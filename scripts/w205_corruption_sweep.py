"""W205 -- is a corrupted checkpoint really the better cheap operator, and if so why?

Tier 48 measured, twice out of sample, that Poseidon-T with Gaussian noise at 3%
of every weight tensor's rms needs FEWER classical calls inside defect correction
than the clean checkpoint: 71 against 99 at six windows, 138 against 222 at
twelve.  In sample at two windows the ordering followed the checkpoint's
integrity -- 25 clean, 32 detuned, 35 with the checkpoint replaced by the
identity, 47 cold.  **One seed, one magnitude.**  Until that inversion is
explained, no claim that a checkpoint contributes is supported in any regime
([[defect-correction-learned-operator]] section 8.2, W205).

This sweeps it.  Three questions in order, cheapest first:

  ``repro``   **the reproduction control, and nothing runs before it passes.**
              sigma = 0.03 at the original seed must return Tier 48's arm to the
              call -- same `DEC_SETTINGS`, same stopping residual, same arms.  A
              sweep whose default cell does not reproduce is measuring its own
              scaffolding.
  ``sweep``   **is the ordering noise?**  Several seeds at one magnitude answers
              that on its own: if the spread across seeds straddles the clean
              checkpoint's count, the Tier 48 cell was a draw.  Then several
              magnitudes, because a MONOTONE response to a knob is not noise and
              a scatter is.
  ``probe``   **if it is not noise, which property of the column is doing it?**
              The candidate named on the page is dissipation on the slow modes
              and NOT accuracy, and both are measured per copy rather than
              argued: the one-step distance from the classical map, and the
              directional derivative of each map along low- and high-wavenumber
              perturbations at the settled state.  That is the quantity
              Theorem 2's rate is actually about, and Tier 48 could only read it
              off the error structure of stalled iterates -- marked
              **[AI Inference]** there.

Everything reuses `scripts/w202_kill_tests.py` unchanged: its `Maps`, its arms,
its `DEC_SETTINGS`, its stopping rule and its reference caches.  The only thing
Tier 49 added to that module is a `set_corruption(sigma, seed)` that re-draws
`Pw`'s noise; both defaults are Tier 48's, so `repro` is a real control.

Writes ``out/w205/w205.json`` after every cell, so a killed run keeps its work.
Nothing is downloaded -- Poseidon-T from the local cache with the hub offline --
no machine is rented and NeuberNet is not loaded.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                         # noqa: E402
import io                                                               # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np                                                      # noqa: E402

import w202_kill_tests as K                                             # noqa: E402
from atlas.cases import wake_array as wa                                # noqa: E402
from atlas.defect_correction import classical_march, defect_correct, shrink, vector_rms  # noqa: E402

OUT = os.path.join(HERE, "out", "w205")

#: **The grid, fixed before the first cell ran.**  Five magnitudes log-spaced
#: about Tier 48's 0.03, so the sweep brackets it on both sides rather than
#: extending it one way; and four seeds, the first of which is Tier 48's own.
#: 0.003 is the continuity control -- a corruption that small must return the
#: clean checkpoint's behaviour, or the knob is not the knob -- and 0.30 is the
#: destruction control, where the network's learned content is gone.
SIGMAS = (0.003, 0.01, 0.03, 0.10, 0.30)
SEEDS = (20260911, 20260912, 20260913, 20260914)
#: Tier 48's cell, which `repro` must reproduce to the call.
TIER48_CELL = (0.03, 20260911)
TIER48_EXPECTED = {6: 71, 12: 138}
#: the clean checkpoint and the no-checkpoint control, from Tier 48's artifact
CLEAN_EXPECTED = {6: 99, 12: 222}
Z_EXPECTED = {6: 112, 12: 240}
COLD = {6: 141, 12: 267}

ALPHA = 0.5                      # the gate's alpha*, fixed on N=2, not re-tuned

#: **The magnitude the question is about.**  W205 asks whether TIER 48'S CELL was
#: a draw, and that cell is sigma = 0.03; the verdict is decided there and every
#: other magnitude is a control on it, not an extra vote.
DECISION_SIGMA = 0.03
#: **The continuity control.**  At a corruption this small the copy must behave
#: like the clean checkpoint -- that is what says the knob is the knob.  The
#: first version of `_verdict` scored this control's FAILURE to beat the clean
#: checkpoint as evidence of noise, which is backwards: it is the control
#: working.  Corrected BEFORE any cell of the grid was read, and recorded here
#: rather than quietly, because a reading repaired after the numbers are in is
#: not a pre-registered reading.
CONTINUITY_SIGMA = 0.003


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _retry(fn, attempts=40, pause=0.25):
    """OneDrive holds a just-written file, so `os.replace` can raise WinError 5."""
    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(pause)


def persist(art):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "w205.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(K._f(art), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_art():
    path = os.path.join(OUT, "w205.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def w202_art():
    with open(os.path.join(HERE, "out", "w202", "w202.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# one cell of the sweep
# ---------------------------------------------------------------------------


class Rung:
    """One rung's maps, reference and stopping rule, built once and reused.

    Building `Maps` loads the checkpoint, so a sweep that rebuilt it per cell
    would pay that once per cell and, worse, would not be able to say that two
    cells differ by the corruption alone.  One `Maps`, `set_corruption` between
    cells, and the clean `P` arm re-run inside the same object as the control
    that the object itself did not drift.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self.M = K.Maps(n)
        z = K._load_ref(n)
        self.ref = np.stack([z["u"], z["v"]])
        rec = w202_art()["reference"][f"N{n}"]
        self.rec = rec
        target = 1e-3 * rec["cold_distance"]
        idx = rec.get("error_index") or list(range(len(rec["error_to_final"])))
        m_hit = next(m for m, e in zip(idx, rec["error_to_final"]) if e <= target)
        self.r_stop = rec["residual"][m_hit]
        self.cold_calls = next(j for j, r in enumerate(rec["residual"])
                               if r <= self.r_stop) + 1
        self.theta = rec["theta"]["theta"]
        self.phi = K._stack(self.M.F)
        self.w0 = np.stack(self.M.freestream())

    def arm(self, op: str, alpha: float = ALPHA, label: str = "") -> dict:
        psi = K._stack({"P": self.M.P, "Pw": self.M.Pw, "Z": self.M.Z,
                        "C": self.M.C, "Pc": self.M.Pc}[op])
        if alpha > 0.0:
            psi = shrink(psi, alpha)
        calls0 = dict(self.M.calls)
        t0 = time.time()
        res = defect_correct(self.phi, psi, self.w0, r_stop=self.r_stop,
                             reference=self.ref, fallback=True, **K.DEC_SETTINGS)
        d = res.as_dict()
        err = vector_rms(res.state - self.ref)
        d.update({
            "operator": op, "alpha": alpha, "label": label,
            "wall_s": time.time() - t0, "final_error": err,
            "model_calls": {k: self.M.calls[k] - calls0[k] for k in self.M.calls},
            "certificate_bound": self.theta * res.residual if res.converged else None,
            "inside_certificate": bool(res.converged
                                       and err <= self.theta * res.residual),
            "constant_it_would_have_needed": (err / res.residual
                                              if res.residual > 0 else None),
        })
        if res.pre_fallback_state is not None:
            e = res.pre_fallback_state - self.ref
            t = self.M.r.tiling
            d["pre_fallback_error"] = vector_rms(e)
            d["pre_fallback_error_structure"] = K.error_structure(
                e, t.weights(), t.offsets, wa.N)
        return d


# ---------------------------------------------------------------------------
# stage 1 -- the reproduction control
# ---------------------------------------------------------------------------


def stage_repro(n: int, art: dict) -> dict:
    """Tier 48's cell, re-run through this driver. Nothing sweeps until it passes."""
    r = Rung(n)
    sig, seed = TIER48_CELL
    r.M.set_corruption(sig, seed)
    pw = r.arm("Pw", label=f"sigma={sig} seed={seed} (Tier 48's cell)")
    clean = r.arm("P", label="the clean checkpoint")
    out = {
        "rung": n, "r_stop": r.r_stop, "theta": r.theta,
        "cold_calls": r.cold_calls,
        "settings": K.DEC_SETTINGS,
        "Pw_phi_calls": pw["phi_calls"], "Pw_expected": TIER48_EXPECTED.get(n),
        "P_phi_calls": clean["phi_calls"], "P_expected": CLEAN_EXPECTED.get(n),
        "Pw_reproduces": pw["phi_calls"] == TIER48_EXPECTED.get(n),
        "P_reproduces": clean["phi_calls"] == CLEAN_EXPECTED.get(n),
        "Pw": pw, "P": clean,
    }
    out["passes"] = bool(out["Pw_reproduces"] and out["P_reproduces"])
    art.setdefault("repro", {})[f"N{n}"] = out
    persist(art)
    print(f"  repro N{n}: Pw {pw['phi_calls']} (expected {TIER48_EXPECTED.get(n)}), "
          f"P {clean['phi_calls']} (expected {CLEAN_EXPECTED.get(n)})  -> "
          f"{'PASS' if out['passes'] else 'FAIL'}", flush=True)
    return out


# ---------------------------------------------------------------------------
# stage 2 -- the grid
# ---------------------------------------------------------------------------


def stage_sweep(n: int, art: dict, sigmas=SIGMAS, seeds=SEEDS,
                rung: Rung | None = None) -> dict:
    """Several seeds at several magnitudes, one rung, everything else held."""
    r = rung or Rung(n)
    cells = art.setdefault("sweep", {}).setdefault(f"N{n}", {})
    meta = art["sweep"].setdefault(f"meta_N{n}", {})
    meta.update({"r_stop": r.r_stop, "theta": r.theta, "cold_calls": r.cold_calls,
                 "alpha": ALPHA, "settings": K.DEC_SETTINGS,
                 "sigmas": list(sigmas), "seeds": list(seeds),
                 "clean_expected": CLEAN_EXPECTED.get(n),
                 "Z_expected": Z_EXPECTED.get(n)})
    for sig in sigmas:
        for seed in seeds:
            key = f"s{sig:g}_seed{seed}"
            if key in cells:
                print(f"    [{key}] cached: phi {cells[key]['phi_calls']}", flush=True)
                continue
            r.M.set_corruption(sig, seed)
            t0 = time.time()
            d = r.arm("Pw", label=f"sigma={sig} seed={seed}")
            d["sigma"], d["seed"] = sig, seed
            cells[key] = d
            persist(art)
            print(f"    [{key}] phi {d['phi_calls']:4d}  psi {d['psi_calls']:4d}  "
                  f"err {d['final_error']:.3e}  status {d['status']}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    art["sweep"][f"summary_N{n}"] = summarise(cells, n)
    persist(art)
    return art["sweep"][f"summary_N{n}"]


def summarise(cells, n):
    """Per magnitude: the spread across seeds, against the clean checkpoint.

    **The reading, fixed before any cell was read.**  W205 asks whether Tier
    48's CELL was a draw, so the verdict is decided at that cell's magnitude,
    `DECISION_SIGMA`, and every other magnitude is a control on it:

      * **NOISE** -- at sigma = 0.03 the across-seed spread covers the clean
        checkpoint's count, so which side of it a single draw lands on is luck.
      * **REAL** -- every seed at sigma = 0.03 beats the clean checkpoint.
      * **REVERSED** -- no seed at sigma = 0.03 beats it, and Tier 48 drew the
        outlier.

    and, when REAL, whether the effect is **GRADED** -- the mean moves with the
    magnitude by more than the across-seed spread, so the corruption is the
    cause -- or **FLAT**, in which case the corruption's SIZE is irrelevant and
    something else about this map is doing the work.  FLAT is a third answer and
    is reported as itself rather than folded into either.

    The continuity control at `CONTINUITY_SIGMA` is reported separately and is
    NOT a vote: a copy corrupted a tenth as hard SHOULD sit nearer the clean
    checkpoint, and it failing to beat it is the control working.
    """
    clean = CLEAN_EXPECTED.get(n)
    by_sigma = {}
    for _key, d in cells.items():
        s_ = "%g" % d["sigma"]
        by_sigma.setdefault(s_, {"sigma": d["sigma"], "phi": [], "seeds": [],
                                 "err": [], "status": []})
        by_sigma[s_]["phi"].append(d["phi_calls"])
        by_sigma[s_]["seeds"].append(d["seed"])
        by_sigma[s_]["err"].append(d["final_error"])
        by_sigma[s_]["status"].append(d["status"])
    for _s, row in by_sigma.items():
        p_ = np.asarray(row["phi"], dtype=float)
        row.update({
            "n_seeds": int(p_.size), "mean": float(p_.mean()),
            "min": int(p_.min()), "max": int(p_.max()),
            "std": float(p_.std(ddof=1)) if p_.size > 1 else None,
            "spread": int(p_.max() - p_.min()),
            "all_beat_the_clean_checkpoint": bool(
                clean is not None and bool((p_ < clean).all())),
            "any_beat_the_clean_checkpoint": bool(
                clean is not None and bool((p_ < clean).any())),
            "spread_covers_the_clean_checkpoint": bool(
                clean is not None and p_.min() <= clean <= p_.max()),
            "distance_of_the_mean_from_the_clean_checkpoint": (
                abs(float(p_.mean()) - clean) if clean is not None else None),
        })
    order = sorted(by_sigma, key=lambda k: by_sigma[k]["sigma"])
    means = [by_sigma[k]["mean"] for k in order]
    floors = [by_sigma[k]["spread"] for k in order if by_sigma[k]["n_seeds"] > 1]
    floor = max(floors) if floors else None
    span = (max(means) - min(means)) if means else None
    dec = by_sigma.get("%g" % DECISION_SIGMA)
    con = by_sigma.get("%g" % CONTINUITY_SIGMA)
    out = {
        "clean_checkpoint": clean, "cold": COLD.get(n), "Z": Z_EXPECTED.get(n),
        "decision_sigma": DECISION_SIGMA, "continuity_sigma": CONTINUITY_SIGMA,
        "by_sigma": by_sigma, "sigma_order": order, "means": means,
        "monotone_decreasing_in_sigma": all(b <= a for a, b in zip(means, means[1:])),
        "monotone_increasing_in_sigma": all(b >= a for a, b in zip(means, means[1:])),
        "across_seed_spread_floor": floor,
        "span_across_sigma": span,
        "span_over_floor": (span / floor) if floor else None,
        "continuity_control": None if con is None else {
            "sigma": CONTINUITY_SIGMA, "mean": con["mean"],
            "distance_from_clean":
                con["distance_of_the_mean_from_the_clean_checkpoint"],
            "nearer_the_clean_checkpoint_than_the_decision_cell": (
                None if dec is None else bool(
                    con["distance_of_the_mean_from_the_clean_checkpoint"]
                    < dec["distance_of_the_mean_from_the_clean_checkpoint"])),
        },
    }
    out["verdict"] = _verdict(dec, floor, span)
    return out


def _verdict(dec, floor, span):
    """NOISE / REAL-AND-GRADED / REAL-BUT-FLAT / REVERSED, by the rule above."""
    if dec is None:
        return "not measured: no cell at the decision magnitude"
    if dec["n_seeds"] < 2:
        return ("not measured: one seed at the decision magnitude is what W205 "
                "already had")
    if dec["spread_covers_the_clean_checkpoint"]:
        return ("NOISE: at sigma = %g the across-seed spread %d-%d covers the "
                "clean checkpoint, so Tier 48's cell was a draw"
                % (dec["sigma"], dec["min"], dec["max"]))
    if not dec["any_beat_the_clean_checkpoint"]:
        return ("REVERSED: no seed at sigma = %g beats the clean checkpoint, so "
                "Tier 48 drew the outlier" % dec["sigma"])
    if not dec["all_beat_the_clean_checkpoint"]:
        return ("MIXED: some seeds at sigma = %g beat the clean checkpoint and "
                "some do not" % dec["sigma"])
    if floor and span is not None and span > 2.0 * floor:
        return ("REAL AND GRADED: every seed at sigma = %g beats the clean "
                "checkpoint, and the magnitude moves the mean by %.0f calls "
                "against an across-seed spread of %d"
                % (dec["sigma"], span, floor))
    return ("REAL BUT FLAT: every seed at sigma = %g beats the clean checkpoint, "
            "and the magnitude does not move it above the across-seed spread -- "
            "so the corruption's SIZE is not what is doing the work"
            % dec["sigma"])


# ---------------------------------------------------------------------------
# stage 3 -- the mechanism
# ---------------------------------------------------------------------------


def _band_perturbation(shape, kx_lo, kx_hi, rng) -> np.ndarray:
    """A divergence-agnostic random field whose energy sits in one band of |k|.

    Built in Fourier space with a flat spectrum inside the band and zero outside,
    then made real.  Normalised to unit rms, so the directional derivative below
    is an amplification factor and not a magnitude.
    """
    _c, ny, nx = shape
    ky = np.abs(np.fft.fftfreq(ny) * ny)[:, None]
    kx = np.abs(np.fft.fftfreq(nx) * nx)[None, :]
    kk = np.sqrt(ky * ky + kx * kx)
    mask = (kk >= kx_lo) & (kk <= kx_hi)
    out = np.empty(shape)
    for c in range(shape[0]):
        amp = rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
        out[c] = np.real(np.fft.ifft2(amp * mask))
    nrm = float(np.sqrt(np.mean(out * out)))
    if nrm <= 0.0:
        raise ValueError(f"band {kx_lo}-{kx_hi} is empty on a {ny}x{nx} grid")
    return out / nrm


#: Wavenumber bands, in cycles across the domain.  "slow" is the band
#: [[defect-correction-learned-operator]] section 5 names -- the lowest modes,
#: where the composed learned column holds each window's incoming mean and the
#: monolith relaxes through its ring and its viscosity.
BANDS = (("slow", 0.0, 2.0), ("mid", 3.0, 8.0), ("fast", 12.0, 32.0))
BAND_INDEX = {b[0]: i for i, b in enumerate(BANDS)}


def stage_probe(n: int, art: dict, sigmas=SIGMAS, seeds=(20260911,),
                eps: float = 1.0e-4, n_draws: int = 3,
                rung: Rung | None = None) -> dict:
    """Accuracy and per-band contraction of every cheap map, at the settled state.

    **This is the measurement Tier 48 could only infer.**  Theorem 2 puts the
    rate through ``rho(I - J_Psi^-1 J_Phi)`` with ``J = I - D(map)``, so what
    decides an arm is the ACTION of each map's derivative on the modes the
    iteration has to remove -- not how close the map is to the classical one.
    Tier 48 read that off the error structure of stalled iterates and marked the
    reading **[AI Inference]**.  Here it is a directional derivative:

        D(map) d  ~  (map(w* + eps d) - map(w*)) / eps

    averaged over ``n_draws`` random unit fields whose energy sits in one band,
    for the classical map and each cheap map.  The quantity reported per band is
    the amplification ``|D(map) d|`` -- below 1 the map damps that band, above 1
    it amplifies it -- and the shrink ``alpha`` is applied afterwards exactly as
    the iteration applies it, so the numbers are the map the arm actually ran.
    """
    r = rung or Rung(n)
    w_star = r.ref
    out = art.setdefault("probe", {}).setdefault(f"N{n}", {})
    out["settings"] = {"eps": eps, "n_draws": n_draws, "alpha": ALPHA,
                       "bands": [list(b) for b in BANDS]}

    maps: list[tuple[str, object, dict]] = [
        ("F (classical)", r.M.F, {}),
        ("P (clean)", r.M.P, {}),
        ("Z (no checkpoint)", r.M.Z, {}),
    ]
    for sig in sigmas:
        for seed in seeds:
            maps.append((f"Pw s{sig:g} seed{seed}", None,
                         {"sigma": sig, "seed": seed}))

    base = {}
    for label, fn, cfg in maps:
        if cfg:
            r.M.set_corruption(cfg["sigma"], cfg["seed"])
            fn = r.M.Pw
        t0 = time.time()
        g = K._stack(fn)
        y0 = g(w_star)
        if label == "F (classical)":
            base["F (classical)"] = y0
        #: ACCURACY: how far this map's one step is from the classical map's, at
        #: the settled state.  The page's claim is that this is NOT what decides
        #: the arm, and the correlation at the end is what tests that.
        row = {"label": label, **cfg, "bands": {},
               "one_step_distance_from_the_classical_map": vector_rms(
                   y0 - base["F (classical)"])}
        for bname, lo, hi in BANDS:
            amps, rayleigh = [], []
            #: a DETERMINISTIC seed per band: `hash` of a str is salted
            #: per process unless PYTHONHASHSEED is set, so using it here
            #: would give every run different perturbations and make the
            #: paired comparison below unreproducible.  Re-seeded identically
            #: for every map, so the comparison is PAIRED on the same fields.
            rng = np.random.default_rng(20490912 + 1000 * BAND_INDEX[bname])
            for _ in range(n_draws):
                d = _band_perturbation(w_star.shape, lo, hi, rng)
                y1 = g(w_star + eps * d)
                jd = (y1 - y0) / eps
                amps.append(vector_rms(jd))
                #: the SIGNED per-band eigenvalue estimate, <d, D d> / <d, d>:
                #: the amplification has no sign and so cannot tell a map that
                #: PRESERVES a mode from one that inverts it, which is exactly
                #: the distinction Theorem 2's denominator turns on.
                rayleigh.append(float(np.sum(jd * d) / np.sum(d * d)))
            #: the shrunk map the iteration actually uses (Corollary 2):
            #: Psi_alpha = (1 - alpha) Psi, so its derivative scales too.
            psi_r = float(np.mean(rayleigh)) * (1.0 - ALPHA)
            row["bands"][bname] = {
                "amplification": float(np.mean(amps)),
                "amplification_std": (float(np.std(amps, ddof=1))
                                      if n_draws > 1 else None),
                "rayleigh": float(np.mean(rayleigh)),
                "rayleigh_std": (float(np.std(rayleigh, ddof=1))
                                 if n_draws > 1 else None),
                "rayleigh_shrunk": psi_r,
                #: **the quantity the rate turns on**: |1 - psi|.  Near zero the
                #: cheap map preserves this band and the iteration's denominator
                #: vanishes on it; near one it damps it and the iteration runs at
                #: the classical rate.
                "one_minus_psi_shrunk": abs(1.0 - psi_r),
                #: below 1 the map DAMPS this band
                "damps": bool(np.mean(amps) < 1.0),
            }
        row["wall_s"] = time.time() - t0
        out[label] = row
        persist(art)
        print("    probe %-24s acc %.4e   " % (
            label, row["one_step_distance_from_the_classical_map"])
            + "  ".join("%s %.4f" % (b, row["bands"][b]["amplification"])
                        for b, _l, _h in BANDS), flush=True)
    out["theorem2_eigenvalues"] = _theorem2(out)
    out["correlation"] = _correlate(art, n)
    persist(art)
    return out


def _theorem2(out):
    """lambda_k = (phi_k - psi_k) / (1 - psi_k), per band, per cheap map.

    Theorem 2's per-mode rate, assembled from the paired Rayleigh quotients
    rather than from an assembled Jacobian -- which is what Tier 48 could not do
    and marked **[AI Inference]** instead.  ``max |lambda|`` over the bands is
    the prediction of which arm converges fastest; whether it actually orders
    the arms is what `_correlate` asks.
    """
    base = out.get("F (classical)")
    if base is None:
        return {"note": "not measured: the classical map was not probed"}
    rows = {}
    SKIP = {"settings", "theorem2_eigenvalues", "correlation"}
    for label, row in out.items():
        if label in SKIP or label.startswith("F ("):
            continue
        if not isinstance(row, dict) or not isinstance(row.get("bands"), dict):
            continue
        per = {}
        for bname, _lo, _hi in BANDS:
            phi = base["bands"][bname]["rayleigh"]
            psi = row["bands"][bname]["rayleigh_shrunk"]
            den = 1.0 - psi
            per[bname] = {
                "phi": phi, "psi_shrunk": psi, "one_minus_psi": abs(den),
                "lambda": (abs((phi - psi) / den) if abs(den) > 1e-12 else None),
            }
        lams = [v["lambda"] for v in per.values() if v["lambda"] is not None]
        rows[label] = {"bands": per,
                       "max_abs_lambda": (max(lams) if lams else None),
                       "converges_if_below_one": (bool(max(lams) < 1.0)
                                                  if lams else None)}
    return rows


def _correlate(art: dict, n: int) -> dict:
    """Do classical calls track accuracy, or track slow-mode damping?

    The whole question of W205 in one table: for every corrupted copy that both
    stages measured, put its classical-call count beside its accuracy and beside
    its slow-band amplification, and report the rank correlation of each.  A
    corruption that helps BECAUSE it damps the slow modes gives a strong
    correlation with the second and a weak one with the first.
    """
    cells = art.get("sweep", {}).get(f"N{n}", {})
    probes = art.get("probe", {}).get(f"N{n}", {})
    rows = []
    for key, d in cells.items():
        label = f"Pw s{d['sigma']:g} seed{d['seed']}"
        p = probes.get(label)
        if p is None:
            continue
        rows.append({
            "cell": key, "sigma": d["sigma"], "seed": d["seed"],
            "phi_calls": d["phi_calls"],
            "accuracy": p["one_step_distance_from_the_classical_map"],
            "slow_amplification": p["bands"]["slow"]["amplification"],
            "mid_amplification": p["bands"]["mid"]["amplification"],
            "fast_amplification": p["bands"]["fast"]["amplification"],
            "slow_one_minus_psi": p["bands"]["slow"].get("one_minus_psi_shrunk"),
            "max_abs_lambda": (probes.get("theorem2_eigenvalues", {})
                               .get(label, {}).get("max_abs_lambda")),
        })
    if len(rows) < 3:
        return {"rows": rows, "note": "fewer than three paired cells: "
                                      "not measured rather than zero"}

    def spearman(a, b):
        ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
        rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
        return float((ra * rb).sum() / den) if den > 0 else None

    phi = [r["phi_calls"] for r in rows]
    return {
        "rows": rows, "n": len(rows),
        "spearman_phi_vs_accuracy": spearman(phi, [r["accuracy"] for r in rows]),
        "spearman_phi_vs_slow_amplification": spearman(
            phi, [r["slow_amplification"] for r in rows]),
        "spearman_phi_vs_mid_amplification": spearman(
            phi, [r["mid_amplification"] for r in rows]),
        "spearman_phi_vs_fast_amplification": spearman(
            phi, [r["fast_amplification"] for r in rows]),
        #: **the two that decide W205.**  If the saving is the corruption
        #: DAMPING the slow modes, classical calls fall as |1 - psi_slow| rises,
        #: so this correlation is strongly NEGATIVE -- and the accuracy one is
        #: weak, because accuracy is not what the rate turns on.
        "spearman_phi_vs_slow_one_minus_psi": spearman(
            phi, [r["slow_one_minus_psi"] for r in rows]
        ) if all(r.get("slow_one_minus_psi") is not None for r in rows) else None,
        "spearman_phi_vs_max_abs_lambda": spearman(
            phi, [r["max_abs_lambda"] for r in rows]
        ) if all(r.get("max_abs_lambda") is not None for r in rows) else None,
    }


# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="repro,sweep",
                    help="repro, sweep, summary, probe (comma separated)")
    ap.add_argument("--rung", type=int, default=6)
    ap.add_argument("--sigmas", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--force", action="store_true",
                    help="sweep even if the reproduction control has not passed")
    a = ap.parse_args(argv)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    sigmas = tuple(float(x) for x in a.sigmas.split(",")) if a.sigmas else SIGMAS
    seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else SEEDS
    n = a.rung

    art = load_art()
    art["what"] = ("W205: is a corrupted checkpoint really the better cheap "
                   "operator, and if so what property is doing it")
    art["date"] = time.strftime("%Y-%m-%d")
    t0 = time.time()
    rung = None

    if "repro" in stages:
        print(f"1. the reproduction control, N={n}", flush=True)
        rep = stage_repro(n, art)
        if not rep["passes"] and not a.force:
            print("   the default cell does NOT reproduce Tier 48; stopping. "
                  "A sweep whose control fails is measuring its own scaffolding. "
                  "Re-run with --force only to diagnose.", flush=True)
            persist(art)
            return

    if "sweep" in stages:
        print(f"2. the grid, N={n}: {len(sigmas)} magnitudes x {len(seeds)} seeds",
              flush=True)
        rung = rung or Rung(n)
        s = stage_sweep(n, art, sigmas, seeds, rung=rung)
        print(f"   verdict: {s['verdict']}", flush=True)
        for key in s["sigma_order"]:
            row = s["by_sigma"][key]
            print("   sigma %-6s n=%d  phi %s  mean %.1f  spread %d  "
                  "all beat clean(%s): %s" % (
                      key, row["n_seeds"], sorted(row["phi"]), row["mean"],
                      row["spread"], s["clean_checkpoint"],
                      row["all_beat_the_clean_checkpoint"]), flush=True)

    if "summary" in stages:
        cells = art.get("sweep", {}).get("N%d" % n, {})
        if not cells:
            print("   no cached cells to summarise", flush=True)
        else:
            sm = summarise(cells, n)
            art.setdefault("sweep", {})["summary_N%d" % n] = sm
            persist(art)
            print("   verdict: %s" % sm["verdict"], flush=True)
            for key in sm["sigma_order"]:
                row = sm["by_sigma"][key]
                print("   sigma %-6s n=%d  phi %s  mean %6.1f  spread %3d  "
                      "all beat clean(%s): %s" % (
                          key, row["n_seeds"], sorted(row["phi"]), row["mean"],
                          row["spread"], sm["clean_checkpoint"],
                          row["all_beat_the_clean_checkpoint"]), flush=True)
            c = sm.get("continuity_control")
            if c:
                print("   continuity control sigma=%g: mean %.1f, nearer clean "
                      "than the decision cell: %s"
                      % (c["sigma"], c["mean"],
                         c["nearer_the_clean_checkpoint_than_the_decision_cell"]),
                      flush=True)

    if "probe" in stages:
        print(f"3. the mechanism, N={n}", flush=True)
        rung = rung or Rung(n)
        p = stage_probe(n, art, sigmas, seeds, rung=rung)
        c = p.get("correlation", {})
        if "spearman_phi_vs_accuracy" in c:
            print("   spearman(phi, accuracy) %.3f   spearman(phi, slow amp) %.3f"
                  % (c["spearman_phi_vs_accuracy"],
                     c["spearman_phi_vs_slow_amplification"]), flush=True)

    art["elapsed_seconds"] = time.time() - t0
    print("wrote", persist(art), "in %.1f s" % art["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main()
