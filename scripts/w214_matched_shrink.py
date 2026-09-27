"""W214 -- the shrink matched to the slow-mode response, with alpha* = 0.5 as the control.

Tier 50 ([[corrupted-checkpoint-and-jacobian-fidelity]] section 4.4) measured, on
the derivative, that the composition layer's shrink overshoots: on the slow band
the classical map's response is phi_slow = 0.4780, the clean column's is
psi_slow = 0.5578, and (1 - 0.5) psi_slow = 0.2789 -- two and a half times below
the classical map.  alpha* = 0.5 was chosen on N = 2 as the only grid value at
which the checkpoint converged without falling back: a STABILITY criterion applied
to a FIDELITY quantity.

This does the cheapest thing that page opened: choose alpha per rung so that

    (1 - alpha) psi_slow  =  phi_slow      =>      alpha = 1 - phi_slow / psi_slow

and re-run the P arm there, with alpha* = 0.5 re-run beside it in the same process
as the reproduction control.  The composition layer with the checkpoint replaced
by the identity (Z) is run at ITS OWN matched alpha, because the question the brief
asks -- "was the checkpoint ever the variable?" -- is answered by comparing the two
at the rule, not at a shared number.

Stages, cheapest first:

  ``probe``  the per-band Rayleigh quotients of F, P and Z at the settled state,
             by `w205_corruption_sweep`'s own perturbations (same bands, eps,
             draws and seeds).  N = 6 must reproduce Tier 50's cached probe; N = 12
             has never been probed.  Writes the matched alphas.
  ``arms``   P at 0.5 (control), P at alpha_P, Z at alpha_Z, per rung.
  ``scan``   optional: P at a few more alphas at N = 6, to read the knob's curve.

Writes ``out/w214/w214.json`` after every arm.  Nothing is downloaded (Poseidon-T
from the local cache, hub offline), no machine is rented, NeuberNet is not loaded.
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
import w205_corruption_sweep as S                                       # noqa: E402
from atlas.defect_correction import vector_rms                          # noqa: E402

OUT = os.path.join(HERE, "out", "w214")

#: Tier 48 / Tier 50's counts, which the control and the predictions are read
#: against.  Every one of these is in `out/w202/w202.json` or `out/w205/w205.json`.
P_HALF = {6: 99, 12: 222}            # P at alpha* = 0.5, the control
Z_HALF = {6: 112, 12: 240}           # Z at alpha* = 0.5
COLD = {6: 141, 12: 267}
BEST_CORRUPTED = {6: 71, 12: 138}    # the minimum over Tier 50's directions
#: the cheapest classical competitor in the slot, C at alpha = 0: classical calls
#: plus c_C times coarse calls, in classical-equivalents (Tier 48's ledger)
COARSE_COST = {6: 24 + 0.13239362522140255 * 211,
               12: 27 + 0.032049938932302645 * 347}
C_PSI = {6: 2.0075269487324916, 12: 0.2959342798372732}   # P per call, / F
C_Z = {6: 0.14868664696200168, 12: 0.03648846693923803}   # Z per call, / F

#: **Registered before any arm or the N = 12 probe ran** (2026-09-26, this file's
#: first version).  Each is evaluated by `evaluate` against the artifact, and the
#: evaluator is written beside the words so the two cannot drift apart.
PREDICTIONS = {
    "M1": "P at alpha* = 0.5 reproduces Tier 48's classical-call count exactly at "
          "both rungs (99 at N=6, 222 at N=12): the reproduction control",
    "M2": "P at its matched alpha needs FEWER classical calls than P at 0.5 at both "
          "rungs -- the brief's prediction, that matching improves the rate",
    "M3": "P at its matched alpha does NOT beat the best corrupted direction "
          "(71 at N=6, 138 at N=12) at either rung",
    "M4": "G5 does not move: P at its matched alpha costs MORE classical-equivalents "
          "(phi + c_P psi) than the coarse classical arm C at alpha 0 at both rungs",
    "M5": "matching recovers at least half of the 28-call gap the corruption found "
          "at N=6: P at its matched alpha needs at most 85 classical calls",
    "M6": "the checkpoint is not the variable at the matched rule: at N=6, Z at its "
          "own matched alpha and P at its matched alpha differ by at most 13 calls "
          "(the learned content Tier 50 measured)",
    "M7": "every arm at N=6 returns a state inside its own certificate "
          "(error <= Theta * residual)",
}


def evaluate(art: dict) -> dict:
    out = {}
    arms = art.get("arms", {})

    def g(n, key):
        return arms.get(f"N{n}", {}).get(key)

    def calls(n, key):
        a = g(n, key)
        return None if a is None else a["phi_calls"]

    def cost(n, key, c):
        a = g(n, key)
        return None if a is None else a["phi_calls"] + c * a["psi_calls"]

    def verdict(ok):
        return "not measured" if ok is None else ("held" if ok else "FAILED")

    r = {n: (calls(n, "P_half"), P_HALF[n]) for n in (6, 12)}
    ok = None if any(v[0] is None for v in r.values()) else all(a == b for a, b in r.values())
    out["M1"] = {"got": {f"N{n}": v[0] for n, v in r.items()}, "verdict": verdict(ok)}

    r = {n: (calls(n, "P_match"), calls(n, "P_half")) for n in (6, 12)}
    ok = None if any(None in v for v in r.values()) else all(a < b for a, b in r.values())
    out["M2"] = {"got": {f"N{n}": v for n, v in r.items()}, "verdict": verdict(ok)}

    r = {n: calls(n, "P_match") for n in (6, 12)}
    ok = None if None in r.values() else all(r[n] >= BEST_CORRUPTED[n] for n in r)
    out["M3"] = {"got": r, "bar": BEST_CORRUPTED, "verdict": verdict(ok)}

    r = {n: cost(n, "P_match", C_PSI[n]) for n in (6, 12)}
    ok = None if None in r.values() else all(r[n] > COARSE_COST[n] for n in r)
    out["M4"] = {"got": r, "coarse": COARSE_COST, "verdict": verdict(ok)}

    c6 = calls(6, "P_match")
    out["M5"] = {"got": c6, "verdict": verdict(None if c6 is None else c6 <= 85)}

    z6, p6 = calls(6, "Z_match"), calls(6, "P_match")
    out["M6"] = {"got": {"Z_match": z6, "P_match": p6},
                 "verdict": verdict(None if None in (z6, p6) else abs(z6 - p6) <= 13)}

    a6 = arms.get("N6", {})
    ins = [a.get("inside_certificate") for k, a in a6.items() if isinstance(a, dict)
           and "phi_calls" in a]
    out["M7"] = {"got": ins, "verdict": verdict(None if not ins else all(ins))}
    return out


# ---------------------------------------------------------------------------


def persist(art):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "w214.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(K._f(art), fh, indent=1)

    S._retry(write)
    S._retry(lambda: os.replace(tmp, path))
    return path


def load_art():
    path = os.path.join(OUT, "w214.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def probe_map(g, w_star, eps=1.0e-4, n_draws=3):
    """Per-band signed Rayleigh quotient of one map at the settled state.

    `w205_corruption_sweep.stage_probe`'s arithmetic exactly -- the same band
    fields from the same deterministic seeds -- so N = 6 can be checked against
    Tier 50's cache rather than trusted to agree with it.
    """
    y0 = g(w_star)
    bands = {}
    for bname, lo, hi in S.BANDS:
        rng = np.random.default_rng(20490912 + 1000 * S.BAND_INDEX[bname])
        amps, ray = [], []
        for _ in range(n_draws):
            d = S._band_perturbation(w_star.shape, lo, hi, rng)
            jd = (g(w_star + eps * d) - y0) / eps
            amps.append(vector_rms(jd))
            ray.append(float(np.sum(jd * d) / np.sum(d * d)))
        bands[bname] = {"rayleigh": float(np.mean(ray)),
                        "amplification": float(np.mean(amps))}
    return y0, bands


def theorem2(phi_b: dict, psi_b: dict, alpha: float) -> dict:
    """lambda per band for the map shrunk by alpha, from the paired quotients."""
    per = {}
    for bname, _lo, _hi in S.BANDS:
        ph = phi_b[bname]["rayleigh"]
        ps = (1.0 - alpha) * psi_b[bname]["rayleigh"]
        den = 1.0 - ps
        per[bname] = abs((ph - ps) / den) if abs(den) > 1e-12 else None
    lams = [v for v in per.values() if v is not None]
    return {"bands": per, "max": max(lams) if lams else None}


def stage_probe(n: int, art: dict, rung: "S.Rung") -> dict:
    w_star = rung.ref
    out = art.setdefault("probe", {}).setdefault(f"N{n}", {})
    t0 = time.time()
    rows = {}
    for label, fn in (("F", rung.M.F), ("P", rung.M.P), ("Z", rung.M.Z)):
        _y, b = probe_map(K._stack(fn), w_star)
        rows[label] = b
        print(f"    probe N{n} {label}: " + "  ".join(
            f"{k} {v['rayleigh']:+.4f}" for k, v in b.items()), flush=True)
    phi_s = rows["F"]["slow"]["rayleigh"]
    alpha_p = 1.0 - phi_s / rows["P"]["slow"]["rayleigh"]
    alpha_z = 1.0 - phi_s / rows["Z"]["slow"]["rayleigh"]
    out.update({"bands": rows, "alpha_P": alpha_p, "alpha_Z": alpha_z,
                "wall_s": time.time() - t0})
    #: the LINEAR prediction at w*, for each alpha an arm will run at.  Tier 48's
    #: alpha = 0 arm at N = 6 stalled and fell back at 144 calls although this
    #: reading gives it max |lambda| < 1, so the prediction is reported, not
    #: trusted: it describes the iteration near the settled state only.
    out["theorem2"] = {
        "P": {f"{a:.4f}": theorem2(rows["F"], rows["P"], a)
              for a in (0.0, alpha_p, 0.5)},
        "Z": {f"{a:.4f}": theorem2(rows["F"], rows["Z"], a)
              for a in (0.0, alpha_z, 0.5)},
    }
    #: the reproduction control for the probe: Tier 50 probed N = 6 with the same
    #: code path, so its cached quotients must come back
    cached = S.load_art().get("probe", {}).get(f"N{n}", {})
    if cached:
        diffs = {}
        for mine, theirs in (("F", "F (classical)"), ("P", "P (clean)"),
                             ("Z", "Z (no checkpoint)")):
            if theirs in cached:
                diffs[mine] = max(abs(rows[mine][b]["rayleigh"]
                                      - cached[theirs]["bands"][b]["rayleigh"])
                                  for b in rows[mine])
        out["reproduces_tier50_probe"] = diffs
        print(f"    probe N{n} against Tier 50's cache, max |diff|: {diffs}", flush=True)
    persist(art)
    print(f"    N{n}: alpha_P = {alpha_p:.4f}   alpha_Z = {alpha_z:.4f}", flush=True)
    return out


def run_arm(n, art, rung, key, op, alpha, label):
    arms = art.setdefault("arms", {}).setdefault(f"N{n}", {})
    if key in arms:
        print(f"    [{key}] cached: phi {arms[key]['phi_calls']}", flush=True)
        return arms[key]
    t0 = time.time()
    d = rung.arm(op, alpha=alpha, label=label)
    arms[key] = d
    persist(art)
    print(f"    [N{n} {key}] {op} alpha={alpha:.4f}: phi {d['phi_calls']:4d}  "
          f"psi {d['psi_calls']:4d}  status {d['status']}  "
          f"inside {d['inside_certificate']}  {time.time() - t0:.0f}s", flush=True)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="probe,arms")
    ap.add_argument("--rungs", default="6,12")
    ap.add_argument("--scan", default="0.25,0.35")
    a = ap.parse_args(argv)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    art = load_art()
    art["what"] = "W214: the shrink matched to the slow-mode response"
    art["predictions"] = PREDICTIONS
    art.setdefault("registered", "2026-09-26, before any arm or the N=12 probe ran")
    t_all = time.time()
    for n in [int(x) for x in a.rungs.split(",")]:
        print(f"== N{n}", flush=True)
        rung = S.Rung(n)
        if "probe" in stages or f"N{n}" not in art.get("probe", {}):
            stage_probe(n, art, rung)
        pr = art["probe"][f"N{n}"]
        if "arms" in stages:
            run_arm(n, art, rung, "P_half", "P", 0.5, "control: alpha* = 0.5")
            run_arm(n, art, rung, "P_match", "P", pr["alpha_P"],
                    "P at its matched alpha")
            run_arm(n, art, rung, "Z_match", "Z", pr["alpha_Z"],
                    "Z at its own matched alpha")
        if "scan" in stages and n == 6:
            for al in [float(x) for x in a.scan.split(",") if x]:
                run_arm(n, art, rung, f"P_{al:g}", "P", al, f"scan alpha={al:g}")
        art["evaluation"] = evaluate(art)
        persist(art)
    art["elapsed_seconds"] = time.time() - t_all
    art["evaluation"] = evaluate(art)
    print("wrote", persist(art), flush=True)
    for k, v in art["evaluation"].items():
        print(f"  {k}: {v['verdict']}  {v['got']}", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
