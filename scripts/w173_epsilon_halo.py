"""W173 -- the epsilon-halo, measured before any rule is written.

`case-study-ladder-to-f1` section 5 proposes the one unbuilt route through R10 and
marks it [AI Inference]:

    "`support_reach` currently asks whether the response is NONZERO past the
     declared radius; the physically meaningful question is whether it is LARGER
     THAN THE DEFECT THE COMPOSITION ALREADY TOLERATES.  An epsilon-halo -- the
     radius beyond which the response falls below eps_tol, with the truncated
     tail carried as an explicit term in sigma -- would be a bounded halo for an
     operator whose support is formally global.  Whether the tail is summable is
     an empirical question about the architecture and is measurable with the
     probe that exists."

This script is that measurement and nothing else.  No rule is written here.

What is measured
----------------

`probe.support_reach` pokes ONE seam cell and counts how many cells move.  That
is one column of the seam operator.  A halo DECLARATION is a statement about the
whole operator -- "cells further than r apart on this seam do not talk" -- so the
quantity the rule would need is the BAND TRUNCATION of the dense seam response:

    S[:, j] = ( respond(base + a e_j) - respond(base) ) / a        (n+1 solves)
    S_r     = S restricted to |i - j| <= r
    E(r)    = || S - S_r ||                                        the tail

`E(r)` is exactly `||Lambda - Lambda~||` in master-error-bound section 4's sigma
factorization, so an epsilon-halo at radius r is not a free approximation: it
adds `C_mu * E(r) * ||lambda*|| / beta` to sigma, which is the "explicit term"
the proposal requires and which this script reports rather than drops.

Subjects.  Three, on purpose -- one graph is an anecdote:

  1. Poseidon-T, a frozen 20.8M neural operator.  Global by ARCHITECTURE.
     Declared reach `stencil_radius * substeps` = 2; W93 measured 64.
  2. WindowNS as-built, a classical solver with an EMBEDDED pressure solve.
     Global by PHYSICS -- an elliptic solve inverts a dense operator.
     Declared reach 20.
  3. WindowNS split-step, the same solver with the elliptic part EXPOSED to the
     composition layer.  Local by construction; the positive control that says
     the instrument can read a compact operator when one is in front of it.

Outputs `out/w173/w173.json`.  Every number in the write-up comes from there.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
import sys
import time

import numpy as np

if __name__ == "__main__":
    # Only when RUN.  Re-wrapping on import garbage-collects the importer's own
    # wrapper, which closes the shared buffer under it -- the second script to
    # do this dies on its first print with "I/O operation on closed file".
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.cases import neural_interface as NI  # noqa: E402
from atlas.cases import poseidon as PO  # noqa: E402
from atlas.cases import window_ns as W  # noqa: E402
from atlas.probe import support_reach  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "w173")


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------


def dense_seam_operator(respond, port_name, base, amplitude=1.0):
    """The full n x n response, one column per seam cell.  n+1 solves.

    A single poke at the middle cell -- which is what `support_reach` does -- is
    column n/2 of this matrix.  It is the right instrument for "is the support
    compact"; it is the wrong one for "how wide a band does the operator live
    in", because an operator that is not translation-invariant along the seam
    can decay from the middle and not from the edge.  The seam here has corners
    and a varying base state, so it is not translation-invariant, and the whole
    matrix is what the declaration would be about.
    """
    base = np.asarray(base, dtype=float).ravel()
    n = base.size
    f0 = np.asarray(respond(port_name, base), dtype=float).ravel()
    S = np.empty((n, n), dtype=float)
    for j in range(n):
        d = np.zeros(n)
        d[j] = float(amplitude)
        fj = np.asarray(respond(port_name, base + d), dtype=float).ravel()
        S[:, j] = (fj - f0) / float(amplitude)
    return S, f0


def band_mask(n, r, circular=False):
    """True where |i - j| <= r.  `circular` uses the wrap-around distance."""
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    d = np.abs(i - j)
    if circular:
        d = np.minimum(d, n - d)
    return d <= r


def truncation_curve(S, radii, circular=False):
    """||S - S_r|| against r, in the two norms the bound could be stated in."""
    n = S.shape[0]
    norm2 = float(np.linalg.norm(S, 2))
    normF = float(np.linalg.norm(S, "fro"))
    rows = []
    for r in radii:
        T = S * ~band_mask(n, int(r), circular)
        rows.append({
            "r": int(r),
            "E2": float(np.linalg.norm(T, 2)),
            "EF": float(np.linalg.norm(T, "fro")),
            "E2_rel": float(np.linalg.norm(T, 2) / norm2) if norm2 else float("nan"),
            "EF_rel": float(np.linalg.norm(T, "fro") / normF) if normF else float("nan"),
            "kept_fraction": float(band_mask(n, int(r), circular).sum()) / (n * n),
        })
    return {"norm2": norm2, "normF": normF, "rows": rows}


def column_decay(S, margin=None):
    """Per-column |S[i, j]| against d = |i - j|, averaged over INTERIOR columns.

    Edge columns have a one-sided tail -- half the far field falls off the
    domain -- so a decay law fitted over all columns is fitted partly on the
    domain's edge and not on the operator.  `margin` drops the columns whose
    far side is truncated.
    """
    n = S.shape[0]
    m = int(margin if margin is not None else n // 4)
    cols = range(m, n - m)
    dmax = n - 1
    acc = np.zeros(dmax + 1)
    cnt = np.zeros(dmax + 1)
    per_col = []
    for j in cols:
        i = np.arange(n)
        d = np.abs(i - j)
        g = np.abs(S[:, j])
        # only distances reachable on BOTH sides of this column
        reach = min(j, n - 1 - j)
        keep = d <= reach
        np.add.at(acc, d[keep], g[keep])
        np.add.at(cnt, d[keep], 1.0)
        per_col.append({"j": int(j), "peak": float(g[j]),
                        "far": float(g[keep].min()) if keep.any() else float("nan")})
    prof = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    return {"n_columns": len(per_col), "margin": m,
            "profile": [None if not np.isfinite(x) else float(x) for x in prof]}


def fit_laws(d, y):
    """Exponential vs power law on the same points.  Report both R^2, pick neither."""
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & (y > 0) & (d > 0)
    d, y = d[ok], y[ok]
    out = {"n_points": int(d.size)}
    if d.size < 4:
        return out
    ly = np.log(y)

    def r2(x, yy):
        A = np.vstack([x, np.ones_like(x)]).T
        c, *_ = np.linalg.lstsq(A, yy, rcond=None)
        pred = A @ c
        ss = float(np.sum((yy - pred) ** 2))
        tot = float(np.sum((yy - yy.mean()) ** 2))
        return c, (1.0 - ss / tot if tot > 0 else float("nan"))

    c_exp, r2_exp = r2(d, ly)
    c_pow, r2_pow = r2(np.log(d), ly)
    out["exponential"] = {"slope": float(c_exp[0]),
                          "ell_cells": float(-1.0 / c_exp[0]) if c_exp[0] < 0 else None,
                          "r2": float(r2_exp)}
    out["power_law"] = {"exponent": float(c_pow[0]), "r2": float(r2_pow)}
    out["better"] = "exponential" if r2_exp >= r2_pow else "power_law"
    return out


def r_star(curve, key, targets, n=None):
    """Smallest radius whose truncation is at or below each target.

    ``r = n - 1`` is not an answer: the band is then the whole matrix and the
    truncation is zero by arithmetic rather than by decay.  It is reported with
    ``trivial: true`` so a table cannot read it as a halo.
    """
    out = []
    rows = curve["rows"]
    for t in targets:
        hit = next((row["r"] for row in rows if row[key] <= t), None)
        out.append({"target": float(t), "r_star": hit,
                    "reached": hit is not None,
                    "trivial": (n is not None and hit is not None and hit >= n - 1)})
    return out


def terminal_slope(curve, key, lo, hi):
    """d(E)/dr over the last measured stretch, and the radius it extrapolates to.

    The question a plateau raises is *how far would the halo have to go*, and the
    honest answer is an extrapolation of the terminal slope, quoted as such.  A
    non-negative slope means the curve has stopped falling and no radius is
    implied at all.
    """
    rows = {r["r"]: r[key] for r in curve["rows"]}
    if lo not in rows or hi not in rows or hi <= lo:
        return None
    slope = (rows[hi] - rows[lo]) / float(hi - lo)
    out = {"lo": lo, "hi": hi, "value_lo": rows[lo], "value_hi": rows[hi],
           "slope_per_cell": float(slope)}
    for t in (1e-1, 1e-2, 1e-3):
        if slope < 0 and rows[hi] > t:
            out[f"extrapolated_r_for_{t:g}"] = float(hi + (rows[hi] - t) / (-slope))
        elif rows[hi] <= t:
            out[f"extrapolated_r_for_{t:g}"] = float(hi)
        else:
            out[f"extrapolated_r_for_{t:g}"] = None
    return out


def far_field_spectrum(S, r, k=8):
    """The singular values of the OFF-BAND block at radius r.

    A truncation error that plateaus can plateau two ways: many small modes that
    never decay, or a few coherent global ones.  They are different findings and
    the spectrum separates them.  ``share_1`` is how much of the off-band
    Frobenius energy the leading mode carries.
    """
    n = S.shape[0]
    T = S * ~band_mask(n, int(r), False)
    sv = np.linalg.svd(T, compute_uv=False)
    tot = float(np.sum(sv ** 2))
    return {"r": int(r),
            "singular_values": [float(x) for x in sv[:k]],
            "share_1": float(sv[0] ** 2 / tot) if tot > 0 else float("nan"),
            "share_top4": float(np.sum(sv[:4] ** 2) / tot) if tot > 0 else float("nan"),
            "effective_rank_99": int(np.searchsorted(
                np.cumsum(sv ** 2) / tot, 0.99) + 1) if tot > 0 else 0}


def amplitude_ladder(respond, port_name, n, amps, columns, reference_amp):
    """Is the measured S an OPERATOR, or the instrument's own floor?

    A linear response satisfies S(a) = S(a/2) exactly, so the disagreement
    between two amplitudes is truncation (nonlinearity, growing with a) plus
    cancellation (round-off, growing as 1/a).  The checkpoint runs in float32,
    so the cancellation floor per entry is about ``eps32 * ||f0|| / a`` and is
    reported beside the measurement rather than argued about.  This is the F5
    methodology from CS-10, reused on a different differencing.
    """
    base = np.zeros(n)
    f0 = np.asarray(respond(port_name, base), dtype=float).ravel()
    eps32 = float(np.finfo(np.float32).eps)
    cols = {}
    for a in amps:
        M = np.empty((n, len(columns)), dtype=float)
        for c, j in enumerate(columns):
            d = np.zeros(n)
            d[j] = float(a)
            fj = np.asarray(respond(port_name, base + d), dtype=float).ravel()
            M[:, c] = (fj - f0) / float(a)
        cols[a] = M
    ref = cols[reference_amp]
    rows = []
    for a in amps:
        M = cols[a]
        rows.append({
            "amplitude": float(a),
            "normF": float(np.linalg.norm(M, "fro")),
            "rel_disagreement_vs_ref": float(
                np.linalg.norm(M - ref, "fro") / np.linalg.norm(ref, "fro")),
            "cancellation_floor_per_entry": float(
                eps32 * float(np.max(np.abs(f0))) / a),
            "cancellation_floor_normF": float(
                eps32 * float(np.max(np.abs(f0))) / a * np.sqrt(M.size)),
        })
    return {"reference_amplitude": float(reference_amp),
            "columns": [int(j) for j in columns],
            "f0_max": float(np.max(np.abs(f0))),
            "eps_float32": eps32, "rows": rows}


# ---------------------------------------------------------------------------
# subjects
# ---------------------------------------------------------------------------


def load_checkpoint(name):
    """A named scOT checkpoint, wrapped exactly as `poseidon.load_expert` wraps T.

    ``FrozenFluidExpert`` already takes the repo id as an argument; only the
    default is Poseidon-T.  Poseidon-B is the same architecture eight layers
    deeper and is the SECOND DONOR this measurement needs, because a tail
    measured on one checkpoint is a fact about that checkpoint.
    """
    import importlib

    W.load_reference()
    ad = importlib.import_module("atlas_windfarm_reference.adapters")
    return ad.FrozenFluidExpert(checkpoint=name, device="cpu", threads=1)


def poseidon_subjects(u_full, v_full, tiling, expert=None):
    ex = expert or PO.load_expert()
    us, vs = tiling.cut(u_full[:tiling.mono_n, :tiling.mono_n]), \
        tiling.cut(v_full[:tiling.mono_n, :tiling.mono_n])
    agents = {}
    for k, name in enumerate(tiling.names):
        agents[name] = PO.PoseidonAgent(
            agent_id=name, u0=us[k], v0=vs[k],
            shared_faces=tiling.artificial_faces(*tiling.offsets[k]), expert=ex)
    return agents


def window_subjects(u_full, v_full, tiling, expose_elliptic):
    us, vs = tiling.cut(u_full[:tiling.mono_n, :tiling.mono_n]), \
        tiling.cut(v_full[:tiling.mono_n, :tiling.mono_n])
    agents = {}
    for k, name in enumerate(tiling.names):
        agents[name] = W.WindowAgent(
            agent_id=name, u0=us[k], v0=vs[k],
            shared_faces=tiling.artificial_faces(*tiling.offsets[k]),
            expose_elliptic=expose_elliptic)
    return agents


def measure(label, agent, port, amplitude, radii, targets, note="", store=None,
            key=None):
    n = agent.n
    base = np.zeros(n)
    t0 = time.perf_counter()
    S, f0 = dense_seam_operator(agent.respond, port, base, amplitude)
    dt_build = time.perf_counter() - t0
    if store is not None and key is not None:
        store[key] = S

    lin = truncation_curve(S, radii, circular=False)
    circ = truncation_curve(S, radii, circular=True)
    dec = column_decay(S)
    prof = dec["profile"]
    d = np.arange(len(prof), dtype=float)
    y = np.array([np.nan if x is None else x for x in prof], dtype=float)
    # fit over the decaying part only: d >= 1, and inside the interior reach
    lim = min(len(prof) - 1, n // 4)
    fits = fit_laws(d[1:lim + 1], y[1:lim + 1])

    sr = support_reach(agent.respond, port, base, amplitude=amplitude,
                       profile_cells=min(64, n // 2))

    return {
        "label": label,
        "agent": agent.agent_id,
        "port": port,
        "n": int(n),
        "amplitude": float(amplitude),
        "build_seconds": float(dt_build),
        "note": note,
        "support_reach": sr.as_dict(),
        "linear": lin,
        "circular": circ,
        "column_decay": dec,
        "decay_fit": fits,
        "r_star_relative_2norm": r_star(lin, "E2_rel", targets["relative"], n),
        "r_star_absolute_2norm": r_star(lin, "E2", targets["absolute"], n),
        "r_star_relative_2norm_circular": r_star(circ, "E2_rel",
                                                 targets["relative"], n),
        "terminal_slope_2norm": terminal_slope(lin, "E2_rel", 48, 64),
        "far_field_spectrum": [far_field_spectrum(S, r) for r in (8, 20, 32)],
        "S_diag_mean": float(np.mean(np.abs(np.diag(S)))),
        "S_offdiag_share": float(
            np.linalg.norm(S - np.diag(np.diag(S)), "fro") / np.linalg.norm(S, "fro")),
    }


# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.perf_counter()

    u_full, v_full = NI.load_state()
    print("state:", u_full.shape, "from out/tier0b/s0_state.npz")

    radii = list(range(0, 65, 1)) + [70, 80, 90, 100, 110, 120, 127]
    # relative targets are dimensionless fractions of ||S||; absolute ones are the
    # defect scales this vault has actually measured on a WindowNS composition
    # (tier0: tau = 3.7014e-4, sigma = 2.6995e-5).  They are quoted as SCALES the
    # tail is compared against, not as this graph's own eps_tol -- see the write-up.
    targets = {"relative": [1e-1, 1e-2, 1e-3, 1e-4, 1e-6],
               "absolute": [3.7014e-4, 2.6995e-5, 1e-6, 1e-9]}

    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state": "out/tier0b/s0_state.npz, developed wake t = 5",
        "radii": radii,
        "targets": targets,
        "subjects": [],
    }
    store = {}

    def report(r):
        results["subjects"].append(r)
        print("%-46s reach %3d  nz %4d/%d  E2rel(20) %.4g  E2rel(64) %.4g  %.1fs"
              % (r["label"], r["support_reach"]["reach"],
                 r["support_reach"]["nonzero"], r["n"],
                 next(x["E2_rel"] for x in r["linear"]["rows"] if x["r"] == 20),
                 next(x["E2_rel"] for x in r["linear"]["rows"] if x["r"] == 64),
                 r["build_seconds"]))

    # --- 1. Poseidon-T, the primary -------------------------------------
    ptil = PO.DEFAULT_TILING
    pag = poseidon_subjects(u_full, v_full, ptil)
    results["poseidon_tiling"] = {"mono_n": ptil.mono_n, "n": ptil.n,
                                  "halo": ptil.halo, "ramp": ptil.ramp,
                                  "declared_reach": 2 * 1}
    for (name, port, amp, note) in [
        ("P00", "xhi:MECH", 1e-2, "primary: tile P00, x-high face"),
        ("P00", "xhi:MECH", 1.0, "amplitude control, 100x"),
        ("P00", "xhi:MECH", 1e-4, "amplitude control, 1/100x"),
        ("P00", "yhi:MECH", 1e-2, "second face, same tile and state"),
        ("P11", "xlo:MECH", 1e-2, "second tile: a DIFFERENT base state"),
    ]:
        ag = pag[name]
        if port.split(":")[0] not in ag.shared_faces:
            print("skip", name, port, "-- not an artificial face here")
            continue
        key = f"poseidon-T_{name}_{port.split(':')[0]}_a{amp:g}"
        r = measure(f"poseidon-T {name} {port} a={amp:g}", ag, port, amp,
                    radii, targets, note, store, key)
        r["subject"] = "poseidon-t"
        r["global_by"] = "architecture"
        r["declared_reach"] = 2
        report(r)

    # --- 1b. Poseidon-B, the SECOND DONOR --------------------------------
    # Same scOT architecture, `depths` [8, 8, 8, 8] against T's [4, 4, 4, 4] and
    # ~5x the parameters, trained on the same corpus.  Its own repo id, its own
    # weights.  If the two agree, the tail is a fact about the architecture; if
    # they disagree, it was a fact about one checkpoint and the ladder's claim --
    # which is about an expert CLASS -- has no support either way.
    try:
        exB = load_checkpoint("camlab-ethz/Poseidon-B")
        results["poseidon_b"] = {"checkpoint": exB.checkpoint,
                                 "n_params": int(exB.n_params)}
        pagB = poseidon_subjects(u_full, v_full, ptil, expert=exB)
        for (name, port, amp, note) in [
            ("P00", "xhi:MECH", 1e-2, "second donor, primary face"),
            ("P00", "yhi:MECH", 1e-2, "second donor, second face"),
            ("P11", "xlo:MECH", 1e-2, "second donor, second tile"),
        ]:
            ag = pagB[name]
            if port.split(":")[0] not in ag.shared_faces:
                continue
            key = f"poseidon-B_{name}_{port.split(':')[0]}_a{amp:g}"
            r = measure(f"poseidon-B {name} {port} a={amp:g}", ag, port, amp,
                        radii, targets, note, store, key)
            r["subject"] = "poseidon-b"
            r["global_by"] = "architecture"
            r["declared_reach"] = 2
            report(r)
    except Exception as exc:                                  # pragma: no cover
        results["poseidon_b"] = {"unavailable": repr(exc)}
        print("Poseidon-B unavailable:", exc)

    # --- 2 & 3. the two classical arrangements --------------------------
    wtil = NI.DEFAULT_TILING
    for expose, sub, gb in [
        (False, "windowns-as-built", "physics (embedded pressure solve)"),
        (True, "windowns-split-step", "nothing -- the elliptic part is exposed"),
    ]:
        wag = window_subjects(u_full, v_full, wtil, expose)
        for (name, port, amp, note) in [
            ("C00", "xhi:MECH", 1e-2, "primary face"),
            ("C11", "xlo:MECH", 1e-2, "second tile: a DIFFERENT base state"),
        ]:
            ag = wag[name]
            if port.split(":")[0] not in ag.shared_faces:
                continue
            key = f"{sub}_{name}_{port.split(':')[0]}_a{amp:g}"
            r = measure(f"{sub} {name} {port} a={amp:g}", ag, port, amp,
                        radii, targets, note, store, key)
            r["subject"] = sub
            r["global_by"] = gb
            r["declared_reach"] = NI.DOMAIN_OF_DEPENDENCE
            report(r)

    # --- 3b. is S an operator, or the instrument's own floor? ------------
    # The three Poseidon amplitudes above disagree, and a linear response cannot.
    # This decides which of them is the operator, on the two solver classes at
    # once so the answer is not a claim about the checkpoint alone.
    amps = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    cols = list(range(8, 128, 8))
    results["amplitude_ladder"] = {}
    for lab, ag in [("poseidon-T P00 xhi", pag["P00"]),
                    ("windowns-as-built C00 xhi",
                     window_subjects(u_full, v_full, wtil, False)["C00"]),
                    ("windowns-split-step C00 xhi",
                     window_subjects(u_full, v_full, wtil, True)["C00"])]:
        lad = amplitude_ladder(ag.respond, "xhi:MECH", ag.n, amps, cols, 1e-2)
        results["amplitude_ladder"][lab] = lad
        print("ladder %-28s " % lab + "  ".join(
            "a=%.0e:%.3g" % (x["amplitude"], x["rel_disagreement_vs_ref"])
            for x in lad["rows"]))

    # --- 4. what a finite halo would buy in the sigma bound -------------
    # Pi(d) is the partition-of-unity weight on cells within d of an artificial
    # face; it is the multiplier in master-error-bound 4.1's overlapping sigma.
    pis = []
    for d in [0, 1, 2, 4, 8, 16, 20, 21, 24, 32, 48, 64, 96, 127]:
        pou = ptil.partition_of_unity(d_cells=d)
        pis.append({"d_cells": int(d),
                    "contaminated_weight": float(pou.contaminated_weight())})
    results["Pi_of_halo"] = pis
    print("Pi(d): " + ", ".join("%d->%.4f" % (p["d_cells"], p["contaminated_weight"])
                                for p in pis))

    results["elapsed_seconds"] = time.perf_counter() - t_all
    npz = os.path.join(OUT, "seam_operators.npz")
    np.savez_compressed(npz, **store)
    results["seam_operators"] = {"path": "out/w173/seam_operators.npz",
                                 "keys": sorted(store)}
    path = os.path.join(OUT, "w173.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", path, "in %.1f s" % results["elapsed_seconds"])


if __name__ == "__main__":
    main()
