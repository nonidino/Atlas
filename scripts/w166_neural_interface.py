"""W166 -- the neural interface predictor, built and measured against its controls.

Runs `atlas.cases.neural_interface` end to end and writes `out/w166/w166.json`.

Five parts, and the order is the order the questions were settled in:

  A  the exchange           what one declaration -- `elliptic_subsolve` -- does to
                            the coupling's convergence RATE, with the projection
                            separated from the exposure as its own control
  B  the halo threshold     where the iterated scheme's contraction crosses one,
                            against the agent's declared domain of dependence
  C  the predictors         skill, sweeps and total cost for five starts, in both
                            arrangements, with the start-independence check that
                            is the safety claim
  D  the accelerators       Richardson, alpha_star undamped and damped, and
                            Newton on the probed S -- priced in EXPERT CALLS
  E  the compile            the verdict, and the structural half of "a predictor
                            is not an agent"

Run:  python scripts/w166_neural_interface.py [--parts A,B,C,D,E] [--out out/w166]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# torch and numpy's MKL each ship an OpenMP runtime and loading both in one
# process aborts the first time numpy takes an SVD after the checkpoint is in
# memory. Set before numpy or torch is imported, because neither reads it after.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from atlas import Budget, compile_scheme                            # noqa: E402
from atlas.cases import neural_interface as NI                      # noqa: E402

#: The two arrangements the whole study contrasts.  ``expose`` is W100's split
#: and it is the field `L2/R10` turns on; ``project`` is the composition layer's
#: own global Leray projection (L6/C2), separated from it so the rate change can
#: be attributed to one of the two rather than to the pair.
MODES = {
    "split-step": dict(expose_elliptic=True, project=True),
    "as-built": dict(expose_elliptic=False, project=False),
}


def _fmt(x, n=5):
    return "None" if x is None else (f"{x:.{n}g}" if isinstance(x, float) else str(x))


# ===========================================================================
# A -- what the elliptic declaration does to the rate
# ===========================================================================


def part_a(u, v):
    print("\n=== A. the exchange, at halo "
          f"{NI.DEFAULT_TILING.halo} ===")
    rows = []
    for label, expose, project in (
        ("as-built (per-window pressure solve)", False, False),
        ("split-step + global projection", True, True),
        ("split-step, no projection [control]", True, False),
    ):
        ex = NI.SchwarzExchange(u, v, expose_elliptic=expose, project=project)
        t0 = time.perf_counter()
        out = ex.march(ex.u0, ex.v0, cap=500, tol=1e-13)
        rows.append({
            "arrangement": label, "expose_elliptic": expose, "project": project,
            "sweeps": out["sweeps"], "contraction": out["contraction"],
            "last_change": out["changes"][-1] if out["changes"] else None,
            "first_change": out["changes"][0] if out["changes"] else None,
            "diverged": out["diverged"], "solves": ex.solves,
            "seconds": time.perf_counter() - t0,
        })
        r = rows[-1]
        print(f"  {label:38s} {r['sweeps']:4d} sweeps  contraction "
              f"{_fmt(r['contraction'])}  last change {_fmt(r['last_change'])}")
    emb = next(r for r in rows if not r["expose_elliptic"])
    exp = next(r for r in rows if r["expose_elliptic"] and r["project"])
    # Sweeps to gain one decade is log(10) / |log(contraction)|, so the ratio of
    # the two is what a practitioner pays for the declaration. Quoted from the
    # asymptotic rate rather than from the sweep counts, because the two
    # arrangements stopped at different places.
    per_decade = {r["arrangement"]: float(np.log(10.0) / abs(np.log(r["contraction"])))
                  for r in rows if r["contraction"] and np.isfinite(r["contraction"])}
    ratio = (per_decade[emb["arrangement"]] / per_decade[exp["arrangement"]]
             if emb["arrangement"] in per_decade and exp["arrangement"] in per_decade
             else None)
    for k, val in per_decade.items():
        print(f"     sweeps per decade, {k:38s} {val:8.2f}")
    print(f"  -> exposing the elliptic part is worth {ratio:.1f}x in sweeps")
    return {"rows": rows, "sweeps_per_decade": per_decade,
            "embedded_over_exposed": ratio}


# ===========================================================================
# B -- the halo threshold
# ===========================================================================


def part_b(u, v):
    print(f"\n=== B. the halo threshold (declared reach "
          f"{NI.DOMAIN_OF_DEPENDENCE} cells) ===")
    out = {}
    for mode, kw in MODES.items():
        rows = NI.halo_convergence(u, v, **kw)
        thr = NI.halo_threshold(rows)
        out[mode] = {"rows": rows, "threshold": thr}
        print(f"  -- {mode} --")
        for r in rows:
            print(f"     halo {r['halo']:3d}  mono_n {r['mono_n']:4d}  "
                  f"{r['sweeps']:3d} sweeps  contraction {_fmt(r['contraction'])}"
                  f"   {'converging' if r['converging'] else 'DIVERGING'}")
        print(f"     bracket: diverges up to {thr['largest_diverging']}, "
              f"converges from {thr['smallest_converging']}; declared reach "
              f"{thr['declared_reach']}; brackets it: {thr['bracket_contains_reach']}")
    return out


# ===========================================================================
# C -- the predictors
# ===========================================================================


def part_c(u, v, tol_frac=1e-3, cap=400, with_neural=True):
    print("\n=== C. the predictors ===")
    out = {}
    for mode, kw in MODES.items():
        ex = NI.SchwarzExchange(u, v, **kw)
        su, sv, fp = ex.fixed_point(cap=500, tol=1e-13)
        star = (su, sv)
        d_cold = NI._rms(ex.u0 - su, ex.v0 - sv)
        tol = tol_frac * d_cold
        print(f"  -- {mode} --  fixed point in {fp['sweeps']} sweeps, last change "
              f"{_fmt(fp['changes'][-1])}, contraction {_fmt(fp['contraction'])}")
        print(f"     ||g_cold - g*|| = {_fmt(d_cold)},  tol = {_fmt(tol)}")

        preds = [NI.HoldOldRing(), NI.OneSweep(ex.tiling)]
        if with_neural:
            preds.append(NI.Neural())
        rows, amp_ref = [], None
        for p in preds:
            r = NI.predictor_skill(ex, p, star, tol, cap=cap)
            if isinstance(p, NI.Neural):
                amp_ref = r["perturbation"]
            rows.append(r)
        # The floor control is matched to the arm it controls: the neural one if
        # it ran, and otherwise the best available field predictor.  A control at
        # some other amplitude is not a control.
        if amp_ref is None:
            amp_ref = max(r["perturbation"] for r in rows)
        rows.append(NI.predictor_skill(
            ex, NI.AmplitudeMatchedNoise(amplitude=amp_ref), star, tol, cap=cap))
        rows.append(NI.predictor_skill(ex, NI.Oracle(star), star, tol, cap=cap))

        print(f"     {'arm':34s} {'skill':>8s} {'sweeps':>7s} {'cost':>8s} "
              f"{'total':>8s}  final")
        for r in rows:
            print(f"     {r['predictor']:34s} {r['skill']:+8.4f} {r['sweeps']:7d} "
                  f"{r['cost_solves']:8.4g} {r['total_solves']:8.4g}  "
                  f"{_fmt(r['final_distance'])}")
        ind = NI.start_independence(rows)
        print(f"     start independence: worst pair "
              f"{_fmt(ind['worst_pair_disagreement'])} over {ind['arms']} arms, "
              f"against tol {_fmt(tol)}")

        cold = next(r for r in rows if r["predictor"].startswith("hold old"))
        one = next(r for r in rows if r["predictor"].startswith("one classical"))
        saved = cold["sweeps"] - one["sweeps"]
        n_win = len(ex.tiling.names)
        print(f"     ceiling: the one-sweep predictor saves {saved} sweep(s) of "
              f"{cold['sweeps']}; a predictor breaks even below "
              f"{saved * n_win} solves")
        neural = next((r for r in rows if r["predictor"].startswith("poseidon")), None)
        if neural is not None and np.isfinite(neural["cost_solves"]):
            print(f"     poseidon costs {neural['cost_solves']:.1f} solves, i.e. "
                  f"{neural['cost_solves'] / max(saved * n_win, 1e-9):.0f}x the "
                  f"break-even")
        out[mode] = {
            "rows": [{k: val for k, val in r.items() if k != "fixed_point"}
                     for r in rows],
            "start_independence": ind, "tol": tol, "d_cold": d_cold,
            "fixed_point_sweeps": fp["sweeps"],
            "fixed_point_last_change": fp["changes"][-1] if fp["changes"] else None,
            "fixed_point_contraction": fp["contraction"],
            "ceiling_sweeps_saved": saved,
            "break_even_solves": saved * n_win,
        }
    return out


# ===========================================================================
# D -- the accelerator ladder
# ===========================================================================


def part_d(u, v):
    print("\n=== D. the accelerators, priced in expert calls ===")
    out = {}
    for mode, kw in MODES.items():
        experts = NI.make_experts(u, v, expose_elliptic=kw["expose_elliptic"])
        seam = NI.MultiplierSeam(a=experts["C00"], b=experts["C10"])
        pr = seam.probe()
        print(f"  -- {mode} --  kappa {pr['kappa']:.4f} -> "
              f"{pr['kappa_preconditioned']:.4f} preconditioned, "
              f"rho(D^-1 S) = {pr['rho_D_inv_S']:.4f}, off-diagonal share "
              f"{pr['off_diagonal_share']:.4f}, probe {pr['probe_calls']} calls")
        lad = NI.accelerator_ladder(seam, pr)
        for r in lad:
            print(f"     tol {r['tol_frac']:g}  {r['accelerator']:34s} "
                  f"{r['iterations']:4d} it  {r['sweep_calls']:5d} + "
                  f"{r['probe_calls']} probe = {r['total_calls']:5d}   "
                  f"{'ok ' if r['converged'] else 'FAIL'} "
                  f"r={_fmt(r['final_residual'])}")
        out[mode] = {
            "probe": {k: (val.tolist() if isinstance(val, np.ndarray) else val)
                      for k, val in pr.items() if k != "blocks"},
            "ladder": lad,
        }
    return out


# ===========================================================================
# E -- the compile
# ===========================================================================


def part_e(u, v):
    print("\n=== E. the compile ===")
    out = {}
    for mode, kw in MODES.items():
        g, _ = NI.build(u, v, expose_elliptic=kw["expose_elliptic"])
        res = compile_scheme(g, budget=Budget(allow_probe=False),
                             probe_state="developed wake, t = 5, dt = 0.05, nu = 1/255")
        rules = res.decisions.cited_rules()
        not_admitted = sorted({f"{d.layer}/{d.rule}"
                               for d in res.decisions.decertifications})
        zero = np.zeros((1, 1))
        preds = [NI.HoldOldRing(), NI.OneSweep(), NI.Oracle((zero, zero)),
                 NI.AmplitudeMatchedNoise(amplitude=0.0)]
        not_agent = {p.name: NI.predictor_is_not_an_agent(g, p) for p in preds}
        print(f"  {mode:12s} verdict {res.verdict}   agents {len(g.agents)}  "
              f"seams {len(g.connections)}   not-admitted rules {not_admitted}")
        for d in res.decisions.refusals[:3]:
            print(f"     REFUSE {d.layer}/{d.rule}: {d.message[:100]}")
        print(f"     predictor is not an agent: {not_agent}")
        out[mode] = {
            "verdict": str(res.verdict),
            "n_agents": len(g.agents), "n_seams": len(g.connections),
            "n_refusals": len(res.decisions.refusals),
            "refusals": [f"{d.layer}/{d.rule}: {d.message}"
                         for d in res.decisions.refusals],
            "not_admitted_rules": not_admitted,
            "cited_rules": rules,
            "predictor_is_not_an_agent": not_agent,
        }
    return out


# ===========================================================================


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,B,C,D,E")
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w166"))
    ap.add_argument("--no-neural", action="store_true",
                    help="skip the checkpoint (parts A, B, D, E need no torch)")
    a = ap.parse_args()
    parts = {p.strip().upper() for p in a.parts.split(",") if p.strip()}
    os.makedirs(a.out, exist_ok=True)

    u, v = NI.load_state(_ROOT)
    result = {
        "state": "out/tier0b/s0_state.npz, developed wake t = 5",
        "mono_n": NI.DEFAULT_TILING.mono_n,
        "window_cells": NI.DEFAULT_TILING.n,
        "halo": NI.DEFAULT_TILING.halo,
        "ramp": NI.DEFAULT_TILING.ramp,
        "domain_of_dependence": NI.DOMAIN_OF_DEPENDENCE,
        "dt": NI.MACRO_DT, "nu": NI.NU,
    }
    t0 = time.perf_counter()
    if "A" in parts:
        result["A_exchange"] = part_a(u, v)
    if "B" in parts:
        result["B_halo"] = part_b(u, v)
    if "C" in parts:
        result["C_predictors"] = part_c(u, v, with_neural=not a.no_neural)
    if "D" in parts:
        result["D_accelerators"] = part_d(u, v)
    if "E" in parts:
        result["E_compile"] = part_e(u, v)
    result["seconds"] = time.perf_counter() - t0

    path = os.path.join(a.out, "w166.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, default=float)
    print(f"\nwrote {path}  ({os.path.getsize(path)} bytes, "
          f"{result['seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
