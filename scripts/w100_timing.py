"""Wall time per macro-step per agent, INTERLEAVED -- Claim B's other half.

Claim B has two halves and the error is only one of them: composition is supposed
to cost **O(1) integration work per added agent**, so a per-agent cost that climbs
with the graph is as much a falsification as a super-linear error would be.

**Why this is a separate script, and why it is built the way it is.**  The driver
times the marches it runs anyway, and those timings are *provenance* rather than
a measurement: its rungs are marched at different times and some are resumed from
a cache. The obvious repair -- run every rung back to back in one process -- was
tried first and is **not enough**, because this is a shared desktop and the load
moves inside a single pass. Two passes of that script, same code, same inputs,
minutes apart, returned per-agent costs of 27.0 and 103.1 ms for the same rung
and per-agent ratios of 4.49x and 3.03x. A ratio that moves by half its own value
between two runs is not a measurement of anything.

So this does three things instead:

  * **interleaves.** One repeat visits every (rung, expert) row in turn, so a
    transient that lands inside a repeat is spread across all rows rather than
    charged to whichever row was unlucky.
  * **takes the minimum** over repeats, which is the standard estimator for a
    wall time that can only be contaminated upward.
  * **reports its own reproducibility** -- the spread between the best and the
    second-best repeat of each row -- so the reader can see whether the trend is
    bigger than the noise. **If it is not, the honest output is that wall time is
    not measurable here**, and this prints exactly that.

It also divides out the solver's own sub-step count. `WindowNS` picks ``n_sub``
from the advective CFL, so a bigger farm with deeper wakes sub-steps more -- and
that is the FLOW getting faster, not the composition costing more per agent.
Cost per agent per SUB-STEP is the arithmetic unit Claim B's O(1) is about.
Poseidon-T is a one-shot map with no sub-steps, so its count is 1 by construction.

    python scripts/w100_timing.py --steps 8 --repeats 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas.cases import scaling_ladder as sl                         # noqa: E402
from atlas.cases import wake_array as wa                             # noqa: E402
from w100_scaling_ladder import OUT, _band, _forcing, composed_step  # noqa: E402

#: Above this relative spread between a row's best and second-best repeat, the
#: trend across rungs is not separable from the machine's own noise and the
#: script says so instead of publishing a ratio.  0.25 is generous: the smallest
#: effect worth reporting here is O(1) against 24x.
REPRODUCIBILITY_LIMIT = 0.25


def timed_march(r, kind, steps):
    """One march, timed, from the freestream with every disk live."""
    ex = (wa.scaled_expert() if kind == "poseidon"
          else wa.reference_solver(wa.NU_REF))
    rotors = {x.rotor_id for x in r.tiling.rotors}
    u = np.ones(r.shape)
    v = np.zeros(r.shape)
    t0 = time.perf_counter()
    for _ in range(steps):
        fx, _rec = _forcing(r, u, rotors)
        u, v, _loc = composed_step(r, u, v, fx, kind, ex)
        u, v = _band(r, u, v)
    return (time.perf_counter() - t0,
            int(getattr(ex, "last_substeps", 1) or 1),
            float(np.max(np.abs(u))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--artifact", default=os.path.join(OUT, "w100.json"))
    args = ap.parse_args(argv)

    rungs = sl.ladder()
    rows = [(r, kind) for r in rungs for kind in ("reference", "poseidon")]

    print("warming both experts (loader, FFT plans, DCT tables) ...", flush=True)
    for kind in ("reference", "poseidon"):
        timed_march(rungs[0], kind, 1)

    samples: dict[tuple[str, str], list] = {(r.label, k): [] for r, k in rows}
    for rep in range(args.repeats):
        t0 = time.perf_counter()
        for r, kind in rows:
            samples[(r.label, kind)].append(timed_march(r, kind, args.steps))
        print(f"  repeat {rep + 1}/{args.repeats} done "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    out_rows, worst_spread = [], 0.0
    for r in rungs:
        row = {"rung": r.label, "n_windows": r.n_windows, "n_cells": r.n_cells,
               "n_overlaps": r.n_overlaps, "steps": args.steps}
        for kind in ("reference", "poseidon"):
            s = sorted(samples[(r.label, kind)], key=lambda x: x[0])
            best, n_sub, umax = s[0]
            second = s[1][0] if len(s) > 1 else best
            spread = (second - best) / best if best else 0.0
            worst_spread = max(worst_spread, spread)
            row[kind] = {
                "seconds_total": best,
                "per_macro_step": best / args.steps,
                "per_macro_step_per_agent": best / args.steps / r.n_windows,
                "per_macro_step_per_cell_ns": best / args.steps / r.n_cells * 1e9,
                "substeps_last": n_sub,
                "u_max": umax,
                "per_substep_per_agent": best / args.steps / r.n_windows / n_sub,
                "all_repeats_s": [x[0] for x in s],
                "spread_best_to_second": spread,
            }
        out_rows.append(row)

    print(f"\n  {'rung':5s} {'win':>4s} {'cells':>8s} "
          f"{'WindowNS':>10s} {'Poseidon':>10s}  ms/step/agent   "
          f"{'n_sub':>5s} {'u_max':>6s}  {'WNS/sub':>8s}")
    for row in out_rows:
        print(f"  {row['rung']:5s} {row['n_windows']:4d} {row['n_cells']:8d} "
              f"{row['reference']['per_macro_step_per_agent']*1e3:10.1f} "
              f"{row['poseidon']['per_macro_step_per_agent']*1e3:10.1f}"
              f"                 {row['reference']['substeps_last']:5d} "
              f"{row['reference']['u_max']:6.3f}  "
              f"{row['reference']['per_substep_per_agent']*1e3:8.2f}")

    base = next((x for x in out_rows if x["n_windows"] == 2), out_rows[0])
    top = out_rows[-1]
    out = {
        "note": ("interleaved: one repeat visits every (rung, expert) row in "
                 "turn, minimum over repeats, on a SHARED desktop"),
        "steps": args.steps, "repeats": args.repeats,
        "date": time.strftime("%Y-%m-%d"), "rows": out_rows,
        "baseline_rung": base["rung"], "top_rung": top["rung"],
        "worst_spread_best_to_second": worst_spread,
        "reproducibility_limit": REPRODUCIBILITY_LIMIT,
        "trustworthy": bool(worst_spread <= REPRODUCIBILITY_LIMIT),
        "per_agent_ratio": {
            k: top[k]["per_macro_step_per_agent"]
            / base[k]["per_macro_step_per_agent"]
            for k in ("reference", "poseidon")},
        "per_substep_per_agent_ratio": {
            k: top[k]["per_substep_per_agent"] / base[k]["per_substep_per_agent"]
            for k in ("reference", "poseidon")},
    }
    # **The verdict is a comparison, not a threshold.** A noise floor does not
    # invalidate a trend larger than it, and a threshold on the floor alone
    # would throw away a 5x effect because one row wobbled by a third. What
    # decides whether a ratio may be quoted is whether the ratio's own distance
    # from 1 exceeds the spread -- so both are computed and both are reported.
    for k in ("reference", "poseidon"):
        ratio = out["per_agent_ratio"][k]
        effect = abs(np.log(ratio))
        out.setdefault("separable", {})[k] = {
            "ratio": ratio,
            "log_effect": effect,
            "log_noise": float(np.log1p(worst_spread)),
            "separable": bool(effect > np.log1p(worst_spread)),
        }
    out["trustworthy"] = bool(worst_spread <= REPRODUCIBILITY_LIMIT)
    out["any_separable"] = any(v["separable"] for v in out["separable"].values())

    print(f"\n  reproducibility: worst best-to-second spread over "
          f"{args.repeats} interleaved repeats = {100 * worst_spread:.1f}%"
          f"  (limit {100 * REPRODUCIBILITY_LIMIT:.0f}%)")
    print(f"  per-agent cost, {top['rung']} against {base['rung']}: "
          f"WindowNS {out['per_agent_ratio']['reference']:.2f}x, "
          f"Poseidon-T {out['per_agent_ratio']['poseidon']:.2f}x  "
          f"(O(1) per added agent is 1.00x)")
    print(f"  per agent per SUB-STEP: "
          f"WindowNS {out['per_substep_per_agent_ratio']['reference']:.2f}x, "
          f"Poseidon-T {out['per_substep_per_agent_ratio']['poseidon']:.2f}x "
          f"-- the flow's own CFL divided out")
    for k, v in out["separable"].items():
        print(f"    {k:10s} |log ratio| = {v['log_effect']:.3f} against a "
              f"log-noise of {v['log_noise']:.3f}: "
              f"{'SEPARABLE -- the direction may be quoted' if v['separable'] else 'NOT separable'}")
    if not out["trustworthy"]:
        print(f"\n  The spread is above the limit, so no VALUE here is a "
              "measurement of this composition:")
        print("  it is a measurement of a shared desktop. Only the ratios marked "
              "separable above")
        print("  carry a direction, and F3 -- integration hours per added expert "
              "-- stays unmeasured.")

    if os.path.isfile(args.artifact):
        with open(args.artifact, encoding="utf-8") as fh:
            art = json.load(fh)
        art["timing"] = out
        with open(args.artifact, "w", encoding="utf-8") as fh:
            json.dump(art, fh, indent=1)
        print(f"\n  merged into {args.artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
