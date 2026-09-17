"""W312 -- one plane seam probed at its DECLARED clock, as the anchor.

Tier 76 justified a reduced probe cadence by measuring that beta moved 0.3% over
a 100x range.  **That measurement was taken on the b-c THERM port, which is a
WALL**, and a wall's response is algebraic in the trace: `generate.py::_wall_flux`
subtracts ``T_wall`` directly, so it responds at any cadence.  Measured here, the
wall port's response is flat to four digits from 8 sub-steps to 678.

A **plane** the flow crosses is not that shape.  Its response is the interior
state after the imposed ghost has influenced it, which needs the wave to cross
cells.  Over the same 100x range the nozzle exit plane's response moves **three
orders** and its support goes from 11 of 96 cells to all 96:

    dt_scale   sub-steps   max |dR/dh0|    cells responding
    1e-3          8        4.66484e-11     11 of 96
    1e-2         72        2.18085e-10     29 of 96
    1e-1        720        1.42928e-07     96 of 96
    1e+0       7274        9.79913e-06     96 of 96

So **Tier 76's cadence justification does not transfer from a wall to a plane**,
and every ADVEC number in `w310`'s seam table is the operator the probe could see
in 8 sub-steps rather than the seam's.  This script pays for one of them.

The cost is the point and it is measured, not estimated: at the declared clock
one response call on agent `e` is ~268 s, so a full ``dim M = 41`` probe of e-f
is 84 calls and about **6.3 hours**.  `--modes` caps the interface space so the
anchor is affordable; the cap is a DECLARATION and it is reported.

    python scripts/w312_plane_seam_converged.py --modes 8 --json out/w312.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE
from atlas.probe import ProbeBudget


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--seam", default="e-f")
    ap.add_argument("--modes", type=int, default=8,
                    help="cap on dim M, so the anchor is affordable; reported")
    ap.add_argument("--dt-scale", type=float, default=1.0,
                    help="1.0 is the config's own declared clock")
    a = ap.parse_args(argv)

    print("=" * 88)
    print("W312 -- %s at dt_scale=%g, dim M capped at %d"
          % (a.seam, a.dt_scale, a.modes))
    print("=" * 88)
    t0 = time.perf_counter()
    graph, experts = RE.build_rocket_real(dt_scale=a.dt_scale, m_cap=a.modes)
    conn = [c for c in graph.connections if c.seam_id == a.seam][0]
    x, y = conn.a[0], conn.b[0]
    print("  %s <-> %s, %s, effort_normal=%r"
          % (conn.a, conn.b, conn.port_type.value, conn.effort_normal))
    print("  agent clocks: %s" % {i: experts[i].dt for i in (x, y)})
    print("  probing ... this is the expensive one; the cost IS the measurement")
    sys.stdout.flush()

    from atlas.graph import CaseGraph
    sub = CaseGraph(
        name=graph.name + ":" + a.seam,
        agents=[g for g in graph.agents if g.agent_id in (x, y)],
        connections=[conn], decomposition=graph.decomposition,
        macro_dt=graph.macro_dt, flux_matching=graph.flux_matching,
    )
    t1 = time.perf_counter()
    # NOT ProbeBudget(max_modes=...): a budget shrinks the space and leaves the
    # DECLARED prolongation at its original width, and `Prolongation.prolong`
    # then raises "dim mismatch". The narrower space is declared instead, via
    # build_rocket_real(m_cap=...), which builds the matching prolongation.
    op, tr = RE.seam_operator(sub, budget=ProbeBudget(),
                              probe_state="w312 dt_scale=%g m_cap=%d"
                                          % (a.dt_scale, a.modes))
    el = time.perf_counter() - t1
    calls = sum(experts[i].solver_calls for i in (x, y))
    subs = sum(experts[i].solver_substeps for i in (x, y)
               if hasattr(experts[i], "solver_substeps"))
    res = {"seam": a.seam, "dt_scale": a.dt_scale, "modes_cap": a.modes,
           "dim_M": op.dim_M, "beta": op.beta, "kappa": op.kappa,
           "omega": op.operator_content, "sign_structure": op.sign_structure,
           "null_dim": op.null_dim, "expected_null_dim": op.expected_null_dim,
           "passivity_defect": op.passivity_defect, "is_empty": bool(op.is_empty),
           "one_sided": op.one_sided, "effort_normal": op.effort_normal,
           "solver_calls": calls, "solver_substeps": subs,
           "probe_seconds": el, "total_seconds": time.perf_counter() - t0,
           "blocks": {k: {"share": b.share, "norm_F": float(np.linalg.norm(b.S, "fro")),
                          "n_solves": b.n_solves}
                      for k, b in op.blocks.items()}}
    print()
    print("  dim M            %s" % res["dim_M"])
    print("  beta             %.6g" % res["beta"])
    print("  kappa            %.6g" % res["kappa"])
    print("  omega            %.6g" % res["omega"])
    print("  sign structure   %s" % res["sign_structure"])
    print("  null dim         %s (declared %s)" % (res["null_dim"],
                                                   res["expected_null_dim"]))
    print("  empty            %s" % res["is_empty"])
    print("  blocks           %s" % {k: round(v["share"], 4)
                                     for k, v in res["blocks"].items()
                                     if v["share"] is not None})
    print()
    print("  COST: %d response calls, %d CFL sub-steps, %.0f s of probe"
          % (calls, subs, el))
    print("  Extrapolated to the full dim M = 41 this seam declares: %.1f hours"
          % (el / max(1, a.modes + 1) * 42 * 2 / 3600.0))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, default=str)
        print()
        print("artifact written to %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
