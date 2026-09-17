"""W310 -- R0 complete: all seven rocket agents on real physics, every seam probed.

Tier 76 wired two of the seven and measured one seam.  This wires the remaining
five and probes all seven seams, which is R0's completion.

Stages:

    setup       the seven agents, their meshes, clocks and operating points
    nozzle      W311's consequence -- the film coefficient on the UNDECLARED
                nozzle wall, against the chamber wall's measured 183.87
    seams       every seam: beta, kappa, the null check against a DECLARED
                expectation, omega, the sign structure, and the cost
    reach       W301's signature on wall ports against plane ports -- the
                cross-check that the saturation is about `wall_noslip` and not
                about `compressible2d`
    compile     the whole graph through the compiler

ASCII output only -- the console is cp1252.

    python scripts/w310_rocket_r0_complete.py --json out/w310.json
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
from atlas.probe import support_reach

#: Every agent's declared clock multiplied by this. A DECLARATION: the chamber
#: alone is 3310 CFL sub-steps at its own 1 ms, and Tier 76 measured beta moving
#: 0.3% over a 100x cadence range against an 84.75x cost in sub-steps.
DT_SCALE = 1.0e-3


def _fmt(x, n=6):
    if x is None:
        return "None"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return ("%." + str(n) + "g") % float(x)


def stage_setup(out):
    print("=" * 96)
    print("SETUP -- all seven agents on real physics")
    print("=" * 96)
    t0 = time.perf_counter()
    graph, experts = RE.build_rocket_real(dt_scale=DT_SCALE)
    el = time.perf_counter() - t0
    rows = []
    print("  %-4s %-9s %-11s %-11s %s" % ("id", "cells", "dt (probe)", "dt (config)", "ports"))
    cfg = RE.rocket_config()
    for a in graph.agents:
        aid = a.agent_id
        ex = experts[aid]
        cells = (int(np.prod(ex._blk.shape)) if aid != "c"
                 else int(np.prod(RE.shell_panel().shape)))
        rows.append({"agent": aid, "cells": cells,
                     "dt_probe": a.capabilities.dt_native,
                     "dt_config": float(cfg.dt_model[aid]),
                     "ports": [p.name for p in a.capabilities.ports]})
        print("  %-4s %-9d %-11.4g %-11.4g %s"
              % (aid, cells, a.capabilities.dt_native, cfg.dt_model[aid],
                 [p.name for p in a.capabilities.ports]))
    print()
    print("  built in %s s;  dt_scale = %s" % (_fmt(el, 3), _fmt(DT_SCALE)))
    print("  UNDECLARED interfaces this graph cannot carry:")
    for k, why in sorted(RE.UNDECLARED_INTERFACES.items()):
        print("    %-5s %s" % (k, why))
    out["setup"] = {"agents": rows, "dt_scale": DT_SCALE, "build_seconds": el,
                    "undeclared": RE.UNDECLARED_INTERFACES}
    return graph, experts


def stage_nozzle(experts, out):
    """W311's consequence: does the undeclared nozzle wall matter?"""
    print()
    print("=" * 96)
    print("NOZZLE -- the film coefficient on the interface nobody declared")
    print("=" * 96)
    res = {}
    for aid, label in (("b", "chamber wall (DECLARED, b-c)"),
                       ("e", "nozzle wall  (UNDECLARED, e-c)")):
        h = np.asarray(experts[aid].wall_h(neighbour="c"), float)
        res[aid] = {"label": label, "mean": float(h.mean()),
                    "min": float(h.min()), "max": float(h.max())}
        print("  %-32s h mean %-12s min %-12s max %s"
              % (label, _fmt(h.mean(), 6), _fmt(h.min(), 6), _fmt(h.max(), 6)))
    r = res["e"]["mean"] / max(1e-300, res["b"]["mean"])
    res["nozzle_over_chamber"] = float(r)
    print()
    print("  nozzle / chamber = %s" % _fmt(r, 5))
    print("  The prediction was >2x. Read the number, not this line.")
    out["nozzle"] = res
    return res


def stage_seams(graph, experts, out):
    print()
    print("=" * 96)
    print("SEAMS -- every declared seam, probed")
    print("=" * 96)
    rows = []
    print("  %-6s %-6s %-5s %-12s %-11s %-9s %-10s %-9s %s"
          % ("seam", "type", "dimM", "beta", "kappa", "omega", "sign", "null", "cost"))
    print("  " + "-" * 92)
    total_calls = 0
    for conn in graph.connections:
        c0 = {a: (experts[a].solver_calls if hasattr(experts[a], "solver_calls") else 0)
              for a in (conn.a[0], conn.b[0])}
        t0 = time.perf_counter()
        try:
            op, _tr = RE.seam_operator(graph_for(graph, conn), probe_state="w310")
        except Exception as exc:                          # noqa: BLE001
            rows.append({"seam": conn.seam_id, "error": "%s: %s"
                         % (type(exc).__name__, str(exc)[:110])})
            print("  %-6s ERROR %s: %s" % (conn.seam_id, type(exc).__name__,
                                           str(exc)[:60]))
            continue
        el = time.perf_counter() - t0
        calls = sum((experts[a].solver_calls if hasattr(experts[a], "solver_calls")
                     else 0) - c0[a] for a in c0)
        total_calls += calls
        null_ok = (op.null_dim == op.expected_null_dim)
        rows.append({"seam": conn.seam_id, "port_type": conn.port_type.value,
                     "dim_M": op.dim_M, "beta": op.beta, "kappa": op.kappa,
                     "omega": op.operator_content,
                     "sign_structure": op.sign_structure,
                     "null_dim": op.null_dim,
                     "expected_null_dim": op.expected_null_dim,
                     "excess_null": op.excess_null_directions,
                     "passivity_defect": op.passivity_defect,
                     "is_empty": bool(op.is_empty),
                     "effort_normal": op.effort_normal,
                     "one_sided": op.one_sided,
                     "solver_calls": calls, "seconds": el})
        print("  %-6s %-6s %-5d %-12s %-11s %-9s %-10s %-9s %d calls / %s s"
              % (conn.seam_id, conn.port_type.value, op.dim_M,
                 _fmt(op.beta, 5), _fmt(op.kappa, 5), _fmt(op.operator_content, 4),
                 op.sign_structure or "-",
                 ("%d==%d" % (op.null_dim, op.expected_null_dim)) if null_ok
                 else ("%d!=%d" % (op.null_dim, op.expected_null_dim)),
                 calls, _fmt(el, 3)))
        if op.is_empty:
            print("        ^ EMPTY: Lambda == 0. The interface problem is empty "
                  "rather than ill-conditioned")
    out["seams"] = rows
    out["total_solver_calls"] = total_calls
    print()
    print("  total across every seam: %d boundary_response calls" % total_calls)
    return rows


def graph_for(graph, conn):
    """A two-agent view of one seam, so `seam_operator` probes exactly it."""
    from atlas.graph import CaseGraph

    ids = {conn.a[0], conn.b[0]}
    return CaseGraph(
        name=graph.name + ":" + conn.seam_id,
        agents=[a for a in graph.agents if a.agent_id in ids],
        connections=[conn],
        decomposition=graph.decomposition,
        macro_dt=graph.macro_dt,
        flux_matching=graph.flux_matching,
        note=graph.note,
    )


def stage_reach(graph, experts, out):
    """W301's signature: wall ports against plane ports."""
    print()
    print("=" * 96)
    print("REACH -- does the trace reach the solver? wall ports vs PLANE ports")
    print("=" * 96)
    print("  `wall_noslip` clamps its ghost at max(2 T_wall - T_i, 20); "
          "`prescribed` has no clamp.")
    print()
    rows = []
    for a in graph.agents:
        if a.agent_id == "c":
            continue
        ex = experts[a.agent_id]
        for p in a.capabilities.ports:
            nb = p.name.split(":")[0]
            side, kind = ex.face(nb)
            base = ex.base_trace(nb)
            sr = support_reach(ex.respond, p.name, base)
            tripped = sr.exact_zeros == sr.n - 1
            rows.append({"agent": a.agent_id, "port": p.name, "kind": kind,
                         "n": sr.n, "nonzero": sr.nonzero,
                         "exact_zeros": sr.exact_zeros, "tripped": bool(tripped),
                         "peak": sr.peak})
            print("  %-3s %-10s %-6s n=%-5d nonzero=%-5d exact_zeros=%-5d "
                  "tripped=%-6s peak=%s"
                  % (a.agent_id, p.name, kind, sr.n, sr.nonzero, sr.exact_zeros,
                     tripped, _fmt(sr.peak, 5)))
    walls = [r for r in rows if r["kind"] == "wall"]
    planes = [r for r in rows if r["kind"] == "plane"]
    print()
    print("  wall  ports: %d of %d trip W301's signature"
          % (sum(r["tripped"] for r in walls), len(walls)))
    print("  plane ports: %d of %d trip W301's signature"
          % (sum(r["tripped"] for r in planes), len(planes)))
    out["reach"] = rows
    return rows


def stage_compile(graph, out):
    from collections import Counter

    from atlas.compiler import compile_scheme

    print()
    print("=" * 96)
    print("COMPILE -- the whole seven-agent graph")
    print("=" * 96)
    t0 = time.perf_counter()
    r = compile_scheme(graph)
    el = time.perf_counter() - t0
    ref = dict(Counter("%s/%s" % (d.layer, d.rule) for d in r.decisions.refusals))
    dec = dict(Counter("%s/%s" % (d.layer, d.rule) for d in r.decisions.decertifications))
    print("  verdict %s   %d refusals, %d decertifications   (%s s)"
          % (r.verdict, len(r.decisions.refusals),
             len(r.decisions.decertifications), _fmt(el, 3)))
    print("  refusals       : %s" % (ref or "none"))
    print("  decertifications:")
    for k, v in sorted(dec.items()):
        print("      %-18s %d" % (k, v))
    out["compile"] = {"verdict": str(r.verdict), "refusals": ref,
                      "decertifications": dec, "seconds": el,
                      "n_refusals": len(r.decisions.refusals),
                      "n_decertifications": len(r.decisions.decertifications)}
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--stage", default="all")
    a = ap.parse_args(argv)
    want = (set(a.stage.split(",")) if a.stage != "all"
            else {"setup", "nozzle", "seams", "reach", "compile"})
    out = {"script": "w310_rocket_r0_complete.py", "date": "2026-09-17",
           "build_repo_commit": "0a407b7", "dt_scale": DT_SCALE}
    graph, experts = stage_setup(out)
    if "nozzle" in want:
        stage_nozzle(experts, out)
    if "seams" in want:
        stage_seams(graph, experts, out)
    if "reach" in want:
        stage_reach(graph, experts, out)
    if "compile" in want:
        stage_compile(graph, out)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print()
        print("artifact written to %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
