"""W348's controls, measured rather than assumed.

    python scripts/w348_controls.py     # -> out/workbench/records/w348/controls-<time>.json

1. **Negative controls.**  The graphs R10 was derived on must still refuse at
   ``L2/R10``: Tier 0's four windows as built (`window_ns.build(mode="as-built")`,
   the graph R10 came out of, on the developed wake ``out/tier0b/s0_state.npz``)
   and CS-S1's arrangement with the pressure solve embedded
   (`neural_interface.build(expose_elliptic=False)`, W168's 466 sweeps).  Neither
   declares ``elliptic_data_from_ports``, and that is checked too.
2. **The positive cases.**  Every workbench example, compiled: none refuses at R10
   any more, and the conduction and elasticity examples that did carry ``L5/R10``
   as a decertification.
3. **The three outcomes of `_r10_scheme`** on ``wall-2``'s graph: the workbench's
   own budget (``direct-schur``) decertifies; with ``allow_direct_schur=False``
   the accelerator sweeps, and the outcome follows the scheme's ``eps_tol``.

Each check is an assertion; the record is written before the first one can fail,
so a failing control leaves its evidence.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                              # noqa: E402

from atlas import Budget, compile_scheme                        # noqa: E402


def r10_rows(result) -> list[list[str]]:
    return [[d.layer, d.rule, d.verdict.value, d.subject or ""]
            for d in result.decisions if d.rule.startswith("R10")]


def summary(result, graph) -> dict:
    return {"graph": graph.name, "agents": len(graph.agents),
            "verdict": result.verdict.value,
            "accelerator": getattr(result.scheme.accelerator, "value", None)
            if result.scheme else None,
            "eps_tol": getattr(result.scheme, "eps_tol", None) if result.scheme else None,
            "r10": r10_rows(result),
            "refusals": sorted({f"{d.layer}/{d.rule}" for d in result.decisions.refusals}),
            "declares_ports": sorted(a.agent_id for a in graph.agents
                                     if a.capabilities.elliptic_data_from_ports)}


def main() -> int:
    out: dict = {"when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "script": "scripts/w348_controls.py"}
    t0 = time.perf_counter()

    # 1. the negative controls
    from atlas.cases import neural_interface as NI
    from atlas.cases import window_ns as W
    st = np.load(os.path.join(ROOT, "out", "tier0b", "s0_state.npz"))
    g0, _ = W.build(st["u"], st["v"], mode="as-built")
    out["tier0_as_built"] = summary(compile_scheme(g0), g0)
    u, v = NI.load_state()
    g1, _ = NI.build(u, v, expose_elliptic=False)
    out["cs_s1_embedded"] = summary(compile_scheme(g1, budget=Budget(allow_probe=False)), g1)

    # 2. every workbench example
    from atlas.workbench.compile import case_graph, compile_case
    from atlas.workbench.spec import EXAMPLES, example_case
    rows = {}
    for key in EXAMPLES:
        s = compile_case(example_case(key))
        rows[key] = {"family": s.family, "verdict": s.verdict,
                     "refused_before": s.refused_before,
                     "r10": [[r["layer"], r["rule"], r["verdict"]] for r in s.other
                             if r["rule"].startswith("R10")]}
    out["workbench"] = rows

    # 3. the three outcomes, on wall-2's graph
    built = case_graph(example_case("wall-2"))
    g2 = built[0] if isinstance(built, tuple) else built
    out["wall2_direct"] = summary(compile_scheme(g2), g2)
    out["wall2_sweeping"] = summary(compile_scheme(g2, budget=Budget(allow_direct_schur=False)), g2)
    out["seconds"] = round(time.perf_counter() - t0, 1)

    folder = os.path.join(ROOT, "out", "workbench", "records", "w348")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"controls-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"written {os.path.relpath(path, ROOT)}")

    # the assertions, after the record
    for k in ("tier0_as_built", "cs_s1_embedded"):
        r = out[k]
        assert r["verdict"] == "refuse", (k, r)
        assert ["L2", "R10", "refuse"] in [x[:3] for x in r["r10"]], (k, r["r10"])
        assert not r["declares_ports"], (k, r["declares_ports"])
        print(f"{k}: {r['graph']}, {r['agents']} agents, refuse at L2/R10 -- held")
    moved = []
    for key, r in rows.items():
        assert not any(x[2] == "refuse" for x in r["r10"]), (key, r)
        if ["L5", "R10", "admit-uncertified"] in r["r10"]:
            moved.append(key)
        print(f"{key}: {r['family']}, {r['verdict']}, R10 {r['r10']}")
    out_d, out_s = out["wall2_direct"], out["wall2_sweeping"]
    assert out_d["accelerator"] == "direct-schur"
    assert ["L5", "R10", "admit-uncertified"] in [x[:3] for x in out_d["r10"]]
    expect = "admit-uncertified" if out_s["eps_tol"] else "refuse"
    assert ["L5", "R10", expect] in [x[:3] for x in out_s["r10"]], out_s
    print(f"wall-2 direct: {out_d['r10']}")
    print(f"wall-2 sweeping ({out_s['accelerator']}, eps_tol={out_s['eps_tol']}): {out_s['r10']}")
    print(f"decertified at L5/R10: {moved}")
    print(f"{out['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
