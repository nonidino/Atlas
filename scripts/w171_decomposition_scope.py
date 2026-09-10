"""W171 -- scoping the per-region decomposition axis.  A census, not a change.

W171 says `CaseGraph.decomposition` is ONE field for the whole graph, that five
rules branch on it and return early on the wrong one, and that rung 9's union
graph -- a tiling plus two circuits -- therefore cannot be built.  It carries an
[AI Inference]:

    "the natural carrier is the AGENT, since each agent knows whether its own
     domain was cut and how -- which would also give `_decomposition_cuts` a
     declaration instead of the same-family proxy R10 currently leans on."

This script checks each clause against the code and the graphs rather than
accepting it, and prices the change.  It writes `out/w171/w171.json`.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import ast
import io
import json
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "out", "w171")


def branch_sites(path):
    """Every function that reads the decomposition axis, and whether it emits.

    The distinction W171's row does not draw: a rule that BRANCHES emits a
    decision on both axes and would say the wrong thing loudly; a rule that
    RETURNS emits nothing and would say nothing at all.  Only the second is
    silent, and the row attributes the second behaviour to all five.
    """
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        seg = ast.get_source_segment(src, fn) or ""
        if "Decomposition." not in seg and ".decomposition" not in seg:
            continue
        reads = ("Decomposition.OVERLAPPING" in seg
                 or "Decomposition.NON_OVERLAPPING" in seg)
        if not reads:
            continue
        # Does an axis branch return having emitted NOTHING?  "Silent" is
        # *emits no decision*, not *ends in a return*: `_cut_policy`'s
        # wrong-axis branch also ends in a bare return and emits `L2/C3` first,
        # so it says the substructuring thing rather than nothing.
        early = False
        for node in ast.walk(fn):
            if not isinstance(node, ast.If):
                continue
            if "Decomposition." not in ast.unparse(node.test):
                continue
            body = "\n".join(ast.unparse(s) for s in node.body)
            emits = any(k in body for k in
                        ("rec.admit", "rec.refuse", "rec.decertify",
                         "record.admit", "record.refuse", "record.decertify"))
            last = node.body[-1]
            if isinstance(last, ast.Return) and last.value is None and not emits:
                early = True
        emits_in_both = ("else:" in seg and ("rec." in seg or "record." in seg))
        out.append({"function": fn.name, "lineno": fn.lineno,
                    "returns_early_on_wrong_axis": early,
                    "emits_on_both_axes": bool(emits_in_both)})
    return out


def graph_axes():
    """Every case graph's declared axis, and whether its seams agree with it."""
    import numpy as np

    from atlas.cases import cooling_loop, front_wing, neural_interface as NI
    from atlas.cases import powertrain, wing_fsi

    u, v = NI.load_state()
    builders = [
        ("front_wing (fixed shape)", lambda: front_wing.build(u, v, motion=False)),
        # `wing_fsi` has its own grid, so it gets its own freestream field --
        # the declaration census needs no developed state.
        ("wing_fsi", lambda: wing_fsi.build(
            np.full((wing_fsi.NY, wing_fsi.NX), wing_fsi.U_INF),
            np.zeros((wing_fsi.NY, wing_fsi.NX)), motion=False)),
        ("cooling_loop", lambda: cooling_loop.build()),
        ("powertrain", lambda: powertrain.build()),
    ]
    rows = []
    for name, mk in builders:
        try:
            g, _ = mk()
        except Exception as exc:                                   # pragma: no cover
            # Named rather than skipped: a census that quietly drops a subject
            # is the failure this package refuses in its own rules.
            rows.append({"graph": name, "error": repr(exc)})
            print("  %-26s NOT CENSUSED -- %s" % (name, exc))
            continue
        fams = {}
        for a in g.agents:
            f = a.capabilities.governing_family or ""
            fams[f] = fams.get(f, 0) + 1
        coincident = [bool(c.geometrically_coincident) for c in g.connections]
        rows.append({
            "graph": name,
            "case": g.name,
            "decomposition": g.decomposition.value,
            "n_agents": len(g.agents),
            "n_seams": len(g.connections),
            "families": fams,
            "n_families": len(fams),
            "cut_families": sum(1 for n in fams.values() if n > 1),
            "sole_agents": sorted(a.agent_id for a in g.agents
                                  if fams[a.capabilities.governing_family or ""] == 1),
            "partition_of_unity": g.partition_of_unity is not None,
            "overlap_cells": g.overlap_cells,
            "geometrically_coincident": {
                "true": sum(coincident), "false": len(coincident) - sum(coincident)},
            "coincident_matches_axis": bool(
                all(not c for c in coincident)
                if g.decomposition.value == "overlapping"
                else all(coincident)),
        })
        print("  %-26s %-16s agents %2d seams %2d families %d (cut %d) "
              "PoU %-5s coincident %d/%d"
              % (name, g.decomposition.value, len(g.agents), len(g.connections),
                 len(fams), rows[-1]["cut_families"],
                 str(rows[-1]["partition_of_unity"]),
                 sum(coincident), len(coincident)))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}

    print("=== clause 1: how many rules branch, and how many are SILENT on the "
          "wrong axis")
    sites = branch_sites(os.path.join(HERE, "atlas", "compiler.py"))
    res["branch_sites"] = sites
    for s in sites:
        print("  %-28s line %-5d  returns-early %-5s  emits-on-both %s"
              % (s["function"], s["lineno"],
                 str(s["returns_early_on_wrong_axis"]), s["emits_on_both_axes"]))
    res["n_sites"] = len(sites)
    res["n_silent"] = sum(1 for s in sites if s["returns_early_on_wrong_axis"])
    print("  %d sites read the axis; %d return early and emit NOTHING"
          % (res["n_sites"], res["n_silent"]))

    print("=== clause 2: is the axis a property of the AGENT, the REGION, or "
          "the SEAM?")
    res["graphs"] = graph_axes()

    # Can two agents of one family be cut differently?  Only if one agent can
    # belong to two regions, and `governing_family` is a single string, so on
    # every graph in the package region == family exactly.
    multi = []
    for r in res["graphs"]:
        if "families" not in r:
            continue
        for fam, n in r["families"].items():
            if n > 1:
                multi.append({"graph": r["graph"], "family": fam, "n_agents": n,
                              "axis": r["decomposition"]})
    res["cut_regions"] = multi
    print("  cut regions across the package:", len(multi))
    for m in multi:
        print("    %-26s %-40s %d agents, %s"
              % (m["graph"], m["family"][:40], m["n_agents"], m["axis"]))

    # Schema reach: what else carries one axis for the whole graph?
    res["schema_reach"] = {
        "CaseGraph.decomposition": "one field, atlas/graph.py:450",
        "Scheme.decomposition": "one field, atlas/scheme.py:101, emitted as "
                                "D_decomposition in as_dict()",
        "_Context.decomposition": "one field, set once by R2's axis lift from "
                                  "`transmission`, which is also graph-global",
        "CaseGraph.partition_of_unity": "one object for the whole graph; the "
                                        "overlapping branch's entire mechanism",
        "CaseGraph.overlap / overlap_cells": "one number for the whole graph",
        "Agent.domain": "free text (str = \"\"), so no structured domain "
                        "declaration exists to hang a per-region axis on",
        "_decomposition_cuts": "derives cut/uncut by grouping on "
                               "governing_family; its own docstring calls this "
                               "a proxy and not a decision procedure",
    }

    res["elapsed_seconds"] = time.perf_counter() - t0
    path = os.path.join(OUT, "w171.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
