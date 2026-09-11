"""W171 -- the per-region decomposition axis, and the control that it moved nothing.

Tier 40 scoped the row and measured its `[AI Inference]` FALSE: the carrier is
not the agent, it is the **connection**.  Tier 41 priced part 1 and did not
build it.  This builds it, and the first thing it does is the control.

**The control comes first, and it is the whole safety argument.**  The axis was
one declared field per graph and is now derived per region.  Every graph written
before W171 has exactly one axis, so *the derivation must reproduce the declared
value on every one of them*.  If it does not, the change is not a schema
extension, it is a behaviour change, and it would move verdicts on graphs nobody
was asking about.  Reproduced on every constructible case graph, that claim is
checkable rather than asserted.

**Then the thing the field could not express.**  A disjoint UNION of two real
graphs -- a fluid tiling and a lumped circuit -- carries two axes at once, which
is rung 9's shape and what `CaseGraph.decomposition` as one field could not say.
Compiled before and after, so the silence W171's row is about is a measurement:
three rules return early on the wrong axis having emitted **nothing**, so on a
union graph they say nothing about the region they skipped.

Writes ``out/w171b/w171b.json``.  numpy only for the fixtures that need it; the
union is built from `cooling_loop` and `powertrain`, which need no field state.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import importlib
import io
import json
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from atlas import Budget, compile_scheme                              # noqa: E402
from atlas.graph import Agent, CaseGraph, Decomposition               # noqa: E402
from atlas.probe import ProbeBudget                                   # noqa: E402

OUT = os.path.join(HERE, "out", "w171b")

#: Case modules whose `build()` needs no argument. The six that need live field
#: state are listed so the count is honest rather than quietly short.
NO_ARG = ["brake_thermal", "cooling_loop", "powertrain", "rocket", "thermal_seam",
          "thermal_strain", "wind_farm"]
NEEDS_STATE = ["channel_ns", "front_wing", "ground_effect", "wind_farm_real",
               "window_ns", "wing_fsi"]


def _unwrap(g):
    """A few builders return (graph, experts); take the graph."""
    return next(x for x in g if hasattr(x, "agents")) if isinstance(g, tuple) else g


def _graph(mod_name):
    mod = importlib.import_module(f"atlas.cases.{mod_name}")
    return _unwrap(mod.build())


# ===========================================================================
# 1 -- the control: the derivation reproduces every declared axis
# ===========================================================================


def control() -> dict:
    """Every constructible case graph: does the derived axis equal the declared one?"""
    rows, disagree = [], []
    for name in NO_ARG:
        try:
            g = _graph(name)
        except Exception as exc:                       # pragma: no cover
            rows.append({"case": name, "skipped": str(exc)[:140]})
            continue
        axes = g.region_axes()
        seams = {c.seam_id: (None if g.seam_axis(c) is None else g.seam_axis(c).value)
                 for c in g.connections}
        row = {
            "case": name,
            "graph": g.name,
            "declared": g.decomposition.value,
            "regions": {k: v.value for k, v in axes.items()},
            "n_regions": len(axes),
            "axis_set": sorted(x.value for x in g.decomposition_axes()),
            "uncut_agents": sorted(
                a.agent_id for a in g.agents if g.agent_axis(a.agent_id) is None),
            "seam_axes": seams,
            "not_a_cut": sorted(s for s, v in seams.items() if v is None),
            "conflicts": g.region_axis_conflicts(),
        }
        row["reproduces_declared"] = all(v.value == g.decomposition.value
                                         for v in axes.values())
        if not row["reproduces_declared"]:
            disagree.append(name)
        rows.append(row)
    return {"rows": rows, "disagree": disagree,
            "n_checked": len([r for r in rows if "skipped" not in r]),
            "needs_field_state_not_checked_here": NEEDS_STATE}


# ===========================================================================
# 2 -- the union: two axes in one graph, which the field could not express
# ===========================================================================


def union_graph(with_pou: bool = False) -> CaseGraph:
    """A disjoint union of `cooling_loop` and a two-window fluid tiling.

    Rung 9's graph is a tiling plus two circuits.  This is the smallest honest
    version of that shape: `cooling_loop`'s four coolant legs are a
    NON_OVERLAPPING region (coincident seams), and a two-agent fluid tiling is
    an OVERLAPPING one (non-coincident).  **The joining seams do not exist** --
    that is W172 and it is not this row -- so the union is disjoint, which is
    enough to put two axes in front of the rules and see what they do.
    """
    from atlas.cases import cooling_loop as CL
    from atlas.cases import wind_farm as WF

    cl, wf = _unwrap(CL.build()), _unwrap(WF.build())

    # Two fluid windows from `wind_farm`, which are a tiling: same family, and
    # their seam is not geometrically coincident.
    fam = "incompressible-navier-stokes-2d"
    fluid_ids = {a.agent_id for a in wf.agents
                 if a.capabilities.governing_family == fam}
    # Pick a pair that is actually JOINED: two agents of one family with no seam
    # between them form a region with nothing to derive an axis from, which is
    # a different case and not the one this union is for.
    pair = next((c.a[0], c.b[0]) for c in wf.connections
                if c.a[0] in fluid_ids and c.b[0] in fluid_ids)
    keep = set(pair)
    fluid_agents = [Agent("F_" + a.agent_id, a.capabilities, domain=a.domain)
                    for a in wf.agents if a.agent_id in keep]
    for a in fluid_agents:
        a.capabilities.expert_id = a.agent_id
    fluid_conns = []
    for c in wf.connections:
        if c.a[0] in keep and c.b[0] in keep:
            import dataclasses
            fluid_conns.append(dataclasses.replace(
                c, seam_id="F_" + c.seam_id,
                a=("F_" + c.a[0], c.a[1]), b=("F_" + c.b[0], c.b[1]),
                # **The declaration W171 adds, and the reason it had to be one.**
                # `wind_farm`'s tiles declare `geometrically_coincident=True`
                # AND an overlapping decomposition, which is why Tier 40's
                # derivation from that flag does not separate the axis. The
                # tiling says what it is.
                cut_axis=Decomposition.OVERLAPPING))
    fluid_conns = fluid_conns[:1]                      # one seam is enough

    agents = list(cl.agents) + fluid_agents
    conns = list(cl.connections) + fluid_conns
    return CaseGraph(
        name="w171-union-circuit-plus-tiling",
        agents=agents,
        connections=conns,
        decomposition=cl.decomposition,                # what the ONE field can say
        overlap=cl.overlap,
        overlap_cells=cl.overlap_cells,
        partition_of_unity=(_tiling_pou(fluid_agents) if with_pou
                            else cl.partition_of_unity),
        cross_points=cl.cross_points,
        macro_dt=cl.macro_dt,
        note="W171: a NON_OVERLAPPING circuit and an OVERLAPPING tiling in one "
             "graph. The joining seams do not exist -- that is W172.",
    )


def _tiling_pou(fluid_agents):
    """A convex two-window partition over the TILING's agents only.

    Deliberately minimal and deliberately wrong in one respect that is the
    point: `CaseGraph.partition_of_unity` is one object for the whole graph, so
    a partition that covers only the tiling still has to be declared as the
    graph's. That is W171 part 2 and this is the fixture that shows why.
    """
    import numpy as np
    from atlas.assembly import PartitionOfUnity

    n = 8
    ids = [a.agent_id for a in fluid_agents]
    w = np.linspace(0.0, 1.0, n)
    return PartitionOfUnity(
        n_global=n,
        restrictions={ids[0]: np.eye(n), ids[1]: np.eye(n)},
        weights={ids[0]: 1.0 - w, ids[1]: w},
        contaminated={ids[0]: (w > 0.5).astype(float),
                      ids[1]: (w <= 0.5).astype(float)},
        ramp_cells=n, profile="linear",
    )


def union_report(g: CaseGraph) -> dict:
    axes = g.region_axes()
    return {
        "name": g.name,
        "n_agents": len(g.agents),
        "n_seams": len(g.connections),
        "declared_single_field": g.decomposition.value,
        "regions": {k: v.value for k, v in axes.items()},
        "axis_set": sorted(x.value for x in g.decomposition_axes()),
        "carries_two_axes": len(g.decomposition_axes()) > 1,
        "uncut_agents": sorted(a.agent_id for a in g.agents
                               if g.agent_axis(a.agent_id) is None),
        "seam_axes": {c.seam_id: (None if g.seam_axis(c) is None
                                  else g.seam_axis(c).value)
                      for c in g.connections},
    }


#: The three rules W171's row is about: each returns early on the wrong axis
#: having emitted NOTHING, so on a union graph it says nothing at all about the
#: region it skipped. Tier 40 named them and this is where they are watched.
SILENT_RULES = ("R10/halo", "R12", "W49")


def rule_coverage(r, g: CaseGraph) -> dict:
    """For each silent rule: did it speak, and about which region?

    The failure W171 names is not a wrong answer, it is an ABSENT one. So the
    quantity is which regions each rule emitted a decision for -- measurable
    before the change and after it, with the same function.
    """
    emitted = {}
    for d in r.decisions.decisions:
        for name in SILENT_RULES:
            if str(d.rule).endswith(name) or str(d.rule) == name:
                emitted.setdefault(name, []).append(
                    {"verdict": d.verdict.value, "subject": d.subject,
                     "message": d.message[:400]})
    axes = g.region_axes()
    return {
        "regions": {k: v.value for k, v in axes.items()},
        "rules": {name: emitted.get(name, []) for name in SILENT_RULES},
        "silent_rules": sorted(n for n in SILENT_RULES if not emitted.get(n)),
        "overlapping_regions_present": sorted(
            k for k, v in axes.items() if v is Decomposition.OVERLAPPING),
    }


def compile_union(g: CaseGraph) -> dict:
    r = compile_scheme(g, Budget(), ProbeBudget())
    return {
        "coverage": rule_coverage(r, g),
        "verdict": r.verdict.value,
        "runnable": bool(r.runnable),
        "n_decisions": len(r.decisions.decisions),
        "rules": sorted({f"{d.layer}/{d.rule}" for d in r.decisions.decisions}),
        "refusals": sorted({f"{d.layer}/{d.rule}"
                            for d in r.decisions.decisions
                            if d.verdict.value == "refuse"}),
        "messages": {f"{d.layer}/{d.rule}/{d.subject}": d.message
                     for d in r.decisions.decisions},
        "D_decomposition": (r.scheme.as_dict().get("D_decomposition")
                            if r.scheme else None),
        "D_decomposition_axes": (r.scheme.as_dict().get("D_decomposition_axes")
                                 if r.scheme else None),
    }


# ===========================================================================


def main() -> None:
    t0 = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    res = {"what": "W171 -- the decomposition axis, per region rather than per graph",
           "date": "2026-09-10"}

    res["control"] = control()
    g = union_graph()
    res["union"] = union_report(g)
    res["union_compile"] = compile_union(g)

    # **Where part 1 stops, measured rather than claimed.** Tier 41 said
    # `partition_of_unity`, `overlap` and `overlap_cells` stay graph-global and
    # are part 2. `_r12_conservative_assembly` and `_w49_sigma_branch` are both
    # gated on a partition of unity BEFORE they reach the axis, so on a union
    # with no PoU they stay silent whatever this change does. Give the union one
    # -- a two-window partition over the tiling only -- and see whether the
    # part-1 change is then enough for them.
    gp = union_graph(with_pou=True)
    res["union_with_pou"] = union_report(gp)
    res["union_with_pou_compile"] = compile_union(gp)
    res["elapsed_seconds"] = time.perf_counter() - t0

    path = os.path.join(OUT, "w171b.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    c = res["control"]
    print("=== 1. the control: derived axis vs declared ===")
    for r in c["rows"]:
        if "skipped" in r:
            print("  %-16s SKIPPED %s" % (r["case"], r["skipped"][:60]))
            continue
        print("  %-16s declared %-16s regions %-46s reproduces=%s"
              % (r["case"], r["declared"], str(r["regions"])[:46],
                 r["reproduces_declared"]))
    print("  checked %d graphs, disagreements: %s"
          % (c["n_checked"], c["disagree"] or "none"))
    print("=== 2. the union ===")
    u = res["union"]
    print("  %s: %d agents, %d seams" % (u["name"], u["n_agents"], u["n_seams"]))
    print("  the ONE field says: %s" % u["declared_single_field"])
    print("  the regions say:    %s" % u["regions"])
    print("  carries two axes:   %s" % u["carries_two_axes"])
    uc = res["union_compile"]
    print("  compiles to %s, %d decisions, refusals %s"
          % (uc["verdict"], uc["n_decisions"], uc["refusals"] or "none"))
    print("wrote", path, "in %.1f s" % res["elapsed_seconds"])


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main()
