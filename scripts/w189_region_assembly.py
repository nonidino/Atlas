"""W189 -- the overlapping mechanism, declared per region, on a REAL tiling.

Tier 44 made the decomposition axis per region and measured where it stopped:
`L6/R12` and `L6/W49` are gated on `graph.partition_of_unity` before they reach
any axis, and that object was one per graph -- so Tier 44's union had to declare
a two-window fixture's partition as the WHOLE graph's.  This builds the union
that cannot dodge it and measures what the per-region form changes.

Five measurements, each with the control that makes it evidence:

  1. **The union.**  `window_ns`'s four `reference.WindowNS` windows -- a real
     tiling, its own grid partition of unity, its own halo -- beside BOTH
     circuits, `cooling_loop` and `powertrain`.  The partition, the overlap and
     the overlap-cell count are filed under the fluid region and nowhere else.
     Which region every assembly and halo decision names, per rule.
  2. **The same union declared the old way**, with the tiling's partition as the
     graph's.  The rules still speak; what they cannot do is say WHICH region.
  3. **The equivalence control.**  `window_ns` alone, declared both ways.  The
     per-region form must reach the same verdict under the same rules -- it is
     the same rule scoped, not a different rule -- or the union's differences
     could be the form rather than the union.
  4. **The two-region positive control.**  A single-region union cannot tell
     "per region" from "the one overlapping region".  `rocket` has TWO overlapping
     regions and no partition; given one per region, every assembly rule must
     speak twice, once naming each.
  5. **The declaration check, fired and not fired.**  Four defective union
     declarations -- a key naming a sole agent's family, a key naming the
     coolant circuit's NON-overlapping region, a partition that blends a circuit
     agent, a partition whose subdomains name no agent -- each beside the clean
     union where the check stays quiet.

And two things the union costs that are not W189's, recorded because W172 will
need them: every graph-global field the three graphs had to agree on, and the
agents whose region membership CHANGED by being put in a union at all.

Writes ``out/w189/w189.json``.  numpy only; the Poseidon-backed graphs are not
used here.  Run ``scripts/w189_artifact_control.py`` for the byte-identity
control, which is separate on purpose: it must be captured before the change.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import dataclasses
import io
import json
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402

from atlas import compile_scheme                                      # noqa: E402
from atlas.assembly import GridPartitionOfUnity                       # noqa: E402
from atlas.graph import (CaseGraph, Decomposition, MeasuredConstants)  # noqa: E402

OUT = os.path.join(HERE, "out", "w189")

FLUID = "incompressible-navier-stokes-2d"
COOLANT = "incompressible-thermal-transport-1d"
CIRCUIT = "lumped-dc-circuit"
OVER, NON = Decomposition.OVERLAPPING, Decomposition.NON_OVERLAPPING

#: The rules whose subject is an assembly or a region -- the ones W189 scopes.
L6_RULES = ("E6", "R11", "L6/C1", "R12", "W49")
L2_RULES = ("R10/halo", "C2", "C3")


# ===========================================================================
# the graphs
# ===========================================================================


def _s0():
    d = np.load(os.path.join(HERE, "out", "tier0b", "s0_state.npz"))
    return d["u"], d["v"]


def tiling_graph():
    """`window_ns` in split-step mode at the Tier 0 state: the real tiling."""
    from atlas.cases import window_ns as WNS

    return WNS.build(*_s0())[0]


def _circuit(name):
    from atlas.cases import cooling_loop, powertrain

    return {"cooling_loop": cooling_loop, "powertrain": powertrain}[name].build()[0]


def _axis(conns, axis):
    return [dataclasses.replace(c, cut_axis=axis) for c in conns]


def union_graph(per_region=True, circuits=("cooling_loop", "powertrain"),
                graph_axis=OVER, partition=None, keys=None, name=None):
    """A real tiling beside both circuits, with no seam joining them (that is W172).

    ``partition`` replaces the tiling's own partition of unity (for the
    declaration-check controls) and ``keys`` replaces the region key each
    per-region field is filed under.  Everything the union had to reconcile
    because a field is one per graph is listed by `reconciliation`.
    """
    t = tiling_graph()
    parts = [_circuit(c) for c in circuits]
    tiling_ids = tuple(a.agent_id for a in t.agents)
    agents = list(t.agents) + [a for g in parts for a in g.agents]
    conns = _axis(t.connections, OVER) + [c for g in parts for c in _axis(g.connections, NON)]
    # W117: `window_ns` declares its pressure field with applies_to=() -- "every
    # agent" -- which was true of a graph holding only its four windows. In a
    # union it would carry the tiling's state into both circuits with no seam,
    # which `L3/global-field` refuses. Scoped to the tiling, which is what
    # `front_wing` and `wake_array` already declare for the same field.
    gfs = [dataclasses.replace(gf, applies_to=tiling_ids) for gf in t.global_fields]
    pou = partition if partition is not None else t.partition_of_unity
    if per_region:
        k = keys or {}
        fields = dict(
            partition_of_unity={k.get("partition_of_unity", FLUID): pou},
            overlap_cells={k.get("overlap_cells", FLUID): t.overlap_cells},
            overlap={k.get("overlap", FLUID): t.overlap},
        )
    else:
        fields = dict(partition_of_unity=pou, overlap_cells=t.overlap_cells,
                      overlap=t.overlap)
    return CaseGraph(
        name=name or ("w189-union-" + ("per-region" if per_region else "graph-scoped")
                      + "-" + "+".join(("window_ns",) + tuple(circuits))),
        agents=agents,
        connections=conns,
        decomposition=graph_axis,
        global_fields=gfs,
        cross_points=tuple(t.cross_points or ()) + tuple(
            cp for g in parts for cp in (g.cross_points or ())),
        loop_gains=tuple(lg for g in parts for lg in g.loop_gains),
        macro_dt=t.macro_dt,
        measured=None,
        note="W189: a real tiling beside both circuits. The joining seams do not "
             "exist -- that is W172.",
        **fields,
    )


def _show(v):
    if isinstance(v, dict):
        return {k: _show(x) for k, x in v.items()}
    if hasattr(v, "subdomains"):
        return f"{type(v).__name__} over {sorted(v.subdomains())}"
    if isinstance(v, Decomposition):
        return v.value
    if isinstance(v, (list, tuple)):
        return [getattr(x, "name", None) or getattr(x, "agents", None) or str(x)
                for x in v]
    if isinstance(v, MeasuredConstants):
        return f"MeasuredConstants(source={v.source!r})"
    return v if isinstance(v, (int, float, str, type(None))) else str(v)


def reconciliation(circuits=("cooling_loop", "powertrain")) -> list[dict]:
    """Every graph-global field, what each part declares, and what the union says.

    Not a rule and not a claim -- a list of what had to be DECIDED to put three
    graphs in one, because each of these is still one value per graph.
    ``parts_disagree`` is True where two parts declare different non-empty
    values, which is where the union had to choose one or scope it.
    """
    t = tiling_graph()
    parts = {"window_ns": t, **{c: _circuit(c) for c in circuits}}
    u = union_graph(circuits=circuits)
    rows = []
    for fname in ("decomposition", "partition_of_unity", "overlap", "overlap_cells",
                  "global_fields", "cross_points", "primal_cross_point_dofs",
                  "loop_gains", "macro_dt", "flux_matching", "measured",
                  "topology_events"):
        shown = {p: _show(getattr(g, fname)) for p, g in parts.items()}
        distinct = {json.dumps(v, sort_keys=True, default=str) for v in shown.values()
                    if v not in (None, [], {}, "")}
        rows.append({
            "field": fname,
            "parts": shown,
            "union": _show(getattr(u, fname)),
            "parts_disagree": len(distinct) > 1,
            "per_region_after_w189": fname in CaseGraph.PER_REGION_FIELDS,
        })
    return rows


def composition_effects(circuits=("cooling_loop", "powertrain")) -> dict:
    """Agents whose cut status or region CHANGED by being put in the union.

    A region is a governing_family with more than one agent (W114's proxy,
    grouped at W171), so it is a property of the GRAPH and not of the agent --
    and putting two graphs together can change it for an agent neither graph
    changed.  Measured rather than argued.
    """
    from atlas.compiler import _decomposition_cuts

    t = tiling_graph()
    parts = {"window_ns": t, **{c: _circuit(c) for c in circuits}}
    u = union_graph(circuits=circuits)
    cut_u, _ = _decomposition_cuts(u)
    region_u = {a: fam for fam, ids in u.regions().items() for a in ids}
    changed = []
    for pname, g in parts.items():
        cut_p, _ = _decomposition_cuts(g)
        region_p = {a: fam for fam, ids in g.regions().items() for a in ids}
        for a in g.agents:
            before = ("cut" if a.agent_id in cut_p else "sole", region_p.get(a.agent_id))
            after = ("cut" if a.agent_id in cut_u else "sole", region_u.get(a.agent_id))
            if before != after:
                changed.append({
                    "agent": a.agent_id, "from_graph": pname,
                    "family": a.capabilities.governing_family,
                    "in_its_own_graph": {"status": before[0], "region": before[1]},
                    "in_the_union": {"status": after[0], "region": after[1],
                                     "region_agents": sorted(u.regions().get(after[1], []))},
                    "stencil_radius": a.capabilities.stencil_radius,
                    "elliptic_subsolve": getattr(a.capabilities.elliptic_subsolve,
                                                 "value", None),
                })
    return {
        "regions_in_parts": {p: {k: sorted(v) for k, v in g.regions().items()}
                             for p, g in parts.items()},
        "regions_in_union": {k: sorted(v) for k, v in u.regions().items()},
        "changed": changed,
        "dt_native": {p: sorted({a.capabilities.dt_native for a in g.agents
                                 if a.capabilities.dt_native is not None})
                      for p, g in parts.items()},
    }


def window_ns_per_region():
    """`window_ns` with its three fields re-declared under its one region."""
    g = tiling_graph()
    g.partition_of_unity = {FLUID: g.partition_of_unity}
    g.overlap_cells = {FLUID: g.overlap_cells}
    g.overlap = {FLUID: g.overlap}
    g.name = g.name + "-per-region"
    return g


def _strip_pou(ids, inner=4, shared=2):
    """A convex one-dimensional tiling over ``ids``: each window owns ``inner``
    cells outright and shares ``shared`` with its right neighbour at weight 1/2.

    Every window has an interior, so ||A|| = 1 exactly as `norm_A`'s docstring
    says it is for any decomposition anybody builds -- the measurement here is
    WHICH region each rule names, and a fixture with no interior would put a
    number beside L6/C1's sentence that the sentence does not describe.
    """
    idx, wts, bad = {}, {}, {}
    start = 0
    n = len(ids)
    for k, aid in enumerate(ids):
        left = shared if k > 0 else 0
        right = shared if k < n - 1 else 0
        cells = np.arange(start - left, start + inner + right)
        w = np.ones(cells.size)
        w[:left] = 0.5
        w[cells.size - right:] = 0.5
        c = np.zeros(cells.size, dtype=bool)
        c[:left] = True
        c[cells.size - right:] = True
        idx[aid], wts[aid], bad[aid] = cells, w, c
        start += inner + right
    total = start
    return GridPartitionOfUnity(n_global=total, indices=idx, weights=wts,
                                contaminated=bad, ramp_cells=shared,
                                profile="step, 1/2 on the shared cells")


def rocket_two_regions(declare=("reacting-compressible-flow",
                                "external-compressible-flow"), c_mu=None):
    """`rocket`, whose two overlapping regions carry one partition each.

    The partitions are a synthetic strip tiling over each region's own agents,
    because the measurement is WHICH region each rule names, not what it
    concludes.  ``c_mu`` declares a graph-level C_mu (and a cut-defect bound):
    the constants a two-region graph cannot attribute to either region (W190).
    """
    from atlas.cases import rocket

    g = rocket.build()
    regions = g.regions()
    g.partition_of_unity = {fam: _strip_pou(regions[fam]) for fam in declare}
    g.overlap_cells = {fam: 4 for fam in declare}
    if c_mu is not None:
        g.measured = MeasuredConstants(
            C_mu=c_mu, cut_defect_bound=1.0e-3, cut_defect_bound_form="chi-weighted",
            probe_state="synthetic", scheme="synthetic", source="w189 two-region control")
    g.name = "w189-rocket-" + str(len(declare)) + "-region" + ("s" if len(declare) > 1 else "")
    return g


def _grid_copy(p, rename=None, extra=None):
    """A copy of a grid partition with subdomains renamed or one appended."""
    rename = rename or {}
    idx = {rename.get(k, k): np.asarray(v) for k, v in p.indices.items()}
    wts = {rename.get(k, k): np.asarray(v) for k, v in p.weights.items()}
    bad = ({rename.get(k, k): np.asarray(v) for k, v in p.contaminated.items()}
           if p.contaminated is not None else None)
    if extra:
        # a subdomain holding one cell at zero weight: the identity and convexity
        # are untouched, so ONLY the index set is wrong -- which is the point
        idx[extra] = np.array([0])
        wts[extra] = np.array([0.0])
        if bad is not None:
            bad[extra] = np.array([False])
    return GridPartitionOfUnity(n_global=p.n_global, indices=idx, weights=wts,
                                kind=p.kind, contaminated=bad,
                                ramp_cells=p.ramp_cells, profile=p.profile)


def defect_variants():
    """The four declaration defects, each on the otherwise-clean union."""
    t = tiling_graph()
    p = t.partition_of_unity
    return {
        "key-names-a-sole-agents-family": union_graph(
            keys={"partition_of_unity": "heat-conduction-2d"},
            name="w189-defect-sole-family-key"),
        "key-names-the-coolant-NON-overlapping-region": union_graph(
            keys={"overlap_cells": COOLANT}, name="w189-defect-non-overlapping-key"),
        "partition-blends-a-circuit-agent": union_graph(
            partition=_grid_copy(p, extra="PASS"), name="w189-defect-foreign-agent"),
        "partition-names-no-agent": union_graph(
            partition=_grid_copy(p, rename={k: "X" + k[1:] for k in p.indices}),
            name="w189-defect-unmatched-keys"),
    }


# ===========================================================================
# the measurements
# ===========================================================================


def scoped(d) -> bool:
    """A decision from one of the rules W189 scopes, including their W-suffixed forms."""
    def hit(rule, names):
        return any(rule == n or rule.startswith(n + "/") for n in names)

    return ((d.layer == "L6" and hit(d.rule, L6_RULES))
            or (d.layer == "L2" and (hit(d.rule, L2_RULES) or d.rule.startswith("W189"))))


def regions_named(d, g) -> list[str]:
    """Which regions a decision is ABOUT: its subject and its `region` evidence.

    A message that names an agent in order to EXCLUDE it -- the halo rule's W136
    clause -- is disclosure, not a statement about that agent, so the message
    text is deliberately not read.
    """
    agent_region = {a: fam for fam, ids in g.regions().items() for a in ids}
    named = set()
    ev = d.evidence.get("region")
    if ev:
        named.add(ev)
    # The declaration check's clean admission is about the graph's per-region
    # DECLARATIONS as a whole, so its subject is the graph -- and the regions it
    # speaks about are the ones it lists as declared.
    for keys in (d.evidence.get("declared") or {}).values():
        named.update(keys)
    subj = d.subject or ""
    for pre in ("<assembly:", "<region:"):
        if subj.startswith(pre) and subj.endswith(">"):
            named.add(subj[len(pre):-1])
    for tok in subj.split(", "):
        if tok in agent_region:
            named.add(agent_region[tok])
    return sorted(named)


def coverage(r, g) -> dict:
    rows = []
    for d in r.decisions.decisions:
        if scoped(d):
            rows.append({"rule": f"{d.layer}/{d.rule}", "verdict": d.verdict.value,
                         "subject": d.subject, "regions_named": regions_named(d, g),
                         "message": d.message})
    by_rule = collections.defaultdict(set)
    for row in rows:
        by_rule[row["rule"]].update(row["regions_named"] or ["<unnamed>"])
    return {"rows": rows, "regions_by_rule": {k: sorted(v) for k, v in by_rule.items()}}


def compile_report(g) -> dict:
    t0 = time.perf_counter()
    r = compile_scheme(g)
    elapsed = time.perf_counter() - t0
    art = json.loads(r.artifact.to_json())
    rules = collections.Counter((d.layer, d.rule, d.verdict.value)
                                for d in r.decisions.decisions)
    return {
        "graph": g.name,
        "n_agents": len(g.agents),
        "n_seams": len(g.connections),
        "regions": {k: sorted(v) for k, v in g.regions().items()},
        "region_axes": {k: v.value for k, v in g.region_axes().items()},
        "verdict": r.verdict.value,
        "n_decisions": len(r.decisions.decisions),
        "refusals": sorted({f"{d.layer}/{d.rule}" for d in r.decisions.refusals}),
        "refusal_messages": {f"{d.layer}/{d.rule}/{d.subject}": d.message
                             for d in r.decisions.refusals},
        "rule_verdicts": sorted([list(k) + [n] for k, n in rules.items()]),
        "coverage": coverage(r, g),
        "scheme": {k: art["scheme"].get(k) for k in
                   ("D_decomposition", "D_decomposition_axes", "overlap",
                    "overlap_by_region")} if art.get("scheme") else None,
        "assembly_kind": art["assembly"].get("kind"),
        "assembly_regions": sorted((art["assembly"].get("regions") or {})),
        "assembly_region_certificates": {
            k: {x: v.get(x) for x in ("pou_residual", "identity_holds", "norm_A",
                                      "condition_holds")}
            for k, v in (art["assembly"].get("regions") or {}).items()},
        "harness": art["bound_terms"]["harness"],
        "envelope_E6": art["envelope"]["E6"],
        "unmeasured": list(r.unmeasured),
        "w189_decisions": [{"verdict": d.verdict.value, "subject": d.subject,
                            "message": d.message, "evidence": d.evidence}
                           for d in r.decisions.decisions
                           if d.layer == "L2" and d.rule.startswith("W189")],
        "compile_s": elapsed,
    }


def speaks_only_about(rep, allowed) -> dict:
    """Does every scoped decision name only regions in ``allowed``, and name one?

    The substructuring branch, L2/C3, is the circuits' and is excluded here; it is
    checked separately to name only the non-overlapping regions.
    """
    stray, unnamed = [], []
    for row in rep["coverage"]["rows"]:
        if row["rule"].startswith("L2/C3"):
            continue
        names = row["regions_named"]
        if not names:
            unnamed.append(row["rule"] + " " + str(row["subject"]))
        elif not set(names) <= set(allowed):
            stray.append(row["rule"] + " " + str(names))
    return {"stray": stray, "unnamed": unnamed, "ok": not stray and not unnamed}


def rule_multiset(rep, drop_prefix=("L2/W189",)) -> list:
    return sorted(x for x in rep["rule_verdicts"]
                  if not any((x[0] + "/" + x[1]).startswith(p) for p in drop_prefix))


def main() -> None:
    t0 = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    res = {"what": "W189 -- partition of unity, overlap and overlap cells, per region",
           "date": time.strftime("%Y-%m-%d")}

    # 1 + 2 -- the union, both ways
    res["union"] = compile_report(union_graph())
    res["union"]["speaks_only_about_the_tiling"] = speaks_only_about(res["union"], [FLUID])
    res["union"]["substructuring_branch_regions"] = sorted({
        n for row in res["union"]["coverage"]["rows"] if row["rule"].startswith("L2/C3")
        for n in row["regions_named"]})
    res["union_graph_scoped"] = compile_report(union_graph(per_region=False))
    res["union_one_circuit"] = compile_report(union_graph(circuits=("cooling_loop",)))
    res["union_one_circuit"]["speaks_only_about_the_tiling"] = speaks_only_about(
        res["union_one_circuit"], [FLUID])
    res["reconciliation"] = reconciliation()
    res["composition_effects"] = composition_effects()

    # the cross-point finding: the same union with the graph-level axis flipped
    res["union_graph_axis_non_overlapping"] = compile_report(
        union_graph(graph_axis=NON, name="w189-union-graph-axis-non-overlapping"))

    # 3 -- equivalence control
    legacy = compile_report(tiling_graph())
    per = compile_report(window_ns_per_region())
    res["equivalence"] = {
        "graph_scoped": legacy, "per_region": per,
        "same_verdict": legacy["verdict"] == per["verdict"],
        "same_rules_and_verdicts": rule_multiset(legacy) == rule_multiset(per),
        "only_in_graph_scoped": [x for x in rule_multiset(legacy) if x not in rule_multiset(per)],
        "only_in_per_region": [x for x in rule_multiset(per) if x not in rule_multiset(legacy)],
    }

    # 4 -- two regions
    two = compile_report(rocket_two_regions())
    one = compile_report(rocket_two_regions(declare=("reacting-compressible-flow",)))
    two_cmu = compile_report(rocket_two_regions(c_mu=1.2))
    res["two_regions"] = {"both_declared": two, "one_declared": one,
                          "both_declared_with_graph_constants": two_cmu}

    # 5 -- declaration defects, beside the clean union
    res["defects"] = {k: compile_report(g) for k, g in defect_variants().items()}
    res["defects_clean_union_w189"] = res["union"]["w189_decisions"]

    res["elapsed_seconds"] = time.perf_counter() - t0
    path = os.path.join(OUT, "w189.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)

    u = res["union"]
    print("=== 1. the union: %d agents, %d seams, verdict %s, refusals %s"
          % (u["n_agents"], u["n_seams"], u["verdict"], u["refusals"]))
    for rule, regs in sorted(u["coverage"]["regions_by_rule"].items()):
        print("  %-18s %s" % (rule, regs))
    print("  speaks only about the tiling:", u["speaks_only_about_the_tiling"])
    print("  agents whose region changed in the union:",
          [(c["agent"], c["in_its_own_graph"], c["in_the_union"]["region"])
           for c in res["composition_effects"]["changed"]])
    e = res["equivalence"]
    print("=== 3. equivalence: same verdict %s, same rules %s"
          % (e["same_verdict"], e["same_rules_and_verdicts"]))
    print("=== 4. two regions:", two["coverage"]["regions_by_rule"])
    print("=== 5. defects:")
    for k, rep in res["defects"].items():
        print("  %-48s %s" % (k, [(d["verdict"], d["subject"]) for d in rep["w189_decisions"]]))
    print("wrote", path, "in %.1f s" % res["elapsed_seconds"])


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main()
