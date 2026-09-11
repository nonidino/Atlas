"""Tier 45 -- W189, W171 part 2: the overlapping mechanism, declared per region.

`CaseGraph.partition_of_unity`, `overlap` and `overlap_cells` were one object each
for the whole graph, and a vehicle is a fluid tiling and two circuits at once.
Tier 44 made the decomposition AXIS per region and measured where that stopped:
`L6/R12` and `L6/W49` are gated on the partition before they reach any axis, so a
partition covering only the tiling still had to be declared as the whole graph's.

What this file pins, in the order the change has to be trusted in.

**First the control, because it is the whole safety argument.**  Every existing
graph must compile to a BYTE-IDENTICAL artifact after the change.  Not the same
verdict and not the same rule set -- the same bytes of `RunArtifact.to_json()`,
because anything weaker would pass a change that rewrote every message.  The
pre-change artifacts were captured from the unchanged tree over every
constructible case graph, twice, under two different ``PYTHONHASHSEED`` values,
and the two captures agreeing is the repeat floor that makes the comparison mean
anything.

Numbers are asserted against ``out/w189/`` where the artifact carries them; run
``python scripts/w189_artifact_control.py capture <label>`` and ``compare`` to
rebuild them.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

OUT = os.path.join(_ROOT, "out", "w189")
BEFORE = os.path.join(OUT, "control_before1.json")
REPEAT = os.path.join(OUT, "compare_before1_vs_before2.json")
AFTER = os.path.join(OUT, "compare_before1_vs_after.json")

#: Recompiled LIVE on every run, against the pre-change manifest.  The full
#: forty are compared by the script and asserted from its record below; these
#: are the ones cheap enough to recompile inside the suite, chosen to cross every
#: code path W189 touches: no assembly at all (`rocket`, `wind_farm`), a
#: co-located pair (`thermal_strain`), a multirate pair (`brake_thermal`), both
#: circuits and the triangle control, Tier 44's two union fixtures -- one with a
#: graph-scoped partition covering only the tiling, which is exactly the
#: declaration W189 exists to replace -- a bare grid partition with the halo rule
#: and L2/C2 live (`window_ns`), and a projected assembly that reaches `admit`
#: (`front_wing`).
LIVE = ("rocket", "wind_farm", "thermal_strain", "brake_thermal", "cooling_loop",
        "cooling_loop-3leg", "powertrain", "tier44-union", "tier44-union-with-pou",
        "window_ns", "front_wing")

#: The field state a live variant needs, when it needs one. Local-only caches
#: (gitignored), so a clone without them skips rather than fails.
NEEDS = {
    "window_ns": ("out", "tier0b", "s0_state.npz"),
    "front_wing": ("out", "w141", "settled.npz"),
}


def _load(path, what):
    if not os.path.exists(path):
        pytest.skip(f"{what} not on disk: run scripts/w189_artifact_control.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def before():
    return _load(BEFORE, "the pre-change manifest")


# --------------------------------------------------------------------------
# 1. the control: every existing graph compiles to the same bytes
# --------------------------------------------------------------------------


def test_W189_control_the_capture_is_the_whole_package(before):
    """Twenty case modules: eighteen build graphs and two do not.

    `seam_placement` is a search over decompositions and `wind_farm_design` an
    optimiser over rollouts; neither constructs a `CaseGraph`.  The census names
    them rather than quietly counting to eighteen.
    """
    cases = sorted(f[:-3] for f in os.listdir(os.path.join(_ROOT, "atlas", "cases"))
                   if f.endswith(".py") and f != "__init__.py")
    assert len(cases) == 20, cases
    built = set(before["modules_constructed"])
    named = set(before["no_graph_modules"])
    assert built.isdisjoint(named)
    assert (built | named) >= set(cases) - {"__init__"}, sorted(set(cases) - built - named)
    rows = before["rows"]
    assert len(rows) == 40
    assert all("sha256" in r for r in rows.values()), [
        k for k, r in rows.items() if "sha256" not in r]


def test_W189_control_the_repeat_floor_is_exactly_zero():
    """Two captures of the unchanged tree, two hash seeds, forty identical artifacts.

    `verdict._jsonable` turns a SET into a list in iteration order, and string
    hashing is randomised per process, so an artifact could in principle differ
    between two runs of the same code.  Measured, none does -- which is what lets
    a before/after difference be read as a change rather than as noise.
    """
    rep = _load(REPEAT, "the repeat-floor comparison")
    assert rep["differ"] == [], rep["differ"]
    assert rep["not_compared"] == [], rep["not_compared"]
    assert len(rep["identical"]) == 40


def test_W189_control_every_existing_graph_compiles_byte_identically_after():
    """**The first assertion of the tier.** Forty artifacts, the same bytes.

    Tier 42 clause 2 and Tier 44 section 3.1: a schema extension moves nothing on
    a graph nobody asked about, and a change that did would be a behaviour change
    wearing a schema change's name.
    """
    cmp_ = _load(AFTER, "the after-change comparison")
    assert cmp_["differ"] == [], cmp_["differ"]
    assert cmp_["not_compared"] == [], cmp_["not_compared"]
    assert len(cmp_["identical"]) == 40


@pytest.mark.parametrize("key", LIVE)
def test_W189_control_live_recompile_matches_the_pre_change_bytes(before, key):
    """The same comparison, recompiled now rather than read from the record."""
    need = NEEDS.get(key)
    if need and not os.path.exists(os.path.join(_ROOT, *need)):
        pytest.skip(f"{os.path.join(*need)} is a local cache and is not on disk")
    import w189_artifact_control as CTRL

    _g, _r, text, _t = CTRL.compile_artifact(key)
    assert CTRL.digest(text) == before["rows"][key]["sha256"], key


# --------------------------------------------------------------------------
# 2. the schema: per region, and the single object still the graph's
# --------------------------------------------------------------------------

FLUID = "incompressible-navier-stokes-2d"
COOLANT = "incompressible-thermal-transport-1d"


@pytest.fixture()
def t44_union():
    """Tier 44's union with its tiling's partition, rebuilt fresh per test.

    Compiles in hundredths of a second, which is why the declaration-check
    controls run on it; the real tiling's union is measured by the script and
    asserted from its record in section 4.
    """
    import w171_region_axis as W

    return W.union_graph(with_pou=True)


def _per_region(g, partition=None, key=FLUID):
    g.partition_of_unity = {key: partition if partition is not None
                            else g.partition_of_unity}
    return g


def test_W189_each_field_takes_a_region_map_and_one_object_still_means_the_graph(t44_union):
    g = t44_union
    pou = g.partition_of_unity
    for name in ("partition_of_unity", "overlap", "overlap_cells"):
        assert not g.per_region(name), name
    assert not g.declares_per_region()
    # the graph-scoped object is NOT read as any region's -- which is the
    # declaration this fixture was forced into and W189 exists to replace
    assert g.partition_for(FLUID) is None
    g.overlap_cells = {FLUID: 7}
    assert g.per_region("overlap_cells") and g.declares_per_region()
    assert g.overlap_cells_for(FLUID) == 7 and g.overlap_cells_for(COOLANT) is None
    _per_region(g, pou)
    assert g.partition_for(FLUID) is pou


def test_W189_the_graph_level_projection_question_raises_on_a_per_region_graph(t44_union):
    """A None would read as "no projection declared" on a graph whose regions
    may each declare one, so the one-answer question refuses to answer."""
    from atlas.graph import GraphError

    g = _per_region(t44_union)
    with pytest.raises(GraphError, match="assembly_projection_for"):
        _ = g.assembly_projection
    assert g.assembly_projection_for(FLUID) is None      # a bare partition here


def test_W189_rescope_reissues_a_rule_s_own_decisions_about_a_region():
    """One mechanism for every per-region rule: same sentence, region named."""
    from atlas.verdict import DecisionRecord

    rec = DecisionRecord()
    rec.admit("L2", "before", "untouched", subject="<graph>")
    start = len(rec)
    rec.admit("L6", "E6", "graph-scoped", subject="<assembly>", x=1)
    rec.decertify("L2", "R10/halo", "about the graph", subject="<graph>")
    rec.refuse("L2", "R10/halo", "about agents", subject="A, B")
    rec.rescope(start, FLUID)
    d = rec.decisions
    assert (d[0].subject, d[0].message, d[0].evidence) == ("<graph>", "untouched", {})
    assert d[1].subject == f"<assembly:{FLUID}>"
    assert d[2].subject == f"<region:{FLUID}>"
    assert d[3].subject == "A, B"                       # already inside the region
    for x in d[1:]:
        assert x.message.startswith(f"region {FLUID!r}: ")
        assert x.evidence["region"] == FLUID
    assert d[1].evidence["x"] == 1
    assert [x.verdict.value for x in d] == ["admit", "admit", "admit-uncertified",
                                            "refuse"]


def test_W189_E6_is_stamped_by_severity_not_by_the_order_regions_are_visited():
    """The control first: the stamp itself IS order-dependent for this pair."""
    from atlas.compiler import _E6Tally
    from atlas.envelope import EnvelopeStamp, Hypothesis, Status

    a, b = EnvelopeStamp(), EnvelopeStamp()
    a.holds(Hypothesis.E6, "x")
    a.unchecked(Hypothesis.E6, "y")
    b.unchecked(Hypothesis.E6, "y")
    b.holds(Hypothesis.E6, "x")
    assert a[Hypothesis.E6] is not b[Hypothesis.E6], "the control lost its teeth"

    def tallied(rows):
        t = _E6Tally()
        for region, verb in rows:
            getattr(t.sink(region), verb)(Hypothesis.E6, verb)
        s = EnvelopeStamp()
        t.stamp(s)
        return s[Hypothesis.E6]

    assert tallied([("R1", "holds"), ("R2", "unchecked")]) is Status.UNCHECKED
    assert tallied([("R2", "unchecked"), ("R1", "holds")]) is Status.UNCHECKED
    assert tallied([("R1", "holds"), ("R2", "fails"), ("R3", "unchecked")]) is Status.FAILS
    assert tallied([("R1", "holds"), ("R2", "holds")]) is Status.HOLDS


def test_W189_the_harness_names_each_region_and_no_graph_wide_chi(t44_union):
    from atlas.emit import HarnessParameters

    g = _per_region(t44_union)
    g.overlap_cells = {FLUID: 4}
    h = HarnessParameters.from_graph(g).as_dict()
    assert h["overlap_cells"] is None and h["chi_shape"] is None
    assert h["assembly_projection"] is None
    assert h["regions"][FLUID] == {"overlap_cells": 4,
                                   "chi_shape": "partition-of-unity, ramp 8 cells",
                                   "assembly_projection": "none"}


# --------------------------------------------------------------------------
# 3. the declaration check -- fired, and beside the case where it is quiet
# --------------------------------------------------------------------------


def _w189(r):
    return [d for d in r.decisions.decisions
            if d.layer == "L2" and d.rule == "W189/regions"]


def _compile(g):
    from atlas import Budget, compile_scheme
    from atlas.probe import ProbeBudget

    return compile_scheme(g, Budget(), ProbeBudget())


@pytest.mark.parametrize("which", ["signed", "not-an-average"])
def test_W189_R11_and_E6_refuse_per_region_and_name_the_region(t44_union, which):
    """The two L6 refusals, fired on a region's own partition.

    The union's real partition is convex and closes, so neither refusal fires
    there -- and a rule that does not fire is evidence only beside a cell where it
    does.  A signed partition still sums to one and must be refused at R11; a
    partition whose weights sum to one half is not an average and must be refused
    at E6.  Each names the region, and E6 is stamped FAILED for the run.
    """
    import numpy as np
    from atlas.assembly import PartitionOfUnity
    from atlas.envelope import Hypothesis, Status

    p = t44_union.partition_of_unity
    ids = list(p.restrictions)
    n = p.n_global
    if which == "signed":
        w = np.linspace(-0.25, 1.25, n)
        weights, rule = {ids[0]: 1.0 - w, ids[1]: w}, "R11"
    else:
        w = np.linspace(0.0, 1.0, n)
        weights, rule = {ids[0]: 0.5 * (1.0 - w), ids[1]: 0.5 * w}, "E6"
    bad = PartitionOfUnity(n_global=n, restrictions=dict(p.restrictions),
                           weights=weights, contaminated=dict(p.contaminated),
                           ramp_cells=p.ramp_cells, profile=which)
    r = _compile(_per_region(t44_union, bad))
    l6 = [(d.rule, d.verdict.value, d.subject) for d in r.decisions.decisions
          if d.layer == "L6" and d.rule in ("E6", "R11", "L6/C1")]
    assert l6 == [(rule, "refuse", f"<assembly:{FLUID}>")], l6
    assert r.envelope[Hypothesis.E6] is Status.FAILS
    assert any(FLUID in e for e in r.envelope.evidence[Hypothesis.E6])


def test_W189_a_graph_declaring_nothing_per_region_gets_no_declaration_decision(t44_union):
    assert _w189(_compile(t44_union)) == []


def test_W189_a_clean_per_region_partition_is_admitted_by_name(t44_union):
    ds = _w189(_compile(_per_region(t44_union)))
    assert [d.verdict.value for d in ds] == ["admit"]
    assert ds[0].evidence["declared"] == {"partition_of_unity": [FLUID]}


@pytest.mark.parametrize("key, verdict, status_word", [
    ("heat-conduction-2d", "refuse", "whose only agent is BLOCK"),
    (COOLANT, "refuse", "NON-OVERLAPPING"),
    ("no-such-family", "refuse", "no agent of this graph"),
])
def test_W189_a_key_that_is_not_an_overlapping_region_is_refused(t44_union, key,
                                                                 verdict, status_word):
    g = _per_region(t44_union, key=key)
    ds = _w189(_compile(g))
    assert [d.verdict.value for d in ds] == [verdict], [d.message for d in ds]
    assert status_word in ds[0].message
    assert ds[0].subject == f"<region:{key}>"


def test_W189_a_partition_that_blends_another_regions_agent_is_refused(t44_union):
    """The identity and convexity both pass -- the weight is zero -- so only the
    NAMES are wrong, and only the names can catch it."""
    import numpy as np
    from atlas.assembly import PartitionOfUnity

    p = t44_union.partition_of_unity
    n = p.n_global
    bad = PartitionOfUnity(
        n_global=n,
        restrictions={**p.restrictions, "PASS": np.eye(n)},
        weights={**p.weights, "PASS": np.zeros(n)},
        contaminated={**p.contaminated, "PASS": np.zeros(n)},
        ramp_cells=p.ramp_cells, profile=p.profile)
    assert bad.identity_residual() <= 1e-12 and bad.chi_min() >= 0.0
    ds = _w189(_compile(_per_region(t44_union, bad)))
    assert [d.verdict.value for d in ds] == ["refuse"]
    assert ds[0].evidence["foreign"] == {"PASS": COOLANT}


def test_W189_a_partition_whose_subdomains_name_no_agent_is_decertified(t44_union):
    from atlas.assembly import PartitionOfUnity

    p = t44_union.partition_of_unity
    rename = {k: "X" + k for k in p.restrictions}
    odd = PartitionOfUnity(
        n_global=p.n_global,
        restrictions={rename[k]: v for k, v in p.restrictions.items()},
        weights={rename[k]: v for k, v in p.weights.items()},
        contaminated={rename[k]: v for k, v in p.contaminated.items()},
        ramp_cells=p.ramp_cells, profile=p.profile)
    ds = _w189(_compile(_per_region(t44_union, odd)))
    assert [d.verdict.value for d in ds] == ["admit-uncertified"]
    assert sorted(ds[0].evidence["unmatched"]) == sorted(rename.values())
    assert "W191" in ds[0].message


# --------------------------------------------------------------------------
# 4. the union on a real tiling, and its controls -- asserted from the record
# --------------------------------------------------------------------------

W189_ARTIFACT = os.path.join(OUT, "w189.json")
CIRCUIT = "lumped-dc-circuit"


@pytest.fixture(scope="module")
def w189():
    return _load(W189_ARTIFACT, "the union measurement (scripts/w189_region_assembly.py)")


def test_W189_the_union_carries_a_real_tiling_beside_both_circuits(w189):
    """`window_ns`'s four windows, its own grid partition and its own halo, beside
    `cooling_loop` and `powertrain`: three regions, the partition filed under one."""
    u = w189["union"]
    assert u["n_agents"] == 14 and u["n_seams"] == 14
    assert u["region_axes"] == {FLUID: "overlapping", COOLANT: "non-overlapping",
                                CIRCUIT: "non-overlapping"}
    assert u["assembly_kind"] == "per-region"
    assert u["assembly_regions"] == [FLUID]
    cert = u["assembly_region_certificates"][FLUID]
    assert cert["identity_holds"] is True and cert["norm_A"] == 1.0
    assert u["scheme"]["overlap"] is None
    assert u["scheme"]["overlap_by_region"] == {FLUID: 0.328125}
    assert u["harness"]["regions"][FLUID]["overlap_cells"] == 21
    assert u["envelope_E6"]["status"] == "holds"


def test_W189_every_rule_speaks_about_its_region_and_none_about_the_one_it_is_not(w189):
    """**The done-when, measured.** Every assembly and halo decision names the
    fluid region and only it; the substructuring criterion names each circuit."""
    u = w189["union"]
    assert u["speaks_only_about_the_tiling"] == {"stray": [], "unnamed": [], "ok": True}
    by_rule = u["coverage"]["regions_by_rule"]
    for rule in ("L2/W189/regions", "L2/R10/halo", "L2/C2", "L6/L6/C1", "L6/R12",
                 "L6/W49"):
        assert by_rule[rule] == [FLUID], (rule, by_rule.get(rule))
    assert u["substructuring_branch_regions"] == sorted([COOLANT, CIRCUIT])
    # and the union's one refusal is the clocks', not the assembly's: the same
    # union with only the coolant circuit -- whose clock matches the tiling's --
    # compiles without it
    assert u["refusals"] == ["L7/R9"]
    one = w189["union_one_circuit"]
    assert one["refusals"] == [] and one["speaks_only_about_the_tiling"]["ok"] is True


def test_W189_the_same_union_declared_the_old_way_names_no_region_at_all(w189):
    """The control that says the difference is the FORM: the graph-scoped union
    reaches the same verdict under the same assembly rules and names no region in
    any of them -- which is W189's defect, measured on the graph it is about."""
    u, gs = w189["union"], w189["union_graph_scoped"]
    assert gs["verdict"] == u["verdict"]
    assert gs["refusals"] == u["refusals"]
    scoped_rows = [r for r in gs["coverage"]["rows"]
                   if r["rule"].startswith(("L6/", "L2/R10/halo", "L2/C2"))]
    assert scoped_rows
    for row in scoped_rows:
        assert row["regions_named"] == [], row
    assert gs["assembly_kind"] == "partition-of-unity"
    assert u["n_decisions"] - gs["n_decisions"] == 2   # W189's admit, and C3 per circuit


def test_W189_equivalence_window_ns_declared_per_region_reaches_the_same_rules(w189):
    e = w189["equivalence"]
    assert e["same_verdict"] is True
    assert e["same_rules_and_verdicts"] is True
    assert e["only_in_graph_scoped"] == [] and e["only_in_per_region"] == []


def test_W189_two_overlapping_regions_every_rule_speaks_once_per_region(w189):
    """The positive control a one-region union cannot give: per region is not
    "the one overlapping region"."""
    regions = ["external-compressible-flow", "reacting-compressible-flow"]
    two = w189["two_regions"]["both_declared"]
    for rule in ("L2/R10/halo", "L2/C2", "L6/L6/C1", "L6/R12", "L6/W49"):
        assert two["coverage"]["regions_by_rule"][rule] == regions, rule
    assert two["assembly_regions"] == regions
    # one region with nothing filed under it is decertified by NAME, and E6 is
    # left unchecked however the regions were ordered
    one = w189["two_regions"]["one_declared"]
    assert one["envelope_E6"]["status"] == "unchecked"
    assert any(x.startswith("norm_A") for x in one["unmeasured"])
    e6 = [r for r in one["coverage"]["rows"] if r["rule"] == "L6/E6"]
    assert [r["regions_named"] for r in e6] == [["external-compressible-flow"]]


def test_W190_one_graph_constant_is_not_quoted_for_two_regions(w189):
    """C_mu and cut_defect_bound are one number per graph; with two overlapping
    regions each carrying a partition neither can say which region it was
    measured on, and each decertifies per region rather than being quoted.  The
    cell where it does not fire: the same graph with no constants declared."""
    withc = w189["two_regions"]["both_declared_with_graph_constants"]
    rules = [r["rule"] for r in withc["coverage"]["rows"]]
    assert rules.count("L6/W49/W190") == 2
    assert rules.count("L2/C2/W190") == 2
    without = w189["two_regions"]["both_declared"]
    assert not any("W190" in r["rule"] for r in without["coverage"]["rows"])


def test_W189_the_declaration_check_fires_on_each_defect_and_not_on_the_clean_union(w189):
    assert [d["verdict"] for d in w189["defects_clean_union_w189"]] == ["admit"]
    expect = {
        "key-names-a-sole-agents-family": "refuse",
        "key-names-the-coolant-NON-overlapping-region": "refuse",
        "partition-blends-a-circuit-agent": "refuse",
        "partition-names-no-agent": "admit-uncertified",
    }
    for key, verdict in expect.items():
        ds = w189["defects"][key]["w189_decisions"]
        assert ds[0]["verdict"] == verdict, key
        assert ds[0]["subject"].startswith("<region:"), key


def test_W192_found_joining_graphs_moves_an_agent_into_a_region_it_was_not_in(w189):
    """**Measured, not fixed.** `powertrain`'s ROTOR is the sole agent of its
    family in its own graph and CUT in the union, because the tiling's windows
    share its family: a region is a property of the GRAPH (W114's proxy), so
    putting two graphs together re-derives it for an agent neither changed.  The
    declaration check discloses it as unblended; no verdict moves on this graph,
    because the rotor has no stencil and no elliptic part."""
    changed = w189["composition_effects"]["changed"]
    assert [c["agent"] for c in changed] == ["ROTOR"]
    c = changed[0]
    assert c["from_graph"] == "powertrain"
    assert c["in_its_own_graph"] == {"status": "sole", "region": None}
    assert c["in_the_union"]["region"] == FLUID
    assert c["stencil_radius"] == 0 and c["elliptic_subsolve"] == "none"
    assert "ROTOR" in w189["union"]["w189_decisions"][0]["message"]


def test_W193_found_a_named_cross_point_is_judged_by_the_graphs_leftover_axis(w189):
    """**Measured, not fixed.** `cross_points` names vertices by label, so a
    vertex has no agents and no region, and `_cp_axis` falls back to the one axis
    field the graph still has.  Flip that field on the same union and the
    tiling's own "centre" is refused."""
    assert "L2/I2/G1" not in w189["union"]["refusals"]
    assert "L2/I2/G1" in w189["union_graph_axis_non_overlapping"]["refusals"]
