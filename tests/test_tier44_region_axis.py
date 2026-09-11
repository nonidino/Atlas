"""Tier 44 -- W171: the decomposition axis, per region rather than per graph.

`CaseGraph.decomposition` was ONE field for the whole graph, and rung 9's graph
does not have one axis: a vehicle is a fluid tiling and two circuits at once.
Seven functions read that field and three of them returned early on the wrong
axis **having emitted nothing**, so a union graph got silence about whichever
region the graph-level field did not name.

What this file pins, in the order the change has to be trusted in.

**First the control, because it is the whole safety argument.**  The axis is now
derived per region, and every graph written before W171 has exactly one axis --
so the derivation must reproduce the declared value on every one of them, and
`D_decomposition` must come out of the artifact unchanged.  If that fails the
change is not a schema extension, it is a behaviour change on graphs nobody was
asking about.

**Then the correction.**  Tier 40 proposed deriving the axis from
`geometrically_coincident` and measured the proposal on four graphs.  Over all
eleven it does not separate the axis, so the axis is a DECLARATION --
`Connection.cut_axis`, defaulting to None = inherit.  That falsification lives
in `test_tier40_epsilon_halo.py` beside the claim it corrects; what is here is
the repair.

**Then the thing the field could not say**: a union graph carrying two axes, and
the seven rules reading the region their subject is in.

**And the part-2 boundary, as a measurement rather than a claim.**
`partition_of_unity` is still one object per graph, so `L6/R12` and `L6/W49` are
gated on it before they ever reach the axis.  Give the union a partition and all
three formerly-silent rules speak; withhold one and two stay quiet for a reason
that is not the axis.  That is exactly where Tier 41 drew part 2.

Numbers are asserted against `out/w171b/w171b.json` where the artifact carries
them; run `python scripts/w171_region_axis.py` first.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from atlas import Budget, compile_scheme                              # noqa: E402
from atlas.compiler import _Context                                   # noqa: E402
from atlas.graph import Connection, Decomposition                     # noqa: E402
from atlas.probe import ProbeBudget                                   # noqa: E402
from atlas.scheme import Scheme                                       # noqa: E402

ARTIFACT = os.path.join(_ROOT, "out", "w171b", "w171b.json")


@pytest.fixture(scope="module")
def w171():
    if not os.path.exists(ARTIFACT):
        pytest.skip("run scripts/w171_region_axis.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def union():
    import w171_region_axis as W

    return W.union_graph(), W.union_graph(with_pou=True)


# --------------------------------------------------------------------------
# 1. the control: the change moved nothing
# --------------------------------------------------------------------------


def test_W171_control_every_existing_graph_keeps_exactly_one_axis(w171):
    """One axis each, equal to the declared one, on every constructible graph."""
    rows = [r for r in w171["control"]["rows"] if "skipped" not in r]
    assert len(rows) >= 7
    assert w171["control"]["disagree"] == []
    for r in rows:
        assert r["reproduces_declared"] is True, r["case"]
        assert len(r["axis_set"]) <= 1, r["case"]
        for axis in r["regions"].values():
            assert axis == r["declared"], r["case"]


def test_W171_control_the_artifact_key_is_unchanged_for_a_single_axis_graph():
    """`D_decomposition` still emits one value, and the new key agrees with it.

    An artifact whose meaning changed for every graph in the vault would be a
    much larger act than this row asked for. The set is emitted BESIDE the axis,
    and for a single-axis graph it is that axis in a one-element list.
    """
    from atlas.cases import cooling_loop as CL

    graph = CL.build()
    graph = next(x for x in graph if hasattr(x, "agents")) if isinstance(
        graph, tuple) else graph
    result = compile_scheme(graph, Budget(), ProbeBudget())
    d = result.scheme.as_dict()
    assert d["D_decomposition"] == "non-overlapping"
    assert d["D_decomposition_axes"] == ["non-overlapping"]


# --------------------------------------------------------------------------
# 2. the repair: a declaration on the connection
# --------------------------------------------------------------------------


def test_W171_the_axis_is_declared_on_the_CONNECTION_and_defaults_to_inherit():
    """Tier 40's carrier finding survives; its derivability finding does not.

    A cut is a relation among several agents, so the agent is the wrong carrier
    -- that reasoning is untouched. What is false is that the field was already
    there: `geometrically_coincident` does not separate the axis over eleven
    graphs (the falsification is in `test_tier40_epsilon_halo.py`). So it is
    declared, and `None` inherits, which is what every prior graph means.
    """
    assert "cut_axis" in Connection.__dataclass_fields__
    assert Connection.__dataclass_fields__["cut_axis"].default is None
    assert "decomposition_axes" in Scheme.__dataclass_fields__


def test_W171_a_cross_family_seam_is_not_a_cut_at_all(w171):
    """Gamma between two families is a physical boundary, not a face a
    decomposition created, so neither axis applies to it."""
    seen = False
    for r in w171["control"]["rows"]:
        if "skipped" in r:
            continue
        for seam, axis in r["seam_axes"].items():
            if seam in r["not_a_cut"]:
                assert axis is None
                seen = True
    assert seen, "no cross-family seam in the census -- the claim is untested"


# --------------------------------------------------------------------------
# 3. the union: two axes in one graph
# --------------------------------------------------------------------------


def test_W171_the_union_carries_two_axes_where_one_field_could_say_one(w171):
    u = w171["union"]
    assert u["carries_two_axes"] is True
    assert sorted(u["axis_set"]) == ["non-overlapping", "overlapping"]
    # and the single field can only say one of them, which is the defect
    assert u["declared_single_field"] in u["axis_set"]
    assert len(u["regions"]) == 2
    assert set(u["regions"].values()) == {"non-overlapping", "overlapping"}


def test_W171_the_artifact_reports_the_SET_beside_the_scheme_axis(w171):
    for key in ("union_compile", "union_with_pou_compile"):
        c = w171[key]
        assert c["D_decomposition_axes"] == ["non-overlapping", "overlapping"], key
        # the scheme's own axis stays ONE value: the interface problem is one
        # problem however many regions the graph has
        assert c["D_decomposition"] in ("non-overlapping", "overlapping"), key


def test_W171_no_rule_is_silent_about_a_region_any_more(w171):
    """The defect, closed, and measured on the graph that exhibits it.

    Before: with an overlapping region present, `R10/halo`, `R12` and `W49` all
    emitted nothing. After, with a partition of unity so that none of the three
    is gated on something else, all three speak.
    """
    after = w171["union_with_pou_compile"]["coverage"]
    assert after["overlapping_regions_present"], "no overlapping region to be silent about"
    assert after["silent_rules"] == [], after["silent_rules"]
    for name in ("R10/halo", "R12", "W49"):
        assert after["rules"][name], name


def test_W171_the_halo_rule_scopes_to_the_overlapping_region(w171):
    """It speaks about the tiling's agents and not about the circuit's."""
    rows = w171["union_compile"]["coverage"]["rules"]["R10/halo"]
    assert rows, "the halo rule went quiet again"
    subject = rows[0]["subject"]
    assert subject.startswith("F_") or "F_" in subject, subject


def test_W171_the_cut_policy_emits_BOTH_branches_on_a_union(w171):
    """L2/C2 is the overlapping criterion and L2/C3 the substructuring one.

    A union graph is on both, and before W171 exactly one was emitted -- the
    wrong one for one of its regions, loudly. Both now appear.
    """
    rules = w171["union_compile"]["rules"]
    assert "L2/C2" in rules
    assert any(r.startswith("L2/C3") for r in rules), rules
    assert "L5/W171/axes" in rules


# --------------------------------------------------------------------------
# 4. the part-2 boundary, measured
# --------------------------------------------------------------------------


def test_W171_part_2_was_the_partition_of_unity_and_W189_files_it_per_region(w171, union):
    """**Closed in Tier 45 (W189). The measurement is kept and the claim moved on.**

    *The diagnosis, which stands as a measurement of Tier 44's tree:* two rules
    stayed silent WITHOUT a partition, and not because of the axis --
    `_r12_conservative_assembly` and `_w49_sigma_branch` are both gated on
    `graph.partition_of_unity` before they reach any axis, and that object was one
    per graph, so a union whose tiling had a partition had to declare it as the
    WHOLE graph's.  That is Tier 41's part 2, and the numbers below are Tier 44's.

    *What changed:* the partition, the overlap and the overlap-cell count can each
    be filed under a REGION.  The graph-scoped form still means what it meant --
    so this fixture, which declares its tiling's partition as the graph's, still
    compiles exactly as Tier 44 measured it (the byte-identity control in
    `test_tier45_region_assembly.py` compiles it and compares every byte) -- and
    the per-region form is what the fixture could not say.
    """
    without = w171["union_compile"]["coverage"]["silent_rules"]
    with_pou = w171["union_with_pou_compile"]["coverage"]["silent_rules"]
    assert sorted(without) == ["R12", "W49"], without
    assert with_pou == [], with_pou
    # the halo rule is NOT gated on the partition, which is why it speaks in both
    assert w171["union_compile"]["coverage"]["rules"]["R10/halo"]

    # the closure: the fixture's single partition is the GRAPH's and is not read
    # as any region's -- and the same object filed under the tiling's region is
    _, with_pou_graph = union
    assert not with_pou_graph.per_region("partition_of_unity")
    fluid = "incompressible-navier-stokes-2d"
    assert with_pou_graph.partition_for(fluid) is None
    import copy

    scoped = copy.copy(with_pou_graph)
    scoped.partition_of_unity = {fluid: with_pou_graph.partition_of_unity}
    assert scoped.per_region("partition_of_unity")
    assert scoped.partition_for(fluid) is with_pou_graph.partition_of_unity


def test_W171_the_fields_that_were_graph_global_now_have_a_per_region_form():
    """Part 2's subjects, asserted so the boundary cannot quietly move -- again.

    Tier 44 pinned that these three were still one per graph.  Tier 45 gives each
    a per-region form, and the control is that nothing ELSE grew one: asking for
    the per-region form of any other field raises rather than answering False.
    """
    from atlas.graph import CaseGraph, GraphError

    assert CaseGraph.PER_REGION_FIELDS == ("partition_of_unity", "overlap",
                                           "overlap_cells")
    for name in CaseGraph.PER_REGION_FIELDS:
        assert name in CaseGraph.__dataclass_fields__, name
    # and the axis is per-seam and per-region, as Tier 44 left it
    assert hasattr(CaseGraph, "region_axes")
    assert hasattr(CaseGraph, "seam_axis")
    assert hasattr(CaseGraph, "agent_axis")
    from atlas.cases import cooling_loop as CL

    g = CL.build()[0]
    with pytest.raises(GraphError):
        g.per_region("cross_points")


# --------------------------------------------------------------------------
# 5. the helper, and R2's lift
# --------------------------------------------------------------------------


def test_W171_R2s_lift_to_the_non_overlapping_view_stays_graph_wide(union):
    """One helper for seven sites, and it respects the one global axis decision.

    R2 lifts the achievable rung to probed-DtN, which requires the
    non-overlapping view -- of the INTERFACE PROBLEM, which is one problem
    however many regions there are. `ctx.region_axis` returns non-overlapping
    everywhere under that lift, which is what keeps every lifted graph behaving
    as it did before W171.
    """
    g, _ = union
    ctx = _Context(graph=g, budget=Budget(), probe_budget=ProbeBudget(),
                   references={}, record=None, stamp=None, holes=None,
                   probe_state="x", depth=0)
    # no lift: the regions speak for themselves
    ctx.decomposition = g.decomposition
    assert set(ctx.axis_summary().values()) == {"non-overlapping", "overlapping"}
    # under R2's lift every region is non-overlapping
    ctx.decomposition = Decomposition.NON_OVERLAPPING
    g.decomposition = Decomposition.OVERLAPPING
    assert set(ctx.axis_summary().values()) == {"non-overlapping"}
    assert ctx.regions_on(Decomposition.OVERLAPPING) == {}
    assert ctx.region_axis("anything") is Decomposition.NON_OVERLAPPING
