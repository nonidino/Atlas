"""Tier 46 -- W172: the joining seams exist, and what joining costs is measured.

Tier 39 counted the union rung 9 would form -- eighteen agents, five families,
forty ports, forty per-agent prolongations, five scale sets, zero per-pair
declarations -- and said of its own count that it was evidence about the algebra
and none about the integration cost, because the seams that would join the three
subsystems did not exist.  `atlas/cases/integration_union.py` builds them and
`scripts/w172_joining_cost.py` counts them.

What this file pins, in the order it has to be trusted in:

1. **The counter** reproduces Tier 39's table before it counts anything new, and
   the disjoint union built from the three `build()` calls IS Tier 39's union.
2. **The coherence check**: the two brief candidates that were replaced are not
   constructible, checked on the records rather than asserted.
3. **The joined union compiles**: its one refusal is the clock's, the same one the
   disjoint union has, and reconciling the clocks admits it.
4. **The brief's prediction, measured**: two new ports per new seam holds in count
   for J1 and J2 and fails in form for J3; "and nothing else" fails for all three;
   the declaration cost is identical at every family count measured, and J3's
   reach through the operating point is not (W196).
5. **The rotor faces four ways**, the native form being the refusing control, and
   the per-pair form in two measures that L3 admits (W195).
6. **The clocks** (W194), **the physics controls**, and the device-down passivity
   defect that `wake_array` already had (W197).

Numbers are asserted against ``out/w172/w172.json``; run
``python scripts/w172_joining_cost.py`` to rebuild it.  The live tests need
``out/w141/settled.npz``, a local-only cache, and skip without it.
"""

from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from atlas import compile_scheme                                      # noqa: E402
from atlas.cases import cooling_loop as CL                            # noqa: E402
from atlas.cases import integration_union as IU                       # noqa: E402

ARTIFACT = os.path.join(_ROOT, "out", "w172", "w172.json")
SETTLED = os.path.join(_ROOT, "out", "w141", "settled.npz")
DEVICE_SEAMS = ("J1_core_up", "J1_core_down", "J3_rotor_up", "J3_rotor_down")


@pytest.fixture(scope="module")
def art():
    if not os.path.exists(ARTIFACT):
        pytest.skip("out/w172/w172.json not on disk: run scripts/w172_joining_cost.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def state():
    if not os.path.exists(SETTLED):
        pytest.skip("out/w141/settled.npz is a local-only cache")
    d = np.load(SETTLED)
    return d["u"], d["v"]


@pytest.fixture(scope="module")
def joined(state):
    g, meta = IU.build(*state)
    return g, meta, compile_scheme(g)


@pytest.fixture(scope="module")
def disjoint(state):
    g, meta = IU.build(*state, joins=())
    return g, meta, compile_scheme(g)


@pytest.fixture(scope="module")
def J():
    import w172_joining_cost

    return w172_joining_cost


def _refusals(r):
    return {f"{d.layer}/{d.rule}" for d in r.decisions.refusals}


def _at(r, subject, layer=None, rule=None):
    return [d for d in r.decisions.decisions
            if d.subject == subject and (layer is None or d.layer == layer)
            and (rule is None or d.rule == rule)]


# --------------------------------------------------------------------------
# 1. the counter, before anything new is counted
# --------------------------------------------------------------------------


def test_W172_the_counter_reproduces_tier_39_before_it_counts_anything(art):
    cc = art["counter_control"]
    assert cc["equal"] is True
    for name, row in cc["published"].items():
        assert cc["measured"][name] == row, name
    assert cc["measured"]["union"] == {
        "K": 18, "families": 5, "ports": 40, "per_agent_prolongations": 40,
        "scale_sets": 5, "per_pair": 0, "pairs": 153}


def test_W172_the_disjoint_union_is_tier_39s_union(disjoint, J):
    g, _meta, _r = disjoint
    c = J.count(g)
    assert {k: c[k] for k in J.COUNT_KEYS + ("pairs",)} == J.TIER39_UNION
    # the only open ports in the three graphs are the rotor's two disk faces
    assert sorted(map(tuple, c["open_ports"])) == [("ROTOR", "down:MECH"),
                                                   ("ROTOR", "up:MECH")]


def test_W172_the_disjoint_union_declares_exactly_what_its_parts_declare(state, J):
    """Declaration for declaration, not just in count: the union is built by the
    three graphs' own `build()` calls, so nothing a join costs can hide in it."""
    from atlas.cases import cooling_loop, front_wing, powertrain

    u, v = state
    g, _ = IU.build(u, v, joins=())
    parts = [front_wing.build(u, v, motion=False)[0], cooling_loop.build()[0],
             powertrain.build()[0]]
    ours = {(a.agent_id, p.name): J.port_sig(p)
            for a in g.agents for p in a.capabilities.ports}
    theirs = {(a.agent_id, p.name): J.port_sig(p)
              for pg in parts for a in pg.agents for p in a.capabilities.ports}
    assert ours == theirs
    assert sorted(c.seam_id for c in g.connections) == sorted(
        c.seam_id for pg in parts for c in pg.connections)


# --------------------------------------------------------------------------
# 2. the coherence check the joins were chosen on
# --------------------------------------------------------------------------


def test_W172_the_two_brief_candidates_that_were_replaced_are_not_constructible(state):
    """THERM on the tiling, and ROT or MECH to the structure."""
    from atlas.cases import front_wing
    from atlas.ports import PortType

    g, _ = front_wing.build(*state, motion=False)
    for a in g.agents:
        if a.capabilities.governing_family == IU.FLUID:
            # no temperature anywhere on the tiling side: MECH is all there is
            assert {p.port_type for p in a.capabilities.ports} == {PortType.MECH}, \
                a.agent_id
    struct = g.agent("STRUCT").capabilities
    assert [p.name for p in struct.ports] == ["wet:MECH"]
    assert ("STRUCT", "wet:MECH") not in g.open_ports()
    assert not struct.ports_of_type(PortType.ROT)


# --------------------------------------------------------------------------
# 3. the joined union compiles
# --------------------------------------------------------------------------


def test_W172_the_joined_union_has_no_open_port_and_no_per_pair_declaration(joined, J, art):
    g, _meta, _r = joined
    c = J.count(g)
    assert (c["K"], c["families"], c["ports"], c["per_agent_prolongations"],
            c["scale_sets"], c["per_pair"], c["seams"]) == (18, 5, 48, 48, 5, 0, 24)
    assert c["open_ports"] == []
    assert art["unions"]["all|J1+J2+J3"]["count"]["ports"] == 48


def test_W172_the_joined_union_compiles_and_its_one_refusal_is_the_clock(joined,
                                                                          disjoint,
                                                                          art):
    """**Rewritten 2026-09-12, Tier 49, and the diagnosis is kept.**  Tier 46
    measured BOTH unions refusing at `L7/R9` and only there, and recorded the
    disjoint one's refusal as W194: every seam in it joins two agents on one
    clock, so nothing crossed a clock boundary and the rule fired anyway,
    because `is_multirate` reads every agent.  Tier 49 scoped R9's premise to
    the multirate SEAM, so the disjoint union is no longer refused.  The joined
    union's refusal is unmoved -- its five joining seams really do cross
    clocks -- and Tier 46's own record still carries what it measured."""
    _g, _m, r = joined
    _gd, _md, rd = disjoint
    assert _refusals(r) == {"L7/R9"}
    assert _refusals(rd) == set()                     # W194 closed, Tier 49
    assert art["unions"]["all"]["verdict"] == "refuse"     # as measured
    for seam in DEVICE_SEAMS + ("J2_heat", "x01_bypass", "x10_bypass"):
        decisions = _at(r, seam)
        assert decisions, seam
        assert all(d.verdict.value != "refuse" for d in decisions), seam
        assert [d.verdict.value for d in _at(r, seam, "L3", "C2/C3/C6")] == ["admit"], seam


def test_W172_reconciling_the_clocks_admits_the_joined_union(state):
    g, _ = IU.build(*state, clocks="reconciled")
    r = compile_scheme(g)
    assert r.verdict.value == "admit-uncertified"
    assert not _refusals(r)


def test_W172_a_split_seam_is_the_whole_face_seam_on_fewer_cells(joined, art):
    _g, _m, r = joined
    for seam in ("x01_bypass", "x10_bypass"):
        dim = _at(r, seam, "L3", "dim-M")
        assert len(dim) == 1 and "= 25" in dim[0].message, seam
    splits = 0
    for m in art["marginal"]:
        for key, x in m["compile"]["existing_seam_decisions_changed"].items():
            assert "->" in key and x["same_rules_and_verdicts"] is True, (m["from"], key)
            splits += 1
    assert splits == 5, "J1 twice and J3 three times, each splitting one seam"


# --------------------------------------------------------------------------
# 4. the prediction, measured
# --------------------------------------------------------------------------

#: "each new seam needs two new ports, one per side": J3 re-declares the rotor's
#: two open ports instead of adding two.
TWO_NEW_PORTS_PER_SEAM = {"J1": True, "J2": True, "J3": False}


def test_W172_two_new_ports_per_new_seam_holds_in_count_for_J1_and_J2_only(art):
    for j, row in art["per_join"].items():
        assert row["contexts"], j
        for c in row["contexts"]:
            assert c["two_new_ports_per_new_seam"] is TWO_NEW_PORTS_PER_SEAM[j], \
                (j, c["from"])


def test_W172_and_nothing_else_fails_for_every_join(art):
    for m in art["marginal"]:
        assert m["prediction"]["and_nothing_else"] is False, m["from"]
        assert m["prediction"]["what_else"], m["from"]
    first = {j: next(m for m in art["marginal"] if m["joins_added"] == [j])
             for j in IU.JOINS}
    assert first["J1"]["prediction"]["what_else"]["existing_seams_redeclared"] == 1
    assert first["J2"]["prediction"]["what_else"]["reconciliation_constants"] == 1
    assert [(x["agent"], x["port"], x["response_moved"])
            for x in first["J2"]["ports_moved_at_their_own_base"]] == [
        ("BLOCK", "wall:THERM", True)]
    assert first["J3"]["tally"]["joining_ports_redeclared"] == 2
    assert {(x["agent"], x["content"]) for x in first["J3"]["joining_sides"]
            if x["status"] == "re-declared"} == {
        ("ROTOR", "taken from the partner, away from the agent's own")}


def test_W172_the_declaration_cost_is_the_same_at_every_family_count(art):
    pj = art["per_join"]
    assert pj["J1"]["family_counts"] == [4, 5]
    assert pj["J2"]["family_counts"] == [4, 5]
    assert pj["J3"]["family_counts"] == [3, 5]
    assert pj["J1"]["identical_in_every_context"] is True
    assert pj["J2"]["identical_in_every_context"] is True
    t1 = pj["J1"]["contexts"][0]["tally"]
    assert (t1["joining_seams"], t1["joining_ports_new"], t1["existing_seams_redeclared"],
            t1["declared_constants"]) == (2, 4, 1, 2)
    t2 = pj["J2"]["contexts"][0]["tally"]
    assert (t2["joining_seams"], t2["joining_ports_new"], t2["reconciliation_constants"],
            t2["declared_constants"]) == (1, 2, 1, 2)
    for j in IU.JOINS:
        for c in pj[j]["contexts"]:
            assert c["tally"]["per_pair_new"] == 0, (j, c["from"])
            assert c["tally"]["scale_sets_new"] == 0, (j, c["from"])
            assert c["tally"]["refusals_new"] == 0, (j, c["from"])


def test_W196_J3s_reach_depends_on_whether_J2_exists(art):
    j3 = art["per_join"]["J3"]
    assert j3["identical_in_every_context"] is False
    assert sorted(j3["columns_that_differ_between_contexts"]) == [
        "ports_moved_at_their_own_base", "ports_moved_on_agents_off_the_new_seams",
        "state_data_rederived"]
    by = {c["from"]: c["tally"] for c in j3["contexts"]}
    assert [by[k]["ports_moved_on_agents_off_the_new_seams"]
            for k in ("fw+pt", "all|J1", "all|J1+J2")] == [3, 3, 6]
    # the control: the declaration columns themselves are identical in all three
    for k in ("joining_seams", "joining_ports_new", "joining_ports_redeclared",
              "existing_seams_redeclared", "per_pair_new", "declared_constants"):
        assert len({t[k] for t in by.values()}) == 1, k


def test_W196_the_state_J3_moves_is_the_state_J2_was_built_on(state):
    """Live and without a compile: the machine's speed, its heat, the block's datum."""
    u, v = state
    _g, without_j3 = IU.build(u, v, joins=("J1", "J2"))
    _g, with_j3 = IU.build(u, v)
    a, b = without_j3["info"], with_j3["info"]
    assert a["mgu_omega"] == 15.0
    assert b["mgu_omega"] == pytest.approx(13.837058774322221, rel=1e-12)
    assert a["p_ref"] == b["p_ref"], "the power unit is held, not re-fitted"
    assert a["q_machine"] == pytest.approx(CL.Q_SOURCE, rel=1e-12)
    assert b["q_machine"] == pytest.approx(134.31119921159222, rel=1e-9)
    assert b["t_case_ref"] < a["t_case_ref"]
    op = b["operating_point"]
    assert op["mgu_valid"] and op["rotor_valid"]
    assert op["torque_residual"] < 1e-12


# --------------------------------------------------------------------------
# 5. the rotor faces, four ways
# --------------------------------------------------------------------------


def test_W172_native_rotor_faces_are_refused_where_the_host_redeclaration_derives(state, joined):
    _g, _m, host = joined
    g, _ = IU.build(*state, rotor_ports="native")
    r = compile_scheme(g)
    for seam in ("J3_rotor_up", "J3_rotor_down"):
        assert [d.verdict.value for d in _at(r, seam, "L3", "C2/C3/C6")] == ["refuse"]
        assert [d.verdict.value for d in _at(host, seam, "L3", "C2/C3/C6")] == ["admit"]
        assert len(_at(host, seam, "L3", "dim-M")) == 1


def test_W195_a_per_pair_transfer_in_two_measures_admits_at_L3(state, art):
    g, _ = IU.build(*state, rotor_ports="per-pair-naive")
    r = compile_scheme(g)
    for seam in ("J3_rotor_up", "J3_rotor_down"):
        P = g.connection(seam).prolongations
        win = next(k for k in P if k != "ROTOR")
        ratio = (np.linalg.norm(P["ROTOR"].matrix, axis=0)
                 / np.linalg.norm(P[win].matrix, axis=0))
        assert np.allclose(ratio, 1.0 / math.sqrt(2.0), rtol=0.0, atol=1e-12)
        assert [d.verdict.value for d in _at(r, seam, "L3", "C2/C3/C6")] == ["admit"]
    # what does notice, and only because both bases here are absolute
    assert [d.verdict.value for d in _at(r, "J3_rotor_up", "L4", "probe-base")] == [
        "admit-uncertified"]
    # the control: the same seam-level form in the host's measure
    ctrl = art["rotor_forms"]["per-pair"]["seams"]["J3_rotor_up"]
    assert ctrl["same_trace_for_the_same_coefficient"] is True
    assert [row[2] for row in ctrl["decisions"] if row[1] == "probe-base"] == ["admit"]


def test_W172_the_rotor_forms_price_the_same_join_three_ways(art):
    rf = art["rotor_forms"]
    tally = {f: rf[f]["cost_against_the_union_without_J3"] for f in rf}
    assert (tally["host"]["joining_ports_redeclared"], tally["host"]["per_pair_new"]) == (2, 0)
    assert (tally["per-pair"]["joining_ports_redeclared"], tally["per-pair"]["per_pair_new"]) == (0, 2)
    assert tally["native"]["refusals_new"] == 2
    host_dim = rf["host"]["seams"]["J3_rotor_up"]["rotor_convention"]["dim_M"]
    pair_dim = rf["per-pair"]["seams"]["J3_rotor_up"]["rotor_convention"]["dim_M"]
    assert (host_dim, pair_dim) == (17, 9), "the per-pair form keeps the coarser cutoff"


# --------------------------------------------------------------------------
# 6. the clocks, the physics controls, and the down seams
# --------------------------------------------------------------------------


def test_W194_a_union_with_no_multirate_seam_is_no_longer_refused(disjoint, J, art):
    """**Rewritten 2026-09-12, Tier 49: the defect this row named is closed, and
    the diagnosis is kept rather than deleted.**  What Tier 46 measured is still
    in `out/w172/w172.json` and asserted below: the disjoint union has NO
    multirate seam and `refuse`d anyway, at `L7/R9` and nothing else, which is
    the false refusal W194 named.  `L7/R9`'s premise is now the multirate seam,
    so the live compile admits -- and re-running `scripts/w172_joining_cost.py`
    today would write a different verdict into that artifact than the one Tier
    46 published, which is recorded on [[joining-seam-cost]] as a dated note."""
    g, _m, r = disjoint
    assert J.count(g)["multirate_seams"] == []
    assert g.is_multirate() is True            # the AGENTS still differ
    assert "L7/R9" not in _refusals(r)         # the SEAMS do not (Tier 49)
    assert art["unions"]["all"]["verdict"] == "refuse"   # as Tier 46 saw it
    assert art["clocks"]["disjoint|reconciled|pointwise"]["verdict"] == "admit-uncertified"
    assert len(art["clocks"]["disjoint|reconciled|pointwise"][
        "dt_native_redeclared_against_native"]) == 10
    # the false refusal itself, as Tier 46 recorded it: no multirate seam, and
    # `L7/R9` fired on the graph anyway
    row = art["clocks"]["disjoint|native|pointwise"]
    assert row["multirate_seams"] == [] and row["verdict"] == "refuse"
    assert row["refusals"] == ["L7/R9@<graph>"]


def test_W194_time_integrated_matching_asks_all_eighteen_not_the_eight_at_a_multirate_seam(art):
    row = art["clocks"]["joined|native|time-integrated"]
    assert row["agents_at_multirate_seams"] == ["BLOCK", "F01", "F10", "F11", "F20",
                                                "MGU", "RAD", "ROTOR"]
    assert len(row["agents_without_integrated_response"]) == 18
    (quad,) = [x for x in row["refusals"] if x.startswith("L7/R9/quadrature@")]
    assert len(quad.split("@", 1)[1].split(", ")) == 18


def test_W172_the_mounted_block_keeps_the_first_law(art):
    balances = art["physics"]["block_first_law_at_the_release_state"]
    assert len(balances) == 3
    for name, bal in balances.items():
        assert bal["relative"] < 1e-6, name


def test_W172_J1_leaves_the_loop_where_it_was_built(art):
    p = art["physics"]
    assert p["loop_gain_unchanged_by_J1_at_the_build_state"] is True
    assert p["core_UA"]["at_build"] == CL.UA_RAD
    assert p["core_UA"]["relative_change"] == pytest.approx(
        1.1 ** IU.UA_EXPONENT - 1.0, rel=1e-12)


def test_W197_every_actuator_down_seam_reads_non_passive_and_wake_array_already_did(art, joined):
    _g, _m, r = joined
    for seam in ("J1_core_down", "J3_rotor_down"):
        assert [d.verdict.value for d in _at(r, seam, "L4", "E7/passivity")] == [
            "admit-uncertified"], seam
    for seam in ("J1_core_up", "J3_rotor_up"):
        assert not _at(r, seam, "L4", "E7/passivity"), seam
    wa = art["device_down_seams"]["wake_array_reference_uniform_flow"]["seams"]
    for rotor in ("R1", "R2", "R3"):
        down = [row for row in wa[f"{rotor}_down"] if row[1] == "E7/passivity"]
        assert [row[2] for row in down] == ["admit-uncertified"], rotor
        assert "2.000e+00" in down[0][3], rotor
        assert not [row for row in wa[f"{rotor}_up"] if row[1] == "E7/passivity"], rotor


# --------------------------------------------------------------------------
# 7. adding a subsystem, in both orders
# --------------------------------------------------------------------------


def test_W172_adding_a_subsystem_adds_exactly_its_own_structure_in_both_orders(art):
    for order, rows in art["orders"].items():
        added = [r for r in rows if r["added_subsystems"]]
        assert len(added) == 2, order
        for r in added:
            assert r["delta_is_exactly_the_added_structure"] is True, (order, r["to"])
            assert r["delta"]["per_pair"] == 0
            assert len(r["union_reconciliation"]["cut_axis_declarations_new"]) == 4


def test_W192_the_rotor_changes_region_when_powertrain_meets_the_tiling_and_no_join_moves_one(art):
    for order, rows in art["orders"].items():
        moved = {a: x for r in rows if r["added_subsystems"]
                 for a, x in r["union_reconciliation"]["agents_whose_region_changed"].items()}
        assert list(moved) == ["ROTOR"], order
        assert moved["ROTOR"]["in_the_union"] == IU.FLUID
    for m in art["marginal"]:
        assert m["agents_whose_region_changed"] == {}, m["from"]
