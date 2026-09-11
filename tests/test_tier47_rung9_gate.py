"""Tier 47 -- rung 9's gate, re-read against the graph that exists.

[[f1-pathmap-and-end-goal]] section 3 gives rung 9 the gate *"R closes;
composition error sub-linear in N"*.  Tier 46 built the full-vehicle graph --
`front_wing`'s tiling and both circuits, joined -- and this tier asked whether
each clause has a subject on it.  Neither does as written, and the replacement is
proposed rather than substituted, which is Tier 43's precedent for rung 3 (W183).

What this file pins:

1. **The clause about N has no variable here**: N = 13 is two different graphs,
   and every step up in N adds a different subsystem's physics.
2. **R(t) cannot be assembled from the records**: no field can carry
   dissipation, the tiling's domain boundaries are not ports, and the three
   clocks carry no unit (W200, W201).
3. **The replacement clause, measured join by join** -- each join's receiving
   subsystem's own balance with the join's term and without it: J2 closes; J3 is
   off by exactly the disk's declared width, and the two declared faces carry no
   net power; J1 carries no power at all.
4. **What closing J3 costs**: a host-sized rotor leaves the declared machine with
   no operating point, and `CircuitSolve` cannot be told the width (W199).

Numbers are asserted against ``out/rung9/rung9.json``; run
``python scripts/rung9_gate.py`` to rebuild it, in about two seconds.  The live
tests need ``out/w141/settled.npz``, a local-only cache, and skip without it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from atlas.capability import ExpertCapabilities                       # noqa: E402
from atlas.graph import CaseGraph                                     # noqa: E402
from atlas.cases import integration_union as IU                       # noqa: E402
from atlas.cases import powertrain as PT                              # noqa: E402
from atlas.cases import wake_array as WA                              # noqa: E402
from atlas.cases import wing_fsi as W                                 # noqa: E402

ARTIFACT = os.path.join(_ROOT, "out", "rung9", "rung9.json")
SETTLED = os.path.join(_ROOT, "out", "w141", "settled.npz")
PATHMAP = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                       "f1-pathmap-and-end-goal.md")
HOST_WIDTH = WA.ROTOR_CELLS * W.DX


@pytest.fixture(scope="module")
def art():
    if not os.path.exists(ARTIFACT):
        pytest.skip("out/rung9/rung9.json not on disk: run scripts/rung9_gate.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def state():
    if not os.path.exists(SETTLED):
        pytest.skip("out/w141/settled.npz is a local-only cache")
    d = np.load(SETTLED)
    return d["u"], d["v"]


@pytest.fixture(scope="module")
def G():
    import rung9_gate

    return rung9_gate


# --------------------------------------------------------------------------
# the gate being restated
# --------------------------------------------------------------------------


def test_W198_the_gate_being_restated_is_the_pathmaps_own():
    """The subject of the restatement, so the page cannot drift from it."""
    with open(PATHMAP, encoding="utf-8") as fh:
        lines = fh.readlines()
    # two tables open a row with rung 9 -- the ladder (section 3) and the status
    # recount (section 3.3) -- and the gate lives in the ladder's
    ladder = [line for line in lines
              if line.startswith("| **9** |") and "**Full-vehicle graph**" in line]
    assert len(ladder) == 1
    assert "$\\mathcal R$ closes; composition error sub-linear in $N$" in ladder[0]


def test_W198_N_is_not_a_variable_on_the_vehicle_graph(art):
    nv = art["n_on_the_vehicle"]
    assert nv["N_values"] == [8, 10, 13, 18]
    assert nv["N_with_more_than_one_graph"] == [13]
    assert nv["families_at_N13"] == [3, 4]
    assert nv["physics_added_with_every_increase_in_N"] is True


# --------------------------------------------------------------------------
# what R(t) needs, against what the records declare
# --------------------------------------------------------------------------


def test_W200_no_record_can_declare_dissipation_and_W201_none_a_unit():
    cap = [f.name for f in dataclasses.fields(ExpertCapabilities)]
    graph = [f.name for f in dataclasses.fields(CaseGraph)]
    assert not [f for f in cap + graph if "dissip" in f.lower()]
    assert not [f for f in cap if "unit" in f.lower()]


def test_W200_the_union_dissipates_through_nothing_declared(art):
    rp = art["r_prerequisites"]
    assert rp["n_storage_declared"] == 18
    assert rp["n_dissipative_by_their_own_constants"] == 13
    assert rp["open_ports"] == []
    assert rp["n_window_faces_that_are_domain_boundaries"] == 10


def test_W200_the_tilings_domain_boundaries_are_live_and_not_ports(state):
    g, meta = IU.build(*state)
    experts = meta["experts"]
    faces = sum(4 - len(experts[n].shared_faces) for n in W.DEFAULT_TILING.names)
    assert faces == 10
    assert g.open_ports() == []


def test_W201_three_clocks_in_three_unit_systems(art):
    rp = art["r_prerequisites"]
    assert rp["clocks_by_subsystem"] == {"front_wing": [0.0125],
                                         "cooling_loop": [0.05],
                                         "powertrain": [0.2]}
    assert set(rp["clock_units_as_the_modules_state_them"]) == {
        "front_wing", "cooling_loop", "powertrain"}


# --------------------------------------------------------------------------
# the replacement clause, join by join
# --------------------------------------------------------------------------


def test_W198_J2_closes_its_receiving_balance_with_its_term_and_not_without(art):
    rows = {k: v for k, v in art["j2_balance"].items()
            if isinstance(v, dict) and "relative" in v}
    assert len(rows) == 3
    for name, row in rows.items():
        assert row["relative"] < 1e-6, name
        assert row["relative_without_the_source_term"] > 0.1, name
        assert row["factor_without_over_with"] > 1e6, name


def test_W199_J3_power_is_off_by_exactly_the_disks_declared_width(G, state):
    ones = np.ones(WA.ROTOR_CELLS)
    native = G._face_accounting(ones, WA.DX, 1.0)
    host = G._face_accounting(ones, W.DX, 1.0)
    assert native["power_ratio_face_over_shaft"] == pytest.approx(1.0, abs=1e-14)
    assert host["power_ratio_face_over_shaft"] == pytest.approx(0.5, abs=1e-14)
    u_up = IU.ring_velocity(state[0], IU.ROTOR_SITE)
    declared = G._face_accounting(u_up, W.DX, 1.0)
    resized = G._face_accounting(u_up, W.DX, HOST_WIDTH)
    nonuni = declared["inflow_nonuniformity_mean_u3_over_mean_u_cubed"]
    assert nonuni > 1.0
    assert declared["power_ratio_face_over_shaft"] == pytest.approx(0.5 * nonuni, rel=1e-12)
    assert resized["power_ratio_face_over_shaft"] == pytest.approx(nonuni, rel=1e-12)


def test_W197_the_two_declared_faces_carry_no_net_power_at_the_rotors_base(G, state):
    u_up = IU.ring_velocity(state[0], IU.ROTOR_SITE)
    row = G._face_accounting(u_up, W.DX, 1.0)
    assert row["declared_two_face_net_power_at_the_rotors_base"] == 0.0
    assert row["shaft_power"] > 0.5
    rotor = WA.RotorDisk(agent_id="ROTOR", u_ref=u_up)
    assert np.array_equal(rotor.respond("up:MECH", u_up),
                          -rotor.respond("down:MECH", u_up))


def test_W199_circuit_solve_cannot_be_told_the_disk_width(G, art):
    assert G._circuit_solve_width_is_hard_coded() is True
    assert art["j3_power"]["circuit_solve_builds_its_disk_at_the_default_width"] is True


def test_W199_a_host_sized_rotor_leaves_the_declared_machine_with_no_operating_point(G, art, state):
    u_up = IU.ring_velocity(state[0], IU.ROTOR_SITE)
    u_mean = float(np.mean(u_up))

    def solve(width=None):
        rotor = WA.RotorDisk(agent_id="ROTOR", u_ref=u_up)
        kw = dict(rotor=rotor, elements=PT.make_elements(), u_ref=u_mean)
        return (PT.CircuitSolve(**kw) if width is None
                else G.SizedCircuitSolve(width, **kw)).solve()

    parent, same, host = solve(), solve(1.0), solve(HOST_WIDTH)
    # the control: the override at the declared width is the parent, bitwise
    assert (same.omega, same.induction, same.current) == (
        parent.omega, parent.induction, parent.current)
    assert parent.rotor_valid is True
    assert host.rotor_valid is False
    so = art["sized_operating_points"]
    assert so["width_1_reproduces_the_parent"] is True
    assert so["width 1, the declared rotor"]["demand_over_largest_supply"] < 1.0
    assert so["width 0.5, the host face"]["demand_over_largest_supply"] > 10.0


def test_W198_J1_carries_no_power_and_its_coupling_moves_the_loop(art):
    j1 = art["j1_loop_response"]
    assert j1["power_crossing_into_the_coolant_through_J1"] == 0.0
    assert j1["identical_at_the_build_state"] is True
    assert j1["t_return_change_K_for_air_plus_10pct"] < -1.0
    assert j1["rad_rejection_change_W_for_air_plus_10pct"] > 10.0


def test_W198_only_J3_joins_a_single_family(art):
    joins = art["referents"]["joining_seams"]
    assert {k: v["same_family"] for k, v in joins.items()} == {
        "J1_core_up": False, "J1_core_down": False, "J2_heat": False,
        "J3_rotor_up": True, "J3_rotor_down": True}
    assert art["referents"]["tiling_monolith_declared"] is True
