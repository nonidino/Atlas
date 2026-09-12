"""Tier 49 -- rung 9: the union marched, and the seven decisions a march needs.

Tier 46 built the joined union and compiled it; Tier 47 restated rung 9's gate
and measured each join's receiving balance at the RELEASE STATE, one step each.
This tier makes the seven decisions `rung9-gate-restated` section 6 lists, and
marches.

What this file pins:

1. **The units (W201)** -- a length and a speed put three unit systems on one
   clock, and the bare numbers `L7/R9` compares understate the real spread by
   25x: 16 against 400.  The spread is ``4 u0 / l0`` and is never below 100
   anywhere in a plausible F1 box, so the decision moves the number and not the
   conclusion.
2. **The rotor and the machine (W199)** -- the similarity that re-sizes the
   machine for a host-sized rotor, with BOTH controls: width 1 must reproduce
   Tier 46's operating point and Tier 47's un-resized failure must reproduce
   its 31x.
3. **The devices as body forces (W94)** -- the donor's own exact sink, with
   ``sum(f dA) == -thrust`` to the bit, and the one-cell thickness floor this
   host needs.
4. **W194** -- `L7/R9` scoped to the multirate SEAM: the disjoint union stops
   being refused for a mismatch no seam carries, and every graph that has a
   multirate seam still refuses.  The controls are the ones that must NOT move.
5. **The coolant clock does not reach the fluid**, bitwise -- which is what
   makes `replay_coolant` a replay and not a second model.
6. **The gate's numbers are the page's numbers** -- every threshold in
   `vehicle_march.GATE` appears in the case study, because a criterion written
   in prose and evaluated in code drifts, and Tier 48 measured that it drifts
   permissive.

The live marches need ``out/w141/settled.npz``, a local-only cache, and skip
without it.  The artifact tests need ``out/tier49/tier49.json``; rebuild it with
``python scripts/tier49_union_march.py``, which is about half an hour because
the five tight arms are 1200 macro-steps each at 0.22 s a step.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from atlas.compiler import compile_scheme                             # noqa: E402
from atlas.cases import cooling_loop as CL                            # noqa: E402
from atlas.cases import integration_union as IU                       # noqa: E402
from atlas.cases import powertrain as PT                              # noqa: E402
from atlas.cases import thermal_seam as TS                            # noqa: E402
from atlas.cases import vehicle_march as VM                           # noqa: E402
from atlas.cases import wake_array as WA                              # noqa: E402
from atlas.cases import wing_fsi as W                                 # noqa: E402

ARTIFACT = os.path.join(_ROOT, "out", "tier49", "tier49.json")
SETTLED = os.path.join(_ROOT, "out", "w141", "settled.npz")
WIKI = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common")
CASE = os.path.join(WIKI, "case-study-vehicle-march-atlas-0.1.md")
SCALE_PAGE = os.path.join(WIKI, "vehicle-scale-and-sizing.md")


@pytest.fixture(scope="module")
def art():
    if not os.path.exists(ARTIFACT):
        pytest.skip("out/tier49/tier49.json not on disk: run "
                    "scripts/tier49_union_march.py")
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
    g, _ = IU.build(*state)
    return g


@pytest.fixture(scope="module")
def disjoint(state):
    g, _ = IU.build(*state, joins=())
    return g


# ==========================================================================
# 1. the units (W201)
# ==========================================================================


def test_W201_the_bare_numbers_understate_the_clock_spread_by_25x():
    """`L7/R9` compares bare numbers; in seconds the spread is 25x larger."""
    t = VM.clock_table()
    assert t["bare_numbers"] == [0.0125, 0.05, 0.2]
    assert t["spread_in_bare_numbers"] == 16.0
    assert t["spread_in_seconds"] == pytest.approx(400.0)
    assert t["spread_in_seconds"] / t["spread_in_bare_numbers"] == pytest.approx(25.0)


def test_W201_the_ordering_of_the_clocks_changes_when_they_get_units():
    """By bare number the coolant circuit is the MIDDLE clock; in seconds it is
    the slowest, by a factor of 50 over the one that looked slowest."""
    rows = VM.clock_table()["subsystems"]
    by_bare = sorted(rows, key=lambda k: rows[k]["dt_native_bare"])
    by_secs = sorted(rows, key=lambda k: rows[k]["dt_native_s"])
    assert by_bare == ["front_wing", "cooling_loop", "powertrain"]
    assert by_secs == ["front_wing", "powertrain", "cooling_loop"]
    assert (rows["cooling_loop"]["dt_native_s"]
            / rows["powertrain"]["dt_native_s"]) == pytest.approx(50.0)


def test_W201_the_clock_ratio_is_4_u0_over_l0_and_never_below_100():
    """The decision moves the number and not the conclusion."""
    alt = VM.VEHICLE.alternatives()
    assert alt["ratio_min"] == pytest.approx(120.0)
    assert alt["ratio_max"] == pytest.approx(1440.0)
    assert alt["ratio_never_below_100"] is True
    for l0 in (0.25, 0.5, 1.0):
        for u0 in (30.0, 50.0, 90.0):
            v = VM.VehicleScale(l0_m=l0, u0_ms=u0)
            assert VM.clock_ratio(v) == pytest.approx(4.0 * u0 / l0)


def test_W201_the_clocks_nest_so_a_sub_cycled_march_is_well_posed():
    assert VM.N_FLUID_PER_COOLANT == 400
    assert VM.clock_ratio(VM.VEHICLE) == pytest.approx(VM.N_FLUID_PER_COOLANT)


def test_W201_kinematic_similarity_is_not_dynamic_similarity():
    """A page with metres on it invites the other reading, so this is pinned."""
    v = VM.VEHICLE
    assert v.reynolds_model == pytest.approx(250.0)
    assert v.reynolds_vehicle > 1.5e6
    assert v.reynolds_vehicle / v.reynolds_model > 6.0e3


# ==========================================================================
# 2. the rotor and the machine (W199)
# ==========================================================================


def test_W199_the_width_1_control_reproduces_Tier_46s_operating_point(state):
    """`SizedCircuitSolve` overrides two reads of the disk and nothing else, so
    at width 1 it must be `CircuitSolve`.  Without this control the similarity
    in the next test could be an artefact of the override."""
    u, _v = state
    ring = IU.ring_velocity(u, IU.ROTOR_SITE)
    res, els, _disk = VM.operating_point(ring, width=1.0, scale=1.0)
    parent = PT.CircuitSolve(rotor=WA.RotorDisk(agent_id="ROTOR", u_ref=ring),
                             elements=PT.make_elements(),
                             u_ref=float(np.mean(ring))).solve()
    for k in ("omega", "induction", "current", "p_mech"):
        assert getattr(res, k) == getattr(parent, k)
    assert res.omega == pytest.approx(13.837059, abs=1e-5)
    assert res.induction == pytest.approx(0.113792, abs=1e-6)
    assert res.current == pytest.approx(0.145686, abs=1e-6)
    p_ref = IU.calibrated_p_ref(PT.MachineAgent())[0]
    q = res.current ** 2 * els["MGU"].resistance * p_ref
    assert q == pytest.approx(134.31, abs=0.01)     # joining-seam-cost section 8.1


def test_W199_the_unsized_machine_reproduces_Tier_47s_31x(state):
    u, _v = state
    ring = IU.ring_velocity(u, IU.ROTOR_SITE)
    res, _els, _d = VM.operating_point(ring, width=VM.HOST_ROTOR_WIDTH, scale=1.0)
    assert res.rotor_valid is False
    assert res.induction == pytest.approx(PT.A_MAX)
    mod = WA.RotorDisk(agent_id="_P", u_ref=ring)._mod
    best = max(float(mod.ActuatorDisk(a=a, area=VM.HOST_ROTOR_WIDTH)(
        float(np.mean(ring))).torque) for a in np.linspace(PT.A_MIN, PT.A_MAX, 800))
    assert res.torque_machine / best == pytest.approx(31.15, abs=0.05)


def test_W199_the_similarity_is_exact_and_the_cost_is_a_factor_of_two(state):
    """k_e = k_t scale with the width and every resistance inversely, so the
    induction and the demand-over-supply ratio do not move at all."""
    u, _v = state
    ring = IU.ring_velocity(u, IU.ROTOR_SITE)
    ctrl, els_c, d_c = VM.operating_point(ring, width=1.0, scale=1.0)
    sized, els_s, d_s = VM.operating_point(ring, width=VM.HOST_ROTOR_WIDTH,
                                           scale=VM.ROTOR_SCALE)
    assert sized.rotor_valid is True
    assert sized.induction == pytest.approx(ctrl.induction, abs=1e-12)
    assert sized.torque_residual == pytest.approx(ctrl.torque_residual, abs=1e-14)
    assert sized.omega == pytest.approx(2.0 * ctrl.omega, rel=1e-12)
    assert sized.current == pytest.approx(0.5 * ctrl.current, rel=1e-12)
    assert d_s.power / d_c.power == pytest.approx(0.5, rel=1e-12)
    q_c = ctrl.current ** 2 * els_c["MGU"].resistance
    q_s = sized.current ** 2 * els_s["MGU"].resistance
    assert q_s / q_c == pytest.approx(0.5, rel=1e-12)


def test_W199_the_machines_heat_is_its_own_windings_not_the_loop_total():
    """R_TOTAL = 2 R_MGU on this circuit, so charging the loop total doubles
    the heat EXACTLY -- and a factor of exactly two is invisible in a ratio.
    `integration_union` and `CooledMachine.heat_flux` both use R_MGU."""
    els = VM.machine_for_rotor(1.0)
    assert els["MGU"].resistance == pytest.approx(PT.R_MGU)
    assert sum(e.resistance for e in els.values()) == pytest.approx(PT.R_TOTAL)
    assert PT.R_TOTAL == pytest.approx(2.0 * PT.R_MGU)


def test_W199_the_parent_CircuitSolve_still_cannot_be_told_the_width():
    """The defect is diagnosed, not closed: `powertrain` is unchanged, so any
    other caller still gets a one-diameter rotor's operating point."""
    import inspect

    src = inspect.getsource(PT.CircuitSolve.torque_at)
    assert "ActuatorDisk(a=float(a))" in src
    assert "area" not in src


# ==========================================================================
# 3. the devices as body forces (W94)
# ==========================================================================


def test_W94_the_body_force_is_the_thrust_to_the_bit():
    """The donor's exact-overlap weighting: a discretization that lost 3% of the
    thrust would read as a 3% interface residual and be blamed on the join."""
    f = VM.DeviceForcing()
    for dev in VM.DEVICES:
        for thrust in (0.05, 0.1093, 0.3):
            fld = f.field(thrust, dev, 0.92)
            assert float(np.sum(fld) * W.DX * W.DX) == pytest.approx(
                -thrust, abs=1e-15)


def test_W94_this_host_needs_a_thickness_floor_and_wake_arrays_did_not():
    """The donor's rule is Delta_d = <U_d> dt.  On `wake_array`'s lattice that
    is 1.6 cells; here the cell is half as wide and the clock sixteen times
    finer, so it is 0.18 of one."""
    f = VM.DeviceForcing()
    used, raw = f.thickness(0.92)
    assert raw / W.DX == pytest.approx(0.182, abs=0.01)
    assert used == pytest.approx(W.DX)
    native = max(1.0, 0.05) * WA.MACRO_DT / WA.DX
    assert native > 1.5


def test_W94_the_disk_reads_its_inflow_upstream_of_the_strip():
    """`disk.disk_average`'s own rule: sampling inside the strip would read a
    velocity the disk's force has already slowed.  Here the ring is 7.5 cells
    upstream, and that gap is the whole of J3's residual."""
    for dev in VM.DEVICES:
        gap = dev.x_plane / W.DX - (dev.i_up + 0.5)
        assert gap == pytest.approx(7.5)
        assert dev.i_up < dev.i_down


# ==========================================================================
# 4. W194 -- L7/R9 scoped to the multirate seam
# ==========================================================================


def _refusals(result):
    return sorted({f"{d.layer}/{d.rule}" for d in result.decisions.refusals})


def test_W194_the_disjoint_union_has_no_multirate_seam_and_is_no_longer_refused(
        disjoint):
    """Tier 46's finding, closed.  Every seam in the disjoint union joins two
    agents on ONE clock and no flux crosses between two of them; the rule
    refused it anyway, because `is_multirate` reads every agent."""
    assert disjoint.is_multirate() is True
    assert disjoint.multirate_seams() == []
    r = compile_scheme(disjoint)
    assert "L7/R9" not in _refusals(r)
    scope = [d for d in r.decisions._decisions if d.rule == "R9/scope"]
    assert len(scope) == 1
    assert "NO SEAM JOINS TWO OF THEM" in scope[0].message


def test_W194_E4_still_fails_on_the_disjoint_union(disjoint):
    """What the narrower premise costs is the refusal, not the hypothesis: the
    agents really do run at different steps."""
    from atlas.envelope import Hypothesis, Status

    r = compile_scheme(disjoint)
    assert r.envelope[Hypothesis.E4] is Status.FAILS
    assert r.scheme.multirate is True


def test_W194_the_joined_union_has_five_multirate_seams_and_still_refuses(joined):
    """The joins make the mismatch real, and the rule must still fire."""
    assert joined.multirate_seams() == ["J1_core_down", "J1_core_up", "J2_heat",
                                        "J3_rotor_down", "J3_rotor_up"]
    assert joined.agents_at_multirate_seams() == [
        "BLOCK", "F01", "F10", "F11", "F20", "MGU", "RAD", "ROTOR"]
    assert "L7/R9" in _refusals(compile_scheme(joined))


def test_W194_the_quadrature_branch_asks_the_eight_not_the_eighteen(state):
    """Tier 46: the requirement named all 18 agents where 8 sit at a multirate
    seam.  `FluxMatching`'s own docstring says every agent AT one."""
    from atlas.graph import FluxMatching

    g, _ = IU.build(*state, flux_matching=FluxMatching.TIME_INTEGRATED)
    r = compile_scheme(g)
    quad = [d for d in r.decisions.refusals if d.rule == "R9/quadrature"]
    assert len(quad) == 1
    named = {a.strip() for a in quad[0].subject.split(",")}
    assert named <= set(g.agents_at_multirate_seams())
    assert len(named) <= 8
    assert len(g.agents) == 18


def test_W194_a_graph_whose_seam_joins_two_clocks_still_refuses():
    """The control that must NOT move: `thermal_seam`'s 500:1 mismatch is
    carried BY its seam, so R9's premise is still met and the narrower rule
    still fires.  Its DEFAULT build declares time-integrated matching and
    supplies the integrated responses, so it admits for Tier 17's reason and
    not for anything Tier 49 did -- both cells are asserted, because only the
    pair says the scoping did not weaken the rule."""
    from atlas.graph import FluxMatching

    gp, _ = TS.build(flux_matching=FluxMatching.POINTWISE)
    assert gp.multirate_seams() == ["cht"]
    assert "L7/R9" in _refusals(compile_scheme(gp))
    gd, _ = TS.build()
    assert gd.flux_matching is FluxMatching.TIME_INTEGRATED
    assert _refusals(compile_scheme(gd)) == []


def test_W194_the_scoping_is_the_narrower_predicate_not_a_weaker_one(joined,
                                                                    disjoint):
    """Every graph with a multirate seam is multirate by agents too, so the new
    premise implies the old one and never the reverse."""
    for g in (joined, disjoint):
        if g.multirate_seams():
            assert g.is_multirate() is True


# ==========================================================================
# 5. the march
# ==========================================================================


def test_the_coolant_clock_does_not_reach_the_fluid(state):
    """**Bitwise**, and it is what makes `replay_coolant` a replay.  Neither the
    loop's temperature nor the block's reaches the fluid -- the air carries no
    temperature, and the machine's R(T) enters its THERM response only -- so
    the fluid trajectory does not depend on how often the coolant steps."""
    u, v = state
    a = VM.march_union(u, v, steps=20, join_coupling="lagged", n_per_coolant=400)
    b = VM.march_union(u, v, steps=20, join_coupling="lagged", n_per_coolant=5)
    assert len(a.coolant) == 0 and len(b.coolant) == 4
    for k in ("u_rotor", "u_core", "induction", "current", "thrust_rotor",
              "power_rotor", "ua"):
        assert np.array_equal(np.asarray(a.trace[k]), np.asarray(b.trace[k])), k


def test_a_repeat_is_bitwise(state):
    """The floor every difference is quoted against."""
    u, v = state
    a = VM.march_union(u, v, steps=12, join_coupling="lagged")
    b = VM.march_union(u, v, steps=12, join_coupling="lagged")
    for k in VM.CROSSING_KEYS:
        if k == "t_wall":
            continue
        assert np.array_equal(np.asarray(a.trace[k]), np.asarray(b.trace[k])), k


def test_the_null_arm_removes_the_term_and_not_the_claim(state):
    """A null arm that cannot fail is not a control.  Removing J3's whole
    DEVICE would leave the shaft claiming nothing and the balance reading 0/0;
    what is withheld is the force on the fluid, while the operating point is
    still solved and the shaft still claims its power."""
    u, v = state
    m = VM.march_union(u, v, steps=8, join_coupling="lagged", null="J3")
    assert np.all(np.asarray(m.trace["power_rotor"]) == 0.0)
    assert np.all(np.asarray(m.trace["shaft_power"]) > 0.0)   # still claimed
    assert np.all(np.asarray(m.trace["q_machine"]) > 0.0)     # J2 still fed
    assert np.any(np.asarray(m.trace["power_core"]) != 0.0)   # J1 untouched
    b = VM.receiver_balances(m)["J3"]
    assert b["residual_with_the_term"] == pytest.approx(1.0)

    # J1 carries no power, so its TERM is the dependence: the null pins UA and
    # leaves the core's drag, or the control would vary two things at once
    n = VM.march_union(u, v, steps=8, join_coupling="lagged", null="J1")
    ua = np.asarray(n.trace["ua"])
    assert ua.max() == ua.min() == CL.UA_RAD          # the dependence is gone
    assert np.all(np.asarray(n.trace["power_core"]) != 0.0)   # the drag is not
    air = np.asarray(n.trace["u_core"])
    assert air.max() - air.min() > 0.0                # so the air still moves
    assert np.any(np.asarray(n.trace["power_rotor"]) != 0.0)


def test_a_tight_joins_inner_fixed_point_actually_iterates(state):
    """The first version re-read the START state, so it converged in one call
    and measured nothing -- `join_residual` was identically zero."""
    u, v = state
    m = VM.march_union(u, v, steps=6, join_coupling="tight")
    jr = np.asarray(m.join_residual)
    assert jr.shape[1] == 2
    assert jr[:, 0].mean() > 0.0
    assert jr[:, 1].mean() < 0.01 * jr[:, 0].mean()


def test_not_measured_is_None_and_never_a_default_False(state):
    """W208's discipline: a clause with no subject reports its own state."""
    u, v = state
    m = VM.march_union(u, v, steps=8, join_coupling="lagged")
    b = VM.receiver_balances(m)
    assert b["J2"]["coolant steps taken"] == 0
    assert b["J2"]["block first law, relative"] is None
    assert "not_measured_because" in b["J2"]


# ==========================================================================
# 6. the gate is the page's gate
# ==========================================================================


def _page(path):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} is not on disk")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_every_pre_registered_threshold_appears_in_the_case_study():
    """**A criterion written in prose and evaluated in code drifts, and Tier 48
    measured that it drifts permissive.**  Every number the code judges by has
    to be findable in the page that states the gate."""
    text = _page(CASE)
    for key, clause in VM.GATE.items():
        assert key in text, key
        for field, value in clause.items():
            if not isinstance(value, float):
                continue
            rendered = ("%g" % value)
            assert rendered in text or ("%s" % value) in text, (key, field, value)


def test_the_prediction_is_recorded_in_the_case_study():
    text = _page(CASE)
    assert "COMPOUND" in text and "ADD" in text
    assert "predict" in text.lower()


def test_the_scale_page_carries_the_declaration_it_is_about():
    text = _page(SCALE_PAGE)
    assert "0.50" in text and "50" in text
    assert "400" in text and "16" in text
    assert "31.152258" in text or "31.15" in text


# ==========================================================================
# 7. the artifact, and the clauses re-evaluated from the raw numbers
# ==========================================================================


def test_the_artifact_reports_the_decisions_with_their_controls(art):
    d = art["decisions"]
    assert d["W199_sizing"]["same_induction_as_the_control"] is True
    assert d["W199_sizing"]["same_demand_over_supply"] is True
    assert d["W199_sizing"]["control_reproduces_Tier_46_omega"] is True
    assert d["W199_sizing"]["shaft_power_ratio_sized_over_control"] == pytest.approx(0.5)
    assert d["W194_multirate_seams"]["disjoint"]["n_seams"] == 0
    assert d["W194_multirate_seams"]["joined"]["n_seams"] == 5
    for row in d["W94_body_force"]["devices"].values():
        assert row["exact_to"] == 0.0


def test_the_gate_verdicts_are_what_the_raw_numbers_say(art):
    """Re-evaluate every clause here from the numbers the run recorded, against
    the thresholds in `vehicle_march.GATE`, so the driver's own judgement is
    checked rather than trusted."""
    g = art.get("gate_stage")
    if g is None:
        pytest.skip("stage 'gate' has not run")
    G = VM.GATE
    g1 = g["G1"]
    want = (g1["with_the_term"] <= G["G1"]["tol_with"]
            and g1["mis_attributed_residual"] >= G["G1"]["tol_without"]
            and g1["null_J3_residual"] >= G["G1"]["tol_without"])
    assert g1["verdict"] == ("pass" if want else "fail")
    g3 = g["G3"]
    want3 = (g3["tracking_residual"] <= G["G3"]["tol_with"]
             and g3["null_UA_is_exactly_constant"]
             and g3["null_air_moved_by"] > 0.0)
    assert g3["verdict"] == ("pass" if want3 else "fail")
    s = g["G5"]["the_well_posed_form"]
    if s["sum_of_the_single_join_errors"] <= 0.0:
        assert s["verdict"] == "not measured"
    elif (g["G5"]["e_all_lagged"]
          > G["G5"]["compound_factor"] * s["sum_of_the_single_join_errors"]):
        assert s["verdict"] == "COMPOUND"
    elif s["gap"] <= G["G5"]["add_tol"] * s["sum_of_the_single_join_errors"]:
        assert s["verdict"] == "ADD"
    else:
        assert s["verdict"] == "NEITHER"
    # the pre-registered three-join form is REPORTED, not deleted
    assert "as_pre_registered_over_three_joins" in g["G5"]
    assert g["G5"]["as_pre_registered_over_three_joins"]["has_no_subject_because"]


def test_the_repeat_floor_is_quoted_beside_every_difference(art):
    g = art.get("gate_stage")
    if g is None:
        pytest.skip("stage 'gate' has not run")
    assert g["G6"]["bitwise_in_every_crossing_quantity"] is True
    assert "repeat_floor" in g["G4"]
    assert set(g["G4"]["errors"]) >= {"J1_lagged", "J3_lagged", "J2_lagged",
                                      "all_lagged"}
    # a bitwise floor makes "10x the floor" vacuous; the clause says so rather
    # than passing quietly, and reports the floor that does bind
    if g["G4"]["repeat_floor"] == 0.0:
        assert g["G4"]["the_clause_as_written_is_vacuous_because_the_floor_is_zero"]
        assert g["G4"]["the_floor_that_binds_instead"]["norm"] > 0.0


def test_the_union_does_march_and_all_three_clocks_tick(art):
    g = art.get("gate_stage")
    if g is None:
        pytest.skip("stage 'gate' has not run")
    ref = g["arms"]["referent"]
    assert ref["coolant_steps"] == 3
    assert g["horizon"] == 1200


def test_the_slow_half_cannot_be_reached_on_the_fluids_clock(art):
    s = art.get("slow")
    if s is None:
        pytest.skip("stage 'slow' has not run")
    assert s["block_thermal_time_constant_s"] > 1.0
    assert s["fluid_macro_steps_to_span_the_block_s_transient"] > 1.0e4
