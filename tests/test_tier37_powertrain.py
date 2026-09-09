"""Tier 37 -- CS-14, a drivetrain on the wake array's open shaft.

`atlas/cases/powertrain.py`.  Rung 7: the first graph to carry `ELEC`, the first
cross-domain energy balance that is not an identity of its own solve, and
W163's second directed cycle.  Five groups:

  * **the port that was open** -- the shaft seam is `wake_array`'s own
    declaration, connected, and the note that said "unconnected" is gone;
  * **the balance** -- it closes, and it FAILS when the one physical fact it
    rests on is broken, which is the difference between a gate and a tautology;
  * **the operating point** -- a real torque balance, solved on the donor's own
    curve, with two `validity` predicates that decline outside it;
  * **W163 on a second graph** -- the compiler still cannot tell the circuit
    from the chain, now on completely different physics;
  * **what the compile said**, recorded as it comes, including three findings
    about `wake_array`'s rotor that only connecting its shaft could surface.

The rotor needs the build repo's windfarm package, so every test that builds one
is skipped without it.  The circuit elements need nothing.
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import compile_scheme                                   # noqa: E402
from atlas.cases import powertrain as P                            # noqa: E402
from atlas.cases import wake_array as WA                           # noqa: E402
from atlas.ports import PortType                                   # noqa: E402
from atlas.verdict import Verdict                                  # noqa: E402


def _have_expert() -> bool:
    try:
        WA.load_reference()
    except Exception:                                              # noqa: BLE001
        return False
    return True


needs_expert = pytest.mark.skipif(
    not _have_expert(),
    reason="the windfarm reference package is not on ATLAS_BUILD_REPO")


# ---------------------------------------------------------------------------
# 1. the port that was open
# ---------------------------------------------------------------------------


@needs_expert
def test_the_shaft_port_is_wake_arrays_own_and_is_now_connected():
    """**The declaration is that module's and is not re-derived here**, which is
    the point: the open port this case study connects has to be the same port
    `wake_array` declares, or connecting it proves nothing.

    Everything about it is unchanged except the direction and the note -- and
    the note has to change, because the sentence it carried was
    *"unconnected: no drivetrain"* and that is what rung 7 exists to make false.
    """
    disk = P._rotor()
    mine = P.rotor_capabilities(disk)
    theirs = WA.rotor_capabilities(disk)

    a = mine.port("shaft:ROT")
    b = theirs.port("shaft:ROT")
    assert a.port_type is b.port_type is PortType.ROT
    assert a.effective_resolution == b.effective_resolution == 1
    assert a.response_half is b.response_half
    assert dict(a.nondim) == dict(b.nondim)

    #: the old note ASSERTED the port was unconnected; the new one quotes that
    #: sentence in order to say it is now false, so the test is on what the note
    #: asserts rather than on which words appear in it
    assert b.note.startswith("unconnected")
    assert a.note.startswith("CONNECTED, as of CS-14")
    assert "what rung 7 exists to make false" in a.note

    #: and the MECH ports are untouched -- this case study connects one port
    for name in ("up:MECH", "down:MECH"):
        assert mine.port(name).note == theirs.port(name).note

    #: the seam exists and it is the one the graph is about
    conns = P.connections()
    shaft = [c for c in conns if c.seam_id == "shaft"]
    assert len(shaft) == 1
    assert shaft[0].a == ("ROTOR", "shaft:ROT")
    assert shaft[0].port_type is PortType.ROT


def test_ELEC_is_carried_for_the_first_time_with_a_complete_scale_set():
    """The fifth port type, exercised. `check_scales` validates the power
    identity from the port list alone, so a scale set that does not close is a
    declaration error and not a run-time surprise."""
    e, f, p = (P.ELEC_SCALES["potential"], P.ELEC_SCALES["current_density"],
               P.ELEC_SCALES["power_area"])
    assert abs(e * f / p - 1.0) < 1e-12
    #: the flow is a current DENSITY, so a lumped terminal needs an area
    assert P.A_TERM > 0.0
    assert abs(f * P.A_TERM - P.I_REF) < 1e-12


@needs_expert
def test_every_declared_port_type_in_this_graph_is_one_of_the_closed_five():
    g, _e = P.build()
    kinds = {p.port_type for a in g.agents for p in a.capabilities.ports}
    assert PortType.ELEC in kinds and PortType.ROT in kinds
    assert kinds <= set(PortType)


# ---------------------------------------------------------------------------
# 2. the balance, and the fact it rests on
# ---------------------------------------------------------------------------


@needs_expert
def test_the_cross_domain_energy_balance_closes():
    """``T omega = sum I^2 R + V_oc I``, with the left side from the build
    repo's own disk and the right from Kirchhoff's laws."""
    r = P.CircuitSolve().solve()
    assert r.rotor_valid and r.mgu_valid, r
    assert r.residual < 1e-12, r.residual
    assert r.kvl_residual < 1e-12, r.kvl_residual
    assert r.torque_residual < 1e-12, r.torque_residual

    #: and it is not a balance between zeros
    assert r.p_mech > 0.1
    assert sum(r.dissipation.values()) > 0.0 and r.p_stored > 0.0
    #: every element is in the sum
    assert set(r.dissipation) == set(P.CIRCUIT_ORDER)
    assert abs(r.p_mech - (sum(r.dissipation.values()) + r.p_stored)) < 1e-12


@needs_expert
def test_the_balance_FAILS_when_ke_and_kt_are_pulled_apart():
    """**This is what makes it a gate rather than a tautology.**

    ``T omega = V_emf I`` holds because ``k_e = k_t`` -- both are the same
    air-gap flux linkage seen from the two sides. It is a theorem about the
    machine, not a definition anybody chose, and a theorem that cannot be
    broken in the code is a theorem the code is not resting on. Pull the two
    constants apart and the bridge residual grows in proportion.
    """
    base = P.CircuitSolve().solve()
    assert base.bridge_residual < 1e-12

    #: **The perturbation has to keep the graph inside its own envelope**, which
    #: is why the ratios are below one. Raising `k_e` raises the loop current,
    #: which demands more torque than `disk.py` delivers at its clamp, so the
    #: operating point stops existing and the residual then mixes two effects.
    #: Lowering it keeps the point valid and isolates the one under test.
    for ratio in (0.95, 0.90):
        els = P.make_elements()
        els["MGU"].k_e = P.K_T * ratio          # k_t left alone
        r = P.CircuitSolve(elements=els).solve()
        assert r.rotor_valid, (ratio, r)
        #: p_electrical = k_e w I and p_mech = k_t I w at the balance, so the
        #: residual is EXACTLY one minus the ratio -- measured, not asserted by
        #: construction, and it comes out to machine precision
        assert abs(r.bridge_residual - (1.0 - ratio)) < 1e-9, (ratio, r)

    #: and pushing it further leaves the envelope, which the predicate reports
    els = P.make_elements()
    els["MGU"].k_e = P.K_T * 0.85
    far = P.CircuitSolve(elements=els).solve()
    assert not far.rotor_valid and far.current < 0.0


@needs_expert
def test_the_two_routes_to_the_loop_current_agree():
    """The positive control on the circuit solve: a closed form COMPOSED from
    the elements' own coefficients, and a relaxation that has to find it."""
    r = P.CircuitSolve().solve()
    assert r.current_gap < 1e-12, r.current_gap
    assert r.iterations > 1, "a control that converges in one step is not one"


def test_a_loop_with_no_resistance_RAISES_rather_than_returning_a_number():
    """An ideal source across an ideal conductor has no operating point, and
    that is a statement about the circuit rather than a solver failure."""
    els = P.make_elements()
    for e in els.values():
        e.resistance = 0.0
    s = P.CircuitSolve.__new__(P.CircuitSolve)
    s.elements = els
    with pytest.raises(RuntimeError, match="no resistance"):
        s._closed_form(15.0)


# ---------------------------------------------------------------------------
# 3. the operating point, and two predicates that decline
# ---------------------------------------------------------------------------


@needs_expert
def test_the_operating_point_is_SOLVED_and_is_not_the_disks_reference():
    """A search that returns the number it started from has not been tested.

    `V_OC` is chosen so the equilibrium is away from `disk.py`'s own reference
    induction of 1/3 -- the module says so and this pins it.
    """
    r = P.CircuitSolve().solve()
    assert P.A_MIN < r.induction < P.A_MAX
    assert abs(r.induction - 1.0 / 3.0) > 0.02, (
        "the operating point has landed on the disk's reference induction, so "
        "the bisection is returning its own starting point")
    #: and it really is the root: the donor's torque there is the circuit's
    solver = P.CircuitSolve()
    assert abs(solver.torque_at(r.induction) - r.torque_machine) < 1e-12


@needs_expert
def test_the_rotor_declines_when_no_operating_point_exists_in_the_clamp():
    """`disk.py` clamps the induction at 0.4 -- momentum theory breaks down in
    the turbulent-wake state and the donor refuses to extrapolate. Demand more
    torque than that clamp can deliver and there is no operating point, which
    the result reports rather than returning the nearest edge as a solution."""
    els = P.make_elements()
    els["BATT"].v_oc = 1.20          # a bigger voltage margin, so more current
    r = P.CircuitSolve(elements=els).solve()
    assert not r.rotor_valid
    assert r.torque_machine > P.CircuitSolve().torque_at(P.A_MAX)


@needs_expert
def test_the_machine_declines_below_the_speed_at_which_it_can_generate():
    """**The first `validity` predicate in this package to catch a real defect.**

    It caught a units error across a seam on the first build of this module: SI
    machine constants against a nondimensional shaft gave ``k_e omega = 0.19``
    against a 3.6 V battery, so the machine motored where the graph declared
    generation. The predicate is kept sharp here.
    """
    mgu = P.MachineAgent()
    mgu.omega = P.V_OC / P.K_E * 1.01
    assert mgu.validity()
    mgu.omega = P.V_OC / P.K_E * 0.99
    assert not mgu.validity()
    #: and the graph's own operating speed is comfortably inside it
    assert P.CircuitSolve().solve().mgu_valid


# ---------------------------------------------------------------------------
# 4. W163 on a second graph
# ---------------------------------------------------------------------------


def test_the_elec_seams_form_one_directed_cycle_through_every_element():
    advec = [c for c in P.connections() if c.port_type is PortType.ELEC]
    order = P.CIRCUIT_ORDER
    assert len(advec) == len(order) == 4
    nxt = {c.a[0]: c.b[0] for c in advec}
    node = order[0]
    seen = []
    for _ in range(len(order)):
        seen.append(node)
        node = nxt[node]
    assert node == order[0]
    assert sorted(seen) == sorted(order)


def test_dropping_the_return_conductor_turns_the_circuit_into_a_chain():
    closed = {c.seam_id for c in P.connections(close_loop=True)}
    chain = {c.seam_id for c in P.connections(close_loop=False)}
    assert closed - chain == {"inv_mgu"}


@needs_expert
def test_the_compiler_reports_NOTHING_different_about_the_cycle_HERE_EITHER():
    """**W163's second graph, which is what that row said it was waiting for.**

    CS-13 found that compiling a circuit and compiling the same agents as a
    chain gives the same verdict, the same rule set and the same per-agent
    decisions. This is the same finding on completely different physics -- a DC
    loop rather than a coolant loop, ELEC rather than ADVEC, a real actuator
    disk rather than a conduction solver. One graph is an anecdote; two on
    unrelated physics is the statement that no layer looks at loop topology.
    """
    gc, _ = P.build(close_loop=True)
    go, _ = P.build(close_loop=False)
    rc, ro = compile_scheme(gc), compile_scheme(go)
    assert rc.verdict is ro.verdict
    assert ({d.rule for d in rc.decisions if d.verdict.value != "admit"}
            == {d.rule for d in ro.decisions if d.verdict.value != "admit"})

    def rows(rec):
        return {(d.layer, d.rule, d.subject, d.verdict.value)
                for d in rec.decisions}
    extra = rows(rc.decisions) - rows(ro.decisions)
    assert extra and all("inv_mgu" in str(x[2]) for x in extra), extra


# ---------------------------------------------------------------------------
# 5. what the compile said
# ---------------------------------------------------------------------------


@needs_expert
def test_the_graph_compiles_with_no_refusal():
    g, _e = P.build()
    r = compile_scheme(g)
    assert r.verdict is Verdict.ADMIT_UNCERTIFIED, r.verdict
    assert not r.decisions.refusals


@needs_expert
def test_the_shaft_seam_is_ONE_SIDED_and_the_compile_says_so():
    """**A finding only connecting the port could surface.**

    `wake_array.RotorDisk.respond` ignores the trace on `shaft:ROT` entirely --
    an actuator disk at a fixed induction and a fixed inflow delivers a torque
    that does not depend on the speed you ask of it, because its own omega is
    tied to the inflow rather than to the load. So the rotor's block is
    identically zero and the machine carries the whole seam operator. That is
    W97's class (a seam one-sided by construction, whose substitution
    certificate is therefore blind) arriving on a ROT port, and `L4/block-share`
    reports it.
    """
    disk = P._rotor()
    w0 = disk.respond("shaft:ROT", np.zeros(1))
    w1 = disk.respond("shaft:ROT", np.full(1, 30.0))
    assert float(w0[0]) == float(w1[0]), (
        "the rotor's shaft response has become speed-dependent, which would "
        "make this seam two-sided and this test's subject wrong")

    g, _e = P.build()
    left = {(d.rule, d.subject) for d in compile_scheme(g).decisions
            if d.verdict.value != "admit"}
    assert ("block-share", "shaft") in left


@needs_expert
def test_the_two_sides_of_the_shaft_linearize_about_DIFFERENT_speeds():
    """W74 on a third seam, and it is `wake_array`'s declaration rather than
    this module's: `RotorDisk.probe_base` returns zeros for `shaft:ROT` -- a
    STALLED turbine -- while the machine declares the operating speed.

    **Left as it is rather than repaired**, because the rotor's declaration
    belongs to `wake_array` and because the instrument that would settle it
    exists: the rotor's response is constant in the trace, so
    `probe.base_sensitivity` would clear the decertification outright. That it
    is not wired into this compile is the finding.
    """
    disk = P._rotor()
    assert float(np.mean(disk.probe_base("shaft:ROT"))) == 0.0
    assert P.MachineAgent().base_for(
        P.rotor_capabilities(disk).port("shaft:ROT"))[0] != 0.0

    g, _e = P.build()
    left = {(d.rule, d.subject) for d in compile_scheme(g).decisions
            if d.verdict.value != "admit"}
    assert ("probe-base", "shaft") in left


@needs_expert
def test_W160_reproduces_here_on_a_THIRD_graph():
    """`L2/R10`'s undeclared-pressure-solve branch fires on the ROTOR: an
    actuator disk declares the flow's `governing_family` -- correctly, since an
    algebraic closure inside a continuum problem is not a different continuum
    problem -- and `stencil_radius = 0`, so it has no field and no pressure
    solve to hide.

    The front wing's suspension and CS-13's four coolant legs are the other two.
    A row that fires on three unrelated graphs is about a class.
    """
    g, _e = P.build()
    r10 = [d for d in compile_scheme(g).decisions if d.rule == "R10"]
    assert r10 and "ROTOR" in r10[0].subject
    caps = g.agent("ROTOR").capabilities
    assert int(caps.stencil_radius) == 0
    assert caps.elliptic_subsolve.value == "none"


@needs_expert
def test_the_ELEC_seams_are_PASSIVE_and_that_was_not_free():
    """The first version flipped the sign on the `lo` terminal, which made one
    block negative and left `L4/E7/passivity` reporting 1.5e-5 on `mgu_bus` --
    W138's shape on a third port type, caught by the rule this session taught to
    see it. With each terminal reporting in its own element's sense, the
    assembled operator at a node is the SERIES resistance of the two elements
    meeting there, which is positive because a resistive network is passive.
    """
    g, _e = P.build()
    r = compile_scheme(g)
    for seam in ("mgu_bus", "bus_batt", "batt_inv", "inv_mgu"):
        op = r.seam_operators[seam]
        assert op.passivity_defect == 0.0, (seam, op.passivity_defect)
        #: and it is the series resistance, positive on both blocks
        for blk in op.blocks.values():
            assert float(np.linalg.eigvalsh(
                0.5 * (blk.S + blk.S.T))[0]) > 0.0, seam
    #: no `effort_normal` is declared on them, because this really is the
    #: own-outward convention rather than a co-oriented pair
    for c in P.connections():
        if c.port_type is PortType.ELEC:
            assert c.effort_normal == ""


@needs_expert
def test_the_quasi_steady_elements_declare_the_macro_step_as_their_own():
    """CS-13 learnt this from `L7/R9` refusing a clock ratio its legs did not
    have. A DC circuit with no inductance settles in microseconds against a
    rotor's tenths of a second, so the element is quasi-steady -- declared up
    front here rather than after the rule says so."""
    g, _e = P.build()
    for name in P.CIRCUIT_ORDER:
        assert g.agent(name).capabilities.dt_native == P.MACRO_DT
    r = compile_scheme(g)
    assert not [d for d in r.decisions.refusals if d.rule == "R9"]
