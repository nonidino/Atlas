"""Tier 36 -- CS-13, a closed coolant circuit.  `atlas/cases/cooling_loop.py`.

Rung 6 of the ladder, and the first graph in this package whose ports form a
**directed cycle**.  Five groups:

  * **the topology** -- the circuit closes, every leg has an upstream and a
    downstream, and the open-chain control differs from it by exactly one seam;
  * **the gates** -- the block's own first law (which can fail), the loop
    balance (which is an identity of the fixed point and is asserted as such),
    and the two independent routes to the fixed point;
  * **path dependence** -- the measurement this case study exists for: one
    Gauss-Seidel sweep per step is order-dependent by 1.5 K, the fixed point is
    not, and the spread contracts at a rate the circuit's own gain sets;
  * **what the compiler says** -- recorded as it comes, including the two
    things it does NOT say;
  * **the declaration** -- the four errors the first compile found, pinned so
    they cannot come back.

The block agent needs the build repo, so every test that builds one is skipped
without it.  The lumped legs need nothing and their tests always run.
"""

from __future__ import annotations

import math
import os
import sys

# Before numpy: the build repo pulls torch in and torch's OpenMP beside numpy's
# MKL aborts the interpreter inside a dense solve.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import compile_scheme                                   # noqa: E402
from atlas.cases import cooling_loop as C                          # noqa: E402
from atlas.ports import PortType                                   # noqa: E402
from atlas.verdict import Verdict                                  # noqa: E402


def _have_expert() -> bool:
    try:
        C.load_solvers()
    except Exception:                                              # noqa: BLE001
        return False
    return True


needs_expert = pytest.mark.skipif(
    not _have_expert(),
    reason="thermostruct2d is not on ATLAS_BUILD_REPO")


# ---------------------------------------------------------------------------
# 1. the topology: it is a circuit
# ---------------------------------------------------------------------------


def test_the_advec_seams_form_one_directed_cycle_through_every_leg():
    """A circuit, not a chain: every leg has exactly one upstream and one
    downstream, and following them returns to where it started."""
    order = C.loop_order(4)
    conns = C.connections(4)
    advec = [c for c in conns if c.port_type is PortType.ADVEC]
    assert len(advec) == len(order) == 4

    nxt = {c.a[0]: c.b[0] for c in advec}
    prv = {c.b[0]: c.a[0] for c in advec}
    assert set(nxt) == set(prv) == set(order)

    seen, node = [], order[0]
    for _ in range(len(order)):
        seen.append(node)
        node = nxt[node]
    assert node == order[0], "the loop does not close"
    assert sorted(seen) == sorted(order), "the loop misses a leg"


def test_dropping_ONE_seam_turns_the_circuit_into_a_chain():
    """The control has to differ by the return edge and by nothing else."""
    closed = {c.seam_id for c in C.connections(4, close_loop=True)}
    chain = {c.seam_id for c in C.connections(4, close_loop=False)}
    assert closed - chain == {"cold_pass"}
    assert chain < closed


def test_the_seam_list_is_DERIVED_from_the_loop_order():
    """`connections` and `LoopSolve` must not be able to disagree about the
    circuit. Reordering the legs has to reorder the seams."""
    for n in (3, 4):
        order = C.loop_order(n)
        advec = [c for c in C.connections(n) if c.port_type is PortType.ADVEC]
        pairs = [(c.a[0], c.b[0]) for c in advec]
        expect = [(order[i], order[(i + 1) % n]) for i in range(n)]
        assert pairs == expect


def test_the_sweep_study_reads_the_circuit_off_the_CONNECTIONS():
    """Not off a hard-coded dict: a study that remembers the topology rather
    than reading it is measuring a topology nobody declared."""
    up = C.SweepStudy().upstream_of()
    order = C.loop_order(4)
    assert up == {order[(i + 1) % 4]: order[i] for i in range(4)}


# ---------------------------------------------------------------------------
# 2. the gates
# ---------------------------------------------------------------------------


@needs_expert
def test_the_blocks_own_FIRST_LAW_closes_across_one_step():
    """**The gate that can fail, and the only one here that is about physics.**

    ``d/dt integral rho cp T dV = Q_outer - Q_wall`` across one `step_thermal`,
    with both fluxes read through the same face machinery the loop uses. Get the
    wetted face wrong, drop the out-of-plane width, or flip a sign and this is
    order one. Nothing in the loop algebra makes it true.
    """
    b = C.BlockAgent()
    bal = b.energy_balance(np.full(C.N_SEAM, C.T_COOLANT_0))
    #: **The floor is the donor's own iterative solve, and it was measured
    #: rather than assumed.** `step_thermal` finishes with Jacobi-preconditioned
    #: CG, so the balance closes to that solve's tolerance and not to round-off.
    #: Swept over four decades of dt the relative residual runs 8.1e-14, 3.3e-8,
    #: 4.1e-12, 5.3e-11 -- NON-MONOTONE, which is the signature of a solver
    #: tolerance and not of a discretisation error, since the latter would have
    #: a trend. 1e-6 is two decades above the worst of those and seven below
    #: anything a modelling defect would produce.
    assert bal["relative"] < 1e-6, bal
    #: and it is not trivially satisfied by everything being zero
    assert abs(bal["q_outer"]) > 100.0 and abs(bal["q_wall"]) > 10.0


@needs_expert
def test_the_first_law_is_SENSITIVE_to_the_things_it_is_checking():
    """A gate that passes whatever you do is not a gate -- and this one has a
    blind spot, which is worth pinning as precisely as its teeth are.

    **What it catches, decisively**: a sign error on the wall flux, and the
    out-of-plane width the docstring warns about. Both are the mistakes that
    cancel out of every ratio in this module and do not cancel out of a first
    law, and each moves the residual by more than the fluxes themselves.

    **What it does NOT catch at this state**: reading the OUTER face where the
    wetted one was meant. The block is nearly uniform through its thickness at
    the release state, so the two face temperatures are close and the swap is
    worth only about 1.7% -- inside no threshold worth setting. That is a real
    limitation of taking the gate at one state, it is measured rather than
    guessed, and it is stated here so nobody reads this test as covering it.
    """
    b = C.BlockAgent()
    Tc = np.full(C.N_SEAM, C.T_COOLANT_0)
    good = b.energy_balance(Tc)
    scale = abs(good["q_outer"])
    #: the discrimination is measured against the residual the gate PASSES at,
    #: which is the only normalisation that says how much room the gate has
    floor = abs(good["residual"])

    #: a SIGN error on the wall flux
    flipped = abs(good["stored_rate"] - (good["q_outer"] + good["q_wall"]))
    assert flipped > 1.0e6 * floor, "a sign error on q_wall must break it"
    assert flipped / scale > 0.1

    #: the out-of-plane WIDTH dropped from the stored term
    no_width = abs(good["stored_rate"] / C.WIDTH
                   - (good["q_outer"] - good["q_wall"]))
    assert no_width > 1.0e6 * floor, "dropping WIDTH must break it"
    assert no_width / scale > 1.0

    #: and the blind spot, measured
    T1 = b._ts.step_thermal(b._T, b.dt, b.h_wall, Tc, C.H_OUTER, b.t_source,
                            radiate=False, T_inf=C.T_AMB)
    q_wrong_face = float(np.sum(b.h_wall * (b._face_T(T1, "outer") - Tc))
                         * C.H_SEAM * C.WIDTH)
    wrong_face = abs(good["stored_rate"] - (good["q_outer"] - q_wrong_face))
    assert wrong_face / scale < 0.05, (
        "if the wrong-face swap has become large, the block is no longer "
        "near-uniform and this test's stated blind spot needs re-measuring")


@needs_expert
def test_the_loop_balance_closes_and_it_is_an_IDENTITY_of_the_fixed_point():
    """The gate the ladder asks for, with its status stated rather than implied.

    ``Q_block + W_pump = sum of the legs' rejections`` holds to round-off. It is
    a consistency check on the arithmetic -- a sign error in any leg's
    `heat_out` breaks it -- and it is NOT independent evidence about the
    physics, because substituting the legs' definitions turns it into
    ``T_return = T_return``. Both halves of that are asserted: the balance
    closes, and perturbing a leg's rejection breaks it.
    """
    r = C.LoopSolve().solve(steps=600)
    assert r.residual < 1e-10, r.residual
    q_in = r.q_block + r.w_pump
    assert q_in > 100.0, "a balance between two zeros is not a balance"
    #: every leg that rejects heat is in the sum, and the radiator dominates
    assert set(r.q_rejected) == {"HOT", "RAD", "COLD"}
    assert r.q_rejected["RAD"] > 10.0 * max(r.q_rejected["HOT"],
                                            r.q_rejected["COLD"])
    #: and it is sensitive: drop one leg's rejection and it stops closing
    partial = sum(v for k, v in r.q_rejected.items() if k != "HOT")
    assert abs(q_in - partial) / q_in > 1e-3


@needs_expert
def test_the_two_routes_to_the_fixed_point_agree():
    """The positive control on the solve itself.

    An iteration that converged to the wrong thing and a closed form that was
    never checked are the same artifact. The closed form is COMPOSED from the
    legs' own `coeffs`, so it cannot drift from the legs it is the closed form
    of, and the iteration is the composition layer's own sweep run to
    convergence.
    """
    r = C.LoopSolve().solve(steps=600)
    assert r.fixed_point_gap < 1e-12, r.fixed_point_gap
    assert abs(r.t_return - r.closed_form) < 1e-9


def test_a_loop_with_no_dissipative_leg_RAISES_rather_than_returning_a_number():
    """Unit gain means no isolated fixed point, and that is a statement about
    the circuit rather than a solver failure. A closed form that silently
    divided by zero would be the silent-wrongness class."""
    s = C.LoopSolve.__new__(C.LoopSolve)
    s.order = ("PASS", "COLD")
    s.legs = {"PASS": C.CoolantLeg(), "COLD": C.PumpLeg(agent_id="COLD", ua=0.0)}
    with pytest.raises(RuntimeError, match="unit gain"):
        s._closed_form(500.0)


# ---------------------------------------------------------------------------
# 3. path dependence -- the measurement CS-13 exists for
# ---------------------------------------------------------------------------


def test_ONE_sweep_per_step_is_ORDER_DEPENDENT():
    """**The finding.** A directed cycle has no first agent, so a composition
    layer that sweeps it once per macro-step has made a modelling choice nothing
    in the declaration licenses -- and the choice is worth real kelvin."""
    st = C.SweepStudy()
    one = st.spread(1)
    assert one["n_distinct"] > 1, "a cycle whose sweeps all agree is not a cycle"
    assert one["range"] > 0.1, one["range"]
    assert one["relative_range"] > 1e-4


def test_the_FIXED_POINT_is_order_independent():
    """And the cure is not an ordering rule: it is solving the loop instead of
    sweeping it. Every order iterated to convergence lands on one state."""
    c = C.SweepStudy().converged()
    assert c["relative_range"] < 1e-11, c["relative_range"]
    assert len(c["orders"]) == 8                      # 4 rotations x 2 senses


def test_the_spread_contracts_at_a_rate_the_CIRCUIT_sets():
    """The decay is not a rate of the scheme's choosing -- and the sharp form
    of that claim did not survive a second leg count.

    **What was claimed and is withdrawn.** On the THREE-leg circuit the
    per-sweep contraction matched ``sqrt(loop gain)`` to seven figures
    (0.9618577 measured against 0.9618576), which read as an exact
    identification. On FOUR legs it does not: the measured rate is 0.97405 over
    256-512 sweeps and 0.97381 over 512-1024, against ``gain^(1/3)`` =
    0.973926 -- agreeing to about 1e-4 and DRIFTING, and the spread underflows
    to bitwise zero by 1536 sweeps before any asymptote is reached. So the
    seven-figure match was true as far as it was marched and is not a law.

    **What survives, and is what this test asserts.** The spread contracts
    geometrically; the rate is strictly inside ``(gain, 1)``; and it sits within
    1e-3 of ``gain^(1/(n-1))``. That is enough for the claim the case study
    actually needs -- the path dependence decays at a rate the circuit's own
    dissipation sets, not one the scheme chose -- and it does not assert an
    identity two leg counts disagree about.
    """
    k = C.SweepStudy().contraction()
    g = k["loop_gain"]
    assert g < k["per_sweep"] < 1.0, k
    n = len(C.loop_order(4))
    assert abs(k["per_sweep"] / g ** (1.0 / (n - 1)) - 1.0) < 1e-3, k

    #: and the three-leg circuit, where the exponent is 1/2
    k3 = C.SweepStudy(n_legs=3).contraction()
    g3 = k3["loop_gain"]
    assert g3 < k3["per_sweep"] < 1.0, k3
    assert abs(k3["per_sweep"] / math.sqrt(g3) - 1.0) < 1e-3, k3


def test_the_spread_does_not_vanish_at_a_practical_number_of_sweeps():
    """A composition layer does not iterate a loop 400 times per macro-step, so
    the number that matters is the spread at a handful -- and it is still there
    after eight."""
    st = C.SweepStudy()
    assert st.spread(8)["range"] > 0.1
    assert st.spread(1)["range"] > 0.1


# ---------------------------------------------------------------------------
# 4. what the compiler says -- and the two things it does not
# ---------------------------------------------------------------------------


@needs_expert
def test_the_circuit_compiles_and_the_verdict_is_recorded_as_it_comes():
    g, _e = C.build()
    r = compile_scheme(g)
    assert r.verdict is Verdict.ADMIT_UNCERTIFIED, r.verdict
    left = {d.rule for d in r.decisions if d.verdict.value != "admit"}
    #: E3 fails at the wall because it is a genuine multiphysics seam; R10 is
    #: W160 on four more agents; W56 is the empty ledger this graph declares.
    assert {"E3", "R10", "W56"} <= left, left
    assert not r.decisions.refusals


@needs_expert
def test_the_compiler_reports_NOTHING_different_about_the_cycle():
    """**The named gap, and it is this case study's main result about Atlas.**

    Compile the circuit and compile the same agents as an open chain. The two
    differ by the seam that closes the loop -- and by every rule the compiler
    applies, they are the same graph: same verdict, same rule set, same
    per-agent decisions. So nine layers of admissibility currently see a
    DIRECTED CYCLE as a list of independent seams.

    That is not a claim that a rule is missing; measuring the consequence
    (1.5 K of order dependence at one sweep per step) is what says the cycle is
    physically real. It is the statement that no layer looks.
    """
    gc, _ = C.build(close_loop=True)
    go, _ = C.build(close_loop=False)
    rc, ro = compile_scheme(gc), compile_scheme(go)
    assert rc.verdict is ro.verdict
    assert ({d.rule for d in rc.decisions if d.verdict.value != "admit"}
            == {d.rule for d in ro.decisions if d.verdict.value != "admit"})
    #: and the ONLY difference in the decision record is the extra seam's own
    #: per-seam rows
    def by_seam(rec):
        return {(d.layer, d.rule, d.subject, d.verdict.value)
                for d in rec.decisions}
    extra = by_seam(rc.decisions) - by_seam(ro.decisions)
    assert extra and all("cold_pass" in str(x[2]) for x in extra), extra


@needs_expert
def test_the_THREE_leg_circuit_is_refused_as_a_CROSS_POINT():
    """A triangle in the agent adjacency is read as a substructuring
    cross-point, and three legs joined by three DISTINCT planes share no point.

    Kept reachable rather than written up and thrown away, because a finding
    whose reproduction is a paragraph is not reproducible.
    """
    g, _e = C.build(n_legs=3)
    r = compile_scheme(g)
    assert r.verdict is Verdict.REFUSE
    ref = [d for d in r.decisions.refusals]
    assert [d.rule for d in ref] == ["I2/G1"], [d.rule for d in ref]
    assert "cross-point" in ref[0].message
    #: the triangle is what triggers it, and it really is one
    assert len(g.detected_cross_points()) == 1
    #: while the four-leg circuit has none
    g4, _ = C.build(n_legs=4)
    assert not g4.detected_cross_points()


# ---------------------------------------------------------------------------
# 5. the declaration: the four errors the first compile found
# ---------------------------------------------------------------------------


def test_the_ADVEC_scale_set_carries_a_pair_PER_PASSENGER():
    """L3/C4 refused the first version. ADVEC is a multibond: the base triple
    describes the port and every passenger needs its own conjugate pair."""
    for key in ("enthalpy", "mass_flux", "power_area",
                "h0_effort", "h0_flow", "h0_power"):
        assert key in C.ADVEC_SCALES, key
    #: and the power identity holds for both the base pair and the passenger's
    for e, f, p in (("enthalpy", "mass_flux", "power_area"),
                    ("h0_effort", "h0_flow", "h0_power")):
        assert abs(C.ADVEC_SCALES[e] * C.ADVEC_SCALES[f]
                   / C.ADVEC_SCALES[p] - 1.0) < 1e-12


@needs_expert
def test_a_LUMPED_leg_declares_ONE_interface_mode():
    """L4/null-space refused 16 modes on a one-state element, reporting all
    fifteen null directions. `PORT_SPECS[ROT]`'s note says the shape out loud:
    a lumped port is the case dim M = 1."""
    g, _e = C.build()
    for a in g.agents:
        for p in a.capabilities.ports:
            if p.port_type is PortType.ADVEC:
                assert p.effective_resolution == C.M_LUMPED == 1
            elif p.port_type is PortType.THERM:
                assert p.effective_resolution == C.M_EFF == 16


@needs_expert
def test_a_QUASI_STEADY_leg_declares_the_MACRO_step_as_its_own():
    """L7/R9 refused a 50:1 clock ratio the legs do not have. ``T_out = T_in +
    Q/(mdot cp)`` contains no time; the leg responds within the macro-step."""
    g, _e = C.build()
    for a in g.agents:
        if a.agent_id != "BLOCK":
            assert a.capabilities.dt_native == C.MACRO_DT
    assert C.RESIDENCE_TIME < C.MACRO_DT, (
        "the residence time is the scale at which quasi-steady stops holding, "
        "and it has to be below the step for the assumption to be the one made")


@needs_expert
def test_the_probe_base_belongs_to_the_PORT_and_not_to_the_expert():
    """W74, and L4/probe-base caught it: a leg's wall port linearizes about a
    temperature and its ADVEC ports about a mass flux. One base for both put
    320 K against 500 kg/(m^2 s) on the same seam."""
    leg = C.CoolantLeg()
    g, _e = C.build()
    caps = g.agent("PASS").capabilities
    therm = caps.port("wall:THERM")
    advec = caps.port("out:ADVEC")
    assert float(np.mean(leg.base_for(therm))) == C.T_COOLANT_0
    assert float(np.mean(leg.base_for(advec))) == pytest.approx(C.MASS_FLUX)
    #: and both live in V, beside the trace, not in M
    assert leg.base_for(therm).shape == (C.N_SEAM,)
    assert leg.base_for(advec).shape == (C.N_SEAM,)


@needs_expert
def test_the_block_is_the_build_repos_solver_imported_unmodified():
    """The one real expert here, and the case study is a fixture without it."""
    g, _e = C.build()
    caps = g.agent("BLOCK").capabilities
    assert "thermostruct2d" in caps.note
    assert caps.elliptic_subsolve.value == "embedded"
    assert caps.bc_channel.value == "robin"
    assert not caps.missing_fields(), caps.missing_fields()


def test_every_leg_is_AFFINE_in_its_inlet_temperature():
    """What makes the closed form exist at all, asserted rather than assumed:
    two probes fix a line, and a third has to lie on it."""
    for leg in C.make_legs(4).values():
        a, b = leg.coeffs(1000.0)
        for t in (300.0, 350.0, 400.0):
            assert abs(leg.outlet(t, 1000.0) - (a * t + b)) < 1e-9, leg.agent_id
        assert 0.0 < a <= 1.0, (leg.agent_id, a)
