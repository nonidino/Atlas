"""Tier 13 -- the conjugate pairing (W66), the substructuring criterion (W57),
the cross-point scope (W64), and E3 against a genuine family disagreement.

Following `test_tier12_topology.py`'s pattern: **the rule is tested against the
algebra wherever the algebra is what the rule is about**, and a case study is
used only where nothing else can stand in for a real object.  The two real-object
runs this tier rests on are harness scripts with persisted output --
`scripts/w57_substructuring.py` and `scripts/w67_thermal_seam.py` -- exactly as
W33's and W6's were, because a 33-column probe of a compressible solver does not
belong in a suite that runs in two seconds.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from atlas import (
    BCChannel, CaseGraph, ClaimType, Connection, Decomposition, Differentiable,
    Direction, EllipticSubsolve, ExpertCapabilities, MotionClass, PortType,
    TimeDiscretization, compile_scheme, port_decl,
)
from atlas.admissibility import check_connection
from atlas.compiler import CROSS_POINT_COUPLING_SCOPE
from atlas.ports import ResponseHalf, check_response_half, spec_for
from atlas.transfer import Prolongation
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE

sys.path.insert(0, os.path.dirname(__file__))
import substructure_model as SM        # noqa: E402

M = 8
SCALES = {"stress": 1.0, "velocity": 1.0, "power_area": 1.0}


# ===========================================================================
# W66 -- the conjugate pairing, on the algebra
# ===========================================================================


def test_w66_both_sides_declaring_the_same_half_is_admissible():
    c = check_response_half(PortType.MECH, ResponseHalf.EFFORT, ResponseHalf.EFFORT)
    assert c.ok and c.declared and c.agree
    assert c.response_half is ResponseHalf.EFFORT
    # the trace is the OTHER half, and that is the whole content of the pairing
    assert c.trace_half is ResponseHalf.FLOW


def test_w66_mixed_halves_are_refused_because_the_operator_adds_them():
    """``Lambda_M = sum_i P_i^* Lambda_i P_i`` ADDS the two sides' responses.

    An effort plus a flow is not a quantity, so this is the silent-wrongness
    class and the only case C9 can be certain about.
    """
    c = check_response_half(PortType.MECH, ResponseHalf.EFFORT, ResponseHalf.FLOW)
    assert not c.ok and c.declared and not c.agree
    assert "ADDS" in c.detail


@pytest.mark.parametrize("a,b", [(None, "effort"), ("flow", None), (None, None)])
def test_w66_an_undeclared_half_is_not_ok_but_is_not_a_refusal(a, b):
    """The pre-existing silence, made audible. It decertifies rather than
    refusing, because an absent declaration establishes nothing either way --
    the same reading `missing storage` gets at E7."""
    c = check_response_half(PortType.THERM, a, b)
    assert not c.ok
    assert not c.declared
    assert c.response_half is ResponseHalf.UNDECLARED
    assert "W66" in c.detail


def test_w66_the_declaration_picks_which_scale_nondimensionalizes_which_side():
    """What the field buys beyond the agreement check: before it, which of
    ``s_e`` and ``s_f`` scaled the probe's response was chosen silently."""
    spec = spec_for(PortType.THERM)
    flow_side = check_response_half(PortType.THERM, "flow", "flow")
    assert flow_side.scale_keys(spec) == ("entropy_flux", "temperature")
    effort_side = check_response_half(PortType.THERM, "effort", "effort")
    assert effort_side.scale_keys(spec) == ("temperature", "entropy_flux")


def test_w66_passivity_is_exactly_invariant_under_a_positive_rescale():
    """**The reason the operator cannot say which half it is in.**

    ``sym(cS) = c sym(S)`` for a scalar ``c > 0``, so every eigenvalue of the
    symmetric part keeps its sign and E7's passivity test is unchanged.  The
    THERM pseudo-bond ``q_n`` differs from the declared flow ``q_n/T`` by exactly
    such a factor (``T > 0``), which is why returning it left the verdict, the
    stamp, the null count and the passivity defect numerically identical on
    `cases/thermal_seam.py` -- measured, and moving only ``beta``, which nothing
    checks.
    """
    rng = np.random.default_rng(0)
    A = rng.normal(size=(12, 12))
    S = A @ A.T + 0.5 * np.eye(12)            # symmetric positive definite
    lo = np.linalg.eigvalsh(0.5 * (S + S.T)).min()
    for c in (1e-3, 0.5, 94.97, 900.0):
        lo_c = np.linalg.eigvalsh(0.5 * (c * S + (c * S).T)).min()
        assert lo_c == pytest.approx(c * lo, rel=1e-10)
        assert (lo_c > 0) == (lo > 0)          # the SIGN, which is what E7 reads


def test_w66_a_nondimensional_scale_set_carries_no_information_about_the_half():
    """**The reason the magnitude cannot say either**, and it is not a quirk of
    one case: the power identity is ``s_e s_f = s_P``, so a properly
    nondimensionalized port with ``s_e = s_f = 1`` has ``s_P = 1`` too, and all
    three halves are declared at the same scale.  Measured on
    `wind_farm_real`, where ``U_INF = 1``: the correct ADVEC response and the
    incorrect one differ by 5%.
    """
    from atlas.ports import check_scales
    scales = {"enthalpy": 1.0, "mass_flux": 1.0, "power_area": 1.0,
              "h0_effort": 1.0, "h0_flow": 1.0, "h0_power": 1.0}
    chk = check_scales(PortType.ADVEC, scales, ("h0",))
    assert chk.ok                       # the scale SET is perfectly valid...
    spec = spec_for(PortType.ADVEC)
    # ...and it assigns the same number to the effort, the flow and the power,
    # so no magnitude test can separate them.
    assert scales[spec.effort_scale_key] == scales[spec.flow_scale_key]
    assert scales[spec.flow_scale_key] == scales[spec.power_scale_key]


# ---------------------------------------------------------------------------
# W66 through L3, on a graph
# ---------------------------------------------------------------------------


def _caps(name, half=ResponseHalf.EFFORT, faces=("xlo", "xhi")):
    n = 8
    P = np.eye(n)[:, :M] if M <= n else np.eye(n)
    return ExpertCapabilities(
        expert_id=name,
        ports=[port_decl(name=f"{f}:MECH", port_type=PortType.MECH, geometry=f,
                         direction=Direction.BIDIRECTIONAL, nondim=dict(SCALES),
                         effective_resolution=M, motion_class=MotionClass.STATIC,
                         response_half=half,
                         prolongation=Prolongation(agent_id=name, port_name=f"{f}:MECH",
                                                   matrix=P))
               for f in faces],
        bc_channel=BCChannel.DIRICHLET, bc_time_varying=True,
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=1, substeps_per_macro_step=1,
        differentiable=Differentiable.NONE, dt_native=1e-2,
        governing_family="toy", claim_types=frozenset({ClaimType.TRAJECTORY}),
        boundary_response=lambda p, t: np.asarray(t, float) * 0.5,
    )


def _two_agent_graph(half_a, half_b):
    from atlas.graph import Agent
    a, b = _caps("a", half_a), _caps("b", half_b)
    return CaseGraph(
        name="pairing", agents=[Agent("a", a), Agent("b", b)],
        connections=[Connection(seam_id="s", a=("a", "xhi:MECH"), b=("b", "xlo:MECH"),
                                port_type=PortType.MECH, derive_space=True,
                                geometrically_coincident=True, expected_null_dim=None)],
        decomposition=Decomposition.OVERLAPPING, overlap=1.0, overlap_cells=4,
        macro_dt=1e-2,
    )


def test_w66_l3_refuses_a_seam_whose_two_sides_return_different_halves():
    g = _two_agent_graph(ResponseHalf.EFFORT, ResponseHalf.FLOW)
    res = check_connection(g, g.connections[0], None)
    c9 = [r for r in res if r.condition == "C9"]
    assert c9 and c9[0].verdict is REFUSE
    r = compile_scheme(g)
    assert r.verdict is REFUSE
    assert any(d.rule == "C9" for d in r.decisions.refusals)


def test_w66_l3_decertifies_a_seam_that_does_not_declare_the_half():
    g = _two_agent_graph(ResponseHalf.UNDECLARED, ResponseHalf.UNDECLARED)
    r = compile_scheme(g)
    hit = [d for d in r.decisions.decertifications if d.rule == "C9/W66"]
    assert hit
    assert "window_ns compiles to `admit`" in hit[0].message
    assert r.verdict is not ADMIT


def test_w66_agreeing_declarations_admit_and_name_the_trace_half():
    g = _two_agent_graph(ResponseHalf.EFFORT, ResponseHalf.EFFORT)
    r = compile_scheme(g)
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C9"]
    assert hit
    assert hit[0].evidence["response_half"] == "effort"
    assert hit[0].evidence["trace_half"] == "flow"


def test_w66_every_shipped_case_declares_the_half():
    """The W61 discipline applied to the new field: a value nobody chose is not
    a declaration, and both fixtures were getting their transmission rung from
    an omission the last time that happened."""
    from atlas.cases import rocket, wind_farm
    for build in (wind_farm.build, rocket.build):
        g = build()
        for agent in g.agents:
            for p in agent.capabilities.ports:
                assert p.response_half is not ResponseHalf.UNDECLARED, (
                    f"{agent.agent_id}/{p.name}")


# ===========================================================================
# W64 -- the cross-point scope, decided
# ===========================================================================


def test_w64_the_scope_statement_says_coupling_is_never_certified():
    assert "never certified" in CROSS_POINT_COUPLING_SCOPE
    assert "1.8%" in CROSS_POINT_COUPLING_SCOPE
    assert "MULTI-VALUEDNESS" in CROSS_POINT_COUPLING_SCOPE


def test_w64_the_l2_refusal_no_longer_rests_on_beta():
    """Section 12.2 measured all three of the old refusal's claims and they were
    wrong: beta does not collapse (1.00x), the conditioning is not degraded, and
    the smallest singular direction carries 0.0097 at the cross-point.  The
    refusal STANDS -- a cross-point does need treatment -- and now cites the
    quantity that was actually measured.
    """
    from atlas.cases import wind_farm
    g = wind_farm.build(boundary_capable=True, declare_transfer=True)
    r = compile_scheme(g)
    hit = [d for d in r.decisions.refusals if d.rule == "I2/G1"]
    assert hit, "the cross-point refusal should still fire"
    d = hit[0]
    assert d.quantity == "cross_point_multivaluedness"
    assert "MULTI-VALUED" in d.message
    assert "does NOT rest on beta" in d.message
    assert d.evidence["measured_beta_collapse"] == 1.00
    assert d.evidence["measured_transverse_spread"] == pytest.approx(0.667)


# ===========================================================================
# W57 -- the substructuring cut criterion, L2/C3
# ===========================================================================


@pytest.fixture(scope="module")
def sub_sweep():
    phys = SM.SteadyPhysics()
    f = phys.source(0)
    u_star = SM.monolith_solve(phys, f)
    rows = [SM.measure(SM.SubstructureDecomposition(phys=phys, m_eff=8, cut=c),
                       f, u_star)
            for c in range(phys.ny // 2)]
    return phys, f, u_star, rows


def _spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, float))).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def test_substructuring_is_exact_at_full_resolution(sub_sweep):
    """The instrument's own premise: with the FULL interface space the Schur
    complement reproduces the monolith, so every defect the sweep reports is the
    DECLARED interface space's truncation and not the model's error."""
    phys, f, u_star, _ = sub_sweep
    for cut in (0, 3, 7):
        assert SM.exact_check(phys, f, u_star, cut=cut) < 1e-12


def test_w57_the_tight_form_is_a_genuine_bound_on_the_trace_error(sub_sweep):
    """``S_M (lam_dag - a*) = -r`` is an identity; ``||lam_dag - a*|| <= ||r||/beta``
    is the one bound taken.  It must hold at every cut, exactly."""
    _phys, _f, _u, rows = sub_sweep
    for r in rows:
        assert r["trace_error"] <= r["q_tight"] * (1 + 1e-9), r["cut"]


def test_w57_the_criterion_ranks_cut_placements(sub_sweep):
    """The experiment section 10.2 ran on the overlapping branch, run here.

    The defect spreads 14x over the placements, so there is something to rank.
    """
    _phys, _f, _u, rows = sub_sweep
    D = [r["composed_defect"] for r in rows]
    Q = [r["q_tight"] for r in rows]
    assert max(D) / min(D) > 5.0
    assert _spearman(Q, D) > 0.8


def test_w57_dividing_by_beta_helps_here_where_it_inverted_the_ranking_there(sub_sweep):
    """`master-error-bound` 4's box says 1/beta is the amplifier only where there
    IS an interface solve.  Section 10.2 measured it inverting the ranking on the
    overlapping branch; this is the other half of that claim, measured."""
    _phys, _f, _u, rows = sub_sweep
    D = [r["composed_defect"] for r in rows]
    assert _spearman([r["q_tight"] for r in rows], D) >= _spearman(
        [r["residual"] for r in rows], D)


def test_w57_section_4s_product_form_does_not_rank(sub_sweep):
    """**The disproof half.**  The loose form is not merely loose -- it is
    anti-correlated, negative in 9 of the 11 configurations swept by
    `scripts/w57_substructuring.py`.  It is also the shape the retired Q had,
    which is why that one ranked at -0.853.
    """
    _phys, _f, _u, rows = sub_sweep
    D = [r["composed_defect"] for r in rows]
    assert _spearman([r["q_product"] for r in rows], D) < 0.0


def test_w57_following_the_criterion_costs_little(sub_sweep):
    """The retired Q cost 92% of the available range when followed."""
    _phys, _f, _u, rows = sub_sweep
    D = np.array([r["composed_defect"] for r in rows])
    Q = np.array([r["q_tight"] for r in rows])
    assert D[int(np.argmin(Q))] / D.min() < 1.2


# ===========================================================================
# E3 -- a genuine governing-family disagreement
# ===========================================================================


def test_e3_fails_on_a_real_cross_family_seam_and_the_probe_still_runs():
    """**The claim the spec makes and no real object had ever tested.**

    All four pre-existing real case studies are 2-D incompressible
    Navier-Stokes, so `governing_family` had only been compared against itself
    and every graph failing E3 was a fixture built to fail it.

    This asserts the RULE on records; `scripts/w67_thermal_seam.py` runs it
    against `compressible2d` and `thermostruct2d` from the build repo, where the
    measured outcome is: verdict `admit-uncertified` with **zero refusals**,
    E3 `fails`, tau `UNDEFINED` on the seam, and the probe assembling a
    well-conditioned operator anyway (beta 4.806, kappa 1.0002, null dim 0).
    """
    from atlas.graph import Agent
    a = _caps("gas")
    b = _caps("shell")
    object.__setattr__(a, "governing_family", "compressible-navier-stokes-2d")
    object.__setattr__(b, "governing_family", "thermoelastic-shell-2d")
    g = CaseGraph(
        name="cross-family", agents=[Agent("gas", a), Agent("shell", b)],
        connections=[Connection(seam_id="cht", a=("gas", "xhi:MECH"),
                                b=("shell", "xlo:MECH"), port_type=PortType.MECH,
                                derive_space=True, geometrically_coincident=True,
                                expected_null_dim=None)],
        decomposition=Decomposition.OVERLAPPING, overlap=1.0, overlap_cells=4,
        macro_dt=1e-2,
    )
    r = compile_scheme(g)
    from atlas.envelope import Hypothesis, Status
    assert r.envelope.values[Hypothesis.E3] is Status.FAILS
    assert r.tau_undefined_seams == ["cht"]
    # ...and the probe ran anyway: probing mentions no governing equation.
    assert "cht" in r.seam_operators
    assert not r.seam_operators["cht"].is_empty
    # E3's failure is a decertification, never a refusal: the composition RUNS.
    assert not [d for d in r.decisions.refusals if d.rule == "E3"]
