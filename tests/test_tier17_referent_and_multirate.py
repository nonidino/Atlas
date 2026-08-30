"""Tier 17 -- W87, W85, W86, and W7's multirate half.

Tier 16 made the reference pair load-bearing and left three things about it
unfinished.  Each is decided here by a measurement, and where the measurement
contradicted the expectation the test records the measurement.

The scope statements are tests rather than caveats on purpose: `_test_lambda_ref`
catches exactly one thing, and a test that pins down what it does NOT catch is
the only way that stays true.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from atlas.conformance import _test_lambda_ref, run_conformance
from atlas.multiphysics import tight_couple
from atlas.ports import ResponseHalf
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

N = 8
A_SLOPE, B_SLOPE, TA, TB = 2.0, 3.0, 300.0, 900.0
EXACT_ROOT = (A_SLOPE * TA - B_SLOPE * TB) / (A_SLOPE - B_SLOPE)


def _A(lam, factor=1.0):
    return factor * A_SLOPE * (np.asarray(lam, dtype=float) - TA)


def _B(lam):
    return -B_SLOPE * (np.asarray(lam, dtype=float) - TB)


def _pair():
    return {"A": _A, "B": _B}


def _caps(**kw):
    from atlas.capability import ExpertCapabilities, PortType, port_decl

    base = dict(
        expert_id="A",
        ports=[port_decl(name="p:MECH", port_type=PortType.MECH,
                         nondim={"velocity": 1.0, "traction": 1.0, "power_area": 1.0},
                         response_half=ResponseHalf.EFFORT)],
        substeps_per_macro_step=1,
        stencil_radius=1,
    )
    base.update(kw)
    return ExpertCapabilities(**base)


TRACE0 = np.full(N, 400.0)


# ---------------------------------------------------------------------------
# W87 -- lambda_ref names an experiment
# ---------------------------------------------------------------------------


def test_an_undeclared_lambda_ref_decertifies_and_has_nothing_to_falsify():
    t = _test_lambda_ref(_caps(lambda_ref=None), _pair(), TRACE0)
    assert t.verdict is ADMIT_UNCERTIFIED
    assert t.measured == "not declared"
    assert "conservative reading" in t.message


def test_a_declared_lambda_ref_with_no_pair_supplied_decertifies():
    """An unrun experiment is not a passing one, and this is the field where
    that matters: a false lambda_ref makes L1/E3 promote."""
    t = _test_lambda_ref(_caps(lambda_ref="the same solver"), None, None)
    assert t.verdict is ADMIT_UNCERTIFIED
    assert "was not built" in t.message


def test_a_pair_that_does_not_name_this_expert_is_not_this_expert_s_referent():
    t = _test_lambda_ref(_caps(expert_id="C", lambda_ref="x"), _pair(), TRACE0)
    assert t.verdict is ADMIT_UNCERTIFIED
    assert "not this agent's referent" in t.message


def test_a_convergent_reference_pair_certifies_lambda_ref():
    t = _test_lambda_ref(_caps(lambda_ref="itself"), _pair(), TRACE0)
    assert t.verdict is ADMIT
    assert t.residual is not None and t.residual < 1e-6


def test_an_empty_interface_problem_refuses_because_no_state_balances_the_bond():
    """Both sides ignore the trace, so the residual is a nonzero constant and no
    interface state balances it.  `CASE-STUDY-GUIDE` mistake 6 arriving through a
    declaration -- and the ONE thing this test falsifies."""
    blind = {"A": lambda lam: np.full(N, 1.0), "B": lambda lam: np.full(N, 2.0)}
    t = _test_lambda_ref(_caps(lambda_ref="a reference that cannot couple"),
                         blind, TRACE0)
    assert t.verdict is REFUSE
    assert t.silent_if_false
    assert "No interface state balances the bond" in t.message


def test_the_check_cannot_see_a_sign_flipped_reference():
    """The scope statement, as a test so it cannot quietly grow.

    A monotone residual has a root whichever way its two halves lean, so a
    reference on the wrong side of its own bond still converges -- to a
    DIFFERENT trace.  Measured on `thermal_seam` at 425.69 K against the true
    372.14 K; here in the algebra the root simply moves.
    """
    flipped = {"A": _A, "B": lambda lam: -_B(lam)}
    t = _test_lambda_ref(_caps(lambda_ref="a reference with a sign error"),
                         flipped, TRACE0)
    assert t.verdict is ADMIT
    moved = tight_couple(lambda l: flipped["A"](l) + flipped["B"](l), TRACE0)
    assert moved.converged
    assert abs(float(np.mean(moved.trace)) - EXACT_ROOT) > 1.0


def test_a_blind_reference_on_one_side_alone_still_converges():
    """Four of five wrong pairs converge, and this is one of them: the other
    side's response still crosses zero."""
    one_blind = {"A": _A, "B": lambda lam: np.full(N, -100.0)}
    t = _test_lambda_ref(_caps(lambda_ref="a reference that ignores its datum"),
                         one_blind, TRACE0)
    assert t.verdict is ADMIT


def test_a_wrong_referent_corrupts_the_magnitude_and_not_the_localization():
    """What the miss costs, and the one reassuring thing about it.

    The ACTUAL pair is its own reference, so the true tau is zero on both
    agents.  Scored against a sign-flipped referent, the defect is exactly 2 --
    ``|P - (-P)| / |P|`` -- and it lands entirely on the corrupted side.
    """
    from atlas.multiphysics import seam_defect_split

    halves = {"A": ResponseHalf.FLOW, "B": ResponseHalf.FLOW}
    flipped = {"A": _A, "B": lambda lam: -_B(lam)}
    d = seam_defect_split("s", _pair(), flipped, halves, TRACE0, TRACE0)
    assert d.converged
    assert d.tau["B"] == pytest.approx(2.0, rel=1e-6)
    assert d.tau["A"] == pytest.approx(0.0, abs=1e-9)


def test_lambda_ref_is_not_a_fourth_unverifiable_declaration():
    """W87's actual result.  The worklist logged it as a fourth field of
    `validity`'s kind because it is a free-form string.  The string is not what
    the rule consumes -- the rule consumes the claim that a pair converges, and
    that is one solve."""
    cert = run_conformance(_caps(lambda_ref="itself", governing_family="x",
                                 validity=lambda *a: True))
    names = {t.field_name for t in cert.tests if t.cost == "none exists"}
    assert names == {"validity", "governing_family"}
    lam = [t for t in cert.tests if t.field_name == "lambda_ref"][0]
    assert lam.cost != "none exists"


# ---------------------------------------------------------------------------
# W86 -- sigma is a function of the lag, and the lag belongs to the run
# ---------------------------------------------------------------------------


def test_the_lag_is_the_max_cellwise_move_between_consecutive_traces():
    """Max, not a mean: the lag enters the response pointwise, and a mean hides a
    lag that is large somewhere and zero elsewhere -- which is exactly the case
    that breaks the uniform-shift picture."""
    from atlas.multiphysics import lag_distance

    prev = np.zeros(6)
    cur = np.array([0.0, 0.0, 3.0, 0.0, 0.0, -1.0])
    assert lag_distance(prev, cur) == pytest.approx(3.0)


def test_the_lag_refuses_two_traces_that_are_not_the_same_interface():
    from atlas.multiphysics import MultiphysicsError, lag_distance

    with pytest.raises(MultiphysicsError):
        lag_distance(np.zeros(4), np.zeros(5))


def test_a_sigma_with_no_declared_lag_has_nothing_to_be_compared_against():
    from atlas.multiphysics import check_sigma_lag

    c = check_sigma_lag(None, 1.0e-3)
    assert c.agrees is None
    assert "not a number" in c.note


def test_a_lag_from_another_regime_falsifies_the_quoted_sigma():
    from atlas.multiphysics import check_sigma_lag

    c = check_sigma_lag(9.83e-4, 5.0e-2)
    assert c.agrees is False
    assert "a lag this run does not carry" in c.note


def test_a_matching_lag_establishes_provenance_and_not_correctness():
    """The scope statement, measured: two consecutive macro-steps of
    `thermal_seam` whose lags agree to 8% carry sigmas 1.48x apart, because
    sigma's argument is the lag PROFILE and this compares a scalar summary."""
    from atlas.multiphysics import check_sigma_lag

    c = check_sigma_lag(1.087e-3, 1.174e-3)
    assert c.agrees is True
    assert "NOT that it is right here" in c.note
    assert "1.48x" in c.note


def test_the_declared_thermal_seam_sigma_is_its_slope_times_its_declared_lag():
    """What the number actually is, pinned so it cannot drift back into prose.

    ``MEASURED_W83.sigma`` is not a sigma measured at any lag a run carries: it
    is the seam's uniform-shift slope, 3.4501e-2 per K and constant to five
    digits across five macro-steps, evaluated at the assumed drift.  The run's
    own sigma over those five steps is 4.07e-6 to 3.13e-5, all below it.
    """
    from atlas.cases.thermal_seam import MEASURED_W83

    slope = 3.4501e-2
    assert MEASURED_W83.sigma_lag == pytest.approx(9.83e-4)
    # To three digits -- both the slope and the assumed drift are quoted to
    # five, so agreement past that would be claiming precision neither has.
    assert MEASURED_W83.sigma == pytest.approx(slope * MEASURED_W83.sigma_lag, rel=1e-3)


def test_the_rollout_derives_its_own_lag_instead_of_trusting_the_declaration():
    """W77's shape where it DOES apply: the traces were already being carried
    from step to step and nothing read them."""
    from atlas.capability import ExpertCapabilities, port_decl
    from atlas.compiler import compile_scheme
    from atlas.graph import Agent, CaseGraph, Connection, MeasuredConstants
    from atlas.ports import PortType
    from atlas.solve import rollout

    m = 4
    scales = {"stress": 2.0, "velocity": 3.0, "power_area": 6.0}

    def expert(name, face, gain, seed):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((m, m)) * 0.1
        A = gain * (A + A.T + 3.0 * np.eye(m))
        bias = rng.standard_normal(m) * 0.5
        return ExpertCapabilities(
            expert_id=name,
            ports=[port_decl(name=f"{face}:MECH", port_type=PortType.MECH,
                             nondim=dict(scales), effective_resolution=m,
                             response_half=ResponseHalf.EFFORT)],
            dt_native=1e-2, governing_family="linear-test",
            boundary_response=lambda _p, t, _A=A, _b=bias: _A @ np.asarray(t, float) + _b,
            substeps_per_macro_step=1, stencil_radius=1,
        )

    graph = CaseGraph(
        name="lag-pair",
        agents=[Agent("a", expert("a", "b", 1.0, 1)),
                Agent("b", expert("b", "a", 0.7, 2))],
        connections=[Connection("a-b", ("a", "b:MECH"), ("b", "a:MECH"),
                                PortType.MECH, derive_space=True,
                                geometrically_coincident=True)],
        macro_dt=1e-2,
        measured=MeasuredConstants(sigma=1.0e-6, sigma_lag=1.0e-3),
    )
    result = compile_scheme(graph)
    _, report, _ = rollout(result, graph, n_steps=3)
    assert "a-b" in report.lag
    entry = report.lag["a-b"]
    assert entry["declared_lag"] == pytest.approx(1.0e-3)
    assert entry["actual_lag"] is not None and entry["actual_lag"] >= 0.0


def test_a_single_step_has_no_lag_because_a_lag_needs_two_traces():
    from atlas.multiphysics import lag_distance

    assert lag_distance(np.ones(3), np.ones(3)) == 0.0


# ---------------------------------------------------------------------------
# W85 -- the seam operator is not a Jacobian, and the reason is a subspace
# ---------------------------------------------------------------------------


def test_a_seam_operator_on_M_is_refused_as_a_dense_jacobian_on_V():
    """It used to fall through to the scalar secant, silently. A dim-M operator
    handed to a dim-V Newton is a shape error and now says so."""
    from atlas.multiphysics import MultiphysicsError, tight_couple

    with pytest.raises(MultiphysicsError) as exc:
        tight_couple(lambda l: _A(l) + _B(l), TRACE0, jacobian=np.eye(N // 2))
    assert "not a Jacobian on V" in str(exc.value)


def test_the_coarse_step_zeroes_what_it_can_reach_and_nothing_else():
    """W85's mechanism, in the algebra it is a fact about.

    ``P S^-1 R r`` lives in ``range(P)``.  Give the residual a component
    orthogonal to that subspace and no number of iterations can touch it --
    which is why no threshold on `operator_content` could have rescued this, and
    why the measured plateau on `thermal_seam` (1.04e-04) is exactly the
    orthogonal part of its residual.
    """
    from atlas.multiphysics import seam_jacobian, tight_couple

    n, m = 8, 3
    rng = np.random.default_rng(0)
    P, _ = np.linalg.qr(rng.standard_normal((n, m)))     # orthonormal columns
    R = P.T
    A = np.eye(n)
    off = np.zeros(n)
    off[m:] = 1.0                                        # outside range(P)
    off = off - P @ (R @ off)
    assert np.linalg.norm(off) > 0.1

    def residual(lam):
        return A @ np.asarray(lam, float).ravel() + off

    S = R @ A @ P
    tc = tight_couple(residual, np.zeros(n), jacobian=seam_jacobian(S, P, R))
    assert not tc.converged
    r = residual(tc.trace)
    assert np.linalg.norm(P @ (R @ r)) < 1e-10           # what it could reach
    assert np.linalg.norm(r - P @ (R @ r)) == pytest.approx(
        float(np.linalg.norm(off)), rel=1e-9)            # what it could not


def test_the_dense_jacobian_solves_the_same_problem_the_coarse_one_cannot():
    """The control: the same residual, converged, when the step spans all of V."""
    from atlas.multiphysics import tight_couple

    n, m = 8, 3
    rng = np.random.default_rng(0)
    P, _ = np.linalg.qr(rng.standard_normal((n, m)))
    off = np.zeros(n)
    off[m:] = 1.0
    off = off - P @ (P.T @ off)
    tc = tight_couple(lambda lam: np.asarray(lam, float).ravel() + off, np.zeros(n))
    assert tc.converged


# ---------------------------------------------------------------------------
# W7 / R9 -- multirate, and the term the rule does not mention
# ---------------------------------------------------------------------------


def _multirate_graph(flux_matching=None, integrated=True, dt_b=1.0e-3):
    from atlas.capability import ExpertCapabilities, port_decl
    from atlas.graph import Agent, CaseGraph, Connection, FluxMatching
    from atlas.ports import PortType

    m = 4
    scales = {"stress": 2.0, "velocity": 3.0, "power_area": 6.0}

    def expert(name, face, gain, seed, dt):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((m, m)) * 0.1
        A = gain * (A + A.T + 3.0 * np.eye(m))
        bias = rng.standard_normal(m) * 0.5

        def respond(_p, t, _A=A, _b=bias):
            return _A @ np.asarray(t, float) + _b

        caps = ExpertCapabilities(
            expert_id=name,
            ports=[port_decl(name=f"{face}:MECH", port_type=PortType.MECH,
                             nondim=dict(scales), effective_resolution=m,
                             response_half=ResponseHalf.EFFORT)],
            dt_native=dt, governing_family="linear-test",
            boundary_response=respond,
            substeps_per_macro_step=1, stencil_radius=1,
        )
        if integrated:
            caps.boundary_response_integrated = (
                lambda p, t, n, _r=respond: _r(p, t))
        return caps

    return CaseGraph(
        name="multirate-pair",
        agents=[Agent("a", expert("a", "b", 1.0, 1, 1.0e-4)),
                Agent("b", expert("b", "a", 0.7, 2, dt_b))],
        connections=[Connection("a-b", ("a", "b:MECH"), ("b", "a:MECH"),
                                PortType.MECH, derive_space=True,
                                geometrically_coincident=True)],
        macro_dt=dt_b,
        flux_matching=flux_matching or FluxMatching.POINTWISE,
    )


def _rules(result, prefix="R9"):
    return {d.rule: d.verdict.value for d in result.decisions._decisions
            if d.rule.startswith(prefix)}


def test_a_pointwise_multirate_graph_is_still_refused():
    """Unchanged, and it is the refusal this tier gives an exit rather than
    removes: across two clocks the pointwise match is not conservative."""
    from atlas.compiler import compile_scheme

    r = compile_scheme(_multirate_graph())
    assert r.verdict is REFUSE
    assert _rules(r)["R9"] == "refuse"


def test_declaring_time_integrated_matching_admits_R9():
    from atlas.compiler import compile_scheme
    from atlas.graph import FluxMatching

    r = compile_scheme(_multirate_graph(FluxMatching.TIME_INTEGRATED))
    assert _rules(r)["R9"] == "admit"
    assert r.scheme.flux_matching == "time-integrated"
    assert r.verdict is not REFUSE


def test_declaring_it_without_the_callable_is_a_contradiction_and_refuses():
    """`boundary_response_jvp`'s rule, applied to the same shape of claim: a
    scheme that matches an integral it cannot compute is refused at L7, not
    discovered by the run."""
    from atlas.compiler import compile_scheme
    from atlas.graph import FluxMatching

    r = compile_scheme(_multirate_graph(FluxMatching.TIME_INTEGRATED,
                                        integrated=False))
    assert r.verdict is REFUSE
    assert _rules(r)["R9/quadrature"] == "refuse"


def test_clocks_that_do_not_nest_are_refused_because_there_is_no_common_integral():
    from atlas.compiler import compile_scheme
    from atlas.graph import FluxMatching

    r = compile_scheme(_multirate_graph(FluxMatching.TIME_INTEGRATED,
                                        dt_b=2.5e-4 * 1.3))
    assert r.verdict is REFUSE
    assert _rules(r)["R9/nesting"] == "refuse"


def test_the_substep_quadrature_is_derived_from_the_declared_clocks():
    from atlas.solve import _substep_quadrature

    g = _multirate_graph()
    assert _substep_quadrature(g, "a", 1.0e-3) == 10
    assert _substep_quadrature(g, "b", 1.0e-3) == 1


def test_R9_is_admitted_and_immediately_decertified_for_the_larger_term():
    """The finding, as a test. R9 refuses over the flux transient; the lag over
    the same interval is 62.4x larger, measured on thermal_seam in interface
    power. Admitting R9 without saying so would read as fixing multirate."""
    from atlas.compiler import compile_scheme
    from atlas.graph import FluxMatching

    r = compile_scheme(_multirate_graph(FluxMatching.TIME_INTEGRATED))
    rules = _rules(r)
    assert rules["R9"] == "admit"
    assert rules["R9/lag"] == "admit-uncertified"
    msg = [d.message for d in r.decisions._decisions if d.rule == "R9/lag"][0]
    assert "62.4" in msg


def test_the_integrated_response_at_one_substep_is_the_plain_response():
    """Slow: two real solvers. The check that makes `boundary_response_integrated`
    a testable declaration rather than a fifth unverifiable one -- one sub-step
    IS the interval, so the two callables must agree exactly."""
    from atlas.cases import thermal_seam as T

    lam = np.full(T.N_SEAM, 400.0)
    for agent, port in ((T.GasAgent(dt=T.DT_GAS), "wall:THERM"),
                        (T.ShellAgent(dt=T.DT_GAS), "inner:THERM")):
        a = np.asarray(agent.respond(port, lam), float)
        b = np.asarray(agent.respond_integrated(port, lam, 1), float)
        assert np.allclose(a, b, rtol=0, atol=0)


def test_the_thermal_seam_at_native_clocks_no_longer_refuses():
    """The target this tier was pointed at: 500:1, and it was `refuse`."""
    from atlas.compiler import compile_scheme
    from atlas.cases import thermal_seam as T

    g, _ = T.build(mode="split-step", clocks="native")
    r = compile_scheme(g, probe_state="duct, T_hot=900 K")
    assert r.verdict is ADMIT_UNCERTIFIED
    assert not [d for d in r.decisions._decisions if d.verdict is REFUSE]
    assert _rules(r)["R9"] == "admit"
    # E4 still fails, and always will: the clocks really do differ.
    assert r.scheme.multirate is True


def test_a_time_integrated_run_actually_calls_the_integrated_response():
    """The compile ADMITS R9 on the grounds that `coupled_step` calls the
    integrated callable.  Nothing checked that it does, and it did not: the
    scheme was not being passed to `_port_fluxes`, so the branch was unreachable
    and the assert guarding it could never fire.  That is the silent-wrongness
    class the verdict split exists to separate -- an admission resting on a
    promise the run does not keep -- so it gets a test rather than a comment.
    """
    from atlas.compiler import compile_scheme
    from atlas.graph import FluxMatching
    from atlas.solve import coupled_step

    seen = {"integrated": 0, "plain": 0, "n": set()}
    g = _multirate_graph(FluxMatching.TIME_INTEGRATED)
    for a in g.agents:
        caps = a.capabilities
        plain, integ = caps.boundary_response, caps.boundary_response_integrated

        def wrapped_plain(p, t, _f=plain):
            seen["plain"] += 1
            return _f(p, t)

        def wrapped_integ(p, t, n, _f=integ):
            seen["integrated"] += 1
            seen["n"].add(n)
            return _f(p, t, n)

        caps.boundary_response = wrapped_plain
        caps.boundary_response_integrated = wrapped_integ

    result = compile_scheme(g)
    seen["plain"] = seen["integrated"] = 0        # ignore the compile's probing
    seen["n"].clear()
    coupled_step(result, g)
    assert seen["integrated"] > 0
    # each side quadrature'd on its OWN clock: 1e-3 / 1e-4 and 1e-3 / 1e-3
    assert seen["n"] == {10, 1}


def test_a_pointwise_run_calls_the_plain_response_and_not_the_integral():
    """The control: the default path is untouched."""
    from atlas.compiler import compile_scheme
    from atlas.graph import Agent, CaseGraph, Connection, MeasuredConstants
    from atlas.solve import coupled_step

    g = _multirate_graph()                        # pointwise -> refused
    single = CaseGraph(
        name="single-clock", agents=g.agents,
        connections=g.connections, macro_dt=1.0e-4,
        measured=MeasuredConstants(sigma=1e-6, sigma_lag=1e-3),
    )
    for a in single.agents:
        a.capabilities.dt_native = 1.0e-4         # one clock, so R9 does not apply
    seen = {"integrated": 0}
    for a in single.agents:
        integ = a.capabilities.boundary_response_integrated

        def wrapped(p, t, n, _f=integ):
            seen["integrated"] += 1
            return _f(p, t, n)

        a.capabilities.boundary_response_integrated = wrapped
    result = compile_scheme(single)
    coupled_step(result, single)
    assert seen["integrated"] == 0
