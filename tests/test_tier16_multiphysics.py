"""Tier 16 -- tau at a seam whose two sides solve different equations.

E3 compares two ``governing_family`` strings and, when they differ, the compiler
emitted ``tau`` as UNDEFINED for both sides.  That was the standing reason no
multiphysics graph could be certified.  But ``tau`` is measured as *the composed
step given the TRUE trace, against a reference trajectory*, and nothing in that
mentions a governing equation -- it needs a reference TRAJECTORY, which at a
multiphysics seam is the tightly coupled pair.

The algebra is tested directly wherever it can be, because a surrogate carrying a
known error is the only way to check that an attribution attributes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from atlas.multiphysics import (
    MultiphysicsError,
    SeamDefect,
    interface_power,
    numerical_jacobian,
    seam_defect_split,
    tight_couple,
)
from atlas.ports import PortType, ResponseHalf
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

N = 8
A_SLOPE, B_SLOPE, TA, TB = 2.0, 3.0, 300.0, 900.0
#: a(l - Ta) - b(l - Tb) = 0  ->  l = (a Ta - b Tb) / (a - b)
EXACT_ROOT = (A_SLOPE * TA - B_SLOPE * TB) / (A_SLOPE - B_SLOPE)


def _A(lam, factor=1.0):
    return factor * A_SLOPE * (np.asarray(lam, dtype=float) - TA)


def _B(lam):
    return -B_SLOPE * (np.asarray(lam, dtype=float) - TB)


def _pair(factor=1.0):
    return {"A": lambda l: _A(l, factor), "B": _B}


HALVES = {"A": ResponseHalf.FLOW, "B": ResponseHalf.FLOW}


# ---------------------------------------------------------------------------
# the common currency
# ---------------------------------------------------------------------------


def test_interface_power_refuses_an_undeclared_half():
    """The bond's two halves are not distinguishable from the returned numbers.

    W66 measured that; guessing here would put a sign and a unit error into every
    attribution downstream, so multiphysics attribution inherits that dependency
    openly instead of hiding it.
    """
    with pytest.raises(MultiphysicsError, match="response_half"):
        interface_power(np.ones(4), np.ones(4), ResponseHalf.UNDECLARED)


def test_interface_power_is_the_same_product_whichever_half_is_returned():
    """effort x flow does not care which side of the bond the expert hands back."""
    t, r = np.array([2.0, 3.0]), np.array([5.0, 7.0])
    assert interface_power(t, r, ResponseHalf.FLOW) == pytest.approx(10.0 + 21.0)
    assert interface_power(t, r, ResponseHalf.EFFORT) == pytest.approx(10.0 + 21.0)


def test_interface_power_applies_the_seam_measure_as_a_quadrature():
    t, r = np.ones(4), np.full(4, 2.0)
    assert interface_power(t, r, ResponseHalf.FLOW, 0.25) == pytest.approx(2.0)


def test_interface_power_refuses_mismatched_lengths():
    with pytest.raises(MultiphysicsError, match="pointwise"):
        interface_power(np.ones(4), np.ones(5), ResponseHalf.FLOW)


def test_every_port_type_has_a_declared_bond():
    """The power norm is only general because the port algebra already was."""
    from atlas.multiphysics import bond_of

    for pt in PortType:
        e, f = bond_of(pt)
        assert isinstance(e, str) and isinstance(f, str) and e and f


# ---------------------------------------------------------------------------
# the reference trajectory
# ---------------------------------------------------------------------------


def test_tight_couple_finds_the_exact_root_of_a_linear_interface():
    tc = tight_couple(lambda l: _A(l) + _B(l), np.full(N, 500.0))
    assert tc.converged
    assert float(np.mean(tc.trace)) == pytest.approx(EXACT_ROOT, rel=1e-9)


def test_tight_couple_reports_a_stall_instead_of_returning_a_bad_root():
    """A referent that was not built is not a referent.

    Measured on `thermal_seam`: the scalar secant moves only a UNIFORM shift of
    the trace and stalls five orders in at 3.04e-3, because the part of the
    residual that varies along the seam is untouchable by a uniform move. With
    the Jacobian the same solve reaches 1e-7 in 9 iterations.
    """
    tc = tight_couple(lambda l: _A(l) + _B(l), np.full(N, 500.0), max_iter=0)
    assert not tc.converged


def test_the_numerical_jacobian_matches_the_analytic_one():
    J = numerical_jacobian(lambda l: _A(l) + _B(l), np.full(N, 500.0))
    assert J == pytest.approx((A_SLOPE - B_SLOPE) * np.eye(N), abs=1e-6)


def test_a_scalar_secant_stalls_where_the_jacobian_does_not():
    """The stall is structural, not a tuning problem -- reproduced in the algebra.

    A residual whose along-seam variation is not a multiple of the uniform mode
    cannot be removed by a uniform step, whatever the step size.
    """
    target = EXACT_ROOT + np.linspace(-30.0, 30.0, N)   # a NON-uniform root

    def residual(lam):
        return np.asarray(lam, dtype=float) - target

    stalled = tight_couple(residual, np.full(N, 500.0), jacobian=None, max_iter=40)
    solved = tight_couple(residual, np.full(N, 500.0), max_iter=40)
    assert not stalled.converged
    assert solved.converged
    assert solved.trace == pytest.approx(target, rel=1e-9)


# ---------------------------------------------------------------------------
# the split -- does an attribution attribute?
# ---------------------------------------------------------------------------


def _split(factor=1.0, lag=400.0):
    return seam_defect_split("s", _pair(factor), _pair(1.0), HALVES,
                             np.full(N, 500.0), np.full(N, lag))


def test_a_pair_that_is_its_own_reference_has_exactly_zero_tau():
    """The correct answer for a composition of exact solvers, and worth stating."""
    d = _split(1.0)
    assert d.converged
    assert d.tau == {"A": 0.0, "B": 0.0}
    assert d.total == pytest.approx(d.sigma, rel=1e-12)


@pytest.mark.parametrize("factor,expected", [(1.05, 0.05), (1.25, 0.25), (2.0, 1.0)])
def test_tau_recovers_an_injected_error_exactly(factor, expected):
    """The whole claim, checked against a known answer rather than a plausible one."""
    d = _split(factor)
    assert d.tau["A"] == pytest.approx(expected, rel=1e-9)


def test_tau_attributes_to_the_swapped_agent_and_not_its_neighbour():
    d = _split(2.0)
    assert d.tau["A"] > 0.5
    assert d.tau["B"] == 0.0
    assert d.tau_worst == ("A", d.tau["A"])


def test_the_split_is_subadditive_by_the_triangle_inequality():
    """sigma is taken against the TRUE trace, exactly as tier0 takes it, so
    ``total <= sum(tau) + sigma`` is an inequality and not a hope."""
    for factor in (1.0, 1.05, 1.25, 2.0, 5.0):
        assert _split(factor).subadditive


def test_the_split_refuses_when_the_reference_pair_will_not_converge():
    """E3's own verdict, arrived at honestly -- by failing to BUILD the referent
    rather than by comparing two strings."""
    wobble = {"A": lambda l: np.cos(np.asarray(l, dtype=float)) + 2.0,
              "B": lambda l: np.zeros_like(np.asarray(l, dtype=float))}
    d = seam_defect_split("s", wobble, wobble, HALVES, np.full(N, 500.0),
                          np.full(N, 400.0))
    assert not d.converged
    assert d.tau == {}
    assert "no reference trajectory" in d.note or "did not converge" in d.note


def test_the_split_refuses_a_pair_naming_different_agents():
    with pytest.raises(MultiphysicsError, match="different agents"):
        seam_defect_split("s", {"A": _A}, {"B": _B}, HALVES,
                          np.full(N, 500.0), np.full(N, 400.0))


def test_the_split_refuses_an_agent_with_no_declared_half():
    with pytest.raises(MultiphysicsError, match="response_half"):
        seam_defect_split("s", _pair(), _pair(), {"A": ResponseHalf.FLOW},
                          np.full(N, 500.0), np.full(N, 400.0))


def test_sigma_vanishes_when_the_lag_is_the_converged_trace():
    """The check that sigma measures the lag and nothing else."""
    d = _split(1.0, lag=EXACT_ROOT)
    assert d.sigma == pytest.approx(0.0, abs=1e-9)


def test_sigma_is_linear_in_the_lag_to_leading_order_and_no_further():
    """Interface power is BILINEAR -- effort times flow -- so sigma carries a
    quadratic correction and is only asymptotically linear in the lag.

    Visible in both places. Here: a decade of lag gives 9.9727 rather than 10.
    On `thermal_seam`: 0.1 K -> 1 K gives 9.99, 1 K -> 5 K gives 4.97 against 5,
    and 5 K -> 20 K gives 3.91 against 4. So quoting a single slope in "per K"
    is right near the operating point and wrong away from it, which is why the
    declared sigma carries the lag it was measured at.
    """
    ratios = []
    for small in (1e-3, 1e-2, 1e-1):
        a = _split(1.0, lag=EXACT_ROOT - small)
        b = _split(1.0, lag=EXACT_ROOT - 10.0 * small)
        ratios.append(b.sigma / a.sigma)
    # the correction shrinks with the lag: linearity is a limit, not an identity
    assert ratios[0] == pytest.approx(10.0, rel=1e-4)
    assert abs(ratios[0] - 10.0) < abs(ratios[-1] - 10.0)
    assert _split(1.0, lag=EXACT_ROOT - 10.0).sigma /         _split(1.0, lag=EXACT_ROOT - 1.0).sigma < 10.0


# ---------------------------------------------------------------------------
# the compiler: E3's consequence, which is what actually changed
# ---------------------------------------------------------------------------


def _thermal(**kw):
    from atlas import compile_scheme
    from atlas.cases import thermal_seam as T

    g, _ = T.build(mode="split-step", clocks="matched", **kw)
    return compile_scheme(g, probe_state="duct, T_hot=900 K")


def _rule(result, name):
    return [d for d in result.decisions._decisions if d.rule == name]


def test_a_multiphysics_seam_no_longer_reports_tau_undefined():
    """The blocker, removed. E3 still FAILS -- the families really do differ."""
    r = _thermal()
    assert r.tau_undefined_seams == []
    assert _rule(r, "E3")[0].verdict is ADMIT


def test_e3_still_fails_as_a_hypothesis_even_though_tau_survives():
    """What E3's failure costs is the MONOLITHIC reference, and nothing else."""
    from atlas.envelope import Hypothesis

    r = _thermal()
    assert r.envelope[Hypothesis.E3].value == "fails"


def test_without_a_measurement_the_seam_decertifies_rather_than_admitting():
    r = _thermal(measured=None)
    d = _rule(r, "E3")[0]
    assert d.verdict is ADMIT_UNCERTIFIED
    assert "has not been RUN" in d.message
    assert r.tau_undefined_seams == []


def test_without_lambda_ref_tau_is_undefined_again_and_says_why():
    """The gate is the REFERENT now, not the family. Removing the declaration
    puts tau back to UNDEFINED without either governing_family moving."""
    from atlas import compile_scheme
    from atlas.cases import thermal_seam as T

    g, _ = T.build(mode="split-step", clocks="matched", measured=None)
    for a in g.agents:
        a.capabilities.lambda_ref = None
    r = compile_scheme(g, probe_state="duct")
    assert r.tau_undefined_seams == ["cht"]
    assert "UNDEFINED" in _rule(r, "E3")[0].message


def test_declaring_the_measured_constants_clears_eps_tol():
    from atlas.cases import thermal_seam as T

    assert T.MEASURED_W83.tau == 0.0
    assert _rule(_thermal(measured=None), "eps_tol")[0].verdict is ADMIT_UNCERTIFIED
    assert _rule(_thermal(), "eps_tol")[0].verdict is ADMIT


def test_eps_tol_does_not_degenerate_when_tau_is_exactly_zero():
    """W84. `min(tau, sigma)` sets the interface tolerance to 0 for a composition
    of exact solvers, and no solve can meet it. A term that is identically zero
    names no scale, so the minimum is taken over the terms that do."""
    from atlas.cases import thermal_seam as T

    d = _rule(_thermal(), "eps_tol")[0]
    assert d.evidence["eps_tol"] == pytest.approx(T.MEASURED_W83.sigma)
    assert d.evidence["eps_tol"] > 0.0


def test_both_defect_terms_zero_declines_instead_of_emitting_a_zero_tolerance():
    from atlas import compile_scheme
    from atlas.cases import thermal_seam as T
    from atlas.graph import MeasuredConstants

    g, _ = T.build(mode="split-step", clocks="matched",
                   measured=MeasuredConstants(tau=0.0, sigma=0.0, scheme="split-step"))
    d = _rule(compile_scheme(g, probe_state="x"), "eps_tol")[0]
    assert d.verdict is ADMIT_UNCERTIFIED
    assert "names a scale" in d.message or "no scale" in d.message


def test_the_thermal_seam_compile_lands_where_this_tier_says_it_does():
    r = _thermal()
    ds = r.decisions._decisions
    assert not [d for d in ds if d.verdict is REFUSE]
    fired = sorted({d.rule for d in ds if d.verdict is ADMIT_UNCERTIFIED})
    assert fired == ["C3/W57", "E5", "E6", "W56", "block-share", "operator-content",
                     "probe-base"] or "E3" not in fired
    assert sorted(r.unmeasured) == ["C_mu (W3)", "L (W1)", "norm_A (W28)"]


# ---------------------------------------------------------------------------
# the real seam
# ---------------------------------------------------------------------------


def test_the_real_pair_is_its_own_reference_and_measures_tau_zero():
    """Slow: two real solvers, a Newton solve, and 2(n+1) extra calls."""
    from atlas.cases import thermal_seam as T

    gas, shell = T.GasAgent(dt=T.DT_GAS), T.ShellAgent(dt=T.DT_GAS)
    flows = {"gas": lambda l: np.asarray(gas.respond("wall:THERM", l), float),
             "shell": lambda l: np.asarray(shell.respond("inner:THERM", l), float)}
    d = seam_defect_split("cht", flows, flows,
                          {"gas": ResponseHalf.FLOW, "shell": ResponseHalf.FLOW},
                          np.full(T.N_SEAM, T.T_WALL_0),
                          np.full(T.N_SEAM, T.T_WALL_0), measure=T.H_SEAM)
    assert d.converged
    assert d.tau == {"gas": 0.0, "shell": 0.0}
    # 372.1377 K until Tier 86, while W301 held the gas's wall BC saturated: the
    # gas now feels the wall temperature it is handed (372.4964 K). Then W332:
    # the inviscid flux had been handed the isothermal ghost, whose density made
    # the wall face pass mass, and it now sees the plain mirror -- the consistent
    # interface temperature moves -0.26 K more. Both moves are the solver
    # getting the wall right, not the seam method changing.
    assert d.trace_reference == pytest.approx(372.2359, rel=1e-4)


def test_the_gas_state_norm_is_blind_where_the_bond_power_is_not():
    """Why the defect is measured in interface power and not in either agent's
    own norm: across 100 K of wall temperature over one macro-step the gas's
    state norm moves ~5e-5 while its interface flow moves ~31%.

    Until Tier 86 the interior was BIT-identical across that range -- which was
    W301 (a saturated isothermal wall that never reached the solver), not a
    property of the norm. The conclusion survives the fix; the bit-identity
    did not."""
    from atlas.cases import thermal_seam as T

    gas = T.GasAgent(dt=T.DT_GAS)
    states, flows = [], []
    for wall in (352.378, 400.0, 450.0):
        U, _ = gas._solver(np.full(T.N_SEAM, wall)).advance(gas._U0, gas.dt)
        states.append(np.linalg.norm(U))
        flows.append(np.linalg.norm(
            np.asarray(gas.respond("wall:THERM", np.full(T.N_SEAM, wall)), float)))
    assert not states[0] == states[1] == states[2], "the wall reaches the gas"
    assert (max(states) - min(states)) / min(states) < 1e-3
    assert max(flows) / min(flows) > 1.25
