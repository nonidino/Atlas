"""Tier 48 -- defect correction's claims, on exact linear problems.

[[defect-correction-learned-operator]] makes three claims about the iteration in
`atlas/defect_correction.py` and one about its certificate.  Each is checked here
where every quantity has a closed form, so a failure is the module's and not a
flow's:

1. the limit is the classical map's fixed point whatever the cheap map is;
2. a constant cheap map reproduces the classical march bit for bit (the null
   replacement, W76, as an identity rather than an approximation);
3. the rate is the spectral radius of ``I - J_psi^{-1} J_phi``;
4. a cheap map that makes the iteration diverge is caught and the classical march
   takes over, returning a certified state;
5. the Banach certificate ``||w - w*|| <= ||phi(w) - w|| / (1 - L)`` holds.

No expert is loaded.  The wake-array measurements are pinned in
``tests/test_tier48_learned_contribution.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.defect_correction import (                                  # noqa: E402
    break_even,
    certificate,
    classical_march,
    defect_correct,
    exchange_rate,
    theta_from_march,
    vector_rms,
)


def _linear(n=12, seed=0, radius=0.95):
    """An affine contraction ``phi(w) = A w + b`` with spectral norm ``radius``."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.linspace(radius, 0.2, n)
    a = q @ np.diag(s) @ q.T
    b = rng.standard_normal(n)
    w_star = np.linalg.solve(np.eye(n) - a, b)
    return a, b, w_star


def _norm(w):
    return float(np.linalg.norm(w))


def _propagator_radius(a, a_cheap):
    n = a.shape[0]
    prop = np.eye(n) - np.linalg.solve(np.eye(n) - a_cheap, np.eye(n) - a)
    return float(np.max(np.abs(np.linalg.eigvals(prop))))


def test_the_limit_is_the_classical_fixed_point_whatever_the_cheap_map_is():
    """Claim 1 -- whatever the cheap map's BIAS, given the rate's precondition."""
    a, b, w_star = _linear()
    rng = np.random.default_rng(1)
    s = rng.standard_normal(a.shape)
    s = 0.5 * (s + s.T)
    s /= np.linalg.norm(s, 2)
    # a cheap map wrong in every entry -- a perturbed matrix and a large bias --
    # whose Jacobian still satisfies the precondition, asserted not assumed
    a_cheap = a + 0.01 * s
    c_cheap = b + 3.0 * rng.standard_normal(b.shape)
    assert _propagator_radius(a, a_cheap) < 1.0
    res = defect_correct(lambda w: a @ w + b, lambda w: a_cheap @ w + c_cheap,
                         np.zeros_like(b), r_stop=1e-13, k_max=200, m_max=400,
                         inner_frac=1e-3, norm=_norm, fallback=False)
    assert res.status == "converged"
    assert np.allclose(res.state, w_star, atol=1e-10)
    # and the bias the cheap map carried is gone from the answer
    w_cheap_star = np.linalg.solve(np.eye(len(b)) - a_cheap, c_cheap)
    assert np.linalg.norm(w_cheap_star - w_star) > 1.0


def test_a_constant_cheap_map_is_the_classical_march_bit_for_bit():
    """W76's null replacement, in this slot, is an identity and not an estimate."""
    a, b, _ = _linear(seed=2)
    phi = lambda w: a @ w + b                                            # noqa: E731
    const = np.full_like(b, 7.0)
    dec = defect_correct(phi, lambda w: const, np.zeros_like(b), r_stop=1e-9,
                         k_max=2000, m_max=5, norm=_norm, fallback=False)
    march = classical_march(phi, np.zeros_like(b), r_stop=1e-9, norm=_norm)
    assert dec.status == march.status == "converged"
    assert dec.phi_calls == march.phi_calls
    assert np.array_equal(dec.state, march.state)
    # one cheap call to form the defect and one inner call per outer iteration,
    # and the inner call returns its start exactly
    assert all(r.get("inner_steps", 1) == 1 for r in dec.rows)


def test_the_rate_is_the_spectral_radius_of_the_error_propagator():
    a, b, w_star = _linear(n=10, seed=3, radius=0.98)
    rng = np.random.default_rng(4)
    a_cheap = a + 0.02 * rng.standard_normal(a.shape)
    n = len(b)
    j_phi = np.eye(n) - a
    j_psi = np.eye(n) - a_cheap
    prop = np.eye(n) - np.linalg.solve(j_psi, j_phi)
    rho = float(np.max(np.abs(np.linalg.eigvals(prop))))
    assert rho < 1.0
    res = defect_correct(lambda w: a @ w + b, lambda w: a_cheap @ w + b,
                         np.zeros_like(b), r_stop=1e-14, k_max=80, m_max=3000,
                         inner_frac=1e-6, norm=_norm, reference=w_star, fallback=False)
    errs = [r["error"] for r in res.rows if r["error"] > 1e-11]
    ratios = [errs[i + 1] / errs[i] for i in range(len(errs) // 2, len(errs) - 1)]
    assert ratios, "the iteration converged too fast to read a rate"
    measured = float(np.exp(np.mean(np.log(ratios))))
    assert abs(measured - rho) / rho < 0.05, (measured, rho)


def test_the_exact_map_as_the_cheap_map_converges_in_one_outer_iteration():
    a, b, w_star = _linear(seed=5, radius=0.6)
    phi = lambda w: a @ w + b                                            # noqa: E731
    # the inner march solves the corrected problem to 1e-9 of the first residual,
    # so the second outer residual is at that level -- r_stop sits above it
    res = defect_correct(phi, phi, np.zeros_like(b), r_stop=1e-6, k_max=10,
                         m_max=500, inner_frac=1e-9, norm=_norm, fallback=False)
    assert res.status == "converged"
    assert len(res.rows) == 2


def test_a_cheap_map_that_diverges_is_caught_and_the_classical_march_takes_over():
    """Loud on failure: a stall is detected and the answer is still certified."""
    a, b, w_star = _linear(seed=6, radius=0.9)
    n = len(b)
    # J_psi nearly singular in a direction J_phi is not: the propagator blows up
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a_bad = q @ np.diag(np.full(n, 0.999)) @ q.T
    res = defect_correct(lambda w: a @ w + b, lambda w: a_bad @ w, np.zeros_like(b),
                         r_stop=1e-10, k_max=40, m_max=50, inner_frac=0.1,
                         norm=_norm, divergence_bound=1e6)
    assert res.status == "fallback_converged"
    assert res.fallback_from is not None
    assert res.residual <= 1e-10
    assert np.allclose(res.state, w_star, atol=1e-8)


def test_the_banach_certificate_holds_and_theta_is_read_off_a_march():
    a, b, w_star = _linear(seed=8, radius=0.9)
    lip = float(np.linalg.norm(a, 2))
    rng = np.random.default_rng(9)
    for _ in range(50):
        w = w_star + rng.standard_normal(len(b)) * rng.uniform(1e-3, 10.0)
        r = _norm(a @ w + b - w)
        assert _norm(w - w_star) <= certificate(r, 1.0 / (1.0 - lip)) * (1 + 1e-12)
    march = classical_march(lambda w: a @ w + b, np.zeros_like(b), r_stop=1e-12,
                            norm=_norm, reference=w_star)
    errs = [row["error"] for row in march.rows]
    ress = [row["residual"] for row in march.rows]
    th = theta_from_march(errs, ress)
    assert th["n"] > 0
    # error/residual tends to 1/(1-L) along the slowest mode, and in the tail it is
    # a ratio of numbers near the reference's round-off: 1e-5 relative, not 1e-9
    assert th["theta"] <= (1.0 / (1.0 - lip)) * (1.0 + 1e-5)


def test_shrinking_toward_the_null_element_interpolates_to_the_classical_march():
    """``shrink(psi, 1)`` is the classical march bit for bit; a shrunk map that
    satisfies the precondition converges where the unshrunk one diverges."""
    from atlas.defect_correction import shrink                        # noqa: PLC0415
    a, b, w_star = _linear(seed=6, radius=0.9)
    n = len(b)
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a_bad = q @ np.diag(np.full(n, 0.999)) @ q.T
    phi = lambda w: a @ w + b                                            # noqa: E731
    one = defect_correct(phi, shrink(lambda w: a_bad @ w, 1.0), np.zeros_like(b),
                         r_stop=1e-9, k_max=5000, m_max=3, norm=_norm, fallback=False)
    march = classical_march(phi, np.zeros_like(b), r_stop=1e-9, norm=_norm)
    assert one.phi_calls == march.phi_calls
    assert np.array_equal(one.state, march.state)
    assert _propagator_radius(a, a_bad) > 1.0
    found = None
    for alpha in (0.3, 0.5, 0.7, 0.9):
        if _propagator_radius(a, (1.0 - alpha) * a_bad) < 1.0:
            found = alpha
            break
    assert found is not None
    res = defect_correct(phi, shrink(lambda w: a_bad @ w, found), np.zeros_like(b),
                         r_stop=1e-10, k_max=400, m_max=400, inner_frac=1e-3,
                         norm=_norm, fallback=False)
    assert res.status == "converged"
    assert np.allclose(res.state, w_star, atol=1e-8)


def test_vector_rms_is_the_rms_of_the_vector_magnitude():
    u = np.array([[3.0, 0.0], [0.0, 3.0]])
    v = np.array([[4.0, 0.0], [0.0, 4.0]])
    assert vector_rms(np.stack([u, v])) == pytest.approx(np.sqrt(np.mean(u * u + v * v)))


def test_break_even_and_the_exchange_rate_read_the_same_way():
    class _R:
        phi_calls, psi_calls = 10, 100

        def cost(self, c):
            return self.phi_calls + c * self.psi_calls

    be = break_even(100, _R(), 0.25)
    assert be["corrected_cost"] == pytest.approx(35.0)
    assert be["saving_ratio"] == pytest.approx(100.0 / 35.0)
    # a cheap map that halves the log-rate at cost multiplier one does not pay
    assert exchange_rate(0.9, 0.81, 0.0, 0.0) == pytest.approx(2.0)
    assert exchange_rate(0.9, 0.81, 1.0, 0.0) == pytest.approx(1.0)
    assert np.isnan(exchange_rate(1.0, 0.5, 0.1, 1.0))


def test_theta_from_a_march_can_under_bound_another_approach():
    """W208's mechanism, in two dimensions and with no fluid in it.

    ``theta_from_march`` reads the constant off the middle half of ONE march, so it
    sees only the direction that march took.  Here phi = diag(0.6, 0.95): the true
    constant is 1/(1-0.95) = 20 along the slow eigenvector and 1/(1-0.6) = 2.5 along
    the fast one.  A march started almost entirely in the fast mode still has the
    fast mode dominant across its whole middle half, so the reading is 2.5 -- and a
    state sitting on the slow eigenvector then lands OUTSIDE the certificate that
    reading issues, by the ratio of the two constants.

    This is the failure the wake array showed at twelve windows.  Pinning it here
    says it is a property of the estimator rather than of that problem.
    """
    lam_fast, lam_slow = 0.6, 0.95
    A = np.diag([lam_fast, lam_slow])

    # a march that stays in the fast mode for its whole middle half
    w = np.array([1.0, 1e-9])
    errors, residuals = [], []
    for _ in range(40):
        errors.append(float(np.linalg.norm(w)))              # w_star is the origin
        residuals.append(float(np.linalg.norm(A @ w - w)))
        w = A @ w
    read = theta_from_march(errors, residuals)

    assert read["theta"] == pytest.approx(1.0 / (1.0 - lam_fast), rel=1e-6)
    assert read["theta"] < 1.0 / (1.0 - lam_slow)            # the true constant

    # a state approached along the slow eigenvector, at the same kind of residual
    v = np.array([0.0, 1e-3])
    r_v = float(np.linalg.norm(A @ v - v))
    err_v = float(np.linalg.norm(v))
    assert err_v / r_v == pytest.approx(1.0 / (1.0 - lam_slow), rel=1e-6)

    # the certificate that march issues is violated, and by the ratio of constants
    bound = certificate(r_v, read["theta"])
    assert err_v > bound
    assert err_v / bound == pytest.approx(
        (1.0 - lam_fast) / (1.0 - lam_slow), rel=1e-6)

    # the reading bounds every iterate of the WINDOW it was read from ...
    n = len(errors)
    lo, hi = n // 4, (3 * n) // 4
    for e, r in zip(errors[lo:hi], residuals[lo:hi]):
        assert e <= certificate(r, read["theta"]) * (1.0 + 1e-9)

    # ... and not the iterates past it.  The slow mode is still growing its share,
    # so the required constant rises along the very march the reading came from --
    # the same drift, at 1e-7 here and at 14% on the wake array at twelve windows.
    tail_ratio = errors[-1] / residuals[-1]
    assert tail_ratio > read["theta"]
    assert errors[-1] > certificate(residuals[-1], read["theta"])
