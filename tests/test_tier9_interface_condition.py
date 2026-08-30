"""Tier 9: W7/W17 -- the interface condition an explicit one-step map poses.

`tier0-measurements` section 5 found that flux balance is not the condition an
explicitly stepped agent satisfies, and section 5.1 promoted W7 (R9's
time-integrated flux) and W17 (waveform relaxation) as the fix.  **Built and
measured 2026-08-28, the fix does not work, and this file holds the algebra that
says why.**

At a shared-layer seam both sides pin the same cell, so with outward normals

    F_A + F_B  =  nu (2 w_G - w_A,in - w_B,in) / h  =  -nu h d2w/dn2

which is a discrete SECOND DERIVATIVE of the state, not a jump.  It is O(h) on
the exact solution and vanishes only where that solution is linear across the
seam.  A spatial defect cannot be removed by integrating in time, which is
exactly what the measurement shows: the reference-trace check fails identically
under the pointwise and the time-integrated convention, at every dt tried.

Written against the algebra, so it runs without the build repo.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.probe import (
    ReferenceTraceCheck,
    reference_trace_check,
    shared_layer_flux_sum,
)

NU, H = 1.0 / 255.0, 2.0 / 128.0


# ---------------------------------------------------------------------------
# the identity: flux balance at a shared layer is a second difference
# ---------------------------------------------------------------------------


def _sample(f, j0=40, n=64):
    """w on the seam column and its two neighbours, for a smooth profile f(x)."""
    x = (np.arange(n) + 0.5) * H
    return f(x), f(x - H), f(x + H)


def test_the_shared_layer_flux_sum_is_minus_nu_h_times_the_second_difference():
    """The identity, on a curved profile. Verified bit-exactly against
    `reference.WindowNS`'s own monolith on all four seams (section 9)."""
    for f in (np.sin, np.cos, lambda x: x ** 2, lambda x: np.exp(-x)):
        w, wa, wb = _sample(f)
        got = shared_layer_flux_sum(w, wa, wb, NU, H)
        d2 = (wa - 2 * w + wb) / H ** 2
        assert np.allclose(got, -NU * H * d2, rtol=0, atol=1e-15)


def test_it_is_zero_exactly_when_the_field_is_linear_across_the_seam():
    """So driving it to zero asks the solution to be straight at every seam."""
    w, wa, wb = _sample(lambda x: 3.0 * x - 1.25)
    assert np.abs(shared_layer_flux_sum(w, wa, wb, NU, H)).max() < 1e-15

    w, wa, wb = _sample(lambda x: x ** 2)
    assert np.abs(shared_layer_flux_sum(w, wa, wb, NU, H)).max() > 1e-9


def test_the_residual_on_the_exact_solution_is_first_order_in_h_not_zero():
    """O(h), so refining the MESH removes it and refining dt does not.

    That is the whole reason time-integration cannot rescue the condition: the
    defect is spatial. Halving h halves the residual.
    """
    x = np.linspace(0.2, 1.8, 64)          # the same PHYSICAL points at every h

    def resid(h):
        w, wa, wb = np.sin(x), np.sin(x - h), np.sin(x + h)
        return float(np.abs(shared_layer_flux_sum(w, wa, wb, NU, h)).max())

    assert resid(H) / resid(H / 2) == pytest.approx(2.0, rel=0.01)
    assert resid(H) / resid(H / 4) == pytest.approx(4.0, rel=0.01)
    assert resid(H) > 0.0


def test_the_second_difference_is_independent_of_any_time_step():
    """It is a function of the STATE. Measured on the real solver: the residual
    moved 2.21e-5 -> 3.31e-5 across a 25x range of dt while the true trace move
    fell 7.57e-5 -> 1.53e-5, i.e. proportionally to dt."""
    w, wa, wb = _sample(np.sin)
    a = shared_layer_flux_sum(w, wa, wb, NU, H)
    b = shared_layer_flux_sum(w, wa, wb, NU, H)      # nothing here takes a dt
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# W46's named measurement
# ---------------------------------------------------------------------------


def test_the_reference_trace_check_passes_a_condition_the_truth_satisfies():
    """A residual that IS minimized at the reference trace passes."""
    chk = reference_trace_check(lambda t: abs(t - 1.0), lagged_trace=0.0,
                                reference_trace=1.0)
    assert chk.passes and chk.ratio == 0.0


def test_the_reference_trace_check_fails_flux_balance_as_measured():
    """The measured numbers, as a regression.

    exposed agent 1.002, embedded 1.028-1.040, under BOTH conventions and at
    dt = 0.05, 0.01 and 0.002. Every one is above 1, i.e. substituting the
    reference's own trace makes the residual WORSE.
    """
    measured = {
        ("exposed", 0.05, "pointwise"): (2.7593e-03, 2.7639e-03),
        ("exposed", 0.05, "integrated"): (1.6376e-03, 1.6403e-03),
        ("embedded", 0.05, "pointwise"): (2.2053e-05, 2.2945e-05),
        ("embedded", 0.05, "integrated"): (2.0316e-05, 2.0889e-05),
        ("embedded", 0.01, "pointwise"): (2.4812e-05, 2.4929e-05),
        ("embedded", 0.01, "integrated"): (2.0289e-05, 2.0389e-05),
        ("embedded", 0.002, "pointwise"): (3.3058e-05, 3.3212e-05),
        ("embedded", 0.002, "integrated"): (2.3942e-05, 2.4062e-05),
    }
    for key, (lag, ref) in measured.items():
        chk = ReferenceTraceCheck(lagged=lag, reference=ref)
        assert not chk.passes, f"{key} must FAIL the check"
        assert chk.ratio > 1.0

    integrated = [k for k in measured if k[2] == "integrated"]
    pointwise = [k for k in measured if k[2] == "pointwise"]
    assert len(integrated) == len(pointwise) == 4
    for i, p in zip(sorted(integrated), sorted(pointwise)):
        ri = ReferenceTraceCheck(*measured[i]).ratio
        rp = ReferenceTraceCheck(*measured[p]).ratio
        # time-integrating does not change the verdict, and barely the number
        assert abs(ri - rp) < 0.02, (i, p, ri, rp)


def test_the_check_reports_rather_than_raises_on_a_zero_baseline():
    chk = ReferenceTraceCheck(lagged=0.0, reference=1e-9)
    assert chk.ratio == float("inf")
    assert not chk.passes


def test_the_check_serializes_with_its_verdict():
    chk = reference_trace_check(lambda t: abs(t), 1.0, 2.0, solved_trace=0.5,
                                note="flux balance, pointwise")
    d = chk.as_dict()
    assert d["passes"] is False and d["solved"] == 0.5
    assert d["note"] == "flux balance, pointwise"
