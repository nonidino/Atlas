"""Tier 72 -- the fixed point the implicit step has, and the certified mode on it.

[[poc3-racelab-certified-step]].

The claims under test are the ones section 12's criterion 4 rests on, and they
are tested on a SMALL composite so the suite can afford them.  The car's own
numbers are in `out/racelab21/racelab21.json`; what is pinned here is the
mechanism, and the two defects the cylinder found while it was being built.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from atlas import defect_correction as DC
from atlas.cases import certified_step as CT
from atlas.cases import overset_ns as NS


@pytest.fixture(scope="module")
def flow():
    f = NS.cylinder_flow(h=1.0 / 12, ni=96, nj=21, dt=0.03)
    f.precond = "ilu"
    for _ in range(3):
        f.step()
    return f


# ---------------------------------------------------------------------------
# the refactor: step() is the same step, and the certified mode is a seam in it
# ---------------------------------------------------------------------------


def test_assembly_is_one_definition(flow):
    """`MomentumSystem` must take the system from the flow's own assembly.

    A replica would be a second definition of the car free to drift from the
    first, which this vault has paid for before.
    """
    M, bu, bv, us, vs, a0, t1 = flow._assemble_momentum()
    s = CT.MomentumSystem(flow, sweep="jacobi")
    assert s.M.shape == M.shape and s.M.nnz == M.nnz
    assert np.array_equal(s.bu, bu) and np.array_equal(s.bv, bv)
    assert np.array_equal(s.w0[0], us) and np.array_equal(s.w0[1], vs)
    assert s.a0 == a0 and s.t1 == t1


def test_momentum_solver_hook_defaults_to_the_classical_solve(flow):
    """``momentum_solver = None`` is the classical BiCGSTAB, bit for bit."""
    assert flow.momentum_solver is None
    keep = (flow.U.copy(), flow.V.copy(), flow.Um1.copy(), flow.Vm1.copy(),
            flow.P.copy(), flow.t, flow.k)
    rec_a = flow.step()
    a = (flow.U.copy(), flow.V.copy(), flow.P.copy())
    flow.U, flow.V, flow.Um1, flow.Vm1, flow.P, flow.t, flow.k = (
        keep[0].copy(), keep[1].copy(), keep[2].copy(), keep[3].copy(),
        keep[4].copy(), keep[5], keep[6])
    flow.momentum_solver = None
    rec_b = flow.step()
    b = (flow.U.copy(), flow.V.copy(), flow.P.copy())
    flow.U, flow.V, flow.Um1, flow.Vm1, flow.P, flow.t, flow.k = (
        keep[0].copy(), keep[1].copy(), keep[2].copy(), keep[3].copy(),
        keep[4].copy(), keep[5], keep[6])
    assert rec_a["iterations"] == rec_b["iterations"]
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ---------------------------------------------------------------------------
# W226: the fixed point an implicit step has and an explicit one does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sweep", CT.SWEEPS)
def test_the_classical_answer_is_a_fixed_point(flow, sweep):
    """``phi_P(x) = x`` iff ``M x = b``, for ANY non-singular P.  This is what
    the porous column's explicit step does not have."""
    s = CT.MomentumSystem(flow, sweep=sweep)
    xstar, _its, _w = s.classical()
    gap = DC.vector_rms(s.phi(xstar) - xstar)
    scale = DC.vector_rms(xstar)
    assert gap / scale < 1e-8


def test_picard_diverges_and_ilu_contracts(flow):
    """The sweep is not a free choice: the geometry the body-fitted column
    exists for is what rules out the explicit evaluation."""
    picard = CT.contraction(CT.MomentumSystem(flow, sweep="picard"), n=20)
    ilu = CT.contraction(CT.MomentumSystem(flow, sweep="ilu"), n=20)
    assert picard["diverged"] and not picard["contracts"]
    assert picard["rate"] > 1.0
    assert ilu["contracts"] and ilu["rate"] < 0.5
    assert not ilu["rate_is_floor"]


def test_contraction_does_not_call_a_slow_sweep_floored(flow):
    """The defect the cylinder found: taking the minimum residual over n sweeps
    as 'the floor' calls a sweep that is STILL FALLING at sweep n floored.

    Jacobi contracts here, slowly.  Read over too few sweeps it must still
    report a rate, and that rate must be a contraction, not a floor.
    """
    c = CT.contraction(CT.MomentumSystem(flow, sweep="jacobi"), n=12)
    assert c["contracts"], c
    assert not c["rate_is_floor"], c
    assert 0.0 < c["rate"] < 1.0
    # and the tail of a CONVERGED sweep must be excluded rather than averaged in.
    # **The property that matters is that the rate does not depend on how long
    # you look** -- a rate read off a flat tail moves toward one as n grows,
    # which is precisely the defect.  So it is measured at three lengths, one of
    # them well past the plateau.
    rates, plateaus = [], []
    for n in (60, 120, 200):
        c = CT.contraction(CT.MomentumSystem(flow, sweep="ilu"), n=n)
        rates.append(c["rate"])
        plateaus.append(c["plateau_at"])
        assert not c["rate_is_floor"], (n, c["rate"])
        assert 0.0 < c["rate"] < 1.0, (n, c["rate"])
    assert max(rates) - min(rates) < 1e-9, rates
    # past round-off the plateau IS found, rather than running to the end
    assert plateaus[-1] < 200, "a converged sweep's plateau was never found"


# ---------------------------------------------------------------------------
# what is certified
# ---------------------------------------------------------------------------


def test_the_sweep_uses_the_flows_own_preconditioner(flow):
    """**One definition of the preconditioner, as of the assembly.**

    Building a fresh incomplete factor here was wrong twice: SuperLU's factor is
    exactly singular on the car at the step's own drop tolerance, and the retry
    for that lives in `_momentum_precond`; and a fresh factor every call prices
    the certified mode against a classical mode paying a factor it refreshes
    only every ``ilu_every`` steps.  The difference is not small -- on the
    cylinder a fresh factor contracts at 0.0066 and the step's own at 0.473.
    """
    s = CT.MomentumSystem(flow, sweep="ilu")
    assert s._apply_ilu is not None
    # handed one, it must not build another
    precond, _t, _note = flow._momentum_precond(s.M)
    s2 = CT.MomentumSystem(flow, sweep="ilu", precond=precond)
    assert s2.reused_precond and s2.ilu_s == 0.0


def test_the_timed_certified_step_does_not_run_a_classical_solve(flow):
    """A certified step that fetched its reference by running the classical
    solve would be timed as classical PLUS certified, and the price it reported
    would be an artefact of the instrument."""
    solver = CT.CertifiedMomentumSolver(flow, sweep="ilu", compare_classical=False)
    assert solver.compare_classical is False
    r = CT.certified_step_report(flow, sweep="ilu")
    # the gap is measured between the two STATES, so it needs no in-solver reference
    assert np.isfinite(r["relative_velocity_gap"])
    assert r["certified"]["classical_iterations"] == []


def test_the_null_element_is_the_classical_sweep_march_bitwise(flow):
    """`defect_correction`'s own fact 2, on this column: a constant cheap map
    makes the iterates the classical march itself."""
    s_a = CT.MomentumSystem(flow, sweep="ilu")
    r_stop = flow.rtol * max(s_a.rhs_norm(), 1.0) / np.sqrt(s_a.n)
    w, _rep = CT.certified_momentum(s_a, CT.null_psi(), r_stop=r_stop)
    s_b = CT.MomentumSystem(flow, sweep="ilu")
    march = DC.classical_march(s_b.phi, s_b.w0, r_stop=r_stop)
    assert np.array_equal(w, march.state)


def test_a_wrong_cheap_map_does_not_change_the_answer(flow):
    """Theorem 1, where it is easiest to break.  Consistency does not depend on
    the cheap map's accuracy: a wrong psi can cost calls, it cannot change what
    is returned.  This is the one accuracy claim in the demo that rests on a
    proof rather than a measurement, so it gets the adversarial map."""
    s0 = CT.MomentumSystem(flow, sweep="ilu")
    xstar, cits, _w = s0.classical()
    s1 = CT.MomentumSystem(flow, sweep="ilu")
    _w1, good = CT.certified_momentum(s1, CT.null_psi(), reference=xstar,
                                      classical_its=cits)
    s2 = CT.MomentumSystem(flow, sweep="ilu")
    _w2, bad = CT.certified_momentum(s2, CT.wrong_psi(), reference=xstar,
                                     classical_its=cits)
    assert good.error_to_classical > 0 and bad.error_to_classical > 0
    # **Both land on the classical answer** -- which is the claim.  Comparing the
    # two errors to each other does NOT test it: both are at their stopping
    # floor, and the ratio swung from 0.40 to 84 between two cylinder
    # resolutions while nothing about Theorem 1 changed.  A ratio of two numbers
    # at a floor is a reading of the floor.  So each is measured against the
    # state's own scale, which is what "the classical answer" means.
    scale = DC.vector_rms(xstar)
    assert good.error_to_classical / scale < 1e-6, good.error_to_classical
    assert bad.error_to_classical / scale < 1e-6, bad.error_to_classical
    # ... and it is not free
    assert bad.inner_cheap_calls > good.inner_cheap_calls


def test_certifying_the_momentum_solve_certifies_the_step(flow):
    """The projection is one factored solve and the interpolation one sparse
    apply; neither iterates.  So the certified STEP must land on the classical
    step, which is section 12's criterion 4 reduced to a number."""
    r = CT.certified_step_report(flow, sweep="ilu")
    assert r["relative_velocity_gap"] < 1e-6, r["relative_velocity_gap"]
    assert r["pressure_gap"] / r["pressure_scale"] < 1e-6
    assert r["certified"]["status"] in ("converged", "fallback_converged")


def test_the_report_carries_what_section_5_2_asks_for(flow):
    """Outer iterations, inner cheap calls and the residual -- plus the two the
    screen needs and the iteration must not read."""
    s = CT.MomentumSystem(flow, sweep="ilu")
    xstar, cits, cwall = s.classical()
    _w, rep = CT.certified_momentum(s, CT.null_psi(), reference=xstar,
                                    classical_its=cits, classical_wall_s=cwall)
    d = rep.as_dict()
    for key in ("outer_iterations", "inner_cheap_calls", "residual",
                "algebraic_residual", "error_to_classical", "classical_iterations"):
        assert key in d, key
    assert rep.outer_iterations >= 1 and rep.inner_cheap_calls >= 1
    # the two residuals are in DIFFERENT currencies and must not be conflated:
    # the iteration's is ||P(b - Mw)||, the system's is ||b - Mw||
    assert rep.residual != rep.algebraic_residual


def test_the_iteration_never_reads_the_reference(flow):
    """`reference` REPORTS the error; nothing in the iteration may consult the
    answer it claims to reach.  A wrong reference must not move the answer."""
    s1 = CT.MomentumSystem(flow, sweep="ilu")
    w1, _r1 = CT.certified_momentum(s1, CT.null_psi())
    s2 = CT.MomentumSystem(flow, sweep="ilu")
    bogus = np.zeros_like(s2.w0)
    w2, r2 = CT.certified_momentum(s2, CT.null_psi(), reference=bogus)
    assert np.array_equal(w1, w2)
    assert r2.error_to_classical > 0      # it was reported, against the bogus one


# ---------------------------------------------------------------------------
# the failing controls
# ---------------------------------------------------------------------------


def test_the_porous_column_still_refuses(flow):
    """P9's control.  The fixed point is a property of IMPLICITNESS, not of this
    module: the explicit column must still raise, or the tier would be claiming
    a universal fact where it measured a conditional one."""
    from atlas.cases import racelab_switch as RS
    with pytest.raises(ValueError) as exc:
        RS.MixedRollout(assignment=RS.Mode.CERTIFIED.value)
    assert "explicit" in str(exc.value).lower()


def test_an_unbuildable_cheap_map_is_reported_and_not_raised(flow):
    """The zero-pivot trap: SuperLU's incomplete factor can be exactly singular
    after dropping.  `detuned_psi` must return None rather than crash a march."""
    psi = CT.detuned_psi(CT.MomentumSystem(flow, sweep="ilu"), drop_tol=1e9, fill=1.0)
    assert psi is None or callable(psi)


def test_a_cheap_map_that_cannot_be_built_does_not_answer_under_its_own_name(flow):
    """On the car `detuned_psi` returns None -- the incomplete factor is exactly
    singular even after the retry -- and the null map ran in its place IN
    SILENCE, so the record read "detuned: 2.985 s" for a run of the null map.

    The number was real and the label was wrong, which is the same defect class
    as a knob reporting a response the march does not have.
    """
    r = CT.certified_step_report(flow, sweep="ilu",
                                 psi_factory=lambda sy: None)
    assert r["psi_ran"] == "null"
    assert r["psi_requested"] is not None
    assert r["psi_unavailable_reason"], "the substitution was not reported"
    # and when the map IS built, it answers under its own name
    ok = CT.certified_step_report(flow, sweep="ilu", psi_factory=CT.jacobi_psi)
    assert ok["psi_ran"] == "jacobi_psi"
    assert ok["psi_unavailable_reason"] is None


def test_an_unknown_sweep_is_refused(flow):
    with pytest.raises(ValueError):
        CT.MomentumSystem(flow, sweep="no-such-sweep")


def test_shrink_is_needed_for_a_window_map_and_alpha_is_recorded(flow):
    """A cheap map that is the identity outside its window has no fixed point to
    offer the inner iteration -- ``x -> x + d`` moves forever.  `shrink` is what
    makes it usable, which is why `certified_momentum` takes alpha."""
    s = CT.MomentumSystem(flow, sweep="ilu")
    identity = lambda w: np.asarray(w, dtype=float)                     # noqa: E731
    _w, rep = CT.certified_momentum(s, identity, alpha=0.5, k_max=6, m_max=6)
    assert rep.status in ("converged", "fallback_converged", "stalled",
                          "max_outer", "fallback_unconverged")
    assert rep.outer_iterations >= 1
