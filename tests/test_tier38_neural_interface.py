"""Tier 38 -- CS-S1, the learned operator on the interface.

`atlas/cases/neural_interface.py`.  The hybrid built and its two halves
separated: the half that is safe by construction, which is measured here rather
than argued, and the half that was supposed to be fast, which is not.

  * **the exchange** -- the unknown is the ring at ``t + dt``, the shipped
    scheme is its zeroth iterate, and a domain-boundary face is never iterated;
  * **the safety claim** -- five starts, one fixed point, including from a
    deliberately wrong start; and the predictor appears in no capability record,
    with a negative control that shows the check can fail;
  * **the ceiling** -- the one-sweep predictor saves exactly one sweep, because
    it IS one sweep, and that bounds the whole class;
  * **the rate lever** -- what `elliptic_subsolve` does to the contraction, with
    the projection separated out as its own control, and the arrangement with
    the slow rate being the one `L2/R10` refuses;
  * **alpha_star** -- the vault's unused optimal-Robin field, used; it diverges
    undamped exactly when its preconditioned spectral radius exceeds two;
  * **the halo** -- R10's rule as the ITERATED scheme's convergence condition;
  * **the multiplier space** -- why the first arrangement measured every field
    predictor as negative, which is a fact about the unknown and not about
    Poseidon.

`reference.WindowNS` lives in the build repo, so every test that steps one is
skipped without it.  Nothing here needs torch: the checkpoint arm is measured by
`scripts/w166_neural_interface.py`, not asserted in the suite, because a test
that loads a 20.8M-parameter checkpoint is not a test.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import Budget, compile_scheme                           # noqa: E402
from atlas.cases import neural_interface as NI                     # noqa: E402
from atlas.cases import window_ns as W                             # noqa: E402
from atlas.transfer import InterfaceSpace                          # noqa: E402
from atlas.verdict import Verdict                                  # noqa: E402


def _have_expert() -> bool:
    try:
        W.load_reference()
        NI.load_state()
    except Exception:                                              # noqa: BLE001
        return False
    return True


needs_expert = pytest.mark.skipif(
    not _have_expert(),
    reason="the windfarm reference package or out/tier0b/s0_state.npz is missing")


@pytest.fixture(scope="module")
def state():
    return NI.load_state()


@pytest.fixture(scope="module")
def split(state):
    """The arrangement the compiler admits: elliptic exposed, projection global."""
    return NI.SchwarzExchange(state[0], state[1], expose_elliptic=True, project=True)


@pytest.fixture(scope="module")
def split_star(split):
    su, sv, _ = split.fixed_point(cap=60, tol=1e-13)
    return su, sv


@pytest.fixture(scope="module")
def as_built(state):
    """The agent as it is, which `L2/R10` refuses.  Kept as the contrast."""
    return NI.SchwarzExchange(state[0], state[1], expose_elliptic=False, project=False)


# ---------------------------------------------------------------------------
# 1. the exchange
# ---------------------------------------------------------------------------


@needs_expert
def test_the_shipped_scheme_is_this_iterations_ZEROTH_iterate(split):
    """One sweep from the cold start is exactly ``step_batch(bc1=None)``.

    The whole study rests on this being the same object the composed rollout
    already runs.  ``bc1=None`` holds ``bc0`` across the macro-step, and the cold
    start writes the old ring into ``bc1``, so the two must agree to the bit --
    and if they did not, every number here would be about a scheme nobody ships.
    """
    got_u, got_v = split.sweep(split.u0, split.v0)

    ou, ov = [], []
    for k in range(len(split.tiling.names)):
        u1, v1 = split._solvers[k].step_batch(
            split.us[k][None], split.vs[k][None], split.dt,
            bc0=(split.us[k][None], split.vs[k][None]), bc1=None)
        ou.append(u1[0])
        ov.append(v1[0])
    want_u, want_v = split.tiling.assemble(np.stack(ou), np.stack(ov))
    from atlas.cases import wake_array as WA
    want_u, want_v = WA.project_assembled(want_u, want_v)

    assert np.array_equal(got_u, want_u)
    assert np.array_equal(got_v, want_v)


@needs_expert
def test_a_domain_boundary_face_is_never_iterated(split):
    """A face that is the domain edge carries a real boundary condition.

    The monolith holds its own ring there, so this iteration holds it too, and
    only on the faces `artificial_faces` does not name.  Measured by wrecking the
    iterate along the whole domain edge and checking the sweep does not notice:
    if it did, the fixed point would be a property of the tiling rather than of
    the problem.
    """
    base = split.sweep(split.u0, split.v0)
    gu, gv = split.u0.copy(), split.v0.copy()
    gu[0, :] += 7.0
    gu[-1, :] += 7.0
    gu[:, 0] += 7.0
    gu[:, -1] += 7.0
    gv[0, :] -= 3.0
    wrecked = split.sweep(gu, gv)
    assert np.array_equal(base[0], wrecked[0])
    assert np.array_equal(base[1], wrecked[1])


@needs_expert
def test_an_artificial_face_IS_iterated(split):
    """The control for the test above: perturbing an interior seam DOES move it.

    Without this, "the sweep ignored my perturbation" is equally consistent with
    a sweep that ignores its argument entirely.
    """
    base = split.sweep(split.u0, split.v0)
    gu, gv = split.u0.copy(), split.v0.copy()
    ox = split.tiling.offsets[0][0]
    gu[:, ox + split.tiling.n - 1] += 0.05          # C00's xhi ring
    moved = split.sweep(gu, gv)
    assert not np.array_equal(base[0], moved[0])


# ---------------------------------------------------------------------------
# 2. the safety claim -- the half that works
# ---------------------------------------------------------------------------


@needs_expert
def test_the_fixed_point_does_not_depend_on_where_the_iteration_starts(
        split, split_star):
    """**The claim the whole arrangement is for.**

    A predictor is a preconditioner, so a wrong guess costs iterations and not
    correctness.  Three starts that differ by more than the tolerance, marched to
    the same tolerance, must land in the same place -- and the disagreement is
    only meaningful beside the tolerance, so both are asserted.
    """
    d_cold = NI._rms(split.u0 - split_star[0], split.v0 - split_star[1])
    tol = 1e-3 * d_cold
    rows = [
        NI.predictor_skill(split, NI.HoldOldRing(), split_star, tol, cap=200),
        NI.predictor_skill(split, NI.OneSweep(split.tiling), split_star, tol, cap=200),
        NI.predictor_skill(split, NI.AmplitudeMatchedNoise(amplitude=10 * d_cold),
                           split_star, tol, cap=200),
    ]
    # the starts really were apart, or the agreement below means nothing
    assert max(r["d0"] for r in rows) > 5.0 * min(r["d0"] for r in rows)
    ind = NI.start_independence(rows)
    assert ind["arms"] == 3
    # 2 * tol and not tol: each arm stops within tol of the limit, so two of them
    # are 2 * tol apart at worst by the triangle inequality. Asserting tol would
    # be asserting something the stopping rule does not promise.
    assert ind["worst_pair_disagreement"] <= 2.0 * tol


@needs_expert
def test_a_deliberately_wrong_start_converges_to_the_same_answer(
        split, split_star):
    """The strongest form: a start that is pure noise, ten times the size of the
    thing being solved for, still lands on the fixed point.

    This is what makes the predictor exempt from a substitution certificate.  It
    is not a convention about what counts as an agent -- it is a measurement that
    the predictor's output cannot reach the answer.
    """
    d_cold = NI._rms(split.u0 - split_star[0], split.v0 - split_star[1])
    r = NI.predictor_skill(
        split, NI.AmplitudeMatchedNoise(amplitude=10 * d_cold),
        split_star, 1e-4 * d_cold, cap=200)
    assert not r["diverged"]
    assert r["skill"] < 0.0                      # it is genuinely a bad start
    assert r["final_distance"] <= 1e-4 * d_cold


@needs_expert
def test_the_predictor_appears_in_no_capability_record(state):
    graph, _ = NI.build(state[0], state[1])
    for p in (NI.HoldOldRing(), NI.OneSweep(), NI.Neural(),
              NI.AmplitudeMatchedNoise(amplitude=0.0)):
        assert NI.predictor_is_not_an_agent(graph, p), p.name


@needs_expert
def test_the_not_an_agent_check_can_FAIL(state):
    """The negative control.  A check that cannot fail is not a check.

    `Neural` is constructed above without touching the checkpoint, which is the
    other half of what this test protects: if the constructor loaded 20.8M
    parameters, nobody would run the check.
    """
    graph, _ = NI.build(state[0], state[1])

    class Impostor(NI.Predictor):
        name = "C00"                               # an agent id on this graph

        def initial(self, ex):
            return ex.u0, ex.v0

    assert NI.predictor_is_not_an_agent(graph, Impostor()) is False
    assert NI.Neural()._expert is None             # never loaded


# ---------------------------------------------------------------------------
# 3. the ceiling on any field predictor
# ---------------------------------------------------------------------------


@needs_expert
def test_the_one_sweep_predictor_saves_exactly_one_sweep(split, split_star):
    """**The harness's own control, and the ceiling for the whole class.**

    `OneSweep` is the first iterate, built from the very solver that owns the
    subdomains, so it is about as good as a prediction of the new field can be.
    It must therefore save one sweep and no more.  A harness in which it saved
    two would be double-counting; one in which it saved none would not be
    measuring the start at all.
    """
    d_cold = NI._rms(split.u0 - split_star[0], split.v0 - split_star[1])
    tol = 1e-6 * d_cold
    cold = NI.predictor_skill(split, NI.HoldOldRing(), split_star, tol, cap=200)
    one = NI.predictor_skill(split, NI.OneSweep(split.tiling), split_star, tol,
                             cap=200)
    assert cold["sweeps"] - one["sweeps"] == 1
    # and it costs exactly the sweep it saves, so it can never pay for itself
    assert one["total_solves"] == pytest.approx(cold["total_solves"])


@needs_expert
def test_the_oracle_costs_one_sweep_and_the_noise_control_does_not_beat_cold(
        split, split_star):
    d_cold = NI._rms(split.u0 - split_star[0], split.v0 - split_star[1])
    tol = 1e-3 * d_cold
    oracle = NI.predictor_skill(split, NI.Oracle(split_star), split_star, tol,
                                cap=50)
    assert oracle["sweeps"] == 1                  # the sweep that shows it is one
    assert oracle["skill"] == pytest.approx(1.0)
    noise = NI.predictor_skill(
        split, NI.AmplitudeMatchedNoise(amplitude=d_cold), split_star, tol, cap=200)
    assert noise["skill"] < 0.0
    assert noise["sweeps"] >= NI.predictor_skill(
        split, NI.HoldOldRing(), split_star, tol, cap=200)["sweeps"]


# ---------------------------------------------------------------------------
# 4. the rate lever: what one declared field does to the coupling
# ---------------------------------------------------------------------------


@needs_expert
def test_exposing_the_elliptic_part_changes_the_contraction_by_two_orders(
        split, as_built):
    """`elliptic_subsolve` is a word on a capability record and it is worth ~150x.

    R10 refuses an embedded pressure solve because the composed ERROR is
    elliptic.  Measured here as a RATE: the same four windows, the same overlap,
    the same state, the same tolerance -- and the contraction goes from ~0.97 per
    sweep to ~0.01.
    """
    a = split.march(split.u0, split.v0, cap=60, tol=1e-13)
    b = as_built.march(as_built.u0, as_built.v0, cap=120, tol=1e-13)
    assert a["contraction"] < 0.05
    assert b["contraction"] > 0.9
    per_decade = lambda c: np.log(10.0) / abs(np.log(c))
    assert per_decade(b["contraction"]) / per_decade(a["contraction"]) > 50.0


@needs_expert
def test_the_global_projection_is_NOT_what_changes_the_rate(state):
    """The control that attributes the 150x to the exposure rather than the pair.

    W100 added two things at once -- take the pressure solve out of the agent,
    and put one global projection in the composition layer.  For the composed
    ERROR both are needed.  For the iteration's RATE only the first is: with the
    projection switched off the contraction is the same to within a factor of two.
    """
    a = NI.SchwarzExchange(state[0], state[1], expose_elliptic=True, project=True)
    b = NI.SchwarzExchange(state[0], state[1], expose_elliptic=True, project=False)
    with_p = a.march(a.u0, a.v0, cap=60, tol=1e-13)
    without = b.march(b.u0, b.v0, cap=60, tol=1e-13)
    assert 0.5 < with_p["contraction"] / without["contraction"] < 2.0


@needs_expert
def test_the_arrangement_with_the_slow_rate_is_the_one_R10_REFUSES(state):
    """The compiler picks the fast one out of the declaration, before any sweep.

    This is the whole argument for the refusal restated in a currency a
    practitioner acts on: `L2/R10` is not only about the composed error, it is
    about how expensive the coupling is, and the compile knows which is which
    from `elliptic_subsolve` alone.
    """
    slow, _ = NI.build(state[0], state[1], expose_elliptic=False)
    fast, _ = NI.build(state[0], state[1], expose_elliptic=True)
    r_slow = compile_scheme(slow, budget=Budget(allow_probe=False))
    r_fast = compile_scheme(fast, budget=Budget(allow_probe=False))
    assert r_slow.verdict is Verdict.REFUSE
    assert r_fast.verdict is Verdict.ADMIT_UNCERTIFIED
    assert any(d.rule == "R10" for d in r_slow.decisions.refusals)


# ---------------------------------------------------------------------------
# 5. alpha_star, the vault's unused optimal-Robin field
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def as_built_probe(state):
    ex = NI.make_experts(state[0], state[1], expose_elliptic=False)
    seam = NI.MultiplierSeam(a=ex["C00"], b=ex["C10"])
    return seam, seam.probe()


@needs_expert
def test_alpha_star_is_the_diagonal_of_the_assembled_seam_operator(as_built_probe):
    seam, pr = as_built_probe
    assert np.allclose(pr["alpha_star"], np.diag(pr["S"]))
    assert pr["probe_calls"] == 2 * (seam.m_eff + 1)
    # and it is NOT a near-diagonal operator, which is why a diagonal
    # preconditioner buys a factor rather than a solve
    assert pr["off_diagonal_share"] > 0.3


@needs_expert
def test_the_undamped_preconditioner_diverges_and_its_spectral_radius_says_why(
        as_built_probe):
    """Richardson converges when the preconditioned spectrum sits inside (0, 2).

    ``alpha_star`` cuts kappa by 3x here and still diverges, because a condition
    number is not an iteration.  Both halves are asserted: the radius that
    predicts it, and the divergence it predicts.
    """
    seam, pr = as_built_probe
    assert pr["kappa_preconditioned"] < pr["kappa"] / 2.0
    assert pr["rho_D_inv_S"] > 2.0
    lad = NI.accelerator_ladder(seam, pr, tol_fracs=(1e-3,), cap=400)
    # "damped" is a substring of "undamped", so match on the whole tail. This
    # test failed once for exactly that reason and reported the undamped arm's
    # divergence as the damped arm's.
    undamped = next(r for r in lad if r["accelerator"].endswith("undamped"))
    damped = next(r for r in lad if r["accelerator"].endswith("star damped"))
    plain = next(r for r in lad if "theta" in r["accelerator"])
    assert undamped["converged"] is False
    assert damped["converged"] is True
    assert plain["converged"] is True            # compared against a real number
    assert damped["total_calls"] < plain["total_calls"] / 2.0


@needs_expert
def test_the_probed_S_beats_every_sweep_in_expert_calls(as_built_probe):
    """The vault's own claim, measured: **the probed S IS the missing coarse space.**

    `atlas-and-standard-dd-theory` 2.2 says a geometric coarse space is
    unavailable for a frozen expert and 3 says the probed Schur complement is the
    substitute, because it is dense across the interface and couples it in one
    solve.  Priced in expert calls INCLUDING its own 34-call probe, it wins.
    """
    seam, pr = as_built_probe
    lad = NI.accelerator_ladder(seam, pr, tol_fracs=(1e-3,), cap=400)
    newton = next(r for r in lad if "newton" in r["accelerator"])
    plain = next(r for r in lad if "theta" in r["accelerator"])
    assert newton["converged"] and plain["converged"]
    assert newton["total_calls"] < plain["total_calls"] / 2.0
    assert newton["total_calls"] < 2 * pr["probe_calls"]


# ---------------------------------------------------------------------------
# 6. the halo, as a convergence condition
# ---------------------------------------------------------------------------


@needs_expert
def test_the_EMBEDDED_scheme_diverges_below_the_declared_domain_of_dependence(state):
    """R10's halo rule, in a regime it was not derived for.

    The rule is an ACCURACY condition on a one-sweep scheme: a halo narrower than
    the agent's reach carries stale data.  Iterated, the same threshold decides
    whether the scheme converges at all -- the driver's eleven-row sweep brackets
    it between halo 18 and halo 20 against a declared reach of 20.  Two rows
    either side is what a test can afford.
    """
    rows = NI.halo_convergence(state[0], state[1], halos=(8, 24),
                               expose_elliptic=False, project=False, cap=40)
    by_halo = {r["halo"]: r for r in rows}
    assert by_halo[8]["converging"] is False
    assert by_halo[24]["converging"] is True
    assert by_halo[8]["covers_reach"] is False
    assert by_halo[24]["covers_reach"] is True
    thr = NI.halo_threshold(rows)
    assert thr["declared_reach"] == NI.DOMAIN_OF_DEPENDENCE == 20


@needs_expert
def test_and_the_EXPOSED_one_converges_at_every_overlap(state):
    """**The control, and it changes what the row above means.**

    With the elliptic part in the composition layer the iteration converges at a
    halo of 8 -- half the agent's reach -- as fast as at 24.  So the threshold
    above is not a property of the stencil: it is the embedded pressure solve
    making the overlap load-bearing, which is R10's own mechanism showing up as a
    convergence condition rather than as an error term.

    The overlap still decides ACCURACY in both arrangements.  That is
    `assembly.py`'s contaminated-cell weight and a different measurement.
    """
    rows = NI.halo_convergence(state[0], state[1], halos=(8, 24),
                               expose_elliptic=True, project=True, cap=40)
    by_halo = {r["halo"]: r for r in rows}
    assert by_halo[8]["converging"] is True
    assert by_halo[8]["covers_reach"] is False
    assert by_halo[8]["contraction"] < 0.1
    assert by_halo[24]["contraction"] < 0.1


# ---------------------------------------------------------------------------
# 7. why the multiplier arrangement measured every predictor as negative
# ---------------------------------------------------------------------------


@needs_expert
def test_the_multiplier_root_is_far_bigger_than_the_fields_own_change(state):
    """**The mechanism behind this study's negative result, and it is structural.**

    `solve.py`'s interface unknown is a common 16-mode datum imposed on BOTH
    sides' rings, whose root balances two outward fluxes measured 20 cells apart
    on an overlapping decomposition.  Its root is nothing like the physical ring
    update, so no predictor of the new FIELD predicts it -- which is why the
    first arrangement scored the classical one-sweep predictor negative too, and
    why the result is not about Poseidon.
    """
    experts = NI.make_experts(state[0], state[1], expose_elliptic=False)
    seam = NI.MultiplierSeam(a=experts["C00"], b=experts["C10"])
    pr = seam.probe()
    lam = np.zeros(seam.m_eff)
    for _ in range(6):
        lam = lam - np.linalg.solve(pr["S"], seam.residual(lam))
    assert np.linalg.norm(seam.residual(lam)) < 1e-10

    # the field's own change over the same macro-step, on the same face, in the
    # same reduced space -- the largest thing a field predictor could offer
    ex = NI.SchwarzExchange(state[0], state[1], expose_elliptic=False, project=False)
    nu_, _nv = ex.sweep(ex.u0, ex.v0)
    ox, oy = ex.tiling.offsets[0]
    n = ex.tiling.n
    # C00's OWN xhi ring: n cells of the window, not the whole domain column
    ring = (slice(oy, oy + n), ox + n - 1)
    change = seam._reduce(nu_[ring] - ex.u0[ring])
    assert np.linalg.norm(lam) > 5.0 * np.linalg.norm(change)


@needs_expert
def test_the_matrix_free_reduce_is_the_declared_prolongations_forced_adjoint(state):
    """`MultiplierSeam._reduce` writes ``h P^T`` out by hand for speed.

    It has to be the same map `window_ns.face_prolongation` declares, or the
    interface residual is on a different space from the one the compiler checked.
    """
    experts = NI.make_experts(state[0], state[1])
    seam = NI.MultiplierSeam(a=experts["C00"], b=experts["C10"])
    P = W.face_prolongation("C00", "xhi:MECH", experts["C00"].n)
    space = InterfaceSpace(seam_id="sx0", dim=W.M_EFF)
    e = np.random.default_rng(0).standard_normal(experts["C00"].n)
    assert np.allclose(seam._reduce(e), P.reduce(e, space), atol=1e-14)


# ---------------------------------------------------------------------------
# 8. what this graph declares, and what it does not
# ---------------------------------------------------------------------------


@needs_expert
def test_the_graph_declares_no_measured_constants(state):
    """Every constant in `MeasuredConstants` is about composition ERROR.

    This study measures the ITERATION.  Carrying `window_ns`'s numbers across
    would make the compile look better and would be about a different question.
    """
    graph, _ = NI.build(state[0], state[1])
    assert graph.measured is None


@needs_expert
def test_no_new_capability_record_is_declared(state):
    """The agents are `window_ns`'s, unchanged.

    A study about a SCHEME that needed a new L1 record would be a study about the
    record.  This is the affordability claim being used rather than restated.
    """
    graph, experts = NI.build(state[0], state[1])
    for a in graph.agents:
        mine = a.capabilities
        theirs = W.window_capabilities(experts[a.agent_id])
        assert mine.weight_hash == theirs.weight_hash
        assert mine.stencil_radius == theirs.stencil_radius
        assert mine.substeps_per_macro_step == theirs.substeps_per_macro_step
        assert [p.name for p in mine.ports] == [p.name for p in theirs.ports]
        assert mine.governing_family == theirs.governing_family


@needs_expert
def test_the_geometry_is_the_checkpoints_own_and_is_not_re_derived(state):
    """`HybridTiling` is `PoseidonTiling` with different agent names.

    The overlap arithmetic, the ramped partition of unity and the contaminated
    sets have to be the same geometry as the checkpoint's graph, or the classical
    column is not a column beside it.
    """
    from atlas.cases import poseidon as PO
    mine = NI.HybridTiling(mono_n=235)
    theirs = PO.PoseidonTiling(mono_n=235)
    assert mine.n == theirs.n == PO.EXPERT_RES
    assert mine.halo == theirs.halo == 21
    assert mine.offsets == theirs.offsets
    assert mine.names != theirs.names
    for a, b in zip(mine.weights(), theirs.weights()):
        assert np.array_equal(a, b)
