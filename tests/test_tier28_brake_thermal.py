"""Tier 25 -- CS-11, a bound for the multirate lag defect, and a horizon law.

`atlas/cases/brake_thermal.py` and `scripts/w131_brake_thermal.py`.  Six groups,
and the first two are what make the rest mean anything:

  * **the duct is graded** -- it is the one expert in this seam that is not a
    build-repo donor, so it is checked against two closed-form oracles (pure
    advection of a step, and the erf half-space solution), against its own
    refinement, and for the discrete conservation its wall flux has to satisfy.
    An expert nobody graded is a fixture wearing a solver's name;
  * **the declarations** -- both sides declare the same `response_half`, the
    same seam base (W74's discipline, applied rather than rediscovered), a
    computed sub-step count rather than a hard-coded one (mistake 5), and an
    interface space at the GRID because the probed seam operator turns out to
    have no spectral cutoff to declare;
  * **the controls** -- the single-rate column's lag defect is EXACTLY zero, the
    Fourier basis is full rank and its Nyquist column is not the zero vector,
    and the objective is bit-reproducible.  Each is a control and none is quoted
    as a floor (W106);
  * **the bound** -- sigma is first order in the exchange interval, the bound
    built from the probe's own slope holds at every graded interval, and the
    clock RATIO moves it by less than a part in a thousand at a fixed interval;
  * **W17** -- a trace carried as a linear waveform reduces the defect, and R3
    admits it here because both agents declare `bc_time_varying`;
  * **W128** -- the design sensitivity is a function of the horizon, its sign is
    undetermined below a stated one, and the long-horizon value agrees with the
    closed-form lumped prediction.

Artifact-backed tests re-derive the published numbers from `out/w131/w131.json`
and skip when it is absent.  The live tests build graphs, probe seams and march
short columns; the long sweeps are not re-run here.
"""

from __future__ import annotations

import json
import math
import os
import sys

# Before numpy.  The build repo pulls torch in and torch's OpenMP beside numpy's
# MKL aborts the interpreter inside a dense solve.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import compile_scheme                                      # noqa: E402
from atlas.capability import (                                        # noqa: E402
    BCChannel, EllipticSubsolve, MotionClass, TimeDiscretization,
)
from atlas.graph import Decomposition, FluxMatching                   # noqa: E402
from atlas.ports import PortType, ResponseHalf                        # noqa: E402

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out", "w131", "w131.json")

brake_thermal = pytest.importorskip("atlas.cases.brake_thermal")
B = brake_thermal


@pytest.fixture(scope="module")
def art():
    if not os.path.isfile(ARTIFACT):
        pytest.skip(f"{ARTIFACT} not present; run scripts/w131_brake_thermal.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def experts():
    return B.DuctAgent(), B.DiscAgent()


@pytest.fixture(scope="module")
def settled():
    return B.settled_state()


# ---------------------------------------------------------------------------
# 1. the duct is a solver, and it is graded
# ---------------------------------------------------------------------------


def test_the_duct_transports_a_step_at_the_declared_speed():
    """Pure advection: with diffusion off, a front moves at u and nowhere else.

    The oracle is the method of characteristics, and what it grades is that the
    advective term carries the declared bulk velocity rather than some multiple
    of it -- the design knob enters here and nowhere else, so a factor lost in
    the profile normalization would move every sensitivity in the study.
    """
    d = B.DuctAgent(u_bulk=40.0)
    d._a = np.zeros_like(d._a)          # diffusion off: characteristics only
    d._af = np.zeros_like(d._af)
    T = np.full((B.N_SEAM, B.NY_DUCT), 300.0)
    wall = np.full(B.N_SEAM, 300.0)
    # T_IN is 420 K, so the inlet injects a step that marches downstream
    t_travel = B.L_R / d._u.max()
    T, _ = d.advance(T, 0.5 * t_travel, wall)
    front = np.argmax(T[:, -1] < 0.5 * (300.0 + B.T_IN))
    expected = 0.5 * B.N_SEAM
    # first-order upwind smears the front, so the tolerance is the smear width
    assert abs(front - expected) <= 0.25 * B.N_SEAM, (front, expected)
    # and nothing has reached the far end
    assert T[-1, -1] == pytest.approx(300.0, abs=1.0)


def test_the_duct_diffuses_like_the_erf_solution():
    """Pure diffusion from a step wall: the erf half-space oracle.

    ``T(y,t) = T_w + (T_0 - T_w) erf(y / (2 sqrt(alpha t)))`` while the front is
    far from the far wall.  Advection off, a uniform alpha, and the comparison is
    against the closed form and not against another run.
    """
    d = B.DuctAgent(u_bulk=40.0)
    d._u = np.zeros_like(d._u)
    alpha = 2.0e-3
    d._a = np.full_like(d._a, alpha)
    d._af = np.full_like(d._af, alpha)
    T0, Tw = 420.0, 300.0
    T = np.full((B.N_SEAM, B.NY_DUCT), T0)
    wall = np.full(B.N_SEAM, Tw)
    t = 2.0e-4
    T, _ = d.advance(T, t, wall)
    y = (np.arange(B.NY_DUCT) + 0.5) * B.DY
    exact = Tw + (T0 - Tw) * np.array([math.erf(v / (2.0 * math.sqrt(alpha * t)))
                                       for v in y])
    got = T[B.N_SEAM // 2]
    rel = np.linalg.norm(got - exact) / np.linalg.norm(exact - Tw)
    # 8 cells across the layer and a first-order wall treatment: this is a grade,
    # not a convergence study, and 15% is what that geometry supports
    assert rel < 0.15, (rel, got, exact)


def test_the_duct_converges_first_order_under_its_own_refinement():
    """Halving the Courant number halves the error against a fine reference.

    **The regime matters and getting it wrong is W106.** Marched from the
    settled field the duct is already at its steady state, every Courant number
    agrees to $10^{-12}$, and a "convergence" test there is comparing roundoff to
    roundoff -- which is what the first version of this test did. The transient
    has to be under-resolved for the discretization error to exist at all, so
    this marches a step wall for two duct native steps from a uniform field.
    """
    T0 = np.full((B.N_SEAM, B.NY_DUCT), B.T_IN)
    wall = np.full(B.N_SEAM, 800.0)
    ref, _ = B.DuctAgent(cfl=0.0125).advance(T0.copy(), 2.0e-4, wall)
    errs = []
    for cfl in (0.4, 0.2, 0.1):
        T, _ = B.DuctAgent(cfl=cfl).advance(T0.copy(), 2.0e-4, wall)
        errs.append(np.linalg.norm(T - ref) / np.linalg.norm(ref))
    assert errs[0] > 1e-5, errs           # the error is resolvable, not roundoff
    for a, b in zip(errs, errs[1:]):
        assert 1.7 < a / b < 2.6, errs    # forward Euler: first order in dt


def test_the_wall_flux_is_the_same_quantity_the_energy_balance_sees():
    """The declared wall flux equals the energy the wall face actually removes.

    `wall_heat_flux` is a formula and the march is a discretization; if they
    disagree the seam is exchanging one number and conserving another, which is
    the silent class R9 exists for.
    """
    d = B.DuctAgent()
    T = np.full((B.N_SEAM, B.NY_DUCT), B.T_IN)
    wall = np.full(B.N_SEAM, 600.0)
    dt = d.dt_stable
    q, _ = d.wall_heat_flux(T, wall)
    T2, n = d.advance(T.copy(), dt, wall)
    assert n == 1
    # the first cell's energy change from the wall face alone, per unit area
    dE = B.RHO_G * B.CP_G * B.DY * (T2[:, 0] - T[:, 0]) / dt
    # the y-face flux into cell 0 from the wall is -q (q is INTO the wall), and
    # the interior face and the x-terms are what remain; on a uniform field the
    # x-terms vanish and the interior face flux is zero, so this is exact
    assert np.allclose(dE, -q, rtol=1e-12), (dE[:3], -q[:3])


def test_the_wall_coefficient_follows_the_dittus_boelter_exponent():
    """h ~ U^0.8, and no constant in the closure was fitted to make it."""
    us = np.array([20.0, 80.0])
    hs = np.array([B.DuctAgent(u_bulk=u).h_wall for u in us])
    e = math.log(hs[1] / hs[0]) / math.log(us[1] / us[0])
    assert 0.75 < e < 0.90, e


def test_the_duct_declines_outside_its_closure_s_own_reynolds_range():
    """Blasius is fitted for 4e3 < Re < 1e5 and the expert says so."""
    assert B.DuctAgent(u_bulk=40.0).validity()
    assert not B.DuctAgent(u_bulk=1.0).validity()
    assert not B.DuctAgent(u_bulk=1000.0).validity()


# ---------------------------------------------------------------------------
# 2. the declarations
# ---------------------------------------------------------------------------


def test_the_disc_declares_the_embedded_solve_it_actually_runs():
    """Backward Euler over the cross-section is EMBEDDED, and L2/R10 refuses it.

    The ladder's CS-11 row asks for a slow EMBEDDED conduction agent; declaring
    anything else to dodge the refusal would be the refusal working correctly and
    the declaration lying.
    """
    g, _ = B.build(mode="as-built", clocks="native")
    disc = [a for a in g.agents if a.agent_id == "disc"][0]
    assert disc.capabilities.elliptic_subsolve is EllipticSubsolve.EMBEDDED
    assert disc.capabilities.time_discretization is TimeDiscretization.IMPLICIT
    r = compile_scheme(g, probe_state="test")
    assert r.verdict.value == "refuse"
    assert any(d.rule == "R10" for d in r.decisions._decisions
               if d.verdict.value == "refuse")


def test_the_split_step_graph_compiles_and_R9_lag_is_what_decertifies_it():
    """The row this case study exists to close is the one that fires."""
    g, _ = B.build(mode="split-step", clocks="native")
    r = compile_scheme(g, probe_state="test")
    assert r.verdict.value == "admit-uncertified"
    rules = [f"{d.layer}/{d.rule}" for d in r.decisions._decisions
             if d.verdict.value != "admit"]
    assert "L7/R9/lag" in rules, rules
    assert not any(d.verdict.value == "refuse" for d in r.decisions._decisions)


def test_a_pointwise_declaration_still_earns_R9_s_refusal():
    """Nothing here weakens R9; the bound is about the term R9 does not cover."""
    g, _ = B.build(mode="split-step", clocks="native",
                   flux_matching=FluxMatching.POINTWISE)
    r = compile_scheme(g, probe_state="test")
    assert r.verdict.value == "refuse"


def test_both_sides_declare_the_same_seam_base():
    """W74: the base belongs to the SEAM, and left alone these two are 120 K apart."""
    _g, ex = B.build(mode="split-step", clocks="native")
    a = ex["duct"].base_trace()
    b = ex["disc"].base_trace()
    assert np.allclose(a, b), (a[0], b[0])
    assert a[0] == pytest.approx(B.seam_base(), rel=1e-12)
    # and the naive per-expert bases really would have disagreed
    assert abs(B.T_IN - B.T_DISC_0) > 100.0


def test_both_sides_declare_the_same_response_half():
    """L3/C9 refuses a disagreement; this asserts the graph does not have one."""
    g, _ = B.build(mode="split-step", clocks="native")
    halves = {p.response_half for a in g.agents for p in a.capabilities.ports}
    assert halves == {ResponseHalf.FLOW}


def test_the_substep_count_is_a_function_of_the_step_and_not_a_constant():
    """`CASE-STUDY-GUIDE` mistake 5, which cost two orders in tau once."""
    d = B.DuctAgent()
    assert d.substeps_at(B.DT_DUCT) < d.substeps_at(10.0 * B.DT_DUCT)
    assert B.disc_substeps_at(B.DT_DISC) > B.disc_substeps_at(B.DT_DISC / 10.0)
    # and it moves with the design knob, which is why it cannot be hard-coded
    assert B.DuctAgent(u_bulk=80.0).substeps_at(B.DT_DUCT) > \
        B.DuctAgent(u_bulk=20.0).substeps_at(B.DT_DUCT)


def test_the_clock_ratio_is_what_the_two_declared_steps_give():
    g, _ = B.build(mode="split-step", clocks="native")
    steps = sorted(a.capabilities.dt_native for a in g.agents)
    assert steps[1] / steps[0] == pytest.approx(B.CLOCK_RATIO, rel=1e-9)
    assert g.is_multirate()
    assert g.flux_matching is FluxMatching.TIME_INTEGRATED
    assert g.decomposition is Decomposition.NON_OVERLAPPING


def test_the_integrated_response_equals_the_plain_one_at_a_single_substep():
    """R9's own falsifiable condition on `boundary_response_integrated`."""
    for agent, port in ((B.DuctAgent(), "wall:THERM"), (B.DiscAgent(), "duct:THERM")):
        base = agent.base_trace()
        a = agent.respond(port, base)
        b = agent.respond_integrated(port, base, 1)
        assert np.allclose(a, b, rtol=0, atol=0), port


def test_the_seam_declares_a_zero_null_space_and_the_probe_agrees():
    """A conjugate-heat seam constrains nothing the way incompressibility does."""
    g, _ = B.build(mode="split-step", clocks="native")
    assert g.connections[0].expected_null_dim == 0
    assert g.connections[0].port_type is PortType.THERM
    r = compile_scheme(g, probe_state="test")
    assert not any(d.rule == "null-space" for d in r.decisions._decisions
                   if d.verdict.value == "refuse")


# ---------------------------------------------------------------------------
# 3. the controls
# ---------------------------------------------------------------------------


def test_the_fourier_basis_is_full_rank_and_its_nyquist_column_is_not_zero():
    """The bug the other case studies could not have hit.

    At ``k = n/2`` the cosine samples at ``cos(pi (i + 1/2))``, which is zero at
    every cell centre, so a naive construction hands back a zero column and every
    projection built on it silently drops a direction.  `window_ns`,
    `thermal_seam` and `wake_array` all declare ``m`` far below ``n``.
    """
    P = B.fourier_basis(B.N_SEAM, B.M_EFF)
    assert P.shape == (B.N_SEAM, B.N_SEAM)
    assert np.linalg.matrix_rank(P) == B.N_SEAM
    assert np.min(np.linalg.norm(P, axis=0)) > 1e-8
    gram = B.H_SEAM * P.T @ P
    assert np.abs(gram - np.eye(B.M_EFF)).max() < 1e-12
    # the full-rank projector is the identity, so truncation costs nothing at m=n
    q = P @ P.T * B.H_SEAM
    assert np.abs(q - np.eye(B.N_SEAM)).max() < 1e-12


def test_the_interface_condition_is_solved_exactly_and_not_to_a_tolerance():
    """CS-10's lesson: on the states at hand both responses are AFFINE, so the
    port's own condition has a closed form and needs no Newton iteration."""
    Tg, Td, _ = B.settled_state()
    duct, disc = B.DuctAgent(), B.DiscAgent()
    lam = B.solve_interface(duct, Tg, disc, Td)
    a = duct.k_wall / (0.5 * B.DY)
    res = a * (Tg[:, 0] - lam) - disc.h_in * (lam - disc.face_T(Td))
    assert np.abs(res).max() < 1e-9, np.abs(res).max()


def test_the_interface_root_does_not_depend_on_the_flux_convention():
    """W66 is about what the bond IS, not about where the interface sits."""
    Tg, Td, _ = B.settled_state()
    disc = B.DiscAgent()
    a = B.solve_interface(B.DuctAgent(flux_convention="entropy"), Tg, disc, Td)
    b = B.solve_interface(B.DuctAgent(flux_convention="heat"), Tg, disc, Td)
    assert np.allclose(a, b, rtol=0, atol=0)


def test_the_objective_is_bit_reproducible():
    """A control, and NOT a level anything is quoted against (W106)."""
    Tg, Td, _ = B.settled_state()
    a = B.composed_rollout().run(8, Tg, Td).objective(8)
    b = B.composed_rollout().run(8, Tg, Td).objective(8)
    assert a == b


def test_the_single_rate_column_has_no_lag_defect_at_all():
    """The control that says the instrument measures the lag and not itself.

    At an exchange interval equal to the fast agent's own step there is nothing
    to be stale about, and the measured defect has to be exactly zero rather than
    small.
    """
    Tg, Td, _ = B.settled_state()
    duct, disc = B.DuctAgent(), B.DiscAgent()
    lam0 = B.solve_interface(duct, Tg, disc, Td)
    held, _ = duct.advance(Tg.copy(), B.DT_DUCT, lam0)
    ref = B.BrakeRollout(exchange=B.DT_DUCT).run(1, Tg, Td)
    assert np.array_equal(ref.trace[0], lam0)
    assert np.array_equal(ref.T_gas, held)


# ---------------------------------------------------------------------------
# 4. the bound  (W90)
# ---------------------------------------------------------------------------


def test_sigma_is_first_order_in_the_exchange_interval(art):
    rows = [r for r in art["sigma_law"]["rows"] if r["sigma"] > 0.0]
    assert len(rows) >= 4
    exps = [r["exponent"] for r in rows if r.get("exponent") == r.get("exponent")]
    exps = [e for e in exps if not math.isnan(e)]
    assert exps, art["sigma_law"]["rows"]
    for e in exps:
        assert 0.85 < e < 1.30, (e, exps)


def test_the_bound_holds_at_every_graded_interval(art):
    assert art["bound_holds"] is True
    tight, worst = art["bound_tightness"]
    assert tight >= 1.0
    # loose by a small factor is a bound; loose by an order is a slogan
    assert worst < 4.0, worst


def test_the_single_rate_control_carries_a_zero_defect(art):
    ctrl = [r for r in art["sigma_law"]["rows"] if r["ratio"] == 1.0]
    assert ctrl and ctrl[0]["sigma"] == 0.0
    assert 1.0 in art.get("bound_control_ratio", [])


def test_the_clock_ratio_is_not_the_variable(art):
    """8x of clock ratio at a fixed interval must not move sigma."""
    r = art["ratio"]
    assert r["ratio_span"] >= 7.0
    assert abs(r["spread"] - 1.0) < 1.0e-3, r["spread"]


def test_the_interval_sweep_spans_three_decades_of_clock_ratio(art):
    ratios = [r["ratio"] for r in art["sigma_law"]["rows"]]
    assert min(ratios) == 1.0
    assert max(ratios) >= 1000.0


def test_the_seam_operator_has_no_spectral_cutoff(art):
    """The measurement behind declaring M_EFF at the grid."""
    for name in ("duct", "disc"):
        blk = art["resolution"]["blocks"][name]
        assert blk["cond"] < 1.1, (name, blk["cond"])
        assert blk["off_over_diag"] < 0.05, (name, blk["off_over_diag"])
    trunc = {r["m"]: r["duct"] for r in art["resolution"]["truncation"]}
    assert trunc[32] < 1e-10
    assert trunc[11] > 0.5            # an 11-mode basis loses most of it
    assert art["resolution"]["m_declared"] == B.N_SEAM


def test_the_truncation_costs_the_space_and_not_the_conditioning(art):
    betas = {r["m"]: r["beta"] for r in art["resolution"]["betas"]}
    assert abs(betas[11] / betas[32] - 1.0) < 0.01, betas


# ---------------------------------------------------------------------------
# 5. W17's W > 1
# ---------------------------------------------------------------------------


def test_R3_admits_a_waveform_on_this_graph():
    """W > 1 needs `bc_time_varying` on every agent at the interface."""
    g, _ = B.build(mode="split-step", clocks="native")
    assert all(a.capabilities.bc_time_varying for a in g.agents)


def test_the_waveform_reduces_the_lag_defect(art):
    rows = art["waveform"]["rows"]
    assert rows
    for r in rows:
        assert r["sigma_wave"] < r["sigma_held"], r
        assert r["reduction"] > 2.0, r


def test_the_waveform_is_built_from_the_past_and_not_from_its_own_endpoint():
    """An extrapolation that reads its own interval's endpoint is the answer.

    Pinned structurally: `BrakeRollout.interval` is handed ``lam_next`` derived
    in `run` from ``lam - lam_prev``, both of which are already known when the
    interval starts.
    """
    Tg, Td, _ = B.settled_state()
    r = B.BrakeRollout(exchange=B.DT_DISC, waveform=2)
    m = r.run(3, Tg, Td)
    assert m.face_T.size == 3
    # the first interval has no past, so its slope is zero and it degenerates to
    # the held scheme -- which is the honest behaviour and not a special case
    held = B.BrakeRollout(exchange=B.DT_DISC, waveform=1).run(1, Tg, Td)
    one = B.BrakeRollout(exchange=B.DT_DISC, waveform=2).run(1, Tg, Td)
    assert one.face_T[0] == pytest.approx(held.face_T[0], rel=1e-12)


# ---------------------------------------------------------------------------
# 6. W128 -- the horizon law
# ---------------------------------------------------------------------------


def test_the_design_sensitivity_depends_on_the_horizon(art):
    rows = art["horizon"]["rows"]
    g = [r["grad_ref"] for r in rows]
    assert max(abs(x) for x in g) > 3.0 * min(abs(x) for x in g)


def test_the_design_sensitivity_changes_sign_over_the_horizons_measured(art):
    """The finding, and the reason a gradient has to be quoted with a horizon."""
    assert art["horizon"]["crossings"], art["horizon"]["rows"]


def test_the_finite_difference_is_on_a_truncation_branch(art):
    """Three FD steps over 4x must agree, or the sweep is in cancellation."""
    assert art["horizon"]["fd_spread"] < 5.0e-2, art["horizon"]["fd_spread"]


def test_the_horizon_law_carries_a_validity_limit(art):
    """W128's own T_pred: the horizon below which the number is not the number."""
    lim = art["horizon"]["validity_limit"]
    assert any(v is not None for v in lim.values()), lim
    got = [(float(k), v) for k, v in lim.items() if v is not None]
    # a tighter tolerance cannot be met at a shorter horizon
    got.sort(key=lambda kv: -kv[0])
    ns = [v for _k, v in got]
    assert ns == sorted(ns), got


def test_the_long_horizon_sensitivity_matches_the_lumped_prediction(art):
    """The closed form was written before the run and it is what the sign means."""
    h = art["horizon"]
    pred = h["lumped"]["dT_eq_dU"]
    got = h["converged"]
    assert pred < 0.0 and got < 0.0, (pred, got)
    assert 0.25 < abs(got / pred) < 4.0, (got, pred)


def test_the_short_and_long_horizon_signs_are_the_ones_the_mechanism_predicts(art):
    """Short horizon follows sign(T_gas - T_0); long follows sign(dT_eq/dU)."""
    h = art["horizon"]
    rows = h["rows"]
    assert h["lumped"]["T_gas_minus_T0"] > 0.0
    assert rows[0]["grad_ref"] > 0.0, rows[0]
    assert rows[-1]["grad_ref"] < 0.0, rows[-1]


def test_the_release_field_choice_does_not_carry_the_gradient(art):
    """CS-10's convention: every design point is released from the same field."""
    assert art["horizon"]["ic_control"] < 1.0e-2, art["horizon"]["ic_control"]


def test_the_objective_is_bit_reproducible_in_the_artifact_too(art):
    assert art["horizon"]["reproducibility"] == 0.0


# ---------------------------------------------------------------------------
# 7. the accumulated defect, marched past the crossing
# ---------------------------------------------------------------------------


def test_the_accumulated_defect_was_marched_and_the_crossing_looked_for(art):
    """Tier 23's standing rule, applied as a gate rather than discovered."""
    rows = art["accumulate"]["rows"]
    assert rows
    for r in rows:
        assert "sign_changes" in r
        assert r["at_step"] >= 1


def test_a_longer_lag_costs_more_than_a_shorter_one(art):
    rows = {r["column"]: r for r in art["accumulate"]["rows"]}
    held = rows["multirate, held"]["max_abs"]
    lag4 = rows["multirate, lag 4"]["max_abs"]
    assert lag4 > held, (held, lag4)


def test_the_waveform_column_beats_the_held_one_over_a_rollout(art):
    rows = {r["column"]: r for r in art["accumulate"]["rows"]}
    assert rows["multirate, waveform"]["max_abs"] < rows["multirate, held"]["max_abs"]


def test_the_referent_moved_enough_for_the_defect_to_mean_something(art):
    """A defect quoted against a state that did not move is quoted against noise."""
    a = art["accumulate"]
    rise = a["referent_end"] - a["referent_start"]
    assert rise > 10.0, rise
    for r in a["rows"]:
        assert abs(r["end"]) < rise
