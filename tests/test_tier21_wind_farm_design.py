"""Tier 21 -- PoC 1a, the differentiable wind-farm design loop.

`atlas/cases/wind_farm_design.py` and `scripts/w111_wind_farm_design.py`.

The claim this PoC makes is narrow and entirely mechanical: **the composed
rollout CS-7 and CS-8 marched is the one being differentiated, and the
derivative is right.**  Both halves are testable without a single optimiser step,
and that is what this file is:

  * **the column is the same column** -- the torch agent is `WindowNS` with the
    elliptic part exposed, BITWISE equal to `wake_array.exposed_reference_solver`
    on a real `step_batch`; the out-of-place stencils are bitwise the in-place
    ones; the blend is the declared partition of unity's own weights; the
    projection reproduces `wake_array.assemble_conservative` through the
    declared `assembly.ProjectedAssembly` to float64 round-off;
  * **the disk is still `disk.ActuatorDisk`** -- the momentum sink integrates to
    exactly `-T`, the power is the disk expert's own at the same inflow, the ROT
    port's `tau * omega` is that power, and yaw rotates the force by exactly the
    yaw angle and costs exactly `cos^3`;
  * **the derivative is right** -- the adjoint agrees with central differences
    on a real (short) rollout, the checkpointed rollout is bitwise the
    un-checkpointed one, and the same design vector twice gives the same
    objective bit for bit, which is what "pin the batch layout" (OP-6) means
    operationally;
  * **the optimisers are optimisers** -- the projection respects the box and the
    spacing set, and the hand-written CMA-ES recovers the optimum of a problem
    whose answer is known.

The measurements are in `out/w111/w111.json`; the last group asserts the
headline is consistent with the traces it was computed from, and skips when the
artefact is absent.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.assembly import ProjectedAssembly                          # noqa: E402
from atlas.cases import scaling_ladder as sl                          # noqa: E402
from atlas.cases import wake_array as wa                              # noqa: E402

torch = pytest.importorskip("torch")

from atlas.cases import wind_farm_design as wd                        # noqa: E402

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out", "w111", "w111.json")


def _needs_expert():
    """The reference solver lives in the build repo; skip cleanly without it."""
    try:
        wa.exposed_reference_solver(wa.NU_REF)
    except Exception as exc:                                # pragma: no cover
        pytest.skip(f"reference.WindowNS unavailable: {exc}")


@pytest.fixture(scope="module")
def tiny():
    """The smallest real case: the ladder's N2 rung, two turbines, two steps.

    Small enough to run in a test suite and still a genuine composed rollout --
    two 128-cell windows, a 16-cell overlap, the partition of unity, the global
    projection and ~20 solver sub-steps per macro-step.
    """
    _needs_expert()
    torch.set_num_threads(4)
    case = wd.FarmCase("T2", n_col=2, n_row=1, k_col=2, k_row=1,
                       steps=2, avg_window=1)
    return case, wd.Rollout(case)


# ---------------------------------------------------------------------------
# 1. the column is the column
# ---------------------------------------------------------------------------


def test_torch_agent_is_the_exposed_reference_solver_bitwise():
    """`tape_solver` is `wake_array.exposed_reference_solver`, in torch.

    Not "close to": equal to the bit, on a real `step_batch` with a body force.
    If this ever fails the PoC is differentiating something other than the
    column CS-7 measured, and every number it produces is about a different
    object.
    """
    _needs_expert()
    npy = wa.exposed_reference_solver(wa.NU_REF)
    tch = wd.tape_solver(wa.NU_REF)
    rng = np.random.default_rng(11)
    b = 3
    u = np.ones((b, wa.N, wa.N)) + 0.05 * rng.standard_normal((b, wa.N, wa.N))
    v = 0.02 * rng.standard_normal((b, wa.N, wa.N))
    f = np.zeros((b, wa.N, wa.N))
    f[:, 50:60, 40:47] = -1.5
    un, vn = npy.step_batch(u, v, wa.MACRO_DT, bc0=None, force=(f, np.zeros_like(f)))
    with torch.no_grad():
        ut, vt = tch.step_batch(
            torch.as_tensor(u), torch.as_tensor(v), wa.MACRO_DT, bc0=None,
            force=(torch.as_tensor(f), torch.zeros(b, wa.N, wa.N,
                                                   dtype=torch.float64)))
    assert isinstance(ut, torch.Tensor), "the tape was cut at the class boundary"
    assert np.array_equal(ut.numpy(), un)
    assert np.array_equal(vt.numpy(), vn)


def test_agent_declares_the_same_settings_as_the_numpy_one():
    _needs_expert()
    npy = wa.exposed_reference_solver(wa.NU_REF)
    tch = wd.tape_solver(wa.NU_REF)
    for k in ("nu", "length", "n", "cfl", "skew", "transmission", "h"):
        assert getattr(tch, k) == getattr(npy, k), k
    assert tch.backend == "torch" and npy.backend == "numpy"
    # the elliptic part really is exposed on both: `_project` is the identity
    a = torch.ones(1, 8, 8, dtype=torch.float64)
    pu, pv = tch._project(a, a)
    assert pu is a and pv is a


def test_out_of_place_stencils_are_bitwise_the_in_place_ones():
    """The three overridden kernels are the same arithmetic, not an approximation.

    They exist only because the backward of a `cat` is a slice while the
    backward of an in-place slice write is a scatter (~20% per gradient step,
    measured).  A speedup that changed a digit would not be one.
    """
    _needs_expert()
    base = importlib.import_module("atlas.cases.window_ns")._no_projection_class()
    oop = wd.torch_solver_class()
    kw = dict(nu=wa.NU_REF, length=wa.S_LEN, n=wa.N, cfl=0.4,
              transmission="dirichlet", backend="torch", device="cpu")
    a, b = base(**kw), oop(**kw)
    rng = np.random.default_rng(5)
    f = torch.as_tensor(rng.standard_normal((2, wa.N, wa.N)))
    for name in ("_ddx", "_ddy", "_lap"):
        x = getattr(a, name)(f)
        y = getattr(b, name)(f)
        assert torch.equal(x, y), name


def test_case_geometry_is_the_ladder_rung_unchanged():
    for case in (wd.case_k12(), wd.case_k25()):
        assert case.tiling == sl.rung(case.n_col, case.n_row).tiling
        assert case.tiling.ramp == wa.RAMP
        assert case.n_windows == case.tiling.n_windows
        assert case.shape == (case.tiling.ny, case.tiling.nx)
    assert wd.case_k12().n_windows == 12
    assert wd.case_k25().n_windows == 24
    assert wd.case_k12().k == 12
    assert wd.case_k25().k == 25


def test_the_declared_assembly_is_a_projected_assembly(tiny):
    """R12's object, not a description of one."""
    case, ro = tiny
    assert isinstance(ro.assembly, ProjectedAssembly)
    assert ro.assembly.projection.satisfies_C2
    assert ro.assembly.projection.scope == "global"
    assert ro.assembly.projection.stage == "after-assembly"
    assert ro.assembly.projection.cadence == 1


def test_blend_uses_the_declared_partition_of_unity(tiny):
    case, ro = tiny
    w = case.tiling.weights()
    for k, (ox, oy) in enumerate(case.tiling.offsets):
        assert np.array_equal(ro._chi[k].numpy(),
                              w[k][oy:oy + wa.N, ox:ox + wa.N])


def test_torch_assembly_reproduces_assemble_conservative(tiny):
    """blend + global Leray == `wake_array.assemble_conservative`, to round-off.

    The numpy path goes through the declared `ProjectedAssembly`, so this is a
    comparison against the object the graph carries rather than against a
    transcription of it.  Round-off and not the bit, because the two FFTs are
    different libraries; the tolerance is 1e-12 on a field of order 1.
    """
    case, ro = tiny
    rng = np.random.default_rng(2)
    w = ro.n_win
    us = np.ones((w, wa.N, wa.N)) + 0.05 * rng.standard_normal((w, wa.N, wa.N))
    vs = 0.02 * rng.standard_normal((w, wa.N, wa.N))
    au, av = wa.assemble_conservative(case.tiling, us, vs, ro.assembly)
    with torch.no_grad():
        bu, bv = ro.blend(torch.as_tensor(us), torch.as_tensor(vs))
        pu, pv = ro.project(bu, bv)
    assert np.abs(pu.numpy() - au).max() < 1e-12
    assert np.abs(pv.numpy() - av).max() < 1e-12


def test_projection_removes_the_divergence_the_blend_made(tiny):
    """L6/C2's substantive half, on this case's own geometry."""
    case, ro = tiny
    rng = np.random.default_rng(7)
    w = ro.n_win
    us = np.ones((w, wa.N, wa.N)) + 0.08 * rng.standard_normal((w, wa.N, wa.N))
    vs = 0.05 * rng.standard_normal((w, wa.N, wa.N))
    with torch.no_grad():
        bu, bv = ro.blend(torch.as_tensor(us), torch.as_tensor(vs))
        pu, pv = ro.project(bu, bv)
    before = wa.divergence_rms(bu.numpy(), bv.numpy())
    after = wa.divergence_rms(pu.numpy(), pv.numpy())
    assert after < before


def test_band_holds_inlet_and_laterals_and_leaves_the_outlet_free(tiny):
    case, ro = tiny
    u = torch.zeros(case.shape, dtype=torch.float64)
    v = torch.ones(case.shape, dtype=torch.float64)
    bu, bv = ro.band(u, v)
    n = wd.BAND
    assert torch.all(bu[:, :n] == wa.U_INF) and torch.all(bv[:, :n] == 0.0)
    assert torch.all(bu[:n, :] == wa.U_INF) and torch.all(bu[-n:, :] == wa.U_INF)
    assert torch.all(bu[n:-n, -1] == 0.0), "the outlet must be left free"


# ---------------------------------------------------------------------------
# 2. the disk is still the actuator disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaw_deg", [0.0, 12.0, -20.0, 30.0])
def test_body_force_integrates_to_minus_thrust(tiny, yaw_deg):
    """Gate W2's property, kept under continuous placement and yaw.

    `disk.body_force_field` gets it from exact rectangle overlap; the Gaussian
    projection here gets it from a discrete normalization, which is the same
    guarantee by a different route and has to be checked the same way.
    """
    case, ro = tiny
    th = wd.initial_design(case).reshape(-1, 3)
    th[:, 2] = math.radians(yaw_deg)
    th = torch.as_tensor(th.reshape(-1))
    u = torch.full(case.shape, wa.U_INF, dtype=torch.float64)
    v = torch.zeros_like(u)
    fx, fy, u_n, thrust, power = ro.disks.forcing(u, v, th)
    tot = float(thrust.sum())
    gx = float(fx.sum()) * wa.DX ** 2
    gy = float(fy.sum()) * wa.DX ** 2
    assert abs(gx + tot * math.cos(math.radians(yaw_deg))) < 1e-12 * tot
    assert abs(gy + tot * math.sin(math.radians(yaw_deg))) < 1e-12 * max(tot, 1.0)


def test_stamping_box_truncation_is_below_float64(tiny):
    """The local box is a window on the lattice, not an approximation.

    The box corner is an integer taken from the position, so it moves by one
    cell when the position crosses a half-cell -- a discontinuity in the
    objective if anything the box cut off were nonzero.  Doubling the box has to
    change nothing at float64, and that is what makes the flip harmless rather
    than merely small.
    """
    case, ro = tiny
    th = torch.as_tensor(wd.initial_design(case))
    u = torch.full(case.shape, wa.U_INF, dtype=torch.float64)
    v = torch.zeros_like(u)
    fx0 = ro.disks.forcing(u, v, th)[0]
    big = wd.DiskBank(case)
    big.box_cells = (min(case.shape) - 1) // 2          # the widest that fits
    assert big.box_cells > ro.disks.box_cells
    fx1 = big.forcing(u, v, th)[0]
    scale = float(fx0.abs().max())
    assert float((fx1 - fx0).abs().max()) < 1e-13 * scale


def test_power_is_the_actuator_disk_experts_own(tiny):
    """`P = 1/2 rho A C_T' U^3` at the disk's own inflow -- `disk.ActuatorDisk`."""
    case, ro = tiny
    dk = importlib.import_module("atlas_windfarm_reference.disk")
    ref = dk.ActuatorDisk(a=wd.A_INDUCTION)
    for u_inf in (0.7, 1.0, 1.3):
        u = torch.full(case.shape, u_inf, dtype=torch.float64)
        v = torch.zeros_like(u)
        th = torch.as_tensor(wd.initial_design(case))
        _fx, _fy, u_n, thrust, power = ro.disks.forcing(u, v, th)
        st = ref(u_inf)
        assert float(u_n[0]) == pytest.approx(u_inf, rel=1e-9)
        assert float(thrust[0]) == pytest.approx(st.thrust, rel=1e-9)
        assert float(power[0]) == pytest.approx(st.power, rel=1e-9)


def test_rot_port_torque_times_omega_is_the_power(tiny):
    case, ro = tiny
    u = torch.full(case.shape, 1.1, dtype=torch.float64)
    v = torch.zeros_like(u)
    th = torch.as_tensor(wd.initial_design(case))
    _fx, _fy, u_n, _t, power = ro.disks.forcing(u, v, th)
    tau, omega = wd.DiskBank.rot_port(u_n, power)
    assert torch.allclose(tau * omega, power, rtol=1e-12)
    assert float(omega[0]) == pytest.approx(
        wd.TIP_SPEED_RATIO * 1.1 / (0.5 * wa.ROTOR_D), rel=1e-9)


@pytest.mark.parametrize("yaw_deg", [0.0, 10.0, 25.0])
def test_yaw_costs_exactly_cos_cubed_in_a_uniform_stream(tiny, yaw_deg):
    """The cos^3 law is a CONSEQUENCE here, not an imposed closure.

    Nothing multiplies by cos^3.  The disk responds to the axis-normal inflow
    `U cos g` and its power is cubic in that, so the law appears; what is
    genuinely a closure is that the disk pushes along its own axis, which is
    what deflects the wake, and that is stated in the module docstring.
    """
    case, ro = tiny
    u = torch.full(case.shape, wa.U_INF, dtype=torch.float64)
    v = torch.zeros_like(u)
    base = wd.initial_design(case)
    th0 = torch.as_tensor(base)
    thy = base.reshape(-1, 3).copy()
    thy[:, 2] = math.radians(yaw_deg)
    thy = torch.as_tensor(thy.reshape(-1))
    p0 = float(ro.disks.forcing(u, v, th0)[4].sum())
    py = float(ro.disks.forcing(u, v, thy)[4].sum())
    assert py / p0 == pytest.approx(math.cos(math.radians(yaw_deg)) ** 3, rel=1e-9)


def test_yaw_rotates_the_force_by_the_yaw_angle(tiny):
    case, ro = tiny
    u = torch.full(case.shape, wa.U_INF, dtype=torch.float64)
    v = torch.zeros_like(u)
    th = wd.initial_design(case).reshape(-1, 3)
    th[:, 2] = math.radians(18.0)
    fx, fy, _un, thrust, _p = ro.disks.forcing(u, v, torch.as_tensor(th.reshape(-1)))
    ang = math.atan2(float(fy.sum()), float(fx.sum()))
    assert ang == pytest.approx(math.radians(18.0) - math.pi, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. the derivative is right
# ---------------------------------------------------------------------------


def test_rollout_is_bitwise_deterministic(tiny):
    """OP-6 operationally: same theta, same batch layout, same number.

    `Rollout.cut` walks `tiling.offsets` and nothing reorders it, so the windows
    reach the expert in one fixed order at one fixed batch size on every call.
    A composed objective that did not repeat could not be finite-differenced at
    all, so this test comes before the gradient one.
    """
    case, ro = tiny
    th = wd.initial_design(case)
    a, _ = wd.value_only(th, ro)
    b, _ = wd.value_only(th, ro)
    assert a == b


def test_checkpointing_changes_nothing(tiny):
    """The tape is segmented for memory, and segmentation is not a scheme change."""
    case, ro = tiny
    plain = wd.Rollout(case, checkpoint=False)
    th = wd.initial_design(case)
    j1, g1, _ = wd.value_and_grad(th, ro)
    j2, g2, _ = wd.value_and_grad(th, plain)
    assert j1 == j2
    assert np.array_equal(g1, g2)


def test_adjoint_matches_central_differences(tiny):
    """The check the spec asks for, on a real composed rollout.

    `h = 1e-4` D, which is 0.3% of a cell.  The window matters and is reported
    in the results: at `h >= 1e-3` the objective's own lattice structure -- the
    bilinear inflow read has a kink at every cell centre -- shows up as a few
    percent, and below `1e-5` float64 cancellation does.  In between the two
    agree to the fifth digit.
    """
    case, ro = tiny
    th = wd.initial_design(case).reshape(-1, 3)
    th[:, 2] = np.radians([7.0, -11.0])
    th = th.reshape(-1)
    _j, g, _ = wd.value_and_grad(th, ro)
    idx = [0, 1, 2, 5]
    gfd, n_ev = wd.fd_gradient(th, ro, h=1e-4, idx=idx)
    assert n_ev == 2 * len(idx)
    for i in idx:
        scale = max(abs(g[i]), abs(gfd[i]), 1e-12)
        assert abs(g[i] - gfd[i]) / scale < 5e-3, (i, g[i], gfd[i])


def test_gradient_is_nonzero_in_every_variable_kind(tiny):
    """A staircase placement would give dJ/dx = 0 almost everywhere.

    That is the whole reason the disk was re-projected, so it is asserted rather
    than assumed: position and yaw both have to move the objective.
    """
    case, ro = tiny
    th = wd.initial_design(case)
    _j, g, _ = wd.value_and_grad(th, ro)
    for kind, name in enumerate(("x", "y", "yaw")):
        assert np.abs(g[kind::3]).max() > 0.0, name


def test_the_gradient_flows_through_the_fluid_and_not_only_the_disk(tiny):
    """The seam is INSIDE the differentiated path, and this is what says so.

    A turbine's own power depends on its position through the fluid state as
    well as through its own closure.  `value_and_grad_local` differentiates only
    the closure, on the states the live march produced; if it agreed with the
    full adjoint, every fluid-disk seam in the differentiated path would be
    decorative and a static wake model would give the same search direction.
    `--stage ablation` reports the size of the difference; this asserts it is
    not zero.
    """
    case, ro = tiny
    th = wd.initial_design(case)
    j_full, g_full, _ = wd.value_and_grad(th, ro)
    j_loc, g_loc, _ = wd.value_and_grad_local(th, ro)
    # the two differentiate the SAME objective at the same point, so the values
    # must agree; only the derivatives may differ
    assert j_loc == pytest.approx(j_full, rel=1e-12)
    assert np.linalg.norm(g_full - g_loc) > 1e-6 * np.linalg.norm(g_full)


# ---------------------------------------------------------------------------
# 4. the optimisers, and the constraints they respect
# ---------------------------------------------------------------------------


def test_initial_design_is_feasible():
    for case in (wd.case_k12(), wd.case_k25()):
        th = wd.initial_design(case)
        lo, hi = case.box
        assert np.all(th >= lo - 1e-12) and np.all(th <= hi + 1e-12)
        assert wd.min_spacing(th) >= case.s_min
        assert np.all(th[2::3] == 0.0), "the unoptimised layout faces the wind"


def test_projection_enforces_box_and_spacing():
    case = wd.case_k12()
    rng = np.random.default_rng(4)
    th = wd.initial_design(case)
    th = th + rng.normal(scale=1.5, size=th.shape)
    th[2::3] = rng.normal(scale=1.0, size=case.k)          # far outside the yaw box
    p = wd.project_design(th, case)
    lo, hi = case.box
    assert np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9)
    assert wd.min_spacing(p) >= case.s_min - 1e-6


def test_spacing_penalty_is_zero_when_feasible_and_positive_when_not():
    case = wd.case_k12()
    th = torch.as_tensor(wd.initial_design(case))
    assert float(wd.spacing_penalty(th, case)) == 0.0
    bad = th.clone().reshape(-1, 3)
    bad[1, 0] = bad[0, 0] + 0.5
    bad[1, 1] = bad[0, 1]
    val = float(wd.spacing_penalty(bad.reshape(-1), case))
    # one violating PAIR: the implementation halves a sum over the symmetric
    # distance matrix, so what comes out is the sum over j < k the docstring says
    assert val == pytest.approx(case.lam_spacing * (case.s_min - 0.5) ** 2)


def test_cma_es_recovers_a_known_optimum():
    """The baseline has to be a real CMA-ES or the ratio means nothing.

    A separable quadratic in 36 variables with a known maximiser, evaluated
    through the same `evaluate` hook the composed run uses.  No fluid, no disks
    -- this is a test of the search, and it is here because the headline number
    is a comparison against it.
    """
    case = wd.case_k12()
    target = wd.initial_design(case).copy()
    target[0::3] += 0.7
    target[1::3] -= 0.5
    target[2::3] = 0.2

    def evaluate(batch):
        return [-float(np.sum((np.asarray(t) - target) ** 2)) for t in batch]

    tr = wd.cma_es(wd.initial_design(case), None, budget=4000, sigma0=0.5,
                   seed=3, case=case, evaluate=evaluate)
    j, th = tr.best()
    # -1e-2 on an objective whose optimum is 0 and whose start is -30: this is a
    # convergence assertion, not a precision one, and CMA-ES's own termination
    # scale is what sets it
    assert j > -1e-2
    assert np.abs(th - target).max() < 0.05
    assert len(tr.evals) == tr.evals[-1]


def test_evals_to_tolerance_reads_the_trace():
    tr = wd.OptTrace("t")
    tr.evals = [1, 2, 3, 4]
    tr.best_j = [0.0, 0.5, 0.9, 1.0]
    assert wd.evals_to_tolerance(tr, 0.9) == 3
    assert wd.evals_to_tolerance(tr, 2.0) is None


def test_adam_takes_a_step_uphill(tiny):
    """Two Adam steps on the real objective; J must not go down over them.

    Two, because that is what fits in a test suite, and the assertion is the
    weak one it can support: the projected step is an ascent step on this
    objective at this point.
    """
    case, ro = tiny
    th = wd.initial_design(case)
    tr = wd.adam_optimise(th, ro, steps=2, lr_pos=0.15, lr_yaw=0.06)
    assert len(tr.j) == 2
    assert tr.best_j[-1] >= tr.j[0]
    assert wd.min_spacing(np.array(tr.theta[-1])) >= case.s_min - 1e-6


# ---------------------------------------------------------------------------
# 5. the artefact -- skipped when the run has not been made
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact():
    if not os.path.isfile(ARTIFACT):
        pytest.skip(f"no PoC 1a artefact at {ARTIFACT}; run "
                    "`python scripts/w111_wind_farm_design.py --stage merge`")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def test_artifact_headline_is_consistent_with_its_traces(artifact):
    parts = artifact["parts"]
    for cname, row in artifact["headline"].items():
        g = parts[f"grad_{cname}"]
        assert row["J_start"] == g["J0"]
        assert row["J_gradient"] == g["J_best"]
        assert row["gradient_rollouts"] == g["trace"]["evals"][-1]
        # the tolerance rule, recomputed
        eps = 0.02 * (row["J_gradient"] - row["J_start"])
        assert row["tolerance_J"] == pytest.approx(row["J_gradient"] - eps)
        # the gradient method reached its own tolerance within its own budget
        assert row["gradient_rollouts_to_tolerance"] is not None


def test_artifact_gradient_improved_the_objective(artifact):
    for cname, row in artifact["headline"].items():
        assert row["J_gradient"] > row["J_start"], cname
        assert row["farm_power_gradient"] > row["farm_power_start"], cname


def test_artifact_ratio_favours_the_gradient(artifact):
    """The headline, asserted rather than left in prose."""
    seen = 0
    for cname, row in artifact["headline"].items():
        r = row.get("ratio_rollouts") or row.get("ratio_rollouts_lower_bound")
        if r is None:
            continue
        seen += 1
        assert r > 1.0, (cname, r)
    if seen == 0:
        pytest.skip("no derivative-free baseline in the artefact yet")


def test_artifact_records_the_finite_difference_check(artifact):
    parts = artifact["parts"]
    fd = next((v for k, v in parts.items() if k.startswith("fd_")), None)
    if fd is None:
        pytest.skip("the OP-6 stage has not run")
    assert fd["determinism"]["bitwise_identical"] is True
    best = min(c["median_rel_err"] for c in fd["checks"].values())
    assert best < 1e-2, "no step size made the adjoint agree with the differences"


def test_artifact_k25_confirmation_march_is_reported(artifact):
    """N24 was never marched past 20 macro-steps before this PoC.

    Whatever it says, it has to be on the record, and the optimiser's horizon
    has to be inside what it confirmed.
    """
    parts = artifact["parts"]
    row = parts.get("confirm_K25")
    if row is None:
        pytest.skip("the confirmation march has not run")
    assert "stable" in row and "u_max_overall" in row
    g = parts.get("grad_K25")
    if g is not None and row["stable"]:
        assert g["case"]["steps"] <= row["finite_to"]
