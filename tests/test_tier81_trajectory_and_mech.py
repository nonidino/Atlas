"""Tier 81 -- W305: the trajectory agent, and MECH on b-c.

The last part of the original brief. What this file pins is mostly the ways
W305's own definition of done turned out not to be answerable as written:

* **the port is 3-dimensional, not 1** -- a planar rigid body has two
  translations and one rotation, and the structure's own `_rigid_modes` returns
  a ``[2n, 3]`` basis;
* **the b-c MECH operator has no null space to measure** -- its singular values
  decay geometrically with no gap, so the null dimension is a property of the
  tolerance (13 at 1e-2, 3 at 1e-8, 0 at 1e-10) and not of the operator. The
  test for this asserts that two tolerances DISAGREE, which is the only way to
  state "there is no answer here" as something that can fail;
* **the gas cannot respond at all** -- `Compressible2D` has no wall-velocity
  boundary condition, so the bond is one-sided by a missing capability;
* **the coupling variable already exists in the build repo** -- `_solve_free`
  computes ``l = V^T f`` as a Lagrange multiplier and slices it off. That
  identity is pinned here as pure algebra on a synthetic system, where K, f and
  V are controlled, rather than on the real shell where it cannot be observed
  without changing the build repo.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from atlas.cases import rocket_experts as RE

W305 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "out", "w305.json")


def _have_expert() -> bool:
    try:
        RE.load_rocket_modules()
        return True
    except Exception:
        return False


needs_expert = pytest.mark.skipif(
    not _have_expert(), reason="the build repo is not importable; set ATLAS_BUILD_REPO")


def _results():
    if not os.path.exists(W305):
        pytest.skip("out/w305.json not present; run scripts/w305_...")
    with open(W305, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def shell():
    sh = RE.ShellAgent(dt=5.0e-2)
    gas = RE.ChamberGasAgent(dt=1.0e-5)
    h = RE.gas_h_on_shell_seam(gas, sh, RE.T_SHELL_COLD)
    sh.march(RE.T_CHAMBER, h, burn_time=RE.T_PROBE_BURN)
    return sh


# ===========================================================================
# 1. the trajectory agent
# ===========================================================================


@needs_expert
def test_the_trajectory_agent_uses_the_build_repo_verbatim():
    """No physics is written here; the module's own docstring says so.

    `solvers/trajectory.py` states it is *"written once and reused verbatim by
    Atlas's non-learned `rigid_body` expert ... not a data-generation helper
    that gets replaced later"*. The wrapper must therefore call it, not
    reimplement it.
    """
    t = RE.TrajectoryAgent()
    mod = RE._trajectory_module()
    assert mod.STATE_FIELDS == ("x", "y", "theta", "vx", "vy", "omega", "m")
    assert t._traj is mod
    src = os.path.join(RE.build_repo(), "src", "atlas", "solvers", "trajectory.py")
    body = " ".join(open(src, encoding="utf-8").read().split())
    assert "reused **verbatim** by Atlas's non-learned `rigid_body` expert" in body
    # And the integrator is RK4 by an explicit instruction not to "upgrade" it.
    # Matched on a fragment that does not span a line break: the note is a
    # block quote, so flattening leaves the "> " markers between its lines.
    assert "RK4 is correct here; do not \"upgrade\" it to" in body
    assert "leapfrog." in body


@needs_expert
def test_W305_the_port_is_three_dimensional_not_one():
    """The brief called it 'lumped, dim M = 1 -- the cheapest of all'.

    A planar rigid body has two translations and one rotation. The structure's
    own rigid-mode basis is the authority and it is [2n, 3].
    """
    t = RE.TrajectoryAgent()
    assert t.n_seam == 3
    assert t.seam_weights().shape == (3,)
    sh = RE.ShellAgent(dt=5.0e-2)
    V = sh._ts._rigid_modes()
    assert V.shape == (2 * sh._ts.mesh.n_nodes, 3)
    np.testing.assert_allclose(V.T @ V, np.eye(3), atol=1e-12)


@needs_expert
def test_the_trajectory_block_is_exactly_diagonal():
    """load -> velocity over one step is dt/m, dt/m, dt/I and nothing else.

    RK4 on this right-hand side is linear in the applied load at fixed mass, so
    the block is analytic. The off-diagonals must be EXACTLY zero: a planar
    body's three modes do not mix under a load.
    """
    t = RE.TrajectoryAgent(dt=5.0e-2)
    base = t.base_trace()
    r0 = t.respond("c:MECH", base)
    B = np.zeros((3, 3))
    for k in range(3):
        e = base.copy()
        e[k] += 1.0e3
        B[:, k] = (t.respond("c:MECH", e) - r0) / 1.0e3
    m, dt = float(t.state[6]), t.dt
    want = np.diag([dt / m, dt / m, dt / t.inertia])
    np.testing.assert_allclose(B, want, rtol=1e-6)
    off = B - np.diag(np.diag(B))
    assert np.all(off == 0.0), "the three modes must not mix"
    # The base response is gravity over one step -- but NOT exactly -g*dt:
    # RK4 samples the right-hand side at four points across the step, the body
    # falls about 0.012 m in that time, and `atmosphere.gravity` varies with
    # altitude. The residual is ~1e-9 relative and it is the altitude
    # dependence, not integrator noise.
    g = RE._atmosphere_module().gravity(float(t.state[1]))
    assert r0[1] == pytest.approx(-g * dt, rel=1e-6)
    assert r0[1] != -g * dt, "exact equality would mean gravity was frozen"
    assert abs(r0[1] / (-g * dt) - 1.0) < 1e-7


@needs_expert
def test_the_trajectory_validity_can_decline():
    """A predicate that cannot say no is not a predicate.

    Two ways it declines, both read off the sources: `atmosphere.py` is the 1976
    standard tabulated to 86 km, and `rhs` floors the mass at 1e-6 kg so a burn
    past the propellant load returns a number rather than failing.
    """
    t = RE.TrajectoryAgent()
    assert t.validity() is True
    s = t.state.copy()
    s[6] = 0.5
    assert t.validity(s) is False, "a spent vehicle must decline"
    s = t.state.copy()
    s[1] = 120.0e3
    assert t.validity(s) is False, "above the atmosphere table must decline"


# ===========================================================================
# 2. MECH on b-c, and the null space it does not have
# ===========================================================================


@needs_expert
def test_W305_the_mech_null_dimension_is_a_property_of_the_tolerance(shell):
    """**The row asked for a number the operator does not have.**

    W305 wanted "a MECH response on b-c whose null space is measured against the
    rigid-body count". Measured, the singular values decay geometrically -- by
    about three per mode after the second -- with no gap anywhere. So the null
    dimension is whatever tolerance is chosen.

    This test asserts the DISAGREEMENT, which is the only way to state "there is
    no answer here" as something that can fail: if the operator ever grew a
    genuine null space, two tolerances four decades apart would agree on it.
    """
    n = shell.n_seam
    base = np.full(n, RE.P_CHAMBER)
    eps = 1.0e-3 * RE.P_CHAMBER
    v0 = shell.respond_mech("b:MECH", base)
    J = np.column_stack([
        (shell.respond_mech("b:MECH", base + eps * np.eye(n)[k]) - v0) / eps
        for k in range(n)])
    s = np.linalg.svd(J, compute_uv=False)

    null = {tol: n - int(np.sum(s > tol * s[0]))
            for tol in (1e-2, 1e-4, 1e-8, 1e-10)}
    assert len(set(null.values())) >= 3, (
        "the null dimension must DISAGREE across tolerances; got %r" % null)
    assert null[1e-2] > null[1e-10], null

    # and the reason: no gap. After the first two modes the largest jump is
    # small, so nothing separates "signal" from "null".
    ratios = s[:-1] / s[1:]
    assert max(ratios[2:]) < 8.0, (
        "a real null space would show a gap here: %r" % ratios.tolist())
    assert s[0] / s[-1] > 1.0e6, "and the operator is severely ill-conditioned"


@needs_expert
def test_the_mech_seam_is_far_worse_conditioned_than_the_therm_one(shell):
    """Same 14 cells, same seam, eight orders of magnitude apart.

    Tier 77 measured b-c THERM at kappa = 3.7141. MECH here is ~1e8. The reason
    is structural: the THERM response is nearly algebraic (W301 -- the wall flux
    subtracts T_wall directly) while the MECH response is a quasi-static
    elliptic solve, which is a SMOOTHING operator and so has geometrically
    decaying singular values.
    """
    n = shell.n_seam
    base = np.full(n, RE.P_CHAMBER)
    eps = 1.0e-3 * RE.P_CHAMBER
    v0 = shell.respond_mech("b:MECH", base)
    J = np.column_stack([
        (shell.respond_mech("b:MECH", base + eps * np.eye(n)[k]) - v0) / eps
        for k in range(n)])
    kappa = float(np.linalg.cond(J))
    assert kappa > 1.0e7, kappa
    # THERM's corrected kappa: 3.7141 at Tier 77 with the old wall, 2.3914
    # re-measured at Tier 86 (W334) -- the comparison only got starker
    assert kappa / 2.3914 > 1.0e6, "MECH is millions of times worse than THERM"


@needs_expert
def test_W305_the_gas_has_no_wall_velocity_boundary_condition():
    """One-sided by a MISSING CAPABILITY, read off the solver rather than claimed.

    `_reflect(no_slip=True)` negates the ghost momentum -- a stationary wall --
    and `wall_noslip` reads only `T_wall` and two per-station masks,
    `no_slip_mask` and (since Tier 86) `isothermal_mask`. There is no channel
    to impose a wall velocity, so the gas can supply a traction and cannot
    respond to one.
    """
    src = os.path.join(RE.build_repo(), "src", "atlas", "solvers",
                       "compressible2d.py")
    if not os.path.exists(src):
        pytest.skip("build repo checkout not present")
    flat = " ".join(open(src, encoding="utf-8").read().split())
    assert "out[..., 1] = -U[..., 1]" in flat and "out[..., 2] = -U[..., 2]" in flat
    for p in RE.WALL_BC_PARAMS:
        assert ('"%s" in bc.params' % p in flat) or ('bc.params.get("%s")' % p in flat), p
    for name in ("u_wall", "v_wall", "wall_velocity", "wall_speed"):
        assert name not in flat, "found a wall-velocity channel after all: %s" % name
    for k in RE.COMPRESSIBLE2D_BC_KINDS:
        assert k in flat


# ===========================================================================
# 3. the coupling variable the build repo computes and discards
# ===========================================================================


def test_the_lagrange_multiplier_is_exactly_V_transpose_f():
    """Pure algebra, on a synthetic system where K, f and V are controlled.

    `_solve_free` solves [[K, V], [V^T, 0]] [u, l] = [f, 0]. Since K V = 0 and V
    is orthonormal, left-multiplying the first block row by V^T gives l = V^T f.
    That is the net rigid load -- what `trajectory.rk4_step` integrates -- and
    the build repo computes it and returns `lu.solve(rhs)[: f.size]`, slicing it
    off on the next character.
    """
    rng = np.random.default_rng(81)
    n = 12
    V, _ = np.linalg.qr(rng.standard_normal((n, 3)))
    P = np.eye(n) - V @ V.T
    A = rng.standard_normal((n, n))
    K = P @ (A @ A.T + np.eye(n)) @ P          # SPD on the complement, null on V
    np.testing.assert_allclose(K @ V, 0.0, atol=1e-10)

    f = rng.standard_normal(n)
    big = np.block([[K, V], [V.T, np.zeros((3, 3))]])
    sol = np.linalg.solve(big, np.concatenate([f, np.zeros(3)]))
    u, lam = sol[:n], sol[n:]
    np.testing.assert_allclose(lam, V.T @ f, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(V.T @ u, 0.0, atol=1e-10)


@needs_expert
def test_a_uniform_pressure_is_a_couple_and_not_a_force(shell):
    """The thickness talking, and it is the check that the map is physical.

    Inner and outer faces have equal projected area, so a uniform pressure
    exerts NO net force; they sit at different radii, so it exerts a couple.
    Both halves are asserted, because only the pair rules out a sign error.
    """
    p_amb = float(RE.ambient_state()[1])
    L = shell.rigid_load(np.full(shell.n_seam, p_amb))
    assert abs(L[0]) < 1e-9 and abs(L[1]) < 1e-9, L
    assert abs(L[2]) > 1e-3, L


@needs_expert
def test_the_rigid_load_map_drives_every_planar_mode(shell):
    """Rank 3: a 14-cell window is enough to move the whole body."""
    n = shell.n_seam
    base = np.full(n, RE.P_CHAMBER)
    eps = 1.0e-3 * RE.P_CHAMBER
    L0 = shell.rigid_load(base)
    G = np.column_stack([
        (shell.rigid_load(base + eps * np.eye(n)[k]) - L0) / eps
        for k in range(n)])
    assert G.shape == (3, n)
    sg = np.linalg.svd(G, compute_uv=False)
    assert int(np.sum(sg > 1e-12 * sg[0])) == 3
    assert sg[0] / sg[-1] > 10.0, "the torque mode is much weaker, and that is real"


# ===========================================================================
# 4. the record
# ===========================================================================


def test_the_record_carries_the_tolerance_sweep():
    res = _results()
    bc = res["bc_mech"]
    nulls = {t: 14 - r for t, r in bc["ranks"].items()}
    assert len(set(nulls.values())) >= 3, nulls
    assert bc["kappa"] > 1e7
    assert res["seam"]["shape"] == [3, 14]
    assert res["seam"]["kappa"] < 1e3
    assert res["compile"]["verdict"] == "refuse"
    assert res["compile"]["counts"]["refuse"] == 4


def test_all_registered_predictions_held():
    res = _results()
    failed = [p["name"] for p in res["predictions"] if not p["held"]]
    assert not failed, failed


def test_R5_prose_is_checked_against_the_tier_79_record():
    """R5 says the seven-agent graph is left 'where Tier 79 left it', and its
    evaluator checks only the verdict and the four refusals -- more permissive
    than its prose, found when W334 re-derived both records (Tier 86) and the
    counts moved, 22 -> 20 uncertified admits, with R5 still reading held.

    The prose's claim is checked here instead, against Tier 79's own record: the
    uncertified admits are Tier 79's decertifications and the refusals are its
    refusals. It held on the old wall (22 = 22) and holds on the fixed one
    (20 = 20); the registered evaluator is left as it was registered.
    """
    res = _results()
    w310 = os.path.join(os.path.dirname(W305), "w310.json")
    if not os.path.exists(w310):
        pytest.skip("out/w310.json not present")
    with open(w310, encoding="utf-8") as fh:
        t79 = json.load(fh)["compile"]
    counts = res["compile"]["counts"]
    assert res["compile"]["verdict"] == t79["verdict"] == "refuse"
    assert counts["refuse"] == t79["n_refusals"]
    assert counts["admit-uncertified"] == t79["n_decertifications"]
