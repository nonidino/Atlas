r"""W305 -- the trajectory agent, and MECH on b-c.

The last part of the original brief. R0's ordering named the trajectory agent
(*"lumped, dim M = 1 -- the cheapest of all"*) and it was never built; MECH is
declared on seven edges of `rocket.py` and no channel has ever carried it.

W305's own definition of done, from the worklist:

    A MECH response on b-c whose null space is measured against the
    rigid-body count, coupled to the trajectory agent rather than ahead
    of it -- the path W97 closed at CS-10.

**Three things this script establishes, and two of them contradict the row that
asked for them.**

1.  **The trajectory agent's port is 3-dimensional, not 1.**  A planar rigid
    body has two translations and one rotation, and `ThermoStruct2D._rigid_modes`
    returns a ``[2n, 3]`` basis.  One would be the count for a single scalar
    channel and this bond is not one.

2.  **The b-c MECH seam has no null space to measure.**  Its singular values
    decay geometrically -- by about a factor of three per mode after the second
    -- with **no gap**, so the null dimension is whatever tolerance is chosen:
    13 at 1e-2, 3 at 1e-8, 0 at 1e-10.  The row asked for a number the operator
    does not have.  That is a statement about `expected_null_dim` as a
    declaration, not about this rocket.

3.  **The gas cannot respond at all.**  `Compressible2D` has eight boundary
    kinds and none takes a wall velocity: `_reflect(no_slip=True)` negates the
    ghost momentum, which is a stationary wall, and `wall_noslip` reads only
    `T_wall` and `no_slip_mask`.  So the MECH bond at every gas-solid seam here
    is one-sided by a **missing capability**, not by a modelling choice -- and a
    declaration cannot see the difference.

**And the coupling variable already exists inside the build repo.**
`_solve_free` solves the bordered system ``[[K, V], [V^T, 0]] [u, l] = [f, 0]``
and returns ``lu.solve(rhs)[: f.size]`` -- it computes the Lagrange multiplier
and slices it off on the next character.  Because ``K V = 0`` and ``V`` is
orthonormal, ``l = V^T f`` exactly: the net force and torque on the body, which
is what `trajectory.rk4_step` integrates.  The structure and the trajectory
already share an interface variable and nothing declares it.

    python scripts/w305_trajectory_and_mech.py --json out/w305.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE

STAGES = ("setup", "bc_mech", "onesided", "rigid", "trajectory", "seam",
          "compile", "constants")

#: Tolerances the effective rank is read at. The point of the sweep is that no
#: choice is privileged, which is only visible if several are shown.
RANK_TOLS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)

#: Relative step for the hand-built Jacobians. MECH is affine in the trace --
#: the thermal load and the ambient background do not vanish -- so every
#: operator here is a DIFFERENCE, never a raw response.
EPS_REL = 1.0e-3


def _fmt(x) -> str:
    if isinstance(x, str):
        return x
    try:
        return "%.6e" % float(x)
    except (TypeError, ValueError):
        return repr(x)


# ---------------------------------------------------------------------------


def stage_setup(out: dict, a) -> None:
    print("=== the two agents this row is about ===")
    t0 = time.perf_counter()
    sh = RE.ShellAgent(dt=a.dt_shell)
    gas = RE.ChamberGasAgent(dt=a.dt_gas)
    h = RE.gas_h_on_shell_seam(gas, sh, RE.T_SHELL_COLD)
    marched = sh.march(RE.T_CHAMBER, h, burn_time=a.burn)
    traj = RE.TrajectoryAgent(dt=a.dt_shell)
    print("  shell: %d inner-face cells, %d on the b-c window; marched %.2f s "
          "to %.4f K" % (sh._n_face, sh.n_seam, marched["burn_time_s"],
                         marched["T_wall_mean_K"]))
    print("  trajectory: state %s" % (RE._trajectory_module().STATE_FIELDS,))
    print("              initial %s" % np.array2string(traj.state, precision=4))
    print("              dt = %g s, port dimension %d, validity %s"
          % (traj.dt, traj.n_seam, traj.validity()))
    print("  setup %.2f s" % (time.perf_counter() - t0))
    out["_shell"], out["_gas"], out["_traj"] = sh, gas, traj
    out["setup"] = dict(
        n_face=int(sh._n_face), n_seam=int(sh.n_seam),
        dt_shell=a.dt_shell, dt_gas=a.dt_gas, burn=a.burn,
        traj_dt=traj.dt, traj_port_dim=int(traj.n_seam),
        traj_state=[float(x) for x in traj.state],
        traj_validity=bool(traj.validity()),
        marched={k: v for k, v in marched.items() if k != "curve"})


def _mech_jacobians(sh, base, eps):
    n = sh.n_seam
    v0, L0 = sh.respond_mech("b:MECH", base), sh.rigid_load(base)
    J = np.zeros((n, n))
    G = np.zeros((3, n))
    for k in range(n):
        e = base.copy()
        e[k] += eps
        J[:, k] = (sh.respond_mech("b:MECH", e) - v0) / eps
        G[:, k] = (sh.rigid_load(e) - L0) / eps
    return J, G, v0, L0


def stage_bc_mech(out: dict, a) -> None:
    print("=== the MECH response on b-c, and the null space it does not have ===")
    sh = out["_shell"]
    n = sh.n_seam
    base = np.full(n, RE.P_CHAMBER)
    t0 = time.perf_counter()
    J, G, v0, _L0 = _mech_jacobians(sh, base, EPS_REL * RE.P_CHAMBER)
    wall = time.perf_counter() - t0
    s = np.linalg.svd(J, compute_uv=False)
    print("  %d x %d operator in %.2f s -- a MECH response is %.3f s, against "
          "minutes for a gas one" % (n, n, wall, wall / (n + 1)))
    print("  the trace is a normal traction [Pa]; the flow is du/dt on the same "
          "cells [m/s]")
    print()
    print("  singular values:")
    for i, x in enumerate(s):
        print("    %2d  %.6e" % (i, x))
    print()
    print("  beta = %.6e   kappa = %.4e" % (s[-1], s[0] / s[-1]))
    print()
    print("  effective rank, read at several tolerances:")
    ranks = {}
    for tol in RANK_TOLS:
        r = int(np.sum(s > tol * s[0]))
        ranks[tol] = r
        print("    tol %-9g rank %2d   null dim %2d" % (tol, r, n - r))
    ratios = (s[:-1] / s[1:]).tolist()
    print()
    print("  consecutive ratios: %s"
          % np.array2string(np.array(ratios), precision=2, max_line_width=100))
    tail = [r for r in ratios[2:]]
    print("  after the second mode the largest ratio is %.2f -- **no gap**, so "
          "the null" % max(tail))
    print("  dimension is a property of the tolerance and not of the operator.")
    out["bc_mech"] = dict(
        singular_values=[float(x) for x in s], beta=float(s[-1]),
        kappa=float(s[0] / s[-1]), ranks={str(k): v for k, v in ranks.items()},
        ratios=[float(r) for r in ratios], max_tail_ratio=float(max(tail)),
        wall_s=wall, calls=int(sh.solver_calls),
        response_at_base=[float(x) for x in v0])
    out["_J"], out["_G"] = J, G


def stage_onesided(out: dict, a) -> None:
    print("=== the half of this bond the build repo cannot carry ===")
    src = os.path.join(RE.build_repo(), "src", "atlas", "solvers",
                       "compressible2d.py")
    body = open(src, encoding="utf-8").read()
    flat = " ".join(body.split())
    #: Matched WITHOUT the set-membership glyph the build repo's docstring
    #: uses: this console is cp1252 and a non-ASCII literal in a script that
    #: prints is a latent crash, which is a standing note here.
    checks = {
        "bc_docstring_lists_eight_kinds":
            all(k in flat for k in RE.COMPRESSIBLE2D_BC_KINDS)
            and "wall_slip, wall_noslip, symmetry, inlet_massflow, "
                "freestream, outflow, prescribed, extrapolate}" in flat,
        "noslip_negates_momentum":
            "out[..., 1] = -U[..., 1]" in flat and "out[..., 2] = -U[..., 2]" in flat,
        "wall_reads_T_wall": '"T_wall" in bc.params' in flat,
        "wall_reads_no_slip_mask": '"no_slip_mask" in bc.params' in flat,
        "no_wall_velocity_param": ("u_wall" not in flat and "v_wall" not in flat
                                   and "wall_velocity" not in flat),
    }
    for k, v in checks.items():
        print("  %-34s %s" % (k, v))
    print()
    print("  So a gas agent can SUPPLY a traction -- it has the wall pressure --")
    print("  and cannot RESPOND to a velocity: there is no channel to impose one.")
    print("  The MECH bond at every gas-solid seam in this graph is one-sided by")
    print("  a MISSING CAPABILITY, and a declaration cannot see the difference")
    print("  between that and a modelling choice. W97 found a field-to-lumped")
    print("  MECH seam one-sided by Re_h; this is one-sided by a boundary")
    print("  condition that does not exist.")
    out["onesided"] = dict(checks=checks, source=src,
                           bc_kinds=list(RE.COMPRESSIBLE2D_BC_KINDS),
                           wall_params=list(RE.WALL_BC_PARAMS))


def stage_rigid(out: dict, a) -> None:
    print("=== the load the projection removes, which is the trajectory's input ===")
    sh = out["_shell"]
    G = out["_G"]
    sg = np.linalg.svd(G, compute_uv=False)
    print("  the rigid-load map d(V^T f)/d(traction) is 3 x %d" % sh.n_seam)
    print("  singular values: %s" % ", ".join("%.6e" % x for x in sg))
    print("  rank %d -- the seam window drives all three planar modes, the"
          % int(np.sum(sg > 1e-12 * sg[0])))
    print("  torque one %.1fx more weakly than the leading translation."
          % (sg[0] / sg[-1]))
    print()
    #: The identity `l = V^T f` is pure algebra and is pinned by the tier's own
    #: test on a synthetic system, where K, f and V are controlled. Here what is
    #: checked is the part that depends on this mesh: that the thermal load is
    #: self-equilibrated, so the rigid load is the pressure integral alone.
    V = sh._ts._rigid_modes()
    orth = float(np.abs(V.T @ V - np.eye(3)).max())
    p_amb = float(RE.ambient_state()[1])
    L_amb = sh.rigid_load(np.full(sh.n_seam, p_amb))
    print("  V is orthonormal to %.2e" % orth)
    print("  at a UNIFORM ambient traction the net rigid load is %s"
          % np.array2string(L_amb, precision=4))
    print("  -- the two FORCES cancel to machine precision and the TORQUE does")
    print("  not. That is right and it is the shell's thickness talking: the")
    print("  inner and outer faces have equal projected area, so a uniform")
    print("  pressure exerts no net force, but they sit at different radii, so")
    print("  it exerts a COUPLE. The trajectory must receive that couple and the")
    print("  structure must not, which is precisely what the projection does.")
    out["rigid"] = dict(
        singular_values=[float(x) for x in sg],
        rank=int(np.sum(sg > 1e-12 * sg[0])),
        torque_weakness=float(sg[0] / sg[-1]),
        V_orthonormality=orth,
        load_at_ambient=[float(x) for x in L_amb])


def stage_trajectory(out: dict, a) -> None:
    print("=== the trajectory agent, probed ===")
    traj = out["_traj"]
    base = traj.base_trace()
    r0 = traj.respond("c:MECH", base)
    eps = 1.0e3
    B = np.zeros((3, 3))
    for k in range(3):
        e = base.copy()
        e[k] += eps
        B[:, k] = (traj.respond("c:MECH", e) - r0) / eps
    m = float(traj.state[6])
    dt = traj.dt
    analytic = np.diag([dt / m, dt / m, dt / traj.inertia])
    err = float(np.abs(B - analytic).max() / np.abs(analytic).max())
    print("  probed block (load -> velocity over one step):")
    print(np.array2string(B, precision=10, max_line_width=100))
    print()
    print("  analytic dt/m = %.10e on the translations, dt/I = %.10e on the "
          "rotation" % (dt / m, dt / traj.inertia))
    print("  max relative departure: %.3e" % err)
    print("  off-diagonal magnitude / diagonal: %.3e"
          % float(np.abs(B - np.diag(np.diag(B))).max() / np.abs(np.diag(B)).max()))
    print()
    print("  response at the base (pure gravity): %s"
          % np.array2string(r0, precision=8))
    print("  storage %.6e J   validity %s" % (traj.storage(), traj.validity()))
    out["trajectory"] = dict(
        block=[[float(x) for x in row] for row in B],
        analytic_diag=[dt / m, dt / m, dt / traj.inertia],
        max_rel_error=err, base_response=[float(x) for x in r0],
        storage=float(traj.storage()), validity=bool(traj.validity()),
        mass=m, dt=dt, inertia=float(traj.inertia))


def stage_seam(out: dict, a) -> None:
    print("=== the c-trajectory seam, assembled ===")
    sh, traj = out["_shell"], out["_traj"]
    G, B = out["_G"], np.array(out["trajectory"]["block"])
    #: The structure's side of this bond maps a seam traction to the rigid load
    #: it hands over; the trajectory's side maps that load to a velocity. The
    #: composition is the seam's own operator in the rigid-mode space.
    S = B @ G
    sv = np.linalg.svd(S, compute_uv=False)
    print("  the structure hands over V^T f (3 x %d) and the body turns it into"
          % sh.n_seam)
    print("  a velocity (3 x 3); composed, the seam is 3 x %d." % sh.n_seam)
    print("  singular values: %s" % ", ".join("%.6e" % x for x in sv))
    print("  beta = %.6e   kappa = %.4f" % (sv[-1], sv[0] / sv[-1]))
    print()
    print("  **This bond is two-sided and the b-c MECH one is not.** The body")
    print("  responds to a load with a velocity natively -- no time derivative")
    print("  is supplied and none is declared -- while the shell's flow had to be")
    print("  du/dt at a stated clock. The contrast is the tier's point.")
    out["seam"] = dict(singular_values=[float(x) for x in sv],
                       beta=float(sv[-1]), kappa=float(sv[0] / sv[-1]),
                       shape=list(S.shape))


def stage_compile(out: dict, a) -> None:
    print("=== what the compiler says about the graph as it stands ===")
    from atlas.compiler import compile_scheme
    t0 = time.perf_counter()
    graph, _experts = RE.build_rocket_real(dt_scale=a.dt_scale)
    res = compile_scheme(graph)
    counts = {}
    for d in res.decisions.decisions:
        counts[d.verdict.value] = counts.get(d.verdict.value, 0) + 1
    print("  verdict %s in %.0f s" % (res.verdict.value, time.perf_counter() - t0))
    print("  %s" % counts)
    print()
    print("  This is the SEVEN-agent graph, unchanged. The trajectory agent is")
    print("  built and measured here and is deliberately NOT added to it -- see")
    print("  the tier's record for why.")
    out["compile"] = dict(verdict=res.verdict.value, counts=counts,
                          wall_s=time.perf_counter() - t0)


def stage_constants(out: dict, a) -> None:
    print("=== what this row publishes ===")
    bc = out.get("bc_mech", {})
    tr = out.get("trajectory", {})
    rg = out.get("rigid", {})
    sm = out.get("seam", {})
    print("  b-c MECH   beta %s  kappa %s" % (_fmt(bc.get("beta")),
                                              _fmt(bc.get("kappa"))))
    print("             null dim: 13 at 1e-2, %s at 1e-8, %s at 1e-10"
          % (bc.get("ranks", {}).get("1e-08") is not None
             and 14 - bc["ranks"]["1e-08"],
             bc.get("ranks", {}).get("1e-10") is not None
             and 14 - bc["ranks"]["1e-10"]))
    print("  c-traj     beta %s  kappa %s  dim M 3"
          % (_fmt(sm.get("beta")), _fmt(sm.get("kappa"))))
    print("  rigid map  rank %s, torque weaker by %s"
          % (rg.get("rank"), _fmt(rg.get("torque_weakness"))))
    print("  trajectory block is diagonal to %s" % _fmt(tr.get("max_rel_error")))
    out["constants"] = dict(bc_beta=bc.get("beta"), bc_kappa=bc.get("kappa"),
                            seam_beta=sm.get("beta"), seam_kappa=sm.get("kappa"),
                            traj_block_error=tr.get("max_rel_error"))


# ---------------------------------------------------------------------------
# predictions, registered for the arms that had not run when they were written
# ---------------------------------------------------------------------------


def predictions(out: dict) -> list[dict]:
    """**Registered 2026-09-21, and the honest scoping matters.**

    The b-c MECH spectrum and the rigid-load rank were measured EXPLORATORILY
    before these were written, so they are not predictions and are not listed as
    such -- they are pinned by the tier's tests instead. What had not run when
    these were registered is the trajectory agent's probed block, the composed
    c-trajectory seam, and the compile.
    """
    tr, sm, cm, rg = (out.get("trajectory", {}), out.get("seam", {}),
                      out.get("compile", {}), out.get("rigid", {}))
    P = []

    def add(name, claim, got, ok):
        P.append(dict(name=name, claim=claim, got=got, held=bool(ok)))

    add("R1", "the trajectory agent's probed block is EXACTLY diagonal "
              "dt/m, dt/m, dt/I to better than 1e-6 relative",
        tr.get("max_rel_error"),
        tr.get("max_rel_error") is not None and tr["max_rel_error"] < 1e-6)
    add("R2", "the c-trajectory seam is 3-dimensional, not the 1 the brief named",
        sm.get("shape"), sm.get("shape") == [3, 14])
    add("R3", "the composed c-trajectory seam is far better conditioned than "
              "b-c MECH: kappa below 1e3 against b-c's 1e8",
        sm.get("kappa"),
        sm.get("kappa") is not None and sm["kappa"] < 1.0e3)
    add("R4", "the rigid-load map has full rank 3 -- the seam window drives "
              "every planar mode",
        rg.get("rank"), rg.get("rank") == 3)
    add("R5", "the seven-agent graph still refuses, and adding nothing to it "
              "leaves the count where Tier 79 left it",
        cm.get("counts"),
        cm.get("verdict") == "refuse" and cm.get("counts", {}).get("refuse") == 4)
    return P


def _persist(out: dict, a) -> None:
    if not a.json:
        return
    path = os.path.abspath(a.json)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = {k: v for k, v in out.items() if not k.startswith("_")}
    tmp = path + ".tmp"
    for attempt in range(5):
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(body, fh, indent=2, default=float)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.6 * (attempt + 1))
    print("  WARNING: could not persist to %s" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH", default=None)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--burn", type=float, default=RE.T_PROBE_BURN)
    ap.add_argument("--dt-shell", type=float, default=5.0e-2)
    ap.add_argument("--dt-gas", type=float, default=1.0e-5)
    ap.add_argument("--dt-scale", type=float, default=1.0e-3)
    a = ap.parse_args(argv)

    want = (list(STAGES) if a.stages == "all"
            else [s.strip() for s in a.stages.split(",")])
    bad = [s for s in want if s not in STAGES]
    if bad:
        ap.error("unknown stage(s): %s" % bad)
    if "setup" not in want:
        want = ["setup"] + want

    print("=" * 84)
    print("W305 -- the trajectory agent, and MECH on b-c")
    print("=" * 84)
    out: dict = {}
    t0 = time.perf_counter()
    for s in want:
        print()
        globals()["stage_" + s](out, a)
        _persist(out, a)

    print()
    print("=" * 84)
    print("PREDICTIONS")
    print("=" * 84)
    P = predictions(out)
    for p in P:
        print("  %-4s %-6s %s" % (p["name"], "HELD" if p["held"] else "FAILED",
                                  p["claim"]))
        print("       got: %s" % (p["got"],))
    print()
    print("  %d of %d held." % (sum(1 for p in P if p["held"]), len(P)))
    out["predictions"] = P
    out["wall_seconds"] = time.perf_counter() - t0
    print("  wall %.0f s" % out["wall_seconds"])
    _persist(out, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
