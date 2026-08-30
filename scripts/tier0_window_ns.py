"""Tier 0 of `gap-worklist`, measured against `reference.WindowNS`.

W1 (fit L), W2 (assemble one real Lambda-tilde) and W3 (the three-way defect
split) run here, on `atlas/cases/window_ns.py` -- the only case study whose
``boundary_response`` is a solver rather than a matrix.

**Rewritten 2026-08-27 after the first run's findings.** That run measured a
`tau` of 3.7e-4 that was 99.8% of the defect while the agent WAS the reference
solver, and a probed-DtN interface solve that made one composed step 8.9x worse
than not solving it. Both are understood and both are fixed:

  R10   a projection method's pressure solve is global over whatever domain it
        runs on, so decomposing the domain decomposes the operator. The
        composition layer takes the elliptic part and runs it globally.
  R2b   flux balance is the interface condition of a boundary-value problem, and
        an explicitly stepped agent poses none over a macro-step. The rung stays
        at `dirichlet` and the coupling is an overlapping halo.
  halo  the overlap must outrun the agent's own domain of dependence, and the
        partition of unity must vanish at each window's artificial edge.

Both schemes are run, because the comparison is the result.

    s0   develop a base state on the monolith (n = 255) and persist it
    w2   Lambda-tilde: kappa, beta, null dim, mu beside pi; the flux-convention
         comparison; Xi and ||Lambda - Lambda_ref|| between the two agents
    w1   paired rollouts; fit ||e^n|| against L^n, per scheme
    w3   one macro-step split into tau + sigma + gamma against the monolith,
         with the assembly certificate and the depth tag

Run:  python scripts/tier0_window_ns.py [--stages s0,w2,w1,w3] [--out out/tier0b]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import ProbeBudget, compile_scheme                      # noqa: E402
from atlas.assembly import certify                                 # noqa: E402
from atlas.cases import window_ns as W                             # noqa: E402
from atlas.probe import assemble_seam, operator_drift              # noqa: E402

SCHEMES = ("as-built", "split-step")


class Out:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.log_path = os.path.join(root, "run.log")

    def say(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def json(self, name: str, obj) -> None:
        path = os.path.join(self.root, name + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=_jsonable)
        self.say(f"wrote {path}")

    def npz(self, name: str, **arrays) -> None:
        p = os.path.join(self.root, name + ".npz")
        np.savez_compressed(p, **arrays)
        self.say(f"wrote {p}")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ---------------------------------------------------------------------------
# the physics harness -- the system under measurement, not part of `atlas/`
# ---------------------------------------------------------------------------


def body_force(x_d=1.0, y_d=1.0, halfspan=0.5, thickness=0.25, thrust=0.4444):
    """An actuator strip in the lower-left window, at the wind farm's thrust.

    Frozen rather than read from the local inflow: a thrust that responds to its
    own wake is a second feedback loop inside every quantity below.
    """
    n, h = W.MONO_N, W.H
    g1 = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(g1, g1)
    g = np.exp(-(((X - x_d) / (0.5 * thickness)) ** 2))
    g = g / (g.sum(axis=1, keepdims=True) * h + 1e-300)
    return -(thrust / (2.0 * halfspan)) * g * (np.abs(Y - y_d) <= halfspan)


def rel_l2(a, b, a2, b2) -> float:
    return float(np.sqrt((np.sum((a - b) ** 2) + np.sum((a2 - b2) ** 2))
                         / (np.sum(b ** 2) + np.sum(b2 ** 2))))


def make_solvers(tiling, nu=W.NU):
    ref = W.load_reference()
    mono = ref.WindowNS(nu=nu, length=W.MONO_L, n=W.MONO_N, cfl=0.4,
                        transmission="dirichlet")
    embedded = ref.WindowNS(nu=nu, length=tiling.n * W.H, n=tiling.n, cfl=0.4,
                            transmission="dirichlet")
    exposed = W._no_projection_class()(nu=nu, length=tiling.n * W.H, n=tiling.n,
                                       cfl=0.4, transmission="dirichlet")
    return mono, embedded, exposed


def mono_step(mono, u, v, dt, fx):
    u1, v1 = mono.step_batch(u[None], v[None], dt, bc0=None, bc1=None,
                             force=(fx[None], np.zeros_like(fx)[None]))
    return u1[0], v1[0]


def _pin_domain(U, V, ring):
    for A, R in ((U, ring[0]), (V, ring[1])):
        A[0, :] = R[0, :]; A[-1, :] = R[-1, :]
        A[:, 0] = R[:, 0]; A[:, -1] = R[:, -1]
    return U, V


def flat_assemble(tiling, us, vs):
    """chi = 1 / (owner count): the assembly the first measurement used."""
    M, n = W.MONO_N, tiling.n
    au, av, c = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    for k, (ox, oy) in enumerate(tiling.offsets):
        au[oy:oy + n, ox:ox + n] += us[k]
        av[oy:oy + n, ox:ox + n] += vs[k]
        c[oy:oy + n, ox:ox + n] += 1.0
    return au / c, av / c


def composed_step(scheme, tiling, solvers, u, v, dt, fx, ring, substeps=None,
                  exact_ring=None):
    """One composed macro-step under either scheme.

    ``exact_ring`` supplies the reference solution's own boundary data, which is
    what makes the result `tau` rather than the total defect.

    **``substeps`` defaults to the agent's own count at this dt, not to 10.**
    R10b: the composition layer owns the elliptic part and applies it once per
    exchange, so exchanging at a different cadence from the agent's own
    sub-stepping makes the composed step a different SPLITTING from the reference.
    Measured 2026-08-28: hard-coding 10 at dt = 0.025 costs two orders in tau
    (8.0e-7 -> 1.6e-4) with sigma unchanged, i.e. the whole defect is charged to
    the agent while being a property of this function.
    """
    mono, embedded, exposed = solvers
    if substeps is None:
        substeps = W.substeps_at(dt, mono)
    fxs, zero = tiling.cut(fx), np.zeros((4, tiling.n, tiling.n))
    if scheme == "as-built":
        us, vs = tiling.cut(u), tiling.cut(v)
        b1 = None if exact_ring is None else (tiling.cut(exact_ring[0]),
                                              tiling.cut(exact_ring[1]))
        u1, v1 = embedded.step_batch(us, vs, dt, bc0=(us, vs), bc1=b1,
                                     force=(fxs, zero))
        return flat_assemble(tiling, u1, v1)

    U, V = u.copy(), v.copy()
    hs = dt / substeps
    for m in range(substeps):
        us, vs = tiling.cut(U), tiling.cut(V)
        if exact_ring is not None:
            # ramp the ring toward the reference across the macro-step
            s = (m + 1) / substeps
            ru = (1 - s) * u + s * exact_ring[0]
            rv = (1 - s) * v + s * exact_ring[1]
            b0 = (tiling.cut(ru), tiling.cut(rv))
        else:
            b0 = (us, vs)
        u1, v1 = exposed.step_batch(us, vs, hs, bc0=b0, bc1=None, force=(fxs, zero))
        U, V = tiling.assemble(u1, v1)
        pu, pv = mono._project(U[None], V[None])
        U, V = _pin_domain(pu[0], pv[0], ring)
    return U, V


def streamfunction_blob(amp, x0=0.5, y0=1.0, r=0.15):
    """A divergence-free perturbation, so it injects no pressure transient."""
    n, h = W.MONO_N, W.H
    g = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(g, g)
    psi = amp * np.exp(-(((X - x0) ** 2 + (Y - y0) ** 2) / r ** 2))
    return np.gradient(psi, h, axis=0), -np.gradient(psi, h, axis=1)


def probe_all_seams(graph, budget, probe_state, references=None):
    from atlas.compiler import _Context, _derive_transfer          # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.verdict import DecisionRecord

    ctx = _Context(graph=graph, budget=None, probe_budget=budget, references={},
                   record=DecisionRecord(), stamp=EnvelopeStamp(), holes=HoleLedger(),
                   probe_state=probe_state, depth=0)
    ops = {}
    for conn in graph.connections:
        tr = _derive_transfer(ctx, conn)
        ops[conn.seam_id] = (assemble_seam(graph, conn, tr, budget, references or {},
                                           expected_null_dim=conn.expected_null_dim,
                                           probe_state=probe_state), tr)
    return ops


# ---------------------------------------------------------------------------
# S0 -- the base state
# ---------------------------------------------------------------------------


def stage_s0(out, t_dev=5.0, dt=W.MACRO_DT, nu=W.NU):
    mono, _e, _x = make_solvers(W.DEFAULT_TILING, nu)
    fx = body_force()
    u = np.full((W.MONO_N, W.MONO_N), W.U_INF)
    v = np.zeros_like(u)
    hist, t0 = [], time.perf_counter()
    n_steps = int(round(t_dev / dt))
    for k in range(n_steps):
        up, vp = u, v
        u, v = mono_step(mono, u, v, dt, fx)
        hist.append(rel_l2(u, up, v, vp) / dt)
        if k % 25 == 0 or k == n_steps - 1:
            out.say(f"  s0 step {k+1}/{n_steps}  |dU/dt|/|U| = {hist[-1]:.3e}")
    j = int(1.0 / W.H)
    out.npz("s0_state", u=u, v=v, fx=fx)
    out.json("s0", {
        "t_dev": t_dev, "dt": dt, "nu": nu, "n": W.MONO_N, "h": W.H,
        "seconds": time.perf_counter() - t0,
        "substeps_per_macro_step": mono.last_substeps,
        "steady_residual_final": hist[-1], "steady_residual_history": hist,
        "centreline_u": {str(x): float(u[j, int(x / W.H)])
                         for x in (0.5, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0)},
        "monolith_divergence": float(mono.last_div),
        "note": "monolith n=255, frozen actuator strip at x=1 y=1, pinned Dirichlet ring",
    })
    return u, v, fx


# ---------------------------------------------------------------------------
# W2 -- assemble one real Lambda-tilde
# ---------------------------------------------------------------------------


def stage_w2(out, u, v, dt=W.MACRO_DT, nu=W.NU, tiling=None):
    tiling = tiling or W.DEFAULT_TILING
    state = "developed wake, t = 5"
    result = {"probe_state": state, "dim_M": W.M_EFF, "dt": dt,
              "tiling": {"n": tiling.n, "halo": tiling.halo, "ramp": tiling.ramp},
              "required_halo": W.STENCIL_RADIUS * W.SUBSTEPS}

    sweep = []
    for s in (1e-2, 1e-4, 1e-6, 1e-8):
        g, _ = W.build(u, v, mode="split-step", tiling=tiling, dt=dt, nu=nu)
        op = probe_all_seams(g, ProbeBudget(fd_step=s), "fd sweep")["sx0"][0]
        sweep.append({"fd_step": s, "beta": op.beta, "kappa": op.kappa,
                      "null_dim": op.null_dim, "mu": op.passivity_lambda_min})
        out.say(f"  w2 fd sweep {s:.0e}: beta={op.beta:.6e} kappa={op.kappa:.4f}")
    result["fd_step_sweep"] = sweep
    fd = 1e-6
    result["fd_step_used"] = fd
    out.json("w2", result)

    # the assembly, both flux conventions (W47) and both agents (R10)
    for mode in ("split-step", "as-built"):
        for flux in ("diffusive", "momentum"):
            g, _ = W.build(u, v, mode=mode, tiling=tiling, dt=dt, nu=nu, flux_mode=flux)
            ops = probe_all_seams(g, ProbeBudget(fd_step=fd), state)
            key = f"{mode}/{flux}"
            result[f"seams::{key}"] = {
                s: {**op.as_dict(),
                    "singular_values": np.linalg.svd(op.S, compute_uv=False),
                    "asymmetry": float(np.linalg.norm(op.S - op.S.T)
                                       / np.linalg.norm(op.S)),
                    "alpha_star_diagonal": np.diag(op.S),
                    "S": op.S}
                for s, (op, _tr) in ops.items()
            }
            out.say(f"  w2 {key}: sx0 beta={ops['sx0'][0].beta:.4e} "
                    f"kappa={ops['sx0'][0].kappa:.3f} null={ops['sx0'][0].null_dim} "
                    f"mu={ops['sx0'][0].passivity_lambda_min:+.4e}")
        out.json("w2", result)

    # W47: do the two flux conventions give the same assembled operator?
    a = np.array(result["seams::split-step/diffusive"]["sx0"]["S"])
    b = np.array(result["seams::split-step/momentum"]["sx0"]["S"])
    result["flux_convention_W47"] = {
        "rel_diff_assembled": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "block_pi_diffusive": {k: v["passivity_defect"] for k, v in
                               result["seams::split-step/diffusive"]["sx0"]["blocks"].items()},
        "block_pi_momentum": {k: v["passivity_defect"] for k, v in
                              result["seams::split-step/momentum"]["sx0"]["blocks"].items()},
        "verdict": "2.2 is normative; 4.1 is a seam-level equivalent and a block-level hazard",
    }

    # Xi and ||Lambda - Lambda_ref||: the exposed agent is the reference, because
    # R10 says the embedded one is solving a different operator.
    ref_graph, _ = W.build(u, v, mode="split-step", tiling=tiling, dt=dt, nu=nu)
    ref_caps = {a.agent_id: a.capabilities for a in ref_graph.agents}
    emb_graph, _ = W.build(u, v, mode="as-built", tiling=tiling, dt=dt, nu=nu)
    ops_emb = probe_all_seams(emb_graph, ProbeBudget(fd_step=fd), state,
                              references=ref_caps)
    ops_ref = probe_all_seams(ref_graph, ProbeBudget(fd_step=fd), state)
    result["Xi_and_operator_defect"] = {
        s: {
            "Xi_blocks": {k: b.Xi for k, b in ops_emb[s][0].blocks.items()},
            "norm_Lambda_embedded": float(np.linalg.norm(ops_emb[s][0].S, 2)),
            "norm_Lambda_exposed": float(np.linalg.norm(ops_ref[s][0].S, 2)),
            "operator_defect": float(np.linalg.norm(ops_emb[s][0].S - ops_ref[s][0].S, 2)),
            "relative": float(np.linalg.norm(ops_emb[s][0].S - ops_ref[s][0].S, 2)
                              / max(np.linalg.norm(ops_ref[s][0].S, 2), 1e-300)),
        } for s in ops_emb
    }
    out.say("  w2 Xi/operator-defect: "
            + ", ".join(f"{s}={v['relative']:.3e}"
                        for s, v in result["Xi_and_operator_defect"].items()))

    # controls
    z = np.zeros((W.MONO_N, W.MONO_N))
    controls = {}
    for label, (bu, bv) in (("zero-state (Stokes limit)", (z, z)),
                            ("uniform flow", (np.full_like(z, W.U_INF), z))):
        g, _ = W.build(bu, bv, mode="split-step", tiling=tiling, dt=dt, nu=nu)
        op = probe_all_seams(g, ProbeBudget(fd_step=fd), label)["sx0"][0]
        controls[label] = {
            "beta": op.beta, "kappa": op.kappa, "null_dim": op.null_dim,
            "mu": op.passivity_lambda_min, "pi": op.passivity_defect,
            "blocks": {k: {"beta": b.beta, "kappa": b.kappa,
                           "mu": b.passivity_lambda_min, "pi": b.passivity_defect}
                       for k, b in op.blocks.items()},
        }
        out.say(f"  w2 control {label}: mu={op.passivity_lambda_min:+.4e} "
                f"beta={op.beta:.4e} null={op.null_dim}")
    result["controls"] = controls

    # W30 drift
    solvers = make_solvers(tiling, nu)
    fx = body_force()
    ring = (u.copy(), v.copy())
    g0, _ = W.build(u, v, mode="split-step", tiling=tiling, dt=dt, nu=nu)
    ops0 = probe_all_seams(g0, ProbeBudget(fd_step=fd), "t")
    uk, vk = u.copy(), v.copy()
    for _ in range(10):
        uk, vk = composed_step("split-step", tiling, solvers, uk, vk, dt, fx, ring)
    gk, _ = W.build(uk, vk, mode="split-step", tiling=tiling, dt=dt, nu=nu)
    opsk = probe_all_seams(gk, ProbeBudget(fd_step=fd), "t + K dt")
    result["operator_drift_W30"] = {
        "K": 10, "dt": dt,
        "per_seam": {s: {"drift": operator_drift(ops0[s][0].S, opsk[s][0].S),
                         "relative": operator_drift(ops0[s][0].S, opsk[s][0].S)
                                     / float(np.linalg.norm(ops0[s][0].S, 2))}
                     for s in ops0},
    }
    out.json("w2", result)
    return result


# ---------------------------------------------------------------------------
# W1 -- fit L
# ---------------------------------------------------------------------------


def _fit_L(errors, dt):
    e = np.asarray(errors, float)
    good = np.isfinite(e) & (e > 0)
    n = np.arange(e.size)[good]
    if n.size < 3:
        return None
    y = np.log(e[good])
    A = np.column_stack([n.astype(float), np.ones(n.size)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    s2 = float(resid @ resid) / max(1, n.size - 2)
    se = float(np.sqrt((s2 * np.linalg.inv(A.T @ A))[0, 0]))
    slope = float(coef[0])
    L = float(np.exp(slope))
    return {"L": L, "L_stderr": float(L * se),
            "L_ci95": [float(np.exp(slope - 1.96 * se)), float(np.exp(slope + 1.96 * se))],
            "eta": L - 1.0, "growth_rate_per_time": float(slope / dt),
            "n_points": int(n.size), "log_residual_rms": float(np.sqrt(s2))}


def _roll(scheme, tiling, solvers, u, v, fx, ring, dt, n_steps, mono=None):
    us, vs = [u.copy()], [v.copy()]
    U, V = u.copy(), v.copy()
    for _ in range(n_steps):
        if scheme == "monolith":
            U, V = mono_step(mono, U, V, dt, fx)
        else:
            U, V = composed_step(scheme, tiling, solvers, U, V, dt, fx, ring)
        us.append(U.copy()); vs.append(V.copy())
    return us, vs


def stage_w1(out, u, v, fx, dt=W.MACRO_DT, nu=W.NU, n_steps=40,
             amplitudes=(1e-3, 1e-4), tiling=None):
    tiling = tiling or W.DEFAULT_TILING
    solvers = make_solvers(tiling, nu)
    mono = solvers[0]
    ring = (u.copy(), v.copy())
    result = {"dt": dt, "n_steps": n_steps,
              "tiling": {"n": tiling.n, "halo": tiling.halo, "ramp": tiling.ramp},
              "perturbation": "divergence-free streamfunction blob at x=0.5, y=1.0, r=0.15",
              "note": "L is a property of a SCHEME; e^n compares perturbed and "
                      "unperturbed trajectories of the SAME scheme"}
    for scheme in ("monolith",) + SCHEMES:
        base = _roll(scheme, tiling, solvers, u, v, fx, ring, dt, n_steps, mono)
        per_amp = {}
        for amp in amplitudes:
            du, dv = streamfunction_blob(amp)
            traj = _roll(scheme, tiling, solvers, u + du, v + dv, fx, ring, dt,
                         n_steps, mono)
            errs = [rel_l2(traj[0][k], base[0][k], traj[1][k], base[1][k])
                    for k in range(n_steps + 1)]
            fit = _fit_L(errs, dt)
            per_amp["%.0e" % amp] = {"errors": errs, "fit": fit}
            out.say("  w1 %-11s amp=%.0e: e0=%.3e e_end=%.3e L=%.6f +/- %.2g"
                    % (scheme, amp, errs[0], errs[-1], fit["L"], fit["L_stderr"]))
        result[scheme] = per_amp
        out.json("w1", result)

    op2 = {"deficit_t2": 0.093, "deficit_t20": 0.259, "dt_macro": 0.25,
           "n_macro_steps": int(round((20.0 - 2.0) / 0.25))}
    op2["L_implied"] = float((op2["deficit_t20"] / op2["deficit_t2"])
                             ** (1.0 / op2["n_macro_steps"]))
    op2["eta_implied"] = op2["L_implied"] - 1.0
    op2["note"] = ("OP-2's drift on the PERIODIC frozen-checkpoint configuration at "
                   "dt=0.25. Three variables differ from this one; the comparison is "
                   "stated as such rather than as a match")
    result["op2_recorded_drift"] = op2
    out.json("w1", result)
    return result


# ---------------------------------------------------------------------------
# W3 -- the three-way split
# ---------------------------------------------------------------------------


def stage_w3_cut_criterion(tiling, pou, u, v, lu, lv, ru, rv):
    """L2/C2, on the four-window tiling: the identity, both bound forms, and Qhat.

    ``D_i = E_i R_i u - R_i E u`` is the restriction defect -- the failure of the
    exact one-interval operator to commute with restriction to window ``i``, and
    the definition G5/W16's *"the exact operator is closest to local"* never had.
    The identity ``A({E_i R_i u}) - E u = sum_i R_i^T chi_i D_i`` is exact given
    the partition of unity, and with ``chi >= 0`` (L6/C1) it bounds the composed
    defect cellwise by ``sum_i chi_i |D_i|``.

    Reported per component and then jointly, because ``v`` is an order smaller
    than ``u`` in a through-flow and normalizing each separately would report the
    transverse component's smallness as a defect.
    """
    ws = tiling.weights()
    n, M = tiling.n, W.MONO_N
    acc = {"w": 0.0, "m": 0.0, "act": 0.0, "hat": 0.0}
    resid = 0.0
    per = {}
    for tag, loc, ref, base in (("u", lu, ru, u), ("v", lv, rv, v)):
        lhs = np.zeros((M, M))
        rhs = np.zeros((M, M))
        wb = np.zeros((M, M))
        mb = np.zeros((M, M))
        glob, own = [], []
        for k, (ox, oy) in enumerate(tiling.offsets):
            sl = (slice(oy, oy + n), slice(ox, ox + n))
            chi = ws[k][sl]
            D = loc[k] - ref[sl]
            lhs[sl] += chi * loc[k]
            rhs[sl] += chi * D
            wb[sl] += chi * np.abs(D)
            mb[sl] = np.maximum(mb[sl], np.abs(D) * (chi > 0.0))
            g = np.zeros((M, M)); o = np.zeros((M, M), dtype=bool)
            g[sl] = loc[k]; o[sl] = chi > 0.0
            glob.append(g); own.append(o)
        lhs = lhs - ref
        hat = np.zeros((M, M))
        for i in range(4):
            for j in range(i + 1, 4):
                both = own[i] & own[j]
                hat = np.maximum(hat, np.abs(glob[i] - glob[j]) * both)
        scale = max(float(np.max(np.abs(lhs))), 1e-300)
        resid = max(resid, float(np.max(np.abs(lhs - rhs))) / scale)
        per[tag] = {
            "identity_residual_rel": float(np.max(np.abs(lhs - rhs))) / scale,
            "cellwise_bound_violation": float(np.max(np.abs(lhs) - mb)),
        }
        acc["w"] += float(np.sum(wb**2))
        acc["m"] += float(np.sum(mb**2))
        acc["act"] += float(np.sum(lhs**2))
        acc["hat"] += float(np.sum(hat**2))
    joint = float(np.sqrt(np.sum(u**2) + np.sum(v**2)))
    wbnd = float(np.sqrt(acc["w"])) / joint
    mbnd = float(np.sqrt(acc["m"])) / joint
    act = float(np.sqrt(acc["act"])) / joint
    hat = float(np.sqrt(acc["hat"])) / joint
    cont = tiling.contaminated(W.STENCIL_RADIUS)
    pair = int(sum((cont[i] & cont[j]).sum()
                   for i in range(4) for j in range(i + 1, 4)))
    return {
        "cut_defect_bound": wbnd,
        "cut_defect_bound_max": mbnd,
        "cut_defect_bound_reference_free": hat,
        "actual_one_interval_defect": act,
        "bound_tightness_chi_weighted": act / max(wbnd, 1e-300),
        "bound_tightness_max_form": act / max(mbnd, 1e-300),
        "surrogate_over_max_form": hat / max(mbnd, 1e-300),
        "identity_residual_rel": resid,
        "per_component": per,
        "contaminated_sets_pairwise_disjoint": pair == 0,
        "contaminated_pairwise_overlap_cells": pair,
        "note": "one exchange interval (MACRO_DT / SUBSTEPS), exposed agent per "
                "window against a no-projection monolith of the same class. The "
                "reference-free surrogate equals the MAX form exactly when the "
                "contaminated sets are pairwise disjoint; here they are not, and "
                "it agrees anyway -- measured, not assumed",
    }


def stage_w3(out, u, v, fx, dt=W.MACRO_DT, nu=W.NU, tiling=None):
    tiling = tiling or W.DEFAULT_TILING
    solvers = make_solvers(tiling, nu)
    mono = solvers[0]
    ring = (u.copy(), v.copy())
    u_ref, v_ref = mono_step(mono, u, v, dt, fx)

    result = {"depth": 0, "dt": dt,
              "reference": "reference.WindowNS n=255 monolith, identical discretization",
              "tiling": {"n": tiling.n, "halo": tiling.halo, "ramp": tiling.ramp,
                         "required_halo": W.STENCIL_RADIUS * W.SUBSTEPS},
              "depth_tag_W34": "all defects below are measured at depth 0; a nested "
                               "run's numbers are not comparable without this field"}

    for scheme in SCHEMES:
        u_c, v_c = composed_step(scheme, tiling, solvers, u, v, dt, fx, ring)
        u_t, v_t = composed_step(scheme, tiling, solvers, u, v, dt, fx, ring,
                                 exact_ring=(u_ref, v_ref))
        total = rel_l2(u_c, u_ref, v_c, v_ref)
        tau = rel_l2(u_t, u_ref, v_t, v_ref)
        sigma = rel_l2(u_c, u_t, v_c, v_t)
        result[scheme] = {
            "defect_total_relative_L2": total,
            "tau_agent": tau,
            "sigma_transmission": sigma,
            "gamma_solve": 0.0,
            "gamma_note": "the rung is dirichlet with a lagged trace, so there is no "
                          "interface system to solve and gamma is identically zero. "
                          "Under the refused probed-DtN rung the direct solve drove the "
                          "residual to 8.4e-20, which was gamma's previous measurement",
        }
        out.say(f"  w3 {scheme:11s}: total={total:.4e} tau={tau:.4e} sigma={sigma:.4e}")

    imp = result["as-built"]["defect_total_relative_L2"] / max(
        result["split-step"]["defect_total_relative_L2"], 1e-300)
    result["improvement_factor"] = imp
    out.say(f"  w3 split-step is {imp:.1f}x better than as-built")

    # W28 / L6/C1: the assembly certificate, on REAL local solves.
    #
    # **Corrected 2026-08-28.** This block used to pass `tiling.cut(u_ref)` as the
    # local solves and `u_ref` as the reference -- i.e. it cut the reference into
    # pieces and blended them back together, which the partition-of-unity identity
    # makes exactly zero for ANY partition. Measured: the emitted blend defect was
    # 0 to 6e-15 for all six partitions tried, including a hard switch and a signed
    # one. It was the `norm_A` failure again, one field over: a number that could
    # only ever be nonzero when the thing it was checking had already failed.
    #
    # The honest inputs are one sub-step of the exposed agent per window against
    # the same sub-step taken by a no-projection monolith of the same class: same
    # map, same state, no projection and no assembly in between, so nothing but
    # L6 is in the comparison.
    pou = tiling.partition_of_unity()
    cls = W._no_projection_class()
    mono_np = cls(nu=nu, length=W.MONO_L, n=W.MONO_N, cfl=0.4, transmission="dirichlet")
    win_np = cls(nu=nu, length=tiling.n * W.H, n=tiling.n, cfl=0.4,
                 transmission="dirichlet")
    hs = dt / W.SUBSTEPS
    us, vs = tiling.cut(u), tiling.cut(v)
    fxs, zero = tiling.cut(fx), np.zeros((4, tiling.n, tiling.n))
    lu, _lv = win_np.step_batch(us, vs, hs, bc0=(us, vs), bc1=None, force=(fxs, zero))
    ru, _rv = mono_np.step_batch(u[None], v[None], hs, bc0=None, bc1=None,
                                 force=(fx[None], np.zeros_like(fx)[None]))
    locals_ = {n: lu[k].reshape(-1) for k, n in enumerate(tiling.names)}
    ov = np.zeros((W.MONO_N, W.MONO_N))
    for ox, oy in tiling.offsets:
        ov[oy:oy + tiling.n, ox:ox + tiling.n] += 1.0
    mask = (ov >= 2.0).reshape(-1)
    cert = certify(pou, locals_=locals_, reference=ru[0].reshape(-1), overlap_mask=mask)

    # L2/C2, measured on the SAME honest inputs -- one exchange interval of the
    # exposed agent per window against the same interval of a no-projection
    # monolith.  The identity `A({E_i R_i u}) - E u = sum_i R_i^T chi_i D_i` is
    # checked here rather than asserted, and the two bound forms are reported
    # separately because they differ by 220x and the difference is what the
    # ramped partition of unity buys.  See `scripts/w16_cut_policy.py` for the
    # derivation and for the falsification of the cut score it replaces.
    cutb = stage_w3_cut_criterion(tiling, pou, u, v, lu, _lv, ru[0], _rv[0])
    result["cut_criterion_L2_C2"] = cutb
    out.say(f"  w3 L2/C2: chi-weighted bound={cutb['cut_defect_bound']:.4e} "
            f"(tightness {cutb['bound_tightness_chi_weighted']:.4f}), max form="
            f"{cutb['cut_defect_bound_max']:.4e}, identity residual="
            f"{cutb['identity_residual_rel']:.2e}, surrogate ratio="
            f"{cutb['surrogate_over_max_form']:.5f}")

    vacuous = certify(pou,
                      locals_={n: tiling.cut(u_ref)[k].reshape(-1)
                               for k, n in enumerate(tiling.names)},
                      reference=u_ref.reshape(-1))
    result["assembly_certificate_W28"] = {
        **cert.as_dict(),
        "identity_norm_vacuous": pou.identity_norm(),
        "blend_defect_vacuous_inputs": vacuous.blend_defect,
        "vacuous_note": "the value this field carried until 2026-08-28: the reference "
                        "blended against itself, which the identity makes zero for any "
                        "partition of unity including a non-convex one",
        "overlap_cells": int(mask.sum()),
        "norm_A_note": "||A|| = max_j sqrt(sum_i chi_ij^2) = 1 for any CONVEX partition "
                       "of unity, attained at every single-owner cell. It is a theorem, "
                       "not a measurement. The word convex is load-bearing: a signed "
                       "partition on this same tiling measures ||A|| = 1.7038",
    }
    out.say(f"  w3 assembly: pou residual={cert.pou_residual:.2e} "
            f"norm_A={cert.norm_A} chi_min={cert.condition.chi_min:.4e} "
            f"L6/C1={cert.condition.holds}")
    out.say(f"  w3 assembly: blend_defect={cert.blend_defect:.3e} (was "
            f"{vacuous.blend_defect:.1e} on vacuous inputs) "
            f"V_chi_min={cert.condition.variance_margin:.3e} "
            f"hull_escape={cert.condition.hull_escape:.3e}")
    out.json("w3", result)
    return result


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="s0,w2,w1,w3")
    ap.add_argument("--out", default=os.path.join("out", "tier0b"))
    ap.add_argument("--t-dev", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=W.MACRO_DT)
    ap.add_argument("--w1-steps", type=int, default=40)
    ap.add_argument("--state", default=None, help="reuse a base state npz")
    a = ap.parse_args(argv)

    out = Out(a.out)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    out.say(f"stages: {stages}   expert: {W.build_repo()}")
    out.say(f"tiling: n={W.DEFAULT_TILING.n} halo={W.DEFAULT_TILING.halo} "
            f"ramp={W.DEFAULT_TILING.ramp} required_halo="
            f"{W.STENCIL_RADIUS * W.SUBSTEPS}")

    state = a.state or os.path.join(a.out, "s0_state.npz")
    if "s0" in stages:
        u, v, fx = stage_s0(out, t_dev=a.t_dev, dt=a.dt)
    else:
        d = np.load(state)
        u, v, fx = d["u"], d["v"], d["fx"]
        out.say(f"loaded base state from {state}")

    if "w2" in stages:
        stage_w2(out, u, v, dt=a.dt)
    if "w1" in stages:
        stage_w1(out, u, v, fx, dt=a.dt, n_steps=a.w1_steps)
    if "w3" in stages:
        stage_w3(out, u, v, fx, dt=a.dt)
    out.say("done")


if __name__ == "__main__":
    main()
