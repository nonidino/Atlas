"""W55 -- the rules and the factorizations, against a second and a third expert.

Every constant in the vault -- `C_mu = 1.2`, `Pi`'s two limits, R10b's cadence
rule, `L`, `tau`, `sigma`, `beta`, `kappa`, `mu`, `cut_defect_bound` -- was
measured against `reference.WindowNS` and its no-projection subclass, plus one
`SpectralNS` control that measures zero by construction. `tier0-measurements`
§9.8 and §10.5 both name that the largest remaining caveat.

**The question is not whether the numbers match.** They will not: a different
discretization has a different truncation error and no reason to reproduce
anybody else's `tau`. The question is whether the **rules** hold structurally --
R10, R10b, R11, R2b -- and whether the **factorizations** do: the bias-variance
identity behind L6/C1, `Pi`'s composition behind W49, and the restriction-defect
identity behind L2/C2. A rule that holds only for one solver's internals is not
a rule, and a constant that moves by orders between two solvers of the same
family is not a constant.

    A   `reference.ChannelNS`  a second DISCRETIZATION: advective-form advection
                               instead of skew-symmetric, a rolled Laplacian, and
                               `pressure.project_outflow` -- a projection with an
                               outlet Dirichlet, hence **non-singular**, where
                               `WindowNS`'s all-Neumann one is singular in the
                               constant mode. That difference is the sharpest
                               single test here: R10's measured MECHANISM (§4.1's
                               compatibility violation) is absent, and R10's
                               stated REASON is untouched.

    B   Poseidon-T             a LEARNED operator: 20.8M parameters, frozen. Not
                               a discretization at all. It is fixed at 128x128,
                               so it cannot supply a same-class monolith and half
                               the Tier 0 stack is structurally unavailable to
                               it -- which is itself the finding.

Run:  python scripts/w55_second_expert.py [--parts A,B] [--out out/w55]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# torch and numpy's MKL each ship an OpenMP runtime, and loading both in one
# process aborts with "libiomp5md.dll already initialized" the first time numpy
# takes an SVD after the checkpoint is in memory. This is the documented
# workaround; it is set BEFORE numpy or torch is imported because neither reads
# it afterwards. Part B verifies afterwards that linear algebra still agrees with
# a torch-free reference, because the warning this suppresses says results can be
# silently wrong and an unchecked suppression would be exactly that.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from atlas import ProbeBudget, compile_scheme                      # noqa: E402
from atlas.assembly import certify, sigma_halo_bound               # noqa: E402
from atlas.cases import channel_ns as C                            # noqa: E402
from atlas.cases import window_ns as W                             # noqa: E402
from atlas.probe import assemble_seam                              # noqa: E402

SCHEMES = ("as-built", "split-step")


def _j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rel_l2(a, b, a2, b2) -> float:
    return float(np.sqrt((np.sum((a - b) ** 2) + np.sum((a2 - b2) ** 2))
                         / (np.sum(b ** 2) + np.sum(b2 ** 2))))


def load_state():
    d = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    return d["u"], d["v"]


# ===========================================================================
# A -- ChannelNS
# ===========================================================================


def _pin_domain(U, V, ring):
    for A, R in ((U, ring[0]), (V, ring[1])):
        A[0, :] = R[0, :]; A[-1, :] = R[-1, :]
        A[:, 0] = R[:, 0]; A[:, -1] = R[:, -1]
    return U, V


def channel_mono_step(u, v, dt, expose=False, substeps=None):
    """One macro-step of the ChannelNS monolith, pinned at the domain edge."""
    mono = C.ChannelWindow(nu=W.NU, n=W.MONO_N, h=W.H, expose_elliptic=expose)
    substeps = substeps or C.substeps_at(dt)
    U, V = u.copy(), v.copy()
    hs = dt / substeps
    for _ in range(substeps):
        U, V = mono.step(U, V, hs, ring=(u, v))
    return U, V


def channel_composed_step(mode, tiling, u, v, dt, ring, substeps=None,
                          exact_ring=None, exchange_substeps=None):
    """One composed macro-step, mirroring `tier0_window_ns.composed_step`.

    ``exchange_substeps`` deliberately overridable: R10b's whole content is that
    the exchange cadence must match the agent's own sub-step cadence, and the
    only way to test the rule is to break it on purpose.
    """
    n = tiling.n
    substeps = substeps or C.substeps_at(dt)
    exch = exchange_substeps or substeps
    if mode == "as-built":
        win = C.ChannelWindow(nu=W.NU, n=n, h=W.H, expose_elliptic=False)
        us, vs = tiling.cut(u), tiling.cut(v)
        src = (u, v) if exact_ring is None else exact_ring
        rs = (tiling.cut(src[0]), tiling.cut(src[1]))
        outs_u, outs_v = [], []
        for k in range(4):
            a, b = win.step_macro(us[k], vs[k], dt, ring=(rs[0][k], rs[1][k]),
                                  substeps=substeps)
            outs_u.append(a); outs_v.append(b)
        return _flat_assemble(tiling, np.stack(outs_u), np.stack(outs_v))

    win = C.ChannelWindow(nu=W.NU, n=n, h=W.H, expose_elliptic=True)
    proj = C.load_pressure().project_outflow
    U, V = u.copy(), v.copy()
    hs = dt / exch
    for m in range(exch):
        us, vs = tiling.cut(U), tiling.cut(V)
        if exact_ring is not None:
            s = (m + 1) / exch
            ru = (1 - s) * u + s * exact_ring[0]
            rv = (1 - s) * v + s * exact_ring[1]
            rs = (tiling.cut(ru), tiling.cut(rv))
        else:
            rs = (us, vs)
        ou, ov = [], []
        for k in range(4):
            a, b = win.step(us[k], vs[k], hs, ring=(rs[0][k], rs[1][k]))
            ou.append(a); ov.append(b)
        U, V = tiling.assemble(np.stack(ou), np.stack(ov))
        U, V, _r = proj(U, V, W.H)
        U, V = _pin_domain(U, V, ring)
    return U, V


def _flat_assemble(tiling, us, vs):
    M, n = W.MONO_N, tiling.n
    au, av, c = np.zeros((M, M)), np.zeros((M, M)), np.zeros((M, M))
    for k, (ox, oy) in enumerate(tiling.offsets):
        au[oy:oy + n, ox:ox + n] += us[k]
        av[oy:oy + n, ox:ox + n] += vs[k]
        c[oy:oy + n, ox:ox + n] += 1.0
    return au / c, av / c


def probe_all_seams(graph, budget, probe_state):
    from atlas.compiler import _Context, _derive_transfer          # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.verdict import DecisionRecord

    ctx = _Context(graph=graph, budget=None, probe_budget=budget, references={},
                   record=DecisionRecord(), stamp=EnvelopeStamp(), holes=HoleLedger(),
                   probe_state=probe_state, depth=0)
    out = {}
    for conn in graph.connections:
        tr = _derive_transfer(ctx, conn)
        out[conn.seam_id] = assemble_seam(graph, conn, tr, budget, {},
                                          expected_null_dim=conn.expected_null_dim,
                                          probe_state=probe_state)
    return out


def part_a(u, v, dt=W.MACRO_DT):
    tiling = W.DEFAULT_TILING
    res = {"expert": "reference.ChannelNS", "dt": dt, "tiling": {
        "n": tiling.n, "halo": tiling.halo, "ramp": tiling.ramp}}

    # -- A1: W2, the probed operator, both modes ---------------------------
    budget = ProbeBudget(fd_step=1e-6)
    for mode in SCHEMES:
        g, ex = C.build(u, v, mode=mode, dt=dt)
        ops = probe_all_seams(g, budget, f"ChannelNS {mode}, developed wake t=5")
        rows = {}
        for sid, op in ops.items():
            asym = float(np.linalg.norm(op.S - op.S.T) / max(np.linalg.norm(op.S), 1e-300))
            rows[sid] = {"beta": op.beta, "kappa": op.kappa, "null_dim": op.null_dim,
                         "mu": op.passivity_lambda_min, "pi": op.passivity_defect,
                         "cut_score_RETIRED": op.cut_score, "asymmetry": asym}
            say(f"  A1 {mode:11s} {sid}: beta={op.beta:.4e} kappa={op.kappa:.4g} "
                f"null={op.null_dim} mu={op.passivity_lambda_min:+.4e} "
                f"asym={asym:.4f}")
        res[f"w2_{mode}"] = rows

    # -- A2/A3: W3, the three-way split, both modes ------------------------
    ring = (u.copy(), v.copy())
    mono_u, mono_v = channel_mono_step(u, v, dt)
    split = {}
    for mode in SCHEMES:
        cu, cv = channel_composed_step(mode, tiling, u, v, dt, ring)
        tu, tv = channel_composed_step(mode, tiling, u, v, dt, ring,
                                       exact_ring=(mono_u, mono_v))
        total = rel_l2(cu, mono_u, cv, mono_v)
        tau = rel_l2(tu, mono_u, tv, mono_v)
        sigma = rel_l2(cu, tu, cv, tv)
        split[mode] = {"total": total, "tau": tau, "sigma": sigma, "gamma": 0.0}
        say(f"  A3 {mode:11s}: total={total:.4e} tau={tau:.4e} sigma={sigma:.4e}")
    split["improvement_factor"] = (split["as-built"]["total"]
                                   / max(split["split-step"]["total"], 1e-300))
    say(f"  A3 split-step is {split['improvement_factor']:.1f}x better than as-built")
    res["w3"] = split

    # -- A2b: R10's signature -- is the embedded defect FLAT in dt? --------
    #
    # This is the test that matters most for W55. §4.1 identified the elliptic
    # defect by three signatures: flat in dt, flat in distance from the cut, and
    # untouched by the halo. `project_outflow` is NON-SINGULAR, so §4.1's
    # specific mechanism -- a Neumann compatibility violation -- cannot occur.
    # If the defect is still flat in dt, R10 is about the elliptic operator
    # being global and not about that solver's compatibility condition.
    flat = {}
    for d in (dt, dt / 5.0, dt / 25.0):
        mu_, mv_ = channel_mono_step(u, v, d)
        cu, cv = channel_composed_step("as-built", tiling, u, v, d, ring,
                                       exact_ring=(mu_, mv_))
        flat[f"{d:.4g}"] = rel_l2(cu, mu_, cv, mv_)
        say(f"  A2b R10 signature: dt={d:.4g} tau_as_built={flat[f'{d:.4g}']:.4e}")
    vals = list(flat.values())
    flat["ratio_over_25x_dt_range"] = float(max(vals) / max(min(vals), 1e-300))
    flat["verdict"] = ("FLAT in dt -- elliptic, R10 holds"
                       if flat["ratio_over_25x_dt_range"] < 5.0
                       else "scales with dt -- NOT elliptic, R10's premise fails here")
    say(f"  A2b -> {flat['verdict']} (ratio {flat['ratio_over_25x_dt_range']:.2f})")
    res["r10_signature"] = flat

    # -- A4: W1, fit L -----------------------------------------------------
    res["w1"] = _fit_L(u, v, dt, tiling, ring)

    # -- A5/A8: L6/C1 and L2/C2 on real local solves -----------------------
    res["assembly_and_cut"] = _assembly_and_cut(u, v, dt, tiling)

    # -- A6: W49's Pi and the implied C_mu ---------------------------------
    res["w49"] = _w49(u, v, dt, tiling, ring, mono_u, mono_v,
                      split["split-step"]["sigma"])
    res["w49_sweep"] = _w49_sweep(u, v, dt, ring)

    # -- A7: R10b, the cadence rule ----------------------------------------
    cad = {}
    for exch in (C.substeps_at(dt), 2 * C.substeps_at(dt), C.substeps_at(dt) // 2):
        cu, cv = channel_composed_step("split-step", tiling, u, v, dt, ring,
                                       exact_ring=(mono_u, mono_v),
                                       exchange_substeps=exch)
        cad[str(exch)] = rel_l2(cu, mono_u, cv, mono_v)
        say(f"  A7 R10b: exchanges={exch:3d} (agent's own {C.substeps_at(dt)}) "
            f"tau={cad[str(exch)]:.4e}")
    matched = cad[str(C.substeps_at(dt))]
    cad["penalty_for_mismatch"] = float(
        max(v_ for k, v_ in cad.items() if k != str(C.substeps_at(dt))
            and not k.startswith("penalty")) / max(matched, 1e-300))
    say(f"  A7 -> mismatching the cadence costs {cad['penalty_for_mismatch']:.1f}x in tau")
    res["r10b"] = cad

    # -- A9: the compile ---------------------------------------------------
    res["compile"] = _compile_both(u, v, dt)
    return res


def _fit_L(u, v, dt, tiling, ring, n_steps=40, amp=1e-3):
    """Paired rollouts, least squares on log||e^n||, per map."""
    def blob(a):
        n, h = W.MONO_N, W.H
        g = (np.arange(n) + 0.5) * h
        X, Y = np.meshgrid(g, g)
        psi = a * np.exp(-(((X - 0.5) ** 2 + (Y - 1.0) ** 2) / 0.15 ** 2))
        return np.gradient(psi, h, axis=0), -np.gradient(psi, h, axis=1)

    du, dv = blob(amp)
    out = {}
    for name in ("monolith", "as-built", "split-step"):
        def step(a, b):
            if name == "monolith":
                return channel_mono_step(a, b, dt)
            return channel_composed_step(name, tiling, a, b, dt, (a.copy(), b.copy()))
        A, B = u.copy(), v.copy()
        P, Q = u + du, v + dv
        errs = []
        for _ in range(n_steps):
            A, B = step(A, B)
            P, Q = step(P, Q)
            errs.append(rel_l2(P, A, Q, B))
        e = np.log(np.asarray(errs))
        k = np.arange(1, n_steps + 1)
        sl, _ic = np.polyfit(k, e, 1)
        resid = e - (sl * k + _ic)
        se = float(np.sqrt(np.sum(resid**2) / (n_steps - 2)
                           / np.sum((k - k.mean())**2)))
        out[name] = {"L": float(np.exp(sl)), "L_stderr": float(np.exp(sl) * se),
                     "e_end": errs[-1]}
        say(f"  A4 W1 {name:11s}: L={out[name]['L']:.6f} +/- {out[name]['L_stderr']:.5f}")
    d = abs(out["split-step"]["L"] - out["monolith"]["L"])
    out["split_step_minus_monolith_in_stderr"] = float(
        d / max(out["monolith"]["L_stderr"], 1e-300))
    say(f"  A4 -> composed L differs from the monolith by "
        f"{out['split_step_minus_monolith_in_stderr']:.2f} standard errors")
    return out


def _assembly_and_cut(u, v, dt, tiling):
    """L6/C1's certificate and L2/C2's bound, on one honest exchange interval."""
    pou = tiling.partition_of_unity()
    win = C.ChannelWindow(nu=W.NU, n=tiling.n, h=W.H, expose_elliptic=True)
    mono = C.ChannelWindow(nu=W.NU, n=W.MONO_N, h=W.H, expose_elliptic=True)
    hs = dt / C.substeps_at(dt)
    us, vs = tiling.cut(u), tiling.cut(v)
    lu, lv = [], []
    for k in range(4):
        a, b = win.step(us[k], vs[k], hs, ring=(us[k], vs[k]))
        lu.append(a); lv.append(b)
    lu, lv = np.stack(lu), np.stack(lv)
    ru, rv = mono.step(u, v, hs, ring=(u, v))

    ov = np.zeros((W.MONO_N, W.MONO_N))
    for ox, oy in tiling.offsets:
        ov[oy:oy + tiling.n, ox:ox + tiling.n] += 1.0
    mask = (ov >= 2.0).reshape(-1)
    cert = certify(pou,
                   locals_={n: lu[k].reshape(-1) for k, n in enumerate(tiling.names)},
                   reference=ru.reshape(-1), overlap_mask=mask)

    from tier0_window_ns import stage_w3_cut_criterion               # noqa: PLC0415
    cut = stage_w3_cut_criterion(tiling, pou, u, v, lu, lv, ru, rv)
    say(f"  A5 L6/C1: pou residual={cert.pou_residual:.2e} norm_A={cert.norm_A} "
        f"chi_min={cert.condition.chi_min:.4e} holds={cert.condition.holds}")
    say(f"  A5 blend_defect={cert.blend_defect:.3e} V_chi_min="
        f"{cert.condition.variance_margin:.3e} hull_escape="
        f"{cert.condition.hull_escape:.3e}")
    say(f"  A8 L2/C2: chi-weighted bound={cut['cut_defect_bound']:.4e} "
        f"(tightness {cut['bound_tightness_chi_weighted']:.4f}), "
        f"identity residual={cut['identity_residual_rel']:.2e}, "
        f"surrogate ratio={cut['surrogate_over_max_form']:.5f}")
    return {"assembly_certificate": cert.as_dict(), "cut_criterion_L2_C2": cut}


def _face_trace(cut_u, cut_v, tiling):
    """The four artificial rings of every window, flattened.

    Copied from `w49_sigma_halo.py` rather than approximated. ``||d_lambda||``
    has to be **the same functional** on both experts or the implied C_mu is a
    comparison of two different quantities -- which is exactly the mistake a
    whole-field norm would make, and it moves the answer by two orders.
    """
    out = []
    for k, (ox, oy) in enumerate(tiling.offsets):
        faces = tiling.artificial_faces(ox, oy)
        for A in (cut_u[k], cut_v[k]):
            if "xlo" in faces:
                out.append(A[:, 0])
            if "xhi" in faces:
                out.append(A[:, -1])
            if "ylo" in faces:
                out.append(A[0, :])
            if "yhi" in faces:
                out.append(A[-1, :])
    return np.concatenate(out)


def _w49_sweep(u, v, dt, ring, configs=((21, 1), (21, 4), (21, 8), (21, 21),
                                        (11, 8), (41, 8))):
    """C_mu over several configurations, because one is a back-fit and not a value.

    §8.8 refused to quote a C_mu obtained from a single measured sigma, and §9.2
    only quoted 1.2 after 16 configurations showed it moving 5x while sigma moved
    4e4. The same discipline applies to a second expert: **one configuration
    gives a number, several give a constant.**

    ``configs`` are ``(halo, ramp)``. Varying the ramp is what moves Pi -- §9.2's
    finding that overlap width is a precondition and the partition of unity is
    the mechanism -- so the sweep is built around it.
    """
    rows = []
    for halo, ramp in configs:
        n = (W.MONO_N + halo) // 2
        if (2 * n - W.MONO_N) != halo:
            n = (W.MONO_N + halo + 1) // 2
        t = W.Tiling(n=n, ramp=ramp)
        mu_, mv_ = channel_mono_step(u, v, dt)
        r = _w49(u, v, dt, t, ring, mu_, mv_, None, quiet=True)
        r.update(halo=t.halo, ramp=ramp, n=t.n)
        rows.append(r)
        say(f"  A6 halo={t.halo:3d} ramp={ramp:3d}  Pi={r['Pi']:.4g}  "
            f"|dlam|={r['d_lambda']:.4e}  sigma={r['sigma_paired']:.4e}  "
            f"implied C_mu={r['implied_C_mu']:.4f}  bound/sigma="
            f"{r['bound_at_C_mu_1_2'] / max(r['sigma_paired'], 1e-300):.2f}")
    imp = [r["implied_C_mu"] for r in rows]
    pis = [r["Pi"] for r in rows]
    sigs = [r["sigma_paired"] for r in rows]
    out = {
        "rows": rows,
        "implied_C_mu_range": [float(min(imp)), float(max(imp))],
        "C_mu_spread_factor": float(max(imp) / max(min(imp), 1e-300)),
        "sigma_spread_factor": float(max(sigs) / max(min(sigs), 1e-300)),
        "Pi_range": [float(min(pis)), float(max(pis))],
        "bound_holds_everywhere": all(r["bound_holds"] for r in rows),
        "windowns_range": [0.228, 1.178],
        "any_inside_windowns_range": any(r["inside_windowns_range"] for r in rows),
    }
    say(f"  A6 -> implied C_mu in [{min(imp):.4f}, {max(imp):.4f}] "
        f"(spread {out['C_mu_spread_factor']:.1f}x) while sigma moves "
        f"{out['sigma_spread_factor']:.0f}x; WindowNS measured [0.228, 1.178]")
    say(f"  A6 -> C_mu = 1.2 bounds every configuration: "
        f"{out['bound_holds_everywhere']}")
    return out


def _w49(u, v, dt, tiling, ring, mono_u, mono_v, _sigma_unused, quiet=False):
    """W49's Pi and the implied C_mu, measured exactly as `w49_sigma_halo` does.

    The lagged-ring and exact-ring composed steps are run **in lockstep**, so
    ``||d_lambda||`` is the datum difference the measured sigma is actually
    caused by rather than a proxy for it, and sigma is re-measured here from the
    same paired rollout rather than borrowed from A3.
    """
    pou = tiling.partition_of_unity()
    pi = pou.contaminated_weight()
    win = C.ChannelWindow(nu=W.NU, n=tiling.n, h=W.H, expose_elliptic=True)
    proj = C.load_pressure().project_outflow
    substeps = C.substeps_at(dt)
    hs = dt / substeps

    U, V = u.copy(), v.copy()
    Ut, Vt = u.copy(), v.copy()
    dlam, lam = [], []
    for m in range(substeps):
        us, vs = tiling.cut(U), tiling.cut(V)
        uts, vts = tiling.cut(Ut), tiling.cut(Vt)
        s = (m + 1) / substeps
        ru = (1 - s) * u + s * mono_u
        rv = (1 - s) * v + s * mono_v
        b_t = (tiling.cut(ru), tiling.cut(rv))
        dlam.append(float(np.linalg.norm(_face_trace(us, vs, tiling)
                                         - _face_trace(*b_t, tiling))))
        lam.append(float(np.linalg.norm(_face_trace(*b_t, tiling))))

        ou, ov, ot, otv = [], [], [], []
        for k in range(4):
            a, b = win.step(us[k], vs[k], hs, ring=(us[k], vs[k]))
            ou.append(a); ov.append(b)
            c_, d_ = win.step(uts[k], vts[k], hs, ring=(b_t[0][k], b_t[1][k]))
            ot.append(c_); otv.append(d_)
        U, V = tiling.assemble(np.stack(ou), np.stack(ov))
        U, V, _r = proj(U, V, W.H)
        U, V = _pin_domain(U, V, ring)
        Ut, Vt = tiling.assemble(np.stack(ot), np.stack(otv))
        Ut, Vt, _r = proj(Ut, Vt, W.H)
        Ut, Vt = _pin_domain(Ut, Vt, ring)

    sigma = rel_l2(U, Ut, V, Vt)
    norm = float(np.sqrt(np.sum(mono_u ** 2) + np.sum(mono_v ** 2)))
    dl = float(max(dlam)) / norm
    bound = sigma_halo_bound(pou, dl)
    implied = sigma / max(pi * dl, 1e-300)
    if not quiet:
        say(f"  A6 W49: Pi={pi:.4g} ||d_lambda||={dl:.4e} sigma={sigma:.4e} "
            f"bound(C_mu=1.2)={bound:.4e} ratio={bound / max(sigma, 1e-300):.2f}")
        say(f"  A6 -> implied C_mu = {implied:.4f}  "
            f"(WindowNS measured [0.228, 1.178] over 16 configurations)")
    return {"Pi": pi, "d_lambda": dl, "sigma_paired": sigma,
            "bound_at_C_mu_1_2": bound, "bound_holds": bool(bound >= sigma),
            "implied_C_mu": implied,
            "inside_windowns_range": bool(0.228 <= implied <= 1.178),
            "note": "d_lambda is `w49_sigma_halo._face_trace`'s functional, max "
                    "over sub-steps, normalized by the reference norm -- the same "
                    "definition WindowNS's [0.228, 1.178] was measured with"}


def _compile_both(u, v, dt):
    out = {}
    for mode in SCHEMES:
        g, _ex = C.build(u, v, mode=mode, dt=dt)
        r = compile_scheme(g)
        ds = list(r.decisions)
        out[mode] = {
            "verdict": r.verdict.value,
            "unmeasured": list(r.unmeasured),
            "stamp": r.envelope.summary(),
            "refusals": [f"{d.layer}/{d.rule}" for d in ds
                         if d.verdict.value == "refuse"],
            "decertifications": [f"{d.layer}/{d.rule}" for d in ds
                                 if d.verdict.value == "admit-uncertified"],
        }
        say(f"  A9 {mode:11s}: {r.verdict.value}  refusals={out[mode]['refusals']}  "
            f"decerts={out[mode]['decertifications']}")
    return out


# ===========================================================================
# B -- Poseidon-T, the learned operator
# ===========================================================================


def part_b(u, v, dt=W.MACRO_DT, m_eff=W.M_EFF):
    from atlas.cases import poseidon as P                          # noqa: PLC0415

    res = {"expert": "Poseidon-T (camlab-ethz), frozen", "dt": dt}
    n = P.EXPERT_RES
    u0 = np.ascontiguousarray(u[:n, :n])
    v0 = np.ascontiguousarray(v[:n, :n])
    # An AGENT, not the raw wrapper: `respond` and the ring bookkeeping live on
    # the agent, and probing the wrapper directly would be probing a different
    # object from the one the compiler sees.
    ag = P.PoseidonAgent(agent_id="probe", u0=u0, v0=v0,
                         shared_faces=("xlo", "xhi", "ylo", "yhi"), dt=dt)
    res["n_params"] = int(ag.expert.n_params)
    res["tiling"] = {"n": n, "mono_n": P.MONO_N, "halo": P.DEFAULT_TILING.halo}

    # -- B1: determinism and batch-position dependence ---------------------
    a1 = ag.step(u0, v0, dt)
    a2 = ag.step(u0, v0, dt)
    same = float(np.max(np.abs(a1[0] - a2[0])))
    batch = P.batch_position_spread(ag, u0, v0, dt)
    say(f"  B1 repeat-call determinism: max|d| = {same:.3e}")
    say(f"  B1 batch-position spread:   max|d| = {batch:.3e}  (OP-6 asserted 1e-6)")
    res["determinism"] = {"repeat_call": same, "batch_position": batch,
                          "op6_asserted": 1e-6,
                          "op6_ratio": float(batch / 1e-6)}

    # The KMP_DUPLICATE_LIB_OK guard above is documented as able to produce
    # silently wrong results, so it is checked rather than trusted: an SVD with
    # a known answer, taken in this process, after the checkpoint is resident.
    _m = np.diag([3.0, 2.0, 1.0]) @ np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    _sv = np.linalg.svd(_m, compute_uv=False)
    res["linalg_sane_under_omp_workaround"] = bool(
        np.allclose(np.sort(_sv)[::-1], [3.0, 2.0, 1.0], atol=1e-12))
    say(f"  B1 linalg sanity under the OpenMP workaround: "
        f"{res['linalg_sane_under_omp_workaround']}")

    # -- B2: the probe floor, measured ------------------------------------
    sweep = P.eps_sweep(ag, u0, v0, dt, m_eff=m_eff)
    res["probe_floor"] = sweep
    for row in sweep["rows"]:
        say(f"  B2 eps={row['eps']:.0e}: beta={row['beta']:.6e} "
            f"kappa={row['kappa']:.4g} ||S||={row['norm_S']:.6e}")
    say(f"  B2 -> trusted window {sweep['trusted_window']}, floor "
        f"{sweep['floor']:.0e} -- and WindowNS's is machine epsilon")

    # -- B3: W2 -----------------------------------------------------------
    res["w2"] = P.probe_seam(ag, u0, v0, dt, eps=sweep["best_eps"], m_eff=m_eff)
    w = res["w2"]
    say(f"  B3 W2: beta={w['beta']:.4e} kappa={w['kappa']:.4g} null={w['null_dim']} "
        f"mu={w['mu']:+.4e} pi={w['pi']:.4e} asym={w['asymmetry']:.4f}")

    # -- B4: Xi against the WindowNS exposed agent ------------------------
    res["xi"] = P.xi_against_window(ag, u, v, dt, eps=sweep["best_eps"], m_eff=m_eff)
    say(f"  B4 Xi = ||Lambda_poseidon|| / ||Lambda_windowns|| = {res['xi']['Xi']:.4f}")

    # -- B5: the compile ---------------------------------------------------
    g = P.build(u, v, dt=dt)[0]
    r = compile_scheme(g)
    ds = list(r.decisions)
    res["compile"] = {
        "verdict": r.verdict.value,
        "unmeasured": list(r.unmeasured),
        "stamp": r.envelope.summary(),
        "refusals": [f"{d.layer}/{d.rule}" for d in ds if d.verdict.value == "refuse"],
        "decertifications": [f"{d.layer}/{d.rule}" for d in ds
                             if d.verdict.value == "admit-uncertified"],
    }
    say(f"  B5 compile: {r.verdict.value} refusals={res['compile']['refusals']}")
    say(f"  B5 decerts={res['compile']['decertifications']}")

    # -- B6: what cannot be measured, and why -----------------------------
    res["structurally_unavailable"] = {
        "tau": "no same-class monolith: the checkpoint is fixed at 128x128",
        "sigma": "same",
        "C_mu": "needs sigma",
        "cut_defect_bound (reference form)": "needs E u, i.e. a monolith",
        "note": "L2/C2's REFERENCE-FREE surrogate is available and is the only "
                "form that is -- which is the case it was derived for",
    }
    return res


# ===========================================================================


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,B")
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w55"))
    ap.add_argument("--dt", type=float, default=W.MACRO_DT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    parts = [p.strip().upper() for p in a.parts.split(",") if p.strip()]

    u, v = load_state()
    result = {"state": "out/tier0b/s0_state.npz, developed wake t=5", "dt": a.dt}

    if "A" in parts:
        say("A -- reference.ChannelNS, a second discretization")
        result["A_channel_ns"] = part_a(u, v, dt=a.dt)
    if "B" in parts:
        say("B -- Poseidon-T, a learned operator")
        try:
            result["B_poseidon"] = part_b(u, v, dt=a.dt)
        except Exception as exc:                                   # noqa: BLE001
            say(f"  B FAILED: {type(exc).__name__}: {exc}")
            result["B_poseidon"] = {"failed": f"{type(exc).__name__}: {exc}"}

    path = os.path.join(a.out, "w55.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=_j)
    say(f"wrote {path}")
    return result


if __name__ == "__main__":
    main()
