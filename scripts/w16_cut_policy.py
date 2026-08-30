"""G5/W16 -- the decomposition policy, derived and then falsified as scalarized.

The last thing standing between `cases/window_ns.py` in `split-step` mode and
`admit` is one decertification, `L2/G5/W16`: *"cut where the exact operator is
closest to local and the conditioning is comfortable"*, adopted as policy with
its **[AI Inference]** status preserved.  `tier0-measurements` §9.7 states the
terms exactly -- identified, never derived, never tested, and agreeing with two
other diagnostics on five configurations of one graph is not a derivation.

This script does both halves of what that leaves open.

  A  the identity        the composed defect over one exchange interval is
                         EXACTLY the chi-weighted assembly of the per-subdomain
                         restriction defects D_i = E_i R_i - R_i E.  Verified to
                         machine precision.  With chi convex (L6/C1) the same
                         cellwise argument that closed the assembly hole bounds
                         the composed defect by max_i |D_i|, which DERIVES the
                         locality half of the policy and gives it a definition.

  B  the falsification   Q = (1/beta) ||S - diag S|| / ||S|| is not a function of
                         the decomposition.  It moves under a change of the
                         declared interface basis, which leaves the scheme
                         bit-identical; and its 1/beta is the SUBSTRUCTURING
                         branch's amplifier, imported into an overlapping scheme
                         whose sigma bound (W49 / master-error-bound §4.1) has no
                         beta in it at all.  Scanned over cut placement, Q ranks
                         backwards.

Run:  python scripts/w16_cut_policy.py [--out out/w16]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from atlas.probe import cut_score                                   # noqa: E402
from strip_model import (                                           # noqa: E402
    STENCIL_RADIUS,
    SUBSTEPS_PER_EXCHANGE,
    Physics,
    StripDecomposition,
    cellwise_defect_bound,
    composed_step,
    neighbour_disagreement,
    reference_step,
    rel_l2,
    restriction_defect,
    seam_operator,
    smooth_field,
)

SEEDS = (0, 1, 2, 3)

#: A criterion is "constant over the scan" below this relative spread.  The
#: probe is a finite difference and a provably-identical operator comes back
#: with a spread near 1e-10, so zero is not the right threshold and neither is
#: 1e-12.  Measured on the uniform-nu control: Q 4.0e-11, beta 1.8e-11.
CONSTANT_TOL = 1e-8


def _j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def say(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# A -- the identity, and the bound it implies
# ---------------------------------------------------------------------------


def stage_identity(phys, dt):
    """The composed defect IS the chi-weighted assembly of the D_i."""
    rows = []
    for cut in (0, 7, 13, 24):
        for seed in SEEDS:
            dec = StripDecomposition(phys=phys, cut=cut)
            u = smooth_field(phys, seed=seed)
            lhs = composed_step(dec, u, dt) - reference_step(dec, u, dt)
            D = restriction_defect(dec, u, dt)
            rhs = np.zeros_like(lhs)
            ws = dec.weights()
            for i in range(dec.n_strips):
                r = dec.rows(i)
                rhs[r, :] += ws[i][r, :] * D[i]
            resid = float(np.max(np.abs(lhs - rhs)))
            scale = float(np.max(np.abs(lhs))) or 1.0
            bound = cellwise_defect_bound(dec, D)
            viol = float(np.max(np.abs(lhs) - bound))
            rows.append({
                "cut": cut, "seed": seed,
                "identity_residual_abs": resid,
                "identity_residual_rel": resid / scale,
                "max_abs_defect": scale,
                "cellwise_bound_violation": viol,
                "bound_holds": bool(viol <= 1e-15 * scale),
                "bound_tightness": float(np.max(np.abs(lhs)) / max(bound.max(), 1e-300)),
            })
    worst = max(r["identity_residual_rel"] for r in rows)
    viol = max(r["cellwise_bound_violation"] for r in rows)
    say(f"  A identity: worst relative residual {worst:.3e} over {len(rows)} runs")
    say(f"  A cellwise bound |A(u)-Eu|_j <= max_i |D_i|_j: worst violation {viol:.3e}")
    return {"rows": rows, "worst_identity_residual_rel": worst,
            "worst_cellwise_violation": viol}


# ---------------------------------------------------------------------------
# B1 -- Q is not a function of the decomposition: the basis moves it
# ---------------------------------------------------------------------------


def stage_basis(phys, dt, cut=0, seed=0, n_rot=64):
    """Same scheme, same interface space, different orthonormal frame for it.

    Replacing the declared prolongation P by P U for orthogonal U declares the
    SAME interface space -- range(P U) = range(P) -- so nothing the scheme does
    changes.  S becomes U^T S U, so beta and ||S|| are invariant and the
    off-diagonal mass is not.  Q therefore has a whole orbit of values on one
    decomposition, and the scheme it is supposed to rank has one.
    """
    dec = StripDecomposition(phys=phys, cut=cut)
    u = smooth_field(phys, seed=seed)
    S = seam_operator(dec, u, dt)
    beta = float(np.linalg.svd(S, compute_uv=False)[-1])
    q_declared = cut_score(S, beta)

    rng = np.random.default_rng(12345)
    qs = []
    for _ in range(n_rot):
        A = rng.normal(size=S.shape)
        U, _r = np.linalg.qr(A)
        Su = U.T @ S @ U
        b = float(np.linalg.svd(Su, compute_uv=False)[-1])
        qs.append(cut_score(Su, b))
    # the eigenbasis of the symmetric part: the extreme of the orbit
    sym = 0.5 * (S + S.T)
    _w, V = np.linalg.eigh(sym)
    S_eig = V.T @ S @ V
    q_eig = cut_score(S_eig, float(np.linalg.svd(S_eig, compute_uv=False)[-1]))

    # the scheme is bit-identical under the re-declaration: nothing in the
    # composed step reads the interface basis at all under the dirichlet rung.
    d_before = rel_l2(composed_step(dec, u, dt), reference_step(dec, u, dt))

    out = {
        "cut": cut, "seed": seed,
        "Q_declared_fourier": q_declared,
        "Q_min_over_random_frames": float(min(qs)),
        "Q_max_over_random_frames": float(max(qs)),
        "Q_in_eigenbasis_of_sym_part": q_eig,
        "orbit_ratio_max_over_min": float(max(qs) / max(min(qs), 1e-300)),
        "beta_invariant": beta,
        "beta_in_eigenbasis": float(np.linalg.svd(S_eig, compute_uv=False)[-1]),
        "composed_defect_unchanged": d_before,
        "note": "S -> U^T S U for orthogonal U is a re-declaration of the same "
                "interface space M; range(P U) = range(P). beta is invariant, "
                "Q is not.",
    }
    say(f"  B1 Q on one decomposition: declared {q_declared:.4g}, orbit "
        f"[{min(qs):.4g}, {max(qs):.4g}], eigenbasis {q_eig:.4g}")
    return out


# ---------------------------------------------------------------------------
# B2 -- scan the cut, and rank
# ---------------------------------------------------------------------------


def stage_scan(phys, dt, cuts, exchanges=8):
    rows = []
    nu = phys.nu_field()
    for cut in cuts:
        dec = StripDecomposition(phys=phys, cut=cut)
        # the operator at the cut, on a fixed reference state
        S = seam_operator(dec, smooth_field(phys, seed=0), dt)
        sv = np.linalg.svd(S, compute_uv=False)
        beta, kappa = float(sv[-1]), float(sv[0] / max(sv[-1], 1e-300))
        Q = cut_score(S, beta)

        d1, dN, qstar, qop, qhat, qw = [], [], [], [], [], []
        for seed in SEEDS:
            u = smooth_field(phys, seed=seed)
            nrm = max(float(np.linalg.norm(u)), 1e-300)
            d1.append(rel_l2(composed_step(dec, u, dt), reference_step(dec, u, dt)))
            D = restriction_defect(dec, u, dt)
            b = cellwise_defect_bound(dec, D)
            qstar.append(float(np.linalg.norm(b)) / nrm)
            qw.append(float(np.linalg.norm(
                cellwise_defect_bound(dec, D, weighted=True))) / nrm)
            qop.append(max(float(np.linalg.norm(x)) for x in D) / nrm)
            qhat.append(float(np.linalg.norm(
                neighbour_disagreement(dec, u, dt))) / nrm)
            # roll out `exchanges` intervals of both
            U, R = u.copy(), u.copy()
            for _ in range(exchanges):
                U = composed_step(dec, U, dt)
                R = reference_step(dec, R, dt)
            dN.append(rel_l2(U, R))

        nu_at_cut = float(np.mean([nu[dec.face_row(i, f)].mean()
                                   for i in (0, 1) for f in ("ylo", "yhi")]))
        off = float(np.linalg.norm(S - np.diag(np.diag(S))) / np.linalg.norm(S))
        rows.append({
            "cut": cut,
            "nu_at_faces": nu_at_cut,
            "beta": beta, "kappa": kappa, "Q_ai_inference": Q,
            # Q's two factors, separately, because they do not agree and the
            # whole finding is which of them is carrying the sign.
            "Q_locality_factor": off,
            "off_diagonal_mass_unnormalized": float(
                np.linalg.norm(S - np.diag(np.diag(S)))),
            "Q_conditioning_factor": 1.0 / beta,
            "Q_star_derived": float(np.mean(qstar)),
            "Q_star_chi_weighted": float(np.mean(qw)),
            "Q_hat_reference_free": float(np.mean(qhat)),
            "D_op_max": float(np.mean(qop)),
            "defect_1_interval": float(np.mean(d1)),
            f"defect_{exchanges}_intervals": float(np.mean(dN)),
        })
        say(f"  B2 cut={cut:3d}  nu={nu_at_cut:.3e}  beta={beta:.4e}  "
            f"Q={Q:9.4g}  Q*={np.mean(qstar):.4e}  "
            f"Qhat={np.mean(qhat):.4e}  defect={np.mean(d1):.4e}")
    return rows


def _rank_report(rows, exchanges):
    """Spearman rank correlation of each criterion against the measured defect."""
    def rank(vals):
        order = np.argsort(np.argsort(np.asarray(vals, dtype=float)))
        return order.astype(float)

    truth1 = rank([r["defect_1_interval"] for r in rows])
    truthN = rank([r[f"defect_{exchanges}_intervals"] for r in rows])
    out = {}
    keys = ("Q_ai_inference", "Q_locality_factor", "off_diagonal_mass_unnormalized",
            "Q_conditioning_factor", "Q_star_derived", "Q_star_chi_weighted",
            "Q_hat_reference_free", "D_op_max", "beta", "kappa")
    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=float)
        # A criterion that takes ONE value over the whole scan induces no
        # ordering, and argsort's arbitrary tie-break would report a correlation
        # for it anyway.  Say "constant" instead: it is the honest reading and it
        # is the finding in the uniform-nu control, where Q is exactly that.
        spread = float(vals.max() - vals.min()) / max(abs(float(vals.mean())), 1e-300)
        # CONSTANT_TOL, not zero: the probe is a finite difference, so a
        # quantity that is constant in exact arithmetic comes back with a
        # relative spread near 1e-10.  Measured, on the uniform-nu control where
        # S is provably identical at every cut: 4.0e-11 for Q, 1.8e-11 for beta.
        constant = spread < CONSTANT_TOL
        out[f"{key}__constant_over_scan"] = constant
        out[f"{key}__relative_spread"] = spread
        rk = rank(vals)
        for name, t in (("one_interval", truth1), (f"{exchanges}_intervals", truthN)):
            out[f"{key}__vs__{name}"] = (None if constant
                                         else float(np.corrcoef(rk, t)[0, 1]))

    d = [r["defect_1_interval"] for r in rows]
    lo, hi = float(min(d)), float(max(d))
    span = max(hi - lo, 1e-300)
    for key, tag in (("Q_ai_inference", "Q"), ("Q_star_derived", "Q_star"),
                     ("Q_hat_reference_free", "Q_hat")):
        k = int(np.argmin([r[key] for r in rows]))
        out[f"argmin_{tag}"] = rows[k]["cut"]
        out[f"defect_at_{tag}_argmin"] = d[k]
        out[f"{tag}_penalty_factor"] = d[k] / max(lo, 1e-300)
        # 0 = picked the best cut available, 1 = picked the worst
        out[f"{tag}_fraction_of_available_range_lost"] = (d[k] - lo) / span
    out["argmin_defect"] = rows[int(np.argmin(d))]["cut"]
    out["defect_best"] = lo
    out["defect_worst"] = hi
    out["defect_spread_factor"] = hi / max(lo, 1e-300)
    return out


# ---------------------------------------------------------------------------
# B4 -- when is the reference-free form EQUAL to the reference-requiring one?
# ---------------------------------------------------------------------------


def stage_halo(phys, dt, halos=(2, 3, 4, 5, 6, 8, 10), cut=10, seed=0):
    """Qhat = Q* exactly when the halo rule holds with room to spare.

    ``D_i`` is supported inside the agent's domain of dependence of its OWN
    artificial faces -- outside it, finite propagation makes ``E_i R_i u`` and
    ``R_i E u`` agree bit for bit.  When the halo exceeds twice that reach the
    two strips' defect supports are **disjoint**, and on every cell at most one
    ``D_i`` is nonzero, so ``max_i |D_i| = |D_i - D_j|``: the reference-free
    disagreement is not an approximation of the bound, it IS the bound.

    Narrow the halo until the supports collide and the two part company -- which
    is the same threshold the halo rule already refuses below, reached from a
    second direction.
    """
    reach = STENCIL_RADIUS * SUBSTEPS_PER_EXCHANGE
    rows = []
    for halo in halos:
        dec = StripDecomposition(phys=phys, halo=halo, ramp=min(4, halo), cut=cut)
        u = smooth_field(phys, seed=seed)
        D = restriction_defect(dec, u, dt)
        qstar = cellwise_defect_bound(dec, D)
        qhat = neighbour_disagreement(dec, u, dt)
        tol = 1e-13 * max(float(np.max(qstar)), 1e-300)
        supports = []
        for i in range(dec.n_strips):
            m = np.zeros((phys.ny, phys.nx), dtype=bool)
            m[dec.rows(i), :] = np.abs(D[i]) > tol
            supports.append(m)
        collide = int((supports[0] & supports[1]).sum())
        rel = float(np.linalg.norm(qhat - qstar) / max(np.linalg.norm(qstar), 1e-300))
        rows.append({
            "halo": halo, "domain_of_dependence": reach,
            "halo_rule_satisfied": bool(halo > reach),
            "supports_disjoint": collide == 0,
            "cells_where_both_D_nonzero": collide,
            "Q_hat_vs_Q_star_relative_gap": rel,
            "defect_1_interval": rel_l2(composed_step(dec, u, dt),
                                        reference_step(dec, u, dt)),
        })
        say(f"  B4 halo={halo:2d} (reach {reach})  supports disjoint="
            f"{str(collide == 0):5s}  |Qhat-Q*|/Q* = {rel:.3e}  "
            f"defect={rows[-1]['defect_1_interval']:.4e}")
    return rows


# ---------------------------------------------------------------------------
# B5 -- a localized feature, which is what a real cut decision is about
# ---------------------------------------------------------------------------


def stage_feature(phys, dt, cuts, exchanges=8, width=0.035):
    """Scan the cut past a localized structure, uniform nu.

    `nu` is uniform, so **every cut sees an identical operator** and Q is
    constant by construction across the whole scan.  The measured defect is not:
    cutting through the feature and cutting away from it differ, and that is the
    decision a wind farm actually faces -- cut through the wake or beside it.
    A criterion computed from the operator alone cannot express the question.
    """
    flat = Physics(nu_y_amp=0.0, nu_x_amp=phys.nu_x_amp)
    y = (np.arange(flat.ny) + 0.5) / flat.ny
    x = (np.arange(flat.nx) + 0.5) / flat.nx
    X, Y = np.meshgrid(x, y)
    blob = np.exp(-(((Y - 0.25) ** 2 + (X - 0.5) ** 2) / width))
    blob = blob / np.sqrt(np.mean(blob**2))

    rows = []
    for cut in cuts:
        dec = StripDecomposition(phys=flat, cut=cut)
        S = seam_operator(dec, blob, dt)
        beta = float(np.linalg.svd(S, compute_uv=False)[-1])
        d1 = rel_l2(composed_step(dec, blob, dt), reference_step(dec, blob, dt))
        qhat = float(np.linalg.norm(neighbour_disagreement(dec, blob, dt))
                     / np.linalg.norm(blob))
        U, R = blob.copy(), blob.copy()
        for _ in range(exchanges):
            U = composed_step(dec, U, dt)
            R = reference_step(dec, R, dt)
        rows.append({
            "cut": cut, "beta": beta, "Q_ai_inference": cut_score(S, beta),
            "Q_hat_reference_free": qhat,
            "defect_1_interval": d1,
            f"defect_{exchanges}_intervals": rel_l2(U, R),
        })
        say(f"  B5 cut={cut:3d}  Q={rows[-1]['Q_ai_inference']:.6g}  "
            f"Qhat={qhat:.4e}  defect={d1:.4e}")
    return rows


# ---------------------------------------------------------------------------
# C -- the same identity against the real expert, on the real tiling
# ---------------------------------------------------------------------------


def stage_real(state="out/tier0b/s0_state.npz"):
    """The restriction-defect identity on `reference.WindowNS`'s four windows.

    The strip model can move a cut and the four-window tiling cannot, which is
    why the falsification lives there.  What must live *here* is the identity
    itself: a derivation that closes to 1e-12 on a linear model problem and is
    never confronted with the real expert is a derivation about the model.

    Inputs are §9.1's honest ones -- one sub-step of the exposed agent per window
    against the same sub-step of a no-projection monolith of the same class, so
    no projection and no macro-step splitting is in the comparison and nothing
    but the restriction is being measured.
    """
    import numpy as np
    from atlas.cases import window_ns as W

    d = np.load(os.path.join(_ROOT, state))
    u, v = d["u"], d["v"]
    tiling = W.DEFAULT_TILING
    cls = W._no_projection_class()
    mono = cls(nu=W.NU, length=W.MONO_L, n=W.MONO_N, cfl=0.4, transmission="dirichlet")
    win = cls(nu=W.NU, length=tiling.n * W.H, n=tiling.n, cfl=0.4,
              transmission="dirichlet")
    hs = W.MACRO_DT / W.SUBSTEPS
    zero4 = np.zeros((4, tiling.n, tiling.n))
    zero1 = np.zeros((1, W.MONO_N, W.MONO_N))

    us, vs = tiling.cut(u), tiling.cut(v)
    lu, lv = win.step_batch(us, vs, hs, bc0=(us, vs), bc1=None, force=(zero4, zero4))
    ru, rv = mono.step_batch(u[None], v[None], hs, bc0=None, bc1=None,
                             force=(zero1, zero1))
    ru, rv = ru[0], rv[0]

    ws = tiling.weights()
    out = {}
    # One number for the graph, not one per component: v is an order smaller
    # than u in a through-flow, so normalizing each separately reports the
    # transverse component's smallness as a defect.  `rel_l2` in the Tier 0
    # driver uses the joint norm and so does every constant on the record.
    joint = float(np.sqrt(np.sum(u**2) + np.sum(v**2)))
    acc = {"bound_sq": 0.0, "qhat_sq": 0.0, "actual_sq": 0.0, "wbound_sq": 0.0}
    for tag, loc, ref, comp0 in (("u", lu, ru, u), ("v", lv, rv, v)):
        lhs = np.zeros((W.MONO_N, W.MONO_N))
        for k, (ox, oy) in enumerate(tiling.offsets):
            chi = ws[k][oy:oy + tiling.n, ox:ox + tiling.n]
            lhs[oy:oy + tiling.n, ox:ox + tiling.n] += chi * loc[k]
        lhs = lhs - ref
        rhs = np.zeros((W.MONO_N, W.MONO_N))
        bound = np.zeros((W.MONO_N, W.MONO_N))
        wbound = np.zeros((W.MONO_N, W.MONO_N))
        supports = []
        for k, (ox, oy) in enumerate(tiling.offsets):
            sl = (slice(oy, oy + tiling.n), slice(ox, ox + tiling.n))
            chi = ws[k][sl]
            D = loc[k] - ref[sl]
            rhs[sl] += chi * D
            bound[sl] = np.maximum(bound[sl], np.abs(D) * (chi > 0.0))
            wbound[sl] += chi * np.abs(D)
            m = np.zeros((W.MONO_N, W.MONO_N), dtype=bool)
            m[sl] = np.abs(D) > 1e-13 * max(float(np.abs(D).max()), 1e-300)
            supports.append(m)
        scale = max(float(np.max(np.abs(lhs))), 1e-300)
        collide = int(sum((supports[i] & supports[j]).sum()
                          for i in range(4) for j in range(i + 1, 4)))
        # The reference-free surrogate: neighbours' disagreement on the overlap.
        qhat = np.zeros((W.MONO_N, W.MONO_N))
        glob, own = [], []
        for k, (ox, oy) in enumerate(tiling.offsets):
            g = np.zeros((W.MONO_N, W.MONO_N))
            o = np.zeros((W.MONO_N, W.MONO_N), dtype=bool)
            g[oy:oy + tiling.n, ox:ox + tiling.n] = loc[k]
            o[oy:oy + tiling.n, ox:ox + tiling.n] = ws[k][oy:oy + tiling.n,
                                                          ox:ox + tiling.n] > 0.0
            glob.append(g); own.append(o)
        for i in range(4):
            for j in range(i + 1, 4):
                both = own[i] & own[j]
                qhat = np.maximum(qhat, np.abs(glob[i] - glob[j]) * both)
        nrm = max(float(np.linalg.norm(comp0)), 1e-300)
        acc["bound_sq"] += float(np.sum(bound**2))
        acc["wbound_sq"] += float(np.sum(wbound**2))
        acc["qhat_sq"] += float(np.sum(qhat**2))
        acc["actual_sq"] += float(np.sum(lhs**2))
        out[tag] = {
            "identity_residual_abs": float(np.max(np.abs(lhs - rhs))),
            "identity_residual_rel": float(np.max(np.abs(lhs - rhs))) / scale,
            "cellwise_bound_violation": float(np.max(np.abs(lhs) - bound)),
            "max_abs_defect": scale,
            "cells_where_two_D_nonzero": collide,
            "supports_disjoint": collide == 0,
            "Q_star_relative": float(np.linalg.norm(bound)) / nrm,
            "Q_hat_relative": float(np.linalg.norm(qhat)) / nrm,
            "Q_hat_over_Q_star": float(np.linalg.norm(qhat)
                                       / max(np.linalg.norm(bound), 1e-300)),
        }
        say(f"  C {tag}: identity residual {out[tag]['identity_residual_rel']:.3e} rel, "
            f"cellwise bound violation {out[tag]['cellwise_bound_violation']:.3e}")
        say(f"  C {tag}: Q* = {out[tag]['Q_star_relative']:.4e}, "
            f"Qhat = {out[tag]['Q_hat_relative']:.4e}, ratio "
            f"{out[tag]['Q_hat_over_Q_star']:.4f}, supports disjoint="
            f"{out[tag]['supports_disjoint']} ({collide} cells collide)")

    # The declared geometry that decides whether the surrogate is exact: are the
    # per-agent contaminated sets pairwise disjoint?  This is checkable at
    # compile time from the declaration, with no run and no state -- it is the
    # same `contaminated` the case study already declares for W49's Pi.
    cont = tiling.contaminated(W.STENCIL_RADIUS)
    pair = int(sum((cont[i].astype(bool) & cont[j].astype(bool)).sum()
                   for i in range(4) for j in range(i + 1, 4)))
    out["contaminated_sets_pairwise_disjoint"] = pair == 0
    out["contaminated_pairwise_overlap_cells"] = pair
    out["halo_cells"] = tiling.halo
    out["reach_per_exchange"] = W.STENCIL_RADIUS
    say(f"  C declared contaminated sets pairwise disjoint: {pair == 0} "
        f"({pair} cells shared) -- the compile-time form of the same test")

    # The constant the record will declare: L2/C2's bound over ONE exchange
    # interval, on the joint (u, v) field, relative.
    out["joint"] = {
        "cut_defect_bound": float(np.sqrt(acc["bound_sq"])) / joint,
        "cut_defect_bound_chi_weighted": float(np.sqrt(acc["wbound_sq"])) / joint,
        "cut_defect_bound_reference_free": float(np.sqrt(acc["qhat_sq"])) / joint,
        "actual_one_interval_defect": float(np.sqrt(acc["actual_sq"])) / joint,
        "exchange_interval": hs,
        "note": "one exchange interval (dt / substeps), exposed agent vs a "
                "no-projection monolith of the same class -- §9.1's honest inputs",
    }
    j = out["joint"]
    j["bound_tightness"] = j["actual_one_interval_defect"] / max(
        j["cut_defect_bound"], 1e-300)
    j["surrogate_ratio"] = j["cut_defect_bound_reference_free"] / max(
        j["cut_defect_bound"], 1e-300)
    j["bound_tightness_chi_weighted"] = j["actual_one_interval_defect"] / max(
        j["cut_defect_bound_chi_weighted"], 1e-300)
    say(f"  C joint: chi-weighted bound {j['cut_defect_bound_chi_weighted']:.4e} "
        f"(tightness {j['bound_tightness_chi_weighted']:.4f})")
    say(f"  C joint: L2/C2 bound {j['cut_defect_bound']:.4e}, actual defect "
        f"{j['actual_one_interval_defect']:.4e} (tightness "
        f"{j['bound_tightness']:.4f}), surrogate/bound "
        f"{j['surrogate_ratio']:.5f}")
    return out


def _rank_report_min(rows, exchanges):
    """The B5 form: only Q and Qhat are defined, and Q is expected constant."""
    def rank(vals):
        return np.argsort(np.argsort(np.asarray(vals, dtype=float))).astype(float)

    d = [r["defect_1_interval"] for r in rows]
    truth = rank(d)
    out = {}
    for key in ("Q_ai_inference", "Q_hat_reference_free", "beta"):
        vals = np.asarray([r[key] for r in rows], dtype=float)
        spread = (float(vals.max() - vals.min())
                  / max(abs(float(vals.mean())), 1e-300))
        const = spread < CONSTANT_TOL
        out[f"{key}__constant_over_scan"] = const
        out[f"{key}__relative_spread"] = spread
        out[f"{key}__vs__one_interval"] = (
            None if const else float(np.corrcoef(rank(vals), truth)[0, 1]))
        k = int(np.argmin(vals))
        out[f"argmin_{key}"] = rows[k]["cut"]
        out[f"defect_at_{key}_argmin"] = d[k]
    lo, hi = float(min(d)), float(max(d))
    out["argmin_defect"] = rows[int(np.argmin(d))]["cut"]
    out["defect_best"], out["defect_worst"] = lo, hi
    out["defect_spread_factor"] = hi / max(lo, 1e-300)
    out["Q_hat_penalty_factor"] = (
        out["defect_at_Q_hat_reference_free_argmin"] / max(lo, 1e-300))
    return out


def _report_min(r):
    q_const = r["Q_ai_inference__constant_over_scan"]
    say(f"       Q is {'CONSTANT' if q_const else 'varying'} over the scan "
        f"(relative spread {r['Q_ai_inference__relative_spread']:.1e}) while "
        f"the defect spreads {r['defect_spread_factor']:.2f}x")
    c = r["Q_hat_reference_free__vs__one_interval"]
    say(f"       Qhat rank corr vs measured defect: "
        f"{'constant' if c is None else f'{c:+.3f}'}")
    say(f"       argmin: Qhat -> cut {r['argmin_Q_hat_reference_free']}, "
        f"truth -> {r['argmin_defect']}, penalty "
        f"{r['Q_hat_penalty_factor']:.3f}x")


def _report(r):
    say("       criterion                  rank corr vs measured defect")
    for key, label in (("Q_ai_inference", "Q  (the [AI Inference])"),
                       ("Q_locality_factor", "  its locality factor (a RATIO)"),
                       ("off_diagonal_mass_unnormalized",
                        "  the same, UN-normalized"),
                       ("Q_conditioning_factor", "  its 1/beta factor"),
                       ("Q_star_derived", "Q* (max_i, needs the reference)"),
                       ("Q_star_chi_weighted", "Q* (chi-weighted, tighter)"),
                       ("Q_hat_reference_free", "Qhat (derived, reference-free)"),
                       ("D_op_max", "max_i ||D_i||")):
        c = r[key + "__vs__one_interval"]
        txt = "CONSTANT over the scan -- ranks nothing" if c is None else f"{c:+.3f}"
        say(f"       {label:34s} {txt}")
    say(f"       argmin: Q -> cut {r['argmin_Q']}, Q* -> {r['argmin_Q_star']}, "
        f"Qhat -> {r['argmin_Q_hat']}, truth -> {r['argmin_defect']}")
    say(f"       defect spread over the scan: {r['defect_spread_factor']:.3f}x")
    say(f"       following Q costs {r['Q_penalty_factor']:.3f}x the best cut "
        f"= {100 * r['Q_fraction_of_available_range_lost']:.0f}% of the "
        f"available range lost (100% = the worst cut on offer)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w16"))
    ap.add_argument("--exchanges", type=int, default=8)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    phys = Physics()
    dt = phys.stable_substep()
    say(f"strip model: {phys.ny}x{phys.nx} torus, nu in "
        f"[{phys.nu_field().min():.3e}, {phys.nu_field().max():.3e}], dt={dt:.4e}")
    say(f"halo 6 vs domain of dependence 2 (STENCIL_RADIUS x 1 substep/exchange)")

    result = {"physics": phys.__dict__, "dt": dt, "seeds": list(SEEDS)}

    say("A -- the restriction-defect identity")
    result["A_identity"] = stage_identity(phys, dt)

    say("B1 -- Q under a re-declaration of the same interface space")
    result["B1_basis"] = stage_basis(phys, dt)

    say("B2 -- scanning the cut")
    cuts = list(range(0, phys.ny // 2, 2))
    rows = stage_scan(phys, dt, cuts, exchanges=args.exchanges)
    result["B2_scan"] = rows
    result["B2_ranking"] = _rank_report(rows, args.exchanges)
    _report(result["B2_ranking"])

    # B3 -- the control.  With nu uniform in y every cut is the same cut, so a
    # criterion that ranks them is ranking noise.  Without this row the B2
    # anti-correlation could be an artifact of the scan rather than of Q.
    say("B3 -- control: nu uniform in y, so every cut placement is equivalent")
    flat = Physics(nu_y_amp=0.0)
    dt_flat = flat.stable_substep()
    rows_flat = stage_scan(flat, dt_flat, cuts, exchanges=args.exchanges)
    result["B3_control_uniform_nu"] = rows_flat
    result["B3_control_ranking"] = _rank_report(rows_flat, args.exchanges)
    dspread = result["B3_control_ranking"]["defect_spread_factor"]
    say(f"  B3 defect spread across all cuts: {dspread:.4f}x")
    _report(result["B3_control_ranking"])

    say("B4 -- the halo at which the reference-free form stops being exact")
    result["B4_halo"] = stage_halo(phys, dt)

    say("B5 -- a localized feature under a cut-independent operator")
    rows_f = stage_feature(phys, dt, cuts, exchanges=args.exchanges)
    result["B5_feature"] = rows_f
    result["B5_feature_ranking"] = _rank_report_min(rows_f, args.exchanges)
    _report_min(result["B5_feature_ranking"])

    say("C -- the identity against reference.WindowNS on the real four-window tiling")
    try:
        result["C_real_expert"] = stage_real()
    except Exception as exc:                                  # noqa: BLE001
        say(f"  C SKIPPED: {type(exc).__name__}: {exc}")
        result["C_real_expert"] = {"skipped": f"{type(exc).__name__}: {exc}"}

    path = os.path.join(args.out, "w16_cut_policy.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=_j)
    say(f"wrote {path}")
    return result


if __name__ == "__main__":
    main()
