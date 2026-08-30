"""W49: a sigma bound that applies to the overlapping (halo) branch.

`master-error-bound` section 4 bounds sigma by ``C_mu/beta * ||Lambda - Lambda~||
* ||lambda*||``.  Measured 2026-08-27 on the halo scheme that bound overestimates
by 4.6e7, because it assumes the transmission error enters through an INTERFACE
SOLVE (hence the 1/beta amplifier) and a halo scheme has none.  Its accuracy comes
from the overlap width, and section 4 has no term for overlap.

The candidate replacement, derived from what actually carries the transmission
error in a halo scheme:

    sigma  <=  C_mu * Pi(chi, d) * ||d_lambda||
    Pi     =  max_j sum_i chi_ij * 1[ b_i(j) <= d_i ]

``b_i(j)`` is the distance in cells from j to agent i's nearest ARTIFICIAL face
and ``d_i = stencil_radius_i * substeps_per_exchange_i`` is how far that face's
datum can reach in one exchange interval.  ``Pi`` is the weight the assembly
gives to cells the stale boundary datum can have reached -- it is zero when the
partition of unity vanishes over the whole contaminated band, and one when the
overlap is narrower than the domain of dependence.  It is computable from the
DECLARATION alone: chi, stencil_radius and the exchange interval.

This script measures, for a sweep over halo width and over partitions of unity:

    Pi                 from the declaration
    ||d_lambda||       the artificial-boundary datum error, measured per sub-step
    sigma              measured, the driver's own definition
    the two bounds     section 4's product form and the candidate, side by side

Run:  python scripts/w49_sigma_halo.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas.cases import window_ns as W                             # noqa: E402
from l6_assembly_condition import assemble_with, pou_flat, pou_ramp  # noqa: E402
from tier0_window_ns import _pin_domain, make_solvers, mono_step, rel_l2  # noqa: E402


# ---------------------------------------------------------------------------
# Pi -- the contaminated weight, from the declaration only
# ---------------------------------------------------------------------------


def artificial_distance(tiling, k):
    """b_i(j): cells from j to window k's nearest ARTIFICIAL face, on the global grid.

    A face that coincides with the domain boundary is a real boundary and carries
    no stale datum, so it does not enter.  Cells outside the window are inf.
    """
    ox, oy = tiling.offsets[k]
    n, Mn = tiling.n, W.MONO_N
    faces = tiling.artificial_faces(ox, oy)
    ii = np.arange(n) + 0.5
    dx = np.full(n, np.inf)
    dy = np.full(n, np.inf)
    if "xlo" in faces:
        dx = np.minimum(dx, ii)
    if "xhi" in faces:
        dx = np.minimum(dx, n - ii)
    if "ylo" in faces:
        dy = np.minimum(dy, ii)
    if "yhi" in faces:
        dy = np.minimum(dy, n - ii)
    b = np.full((Mn, Mn), np.inf)
    b[oy:oy + n, ox:ox + n] = np.minimum(dy[:, None], dx[None, :])
    return b


def contaminated_weight(ws, tiling, d_cells):
    """Pi = max_j sum_i chi_ij 1[b_i(j) <= d_i], and the same weighted by an
    exponential attenuation exp(-b/d) as the sharper variant."""
    acc = np.zeros((W.MONO_N, W.MONO_N))
    soft = np.zeros((W.MONO_N, W.MONO_N))
    for k in range(len(tiling.offsets)):
        b = artificial_distance(tiling, k)
        acc += ws[k] * (b <= d_cells)
        soft += ws[k] * np.exp(-np.where(np.isfinite(b), b, 1e6) / max(d_cells, 1e-12))
    return float(acc.max()), float(soft.max())


# ---------------------------------------------------------------------------
# the composed step, instrumented for the boundary datum error
# ---------------------------------------------------------------------------


def _face_trace(cut_u, cut_v, tiling):
    """The four artificial rings of every window, flattened."""
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


def paired_step(ws, tiling, solvers, u, v, dt, fx, ring, substeps=None):
    """The lagged-ring and exact-ring composed steps, run together.

    Running them in lockstep is what makes ``||d_lambda||`` the datum difference
    the measured sigma is actually caused by, rather than a proxy for it.

    ``substeps`` defaults to the agent's own count at this dt -- R10b. Hard-coding
    it costs two orders in tau at any other dt.
    """
    mono, _emb, exposed = solvers
    if substeps is None:
        substeps = W.substeps_at(dt, mono)
    fxs, zero = tiling.cut(fx), np.zeros((4, tiling.n, tiling.n))
    u_ref, v_ref = mono_step(mono, u, v, dt, fx)

    U, V = u.copy(), v.copy()
    Ut, Vt = u.copy(), v.copy()
    hs = dt / substeps
    dlam, lam = [], []
    for m in range(substeps):
        us, vs = tiling.cut(U), tiling.cut(V)
        uts, vts = tiling.cut(Ut), tiling.cut(Vt)
        s = (m + 1) / substeps
        ru = (1 - s) * u + s * u_ref
        rv = (1 - s) * v + s * v_ref
        b_t = (tiling.cut(ru), tiling.cut(rv))

        dlam.append(float(np.linalg.norm(_face_trace(us, vs, tiling)
                                         - _face_trace(*b_t, tiling))))
        lam.append(float(np.linalg.norm(_face_trace(*b_t, tiling))))

        u1, v1 = exposed.step_batch(us, vs, hs, bc0=(us, vs), bc1=None,
                                    force=(fxs, zero))
        U, V = assemble_with(ws, tiling, u1), assemble_with(ws, tiling, v1)
        pu, pv = mono._project(U[None], V[None])
        U, V = _pin_domain(pu[0], pv[0], ring)

        u1t, v1t = exposed.step_batch(uts, vts, hs, bc0=b_t, bc1=None,
                                      force=(fxs, zero))
        Ut, Vt = assemble_with(ws, tiling, u1t), assemble_with(ws, tiling, v1t)
        put, pvt = mono._project(Ut[None], Vt[None])
        Ut, Vt = _pin_domain(put[0], pvt[0], ring)

    sigma = rel_l2(U, Ut, V, Vt)
    tau = rel_l2(Ut, u_ref, Vt, v_ref)
    total = rel_l2(U, u_ref, V, v_ref)
    norm = float(np.sqrt(np.sum(u_ref ** 2) + np.sum(v_ref ** 2)))
    return {
        "sigma": sigma, "tau": tau, "total": total,
        # relative, to match how sigma itself is normalized
        "dlambda_rel": float(max(dlam)) / norm,
        "dlambda_abs": float(max(dlam)),
        "lambda_norm": float(max(lam)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=os.path.join("out", "tier0_verify", "s0_state.npz"))
    ap.add_argument("--out", default=os.path.join("out", "l6"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.state)
    u, v, fx = d["u"], d["v"], d["fx"]
    dt = W.MACRO_DT
    # the split-step scheme exchanges every SUB-step, so the datum reaches
    # stencil_radius cells per exchange, not stencil_radius * substeps.
    d_cells = W.STENCIL_RADIUS
    print(f"domain of dependence per exchange: d = {d_cells} cells "
          f"(stencil_radius {W.STENCIL_RADIUS} x 1 sub-step per exchange)\n")

    rows = []
    cases = []
    for n in (128, 133, 138, 143, 148, 158):
        cases.append((n, 8, f"halo {2 * n - W.MONO_N:>3d}, ramp  8"))
    for ramp, label in ((0, "flat"), (1, "ramp  1"), (4, "ramp  4"),
                        (8, "ramp  8"), (21, "ramp 21")):
        cases.append((138, ramp, f"halo  21, {label}"))

    print(f"{'configuration':24s} {'halo':>5s} {'Pi':>10s} {'Pi_soft':>10s} "
          f"{'|dlambda|':>11s} {'sigma':>11s} {'bound':>11s} {'ratio':>10s}")
    for n, ramp, label in cases:
        tiling = W.Tiling(n=n, ramp=max(ramp, 1))
        ws = pou_flat(tiling) if ramp == 0 else pou_ramp(tiling, ramp)
        solvers = make_solvers(tiling)
        ring = (u.copy(), v.copy())
        m = paired_step(ws, tiling, solvers, u, v, dt, fx, ring)
        pi, pi_soft = contaminated_weight(ws, tiling, d_cells)
        bound = pi * m["dlambda_rel"]
        ratio = bound / max(m["sigma"], 1e-300)
        rows.append({"n": n, "ramp": ramp, "halo": tiling.halo, "label": label,
                     "Pi": pi, "Pi_soft": pi_soft, "bound_halo": bound,
                     "ratio": ratio, **m})
        print(f"{label:24s} {tiling.halo:5d} {pi:10.4e} {pi_soft:10.4e} "
              f"{m['dlambda_rel']:11.4e} {m['sigma']:11.4e} {bound:11.4e} "
              f"{ratio:10.2f}")

    with open(os.path.join(a.out, "w49_sigma.json"), "w", encoding="utf-8") as fh:
        json.dump({"d_cells": d_cells, "rows": rows}, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(a.out, 'w49_sigma.json')}")

    # the table above prints the bound at C_mu = 1, which is the raw factorization.
    # The recorded constant is C_MU_HALO = 1.2, the sampled maximum rounded up.
    from atlas.assembly import C_MU_HALO                          # noqa: PLC0415
    at1 = [r for r in rows if r["ratio"] >= 1.0]
    at_c = [r for r in rows if C_MU_HALO * r["ratio"] >= 1.0]
    lo = min(r["ratio"] for r in rows)
    hi = max(r["ratio"] for r in rows)
    print(f"\nat C_mu = 1:   holds on {len(at1)}/{len(rows)}, "
          f"overestimate {lo:.2f}x to {hi:.2f}x")
    print(f"at C_mu = {C_MU_HALO}: holds on {len(at_c)}/{len(rows)}, "
          f"overestimate {C_MU_HALO * lo:.2f}x to {C_MU_HALO * hi:.2f}x")
    imp = [r["sigma"] / (r["Pi"] * r["dlambda_rel"]) for r in rows]
    sig = [r["sigma"] for r in rows]
    print(f"C_mu implied:  [{min(imp):.4f}, {max(imp):.4f}], spread "
          f"{max(imp) / min(imp):.2f}x, while sigma moves "
          f"{max(sig) / min(sig):.3g}x")


if __name__ == "__main__":
    main()
