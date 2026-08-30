"""L6: what condition must a partition-of-unity blend satisfy?

`AssemblyCertificate.condition` is the single named hole blocking every graph
from `admit`.  This script measures the quantities a candidate condition would
be stated in, on the SAME configuration `tier0_window_ns.py` reports, before any
rule is written.

Four measurements:

  M1  the blend defect as currently emitted is VACUOUS -- it cuts the reference
      into local pieces and blends them, so the identity makes it zero for ANY
      partition of unity, including a terrible one
  M2  an honest blend defect, from real local solves at one sub-step against a
      no-projection monolith of the same class, for six partitions of unity
  M3  the candidate condition's quantities: chi_min, C_inf, G = max|grad chi|,
      Delta_ov, Theta -- all computable with NO reference
  M4  the convexity theorem, tested cellwise: does chi >= 0 actually imply the
      blend is inside the hull of what it blends, and does a signed partition
      break it?

Run:  python scripts/l6_assembly_condition.py [--state out/tier0_verify/s0_state.npz]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.assembly import certify                                 # noqa: E402
from atlas.cases import window_ns as W                             # noqa: E402


# ---------------------------------------------------------------------------
# partitions of unity to compare, all on the same tiling
# ---------------------------------------------------------------------------


def _raw_ramp(tiling, ramp, power=2):
    """The case study's own shape, with the ramp width as a parameter."""
    raw = []
    for ox, oy in tiling.offsets:
        r = max(ramp, 1)
        idx = np.arange(tiling.n) + 0.5
        wx = np.ones(tiling.n)
        wy = np.ones(tiling.n)
        if ox > 0:
            wx = np.minimum(wx, np.clip(idx / r, 0.0, 1.0))
        if ox + tiling.n < W.MONO_N:
            wx = np.minimum(wx, np.clip((tiling.n - idx) / r, 0.0, 1.0))
        if oy > 0:
            wy = np.minimum(wy, np.clip(idx / r, 0.0, 1.0))
        if oy + tiling.n < W.MONO_N:
            wy = np.minimum(wy, np.clip((tiling.n - idx) / r, 0.0, 1.0))
        w = np.zeros((W.MONO_N, W.MONO_N))
        w[oy:oy + tiling.n, ox:ox + tiling.n] = np.minimum(wy[:, None], wx[None, :]) ** power
        raw.append(w)
    return raw


def _normalize(raw):
    total = np.sum(raw, axis=0)
    total = np.where(total <= 0.0, 1.0, total)
    return [w / total for w in raw]


def pou_flat(tiling):
    """chi = 1 / (owner count): full weight right up to the artificial edge.

    This is what the first (as-built) measurement used, and section 8.1 of
    `tier0-measurements` records it as one of the three things that had to change.
    """
    raw = []
    for ox, oy in tiling.offsets:
        w = np.zeros((W.MONO_N, W.MONO_N))
        w[oy:oy + tiling.n, ox:ox + tiling.n] = 1.0
        raw.append(w)
    return _normalize(raw)


def pou_ramp(tiling, ramp):
    return _normalize(_raw_ramp(tiling, ramp))


def pou_signed(tiling, ramp=8, excess=0.6):
    """A partition that sums to one and has a NEGATIVE weight.

    Extrapolatory blending: lean past one window by ``excess`` and take the
    deficit off the other.  The identity holds exactly, so every check the
    framework has today passes it.  It is the deliberate bad blend.
    """
    ws = pou_ramp(tiling, ramp)
    out = [w.copy() for w in ws]
    both = (ws[0] > 1e-12) & (ws[1] > 1e-12)
    out[0] = np.where(both, ws[0] + excess, ws[0])
    out[1] = np.where(both, ws[1] - excess, ws[1])
    return out


PARTITIONS = {
    "flat (1/owner count)": lambda t: pou_flat(t),
    "ramp 1 (hard switch)": lambda t: pou_ramp(t, 1),
    "ramp 4": lambda t: pou_ramp(t, 4),
    "ramp 8 (the working one)": lambda t: pou_ramp(t, 8),
    "ramp 21 (full overlap)": lambda t: pou_ramp(t, 21),
    "signed (extrapolatory)": lambda t: pou_signed(t, 8, 0.6),
}


# ---------------------------------------------------------------------------
# the quantities, computed from chi and the local solves only
# ---------------------------------------------------------------------------


def grad_inf(field):
    """max cell-to-cell difference, both axes: the discrete ||grad||_inf * h."""
    return float(max(np.abs(np.diff(field, axis=0)).max(),
                     np.abs(np.diff(field, axis=1)).max()))


def chi_stats(ws, tiling):
    """chi_min, C_inf, G = max_i ||grad chi_i||_inf, and the identity residual."""
    total = np.sum(ws, axis=0)
    return {
        "chi_min": float(min(w.min() for w in ws)),
        "C_inf": float(max(w.max() for w in ws)),
        "G_max_grad_chi": float(max(grad_inf(w) for w in ws)),
        "identity_residual": float(np.abs(total - 1.0).max()),
        "n_negative_cells": int(sum(int((w < -1e-12).sum()) for w in ws)),
        "norm_A_assembly": float(np.sqrt(np.sum([w ** 2 for w in ws], axis=0).max())),
    }


def assemble_with(ws, tiling, locals_):
    """sum_i R_i^T chi_i u_i for per-window fields ``locals_`` (4, n, n)."""
    out = np.zeros((W.MONO_N, W.MONO_N))
    for k, (ox, oy) in enumerate(tiling.offsets):
        chi = ws[k][oy:oy + tiling.n, ox:ox + tiling.n]
        out[oy:oy + tiling.n, ox:ox + tiling.n] += chi * locals_[k]
    return out


def lifted(tiling, locals_):
    """Each local field on the global grid, plus its support mask."""
    fields, masks = [], []
    for k, (ox, oy) in enumerate(tiling.offsets):
        f = np.full((W.MONO_N, W.MONO_N), np.nan)
        m = np.zeros((W.MONO_N, W.MONO_N), bool)
        f[oy:oy + tiling.n, ox:ox + tiling.n] = locals_[k]
        m[oy:oy + tiling.n, ox:ox + tiling.n] = True
        fields.append(f)
        masks.append(m)
    return fields, masks


def overlap_mask(tiling):
    c = np.zeros((W.MONO_N, W.MONO_N))
    for ox, oy in tiling.offsets:
        c[oy:oy + tiling.n, ox:ox + tiling.n] += 1.0
    return c >= 2.0


def defects(ws, tiling, locals_, reference, mask):
    """Everything the certificate could report, with and without a reference.

    Returns the two competing alphas -- max-of-norms (what `blend_defect`
    computes today) and norm-of-cellwise-max (what the convexity theorem
    actually bounds) -- plus the reference-free quantities.
    """
    blend = assemble_with(ws, tiling, locals_)
    fields, masks = lifted(tiling, locals_)

    err = np.abs(blend - reference)
    per_local = [np.where(m, np.abs(f - reference), 0.0) for f, m in zip(fields, masks)]
    cellmax = np.max(per_local, axis=0)

    n2 = lambda a: float(np.linalg.norm(a[mask]))                   # noqa: E731
    ninf = lambda a: float(np.abs(a[mask]).max())                   # noqa: E731

    max_of_norms = max(float(np.linalg.norm(np.abs(f - reference)[mask & m]))
                       for f, m in zip(fields, masks))
    max_of_infs = max(float(np.abs(f - reference)[mask & m].max())
                      for f, m in zip(fields, masks))

    # reference-free: disagreement of each local with the blend, on its support
    disagree = max(float(np.abs(f - blend)[mask & m].max())
                   for f, m in zip(fields, masks))
    stats = chi_stats(ws, tiling)
    g = stats["G_max_grad_chi"]
    field_grad = grad_inf(blend)

    # the hull test: is the blend inside the range of what it blends, cellwise?
    lo = np.min([np.where(m, f, np.inf) for f, m in zip(fields, masks)], axis=0)
    hi = np.max([np.where(m, f, -np.inf) for f, m in zip(fields, masks)], axis=0)
    esc = np.maximum(blend - hi, lo - blend)

    # the bias-variance identity, which is what actually settles the condition:
    #   |A(u) - u*|^2  =  sum_i chi_i |u_i - u*|^2  -  V_chi
    # with V_chi = sum_i chi_i u_i^2 - (sum_i chi_i u_i)^2 the chi-weighted
    # variance of the local values.  It is an identity for ANY weights summing to
    # one; V_chi >= 0 for all data iff chi >= 0.  Computable with no reference.
    wsq = np.zeros_like(blend)
    wmean_err = np.zeros_like(blend)
    for k, (f, m) in enumerate(zip(fields, masks)):
        chi = np.where(m, ws[k], 0.0)
        val = np.where(m, f, 0.0)
        wsq += chi * val ** 2
        wmean_err += chi * np.where(m, (f - reference) ** 2, 0.0)
    v_chi = wsq - blend ** 2
    identity_gap = float(np.abs((wmean_err - v_chi) - (blend - reference) ** 2)[mask].max())
    return {
        **stats,
        "blend_err_L2": n2(err),
        "blend_err_inf": ninf(err),
        "max_of_local_norms_L2": max_of_norms,
        "norm_of_cellmax_L2": n2(cellmax),
        "max_of_local_infs": max_of_infs,
        "alpha_max_of_norms": n2(err) - max_of_norms,          # today's definition
        "alpha_cellmax_L2": n2(err) - n2(cellmax),             # what convexity bounds
        "alpha_inf": ninf(err) - max_of_infs,                  # what convexity bounds
        "hull_escape_max": float(esc[mask].max()),
        "Delta_ov": disagree,
        "blend_grad_defect": g * disagree,
        "field_grad": field_grad,
        "Theta": g * disagree / max(field_grad, 1e-300),
        # the bias-variance identity and its margin
        "V_chi_min": float(v_chi[mask].min()),
        "V_chi_L2": float(np.sqrt(np.clip(v_chi, 0.0, None)[mask].sum())),
        "V_chi_negative_cells": int((v_chi[mask] < -1e-18).sum()),
        "weighted_mean_local_err_L2": float(np.sqrt(wmean_err[mask].sum())),
        "bias_variance_identity_gap": identity_gap,
    }


# ---------------------------------------------------------------------------
# M1 -- the emitted blend defect is vacuous
# ---------------------------------------------------------------------------


def m1_vacuous(out, tiling, u_ref):
    """`certify(pou, locals_=cut(reference), reference=reference)` for every chi."""
    rows = {}
    for name, make in PARTITIONS.items():
        ws = make(tiling)
        idx, wts = {}, {}
        for k, (ox, oy) in enumerate(tiling.offsets):
            r, c = np.meshgrid(np.arange(oy, oy + tiling.n),
                               np.arange(ox, ox + tiling.n), indexing="ij")
            idx[tiling.names[k]] = (r * W.MONO_N + c).reshape(-1)
            wts[tiling.names[k]] = ws[k][oy:oy + tiling.n, ox:ox + tiling.n].reshape(-1)
        from atlas.assembly import GridPartitionOfUnity
        pou = GridPartitionOfUnity(W.MONO_N * W.MONO_N, idx, wts)
        us = tiling.cut(u_ref)
        locals_ = {n: us[k].reshape(-1) for k, n in enumerate(tiling.names)}
        cert = certify(pou, locals_=locals_, reference=u_ref.reshape(-1))
        rows[name] = {"pou_residual": cert.pou_residual,
                      "norm_A": float(cert.norm_A),
                      "blend_defect_as_emitted": cert.blend_defect}
        print(f"  M1 {name:26s} residual={cert.pou_residual:.2e} "
              f"norm_A={float(cert.norm_A):.4f} blend_defect={cert.blend_defect:.3e}")
    return rows


# ---------------------------------------------------------------------------
# M2/M3/M4 -- one honest sub-step
# ---------------------------------------------------------------------------


def local_solves(tiling, u, v, fx, nu=W.NU, dt=W.MACRO_DT, substeps=W.SUBSTEPS):
    """One sub-step of the exposed (no-projection) agent, per window, plus the
    same sub-step taken by a no-projection monolith of the same class.

    No projection anywhere and no assembly in between, so the local fields and
    the reference are the SAME map applied to the same state -- which is what
    isolates L6 from everything else.
    """
    cls = W._no_projection_class()
    mono = cls(nu=nu, length=W.MONO_L, n=W.MONO_N, cfl=0.4, transmission="dirichlet")
    win = cls(nu=nu, length=tiling.n * W.H, n=tiling.n, cfl=0.4,
              transmission="dirichlet")
    hs = dt / substeps
    us, vs = tiling.cut(u), tiling.cut(v)
    fxs, zero = tiling.cut(fx), np.zeros((4, tiling.n, tiling.n))
    u1, v1 = win.step_batch(us, vs, hs, bc0=(us, vs), bc1=None, force=(fxs, zero))
    ur, vr = mono.step_batch(u[None], v[None], hs, bc0=None, bc1=None,
                             force=(fx[None], np.zeros_like(fx)[None]))
    return (u1, v1), (ur[0], vr[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=os.path.join("out", "tier0_verify", "s0_state.npz"))
    ap.add_argument("--out", default=os.path.join("out", "l6"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.state)
    u, v, fx = d["u"], d["v"], d["fx"]
    tiling = W.DEFAULT_TILING
    print(f"tiling n={tiling.n} halo={tiling.halo} ramp={tiling.ramp} "
          f"required_halo={W.STENCIL_RADIUS * W.SUBSTEPS}")

    result = {"tiling": {"n": tiling.n, "halo": tiling.halo, "ramp": tiling.ramp},
              "state": a.state}

    print("\nM1 -- the blend defect as currently emitted")
    mono_full = W.load_reference().WindowNS(
        nu=W.NU, length=W.MONO_L, n=W.MONO_N, cfl=0.4, transmission="dirichlet")
    ur, vr = mono_full.step_batch(u[None], v[None], W.MACRO_DT, bc0=None, bc1=None,
                                 force=(fx[None], np.zeros_like(fx)[None]))
    result["M1_vacuous_blend_defect"] = m1_vacuous(out=None, tiling=tiling, u_ref=ur[0])

    print("\nM2/M3/M4 -- one honest sub-step, six partitions of unity")
    (lu, lv), (ru, rv) = local_solves(tiling, u, v, fx)
    mask = overlap_mask(tiling)
    print(f"  overlap cells: {int(mask.sum())} of {W.MONO_N ** 2}")

    rows = {}
    for name, make in PARTITIONS.items():
        ws = make(tiling)
        du = defects(ws, tiling, lu, ru, mask)
        dv = defects(ws, tiling, lv, rv, mask)
        rows[name] = {"u": du, "v": dv}
        print(f"\n  [{name}]")
        print(f"    chi_min={du['chi_min']:+.4f}  C_inf={du['C_inf']:.4f}  "
              f"G={du['G_max_grad_chi']:.4f}  identity={du['identity_residual']:.1e}")
        print(f"    blend_err_L2={du['blend_err_L2']:.4e}   "
              f"max_of_local_norms={du['max_of_local_norms_L2']:.4e}   "
              f"norm_of_cellmax={du['norm_of_cellmax_L2']:.4e}")
        print(f"    alpha(max-of-norms)={du['alpha_max_of_norms']:+.4e}   "
              f"alpha(cellmax)={du['alpha_cellmax_L2']:+.4e}   "
              f"alpha(inf)={du['alpha_inf']:+.4e}")
        print(f"    hull_escape={du['hull_escape_max']:.4e}   "
              f"Delta_ov={du['Delta_ov']:.4e}   field_grad={du['field_grad']:.4e}   "
              f"Theta={du['Theta']:.4e}")
    result["M2_M3_M4"] = rows

    with open(os.path.join(a.out, "l6_condition.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(f"\nwrote {os.path.join(a.out, 'l6_condition.json')}")


if __name__ == "__main__":
    main()
