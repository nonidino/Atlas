"""Proposition 3's discrete certificate, and the envelope an Expert Card declares.

**Proposition 3.**  A bilinearly interpolated chart whose Jacobian is positive at the
four corners of every cell, and whose boundary nodes form a simple, positively
oriented polygon, is a homeomorphism of the closed square onto the polygon's closed
interior.  The bilinear Jacobian is affine in each cell (the ``xi eta`` term cancels),
so positivity at the four corners is positivity on the cell; a local homeomorphism
whose boundary winds once, positively, is a bijection (a degree argument).  The cost
is O(n^2) for the Jacobians and O(n^2) segment pairs for simplicity.  With bilinear
elements (decision D7) the same condition is the validity of the physical mesh.

**The envelope.**  Where an expert is trusted, declared on its card: ``sup |mu| <= k_mu``,
``max log J - min log J <= l_J``, ``||grad log J||_inf + ||grad mu||_inf <= s`` (per unit
reference length), ``|log kappa_hat| <= l_kappa`` (the conductivity normalised by its
geometric mean on the piece), and the Fourier number in ``[fo_min, fo_max]`` or steady.
The host refuses a piece outside it and names the quantity (`refusals`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from . import chart as ch
from . import fe


@dataclass(frozen=True)
class Certificate:
    ok: bool
    min_corner_jacobian_rel: float
    nonpositive_corners: int
    boundary_simple: bool
    signed_area: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def corner_jacobians(X: np.ndarray) -> np.ndarray:
    """The bilinear Jacobian at the four corners of every cell, shape (n, n, 4)."""
    P00, P10 = X[:-1, :-1], X[:-1, 1:]
    P11, P01 = X[1:, 1:], X[1:, :-1]

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return np.stack([cross(P10 - P00, P01 - P00),       # at (0, 0)
                     cross(P10 - P00, P11 - P10),       # at (1, 0)
                     cross(P11 - P01, P11 - P10),       # at (1, 1)
                     cross(P11 - P01, P01 - P00)],      # at (0, 1)
                    axis=-1)


def signed_area(P: np.ndarray) -> float:
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_simple(P: np.ndarray) -> bool:
    """No two non-adjacent edges of the closed polygon P meet, checked over all pairs.

    Two segments meet if they cross properly, or if an end point of one lies on the
    other (orientation zero to rounding and inside its bounding box).  Collinear
    segments that do not overlap -- every straight side is a run of them -- do not
    meet."""
    N = len(P)
    A = P
    B = np.roll(P, -1, axis=0)
    scale = float(np.ptp(P, axis=0).max())
    eps = 1e-12 * scale * scale
    tol = 1e-12 * scale

    def orient(p, q, r):
        return ((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    def on_seg(p, q, r):
        """r within the bounding box of segment pq (with tolerance)."""
        return ((np.minimum(p[..., 0], q[..., 0]) - tol <= r[..., 0])
                & (r[..., 0] <= np.maximum(p[..., 0], q[..., 0]) + tol)
                & (np.minimum(p[..., 1], q[..., 1]) - tol <= r[..., 1])
                & (r[..., 1] <= np.maximum(p[..., 1], q[..., 1]) + tol))
    for k0 in range(0, N, 256):
        a, b = A[k0:k0 + 256, None, :], B[k0:k0 + 256, None, :]
        c, d = A[None, :, :], B[None, :, :]
        o1, o2 = orient(a, b, c), orient(a, b, d)
        o3, o4 = orient(c, d, a), orient(c, d, b)
        s1, s2 = np.sign(np.where(np.abs(o1) <= eps, 0, o1)), np.sign(np.where(np.abs(o2) <= eps, 0, o2))
        s3, s4 = np.sign(np.where(np.abs(o3) <= eps, 0, o3)), np.sign(np.where(np.abs(o4) <= eps, 0, o4))
        proper = (s1 * s2 < 0) & (s3 * s4 < 0)
        touch = (((s1 == 0) & on_seg(a, b, c)) | ((s2 == 0) & on_seg(a, b, d))
                 | ((s3 == 0) & on_seg(c, d, a)) | ((s4 == 0) & on_seg(c, d, b)))
        ii = np.arange(k0, min(k0 + 256, N))[:, None]
        jj = np.arange(N)[None, :]
        adjacent = (ii == jj) | ((ii + 1) % N == jj) | ((jj + 1) % N == ii)
        if np.any((proper | touch) & ~adjacent):
            return False
    return True


def certify(X: np.ndarray) -> Certificate:
    Jc = corner_jacobians(X)
    n = X.shape[0] - 1
    mean = float(np.mean(Jc))
    bad = int(np.sum(Jc <= 0.0))
    P = X.reshape(-1, 2)[fe.grid(n)[0]]
    area = signed_area(P)
    simple = polygon_simple(P)
    ok = bad == 0 and simple and area > 0.0
    if bad:
        reason = f"{bad} cell corners with a non-positive bilinear Jacobian"
    elif not simple:
        reason = "the boundary polygon intersects itself"
    elif area <= 0.0:
        reason = "the boundary polygon is not positively oriented"
    else:
        reason = "certified: positive bilinear Jacobian at every cell corner, simple positive boundary"
    return Certificate(ok, float(Jc.min()) / mean if mean != 0 else float("nan"), bad,
                       simple, area, reason)


# ---------------------------------------------------------------------------
# the envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Envelope:
    k_mu: float
    l_J: float
    s_grad: float
    l_kappa: float
    fo_min: float
    fo_max: float
    steady: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def _grad_sup(f: np.ndarray) -> float:
    """sup over elements of |grad f| per unit reference length, f nodal (real or complex)."""
    n = f.shape[0] - 1
    fx = 0.5 * ((f[:-1, 1:] - f[:-1, :-1]) + (f[1:, 1:] - f[1:, :-1])) * n
    fy = 0.5 * ((f[1:, :-1] - f[:-1, :-1]) + (f[1:, 1:] - f[:-1, 1:])) * n
    return float(np.sqrt(np.abs(fx) ** 2 + np.abs(fy) ** 2).max())


def measure(c: ch.Chart, log_kappa_hat: np.ndarray | None = None,
            fo: float | None = None) -> dict:
    """The envelope quantities of one charted piece."""
    el = c.element_fields()
    nd = c.nodal_fields()
    mu_e = ch.beltrami(ch.element_jacobian(c.X))[1]
    logJn = np.log(nd["J"])
    out = {
        "sup_abs_mu": float(max(np.abs(nd["mu"]).max(), np.abs(mu_e).max())),
        "logJ_range": float(np.log(el["J"]).max() - np.log(el["J"]).min()),
        "grad_logJ": _grad_sup(logJn),
        "grad_mu": _grad_sup(nd["mu"]),
    }
    out["s_grad"] = out["grad_logJ"] + out["grad_mu"]
    if log_kappa_hat is not None:
        out["abs_log_kappa"] = float(np.abs(log_kappa_hat).max())
    out["fo"] = fo
    return out


def refusals(m: dict, env: Envelope) -> list[str]:
    """Each violated quantity, named, with its value and the declared bound."""
    out = []
    if m["sup_abs_mu"] > env.k_mu:
        out.append(f"sup|mu| = {m['sup_abs_mu']:.4g} exceeds k_mu = {env.k_mu:g}")
    if m["logJ_range"] > env.l_J:
        out.append(f"max log J - min log J = {m['logJ_range']:.4g} exceeds l_J = {env.l_J:g}")
    if m["s_grad"] > env.s_grad:
        out.append(f"|grad log J| + |grad mu| = {m['s_grad']:.4g} exceeds s = {env.s_grad:g}")
    if m.get("abs_log_kappa") is not None and m["abs_log_kappa"] > env.l_kappa:
        out.append(f"|log kappa_hat| = {m['abs_log_kappa']:.4g} exceeds l_kappa = {env.l_kappa:g}")
    fo = m.get("fo")
    if fo is None or (isinstance(fo, float) and math.isinf(fo)):
        if not env.steady:
            out.append("a steady piece, and the card declares no steady range")
    elif not (env.fo_min <= fo <= env.fo_max):
        out.append(f"Fo = {fo:.4g} outside the declared range [{env.fo_min:g}, {env.fo_max:g}]")
    return out
