"""A numerical conformal chart of a four-sided piece (plan §5: "a general numerical
conformal map onto a square is new work"; the owner, 2026-11-16: build one).

A piece is a Jordan domain with four marked corners, its sides in chartop's loop order
(`atlas.chartop.chart`).  Its **conformal module** ``m`` and its conformal map onto
the rectangle ``[0, m] x [0, 1]`` come from two harmonic measures (Gaier's method,
*Konstruktive Methoden der konformen Abbildung*, Springer 1964, ch. IV):

* ``v``: harmonic, ``v = 0`` on side 0 and ``v = 1`` on side 2, ``dv/dn = 0`` on
  sides 1 and 3;
* ``w``: harmonic, ``w = 0`` on side 3 and ``w = 1`` on side 1, ``dw/dn = 0`` on
  sides 0 and 2.

``v``'s harmonic conjugate is constant on the sides where ``v`` has zero flux, so it is
``m w`` with ``m = E(v) = ∫|∇v|²``, the module; and ``E(w) = 1/m``.  The map
``z -> (m w, v)`` is conformal onto the rectangle, so ``(w, v)`` is a conformal map
onto the unit square stretched by ``m`` along ``xi``: the chart chartop's
``harmonic(sides, n, aniso=m)`` computes when its sides are parametrised by this
**conformal boundary correspondence** (``w`` along sides 0 and 2, ``v`` along 1 and 3).
The product ``E(v) E(w)`` is 1 for exact solves; its distance from 1 is reported as
the discretisation's own error estimate.

Both measures are solved on a fine Winslow chart of the piece (a Coons patch when
Winslow does not converge) with chartop's bilinear elements (`PieceProblem.matrix`,
the physical Laplacian pulled back), so nothing here discretises anything new.  An
annular sector is recognised and charted by its closed-form log-polar map, the exact
conformal map (as `benchmarks.bend3` does).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse.linalg as spla

from atlas.chartop import chart as ch
from atlas.chartop.piece import PieceProblem


class ConformalError(ValueError):
    """The piece has no conformal chart this module can compute; the message says why."""


def _solve(K, fixed, values, N):
    free = np.setdiff1d(np.arange(N), fixed)
    u = np.zeros(N)
    u[fixed] = values
    K = K.tocsr()
    u[free] = spla.splu(K[free][:, free].tocsc()).solve(-K[free][:, fixed] @ values)
    return u


def measures(sides, fine: int = 128) -> dict:
    """The two harmonic measures on a fine chart of the piece, and the module."""
    try:
        Xf, _info = ch.winslow(sides, fine)
        made = "winslow"
    except Exception:
        Xf = ch.transfinite(sides, fine)
        made = "transfinite"
    p = PieceProblem(ch.Chart(Xf, made))
    el = p.element_fields()
    if np.any(el["J"] <= 0):
        raise ConformalError("the fine mesh of this piece folds: cut it so each side is "
                             "closer to straight")
    K = p.matrix()
    n1 = fine + 1
    idx = np.arange(n1 * n1).reshape(n1, n1)                 # [j, i]
    N = n1 * n1
    bot, top, lft, rgt = idx[0, :], idx[-1, :], idx[:, 0], idx[:, -1]
    v = _solve(K, np.concatenate([bot, top]), np.concatenate([np.zeros(n1), np.ones(n1)]), N)
    w = _solve(K, np.concatenate([lft, rgt]), np.concatenate([np.zeros(n1), np.ones(n1)]), N)
    Ev = float(v @ (K @ v))
    Ew = float(w @ (K @ w))
    return {"X": Xf, "v": v.reshape(n1, n1), "w": w.reshape(n1, n1), "module": Ev,
            "Ev": Ev, "Ew": Ew, "duality_defect": abs(Ev * Ew - 1.0), "fine_mesh": made}


def _monotone(a: np.ndarray) -> np.ndarray:
    a = np.maximum.accumulate(np.asarray(a, float))
    a = (a - a[0]) / max(a[-1] - a[0], 1e-300)
    return a


def correspondence(sides, M: dict) -> list:
    """The sides re-parametrised by the conformal boundary correspondence."""
    fine = M["v"].shape[0] - 1
    t = np.arange(fine + 1) / fine
    w, v = M["w"], M["v"]
    # each side's own parameter at the fine nodes is t (side 0, 1) or 1 - t (sides 2, 3,
    # which chartop's boundary_nodes reads backwards)
    along = [(t, _monotone(w[0, :])),                       # side 0: xi from 0 to 1
             (t, _monotone(v[:, -1])),                      # side 1: eta from 0 to 1
             (t, _monotone((1.0 - w[-1, ::-1]))),           # side 2: from corner 2 to 3
             (t, _monotone((1.0 - v[::-1, 0])))]            # side 3: from corner 3 to 0
    out = []
    for s, (tn, cn) in zip(sides, along):
        out.append(s.reparametrised(lambda tt, tn=tn, cn=cn: np.interp(tt, tn, cn)))
    return out


def annular_sector(sides, rtol: float = 1e-9):
    """(centre, r0, r1, a0, a1, radial_first) when the piece is an annular sector
    with two radial sides and two arcs about one centre; else None."""
    P = [np.asarray(s.P, float) for s in sides]
    for first in (0, 1):
        arcs = [P[(first + 1) % 4], P[(first + 3) % 4]]
        rays = [P[first], P[(first + 2) % 4]]
        # the centre: where the two radial lines meet
        a, b = rays[0][0], rays[0][-1]
        c, d = rays[1][0], rays[1][-1]
        A = np.array([b - a, -(d - c)]).T
        if abs(np.linalg.det(A)) < 1e-14:
            continue
        s_, _t = np.linalg.solve(A, c - a)
        ctr = a + s_ * (b - a)
        ok = True
        for arc in arcs:
            r = np.hypot(*(arc - ctr).T)
            if np.ptp(r) > rtol * max(r.max(), 1e-300):
                ok = False
        for ray in rays:
            d0 = ray - ray[0]
            cross = d0[:, 0] * (ray[-1] - ray[0])[1] - d0[:, 1] * (ray[-1] - ray[0])[0]
            if np.abs(cross).max() > rtol * np.hypot(*(ray[-1] - ray[0])) ** 2:
                ok = False
        if ok:
            return ctr, first
    return None


def conformal_chart(sides, n: int, fine: int = 128) -> ch.Chart:
    """The conformal chart of a four-sided piece at the reference grid's (n+1)^2 nodes."""
    sec = annular_sector(sides)
    if sec is not None:
        ctr, first = sec
        corners = [np.asarray(s.P[0], float) for s in sides]
        rel = [c - ctr for c in corners]
        rr = [math.hypot(*q) for q in rel]
        th = [math.atan2(q[1], q[0]) for q in rel]
        if first == 0:                        # side 0 radial: xi along r, eta along theta
            rho0, rho1 = math.log(rr[0]), math.log(rr[1])
            t0, t1 = th[0], th[3]
        else:                                 # side 0 an arc: xi along theta, eta along r
            rho0, rho1 = math.log(rr[0]), math.log(rr[3])
            t0, t1 = th[0], th[1]
        if t1 < t0 - math.pi:
            t1 += 2 * math.pi
        if first == 0:
            def F(xi, eta):
                r = np.exp(rho0 + (rho1 - rho0) * xi)
                a = t0 + (t1 - t0) * eta
                return ctr[0] + r * np.cos(a), ctr[1] + r * np.sin(a)
            mod = (rho1 - rho0) / (t1 - t0)
        else:
            def F(xi, eta):
                a = t0 + (t1 - t0) * xi
                r = np.exp(rho0 + (rho1 - rho0) * eta)
                return ctr[0] + r * np.cos(a), ctr[1] + r * np.sin(a)
            mod = (t1 - t0) / (rho1 - rho0)
        return ch.Chart(ch.from_map(F, n), "conformal-log-polar", {"modulus": abs(mod)})
    M = measures(sides, fine)
    new = correspondence(sides, M)
    X = ch.harmonic(new, n, aniso=M["module"])
    return ch.Chart(X, "conformal-numerical", {"modulus": M["module"],
                                               "duality_defect": M["duality_defect"],
                                               "fine": fine, "fine_mesh": M["fine_mesh"]})


__all__ = ["ConformalError", "annular_sector", "conformal_chart", "correspondence", "measures"]
