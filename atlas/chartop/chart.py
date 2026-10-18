"""Charts: maps from the reference square onto a four-sided piece, and what they give
the operator.

A piece is a Jordan domain with four marked corners, given by its four sides in loop
order (counter-clockwise): side 0 from corner 0 to corner 1 is the image of
``eta = 0``, side 1 of ``xi = 1``, side 2 of ``eta = 1`` (traversed with ``xi`` falling),
side 3 of ``xi = 0`` (``eta`` falling).  Each side carries a parametrisation ``t`` in
[0, 1]: by default its normalised arc length, or the parametrisation a neighbour
imposes on a shared side, so that the two pieces' nodes coincide (a conforming seam,
Theorem 2's setting).

A chart is sampled at the reference grid's ``(n+1)^2`` vertices, ``X[j, i] = F(i/n, j/n)``,
and interpolated bilinearly; `certificate.certify` decides whether that is a
homeomorphism (Proposition 3).  Three ways to make one (decision (b), the chart rule
of [[chart-operator-architecture]] section 3.2):

- ``transfinite``: the Coons patch of the four sides, explicit;
- ``winslow``: the harmonic chart with the sides' parametrisations as boundary data
  (the reference coordinates are harmonic functions on the piece, so the map is
  injective in the continuum by Rado-Kneser-Choquet), solved in its forward form
  ``g22 x_xixi - 2 g12 x_xieta + g11 x_etaeta = 0`` by Picard iteration from the Coons
  patch (Thompson, Warsi and Mastin, *Numerical Grid Generation*, 1985, ch. 6);
- ``from_map``: an explicit map, used for conformal charts known in closed form (the
  bend's log-polar map) and for the generator's analytic maps.

**What the operator sees.**  With ``DF`` the chart's Jacobian, ``J = det DF`` and the
complex derivatives ``F_z = (F_xi - i F_eta) / 2``, ``F_zbar = (F_xi + i F_eta) / 2``,
the Beltrami coefficient is ``mu = F_zbar / F_z`` and the pulled-back tensor of an
isotropic conductivity is ``kappa_hat J G^-1 = (kappa_hat / J) [[g22, -g12], [-g12, g11]]``
with ``g = DF^T DF``.  Its determinant is ``kappa_hat^2``: geometry changes only its
conformal class, which ``mu`` carries.  The element tensors come from ``DF`` at each
element's centre (exact for the bilinear element's area, since a bilinear map's
Jacobian is affine); the network's nodal inputs from the average of the adjacent
elements' ``DF``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# sides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Side:
    """A curve with a parametrisation: dense points ``P`` (k, 2) at increasing
    parameters ``t`` (k,) from 0 to 1, evaluated by linear interpolation."""
    P: np.ndarray
    t: np.ndarray

    def __call__(self, tq) -> np.ndarray:
        tq = np.asarray(tq, float)
        return np.stack([np.interp(tq, self.t, self.P[:, 0]),
                         np.interp(tq, self.t, self.P[:, 1])], axis=-1)

    @property
    def start(self) -> np.ndarray:
        return self.P[0]

    @property
    def end(self) -> np.ndarray:
        return self.P[-1]

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(*np.diff(self.P, axis=0).T)))

    def reversed(self) -> "Side":
        return Side(self.P[::-1].copy(), (1.0 - self.t[::-1]).copy())

    def reparametrised(self, s_of_t) -> "Side":
        """The same curve with new parameter values ``s_of_t(t)`` at the current
        samples, renormalised to run from 0 to 1 (``s_of_t`` must be increasing)."""
        new = np.asarray(s_of_t(self.t), float)
        return Side(self.P.copy(), (new - new[0]) / (new[-1] - new[0]))


def arclength(P: np.ndarray) -> np.ndarray:
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    return s


def side_from_polyline(P, dense: int = 2049) -> Side:
    """A side parametrised by normalised arc length, resampled to ``dense`` points."""
    P = np.asarray(P, float)
    s = arclength(P)
    s = s / s[-1]
    t = np.linspace(0.0, 1.0, dense)
    Q = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], axis=-1)
    Q[0], Q[-1] = P[0], P[-1]
    return Side(Q, t)


def side_from_function(f, dense: int = 2049, arclength_param: bool = True) -> Side:
    """A side from a function of a parameter on [0, 1]; by default re-parametrised by
    normalised arc length, otherwise keeping the function's own parameter."""
    t = np.linspace(0.0, 1.0, dense)
    P = np.asarray(f(t), float)
    if not arclength_param:
        return Side(P, t)
    s = arclength(P)
    return Side(P, s / s[-1])


def line(a, b, dense: int = 2049) -> Side:
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0.0, 1.0, dense)
    return Side(a[None, :] + t[:, None] * (b - a)[None, :], t)


def check_loop(sides) -> float:
    """The largest gap between consecutive sides' ends, relative to the perimeter."""
    per = sum(s.length for s in sides)
    gap = max(float(np.hypot(*(sides[k].end - sides[(k + 1) % 4].start))) for k in range(4))
    return gap / per


# ---------------------------------------------------------------------------
# chart constructions
# ---------------------------------------------------------------------------


def boundary_nodes(sides, n: int) -> dict:
    """The sides evaluated at t = i/n: B(xi), R(eta), T(xi), L(eta) as (n+1, 2)."""
    t = np.arange(n + 1) / n
    B = sides[0](t)
    R = sides[1](t)
    T = sides[2](1.0 - t)
    L = sides[3](1.0 - t)
    return {"B": B, "R": R, "T": T, "L": L}


def transfinite(sides, n: int) -> np.ndarray:
    """The Coons patch on the (n+1)^2 vertices, X[j, i] = F(i/n, j/n)."""
    bn = boundary_nodes(sides, n)
    B, R, T, L = bn["B"], bn["R"], bn["T"], bn["L"]
    t = np.arange(n + 1) / n
    XI, ETA = np.meshgrid(t, t)                         # [j, i]
    p00, p10, p11, p01 = B[0], B[-1], T[-1], T[0]
    X = np.empty((n + 1, n + 1, 2))
    for k in range(2):
        X[..., k] = ((1 - ETA) * B[None, :, k] + ETA * T[None, :, k]
                     + (1 - XI) * L[:, None, k] + XI * R[:, None, k]
                     - ((1 - XI) * (1 - ETA) * p00[k] + XI * (1 - ETA) * p10[k]
                        + XI * ETA * p11[k] + (1 - XI) * ETA * p01[k]))
    _set_boundary(X, bn)
    return X


def _set_boundary(X: np.ndarray, bn: dict) -> None:
    X[0, :, :] = bn["B"]
    X[-1, :, :] = bn["T"]
    X[:, 0, :] = bn["L"]
    X[:, -1, :] = bn["R"]


class WinslowDiverged(RuntimeError):
    pass


def winslow(sides, n: int, X0: np.ndarray | None = None, tol: float = 1e-11,
            maxit: int = 400, omega: float = 1.0) -> tuple[np.ndarray, dict]:
    """The harmonic (Winslow) chart with the sides' parametrisations as Dirichlet data.

    Picard iteration: freeze ``g11, g12, g22`` from the current nodes (central
    differences, unit spacing -- the equation is homogeneous), solve the 9-point
    linear system for the interior x and y with one sparse LU, relax by ``omega``.
    Stops when the largest node move is below ``tol`` times the piece's diameter;
    raises `WinslowDiverged` otherwise.  Returns the nodes and an iteration record."""
    bn = boundary_nodes(sides, n)
    X = transfinite(sides, n) if X0 is None else np.array(X0, float)
    _set_boundary(X, bn)
    diam = float(np.ptp(np.concatenate([bn[k] for k in "BRTL"]), axis=0).max())
    m = n - 1
    idx = np.arange(m * m).reshape(m, m)                # interior [j-1, i-1]
    hist = []
    for it in range(1, maxit + 1):
        x, y = X[..., 0], X[..., 1]
        x_xi = 0.5 * (x[1:-1, 2:] - x[1:-1, :-2])
        y_xi = 0.5 * (y[1:-1, 2:] - y[1:-1, :-2])
        x_et = 0.5 * (x[2:, 1:-1] - x[:-2, 1:-1])
        y_et = 0.5 * (y[2:, 1:-1] - y[:-2, 1:-1])
        g11 = x_xi ** 2 + y_xi ** 2
        g22 = x_et ** 2 + y_et ** 2
        g12 = x_xi * x_et + y_xi * y_et
        al, be, ga = g22.ravel(), g12.ravel(), g11.ravel()
        rows, cols, vals = [], [], []
        rhs = np.zeros((m * m, 2))
        jj, ii = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
        jj, ii = jj.ravel(), ii.ravel()
        me = idx.ravel()

        def add(dj, di, coef):
            tj, ti = jj + dj, ii + di
            inside = (tj >= 0) & (tj < m) & (ti >= 0) & (ti < m)
            rows.append(me[inside])
            cols.append(idx[tj[inside], ti[inside]])
            vals.append(coef[inside])
            out = ~inside
            if np.any(out):
                # boundary node: X index (tj+1, ti+1)
                bx = X[tj[out] + 1, ti[out] + 1]
                np.subtract.at(rhs, me[out], coef[out][:, None] * bx)
        add(0, 0, -2.0 * (al + ga))
        add(0, 1, al)
        add(0, -1, al)
        add(1, 0, ga)
        add(-1, 0, ga)
        add(1, 1, -0.5 * be)
        add(-1, -1, -0.5 * be)
        add(1, -1, 0.5 * be)
        add(-1, 1, 0.5 * be)
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m * m, m * m))
        sol = spla.splu(A).solve(rhs)
        new = sol.reshape(m, m, 2)
        move = float(np.max(np.abs(new - X[1:-1, 1:-1])))
        X[1:-1, 1:-1] = (1.0 - omega) * X[1:-1, 1:-1] + omega * new
        hist.append(move / diam)
        if not np.all(np.isfinite(X)) or move / diam > 1e3:
            raise WinslowDiverged(f"Winslow diverged at iteration {it}: move {move / diam:.3g}")
        if move / diam < tol:
            return X, {"iterations": it, "final_move": move / diam}
    raise WinslowDiverged(f"Winslow did not converge in {maxit} iterations "
                          f"(last move {hist[-1]:.3g} of the diameter)")


def harmonic(sides, n: int, aniso: float = 1.0) -> np.ndarray:
    """The forward map whose components solve ``x_xixi / aniso^2 + x_etaeta = 0`` with
    the sides as Dirichlet data (5 points, one sparse solve).  With a conformal
    boundary correspondence of a piece of modulus ``aniso`` this is the conformal
    chart: a conformal map onto a rectangle of modulus ``aniso``, composed with the
    stretch onto the square, has components harmonic in that anisotropic sense, and
    the Dirichlet problem's solution is unique."""
    bn = boundary_nodes(sides, n)
    X = transfinite(sides, n)
    _set_boundary(X, bn)
    m = n - 1
    idx = np.arange(m * m).reshape(m, m)
    w = 1.0 / aniso ** 2
    rows, cols, vals = [], [], []
    rhs = np.zeros((m * m, 2))
    for j in range(m):
        for i in range(m):
            r = idx[j, i]
            rows.append(r); cols.append(r); vals.append(-2.0 * (w + 1.0))
            for dj, di, c in ((0, 1, w), (0, -1, w), (1, 0, 1.0), (-1, 0, 1.0)):
                tj, ti = j + dj, i + di
                if 0 <= tj < m and 0 <= ti < m:
                    rows.append(r); cols.append(idx[tj, ti]); vals.append(c)
                else:
                    rhs[r] -= c * X[tj + 1, ti + 1]
    A = sp.csc_matrix((vals, (rows, cols)), shape=(m * m, m * m))
    X[1:-1, 1:-1] = spla.splu(A).solve(rhs).reshape(m, m, 2)
    return X


def from_map(F, n: int) -> np.ndarray:
    """X[j, i] = F(i/n, j/n) for a vectorised map F(xi, eta) -> (x, y)."""
    t = np.arange(n + 1) / n
    XI, ETA = np.meshgrid(t, t)
    x, y = F(XI, ETA)
    return np.stack([np.asarray(x, float), np.asarray(y, float)], axis=-1)


def resample(X: np.ndarray, n: int) -> np.ndarray:
    """The chart on another grid: every other node when halving a 2^k grid, bicubic
    interpolation of the nodes otherwise (any smooth map is a chart once certified)."""
    n0 = X.shape[0] - 1
    if n == n0:
        return X.copy()
    if n0 % n == 0:
        r = n0 // n
        return X[::r, ::r].copy()
    from scipy.interpolate import RectBivariateSpline
    t0 = np.arange(n0 + 1) / n0
    t = np.arange(n + 1) / n
    out = np.empty((n + 1, n + 1, 2))
    for k in range(2):
        out[..., k] = RectBivariateSpline(t0, t0, X[..., k], kx=3, ky=3)(t, t)
    return out


# ---------------------------------------------------------------------------
# what the chart gives the operator
# ---------------------------------------------------------------------------


def element_jacobian(X: np.ndarray) -> np.ndarray:
    """DF at each element's centre, per unit reference length: shape (n, n, 2, 2)
    with DF[..., :, 0] = d(x, y)/dxi and DF[..., :, 1] = d(x, y)/deta."""
    n = X.shape[0] - 1
    d_xi = 0.5 * ((X[:-1, 1:] - X[:-1, :-1]) + (X[1:, 1:] - X[1:, :-1])) * n
    d_et = 0.5 * ((X[1:, :-1] - X[:-1, :-1]) + (X[1:, 1:] - X[:-1, 1:])) * n
    return np.stack([d_xi, d_et], axis=-1)


def nodal_jacobian(X: np.ndarray) -> np.ndarray:
    """DF at the nodes: the average of the adjacent elements' centre values."""
    De = element_jacobian(X)
    n = X.shape[0] - 1
    acc = np.zeros((n + 1, n + 1, 2, 2))
    cnt = np.zeros((n + 1, n + 1, 1, 1))
    for dj in (0, 1):
        for di in (0, 1):
            acc[dj:dj + n, di:di + n] += De
            cnt[dj:dj + n, di:di + n] += 1.0
    return acc / cnt


def beltrami(DF: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(J, mu) from DF (..., 2, 2)."""
    x_xi, y_xi = DF[..., 0, 0], DF[..., 1, 0]
    x_et, y_et = DF[..., 0, 1], DF[..., 1, 1]
    F_xi = x_xi + 1j * y_xi
    F_et = x_et + 1j * y_et
    Fz = 0.5 * (F_xi - 1j * F_et)
    Fzb = 0.5 * (F_xi + 1j * F_et)
    J = x_xi * y_et - x_et * y_xi
    return J, Fzb / Fz


def metric_tensor(DF: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(a11, a12, a22, J) of J G^-1 = (1/J) [[g22, -g12], [-g12, g11]]."""
    x_xi, y_xi = DF[..., 0, 0], DF[..., 1, 0]
    x_et, y_et = DF[..., 0, 1], DF[..., 1, 1]
    g11 = x_xi ** 2 + y_xi ** 2
    g22 = x_et ** 2 + y_et ** 2
    g12 = x_xi * x_et + y_xi * y_et
    J = x_xi * y_et - x_et * y_xi
    return g22 / J, -g12 / J, g11 / J, J


def tensor_from_mu(mu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """J G^-1 from mu alone: (1 / (1 - |mu|^2)) [[|1 - mu|^2, -2 Im mu], [-2 Im mu, |1 + mu|^2]]."""
    den = 1.0 - np.abs(mu) ** 2
    return np.abs(1 - mu) ** 2 / den, -2.0 * mu.imag / den, np.abs(1 + mu) ** 2 / den


@dataclass
class Chart:
    """A sampled chart and everything derived from it.  ``method`` names how it was
    made (the Expert Card's chart methods)."""
    X: np.ndarray
    method: str
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0] - 1

    def element_fields(self) -> dict:
        De = element_jacobian(self.X)
        a11, a12, a22, J = metric_tensor(De)
        n = self.n
        area = float(J.sum()) / n ** 2
        return {"a11": a11, "a12": a12, "a22": a22, "J": J, "area": area,
                "Jt": J / area}

    def nodal_fields(self) -> dict:
        Dn = nodal_jacobian(self.X)
        J, mu = beltrami(Dn)
        return {"J": J, "mu": mu}

    def corners(self) -> np.ndarray:
        X = self.X
        return np.array([X[0, 0], X[0, -1], X[-1, -1], X[-1, 0]])

    def loop_points(self) -> np.ndarray:
        """The 4n boundary nodes in `fe.grid` loop order."""
        n = self.n
        P = self.X.reshape(-1, 2)
        from .fe import grid
        return P[grid(n)[0]]


def corner_angles(X: np.ndarray) -> np.ndarray:
    """The piece's interior angle at each corner (degrees), from the first boundary
    segment of each side: corner 0 at (0, 0), then counter-clockwise."""
    n = X.shape[0] - 1
    c = [X[0, 0], X[0, n], X[n, n], X[n, 0]]
    nxt = [X[0, 1], X[1, n], X[n, n - 1], X[n - 1, 0]]       # along the side leaving corner c
    prv = [X[1, 0], X[0, n - 1], X[n - 1, n], X[n, 1]]       # along the side arriving at c
    out = []
    for k in range(4):
        a = nxt[k] - c[k]
        b = prv[k] - c[k]
        ang = math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
        out.append(math.degrees(ang) % 360.0)
    return np.array(out)
