"""Body-fitted overset grids for RaceLab: curved grids wrapped around the car's
parts, a Cartesian background behind them, and one pressure system across all
of them.

PoC 3, Tier 60.  [[poc3-racelab-body-fitted-grids]].

**Why this module exists.**  Until Tier 59 every RaceLab fluid window was a
128x128 box on one uniform lattice, and every body was a porous body force
stamped into it.  On 2026-09-14 the user decided the windows must follow the
car, and chose body-fitted grids over shaped regions on the same lattice:
curved grids around each part, overlapping a background grid, with the bodies
as real walls.  This module is the foundation that decision needs -- the grids,
the overlap between them, and a pressure solve that couples all of them --
verified on shapes with known answers before any car part is put on it.

What is here, and what is not
-----------------------------

* `CartesianGrid`: the background.  Its points are the CELL CENTRES of the
  lattice RaceLab already marches, so a background field and a RaceLab field
  are the same array.
* `CurvilinearGrid`: a structured grid around one closed body.  ``i`` runs
  CLOCKWISE around the body and is periodic; ``j`` runs from the wall
  (``j = 0``) outward.  That orientation makes the mapping right-handed, so
  the Jacobian is positive, which the constructor requires.
* `ogrid_annulus` builds one analytically (a circle, with optional wall
  clustering and a twist that makes it non-orthogonal); `ogrid_from_outline`
  builds one by marching outward from any closed outline, and refuses a grid
  whose cells fold.
* `Overset`: hole cutting in the background, interpolation points on both
  sides of every overlap, their donors and weights, and a global numbering.
  A point that needs a donor and has none is an ORPHAN, and the constructor
  raises rather than returning a grid that silently leaks.
* `Overset.poisson`: the composite system.  Five-point Laplacian on the
  background, a nine-point conservative curvilinear Laplacian on each body
  grid, a Neumann (or Dirichlet) row on each wall, and one interpolation
  equation per interpolation point.
* `manufactured_poisson`: the verification, against an exact solution, with
  three single-grid controls beside the composite.

**Not here:** the flow solver (Tier 61), several bodies whose grids overlap
each other or the road (Tier 62), and anything learned.  A component grid that
leaves the box or enters another body is refused, not handled.

Conventions
-----------

Arrays are indexed ``[j, i]`` -- row first -- exactly as RaceLab's fields are
``[ny, nx]``.  Computational spacing is one index unit in both directions.
With ``x_xi = dx/di`` and so on, ``J = x_xi y_eta - x_eta y_xi`` and the
contravariant metric ``g^ab = grad(xi^a) . grad(xi^b)``; the grid stores
``a11 = J g^11 = (x_eta^2 + y_eta^2) / J``, ``a12 = J g^12 = -(x_xi x_eta +
y_xi y_eta) / J`` and ``a22 = J g^22 = (x_xi^2 + y_xi^2) / J``.  The Laplacian
is then ``(1/J) [ d_xi(a11 p_xi + a12 p_eta) + d_eta(a12 p_xi + a22 p_eta) ]``.

The wall normal ``n`` points OUT OF THE FLUID, into the body: ``n = -grad(eta)
/ |grad(eta)|``.  A Neumann value is ``dp/dn`` with that ``n``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

__all__ = [
    "CartesianGrid", "CurvilinearGrid", "Overset", "OversetError",
    "GridQualityError", "ogrid_annulus", "ogrid_from_outline", "grid_quality",
    "orient_clockwise", "resample_closed", "points_in_polygon",
    "distance_to_polygon", "lagrange_weights", "solve_sparse",
    "manufactured_poisson", "MS", "MS_BOX", "MS_BODY", "MS_THICKNESS",
    "MS_HOLE_MARGIN", "MS_ELLIPSE", "N_MASTER", "HOLE", "DISC", "INTERP", "WALL",
    "OUTER_BC", "STATUS_NAMES", "circle_outline", "ellipse_outline",
    "rounded_plate_outline", "naca4_outline", "dented_circle_outline",
    "signed_area", "KAPPA",
]

#: Point classes.  ``OUTER_BC`` is a component's outer ring when it carries a
#: Dirichlet value instead of an interpolation equation -- the single-grid
#: control, never a composite grid.
HOLE, DISC, INTERP, WALL, OUTER_BC = 0, 1, 2, 3, 4
STATUS_NAMES = {HOLE: "hole", DISC: "discretisation", INTERP: "interpolation",
                WALL: "wall", OUTER_BC: "outer-dirichlet"}


class OversetError(ValueError):
    """The overlap cannot be closed: some point needs a donor and has none."""


class GridQualityError(ValueError):
    """A grid whose cells fold, or whose Jacobian is not positive."""


# ===========================================================================
# outlines
# ===========================================================================


def signed_area(poly: np.ndarray) -> float:
    """Shoelace area: positive for a counter-clockwise outline."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def orient_clockwise(poly: np.ndarray) -> np.ndarray:
    """The same closed outline, running clockwise, without a repeated end point."""
    p = np.asarray(poly, dtype=float)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3:
        raise ValueError("an outline needs at least three distinct points")
    return p[::-1].copy() if signed_area(p) > 0 else p.copy()


def resample_closed(poly: np.ndarray, n: int) -> np.ndarray:
    """``n`` points spaced evenly by arc length round a closed outline."""
    p = np.asarray(poly, dtype=float)
    q = np.vstack([p, p[:1]])
    seg = np.hypot(np.diff(q[:, 0]), np.diff(q[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        raise ValueError("an outline of zero length")
    t = np.arange(n) * total / n
    return np.column_stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])])


def circle_outline(cx: float, cy: float, r: float, n: int = 4000) -> np.ndarray:
    th = -2.0 * math.pi * np.arange(n) / n
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


def ellipse_outline(cx: float, cy: float, a: float, b: float, angle_deg: float = 0.0,
                    n: int = 4000) -> np.ndarray:
    th = -2.0 * math.pi * np.arange(n) / n
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    x, y = a * np.cos(th), b * np.sin(th)
    return np.column_stack([cx + ca * x - sa * y, cy + sa * x + ca * y])


def _place(x: np.ndarray, y: np.ndarray, x_le: float, y_le: float,
           alpha_deg: float) -> np.ndarray:
    """Rotate a chord-aligned shape about its leading edge; POSITIVE alpha lifts
    the trailing edge, this project's convention (`ground_effect` line 211)."""
    ca, sa = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))
    return np.column_stack([x_le + ca * x - sa * y, y_le + sa * x + ca * y])


def rounded_plate_outline(x_le: float, y_le: float, chord: float, thickness: float,
                          alpha_deg: float, n: int = 4000) -> np.ndarray:
    """A plate of finite thickness with semicircular ends: what a RaceLab
    `FlexWing` plate has to become before a grid can be wrapped round it."""
    r = 0.5 * thickness
    k = max(8, n // 8)
    top = np.column_stack([np.linspace(0.0, chord, 3 * k), np.full(3 * k, r)])
    te = np.column_stack([chord + r * np.sin(np.linspace(0, math.pi, k)),
                          r * np.cos(np.linspace(0, math.pi, k))])
    bot = np.column_stack([np.linspace(chord, 0.0, 3 * k), np.full(3 * k, -r)])
    le = np.column_stack([-r * np.sin(np.linspace(0, math.pi, k)),
                          -r * np.cos(np.linspace(0, math.pi, k))])
    p = np.vstack([top[:-1], te[:-1], bot[:-1], le[:-1]])
    return _place(p[:, 0], p[:, 1], x_le, y_le, alpha_deg)


def naca4_outline(x_le: float, y_le: float, chord: float, t: float = 0.12,
                  alpha_deg: float = 0.0, n: int = 4000) -> np.ndarray:
    """A symmetric NACA four-digit section with the closed trailing edge
    (coefficient -0.1036): a SHARP convex corner, which a car's wing has."""
    m = n // 2
    beta = np.linspace(0.0, math.pi, m)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                    + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)
    upper = np.column_stack([xc, yt])
    lower = np.column_stack([xc[::-1], -yt[::-1]])
    p = np.vstack([upper[:-1], lower[:-1]]) * chord
    return _place(p[:, 0], p[:, 1], x_le, y_le, alpha_deg)


def dented_circle_outline(cx: float, cy: float, r: float, depth: float = 0.6,
                          width: float = 0.25, n: int = 4000) -> np.ndarray:
    """A circle with a deep, narrow dent: a concave region tighter than any
    reasonable grid thickness, which a generator must refuse."""
    th = -2.0 * math.pi * np.arange(n) / n
    wrapped = np.angle(np.exp(1j * th))
    rr = r * (1.0 - depth * np.exp(-(wrapped / width) ** 2))
    return np.column_stack([cx + rr * np.cos(th), cy + rr * np.sin(th)])


def points_in_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray,
                      max_elements: int = 4_000_000) -> np.ndarray:
    """Even-odd rule, vectorised over points AND edges.  A point on an edge may go
    either way.

    The parity of the crossings is summed over chunks of edges.  Each crossing
    is decided by the same floating-point expression, in the same order, as the
    one-edge-at-a-time loop this replaced, so the answer is bitwise that loop's
    -- which Tier 60's record was built with -- at a small fraction of its cost:
    at the car's size that loop was 31 of the 47 seconds an `Overset` took.
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    shape = px.shape
    px = px.ravel()
    py = py.ravel()
    count = np.zeros(px.size, dtype=np.int64)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    step = max(1, max_elements // max(1, px.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(0, len(x1), step):
            a, b = x1[k:k + step][None, :], y1[k:k + step][None, :]
            c, d = x2[k:k + step][None, :], y2[k:k + step][None, :]
            Y = py[:, None]
            crosses = (b > Y) != (d > Y)
            xc = a + (Y - b) * (c - a) / (d - b)
            count += np.count_nonzero(crosses & (px[:, None] < xc), axis=1)
    return (count % 2 == 1).reshape(shape)


def distance_to_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray,
                        chunk: int = 200_000) -> np.ndarray:
    """Distance from each point to the nearest EDGE of a closed outline."""
    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    ax, ay = poly[:, 0], poly[:, 1]
    bx, by = np.roll(ax, -1), np.roll(ay, -1)
    ex, ey = bx - ax, by - ay
    ll = np.maximum(ex * ex + ey * ey, 1e-300)
    out = np.empty(px.size)
    step = max(1, chunk // max(1, len(poly)))
    for k in range(0, px.size, step):
        X = px[k:k + step, None]
        Y = py[k:k + step, None]
        t = np.clip(((X - ax) * ex + (Y - ay) * ey) / ll, 0.0, 1.0)
        dx = X - (ax + t * ex)
        dy = Y - (ay + t * ey)
        out[k:k + step] = np.sqrt((dx * dx + dy * dy).min(axis=1))
    return out


# ===========================================================================
# grids
# ===========================================================================


@dataclass
class CartesianGrid:
    """The background: cell centres of a uniform lattice.

    ``x0, y0`` is the box's lower-left CORNER, so the first point is at
    ``(x0 + h/2, y0 + h/2)``.  The box faces carry Dirichlet data in this tier.
    """

    name: str
    nx: int
    ny: int
    h: float
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("a background needs at least three points each way")
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.h
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.h
        self.X, self.Y = np.meshgrid(xs, ys)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x0, self.x0 + self.nx * self.h, self.y0, self.y0 + self.ny * self.h)

    def to_index(self, px, py) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (i, j) of physical points: exact, the lattice is uniform."""
        return ((np.asarray(px) - self.x0) / self.h - 0.5,
                (np.asarray(py) - self.y0) / self.h - 0.5)

    def gradient(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Central differences inside, second-order one-sided on the box edges."""
        return (_d_open(p, 1) / self.h, _d_open(p, 0) / self.h)


def _d_periodic(a: np.ndarray, axis: int) -> np.ndarray:
    return 0.5 * (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis))


def _d_open(a: np.ndarray, axis: int) -> np.ndarray:
    """Central inside, (-3, 4, -1)/2 and (3, -4, 1)/2 at the two ends."""
    a = np.moveaxis(np.asarray(a, dtype=float), axis, 0)
    d = np.empty_like(a)
    d[1:-1] = 0.5 * (a[2:] - a[:-2])
    d[0] = 0.5 * (-3.0 * a[0] + 4.0 * a[1] - a[2])
    d[-1] = 0.5 * (3.0 * a[-1] - 4.0 * a[-2] + a[-3])
    return np.moveaxis(d, 0, axis)


@dataclass
class CurvilinearGrid:
    """A structured grid around one closed body.

    ``x`` and ``y`` are ``(nj, ni)``: ``i`` clockwise round the body and
    periodic, ``j`` from the wall outward.  ``wall`` is the row-0 condition the
    pressure system writes (``'neumann'`` or ``'dirichlet'``); ``outer`` is
    ``'interp'`` inside a composite and ``'dirichlet'`` for the single-grid
    control.  ``body`` is the closed outline the grid was built on, clockwise,
    and it is what cuts the background.
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    body: np.ndarray | None = None
    wall: str = "neumann"
    outer: str = "interp"
    meta: dict = field(default_factory=dict)
    #: ``False`` for a grid that does not close round a body -- Tier 62's road
    #: patch, whose row 0 lies along the road and whose other three sides are
    #: interpolation.  Its ``i`` derivatives are one-sided at the two ends.
    periodic: bool = True

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape or self.x.ndim != 2:
            raise ValueError("x and y must be matching (nj, ni) arrays")
        nj, ni = self.x.shape
        if nj < 4 or ni < 4:
            raise ValueError("a body grid needs at least four points each way")
        if self.wall not in ("neumann", "dirichlet"):
            raise ValueError("wall must be 'neumann' or 'dirichlet'")
        if self.outer not in ("interp", "dirichlet"):
            raise ValueError("outer must be 'interp' or 'dirichlet'")
        d_xi = _d_periodic if self.periodic else _d_open
        self.x_xi = d_xi(self.x, 1)
        self.y_xi = d_xi(self.y, 1)
        self.x_eta = _d_open(self.x, 0)
        self.y_eta = _d_open(self.y, 0)
        self.J = self.x_xi * self.y_eta - self.x_eta * self.y_xi
        if not np.all(self.J > 0):
            bad = int(np.sum(self.J <= 0))
            raise GridQualityError(
                f"{self.name}: the Jacobian is not positive at {bad} points -- "
                "the grid is left-handed or folds (i must run clockwise round "
                "the body and j outward)")
        self.a11 = (self.x_eta ** 2 + self.y_eta ** 2) / self.J
        self.a12 = -(self.x_xi * self.x_eta + self.y_xi * self.y_eta) / self.J
        self.a22 = (self.x_xi ** 2 + self.y_xi ** 2) / self.J
        if self.body is None:
            self.body = np.column_stack([self.x[0], self.y[0]])
        self._tree = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape

    @property
    def nj(self) -> int:
        return self.x.shape[0]

    @property
    def ni(self) -> int:
        return self.x.shape[1]

    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(np.column_stack([self.x.ravel(), self.y.ravel()]))
        return self._tree

    def wall_normal(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit normal at each wall point, pointing out of the fluid."""
        tx, ty = self.x_xi[0], self.y_xi[0]
        m = np.hypot(tx, ty)
        return ty / m, -tx / m

    def gradient(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``grad p`` at every point from the grid's own metrics."""
        p_xi = (_d_periodic if self.periodic else _d_open)(p, 1)
        p_eta = _d_open(p, 0)
        px = (self.y_eta * p_xi - self.y_xi * p_eta) / self.J
        py = (-self.x_eta * p_xi + self.x_xi * p_eta) / self.J
        return px, py

    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Conservative form: ``(1/J) [ d_xi(y_eta u - x_eta v) + d_eta(-y_xi u + x_xi v) ]``."""
        U = self.y_eta * u - self.x_eta * v
        V = -self.y_xi * u + self.x_xi * v
        return ((_d_periodic if self.periodic else _d_open)(U, 1) + _d_open(V, 0)) / self.J


def ogrid_annulus(name: str, cx: float, cy: float, r_in: float, r_out: float,
                  ni: int, nj: int, beta: float = 0.0, twist: float = 0.0,
                  **kw) -> CurvilinearGrid:
    """An O-grid round a circle, built analytically.

    ``beta > 0`` clusters the rows toward the wall, ``r = r_in + (r_out - r_in)
    (exp(beta s) - 1)/(exp(beta) - 1)``; ``twist`` rotates each row by
    ``twist * s`` radians, which leaves the Jacobian exactly as it was and makes
    the grid lines non-orthogonal -- the control that exercises ``a12``.
    """
    s = np.linspace(0.0, 1.0, nj)
    phi = s if beta == 0 else np.expm1(beta * s) / np.expm1(beta)
    r = r_in + (r_out - r_in) * phi
    theta0 = -2.0 * math.pi * np.arange(ni) / ni
    th = theta0[None, :] + twist * s[:, None]
    x = cx + r[:, None] * np.cos(th)
    y = cy + r[:, None] * np.sin(th)
    body = np.column_stack([x[0], y[0]])
    g = CurvilinearGrid(name, x, y, body=body, **kw)
    g.meta.update(kind="annulus", cx=cx, cy=cy, r_in=r_in, r_out=r_out,
                  beta=beta, twist=twist)
    return g


#: The dense resampling every generated grid is built from.  A power of two,
#: so any power-of-two ``ni`` up to it samples it exactly.
N_MASTER = 16384


#: How fast the normal smoothing widens with distance from the wall.
#:
#: **Measured, not chosen for looks (Tier 60, G4).**  At ``kappa = 1`` the
#: manufactured solution on a grid round an ellipse four times as long as it is
#: thick, 0.45 deep, converged at 2.02, 1.88, 1.67 on that grid alone (the
#: composite at 1.74), with the largest error on the tip, whose radius of
#: curvature is 0.0375.  With nothing else changed, the grid alone read 1.89,
#: 1.97, 1.99 at ``kappa = 0.25`` and 1.14, 1.56, 1.01 at ``kappa = 4``: a normal
#: field smoothed over a width comparable to a tight tip bends the grid lines
#: hard enough to hold the error off its asymptote.  A thinner grid (0.15 deep)
#: or a blunter tip (radius 0.2) at ``kappa = 1`` converged too, and a
#: 64000-point outline changed nothing.  Smoothing is
#: still what keeps a mildly concave region from folding, and how much of it
#: the car's shapes need is measured on them, not here.
KAPPA = 0.25


def _clustered_positions(M: np.ndarray, L: float, ni: int, cluster: float,
                         width: float) -> np.ndarray:
    """Fractional master indices of ``ni`` columns spaced by a density that rises
    with how fast the outline TURNS -- ``1 + cluster * width * rate``, the turning
    rate smoothed along the outline by a Gaussian of ``width`` -- so a tight
    corner gets columns in proportion to the angle it turns through.  The
    density is a property of the outline, never of ``ni``: refining samples the
    same distribution."""
    n = M.shape[0]
    ds = L / n
    ex = np.roll(M[:, 0], -1) - M[:, 0]
    ey = np.roll(M[:, 1], -1) - M[:, 1]
    heading = np.arctan2(ey, ex)
    turn = np.angle(np.exp(1j * (heading - np.roll(heading, 1))))
    k = 2.0 * math.pi * np.fft.fftfreq(n, d=ds)
    rate = np.real(np.fft.ifft(np.fft.fft(np.abs(turn) / ds) * np.exp(-0.5 * (width * k) ** 2)))
    density = 1.0 + cluster * width * np.maximum(rate, 0.0)
    F = np.concatenate([[0.0], np.cumsum(density)])
    targets = np.arange(ni) * (F[-1] / ni)
    return np.interp(targets, F, np.arange(n + 1, dtype=float))


def _room_along_normals(M: np.ndarray, nx0: np.ndarray, ny0: np.ndarray, body: np.ndarray,
                        L: float) -> np.ndarray:
    """How far each master point can go along its outward normal before meeting
    the outline again -- ``inf`` where the ray escapes.  A cavity's width, or the
    distance across a concave corner."""
    P = resample_closed(body, 4096)
    ax, ay = P[:, 0], P[:, 1]
    ex, ey = np.roll(ax, -1) - ax, np.roll(ay, -1) - ay
    tmin = 0.5 * L / 4096
    out = np.full(M.shape[0], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k0 in range(0, M.shape[0], 256):
            ox, oy = M[k0:k0 + 256, 0][:, None], M[k0:k0 + 256, 1][:, None]
            dx, dy = nx0[k0:k0 + 256][:, None], ny0[k0:k0 + 256][:, None]
            den = -dx * ey + dy * ex
            wx, wy = ax - ox, ay - oy
            t = (-wx * ey + wy * ex) / den
            u = (dx * wy - dy * wx) / den
            hit = (np.abs(den) > 1e-14) & (u >= 0.0) & (u <= 1.0) & (t > tmin)
            out[k0:k0 + 256] = np.where(hit, t, np.inf).min(axis=1)
    return out


def ogrid_from_outline(name: str, outline: np.ndarray, ni: int, nj: int,
                       thickness: float, beta: float = 2.0,
                       sigma_wall: float = 0.0, kappa: float = KAPPA,
                       cluster: float = 0.0, cluster_width: float | None = None,
                       room: float = 0.0, room_floor: float = 0.0,
                       **kw) -> CurvilinearGrid:
    """An O-grid offset outward from a closed outline, defined by the outline alone.

    ``x(s, d) = P(s) + d N(s, d)``: ``P`` is the outline by arc length ``s``,
    ``d`` runs from 0 at the wall to ``thickness`` with the rows clustered by
    ``beta``, and ``N(s, d)`` is the outward normal smoothed along the outline
    by a periodic Gaussian of width ``sigma_wall + kappa * d`` and renormalised.
    Smoothing grows with distance from the wall, where orthogonality matters
    least and folding in a concave region matters most; `KAPPA` says why the
    default is gentle.

    **The mapping does not depend on ``ni`` or ``nj``**: it is evaluated on a
    fixed `N_MASTER`-point resampling and sampled.  So refining a generated grid
    refines ONE mapping, which a convergence study needs.  (A first version
    marched layer by layer and smoothed each one; it shrank a circle by
    ``1.1e-2`` at 12 layers and its smoothing grew with the layer count, so a
    finer grid was a different grid.)

    **A folded grid is refused**, not repaired: every cell's signed area must be
    positive.  The outline is what cuts the background, and row 0 IS it.

    ``cluster > 0`` (Tier 62) spaces the columns by `_clustered_positions`
    instead of evenly by arc length, over ``cluster_width`` (default: the
    thickness).  Zero -- the default -- is Tier 60's even spacing, bit for bit.

    ``room > 0`` (Tier 62) lets the thickness VARY round the body: at each
    column it is at most ``room`` times the distance along the normal to the
    outline itself (`_room_along_normals`) and never under ``room_floor``, taken
    as a running minimum over one thickness and then smoothed over half of one,
    so a grid reaching into a cavity or a concave corner of its own body stops
    short of the far wall instead of folding.  Zero is a uniform thickness.
    """
    body = orient_clockwise(np.asarray(outline, dtype=float))
    M = resample_closed(body, N_MASTER)
    seg = np.hypot(np.roll(M[:, 0], -1) - M[:, 0], np.roll(M[:, 1], -1) - M[:, 1])
    L = float(seg.sum())
    tx = _d_periodic(M[:, 0], 0)
    ty = _d_periodic(M[:, 1], 0)
    m = np.hypot(tx, ty)
    if np.any(m == 0):
        raise GridQualityError(f"{name}: two coincident outline points")
    nx0, ny0 = -ty / m, tx / m                         # outward, for clockwise
    k = 2.0 * math.pi * np.fft.fftfreq(N_MASTER, d=L / N_MASTER)
    fx, fy = np.fft.fft(nx0), np.fft.fft(ny0)
    s = np.linspace(0.0, 1.0, nj)
    phi = s if beta == 0 else np.expm1(beta * s) / np.expm1(beta)
    dist = thickness * phi
    if cluster > 0:
        pos = _clustered_positions(M, L, ni, cluster,
                                   thickness if cluster_width is None else cluster_width)
        pos = np.minimum(pos, np.nextafter(float(N_MASTER), 0.0))
    else:
        pos = np.arange(ni) * (N_MASTER / ni)
    lo = np.floor(pos).astype(np.int64)
    fr = pos - lo
    hi = np.mod(lo + 1, N_MASTER)

    def sample(a):
        return (1.0 - fr) * a[lo] + fr * a[hi]

    X = np.empty((nj, ni))
    Y = np.empty((nj, ni))
    px0, py0 = sample(M[:, 0]), sample(M[:, 1])
    scale = None
    if room > 0:
        from scipy.ndimage import minimum_filter1d
        reach = _room_along_normals(M, nx0, ny0, body, L)
        local = np.clip(room * reach, room_floor, thickness)
        half = max(1, int(math.ceil(thickness / (L / N_MASTER))))
        local = minimum_filter1d(local, size=2 * half + 1, mode="wrap")
        local = np.real(np.fft.ifft(np.fft.fft(local) * np.exp(-0.5 * (0.5 * thickness * k) ** 2)))
        local = np.clip(local, room_floor, thickness)
        scale = sample(local / thickness)
    for j, d in enumerate(dist):
        sig = sigma_wall + kappa * d
        g = np.exp(-0.5 * (sig * k) ** 2)
        nx_ = np.real(np.fft.ifft(fx * g))
        ny_ = np.real(np.fft.ifft(fy * g))
        mm = np.hypot(nx_, ny_)
        if np.any(mm < 1e-6):
            raise GridQualityError(f"{name}: the smoothed normal vanishes {d:.4f} from "
                                   "the wall -- the smoothing is wider than the body")
        dd = d if scale is None else d * scale
        X[j] = px0 + dd * sample(nx_ / mm)
        Y[j] = py0 + dd * sample(ny_ / mm)
    q = _cell_areas(X, Y)
    if not np.all(q > 0):
        raise GridQualityError(
            f"{name}: {int(np.sum(q <= 0))} cells fold within {thickness} of the "
            "outline -- a concave region too tight for this thickness, or too few "
            "points round it")
    g = CurvilinearGrid(name, X, Y, body=body, **kw)
    g.meta.update(kind="offset", thickness=thickness, beta=beta,
                  sigma_wall=sigma_wall, kappa=kappa, perimeter=L)
    if cluster > 0:
        g.meta.update(cluster=cluster, cluster_width=cluster_width)
    if scale is not None:
        g.meta.update(room=room, room_floor=room_floor,
                      thickness_min=float(thickness * scale.min()))
    return g


def _cell_areas(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Signed area of every cell of a periodic-in-i grid, positive when right-handed."""
    x00, y00 = X[:-1], Y[:-1]
    x10, y10 = np.roll(X[:-1], -1, axis=1), np.roll(Y[:-1], -1, axis=1)
    x11, y11 = np.roll(X[1:], -1, axis=1), np.roll(Y[1:], -1, axis=1)
    x01, y01 = X[1:], Y[1:]
    return 0.5 * ((x00 * y10 - x10 * y00) + (x10 * y11 - x11 * y10)
                  + (x11 * y01 - x01 * y11) + (x01 * y00 - x00 * y01))


def grid_quality(g: CurvilinearGrid) -> dict[str, float]:
    """What a grid is like: folding, stretching, and how far from orthogonal."""
    area = _cell_areas(g.x, g.y)
    le_xi = np.hypot(np.roll(g.x, -1, axis=1) - g.x, np.roll(g.y, -1, axis=1) - g.y)
    le_eta = np.hypot(np.diff(g.x, axis=0), np.diff(g.y, axis=0))
    aspect = np.maximum(le_xi[:-1], le_eta) / np.maximum(np.minimum(le_xi[:-1], le_eta), 1e-300)
    cosang = ((g.x_xi * g.x_eta + g.y_xi * g.y_eta)
              / (np.hypot(g.x_xi, g.y_xi) * np.hypot(g.x_eta, g.y_eta)))
    skew = np.degrees(np.abs(np.arcsin(np.clip(cosang, -1.0, 1.0))))
    return {
        "min_cell_area": float(area.min()),
        "folded_cells": int(np.sum(area <= 0)),
        "min_jacobian": float(g.J.min()),
        "max_aspect_ratio": float(aspect.max()),
        "max_skew_deg": float(skew.max()),
        "max_wall_skew_deg": float(skew[0].max()),
        "mean_wall_spacing": float(le_eta[0].mean()),
    }


# ===========================================================================
# interpolation
# ===========================================================================


def lagrange_weights(t: np.ndarray, width: int) -> np.ndarray:
    """Lagrange basis on nodes ``0 .. width-1`` evaluated at ``t``: ``(n, width)``."""
    t = np.asarray(t, dtype=float)
    w = np.ones(t.shape + (width,))
    for k in range(width):
        for m in range(width):
            if m != k:
                w[..., k] *= (t - m) / (k - m)
    return w


def _lagrange_dweights(t: np.ndarray, width: int) -> np.ndarray:
    """d/dt of `lagrange_weights`."""
    t = np.asarray(t, dtype=float)
    d = np.zeros(t.shape + (width,))
    for k in range(width):
        for q in range(width):
            if q == k:
                continue
            term = np.full(t.shape, 1.0 / (k - q))
            for m in range(width):
                if m != k and m != q:
                    term = term * (t - m) / (k - m)
            d[..., k] += term
    return d


def _stencil_start(xi: np.ndarray, width: int) -> np.ndarray:
    """The first node of the ``width``-point stencil centred on ``xi``."""
    return np.floor(np.asarray(xi) - 0.5 * width + 1.0).astype(np.int64)


# ===========================================================================
# the composite
# ===========================================================================


class Overset:
    """A background and body grids, cut and joined.

    ``hole_margin`` is how far from each body the background is removed: every
    background point inside a body, or closer to it than this, is a HOLE.  The
    background points beside a hole become INTERPOLATION points, whose values
    come from the body grid; each body grid's outer ring does the same from the
    background.  **Interpolation is explicit**: a donor must be a point that
    carries its own equation -- a discretisation point, or a wall point on a
    body grid -- never another interpolation point, so the overlap must be wide
    enough for the two rings not to read each other.  When it is not, the
    points that cannot be served are ORPHANS and the constructor raises.

    ``background=None`` with one body grid whose ``outer`` is ``'dirichlet'``
    is the curvilinear single-grid control; ``components=()`` is the Cartesian
    one.  Both run through the same assembly as the composite.
    """

    def __init__(self, background: CartesianGrid | None,
                 components: Sequence[CurvilinearGrid] = (), *,
                 hole_margin: float = 0.0, width: int = 3) -> None:
        if width not in (2, 3, 4):
            raise ValueError("interpolation width must be 2, 3 or 4")
        self.bg = background
        self.comps = list(components)
        self.width = int(width)
        self.hole_margin = float(hole_margin)
        if self.bg is None and not self.comps:
            raise ValueError("an overset grid needs at least one grid")
        names = ([self.bg.name] if self.bg else []) + [c.name for c in self.comps]
        if len(set(names)) != len(names):
            raise ValueError("grid names must be unique")
        for c in self.comps:
            if c.outer == "interp" and self.bg is None:
                raise ValueError(f"{c.name}: an interpolated outer ring needs a background")
        self.grids = ([self.bg] if self.bg else []) + self.comps
        self.status: dict[str, np.ndarray] = {}
        self.donors: dict[str, dict[str, Any]] = {}
        t0 = time.perf_counter()
        self._classify()
        self._find_donors()
        self._number()
        self.build_s = time.perf_counter() - t0

    # -- classification -----------------------------------------------------

    def _classify(self) -> None:
        if self.bg is not None:
            S = np.full(self.bg.shape, DISC, dtype=np.int8)
            x0, x1, y0, y1 = self.bg.extent
            for c in self.comps:
                if (c.x.min() < x0 or c.x.max() > x1 or c.y.min() < y0 or c.y.max() > y1):
                    raise OversetError(
                        f"{c.name} leaves the background box; cutting a body "
                        "grid by the box is not built in this tier")
                body = c.body
                bx0, bx1 = body[:, 0].min(), body[:, 0].max()
                by0, by1 = body[:, 1].min(), body[:, 1].max()
                m = self.hole_margin
                near = ((self.bg.X >= bx0 - m) & (self.bg.X <= bx1 + m)
                        & (self.bg.Y >= by0 - m) & (self.bg.Y <= by1 + m))
                jj, ii = np.nonzero(near)
                px, py = self.bg.X[jj, ii], self.bg.Y[jj, ii]
                cut = points_in_polygon(px, py, body)
                if m > 0:
                    # a point already inside is cut whatever its distance, so
                    # only the rest need one -- the same answer, fewer edges
                    rest = np.nonzero(~cut)[0]
                    cut[rest] = distance_to_polygon(px[rest], py[rest], body) < m
                S[jj[cut], ii[cut]] = HOLE
            for a, c in enumerate(self.comps):
                cx_, cy_ = c.x.ravel(), c.y.ravel()
                for b, d in enumerate(self.comps):
                    if a == b:
                        continue
                    db = d.body
                    box = ((cx_ >= db[:, 0].min()) & (cx_ <= db[:, 0].max())
                           & (cy_ >= db[:, 1].min()) & (cy_ <= db[:, 1].max()))
                    if np.any(box) and np.any(points_in_polygon(cx_[box], cy_[box], db)):
                        raise OversetError(
                            f"{c.name} reaches inside {d.name}'s body; cutting "
                            "one body grid by another is not built in this tier")
            hole = S == HOLE
            nb = np.zeros_like(hole)
            nb[1:, :] |= hole[:-1, :]
            nb[:-1, :] |= hole[1:, :]
            nb[:, 1:] |= hole[:, :-1]
            nb[:, :-1] |= hole[:, 1:]
            S[(S == DISC) & nb] = INTERP
            if np.any(S[0, :] == HOLE) or np.any(S[-1, :] == HOLE) or \
                    np.any(S[:, 0] == HOLE) or np.any(S[:, -1] == HOLE):
                raise OversetError(f"a hole reaches the edge of {self.bg.name}")
            self.status[self.bg.name] = S
        for c in self.comps:
            S = np.full(c.shape, DISC, dtype=np.int8)
            S[0, :] = WALL
            S[-1, :] = INTERP if c.outer == "interp" else OUTER_BC
            self.status[c.name] = S

    # -- donors ---------------------------------------------------------------

    def _find_donors(self) -> None:
        orphans: list[str] = []
        w = self.width
        if self.bg is not None:
            S = self.status[self.bg.name]
            jj, ii = np.nonzero(S == INTERP)
            if jj.size:
                px, py = self.bg.X[jj, ii], self.bg.Y[jj, ii]
                # Try the body grids nearest first: the hole a point borders is
                # usually its nearest body's, but between two close bodies the
                # grid that CONTAINS the point may be the other one.
                dists = np.stack([distance_to_polygon(px, py, c.body) for c in self.comps])
                rank = np.argsort(dists, axis=0)
                open_ = np.ones(jj.size, dtype=bool)
                why_last = np.array([""] * jj.size, dtype=object)
                entries = []
                for r in range(len(self.comps)):
                    for k, c in enumerate(self.comps):
                        sel = np.nonzero(open_ & (rank[r] == k))[0]
                        if not sel.size:
                            continue
                        res = self._curvilinear_donors(c, px[sel], py[sel])
                        good = res["ok"]
                        why_last[sel[~good]] = [f"{c.name}: {w}" for w in res["why"][~good]]
                        if np.any(good):
                            g_sel = sel[good]
                            entries.append(dict(j=jj[g_sel], i=ii[g_sel], donor=c.name,
                                                flat=res["flat"][good], weights=res["weights"][good],
                                                xi=res["xi"][good], eta=res["eta"][good]))
                            open_[g_sel] = False
                for q in np.nonzero(open_)[0]:
                    orphans.append(f"{self.bg.name}[{jj[q]},{ii[q]}] at ({px[q]:.4f}, {py[q]:.4f}) "
                                   f"has no donor in any body grid; nearest said {why_last[q]}")
                self.donors[self.bg.name] = {"entries": entries}
        for c in self.comps:
            if c.outer != "interp":
                continue
            px, py = c.x[-1], c.y[-1]
            res = self._cartesian_donors(px, py)
            for q in np.nonzero(~res["ok"])[0]:
                orphans.append(f"{c.name}[outer,{q}] at ({px[q]:.4f}, {py[q]:.4f}) has no "
                               f"donor in {self.bg.name}: {res['why'][q]}")
            self.donors[c.name] = {"entries": [dict(
                j=np.full(c.ni, c.nj - 1), i=np.arange(c.ni), donor=self.bg.name,
                flat=res["flat"], weights=res["weights"], xi=res["xi"], eta=res["eta"])]}
        self.orphans = orphans
        if orphans:
            head = "; ".join(orphans[:4])
            raise OversetError(f"{len(orphans)} orphan point(s) -- the overlap is too "
                               f"narrow or a grid is too short: {head}")

    def _cartesian_donors(self, px, py) -> dict[str, np.ndarray]:
        g, w = self.bg, self.width
        S = self.status[g.name]
        xi, eta = g.to_index(px, py)
        n = xi.size
        flat = np.zeros((n, w * w), dtype=np.int64)
        weights = np.zeros((n, w * w))
        ok = np.zeros(n, dtype=bool)
        why = np.array([""] * n, dtype=object)
        i_base, j_base = _stencil_start(xi, w), _stencil_start(eta, w)
        for q in range(n):
            found = False
            for dj in (0, -1, 1):
                for di in (0, -1, 1):
                    i0, j0 = i_base[q] + di, j_base[q] + dj
                    ti, tj = xi[q] - i0, eta[q] - j0
                    if not (-1e-12 <= ti <= w - 1 + 1e-12 and -1e-12 <= tj <= w - 1 + 1e-12):
                        continue
                    if i0 < 0 or j0 < 0 or i0 + w > g.nx or j0 + w > g.ny:
                        continue
                    block = S[j0:j0 + w, i0:i0 + w]
                    if np.all(block == DISC):
                        wi = lagrange_weights(np.array([ti]), w)[0]
                        wj = lagrange_weights(np.array([tj]), w)[0]
                        jjs, iis = np.meshgrid(np.arange(j0, j0 + w), np.arange(i0, i0 + w), indexing="ij")
                        flat[q] = (jjs * g.nx + iis).ravel()
                        weights[q] = np.outer(wj, wi).ravel()
                        ok[q] = found = True
                        break
                if found:
                    break
            if not found:
                why[q] = "every stencil holding it touches a hole, an interpolation point or the box edge"
        return dict(flat=flat, weights=weights, ok=ok, why=why, xi=xi, eta=eta)

    def _curvilinear_donors(self, c: CurvilinearGrid, px, py) -> dict[str, np.ndarray]:
        """Locate points in a body grid and build their interpolation stencils.

        A bilinear inversion over the candidate cells round the nearest nodes
        finds the cell; Newton on the width-``w`` Lagrange map of the grid's own
        coordinates then finds ``(xi, eta)`` so that the interpolated
        coordinates ARE the point -- which makes the weights exact for any field
        linear in ``x`` and ``y``, curved grid or not.
        """
        w = self.width
        nj, ni = c.shape
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        n = px.size
        k = min(8, nj * ni)          # stretched cells: the nearest node need not be a corner
        _d, nn = c.tree().query(np.column_stack([px, py]), k=k)
        nn = nn.reshape(n, k)
        jn, inn = np.divmod(nn, ni)
        cj = (jn[:, :, None, None] - np.array([0, 1])[None, None, :, None]) \
            + np.zeros((1, 1, 1, 2), dtype=np.int64)
        ci = (inn[:, :, None, None] - np.array([0, 1])[None, None, None, :]) \
            + np.zeros((1, 1, 2, 1), dtype=np.int64)
        cj = cj.reshape(n, -1)
        ci = np.mod(ci.reshape(n, -1), ni)
        valid = (cj >= 0) & (cj <= nj - 2)
        j0 = np.clip(cj, 0, nj - 2)
        i1 = np.mod(ci + 1, ni)
        P00x, P00y = c.x[j0, ci], c.y[j0, ci]
        P10x, P10y = c.x[j0, i1], c.y[j0, i1]
        P01x, P01y = c.x[j0 + 1, ci], c.y[j0 + 1, ci]
        P11x, P11y = c.x[j0 + 1, i1], c.y[j0 + 1, i1]
        s = np.full(cj.shape, 0.5)
        t = np.full(cj.shape, 0.5)
        X = px[:, None]
        Y = py[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            for _ in range(20):
                bx = (1 - s) * (1 - t) * P00x + s * (1 - t) * P10x + (1 - s) * t * P01x + s * t * P11x
                by = (1 - s) * (1 - t) * P00y + s * (1 - t) * P10y + (1 - s) * t * P01y + s * t * P11y
                sx = (1 - t) * (P10x - P00x) + t * (P11x - P01x)
                sy = (1 - t) * (P10y - P00y) + t * (P11y - P01y)
                tx = (1 - s) * (P01x - P00x) + s * (P11x - P10x)
                ty = (1 - s) * (P01y - P00y) + s * (P11y - P10y)
                det = sx * ty - sy * tx
                rx, ry = bx - X, by - Y
                s = s - (ty * rx - tx * ry) / det
                t = t - (-sy * rx + sx * ry) / det
        tol = 1e-9
        inside = valid & np.isfinite(s) & np.isfinite(t) & \
            (s >= -tol) & (s <= 1 + tol) & (t >= -tol) & (t <= 1 + tol)
        has = inside.any(axis=1)
        pick = np.argmax(inside, axis=1)
        rows = np.arange(n)
        xi = np.where(has, ci[rows, pick] + s[rows, pick], np.nan)
        eta = np.where(has, j0[rows, pick] + t[rows, pick], np.nan)
        why = np.array([""] * n, dtype=object)
        why[~has] = "not inside any cell of the grid"
        # The stencil is fixed from the bilinear estimate, and the SAME stencil
        # defines the coordinate map Newton inverts and the weights returned:
        # that is what makes the weights reproduce x and y, hence any linear
        # field, to round-off.  Row nj-1 is the interpolation ring and is never
        # a donor, so the eta stencil stops at nj-2.
        ok = has.copy()
        sel = np.nonzero(ok)[0]
        i_start = np.zeros(n, dtype=np.int64)
        j_start = np.zeros(n, dtype=np.int64)
        i_start[sel] = _stencil_start(xi[sel], w)
        j_start[sel] = np.clip(_stencil_start(eta[sel], w), 0, nj - 1 - w)
        for _ in range(8):
            if not sel.size:
                break
            fx, fy, dxx, dxy, dyx, dyy = self._lagrange_map(c, xi[sel], eta[sel],
                                                            i_start[sel], j_start[sel])
            det = dxx * dyy - dxy * dyx
            rx, ry = fx - px[sel], fy - py[sel]
            with np.errstate(divide="ignore", invalid="ignore"):
                dxi = (dyy * rx - dxy * ry) / det
                deta = (-dyx * rx + dxx * ry) / det
            xi[sel] = xi[sel] - dxi
            eta[sel] = eta[sel] - deta
        flat = np.zeros((n, w * w), dtype=np.int64)
        weights = np.zeros((n, w * w))
        tol = 1e-6
        for q in sel:
            ti = xi[q] - i_start[q]
            tj = eta[q] - j_start[q]
            if not np.isfinite(ti) or not np.isfinite(tj) or \
                    not (-tol <= tj <= w - 1 + tol) or not (-tol <= ti <= w - 1 + tol):
                ok[q] = False
                why[q] = (f"eta = {eta[q]:.3f} of {nj - 1}: no {w}-point stencil of "
                          "discretisation points holds it without extrapolating -- "
                          "too close to the outer ring")
                continue
            wi = lagrange_weights(np.array([ti]), w)[0]
            wj = lagrange_weights(np.array([tj]), w)[0]
            jjs, iis = np.meshgrid(np.arange(j_start[q], j_start[q] + w),
                                   np.mod(np.arange(i_start[q], i_start[q] + w), ni),
                                   indexing="ij")
            flat[q] = (jjs * ni + iis).ravel()
            weights[q] = np.outer(wj, wi).ravel()
        return dict(flat=flat, weights=weights, ok=ok, why=why, xi=xi, eta=eta)

    def _lagrange_map(self, c: CurvilinearGrid, xi, eta, i0, j0):
        """The width-``w`` Lagrange interpolant of the coordinates on the stencil
        starting at ``(i0, j0)``, and its derivatives."""
        w = self.width
        nj, ni = c.shape
        ti = xi - i0
        tj = eta - j0
        Li, dLi = lagrange_weights(ti, w), _lagrange_dweights(ti, w)
        Lj, dLj = lagrange_weights(tj, w), _lagrange_dweights(tj, w)
        ii = np.mod(i0[:, None] + np.arange(w)[None, :], ni)
        jj = j0[:, None] + np.arange(w)[None, :]
        Xs = c.x[jj[:, :, None], ii[:, None, :]]
        Ys = c.y[jj[:, :, None], ii[:, None, :]]
        fx = np.einsum("nb,na,nba->n", Lj, Li, Xs)
        fy = np.einsum("nb,na,nba->n", Lj, Li, Ys)
        dxx = np.einsum("nb,na,nba->n", Lj, dLi, Xs)
        dxy = np.einsum("nb,na,nba->n", dLj, Li, Xs)
        dyx = np.einsum("nb,na,nba->n", Lj, dLi, Ys)
        dyy = np.einsum("nb,na,nba->n", dLj, Li, Ys)
        return fx, fy, dxx, dxy, dyx, dyy

    # -- numbering and fields ----------------------------------------------

    def _number(self) -> None:
        self.index: dict[str, np.ndarray] = {}
        self.offset: dict[str, int] = {}
        n = 0
        for g in self.grids:
            S = self.status[g.name]
            idx = np.full(S.shape, -1, dtype=np.int64)
            live = S != HOLE
            idx[live] = n + np.arange(int(live.sum()))
            self.index[g.name] = idx
            self.offset[g.name] = n
            n += int(live.sum())
        self.n_unknowns = n

    def scatter(self, vec: np.ndarray) -> dict[str, np.ndarray]:
        """A solution vector as one array per grid, NaN in the holes."""
        out = {}
        for g in self.grids:
            idx = self.index[g.name]
            a = np.full(idx.shape, np.nan)
            live = idx >= 0
            a[live] = vec[idx[live]]
            out[g.name] = a
        return out

    def report(self) -> dict[str, Any]:
        rep: dict[str, Any] = {"n_unknowns": self.n_unknowns, "width": self.width,
                               "hole_margin": self.hole_margin, "build_s": self.build_s,
                               "grids": {}}
        for g in self.grids:
            S = self.status[g.name]
            rep["grids"][g.name] = {STATUS_NAMES[k]: int(np.sum(S == k))
                                    for k in STATUS_NAMES if np.any(S == k)}
        return rep

    # -- the composite pressure system ---------------------------------------

    def poisson(self, rhs: Callable[[np.ndarray, np.ndarray], np.ndarray],
                box_value: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                wall_value: Callable[[CurvilinearGrid], np.ndarray] | None = None,
                outer_value: Callable[[CurvilinearGrid], np.ndarray] | None = None,
                ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Assemble ``A p = b`` for ``lap p = rhs`` on the composite grid.

        ``box_value(x, y)`` is the Dirichlet value on the background's faces,
        read at each face's midpoint and imposed through a ghost point;
        ``wall_value(grid)`` is each wall's Neumann or Dirichlet data (length
        ``ni``); ``outer_value(grid)`` the single-grid control's outer ring.
        """
        R: list[np.ndarray] = []
        C: list[np.ndarray] = []
        V: list[np.ndarray] = []
        b = np.zeros(self.n_unknowns)
        if self.bg is not None:
            self._assemble_background(rhs, box_value, R, C, V, b)
        for c in self.comps:
            self._assemble_component(c, rhs, wall_value, outer_value, R, C, V, b)
        for name, spec in self.donors.items():
            idx = self.index[name]
            for e in spec["entries"]:
                rows = idx[e["j"], e["i"]]
                donor_idx = self.index[e["donor"]].ravel()[e["flat"]]
                if np.any(donor_idx < 0):
                    raise OversetError(f"{name}: a donor in {e['donor']} is a hole")
                R += [rows, np.repeat(rows, e["flat"].shape[1])]
                C += [rows, donor_idx.ravel()]
                V += [np.ones(rows.size), -e["weights"].ravel()]
                b[rows] = 0.0
        A = sp.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                          shape=(self.n_unknowns, self.n_unknowns)).tocsr()
        A.sum_duplicates()
        return A, b

    def _assemble_background(self, rhs, box_value, R, C, V, b) -> None:
        g = self.bg
        S = self.status[g.name]
        idx = self.index[g.name]
        jj, ii = np.nonzero(S == DISC)
        rows = idx[jj, ii]
        ih2 = 1.0 / g.h ** 2
        centre = np.full(rows.size, -4.0 * ih2)
        b[rows] = rhs(g.X[jj, ii], g.Y[jj, ii])
        x0, x1, y0, y1 = g.extent
        for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nj_, ni_ = jj + dj, ii + di
            inbox = (nj_ >= 0) & (nj_ < g.ny) & (ni_ >= 0) & (ni_ < g.nx)
            q = np.nonzero(inbox)[0]
            nb = idx[nj_[q], ni_[q]]
            if np.any(nb < 0):
                raise OversetError(f"{g.name}: a discretisation point touches a hole")
            R.append(rows[q]); C.append(nb); V.append(np.full(q.size, ih2))
            e = np.nonzero(~inbox)[0]
            if e.size:
                if box_value is None:
                    raise ValueError("the background reaches its box and no box_value was given")
                if di == 1:
                    fx, fy = np.full(e.size, x1), g.Y[jj[e], ii[e]]
                elif di == -1:
                    fx, fy = np.full(e.size, x0), g.Y[jj[e], ii[e]]
                elif dj == 1:
                    fx, fy = g.X[jj[e], ii[e]], np.full(e.size, y1)
                else:
                    fx, fy = g.X[jj[e], ii[e]], np.full(e.size, y0)
                # ghost = (8 g - 6 p0 + p1) / 3, the quadratic through the face
                # value and the first two points in.  The linear ghost 2 g - p0
                # is enough for p at second order but leaves the GRADIENT first
                # order along the box -- measured, 1.00 -- and the flow solver
                # reads the gradient at its outlet.
                centre[e] -= 2.0 * ih2
                inward = idx[jj[e] - dj, ii[e] - di]
                R.append(rows[e]); C.append(inward); V.append(np.full(e.size, ih2 / 3.0))
                b[rows[e]] -= (8.0 / 3.0) * ih2 * box_value(fx, fy)
        R.append(rows); C.append(rows); V.append(centre)

    def laplacian_triplets(self, c: CurvilinearGrid
                           ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray],
                                      np.ndarray, np.ndarray]:
        """The nine-point conservative Laplacian at a body grid's discretisation
        points, as COO pieces in global numbering, and the points' ``(j, i)``.

        One copy of the stencil, read by the pressure system here and by the
        flow solver (`overset_ns`), so the two cannot drift apart.
        """
        S = self.status[c.name]
        idx = self.index[c.name]
        nj, ni = c.shape
        J, a11, a12, a22 = c.J, c.a11, c.a12, c.a22
        ip = lambda a: np.roll(a, -1, axis=1)             # noqa: E731
        im = lambda a: np.roll(a, 1, axis=1)              # noqa: E731
        A_p, A_m = 0.5 * (a11 + ip(a11)), 0.5 * (a11 + im(a11))
        B_p, B_m = 0.5 * (a12 + ip(a12)), 0.5 * (a12 + im(a12))
        C_p = np.zeros_like(a22); C_m = np.zeros_like(a22)
        D_p = np.zeros_like(a12); D_m = np.zeros_like(a12)
        C_p[:-1] = 0.5 * (a22[:-1] + a22[1:]); C_m[1:] = 0.5 * (a22[1:] + a22[:-1])
        D_p[:-1] = 0.5 * (a12[:-1] + a12[1:]); D_m[1:] = 0.5 * (a12[1:] + a12[:-1])
        jj, ii = np.nonzero(S == DISC)
        rows = idx[jj, ii]
        Jc = J[jj, ii]
        ap, am = A_p[jj, ii], A_m[jj, ii]
        bp, bm = B_p[jj, ii], B_m[jj, ii]
        cp, cm = C_p[jj, ii], C_m[jj, ii]
        dp, dm = D_p[jj, ii], D_m[jj, ii]
        coef = {
            (0, 0): -(ap + am + cp + cm),
            (0, 1): ap + 0.25 * (dp - dm),
            (0, -1): am - 0.25 * (dp - dm),
            (1, 0): cp + 0.25 * (bp - bm),
            (-1, 0): cm - 0.25 * (bp - bm),
            (1, 1): 0.25 * (bp + dp),
            (-1, 1): -0.25 * (bp + dm),
            (1, -1): -0.25 * (bm + dp),
            (-1, -1): 0.25 * (bm + dm),
        }
        R: list[np.ndarray] = []
        C: list[np.ndarray] = []
        V: list[np.ndarray] = []
        for (dj, di), val in coef.items():
            nb = idx[jj + dj, np.mod(ii + di, ni)]
            if np.any(nb < 0):                              # pragma: no cover
                raise OversetError(f"{c.name}: a stencil reaches a hole")
            R.append(rows); C.append(nb); V.append(val / Jc)
        return R, C, V, jj, ii

    def neumann_wall_triplets(self, c: CurvilinearGrid
                              ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """``dp/dn`` at a body grid's wall points, one-sided second order in ``eta``,
        with ``n`` pointing out of the fluid -- the rows' left-hand side."""
        idx = self.index[c.name]
        ni = c.ni
        J, a12, a22 = c.J, c.a12, c.a22
        wrow = idx[0, :]
        g12 = a12[0] / J[0]
        g22 = a22[0] / J[0]
        sq = np.sqrt(g22)
        i_all = np.arange(ni)
        R = [wrow] * 5
        C = [idx[0, np.mod(i_all + 1, ni)], idx[0, np.mod(i_all - 1, ni)],
             idx[0, i_all], idx[1, i_all], idx[2, i_all]]
        V = [-0.5 * g12 / sq, 0.5 * g12 / sq, 1.5 * sq, -2.0 * sq, 0.5 * sq]
        return R, C, V

    def _assemble_component(self, c, rhs, wall_value, outer_value, R, C, V, b) -> None:
        idx = self.index[c.name]
        nj, ni = c.shape
        LR, LC, LV, jj, ii = self.laplacian_triplets(c)
        R += LR; C += LC; V += LV
        b[idx[jj, ii]] = rhs(c.x[jj, ii], c.y[jj, ii])
        # the wall, row 0
        wrow = idx[0, :]
        if wall_value is None:
            raise ValueError(f"{c.name}: no wall data")
        gw = np.asarray(wall_value(c), dtype=float)
        if c.wall == "dirichlet":
            R.append(wrow); C.append(wrow); V.append(np.ones(ni))
            b[wrow] = gw
        else:
            NR, NC, NV = self.neumann_wall_triplets(c)
            R += NR; C += NC; V += NV
            b[wrow] = gw
        if c.outer == "dirichlet":
            orow = idx[nj - 1, :]
            if outer_value is None:
                raise ValueError(f"{c.name}: an outer Dirichlet ring needs outer_value")
            R.append(orow); C.append(orow); V.append(np.ones(ni))
            b[orow] = np.asarray(outer_value(c), dtype=float)


# ===========================================================================
# solving
# ===========================================================================


def solve_sparse(A: sp.spmatrix, b: np.ndarray, method: str = "splu",
                 rtol: float = 1e-11, x0: np.ndarray | None = None
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the composite system and report what it cost.

    ``'splu'`` is SuperLU, factor then solve, and reports both; ``'amg'`` is
    pyamg's smoothed aggregation preconditioning BiCGSTAB (pyamg is NOT a
    bundle dependency, so this path is for pricing, not shipping);
    ``'ilu-gmres'`` is SciPy's incomplete LU preconditioning GMRES.
    """
    info: dict[str, Any] = {"method": method, "n": int(A.shape[0]), "nnz": int(A.nnz)}
    if method == "splu":
        t0 = time.perf_counter()
        lu = spla.splu(sp.csc_matrix(A), permc_spec="COLAMD")
        info["factor_s"] = time.perf_counter() - t0
        info["fill_nnz"] = int(lu.L.nnz + lu.U.nnz)
        t0 = time.perf_counter()
        x = lu.solve(b)
        info["solve_s"] = time.perf_counter() - t0
        info["_lu"] = lu
    elif method == "amg":
        import pyamg
        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(A), symmetry="nonsymmetric")
        info["setup_s"] = time.perf_counter() - t0
        it = [0]
        t0 = time.perf_counter()
        x, flag = spla.bicgstab(A, b, x0=x0, M=ml.aspreconditioner(), rtol=rtol,
                                maxiter=500, callback=lambda _x: it.__setitem__(0, it[0] + 1))
        info.update(solve_s=time.perf_counter() - t0, iterations=it[0], flag=int(flag))
    elif method == "ilu-gmres":
        t0 = time.perf_counter()
        ilu = spla.spilu(sp.csc_matrix(A), drop_tol=1e-5, fill_factor=10)
        M = spla.LinearOperator(A.shape, ilu.solve)
        info["setup_s"] = time.perf_counter() - t0
        it = [0]
        t0 = time.perf_counter()
        x, flag = spla.gmres(A, b, x0=x0, M=M, rtol=rtol, restart=60, maxiter=50,
                             callback=lambda _r: it.__setitem__(0, it[0] + 1),
                             callback_type="pr_norm")
        info.update(solve_s=time.perf_counter() - t0, iterations=it[0], flag=int(flag))
    else:
        raise ValueError(f"unknown method {method!r}")
    info["relative_residual"] = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    return x, info


# ===========================================================================
# verification: a manufactured solution with three single-grid controls
# ===========================================================================

#: The manufactured pressure: a smooth wave over the whole box plus a Gaussian
#: centred on the body, so the solution varies where the grids overlap and at
#: the wall -- a field that is nearly flat there would verify nothing.
MS = {"a": 1.1, "b": 0.3, "c": 0.7, "d": 0.2, "amp": 0.3, "k": 2.0}


def _ms(x, y, cx, cy):
    a, bb, c, d, amp, k = (MS[q] for q in ("a", "b", "c", "d", "amp", "k"))
    wave = np.sin(a * x + bb) * np.cos(c * y + d)
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    bump = amp * np.exp(-k * r2)
    p = wave + bump
    px = a * np.cos(a * x + bb) * np.cos(c * y + d) - 2 * k * (x - cx) * bump
    py = -c * np.sin(a * x + bb) * np.sin(c * y + d) - 2 * k * (y - cy) * bump
    lap = -(a * a + c * c) * wave + (4 * k * k * r2 - 4 * k) * bump
    return p, px, py, lap


#: The verification geometry, in tiling units: a 4 x 3 box with a circle of
#: radius 0.3 at its centre and a body grid 0.45 thick round it.
MS_BOX = (4.0, 3.0)
MS_BODY = (2.0, 1.5, 0.3)
MS_THICKNESS = 0.45
MS_HOLE_MARGIN = 0.15


#: The generated-grid verification body: an ellipse four times as long as it is
#: thick, turned ten degrees, in the same box and at the same centre.
MS_ELLIPSE = (0.6, 0.15, 10.0)


def manufactured_poisson(case: str, level: int, *, width: int = 3,
                         twist: float = 0.0, beta: float = 1.0,
                         method: str = "splu", body: str = "circle",
                         ellipse: tuple[float, float, float] = MS_ELLIPSE,
                         thickness: float = MS_THICKNESS, kappa: float = 1.0,
                         n_outline: int = 4000) -> dict[str, Any]:
    """Solve the manufactured problem at ``h = 1/(16 2^(level-1))`` and measure it.

    ``case`` is ``'box'`` (the Cartesian control, no body), ``'annulus'`` (the
    body grid alone, Dirichlet on its outer ring: the curvilinear control) or
    ``'composite'`` (both, joined by interpolation).  ``body`` is ``'circle'``
    (the analytic O-grid, `ogrid_annulus`) or ``'ellipse'`` (a GENERATED grid,
    `ogrid_from_outline`, with ``ellipse = (a, b, angle)``, ``thickness``,
    ``kappa`` and an ``n_outline``-point input polygon).  Errors are the max and
    rms of ``p - p_exact`` over every point that carries an equation, per grid,
    with where the max sits, and of the discrete gradient against the exact
    one over discretisation points.

    ``kappa`` defaults to 1.0 here, NOT to `KAPPA`: 1.0 is what Tier 60's
    registered G4 ran at, and this default keeps that record reproducible.
    """
    if case not in ("box", "annulus", "composite"):
        raise ValueError("case must be 'box', 'annulus' or 'composite'")
    if body not in ("circle", "ellipse"):
        raise ValueError("body must be 'circle' or 'ellipse'")
    n16 = 2 ** (level - 1)
    h = 1.0 / (16 * n16)
    Lx, Ly = MS_BOX
    cx, cy, R = MS_BODY
    bg = None
    comps: list[CurvilinearGrid] = []
    if case in ("box", "composite"):
        bg = CartesianGrid("bg", int(round(Lx / h)), int(round(Ly / h)), h)
    outer = "interp" if case == "composite" else "dirichlet"
    if case in ("annulus", "composite") and body == "circle":
        comps.append(ogrid_annulus("body", cx, cy, R, R + thickness, 48 * n16, 6 * n16 + 1,
                                   beta=beta, twist=twist, outer=outer))
    elif case in ("annulus", "composite"):
        a, bb, ang = ellipse
        comps.append(ogrid_from_outline("body", ellipse_outline(cx, cy, a, bb, ang, n=n_outline),
                                        96 * n16, 6 * n16 + 1, thickness, beta=beta,
                                        kappa=kappa, outer=outer))
    ov = Overset(bg, comps, hole_margin=MS_HOLE_MARGIN if case == "composite" else 0.0,
                 width=width)

    def rhs(x, y):
        return _ms(x, y, cx, cy)[3]

    def box_value(x, y):
        return _ms(x, y, cx, cy)[0]

    def wall_value(g):
        _p, gx, gy, _l = _ms(g.x[0], g.y[0], cx, cy)
        nx_, ny_ = g.wall_normal()
        return nx_ * gx + ny_ * gy

    def outer_value(g):
        return _ms(g.x[-1], g.y[-1], cx, cy)[0]

    t0 = time.perf_counter()
    A, b = ov.poisson(rhs, box_value, wall_value, outer_value)
    t_asm = time.perf_counter() - t0
    x, info = solve_sparse(A, b, method)
    info.pop("_lu", None)
    fields = ov.scatter(x)
    out: dict[str, Any] = {"case": case, "level": level, "h": h, "width": width,
                           "twist": twist, "beta": beta, "n_unknowns": ov.n_unknowns,
                           "assemble_s": t_asm, "solve": info, "overset": ov.report(),
                           "grids": {}}
    for g in ov.grids:
        S = ov.status[g.name]
        if isinstance(g, CartesianGrid):
            X, Y = g.X, g.Y
        else:
            X, Y = g.x, g.y
        pe, gxe, gye, _l = _ms(X, Y, cx, cy)
        p = fields[g.name]
        live = S != HOLE
        full = np.where(live, np.abs(p - pe), -1.0)
        err = full[live]
        jm, im_ = np.unravel_index(int(np.argmax(full)), full.shape)
        rec = {"max_err": float(err.max()), "rms_err": float(np.sqrt(np.mean(err ** 2))),
               "max_at": {"j": int(jm), "i": int(im_), "x": float(X[jm, im_]),
                          "y": float(Y[jm, im_]), "status": STATUS_NAMES[int(S[jm, im_])]}}
        disc = S == DISC
        gx, gy = g.gradient(np.where(live, p, 0.0))
        # a gradient is only meaningful where its whole stencil is live
        ok = disc.copy()
        if isinstance(g, CartesianGrid):
            for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                sh = np.roll(np.roll(live, -dj, axis=0), -di, axis=1)
                ok &= sh
        gerr = np.hypot(gx - gxe, gy - gye)[ok]
        rec["grad_max_err"] = float(gerr.max()) if gerr.size else None
        rec["grad_rms_err"] = float(np.sqrt(np.mean(gerr ** 2))) if gerr.size else None
        out["grids"][g.name] = rec
    return out
