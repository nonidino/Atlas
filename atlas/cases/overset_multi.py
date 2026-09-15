"""Several bodies and a road on body-fitted overset grids.

PoC 3, Tier 62.  [[poc3-racelab-car-solids]].

Tier 60's `Overset` joins one kind of overlap: body grids that sit inside a
background and cut holes in it -- never in each other, never through the box.
The car needs three more things, and this module adds them in a subclass, so
the Tier 60 class keeps joining the grids it joined, bit for bit:

* **grids cut by OTHER bodies.**  A wheel's grid crosses the floor's; a wing
  element's grid crosses its neighbour's.
* **grids cut by the ROAD.**  A wheel two percent of its radius above the road
  puts the bottom of its own grid below the road.
* **road patches** (`road_patch`).  A thin grid lying along the road under a
  wheel, resolving the gap between tyre and road, which the background -- a
  sixty-fourth of a unit per cell on the car's lattice -- cannot.  Its row 0 IS
  the road, and its other three sides are interpolation.

How the grids are joined
------------------------

1. **Boundary rows.**  A body grid's row 0 is WALL and its outer row
   INTERPOLATION; a road patch's row 0 is WALL and its two ends and its top row
   INTERPOLATION.
2. **Holes.**  A background point inside any body, or closer to one than
   ``hole_margin``; a grid's point inside any OTHER body (or closer than
   ``component_margin``), or below the road.  **A wall point in a hole is
   refused** -- two bodies overlap.  A background EDGE cell whose first or
   second cell inward is a hole becomes a hole too, because an edge cell
   carries the box's condition through exactly those two cells; this is
   allowed on the road only, and a hole reaching the inlet, the outlet or the
   top is refused.
3. **Fringe.**  A background discretisation point with a hole among its four
   neighbours, or a body grid's with a hole in its 3 x 3 block (the curvilinear
   Laplacian is nine-point), becomes an interpolation point.  **A wall point
   whose one-sided stencil reaches a hole is refused** -- a gap narrower than
   two grid rows, which no interpolation can close.
4. **Donors**, explicit as in Tier 60: every node of a donor stencil carries
   its own equation.  Candidates are tried in order --

   * a point next to a hole: the grid of whatever cut the hole first (a road
     patch, for the road), then the other body grids nearest first, then the
     background;
   * a grid's own boundary: the background first, then the body grids nearest
     first;
   * the background: the body grids nearest first --

   and a stencil that touches a hole or an interpolation point is shifted one
   node each way before a candidate is given up.  A point no candidate can
   serve is an ORPHAN, and the constructor raises, as Tier 60's does.

With one body grid and no road the order above IS Tier 60's, and so are the
stencils: the first stencil tried is Tier 60's, and it is the one kept
whenever every node of it is usable (`test_tier62_car_solids` asserts the
weights against `Overset` on Tier 60's geometry).
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .overset import (CartesianGrid, CurvilinearGrid, DISC, HOLE, INTERP, OUTER_BC,
                      Overset, OversetError, STATUS_NAMES, WALL, _ms, _stencil_start,
                      distance_to_polygon, lagrange_weights, naca4_outline,
                      ogrid_annulus, ogrid_from_outline, points_in_polygon,
                      rounded_plate_outline, solve_sparse)

__all__ = [
    "MultiOverset", "road_patch", "aerofoil_outline", "ROAD", "SHIFTS", "WALL_REACH",
    "MULTI_BOX", "MULTI_WHEEL", "MULTI_PANEL", "MULTI_AEROFOIL", "MULTI_HOLE_MARGIN",
    "MULTI_BUMP", "MULTI_GRID", "multi_geometry", "manufactured_poisson_multi", "mms_flow_multi",
]

#: What cut a hole, when it was not a body grid: the road.
ROAD = -2

#: The order stencils are shifted in when the first one touches a hole or an
#: interpolation point: unshifted first, then one node along each axis, then
#: diagonally.
SHIFTS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))

#: How far past a wall row, in rows, a donor stencil starting at that row may
#: reach -- for points between a wall's polygon and its interpolant's curve.
WALL_REACH = 0.25


def road_patch(name: str, x0: float, x1: float, height: float, ni: int, nj: int, *,
               road: float = 0.0, beta: float = 0.0, wall: str = "neumann") -> CurvilinearGrid:
    """A grid along the road: ``x`` uniform from ``x0`` to ``x1``, ``y`` from the
    road to ``road + height`` with the rows clustered toward the road by
    ``beta`` exactly as `ogrid_annulus` clusters them toward a wall.

    Row 0 is the road and is the patch's wall; its normal points down, out of
    the fluid, the convention every wall here keeps.  The patch cuts nothing:
    what it resolves is the road, and the road is cut by `MultiOverset` itself.
    """
    s = np.linspace(0.0, 1.0, nj)
    phi = s if beta == 0 else np.expm1(beta * s) / np.expm1(beta)
    X, Y = np.meshgrid(np.linspace(x0, x1, ni), road + height * phi)
    g = CurvilinearGrid(name, X, Y, body=None, wall=wall, outer="interp", periodic=False)
    g.meta.update(kind="road-patch", x0=x0, x1=x1, height=height, beta=beta, road=road)
    return g


def aerofoil_outline(x_le: float, y_le: float, chord: float, t: float = 0.12,
                     alpha_deg: float = 0.0, te_thickness: float = 0.01,
                     n: int = 4000) -> np.ndarray:
    """A NACA four-digit thickness form with a ROUNDED trailing edge.

    Tier 60's `naca4_outline` closes the trailing edge to a point, and behind a
    point an offset grid's columns fan out: Tier 62 measured the manufactured
    pressure converging at order 1.1 in the fan, on both the aerofoil's grid and
    the background it feeds, while every smooth body converged at 2.1-2.3.
    Here the half-thickness gains ``te_thickness / 2`` times ``x/c`` -- nothing at
    the leading edge, ``te_thickness`` (a fraction of the chord) at the trailing
    edge -- and a semicircle closes it.
    """
    m = max(8, n // 2)
    beta = np.linspace(0.0, math.pi, m)
    xc = 0.5 * (1.0 - np.cos(beta))
    r = 0.5 * te_thickness
    yt = 5.0 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2
                    + 0.2843 * xc ** 3 - 0.1036 * xc ** 4) + r * xc
    upper = np.column_stack([xc, yt])
    phi = np.linspace(0.5 * math.pi, -0.5 * math.pi, max(8, n // 16))
    cap = np.column_stack([1.0 + r * np.cos(phi), r * np.sin(phi)])
    lower = np.column_stack([xc[::-1], -yt[::-1]])
    p = np.vstack([upper[:-1], cap[:-1], lower[:-1]]) * chord
    ca, sa = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))
    return np.column_stack([x_le + ca * p[:, 0] - sa * p[:, 1], y_le + sa * p[:, 0] + ca * p[:, 1]])


def _dilate(mask: np.ndarray, periodic: bool, diagonal: bool) -> np.ndarray:
    """Points with a True neighbour: four neighbours, or eight with ``diagonal``;
    ``i`` wraps when ``periodic``."""
    out = np.zeros_like(mask)
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    nj, ni = mask.shape
    for dj, di in steps:
        src = mask
        if periodic:
            src = np.roll(src, di, axis=1)
        else:
            sh = np.zeros_like(src)
            if di > 0:
                sh[:, di:] = src[:, :-di]
            elif di < 0:
                sh[:, :di] = src[:, -di:]
            else:
                sh = src
            src = sh
        sh = np.zeros_like(src)
        if dj > 0:
            sh[dj:, :] = src[:-dj, :]
        elif dj < 0:
            sh[:dj, :] = src[-dj:, :]
        else:
            sh = src
        out |= sh
    return out


class MultiOverset(Overset):
    """A background, several body grids, a road and road patches, cut and joined.

    ``road`` is the height of the road, the background's lower face; every grid
    point below it is a hole.  ``component_margin`` is how far from another
    body a body grid's points are removed (zero: only the points inside it).
    Everything else is `Overset`'s: ``hole_margin`` for the background,
    ``width`` for the interpolation, orphans refused.
    """

    def __init__(self, background: CartesianGrid, components: Sequence[CurvilinearGrid] = (), *,
                 hole_margin: float = 0.0, width: int = 3, road: float | None = None,
                 component_margin: float = 0.0) -> None:
        if background is None:
            raise ValueError("a MultiOverset needs a background")
        self.road = None if road is None else float(road)
        self.component_margin = float(component_margin)
        super().__init__(background, components, hole_margin=hole_margin, width=width)

    @staticmethod
    def _xy(g) -> tuple[np.ndarray, np.ndarray]:
        return (g.X, g.Y) if isinstance(g, CartesianGrid) else (g.x, g.y)

    # -- classification -----------------------------------------------------

    def _cut(self, px: np.ndarray, py: np.ndarray, body: np.ndarray, margin: float) -> np.ndarray:
        inside = points_in_polygon(px, py, body)
        if margin > 0:
            rest = np.nonzero(~inside)[0]
            if rest.size:
                inside[rest] = distance_to_polygon(px[rest], py[rest], body) < margin
        return inside

    def _classify(self) -> None:
        bg, road = self.bg, self.road
        self.bodies = [c for c in self.comps if c.periodic]
        self.patches = [c for c in self.comps if not c.periodic]
        if self.patches and road is None:
            raise OversetError("a road patch needs a road")
        for p in self.patches:
            if np.max(np.abs(p.y[0] - road)) > 1e-12:
                raise OversetError(f"{p.name}: a road patch's row 0 must lie on the road")
        x0, x1, y0, y1 = bg.extent
        for c in self.comps:
            if c.x.min() < x0 or c.x.max() > x1 or c.y.max() > y1:
                raise OversetError(f"{c.name} leaves the background box through its inlet, "
                                   "outlet or top, which nothing cuts")
            if c.y.min() < y0 and road is None:
                raise OversetError(f"{c.name} leaves the background box through its floor, "
                                   "and there is no road to cut it")
        self.cut_by: dict[str, np.ndarray] = {}

        # -- the background
        S = np.full(bg.shape, DISC, dtype=np.int8)
        cut = np.full(bg.shape, -1, dtype=np.int64)
        m = self.hole_margin
        for k, c in enumerate(self.comps):
            if not c.periodic:
                continue
            body = c.body
            near = ((bg.X >= body[:, 0].min() - m) & (bg.X <= body[:, 0].max() + m)
                    & (bg.Y >= body[:, 1].min() - m) & (bg.Y <= body[:, 1].max() + m))
            jj, ii = np.nonzero(near)
            if not jj.size:
                continue
            inside = self._cut(bg.X[jj, ii], bg.Y[jj, ii], body, m)
            fresh = inside & (cut[jj, ii] < 0)
            cut[jj[fresh], ii[fresh]] = k
            S[jj[inside], ii[inside]] = HOLE
        if road is not None:
            below = bg.Y < road - 1e-12
            cut[below & (cut < 0)] = ROAD
            S[below] = HOLE
        ny, nx = bg.shape
        JJ, II = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        on_edge = (II == 0) | (II == nx - 1) | (JJ == 0) | (JJ == ny - 1)
        j1, i1 = np.clip(JJ, 1, ny - 2), np.clip(II, 1, nx - 2)
        j2 = np.where(JJ == 0, 2, np.where(JJ == ny - 1, ny - 3, JJ))
        i2 = np.where(II == 0, 2, np.where(II == nx - 1, nx - 3, II))
        lost = on_edge & (S != HOLE) & ((S[j1, i1] == HOLE) | (S[j2, i2] == HOLE))
        heir = np.where(cut[j1, i1] != -1, cut[j1, i1], cut[j2, i2])
        cut[lost] = heir[lost]
        S[lost] = HOLE
        road_row = (JJ == 0) & (road is not None)
        if np.any(on_edge & ~road_row & (S == HOLE)):
            raise OversetError(f"a hole reaches the inlet, outlet or top of {bg.name}")
        S[(S == DISC) & _dilate(S == HOLE, periodic=False, diagonal=False)] = INTERP
        if np.any(on_edge & ~road_row & (S == INTERP)):
            raise OversetError(f"an interpolation point reaches the inlet, outlet or top of {bg.name}")
        self.status[bg.name] = S
        self.cut_by[bg.name] = cut

        # -- the body grids and the patches
        mc = self.component_margin
        for a, c in enumerate(self.comps):
            nj, ni = c.shape
            S = np.full(c.shape, DISC, dtype=np.int8)
            S[0, :] = WALL
            if c.periodic:
                S[-1, :] = INTERP if c.outer == "interp" else OUTER_BC
                wall_cols = np.arange(ni)
            else:
                S[:, 0] = INTERP
                S[:, -1] = INTERP
                S[-1, :] = INTERP
                wall_cols = np.arange(1, ni - 1)
            cut = np.full(c.shape, -1, dtype=np.int64)
            for b, d in enumerate(self.comps):
                if b == a or not d.periodic:
                    continue
                db = d.body
                box = ((c.x >= db[:, 0].min() - mc) & (c.x <= db[:, 0].max() + mc)
                       & (c.y >= db[:, 1].min() - mc) & (c.y <= db[:, 1].max() + mc))
                jj, ii = np.nonzero(box)
                if not jj.size:
                    continue
                inside = self._cut(c.x[jj, ii], c.y[jj, ii], db, mc)
                fresh = inside & (cut[jj, ii] < 0)
                cut[jj[fresh], ii[fresh]] = b
                S[jj[inside], ii[inside]] = HOLE
            if road is not None:
                below = c.y < road - 1e-12
                cut[below & (cut < 0)] = ROAD
                S[below] = HOLE
            hole = S == HOLE
            if np.any(hole[0, wall_cols]):
                q = int(wall_cols[np.argmax(hole[0, wall_cols])])
                who = cut[0, q]
                raise OversetError(
                    f"{c.name}: its wall at ({c.x[0, q]:.4f}, {c.y[0, q]:.4f}) lies "
                    f"{'below the road' if who == ROAD else 'inside ' + self.comps[who].name} "
                    "-- two bodies overlap")
            reach = hole[1, wall_cols] | hole[2, wall_cols] | \
                hole[0, np.mod(wall_cols + 1, ni)] | hole[0, np.mod(wall_cols - 1, ni)]
            if np.any(reach):
                q = int(wall_cols[np.argmax(reach)])
                raise OversetError(
                    f"{c.name}: the wall at ({c.x[0, q]:.4f}, {c.y[0, q]:.4f}) is closer than "
                    "two grid rows to another body or the road -- a gap no interpolation "
                    "can close")
            S[(S == DISC) & _dilate(hole, periodic=c.periodic, diagonal=True)] = INTERP
            self.status[c.name] = S
            self.cut_by[c.name] = cut

    # -- donors ---------------------------------------------------------------

    def _find_donors(self) -> None:
        orphans: list[str] = []
        trees = {c.name: cKDTree(np.column_stack([c.x[0], c.y[0]]))
                 for c in self.comps if c.periodic}
        extent = {c.name: (c.x.min(), c.x.max(), c.y.min(), c.y.max()) for c in self.comps}

        def distance(c, px, py):
            e = extent[c.name]
            within = (px >= e[0]) & (px <= e[1]) & (py >= e[2]) & (py <= e[3])
            d = np.full(px.shape, np.inf)
            if np.any(within):
                if c.periodic:
                    d[within] = trees[c.name].query(np.column_stack([px[within], py[within]]))[0]
                else:
                    d[within] = np.abs(py[within] - self.road)
            return d

        comp_pos = {c.name: k for k, c in enumerate(self.comps)}
        for g in self.grids:
            S = self.status[g.name]
            jj, ii = np.nonzero(S == INTERP)
            entries: list[dict[str, Any]] = []
            self.donors[g.name] = {"entries": entries}
            if not jj.size:
                continue
            gx, gy = self._xy(g)
            px, py = gx[jj, ii], gy[jj, ii]
            n = jj.size
            if g is self.bg:
                cands: list[Any] = list(self.comps)
                key = np.stack([distance(c, px, py) for c in cands])
            else:
                cands = [c for c in self.comps if c is not g] + [self.bg]
                cut = self.cut_by[g.name]
                nj_, ni_ = g.shape
                seen = []
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        qj, qi = jj + dj, ii + di
                        if g.periodic:
                            qi = np.mod(qi, ni_)
                            inr = (qj >= 0) & (qj < nj_)
                        else:
                            inr = (qj >= 0) & (qj < nj_) & (qi >= 0) & (qi < ni_)
                        v = np.full(n, -1, dtype=np.int64)
                        v[inr] = cut[qj[inr], qi[inr]]
                        seen.append(v)
                seen = np.stack(seen)                          # (9, n)
                fringe = np.any(seen != -1, axis=0)
                key = np.empty((len(cands), n))
                for k, c in enumerate(cands):
                    if c is self.bg:
                        key[k] = np.where(fringe, 2e6, 0.0)
                        continue
                    d = distance(c, px, py)
                    if c.periodic:
                        cutter = np.any(seen == comp_pos[c.name], axis=0)
                    else:
                        cutter = np.any(seen == ROAD, axis=0)
                    klass = np.where(fringe & cutter, 0.0, 1e6)
                    key[k] = np.where(np.isfinite(d), klass + d, np.inf)
            order = np.argsort(key, axis=0, kind="stable")
            ranked = np.take_along_axis(key, order, axis=0)
            open_ = np.ones(n, dtype=bool)
            why_last = np.array(["no candidate grid contains it"] * n, dtype=object)
            for r in range(len(cands)):
                for k, cand in enumerate(cands):
                    sel = np.nonzero(open_ & (order[r] == k) & np.isfinite(ranked[r]))[0]
                    if not sel.size:
                        continue
                    if isinstance(cand, CartesianGrid):
                        res = self._cartesian_donors(px[sel], py[sel])
                    else:
                        res = self._curvilinear_donors(cand, px[sel], py[sel])
                    good = res["ok"]
                    why_last[sel[~good]] = [f"{cand.name}: {m}" for m in res["why"][~good]]
                    if np.any(good):
                        gs = sel[good]
                        entries.append(dict(j=jj[gs], i=ii[gs], donor=cand.name,
                                            flat=res["flat"][good], weights=res["weights"][good],
                                            xi=res["xi"][good], eta=res["eta"][good]))
                        open_[gs] = False
            for q in np.nonzero(open_)[0]:
                orphans.append(f"{g.name}[{jj[q]},{ii[q]}] at ({px[q]:.4f}, {py[q]:.4f}): "
                               f"{why_last[q]}")
        self.orphans = orphans
        if orphans:
            head = "; ".join(orphans[:4])
            raise OversetError(f"{len(orphans)} orphan point(s): {head}")

    def _curvilinear_donors(self, c: CurvilinearGrid, px, py) -> dict[str, np.ndarray]:
        """Tier 60's location and Newton, on a grid that may have holes, fringe
        points and (for a road patch) two open ends: every node of the stencil
        kept must carry its own equation, and a stencil that does not is shifted
        by `SHIFTS` -- Newton re-run on each, so the weights reproduce linear
        fields on whichever stencil is kept."""
        w = self.width
        nj, ni = c.shape
        usable = np.isin(self.status[c.name], (DISC, WALL))
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        n = px.size
        k = min(8, nj * ni)
        _d, nn = c.tree().query(np.column_stack([px, py]), k=k)
        nn = nn.reshape(n, k)
        jn, inn = np.divmod(nn, ni)
        cj = (jn[:, :, None, None] - np.array([0, 1])[None, None, :, None]) \
            + np.zeros((1, 1, 1, 2), dtype=np.int64)
        ci = (inn[:, :, None, None] - np.array([0, 1])[None, None, None, :]) \
            + np.zeros((1, 1, 2, 1), dtype=np.int64)
        cj = cj.reshape(n, -1)
        ci = ci.reshape(n, -1)
        valid = (cj >= 0) & (cj <= nj - 2)
        if c.periodic:
            ci = np.mod(ci, ni)
        else:
            valid &= (ci >= 0) & (ci <= ni - 2)
            ci = np.clip(ci, 0, ni - 2)
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
        xi0 = np.where(has, ci[rows, pick] + s[rows, pick], np.nan)
        eta0 = np.where(has, j0[rows, pick] + t[rows, pick], np.nan)
        why = np.array([""] * n, dtype=object)
        why[~has] = "not inside any cell of the grid"
        ok = np.zeros(n, dtype=bool)
        flat = np.zeros((n, w * w), dtype=np.int64)
        weights = np.zeros((n, w * w))
        xi = np.full(n, np.nan)
        eta = np.full(n, np.nan)
        todo = np.nonzero(has)[0]
        base_i = np.zeros(n, dtype=np.int64)
        base_j = np.zeros(n, dtype=np.int64)
        base_i[todo] = _stencil_start(xi0[todo], w)
        base_j[todo] = np.clip(_stencil_start(eta0[todo], w), 0, nj - 1 - w)
        blocked = np.zeros(n, dtype=bool)
        ar = np.arange(w)
        for dj, di in SHIFTS:
            if not todo.size:
                break
            i0 = base_i[todo] + di
            j0s = base_j[todo] + dj
            keep = (j0s >= 0) & (j0s <= nj - 1 - w)
            if not c.periodic:
                keep &= (i0 >= 0) & (i0 <= ni - w)
            sel, i0, j0s = todo[keep], i0[keep], j0s[keep]
            if not sel.size:
                continue
            jjs = j0s[:, None, None] + ar[None, :, None]
            iis = np.mod(i0[:, None, None] + ar[None, None, :], ni)
            nodes = usable[jjs, iis].all(axis=(1, 2))
            blocked[sel[~nodes]] = True
            sel, i0, j0s, jjs, iis = sel[nodes], i0[nodes], j0s[nodes], jjs[nodes], iis[nodes]
            if not sel.size:
                continue
            xq, eq = xi0[sel].copy(), eta0[sel].copy()
            for _ in range(8):
                fx, fy, dxx, dxy, dyx, dyy = self._lagrange_map(c, xq, eq, i0, j0s)
                det = dxx * dyy - dxy * dyx
                rx, ry = fx - px[sel], fy - py[sel]
                with np.errstate(divide="ignore", invalid="ignore"):
                    xq = xq - (dyy * rx - dxy * ry) / det
                    eq = eq - (-dyx * rx + dxx * ry) / det
            ti, tj = xq - i0, eq - j0s
            tl = 1e-6
            # A point can sit between a wall's polygon, which is what cuts, and the
            # curve the wall row's quadratic interpolant follows, which bulges past
            # the polygon's chords: outside the body by the cut and at eta just
            # below zero by the map (Tier 62 found two such road-patch points at
            # the tyre).  A stencil that starts at the wall may reach a quarter
            # of a row past it for them; nothing else extrapolates.
            floor = np.where(j0s == 0, -WALL_REACH, -tl)
            fits = (np.isfinite(ti) & np.isfinite(tj) & (ti >= -tl) & (ti <= w - 1 + tl)
                    & (tj >= floor) & (tj <= w - 1 + tl))
            acc = sel[fits]
            if acc.size:
                wi = lagrange_weights(ti[fits], w)
                wj = lagrange_weights(tj[fits], w)
                weights[acc] = (wj[:, :, None] * wi[:, None, :]).reshape(acc.size, -1)
                flat[acc] = (jjs[fits] * ni + iis[fits]).reshape(acc.size, -1)
                ok[acc] = True
                xi[acc] = xq[fits]
                eta[acc] = eq[fits]
                todo = np.setdiff1d(todo, acc, assume_unique=True)
        for q in todo:
            why[q] = ("every stencil holding it touches a hole or an interpolation point"
                      if blocked[q] else "no stencil holds it without extrapolating")
        return dict(flat=flat, weights=weights, ok=ok, why=why, xi=xi, eta=eta)

    # -- the rows a wall writes ---------------------------------------------

    def wall_columns(self, c: CurvilinearGrid) -> np.ndarray:
        """The ``i`` of every WALL point in row 0: all of them round a body, all
        but the two ends along a road patch."""
        return np.nonzero(self.status[c.name][0] == WALL)[0]

    def neumann_wall_triplets(self, c: CurvilinearGrid):
        """Tier 60's one-sided ``dp/dn`` row, at wall points only."""
        idx = self.index[c.name]
        ni = c.ni
        iw = self.wall_columns(c)
        wrow = idx[0, iw]
        g12 = c.a12[0, iw] / c.J[0, iw]
        g22 = c.a22[0, iw] / c.J[0, iw]
        sq = np.sqrt(g22)
        cols = [idx[0, np.mod(iw + 1, ni)], idx[0, np.mod(iw - 1, ni)], wrow, idx[1, iw], idx[2, iw]]
        if any(np.any(col < 0) for col in cols):                   # pragma: no cover
            raise OversetError(f"{c.name}: a wall row reaches a hole")
        return ([wrow] * 5, cols, [-0.5 * g12 / sq, 0.5 * g12 / sq, 1.5 * sq, -2.0 * sq, 0.5 * sq])

    def _assemble_component(self, c, rhs, wall_value, outer_value, R, C, V, b) -> None:
        idx = self.index[c.name]
        LR, LC, LV, jj, ii = self.laplacian_triplets(c)
        R += LR; C += LC; V += LV
        b[idx[jj, ii]] = rhs(c.x[jj, ii], c.y[jj, ii])
        if wall_value is None:
            raise ValueError(f"{c.name}: no wall data")
        iw = self.wall_columns(c)
        wrow = idx[0, iw]
        gw = np.asarray(wall_value(c), dtype=float)[iw]
        if c.wall == "dirichlet":
            R.append(wrow); C.append(wrow); V.append(np.ones(iw.size))
        else:
            NR, NC, NV = self.neumann_wall_triplets(c)
            R += NR; C += NC; V += NV
        b[wrow] = gw
        if c.outer == "dirichlet":
            raise ValueError(f"{c.name}: a MultiOverset joins every grid by interpolation")

    def report(self) -> dict[str, Any]:
        rep = super().report()
        rep["road"] = self.road
        rep["component_margin"] = self.component_margin
        for g in self.grids:
            cut = self.cut_by[g.name]
            S = self.status[g.name]
            by = {}
            for v in np.unique(cut[S == HOLE]):
                label = "road" if v == ROAD else ("?" if v < 0 else self.comps[int(v)].name)
                by[label] = int(np.sum((cut == v) & (S == HOLE)))
            rep["grids"][g.name]["holes_cut_by"] = by
            donors = {}
            for e in self.donors.get(g.name, {}).get("entries", []):
                donors[e["donor"]] = donors.get(e["donor"], 0) + int(len(e["j"]))
            rep["grids"][g.name]["donor_grids"] = donors
        return rep


# ===========================================================================
# verification: several bodies, the road and a patch, with known answers
# ===========================================================================

#: The verification box in tiling units, the road its lower face.
MULTI_BOX = (3.0, 1.2)
#: A wheel: centre x, radius, and its gap above the road as a fraction of the
#: radius -- the car's rule, two percent.
MULTI_WHEEL = (0.9, 0.3, 0.02)
#: A thick panel beside the wheel: leading and trailing points, thickness.  Its
#: nose comes within 0.027 of the tyre, under two background cells at h = 1/64.
MULTI_PANEL = (1.22, 0.16, 2.30, 0.18, 0.05)
#: An aerofoil above the panel: leading edge, chord, thickness fraction, angle.
MULTI_AEROFOIL = (1.50, 0.30, 0.50, 0.12, 12.0)
MULTI_HOLE_MARGIN = 0.04
#: Where the manufactured pressure's bump sits: between the three bodies.
MULTI_BUMP = (1.3, 0.25)


#: **The grid settings the verification runs at are the car's** (`car_solids.GRID`,
#: which a test holds equal to these): at level 3, where ``h = 1/64``, every grid
#: here is built the way the car's are.  Two of them were chosen on this very
#: geometry, and the record says so: a NACA trailing edge closed to a point
#: converged at order 1.1 behind it; rounding it to 1% of the chord took it to
#: 1.7, clustering the columns where the outline turns to 1.8, and smoothing
#: the normals at the wall (0.02, with ``kappa`` 0.5) to 2.0 -- all at L3->L4.
MULTI_GRID: dict[str, Any] = {
    "body_thickness": 0.10, "body_beta": 2.0, "cluster": 4.0, "sigma_wall": 0.02, "kappa": 0.5,
    "room": 0.4, "room_floor": 0.03, "wheel_thickness": 0.15, "wheel_beta": 3.0,
    "patch_half_width_r": 0.8, "patch_height_r": 0.25, "patch_spacing_h": 0.175,
    "te_thickness": 0.01,
}


def multi_geometry(level: int, *, road: bool = True, **override) -> MultiOverset:
    """The verification composite at ``h = 1/(16 2^(level-1))``.

    Every mapping is fixed and only sampled more finely with ``level`` (Tier
    60's lesson): the wheel is `ogrid_annulus`, the panel and the aerofoil are
    `ogrid_from_outline`, and the patch is `road_patch`, all at `MULTI_GRID`
    unless ``override`` says otherwise (``te_thickness=None`` is a NACA section
    closed to a point).  The background's rows sit AT ``y = j h``, so its first
    row is the road.
    """
    G = dict(MULTI_GRID)
    unknown = set(override) - set(G)
    if unknown:
        raise ValueError(f"unknown settings {sorted(unknown)}")
    G.update(override)
    n16 = 2 ** (level - 1)
    h = 1.0 / (16 * n16)
    Lx, Ly = MULTI_BOX
    bg = CartesianGrid("bg", int(round(Lx / h)), int(round(Ly / h)) + 1, h, y0=-0.5 * h)
    cx, R, gap = MULTI_WHEEL
    comps = [ogrid_annulus("wheel", cx, R * (1.0 + gap), R, R + G["wheel_thickness"],
                           96 * n16, 8 * n16 + 1, beta=G["wheel_beta"])]
    body_kw = dict(beta=G["body_beta"], cluster=G["cluster"], sigma_wall=G["sigma_wall"],
                   kappa=G["kappa"], room=G["room"], room_floor=G["room_floor"])
    x0, y0, x1, y1, th = MULTI_PANEL
    panel = rounded_plate_outline(x0, y0, math.hypot(x1 - x0, y1 - y0), th,
                                  math.degrees(math.atan2(y1 - y0, x1 - x0)))
    comps.append(ogrid_from_outline("panel", panel, 128 * n16, 6 * n16 + 1, G["body_thickness"],
                                    **body_kw))
    ax, ay, chord, tf, alpha = MULTI_AEROFOIL
    foil = (naca4_outline(ax, ay, chord, tf, alpha) if G["te_thickness"] is None
            else aerofoil_outline(ax, ay, chord, tf, alpha, G["te_thickness"]))
    comps.append(ogrid_from_outline("aerofoil", foil, 64 * n16, 6 * n16 + 1, G["body_thickness"],
                                    **body_kw))
    if road:
        half = G["patch_half_width_r"] * R
        ni = int(round(2 * half / (G["patch_spacing_h"] * h))) + 1
        comps.append(road_patch("road", cx - half, cx + half, G["patch_height_r"] * R, ni,
                                8 * n16 + 1, beta=G["wheel_beta"]))
    return MultiOverset(bg, comps, hole_margin=MULTI_HOLE_MARGIN, width=3,
                        road=0.0 if road else None)


def manufactured_poisson_multi(level: int, **kw) -> dict[str, Any]:
    """Tier 60's manufactured pressure on the multi-body composite: Neumann on
    every body's wall, Dirichlet on the road patch's wall and on the box, and
    the error per grid against the exact solution."""
    ov = multi_geometry(level, **kw)
    for c in ov.comps:
        if not c.periodic:
            c.wall = "dirichlet"
    bx, by = MULTI_BUMP

    def rhs(x, y):
        return _ms(x, y, bx, by)[3]

    def box_value(x, y):
        return _ms(x, y, bx, by)[0]

    def wall_value(g):
        p, gx, gy, _l = _ms(g.x[0], g.y[0], bx, by)
        if g.wall == "dirichlet":
            return p
        nx_, ny_ = g.wall_normal()
        return nx_ * gx + ny_ * gy

    t0 = time.perf_counter()
    A, b = ov.poisson(rhs, box_value, wall_value)
    t_asm = time.perf_counter() - t0
    x, info = solve_sparse(A, b, "splu")
    info.pop("_lu", None)
    fields = ov.scatter(x)
    out: dict[str, Any] = {"level": level, "h": ov.bg.h, "n_unknowns": ov.n_unknowns,
                           "build_s": ov.build_s, "assemble_s": t_asm, "solve": info,
                           "overset": ov.report(), "grids": {}}
    for g in ov.grids:
        S = ov.status[g.name]
        X, Y = MultiOverset._xy(g)
        pe = _ms(X, Y, bx, by)[0]
        live = S != HOLE
        full = np.where(live, np.abs(fields[g.name] - pe), -1.0)
        err = full[live]
        jm, im_ = np.unravel_index(int(np.argmax(full)), full.shape)
        out["grids"][g.name] = {
            "max_err": float(err.max()), "rms_err": float(np.sqrt(np.mean(err ** 2))),
            "max_at": {"j": int(jm), "i": int(im_), "x": float(X[jm, im_]), "y": float(Y[jm, im_]),
                       "status": STATUS_NAMES[int(S[jm, im_])]}}
    return out


def mms_flow_multi(level: int, dt: float, T: float, *, nu: float = 0.01, chi: float = 1.0,
                   **kw) -> dict[str, Any]:
    """Tier 61's manufactured flow on the multi-body composite: every wall --
    the tyre, the panel, the aerofoil and the road patch -- moves with the exact
    velocity, every box face carries it, and the error is measured per grid."""
    from . import overset_ns as NS
    ov = multi_geometry(level, **kw)
    flow = NS.OversetFlow(
        ov, nu, dt, box={f: "dirichlet" for f in NS.FACES},
        box_velocity=lambda x, y, t: NS.mms_fields(x, y, t, nu)[:2],
        wall_velocity=lambda g, t: NS.mms_fields(g.x[0], g.y[0], t, nu)[:2],
        forcing=lambda x, y, t: NS.mms_fields(x, y, t, nu)[3:], chi=chi)
    u0, v0, p0, _a, _b = NS.mms_fields(flow.X, flow.Y, 0.0, nu)
    um, vm, _p, _c, _d = NS.mms_fields(flow.X, flow.Y, -dt, nu)
    flow.set_state(u0, v0, p0, u_prev=um, v_prev=vm)
    steps = int(round(T / dt))
    its = []
    t0 = time.perf_counter()
    for _ in range(steps):
        its.append(max(flow.step()["iterations"]))
    wall_s = time.perf_counter() - t0
    ue, ve, pe, _e, _f = NS.mms_fields(flow.X, flow.Y, flow.t, nu)
    dp = flow.P - pe
    dp = dp - dp.mean()
    out: dict[str, Any] = {"level": level, "h": ov.bg.h, "dt": dt, "T": flow.t, "steps": steps,
                           "n_unknowns": ov.n_unknowns, "wall_s": wall_s, "setup_s": flow.setup_s,
                           "momentum_iterations_max": int(max(its)) if its else 0,
                           "divergence": flow.divergence_report(), "grids": {}}
    for g in ov.grids:
        ix = ov.index[g.name]
        rows = ix[ix >= 0]
        eu = np.maximum(np.abs(flow.U[rows] - ue[rows]), np.abs(flow.V[rows] - ve[rows]))
        ep = np.abs(dp[rows])
        ku, kp = int(rows[np.argmax(eu)]), int(rows[np.argmax(ep)])
        out["grids"][g.name] = {
            "u_max_err": float(eu.max()), "u_rms_err": float(np.sqrt(np.mean(eu ** 2))),
            "p_max_err": float(ep.max()), "p_rms_err": float(np.sqrt(np.mean(ep ** 2))),
            "u_max_at": [float(flow.X[ku]), float(flow.Y[ku])],
            "p_max_at": [float(flow.X[kp]), float(flow.Y[kp])]}
    return out
