"""A plate with channels: cut across at stations, and along every channel's walls (the
coolant plan's Phase 2, [[chart-operator-stage-0b-and-1-plan]]; the owner, 2026-11-19:
"use the same domain decomposition tools we've already developed").

The grid cut (`pieces.cut_grid`) cuts a rectangle along straight lines at every
rectangular region's edges.  This cut keeps its columns and replaces its horizontal
lines with **interfaces**: the plate's bottom and top, the straight lines along a
rectangular region's horizontal edges, and both walls of every channel, whatever
their shape.  A channel is any region of a coolant material (one drawn with the
Channel tool, a polygon or a rectangle) whose outline has four sharp corners, two on
the plate's left edge and two on its right, with walls that run from left to right as
graphs over x.  Each column between two stations is cut at every interface it
crosses, so each piece is a four-sided curved quadrilateral:

    bottom: an interface over [x_c, x_c+1], parametrised by x;
    right:  the station line x = x_c+1 between the two interfaces;
    top:    the next interface, back from x_c+1 to x_c;
    left:   the station line x = x_c.

**Conforming by construction.**  Two pieces that share a side share the same `Side`
object (one reversed), so every chart places its boundary nodes at the same points
(`chart.boundary_nodes` samples a side at t = i/n), and every seam conforms.

**Charted by the tools already built**: TFI (the Coons patch, here a vertical shear of
the column, the default), and the Pieces screen's Laplacian or conformal on request
(`pieces.chart_of`).  A TFI chart that Proposition 3 does not certify is replaced by
Winslow's.

**Refused, named:** a drawn plate outline (the plate is the grid's rectangle here), a
hole, a channel that does not cross the plate from left to right or whose wall turns
back on itself, a region of another material that is not a rectangle, two interfaces
that cross or touch, and a piece whose corners hold two materials.

**Not built:** a serpentine that turns back on itself (U-turns): its walls are not
graphs over x, and cutting round a turn needs blocks of another shape.
"""

from __future__ import annotations

import math

import numpy as np

from atlas.chartop import certificate as cert
from atlas.chartop import chart as ch

from . import pieces as P


def _materials_flowing(spec) -> set:
    mats = P._materials(spec)
    return {k for k, v in mats.items() if float(v.get("flows", 0.0)) >= 0.5}


def _region_edges(spec, r) -> list:
    """A region's outline as sampled edges (metres), counter-clockwise."""
    from atlas.workbench import shapes
    dx = float(spec.domain.dx)
    if r.shape == "curve" and r.outline is not None:
        o = r.outline
        E = [np.asarray(e, float) * dx for e in shapes.edges_sampled(o.points, o.edges, o.bulge, step=0.02)]
    elif r.shape == "polygon" and r.points is not None:
        p = np.asarray(r.points, float) * dx
        E = [np.vstack([p[k], p[(k + 1) % len(p)]]) for k in range(len(p))]
    else:
        x0, y0, x1, y1 = r.x0 * dx, r.y0 * dx, (r.x0 + r.nx) * dx, (r.y0 + r.ny) * dx
        c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        E = [np.array([c[k], c[(k + 1) % 4]], float) for k in range(4)]
    # counter-clockwise
    pts = np.vstack([e[:-1] for e in E])
    area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    if area < 0:
        E = [e[::-1].copy() for e in E[::-1]]
    return E


def _four_sides(E) -> list:
    cs = P._sharpest_corners(E)
    groups = []
    for q in range(4):
        k0, k1 = cs[q], cs[(q + 1) % 4]
        ks = list(range(k0, k1)) if k1 > k0 else list(range(k0, len(E))) + list(range(0, k1))
        groups.append(np.vstack([E[ks[0]]] + [E[k][1:] for k in ks[1:]]))
    return groups


def channel_walls(spec, r, W: float, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """A channel region's two walls, each from x = 0 to x = W with x rising: (lower,
    upper), by their mean height."""
    groups = _four_sides(_region_edges(spec, r))
    ends = [g for g in groups if np.all(np.abs(g[:, 0]) < tol) or np.all(np.abs(g[:, 0] - W) < tol)]
    walls = [g for g in groups if not (np.all(np.abs(g[:, 0]) < tol) or np.all(np.abs(g[:, 0] - W) < tol))]
    if len(ends) != 2 or len(walls) != 2:
        raise P.CutError(f"channel {r.id} does not cross the plate from its left edge to its right "
                         "edge (two ends on those edges, two walls between them): not cut")
    out = []
    for w in walls:
        w = w if w[0, 0] < w[-1, 0] else w[::-1]
        if np.any(np.diff(w[:, 0]) <= -tol):
            raise P.CutError(f"a wall of channel {r.id} turns back on itself (a U-turn): a serpentine "
                             "that doubles back is not cut here")
        keep = np.concatenate([[True], np.diff(w[:, 0]) > 1e-15])
        w = w[keep]
        w[0, 0], w[-1, 0] = 0.0, W
        out.append(w)
    out.sort(key=lambda w: float(np.mean(w[:, 1])))
    return out[0], out[1]


class _Interface:
    """A curve across the plate, a graph y(x) from x = 0 to W."""

    def __init__(self, P_: np.ndarray, name: str):
        self.P = np.asarray(P_, float)
        self.name = name

    def y(self, x):
        return np.interp(x, self.P[:, 0], self.P[:, 1])

    def segment(self, xa: float, xb: float) -> ch.Side:
        """From x = xa to x = xb (xa < xb), parametrised by x: t = (x - xa) / (xb - xa),
        sampled densely and at the curve's own vertices, so a kink stays exact."""
        inner = self.P[(self.P[:, 0] > xa) & (self.P[:, 0] < xb), 0]
        t = np.unique(np.concatenate([np.linspace(0.0, 1.0, 2049), (inner - xa) / (xb - xa)]))
        x = xa + t * (xb - xa)
        return ch.Side(np.column_stack([x, self.y(x)]), t)


def cut_channels(spec, n: int = 64, cols: int = 1, rows: int = 1, envelope=None) -> P.Cut:
    """The plate cut at ``cols`` equal columns (and at every rectangular region's
    vertical edges), along every channel's walls and every rectangular region's
    horizontal edges, and each layer between two of those split into ``rows`` by the
    curves at equal fractions between them.  With TFI charts, ``cols`` and ``rows``
    doubled at half the ``n`` give the same nodes: a finer cut of the same mesh."""
    d = spec.domain
    if d.outline is not None or d.holes:
        raise P.CutError("the channel cut takes the grid's rectangle as the plate; this domain has "
                         "a drawn outline or holes")
    rows = max(1, int(rows))
    dx = float(d.dx)
    W, H = d.nx * dx, d.ny * dx
    tol = 0.02 * dx
    wet = _materials_flowing(spec)
    interfaces = [_Interface(np.array([[0.0, 0.0], [W, 0.0]]), "bottom"),
                  _Interface(np.array([[0.0, H], [W, H]]), "top")]
    xs = set([0.0, W] + [round(k * W / max(1, cols), 15) for k in range(1, max(1, cols))])
    walls = []                                   # (lower, upper, material) per channel
    for r in spec.regions:
        if r.material in wet:
            lo, hi = channel_walls(spec, r, W, tol)
            a_, b_ = _Interface(lo, f"{r.id}'s lower wall"), _Interface(hi, f"{r.id}'s upper wall")
            interfaces += [a_, b_]
            walls.append((a_, b_, r.material))
        elif getattr(r, "shape", "rect") == "rect" and r.points is None:
            for v in (r.y0 * dx, (r.y0 + r.ny) * dx):
                if tol < v < H - tol:
                    interfaces.append(_Interface(np.array([[0.0, v], [W, v]]), f"{r.id}'s edge"))
            for v in (r.x0 * dx, (r.x0 + r.nx) * dx):
                if tol < v < W - tol:
                    xs.add(round(v, 15))
        else:
            raise P.CutError(f"region {r.id} is neither a coolant channel nor a rectangle: the channel "
                             "cut follows channels and rectangular regions only")
    if not walls:
        raise P.CutError("no coolant channel to follow: use the grid cut")
    xs = sorted(xs)
    # interfaces in order of height, apart at every point across the plate
    probe = np.unique(np.concatenate([np.linspace(0.0, W, 257), xs]))
    Y = np.array([f.y(probe) for f in interfaces])
    order = np.argsort(Y[:, len(probe) // 2])
    Ys = Y[order]
    keep = [0]
    for k in range(1, len(order)):
        g = Ys[k] - Ys[keep[-1]]
        if np.all(g <= tol):                     # one lying on another: merged away
            continue
        if np.any(g <= tol):
            a_, b_ = interfaces[order[keep[-1]]].name, interfaces[order[k]].name
            raise P.CutError(f"{a_} and {b_} cross or touch: the channel cut needs interfaces that "
                             "stay apart across the whole plate")
        keep.append(k)
    base = [interfaces[order[k]] for k in keep]
    # what each layer between consecutive base interfaces is: a channel's, or the plate's
    layer_mat = []
    for k in range(len(base) - 1):
        m = next((mat for lo, hi, mat in walls if lo is base[k] and hi is base[k + 1]), None)
        layer_mat.append(m)
    # split every layer into rows by the curves at equal fractions between its sides
    lines, owner = [base[0]], []
    for k in range(len(base) - 1):
        lo, hi = base[k], base[k + 1]
        xv = np.unique(np.concatenate([lo.P[:, 0], hi.P[:, 0], xs]))
        for j in range(1, rows):
            f = j / rows
            lines.append(_Interface(np.column_stack([xv, (1 - f) * lo.y(xv) + f * hi.y(xv)]),
                                    f"row {j} of layer {k}"))
            owner.append(k)
        lines.append(hi)
        owner.append(k)
    nL, nC = len(lines), len(xs) - 1
    seg = {(k, c): lines[k].segment(xs[c], xs[c + 1]) for k in range(nL) for c in range(nC)}
    vert = {}
    for c in range(nC + 1):
        for k in range(nL - 1):
            x = xs[c]
            vert[(c, k)] = ch.line((x, float(lines[k].y(x))), (x, float(lines[k + 1].y(x))))
    pieces, refolded = [], 0
    for k in range(nL - 1):
        lay = owner[k]
        for c in range(nC):
            s0 = seg[(k, c)]
            s1 = vert[(c + 1, k)]
            s2 = seg[(k + 1, c)].reversed()
            s3 = vert[(c, k)].reversed()
            sides = [s0, s1, s2, s3]
            X = ch.transfinite(sides, n)
            chart, own = ch.Chart(X, "transfinite"), "transfinite"
            if not cert.certify(X).ok:
                Xw, info = ch.winslow(sides, n)
                chart, own = ch.Chart(Xw, "winslow-arc-length", info), "winslow-arc-length"
                refolded += 1
            if layer_mat[lay] is not None:
                mat = layer_mat[lay]
            else:
                mats = set()
                for fx in (0.25, 0.5, 0.75):
                    xq = xs[c] + fx * (xs[c + 1] - xs[c])
                    mats.add(P.material_at(spec, xq, 0.5 * float(lines[k].y(xq) + lines[k + 1].y(xq))))
                mats.discard(None)
                if len(mats) > 1:
                    raise P.CutError(f"column {c}, layer {k} holds {sorted(mats)}: a region's edge crosses "
                                     "it; draw regions whose edges are channel walls or straight lines "
                                     "across the plate")
                mat = next(iter(mats)) if mats else ""
            pieces.append(P.Piece(sides, mat, f"col {c}, layer {k}", own, chart))
    notes = [f"{len(pieces)} pieces: {nC} columns, {nL - 1} layers"
             + (f" ({rows} rows in each of {len(base) - 1})" if rows > 1 else "")
             + f"; {len(walls)} channel{'s' if len(walls) != 1 else ''}, cut along both walls"]
    if refolded:
        notes.append(f"{refolded} pieces charted by Winslow: their TFI was not certified")
    return P.Cut("channels", {"cols": cols, "rows": rows}, n, pieces, notes)


__all__ = ["channel_walls", "cut_channels"]
