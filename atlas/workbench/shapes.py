"""Closed shapes whose edges are straight lines, circular arcs or splines.

What a person draws in the geometry section -- the domain's outline, the holes
cut in it, material regions, and windows -- is one kind of object: a list of
vertices, and for each edge between two consecutive vertices how it is drawn.
The owner's decision (2026-09-29): any shape, drawn in the page, with polygon
edges, arcs and splines, and the solvers get **the cells whose centres lie
inside it** (`geometry._polygon_mask`, the rule polygon regions have used since
case file 0.2).  So a curve is resolved to the grid's cells, and the canvas
draws the exact curve over them.

Units are CELLS: the grid's lower-left corner is (0, 0), and cell ``(i, j)``
spans ``[i, i+1] x [j, j+1]``.

A shape is ``(points, kinds, bulges)``:

``points``  the vertices in order; edge ``k`` runs from ``points[k]`` to
            ``points[(k + 1) % n]``, so the last edge closes the shape;
``kinds``   per edge, ``"line"``, ``"arc"`` or ``"spline"``;
``bulges``  per edge, an arc's **bulge** ``b = tan(theta / 4)``, ``theta`` its
            included angle -- the DXF convention, so an arc keeps its shape when
            an end point moves.  Its sagitta is ``s = b c / 2`` for a chord of
            length ``c``, and a positive bulge bows to the RIGHT of the edge's
            direction of travel: outward, on a counter-clockwise shape.  A
            semicircle is ``|b| = 1``.  Ignored on other kinds.

A **spline** edge is the centripetal Catmull-Rom segment between its two
vertices, shaped by the vertices either side, so a run of spline edges is one
smooth curve through its vertices (and centripetal, so it neither cusps nor
self-intersects within a segment).  Next to a straight edge it meets the line
at the shared vertex without matching its direction.

Every function here is pure: it takes lists and returns new ones, so an edit
can be applied to a case, undone and re-applied like any other.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

EDGE_KINDS = ("line", "arc", "spline")

#: the sampling step, in cells, used to turn a shape into a polygon.  Half a
#: cell: between two samples a circular arc of radius R strays at most
#: step^2 / (8 R) from its chord -- 1/32 of a cell at R = 1 -- so which cell
#: centres fall inside cannot depend on it except within a hair of a centre.
STEP = 0.5

#: the largest bulge accepted: an arc of about 304 degrees on its chord
MAX_BULGE = 4.0

Point = tuple[float, float]


def _norm_kinds(n: int, kinds, bulges) -> tuple[list[str], list[float]]:
    ks = list(kinds) if kinds is not None else ["line"] * n
    bs = [float(b) for b in bulges] if bulges is not None else [0.0] * n
    if len(ks) != n or len(bs) != n:
        raise ValueError(f"a shape with {n} vertices has {n} edges; got {len(ks)} kinds "
                         f"and {len(bs)} bulges")
    for k in ks:
        if k not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind {k!r}; one of {', '.join(EDGE_KINDS)}")
    return ks, bs


# ---------------------------------------------------------------------------
# one edge
# ---------------------------------------------------------------------------


def _line(a: np.ndarray, b: np.ndarray, step: float) -> np.ndarray:
    c = float(np.hypot(*(b - a)))
    n = max(1, int(math.ceil(c / step)))
    t = np.arange(n)[:, None] / n
    return a[None, :] + t * (b - a)[None, :]


def circumcentre(a, m, b) -> np.ndarray | None:
    """The centre of the circle through three points, or None when they are in line."""
    ax, ay = a
    bx, by = m
    cx, cy = b
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12 * max(1.0, abs(ax) + abs(bx) + abs(cx) + abs(ay) + abs(by) + abs(cy)):
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.array([ux, uy])


def arc_midpoint(a, b, bulge: float) -> np.ndarray:
    """The point halfway along an arc: the chord's midpoint pushed out by the sagitta."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    c = float(np.hypot(*d))
    if c == 0.0:
        return a.copy()
    right = np.array([d[1], -d[0]]) / c
    return 0.5 * (a + b) + (bulge * c / 2.0) * right


def _arc(a: np.ndarray, b: np.ndarray, bulge: float, step: float) -> np.ndarray:
    if abs(bulge) < 1e-9:
        return _line(a, b, step)
    m = arc_midpoint(a, b, bulge)
    o = circumcentre(a, m, b)
    if o is None:
        return _line(a, b, step)
    r = float(np.hypot(*(a - o)))
    aa = math.atan2(a[1] - o[1], a[0] - o[0])
    am = math.atan2(m[1] - o[1], m[0] - o[0])
    ab = math.atan2(b[1] - o[1], b[0] - o[0])
    ccw = (ab - aa) % (2.0 * math.pi)
    sweep = ccw if (am - aa) % (2.0 * math.pi) < ccw else ccw - 2.0 * math.pi
    n = max(2, int(math.ceil(abs(sweep) * r / step)))
    ang = aa + sweep * (np.arange(n) / n)
    return np.column_stack([o[0] + r * np.cos(ang), o[1] + r * np.sin(ang)])


def _catmull_rom(p0, p1, p2, p3, t: np.ndarray) -> np.ndarray:
    """Centripetal Catmull-Rom (alpha = 1/2) between p1 and p2, at fractions t of it."""
    def knot(ti, pa, pb):
        return ti + max(float(np.hypot(*(pb - pa))) ** 0.5, 1e-9)
    t0 = 0.0
    t1 = knot(t0, p0, p1)
    t2 = knot(t1, p1, p2)
    t3 = knot(t2, p2, p3)
    tt = (t1 + (t2 - t1) * t)[:, None]
    a1 = (t1 - tt) / (t1 - t0) * p0 + (tt - t0) / (t1 - t0) * p1
    a2 = (t2 - tt) / (t2 - t1) * p1 + (tt - t1) / (t2 - t1) * p2
    a3 = (t3 - tt) / (t3 - t2) * p2 + (tt - t2) / (t3 - t2) * p3
    b1 = (t2 - tt) / (t2 - t0) * a1 + (tt - t0) / (t2 - t0) * a2
    b2 = (t3 - tt) / (t3 - t1) * a2 + (tt - t1) / (t3 - t1) * a3
    return (t2 - tt) / (t2 - t1) * b1 + (tt - t1) / (t2 - t1) * b2


def _spline(p0, p1, p2, p3, step: float) -> np.ndarray:
    coarse = _catmull_rom(p0, p1, p2, p3, np.linspace(0.0, 1.0, 17))
    length = float(np.sum(np.hypot(*np.diff(coarse, axis=0).T)))
    n = max(2, int(math.ceil(length / step)))
    return _catmull_rom(p0, p1, p2, p3, np.arange(n) / n)


def edge_points(points: Sequence[Point], kinds, bulges, k: int,
                step: float = STEP) -> np.ndarray:
    """Edge ``k`` sampled from its first vertex up to (not including) its last."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    ks, bs = _norm_kinds(n, kinds, bulges)
    a, b = p[k], p[(k + 1) % n]
    if ks[k] == "arc":
        return _arc(a, b, bs[k], step)
    if ks[k] == "spline" and n >= 3:
        return _spline(p[(k - 1) % n], a, b, p[(k + 2) % n], step)
    return _line(a, b, step)


def sample(points: Sequence[Point], kinds=None, bulges=None,
           step: float = STEP) -> np.ndarray:
    """The whole shape as a closed polygon, ``(M, 2)``, first point not repeated."""
    n = len(points)
    if n < 2:
        raise ValueError("a shape needs at least two vertices")
    parts = [edge_points(points, kinds, bulges, k, step) for k in range(n)]
    out = np.concatenate(parts)
    # drop consecutive duplicates (a zero-length edge)
    keep = np.ones(len(out), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(out, axis=0)) > 1e-12, axis=1)
    return out[keep]


def edges_sampled(points, kinds=None, bulges=None, step: float = STEP) -> list[np.ndarray]:
    """Each edge as a polyline INCLUDING both end points, for drawing an edge alone
    (a boundary condition on it) and for finding the edge nearest a face."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    return [np.vstack([edge_points(points, kinds, bulges, k, step), p[(k + 1) % n][None, :]])
            for k in range(n)]


def edge_midpoint(points, kinds, bulges, k: int) -> np.ndarray:
    """The point halfway along edge ``k``, where its handle is drawn."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    ks, bs = _norm_kinds(n, kinds, bulges)
    a, b = p[k], p[(k + 1) % n]
    if ks[k] == "arc":
        return arc_midpoint(a, b, bs[k])
    if ks[k] == "spline" and n >= 3:
        return _catmull_rom(p[(k - 1) % n], a, b, p[(k + 2) % n], np.array([0.5]))[0]
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# the polygon's own properties
# ---------------------------------------------------------------------------


def signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def self_intersects(ring: np.ndarray) -> bool:
    """Whether two non-adjacent sides of a closed polygon cross or touch."""
    m = len(ring)
    if m < 4:
        return False
    a = ring
    b = np.roll(ring, -1, axis=0)
    # every pair (i, j) with j > i + 1, leaving out the closing pair (0, m - 1)
    i, j = np.triu_indices(m, k=2)
    keep = ~((i == 0) & (j == m - 1))
    i, j = i[keep], j[keep]
    step = 200_000
    for s in range(0, len(i), step):
        ii, jj = i[s:s + step], j[s:s + step]
        p, r = a[ii], b[ii] - a[ii]
        q, t = a[jj], b[jj] - a[jj]
        den = r[:, 0] * t[:, 1] - r[:, 1] * t[:, 0]
        qp = q - p
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (qp[:, 0] * t[:, 1] - qp[:, 1] * t[:, 0]) / den
            v = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / den
        hit = (den != 0.0) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        if np.any(hit):
            return True
    return False


def contains(ring: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for many points at once."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    xa, ya = ring[:, 0], ring[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    for k in range(len(ring)):
        if ya[k] == yb[k]:
            continue
        crosses = (ya[k] > y) != (yb[k] > y)
        xc = xa[k] + (y - ya[k]) * (xb[k] - xa[k]) / (yb[k] - ya[k])
        inside ^= crosses & (x < xc)
    return inside


def problems(points, kinds=None, bulges=None) -> list[str]:
    """Why a shape cannot be used, or an empty list."""
    out = []
    n = len(points)
    try:
        ks, bs = _norm_kinds(n, kinds, bulges)
    except ValueError as exc:
        return [str(exc)]
    if n < 2:
        return ["a shape needs at least two vertices"]
    p = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(p)):
        return ["a vertex is not a finite number"]
    if n < 3 and not any(k == "arc" and abs(b) > 1e-9 for k, b in zip(ks, bs)):
        return ["two vertices joined by straight edges enclose nothing: add a vertex or "
                "bend an edge into an arc"]
    for k, b in zip(ks, bs):
        if k == "arc" and abs(b) > MAX_BULGE:
            out.append(f"an arc bulges by {b:g}; at most {MAX_BULGE:g} (about 300 degrees)")
    ring = sample(points, ks, bs, step=1.0)
    # crossing first: a figure-of-eight's two lobes cancel, so its signed area is
    # zero and "no area" would name the symptom instead of the cause
    if len(ring) >= 4 and self_intersects(ring):
        out.append("the outline crosses itself")
    elif len(ring) < 3 or abs(signed_area(ring)) < 1e-9:
        out.append("the shape encloses no area")
    return out


# ---------------------------------------------------------------------------
# edits: each returns (points, kinds, bulges) and, where edges are renumbered,
# which old edge each new edge came from
# ---------------------------------------------------------------------------


def bulge_from_point(a, b, p) -> float:
    """The bulge of the arc on chord ``a -> b`` whose midpoint is as far from the
    chord as ``p`` is: a handle dragged anywhere sets the sagitta to its distance
    from the chord, on its side."""
    a, b, p = (np.asarray(v, float) for v in (a, b, p))
    d = b - a
    c = float(np.hypot(*d))
    if c == 0.0:
        return 0.0
    right = np.array([d[1], -d[0]]) / c
    s = float(np.dot(p - 0.5 * (a + b), right))
    return float(np.clip(2.0 * s / c, -MAX_BULGE, MAX_BULGE))


def bend(points, kinds, bulges, k: int, p, straight_within: float = 0.5):
    """Edge ``k`` made the arc whose midpoint is level with ``p``; back to a straight
    line when ``p`` is within ``straight_within`` cells of the chord."""
    n = len(points)
    ks, bs = _norm_kinds(n, kinds, bulges)
    a, b = points[k], points[(k + 1) % n]
    bl = bulge_from_point(a, b, p)
    c = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    ks, bs = list(ks), list(bs)
    if abs(bl) * c / 2.0 < straight_within:
        ks[k], bs[k] = "line", 0.0
    else:
        ks[k], bs[k] = "arc", bl
    return [tuple(map(float, q)) for q in points], ks, bs


def split(points, kinds, bulges, k: int):
    """A vertex added halfway along edge ``k``; both halves keep its kind (an arc's
    halves are arcs of half the angle, so the curve does not move)."""
    n = len(points)
    ks, bs = _norm_kinds(n, kinds, bulges)
    mid = edge_midpoint(points, ks, bs, k)
    half = math.tan(math.atan(bs[k]) / 2.0) if ks[k] == "arc" else 0.0
    pts = [tuple(map(float, q)) for q in points]
    pts.insert(k + 1, (float(mid[0]), float(mid[1])))
    ks2 = ks[:k] + [ks[k], ks[k]] + ks[k + 1:]
    bs2 = bs[:k] + [half, half] + bs[k + 1:] if ks[k] == "arc" else bs[:k] + [0.0, 0.0] + bs[k + 1:]
    origin = list(range(k)) + [k, k] + list(range(k + 1, n))
    return pts, ks2, bs2, origin


def remove_vertex(points, kinds, bulges, i: int):
    """Vertex ``i`` removed; the edge before it now runs to the vertex after it, with
    its own kind (an arc's bulge is kept)."""
    n = len(points)
    ks, bs = _norm_kinds(n, kinds, bulges)
    if n <= 2:
        raise ValueError("a shape keeps at least two vertices")
    pts = [tuple(map(float, q)) for j, q in enumerate(points) if j != i]
    # new edge j is old edge keep[j]: old edge i is dropped, and old edge i - 1
    # (the one ending at the removed vertex) now runs on to the vertex after it
    keep = [j for j in range(n) if j != i]
    return pts, [ks[j] for j in keep], [bs[j] for j in keep], keep


def move_vertex(points, kinds, bulges, i: int, x: float, y: float):
    n = len(points)
    ks, bs = _norm_kinds(n, kinds, bulges)
    pts = [tuple(map(float, q)) for q in points]
    pts[i] = (float(x), float(y))
    return pts, list(ks), list(bs)


def set_kind(points, kinds, bulges, k: int, kind: str, bulge: float | None = None):
    n = len(points)
    ks, bs = _norm_kinds(n, kinds, bulges)
    if kind not in EDGE_KINDS:
        raise ValueError(f"unknown edge kind {kind!r}")
    ks, bs = list(ks), list(bs)
    ks[k] = kind
    if kind == "arc":
        bs[k] = float(bulge) if bulge is not None else (bs[k] if bs[k] else 0.35)
    else:
        bs[k] = 0.0
    return [tuple(map(float, q)) for q in points], ks, bs


def translate(points, dx: float, dy: float):
    return [(float(p[0]) + dx, float(p[1]) + dy) for p in points]


# ---------------------------------------------------------------------------
# ready-made shapes
# ---------------------------------------------------------------------------


def circle(cx: float, cy: float, r: float):
    """Four vertices joined by four quarter arcs: an exact circle."""
    pts = [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)]
    q = math.tan(math.pi / 8.0)             # counter-clockwise, so outward is to the right
    return [tuple(map(float, p)) for p in pts], ["arc"] * 4, [q] * 4


def ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 8):
    """``n`` vertices on the ellipse joined by splines (the spline passes through
    them, so the shape is the ellipse to well under a cell at showcase sizes)."""
    ang = 2.0 * math.pi * np.arange(n) / n
    pts = [(float(cx + rx * math.cos(a)), float(cy + ry * math.sin(a))) for a in ang]
    return pts, ["spline"] * n, [0.0] * n


def rectangle(x0: float, y0: float, w: float, h: float):
    pts = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    return [tuple(map(float, p)) for p in pts], ["line"] * 4, [0.0] * 4


def regular(cx: float, cy: float, r: float, n: int = 6):
    ang = 2.0 * math.pi * np.arange(n) / n + math.pi / 2.0
    pts = [(float(cx + r * math.cos(a)), float(cy + r * math.sin(a))) for a in ang]
    return pts, ["line"] * n, [0.0] * n


def annulus_sector(cx: float, cy: float, r_in: float, r_out: float, a0: float, a1: float):
    """A ring's sector between angles ``a0`` and ``a1`` (radians, counter-clockwise):
    two straight radial edges and two arcs, exact."""
    theta = a1 - a0
    q = math.tan(theta / 4.0)
    pts = [(cx + r_in * math.cos(a0), cy + r_in * math.sin(a0)),
           (cx + r_out * math.cos(a0), cy + r_out * math.sin(a0)),
           (cx + r_out * math.cos(a1), cy + r_out * math.sin(a1)),
           (cx + r_in * math.cos(a1), cy + r_in * math.sin(a1))]
    # the shape runs counter-clockwise.  The outer arc bows away from the centre,
    # to the right of its travel: +q.  The inner arc runs back clockwise, and also
    # bows away from the centre (its midpoint is further out than its chord's),
    # which is to the LEFT of its travel: -q.
    return ([tuple(map(float, p)) for p in pts], ["line", "arc", "line", "arc"],
            [0.0, q, 0.0, -q])


__all__ = ["EDGE_KINDS", "STEP", "MAX_BULGE", "sample", "edge_points", "edges_sampled",
           "edge_midpoint", "arc_midpoint", "circumcentre", "signed_area", "self_intersects",
           "contains", "problems", "bulge_from_point", "bend", "split", "remove_vertex",
           "move_vertex", "set_kind", "translate", "circle", "ellipse", "rectangle",
           "regular", "annulus_sector"]
