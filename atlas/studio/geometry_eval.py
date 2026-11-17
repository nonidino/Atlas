"""The feature tree evaluated: the domain, holes and regions the engine solves on.

The page draws the tree with Paper.js; this module is the truth the solvers get
(integrated UI plan, section 3.3).  Each feature is a closed ring of **sides** (a
rectangle's four lines, a circle's four quarter arcs, a path's own edges), sampled
finely enough that no cell centre can tell the samples from the curve.  Shapely
(GEOS) does the booleans: the domain is the union of what adds minus the union of
what cuts.

**The engine's edges are kept as the person drew them.**  A boundary condition in a
case file names an edge of the outline (``outline:3``), so an outline made of a
thousand sampled segments would hand the person a thousand edges.  Every segment of a
boolean's result is traced back to the side it came from (an STR-tree over the input
segments); consecutive segments from one side become ONE engine edge again: a line, or
an arc with its bulge (the DXF convention the engine reads).  A side that survives
whole, of a ring that survives whole, is written exactly as drawn, splines included,
so a migrated 0.6 outline comes back with its own edges in its own order.  Cubic
curves and ellipses, which the engine has no edge for, become runs of short lines.

**Grid.**  The engine's cells start at the origin: cell ``(i, j)`` spans
``[i dx, (i+1) dx] x [j dx, (j+1) dx]``.  The grid is the domain's bounding box from
the origin, in whole cells; a domain reaching below or left of the origin is refused
with a fix that moves the drawing.  A domain that is exactly the grid's rectangle is
written as no outline at all (the whole grid), as a 0.6 case holds it.

Exact areas, for the check, come from Gmsh's OpenCASCADE kernel when Gmsh is
installed (`occ.py`), and from Shapely otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from atlas.workbench import geometry as wgeo
from atlas.workbench import registry, shapes
from atlas.workbench.spec import CaseSpec, Domain, Outline, Region, slug

from .document import Feature, Geometry

#: sampling tolerance as a fraction of a cell: a curve and its samples differ by less
#: than this, so which cell centres fall inside cannot depend on the sampling
TOL_CELLS = 1.0 / 64.0
#: the largest number of samples one side gets
MAX_SAMPLES = 4000


class GeometryError(ValueError):
    """The tree cannot be turned into a domain; the message says why, for a person."""


# ---------------------------------------------------------------------------
# sides
# ---------------------------------------------------------------------------


@dataclass
class Side:
    """One side of a feature's ring, from ``p0`` to ``p1``, parametrised by t in [0, 1]."""
    fid: str
    index: int
    kind: str                                   # line, arc, cubic, spline, ell
    p0: np.ndarray
    p1: np.ndarray
    centre: np.ndarray | None = None            # arc, ell
    radius: float = 0.0                         # arc
    a0: float = 0.0                             # arc, ell: start angle
    sweep: float = 0.0                          # arc, ell: signed, counter-clockwise > 0
    rx: float = 0.0                             # ell
    ry: float = 0.0
    c1: np.ndarray | None = None                # cubic
    c2: np.ndarray | None = None
    q0: np.ndarray | None = None                # spline: the neighbours
    q3: np.ndarray | None = None
    bulge: float = 0.0                          # arc: as drawn (for an exact rewrite)
    pts: np.ndarray | None = None               # poly: an explicit polyline, p0 .. p1

    def at(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.kind == "poly":
            s = np.linspace(0.0, 1.0, len(self.pts))
            return np.column_stack([np.interp(t, s, self.pts[:, 0]), np.interp(t, s, self.pts[:, 1])])
        if self.kind == "line":
            return self.p0[None, :] + t[:, None] * (self.p1 - self.p0)[None, :]
        if self.kind == "arc":
            a = self.a0 + self.sweep * t
            return np.column_stack([self.centre[0] + self.radius * np.cos(a),
                                    self.centre[1] + self.radius * np.sin(a)])
        if self.kind == "ell":
            a = self.a0 + self.sweep * t
            return np.column_stack([self.centre[0] + self.rx * np.cos(a),
                                    self.centre[1] + self.ry * np.sin(a)])
        if self.kind == "cubic":
            u = 1.0 - t
            return ((u ** 3)[:, None] * self.p0 + (3 * u * u * t)[:, None] * self.c1
                    + (3 * u * t * t)[:, None] * self.c2 + (t ** 3)[:, None] * self.p1)
        if self.kind == "spline":
            return shapes._catmull_rom(self.q0, self.p0, self.p1, self.q3, t)
        raise ValueError(self.kind)

    def count(self, tol: float) -> int:
        """Intervals so that the chord error is below ``tol``."""
        if self.kind == "line":
            return 1
        if self.kind in ("arc", "ell"):
            r = self.radius if self.kind == "arc" else min(self.rx, self.ry) ** 2 / max(self.rx, self.ry)
            r = max(r, 1e-15)
            step = 2.0 * math.acos(max(-1.0, 1.0 - min(tol / r, 1.0)))
            return int(min(MAX_SAMPLES, max(2, math.ceil(abs(self.sweep) / max(step, 1e-9)))))
        coarse = self.at(np.linspace(0, 1, 33))
        length = float(np.sum(np.hypot(*np.diff(coarse, axis=0).T)))
        if self.kind == "cubic":
            ctrl = (np.hypot(*(self.c1 - self.p0)) + np.hypot(*(self.c2 - self.c1))
                    + np.hypot(*(self.p1 - self.c2)))
            length = max(length, float(ctrl))
        n = 2.0 * math.sqrt(max(length, 1e-15) / (8.0 * tol))
        return int(min(MAX_SAMPLES, max(4, math.ceil(n))))

    def samples(self, tol: float) -> tuple[np.ndarray, np.ndarray]:
        """Points from p0 up to, not including, p1, and their parameters."""
        if self.kind == "poly":
            return self.pts[:-1].copy(), np.linspace(0.0, 1.0, len(self.pts))[:-1]
        n = self.count(tol)
        t = np.arange(n) / n
        pts = self.at(t)
        pts[0] = self.p0
        return pts, t

    def t_of(self, p: np.ndarray, near: float) -> float:
        """The parameter of a point on this side (``near``: the parameter it is close to)."""
        if self.kind == "line":
            d = self.p1 - self.p0
            L2 = float(d @ d)
            return float(((p - self.p0) @ d) / L2) if L2 > 0 else 0.0
        if self.kind == "arc":
            a = math.atan2(p[1] - self.centre[1], p[0] - self.centre[0])
            guess = self.a0 + self.sweep * near
            a = guess + ((a - guess + math.pi) % (2.0 * math.pi) - math.pi)
            return (a - self.a0) / self.sweep if self.sweep else 0.0
        return near


def _arc_side(fid, k, a, b, bulge) -> Side:
    """An arc by its chord and bulge, as the engine draws it (`shapes._arc`)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if abs(bulge) < 1e-12:
        return Side(fid, k, "line", a, b)
    m = shapes.arc_midpoint(a, b, bulge)
    o = shapes.circumcentre(a, m, b)
    if o is None:
        return Side(fid, k, "line", a, b)
    r = float(np.hypot(*(a - o)))
    a0 = math.atan2(a[1] - o[1], a[0] - o[0])
    # the engine's own rule for which way round: the way that passes the midpoint
    aa, am = a0, math.atan2(m[1] - o[1], m[0] - o[0])
    ab = math.atan2(b[1] - o[1], b[0] - o[0])
    ccw = (ab - aa) % (2.0 * math.pi)
    sweep = ccw if (am - aa) % (2.0 * math.pi) < ccw else ccw - 2.0 * math.pi
    return Side(fid, k, "arc", a, b, centre=o, radius=r, a0=a0, sweep=sweep, bulge=float(bulge))


def _circle_arc(fid, k, c, r, a0, sweep) -> Side:
    c = np.asarray(c, float)
    p0 = c + r * np.array([math.cos(a0), math.sin(a0)])
    p1 = c + r * np.array([math.cos(a0 + sweep), math.sin(a0 + sweep)])
    return Side(fid, k, "arc", p0, p1, centre=c, radius=float(r), a0=a0, sweep=sweep,
                bulge=math.tan(sweep / 4.0))


class Placement:
    """A rigid motion: rotation by ``angle`` about the origin, after a reflection of y
    (``flip``), then a translation.  Patterns and mirrors place copies of a shape's
    sides with it, so a copy of a circle is still four arcs."""

    def __init__(self, angle: float = 0.0, flip: bool = False, tx: float = 0.0, ty: float = 0.0):
        self.c, self.s = math.cos(angle), math.sin(angle)
        self.flip, self.tx, self.ty = flip, tx, ty

    def __call__(self, p) -> np.ndarray:
        x, y = float(p[0]), float(p[1])
        if self.flip:
            y = -y
        return np.array([self.c * x - self.s * y + self.tx, self.s * x + self.c * y + self.ty])

    def side(self, s: Side, fid: str, index: int) -> Side:
        T = self
        if s.kind == "line":
            return Side(fid, index, "line", T(s.p0), T(s.p1))
        if s.kind == "arc":
            c = T(s.centre)
            p0 = T(s.p0)
            a0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
            sweep = -s.sweep if self.flip else s.sweep
            return Side(fid, index, "arc", p0, T(s.p1), centre=c, radius=s.radius, a0=a0,
                        sweep=sweep, bulge=math.tan(sweep / 4.0))
        if s.kind == "cubic":
            return Side(fid, index, "cubic", T(s.p0), T(s.p1), c1=T(s.c1), c2=T(s.c2))
        if s.kind == "spline":                       # centripetal CR is unchanged by motions
            return Side(fid, index, "spline", T(s.p0), T(s.p1), q0=T(s.q0), q3=T(s.q3))
        pts = s.pts if s.kind == "poly" else np.vstack([s.at(np.linspace(0, 1, 65))])
        moved = np.array([T(p) for p in pts])
        return Side(fid, index, "poly", moved[0], moved[-1], pts=moved)


def scatter_strip_sides(f: Feature, k: int, item) -> list[Side]:
    """A scatter's channel inclusion (cx, cy, r, angle): a straight strip of length
    2r and width 2r over the scatter's aspect, turned by the angle; four sides."""
    from .resolve import SCATTER_DEFAULTS
    cx, cy, r, a = (float(t) for t in item[:4])
    aspect = float({**SCATTER_DEFAULTS, **f.params}.get("aspect", 8.0))
    u = np.array([math.cos(a), math.sin(a)]) * r
    v = np.array([-math.sin(a), math.cos(a)]) * (r / max(aspect, 1.0))
    c = np.array([cx, cy])
    pts = [c - u - v, c + u - v, c + u + v, c - u + v]
    return [Side(f.id, 4 * k + j, "line", pts[j], pts[(j + 1) % 4]) for j in range(4)]


def channel_sides(f: Feature, tol: float) -> list[Side]:
    """A channel: its centre line offset by half its width each way, with straight ends
    (an inlet and an outlet): four sides."""
    centre = Feature(id=f.id, kind="path", role="guide", points=f.points, edges=f.edges,
                     bulge=f.bulge, handles=f.handles, closed=False)
    parts = [s.samples(tol)[0] for s in feature_sides(centre)]
    line = np.vstack(parts + [np.asarray(f.points[-1], float)[None, :]])
    keep = np.ones(len(line), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(line, axis=0)) > 1e-15, axis=1)
    line = line[keep]
    if len(line) < 2:
        raise GeometryError(f"{f.label}: its centre line has no length")
    d = np.gradient(line, axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-300)[:, None]
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    left = line + 0.5 * f.w * nrm
    right = line - 0.5 * f.w * nrm
    return [Side(f.id, 0, "poly", right[0], right[-1], pts=right),
            Side(f.id, 1, "line", right[-1], left[-1]),
            Side(f.id, 2, "poly", left[-1], left[0], pts=left[::-1].copy()),
            Side(f.id, 3, "line", left[0], right[0])]


def feature_sides(f: Feature) -> list[Side]:
    """A plain feature's closed ring of sides, in drawing order."""
    if f.kind == "rect":
        x, y, w, h, r = f.x, f.y, f.w, f.h, min(f.r, 0.5 * min(f.w, f.h))
        if r <= 1e-15:
            c = [np.array(p, float) for p in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
            return [Side(f.id, k, "line", c[k], c[(k + 1) % 4]) for k in range(4)]
        out: list[Side] = []
        pi2 = math.pi / 2

        def line(a, b):
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-12 * max(w, h):
                out.append(Side(f.id, len(out), "line", np.array(a, float), np.array(b, float)))
        line((x + r, y), (x + w - r, y))
        out.append(_circle_arc(f.id, len(out), (x + w - r, y + r), r, -pi2, pi2))
        line((x + w, y + r), (x + w, y + h - r))
        out.append(_circle_arc(f.id, len(out), (x + w - r, y + h - r), r, 0.0, pi2))
        line((x + w - r, y + h), (x + r, y + h))
        out.append(_circle_arc(f.id, len(out), (x + r, y + h - r), r, pi2, pi2))
        line((x, y + h - r), (x, y + r))
        out.append(_circle_arc(f.id, len(out), (x + r, y + r), r, math.pi, pi2))
        for k, s in enumerate(out):
            s.index = k
        return out
    if f.kind == "ellipse":
        c = np.array((f.cx, f.cy), float)
        if abs(f.rx - f.ry) <= 1e-12 * max(f.rx, f.ry):
            return [_circle_arc(f.id, k, c, f.rx, k * math.pi / 2, math.pi / 2) for k in range(4)]
        out = []
        for k in range(4):
            a0 = k * math.pi / 2
            p0 = c + np.array([f.rx * math.cos(a0), f.ry * math.sin(a0)])
            a1 = a0 + math.pi / 2
            p1 = c + np.array([f.rx * math.cos(a1), f.ry * math.sin(a1)])
            out.append(Side(f.id, k, "ell", p0, p1, centre=c, a0=a0, sweep=math.pi / 2,
                            rx=f.rx, ry=f.ry))
        return out
    if f.kind == "path":
        pts = [np.array(p, float) for p in f.points]
        n = len(pts)
        m = n if f.closed else n - 1
        out = []
        for k in range(m):
            a, b = pts[k], pts[(k + 1) % n]
            kind = f.edges[k]
            if kind == "arc":
                out.append(_arc_side(f.id, k, a, b, f.bulge[k]))
            elif kind == "cubic":
                h = f.handles[k]
                out.append(Side(f.id, k, "cubic", a, b, c1=np.array(h[:2], float),
                                c2=np.array(h[2:], float)))
            elif kind == "spline" and n >= 3:
                q0 = pts[(k - 1) % n] if (f.closed or k > 0) else a
                q3 = pts[(k + 2) % n] if (f.closed or k + 2 < n) else b
                out.append(Side(f.id, k, "spline", a, b, q0=q0, q3=q3))
            else:
                out.append(Side(f.id, k, "line", a, b))
        return out
    raise GeometryError(f"{f.label}: a {f.kind} has no sides of its own")


# ---------------------------------------------------------------------------
# the tree as polygons, with every segment traceable to its side
# ---------------------------------------------------------------------------


@dataclass
class Tagged:
    """Every input segment, for tracing a boolean's output back to its sides."""
    sides: dict[tuple[str, int], Side] = field(default_factory=dict)
    seg_side: list[tuple[str, int]] = field(default_factory=list)
    seg_t: list[tuple[float, float]] = field(default_factory=list)
    lines: list[LineString] = field(default_factory=list)
    tree: object = None
    rings: dict[str, list[Side]] = field(default_factory=dict)

    def add(self, side: Side, tol: float) -> np.ndarray:
        self.sides[(side.fid, side.index)] = side
        pts, ts = side.samples(tol)
        full = np.vstack([pts, side.p1[None, :]])
        tt = np.append(ts, 1.0)
        for i in range(len(pts)):
            self.lines.append(LineString([full[i], full[i + 1]]))
            self.seg_side.append((side.fid, side.index))
            self.seg_t.append((float(tt[i]), float(tt[i + 1])))
        return pts

    def build(self):
        self.tree = shapely.STRtree(self.lines)
        self.fids = np.array([s[0] for s in self.seg_side], dtype=object)
        self._trees: dict[frozenset, tuple] = {}

    def tree_for(self, allowed: frozenset):
        """An STR-tree over the segments of the allowed features only: a ring is traced
        to the sides that build it, never to another feature's side lying on it."""
        if allowed not in self._trees:
            keep = np.flatnonzero(np.isin(self.fids, list(allowed)))
            self._trees[allowed] = (shapely.STRtree([self.lines[i] for i in keep]), keep)
        return self._trees[allowed]


def _ring_polygon(sides: list[Side], tagged: Tagged, tol: float, label: str,
                  notes: list[dict]) -> Polygon:
    pts = np.vstack([tagged.add(s, tol) for s in sides])
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-15, axis=1)
    pts = pts[keep]
    if len(pts) < 3:
        raise GeometryError(f"{label} encloses no area")
    poly = Polygon(pts)
    if not poly.is_valid:
        notes.append({"severity": "error", "kind": "crossing", "feature": sides[0].fid,
                      "message": f"{label}'s outline crosses itself"})
        poly = shapely.make_valid(poly)
        polys = [g for g in getattr(poly, "geoms", [poly]) if isinstance(g, Polygon)]
        poly = unary_union(polys) if polys else Polygon()
    return poly


class Evaluator:
    """The tree's shapes as Shapely geometry, every segment traceable to its side."""

    def __init__(self, geom: Geometry, tol: float):
        self.geom = geom
        self.tol = tol
        self.tagged = Tagged()
        self.notes: list[dict] = []
        self.by_id = geom.by_id()
        self.shapes: dict[str, object] = {}
        consumed = geom.consumed()
        self.top = [f for f in geom.features if f.id not in consumed and not f.hidden]
        for f in geom.features:
            self._shape(f)
        self.tagged.build()

    def _shape(self, f: Feature):
        if f.id in self.shapes:
            return self.shapes[f.id]
        if f.kind == "combine":
            parts = [self._shape(self.by_id[i]) for i in f.inputs]
            g = parts[0]
            for p in parts[1:]:
                if f.op == "union":
                    g = g.union(p)
                elif f.op == "subtract":
                    g = g.difference(p)
                else:
                    g = g.intersection(p)
            g = _polygonal(g)
        elif f.kind == "scatter":
            parts = []
            for k, item in enumerate(f.items):
                if len(item) >= 4:                   # a channel: a strip at an angle
                    sides = scatter_strip_sides(f, k, item)
                else:
                    cx, cy, r = item[:3]
                    sides = [_circle_arc(f.id, 4 * k + j, (cx, cy), r, j * math.pi / 2,
                                         math.pi / 2) for j in range(4)]
                parts.append(_ring_polygon(sides, self.tagged, self.tol, f.label, self.notes))
            g = _polygonal(unary_union(parts)) if parts else Polygon()
        elif f.kind in ("pattern", "mirror"):
            src = self.by_id[f.inputs[0]]
            base = self._shape(src)
            places = placements(f, base)
            sides = self.tagged.rings.get(src.id)
            parts = [] if f.kind == "pattern" else [base]
            for k, T in enumerate(places):
                if sides is not None:
                    moved = [T.side(s, f.id, k * len(sides) + j) for j, s in enumerate(sides)]
                    parts.append(_ring_polygon(moved, self.tagged, self.tol, f.label, self.notes))
                else:                                 # a combined input: its outline, moved
                    parts.append(_moved_polygon(base, T))
            g = _polygonal(unary_union(parts)) if parts else Polygon()
        elif f.kind == "channel":
            sides = channel_sides(f, self.tol)
            self.tagged.rings[f.id] = sides
            g = _ring_polygon(sides, self.tagged, self.tol, f.label, self.notes)
        elif f.kind == "path" and not f.closed:
            g = Polygon()                                    # a guide line encloses nothing
        else:
            sides = feature_sides(f)
            self.tagged.rings[f.id] = sides
            g = _ring_polygon(sides, self.tagged, self.tol, f.label, self.notes)
        self.shapes[f.id] = g
        return g

    def of_role(self, role: str) -> list[Feature]:
        return [f for f in self.top if f.role == role]

    def domain(self):
        adds = [self.shapes[f.id] for f in self.of_role("add")]
        cuts = [self.shapes[f.id] for f in self.of_role("cut")]
        d = unary_union(adds) if adds else Polygon()
        if cuts:
            d = d.difference(unary_union(cuts))
        return _polygonal(d)

    # -- tracing a result ring back to its sides ---------------------------------
    def sources(self, fids) -> frozenset:
        """The features whose sides can lie on a ring built from ``fids``: those
        features and, through combines, their inputs."""
        out: set[str] = set()
        todo = list(fids)
        while todo:
            f = self.by_id.get(todo.pop())
            if f is None or f.id in out:
                continue
            out.add(f.id)
            if f.kind == "combine":
                todo += list(f.inputs)
        return frozenset(out)

    def pieces(self, coords: np.ndarray, allowed: frozenset) -> list[dict]:
        """A closed ring (its points, not repeated) as runs of segments from one side
        each: ``{"side": (fid, k) or None, "points": [...]}``."""
        n = len(coords)
        a = coords
        b = np.roll(coords, -1, axis=0)
        mids = shapely.points(0.5 * (a + b))
        tree, keep = self.tagged.tree_for(allowed)
        idx, dist = tree.query_nearest(mids, return_distance=True, all_matches=False)
        # query_nearest returns (input index, tree index) pairs
        seg = np.full(n, -1, dtype=int)
        seg[idx[0]] = keep[idx[1]]
        dd = np.full(n, np.inf)
        dd[idx[0]] = dist
        scale = max(1.0, float(np.max(np.abs(coords))))
        tags = [self.tagged.seg_side[s] if s >= 0 and dd[i] <= 1e-7 * scale else None
                for i, s in enumerate(seg)]
        # start at a change of side so no run is split across the ring's start
        start = 0
        for i in range(n):
            if tags[i] != tags[i - 1]:
                start = i
                break
        order = [(start + i) % n for i in range(n)]
        runs: list[dict] = []
        for i in order:
            if runs and runs[-1]["side"] == tags[i]:
                runs[-1]["points"].append(b[i])
            else:
                runs.append({"side": tags[i], "points": [a[i], b[i]]})
        return runs


def placements(f: Feature, base) -> list[Placement]:
    """Where a pattern puts its copies, or where a mirror puts its one."""
    p = f.params
    if f.kind == "mirror":
        axis = str(p.get("axis", "x"))
        at = float(p.get("at", 0.0))
        if axis == "x":                    # about a vertical line x = at
            return [Placement(math.pi, True, 2 * at, 0.0)]
        return [Placement(0.0, True, 0.0, 2 * at)]          # about y = at
    mode = str(p.get("mode", "line"))
    n = int(p.get("count", 3))
    if not 1 <= n <= 200:
        raise GeometryError(f"{f.label}: 1 to 200 copies")
    if mode == "line":
        dx, dy = float(p.get("dx", 0.0)), float(p.get("dy", 0.0))
        return [Placement(0.0, False, k * dx, k * dy) for k in range(n)]
    if mode == "grid":
        nx, ny = int(p.get("nx", n)), int(p.get("ny", 1))
        sx, sy = float(p.get("sx", 0.0)), float(p.get("sy", 0.0))
        return [Placement(0.0, False, i * sx, j * sy) for j in range(ny) for i in range(nx)]
    if mode == "circle":
        cx, cy = float(p.get("cx", 0.0)), float(p.get("cy", 0.0))
        step = math.radians(float(p.get("angle", 360.0 / n)))
        out = []
        for k in range(n):
            a = k * step
            c, s = math.cos(a), math.sin(a)
            out.append(Placement(a, False, cx - (c * cx - s * cy), cy - (s * cx + c * cy)))
        return out
    raise GeometryError(f"{f.label}: a pattern is a line, a circle or a grid")


def _moved_polygon(poly, T: Placement):
    from shapely import affinity
    g = poly
    if T.flip:
        g = affinity.scale(g, 1.0, -1.0, origin=(0, 0))
    g = affinity.affine_transform(g, [T.c, -T.s, T.s, T.c, T.tx, T.ty])
    return g


def _polygonal(g):
    """Only the polygons of a result (a boolean can leave lines and points behind)."""
    if g.is_empty:
        return Polygon()
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    polys = [p for p in getattr(g, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
    return unary_union(polys) if polys else Polygon()


# ---------------------------------------------------------------------------
# engine outlines
# ---------------------------------------------------------------------------


def _edges_of_run(ev: Evaluator, run: dict, edges: list):
    """One run of segments from one side, as engine edges (world units): a line or an
    arc with its bulge; a curve the engine has no edge for, as short lines."""
    pts = run["points"]
    a, b = np.asarray(pts[0], float), np.asarray(pts[-1], float)
    side = ev.tagged.sides.get(run["side"]) if run["side"] else None
    if side is not None and side.kind == "line":
        edges.append((a, "line", 0.0))
        return
    if side is not None and side.kind == "arc":
        # the run's sweep, from where its ends sit on the circle (each within half a
        # turn of the arc's middle, so an arc of up to a full turn reads right)
        t0 = side.t_of(a, 0.5)
        t1 = side.t_of(b, 0.5)
        sweep = side.sweep * (t1 - t0)
        if abs(sweep) < 1e-12:
            edges.append((a, "line", 0.0))
            return
        edges.append((a, "arc", math.tan(sweep / 4.0)))
        return
    for p in pts[:-1]:
        edges.append((np.asarray(p, float), "line", 0.0))


def _intact(ev: Evaluator, runs: list[dict]) -> str | None:
    """The feature whose whole ring this is, every side whole and in order, else None."""
    fids = {r["side"][0] for r in runs if r["side"]}
    if len(fids) != 1 or any(r["side"] is None for r in runs):
        return None
    fid = fids.pop()
    ring = ev.tagged.rings.get(fid)
    if ring is None or len(runs) != len(ring):
        return None
    ks = [r["side"][1] for r in runs]
    n = len(ring)
    if sorted(ks) != list(range(n)):
        return None
    fwd = all((ks[(i + 1) % n] - ks[i]) % n == 1 for i in range(n))
    back = all((ks[i] - ks[(i + 1) % n]) % n == 1 for i in range(n))
    if not (fwd or back):
        return None
    for r in runs:                    # whole: the run starts and ends at the side's ends
        s = ev.tagged.sides[r["side"]]
        ends = {tuple(np.round(s.p0, 12)), tuple(np.round(s.p1, 12))}
        got = {tuple(np.round(np.asarray(r["points"][0], float), 12)),
               tuple(np.round(np.asarray(r["points"][-1], float), 12))}
        if got != ends:
            return None
    return fid


def _exact_edges(ev: Evaluator, fid: str, tol: float) -> list:
    """A whole feature's ring as drawn: its own sides, in its own order."""
    out = []
    for s in ev.tagged.rings[fid]:
        if s.kind == "line":
            out.append((s.p0, "line", 0.0))
        elif s.kind == "arc":
            out.append((s.p0, "arc", s.bulge if s.bulge else math.tan(s.sweep / 4.0)))
        elif s.kind == "spline":
            out.append((s.p0, "spline", 0.0))
        elif s.kind == "poly":
            for p in s.pts[:-1]:
                out.append((p, "line", 0.0))
        else:
            pts, _t = s.samples(tol)
            for p in pts:
                out.append((p, "line", 0.0))
    return out


def ring_edges(ev: Evaluator, coords: np.ndarray, tol: float,
               allowed: frozenset) -> tuple[list, list]:
    """A result ring as engine edges (world units), and each edge's side."""
    runs = ev.pieces(coords, allowed)
    fid = _intact(ev, runs)
    if fid is not None:
        edges = _exact_edges(ev, fid, tol)
        prov = []
        for s in ev.tagged.rings[fid]:
            n = (1 if s.kind in ("line", "arc", "spline") else len(s.pts) - 1
                 if s.kind == "poly" else len(s.samples(tol)[0]))
            prov += [(s.fid, s.index)] * n
        return edges, prov
    edges, prov = [], []
    for r in runs:
        before = len(edges)
        _edges_of_run(ev, r, edges)
        prov += [r["side"]] * (len(edges) - before)
    # drop edges too short to matter (a boolean's near-coincident vertices)
    keep_e, keep_p = [], []
    for i, e in enumerate(edges):
        nxt = edges[(i + 1) % len(edges)][0]
        if np.hypot(*(nxt - e[0])) > 1e-9 * max(1.0, tol * 64) or len(edges) <= 3:
            keep_e.append(e)
            keep_p.append(prov[i])
    return _merge_collinear(keep_e, keep_p)


def _merge_collinear(edges: list, prov: list) -> tuple[list, list]:
    """Consecutive straight edges on one line are one edge.  Where two shapes' sides
    overlap (two rectangles sharing a bottom), the traced segments alternate between
    the sides, and a straight side would otherwise come back in pieces."""
    if len(edges) <= 3:
        return edges, prov
    pts = [e[0] for e in edges]
    scale = max(1e-300, float(np.max(np.abs(np.asarray(pts)))))
    changed = True
    while changed and len(edges) > 3:
        changed = False
        n = len(edges)
        for i in range(n):
            j = (i + 1) % n
            if edges[i][1] != "line" or edges[j][1] != "line":
                continue
            a, b, c = edges[i][0], edges[j][0], edges[(j + 1) % n][0]
            u, v = b - a, c - b
            cross = float(u[0] * v[1] - u[1] * v[0])
            if abs(cross) <= 1e-12 * scale * scale and float(u @ v) > 0:
                del edges[j]
                del prov[j]
                changed = True
                break
    return edges, prov


def to_outline(edges: list, dx: float) -> Outline:
    return Outline(points=[(float(p[0] / dx), float(p[1] / dx)) for p, _k, _b in edges],
                   edges=[k for _p, k, _b in edges],
                   bulge=[float(b) for _p, _k, b in edges])


# ---------------------------------------------------------------------------
# the whole evaluation
# ---------------------------------------------------------------------------


@dataclass
class Result:
    domain: Domain | None
    regions: list[Region]
    notes: list[dict]
    shape: object                       # the domain, Shapely, world units
    provenance: dict[str, list]         # "outline" / "hole<k>" -> per engine edge, its side
    materials: dict[str, dict[str, float]]
    #: the feature each of ``regions`` was made from, in order
    region_features: list[str] = field(default_factory=list)
    evaluator: object = None
    #: per coefficient field: its property, its range over the cells, its levels
    fields: dict = field(default_factory=dict)


def default_material(spec: CaseSpec) -> str | None:
    lib = registry.family(spec.physics.family).material_library()
    if spec.regions:
        return spec.regions[0].material
    if spec.materials:
        return next(iter(spec.materials))
    return next(iter(lib), None)


def evaluate(geom: Geometry, spec: CaseSpec, dx: float | None = None,
             keep: tuple[int, int] | None = None) -> Result:
    """The tree as the engine's domain and regions, at cell size ``dx`` (the case's own
    by default).  ``keep``: the grid never shrinks below these many cells (a case's own
    margins stay; None fits the grid to the shape).  Never raises for a person's
    drawing: what cannot be built is a note with severity ``error`` and ``domain`` None."""
    dx = float(dx or spec.domain.dx)
    tol = dx * TOL_CELLS
    notes: list[dict] = []
    try:
        ev = Evaluator(geom, tol)
    except GeometryError as exc:
        return Result(None, [], [{"severity": "error", "kind": "shape", "message": str(exc)}],
                      Polygon(), {}, {})
    notes += ev.notes
    if not ev.of_role("add"):
        notes.append({"severity": "error", "kind": "empty",
                      "message": "No shape yet: pick a start, draw, or import"})
        return Result(None, [], notes, Polygon(), {}, {})
    dom = ev.domain()
    if dom.is_empty:
        notes.append({"severity": "error", "kind": "empty",
                      "message": "the cuts remove the whole domain"})
        return Result(None, [], notes, dom, {}, {})
    parts = list(getattr(dom, "geoms", [dom]))
    if len(parts) > 1:
        parts.sort(key=lambda p: -p.area)
        notes.append({"severity": "error", "kind": "pieces",
                      "message": f"the domain is in {len(parts)} separate pieces: join them, "
                                 f"or remove all but one"})
    main = orient(parts[0], 1.0)
    x0, y0, x1, y1 = main.bounds
    if x0 < -1e-9 * max(1.0, abs(x1)) or y0 < -1e-9 * max(1.0, abs(y1)):
        notes.append({"severity": "error", "kind": "origin", "fix": "origin",
                      "message": "part of the domain lies left of or below the origin, where "
                                 "the grid has no cells"})
    nx = max(1, int(math.ceil(x1 / dx - 1e-7)))
    ny = max(1, int(math.ceil(y1 / dx - 1e-7)))
    if keep is not None:
        nx, ny = max(nx, int(keep[0])), max(ny, int(keep[1]))
    W, H = nx * dx, ny * dx
    prov: dict[str, list] = {}
    full = box(0.0, 0.0, W, H)
    outer = Polygon(main.exterior)
    is_grid = (abs(outer.area - full.area) <= 1e-9 * full.area
               and outer.symmetric_difference(full).area <= 1e-9 * full.area)
    holes_all = [np.asarray(h.coords[:-1], float) for h in main.interiors]
    domain_src = ev.sources(f.id for f in ev.of_role("add") + ev.of_role("cut"))
    if is_grid:
        outline = None
    else:
        e, p = ring_edges(ev, np.asarray(main.exterior.coords[:-1], float), tol, domain_src)
        outline = to_outline(e, dx)
        prov["outline"] = p
    holes = []
    for k, ring in enumerate(holes_all):
        e, p = ring_edges(ev, ring, tol, domain_src)
        holes.append(to_outline(e, dx))
        prov[f"hole{k}"] = p
    domain = Domain(nx=nx, ny=ny, dx=dx, outline=outline, holes=holes)

    # -- regions: what each adding shape is made of, then the regions in order --------
    fam = registry.family(spec.physics.family)
    regions: list[Region] = []
    region_fids: list[str] = []
    mats: dict[str, dict[str, float]] = {}
    fields: dict[str, dict] = {}
    if "regions" in fam.layers or spec.regions:
        lib = fam.material_library()
        base = default_material(spec)
        grid_box = box(0.0, 0.0, W, H)
        used_ids: set[str] = set()
        for f in ev.top:
            if f.role not in ("add", "region"):
                continue
            m = f.material or (base if f.role == "add" else None)
            if m is None:
                notes.append({"severity": "warning", "kind": "material", "feature": f.id,
                              "message": f"{f.label} has no material"})
                continue
            if m not in spec.materials and m in lib:
                mats[m] = dict(lib[m])
            shp = ev.shapes[f.id]
            if shp.is_empty:
                continue
            rid = slug(f.name) if f.name else f.id
            if rid in used_ids or not rid:
                rid = f.id
            used_ids.add(rid)
            reg = _region(ev, f, shp, grid_box, rid, m, dx, tol, nx, ny)
            if reg is None:
                continue
            if f.field and f.field.get("kind") in ("formula", "image"):
                props = dict(spec.materials.get(m) or lib.get(m) or {})
                try:
                    regs, more, info = field_regions(f, reg, domain, props, geom)
                except Exception as exc:                       # a person's formula
                    notes.append({"severity": "error", "kind": "field", "feature": f.id,
                                  "message": f"{f.label}'s field: {exc}"})
                    continue
                regions += regs
                region_fids += [f.id] * len(regs)
                mats.update(more)
                fields[f.id] = info
                continue
            regions.append(reg)
            region_fids.append(f.id)
    _notes_inside(ev, main, notes)
    res = Result(domain, regions, notes, dom, prov, mats, region_fids, ev)
    res.fields = fields
    return res


#: a coefficient field becomes at most this many materials, log-spaced
FIELD_LEVELS = 16


def field_cells(reg: Region, domain: Domain):
    """The cells a region holds in the domain: their row and column indices."""
    mask = wgeo.region_mask(reg, domain.nx, domain.ny) & wgeo.domain_mask(domain)
    return np.nonzero(mask)


def image_values(field: dict, x, y):
    """An image field at points (x, y): the grey level of the pixel each point falls
    in (nearest, row 0 at the top), mapped onto [min, max]: linearly, or on a log
    scale when ``scale`` is "log" (one decade per equal step of grey).  Points
    outside the image's box take the nearest edge pixel."""
    import base64 as _b64
    g = field.get("grid") or {}
    try:
        nx, ny = int(g["nx"]), int(g["ny"])
        raw = np.frombuffer(_b64.b64decode(g["data"], validate=True), dtype=np.uint8)
        x0, y0, x1, y1 = (float(t) for t in field["box"])
        lo, hi = float(field["min"]), float(field["max"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("an image field needs its grid (nx, ny, data), its box and a "
                         "min and max") from None
    if not (1 <= nx <= 1024 and 1 <= ny <= 1024) or raw.size != nx * ny:
        raise ValueError(f"the image grid is {nx} x {ny} but holds {raw.size} values")
    if not (x1 > x0 and y1 > y0):
        raise ValueError("the image's box has no area")
    grey = raw.reshape(ny, nx).astype(float) / 255.0
    i = np.clip(np.floor((np.asarray(x) - x0) / (x1 - x0) * nx).astype(int), 0, nx - 1)
    j = np.clip(np.floor((y1 - np.asarray(y)) / (y1 - y0) * ny).astype(int), 0, ny - 1)
    t = grey[j, i]
    if field.get("scale") == "log":
        if not (lo > 0 and hi > 0):
            raise ValueError("a log scale needs min and max above 0")
        return lo * (hi / lo) ** t
    return lo + (hi - lo) * t


def field_values(f: Feature, x, y, geom: Geometry):
    """A coefficient field at points (x, y): a formula with the tree's parameters, or
    an image's grey levels (`image_values`)."""
    from . import formula
    if f.field.get("kind") == "image":
        return np.asarray(image_values(f.field, x, y), dtype=float)
    expr = f.field.get("expr") or ""
    tree = formula.parse(expr)
    env = {p.name: float(p.value) for p in geom.parameters}
    bad = formula.names(tree) - set(env) - {"x", "y"} - set(formula.CONSTANTS)
    if bad:
        name = sorted(bad)[0]
        raise formula.FormulaError("T is not a variable of any kind that runs here: the "
                                   "properties are constant in time" if name == "T" else
                                   f"no parameter called {name!r}")
    v = formula.evaluate(tree, {**env, "x": x, "y": y})
    return np.broadcast_to(np.asarray(v, dtype=float), np.shape(x)).copy()


def field_regions(f: Feature, reg: Region, domain: Domain, props: dict, geom: Geometry):
    """A region whose ``prop`` is a formula of position, as the engine can run it: the
    formula at every cell centre, binned into at most `FIELD_LEVELS` levels (log-spaced
    when every value is positive), each level a material whose ``prop`` is the mean of
    its cells' values, and each level's cells a polygon region (cell-exact, so the
    engine's mask is exactly those cells).  Returns (regions, materials, the range)."""
    from shapely.geometry import box as _box
    prop = f.field.get("prop")
    if props and prop not in props:
        raise ValueError(f"its material has no property {prop!r} (it has "
                         f"{', '.join(sorted(props))})")
    rows, cols = field_cells(reg, domain)
    if not len(rows):
        return [], {}, {"prop": prop, "min": None, "max": None, "levels": 0, "cells": 0}
    dx = domain.dx
    x = (cols + 0.5) * dx
    y = (rows + 0.5) * dx
    v = field_values(f, x, y, geom)
    if not np.all(np.isfinite(v)):
        raise ValueError("the formula is not finite at every cell centre")
    if np.any(v <= 0) and prop in ("k", "sigma", "rho", "cp", "E", "depth", "c"):
        raise ValueError(f"{prop} must be above 0 everywhere; the formula reaches {v.min():g}")
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo * (1 + 1e-12):
        idx = np.zeros(len(v), dtype=int)
        levels = 1
    else:
        levels = int(min(FIELD_LEVELS, len(np.unique(np.round(v, 12)))))
        if lo > 0:
            edges = np.geomspace(lo, hi, levels + 1)
        else:
            edges = np.linspace(lo, hi, levels + 1)
        idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, levels - 1)
    regs, mats = [], {}
    for b in range(levels):
        sel = idx == b
        if not sel.any():
            continue
        name = f"{f.id}-{b + 1}"
        mats[name] = {**props, prop: float(v[sel].mean())}
        # the level's cells, row by row, merged into polygons on cell boundaries
        rr, cc = rows[sel], cols[sel]
        boxes = []
        order = np.lexsort((cc, rr))
        rr, cc = rr[order], cc[order]
        start = 0
        for i in range(1, len(rr) + 1):
            if i == len(rr) or rr[i] != rr[i - 1] or cc[i] != cc[i - 1] + 1:
                boxes.append(_box(cc[start], rr[start], cc[i - 1] + 1, rr[start] + 1))
                start = i
        merged = unary_union(boxes)
        for k, poly in enumerate(getattr(merged, "geoms", [merged])):
            poly = orient(poly, 1.0)
            pts = [(float(a), float(c)) for a, c in poly.exterior.coords[:-1]]
            holes = [[(float(a), float(c)) for a, c in h.coords[:-1]] for h in poly.interiors]
            regs.append(Region(id=f"{reg.id}-{b + 1}-{k + 1}", material=name, shape="polygon",
                               points=pts, holes=holes or None))
    return regs, mats, {"prop": prop, "min": lo, "max": hi, "levels": len(mats),
                        "cells": int(len(v))}


def _region(ev: Evaluator, f: Feature, shp, grid_box, rid, material, dx, tol, nx, ny):
    """One feature as an engine region: a rectangle in whole cells as a rectangle, a
    plain shape inside the grid as its own outline, anything else as a polygon."""
    if f.kind == "rect" and f.r <= 0:
        c = [f.x / dx, f.y / dx, (f.x + f.w) / dx, (f.y + f.h) / dx]
        rc = [round(v) for v in c]
        if all(abs(v - r) < 1e-7 for v, r in zip(c, rc)):
            ix0, iy0 = max(0, rc[0]), max(0, rc[1])
            ix1, iy1 = min(nx, rc[2]), min(ny, rc[3])
            if ix1 > ix0 and iy1 > iy0:
                return Region(id=rid, material=material, shape="rect", x0=ix0, y0=iy0,
                              nx=ix1 - ix0, ny=iy1 - iy0)
            return None
    clipped = _polygonal(shp.intersection(grid_box))
    if clipped.is_empty:
        return None
    polys = sorted(getattr(clipped, "geoms", [clipped]), key=lambda p: -p.area)
    p0 = orient(polys[0], 1.0)
    if f.kind != "combine" and not p0.interiors and len(polys) == 1:
        e, _p = ring_edges(ev, np.asarray(p0.exterior.coords[:-1], float), tol,
                           ev.sources([f.id]))
        return Region(id=rid, material=material, shape="curve", outline=to_outline(e, dx))
    pts = [(float(x / dx), float(y / dx)) for x, y in p0.exterior.coords[:-1]]
    holes = [[(float(x / dx), float(y / dx)) for x, y in h.coords[:-1]] for h in p0.interiors]
    return Region(id=rid, material=material, shape="polygon", points=pts, holes=holes or None)


def _notes_inside(ev: Evaluator, dom, notes: list[dict]) -> None:
    """Cuts that miss the domain, regions that leave it."""
    for f in ev.top:
        shp = ev.shapes[f.id]
        if shp.is_empty:
            continue
        if f.role == "cut" and not shp.intersects(dom.buffer(0)) and not dom.is_empty:
            inside_any = any(shp.intersects(ev.shapes[a.id]) for a in ev.of_role("add"))
            if not inside_any:
                notes.append({"severity": "warning", "kind": "outside", "feature": f.id,
                              "message": f"{f.label} cuts nothing: it is outside the domain"})
        if f.role == "region":
            out = shp.difference(dom).area
            if out > 1e-9 * max(shp.area, 1e-300):
                notes.append({"severity": "info", "kind": "outside", "feature": f.id,
                              "message": f"{f.label} reaches outside the domain; only its "
                                         f"inside counts"})


def apply(spec: CaseSpec, geom: Geometry, dx: float | None = None,
          keep: tuple[int, int] | None = None) -> tuple[CaseSpec, Result]:
    """A copy of ``spec`` whose domain and regions are the tree's (when it has a
    domain); the result's notes say what could not be built."""
    res = evaluate(geom, spec, dx, keep)
    new = spec.copy_deep()
    if res.domain is not None:
        new.domain = res.domain
        new.regions = res.regions
        # a field's levels are materials of its own, made again on every evaluation
        field_ids = {f.id for f in geom.features}
        for name in list(new.materials):
            fid, _, k = name.rpartition("-")
            if fid in field_ids and k.isdigit():
                del new.materials[name]
        for m, props in res.materials.items():
            fid, _, k = m.rpartition("-")
            if fid in field_ids and k.isdigit():
                new.materials[m] = props
            else:
                new.materials.setdefault(m, props)
    return new, res


# ---------------------------------------------------------------------------
# boundaries follow the sides they were set on
# ---------------------------------------------------------------------------


def edge_sides(prov: dict[str, list]) -> dict[str, tuple]:
    """Engine edge name (``outline:3``, ``hole0:2``) -> the side it came from."""
    out = {}
    for ring, sides in prov.items():
        for k, s in enumerate(sides):
            if s is not None:
                out[f"{ring}:{k}"] = tuple(s)
    return out


def remap_boundaries(boundaries: list, old_prov: dict, new_prov: dict) -> tuple[list, list[str]]:
    """Boundaries on drawn edges, moved to the edges of the same sides after the
    geometry changed.  A side cut into several edges carries the condition on each; a
    side that is gone loses it (said in the returned notes)."""
    old = edge_sides(old_prov)
    new_by_side: dict[tuple, list[str]] = {}
    for name, side in edge_sides(new_prov).items():
        new_by_side.setdefault(side, []).append(name)
    out, notes = [], []
    taken: set[str] = set()
    for b in boundaries:
        if not wgeo.DRAWN_EDGE.match(b.edge):
            out.append(b)
            continue
        side = old.get(b.edge)
        targets = [t for t in new_by_side.get(side, []) if t not in taken] if side else []
        if not targets:
            notes.append(f"the {b.kind} on {b.edge} was on a side that is gone")
            continue
        for j, t in enumerate(targets):
            nb = b.model_copy(deep=True)
            nb.edge = t
            if j:
                nb.id = f"{b.id}-{j + 1}"
            taken.add(t)
            out.append(nb)
    ids = [b.id for b in out]
    for i, b in enumerate(out):
        if ids.count(b.id) > 1:
            b.id = f"{b.id}-{i + 1}"
    return out, notes


__all__ = ["Evaluator", "GeometryError", "Result", "Side", "apply", "evaluate", "feature_sides",
           "remap_boundaries", "edge_sides", "TOL_CELLS"]
