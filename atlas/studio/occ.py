"""Gmsh's OpenCASCADE kernel: the tree's exact areas, and STEP import.

Gmsh 4.15 (GPL-2.0-or-later, an optional install) is CALLED here, never copied.  Its
kernel builds each feature from its true curves (lines, circle arcs, ellipses, Bezier
curves), applies the booleans exactly, and measures the result.  The check compares
these areas with the area the solver's cells hold (`geocheck.py`), so the difference
the page shows is the grid's, not the sampling's.

Gmsh is a process-wide state machine, so every call holds `_LOCK` (the workbench's
Gmsh import shares the same lock) and builds a fresh model.  Without Gmsh,
`available()` is False and the check says the areas are Shapely's.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from atlas.workbench.gmsh_import import _LOCK

from .document import Feature, Geometry


def available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except (ImportError, OSError):            # Linux wheels can fail on libGLU
        return False


class _Model:
    """A fresh Gmsh model for the duration of a `with` block, under the lock."""

    def __enter__(self):
        import gmsh
        _LOCK.acquire()
        self.gmsh = gmsh
        self.started = not gmsh.isInitialized()
        if self.started:
            # not interruptible: a SIGINT handler can only be set from the main thread,
            # and the server calls this from a worker (the workbench's import does too)
            gmsh.initialize(readConfigFiles=False, interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("atlas-studio")
        return gmsh

    def __exit__(self, *exc):
        try:
            self.gmsh.model.remove()
            if self.started:
                self.gmsh.finalize()
        finally:
            _LOCK.release()
        return False


def _point(occ, cache, p):
    key = (round(float(p[0]), 12), round(float(p[1]), 12))
    if key not in cache:
        cache[key] = occ.addPoint(float(p[0]), float(p[1]), 0.0)
    return cache[key]


def _surface(occ, f: Feature, by_id) -> list[tuple[int, int]]:
    """One feature as OCC surfaces (dimTags)."""
    if f.kind == "rect":
        return [(2, occ.addRectangle(f.x, f.y, 0.0, f.w, f.h,
                                     roundedRadius=min(f.r, 0.5 * min(f.w, f.h))))]
    if f.kind == "ellipse":
        if f.rx >= f.ry:
            return [(2, occ.addDisk(f.cx, f.cy, 0.0, f.rx, f.ry))]
        tag = occ.addDisk(f.cx, f.cy, 0.0, f.ry, f.rx)
        occ.rotate([(2, tag)], f.cx, f.cy, 0.0, 0.0, 0.0, 1.0, math.pi / 2)
        return [(2, tag)]
    if f.kind == "path":
        from . import geometry_eval as GE
        cache: dict = {}
        curves = []
        for s in GE.feature_sides(f):
            a, b = _point(occ, cache, s.p0), _point(occ, cache, s.p1)
            if s.kind == "line":
                curves.append(occ.addLine(a, b))
            elif s.kind == "arc":
                c = occ.addPoint(float(s.centre[0]), float(s.centre[1]), 0.0)
                # a circle arc is drawn the short way: split one of more than half a turn
                if abs(s.sweep) < math.pi * 0.999:
                    curves.append(occ.addCircleArc(a, c, b))
                else:
                    m = s.at(0.5)[0]
                    mt = _point(occ, cache, m)
                    curves.append(occ.addCircleArc(a, c, mt))
                    curves.append(occ.addCircleArc(mt, c, b))
            elif s.kind == "cubic":
                c1 = occ.addPoint(float(s.c1[0]), float(s.c1[1]), 0.0)
                c2 = occ.addPoint(float(s.c2[0]), float(s.c2[1]), 0.0)
                curves.append(occ.addBezier([a, c1, c2, b]))
            else:                                   # a spline: through its own samples
                pts, _t = s.samples(1e-6 * max(1.0, float(np.max(np.abs(s.p0)))))
                tags = [a] + [occ.addPoint(float(x), float(y), 0.0) for x, y in pts[1:]] + [b]
                curves.append(occ.addSpline(tags))
        loop = occ.addCurveLoop(curves)
        return [(2, occ.addPlaneSurface([loop]))]
    if f.kind == "combine":
        parts = [_surface(occ, by_id[i], by_id) for i in f.inputs]
        base = parts[0]
        for p in parts[1:]:
            if f.op == "union":
                base, _ = occ.fuse(base, p)
            elif f.op == "subtract":
                base, _ = occ.cut(base, p)
            else:
                base, _ = occ.intersect(base, p)
        return base
    if f.kind == "scatter":
        from .generators import scatter_features
        out = []
        for g in scatter_features(f):
            out += _surface(occ, g, by_id)
        return out
    return []


def _area(gmsh, dimtags) -> float:
    gmsh.model.occ.synchronize()
    return float(sum(gmsh.model.occ.getMass(d, t) for d, t in dimtags if d == 2))


def areas(geom: Geometry) -> dict:
    """Exact areas: the domain, and each region feature inside it.  Raises when Gmsh
    is not installed (check `available()` first)."""
    by_id = geom.by_id()
    consumed = geom.consumed()
    top = [f for f in geom.features if f.id not in consumed and not f.hidden]
    out: dict = {"domain": 0.0, "features": {}}
    with _Model() as gmsh:
        occ = gmsh.model.occ
        adds = [dt for f in top if f.role == "add" for dt in _surface(occ, f, by_id)]
        cuts = [dt for f in top if f.role == "cut" for dt in _surface(occ, f, by_id)]
        if not adds:
            return out
        dom = adds
        if len(adds) > 1:
            dom, _ = occ.fuse(adds[:1], adds[1:])
        if cuts:
            dom, _ = occ.cut(dom, cuts)
        out["domain"] = _area(gmsh, dom)
        for f in top:
            if f.role not in ("region", "add"):
                continue
            surf = _surface(occ, f, by_id)
            inside, _ = occ.intersect(surf, dom, removeObject=True, removeTool=False)
            out["features"][f.id] = _area(gmsh, inside)
    return out


def import_step(data: bytes, name: str = "upload.step") -> list[dict]:
    """A STEP file's faces as closed paths (world units, the file's own): each face's
    outer boundary and its holes.  A solid is cut at half its height and the section
    taken (plan section 3.6).  Lines stay lines and circle arcs become arcs; every
    other curve is sampled."""
    out: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, os.path.basename(name) or "upload.step")
        with open(path, "wb") as fh:
            fh.write(data)
        with _Model() as gmsh:
            occ = gmsh.model.occ
            # STEP is usually in millimetres; the studio's unit is the metre
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")
            ents = occ.importShapes(path)
            occ.synchronize()
            vols = [t for d, t in ents if d == 3]
            if vols:
                x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(-1, -1)
                zc = 0.5 * (z0 + z1)
                big = max(x1 - x0, y1 - y0) * 4 + 1.0
                plane = occ.addRectangle(x0 - big, y0 - big, zc, (x1 - x0) + 2 * big,
                                         (y1 - y0) + 2 * big)
                faces, _ = occ.intersect([(3, v) for v in vols], [(2, plane)])
                occ.synchronize()
                faces = [t for d, t in faces if d == 2]
            else:
                faces = [t for d, t in ents if d == 2]
            if not faces:
                raise ValueError(f"{name} has no faces or solids")
            for face in faces:
                loops = []
                for _d, curve in gmsh.model.getBoundary([(2, face)], oriented=True):
                    loops.append(curve)
                rings = _rings(gmsh, loops)
                rings.sort(key=lambda r: -abs(_signed_area(r["points"])))
                if rings:
                    out.append({"outer": rings[0], "holes": rings[1:]})
    return out


def _signed_area(pts) -> float:
    p = np.asarray(pts, float)
    if len(p) < 3:
        return 0.0
    return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _rings(gmsh, curves) -> list[dict]:
    """Oriented boundary curves chained into closed rings of path edges."""
    segs = []
    for c in curves:
        tag = abs(c)
        kind = gmsh.model.getType(1, tag)
        lo, hi = gmsh.model.getParametrizationBounds(1, tag)
        ts = np.linspace(lo[0], hi[0], 65)
        xyz = np.asarray(gmsh.model.getValue(1, tag, ts), float).reshape(-1, 3)[:, :2]
        if c < 0:
            xyz = xyz[::-1]
        from atlas.workbench import shapes
        o = None
        if kind in ("Circle", "Ellipse"):
            # an ellipse with equal radii (a disk's edge in OpenCASCADE) is a circle; a
            # closed one's ends coincide, so its centre is found from points at thirds
            a, m, b = xyz[0], xyz[len(xyz) // 2], xyz[-1]
            if np.allclose(a, b):
                o = shapes.circumcentre(xyz[0], xyz[len(xyz) // 3], xyz[2 * len(xyz) // 3])
            else:
                o = shapes.circumcentre(a, m, b)
            if o is not None:
                rr = np.hypot(*(xyz - o).T)
                if np.ptp(rr) > 1e-9 * max(1.0, float(rr.mean())):
                    o = None
        if kind == "Line":
            segs.append([(xyz[0], "line", 0.0)])
        elif kind in ("Circle", "Ellipse"):
            a, m, b = xyz[0], xyz[len(xyz) // 2], xyz[-1]
            if o is None:
                segs.append([(p, "line", 0.0) for p in xyz[:-1]])
            else:
                ang = lambda p: math.atan2(p[1] - o[1], p[0] - o[0])     # noqa: E731
                sweep = 0.0
                prev = ang(xyz[0])
                for p in xyz[1:]:
                    cur = ang(p)
                    sweep += (cur - prev + math.pi) % (2 * math.pi) - math.pi
                    prev = cur
                if abs(sweep) > 1.5 * math.pi:            # a long arc: two edges
                    segs.append([(a, "arc", math.tan(sweep / 8.0)), (m, "arc", math.tan(sweep / 8.0))])
                else:
                    segs.append([(a, "arc", math.tan(sweep / 4.0))])
        else:
            segs.append([(p, "line", 0.0) for p in xyz[:-1]])
        segs[-1].append(("end", xyz[-1]))
    rings, cur = [], []
    start = None
    for s in segs:
        end = s[-1][1]
        body = s[:-1]
        if start is None:
            start = body[0][0]
        cur += body
        if np.hypot(*(np.asarray(end) - np.asarray(start))) < 1e-9 * max(1.0, float(np.abs(start).max())):
            rings.append({"points": [list(map(float, p)) for p, _k, _b in cur],
                          "edges": [k for _p, k, _b in cur],
                          "bulge": [float(b) for _p, _k, b in cur]})
            cur, start = [], None
    return rings


__all__ = ["available", "areas", "import_step"]
