"""What the page draws of a case's engine geometry, in the physics' length unit.

The case file keeps its shapes in cells (`spec.py`); the page draws in the
physics' own unit (metres for every family but the wind farm, whose unit is the
rotor diameter), so every coordinate here is cells times ``domain.dx``.  Curves
are sent as the rings the engine itself samples (`shapes.sample`, half a cell
apart): the page draws exactly the boundary the solver's cells are cut from.

Windows generated from the geometry (``shape = "cells"``) are sent as the outline
of their cells, merged with Shapely, so a window of ten thousand cells is a few
rings rather than ten thousand squares.
"""

from __future__ import annotations

import numpy as np

from atlas.workbench import geometry as geo
from atlas.workbench import shapes


def _ring(points, dx: float) -> list[list[float]]:
    a = np.asarray(points, dtype=float) * dx
    return [[round(float(x), 9), round(float(y), 9)] for x, y in a]


def outline_ring(o, dx: float) -> list[list[float]]:
    return _ring(shapes.sample(o.points, o.edges, o.bulge, shapes.STEP), dx)


def _box(x0, y0, nx, ny, dx: float) -> list[list[float]]:
    return _ring([(x0, y0), (x0 + nx, y0), (x0 + nx, y0 + ny), (x0, y0 + ny)], dx)


def _runs_rings(runs, dx: float) -> list[list[list[float]]]:
    """The outline of a set of cells given as row runs, merged."""
    import shapely
    from shapely.geometry import box
    if not runs:
        return []
    geom = shapely.unary_union([box(a, r, b, r + 1) for r, a, b in runs])
    polys = list(getattr(geom, "geoms", [geom]))
    rings = []
    for p in polys:
        rings.append(_ring(p.exterior.coords[:-1], dx))
        for h in p.interiors:
            rings.append(_ring(h.coords[:-1], dx))
    return rings


def region_rings(r, dx: float) -> list[list[list[float]]]:
    if r.shape == "curve":
        return [outline_ring(r.outline, dx)]
    if r.shape == "polygon":
        return [_ring(r.points, dx)] + [_ring(h, dx) for h in (r.holes or [])]
    return [_box(r.x0, r.y0, r.nx, r.ny, dx)]


def window_rings(w, dx: float) -> list[list[list[float]]]:
    if w.shape == "curve":
        return [outline_ring(w.outline, dx)] + [outline_ring(h, dx) for h in w.holes]
    if w.shape == "cells":
        return _runs_rings(w.runs, dx)
    return [_box(w.x0, w.y0, w.nx, w.ny, dx)]


def boundary_segments(spec) -> list[dict]:
    """Each boundary condition as the polyline it holds on."""
    d = spec.domain
    out = []
    for b in spec.boundaries:
        if b.drawn:
            m = geo.DRAWN_EDGE.match(b.edge)
            o = d.outline if m.group(1) == "outline" else (
                d.holes[int(m.group(2))] if m.group(2) is not None
                and int(m.group(2)) < len(d.holes) else None)
            k = int(m.group(3))
            if o is None or k >= len(o.points):
                continue
            pts = shapes.edge_points(o.points, o.edges, o.bulge, k, shapes.STEP)
            end = np.asarray(o.points[(k + 1) % len(o.points)], dtype=float)[None, :]
            out.append({"id": b.id, "kind": b.kind, "value": b.value, "edge": b.edge,
                        "line": _ring(np.vstack([pts, end]), d.dx)})
        else:
            stop = b.stop if b.stop is not None else geo.edge_length(b.edge, d.nx, d.ny)
            x0, y0, x1, y1 = geo.edge_segment_xy(b.edge, b.start, stop, d.nx, d.ny)
            out.append({"id": b.id, "kind": b.kind, "value": b.value, "edge": b.edge,
                        "line": _ring([(x0, y0), (x1, y1)], d.dx)})
    return out


def drawing(spec) -> dict:
    """The engine's geometry of ``spec``, for the page: the grid, the domain, its
    holes, the material regions in stacking order, the windows, the boundaries and
    the rotors."""
    d = spec.domain
    dx = float(d.dx)
    return {
        "dx": dx, "nx": d.nx, "ny": d.ny,
        "width": d.nx * dx, "height": d.ny * dx,
        "outline": outline_ring(d.outline, dx) if d.outline is not None else None,
        "holes": [outline_ring(h, dx) for h in d.holes],
        "regions": [{"id": r.id, "material": r.material, "rings": region_rings(r, dx)}
                    for r in spec.regions],
        "windows": [{"id": w.id, "rings": window_rings(w, dx)} for w in spec.windows],
        "boundaries": boundary_segments(spec),
        "devices": [{"id": v.id, "x": v.x, "y": v.y, "diameter": v.diameter,
                     "yaw": v.yaw_deg} for v in spec.devices],
    }


__all__ = ["drawing", "outline_ring", "region_rings", "window_rings", "boundary_segments"]
