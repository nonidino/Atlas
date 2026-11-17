"""The Check view: the geometry as the solver sees it, at a chosen grid (plan §3.5).

At ``n`` cells along the grid's longer side, the tree is evaluated again
(`geometry_eval.evaluate`) and the engine's own mask rules give each region's cells
(`atlas.workbench.geometry.domain_mask`, `region_owner`): what a grid solver actually
sees.  Beside them:

* **the checks**: every outline closed, none crossing itself, the holes and the
  regions inside the domain;
* **thin features**: a part of the domain or of a region narrower than one cell
  (found as what a morphological opening of half a cell removes);
* **slivers**: a gap narrower than one cell, which can vanish or cut off a bit;
* **near-touching**: two edges closer than one cell without meeting;
* **areas**: each region's exact area (Gmsh's OpenCASCADE kernel when installed, else
  Shapely's polygons) against the area of its cells, and the difference.

Each note carries where it is (for its numbered pin on the canvas) and the fixes the
page offers: a finer grid, widen to one cell (rectangles and ellipses), or keep.  No
note blocks a run unless `spec.check` refuses the case.
"""

from __future__ import annotations

import base64
import math

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from atlas.workbench import geometry as wgeo
from atlas.workbench.spec import CaseSpec

from . import geometry_eval as GE
from . import occ
from .document import Geometry

#: the grids the Check view offers
GRIDS = (32, 64, 128, 256)


def grid_n(spec: CaseSpec) -> int:
    """The case's own grid as cells along its longer side."""
    return max(spec.domain.nx, spec.domain.ny)


def dx_for(spec: CaseSpec, n: int) -> float:
    L = max(spec.domain.nx, spec.domain.ny) * spec.domain.dx
    return L / float(n)


def _parts(g):
    return [p for p in getattr(g, "geoms", [g]) if not p.is_empty]


def _opening_loss(shape, h: float, min_area: float):
    """What an opening by just under half a cell removes: the parts strictly narrower
    than one cell (a strip exactly one cell wide survives, as it should)."""
    if shape.is_empty:
        return []
    r = 0.49 * h
    opened = shape.buffer(-r, join_style="mitre").buffer(r, join_style="mitre")
    lost = shape.difference(opened)
    return [p for p in _parts(lost) if p.area > min_area]


def _width(p) -> float:
    """A thin part's width: twice its area over its perimeter (exact for a long strip)."""
    return 2.0 * p.area / max(p.length, 1e-300)


def check_view(spec: CaseSpec, geom: Geometry, n: int | None = None,
               page_areas: dict | None = None) -> dict:
    own = grid_n(spec)
    n = int(n or own)
    dx = spec.domain.dx if n == own else dx_for(spec, n)
    scale = spec.domain.dx / dx
    keep = (int(math.ceil(spec.domain.nx * scale - 1e-7)), int(math.ceil(spec.domain.ny * scale - 1e-7)))
    res = GE.evaluate(geom, spec, dx=dx, keep=keep)
    out: dict = {"type": "check", "n": n, "own_n": own, "grids": sorted(set(GRIDS) | {own}),
                 "dx": dx, "notes": [], "checks": {}, "areas": [], "kernel": None}
    notes = [dict(x) for x in res.notes]
    if res.domain is None:
        out["notes"] = notes
        out["checks"] = {"closed": not any(x["kind"] == "shape" for x in notes)}
        return out
    d = res.domain
    dom_mask = wgeo.domain_mask(d)
    owner = wgeo.region_owner(res.regions, d.nx, d.ny)
    raster = np.zeros((d.ny, d.nx), dtype=np.uint8)
    raster[dom_mask] = 1
    inside = dom_mask & (owner >= 0)
    raster[inside] = (np.minimum(owner[inside], 250) + 2).astype(np.uint8)
    out["cells"] = {"nx": d.nx, "ny": d.ny, "count": int(dom_mask.sum()),
                    "data": base64.b64encode(raster.tobytes()).decode("ascii"),
                    "legend": [{"value": k + 2, "region": r.id, "material": r.material,
                                "feature": res.region_features[k]}
                               for k, r in enumerate(res.regions)]}
    ev = res.evaluator
    D = res.shape
    h = dx
    tiny = 0.02 * h * h
    by_id = geom.by_id()

    def owner_of(part, roles=("region", "add")):
        best, area = None, 0.0
        for f in ev.top:
            if f.role in roles:
                a = ev.shapes[f.id].intersection(part).area
                if a > area:
                    best, area = f, a
        return best

    def pin(p):
        c = p.representative_point()
        return [round(c.x, 9), round(c.y, 9)]

    # a thin part of the domain belongs to the shape that adds it; a thin part of a
    # region to that region
    thin = [(p, owner_of(p, ("add",))) for p in _opening_loss(D, h, tiny)]
    for f in ev.top:
        if f.role == "region":
            thin += [(p, f) for p in _opening_loss(ev.shapes[f.id].intersection(D), h, tiny)]
    for p, f in thin:
        w = _width(p) / h
        name = f.label if f else "a part of the domain"
        fixes = [{"key": f"grid:{_finer(n, w)}", "label": f"Use n = {_finer(n, w)}"}]
        if f is not None and f.kind in ("rect", "ellipse"):
            fixes.append({"key": f"widen:{f.id}", "label": "Widen to 1 cell"})
        fixes.append({"key": "keep", "label": "Keep",
                      "tip": "Keep it: the chart operator resolves it on its pieces"})
        notes.append({"severity": "warning", "kind": "thin", "feature": f.id if f else None,
                      "at": pin(p), "width_cells": round(w, 3),
                      "message": f"{name} is {w:.1f} cells wide",
                      "detail": f"At n = {n} it shows as a broken staircase.", "fixes": fixes})
    frame = box(*D.buffer(2 * h).bounds)
    gaps = frame.difference(D)
    for p in _opening_loss(gaps, h, tiny):
        if p.intersects(frame.exterior):
            continue                          # the outside, not a gap
        w = _width(p) / h
        notes.append({"severity": "warning", "kind": "sliver", "feature": None,
                      "at": pin(p), "width_cells": round(w, 3),
                      "message": f"a gap {w:.1f} cells wide",
                      "detail": "A gap narrower than one cell can vanish or cut off a bit.",
                      "fixes": [{"key": f"grid:{_finer(n, w)}", "label": f"Use n = {_finer(n, w)}"},
                                {"key": "keep", "label": "Keep"}]})
    for poly in _parts(D):
        rings = [("the outline", poly.exterior)] + [(f"hole {k + 1}", r)
                                                    for k, r in enumerate(poly.interiors)]
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                dist = rings[i][1].distance(rings[j][1])
                if 0 < dist < h:
                    a, b = rings[i][1], rings[j][1]
                    from shapely.ops import nearest_points
                    pa, pb = nearest_points(a, b)
                    notes.append({"severity": "warning", "kind": "near", "feature": None,
                                  "at": [round((pa.x + pb.x) / 2, 9), round((pa.y + pb.y) / 2, 9)],
                                  "message": f"{rings[j][0].capitalize()} comes within "
                                             f"{dist / h:.1f} cells of {rings[i][0]}",
                                  "detail": "Two edges closer than one cell without meeting.",
                                  "fixes": [{"key": f"grid:{_finer(n, dist / h)}",
                                             "label": f"Use n = {_finer(n, dist / h)}"},
                                            {"key": "keep", "label": "Keep"}]})
    for k, x in enumerate(notes):
        x["pin"] = k + 1
    out["notes"] = notes
    kinds = [x["kind"] for x in notes]
    out["checks"] = {
        "closed": all(f.closed or f.role == "guide" for f in geom.features if f.kind == "path"),
        "no_crossings": "crossing" not in kinds,
        "holes_inside": not any(x["kind"] == "outside" and by_id.get(x.get("feature")) is not None
                                and by_id[x["feature"]].role == "cut" for x in notes),
        "regions_inside": not any(x["kind"] == "outside" and by_id.get(x.get("feature")) is not None
                                  and by_id[x["feature"]].role == "region" for x in notes),
        "thin": kinds.count("thin"), "slivers": kinds.count("sliver"),
        "near": kinds.count("near"),
    }
    # -- areas: exact against cells ------------------------------------------------
    exact = None
    if occ.available():
        try:
            exact = occ.areas(geom)
            out["kernel"] = "OpenCASCADE (Gmsh)"
        except Exception as exc:                       # the check still reads Shapely's
            out["kernel_error"] = f"{type(exc).__name__}: {exc}"
    if exact is None:
        out["kernel"] = "Shapely"
        exact = {"domain": D.area,
                 "features": {f.id: ev.shapes[f.id].intersection(D).area
                              for f in ev.top if f.role in ("add", "region")}}
    cell = dx * dx
    rows = [{"name": "domain", "feature": None, "exact": exact["domain"],
             "cells": float(dom_mask.sum()) * cell}]
    # what each feature's region holds after the later regions took their cells
    for k, fid in enumerate(res.region_features):
        f = by_id[fid]
        later = [ev.shapes[x].intersection(D) for x in res.region_features[k + 1:]]
        mine = ev.shapes[fid].intersection(D)
        if later:
            mine = mine.difference(unary_union(later))
        a_exact = mine.area if out["kernel"] == "Shapely" or later else exact["features"].get(fid, mine.area)
        rows.append({"name": f.label, "feature": fid, "exact": a_exact,
                     "cells": float(((owner == k) & dom_mask).sum()) * cell})
    for r in rows:
        r["difference"] = (r["cells"] - r["exact"]) / r["exact"] if r["exact"] > 0 else None
        if page_areas and (r["feature"] or "domain") in page_areas:
            r["page"] = float(page_areas[r["feature"] or "domain"])
    out["areas"] = rows
    return out


def _finer(n: int, width_cells: float) -> int:
    """The smallest offered grid at which a feature this many cells wide is one cell."""
    need = n / max(width_cells, 1e-9)
    for g in GRIDS + (512, 1024):
        if g >= need and g > n:
            return g
    return int(2 ** math.ceil(math.log2(max(need, n + 1))))


__all__ = ["GRIDS", "check_view", "dx_for", "grid_n"]
