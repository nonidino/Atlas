"""The drawn car as solids, on body-fitted overset grids.

PoC 3, Tier 62.  [[poc3-racelab-car-solids]].

Until Tier 61 every part of RaceLab's car was a porous plate: a line segment
carrying a normal traction, with flow passing through it.  A body-fitted grid
needs a closed outline to wrap, so a plate must become a SOLID first.  On
2026-09-15 the user chose the rule, and chose it editable:

* **wing elements are aerofoil-like**, about 12% of their chord thick;
* **body and floor panels are a few cells thick**;
* **shell plates that touch are welded into one closed body**;
* **every thickness is data**, in the ``solids`` block of `car_geometry.json`;
* **the wheels sit a small gap above the road**, about 2% of their radius, and
  still roll at road speed.

What the rule does, step by step (`car_solids`)
-----------------------------------------------

1. Every plate becomes a polygon, in cells: an aerofoil
   (`overset_multi.aerofoil_outline`, thickness ``aerofoil_thickness`` of the
   chord, trailing edge rounded to ``aerofoil_te_thickness``) if its id is in
   ``aerofoil``, otherwise a panel -- the segment thickened to
   ``panel_thickness_cells`` (or its own entry in ``thickness_cells``) with
   round ends.
2. **Welding.**  The polygons are united, then closed by ``weld_cells``: parts
   closer than that become one body, and every concave corner gets a fillet of
   half that radius.
3. **Wheels** are circles raised to ``wheel_gap_fraction`` of their radius
   above the road.  A stationary part is trimmed back to
   ``wheel_clearance_cells`` from any wheel, because a wheel turns and a part
   welded to it could not -- and because a gap narrower than two grid rows is
   one no interpolation closes (`MultiOverset` refuses it).
4. **A fluid pocket sealed inside a body is filled** -- no flow can reach it --
   and a sliver smaller than ``min_area_cells`` is dropped.  Both are recorded.

Every choice above that is not the user's own is a modelling choice this tier
made, and each is a named field, so the page can list them and the editor can
change them.

What a solid gets
-----------------

`car_grids` wraps each solid in `ogrid_from_outline` (columns clustered where
the outline turns, normals smoothed at the wall -- both measured on the
multi-body manufactured solution, where they took a rounded trailing edge from
order 1.7 to 2.0), each wheel in `ogrid_annulus`, and lays a `road_patch`
under each wheel.  `car_flow` marches it with `overset_ns.OversetFlow`: free
stream on the inlet and the top, the rolling road on the floor, an outflow
behind, each wheel's wall moving at ``omega x r`` with ``omega = U / R``.

Not here: the devices (the radiator core and the recovery turbine), the joins,
the agents and the demo -- Tier 63 -- and anything learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import ground_effect as GE
from . import overset as OV
from . import overset_multi as OM

__all__ = [
    "SOLIDS_DEFAULT", "solids_rule", "Solid", "car_solids", "car_grids", "car_overset",
    "car_flow", "GRID", "wall_members", "forces_by_member", "probe",
]

#: The rule's defaults.  `car_geometry.json` may carry a ``solids`` block that
#: overrides any of them; `solids_rule` merges the two and reports which came
#: from where.
SOLIDS_DEFAULT: dict[str, Any] = {
    "aerofoil": ["FW_MAIN", "FW_FLAP", "RW_FLAP"],
    "aerofoil_thickness": 0.12,
    "aerofoil_te_thickness": 0.01,
    "panel_thickness_cells": 3.0,
    "thickness_cells": {},
    "weld_cells": 1.5,
    "fillet_cells": "auto",
    "fillet_by_member": {},
    "wheel_gap_fraction": 0.02,
    "wheel_clearance_cells": 4.0,
    "min_area_cells": 1.0,
    #: **The duct's openings (Tier 63, the user's decision after Tier 62's sealed
    #: pod flowed backwards).**  Each removes the stretch ``[from, to]`` of a
    #: panel's centreline, as fractions of its length, before the panel is
    #: thickened -- so an opening's two edges are round caps.  An inlet in the
    #: chassis's forward-facing slope ahead of the duct, an outlet at the end
    #: of the pod's top just behind the duct's exit.
    "openings": [
        {"plate": "CHASSIS", "from": 0.25, "to": 0.75, "role": "inlet"},
        {"plate": "POD_UP", "from": 0.80, "to": 1.00, "role": "outlet"},
    ],
    #: Section 3.3's *radiator duct area, 0.3 to 1.5 of nominal*, wired in Tier
    #: 69.  It scales every opening's span about its own centre; `1.0` IS the
    #: drawn duct, so the nominal car and every cache keyed on its fingerprint
    #: are unchanged by the knob existing.
    "duct_area": 1.0,
}

#: The fillet radii ``"auto"`` tries, smallest first, in cells.
FILLET_LADDER = (0.0, 1.5, 3.0, 4.5, 6.0, 8.0, 10.0)

#: How the grids are built round the solids, in tiling units.  **Measured on
#: the multi-body manufactured solution at the car's own spacing** (h = 1/64 is
#: its level 3): body grids 0.15 thick with 25 rows, wheels 0.15 thick with 33
#: rows clustered harder, columns about 0.3 h apart at the wall, a road patch
#: 1.17 radii either side of the contact and 0.4 radii tall.
GRID: dict[str, Any] = {
    "h": GE.DX,
    "body_thickness": 0.10, "body_nj": 25, "body_beta": 2.0,
    "room": 0.4, "room_floor": 0.03,
    "cluster": 4.0, "sigma_wall": 0.02, "kappa": 0.5,
    "wheel_thickness": 0.15, "wheel_nj": 33, "wheel_beta": 3.0,
    "wall_spacing_h": 0.3,
    "patch_half_width_r": 0.8, "patch_height_r": 0.25, "patch_nj": 33,
    "patch_spacing_h": 0.7 / 256 * 64,
    "hole_margin": 0.04,
    "max_skew_deg": 70.0,
}


def solids_rule(geometry: dict | None = None) -> dict[str, Any]:
    """The rule in force: `SOLIDS_DEFAULT` overridden by the geometry's block."""
    from . import racelab as RL
    doc = RL.load_geometry() if geometry is None else geometry
    block = dict(doc.get("solids", {}))
    rule = dict(SOLIDS_DEFAULT)
    unknown = sorted(set(block) - set(SOLIDS_DEFAULT) - {"note"})
    if unknown:
        raise ValueError(f"car_geometry.json 'solids' has unknown keys {unknown}")
    for k, v in block.items():
        if k != "note":
            rule[k] = v
    rule["from_file"] = sorted(k for k in block if k != "note")
    return rule


@dataclass
class Solid:
    """One closed body of the car, in TILING units, clockwise."""

    name: str
    outline: np.ndarray
    members: list[str]
    groups: list[str]
    kind: str                       # "shell" or "wheel"
    area_cells: float
    centre: tuple[float, float] | None = None
    radius: float | None = None
    omega: float = 0.0
    notes: list[str] = field(default_factory=list)


def scaled_opening(a0: float, a1: float, area: float) -> tuple[float, float]:
    """One opening's span, scaled about its own centre and kept inside the panel.

    `duct_area` is section 3.3's *radiator duct area, 0.3 to 1.5 of nominal*,
    and until Tier 69 it reached nothing: these spans were literals and no code
    multiplied them.  Scaling about the CENTRE keeps the inlet where it was
    drawn and only opens or closes it.

    **It clamps, and the clamp is why this returns the achieved span rather
    than the asked-for one.**  The outlet is drawn from 0.80 to 1.00, so at
    1.5x it would run to 1.05 -- off the end of the panel.  The span is held
    inside ``[0, 1]``, which means a large `duct_area` delivers less than it
    asks for, and `openings_report` records both so the difference is visible
    rather than assumed away.
    """
    area = float(area)
    if area <= 0.0:
        raise ValueError(f"duct_area must be positive, got {area}")
    mid = 0.5 * (a0 + a1)
    half = 0.5 * (a1 - a0) * area
    lo, hi = mid - half, mid + half
    if lo < 0.0:
        lo, hi = 0.0, min(1.0, hi - lo)
    if hi > 1.0:
        lo, hi = max(0.0, lo - (hi - 1.0)), 1.0
    return (lo, hi)


def _openings_for(b, rule) -> list[tuple[float, float]]:
    area = float(rule.get("duct_area", 1.0))
    out = []
    for o in rule.get("openings", []):
        if o["plate"] != b.body_id:
            continue
        a0, a1 = float(o["from"]), float(o["to"])
        if not 0.0 <= a0 < a1 <= 1.0:
            raise ValueError(f"opening on {b.body_id}: need 0 <= from < to <= 1, got {a0}, {a1}")
        if area != 1.0:
            a0, a1 = scaled_opening(a0, a1, area)
        out.append((a0, a1))
    return out


def openings_report(rule: dict | None = None) -> list[dict]:
    """Each opening's nominal and achieved span at this `duct_area`.

    The knob's probe: a `duct_area` that changes no achieved span has not
    reached the duct, whatever the rule says.
    """
    r = dict(SOLIDS_DEFAULT if rule is None else rule)
    area = float(r.get("duct_area", 1.0))
    rows = []
    for o in r.get("openings", []):
        a0, a1 = float(o["from"]), float(o["to"])
        b0, b1 = scaled_opening(a0, a1, area) if area != 1.0 else (a0, a1)
        rows.append({"plate": o["plate"], "role": o.get("role"),
                     "duct_area": area,
                     "nominal": [a0, a1], "nominal_span": a1 - a0,
                     "achieved": [b0, b1], "achieved_span": b1 - b0,
                     "clamped": bool(abs((b1 - b0) - (a1 - a0) * area) > 1e-12)})
    return rows


def _plate_polygon(b, rule, keep_out=None, openings: bool = True):
    """A plate's solid, in cells.  ``keep_out`` (the wheels' clearance discs) trims
    a PANEL along its centreline before it is thickened, so a trimmed end is a
    round cap like any other.  **Why not cut the solid (Tier 62).**  The first
    version subtracted the discs from the thickened panels, which left two
    square corners where the nose and the diffuser met a wheel's clearance; the
    car's first steps put their largest divergence, growing from 3.5 to 28 in
    six steps, exactly on those two corners."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import substring, unary_union
    cuts = _openings_for(b, rule) if openings else []
    if b.body_id in rule["aerofoil"]:
        if cuts:
            raise ValueError(f"{b.body_id}: an opening is cut into a panel's centreline, and "
                             "this plate is an aerofoil")
        xy = OM.aerofoil_outline(b.x_le, b.y_le, b.chord, rule["aerofoil_thickness"],
                                 b.alpha_deg, rule["aerofoil_te_thickness"], n=1600)
        poly = Polygon(xy)
        if keep_out is not None and poly.intersects(keep_out):
            raise ValueError(f"{b.body_id}: an aerofoil reaches a wheel's clearance; "
                             "move it or shorten it in car_geometry.json")
        return poly
    t = float(rule["thickness_cells"].get(b.body_id, rule["panel_thickness_cells"]))
    line = LineString([(b.x_le, b.y_le), (b.x_te, b.y_te)])
    for a0, a1 in cuts:
        full = LineString([(b.x_le, b.y_le), (b.x_te, b.y_te)])
        gap = substring(full, a0 * full.length, a1 * full.length)
        line = line.difference(gap.buffer(1e-9, cap_style=2))
    if keep_out is not None:
        line = line.difference(keep_out.buffer(0.5 * t, quad_segs=64))
    if line.is_empty:
        return None
    return unary_union([line.buffer(0.5 * t, quad_segs=32)])


def _body_grid(name: str, outline: np.ndarray, G: dict, extra: dict | None = None):
    """A shell's grid, exactly as `car_grids` builds it."""
    h = G["h"]
    seg = np.hypot(np.roll(outline[:, 0], -1) - outline[:, 0], np.roll(outline[:, 1], -1) - outline[:, 1])
    ni = int(64 * math.ceil(float(seg.sum()) / (G["wall_spacing_h"] * h) / 64))
    kw = dict(beta=G["body_beta"], cluster=G["cluster"], sigma_wall=G["sigma_wall"],
              kappa=G["kappa"], room=G["room"], room_floor=G["room_floor"])
    kw.update(extra or {})
    return OV.ogrid_from_outline(name, outline, ni, G["body_nj"], G["body_thickness"], **kw), kw


def _grid_verdict(outline: np.ndarray, G: dict) -> dict[str, Any]:
    """Can this outline be gridded: no fold, no cell skewed past
    ``G["max_skew_deg"]``, and -- **with the background alone** -- no orphan.  The
    last is what a narrow corner of a body's own fails: its grid, thinned to
    stay clear of the far wall, ends where the background is all hole and
    fringe, and a grid cannot be its own donor (Tier 62 found 61 such points
    inside the rear wing at a 4.5-cell fillet that passed the first two)."""
    from . import racelab as RL
    try:
        g, _kw = _body_grid("probe", outline, G)
    except OV.GridQualityError as exc:
        return {"ok": False, "folds": True, "why": str(exc)[:160]}
    q = OV.grid_quality(g)
    out = {"ok": q["max_skew_deg"] <= G["max_skew_deg"], "folds": False,
           "max_skew_deg": round(q["max_skew_deg"], 2),
           "thickness_min": round(float(g.meta.get("thickness_min", G["body_thickness"])), 4)}
    if out["ok"]:
        h = G["h"]
        bg = OV.CartesianGrid("bg", RL.RNX, RL.RNY + 1, h, y0=-0.5 * h)
        try:
            OM.MultiOverset(bg, [g], hole_margin=G["hole_margin"], width=3, road=0.0)
            out["orphans_alone"] = 0
        except OV.OversetError as exc:
            out["ok"] = False
            out["orphans_alone"] = int(str(exc).split(" ")[0]) if str(exc)[0].isdigit() else -1
    return out


def car_solids(p=None, geometry: dict | None = None, u_inf: float = GE.U_INF,
               grid: dict | None = None, check: bool = True) -> tuple[list[Solid], dict]:
    """The car's solids by `solids_rule`, and a record of what the rule did.

    ``check=False`` skips building a trial grid for a shell whose fillet is FIXED
    (``fillet_cells`` a number, or ``fillet_by_member``) -- the ladder still
    checks every rung it climbs -- so a caller that already knows the fillets
    gets the shapes in a fraction of a second."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from . import racelab as RL
    doc = RL.load_geometry() if geometry is None else geometry
    rule = solids_rule(doc)
    objs, _flat = RL.car_bodies(p, geometry=doc)
    plates = [o.body for o in objs if isinstance(o, RL.PlateBody)]
    wheels = [o for o in objs if isinstance(o, RL.WheelBody)]
    unknown = sorted(set(rule["aerofoil"]) - {b.body_id for b in plates})
    if unknown:
        raise ValueError(f"solids.aerofoil names plates the car does not have: {unknown}")
    groups = {b.body_id: str(b.group) for b in plates}
    record: dict[str, Any] = {"rule": {k: v for k, v in rule.items()}, "filled_pockets": [],
                              "dropped_slivers": [], "trimmed": {}}
    circles = []
    for wh in wheels:
        cy = wh.r * (1.0 + float(rule["wheel_gap_fraction"]))
        circles.append((wh, cy, Point(wh.xc, cy).buffer(wh.r, quad_segs=256)))
    clear = float(rule["wheel_clearance_cells"])
    keep_out = unary_union([c.buffer(clear, quad_segs=64) for _w, _cy, c in circles]) if circles else None
    polys = {}
    record["openings"] = []
    for o in rule.get("openings", []):
        b = next((q for q in plates if q.body_id == o["plate"]), None)
        if b is None:
            raise ValueError(f"an opening names a plate the car does not have: {o['plate']}")
        record["openings"].append(dict(o, length_cells=round(float(b.chord) * (float(o["to"]) - float(o["from"])), 3)))
    for b in plates:
        whole = _plate_polygon(b, rule)
        trimmed = _plate_polygon(b, rule, keep_out)
        if trimmed is None:
            record["trimmed"][b.body_id] = "removed: inside a wheel's clearance"
            continue
        if whole.area - trimmed.area > 1e-6:
            record["trimmed"][b.body_id] = round(float(whole.area - trimmed.area), 3)
        polys[b.body_id] = trimmed
    union = unary_union(list(polys.values()))
    w = float(rule["weld_cells"])
    if w > 0:
        union = union.buffer(0.5 * w, quad_segs=32).buffer(-0.5 * w, quad_segs=32)
    if keep_out is not None:
        # welding can reach back into a clearance; the discs win, and any area
        # they take back is recorded, because a cut there leaves a corner
        before = union.area
        union = union.difference(keep_out)
        record["clearance_cut_after_weld_cells"] = round(float(before - union.area), 4)
    parts = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
    shells = []
    for g in parts:
        if g.area < float(rule["min_area_cells"]):
            record["dropped_slivers"].append({"area_cells": float(g.area),
                                              "at": [float(v) for v in g.centroid.coords[0]]})
            continue
        solid = Polygon(g.exterior)
        for hole in g.interiors:
            record["filled_pockets"].append({"area_cells": float(Polygon(hole).area),
                                             "at": [float(v) for v in Polygon(hole).centroid.coords[0]]})
        shells.append(solid)
    shells.sort(key=lambda s: (s.bounds[0], s.bounds[1]))
    h = GE.DX
    G = dict(GRID if grid is None else grid)
    # Fillets, one shell at a time: closing a shell by itself rounds its own
    # concave corners and fills its own notches narrower than twice the radius,
    # and cannot weld it to a neighbour the way closing the union would.  With
    # ``fillet_cells = "auto"`` each shell takes the SMALLEST radius on
    # `FILLET_LADDER` whose grid, built exactly as `car_grids` builds it, neither
    # folds nor skews past ``GRID["max_skew_deg"]`` -- the least change to the
    # drawing that a grid can wrap, and every attempt is recorded.
    record["fillets"] = []
    out: list[Solid] = []
    for k, s in enumerate(shells):
        others = unary_union([o for q, o in enumerate(shells) if q != k]) if len(shells) > 1 else None
        members = [pid for pid, poly in polys.items() if poly.buffer(0.5 * w + 1e-6).intersects(s)]

        def filleted(r):
            if r <= 0:
                return s, 0.0
            f = s.buffer(r, quad_segs=32).buffer(-r, quad_segs=32)
            grown = f.area
            if keep_out is not None:
                f = f.difference(keep_out)
            if others is not None:
                f = f.difference(others.buffer(clear, quad_segs=32))
            if f.geom_type == "MultiPolygon":
                f = max(f.geoms, key=lambda q: q.area)
            return Polygon(f.exterior), float(grown - f.area)

        fixed = rule["fillet_by_member"]
        chosen = [fixed[m] for m in members if m in fixed]
        if chosen:
            ladder = (float(max(chosen)),)
        elif rule["fillet_cells"] == "auto":
            ladder = FILLET_LADDER
        else:
            ladder = (float(rule["fillet_cells"]),)
        attempts = []
        pick = None
        for r in ladder:
            f, cut_back = filleted(r)
            xy = OV.orient_clockwise(np.asarray(f.exterior.coords)[:-1] * h)
            verdict = (_grid_verdict(xy, G) if (check or len(ladder) > 1)
                       else {"ok": None, "unchecked": True})
            attempts.append({"fillet_cells": r, "added_cells": round(float(f.area - s.area), 2),
                             "cut_back_cells": round(cut_back, 4), **verdict})
            if verdict["ok"] or len(ladder) == 1:
                pick = (r, f, xy)
                break
        if pick is None:
            raise OV.GridQualityError(f"the shell holding {members}: no fillet on the ladder "
                                      f"gives a grid that neither folds nor skews: {attempts}")
        r, f, xy = pick
        record["fillets"].append({"members": members, "attempts": attempts, "fillet_cells": r})
        grp = []
        for pid in members:
            if groups[pid] not in grp:
                grp.append(groups[pid])
        out.append(Solid(name=f"shell{k}", outline=xy, members=members, groups=grp, kind="shell",
                         area_cells=float(f.area), notes=[f"fillet {r:g} cells"]))
    for wh, cy, c in circles:
        R = wh.r * h
        out.append(Solid(name=wh.body_id, outline=OV.circle_outline(wh.xc * h, cy * h, R, n=4096),
                         members=[wh.body_id], groups=["wheel"], kind="wheel",
                         area_cells=float(c.area), centre=(wh.xc * h, cy * h), radius=R,
                         omega=u_inf / R))
    record["solids"] = [{"name": s.name, "kind": s.kind, "members": s.members, "groups": s.groups,
                         "area_cells": round(s.area_cells, 2)} for s in out]
    return out, record


def car_grids(solids: Sequence[Solid], grid: dict | None = None) -> tuple[OV.CartesianGrid, list, dict]:
    """The background, one grid per solid, and a road patch under each wheel."""
    from . import racelab as RL
    G = dict(GRID if grid is None else grid)
    h = G["h"]
    bg = OV.CartesianGrid("bg", RL.RNX, RL.RNY + 1, h, y0=-0.5 * h)
    comps: list = []
    rec: dict[str, Any] = {}
    for s in solids:
        if s.kind == "wheel":
            cx, cy = s.centre
            R = s.radius
            ni = int(64 * math.ceil(2 * math.pi * R / (G["wall_spacing_h"] * h) / 64))
            g = OV.ogrid_annulus(s.name, cx, cy, R, R + G["wheel_thickness"], ni, G["wheel_nj"],
                                 beta=G["wheel_beta"])
            comps.append(g)
            half, height = G["patch_half_width_r"] * R, G["patch_height_r"] * R
            pni = int(2 * half / (G["patch_spacing_h"] * h)) + 1
            comps.append(OM.road_patch(f"road_{s.name}", cx - half, cx + half, height, pni,
                                       G["patch_nj"], beta=G["wheel_beta"]))
            rec[s.name] = {"shape": list(g.shape), "patch_shape": [G["patch_nj"], pni]}
            continue
        g, kw = _body_grid(s.name, s.outline, G)
        comps.append(g)
        rec[s.name] = {"shape": list(g.shape), "settings": kw, "quality": OV.grid_quality(g),
                       "thickness_min": float(g.meta.get("thickness_min", G["body_thickness"]))}
    return bg, comps, rec


def car_overset(solids: Sequence[Solid], grid: dict | None = None):
    G = dict(GRID if grid is None else grid)
    bg, comps, rec = car_grids(solids, G)
    ov = OM.MultiOverset(bg, comps, hole_margin=G["hole_margin"], width=3, road=0.0)
    return ov, rec


#: How long the start takes.  **The car's walls are brought to rest; the stream
#: is never started (Tier 62).**  At ``t = 0`` every wall moves with the free
#: stream, so the uniform stream is an exact solution of the discrete equations
#: -- Tier 61 asserts it to 1e-12 -- and over ``RAMP`` each wall's velocity
#: turns into its own, ``(1 - s) U + s u_wall`` with ``s = (1 - cos(pi t /
#: RAMP)) / 2``: rest for a shell, rolling for a wheel.  The road, the inlet and
#: the top stay at the stream throughout.
#:
#: Two starts came first and both failed on the car.  **Impulsive** (the stream
#: everywhere but on the walls, Tier 61's cylinder start): the jump across a
#: wall row 0.0014 thick put the first step's divergence at 2.3e3 and its speed
#: at 23, and the second step's momentum solve did not converge.  **From rest**,
#: the stream, road and wheels ramped together: a pressure-correction step
#: accelerates the interior only through the lagging pressure, so the road grew
#: a spurious boundary layer that the road patches resolve and the background
#: does not, and the divergence at a patch's end grew from 0.5 to 2.5 in six
#: steps.  With the walls ramped instead, nothing that is far from the car ever
#: changes.
RAMP = 0.5


def ramp(t: float, tau: float = RAMP) -> float:
    return 1.0 if tau <= 0 or t >= tau else 0.5 * (1.0 - math.cos(math.pi * t / tau))


def car_flow(ov, solids: Sequence[Solid], *, nu: float = GE.NU, dt: float = GE.MACRO_DT,
             u_inf: float = GE.U_INF, chi: float = 0.5, ramp_time: float = RAMP,
             **flow_kw):
    """The car's march: the uniform stream, with the car's walls brought from the
    stream's velocity to their own by `ramp`."""
    from . import overset_ns as NS
    wheels = {s.name: s for s in solids if s.kind == "wheel"}

    def wall_velocity(g, t):
        n = g.ni
        if not g.periodic:
            return np.full(n, u_inf), np.zeros(n)
        a = ramp(t, ramp_time)
        s = wheels.get(g.name)
        if s is None:
            return np.full(n, (1.0 - a) * u_inf), np.zeros(n)
        cx, cy = s.centre
        return ((1.0 - a) * u_inf - a * s.omega * (g.y[0] - cy),
                a * s.omega * (g.x[0] - cx))

    flow = NS.OversetFlow(ov, nu, dt,
                          box={"xlo": "dirichlet", "ylo": "dirichlet", "yhi": "dirichlet",
                               "xhi": "outflow"},
                          box_velocity=lambda x, y, t: (np.full_like(x, u_inf), np.zeros_like(x)),
                          wall_velocity=wall_velocity, chi=chi, **flow_kw)
    u = np.full(ov.n_unknowns, u_inf)
    v = np.zeros(ov.n_unknowns)
    # steady at t = 0, so BDF2 may start at once
    flow.set_state(u, v, u_prev=u, v_prev=v)
    return flow


def wall_members(ov, solids: Sequence[Solid], geometry: dict | None = None, p=None) -> dict[str, np.ndarray]:
    """For each shell's wall points, the plate each belongs to: the nearest
    plate polygon.  A welded body's force can then be read by part."""
    from shapely.geometry import Point
    from . import racelab as RL
    doc = RL.load_geometry() if geometry is None else geometry
    rule = solids_rule(doc)
    objs, _flat = RL.car_bodies(p, geometry=doc)
    plates = {o.body.body_id: _plate_polygon(o.body, rule, openings=False)
              for o in objs if isinstance(o, RL.PlateBody)}
    out = {}
    h = GE.DX
    for s in solids:
        if s.kind != "shell":
            continue
        c = next(g for g in ov.comps if g.name == s.name)
        names = s.members
        pts = np.column_stack([c.x[0], c.y[0]]) / h
        d = np.array([[plates[m].distance(Point(float(x), float(y))) for x, y in pts] for m in names])
        out[s.name] = np.array(names, dtype=object)[np.argmin(d, axis=0)]
    return out


def forces_by_member(flow, solid: Solid, members: np.ndarray | None) -> dict[str, dict[str, float]]:
    """The force on each part of a solid: ``F = -sum (sigma . n) ds`` over the wall
    points that belong to it (all of them, for a wheel)."""
    c = next(g for g in flow.ov.comps if g.name == solid.name)
    ic = flow.ov.index[c.name]
    u, v, p = flow.U[ic], flow.V[ic], flow.P[ic]
    ux, uy = c.gradient(u)
    vx, vy = c.gradient(v)
    nx_, ny_ = c.wall_normal()
    ds = np.hypot(c.x_xi[0], c.y_xi[0])
    nu = flow.nu
    tx = -(-p[0] * nx_ + 2 * nu * ux[0] * nx_ + nu * (uy[0] + vx[0]) * ny_) * ds
    ty = -(-p[0] * ny_ + nu * (uy[0] + vx[0]) * nx_ + 2 * nu * vy[0] * ny_) * ds
    labels = np.array([solid.name] * c.ni, dtype=object) if members is None else members
    out = {}
    for m in dict.fromkeys(labels):
        sel = labels == m
        out[str(m)] = {"fx": float(tx[sel].sum()), "fy": float(ty[sel].sum())}
    return out


def probe(ov, field: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """A composite field at arbitrary points: from the body grid or patch nearest
    that holds each point with a usable stencil, else from the background."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    out = np.full(px.shape, np.nan)
    open_ = np.ones(px.shape, dtype=bool)
    for c in list(ov.comps) + [ov.bg]:
        sel = np.nonzero(open_)[0]
        if not sel.size:
            break
        if c is ov.bg:
            res = ov._cartesian_donors(px[sel], py[sel])
        else:
            res = ov._curvilinear_donors(c, px[sel], py[sel])
        good = res["ok"]
        if np.any(good):
            idx = ov.index[c.name].ravel()[res["flat"][good]]
            out[sel[good]] = np.sum(field[idx] * res["weights"][good], axis=1)
            open_[sel[good]] = False
    return out
