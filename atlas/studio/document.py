"""Case file 0.7: a case's geometry as a feature tree (integrated UI plan, section 3.1).

A 0.7 file is a 0.6 case file (`atlas.workbench.spec.CaseSpec`, unchanged) with one
more section, ``geometry``: the shapes a person drew, each a **feature** with a kind,
a role and its own numbers, in the order they apply.  The 0.6 part keeps the engine's
own copy of the geometry in cells (``domain.outline``, ``domain.holes``, the drawn
regions), regenerated from the tree whenever the tree changes (`geometry_eval`), so a
0.7 file is still a complete engine case.

**Features** (coordinates in the physics' length unit, metres for every family but
the wind farm):

* ``rect``: ``x, y`` its lower-left corner, ``w, h``, and ``r`` a corner radius;
* ``ellipse``: ``cx, cy, rx, ry`` (a circle when ``rx == ry``);
* ``path``: ``points`` joined by ``edges`` (``line``; ``arc`` with its ``bulge``,
  the engine's DXF convention; ``cubic`` with its two ``handles``; ``spline``, the
  engine's centripetal Catmull-Rom), closed unless ``closed`` is false;
* ``combine``: ``op`` (``union``, ``subtract``, ``intersect``) of ``inputs``, the
  ids of other features; the first input is the base, and the inputs stay editable;
* ``scatter`` (P2): seeded inclusions.

**Roles**: ``add`` (the shape adds to the domain, filled with its ``material``),
``cut`` (it cuts a hole or a notch), ``region`` (a material region inside the
domain), ``guide`` (construction: drawn, never part of the domain).  The domain is
the union of what adds, minus the union of what cuts; regions stack in order, later
on top, as the engine's regions always have.

**0.6 files migrate** (`migrate`) as features built from their polygons: the grid or
the drawn outline, each hole, each region.  A tree that is still exactly the
migration of the case's own geometry is saved as 0.6, so a 0.6 file opened and saved
is the same bytes (`dumps`).
"""

from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from atlas.workbench.spec import SCHEMA_ID as SCHEMA_06
from atlas.workbench.spec import CaseSpec

SCHEMA_07 = "atlas-workbench/case@0.7"
EDGE_KINDS = ("line", "arc", "cubic", "spline")
ROLES = ("add", "cut", "region", "guide")
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,40}$")


class Parameter(BaseModel):
    """A named value (P2): a number, or a formula of other parameters."""
    model_config = ConfigDict(extra="forbid")
    name: str
    value: float = 0.0
    expr: Optional[str] = None
    sweep: Optional[list[float]] = None


class Constraint(BaseModel):
    """A sketch constraint the page solves with planegcs (P2): between features'
    points, or a driving dimension.  ``refs`` name points as ``<feature>:<point>``
    (a rectangle's corners are ``c0`` to ``c3``, an ellipse's centre ``c``, a path's
    points ``p0``, ``p1`` ...) and lines as ``<feature>:e<k>``."""
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["fix", "coincident", "horizontal", "vertical", "parallel", "perpendicular",
                  "equal", "tangent", "distance"]
    refs: list[str] = Field(default_factory=list)
    value: Optional[float] = None
    param: Optional[str] = None


class Feature(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["rect", "ellipse", "path", "combine", "scatter", "pattern", "mirror",
                  "channel"]
    role: Literal["add", "cut", "region", "guide"] = "add"
    name: str = ""
    material: Optional[str] = None
    hidden: bool = False
    # rect
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    r: float = 0.0
    # ellipse
    cx: float = 0.0
    cy: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    # path
    points: list[tuple[float, float]] = Field(default_factory=list)
    edges: list[Literal["line", "arc", "cubic", "spline"]] = Field(default_factory=list)
    bulge: list[float] = Field(default_factory=list)
    #: per edge: a cubic's two control points (x1, y1, x2, y2), else None
    handles: list[Optional[tuple[float, float, float, float]]] = Field(default_factory=list)
    closed: bool = True
    # combine; pattern and mirror (one input)
    op: Optional[Literal["union", "subtract", "intersect"]] = None
    inputs: list[str] = Field(default_factory=list)
    #: scatter (count, rmin, rmax, gap, seed, x, y, w, h), pattern (mode, count, dx, dy,
    #: cx, cy, angle, nx, ny), mirror (axis, at)
    params: dict[str, float | str] = Field(default_factory=dict)
    #: a scatter's inclusions as the seed placed them: (cx, cy, r), written by the server
    items: list[tuple[float, float, float]] = Field(default_factory=list)
    #: P2: fields driven by formulas of the parameters, field -> formula ("w": "W")
    bind: dict[str, str] = Field(default_factory=dict)
    #: P2: a coefficient field over the region: {"kind": "formula", "prop": "k",
    #: "expr": "..."}; its values are binned into materials for the engine
    field: Optional[dict] = None

    @model_validator(mode="after")
    def _consistent(self):
        if not _ID.match(self.id):
            raise ValueError(f"feature id {self.id!r}: a letter, then letters, digits, - or _")
        if self.kind == "rect":
            if not (self.w > 0 and self.h > 0):
                raise ValueError(f"{self.label}: a rectangle needs a width and a height above 0")
            if not 0 <= self.r <= 0.5 * min(self.w, self.h) + 1e-12:
                raise ValueError(f"{self.label}: the corner radius is at most half the "
                                 f"shorter side")
        elif self.kind == "ellipse":
            if not (self.rx > 0 and self.ry > 0):
                raise ValueError(f"{self.label}: an ellipse needs both radii above 0")
        elif self.kind == "path":
            n = len(self.points)
            m = n if self.closed else n - 1
            if n < 2:
                raise ValueError(f"{self.label}: a path needs at least two points")
            if not self.edges:
                self.edges = ["line"] * m
            if len(self.edges) != m:
                raise ValueError(f"{self.label}: {n} points {'closed ' if self.closed else ''}"
                                 f"make {m} edges, not {len(self.edges)}")
            if not self.bulge:
                self.bulge = [0.0] * m
            if not self.handles:
                self.handles = [None] * m
            if len(self.bulge) != m or len(self.handles) != m:
                raise ValueError(f"{self.label}: one bulge and one handle pair per edge")
            for k, e in enumerate(self.edges):
                if e == "cubic" and self.handles[k] is None:
                    raise ValueError(f"{self.label}: edge {k} is a curve with no handles")
                if e == "arc" and abs(self.bulge[k]) > 4.0:
                    raise ValueError(f"{self.label}: an arc bulges at most 4 (about 300 degrees)")
            if self.closed and n < 3 and not any(e in ("arc", "cubic") for e in self.edges):
                raise ValueError(f"{self.label}: two points joined by straight edges enclose "
                                 f"nothing")
            if not self.closed and self.role in ("add", "cut", "region"):
                raise ValueError(f"{self.label}: an open path cannot {self.role}: close it, "
                                 f"or keep it as a guide")
        elif self.kind == "combine":
            if self.op is None or len(self.inputs) < 2:
                raise ValueError(f"{self.label}: a combine needs an operation and two inputs")
        elif self.kind in ("pattern", "mirror"):
            if len(self.inputs) != 1:
                raise ValueError(f"{self.label}: a {self.kind} repeats exactly one shape")
        elif self.kind == "channel":
            if len(self.points) < 2 or not self.w > 0:
                raise ValueError(f"{self.label}: a channel needs a centre line of two or more "
                                 f"points and a width above 0")
            self.closed = False
            m = len(self.points) - 1
            if not self.edges:
                self.edges = ["line"] * m
            if not self.bulge:
                self.bulge = [0.0] * m
            if not self.handles:
                self.handles = [None] * m
        elif self.kind == "scatter":
            if not (self.w > 0 and self.h > 0):
                raise ValueError(f"{self.label}: a scatter needs its box (w, h above 0)")
        if self.field is not None:
            if self.field.get("kind") not in ("formula", "image") or not self.field.get("prop"):
                raise ValueError(f"{self.label}: a field is a formula or an image of a property")
        return self

    @property
    def label(self) -> str:
        return self.name or self.id


class Geometry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameters: list[Parameter] = Field(default_factory=list)
    features: list[Feature] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _tree(self):
        ids = [f.id for f in self.features]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"two features share the id {dup[0]!r}")
        pnames = [p.name for p in self.parameters]
        for n in pnames:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,30}$", n) or n in ("x", "y", "T", "pi", "e"):
                raise ValueError(f"{n!r} cannot name a parameter: a letter first, then letters, "
                                 f"digits or _ (and not x, y, T, pi or e)")
        if len(set(pnames)) != len(pnames):
            raise ValueError("two parameters share a name")
        known = set(ids)
        used: dict[str, str] = {}
        index = {f.id: k for k, f in enumerate(self.features)}
        for f in self.features:
            if f.kind not in ("combine", "pattern", "mirror"):
                continue
            for i in f.inputs:
                if i not in known:
                    raise ValueError(f"{f.label} combines {i!r}, which is not a feature")
                if i == f.id:
                    raise ValueError(f"{f.label} combines itself")
                if i in used:
                    raise ValueError(f"{i!r} is an input of both {used[i]} and {f.label}")
                if index[i] > index[f.id]:
                    raise ValueError(f"{f.label} combines {i!r}, which comes after it")
                used[i] = f.label
        return self

    def by_id(self) -> dict[str, Feature]:
        return {f.id: f for f in self.features}

    def consumed(self) -> set[str]:
        """Features that are inputs of a combine, a pattern or a mirror: evaluated only
        through it."""
        return {i for f in self.features if f.kind in ("combine", "pattern", "mirror")
                for i in f.inputs}


def next_id(geom: Geometry, prefix: str = "f") -> str:
    ids = {f.id for f in geom.features}
    k = len(ids) + 1
    while f"{prefix}{k}" in ids:
        k += 1
    return f"{prefix}{k}"


# ---------------------------------------------------------------------------
# migration: a 0.6 case's own geometry as features
# ---------------------------------------------------------------------------


def _outline_feature(fid: str, o, dx: float, role: str, name: str, material=None) -> Feature:
    pts = [(float(x) * dx, float(y) * dx) for x, y in o.points]
    n = len(pts)
    kinds = list(o.edges) if o.edges is not None else ["line"] * n
    bulges = [float(b) for b in o.bulge] if o.bulge is not None else [0.0] * n
    return Feature(id=fid, kind="path", role=role, name=name, material=material,
                   points=pts, edges=kinds, bulge=bulges, handles=[None] * n)


def migrate(spec: CaseSpec) -> Geometry:
    """The feature tree a 0.6 case's own geometry is: what the page draws and edits.

    The grid (no outline) is a rectangle at the origin; a drawn outline is a path with
    the engine's own edges (lines, arcs by their bulge, splines); each hole is a cut;
    each region is a region feature.  A region covering the whole grid that comes first
    (the starter's ``base``) becomes the domain's own material instead."""
    d = spec.domain
    dx = float(d.dx)
    feats: list[Feature] = []
    regions = list(spec.regions)
    base_material = None
    if (regions and regions[0].shape == "rect" and regions[0].x0 == 0 and regions[0].y0 == 0
            and regions[0].nx >= d.nx and regions[0].ny >= d.ny):
        base_material = regions[0].material
        regions = regions[1:]
    if d.outline is None:
        feats.append(Feature(id="f1", kind="rect", role="add", name="Plate",
                             material=base_material, x=0.0, y=0.0, w=d.nx * dx, h=d.ny * dx))
    else:
        feats.append(_outline_feature("f1", d.outline, dx, "add", "Outline", base_material))
    for k, h in enumerate(d.holes):
        feats.append(_outline_feature(f"f{len(feats) + 1}", h, dx, "cut", f"Hole {k + 1}"))
    for r in regions:
        fid = f"f{len(feats) + 1}"
        name = r.id
        if r.shape == "rect":
            feats.append(Feature(id=fid, kind="rect", role="region", name=name,
                                 material=r.material, x=r.x0 * dx, y=r.y0 * dx,
                                 w=r.nx * dx, h=r.ny * dx))
        elif r.shape == "curve":
            feats.append(_outline_feature(fid, r.outline, dx, "region", name, r.material))
        else:
            pts = [(float(x) * dx, float(y) * dx) for x, y in r.points]
            body = Feature(id=fid, kind="path", role="region", name=name, material=r.material,
                           points=pts)
            if not r.holes:
                feats.append(body)
            else:
                cuts = []
                feats.append(body)
                for j, hole in enumerate(r.holes):
                    hid = f"f{len(feats) + 1}"
                    feats.append(Feature(id=hid, kind="path", role="cut", name=f"{name} hole {j + 1}",
                                         points=[(float(x) * dx, float(y) * dx) for x, y in hole]))
                    cuts.append(hid)
                feats.append(Feature(id=f"f{len(feats) + 1}", kind="combine", role="region",
                                     name=name, material=r.material, op="subtract",
                                     inputs=[fid, *cuts]))
    return Geometry(features=feats)


# ---------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------


def loads(text: str) -> tuple[CaseSpec, Geometry, str]:
    """A case file of any readable schema: (the engine's case, its feature tree, the
    schema it was read as)."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("a case file is a JSON object")
    if data.get("schema_id") == SCHEMA_07:
        geom_data = data.pop("geometry", None) or {}
        data["schema_id"] = SCHEMA_06
        spec = CaseSpec.model_validate(data)
        try:
            geom = Geometry.model_validate(geom_data)
        except ValidationError as exc:
            raise ValueError(f"its geometry section: {exc.errors()[0].get('msg')}") from None
        return spec, geom, SCHEMA_07
    spec = CaseSpec.model_validate(data)
    return spec, migrate(spec), spec.schema_id if data.get("schema_id") else SCHEMA_06


def is_migration(spec: CaseSpec, geom: Geometry) -> bool:
    """Whether the tree is exactly the migration of the case's own geometry: nothing
    a 0.6 file cannot hold."""
    return (geom.model_dump() == migrate(spec).model_dump())


def dumps(spec: CaseSpec, geom: Geometry) -> str:
    """The file's text: 0.6 when the tree adds nothing to the engine's own geometry
    (so a 0.6 file opened and saved is the same bytes), else 0.7."""
    if is_migration(spec, geom):
        return spec.to_json()
    data = json.loads(spec.to_json())
    out = {"schema_id": SCHEMA_07, "geometry": json.loads(geom.model_dump_json())}
    out.update({k: v for k, v in data.items() if k != "schema_id"})
    return json.dumps(out, indent=2, ensure_ascii=False)


def save(path: str, spec: CaseSpec, geom: Geometry) -> str:
    """Write atomically, in text mode as `CaseSpec.save` does (so a 0.6 file opened
    and saved on one machine is the same bytes); OneDrive can hold a just-written
    file, so the replace is retried."""
    import os
    import time
    if is_migration(spec, geom):
        return spec.save(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(dumps(spec, geom))
    for k in range(40):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if k == 39:
                raise
            time.sleep(0.25)
    return path


__all__ = ["Constraint", "EDGE_KINDS", "Feature", "Geometry", "Parameter", "ROLES", "SCHEMA_06",
           "SCHEMA_07", "dumps", "is_migration", "loads", "migrate", "next_id", "save"]
