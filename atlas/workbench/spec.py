"""The case a person builds in the workbench, as data.

A case is one JSON document: the domain, the physics, the windows it is cut
into, the devices in it, how the windows are coupled, how to run it, and what to
compare it against.  Everything the workbench shows is a view of this object, and
everything a person does in it is an edit to this object -- so a case can be
saved, diffed, reloaded and run without the GUI.

Units, chosen so that nothing the grid has to agree with is a float:
  * the domain and every window are in CELLS (a window must align with the grid);
  * devices are in the physics' own length unit (for the wind-farm family, rotor
    diameters D), because a rotor does not have to sit on a cell boundary;
  * ``domain.dx`` converts between them (the wind-farm family: 1/32 D).

Schema ``atlas-workbench/case@0.2`` added the geometry section's two layers:
material **regions** (rectangles drawn in the page, or polygons imported from
Gmsh) and **boundaries** (conditions on segments of the domain's edges).

Schema ``atlas-workbench/case@0.3`` makes the case file hold any family, not
only the wind farm:

* ``physics.params`` holds the family's scalar parameters by the names its
  registry entry declares (0.2 had the wind farm's ``nu`` and ``u_inf`` as
  fields);
* ``materials`` names each material's properties, which the regions refer to;
* ``attachments`` are lumped parts -- batteries and resistors -- wired between
  electrodes on the domain's edge and free circuit nodes (showcase case 5);
* ``coupling.style`` is the showcase plan's coupling style (A, B, C, D), with
  the iteration's tolerance, relaxation and Dirichlet side;
* ``run.mode`` is steady or transient.

Schema ``atlas-workbench/case@0.4`` (2026-09-29, the owner's request: "draw
nonregular geometry edges, and then have NONRECTANGULAR windows") adds **drawn
shapes** (`Outline`: vertices in cells joined by straight lines, circular arcs
or splines; `shapes.py`):

* ``domain.outline`` and ``domain.holes``: the domain is the cells inside its
  outline (the whole grid when there is none) and outside its holes;
* a window or a region may be a drawn shape (``shape = "curve"``), in which case
  its rectangle is its bounding box, derived;
* a boundary may sit on a drawn edge: ``edge = "outline:<k>"`` or
  ``"hole<h>:<k>"``, the whole of that edge.

The solvers see the cells whose centres a shape contains; nothing else about a
0.3 file changes, and a case with no drawn shape runs on the same arithmetic as
before (`geometry.is_plain`).

Schema ``atlas-workbench/case@0.5`` (2026-09-29, the owner's next question: "why
doesn't their shape itself ADAPT to the geometry of the curve smoothly?") adds
**windows generated from the geometry** (`layout.py`):

* ``layout``: how the windows follow the domain -- cut ``along`` it into
  pieces (and ``across`` it) at level curves of its own harmonic coordinates, or
  one piece per material -- and a fingerprint of what they were generated
  from, so they are generated again whenever the geometry changes;
* a generated window is ``shape = "cells"``: its cells, as runs along rows.

Older files still load: a 0.1 file gains its family's fixed boundaries, and a
0.2 file's ``nu`` and ``u_inf`` move into ``params`` and its style is its
family's.  The geometry rules themselves live in `geometry.py`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from . import geometry as geo
from . import registry
from . import shapes

SCHEMA_ID = "atlas-workbench/case@0.5"
#: every schema this version reads; older ones are migrated on load
READABLE = ("atlas-workbench/case@0.1", "atlas-workbench/case@0.2",
            "atlas-workbench/case@0.3", "atlas-workbench/case@0.4", SCHEMA_ID)


class Outline(BaseModel):
    """A closed shape drawn in the page (`shapes.py`), in cells.

    ``points`` are its vertices; edge ``k`` runs from point ``k`` to point
    ``k + 1``, and the last edge closes the shape.  ``edges`` says whether each
    edge is a straight ``line``, a circular ``arc`` or a ``spline`` (all lines
    when omitted), and ``bulge`` gives each arc's bulge ``tan(theta / 4)``,
    positive when it bows to the right of the edge's direction (0 when omitted).
    """

    points: list[tuple[float, float]]
    edges: Optional[list[Literal["line", "arc", "spline"]]] = None
    bulge: Optional[list[float]] = None

    @model_validator(mode="after")
    def _consistent(self):
        n = len(self.points)
        if n < 2:
            raise ValueError("a drawn shape needs at least two vertices")
        if self.edges is not None and len(self.edges) != n:
            raise ValueError(f"a drawn shape with {n} vertices has {n} edges, "
                             f"not {len(self.edges)}")
        if self.bulge is not None and len(self.bulge) != n:
            raise ValueError(f"a drawn shape with {n} vertices has {n} bulges, "
                             f"not {len(self.bulge)}")
        return self

    def kinds(self) -> list[str]:
        return list(self.edges) if self.edges is not None else ["line"] * len(self.points)

    def bulges(self) -> list[float]:
        return ([float(b) for b in self.bulge] if self.bulge is not None
                else [0.0] * len(self.points))

    def ring(self, step: float = shapes.STEP) -> np.ndarray:
        return shapes.sample(self.points, self.edges, self.bulge, step)

    def bbox(self) -> tuple[float, float, float, float]:
        r = self.ring()
        return (float(r[:, 0].min()), float(r[:, 1].min()), float(r[:, 0].max()),
                float(r[:, 1].max()))

    def problems(self) -> list[str]:
        return shapes.problems(self.points, self.edges, self.bulge)

    @classmethod
    def of(cls, shape) -> "Outline":
        """From a `shapes` triple ``(points, kinds, bulges)``."""
        pts, ks, bs = shape[:3]
        return cls(points=[tuple(map(float, p)) for p in pts], edges=list(ks),
                   bulge=[float(b) for b in bs])


def _cell_box(o: Outline) -> tuple[int, int, int, int]:
    """The bounding box of a drawn shape in whole cells, clipped at zero."""
    x0, y0, x1, y1 = o.bbox()
    bx, by = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    return bx, by, max(1, int(np.ceil(x1)) - bx), max(1, int(np.ceil(y1)) - by)


class Domain(BaseModel):
    nx: int = Field(gt=0, description="cells along x")
    ny: int = Field(gt=0, description="cells along y")
    dx: float = Field(gt=0, description="cell size in the physics' length unit")
    #: schema 0.4: the domain's drawn outline, in cells (None: the whole grid),
    #: and holes cut in it.  The domain is the cells whose centres are inside.
    outline: Optional[Outline] = None
    holes: list[Outline] = Field(default_factory=list)


class Physics(BaseModel):
    """The family and its scalar parameters, by the names its registry entry declares."""

    family: str = "incompressible-2d"
    params: dict[str, float] = Field(default_factory=dict)

    def get(self, name: str) -> float:
        """A parameter, or the family's default when the case does not set it."""
        if name in self.params:
            return float(self.params[name])
        return float(registry.family(self.family).param(name).default)

    # the wind farm's two, read as they were in 0.2
    @property
    def nu(self) -> float:
        return self.get("nu")

    @property
    def u_inf(self) -> float:
        return self.get("u_inf")


class Window(BaseModel):
    """A window: a rectangle of cells (``x0, y0, nx, ny``), or -- schema 0.4 -- a
    drawn shape (``shape = "curve"``: an ``outline`` and optional ``holes``), in
    which case its rectangle is its bounding box, derived, not edited.  A window
    holds the cells of the domain whose centres it contains."""

    id: str
    shape: Literal["rect", "curve", "cells"] = "rect"
    x0: int = Field(0, ge=0)
    y0: int = Field(0, ge=0)
    nx: int = Field(1, gt=0)
    ny: int = Field(1, gt=0)
    outline: Optional[Outline] = None
    holes: list[Outline] = Field(default_factory=list)
    #: schema 0.5: a generated window's cells (``shape = "cells"``), as runs
    #: ``(row, first, stop)`` -- the cells ``first .. stop - 1`` of that row
    runs: Optional[list[tuple[int, int, int]]] = None

    @model_validator(mode="after")
    def _shape_matches(self):
        if self.shape == "curve":
            if self.outline is None:
                raise ValueError(f"window {self.id} is drawn but has no outline")
            if self.runs is not None:
                raise ValueError(f"window {self.id} is drawn: its cells are its outline's")
            self.x0, self.y0, self.nx, self.ny = _cell_box(self.outline)
        elif self.shape == "cells":
            if not self.runs:
                raise ValueError(f"window {self.id} is generated but holds no cells")
            if self.outline is not None or self.holes:
                raise ValueError(f"window {self.id} is generated: it has cells, not an "
                                 f"outline")
            if any(r < 0 or a < 0 or b <= a for r, a, b in self.runs):
                raise ValueError(f"window {self.id}: a run of cells is empty or negative")
            rows = [r for r, _a, _b in self.runs]
            x0, x1 = min(a for _r, a, _b in self.runs), max(b for _r, _a, b in self.runs)
            self.x0, self.y0, self.nx, self.ny = x0, min(rows), x1 - x0, max(rows) - min(rows) + 1
        elif self.outline is not None or self.holes or self.runs is not None:
            raise ValueError(f"window {self.id} is a rectangle and has no outline, holes or "
                             f"cells")
        return self


class Region(BaseModel):
    """A piece of the domain made of one material.

    A rectangle (``x0, y0, nx, ny`` in cells), a polygon (``points`` in cells,
    with optional ``holes``, usually imported from Gmsh), or -- schema 0.4 -- a
    drawn shape (``shape = "curve"``, its ``outline``); for the last two the
    rectangle is the bounding box and is derived, not edited.

    **Regions stack in list order**: where two overlap, the later one takes the
    cells.  So a plate with an insert is the plate, then the insert, with no need
    to cut the plate around it.  A cell's centre decides which region it is in.
    """
    id: str
    material: str = "material-1"
    shape: Literal["rect", "polygon", "curve"] = "rect"
    x0: int = Field(0, ge=0)
    y0: int = Field(0, ge=0)
    nx: int = Field(1, gt=0)
    ny: int = Field(1, gt=0)
    points: Optional[list[tuple[float, float]]] = None
    holes: Optional[list[list[tuple[float, float]]]] = None
    outline: Optional[Outline] = None

    @model_validator(mode="after")
    def _shape_matches(self):
        if self.shape == "curve":
            if self.outline is None:
                raise ValueError(f"region {self.id} is drawn but has no outline")
            if self.points is not None or self.holes is not None:
                raise ValueError(f"region {self.id} is drawn: its shape is its outline, "
                                 f"not points and holes")
            self.x0, self.y0, self.nx, self.ny = _cell_box(self.outline)
            return self
        if self.outline is not None:
            raise ValueError(f"region {self.id} has an outline but is not drawn")
        if self.shape == "polygon":
            if not self.points or len(self.points) < 3:
                raise ValueError("a polygon region needs at least three points")
            if any(len(h) < 3 for h in self.holes or []):
                raise ValueError("a hole needs at least three points")
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            if min(xs) < 0 or min(ys) < 0:
                raise ValueError("a polygon region has a point below zero")
            x0, y0 = int(np.floor(min(xs))), int(np.floor(min(ys)))
            self.x0, self.y0 = x0, y0
            self.nx = max(1, int(np.ceil(max(xs))) - x0)
            self.ny = max(1, int(np.ceil(max(ys))) - y0)
        elif self.points is not None or self.holes is not None:
            raise ValueError("a rectangular region has no points or holes")
        return self


class Boundary(BaseModel):
    """A condition on a segment of one domain edge, in cells along the edge.

    ``start`` and ``stop`` run left to right on the bottom and top edges, and
    bottom to top on the left and right edges; ``stop = None`` means the end.

    Schema 0.4: on a domain with drawn shapes, ``edge`` may name a drawn edge --
    ``outline:<k>`` or ``hole<h>:<k>`` -- and the condition holds on the whole of
    it (``start`` and ``stop`` are not used).  Each face of the domain's boundary
    belongs to the drawn edge nearest it.
    """
    id: str
    edge: str
    kind: str
    start: int = Field(0, ge=0)
    stop: Optional[int] = Field(None, gt=0)
    value: Optional[float] = None

    @field_validator("edge")
    @classmethod
    def _known_edge(cls, v: str) -> str:
        if v in geo.EDGES or geo.DRAWN_EDGE.match(v):
            return v
        raise ValueError(f"unknown edge {v!r}: left, right, bottom, top, outline:<k> or "
                         f"hole<h>:<k>")

    @property
    def drawn(self) -> bool:
        return self.edge not in geo.EDGES


class Device(BaseModel):
    id: str
    kind: Literal["actuator-disk"] = "actuator-disk"
    x: float = Field(description="disk plane, in the physics' length unit")
    y: float = Field(description="disk centre, in the physics' length unit")
    diameter: float = Field(1.0, gt=0)
    yaw_deg: float = 0.0


class Attachment(BaseModel):
    """A lumped part wired between two circuit nodes (schema 0.3, showcase case 5).

    A node is an **electrode** -- a boundary segment of kind ``electrode``, named
    by its id -- or a free node of the circuit (any other name).  The node
    ``ground`` is at 0 V; with no ground, the first electrode is.  A battery's
    EMF drives current out of its ``a`` terminal through the external circuit
    and back into ``b``: ``a`` is its positive terminal.
    """
    id: str
    kind: Literal["battery", "resistor"]
    value: float = Field(gt=0, description="EMF in V (battery), resistance in ohm")
    internal: float = Field(0.0, ge=0, description="a battery's internal resistance, ohm")
    a: str
    b: str


class Coupling(BaseModel):
    #: the showcase plan's coupling style (A, B, C, D) -- see `registry.STYLES`
    style: Literal["A", "B", "C", "D", "split"] = "A"
    ramp_cells: int = Field(8, ge=1, description="partition-of-unity ramp width")
    assembly: Literal["projected", "blend"] = "projected"
    elliptic: Literal["exposed", "embedded"] = "exposed"
    #: styles B, C, D: iterate until the update (max norm, relative to the
    #: field's scale) is below this
    tolerance: float = Field(1e-10, gt=0, lt=1)
    max_iterations: int = Field(500, ge=1)
    #: styles C and D: the first relaxation factor, and whether Aitken's rule
    #: adapts it from the second iteration on
    relaxation: float = Field(0.5, gt=0, le=2)
    aitken: bool = True
    #: style C: which window is the Dirichlet side ("auto": the one with the
    #: lower mean conductivity, the side the 1-D analysis says contracts)
    dirichlet_side: str = "auto"


class Layout(BaseModel):
    """Windows generated from the geometry (schema 0.5, `layout.py`).

    ``cut``: ``along`` cuts the domain into ``along`` pieces along its length at
    level curves of its harmonic coordinate between its two ends, and each of
    those into ``across`` pieces between its two sides; ``materials`` makes one
    piece per material.  For styles A and B each piece grows into the domain
    until every cell has a window at full weight; for C, D and the split the
    pieces meet along faces.

    ``source`` fingerprints what the windows were generated from (the domain,
    the materials, the style, the ramp, these settings): an edit that changes it
    generates them again, and an edit that does not leaves them as they are.
    ``grown`` is how far each piece reached, in cells.
    """

    cut: Literal["along", "materials"] = "along"
    along: int = Field(3, ge=1, le=32)
    across: int = Field(1, ge=1, le=8)
    source: str = ""
    grown: int = 0


class RunSettings(BaseModel):
    macro_dt: float = Field(0.2, gt=0)
    #: macro-steps of a transient run; timed repeats of the solve of a steady one
    steps: int = Field(40, gt=0)
    threads: int = Field(4, ge=1)
    start: Literal["freestream"] = "freestream"
    mode: Literal["transient", "steady"] = "transient"


class Compare(BaseModel):
    full_domain: bool = True
    metrics: list[str] = Field(default_factory=lambda: ["farm_power", "rms_velocity",
                                                        "time_per_step"])


class CaseSpec(BaseModel):
    schema_id: str = SCHEMA_ID
    name: str = "untitled"
    description: str = ""
    domain: Domain
    physics: Physics
    #: material name -> {property: value}, SI; the regions refer to these names
    materials: dict[str, dict[str, float]] = Field(default_factory=dict)
    regions: list[Region] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    devices: list[Device] = Field(default_factory=list)
    boundaries: list[Boundary] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    coupling: Coupling = Field(default_factory=Coupling)
    run: RunSettings = Field(default_factory=RunSettings)
    compare: Compare = Field(default_factory=Compare)
    #: schema 0.5: when set, the windows are generated from the geometry and follow
    #: it (`layout.py`); None when the windows are the case's own
    layout: Optional[Layout] = None

    @field_validator("schema_id")
    @classmethod
    def _known_schema(cls, v: str) -> str:
        if v not in READABLE:
            raise ValueError(f"unknown case schema {v!r}; this workbench reads "
                             f"{', '.join(READABLE)}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data):
        """0.1 -> 0.2 -> 0.3 -> 0.4, one step at a time.

        0.1 -> 0.2: the family's fixed boundaries, no regions.
        0.2 -> 0.3: ``nu`` and ``u_inf`` move into ``physics.params``; the style
        is the family's first; no materials, no attachments; a transient run.
        0.3 -> 0.4: nothing to move -- every drawn shape is optional, so a 0.3
        case is a 0.4 case with none.
        0.4 -> 0.5: nothing to move -- a 0.4 case's windows are its own (no
        layout).
        """
        if not isinstance(data, dict):
            return data
        if data.get("schema_id") == "atlas-workbench/case@0.1":
            data = dict(data)
            fam = (data.get("physics") or {}).get("family", "incompressible-2d")
            data.setdefault("boundaries", [b.model_dump() for b in family_boundaries(fam)])
            data.setdefault("regions", [])
            data["schema_id"] = "atlas-workbench/case@0.2"
        if data.get("schema_id") == "atlas-workbench/case@0.2":
            data = dict(data)
            ph = dict(data.get("physics") or {})
            params = dict(ph.get("params") or {})
            for key in ("nu", "u_inf"):
                if key in ph:
                    params[key] = ph.pop(key)
            ph["params"] = params
            data["physics"] = ph
            cp = dict(data.get("coupling") or {})
            cp.setdefault("style", _first_style(ph.get("family", "incompressible-2d")))
            data["coupling"] = cp
            data.setdefault("materials", {})
            data.setdefault("attachments", [])
            data["schema_id"] = "atlas-workbench/case@0.3"
        if data.get("schema_id") == "atlas-workbench/case@0.3":
            data = dict(data)
            data["schema_id"] = "atlas-workbench/case@0.4"
        if data.get("schema_id") == "atlas-workbench/case@0.4":
            data = dict(data)                      # 0.4 -> 0.5: the layout is optional
            data["schema_id"] = SCHEMA_ID
        return data

    # -- persistence ---------------------------------------------------------
    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "CaseSpec":
        """Parsed to a dict first, so the migration always sees the old shape."""
        return cls.model_validate(json.loads(text))

    def save(self, path: str) -> str:
        """Write atomically; OneDrive can hold a just-written file, so retry."""
        import time
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        for k in range(40):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if k == 39:
                    raise
                time.sleep(0.25)
        return path

    @classmethod
    def load(cls, path: str) -> "CaseSpec":
        with open(path, encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    def copy_deep(self) -> "CaseSpec":
        return self.model_copy(deep=True)


def _first_style(fid: str) -> str:
    try:
        fam = registry.family(fid)
    except KeyError:
        return "A"
    return fam.styles[0] if fam.styles else "A"


def family_boundaries(fid: str) -> list[Boundary]:
    """The boundaries a family's solver fixes, as whole-edge segments; for a family
    whose boundaries are editable, the ones a new case of it starts with."""
    try:
        fam = registry.family(fid)
    except KeyError:
        return []
    if fam.fixed_boundaries:
        return [Boundary(id=f"B-{b.edge}", edge=b.edge, kind=b.kind)
                for b in fam.fixed_boundaries]
    return [Boundary(id=f"B-{edge}", edge=edge, kind=kind, value=value)
            for edge, kind, value in fam.default_boundaries]


def derived_kind(spec: "CaseSpec", name: str) -> str:
    """The condition a family gives the drawn edge ``name`` of its own accord: the
    kind the family's solver fixes on the grid edge the drawn edge lies along
    (`geometry.along_grid_edge`), or its ``drawn_default``.  A river drawn from the
    grid's left edge to its right edge enters on the left and leaves on the right;
    a wind farm's drawn edges away from the grid's edges are walls."""
    fam = registry.family(spec.physics.family)
    e = geo.along_grid_edge(spec.domain, name)
    for f in fam.fixed_boundaries:
        if f.edge == e:
            return f.kind
    if fam.drawn_default:
        return fam.drawn_default
    return fam.boundary_kinds[0] if fam.boundary_kinds else "insulated"


def default_boundaries(c: "CaseSpec", fid: str | None = None) -> list[Boundary]:
    """A family's boundaries for the case's domain: `family_boundaries` on the grid's
    edges (when the domain has no drawn outline), and one per drawn edge, of its
    `derived_kind`."""
    fid = fid or c.physics.family
    d = c.domain
    out = [] if d.outline is not None else family_boundaries(fid)
    if d.outline is not None or d.holes:
        probe = c.model_copy(update={"physics": c.physics.model_copy(update={"family": fid})})
        taken = {b.id for b in out}
        for name in geo.drawn_edge_names(d):
            bid = "B-" + name.replace(":", "-")
            k = 2
            while bid in taken:
                bid, k = f"B-{name.replace(':', '-')}-{k}", k + 1
            taken.add(bid)
            out.append(Boundary(id=bid, edge=name, kind=derived_kind(probe, name)))
    return out


def derive_boundaries(c: "CaseSpec") -> bool:
    """For a family whose solver fixes its drawn edges' conditions (``drawn_derived``),
    give every drawn edge its `derived_kind`; True when anything changed.  Called on
    every edit, so moving a vertex off the grid's edge turns an inlet into a wall."""
    try:
        fam = registry.family(c.physics.family)
    except KeyError:
        return False
    d = c.domain
    if not fam.drawn_derived or (d.outline is None and not d.holes):
        return False
    changed = False
    for b in c.boundaries:
        if b.drawn:
            want = derived_kind(c, b.edge)
            if b.kind != want:
                b.kind, b.value = want, None
                changed = True
    return changed


@dataclass(frozen=True)
class FamilySetup:
    """What a family's first example runs with: the scale its parameters' defaults
    belong to, the coupling it converges with, and how many windows it is cut into."""
    example: str
    dx: float
    macro_dt: float
    steps: int
    mode: str
    coupling: dict
    windows: int


_SETUPS: dict[str, FamilySetup | None] = {}


def family_setup(fid: str) -> FamilySetup | None:
    """The family's first example's setup (`FamilySetup`), or None without one."""
    if fid not in _SETUPS:
        _SETUPS[fid] = None
        for key, ex in EXAMPLES.items():
            if ex.family == fid:
                s = example_case(key)
                _SETUPS[fid] = FamilySetup(key, float(s.domain.dx), float(s.run.macro_dt),
                                           int(s.run.steps), s.run.mode,
                                           s.coupling.model_dump(), len(s.windows))
                break
    return _SETUPS[fid]


def adapt_to_family(c: "CaseSpec", fid: str) -> list[str]:
    """Make a case fit a newly chosen family, keeping its shape; returns what changed
    that the person should be told.

    A family actually changed starts from its first example's setup
    (`family_setup`): its cell size, macro-step, step count and mode, the coupling
    it converges with, and automatic windows as many as it has (the wind farm keeps
    its rectangles, the tiling its graph compiles).  The owner's first scenario
    (2026-09-30) is why: a wind farm's cell of 0.03125 rotor diameters kept as metres
    made a river 11 m long with its outfall's default outside it, and the farm's six
    windows and 500 iterations left a drawn plate's Schwarz unconverged.

    Rotors, circuit parts and material regions the family does not read are removed
    (the farm's rotors, checked by a river, were errors no one could see the point
    of).  Parameters the family declares keep a value the case already has; the
    style and the run mode fall back to the family's first if the case's is not one
    it runs; boundaries are replaced when the family fixes its own or cannot impose
    one the case has (on a drawn domain, by one per drawn edge); region materials
    the case does not define are taken from the family's showcase library."""
    fam = registry.family(fid)
    changed = c.physics.family != fid
    notes: list[str] = []
    for layer, what in (("devices", "rotor"), ("attachments", "circuit part"),
                        ("regions", "material region")):
        items = getattr(c, layer)
        if changed and items and layer not in fam.layers:
            notes.append(f"the {len(items)} {what}{'s' * (len(items) != 1)} removed, which "
                         f"this physics does not read")
            setattr(c, layer, [])
    c.physics.family = fid
    c.physics.params = {p.name: float(c.physics.params.get(p.name, p.default))
                        for p in fam.params}
    setup = family_setup(fid) if changed else None
    if setup is not None:
        c.domain.dx, c.run.macro_dt, c.run.steps = setup.dx, setup.macro_dt, setup.steps
        if setup.mode in fam.modes:
            c.run.mode = setup.mode
        c.coupling = Coupling(**setup.coupling)
        unit = fam.length_unit or "cells"
        notes.append(f"the {setup.example} example's scale: cells of {setup.dx:g} {unit}, "
                     f"{setup.steps:,} steps of {setup.macro_dt:g}"
                     + (" s" if unit == "m" else ""))
        if fid != "incompressible-2d":
            c.layout = Layout(cut="along", along=setup.windows, across=1)
            notes.append(f"{setup.windows} window{'s' * (setup.windows != 1)}, cut "
                         f"automatically from the shape")
    if c.coupling.style not in fam.styles and fam.styles:
        c.coupling.style = fam.styles[0]
    if c.run.mode not in fam.modes:
        c.run.mode = fam.modes[0]
    if fam.fixed_boundaries or any(b.kind not in fam.boundary_kinds for b in c.boundaries):
        c.boundaries = default_boundaries(c, fid)
    lib = fam.material_library()
    for r in c.regions:
        if r.material not in c.materials and r.material in lib:
            c.materials[r.material] = dict(lib[r.material])
    return notes


# ---------------------------------------------------------------------------
# checking a case
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    severity: Literal["error", "warning", "info"]
    step: str          # which workflow step fixes it: "case", "geometry", ...
    message: str
    #: a repair the page can make in one click (a key of `starter.FIXES`), and that
    #: a newly drawn shape or newly chosen physics gets without asking
    fix: str | None = None


def check(spec: CaseSpec) -> list[Issue]:
    """What is wrong with a case, in the order a person would fix it.

    Errors stop a run; warnings are worth reading; info is context.  The window
    rules are `geometry.analyse_windows`, the same computation the canvas draws.
    """
    out: list[Issue] = []
    d = spec.domain
    if not spec.name.strip():
        out.append(Issue("warning", "case", "the case has no name"))

    ids = [w.id for w in spec.windows]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        out.append(Issue("error", "geometry", f"two windows are both called {dup!r}"))
    dids = [v.id for v in spec.devices]
    for dup in sorted({i for i in dids if dids.count(i) > 1}):
        out.append(Issue("error", "geometry", f"two devices are both called {dup!r}"))

    plain = geo.is_plain(spec)
    if not plain:
        out += _check_drawn(spec)
    if not spec.windows:
        out.append(Issue("error", "geometry", "the domain is not cut into any windows yet "
                         "(Windows)", fix="windows"))
    if plain:
        covered = np.zeros((d.ny, d.nx), dtype=bool)
        for w in spec.windows:
            if w.x0 + w.nx > d.nx or w.y0 + w.ny > d.ny:
                out.append(Issue("error", "geometry",
                                 f"window {w.id} reaches past the domain "
                                 f"(x {w.x0}..{w.x0 + w.nx}, y {w.y0}..{w.y0 + w.ny} "
                                 f"against {d.nx} x {d.ny} cells)"))
            covered[w.y0:w.y0 + w.ny, w.x0:w.x0 + w.nx] = True
        if spec.windows:
            gap = int((~covered).sum())
            if gap:
                out.append(Issue("error", "geometry",
                                 f"{gap} of {d.nx * d.ny} cells lie in no window (Windows)",
                                 fix="windows"))
    else:
        act = geo.domain_mask(d)
        covered = np.zeros_like(act)
        for wid, m in geo.window_masks(spec):
            if not m.any():
                out.append(Issue("error", "geometry",
                                 f"window {wid} holds no cell of the domain: no cell centre "
                                 f"lies inside both"))
            covered |= m
        if spec.windows:
            gap = int((act & ~covered).sum())
            if gap:
                out.append(Issue("error", "geometry",
                                 f"{gap} of the domain's {int(act.sum())} cells lie in no "
                                 f"window (Windows)", fix="windows"))

    ramp = spec.coupling.ramp_cells
    style = spec.coupling.style
    if spec.windows and style in ("C", "D"):
        out += _check_pieces(spec)
    elif spec.windows and style == "split":
        if plain:
            whole = [w for w in spec.windows
                     if (w.x0, w.y0, w.nx, w.ny) == (0, 0, d.nx, d.ny)]
        else:
            act = geo.domain_mask(d)
            whole = [wid for wid, m in geo.window_masks(spec) if np.array_equal(m, act)]
        if len(spec.windows) != 1 or not whole:
            out.append(Issue("error", "geometry",
                             "a split by physics shares the whole domain between its "
                             "agents, so the case is one window covering the domain; this "
                             f"one has {len(spec.windows)} (Windows)", fix="windows"))
    elif spec.windows:
        an = geo.analyse_case(spec)
        n_ramp = int(an.ramp_only.sum())
        if n_ramp:
            names = ", ".join(sorted(an.ramp_only_in)[:8])
            more = "" if len(an.ramp_only_in) <= 8 else f" and {len(an.ramp_only_in) - 8} more"
            out.append(Issue("error", "geometry",
                             f"{n_ramp} cells have no window at full weight (in {names}{more}): "
                             f"where windows meet they must overlap by at least twice the "
                             f"ramp ({2 * ramp} cells), as every measured tiling does "
                             f"(Windows)", fix="windows"))
        for a, b, t in an.thin_pairs:
            what = "touch without overlapping" if t == 0 else f"overlap by only {t} cells"
            out.append(Issue("warning", "geometry",
                             f"windows {a} and {b} {what}; the ramp needs {2 * ramp}"))
        for inner, outer in an.nested:
            out.append(Issue("warning", "geometry",
                             f"window {inner} lies entirely inside {outer}: it costs a solve "
                             f"and covers nothing new"))
        if an.cross_points:
            out.append(Issue("info", "geometry",
                             f"{len(an.cross_points)} cross-point(s), where three or more "
                             f"windows overlap; the compiler's rules for them are L2/I2/G1"))

    wx, wy = d.nx * d.dx, d.ny * d.dx
    for v in spec.devices:
        if not (0.0 <= v.x <= wx and v.diameter / 2 <= v.y <= wy - v.diameter / 2):
            out.append(Issue("error", "geometry",
                             f"device {v.id} is not fully inside the domain"))
        if v.yaw_deg != 0.0:
            out.append(Issue("error", "geometry",
                             f"device {v.id} is yawed {v.yaw_deg:g} degrees; the actuator "
                             f"disk here has no yaw model (disk.ActuatorDisk), so set it "
                             f"to 0"))
    if spec.physics.family == "incompressible-2d" and (d.outline is not None or d.holes):
        # a drawn domain: the fluid is its cells, the rest a solid held at rest
        act = geo.domain_mask(d)
        for v in spec.devices:
            j0 = int(np.floor((v.y - v.diameter / 2) / d.dx))
            j1 = int(np.ceil((v.y + v.diameter / 2) / d.dx))
            i = int(np.floor(v.x / d.dx))
            if 0 <= i < d.nx and not act[max(j0, 0):min(j1, d.ny), i].all():
                out.append(Issue("error", "geometry",
                                 f"rotor {v.id}'s disk reaches outside the drawn domain, into "
                                 f"the solid: move it into the air (Rotors)"))
        if d.outline is not None:
            along = {geo.along_grid_edge(d, n) for n in geo.drawn_edge_names(d)}
            if "left" not in along:
                out.append(Issue("error", "geometry",
                                 "the flow enters on the grid's left edge, and no edge of the "
                                 "drawn domain runs along it: draw the shape so one of its "
                                 "edges lies along the grid's left edge (Shape)"))
            if "right" not in along:
                # the page's first drawn farm reached the right edge at one vertex only,
                # and the check said nothing (2026-09-29)
                out.append(Issue("error", "geometry",
                                 "the flow leaves on the grid's right edge, and no edge of the "
                                 "drawn domain runs along it: draw the shape so one of its "
                                 "edges lies along the grid's right edge (Shape)"))
    if spec.physics.family == "incompressible-2d":
        re_cell = d.dx * spec.physics.u_inf / spec.physics.nu
        if re_cell > 8.0:
            out.append(Issue("warning", "physics",
                             f"the cell Reynolds number dx U / nu is {re_cell:.3g}; the "
                             f"window solver's own validity predicate asks for at most 8 "
                             f"(wake_array.FluidWindow.reference_validity)"))

    for part in (_check_regions, _check_boundaries, _check_physics, _check_attachments):
        try:
            out += part(spec)
        except Exception as exc:
            # a check must never take the page down: it runs on every edit, and one
            # that raised left a newly drawn block unusable (2026-09-30)
            out.append(Issue("error", "physics",
                             f"the case could not be checked ({part.__name__[1:]}): "
                             f"{type(exc).__name__}: {exc}"))
    return out


def _check_pieces(spec: CaseSpec) -> list[Issue]:
    """Styles C and D cut the domain into pieces that do not overlap.

    Style C couples exactly two pieces across one interface (Dirichlet-Neumann
    is a two-piece method; more pieces would need cross-point rules this runner
    does not have).  Style D joins ONE field to lumped parts, so its window is
    the whole plate.
    """
    out: list[Issue] = []
    d = spec.domain
    plain = geo.is_plain(spec)
    count = np.zeros((d.ny, d.nx), dtype=np.int32)
    if plain:
        for w in spec.windows:
            count[w.y0:w.y0 + w.ny, w.x0:w.x0 + w.nx] += 1
    else:
        masks = dict(geo.window_masks(spec))
        for m in masks.values():
            count += m
    over = int((count > 1).sum())
    if over:
        out.append(Issue("error", "geometry",
                         f"{over} cells lie in more than one window; style "
                         f"{spec.coupling.style} cuts the domain into pieces that meet "
                         f"along faces and do not overlap (Windows)", fix="windows"))
    if spec.coupling.style == "C" and len(spec.windows) != 2:
        out.append(Issue("error", "geometry",
                         f"style C (Dirichlet-Neumann) couples exactly two pieces; this "
                         f"case has {len(spec.windows)} (Windows)", fix="windows"))
    if spec.coupling.style == "C" and len(spec.windows) == 2 and not over:
        a, b = spec.windows
        if plain:
            touch_x = (a.x0 + a.nx == b.x0 or b.x0 + b.nx == a.x0) and not (
                a.y0 + a.ny <= b.y0 or b.y0 + b.ny <= a.y0)
            touch_y = (a.y0 + a.ny == b.y0 or b.y0 + b.ny == a.y0) and not (
                a.x0 + a.nx <= b.x0 or b.x0 + b.nx <= a.x0)
            meet = touch_x or touch_y
        else:
            meet = geo.faces_between(masks[a.id], masks[b.id]) > 0
        if not meet:
            out.append(Issue("error", "geometry",
                             f"windows {a.id} and {b.id} do not meet along a face "
                             f"(Windows)", fix="windows"))
        side = spec.coupling.dirichlet_side
        if side != "auto" and side not in (a.id, b.id):
            out.append(Issue("error", "physics",
                             f"the Dirichlet side {side!r} is not one of the two windows "
                             f"({a.id}, {b.id})"))
    if spec.coupling.style == "D" and len(spec.windows) != 1:
        out.append(Issue("error", "geometry",
                         f"style D joins one field to lumped parts, so the plate is one "
                         f"window covering the domain; this case has "
                         f"{len(spec.windows)} (Windows)", fix="windows"))
    return out


def drawn_shapes(spec: CaseSpec) -> list[tuple[str, Outline]]:
    """Every drawn shape in the case, named the way the check names it."""
    d = spec.domain
    out: list[tuple[str, Outline]] = []
    if d.outline is not None:
        out.append(("the domain's outline", d.outline))
    for h, o in enumerate(d.holes):
        out.append((f"hole {h}", o))
    for w in spec.windows:
        if w.shape == "curve":
            out.append((f"window {w.id}", w.outline))
            for h, o in enumerate(w.holes):
                out.append((f"window {w.id}'s hole {h}", o))
    for r in spec.regions:
        if r.shape == "curve":
            out.append((f"region {r.id}", r.outline))
    return out


def _check_drawn(spec: CaseSpec) -> list[Issue]:
    """Drawn shapes: each one well formed and on the grid, the holes inside the
    outline, and a family whose solvers can run on them."""
    out: list[Issue] = []
    d = spec.domain
    fam = _family(spec)
    if fam is not None and not fam.drawn_shapes:
        out.append(Issue("error", "geometry",
                         f"the {fam.label} family runs on rectangles only, so its domain "
                         f"and windows cannot be drawn shapes: {fam.drawn_why}"))
    for name, o in drawn_shapes(spec):
        why = o.problems()
        for p in why:
            out.append(Issue("error", "geometry", f"{name}: {p}"))
        if why:
            continue
        # a window or a region may reach past the grid: it holds only the domain's
        # cells whose centres it contains.  The domain itself may not: the part
        # past the grid would silently not be in it.
        if not (name == "the domain's outline" or name.startswith("hole ")):
            continue
        x0, y0, x1, y1 = o.bbox()
        if x0 < -1e-9 or y0 < -1e-9 or x1 > d.nx + 1e-9 or y1 > d.ny + 1e-9:
            out.append(Issue("error", "geometry",
                             f"{name} reaches past the {d.nx} x {d.ny} grid; draw it inside, "
                             f"or make the grid bigger (Model, Physics)"))
    if d.outline is not None and not d.outline.problems():
        ring = d.outline.ring()
        for h, o in enumerate(d.holes):
            if not o.problems() and not np.all(shapes.contains(ring, o.ring(step=1.0))):
                out.append(Issue("error", "geometry",
                                 f"hole {h} is not inside the domain's outline"))
    if not any(i.severity == "error" for i in out):
        act = geo.domain_mask(d)
        if not act.any():
            out.append(Issue("error", "geometry", "the domain contains no cell centre"))
    return out


def _check_physics(spec: CaseSpec) -> list[Issue]:
    out: list[Issue] = []
    fam = _family(spec)
    if fam is None:
        return [Issue("error", "case", f"unknown physics family {spec.physics.family!r}")]
    if fam.status != "ready-to-wire":
        out.append(Issue("error", "case",
                         f"the {fam.label} family is not available yet: {fam.note}"))
        return out
    if spec.coupling.style not in fam.styles:
        out.append(Issue("error", "physics",
                         f"the {fam.id} family runs style "
                         f"{' or '.join(fam.styles)}, not {spec.coupling.style}"))
    if spec.run.mode not in fam.modes:
        out.append(Issue("error", "physics",
                         f"the {fam.id} family runs {' or '.join(fam.modes)}, not "
                         f"{spec.run.mode}"))
    for p in fam.params:
        why = p.problem(spec.physics.get(p.name))
        if why:
            out.append(Issue("error", "physics", why))
    unknown = sorted(set(spec.physics.params) - {p.name for p in fam.params})
    if unknown:
        out.append(Issue("warning", "physics",
                         f"the {fam.id} family does not read {', '.join(unknown)}"))
    if "regions" in fam.layers:
        used = sorted({r.material for r in spec.regions})
        for m in used:
            props = spec.materials.get(m)
            if props is None:
                out.append(Issue("error", "physics",
                                 f"material {m!r} is used by a region and has no "
                                 f"properties (Model, Materials)"))
                continue
            for p in fam.material_props:
                if p.name not in props:
                    out.append(Issue("error", "physics",
                                     f"material {m!r} has no {p.label.lower()} "
                                     f"({p.name}, {p.unit})"))
                    continue
                why = p.problem(props[p.name])
                if why:
                    out.append(Issue("error", "physics", f"material {m!r}: {why}"))
    if fam.id == "transport-2d":
        out += _check_transport(spec, materials_ok=not any(i.severity == "error"
                                                           for i in out))
    if fam.id == "acoustics-2d":
        out += _check_acoustics(spec, materials_ok=not any(i.severity == "error"
                                                           for i in out))
    if fam.id == "conjugate-heat-2d":
        out += _check_cooling(spec, materials_ok=not any(i.severity == "error" for i in out))
    if (fam.id in ("conduction-2d", "conjugate-heat-2d") and spec.coupling.style == "C"
            and spec.run.mode == "steady" and not any(i.severity == "error" for i in out)
            and not any(i.severity == "error" for i in _check_pieces(spec))):
        out += _check_floating(spec)
    if fam.id == "elasticity-2d":
        if not any(b.kind == "clamped" for b in spec.boundaries):
            out.append(Issue("error", "geometry",
                             "nothing holds the plate: clamp at least one edge segment (this "
                             "family does not remove the rigid-body motions of a free body) "
                             "(Boundaries)", fix="clamp"))
        if not any(b.kind in ("load-x", "load-y") and b.value for b in spec.boundaries):
            out.append(Issue("warning", "geometry",
                             "no edge is loaded, so the plate will not move (Boundaries)",
                             fix="load"))
    if fam.id in ("conduction-2d", "thermoelastic-2d") and not any(
            b.kind in ("fixed-temperature", "heat-flux") for b in spec.boundaries):
        # a new shape's edges start insulated: nothing would warm or cool it
        out.append(Issue("warning", "geometry",
                         "every edge is insulated, so nothing heats or cools the domain: "
                         "hold an edge at a temperature, or give it a heat flux "
                         "(Boundaries)", fix="temperatures"))
    if fam.id == "incompressible-2d":
        if spec.coupling.assembly == "blend":
            out.append(Issue("warning", "physics",
                             "a blend without the global projection leaves the band at "
                             "macro-step 70-80 at six windows (W100)"))
        if spec.coupling.elliptic == "embedded":
            out.append(Issue("warning", "physics",
                             "an embedded pressure solve in every window is refused by "
                             "L2/R10 and is unstable composed (W100)"))
    return out


def _check_transport(spec: CaseSpec, materials_ok: bool) -> list[Issue]:
    """The river: the outfall is in it, and the explicit step is under its limit."""
    out: list[Issue] = []
    d = spec.domain
    sx, sy = spec.physics.get("source_x"), spec.physics.get("source_y")
    lx, ly = d.nx * d.dx, d.ny * d.dx
    drawn = d.outline is not None or bool(d.holes)
    if not (0.0 <= sx < lx and 0.0 <= sy < ly):
        out.append(Issue("error", "physics",
                         f"the outfall, where the pollutant is released, is at ({sx:g} m, "
                         f"{sy:g} m): not in the river, outside the grid's {lx:g} m x "
                         f"{ly:g} m (Physics)", fix="outfall"))
    elif drawn and not geo.domain_mask(d)[int(sy // d.dx), int(sx // d.dx)]:
        out.append(Issue("error", "physics",
                         f"the outfall, where the pollutant is released, is at ({sx:g} m, "
                         f"{sy:g} m): on dry ground, outside the drawn river (Physics)",
                         fix="outfall"))
    if drawn:
        # a drawn river's flow is solved from its inlets to its outlets (flow.py)
        kinds = {b.kind for b in spec.boundaries if b.drawn or d.outline is None}
        for kind, what, fix in (("river-inlet", "no inlet: mark the edge where the water "
                                 "comes in", "river-inlet"),
                                ("river-outlet", "no outlet: mark the edge where the water "
                                 "leaves", "river-outlet")):
            if kind not in kinds:
                out.append(Issue("error", "geometry",
                                 f"the river has {what} as {kind} (Boundaries)", fix=fix))
        if any(i.severity == "error" for i in out) or any(
                i.severity == "error" for i in _check_boundaries(spec)):
            return out
        from .families.plume import river_flow
        from .flow import FlowError
        try:
            river_flow(spec)
        except FlowError as exc:
            return out + [Issue("error", "geometry", f"the river's flow: {exc}")]
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    if materials_ok and spec.regions and not np.any((owner < 0) & geo.domain_mask(d)):
        from .families.plume import explicit_limit       # the family's own arithmetic
        lim = explicit_limit(spec)
        if spec.run.macro_dt > lim:
            out.append(Issue("error", "physics",
                             f"the macro-step {spec.run.macro_dt:g} s is over the explicit "
                             f"step's stability limit here, {lim:.4g} s (min over the cells "
                             f"of depth dx^2 over the cell's outflow and mixing "
                             f"conductances); take at most that (Physics)", fix="time-step"))
    return out


def _check_acoustics(spec: CaseSpec, materials_ok: bool) -> list[Issue]:
    """Two pieces side by side, a step under the leapfrog's limit, and a pulse the
    reflection can be read from."""
    from .families import acoustics as ac                # the family's own arithmetic
    out: list[Issue] = []
    d = spec.domain
    plain = geo.is_plain(spec)
    m = ac.cut_column(spec) if plain else None
    if spec.windows and plain and m is None:
        out.append(Issue("error", "geometry",
                         "the acoustics family's pieces on a rectangle are two windows side "
                         "by side, each the full height of the domain: the wave runs along x "
                         "(draw the pieces, or the domain, for any other cut)"))
    if not plain:
        out.append(Issue("info", "physics",
                         "a drawn domain or pieces: the reflection is not read against the "
                         "textbook (that needs two uniform media meeting at a straight cut); "
                         "the energy and the pieces' agreement with the full domain are"))
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    if not (materials_ok and spec.regions and not np.any((owner < 0) & geo.domain_mask(d))):
        return out
    lim = ac.stable_dt(spec)
    if spec.run.macro_dt > lim:
        out.append(Issue("error", "physics",
                         f"the macro-step {spec.run.macro_dt:.4g} s is over the leapfrog's "
                         f"stability limit here, {lim:.4g} s (Gershgorin on the grid's "
                         f"operator; dx / (c sqrt 2) in one medium); take at most that "
                         f"(Physics)", fix="time-step"))
    if m is not None:
        p = ac.pulse(spec)
        if not 0.0 < p.x0 < m * d.dx:
            out.append(Issue("error", "physics",
                             f"the pulse starts at {p.x0:g} m, outside the first piece "
                             f"(0 to {m * d.dx:g} m)"))
        else:
            r, why = ac.closed_form_reflection(spec)
            if r is None:
                out.append(Issue("warning", "physics",
                                 f"the reflection will not be read against the textbook: "
                                 f"{why}"))
            elif ac.plateau(spec, spec.run.macro_dt) is None:
                out.append(Issue("warning", "physics",
                                 "the domain is too short for the reflected pulse to be "
                                 "alone in the first medium, so the reflection will not be "
                                 "read"))
    return out


def _check_floating(spec: CaseSpec) -> list[Issue]:
    """A steady Dirichlet-Neumann piece given only a flow must have something of its
    own that sets its temperature level (conduction.floating); the case's choice of
    Dirichlet side must not leave the Neumann piece floating."""
    from .families import conduction as cd
    from .flow import FlowError
    d = spec.domain
    try:
        if spec.physics.family == "conjugate-heat-2d":
            from .families import cooling as co
            f = co.field_from_case(spec)
        else:
            f = cd.field_from_case(spec)
    except (FlowError, ValueError):
        # the coolant's flow is not solvable yet (no inlet, no channel): the checks
        # of the coolant say why; this one needs the field (seen: the check crashed
        # on a newly drawn block, 2026-09-30)
        return []
    wins = spec.windows
    fl = {w.id: cd.floating(f, cd.cells_of(spec, w)) for w in wins}
    if all(fl.values()):
        return [Issue("error", "physics",
                      "both pieces float: nothing on the domain's edges sets a temperature "
                      "(a fixed temperature, or a coolant that leaves), so no steady state "
                      "exists (Boundaries)", fix="temperatures")]
    _d_id, n_id = cd.dirichlet_side(spec, f)
    if fl[n_id]:
        return [Issue("error", "physics",
                      f"the Neumann side, {n_id}, floats: nothing of its own sets its "
                      f"temperature level, so its solve is singular. Make it the Dirichlet "
                      f"side (or choose auto)")]
    return []


#: what a newly drawn block lacks, and where to add it
_NO_COOLANT = ("nothing cools the block: no material flows. Draw the channel as a "
               "region made of water (Materials), then mark where the water enters "
               "and leaves: coolant-inlet and coolant-outlet (Boundaries)")


def _check_cooling(spec: CaseSpec, materials_ok: bool) -> list[Issue]:
    """The coolant fills whole rows, enters on the left and leaves on the right of
    exactly those rows, and the seam between the pieces does not cut its flow."""
    from .families import cooling as co                  # the family's own arithmetic
    out: list[Issue] = []
    d = spec.domain
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    if not (materials_ok and spec.regions
            and not np.any((owner < 0) & geo.domain_mask(d))):
        return out
    if not co.is_plug(spec):
        return _check_coolant_flow(spec)
    rows, why = co.coolant_rows(spec)
    if why:
        return [Issue("error", "geometry", why)]
    if rows.size == 0:
        out.append(Issue("error", "physics", _NO_COOLANT))
    for edge, kind in (("left", "coolant-inlet"), ("right", "coolant-outlet")):
        kinds = np.full(d.ny, "", dtype=object)
        for b in spec.boundaries:
            if b.edge == edge:
                kinds[b.start:(d.ny if b.stop is None else b.stop)] = b.kind
        wrong = [j for j in rows if kinds[j] != kind]
        if wrong:
            out.append(Issue("error", "geometry",
                             f"the coolant runs in rows {rows.min()}-{rows.max()}, so the "
                             f"{edge} edge there must be a {kind}"))
        stray = [j for j in range(d.ny) if kinds[j] == kind and j not in set(rows.tolist())]
        if stray:
            out.append(Issue("error", "geometry",
                             f"a {kind} on the {edge} edge covers rows the coolant does not "
                             f"run in"))
    if not np.any(co.material_grid(spec, "heat") > 0.0):
        out.append(Issue("warning", "physics", "no material generates heat, so nothing warms"))
    if len(spec.windows) == 2 and rows.size:
        flow_rows = set(rows.tolist())
        for w in spec.windows:
            inside = set(range(w.y0, w.y0 + w.ny)) & flow_rows
            if inside and (w.x0 != 0 or w.x0 + w.nx != d.nx):
                out.append(Issue("error", "geometry",
                                 f"window {w.id} cuts the coolant's flow; make one window the "
                                 f"channel and the other the block, meeting at the wall"))
    return out


def _check_coolant_flow(spec: CaseSpec) -> list[Issue]:
    """A coolant that is not whole rows, or a drawn domain (case file 0.4): its flow is
    solved through its own cells from its inlets to its outlets (`flow.py`).  The
    inlets and outlets must be on the coolant, the flow must be solvable, and the
    coolant must lie in one piece: no flow may cross the seam."""
    from .families import cooling as co
    from .flow import FlowError, case_faces
    out: list[Issue] = []
    carrier = co.coolant_mask(spec)
    if not carrier.any():
        return [Issue("error", "physics", _NO_COOLANT)]
    bf, kinds = case_faces(spec, "coolant-inlet", "coolant-outlet")
    on = carrier.ravel()[bf.cell]
    if np.any((kinds != "") & ~on):
        out.append(Issue("error", "geometry",
                         "a coolant inlet or outlet lies on the block's cells: the coolant "
                         "enters and leaves through the edges of its own cells"))
    if not np.any(co.material_grid(spec, "heat") > 0.0):
        out.append(Issue("warning", "physics", "no material generates heat, so nothing warms"))
    if spec.windows and len(spec.windows) == 2:
        holders = [(w.id, m) for w, (_wid, m) in zip(spec.windows, geo.window_masks(spec))
                   if (m & carrier).any()]
        if len(holders) != 1 or (carrier & ~holders[0][1]).any():
            out.append(Issue("error", "geometry",
                             "the coolant runs across the seam between the two windows; make "
                             "one window the channel and the other the block (Windows: one "
                             "piece per material)"))
    if any(i.severity == "error" for i in out) or any(
            i.severity == "error" for i in _check_boundaries(spec)):
        return out
    try:
        co.coolant_flow(spec)
    except FlowError as exc:
        out.append(Issue("error", "geometry", f"the coolant's flow: {exc}"))
    return out


def _check_attachments(spec: CaseSpec) -> list[Issue]:
    """The circuit: parts, nodes, and that it closes through the plate."""
    out: list[Issue] = []
    fam = _family(spec)
    reads = fam is not None and "attachments" in fam.layers
    if spec.attachments and not reads:
        return [Issue("warning", "geometry",
                      f"the {fam.id if fam else '?'} family does not read lumped "
                      f"attachments; the {len(spec.attachments)} here would be ignored")]
    if not reads:
        return out
    aids = [a.id for a in spec.attachments]
    for dup in sorted({i for i in aids if aids.count(i) > 1}):
        out.append(Issue("error", "geometry", f"two attachments are both called {dup!r}"))
    electrodes = [b.id for b in spec.boundaries if b.kind == "electrode"]
    if not electrodes:
        out.append(Issue("error", "geometry",
                         "the plate has no electrode: mark the edges where the circuit "
                         "attaches as electrodes (Boundaries)", fix="electrodes"))
    if not any(a.kind == "battery" for a in spec.attachments):
        out.append(Issue("error", "geometry",
                         "the circuit has no battery to drive it (Circuit)", fix="circuit"))
    for a in spec.attachments:
        if a.a == a.b:
            out.append(Issue("error", "geometry",
                             f"attachment {a.id} connects node {a.a!r} to itself"))
    # connectivity: every electrode is joined to the rest through the plate, and
    # every part must reach an electrode, or it carries no current
    nodes = set(electrodes) | {n for a in spec.attachments for n in (a.a, a.b)}
    parent = {n: n for n in nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for e in electrodes[1:]:
        parent[find(e)] = find(electrodes[0])
    for a in spec.attachments:
        parent[find(a.a)] = find(a.b)
    if electrodes:
        root = find(electrodes[0])
        lost = sorted({a.id for a in spec.attachments if find(a.a) != root})
        if lost:
            out.append(Issue("error", "geometry",
                             f"attachments {', '.join(lost)} are not connected to the "
                             f"plate's electrodes"))
        unused = [e for e in electrodes if not any(e in (a.a, a.b) for a in spec.attachments)]
        if unused and spec.attachments:
            out.append(Issue("warning", "geometry",
                             f"electrode {', '.join(unused)} has nothing wired to it and "
                             f"carries no current"))
    return out


def _family(spec: CaseSpec):
    try:
        return registry.family(spec.physics.family)
    except KeyError:
        return None


def _check_regions(spec: CaseSpec) -> list[Issue]:
    out: list[Issue] = []
    d = spec.domain
    fam = _family(spec)
    rids = [r.id for r in spec.regions]
    for dup in sorted({i for i in rids if rids.count(i) > 1}):
        out.append(Issue("error", "geometry", f"two regions are both called {dup!r}"))
    if spec.regions and fam is not None and "regions" not in fam.layers:
        out.append(Issue("warning", "geometry",
                         f"the {fam.id} family does not read material regions, so its "
                         f"solver would ignore the {len(spec.regions)} drawn here"))
    for r in spec.regions:
        # a drawn region may reach past the grid: it holds the cells whose centres
        # it contains (case file 0.4), like a drawn window
        if r.shape != "curve" and (r.x0 + r.nx > d.nx or r.y0 + r.ny > d.ny):
            out.append(Issue("error", "geometry", f"region {r.id} reaches past the domain"))
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    for k, r in enumerate(spec.regions):
        drawn = int(geo.region_mask(r, d.nx, d.ny).sum())
        kept = int((owner == k).sum())
        if drawn == 0:
            out.append(Issue("error", "geometry", f"region {r.id} contains no cell centre"))
        elif kept == 0:
            out.append(Issue("warning", "geometry",
                             f"region {r.id} is entirely covered by regions listed after it"))
        elif kept < drawn:
            out.append(Issue("info", "geometry",
                             f"regions listed after {r.id} take {drawn - kept} of its "
                             f"{drawn} cells (later regions stack on top)"))
    if fam is not None and "regions" in fam.layers:
        free = int(((owner < 0) & geo.domain_mask(d)).sum())
        if free:
            out.append(Issue("error", "geometry",
                             f"{free:,} cells of the domain have no material yet: every cell "
                             f"needs one (Materials)", fix="materials"))
    return out


def _check_boundaries(spec: CaseSpec) -> list[Issue]:
    out: list[Issue] = []
    d = spec.domain
    fam = _family(spec)
    bids = [b.id for b in spec.boundaries]
    for dup in sorted({i for i in bids if bids.count(i) > 1}):
        out.append(Issue("error", "geometry", f"two boundaries are both called {dup!r}"))
    known = {k.id for k in registry.BOUNDARY_KINDS}
    spans: dict[str, list[tuple[int, int, str]]] = {e: [] for e in geo.EDGES}
    drawn = geo.drawn_edge_names(d) if (d.outline is not None or d.holes) else []
    on_drawn: dict[str, list[str]] = {e: [] for e in drawn}
    for b in spec.boundaries:
        if b.kind not in known:
            out.append(Issue("error", "geometry", f"boundary {b.id}: unknown kind {b.kind!r}"))
        elif fam is not None and b.kind not in fam.boundary_kinds:
            out.append(Issue("error", "geometry",
                             f"boundary {b.id}: the {fam.id} family cannot impose "
                             f"{b.kind!r} (it can: {', '.join(fam.boundary_kinds) or 'none'})"))
        elif registry.boundary_kind(b.kind).needs_value and b.value is None:
            out.append(Issue("error", "geometry", f"boundary {b.id}: {b.kind} needs a value"))
        if b.drawn:
            if b.edge not in on_drawn:
                out.append(Issue("error", "geometry",
                                 f"boundary {b.id} is on {b.edge}, an edge the domain does "
                                 f"not have"))
            else:
                on_drawn[b.edge].append(b.id)
            continue
        if d.outline is not None:
            out.append(Issue("error", "geometry",
                             f"boundary {b.id} is on the grid's {b.edge} edge, but the domain "
                             f"has a drawn outline: its boundaries are on the outline's edges"))
            continue
        n = geo.edge_length(b.edge, d.nx, d.ny)
        stop = n if b.stop is None else b.stop
        if not (0 <= b.start < stop <= n):
            out.append(Issue("error", "geometry",
                             f"boundary {b.id} runs {b.start}..{stop} on the {b.edge} edge, "
                             f"which is {n} cells long"))
            continue
        spans[b.edge].append((b.start, stop, b.id))
    for edge, ids in on_drawn.items():
        if len(ids) > 1:
            out.append(Issue("error", "geometry",
                             f"boundaries {', '.join(ids)} are all on {edge}; a drawn edge "
                             f"takes one condition (add a vertex to split it)"))
        elif not ids and fam is not None and fam.boundary_kinds:
            out.append(Issue("error", "geometry", f"the drawn edge {edge} has no boundary "
                                                  f"condition"))
    for edge, segs in spans.items():
        if d.outline is not None:
            break                       # a drawn outline: the grid's edges are not used
        segs.sort()
        for (_s0, e0, i0), (s1, _e1, i1) in zip(segs, segs[1:]):
            if s1 < e0:
                out.append(Issue("error", "geometry",
                                 f"boundaries {i0} and {i1} overlap on the {edge} edge"))
        n = geo.edge_length(edge, d.nx, d.ny)
        covered = np.zeros(n, dtype=bool)
        for s0, e0, _ in segs:
            covered[s0:e0] = True
        gap = int((~covered).sum())
        if gap and fam is not None and fam.boundary_kinds:
            out.append(Issue("error", "geometry",
                             f"{gap} of the {n} cells on the {edge} edge have no boundary "
                             f"condition"))
    if fam is not None and fam.fixed_boundaries and d.outline is None:
        # the grid's own edges; a drawn outline replaces them (below)
        grid_b = [b for b in spec.boundaries if not b.drawn]
        want = sorted((f.edge, f.kind) for f in fam.fixed_boundaries)
        have = sorted((b.edge, b.kind) for b in grid_b
                      if b.start == 0 and b.stop in (None, geo.edge_length(b.edge, d.nx, d.ny)))
        if have != want or len(grid_b) != len(want):
            fixed = ", ".join(f"{f.edge} {f.kind}" for f in fam.fixed_boundaries)
            out.append(Issue("error", "geometry",
                             f"the {fam.id} solver fixes its outer boundary ({fixed}); a "
                             f"case of this family cannot change it (Model, Boundaries: "
                             f"use the family's)"))
    if fam is not None and fam.drawn_derived and drawn:
        for b in spec.boundaries:
            if not (b.drawn and b.edge in on_drawn):
                continue
            want_kind = derived_kind(spec, b.edge)
            if b.kind != want_kind:
                along = geo.along_grid_edge(d, b.edge)
                where = (f"it lies along the grid's {along} edge" if along else
                         "it is not along one of the grid's edges")
                out.append(Issue("error", "geometry",
                                 f"boundary {b.id} on {b.edge}: the {fam.id} solver fixes it "
                                 f"as {want_kind}, since {where}"))
    return out


def summary(issues: list[Issue]) -> dict[str, int]:
    return {s: sum(1 for i in issues if i.severity == s) for s in ("error", "warning", "info")}


# ---------------------------------------------------------------------------
# examples, taken from the real tilings rather than retyped
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """One entry of File > New from example."""

    key: str
    label: str
    description: str
    family: str
    style: str
    #: family-specific numbers the builder reads (for the wind farm: the tiling's
    #: columns and rows, the macro-steps and the threads)
    params: tuple[tuple[str, object], ...] = ()

    def param(self, name: str):
        return dict(self.params)[name]


#: **The wind-farm examples' step counts are the owner's decision (2026-09-28).**
#: W346 marched 40 macro-steps; with all three arms taking turns that is about
#: 8 minutes at 21 rotors and 4.5 at 12 on this laptop, against the showcase's
#: 2-minute rule.  The owner chose short marches (proposed as 14 and 8
#: macro-steps, "about 100 s each"), with all three arms over the whole run, so
#: the speed and the serial/threaded control cover the run and the farm-power
#: comparison covers the start-up only (the first wake needs about 17 steps to
#: reach the next row).  **Recalibrated the same day on measured runs in the
#: page**: 8 steps at 21 rotors took 122.1 s of wall time and 14 at 12 rotors
#: 112.6 s; 7 at 21 rotors then took 120.8 s, because the laptop's per-step cost
#: rose about 15% between back-to-back runs (the ratios did not move).  So the
#: counts are sized to the SLOWEST per-step cost measured that day, with room
#: for the compile: 6 steps at 21 rotors (16.8 s a step, about 104 s) and 12 at
#: 12 rotors.  The thread counts are the best measured on this laptop before
#: the choice (4, 4, 8).
EXAMPLES: dict[str, Example] = {e.key: e for e in (
    Example("wake-array-3", "Wake array: 3 rotors, 6 windows (CS-7, N = 6)",
            "Three actuator disks in an L on six overlapping windows; the case every "
            "wind-farm measurement started from.", "incompressible-2d", "A",
            (("cols", 3), ("rows", 2), ("steps", 40), ("threads", 4))),
    Example("farm-12", "Wind farm: 12 rotors, 24 windows (CS-7, N = 24)",
            "Twelve rotors on 24 windows, where the full-domain solve has left the "
            "cache and threads start to pay.", "incompressible-2d", "A",
            (("cols", 6), ("rows", 4), ("steps", 12), ("threads", 4))),
    Example("farm-21", "Wind farm: 21 rotors, 48 windows (W346)",
            "W346's 21-rotor farm: the decomposition run on threads against the full "
            "domain, live.", "incompressible-2d", "A",
            (("cols", 8), ("rows", 6), ("steps", 6), ("threads", 8))),
    Example("wall-2", "Two-layer wall: steel and copper (style C, steady)",
            "Heat through a steel layer and a copper layer, cut at the material "
            "interface and joined by Dirichlet-Neumann; checked against the closed form "
            "q = dT / sum L/k.", "conduction-2d", "C"),
    Example("plate-insert", "Copper insert in a steel plate (style B, transient)",
            "A steel plate with a copper insert warming from one side, on two overlapping "
            "windows iterated to agreement every time step.", "conduction-2d", "B"),
    Example("plate-circuit", "Resistive film on a battery and a resistor (style D)",
            "Current spreading between two electrodes of a graphite film wired to a "
            "battery and a resistor: the electrodes' currents are the circuit's.",
            "electric-2d", "D"),
    Example("plume-2", "Pollutant plume: a shallow reach into a deep one (style A)",
            "A release carried down a river that slows and mixes faster where it deepens, "
            "on two windows overlapping across the seam: one exchange per explicit step.",
            "transport-2d", "A"),
    Example("sound-air-water", "Sound from air into water (style C, explicit)",
            "A pressure pulse in air hits water: almost all of it reflects, the pressure "
            "that enters is doubled, and R is read against (Z2 - Z1) / (Z2 + Z1).",
            "acoustics-2d", "C"),
    Example("bracket-2", "Steel-and-aluminium bracket under load (style B)",
            "A plate clamped on one edge and loaded on the other, steel near the clamp and "
            "aluminium at the tip, on two overlapping windows iterated to agreement.",
            "elasticity-2d", "B"),
    Example("heated-strip", "Heated bimetal strip (split by physics)",
            "A steel-and-copper strip heated at one end bends as it warms: conduction and "
            "elasticity as two agents on one mesh, synchronous and lagged a step.",
            "thermoelastic-2d", "split"),
    Example("cooled-block", "Heated block cooled by a channel flow (style C, two physics)",
            "A chip on a copper block under a water channel: the coolant's advection and "
            "the block's conduction meet at the wall, and every watt leaves in the water.",
            "conjugate-heat-2d", "C"),
    Example("bend-3", "Heat round a pipe bend: three curved windows (style B, drawn)",
            "A quarter ring of steel drawn with arcs, hot at one end and cold at the other, "
            "cut into three curved windows that overlap along the bend and iterate to "
            "agreement.", "conduction-2d", "B"),
    Example("insert-round", "Round copper insert in a steel plate (style C, drawn)",
            "A copper disc in a steel plate, cut along the circle: the disc and the plate "
            "around it meet at a round interface, joined by Dirichlet-Neumann.",
            "conduction-2d", "C"),
    Example("s-channel", "Heat along an S-shaped channel: windows that follow it (style B)",
            "A steel channel drawn with splines, hot at one end and cold at the other. Its "
            "four windows are generated from its own shape, cut across its length at level "
            "curves of its harmonic coordinate, and follow it when it is reshaped.",
            "conduction-2d", "B"),
    # -- drawn shapes in every other family (2026-09-29, the owner: "make the smooth
    #    domain/spline/other stuff available for all cases")
    Example("ring-film", "Current round a curved film (style D, drawn)",
            "A quarter ring of graphite film with an electrode on each straight end, wired "
            "to a battery and a resistor: the current turns the corner, and the film's "
            "resistance is read against theta / (sigma t ln(ro / ri)).", "electric-2d", "D"),
    Example("river-bend", "A plume round a winding river (style A, drawn)",
            "A river drawn with splines, shallow then deep, its flow solved from its inlet to "
            "its outlet so no water crosses a bank; a release carried round the bends on two "
            "windows that follow the river.", "transport-2d", "A"),
    Example("sound-lens", "Sound on a curved water surface (style C, drawn)",
            "A pressure pulse runs along a round-ended air duct and meets a curved water "
            "surface; the pieces are the two media, and the rigid drawn walls keep the "
            "energy to round-off.", "acoustics-2d", "C"),
    Example("plate-hole", "A plate with a hole in tension (style B, drawn)",
            "A steel plate clamped at one end and pulled at the other, a round hole in its "
            "middle concentrating the stress round it; two windows cut along it.",
            "elasticity-2d", "B"),
    Example("bimetal-arc", "A curved bimetal strip heated at one end (split, drawn)",
            "A quarter ring of steel inside copper, one end held hot: conduction and "
            "elasticity on the ring's own cells, split by physics.", "thermoelastic-2d",
            "split"),
    Example("cooled-winding", "A block cooled by a winding channel (style C, drawn)",
            "A copper block with a chip at its base and a water channel drawn as a winding "
            "band across it; the coolant's flow is solved through the channel, and the pieces "
            "are its two physics.", "conjugate-heat-2d", "C"),
    Example("farm-hill", "Wind farm over a hill (style A, drawn)",
            "The three-rotor array over terrain drawn with splines: the ground is a no-slip "
            "wall held by penalization, the grid's edges keep their inlet, outlet and "
            "freestream.", "incompressible-2d", "A", (("steps", 40), ("threads", 4))),
)}


def example_case(key: str = "wake-array-3") -> CaseSpec:
    """A case built from `scaling_ladder`'s own tiling and rotor rule, or one of the
    other families' showcase cases.

    The wind-farm examples' windows, rotors and physical constants are read from
    the case modules the measurements were taken on, so an example cannot drift
    from the geometry the record describes.
    """
    ex = EXAMPLES[key]
    if key == "farm-hill":
        return _farm_hill_example(ex)
    if ex.family == "incompressible-2d":
        return _farm_example(ex)
    builder = {"wall-2": _wall_example, "plate-insert": _insert_example,
               "plate-circuit": _circuit_example, "plume-2": _plume_example,
               "sound-air-water": _sound_example, "bracket-2": _bracket_example,
               "heated-strip": _strip_example, "cooled-block": _block_example,
               "bend-3": _bend_example, "insert-round": _round_insert_example,
               "s-channel": _s_channel_example, "ring-film": _ring_film_example,
               "river-bend": _river_bend_example, "sound-lens": _sound_lens_example,
               "plate-hole": _plate_hole_example, "bimetal-arc": _bimetal_arc_example,
               "cooled-winding": _cooled_winding_example}.get(key)
    if builder is None:                                    # pragma: no cover
        raise KeyError(key)
    return builder(ex)


def _materials(fid: str, *names: str) -> dict[str, dict[str, float]]:
    lib = registry.family(fid).material_library()
    return {n: dict(lib[n]) for n in names}


def _wall_example(ex: Example) -> CaseSpec:
    """0.4 m x 0.1 m: 0.24 m of steel, then 0.16 m of copper, at 2.5 mm cells.
    Hot on the left, cold on the right, insulated top and bottom; cut at the
    interface, so the two pieces are the two materials."""
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=160, ny=40, dx=0.0025),
        physics=Physics(family="conduction-2d", params={"T0": 300.0}),
        materials=_materials("conduction-2d", "steel", "copper"),
        regions=[Region(id="R1", material="steel", x0=0, y0=0, nx=96, ny=40),
                 Region(id="R2", material="copper", x0=96, y0=0, nx=64, ny=40)],
        windows=[Window(id="steel", x0=0, y0=0, nx=96, ny=40),
                 Window(id="copper", x0=96, y0=0, nx=64, ny=40)],
        boundaries=family_boundaries("conduction-2d"),
        coupling=Coupling(style="C", tolerance=1e-10, max_iterations=200, relaxation=0.5,
                          aitken=True, dirichlet_side="auto"),
        run=RunSettings(mode="steady", steps=5, threads=1, macro_dt=1.0),
    )


def _insert_example(ex: Example) -> CaseSpec:
    """0.4 m x 0.24 m of steel with a 0.12 m x 0.08 m copper insert, warming from
    300 K with the left edge held at 400 K; two windows overlapping by 16 cells."""
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=160, ny=96, dx=0.0025),
        physics=Physics(family="conduction-2d", params={"T0": 300.0}),
        materials=_materials("conduction-2d", "steel", "copper"),
        regions=[Region(id="plate", material="steel", x0=0, y0=0, nx=160, ny=96),
                 Region(id="insert", material="copper", x0=56, y0=32, nx=48, ny=32)],
        windows=[Window(id=n, x0=b[0], y0=b[1], nx=b[2], ny=b[3])
                 for n, b in geo.tile(160, 96, 2, 1, 16)],
        boundaries=family_boundaries("conduction-2d"),
        coupling=Coupling(style="B", ramp_cells=8, tolerance=1e-10, max_iterations=500),
        run=RunSettings(mode="transient", macro_dt=120.0, steps=40, threads=2),
    )


def _bend_example(ex: Example) -> CaseSpec:
    """A quarter ring, radii 0.2 m and 0.5 m (40 and 100 cells of 5 mm), drawn with
    two arcs and two straight ends: the radial end along x held at 400 K, the one
    along y at 300 K, the arcs insulated.  Three windows, each a ring sector
    reaching past the ring (only the ring's cells count), overlapping their
    neighbours by 26 degrees -- 18 cells at the inner radius, over the 16 two
    ramps need -- and nothing else, so there is no cross-point."""
    import math
    c, deg = (4.0, 4.0), math.pi / 180.0

    def sector(r0: float, r1: float, a0: float, a1: float) -> Outline:
        return Outline.of(shapes.annulus_sector(c[0], c[1], r0, r1, a0 * deg, a1 * deg))
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=108, ny=108, dx=0.005, outline=sector(40.0, 100.0, 0.0, 90.0)),
        physics=Physics(family="conduction-2d", params={"T0": 300.0}),
        materials=_materials("conduction-2d", "steel"),
        regions=[Region(id="ring", material="steel", x0=0, y0=0, nx=108, ny=108)],
        windows=[Window(id="W1", shape="curve", outline=sector(34.0, 106.0, -5.0, 44.0)),
                 Window(id="W2", shape="curve", outline=sector(34.0, 106.0, 18.0, 72.0)),
                 Window(id="W3", shape="curve", outline=sector(34.0, 106.0, 46.0, 95.0))],
        boundaries=[Boundary(id="hot", edge="outline:0", kind="fixed-temperature",
                             value=400.0),
                    Boundary(id="outer", edge="outline:1", kind="insulated"),
                    Boundary(id="cold", edge="outline:2", kind="fixed-temperature",
                             value=300.0),
                    Boundary(id="inner", edge="outline:3", kind="insulated")],
        coupling=Coupling(style="B", ramp_cells=8, tolerance=1e-10, max_iterations=2000),
        run=RunSettings(mode="steady", steps=5, threads=2),
    )


def _round_insert_example(ex: Example) -> CaseSpec:
    """0.4 m x 0.25 m of steel with a copper disc 0.125 m across (a radius of 25
    cells of 2.5 mm), held at 400 K on the left and 300 K on the right.  The two
    pieces are the disc and the plate with the disc as its hole, so the interface
    is the circle's staircase of cell faces; the disc has no edge of its own that
    sets its temperature, so it is the Dirichlet side (`conduction.dirichlet_side`)."""
    disc = Outline.of(shapes.circle(80.0, 50.0, 25.0))
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=160, ny=100, dx=0.0025),
        physics=Physics(family="conduction-2d", params={"T0": 300.0}),
        materials=_materials("conduction-2d", "steel", "copper"),
        regions=[Region(id="plate", material="steel", x0=0, y0=0, nx=160, ny=100),
                 Region(id="insert", material="copper", shape="curve", outline=disc)],
        windows=[Window(id="plate", shape="curve",
                        outline=Outline.of(shapes.rectangle(0.0, 0.0, 160.0, 100.0)),
                        holes=[disc]),
                 Window(id="insert", shape="curve", outline=disc)],
        boundaries=family_boundaries("conduction-2d"),
        coupling=Coupling(style="C", tolerance=1e-10, max_iterations=2000, relaxation=0.5,
                          aitken=True),
        run=RunSettings(mode="steady", steps=5, threads=2),
    )


def _s_channel_example(ex: Example) -> CaseSpec:
    """A channel 28 cells wide (14 cm at 5 mm) whose middle follows
    ``y = 55 + 28 sin(2 pi (x - 10) / 180)`` from x = 10 to 190 cells: each wall is six
    splines through seven points, and the two ends are straight.  Steel, the left end
    held at 400 K and the right at 300 K, the walls insulated, steady.  The windows
    are not drawn: the layout cuts the channel into four along its length
    (`layout.py`), so they bend with it."""
    import math

    def mid(x: float) -> float:
        return 55.0 + 28.0 * math.sin(2.0 * math.pi * (x - 10.0) / 180.0)
    xs = [10.0, 40.0, 70.0, 100.0, 130.0, 160.0, 190.0]
    n = len(xs)
    pts = [(x, mid(x) - 14.0) for x in xs] + [(x, mid(x) + 14.0) for x in reversed(xs)]
    # along the bottom wall, up the right end, back along the top wall, down the left end
    kinds = ["spline"] * (n - 1) + ["line"] + ["spline"] * (n - 1) + ["line"]
    right, left = n - 1, 2 * n - 1
    bnds = [Boundary(id="hot", edge=f"outline:{left}", kind="fixed-temperature", value=400.0),
            Boundary(id="cold", edge=f"outline:{right}", kind="fixed-temperature", value=300.0)]
    bnds += [Boundary(id=f"wall{k}", edge=f"outline:{k}", kind="insulated")
             for k in range(2 * n) if k not in (right, left)]
    spec = CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=200, ny=110, dx=0.005,
                      outline=Outline(points=pts, edges=kinds, bulge=[0.0] * len(pts))),
        physics=Physics(family="conduction-2d", params={"T0": 300.0}),
        materials=_materials("conduction-2d", "steel"),
        regions=[Region(id="channel", material="steel", x0=0, y0=0, nx=200, ny=110)],
        boundaries=bnds, layout=Layout(cut="along", along=4),
        coupling=Coupling(style="B", ramp_cells=8, tolerance=1e-10, max_iterations=2000),
        run=RunSettings(mode="steady", steps=5, threads=2),
    )
    from .layout import refresh
    refresh(spec)                           # the windows, from the channel's own shape
    return spec


def _band(xs: list[float], mid, half: float) -> Outline:
    """A band of half-width ``half`` round the curve ``y = mid(x)`` from ``xs[0]`` to
    ``xs[-1]``: each side a spline through the points over ``xs``, the two ends
    straight."""
    n = len(xs)
    pts = [(x, mid(x) - half) for x in xs] + [(x, mid(x) + half) for x in reversed(xs)]
    kinds = ["spline"] * (n - 1) + ["line"] + ["spline"] * (n - 1) + ["line"]
    return Outline(points=pts, edges=kinds, bulge=[0.0] * len(pts))


def _ring_film_example(ex: Example) -> CaseSpec:
    """A quarter ring of graphite film, radii 50 mm and 125 mm (40 and 100 cells of
    1.25 mm), 30 um thick.  Electrode E1 is the straight end along x, E2 the one
    along y, the arcs carry no current; the same 12 V battery (0.5 ohm inside) and 1
    ohm resistor as the flat film close the loop.  The current runs round the ring,
    so the film's resistance is ``theta / (sigma t ln(ro / ri))``: 0.571 ohm."""
    import math
    ring = Outline.of(shapes.annulus_sector(4.0, 4.0, 40.0, 100.0, 0.0, math.pi / 2))
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=108, ny=108, dx=0.00125, outline=ring),
        physics=Physics(family="electric-2d", params={"thickness": 3.0e-5}),
        materials=_materials("electric-2d", "graphite"),
        regions=[Region(id="film", material="graphite", x0=0, y0=0, nx=108, ny=108)],
        windows=[Window(id="plate", shape="curve", outline=ring)],
        boundaries=[Boundary(id="E1", edge="outline:0", kind="electrode"),
                    Boundary(id="outer", edge="outline:1", kind="no-current"),
                    Boundary(id="E2", edge="outline:2", kind="electrode"),
                    Boundary(id="inner", edge="outline:3", kind="no-current")],
        attachments=[Attachment(id="B1", kind="battery", value=12.0, internal=0.5,
                                a="n1", b="E2"),
                     Attachment(id="R1", kind="resistor", value=1.0, a="n1", b="E1")],
        coupling=Coupling(style="D", tolerance=1e-10, max_iterations=200, relaxation=0.5,
                          aitken=True),
        run=RunSettings(mode="steady", steps=5, threads=1, macro_dt=1.0),
    )


def _river_bend_example(ex: Example) -> CaseSpec:
    """A river 1.2 km long and 200 m wide at 5 m cells, winding once each way:
    its middle follows ``y = 50 + 22 sin(2 pi x / 240)`` cells, each bank a spline,
    its two ends straight on the grid's left and right edges -- so it enters on the
    left and leaves on the right (`derived_kind`).  Shallow and fast (1.25 m) for the
    first 600 m, then deep and slow (2.5 m).  The outfall releases 10 g/s 150 m below
    the inlet, in the middle of the river.  The two windows are generated along the
    river (`layout.py`), so they bend with it.  Steps of 1.5 s, under the explicit
    limit the check computes for the solved flow (1.53 s: the water runs fastest on
    the inside of the bends); 2000 of them, 3000 s, longer than the water takes from
    the inlet to the outlet."""
    import math

    def mid(x: float) -> float:
        return 50.0 + 22.0 * math.sin(2.0 * math.pi * x / 240.0)
    river = _band([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0], mid, 20.0)
    spec = CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=240, ny=100, dx=5.0, outline=river),
        physics=Physics(family="transport-2d",
                        params={"q": 1.0, "release": 10.0, "source_x": 150.0,
                                "source_y": 5.0 * mid(30.0)}),
        materials=_materials("transport-2d", "shallow-fast", "deep-slow"),
        regions=[Region(id="upper", material="shallow-fast", x0=0, y0=0, nx=120, ny=100),
                 Region(id="lower", material="deep-slow", x0=120, y0=0, nx=120, ny=100)],
        layout=Layout(cut="along", along=2),
        coupling=Coupling(style="A", ramp_cells=8),
        run=RunSettings(mode="transient", macro_dt=1.5, steps=2000, threads=2),
    )
    spec.boundaries = default_boundaries(spec)
    from .layout import refresh
    refresh(spec)
    return spec


def _sound_lens_example(ex: Example) -> CaseSpec:
    """A duct of air 3 m long and 0.8 m tall with round ends, at 1 cm cells, drawn as
    two straight sides and two semicircles; water fills the grid right of a circle of
    radius 1.1 m, so the water's surface is a shallow curve across the duct.  A
    Gaussian pulse 0.1 m wide starts 1 m in; the pieces are the two media
    (`layout.py`, one per material).  Steps of 4 us, under the leapfrog's limit of
    about 4.8 us (water's sound speed, on the open faces)."""
    duct = Outline(points=[(60.0, 20.0), (240.0, 20.0), (240.0, 100.0), (60.0, 100.0)],
                   edges=["line", "arc", "line", "arc"], bulge=[0.0, 1.0, 0.0, 1.0])
    spec = CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=300, ny=120, dx=0.01, outline=duct),
        physics=Physics(family="acoustics-2d",
                        params={"amplitude": 1.0, "pulse_x": 1.0, "pulse_width": 0.1}),
        materials=_materials("acoustics-2d", "air", "water"),
        regions=[Region(id="air", material="air", x0=0, y0=0, nx=300, ny=120),
                 Region(id="water", material="water", shape="curve",
                        outline=Outline.of(shapes.circle(300.0, 60.0, 110.0)))],
        layout=Layout(cut="materials"),
        coupling=Coupling(style="C"),
        run=RunSettings(mode="transient", macro_dt=4.0e-6, steps=1500, threads=1),
    )
    spec.boundaries = default_boundaries(spec)
    from .layout import refresh
    refresh(spec)
    return spec


def _plate_hole_example(ex: Example) -> CaseSpec:
    """A steel plate 0.4 m x 0.2 m at 2.5 mm cells, 10 mm thick, with a round hole of
    radius 30 mm (12 cells) in its middle: clamped on the left, pulled on the right by
    10 MPa.  Far from the hole the stress is the pull; at the hole's top and bottom
    it rises to about three times that (Kirsch's plate, less for a finite width).
    Two windows generated along the plate, iterated to 1e-12 of the largest
    displacement, the tolerance the force check's registration assumes."""
    spec = CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=160, ny=80, dx=0.0025,
                      holes=[Outline.of(shapes.circle(80.0, 40.0, 12.0))]),
        physics=Physics(family="elasticity-2d", params={"thickness": 0.01}),
        materials=_materials("elasticity-2d", "steel"),
        regions=[Region(id="plate", material="steel", x0=0, y0=0, nx=160, ny=80)],
        layout=Layout(cut="along", along=2),
        coupling=Coupling(style="B", ramp_cells=8, tolerance=1e-12, max_iterations=5000),
        run=RunSettings(mode="steady", steps=5, threads=2, macro_dt=1.0),
    )
    spec.boundaries = [Boundary(id="clamp", edge="left", kind="clamped"),
                       Boundary(id="pull", edge="right", kind="load-x", value=1.0e7),
                       Boundary(id="bottom", edge="bottom", kind="free"),
                       Boundary(id="top", edge="top", kind="free")]
    spec.boundaries += [Boundary(id=f"hole-{k}", edge=f"hole0:{k}", kind="free")
                        for k in range(4)]
    from .layout import refresh
    refresh(spec)
    return spec


def _bimetal_arc_example(ex: Example) -> CaseSpec:
    """A quarter ring of radii 60 mm and 100 mm at 1 mm cells: steel inside, copper
    outside the 80 mm arc.  At 300 K, its end along x held at 400 K from the start,
    every other edge insulated; copper expands more than steel, so the arc opens as
    the heat runs round it.  One window, the ring (a split by physics); 60 steps of
    5 s."""
    import math
    ring = Outline.of(shapes.annulus_sector(4.0, 4.0, 60.0, 100.0, 0.0, math.pi / 2))
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=108, ny=108, dx=0.001, outline=ring),
        physics=Physics(family="thermoelastic-2d", params={"T0": 300.0}),
        materials=_materials("thermoelastic-2d", "steel", "copper"),
        regions=[Region(id="steel", material="steel", x0=0, y0=0, nx=108, ny=108),
                 Region(id="copper", material="copper", shape="curve",
                        outline=Outline.of(shapes.annulus_sector(4.0, 4.0, 80.0, 110.0,
                                                                 -0.1, 1.7)))],
        windows=[Window(id="plate", shape="curve", outline=ring)],
        boundaries=[Boundary(id="hot", edge="outline:0", kind="fixed-temperature",
                             value=400.0),
                    Boundary(id="outer", edge="outline:1", kind="insulated"),
                    Boundary(id="end", edge="outline:2", kind="insulated"),
                    Boundary(id="inner", edge="outline:3", kind="insulated")],
        coupling=Coupling(style="split"),
        run=RunSettings(mode="transient", macro_dt=5.0, steps=60, threads=2),
    )


def _cooled_winding_example(ex: Example) -> CaseSpec:
    """The cooled block's copper and chip (0.2 m x 60 mm at 1 mm cells, the chip
    40 mm x 5 mm in the middle of the base generating 10 W/cm^3), the block drawn
    with rounded corners (radius 6 mm), and the water drawn as a winding channel
    10 mm wide whose middle follows ``y = 40 + 8 sin(pi x / 100)`` cells.  The
    block's left and right edges each have a vertex at either side of the channel,
    so the stretch between them -- read off the channel's cells -- takes the
    coolant's inlet (1 cm/s from 300 K) or outlet; every other edge is insulated.
    The coolant's flow is solved through the channel (`flow.py`); the pieces are
    the block and the channel (`layout.py`, cut by physics)."""
    import math

    def mid(x: float) -> float:
        return 40.0 + 8.0 * math.sin(math.pi * x / 100.0)
    channel = _band([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0], mid, 5.0)
    water = Region(id="channel", material="water", shape="curve", outline=channel)
    wet = geo.region_mask(water, 200, 60)
    rows_l, rows_r = np.flatnonzero(wet[:, 0]), np.flatnonzero(wet[:, -1])
    lo_l, hi_l = float(rows_l.min()), float(rows_l.max() + 1)
    lo_r, hi_r = float(rows_r.min()), float(rows_r.max() + 1)
    q = math.tan(math.pi / 8)                     # a quarter circle's bulge
    r_ = 6.0
    pts = [(r_, 0.0), (200.0 - r_, 0.0), (200.0, r_), (200.0, lo_r), (200.0, hi_r),
           (200.0, 60.0 - r_), (200.0 - r_, 60.0), (r_, 60.0), (0.0, 60.0 - r_), (0.0, hi_l),
           (0.0, lo_l), (0.0, r_)]
    kinds = ["line", "arc", "line", "line", "line", "arc", "line", "arc", "line", "line",
             "line", "arc"]
    bulge = [q if k == "arc" else 0.0 for k in kinds]
    block = Outline(points=pts, edges=kinds, bulge=bulge)
    spec = CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=200, ny=60, dx=0.001, outline=block),
        physics=Physics(family="conjugate-heat-2d", params={"u_coolant": 0.01}),
        materials=_materials("conjugate-heat-2d", "copper", "chip", "water"),
        regions=[Region(id="block", material="copper", x0=0, y0=0, nx=200, ny=60),
                 Region(id="chip", material="chip", x0=80, y0=0, nx=40, ny=5), water],
        layout=Layout(cut="materials"),
        coupling=Coupling(style="C", tolerance=1e-10, max_iterations=2000, relaxation=0.5,
                          aitken=True, dirichlet_side="auto"),
        run=RunSettings(mode="steady", steps=5, threads=1, macro_dt=1.0),
    )
    spec.boundaries = [Boundary(id=f"wall{k}", edge=f"outline:{k}", kind="insulated")
                       for k in range(len(pts))]
    spec.boundaries[3] = Boundary(id="outlet", edge="outline:3", kind="coolant-outlet")
    spec.boundaries[9] = Boundary(id="inlet", edge="outline:9", kind="coolant-inlet",
                                  value=300.0)
    from .layout import refresh
    refresh(spec)
    return spec


def _farm_hill_example(ex: Example) -> CaseSpec:
    """The three-rotor array (`wake-array-3`: its grid, rotors, six windows and
    constants) over a hill: the domain's bottom is terrain drawn as splines through
    nine points, ``y = 12 + 30 exp(-((x - 194) / 60)^2)`` cells, rising 30 cells (0.94
    D) under the middle of the array; its other three edges are the grid's.  The
    terrain is a no-slip wall (the solid below it is held at rest every sub-step);
    the grid's edges keep the solver's inlet, outlet and freestream
    (`derived_kind`).  40 macro-steps, W346's march."""
    import math
    spec = _farm_example(EXAMPLES["wake-array-3"])
    spec.name, spec.description = ex.key, ex.label
    d = spec.domain
    xs = [d.nx * k / 8.0 for k in range(9)]
    pts = [(x, 12.0 + 30.0 * math.exp(-((x - 0.55 * d.nx) / 60.0) ** 2)) for x in xs]
    pts += [(float(d.nx), float(d.ny)), (0.0, float(d.ny))]
    kinds = ["spline"] * 8 + ["line"] * 3
    d.outline = Outline(points=pts, edges=kinds, bulge=[0.0] * len(pts))
    spec.boundaries = default_boundaries(spec)
    spec.run.steps = ex.param("steps")
    spec.run.threads = ex.param("threads")
    return spec


def _circuit_example(ex: Example) -> CaseSpec:
    """A 0.2 m x 0.1 m graphite film, 30 um thick, at 1.25 mm cells.  Electrode E1
    is the middle of the left edge, E2 the upper part of the right edge, so the
    current spreads across the film diagonally.  A 12 V battery with 0.5 ohm
    inside and a 1 ohm resistor close the loop: n1 is the battery's + terminal,
    the resistor runs from n1 to E1, and the battery's - terminal is E2.  The film
    is thin enough that its resistance is comparable to the circuit's, so the
    plate matters and the style-D iteration needs its relaxation."""
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=160, ny=80, dx=0.00125),
        physics=Physics(family="electric-2d", params={"thickness": 3.0e-5}),
        materials=_materials("electric-2d", "graphite"),
        regions=[Region(id="film", material="graphite", x0=0, y0=0, nx=160, ny=80)],
        windows=[Window(id="plate", x0=0, y0=0, nx=160, ny=80)],
        boundaries=[Boundary(id="left-low", edge="left", kind="no-current", start=0, stop=30),
                    Boundary(id="E1", edge="left", kind="electrode", start=30, stop=50),
                    Boundary(id="left-high", edge="left", kind="no-current", start=50),
                    Boundary(id="right-low", edge="right", kind="no-current", start=0,
                             stop=50),
                    Boundary(id="E2", edge="right", kind="electrode", start=50),
                    Boundary(id="bottom", edge="bottom", kind="no-current"),
                    Boundary(id="top", edge="top", kind="no-current")],
        attachments=[Attachment(id="B1", kind="battery", value=12.0, internal=0.5,
                                a="n1", b="E2"),
                     Attachment(id="R1", kind="resistor", value=1.0, a="n1", b="E1")],
        coupling=Coupling(style="D", tolerance=1e-10, max_iterations=200, relaxation=0.5,
                          aitken=True),
        run=RunSettings(mode="steady", steps=5, threads=1, macro_dt=1.0),
    )


def _plume_example(ex: Example) -> CaseSpec:
    """A river 2.4 km long and 300 m wide at 5 m cells, carrying 1 m^2/s per metre of
    width: a shallow reach (1.25 m, so 0.8 m/s) for the first 1.2 km, then a deep one
    (2.5 m, so 0.4 m/s) that mixes four times as fast.  The outfall releases 10 g/s
    200 m below the inlet, a third of the way across.  Two windows overlap by 48 cells
    across the seam between the reaches.  6000 steps of 2 s: the water takes 4250 s
    from the outfall to the outlet, so the run ends with the plume steady and the
    outflow equal to the release (measured: to 8e-14, in 7 s of wall time
    outside the page)."""
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=480, ny=60, dx=5.0),
        physics=Physics(family="transport-2d",
                        params={"q": 1.0, "release": 10.0, "source_x": 200.0,
                                "source_y": 100.0}),
        materials=_materials("transport-2d", "shallow-fast", "deep-slow"),
        regions=[Region(id="upper", material="shallow-fast", x0=0, y0=0, nx=240, ny=60),
                 Region(id="lower", material="deep-slow", x0=240, y0=0, nx=240, ny=60)],
        windows=[Window(id="upper", x0=0, y0=0, nx=264, ny=60),
                 Window(id="lower", x0=216, y0=0, nx=264, ny=60)],
        boundaries=family_boundaries("transport-2d"),
        coupling=Coupling(style="A", ramp_cells=8),
        run=RunSettings(mode="transient", macro_dt=2.0, steps=6000, threads=2),
    )


def _sound_example(ex: Example) -> CaseSpec:
    """1.4 m of air, then 6 m of water, 0.4 m tall, at 1 cm cells.  A Gaussian pulse
    0.1 m wide starts in the middle of the air, seven widths clear of the wall and
    of the water.  The water is long enough that what enters it cannot come back
    before the air holds the reflected pulse alone (steps 1021 to 2027 at 4 us);
    2000 steps of 4 us, against a stability limit of about 4.8 us."""
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=740, ny=40, dx=0.01),
        physics=Physics(family="acoustics-2d",
                        params={"amplitude": 1.0, "pulse_x": 0.7, "pulse_width": 0.1}),
        materials=_materials("acoustics-2d", "air", "water"),
        regions=[Region(id="air", material="air", x0=0, y0=0, nx=140, ny=40),
                 Region(id="water", material="water", x0=140, y0=0, nx=600, ny=40)],
        windows=[Window(id="air", x0=0, y0=0, nx=140, ny=40),
                 Window(id="water", x0=140, y0=0, nx=600, ny=40)],
        boundaries=family_boundaries("acoustics-2d"),
        coupling=Coupling(style="C"),
        run=RunSettings(mode="transient", macro_dt=4.0e-6, steps=2000, threads=1),
    )


def _bracket_example(ex: Example) -> CaseSpec:
    """0.4 m x 0.1 m at 2.5 mm cells, 10 mm thick: 0.24 m of steel from the clamped
    left edge, then 0.16 m of aluminium to the right edge, which carries a downward
    shear traction of 5 MPa (5 kN on the plate).  Two windows overlap by 32 cells;
    Schwarz stops at an update of 1e-12 of the largest displacement, the tolerance
    the force check's registration assumes (elasticity.CHECKS)."""
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=160, ny=40, dx=0.0025),
        physics=Physics(family="elasticity-2d", params={"thickness": 0.01}),
        materials=_materials("elasticity-2d", "steel", "aluminium"),
        regions=[Region(id="root", material="steel", x0=0, y0=0, nx=96, ny=40),
                 Region(id="tip", material="aluminium", x0=96, y0=0, nx=64, ny=40)],
        windows=[Window(id="clamp-side", x0=0, y0=0, nx=96, ny=40),
                 Window(id="tip-side", x0=64, y0=0, nx=96, ny=40)],
        boundaries=family_boundaries("elasticity-2d"),
        coupling=Coupling(style="B", ramp_cells=8, tolerance=1e-12, max_iterations=5000),
        run=RunSettings(mode="steady", steps=5, threads=2, macro_dt=1.0),
    )


def _strip_example(ex: Example) -> CaseSpec:
    """A bimetal strip 0.2 m x 20 mm at 1 mm cells: 10 mm of steel under 10 mm of
    copper, at 300 K, its left end held at 400 K from the start and every other edge
    insulated.  Copper expands more than steel, so the strip bends as the heat runs
    along it.  60 steps of 5 s."""
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=200, ny=20, dx=0.001),
        physics=Physics(family="thermoelastic-2d", params={"T0": 300.0}),
        materials=_materials("thermoelastic-2d", "steel", "copper"),
        regions=[Region(id="steel", material="steel", x0=0, y0=0, nx=200, ny=10),
                 Region(id="copper", material="copper", x0=0, y0=10, nx=200, ny=10)],
        windows=[Window(id="plate", x0=0, y0=0, nx=200, ny=20)],
        boundaries=family_boundaries("thermoelastic-2d"),
        coupling=Coupling(style="split"),
        run=RunSettings(mode="transient", macro_dt=5.0, steps=60, threads=2),
    )


def _block_example(ex: Example) -> CaseSpec:
    """A copper block 0.2 m x 50 mm at 1 mm cells, a 40 mm x 5 mm chip in the middle of
    its base generating 10 W/cm^3 (2 kW per metre of depth), and a 10 mm water channel
    along its top at 1 cm/s entering at 300 K.  The water's mdot cp is 418 W/K per
    metre, so its bulk temperature rises 4.8 K.  Every other edge is insulated, so
    all the heat leaves in the water.

    The block floats (nothing of its own sets its temperature level), so it takes
    the Dirichlet side, and copper against water puts the 1-D factor at 132: Aitken
    converged in 330 iterations when measured.  The first run of this example
    allowed 200 and failed its reference check (unconverged; the block was then also
    the Neumann side, since fixed) -- so it allows 2000, and the check still fails
    any run that stops unconverged."""
    return CaseSpec(
        name=ex.key, description=ex.label,
        domain=Domain(nx=200, ny=60, dx=0.001),
        physics=Physics(family="conjugate-heat-2d", params={"u_coolant": 0.01}),
        materials=_materials("conjugate-heat-2d", "copper", "chip", "water"),
        regions=[Region(id="block", material="copper", x0=0, y0=0, nx=200, ny=50),
                 Region(id="chip", material="chip", x0=80, y0=0, nx=40, ny=5),
                 Region(id="channel", material="water", x0=0, y0=50, nx=200, ny=10)],
        windows=[Window(id="block", x0=0, y0=0, nx=200, ny=50),
                 Window(id="channel", x0=0, y0=50, nx=200, ny=10)],
        boundaries=[Boundary(id="left-block", edge="left", kind="insulated", start=0, stop=50),
                    Boundary(id="inlet", edge="left", kind="coolant-inlet", start=50,
                             value=300.0),
                    Boundary(id="right-block", edge="right", kind="insulated", start=0,
                             stop=50),
                    Boundary(id="outlet", edge="right", kind="coolant-outlet", start=50),
                    Boundary(id="base", edge="bottom", kind="insulated"),
                    Boundary(id="lid", edge="top", kind="insulated")],
        coupling=Coupling(style="C", tolerance=1e-10, max_iterations=2000, relaxation=0.5,
                          aitken=True, dirichlet_side="auto"),
        run=RunSettings(mode="steady", steps=5, threads=1, macro_dt=1.0),
    )


def _farm_example(ex: Example) -> CaseSpec:
    from atlas.cases import scaling_ladder as sl          # light: no torch, no build repo
    from atlas.cases import wake_array as wa

    t = sl.rung(ex.param("cols"), ex.param("rows")).tiling
    return CaseSpec(
        name=ex.key,
        description=ex.label,
        domain=Domain(nx=t.nx, ny=t.ny, dx=wa.DX),
        physics=Physics(family="incompressible-2d",
                        params={"nu": wa.NU_REF, "u_inf": wa.U_INF}),
        windows=[Window(id=n, x0=ox, y0=oy, nx=wa.N, ny=wa.N)
                 for n, (ox, oy) in zip(t.names, t.offsets)],
        devices=[Device(id=r.rotor_id, x=r.x_plane, y=r.y_centre, diameter=1.0)
                 for r in t.rotors],
        boundaries=family_boundaries("incompressible-2d"),
        coupling=Coupling(ramp_cells=wa.RAMP, assembly="projected", elliptic="exposed"),
        run=RunSettings(macro_dt=wa.MACRO_DT, steps=ex.param("steps"),
                        threads=ex.param("threads")),
    )


def blank_case(nx: int = 352, ny: int = 240) -> CaseSpec:
    """An empty domain in the wind-farm family's units, with no windows yet."""
    from atlas.cases import wake_array as wa
    return CaseSpec(name="untitled", description="",
                    domain=Domain(nx=nx, ny=ny, dx=wa.DX),
                    physics=Physics(family="incompressible-2d",
                                    params={"nu": wa.NU_REF, "u_inf": wa.U_INF}),
                    boundaries=family_boundaries("incompressible-2d"))


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    return s or "untitled"


__all__ = ["SCHEMA_ID", "READABLE", "CaseSpec", "Domain", "Physics", "Region", "Window",
           "Device", "Boundary", "Attachment", "Coupling", "RunSettings", "Compare",
           "Issue", "check", "summary", "family_boundaries", "adapt_to_family", "Example",
           "EXAMPLES", "example_case", "blank_case", "slug"]
