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

Schema ``atlas-workbench/case@0.2`` adds the geometry section's two new layers:
material **regions** (rectangles drawn in the page, or polygons imported from
Gmsh) and **boundaries** (conditions on segments of the domain's edges).  A 0.1
file still loads: it gains the boundaries its family's solver fixes, and no
regions.  The geometry rules themselves live in `geometry.py`.
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

SCHEMA_ID = "atlas-workbench/case@0.2"
#: every schema this version reads; older ones are migrated on load
READABLE = ("atlas-workbench/case@0.1", SCHEMA_ID)


class Domain(BaseModel):
    nx: int = Field(gt=0, description="cells along x")
    ny: int = Field(gt=0, description="cells along y")
    dx: float = Field(gt=0, description="cell size in the physics' length unit")


class Physics(BaseModel):
    family: str = "incompressible-2d"
    nu: float = Field(gt=0, description="kinematic viscosity")
    u_inf: float = Field(gt=0, description="freestream speed")


class Window(BaseModel):
    id: str
    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    nx: int = Field(gt=0)
    ny: int = Field(gt=0)


class Region(BaseModel):
    """A piece of the domain made of one material.

    A rectangle (``x0, y0, nx, ny`` in cells), or a polygon (``points`` in cells,
    with optional ``holes``, usually imported from Gmsh), in which case the
    rectangle is its bounding box and is derived, not edited.

    **Regions stack in list order**: where two overlap, the later one takes the
    cells.  So a plate with an insert is the plate, then the insert, with no need
    to cut the plate around it.  A cell's centre decides which region it is in.
    """
    id: str
    material: str = "material-1"
    shape: Literal["rect", "polygon"] = "rect"
    x0: int = Field(0, ge=0)
    y0: int = Field(0, ge=0)
    nx: int = Field(1, gt=0)
    ny: int = Field(1, gt=0)
    points: Optional[list[tuple[float, float]]] = None
    holes: Optional[list[list[tuple[float, float]]]] = None

    @model_validator(mode="after")
    def _shape_matches(self):
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
    """
    id: str
    edge: Literal["left", "right", "bottom", "top"]
    kind: str
    start: int = Field(0, ge=0)
    stop: Optional[int] = Field(None, gt=0)
    value: Optional[float] = None


class Device(BaseModel):
    id: str
    kind: Literal["actuator-disk"] = "actuator-disk"
    x: float = Field(description="disk plane, in the physics' length unit")
    y: float = Field(description="disk centre, in the physics' length unit")
    diameter: float = Field(1.0, gt=0)
    yaw_deg: float = 0.0


class Coupling(BaseModel):
    ramp_cells: int = Field(8, ge=1, description="partition-of-unity ramp width")
    assembly: Literal["projected", "blend"] = "projected"
    elliptic: Literal["exposed", "embedded"] = "exposed"


class RunSettings(BaseModel):
    macro_dt: float = Field(0.2, gt=0)
    steps: int = Field(40, gt=0)
    threads: int = Field(4, ge=1)
    start: Literal["freestream"] = "freestream"


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
    regions: list[Region] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    devices: list[Device] = Field(default_factory=list)
    boundaries: list[Boundary] = Field(default_factory=list)
    coupling: Coupling = Field(default_factory=Coupling)
    run: RunSettings = Field(default_factory=RunSettings)
    compare: Compare = Field(default_factory=Compare)

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
        """0.1 -> 0.2: the family's fixed boundaries, no regions."""
        if isinstance(data, dict) and data.get("schema_id") == "atlas-workbench/case@0.1":
            data = dict(data)
            fam = (data.get("physics") or {}).get("family", "incompressible-2d")
            data.setdefault("boundaries", [b.model_dump() for b in family_boundaries(fam)])
            data.setdefault("regions", [])
            data["schema_id"] = SCHEMA_ID
        return data

    # -- persistence ---------------------------------------------------------
    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "CaseSpec":
        return cls.model_validate_json(text)

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


def family_boundaries(fid: str) -> list[Boundary]:
    """The boundaries a family's solver fixes, as whole-edge segments."""
    try:
        fam = registry.family(fid)
    except KeyError:
        return []
    return [Boundary(id=f"B-{b.edge}", edge=b.edge, kind=b.kind)
            for b in fam.fixed_boundaries]


# ---------------------------------------------------------------------------
# checking a case
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    severity: Literal["error", "warning", "info"]
    step: str          # which workflow step fixes it: "case", "geometry", ...
    message: str


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

    if not spec.windows:
        out.append(Issue("error", "geometry", "the domain is not cut into any windows yet"))
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
                             f"{gap} of {d.nx * d.ny} cells lie in no window"))

    ramp = spec.coupling.ramp_cells
    if spec.windows:
        an = geo.analyse_windows(d.nx, d.ny, [(w.id, (w.x0, w.y0, w.nx, w.ny))
                                              for w in spec.windows], ramp)
        n_ramp = int(an.ramp_only.sum())
        if n_ramp:
            names = ", ".join(sorted(an.ramp_only_in)[:8])
            more = "" if len(an.ramp_only_in) <= 8 else f" and {len(an.ramp_only_in) - 8} more"
            out.append(Issue("error", "geometry",
                             f"{n_ramp} cells have no window at full weight (in {names}{more}): "
                             f"where windows meet they must overlap by at least twice the "
                             f"ramp ({2 * ramp} cells), as every measured tiling does"))
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
    if spec.physics.family == "incompressible-2d":
        re_cell = d.dx * spec.physics.u_inf / spec.physics.nu
        if re_cell > 8.0:
            out.append(Issue("warning", "physics",
                             f"the cell Reynolds number dx U / nu is {re_cell:.3g}; the "
                             f"window solver's own validity predicate asks for at most 8 "
                             f"(wake_array.FluidWindow.reference_validity)"))

    out += _check_regions(spec)
    out += _check_boundaries(spec)

    if spec.physics.family != "incompressible-2d":
        out.append(Issue("error", "physics",
                         f"physics family {spec.physics.family!r} is not available yet"))
    if spec.coupling.assembly == "blend":
        out.append(Issue("warning", "physics",
                         "a blend without the global projection leaves the band at "
                         "macro-step 70-80 at six windows (W100)"))
    if spec.coupling.elliptic == "embedded":
        out.append(Issue("warning", "physics",
                         "an embedded pressure solve in every window is refused by "
                         "L2/R10 and is unstable composed (W100)"))
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
        if r.x0 + r.nx > d.nx or r.y0 + r.ny > d.ny:
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
        free = int((owner < 0).sum())
        if free:
            out.append(Issue("error", "geometry",
                             f"{free} cells belong to no region; every cell needs a material"))
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
    for b in spec.boundaries:
        n = geo.edge_length(b.edge, d.nx, d.ny)
        stop = n if b.stop is None else b.stop
        if b.kind not in known:
            out.append(Issue("error", "geometry", f"boundary {b.id}: unknown kind {b.kind!r}"))
        elif fam is not None and b.kind not in fam.boundary_kinds:
            out.append(Issue("error", "geometry",
                             f"boundary {b.id}: the {fam.id} family cannot impose "
                             f"{b.kind!r} (it can: {', '.join(fam.boundary_kinds) or 'none'})"))
        elif registry.boundary_kind(b.kind).needs_value and b.value is None:
            out.append(Issue("error", "geometry", f"boundary {b.id}: {b.kind} needs a value"))
        if not (0 <= b.start < stop <= n):
            out.append(Issue("error", "geometry",
                             f"boundary {b.id} runs {b.start}..{stop} on the {b.edge} edge, "
                             f"which is {n} cells long"))
            continue
        spans[b.edge].append((b.start, stop, b.id))
    for edge, segs in spans.items():
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
    if fam is not None and fam.fixed_boundaries:
        want = sorted((f.edge, f.kind) for f in fam.fixed_boundaries)
        have = sorted((b.edge, b.kind) for b in spec.boundaries
                      if b.start == 0 and b.stop in (None, geo.edge_length(b.edge, d.nx, d.ny)))
        if have != want or len(spec.boundaries) != len(want):
            fixed = ", ".join(f"{f.edge} {f.kind}" for f in fam.fixed_boundaries)
            out.append(Issue("error", "geometry",
                             f"the {fam.id} solver fixes its outer boundary ({fixed}); a "
                             f"case of this family cannot change it (Geometry, Boundaries: "
                             f"use the family's)"))
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
)}


def example_case(key: str = "wake-array-3") -> CaseSpec:
    """A case built from `scaling_ladder`'s own tiling and rotor rule.

    The windows, the rotors and every physical constant are read from the case
    modules the measurements were taken on, so an example cannot drift from the
    geometry the record describes.
    """
    ex = EXAMPLES[key]
    if ex.family == "incompressible-2d":
        return _farm_example(ex)
    raise KeyError(key)                                    # pragma: no cover


def _farm_example(ex: Example) -> CaseSpec:
    from atlas.cases import scaling_ladder as sl          # light: no torch, no build repo
    from atlas.cases import wake_array as wa

    t = sl.rung(ex.param("cols"), ex.param("rows")).tiling
    return CaseSpec(
        name=ex.key,
        description=ex.label,
        domain=Domain(nx=t.nx, ny=t.ny, dx=wa.DX),
        physics=Physics(family="incompressible-2d", nu=wa.NU_REF, u_inf=wa.U_INF),
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
                    physics=Physics(family="incompressible-2d", nu=wa.NU_REF, u_inf=wa.U_INF),
                    boundaries=family_boundaries("incompressible-2d"))


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    return s or "untitled"


__all__ = ["SCHEMA_ID", "READABLE", "CaseSpec", "Domain", "Physics", "Region", "Window",
           "Device", "Boundary", "Coupling", "RunSettings", "Compare", "Issue", "check",
           "summary", "family_boundaries", "Example", "EXAMPLES", "example_case",
           "blank_case", "slug"]
