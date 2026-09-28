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

This is the shell's version of the schema (``atlas-workbench/case@0.1``).  The
geometry rules beyond "inside the domain, covering it, uniquely named" are left
for the geometry section, which is being designed with the owner.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

SCHEMA_ID = "atlas-workbench/case@0.1"


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
    windows: list[Window] = Field(default_factory=list)
    devices: list[Device] = Field(default_factory=list)
    coupling: Coupling = Field(default_factory=Coupling)
    run: RunSettings = Field(default_factory=RunSettings)
    compare: Compare = Field(default_factory=Compare)

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

    Structural only, for the shell: names unique, windows inside the domain and
    covering it, devices inside the domain.  The coupling rules a window layout
    must satisfy (overlap against ramp, cross-points) belong to the geometry
    section and are not decided here.
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

    wx, wy = d.nx * d.dx, d.ny * d.dx
    for v in spec.devices:
        if not (0.0 <= v.x <= wx and v.diameter / 2 <= v.y <= wy - v.diameter / 2):
            out.append(Issue("error", "geometry",
                             f"device {v.id} is not fully inside the domain"))

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


def summary(issues: list[Issue]) -> dict[str, int]:
    return {s: sum(1 for i in issues if i.severity == s) for s in ("error", "warning", "info")}


# ---------------------------------------------------------------------------
# examples, taken from the real tilings rather than retyped
# ---------------------------------------------------------------------------

#: key -> (label, window columns, window rows)
EXAMPLES = {
    "wake-array-3": ("Wake array: 3 rotors, 6 windows (CS-7, N = 6)", 3, 2),
    "farm-12": ("Wind farm: 12 rotors, 24 windows (CS-7, N = 24)", 6, 4),
    "farm-21": ("Wind farm: 21 rotors, 48 windows (W346)", 8, 6),
}


def example_case(key: str = "wake-array-3") -> CaseSpec:
    """A case built from `scaling_ladder`'s own tiling and rotor rule.

    The windows, the rotors and every physical constant are read from the case
    modules the measurements were taken on, so an example cannot drift from the
    geometry the record describes.
    """
    from atlas.cases import scaling_ladder as sl          # light: no torch, no build repo
    from atlas.cases import wake_array as wa

    label, cols, rows = EXAMPLES[key]
    t = sl.rung(cols, rows).tiling
    return CaseSpec(
        name=key,
        description=label,
        domain=Domain(nx=t.nx, ny=t.ny, dx=wa.DX),
        physics=Physics(family="incompressible-2d", nu=wa.NU_REF, u_inf=wa.U_INF),
        windows=[Window(id=n, x0=ox, y0=oy, nx=wa.N, ny=wa.N)
                 for n, (ox, oy) in zip(t.names, t.offsets)],
        devices=[Device(id=r.rotor_id, x=r.x_plane, y=r.y_centre, diameter=1.0)
                 for r in t.rotors],
        coupling=Coupling(ramp_cells=wa.RAMP, assembly="projected", elliptic="exposed"),
        run=RunSettings(macro_dt=wa.MACRO_DT, steps=40, threads=4),
    )


def blank_case(nx: int = 352, ny: int = 240) -> CaseSpec:
    """An empty domain in the wind-farm family's units, with no windows yet."""
    from atlas.cases import wake_array as wa
    return CaseSpec(name="untitled", description="",
                    domain=Domain(nx=nx, ny=ny, dx=wa.DX),
                    physics=Physics(family="incompressible-2d", nu=wa.NU_REF, u_inf=wa.U_INF))


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    return s or "untitled"


__all__ = ["SCHEMA_ID", "CaseSpec", "Domain", "Physics", "Window", "Device", "Coupling",
           "RunSettings", "Compare", "Issue", "check", "summary", "EXAMPLES",
           "example_case", "blank_case", "slug"]
