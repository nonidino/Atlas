"""Which physics families the workbench knows, and how far each is wired.

Step 1 of [[outcome-c4-path-to-declarative-cases]]: Python written once per
solver, never per case.  In the shell this is a catalogue, not yet a factory: it
names each family, the solvers and devices it would use (all of which exist in
`atlas/cases/`), and **what is missing before the workbench can run it**, so that
the GUI can show an unavailable option with its reason instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundaryKind:
    id: str
    physics: str                # "flow" | "heat" | "electric"
    meaning: str
    needs_value: bool = False
    unit: str = ""


#: The boundary conditions a case may name.  A family says which of them its
#: solver can do; the check refuses the rest.
BOUNDARY_KINDS: tuple[BoundaryKind, ...] = (
    BoundaryKind("inlet", "flow", "inflow held at the freestream (U, 0)"),
    BoundaryKind("freestream", "flow", "lateral edge held at the freestream (U, 0)"),
    BoundaryKind("outlet", "flow", "outflow"),
    BoundaryKind("wall", "flow", "no-slip wall"),
    BoundaryKind("fixed-temperature", "heat", "temperature held", True, "K"),
    BoundaryKind("insulated", "heat", "no heat flux"),
    BoundaryKind("heat-flux", "heat", "heat flux into the domain", True, "W/m^2"),
    BoundaryKind("fixed-potential", "electric", "potential held", True, "V"),
    BoundaryKind("no-current", "electric", "no current through the edge"),
)


def boundary_kind(kid: str) -> BoundaryKind:
    for k in BOUNDARY_KINDS:
        if k.id == kid:
            return k
    raise KeyError(kid)


@dataclass(frozen=True)
class FixedBoundary:
    edge: str                   # "left" | "right" | "bottom" | "top"
    kind: str


@dataclass(frozen=True)
class Family:
    id: str
    label: str
    status: str                 # "ready-to-wire" | "planned"
    solvers: tuple[str, ...]
    devices: tuple[str, ...] = ()
    has_full_domain: bool = False
    note: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)
    #: which geometry layers the family's solvers read
    layers: tuple[str, ...] = ("windows",)
    #: the boundary kinds its solvers can impose
    boundary_kinds: tuple[str, ...] = ()
    #: when the solver fixes its outer boundary, what it is (whole edges); a case
    #: of this family may not change it
    fixed_boundaries: tuple[FixedBoundary, ...] = ()
    fixed_boundaries_why: str = ""


FAMILIES: tuple[Family, ...] = (
    Family(
        id="incompressible-2d",
        label="2-D incompressible flow with actuator disks (wind farm)",
        status="ready-to-wire",
        solvers=("WindowNS, elliptic part exposed (one window)",
                 "RectangularNS on the undivided domain (the full-domain reference)"),
        devices=("actuator-disk",),
        has_full_domain=True,
        note=("Every piece exists and is tested: CS-7's tilings, the projected assembly, "
              "W346's serial and threaded columns and its timing harness. The workbench "
              "runner that drives them from a case file is plan step 3 and is not "
              "built yet."),
        sources=("atlas/cases/scaling_ladder.py", "atlas/cases/wake_array.py",
                 "scripts/w346_rotor_count_speed.py"),
        layers=("windows", "devices", "boundaries"),
        boundary_kinds=("inlet", "freestream", "outlet"),
        fixed_boundaries=(FixedBoundary("left", "inlet"), FixedBoundary("right", "outlet"),
                          FixedBoundary("bottom", "freestream"),
                          FixedBoundary("top", "freestream")),
        fixed_boundaries_why=("The march holds the inlet and both lateral edges at the "
                              "freestream (U, 0), and the global projection tapers the "
                              "outflow downstream (`wake_array.leray_projection`, "
                              "`wake_array._extend`). The projection is built for this "
                              "outer boundary and does not travel to another, so a case "
                              "of this family cannot change it."),
    ),
    Family(
        id="conduction-2d",
        label="2-D heat conduction in several materials (showcase case 2)",
        status="planned",
        solvers=("a Q1 conduction solve per window or region", "the same solve on the "
                 "whole domain (the full-domain reference)"),
        has_full_domain=True,
        note=("The first family that reads material regions: a copper insert in a steel "
              "plate, steady and transient. Not built: it needs the solver-family "
              "interface and coupling styles B and C of the showcase plan."),
        sources=("wiki/concepts/Atlas 0.1/atlas-0.1-outcome/showcase-library-plan.md",
                 "atlas/cases/thermal_seam.py"),
        layers=("regions", "windows", "boundaries"),
        boundary_kinds=("fixed-temperature", "insulated", "heat-flux"),
    ),
    Family(
        id="conduction-loop",
        label="Conduction block on a coolant loop (CS-13)",
        status="planned",
        solvers=("thermostruct2d.step_thermal", "lumped coolant legs"),
        note=("Needs a registry entry with a real boundary response and a runner for "
              "a field coupled to lumped legs (plan scope B). No full-domain reference: "
              "the loop is lumped."),
        sources=("atlas/cases/cooling_loop.py",),
    ),
    Family(
        id="wing-fsi",
        label="Flexible wing in two-way flow (CS-12)",
        status="planned",
        solvers=("FSIRollout fluid windows", "ThermoStruct2D.solve_mechanical"),
        note=("The case's resolution is module constants; a workbench version needs "
              "them as parameters (Tier 89 built a half-resolution copy by patching "
              "them). Plan scope B."),
        sources=("atlas/cases/wing_fsi.py",),
    ),
    Family(
        id="learned-poseidon",
        label="Learned windows: Poseidon-T (frozen)",
        status="planned",
        solvers=("adapters.FrozenFluidExpert",),
        note=("Uniform 128 x 128 windows at one native step only, and CC-BY-NC-4.0 "
              "weights (research use). Plan scope D."),
        sources=("atlas/cases/poseidon.py",),
    ),
)


def family(fid: str) -> Family:
    for f in FAMILIES:
        if f.id == fid:
            return f
    raise KeyError(fid)


def available_ids() -> list[str]:
    return [f.id for f in FAMILIES if f.status == "ready-to-wire"]


__all__ = ["BoundaryKind", "BOUNDARY_KINDS", "boundary_kind", "FixedBoundary",
           "Family", "FAMILIES", "family", "available_ids"]
