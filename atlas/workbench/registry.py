"""Which physics families the workbench knows, and how far each is wired.

Step 1 of [[outcome-c4-path-to-declarative-cases]]: Python written once per
solver, never per case.  Each family names the solvers and devices it uses,
**what is missing before the workbench can run it** (so the GUI can show an
unavailable option with its reason instead of hiding it), and, once it runs, the
adapter module that marches a case of it (``adapter``; see `families/`).  That
field is what makes this a factory rather than a catalogue: the runner asks
the registry for a family's adapter and never names one itself.

A family also declares what a case of it may set, so the page can draw the
Physics step from the registry rather than from code per family: its scalar
parameters (``params``), the properties it reads from each material
(``material_props``) with a small library of showcase materials, the run modes
its adapter supports (steady, transient), and the coupling styles.
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
    BoundaryKind("electrode", "electric",
                 "a terminal: the lumped circuit wired to it sets its potential, and the "
                 "current through it is the circuit's"),
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
class Param:
    """A number a case of a family may set, with its unit and its range."""

    name: str
    label: str
    unit: str
    default: float
    #: lower bound; the value must be STRICTLY above it when ``strict``
    minimum: float | None = None
    strict: bool = True
    maximum: float | None = None
    help: str = ""

    def problem(self, value) -> str | None:
        """Why ``value`` is not allowed, or None."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return f"{self.label} is not a number"
        if v != v or v in (float("inf"), float("-inf")):
            return f"{self.label} is not finite"
        if self.minimum is not None and (v <= self.minimum if self.strict
                                         else v < self.minimum):
            return (f"{self.label} must be {'above' if self.strict else 'at least'} "
                    f"{self.minimum:g} {self.unit}".rstrip())
        if self.maximum is not None and v > self.maximum:
            return f"{self.label} must be at most {self.maximum:g} {self.unit}".rstrip()
        return None


#: What each coupling style is, for the page.
STYLES = {
    "A": "overlapping windows, one exchange per step",
    "B": "overlapping windows, iterated until they agree (additive Schwarz)",
    "C": "two pieces meeting at an interface, no overlap (Dirichlet-Neumann)",
    "D": "a field joined to lumped parts through its boundary integrals",
    "split": "one domain split by physics rather than by space",
}


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
    #: the module that runs a case of this family (``build(spec, arms, threads)``);
    #: empty until it runs
    adapter: str = ""
    #: the showcase plan's coupling styles its adapter runs (A, B, C, D, split)
    styles: tuple[str, ...] = ()
    #: the scalar parameters a case sets (``CaseSpec.physics.params``)
    params: tuple[Param, ...] = ()
    #: the properties it reads from each region's material
    material_props: tuple[Param, ...] = ()
    #: showcase materials: name -> ((property, value), ...); textbook round values
    materials: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    #: run modes the adapter supports
    modes: tuple[str, ...] = ("transient",)
    #: the length unit its cell size is in
    length_unit: str = "m"
    #: the boundaries a new case of the family starts with (edge -> (kind, value))
    default_boundaries: tuple[tuple[str, str, float | None], ...] = ()

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def default_params(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}

    def material_library(self) -> dict[str, dict[str, float]]:
        return {name: dict(props) for name, props in self.materials}


_WIND_FARM = Family(
    id="incompressible-2d",
    label="2-D incompressible flow with actuator disks (wind farm)",
    status="ready-to-wire",
    solvers=("WindowNS, elliptic part exposed (one window)",
             "RectangularNS on the undivided domain (the full-domain reference)"),
    devices=("actuator-disk",),
    has_full_domain=True,
    note=("Runs: overlapping windows of any size, one exchange per macro-step "
          "(style A), marched serially, on threads and on the undivided domain in "
          "turns. The march is W346's, lifted: on the example tilings the serial "
          "and full-domain arms are W346's own columns to the bit."),
    sources=("atlas/workbench/families/windfarm.py", "atlas/cases/scaling_ladder.py",
             "atlas/cases/wake_array.py", "scripts/w346_rotor_count_speed.py"),
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
    adapter="atlas.workbench.families.windfarm",
    styles=("A",),
    params=(Param("nu", "Viscosity nu", "D U", 3.92e-3, 0.0,
                  help="W0 4.2's grid-scale fit, the referent's own (Re_D = 255)"),
            Param("u_inf", "Freestream speed", "U", 1.0, 0.0)),
    modes=("transient",),
    length_unit="D",
)

_CONDUCTION = Family(
    id="conduction-2d",
    label="2-D heat conduction in several materials",
    status="ready-to-wire",
    solvers=("cell-centred finite volumes per window or piece (fv.py), a sparse LU each",
             "the same finite volumes on the whole domain (the full-domain reference)"),
    has_full_domain=True,
    note=("Runs: steady or transient (backward Euler) conduction with a material per "
          "region, decomposed by overlapping windows iterated to agreement (style B, "
          "restricted additive Schwarz) or by two pieces meeting at an interface "
          "(style C, Dirichlet-Neumann with stated relaxation). Harmonic-mean faces make "
          "a layered wall exact, so a wall of layers is checked against its closed form."),
    sources=("atlas/workbench/families/conduction.py", "atlas/workbench/fv.py",
             "atlas/workbench/styles.py"),
    layers=("regions", "windows", "boundaries"),
    boundary_kinds=("fixed-temperature", "insulated", "heat-flux"),
    adapter="atlas.workbench.families.conduction",
    styles=("B", "C"),
    params=(Param("T0", "Initial temperature", "K", 300.0, 0.0,
                  help="the whole plate starts here (and it is the first guess of a "
                       "steady solve)"),),
    material_props=(Param("k", "Conductivity", "W/(m K)", 1.0, 0.0),
                    Param("rho", "Density", "kg/m^3", 1000.0, 0.0),
                    Param("cp", "Specific heat", "J/(kg K)", 1000.0, 0.0)),
    materials=(("steel", (("k", 45.0), ("rho", 7850.0), ("cp", 490.0))),
               ("copper", (("k", 400.0), ("rho", 8960.0), ("cp", 385.0))),
               ("aluminium", (("k", 205.0), ("rho", 2700.0), ("cp", 900.0)))),
    modes=("steady", "transient"),
    length_unit="m",
    default_boundaries=(("left", "fixed-temperature", 400.0),
                        ("right", "fixed-temperature", 300.0),
                        ("bottom", "insulated", None), ("top", "insulated", None)),
)

_ELECTRIC = Family(
    id="electric-2d",
    label="2-D current spreading in a resistive plate on a lumped circuit",
    status="ready-to-wire",
    solvers=("cell-centred finite volumes for div(sigma grad phi) = 0 (fv.py)",
             "a lumped circuit of batteries and resistors (nodal analysis)",
             "both in one sparse system (the full-domain reference)"),
    has_full_domain=True,
    note=("Runs: a conducting plate whose electrodes are wired to batteries and "
          "resistors, the field and the circuit joined through the electrodes' "
          "currents (style D: a field's boundary integral is a lumped part's port "
          "variable), against the plate and the circuit solved as one system."),
    sources=("atlas/workbench/families/electric.py", "atlas/workbench/fv.py",
             "atlas/workbench/styles.py"),
    layers=("regions", "windows", "boundaries", "attachments"),
    boundary_kinds=("electrode", "fixed-potential", "no-current"),
    adapter="atlas.workbench.families.electric",
    styles=("D",),
    params=(Param("thickness", "Plate thickness", "m", 1.0e-3, 0.0,
                  help="the plate is 2-D; its currents per metre of depth are "
                       "multiplied by this to meet the circuit's amperes"),),
    material_props=(Param("sigma", "Electrical conductivity", "S/m", 1.0, 0.0),),
    #: from textbook resistivities at room temperature, rounded: nichrome
    #: 1.10e-6 ohm m, constantan 4.9e-7 ohm m, polycrystalline graphite ~1e-5
    materials=(("graphite", (("sigma", 1.0e5),)),
               ("nichrome", (("sigma", 9.1e5),)),
               ("constantan", (("sigma", 2.04e6),))),
    modes=("steady",),
    length_unit="m",
    default_boundaries=(("left", "no-current", None), ("right", "no-current", None),
                        ("bottom", "no-current", None), ("top", "no-current", None)),
)

FAMILIES: tuple[Family, ...] = (
    _WIND_FARM,
    _CONDUCTION,
    _ELECTRIC,
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


__all__ = ["BoundaryKind", "BOUNDARY_KINDS", "boundary_kind", "FixedBoundary", "Param",
           "STYLES", "Family", "FAMILIES", "family", "available_ids"]
