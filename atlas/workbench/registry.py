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
    BoundaryKind("river-inlet", "transport", "the river enters, carrying none of the pollutant"),
    BoundaryKind("river-outlet", "transport", "the river leaves, carrying what it holds"),
    BoundaryKind("bank", "transport", "a bank: no water and no pollutant cross it"),
    BoundaryKind("rigid-wall", "acoustic", "a rigid wall: no velocity through it"),
    BoundaryKind("clamped", "mechanical", "held fixed: no displacement"),
    BoundaryKind("free", "mechanical", "free: no load"),
    BoundaryKind("load-x", "mechanical", "a uniform traction along x", True, "Pa"),
    BoundaryKind("load-y", "mechanical", "a uniform traction along y (negative is down)",
                 True, "Pa"),
    BoundaryKind("coolant-inlet", "heat", "the coolant enters at this temperature", True, "K"),
    BoundaryKind("coolant-outlet", "heat", "the coolant leaves, carrying its heat"),
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
    #: schema 0.4: whether its adapter runs on drawn shapes (a drawn outline, holes,
    #: curved windows), and if not, why; and the condition a newly drawn edge
    #: starts with
    drawn_shapes: bool = False
    drawn_why: str = "its adapter has not been taught drawn shapes yet"
    drawn_default: str = ""
    #: what the Physics step calls the regions' materials, and what it says of the
    #: library's values
    #: its coupling needs no iteration (an explicit exchange every step), so the
    #: page offers no tolerance, relaxation or convergence curve for it
    explicit_coupling: bool = False
    materials_title: str = "Materials"
    materials_note: str = ("Library values are textbook round values at room temperature, "
                           "for a showcase, not a datasheet. A region's material is set in "
                           "Geometry, Regions.")

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
    drawn_why=("W346's window solver (wake_array.WindowNS) marches rectangular windows, "
               "batched by shape, and the global projection is built for the rectangle's "
               "outer boundary"),
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
    drawn_shapes=True,
    drawn_why="",
    drawn_default="insulated",
)

_ELECTRIC = Family(
    id="electric-2d",
    drawn_why=("its electrodes' currents are read on the grid's four edges; drawn edges "
               "are not wired into it yet"),
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

_TRANSPORT = Family(
    id="transport-2d",
    drawn_why=("the river's flow is given per reach across the whole grid, not solved, so "
               "a drawn bank would have no flow that follows it"),
    label="2-D pollutant plume down a river (advection and mixing)",
    status="ready-to-wire",
    solvers=("explicit upwinded finite volumes per window (fv.py), one step per exchange",
             "the same explicit finite volumes on the whole river (the full-domain reference)"),
    has_full_domain=True,
    note=("Runs: a pollutant released at an outfall, carried by a river whose reaches (the "
          "regions) differ in depth and mixing, on overlapping windows with one exchange per "
          "explicit step (style A), marched serially, on threads and on the whole river in "
          "turns. For an explicit step that is the full-domain step, so the arms agree to "
          "round-off."),
    sources=("atlas/workbench/families/plume.py", "atlas/workbench/fv.py"),
    layers=("regions", "windows", "boundaries"),
    boundary_kinds=("river-inlet", "river-outlet", "bank"),
    fixed_boundaries=(FixedBoundary("left", "river-inlet"),
                      FixedBoundary("right", "river-outlet"),
                      FixedBoundary("bottom", "bank"), FixedBoundary("top", "bank")),
    fixed_boundaries_why=("The river runs along +x at the discharge the case sets, so its "
                          "water enters on the left, leaves on the right, and does not cross "
                          "the banks. A case of this family cannot turn the flow, so it "
                          "cannot change these."),
    adapter="atlas.workbench.families.plume",
    styles=("A",),
    params=(Param("q", "Discharge per metre of width", "m^2/s", 1.0, 0.0,
                  help="uniform along the river, so the water runs at q / depth in each "
                       "reach"),
            Param("release", "Release at the outfall", "g/s", 10.0, 0.0),
            Param("source_x", "Outfall, along the river", "m", 200.0, 0.0, strict=False),
            Param("source_y", "Outfall, across the river", "m", 100.0, 0.0, strict=False)),
    material_props=(Param("depth", "Depth", "m", 2.0, 0.0),
                    Param("mixing", "Mixing coefficient", "m^2/s", 1.0, 0.0)),
    #: round values chosen for a plume that visibly slows and widens at the seam
    materials=(("shallow-fast", (("depth", 1.25), ("mixing", 0.5))),
               ("deep-slow", (("depth", 2.5), ("mixing", 2.0))),
               ("pool", (("depth", 4.0), ("mixing", 4.0)))),
    modes=("transient",),
    length_unit="m",
    materials_title="Reaches (the regions' materials)",
    materials_note=("Showcase reach types: a depth and a depth-averaged mixing coefficient "
                    "each, round values chosen so the plume visibly slows and widens, not "
                    "calibrated to a river. A region's reach is set in Geometry, Regions."),
)

_ACOUSTICS = Family(
    id="acoustics-2d",
    drawn_why=("the staggered-grid leapfrog runs on the whole rectangle, cut into two "
               "full-height pieces side by side"),
    label="2-D sound through two media (linear acoustics)",
    status="ready-to-wire",
    solvers=("staggered-grid leapfrog per piece, trading the interface's pressures and "
             "velocities every step (acoustics.py)",
             "the same leapfrog on the whole domain (the full-domain reference)"),
    has_full_domain=True,
    note=("Runs: a plane pulse crossing from one medium into another, on two pieces that "
          "meet at the interface and trade its pressures and velocities once a step "
          "(style C with an explicit step, so nothing iterates), against the same leapfrog "
          "on the whole domain: the two agree to the bit. The reflection is read against "
          "(Z2 - Z1) / (Z2 + Z1)."),
    sources=("atlas/workbench/families/acoustics.py",),
    layers=("regions", "windows", "boundaries"),
    boundary_kinds=("rigid-wall",),
    fixed_boundaries=(FixedBoundary("left", "rigid-wall"), FixedBoundary("right", "rigid-wall"),
                      FixedBoundary("bottom", "rigid-wall"), FixedBoundary("top", "rigid-wall")),
    fixed_boundaries_why=("Every edge is a rigid wall, with no velocity through it, so the "
                          "domain is closed and keeps its energy, which is the family's "
                          "balance check. It has no open boundary to offer."),
    adapter="atlas.workbench.families.acoustics",
    styles=("C",),
    params=(Param("amplitude", "Pulse amplitude", "Pa", 1.0, 0.0),
            Param("pulse_x", "Pulse centre", "m", 0.7, 0.0,
                  help="the pulse starts here in the first piece and runs towards +x"),
            Param("pulse_width", "Pulse width (standard deviation)", "m", 0.1, 0.0)),
    material_props=(Param("rho", "Density", "kg/m^3", 1.2, 0.0),
                    Param("c", "Speed of sound", "m/s", 343.0, 0.0)),
    #: room-temperature textbook values
    materials=(("air", (("rho", 1.2), ("c", 343.0))),
               ("water", (("rho", 1000.0), ("c", 1480.0))),
               ("helium", (("rho", 0.166), ("c", 1007.0)))),
    modes=("transient",),
    length_unit="m",
    explicit_coupling=True,
    materials_title="Media (the regions' materials)",
    materials_note=("Textbook round values at room temperature, for a showcase. A region's "
                    "medium is set in Geometry, Regions."),
)

#: structural metals, textbook round values at room temperature: Young's modulus,
#: Poisson's ratio, expansion, conductivity, density, specific heat
_METALS = (("steel", (("E", 200e9), ("nu", 0.30), ("alpha", 12e-6), ("k", 45.0),
                      ("rho", 7850.0), ("cp", 490.0))),
           ("aluminium", (("E", 70e9), ("nu", 0.33), ("alpha", 23e-6), ("k", 205.0),
                          ("rho", 2700.0), ("cp", 900.0))),
           ("copper", (("E", 120e9), ("nu", 0.34), ("alpha", 17e-6), ("k", 400.0),
                       ("rho", 8960.0), ("cp", 385.0))))

_ELASTICITY = Family(
    id="elasticity-2d",
    drawn_why=("fe.py's elements cover the whole grid; drawn shapes need an element "
               "mask, not built yet"),
    label="2-D plane-stress elasticity in several materials (a loaded bracket)",
    status="ready-to-wire",
    solvers=("Q1 plane-stress elements per window, a sparse LU each (fe.py, the build "
             "repo's ThermoStruct2D element with a material per element)",
             "the same elements on the whole plate, solved directly (the full-domain "
             "reference)"),
    has_full_domain=True,
    note=("Runs: a plate of several materials, clamped and loaded on its edges, on "
          "overlapping windows iterated to agreement (style B, restricted additive Schwarz "
          "on the stiffness), against the same elements solved directly. Steady: each "
          "repeat is the whole solve. Checked: the supports balance the loads."),
    sources=("atlas/workbench/families/elasticity.py", "atlas/workbench/fe.py",
             "atlas/workbench/styles.py"),
    layers=("regions", "windows", "boundaries"),
    boundary_kinds=("clamped", "free", "load-x", "load-y"),
    adapter="atlas.workbench.families.elasticity",
    styles=("B",),
    params=(Param("thickness", "Plate thickness", "m", 0.01, 0.0,
                  help="plane stress: the displacements do not depend on it; the forces "
                       "reported do"),),
    material_props=(Param("E", "Young's modulus", "Pa", 200e9, 0.0),
                    Param("nu", "Poisson's ratio", "", 0.3, -1.0, maximum=0.4999)),
    materials=_METALS,
    modes=("steady",),
    length_unit="m",
    default_boundaries=(("left", "clamped", None), ("right", "load-y", -5.0e6),
                        ("bottom", "free", None), ("top", "free", None)),
)

_THERMOELASTIC = Family(
    id="thermoelastic-2d",
    drawn_why=("fe.py's elements cover the whole grid; drawn shapes need an element "
               "mask, not built yet"),
    label="2-D heated plate that expands (conduction, then thermal strain)",
    status="ready-to-wire",
    solvers=("Q1 backward-Euler conduction (fe.py): the conduction agent",
             "Q1 quasi-static plane stress of a free body under thermal strain (fe.py): the "
             "elasticity agent",
             "the two in one object, stepped as the build repo's ThermoStruct2D steps them "
             "(the unsplit reference)"),
    has_full_domain=True,
    note=("Runs: a plate warming from its edges and deforming as it warms, split by physics "
          "rather than space: a conduction agent and an elasticity agent on one mesh, the "
          "whole temperature field crossing between them, run synchronously and lagged by a "
          "step (on two threads), against the unsplit solver."),
    sources=("atlas/workbench/families/thermoelastic.py", "atlas/workbench/fe.py",
             "atlas/cases/thermal_strain.py"),
    layers=("regions", "windows", "boundaries"),
    boundary_kinds=("fixed-temperature", "insulated", "heat-flux"),
    adapter="atlas.workbench.families.thermoelastic",
    styles=("split",),
    params=(Param("T0", "Initial temperature", "K", 300.0, 0.0,
                  help="the plate starts here, and is free of strain here"),),
    material_props=(Param("E", "Young's modulus", "Pa", 200e9, 0.0),
                    Param("nu", "Poisson's ratio", "", 0.3, -1.0, maximum=0.4999),
                    Param("alpha", "Thermal expansion", "1/K", 12e-6, 0.0, strict=False),
                    Param("k", "Conductivity", "W/(m K)", 45.0, 0.0),
                    Param("rho", "Density", "kg/m^3", 7850.0, 0.0),
                    Param("cp", "Specific heat", "J/(kg K)", 490.0, 0.0)),
    materials=_METALS,
    modes=("transient",),
    length_unit="m",
    default_boundaries=(("left", "fixed-temperature", 400.0),
                        ("right", "insulated", None), ("bottom", "insulated", None),
                        ("top", "insulated", None)),
)

_COOLING = Family(
    id="conjugate-heat-2d",
    drawn_why=("the coolant fills whole rows from the left edge to the right; a drawn "
               "channel needs a flow field, not built yet"),
    label="2-D heated block cooled by channel flow (conjugate heat transfer)",
    status="ready-to-wire",
    solvers=("cell-centred finite volumes with the coolant's advection (fv.py): the channel",
             "the same finite volumes without it: the block",
             "both on every cell in one system (the full-domain reference)"),
    has_full_domain=True,
    note=("Runs: a block with a heat source cooled by a plug flow in a channel beside it, "
          "the channel (advection and diffusion) and the block (conduction) as two pieces "
          "meeting at the wall and coupled by Dirichlet-Neumann (style C at a seam between "
          "two physics), against both on every cell solved directly. Steady. Checked: every "
          "watt generated leaves in the coolant."),
    sources=("atlas/workbench/families/cooling.py", "atlas/workbench/fv.py",
             "atlas/workbench/styles.py"),
    layers=("regions", "windows", "boundaries"),
    boundary_kinds=("coolant-inlet", "coolant-outlet", "insulated", "fixed-temperature",
                    "heat-flux"),
    adapter="atlas.workbench.families.cooling",
    styles=("C",),
    params=(Param("u_coolant", "Coolant speed", "m/s", 0.01, 0.0,
                  help="a plug flow along +x in the coolant's rows"),),
    material_props=(Param("k", "Conductivity", "W/(m K)", 1.0, 0.0),
                    Param("rho", "Density", "kg/m^3", 1000.0, 0.0),
                    Param("cp", "Specific heat", "J/(kg K)", 1000.0, 0.0),
                    Param("flows", "Flows as the coolant (1) or is solid (0)", "", 0.0, 0.0,
                          strict=False, maximum=1.0),
                    Param("heat", "Heat generated", "W/m^3", 0.0, 0.0, strict=False)),
    materials=(("water", (("k", 0.6), ("rho", 1000.0), ("cp", 4180.0), ("flows", 1.0),
                          ("heat", 0.0))),
               ("copper", (("k", 400.0), ("rho", 8960.0), ("cp", 385.0), ("flows", 0.0),
                           ("heat", 0.0))),
               ("aluminium", (("k", 205.0), ("rho", 2700.0), ("cp", 900.0), ("flows", 0.0),
                              ("heat", 0.0))),
               #: silicon's conductivity, generating 10 W per cubic centimetre
               ("chip", (("k", 150.0), ("rho", 2330.0), ("cp", 700.0), ("flows", 0.0),
                         ("heat", 1.0e7)))),
    modes=("steady",),
    length_unit="m",
    default_boundaries=(("left", "insulated", None), ("right", "insulated", None),
                        ("bottom", "insulated", None), ("top", "insulated", None)),
)

FAMILIES: tuple[Family, ...] = (
    _WIND_FARM,
    _CONDUCTION,
    _ELECTRIC,
    _TRANSPORT,
    _ACOUSTICS,
    _ELASTICITY,
    _THERMOELASTIC,
    _COOLING,
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
