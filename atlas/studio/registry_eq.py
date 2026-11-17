"""The equation registry: every PDE the family catalog lists, as data (plan §4).

Each equation is one row of the Physics screen, from the catalog's groups A to L
([[chart-operator-family-catalog]] §2): its type, its group's chart-operator family
and guarantee class (G0, G0′, G1, G2, G3; catalog §1.3), whether a chart-operator
expert exists for it (CO: only CO-D's, today), its properties (order, linearity,
self-adjointness, energy, maximum principle, what it conserves, signal speed, which
equations share its operator, where Atlas uses it), and **either the workbench family
that runs it or why nothing here does** (the page shows that reason on hover).

`regime(spec)` computes the case's regime numbers with their definitions: the
coefficient contrast, the Fourier numbers (per cell, and per domain as the chart
operator's card reads it), the Péclet and Reynolds numbers for flow and transport, and
the wavenumber for waves.  The test computes each by hand on one case.

Nothing here runs a solver; the data is checked against the registry of runnable
families (`atlas.workbench.registry`) by the tests.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

#: the groups of catalog §2, with their chart-operator family and guarantee class
GROUPS = {
    "A": ("Linear diffusion", "CO-D", "G0"),
    "B": ("Nonlinear diffusion and phase change", "CO-DN", "G1"),
    "C": ("Transport", "CO-T", "G0′ / G3"),
    "D": ("Solids", "CO-E", "G0"),
    "E": ("Thin structures", "CO-P", "G0"),
    "F": ("Incompressible flow", "CO-S / CO-NS", "G0 / G2"),
    "G": ("Compressible and reacting flow", "CO-G", "G3"),
    "H": ("Waves", "CO-W", "G0′ / G3"),
    "I": ("3-D electromagnetics", "CO-M", "G0 / G0′"),
    "J": ("Co-located multiphysics: bonds", "bond", "per family"),
    "K": ("Lumped components", "—", "passive"),
    "L": ("Outside the catalog", "—", "—"),
}

#: what each guarantee class promises (catalog §1.3), for the hover
CLASSES = {
    "G0": "Guarantee class G0: linear, symmetric, has an energy. The assembled interface "
          "matrix stays positive semidefinite whatever the network outputs; Céa constant 1; "
          "machine-checked in Lean.",
    "G0′": "Guarantee class G0′: linear but not symmetric (or complex-symmetric). "
           "Petrov–Galerkin with adjoint modes; no positivity guarantee.",
    "G1": "Guarantee class G1: a convex energy. An energy upper bound; at most √(L/m) times "
          "the best; never worse than one classical step.",
    "G2": "Guarantee class G2: no energy. The true residual is exact; no certified error bound.",
    "G3": "Guarantee class G3: explicit hyperbolic. Exact conservation through face fluxes; "
          "positivity by convex limiting.",
}

NOT_BUILT = "Designed in the family catalog, not built: no solver in Atlas runs it yet."
THREE_D = "Three-dimensional: the studio draws and runs two-dimensional cases."
LUMPED = ("A lumped component, not a chart operator: it couples to fields as a "
          "port-Hamiltonian system at the host, and has no geometry to draw.")
OUTSIDE = "Outside the catalog: no chart-operator family is designed for it."


@dataclass(frozen=True)
class Equation:
    id: str
    name: str
    group: str
    kind: str                       # elliptic, parabolic, hyperbolic, mixed
    #: the equation, as HTML with every symbol explained on hover (static data)
    html: str
    guarantee: str
    runs_with: str | None = None    # a workbench family id
    why_not: str = ""               # when nothing here runs it
    co: bool = False                # a chart-operator expert exists for it
    order: str = ""
    linear: bool | None = None
    self_adjoint: bool | None = None
    energy: str = ""
    max_principle: bool | None = None
    conserves: str = ""
    signal_speed: str = ""
    same_operator: tuple[str, ...] = ()
    uses: str = ""
    steady_kind: str = ""           # the type when steady, if different
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        d = asdict(self)
        g = GROUPS[self.group]
        d.update(group_label=g[0], family=g[1], class_tip=CLASSES.get(self.guarantee.split(" ")[0], ""))
        return d


def _a(sym: str, tip: str) -> str:
    return f'<abbr data-tip="{tip}">{sym}</abbr>'


K = _a("<i>κ</i>", "κ: thermal conductivity, W/(m·K)")
RHOC = _a("ρ<i>c</i>", "ρc: volumetric heat capacity, J/(m³·K)")
F = _a("<i>f</i>", "f: heat source per volume, W/m³")
SIG = _a("<i>σ</i>", "σ: electrical conductivity, S/m")

EQUATIONS: tuple[Equation, ...] = (
    # -- A. linear diffusion, CO-D, G0 ----------------------------------------------------
    Equation("heat", "Heat", "A", "parabolic",
             f"{RHOC} <i>∂T</i>/<i>∂t</i> − ∇·({K} ∇<i>T</i>) = {F}", "G0",
             runs_with="conduction-2d", co=True, order="2 in space, 1 in time", linear=True,
             self_adjoint=True, energy="½∫κ|∇T|² − ∫fT", max_principle=True, conserves="heat",
             signal_speed="infinite", same_operator=("darcy", "steady-current", "electrostatics",
                                                     "magnetostatics", "fick"),
             uses="brakes, conductors and motor sections, the nozzle, the airframe",
             steady_kind="elliptic"),
    Equation("steady-current", "Steady current", "A", "elliptic",
             f"−∇·({SIG} ∇<i>φ</i>) = 0", "G0", runs_with="electric-2d", co=True,
             order="2", linear=True, self_adjoint=True, energy="½∫σ|∇φ|² (the Joule power, halved)",
             max_principle=True, conserves="charge", signal_speed="infinite (quasi-static)",
             same_operator=("heat", "electrostatics", "darcy"), uses="conductors, motor sections"),
    Equation("darcy", "Darcy flow", "A", "elliptic",
             "−∇·(" + _a("<i>k</i>/<i>μ</i>", "k/μ: permeability over viscosity, m²/(Pa·s)")
             + " ∇<i>p</i>) = " + _a("<i>q</i>", "q: volume source, 1/s"), "G0", co=True,
             why_not="Same operator as heat (CO-D), but no Darcy case is built here: its "
                     "permeabilities, wells and pressure boundaries have no workbench family.",
             order="2", linear=True, self_adjoint=True, energy="½∫(k/μ)|∇p|² − ∫qp",
             max_principle=True, conserves="fluid volume", signal_speed="infinite",
             same_operator=("heat", "steady-current"), uses="porous cooling, filters"),
    Equation("electrostatics", "Electrostatics", "A", "elliptic",
             "−∇·(" + _a("<i>ε</i>", "ε: permittivity, F/m") + " ∇<i>φ</i>) = "
             + _a("<i>ρ</i>", "ρ: charge density, C/m³"), "G0", co=True,
             why_not="Runs as Steady current: the same operator, with σ in place of ε; no "
                     "separate electrostatics case is built.",
             order="2", linear=True, self_adjoint=True, energy="½∫ε|∇φ|² − ∫ρφ",
             max_principle=True, conserves="charge", signal_speed="infinite (quasi-static)",
             same_operator=("heat", "steady-current"), uses="insulation, sensors"),
    Equation("magnetostatics", "Magnetostatics 2-D", "A", "elliptic",
             "−∇·(" + _a("<i>ν</i>", "ν: magnetic reluctivity, 1/μ, m/H") + " ∇<i>A<sub>z</sub></i>) = "
             + _a("<i>J<sub>z</sub></i>", "J_z: current density out of the plane, A/m²"), "G0", co=True,
             why_not="Same operator as heat (CO-D); no magnetostatics case (coils, iron, "
                     "air gaps) is built.", order="2", linear=True, self_adjoint=True,
             energy="½∫ν|∇A_z|² − ∫J_z A_z", max_principle=True, conserves="flux",
             same_operator=("heat",), uses="motor sections"),
    Equation("potential-flow", "Potential flow", "A", "elliptic", "∇²<i>φ</i> = 0", "G0",
             co=True, why_not="Same operator as heat with κ = 1; no potential-flow case is built.",
             order="2", linear=True, self_adjoint=True, energy="½∫|∇φ|²", max_principle=True,
             conserves="mass", same_operator=("heat",)),
    Equation("torsion", "Prandtl torsion", "A", "elliptic",
             "−∇²<i>ψ</i> = 2" + _a("<i>Gθ</i>", "G: shear modulus; θ: twist per length"), "G0",
             co=True, why_not="Same operator as heat; no torsion case is built.", order="2",
             linear=True, self_adjoint=True, energy="½∫|∇ψ|² − 2Gθ∫ψ", max_principle=True,
             same_operator=("heat",)),
    Equation("fick", "Fick diffusion", "A", "parabolic",
             "<i>∂c</i>/<i>∂t</i> − ∇·(" + _a("<i>D</i>", "D: diffusivity, m²/s") + " ∇<i>c</i>) = <i>s</i>",
             "G0", co=True, why_not="Same operator as heat; runs as Heat with ρc = 1 and κ = D.",
             order="2 in space, 1 in time", linear=True, self_adjoint=True,
             energy="½∫D|∇c|² − ∫sc", max_principle=True, conserves="mass",
             signal_speed="infinite", same_operator=("heat",), steady_kind="elliptic"),
    Equation("newmark-acoustic", "Implicit acoustic step", "A", "elliptic",
             "<i>p</i>/(β<i>c</i>²Δ<i>t</i>²) − ∇²<i>p</i> = <i>s</i>", "G0", co=True,
             why_not="One implicit Newmark step is CO-D's operator; the explicit acoustics "
                     "family runs instead.", order="2", linear=True, self_adjoint=True,
             same_operator=("heat",)),
    # -- B. nonlinear diffusion, CO-DN, G1 --------------------------------------------------
    Equation("heat-nonlinear", "Heat, κ(T)", "B", "parabolic",
             f"{RHOC} <i>∂T</i>/<i>∂t</i> − ∇·(<i>κ</i>(<i>T</i>) ∇<i>T</i>) = {F}", "G1",
             why_not=NOT_BUILT + " The Kirchhoff and frozen-coefficient routes come first "
                                 "(catalog §6).", order="2 in space, 1 in time", linear=False,
             self_adjoint=True, energy="convex in the Kirchhoff variable", max_principle=True,
             conserves="heat", signal_speed="infinite", uses="nozzle, heat shield, hot brakes"),
    Equation("radiation-bc", "Radiating boundaries", "B", "parabolic",
             "−<i>κ</i> ∂<i>T</i>/∂<i>n</i> = <i>εσ</i>(<i>T</i>⁴ − <i>T</i>∞⁴)", "G1",
             why_not=NOT_BUILT, linear=False, energy="∮εσ(⅕T⁵ − T∞⁴T), convex for T ≥ 0"),
    Equation("stefan", "Stefan problem (enthalpy)", "B", "parabolic",
             "<i>∂H</i>/<i>∂t</i> − ∇·(<i>κ</i>∇<i>T</i>(<i>H</i>)) = 0", "G1", why_not=NOT_BUILT,
             linear=False, conserves="enthalpy", uses="ablation"),
    Equation("magnetostatics-sat", "Saturating magnetostatics", "B", "elliptic",
             "−∇·(<i>ν</i>(|∇<i>A</i>|) ∇<i>A</i>) = <i>J</i>", "G1", why_not=NOT_BUILT,
             linear=False, uses="motors"),
    Equation("richards", "Richards equation", "B", "parabolic",
             "∂θ(<i>h</i>)/∂<i>t</i> − ∇·(<i>K</i>(<i>h</i>)∇(<i>h</i> + <i>z</i>)) = 0", "G1",
             why_not=NOT_BUILT, linear=False),
    # -- C. transport, CO-T ------------------------------------------------------------------
    Equation("advection-diffusion", "Advection–diffusion", "C", "parabolic",
             "<i>∂c</i>/<i>∂t</i> + " + _a("<i>u</i>", "u: the advecting velocity, m/s")
             + "·∇<i>c</i> − ∇·(" + _a("<i>D</i>", "D: mixing (diffusivity), m²/s") + "∇<i>c</i>) = <i>s</i>",
             "G0′ (G3 when advection dominates)", runs_with="transport-2d",
             order="2 in space, 1 in time", linear=True, self_adjoint=False, max_principle=True,
             conserves="mass", signal_speed="infinite (diffusion); |u| along the flow",
             uses="coolant, wakes, plume species", notes=("Self-adjoint only without advection.",)),
    Equation("conjugate-heat", "Conjugate heat transfer", "C", "parabolic",
             f"{RHOC}(<i>∂T</i>/<i>∂t</i> + <i>u</i>·∇<i>T</i>) − ∇·({K}∇<i>T</i>) = {F}",
             "G0′ (G3 when advection dominates)", runs_with="conjugate-heat-2d",
             order="2", linear=True, self_adjoint=False, max_principle=True, conserves="heat",
             uses="the coolant beside a heated block (case study cooling loop)",
             steady_kind="elliptic"),
    Equation("advection", "Pure advection, level sets", "C", "hyperbolic",
             "<i>∂c</i>/<i>∂t</i> + <i>u</i>·∇<i>c</i> = 0", "G3", why_not=NOT_BUILT,
             linear=True, conserves="mass", signal_speed="|u|"),
    # -- D. solids ---------------------------------------------------------------------------
    Equation("elasticity", "Linear elasticity", "D", "elliptic",
             "−∇·" + _a("<b>σ</b>", "σ = λ tr(ε) I + 2με, from Young's modulus E and Poisson's ratio ν")
             + "(<b>u</b>) = <b>f</b>", "G0", runs_with="elasticity-2d", order="2", linear=True,
             self_adjoint=True, energy="½∫σ:ε − ∫f·u", max_principle=False,
             conserves="momentum (equilibrium)", signal_speed="infinite (static)",
             uses="airframe, wing, brake pads"),
    Equation("axisymmetric", "Axisymmetric elasticity", "D", "elliptic",
             "−∇·<b>σ</b> + hoop terms (weight <i>r</i>)", "G0", why_not=NOT_BUILT, linear=True,
             self_adjoint=True),
    Equation("elastodynamics", "Implicit elastodynamics", "D", "hyperbolic",
             "ρ<b>ü</b> − ∇·<b>σ</b>(<b>u</b>) = <b>f</b>", "G0", why_not=NOT_BUILT, linear=True,
             self_adjoint=True, signal_speed="the elastic wave speeds"),
    Equation("hyperelasticity", "Hyperelasticity", "D", "elliptic",
             "−∇·<b>P</b>(<b>F</b>) = <b>f</b>", "G1", why_not=NOT_BUILT, linear=False,
             uses="tyre (CS-15)"),
    Equation("plasticity", "Incremental plasticity", "D", "elliptic",
             "min over the increment of a convex incremental energy", "G1", why_not=NOT_BUILT,
             linear=False),
    Equation("contact", "Contact and friction", "D", "elliptic",
             "unilateral contact (host); Coulomb friction (host, G2)", "G1 / G2",
             why_not=NOT_BUILT, linear=False),
    # -- E. thin structures --------------------------------------------------------------------
    Equation("kirchhoff-love", "Kirchhoff–Love plate", "E", "elliptic",
             "∇·∇·(𝔻:∇∇<i>w</i>) = <i>q</i>", "G0", why_not=NOT_BUILT + " (the biharmonic network)",
             order="4", linear=True, self_adjoint=True, energy="½∫𝔻(∇∇w, ∇∇w) − ∫qw",
             uses="rocket skin, wing skins"),
    Equation("reissner-mindlin", "Reissner–Mindlin plate", "E", "elliptic",
             "(<i>w</i>, θ) with shear", "G0", why_not=NOT_BUILT, linear=True, self_adjoint=True),
    Equation("koiter", "Koiter shell", "E", "elliptic", "membrane and bending of a midsurface",
             "G0", why_not=NOT_BUILT + " " + THREE_D, linear=True, self_adjoint=True),
    # -- F. incompressible flow ------------------------------------------------------------------
    Equation("stokes", "Stokes 2-D", "F", "elliptic", "<i>ν</i>∇⁴<i>ψ</i> = <i>ẑ</i>·∇×<b>f</b>",
             "G0", why_not=NOT_BUILT + " (the plate network relabelled)", order="4", linear=True,
             self_adjoint=True, energy="½∫ν|∇∇ψ|²", uses="slow flow in ducts"),
    Equation("navier-stokes", "Navier–Stokes", "F", "mixed",
             "<i>∂</i><b>u</b>/<i>∂t</i> + (<b>u</b>·∇)<b>u</b> = −∇<i>p</i> + "
             + _a("<i>ν</i>", "ν: kinematic viscosity, m²/s") + "∇²<b>u</b>, ∇·<b>u</b> = 0",
             "G2", runs_with="incompressible-2d", order="2 in space, 1 in time", linear=False,
             self_adjoint=False, energy="none (kinetic energy dissipates)", max_principle=False,
             conserves="mass and momentum", signal_speed="infinite (incompressible)",
             uses="ground effect, ducts, the wind farm"),
    Equation("rans", "RANS", "F", "mixed", "averaged Navier–Stokes with a turbulence model", "G2",
             why_not=NOT_BUILT, linear=False),
    Equation("oseen", "Oseen step", "F", "elliptic", "linearised Navier–Stokes about a known flow",
             "G0′", why_not=NOT_BUILT, linear=True, self_adjoint=False),
    # -- G. compressible -----------------------------------------------------------------------
    Equation("euler", "Euler equations", "G", "hyperbolic",
             "<i>∂</i><b>U</b>/<i>∂t</i> + ∇·<b>F</b>(<b>U</b>) = 0", "G3",
             why_not="Runs in the rocket case study (compressible2d), not as a family the "
                     "studio can draw.", linear=False, conserves="mass, momentum, energy",
             signal_speed="|u| + a", uses="rocket nozzle and plume"),
    Equation("compressible-ns", "Compressible Navier–Stokes", "G", "mixed",
             "Euler plus viscous fluxes (G0 per step)", "G3", why_not=NOT_BUILT, linear=False),
    Equation("reacting", "Reacting flow", "G", "mixed",
             "∂(ρ<i>Y<sub>k</sub></i>)/∂<i>t</i> + ∇·(ρ<b>u</b><i>Y<sub>k</sub></i> − ρ<i>D<sub>k</sub></i>∇<i>Y<sub>k</sub></i>) = ω̇<sub>k</sub>",
             "G3", why_not=NOT_BUILT, linear=False),
    # -- H. waves ------------------------------------------------------------------------------
    Equation("acoustics", "Acoustics", "H", "hyperbolic",
             "<i>∂²p</i>/<i>∂t²</i> − ∇·(" + _a("<i>c</i>²", "c: sound speed, m/s") + "∇<i>p</i>) = 0",
             "G3", runs_with="acoustics-2d", order="2 in space, 2 in time", linear=True,
             self_adjoint=True, energy="½∫(p_t²/c² + |∇p|²)", max_principle=False,
             conserves="acoustic energy (without loss)", signal_speed="c",
             uses="lift-off and exhaust acoustics", notes=("Explicit, CFL-limited.",)),
    Equation("helmholtz", "Helmholtz", "H", "elliptic",
             "−∇²<i>p</i> − <i>k</i>²<i>p</i> = <i>s</i>", "G0′", why_not=NOT_BUILT, linear=True,
             self_adjoint=False, notes=("Complex-symmetric with impedance ports.",)),
    # -- I. 3-D EM -------------------------------------------------------------------------------
    Equation("eddy", "Eddy currents", "I", "parabolic",
             "∇×(<i>ν</i>∇×<b>A</b>) + <i>σ</i>∂<b>A</b>/∂<i>t</i> = <b>J</b><sub>s</sub>", "G0",
             why_not=THREE_D, linear=True, self_adjoint=True, uses="motor end regions"),
    Equation("maxwell-harmonic", "Time-harmonic Maxwell", "I", "elliptic",
             "∇×(μ⁻¹∇×<b>E</b>) − ω²ε<b>E</b> = −iω<b>J</b>", "G0′", why_not=THREE_D, linear=True),
    # -- J. bonds ----------------------------------------------------------------------------------
    Equation("thermoelasticity", "Thermoelasticity", "J", "parabolic",
             "heat, then −∇·<b>σ</b>(<b>u</b>, <i>T</i>) = 0 with thermal strain α(<i>T</i> − <i>T</i>₀)",
             "G0", runs_with="thermoelastic-2d", order="2", linear=True, self_adjoint=True,
             conserves="heat; momentum", uses="thermal strain, the airframe",
             notes=("A bond between CO-D and CO-E (I-3).",)),
    Equation("biot", "Biot poroelasticity", "J", "parabolic", "elasticity bonded to Darcy flow",
             "G0", why_not=NOT_BUILT, linear=True),
    Equation("piezo", "Piezoelectricity", "J", "elliptic", "elasticity bonded to electrostatics",
             "G0", why_not=NOT_BUILT, linear=True),
    Equation("joule", "Joule heating", "J", "parabolic", "current with σ(T) heating the solid",
             "G1", why_not=NOT_BUILT, linear=False, uses="conductors"),
    # -- K, L ----------------------------------------------------------------------------------
    Equation("lumped", "Circuits, pipe and thermal networks", "K", "mixed",
             "<i>ẋ</i> = (𝒥 − ℛ)∇<i>H</i> + <i>Bu</i>, <i>y</i> = <i>B</i>ᵀ∇<i>H</i>", "passive",
             why_not=LUMPED),
    Equation("radiative-transfer", "Radiative transfer", "L", "hyperbolic",
             "transport of intensity in angle and space", "—", why_not=OUTSIDE),
    Equation("plasma", "Plasma and MHD", "L", "mixed", "fluid and Maxwell coupled", "—",
             why_not=OUTSIDE),
)

BY_ID = {e.id: e for e in EQUATIONS}


def equations(available: list[str] | None = None) -> list[dict]:
    """The rows, with whether each runs here (its family is available)."""
    if available is None:
        from atlas.workbench import registry
        available = registry.available_ids()
    out = []
    for e in EQUATIONS:
        d = e.as_dict()
        runs = e.runs_with is not None and e.runs_with in available
        d["runs"] = runs
        if not runs and not d["why_not"]:
            d["why_not"] = f"{e.runs_with} is not available on this machine."
        out.append(d)
    return out


def of_family(family: str) -> str | None:
    """The equation a workbench family runs."""
    for e in EQUATIONS:
        if e.runs_with == family:
            return e.id
    return None


# ---------------------------------------------------------------------------
# regime numbers
# ---------------------------------------------------------------------------

#: the coefficient whose contrast matters, per family
COEFFICIENT = {"conduction-2d": "k", "thermoelastic-2d": "k", "conjugate-heat-2d": "k",
               "electric-2d": "sigma", "elasticity-2d": "E", "transport-2d": "mixing",
               "acoustics-2d": "c"}
SYMBOL = {"k": "κ", "sigma": "σ", "E": "E", "mixing": "D", "c": "c"}


def _used_materials(spec) -> dict:
    mats = dict(spec.materials)
    used = [r.material for r in spec.regions if r.material in mats]
    return {m: mats[m] for m in dict.fromkeys(used)} or mats


def regime(spec, envelope=None) -> dict:
    """The case's regime numbers, each with its definition, the inputs it read and,
    where an expert's card declares a range, that range and whether the number is in it.

    * contrast: the largest coefficient over the smallest, over the materials in use;
    * Fo (cell) = α Δt / h², α = κ/(ρc) the largest diffusivity, h the cell size;
    * Fo (domain) = κ Δt / (ρc |Ω|), as the chart operator's card reads it per piece
      (here the whole domain is one piece);
    * Pe = |u| L / D for transport, with the cell Péclet |u| h / D beside it;
    * Re_D = U D / ν for the wind farm's flow (lengths in rotor diameters);
    * k h for waves, with k = 3/w the wavenumber where a Gaussian pulse of width w has
      fallen to about 1% of its peak.
    """
    fam = spec.physics.family
    p = dict(spec.physics.params)
    mats = _used_materials(spec)
    dx = float(spec.domain.dx)
    L = float(max(spec.domain.nx, spec.domain.ny)) * dx
    area = float(spec.domain.nx * spec.domain.ny) * dx * dx
    if spec.domain.outline:
        try:
            from shapely.geometry import Polygon
            area = float(Polygon(spec.domain.outline).area)
            for hole in spec.domain.holes:
                area -= float(Polygon(hole).area)
        except Exception:
            pass
    transient = spec.run.mode != "steady"
    dt = float(spec.run.macro_dt)
    out: dict = {"family": fam, "h": dx, "L": L, "area": area, "transient": transient, "dt": dt}
    coef = COEFFICIENT.get(fam)
    if coef:
        vals = [float(m[coef]) for m in mats.values() if coef in m and float(m[coef]) > 0]
        if vals:
            out["contrast"] = {"value": max(vals) / min(vals), "max": max(vals), "min": min(vals),
                               "symbol": SYMBOL[coef],
                               "definition": f"largest {SYMBOL[coef]} over smallest, over the materials in use"}
    if fam in ("conduction-2d", "thermoelastic-2d", "conjugate-heat-2d"):
        alphas = {m: float(v["k"]) / (float(v["rho"]) * float(v["cp"])) for m, v in mats.items()
                  if all(k in v for k in ("k", "rho", "cp")) and float(v["rho"]) * float(v["cp"]) > 0}
        if alphas:
            m = max(alphas, key=alphas.get)
            a = alphas[m]
            out["alpha"] = {"value": a, "material": m}
            if transient:
                out["fo_cell"] = {"value": a * dt / dx ** 2,
                                  "definition": "Fo = κΔt / (ρc h²): how far heat spreads in one step, relative to a cell"}
                fo_dom = a * dt / area
                env = envelope if envelope is not None else _card_envelope()
                d = {"value": fo_dom, "definition": "Fo = κΔt / (ρc |Ω|): the chart operator card's Fourier number, the whole domain as one piece"}
                if env is not None:
                    d.update(range=[env["fo_min"], env["fo_max"]], within=env["fo_min"] <= fo_dom <= env["fo_max"])
                out["fo_domain"] = d
            else:
                env = envelope if envelope is not None else _card_envelope()
                out["fo_domain"] = {"value": None, "definition": "steady: 1/Fo = 0",
                                    "within": bool(env and env.get("steady"))}
        if fam == "conjugate-heat-2d":
            u = float(p.get("u_coolant", 0.0))
            fl = {m: v for m, v in mats.items() if float(v.get("flows", 0)) > 0.5}
            if fl and u > 0:
                m, v = next(iter(fl.items()))
                D = float(v["k"]) / (float(v["rho"]) * float(v["cp"]))
                out["pe"] = {"value": u * L / D, "cell": u * dx / D, "material": m,
                             "definition": "Pe = |u|L/α: advection over diffusion in the coolant"}
    if fam == "transport-2d":
        q = float(p.get("q", 0.0))
        best = None
        for m, v in mats.items():
            depth, D = float(v.get("depth", 0)), float(v.get("mixing", 0))
            if depth > 0 and D > 0 and q > 0:
                u = q / depth                    # each reach runs at q / depth
                if best is None or u * L / D > best["value"]:
                    best = {"value": u * L / D, "cell": u * dx / D, "u": u, "D": D, "material": m,
                            "definition": "Pe = |u|L/D per reach, u = q/depth its speed; the largest"}
        if best:
            out["pe"] = best
    if fam == "incompressible-2d":
        nu = float(p.get("nu", 0.0))
        u = float(p.get("u_inf", 0.0))
        if nu > 0:
            # lengths are in rotor diameters D, so Re_D = U * 1 / nu
            out["re"] = {"value": u * 1.0 / nu, "cell": u * dx / nu,
                         "definition": "Re_D = U D/ν, D the rotor diameter (the length unit); the cell Reynolds number U h/ν beside it"}
    if fam == "acoustics-2d":
        w = float(p.get("pulse_width", 0.0))
        if w > 0:
            k = 3.0 / w
            out["k"] = {"value": k, "kh": k * dx,
                        "definition": "k = 3/w, w the pulse's standard deviation: where its spectrum exp(−k²w²/2) falls to about 1%; kh against the CO-W card's π√2 limit for value ports",
                        "range": [0.0, math.pi * math.sqrt(2.0)], "within": k * dx < math.pi * math.sqrt(2.0)}
    return out


def _card_envelope() -> dict | None:
    """The CO-D card's envelope (atlas/chartop/data.py), read, never changed."""
    try:
        from atlas.chartop.data import ENVELOPE
        return ENVELOPE.as_dict()
    except Exception:
        return None


__all__ = ["BY_ID", "CLASSES", "EQUATIONS", "Equation", "GROUPS", "equations", "of_family",
           "regime"]
