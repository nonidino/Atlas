"""The 2D rocket ascent: seven agents, seven typed edges, no ROT or ELEC ports
anywhere -- which is a correct statement about a solid-propellant vehicle.

**This docstring used to say the rocket "should not be the next build", because
every refusal it drew was a genuine missing rule and three of them were research.
That was written before CS-9 through CS-14 and most of it is no longer true.**
Corrected at Tier 76, 2026-09-17, with what is left stated exactly:

    L2/InterfaceMotion  x9   the combustion front `a.b` (solution_dependent) and
                             the plume boundary `f.e`, `f.g` (prescribed).
                             STILL RESEARCH. Stage C. Untouched here, and it
                             should stay refusing -- the refusal is the product.
    L7/R9               x1   the time-integrated flux. NOT research any more:
                             CS-11 closed the bound and
                             `boundary_response_integrated` is an existing field.
    the 49 decerts           almost all MISSING DECLARATIONS, not missing
                             science -- and Tier 76 measured exactly how many.

**And the thing the old docstring did not say at all**, which mattered more than
anything it did: every agent's ``boundary_response`` was `linear_response` on a
seeded random matrix, so **no rocket physics had ever been run through Atlas.**

`rocket_experts.py` closes that for the b-c seam -- agent `c`, the airframe
shell (`thermostruct2d`), and agent `b`, the combustion chamber
(`compressible2d`), both build-repo solvers imported unmodified on the build
repo's own meshes.  The other five agents are still stubs and `build` says which
is which rather than leaving a reader to find out.

---

## The three declaration levels, and why the middle one is a warning

``declarations`` selects what the record claims:

``fixture``   as the graph was written before Tier 76.  Kept so the pre-Tier-76
              compile reproduces exactly.
``known``     every field that is a fact about the solver the agent STANDS FOR,
              read off the build repo's source: `time_discretization`,
              `stencil_radius`, `substeps_per_macro_step`, `elliptic_subsolve`,
              `lambda_ref`.  **These are true of the intended expert and
              unverifiable against the current one** -- W69's class exactly.
``real``      ``known`` for the five stub agents, plus real physics, real
              `storage` and real `validity` on `b` and `c`.  The default.

**The warning is the `known` level and it is worth stating out loud.**  It lifts
a large fraction of the decertifications on a graph whose agents are still
seeded random matrices, and nothing in the compiler can tell.  That is
`governing_family`'s promotion problem (W69) generalised: a declaration layer
improves the verdict without any physics arriving, so the verdict movement
between ``fixture`` and ``known`` is a measure of the paperwork and must never be
read as a measure of the model.

## Two declarations Tier 76 corrected as WRONG, not merely absent

1. **The clocks.**  This file declared ``dt_native`` 1e-5 / 1e-4 / 1e-3 with
   ``macro_dt`` 1e-3, a spread of 100:1.  ``config/atlas_0_1.yaml`` -- which
   says in its own header that it is the single source of truth, and which both
   plan documents quote -- declares 1e-3 / 5e-3 / 5e-2 with ``dt_macro`` 5e-2, a
   spread of **50:1**.  The YAML wins.
2. **The decomposition.**  This file declared `OVERLAPPING`.  The build repo's
   `geometry/domains.verify` asserts **at import** that every active cell centre
   lies in exactly one agent's domain (``np.all(hits == 1)``), and agent `g`
   carries the plume as a blanked hole precisely so that it does not overlap
   `f`.  The rocket is a **disjoint partition**, so it is `NON_OVERLAPPING` and
   there is no partition of unity to declare -- the substructuring branch, where
   W57 says no cut criterion exists.
"""

from __future__ import annotations

import numpy as np

from ..capability import (
    BCChannel,
    ClaimType,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    MotionClass,
    TimeDiscretization,
    linear_response,
    port_decl,
)
from ..graph import (
    Agent,
    CaseGraph,
    Connection,
    Decomposition,
    DeclaredTopologyEvent,
    FluxMatching,
    GlobalField,
)
from ..ports import PortType, ResponseHalf

M_EFF = 12

MECH_SCALES = {"stress": 1.0e6, "velocity": 3.0e3, "power_area": 3.0e9}
THERM_SCALES = {"temperature": 3.0e3, "entropy_flux": 2.0e3, "power_area": 6.0e6}
ADVEC_SCALES = {
    "enthalpy": 4.0e6, "mass_flux": 1.5e3, "power_area": 6.0e9,
    "h0_effort": 4.0e6, "h0_flow": 1.5e3, "h0_power": 6.0e9,
    "Yk_effort": 4.0e6, "Yk_flow": 1.5e3, "Yk_power": 6.0e9,
}

DECLARATION_LEVELS = ("fixture", "known", "real")

#: `config/atlas_0_1.yaml` ``dt_model`` and ``dt_macro``.  Hard-coded here rather
#: than read, because `atlas` must import without the build repo present; the
#: numbers are asserted against the YAML by
#: `tests/test_tier76_rocket_experts.py` whenever that checkout is available, so
#: a drift is a failing test rather than a silent disagreement.
CLOCKS_YAML = {"a": 1.0e-3, "b": 1.0e-3, "e": 1.0e-3,
               "d": 5.0e-3, "f": 5.0e-3, "g": 5.0e-3, "c": 5.0e-2}
MACRO_DT_YAML = 5.0e-2

#: The pre-Tier-76 clocks, kept so ``declarations="fixture"`` reproduces exactly.
CLOCKS_FIXTURE = {"a": 1.0e-5, "b": 1.0e-5, "e": 1.0e-5,
                  "d": 1.0e-4, "f": 1.0e-4, "g": 1.0e-4, "c": 1.0e-3}
MACRO_DT_FIXTURE = 1.0e-3

#: `data/generate.py` line 39: ``GAS_AGENTS = ("a", "b", "e", "d", "f", "g")``,
#: every one a `Compressible2D`; `c` alone is `ThermoStruct2D`.  So the two
#: `time_discretization` values below are read off that file, not chosen.
GAS_AGENTS = ("a", "b", "e", "d", "f", "g")

#: The chamber's measured CFL sub-step at the operating point, 2.99e-7 s
#: (`scripts/w300_rocket_bc_seam.py`, stage cadence).  ``substeps_per_macro_step``
#: is ``dt_native / dt_cfl`` and it is NOT 1: R10's halo is radius x substeps, so
#: declaring 1 would under-report the domain of dependence by three orders --
#: W46's failure mode with the declaration too SMALL rather than too large.
DT_CFL_GAS = 2.99e-7

#: **The cadence agent `b`'s real response marches when `build` is not told
#: otherwise, and the measurement that justifies it.**
#:
#: The YAML's own ``dt_model['b'] = 1e-3`` is ~3350 CFL sub-steps, about 100 s a
#: probe column and a quarter of an hour for one seam on this machine.  Measured
#: over a 100x cadence range (`scripts/w300_rocket_bc_seam.py`, stage cadence):
#:
#:     dt_gas     1e-6      1e-5      1e-4
#:     beta       0.149201  0.149619  0.149639   spread 1.003x
#:     substeps   36        306       3051       spread 84.75x
#:
#: The cost is in SUB-STEPS, not seconds, because the seconds do not hold still:
#: two runs of this identical sweep gave wall-clock ratios of 87.0 and 78.4 while
#: the sub-step ratio was 84.75 both times.
#:
#: So three orders of cadence buy 0.3% of beta and cost two orders of work.  The
#: reduced cadence is therefore a MEASURED trade rather than a convenience, and
#: ``build(gas_dt=None)`` takes the YAML's clock for anyone who wants to pay it.
#: ``dt_native`` follows whichever is chosen -- a record whose declared step is
#: not the step its callable takes is undetectable downstream.
PROBE_DT_GAS = 1.0e-5


#: **The governing families, read off `data/generate.py` rather than guessed.**
#: That file sets ``rxn = self.reaction if a == "a" else None`` -- so agent `a`
#: alone carries the progress-variable source and is the only REACTING agent.
#: `b`, `e` and `f` get ``gas_cfg`` (combustion products) and `d`, `g` get
#: ``air_cfg``, which are different PARAMETERS of the same equations, not
#: different equations.
#:
#: The fixture labelled `a`, `b` and `e` alike as reacting, which is wrong for
#: two of the three, and the error was INVISIBLE because it made a-b and e-b
#: agree.  Correcting only `b` -- which is what wiring its real solver does --
#: takes L1/E3 from 7 records to 13, because the disagreement it had been hiding
#: becomes visible before the matching correction to `e` arrives.  A verdict can
#: get worse on the way to being right, and that is worth one line in a
#: certificate rather than a silent simultaneous edit.
GOVERNING_FAMILY = {
    "a": "reacting-compressible-flow",
    "b": "compressible-navier-stokes-2d",
    "e": "compressible-navier-stokes-2d",
    "f": "compressible-navier-stokes-2d",
    "d": "compressible-navier-stokes-2d",
    "g": "compressible-navier-stokes-2d",
    "c": "parabolic-conduction-quasistatic-elasticity",
}


#: Which faces carry species as ADVEC passengers, from the case study's edge
#: table. Passengers are declared PER FACE, not per agent: the plume carries
#: species toward the combustion outflow and enthalpy only toward the wake, and
#: flattening that to one list per agent is what makes the two sides of a seam
#: disagree about the type.
SPECIES_FACES = {("a", "b"), ("b", "a"), ("b", "e"), ("e", "b"), ("e", "f"), ("f", "e")}


def _passengers_for(owner: str, face: str) -> tuple[str, ...]:
    return ("h0", "Yk") if (owner, face) in SPECIES_FACES else ("h0",)


def _ports(
    owner: str,
    faces: list[str],
    types: tuple[PortType, ...],
    motion: MotionClass = MotionClass.STATIC,
    resolution: int = M_EFF,
) -> list:
    out = []
    for f in faces:
        passengers = _passengers_for(owner, f)
        for t in types:
            scales = {
                PortType.MECH: MECH_SCALES,
                PortType.THERM: THERM_SCALES,
                PortType.ADVEC: ADVEC_SCALES,
            }[t]
            out.append(
                port_decl(
                    name=f"{f}:{t.value}",
                    port_type=t,
                    geometry=f,
                    direction=Direction.BIDIRECTIONAL,
                    nondim=dict(scales),
                    passengers=passengers if t is PortType.ADVEC else (),
                    effective_resolution=resolution,
                    motion_class=motion,
                    # W66 / L3-C9, declared on a fixture for W61's reason.
                    response_half=ResponseHalf.EFFORT,
                )
            )
    return out


def _responder(n: int, seed: int, gain: float = 1.0):
    """A declared boundary response and its exact JVP, as one consistent pair.

    A record that claims a JVP and cannot supply one is a contradiction the
    compiler catches at L1, so a fixture must not create one by accident.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.1
    A = gain * (A + A.T + 3.0 * np.eye(n))
    return linear_response(A), (lambda _p, _t, d: A @ np.asarray(d, dtype=float))


def _known_fields(agent_id: str, level: str) -> dict:
    """The fields that are facts about the solver the agent stands for.

    Empty at ``fixture``.  At ``known`` and above they are read off the build
    repo: `compressible2d` is MUSCL + minmod + HLLC under SSP-RK2 with NG = 2
    ghosts, an explicit finite-volume march with no solve of any kind;
    `thermostruct2d` is backward-Euler conduction over Q1 elements, which is one
    global solve per step and therefore EMBEDDED in R10's exact sense.
    """
    #: The fixture's own labels, kept so `fixture` reproduces exactly. Two of
    #: them are wrong -- see GOVERNING_FAMILY -- and that is the point of the
    #: level existing.
    fixture_family = {"a": "reacting-compressible-flow",
                      "b": "reacting-compressible-flow",
                      "e": "reacting-compressible-flow",
                      "d": "external-compressible-flow",
                      "f": "external-compressible-flow",
                      "g": "external-compressible-flow",
                      "c": "parabolic-conduction-quasistatic-elasticity"}
    if level == "fixture":
        return {"governing_family": fixture_family[agent_id]}
    dt = CLOCKS_YAML[agent_id]
    if agent_id in GAS_AGENTS:
        return {
            "governing_family": GOVERNING_FAMILY[agent_id],
            "time_discretization": TimeDiscretization.EXPLICIT,
            "elliptic_subsolve": EllipticSubsolve.NONE,
            "stencil_radius": 2,                       # NG = 2; MUSCL needs two
            "substeps_per_macro_step": max(1, int(np.ceil(dt / DT_CFL_GAS))),
            "lambda_ref": f"compressible2d.Compressible2D, agent {agent_id}'s own "
                          "block and discretization",
        }
    return {
        "governing_family": GOVERNING_FAMILY[agent_id],
        "time_discretization": TimeDiscretization.IMPLICIT,
        # Backward Euler solves (M/dt + K + K_robin) T over the whole panel.
        # Measured, not asserted: probe.support_reach reads fraction = 1.0000 and
        # consistent_with = ('embedded',) on this exact agent (W300, stage reach).
        "elliptic_subsolve": EllipticSubsolve.EMBEDDED,
        "stencil_radius": 1,                           # Q1 elements, one ring
        "substeps_per_macro_step": 1,
        "lambda_ref": "thermostruct2d.ThermoStruct2D, the airframe panel and its "
                      "own discretization",
    }


def reacting_flow(agent_id: str, faces: list[str], motion: MotionClass, seed: int,
                  level: str = "fixture", dt: float | None = None):
    """Reacting / internal compressible flow: agents a, interior b, e."""
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=_ports(agent_id, faces, (PortType.MECH, PortType.THERM, PortType.ADVEC), motion),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        differentiable=Differentiable.JVP,
        dt_native=CLOCKS_FIXTURE[agent_id] if dt is None else dt,
        L_native=0.2,
        storage=None,
        validity=None,
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        boundary_response=_responder(M_EFF, seed)[0],
        boundary_response_jvp=_responder(M_EFF, seed)[1],
        note=("trained from scratch; the fastest clock in the graph"
              if level == "fixture" else
              "STUB: a seeded random matrix, not physics. The fastest clock in the graph"),
        **_known_fields(agent_id, level),
    )


def thermal_structural(agent_id: str, faces: list[str], seed: int,
                       level: str = "fixture", dt: float | None = None):
    """Thermal-structural: agent c, the airframe. A DIFFERENT governing family."""
    known = _known_fields(agent_id, level)
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=_ports(agent_id, faces, (PortType.MECH, PortType.THERM)),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        differentiable=Differentiable.NONE,
        dt_native=CLOCKS_FIXTURE[agent_id] if dt is None else dt,
        L_native=1.0,
        storage=None,
        validity=None,
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        boundary_response=_responder(M_EFF, seed, gain=0.6)[0],
        note=("bootstrapped; parabolic conduction plus quasi-static elasticity"
              if level == "fixture" else
              "STUB: a seeded random matrix, not physics. Parabolic conduction "
              "plus quasi-static elasticity"),
        # at `fixture` there is no SeamReference and tau at b-c has no referent.
        **({"lambda_ref": None} if not known else known),
    )


def external_flow(agent_id: str, faces: list[str], motion: MotionClass, seed: int,
                  level: str = "fixture", dt: float | None = None):
    """External compressible flow and plume: agents d, f, g."""
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=_ports(agent_id, faces, (PortType.MECH, PortType.THERM, PortType.ADVEC), motion),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        differentiable=Differentiable.JVP,
        dt_native=CLOCKS_FIXTURE[agent_id] if dt is None else dt,
        L_native=2.0,
        storage=None,
        validity=None,
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        boundary_response=_responder(M_EFF, seed, gain=0.8)[0],
        boundary_response_jvp=_responder(M_EFF, seed, gain=0.8)[1],
        note=("" if level == "fixture"
              else "STUB: a seeded random matrix, not physics"),
        **_known_fields(agent_id, level),
    )


AGENTS = {
    "a": ("reacting", ["b"], MotionClass.SOLUTION_DEPENDENT),   # combustion front moves
    "b": ("reacting", ["a", "c", "e"], MotionClass.STATIC),
    "c": ("structural", ["b", "d"], MotionClass.STATIC),
    "d": ("external", ["c", "g"], MotionClass.STATIC),
    "e": ("reacting", ["b", "f"], MotionClass.STATIC),
    "f": ("external", ["e", "g"], MotionClass.PRESCRIBED),      # plume boundary moves
    "g": ("external", ["d", "f"], MotionClass.STATIC),
}

EDGES: list[tuple[str, str, str, tuple[PortType, ...]]] = [
    ("a-b", "a", "b", (PortType.MECH, PortType.THERM, PortType.ADVEC)),
    ("b-c", "b", "c", (PortType.MECH, PortType.THERM)),
    ("c-d", "c", "d", (PortType.MECH, PortType.THERM)),
    ("d-g", "d", "g", (PortType.MECH, PortType.THERM, PortType.ADVEC)),
    ("e-f", "e", "f", (PortType.MECH, PortType.THERM, PortType.ADVEC)),
    ("e-b", "e", "b", (PortType.MECH, PortType.THERM, PortType.ADVEC)),
    ("g-f", "g", "f", (PortType.MECH, PortType.THERM, PortType.ADVEC)),
]


def _dispatch(real_port: str, real_fn, stub_fn):
    """Route one port to the real expert and leave every other port on the stub.

    **A mixed record, labelled as one.**  Agent `b` has six ports and exactly one
    of them -- ``c:THERM`` -- is backed by `compressible2d`; the rest are still
    seeded random matrices of a different SHAPE (M_EFF against the wall's own
    cell count).  A reader of the certificate has to be able to tell which port
    is physics, so the dispatch is explicit rather than the real response being
    broadcast over faces it was never computed for.
    """
    def respond(port_name, trace, *a, **k):
        if port_name == real_port:
            return real_fn(port_name, trace, *a, **k)
        return stub_fn(port_name, trace, *a, **k)
    return respond


def _integrated_on(real_port: str, real_fn, agent_id: str):
    """The R9 integral, on the one port that has one.

    See `_dispatch`. There is no stub for this callable and there should not be:
    `boundary_response` at least has a seeded matrix of the right shape behind
    it, while an integral over a stub's sub-steps is not a quantity that exists.
    """
    def integrated(port_name, trace, n_substeps):
        if port_name != real_port:
            raise NotImplementedError(
                f"agent {agent_id!r}: only {real_port!r} is backed by a real expert, "
                f"so it is the only port with a time-integrated response; "
                f"{port_name!r} is still a seeded random matrix (Tier 76)"
            )
        return real_fn(port_name, trace, n_substeps)
    return integrated


def _real_bc_capabilities(agent_id: str, faces: list[str], level: str,
                          gas_dt: float | None):
    """Real `b` / `c` records, with the rocket's port names and face list kept.

    The b-c port is replaced wholesale -- declaration, prolongation, effective
    resolution and callables -- by `rocket_experts`' own, because its geometry is
    the real seam (14 airframe cells against 112 chamber cells) and the fixture's
    uniform ``M_EFF = 12`` is not attainable there.
    """
    from . import rocket_experts as RE

    if agent_id == "c":
        expert = RE.ShellAgent(dt=CLOCKS_YAML["c"])
        # h is measured at DT_H_MEASURE, not at the YAML's 1 ms clock: it moves
        # 2.5% over a 100x cadence range and the YAML clock is ~3350 CFL
        # sub-steps, about a minute a call. The spread is recorded beside it.
        gas = RE.ChamberGasAgent(dt=RE.DT_H_MEASURE)
        expert.march(RE.T_CHAMBER, RE.gas_h_on_shell_seam(gas, expert, RE.T_SHELL_COLD))
        caps = thermal_structural(agent_id, faces, 102, level=level,
                                  dt=CLOCKS_YAML[agent_id])
        real = RE.shell_capabilities(expert, m_eff=RE.bc_dim_M())
        caps.bc_channel = BCChannel.ROBIN        # `_robin` is a Robin condition
        caps.weight_hash = "thermostruct2d/1-backward-euler"
        real_port, other = "b:THERM", "c:THERM"
    else:
        expert = RE.ChamberGasAgent(dt=gas_dt or CLOCKS_YAML["b"])
        expert.T_wall_base = RE.T_SHELL_COLD
        caps = reacting_flow(agent_id, faces, AGENTS[agent_id][2], 101, level=level,
                             dt=expert.dt)
        # dt_native FOLLOWS the cadence the response actually marches. A record
        # whose declared step is not the step its callable takes is the kind of
        # inconsistency nothing downstream can detect, so the two move together
        # and the YAML's own clock is quoted in the note instead.
        caps.substeps_per_macro_step = max(1, int(np.ceil(expert.dt / DT_CFL_GAS)))
        real = RE.chamber_gas_capabilities(expert, m_eff=RE.bc_dim_M())
        caps.weight_hash = "compressible2d/2-hllc-minmod-ssprk2"
        caps.governing_family = "compressible-navier-stokes-2d"
        real_port, other = "c:THERM", "b:THERM"

    # swap the b-c THERM port for the real one, keeping every other face
    caps.ports = [real.ports[0] if p.name == real_port else p for p in caps.ports]
    stub = caps.boundary_response
    caps.boundary_response = _dispatch(real_port, real.boundary_response, stub)
    # The integrated response exists ONLY on the real port. Handing the real
    # one to every port would return a 14- or 112-vector where a stub port's
    # 12 is expected -- a shape error waiting for the first caller, and the
    # kind of latent mismatch that reads as physics when it finally fires. It
    # raises instead, naming the port, because `L7/R9/quadrature` already
    # reports the absence and a wrong answer is worse than a refusal.
    caps.boundary_response_integrated = _integrated_on(
        real_port, real.boundary_response_integrated, agent_id)
    caps.probe_base = lambda p, _e=expert, _r=real_port: (
        np.asarray(_e.base_trace(), dtype=float) if p == _r else np.zeros(M_EFF))
    caps.storage = expert.storage
    caps.validity = expert.validity
    # the JVP claim was true of the stub and is NOT true of the real expert:
    # neither build-repo solver is differentiable. Dropping the claim rather
    # than leaving a record that promises a JVP it cannot supply.
    caps.differentiable = Differentiable.NONE
    caps.boundary_response_jvp = None
    caps.note = (f"MIXED: {real_port} is REAL "
                 f"({'thermostruct2d' if agent_id == 'c' else 'compressible2d'}, "
                 f"build repo 0a407b7, unmodified, on grid.build_blocks' own mesh); "
                 f"every other face including {other} is still a seeded random matrix")
    del other
    return caps, expert


#: **W303, measured 2026-09-17 -- and declared ONLY where it was measured.**
#:
#: `assemble_seam` sums the two blocks unless a seam names the agent whose
#: outward normal both sides report against.  Undeclared through Tier 76, the
#: rocket summed every seam.  On b-c:THERM that is the wrong object, and the
#: evidence is a root rather than an argument: scanned over the admissible
#: interval [250, 2800] K, the SUM residual has **no sign change at all** while
#: the DIFFERENCE has one, at 1035.487 K.
#:
#: The sum cannot have a root there because `generate.py` hands the GAS's own
#: conduction-limited h to the SHELL's Robin channel, so both sides carry the
#: same film coefficient and the interface state cancels out of their sum to
#: first order.  Both sides return heat POSITIVE IN THE DIRECTION GAS -> SHELL,
#: which is +y at this wall, which is agent `b`'s own outward normal (its jmax
#: face normal is [0, +1]) and is against the shell's inner-face normal.
#:
#: **Which of the two agents is named is a convention with no physical content**
#: -- it flips a global sign on S, and the interface problem S lam = chi is
#: unchanged because chi flips with it.  What it does change is whether
#: `L4/E7/passivity`'s ``lambda_min > 0`` test passes, which is **W306**: that
#: test cannot distinguish a one-signed spectrum from a genuinely mixed one.
#:
#: The other six edges are NOT declared. Their sign structure has not been
#: measured, and a declaration copied across a graph on the strength of one
#: seam is the kind of defaulting this vault keeps finding.
EFFORT_NORMAL = {"b-c:THERM": "b"}


def build(with_staging: bool = False, declarations: str = "real",
          counting_control: bool = False,
          gas_dt: float | None = PROBE_DT_GAS) -> CaseGraph:
    """The rocket ascent as a case graph.

    ``with_staging``      adds the stage-separation event the 2D case study is
                          scoped to exclude.  Offered because it is the fourth
                          named slot, and a slot with no way to reach it is
                          untested.
    ``declarations``      ``fixture`` | ``known`` | ``real`` -- see the module
                          docstring.  ``real`` is the default from Tier 76.
    ``counting_control``  attaches trivially-true `storage` and `validity` to
                          ALL SEVEN agents.  **It is an instrument, not a
                          claim**: it exists so the number of decertification
                          records those two fields control can be counted
                          exactly, and it must never be used to report a verdict
                          about the rocket.  A stub's storage is fiction.
    """
    if declarations not in DECLARATION_LEVELS:
        raise ValueError(f"declarations must be one of {DECLARATION_LEVELS}, "
                         f"got {declarations!r}")
    makers = {
        "reacting": reacting_flow,
        "structural": lambda aid, faces, seed, level="fixture", dt=None:
            thermal_structural(aid, faces, seed, level=level, dt=dt),
        "external": external_flow,
    }
    clocks = CLOCKS_FIXTURE if declarations == "fixture" else CLOCKS_YAML
    real_ids = ("b", "c") if declarations == "real" else ()
    experts: dict = {}
    agents = []
    for i, (aid, (kind, faces, motion)) in enumerate(AGENTS.items()):
        if aid in real_ids:
            caps, expert = _real_bc_capabilities(aid, faces, declarations, gas_dt)
            experts[aid] = expert
        elif kind == "structural":
            caps = makers[kind](aid, faces, 100 + i, level=declarations,
                                dt=clocks[aid])
        else:
            caps = makers[kind](aid, faces, motion, 100 + i, level=declarations,
                                dt=clocks[aid])
        if counting_control:
            # trivially true, and that is the point: they measure the RULE, not
            # the rocket. `1.0` is not an energy and `True` is not a predicate.
            caps.storage = caps.storage or (lambda _s=None: 1.0)
            caps.validity = caps.validity or (lambda _s=None, _c=None: True)
        agents.append(Agent(aid, caps, domain=aid))

    connections = []
    for seam_id, a, b, types in EDGES:
        for t in types:
            connections.append(
                Connection(
                    seam_id=f"{seam_id}:{t.value}",
                    a=(a, f"{b}:{t.value}"),
                    b=(b, f"{a}:{t.value}"),
                    port_type=t,
                    geometrically_coincident=True,
                    derive_space=True,
                    # NOT at the `fixture` level: W189 compares that
                    # artifact byte for byte against a pre-Tier-76
                    # capture, and a Tier-77 convention must not reach
                    # back into it.
                    effort_normal=("" if declarations == "fixture"
                                   else EFFORT_NORMAL.get(f"{seam_id}:{t.value}", "")),
                )
            )

    events = []
    if with_staging:
        events.append(
            DeclaredTopologyEvent(
                event_id="stage-separation",
                t=120.0,
                graph_before=tuple(AGENTS),
                graph_after=tuple(k for k in AGENTS if k != "a"),
                state_map=None,     # declared without one, so it is refused
                ledger=None,        # and without the ledger the missing rule will constrain
            )
        )

    # The decomposition. `fixture` keeps the pre-Tier-76 declaration so that
    # compile reproduces; everything else states what the build repo asserts at
    # import -- a disjoint partition, hence the substructuring branch.
    decomp = (Decomposition.OVERLAPPING if declarations == "fixture"
              else Decomposition.NON_OVERLAPPING)
    macro = MACRO_DT_FIXTURE if declarations == "fixture" else MACRO_DT_YAML
    # 50:1 across the graph, so R9 wants the integral rather than the pointwise
    # flux. Declaring it is not enough on its own -- each side of every multirate
    # seam must also SUPPLY boundary_response_integrated, which only b and c do.
    flux = (FluxMatching.POINTWISE if declarations == "fixture"
            else FluxMatching.TIME_INTEGRATED)

    # **The name and the note below are the fixture's OWN, verbatim, at
    # `fixture`.**  `w189_artifact_control` compares this artifact byte for byte
    # against a capture taken before Tier 76, and the claim that the fixture
    # level "reproduces the pre-Tier-76 compile" is worth far more as a BYTE
    # identity than as a count of matching decisions -- so a level tag in the
    # name or the note, which would cost nothing to a reader, is not spent here.
    graph = CaseGraph(
        name="rocket-ascent-2d" + ("-with-staging" if with_staging else "")
             + ("" if declarations in ("fixture", "real") else f"-{declarations}"),
        agents=agents,
        connections=connections,
        decomposition=decomp,
        global_fields=[
            GlobalField(
                "gravity",
                # W117: external. Gravity is not produced by any agent in this
                # graph, so there is no seam being routed around it.
                produced_by=(),
                note="a global field bypasses the port mechanism entirely: it enters each "
                     "agent's update directly and contributes to the external power term "
                     "of the residual and to nothing else",
            )
        ],
        topology_events=events,
        macro_dt=macro,
        flux_matching=flux,
        note=("powered ascent; millisecond combustion against minute-scale ascent"
              if declarations == "fixture" else
              "powered ascent; millisecond combustion against minute-scale ascent. "
              f"declarations={declarations}"
              + (", b and c backed by real build-repo solvers" if real_ids else
                 ", every agent still a seeded random matrix")),
    )
    graph.rocket_experts = experts          # so a caller can reach the solvers
    return graph
