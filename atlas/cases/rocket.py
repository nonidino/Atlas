"""Fixture: the 2D rocket ascent, as specified.

case-study-rocket-ascent-2d-atlas-0.1: seven agents, seven typed edges, no ROT or
ELEC ports anywhere -- which is a correct statement about a solid-propellant
vehicle.

A **fixture, not a target**.  end-to-end-architecture-spec §13.2 runs this case
through the compiler on paper and finds that **it does not compile**, tripping
three of the five named slots in its declared scope with the fourth arriving with
staging:

    L1   different governing families at the fluid-structure seam b-c
    L2   the combustion front and the plume boundary are not static
         -> InterfaceMotion -> refuse
    L3   the rigid-body trajectory expert is lumped, coupling to field agents
         -> the field-to-lumped case, dim M = 1
    L4   E3 fails at b-c -> tau UNDEFINED -> admit-uncertified.
         The RUNG CHOICE survives: probing never mentions a governing equation
    L7   differing native steps -> multirate -> R9 -> refuse unless the flux is
         matched time-integrated
    L9   UNTYPED
    later  staging trips E1 -> TopologyEvent -> refuse without a ledger

The ordering this implies is that the rocket should not be the next build: every
refusal above is a genuine missing rule rather than missing code, and three of
them are research.
"""

from __future__ import annotations

import numpy as np

from ..capability import (
    BCChannel,
    ClaimType,
    Differentiable,
    Direction,
    ExpertCapabilities,
    MotionClass,
    linear_response,
    port_decl,
)
from ..graph import (
    Agent,
    CaseGraph,
    Connection,
    Decomposition,
    DeclaredTopologyEvent,
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


def reacting_flow(agent_id: str, faces: list[str], motion: MotionClass, seed: int):
    """Reacting / internal compressible flow: agents a, interior b, e."""
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=_ports(agent_id, faces, (PortType.MECH, PortType.THERM, PortType.ADVEC), motion),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        differentiable=Differentiable.JVP,
        dt_native=1.0e-5,             # millisecond-scale combustion
        L_native=0.2,
        storage=None,
        validity=None,
        governing_family="reacting-compressible-flow",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        boundary_response=_responder(M_EFF, seed)[0],
        boundary_response_jvp=_responder(M_EFF, seed)[1],
        note="trained from scratch; the fastest clock in the graph",
    )


def thermal_structural(agent_id: str, faces: list[str], seed: int):
    """Thermal-structural: agent c, the airframe. A DIFFERENT governing family."""
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=_ports(agent_id, faces, (PortType.MECH, PortType.THERM)),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        differentiable=Differentiable.NONE,
        dt_native=1.0e-3,
        L_native=1.0,
        storage=None,
        validity=None,
        governing_family="parabolic-conduction-quasistatic-elasticity",
        lambda_ref=None,              # no SeamReference: tau at b-c has no referent
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        boundary_response=_responder(M_EFF, seed, gain=0.6)[0],
        note="bootstrapped; parabolic conduction plus quasi-static elasticity",
    )


def external_flow(agent_id: str, faces: list[str], motion: MotionClass, seed: int):
    """External compressible flow and plume: agents d, f, g."""
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=_ports(agent_id, faces, (PortType.MECH, PortType.THERM, PortType.ADVEC), motion),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        differentiable=Differentiable.JVP,
        dt_native=1.0e-4,
        L_native=2.0,
        storage=None,
        validity=None,
        governing_family="external-compressible-flow",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        boundary_response=_responder(M_EFF, seed, gain=0.8)[0],
        boundary_response_jvp=_responder(M_EFF, seed, gain=0.8)[1],
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


def build(with_staging: bool = False) -> CaseGraph:
    """The rocket ascent as a case graph.

    ``with_staging`` adds the stage-separation event the 2D case study is scoped
    to exclude.  It is offered because it is the fourth named slot, and because a
    slot with no way to reach it is untested.
    """
    makers = {
        "reacting": reacting_flow,
        "structural": lambda aid, faces, seed: thermal_structural(aid, faces, seed),
        "external": external_flow,
    }
    agents = []
    for i, (aid, (kind, faces, motion)) in enumerate(AGENTS.items()):
        if kind == "structural":
            caps = makers[kind](aid, faces, 100 + i)
        else:
            caps = makers[kind](aid, faces, motion, 100 + i)
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

    return CaseGraph(
        name="rocket-ascent-2d" + ("-with-staging" if with_staging else ""),
        agents=agents,
        connections=connections,
        decomposition=Decomposition.OVERLAPPING,
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
        macro_dt=1.0e-3,
        note="powered ascent; millisecond combustion against minute-scale ascent",
    )
