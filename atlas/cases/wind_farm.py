"""Fixture: the wind farm, as built.

spec-wind-farm-wake-atlas-0.1 §3 and §5, run through the compiler.  Eight agents
forming a disjoint cover, fifteen typed edges, two open ROT ports.

This is a **fixture, not a target**.  It exists so the compiler has a concrete
graph to be tested against, and it is declarations only -- there is no coupling
code in this file, and there is none anywhere else either.

What it should produce, per end-to-end-architecture-spec §13.1:

    L1   bc_channel none, storage none, validity none  -> decertifications
    L3   refuse: no interface space or prolongation declared
    L4   Xi = 0 -> the word "coupled" is refused on the output
    L5   R6 refuses krylov and direct-schur; richardson at k = 1 survives
    L6   ||A|| never computed -> admit-uncertified, E6 unchecked
    L9   L unmeasured -> UNTYPED
    stamp  E1 holds, E2 holds, E3 holds, E4 holds, E5/E6/E7 unchecked

and the finding worth the whole exercise: the negative composed result was a
**compile-time refusal, not a measurement**.  The frozen checkpoint's composability
index is zero, which says the interface problem is *empty* rather than *hard*, and
it is decidable from one probe costing minutes.
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
    TimeDiscretization,
    linear_response,
    port_decl,
    zero_response,
)
from ..graph import Agent, CaseGraph, Connection, Decomposition, GlobalField
from ..ports import PortType, ResponseHalf

#: Interface resolution from the expert's own measured spectral cutoff: a 2D-long
#: tile seam, truncated where the expert stops representing flow, gives about 16
#: modes per component. The multiplier space and the probe basis are one space.
M_EFF = 16

#: Nondimensionalization: the wind farm's ruler. Length D, velocity U_inf,
#: density rho_inf. The scales below satisfy s_e * s_f = s_P by construction, so
#: the port's power bond survives the conversion.
RHO, U_INF = 1.225, 9.0
MECH_SCALES = {
    "stress": RHO * U_INF**2,
    "velocity": U_INF,
    "power_area": RHO * U_INF**3,
}
ADVEC_SCALES = {
    "enthalpy": U_INF**2,
    "mass_flux": RHO * U_INF,
    "power_area": RHO * U_INF**3,
    "h0_effort": U_INF**2,
    "h0_flow": RHO * U_INF,
    "h0_power": RHO * U_INF**3,
}
ROT_SCALES = {
    "torque": RHO * U_INF**2,
    "angular_velocity": U_INF,
    "power": RHO * U_INF**3,
}


def _fluid_ports(names: list[str]) -> list:
    ports = []
    for n in names:
        ports.append(
            port_decl(
                name=f"{n}:MECH",
                port_type=PortType.MECH,
                geometry=n,
                direction=Direction.BIDIRECTIONAL,
                nondim=dict(MECH_SCALES),
                effective_resolution=M_EFF,
                motion_class=MotionClass.STATIC,
                # W66 / L3-C9. Declared on a fixture for W61's reason: a field
                # left at its default is a value nobody chose, and the last time
                # that happened the fixture whose whole purpose was the rung-lift
                # chain was getting its lift from an omission.
                response_half=ResponseHalf.EFFORT,
            )
        )
        ports.append(
            port_decl(
                name=f"{n}:ADVEC",
                port_type=PortType.ADVEC,
                geometry=n,
                direction=Direction.BIDIRECTIONAL,
                nondim=dict(ADVEC_SCALES),
                passengers=("h0",),
                effective_resolution=M_EFF,
                motion_class=MotionClass.STATIC,
                response_half=ResponseHalf.EFFORT,
            )
        )
    return ports


def frozen_fluid(agent_id: str, faces: list[str]) -> ExpertCapabilities:
    """The frozen checkpoint. One field is the formal cause of everything below.

    ``bc_channel: none`` -- a field-in / field-out operator trained on periodic
    windows has nowhere to put a boundary ring at all.  Its response to imposed
    boundary data therefore does not depend on that data, and the probe measures
    exactly zero.  That is not a modelling choice made here; it is what
    ``zero_response`` implements, and the composability index falls out of it.
    """
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=_fluid_ports(faces),
        bc_channel=BCChannel.NONE,
        bc_time_varying=False,
        differentiable=Differentiable.NONE,
        dt_native=1.0e-2,
        L_native=2.0,
        regime_law=lambda T_s, nu_p, L: T_s / (nu_p * L**2),
        storage=None,
        equivariances=("mirror-y",),
        validity=None,
        governing_family="incompressible-navier-stokes-2d",
        lambda_ref=None,
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="frozen-windowns-checkpoint-2026",
        boundary_response=zero_response,
        reproducibility_floor=1.0e-6,
        deterministic=True,
        note="frozen periodic-window checkpoint; the composability index is zero",
    )


def boundary_capable_fluid(agent_id: str, faces: list[str]) -> ExpertCapabilities:
    """The same agent with a real boundary channel, for the contrast.

    A declared linear response stands in for a validated reference solver.  The
    probe cannot tell it from a neural expert: it hands over a trace and takes
    back a flux.
    """
    n = M_EFF
    rng = np.random.default_rng(7)
    A = rng.standard_normal((n, n)) * 0.05
    A = A + A.T + 2.0 * np.eye(n)      # symmetric positive definite: passive
    # Incompressibility constrains the interface trace to have zero net flux, so
    # the true operator is singular in exactly one direction. Projecting it out
    # here is what makes the fixture's declared expected_null_dim = 1 TRUE rather
    # than aspirational -- and the probe's null-space count is the free
    # correctness check that would catch it if it were not.
    e = np.ones(n) / np.sqrt(n)
    Pi = np.eye(n) - np.outer(e, e)
    A = Pi @ A @ Pi
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=_fluid_ports(faces),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # **Declared explicitly since W61, and it was load-bearing by omission
        # before that.** This fixture's whole purpose is the chain R2 lifts the
        # rung to probed-DtN -> the axis switches to non-overlapping -> the
        # cross-point difficulty appears. R2b gates that lift on the agent posing
        # a boundary-value problem over a macro-step, which is what an implicit
        # macro-step means -- so the fixture has to SAY so. It used to get the
        # lift from `time_discretization` defaulting to `unknown`, which W61
        # changed from "lift anyway and decertify" to "hold the rung down", and
        # that is the right default: an undecided premise is not a favourable one.
        time_discretization=TimeDiscretization.IMPLICIT,
        differentiable=Differentiable.JVP,
        dt_native=1.0e-2,
        L_native=2.0,
        regime_law=lambda T_s, nu_p, L: T_s / (nu_p * L**2),
        storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
        equivariances=("mirror-y",),
        validity=lambda state, cond=None: True,
        governing_family="incompressible-navier-stokes-2d",
        lambda_ref="classical-reference-windowns",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="boundary-capable-reference",
        # The zero-probe response is not zero: with no interface datum imposed a
        # real solver still returns the freestream flux, and subtracting that term
        # is what removes every trace-independent bias from the probe.
        boundary_response=linear_response(A, np.full(n, 0.05)),
        boundary_response_jvp=lambda _p, _t, d: A @ np.asarray(d, dtype=float),
        deterministic=True,
        note="accepts a Dirichlet ring, so R2 lifts the achievable rung to probed-DtN",
    )


def actuator_disk(agent_id: str) -> ExpertCapabilities:
    """Closed-form actuator disk: a LUMPED agent coupling to field agents.

    The interface is a field-to-lumped reduction, which is the case dim M = 1.
    The disk was never missing a check -- it was missing a declaration: nobody
    declared a prolongation, so nothing could say what the reduction had to be,
    and one was written by hand.  This fixture declares neither, so the compiler
    refuses for want of the declaration.
    """
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=[
            port_decl(
                name="face:MECH",
                port_type=PortType.MECH,
                geometry="rotor face",
                direction=Direction.BIDIRECTIONAL,
                nondim=dict(MECH_SCALES),
                effective_resolution=M_EFF,
                response_half=ResponseHalf.EFFORT,
            ),
            port_decl(
                name="face:ADVEC",
                port_type=PortType.ADVEC,
                geometry="rotor face",
                nondim=dict(ADVEC_SCALES),
                passengers=("h0",),
                effective_resolution=M_EFF,
                response_half=ResponseHalf.EFFORT,
            ),
            port_decl(
                name="shaft:ROT",
                port_type=PortType.ROT,
                geometry="shaft",
                direction=Direction.OUT,
                nondim=dict(ROT_SCALES),
                effective_resolution=1,
                response_half=ResponseHalf.EFFORT,
                note="unconnected: no drivetrain. The extracted power leaves the system "
                     "here, and under the port algebra that is an OPEN PORT with a "
                     "measurable power flow rather than an absence",
            ),
        ],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # W61: the rung is decided over ALL agents, so one agent leaving
        # `time_discretization` at its `unknown` default now holds the whole
        # graph's rung down. An actuator disk is an algebraic closure solved
        # together with the flow it closes, which is an implicit macro-step --
        # the same reason its `governing_family` is the flow's rather than one of
        # its own. Declared rather than defaulted, which is what W61 is about.
        time_discretization=TimeDiscretization.IMPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=1.0e-2,
        L_native=0.1,
        storage=None,
        validity=lambda state, cond=None: True,
        # E3 is decided by this string. The disk is an algebraic CLOSURE within the
        # incompressible problem, not a different continuum problem, so it declares
        # the same family and the seam is not a multiphysics seam. Declaring a
        # separate family here would fail E3 and mark tau UNDEFINED at every rotor
        # face -- which is why the string is load-bearing and worth being right about.
        governing_family="incompressible-navier-stokes-2d",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        boundary_response=linear_response(np.eye(M_EFF) * 0.4),
        note="closed form, zero parameters",
    )


#: The fifteen edges of the spec's §5 table, by the agents they join.
EDGES: list[tuple[str, str, str, PortType]] = [
    ("e1", "I", "R1", PortType.MECH),
    ("e1a", "I", "R1", PortType.ADVEC),
    ("e2", "I", "Bp", PortType.MECH),
    ("e3", "I", "Bm", PortType.MECH),
    ("e4", "R1", "N", PortType.MECH),
    ("e4a", "R1", "N", PortType.ADVEC),
    ("e5", "N", "F", PortType.MECH),
    ("e5a", "N", "F", PortType.ADVEC),
    ("e6", "F", "R2", PortType.MECH),
    ("e7", "R2", "W", PortType.MECH),
    ("e7a", "R2", "W", PortType.ADVEC),
    ("e8", "N", "Bp", PortType.MECH),
    ("e9", "N", "Bm", PortType.MECH),
    ("e10", "F", "Bp", PortType.MECH),
    ("e11", "W", "Bp", PortType.MECH),
]

_FACES = {
    "I": ["inlet", "R1", "Bp", "Bm"],
    "N": ["R1", "F", "Bp", "Bm"],
    "F": ["N", "R2", "Bp"],
    "W": ["R2", "Bp"],
    "Bp": ["I", "N", "F", "W"],
    "Bm": ["I", "N"],
}


def build(boundary_capable: bool = False, declare_transfer: bool = False) -> CaseGraph:
    """The wind farm as a case graph.

    ``boundary_capable`` swaps the frozen checkpoint for an expert with a real
    boundary channel -- the "wind farm, next" row of the coupling scheme's
    instantiation table.  ``declare_transfer`` opts the seams into having the
    compiler derive the interface space from the declared effective resolutions,
    which is what makes the graph admissible with no further declaration.
    """
    make_fluid = boundary_capable_fluid if boundary_capable else frozen_fluid
    fluid_ids = ["I", "N", "F", "W", "Bp", "Bm"]
    agents = [Agent(a, make_fluid(a, _FACES[a]), domain=a) for a in fluid_ids]
    agents += [Agent(r, actuator_disk(r), domain=r, role="rotor") for r in ("R1", "R2")]

    connections = []
    for seam_id, a, b, ptype in EDGES:
        connections.append(
            Connection(
                seam_id=seam_id,
                a=(a, _port_name(a, b, ptype)),
                b=(b, _port_name(b, a, ptype)),
                port_type=ptype,
                # The tiles coincide, which satisfies E2's coincidence half and
                # does NOT satisfy the amended connection rule.
                geometrically_coincident=True,
                derive_space=declare_transfer,
                # Incompressibility leaves a rank-one null space in the assembled
                # operator, and a measured null space of any other dimension
                # indicts the probe rather than the physics. It is a property of
                # the ASSEMBLED seam, not of one side: a lumped rotor responds in
                # the direction two fluids leave open, so a fluid-rotor seam is
                # not expected to be singular.
                expected_null_dim=(
                    1
                    if ptype is PortType.MECH and a not in ("R1", "R2") and b not in ("R1", "R2")
                    else None
                ),
            )
        )

    return CaseGraph(
        name="wind-farm-wake-2d" + ("-boundary-capable" if boundary_capable else "-as-built"),
        agents=agents,
        connections=connections,
        decomposition=Decomposition.OVERLAPPING,
        overlap=0.5,
        global_fields=[GlobalField("none", produced_by=(),
                                   note="isothermal, incompressible, no gravity term. "
                                        "W117: external by vacuity -- a declared absence")],
        macro_dt=1.0e-2,
        note="hub-height horizontal plane; two turbines at 7D spacing",
    )


def _port_name(owner: str, neighbour: str, ptype: PortType) -> str:
    """Resolve which declared port of ``owner`` faces ``neighbour``."""
    if owner in ("R1", "R2"):
        return "face:MECH" if ptype is PortType.MECH else "face:ADVEC"
    face = neighbour if neighbour in _FACES[owner] else _FACES[owner][0]
    return f"{face}:{'MECH' if ptype is PortType.MECH else 'ADVEC'}"
