"""The FOURTH real case study: the wind farm's own topology, with real experts.

`wind_farm.py` is a **fixture** -- eight agents, fifteen typed edges, two open
`ROT` ports, and every `boundary_response` a matrix somebody wrote down.  It has
been the only graph in this vault exercising `ADVEC` passengers, a `ROT` port, a
field-to-lumped seam, or more than one cross-point, and none of that has ever run
against a solver.  `gap-worklist` W5, W6 and the port algebra's harder cases are
all *fixture-tested only*, and [[tier0-measurements]] §9.8 and §11.8 both record
**"one topology"** as a standing caveat next to W55's "one expert".

This file keeps the fixture's **topology** exactly -- the same eight agents, the
same fifteen edges, the same port types, the same adjacency -- and replaces every
`boundary_response` with a real one:

    six fluid regions   `reference.WindowNS` windows, cut from the developed
                        state at six different places, so each carries its own
                        real flow rather than a shared stand-in
    two rotors          `disk.ActuatorDisk` -- algebraic, **zero fitted
                        parameters**, the same object the coupled system uses

**What is real and what is schematic, stated plainly.** The *experts* are real,
the *ports* are real, the *passengers* are real, and the *responses* are real
forward passes.  The *geometry* is schematic: the six regions are six windows of
one field rather than a partition of the wind farm's actual domain, because the
question this file exists to answer is about port types and junction structure,
not about wake physics.  A number measured here is a number about the port
algebra.  It is not a wind-farm result and must not be quoted as one.

The three things it exercises that nothing else has
---------------------------------------------------

**`ADVEC` with a real passenger, per face.**  `CASE-STUDY-GUIDE` warns that the
passenger list is per *face*, not per agent, and that a single per-agent list
cannot express an agent meeting one neighbour with `(h0, Y_k)` and another with
`(h0)`.  Here `h0` is computed from the window's own state -- specific total
enthalpy, `|u|^2 / 2` for an incompressible flow whose pressure the composition
layer holds -- and the flux is the advected one, `(u.n) h0`.

**A `ROT` port that is genuinely open.**  Each rotor's shaft carries extracted
power out of the system with nothing on the other side.  Under the port algebra
that is an open port with a measurable power flow rather than an absence, and the
`ActuatorDisk` gives a real torque and a real omega for it.

**A cross-point count that is not what it looked like.**  This section began as
*"six cross-points instead of one"* and the number was written before it was
derived.  Derived from the adjacency (`_cross_points` below), the fifteen edges
contain exactly **one** 3-clique -- ``{N, F, Bp}`` -- so the wind farm's own
topology has the **same** cross-point count as `window_ns`'s four-window tiling,
not more.  The interesting structure here is the *port types*, not the junction
count, and saying so is worth more than the sentence it replaces.

What it found on its first compile
----------------------------------

**A port whose trace and flux are not power-conjugate, and nothing checks
(W66).**  The first version of this file returned ``(u.n) h0`` from its `ADVEC`
port -- the enthalpy *power* -- against a normal-velocity trace.  E7 then failed
with a passivity defect of ``3.915e-02`` on the **assembled** seam ``e5a``, and
the number was real: the pairing was (velocity x power), which is not a power, so
the passivity eigenvalue was not measuring dissipation.

`check_scales` validates the declared scale *set* -- `s_e * s_f = s_P` on the
`nondim` dictionary -- and **nothing anywhere checks that `boundary_response`
returns the declared flow variable**.  For `MECH` the trace-flux pair lines up
with the scale set by convention and the gap is invisible.  For `ADVEC`, whose
conjugate pair is (specific total enthalpy, mass flux), it is not, and the whole
port algebra passed a record whose callable returned the wrong quantity.

This is §2.2 / W47 arriving on a new port type.  There, a *transport* term in the
MECH flux convention produced an apparent passivity defect of 2.39 on a block
that was passive, and the answer was to pin the diffusive form as normative.  An
`ADVEC` port has no diffusive form to fall back on -- transport is the whole of
what it carries -- so the only defence is the conjugate pairing, and the pairing
is the thing nothing validates.

**R10's plausibility check misfires on an algebraic closure (W65).**  The rotors
declare `elliptic_subsolve = none`, which is **true** -- an `ActuatorDisk` is a
closed-form zero-parameter law with no solve of any kind -- and they declare
`governing_family = "incompressible-navier-stokes-2d"`, which they **must**, or
E3 fails at every rotor face and tau goes `UNDEFINED` there.  R10's guard
challenges exactly that combination: *"an incompressible solver almost always
contains a pressure solve"*.  It is right about solvers and wrong about closures,
and the guide's own instruction to declare the flow's family is what walks an
agent into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    port_decl,
)
from ..graph import Agent, CaseGraph, Connection, Decomposition, GlobalField
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation
from .window_ns import (
    H,
    M_EFF,
    MACRO_DT,
    MONO_N,
    NU,
    U_INF,
    fourier_basis,
    load_reference,
)

#: The fixture's own topology, imported rather than copied, so the two cannot
#: drift apart.  If `wind_farm.py`'s adjacency changes, this graph changes with
#: it -- which is the point: it is the *same* topology with different experts.
from .wind_farm import EDGES, _FACES, _port_name           # noqa: E402, PLC2701

MECH_SCALES = {"stress": U_INF**2, "velocity": U_INF, "power_area": U_INF**3}
ADVEC_SCALES = {
    "enthalpy": U_INF**2,
    "mass_flux": U_INF,
    "power_area": U_INF**3,
    "h0_effort": U_INF**2,
    "h0_flow": U_INF,
    "h0_power": U_INF**3,
}
ROT_SCALES = {"torque": U_INF**2, "angular_velocity": U_INF, "power": U_INF**3}

FLUID_IDS = ("I", "N", "F", "W", "Bp", "Bm")
ROTOR_IDS = ("R1", "R2")

#: Window size for the six fluid regions.  Smaller than `window_ns`'s 138 so six
#: of them fit inside the 255-cell developed state at distinct, non-identical
#: locations -- each region must carry a *different* flow or the six agents are
#: one agent six times and the topology proves nothing.
REGION_N = 96

#: Where each region's window is cut from, in global cells.  Laid out along the
#: streamwise direction so `I` sits upstream of `R1`, `N` downstream of it, and
#: the two bypass regions off the centreline, which is the fixture's own picture.
REGION_ORIGIN = {
    "I":  (80, 0),
    "N":  (80, 100),
    "F":  (80, 155),
    "W":  (80, 159),
    "Bp": (155, 80),
    "Bm": (4, 80),
}

STENCIL_RADIUS = 2
SUBSTEPS = 10

_FACE_GEOM = {"xlo": (-1.0, "x"), "xhi": (+1.0, "x"),
              "ylo": (-1.0, "y"), "yhi": (+1.0, "y")}

#: Which of the window's four physical faces each logical neighbour is reached
#: through.  A logical face name in `_FACES` is a neighbour id; the probe needs a
#: geometric ring, so the two are mapped here rather than assumed to coincide.
_GEOM_FACE = ("xlo", "xhi", "ylo", "yhi")


def _geom_face_for(agent_id: str, logical: str) -> str:
    """Deterministic assignment of a logical neighbour to a physical ring.

    Arbitrary but **stable and injective per agent**, which is all the port
    algebra needs: what matters is that two faces of one agent are two different
    rings, so an `ADVEC` passenger list attached to one is not attached to the
    other. That is the distinction `CASE-STUDY-GUIDE` says a per-agent list
    cannot express, and it is what this graph is here to exercise.
    """
    faces = _FACES[agent_id]
    return _GEOM_FACE[faces.index(logical) % len(_GEOM_FACE)]


# ---------------------------------------------------------------------------
# the fluid agent: a real window, with MECH and ADVEC on each face
# ---------------------------------------------------------------------------


@dataclass
class RegionAgent:
    """One `reference.WindowNS` window standing for one region of the farm.

    Carries **two ports per face** where the topology asks for both a `MECH` and
    an `ADVEC` edge, and the `ADVEC` passenger list is attached to the *face*,
    not to the agent.
    """

    agent_id: str
    u0: np.ndarray
    v0: np.ndarray
    logical_faces: tuple[str, ...]
    dt: float = MACRO_DT
    nu: float = NU
    expose_elliptic: bool = False
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.u0 = np.asarray(self.u0, dtype=float)
        self.v0 = np.asarray(self.v0, dtype=float)
        self.n = int(self.u0.shape[0])
        self.h = H
        from .window_ns import _no_projection_class          # noqa: PLC0415
        cls = (_no_projection_class() if self.expose_elliptic
               else load_reference().WindowNS)
        self._solver = cls(nu=self.nu, length=self.n * H, n=self.n, cfl=0.4,
                           transmission="dirichlet")

    # -- geometry ---------------------------------------------------------

    def _ring_index(self, face: str):
        return {
            "xlo": ((slice(None), 0), (slice(None), 1)),
            "xhi": ((slice(None), -1), (slice(None), -2)),
            "ylo": ((0, slice(None)), (1, slice(None))),
            "yhi": ((-1, slice(None)), (-2, slice(None))),
        }[face]

    @staticmethod
    def parse(port_name: str) -> tuple[str, str]:
        """``'<geom>:<TYPE>'`` -> (geometric face, port type)."""
        face, kind = port_name.split(":", 1)
        return face, kind

    # -- the boundary channel ---------------------------------------------

    def _step_with(self, face: str, trace: np.ndarray):
        r_u, r_v = self.u0.copy(), self.v0.copy()
        ring, _ = self._ring_index(face)
        if _FACE_GEOM[face][1] == "x":
            r_u[ring] = r_u[ring] + trace
        else:
            r_v[ring] = r_v[ring] + trace
        u1, v1 = self._solver.step_batch(self.u0[None], self.v0[None], self.dt,
                                         bc0=(r_u[None], r_v[None]), bc1=None)
        self.n_calls += 1
        return u1[0], v1[0]

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(port, trace) -> flux, for both port types this agent carries.

        `MECH` returns the Steklov-Poincare flux `nu du/dn`, §2.2's normative
        form, exactly as `window_ns` does.

        `ADVEC` returns the **advected passenger flux** `(u.n) h0` with
        `h0 = |u|^2 / 2`: the specific total enthalpy an incompressible window
        carries, its pressure part being the composition layer's under R10. It is
        a genuinely different functional of the same solve, which is what makes
        the two ports two ports rather than one printed twice.
        """
        face, kind = self.parse(port_name)
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if trace.shape[0] != self.n:
            raise ValueError(
                f"trace on {port_name} has length {trace.shape[0]}, expected {self.n}")
        u1, v1 = self._step_with(face, trace)
        ring, interior = self._ring_index(face)
        sign, axis = _FACE_GEOM[face]
        w = u1 if axis == "x" else v1
        if kind == "MECH":
            return self.nu * (w[ring] - w[interior]) / self.h
        if kind == "ADVEC":
            # **The EFFORT, not the power (W66).** The declared scale set is
            # `h0_effort: U^2`, `h0_flow: U`, `h0_power: U^3`, so the conjugate
            # pair on this port is (specific total enthalpy, mass flux) and their
            # product is the power. The trace imposed here is a normal-velocity
            # perturbation -- the FLOW -- so the response must be the EFFORT for
            # the pairing to be a power at all.
            #
            # Returning `(u.n) h0` instead, which is the power itself, was the
            # first thing this file did, and it made E7 fail on seam `e5a` with a
            # passivity defect of 3.915e-02. **Nothing caught it**: `check_scales`
            # validates the declared scale SET (`s_e * s_f = s_P`) and nothing
            # anywhere checks that `boundary_response` returns the declared flow
            # variable. For MECH the two line up by accident of convention; for
            # ADVEC they do not.
            h0 = 0.5 * (u1**2 + v1**2)
            return h0[ring]
        raise ValueError(f"{self.agent_id} has no port kind {kind!r}")

    def storage(self, u: Any = None, v: Any = None) -> float:
        if u is None:
            u, v = self.u0, self.v0
        u = np.asarray(u, dtype=float)
        v = self.v0 if v is None else np.asarray(v, dtype=float)
        return 0.5 * float(np.sum(u * u + v * v)) * self.h**2

    def validity(self, state: Any = None, cond: Any = None) -> bool:
        u, v = (self.u0, self.v0) if state is None else state
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            return False
        return bool(self.h * float(np.max(np.hypot(u, v))) / self.nu <= 8.0)


# ---------------------------------------------------------------------------
# the rotor agent: a real ActuatorDisk
# ---------------------------------------------------------------------------


@dataclass
class RotorAgent:
    """`disk.ActuatorDisk`, exposed as a boundary response on two port types.

    Algebraic and stateless -- `spec` §8.2 -- so its response is a derivative of
    a closed form rather than of a solve, and it is exact to machine precision at
    any probe step.

    **`MECH`** returns the streamwise momentum sink the disk applies for the
    inflow it is shown: perturb the face velocity, get back the change in the
    force density. **`ROT`** returns the shaft torque, and its port is
    **unconnected** -- extracted power leaving the system through an open port
    with a measurable flow, which is what the port algebra says an absent
    drivetrain is.
    """

    agent_id: str
    u_ref: float
    m_eff: int = M_EFF
    n_cells: int = REGION_N
    dt: float = MACRO_DT
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        disk = load_reference()
        import importlib

        mod = importlib.import_module("atlas_windfarm_reference.disk")
        self._disk = mod.ActuatorDisk()
        self._state = self._disk(self.u_ref)

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        face, kind = port_name.split(":", 1)
        trace = np.asarray(trace, dtype=float).reshape(-1)
        base = self._disk(self.u_ref)
        self.n_calls += 1
        if kind == "MECH":
            # thrust per unit area, as a function of the perturbed local inflow
            out = np.empty(trace.shape[0])
            for i, dv in enumerate(trace):
                st = self._disk(self.u_ref + float(dv))
                out[i] = self._disk.force_density(st.thrust)
            return out - self._disk.force_density(base.thrust)
        if kind == "ADVEC":
            out = np.empty(trace.shape[0])
            for i, dv in enumerate(trace):
                st = self._disk(self.u_ref + float(dv))
                out[i] = st.power
            return out - base.power
        if kind == "ROT":
            # one scalar: the shaft torque. The port's effective resolution is 1.
            st = self._disk(self.u_ref + float(np.mean(trace)))
            return np.array([st.torque - base.torque])
        raise ValueError(f"{self.agent_id} has no port kind {kind!r}")

    def storage(self, u: Any = None, v: Any = None) -> float:
        """Kinetic energy of the disk-averaged flow it sees. Real, and small."""
        return 0.5 * float(self.u_ref**2)

    def validity(self, state: Any = None, cond: Any = None) -> bool:
        """`clamp_induction` is the expert's own envelope, and it is declared.

        The disk clamps its induction factor rather than extrapolating, and
        reports that it did. A clamped state is outside the regime the closed
        form is valid in, so the predicate reports it rather than hiding it --
        which is the difference between `validity` and a finiteness check.
        """
        return not bool(self._state.clamped)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def _prolongation(agent_id, port_name, n_cells, m=M_EFF):
    return Prolongation(
        agent_id=agent_id, port_name=port_name,
        matrix=fourier_basis(n_cells, m),
        gram_V=H * np.eye(n_cells),
        label=f"{m}-mode real Fourier basis (probed-dtn-coupling 2.2)",
    )


def _edges_of(agent_id):
    """{(neighbour, port_type)} for every edge this agent is an endpoint of."""
    out = set()
    for _sid, a, b, ptype in EDGES:
        if a == agent_id:
            out.add((b, ptype))
        if b == agent_id:
            out.add((a, ptype))
    return out


def region_capabilities(expert: RegionAgent) -> ExpertCapabilities:
    """One record per fluid region, with `ADVEC` passengers attached PER FACE."""
    ports = []
    for neighbour, ptype in sorted(_edges_of(expert.agent_id),
                                   key=lambda x: (x[0], x[1].value)):
        gf = _geom_face_for(expert.agent_id, neighbour)
        name = f"{gf}:{ptype.value.upper()}"
        if any(p.name == name for p in ports):
            continue
        kw = {}
        if ptype is PortType.ADVEC:
            # The passenger list belongs to THIS face. `I->R1` carries the
            # enthalpy into the rotor and `N->F` carries it downstream; they are
            # separate declarations on separate rings even when one agent owns
            # both, which is the thing a per-agent list cannot express.
            kw["passengers"] = ("h0",)
        ports.append(port_decl(
            name=name, port_type=ptype,
            geometry=f"{gf} face of {expert.agent_id} (towards {neighbour})",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(ADVEC_SCALES if ptype is PortType.ADVEC else MECH_SCALES),
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            # W66 / L3-C9. MECH returns nu du/dn and ADVEC returns h0: BOTH are
            # the EFFORT. The ADVEC one is the fix section 12.3 made by hand and
            # this field is what now carries it as a declaration.
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, name, expert.n),
            **kw,
        ))
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=ports,
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        elliptic_subsolve=(EllipticSubsolve.EXPOSED if expert.expose_elliptic
                           else EllipticSubsolve.EMBEDDED),
        stencil_radius=STENCIL_RADIUS,
        substeps_per_macro_step=SUBSTEPS,
        time_discretization=TimeDiscretization.EXPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=expert.n * H,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        governing_family="incompressible-navier-stokes-2d",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"windowns-region-{expert.agent_id}-n{expert.n}",
        boundary_response=expert.respond,
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="reference.WindowNS window standing for one region; MECH and ADVEC",
    )


def rotor_capabilities(expert: RotorAgent) -> ExpertCapabilities:
    """The rotor's record, including the OPEN `ROT` port."""
    ports = []
    for neighbour, ptype in sorted(_edges_of(expert.agent_id),
                                   key=lambda x: (x[0], x[1].value)):
        name = f"face:{ptype.value.upper()}"
        if any(p.name == name for p in ports):
            continue
        kw = {"passengers": ("h0",)} if ptype is PortType.ADVEC else {}
        ports.append(port_decl(
            name=name, port_type=ptype,
            geometry=f"rotor face of {expert.agent_id}",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(ADVEC_SCALES if ptype is PortType.ADVEC else MECH_SCALES),
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, name, expert.n_cells),
            **kw,
        ))
    ports.append(port_decl(
        name="shaft:ROT", port_type=PortType.ROT, geometry="shaft",
        direction=Direction.OUT, nondim=dict(ROT_SCALES),
        effective_resolution=1,
        # The torque, the ROT effort. Unconnected, so C9 never runs on it --
        # declared anyway, because an open port is still a port.
        response_half=ResponseHalf.EFFORT,
        note="unconnected: no drivetrain. The extracted power leaves the system "
             "here, and under the port algebra that is an OPEN PORT with a "
             "measurable power flow rather than an absence",
    ))
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=ports,
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # A closed-form algebraic closure contains no elliptic solve of its own.
        # Unlike the checkpoint (W60) this is KNOWN rather than undeclarable: the
        # disk is `spec` §8.2's zero-parameter formula and it is written down.
        elliptic_subsolve=EllipticSubsolve.NONE,
        stencil_radius=0,
        substeps_per_macro_step=1,
        time_discretization=TimeDiscretization.IMPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=1.0,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        # The SAME family as the flow it closes -- `wind_farm.py` has a comment on
        # exactly this, put there after getting it wrong once. An algebraic
        # closure WITHIN a continuum problem is not a different continuum problem,
        # and declaring otherwise fails E3 and marks tau UNDEFINED at every rotor.
        governing_family="incompressible-navier-stokes-2d",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"actuatordisk-a{expert._disk.a:.6g}",
        boundary_response=expert.respond,
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="disk.ActuatorDisk, closed form, zero fitted parameters",
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def make_experts(u_full, v_full, dt=MACRO_DT, nu=NU, expose_elliptic=False):
    out: dict[str, Any] = {}
    for rid in FLUID_IDS:
        oy, ox = REGION_ORIGIN[rid]
        out[rid] = RegionAgent(
            agent_id=rid,
            u0=u_full[oy:oy + REGION_N, ox:ox + REGION_N],
            v0=v_full[oy:oy + REGION_N, ox:ox + REGION_N],
            logical_faces=tuple(_FACES[rid]), dt=dt, nu=nu,
            expose_elliptic=expose_elliptic)
    # Each rotor sees the disk-averaged inflow of the region upstream of it.
    for rid, upstream in (("R1", "I"), ("R2", "F")):
        u_disk = float(np.mean(out[upstream].u0))
        out[rid] = RotorAgent(agent_id=rid, u_ref=u_disk, dt=dt)
    return out


def build(u_full: np.ndarray, v_full: np.ndarray, mode: str = "split-step",
          dt: float = MACRO_DT, nu: float = NU, experts=None, measured=None):
    """The eight-agent wind-farm topology, backed by real experts.

    ``measured`` defaults to None: none of these constants has been measured for
    this graph, and borrowing `window_ns`'s would be W56 on purpose.
    """
    if mode not in ("as-built", "split-step"):
        raise ValueError(mode)
    experts = experts or make_experts(u_full, v_full, dt, nu,
                                      expose_elliptic=(mode == "split-step"))
    agents = [Agent(a, region_capabilities(experts[a]), domain=a) for a in FLUID_IDS]
    agents += [Agent(r, rotor_capabilities(experts[r]), domain=r, role="rotor")
               for r in ROTOR_IDS]

    connections = []
    for seam_id, a, b, ptype in EDGES:
        pa = (f"face:{ptype.value.upper()}" if a in ROTOR_IDS
              else f"{_geom_face_for(a, b)}:{ptype.value.upper()}")
        pb = (f"face:{ptype.value.upper()}" if b in ROTOR_IDS
              else f"{_geom_face_for(b, a)}:{ptype.value.upper()}")
        connections.append(Connection(
            seam_id=seam_id, a=(a, pa), b=(b, pb), port_type=ptype,
            geometrically_coincident=True, derive_space=True,
            # n_0(Gamma) is a property of the SEAM. A fluid-fluid MECH seam under
            # an exposed elliptic part is 0 (section 8.4); a field-to-lumped seam
            # is 0 because a rotor responds in exactly the direction
            # incompressibility leaves open (`CASE-STUDY-GUIDE`, mistake 3).
            # ADVEC seams are left None: nothing here has reasoned one through,
            # and a disabled check is honest where a guessed one is not.
            expected_null_dim=(0 if ptype is PortType.MECH else None),
            note=f"{ptype.value} edge of the spec's section 5 table, real experts",
        ))

    return CaseGraph(
        name=f"wind-farm-real-8-agent-{mode}",
        agents=agents,
        connections=connections,
        decomposition=Decomposition.OVERLAPPING,
        overlap=21 * H, overlap_cells=21,
        global_fields=[GlobalField(
            "pressure",
            # **W117.** Produced by the six fluid regions and applied to exactly
            # those six. It is NOT applied to the rotors: a rotor expert has no
            # pressure input and no pressure state -- it reads a velocity and
            # returns a traction -- so the rotor coupling runs through the declared
            # MECH seams and not through this field. Leaving applies_to empty
            # (= every agent) would have declared the opposite.
            produced_by=FLUID_IDS,
            applies_to=FLUID_IDS,
            note="global under mode='split-step'; per-region under 'as-built', "
                 "which R10 refuses. The point of this graph is the port algebra "
                 "and the elliptic question is window_ns's, so both are offered")],
        cross_points=tuple(_cross_points()),
        macro_dt=dt,
        measured=measured,
        note=("the fixture's topology with real experts: six reference.WindowNS "
              "windows and two disk.ActuatorDisk rotors, 15 typed edges, two "
              "open ROT ports. Geometry schematic, ports and experts real"),
    )


def _cross_points():
    """Vertices where three or more regions meet, from the adjacency alone.

    Derived rather than declared, because a hand-written list is exactly the
    thing L2's cross-point check exists to not trust -- and because the count is
    what this graph has that `window_ns` does not.
    """
    adj: dict[str, set[str]] = {a: set() for a in FLUID_IDS + ROTOR_IDS}
    for _sid, a, b, _pt in EDGES:
        adj[a].add(b)
        adj[b].add(a)
    tri = set()
    ids = sorted(adj)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if b not in adj[a]:
                continue
            for c in sorted(adj[a] & adj[b]):
                tri.add(frozenset((a, b, c)))
    return sorted("-".join(sorted(t)) for t in tri)
