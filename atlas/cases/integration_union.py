"""W172 -- the joining seams: a real tiling, both circuits, and what joining costs.

Tier 39 counted the port algebra's promise over the union rung 9 would form --
`front_wing` (six exposed `WindowNS` windows, the plate structure, the
suspension), `cooling_loop` and `powertrain`: eighteen agents, five families,
forty ports each carrying its own prolongation, five scale sets, and **zero
per-pair declarations** against 153 pairs.  It counted the union's EXISTING
structure.  The seams that would join the three subsystems did not exist, so
nothing had measured what JOINING costs.  This module builds them.

**What joins what was checked against the port algebra before anything was
built, and two of the three candidates the brief named are not constructible.**

  ==========================  ===============================  ================
  candidate                   what the two sides declare       verdict
  ==========================  ===============================  ================
  aero tiling -- cooling,     `FSIFlowWindow` carries u and v  THERM not
  THERM (a radiator in the    and nothing else; its ports are  constructible:
  flow)                       the artificial faces, the        the tiling has
                              wetted surface and the mount,    no temperature
                              all MECH
  cooling -- powertrain,      BLOCK's dry face is a real       constructible
  THERM (waste heat)          Robin face of `step_thermal`;
                              the machine's I^2 R is computed
  powertrain -- structure,    STRUCT is clamped at its         ROT not
  ROT or MECH                 leading edge and has one port,   constructible
                              `wet:MECH`, which is connected   (no rotational
                                                               DOF); no second
                                                               loaded face
  ==========================  ===============================  ================

So the three joins, one per pair of subsystems, are:

  **J1 -- aero tiling to cooling loop, MECH.**  The radiator core sits in the
  flow as a porous plane on the x01 overlap: ``dp = 1/2 K u^2`` across it, the
  actuator disk's form with a declared loss coefficient, and the radiator's
  conductance scales with the air speed through it as ``(u/u_ref)^0.8``.  The
  replacement for THERM, recorded rather than slipped in.
  **J2 -- cooling loop to powertrain, THERM.**  The machine's heat path into the
  block.  The block's DRY face stops carrying a declared ~22.9 kK gas temperature
  behind an 8 W/(m^2 K) film -- `BlockAgent.t_source`, a stand-in for a source
  `step_thermal` cannot take -- and carries the motor-generator's casing instead,
  through a bolted-mount conductance.
  **J3 -- powertrain to aero tiling, MECH.**  `powertrain`'s rotor, whose two
  disk faces are the only OPEN ports in the three graphs, placed on the x10
  overlap behind the wing.  The replacement for "powertrain to structure".

A device in an overlapping tiling sits between two rings, `wake_array`'s
convention exactly: its ``up`` face meets the RIGHT window's ``xlo`` ring
(upstream of the plane) and its ``down`` face the LEFT window's ``xhi`` ring,
and the whole-face seam the two rings shared is split into a ``bypass`` seam and
the device's two.  That split re-declares an EXISTING seam, and it is counted.

Nothing here edits `front_wing`, `cooling_loop`, `powertrain` or `wake_array`.
Every subsystem is built by its own ``build()`` and every join is a subclass or a
re-declaration made in this module, so the disjoint union IS the three existing
graphs and a join's cost is visible as a difference against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from ..capability import Direction, MotionClass, port_decl
from ..graph import CaseGraph, Connection, Decomposition, FluxMatching
from ..ports import PortType, ResponseHalf
from ..transfer import InterfaceSpace, Prolongation
from . import cooling_loop as CL
from . import front_wing as FW
from . import ground_effect as G
from . import powertrain as PT
from . import wake_array as WA
from . import wing_fsi as W

__all__ = [
    "FLUID", "SOLID", "BLOCK_FAMILY", "COOLANT", "CIRCUIT", "SUBSYSTEMS",
    "DeviceSite", "ROTOR_SITE", "CORE_SITE", "JOINS", "JOIN_PAIRS", "JOIN_LEDGER",
    "ROTOR_PORT_FORMS", "SegmentedWindow", "CoreRadiator", "MountedBlock",
    "CooledMachine", "K_CORE", "UA_EXPONENT", "H_MOUNT", "ALPHA_CU", "T_REF_CU",
    "build", "ring_velocity", "calibrated_p_ref",
]

FLUID = "incompressible-navier-stokes-2d"
SOLID = "plane-stress-elasticity-2d"
BLOCK_FAMILY = "heat-conduction-2d"
COOLANT = "incompressible-thermal-transport-1d"
CIRCUIT = "lumped-dc-circuit"

SUBSYSTEMS = ("front_wing", "cooling_loop", "powertrain")
TILING = W.DEFAULT_TILING

#: Both devices span 32 cells of an 80-cell ring: the rotor's own span, which
#: `WA.RotorDisk` checks, and the core is given the same so the two MECH joins
#: are the same size and differ only in what is on the other side.
DEVICE_CELLS = WA.ROTOR_CELLS

OVER, NON = Decomposition.OVERLAPPING, Decomposition.NON_OVERLAPPING


@dataclass(frozen=True)
class DeviceSite:
    """Where a device sits: on the overlap of one tiling x-seam."""

    device: str          # the device's agent id
    segment: str         # the segment name it gives the two rings
    seam: str            # the tiling seam whose face it splits
    left: str            # window whose xhi ring is DOWNSTREAM of the plane
    right: str           # window whose xlo ring is UPSTREAM of the plane
    first_cell: int      # first local cell on the 80-cell ring

    @property
    def cells(self) -> np.ndarray:
        return np.arange(self.first_cell, self.first_cell + DEVICE_CELLS)


#: Row 0, on the [128, 144) overlap behind the plate (x 88-120), local cells
#: 12-44: clear of the plate's stamping box and of the y-overlap band [64, 80).
ROTOR_SITE = DeviceSite("ROTOR", "rotor", "x10", "F10", "F20", 12)
#: Row 1, on the [64, 80) overlap, local cells 32-64 (global y 96-128): above
#: the plate and clear of the y-overlap band.
CORE_SITE = DeviceSite("RAD", "core", "x01", "F01", "F11", 32)

JOINS = ("J1", "J2", "J3")
JOIN_PAIRS = {"J1": ("front_wing", "cooling_loop"),
              "J2": ("cooling_loop", "powertrain"),
              "J3": ("front_wing", "powertrain")}

#: How J3's rotor faces meet a host whose discretization is not the one they
#: were declared at.  ``host`` re-declares them at the host's cutoff and cell
#: measure; ``native`` leaves them as `wake_array` declared them (the control:
#: L3 has no space to derive); ``per-pair`` leaves the ports native and declares
#: the interface space and both prolongations on each CONNECTION, in the host's
#: measure; ``per-pair-naive`` does the same with each side in its OWN measure.
ROTOR_PORT_FORMS = ("host", "native", "per-pair", "per-pair-naive")


def ring_velocity(u_full: np.ndarray, site: DeviceSite) -> np.ndarray:
    """The streamwise velocity on the ring upstream of a device's plane."""
    k = TILING.names.index(site.right)
    ox, oy = TILING.offsets[k]
    return np.asarray(u_full, dtype=float)[oy + site.cells, ox].copy()


# ---------------------------------------------------------------------------
# the tiling side of J1 and J3 -- a ring split into a bypass and a device
# ---------------------------------------------------------------------------


@dataclass
class SegmentedWindow(FW.RidingFlowWindow):
    """`front_wing`'s window with a device plane bordering one of its rings.

    The bypass segment keeps the face's own PERTURBATION convention -- the trace
    is added to the ring, as `FSIFlowWindow` does for the whole face -- and the
    device segment is ABSOLUTE, because a device's response is a function of the
    inflow itself (the disk's and the core's are quadratic in it).  Two
    conventions on one ring, each matching the side it meets, which is
    `wake_array`'s arrangement.
    """

    #: face -> DeviceSite, for the faces a device plane borders.
    sites: dict = field(default_factory=dict)

    def segment_index(self, face: str, segment: str) -> np.ndarray:
        n_face = self.nx_c if face in ("ylo", "yhi") else self.ny_c
        site = self.sites.get(face)
        if site is None:
            if segment != "full":
                raise ValueError(f"{self.agent_id}.{face} is not segmented")
            return np.arange(n_face)
        if segment == site.segment:
            return site.cells
        if segment == "bypass":
            return np.setdiff1d(np.arange(n_face), site.cells)
        raise ValueError(f"{self.agent_id}.{face} has no segment {segment!r}")

    def probe_base(self, name: str) -> np.ndarray:
        parts = name.split(":")
        if len(parts) != 3:
            return super().probe_base(name)
        face, segment, _kind = parts
        idx = self.segment_index(face, segment)
        if segment == "bypass":
            return np.zeros(idx.size)
        ring, _interior = W._RING[face]
        w = self.u0 if W._FACE_GEOM[face][1] == "x" else self.v0
        return np.asarray(w[ring], dtype=float)[idx]

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        parts = port_name.split(":")
        if len(parts) != 3:
            return super().respond(port_name, trace)
        face, segment, kind = parts
        if kind != "MECH":
            raise ValueError(f"{self.agent_id} has no port kind {kind!r}")
        self.n_calls += 1
        idx = self.segment_index(face, segment)
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if trace.shape[0] != idx.size:
            raise ValueError(f"trace on {port_name} has {trace.shape[0]} cells, "
                             f"expected {idx.size}")
        ring, interior = W._RING[face]
        axis = W._FACE_GEOM[face][1]
        r_u, r_v = self.u0.copy(), self.v0.copy()
        target = r_u if axis == "x" else r_v
        col = np.array(target[ring], dtype=float)
        if segment == "bypass":
            col[idx] = col[idx] + trace
        else:
            col[idx] = trace
        target[ring] = col
        opt = dict(dtype=W.TORCH_DTYPE, device=self.device)
        u1, v1 = self._step(torch.as_tensor(r_u, **opt), torch.as_tensor(r_v, **opt))
        w = (u1 if axis == "x" else v1).detach().cpu().numpy()
        return (self.nu * (w[ring] - w[interior]) / self.h_solver)[idx]


def segmented_flow_ports(expert: SegmentedWindow, motion: MotionClass) -> list:
    """`front_wing`'s port list with each device-bordered face split in two."""
    out = []
    for p in FW.flow_ports(expert, motion):
        face = p.name.split(":", 1)[0]
        site = expert.sites.get(face)
        if site is None or p.name != f"{face}:MECH":
            out.append(p)
            continue
        for segment in ("bypass", site.segment):
            n_cells = int(expert.segment_index(face, segment).size)
            name = f"{face}:{segment}:MECH"
            out.append(port_decl(
                name=name, port_type=PortType.MECH,
                geometry=f"{face} ring of {expert.agent_id}, {segment} segment "
                         f"({n_cells} cells)",
                direction=Direction.BIDIRECTIONAL, nondim=dict(W.MECH_SCALES),
                effective_resolution=G.modes_for(n_cells),
                motion_class=MotionClass.STATIC,
                response_half=ResponseHalf.EFFORT,
                prolongation=W._prolongation(expert.agent_id, name, n_cells),
                note=("the face minus the device plane: a perturbation of the "
                      "ring, as the whole face was" if segment == "bypass" else
                      f"the ring under {site.device}'s plane: the ABSOLUTE inflow "
                      "the device's response is a function of")))
    return out


def front_wing_experts(u_full: np.ndarray, v_full: np.ndarray,
                       sites: tuple[DeviceSite, ...] = ()) -> dict[str, Any]:
    """`front_wing.make_experts`, with the windows a device borders segmented."""
    experts = FW.make_experts(u_full, v_full, TILING)
    by_window: dict[str, dict[str, DeviceSite]] = {}
    for s in sites:
        by_window.setdefault(s.left, {})["xhi"] = s
        by_window.setdefault(s.right, {})["xlo"] = s
    for name, faces in by_window.items():
        old = experts[name]
        experts[name] = SegmentedWindow(
            agent_id=name, u0=old.u0, v0=old.v0, shared_faces=old.shared_faces,
            has_wing=old.has_wing, ox=old.ox, oy=old.oy, delta=old.delta,
            nu=old.nu, flux_mode=old.flux_mode, ride_height=old.ride_height,
            sites=faces)
    return experts


# ---------------------------------------------------------------------------
# J1 -- the radiator core in the flow
# ---------------------------------------------------------------------------

#: The core's loss coefficient in dynamic pressures, ``dp = 1/2 rho K u^2``.
#: A DECLARED constant of the closure, in the range a louvred-fin automotive
#: core sits in, and not fitted to anything.
K_CORE = 1.2
#: The air-side film coefficient's dependence on the core velocity, the
#: turbulent Dittus-Boelter / Colburn exponent.  Declared, not fitted.
UA_EXPONENT = 0.8


@dataclass
class CoreRadiator(CL.RadiatorLeg):
    """`cooling_loop`'s radiator with its core in the flow.

    Two MECH faces returning ``+/- 1/2 K u^2`` -- the actuator disk's form with a
    loss coefficient instead of an induction, so zero fitted parameters -- and a
    conductance ``UA(u) = UA_RAD (u / u_ref)^0.8`` with ``u_ref`` the core's own
    inflow at the state it is built at.  At that state ``UA = UA_RAD`` exactly,
    so the loop's operating point does not move and what the join adds is the
    DEPENDENCE on the air.  The heat the core rejects still leaves to `T_AMB`:
    the air carries no temperature, which is why this bond is MECH.
    """

    u_air: Any = None
    k_core: float = K_CORE

    def __post_init__(self) -> None:
        u = np.ones(DEVICE_CELLS) if self.u_air is None else self.u_air
        u = np.asarray(u, dtype=float).reshape(-1)
        if u.size != DEVICE_CELLS:
            raise ValueError(f"core inflow has {u.size} cells, expected {DEVICE_CELLS}")
        self.u_air = u
        self.u_air_ref = float(np.mean(np.abs(u)))

    @property
    def ua(self) -> float:
        u = max(float(np.mean(np.abs(self.u_air))), 1.0e-9)
        return CL.UA_RAD * (u / max(self.u_air_ref, 1.0e-9)) ** UA_EXPONENT

    @property
    def decay(self) -> float:
        return math.exp(-self.ua / (CL.MDOT * CL.CP_COOLANT))

    def traction(self, u: np.ndarray) -> np.ndarray:
        return 0.5 * self.k_core * np.asarray(u, dtype=float) ** 2

    def respond_mech(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        # The core removes streamwise momentum: the sign is the outward
        # normal's, exactly as `RotorDisk.respond` has it.
        sign = -1.0 if port_name == "down:MECH" else +1.0
        return sign * self.traction(np.asarray(trace, dtype=float).reshape(-1))

    def base_for(self, port) -> np.ndarray:
        name = getattr(port, "name", str(port))
        if name.endswith(":MECH"):
            return self.u_air.copy()
        return super().base_for(port)


def core_capabilities(expert: CoreRadiator, caps):
    """RAD's record from `cooling_loop.build`, with the two faces in the flow."""
    mech = tuple(port_decl(
        name=name, port_type=PortType.MECH,
        geometry=f"{side} face of the radiator core in front_wing's flow "
                 f"({DEVICE_CELLS} cells at dx = {W.DX})",
        direction=Direction.BIDIRECTIONAL, nondim=dict(W.MECH_SCALES),
        effective_resolution=G.modes_for(DEVICE_CELLS),
        motion_class=MotionClass.STATIC, response_half=ResponseHalf.EFFORT,
        prolongation=W._prolongation(expert.agent_id, name, DEVICE_CELLS),
        note="the core's pressure drop 1/2 K u^2 against the absolute air "
             "velocity through it; the ring's cells, cutoff and cell measure "
             "are front_wing's, taken from the partner")
        for side, name in (("upstream", "up:MECH"), ("downstream", "down:MECH")))
    inner = caps.boundary_response

    def respond(port_name, trace):
        if port_name.endswith(":MECH"):
            return expert.respond_mech(port_name, trace)
        return inner(port_name, trace)

    return replace(caps, ports=list(caps.ports) + list(mech),
                   boundary_response=respond,
                   probe_base=lambda port, _e=expert: _e.base_for(port),
                   weight_hash=f"{caps.weight_hash}-core-K{expert.k_core:.6g}")


# ---------------------------------------------------------------------------
# J2 -- the machine's heat path into the block
# ---------------------------------------------------------------------------

#: Bolted casing-to-block contact with interface material, W/(m^2 K).  Declared.
H_MOUNT = 5000.0
#: Copper's temperature coefficient of resistance at its reference, 1/K.
ALPHA_CU = 3.93e-3
T_REF_CU = 293.15


@dataclass
class MountedBlock(CL.BlockAgent):
    """The block with the motor-generator bolted to its dry face.

    `BlockAgent` carries its heat source as a raised outer gas temperature,
    ``t_source`` -- about 22.9 kK behind an 8 W/(m^2 K) film, because
    `step_thermal` takes Robin data on two faces and no volumetric source.
    Joined, the outer face's Robin data is the machine's CASING temperature
    behind the mount conductance, so the source is a coupled quantity rather than
    a declared one.

    **This changes an existing port's response**, and that is part of the join's
    cost: the wetted face's response is one `step_thermal` whose outer datum is
    now the casing.  ``t_case`` is a state, initialised where the mount carries
    ``q_machine`` from the release wall temperature, which is the same
    declaration of the operating point `t_source` was.
    """

    h_mount: float = H_MOUNT
    q_machine: float = CL.Q_SOURCE
    t_case: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.t_case is None:
            tw = float(np.mean(self._face_T(self._T, "outer")))
            self.t_case = tw + self.q_machine / (self.h_mount * self.area)

    def _outer(self) -> tuple[float, np.ndarray]:
        return self.h_mount, np.full(CL.N_SEAM, float(self.t_case))

    def step(self, T_coolant: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = self.dt if dt is None else dt
        h_o, t_o = self._outer()
        self._T = self._ts.step_thermal(
            self._T, dt, self.h_wall, np.asarray(T_coolant, dtype=float), h_o, t_o,
            radiate=False, T_inf=CL.T_AMB)
        return self._T

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        if port_name == "outer:THERM":
            t_case = np.asarray(trace, dtype=float).reshape(CL.N_SEAM)
            t_cool = np.full(CL.N_SEAM, CL.T_COOLANT_0)
            T = self._ts.step_thermal(self._T, self.dt, self.h_wall, t_cool,
                                      self.h_mount, t_case, radiate=False,
                                      T_inf=CL.T_AMB)
            tw = self._face_T(T, "outer")
            q = self.h_mount * (t_case - tw)       # W/m^2 INTO the block
            return CL._as_entropy_flux(q, 0.5 * (t_case + tw))
        tc = np.asarray(trace, dtype=float).reshape(CL.N_SEAM)
        h_o, t_o = self._outer()
        T = self._ts.step_thermal(self._T, self.dt, self.h_wall, tc, h_o, t_o,
                                  radiate=False, T_inf=CL.T_AMB)
        tw = self._face_T(T, "inner")
        q = self.h_wall * (tw - tc)
        return CL._as_entropy_flux(q, 0.5 * (tw + tc))

    def heat_outer(self, T: np.ndarray | None = None) -> float:
        """Watts crossing the dry face into the block, through the mount."""
        T = self._T if T is None else T
        tw = self._face_T(T, "outer")
        return float(np.sum(self.h_mount * (float(self.t_case) - tw))
                     * CL.H_SEAM * CL.WIDTH)

    def energy_balance(self, T_coolant: np.ndarray, dt: float | None = None) -> dict:
        """`BlockAgent.energy_balance`'s first law, with the mount as the source."""
        dt = self.dt if dt is None else dt
        tc = np.asarray(T_coolant, dtype=float).reshape(CL.N_SEAM)
        T0 = self._T.copy()
        h_o, t_o = self._outer()
        T1 = self._ts.step_thermal(T0, dt, self.h_wall, tc, h_o, t_o,
                                   radiate=False, T_inf=CL.T_AMB)
        stored = CL.WIDTH * (self.thermal_energy(T1) - self.thermal_energy(T0)) / dt
        q_out = self.heat_outer(T1)
        q_wall = self.heat_in(tc, T1)
        resid = stored - (q_out - q_wall)
        scale = max(abs(q_out), abs(q_wall), 1.0e-30)
        return {"stored_rate": stored, "q_outer": q_out, "q_wall": q_wall,
                "residual": resid, "relative": abs(resid) / scale}

    def base_for(self, port) -> np.ndarray:
        name = getattr(port, "name", str(port))
        if name == "outer:THERM":
            return np.full(CL.N_SEAM, float(self.t_case))
        return self.base_trace()


def mounted_block_capabilities(expert: MountedBlock, caps):
    """BLOCK's record from `cooling_loop.build`, with the dry face as a port."""
    outer = port_decl(
        name="outer:THERM", port_type=PortType.THERM,
        geometry=f"dry face of the block where the machine casing is bolted, "
                 f"{CL.N_SEAM} line elements",
        direction=Direction.BIDIRECTIONAL, nondim=dict(CL.THERM_SCALES),
        effective_resolution=CL.M_EFF, motion_class=MotionClass.STATIC,
        response_half=ResponseHalf.FLOW,
        prolongation=CL.face_prolongation(expert.agent_id, "outer:THERM"),
        note="the mount conductance's Robin term, -k dT/dn = h_mount (T - T_case)")
    return replace(caps, ports=list(caps.ports) + [outer],
                   boundary_response=expert.respond,
                   probe_base=lambda port, _e=expert: _e.base_for(port),
                   weight_hash=f"{caps.weight_hash}-mounted-h{expert.h_mount:.6g}")


@dataclass
class CooledMachine(PT.MachineAgent):
    """The motor-generator with its casing bolted to the block.

    Its THERM response is its own dissipation, ``I^2 R(T)``, at the current its
    circuit carries at its declared shaft speed, through the mount area.  A
    lumped casing has one temperature, so the response is uniform over the face.

    ``R(T)`` is copper's, referenced at the casing temperature the join is built
    at: ``R(T) = R_MGU (1 + alpha_ref (T - T_case_ref))`` with ``alpha_ref`` the
    20 C coefficient carried to that temperature, which is exact for a linear
    ``R(T)``.  So at the build state the machine's resistance is `R_MGU` and the
    circuit's operating point -- and its ELEC ports' responses -- do not move.

    **One constant is a reconciliation, not a property of either side.**
    `powertrain` is nondimensional and the block works in watts.  ``p_ref`` is
    watts per nondimensional power unit, fixed ONCE at `powertrain`'s own
    reference operating point (``U_REF = 1``, ``omega = 15``), where it makes the
    machine dissipate the block's `Q_SOURCE` -- chosen by looking at one answer,
    and then held, because a unit does not move with the operating point.  When
    J3 moves the operating point the machine's heat moves with it and the block's
    ``q_machine`` is re-derived from it: a cost J3 lays on J2's agents.
    """

    current_op: float = 0.0
    p_ref: float = 1.0
    t_case_ref: float = 350.0
    mount_area: float = CL.L_Z * CL.WIDTH

    @property
    def alpha_ref(self) -> float:
        return ALPHA_CU / (1.0 + ALPHA_CU * (self.t_case_ref - T_REF_CU))

    def resistance_at(self, t_case: float) -> float:
        return self.resistance * (1.0 + self.alpha_ref * (float(t_case)
                                                          - self.t_case_ref))

    def heat_flux(self, t_case: float) -> float:
        """W/m^2 leaving the casing into the mount."""
        return (self.current_op ** 2 * self.resistance_at(t_case) * self.p_ref
                / self.mount_area)

    def respond_therm(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        t = np.asarray(trace, dtype=float).reshape(CL.N_SEAM)
        q = self.heat_flux(float(np.mean(t)))
        return CL._as_entropy_flux(np.full(CL.N_SEAM, q), t)

    def base_for(self, port) -> np.ndarray:
        name = getattr(port, "name", str(port))
        if name.endswith(":THERM"):
            return np.full(CL.N_SEAM, self.t_case_ref)
        return super().base_for(port)


def cooled_machine_capabilities(expert: CooledMachine, caps):
    """MGU's record from `powertrain.build`, with the casing as a THERM port."""
    therm = port_decl(
        name="case:THERM", port_type=PortType.THERM,
        geometry=f"the machine casing's mounting face, over the block's "
                 f"{CL.N_SEAM}-element dry face",
        direction=Direction.BIDIRECTIONAL, nondim=dict(CL.THERM_SCALES),
        effective_resolution=CL.M_EFF, motion_class=MotionClass.STATIC,
        response_half=ResponseHalf.FLOW,
        prolongation=CL.face_prolongation(expert.agent_id, "case:THERM"),
        note="I^2 R(T) through the mount area, as an entropy flux; the face's "
             "cells, cutoff and cell measure are the BLOCK's, taken from the "
             "partner")
    inner = caps.boundary_response

    def respond(port_name, trace):
        if port_name.endswith(":THERM"):
            return expert.respond_therm(port_name, trace)
        return inner(port_name, trace)

    return replace(caps, ports=list(caps.ports) + [therm],
                   boundary_response=respond,
                   probe_base=lambda port, _e=expert: _e.base_for(port),
                   weight_hash=f"{caps.weight_hash}-cooled-pref{expert.p_ref:.6g}")


def calibrated_p_ref(machine: PT.MachineAgent,
                     q_target: float = CL.Q_SOURCE) -> tuple[float, float]:
    """(p_ref, current): the power unit at which ``machine`` dissipates ``q_target``.

    Evaluated at the machine's own declared speed, so called on a fresh
    `MachineAgent` it is `powertrain`'s reference operating point and nothing a
    join moved.
    """
    current = machine.current_at(machine.omega)
    denom = current * current * machine.resistance
    if denom <= 0.0:
        raise ValueError("the machine carries no current at its declared speed, "
                         "so no power scale makes it a heat source")
    return q_target / denom, current


# ---------------------------------------------------------------------------
# J3 -- the rotor in the flow
# ---------------------------------------------------------------------------


def host_rotor_capabilities(expert: WA.RotorDisk, caps):
    """ROTOR's record from `powertrain.build`, disk faces re-declared for the host.

    `wake_array` declared these faces at ITS spectral cutoff (8 cells, so 9 modes
    on 32 cells) and ITS cell measure (1/32 D).  `front_wing`'s rings declare a
    4-cell cutoff (17 modes) at 1/64, so ``dim M = min_i m_i_eff`` is no longer a
    derivation.  A port's discretization is its host's, so the re-declaration is
    the host's cutoff and measure on the rotor's own 32 cells.
    """
    ports = []
    for p in caps.ports:
        if p.name in ("up:MECH", "down:MECH"):
            side = "upstream" if p.name.startswith("up") else "downstream"
            p = replace(
                p, effective_resolution=G.modes_for(DEVICE_CELLS),
                nondim=dict(W.MECH_SCALES),
                prolongation=W._prolongation(expert.agent_id, p.name, DEVICE_CELLS),
                geometry=f"{side} face of {expert.agent_id} in front_wing's flow "
                         f"({DEVICE_CELLS} cells at dx = {W.DX})",
                note="RE-DECLARED for the host (W172): wake_array declared this "
                     "face at its own 8-cell cutoff and 1/32 cell measure")
        ports.append(p)
    return replace(caps, ports=ports)


def _per_pair_transfer(seam_id: str, rotor_port: str, window: str,
                       window_port: str, naive: bool) -> dict[str, Any]:
    """The seam-level form: M and both prolongations declared on the connection.

    ``dim M`` is ``min(9, 17) = 9``, because `C2/C3/C6` refuses a space finer than
    the coarser side's declared resolution.  In the host measure both sides share
    one basis.  ``naive`` puts each side in its OWN measure -- the rotor's native
    prolongation beside the window's truncated to nine modes -- which is what
    copying each port's declaration onto the connection produces.
    """
    d = min(WA.modes_for(DEVICE_CELLS), G.modes_for(DEVICE_CELLS))
    host = dict(matrix=G.fourier_basis(DEVICE_CELLS, d, W.DX),
                gram_V=W.DX * np.eye(DEVICE_CELLS))
    rotor = (dict(matrix=WA.fourier_basis(DEVICE_CELLS, d, WA.DX),
                  gram_V=WA.DX * np.eye(DEVICE_CELLS)) if naive else host)
    return {
        "space": InterfaceSpace(seam_id=seam_id, dim=d,
                                note="declared per pair (W172): the rotor's native "
                                     "cutoff against the host's"),
        "prolongations": {
            "ROTOR": Prolongation(agent_id="ROTOR", port_name=rotor_port,
                                  label=("native measure" if naive else
                                         "host measure") + f", {d} modes", **rotor),
            window: Prolongation(agent_id=window, port_name=window_port,
                                 label=f"host measure, {d} modes", **host),
        },
    }


# ---------------------------------------------------------------------------
# the ledger: what the count cannot derive from a difference
# ---------------------------------------------------------------------------

#: WHY a declaration has the content it has.  Everything countable is derived by
#: `scripts/w172_joining_cost.py` from the graphs themselves.
JOIN_LEDGER: dict[str, dict[str, Any]] = {
    "J1": {
        "pair": JOIN_PAIRS["J1"], "bond": "MECH",
        "seams": ("J1_core_up", "J1_core_down"), "splits": (CORE_SITE.seam,),
        "brief_candidate": "THERM, a radiator in the flow",
        "why_replaced": "FSIFlowWindow carries u and v and no temperature, so no "
                        "THERM effort exists on the tiling side; MECH is the only "
                        "bond both sides can declare",
        "declared_constants": {"K_CORE": K_CORE, "UA_EXPONENT": UA_EXPONENT},
        "reconciliation_constants": {},
        "responses_changed": {
            "RAD": "gains a MECH response, and its coefficients now depend on the "
                   "air speed through UA(u); unchanged at the build state",
        },
        "state_read_from_partner": {"RAD": "u_air, the host ring's velocity"},
    },
    "J2": {
        "pair": JOIN_PAIRS["J2"], "bond": "THERM",
        "seams": ("J2_heat",), "splits": (),
        "brief_candidate": "THERM, waste heat",
        "why_replaced": None,
        "declared_constants": {"H_MOUNT": H_MOUNT, "ALPHA_CU": ALPHA_CU},
        "reconciliation_constants": {
            "p_ref": "watts per nondimensional power unit, fixed at powertrain's "
                     "own reference operating point (U_REF = 1, omega = 15) where "
                     "the machine dissipates Q_SOURCE; held, not re-fitted, when "
                     "the operating point moves"},
        "responses_changed": {
            "BLOCK": "the wetted face's outer Robin datum is the casing "
                     "temperature behind h_mount instead of t_source behind "
                     "H_OUTER",
            "MGU": "gains a THERM response I^2 R(T)",
        },
        "state_read_from_partner": {
            "MGU": "t_case_ref, the block's dry-face temperature plus the mount "
                   "drop",
            "BLOCK": "q_machine, the machine's I^2 R at the operating point the "
                     "union solves"},
    },
    "J3": {
        "pair": JOIN_PAIRS["J3"], "bond": "MECH",
        "seams": ("J3_rotor_up", "J3_rotor_down"), "splits": (ROTOR_SITE.seam,),
        "brief_candidate": "ROT or MECH, powertrain to the structure",
        "why_replaced": "STRUCT is clamped at its leading edge, so it has no "
                        "rotational degree of freedom for ROT, and its one port "
                        "(wet:MECH) is the aerodynamic seam; the rotor's two disk "
                        "faces are the only OPEN ports in the three graphs",
        "declared_constants": {},
        "reconciliation_constants": {},
        "responses_changed": {},
        "state_read_from_partner": {
            "ROTOR": "u_ref, the host ring's velocity, where powertrain declared "
                     "U_REF = 1",
            "MGU": "omega, re-solved by CircuitSolve at the host inflow, where "
                   "powertrain declared 15"},
    },
}


# ---------------------------------------------------------------------------
# the union
# ---------------------------------------------------------------------------


def _split(conns: list[Connection], site: DeviceSite) -> list[Connection]:
    out = []
    for c in conns:
        if c.seam_id != site.seam:
            out.append(c)
            continue
        out.append(Connection(
            seam_id=f"{c.seam_id}_bypass",
            a=(site.left, "xhi:bypass:MECH"), b=(site.right, "xlo:bypass:MECH"),
            port_type=PortType.MECH, derive_space=True,
            geometrically_coincident=False, expected_null_dim=0,
            note=f"the open flow beside {site.device}: {c.seam_id}'s face minus "
                 "the device plane"))
    return out


def _device_seams(site: DeviceSite, join: str, what: str,
                  up: dict | None = None, down: dict | None = None
                  ) -> list[Connection]:
    return [
        Connection(seam_id=f"{join}_{site.segment}_up", a=(site.device, "up:MECH"),
                   b=(site.right, f"xlo:{site.segment}:MECH"),
                   port_type=PortType.MECH, derive_space=True,
                   geometrically_coincident=False, expected_null_dim=0,
                   note=f"{join}: {what} reads its inflow from the ring upstream "
                        "of its plane", **(up or {})),
        Connection(seam_id=f"{join}_{site.segment}_down",
                   a=(site.device, "down:MECH"),
                   b=(site.left, f"xhi:{site.segment}:MECH"),
                   port_type=PortType.MECH, derive_space=True,
                   geometrically_coincident=False, expected_null_dim=0,
                   note=f"{join}: {what}'s momentum sink, imposed on the ring "
                        "downstream of its plane", **(down or {})),
    ]


def build(u_full: np.ndarray, v_full: np.ndarray,
          subsystems=SUBSYSTEMS, joins=JOINS, rotor_ports: str = "host",
          clocks: str = "native",
          flux_matching: FluxMatching = FluxMatching.POINTWISE,
          name: str | None = None) -> tuple[CaseGraph, dict[str, Any]]:
    """The union of ``subsystems`` with the listed ``joins`` built.

    ``joins=()`` is the DISJOINT union of the same subsystems, which is the
    control every join's cost is a difference against.  ``clocks="reconciled"``
    re-declares every agent's native step at the tiling's, which is one of the
    two ways to pay for three clocks; ``"native"`` keeps each subsystem's own and
    leaves `L7/R9` to say what that costs.
    """
    subsystems = tuple(s for s in SUBSYSTEMS if s in tuple(subsystems))
    joins = tuple(j for j in JOINS if j in tuple(joins))
    for j in joins:
        missing = [s for s in JOIN_PAIRS[j] if s not in subsystems]
        if missing:
            raise ValueError(f"{j} joins {JOIN_PAIRS[j]} and {missing} are not in "
                             "the union")
    if clocks not in ("native", "reconciled"):
        raise ValueError(f"clocks is 'native' or 'reconciled', not {clocks!r}")
    if rotor_ports not in ROTOR_PORT_FORMS:
        raise ValueError(f"rotor_ports is one of {ROTOR_PORT_FORMS}")
    parts: dict[str, CaseGraph] = {}
    experts: dict[str, Any] = {}
    info: dict[str, Any] = {"subsystems": subsystems, "joins": joins,
                            "rotor_ports": rotor_ports, "clocks": clocks,
                            "flux_matching": flux_matching.value}
    sites = tuple(s for j, s in (("J3", ROTOR_SITE), ("J1", CORE_SITE)) if j in joins)

    if "front_wing" in subsystems:
        fw = front_wing_experts(u_full, v_full, sites)
        g, _ = FW.build(u_full, v_full, motion=False, tiling=TILING,
                        measured=None, experts=fw)
        agents = []
        for a in g.agents:
            ex = fw.get(a.agent_id)
            if isinstance(ex, SegmentedWindow):
                a = replace(a, capabilities=replace(
                    a.capabilities,
                    ports=segmented_flow_ports(ex, MotionClass.STATIC)))
            agents.append(a)
        conns = list(g.connections)
        for s in sites:
            conns = _split(conns, s)
        parts["front_wing"] = replace(g, agents=agents, connections=conns)
        experts.update(fw)

    # The powertrain's operating point is settled before either circuit is
    # built, because two joins read it: J3 moves it (the rotor's inflow is the
    # host ring's, so the shaft speed is re-solved) and J2 turns its dissipation
    # into the block's heat source.
    pt: dict[str, Any] = {}
    q_machine = None
    if "powertrain" in subsystems:
        pt = dict(PT.make_elements())
        if "J3" in joins:
            u_rotor = ring_velocity(u_full, ROTOR_SITE)
            pt["ROTOR"] = WA.RotorDisk(agent_id="ROTOR", u_ref=u_rotor)
            op = PT.CircuitSolve(rotor=pt["ROTOR"], elements=pt,
                                 u_ref=float(np.mean(u_rotor))).solve()
            # `solve` sets MGU.omega to the shaft speed the host inflow gives
            info["operating_point"] = op.as_dict()
        else:
            pt["ROTOR"] = PT._rotor()
        info["mgu_omega"] = float(pt["MGU"].omega)
        info["rotor_u_ref_mean"] = float(np.mean(pt["ROTOR"].u_ref))
        if "J2" in joins:
            p_ref, _i_ref = calibrated_p_ref(PT.MachineAgent())
            current = pt["MGU"].current_at(pt["MGU"].omega)
            q_machine = current * current * pt["MGU"].resistance * p_ref
            info.update(p_ref=p_ref, current_op=current, q_machine=q_machine)

    block = None
    if "cooling_loop" in subsystems:
        dt_c = W.MACRO_DT if clocks == "reconciled" else CL.MACRO_DT
        cl = dict(CL.make_legs(4))
        if "J1" in joins:
            cl["RAD"] = CoreRadiator(u_air=ring_velocity(u_full, CORE_SITE))
        for leg in cl.values():
            leg.dt = dt_c
        block = (MountedBlock(dt=dt_c, q_machine=q_machine) if "J2" in joins
                 else CL.BlockAgent(dt=dt_c))
        cl["BLOCK"] = block
        info["rad_ua"] = -math.log(cl["RAD"].decay) * CL.MDOT * CL.CP_COOLANT
        info["block_outer_h"], info["block_outer_T"] = (
            (block.h_mount, float(block.t_case)) if isinstance(block, MountedBlock)
            else (CL.H_OUTER, float(block.t_source)))
        g, _ = CL.build(experts=cl, measured=None)
        agents = []
        for a in g.agents:
            if a.agent_id == "RAD" and "J1" in joins:
                a = replace(a, capabilities=core_capabilities(cl["RAD"],
                                                              a.capabilities))
            if a.agent_id == "BLOCK" and "J2" in joins:
                a = replace(a, capabilities=mounted_block_capabilities(
                    block, a.capabilities))
            agents.append(a)
        parts["cooling_loop"] = replace(g, agents=agents)
        experts.update(cl)

    if "powertrain" in subsystems:
        dt_p = W.MACRO_DT if clocks == "reconciled" else PT.MACRO_DT
        if "J2" in joins:
            base = pt["MGU"]
            pt["MGU"] = CooledMachine(current_op=info["current_op"],
                                      p_ref=info["p_ref"],
                                      t_case_ref=float(block.t_case),
                                      omega=base.omega)
            info["t_case_ref"] = float(block.t_case)
        for e in pt.values():
            e.dt = dt_p
        g, _ = PT.build(experts=pt, measured=None)
        agents = []
        for a in g.agents:
            if a.agent_id == "ROTOR" and "J3" in joins and rotor_ports == "host":
                a = replace(a, capabilities=host_rotor_capabilities(
                    pt["ROTOR"], a.capabilities))
            if a.agent_id == "MGU" and "J2" in joins:
                a = replace(a, capabilities=cooled_machine_capabilities(
                    pt["MGU"], a.capabilities))
            agents.append(a)
        parts["powertrain"] = replace(g, agents=agents)
        experts.update(pt)

    joining: list[Connection] = []
    if "J1" in joins:
        joining += _device_seams(CORE_SITE, "J1", "the radiator core")
    if "J2" in joins:
        joining.append(Connection(
            seam_id="J2_heat", a=("MGU", "case:THERM"), b=("BLOCK", "outer:THERM"),
            port_type=PortType.THERM, derive_space=True,
            geometrically_coincident=True, expected_null_dim=0,
            # W138: both sides return the heat crossing INTO THE BLOCK -- the
            # machine what it rejects, the block what it takes -- so the
            # condition is their difference, `cooling_loop`'s wall seam's
            # declaration on this seam's own physics.
            effort_normal="BLOCK",
            note="J2: the machine's I^2 R into the block's dry face through the "
                 "mount conductance"))
    if "J3" in joins:
        up = down = None
        if rotor_ports.startswith("per-pair"):
            naive = rotor_ports == "per-pair-naive"
            up = _per_pair_transfer("J3_rotor_up", "up:MECH", ROTOR_SITE.right,
                                    "xlo:rotor:MECH", naive)
            down = _per_pair_transfer("J3_rotor_down", "down:MECH",
                                      ROTOR_SITE.left, "xhi:rotor:MECH", naive)
        joining += _device_seams(ROTOR_SITE, "J3", "the rotor disk", up, down)

    agents = [a for p in parts.values() for a in p.agents]
    families = {a.agent_id: a.capabilities.governing_family for a in agents}
    has_fw = "front_wing" in subsystems
    graph_axis = OVER if has_fw else NON

    def axis(c: Connection):
        # The minimal declaration: only a same-family seam whose axis is NOT the
        # graph's needs to say so (W171).  Every other seam inherits.
        fa, fb = families[c.a[0]], families[c.b[0]]
        if fa != fb:
            return None
        own = OVER if fa == FLUID else NON
        return None if own is graph_axis else own

    conns = [replace(c, cut_axis=axis(c))
             for c in [c for p in parts.values() for c in p.connections] + joining]

    fw_graph = parts.get("front_wing")
    macro = (W.MACRO_DT if (has_fw or clocks == "reconciled")
             else max(p.macro_dt for p in parts.values()))
    graph = CaseGraph(
        name=name or ("w172-" + "+".join(subsystems) + "-"
                      + ("+".join(joins) if joins else "disjoint")
                      + ("" if rotor_ports == "host" else f"-rotor-{rotor_ports}")
                      + ("" if clocks == "native" else f"-clocks-{clocks}")
                      + ("" if flux_matching is FluxMatching.POINTWISE
                         else f"-{flux_matching.value}")),
        agents=agents,
        connections=conns,
        decomposition=graph_axis,
        overlap={FLUID: fw_graph.overlap} if has_fw else None,
        overlap_cells={FLUID: fw_graph.overlap_cells} if has_fw else None,
        partition_of_unity=({FLUID: fw_graph.partition_of_unity} if has_fw else None),
        global_fields=list(fw_graph.global_fields) if has_fw else [],
        cross_points=tuple(cp for p in parts.values() for cp in (p.cross_points or ())),
        loop_gains=tuple(lg for p in parts.values() for lg in p.loop_gains),
        macro_dt=macro,
        flux_matching=flux_matching,
        measured=None,
        note="W172: " + ("joined by " + ", ".join(joins) if joins
                         else "the disjoint union, the control every join is a "
                              "difference against"),
    )
    info["parts"] = {k: {"agents": [a.agent_id for a in p.agents],
                         "seams": [c.seam_id for c in p.connections]}
                     for k, p in parts.items()}
    return graph, {"experts": experts, "info": info, "parts": parts}
