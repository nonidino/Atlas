"""Tier 49 -- rung 9: the union MARCHED, and the seven decisions a march needs.

Tier 46 built the joined union (`integration_union`) and compiled it.  Tier 47
held rung 9's gate against it and restated the gate, measuring each join's
receiving balance **at the release state, one step each**.  Nothing had ever
marched it, and `rung9-gate-restated` section 6 lists seven decisions a march
needs first.  This module makes them, and marches.

The seven, and where each one lives here:

  ===  =========================================  ==========================
  #    decision (worklist row)                    where
  ===  =========================================  ==========================
  1    time and length units (W201)               `VehicleScale`, `VEHICLE`
  2    a rotor sized for its host, and a machine  `HOST_ROTOR_WIDTH`,
       sized for that rotor (W199)                `machine_for_rotor`,
                                                  `SizedCircuitSolve`
  3    dissipation and domain-boundary power      `receiver_balances`, and
       (W200)                                     the named hole in its
                                                  docstring
  4    the down faces' orientation (W197)         `rotor_power_paths` --
                                                  measured, not repaired
  5    the devices applied as body forces (W94)   `DeviceForcing`
  6    the shared operating point (W196)          `UnionRollout.refresh`,
                                                  `join_coupling`
  7    `L7/R9` scoped to the multirate seams      `atlas/compiler.py`, and
       (W194)                                     `multirate_seams` here
  ===  =========================================  ==========================

**Decisions 1 and 2 are not framework work.**  They are the choice of WHICH
VEHICLE this is -- its speed, its front wing's chord, its rotor -- and none of
the three case studies was written at one physical scale.  They are made here,
in one place, as declarations rather than as constants buried in the march, with
what it would have cost to choose otherwise recorded beside them
(`VehicleScale.alternatives`).  They are flagged so they can be overruled
cheaply: change `VEHICLE` and `HOST_ROTOR_WIDTH` and every number downstream
moves with them.

Nothing here edits `front_wing`, `cooling_loop`, `powertrain`, `wake_array` or
`integration_union`.  The rollout SUBCLASSES `front_wing.FrontWingRollout` and
inherits its four composition-layer operators -- cut, blend, project, band -- so
that the union's fluid is the same column CS-12 and PoC 2 published and not a
second copy of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from . import cooling_loop as CL
from . import front_wing as FW
from . import integration_union as IU
from . import powertrain as PT
from . import wake_array as WA
from . import wing_fsi as W

__all__ = [
    "VehicleScale", "VEHICLE", "CLOCKS_NATIVE", "clock_table", "clock_ratio",
    "HOST_ROTOR_WIDTH", "ROTOR_SCALE", "machine_for_rotor", "SizedCircuitSolve",
    "operating_point", "DeviceSpec", "ROTOR_DEVICE", "CORE_DEVICE", "DEVICES",
    "DeviceForcing", "UnionRollout", "UnionMarch", "march_union",
    "LoopMarch", "march_loop", "replay_coolant", "receiver_balances",
    "rotor_power_paths",
    "multirate_seams", "CROSSING_KEYS", "crossing_vector", "composition_error",
    "JOIN_OF_DEVICE", "N_FLUID_PER_COOLANT", "GATE", "PREDICTION",
]


# ===========================================================================
# DECISION 1 -- the vehicle's units (W201)
# ===========================================================================


@dataclass(frozen=True)
class VehicleScale:
    """The two numbers that turn three unrelated unit systems into one.

    `rung9-gate-restated` section 3: the union's three clocks are ``0.0125`` in
    the tiling's convective units, ``0.05`` s, and ``0.2`` in `wake_array`'s
    rotor-diameter units, and `L7/R9` compares them as BARE NUMBERS.  Tier 46's
    "reconciled clocks" therefore admitted equal NUMBERS and not equal TIMES.
    Relating them needs a length and a speed, and choosing those is choosing
    which vehicle this is.

    **The choice, stated once.**

      ``l0_m``   the tiling's length unit -- 64 cells -- in metres.  The plate
                 spans 32 cells, so the front wing's chord is ``l0_m / 2``.
                 0.50 m puts the chord at 0.25 m and the domain at
                 1.625 m x 1.125 m, which is a front-wing-scale box.
      ``u0_ms``  the tiling's speed unit, ``U_inf = 1``, in m/s.  50 m/s is
                 180 km/h: a medium-speed corner, where a front wing is
                 working and the car is not at its straight-line ride height.

    Everything else follows: the time unit is ``l0_m / u0_ms``, and
    `wake_array`'s length unit is the rotor's diameter, which decision 2 sizes
    to its host face at ``l0_m / 2``.

    **What the declaration does NOT buy: dynamic similarity.**  The tiling
    marches at ``nu = 4e-3`` in its own units, so its Reynolds number is
    ``U L / nu = 250``.  Air at 50 m/s over a 0.5 m length is at
    ``1.7e6``.  Fixing lengths, times and speeds does not fix ``Re``, and
    matching it would need ``nu = 6e-7`` in code units -- a grid this
    laptop cannot march.  So: **kinematic similarity yes, dynamic similarity
    no**, declared rather than implied, and every second quoted downstream is a
    second of a 2-D laminar model of a vehicle, not of the vehicle.
    """

    l0_m: float = 0.50
    u0_ms: float = 50.0
    #: air's kinematic viscosity at 25 C, for the Reynolds honesty above only.
    nu_air_si: float = 1.55e-5

    @property
    def t0_s(self) -> float:
        """The tiling's time unit in seconds."""
        return self.l0_m / self.u0_ms

    @property
    def chord_m(self) -> float:
        """The plate spans 32 cells of 64 to the length unit."""
        return 0.5 * self.l0_m

    @property
    def cell_m(self) -> float:
        return W.DX * self.l0_m

    @property
    def domain_m(self) -> tuple[float, float]:
        t = W.DEFAULT_TILING
        return (t.nx * self.cell_m, t.ny * self.cell_m)

    @property
    def reynolds_model(self) -> float:
        """What the tiling actually marches at, in its own units."""
        return W.U_INF * 1.0 / W.NU

    @property
    def reynolds_vehicle(self) -> float:
        return self.u0_ms * self.l0_m / self.nu_air_si

    def seconds(self, t_units: float, system: str) -> float:
        """A bare number in one subsystem's clock, in seconds.

        ``system`` is ``"front_wing"`` (the tiling's convective units),
        ``"powertrain"`` (`wake_array`'s: the rotor diameter over the same
        freestream) or ``"cooling_loop"`` (SI already).
        """
        if system == "front_wing":
            return float(t_units) * self.t0_s
        if system == "powertrain":
            return float(t_units) * (self.rotor_diameter_m / self.u0_ms)
        if system == "cooling_loop":
            return float(t_units)
        raise ValueError(f"no unit system {system!r}")

    def metres(self, l_units: float, system: str) -> float:
        if system == "front_wing":
            return float(l_units) * self.l0_m
        if system == "powertrain":
            return float(l_units) * self.rotor_diameter_m
        if system == "cooling_loop":
            return float(l_units)
        raise ValueError(f"no unit system {system!r}")

    @property
    def rotor_diameter_m(self) -> float:
        """Decision 2's rotor, in metres: its host face is 32 cells."""
        return HOST_ROTOR_WIDTH * self.l0_m

    def alternatives(self) -> dict[str, Any]:
        """What choosing otherwise would have cost -- recorded, not argued.

        The only thing downstream that the pair ``(l0_m, u0_ms)`` moves is the
        RATIO of the coolant circuit's clock to the tiling's, because the
        circuit is the one subsystem already written in seconds:

            ratio = dt_cool / (dt_tiling * l0 / u0) = 4 * u0 / l0

        over a plausible F1 box -- chord 0.125 m to 0.5 m, so ``l0`` in
        [0.25, 1.0] m, and 30 m/s to 90 m/s -- the ratio runs from 120 to 1440
        and is never below 100.  **The choice moves the number and not the
        conclusion**, which is what makes it safe to make here rather than ask.
        """
        box_l = (0.25, 1.0)
        box_u = (30.0, 90.0)
        corners = {}
        for lo in box_l:
            for uo in box_u:
                v = VehicleScale(l0_m=lo, u0_ms=uo)
                corners[f"l0={lo} m, u0={uo} m/s"] = {
                    "t0_s": v.t0_s,
                    "chord_m": v.chord_m,
                    "coolant_over_fluid_clock": clock_ratio(v),
                }
        rs = [c["coolant_over_fluid_clock"] for c in corners.values()]
        return {
            "box": {"l0_m": list(box_l), "u0_ms": list(box_u)},
            "corners": corners,
            "ratio_min": min(rs), "ratio_max": max(rs),
            "ratio_never_below_100": bool(min(rs) >= 100.0),
            "chosen": {"l0_m": VEHICLE.l0_m, "u0_ms": VEHICLE.u0_ms,
                       "coolant_over_fluid_clock": clock_ratio(VEHICLE)},
        }


#: **The declaration.**  Overrule it here and every number downstream moves.
VEHICLE = VehicleScale()

#: Each subsystem's native macro-step, as the bare number its own module states,
#: beside the unit system that number is written in.
CLOCKS_NATIVE = {
    "front_wing": (W.MACRO_DT, "front_wing"),
    "cooling_loop": (CL.MACRO_DT, "cooling_loop"),
    "powertrain": (PT.MACRO_DT, "powertrain"),
}

#: The same three, as native LENGTHS.
LENGTHS_NATIVE = {
    "front_wing": (W.DEFAULT_TILING.wx * W.DX, "front_wing"),   # one window
    "cooling_loop": (CL.L_Z, "cooling_loop"),                   # the passage
    "powertrain": (1.0, "powertrain"),                          # the diameter
}


def clock_table(scale: VehicleScale = None) -> dict[str, Any]:
    """The three clocks as bare numbers and as seconds, with their ratios."""
    scale = VEHICLE if scale is None else scale
    rows = {}
    for sub, (dt, system) in CLOCKS_NATIVE.items():
        ln, lsys = LENGTHS_NATIVE[sub]
        rows[sub] = {
            "dt_native_bare": dt, "unit_system": system,
            "dt_native_s": scale.seconds(dt, system),
            "L_native_bare": ln, "L_native_m": scale.metres(ln, lsys),
        }
    fast = min(r["dt_native_s"] for r in rows.values())
    for r in rows.values():
        r["over_the_fastest_clock"] = r["dt_native_s"] / fast
    bare = sorted({dt for dt, _ in CLOCKS_NATIVE.values()})
    secs = sorted({r["dt_native_s"] for r in rows.values()})
    return {
        "scale": {"l0_m": scale.l0_m, "u0_ms": scale.u0_ms, "t0_s": scale.t0_s,
                  "chord_m": scale.chord_m, "cell_m": scale.cell_m,
                  "domain_m": list(scale.domain_m),
                  "rotor_diameter_m": scale.rotor_diameter_m,
                  "reynolds_model": scale.reynolds_model,
                  "reynolds_vehicle": scale.reynolds_vehicle},
        "subsystems": rows,
        "bare_numbers": bare,
        "seconds": secs,
        "spread_in_bare_numbers": max(bare) / min(bare),
        "spread_in_seconds": max(secs) / min(secs),
        #: Tier 46 reconciled the clocks by re-declaring ten agents' bare
        #: numbers at the tiling's 0.0125.  In seconds that puts the coolant
        #: circuit at the tiling's step, which is this factor away from its own.
        "what_reconciling_the_bare_numbers_does_to_the_coolant_clock":
            scale.seconds(CL.MACRO_DT, "cooling_loop")
            / scale.seconds(W.MACRO_DT, "front_wing"),
    }


def clock_ratio(scale: VehicleScale = None) -> float:
    """The coolant circuit's clock over the tiling's, in seconds."""
    scale = VEHICLE if scale is None else scale
    return (scale.seconds(CL.MACRO_DT, "cooling_loop")
            / scale.seconds(W.MACRO_DT, "front_wing"))


#: How many fluid macro-steps make one coolant step, at the declared scale.
#: A whole number, so the clocks NEST -- which is what `L7/R9`'s quadrature
#: branch asks for and what makes a sub-cycled march well posed.
N_FLUID_PER_COOLANT = int(round(clock_ratio(VEHICLE)))


# ===========================================================================
# DECISION 2 -- a rotor sized for its host, and a machine sized for it (W199)
# ===========================================================================

#: The rotor's face in `front_wing`'s tiling: 32 cells at 1/64.  Tier 46
#: re-declared the rotor's PORTS for their host and left the rotor itself a
#: one-diameter disk, so the shaft delivered exactly twice the power the flow
#: gave up through the face.  The device is now its host's size too.
HOST_ROTOR_WIDTH = WA.ROTOR_CELLS * W.DX          # 0.5
ROTOR_SCALE = HOST_ROTOR_WIDTH / 1.0              # A / A_declared


def machine_for_rotor(scale: float = ROTOR_SCALE) -> dict[str, Any]:
    """`powertrain`'s elements, scaled so an operating point exists again.

    **The problem (W199).**  `disk.ActuatorDisk` has ``radius = area / 2`` and
    ``omega = lambda U_d / radius``, so halving the swept width DOUBLES the
    shaft speed; and ``torque = thrust * radius / lambda`` with
    ``thrust = 1/2 rho A C_T' U_d^2``, so the torque it can deliver falls as
    ``A^2`` -- a quarter.  The machine meanwhile sits just above the battery's
    open-circuit voltage, ``I = (k_e omega - V_oc) / R``, so its current is a
    small difference of two large numbers and doubling ``omega`` multiplies it
    by thirty.  Measured by Tier 47: the declared machine demands **31x** the
    largest torque a host-sized disk delivers anywhere in its induction clamp,
    so the powertrain has no operating point at all.

    **The choice, and it is a similarity rather than a fit.**  Ask that the
    re-sized machine reach the SAME induction at the SAME inflow -- that is,
    that the whole electrical solution be similar and only its size change.
    Two conditions decide it:

      * the back-EMF must be unchanged, ``k_e omega`` invariant, and
        ``omega ~ 1/A``, so ``k_e -> A k_e``; and since ``k_e = k_t`` is the
        same air-gap flux linkage, ``k_t -> A k_t`` with it;
      * the torque must follow the disk's ``A^2``.  With ``k_t ~ A`` that needs
        ``I ~ A``, and the numerator ``k_e omega - V_oc`` is invariant, so
        ``R -> R / A``.

    At ``A = 1/2``: ``k_e = k_t = 0.05``, every element's resistance doubled,
    ``V_oc`` untouched.  The result is checked rather than asserted -- at
    ``A = 1`` it must reproduce `powertrain`'s own elements to the bit, and at
    ``A = 1/2`` it must return Tier 46's induction, which it does to twelve
    digits.

    **What it costs.**  The shaft power falls as ``A`` -- 0.2016 to 0.1008 in
    the tiling's units -- and the machine's ``I^2 R`` falls with it, so with
    Tier 46's power unit ``p_ref`` HELD (the decision that made it a unit rather
    than a fit) the heat the machine puts into the block halves.  That is a
    smaller vehicle, not a different framework.
    """
    s = float(scale)
    if not (s > 0.0):
        raise ValueError(f"the rotor's width scale must be positive, got {s}")
    mgu = _ScaledMachine(resistance=PT.R_MGU / s, k_e=PT.K_E * s,
                         k_t=PT.K_T * s, r_total=PT.R_TOTAL / s)
    return {"MGU": mgu,
            "BUS": PT.BusLeg(resistance=PT.R_BUS / s),
            "BATT": PT.BatteryLeg(resistance=PT.R_BATT / s),
            "INV": PT.InverterLeg(resistance=PT.R_INV / s)}


@dataclass
class _ScaledMachine(PT.MachineAgent):
    """`MachineAgent` whose loop resistance is its own rather than the module's.

    `MachineAgent.current_at` reads the module-level ``R_TOTAL``, so scaling the
    elements' resistances would leave the machine solving its own current
    against the unscaled loop.  This carries the total it is actually in.
    """

    r_total: float = PT.R_TOTAL

    def current_at(self, omega: float) -> float:
        return (self.k_e * float(omega) - PT.V_OC) / self.r_total


class SizedCircuitSolve(PT.CircuitSolve):
    """`CircuitSolve` with the disk's swept width declared.

    The parent builds ``ActuatorDisk(a=a)`` at the donor's default width and
    reads ``omega`` off ``rotor._disk``, which is the same default, so its
    operating point is a one-diameter rotor's wherever the rotor is placed
    (W199, and `scripts/rung9_gate.py` first).  This overrides exactly those two
    reads and nothing else, so at ``width = 1`` it reproduces the parent.
    """

    def __init__(self, width: float, **kw) -> None:
        super().__init__(**kw)
        self.width = float(width)

    def torque_at(self, a: float) -> float:
        disk = self.rotor._mod.ActuatorDisk(a=float(a), area=self.width)
        return float(disk(self.u_ref).torque)

    def omega(self) -> float:
        disk = self.rotor._mod.ActuatorDisk(a=self.rotor.a, area=self.width)
        return float(disk(self.u_ref).omega)


def operating_point(u_ring: np.ndarray, width: float = HOST_ROTOR_WIDTH,
                    scale: float | None = None,
                    elements: dict | None = None) -> tuple[Any, dict, Any]:
    """Solve the union's shared operating point at this rotor inflow (W196).

    Returns ``(CircuitResult, elements, DiskState)``.  The disk state is the
    donor's own, evaluated at the solved induction and the declared width, so
    the thrust the march applies and the power the shaft claims come from one
    evaluation rather than two.
    """
    u_ring = np.asarray(u_ring, dtype=float).reshape(-1)
    u_mean = float(np.mean(u_ring))
    scale = (width / 1.0) if scale is None else float(scale)
    els = machine_for_rotor(scale) if elements is None else elements
    rotor = WA.RotorDisk(agent_id="ROTOR", u_ref=u_ring)
    res = SizedCircuitSolve(width, rotor=rotor, elements=els,
                            u_ref=u_mean).solve()
    disk = rotor._mod.ActuatorDisk(a=res.induction, area=width)
    return res, els, disk(u_mean)


# ===========================================================================
# DECISION 5 -- the devices applied as BODY FORCES (W94)
# ===========================================================================


@dataclass(frozen=True)
class DeviceSpec:
    """Where a device's plane is in the host, and how wide it is.

    The plane sits midway between the two rings `integration_union` declares
    its ports on -- the downstream window's ``xlo`` ring UPSTREAM of the plane
    and the upstream window's ``xhi`` ring downstream of it -- which is
    `wake_array`'s own convention, and the inflow is read on the upstream ring
    exactly as `disk.disk_average` insists ("deliberately upstream of the strip:
    sampling inside it would read a velocity the disk's own body force has
    already slowed").
    """

    join: str
    site: IU.DeviceSite
    label: str

    @property
    def rows(self) -> np.ndarray:
        t = W.DEFAULT_TILING
        _ox, oy = t.offsets[t.names.index(self.site.right)]
        return oy + self.site.cells

    @property
    def i_up(self) -> int:
        """Column index of the ring the inflow is read on."""
        t = W.DEFAULT_TILING
        ox, _oy = t.offsets[t.names.index(self.site.right)]
        return int(ox)

    @property
    def i_down(self) -> int:
        t = W.DEFAULT_TILING
        ox, _oy = t.offsets[t.names.index(self.site.left)]
        return int(ox + t.wx - 1)

    @property
    def x_plane(self) -> float:
        """Physical x of the device plane: midway between the two rings."""
        return 0.5 * ((self.i_up + 0.5) + (self.i_down + 0.5)) * W.DX

    @property
    def y0(self) -> float:
        """Physical y of the strip's lower edge."""
        return float(self.rows[0]) * W.DX

    @property
    def width(self) -> float:
        return float(self.rows.size) * W.DX


ROTOR_DEVICE = DeviceSpec("J3", IU.ROTOR_SITE, "the rotor disk")
CORE_DEVICE = DeviceSpec("J1", IU.CORE_SITE, "the radiator core")
DEVICES = (ROTOR_DEVICE, CORE_DEVICE)
JOIN_OF_DEVICE = {d.site.device: d.join for d in DEVICES}


class DeviceForcing:
    """The two devices' streamwise momentum sinks, on the host lattice.

    **The donor's own body force, not a second copy of it.**
    `disk.ActuatorDisk.body_force_field` weights each cell by the exact
    rectangle overlap with the strip, so ``sum(f_x * cell_area) == -thrust`` to
    floating point on any lattice -- gate W2's check in the build repo, and this
    class asserts it again here on every call, because a discretization that
    lost 3% of the thrust would read as a 3% interface residual in section 5's
    receiver balance and be blamed on the join.

    **The smearing thickness, and the one place this departs from
    `w100_scaling_ladder._forcing`.**  That rule is
    ``Delta_d = <U_d> * dt``, the distance a parcel travels while the impulse is
    applied, which is what makes the impulse ``T dt / (A Delta_d)`` agree with
    the ``T / (A <U_d>)`` momentum theory allows.  On `wake_array`'s own lattice
    that is 1.6 cells.  On this host the cell is half as wide and the clock is
    sixteen times finer, so the same rule gives **0.18 cells** -- a strip inside
    one cell.  The rule is kept and a floor of one cell is DECLARED on top of
    it, so the sink is never narrower than the grid can represent.  The floor
    changes no total: the exact-overlap weighting conserves the thrust either
    way, and what it changes is how sharply the sink is presented to the
    projection.  `thickness_report` records both numbers on every march.
    """

    def __init__(self, devices=DEVICES, dt_apply: float = None,
                 cell_floor: float = 1.0) -> None:
        self.devices = tuple(devices)
        self.dt_apply = (W.MACRO_DT / W.EXCHANGES) if dt_apply is None else float(dt_apply)
        self.cell_floor = float(cell_floor)
        t = W.DEFAULT_TILING
        self.ny, self.nx = t.ny, t.nx
        self.x_c = (np.arange(self.nx) + 0.5) * W.DX
        self.y_c = (np.arange(self.ny) + 0.5) * W.DX
        self._mod = WA.RotorDisk(agent_id="_PROBE",
                                 u_ref=np.ones(WA.ROTOR_CELLS))._mod
        self.last: dict[str, dict] = {}

    def thickness(self, u_disk: float) -> tuple[float, float]:
        """(the thickness used, the thickness the donor's rule alone gives)."""
        raw = max(float(u_disk), 0.05) * self.dt_apply
        return max(raw, self.cell_floor * W.DX), raw

    def field(self, thrust: float, dev: DeviceSpec, u_disk: float) -> np.ndarray:
        """``f_x`` over the whole domain for one device, as an exact sink."""
        thick, _raw = self.thickness(u_disk)
        d = self._mod.ActuatorDisk(area=dev.width, thickness=thick)
        fx = d.body_force_field(self.x_c, self.y_c, W.DX, W.DX, float(thrust),
                                x0=dev.x_plane - 0.5 * thick, y0=dev.y0)
        got = float(np.sum(fx) * W.DX * W.DX)
        if abs(got + float(thrust)) > 1e-9 * max(abs(float(thrust)), 1e-30):
            raise RuntimeError(
                f"{dev.label}'s body force integrates to {got:.12g} against a "
                f"thrust of {-float(thrust):.12g}: the discrete sink is not the "
                "thrust it is supposed to be, and the difference would be "
                "charged to the join")
        return fx

    def thickness_report(self) -> dict[str, Any]:
        out = {}
        for dev in self.devices:
            rec = self.last.get(dev.site.device, {})
            ud = rec.get("u_disk", 1.0)
            used, raw = self.thickness(ud)
            out[dev.site.device] = {
                "u_disk": ud, "thickness_used": used,
                "thickness_from_the_donors_rule": raw,
                "cells_used": used / W.DX, "cells_from_the_rule": raw / W.DX,
                "floor_bound": bool(used > raw),
                "dt_apply": self.dt_apply,
            }
        return out


# ===========================================================================
# DECISION 6 -- the shared operating point, solved across the union (W196)
# ===========================================================================


@dataclass
class JoinState:
    """What crosses one join, and when it was last read.

    The union's three joins carry six numbers between them, and `joining-seam-cost`
    section 5.4's whole point is that these are STATE a join re-derives, that the
    re-derivation is not local to the join, and that no rule reads any of it.  A
    march has to read all of it, which is why they are one object here.
    """

    #: J3 -- the rotor
    u_rotor: float = 0.0
    induction: float = 0.0
    omega: float = 0.0
    current: float = 0.0
    thrust_rotor: float = 0.0
    shaft_power: float = 0.0
    rotor_valid: bool = True
    #: J1 -- the radiator core
    u_core: float = 0.0
    ua: float = CL.UA_RAD
    thrust_core: float = 0.0
    core_flow_power: float = 0.0
    #: J2 -- the machine's heat
    q_machine: float = 0.0
    #: bookkeeping
    step: int = -1
    exchange: int = -1

    def as_dict(self) -> dict[str, float]:
        return {k: (float(v) if not isinstance(v, bool) else bool(v))
                for k, v in self.__dict__.items()}


#: The six quantities that cross the union's joins, each in its own currency.
#: Composition error is measured on the vector of their RELATIVE differences
#: (section `composition_error`), which is the only dimensionless thing all
#: three joins move.
CROSSING_KEYS = ("u_core", "ua", "u_rotor", "current", "q_machine", "t_wall")


def crossing_vector(js: JoinState, t_wall: float) -> np.ndarray:
    d = js.as_dict()
    d["t_wall"] = float(t_wall)
    return np.array([float(d[k]) for k in CROSSING_KEYS], dtype=float)


def composition_error(test: np.ndarray, referent: np.ndarray) -> float:
    """The union's composition error: relative, per crossing quantity, in l2.

    Each of `CROSSING_KEYS` is divided by the REFERENT's value of it, so the
    six add without a unit reconciliation -- the one thing the union has no
    field for (W200, W201).  A temperature is referenced to its own value in
    kelvin rather than to a rise, which makes `t_wall` the least sensitive
    entry by construction; it is reported per component as well as in norm so
    that is visible rather than hidden in a scalar.
    """
    test = np.asarray(test, dtype=float)
    referent = np.asarray(referent, dtype=float)
    denom = np.where(np.abs(referent) > 1e-300, np.abs(referent), 1.0)
    return float(np.linalg.norm((test - referent) / denom))


# ===========================================================================
# the union's fluid: front_wing's column with two devices in it
# ===========================================================================


class UnionRollout(FW.FrontWingRollout):
    """`front_wing`'s six-window column with the union's two device planes in it.

    **It subclasses and overrides `exchange` and nothing else in the fluid.**
    The cut, the partition of unity, the one global spectral Leray projection
    per exchange and the boundary band are CS-12's, which are CS-10's; the plate
    is still applied as a body force by `FlexWing.forcing`, and the two devices
    are added to that same ``f_x`` before it is cut.  A device is therefore
    applied exactly the way the plate already was, which is decision 5.

    **The joins' cadence is the knob** (`join_coupling`).  ``"tight"`` re-reads
    a join's crossing data and re-solves the operating point at EVERY exchange
    and iterates the device's own fixed point ``n_join_inner`` times;
    ``"lagged"`` reads once per macro-step and holds.  That is `FSIRollout`'s
    own tight/lagged distinction carried to the joins, and the difference
    between the two is what section `composition_error` measures.

    **The march is not differentiable through the devices.**  The donor's body
    force is numpy, and the operating point is a bisection on it; both are
    converted rather than taped.  `front_wing`'s own gradient path is untouched
    and unused here.  Named, not hidden.
    """

    def __init__(self, *, joins=("J1", "J2", "J3"),
                 join_coupling: str | dict = "tight",
                 rotor_width: float = HOST_ROTOR_WIDTH,
                 machine_scale: float | None = None,
                 n_join_inner: int = 3,
                 p_ref: float | None = None, null: str | None = None,
                 devices=DEVICES, forcing: DeviceForcing | None = None,
                 **kw) -> None:
        kw.setdefault("coupling", "tight")
        kw.setdefault("motion", False)
        super().__init__(**kw)
        self.joins = tuple(j for j in ("J1", "J2", "J3") if j in tuple(joins))
        if isinstance(join_coupling, str):
            join_coupling = {j: join_coupling for j in ("J1", "J2", "J3")}
        bad = [v for v in join_coupling.values() if v not in ("tight", "lagged")]
        if bad:
            raise ValueError(f"a join's coupling is 'tight' or 'lagged', not {bad}")
        self.join_coupling = dict(join_coupling)
        self.rotor_width = float(rotor_width)
        self.machine_scale = (self.rotor_width if machine_scale is None
                              else float(machine_scale))
        self.n_join_inner = int(n_join_inner)
        self.elements = machine_for_rotor(self.machine_scale)
        self.p_ref = (IU.calibrated_p_ref(PT.MachineAgent())[0] if p_ref is None
                      else float(p_ref))
        #: **The join whose TERM is removed, receiver-side, with everything
        #: that reads it left in place.**  Removing the whole device instead
        #: would leave the shaft claiming nothing, and the balance would read
        #: 0/0 rather than failing -- a null arm that cannot fail is not a
        #: control.  So the rotor's operating point is still solved and its
        #: shaft still claims its power; what is withheld is the force on the
        #: fluid.  For J1 the term is the DEPENDENCE, so UA is pinned at the
        #: declared radiator's UA_RAD while the core's drag still moves the air.
        self.null = null if null in ("J1", "J3") else None
        self.devices = tuple(d for d in devices if d.join in self.joins)
        self.forcing_ = DeviceForcing(self.devices) if forcing is None else forcing
        self.state = JoinState()
        self._fx_dev = np.zeros((self.ny, self.nx))
        self._fx_by_dev: dict[str, np.ndarray] = {}
        self._opt_t = dict(dtype=W.TORCH_DTYPE, device=self.device)
        #: **The core's reference air speed is the state the join was BUILT at**,
        #: and it is held.  `CoreRadiator.__post_init__` sets ``u_air_ref`` from
        #: whatever ring it is constructed with, so building a fresh one every
        #: refresh would make ``UA(u) = UA_RAD (u/u)^0.8`` identically `UA_RAD`
        #: and the join would carry no dependence at all -- the failure this
        #: line exists to stop.  One object, built once, its ``u_air`` updated.
        self._core: IU.CoreRadiator | None = None
        #: per-exchange trace of the join residual, so a "tight" arm's inner
        #: fixed point is defended rather than assumed
        self.join_residual: list[float] = []
        self.work_trace: list[dict] = []
        self._exchange_i = 0

    # -- reading the rings -------------------------------------------------

    def ring(self, u, dev: DeviceSpec) -> np.ndarray:
        """The streamwise velocity on the ring upstream of a device's plane."""
        w = u.detach().cpu().numpy() if torch.is_tensor(u) else np.asarray(u)
        return np.asarray(w, dtype=float)[dev.rows, dev.i_up].copy()

    def ring_down(self, u, dev: DeviceSpec) -> np.ndarray:
        w = u.detach().cpu().numpy() if torch.is_tensor(u) else np.asarray(u)
        return np.asarray(w, dtype=float)[dev.rows, dev.i_down].copy()

    # -- the joins ---------------------------------------------------------

    def refresh(self, u, step: int, which=("J1", "J3")) -> None:
        """Re-read the listed joins' crossing data and rebuild the body force.

        This IS decision 6: the operating point is solved ACROSS THE UNION, at
        the inflow the host's ring actually carries, every time this is called.
        `joining-seam-cost` section 5.4 measured that re-derivation reaching six
        ports on agents off J3's own seams; here it reaches the fluid as a
        thrust and the block as a heat, on the same call.
        """
        s = self.state
        fx = np.zeros((self.ny, self.nx))
        for dev in self.devices:
            if dev.join == "J3" and "J3" in which:
                ring = self.ring(u, dev)
                res, _els, disk = operating_point(
                    ring, width=self.rotor_width, scale=self.machine_scale,
                    elements=self.elements)
                s.u_rotor = float(np.mean(ring))
                s.induction = float(res.induction)
                s.omega = float(res.omega)
                s.current = float(res.current)
                s.thrust_rotor = float(disk.thrust)
                s.shaft_power = float(disk.power)
                s.rotor_valid = bool(res.rotor_valid)
                if "J2" in self.joins:
                    #: **The machine's own windings, not the loop's total.**
                    #: `CooledMachine.heat_flux` dissipates ``I^2 R(T)`` with
                    #: ``R`` the MACHINE's resistance -- the bus, battery and
                    #: inverter losses happen elsewhere and are not in the
                    #: casing the block is bolted to -- and `integration_union`
                    #: builds J2's heat the same way.  Charging the loop total
                    #: here doubles it exactly, because ``R_TOTAL = 2 R_MGU``,
                    #: and the doubling is invisible in a relative error.
                    s.q_machine = float(s.current ** 2
                                        * self.elements["MGU"].resistance
                                        * self.p_ref)
                self.forcing_.last[dev.site.device] = {"u_disk": s.u_rotor}
            elif dev.join == "J1" and "J1" in which:
                ring = self.ring(u, dev)
                s.u_core = float(np.mean(ring))
                if self._core is None:
                    self._core = IU.CoreRadiator(u_air=ring)   # fixes u_air_ref
                core = self._core
                core.u_air = ring
                s.ua = (CL.UA_RAD if self.null == "J1" else float(core.ua))
                tau = core.traction(ring)
                s.thrust_core = float(np.sum(tau) * W.DX)
                s.core_flow_power = float(np.sum(tau * ring) * W.DX)
                self.forcing_.last[dev.site.device] = {"u_disk": s.u_core}
        # rebuild the whole field from the current state, so a lagged join
        # keeps its held value and a tight one gets the new one.  Kept PER
        # DEVICE as well as summed, so each join's receiver balance is charged
        # its own force's work rather than the pair's minus an estimate of the
        # other -- the two devices sit in the same domain and subtracting one
        # ring-side estimate from a field-side total mixes two conventions.
        by_dev: dict[str, np.ndarray] = {}
        for dev in self.devices:
            #: **Each join's TERM is a different kind of thing, so each null
            #: withholds a different thing.**  J3's term is the power path: the
            #: rotor's force on the fluid, withheld while the shaft goes on
            #: claiming its power, which is what makes the balance fail rather
            #: than read 0/0.  J1's term is the DEPENDENCE -- `rung9-gate-restated`
            #: section 4.4 established that no power crosses J1 -- so its null
            #: pins UA at the declared radiator's value and leaves the core's
            #: drag in place.  Withholding the drag too would move the air and
            #: the conductance in one step and the control would vary two things.
            if dev.join == "J3" and self.null == "J3":
                continue
            if dev.join == "J3" and s.thrust_rotor:
                by_dev[dev.site.device] = self.forcing_.field(
                    s.thrust_rotor, dev, s.u_rotor)
            elif dev.join == "J1" and s.thrust_core:
                by_dev[dev.site.device] = self.forcing_.field(
                    s.thrust_core, dev, s.u_core)
        for f in by_dev.values():
            fx = fx + f
        self._fx_by_dev = by_dev
        self._fx_dev = fx
        s.step = int(step)

    # -- the exchange ------------------------------------------------------

    def _advance(self, u, v, delta, h, w_plate):
        """One fluid exchange at the device force currently held.

        Split out of `exchange` so a TIGHT join can take it more than once from
        the same starting state -- which is the only way the device's own fixed
        point is a fixed point: the thrust depends on the inflow at the END of
        the exchange, and that inflow depends on the thrust.  Re-reading the
        START state, as the first version did, converges in one call and
        measures nothing (`join_residual` was identically zero).
        """
        fx, fy, w, load = self.wing.forcing(u, v, delta, w_plate, self.ny,
                                            self.nx, h=h)
        fdev = torch.as_tensor(self._fx_dev, **self._opt_t)
        fx = fx + fdev
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)
        u1, v1 = self.solver.step_batch(us, vs, self.dt_ex, bc0=None,
                                        force=(fxs, fys))
        self.substep_log.append(int(self.solver.last_substeps))
        au, av = self.blend(u1, v1)
        au, av = self.project(au, av)
        bu, bv = self.band(au, av)
        return bu, bv, w, load, fdev

    def exchange(self, u, v, delta, h, w_plate):
        """`FrontWingRollout.exchange` with the devices added to ``f_x``.

        **Tight and lagged differ in WHICH state the device reads.**  A lagged
        join computes its thrust from the ring at the start of the MACRO-STEP
        and holds it for all `EXCHANGES` of them.  A tight join computes it
        from the ring at the END of the exchange, found by a fixed-point
        iteration of a declared, fixed count, with the residual recorded so the
        count is defended rather than assumed.  This is `FSIRollout`'s own
        tight/lagged distinction carried to the joins, and the difference
        between the two arms is the composition error G3 is about.
        """
        tight = [j for j in ("J1", "J3")
                 if j in self.joins and self.join_coupling.get(j) == "tight"]
        u0 = u
        if tight:
            prev, r_first, r_last = None, None, 0.0
            for _ in range(max(1, self.n_join_inner) + 1):
                bu, bv, w, load, fdev = self._advance(u, v, delta, h, w_plate)
                self.refresh(bu, self.state.step, which=tight)
                cur = np.array([self.state.u_rotor, self.state.u_core])
                if prev is not None:
                    r_last = float(np.linalg.norm(cur - prev))
                    r_first = r_last if r_first is None else r_first
                prev = cur
            #: the LAST advance uses the force the last refresh built, so the
            #: state returned and the thrust that drove it are consistent
            bu, bv, w, load, fdev = self._advance(u, v, delta, h, w_plate)
            self.join_residual.append((float(r_first or 0.0), r_last))
        else:
            bu, bv, w, load, fdev = self._advance(u, v, delta, h, w_plate)
            self.join_residual.append((0.0, 0.0))
        #: **The power the devices take out of the fluid, measured on the
        #: march.**  The force is explicit over the exchange, so the work it
        #: does is quoted three ways -- at the start of the exchange, at the
        #: end, and at the midpoint -- and the spread between them is the O(dt)
        #: ambiguity, reported rather than argued away.  This is the term
        #: section 5's J3 balance charges the shaft against, and it is read off
        #: the BODY FORCE rather than off the two declared ports, which is what
        #: dissolves W197 for a march (the ports' net power is exactly zero).
        with torch.no_grad():
            a_ = float((fdev * u0).sum() * W.DX * W.DX)
            b_ = float((fdev * bu).sum() * W.DX * W.DX)
            per = {}
            for name, f in self._fx_by_dev.items():
                ft = torch.as_tensor(f, **self._opt_t)
                pa = float((ft * u0).sum() * W.DX * W.DX)
                pb = float((ft * bu).sum() * W.DX * W.DX)
                per[name] = 0.5 * (pa + pb)
                per[name + "_start"] = pa
                per[name + "_end"] = pb
        self.work_trace.append({
            "exchange": self._exchange_i, "step": self.state.step,
            "power_on_the_fluid_start": a_, "power_on_the_fluid_end": b_,
            "power_on_the_fluid_mid": 0.5 * (a_ + b_),
            "shaft_power": self.state.shaft_power,
            "core_flow_power": self.state.core_flow_power,
            "per_device": per,
        })
        self._exchange_i += 1
        f = self.wing.normal_traction(w)
        drag = (f * self.n_hat_x * self.ds).sum()
        return bu, bv, load, f, drag

    def macro_step(self, u, v, delta, h, w_delta, v_mount, e_star, tc, k, h0,
                   step: int = 0):
        lagged = [j for j in ("J1", "J3")
                  if j in self.joins and self.join_coupling.get(j) == "lagged"]
        self.state.step = int(step)
        if lagged or self.state.step <= 0:
            self.refresh(u, step, which=tuple(lagged) if lagged else ())
        return super().macro_step(u, v, delta, h, w_delta, v_mount,
                                  e_star, tc, k, h0, step)


# ===========================================================================
# the union's march
# ===========================================================================


@dataclass
class UnionMarch:
    """What one march of the union returns."""

    steps: int
    join_coupling: dict
    joins: tuple
    #: per macro-step traces of everything that crosses a join
    trace: dict[str, np.ndarray] = field(default_factory=dict)
    #: the coolant circuit's own steps, taken every `N_FLUID_PER_COOLANT`
    coolant: list = field(default_factory=list)
    #: the receiver balances, per coolant step and per macro-step
    balances: dict = field(default_factory=dict)
    u: Any = None
    v: Any = None
    block_T: Any = None
    t_in: float = CL.T_COOLANT_0
    wall_s: float = 0.0
    thickness: dict = field(default_factory=dict)
    join_residual: Any = None
    notes: dict = field(default_factory=dict)

    def settled(self, frac: float = 0.25) -> dict[str, float]:
        n = max(1, int(round(frac * self.steps)))
        return {k: float(np.asarray(v)[-n:].mean())
                for k, v in self.trace.items() if np.asarray(v).ndim == 1}

    def crossing(self) -> np.ndarray:
        s = self.settled()
        s.setdefault("t_wall", float(np.mean(self.trace["t_wall"][-1])))
        return np.array([float(s[k]) for k in CROSSING_KEYS], dtype=float)


def march_union(u0: np.ndarray, v0: np.ndarray, steps: int = 1200,
                joins=("J1", "J2", "J3"), join_coupling="tight",
                rotor_width: float = HOST_ROTOR_WIDTH,
                machine_scale: float | None = None,
                null: str | None = None,
                n_per_coolant: int = N_FLUID_PER_COOLANT,
                progress=None, **kw) -> UnionMarch:
    """March the union: the fluid on its clock, the circuit algebraic, the
    coolant circuit sub-cycled at the ratio the units declaration gives.

    ``null`` removes ONE join's term while leaving everything that reads it in
    place -- the control every clause on this page needs:

      ``"J3"``  the rotor's body force is not applied, and the shaft still
                claims its power.  J3's receiver balance must then FAIL.
      ``"J2"``  the block's mount term is removed and its own declared source
                is not put back.  The block's first law must then FAIL.
      ``"J1"``  the core's drag is not applied and ``UA`` is held at `UA_RAD`.
                The loop must then stop following the air.

    A gate a broken graph passes is not a gate, so every arm is run beside its
    null and both are reported.
    """
    r = UnionRollout(joins=joins, join_coupling=join_coupling, null=null,
                     rotor_width=rotor_width, machine_scale=machine_scale, **kw)
    opt = r._opt
    kn = r.knobs(None)
    e_star, tc, k, h0 = kn["e_star"], kn["tc"], kn["k"], kn["h0"]
    u = torch.as_tensor(np.asarray(u0), **opt)
    v = torch.as_tensor(np.asarray(v0), **opt)
    delta = torch.zeros(W.N_STATION, **opt)
    h = r.release_height(h0, k)
    w_delta = torch.zeros(W.N_STATION, **opt)
    v_mount = torch.zeros((), **opt)
    r.substep_log = []

    block = (IU.MountedBlock(q_machine=0.0) if ("J2" in joins and null != "J2")
             else CL.BlockAgent())
    loop = CL.LoopSolve(block=block)
    t_in = CL.T_COOLANT_0

    keys = ("u_rotor", "u_core", "ua", "induction", "omega", "current",
            "thrust_rotor", "shaft_power", "thrust_core", "core_flow_power",
            "q_machine", "load", "drag", "u_max", "power_on_the_fluid")
    trace: dict[str, list] = {kk: [] for kk in keys}
    trace["t_wall"] = []
    trace["power_rotor"] = []
    trace["power_core"] = []
    trace["u_at_the_rotor_plane"] = []
    coolant: list[dict] = []
    import time
    t0 = time.perf_counter()
    r.refresh(u, 0, which=tuple(j for j in ("J1", "J3") if j in r.joins))
    #: J1's radiator is the rollout's OWN core object, so the ``UA`` the loop
    #: integrates is the one the fluid trace records, referenced to the release
    #: ring and not to a fresh one (which would make the dependence vanish).
    if "J1" in joins and null != "J1" and r._core is not None:
        loop.legs["RAD"] = r._core
    for s in range(steps):
        n_before = len(r.work_trace)
        out = r.macro_step(u, v, delta, h, w_delta, v_mount,
                           e_star, tc, k, h0, s)
        (u, v, delta, h, w_delta, v_mount, load, drag, _f, _vm,
         _res, _r0, _p, _half) = out
        st = r.state
        wk = r.work_trace[n_before:]
        for kk in ("u_rotor", "u_core", "ua", "induction", "omega", "current",
                   "thrust_rotor", "shaft_power", "thrust_core",
                   "core_flow_power", "q_machine"):
            trace[kk].append(float(getattr(st, kk)))
        trace["load"].append(float(load.detach()))
        trace["drag"].append(float(drag.detach()))
        trace["u_max"].append(float(torch.max(torch.hypot(u, v)).detach()))
        trace["power_on_the_fluid"].append(
            float(np.mean([x["power_on_the_fluid_mid"] for x in wk])) if wk else 0.0)
        trace["power_rotor"].append(float(np.mean(
            [x["per_device"].get("ROTOR", 0.0) for x in wk])) if wk else 0.0)
        trace["power_core"].append(float(np.mean(
            [x["per_device"].get("RAD", 0.0) for x in wk])) if wk else 0.0)
        #: the streamwise velocity in the cell the rotor's sink actually sits
        #: in, against the ring the disk reads 8 cells upstream -- the two
        #: velocities J3's balance turns on
        i_pl = int(round(ROTOR_DEVICE.x_plane / W.DX - 0.5))
        trace["u_at_the_rotor_plane"].append(float(
            u.detach().cpu().numpy()[ROTOR_DEVICE.rows, i_pl].mean()))
        if not torch.isfinite(u).all():
            raise RuntimeError(f"the union's fluid is not finite at macro-step {s}")
        # -- the coolant circuit, on its own clock -------------------------
        if (s + 1) % n_per_coolant == 0:
            if isinstance(block, IU.MountedBlock):
                block.q_machine = float(st.q_machine)
                tw = float(np.mean(block._face_T(block._T, "outer")))
                block.t_case = tw + block.q_machine / (block.h_mount * block.area)
            row = _coolant_step(loop, block, t_in)
            t_in = row["t_in"]
            row["fluid_step"] = s
            coolant.append(row)
        trace["t_wall"].append(float(np.mean(
            block._face_T(block._T, "inner"))))
        if progress is not None:
            progress(s, time.perf_counter() - t0)
    return UnionMarch(
        steps=steps, join_coupling=dict(r.join_coupling), joins=tuple(r.joins),
        trace={kk: np.asarray(vv) for kk, vv in trace.items()},
        coolant=coolant, u=u.detach().cpu().numpy().copy(),
        v=v.detach().cpu().numpy().copy(), block_T=block._T.copy(), t_in=t_in,
        wall_s=time.perf_counter() - t0,
        thickness=r.forcing_.thickness_report(),
        join_residual=np.asarray(r.join_residual),
        notes={"null": null, "n_per_coolant": n_per_coolant,
               "rotor_width": r.rotor_width, "machine_scale": r.machine_scale,
               "p_ref": r.p_ref, "n_join_inner": r.n_join_inner,
               "coolant_steps_taken": len(coolant)})


def _coolant_step(loop: CL.LoopSolve, block, t_in: float) -> dict:
    """One step of `LoopSolve.solve`'s body, so the march can interleave it."""
    loop.legs["PASS"].t_in = t_in
    wall_t = loop.legs["PASS"].wall_temperature()
    block.step(wall_t)
    q_block = block.heat_in(wall_t)
    t_new = loop._closed_form(q_block)
    temps = loop._once(t_new, q_block)
    rejected = {n: float(loop.legs[n].heat_out())
                for n in loop.order if hasattr(loop.legs[n], "heat_out")}
    q_out = sum(rejected.values())
    q_src = q_block + CL.W_PUMP
    bal = block.energy_balance(loop.legs["PASS"].wall_temperature())
    return {"t_in": float(t_new), "q_block": float(q_block),
            "t_return": float(temps[loop.order[-1]]),
            "q_rejected": rejected, "q_rejected_total": float(q_out),
            "loop_residual": float(abs(q_src - q_out) / max(abs(q_src), 1e-30)),
            "ua_rad": float(getattr(loop.legs["RAD"], "ua", CL.UA_RAD)),
            "block_first_law": {kk: float(vv) for kk, vv in bal.items()
                                if isinstance(vv, (int, float))},
            "t_wall_mean": float(np.mean(block._face_T(block._T, "inner"))),
            "t_outer_mean": float(np.mean(block._face_T(block._T, "outer")))}


# ===========================================================================
# the slow subsystem alone, on its own clock
# ===========================================================================


@dataclass
class LoopMarch:
    steps: int
    rows: list
    wall_s: float = 0.0
    notes: dict = field(default_factory=dict)


def replay_coolant(m: UnionMarch, n_per_coolant: int = N_FLUID_PER_COOLANT,
                   mounted: bool = True, q_lag: int = 1,
                   ua_from_fluid: bool = True) -> LoopMarch:
    """Run the coolant circuit again on a march's recorded fluid trace.

    **Legitimate because the coupling is one-way on this side, and that is
    asserted rather than assumed.**  Neither the loop's temperature nor the
    block's reaches the fluid: the air carries no temperature (which is why J1
    is a `MECH` bond at all) and the machine's ``R(T)`` enters its `THERM`
    response only, not its `ELEC` one -- Tier 46 named that as not built and it
    is still not built.  So the fluid trajectory does not depend on the coolant
    clock, and `tests/test_tier49_union_march.py` pins the two marches'
    fluid traces bitwise identical at two different ``n_per_coolant``.

    That makes the second vehicle scale, J2's lag arm and J2's null arm free:
    each is this replay on one already-computed fluid trace.
    """
    import time
    t0 = time.perf_counter()
    block = (IU.MountedBlock(q_machine=0.0) if mounted else CL.BlockAgent())
    loop = CL.LoopSolve(block=block)
    if ua_from_fluid and "ua" in m.trace:
        core = IU.CoreRadiator(u_air=np.full(WA.ROTOR_CELLS,
                                             float(m.trace["u_core"][0])))
        loop.legs["RAD"] = core
    t_in = CL.T_COOLANT_0
    rows, held_q = [], None
    n = int(n_per_coolant)
    for s in range(m.steps):
        if (s + 1) % n:
            continue
        if ua_from_fluid and isinstance(loop.legs["RAD"], IU.CoreRadiator):
            loop.legs["RAD"].u_air = np.full(WA.ROTOR_CELLS,
                                             float(m.trace["u_core"][s]))
        if mounted:
            k = len(rows)
            if k % max(1, int(q_lag)) == 0:
                held_q = float(m.trace["q_machine"][s])
            block.q_machine = held_q
            tw = float(np.mean(block._face_T(block._T, "outer")))
            block.t_case = tw + block.q_machine / (block.h_mount * block.area)
        row = _coolant_step(loop, block, t_in)
        t_in = row["t_in"]
        row["fluid_step"] = s
        rows.append(row)
    return LoopMarch(steps=len(rows), rows=rows,
                     wall_s=time.perf_counter() - t0,
                     notes={"replayed_on": m.notes.get("null"),
                            "n_per_coolant": n, "mounted": mounted,
                            "q_lag": q_lag, "fluid_steps": m.steps,
                            "horizon_s": len(rows) * CL.MACRO_DT})


def march_loop(steps: int = 1200, q_machine: float = None, ua: float = None,
               mounted: bool = True, lag: int = 1) -> LoopMarch:
    """The coolant circuit and the block on THEIR clock, the fluid quasi-steady.

    The union's clocks are `N_FLUID_PER_COOLANT` apart, so a march that resolves
    the fluid reaches a small fraction of the block's own thermal transient.
    This marches the slow half alone with the fluid's crossing data HELD at the
    value the union march settled on -- a quasi-steady closure of the fluid,
    which is the standard multirate one and which the measured settling time
    justifies.  ``mounted=False`` is J2's null arm: the block with no mount
    term.  ``lag`` holds the machine's heat for that many coolant steps, which
    is J2's lagged arm.
    """
    import time
    block = (IU.MountedBlock(q_machine=float(q_machine or 0.0)) if mounted
             else CL.BlockAgent())
    loop = CL.LoopSolve(block=block)
    if ua is not None:
        core = IU.CoreRadiator(u_air=np.ones(WA.ROTOR_CELLS))
        core.u_air_ref = 1.0
        core.u_air = np.full(WA.ROTOR_CELLS,
                             (float(ua) / CL.UA_RAD) ** (1.0 / IU.UA_EXPONENT))
        loop.legs["RAD"] = core
    t_in = CL.T_COOLANT_0
    rows = []
    t0 = time.perf_counter()
    held = None
    for s in range(steps):
        if mounted:
            if s % max(1, int(lag)) == 0:
                tw = float(np.mean(block._face_T(block._T, "outer")))
                held = tw + block.q_machine / (block.h_mount * block.area)
            block.t_case = held
        row = _coolant_step(loop, block, t_in)
        t_in = row["t_in"]
        row["step"] = s
        rows.append(row)
    return LoopMarch(steps=steps, rows=rows, wall_s=time.perf_counter() - t0,
                     notes={"mounted": mounted, "q_machine": q_machine,
                            "ua": ua, "lag": lag,
                            "dt_s": CL.MACRO_DT,
                            "horizon_s": steps * CL.MACRO_DT})


# ===========================================================================
# DECISIONS 3 and 4 -- the receiver balances, and the rotor's power path
# ===========================================================================


def receiver_balances(m: UnionMarch, frac: float = 0.25) -> dict[str, Any]:
    """Each join's receiving subsystem against its OWN balance, over the march.

    **This is the alternative decision 3 takes (W200).**  A global `R(t)` cannot
    be assembled from the union's records -- no field carries dissipation, 13 of
    18 agents dissipate by their own constants, ten window faces are the fluid
    domain's physical boundaries and carry energy through no port, and the three
    clocks were in three unit systems until `VehicleScale`.  So no declaration
    is added and each receiver's balance is written by hand, which is what CS-9,
    CS-12 and the front-wing demo each did.  **The hole is named and not
    filled**: `ExpertCapabilities` still has no dissipation field and a tiling's
    domain boundaries are still not ports.

    J3's balance is the one with a real receiver: the power the flow gives up
    at the rotor's plane, measured as the work the BODY FORCE does on the fluid,
    against the power the shaft claims.
    """
    n = max(1, int(round(frac * m.steps)))
    tr = m.trace
    out: dict[str, Any] = {}

    # -- J3 ----------------------------------------------------------------
    w_fluid = -float(np.mean(tr["power_on_the_fluid"][-n:]))   # taken OUT
    p_shaft = float(np.mean(tr["shaft_power"][-n:]))
    p_core = float(np.mean(tr["core_flow_power"][-n:]))
    w_rotor = -float(np.mean(tr["power_rotor"][-n:]))
    w_core = -float(np.mean(tr["power_core"][-n:]))
    u_ring = float(np.mean(tr["u_rotor"][-n:]))
    u_plane = float(np.mean(tr["u_at_the_rotor_plane"][-n:]))
    out["J3"] = {
        "receiver": "the powertrain's shaft",
        "the join's term: work the rotor's body force does on the fluid": w_rotor,
        "the shaft's claim, T <U_d>": p_shaft,
        "ratio_with_the_term": (w_rotor / p_shaft) if p_shaft else None,
        "residual_with_the_term": (abs(w_rotor - p_shaft) / abs(p_shaft)
                                   if p_shaft else None),
        "residual_without_the_term": 1.0 if p_shaft else None,
        #: the two velocities the balance turns on: the disk reads its inflow
        #: on a ring 8 cells UPSTREAM of the plane (`disk.disk_average`'s own
        #: rule, to avoid reading a velocity its force has already slowed),
        #: while the force does its work inside the strip, where it has.
        "u on the ring the disk reads": u_ring,
        "u in the cell the sink sits in": u_plane,
        "u_plane / u_ring": (u_plane / u_ring) if u_ring else None,
        "both devices' work on the fluid": w_fluid,
        "the core's share of it": w_core,
        "horizon_macro_steps": n,
    }

    # -- J1 ----------------------------------------------------------------
    ua = float(np.mean(tr["ua"][-n:]))
    u_core = float(np.mean(tr["u_core"][-n:]))
    ua0 = float(tr["ua"][0]) if tr["ua"].size else CL.UA_RAD
    u0 = float(tr["u_core"][0]) if tr["u_core"].size else 1.0
    pred = ua0 * (u_core / u0) ** IU.UA_EXPONENT if u0 else None
    out["J1"] = {
        "receiver": "the coolant circuit's radiator",
        "no power crosses": True,
        "the join's term: UA follows the air": ua,
        "UA at release": ua0,
        "air through the core, release -> settled": [u0, u_core],
        "UA predicted by the declared exponent": pred,
        "tracking_residual": (abs(ua - pred) / abs(pred)
                              if pred not in (None, 0.0) else None),
        "the air's mechanical loss across the core, unaccounted": p_core,
    }

    # -- J2 ----------------------------------------------------------------
    if m.coolant:
        last = m.coolant[-1]
        out["J2"] = {
            "receiver": "the cooled block",
            "coolant steps taken": len(m.coolant),
            "block first law, relative": last["block_first_law"].get("relative"),
            "q_machine_W": float(np.mean(tr["q_machine"][-n:])),
            "t_wall_K": last["t_wall_mean"],
            "loop residual": last["loop_residual"],
        }
    else:
        out["J2"] = {"receiver": "the cooled block",
                     "coolant steps taken": 0,
                     "block first law, relative": None,
                     "not_measured_because":
                         "the march is shorter than one coolant step"}
    return out


def rotor_power_paths(m: UnionMarch, frac: float = 0.25) -> dict[str, Any]:
    """W197, seen on a march: what the two declared PORTS carry against the force.

    Tier 47 measured the two declared faces carrying exactly zero net power at
    the rotor's own base and 0.0077 at the two host rings, against a shaft power
    of 0.785 -- *"the power the shaft delivers enters through no declared
    port"*.  A march does not go through the ports at all: the device is a body
    force (decision 5), so the power path is the force's work on the fluid and
    is read off that.  **That dissolves W197 for a march and leaves it open for
    a compile**, and the number below is the size of the gap the compile has.
    """
    n = max(1, int(round(frac * m.steps)))
    tr = m.trace
    return {
        "power the body force takes out of the fluid": -float(
            np.mean(tr["power_on_the_fluid"][-n:])),
        "the shaft's claim": float(np.mean(tr["shaft_power"][-n:])),
        "what the two declared ports carry at the rotor's own base": 0.0,
        "the march reads the power path from the force, not the ports": True,
        "so W197 blocks a compile's accounting and not a march": True,
    }


# ===========================================================================
# the gate, PRE-REGISTERED
# ===========================================================================

#: **Rung 9's gate for a march, as numbers, fixed before any arm ran.**
#:
#: Written after the instrument runs of stage ``calib`` -- a 40-macro-step
#: march, its bitwise repeat, and the settling report -- and BEFORE the
#: 1200-step arms and the second vehicle scale, which are the out-of-sample
#: cells.  That is `defect-correction-learned-operator` section 7's discipline,
#: and its weakness here is named on the page: this graph has one geometry, so
#: "out of sample" means a longer horizon and a different clock ratio, not a
#: different rung.
#:
#: **Every threshold here is justified by a number that existed before this
#: tier**, so that none of them is the answer read backwards:
#:
#:   G1  0.075 is the inflow non-uniformity `rung9-gate-restated` section 4.2
#:       measured at the release state and declared *a property of the model,
#:       not a leak*.  The march is held to the same size.
#:   G2  1e-6 and 1e-2 bracket Tier 47's release-state pair, 3.0e-8 with the
#:       mount term and 0.269 without it, with two orders of margin on each.
#:   G3  1e-6 is an algebraic identity's tolerance: ``UA(u)`` IS
#:       ``UA_RAD (u/u_ref)^0.8`` and nothing else, so anything above round-off
#:       means the march is not evaluating the declaration it says it is.
#:   G4  10x the repeat floor is this vault's standing rule that a difference
#:       smaller than the instrument's own floor is not a signal.
#:   G5  0.25 and 2.0 are stated so that "the errors add" and "the errors
#:       compound" are decidable, and so that NEITHER is a reportable outcome
#:       rather than a default to the convenient one.
#:
#: `tests/test_tier49_union_march.py` asserts every number below appears in
#: `case-study-vehicle-march-atlas-0.1.md`, because a criterion written in prose
#: and evaluated in code drifts, and Tier 48 measured that it drifts permissive.
GATE: dict[str, dict[str, Any]] = {
    "G1": {
        "clause": "J3's receiving balance, over a march",
        "measured_as": "the work the rotor's body force does on the fluid over "
                       "the settle window, against the shaft power the union's "
                       "shared operating point claims",
        "passes_if": "relative residual <= 0.075 with the join's term, and "
                     ">= 0.5 both without it and when the shaft is charged "
                     "against the OTHER device's work",
        "tol_with": 0.075, "tol_without": 0.5,
    },
    "G2": {
        "clause": "J2's receiving balance, over a march",
        "measured_as": "the block's first law over the coolant circuit's own "
                       "march, with the mount term and without it",
        "passes_if": "relative residual <= 1e-6 with the mount term and "
                     ">= 1e-2 without it",
        "tol_with": 1.0e-6, "tol_without": 1.0e-2,
    },
    "G3": {
        "clause": "J1's parametric check, over a march",
        "measured_as": "UA against the declared exponent's prediction from the "
                       "measured air, over the march; the null holds UA at UA_RAD",
        "passes_if": "tracking residual <= 1e-6 with the join's term, and UA "
                     "exactly constant without it while the air moves",
        "tol_with": 1.0e-6, "tol_without": 0.0,
    },
    "G4": {
        "clause": "composition error, per join, at vehicle scale",
        "measured_as": "the crossing vector's relative l2 error of each "
                       "single-join-lagged arm against the all-tight referent",
        "passes_if": "each join's error exceeds 10x the repeat floor, or is "
                     "reported NOT RESOLVED rather than as a number",
        "floor_multiple": 10.0,
    },
    "G5": {
        "clause": "do the joins' errors add",
        "measured_as": "the all-lagged arm's error against the sum of the "
                       "single-join-lagged arms' errors",
        "passes_if": "ADD if |e_all - sum(e_i)| <= 0.25 sum(e_i); COMPOUND -- "
                     "Claim B failing at vehicle scale -- if e_all > 2.0 "
                     "sum(e_i); otherwise NEITHER, reported as neither",
        "add_tol": 0.25, "compound_factor": 2.0,
    },
    "G6": {
        "clause": "the repeat floor",
        "measured_as": "the referent marched twice as two independent "
                       "constructions in one process -- a fresh rollout, "
                       "solver, circuit and block each time. Cross-process "
                       "reproducibility is NOT tested here",
        "passes_if": "bitwise identical in every crossing quantity, and the "
                     "floor quoted beside every difference G4 reports",
    },
}

#: What this tier predicted, recorded with the gate and before the arms ran.
PREDICTION = (
    "G1 passes, at a residual LARGER than the 0.030 the 40-step instrument run "
    "showed, because the rotor's own wake deepens over a longer horizon and the "
    "whole residual is the gap between the ring the disk reads and the cell its "
    "force sits in. G2 and G3 pass. G4 resolves all three joins. G5 returns "
    "ADD rather than COMPOUND, because two of the three joins are one-way and "
    "the third is algebraic, so there is no loop for an error to go round -- "
    "and COMPOUND would have been the more valuable result, since it is Claim B "
    "failing at vehicle scale, which f1-pathmap-and-end-goal section 7 calls "
    "the most likely failure on the ladder."
)


# ===========================================================================
# DECISION 7 -- L7/R9 scoped to the multirate seams (W194)
# ===========================================================================


def multirate_seams(graph) -> dict[str, Any]:
    """Which seams actually join two clocks, and which agents sit at one.

    `CaseGraph.is_multirate` reads every agent's native step, so the DISJOINT
    union -- in which no seam joins two clocks -- is refused; and `L7/R9`'s
    time-integrated branch asks for `boundary_response_integrated` from all
    eighteen agents where the joined union's five multirate seams touch eight.
    `FluxMatching`'s own docstring states the requirement as every agent AT a
    multirate seam.  This computes the seam-scoped answer, which
    `atlas/compiler.py` now reads (W194).
    """
    dt = {a.agent_id: a.capabilities.dt_native for a in graph.agents}
    seams, agents = [], set()
    for c in graph.connections:
        a, b = dt.get(c.a[0]), dt.get(c.b[0])
        if a is None or b is None or a <= 0.0 or b <= 0.0:
            continue
        if abs(a - b) > 1e-12 * max(a, b):
            seams.append(c.seam_id)
            agents.update((c.a[0], c.b[0]))
    return {"multirate_seams": sorted(seams),
            "agents_at_a_multirate_seam": sorted(agents),
            "n_seams": len(seams), "n_agents": len(agents),
            "n_agents_in_the_graph": len(graph.agents),
            "graph_is_multirate_by_agents": graph.is_multirate(),
            "graph_is_multirate_by_seams": bool(seams)}
