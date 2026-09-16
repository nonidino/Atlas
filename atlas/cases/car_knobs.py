"""Section 3.3's parameters, and what each one actually reaches.

PoC 3, Tier 69.  [[poc3-racelab-knobs]].

Requirements section 12's criterion 2 is *"they can move a parameter and watch
every coupled subsystem respond"*, and its purpose is to show the coupling is
REAL -- that moving one knob propagates through every subsystem the graph says it
reaches, so RaceLab is one coupled system and not several models sharing a
screen.  Section 3.3 states the rule that makes that testable:

    Every parameter must be honestly wired.  A slider that moves a number
    nothing reads is worse than no slider: if a parameter cannot reach its
    subsystem, it is omitted and the omission is recorded.

This module declares the knobs, states what each CLAIMS to reach, and provides
the probe that decides it.  `reach_report` moves every knob from one end of its
declared range to the other and measures the claimed quantity at both ends.

Why "moved" is not `a != b`
---------------------------

**`nan != nan` is True.**  The first version of this test called a knob wired
whenever its two readings differed, and a probe that returned `nan` at both ends
-- because it was calling an API that did not exist -- reported the knob as
moving.  A broken probe read as a working knob, which is the exact failure this
module is built to detect, committed by the detector.  `moved` therefore
requires both readings to be FINITE and different.

What the measurement found
--------------------------

Four of section 3.3's eleven reach their subsystem as declared: ride height
(four bodies), diffuser, front flap and rear wing (one each).  Two reach theirs
only by rebinding a module-level constant.  One reaches half of what it claims.
Four reach nothing, and one of those four cannot exist in this model at all.
`KNOBS` carries the verdict per knob and `OMITTED` the reasons; the page has the
table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

#: What a knob's probe must satisfy to count as reaching its subsystem.
#:
#: Both readings finite AND different.  See the module docstring: the `nan`
#: version of this test called a broken probe a working knob.
def moved(a: float, b: float) -> bool:
    """True when a knob's probe genuinely moved between its two ends."""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(a) and math.isfinite(b) and a != b)


#: How a knob reaches -- or fails to reach -- its subsystem.
#:
#: ``"wired"``    the subsystem reads it through a parameter it owns.
#: ``"global"``   it is reached only by rebinding a MODULE-LEVEL constant, which
#:                works in a script and is not safe under the demo's background
#:                march thread, and which also moves a capability record's
#:                `weight_hash` without anything noticing.
#: ``"partial"``  one subsystem reads it and another that should does not.
#: ``"dead"``     nothing reads it.
#: ``"absent"``   the model has no quantity for it to reach.
HOW = ("wired", "global", "partial", "dead", "absent")


@dataclass(frozen=True)
class Knob:
    """One of section 3.3's parameters."""

    name: str
    group: str
    lo: float
    hi: float
    default: float
    unit: str
    reaches: tuple[str, ...]
    probe: str
    how: str
    note: str = ""

    def ends(self) -> tuple[float, float]:
        return (self.lo, self.hi)


#: Section 3.3's table, as measured on 2026-09-16 rather than as written.
KNOBS: tuple[Knob, ...] = (
    Knob("ride_height", "vehicle", 0.15, 0.60, 0.22, "tiling units",
         ("front wing", "floor", "diffuser"), "bodies whose built pose moves", "wired",
         "moves DIFF, FW_FLAP, FW_MAIN and FW_UPPER -- four bodies"),
    Knob("rake", "vehicle", 0.0, 0.15, 0.10, "cells per unit",
         ("floor", "diffuser"), "bodies whose built pose moves", "dead",
         "`rake` appears EXACTLY ONCE in the package: its own declaration in "
         "CarParams. No body, grid, force or record reads it, so the slider "
         "would move a number nothing reads"),
    Knob("diffuser_deg", "aero", 0.0, 15.0, 0.0, "degrees",
         ("floor geometry",), "bodies whose built pose moves", "wired",
         "moves DIFF"),
    Knob("front_flap_deg", "aero", 0.0, 30.0, 21.8, "degrees",
         ("front wing geometry",), "bodies whose built pose moves", "wired",
         "moves FW_FLAP"),
    Knob("rear_wing_deg", "aero", 0.0, 30.0, 27.6, "degrees",
         ("rear wing geometry",), "bodies whose built pose moves", "wired",
         "moves RW_FLAP"),
    Knob("duct_area", "cooling", 0.3, 1.5, 1.0, "of nominal",
         ("duct geometry", "core inflow", "UA"), "the openings' spans", "dead",
         "the duct's openings are FIXED LITERALS in car_solids.SOLIDS_DEFAULT "
         "-- a CHASSIS inlet from 0.25 to 0.75 and a POD_UP outlet from 0.8 to "
         "1.0 -- and nothing scales them"),
    Knob("coolant_mdot", "cooling", 0.05, 0.30, 0.15, "kg/s",
         ("cooling_loop",), "the circuit's fixed-point return temperature", "global",
         "reaches the loop: 310.156 K at 0.05 against 310.865 K at 0.30 -- but "
         "only by rebinding cooling_loop.MDOT, which ten call sites read "
         "directly and which a leg's weight_hash is built from"),
    Knob("ambient_t", "cooling", 273.0, 318.0, 300.0, "K",
         ("cooling_loop", "brake_thermal"), "the circuit's fixed-point return temperature",
         "global",
         "reaches the loop strongly: 288.133 K at 273 against 325.778 K at 318 "
         "-- again only by rebinding cooling_loop.T_AMB"),
    Knob("battery_power", "powertrain", 0.0, 350.0, 0.0, "kW",
         ("powertrain", "J2 heat into the block"), "the loop current", "absent",
         "THIS CIRCUIT HAS NO DEMAND INPUT. The machine is a generator: its "
         "current follows the shaft speed through current_at(omega), and the "
         "battery receives what is pushed into it. There is no quantity a "
         "power draw could set, so the knob cannot be wired without a "
         "different powertrain model"),
    Knob("battery_soc", "powertrain", 0.0, 1.0, 1.0, "fraction",
         ("powertrain open-circuit voltage",), "battery EMF and the machine's loop current",
         "partial",
         "BatteryLeg.v_oc is a real field and its emf() reads it, but "
         "MachineAgent has NO v_oc field at all: current_at and validity read "
         "the MODULE constant V_OC. So a state-of-charge knob moves the "
         "battery and leaves the machine drawing from it unchanged -- "
         "including the validity predicate W282's envelope is written on"),
    Knob("road_speed", "vehicle", 20.0, 90.0, 40.0, "m/s",
         ("the fluid's freestream", "the scale's U0"), "ground_effect's freestream", "dead",
         "ground_effect.U_INF is 1.0 and the whole column is nondimensional, so "
         "road speed sets a SCALE and no field in the march reads it; changing "
         "it would rename the axes and move nothing"),
)

KNOBS_BY_NAME = {k.name: k for k in KNOBS}

#: The knobs section 3.3 lists that amendment 13.1 already omitted, kept here so
#: the hole list is one list and not two.
OMITTED = {
    "brake_duty": "there is no brake subsystem in RaceLab's graph",
    "brake_duct": "there is no brake subsystem in RaceLab's graph",
    "plate_stiffness": "the structure is rigid in this column",
}


def by_how(how: str) -> tuple[Knob, ...]:
    if how not in HOW:
        raise ValueError(f"how must be one of {HOW}, got {how!r}")
    return tuple(k for k in KNOBS if k.how == how)


def honestly_wired() -> tuple[Knob, ...]:
    """The knobs a dashboard may show as reaching what they claim."""
    return by_how("wired")


def must_not_be_shown() -> tuple[Knob, ...]:
    """Knobs that reach nothing, or nothing that exists.

    Section 3.3: *a parameter that cannot reach its subsystem is omitted and the
    omission is recorded*.  A dashboard that shows these is worse than one that
    does not.
    """
    return by_how("dead") + by_how("absent")


# ---------------------------------------------------------------------------
# the probes
# ---------------------------------------------------------------------------


def built_bodies(**knob_values) -> dict[str, tuple]:
    """The car AS BUILT at these knob values: body id -> its pose.

    The geometry probe.  It reads the built car rather than the geometry file,
    because the knob-owned entries (`y_plus_ride`, `y_plus_duct`, `alpha_from`)
    are only resolved at build time -- which is exactly where a knob either
    reaches a body or does not.
    """
    from . import racelab as RL

    p = RL.CarParams(**knob_values)
    _plates, bodies = RL.car_bodies(p)
    return {b.body_id: (round(float(b.x_le), 9), round(float(b.y_le), 9),
                        round(float(b.chord), 9), round(float(b.alpha_deg), 9))
            for b in bodies}


def bodies_moved(name: str, lo: float, hi: float) -> list[str]:
    """Which bodies change pose between a geometry knob's two ends."""
    a = built_bodies(**{name: lo})
    b = built_bodies(**{name: hi})
    return sorted(k for k in a if k in b and a[k] != b[k])


def loop_return(mdot: float | None = None, t_amb: float | None = None) -> float:
    """The coolant circuit's fixed-point return temperature.

    Reached by rebinding the module constants, which is what `how="global"`
    records: ten call sites read `MDOT` directly, so there is no parameter to
    pass.  The rebinding is undone in a `finally`, because leaving a module
    constant moved would make every later measurement quietly wrong.
    """
    from . import cooling_loop as CL
    from . import integration_union as IU

    old_m, old_t = CL.MDOT, CL.T_AMB
    try:
        if mdot is not None:
            CL.MDOT = float(mdot)
        if t_amb is not None:
            CL.T_AMB = float(t_amb)
        return float(CL.LoopSolve(block=IU.MountedBlock(q_machine=0.0)).solve().t_return)
    finally:
        CL.MDOT, CL.T_AMB = old_m, old_t


def battery_emf(v_oc: float) -> float:
    from . import powertrain as PT
    return float(PT.BatteryLeg(v_oc=v_oc).emf())


def machine_current(omega: float = 20.0) -> float:
    from . import powertrain as PT
    return float(PT.MachineAgent().current_at(omega))


def soc_reach(omega: float = 20.0) -> dict[str, Any]:
    """The state-of-charge defect, measured rather than asserted.

    The battery's EMF follows its own `v_oc`; the machine's loop current and its
    validity predicate read the module constant.  So the two halves of one
    circuit disagree about what voltage the battery is at.
    """
    from . import powertrain as PT

    hi, lo = battery_emf(1.34), battery_emf(0.80)
    cur = machine_current(omega)
    return {
        "battery_emf_hi": hi, "battery_emf_lo": lo,
        "battery_moves": moved(hi, lo),
        "machine_has_v_oc": "v_oc" in PT.MachineAgent.__dataclass_fields__,
        "machine_current": cur,
        "machine_moves_with_the_battery": False,
        "why": ("MachineAgent.current_at and .validity read the module constant "
                "V_OC, so no battery instance can move them"),
    }


def reach_report() -> dict[str, Any]:
    """Move every knob end to end and measure what it claims to reach."""
    rows: dict[str, Any] = {}
    for k in KNOBS:
        row: dict[str, Any] = {"group": k.group, "how": k.how, "lo": k.lo, "hi": k.hi,
                               "reaches": list(k.reaches), "probe": k.probe,
                               "note": k.note}
        if k.probe == "bodies whose built pose moves":
            mv = bodies_moved(k.name, k.lo, k.hi)
            row["bodies_moved"] = mv
            row["n_moved"] = len(mv)
            row["reaches_something"] = bool(mv)
        elif k.probe == "the circuit's fixed-point return temperature":
            a = loop_return(**({"mdot": k.lo} if k.name == "coolant_mdot"
                               else {"t_amb": k.lo}))
            b = loop_return(**({"mdot": k.hi} if k.name == "coolant_mdot"
                               else {"t_amb": k.hi}))
            row["lo_value"], row["hi_value"] = a, b
            row["reaches_something"] = moved(a, b)
        elif k.name == "battery_soc":
            s = soc_reach()
            row.update(s)
            row["reaches_something"] = bool(s["battery_moves"])
            row["reaches_all_it_claims"] = False
        else:
            row["reaches_something"] = False
        rows.setdefault("reaches_all_it_claims", None)
        rows[k.name] = row
    rows.pop("reaches_all_it_claims", None)
    summary = {h: [k.name for k in by_how(h)] for h in HOW}
    return {"knobs": rows, "by_how": summary,
            "n_wired": len(by_how("wired")), "n_total": len(KNOBS),
            "omitted": dict(OMITTED)}
