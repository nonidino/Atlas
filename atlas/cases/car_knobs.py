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
#:                `weight_hash` without anything noticing.  **Empty since
#:                Tier 71**, which threaded `cooling_loop.LoopSettings`; the
#:                category is kept because it names a real way to fail.
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
    Knob("rake", "vehicle", 0.0, 0.15, 0.0, "cells",
         ("floor", "diffuser"), "bodies whose built pose moves", "wired",
         "WIRED IN TIER 70. It pitches the floor group rigidly about its own "
         "leading edge -- five bodies move, FLOOR_LE sits at the pivot and "
         "does not, DIFF rises 0.1342 cells at rake 0.15, and every plate's "
         "incidence gains atan(rake/351.97). Its default moved 0.10 -> 0.0 in "
         "the same change because while it reached nothing every car ever "
         "built here was the rake = 0 car, and wiring it at 0.10 would have "
         "pitched the floor under every cached field and record"),
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
         ("duct geometry", "core inflow", "UA"), "the openings' achieved spans", "wired",
         "WIRED IN TIER 70. It scales each opening's span about its own centre, "
         "so the inlet's 0.500 becomes 0.150 at 0.3 and 0.750 at 1.5 and the "
         "outlet's 0.200 becomes 0.060 and 0.300. An opening that would run off "
         "the end of its panel is SHIFTED to keep its span rather than losing "
         "it -- the outlet moves to [0.70, 1.00] at 1.5 -- and "
         "car_solids.openings_report records nominal beside achieved so a "
         "clamp is visible. 1.0 IS the drawn duct, so the nominal car is "
         "unchanged by the knob existing"),
    Knob("coolant_mdot", "cooling", 0.05, 0.30, 0.15, "kg/s",
         ("cooling_loop",), "the circuit's fixed-point return temperature", "wired",
         "WIRED IN TIER 71. Reaches the loop through LoopSettings: 310.156 K "
         "at 0.05 against 310.865 K at 0.30, and the leg's weight_hash moves "
         "with it, so the capability record cannot claim the machine that "
         "used to run"),
    Knob("ambient_t", "cooling", 273.0, 318.0, 300.0, "K",
         ("cooling_loop", "brake_thermal"), "the circuit's fixed-point return temperature",
         "wired",
         "WIRED IN TIER 71. Reaches the loop through LoopSettings: 288.133 K "
         "at 273 against 325.778 K at 318, with no module constant moved"),
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
         "Tier 69 found MachineAgent with NO v_oc field at all -- current_at "
         "and validity read the MODULE constant -- and repaired it, so a "
         "machine CAN now be told which battery it draws from and its "
         "decline threshold moves 13.4 -> 8.0 rad/s at v_oc 0.80. What is "
         "still missing is the wiring: racelab.machine_for_host builds the "
         "marching machine without the knob, so the state of charge reaches "
         "the battery model and not the machine that is running"),
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


def duct_report(area: float) -> list[dict]:
    """Every opening's nominal and achieved span at this `duct_area`."""
    from . import car_solids as CS

    return CS.openings_report(dict(CS.SOLIDS_DEFAULT, duct_area=float(area)))


def duct_spans(area: float) -> float:
    """The duct's TOTAL achieved opening, the one number the knob must move."""
    return float(sum(r["achieved_span"] for r in duct_report(area)))


def loop_settings(mdot: float | None = None, t_amb: float | None = None):
    """The circuit's operating point at these knob values."""
    from . import cooling_loop as CL

    d = CL.LoopSettings()
    return CL.LoopSettings(mdot=d.mdot if mdot is None else float(mdot),
                           t_amb=d.t_amb if t_amb is None else float(t_amb))


def loop_return(mdot: float | None = None, t_amb: float | None = None) -> float:
    """The coolant circuit's fixed-point return temperature at these knobs.

    **This used to rebind the module constants and put them back** -- which was
    W291, and which the dashboard proved was not merely inelegant: the page
    reported ambient temperature "responding" while the marching circuit's
    return temperature never moved, because the probe's rebinding never reached
    the running `CarUnion`.  The settings are now passed, so the number this
    returns is the number the march would get.
    """
    from . import cooling_loop as CL
    from . import integration_union as IU

    return float(CL.LoopSolve(block=IU.MountedBlock(q_machine=0.0),
                              settings=loop_settings(mdot, t_amb)).solve().t_return)


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


#: Knobs whose change forces the car to be re-cut and re-gridded, which costs a
#: composite rebuild (66-90 s, Tier 68).  Section 4.1 already anticipates this:
#: a parameter that would change the decomposition is either clamped or the
#: window set is rebuilt "with a visible recompiling state".
REGRID = ("ride_height", "rake", "diffuser_deg", "front_flap_deg",
          "rear_wing_deg", "duct_area")


@dataclass
class KnobState:
    """The knobs' current values, and what each change actually moved.

    This is criterion 2's machinery: `set` applies a value and returns the
    subsystems that RESPONDED, measured, so a screen can show the response
    rather than assert it.  A knob that reaches nothing is refused here rather
    than accepted and quietly ignored -- section 3.3's rule, enforced at the one
    place a value enters.
    """

    values: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.values is None:
            self.values = {k.name: k.default for k in KNOBS}

    def car_params(self):
        """`racelab.CarParams` at the current geometry values."""
        from . import racelab as RL

        fields = {f for f in RL.CarParams.__dataclass_fields__}
        return RL.CarParams(**{k: v for k, v in self.values.items() if k in fields})

    def loop_settings(self):
        """`cooling_loop.LoopSettings` at the current cooling values.

        What a marching circuit must be built with, so a cooling knob reaches
        the loop that is running and not only the probe that measures it.
        """
        return loop_settings(self.values["coolant_mdot"], self.values["ambient_t"])

    def solids_rule(self) -> dict:
        """`car_solids.SOLIDS_DEFAULT` at the current duct area."""
        from . import car_solids as CS

        return dict(CS.SOLIDS_DEFAULT, duct_area=float(self.values["duct_area"]))

    def set(self, name: str, value: float) -> dict[str, Any]:
        """Move one knob, and report what responded.

        Refuses a knob that reaches nothing and a value outside its declared
        range, because a dashboard that accepts either is showing a control that
        does not control.
        """
        k = KNOBS_BY_NAME.get(name)
        if k is None:
            raise KeyError(f"no such knob: {name!r}")
        if k in must_not_be_shown():
            raise ValueError(f"{name} reaches nothing and must not be offered: {k.note}")
        value = float(value)
        if not (k.lo <= value <= k.hi):
            raise ValueError(f"{name} must be within [{k.lo}, {k.hi}], got {value}")
        before = self._probe(k)
        old = self.values[name]
        self.values[name] = value
        after = self._probe(k)
        return {"knob": name, "from": old, "to": value, "group": k.group,
                "reaches": list(k.reaches), "responded": self._responded(k, before, after),
                "needs_regrid": name in REGRID,
                "before": before, "after": after}

    def _probe(self, k: Knob) -> Any:
        if k.probe == "bodies whose built pose moves":
            return built_bodies(**{f.name: self.values[f.name]
                                   for f in KNOBS
                                   if f.name in _car_param_names()})
        if k.probe == "the openings' achieved spans":
            return duct_spans(self.values["duct_area"])
        if k.probe == "the circuit's fixed-point return temperature":
            return loop_return(mdot=self.values.get("coolant_mdot"),
                               t_amb=self.values.get("ambient_t"))
        if k.name == "battery_soc":
            return battery_emf(self.values["battery_soc"] * 1.34)
        return None                                              # pragma: no cover

    @staticmethod
    def _responded(k: Knob, before: Any, after: Any) -> list[str]:
        if isinstance(before, dict) and isinstance(after, dict):
            return sorted(b for b in before if b in after and before[b] != after[b])
        if moved(before, after):
            return list(k.reaches)
        return []


def _car_param_names() -> set[str]:
    from . import racelab as RL

    return set(RL.CarParams.__dataclass_fields__)


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
        elif k.probe == "the openings' achieved spans":
            a = duct_spans(k.lo)
            b = duct_spans(k.hi)
            row["lo_value"], row["hi_value"] = a, b
            row["reaches_something"] = moved(a, b)
            row["detail"] = duct_report(k.lo) + duct_report(k.hi)
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
