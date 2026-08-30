"""The closed port vocabulary, and what a scale set must contain.

port-algebra-atlas-0.1 §3.  Five port types, and the list is closed: a sixth
enters only through the six-field ``PortAmendment`` procedure, which is a named
hole (holes.PORT_AMENDMENT) rather than a rule anyone has exercised.

Every port is a power bond: an effort and a flow whose product is a power.  Two
consequences this module implements, and both are compile-time checks that cost
a multiplication:

  * **Scale-set completeness is decidable per port type.**  MECH needs a stress
    scale and a velocity scale; THERM a temperature scale and a heat-flux scale;
    ADVEC one pair per passenger.  So the compiler can determine, from the port
    list alone, exactly which scales a record must contain and refuse one that
    is missing a scale, before any solve.  (interface-transfer-theory §6)

  * **The power identity s_e * s_f = s_P.**  Converting effort and flow with
    independently chosen scales is dimensionally plausible and silently destroys
    the power bond, after which the global power residual reports a unit error
    in watts.  Refused at compile time from the declaration alone.

Nothing in this module knows any physics beyond the bond structure.  A new case
study adds no code here.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .holes import PORT_AMENDMENT, NamedHoleError


class PortType(enum.Enum):
    """The closed vocabulary. Adding a member is a PortAmendment, not an edit."""

    MECH = "MECH"
    ROT = "ROT"
    THERM = "THERM"
    ELEC = "ELEC"
    ADVEC = "ADVEC"


class Role(enum.Enum):
    """Which half of the bond a quantity is.

    The mapping class follows from this and from the declared prolongation ---
    it is never declared.  See transfer.mapping_class.
    """

    EFFORT = "effort"
    FLOW = "flow"


class MappingClass(enum.Enum):
    """Derived, never declared. interface-transfer-theory §2.2."""

    CONSISTENT = "consistent"        # interpolation; reproduces constants; efforts
    CONSERVATIVE = "conservative"    # preserves integrals; the adjoint; flows


@dataclass(frozen=True)
class PortSpec:
    """The bond structure of one port type."""

    port_type: PortType
    effort: str
    flow: str
    effort_unit: str
    flow_unit: str
    power_unit: str
    effort_scale_key: str
    flow_scale_key: str
    power_scale_key: str
    lumped: bool                 # a ROT port is lumped: dim M = 1 by nature
    multibond: bool = False      # ADVEC only
    note: str = ""

    def required_scale_keys(self, passengers: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Exactly which scales a record must carry for this port.

        For ADVEC the passenger list extends the requirement: one conjugate pair
        per passenger.  That is also the leading indicator of vocabulary
        pressure the PortAmendment slot asks to be counted.
        """
        keys = [self.effort_scale_key, self.flow_scale_key, self.power_scale_key]
        if self.multibond:
            for p in passengers:
                keys.extend([f"{p}_effort", f"{p}_flow", f"{p}_power"])
        return tuple(keys)


PORT_SPECS: dict[PortType, PortSpec] = {
    PortType.MECH: PortSpec(
        port_type=PortType.MECH,
        effort="traction  t = sigma . n",
        flow="velocity  v",
        effort_unit="Pa",
        flow_unit="m/s",
        power_unit="W/m^2",
        effort_scale_key="stress",
        flow_scale_key="velocity",
        power_scale_key="power_area",
        lumped=False,
        note=(
            "Absorbs the old pressure / stress / shear / momentum labels exactly: "
            "pressure is the isotropic part of the stress tensor, viscous and elastic "
            "stress the deviatoric part, shear the tangential component of one traction."
        ),
    ),
    PortType.ROT: PortSpec(
        port_type=PortType.ROT,
        effort="torque",
        flow="angular velocity",
        effort_unit="N m",
        flow_unit="1/s",
        power_unit="W",
        effort_scale_key="torque",
        flow_scale_key="angular_velocity",
        power_scale_key="power",
        lumped=True,
        note="Lumped rotating machinery. A lumped port is the case dim M = 1.",
    ),
    PortType.THERM: PortSpec(
        port_type=PortType.THERM,
        effort="temperature",
        flow="entropy flux  q_n / T",
        effort_unit="K",
        flow_unit="W/(m^2 K)",
        power_unit="W/m^2",
        effort_scale_key="temperature",
        flow_scale_key="entropy_flux",
        power_scale_key="power_area",
        lumped=False,
        note=(
            "The TRUE bond (T, q_n/T), not the pseudo-bond (T, q_n) most co-simulation "
            "codes exchange. The pseudo-bond's product is not a power and cannot enter "
            "the global power residual; the adapter between them is one division."
        ),
    ),
    PortType.ELEC: PortSpec(
        port_type=PortType.ELEC,
        effort="potential",
        flow="current density  j . n",
        effort_unit="V",
        flow_unit="A/m^2",
        power_unit="W/m^2",
        effort_scale_key="potential",
        flow_scale_key="current_density",
        power_scale_key="power_area",
        lumped=False,
    ),
    PortType.ADVEC: PortSpec(
        port_type=PortType.ADVEC,
        effort="total enthalpy  h0   (chemical potential per species)",
        flow="mass flux  rho u . n",
        effort_unit="J/kg",
        flow_unit="kg/(m^2 s)",
        power_unit="W/m^2",
        effort_scale_key="enthalpy",
        flow_scale_key="mass_flux",
        power_scale_key="power_area",
        lumped=False,
        multibond=True,
        note=(
            "Deliberately not a simple effort-flow pair: a mass flux carries several "
            "conserved quantities at once. The variable passenger list is where "
            "case-study-specific growth reappears if it is not disciplined."
        ),
    ),
}


def spec_for(port_type: PortType | str) -> PortSpec:
    """Look up a port type, refusing anything outside the closed vocabulary."""
    if isinstance(port_type, PortType):
        return PORT_SPECS[port_type]
    try:
        return PORT_SPECS[PortType(port_type)]
    except ValueError as exc:
        raise NamedHoleError(
            PORT_AMENDMENT,
            f"port type {port_type!r} is not one of "
            f"{[t.value for t in PortType]}; a sixth type enters only through the "
            "six-field amendment procedure, which has never been exercised",
        ) from exc


@dataclass(frozen=True)
class ScaleCheck:
    """The outcome of checking one port's nondimensionalization declaration."""

    complete: bool
    missing: tuple[str, ...]
    power_identity_residual: float | None   # |s_e * s_f / s_P - 1|
    ok: bool
    detail: str = ""


#: Relative tolerance on the power identity.  A scale set is a human-entered
#: declaration, so this is a "did you mean it" tolerance, not a numerical one.
POWER_IDENTITY_TOL = 1e-9


def check_scales(
    port_type: PortType,
    scales: Mapping[str, float] | None,
    passengers: tuple[str, ...] = (),
) -> ScaleCheck:
    """Scale-set completeness and the power identity, from the declaration alone.

    Returns a ScaleCheck; the caller (L3 condition C4) turns it into a verdict.
    A missing scale and a violated identity are different failures and are
    reported separately.
    """
    spec = spec_for(port_type)
    required = spec.required_scale_keys(passengers)
    scales = dict(scales or {})
    missing = tuple(k for k in required if k not in scales)
    if missing:
        return ScaleCheck(
            complete=False,
            missing=missing,
            power_identity_residual=None,
            ok=False,
            detail=(
                f"{port_type.value} requires {list(required)}; missing {list(missing)}. "
                "Completeness is decidable from the port list alone, so this is a "
                "compile-time refusal."
            ),
        )

    residuals: list[tuple[str, float]] = []
    pairs = [(spec.effort_scale_key, spec.flow_scale_key, spec.power_scale_key)]
    for p in passengers if spec.multibond else ():
        pairs.append((f"{p}_effort", f"{p}_flow", f"{p}_power"))

    for e_key, f_key, p_key in pairs:
        s_e, s_f, s_p = float(scales[e_key]), float(scales[f_key]), float(scales[p_key])
        if s_p == 0.0:
            residuals.append((p_key, math.inf))
            continue
        residuals.append((p_key, abs(s_e * s_f / s_p - 1.0)))

    worst_key, worst = max(residuals, key=lambda kv: kv[1])
    ok = worst <= POWER_IDENTITY_TOL
    return ScaleCheck(
        complete=True,
        missing=(),
        power_identity_residual=worst,
        ok=ok,
        detail=(
            ""
            if ok
            else (
                f"power identity s_e * s_f = s_P violated at {worst_key} by relative "
                f"{worst:.3e}. Independently chosen effort and flow scales are "
                "dimensionally plausible and silently destroy the power bond; after that "
                "the global power residual reports a unit error in watts."
            )
        ),
    )


def compose_scales(
    port_type: PortType,
    scales_a: Mapping[str, float] | None,
    scales_b: Mapping[str, float] | None,
    passengers: tuple[str, ...] = (),
) -> tuple[bool, dict[str, float], str]:
    """Whether two experts' scale sets compose to a defined conversion at a port.

    Two experts on different reference scales will not agree at a port, and the
    residual will not say why (L3 condition C4).  The conversion factor per key
    is emitted so the failure is attributable.
    """
    spec = spec_for(port_type)
    required = spec.required_scale_keys(passengers)
    a, b = dict(scales_a or {}), dict(scales_b or {})
    missing = [k for k in required if k not in a or k not in b]
    if missing:
        return False, {}, f"cannot compose: one side is missing {missing}"
    ratios = {}
    for k in required:
        if float(b[k]) == 0.0:
            return False, {}, f"cannot compose: scale {k} is zero on one side"
        ratios[k] = float(a[k]) / float(b[k])
    return True, ratios, ""


def advec_passenger_pressure(ports: list[Any]) -> int:
    """Count interfaces the five types cannot express without an ADVEC extension.

    This is the field-5 measurement PortAmendment asks for: the leading indicator
    of vocabulary pressure, and it is countable today.  A passenger list beyond
    the base energy channel is a vocabulary growing per case while wearing a
    single type name.
    """
    count = 0
    for p in ports:
        if getattr(p, "port_type", None) is PortType.ADVEC:
            passengers = tuple(getattr(p, "passengers", ()) or ())
            extra = [x for x in passengers if x not in ("h0",)]
            if extra:
                count += 1
    return count


# ---------------------------------------------------------------------------
# W66 -- which half of the bond the callable returns
# ---------------------------------------------------------------------------


class ResponseHalf(enum.Enum):
    """Which half of the conjugate pair ``boundary_response`` returns.

    **The field exists because nothing else can supply the information.**  W66,
    opened 2026-08-28 and closed 2026-08-29 by declaration after both cheaper
    routes were measured and failed:

    * *The operator cannot say.*  A response scaled by any positive constant
      gives ``sym(cS) = c sym(S)``, so every eigenvalue of the symmetric part
      keeps its sign and E7's passivity test is exactly invariant.  Measured on
      the ``THERM`` seam of `cases/thermal_seam.py`: returning the *pseudo-bond*
      ``q_n`` where the declared flow is the entropy flux ``q_n/T`` leaves the
      verdict, the stamp, the null count and the passivity defect **numerically
      identical**, and moves only ``beta``, which nothing checks.

    * *The magnitude cannot say either.*  Nondimensionalize properly and
      ``s_e = s_f = s_P = 1``, so the declared scales carry no information about
      which half a number is.  Measured on `cases/wind_farm_real.py`, where
      ``U_INF = 1``: the correct ``ADVEC`` response ``h0`` and the incorrect
      ``(u.n) h0`` differ by **5%**.  A magnitude guard is vacuous exactly where
      the port algebra is used correctly.

    So it is declared, and the declaration is load-bearing in one place a
    declaration *can* be: the two sides of a seam must return the **same** half,
    because ``Lambda_M = sum_i P_i^* Lambda_i P_i`` adds them.  Adding a traction
    to a velocity is not a quantity, and before this field existed `window_ns`
    compiled to **`admit`** with one side of every seam returning the flow.
    """

    EFFORT = "effort"
    FLOW = "flow"
    UNDECLARED = "undeclared"

    @property
    def conjugate(self) -> "ResponseHalf":
        if self is ResponseHalf.EFFORT:
            return ResponseHalf.FLOW
        if self is ResponseHalf.FLOW:
            return ResponseHalf.EFFORT
        return ResponseHalf.UNDECLARED


@dataclass(frozen=True)
class PairingCheck:
    """The outcome of checking a seam's declared conjugate pairing."""

    declared: bool
    agree: bool
    response_half: ResponseHalf
    trace_half: ResponseHalf
    ok: bool
    detail: str = ""

    def scale_keys(self, spec: PortSpec) -> tuple[str | None, str | None]:
        """(scale key for the response, scale key for the trace), or (None, None).

        Which scale nondimensionalizes which side of the probe is exactly the
        thing the declaration fixes, and it was being chosen silently before.
        """
        if not self.declared:
            return (None, None)
        e, f = spec.effort_scale_key, spec.flow_scale_key
        if self.response_half is ResponseHalf.EFFORT:
            return (e, f)
        return (f, e)


def check_response_half(
    port_type: PortType,
    half_a: "ResponseHalf | str | None",
    half_b: "ResponseHalf | str | None",
) -> PairingCheck:
    """L3/C9 -- do both sides of a seam return the same half of the bond?

    Three outcomes, and the middle one is the whole point:

    * either side ``UNDECLARED`` -> **not ok**, and the caller decertifies.  This
      is the status quo made visible rather than a new failure: nothing has ever
      validated what a callable returns.
    * both declared and **different** -> **not ok**, and the caller refuses.  The
      assembled operator would sum an effort and a flow.
    * both declared and the same -> ok, and the trace is the conjugate half.

    It is deliberately *not* a check on the callable's values: no property of the
    returned numbers distinguishes the halves (see `ResponseHalf`), so a check
    that inspected them would be a guard that fires on the wrong thing.
    """
    spec = spec_for(port_type)

    def _coerce(h) -> ResponseHalf:
        if h is None:
            return ResponseHalf.UNDECLARED
        if isinstance(h, ResponseHalf):
            return h
        return ResponseHalf(str(h))

    a, b = _coerce(half_a), _coerce(half_b)
    undeclared = [n for n, h in (("a", a), ("b", b)) if h is ResponseHalf.UNDECLARED]

    if undeclared:
        return PairingCheck(
            declared=False, agree=False,
            response_half=ResponseHalf.UNDECLARED,
            trace_half=ResponseHalf.UNDECLARED,
            ok=False,
            detail=(
                f"{port_type.value}: side(s) {undeclared} do not declare which half of "
                f"({spec.effort}, {spec.flow}) boundary_response returns. "
                "check_scales validates the scale SET and never sees the callable, so "
                "nothing here establishes that the pairing is a power (W66)."
            ),
        )

    if a is not b:
        return PairingCheck(
            declared=True, agree=False, response_half=a, trace_half=a.conjugate,
            ok=False,
            detail=(
                f"{port_type.value}: one side returns the {a.value} and the other the "
                f"{b.value}. Lambda_M = sum_i P_i^* Lambda_i P_i ADDS the two responses, "
                f"so the assembled operator would be a {spec.effort} plus a {spec.flow}, "
                "which is not a quantity."
            ),
        )

    return PairingCheck(
        declared=True, agree=True, response_half=a, trace_half=a.conjugate, ok=True,
        detail=(
            f"{port_type.value}: both sides return the {a.value} "
            f"({spec.effort if a is ResponseHalf.EFFORT else spec.flow}); the imposed "
            f"trace is therefore the {a.conjugate.value} and the pairing is a power."
        ),
    )
