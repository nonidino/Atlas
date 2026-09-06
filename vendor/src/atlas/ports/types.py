"""The closed set of Atlas port types.

This module is **not** case-study code. It is the port algebra of
`port-algebra-atlas-0.1` and every later case study reuses it, so nothing about
turbines, wakes or rotor disks belongs here.

A port is an effort/flow pair whose product is a power density. Two agents
coupled through a port agree on both, and the pair being a power conjugate is
what makes the global residual of `port-algebra-atlas-0.1` section 6 a single
scalar rather than a pile of incommensurable numbers.

Mapping mode is the load-bearing field
--------------------------------------
It is preCICE's `conservative` / `consistent` distinction, and getting it
backwards is a classic partitioned-coupling bug that produces plausible-looking
and quietly wrong answers:

* a **flow** must map so that its *integral* over the interface is preserved --
  splitting one coarse face into two fine ones splits the mass flux between
  them, it does not duplicate it;
* an **effort** must map so that its *pointwise values* are preserved --
  splitting one coarse face into two fine ones gives both the same traction.

`MappingMode.apply` implements exactly that difference, so the two never get
confused at a call site.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MappingMode(str, Enum):
    """How a quantity is transferred between two discretizations of one curve."""

    CONSERVATIVE = "conservative"   # preserve the integral (flows)
    CONSISTENT = "consistent"       # preserve pointwise values (efforts)

    def apply(self, values: np.ndarray, w_src: np.ndarray, w_dst: np.ndarray,
              weights: np.ndarray) -> np.ndarray:
        """Map `values` from a source discretization to a destination one.

        `weights[i, j]` is the overlap length of source segment `i` with
        destination segment `j`; `w_src`, `w_dst` are the segment lengths."""
        if self.value == "conservative":
            # split each source segment's total across the destinations it overlaps
            total = values * w_src
            share = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-300)
            return (total[:, None] * share).sum(axis=0) / np.maximum(w_dst, 1e-300)
        # consistent: length-weighted average of the source values covering each dst
        num = (values[:, None] * weights).sum(axis=0)
        return num / np.maximum(weights.sum(axis=0), 1e-300)


@dataclass(frozen=True)
class PortType:
    name: str
    effort: str
    effort_units: str
    effort_mapping: MappingMode | None
    flow: str
    flow_units: str
    flow_mapping: MappingMode
    conjugate: str                  # what effort * flow integrates to
    vector: bool                    # effort/flow are vectors on the interface

    def check(self) -> None:
        if self.flow_mapping is not MappingMode.CONSERVATIVE:
            raise AssertionError(f"port {self.name}: a flow must map conservatively")
        if self.effort_mapping not in (None, MappingMode.CONSISTENT):
            raise AssertionError(f"port {self.name}: an effort must map consistently")


#: Traction and velocity across a mechanical interface. Both the normal and the
#: tangential components live here: `port-algebra-atlas-0.1` resolved the old
#: separate `shear` label by observing they are two components of one traction
#: vector `t = sigma . n`, not two quantities.
MECH = PortType(
    name="MECH",
    effort="traction", effort_units="rho U^2", effort_mapping=MappingMode.CONSISTENT,
    flow="velocity", flow_units="U", flow_mapping=MappingMode.CONSERVATIVE,
    conjugate="mechanical power", vector=True,
)

#: Advected transport. At constant density in an isothermal incompressible
#: setting this degenerates to volumetric flux and carries no independent
#: effort, which is why `effort_mapping` is None rather than a mode.
ADVEC = PortType(
    name="ADVEC",
    effort="stagnation enthalpy (carrier)", effort_units="U^2", effort_mapping=None,
    flow="mass flux", flow_units="rho U", flow_mapping=MappingMode.CONSERVATIVE,
    conjugate="enthalpy flux", vector=False,
)

#: A lumped rotational port. Unconnected ports are how power leaves the model
#: without vanishing -- see `port-algebra-atlas-0.1` on P_ext.
ROT = PortType(
    name="ROT",
    effort="torque", effort_units="rho U^2 D^3", effort_mapping=MappingMode.CONSISTENT,
    flow="angular velocity", flow_units="U/D", flow_mapping=MappingMode.CONSERVATIVE,
    conjugate="shaft power", vector=False,
)

THERM = PortType(
    name="THERM",
    effort="temperature", effort_units="T_ref", effort_mapping=MappingMode.CONSISTENT,
    flow="heat flux", flow_units="rho U^3", flow_mapping=MappingMode.CONSERVATIVE,
    conjugate="thermal power", vector=False,
)

ELEC = PortType(
    name="ELEC",
    effort="voltage", effort_units="V_ref", effort_mapping=MappingMode.CONSISTENT,
    flow="current", flow_units="I_ref", flow_mapping=MappingMode.CONSERVATIVE,
    conjugate="electrical power", vector=False,
)

#: The closed set. Nothing outside it may appear on an edge; a case study that
#: needs a sixth type is a change to the port algebra, made here, deliberately.
PORT_TYPES: dict[str, PortType] = {p.name: p for p in (MECH, ADVEC, ROT, THERM, ELEC)}

for _p in PORT_TYPES.values():
    _p.check()


def port(name: str) -> PortType:
    try:
        return PORT_TYPES[name]
    except KeyError:
        raise KeyError(f"unknown port type {name!r}; the closed set is {sorted(PORT_TYPES)}") from None


__all__ = ["MappingMode", "PortType", "PORT_TYPES", "port",
           "MECH", "ADVEC", "ROT", "THERM", "ELEC"]
