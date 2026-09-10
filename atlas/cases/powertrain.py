"""CS-14: a drivetrain on the wake array's open shaft, and the ELEC port at last.

Rung 7 of `f1-pathmap-and-end-goal`'s ladder.  `wake_array.rotor_capabilities`
declares a `shaft:ROT` port and says of it, in its own note:

    unconnected: no drivetrain.  The extracted power leaves the system here, and
    under the port algebra that is an OPEN PORT with a measurable power flow
    rather than an absence.

**This case study connects it.**  The rotor's shaft drives a motor-generator,
the machine sits in a DC circuit with a bus, a battery and an inverter, and the
power that used to leave the system is followed to where it is stored and to
where it is lost.

What only this case study can say
---------------------------------

**1. It closes the port vocabulary, and TWO types were open rather than one.**
Censused rather than recalled: `MECH`, `THERM` and `ADVEC` had carried real
connections.  **`ROT` had been declared three times -- `wind_farm`,
`wind_farm_real`, `wake_array` -- and every one of those declarations says
"unconnected: no drivetrain"**, so the type had never appeared in a
`Connection`.  **`ELEC` had never been declared at all**; `rocket.py`'s own
docstring says so in its fourth line, and correctly, because a solid-propellant
rocket has no electrical bond.

This graph gives both their first seam, so all five of the closed vocabulary
are now exercised.  **The cost was one scale set and one prolongation each**,
which is the affordability claim being tested rather than restated: the port
algebra promises the twentieth expert costs what the fourth did, `O(K)`
declarations rather than `O(K^2)` adapters, and until this session it had never
run past two governing families.

**2. It is the first CROSS-DOMAIN energy balance in this package**, and unlike
CS-13's loop balance it is **not** an identity of the solve.  The mechanical
side is read off the build repo's own `disk.ActuatorDisk`, which knows nothing
about circuits; the electrical side is solved from Kirchhoff's laws, which know
nothing about wakes; and the bridge between them is one physical fact --

    k_e = k_t

the machine's back-EMF constant equals its torque constant, in SI, because both
are the same flux linkage seen from the two sides of the same air gap.  So

    T omega  =  (k_t I) omega  =  (k_e omega) I  =  V_emf I

is a **theorem about the machine** rather than a definition anybody chose, and
the balance can fail: set ``k_e != k_t`` and it does, in proportion.  That is
asserted in `tests/test_tier37_powertrain.py`.

**3. It is W163's second cyclic graph**, which is the thing that row asks for.
A DC circuit is a loop -- current leaves the battery, passes the inverter,
crosses the machine and returns through the bus -- so the four electrical agents
form a directed cycle exactly as CS-13's coolant does, on completely different
physics.  **The compiler still cannot see it**: compiled with the return
conductor and without it, same verdict, same rule set, same per-agent decisions.
Two graphs now say so, which is what W163 said it was waiting for, and the rule
is still not written here, because writing it is the next tier's decision and
not a side effect of this one.

**4. And the machine has a real `validity` predicate, not a placeholder.**  A
generator can only push current into a battery whose terminal voltage it exceeds
-- ``k_e omega > V_oc + I R`` -- and below that speed the circuit reverses and
the machine motors instead.  `MGU.validity` declines outside it, which is what
`spec` §3.4 asks every expert for and what most fixtures here supply as
``return True``.

The topology
------------

    ROTOR --ROT--  MGU  --ELEC-->  BUS  --ELEC-->  BATT
                    ^                                |
                    +---------- INV <----ELEC--------+

`ROTOR` is `wake_array.RotorDisk` -- the build repo's zero-parameter closed form
-- and is the only agent here that is not four lines of algebra.  The four
electrical agents are `disk.ActuatorDisk`'s class: closed form, zero fitted
parameters.  **They buy a port type and a topology, not physics**, and this
module says so rather than implying otherwise.

Run::

    python -c "from atlas.cases import powertrain as P; print(P.compile_report())"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..capability import (BCChannel, ClaimType, Differentiable, Direction,
                          EllipticSubsolve, ExpertCapabilities, MotionClass,
                          TimeDiscretization, port_decl)
from ..graph import (Agent, CaseGraph, Connection, Decomposition,
                     DeclaredLoopGain, MeasuredConstants)
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation
from . import wake_array as WA

__all__ = [
    "MachineAgent", "BusLeg", "BatteryLeg", "InverterLeg",
    "CircuitSolve", "CircuitResult",
    "build", "connections", "compile_report", "make_agents",
    "CIRCUIT_ORDER", "K_E", "K_T", "R_MGU", "R_BUS", "R_BATT", "R_INV",
    "V_OC", "A_TERM", "MACRO_DT", "M_LUMPED",
]


# ---------------------------------------------------------------------------
# the drivetrain, in numbers
# ---------------------------------------------------------------------------

#: The macro-step is the rotor's, which is the agent with a clock.  Every
#: electrical element here is quasi-steady -- a DC circuit with no inductance
#: settles in microseconds against a rotor's tenths of a second -- so they
#: declare the macro-step as their own native step, which is `L7/R9`'s lesson
#: from CS-13 applied before the rule had to teach it again.
MACRO_DT = WA.MACRO_DT

#: **The machine constants, and they are ONE number twice.** In SI the back-EMF
#: constant and the torque constant of a given machine are equal, because both
#: are the same air-gap flux linkage read from the two sides of it. They are
#: written as two names because the balance in section 2 of the module docstring
#: is a theorem *given* that equality, and a test that cannot break the equality
#: cannot check the theorem.
#:
#: **Nondimensional, like everything on the other side of the shaft.**  The
#: first version of this module gave the machine SI constants and the rotor a
#: nondimensional shaft, and `MachineAgent.validity` DECLINED -- ``k_e omega``
#: came to 0.19 against a 3.6 V battery, so the machine motored where the graph
#: declared generation.  That is the first time a `validity` predicate in this
#: package has caught a defect rather than returning ``True``, and it caught a
#: units error across a seam, which is exactly what the nondimensionalisation
#: half of the port algebra exists for.  `wake_array` is nondimensional
#: throughout (`ROT_SCALES` is all 1.0), so this side is too, and the
#: dimensional reading lives in `ELEC_SCALES` where the algebra puts it.
K_T = 0.10                       # torque per unit current
K_E = K_T                        # volts per unit shaft speed -- the SAME number

#: Series resistances round the loop.  A real drivetrain's losses live in four
#: different places and the circuit is what adds them up.
R_MGU = 0.15                     # machine windings
R_BUS = 0.05                     # the DC harness
R_BATT = 0.07                    # battery internal resistance
R_INV = 0.03                     # inverter conduction loss

#: The battery's open-circuit voltage, and it is the one constant here that was
#: chosen by looking at the answer -- so the reasoning is on the record.
#:
#: **It must not be 1.30.**  At ``V_oc = 1.30`` the machine demands exactly the
#: torque `disk.py` delivers at its own reference induction ``a = 1/3``, the
#: bisection below returns the number it started from, and an untested code path
#: passes.  **And it must not be 1.28**, which the first version used: that
#: demands ``0.0733`` against a maximum of ``0.0718`` at the donor's clamp
#: ``a = 0.4``, so no operating point exists and `rotor_valid` declines -- which
#: is the predicate working, and is not a graph anybody can march.
#:
#: ``1.34`` demands ``0.0533``, which the disk delivers near ``a = 0.285``:
#: inside the clamp with room, and away from the reference induction, so the
#: search has to find it.
V_OC = 1.34

#: The loop's total series resistance, summed here so the circuit and the
#: machine cannot come to disagree about it.
R_TOTAL = R_MGU + R_BUS + R_BATT + R_INV

#: The terminal cross-section.  `PORT_SPECS[ELEC]`'s declared flow is a CURRENT
#: DENSITY and its power is per unit area, so a lumped circuit terminal needs an
#: area for the units to close -- and a real terminal has one. Declaring it is
#: what lets `check_scales` verify s_e * s_f = s_P from the port list alone.
A_TERM = 1.5e-4                  # m^2

#: The rotor's inflow.  The disk's response is quadratic in it, so this is also
#: the state its block is linearized about (W74, on a rotor -- `RotorDisk`'s own
#: `probe_base` says so).
U_REF = 1.0

#: A lumped port is the case dim M = 1 -- `PORT_SPECS[ROT]`'s own note, and
#: CS-13 learnt it from `L4/null-space` reporting fifteen null directions on a
#: one-state leg. Declared up front here rather than after the rule says so.
M_LUMPED = 1
N_TERM = 8                       # cells across a terminal, for the V space


# ---------------------------------------------------------------------------
# the scale sets
# ---------------------------------------------------------------------------

#: (T, omega) in N m and rad/s: s_e s_f = s_P with s_P in W.  Carried from
#: `wake_array` unchanged, because the rotor's side of this seam is that
#: module's port and comparing one expert through two declarations confounds the
#: expert with the presentation.
ROT_SCALES = dict(WA.ROT_SCALES)

#: (V, j.n): s_e s_f = s_P.  The port algebra checks the IDENTITY, not the
#: units, so a consistent nondimensional set is a legal declaration -- and the
#: dimensional reading is one multiplication away, which is the point of
#: declaring scales at all rather than hard-coding volts.
I_REF = 1.0                      # a reference current for the scale set
ELEC_SCALES = {"potential": V_OC,
               "current_density": I_REF / A_TERM,
               "power_area": V_OC * I_REF / A_TERM}


def terminal_prolongation(agent_id: str, port_name: str) -> Prolongation:
    """The one-dimensional prolongation of a lumped terminal: the constant mode.

    A circuit node has one potential, so its interface space is
    one-dimensional. `gram_V` is the terminal's own area measure, so the power
    pairing on this seam is an integral over the terminal rather than a sum over
    an index set that happens to have `N_TERM` entries.
    """
    h = A_TERM / N_TERM
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=np.full((N_TERM, 1), 1.0 / math.sqrt(A_TERM)),
        gram_V=h * np.eye(N_TERM),
        label="the constant mode over the terminal: a lumped port is dim M = 1",
    )


def shaft_prolongation(agent_id: str) -> Prolongation:
    """A shaft has one angular coordinate.  `ROT` is `lumped=True` by spec."""
    return Prolongation(
        agent_id=agent_id,
        port_name="shaft:ROT",
        matrix=np.ones((1, 1)),
        gram_V=np.ones((1, 1)),
        label="a shaft is one degree of freedom (PORT_SPECS[ROT].lumped)",
    )


# ---------------------------------------------------------------------------
# the electrical agents -- four lines of algebra each
# ---------------------------------------------------------------------------


@dataclass
class _Element:
    """One circuit element: a series EMF and a series resistance.

    Every element in this loop has the form ``V_out = V_in + e - I R`` with
    ``e`` and ``R`` closed-form functions of declared constants. **That
    affineness is what makes the loop current available in closed form** beside
    the iteration that finds it, exactly as CS-13's legs make the return
    temperature available -- and for the same reason: an iteration that
    converged to the wrong thing and a closed form nobody checked are the same
    artifact.
    """

    agent_id: str = "ELEM"
    dt: float = MACRO_DT
    resistance: float = 0.0
    v_in: float = 0.0

    def emf(self, omega: float = 0.0) -> float:
        """Volts this element ADDS to the loop.  Zero for a passive one."""
        return 0.0

    def drop(self, current: float, omega: float = 0.0) -> float:
        """Volts across the element, positive in the direction of the current."""
        return self.emf(omega) - current * self.resistance

    def dissipation(self, current: float) -> float:
        """Watts turned to heat here.  ``I^2 R``, and nothing else is a loss."""
        return float(current * current * self.resistance)

    # -- ports -------------------------------------------------------------

    def respond_elec(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(terminal current density) -> (terminal potential).

        `ELEC`'s bond is ``(potential, j . n)``; this element's response half is
        the EFFORT, so the imposed trace is the current density and what comes
        back is the potential this terminal must sit at for the element to carry
        that current **into itself**.

        **Both terminals report in their own element's sense, and that is what
        makes the seam passive.**  The first version flipped the sign on the
        ``lo`` terminal, which made one of the two blocks NEGATIVE and left
        `L4/E7/passivity` reporting a defect of 1.5e-5 on `mgu_bus` -- W138's
        shape on a third port type, caught by the rule this session taught to
        see it.  With each side reporting ``+I R`` the assembled operator at a
        node is the SERIES resistance of the two elements meeting there, which
        is positive because a resistive network is passive, and no
        `effort_normal` is needed: this really is the own-outward convention.
        """
        j = np.asarray(trace, dtype=float).reshape(-1)
        current = float(np.mean(j)) * A_TERM
        return np.full(j.shape,
                       self.v_in - self.emf() + current * self.resistance)

    def storage(self, state=None) -> float:
        """``1/2 C V^2``: an element's stored electrostatic energy, positive
        definite.  A pure resistance stores nothing and declares the terminal
        capacitance it actually has rather than zero, because zero would make
        the passivity argument rest on an absence."""
        v = self.v_in if state is None else float(np.mean(np.asarray(state)))
        return float(0.5 * C_TERM * v * v)

    def validity(self, state=None, cond=None) -> bool:
        return True

    def base_for(self, port) -> np.ndarray:
        """W74: the operating point of this port's own variable.

        Every port on an electrical element carries a current density, so unlike
        CS-13's legs there is no dispatch to get wrong here -- and the method
        exists anyway, because an element that grows a second port type should
        have to come past it.
        """
        return np.full(N_TERM, I_REF / A_TERM)


#: Terminal capacitance, F.  Small and real: it is what a `storage` function has
#: to be built on for a circuit node, and declaring zero would make E7's
#: certificate rest on an absence rather than on a measurement.
C_TERM = 4.7e-6


@dataclass
class BusLeg(_Element):
    """The DC harness.  A pure series resistance."""

    agent_id: str = "BUS"
    resistance: float = R_BUS


@dataclass
class InverterLeg(_Element):
    """The inverter.  A series conduction loss; switching is not modelled."""

    agent_id: str = "INV"
    resistance: float = R_INV


@dataclass
class BatteryLeg(_Element):
    """The battery.  ``V = V_oc - I R_int``, and it is the loop's only sink.

    Its EMF **opposes** the machine's, which is what makes charging cost work:
    the current the generator pushes has to climb the open-circuit voltage
    before any of it is stored.
    """

    agent_id: str = "BATT"
    resistance: float = R_BATT
    v_oc: float = V_OC

    def emf(self, omega: float = 0.0) -> float:
        return -self.v_oc

    def stored(self, current: float) -> float:
        """Watts into the chemistry.  ``V_oc I`` -- not ``V_term I``, which
        includes the internal resistance's heat and is what a terminal
        measurement would report."""
        return float(self.v_oc * current)


@dataclass
class MachineAgent(_Element):
    """The motor-generator: the one agent with a port in each domain.

    ``V_emf = k_e omega`` on the electrical side and ``T = k_t I`` on the
    mechanical one, with ``k_e = k_t`` because both are the same air-gap flux
    linkage. That single equality is the whole bridge, and it is why the energy
    balance across the two domains is a **theorem** rather than a definition.
    """

    agent_id: str = "MGU"
    resistance: float = R_MGU
    k_e: float = K_E
    k_t: float = K_T
    omega: float = 15.0

    def emf(self, omega: float | None = None) -> float:
        return self.k_e * (self.omega if omega is None else float(omega))

    def torque_at(self, current: float) -> float:
        """The mechanical torque the machine takes to carry this current."""
        return float(self.k_t * current)

    def current_for(self, torque: float) -> float:
        """The current the machine must carry to absorb this torque."""
        return float(torque / self.k_t)

    def respond_rot(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(shaft angular velocity) -> (reaction torque).

        `ROT`'s bond is ``(torque, omega)`` and this port's response half is the
        EFFORT, matching the rotor's own declaration on the other side.  The
        machine's reaction torque at a given speed is what the circuit lets it
        carry: at speed ``w`` it develops ``k_e w`` against the battery's
        ``V_oc``, the loop resistance sets the current, and ``k_t I`` is the
        torque that current costs.

        **This is the seam's whole content**, and it is why the operating point
        is a solve rather than a pair of numbers: the rotor's torque is a
        function of its induction and the machine's is a function of the shaft
        speed, and the two have to agree.
        """
        w = float(np.mean(np.asarray(trace, dtype=float)))
        return np.array([self.k_t * self.current_at(w)])

    def current_at(self, omega: float) -> float:
        """The loop current the circuit carries at this shaft speed."""
        return (self.k_e * float(omega) - V_OC) / R_TOTAL

    def validity(self, state=None, cond=None) -> bool:
        """**A real predicate, not a placeholder.**

        A generator can only push current into the battery while its back-EMF
        exceeds the open-circuit voltage. Below that speed the loop current
        reverses and the machine MOTORS -- which is a legitimate mode and a
        different one from the mode this graph declares, so the record declines
        rather than reporting a negative generated power as if it were
        generation.
        """
        w = self.omega if state is None else float(np.mean(np.asarray(state)))
        return bool(self.k_e * w > V_OC)

    def base_for(self, port) -> np.ndarray:
        name = getattr(port, "name", str(port))
        if name.endswith(":ROT"):
            return np.full(1, self.omega)
        return np.full(N_TERM, I_REF / A_TERM)


# ---------------------------------------------------------------------------
# the operating point: a torque balance across the shaft
# ---------------------------------------------------------------------------

#: The loop in current order, and everything that needs the topology reads it
#: here: `connections` builds one ELEC seam per consecutive pair and one more to
#: close the ring, and `CircuitSolve` sums round it in this order.
CIRCUIT_ORDER: tuple[str, ...] = ("MGU", "BUS", "BATT", "INV")

#: `disk.py`'s own clamp on the induction factor -- momentum theory breaks down
#: in the turbulent-wake state and the donor refuses to extrapolate. It is read
#: here as the ROTOR's `validity` bound rather than restated as a number,
#: because a bound copied out of a donor is a bound that can drift from it.
A_MIN, A_MAX = 0.02, 0.40


@dataclass
class CircuitResult:
    """One solved operating point, with every term of the cross-domain balance."""

    omega: float
    induction: float                 # the controller's lever, SOLVED for
    torque_rotor: float              # from disk.py at that induction
    torque_machine: float            # k_t I, from the circuit
    p_mech: float                    # T omega
    current: float
    v_emf: float
    p_electrical: float              # V_emf I
    dissipation: dict                # element -> I^2 R
    p_stored: float                  # V_oc I, into the chemistry
    residual: float                  # |p_mech - (losses + stored)| / p_mech
    bridge_residual: float           # |p_mech - p_electrical| / p_mech
    torque_residual: float           # |T_rotor - T_machine| / T_machine
    kvl_residual: float              # |sum of drops round the loop|
    closed_form: float               # the loop current, solved directly
    current_gap: float               # |iterated - closed form| / |closed form|
    mgu_valid: bool
    rotor_valid: bool
    iterations: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class CircuitSolve:
    """Find the operating point, and check the two domains against each other.

    **The shaft is a balance and not a pair of numbers.**  The rotor's torque is
    a function of the induction its controller sets; the machine's is a function
    of the shaft speed, through the circuit.  The operating point is where they
    agree, and finding it is a one-unknown solve on the donor's own closed form:

        T_rotor(a, u)  =  k_t * (k_e * omega - V_oc) / R_total

    with ``omega = lambda u`` fixed by the tip-speed ratio the disk declares.
    ``a`` is the controller's lever and `disk.py` clamps it at 0.4 -- momentum
    theory breaks down above that and the donor refuses to extrapolate -- so an
    operating point outside the clamp does not exist and the ROTOR's `validity`
    declines rather than the search returning a number nobody can use.

    **The circuit is solved rather than swept**, for CS-13's reason carried to a
    second cycle: a directed loop has no first element, so every sweep order is
    a modelling choice nothing in the declaration licenses. The loop map is
    affine in the current, so its fixed point is available in closed form --

        I  =  (sum of the elements' EMFs) / (sum of their resistances)

    -- **assembled from the elements' own ``emf`` and ``resistance``** rather
    than written out, so the closed form cannot drift from the elements it is
    the closed form of.  The relaxation is run beside it and the pair is the
    control.

    **What the balance is, and this time it is NOT an identity.**  CS-13's loop
    balance reduced to ``T_return = T_return`` on substitution and its page said
    so.  This one does not.  ``torque_rotor`` comes from the build repo's
    `disk.ActuatorDisk`, which contains no circuit; ``torque_machine`` comes
    from Kirchhoff's laws, which contain no wake; and they meet only through
    ``k_e = k_t``.  Breaking that equality breaks the balance in proportion,
    which `tests/test_tier37_powertrain.py` asserts by breaking it.
    """

    def __init__(self, rotor: Any | None = None,
                 elements: dict | None = None,
                 u_ref: float = None,
                 tol: float = 1.0e-14, max_iter: int = 200) -> None:
        self.rotor = rotor if rotor is not None else _rotor()
        self.elements = elements if elements is not None else make_elements()
        self.u_ref = U_REF if u_ref is None else float(u_ref)
        self.tol = tol
        self.max_iter = max_iter

    # -- the rotor's side, from the real expert -----------------------------

    def torque_at(self, a: float) -> float:
        """``disk.py``'s own torque at this induction and this inflow.

        A fresh `ActuatorDisk` per evaluation, from the donor's own module: the
        induction is a constructor argument there and reaching around it would
        be this case study writing physics rather than declaring it.
        """
        disk = self.rotor._mod.ActuatorDisk(a=float(a))
        return float(disk(self.u_ref).torque)

    def omega(self) -> float:
        """The shaft speed the disk declares at this inflow.

        Read off the donor rather than computed from a tip-speed ratio written
        down here, so a change to `disk.py`'s convention reaches this graph.
        """
        return float(self.rotor._disk(self.u_ref).omega)

    # -- the circuit --------------------------------------------------------

    def _closed_form(self, omega: float) -> float:
        """``sum(emf) / sum(R)``, composed from the elements themselves."""
        e = sum(self.elements[n].emf(omega) for n in CIRCUIT_ORDER)
        r = sum(self.elements[n].resistance for n in CIRCUIT_ORDER)
        if r <= 0.0:
            raise RuntimeError(
                "the loop has no resistance, so its current is unbounded. That "
                "is a statement about the circuit and not a solver failure: an "
                "ideal source across an ideal conductor has no operating point"
            )
        return e / r

    def _relax(self, omega: float) -> tuple[float, int]:
        """The same current by relaxation, as the control on the formula."""
        e = sum(self.elements[n].emf(omega) for n in CIRCUIT_ORDER)
        r = sum(self.elements[n].resistance for n in CIRCUIT_ORDER)
        i, k = 0.0, 0
        for k in range(1, self.max_iter + 1):
            nxt = i + 0.5 * (e - i * r) / r
            if abs(nxt - i) < self.tol * max(1.0, abs(i)):
                i = nxt
                break
            i = nxt
        return i, k

    # -- the balance --------------------------------------------------------

    def solve(self) -> CircuitResult:
        mgu = self.elements["MGU"]
        w = self.omega()
        mgu.omega = w

        i_closed = self._closed_form(w)
        i_relax, iters = self._relax(w)
        current = i_closed
        t_machine = mgu.torque_at(current)

        #: the controller's lever, found by bisection on the donor's own curve.
        #: Monotone in `a` over the clamp, so bisection is the right tool and
        #: needs no derivative the donor does not expose.
        lo, hi = A_MIN, A_MAX
        f_lo = self.torque_at(lo) - t_machine
        f_hi = self.torque_at(hi) - t_machine
        rotor_valid = bool(f_lo * f_hi <= 0.0)
        if rotor_valid:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if (self.torque_at(mid) - t_machine) * f_lo <= 0.0:
                    hi = mid
                else:
                    lo, f_lo = mid, self.torque_at(mid) - t_machine
                if hi - lo < 1e-15:
                    break
            a_star = 0.5 * (lo + hi)
        else:
            #: no operating point inside the donor's clamp. The nearest edge is
            #: reported so the decline is legible, and `rotor_valid` says it is
            #: not a solution.
            a_star = A_MAX if f_hi < 0.0 else A_MIN

        t_rotor = self.torque_at(a_star)
        p_mech = t_rotor * w
        v_emf = mgu.emf(w)
        diss = {n: self.elements[n].dissipation(current) for n in CIRCUIT_ORDER}
        stored = self.elements["BATT"].stored(current)
        kvl = sum(self.elements[n].drop(current, w) for n in CIRCUIT_ORDER)
        scale = max(abs(p_mech), 1.0e-30)
        return CircuitResult(
            omega=w, induction=a_star,
            torque_rotor=t_rotor, torque_machine=t_machine,
            p_mech=p_mech, current=current, v_emf=v_emf,
            p_electrical=v_emf * current,
            dissipation=diss, p_stored=stored,
            residual=abs(p_mech - (sum(diss.values()) + stored)) / scale,
            bridge_residual=abs(p_mech - v_emf * current) / scale,
            torque_residual=(abs(t_rotor - t_machine)
                             / max(abs(t_machine), 1.0e-30)),
            kvl_residual=abs(kvl),
            closed_form=i_closed,
            current_gap=abs(i_relax - i_closed) / max(abs(i_closed), 1e-30),
            mgu_valid=bool(mgu.validity()),
            rotor_valid=rotor_valid,
            iterations=iters,
        )


def composed_loop_gain(elements: dict, order: tuple[str, ...],
                       omega: float = 0.0, current: float = 1.0) -> float:
    """The gain of one traversal, composed from the elements' OWN map.

    **W163, and the answer is 1 for a structural reason.**  The ELEC seam
    carries the terminal POTENTIAL, and each element's declared map is
    ``V -> V - emf + I R`` -- a translation.  Its derivative is one, so the
    derivative round the ring is one whatever the emfs and resistances are.
    Composed here rather than asserted, so that an element which stopped being
    affine in the potential would change this number instead of leaving a stale
    literal behind.

    A gain of exactly one is R13's "no isolated fixed point" case, and on this
    circuit that is correct physics rather than a defect: Kirchhoff's voltage
    law round a loop is a CONSTRAINT, and the potential has an arbitrary datum,
    so there is nothing in it to converge to.  The determined quantity is the
    current, which `CircuitSolve._closed_form` solves for directly -- and that
    direct solve is exactly the repair R13's refusal names.
    """
    v = 1.0
    eps = 1.0

    def traverse(start: float) -> float:
        v = start
        for name in order:
            e = elements[name]
            e.v_in = v
            v = v - e.emf(omega) + current * e.resistance
        return v

    return float((traverse(v + eps) - traverse(v)) / eps)


def make_elements() -> dict:
    return {"MGU": MachineAgent(), "BUS": BusLeg(),
            "BATT": BatteryLeg(), "INV": InverterLeg()}


def _rotor() -> Any:
    return WA.RotorDisk(agent_id="ROTOR", u_ref=np.full(WA.ROTOR_CELLS, U_REF))


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------


def _elec_ports(expert: _Element, upstream: str, downstream: str,
                motion: MotionClass) -> list:
    """One terminal per face.  **Two ports, not one**: an element has two
    terminals, they meet different agents, and flattening them to one port per
    agent is what `port-algebra-atlas-0.1` §3.2 says a per-agent list cannot
    express."""
    out = []
    for face, other in (("lo", upstream), ("hi", downstream)):
        out.append(port_decl(
            name=f"{face}:ELEC", port_type=PortType.ELEC,
            geometry=f"{face} terminal of {expert.agent_id}, meeting {other}",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(ELEC_SCALES),
            effective_resolution=M_LUMPED,
            motion_class=motion,
            # ELEC returns the EFFORT (a potential) against a current-density
            # trace, which is the same half MECH and ADVEC return in this
            # package and the opposite of THERM's.
            response_half=ResponseHalf.EFFORT,
            prolongation=terminal_prolongation(expert.agent_id,
                                               f"{face}:ELEC"),
            note="a lumped circuit terminal: one potential over a declared "
                 "cross-section, which is what makes the current DENSITY the "
                 "port's flow variable rather than the current",
        ))
    return out


def element_capabilities(expert: _Element, upstream: str, downstream: str,
                         motion: MotionClass = MotionClass.STATIC
                         ) -> ExpertCapabilities:
    ports = _elec_ports(expert, upstream, downstream, motion)
    rot = getattr(expert, "respond_rot", None)
    if rot is not None:
        ports.append(port_decl(
            name="shaft:ROT", port_type=PortType.ROT,
            geometry="the machine's shaft",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(ROT_SCALES),
            effective_resolution=M_LUMPED,
            motion_class=motion,
            response_half=ResponseHalf.EFFORT,
            prolongation=shaft_prolongation(expert.agent_id),
            note="the reaction torque the circuit lets the machine carry at "
                 "this shaft speed: T = k_t I with I set by (k_e w - V_oc)/R",
        ))

    def respond(port_name: str, trace):
        if port_name.endswith(":ROT") and rot is not None:
            return rot(port_name, trace)
        return expert.respond_elec(port_name, trace)

    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=ports,
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # KNOWN, unlike a checkpoint's: Ohm's law contains no solve.
        elliptic_subsolve=EllipticSubsolve.NONE,
        stencil_radius=0,
        # A DC circuit with no inductance settles in microseconds against a
        # rotor's tenths of a second, so the element is quasi-steady and its
        # native step IS the macro-step. CS-13 learnt this from `L7/R9`
        # refusing a clock ratio its legs did not have; declared up front here.
        substeps_per_macro_step=1,
        time_discretization=TimeDiscretization.IMPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=1.0,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        # One family for the whole circuit, including the machine: a
        # motor-generator IS a circuit element with an EMF, and describing it as
        # its own continuum problem would fail E3 at both of its terminals for
        # no gain. The SHAFT seam is genuinely cross-family and fails E3, which
        # is correct and is what a drivetrain seam is.
        governing_family="lumped-dc-circuit",
        lambda_ref="itself: a closed form has no infidelity to measure",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"powertrain-{expert.agent_id}-R{expert.resistance:.6g}",
        boundary_response=respond,
        probe_base=lambda port, _e=expert: _e.base_for(port),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="closed form, zero fitted parameters, the disk.ActuatorDisk pattern",
    )


def rotor_capabilities(expert: Any,
                       motion: MotionClass = MotionClass.STATIC
                       ) -> ExpertCapabilities:
    """`wake_array`'s own record, with the shaft port's note corrected.

    **The declaration is that module's and is not re-derived here**, which is
    the point: the open port this case study connects has to be the same port
    `wake_array` declares, or connecting it proves nothing. Only the note
    changes, because the sentence *"unconnected: no drivetrain"* is now false.
    """
    from dataclasses import replace as _replace

    caps = WA.rotor_capabilities(expert)
    ports = []
    for p in caps.ports:
        if p.name == "shaft:ROT":
            p = _replace(
                p,
                direction=Direction.BIDIRECTIONAL,
                prolongation=shaft_prolongation(expert.agent_id),
                note="CONNECTED, as of CS-14: the shaft drives a "
                     "motor-generator and the extracted power is followed to "
                     "where it is stored and to where it is lost. The note this "
                     "replaces said 'unconnected: no drivetrain', and that "
                     "sentence is what rung 7 exists to make false",
            )
        ports.append(p)
    return _replace(caps, ports=tuple(ports))


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def connections(close_loop: bool = True) -> list[Connection]:
    """The shaft seam, and one ELEC seam per element of the circuit.

    Built from `CIRCUIT_ORDER` rather than written out, so the seam list and the
    solve cannot come to disagree about which element feeds which. The last seam
    closes the ring; `close_loop=False` drops it, which is W163's control.
    """
    conns = [Connection(
        seam_id="shaft", a=("ROTOR", "shaft:ROT"), b=("MGU", "shaft:ROT"),
        port_type=PortType.ROT, derive_space=True,
        geometrically_coincident=True,
        # A shaft has one degree of freedom and a nonzero speed always costs a
        # nonzero torque here, so nothing in the trace space is in the kernel.
        expected_null_dim=0,
        # **W138.** Both sides return the torque about the SAME axis in the same
        # sense -- the rotor what it delivers, the machine what it takes -- so
        # the well-posed condition is their difference, exactly as `wing_fsi`'s
        # wet seam and `cooling_loop`'s wall.
        effort_normal="MGU",
        note="the seam rung 7 exists for: `wake_array`'s open shaft port, "
             "connected. Fluid on one side, a circuit on the other, and E3 "
             "fails across it because they really do solve different equations")]
    n = len(CIRCUIT_ORDER)
    for i in range(n if close_loop else n - 1):
        a, b = CIRCUIT_ORDER[i], CIRCUIT_ORDER[(i + 1) % n]
        closes = (i == n - 1)
        conns.append(Connection(
            seam_id=f"{a.lower()}_{b.lower()}",
            a=(a, "hi:ELEC"), b=(b, "lo:ELEC"),
            port_type=PortType.ELEC, derive_space=True,
            geometrically_coincident=True, expected_null_dim=0,
            orientation=f"current goes {a} -> {b}",
            note=("the RETURN conductor, and it is what makes this a circuit "
                  "rather than a chain: the current re-enters the machine it "
                  "left, so the graph has a DIRECTED cycle and no element can "
                  "go first. W163's second such graph, on different physics"
                  if closes else
                  f"current leaves {a} and enters {b}")))
    return conns


#: Nothing is measured on this graph, and the record says so rather than
#: carrying another graph's numbers.  W56's backstop is what turns an absent
#: ledger into `admit-uncertified` rather than a silent pass.
MEASURED_NONE = MeasuredConstants(
    probe_state=f"rotor at u = {U_REF}, a = 1/3; battery at V_oc = {V_OC} V",
    scheme="one real actuator disk and four lumped circuit elements; ROT at "
           "the shaft and ELEC round the loop, solved rather than swept",
    depth=0,
    source="not measured: CS-14 declares the topology and the port type and "
           "does not claim a constant. L, sigma, C_mu and tau are open here")


def make_agents(motion: MotionClass = MotionClass.STATIC,
                experts: dict | None = None) -> tuple[list[Agent], dict]:
    if experts is None:
        experts = dict(make_elements())
        experts["ROTOR"] = _rotor()
    agents = [Agent("ROTOR", rotor_capabilities(experts["ROTOR"], motion),
                    domain="the rotor disk plane", role="turbine")]
    n = len(CIRCUIT_ORDER)
    for i, name in enumerate(CIRCUIT_ORDER):
        up = CIRCUIT_ORDER[(i - 1) % n]
        down = CIRCUIT_ORDER[(i + 1) % n]
        agents.append(Agent(
            name, element_capabilities(experts[name], up, down, motion),
            domain=_ELEMENT_DOMAIN[name], role="electrical"))
    return agents, experts


_ELEMENT_DOMAIN = {
    "MGU": "the motor-generator's windings and air gap",
    "BUS": "the DC harness between the machine and the battery",
    "BATT": "the battery pack",
    "INV": "the inverter's conduction path",
}


def build(motion: bool = False,
          experts: dict | None = None,
          measured: MeasuredConstants | None = "default",
          close_loop: bool = True,
          ) -> tuple[CaseGraph, dict]:
    """The drivetrain as a graph.

    ``close_loop=False`` is **W163's control on a second graph**: drop the
    return conductor and the same five agents form a chain. Everything else is
    identical, so anything the compile says about one and not the other is a
    statement about the CYCLE and about nothing else. On CS-13 it said nothing;
    whether it says nothing here too is the point of running it.
    """
    mc = MotionClass.SOLUTION_DEPENDENT if motion else MotionClass.STATIC
    if measured == "default":
        measured = MEASURED_NONE
    agents, experts = make_agents(mc, experts)
    return (
        CaseGraph(
            name="powertrain" + ("" if close_loop else "-open-control"),
            agents=agents,
            connections=connections(close_loop),
            # **W163, 2026-09-09.** Declared, and it is NOT a contraction -- see
            # `composed_loop_gain`. `L2/R13` therefore refuses the swept scheme
            # on this graph, which is the conclusion `CircuitSolve` had already
            # reached by hand when it chose to solve the loop rather than sweep
            # it. The declaration is what lets the compiler reach it too.
            loop_gains=() if not close_loop else (DeclaredLoopGain(
                agents=tuple(CIRCUIT_ORDER),
                gain=composed_loop_gain(experts, CIRCUIT_ORDER),
                source="composed from the elements' own emf/resistance map, "
                       "V -> V - emf + I R, which is a translation in the "
                       "potential the ELEC seam carries",
                note="exactly 1, and structurally so: KVL round a loop is a "
                     "constraint and the potential has an arbitrary datum, so "
                     "there is no isolated fixed point in it. The determined "
                     "quantity is the current, and CircuitSolve solves for it "
                     "directly rather than sweeping"),),
            # **W162, 2026-09-09**, and for `cooling_loop`'s reason on different
            # physics: declared NONE rather than left silent. A circuit's
            # elements meet at TERMINALS -- one pair per seam, at distinct
            # places round the loop -- so no cell belongs to two seams and there
            # is no vertex for a multiplier to be multi-valued at. The adjacency
            # of a closed loop of four elements has no triangle, so this
            # declaration changes no verdict here; it is written because the
            # claim is a property of the CIRCUIT and not of its length, and a
            # five-element loop with a shunt would have one.
            cross_points=(),
            # Five agents, five disjoint elements, no region cut and no
            # partition of unity -- `thermal_seam`'s and `cooling_loop`'s shape.
            decomposition=Decomposition.NON_OVERLAPPING,
            macro_dt=MACRO_DT,
            measured=measured,
            note=("CS-14: a drivetrain on the wake array's open shaft. The "
                  "first graph to carry ELEC, the first cross-domain energy "
                  "balance that is not an identity of its own solve, and "
                  "W163's second directed cycle")),
        experts,
    )


def compile_report(close_loop: bool = True) -> str:
    from ..compiler import compile_scheme
    g, _e = build(close_loop=close_loop)
    return compile_scheme(g).report()
