"""CS-13: a CLOSED coolant circuit -- the first graph whose ports form a cycle.

Rung 6 of `f1-pathmap-and-end-goal`'s ladder.  A solid block is cooled by a
liquid loop that leaves it hot, rejects heat at a radiator, is driven by a pump
and **returns to the block it started from**.

What only this case study can say
---------------------------------

**It is the first graph in which information circulates.**
`spec-wind-farm-wake` section 5.1 raised path-dependence of message passing as
a real concern -- *"I -> N -> F -> W and I -> B+ -> ... -> W are two distinct
paths between the same agents; information can circulate"* -- and then
`wind-farm-implementation-log` **withdrew** the one measurement that had been
read as evidence for it: the mirror residual was the frozen operator's own
asymmetry, reproduced with no graph, no ports and no agents anywhere in the
loop.  The log's own words are that sweep-order path dependence *"may still be
present -- nothing here rules it out. What is ruled out is that it is NEEDED."*

So the concern has been open since 2026-08-23 with **no graph that can test
it**, and the reason is specific: every cyclic graph this package has built is
an **overlapping tiling**, where the cycle is undirected and the scheme is
ADDITIVE.  `front_wing`'s six fluid windows contain the 4-cycle
``F00 - F10 - F11 - F01 - F00``, and it tests nothing, because all six windows
step from the same assembled field and are blended: no window's input is
another window's output, so there is no sweep and no order for an answer to
depend on.

A coolant loop is the opposite and it is the minimal case.  The cycle is
**directed**, it is carried by `ADVEC` -- a mass flux, which has a direction --
and every agent's inlet is another agent's outlet:

    BLOCK --THERM-- PASS --ADVEC--> RAD --ADVEC--> PUMP --ADVEC--> PASS
                     ^                                              |
                     +----------------------------------------------+

**There is no agent that can go first.**  That is the property, it is a
property of the topology rather than of any expert, and it is what no graph in
this package has had.

**Measured, it is worth 1.5 K.**  A composition layer that sweeps the circuit
once per macro-step -- which is what one does when nobody thinks about it --
gives eight different answers for eight equally defensible orders, spread over
1.54 K (4.8e-3 relative), and the spread is still 1.3 K after eight sweeps.
Solving the loop instead of sweeping it removes it entirely: every order
iterated to convergence agrees to 1e-11 relative.  So the cure is not an
ordering rule, it is `general-coupling-scheme` section 4.2's rule -- write the
residual in the port's own conjugate variables and solve it there -- applied to
a cycle rather than to a seam.

**And the compiler cannot tell this circuit from a chain.**  Compiled with the
return seam and compiled without it, the graph gets the same verdict, the same
rule set and the same per-agent decisions; the only difference in the whole
decision record is the extra seam's own per-seam rows.  Nine layers of
admissibility currently see a directed cycle as a list of independent seams.
That is recorded as a gap and **no rule was written for it here**, because a
rule invented the same day the first cyclic graph appeared would be a rule
validated on one graph.

The gate: thermal balance closes around the loop
------------------------------------------------

At the fixed point of the loop, the coolant returns to the state it left, so

    Q_block  =  Q_rad  +  W_pump                                   (1)

exactly, in watts.  Every term is read from a different agent, and (1) is the
statement that the cycle is consistent -- an assembly whose loop does not close
is manufacturing or destroying energy at a seam, which is the failure the whole
port algebra exists to make visible.  It is checked in
`tests/test_tier36_cooling_loop.py` against a residual normalised by ``Q_block``.

**And the loop's fixed point is the thing that has to exist at all.**  A
directed cycle is a fixed-point problem in the return temperature and not a
sweep, so `LoopSolve` solves it -- and because every lumped leg is affine in
its inlet temperature, the loop map is affine and its fixed point is available
**in closed form** beside the iteration.  The two agree to round-off, which is
this case study's own positive control: an iteration that converged to the
wrong thing and a closed form that was never checked are the same artifact.

What is NOT claimed
-------------------

**No compiler layer and no rule was added for this graph**, and that is
deliberate.  If a directed cycle needs a rule -- an ordering constraint, a
fixed-point admissibility condition, a statement about which sweeps are legal
-- then the honest output of this case study is a **named gap** and not a rule
written the same day the first cyclic graph appeared.  What the compile reports
is recorded as it comes.

**The lumped legs are `disk.ActuatorDisk`'s class**: closed-form, zero fitted
parameters, and each one is four lines of algebra.  They buy a port TYPE and a
topology, not physics.  The one real expert is `BLOCK` --
`thermostruct2d.ThermoStruct2D.step_thermal`, the build repo's Q1 conduction
solver, imported unmodified -- and it is what makes the `THERM` seam a
conjugate-heat-transfer seam rather than a fixture.

Run::

    python -c "from atlas.cases import cooling_loop as C; \\
               print(C.compile_report())"
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ..capability import (BCChannel, ClaimType, Differentiable, Direction,
                          EllipticSubsolve, ExpertCapabilities, MotionClass,
                          TimeDiscretization, port_decl)
from ..graph import Agent, CaseGraph, Connection, Decomposition, MeasuredConstants
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation

__all__ = [
    "BlockAgent", "CoolantLeg", "RadiatorLeg", "LineLeg", "PumpLeg",
    "LoopSolve", "LoopResult", "SweepStudy", "SWEEP_ORDERS",
    "build", "connections", "compile_report",
    "N_SEAM", "M_EFF", "MDOT", "CP_COOLANT", "Q_SOURCE", "T_AMB",
    "UA_RAD", "UA_LINE", "W_PUMP", "H_WALL", "MACRO_DT", "LOOP_ORDER",
]

DEFAULT_BUILD_REPO = os.path.join(os.path.expanduser("~"),
                                  "physics-foundation-model")


# ---------------------------------------------------------------------------
# the circuit, in numbers
# ---------------------------------------------------------------------------

#: Cells along the wetted passage.  The same 48 as `thermal_seam`, so the
#: seam's discretization is a declaration this package has already exercised
#: rather than a new one chosen here.
N_SEAM = 48
M_EFF = 16                       # declared interface resolution on the WALL

#: **The ADVEC ports are LUMPED and their resolution is 1.**  `L4/null-space`
#: refused the first version of this case study, which declared `M_EFF` on
#: them: a leg with one state responds only to the mean of its inlet, so the
#: probed 16-mode block had 15 null directions and the rule reported all
#: fifteen.  `PORT_SPECS[ROT]`'s note says the shape out loud -- *"a lumped
#: port is the case dim M = 1"* -- and declaring 16 modes on a one-state
#: element is `CASE-STUDY-GUIDE`'s *"claiming resolution nobody measured"* with
#: the arithmetic to prove it.
M_LUMPED = 1
L_Z = 0.20                       # m, passage length
H_SEAM = L_Z / N_SEAM
T_BLOCK = 0.02                   # m, block thickness
NJ_BLOCK = 6                     # elements through the thickness
WIDTH = 0.05                     # m, passage width out of plane -- the wetted
                                 # area is L_Z * WIDTH

#: The coolant.  Water-glycol-like; reference defaults, not a spec.
MDOT = 0.15                      # kg/s
CP_COOLANT = 3600.0              # J/(kg K)

#: The block's heat source, the radiator's conductance and the pump's work.
Q_SOURCE = 1800.0                # W, dissipated in the block
UA_RAD = 42.0                    # W/K, radiator conductance to ambient
UA_LINE = 0.4                    # W/K, a pipe run's own loss to ambient
W_PUMP = 45.0                    # W, pump work, all of it into the coolant
T_AMB = 300.0                    # K
T_COOLANT_0 = 320.0              # K, the loop's release temperature

#: The conjugate-heat-transfer film coefficient on the wetted face.
H_WALL = 1200.0                  # W/(m^2 K)
H_OUTER = 8.0                    # W/(m^2 K), the block's dry face to ambient

#: Clocks.  The block conducts slowly and the coolant transports fast; the
#: macro-step is the block's, which is the slow agent, exactly as `thermal_seam`
#: and `brake_thermal` do it.
MACRO_DT = 5.0e-2                # s

#: **A quasi-steady algebraic leg has no clock of its own, and `L7/R9` is what
#: made that explicit.**  The first version declared a residence-scale step of
#: 1 ms, which put a 50:1 clock ratio on every ADVEC seam and R9 refused the
#: graph for pointwise flux matching across different clocks -- correctly, for
#: a ratio the legs do not have.  ``T_out = T_in + Q / (mdot cp)`` contains no
#: time: the leg responds within the macro-step and its native step IS the
#: macro-step.  The residence time is kept as a declared property because it is
#: the scale at which that quasi-steady assumption would stop holding.
DT_LEG = MACRO_DT                # s, the leg's native clock: it has no other
RESIDENCE_TIME = 2.0e-3          # s, L / u; the quasi-steady assumption's edge

#: (T, q_n/T) in K and W/(m^2 K): s_e s_f = s_P with s_P in W/m^2.
THERM_SCALES = {"temperature": 350.0, "entropy_flux": 1.0e4 / 350.0,
                "power_area": 1.0e4}

#: (h0, rho u.n) in J/kg and kg/(m^2 s): s_e s_f = s_P with s_P in W/m^2.
#: The passage cross-section is `WIDTH * T_PASSAGE`, so the mass FLUX is
#: `MDOT / area`; declaring the flux rather than the rate is what keeps the
#: power scale an area density like every other port in the package.
T_PASSAGE = 0.006                # m, passage height
A_PASSAGE = WIDTH * T_PASSAGE    # m^2
MASS_FLUX = MDOT / A_PASSAGE     # kg/(m^2 s)
#: **L3/C4 refused the first version of this dict and it was right to.**
#: ADVEC is a MULTIBOND: the base triple describes the port, and every
#: passenger needs its own conjugate pair on top, because a mass flux carries
#: several conserved quantities at once and each one has its own power. With
#: one passenger the passenger pair equals the base pair, which is exactly why
#: leaving it out looked harmless and is not: a two-passenger port written the
#: same way would silently price the second passenger at the first's scale.
ADVEC_SCALES = {"enthalpy": CP_COOLANT * 350.0,
                "mass_flux": MASS_FLUX,
                "power_area": CP_COOLANT * 350.0 * MASS_FLUX,
                "h0_effort": CP_COOLANT * 350.0,
                "h0_flow": MASS_FLUX,
                "h0_power": CP_COOLANT * 350.0 * MASS_FLUX}


def build_repo() -> str:
    return os.environ.get("ATLAS_BUILD_REPO", DEFAULT_BUILD_REPO)


@lru_cache(maxsize=1)
def load_solvers(module_name: str = "atlas_build_solvers"):
    """Import the build repo's `atlas` package under a private name.

    The same loader `thermal_seam` uses, and for the same reason: both
    repositories have a top-level package called ``atlas``, and the solvers
    reach up into their own package (``grid.py`` imports ``..config``), so the
    whole package is what has to be bound, under a name that collides with
    nothing.  Shared by module name, so a process that has loaded it once for
    `thermal_seam` does not load it twice.
    """
    pkg_dir = os.path.join(build_repo(), "src", "atlas")
    init = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init):
        raise RuntimeError(
            f"the expert is not where this case study expects it: {init!r} does "
            "not exist. thermostruct2d lives in the build repo "
            "(github.com/nonidino/physics-foundation-model, src/atlas/solvers/); "
            "set ATLAS_BUILD_REPO to the checkout."
        )
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, init, submodule_search_locations=[pkg_dir]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(module_name + ".solvers.thermostruct2d")


# ---------------------------------------------------------------------------
# the interface basis -- identical to every other seam in the package
# ---------------------------------------------------------------------------


def fourier_basis(n_cells: int = N_SEAM, m: int = M_EFF) -> np.ndarray:
    """The same 16-mode real Fourier prolongation five case studies already use.

    Identical to `thermal_seam.fourier_basis` on purpose: the basis is a
    declaration, and comparing experts through two different declarations
    confounds the expert with the presentation.
    """
    length = n_cells * H_SEAM
    y = (np.arange(n_cells) + 0.5) * H_SEAM
    cols = [np.full(n_cells, 1.0 / np.sqrt(length))]
    k = 1
    while len(cols) < m:
        w = 2.0 * np.pi * k * y / length
        cols.append(np.sqrt(2.0 / length) * np.cos(w))
        if len(cols) < m:
            cols.append(np.sqrt(2.0 / length) * np.sin(w))
        k += 1
    return np.column_stack(cols[:m])


def face_prolongation(agent_id: str, port_name: str) -> Prolongation:
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=fourier_basis(),
        gram_V=H_SEAM * np.eye(N_SEAM),
        label="16-mode real Fourier basis (probed-dtn-coupling 2.2)",
    )


def lumped_prolongation(agent_id: str, port_name: str) -> Prolongation:
    """The one-dimensional prolongation of a lumped port: the constant mode.

    A leg has one state, so its interface space is one-dimensional and the
    prolongation is the normalised constant. `gram_V` is the same face measure
    the wall uses, so the two seams are stated in one metric.
    """
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=np.full((N_SEAM, 1), 1.0 / np.sqrt(N_SEAM * H_SEAM)),
        gram_V=H_SEAM * np.eye(N_SEAM),
        label="the constant mode: a lumped port is the case dim M = 1",
    )


def _as_entropy_flux(q_n: np.ndarray, T_face: np.ndarray) -> np.ndarray:
    """THERM's declared FLOW: the TRUE bond's ``q_n / T``, not the pseudo-bond.

    W66's measurement is that returning ``q_n`` instead makes `response_half`
    a lie nothing numerical can catch, so the conversion happens here, once,
    in the one place both agents call.
    """
    return q_n / np.maximum(T_face, 1.0)


# ---------------------------------------------------------------------------
# agent 1 -- the block.  The only real solver in this graph.
# ---------------------------------------------------------------------------


@dataclass
class BlockAgent:
    """`thermostruct2d.ThermoStruct2D` on the block section, seam on the wetted face.

    Conjugate heat transfer in the ordinary sense: the coolant presents a bulk
    temperature, the block's own **Robin** channel takes it as ``T_gas`` with
    the wall film coefficient, and what comes back is the heat that crossed the
    wetted face.  `step_thermal` is called unmodified.

    The heat source is carried as a raised **outer**-face gas temperature rather
    than as a body term, because `step_thermal` takes Robin data on two faces
    and no volumetric source, and inventing a source would mean editing the
    donor.  ``T_SOURCE`` is set so the steady dissipation through the dry face
    is ``Q_SOURCE`` at the release state -- a declaration of the operating
    point, and `heat_in` measures what actually crosses rather than assuming it.
    """

    agent_id: str = "BLOCK"
    dt: float = MACRO_DT
    expose_elliptic: bool = False
    h_wall: float = H_WALL
    _ts: Any = field(default=None, repr=False)
    _mesh: Any = field(default=None, repr=False)
    _T: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        TS = load_solvers()
        nodes = np.stack(np.meshgrid(np.linspace(0.0, L_Z, N_SEAM + 1),
                                     np.linspace(0.0, T_BLOCK, NJ_BLOCK + 1),
                                     indexing="ij"), -1)
        self._mesh = TS.ShellMesh(nodes)
        self._ts = TS.ThermoStruct2D(self._mesh, TS.SolidMaterial())
        self._T = np.full(self._mesh.n_nodes, T_COOLANT_0 + 40.0)

    # -- the source, declared so the steady state is the one intended --------

    @property
    def area(self) -> float:
        """Wetted area of the passage face, m^2."""
        return L_Z * WIDTH

    @property
    def t_source(self) -> float:
        """Outer-face gas temperature that drives ``Q_SOURCE`` through the block.

        A declaration of the operating point: at steady state the dry face
        carries ``H_OUTER * A * (T_source - T_wall)``, so inverting it at the
        release wall temperature fixes the number.  What the block actually
        delivers is MEASURED by `heat_in`; this only sets where the run sits.
        """
        return T_COOLANT_0 + 40.0 + Q_SOURCE / (H_OUTER * self.area)

    def _face_T(self, T: np.ndarray, which: str = "inner") -> np.ndarray:
        a, b, _L = self._ts._face(which)
        return 0.5 * (T[a] + T[b])

    def step(self, T_coolant: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Advance the block one step against a coolant bulk temperature."""
        dt = self.dt if dt is None else dt
        self._T = self._ts.step_thermal(
            self._T, dt, self.h_wall, np.asarray(T_coolant, dtype=float),
            H_OUTER, self.t_source, radiate=False, T_inf=T_AMB)
        return self._T

    def heat_in(self, T_coolant: np.ndarray, T: np.ndarray | None = None) -> float:
        """Watts crossing the wetted face into the coolant.  Positive = cooling.

        The integral of ``h (T_wall - T_coolant)`` over the face, with the cell
        measure ``H_SEAM * WIDTH``.  This is the term (1) reads as ``Q_block``,
        and it is measured from the block's own field rather than assumed equal
        to ``Q_SOURCE``: at a fixed point they agree, and away from one they do
        not, which is what makes the balance a gate.
        """
        T = self._T if T is None else T
        Tw = self._face_T(T, "inner")
        Tc = np.asarray(T_coolant, dtype=float).reshape(N_SEAM)
        return float(np.sum(self.h_wall * (Tw - Tc)) * H_SEAM * WIDTH)

    # -- the port ----------------------------------------------------------

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(coolant temperature on the seam) -> (THERM flow on the seam).

        One `step_thermal` from the agent's stored field, exactly as
        `thermal_seam.ShellAgent.respond` does it: the probe must not advance
        the state it is linearizing about.
        """
        Tc = np.asarray(trace, dtype=float).reshape(N_SEAM)
        T = self._ts.step_thermal(self._T, self.dt, self.h_wall, Tc,
                                  H_OUTER, self.t_source, radiate=False,
                                  T_inf=T_AMB)
        Tw = self._face_T(T, "inner")
        q = self.h_wall * (Tw - Tc)                # W/m^2, into the coolant
        return _as_entropy_flux(q, 0.5 * (Tw + Tc))

    def heat_outer(self, T: np.ndarray | None = None) -> float:
        """Watts crossing the DRY face into the block.  The source term."""
        T = self._T if T is None else T
        Tw = self._face_T(T, "outer")
        return float(np.sum(H_OUTER * (self.t_source - Tw)) * H_SEAM * WIDTH)

    def thermal_energy(self, T: np.ndarray | None = None) -> float:
        """``T^T M T / 2``'s linear partner: the block's stored heat, in joules.

        ``M_th`` is the Q1 mass matrix carrying ``rho c_p``, so ``1^T M T`` is
        the volume integral of ``rho c_p T`` -- the extensive quantity a first
        law is written about.  `storage` returns the quadratic Lyapunov
        functional instead, which is the right object for E7 and the wrong one
        here; they are different questions and both are declared.
        """
        T = self._T if T is None else np.asarray(T, dtype=float)
        return float(np.sum(self._ts.M_th.dot(T)))

    def energy_balance(self, T_coolant: np.ndarray, dt: float | None = None
                       ) -> dict:
        """**The gate that can fail.**  First law across one step of the block.

        ``d/dt integral rho c_p T dV  =  Q_outer - Q_wall``, with both fluxes
        read through the same face machinery `heat_in` uses.  Nothing in the
        loop algebra makes this true: it is a statement about `step_thermal`
        and about whether this module reads its faces, its signs and its cell
        measure correctly.  Get the wetted face wrong, drop the out-of-plane
        width, or flip a sign, and the residual is order one.

        The out-of-plane depth is `WIDTH` for the fluxes and **1 m for the mass
        matrix**, which is what a plane 2-D FE assembly integrates over, so the
        stored term is scaled by `WIDTH` to put both sides in the same watts.
        That factor is the one thing here worth stating out loud: it is the
        kind of bookkeeping that cancels out of every ratio and does not cancel
        out of a first law.
        """
        dt = self.dt if dt is None else dt
        Tc = np.asarray(T_coolant, dtype=float).reshape(N_SEAM)
        T0 = self._T.copy()
        T1 = self._ts.step_thermal(T0, dt, self.h_wall, Tc, H_OUTER,
                                   self.t_source, radiate=False, T_inf=T_AMB)
        stored = WIDTH * (self.thermal_energy(T1) - self.thermal_energy(T0)) / dt
        #: backward Euler evaluates its fluxes at the END of the step, so the
        #: balance is taken there. Reading them at T0 is a first-order error and
        #: would make this gate report the scheme's own time discretisation as a
        #: conservation defect.
        q_out = self.heat_outer(T1)
        q_wall = self.heat_in(Tc, T1)
        resid = stored - (q_out - q_wall)
        scale = max(abs(q_out), abs(q_wall), 1.0e-30)
        return {"stored_rate": stored, "q_outer": q_out, "q_wall": q_wall,
                "residual": resid, "relative": abs(resid) / scale}

    def base_trace(self) -> np.ndarray:
        """W74: this port's effort is an ABSOLUTE temperature, so the probe's
        default zero trace is 0 K and the operating point has to be declared."""
        return np.full(N_SEAM, T_COOLANT_0)

    def storage(self, state=None) -> float:
        """``1/2 T^T M T`` -- the conduction operator's own Lyapunov functional,
        SPD by construction from the Q1 mass matrix."""
        T = self._T if state is None else np.asarray(state)
        return float(0.5 * T @ self._ts.M_th.dot(T))

    def validity(self, state=None, cond=None) -> bool:
        return True


# ---------------------------------------------------------------------------
# agents 2-4 -- the loop's three lumped legs
# ---------------------------------------------------------------------------


@dataclass
class _Leg:
    """A lumped coolant leg: an affine map from inlet to outlet temperature.

    Every leg in this circuit has the form ``T_out = a T_in + b`` with ``a`` and
    ``b`` closed-form functions of declared constants and, for the passage, of
    the heat the block delivers.  **That affineness is not a convenience, it is
    what makes the loop's fixed point available in closed form** beside the
    iteration that finds it -- see `LoopSolve`.

    Zero fitted parameters, four lines of algebra: `disk.ActuatorDisk`'s class.
    """

    agent_id: str = "LEG"
    dt: float = DT_LEG
    t_in: float = T_COOLANT_0

    # -- the affine coefficients, which is all a leg IS --------------------

    def coeffs(self, q_w: float = 0.0) -> tuple[float, float]:
        raise NotImplementedError

    def outlet(self, t_in: float | None = None, q_w: float = 0.0) -> float:
        a, b = self.coeffs(q_w)
        return a * (self.t_in if t_in is None else float(t_in)) + b

    # -- ports -------------------------------------------------------------

    def h0(self, t: float) -> float:
        """ADVEC's declared EFFORT: specific total enthalpy, J/kg."""
        return CP_COOLANT * float(t)

    def respond_advec(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(inlet mass flux) -> (outlet specific total enthalpy).

        The ADVEC bond is ``(h0, rho u.n)``; this leg's response half is the
        EFFORT, matching `rocket.py`'s convention for every ADVEC port in the
        package.  The leg is a one-dimensional transport element, so the
        response is uniform along the port and the trace enters through the
        mass flux it carries.
        """
        m = np.asarray(trace, dtype=float).reshape(-1)
        scale = float(np.mean(m)) / MASS_FLUX if MASS_FLUX else 0.0
        return np.full(m.shape, self.h0(self.outlet()) * (1.0 + scale))

    def storage(self, state=None) -> float:
        """``m cp T^2 / 2``: the leg's own thermal energy, positive definite."""
        t = self.t_in if state is None else float(np.mean(np.asarray(state)))
        return float(0.5 * MDOT * self.dt * CP_COOLANT * t * t)

    def validity(self, state=None, cond=None) -> bool:
        return True

    def base_trace(self) -> np.ndarray:
        return np.full(N_SEAM, MASS_FLUX)

    def base_for(self, port) -> np.ndarray:
        """The operating point of THIS port's own variable (W74).

        A leg carries two kinds of port and they linearize about two different
        physical quantities. Dispatching on the port name is the whole fix, and
        the reason it is worth a method rather than a lambda is that a leg that
        grows a third port type should have to come past this.
        """
        name = getattr(port, "name", str(port))
        if name.endswith(":THERM"):
            return np.full(N_SEAM, T_COOLANT_0)
        #: in V, beside the trace -- the prolongation is what takes it to M,
        #: and `L4/probe` refuses a base of the wrong length rather than
        #: reshaping it
        return np.full(N_SEAM, MASS_FLUX)


@dataclass
class CoolantLeg(_Leg):
    """The passage through the block.  Picks up whatever the wall delivers.

        T_out = T_in + Q_wall / (mdot cp)

    ``Q_wall`` comes from the block through the THERM seam, so this leg is the
    one whose coefficients depend on another agent -- which is exactly the edge
    that makes the graph a loop rather than a chain.
    """

    agent_id: str = "PASS"

    def coeffs(self, q_w: float = 0.0) -> tuple[float, float]:
        return 1.0, float(q_w) / (MDOT * CP_COOLANT)

    def wall_temperature(self) -> np.ndarray:
        """The bulk temperature this leg presents to the block, per seam cell.

        Uniform: a lumped leg has one state, and declaring a profile it does not
        have would be claiming resolution nobody measured.  The port still
        carries `N_SEAM` cells because the BLOCK side has that many, and the
        prolongation is what reconciles them.
        """
        return np.full(N_SEAM, self.t_in)

    def respond_therm(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(wall temperature) -> (THERM flow the coolant accepts).

        The mirror of the block's response through the same film law, which is
        what makes this a conjugate seam: ``q = h (T_wall - T_bulk)``, positive
        into the coolant, converted to the declared entropy flux.
        """
        Tw = np.asarray(trace, dtype=float).reshape(N_SEAM)
        Tb = self.wall_temperature()
        q = H_WALL * (Tw - Tb)
        return _as_entropy_flux(q, 0.5 * (Tw + Tb))


@dataclass
class RadiatorLeg(_Leg):
    """The radiator.  Effectiveness-NTU, in closed form.

        T_out = T_amb + (T_in - T_amb) exp(-UA / (mdot cp))

    and the heat rejected is ``mdot cp (T_in - T_out)``.  One exponential and no
    fitted parameters.
    """

    agent_id: str = "RAD"

    @property
    def decay(self) -> float:
        return math.exp(-UA_RAD / (MDOT * CP_COOLANT))

    def coeffs(self, q_w: float = 0.0) -> tuple[float, float]:
        d = self.decay
        return d, T_AMB * (1.0 - d)

    def heat_out(self, t_in: float | None = None) -> float:
        """Watts rejected to ambient.  Term ``Q_rad`` of the loop balance."""
        ti = self.t_in if t_in is None else float(t_in)
        return float(MDOT * CP_COOLANT * (ti - self.outlet(ti)))


@dataclass
class LineLeg(_Leg):
    """A run of pipe between two components.  The radiator's law, smaller ``UA``.

        T_out = T_amb + (T_in - T_amb) exp(-UA_line / (mdot cp))

    A line is not adiabatic and declaring it so would make it the identity,
    which is an agent that cannot be wrong.  ``UA_LINE`` is two orders below the
    radiator's, so it moves the balance without dominating it -- and it puts a
    second dissipative leg on the cycle, which is what stops the loop balance
    from being the two-term statement a sign error cancels out of.
    """

    agent_id: str = "HOT"
    ua: float = 0.0

    def coeffs(self, q_w: float = 0.0) -> tuple[float, float]:
        d = math.exp(-self.ua / (MDOT * CP_COOLANT))
        return d, T_AMB * (1.0 - d)

    def heat_out(self, t_in: float | None = None) -> float:
        ti = self.t_in if t_in is None else float(t_in)
        return float(MDOT * CP_COOLANT * (ti - self.outlet(ti)))


@dataclass
class PumpLeg(_Leg):
    """The pump.  All the shaft work lands in the coolant.

        T_out = T_in + W_pump / (mdot cp)

    It is in the loop because a circuit needs one, and because it puts a
    **source** on the cycle: without it the balance is trivially
    ``Q_block = Q_rad`` and a sign error in either would cancel.
    """

    agent_id: str = "COLD"
    ua: float = 0.0

    def coeffs(self, q_w: float = 0.0) -> tuple[float, float]:
        """The return line: it loses heat to ambient AND gains the pump work.

        Both on one leg because a circuit's cold side is one run of pipe with a
        pump in it, and splitting them into two agents would be adding a node
        to make a diagram tidier rather than to model anything.
        """
        d = math.exp(-self.ua / (MDOT * CP_COOLANT))
        return d, T_AMB * (1.0 - d) + W_PUMP / (MDOT * CP_COOLANT)

    def heat_out(self, t_in: float | None = None) -> float:
        """Heat rejected to ambient, NOT counting the pump work put in."""
        ti = self.t_in if t_in is None else float(t_in)
        d = math.exp(-self.ua / (MDOT * CP_COOLANT))
        return float(MDOT * CP_COOLANT * (ti - (d * ti + T_AMB * (1.0 - d))))


# ---------------------------------------------------------------------------
# the loop's own solve -- a fixed point, not a sweep
# ---------------------------------------------------------------------------


#: **The circuit, in one place.**  The legs in flow order, and everything that
#: needs to know the topology reads it from here: `connections` builds one
#: `ADVEC` seam per consecutive pair and one more to close the ring, `LoopSolve`
#: composes the legs in this order, and `SweepStudy` enumerates its rotations.
#: A case study whose graph and whose solve disagree about the circuit is
#: measuring neither.
#:
#: **Four legs and not three, and the reason is a finding.**  The first version
#: had three -- passage, radiator, pump -- and `L2/I2/G1` REFUSED it, because
#: `CaseGraph.detected_cross_points` infers a cross-point from a TRIANGLE in the
#: agent adjacency.  That proxy is right on a tiling, where three subdomains
#: meeting pairwise really do share a corner, and wrong on a circuit, where
#: three legs joined by three DISTINCT planes share no point at all.  Four legs
#: is also the more honest circuit -- a cooling loop has a hot line and a cold
#: line, not a radiator bolted to a pump -- so the fix is the better model, and
#: the triangle is kept reachable as `n_legs=3` rather than being written up
#: and thrown away.
LOOP_ORDER: tuple[str, ...] = ("PASS", "HOT", "RAD", "COLD")
LOOP_ORDER_3: tuple[str, ...] = ("PASS", "RAD", "COLD")


def loop_order(n_legs: int = 4) -> tuple[str, ...]:
    if n_legs == 4:
        return LOOP_ORDER
    if n_legs == 3:
        return LOOP_ORDER_3
    raise ValueError("n_legs is 4 (the circuit) or 3 (the triangle control)")


def make_legs(n_legs: int = 4) -> dict[str, "_Leg"]:
    """One expert per leg, keyed by the name `LOOP_ORDER` uses."""
    made = {"PASS": CoolantLeg(), "RAD": RadiatorLeg(),
            "HOT": LineLeg(agent_id="HOT", ua=UA_LINE),
            "COLD": PumpLeg(agent_id="COLD", ua=UA_LINE)}
    return {k: made[k] for k in loop_order(n_legs)}


@dataclass
class LoopResult:
    """One converged circuit state, with every term of the loop balance."""

    order: tuple[str, ...]
    temps: dict                      # leg -> its OUTLET temperature
    t_return: float                  # what re-enters the passage
    q_block: float                   # watts the block delivers to the coolant
    q_rejected: dict                 # leg -> watts it rejects to ambient
    w_pump: float
    residual: float                  # |Q_in - Q_out| / Q_in, around the loop
    iterations: int
    closed_form: float               # the fixed point, solved directly
    fixed_point_gap: float           # |iterated - closed form| / |closed form|
    energy_balance: dict = field(default_factory=dict)
    t_wall: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def as_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.__dict__.items()}


class LoopSolve:
    """Find the circuit's fixed point, and check it against the closed form.

    **A directed cycle has no first agent**, so the loop is not swept -- it is
    solved.  The unknown is the temperature at which coolant re-enters the
    passage; going once round the loop maps it to itself, and the fixed point of
    that map is the circuit's state.

    **Both routes are computed and the pair is the control.**  Every leg is
    affine in its inlet temperature, so the composed loop map is affine,
    ``T -> A T + B``, with ``A`` and ``B`` **composed from the legs' own
    ``coeffs``** rather than re-derived here -- a closed form written out by
    hand is a second source of truth, and this vault has W148's row about what
    that costs.  Its fixed point is ``B / (1 - A)`` exactly.  An iteration that
    converged to the wrong thing and a closed form that was never checked are
    the same artifact; each catches the other.

    ``Q_block`` is the one coefficient that is not a constant: it is what the
    real solver delivers at the current wall state, so the outer march re-reads
    it every macro-step.  That is the coupling, and it is why this is a fixed
    point in the coolant temperature rather than a formula.

    **What the loop balance is, and what it is not.**  At the fixed point

        Q_block + W_pump  =  sum over legs of Q_rejected                  (1)

    holds, and it is worth being exact about its status: **it is an identity of
    the fixed point, not an independent physical check.**  Substituting the
    legs' own definitions turns (1) into ``T_return = T_return``.  It is a
    consistency check on the arithmetic -- a sign error in any leg's
    ``heat_out`` breaks it -- and it is not evidence that the coupled physics is
    right.  The gate that CAN fail on physics is `BlockAgent.energy_balance`,
    which is a first law across the real solver and is reported beside it.
    """

    def __init__(self, block: BlockAgent | None = None, n_legs: int = 4,
                 tol: float = 1.0e-12, max_iter: int = 400) -> None:
        self.block = block if block is not None else BlockAgent()
        self.order = loop_order(n_legs)
        self.legs = make_legs(n_legs)
        self.tol = tol
        self.max_iter = max_iter

    # -- one pass round the loop -------------------------------------------

    def _once(self, t_in: float, q_block: float) -> dict:
        out, t = {}, t_in
        for name in self.order:
            leg = self.legs[name]
            leg.t_in = t
            t = leg.outlet(t, q_block if name == "PASS" else 0.0)
            out[name] = t
        return out

    def _closed_form(self, q_block: float) -> float:
        """``B / (1 - A)`` with A and B COMPOSED from the legs' own coeffs."""
        a_total, b_total = 1.0, 0.0
        for name in self.order:
            a, b = self.legs[name].coeffs(q_block if name == "PASS" else 0.0)
            a_total, b_total = a * a_total, a * b_total + b
        if abs(1.0 - a_total) < 1.0e-15:
            raise RuntimeError(
                "the loop map has unit gain, so it has no isolated fixed point. "
                "That is a statement about the circuit and not a solver failure: "
                "a loop with no dissipative leg does not settle"
            )
        return b_total / (1.0 - a_total)

    @property
    def loop_gain(self) -> float:
        """The composed contraction of one traversal: the product of the legs'
        own ``a`` coefficients.  Below one because the radiator dissipates."""
        g = 1.0
        for name in self.order:
            g *= self.legs[name].coeffs(0.0)[0]
        return g

    # -- the fixed point ---------------------------------------------------

    def solve(self, steps: int = 400) -> LoopResult:
        """March the block with the loop CLOSED at every macro-step."""
        t_in = T_COOLANT_0
        q_block = 0.0
        it = 0
        for it in range(1, steps + 1):
            self.legs["PASS"].t_in = t_in
            wall_t = self.legs["PASS"].wall_temperature()
            self.block.step(wall_t)
            q_block = self.block.heat_in(wall_t)
            t_new = self._closed_form(q_block)
            done = abs(t_new - t_in) < self.tol * max(1.0, abs(t_in))
            t_in = t_new
            if done:
                break

        #: the iterated route, run to convergence at the FINAL q_block, so the
        #: two routes are compared at the same coefficient
        t_it = t_in
        for _ in range(self.max_iter):
            nxt = self._once(t_it, q_block)[self.order[-1]]
            if abs(nxt - t_it) < self.tol * max(1.0, abs(t_it)):
                t_it = nxt
                break
            t_it = nxt
        closed = self._closed_form(q_block)

        temps = self._once(t_in, q_block)
        rejected = {n: float(self.legs[n].heat_out())
                    for n in self.order if hasattr(self.legs[n], "heat_out")}
        q_in = q_block + W_PUMP
        resid = abs(q_in - sum(rejected.values())) / max(abs(q_in), 1.0e-30)
        wall_t = self.legs["PASS"].wall_temperature()
        return LoopResult(
            order=self.order, temps=temps, t_return=temps[self.order[-1]],
            q_block=q_block, q_rejected=rejected, w_pump=W_PUMP,
            residual=resid, iterations=it, closed_form=closed,
            fixed_point_gap=abs(t_it - closed) / max(abs(closed), 1.0e-30),
            energy_balance=self.block.energy_balance(wall_t),
            t_wall=self.block._face_T(self.block._T, "inner").copy(),
        )


# ---------------------------------------------------------------------------
# what the cycle actually costs: sweep order, measured both ways
# ---------------------------------------------------------------------------


def sweep_orders(n_legs: int = 4) -> tuple[tuple[str, ...], ...]:
    """Every rotation of the circuit, in both directions.

    A cycle has no distinguished first agent, so these are ``2 n`` equally
    defensible schemes and nothing in the declaration prefers one -- which is
    `spec-wind-farm-wake` section 5.1's concern stated as a list.
    """
    o = loop_order(n_legs)
    n = len(o)
    fwd = tuple(tuple(o[(i + k) % n] for k in range(n)) for i in range(n))
    rev = tuple(tuple(reversed(x)) for x in fwd)
    return fwd + rev


SWEEP_ORDERS = sweep_orders(4)


@dataclass
class SweepStudy:
    """Does the answer depend on where you start going round the loop?

    **This is the measurement CS-13 exists for**, and it has two halves that
    point opposite ways.  Both are run, because reporting only one would be
    picking the answer.

    **Half 1 -- one sweep per macro-step, which is what a naive composition
    layer does.**  Visit the legs in some order, each using whatever its
    upstream neighbour most recently produced.  A leg visited before its
    upstream sees last pass's value and one visited after it sees this pass's,
    so the order is a modelling choice made by whoever wrote the loop and by
    nothing in the declaration.  The orders disagree, and the spread is the
    number.

    **Half 2 -- the fixed point, which is what `LoopSolve` does.**  Iterate any
    order to convergence and they all land on the same state, because a
    convergent fixed-point iteration forgets its starting order.  So the cure
    for path dependence on this graph is not an ordering rule: it is *solving
    the loop instead of sweeping it*, which is `general-coupling-scheme`
    section 4.2's rule -- write the residual in the port's own conjugate
    variables and solve it there -- applied to a cycle rather than to a seam.

    **[AI Inference]:** that generalises only as far as the loop map being a
    contraction.  Here it is, and demonstrably so: the composed gain is the
    product of the legs' own ``a`` coefficients, strictly below one because the
    radiator dissipates.  A circuit with no dissipative leg has a unit-gain loop
    map, no isolated fixed point, and nothing for an iteration to converge to --
    `LoopSolve._closed_form` raises there rather than returning a number, which
    is a statement about the circuit and not a solver failure.  Where the
    boundary between those regimes sits is not measured here.

    **And how fast it decays is bracketed rather than identified** -- see
    `contraction`, where a seven-figure match on three legs did not survive
    four.
    """

    q_block: float = 900.0
    t_start: float = T_COOLANT_0
    n_legs: int = 4

    def _legs(self) -> dict:
        return make_legs(self.n_legs)

    def upstream_of(self) -> dict[str, str]:
        """Who feeds whom, read from the CONNECTION LIST rather than from memory.

        The ADVEC seams are the answer, and reading them here is what keeps this
        study and the graph from drifting apart about the circuit.
        """
        up = {}
        for c in connections(self.n_legs):
            if c.port_type is PortType.ADVEC:
                up[c.b[0]] = c.a[0]
        return up

    def one_sweep(self, order: tuple[str, ...], passes: int = 1) -> dict:
        """``passes`` Gauss-Seidel sweeps in ``order``, from a cold start."""
        up = self.upstream_of()
        legs = self._legs()
        state = {k: self.t_start for k in legs}
        for _ in range(passes):
            for name in order:
                q = self.q_block if name == "PASS" else 0.0
                state[name] = legs[name].outlet(state[up[name]], q)
        return dict(state)

    def spread(self, passes: int = 1) -> dict:
        """How far apart the orders land, at this many sweeps each."""
        orders = sweep_orders(self.n_legs)
        last = loop_order(self.n_legs)[-1]
        outs = {o: self.one_sweep(o, passes) for o in orders}
        vals = [s[last] for s in outs.values()]
        lo, hi = min(vals), max(vals)
        return {
            "passes": passes,
            "orders": {"-".join(o): s for o, s in outs.items()},
            "min": lo, "max": hi, "range": hi - lo,
            "relative_range": (hi - lo) / max(abs(hi), 1.0e-30),
            "n_distinct": len({round(v, 12) for v in vals}),
        }

    def converged(self, tol: float = 1.0e-13, max_passes: int = 2000) -> dict:
        """Every order iterated to convergence, and where each one lands."""
        orders = sweep_orders(self.n_legs)
        last = loop_order(self.n_legs)[-1]
        out = {}
        for o in orders:
            prev, state, p = None, None, 0
            for p in range(1, max_passes + 1):
                state = self.one_sweep(o, p)
                if prev is not None and abs(state[last] - prev) < tol:
                    break
                prev = state[last]
            out["-".join(o)] = {"T": state[last], "passes": p}
        vals = [v["T"] for v in out.values()]
        return {
            "orders": out,
            "range": max(vals) - min(vals),
            "relative_range": ((max(vals) - min(vals))
                               / max(abs(max(vals)), 1.0e-30)),
        }

    def contraction(self, lo: int = 256, hi: int = 512) -> dict:
        """The measured per-sweep contraction of the spread, and the loop gain.

        Reported as a pair because the interesting thing is their ratio: the
        path dependence decays at a rate the circuit's own dissipation sets
        rather than at one the scheme chose.

        **A sharper claim was measured here and then withdrawn, and the
        withdrawal is the more useful record.**  On the THREE-leg circuit the
        rate matched ``sqrt(loop gain)`` to seven figures -- 0.9618577 against
        0.9618576 -- which reads as an exact identification and was very nearly
        written down as one.  On FOUR legs it does not hold: the rate is
        0.97405 over 256-512 sweeps and 0.97381 over 512-1024, against
        ``gain^(1/3)`` = 0.973926, so it agrees to about 1e-4 and **drifts**,
        and the spread underflows to bitwise zero by 1536 sweeps before any
        asymptote is reached.

        So the seven-figure match was **true as far as it was marched** and is
        not a law -- `positive controls need a horizon`, on a number that was
        going to be quoted.  What survives is the bracket: the rate is strictly
        inside ``(gain, 1)`` and within 1e-3 of ``gain^(1/(n-1))`` at both leg
        counts, which is all the case study's actual claim needs.
        """
        a, b = self.spread(lo)["range"], self.spread(hi)["range"]
        rate = (b / a) ** (1.0 / (hi - lo)) if a > 0 else None
        g = _gain(self.n_legs)
        return {"lo": lo, "hi": hi, "range_lo": a, "range_hi": b,
                "per_sweep": rate, "loop_gain": g,
                "sqrt_loop_gain": math.sqrt(g),
                "per_sweep_over_sqrt_gain": (rate / math.sqrt(g)
                                             if rate else None)}


def _gain(n_legs: int = 4) -> float:
    """The composed contraction of one loop traversal, from the legs alone."""
    g = 1.0
    for name, leg in make_legs(n_legs).items():
        g *= leg.coeffs(0.0)[0]
    return g


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------


def block_capabilities(expert: BlockAgent,
                       motion: MotionClass = MotionClass.STATIC
                       ) -> ExpertCapabilities:
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="wall:THERM", port_type=PortType.THERM,
            geometry=f"wetted face of the block passage, {N_SEAM} line elements",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(THERM_SCALES),
            effective_resolution=M_EFF,
            motion_class=motion,
            # W66 / L3-C9: the response is the entropy flux q_n/T -- THERM's
            # declared FLOW -- so the imposed trace is the temperature.
            response_half=ResponseHalf.FLOW,
            prolongation=face_prolongation(expert.agent_id, "wall:THERM"),
            note="the agent's own Robin surface term, -k dT/dn = h (T - T_c)",
        )],
        # `_robin` is a Robin condition and not an emulation of one.
        bc_channel=BCChannel.ROBIN,
        bc_time_varying=True,
        # Backward Euler solves (M/dt + K + K_robin) T = rhs over the whole
        # block: EMBEDDED in R10's exact sense. It is the SOLE agent of its
        # governing_family here, so W114's narrowing clears it -- nothing has
        # been decomposed, the solve runs over exactly the region a monolith
        # would run it over.
        elliptic_subsolve=EllipticSubsolve.EMBEDDED,
        time_discretization=TimeDiscretization.IMPLICIT,
        stencil_radius=1,                     # Q1 elements, one ring
        substeps_per_macro_step=1,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=L_Z,
        storage=expert.storage,
        equivariances=("translation-z",),
        validity=expert.validity,
        governing_family="heat-conduction-2d",
        lambda_ref="thermostruct2d.ThermoStruct2D, same block and mesh; this "
                   "agent is its own reference and its infidelity is zero by "
                   "construction",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="thermostruct2d/1-q1-backward-euler",
        boundary_response=expert.respond,
        # W74: the effort is an absolute temperature, so the default zero trace
        # is 0 K and the operating point has to be declared.
        probe_base=lambda _port, _e=expert: _e.base_trace(),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/thermostruct2d.py, build repo, imported unmodified; the "
             "conduction half, Robin on both faces",
    )


def _leg_ports(expert: _Leg, upstream: str, downstream: str,
               motion: MotionClass, therm: bool = False) -> list:
    """One ADVEC port per face of the leg, plus the wetted THERM port if any.

    **Two ADVEC ports, not one**, and that is the loop's whole shape: a leg has
    an inlet and an outlet, they meet different agents, and flattening them to
    one port per agent is exactly what `port-algebra-atlas-0.1` section 3.2 says
    a single per-agent list cannot express.
    """
    out = []
    for face, other in (("in", upstream), ("out", downstream)):
        out.append(port_decl(
            name=f"{face}:ADVEC", port_type=PortType.ADVEC,
            geometry=f"{face}let plane of {expert.agent_id}, meeting {other}",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(ADVEC_SCALES),
            # One passenger: a single-species liquid coolant carries enthalpy
            # and nothing else. The list is per FACE and both of this leg's
            # faces carry the same one, which is the easy case -- `rocket.py`
            # has the one that is not.
            passengers=("h0",),
            effective_resolution=M_LUMPED,
            motion_class=motion,
            # ADVEC returns the EFFORT (a specific total enthalpy) against a
            # mass-flux trace, which is `rocket.py`'s convention for every ADVEC
            # port in this package.
            response_half=ResponseHalf.EFFORT,
            prolongation=lumped_prolongation(expert.agent_id, f"{face}:ADVEC"),
            note="single-species liquid coolant: h0 = cp T, one passenger",
        ))
    if therm:
        out.append(port_decl(
            name="wall:THERM", port_type=PortType.THERM,
            geometry=f"wetted face of the passage, {N_SEAM} cells",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(THERM_SCALES),
            effective_resolution=M_EFF,
            motion_class=motion,
            response_half=ResponseHalf.FLOW,
            prolongation=face_prolongation(expert.agent_id, "wall:THERM"),
            note="the film law from the coolant side: q = h (T_wall - T_bulk)",
        ))
    return out


def leg_capabilities(expert: _Leg, upstream: str, downstream: str,
                     motion: MotionClass = MotionClass.STATIC,
                     therm: bool = False) -> ExpertCapabilities:
    responder = (expert.respond_therm if therm and hasattr(expert, "respond_therm")
                 else None)

    def respond(port_name: str, trace):
        if port_name.endswith(":THERM") and responder is not None:
            return responder(port_name, trace)
        return expert.respond_advec(port_name, trace)

    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=_leg_ports(expert, upstream, downstream, motion, therm),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # Two lines of algebra contain no solve. KNOWN, unlike a checkpoint's --
        # `ground_effect.Suspension`'s comment, on the same class of agent.
        elliptic_subsolve=EllipticSubsolve.NONE,
        stencil_radius=0,
        substeps_per_macro_step=1,
        time_discretization=TimeDiscretization.IMPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=L_Z,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        # The SAME family as the transport it closes. A lumped algebraic leg
        # WITHIN a convection problem is not a different continuum problem, and
        # declaring otherwise fails E3 at every seam of the loop --
        # `wind_farm.actuator_disk` has a comment on exactly this.
        governing_family="incompressible-thermal-transport-1d",
        lambda_ref="itself: a closed form has no infidelity to measure",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"loop-leg-{expert.agent_id}-mdot{MDOT:.6g}",
        boundary_response=respond,
        # **W74, and `L4/probe-base` caught this one too.** `probe_base` is one
        # callable per expert and it takes the PORT, which matters here because
        # a leg's two kinds of port linearize about different variables: the
        # wall port's trace is a temperature and the ADVEC ports' is a mass
        # flux. Returning one for both put a 320 K base against a 500 kg/(m^2 s)
        # one on the same seam, and the rule reported the two sides as 56% of
        # the base norm apart -- which they were.
        probe_base=lambda port, _e=expert: _e.base_for(port),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="closed form, zero fitted parameters, the disk.ActuatorDisk pattern",
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def connections(n_legs: int = 4, close_loop: bool = True) -> list[Connection]:
    """One THERM seam at the wall, and one ADVEC seam per leg of the circuit.

    **Built from `LOOP_ORDER` rather than written out**, so the seam list and
    the solve cannot come to disagree about which leg feeds which -- and so the
    three-leg control is the same code path as the circuit it controls.  The
    last seam is the one that closes the ring, and `close_loop=False` drops it.
    """
    order = loop_order(n_legs)
    conns = [Connection(
        seam_id="wall", a=("BLOCK", "wall:THERM"), b=("PASS", "wall:THERM"),
        port_type=PortType.THERM, derive_space=True,
        geometrically_coincident=True,
        # Conjugate heat transfer across a solid-liquid wall. A uniform
        # temperature on this seam drives a nonzero flux -- the film law is not
        # degenerate in any direction of the trace space -- so n_0 = 0.
        expected_null_dim=0,
        # **W138.** Both sides return the heat crossing the wall in the SAME
        # direction (into the coolant), so the well-posed condition is their
        # difference and not their sum, exactly as `wing_fsi`'s wet seam.
        effort_normal="PASS",
        note="conjugate heat transfer: the block's Robin face against the "
             "coolant's film law, the only seam here with a real solver on "
             "one side")]
    n = len(order)
    for i in range(n if close_loop else n - 1):
        a, b = order[i], order[(i + 1) % n]
        closes = (i == n - 1)
        conns.append(Connection(
            seam_id=f"{a.lower()}_{b.lower()}",
            a=(a, "out:ADVEC"), b=(b, "in:ADVEC"),
            port_type=PortType.ADVEC, derive_space=True,
            geometrically_coincident=True,
            # A lumped leg has ONE interface degree of freedom, so dim M = 1
            # and a one-dimensional response has no null direction to spare.
            expected_null_dim=0,
            orientation=f"flow goes {a} -> {b}",
            note=("the RETURN seam, and it is what makes this a circuit: the "
                  "coolant re-enters the passage it left, so the graph has a "
                  "DIRECTED cycle and no agent's inputs are all available "
                  "before the others have run"
                  if closes else
                  f"the coolant leaves {a} and enters {b}")))
    return conns


#: Nothing is measured on this graph yet, and the record says so rather than
#: carrying another graph's numbers.  A constant measured at one state, one
#: scheme and one depth is not the same constant somewhere else --
#: `MeasuredConstants`' own docstring -- and W56's backstop is what turns an
#: absent ledger into `admit-uncertified` rather than a silent pass.
MEASURED_NONE = MeasuredConstants(
    probe_state=f"loop released at T = {T_COOLANT_0} K, mdot = {MDOT} kg/s, "
                f"UA = {UA_RAD} W/K, dt = {MACRO_DT} s",
    scheme="one real conduction agent and three lumped legs, THERM at the wall "
           "and ADVEC round the circuit; the loop closed at every macro-step",
    depth=0,
    source="not measured: CS-13 declares the topology and does not claim a "
           "constant. L, sigma, C_mu and tau are open on this graph")


def build(motion: bool = False,
          experts: dict[str, Any] | None = None,
          measured: MeasuredConstants | None = "default",
          close_loop: bool = True,
          n_legs: int = 4,
          ) -> tuple[CaseGraph, dict[str, Any]]:
    """The circuit as a graph.

    Two controls, both parameters of `build` rather than separate modules, so
    that each runs the same code path as the graph it is a control for:

    ``close_loop=False`` drops the return seam and the same agents form a
    **chain**.  Everything else is identical, so anything the compile says
    about one and not the other is a statement about the CYCLE and about
    nothing else.

    ``n_legs=3`` is the **triangle**: three legs pairwise joined, which
    `CaseGraph.detected_cross_points` reads as a substructuring cross-point and
    `L2/I2/G1` refuses.  Three legs share no geometric point -- their three
    seams are three distinct planes -- so that refusal is the adjacency proxy
    over-firing on a circuit, and keeping the build reachable is what makes the
    finding reproducible rather than a paragraph.
    """
    mc = MotionClass.SOLUTION_DEPENDENT if motion else MotionClass.STATIC
    if measured == "default":
        measured = MEASURED_NONE
    order = loop_order(n_legs)
    if experts is None:
        experts = dict(make_legs(n_legs))
        experts["BLOCK"] = BlockAgent()
    agents = [Agent("BLOCK", block_capabilities(experts["BLOCK"], mc),
                    domain="Omega_block: the whole cooled section",
                    role="solid")]
    n = len(order)
    for i, name in enumerate(order):
        up = order[(i - 1) % n]
        down = order[(i + 1) % n]
        agents.append(Agent(
            name,
            leg_capabilities(experts[name], up, down, mc,
                             therm=(name == "PASS")),
            domain=_LEG_DOMAIN[name], role="coolant"))
    return (
        CaseGraph(
            name=("cooling-loop" if close_loop else "cooling-loop-open-control")
                 + ("" if n_legs == 4 else f"-{n_legs}leg"),
            agents=agents,
            connections=connections(n_legs, close_loop),
            # Four agents, four disjoint regions, no region cut and no partition
            # of unity. `NON_OVERLAPPING` is what `thermal_seam` and
            # `brake_thermal` declare for exactly this shape -- there is no third
            # member of the enum and adding one would be a schema change this
            # case study is not entitled to make. It is also what keeps L2/C2 and
            # the halo rule off a graph they have no subject on.
            decomposition=Decomposition.NON_OVERLAPPING,
            macro_dt=MACRO_DT,
            measured=measured,
            note=("CS-13: a closed coolant circuit. The first graph in this "
                  "package whose ports form a DIRECTED cycle -- the coolant "
                  "returns to the passage it left, so no agent's inputs are "
                  "all available before the others have run")),
        experts,
    )


#: What each leg is, for the record's `domain` field.
_LEG_DOMAIN = {
    "PASS": "the coolant passage through the block",
    "HOT": "the hot line, block to radiator",
    "RAD": "the radiator core",
    "COLD": "the cold line and the pump, radiator back to the block",
}


def compile_report(close_loop: bool = True, n_legs: int = 4) -> str:
    """Compile the circuit and return the decision record, for a quick look."""
    from ..compiler import compile_scheme
    g, _e = build(close_loop=close_loop, n_legs=n_legs)
    return compile_scheme(g).report()
