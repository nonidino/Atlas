"""The electric family, run from a case file: style D, a field joined to lumped parts.

A thin resistive plate, ``div(sigma grad phi) = 0`` on the case's grid with a
material per region (`fv.py`), whose **electrodes** -- boundary segments of kind
``electrode`` -- are wired to a lumped circuit of batteries and resistors (the
case's ``attachments``).  The plate is 2-D, so its currents come per metre of
depth; the ``thickness`` parameter turns them into the circuit's amperes.

**Style D, as the showcase plan states it: a field's boundary integral is a
lumped part's port variable.**  The port is an electrode.  Its effort is the
electrode's potential and its flow is the current through it, which is the
integral of ``sigma dphi/dn`` over the electrode's faces:

* the plate is given the electrode potentials and returns the currents
  (a sparse LU of the plate, factorized once);
* the circuit is given those currents and returns the potentials its nodes
  settle at (modified nodal analysis: resistors, batteries with an internal
  resistance as Norton sources, ideal batteries as voltage-source rows);
* the potentials are relaxed toward the circuit's (`styles.field_lumped`): the
  case's first factor, then Aitken's rule if the case asks for it.  The page
  shows the factor every iteration.  For one port the unrelaxed iteration
  contracts only if ``G (r + R) < 1`` -- the plate's conductance times the
  circuit's resistance -- which is why the relaxation is stated, not hidden.

**The full-domain reference** is the plate and the circuit in ONE sparse
system: every cell's potential, every circuit node's, and every ideal source's
current, solved together.  A converged style-D answer is that system's answer to
the iteration's tolerance.

The checks, registered before any run: Kirchhoff (the currents at every
circuit node sum to zero, the plate's electrode currents included), the energy
balance (what the batteries deliver is what the resistors, the batteries'
internal resistance and the plate dissipate), and agreement with the full
system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .. import fv, styles
from ..checks import CheckSpec, judge
from .conduction import field_from_case, window_cells

FAMILY = "electric-2d"
STYLE = "D"
ARMS = ("serial", "full")
FIELD_LABEL = "potential (V)"
LENGTH_UNIT = "m"
SERIES = {"current": "Circuit current per repeat | A"}

REGISTERED = "2026-09-28, before the electric family's first run"

CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        key="kirchhoff", title="Kirchhoff: the currents at every node sum to zero",
        kind="balance", tolerance=1e-6, registered=REGISTERED,
        measure=("max over circuit nodes of |sum of the currents leaving the node|, the "
                 "plate's electrode currents included, over the largest branch current"),
        why=("the full system solves Kirchhoff's law at every node exactly, so it closes "
             "to round-off; style D closes to its iteration's floor, the node potentials "
             "moving by less than 1e-10 of the largest EMF between the last two "
             "iterations, which moves a branch current by that over the branch's "
             "resistance -- far below 1e-6 of the current for the showcase circuit")),
    CheckSpec(
        key="energy", title="Energy: power delivered = power dissipated",
        kind="balance", tolerance=1e-6, registered=REGISTERED,
        measure=("|sum of EMF x current over the batteries - (I^2 R over the resistors + "
                 "I^2 r inside the batteries + the plate's Joule heat)| over the power "
                 "delivered"),
        why=("Tellegen's theorem makes it an identity of any solution that satisfies "
             "Kirchhoff's laws and the branch laws, so it closes where Kirchhoff does; "
             "the plate's heat is summed from the same faces its currents are")),
    CheckSpec(
        key="reference", title="Current agrees with the plate and circuit solved as one",
        kind="reference", tolerance=1e-6, registered=REGISTERED,
        measure="max over branches |I_D - I_full| over the largest branch current",
        why=("both solve the same finite volumes and the same nodal equations; the "
             "style-D answer converges to the joint solve's")),
)


def step_label(spec) -> str:
    return "timed repeat"


def available_arms(spec) -> tuple[tuple[str, ...], dict[str, str]]:
    return ARMS, {"parallel": "style D joins one field to one circuit: the field solve "
                              "and the circuit solve take turns, so there is nothing to "
                              "run side by side"}


_KIND = {"electrode": fv.FIXED, "fixed-potential": fv.FIXED, "no-current": fv.NO_FLUX}


@dataclass
class Circuit:
    """The case's attachments as a netlist, and its nodal equations."""

    nodes: list[str]                     # every node but the ground, in order
    ground: str
    parts: list[Any]                     # spec.attachments
    electrodes: list[str]

    @classmethod
    def from_case(cls, spec) -> "Circuit":
        electrodes = [b.id for b in spec.boundaries if b.kind == "electrode"]
        names: list[str] = []
        for n in electrodes + [x for a in spec.attachments for x in (a.a, a.b)]:
            if n not in names:
                names.append(n)
        ground = "ground" if "ground" in names else electrodes[0]
        return cls([n for n in names if n != ground], ground, list(spec.attachments),
                   electrodes)

    def index(self, n: str) -> int | None:
        return None if n == self.ground else self.nodes.index(n)

    def ideal(self) -> list[Any]:
        return [p for p in self.parts if p.kind == "battery" and p.internal == 0.0]

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """``(M, rhs)`` of modified nodal analysis: node potentials, then one current
        per ideal battery (flowing from its + terminal into the source)."""
        n, ideal = len(self.nodes), self.ideal()
        M = np.zeros((n + len(ideal), n + len(ideal)))
        rhs = np.zeros(n + len(ideal))

        def stamp(a, b, g):
            ia, ib = self.index(a), self.index(b)
            for i, s in ((ia, 1.0), (ib, -1.0)):
                if i is None:
                    continue
                for j, t in ((ia, 1.0), (ib, -1.0)):
                    if j is not None:
                        M[i, j] += s * t * g
        for p in self.parts:
            if p.kind == "resistor":
                stamp(p.a, p.b, 1.0 / p.value)
            elif p.internal > 0.0:
                g = 1.0 / p.internal
                stamp(p.a, p.b, g)                         # Norton: g in parallel ...
                ia, ib = self.index(p.a), self.index(p.b)
                if ia is not None:
                    rhs[ia] += p.value * g                 # ... with E g driven into +
                if ib is not None:
                    rhs[ib] -= p.value * g
        for k, p in enumerate(ideal):
            row = n + k
            ia, ib = self.index(p.a), self.index(p.b)
            if ia is not None:
                M[ia, row] += 1.0
                M[row, ia] += 1.0
            if ib is not None:
                M[ib, row] -= 1.0
                M[row, ib] -= 1.0
            rhs[row] = p.value
        return M, rhs

    def potentials(self, x: np.ndarray) -> dict[str, float]:
        out = {self.ground: 0.0}
        out.update({n: float(x[i]) for i, n in enumerate(self.nodes)})
        return out

    def branch_currents(self, pot: dict[str, float], x: np.ndarray) -> dict[str, float]:
        """Each part's current, positive from its ``a`` end to its ``b`` end
        THROUGH the part (for a battery: through the source, so a discharging
        battery carries a negative current from + to - inside it)."""
        out = {}
        ideal = self.ideal()
        for p in self.parts:
            va, vb = pot[p.a], pot[p.b]
            if p.kind == "resistor":
                out[p.id] = (va - vb) / p.value
            elif p.internal > 0.0:
                out[p.id] = (va - vb - p.value) / p.internal
            else:
                out[p.id] = float(x[len(self.nodes) + ideal.index(p)])
        return out


@dataclass
class ElecState:
    phi: np.ndarray
    potentials: dict[str, float] = field(default_factory=dict)
    currents: dict[str, float] = field(default_factory=dict)     # per part, A
    electrode_currents: dict[str, float] = field(default_factory=dict)   # into the plate
    iterations: int = 0
    converged: bool = True
    history: list[float] = field(default_factory=list)
    relaxation: list[float] = field(default_factory=list)


class ElectricRun:
    family = FAMILY

    def __init__(self, spec, arms=ARMS, threads: int = 1):
        self.spec = spec
        self.arms = tuple(a for a in ARMS if a in arms)
        self.t = float(spec.physics.get("thickness"))
        self.f = field_from_case(spec, prop_k="sigma", kinds=_KIND)
        self.nx, self.ny, self.dx = self.f.nx, self.f.ny, self.f.dx
        self.circuit = Circuit.from_case(spec)
        self.emf = max([p.value for p in spec.attachments if p.kind == "battery"] + [1.0])
        self.tol = float(spec.coupling.tolerance)
        self.max_it = int(spec.coupling.max_iterations)
        # the electrode faces: which boundary cells belong to which electrode
        self.faces = electrode_faces(spec, self.f)
        e_ids = self.circuit.electrodes
        self.e_ids = e_ids
        # the plate alone (electrodes as fixed potentials whose values change)
        self.plate = fv.assemble(self.f)
        self.plate_lu = styles.Factor(self.plate.A)
        # the circuit alone
        self.M, self.rhs = self.circuit.matrix()
        self.M_lu = np.linalg.inv(self.M) if self.M.size else None
        if "full" in self.arms:
            self.joint, self.joint_rhs = joint_system(self)
            self.joint_lu = styles.Factor(self.joint)

    # -- the two solves style D alternates ----------------------------------

    def _plate_rhs(self, pot: dict[str, float]) -> np.ndarray:
        """The plate's right-hand side with the electrodes at ``pot``: the fixed-
        potential segments keep their values, the electrode faces take their node's."""
        b = self.plate.b.copy()
        for e in self.e_ids:
            cells, g = self.faces[e]
            np.add.at(b, cells, g * pot[e])
        return b

    def plate_solve(self, x: np.ndarray):
        """Electrode potentials (``x``, in the order of ``e_ids``) -> the current INTO
        the plate through each electrode, in amperes, and the potential field."""
        pot = {e: float(v) for e, v in zip(self.e_ids, x)}
        phi = self.plate_lu.solve(self._plate_rhs(pot))
        cur = np.array([float(np.sum(self.faces[e][1] * (pot[e] - phi[self.faces[e][0]])))
                        * self.t for e in self.e_ids])
        return cur, phi

    def circuit_solve(self, into_plate: np.ndarray) -> np.ndarray:
        """Currents drawn into the plate at each electrode -> the electrodes'
        potentials, from the circuit's nodal equations with those as sinks."""
        rhs = self.rhs.copy()
        for e, i in zip(self.e_ids, into_plate):
            k = self.circuit.index(e)
            if k is not None:
                rhs[k] -= float(i)
        x = self.M_lu @ rhs
        pot = self.circuit.potentials(x)
        self._last_x = x
        return np.array([pot[e] for e in self.e_ids])

    # -- the arms -----------------------------------------------------------------

    def initial(self, arm: str) -> ElecState:
        return ElecState(np.zeros(self.f.n))

    def step(self, arm: str, s: ElecState) -> ElecState:
        """The whole solve (steady): style D from zero potentials, or the joint system."""
        if arm == "full":
            y = self.joint_lu.solve(self.joint_rhs)
            n = self.f.n
            phi = y[:n]
            x = y[n:]
            pot = self.circuit.potentials(x)
            return self._state(phi, pot, x, 1, True, [], [])
        x0 = np.zeros(len(self.e_ids))
        it = styles.field_lumped(self.plate_solve, self.circuit_solve, x0, self.tol,
                                 self.max_it, self.emf,
                                 theta0=self.spec.coupling.relaxation,
                                 aitken=self.spec.coupling.aitken)
        # the answer the iteration stopped at: the plate solved at the last efforts,
        # the circuit at the currents they drew
        cur, phi = self.plate_solve(it.extra["efforts"])
        self.circuit_solve(cur)
        x = self._last_x
        pot = self.circuit.potentials(x)
        # the electrodes' potentials are the ones the plate was solved with
        for e, v in zip(self.e_ids, it.extra["efforts"]):
            pot[e] = float(v)
        return self._state(phi, pot, x, it.iterations, it.converged, it.history,
                           it.relaxation)

    def _state(self, phi, pot, x, iters, conv, hist, relax) -> ElecState:
        cur = self.circuit.branch_currents(pot, x)
        e_cur = {e: float(np.sum(self.faces[e][1] * (pot[e] - phi[self.faces[e][0]])))
                 * self.t for e in self.e_ids}
        return ElecState(phi, pot, cur, e_cur, iters, conv, list(hist), list(relax))

    # -- instruments ---------------------------------------------------------------

    def kirchhoff(self, s: ElecState) -> float:
        """max over nodes |sum of currents leaving| / the largest branch current."""
        leave = {n: 0.0 for n in [self.circuit.ground] + self.circuit.nodes}
        for p in self.spec.attachments:
            i = s.currents[p.id]
            leave[p.a] += i
            leave[p.b] -= i
        for e, i in s.electrode_currents.items():
            leave[e] += i                                  # into the plate leaves the node
        scale = max([abs(v) for v in s.currents.values()] + [1e-300])
        return max(abs(v) for v in leave.values()) / scale

    def powers(self, s: ElecState) -> dict[str, float]:
        delivered = 0.0
        lumped = 0.0
        for p in self.spec.attachments:
            i = s.currents[p.id]
            if p.kind == "battery":
                delivered += -i * p.value                  # current out of + through the load
                lumped += i * i * p.internal
            else:
                lumped += i * i * p.value
        plate = plate_heat(self, s.phi, s.potentials)
        return {"delivered": delivered, "lumped": lumped, "plate": plate,
                "balance": abs(delivered - lumped - plate) / max(abs(delivered), 1e-300)}

    def observe(self, arm: str, s: ElecState, prev: ElecState | None = None) -> dict:
        pw = self.powers(s)
        batt = [p for p in self.spec.attachments if p.kind == "battery"]
        current = abs(s.currents[batt[0].id]) if batt else 0.0
        return {"current": current, "plate_heat_W": pw["plate"],
                "power_delivered_W": pw["delivered"], "energy": pw["balance"],
                "kirchhoff": self.kirchhoff(s), "iterations": float(s.iterations),
                "converged": float(s.converged), "convergence": list(s.history),
                "relaxation": list(s.relaxation)}

    def bitwise_equal(self, a: ElecState, b: ElecState) -> bool:
        return bool(np.array_equal(a.phi, b.phi))

    def field(self, s: ElecState) -> np.ndarray:
        return s.phi.reshape(self.ny, self.nx)

    def close(self) -> None:
        pass

    # -- the end of a run -----------------------------------------------------------

    def compare(self, states, history, bitwise):
        metrics: dict[str, Any] = {}
        for arm, rows in history.items():
            if not rows:
                continue
            r = rows[-1]
            s = states[arm]
            metrics[arm] = {"current_A": r["current"], "plate_heat_W": r["plate_heat_W"],
                            "power_delivered_W": r["power_delivered_W"],
                            "kirchhoff_max": max(x["kirchhoff"] for x in rows),
                            "energy_max": max(x["energy"] for x in rows),
                            "iterations_last": r["iterations"],
                            "all_converged": all(x["converged"] for x in rows),
                            "electrode_potentials_V": {e: s.potentials[e] for e in self.e_ids}}
            if r["current"] > 0:
                drop = abs(s.potentials[self.e_ids[0]] - s.potentials[self.e_ids[-1]])
                metrics[arm]["plate_resistance_ohm"] = drop / r["current"] if len(
                    self.e_ids) > 1 else None
        checks = [judge(CHECKS[0], max((m["kirchhoff_max"] for m in metrics.values()),
                                       default=None),
                        "; ".join(f"{a}: {m['kirchhoff_max']:.3g}" for a, m in metrics.items())),
                  judge(CHECKS[1], max((m["energy_max"] for m in metrics.values()),
                                       default=None),
                        "; ".join(f"{a}: {m['energy_max']:.3g}" for a, m in metrics.items()))]
        if "serial" in states and "full" in states:
            a, b = states["serial"].currents, states["full"].currents
            scale = max(abs(v) for v in b.values())
            dev = max(abs(a[k] - b[k]) for k in b) / scale
            metrics["serial"]["current_vs_full"] = ((metrics["serial"]["current_A"]
                                                     - metrics["full"]["current_A"])
                                                    / metrics["full"]["current_A"])
            checks.append(judge(CHECKS[2], dev, f"largest branch current {scale:.6g} A"))
        else:
            checks.append(judge(CHECKS[2], None, "needs style D and the full system"))
        return metrics, checks

    def notes(self, done: int) -> list[str]:
        g = None
        cur, _phi = self.plate_solve(np.array([1.0 if i == 0 else 0.0
                                               for i in range(len(self.e_ids))]))
        if len(self.e_ids) == 2:
            g = abs(cur[0])
        out = [f"The circuit's ground is {self.circuit.ground!r}. First relaxation factor "
               f"{self.spec.coupling.relaxation:g}, then "
               + ("Aitken's rule." if self.spec.coupling.aitken else "held fixed.")]
        if g is not None:
            loop = sum(p.value if p.kind == "resistor" else p.internal
                       for p in self.spec.attachments)
            out.append(f"The plate's conductance between its two electrodes is {g:.4g} S, "
                       f"and the circuit's series resistance {loop:.4g} ohm, so the "
                       f"unrelaxed iteration's factor G (r + R) is {g * loop:.3g}"
                       + (" (under 1: it would contract)." if g * loop < 1 else
                          " (over 1: unrelaxed it would diverge; the relaxation carries "
                          "it)."))
        out.append("Steady: every 'step' is the whole solve again from zero potentials, "
                   "repeated to time it.")
        return out

    def describe(self) -> dict[str, Any]:
        return {"style": "D", "cells": self.f.n, "thickness_m": self.t,
                "electrodes": {e: int(self.faces[e][0].size) for e in self.e_ids},
                "circuit_nodes": [self.circuit.ground] + self.circuit.nodes,
                "ground": self.circuit.ground, "tolerance": self.tol,
                "relaxation0": self.spec.coupling.relaxation,
                "aitken": self.spec.coupling.aitken}


# ---------------------------------------------------------------------------
# the pieces the two arms share
# ---------------------------------------------------------------------------


def electrode_faces(spec, f: fv.Field) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """electrode id -> (the boundary cells its faces belong to, each face's half-cell
    conductance ``2 sigma`` per unit depth)."""
    out: dict[str, tuple[list, list]] = {}
    nx, ny = f.nx, f.ny
    k = f.k.ravel()
    for b in spec.boundaries:
        if b.kind != "electrode":
            continue
        n = ny if b.edge in ("left", "right") else nx
        along = np.arange(b.start, n if b.stop is None else b.stop)
        cells = {"left": along * nx, "right": along * nx + nx - 1, "bottom": along,
                 "top": (ny - 1) * nx + along}[b.edge]
        c, g = out.setdefault(b.id, ([], []))
        c.append(cells)
        g.append(2.0 * k[cells])
    return {e: (np.concatenate(c), np.concatenate(g)) for e, (c, g) in out.items()}


def joint_system(run: ElectricRun):
    """The plate and the circuit as one sparse system, in amperes.

    Unknowns: every cell's potential, then the circuit's (non-ground) node
    potentials and ideal-source currents.  The plate's rows are its finite
    volumes times the thickness, with each electrode face coupling its cell to
    its node; each electrode node's row adds the current it sends into the
    plate.  Built with the ground's potential fixed at zero.
    """
    n = run.f.n
    t = run.t
    A = run.plate.A * t
    b = run.plate.b * t
    m = run.M.shape[0]
    rows, cols, vals = [], [], []
    rhs_c = run.rhs.copy()
    for e in run.e_ids:
        cells, g = run.faces[e]
        k = run.circuit.index(e)
        if k is None:                   # a grounded electrode: its faces are fixed at 0 V
            continue
        rows += [cells, np.full(cells.size, n + k)]
        cols += [np.full(cells.size, n + k), cells]
        vals += [-g * t, -g * t]
        rows.append(np.array([n + k]))
        cols.append(np.array([n + k]))
        vals.append(np.array([float(np.sum(g)) * t]))
    J = sp.bmat([[A, None], [None, sp.csr_matrix(run.M)]], format="csr")
    if rows:
        J = J + sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                      np.concatenate(cols))),
                              shape=(n + m, n + m))
    return J.tocsr(), np.concatenate([b, rhs_c])


def plate_heat(run: ElectricRun, phi: np.ndarray, pot: dict[str, float]) -> float:
    """Joule heat in the plate, in watts: every interior face's ``G dphi^2`` and every
    electrode face's half-cell, times the thickness -- the same faces the currents
    are summed from."""
    f = run.f
    u = phi.reshape(f.ny, f.nx)
    gx = fv.harmonic(f.k[:, :-1], f.k[:, 1:])
    gy = fv.harmonic(f.k[:-1, :], f.k[1:, :])
    heat = float(np.sum(gx * (u[:, :-1] - u[:, 1:]) ** 2) + np.sum(gy * (u[:-1, :]
                                                                         - u[1:, :]) ** 2))
    for e in run.e_ids:
        cells, g = run.faces[e]
        heat += float(np.sum(g * (pot[e] - phi[cells]) ** 2))
    return heat * run.t


def circuit_dtn(spec, held: dict[str, float]) -> dict[str, float]:
    """The circuit alone with every electrode held at a potential: the current INTO
    the circuit at each electrode (A), from the plate's side.

    Modified nodal analysis over the free nodes and the ideal batteries' currents,
    the electrodes known: no ground is needed, because the held electrodes fix the
    level.  Positive into the circuit, so a passive network's response rises with
    the potential it is held at (the Steklov-Poincare convention the compiler's
    probe reads)."""
    parts = list(spec.attachments)
    free = []
    for p in parts:
        for n in (p.a, p.b):
            if n not in held and n not in free:
                free.append(n)
    ideal = [p for p in parts if p.kind == "battery" and p.internal == 0.0]
    nf, ni = len(free), len(ideal)
    M = np.zeros((nf + ni, nf + ni))
    r = np.zeros(nf + ni)

    def conductance(a, b, g, source=0.0):
        """g between a and b, and a current ``source`` pushed from b to a through
        it (a Norton battery's E g)."""
        for x, s in ((a, 1.0), (b, -1.0)):
            if x in held:
                continue
            i = free.index(x)
            for y, t in ((a, 1.0), (b, -1.0)):
                if y in held:
                    r[i] -= s * t * g * held[y]
                else:
                    M[i, free.index(y)] += s * t * g
            r[i] += s * source
    for p in parts:
        if p.kind == "resistor":
            conductance(p.a, p.b, 1.0 / p.value)
        elif p.internal > 0.0:
            conductance(p.a, p.b, 1.0 / p.internal, p.value / p.internal)
    for k, p in enumerate(ideal):
        row = nf + k
        for x, s in ((p.a, 1.0), (p.b, -1.0)):
            if x in held:
                r[row] -= s * held[x]
            else:
                M[free.index(x), row] += s
                M[row, free.index(x)] += s
        r[row] += p.value
    sol = np.linalg.solve(M, r) if M.size else np.zeros(0)
    pot = dict(held)
    pot.update({n: float(sol[i]) for i, n in enumerate(free)})
    into = {e: 0.0 for e in held}
    for p in parts:
        if p.kind == "resistor":
            i_ab = (pot[p.a] - pot[p.b]) / p.value
        elif p.internal > 0.0:
            i_ab = (pot[p.a] - pot[p.b] - p.value) / p.internal
        else:
            i_ab = float(sol[nf + ideal.index(p)])
        # a current i_ab leaves a into the part and arrives at b
        if p.a in into:
            into[p.a] += i_ab
        if p.b in into:
            into[p.b] -= i_ab
    return into


def case_graph(spec):
    """The case for the compiler (`compile.py`): the plate and the circuit as two
    agents meeting at every electrode, an ELEC seam each, lumped (one value per
    electrode: its potential and the current through it).

    The plate solves its whole field directly (an embedded elliptic solve, and it
    is the only agent of its physics); the circuit is algebra (no stencil, no
    solve over a field).  Each responds to an electrode's potential with the
    current into itself there, the others held at the probe base, per unit of
    the electrode's area."""
    from atlas.capability import (BCChannel, ClaimType, Direction, EllipticSubsolve,
                                  ExpertCapabilities, MotionClass, TimeDiscretization,
                                  port_decl)
    from atlas.graph import Agent, CaseGraph, Connection, Decomposition
    from atlas.ports import PortType, ResponseHalf

    from ..compile import face_prolongation
    run = ElectricRun(spec, arms=("serial",))
    d = spec.domain
    area = {b.id: ((d.ny if b.edge in ("left", "right") else d.nx)
                   if b.stop is None else b.stop) - b.start
            for b in spec.boundaries if b.kind == "electrode"}
    area = {e: n * d.dx * run.t for e, n in area.items()}          # m^2
    length = {e: a / run.t for e, a in area.items()}
    base = {e: 0.0 for e in run.e_ids}
    r_loop = sum(p.value if p.kind == "resistor" else p.internal for p in spec.attachments)
    i_scale = run.emf / max(r_loop, 1e-9)
    j_scale = i_scale / max(min(area.values()), 1e-30)
    scales = {"potential": run.emf, "current_density": j_scale,
              "power_area": run.emf * j_scale}

    def plate_respond(port, trace):
        e = port.split(":")[0]
        x = np.array([float(np.ravel(trace)[0]) if k == e else base[k] for k in run.e_ids])
        cur, _phi = run.plate_solve(x)
        return np.array([cur[run.e_ids.index(e)] / area[e]])

    def circuit_respond(port, trace):
        e = port.split(":")[0]
        held = {k: (float(np.ravel(trace)[0]) if k == e else base[k]) for k in run.e_ids}
        return np.array([circuit_dtn(spec, held)[e] / area[e]])

    def ports(agent):
        return [port_decl(
            name=f"{e}:ELEC", port_type=PortType.ELEC,
            geometry=f"electrode {e}, {length[e]:.4g} m of edge, lumped",
            direction=Direction.BIDIRECTIONAL, nondim=dict(scales), effective_resolution=1,
            motion_class=MotionClass.STATIC, response_half=ResponseHalf.FLOW,
            prolongation=face_prolongation(agent, f"{e}:ELEC", 1, length[e]),
            note="the current density into this side through the electrode")
            for e in run.e_ids]

    def valid(state=None, cond=None):
        return True

    common = dict(bc_channel=BCChannel.DIRICHLET, bc_time_varying=True,
                  time_discretization=TimeDiscretization.IMPLICIT,
                  substeps_per_macro_step=1, dt_native=1.0, validity=valid,
                  claim_types=frozenset({ClaimType.TRAJECTORY}),
                  reproducibility_floor=float(np.finfo(float).eps), deterministic=True)
    plate = ExpertCapabilities(
        expert_id="plate", ports=ports("plate"),
        elliptic_subsolve=EllipticSubsolve.EMBEDDED, stencil_radius=1,
        governing_family="electric-conduction-2d",
        lambda_ref="the plate and the circuit as one sparse system, the full-domain arm",
        weight_hash="workbench-fv/electric-plate", boundary_response=plate_respond,
        probe_base=lambda port: np.zeros(1),
        note="atlas/workbench/fv.py: div(sigma grad phi) = 0 on the plate", **common)
    circuit = ExpertCapabilities(
        expert_id="circuit", ports=ports("circuit"),
        elliptic_subsolve=EllipticSubsolve.NONE, stencil_radius=0,
        governing_family="lumped-circuit",
        lambda_ref="itself: nodal analysis is exact for its parts",
        weight_hash="workbench/electric-circuit", boundary_response=circuit_respond,
        probe_base=lambda port: np.zeros(1),
        note="modified nodal analysis of the case's batteries and resistors", **common)
    conns = [Connection(
        seam_id=e, a=("plate", f"{e}:ELEC"), b=("circuit", f"{e}:ELEC"),
        port_type=PortType.ELEC, derive_space=True, geometrically_coincident=True,
        expected_null_dim=0, cut_axis=Decomposition.NON_OVERLAPPING,
        note="field to lumped: the electrode's potential and the current through it")
        for e in run.e_ids]
    return CaseGraph(
        name=f"workbench-{spec.name}",
        agents=[Agent("plate", plate, domain="the plate"),
                Agent("circuit", circuit, domain="the lumped circuit", role="circuit")],
        connections=conns, decomposition=Decomposition.NON_OVERLAPPING,
        # W162: a circuit has no geometric cross-points, and the adjacency proxy is
        # wrong on one (cooling_loop's first refusal); declared none
        cross_points=(), macro_dt=1.0,
        note=f"the workbench case {spec.name!r}: a field joined to lumped parts (style D)")


def build(spec, arms=ARMS, threads: int = 1) -> ElectricRun:
    return ElectricRun(spec, arms=arms, threads=threads)


__all__ = ["FAMILY", "STYLE", "ARMS", "CHECKS", "ElectricRun", "ElecState", "Circuit",
           "build", "available_arms", "step_label", "electrode_faces", "joint_system",
           "plate_heat"]
