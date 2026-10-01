"""The thermoelastic family, run from a case file: a split by physics, not space.

A heated plate that expands (showcase case 7): conduction, backward Euler, and
quasi-static plane-stress elasticity with thermal strain, on the same Q1 mesh
with a material per region (`fe.py`, the build repo's `ThermoStruct2D` element).
The plate is a free body: its rigid-body motions are removed by a bordered
system, `ThermoStruct2D._solve_free`'s construction, because pinning it would
manufacture stress in a plate that should have none.  Temperatures are set on
the edges (fixed, insulated or a heat flux); the strain-free temperature is
the initial one.

**Two agents on one mesh.**  The conduction agent owns the temperature and
steps it; the elasticity agent owns the displacement and solves for it from a
temperature it is handed.  The bond between them is the whole temperature
field: the thermal strain is a volume term, which the thermal-strain case study
([[case-study-thermal-strain-atlas-0.1]]) found is a bond and not a port.  Three
arms:

``serial``    the **synchronous split**: conduction steps, hands its new
              temperature over, elasticity solves from it.  The same arithmetic
              as the unsplit solver, so the two agree to the bit: the control
              that the bond carries the whole coupling.
``parallel``  the **lagged split**: both agents run at once on two threads,
              elasticity from the temperature of the step before.  Its
              displacement is the synchronous one a step late, exactly (the
              second control), so its error is the displacement's change over
              one step: first order in the step.
``full``      the **unsplit solver**: one object that steps the temperature and
              then solves the displacement, as `ThermoStruct2D` does.

The page draws the von Mises stress, ``D (eps(u) - alpha dT (1, 1, 0))`` at each
element's centre, from the temperature the displacement was solved from; the
difference panel is the lagged split against the unsplit solver.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from .. import fe, styles
from ..checks import CheckSpec, exact, judge
from ..fast import FastBars
from .. import geometry as geo
from ..fe import rigid_modes_on
from .elasticity import active_elements, element_props, inactive_nodes

FAMILY = "thermoelastic-2d"
STYLE = "split"
ARMS = ("serial", "parallel", "full")
#: **The Fast example's bars** (demo item 1.4), one per mechanism this family may
#: use, in the order they are tried (`fast.py`; demo-fast-examples-plan section 3).
#: Registered 2026-09-30, before the first timed run, and never loosened.
FAST: tuple[FastBars, ...] = (
    FastBars("X", 3.0, None, "the family's own controls: the lagged split is the synchronous one a step late, bit for bit", ceiling="a split by physics can at most halve the time: s <= 2 with two physics"),
)
ARM_LABELS = {"serial": "Split, synchronous", "parallel": "Split, lagged (two threads)",
              "full": "Unsplit solver"}
#: the page draws the lagged split (the arm that differs) against the unsplit solver,
#: or the synchronous one when the lagged arm is not ticked
PANEL_NAMES = {"parallel": "Lagged split", "serial": "Synchronous split", "full": "Unsplit",
               "difference_parallel": "Lagged minus unsplit",
               "difference_serial": "Synchronous minus unsplit"}
FIELD_LABEL = "von Mises stress (MPa)"
LENGTH_UNIT = "m"
SERIES = {"max_stress": "Largest von Mises stress | MPa"}

REGISTERED = "2026-09-29, before the thermoelastic family's first run"

CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        key="energy", title="Heat: in through the edges = stored", kind="balance",
        tolerance=1e-9, registered=REGISTERED,
        measure=("|heat stored in the step - dt (flux through the edges + the heat the "
                 "fixed-temperature edges supply)| over the larger of the two, every step, "
                 "worst arm"),
        why=("the conduction matrix annihilates a uniform temperature, so the heat the "
             "fixed edges supply is whatever the step stores beyond the given flux, and "
             "the balance is the free nodes' residual of one sparse direct solve: "
             "round-off")),
    CheckSpec(
        key="reference", title="The synchronous split is the unsplit solver, bit for bit",
        kind="control", tolerance=None, registered=REGISTERED,
        measure="np.array_equal on the temperature and the displacement after every step",
        why=("the split hands the whole temperature field across, and each agent does the "
             "arithmetic the unsplit solver does, in the same order")),
    CheckSpec(
        key="lag", title="The lagged split is the synchronous one a step late, bit for bit",
        kind="control", tolerance=None, registered=REGISTERED,
        measure=("np.array_equal between the lagged split's displacement after step k and "
                 "the synchronous split's after step k - 1, every step"),
        why=("the lagged elasticity agent solves from the temperature of the step before, "
             "which is the temperature the synchronous agent solved from then; the "
             "difference between the two splits is therefore exactly one step of change")),
)

_KIND_T = ("fixed-temperature", "insulated", "heat-flux")


@dataclass
class ThermoState:
    T: np.ndarray            # nodal temperature
    u: np.ndarray            # nodal displacement [2 n_nodes]
    T_u: np.ndarray          # the temperature u was solved from


class ThermoelasticRun:
    family = FAMILY

    def __init__(self, spec, arms=ARMS, threads: int = 2):
        self.spec = spec
        d = spec.domain
        self.g = g = fe.QuadGrid(d.nx, d.ny, float(d.dx))
        self.E, self.nu, self.alpha, k, rho, cp = element_props(
            spec, "E", "nu", "alpha", "k", "rho", "cp")
        self.T0 = float(spec.physics.get("T0"))
        self.dt = float(spec.run.macro_dt)
        #: a drawn domain (case file 0.4): its cells are the elements; the nodes no
        #: element touches leave both solves (held at T0, and at rest)
        self.active = active_elements(spec)
        if self.active is not None:
            self.E = np.where(self.active, self.E, 0.0)
            k = np.where(self.active, k, 0.0)
            rho = np.where(self.active, rho, 0.0)
        self.off = inactive_nodes(spec)
        # -- conduction: backward Euler with the fixed-temperature nodes eliminated
        Kth, Mth = fe.assemble_thermal(g, k, rho * cp)
        self.Kth, self.Mth = Kth, Mth
        fixed_T: dict[int, float] = {}
        self.flux = np.zeros(g.n_nodes)
        if self.active is None:
            for b in spec.boundaries:
                nodes = g.edge_nodes(b.edge, b.start, b.stop)
                if b.kind == "fixed-temperature":
                    for n in nodes.tolist():
                        fixed_T[n] = float(b.value)
                elif b.kind == "heat-flux":
                    q = float(b.value) * 0.5 * g.dx
                    np.add.at(self.flux, nodes[:-1], q)
                    np.add.at(self.flux, nodes[1:], q)
        else:
            # every boundary face its edge's condition; a flux per unit length of the
            # edge as drawn (`geometry.staircase_scale`)
            bf = geo.boundary_faces(d)
            owner = geo.face_conditions(spec.boundaries, bf, d.nx, d.ny)
            na, nb = bf.nodes(d.nx)
            scale = geo.staircase_scale(d, bf)
            for i, b in enumerate(spec.boundaries):
                sel = owner == i
                if not sel.any():
                    continue
                if b.kind == "fixed-temperature":
                    for n in np.concatenate([na[sel], nb[sel]]).tolist():
                        fixed_T[n] = float(b.value)
                elif b.kind == "heat-flux":
                    q = float(b.value) * 0.5 * g.dx * np.array(
                        [scale.get(e, 1.0) for e in bf.edge[sel]])
                    np.add.at(self.flux, na[sel], q)
                    np.add.at(self.flux, nb[sel], q)
        self.fixed = np.array(sorted(fixed_T), dtype=np.int64)
        self.T_fixed = np.array([fixed_T[n] for n in self.fixed.tolist()])
        self.free = np.setdiff1d(np.arange(g.n_nodes), np.union1d(self.fixed, self.off))
        A = (Mth / self.dt + Kth).tocsr()
        self.A = A
        self.A_ff = A[self.free][:, self.free]
        self.A_fd = A[self.free][:, self.fixed]
        self.th_lu = styles.Factor(self.A_ff)
        # -- elasticity: a free body, the rigid-body motions removed by a bordered system
        #    (over the domain's own nodes when it is drawn)
        self.K = fe.assemble_elastic(g, self.E, self.nu)
        self.G = fe.thermal_load_matrix(g, self.E, self.nu, self.alpha)
        if self.active is None:
            self.dofs = None
            V = sp.csc_matrix(fe.rigid_modes(g))
            self.me_lu = styles.Factor(sp.bmat([[self.K, V], [V.T, None]], format="csc"))
        else:
            on = np.setdiff1d(np.arange(g.n_nodes), self.off)
            self.dofs = np.stack([2 * on, 2 * on + 1], axis=1).ravel()
            V = sp.csc_matrix(rigid_modes_on(g, on))
            Ka = self.K[self.dofs][:, self.dofs]
            self.me_lu = styles.Factor(sp.bmat([[Ka, V], [V.T, None]], format="csc"))
        self.alpha_e = self.alpha
        self.arms = tuple(a for a in ARMS if a in arms)
        self.pool = (ThreadPoolExecutor(max_workers=2, thread_name_prefix="wb-thermo")
                     if "parallel" in self.arms else None)
        #: every arm's displacement and temperature after each step, for the two
        #: bitwise controls (references, not copies: steps return new arrays)
        self.hist: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {a: [] for a in ARMS}

    # -- the two agents' work -------------------------------------------------

    def _thermal(self, T: np.ndarray) -> np.ndarray:
        """One backward-Euler conduction step: the conduction agent's own."""
        rhs = (self.Mth @ T) / self.dt + self.flux
        out = np.empty_like(T)
        out[self.fixed] = self.T_fixed
        out[self.free] = self.th_lu.solve(rhs[self.free] - self.A_fd @ self.T_fixed)
        if self.off.size:
            out[self.off] = T[self.off]              # off the drawn domain: untouched
        return out

    def _mech(self, T: np.ndarray) -> np.ndarray:
        """The free body's displacement under the thermal strain of ``T``: the
        elasticity agent's own solve."""
        f = self.G @ (T - self.T0)
        if self.dofs is None:
            sol = self.me_lu.solve(np.concatenate([f, np.zeros(3)]))
            return sol[: f.size]
        sol = self.me_lu.solve(np.concatenate([f[self.dofs], np.zeros(3)]))
        u = np.zeros(f.size)
        u[self.dofs] = sol[: self.dofs.size]
        return u

    # -- the arms -------------------------------------------------------------

    def initial(self, arm: str) -> ThermoState:
        T = np.full(self.g.n_nodes, self.T0)
        for n, v in zip(self.fixed.tolist(), self.T_fixed.tolist()):
            T[n] = v            # the edges are held from the start
        return ThermoState(T, self._mech(T), T)

    def step(self, arm: str, s: ThermoState) -> ThermoState:
        if arm == "full":                                   # the unsplit solver
            T1 = self._thermal(s.T)
            out = ThermoState(T1, self._mech(T1), T1)
        elif arm == "serial":                               # the synchronous split
            T1 = self._thermal(s.T)                         # the conduction agent
            bond = np.array(T1, copy=True)                  # the field crosses
            out = ThermoState(T1, self._mech(bond), bond)   # the elasticity agent
        else:                                               # the lagged split
            ft = self.pool.submit(self._thermal, s.T)
            fu = self.pool.submit(self._mech, s.T)
            out = ThermoState(ft.result(), fu.result(), s.T)
        self.hist[arm].append((out.T, out.u))
        return out

    # -- instruments ---------------------------------------------------------

    def heat_balance(self, T_old: np.ndarray, T_new: np.ndarray) -> float:
        stored = float(np.sum(self.Mth @ (T_new - T_old)))
        r = self.A @ T_new - (self.Mth @ T_old) / self.dt - self.flux
        supplied = float(np.sum(r[self.fixed]))            # the fixed edges' heat, W/m
        into = self.dt * (float(np.sum(self.flux)) + supplied)
        through = self.dt * float(np.sum(np.abs(r[self.fixed])) + np.sum(np.abs(self.flux)))
        return abs(stored - into) / max(abs(stored), abs(into), through, 1e-300)

    def stress(self, s: ThermoState) -> np.ndarray:
        dT = s.T_u[self.g.conn()].mean(axis=1) - self.T0    # at each element's centre
        return fe.element_stress(self.g, s.u, self.E, self.nu, eps0=self.alpha_e * dT)

    def field(self, s: ThermoState) -> np.ndarray:
        vm = fe.von_mises(self.stress(s)) / 1e6
        if self.active is not None:
            vm = np.where(self.active, vm, np.nan)    # outside the drawn domain: none
        return vm.reshape(self.g.ny, self.g.nx)

    def observe(self, arm: str, s: ThermoState, prev: ThermoState | None = None) -> dict:
        T_old = prev.T if prev is not None else self.initial(arm).T
        mean_T = (float(np.mean(s.T)) if self.active is None
                  else float(np.mean(np.delete(s.T, self.off))))
        return {"balance": self.heat_balance(T_old, s.T),
                "max_stress": float(np.nanmax(self.field(s))),
                "mean_temperature": mean_T}

    def bitwise_equal(self, a: ThermoState, b: ThermoState) -> bool:
        return bool(np.array_equal(a.T, b.T) and np.array_equal(a.u, b.u)
                    and np.array_equal(a.T_u, b.T_u))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    # -- the end of a run ---------------------------------------------------

    def compare(self, states, history, bitwise):
        metrics: dict[str, Any] = {}
        for arm, rows in history.items():
            if not rows:
                continue
            metrics[arm] = {"max_stress_MPa": rows[-1]["max_stress"],
                            "mean_temperature_K": rows[-1]["mean_temperature"],
                            "balance_max": max(r["balance"] for r in rows)}
        checks = [judge(CHECKS[0], max((m["balance_max"] for m in metrics.values()),
                                       default=None),
                        "; ".join(f"{a}: {m['balance_max']:.3g}" for a, m in metrics.items()))]
        done = min((len(r) for r in history.values()), default=0)
        h = {a: self.hist[a][:done] for a in self.arms}
        if "serial" in h and "full" in h and done:
            bad = next((k + 1 for k in range(done)
                        if not (np.array_equal(h["serial"][k][0], h["full"][k][0])
                                and np.array_equal(h["serial"][k][1], h["full"][k][1]))),
                       None)
            checks.append(exact(CHECKS[1], bad is None,
                                f"equal after all {done} steps" if bad is None
                                else f"first differs after step {bad}"))
        else:
            checks.append(exact(CHECKS[1], None, "needs the synchronous split and the "
                                                 "unsplit solver"))
        if "serial" in h and "parallel" in h and done:
            u0 = self.initial("serial").u
            bad = next((k + 1 for k in range(done)
                        if not np.array_equal(h["parallel"][k][1],
                                              h["serial"][k - 1][1] if k else u0)), None)
            checks.append(exact(CHECKS[2], bad is None,
                                f"equal, one step late, after all {done} steps"
                                if bad is None else f"first differs after step {bad}"))
            ref = "full" if "full" in states else "serial"
            s_ref = self.field(states[ref])
            s_lag = self.field(states["parallel"])
            metrics["parallel"]["lag_stress_error"] = float(
                np.nanmax(np.abs(s_lag - s_ref)) / np.nanmax(np.abs(s_ref)))
        else:
            checks.append(exact(CHECKS[2], None, "needs both splits"))
        return metrics, checks

    def notes(self, done: int) -> list[str]:
        out = [f"Backward-Euler conduction steps of {self.dt:g} s; after each, the free "
               f"plate's quasi-static displacement under its thermal strain (reference "
               f"temperature {self.T0:g} K, the initial one). The bond between the two "
               f"agents is the whole temperature field."]
        out.append("The lagged split runs both agents at once, elasticity one step behind; "
                   "its stress error is the stress's change over one step, first order in "
                   "the step. The thermal-strain case study measured 7.1e-3 of the stress "
                   "for one lagged macro-step on its own shell.")
        out.append("The synchronous split does the unsplit solver's arithmetic and one copy "
                   "of the temperature field, so the ratio of their times measures this "
                   "machine's run-to-run noise at this size, not a method.")
        return out

    def describe(self) -> dict[str, Any]:
        return {"style": "split", "elements": self.g.n_elem, "nodes": self.g.n_nodes,
                "dt_s": self.dt, "T_ref_K": self.T0,
                "fixed_temperature_nodes": int(self.fixed.size)}


def build(spec, arms=ARMS, threads: int = 2) -> ThermoelasticRun:
    return ThermoelasticRun(spec, arms=arms, threads=threads)


def case_graph(spec):
    """The case for the compiler: refused before it, by the port vocabulary itself.

    The split's two agents share the whole domain and exchange the whole
    temperature field, a VOLUMETRIC bond.  The thermal-strain case study
    ([[case-study-thermal-strain-atlas-0.1]]) asked the package for that bond and
    it refused -- a sixth port type enters only through the six-field amendment
    procedure -- and this asks the same question of the same function
    (`ports.spec_for`), so the refusal is the package's, word for word, rather
    than a sentence written here.  The two routes the closed vocabulary does
    offer are both wrong in that study's measurement: as a surface MECH traction
    by 1279x in stress, and as a GlobalField exact but refused at
    L3/global-field since W117."""
    from atlas.ports import spec_for
    spec_for("VOLUMETRIC")            # raises holes.NamedHoleError (PORT_AMENDMENT)
    raise AssertionError("the port vocabulary accepted a volumetric bond")  # pragma: no cover


__all__ = ["FAMILY", "STYLE", "ARMS", "ARM_LABELS", "CHECKS", "ThermoelasticRun",
           "ThermoState", "build"]
