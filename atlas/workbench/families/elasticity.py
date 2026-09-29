"""The elasticity family, run from a case file: style B on a structure.

A two-material bracket under load (showcase case 6): quasi-static plane-stress
elasticity on the case's grid, Q1 elements with a material per region
(`fe.py`, the build repo's `ThermoStruct2D` element with a material per
element).  Edges are clamped, free, or loaded by a uniform traction.

**Style B: overlapping windows iterated until they agree** -- restricted
additive Schwarz (`styles.schwarz`) on the stiffness with the clamped degrees of
freedom removed.  A window is the nodes of its cells; its local matrix is the
global stiffness restricted to them (a principal submatrix of a positive
definite matrix, so every window has a unique solution even when it touches no
support), and the coupling to the nodes outside it is kept as a matrix, so the
right-hand side is ``b - C u`` from the current iterate.  The partition of unity
is the case's, built on the node grid (`tiling.RectangleTiling` over the
windows' node boxes).  The iteration stops when the update is below the
tolerance relative to the iterate's own largest displacement.  Additive, so a
threaded arm is the serial arm to the bit.

The full domain is the same stiffness solved directly, once factorized.  The
problem is steady: every step is the whole solve again from zero, repeated to
time it.

**The displayed field** is the von Mises stress at each element's centre, which
is where a material interface shows: the stiffer material carries more of the
load across the section.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .. import fe, styles
from .. import geometry as geo
from ..checks import CheckSpec, exact, judge
from ..fv import LocalSystem
from ..tiling import RectangleTiling

FAMILY = "elasticity-2d"
STYLE = "B"
ARMS = ("serial", "parallel", "full")
FIELD_LABEL = "von Mises stress (MPa)"
LENGTH_UNIT = "m"
SERIES: dict[str, str] = {}

REGISTERED = "2026-09-29, before the elasticity family's first run"

CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        key="force", title="Forces: the supports balance the loads", kind="balance",
        tolerance=1e-6, registered=REGISTERED,
        measure=("|sum of the support reactions + sum of the applied loads| over |sum of "
                 "the applied loads|, as vectors, every repeat, worst arm"),
        why=("the stiffness annihilates a rigid translation, so reactions plus loads are "
             "minus the net force of the residual at the free nodes: round-off for the "
             "direct solve, and for the decomposed arm the residual its iteration stops "
             "at -- at most about 4 E times the last update per degree of freedom, 8e-4 "
             "N/m for the example at an update of 1e-12 of its millimetre displacement, "
             "which summed over its 13,000 degrees of freedom with independent signs is "
             "2e-7 of the load. With one sign everywhere it would be 2e-5 and this check "
             "would fail, and say so")),
    CheckSpec(
        key="reference", title="Displacements agree with the full domain", kind="reference",
        tolerance=1e-6, registered=REGISTERED,
        measure="max |u_decomposed - u_full| over max |u_full|, at the last repeat",
        why=("both solve the same stiffness; the converged Schwarz iterate is the direct "
             "solution to the iteration's tolerance over one minus its contraction "
             "factor")),
    CheckSpec(
        key="bitwise", title="Threaded equals serial, bit for bit", kind="control",
        tolerance=None, registered=REGISTERED,
        measure="np.array_equal on the displacements after every repeat",
        why=("additive Schwarz solves every window from the same iterate and blends in "
             "window order, so the order the threads finish in cannot change a bit")),
)


def element_props(spec, *names: str) -> list[np.ndarray]:
    """Each named material property per element (= cell), in cell order."""
    d = spec.domain
    owner = geo.region_owner(spec.regions, d.nx, d.ny).ravel()
    out = []
    for name in names:
        v = np.zeros(d.nx * d.ny)
        for i, r in enumerate(spec.regions):
            v[owner == i] = float(spec.materials[r.material][name])
        out.append(v)
    return out


def supports_and_loads(spec, g: fe.QuadGrid) -> tuple[np.ndarray, np.ndarray]:
    """(the clamped degrees of freedom, the nodal load vector per metre of thickness)."""
    fixed: set[int] = set()
    f = np.zeros(2 * g.n_nodes)
    for b in spec.boundaries:
        if b.kind == "clamped":
            nodes = g.edge_nodes(b.edge, b.start, b.stop)
            fixed.update((2 * nodes).tolist())
            fixed.update((2 * nodes + 1).tolist())
        elif b.kind in ("load-x", "load-y"):
            t = float(b.value or 0.0)
            f += fe.edge_force(g, b.edge, b.start, b.stop, t if b.kind == "load-x" else 0.0,
                               t if b.kind == "load-y" else 0.0)
    return np.array(sorted(fixed), dtype=np.int64), f


def node_box(w) -> tuple[int, int, int, int]:
    """A window's nodes: the corners of its cells."""
    return (w.x0, w.y0, w.nx + 1, w.ny + 1)


@dataclass
class ElasState:
    u: np.ndarray                         # [2 n_nodes], clamped dofs included (zero)
    iterations: int = 0
    converged: bool = True
    history: list[float] = field(default_factory=list)


class ElasticityRun:
    family = FAMILY

    def __init__(self, spec, arms=ARMS, threads: int = 2):
        self.spec = spec
        d = spec.domain
        self.g = fe.QuadGrid(d.nx, d.ny, float(d.dx))
        self.E, self.nu = element_props(spec, "E", "nu")
        self.K = fe.assemble_elastic(self.g, self.E, self.nu)
        self.fixed, self.f = supports_and_loads(spec, self.g)
        n = 2 * self.g.n_nodes
        self.free = np.setdiff1d(np.arange(n), self.fixed)
        loc = np.full(n, -1, dtype=np.int64)
        loc[self.free] = np.arange(self.free.size)
        self.loc = loc
        Kff = self.K[self.free][:, self.free].tocsr()
        self.Kff, self.ff = Kff, self.f[self.free]
        self.thickness = float(spec.physics.get("thickness"))
        self.tol = float(spec.coupling.tolerance)
        self.max_it = int(spec.coupling.max_iterations)
        self.arms = tuple(a for a in ARMS if a in arms)
        self.threads = max(1, int(threads))
        self.full_lu = styles.Factor(Kff) if "full" in self.arms else None
        # the windows, on the node grid
        self.windows = [(w.id, node_box(w)) for w in spec.windows]
        self.tiling = RectangleTiling(d.nx + 1, d.ny + 1, self.windows,
                                      spec.coupling.ramp_cells)
        self.certificate = self.tiling.certify()
        self.systems, self.chi = [], []
        for (_name, (x0, y0, w, h)), chi in zip(self.windows, self.tiling.chi):
            jj, ii = np.meshgrid(np.arange(y0, y0 + h), np.arange(x0, x0 + w), indexing="ij")
            nodes = (jj * (d.nx + 1) + ii).ravel()
            dofs = np.stack([2 * nodes, 2 * nodes + 1], axis=1).ravel()
            keep = loc[dofs] >= 0
            idx = loc[dofs[keep]]
            order = np.argsort(idx)
            idx = idx[order]
            wts = np.repeat(chi.ravel(), 2)[keep][order]
            A = Kff[idx][:, idx].tocsr()
            mask = np.ones(Kff.shape[0], dtype=bool)
            mask[idx] = False
            C = Kff[idx].tocsr()
            C = (C @ sp.diags(mask.astype(float))).tocsr()
            C.eliminate_zeros()
            empty = np.zeros(0, dtype=np.int64)
            self.systems.append(LocalSystem(idx, A, self.ff[idx].copy(),
                                            C if C.nnz else None, empty, empty,
                                            np.zeros(0), "neighbour"))
            self.chi.append(wts)
        self.factors = [styles.Factor(s.A) for s in self.systems]
        self.pool = None
        if "parallel" in self.arms:
            self.pool = ThreadPoolExecutor(max_workers=min(self.threads, len(self.systems)),
                                           thread_name_prefix="wb-elas")

    # -- the arms -------------------------------------------------------------

    def initial(self, arm: str) -> ElasState:
        return ElasState(np.zeros(2 * self.g.n_nodes))

    def _full_vector(self, uf: np.ndarray) -> np.ndarray:
        u = np.zeros(2 * self.g.n_nodes)
        u[self.free] = uf
        return u

    def step(self, arm: str, s: ElasState) -> ElasState:
        """One whole solve from zero (the problem is steady)."""
        if arm == "full":
            return ElasState(self._full_vector(self.full_lu.solve(self.ff)), 1, True)
        it = styles.schwarz(self.systems, self.factors, self.chi,
                            np.zeros(self.free.size), self.tol, self.max_it, None,
                            pool=self.pool if arm == "parallel" else None)
        return ElasState(self._full_vector(it.u), it.iterations, it.converged, it.history)

    # -- instruments ---------------------------------------------------------

    def forces(self, s: ElasState) -> dict[str, float]:
        """The applied load and the support reactions, N (thickness included)."""
        r = self.K @ s.u - self.f                      # reactions at the supports
        t = self.thickness
        load = np.array([self.f[0::2].sum(), self.f[1::2].sum()]) * t
        react = np.array([r[self.fixed[self.fixed % 2 == 0]].sum(),
                          r[self.fixed[self.fixed % 2 == 1]].sum()]) * t
        imbalance = float(np.linalg.norm(load + react)) / max(float(np.linalg.norm(load)),
                                                              1e-300)
        return {"load_N": float(np.linalg.norm(load)), "reaction_x_N": float(react[0]),
                "reaction_y_N": float(react[1]), "balance": imbalance}

    def observe(self, arm: str, s: ElasState, prev: ElasState | None = None) -> dict:
        f = self.forces(s)
        tip = float(np.max(np.abs(s.u)))
        return {"balance": f["balance"], "load_N": f["load_N"],
                "reaction_y_N": f["reaction_y_N"], "max_displacement_m": tip,
                "iterations": float(s.iterations), "converged": float(s.converged),
                "convergence": list(s.history)}

    def bitwise_equal(self, a: ElasState, b: ElasState) -> bool:
        return bool(np.array_equal(a.u, b.u))

    def stress(self, s: ElasState) -> np.ndarray:
        return fe.element_stress(self.g, s.u, self.E, self.nu)

    def field(self, s: ElasState) -> np.ndarray:
        return (fe.von_mises(self.stress(s)) / 1e6).reshape(self.g.ny, self.g.nx)

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
            metrics[arm] = {"max_displacement_mm": 1e3 * rows[-1]["max_displacement_m"],
                            "load_N": rows[-1]["load_N"],
                            "balance_max": max(r["balance"] for r in rows),
                            "iterations_last": rows[-1]["iterations"],
                            "all_converged": all(r["converged"] for r in rows),
                            "max_stress_MPa": float(np.max(self.field(states[arm])))}
        checks = [judge(CHECKS[0], max((m["balance_max"] for m in metrics.values()),
                                       default=None),
                        "; ".join(f"{a}: {m['balance_max']:.3g}" for a, m in metrics.items()))]
        dec = "parallel" if "parallel" in metrics else ("serial" if "serial" in metrics
                                                        else None)
        if dec and "full" in metrics:
            uf = states["full"].u
            dev = float(np.max(np.abs(states[dec].u - uf))) / float(np.max(np.abs(uf)))
            metrics[dec]["displacement_vs_full"] = dev
            metrics[dec]["stress_vs_full_MPa"] = float(np.max(np.abs(
                self.field(states[dec]) - self.field(states["full"]))))
            checks.append(judge(CHECKS[1], dev, f"largest displacement "
                                                f"{1e3 * float(np.max(np.abs(uf))):.4g} mm"))
        else:
            checks.append(judge(CHECKS[1], None, "needs a decomposed arm and the full domain"))
        if bitwise.get("steps"):
            ok = bitwise["first_difference"] is None
            checks.append(exact(CHECKS[2], ok, f"equal after all {bitwise['steps']} repeats"
                                if ok else f"first differs after repeat "
                                           f"{bitwise['first_difference']}"))
        else:
            checks.append(exact(CHECKS[2], None, "needs both decomposed arms"))
        return metrics, checks

    def notes(self, done: int) -> list[str]:
        f = self.forces(ElasState(np.zeros(2 * self.g.n_nodes)))
        return [f"Steady: every repeat is the whole solve again from zero displacement. The "
                f"load is {f['load_N']:.4g} N on a plate {self.thickness * 1e3:g} mm thick "
                f"(plane stress, so the displacements do not depend on the thickness).",
                f"Schwarz stops when the update is below {self.tol:g} of the iterate's "
                f"largest displacement; {len(self.systems)} windows on the node grid, "
                f"{self.free.size:,} free degrees of freedom."]

    def describe(self) -> dict[str, Any]:
        return {"style": "B", "mode": "steady", "elements": self.g.n_elem,
                "free_dofs": int(self.free.size), "windows": len(self.systems),
                "tolerance": self.tol, "max_iterations": self.max_it,
                "thickness_m": self.thickness,
                "partition_of_unity": self.certificate.as_dict()}


def step_label(spec) -> str:
    return "timed repeat"


def build(spec, arms=ARMS, threads: int = 2) -> ElasticityRun:
    return ElasticityRun(spec, arms=arms, threads=threads)


__all__ = ["FAMILY", "STYLE", "ARMS", "CHECKS", "ElasticityRun", "ElasState", "build",
           "element_props", "supports_and_loads", "node_box", "step_label"]
