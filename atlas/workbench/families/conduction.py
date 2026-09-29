"""The conduction family, run from a case file: styles B and C, steady or transient.

Heat conduction in several materials, ``rho cp dT/dt = div(k grad T)``, on the
case's grid with a material per region (`fv.py`: harmonic-mean faces, so a
wall of layers is exact).  The second real family the solver-family interface
was derived from (`styles.py`'s table), and the first that reads regions.

Two ways to decompose it:

``B``  overlapping windows iterated until they agree: restricted additive
       Schwarz (`styles.schwarz`), every window a sparse LU of its own
       restriction, blended by the case's partition of unity.  Additive, so a
       threaded arm is the serial arm to the bit (checked every step).
``C``  two windows that meet along a face, coupled by Dirichlet-Neumann
       (`styles.dirichlet_neumann`): the Dirichlet piece gets the interface's
       temperatures, the Neumann piece the heat the Dirichlet piece sends
       across.  **The relaxation is stated**: the case's first factor, then
       Aitken's rule if the case asks for it; the page shows the factor every
       iteration and the 1-D contraction factor ``rho = k_D L_N / (k_N L_D)``
       that says whether the unrelaxed iteration would converge.  Sequential by
       construction (the Neumann piece needs the Dirichlet piece's heat), so it
       has no threaded arm.

**Steady** runs solve once per timed repeat, from the case's initial
temperature; **transient** runs take backward-Euler steps of ``run.macro_dt``
seconds, each iterated to the tolerance, warm-started from the step before.
The full domain is the same finite volumes on every cell, factorized once.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import fv, styles
from .. import geometry as geo
from ..checks import CheckSpec, exact, judge
from ..tiling import RectangleTiling

FAMILY = "conduction-2d"
STYLE = "B or C"
ARMS = ("serial", "parallel", "full")
FIELD_LABEL = "temperature (K)"
LENGTH_UNIT = "m"
SERIES = {"heat_in": "Heat into the plate per step | W per metre of depth"}

REGISTERED = "2026-09-28, before the conduction family's first run"

CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        key="balance", title="Energy: heat in = heat stored", kind="balance",
        tolerance=1e-6, registered=REGISTERED,
        measure=("|heat into the domain through its edges - heat stored| over the heat "
                 "through the plate, every step, worst arm (steady: stored = 0)"),
        why=("the full domain closes to the direct solver's round-off. A decomposed arm "
             "closes to its iteration's floor: the solve stops when the update is below "
             "1e-10 of the temperature scale, and a heat flow is a temperature difference "
             "over a half-cell, so its relative error is that times T/dT, with the "
             "material contrast, well under 1e-6")),
    CheckSpec(
        key="reference", title="Temperatures agree with the full domain", kind="reference",
        tolerance=1e-6, registered=REGISTERED,
        measure=("max |T_decomposed - T_full| over the temperature span of the fixed "
                 "boundaries, at the last step"),
        why=("both solve the same finite volumes; the converged decomposition IS the "
             "full-domain solution up to the iteration's tolerance (Dirichlet-Neumann's "
             "converged face flow is the harmonic-mean face exactly)")),
    CheckSpec(
        key="closed_form", title="Heat flow = dT / sum(L/k) (a layered wall)",
        kind="reference", tolerance=1e-6, registered=REGISTERED,
        measure=("|Q - Q_closed| / Q_closed for every arm, where Q_closed = dT H / sum_i "
                 "L_i / k_i; only for a steady wall of layers across x with fixed "
                 "temperatures left and right and insulated top and bottom"),
        why=("harmonic-mean faces add the half-cells' resistances in series, so the "
             "discrete heat flow through layers on cell faces is the closed form to "
             "round-off; a decomposed arm adds its iteration floor")),
    CheckSpec(
        key="bitwise", title="Threaded equals serial, bit for bit (style B)",
        kind="control", tolerance=None, registered=REGISTERED,
        measure="np.array_equal on the temperature after every step",
        why=("additive Schwarz solves every window from the same iterate and blends in "
             "window order, so the order the threads finish in cannot change a bit")),
)

_KIND = {"fixed-temperature": fv.FIXED, "insulated": fv.NO_FLUX, "heat-flux": fv.FLUX}


def field_from_case(spec, prop_k: str = "k", kinds: dict[str, int] | None = None) -> fv.Field:
    """The case's grid, materials and boundaries as an `fv.Field` (no capacity)."""
    d = spec.domain
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    k = np.zeros((d.ny, d.nx))
    for i, r in enumerate(spec.regions):
        k[owner == i] = float(spec.materials[r.material][prop_k])
    kinds = kinds or _KIND
    bc = {}
    for e in fv.EDGES:
        n = geo.edge_length(e, d.nx, d.ny)
        kd, vv = np.zeros(n, dtype=np.int8), np.zeros(n)
        for b in spec.boundaries:
            if b.edge != e:
                continue
            stop = n if b.stop is None else b.stop
            kd[b.start:stop] = kinds[b.kind]
            vv[b.start:stop] = 0.0 if b.value is None else float(b.value)
        bc[e] = (kd, vv)
    return fv.Field(d.nx, d.ny, float(d.dx), k, bc=bc)


def capacity(spec) -> np.ndarray:
    d = spec.domain
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    cap = np.zeros((d.ny, d.nx))
    for i, r in enumerate(spec.regions):
        m = spec.materials[r.material]
        cap[owner == i] = float(m["rho"]) * float(m["cp"])
    return cap


def window_cells(nx: int, box) -> np.ndarray:
    x0, y0, w, h = box
    jj, ii = np.meshgrid(np.arange(y0, y0 + h), np.arange(x0, x0 + w), indexing="ij")
    return (jj * nx + ii).ravel()


@dataclass
class CondState:
    u: np.ndarray
    iterations: int = 0
    converged: bool = True
    history: list[float] = field(default_factory=list)
    relaxation: list[float] = field(default_factory=list)
    lam: np.ndarray | None = None          # style C: the interface's face values


class ConductionRun:
    family = FAMILY

    def __init__(self, spec, arms=ARMS, threads: int = 2):
        self.spec = spec
        self.style = spec.coupling.style
        self.mode = spec.run.mode
        self.f = field_from_case(spec)
        self.nx, self.ny, self.dx = self.f.nx, self.f.ny, self.f.dx
        self.T0 = float(spec.physics.get("T0"))
        fixed = [b.value for b in spec.boundaries if b.kind == "fixed-temperature"
                 and b.value is not None]
        self.scale = max([abs(self.T0)] + [abs(float(v)) for v in fixed])
        self.span = (max(fixed) - min(fixed)) if len(fixed) >= 2 else max(
            [abs(float(v) - self.T0) for v in fixed] + [1.0])
        self.tol = float(spec.coupling.tolerance)
        self.max_it = int(spec.coupling.max_iterations)
        self.dt = float(spec.run.macro_dt)
        self.transient = self.mode == "transient"
        self.m = (capacity(spec) * self.dx * self.dx / self.dt).ravel() if self.transient \
            else None
        allowed = available_arms(spec)[0]
        self.arms = tuple(a for a in ARMS if a in arms and a in allowed)
        self.threads = max(1, int(threads))
        self.pool = None
        # the full domain: the same assembly on every cell, factorized once
        self.full_sys = fv.assemble(self.f, diag_add=self.m)
        self.full_lu = styles.Factor(self.full_sys.A) if "full" in self.arms else None
        self.windows = [(w.id, (w.x0, w.y0, w.nx, w.ny)) for w in spec.windows]
        if self.style == "B":
            self.tiling = RectangleTiling(self.nx, self.ny, self.windows,
                                          spec.coupling.ramp_cells)
            self.certificate = self.tiling.certify()
            self.systems = [fv.assemble(self.f, window_cells(self.nx, b), diag_add=self.m)
                            for _n, b in self.windows]
            self.factors = [styles.Factor(s.A) for s in self.systems]
            self.chi = [c.ravel() for c in self.tiling.chi]
            if "parallel" in self.arms:
                self.pool = ThreadPoolExecutor(max_workers=min(self.threads,
                                                               len(self.systems)),
                                               thread_name_prefix="wb-cond")
        else:
            self.certificate = None
            d_id, n_id = dirichlet_side(spec, self.f, self.m)
            box = dict(self.windows)
            self.d_id, self.n_id = d_id, n_id
            self.sd = fv.assemble(self.f, window_cells(self.nx, box[d_id]), "dirichlet",
                                  diag_add=self.m)
            self.sn = fv.assemble(self.f, window_cells(self.nx, box[n_id]), "neumann",
                                  diag_add=self.m)
            self.fd, self.fn = styles.Factor(self.sd.A), styles.Factor(self.sn.A)
            self.pairing = styles.pair_faces(self.sd, self.sn)
            self.rho_1d = rho_1d(spec, d_id, n_id)

    # -- the arms -------------------------------------------------------------

    def initial(self, arm: str) -> CondState:
        return CondState(np.full(self.f.n, self.T0))

    def _rhs_extra(self, u_old: np.ndarray):
        return None if self.m is None else self.m * u_old

    def step(self, arm: str, s: CondState) -> CondState:
        """One time step (transient) or one solve from the start (steady)."""
        u_old = s.u if self.transient else np.full(self.f.n, self.T0)
        extra = self._rhs_extra(u_old)
        if arm == "full":
            b = self.full_sys.b if extra is None else self.full_sys.b + extra
            return CondState(self.full_lu.solve(b), 1, True)
        if self.style == "B":
            eb = None if extra is None else [extra[sys_.idx] for sys_ in self.systems]
            it = styles.schwarz(self.systems, self.factors, self.chi, u_old, self.tol,
                                self.max_it, self.scale, extra_b=eb,
                                pool=self.pool if arm == "parallel" else None)
            return CondState(it.u, it.iterations, it.converged, it.history)
        lam0 = (s.lam if (self.transient and s.lam is not None)
                else np.full(self.sd.face_rows.size, float(np.mean(u_old))))
        ed = None if extra is None else extra[self.sd.idx]
        en = None if extra is None else extra[self.sn.idx]
        it = styles.dirichlet_neumann(self.sd, self.fd, self.sn, self.fn, lam0, self.tol,
                                      self.max_it, self.scale,
                                      theta0=self.spec.coupling.relaxation,
                                      aitken=self.spec.coupling.aitken, extra_bd=ed,
                                      extra_bn=en, pairing=self.pairing)
        return CondState(it.u, it.iterations, it.converged, it.history, it.relaxation,
                         it.extra["lam"])

    # -- instruments ---------------------------------------------------------

    def heat(self, s: CondState, u_old: np.ndarray) -> dict[str, float]:
        q = fv.boundary_inflow(self.f, s.u)
        into = float(sum(q.values())) + fv.total_source(self.f)
        stored = float(np.sum(self.m * (s.u - u_old))) if self.m is not None else 0.0
        through = 0.5 * sum(abs(v) for v in q.values())
        scale = max(through, abs(stored), 1e-300)
        return {"heat_in": float(sum(v for v in q.values() if v > 0.0)),
                "balance": abs(into - stored) / scale, "through": through}

    def observe(self, arm: str, s: CondState, prev: CondState | None = None) -> dict:
        u_old = prev.u if (self.transient and prev is not None) else np.full(self.f.n,
                                                                              self.T0)
        h = self.heat(s, u_old)
        return {"heat_in": h["heat_in"], "balance": h["balance"],
                "iterations": float(s.iterations), "converged": float(s.converged),
                "convergence": list(s.history),
                "relaxation": list(s.relaxation)}

    def bitwise_equal(self, a: CondState, b: CondState) -> bool:
        return bool(np.array_equal(a.u, b.u))

    def field(self, s: CondState) -> np.ndarray:
        return s.u.reshape(self.ny, self.nx)

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
            metrics[arm] = {"heat_in": rows[-1]["heat_in"],
                            "balance_max": max(r["balance"] for r in rows),
                            "iterations_last": rows[-1]["iterations"],
                            "iterations_max": max(r["iterations"] for r in rows),
                            "all_converged": all(r["converged"] for r in rows)}
        checks = []
        worst = max((m["balance_max"] for m in metrics.values()), default=None)
        checks.append(judge(CHECKS[0], worst, "; ".join(
            f"{a}: {m['balance_max']:.3g}" for a, m in metrics.items())))
        dec = "parallel" if "parallel" in metrics else ("serial" if "serial" in metrics
                                                        else None)
        if dec and "full" in metrics:
            dev = float(np.max(np.abs(states[dec].u - states["full"].u))) / self.span
            metrics[dec]["max_temperature_difference_K"] = dev * self.span
            metrics[dec]["heat_in_vs_full"] = ((metrics[dec]["heat_in"]
                                                - metrics["full"]["heat_in"])
                                               / metrics["full"]["heat_in"])
            checks.append(judge(CHECKS[1], dev, f"max |dT| = {dev * self.span:.3g} K over a "
                                                f"span of {self.span:g} K"))
        else:
            checks.append(judge(CHECKS[1], None, "needs a decomposed arm and the full "
                                                 "domain"))
        q_closed = closed_form_heat(self.spec)
        if q_closed is None:
            checks.append(judge(CHECKS[2], None, "not a steady wall of layers across x, so "
                                                 "there is no closed form"))
        else:
            errs = {a: abs(m["heat_in"] - q_closed) / q_closed for a, m in metrics.items()}
            for a in metrics:
                metrics[a]["heat_in_vs_closed_form"] = (metrics[a]["heat_in"]
                                                        - q_closed) / q_closed
            checks.append(judge(CHECKS[2], max(errs.values()), "; ".join(
                f"{a}: {e:.3g}" for a, e in errs.items())
                + f" (closed form {q_closed:.6g} W/m)", q_closed=q_closed))
        if self.style == "B" and bitwise.get("steps"):
            ok = bitwise["first_difference"] is None
            checks.append(exact(CHECKS[3], ok, f"equal after all {bitwise['steps']} steps"
                                if ok else f"first differs after step "
                                           f"{bitwise['first_difference']}"))
        else:
            checks.append(exact(CHECKS[3], None, "style C has no threaded arm" if
                                self.style == "C" else "needs both decomposed arms"))
        return metrics, checks

    def notes(self, done: int) -> list[str]:
        out = []
        if self.style == "C":
            out.append(f"Dirichlet side: {self.d_id}; Neumann side: {self.n_id}. The 1-D "
                       f"contraction factor rho = k_D L_N / (k_N L_D) = {self.rho_1d:.3g}"
                       + (" (under 1: the unrelaxed iteration would contract)"
                          if self.rho_1d < 1 else
                          " (over 1: unrelaxed it would diverge; the relaxation carries it)")
                       + f". First relaxation factor {self.spec.coupling.relaxation:g}, "
                       + ("then Aitken's rule." if self.spec.coupling.aitken else
                          "held fixed."))
        if not self.transient:
            out.append("Steady: every 'step' is the whole solve again from the initial "
                       "temperature, repeated to time it; the answer is the same each time.")
        return out

    def describe(self) -> dict[str, Any]:
        out = {"style": self.style, "mode": self.mode, "cells": self.f.n,
               "windows": len(self.windows), "tolerance": self.tol,
               "max_iterations": self.max_it, "scale_K": self.scale}
        if self.certificate is not None:
            out["partition_of_unity"] = self.certificate.as_dict()
        if self.style == "C":
            out.update({"dirichlet": self.d_id, "neumann": self.n_id,
                        "rho_1d": self.rho_1d, "relaxation0": self.spec.coupling.relaxation,
                        "aitken": self.spec.coupling.aitken,
                        "interface_faces": int(self.sd.face_rows.size)})
        return out


# ---------------------------------------------------------------------------
# helpers the page and the checks share
# ---------------------------------------------------------------------------


def step_label(spec) -> str:
    """A transient run's step is a macro-step; a steady run's is a timed repeat of
    the whole solve (the answer is the same every repeat)."""
    return "macro-step" if spec.run.mode == "transient" else "timed repeat"


def available_arms(spec) -> tuple[tuple[str, ...], dict[str, str]]:
    """The arms this case can run, and why the others cannot."""
    if spec.coupling.style == "C":
        return ("serial", "full"), {"parallel": "Dirichlet-Neumann is sequential by "
                                                "construction: the Neumann piece needs the "
                                                "heat the Dirichlet piece sends"}
    return ARMS, {}


def _mean_k(spec, box) -> float:
    d = spec.domain
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    x0, y0, w, h = box
    ks = [float(spec.materials[spec.regions[i].material]["k"])
          for i in owner[y0:y0 + h, x0:x0 + w].ravel() if i >= 0]
    return float(np.mean(ks)) if ks else 1.0


def floating(f: fv.Field, cells: np.ndarray, diag_add: np.ndarray | None = None) -> bool:
    """Whether a piece given only a flow on its cut faces has nothing to set its level:
    no fixed temperature, no outflow, no storage -- every row of its matrix sums to
    zero, so the matrix is singular and a Neumann solve of it is meaningless."""
    A = fv.assemble(f, cells, "neumann", diag_add=diag_add).A
    rows = np.abs(np.asarray(A.sum(axis=1)).ravel())
    return bool(float(rows.max()) <= 1e-12 * float(np.max(np.abs(A.diagonal()))))


def dirichlet_side(spec, f: fv.Field | None = None,
                   diag_add: np.ndarray | None = None) -> tuple[str, str]:
    """(Dirichlet window, Neumann window): the case's choice, or ``auto``.

    ``auto``: a piece that would FLOAT as the Neumann side (`floating`: nothing of
    its own sets its level) takes the Dirichlet side, which the interface's values
    pin.  Otherwise the window with the lower mean conductivity is the Dirichlet
    side, the side under which the 1-D iteration contracts without relaxation when
    the lengths are comparable.  The floating rule came from a failed run: the
    cooled block's first run made the insulated block the Neumann side on the
    conductivity rule alone, and the iteration ran out (2026-09-29)."""
    a, b = spec.windows[0], spec.windows[1]
    side = spec.coupling.dirichlet_side
    if side == a.id:
        return a.id, b.id
    if side == b.id:
        return b.id, a.id
    if f is not None:
        fa = floating(f, window_cells(f.nx, (a.x0, a.y0, a.nx, a.ny)), diag_add)
        fb = floating(f, window_cells(f.nx, (b.x0, b.y0, b.nx, b.ny)), diag_add)
        if fa != fb:
            return (a.id, b.id) if fa else (b.id, a.id)
    ka = _mean_k(spec, (a.x0, a.y0, a.nx, a.ny))
    kb = _mean_k(spec, (b.x0, b.y0, b.nx, b.ny))
    return (a.id, b.id) if ka <= kb else (b.id, a.id)


def rho_1d(spec, d_id: str, n_id: str) -> float:
    """``k_D L_N / (k_N L_D)``: the 1-D Dirichlet-Neumann contraction factor, with
    each piece's mean conductivity and its length normal to the interface."""
    w = {x.id: x for x in spec.windows}
    a, b = w[d_id], w[n_id]
    along_x = a.x0 + a.nx == b.x0 or b.x0 + b.nx == a.x0
    la, lb = (a.nx, b.nx) if along_x else (a.ny, b.ny)
    ka = _mean_k(spec, (a.x0, a.y0, a.nx, a.ny))
    kb = _mean_k(spec, (b.x0, b.y0, b.nx, b.ny))
    return (ka * lb) / (kb * la)


def closed_form_heat(spec) -> float | None:
    """``dT H / sum_i L_i / k_i`` when the case is a steady wall of layers across x.

    Applies only when: the run is steady; the left and right edges are each one
    fixed-temperature segment over the whole edge; the top and bottom are
    insulated over the whole edge; and every column is one material (so the
    layers run top to bottom).  None otherwise.
    """
    if spec.run.mode != "steady":
        return None
    d = spec.domain
    by_edge: dict[str, list] = {e: [] for e in fv.EDGES}
    for b in spec.boundaries:
        by_edge[b.edge].append(b)
    for e in ("left", "right"):
        segs = by_edge[e]
        if len(segs) != 1 or segs[0].kind != "fixed-temperature" or segs[0].start != 0 \
                or segs[0].stop not in (None, d.ny):
            return None
    for e in ("bottom", "top"):
        if not by_edge[e] or any(b.kind != "insulated" for b in by_edge[e]):
            return None
        cov = np.zeros(d.nx, dtype=bool)
        for b in by_edge[e]:
            cov[b.start:(d.nx if b.stop is None else b.stop)] = True
        if not cov.all():
            return None
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    if np.any(owner < 0) or np.any(owner != owner[0:1, :]):
        return None
    k_col = np.array([float(spec.materials[spec.regions[i].material]["k"])
                      for i in owner[0]])
    dT = abs(float(by_edge["left"][0].value) - float(by_edge["right"][0].value))
    return dT * (d.ny * d.dx) / float(np.sum(d.dx / k_col))


def build(spec, arms=ARMS, threads: int = 2) -> ConductionRun:
    return ConductionRun(spec, arms=arms, threads=threads)


def case_graph(spec):
    """The case for the compiler (`compile.py`): one agent per window, each its own
    finite volumes, THERM seams derived from the geometry.

    A window's step is a sparse direct solve over the window -- steady, or one
    backward-Euler step from the initial temperature -- so it declares an
    EMBEDDED elliptic solve, which is true.  Its response to the temperatures on
    one port's faces is the entropy flux ``q_n / T`` through them after that
    solve.  The probe linearizes about the case's own temperature level (W74:
    a temperature has an absolute origin)."""
    from ..compile import fv_graph
    d = spec.domain
    f = field_from_case(spec)
    transient = spec.run.mode == "transient"
    T0 = float(spec.physics.get("T0"))
    fixed = [float(b.value) for b in spec.boundaries
             if b.kind == "fixed-temperature" and b.value is not None]
    base = T0 if (transient or not fixed) else float(np.mean(fixed))
    scale = max([abs(T0)] + [abs(v) for v in fixed])
    span = (max(fixed) - min(fixed)) if len(fixed) >= 2 else max(
        [abs(v - T0) for v in fixed] + [1.0])
    k_max = max(float(m["k"]) for m in spec.materials.values())
    power = k_max * span / (max(d.nx, d.ny) * d.dx)          # W/m^2, a flux's size
    m = (capacity(spec) * d.dx * d.dx / float(spec.run.macro_dt)).ravel() \
        if transient else None
    graph, _agents = fv_graph(
        spec, f, family=FAMILY, port_type="THERM",
        scales={"temperature": scale, "entropy_flux": power / scale, "power_area": power},
        implicit=True, base=base, per_effort=True,
        lambda_ref="the same finite volumes on every cell (fv.assemble), the full-domain arm",
        diag_add=m, extra=None if m is None else m * T0,
        response_note="the window's entropy flux q_n / T into it through these faces "
                      "after its own solve, every other cut face at the probe base",
        validity=_valid_everywhere)
    return graph


def _valid_everywhere(state=None, cond=None) -> bool:
    """Linear conduction with positive conductivities: a direct solve of it holds for
    every temperature field, and the case's own check has already refused a
    conductivity that is not positive."""
    return True


__all__ = ["FAMILY", "STYLE", "ARMS", "CHECKS", "ConductionRun", "CondState", "build",
           "field_from_case", "capacity", "available_arms", "step_label", "dirichlet_side",
           "floating", "rho_1d", "closed_form_heat", "window_cells"]
