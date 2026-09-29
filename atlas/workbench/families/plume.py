"""The river-plume family, run from a case file: style A with an explicit step.

A pollutant released at an outfall is carried down a river and mixed across it
(showcase case 3).  Depth-averaged advection and mixing,

    h dc/dt  +  div(q c)  -  div(h D grad c)  =  s,

on the case's grid.  The discharge per metre of width ``q`` is uniform along the
river (continuity), and each reach -- a region -- has its own depth ``h`` and
mixing coefficient ``D``, so the water runs at ``u = q / h``: faster in a
shallow reach, slower in a deep one.  It is `fv.py`'s operator with capacity
``h``, diffusion coefficient ``h D`` and face flux ``F = q``; the release ``s``
is in the outfall's cell.

**One explicit step, one exchange.**  Forward Euler on the upwinded finite
volumes, at a step below the explicit limit ``h dx^2 / A_ii`` (the case check
refuses a larger one and says what the limit is).  Every macro-step each window
is cut from the global concentration and takes ONE step, its cut faces closed
by the global values outside it (`fv.assemble(..., cut="neighbour")`, the
Schwarz restriction with the outside coupling kept as a matrix ``C``); the
windows are then blended by the case's partition of unity.

That is style A -- overlapping windows, one exchange per step -- and for an
explicit scheme it is exact.  A cell's new value depends only on its own and
its four neighbours' old values, and every window cell's neighbours are in the
window or supplied by the exchange, so each window computes the full-domain
step on its own cells, and the blend of equal values is that value.  The
decomposed river is the full-domain river to round-off, with nothing to
iterate.  That is the plume's contrast with the wind farm, whose windows march
several sub-steps on one exchange and are therefore NOT the full domain (W346:
2-8% in farm power).

**One-way, nearly.**  Across a face with the flow running from P to Q, P's
value reaches Q through ``u + D/dx`` and Q's reaches P through ``D/dx`` alone:
a ratio ``1 / (1 + Pe)`` with the cell Peclet number ``Pe = u dx / D``.  The
run's notes give it per reach.

**What the showcase does not claim.**  First-order upwinding adds a mixing of
``u dx (1 - u dt/dx) / 2`` along the flow, comparable with the physical one at
these cells, so the plume's front is more smeared than ``D`` alone would make
it.  Both arms carry exactly that scheme, so the comparison between them is
unaffected; the reach values are showcase numbers, not a calibrated river.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import fv
from .. import geometry as geo
from ..checks import CheckSpec, exact, judge
from ..tiling import MaskTiling, RectangleTiling
from .conduction import window_cells

FAMILY = "transport-2d"
STYLE = "A"
ARMS = ("serial", "parallel", "full")
FIELD_LABEL = "concentration (mg/L)"
#: the outfall's cell is some fifty times the mixed river: drawn on a log scale
FIELD_SCALE = "log"
FIELD_DECADES = 3.0
LENGTH_UNIT = "m"
SERIES = {"outflow": "Pollutant leaving at the outlet | g/s"}

REGISTERED = "2026-09-29, before the transport family's first run"

CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        key="mass", title="Mass: released = held + carried out", kind="balance",
        tolerance=1e-9, registered=REGISTERED,
        measure=("|change in the pollutant the river holds - dt (release + what crosses "
                 "the edges)| over the mass released in the step, every step, worst arm"),
        why=("the explicit finite-volume step moves mass face by face, what leaves one "
             "cell entering the next, so the full domain closes to the round-off of "
             "summing the field; the decomposed arm is the same step to round-off (the "
             "next check). 1e-9 leaves about four orders above that round-off")),
    CheckSpec(
        key="reference", title="Concentration agrees with the full domain", kind="reference",
        tolerance=1e-10, registered=REGISTERED,
        measure=("max |c_decomposed - c_full| over the full domain's largest "
                 "concentration, at the last step"),
        why=("one explicit step on a window's cells is the full-domain step on those "
             "cells, so the arms differ only in the order of floating-point additions "
             "(the cut faces' rows, and the blend in the overlap): about one unit in the "
             "last place per step, which the monotone step never amplifies -- a few "
             "1e-13 after a few thousand steps")),
    CheckSpec(
        key="bitwise", title="Threaded equals serial, bit for bit", kind="control",
        tolerance=None, registered=REGISTERED,
        measure="np.array_equal on the concentration after every step",
        why=("each window's step reads only the previous step's field, and the blend adds "
             "the windows in their order, so the order the threads finish in cannot "
             "change a bit")),
)

_KIND = {"river-inlet": fv.INLET, "river-outlet": fv.OUTLET, "bank": fv.NO_FLUX}


# ---------------------------------------------------------------------------
# the case as finite volumes
# ---------------------------------------------------------------------------


def reach_properties(spec) -> tuple[np.ndarray, np.ndarray]:
    """(depth ``h``, mixing ``D``) per cell, ``(ny, nx)``, from each region's reach."""
    d = spec.domain
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    h = np.zeros((d.ny, d.nx))
    mix = np.zeros((d.ny, d.nx))
    for i, r in enumerate(spec.regions):
        m = spec.materials[r.material]
        h[owner == i] = float(m["depth"])
        mix[owner == i] = float(m["mixing"])
    return h, mix


def outfall_cell(spec) -> tuple[int, int]:
    """(column, row) of the cell the outfall releases into."""
    d = spec.domain
    i = int(np.floor(spec.physics.get("source_x") / d.dx))
    j = int(np.floor(spec.physics.get("source_y") / d.dx))
    return min(max(i, 0), d.nx - 1), min(max(j, 0), d.ny - 1)


def is_drawn(spec) -> bool:
    d = spec.domain
    return d.outline is not None or bool(d.holes)


def river_flow(spec):
    """A drawn river's flow (case file 0.4): the potential flow from its inlet edges
    to its outlet edges (`flow.py`), carrying ``q`` per metre of the inlets' width
    -- their width as drawn, not their staircase's.  No water crosses a bank."""
    from .. import flow as fl
    d = spec.domain
    bf, kinds = fl.case_faces(spec, "river-inlet", "river-outlet")
    scale = geo.staircase_scale(d, bf)
    width = float(sum(scale.get(n, 1.0) for n, k in zip(bf.edge.tolist(), kinds.tolist())
                      if k == "in")) * d.dx
    return fl.potential_flow(geo.domain_mask(d), bf, kinds, spec.physics.get("q") * width,
                             float(d.dx))


def field_from_case(spec) -> tuple[fv.Field, np.ndarray]:
    """The river as an `fv.Field` -- ``k = h D``, ``F = q`` through every x-face,
    the release in the outfall's cell -- and its capacity ``h`` per cell.

    A drawn river (case file 0.4) is its cells, with the solved flow of
    `river_flow` in place of the uniform one and each face the condition of its
    edge: an inlet, an outlet or a bank."""
    d = spec.domain
    h, mix = reach_properties(spec)
    q = spec.physics.get("q")
    src = np.zeros((d.ny, d.nx))
    i, j = outfall_cell(spec)
    src[j, i] = spec.physics.get("release") / (d.dx * d.dx)
    if is_drawn(spec):
        from .conduction import boundary_arrays
        act = geo.domain_mask(d)
        h = np.where(act, h, 0.0)
        bc, void = boundary_arrays(spec, _KIND)
        fl = river_flow(spec)
        f = fv.Field(d.nx, d.ny, float(d.dx), h * np.where(act, mix, 0.0), cap=h,
                     source=np.where(act, src, 0.0), fx=fl.fx, fy=fl.fy, bc=bc,
                     active=act, void=void)
        return f, h
    bc = {}
    for e in fv.EDGES:
        n = geo.edge_length(e, d.nx, d.ny)
        kd = np.zeros(n, dtype=np.int8)
        for b in spec.boundaries:
            if b.edge == e:
                kd[b.start:(n if b.stop is None else b.stop)] = _KIND[b.kind]
        bc[e] = (kd, np.zeros(n))                  # the river enters clean
    f = fv.Field(d.nx, d.ny, float(d.dx), h * mix, cap=h, source=src,
                 fx=np.full((d.ny, d.nx + 1), q), bc=bc)
    return f, h


def explicit_limit(spec) -> float:
    """The largest stable explicit step, ``min_i h_i dx^2 / A_ii``: the step at which
    a cell's own old value still has a non-negative weight in its new one.

    ``A_ii`` without assembling: the harmonic-mean conductance of each of the
    cell's interior faces, plus ``q dx`` for the one face the water leaves by
    (its +x face, interior or the outlet); the inlet and the banks add nothing.
    A test checks it against the assembled diagonal.  A drawn river's flow varies
    from face to face, so its limit is read off the assembled diagonal itself.
    """
    d = spec.domain
    if is_drawn(spec):
        f, h = field_from_case(spec)
        s = fv.assemble(f)
        return float(np.min(h.ravel()[s.idx] * d.dx * d.dx / s.A.diagonal()))
    h, mix = reach_properties(spec)
    k = h * mix
    diag = np.full((d.ny, d.nx), spec.physics.get("q") * d.dx)
    gx = fv.harmonic(k[:, :-1], k[:, 1:])
    diag[:, :-1] += gx
    diag[:, 1:] += gx
    gy = fv.harmonic(k[:-1, :], k[1:, :])
    diag[:-1, :] += gy
    diag[1:, :] += gy
    return float(np.min(h * d.dx * d.dx / diag))


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


@dataclass
class PlumeState:
    u: np.ndarray                 # concentration per cell, flat, g/m^3 (= mg/L)


class PlumeRun:
    family = FAMILY

    def __init__(self, spec, arms=ARMS, threads: int = 2):
        self.spec = spec
        self.f, h = field_from_case(spec)
        self.nx, self.ny, self.dx = self.f.nx, self.f.ny, self.f.dx
        self.dt = float(spec.run.macro_dt)
        self.limit = explicit_limit(spec)
        if self.dt > self.limit:                          # the case check refuses it first
            raise ValueError(f"the macro-step {self.dt:g} s is over the explicit step's "
                             f"stability limit {self.limit:.4g} s")
        self.h = h.ravel()
        self.q = float(spec.physics.get("q"))
        self.release = float(spec.physics.get("release"))
        #: a drawn river (case file 0.4): its cells only, the rest of the grid dry
        self.drawn = is_drawn(spec)
        self.void = self.f.void_cells()
        #: dt / (h dx^2): what turns a cell's net inflow into its change of concentration
        if self.drawn:
            self.coef = np.zeros(self.f.n)
            wet = self.f.cells()
            self.coef[wet] = self.dt / (self.h[wet] * self.dx * self.dx)
        else:
            self.coef = self.dt / (self.h * self.dx * self.dx)
        self.arms = tuple(a for a in ARMS if a in arms)
        self.threads = max(1, int(threads))
        self.full_sys = fv.assemble(self.f)
        self.windows = [(w.id, (w.x0, w.y0, w.nx, w.ny)) for w in spec.windows]
        if self.drawn or not geo.is_plain(spec):
            from .conduction import cells_of
            self.tiling = MaskTiling(geo.domain_mask(spec.domain), geo.window_masks(spec),
                                     spec.coupling.ramp_cells)
            self.systems = [fv.assemble(self.f, cells_of(spec, w)) for w in spec.windows]
        else:
            self.tiling = RectangleTiling(self.nx, self.ny, self.windows,
                                          spec.coupling.ramp_cells)
            self.systems = [fv.assemble(self.f, window_cells(self.nx, b))
                            for _n, b in self.windows]
        self.certificate = self.tiling.certify()
        self.coefs = [self.coef[s.idx] for s in self.systems]
        #: the outlet faces, for what leaves the river (a drawn outlet can be any edge)
        bc_cell, bc_kind, _v, _g, bc_in = fv.boundary_faces(self.f)
        out = bc_kind == fv.OUTLET
        self._out_cells, self._out_in = bc_cell[out], bc_in[out]
        self.pool = None
        if "parallel" in self.arms:
            self.pool = ThreadPoolExecutor(max_workers=min(self.threads, len(self.systems)),
                                           thread_name_prefix="wb-plume")

    # -- the arms -------------------------------------------------------------

    def initial(self, arm: str) -> PlumeState:
        return PlumeState(np.zeros(self.f.n))

    def _window(self, k: int, u: np.ndarray) -> np.ndarray:
        """Window ``k``'s one explicit step from the global field ``u``: its own cells
        from ``u``, and its cut faces closed by ``u``'s values outside it."""
        s = self.systems[k]
        uw = u[s.idx]
        r = s.b if s.C is None else s.b - s.C @ u
        new = uw + self.coefs[k] * (r - s.A @ uw)
        if isinstance(self.tiling, MaskTiling):
            return new                            # a window of any shape: its own cells
        _x0, _y0, w, h = self.windows[k][1]
        return new.reshape(h, w)

    def step(self, arm: str, s: PlumeState) -> PlumeState:
        u = s.u
        if arm == "full":
            sys_ = self.full_sys
            if not self.void.size:
                return PlumeState(u + self.coef * (sys_.b - sys_.A @ u))
            new = u.copy()                        # the dry grid outside a drawn river
            ui = u[sys_.idx]
            new[sys_.idx] = ui + self.coef[sys_.idx] * (sys_.b - sys_.A @ ui)
            return PlumeState(new)
        n = len(self.systems)
        if arm == "parallel" and self.pool is not None:
            locals_ = list(self.pool.map(self._window, range(n), [u] * n))
        else:
            locals_ = [self._window(k, u) for k in range(n)]
        return PlumeState(self.tiling.assemble(locals_).ravel())

    # -- instruments ---------------------------------------------------------

    def observe(self, arm: str, s: PlumeState, prev: PlumeState | None = None) -> dict:
        u = s.u
        u0 = prev.u if prev is not None else np.zeros_like(u)
        area = self.dx * self.dx
        held = float(np.sum(self.h * (u - u0))) * area
        #: the flows the step moved across the edges: the OLD field's, as it used them
        flows = fv.boundary_inflow(self.f, u0)
        into = self.release + float(sum(flows.values()))
        scale = max(self.dt * self.release, abs(held), 1e-300)
        if self.drawn:
            # every outlet face, wherever the drawn outlet is
            out = -float(np.sum(np.minimum(self._out_in, 0.0) * u0[self._out_cells]))
        else:
            out = 0.0 - flows["right"]
        return {"mass": float(np.sum(self.h * u)) * area, "outflow": out,
                "balance": abs(held - self.dt * into) / scale}

    def bitwise_equal(self, a: PlumeState, b: PlumeState) -> bool:
        return bool(np.array_equal(a.u, b.u))

    def field(self, s: PlumeState) -> np.ndarray:
        if not self.void.size:
            return s.u.reshape(self.ny, self.nx)
        u = s.u.copy()
        u[self.void] = np.nan                     # the dry grid: not drawn
        return u.reshape(self.ny, self.nx)

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
            metrics[arm] = {"mass_g": rows[-1]["mass"], "outflow_g_per_s": rows[-1]["outflow"],
                            "balance_max": max(r["balance"] for r in rows)}
        checks = []
        worst = max((m["balance_max"] for m in metrics.values()), default=None)
        checks.append(judge(CHECKS[0], worst, "; ".join(
            f"{a}: {m['balance_max']:.3g}" for a, m in metrics.items())))
        dec = "parallel" if "parallel" in metrics else ("serial" if "serial" in metrics
                                                        else None)
        if dec and "full" in metrics:
            peak = float(np.max(np.abs(states["full"].u)))
            dev = float(np.max(np.abs(states[dec].u - states["full"].u)))
            metrics[dec]["max_concentration_difference"] = dev
            if metrics["full"]["mass_g"]:
                metrics[dec]["mass_vs_full"] = ((metrics[dec]["mass_g"]
                                                 - metrics["full"]["mass_g"])
                                                / metrics["full"]["mass_g"])
            checks.append(judge(CHECKS[1], dev / peak if peak > 0 else dev,
                                f"max |dc| = {dev:.3g} mg/L against a peak of {peak:.4g} mg/L"))
        else:
            checks.append(judge(CHECKS[1], None, "needs a decomposed arm and the full domain"))
        if bitwise.get("steps"):
            ok = bitwise["first_difference"] is None
            checks.append(exact(CHECKS[2], ok, f"equal after all {bitwise['steps']} steps" if ok
                                else f"first differs after step {bitwise['first_difference']}"))
        else:
            checks.append(exact(CHECKS[2], None, "needs both decomposed arms"))
        return metrics, checks

    def arrival_seconds(self) -> float | None:
        """How long the water takes from the outfall to the outlet along its row:
        the sum of ``dx / u`` over the cells, ``u = q / h``.  None for a drawn river,
        whose water does not run along a row."""
        if self.drawn:
            return None
        i, j = outfall_cell(self.spec)
        h_row = self.h.reshape(self.ny, self.nx)[j, i:]
        return float(np.sum(self.dx * h_row / self.q))

    def speeds(self) -> np.ndarray:
        """The water's speed in each cell, from the flow through its four faces (the
        mean of each axis's two faces, over the depth); zero off the river."""
        fx, fy = self.f.fx, self.f.fy
        vx = 0.5 * (fx[:, :-1] + fx[:, 1:])
        vy = 0.5 * (fy[:-1, :] + fy[1:, :])
        h = self.h.reshape(self.ny, self.nx)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(h > 0, np.hypot(vx, vy) / np.where(h > 0, h, 1.0), 0.0)

    def notes(self, done: int) -> list[str]:
        out = [f"One explicit step of {self.dt:g} s per macro-step, one exchange, no "
               f"iteration: the stability limit here is {self.limit:.4g} s (min h dx^2 / "
               f"A_ii over the cells). Each window takes the full-domain step on its own "
               f"cells, so the decomposed river is the full-domain river to round-off."]
        if self.drawn:
            out.append("The river is drawn, so its flow is solved: the potential flow from "
                       "its inlets to its outlets (atlas/workbench/flow.py), carrying the "
                       "discharge per metre of the inlets' width. No water crosses a bank. "
                       "It is a potential flow: it has no viscosity and turns corners "
                       "without separating.")
        seen = []
        speed = self.speeds() if self.drawn else None
        owner = geo.region_owner(self.spec.regions, self.nx, self.ny)
        for k, r in enumerate(self.spec.regions):
            if r.material in seen:
                continue
            seen.append(r.material)
            m = self.spec.materials[r.material]
            if self.drawn:
                ks = [i for i, x in enumerate(self.spec.regions) if x.material == r.material]
                here = np.isin(owner, ks) & (self.h.reshape(self.ny, self.nx) > 0)
                if not here.any():
                    continue
                u = float(np.mean(speed[here]))
                how = f"so the water runs at {u:.3g} m/s on average"
            else:
                u = self.q / float(m["depth"])
                how = f"so the water runs at {u:.3g} m/s"
            pe = u * self.dx / float(m["mixing"])
            out.append(f"Reach {r.material}: depth {float(m['depth']):g} m, {how}; mixing "
                       f"{float(m['mixing']):g} m^2/s; cell Peclet number u dx / D = "
                       f"{pe:.3g}, so what runs upstream across a face is 1/(1 + Pe) = "
                       f"{1.0 / (1.0 + pe):.2g} of what runs down. Upwinding adds "
                       f"{u * self.dx * (1.0 - u * self.dt / self.dx) / 2.0:.3g} m^2/s of "
                       f"mixing along the flow.")
        t_arr = self.arrival_seconds()
        t_run = done * self.dt
        if t_arr is not None:
            out.append(f"The water takes {t_arr:.0f} s from the outfall to the outlet "
                       f"(macro-step {int(np.ceil(t_arr / self.dt))}); this run covered "
                       f"{t_run:.0f} s"
                       + (", so the plume has reached the outlet." if t_run >= t_arr else
                          ", so the plume has not reached the outlet yet and nothing has "
                          "left."))
        return out

    def describe(self) -> dict[str, Any]:
        i, j = outfall_cell(self.spec)
        return {"style": "A", "cells": int(self.f.cells().size), "windows": len(self.windows),
                "dt_s": self.dt, "explicit_limit_s": self.limit, "q_m2_per_s": self.q,
                "release_g_per_s": self.release, "outfall_cell": [i, j],
                "arrival_s": self.arrival_seconds(), "drawn": self.drawn,
                "partition_of_unity": self.certificate.as_dict()}


def build(spec, arms=ARMS, threads: int = 2) -> PlumeRun:
    return PlumeRun(spec, arms=arms, threads=threads)


def case_graph(spec):
    """The case for the compiler (`compile.py`): one agent per window, ADVEC seams
    carrying the pollutant as a passenger of the river's mass flux.

    A window's step is one explicit step -- no solve of any kind, a one-cell
    stencil, one sub-step per exchange -- so it declares no elliptic solve and an
    explicit time discretization.  Its response to the concentration on one
    port's faces is the pollutant flux its step would send INTO it through them:
    diffusion from that outside value and the upwinded advection, at the probe
    state (a river at the release's fully mixed concentration)."""
    from ..compile import fv_graph
    d = spec.domain
    f, h = field_from_case(spec)
    q = float(spec.physics.get("q"))
    release = float(spec.physics.get("release"))
    if is_drawn(spec):
        c_mixed = release / river_flow(spec).inflow   # the drawn inlets' discharge
    else:
        width = d.ny * d.dx
        c_mixed = release / (q * width)              # g/m^3 once mixed across the river
    u = q / float(np.min(h))
    flux = c_mixed * u                               # g/(m^2 s), the pollutant's flux
    water = 1000.0 * u                               # kg/(m^2 s), the river's mass flux
    enthalpy = 4180.0 * 300.0                        # J/kg, the water's, a scale only
    graph, _agents = fv_graph(
        spec, f, family=FAMILY, port_type="ADVEC",
        scales={"enthalpy": enthalpy, "mass_flux": water, "power_area": enthalpy * water,
                "pollutant_effort": c_mixed, "pollutant_flow": flux,
                "pollutant_power": c_mixed * flux},
        implicit=False, base=c_mixed, per_effort=False, passengers=("pollutant",),
        state=np.full(f.n, c_mixed),
        lambda_ref="the same explicit finite volumes on every cell, the full-domain arm",
        response_note="the pollutant flux the window's explicit step sends into it "
                      "through these faces, the trace the outside concentration",
        validity=lambda state=None, cond=None: float(spec.run.macro_dt)
        <= explicit_limit(spec))
    return graph


__all__ = ["FAMILY", "STYLE", "ARMS", "CHECKS", "PlumeRun", "PlumeState", "build",
           "field_from_case", "reach_properties", "outfall_cell", "explicit_limit"]
