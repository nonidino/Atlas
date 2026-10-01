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
from ..fast import FastBars
from ..tiling import MaskTiling, RectangleTiling
from .conduction import window_cells

FAMILY = "transport-2d"
STYLE = "A"
ARMS = ("serial", "parallel", "full")
#: **The Fast example's bars** (demo item 1.4), one per mechanism this family may
#: use, in the order they are tried (`fast.py`; demo-fast-examples-plan section 3).
#: Registered 2026-09-30, before the first timed run, and never loosened.
FAST: tuple[FastBars, ...] = (
    FastBars("P", 3.0, 1e-9, "concentration against the full domain, over the peak"),
    FastBars("M", 3.0, 1e-3, "concentration against the full domain, over the peak; the mass balance keeps its own 1e-9"),
)
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
    CheckSpec(
        key="continuity", title="The river's flow is continuous, cell by cell", kind="balance",
        tolerance=1e-12, registered=("2026-09-30, before the forked river's first run "
                                     "(demo-finish-plan section 2.4)"),
        measure=("the largest |sum of a cell's flows| over the river's discharge "
                 "(flow.py's residual), for a drawn river's solved flow"),
        why=("the potential flow is one sparse direct solve, and each cell's flows are its "
             "own potential's differences, so they sum to the solve's round-off; a fork "
             "adds outlets, not arithmetic. River-bend measured 1.4e-14")),
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


def dead_branches(spec, fl=None) -> list[np.ndarray]:
    """The parts of a drawn river where no water runs (`layout.dead_regions` of its
    potential): the far end of a branch that nothing lets out.  A potential flow
    decays into such a branch like ``exp(-pi x / w)``, so what it carries there is
    under a thousandth of the river's mean, and the pollutant only creeps in by
    mixing (2026-09-30: a forked river with one outlet)."""
    from .. import layout as lay
    fl = river_flow(spec) if fl is None else fl
    act = geo.domain_mask(spec.domain)
    return lay.dead_regions(lay.cell_speed(np.where(act, fl.phi, np.nan), act), act)


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


#: With several steps per exchange (``run.exchange_every``), every arm sets a
#: concentration under this (g/m^3) to zero once a macro-step (a value above it cannot
#: fall ~280 decades to a subnormal within one macro-step's steps).  The plume's far tail
#: decays into subnormal floats, whose arithmetic is many times slower: a 300-macro-step
#: march of the long river went from 80 to 269 ms a step as 4,268 cells turned
#: subnormal (seen, 2026-09-30), unevenly among the windows.  Flushing them is what
#: hardware flush-to-zero would do; numpy has no switch for it.
FLUSH = 1e-30


@dataclass
class PlumeState:
    u: np.ndarray                 # concentration per cell, flat, g/m^3 (= mg/L)
    #: several explicit steps per macro-step (``run.exchange_every``): the pollutant
    #: that entered through the boundary over the macro-step, g per metre of depth,
    #: summed step by step (None: one step, read from the old field as before)
    q_in: float | None = None


class PlumeRun:
    family = FAMILY

    def __init__(self, spec, arms=ARMS, threads: int = 2):
        self.spec = spec
        self.f, h = field_from_case(spec)
        self.nx, self.ny, self.dx = self.f.nx, self.f.ny, self.f.dx
        self.dt = float(spec.run.macro_dt)
        #: explicit steps per macro-step (demo item 1.4); the windows exchange once
        #: per macro-step on a halo this deep, so each one's own cells stay exact
        self.sub = max(1, int(getattr(spec.run, "exchange_every", 1)))
        self.dts = self.dt / self.sub
        self.limit = explicit_limit(spec)
        if self.dts > self.limit:                         # the case check refuses it first
            raise ValueError(f"the step {self.dts:g} s is over the explicit step's "
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
            self.coef[wet] = self.dts / (self.h[wet] * self.dx * self.dx)
        else:
            self.coef = self.dts / (self.h * self.dx * self.dx)
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
        self._halo_boxes = None
        if self.sub > 1:
            self._init_halo(spec)
        self.certificate = self.tiling.certify()
        self.coefs = [self.coef[s.idx] for s in self.systems]
        #: per window, the rows its cut faces reach (`C`'s nonzero rows) as a matrix of
        #: their own: `C @ u` walked all of a window's rows for a few dozen entries,
        #: a third of a window's step on a long river (demo item 1.4)
        self._ring = []
        for s in self.systems:
            if s.C is None:
                self._ring.append(None)
            else:
                rows = np.flatnonzero(np.diff(s.C.indptr))
                self._ring.append((rows, s.C[rows]))
        #: the outlet faces, for what leaves the river (a drawn outlet can be any edge)
        bc_cell, bc_kind, _v, _g, bc_in = fv.boundary_faces(self.f)
        out = bc_kind == fv.OUTLET
        self._out_cells, self._out_in = bc_cell[out], bc_in[out]
        #: a drawn river: which outlet each outlet face is (its boundary's index), for
        #: each outlet's share, and its flow's continuity residual.  `fv.boundary_faces`
        #: and `geometry.boundary_faces` list the same faces in the same order
        self._out_owner = None
        self.flow_residual = None
        if self.drawn:
            bf = geo.boundary_faces(spec.domain)
            owner = geo.face_conditions(spec.boundaries, bf, self.nx, self.ny)
            if owner.size == out.size:
                self._out_owner = owner[out]
            self.flow_residual = float(river_flow(spec).residual)
        self.pool = None
        if "parallel" in self.arms:
            self.pool = ThreadPoolExecutor(max_workers=min(self.threads, len(self.systems)),
                                           thread_name_prefix="wb-plume")

    def _init_halo(self, spec) -> None:
        """``exchange_every`` steps per exchange (demo item 1.4).  Each window computes
        on its own cells and every cell within that many steps of them (its halo), read
        from the global field at the exchange.  The explicit stencil reaches one cell a
        step, so what the halo's open edge gets wrong reaches no deeper than the steps,
        and the window's own cells come out as the full domain's.  The ledger: each
        boundary face is answered for by the first window holding its cell."""
        from scipy import ndimage
        k, n = self.sub, self.f.n
        own_cells, ext_cells = [], []
        if isinstance(self.tiling, RectangleTiling):
            boxes = []
            for _n, (x0, y0, w, h) in self.windows:
                ex0, ey0 = max(0, x0 - k), max(0, y0 - k)
                ex1, ey1 = min(self.nx, x0 + w + k), min(self.ny, y0 + h + k)
                ext = (ex0, ey0, ex1 - ex0, ey1 - ey0)
                boxes.append((ext, (x0 - ex0, y0 - ey0, w, h)))
                own_cells.append(window_cells(self.nx, (x0, y0, w, h)))
                ext_cells.append(window_cells(self.nx, ext))
            self._halo_boxes = boxes
        else:
            from .conduction import cells_of
            act = geo.domain_mask(spec.domain)
            cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
            for w in spec.windows:
                own = cells_of(spec, w)
                m = np.zeros(n, dtype=bool)
                m[own] = True
                m = ndimage.binary_dilation(m.reshape(self.ny, self.nx), structure=cross,
                                            iterations=k) & act
                own_cells.append(own)
                ext_cells.append(np.flatnonzero(m.ravel()))
        self.systems = [fv.assemble(self.f, e) for e in ext_cells]
        self._own_pos = [np.searchsorted(e, o) for e, o in zip(ext_cells, own_cells)]
        bc = fv.boundary_faces(self.f)
        taken = np.zeros(n, dtype=bool)
        self._bf = []
        for e, o in zip(ext_cells, own_cells):
            mine = np.zeros(n, dtype=bool)
            mine[o] = True
            mine &= ~taken
            taken |= mine
            sel = mine[bc[0]]
            c, a, b = fv.face_inflow_affine(self.f, bc[0][sel], *(x[sel] for x in bc[1:]))
            self._bf.append((np.searchsorted(e, c), a, b))
        self._bf_all = fv.face_inflow_affine(self.f, *bc)

    def _window_halo(self, k: int, u: np.ndarray) -> tuple[np.ndarray, float]:
        """Window ``k``'s ``exchange_every`` steps on its cells and its halo, from the
        global field ``u``: its own cells (as `_window` returns them), and what entered
        through the boundary faces it answers for, step by step."""
        s = self.systems[k]
        if self._halo_boxes is not None:
            (ex0, ey0, ew, eh), (ox, oy, w, h) = self._halo_boxes[k]
            ue = u.reshape(self.ny, self.nx)[ey0:ey0 + eh, ex0:ex0 + ew].ravel()
        else:
            ue = u[s.idx]
        ring = self._ring[k]
        if ring is None:
            r = s.b
        else:
            rows, c_rows = ring
            r = s.b.copy()
            r[rows] = s.b[rows] - c_rows @ u
        c, (pos, a, b) = self.coefs[k], self._bf[k]
        q = 0.0
        for _ in range(self.sub):
            if pos.size or b:
                q += self.dts * (float(a @ ue[pos]) + b)
            t = s.A @ ue
            np.subtract(r, t, out=t)
            np.multiply(c, t, out=t)
            np.add(ue, t, out=t)
            ue = t
        ue[np.abs(ue) < FLUSH] = 0.0                      # once a macro-step, every arm
        if self._halo_boxes is not None:
            return ue.reshape(eh, ew)[oy:oy + h, ox:ox + w], q
        return ue[self._own_pos[k]], q

    def _full_once(self, u: np.ndarray) -> np.ndarray:
        sys_ = self.full_sys
        if not self.void.size:
            return u + self.coef * (sys_.b - sys_.A @ u)
        new = u.copy()                            # the dry grid outside a drawn river
        ui = u[sys_.idx]
        new[sys_.idx] = ui + self.coef[sys_.idx] * (sys_.b - sys_.A @ ui)
        return new

    # -- the arms -------------------------------------------------------------

    def initial(self, arm: str) -> PlumeState:
        return PlumeState(np.zeros(self.f.n))

    def _window(self, k: int, u: np.ndarray) -> np.ndarray:
        """Window ``k``'s one explicit step from the global field ``u``: its own cells
        from ``u``, and its cut faces closed by ``u``'s values outside it."""
        s = self.systems[k]
        rect = not isinstance(self.tiling, MaskTiling)
        if rect:
            # a rectangle's cells in `window_cells`' order: a slice, not a gather
            x0, y0, w, h = self.windows[k][1]
            uw = u.reshape(self.ny, self.nx)[y0:y0 + h, x0:x0 + w].ravel()
        else:
            uw = u[s.idx]
        ring = self._ring[k]
        if ring is None:
            r = s.b
        else:
            # `b - C u` on the rows the cut faces reach; every other row's is `b - 0`
            rows, c_rows = ring
            r = s.b.copy()
            r[rows] = s.b[rows] - c_rows @ u
        # `uw + coef (r - A uw)`, the same operations in the same order, in place
        new = s.A @ uw
        np.subtract(r, new, out=new)
        np.multiply(self.coefs[k], new, out=new)
        np.add(uw, new, out=new)
        if not rect:
            return new                            # a window of any shape: its own cells
        return new.reshape(h, w)

    def step(self, arm: str, s: PlumeState) -> PlumeState:
        u = s.u
        if arm == "full":
            if self.sub == 1:
                return PlumeState(self._full_once(u))
            q = 0.0
            c, a, b = self._bf_all
            for _ in range(self.sub):
                q += self.dts * (float(a @ u[c]) + b)
                u = self._full_once(u)
            u[np.abs(u) < FLUSH] = 0.0                    # once a macro-step, every arm
            return PlumeState(u, q_in=q)
        n = len(self.systems)
        if self.sub > 1:
            win = self._window_halo
        else:
            def win(k, u_):
                return self._window(k, u_), 0.0
        threaded = arm == "parallel" and self.pool is not None
        if isinstance(self.tiling, RectangleTiling):
            # each window writes the cells it alone covers from its own thread, and
            # the overlaps are blended in window order after: `assemble`'s value on
            # every cell, its larger part no longer serial (demo item 1.4)
            out = np.zeros((self.ny, self.nx))

            def one(k):
                loc, q_ = win(k, u)
                self.tiling.write_own(out, k, loc)
                return loc, q_
            res = list(self.pool.map(one, range(n))) if threaded else [one(k)
                                                                        for k in range(n)]
            self.tiling.add_shared(out, [r_[0] for r_ in res])
            field_ = out.ravel()
        else:
            res = (list(self.pool.map(win, range(n), [u] * n)) if threaded
                   else [win(k, u) for k in range(n)])
            field_ = self.tiling.assemble([r_[0] for r_ in res]).ravel()
        return PlumeState(field_, q_in=None if self.sub == 1 else
                          float(sum(r_[1] for r_ in res)))

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
        if s.q_in is not None:
            # several steps per macro-step: what entered, summed step by step
            into = self.release + s.q_in / self.dt
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
        if self.flow_residual is None:
            checks.append(judge(CHECKS[3], None, "the plain river's flow is uniform along x, "
                                                 "so it is continuous by construction"))
        else:
            checks.append(judge(CHECKS[3], self.flow_residual,
                                f"largest |sum of a cell's flows| = {self.flow_residual:.3g} "
                                f"of the discharge"))
        #: a drawn river with several outlets: each one's share (2026-09-30, the fork)
        self._shares = None
        for arm in metrics:
            shares = self.outlet_shares(states[arm].u)
            if shares:
                metrics[arm]["outlets"] = shares
                if arm == "full" or self._shares is None:
                    self._shares = shares
        return metrics, checks

    def outlet_shares(self, u: np.ndarray) -> dict[str, dict[str, float]] | None:
        """Each outlet's share of the water leaving and of the pollutant leaving, for
        the concentration ``u``: by boundary id, the water's share from the solved flow
        (a potential flow splits by the channels' shapes alone), the pollutant's in g/s
        and as a share.  None for a plain river, which has one outlet."""
        if self._out_owner is None or not self._out_in.size:
            return None
        water = -np.minimum(self._out_in, 0.0)                # what leaves, face by face
        poll = water * u[self._out_cells]
        total_w, total_p = float(np.sum(water)), float(np.sum(poll))
        out = {}
        for b in dict.fromkeys(self._out_owner.tolist()):
            sel = self._out_owner == b
            bid = self.spec.boundaries[b].id if b >= 0 else "?"
            w, p = float(np.sum(water[sel])), float(np.sum(poll[sel]))
            out[bid] = {"edge": self.spec.boundaries[b].edge if b >= 0 else "",
                        "water_share": w / total_w if total_w > 0 else None,
                        "pollutant_g_per_s": p,
                        "pollutant_share": p / total_p if total_p > 0 else None}
        return out

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
        if self.sub == 1:
            out = [f"One explicit step of {self.dt:g} s per macro-step, one exchange, no "
                   f"iteration: the stability limit here is {self.limit:.4g} s (min h "
                   f"dx^2 / A_ii over the cells). Each window takes the full-domain step "
                   f"on its own cells, so the decomposed river is the full-domain river "
                   f"to round-off."]
        else:
            out = [f"{self.sub} explicit steps of {self.dts:g} s per macro-step and one "
                   f"exchange: the stability limit here is {self.limit:.4g} s. Each "
                   f"window steps its own cells and a halo {self.sub} cells deep read "
                   f"from the exchange, and the stencil reaches a cell a step, so its own "
                   f"cells are the full domain's to round-off and stay in the "
                   f"processor's cache for all {self.sub} steps."]
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
        shares = getattr(self, "_shares", None)
        if shares and len(shares) > 1:
            parts = []
            for bid, s in shares.items():
                pw = "-" if s["water_share"] is None else f"{100.0 * s['water_share']:.1f}%"
                pp = ("nothing yet" if s["pollutant_share"] is None else
                      f"{100.0 * s['pollutant_share']:.1f}%")
                parts.append(f"outlet {bid} ({s['edge']}) takes {pw} of the water and {pp} "
                             f"of the pollutant leaving ({s['pollutant_g_per_s']:.4g} g/s)")
            out.append("The river leaves by " + str(len(shares)) + " outlets: "
                       + "; ".join(parts) + ". A potential flow splits by the channels' "
                       "shapes alone, which is a stated guess, as the inlet on the left is.")
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
                "dt_s": self.dt, "exchange_every": self.sub, "step_s": self.dts,
                "explicit_limit_s": self.limit, "q_m2_per_s": self.q,
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
    #: the shallowest water on the river's own cells: a drawn river's depth is zero
    #: past its banks, and the compile divided by it (seen: river-bend's compile)
    u = q / float(np.min(h[h > 0]))
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
        substeps=int(spec.run.exchange_every),
        validity=lambda state=None, cond=None: float(spec.run.macro_dt)
        / int(spec.run.exchange_every) <= explicit_limit(spec))
    return graph


__all__ = ["FAMILY", "STYLE", "ARMS", "CHECKS", "PlumeRun", "PlumeState", "build",
           "field_from_case", "reach_properties", "outfall_cell", "explicit_limit",
           "river_flow", "dead_branches"]
