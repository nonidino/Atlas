"""The wind-farm family, marched from a case file: style A of the showcase plan.

Overlapping windows, one exchange per macro-step: cut the global state into
windows, step every window (`WindowNS` with its elliptic part exposed), blend
them by the partition of unity, project the blend once on the whole domain, and
hold the freestream band.  This is `scripts/w346_rotor_count_speed.py`'s `Rung`
lifted into the workbench, with one generalization: **any rectangles**, from
`tiling.RectangleTiling`, instead of CS-7's regular tiling.

The three arms, as W346 names them:

``serial``    W346's ``E``: the windows stepped as one batch per window shape.
``parallel``  W346's ``Ep<T>``: the same windows split across ``T`` threads.
              Every chunk is forced to the whole batch's sub-step count
              (`_Umax`), so the result is **bit for bit** the serial arm's; the
              run checks that at every macro-step.
``full``      W346's ``F``: `RectangularNS` on the undivided domain, the same
              discretization, as built.

**What is W346's and what is new.**  The forcing, the band, the blend, the
projection and the three step functions are transcriptions of W346's (and of
`w100_scaling_ladder._forcing` and ``_band``), in the same order of operations,
so on an example case the serial arm is W346's ``E`` to the bit and the full
arm is its ``F`` to the bit (`tests/test_workbench_runner.py` checks both
against W346's own `Rung`).  New: windows of several sizes (stepped one shape
group at a time, all at the global sub-step count), rotors of any diameter
(the disk's swept width is its diameter; W346's were all 1 D), and the
freestream speed and viscosity read from the case.

**The outer boundary is the solver's, and it does not travel.**  The band
holds the inlet and both laterals at ``(U, 0)`` and the global projection
tapers the outflow (`wake_array._extend`), which is why the family's
boundaries are fixed in the case file.
"""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from .. import geometry as geo
from ..checks import CheckSpec, exact, judge
from ..fast import FastBars
from ..tiling import RectangleTiling

FAMILY = "incompressible-2d"
STYLE = "A"
ARMS = ("serial", "parallel", "full")
#: **The Fast example's bars** (demo item 1.4), one per mechanism this family may
#: use, in the order they are tried (`fast.py`; demo-fast-examples-plan section 3).
#: Registered 2026-09-30, before the first timed run, and never loosened.
FAST: tuple[FastBars, ...] = (
    FastBars("P", 3.0, 0.03, "farm power against the full domain, relative"),
)
#: what the page draws and plots
FIELD_LABEL = "streamwise velocity u / U"
LENGTH_UNIT = "D"
#: key -> "title | y-axis label"
SERIES = {"power": "Farm power per macro-step | sum of the disks' P = C_T' A U_d^3 / 2"}

#: The freestream band, in cells: `w100_scaling_ladder.BAND`, W93's number.
BAND = 8
#: The disk reads its inflow this far upstream of its plane, in the physics'
#: length unit (D): `w100_scaling_ladder._forcing`'s 0.25.
INFLOW_OFFSET = 0.25
#: Padding of the global projection, in cells: `wake_array.PAD_CELLS`.
PAD = 128
#: The window solver's CFL number and transmission, as every measurement ran.
CFL = 0.4
TRANSMISSION = "dirichlet"

# ---------------------------------------------------------------------------
# the checks, registered 2026-09-28 before the runner had marched anything
# ---------------------------------------------------------------------------

REGISTERED = "2026-09-28, before the workbench runner's first march"

CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        key="mass", title="Mass: incompressibility closes", kind="balance",
        tolerance=1e-9, registered=REGISTERED,
        measure=("the largest divergence, times dx / U, of the field each arm's own step "
                 "produces every macro-step, before the freestream band is re-imposed, "
                 "measured in the operator that enforced it: the full domain's own wide "
                 "centred difference (the solver's last_div), and for a projected "
                 "assembly the global projection's own spectral operator on the padded "
                 "domain"),
        why=("both are projections, so what is left is round-off: the full-domain "
             "solver's Neumann mean-mismatch vanishes because the band holds equal "
             "velocities on opposite faces of the ring, and a spectral projection leaves "
             "~1e-15. 1e-9 leaves six orders for accumulation. An assembly that is only "
             "a blend has nothing enforcing incompressibility after it, and fails this "
             "(L6/C2, W100). The band itself -- (U, 0) re-imposed in 8-cell strips after "
             "every step, W93's outer boundary -- is not divergence-free at its inner "
             "edge, identically in every arm, and is not what this measures")),
    CheckSpec(
        key="power", title="Farm power agrees with the full domain", kind="reference",
        tolerance=0.25, registered="2026-09-26, W346's prediction B5, before W346 ran",
        measure=("|P_decomposed - P_full| / P_full, farm power averaged over the last five "
                 "macro-steps (or all, if fewer)"),
        why=("W346's own registered bound for the composed column against the monolith; "
             "it measured 2.3-8.3% at 40 macro-steps from 1 to 21 rotors")),
    CheckSpec(
        key="bitwise", title="Threaded equals serial, bit for bit", kind="control",
        tolerance=None, registered="2026-09-26, W346's prediction B1, before W346 ran",
        measure="np.array_equal on both velocity components after every macro-step",
        why=("the threaded arm is the serial arm's arithmetic split across threads; any "
             "difference is a defect in the split, not a scheme difference")),
)


# ---------------------------------------------------------------------------
# the solvers
# ---------------------------------------------------------------------------


def _wa():
    from atlas.cases import wake_array as wa
    return wa


def _sl():
    from atlas.cases import scaling_ladder as sl
    return sl


class _Umax:
    """W346's backend proxy: ``amax`` returns a fixed value.

    ``WindowNS.step_batch`` reads its sub-step count off
    ``b.amax(b.hypot(u, v))`` of the batch it is handed -- the class's only call
    to ``amax`` -- so a chunk of the batch would otherwise pick its own count.
    Handing every chunk (and every shape group) the whole tiling's maximum makes
    them all take the same sub-steps, which is what makes the arms bitwise.
    """

    def __init__(self, inner, value):
        self._inner, self._value = inner, value

    def amax(self, _x):
        return self._value

    def __getattr__(self, name):
        return getattr(self._inner, name)


@lru_cache(maxsize=1)
def exposed_class():
    """`RectangularNS` with its projection removed: W100's exposed window, any shape.

    `window_ns._no_projection_class` does the same to the square `WindowNS`; a
    window of the case may be any rectangle, so the rectangular class is the
    base.  Without the projection the Poisson symbol is never read, so the two
    are the same arithmetic on a square window (checked against W346's ``E``).
    """
    base = _sl()._rect_class()

    class ExposedRectangularNS(base):
        def _project(self, u, v):
            return u, v

    return ExposedRectangularNS


def window_solver(nu: float, dx: float, nx: int, ny: int, exposed: bool):
    cls = exposed_class() if exposed else _sl()._rect_class()
    return cls(nu=nu, length=nx * dx, n=nx, ny=ny, cfl=CFL, transmission=TRANSMISSION)


@lru_cache(maxsize=2)
def penalized_class(exposed: bool):
    """The window solver (exposed or not) with the solid held at rest: a drawn
    domain's cells outside it (case file 0.4).

    Brinkman penalization in its stiff limit: every sub-step, just before the
    projection (the solver's ``_project``, once per sub-step), the velocity in the
    solid is set to zero -- ``u <- chi u`` with ``chi`` 1 in the fluid and 0 in the
    solid -- and the projection then makes the field divergence-free again.  So a
    drawn edge is a no-slip wall, first order in the sub-step: the projection lets
    a little velocity back into the solid, and the next sub-step removes it.  The
    mask is ``fluid``, ``[B, ny, nx]`` or ``[1, ny, nx]``, set before each
    ``step_batch``; with no mask the class is its base, to the bit."""
    base = exposed_class() if exposed else _sl()._rect_class()

    class PenalizedNS(base):
        fluid = None

        def _project(self, u, v):
            if self.fluid is not None:
                u, v = u * self.fluid, v * self.fluid
            return super()._project(u, v)

    return PenalizedNS


def penalized_solver(nu: float, dx: float, nx: int, ny: int, exposed: bool):
    return penalized_class(exposed)(nu=nu, length=nx * dx, n=nx, ny=ny, cfl=CFL,
                                    transmission=TRANSMISSION)


class MaskedBoxes:
    """Windows of any shape on a domain of any shape (case file 0.4), for a batch
    solver that marches rectangles: each window marches its bounding box, and the
    windows are blended by `tiling.MaskTiling`'s partition of unity on their own
    cells (zero weight on the rest of the box, which only gives the window room).

    ``boxes`` keep the case's order, and that order is the order of every sum."""

    def __init__(self, spec, ramp: int):
        from ..tiling import MaskTiling
        d = spec.domain
        self.nx, self.ny = d.nx, d.ny
        self.fluid = geo.domain_mask(d)
        masks = geo.window_masks(spec)
        self.mask_tiling = MaskTiling(self.fluid, masks, ramp)
        self.names = [n for n, _m in masks]
        self.boxes: list[tuple[int, int, int, int]] = []
        #: where each window's weighted cells sit in its box, flat
        self.sel: list[np.ndarray] = []
        #: a window's box is the window AS DRAWN, solid included, and the penalization
        #: holds that solid at rest inside it, as the full domain's does.  Boxed on its
        #: fluid cells alone, a window over the ground never held the ground: its
        #: box's edge stood where the solid began, and one window over the whole
        #: drawn farm was not the full domain (max |dv| 0.61 U after one macro-step)
        drawn = [geo.window_mask(w, d.nx, d.ny) for w in spec.windows]
        for m, idx in zip(drawn, self.mask_tiling.idx):
            ys, xs = np.nonzero(m)
            x0, y0 = int(xs.min()), int(ys.min())
            w, h = int(xs.max()) + 1 - x0, int(ys.max()) + 1 - y0
            self.boxes.append((x0, y0, w, h))
            j, i = idx // self.nx, idx % self.nx
            self.sel.append((j - y0) * w + (i - x0))
        self.chi = self.mask_tiling.chi

    @property
    def n_windows(self) -> int:
        return len(self.boxes)

    def shapes(self) -> dict[tuple[int, int], list[int]]:
        out: dict[tuple[int, int], list[int]] = {}
        for k, (_x0, _y0, w, h) in enumerate(self.boxes):
            out.setdefault((h, w), []).append(k)
        return out

    def cut_one(self, f: np.ndarray, k: int) -> np.ndarray:
        x0, y0, w, h = self.boxes[k]
        return f[y0:y0 + h, x0:x0 + w]

    def cut(self, f: np.ndarray, idx) -> np.ndarray:
        return np.stack([self.cut_one(f, k) for k in idx])

    def assemble(self, locals_) -> np.ndarray:
        out = np.zeros(self.nx * self.ny)
        for k, idx in enumerate(self.mask_tiling.idx):
            out[idx] += self.chi[k] * np.asarray(locals_[k]).ravel()[self.sel[k]]
        return out.reshape(self.ny, self.nx)

    def certify(self):
        return self.mask_tiling.certify()

    def partition_of_unity(self):
        return self.mask_tiling.partition_of_unity()


def full_solver(nu: float, dx: float, nx: int, ny: int):
    """The undivided domain: `scaling_ladder.reference_monolith` (cached) at the
    measured cell, otherwise the same class built at this cell."""
    sl = _sl()
    if dx == _wa().DX:
        return sl.reference_monolith(nx, ny, nu)
    return sl.RectangularNS(nu=nu, length=nx * dx, n=nx, ny=ny, cfl=CFL,
                            transmission=TRANSMISSION)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rotor:
    rotor_id: str
    x_plane: float
    y_centre: float
    diameter: float


@dataclass
class FarmState:
    u: np.ndarray
    v: np.ndarray
    #: each rotor's disk-averaged inflow, read at the START of the step that made this
    rec: dict[str, float] = field(default_factory=dict)
    #: what `observe` needs and `step` already had, by reference: never computed
    #: inside the timer.  ``locals_`` are the windows' own outputs, which `observe`
    #: re-blends (the band mutates the handed-on field in place, so the pre-band
    #: blend is not kept as such)
    substeps: int = 0
    locals_: tuple[list, list] | None = None
    solver_div: float | None = None


class WindFarmRun:
    """One committed case, prepared to march.

    Built once per Run, from a copy of the case, so an edit made while it
    marches cannot reach it (rebuild on commit, not on drag).
    """

    family = FAMILY
    style = STYLE

    def __init__(self, spec, arms=ARMS, threads: int = 4):
        d = spec.domain
        self.spec = spec
        self.arms = tuple(a for a in ARMS if a in arms)
        self.nx, self.ny, self.dx = d.nx, d.ny, float(d.dx)
        self.nu = float(spec.physics.nu)
        self.u_inf = float(spec.physics.u_inf)
        self.dt = float(spec.run.macro_dt)
        self.projected = spec.coupling.assembly == "projected"
        self.exposed = spec.coupling.elliptic == "exposed"
        self.threads = max(1, int(threads))
        #: a drawn domain or drawn windows (case file 0.4): windows march their boxes
        #: and blend by the mask partition of unity; a drawn domain's solid is held at
        #: rest in every solver (`penalized_class`).  A plain case is W346's, to the bit.
        self.plain = geo.is_plain(spec)
        d = spec.domain
        self.drawn = d.outline is not None or bool(d.holes)
        self.fluid = geo.domain_mask(d) if self.drawn else None
        if self.plain:
            self.tiling = RectangleTiling(self.nx, self.ny,
                                          [(w.id, (w.x0, w.y0, w.nx, w.ny))
                                           for w in spec.windows], spec.coupling.ramp_cells)
        else:
            self.tiling = MaskedBoxes(spec, spec.coupling.ramp_cells)
        self.certificate = self.tiling.certify()
        self.rotors = [Rotor(v.id, float(v.x), float(v.y), float(v.diameter))
                       for v in spec.devices]
        self.x_c = (np.arange(self.nx) + 0.5) * self.dx
        self.y_c = (np.arange(self.ny) + 0.5) * self.dx
        self._disk = _load_disk()

        make = ((lambda nx_, ny_: penalized_solver(self.nu, self.dx, nx_, ny_, self.exposed))
                if self.drawn else
                (lambda nx_, ny_: window_solver(self.nu, self.dx, nx_, ny_, self.exposed)))
        self.groups = self.tiling.shapes()
        self.serial = {shape: make(shape[1], shape[0]) for shape in self.groups}
        self.chunks: list[tuple[tuple[int, int], np.ndarray]] = []
        for shape, idx in self.groups.items():
            for c in np.array_split(np.arange(len(idx)), self.threads):
                if c.size:
                    self.chunks.append((shape, c))
        self.chunk_solvers = [make(shape[1], shape[0]) for shape, _c in self.chunks]
        #: each shape group's solid mask, window by window ([B, h, w]), for the
        #: penalized solvers
        self.solid_masks = ({shape: self.tiling.cut(self.fluid.astype(float), idx)
                             for shape, idx in self.groups.items()} if self.drawn else {})
        self.pool = (ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="wb-farm")
                     if "parallel" in self.arms else None)
        self.full = None
        if "full" in self.arms:
            if self.drawn:
                self.full = penalized_solver(self.nu, self.dx, self.nx, self.ny, exposed=False)
                self.full.fluid = self.fluid.astype(float)[None]
            else:
                self.full = full_solver(self.nu, self.dx, self.nx, self.ny)

    # -- W346's shared pieces, transcribed ---------------------------------

    def forcing(self, u: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """The disks' body force on the whole domain, and each disk's inflow
        (`w100_scaling_ladder._forcing`, with each rotor's own diameter)."""
        dk = self._disk
        fx = np.zeros((self.ny, self.nx))
        rec: dict[str, float] = {}
        for rot in self.rotors:
            rows = np.abs(self.y_c - rot.y_centre) <= 0.5 * rot.diameter + 1e-9
            i_up = int(np.argmin(np.abs(self.x_c - (rot.x_plane - INFLOW_OFFSET))))
            ud = float(np.mean(u[rows, i_up]))
            rec[rot.rotor_id] = ud
            thick = max(ud, 0.05) * self.dt
            disk = dk.ActuatorDisk(area=rot.diameter, thickness=thick)
            st = disk(max(ud, 1e-3))
            fx = fx + disk.body_force_field(self.x_c, self.y_c, self.dx, self.dx, st.thrust,
                                            x0=rot.x_plane - 0.5 * thick,
                                            y0=rot.y_centre - 0.5 * rot.diameter)
        return fx, rec

    def band(self, u: np.ndarray, v: np.ndarray):
        """Inlet and both laterals held at the freestream, and the outlet column
        (`w100_scaling_ladder._band`, then W346's ``Rung.band``)."""
        b, U = BAND, self.u_inf
        u[:, :b] = U
        v[:, :b] = 0.0
        u[:b, :] = U
        v[:b, :] = 0.0
        u[-b:, :] = U
        v[-b:, :] = 0.0
        u[:, -1] = U
        v[:, -1] = 0.0
        if self.fluid is not None:
            # a drawn domain: the band holds the freestream only where the fluid meets
            # the grid's edges, and the solid is at rest
            u *= self.fluid
            v *= self.fluid
        return u, v

    def project(self, u: np.ndarray, v: np.ndarray):
        """One global Leray projection of the assembled field: `wake_array.project_assembled`
        with the case's freestream (the same call when it is 1).  On a drawn domain the
        solid is set to rest first, as every solver does before its projection."""
        if self.fluid is not None:
            u, v = u * self.fluid, v * self.fluid
        uf, vf = _wa().transport_and_project(u - self.u_inf, v, dt=0.0, u_inf=0.0,
                                             project=True)
        return self.u_inf + uf, vf

    def power(self, rec: dict[str, float]) -> float:
        """Farm power: the disks' own ``P = 1/2 C_T' A <U_d>^3``, summed."""
        dk = self._disk
        by_id = {r.rotor_id: r for r in self.rotors}
        return float(sum(dk.ActuatorDisk(area=by_id[k].diameter)(max(float(ud), 1e-6)).power
                         for k, ud in rec.items())) if rec else 0.0

    # -- the arms -----------------------------------------------------------

    def initial(self, arm: str) -> FarmState:
        u = np.full((self.ny, self.nx), self.u_inf)
        if self.fluid is not None:
            u *= self.fluid                        # a drawn domain's solid starts at rest
        return FarmState(u, np.zeros((self.ny, self.nx)))

    def step(self, arm: str, s: FarmState) -> FarmState:
        """One macro-step of one arm.  This is all the runner times."""
        if arm == "full":
            return self._step_full(s)
        return self._step_decomposed(s, parallel=(arm == "parallel"))

    def _step_full(self, s: FarmState) -> FarmState:
        fx, rec = self.forcing(s.u)
        uu, vv = self.full.step_batch(s.u[None], s.v[None], self.dt, bc0=None,
                                      force=(fx[None], np.zeros_like(fx)[None]))
        u1, v1 = self.band(uu[0], vv[0])
        return FarmState(u1, v1, rec, int(self.full.last_substeps),
                         solver_div=float(self.full.last_div))

    def _step_decomposed(self, s: FarmState, parallel: bool) -> FarmState:
        t = self.tiling
        fx, rec = self.forcing(s.u)
        cut = {shape: (t.cut(s.u, idx), t.cut(s.v, idx), t.cut(fx, idx))
               for shape, idx in self.groups.items()}
        umax = max(float(np.max(np.hypot(us, vs))) for us, vs, _f in cut.values())
        out_u: list[Any] = [None] * t.n_windows
        out_v: list[Any] = [None] * t.n_windows
        subs = 0
        if not parallel:
            for shape, idx in self.groups.items():
                us, vs, fs = cut[shape]
                sol = self.serial[shape]
                sol.b = _Umax(getattr(sol.b, "_inner", sol.b), umax)
                if self.drawn:
                    sol.fluid = self.solid_masks[shape]
                a, b = sol.step_batch(us, vs, self.dt, bc0=None, force=(fs, np.zeros_like(fs)))
                subs = int(sol.last_substeps)
                for j, k in enumerate(idx):
                    out_u[k], out_v[k] = a[j], b[j]
        else:
            res: list[Any] = [None] * len(self.chunks)

            def work(i):
                shape, pos = self.chunks[i]
                us, vs, fs = cut[shape]
                sol = self.chunk_solvers[i]
                sol.b = _Umax(getattr(sol.b, "_inner", sol.b), umax)
                if self.drawn:
                    sol.fluid = self.solid_masks[shape][pos]
                a, b = sol.step_batch(us[pos], vs[pos], self.dt, bc0=None,
                                      force=(fs[pos], np.zeros_like(fs[pos])))
                res[i] = (a, b, int(sol.last_substeps))

            list(self.pool.map(work, range(len(self.chunks))))
            for (shape, pos), (a, b, n) in zip(self.chunks, res):
                idx = self.groups[shape]
                for j, p in enumerate(pos):
                    out_u[idx[p]], out_v[idx[p]] = a[j], b[j]
                subs = n
        au, av = t.assemble(out_u), t.assemble(out_v)
        if self.projected:
            u1, v1 = self.project(au, av)
        else:
            u1, v1 = au, av
        u1, v1 = self.band(u1, v1)
        return FarmState(u1, v1, rec, subs, locals_=(out_u, out_v))

    # -- what is read after the timer ---------------------------------------

    def observe(self, arm: str, s: FarmState, prev: FarmState | None = None) -> dict[str, float]:
        """Per-step diagnostics, computed OUTSIDE the timed step."""
        out = {"power": self.power(s.rec), "substeps": float(s.substeps),
               "mass": self.mass_measure(arm, s)}
        return out

    def mass_measure(self, arm: str, s: FarmState) -> float:
        """The divergence of the handed-on field, times dx / U, in the enforcing operator."""
        if arm == "full":
            return float(s.solver_div) * self.dx / self.u_inf
        au, av = self.tiling.assemble(s.locals_[0]), self.tiling.assemble(s.locals_[1])
        if self.fluid is not None:
            au, av = au * self.fluid, av * self.fluid     # what the projection was given
        if self.projected:
            return spectral_divergence_after_projection(au, av, self.u_inf) / self.u_inf
        # nothing enforces incompressibility after a blend: measure the blend in the
        # windows' own operator, which is what the check then refuses
        return wide_divergence_max(au, av) / self.u_inf

    def bitwise_equal(self, a: FarmState, b: FarmState) -> bool:
        return bool(np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v))

    def field(self, s: FarmState) -> np.ndarray:
        """What the page draws: the streamwise velocity (not drawn in a solid)."""
        if self.fluid is None:
            return s.u
        return np.where(self.fluid, s.u, np.nan)

    def field_label(self) -> str:
        return "streamwise velocity u / U"

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    # -- the end of a run ---------------------------------------------------

    def compare(self, states: dict[str, FarmState], history: dict[str, list[dict]],
                bitwise: dict[str, Any]) -> tuple[dict[str, Any], list]:
        """Metrics per arm and the checks, from the final states and the per-step record."""
        metrics: dict[str, Any] = {}
        for arm, rows in history.items():
            if not rows:
                continue
            p = [r["power"] for r in rows]
            tail = p[-5:]
            metrics[arm] = {"farm_power": float(np.mean(tail)),
                            "farm_power_steps_averaged": len(tail),
                            "substeps_last": rows[-1]["substeps"],
                            "mass_max": float(max(r["mass"] for r in rows))}
        dec = "parallel" if "parallel" in metrics else ("serial" if "serial" in metrics else None)
        checks = []
        # mass: the worst arm.  A serial state shown equal to the bit to the parallel
        # one was measured once, as the parallel arm (the runner says so per row)
        masses = {a: m["mass_max"] for a, m in metrics.items()}
        worst = max(masses.values()) if masses else None
        shared = bool(history.get("serial")) and all(
            r.get("observed_as") == "parallel" for r in history["serial"])
        checks.append(judge(CHECKS[0], worst, "; ".join(
            (f"{a}: the parallel arm's state, bit for bit" if a == "serial" and shared
             else f"{a}: {v:.3g}") for a, v in masses.items())))
        if dec and "full" in metrics:
            pf = metrics["full"]["farm_power"]
            # every decomposed arm against the full domain (both components of the
            # velocity, the measure outcome C1 quotes); the check reads `dec`
            for arm in (a for a in ("serial", "parallel") if a in metrics and a in states):
                pa = metrics[arm]["farm_power"]
                metrics[arm]["farm_power_vs_full"] = (pa - pf) / pf if pf else None
                du = states[arm].u - states["full"].u
                dv = states[arm].v - states["full"].v
                sq = du * du + dv * dv
                metrics[arm]["rms_velocity_difference"] = float(
                    np.sqrt(np.mean(sq if self.fluid is None else sq[self.fluid])))
            pd = metrics[dec]["farm_power"]
            rel = abs(pd - pf) / pf if pf else None
            checks.append(judge(CHECKS[1], rel, f"{dec} {pd:.4f} against full {pf:.4f}"))
        else:
            checks.append(judge(CHECKS[1], None, "needs a decomposed arm and the full domain"))
        if "serial" in history and "parallel" in history and bitwise.get("steps"):
            ok = bitwise["first_difference"] is None
            detail = (f"equal after all {bitwise['steps']} macro-steps" if ok else
                      f"first differs after macro-step {bitwise['first_difference']}")
            checks.append(exact(CHECKS[2], ok, detail))
        else:
            checks.append(exact(CHECKS[2], None, "needs both decomposed arms"))
        return metrics, checks

    def wake_steps(self) -> float | None:
        """Macro-steps a wake needs, at the freestream speed, to reach the nearest
        rotor downstream of another whose disk it overlaps; None with no such pair."""
        gaps = [b.x_plane - a.x_plane for a in self.rotors for b in self.rotors
                if b.x_plane > a.x_plane
                and abs(b.y_centre - a.y_centre) < 0.5 * (a.diameter + b.diameter)]
        return min(gaps) / (self.u_inf * self.dt) if gaps else None

    def notes(self, done: int) -> list[str]:
        out = []
        need = self.wake_steps()
        if need is not None and done < need:
            out.append(f"The farm-power comparison covers the start-up only: at the freestream "
                       f"speed a wake needs at least {need:.1f} macro-steps to reach the next "
                       f"rotor downstream, and this run marched {done}. For the developed "
                       f"wake, W346's 40-step comparison (decomposition-speed-by-rotor-count "
                       f"section 3) remains the record.")
        if not (self.exposed and self.projected):
            out.append("This arrangement is not the measured one (elliptic part exposed, "
                       "projected assembly); W100 found every other combination leaves its "
                       "band within 80 macro-steps at six windows.")
        return out

    def describe(self) -> dict[str, Any]:
        return {"windows": self.tiling.n_windows, "rotors": len(self.rotors),
                "cells": (self.nx * self.ny if self.fluid is None
                          else int(self.fluid.sum())), "shape_groups": len(self.groups),
                "drawn": self.drawn,
                "chunks": len(self.chunks), "threads": self.threads,
                "assembly": "projected" if self.projected else "blend",
                "elliptic": "exposed" if self.exposed else "embedded",
                "partition_of_unity": self.certificate.as_dict()}


# ---------------------------------------------------------------------------
# instruments (never inside a timed step)
# ---------------------------------------------------------------------------


def _load_disk():
    _wa().load_reference()                                # registers the private package
    return importlib.import_module("atlas_windfarm_reference.disk")


def wide_divergence_max(u: np.ndarray, v: np.ndarray) -> float:
    """``max |du/dx + dv/dy| * dx`` on the interior, with `WindowNS`'s own wide
    centred difference and even-extension edges -- the operator its projection
    inverts, measured where `WindowNS.last_div` measures it.  Per CELL, so the
    caller divides by U only."""
    c = 0.5
    du = np.empty_like(u)
    du[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * c
    du[:, 0] = (u[:, 1] - u[:, 0]) * c
    du[:, -1] = (u[:, -1] - u[:, -2]) * c
    dv = np.empty_like(v)
    dv[1:-1, :] = (v[2:, :] - v[:-2, :]) * c
    dv[0, :] = (v[1, :] - v[0, :]) * c
    dv[-1, :] = (v[-1, :] - v[-2, :]) * c
    d = du + dv
    return float(np.abs(d[2:-2, 2:-2]).max())


def spectral_divergence_after_projection(au: np.ndarray, av: np.ndarray,
                                         u_inf: float) -> float:
    """The divergence the global projection leaves, in ITS operator on ITS padded domain.

    `wake_array.transport_and_project` at ``dt = 0``, re-run on the assembled
    field and stopped after the projection: the same padding, the same
    wavenumbers with the Nyquist mode zeroed for the derivative, then
    ``max |ifft(k_x u + k_y v)|``.  A projection leaves round-off here; a blend
    that was never projected would not reach this function.

    Returned per CELL (the divergence times `wake_array.DX`, the cell the
    wavenumbers are built on), so the caller divides by U only.  The projection
    is the same whatever cell the wavenumbers assume -- rescaling every k by one
    factor leaves ``I - k k^T / |k|^2`` unchanged -- so this is the projection
    the arm applied at any cell size.
    """
    wa = _wa()
    ny, nx = au.shape
    big_u, big_v = wa._extend(au - u_inf, PAD), wa._extend(av, PAD)
    nxp = nx + PAD
    kx = 2.0 * np.pi * np.fft.fftfreq(nxp, d=(nxp * wa.DX) / nxp)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=(ny * wa.DX) / ny)
    uh, vh = np.fft.fft2(big_u), np.fft.fft2(big_v)
    kxd, kyd = kx.copy(), ky.copy()
    if nxp % 2 == 0:
        kxd[nxp // 2] = 0.0
    if ny % 2 == 0:
        kyd[ny // 2] = 0.0
    KX, KY = kxd[None, :], kyd[:, None]
    k2 = np.where((KX ** 2 + KY ** 2) == 0.0, 1.0, KX ** 2 + KY ** 2)
    div = KX * uh + KY * vh
    uh, vh = uh - KX * div / k2, vh - KY * div / k2
    after = np.real(np.fft.ifft2(KX * uh + KY * vh))
    return float(np.abs(after).max()) * wa.DX


#: the learned case's two arms (demo item 1.5, `learned_case`); every other farm
#: keeps the runner's own names for its three
from ..learned_case import ARM_LABELS                                  # noqa: E402


def available_arms(spec) -> tuple[tuple[str, ...], dict[str, str]]:
    """The arms a farm case can run: the three, or the learned case's four."""
    from .. import learned_case
    if learned_case.is_learned(spec):
        return learned_case.available_arms(spec)
    return ARMS, {}


def build(spec, arms=ARMS, threads: int = 4):
    from .. import learned_case
    if learned_case.is_learned(spec):
        return learned_case.LearnedCase(spec, arms=arms, threads=threads)
    return WindFarmRun(spec, arms=arms, threads=threads)


def ladder_rung(spec):
    """The scaling ladder's rung whose tiling and rotors this case is, or None."""
    from atlas.cases import scaling_ladder as sl
    wa = _wa()
    d = spec.domain
    wins = [(w.x0, w.y0, w.nx, w.ny) for w in spec.windows]
    for cols in range(1, 13):
        for rows in range(1, 9):
            try:
                r = sl.rung(cols, rows)
            except Exception:                       # not a rung the ladder builds
                continue
            t = r.tiling
            if (t.nx, t.ny) != (d.nx, d.ny) or len(t.offsets) != len(wins):
                continue
            if [(ox, oy, wa.N, wa.N) for ox, oy in t.offsets] != wins:
                continue
            rot = sorted((round(x.x_plane, 9), round(x.y_centre, 9)) for x in t.rotors)
            dev = sorted((round(v.x, 9), round(v.y, 9)) for v in spec.devices)
            if rot == dev:
                return r
    return None


def case_graph(spec):
    """The case for the compiler: `wake_array`'s own graph, which is what W346's
    arithmetic was compiled as -- fluid windows with the reference `WindowNS`
    whose pressure solve is EXPOSED (the workbench's arrangement; an embedded
    choice is declared as embedded, and R10 answers it), the actuator disks as
    lumped agents, the projected assembly when the case asks for it, and its own
    cross-points declared (W162).  The probe linearizes about the freestream.

    That graph declares a regular tiling of one window size (`ArrayTiling`, a
    scaling-ladder rung), so a case whose windows are not one is refused with
    that reason rather than compiled as something else."""
    from atlas.capability import EllipticSubsolve
    from atlas.cases import scaling_ladder as sl

    from ..compile import CompileRefused
    d = spec.domain
    if d.outline is not None or d.holes:
        # the rung's graph is its full rectangle of fluid; compiled, a drawn farm was
        # judged as the plain one, its terrain nowhere in the graph (seen: farm-hill)
        raise CompileRefused(
            "the wind-farm family's graph is wake_array's, a full rectangle of fluid; a "
            "drawn domain's solid, held at rest by penalization, is not in it, and a "
            "drawn farm has no declared graph yet")
    r = ladder_rung(spec)
    if r is None:
        raise CompileRefused(
            "the wind-farm family's graph is wake_array's, which declares a regular "
            "tiling of 128-cell windows and its rotors (a scaling-ladder rung); these "
            "windows or rotors are not one, and an irregular tiling has no declared graph "
            "yet")
    t = r.tiling
    u = np.full((t.ny, t.nx), float(spec.physics.u_inf))
    exposed = spec.coupling.elliptic == "exposed"
    graph, _experts = sl.build(
        u, np.zeros_like(u), r, kind="reference_exposed" if exposed else "reference",
        dt=float(spec.run.macro_dt), nu=float(spec.physics.nu),
        elliptic=EllipticSubsolve.EXPOSED if exposed else EllipticSubsolve.EMBEDDED,
        assembly_projection=spec.coupling.assembly == "projected")
    return graph


__all__ = ["FAMILY", "STYLE", "ARMS", "ARM_LABELS", "CHECKS", "WindFarmRun", "FarmState",
           "Rotor", "available_arms", "build", "window_solver", "full_solver", "exposed_class",
           "wide_divergence_max", "spectral_divergence_after_projection"]
