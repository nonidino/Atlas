"""The union on the body-fitted car: the radiator core and the turbine in the opened duct.

PoC 3, Tier 64.  [[poc3-racelab-car-union]].

Tier 63 cut an inlet and an outlet into the solid car's shell and measured the
duct's flow on solid walls, forward and weak (W281); the user accepted that
flow for now and asked for the next steps.  This module puts the two devices
back into that duct and joins them to the rest of the car exactly as RaceLab's
porous column does (`racelab.RaceRollout`), so that the SAME functions read the
march -- `vehicle_march.receiver_balances` takes a `vehicle_march.UnionMarch`
built from this module's trace with no change:

  J1  the radiator core in the flow: a streamwise momentum sink ``1/2 K u^2``
      and a conductance ``UA(u) = UA_RAD (u / u_ref)^0.8`` that follows the air
      (`integration_union.CoreRadiator`, ``u_ref`` held at the state it is built
      at);
  J3  the recovery turbine: `vehicle_march.operating_point`, solved across the
      powertrain at the inflow the duct delivers, its thrust applied to the fluid
      and ``T <u_ring>`` claimed by the shaft;
  J2  the machine's winding heat ``I^2 R_MGU p_ref`` into the cooled block
      (`integration_union.MountedBlock`), the coolant circuit sub-cycled every
      `vehicle_march.N_FLUID_PER_COOLANT` fluid steps (`vehicle_march._coolant_step`).

Every join is LAGGED: re-read once per fluid step from the state at its start,
which is the porous column's lagged cadence with one exchange per step.

What had to be decided again on body-fitted grids, and why
----------------------------------------------------------

1. **A device is a continuous force density, not a set of cells.**  On the
   porous column a device was an exact-overlap strip on ONE lattice, so
   ``sum(f dA) = -T`` to floating point.  Here several grids overlap in the duct
   -- the background in its middle, each wall's body grid within 0.10 of it --
   and every grid solves the momentum equation where it has discretisation
   points, so there is no single lattice to be exact on.  The force is written
   once, as a function of position,

       f_x(x, y) = -T phi(x - x_p) chi(y) / W,

   with ``phi`` a raised cosine of half-width `HALF_WIDTH_CELLS` (integral one)
   and ``chi`` the open band of the duct, and each grid samples it at its own
   points.  The identity survives where it can: on the background, whose columns
   are uniform, the raised cosine's samples sum to one for ANY offset of the
   plane (its shifted copies are a partition of unity), and on the quadrature
   lattice `DuctDevice.strip_points` uses to read the force's work it holds to
   round-off (`DuctDevices.quadrature_identity`).

2. **The device is the duct's OPEN width, 29 cells and not 32.**  The solids
   rule thickened DUCT_LO and DUCT_UP by half a panel each into the duct.
   Decision 2 (W199) is that a device is its host's size, so the disk's area and
   the machine's scale follow the open width, and decision 8 (W222) sizes the
   machine for the inflow the host delivers: `racelab.machine_for_host(u_host,
   scale=width)`, both similarities at once.

3. **The ring is read 8 cells upstream of the plane**, on 32 points at the
   midpoints of 32 equal parts of the open band -- `disk.disk_average`'s rule
   (read where the force has not yet slowed the flow) at the porous column's
   distance (7.5 cells) and cell count.  In a SOLID duct the rule matters less
   than it did: continuity holds the plane's mean velocity at the ring's.

4. **The force's work on the fluid is read by quadrature**, ``-T times the
   force-weighted mean of u over the strip``, on a lattice of `QUAD_PER_CELL`
   points per cell interpolated from the composite by a fixed sparse matrix
   (`probe_matrix`, `car_solids.probe`'s rule for which grid holds a point).

5. **The envelopes are consulted every step and recorded, not enforced.**  The
   machine's generating band is narrow -- measured by `machine_band`, from 2.5%
   below its sizing inflow to about 12% above it -- and the duct's flow is still
   falling, so a release that is not settled for the machine is outside the band
   by design.  `racelab.RaceRollout`'s three states are kept: True inside, False
   declined, None not consultable.

Not here: the tight coupling and the composition-error arms, the graph
declaration and its compile, a learned expert, the demo.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from . import car_solids as CS
from . import cooling_loop as CL
from . import ground_effect as GE
from . import integration_union as IU
from . import overset as OV
from . import powertrain as PT
from . import racelab as RL
from . import vehicle_march as VM

__all__ = [
    "DEVICE_X_CELLS", "RING_UPSTREAM_CELLS", "HALF_WIDTH_CELLS", "QUAD_PER_CELL", "RING_SAMPLES",
    "duct_band", "open_band", "probe_matrix", "DuctDevice", "DuctDevices", "car_devices",
    "machine_band", "CarUnion", "TRACE_KEYS", "save_state", "load_state",
]

#: The device planes, in cells: the porous column's (Tier 59's derived layout),
#: where Tiers 62 and 63 read the duct's flow.
DEVICE_X_CELLS = {"RAD": 280.0, "ROTOR": 392.0}
JOIN_OF = {"RAD": "J1", "ROTOR": "J3"}
LABEL_OF = {"RAD": "the radiator core", "ROTOR": "the recovery turbine"}
#: How far upstream of its plane a device reads its inflow, in cells.
RING_UPSTREAM_CELLS = 8.0
#: The force's streamwise profile: a raised cosine this many cells either side.
#: The porous column's strip was one cell thick by a declared floor; a raised
#: cosine two cells either side has about that width at half height and no jump
#: for a curved grid's points to straddle.
HALF_WIDTH_CELLS = 2.0
#: Quadrature points per cell, each way, for the force's work.
QUAD_PER_CELL = 4
#: Ring points: `integration_union.DEVICE_CELLS`, which `CoreRadiator` and
#: `wake_array.RotorDisk` both insist on.
RING_SAMPLES = IU.DEVICE_CELLS

#: The trace keys `racelab.march` writes, so `vehicle_march.receiver_balances`
#: reads this march unchanged.
TRACE_KEYS = ("u_rotor", "u_core", "ua", "induction", "omega", "current", "thrust_rotor",
              "shaft_power", "thrust_core", "core_flow_power", "q_machine", "load", "drag",
              "u_max", "power_on_the_fluid", "t_wall", "power_rotor", "power_core",
              "u_at_the_rotor_plane")


# ---------------------------------------------------------------------------
# where the devices sit
# ---------------------------------------------------------------------------


def duct_band(geometry: dict | None = None) -> dict[str, float]:
    """The duct's open band and its length, in cells, from the solids rule."""
    doc = RL.load_geometry() if geometry is None else geometry
    rule = CS.solids_rule(doc)
    objs, _flat = RL.car_bodies(geometry=doc)
    plates = {o.body.body_id: o.body for o in objs if isinstance(o, RL.PlateBody)}
    lo, up = plates["DUCT_LO"], plates["DUCT_UP"]
    for b in (lo, up):
        if abs(float(b.y_te) - float(b.y_le)) > 1e-9:
            raise ValueError(f"{b.body_id} is not horizontal; a device band needs a straight duct")
    t_lo = float(rule["thickness_cells"].get("DUCT_LO", rule["panel_thickness_cells"]))
    t_up = float(rule["thickness_cells"].get("DUCT_UP", rule["panel_thickness_cells"]))
    return {"y_lo": float(lo.y_le) + 0.5 * t_lo, "y_hi": float(up.y_le) - 0.5 * t_up,
            "x_from": max(float(lo.x_le), float(up.x_le)), "x_to": min(float(lo.x_te), float(up.x_te)),
            "wall_thickness": [t_lo, t_up]}


def open_band(solids: Sequence[CS.Solid], x_cells: float, y_from: float, y_to: float,
              h: float = GE.DX) -> list[tuple[float, float]]:
    """The fluid intervals, in cells, on the vertical line ``x = x_cells`` between
    ``y_from`` and ``y_to``, from the solids' own outlines."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
    line = LineString([(x_cells, y_from), (x_cells, y_to)])
    body = unary_union([Polygon(np.asarray(s.outline) / h) for s in solids])
    rest = line.difference(body)
    parts = [rest] if rest.geom_type == "LineString" else list(getattr(rest, "geoms", []))
    out = []
    for p in parts:
        if p.is_empty:
            continue
        ys = [c[1] for c in p.coords]
        out.append((float(min(ys)), float(max(ys))))
    return sorted(out)


def probe_matrix_masked(ov, px, py) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """The probe operator, with the unreachable points REPORTED rather than fatal.

    Row k interpolates the composite at point k from the first body grid (in
    `ov.comps` order) that holds it with a usable stencil, else from the
    background.  Returns ``(M, missing, source)``: ``missing[k]`` is True where
    no grid could hold the point -- its row of ``M`` is empty, so ``M @ vec`` is
    zero there and the caller must not read it -- and ``source[k]`` indexes
    ``list(ov.comps) + [ov.bg]``, or is -1 where ``missing``.

    **Why this is the primitive and `probe_matrix` is the wrapper (Tier 66).**
    A device's strip must reach every one of its points or the join is being
    applied somewhere the composite cannot evaluate, so `probe_matrix` raises.
    A PICTURE of the composite is the same operator asked over a raster that
    covers the whole domain, and a raster necessarily lands inside the bodies,
    where there is no fluid and nothing to draw.  Both need the same donor
    search in the same grid order; if they were written twice they could come
    to disagree, and then the picture would stop being a picture of the solve.
    """
    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    m = px.size
    rows, cols, vals = [], [], []
    open_ = np.ones(m, dtype=bool)
    source = np.full(m, -1, dtype=np.int64)
    for gi, c in enumerate(list(ov.comps) + [ov.bg]):
        sel = np.nonzero(open_)[0]
        if not sel.size:
            break
        res = (ov._cartesian_donors(px[sel], py[sel]) if c is ov.bg
               else ov._curvilinear_donors(c, px[sel], py[sel]))
        good = res["ok"]
        if np.any(good):
            idx = ov.index[c.name].ravel()[res["flat"][good]]
            if np.any(idx < 0):                                  # pragma: no cover
                raise OV.OversetError("a probe stencil reaches a point with no unknown")
            q = sel[good]
            rows.append(np.repeat(q, idx.shape[1]))
            cols.append(idx.ravel())
            vals.append(res["weights"][good].ravel())
            open_[q] = False
            source[q] = gi
    if rows:
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(m, ov.n_unknowns))
    else:                                                        # pragma: no cover
        M = sp.csr_matrix((m, ov.n_unknowns))
    return M, open_, source


def probe_matrix(ov, px, py) -> sp.csr_matrix:
    """`car_solids.probe` as a sparse matrix: row k interpolates the composite at
    point k from the first body grid (in `ov.comps` order) that holds it with a
    usable stencil, else from the background.  Built once; applied every step.

    Every point must be reachable; a point that is not is an error here, because
    the caller is a device's strip and not a picture.  `probe_matrix_masked` is
    the same search without that requirement."""
    M, missing, _source = probe_matrix_masked(ov, px, py)
    if np.any(missing):
        raise OV.OversetError(f"{int(missing.sum())} probe point(s) have no usable stencil on any grid")
    return M


@dataclass(frozen=True)
class DuctDevice:
    """One device's plane, band and profile, in TILING units."""

    key: str
    x_plane: float
    y_lo: float
    y_hi: float
    ring_upstream: float
    half_width: float

    @property
    def join(self) -> str:
        return JOIN_OF[self.key]

    @property
    def label(self) -> str:
        return LABEL_OF[self.key]

    @property
    def width(self) -> float:
        return self.y_hi - self.y_lo

    @property
    def x_ring(self) -> float:
        return self.x_plane - self.ring_upstream

    def profile_x(self, x) -> np.ndarray:
        """The raised cosine, integral one."""
        s = (np.asarray(x, dtype=float) - self.x_plane) / self.half_width
        return np.where(np.abs(s) < 1.0, (1.0 + np.cos(np.pi * s)) / (2.0 * self.half_width), 0.0)

    def shape(self, x, y) -> np.ndarray:
        """The force density per unit thrust: integral one over the plane."""
        y = np.asarray(y, dtype=float)
        inside = (y >= self.y_lo) & (y <= self.y_hi)
        return self.profile_x(x) * inside / self.width

    def line_points(self, x: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``n`` points across the open band at ``x``, at the midpoints of ``n``
        equal parts, and each one's share of the width."""
        ys = self.y_lo + (np.arange(n) + 0.5) * self.width / n
        return np.full(n, float(x)), ys, np.full(n, self.width / n)

    def ring_points(self, n: int = RING_SAMPLES):
        return self.line_points(self.x_ring, n)

    def strip_points(self, h: float = GE.DX, q: int = QUAD_PER_CELL):
        """A midpoint lattice over the strip and each point's weight, so that
        ``sum(w u)`` is the force-weighted mean of ``u``: ``sum(w) = 1``."""
        nx = int(round(2.0 * self.half_width / h)) * q
        ny = int(round(self.width / h)) * q
        dx = 2.0 * self.half_width / nx
        dy = self.width / ny
        xs = self.x_plane - self.half_width + (np.arange(nx) + 0.5) * dx
        ys = self.y_lo + (np.arange(ny) + 0.5) * dy
        X, Y = np.meshgrid(xs, ys)
        w = self.shape(X, Y) * dx * dy
        return X.ravel(), Y.ravel(), w.ravel()


def car_devices(geometry: dict | None = None, h: float = GE.DX) -> list[DuctDevice]:
    band = duct_band(geometry)
    out = []
    for key, xc in DEVICE_X_CELLS.items():
        if not (band["x_from"] + RING_UPSTREAM_CELLS + HALF_WIDTH_CELLS < xc
                < band["x_to"] - HALF_WIDTH_CELLS):
            raise ValueError(f"{key}'s plane at {xc} cells leaves the duct {band['x_from']}..{band['x_to']} "
                             "with its ring or its strip")
        out.append(DuctDevice(key, xc * h, band["y_lo"] * h, band["y_hi"] * h,
                              RING_UPSTREAM_CELLS * h, HALF_WIDTH_CELLS * h))
    return out


class DuctDevices:
    """The devices on one composite: the force, the rings, the work."""

    def __init__(self, ov, devices: Sequence[DuctDevice], h: float = GE.DX) -> None:
        t0 = time.perf_counter()
        self.ov = ov
        self.h = float(h)
        self.devices = {d.key: d for d in devices}
        #: the thrust each device currently applies
        self.amplitude = {k: 0.0 for k in self.devices}
        self.Q_ring, self.w_ring, self.Q_strip, self.w_strip = {}, {}, {}, {}
        for k, d in self.devices.items():
            x, y, w = d.ring_points()
            self.Q_ring[k], self.w_ring[k] = probe_matrix(ov, x, y), w
            x, y, w = d.strip_points(h)
            self.Q_strip[k], self.w_strip[k] = probe_matrix(ov, x, y), w
        self._lines: dict[tuple[str, float], tuple[sp.csr_matrix, np.ndarray]] = {}
        self.calls = 0
        self.build_s = time.perf_counter() - t0

    def forcing(self, x, y, t):
        """`overset_ns.OversetFlow`'s ``forcing(x, y, t)``."""
        fx = np.zeros_like(x)
        for k, d in self.devices.items():
            T = self.amplitude[k]
            if T:
                near = np.abs(x - d.x_plane) < d.half_width
                fx[near] -= T * d.shape(x[near], y[near])
        self.calls += 1
        return fx, np.zeros_like(x)

    def ring(self, U: np.ndarray, key: str) -> np.ndarray:
        return self.Q_ring[key] @ U

    def plane_mean(self, U: np.ndarray, key: str) -> float:
        """The force-weighted mean of ``U`` over the strip."""
        return float(self.w_strip[key] @ (self.Q_strip[key] @ U))

    def power(self, U: np.ndarray, key: str) -> float:
        """The work the device's force does on the fluid per unit time (negative:
        the force opposes the flow)."""
        return -float(self.amplitude[key]) * self.plane_mean(U, key)

    def line_mean(self, F: np.ndarray, key: str, x_cells_from_plane: float, n: int | None = None) -> float:
        """The mean of a composite field across the open band, that many cells
        from the device's plane."""
        d = self.devices[key]
        n = int(round(d.width / self.h)) * QUAD_PER_CELL if n is None else int(n)
        ck = (key, float(x_cells_from_plane))
        if ck not in self._lines:
            x, y, w = d.line_points(d.x_plane + x_cells_from_plane * self.h, n)
            self._lines[ck] = (probe_matrix(self.ov, x, y), w / d.width)
        Q, w = self._lines[ck]
        return float(w @ (Q @ F))

    def quadrature_identity(self) -> dict[str, float]:
        """``sum(w) - 1`` per device: the strip lattice integrates the force exactly."""
        return {k: float(self.w_strip[k].sum() - 1.0) for k in self.devices}

    def discrete_integrals(self) -> dict[str, dict[str, float]]:
        """Per device and per grid, ``sum(shape dA)`` over that grid's
        discretisation points inside the duct, where a grid's cell area is known:
        exactly the background's.  A body grid's is reported as its point count."""
        out: dict[str, dict[str, float]] = {}
        bg = self.ov.bg
        S = self.ov.status[bg.name]
        for k, d in self.devices.items():
            s = d.shape(bg.X, bg.Y)
            live = S == OV.DISC
            rec = {"background_all_points": float(s.sum() * bg.h * bg.h),
                   "background_discretisation_points": float(s[live].sum() * bg.h * bg.h)}
            for c in self.ov.comps:
                sc = d.shape(c.x, c.y)
                Sc = self.ov.status[c.name]
                rec[f"points_forced_on_{c.name}"] = int(np.count_nonzero(sc[Sc == OV.DISC]))
            out[k] = rec
        return out


# ---------------------------------------------------------------------------
# the machine's band
# ---------------------------------------------------------------------------


def _operating(u: float, u_host: float, width: float) -> dict[str, Any]:
    els = RL.machine_for_host(u_host, scale=width)
    ring = np.full(RING_SAMPLES, float(u))
    res, _e, disk = VM.operating_point(ring, width=width, scale=width, elements=els)
    mgu = els["MGU"]
    return {"induction": float(res.induction), "omega": float(res.omega), "current": float(res.current),
            "rotor_valid": bool(res.rotor_valid), "mgu_valid": bool(mgu.validity(np.full(1, res.omega))),
            "thrust": float(disk.thrust), "power": float(disk.power)}


def machine_band(u_host: float, width: float, tol: float = 1e-6) -> dict[str, Any]:
    """The inflow ratios ``u / u_host`` over which the machine sized for
    ``u_host`` generates AND the disk has an operating point, by bisection from
    the sizing point (where both hold by the similarity's construction)."""
    def ok(r):
        try:
            p = _operating(r * u_host, u_host, width)
        except ValueError:
            # `disk.clamp_induction` refuses an induction past its validity limit
            return False
        return p["rotor_valid"] and p["mgu_valid"]

    at = _operating(u_host, u_host, width)
    if not (at["rotor_valid"] and at["mgu_valid"]):
        raise RuntimeError("the machine has no generating operating point at its own sizing inflow")

    def edge(inside, outside):
        while abs(outside - inside) > tol:
            mid = 0.5 * (inside + outside)
            if ok(mid):
                inside = mid
            else:
                outside = mid
        return inside

    lo_out = 0.5
    hi_out = 2.0
    if ok(lo_out) or ok(hi_out):                                 # pragma: no cover
        raise RuntimeError("the band's bracket did not bracket it")
    return {"u_host": float(u_host), "width": float(width), "ratio_lo": edge(1.0, lo_out),
            "ratio_hi": edge(1.0, hi_out), "at_the_sizing_point": at}


# ---------------------------------------------------------------------------
# the joins, marched
# ---------------------------------------------------------------------------


class CarUnion:
    """The three joins on a body-fitted march, LAGGED.

    ``flow`` is a `car_solids.car_flow`; the devices are attached as its
    ``forcing`` at construction.  `step` advances the flow one step; from
    ``t_on`` on it first re-reads the joins from the state at the start of the
    step (`refresh`), then steps, then records the work, the envelope and the
    coolant circuit.  Before ``t_on`` the devices apply nothing and nothing is
    recorded.
    """

    def __init__(self, flow, devices: DuctDevices, u_host: float, *, solids: Sequence[str] = (),
                 t_on: float = 0.0, p_ref: float | None = None,
                 n_per_coolant: int = VM.N_FLUID_PER_COOLANT) -> None:
        self.flow = flow
        self.dev = devices
        self.width = devices.devices["ROTOR"].width
        if abs(devices.devices["RAD"].width - self.width) > 1e-12:
            raise ValueError("the two devices span different bands")
        self.u_host = float(u_host)
        self.elements = RL.machine_for_host(self.u_host, scale=self.width)
        self.p_ref = (IU.calibrated_p_ref(PT.MachineAgent())[0] if p_ref is None else float(p_ref))
        self.solids = tuple(solids)
        self.t_on = float(t_on)
        self.n_per_coolant = int(n_per_coolant)
        self.state = VM.JoinState()
        self._core: IU.CoreRadiator | None = None
        self.block = IU.MountedBlock(q_machine=0.0)
        self.loop = CL.LoopSolve(block=self.block)
        self.t_in = CL.T_COOLANT_0
        self.coolant: list[dict] = []
        self.trace: dict[str, list] = {k: [] for k in ("t",) + TRACE_KEYS + (
            "power_rotor_start", "power_rotor_end", "power_core_start", "power_core_end",
            "u_at_the_core_plane", "mgu_valid", "rotor_valid", "rad_valid", "fluid_valid",
            "back_emf", "union_s")}
        self.steps_on = 0
        self.outside_steps = 0
        self.outside_first: dict | None = None
        self.envelope: dict | None = None
        self.last_forces: dict[str, dict[str, float]] = {}
        flow.forcing = devices.forcing

    # -- the joins ----------------------------------------------------------

    def refresh(self, U: np.ndarray) -> None:
        """`racelab.RaceRollout.refresh` on the body-fitted duct."""
        s = self.state
        ring = self.dev.ring(U, "ROTOR")
        if not np.mean(ring) > 0.0:
            raise RuntimeError(f"the turbine's ring reads a mean inflow of {np.mean(ring):.6g}; "
                               "the operating point is defined for forward flow only")
        res, _els, disk = VM.operating_point(ring, width=self.width, scale=self.width,
                                             elements=self.elements)
        s.u_rotor = float(np.mean(ring))
        s.induction = float(res.induction)
        s.omega = float(res.omega)
        s.current = float(res.current)
        s.thrust_rotor = float(disk.thrust)
        s.shaft_power = float(disk.power)
        s.rotor_valid = bool(res.rotor_valid)
        #: the machine's own windings, as `vehicle_march.UnionRollout.refresh` has it
        s.q_machine = float(s.current ** 2 * self.elements["MGU"].resistance * self.p_ref)
        ring_c = self.dev.ring(U, "RAD")
        s.u_core = float(np.mean(ring_c))
        if self._core is None:
            #: built ONCE, at the first refresh, so ``u_air_ref`` is held
            self._core = IU.CoreRadiator(u_air=ring_c)
            self.loop.legs["RAD"] = self._core
        core = self._core
        core.u_air = ring_c
        s.ua = float(core.ua)
        tau = core.traction(ring_c)
        dy = self.dev.w_ring["RAD"]
        s.thrust_core = float(np.sum(tau * dy))
        s.core_flow_power = float(np.sum(tau * ring_c * dy))
        s.step = int(self.steps_on)
        self.dev.amplitude["ROTOR"] = s.thrust_rotor
        self.dev.amplitude["RAD"] = s.thrust_core

    def validity_report(self, u_max: float, finite: bool) -> dict[str, Any]:
        """`racelab.RaceRollout.validity_report`'s predicates at this state."""
        s = self.state
        mgu = self.elements["MGU"]
        out: dict[str, Any] = {}
        ok = bool(mgu.validity(np.full(1, s.omega)))
        out["MGU"] = {"valid": ok, "omega": float(s.omega), "back_emf": float(mgu.k_e * s.omega),
                      "V_oc": float(PT.V_OC), "current": float(s.current)}
        out["ROTOR"] = {"valid": bool(s.rotor_valid), "induction": float(s.induction)}
        fl = self.flow
        out["FLUID"] = {"valid": bool(finite and u_max <= RL.U_MAX_BAND), "u_max": float(u_max),
                        "band": RL.U_MAX_BAND, "finite": bool(finite),
                        "cell_reynolds_on_the_background": float(fl.ov.bg.h * u_max / fl.nu),
                        "note": "the body-fitted solver declares no cell-Reynolds bound; the bound "
                                "of 8 is WindowNS's, whose windows are not in this march"}
        try:
            out["RAD"] = {"valid": bool(self._core.validity()), "ua": float(s.ua)}
        except Exception as exc:                                 # pragma: no cover
            out["RAD"] = {"valid": None, "why": f"not consultable: {exc}"}
        out["_declined"] = sorted(k for k, r in out.items() if not k.startswith("_") and r.get("valid") is False)
        out["_unconsultable"] = sorted(k for k, r in out.items() if not k.startswith("_") and r.get("valid") is None)
        return out

    # -- the step -----------------------------------------------------------

    def step(self) -> dict[str, Any]:
        flow = self.flow
        on = flow.t + 0.5 * flow.dt >= self.t_on
        t0 = time.perf_counter()
        if on:
            self.refresh(flow.U)
            p_start = {k: self.dev.power(flow.U, k) for k in ("ROTOR", "RAD")}
        t_union = time.perf_counter() - t0
        rec = flow.step()
        forces = {}
        fx = fy = 0.0
        for name in self.solids:
            f = flow.forces(name)
            forces[name] = f
            fx += f["fx"]
            fy += f["fy"]
        self.last_forces = forces
        if not on:
            return rec
        t0 = time.perf_counter()
        s = self.state
        U = flow.U
        p_end = {k: self.dev.power(U, k) for k in ("ROTOR", "RAD")}
        sp_ = np.hypot(flow.U, flow.V)
        u_max = float(np.max(sp_))
        finite = bool(np.all(np.isfinite(flow.U)) and np.all(np.isfinite(flow.V)))
        rep = self.validity_report(u_max, finite)
        self.envelope = rep
        if rep["_declined"]:
            self.outside_steps += 1
            if self.outside_first is None:
                self.outside_first = {"t": float(rec["t"]), "declined": list(rep["_declined"])}
        self.steps_on += 1
        if self.steps_on % self.n_per_coolant == 0:
            self.block.q_machine = float(s.q_machine)
            tw = float(np.mean(self.block._face_T(self.block._T, "outer")))
            self.block.t_case = tw + self.block.q_machine / (self.block.h_mount * self.block.area)
            row = VM._coolant_step(self.loop, self.block, self.t_in)
            self.t_in = row["t_in"]
            row["t"] = float(rec["t"])
            row["union_step"] = int(self.steps_on)
            self.coolant.append(row)
        tr = self.trace
        tr["t"].append(float(rec["t"]))
        for k in ("u_rotor", "u_core", "ua", "induction", "omega", "current", "thrust_rotor",
                  "shaft_power", "thrust_core", "core_flow_power", "q_machine"):
            tr[k].append(float(getattr(s, k)))
        tr["load"].append(float(fy))
        tr["drag"].append(float(fx))
        tr["u_max"].append(u_max)
        pr = 0.5 * (p_start["ROTOR"] + p_end["ROTOR"])
        pc = 0.5 * (p_start["RAD"] + p_end["RAD"])
        tr["power_rotor"].append(pr)
        tr["power_core"].append(pc)
        tr["power_on_the_fluid"].append(pr + pc)
        tr["power_rotor_start"].append(p_start["ROTOR"])
        tr["power_rotor_end"].append(p_end["ROTOR"])
        tr["power_core_start"].append(p_start["RAD"])
        tr["power_core_end"].append(p_end["RAD"])
        tr["u_at_the_rotor_plane"].append(self.dev.plane_mean(U, "ROTOR"))
        tr["u_at_the_core_plane"].append(self.dev.plane_mean(U, "RAD"))
        tr["t_wall"].append(float(np.mean(self.block._face_T(self.block._T, "inner"))))
        tr["mgu_valid"].append(bool(rep["MGU"]["valid"]))
        tr["rotor_valid"].append(bool(rep["ROTOR"]["valid"]))
        tr["rad_valid"].append(rep["RAD"]["valid"])
        tr["fluid_valid"].append(bool(rep["FLUID"]["valid"]))
        tr["back_emf"].append(float(rep["MGU"]["back_emf"]))
        tr["union_s"].append(t_union + time.perf_counter() - t0)
        return rec

    # -- what the balances read ---------------------------------------------

    def union_march(self, t_from: float, t_to: float) -> VM.UnionMarch:
        """The trace over ``[t_from, t_to]`` as a `vehicle_march.UnionMarch`, so
        `vehicle_march.receiver_balances(m, frac=1.0)` reads exactly that window."""
        t = np.asarray(self.trace["t"])
        sel = (t >= t_from - 1e-9) & (t <= t_to + 1e-9)
        trace = {k: np.asarray(self.trace[k], dtype=float)[sel] for k in TRACE_KEYS}
        return VM.UnionMarch(
            steps=int(sel.sum()), join_coupling={"J1": "lagged", "J2": "lagged", "J3": "lagged"},
            joins=("J1", "J2", "J3"), trace=trace, coolant=list(self.coolant),
            block_T=self.block._T.copy(), t_in=self.t_in,
            notes={"n_per_coolant": self.n_per_coolant, "rotor_width": self.width,
                   "machine_scale": self.width, "host_inflow": self.u_host, "p_ref": self.p_ref,
                   "window": [t_from, t_to], "coupling": "lagged, one refresh per fluid step"})


# ---------------------------------------------------------------------------
# a march's state, kept
# ---------------------------------------------------------------------------


def save_state(flow, path: str) -> str:
    """The flow's state -- enough for `load_state` to continue it with BDF2."""
    if not path.endswith(".npz"):
        raise ValueError("np.savez appends .npz; name the file with it")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, U=flow.U, V=flow.V, P=flow.P, Um1=flow.Um1, Vm1=flow.Vm1,
             t=np.array(flow.t), k=np.array(flow.k))
    for k in range(6):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if k == 5:
                raise
            time.sleep(0.35 * (k + 1))
    return path


def load_state(flow, path: str) -> dict[str, Any]:
    with np.load(path) as z:
        flow.set_state(z["U"], z["V"], z["P"], u_prev=z["Um1"], v_prev=z["Vm1"], t=float(z["t"]))
        return {"t": float(z["t"]), "k": int(z["k"])}
