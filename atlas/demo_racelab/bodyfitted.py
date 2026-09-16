"""The body-fitted car as a demo column: march it, draw it, and say what is missing.

PoC 3, Tier 68.  [[poc3-racelab-bodyfitted-demo]].

The demo's column has always been the porous one: fourteen rectangular windows on
one lattice, each flippable between a classical and a learned expert.  Tiers 60
to 67 built a second column -- curved grids wrapped round the car's parts, the
bodies as real walls, the duct open, the radiator core and the recovery turbine
marching in it -- and it has never been on the screen.  This module puts it
there, as a SECOND column beside the porous one rather than in place of it
(requirements 13.3), because the porous column is the only place the
window-flipping story of section 12 can be told over fourteen windows.

What this column can and cannot offer, stated once
--------------------------------------------------

**The fluid is one expert** (Tier 65): the composite is a single implicit solve
whose interpolation rows sit in the same matrix as its momentum rows, so no grid
can be stepped alone and there is nothing to flip per window inside it.

**A learned expert can reach at most a third of it** (Tier 67): Poseidon-T
accepts nothing but a uniform 128 x 128 grid, the body-fitted grids hold 61.0% of
the unknowns, and a uniform grid cannot accept a curvilinear patch in any
arrangement.  Four or five background windows exist, they cover about a sixth of
the unknowns under a real tiling, and none of them is near the car.

So this column's honest offer is: the real physics, drawn; the devices and the
joins, live; and a **labelled** statement of the 61% no learned expert can ever
touch.  `coverage` carries the numbers and `LEARNED_HOLE` the reason, in the
shape section 4.3 already uses for families with no learned option -- greyed out
WITH the reason, never hidden.

Why the referent is off, and it is not only about speed
-------------------------------------------------------

Section 5.3 wants the global rms error against an all-classical march.  Here that
is worth saying carefully: **with no window running a learned expert, the
all-classical march IS this march**, so a referent would cost a second 0.6 s
solve per step to measure a number that is identically zero.  It is refused with
that reason until a learned window exists, rather than run to produce a
confident-looking 0.0 -- and section 5.4's third rule is untouched, because the
per-window one-step error needs no referent at all.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..cases import car_render as CR
from ..cases import car_windows as CW

#: The fields this column can draw.  `vorticity` is absent ON PURPOSE and the
#: page is told why: it is a derivative, and on twelve overlapping curvilinear
#: grids it is not another `sample` call but an operator nobody has built.  A
#: field that cannot be drawn honestly is named, not quietly missing.
FIELDS = ("speed", "u", "v", "p")

FIELD_HOLES = {
    "vorticity": ("a derivative on twelve overlapping curvilinear grids, which is an "
                  "operator this column does not have -- not another sample of the "
                  "probe operator"),
    "temperature": ("the coolant loop is lumped and owns no region, so there is no "
                    "thermal field over the fluid; the return, block and disc "
                    "temperatures are numbers in the telemetry instead"),
    "error": ("the all-classical referent IS this march until a window runs a learned "
              "expert, so the error field would be identically zero"),
}

#: Why the learned switch is not offered over most of this column.
LEARNED_HOLE = (
    "Poseidon-T accepts nothing but a uniform 128 x 128 grid.  The body-fitted "
    "grids -- the boundary layers, the wheel clearances and the cooling duct -- "
    "hold 61.0% of this column's unknowns, and a uniform grid cannot accept a "
    "curvilinear patch in any arrangement, so no tiling however clever can reach "
    "them (Tier 67, W288)."
)

#: The scale the speed field is drawn on.  The field reaches about 2.8; fixing
#: the top at 2.0 keeps the free stream mid-scale where the wake is legible, and
#: `frame` reports how many pixels that clips so the picture is not mistaken for
#: a measurement above it.
SPEED_LO, SPEED_HI = 0.0, 2.0


@dataclass
class BodyFittedConfig:
    nx: int = CR.NX
    ny: int = CR.NY
    #: Tier 64's cached prefix: DEVICE-FREE, at t = 8.
    prefix: str = os.path.join("out", "racelab13", "cache", "prefix_t8.npz")
    #: Where the demo actually releases from, and why it is not the prefix.
    #:
    #: **A release state belongs to its machine.**  The prefix was marched with
    #: the devices applying nothing, so its duct runs at the device-free speed --
    #: measured here at 1.292 of what the machine is sized for, against a band
    #: that ends at 1.1408.  Releasing there puts the rotor outside its envelope
    #: on step 1 and it closes at about 0.0013 a step, so the demo would be
    #: stamped OUTSIDE for roughly 300 steps, three minutes, before it said
    #: anything trustworthy.  Tier 64 saw the same transient and judged its arms
    #: over a window starting at t = 12 for exactly this reason.  So the demo
    #: settles ONCE to `release_t` with the devices on, caches that, and releases
    #: from it forever after.
    release_t: float = 12.0
    settled: str | None = None            # None -> out/cache/bodyfitted_t<release_t>.npz
    #: where the probe operator is cached (W287); None picks out/cache/
    cache_path: str | None = None
    #: OFF, and refused with a reason while it would measure zero.  See the
    #: module docstring.
    referent: bool = False
    field: str = "speed"


@dataclass
class BodyFittedColumn:
    """Build once, step forever, and hand the page a frame and the truth."""

    cfg: BodyFittedConfig = field(default_factory=BodyFittedConfig)
    ov: Any = None
    solids: Any = None
    flow: Any = None
    union: Any = None
    raster: Any = None
    coverage: dict = field(default_factory=dict)
    build_report: dict = field(default_factory=dict)
    step_i: int = 0
    step_s: float | None = None
    steps: list = field(default_factory=list)

    # -- building -----------------------------------------------------------

    def build(self, root: str = ".") -> dict:
        """The composite, the flow, the devices and the probe operator.

        The composite is the expensive half and is NOT cached -- it is a live
        object graph of grids, donor searches and index maps, and pickling it
        would be a second definition of the car that could drift from the first.
        The probe operator IS cached, and it validates itself against this
        composite rather than trusting its key (W287).
        """
        import sys

        scripts = os.path.join(os.path.abspath(root), "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import tier62_car_solids as T62
        import tier63_duct_openings as T63
        T62.GEOMETRY = T63.geometry

        from ..cases import car_solids as CS
        from ..cases import car_union as CU

        rep: dict[str, Any] = {}
        t0 = time.perf_counter()
        C = T62._car()
        self.ov, self.solids = C["ov"], C["solids"]
        rep["composite_s"] = time.perf_counter() - t0
        rep["grids"] = len(self.ov.grids)
        rep["n_unknowns"] = int(self.ov.n_unknowns)

        t0 = time.perf_counter()
        self.raster, rep["raster"] = CR.cached_raster(
            self.ov, self.cfg.nx, self.cfg.ny, path=self.cfg.cache_path)
        rep["raster_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.flow = CS.car_flow(self.ov, self.solids, precond="ilu")
        rep["flow_s"] = time.perf_counter() - t0

        settled = self.cfg.settled or os.path.join(
            os.path.abspath(root), "out", "cache",
            "bodyfitted_t%g.npz" % self.cfg.release_t)
        prefix = os.path.join(os.path.abspath(root), self.cfg.prefix)

        t0 = time.perf_counter()
        D = _devices(self.ov, root)
        self.union = CU.CarUnion(self.flow, D, _sizing(root),
                                 solids=[s.name for s in self.solids], t_on=0.0)
        rep["devices_s"] = time.perf_counter() - t0

        if os.path.isfile(settled):
            st = CU.load_state(self.flow, settled)
            rep["released_from"] = {"file": os.path.relpath(settled, os.path.abspath(root)),
                                    "t": float(st["t"]), "k": int(st["k"]),
                                    "kind": "settled with the devices on"}
            rep["settle_s"] = 0.0
        elif os.path.isfile(prefix):
            st = CU.load_state(self.flow, prefix)
            rep["settle_s"] = self._settle_to(settled, root)
            rep["released_from"] = {"file": os.path.relpath(settled, os.path.abspath(root)),
                                    "t": float(self.flow.t), "k": int(self.flow.k),
                                    "kind": "settled here, from the device-free prefix at "
                                            "t = %.3f" % float(st["t"])}
        else:
            rep["released_from"] = {"file": None,
                                    "why": "neither a settled state nor the cached prefix is "
                                           "on disk; this column starts from the solver's "
                                           "own initial field"}
            rep["settle_s"] = 0.0
        rep["u_host_sized"] = float(self.union.u_host)

        self.coverage = _coverage(self.ov)
        rep["coverage"] = self.coverage
        rep["total_s"] = (rep["composite_s"] + rep["raster_s"] + rep["flow_s"]
                          + rep["devices_s"])
        self.build_report = rep
        return rep

    def _settle_to(self, path: str, root: str) -> float:
        """March the devices on from the prefix until `release_t`, then cache it.

        Paid once per car.  The state that comes out belongs to the machine that
        will run on it, which is the whole point -- see `BodyFittedConfig`.
        """
        from ..cases import car_union as CU

        t0 = time.perf_counter()
        guard = 0
        while self.flow.t + 0.5 * self.flow.dt < self.cfg.release_t:
            self.union.step()
            guard += 1
            if guard > 20000:                                    # pragma: no cover
                raise RuntimeError("settling did not reach release_t")
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        CU.save_state(self.flow, path)
        # the settle's own declines are history, not this run's
        self.union.outside_steps = 0
        self.union.outside_first = None
        self.union.steps_on = 0
        for k in self.union.trace:
            self.union.trace[k].clear()
        return time.perf_counter() - t0

    # -- marching -----------------------------------------------------------

    def step(self) -> dict:
        if self.union is None:
            raise RuntimeError("build() first")
        t0 = time.perf_counter()
        rec = self.union.step()
        self.step_s = time.perf_counter() - t0
        self.step_i += 1
        row = {"i": self.step_i, "t": float(rec.get("t", 0.0)),
               "wall_s": self.step_s,
               "solver_step_s": float(rec.get("step_s", 0.0) or 0.0),
               "ilu_s": float(rec.get("ilu_s", 0.0) or 0.0),
               "max_div": float(rec.get("max_div", 0.0) or 0.0),
               "u_max": float(rec.get("u_max", 0.0) or 0.0)}
        self.steps.append(row)
        if len(self.steps) > 512:
            del self.steps[:-512]
        return row

    def steps_per_second(self, n: int = 20) -> float | None:
        """Measured over the steps that did NOT refactor the incomplete LU.

        The first step after a release refactors and is about five times a
        typical one (Tier 66), so including it would report a rate the column
        never actually runs at.
        """
        plain = [s["wall_s"] for s in self.steps[-n:] if s["ilu_s"] == 0.0]
        if not plain:
            return None
        return 1.0 / float(np.median(plain))

    # -- drawing ------------------------------------------------------------

    def frame(self, which: str | None = None) -> tuple[bytes, dict]:
        which = which or self.cfg.field
        if which not in FIELDS:
            raise ValueError(f"field must be one of {FIELDS}, got {which!r}")
        F = self.raster.fields(self.flow.U, self.flow.V, self.flow.P)
        a = F["speed"] if which == "speed" else F[which]
        if which == "speed":
            lo, hi = SPEED_LO, SPEED_HI
        else:
            ok = np.isfinite(a)
            lo = float(np.nanpercentile(a[ok], 1.0)) if ok.any() else 0.0
            hi = float(np.nanpercentile(a[ok], 99.0)) if ok.any() else 1.0
            if hi <= lo:
                hi = lo + 1.0
        png = CR.field_png(a, lo, hi, lut="speed" if which == "speed" else "signed")
        drawn = np.isfinite(a)
        payload = {
            "field": which, "lo": lo, "hi": hi,
            "width": self.raster.nx, "height": self.raster.ny,
            "clipped_pixels": int(np.nansum(a[drawn] > hi)) if drawn.any() else 0,
            "field_max": float(np.nanmax(a[drawn])) if drawn.any() else None,
            "masked_pixels": int(self.raster.mask.sum()),
            "fields": list(FIELDS), "field_holes": FIELD_HOLES,
        }
        return png, payload

    # -- what the page must say ---------------------------------------------

    def telemetry(self) -> dict:
        tr = getattr(self.union, "trace", {}) or {}

        def last(k):
            v = tr.get(k)
            return float(v[-1]) if v else None

        rate = self.steps_per_second()
        return {
            "step_i": self.step_i,
            "t": last("t"),
            "steps_per_second": rate,
            "step_s": self.step_s,
            "load": last("load"), "drag": last("drag"),
            "u_at_the_rotor_plane": last("u_at_the_rotor_plane"),
            "u_at_the_core_plane": last("u_at_the_core_plane"),
            "shaft_power": last("shaft_power"),
            "q_machine": last("q_machine"),
            "coolant_return_t": (float(self.union.coolant[-1]["t_in"])
                                 if getattr(self.union, "coolant", None) else None),
            "envelope": getattr(self.union, "envelope", None),
            "outside_steps": getattr(self.union, "outside_steps", 0),
            "coverage": self.coverage,
            "learned_hole": LEARNED_HOLE,
            "referent": self.referent_state(),
        }

    def referent_state(self) -> dict:
        """Off, and refused with a reason while it would measure zero."""
        learned = int(self.coverage.get("windows_learned", 0))
        if learned == 0:
            return {"on": False, "available": False,
                    "why": ("no window is running a learned expert, so the all-classical "
                            "referent IS this march and its error is identically zero; "
                            "running it would cost a second solve a step to measure "
                            "nothing")}
        return {"on": bool(self.cfg.referent), "available": True,
                "why": ("a second all-classical march, about 0.6 s a step, so turning it "
                        "on roughly halves the frame rate")}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _coverage(ov) -> dict:
    """Tier 67's territory, in the shape the page needs."""
    t = CW.territory(ov, CW.POSEIDON_CELLS, order="row", require="owned")
    return {
        "window_cells": t["window_cells"], "window_span": t["window_span"],
        "n_windows": t["n_windows"],
        "windows": t["windows"],
        "reach_frac_of_composite": t["reach_frac_of_composite"],
        "tiled_frac_of_composite": t["tiled_frac_of_composite"],
        "body_grid_frac": t["body_grid_frac"],
        "body_grid_unknowns": t["body_grid_unknowns"],
        "composite_unknowns": t["composite_unknowns"],
        "windows_learned": 0,
        "note": ("a learned expert could cover at most reach_frac_of_composite of this "
                 "column and body_grid_frac of it can never be covered at all"),
    }


def _devices(ov, root: str):
    """The two devices on THIS composite, sized from the same geometry it was cut
    from -- `tier64_car_union._devices` builds them exactly this way."""
    import sys
    scripts = os.path.join(os.path.abspath(root), "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import tier63_duct_openings as T63
    from ..cases import car_union as CU
    return CU.DuctDevices(ov, CU.car_devices(T63.geometry()))


def _sizing(root: str) -> float:
    """The inflow the machine is sized for.

    Tier 64 had to ITERATE this: the machine generates only between 0.9731 and
    1.1408 of its sizing inflow (W282), and the duct drifts across the window, so
    a sizing taken from a device-free duct spends most of the window motoring.
    The demo takes Tier 64's admitted arm rather than re-deriving it.
    """
    import json
    p = os.path.join(os.path.abspath(root), "out", "racelab13", "racelab13.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            rec = json.load(fh)
        arms = rec.get("arms", {}).get("list") or []
        for a in reversed(arms):
            if a.get("admitted") and a.get("u_host") is not None:
                return float(a["u_host"])
        if arms and arms[-1].get("u_host") is not None:
            return float(arms[-1]["u_host"])
    return 0.08757188607424089
