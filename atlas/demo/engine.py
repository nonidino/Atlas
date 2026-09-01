"""The demo's simulation engine: a live composed march you can edit and optimise.

This sits ON TOP of `atlas.cases.wind_farm_design` and changes nothing in it.
Everything here is either a call into that module or a subclass of one of its
classes, and the two places where a subclass overrides behaviour are pinned
bitwise against the parent in `tests/test_tier22_demo.py`.

Why a live march is possible at all, and why this file quotes no fixed speed
---------------------------------------------------------------------------

One composed macro-step, forward, float64, is the unit of everything here.  An
earlier version of this docstring gave a table of what that costs -- 212 ms at
6 windows, 259 at 12, 445 at 24 -- and the table was **wrong the next time
anyone measured it**: the same code on the same box, unchanged, came back at
~890 ms at 6 windows.  Nothing had regressed.  The box is a Core Ultra 7 155H
and it was running at 1.4 GHz against a 3.8 GHz maximum.

The lesson is kept rather than the table.  A wall-clock number is a claim about
a machine in a power state, this demo is meant to be carried onto other
people's machines, and so **the engine measures itself**: `Engine.eta_s`
reports an exponential moving average of what iterations have actually cost
here, `_publish` ships it, and the screen marks it as an estimate only until the
first real one lands.  Order of magnitude, which is what the design rests on:
one forward macro-step is a fraction of a second to about a second, a gradient
macro-step is ~5.5x that, and both scale with the window count.

The FIELD is therefore real-time or nearly so: drag a turbine and the wake
re-forms over the next few seconds of wall-clock, because those are the same
seconds of simulated time.  A gradient over a horizon of H macro-steps is
H x 5.5 forward steps, which is why the optimiser here is **warm-started and
short-horizon** rather than the PoC's freestream-start 50-step estimator:

  * the PoC's objective (`wind_farm_design.objective`) marches from the
    freestream every evaluation, which makes J a clean deterministic function of
    theta alone and costs 40-50 macro-steps an evaluation -- tens of seconds
    on any machine this has run on.  That is the right estimator for a
    measurement and the wrong one for a demonstration, where nothing may take
    twenty seconds to show a first frame;
  * this engine holds ONE persistent field, and each optimiser iteration
    differentiates the next H macro-steps of it.  The rollout the gradient is
    taken through **is** the live march, so the field never stops animating and
    the user sees the wake respond inside the optimiser step rather than after
    it.

**That is a different estimator and the demo says so on screen.**  It is
warm-started, so it carries the history of every layout the user has already
tried, and it is short, so it sees less of the downstream interaction than the
50-step rollout does.  The number it reports is a demo number.  The
**verification panel** is what converts it back into a defensible one: on
demand, the same layout is marched from the freestream by the composed column
AND by `scaling_ladder.reference_monolith` -- the undivided classical solver,
which has no composition error because it has no cut -- and the two are shown
side by side.  That is [[prior-art-and-novelty-atlas-0.1]] section 5's
"search wide and cheap, verify the shortlist with the classical stack", on
screen instead of in a caption.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np

from ..cases import scaling_ladder as sl
from ..cases import wake_array as wa
from ..cases import wind_farm_design as wd

torch = wd.torch

__all__ = [
    "DOMAINS", "LIMITS", "DEVICES", "REPLAY_HOLD", "Comparison", "DemoConfig",
    "DemoRollout", "Engine", "capacity", "layout_grid", "layout_staggered",
    "colour_lut", "field_png", "validity",
]

# ---------------------------------------------------------------------------
# what the user may change, and how far
# ---------------------------------------------------------------------------

#: Domain presets, as (n_col, n_row) window grids on the CS-7 ladder.  These are
#: the rungs the scaling ladder actually measured, not arbitrary boxes, so a
#: number produced at any of them is comparable with something.
DOMAINS: dict[str, tuple[int, int]] = {
    # The 2x1 rung is deliberately absent. Its box is 4.0 x 1.0 D once the
    # freestream band and the outlet margin are taken out, which holds THREE
    # turbines at the 2 D packing limit -- fewer than the minimum this demo
    # offers. A preset that cannot host a legal layout is not a preset.
    "small":  (3, 2),      # 352x240 cells, 11.0 x  7.5 D,  6 windows
    "medium": (4, 3),      # 464x352,       14.5 x 11.0 D, 12 windows
    "large":  (6, 4),      # 688x464,       21.5 x 14.5 D, 24 windows
}

#: Every knob is clamped, and every clamp has a reason on the record.
LIMITS = {
    # 4 is the smallest array with an in-line pair AND a clean-inflow control;
    # 25 is the PoC's upper rung.
    "k": (4, 25),
    # `reference_validity` is `dx |u| / nu <= 8` and the case study's own
    # freestream value is 7.97 at u_inf = 1.  Above 1.0 the reference
    # discretization is outside the envelope it declares for itself.
    "u_inf": (0.5, 1.4),
    # The composition layer's Leray projection declares its hypothesis as "the
    # march holds the inlet and both laterals at (U_INF, 0)".  A yawed
    # freestream is a different kernel, so this is deliberately a SMALL range
    # with a warning across most of it.
    "inflow_deg": (-15.0, 15.0),
    # Adam's step is its learning rate, in rotor diameters / radians per step.
    "lr_pos": (0.02, 0.40),
    "lr_yaw": (0.005, 0.12),
    # The optimiser horizon, in macro-steps.  Below 3 the wake has not moved;
    # above 15 an iteration takes longer than a demo's attention span.
    "horizon": (3, 15),
    "yaw_deg": (-30.0, 30.0),
    "s_min": (1.5, 6.0),
    "verify_steps": (10, 60),
}

#: Where the coupled graph runs.  A launch-time choice, not a live knob: moving
#: devices rebuilds every solver.  The CLASSICAL side of the head-to-head is
#: numpy and therefore always on the CPU, so on a GPU box the two columns of the
#: comparison are not running on the same hardware -- which the panel says,
#: rather than quietly reporting the ratio as if they were.
DEVICES = ("cpu", "cuda")

#: Macro-steps of flow marched per replay frame when the replay is playing.
#: One would run the trajectory back faster than the wake can follow it: the
#: layout would be three optimiser steps ahead of the field it is supposed to
#: explain.  Two is slow enough that the wakes visibly track the turbines.
REPLAY_HOLD = 2


def _cuda_ok() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:                                          # pragma: no cover
        return False

#: The wake-array viewer's ramp, stop for stop (`out/w93/wake-array-template.html`).
#: Dark and cool in the wake, NEUTRAL GREY at the freestream, warm where the flow
#: speeds up round the array.  Fixed, never auto-scaled: a colour that means a
#: different velocity in every frame is a colour that means nothing.
RAMP_STOPS = [
    (0.000, (10, 22, 38)),
    (0.190, (16, 60, 92)),
    (0.380, (26, 112, 120)),
    (0.520, (125, 166, 168)),
    (0.615, (206, 214, 214)),        # u = 1.0 exactly: the freestream, quiet
    (0.790, (214, 150, 62)),
    (1.000, (246, 222, 168)),
]

#: The velocity window the ramp spans.  Chosen so u = 1 lands on the neutral
#: stop: (1 - 0.20) / (1.50 - 0.20) = 0.6154.
U_LO, U_HI = 0.20, 1.50


def colour_lut() -> np.ndarray:
    """256x3 uint8 lookup table for `RAMP_STOPS`."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(1, len(RAMP_STOPS)):
            a, ca = RAMP_STOPS[k - 1]
            b, cb = RAMP_STOPS[k]
            if t <= b:
                f = 0.0 if b == a else (t - a) / (b - a)
                lut[i] = [round(ca[j] + f * (cb[j] - ca[j])) for j in range(3)]
                break
        else:
            lut[i] = RAMP_STOPS[-1][1]
    return lut


_LUT = colour_lut()


def field_png(u: np.ndarray, stride: int = 1) -> bytes:
    """Colour-map the streamwise field and encode it as a PNG.

    Row 0 of the array is y = 0, and an image's row 0 is the top, so the array
    is flipped here -- the client then draws it with y up, which is how a plan
    view of a farm is read.
    """
    from PIL import Image
    a = u[::stride, ::stride]
    t = np.clip((a - U_LO) / (U_HI - U_LO), 0.0, 1.0)
    idx = (t * 255.0).astype(np.uint8)
    rgb = _LUT[idx][::-1]                       # flip y for image coordinates
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the configuration
# ---------------------------------------------------------------------------


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _ema(prev: float | None, x: float, a: float = 0.25) -> float:
    """Exponential moving average, seeded by its first sample."""
    return float(x) if prev is None else float((1 - a) * prev + a * x)


@dataclass
class DemoConfig:
    domain: str = "medium"
    k: int = 6
    u_inf: float = 1.0
    inflow_deg: float = 0.0
    s_min: float = 2.0
    horizon: int = 7
    lr_pos: float = 0.12
    lr_yaw: float = 0.035
    verify_steps: int = 30
    device: str = "cpu"

    def clamped(self) -> "DemoConfig":
        d = asdict(self)
        d["domain"] = self.domain if self.domain in DOMAINS else "medium"
        d["k"] = int(_clamp(int(self.k), *LIMITS["k"]))
        for key in ("u_inf", "inflow_deg", "lr_pos", "lr_yaw", "s_min"):
            d[key] = float(_clamp(float(d[key]), *LIMITS[key]))
        for key in ("horizon", "verify_steps"):
            d[key] = int(_clamp(int(d[key]), *LIMITS[key]))
        dev = str(d.get("device") or "cpu")
        d["device"] = dev if dev in DEVICES and (dev == "cpu" or _cuda_ok()) else "cpu"
        return DemoConfig(**d)

    # -- the FarmCase this maps onto ---------------------------------------
    def case(self) -> wd.FarmCase:
        n_col, n_row = DOMAINS[self.domain]
        # k_col x k_row must multiply to k; a 1-row "grid" is the general case
        # and the demo supplies its own layouts, so `initial_design` is unused.
        return wd.FarmCase(f"demo-{self.domain}-{self.k}", n_col=n_col,
                           n_row=n_row, k_col=int(self.k), k_row=1,
                           steps=max(10, self.horizon), avg_window=3,
                           s_min=float(self.s_min))

    @property
    def horizon_eta_s(self) -> float:
        """A COLD-START guess at one optimiser iteration, used until it is measured.

        The engine replaces this with an exponential moving average of what
        iterations on THIS machine actually cost the moment it has one, and the
        reason is a number that was in this file: a table of per-step costs
        measured once on the development box, quoted as fact, and wrong by 4x
        the next time anyone looked -- because the box is a Core Ultra 7 155H
        and it had clocked itself down from 3.8 GHz to 1.4 GHz.  Any hard-coded
        wall-clock is a claim about a machine in a power state, and the demo is
        meant to be run on other people's machines.
        """
        per = {"small": 4.5, "medium": 5.9, "large": 9.2}
        return per.get(self.domain, 5.9) * self.horizon


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------


def _box(case: wd.FarmCase):
    lo, hi = case.box
    return (lo[0], hi[0], lo[1], hi[1])


def capacity(case: wd.FarmCase) -> int:
    """How many turbines this box holds at its own minimum spacing.

    A rectangular-lattice count, because a rectangular lattice is what the two
    presets lay out: a capacity computed from area would promise positions no
    grid can hold, and the first thing the user would see is a spacing
    violation on the DEFAULT layout. (It was, before this existed.)
    """
    x0, x1, y0, y1 = _box(case)
    s = max(case.s_min, 1e-6)
    return max(1, (int((x1 - x0) / s) + 1) * (int((y1 - y0) / s) + 1))


def _shape_for(case: wd.FarmCase, k: int) -> tuple[int, int]:
    """Columns x rows for `k`: as close to square-celled as the box allows,
    and never tighter than `s_min` in either direction."""
    x0, x1, y0, y1 = _box(case)
    w, h, s = x1 - x0, y1 - y0, max(case.s_min, 1e-6)
    max_col, max_row = int(w / s) + 1, int(h / s) + 1
    best = None
    for n_col in range(1, max_col + 1):
        n_row = int(math.ceil(k / n_col))
        if n_row > max_row:
            continue
        dx = w / max(1, n_col - 1) if n_col > 1 else w
        dy = h / max(1, n_row - 1) if n_row > 1 else h
        score = abs(math.log(max(dx, 1e-6) / max(dy, 1e-6)))
        if best is None or score < best[0]:
            best = (score, n_col, n_row)
    return (best[1], best[2]) if best else (max_col, max_row)


def _grid_shape(case: wd.FarmCase) -> tuple[int, int]:
    """The shape actually used, which may be denser than `s_min` allows.

    When the caller asks for more turbines than the box holds, this returns a
    shape that still places every one of them at a DISTINCT position, tighter
    than the packing limit. Stacking them on one point instead would report a
    minimum spacing of zero and a farm power that is nonsense; a legible
    over-packed grid reports a real spacing violation, which the validity panel
    is there to show. The engine clamps `k` to `capacity` before this is reached,
    so it is the defensive path rather than the usual one.
    """
    k = case.k
    if k <= capacity(case):
        return _shape_for(case, k)
    x0, x1, y0, y1 = _box(case)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    n_col = max(1, int(round(math.sqrt(k * w / h))))
    return n_col, int(math.ceil(k / n_col))


def layout_grid(case: wd.FarmCase) -> np.ndarray:
    """A regular grid, every turbine facing the wind: the thing to beat."""
    n_col, n_row = _grid_shape(case)
    x0, x1, y0, y1 = _box(case)
    xs = np.linspace(x0, x1, n_col) if n_col > 1 else np.array([(x0 + x1) / 2])
    ys = np.linspace(y0, y1, n_row) if n_row > 1 else np.array([(y0 + y1) / 2])
    out = [(xs[i], ys[j], 0.0) for j in range(n_row) for i in range(n_col)]
    return np.array(out[:case.k], dtype=float).reshape(-1)


def layout_staggered(case: wd.FarmCase) -> np.ndarray:
    """The grid with alternate columns pushed half a row spacing downwind-ward.

    The offset is taken out of the row range BEFORE the rows are placed, not
    added to a finished grid and clamped at the wall -- clamping is what turned
    a 12-turbine stagger into two turbines on the same point (measured 1.12 D
    against a 2 D limit), because the top row had nowhere to go.
    """
    n_col, n_row = _grid_shape(case)
    x0, x1, y0, y1 = _box(case)
    span = y1 - y0
    # the largest offset that still leaves s_min between rows in a column
    room = max(0.0, span - case.s_min * max(1, n_row - 1))
    base = span / max(1, n_row - 1) if n_row > 1 else span
    off = min(0.5 * base, room)
    ys = (np.linspace(y0, y1 - off, n_row) if n_row > 1
          else np.array([(y0 + y1) / 2 - off / 2]))
    xs = np.linspace(x0, x1, n_col) if n_col > 1 else np.array([(x0 + x1) / 2])
    out = [(xs[i], ys[j] + (off if i % 2 else 0.0), 0.0)
           for j in range(n_row) for i in range(n_col)]
    return np.array(out[:case.k], dtype=float).reshape(-1)


LAYOUTS: dict[str, Callable[[wd.FarmCase], np.ndarray]] = {
    "grid": layout_grid,
    "staggered": layout_staggered,
}


# ---------------------------------------------------------------------------
# the rollout, with an inflow the user may point and scale
# ---------------------------------------------------------------------------


class DemoRollout(wd.Rollout):
    """`wind_farm_design.Rollout` with a settable freestream vector.

    The parent hard-codes `wake_array.U_INF` (= 1) along +x in two places, and
    both are declarations rather than incidental: `band` is the boundary
    condition the march imposes, and `project` removes the freestream before the
    global Leray step because the projection is periodic-compatible only about a
    uniform state.  The demo needs both to move, so both are overridden HERE
    rather than in the module.

    **At `u_inf = 1, alpha = 0` this class is the parent, bitwise** -- asserted
    in `tests/test_tier22_demo.py` on a real `macro_step`. That assertion is the
    entire licence for this subclass: everything the PoC measured still applies
    at the default, and every departure from it is a knob the user moved and the
    validity panel is reporting.
    """

    def __init__(self, case, u_inf: float = 1.0, inflow_deg: float = 0.0,
                 **kw) -> None:
        super().__init__(case, **kw)
        self.set_inflow(u_inf, inflow_deg)

    def set_inflow(self, u_inf: float, inflow_deg: float) -> None:
        self.u_inf = float(u_inf)
        self.inflow_deg = float(inflow_deg)
        a = math.radians(self.inflow_deg)
        self.free_u = self.u_inf * math.cos(a)
        self.free_v = self.u_inf * math.sin(a)

    def band(self, u, v):
        zu = torch.full((), self.free_u, dtype=u.dtype, device=u.device)
        zv = torch.full((), self.free_v, dtype=u.dtype, device=u.device)
        return torch.where(self._band, zu, u), torch.where(self._band, zv, v)

    def project(self, u, v):
        uf, vf = u - self.free_u, v - self.free_v
        bu = torch.cat((uf, uf[:, -1:] * self._taper), dim=1)
        bv = torch.cat((vf, vf[:, -1:] * self._taper), dim=1)
        uh, vh = torch.fft.fft2(bu), torch.fft.fft2(bv)
        div = self._kx * uh + self._ky * vh
        uh = uh - self._kx * div / self._k2
        vh = vh - self._ky * div / self._k2
        bu = torch.fft.ifft2(uh).real
        bv = torch.fft.ifft2(vh).real
        return self.free_u + bu[:, :self.nx], self.free_v + bv[:, :self.nx]

    def freestream(self):
        u = torch.full((self.ny, self.nx), self.free_u, dtype=wd.TORCH_DTYPE,
                       device=self.device)
        v = torch.full_like(u, self.free_v)
        return u, v


# ---------------------------------------------------------------------------
# validity -- declared predicates, evaluated live
# ---------------------------------------------------------------------------


def validity(cfg: DemoConfig, case: wd.FarmCase, theta: np.ndarray,
             u_max: float | None) -> dict:
    """Every check here is a predicate some page already declared.

    Nothing is invented for the demo, and each row carries the number, the
    limit, and where the limit comes from -- so "outside validated range" is a
    citation rather than a mood.
    """
    rows = []

    def add(key, label, value, limit, ok, why, unit="", gate=True):
        rows.append({"key": key, "label": label, "value": value, "limit": limit,
                     "ok": bool(ok), "why": why, "unit": unit, "gate": bool(gate)})

    cell_re = wa.DX * cfg.u_inf / wa.NU_REF
    add("cell_re", "cell Reynolds number (freestream)", cell_re, 8.0,
        cell_re <= 8.0 + 1e-9,
        "reference.WindowNS declares h|u|/nu <= 8 as its own validity "
        "predicate, and the wake array's freestream value is 7.97 at "
        "u_inf = 1 -- so this knob has almost no headroom above 1.0 and that "
        "is a property of the checkpoint-derived viscosity, not of the demo")

    if u_max is not None:
        live = wa.DX * u_max / wa.NU_REF
        # NOT a gate. The same predicate evaluated on the state rather than on
        # the freestream is above 8 in ANY developed wake -- it is ~12.7 at the
        # default inflow -- which the case study records rather than hides. A
        # panel that goes red for a condition the user cannot act on and the
        # measurement already accepted is a panel nobody reads.
        add("cell_re_live", "cell Reynolds number (live max |u|)", live, 8.0,
            True,
            "the same predicate on the state rather than the freestream. A "
            "developed wake is ALREADY above 8 at the default inflow, so this "
            "is reported as a diagnostic and not as a gate: the reference "
            "discretization is outside its own declared envelope wherever the "
            "flow is fastest, which is a standing caveat of the case study",
            gate=False)
        add("band", "max |u| inside the stability band", u_max, 3.0,
            u_max <= 3.0,
            "w100_scaling_ladder's own band: past it the composed rollout is "
            "on a trajectory that does not stay finite")

    ang = abs(cfg.inflow_deg)
    add("inflow_deg", "inflow yaw off the x axis", cfg.inflow_deg, 2.0,
        ang <= 2.0, "the global Leray projection declares its hypothesis as "
        "'the march holds the inlet and both laterals at (U_INF, 0)'. A yawed "
        "freestream projects onto a different kernel and the outflow taper no "
        "longer points downstream", unit=" deg")

    ms = wd.min_spacing(theta) if len(theta) >= 6 else float("inf")
    add("spacing", "minimum turbine separation", ms, cfg.s_min, ms >= cfg.s_min - 1e-6,
        "the demo's own packing limit; actuator-disk momentum theory has no "
        "near-field wake model to lean on below it", unit=" D")

    lo, hi = case.box
    inside = bool(np.all(theta >= lo - 1e-6) and np.all(theta <= hi + 1e-6))
    add("box", "turbines inside the domain box", 0.0 if inside else 1.0, 0.0,
        inside, "FarmCase.box keeps every rotor clear of the freestream band "
        "and the outlet")

    worst = [r for r in rows if r["gate"] and not r["ok"]]
    return {"ok": not worst, "rows": rows,
            "headline": ("inside the validated range" if not worst
                         else "OUTSIDE validated range: "
                              + "; ".join(r["label"] for r in worst))}


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    seq: int = 0
    png: bytes = b""
    width: int = 0
    height: int = 0
    stride: int = 1
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# the head-to-head: one layout, solved two ways, timed
# ---------------------------------------------------------------------------


def _ms_stats(ms: list[float]) -> dict:
    """Per-step cost, with the first step held out.

    Step 1 pays for everything lazy -- FFT plans, workspace allocation, the
    first CUDA kernel launch -- and on a 30-step march it is 5-10x the others.
    Averaging it in would report a cost nobody pays after the first second, so
    the headline is the MEDIAN of the remaining steps and the warm-up is shown
    separately rather than dropped.
    """
    if not ms:
        return {"n": 0}
    warm = ms[1:] if len(ms) > 1 else ms
    a = np.asarray(warm, dtype=float)
    return {"n": len(ms), "first_ms": float(ms[0]), "median_ms": float(np.median(a)),
            "mean_ms": float(a.mean()), "min_ms": float(a.min()),
            "total_ms": float(np.sum(ms))}


@dataclass
class Comparison:
    """The same layout marched by both solvers, in lockstep, one step at a time.

    **Why lockstep and not a race.**  Running the two solvers concurrently would
    make a better animation and a worthless measurement: they would be competing
    for the same cores, and each per-step number would be a function of what the
    other one happened to be doing.  They alternate instead -- composed step i,
    then classical step i -- so every timed region has the machine to itself, and
    because both are at step `i` at the same moment their two fields are
    directly comparable, which is where `linf` comes from.
    """

    hash: str = ""
    state: str = "running"                  # running | done | error
    steps: int = 0
    i: int = 0
    composed_power: list[float] = field(default_factory=list)
    classical_power: list[float] = field(default_factory=list)
    composed_ms: list[float] = field(default_factory=list)
    classical_ms: list[float] = field(default_factory=list)
    linf: list[float] = field(default_factory=list)
    seq: int = 0
    png: dict = field(default_factory=dict)       # "composed" / "classical" -> bytes
    width: int = 0
    height: int = 0
    domain_D: tuple = (1.0, 1.0)
    theta: list = field(default_factory=list)
    at_iter: int = 0
    k: int = 0
    n_windows: int = 0
    cells: tuple = (0, 0)
    device: str = "cpu"
    paused_live: bool = False
    started: float = 0.0
    #: Wall-clock at completion. Without it `elapsed_s` is recomputed on every
    #: poll and a finished comparison goes on counting: a 46 s run was reporting
    #: "118 s total" simply because the page had been open for 118 s.
    finished: float = 0.0
    error: str = ""

    def view(self) -> dict:
        """Everything the second window draws, minus the images."""
        end = self.finished or time.time()
        el = end - self.started if self.started else 0.0
        cs, ks = _ms_stats(self.composed_ms), _ms_stats(self.classical_ms)
        cp = float(np.mean(self.composed_power[-5:])) if self.composed_power else 0.0
        kp = float(np.mean(self.classical_power[-5:])) if self.classical_power else 0.0
        return {
            "hash": self.hash, "state": self.state, "steps": self.steps,
            "i": self.i, "seq": self.seq, "width": self.width,
            "height": self.height, "domain_D": list(self.domain_D),
            "theta": self.theta, "at_iter": self.at_iter, "k": self.k,
            "n_windows": self.n_windows, "cells": list(self.cells),
            "device": self.device, "paused_live": self.paused_live,
            "elapsed_s": el, "error": self.error,
            "composed": {"power": cp, "trace": self.composed_power, "ms": cs,
                         "where": self.device},
            "classical": {"power": kp, "trace": self.classical_power, "ms": ks,
                          "where": "cpu"},
            "linf": self.linf,
            "linf_final": self.linf[-1] if self.linf else None,
            "delta": cp - kp,
            "delta_pct": (100.0 * (cp - kp) / kp) if abs(kp) > 1e-12 else None,
            "ms_ratio": (cs.get("median_ms", 0.0) / ks["median_ms"]
                         if ks.get("median_ms") else None),
        }

    def brief(self) -> dict:
        """The one-line version the MAIN window carries, so it can show progress."""
        return {"hash": self.hash, "state": self.state, "i": self.i,
                "steps": self.steps, "paused_live": self.paused_live}


class Engine:
    """One live composed march, editable, optimisable, and verifiable.

    Threading: a single worker thread owns the field and theta and is the only
    thing that touches torch.  Every request from the web layer is a message
    dropped into `self._pending` under a lock and applied by the worker between
    macro-steps -- so a drag never lands halfway through a solve, and the
    optimiser is never re-entered.  Verification runs on its own thread with its
    own solver and never touches engine state.
    """

    def __init__(self, cfg: DemoConfig | None = None) -> None:
        self.lock = threading.Lock()
        self.cfg = (cfg or DemoConfig()).clamped()
        self.mode = "run"            # run | optimize | step | paused | replay
        self.seq = 0
        self.frame = Frame()
        self.history: list[dict] = []          # one entry per optimiser iteration
        self.power_trace: list[float] = []     # one per macro-step
        self.opt_iter = 0
        self.step_ms = 0.0
        # Measured, not tabulated. See `DemoConfig.horizon_eta_s` for why.
        self.fwd_ms: float | None = None        # EMA, one forward macro-step
        self.opt_ms: float | None = None        # EMA, one optimiser iteration
        self.grad_norm = 0.0
        self.notice = ""
        self._pending: list[tuple[str, Any]] = []
        self._stop = threading.Event()
        self._verify_cache: dict[str, dict] = {}
        self._compare_cache: dict[str, Comparison] = {}
        self._verify_running: set[str] = set()
        self.last_verify: dict | None = None
        self._adam = None
        # replay
        self.replay_i = 0
        self.replay_play = False
        self._replay_tick = 0
        # head-to-head. `_parked` is how the timing thread knows the worker has
        # actually stopped -- asking it to stop and assuming it did would time
        # the first step or two against an optimiser iteration still in flight.
        self.compare: Comparison | None = None
        self._parked = threading.Event()
        self._build(reset=True)
        self.worker = threading.Thread(target=self._loop, name="demo-sim",
                                       daemon=True)

    # -- construction ------------------------------------------------------

    def _build(self, reset: bool) -> None:
        self.case = self.cfg.case()
        cap = capacity(self.case)
        if self.cfg.k > cap:
            self.notice = (f"this farm size holds {cap} turbines at "
                           f"{self.cfg.s_min:g} D spacing, so the count was "
                           f"clamped from {self.cfg.k} to {cap}")
            self.cfg = DemoConfig(**{**asdict(self.cfg), "k": cap}).clamped()
            self.case = self.cfg.case()
        self.ro = DemoRollout(self.case, u_inf=self.cfg.u_inf,
                              inflow_deg=self.cfg.inflow_deg,
                              device=self.cfg.device)
        self.stride = 2 if self.case.shape[1] > 500 else 1
        if reset:
            self.theta = wd.project_design(layout_grid(self.case), self.case)
            self.u, self.v = self.ro.freestream()
            self.power_trace = []
            self.history = []
            self.opt_iter = 0
            self.replay_i = 0
            self.replay_play = False
            self._adam = None

    def _theta_t(self, theta: np.ndarray | None = None):
        """theta as a tensor ON THE ROLLOUT'S DEVICE.

        Built without `device=` this is a CPU tensor, and every op it touches on
        a CUDA run raises. That is not hypothetical -- it is the bug that took
        out the first GPU confirmation march of the PoC.
        """
        th = self.theta if theta is None else theta
        return torch.as_tensor(th, dtype=wd.TORCH_DTYPE, device=self.ro.device)

    def start(self) -> None:
        if not self.worker.is_alive():
            self.worker.start()

    def stop(self) -> None:
        self._stop.set()

    # -- the message queue -------------------------------------------------

    def post(self, kind: str, payload: Any = None) -> None:
        with self.lock:
            self._pending.append((kind, payload))

    def _peek_mode(self) -> str | None:
        """The mode the user has asked for but the worker has not reached yet.

        Messages are only drained BETWEEN iterations, and an optimiser iteration
        is the longest thing this engine does -- so without this, Stop appears
        to do nothing for however long the gradient takes. Peeking rather than
        consuming leaves the message in the queue to be applied normally.
        """
        with self.lock:
            for kind, payload in reversed(self._pending):
                if kind == "mode":
                    return payload
        return None

    def _drain(self) -> None:
        with self.lock:
            msgs, self._pending = self._pending, []
        for kind, payload in msgs:
            try:
                self._apply(kind, payload)
            except Exception as exc:                       # pragma: no cover
                self.notice = f"{kind} failed: {exc}"

    def _apply(self, kind: str, payload: Any) -> None:
        if kind == "theta":
            th = np.array(payload, dtype=float).reshape(-1)
            if th.size == self.theta.size:
                self.theta = wd.project_design(th, self.case)
                self._adam = None          # the moment differs; forget momentum
        elif kind == "config":
            old = self.cfg
            self.cfg = DemoConfig(**{**asdict(old), **payload}).clamped()
            hard = (self.cfg.domain != old.domain or self.cfg.k != old.k)
            if hard:
                self._build(reset=True)
                self.notice = "domain or turbine count changed: field reset"
            else:
                self.case = self.cfg.case()
                self.ro.set_inflow(self.cfg.u_inf, self.cfg.inflow_deg)
                self.theta = wd.project_design(self.theta, self.case)
        elif kind == "layout":
            fn = LAYOUTS.get(payload, layout_grid)
            self.theta = wd.project_design(fn(self.case), self.case)
            self._adam = None
        elif kind == "reset":
            self._build(reset=True)
            self.mode = "run"
            self.notice = "reset to the default grid layout"
        elif kind == "reseed":
            self.u, self.v = self.ro.freestream()
            self.power_trace = []
            self.notice = "flow field re-seeded from the freestream"
        elif kind == "mode":
            self.mode = payload
            if payload != "replay":
                self.replay_play = False
        elif kind == "scrub":
            self._seek(int(payload))
        elif kind == "replay":
            p = payload if isinstance(payload, dict) else {}
            if "index" in p:
                self._seek(int(p["index"]))
            if "play" in p:
                self._set_replay_play(bool(p["play"]))

    # -- replay ------------------------------------------------------------

    def _seek(self, i: int, announce: bool = True) -> None:
        """Put back a layout the optimiser actually visited, and hold there.

        The field is NOT restored -- only theta is recorded, and a stored flow
        field per iteration would be 1.3 MB a step. So the replay re-marches:
        the turbines jump to where they were at step i and the wake re-forms
        around them, which is why `REPLAY_HOLD` exists.
        """
        if not self.history:
            self.notice = "nothing to replay yet -- press Optimise first"
            return
        i = max(0, min(int(i), len(self.history) - 1))
        self.replay_i = i
        h = self.history[i]
        self.theta = wd.project_design(np.array(h["theta"], dtype=float), self.case)
        self._adam = None                     # the moment belongs to another theta
        self.mode = "replay"
        if announce:
            self.notice = (f"replay: the layout at optimiser step {h['iter']} of "
                           f"{self.history[-1]['iter']}")

    def _set_replay_play(self, play: bool) -> None:
        if play and not self.history:
            self.notice = "nothing to replay yet -- press Optimise first"
            return
        if play and self.replay_i >= len(self.history) - 1:
            self._seek(0, announce=False)     # at the end: play from the start
        self.replay_play = bool(play)
        self._replay_tick = 0
        if play:
            self.mode = "replay"
            self.notice = "replaying the optimiser's trajectory"
        elif self.mode == "replay":
            h = self.history[self.replay_i]
            self.notice = f"replay paused at optimiser step {h['iter']}"

    def _replay_step(self) -> None:
        """Advance the cursor every `REPLAY_HOLD` macro-steps, and always march.

        Marching while paused is deliberate: pausing on a layout lets its wakes
        settle, which is the only way to see what that layout was actually
        doing rather than what it looked like in passing.
        """
        if self.replay_play and self.history:
            self._replay_tick += 1
            if self._replay_tick >= REPLAY_HOLD:
                self._replay_tick = 0
                if self.replay_i >= len(self.history) - 1:
                    self.replay_play = False
                    self.notice = "replay finished at the last optimiser step"
                else:
                    self._seek(self.replay_i + 1, announce=False)
        self._march_once()

    # -- the loop ----------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._drain()
            try:
                if self._timing_now():
                    # A head-to-head is being TIMED. Marching here would put the
                    # live simulation in competition with the thing measuring
                    # it, and a per-step cost measured against a moving load is
                    # not a measurement. The field is still published, so the
                    # page keeps its last frame instead of going blank.
                    self._parked.set()
                    self._publish()
                    time.sleep(0.25)
                    continue
                self._parked.clear()
                if self.mode in ("optimize", "step"):
                    self._optimise_once()
                    if self.mode == "step":
                        self.mode = "paused"
                elif self.mode == "run":
                    self._march_once()
                elif self.mode == "replay":
                    self._replay_step()
                else:
                    self._publish()
                    time.sleep(0.05)
            except Exception as exc:                       # pragma: no cover
                self.notice = f"simulation error: {exc}"
                self.mode = "paused"
                time.sleep(0.2)

    def _timing_now(self) -> bool:
        c = self.compare
        return bool(c is not None and c.state == "running" and c.paused_live)

    def _march_once(self) -> None:
        t0 = time.perf_counter()
        th = self._theta_t()
        with torch.no_grad():
            u, v, power, _un = self.ro.macro_step(self.u, self.v, th)
        self.u, self.v = u, v
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self.fwd_ms = _ema(self.fwd_ms, self.step_ms)
        self.power_trace.append(float(power.sum()))
        self._publish()

    def _optimise_once(self) -> None:
        """One Adam step, differentiated through the next `horizon` macro-steps.

        The rollout is the live march: every macro-step inside it publishes a
        frame, so the field animates while the gradient is being taken rather
        than freezing until it lands.
        """
        t0 = time.perf_counter()
        cfg = self.cfg
        th = self._theta_t().requires_grad_(True)
        u, v = self.u, self.v
        powers = []
        for _ in range(cfg.horizon):
            u, v, power, _un = torch.utils.checkpoint.checkpoint(
                self.ro.macro_step, u, v, th, use_reentrant=False)
            powers.append(power)
            self.power_trace.append(float(power.sum().detach()))
            self.u, self.v = u.detach(), v.detach()
            self._publish()
            # Stop, mid-flight. The macro-steps already taken are kept -- they
            # are the live march either way -- but no gradient is applied, so
            # the layout is exactly where the user stopped it.
            if self._peek_mode() not in (None, "optimize", "step"):
                self.notice = ("stopped part-way through an optimiser step; "
                               "the layout is unchanged")
                self.step_ms = (time.perf_counter() - t0) * 1000.0
                return
        avg = min(3, len(powers))
        j = torch.stack([p.sum() for p in powers[-avg:]]).mean()
        j = j - wd.spacing_penalty(th, self.case)
        g, = torch.autograd.grad(j, th)
        grad = g.detach().cpu().numpy()
        self.grad_norm = float(np.linalg.norm(grad))
        self.theta = self._adam_step(self.theta, grad)
        self.u, self.v = u.detach(), v.detach()
        self.opt_iter += 1
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self.opt_ms = _ema(self.opt_ms, self.step_ms)
        self.history.append({
            "iter": self.opt_iter, "J": float(j.detach()),
            "power": float(sum(float(p.sum().detach()) for p in powers) / len(powers)),
            "grad_norm": self.grad_norm,
            "theta": [float(x) for x in self.theta],
            "wall_s": self.step_ms / 1000.0,
        })
        del self.history[:-400]
        self.replay_i = len(self.history) - 1     # the scrubber follows the live run
        self._publish()

    def _adam_step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:
        b1, b2, eps = 0.9, 0.999, 1e-8
        if self._adam is None or self._adam[0].shape != theta.shape:
            self._adam = (np.zeros_like(theta), np.zeros_like(theta), 0)
        m, v, t = self._adam
        t += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        lr = np.empty_like(theta)
        lr[0::3] = self.cfg.lr_pos
        lr[1::3] = self.cfg.lr_pos
        lr[2::3] = self.cfg.lr_yaw
        step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        self._adam = (m, v, t)
        return wd.project_design(theta + step, self.case)     # ASCENT

    # -- what the client sees ---------------------------------------------

    #: A gradient macro-step costs this many forward ones. Only used to turn a
    #: forward-step measurement into an optimiser estimate BEFORE the first
    #: optimiser iteration has been timed; after that the measurement wins.
    GRAD_OVER_FWD = 5.5

    def eta_s(self) -> float:
        """Seconds per optimiser iteration, measured on this machine if possible."""
        if self.opt_ms is not None:
            return self.opt_ms / 1000.0
        if self.fwd_ms is not None:
            return self.cfg.horizon * self.GRAD_OVER_FWD * self.fwd_ms / 1000.0
        return self.cfg.horizon_eta_s

    def _publish(self) -> None:
        u = self.u.detach().cpu().numpy()
        umax = float(np.abs(u).max())
        png = field_png(u, self.stride)
        lx, ly = self.case.extent
        th = self.theta.reshape(-1, 3)
        power = self.power_trace[-1] if self.power_trace else 0.0
        self.seq += 1
        payload = {
            "seq": self.seq,
            "mode": self.mode,
            "config": asdict(self.cfg),
            "domain_D": [lx, ly],
            "grid": [int(self.case.shape[1]), int(self.case.shape[0])],
            "stride": self.stride,
            "turbines": [{"x": float(a), "y": float(b),
                          "yaw_deg": math.degrees(float(c))} for a, b, c in th],
            "power": power,
            "power_ideal": float(self.cfg.k * 0.5 * wd.C_T_PRIME
                                 * self.cfg.u_inf ** 3),
            "power_trace": self.power_trace[-240:],
            "opt_iter": self.opt_iter,
            "grad_norm": self.grad_norm,
            "step_ms": self.step_ms,
            "u_max": umax,
            "n_windows": int(self.ro.n_win),
            "capacity": int(capacity(self.case)),
            "history_len": len(self.history),
            "history": [{"iter": h["iter"], "power": h["power"], "J": h["J"]}
                        for h in self.history[-240:]],
            "replay": {"index": int(self.replay_i), "playing": bool(self.replay_play),
                       "n": len(self.history), "hold": REPLAY_HOLD,
                       "iter": (self.history[self.replay_i]["iter"]
                                if self.history and
                                self.replay_i < len(self.history) else 0)},
            "device": self.cfg.device,
            "compare": self.compare.brief() if self.compare else None,
            "validity": validity(self.cfg, self.case, self.theta, umax),
            "notice": self.notice,
            "ramp": {"lo": U_LO, "hi": U_HI, "stops": RAMP_STOPS},
            "eta_s": self.eta_s(),
            "eta_measured": self.opt_ms is not None or self.fwd_ms is not None,
            "fwd_ms": self.fwd_ms,
            "verify": self._verify_view(),
        }
        self.frame = Frame(seq=self.seq, png=png,
                           width=int(self.case.shape[1] // self.stride),
                           height=int(self.case.shape[0] // self.stride),
                           stride=self.stride, payload=payload)

    # -- verification ------------------------------------------------------

    def layout_hash(self, theta=None, cfg=None) -> str:
        cfg = cfg or self.cfg
        theta = self.theta if theta is None else theta
        blob = json.dumps({
            "d": cfg.domain, "k": cfg.k, "u": round(cfg.u_inf, 6),
            "a": round(cfg.inflow_deg, 4), "n": cfg.verify_steps,
            "t": [round(float(x), 5) for x in theta],
        }, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _verify_view(self) -> dict:
        """The check for the CURRENT layout, plus the last one that finished.

        A verification is keyed by layout, and the optimiser moves the layout
        every few seconds -- so a result requested mid-run stops matching the
        moment it lands, and the panel reverted to "idle" with the user's answer
        thrown away. `last` keeps it, labelled with the optimiser step it
        belongs to and marked stale, which is the honest thing to show: the
        number is real, it is just not about what is on the screen now.
        """
        h = self.layout_hash()
        view: dict = {"hash": h}
        if h in self._verify_cache:
            view.update(state="done", **self._verify_cache[h])
        elif h in self._verify_running:
            view["state"] = "running"
        else:
            view["state"] = "idle"
        if self.last_verify:
            view["last"] = {**self.last_verify,
                            "stale": self.last_verify.get("hash") != h}
        return view

    def request_verify(self, force: bool = False, pause_live: bool = True) -> str:
        """Start (or re-show) the head-to-head for the layout on screen.

        `force=False` is the main window's button: a layout already checked
        comes straight back out of the cache, which is what makes asking twice
        instant.  `force=True` is the second window's, because there the point
        IS the timing and a cached number is not a fresh measurement.
        """
        h = self.layout_hash()
        if not force and h in self._compare_cache:
            self.compare = self._compare_cache[h]
            return h
        if h in self._verify_running:
            return h
        if not force and h in self._verify_cache:
            return h
        self._verify_running.add(h)
        cmp_ = Comparison(hash=h, state="running", steps=int(self.cfg.verify_steps),
                          theta=[float(x) for x in self.theta],
                          at_iter=int(self.opt_iter), k=int(self.cfg.k),
                          n_windows=int(self.case.n_windows),
                          cells=(int(self.case.shape[1]), int(self.case.shape[0])),
                          domain_D=tuple(float(x) for x in self.case.extent),
                          device=self.cfg.device, paused_live=bool(pause_live),
                          started=time.time())
        self.compare = cmp_
        threading.Thread(target=self._verify,
                         args=(h, self.cfg, self.theta.copy(), self.opt_iter, cmp_),
                         name=f"verify-{h}", daemon=True).start()
        return h

    def _verify(self, h: str, cfg: DemoConfig, theta: np.ndarray,
                at_iter: int, cmp_: Comparison) -> None:
        """Composed column vs the undivided classical solver, same layout, timed.

        Both march from the freestream for `verify_steps` macro-steps, which is
        the PoC's protocol rather than this demo's warm-started one -- so the
        composed number here is comparable with the paper's, and the classical
        one is the referent it should be read against.

        They alternate, one macro-step each, for the reason `Comparison`
        documents: a per-step cost measured while the other solver is running is
        a measurement of the contention, not of the solver.  For the same reason
        this waits for the live march to actually stop before it starts timing
        anything -- posting the mode change is not the same as the worker having
        acted on it.

        Its own solvers and its own fields: this touches no engine state except
        the comparison it was handed and the caches.
        """
        out: dict = {}
        try:
            if cmp_.paused_live and self.worker.is_alive():
                # Setting `compare` already tells the worker to park; this waits
                # for it to have actually done so, which can take one optimiser
                # iteration (tens of seconds on a large domain with a long
                # horizon). If it never parks, the run still happens -- but it
                # is relabelled as contended rather than reported as a clean
                # timing it is not.
                if not self._parked.wait(120.0):           # pragma: no cover
                    cmp_.paused_live = False
                time.sleep(0.15)              # let the step in flight retire

            case = cfg.case()
            n = int(cfg.verify_steps)
            stride = 2 if case.shape[1] > 500 else 1

            #: The composed side runs where the user put it.  The disk forcing
            #: fed to the CLASSICAL side must be built on the CPU whatever that
            #: is, because the monolith is numpy: shuttling its field to a GPU
            #: and back every step would be timing the transfer, not the solver.
            ro = DemoRollout(case, u_inf=cfg.u_inf, inflow_deg=cfg.inflow_deg,
                             device=cfg.device)
            ro_cpu = ro if cfg.device == "cpu" else DemoRollout(
                case, u_inf=cfg.u_inf, inflow_deg=cfg.inflow_deg, device="cpu")
            th = torch.as_tensor(theta, dtype=wd.TORCH_DTYPE, device=ro.device)
            th_cpu = th if ro_cpu is ro else torch.as_tensor(
                theta, dtype=wd.TORCH_DTYPE)

            u, v = ro.freestream()
            mono = sl.reference_monolith(case.tiling.nx, case.tiling.ny, wa.NU_REF)
            uu = np.full(case.shape, ro_cpu.free_u)
            vv = np.full(case.shape, ro_cpu.free_v)
            band = ro_cpu._band.cpu().numpy()

            composed_s = classical_s = 0.0
            for i in range(n):
                # ---- the coupled graph, one macro-step, alone on the machine
                t0 = time.perf_counter()
                with torch.no_grad():
                    u, v, power, _ = ro.macro_step(u, v, th)
                if cfg.device != "cpu":
                    torch.cuda.synchronize()  # or the timer measures the queue
                dt_c = time.perf_counter() - t0
                composed_s += dt_c

                # ---- the undivided classical solver, the same macro-step
                t0 = time.perf_counter()
                tu = torch.as_tensor(uu, dtype=wd.TORCH_DTYPE)
                tv = torch.as_tensor(vv, dtype=wd.TORCH_DTYPE)
                fx, fy, _un, _T, p = ro_cpu.disks.forcing(tu, tv, th_cpu)
                u1, v1 = mono.step_batch(
                    uu[None], vv[None], wa.MACRO_DT, bc0=None,
                    force=(fx.numpy()[None], fy.numpy()[None]))
                uu, vv = u1[0], v1[0]
                uu = np.where(band, ro_cpu.free_u, uu)
                vv = np.where(band, ro_cpu.free_v, vv)
                dt_k = time.perf_counter() - t0
                classical_s += dt_k

                if not np.all(np.isfinite(uu)):
                    raise RuntimeError("the classical monolith left the band")

                # ---- everything below is OUTSIDE both timers
                un = u.detach().cpu().numpy()
                cmp_.composed_power.append(float(power.sum()))
                cmp_.classical_power.append(float(p.sum()))
                cmp_.composed_ms.append(dt_c * 1000.0)
                cmp_.classical_ms.append(dt_k * 1000.0)
                cmp_.linf.append(float(np.abs(un - uu).max()))
                cmp_.png["composed"] = field_png(un, stride)
                cmp_.png["classical"] = field_png(uu, stride)
                cmp_.width = int(case.shape[1] // stride)
                cmp_.height = int(case.shape[0] // stride)
                cmp_.i = i + 1
                cmp_.seq += 1

            trace, mtrace = cmp_.composed_power, cmp_.classical_power
            composed = float(np.mean(trace[-5:]))
            classical = float(np.mean(mtrace[-5:]))
            cs, ks = _ms_stats(cmp_.composed_ms), _ms_stats(cmp_.classical_ms)

            out = {
                "composed_power": composed, "classical_power": classical,
                "delta": composed - classical,
                "delta_pct": (100.0 * (composed - classical) / classical
                              if abs(classical) > 1e-12 else None),
                "composed_wall_s": composed_s, "classical_wall_s": classical_s,
                "speedup": classical_s / composed_s if composed_s > 0 else None,
                "composed_ms_step": cs["median_ms"], "classical_ms_step": ks["median_ms"],
                "ms_ratio": (cs["median_ms"] / ks["median_ms"]
                             if ks["median_ms"] else None),
                "composed_ms": cmp_.composed_ms, "classical_ms": cmp_.classical_ms,
                "linf": cmp_.linf, "linf_final": cmp_.linf[-1] if cmp_.linf else None,
                "device": cfg.device, "timed_alone": bool(cmp_.paused_live),
                "steps": n, "k": cfg.k, "n_windows": case.n_windows,
                "composed_trace": list(trace), "classical_trace": list(mtrace),
                "note": ("Both marched from still air for the same number of "
                         "steps, alternating so neither was competing with the "
                         "other for cores. The classical side is "
                         "scaling_ladder.reference_monolith: the SAME "
                         "discretization with no cut, so the difference between "
                         "the two numbers is the cost of splitting the domain "
                         "up and re-assembling it, not a disagreement about the "
                         "physics."),
            }
            cmp_.state = "done"
        except Exception as exc:
            out = {"error": f"{type(exc).__name__}: {exc}"}
            cmp_.error = out["error"]
            cmp_.state = "error"
        finally:
            cmp_.finished = time.time()
            out["hash"] = h
            out["at_iter"] = int(at_iter)
            out["theta"] = [float(x) for x in theta]
            self._verify_cache[h] = out
            if "error" not in out:
                self.last_verify = out
                self._compare_cache[h] = cmp_
                del_keys = list(self._compare_cache)[:-6]      # keep the images bounded
                for k in del_keys:
                    self._compare_cache.pop(k, None)
            self._verify_running.discard(h)
            # `state` is no longer "running", so `_timing_now()` goes false and
            # the worker picks up exactly where it left off -- no mode to restore
            # and therefore no way to restore the wrong one.
            self._parked.clear()
