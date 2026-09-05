"""PoC 2's demo engine: a live front-wing march you can watch and optimise.

This sits ON TOP of `atlas.cases.front_wing` and changes nothing in it.
Everything here is a call into that module; nothing about the physics, the
composition layer or the interface system lives in this file.  `atlas/demo/`'s
engine is the template and three of its rules are inherited verbatim:

  * **the engine measures itself and this file quotes no fixed speed.**  A
    wall-clock number is a claim about a machine in a power state -- the PoC 1a
    demo shipped a table that was wrong by 4x the next time anyone looked -- so
    `Engine.eta_s` reports a moving average of what steps have actually cost
    here and the screen marks it as an estimate until the first real one lands;
  * **the optimiser's rollout IS the live march**, warm-started and short, so the
    field animates inside the gradient step rather than freezing until it lands.
    That is a different estimator from `w141_poc2_frontwing.py`'s and the screen
    says so;
  * **nothing here owns a verdict.**  The certification panel is
    `compile_scheme`'s own output, grouped per seam by
    `w141_poc2_frontwing.seam_verdicts`, and this file's only job is to run the
    compile on a worker and hand the answer to the page.

What is new, and it is the point of this demo
---------------------------------------------

**A per-seam certified indicator, driven by the compiler and not by a rule of
thumb.**  Every one of the nine seams gets its own verdict -- `admit`,
`admit-uncertified` or `refuse` -- from the decisions whose subject reaches it,
and the page draws green/amber/red from that.  Two things it must not overstate,
both measured in `w141`'s `compile` stage and both said on screen:

1. **The verdict is a property of the DECLARATION, not of the design point.**
   Over the sixteen corners of the design box, zero produce a different per-seam
   verdict map.  So the colours change when the interface is declared to move --
   the two surface seams go red at `L2/InterfaceMotion` and the seven
   fluid-fluid ones stay amber -- and they do NOT flicker as a knob turns.  The
   compile is still re-run on every design change, on a worker thread, because
   the honest way to show that is to run it rather than to assume it.
2. **What moves with the design is the quantity underneath.**  The one-sidedness
   of each seam, the interface residual and the two constraint margins are live,
   and they are what a designer would actually watch.
"""
from __future__ import annotations

import io
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from ..cases import front_wing as F
from ..cases import wing_fsi as W
from ..compiler import compile_scheme

# ---------------------------------------------------------------------------
# the colour ramp for the flow field -- `atlas/demo/engine.py`'s, unchanged
# ---------------------------------------------------------------------------

U_LO, U_HI = 0.0, 1.6
RAMP_STOPS = (
    (0.00, (12, 20, 48)), (0.25, (26, 76, 128)), (0.50, (44, 148, 160)),
    (0.72, (150, 200, 130)), (0.88, (240, 214, 110)), (1.00, (250, 250, 235)),
)

#: The von Mises ramp is SEPARATE and it is anchored on the CEILING, not on the
#: frame's own maximum.  A ramp that renormalises every frame makes a design at
#: 20% of the ceiling look identical to one at 99% of it, which is the one thing
#: this panel exists to distinguish.
STRESS_STOPS = (
    (0.00, (40, 54, 84)), (0.35, (60, 130, 170)), (0.62, (140, 190, 120)),
    (0.85, (238, 190, 80)), (1.00, (226, 74, 60)),
)


def _lut(stops) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(1, len(stops)):
            a, ca = stops[k - 1]
            b, cb = stops[k]
            if t <= b:
                f = 0.0 if b == a else (t - a) / (b - a)
                lut[i] = [round(ca[j] + f * (cb[j] - ca[j])) for j in range(3)]
                break
        else:
            lut[i] = stops[-1][1]
    return lut


_FIELD_LUT = _lut(RAMP_STOPS)
_STRESS_LUT = _lut(STRESS_STOPS)


def stress_colour(vm: float, ceiling: float = None) -> str:
    ceiling = F.SIGMA_CEIL if ceiling is None else ceiling
    t = float(np.clip(vm / max(ceiling, 1e-30), 0.0, 1.0))
    r, g, b = _STRESS_LUT[int(t * 255)]
    return f"#{r:02x}{g:02x}{b:02x}"


def field_png(u: np.ndarray, v: np.ndarray, stride: int = 1) -> bytes:
    """Colour-map the speed and encode it as a PNG.

    Row 0 of the array is y = 0 and an image's row 0 is the top, so the array is
    flipped here; the client draws it with y up, which is how a wing over a floor
    is read.
    """
    from PIL import Image
    a = np.hypot(u, v)[::stride, ::stride]
    t = np.clip((a - U_LO) / (U_HI - U_LO), 0.0, 1.0)
    rgb = _FIELD_LUT[(t * 255.0).astype(np.uint8)][::-1]
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the configuration
# ---------------------------------------------------------------------------


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _ema(prev, x, a: float = 0.25):
    return float(x) if prev is None else float((1 - a) * prev + a * x)


@dataclass
class DemoConfig:
    #: the composed column by default; `single` is the referent, and switching
    #: between them live is what makes the cut's cost visible rather than quoted
    tiling: str = "six"
    coupling: str = "tight"
    lag: int = 1
    #: the optimiser's rollout length.  SHORT and warm-started: see the module
    #: docstring, and the screen says which estimator it is
    horizon: int = 12
    lr: float = 0.05
    penalty: float = 40.0
    stride: int = 1
    spin: int = 40

    def clamped(self) -> "DemoConfig":
        return DemoConfig(
            tiling=self.tiling if self.tiling in ("six", "single") else "six",
            coupling=(self.coupling
                      if self.coupling in ("tight", "lagged", "split") else "tight"),
            lag=int(_clamp(self.lag, 1, 32)),
            horizon=int(_clamp(self.horizon, 2, 60)),
            lr=float(_clamp(self.lr, 1e-3, 0.3)),
            penalty=float(_clamp(self.penalty, 0.0, 500.0)),
            stride=int(_clamp(self.stride, 1, 4)),
            spin=int(_clamp(self.spin, 0, 400)))


@dataclass
class Frame:
    seq: int = 0
    png: bytes = b""
    width: int = 0
    height: int = 0
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# the certification panel
# ---------------------------------------------------------------------------

GRAPH_SUBJECTS = ("<graph>", "<assembly>", "<run>", "<scheme>")


def seam_verdicts(graph, result) -> dict:
    """Every seam's own verdict, from the compiler's own decisions.

    Kept identical to `scripts/w141_poc2_frontwing.seam_verdicts` on purpose --
    a demo that groups the decisions differently from the driver is showing a
    different thing from the one the results page reports.  A test asserts the
    two agree on this graph.
    """
    from ..verdict import Verdict
    out: dict[str, dict] = {}
    for c in graph.connections:
        ports = {f"{c.a[0]}.{c.a[1]}", f"{c.b[0]}.{c.b[1]}"}
        agents = {c.a[0], c.b[0]}
        rows = {"refuse": [], "admit-uncertified": [], "global": []}
        for d in result.decisions.decisions:
            if d.verdict is Verdict.ADMIT:
                continue
            subs = {s.strip() for s in str(d.subject or "").split(",")}
            tag = f"{d.layer}/{d.rule}"
            if subs & ({c.seam_id} | ports | agents):
                rows[d.verdict.value].append(tag)
            elif subs & set(GRAPH_SUBJECTS):
                rows["global"].append(tag)
        verdict = ("refuse" if rows["refuse"]
                   else "admit-uncertified"
                   if (rows["admit-uncertified"] or rows["global"]) else "admit")
        out[c.seam_id] = dict(
            verdict=verdict,
            colour={"refuse": "red", "admit-uncertified": "amber",
                    "admit": "green"}[verdict],
            kind=("fluid-structure" if c.seam_id == "wet"
                  else "field-lumped" if c.seam_id == "mount"
                  else "fluid-fluid"),
            a=f"{c.a[0]}.{c.a[1]}", b=f"{c.b[0]}.{c.b[1]}",
            refusals=sorted(set(rows["refuse"])),
            decertifications=sorted(set(rows["admit-uncertified"])),
            global_decertifications=sorted(set(rows["global"])))
    return out


#: What each rule means, in one line, because a panel that shows `L2/R10/halo`
#: and nothing else is a panel nobody can act on.
RULE_NOTE = {
    "L2/InterfaceMotion": "the interface's geometry is a function of the "
                          "solution and no rule certifies a cached operator "
                          "across that motion. CS-10 priced it: at a 2% "
                          "staleness tolerance the cache does not survive one "
                          "exchange",
    "L2/R10": "an agent declares an incompressible family with no elliptic "
              "sub-solve, or embeds one the decomposition cut. Narrowed at "
              "W114 to check its own premise",
    "L2/R10/halo": "the agent is implicit with a nonzero stencil, so "
                   "stencil x substeps is not its domain of dependence and the "
                   "halo is undecidable. W136: it over-fires here, because this "
                   "seam is a PHYSICAL boundary with no overlap to outrun",
    "L2/C2": "the cut is admissible and is not certified optimal",
    "L4/E7/passivity": "the symmetric part of the assembled response is "
                       "indefinite. W138: at this seam that is an ORIENTATION "
                       "artefact -- the two blocks are added where the "
                       "interface residual subtracts them -- and flipping one "
                       "sign takes the defect to exactly zero",
    "L5/eps_tol": "the accelerator's tolerance rests on a defect term that is "
                  "identically zero here",
    "L6/R12": "the partition of unity's contaminated set bounds sigma and the "
              "bound is not certified",
    "L6/W49": "C_mu is unmeasured",
    "L9/E5": "the run-level claim has no measured L (W1)",
    "L8/W56": "at least one bound constant is unmeasured, which has forced "
              "admit-uncertified on every graph in this package since W56",
}


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


class Engine:
    """One persistent coupled state, marched or optimised on a worker thread."""

    def __init__(self, cfg: DemoConfig | None = None) -> None:
        self.cfg = (cfg or DemoConfig()).clamped()
        self.design = dict(F.DESIGN_REF)
        self.mode = "run"
        self.notice = ""
        self.frame = Frame()
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self.step_ms = None
        self.fwd_ms = None
        self.opt_ms = None
        self.iteration = 0
        self.history: list[dict] = []
        self.field_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "out", "w141", "settled.npz")
        self._op_cache: dict[float, dict] = {}
        self._build(reset=True)
        # the certification panel, computed on its own worker
        self.cert: dict = {"state": "pending", "seams": {}, "verdict": None}
        self._cert_lock = threading.Lock()
        self._cert_thread: threading.Thread | None = None
        self._cert_want = 0
        self._cert_done = -1
        self.request_compile()

    # -- state -------------------------------------------------------------

    def _build(self, reset: bool) -> None:
        tiling = (F.SINGLE_TILING if self.cfg.tiling == "single"
                  else F.DEFAULT_TILING)
        self.ro = F.FrontWingRollout(tiling=tiling, coupling=self.cfg.coupling,
                                     lag=self.cfg.lag, motion=True,
                                     design=self.design)
        if not reset:
            return
        opt = dict(dtype=F.TORCH_DTYPE)
        if os.path.isfile(self.field_path):
            d = np.load(self.field_path)
            self.u = torch.as_tensor(d["u"], **opt)
            self.v = torch.as_tensor(d["v"], **opt)
            self.notice = "released from the settled fixed-shape field"
        else:
            self.u = torch.full((self.ro.ny, self.ro.nx), F.U_INF, **opt)
            self.v = torch.zeros((self.ro.ny, self.ro.nx), **opt)
            self.notice = ("no settled field on disk: released from the "
                           "freestream, which is not the state any number on "
                           "the results page was measured at")
        self.delta = torch.zeros(F.N_STATION, **opt)
        kn = self.ro.knobs()
        self.h = F.FrontWingRollout.release_height(kn["h0"], kn["k"]).clone()
        self.w_delta = torch.zeros(F.N_STATION, **opt)
        self.v_mount = torch.zeros((), **opt)
        self.load = self.drag = self.vm = 0.0
        self.trace: list[dict] = []
        self.iteration = 0

    # -- the message queue --------------------------------------------------

    def post(self, kind: str, payload: Any = None) -> None:
        self._q.put((kind, payload))

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self._q.get_nowait()
            except queue.Empty:
                return
            self._apply(kind, payload)

    def _apply(self, kind: str, payload: Any) -> None:
        if kind == "mode":
            self.mode = payload
        elif kind == "design":
            d = dict(self.design)
            for k, v in (payload or {}).items():
                if k in F.DESIGN_BOX:
                    d[k] = float(_clamp(float(v), *F.DESIGN_BOX[k]))
            if d != self.design:
                self.design = d
                self._build(reset=False)
                # the ride height is a STATE, so a change of h0 or k does not
                # teleport it -- the suspension moves it, which is the coupling
                self.request_compile()
        elif kind == "config":
            cur = asdict(self.cfg)
            cur.update({k: v for k, v in (payload or {}).items() if k in cur})
            new = DemoConfig(**cur).clamped()
            reset = new.tiling != self.cfg.tiling
            self.cfg = new
            self._build(reset=reset)
            self.request_compile()
        elif kind == "reset":
            self._build(reset=True)
            self.request_compile()
        elif kind == "compile":
            self.request_compile()

    # -- the certification panel, on its own worker -------------------------

    def request_compile(self) -> None:
        with self._cert_lock:
            self._cert_want += 1
            want = self._cert_want
            self.cert = dict(self.cert, state="running")
        if self._cert_thread is None or not self._cert_thread.is_alive():
            self._cert_thread = threading.Thread(
                target=self._compile_loop, daemon=True)
            self._cert_thread.start()
        return want

    def _compile_loop(self) -> None:
        while not self._stop.is_set():
            with self._cert_lock:
                want, done = self._cert_want, self._cert_done
                design = dict(self.design)
            if want == done:
                return
            t0 = time.perf_counter()
            try:
                payload = self._compile_now(design)
                payload["wall_s"] = time.perf_counter() - t0
            except Exception as exc:                        # pragma: no cover
                payload = dict(state="error", error=str(exc)[:300], seams={})
            with self._cert_lock:
                self._cert_done = want
                self.cert = payload

    def _compile_now(self, design: dict) -> dict:
        u = self.u.detach().cpu().numpy()
        v = self.v.detach().cpu().numpy()
        h = float(self.h.detach())
        out = {}
        for tag, motion in (("riding", True), ("fixed-shape", False)):
            g, _e = F.build(u, v, motion=motion, design=design, h=h,
                            tiling=self.ro.tiling)
            r = compile_scheme(g)
            out[tag] = dict(verdict=r.verdict.value,
                            seams=seam_verdicts(g, r),
                            unmeasured=sorted(getattr(r, "unmeasured", ()) or ()))
        live = out["riding"]
        return dict(state="ok", verdict=live["verdict"], seams=live["seams"],
                    unmeasured=live["unmeasured"],
                    fixed_shape=out["fixed-shape"]["seams"],
                    fixed_shape_verdict=out["fixed-shape"]["verdict"],
                    design=design, notes=dict(RULE_NOTE))

    # -- the march ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self, join_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(join_s)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._drain()
            try:
                if self.mode in ("optimize", "step"):
                    self._optimise_once()
                    if self.mode == "step":
                        self.mode = "paused"
                elif self.mode == "run":
                    self._march_once()
                else:
                    self._publish()
                    time.sleep(0.05)
            except RuntimeError as exc:
                # an expert DECLINING is not the same as the scheme diverging,
                # and the two look identical from outside a try block -- CS-12
                # section 7.1's rule, on screen
                msg = str(exc)
                #: **`suspension` is tested FIRST**, because both envelope
                #: messages contain the word "envelope" and testing that first
                #: labelled every ride-height decline as a structural one
                kind = ("the suspension expert declined" if "suspension" in msg
                        else "the structural expert declined" if "envelope" in msg
                        else "the scheme went non-finite" if "finite" in msg
                        else "the march stopped")
                self.notice = f"{kind}: {msg[:220]}"
                self.mode = "paused"
                self._build(reset=True)
                self._publish()
                time.sleep(0.3)
            except Exception as exc:                        # pragma: no cover
                self.notice = f"simulation error: {exc}"
                self.mode = "paused"
                time.sleep(0.2)

    def _step_once(self, grad: bool = False):
        kn = self.ro.knobs()
        return self.ro.macro_step(self.u, self.v, self.delta, self.h,
                                  self.w_delta, self.v_mount,
                                  kn["e_star"], kn["tc"], kn["k"], kn["h0"],
                                  self.iteration)

    def _march_once(self) -> None:
        t0 = time.perf_counter()
        with torch.no_grad():
            (u, v, delta, h, wd, vm_, load, drag, f, vm,
             res, r0, _p, _hs) = self._step_once()
        self._absorb(u, v, delta, h, wd, vm_, load, drag, f, vm, res)
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self.fwd_ms = _ema(self.fwd_ms, self.step_ms)
        self._publish()

    def _absorb(self, u, v, delta, h, wd, vm_, load, drag, f, vm, res) -> None:
        #: **W145, on the demo's own paths.** Both `_march_once` and
        #: `_optimise_once` drive `macro_step` directly, so neither went through
        #: `run`'s envelope checks -- the live optimiser could walk the wing
        #: through the suspension's declared floor and the screen would show a
        #: downforce for a design the model declines. Every path funnels through
        #: `_absorb`, so the check goes here, once. `_loop` already catches the
        #: `RuntimeError` and puts the reason on screen.
        self.ro.check_envelopes(self.iteration, delta.detach(),
                                float(h.detach()),
                                float(self.design["h0"]))
        self.u, self.v = u.detach(), v.detach()
        self.delta = delta.detach()
        self.h = h.detach()
        self.w_delta = wd.detach()
        self.v_mount = vm_.detach()
        self.load = float(load.detach())
        self.drag = float(drag.detach())
        self.vm = float(vm.detach())
        self.traction = f.detach()
        self.residual = float(res.detach())
        self.iteration += 1
        self.trace.append(dict(i=self.iteration, load=self.load, drag=self.drag,
                               h=float(self.h), tip=float(self.delta[-1]),
                               vm=self.vm))
        if len(self.trace) > 600:
            del self.trace[:200]

    def _optimise_once(self) -> None:
        """One Adam step, differentiated through the next `horizon` macro-steps.

        The rollout IS the live march: it starts from the state on screen and
        the state it ends at is the state that stays, so the field animates
        through the gradient rather than freezing until it lands.  That makes it
        a warm-started short-horizon estimator and NOT the one
        `w141_poc2_frontwing.py` measures with -- the screen says so.
        """
        t0 = time.perf_counter()
        opt = dict(dtype=F.TORCH_DTYPE)
        leaves = {k: torch.tensor(float(self.design[k]), requires_grad=True, **opt)
                  for k in F.DESIGN_KEYS}
        u, v = self.u, self.v
        delta, h = self.delta, self.h
        wd, vm_ = self.w_delta, self.v_mount
        loads, vms, dmax = [], [], []
        for s in range(self.cfg.horizon):
            out = self.ro.macro_step(u, v, delta, h, wd, vm_,
                                     leaves["e_star"], leaves["tc"],
                                     leaves["k"], leaves["h0"],
                                     self.iteration + s)
            (u, v, delta, h, wd, vm_, load, drag, f, vm, res, r0, _p, _hs) = out
            loads.append(load); vms.append(vm)
            dmax.append(torch.max(torch.abs(delta)))
        J = torch.stack(loads).mean()
        sig, dl = torch.stack(vms), torch.stack(dmax)
        #: the SMOOTH margins drive the step and the TRUE ones are what the
        #: readout calls feasible -- `logsumexp` overshoots the maximum it
        #: smooths, and a constraint is not the place to leave that unmeasured
        g_sig = (F._softmax_over(sig, F.SIGMA_CEIL * F.SOFTMAX_FRAC)
                 / F.SIGMA_CEIL - 1.0)
        g_del = (F._softmax_over(dl, F.DELTA_CEIL * F.SOFTMAX_FRAC)
                 / F.DELTA_CEIL - 1.0)
        L = F._penalised(J, g_sig, g_del, self.cfg.penalty)
        grads = torch.autograd.grad(L, list(leaves.values()), allow_unused=True)
        self._absorb(u, v, delta, h, wd, vm_, load, drag, f, vm, res)
        self.margins = dict(
            smooth_sigma=float(g_sig), smooth_delta=float(g_del),
            true_sigma=float(sig.max()) / F.SIGMA_CEIL - 1.0,
            true_delta=float(dl.max()) / F.DELTA_CEIL - 1.0)
        # Adam, in the box's own unit cube so one learning rate serves knobs
        # whose scales differ by four orders
        st = getattr(self, "_adam", None)
        if st is None or st["n"] != len(leaves):
            st = self._adam = dict(m=np.zeros(len(leaves)),
                                   v=np.zeros(len(leaves)), t=0,
                                   n=len(leaves))
        span = np.array([F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0]
                         for k in F.DESIGN_KEYS])
        g = np.array([0.0 if x is None else float(x) for x in grads]) * span
        st["t"] += 1
        st["m"] = 0.9 * st["m"] + 0.1 * g
        st["v"] = 0.999 * st["v"] + 0.001 * g * g
        mh = st["m"] / (1 - 0.9 ** st["t"])
        vh = st["v"] / (1 - 0.999 ** st["t"])
        x = np.array([(self.design[k] - F.DESIGN_BOX[k][0]) / span[i]
                      for i, k in enumerate(F.DESIGN_KEYS)])
        x = np.clip(x + self.cfg.lr * mh / (np.sqrt(vh) + 1e-8), 0.0, 1.0)
        self.post("design", {k: F.DESIGN_BOX[k][0] + x[i] * span[i]
                             for i, k in enumerate(F.DESIGN_KEYS)})
        self.history.append(dict(t=st["t"], J=float(J),
                                 g_sigma=float(g_sig), g_delta=float(g_del),
                                 **dict(self.design)))
        if len(self.history) > 400:
            del self.history[:150]
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self.opt_ms = _ema(self.opt_ms, self.step_ms)
        self._publish()

    # -- what the page draws -------------------------------------------------

    def geometry(self) -> dict:
        """The plate's own quads and their von Mises, in DOMAIN coordinates.

        Built from `FlexWing.stations(delta, h)` -- the same call the physics
        makes -- so the picture is the model's geometry and not a second one.
        """
        wing = self.ro.wing
        d = self.delta.detach()
        h = self.h.detach()
        cx, cy = wing.stations(d, h)
        cx = cx.cpu().numpy(); cy = cy.cpu().numpy()
        nx_h, ny_h = float(wing.n_hat[0]), float(wing.n_hat[1])
        half = 0.5 * self.design["tc"] * F.CHORD
        vm = self._von_mises_stations()
        quads = []
        for k in range(len(cx) - 1):
            x0, y0, x1, y1 = cx[k], cy[k], cx[k + 1], cy[k + 1]
            quads.append(dict(
                p=[[x0 + half * nx_h, y0 + half * ny_h],
                   [x1 + half * nx_h, y1 + half * ny_h],
                   [x1 - half * nx_h, y1 - half * ny_h],
                   [x0 - half * nx_h, y0 - half * ny_h]],
                vm=float(vm[k]), colour=stress_colour(vm[k])))
        return dict(quads=quads, chord_x=[float(cx[0]), float(cx[-1])],
                    chord_y=[float(cy[0]), float(cy[-1])],
                    thickness=2 * half, n=len(quads))

    def _von_mises_stations(self) -> np.ndarray:
        """Per chordwise station: the max over the two elements through the
        thickness.  `sigma_map` is the expert's own answer to a unit station
        traction, so this re-enters no solver."""
        tc = float(self.design["tc"])
        key = round(tc, 12)
        op = self._op_cache.get(key)
        if op is None:
            op = self._op_cache[key] = F.surface_operator(tc)
            if len(self._op_cache) > 24:
                self._op_cache.pop(next(iter(self._op_cache)))
        q = getattr(self, "traction", None)
        if q is None:
            return np.zeros(F.N_STATION)
        sig = np.einsum("k,kea->ea", q.cpu().numpy(), op["sigma_map"])
        vm = F.von_mises(sig)
        return vm.reshape(F.N_STATION, -1).max(axis=1)

    def seam_quantities(self) -> dict:
        """What actually MOVES with the design, under a verdict that does not.

        Both are one-sidedness ratios -- the partner's block against the fluid's
        -- which is the quantity `SubstitutionCertificate` is blind in proportion
        to (W97, W137).  They are computed from the design and the current
        traction rather than probed, because a probe is seconds and this panel is
        per frame; the driver's `compile` stage probes them properly.
        """
        kn = self.ro.knobs()
        S_e, _m = self.ro.structure(kn["e_star"], kn["tc"])
        q = getattr(self, "traction", None)
        w = float(np.abs(q.cpu().numpy()).mean()) if q is not None else 0.0
        # the fluid's own aerodynamic damping at the seam, diag(C_N |w|)
        fluid = F.C_N * max(w, 1e-30) ** 0.5
        struct = float(torch.linalg.matrix_norm(S_e, 2))
        return {
            "wet": dict(label="structure / fluid, operator norm",
                        value=struct / max(fluid, 1e-30),
                        note="W137: the fluid's share of this seam is the "
                             "reciprocal, and SubstitutionCertificate is blind "
                             "in proportion to it"),
            "mount": dict(label="spring / fluid, rate",
                          value=float(self.design["k"]) / max(fluid, 1e-30),
                          note="W97 closed the same reading at CS-10's seam by "
                               "repairing an effort convention"),
        }

    def eta_s(self):
        ms = self.opt_ms if self.mode in ("optimize", "step") else self.fwd_ms
        return None if ms is None else ms / 1000.0

    def _publish(self) -> None:
        self._seq += 1
        u = self.u.detach().cpu().numpy()
        v = self.v.detach().cpu().numpy()
        png = field_png(u, v, self.cfg.stride)
        with self._cert_lock:
            cert = dict(self.cert)
        tip = float(self.delta[-1]) if self.delta is not None else 0.0
        dmax = float(torch.max(torch.abs(self.delta)))
        payload = dict(
            seq=self._seq, mode=self.mode, notice=self.notice,
            iteration=self.iteration,
            design=dict(self.design), box={k: list(v) for k, v in
                                           F.DESIGN_BOX.items()},
            config=asdict(self.cfg),
            readout=dict(
                downforce=self.load, drag=self.drag,
                lift_to_drag=(self.load / self.drag) if self.drag else 0.0,
                ride_height=float(self.h), tip=tip, delta_max=dmax,
                vm_max=self.vm, vm_ceiling=F.SIGMA_CEIL,
                delta_ceiling=F.DELTA_CEIL,
                delta_envelope=F.DELTA_MAX,
                stress_margin=self.vm / F.SIGMA_CEIL - 1.0,
                deflection_margin=dmax / F.DELTA_CEIL - 1.0,
                interface_residual=getattr(self, "residual", 0.0),
                u_max=float(np.hypot(u, v).max())),
            geometry=self.geometry(),
            seam_quantities=self.seam_quantities(),
            certification=cert,
            trace=self.trace[-240:],
            history=self.history[-160:],
            step_ms=self.step_ms, eta_s=self.eta_s(),
            domain=dict(nx=int(u.shape[1]), ny=int(u.shape[0]), dx=F.DX,
                        stride=self.cfg.stride),
            ramp=dict(lo=U_LO, hi=U_HI,
                      stops=[[a, list(c)] for a, c in RAMP_STOPS],
                      stress=[[a, list(c)] for a, c in STRESS_STOPS]),
        )
        self.frame = Frame(seq=self._seq, png=png, width=u.shape[1],
                           height=u.shape[0], payload=payload)
