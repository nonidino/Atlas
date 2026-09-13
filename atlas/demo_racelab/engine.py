"""The persistent march behind PoC 3, and everything the screen reads.

One worker thread marches the car and never stops; the server reads whatever is
latest.  Inbound messages -- a mode flip, a preset, a field selector, pause,
step, reset -- are posted to a queue and applied **between macro-steps**, so no
message can land inside an exchange and leave the composition layer half done.

**Three things this engine is built to make impossible to miss**, because CS-20
measured all three and a dashboard that hid any of them would be the artifact
this project exists to not produce:

1. **The lead ratio.**  A learned window is asked for a step of `dt_ex`, which
   on this tiling is **1/32** of Poseidon-T's native lead.  `LEAD_RATIO` is on
   the wire in every frame and the page shows it beside the switch.
2. **The envelope.**  Every declared predicate is consulted every macro-step by
   `racelab.RaceRollout._absorb`.  The engine runs with ``enforce=False`` --
   a dashboard that stops reports nothing -- and **stamps** every frame taken
   outside, which is PoC 2's `OUTSIDE THE MODEL` pattern.
3. **Four of five families have no learned option.**  The switch is greyed out
   for them **with the reason**, never hidden.

The accuracy story, as section 5.4 asks for it
----------------------------------------------

  * the **per-window one-step error** is computed against `WindowNS` on the
    SAME input state, needs no global referent, and is the number to trust.
  * the **global** rms error is against an all-classical march held beside the
    live one from the same release state -- a real referent, and it costs a
    second march, which the frame reports.
  * **no monolith** is run: the composed classical answer carries its own
    composition defect and this demo does not claim otherwise.
"""

from __future__ import annotations

import io
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..cases import ground_effect as GE
from ..cases import racelab as RL
from ..cases import racelab_switch as SW

#: The checkpoint's native lead over the step the composition layer asks for.
#: Arithmetic, not a measurement: one window spans `wx*dx` length units, so
#: `Scaling.time` is 1.0 and `dt_ex / 0.1` is 1/32 on this tiling.
LEAD_RATIO = (GE.MACRO_DT / GE.EXCHANGES) / 0.1

#: The inflow the machine is sized for (CS-20 §2.2).  Without it the powertrain
#: is outside its declared envelope from the first macro-step.
#:
#: **Re-measured 2026-09-13 for the car the user DREW** (Tier 56, stage
#: ``spinup`` of ``out/racelab5``), and it is the release state's value because
#: nothing better exists for this car.  ``0.654077065086774`` was the traced
#: car's horizon minimum; on the drawn car the duct carries 0.5317 at release
#: and the machine sized for 0.654 motors from macro-step 0.  **W228's
#: procedure -- size for the horizon's MINIMUM -- has no admissible answer
#: here**: ``u_rotor`` falls 16.6% over 600 macro-steps, the machine sized for
#: that minimum puts the disk's induction on its UPPER clamp at macro-step 0,
#: and sized for the release state it reaches the LOWER clamp at macro-step 20.
#: No constant spans a fall that wide (W244).  This value buys the first twenty
#: macro-steps, and the page stamps every one after them.
U_DUCT = 0.5317254889754462

#: Colour ramp bounds for the speed overlay, in free streams.
U_LO, U_HI = 0.0, 2.0

FIELDS = ("speed", "u", "v", "vorticity", "error")

LICENCES = (
    {"name": "reference.WindowNS",
     "what": "the classical incompressible expert, this project's own",
     "licence": "private build repo, vendored"},
    {"name": "Poseidon-T",
     "what": "the learned expert, frozen, camlab-ethz",
     "licence": "CC-BY-NC-4.0 -- research use, no commercial use"},
)


def _lut():
    t = np.linspace(0.0, 1.0, 256)[:, None]
    a = np.array([[12, 16, 40]], dtype=float)
    b = np.array([[40, 150, 220]], dtype=float)
    c = np.array([[250, 240, 120]], dtype=float)
    lo = a + (b - a) * np.clip(t / 0.55, 0, 1)
    hi = b + (c - b) * np.clip((t - 0.55) / 0.45, 0, 1)
    return np.where(t < 0.55, lo, hi).astype(np.uint8)


_FIELD_LUT = _lut()
_ERR_LUT = np.stack([
    np.clip(np.linspace(0, 1, 256) * 512, 0, 255),
    np.clip(255 - np.linspace(0, 1, 256) * 400, 0, 255),
    np.full(256, 60.0)], axis=1).astype(np.uint8)


def field_array(u: np.ndarray, v: np.ndarray, which: str,
                ref: tuple | None = None) -> np.ndarray:
    """The scalar the overlay draws, per cell."""
    if which == "u":
        return u
    if which == "v":
        return v
    if which == "vorticity":
        dvdx = np.zeros_like(v)
        dudy = np.zeros_like(u)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * RL.DX)
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * RL.DX)
        return dvdx - dudy
    if which == "error":
        if ref is None:
            return np.zeros_like(u)
        return np.hypot(u - ref[0], v - ref[1])
    return np.hypot(u, v)


def field_png(u: np.ndarray, v: np.ndarray, which: str = "speed",
              ref: tuple | None = None, stride: int = 1) -> bytes:
    """Colour-map a field and encode it.

    Row 0 of the array is ``y = 0`` and an image's row 0 is the top, so the
    array is flipped here and the client draws it with y up -- which is how a
    car on a road is read.
    """
    from PIL import Image
    a = field_array(u, v, which, ref)[::stride, ::stride]
    if which == "error":
        t = np.clip(a / max(1e-9, float(np.percentile(a, 99.5)) or 1e-9), 0, 1)
        rgb = _ERR_LUT[(t * 255.0).astype(np.uint8)][::-1]
    elif which in ("u", "v", "vorticity"):
        s = max(1e-9, float(np.percentile(np.abs(a), 99.0)))
        t = np.clip(0.5 + 0.5 * a / s, 0, 1)
        rgb = _FIELD_LUT[(t * 255.0).astype(np.uint8)][::-1]
    else:
        t = np.clip((a - U_LO) / (U_HI - U_LO), 0.0, 1.0)
        rgb = _FIELD_LUT[(t * 255.0).astype(np.uint8)][::-1]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb), mode="RGB").save(
        buf, format="PNG", compress_level=1)
    return buf.getvalue()


#: Where a settled field is looked for, newest first.  Each directory holds the
#: release state one tier's spin-up produced for the car built at the time.
RELEASE_TIERS = ("racelab5", "racelab4", "racelab2")

#: How to regenerate the release state for whatever car is built now.
RESPIN = "python scripts/tier54_traced_car.py --out out/racelab5 --stages spinup"


def find_release(root: str, fingerprint: str, tiers=RELEASE_TIERS):
    """``(u, v, note, is_this_car_s)`` -- the settled field for THIS car.

    Every cache that exists is opened and the first whose recorded fingerprint
    is the car's is returned, with ``True``.  If none is, the newest cache is
    returned with ``False`` and a note that says, in capitals, that it was
    settled around a different car -- a transient on screen is better than a
    freestream one, and saying so is what makes it acceptable.  With no cache
    at all it is ``(None, None, note, None)``: not "no", but "nothing to ask".

    A field written before fingerprints existed carries none, and is never
    taken for this car's: an unknown provenance is not a matching one.
    """
    import os
    found = []
    for tier in tiers:
        p = os.path.join(root, "out", tier, "cache", "settled.npz")
        if not os.path.isfile(p):
            continue
        with np.load(p) as d:
            fp = str(d["geometry"]) if "geometry" in d.files else None
            u, v = d["u"], d["v"]
        if fp == fingerprint:
            return (u, v, "the settled field in out/%s/cache, settled around "
                          "THIS car (fingerprint %s)" % (tier, fp[:12]), True)
        found.append((tier, fp, u, v))
    if found:
        tier, fp, u, v = found[0]
        return (u, v,
                "the settled field in out/%s/cache -- WHICH WAS SETTLED AROUND A "
                "DIFFERENT CAR (%s; the car built now is %s), so the flow "
                "through the duct is another car's and the first macro-steps "
                "are a transient. Run `%s`"
                % (tier, ("fingerprint " + fp[:12]) if fp else
                   "it carries no fingerprint", fingerprint[:12], RESPIN),
                False)
    return (None, None,
            "THE FREESTREAM -- no settled cache, so this is a transient no "
            "recorded number was measured at. Run `%s`" % RESPIN, None)


@dataclass
class RaceConfig:
    #: the cadence a dashboard marches.  `tight` exists so the composition
    #: error is measurable and was never an interactive column: CS-19 measured
    #: it at 1.7 s a macro-step against the 0.5 s ceiling.
    coupling: str = "lagged"
    #: the machine sized for the inflow its host delivers (CS-20 §2.2).
    #: `None` is the Tier 51 configuration, which is outside its envelope here.
    host_inflow: float | None = U_DUCT
    #: **OFF by design, and stamped.**  A dashboard that stops reports nothing;
    #: what it must never do is show a number from outside an envelope without
    #: saying so.
    enforce: bool = False
    #: hold a second, all-classical march beside the live one so the global rms
    #: error has a real referent.  It costs a second march.
    referent: bool = True
    field: str = "speed"
    paused: bool = False
    #: **Phase 4.**  `"3d"` marches the half-car in `racelab3d` instead, and the
    #: learned switch goes with it: Poseidon-T is a 2-D operator at a fixed
    #: 128x128 and there is no 3-D checkpoint in this project, so in three
    #: dimensions the column is classical by necessity and the page says why
    #: rather than simply offering nothing.
    dims: str = "2d"


@dataclass
class Frame:
    seq: int = 0
    png: bytes = b""
    width: int = 0
    height: int = 0
    payload: dict = field(default_factory=dict)


class Engine:
    """The march, the switch, and the telemetry the page reads."""

    def __init__(self, cfg: RaceConfig | None = None) -> None:
        self.cfg = cfg or RaceConfig()
        self.q: queue.Queue = queue.Queue()
        self.frame = Frame()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lock = threading.Lock()

        self.tiling, self.layout_info = RL.layout()
        self.names = list(self.tiling.names)
        self.assignment = {n: SW.Mode.CLASSICAL.value for n in self.names}
        self.selected = self.names[0]
        self.step_i = 0
        self.step_s: float | None = None
        self.base_step_s: float | None = None
        self.outside: dict | None = None
        self.outside_steps = 0
        self.window_error: dict[str, float | None] = {n: None for n in self.names}
        self.window_error_at: int | None = None
        self.rms_vs_referent: float | None = None
        self.stack: SW.WindowStack | None = None
        self.stack_error: str | None = None
        self._m3 = None                 # the 3-D march, built on first use
        self.families: dict = {}
        self.seam_verdicts: dict = {}
        self._u = self._v = None
        self._ru = self._rv = None
        self._roll = None
        self._ref_roll = None
        self.notes = {
            "lead_ratio": LEAD_RATIO,
            "lead_ratio_note":
                "the composition layer exchanges %d times a macro-step, so a "
                "learned window is asked for 1/%d of the checkpoint's native "
                "lead -- and CS-20 measured the per-window error at a MINIMUM "
                "at the native lead and 4.3x that minimum at an eighth of it"
                % (GE.EXCHANGES, int(round(1.0 / LEAD_RATIO))),
            "n_windows": self.tiling.n_windows,
            "domain": [self.tiling.nx, self.tiling.ny],
            "window": [self.tiling.wx, self.tiling.wy],
            "licences": list(LICENCES),
        }

    # -- geometry the page draws -------------------------------------------

    def geometry(self) -> dict:
        objs, flat = RL.car_bodies()
        bodies = []
        for b in flat:
            bodies.append({"id": b.body_id, "group": b.group,
                           "x0": b.x_le, "y0": b.y_le,
                           "x1": b.x_te, "y1": b.y_te})
        wins = []
        for k, (ox, oy) in enumerate(self.tiling.offsets):
            wins.append({"name": self.names[k], "x": ox, "y": oy,
                         "w": self.tiling.wx, "h": self.tiling.wy})
        sites = RL.device_sites(self.tiling)
        devices = []
        for d, s in sites.items():
            spec = RL.RaceDeviceSpec("J1" if d == "RAD" else "J3", s, d,
                                     self.tiling)
            devices.append({"id": d, "x": spec.x_plane / RL.DX,
                            "y0": int(spec.rows[0]), "y1": int(spec.rows[-1]),
                            "ring_to_plane_cells": spec.ring_to_plane_cells})
        return {"bodies": bodies, "windows": wins, "devices": devices,
                "nx": self.tiling.nx, "ny": self.tiling.ny,
                "x_bands": [list(b) for b in self.tiling.x_bands()],
                "y_bands": [list(b) for b in self.tiling.y_bands()],
                "occupancy_in_a_cut":
                    self.layout_info["banded_force_fraction"]}

    # -- the worker --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="racelab-engine")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def post(self, msg: dict) -> None:
        self.q.put(msg)

    def _build(self) -> None:
        import torch
        torch.set_num_threads(2)
        u, v = self._release()
        self._u, self._v = u.copy(), v.copy()
        self._ru, self._rv = u.copy(), v.copy()
        self._roll = self._rollout(self.assignment)
        self._ref_roll = self._rollout(
            {n: SW.Mode.CLASSICAL.value for n in self.names})
        self.step_i = 0
        self.outside = None
        self.outside_steps = 0

    def _rollout(self, assignment):
        return SW.MixedRollout(
            tiling=self.tiling, assignment=assignment, stack=self._stack(),
            join_coupling=self.cfg.coupling, host_inflow=self.cfg.host_inflow,
            enforce=self.cfg.enforce)

    def _stack(self):
        if self.stack is None and self.stack_error is None:
            try:
                self.stack = SW.WindowStack(tiling=self.tiling,
                                            threads_learned=2,
                                            threads_classical=2)
            except Exception as exc:
                self.stack_error = str(exc)[:300]
        return self.stack

    def _release(self):
        """The settled field THIS car's spin-up produced, or the freestream.

        **Absent, the demo says so on screen.**  PoC 2's README: without the
        settled cache the demo *"releases from the freestream, says so on
        screen, and shows a transient that no number on the results page was
        measured at"* -- and here that transient leaves the disk's clamp for 58
        macro-steps, which the envelope stamp will report.

        **A settled field belongs to a GEOMETRY, and it is matched by the
        fingerprint it carries rather than by the directory it sits in.**
        This demo released the traced car from the hand-drawn car's field in
        Tier 54 and the drawn car from the traced car's in Tier 55: the
        directory order said which cache was newest, not which car it
        belonged to.  See `find_release`.
        """
        import os
        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        u, v, note, matches = find_release(root, RL.geometry_fingerprint())
        self.notes["release"] = note
        self.notes["release_is_this_car_s"] = matches
        if u is None:
            return (np.full((self.tiling.ny, self.tiling.nx), GE.U_INF),
                    np.zeros((self.tiling.ny, self.tiling.nx)))
        return u, v

    def _loop(self) -> None:
        import torch
        self._build()
        self._compile_once()
        last = 0.0
        while not self._stop.is_set():
            self._drain()
            if self.cfg.paused:
                time.sleep(0.03)
                if time.perf_counter() - last > 0.25:
                    self._publish()
                    last = time.perf_counter()
                continue
            try:
                with torch.no_grad():
                    self._march_once()
            except Exception as exc:                          # pragma: no cover
                self.notes["error"] = str(exc)[:300]
                self.cfg.paused = True
            if time.perf_counter() - last > 0.08:
                self._publish()
                last = time.perf_counter()

    def _march3d_once(self) -> None:
        """One macro-step of the half-car in three dimensions."""
        from atlas.cases import racelab3d as R3
        if self._m3 is None:
            self._m3 = R3.March3D()
        self.step_s = self._m3.step()
        self.step_i = self._m3.step_i

    def _march_once(self) -> None:
        if self.cfg.dims == "3d":
            return self._march3d_once()
        import torch
        opt = dict(dtype=RL.W.TORCH_DTYPE, device="cpu")
        u = torch.as_tensor(self._u, **opt)
        v = torch.as_tensor(self._v, **opt)
        t0 = time.perf_counter()
        u, v, load, drag = self._roll.macro_step(u, v, self.step_i)
        self.step_s = time.perf_counter() - t0
        self._u = u.detach().cpu().numpy()
        self._v = v.detach().cpu().numpy()
        #: **the stamp, and it is sticky within a run.**  A column that left
        #: the envelope at step 4 is still a column that left it at step 40.
        if self._roll.outside is not None:
            self.outside_steps += 1
            self.outside = dict(self._roll.outside)
        if self.cfg.referent and self._ref_roll is not None:
            ru = torch.as_tensor(self._ru, **opt)
            rv = torch.as_tensor(self._rv, **opt)
            t0 = time.perf_counter()
            ru, rv, _l, _d = self._ref_roll.macro_step(ru, rv, self.step_i)
            self.base_step_s = time.perf_counter() - t0
            self._ru = ru.detach().cpu().numpy()
            self._rv = rv.detach().cpu().numpy()
            den = float(np.sqrt(np.mean((self._ru - GE.U_INF) ** 2
                                        + self._rv ** 2)))
            self.rms_vs_referent = float(
                np.sqrt(np.mean((self._u - self._ru) ** 2
                                + (self._v - self._rv) ** 2)) / max(den, 1e-30))
        self.step_i += 1
        self.last_load = float(load.detach())
        self.last_drag = float(drag.detach())

    # -- section 5.2's per-window error, on demand --------------------------

    def measure_windows(self) -> bool:
        """One-step error per window against `WindowNS` on the SAME state.

        Exact, cheap, and needs no global referent -- section 5.4's third
        clause and the number the inspector leads with.
        """
        st = self._stack()
        if st is None or st.ex is None or self._u is None:
            return False
        try:
            r = SW.one_step_errors(st, self._u, self._v, 1)
        except Exception as exc:                              # pragma: no cover
            self.stack_error = str(exc)[:300]
            return False
        with self.lock:
            self.window_error = dict(r["per_window"])
            self.window_error_at = self.step_i
            self.notes["window_error_meta"] = {
                "lead_over_native": r["lead_over_native"],
                "classical_s": r["classical_s"], "learned_s": r["learned_s"],
                "median": r["median"], "max": r["max"],
                "at_step": self.step_i,
                "note": "one macro-step, which is 1/8 of the checkpoint's "
                        "native lead; the composition layer asks for 1/32",
            }
        return True

    # -- the compile, once --------------------------------------------------

    def _compile_once(self) -> None:
        try:
            from .. import compiler as C
            g, _aux = RL.build(self._u, self._v)
            res = C.compile_scheme(g)
            self.families = SW.family_switch_table(g)
            refus = sorted({"%s/%s" % (d.layer, d.rule) for d in res.decisions
                            if d.verdict.value == "refuse"})
            self.seam_verdicts = {
                "graph_verdict": res.verdict.value,
                "refusals": refus,
                "n_agents": len(g.agents), "n_seams": len(g.connections),
                "how_to_read":
                    "L7/R9's subject is <graph>: it is ONE decision. A per-seam "
                    "map paints it across every seam so it can be seen, and N "
                    "red tiles from it are N views of one refusal, not N "
                    "verdicts (PoC 2's W177)",
            }
        except Exception as exc:                              # pragma: no cover
            self.seam_verdicts = {"error": str(exc)[:300]}

    # -- messages -----------------------------------------------------------

    # -- the learned expert, and what happens when it is not here -----------

    def learned_available(self) -> bool:
        """Whether a window can actually be switched to the learned expert."""
        return bool(self.cfg.dims != "3d" and self._stack() is not None
                    and self.stack.ex is not None)

    def _refuse_learned(self) -> None:
        self.notes["learned_refused"] = (
            "the learned expert is NOT AVAILABLE on this machine, so no window "
            "was switched: %s. The classical column runs; the switch is greyed "
            "out rather than left to do nothing"
            % (self.stack_error or "the window stack was not built"))

    def _drain(self) -> None:
        #: **W243.**  A mode flip used to change `assignment` first and build
        #: the rollout after, catching the failure into a note the page never
        #: drew -- so with the learned expert missing, "all learned" tinted
        #: every window learned and counted them learned in the ledger while
        #: the march went on classical.  A control that visibly does something
        #: it did not do.  Now the flip is refused BEFORE the assignment moves,
        #: and a rollout that still cannot be built puts the old one back.
        before = dict(self.assignment)
        rebuilt = False
        while True:
            try:
                msg = self.q.get_nowait()
            except queue.Empty:
                break
            kind = msg.get("kind")
            if kind == "mode":
                n, m = msg.get("window"), msg.get("mode")
                if n in self.assignment and m in SW.MODES:
                    if m == SW.Mode.CERTIFIED.value:
                        self.notes["certified_note"] = (
                            "CERTIFIED is a STEADY-STATE mode: defect "
                            "correction certifies a fixed point and an "
                            "explicit time step has none to certify. It is "
                            "measured per window by "
                            "`racelab_switch.certified_window`, not marched "
                            "(W226)")
                        continue
                    if m == SW.Mode.LEARNED.value and not self.learned_available():
                        self._refuse_learned()
                        continue
                    self.notes.pop("certified_note", None)
                    self.notes.pop("learned_refused", None)
                    self.assignment[n] = m
                    rebuilt = True
            elif kind == "preset":
                try:
                    want = SW.assignment_from(msg.get("name"), self.tiling)
                except ValueError:
                    continue
                if (SW.Mode.LEARNED.value in want.values()
                        and not self.learned_available()):
                    self._refuse_learned()
                    continue
                self.assignment = want
                self.notes.pop("certified_note", None)
                self.notes.pop("learned_refused", None)
                rebuilt = True
            elif kind == "select":
                if msg.get("window") in self.assignment:
                    self.selected = msg["window"]
            elif kind == "dims":
                if msg.get("value") in ("2d", "3d"):
                    self.cfg.dims = msg["value"]
                    rebuilt = True
            elif kind == "field":
                if msg.get("name") in FIELDS:
                    self.cfg.field = msg["name"]
            elif kind == "pause":
                self.cfg.paused = bool(msg.get("value", True))
            elif kind == "step":
                self.cfg.paused = True
                try:
                    self._march_once()
                except Exception as exc:                      # pragma: no cover
                    self.notes["error"] = str(exc)[:300]
            elif kind == "reset":
                self._build()
            elif kind == "measure":
                self.measure_windows()
            elif kind == "referent":
                self.cfg.referent = bool(msg.get("value", True))
        if rebuilt:
            try:
                self._roll = self._rollout(self.assignment)
            except Exception as exc:
                #: the march did not change, so neither may what the page says
                #: is marching
                self.assignment = before
                self.notes["error"] = ("the switch was NOT applied: %s"
                                       % str(exc)[:300])

    # -- the frame ----------------------------------------------------------

    def _publish(self) -> None:
        if self._u is None and not (self.cfg.dims == "3d"
                                    and self._m3 is not None):
            return
        ref = (self._ru, self._rv) if self.cfg.referent else None
        three_d = self.cfg.dims == "3d" and self._m3 is not None
        try:
            if three_d:
                # a z-slice through the half-car, transposed into the (y, x)
                # order the 2-D overlay and the page already use
                sl = self._m3.speed_slice().T
                png = field_png(sl, np.zeros_like(sl), "speed", None)
            else:
                png = field_png(self._u, self._v, self.cfg.field, ref)
        except Exception:                                     # pragma: no cover
            png = b""
        ledger = {m: sum(1 for x in self.assignment.values() if x == m)
                  for m in SW.MODES}
        speed = (self.base_step_s / self.step_s
                 if (self.step_s and self.base_step_s) else None)
        payload = {
            "seq": self.frame.seq + 1,
            "step": self.step_i,
            "assignment": dict(self.assignment),
            "selected": self.selected,
            "ledger": ledger,
            "field": self.cfg.field,
            "fields": list(FIELDS),
            "paused": self.cfg.paused,
            "s_per_macro_step": self.step_s,
            "macro_steps_per_s": (1.0 / self.step_s) if self.step_s else None,
            "referent_s_per_macro_step": self.base_step_s,
            "speed_ratio_vs_all_classical": speed,
            "rms_vs_all_classical": self.rms_vs_referent,
            "window_error": dict(self.window_error),
            "window_error_at": self.window_error_at,
            "outside": self.outside,
            "outside_steps": self.outside_steps,
            "load": getattr(self, "last_load", None),
            "drag": getattr(self, "last_drag", None),
            "state": self._roll.state.as_dict() if self._roll else {},
            "notes": dict(self.notes),
            "families": self.families,
            "verdicts": self.seam_verdicts,
            "stack_error": self.stack_error,
            "learned_available": (False if self.cfg.dims == "3d"
                                  else bool(self.stack and self.stack.ex)),
            "dims": self.cfg.dims,
            "three_d": (self._m3.as_dict() if self._m3 is not None else None),
            "config": asdict(self.cfg),
        }
        with self.lock:
            w, h = ((self._m3.tiling.nx, self._m3.tiling.ny) if three_d
                    else (self.tiling.nx, self.tiling.ny))
            self.frame = Frame(seq=self.frame.seq + 1, png=png,
                               width=w, height=h, payload=payload)
