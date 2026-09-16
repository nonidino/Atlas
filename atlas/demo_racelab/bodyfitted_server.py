"""The body-fitted column's dashboard: a march in a thread, a page, a commit.

PoC 3, Tier 71.  [[poc3-racelab-dashboard]].

Tier 68 built the column's engine, Tiers 69-70 wired section 3.3's knobs and
measured what each one reaches.  Nothing rendered a slider, so criterion 2 --
*move a parameter and watch every coupled subsystem respond* -- could not be
demonstrated by a person.  This is the screen.

**Rebuild on commit, not on slider move** (the user's decision, 2026-09-16).  A
geometry change re-cuts the car, which costs 90 s for the composite and up to
63 s for the probe operator -- it cannot run inside a frame and it must not run
on every drag of a slider.  So:

  * a slider moves freely and nothing is re-cut; the knob's value is recorded
    and the subsystems that RESPONDED are reported straight back;
  * a geometry knob marks the column `pending_regrid`, and the page shows,
    plainly, that it is **marching the car before the change**;
  * **Commit** re-cuts the car, with the stages streamed to the page so the
    wait is a visible recompiling state and not a hang.

What the page must not let a viewer believe
-------------------------------------------

Three things, all of which the engine already knows and the page is obliged to
show rather than quietly drop:

1. **The 61%.**  A uniform 128 x 128 window cannot accept a curvilinear patch, so
   the learned expert can never reach the boundary layers, the wheel clearances
   or the duct (W288).  The page says so beside the switch, not in a footnote.
2. **The fields it cannot draw** -- vorticity, temperature, the error field --
   each with its reason, which is section 4.3's rule applied to fields.
3. **Which car is marching.**  After a knob moves and before a commit, the
   picture is of the OLD car.  Saying nothing there would be the defect this
   project keeps finding.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from ..cases import car_knobs as CK
from .bodyfitted import FIELD_HOLES, FIELDS, LEARNED_HOLE, BodyFittedColumn

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "static", "bodyfitted.html")
ROOT = os.path.dirname(os.path.dirname(HERE))


class BodyFittedEngine:
    """One worker thread: build, then march until stopped.

    A commit is handled ON the worker, between steps, so the composite is never
    rebuilt underneath a march that is reading it.
    """

    def __init__(self, root: str = ROOT, autostart: bool = True) -> None:
        self.root = root
        self.col = BodyFittedColumn()
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = "waiting"
        self.progress: dict = {"stage": "not started", "fraction": 0.0}
        self.built = False
        self.build_report: dict = {}
        self.last_error: str | None = None
        self.frame_seq = 0
        self.png: bytes = b""
        self.payload: dict = {}
        self.paused = False
        self.field = "speed"
        self.events: list = []
        self.autostart = autostart

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def post(self, msg: dict) -> None:
        self.q.put(msg)

    def _note(self, kind: str, **kw) -> None:
        with self.lock:
            self.events.append(dict(kw, kind=kind, at=time.time()))
            del self.events[:-40]

    def _set_progress(self, p: dict) -> None:
        with self.lock:
            self.progress = dict(p)

    # -- the worker ---------------------------------------------------------

    def _loop(self) -> None:
        try:
            self.status = "building"
            rep = self.col.build(root=self.root, progress=self._set_progress)
            with self.lock:
                self.build_report = rep
                self.built = True
            self.status = "marching"
            self._set_progress({"stage": "ready", "fraction": 1.0})
        except Exception as exc:                                 # pragma: no cover
            self.status = "failed"
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._note("error", why=self.last_error)
            return
        while not self._stop.is_set():
            self._drain()
            if self.paused:
                time.sleep(0.05)
                continue
            try:
                self.col.step()
            except Exception as exc:                             # pragma: no cover
                self.status = "failed"
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._note("error", why=self.last_error)
                return
            self._publish()

    def _drain(self) -> None:
        while True:
            try:
                msg = self.q.get_nowait()
            except queue.Empty:
                return
            kind = msg.get("kind")
            try:
                if kind == "knob":
                    r = self.col.set_knob(str(msg["name"]), float(msg["value"]))
                    self._note("knob", **r)
                elif kind == "commit":
                    self._commit()
                elif kind == "field":
                    if msg.get("name") in FIELDS:
                        self.field = msg["name"]
                elif kind == "pause":
                    self.paused = bool(msg.get("value", True))
                elif kind == "step":
                    self.col.step()
                    self._publish()
            except Exception as exc:
                self._note("refused", why=str(exc)[:300], name=msg.get("name"))

    def _commit(self) -> None:
        """Re-cut the car from the committed knobs, between steps."""
        if not self.col.pending_regrid:
            self._note("commit", skipped="nothing pending")
            return
        self.status = "recompiling"
        t0 = time.perf_counter()
        try:
            rep = self.col.regrid(root=self.root, progress=self._set_progress)
        except Exception as exc:                                 # pragma: no cover
            self.status = "failed"
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._note("error", why=self.last_error)
            return
        with self.lock:
            self.build_report = rep
        self.status = "marching"
        self._note("commit", wall_s=time.perf_counter() - t0,
                   fingerprint=rep.get("fingerprint_after", "")[:12],
                   released_from=rep.get("released_from"))

    def _publish(self) -> None:
        try:
            png, payload = self.col.frame(self.field)
        except Exception:                                        # pragma: no cover
            return
        tel = self.col.telemetry()
        with self.lock:
            self.frame_seq += 1
            self.png = png
            self.payload = {"seq": self.frame_seq, "overlay": payload,
                            "telemetry": tel, "status": self.status,
                            "pending_regrid": list(self.col.pending_regrid),
                            "knobs": dict(self.col.knob_state().values),
                            "fingerprint": self.col._fingerprint[:12],
                            "released_from": self.build_report.get("released_from"),
                            "events": list(self.events[-6:])}

    # -- what the page reads ------------------------------------------------

    def meta(self) -> dict:
        offerable, refused = [], []
        for k in CK.KNOBS:
            row = {"name": k.name, "group": k.group, "lo": k.lo, "hi": k.hi,
                   "default": k.default, "unit": k.unit, "how": k.how,
                   "reaches": list(k.reaches), "note": k.note,
                   "regrid": k.name in CK.REGRID}
            (refused if k in CK.must_not_be_shown() else offerable).append(row)
        return {"title": "RaceLab - the body-fitted column",
                "fields": list(FIELDS), "field_holes": dict(FIELD_HOLES),
                "learned_hole": LEARNED_HOLE,
                "knobs": offerable, "refused_knobs": refused,
                "omitted": dict(CK.OMITTED),
                "status": self.status, "progress": self.progress,
                "build": {k: v for k, v in self.build_report.items()
                          if k in ("fingerprint", "grids", "n_unknowns", "coverage",
                                   "released_from", "total_s", "regrid_s")}}

    def snapshot(self) -> dict:
        with self.lock:
            return {"status": self.status, "progress": dict(self.progress),
                    "built": self.built, "error": self.last_error,
                    **({k: v for k, v in self.payload.items()} if self.payload else {})}


def create_app(engine: BodyFittedEngine | None = None):
    # **FastAPI is imported at MODULE level and it has to be.**  This
    # module carries `from __future__ import annotations`, so
    # `sock: WebSocket` is the STRING "WebSocket" and FastAPI resolves it
    # with `get_type_hints` against the module's globals -- not against
    # this function's locals.  Imported in here, the name does not
    # resolve, the parameter is not recognised as the socket, and every
    # upgrade to /ws is refused with **403**: the page loads, /api/meta
    # answers, and no frame ever arrives.  Found by opening the page.
    eng = engine or BodyFittedEngine()
    app = FastAPI(title="Atlas PoC 3 - RaceLab, body-fitted")

    @app.on_event("startup")
    async def _startup():
        if eng.autostart:
            eng.start()

    @app.get("/")
    async def index():
        return FileResponse(PAGE)

    @app.get("/api/meta")
    async def meta():
        return JSONResponse(eng.meta())

    @app.get("/api/state")
    async def state():
        return JSONResponse(eng.snapshot())

    @app.get("/api/frame.png")
    async def frame():
        with eng.lock:
            png = eng.png
        return Response(content=png or b"", media_type="image/png")

    @app.post("/api/knob")
    async def knob(body: dict):
        eng.post({"kind": "knob", "name": body.get("name"), "value": body.get("value")})
        return JSONResponse({"queued": True})

    @app.post("/api/commit")
    async def commit():
        eng.post({"kind": "commit"})
        return JSONResponse({"queued": True})

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        last = -1

        async def receive():
            try:
                while True:
                    msg = json.loads(await sock.receive_text())
                    eng.post(msg)
            except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
                return

        task = asyncio.create_task(receive())
        try:
            while True:
                with eng.lock:
                    seq, png, payload = eng.frame_seq, eng.png, dict(eng.payload)
                    prog = dict(eng.progress)
                    status, err = eng.status, eng.last_error
                if seq != last and png:
                    last = seq
                    await sock.send_text(json.dumps(payload))
                    await sock.send_bytes(png)
                else:
                    await sock.send_text(json.dumps(
                        {"seq": seq, "status": status, "progress": prog, "error": err}))
                await asyncio.sleep(0.2)
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            task.cancel()

    app.state.engine = eng
    return app
