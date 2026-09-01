"""FastAPI + websocket front door for the live composed march.

Two channels and the split matters:

  * **binary websocket frames** carry the field. The engine colour-maps and PNG-
    encodes on its own thread, and the socket sends whatever is latest at a
    fixed rate, so a slow client throttles itself rather than the simulation.
  * **JSON websocket messages** carry everything else, in both directions --
    turbine drags and control clicks up, state and validity down.

Nothing here owns simulation state. Every inbound message is posted to the
engine's queue and applied by the worker between macro-steps.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from .engine import (DEVICES, DOMAINS, LAYOUTS, LIMITS, RAMP_STOPS, REPLAY_HOLD,
                     U_HI, U_LO, DemoConfig, Engine)

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

#: Field frames per second pushed to a client. The engine runs at whatever the
#: physics costs (2-12 Hz measured); this is the cap on the wire.
FRAME_HZ = 12.0


def create_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="Atlas PoC 1a - live wind-farm design")
    app.state.engine = engine or Engine()
    app.state.engine.start()

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/compare")
    async def compare_page():
        """The second window: one layout, solved both ways, timed."""
        return FileResponse(os.path.join(STATIC, "compare.html"))

    @app.get("/api/meta")
    async def meta():
        """Everything the client needs to build its controls, and the clamps."""
        return JSONResponse({
            "domains": {k: {"windows": list(v),
                            "cells": [ (v[0]-1)*112+128, (v[1]-1)*112+128 ]}
                        for k, v in DOMAINS.items()},
            "limits": {k: list(v) for k, v in LIMITS.items()},
            "layouts": sorted(LAYOUTS),
            "devices": list(DEVICES),
            "replay_hold": REPLAY_HOLD,
            # the second window draws its own legend and must draw THIS ramp
            "ramp": {"lo": U_LO, "hi": U_HI, "stops": RAMP_STOPS},
            # The config the server is ACTUALLY running, not the dataclass
            # defaults. The second window seeds its step count from this, and
            # seeding it from the defaults meant a server launched with
            # `--steps 14` quietly ran 30 and labelled the result 30.
            "config": asdict(app.state.engine.cfg),
            "defaults": DemoConfig().__dict__,
        })

    @app.get("/api/state")
    async def state():
        return JSONResponse(app.state.engine.frame.payload or {})

    @app.post("/api/verify")
    async def verify():
        return JSONResponse({"hash": app.state.engine.request_verify()})

    # -- the head-to-head ---------------------------------------------------

    @app.get("/api/compare")
    async def compare_state():
        """Everything the second window draws, except the two field images.

        Polled rather than pushed: the comparison advances a step at a time at
        human speed, and a poll keeps the second window entirely independent of
        the main one's socket -- close the main window mid-run and the timing
        still finishes.
        """
        eng: Engine = app.state.engine
        c = eng.compare
        return JSONResponse(c.view() if c else {"state": "idle"})

    @app.post("/api/compare")
    async def compare_start(req: Request):
        eng: Engine = app.state.engine
        try:
            body = await req.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        if "steps" in body:
            eng.post("config", {"verify_steps": int(body["steps"])})
            # the config lands on the worker thread; wait for it, or this run
            # would be timed at the OLD step count and labelled with the new one
            for _ in range(40):
                if int(eng.cfg.verify_steps) == int(body["steps"]):
                    break
                await asyncio.sleep(0.05)
        h = eng.request_verify(force=bool(body.get("force", True)),
                               pause_live=bool(body.get("pause_live", True)))
        return JSONResponse({"hash": h})

    @app.get("/api/compare/field")
    async def compare_field(side: str = "composed", seq: int = 0):
        """One side's latest field, as a PNG. `seq` is a cache-buster."""
        eng: Engine = app.state.engine
        c = eng.compare
        png = (c.png.get(side) if c else None) or b""
        if not png:
            return Response(status_code=404)
        return Response(png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        eng: Engine = app.state.engine
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()

        async def receive():
            try:
                while True:
                    raw = await sock.receive_text()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    _dispatch(eng, msg)
            except WebSocketDisconnect:
                stop.set()
            except Exception:
                stop.set()

        async def send():
            last = -1
            period = 1.0 / FRAME_HZ
            try:
                while not stop.is_set():
                    frame = eng.frame
                    if frame.seq != last and frame.png:
                        last = frame.seq
                        await sock.send_text(json.dumps(
                            {"type": "state", **frame.payload}))
                        await sock.send_bytes(frame.png)
                    await asyncio.sleep(period)
            except Exception:
                stop.set()

        rx = loop.create_task(receive())
        tx = loop.create_task(send())
        await stop.wait()
        for t in (rx, tx):
            t.cancel()
        try:
            await sock.close()
        except Exception:
            pass

    return app


def _dispatch(eng: Engine, msg: dict[str, Any]) -> None:
    kind = msg.get("type")
    if kind == "theta":
        eng.post("theta", msg.get("theta"))
    elif kind == "config":
        eng.post("config", msg.get("config", {}))
    elif kind == "layout":
        eng.post("layout", msg.get("name", "grid"))
    elif kind == "reset":
        eng.post("reset")
    elif kind == "reseed":
        eng.post("reseed")
    elif kind == "mode":
        m = msg.get("mode")
        if m in ("run", "optimize", "step", "paused", "replay"):
            eng.post("mode", m)
    elif kind == "scrub":
        eng.post("scrub", msg.get("index", 0))
    elif kind == "replay":
        p = {}
        if "index" in msg:
            p["index"] = int(msg["index"])
        if "play" in msg:
            p["play"] = bool(msg["play"])
        eng.post("replay", p)
    elif kind == "verify":
        eng.request_verify()
    elif kind == "compare":
        eng.request_verify(force=bool(msg.get("force", True)),
                           pause_live=bool(msg.get("pause_live", True)))


def run(host: str = "127.0.0.1", port: int = 8011, cfg: DemoConfig | None = None,
        log_level: str = "warning", open_browser: bool = False) -> None:
    import uvicorn
    app = create_app(Engine(cfg))
    url = f"http://{host}:{port}/"
    print("\n  Atlas PoC 1a - live wind-farm design")
    print(f"  open  {url}")
    print(f"  the classical-vs-coupled window is at  {url}compare\n", flush=True)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level=log_level)
