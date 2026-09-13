"""FastAPI + websocket front door for the live RaceLab march.

`atlas/demo_frontwing/server.py`'s split, unchanged and for the same reason:

  * **binary websocket frames** carry the field.  The engine colour-maps and
    PNG-encodes on its own thread and the socket sends whatever is latest at a
    fixed rate, so a slow client throttles itself rather than the simulation.
  * **JSON messages** carry everything else, in both directions -- mode flips,
    presets, the field selector and pause/step/reset up; the telemetry, the
    ledger, the per-window error, the envelope stamp and the family table down.

Nothing here owns simulation state and nothing here owns a verdict: every
inbound message is posted to the engine's queue and applied between
macro-steps, and every colour comes from a measurement or from
`compile_scheme`'s own decisions.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..cases import racelab_switch as SW
from .engine import FIELDS, LICENCES, Engine, RaceConfig

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

#: Field frames per second pushed to a client.  The engine runs at whatever the
#: physics costs; this is the cap on the wire.
FRAME_HZ = 10.0


def create_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="Atlas PoC 3 - RaceLab")
    app.state.engine = engine or Engine()
    app.state.engine.start()
    NO_STORE = {"Cache-Control": "no-store"}

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC, "index.html"),
                            headers=NO_STORE)

    @app.get("/api/meta")
    async def meta():
        eng: Engine = app.state.engine
        return JSONResponse({
            "geometry": eng.geometry(),
            "modes": list(SW.MODES),
            "fields": list(FIELDS),
            "presets": ["classical", "upper-learned", "wake-learned",
                        "learned"],
            "licences": list(LICENCES),
            "no_learned_option": dict(SW.NO_LEARNED_OPTION),
            "learned_families": list(SW.LEARNED_FAMILIES),
            "config": asdict(eng.cfg),
            "notes": dict(eng.notes),
            "layout": {k: v for k, v in eng.layout_info.items()
                       if k in ("cols", "rows", "n_windows", "halo",
                                "overlaps", "banded_force_fraction",
                                "min_cut_to_body_clearance_cells",
                                "y_cut_to_body_clearance_cells",
                                "device_planes", "domain_metres")},
        })

    @app.get("/api/state")
    async def state():
        return JSONResponse(app.state.engine.frame.payload or {})

    @app.post("/api/measure")
    async def measure():
        return JSONResponse({"ok": app.state.engine.measure_windows()})

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        eng: Engine = app.state.engine
        stop = asyncio.Event()

        async def receive():
            try:
                while True:
                    raw = await sock.receive_text()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    eng.post(msg)
            except WebSocketDisconnect:
                stop.set()
            except Exception:
                stop.set()

        async def send():
            last = -1
            try:
                while not stop.is_set():
                    fr = eng.frame
                    if fr.seq != last and fr.payload:
                        last = fr.seq
                        await sock.send_text(json.dumps(fr.payload))
                        if fr.png:
                            await sock.send_bytes(fr.png)
                    await asyncio.sleep(1.0 / FRAME_HZ)
            except Exception:
                stop.set()

        rx = asyncio.create_task(receive())
        tx = asyncio.create_task(send())
        await stop.wait()
        for t in (rx, tx):
            t.cancel()
        try:
            await sock.close()
        except Exception:
            pass

    return app


def run(host: str = "127.0.0.1", port: int = 8013, open_browser: bool = False,
        log_level: str = "warning") -> None:
    import uvicorn
    app = create_app()
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(
            "http://%s:%d/" % (host, port))).start()
    uvicorn.run(app, host=host, port=port, log_level=log_level)
