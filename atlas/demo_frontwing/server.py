"""FastAPI + websocket front door for the live front-wing march.

`atlas/demo/server.py`'s split, unchanged and for the same reason:

  * **binary websocket frames** carry the flow field.  The engine colour-maps
    and PNG-encodes on its own thread and the socket sends whatever is latest at
    a fixed rate, so a slow client throttles itself rather than the simulation.
  * **JSON messages** carry everything else, in both directions -- design-knob
    drags and mode clicks up, the readout, the structure's geometry with its von
    Mises colours, and the per-seam certification panel down.

Nothing here owns simulation state, and nothing here owns a verdict: every
inbound message is posted to the engine's queue and applied by the worker
between macro-steps, and every colour in the certification panel comes from
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

from ..cases import front_wing as F
from . import explain as EX
from . import substitution as SUB
from .engine import RULE_NOTE, DemoConfig, Engine

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

#: Field frames per second pushed to a client.  The engine runs at whatever the
#: physics costs; this is the cap on the wire.
FRAME_HZ = 12.0


def create_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="Atlas PoC 2 - live front-wing design")
    app.state.engine = engine or Engine()
    app.state.engine.start()

    NO_STORE = {"Cache-Control": "no-store"}

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(STATIC, "index.html"), headers=NO_STORE)

    @app.get("/api/meta")
    async def meta():
        eng: Engine = app.state.engine
        return JSONResponse({
            "box": {k: list(v) for k, v in F.DESIGN_BOX.items()},
            "keys": list(F.DESIGN_KEYS),
            "reference": dict(F.DESIGN_REF),
            "ceilings": {"sigma": F.SIGMA_CEIL, "delta": F.DELTA_CEIL,
                         "envelope": F.DELTA_MAX},
            "labels": {
                "e_star": "wing stiffness E* (flow units)",
                "tc": "plate thickness t/c",
                "k": "spring rate k",
                "h0": "free ride height h0",
            },
            "notes": dict(RULE_NOTE),
            "scope": {"thickness": F.THICKNESS_SCOPE,
                      "r10": F.R10_ASSEMBLY_SCOPE,
                      "structure": W_SCOPE},
            "config": asdict(eng.cfg),
            "defaults": asdict(DemoConfig()),
            #: the plain-language layer, authored in `explain.py` so it can be
            #: reviewed and tested rather than buried in a template
            "explain": EX.payload(),
            #: the candidate experts beat 1 offers, and the measurement the
            #: middle one rests on -- quoted with its provenance, never re-run
            "candidates": [dict(key=c.key, name=c.name, plain=c.plain,
                                provenance=c.provenance, caveat=c.caveat,
                                is_incumbent=c.is_incumbent)
                           for c in SUB.CANDIDATES],
            "w93": dict(SUB.W93_MEASUREMENT),
            #: the full-scale run, so no panel's live number is the only number
            "recorded": eng.recorded,
        })

    @app.get("/api/state")
    async def state():
        return JSONResponse(app.state.engine.frame.payload or {})

    @app.get("/api/recorded")
    async def recorded():
        return JSONResponse(app.state.engine.recorded)

    @app.post("/api/compile")
    async def recompile():
        app.state.engine.request_compile()
        return JSONResponse({"ok": True})

    @app.post("/api/substitution")
    async def substitution():
        return JSONResponse({"started": app.state.engine.request_substitution()})

    @app.post("/api/ablation")
    async def ablation():
        return JSONResponse({"started": app.state.engine.request_ablation()})

    @app.post("/api/race")
    async def race():
        return JSONResponse({"started": app.state.engine.request_race()})

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


W_SCOPE = (
    "A 32x2 Q1 plane-stress cantilever clamped at the leading edge, and Q1 "
    "elements lock in bending. E* is a declared design knob in FLOW units, "
    "calibrated to a deflection rather than taken from a material, so no stress "
    "number on this screen is a statement about an alloy -- the ceiling is a "
    "level the reference design sits inside and the downforce-seeking direction "
    "runs into, which is the only property a ceiling needs for a constrained "
    "search to be about the constraint."
)


def _dispatch(eng: Engine, msg: dict[str, Any]) -> None:
    kind = msg.get("type")
    if kind == "design":
        eng.post("design", msg.get("design", {}))
    elif kind == "config":
        eng.post("config", msg.get("config", {}))
    elif kind == "mode":
        m = msg.get("mode")
        if m in ("run", "optimize", "step", "paused"):
            eng.post("mode", m)
    elif kind == "reset":
        eng.post("reset")
    elif kind == "compile":
        eng.post("compile")
    elif kind in ("substitution", "ablation", "race"):
        #: posted to the queue rather than started here, so a beat always begins
        #: between macro-steps and reads a state no half-finished step is inside
        eng.post(kind, msg.get("params") or {})


def run(host: str = "127.0.0.1", port: int = 8012,
        cfg: DemoConfig | None = None, log_level: str = "warning",
        open_browser: bool = False) -> None:
    import uvicorn
    app = create_app(Engine(cfg))
    url = f"http://{host}:{port}/"
    print("\n  Atlas PoC 2 - live front-wing design")
    print(f"  open  {url}\n", flush=True)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level=log_level)
