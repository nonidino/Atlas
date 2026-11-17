"""The studio's server: the page, its files, and one WebSocket per open page.

    python -m atlas.studio --open          # http://127.0.0.1:8021/

FastAPI and uvicorn serve one HTML page and its plain JavaScript modules (no build
step); every edit, run and file action is a message over ``/ws`` (`protocol.py`).

Three lessons from the earlier dashboards are built in here:

* **FastAPI is imported at module level**, and this module has no ``from __future__
  import annotations``.  PoC 3's body-fitted server had every upgrade to ``/ws``
  refused with 403 because ``sock: WebSocket`` was a string FastAPI could not
  resolve, and the page loaded and never received a frame.
* **The MIME types are set here, not left to the machine.**  Windows' registry can
  map ``.js`` to ``text/plain``, which a browser refuses for a module script, and
  ``.wasm`` must be ``application/wasm`` for streaming compilation.
* **Every static file is revalidated** (``Cache-Control: no-cache``), and the page's
  own URLs carry the server's start stamp (``?v=``): a restarted server is never
  shown through a cached page (the vault's "preview servers and tabs go stale").

A page that reconnects (a dropped socket, a reload in the same tab) gets its own
session back for an hour, so a theme switch or a lost connection never loses a case.
"""

import asyncio
import json
import mimetypes
import os
import secrets
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import fonts, protocol
from .session import Session

for _ext, _type in ((".js", "text/javascript"), (".mjs", "text/javascript"),
                    (".css", "text/css"), (".wasm", "application/wasm"),
                    (".woff2", "font/woff2"), (".json", "application/json"),
                    (".svg", "image/svg+xml")):
    mimetypes.add_type(_type, _ext)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
#: how long a page's session outlives its socket
KEEP_S = 3600.0
#: how often a running job's progress is looked at
POLL_S = 0.25
#: the header's mark: four squares, two filled
FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
           '<path d="M4 4h7v7H4z" fill="#8E72EE"/><path d="M13 4h7v7h-7z" fill="none" '
           'stroke="#8E72EE" stroke-width="1.6"/><path d="M4 13h7v7H4z" fill="none" '
           'stroke="#00D7D2" stroke-width="1.6"/><path d="M13 13h7v7h-7z" fill="#00D7D2"/>'
           '</svg>')


class _Static(StaticFiles):
    """Static files that the browser must revalidate."""

    def file_response(self, *args, **kwargs):
        r = super().file_response(*args, **kwargs)
        r.headers["Cache-Control"] = "no-cache"
        return r


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, default=str)


def _safe(obj):
    """NaN and infinity become null: the page reads JSON, which has neither."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj


def create_app(cases_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Atlas", docs_url=None, redoc_url=None, openapi_url=None)
    stamp = str(int(time.time()))
    sessions: dict[str, dict] = {}
    app.state.sessions = sessions
    app.state.stamp = stamp

    app.mount("/static", _Static(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
            html = fh.read().replace("__V__", stamp)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/favicon.ico")
    async def favicon():
        # browsers ask for it whatever the page names; a 404 is noise in the console
        return Response(FAVICON, media_type="image/svg+xml",
                        headers={"Cache-Control": "max-age=86400"})

    @app.get("/fonts.css")
    async def font_css():
        return Response(fonts.font_css(), media_type="text/css",
                        headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    async def health():
        return JSONResponse({"ok": True, "app": "atlas-studio", "stamp": stamp,
                             "sessions": len(sessions)})

    def _expire():
        now = time.time()
        for sid in [k for k, v in sessions.items()
                    if not v["attached"] and now - v["seen"] > KEEP_S]:
            sessions.pop(sid)["session"].close()

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        send_lock = asyncio.Lock()

        async def send(obj):
            async with send_lock:
                await sock.send_text(_dumps(_safe(obj)))

        # the first message must be hello: a new session, or this page's own back
        try:
            first = json.loads(await sock.receive_text())
        except (WebSocketDisconnect, RuntimeError, ValueError):
            return
        _expire()
        want = first.get("session") if isinstance(first, dict) else None
        entry = sessions.get(want) if isinstance(want, str) else None
        if entry is None or entry["attached"]:
            sid = secrets.token_hex(12)
            session = await asyncio.to_thread(Session, cases_dir)
            entry = {"session": session, "attached": True, "seen": time.time()}
            sessions[sid] = entry
            resumed = False
        else:
            sid, session, resumed = want, entry["session"], True
            entry["attached"] = True
        await send({"type": "hello", "session": sid, "resumed": resumed, "stamp": stamp})
        await send(Session.catalog())
        await send(await asyncio.to_thread(session.state))
        watch = protocol.Watch()
        last_status: dict = {}

        async def pusher():
            nonlocal last_status
            while True:
                try:
                    frames = await asyncio.to_thread(protocol.run_frame, session, watch)
                    for f in frames:
                        await send(f)
                    st = await asyncio.to_thread(session.status)
                    if st != last_status:
                        last_status = st
                        await send({"type": "status", "status": st})
                except (WebSocketDisconnect, RuntimeError):
                    return
                await asyncio.sleep(POLL_S)

        task = asyncio.create_task(pusher())
        try:
            while True:
                text = await sock.receive_text()
                try:
                    msg = json.loads(text)
                except ValueError:
                    await send({"type": "reply", "re": None, "ok": False,
                                "error": "not JSON"})
                    continue
                reply, pushes = await asyncio.to_thread(protocol.handle, session, msg)
                for p in pushes:
                    await send(p)
                await send(reply)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            task.cancel()
            entry["attached"] = False
            entry["seen"] = time.time()

    return app


def serve(port: int = 8021, host: str = "127.0.0.1", open_browser: bool = False,
          cases_dir: str | None = None) -> None:
    import uvicorn
    app = create_app(cases_dir)
    url = f"http://{host}:{port}/"
    print(f"Atlas studio on {url}  (Ctrl+C stops it)", flush=True)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info", ws="websockets")


__all__ = ["create_app", "serve", "STATIC"]
