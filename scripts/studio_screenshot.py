"""A screenshot of the studio's real page, at an exact size, after its state arrives.

    python scripts/studio_screenshot.py out/studio/p1.png --theme dark \
        --js "window.atlas.store.tab" --size 1090x620

Headless Edge (or Chrome) is started with its DevTools port, the page is loaded,
the script waits until the page holds the server's state (``window.atlas.store.state``),
runs the optional JavaScript (to open a tab, a view or a case), waits for the server
to settle, and saves the PNG.  A plain ``--screenshot`` cannot wait for a WebSocket:
it captured an empty canvas.  The page is the running server's; nothing is changed
but what the JavaScript asks.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

EDGE = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def _browser() -> str:
    for p in EDGE:
        if os.path.isfile(p):
            return p
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    raise SystemExit("no Edge or Chrome found")


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--url", default="http://127.0.0.1:8021/")
    ap.add_argument("--size", default="1090x620")
    ap.add_argument("--theme", choices=("light", "dark"), default="light")
    ap.add_argument("--js", action="append", default=[], help="JavaScript to run, in order")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds after the last step")
    ap.add_argument("--timeout", type=float, default=90.0)
    a = ap.parse_args(argv)
    from websockets.sync.client import connect
    w, h = (int(x) for x in a.size.split("x"))
    port = _port()
    prof = tempfile.mkdtemp(prefix="studio-shot-")
    args = [_browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
            f"--window-size={w},{h}", "about:blank"]
    if a.theme == "dark":
        args.insert(1, "--force-dark-mode")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        t0 = time.time()
        while True:
            try:
                tabs = json.loads(direct.open(f"http://127.0.0.1:{port}/json", timeout=2).read())
                page = next(t for t in tabs if t.get("type") == "page")
                break
            except Exception:
                if time.time() - t0 > 30:
                    raise SystemExit("the browser's DevTools port did not answer")
                time.sleep(0.3)
        with connect(page["webSocketDebuggerUrl"], max_size=None, proxy=None) as ws:
            n = [0]

            def call(method, **params):
                n[0] += 1
                ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
                while True:
                    m = json.loads(ws.recv(timeout=a.timeout))
                    if m.get("id") == n[0]:
                        if "error" in m:
                            raise RuntimeError(f"{method}: {m['error']}")
                        return m.get("result", {})

            def js(expr):
                r = call("Runtime.evaluate", expression=expr, awaitPromise=True,
                         returnByValue=True)
                return (r.get("result") or {}).get("value")

            call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1,
                 mobile=False)
            call("Emulation.setEmulatedMedia", features=[
                {"name": "prefers-color-scheme", "value": a.theme}])
            call("Page.enable")
            call("Page.navigate", url=a.url)
            t0 = time.time()
            while not js("!!(window.atlas && window.atlas.store && window.atlas.store.state)"):
                if time.time() - t0 > a.timeout:
                    raise SystemExit("the page never received the server's state")
                time.sleep(0.3)
            js(f"document.documentElement.dataset.theme = '{a.theme}'")
            for step in a.js:
                js(f"(async () => {{ {step} }})()")
                time.sleep(0.5)
            time.sleep(a.settle)
            shot = call("Page.captureScreenshot", format="png")
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        print(f"saved {a.out} ({w}x{h}, {a.theme})")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
