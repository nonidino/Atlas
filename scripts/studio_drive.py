"""Drive the studio's real page with real input, headless: a walk of clicks, drags and
keys, checked as it goes, with screenshots and every console error.

    python scripts/studio_drive.py scripts/studio_walks/p2.py --out out/studio/p2
    python scripts/studio_drive.py WALK --theme dark --size 1090x620

Headless Edge (or Chrome) is started with its DevTools port and loads the running
server's page (``--url``, the studio on 8021 by default).  The walk is a Python file
with a ``walk(page)`` function; ``page`` sends **trusted** input through the DevTools
protocol (``Input.dispatchMouseEvent`` and ``Input.dispatchKeyEvent``: the browser
treats them as a person's, unlike events a script dispatches), reads the page with
JavaScript, checks what it should (``page.check``), and saves screenshots.  The
record (``walk.json``: every check, every console error, the screenshots) goes to
``--out``.  The run fails if a check fails or the console shows any error.

This exists because the app's browser pane cannot take input while it is hidden; a
page check must not depend on a window being in front.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
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


class CheckFailed(AssertionError):
    pass


class Page:
    """The page under test: trusted input, JavaScript, checks and screenshots."""

    def __init__(self, ws, out: str, timeout: float):
        self.ws = ws
        self.out = out
        self.timeout = timeout
        self.n = 0
        self.events: list[dict] = []
        self.checks: list[dict] = []
        self.errors: list[str] = []
        self.shots: list[str] = []
        self.clicks = 0
        self.chooser = None                 # the last file chooser the page opened

    # -- the protocol --------------------------------------------------------------
    def call(self, method, **params):
        self.n += 1
        mid = self.n
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv(timeout=self.timeout))
            if m.get("id") == mid:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})
            self._event(m)

    def _event(self, m):
        meth = m.get("method", "")
        p = m.get("params", {})
        if meth == "Page.fileChooserOpened":
            self.chooser = p.get("backendNodeId")
        elif meth == "Runtime.exceptionThrown":
            d = p.get("exceptionDetails", {})
            self.errors.append("exception: " + str((d.get("exception") or {}).get("description") or d.get("text")))
        elif meth == "Runtime.consoleAPICalled" and p.get("type") == "error":
            self.errors.append("console.error: " + " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", [])))
        elif meth == "Log.entryAdded" and (p.get("entry") or {}).get("level") == "error":
            e = p["entry"]
            self.errors.append(f"log: {e.get('text')} {e.get('url', '')}")

    def pump(self, seconds: float):
        """Read events for a while (the console) without sending anything."""
        end = time.time() + seconds
        while time.time() < end:
            try:
                m = json.loads(self.ws.recv(timeout=max(0.01, end - time.time())))
            except TimeoutError:
                break
            self._event(m)

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=f"(async () => {{ return ({expr}); }})()",
                      awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"js failed: {expr[:80]}: {r['exceptionDetails'].get('exception', {}).get('description')}")
        return (r.get("result") or {}).get("value")

    def wait_for(self, expr: str, timeout: float = 30.0, what: str = ""):
        t0 = time.time()
        while True:
            v = self.js(expr)
            if v:
                return v
            if time.time() - t0 > timeout:
                raise CheckFailed(f"timed out waiting for {what or expr}")
            self.pump(0.2)

    def settle(self, seconds: float = 0.6):
        """Until the page has no edit in flight (the Rebuilding chip is hidden)."""
        self.pump(0.25)
        t0 = time.time()
        while self.js("!!document.querySelector('.busy:not([hidden])')") and time.time() - t0 < 60:
            self.pump(0.2)
        self.pump(seconds)

    # -- input -----------------------------------------------------------------------
    def mouse(self, kind, x, y, button="left", buttons=1, mods=0, count=1):
        self.call("Input.dispatchMouseEvent", type=kind, x=float(x), y=float(y), button=button,
                  buttons=buttons, modifiers=mods, clickCount=count, pointerType="mouse")

    def click(self, x, y, mods=0, count=1):
        self.mouse("mouseMoved", x, y, button="none", buttons=0, mods=mods)
        self.mouse("mousePressed", x, y, mods=mods, count=count)
        self.mouse("mouseReleased", x, y, buttons=0, mods=mods, count=count)
        self.clicks += 1
        self.pump(0.05)

    def dblclick(self, x, y):
        self.click(x, y, count=1)
        self.click(x, y, count=2)

    def drag(self, pts, mods=0, steps=8):
        (x0, y0) = pts[0]
        self.mouse("mouseMoved", x0, y0, button="none", buttons=0, mods=mods)
        self.mouse("mousePressed", x0, y0, mods=mods)
        last = (x0, y0)
        for (x, y) in pts[1:]:
            for k in range(1, steps + 1):
                xi = last[0] + (x - last[0]) * k / steps
                yi = last[1] + (y - last[1]) * k / steps
                self.mouse("mouseMoved", xi, yi, buttons=1, mods=mods)
                self.pump(0.01)
            last = (x, y)
        self.mouse("mouseReleased", last[0], last[1], buttons=0, mods=mods)
        self.clicks += 1
        self.pump(0.05)

    # Enter carries no text: in a one-line field a real key inserts nothing, and a
    # carriage return sent as text lands in the field's value
    KEYS = {"Enter": (13, ""), "Escape": (27, ""), "Backspace": (8, ""), "Tab": (9, ""), "Delete": (46, ""),
            "ArrowLeft": (37, ""), "ArrowRight": (39, ""), "ArrowUp": (38, ""), "ArrowDown": (40, "")}

    #: the editing command a shortcut performs (Ctrl or Cmd held)
    COMMANDS = {"a": "selectAll", "z": "undo", "c": "copy", "v": "paste", "x": "cut"}

    def key(self, key: str, mods: int = 0):
        """One key pressed and released.  ``mods``: 1 Alt, 2 Ctrl, 4 Meta, 8 Shift.  A
        shortcut (Ctrl or Meta held) types no text, as a real one does not."""
        code, text = self.KEYS.get(key, (ord(key.upper()) if len(key) == 1 else 0, key if len(key) == 1 else ""))
        base = {"key": key, "windowsVirtualKeyCode": code, "modifiers": mods}
        if len(key) == 1:
            base["code"] = f"Key{key.upper()}" if key.isalpha() else ("Digit" + key if key.isdigit() else "")
        extra = {}
        if mods & 6:
            text = ""
            if key.lower() in self.COMMANDS:
                extra["commands"] = [self.COMMANDS[key.lower()]]
        self.call("Input.dispatchKeyEvent", type="keyDown" if text else "rawKeyDown", text=text, **base, **extra)
        self.call("Input.dispatchKeyEvent", type="keyUp", **base)
        self.pump(0.05)

    def type(self, text: str):
        for ch in text:
            self.call("Input.dispatchKeyEvent", type="keyDown", key=ch, text=ch)
            self.call("Input.dispatchKeyEvent", type="keyUp", key=ch)
        self.pump(0.05)

    # -- the page's own geometry --------------------------------------------------------
    def at(self, selector: str, index: int = 0):
        """The centre of an element on screen."""
        r = self.js(f"(() => {{ const e = document.querySelectorAll({json.dumps(selector)})[{index}]; if (!e) return null; "
                    f"e.scrollIntoView({{block: 'nearest'}}); const b = e.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2, b.width, b.height]; }})()")
        if not r:
            raise CheckFailed(f"no element {selector} [{index}]")
        if r[2] <= 0 or r[3] <= 0:
            raise CheckFailed(f"{selector} [{index}] has no size on screen")
        return r[0], r[1]

    def click_on(self, selector: str, index: int = 0):
        x, y = self.at(selector, index)
        self.click(x, y)

    def button(self, text: str, within: str = "body"):
        """The centre of the visible button whose text is exactly this."""
        r = self.js(f"(() => {{ const b = [...document.querySelectorAll({json.dumps(within)} + ' button')].find(e => e.textContent.trim() === {json.dumps(text)} && e.offsetParent !== null); "
                    f"if (!b) return null; b.scrollIntoView({{block: 'nearest'}}); const r = b.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }})()")
        if not r:
            raise CheckFailed(f"no visible button '{text}' in {within}")
        return r

    def dismiss_toasts(self):
        """Close every notice with its own close button, as a person would."""
        for _ in range(8):
            r = self.js("(() => { const x = document.querySelector('.toast .x'); if (!x) return null; "
                        "const b = x.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; })()")
            if not r:
                return
            self.click(r[0], r[1])
            self.pump(0.15)

    def covered(self, x, y, selector_js: str):
        """What covers a point, if it is not the element the selector names."""
        return self.js(f"(() => {{ const want = {selector_js}; const top = document.elementFromPoint({x}, {y}); "
                       f"if (!want || !top || want === top || want.contains(top)) return null; "
                       f"return (top.className || top.tagName) + ': ' + (top.textContent || '').trim().slice(0, 40); }})()")

    def press(self, text: str, within: str = "body"):
        x, y = self.button(text, within)
        find = (f"[...document.querySelectorAll({json.dumps(within)} + ' button')].find(e => e.textContent.trim() === "
                f"{json.dumps(text)} && e.offsetParent !== null)")
        if self.covered(x, y, find):
            self.dismiss_toasts()
            x, y = self.button(text, within)
            what = self.covered(x, y, find)
            if what:
                raise CheckFailed(f"'{text}' is covered by {what}")
        self.click(x, y)

    def world(self, x: float, y: float):
        """A world point (the drawing's units, y up) on screen, through the editor's view."""
        return self.js(f"(() => {{ const ed = window.atlas.geometry.editor; const cv = ed.stage.canvas.getBoundingClientRect(); "
                       f"const p = ed.stage.view.projectToView(new paper.Point({x}, {-y})); return [cv.left + p.x, cv.top + p.y]; }})()")

    def fill(self, selector: str, text: str, index: int = 0, enter: bool | None = None):
        """Type into an input as a person would: click, select all, type, then Enter
        (or, inside a dialog, where Enter would submit it, Tab to the next field)."""
        x, y = self.at(selector, index)
        self.click(x, y)
        self.key("a", mods=2)
        if text:
            self.type(text)
        else:
            self.key("Backspace")                    # what clears a selection
        if enter is None:
            enter = not selector.lstrip().startswith(".dialog")
        self.key("Enter" if enter else "Tab")

    def choose_file(self, path: str, click):
        """A file picker the page opens on ``click()``, answered with ``path`` as a
        person would pick it (the browser's own file chooser, intercepted)."""
        self.call("Page.setInterceptFileChooserDialog", enabled=True)
        self.chooser = None
        try:
            click()
            t0 = time.time()
            while self.chooser is None:
                if time.time() - t0 > 20:
                    raise CheckFailed("no file chooser opened")
                self.pump(0.1)
            node = self.chooser
            self.call("DOM.enable")
            self.call("DOM.setFileInputFiles", files=[os.path.abspath(path)], backendNodeId=node)
        finally:
            self.call("Page.setInterceptFileChooserDialog", enabled=False)
        self.pump(0.3)

    # -- checks and screenshots -----------------------------------------------------------
    def check(self, name: str, ok, detail=None):
        self.checks.append({"check": name, "passed": bool(ok), "detail": detail})
        mark = "pass" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f": {detail}" if detail is not None else ""), flush=True)
        if not ok:
            raise CheckFailed(f"{name}: {detail}")

    def shot(self, name: str):
        d = self.call("Page.captureScreenshot", format="png")
        path = os.path.join(self.out, f"{name}.png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(d["data"]))
        self.shots.append(path.replace("\\", "/"))
        return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("walk", help="a Python file with walk(page)")
    ap.add_argument("--url", default="http://127.0.0.1:8021/")
    ap.add_argument("--size", default="1090x620")
    ap.add_argument("--theme", choices=("light", "dark"), default="light")
    ap.add_argument("--out", default=None, help="where the record and screenshots go")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = importlib.util.spec_from_file_location("walk", a.walk)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = a.out or os.path.join("out", "studio", os.path.splitext(os.path.basename(a.walk))[0])
    os.makedirs(out, exist_ok=True)
    from websockets.sync.client import connect
    w, h = (int(x) for x in a.size.split("x"))
    port = _port()
    prof = tempfile.mkdtemp(prefix="studio-drive-")
    args = [_browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
            f"--window-size={w},{h}", "about:blank"]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    record = {"walk": a.walk.replace("\\", "/"), "url": a.url, "size": a.size, "theme": a.theme,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    page = None
    status = 1
    try:
        t0 = time.time()
        while True:
            try:
                tabs = json.loads(direct.open(f"http://127.0.0.1:{port}/json", timeout=2).read())
                target = next(t for t in tabs if t.get("type") == "page")
                break
            except Exception:
                if time.time() - t0 > 30:
                    raise SystemExit("the browser's DevTools port did not answer")
                time.sleep(0.3)
        with connect(target["webSocketDebuggerUrl"], max_size=None, proxy=None) as ws:
            page = Page(ws, out, a.timeout)
            page.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)
            page.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": a.theme}])
            for dom in ("Page", "Runtime", "Log"):
                page.call(f"{dom}.enable")
            page.call("Page.navigate", url=a.url)
            # the first page after a server starts waits for the server's first imports
            page.wait_for("!!(window.atlas && window.atlas.store && window.atlas.store.state)", 300, "the server's state")
            page.js(f"document.documentElement.dataset.theme = '{a.theme}'")
            page.pump(0.5)
            try:
                mod.walk(page)
                page.pump(0.5)
                page.check("no console errors", not page.errors, page.errors[:5] or None)
                status = 0
            except CheckFailed as exc:
                print(f"FAILED: {exc}", flush=True)
                try:
                    page.shot("failure")
                except Exception:
                    pass
    finally:
        if page is not None:
            record.update({"checks": page.checks, "console_errors": page.errors, "screenshots": page.shots,
                           "trusted_clicks_and_drags": page.clicks, "passed": status == 0,
                           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
            with open(os.path.join(out, "walk.json"), "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=1)
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)
    print(f"{'passed' if status == 0 else 'FAILED'}: {len(page.checks) if page else 0} checks, "
          f"{len(page.errors) if page else '?'} console errors; record in {out}/walk.json", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
