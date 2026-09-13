"""Edit the car's geometry with the mouse.

    python scripts/car_editor.py

opens a page at http://127.0.0.1:8017/ that draws `atlas/cases/car_geometry.json`
and lets you drag it.  Save writes the JSON back; Check runs the REAL model --
the same `car_bodies` and `windows_from_geometry` the march uses -- and tells
you whether what you drew is something this model can march.

**Why this exists.**  A geometry that only lives as literals inside a Python
function can only be changed by someone willing to read that function, and the
person who knows what a Formula One car looks like is not necessarily that
person.  Three attempts at tracing one automatically produced a car that was
not good enough; this hands the shape back to the eye that can judge it.

Nothing here is part of the march.  It is a tool, it is not imported by any
case, and it writes exactly one file.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

GEOM = os.path.join(ROOT, "atlas", "cases", "car_geometry.json")
PAGE = os.path.join(HERE, "car_editor.html")
BACKDROPS = os.path.join(ROOT, "out", "backdrops")


def _load():
    with open(GEOM, encoding="utf-8") as fh:
        return json.load(fh)


def _save(doc):
    """Write the geometry, keeping the previous one beside it."""
    if os.path.isfile(GEOM):
        bak = GEOM + ".bak"
        shutil.copyfile(GEOM, bak)
    tmp = GEOM + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    for k in range(6):                      # OneDrive holds fresh files
        try:
            os.replace(tmp, GEOM)
            break
        except PermissionError:
            if k == 5:
                raise
            time.sleep(0.3 * (k + 1))
    return GEOM


def check(doc):
    """Run the model's own checks on a geometry, without saving it."""
    from atlas.cases import racelab as RL

    rep: dict = {"errors": [], "warnings": [], "ok": []}
    try:
        objs, flat = RL.car_bodies(geometry=doc)
    except Exception as exc:
        rep["errors"].append("car_bodies failed: %s" % exc)
        return rep

    wheels = [(float(w["x"]), float(w["y"]), float(w["r"]))
              for w in doc.get("wheels", [])]
    plates = [b for b in flat if str(b.group) != "wheel"]

    # 1. the tiling's horizontal seam
    top = max(max(b.y_le, b.y_te) for b in plates)
    seam = 112.0
    (rep["ok"] if top < seam - 4 else rep["errors"]).append(
        "body tops at y=%.1f against the tiling seam at %.0f (needs 4 cells "
        "of clearance; two 128-tall rows in a 240-tall box force the seam "
        "there, so it cannot move)" % (top, seam))

    # 2. nothing inside a wheel
    bad = []
    for b in plates:
        for k in range(21):
            t = k / 20.0
            px = b.x_le + t * (b.x_te - b.x_le)
            py = b.y_le + t * (b.y_te - b.y_le)
            if any((px - cx) ** 2 + (py - cy) ** 2 < (r - 1.0) ** 2
                   for cx, cy, r in wheels):
                bad.append(b.body_id)
                break
    (rep["ok"] if not bad else rep["errors"]).append(
        "plates inside a wheel: %s (a plate inside a wheel double-counts that "
        "wheel's drag)" % (", ".join(sorted(set(bad))) or "none"))

    # 3. the duct has to sit inside the body
    ducts = [b for b in plates if str(b.group) == "duct"]
    if len(ducts) == 2:
        lo_, hi_ = sorted(ducts, key=lambda b: b.y_le)
        x0 = max(min(lo_.x_le, lo_.x_te), min(hi_.x_le, hi_.x_te))
        x1 = min(max(lo_.x_le, lo_.x_te), max(hi_.x_le, hi_.x_te))
        shell = [b for b in plates if str(b.group) in ("body", "floor")]
        worst, at = 1e9, None
        x = x0
        while x <= x1:
            ups = [b for b in shell
                   if min(b.x_le, b.x_te) <= x <= max(b.x_le, b.x_te)]
            if ups:
                ys = []
                for b in ups:
                    dx = b.x_te - b.x_le
                    t = 0.0 if abs(dx) < 1e-9 else (x - b.x_le) / dx
                    ys.append(b.y_le + t * (b.y_te - b.y_le))
                m = max(ys) - max(lo_.y_le, hi_.y_le)
                if m < worst:
                    worst, at = m, x
            x += 2.0
        if at is None:
            rep["warnings"].append("no body found over the duct's x-range")
        else:
            (rep["ok"] if worst > 0 else rep["errors"]).append(
                "duct roof clears the body by %+.1f cells (worst at x=%.0f); "
                "the 32 cells between the duct walls are DEVICE_CELLS, where "
                "the radiator and turbine planes sit" % (worst, at))
    else:
        rep["warnings"].append("expected two duct plates, found %d" % len(ducts))

    # 4. the front wing must not PENETRATE the front tyre.
    #    Comparing x-extents is the wrong test and said so: a wing whose
    #    trailing edge sits exactly ON the tread overlaps in x while touching
    #    nothing.  What matters is the distance to the tyre's surface.
    fw = [b for b in plates if str(b.group) == "front-wing"]
    if fw and wheels:
        cx, cy, r = min(wheels, key=lambda w: w[0])
        worst, who = 1e9, None
        for b in fw:
            for k in range(21):
                t = k / 20.0
                px = b.x_le + t * (b.x_te - b.x_le)
                py = b.y_le + t * (b.y_te - b.y_le)
                gap = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 - r
                if gap < worst:
                    worst, who = gap, b.body_id
        (rep["ok"] if worst >= -0.5 else rep["errors"]).append(
            "front wing clears the front tyre's surface by %+.2f cells "
            "(closest: %s); touching is fine, penetrating double-counts the "
            "tyre's drag" % (worst, who))

    # 5. the rear tyre must clear the earliest turbine cut
    if wheels:
        cx, cy, r = max(wheels, key=lambda w: w[0])
        (rep["ok"] if cx - r > 400.0 else rep["errors"]).append(
            "rear tyre leading edge x=%.1f against the turbine cut's end at "
            "x=400 (wx=128 and device_overlap_max=16 force the two device "
            "cuts exactly 112 cells apart)" % (cx - r))

    # 6. the domain
    xs = [v for b in plates for v in (b.x_le, b.x_te)]
    ys = [v for b in plates for v in (b.y_le, b.y_te)]
    inside = (min(xs) > 2 and max(xs) < RL.RNX - 2
              and min(ys) > -1 and max(ys) < RL.RNY - 2)
    (rep["ok"] if inside else rep["errors"]).append(
        "car occupies x %.0f..%.0f, y %.0f..%.0f in a %dx%d box"
        % (min(xs), max(xs), min(ys), max(ys), RL.RNX, RL.RNY))

    # 7. the layout -- the authoritative one, the same call the march makes
    try:
        tiling, info = RL.windows_from_geometry(flat, objects=objs)
        rep["ok"].append(
            "the window layout resolves: %d windows, cuts at %s, %.1f%% of the "
            "car inside a cut"
            % (tiling.n_windows,
               "/".join(str(int(o)) for o, _ in tiling.offsets[:7]),
               100 * info["banded_force_fraction"]))
        rep["layout"] = {"windows": tiling.n_windows,
                         "banded": info["banded_force_fraction"]}
    except Exception as exc:
        rep["errors"].append("NO WINDOW LAYOUT: %s" % exc)

    # 8. chain continuity, informational
    gaps = []
    ends = [(b.body_id, (b.x_te, b.y_te)) for b in plates]
    starts = [(b.body_id, (b.x_le, b.y_le)) for b in plates]
    for bid, e in ends:
        near = min((((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2) ** 0.5, sid)
                   for sid, s in starts if sid != bid)
        if 0.75 < near[0] < 14.0:
            gaps.append("%s -> %s is %.1f cells" % (bid, near[1], near[0]))
    if gaps:
        rep["warnings"].append("small gaps in the outline: " + "; ".join(gaps[:6]))
    return rep


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/car_editor.html"):
            with open(PAGE, encoding="utf-8") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if path == "/geometry":
            return self._send(200, _load())
        if path == "/backdrops":
            names = []
            if os.path.isdir(BACKDROPS):
                names = sorted(f for f in os.listdir(BACKDROPS)
                               if f.lower().endswith((".svg", ".png", ".jpg")))
            return self._send(200, names)
        if path.startswith("/backdrop/"):
            name = os.path.basename(path[len("/backdrop/"):])
            f = os.path.join(BACKDROPS, name)
            if not os.path.isfile(f):
                return self._send(404, {"error": "no such backdrop"})
            ct = ("image/svg+xml" if name.lower().endswith(".svg")
                  else "image/png" if name.lower().endswith(".png")
                  else "image/jpeg")
            with open(f, "rb") as fh:
                return self._send(200, fh.read(), ct)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            doc = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            return self._send(400, {"error": "bad json: %s" % exc})
        if self.path == "/check":
            try:
                return self._send(200, check(doc))
            except Exception as exc:
                return self._send(200, {"errors": ["check crashed: %s" % exc],
                                        "warnings": [], "ok": []})
        if self.path == "/save":
            try:
                _save(doc)
                from atlas.cases import racelab as RL
                RL.load_geometry(reload=True)
                return self._send(200, {"saved": GEOM,
                                        "backup": GEOM + ".bak"})
            except Exception as exc:
                return self._send(500, {"error": str(exc)})
        return self._send(404, {"error": "not found"})


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(BACKDROPS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    url = "http://127.0.0.1:%d/" % a.port
    print("car editor on %s" % url)
    print("  geometry : %s" % GEOM)
    print("  backdrops: %s  (drop a .svg or .png in here to trace over)"
          % BACKDROPS)
    print("  Ctrl-C to stop")
    if a.open:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
