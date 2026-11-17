"""P2's page check (integrated-ui-plan §8): geometry II, walked with real input.

Done-when, each checked in the page:
* a dimensioned plate reaches "fully defined";
* changing W moves every dependent shape;
* the same seed gives the same inclusions twice (byte for byte in the case file);
* a formula field's κ range matches the server's evaluation.
And every control P2 added is used once: the Dimension tool on an edge, a hole and two
points; Fix; Pattern; Mirror; Split; Channel; Scatter; a path's constraints held while
its point is dragged; a fillet; an image field; an image to trace; a sweep from Run.

    python scripts/studio_drive.py scripts/studio_walks/p2.py --out out/studio/p2
"""

import json
import os
import struct
import zlib

W0, H0 = 0.6, 0.3


def _png(path, w=32, h=16):
    """A left-to-right grey ramp, written without any imaging library."""
    row = b"\x00" + b"".join(bytes([round(255 * i / (w - 1))] * 3) for i in range(w))
    rows = row * h

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(data)


def feats(page):
    return page.js("window.atlas.store.state.geometry.features")


def feat(page, kind, n=0):
    fs = [f for f in feats(page) if f["kind"] == kind]
    return fs[n] if len(fs) > n else None


def params(page):
    return {p["name"]: p for p in page.js("window.atlas.store.state.geometry.parameters")}


def dof(page):
    page.wait_for("window.atlas.geometry.freedom() !== null", 20, "the degrees of freedom")
    page.pump(0.4)
    return page.js("window.atlas.geometry.freedom()")


def tree(page, fid, shift=False):
    """Select a shape by its row in the Features tree, with a real click."""
    r = page.js(f"""(() => {{
        const name = (window.atlas.store.state.geometry.features.find(f => f.id === '{fid}') || {{}}).name;
        const list = document.querySelector('#geo-inspector').parentElement.querySelector('aside.panel .items');
        const row = [...list.querySelectorAll('.item')].find(e => e.querySelector('.name')?.textContent === name);
        if (!row) return null;
        row.scrollIntoView({{block: 'nearest'}});
        const b = row.getBoundingClientRect();
        return [b.left + 40, b.top + b.height / 2];
    }})()""")
    if not r:
        raise AssertionError(f"no tree row for {fid}")
    page.click(r[0], r[1], mods=8 if shift else 0)
    page.pump(0.3)


def tool(page, t):
    page.click_on(f'.rail .tool[data-tool="{t}"]')
    page.pump(0.2)


def walk(page):
    # -- a plate --------------------------------------------------------------------
    if not page.js("!document.querySelector('.startover').hidden"):
        page.press("New geometry…", "#geo-inspector")
        page.wait_for("!document.querySelector('.startover').hidden", 10, "the start screen")
    page.fill('.startover input[aria-label="width"]', str(W0))
    page.fill('.startover input[aria-label="height"]', str(H0))
    page.click_on('.startcard[data-start="plate"]')
    page.wait_for(f"(() => {{ const f = window.atlas.store.state.geometry.features; return f.length === 1 && Math.abs(f[0].w - {W0}) < 1e-9; }})()", 60, "the plate")
    page.settle()
    page.check("a plate has 4 degrees of freedom", dof(page) == 4, dof(page))

    # -- dimension it: width, height, fix ----------------------------------------------
    tool(page, "dimension")
    x, y = page.world(0.3 * W0, H0)
    page.click(x, y)
    page.settle()
    x, y = page.world(W0, 0.45 * H0)
    page.click(x, y)
    page.settle()
    plate = feats(page)[0]
    ps = params(page)
    page.check("the Dimension tool made W and H and bound the plate to them",
               plate["bind"] == {"w": "W", "h": "H"} and abs(ps["W"]["value"] - W0) < 1e-12 and abs(ps["H"]["value"] - H0) < 1e-12, plate["bind"])
    page.check("width and height take 2 of the 4", dof(page) == 2, dof(page))
    page.key("v")
    x, y = page.world(0.15 * W0, 0.5 * H0)
    page.click(x, y)
    page.pump(0.3)
    page.press("Fix", "#geo-inspector")
    page.settle()
    page.check("a dimensioned, fixed plate is fully defined", dof(page) == 0, dof(page))
    chip = page.js("document.querySelector('.dofchip')?.textContent")
    page.check("the bar says so", chip == "✓ fully defined", chip)
    page.shot("p2-fully-defined")

    # -- a hole, dimensioned against the plate -------------------------------------------
    page.key("Escape")
    page.press("Cut", ".stage .bar")
    tool(page, "ellipse")
    a = page.world(0.40, 0.11)
    b = page.world(0.48, 0.19)
    page.drag([a, b], mods=8)                    # Shift: a circle
    page.settle()
    hole = feat(page, "ellipse")
    page.check("the hole was drawn by dragging", hole is not None and hole["role"] == "cut", hole and [hole["cx"], hole["cy"], hole["rx"]])
    page.check("a hole adds 4 degrees of freedom", dof(page) == 4, dof(page))
    tool(page, "dimension")
    x, y = page.world(hole["cx"] + 0.8 * hole["rx"], hole["cy"])
    page.click(x, y)                             # the hole's outline: its diameter
    page.settle()
    x, y = page.world(W0, 0.0)                   # the plate's lower-right corner
    page.click(x, y)
    page.pump(0.3)
    hole = feat(page, "ellipse")
    x, y = page.world(hole["cx"], hole["cy"])    # the hole's centre, Shift: both directions
    page.click(x, y, mods=8)
    page.settle()
    hole = feat(page, "ellipse")
    ps = params(page)
    page.check("the hole's diameter and position are parameters",
               "d" in ps and set(hole["bind"]) >= {"rx", "ry", "cx", "cy"} and "W" in hole["bind"]["cx"], hole["bind"])
    page.check("the plate with its hole is fully defined", dof(page) == 0, dof(page))
    page.shot("p2-dimensions")

    # -- change W: every dependent shape moves -------------------------------------------
    page.key("Escape")
    cx0 = hole["cx"]
    page.fill('input[aria-label="Value of W"]', "0.8")
    page.settle()
    plate, hole = feats(page)[0], feat(page, "ellipse")
    page.check("changing W moves every dependent shape",
               abs(plate["w"] - 0.8) < 1e-12 and abs(hole["cx"] - (cx0 + 0.2)) < 1e-9,
               {"plate.w": plate["w"], "hole.cx": [cx0, hole["cx"]]})
    page.check("still fully defined", dof(page) == 0, dof(page))
    page.shot("p2-w-changed")

    # -- scatter: the same seed twice --------------------------------------------------
    tool(page, "select")
    page.key("Escape")
    page.pump(0.3)
    page.fill('#geo-inspector input[aria-label="count"]', "10")
    page.fill('#geo-inspector input[aria-label="seed"]', "7")
    page.press("Scatter", "#geo-inspector")
    page.settle()
    sc = feat(page, "scatter")
    page.check("a scatter placed its inclusions", sc is not None and len(sc["items"]) > 0, sc and len(sc["items"]))
    text1 = page.js("window.atlas.net.request('case.export').then(r => r.text)")
    page.key("z", mods=2)                        # undo it
    page.settle()
    page.check("undone", feat(page, "scatter") is None)
    page.fill('#geo-inspector input[aria-label="count"]', "10")
    page.fill('#geo-inspector input[aria-label="seed"]', "7")
    page.press("Scatter", "#geo-inspector")
    page.settle()
    text2 = page.js("window.atlas.net.request('case.export').then(r => r.text)")
    i1 = json.dumps([f for f in json.loads(text1)["geometry"]["features"] if f["kind"] == "scatter"][0]["items"])
    i2 = json.dumps([f for f in json.loads(text2)["geometry"]["features"] if f["kind"] == "scatter"][0]["items"])
    page.check("the same seed gives the same inclusions, byte for byte in the case file", i1 == i2, f"{len(i1)} bytes")
    page.shot("p2-scatter")

    # -- a region with a formula field: the page's range against the server's ---------
    page.key("Escape")
    page.press("Region", ".stage .bar")
    tool(page, "rect")
    page.drag([page.world(0.05, 0.05), page.world(0.25, 0.15)])
    page.settle()
    page.press("Formula", "#geo-inspector")
    page.settle()
    expr = "45 + 955 * exp(-((x - 0.15)^2 + (y - 0.1)^2) / 0.002)"
    page.fill('#geo-inspector input[aria-label^="Formula for"]', expr)
    page.settle(1.0)
    page.wait_for("[...document.querySelectorAll('#geo-inspector .ok-text, #geo-inspector .warn-text')].some(e => e.textContent.includes('server'))", 30, "the server's range")
    verdict = page.js("[...document.querySelectorAll('#geo-inspector .ok-text, #geo-inspector .warn-text')].map(e => e.textContent).find(t => t.includes('server'))")
    page.check("a formula field's κ range matches the server's evaluation", verdict.startswith("✓ server: same range"), verdict)
    page.shot("p2-formula-field")

    # -- an image field ---------------------------------------------------------------------
    img = os.path.join(page.out, "ramp.png")
    _png(img)
    page.choose_file(img, lambda: page.press("Image", "#geo-inspector"))
    page.settle(1.0)
    reg = [f for f in feats(page) if f["kind"] == "rect" and f["role"] == "region"][0]
    page.check("an image field maps grey onto a range", reg["field"] and reg["field"]["kind"] == "image", reg["field"] and reg["field"]["kind"])
    info = page.js(f"window.atlas.store.state.fields['{reg['id']}']")
    page.check("the server binned the image", info and info["levels"] >= 2, info)

    # -- split the plate by the region: what is outside, and the overlap -------------
    page.key("Escape")
    tool(page, "select")
    region = [f for f in feats(page) if f["kind"] == "rect" and f["role"] == "region"][0]
    tree(page, feats(page)[0]["id"])
    tree(page, region["id"], shift=True)
    n0 = len(feats(page))
    page.press("Split", "#geo-inspector")
    page.settle()
    page.check("Split cut one shape by another into two", len(feats(page)) == n0 + 4, len(feats(page)) - n0)
    page.key("z", mods=2)
    page.settle()

    # -- pattern the hole; mirror the region ------------------------------------------------
    page.key("Escape")
    hole = feat(page, "ellipse")
    tree(page, hole["id"])
    page.press("Pattern…", "#geo-inspector")
    page.wait_for("!!document.querySelector('.dialog')", 5, "the pattern dialog")
    page.fill(".dialog input.field", "2")         # copies
    page.fill(".dialog input.field", "-0.12", 1)  # step x
    page.press("Pattern", ".dialog")
    page.settle()
    pat = feat(page, "pattern")
    holes = page.js("window.atlas.store.state.case.domain.holes.length")
    page.check("a pattern repeats the hole", pat is not None and holes >= 2, {"holes": holes})
    page.key("Escape")
    tree(page, region["id"])
    page.press("Mirror…", "#geo-inspector")
    page.wait_for("!!document.querySelector('.dialog')", 5, "the mirror dialog")
    page.fill(".dialog input.field", "0.4")
    page.press("Mirror", ".dialog")
    page.settle()
    mir = feat(page, "mirror")
    page.check("a mirror reflects the region", mir is not None and mir["inputs"] == [region["id"]], mir and mir["inputs"])

    # -- a channel ---------------------------------------------------------------------------
    page.key("Escape")
    page.press("Add", ".stage .bar")
    tool(page, "channel")
    for (wx, wy) in [(0.05, 0.25), (0.3, 0.27), (0.55, 0.24)]:
        x, y = page.world(wx, wy)
        page.click(x, y)
    page.key("Enter")
    page.settle()
    ch = feat(page, "channel")
    page.check("a channel from a drawn centre line", ch is not None and ch["w"] > 0 and len(ch["points"]) == 3, ch and ch["w"])

    # -- a path held by its constraints while a point is dragged -------------------------
    page.key("Escape")
    page.press("Region", ".stage .bar")
    tool(page, "pen")
    quad = [(0.30, 0.04), (0.42, 0.05), (0.43, 0.13), (0.31, 0.12)]
    for (wx, wy) in quad:
        x, y = page.world(wx, wy)
        page.click(x, y)
    x, y = page.world(*quad[0])
    page.click(x, y)                              # the first point closes it
    page.settle()
    path = [f for f in feats(page) if f["kind"] == "path"][-1]
    page.check("the pen closed a shape", path["closed"] and len(path["points"]) == 4)
    sel = "#geo-inspector select[aria-label='Constraint']"
    page.js(f"(() => {{ const s = document.querySelector(\"{sel}\"); s.value = 'horizontal'; s.dispatchEvent(new Event('change')); }})()")
    page.click_on("#geo-inspector .add-constraint")
    page.settle()
    path = [f for f in feats(page) if f["kind"] == "path"][-1]
    page.check("a horizontal constraint levels e0", abs(path["points"][0][1] - path["points"][1][1]) < 1e-9, path["points"][:2])
    p1 = path["points"][1]
    tool(page, "select")                          # the pen stays on until Select
    tree(page, path["id"])
    a = page.world(*p1)
    b = page.world(p1[0] + 0.03, p1[1] + 0.04)
    page.drag([a, b], steps=10)
    page.settle()
    path = [f for f in feats(page) if f["kind"] == "path"][-1]
    page.check("dragging a point keeps e0 level (planegcs solved the drag)",
               abs(path["points"][0][1] - path["points"][1][1]) < 1e-9 and abs(path["points"][1][0] - (p1[0] + 0.03)) < 0.01,
               path["points"][:2])
    page.fill("#geo-inspector input[aria-label='Radius']", "0.01")
    page.press("Fillet", "#geo-inspector")
    page.settle()
    path = [f for f in feats(page) if f["kind"] == "path"][-1]
    page.check("a fillet rounds a corner with an arc", "arc" in path["edges"] and len(path["points"]) == 5, path["edges"])
    page.shot("p2-path")

    # -- an image to trace -------------------------------------------------------------------
    page.key("Escape")
    page.pump(0.3)
    page.choose_file(img, lambda: page.press("Image to trace…", "#geo-inspector"))
    page.wait_for("!!document.querySelector('.dialog')", 5, "the trace dialog")
    page.fill(".dialog input.field", "0.4")
    page.press("Place", ".dialog")
    page.pump(1.0)
    page.check("the picture shows under the drawing", page.js("window.atlas.geometry.under.active"))
    page.shot("p2-trace")

    # -- a sweep of W, from Run ----------------------------------------------------------------
    page.click_on('input[aria-label="Sweep W"]')
    page.wait_for("!!document.querySelector('.dialog')", 5, "the sweep dialog")
    page.fill(".dialog input.field", "0.7")
    page.fill(".dialog input.field", "0.8", 1)
    page.fill(".dialog input.field", "2", 2)
    page.press("Set", ".dialog")
    page.settle()
    page.check("W carries a sweep", params(page)["W"]["sweep"] == [0.7, 0.8], params(page)["W"]["sweep"])
    page.click_on(".tab", 3)
    page.pump(0.5)
    page.press("Run sweep")
    page.wait_for("window.atlas.store.sweep && ['done', 'stopped', 'failed'].includes(window.atlas.store.sweep.status)", 900, "the sweep")
    sw = page.js("window.atlas.store.sweep")
    page.check("the sweep ran the case once per value", sw["status"] == "done" and len(sw["results"]) == 2 and all(r["status"] == "done" for r in sw["results"]),
               [(r["value"], r["status"]) for r in sw["results"]])
    page.pump(0.5)
    page.shot("p2-sweep")

    # -- the dark theme, the Parameters panel --------------------------------------------------
    page.click_on(".tab", 0)
    page.js("document.documentElement.dataset.theme = 'dark'")
    tool(page, "dimension")
    page.pump(1.0)
    page.shot("p2-dark-parameters")
    fit = page.js("(() => { const o = []; for (const e of document.querySelectorAll('.stage .bar, #geo-inspector')) if (e.scrollWidth > e.clientWidth + 1) o.push(e.className || e.id); return o; })()")
    page.check("nothing overflows sideways at 1090 x 620", not fit, fit)
