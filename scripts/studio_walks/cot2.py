"""The coolant plan's Phase 2 in the page: a development layout drawn with the Channel
tool, cut along its channel's walls, walked with real input
([[chart-operator-stage-0b-and-1-plan]]; `atlas/studio/channel_cut.py`, `cot_layouts.py`).

Done-when, checked in the page:
- ``dev-winding-slow`` (an aluminium plate, one winding channel, a heated device)
  opens from its case file and cuts automatically along the channel's walls: 32
  certified pieces, eight of them coolant, seven seams carrying the flow;
- a coolant piece on the bend shows its Peclet numbers inside the CO-T envelope;
- the coolant's inlet is Danckwerts': no side of the left edge is held;
- Check the coupling: whole traces agree with the monolithic solve to round-off;
- Run's Compare of the classical pieces against the full domain reports class G0' and
  the outlet's temperature.

    python scripts/studio_drive.py scripts/studio_walks/cot2.py --out out/studio/cot2
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CASE = os.path.join(ROOT, "out", "chartop", "cot", "layouts", "dev-winding-slow.json")
LAST = {"id": 0}


def wait_pieces(page, what, timeout=600):
    page.wait_for(f"window.atlas.store.pieces && window.atlas.store.pieces.job && window.atlas.store.pieces.job.id > {LAST['id']} && "
                  f"['done', 'failed'].includes(window.atlas.store.pieces.job.status)", timeout, what)
    LAST["id"] = page.js("window.atlas.store.pieces.job.id")
    job = page.js("window.atlas.store.pieces.job")
    page.check(f"{what}: done", job["status"] == "done", job.get("error"))
    page.pump(0.5)


def wait_run(page, what, timeout=900):
    page.pump(0.6)
    page.wait_for("window.atlas.store.corun && ['done', 'failed', 'stopped'].includes(window.atlas.store.corun.status) "
                  "&& window.atlas.store.corun.result !== undefined", timeout, what)
    cr = page.js("({status: window.atlas.store.corun.status, error: window.atlas.store.corun.error})")
    page.check(f"{what}: done", cr["status"] == "done", cr.get("error"))
    page.pump(0.8)


def card(page, label):
    return page.js(f"(() => {{ const r = [...document.querySelectorAll('.view:not([hidden]) .kvrow')].find(e => e.children[0].textContent.trim() === {json.dumps(label)}); return r ? r.children[1].textContent.trim() : null; }})()")


def walk(page):
    text = open(CASE, encoding="utf-8").read()
    page.js(f"window.atlas.net.request('case.import', {{text: {json.dumps(text)}, name: 'dev-winding-slow.json'}})")
    page.wait_for("window.atlas.store.state && window.atlas.store.state.case && "
                  "window.atlas.store.state.case.physics.family === 'conjugate-heat-2d' && "
                  "window.atlas.store.state.case.physics.params.u_coolant < 0.001", 120, "the layout")
    page.pump(1.0)
    feats = page.js("(window.atlas.store.state.geometry || {}).features || []")
    page.check("the drawing holds a Channel-tool channel", any(f.get("kind") == "channel" for f in feats),
               [f.get("kind") for f in feats])
    page.click_on(".tab", 2)
    page.wait_for("window.atlas.store.pieces && window.atlas.store.pieces.experts", 60, "the experts")
    page.js("(() => { const s = document.querySelector('select[aria-label=\"Resolution n\"]'); s.value = '32'; s.dispatchEvent(new Event('change')); })()")
    page.fill('input[aria-label="grid columns"]', "8", enter=False)
    page.js("document.querySelector('input[aria-label=\"grid columns\"]').dispatchEvent(new Event('change'))")
    page.press("Cut automatically", "aside.panel")
    wait_pieces(page, "the winding plate cut")
    s = page.js("window.atlas.store.pieces.summary")
    page.check("cut along the channel's walls: 32 certified pieces",
               s["cutter"] == "channels" and len(s["pieces"]) == 32 and s["certified"] == [32, 32],
               {"cutter": s.get("cutter"), "pieces": len(s["pieces"]), "certified": s.get("certified")})
    water = [p for p in s["pieces"] if p["material"] == "water"]
    page.check("eight coolant pieces, seven seams carrying the flow",
               len(water) == 8 and s["advective_seams"] == 7, {"water": len(water), "advective": s.get("advective_seams")})
    held = [sd for p in s["pieces"] for sd in p["sides"] if sd["to"] is None and sd.get("kind") == "dirichlet"]
    page.check("the inlet is Danckwerts': no side is held", not held, len(held))
    page.shot("cot2-pieces")
    p = water[3]
    x, y = page.js(f"window.atlas.pieces.world({p['centre'][0]}, {p['centre'][1]})")
    page.click(x, y)
    page.pump(0.6)
    pe = card(page, "Pe")
    page.check("a coolant piece on the bend shows its Peclet number", pe is not None and abs(float(pe) - p["flow"]["pe"]) <= 0.01 * p["flow"]["pe"],
               {"page": pe, "summary": p["flow"]["pe"]})
    page.check("inside the CO-T envelope", card(page, "in CO-T envelope") == "✓", card(page, "in CO-T envelope"))
    page.shot("cot2-inspector")
    page.press("Check the coupling", "aside.panel")
    wait_pieces(page, "the coupling check", timeout=900)
    c = page.js("window.atlas.store.pieces.summary.check")
    page.check("whole traces agree with the monolithic solve to round-off", c["full"]["max_diff_rel"] <= 1e-9,
               c["full"]["max_diff_rel"])
    page.click_on(".tab", 3)
    page.pump(0.6)
    page.press("Compare", "main.panel")
    page.press("Full", ".cowin:nth-of-type(1)")
    page.press("Chart op.", ".cowin:nth-of-type(2)")
    page.js("(() => { const s = document.querySelector('select[aria-label=\"Solver, B\"]'); s.value = 'classical'; s.dispatchEvent(new Event('change')); })()")
    page.press("Run", "main.panel")
    wait_run(page, "classical CO-T pieces on the winding plate")
    r = page.js("window.atlas.store.corun.result")
    page.check("class G0', the outlet's temperature shown", r["certificate"]["class"] == "G0′" and card(page, "outlet T") is not None,
               {"class": r["certificate"]["class"], "outlet": card(page, "outlet T")})
    page.check("the energy error is the check's 16-mode floor",
               abs(r["accuracy"]["energy_rel"] - c["ports16"]["energy_rel"]) <= 1e-9 * max(c["ports16"]["energy_rel"], 1e-30),
               {"run": r["accuracy"]["energy_rel"], "check": c["ports16"]["energy_rel"]})
    page.shot("cot2-compare")
    fit = page.js("(() => { const o = []; for (const e of document.querySelectorAll('.view:not([hidden]) .panel')) if (e.scrollWidth > e.clientWidth + 1) o.push(e.className); return o; })()")
    page.check("nothing overflows sideways at 1090 x 620", not fit, fit)
