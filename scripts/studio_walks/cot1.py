"""The coolant plan's Phase 1 in the page: CO-T on a conjugate-heat case, walked with
real input ([[chart-operator-stage-0b-and-1-plan]] section 5; `atlas/studio/cot.py`).

Done-when, checked in the page:
- the workbench's ``cooled-block`` (the coolant slowed to 0.1 mm/s, so every coolant
  piece is under the Peclet cap of 200) cuts into twelve pieces, four of them coolant,
  with three seams that carry the flow, each marked THERM+ADVEC;
- a coolant piece's inspector shows its Peclet and cell Peclet numbers and the CO-T
  envelope; the classical expert is the only one offered;
- Check the coupling: whole traces agree with the monolithic solve to round-off;
- Run's Compare of the classical CO-T pieces against the full domain reports the same
  energy error the host gives, the outlet's mixing-cup temperature, the ADVEC seams,
  class G0', and a price ratio with its A/A spread;
- at the example's own 1 cm/s the coolant pieces are refused by name, and the page says
  to cut the channel into shorter pieces.

    python scripts/studio_drive.py scripts/studio_walks/cot1.py --out out/studio/cot1
"""

import json

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


def speed(page, u):
    page.js("window.atlas.net.request('edit.apply', {ops: [{op: 'set', path: ['physics', 'params', 'u_coolant'], "
            f"value: {u}}}], label: 'coolant speed'}})")
    page.wait_for(f"window.atlas.store.state.case.physics.params.u_coolant === {u}", 60, "the speed")
    page.pump(1.0)


def card(page, label):
    # only the visible view's cards: the Physics view (hidden) has a Pe of its own
    return page.js(f"(() => {{ const r = [...document.querySelectorAll('.view:not([hidden]) .kvrow')].find(e => e.children[0].textContent.trim() === {json.dumps(label)}); return r ? r.children[1].textContent.trim() : null; }})()")


def cut(page, n, cols):
    page.js(f"(() => {{ const s = document.querySelector('select[aria-label=\"Resolution n\"]'); s.value = '{n}'; s.dispatchEvent(new Event('change')); }})()")
    page.fill('input[aria-label="grid columns"]', str(cols), enter=False)
    page.js("document.querySelector('input[aria-label=\"grid columns\"]').dispatchEvent(new Event('change'))")
    page.press("Cut automatically", "aside.panel")


def walk(page):
    page.js("window.atlas.net.request('case.example', {key: 'cooled-block'})")
    page.wait_for("window.atlas.store.state.meta.key === 'cooled-block'", 60, "cooled-block")
    page.pump(1.0)
    speed(page, 0.0001)
    page.click_on(".tab", 2)
    page.wait_for("window.atlas.store.pieces && window.atlas.store.pieces.experts", 60, "the experts")
    page.check("Pieces opens for a conjugate-heat case", page.js("!document.querySelector('.view:not([hidden]) aside.panel')?.hidden"))
    offered = [e["key"] for e in page.js("window.atlas.store.pieces.experts") if e["offered"]]
    page.check("the classical CO-T expert is the only one offered", offered == ["classical"], offered)
    cut(page, 64, 2)
    wait_pieces(page, "cooled-block cut")
    s = page.js("window.atlas.store.pieces.summary")
    page.check("CO-T, twelve pieces, three seams carry the flow",
               s["physics"] == "CO-T" and len(s["pieces"]) == 12 and s["advective_seams"] == 3,
               {"physics": s.get("physics"), "pieces": len(s["pieces"]), "advective": s.get("advective_seams")})
    water = [p["index"] for p in s["pieces"] if p["material"] == "water"]
    page.check("four coolant pieces", len(water) == 4, water)
    page.shot("cot1-pieces")

    # -- a coolant piece's inspector: select it on the canvas ---------------------------------
    p = s["pieces"][water[1]]
    x, y = page.js(f"window.atlas.pieces.world({p['centre'][0]}, {p['centre'][1]})")
    page.click(x, y)
    page.pump(0.6)
    pe = card(page, "Pe")
    want = p["flow"]["pe"]
    page.check("the inspector shows the piece's Peclet number", pe is not None and abs(float(pe) - want) <= 0.01 * want,
               {"page": pe, "summary": want})
    page.check("and its cell Peclet number, at most 2", card(page, "cell Pe") is not None and float(card(page, "cell Pe")) <= 2.0,
               card(page, "cell Pe"))
    page.check("inside the CO-T envelope", card(page, "in CO-T envelope") == "✓", card(page, "in CO-T envelope"))
    ports = page.js("[...document.querySelectorAll('.view:not([hidden]) aside.panel abbr')].map(e => e.textContent).filter(t => t.startsWith('THERM'))")
    page.check("its cuts across the channel are THERM+ADVEC, its wall THERM",
               ports.count("THERM+ADVEC") == 2 and ports.count("THERM") == 1, ports)
    page.shot("cot1-inspector")

    # -- the check ---------------------------------------------------------------------------------
    page.press("Check the coupling", "aside.panel")
    wait_pieces(page, "the coupling check", timeout=900)
    c = page.js("window.atlas.store.pieces.summary.check")
    page.check("whole traces agree with the monolithic solve to round-off", c["full"]["max_diff_rel"] <= 1e-9,
               c["full"]["max_diff_rel"])
    page.check("the check shows the outlet and the balance at 16 modes",
               card(page, "outlet, 16 modes") is not None and card(page, "balance, 16 modes") is not None)

    # -- Run: Compare the classical CO-T pieces against the full domain ----------------------------
    page.click_on(".tab", 3)
    page.pump(0.6)
    page.check("Run opens on the pieces", page.js("!!document.querySelector('.cowins')"))
    page.press("Compare", "main.panel")
    page.press("Full", ".cowin:nth-of-type(1)")
    page.press("Chart op.", ".cowin:nth-of-type(2)")
    opts = page.js("[...document.querySelector('select[aria-label=\"Solver, B\"]').options].map(o => o.value)")
    page.check("B offers the classical CO-T expert, no network", "classical" in opts and "TL-s0" not in opts, opts)
    page.js("(() => { const s = document.querySelector('select[aria-label=\"Solver, B\"]'); s.value = 'classical'; s.dispatchEvent(new Event('change')); })()")
    page.press("Run", "main.panel")
    wait_run(page, "classical CO-T pieces against the full domain")
    r = page.js("window.atlas.store.corun.result")
    page.check("the result is CO-T's, class G0'", r["physics"] == "CO-T" and r["certificate"]["class"] == "G0′",
               r["certificate"]["class"])
    page.check("the energy error is the check's 16-mode floor",
               abs(r["accuracy"]["energy_rel"] - c["ports16"]["energy_rel"]) <= 1e-9 * c["ports16"]["energy_rel"],
               {"run": r["accuracy"]["energy_rel"], "check": c["ports16"]["energy_rel"]})
    page.check("the cards show the outlet and the ADVEC seams",
               card(page, "outlet T") is not None and card(page, "ADVEC seams") == "3",
               {"outlet": card(page, "outlet T"), "advec": card(page, "ADVEC seams")})
    price = r["price"]
    page.check("the price is a ratio with its A/A spread, no seconds",
               price["ratio"] > 0 and price["aa_spread"] is not None and "seconds" not in json.dumps(r), price)
    page.check("the positive-real certificate", r["certificate"]["min_eig"] > 0, r["certificate"]["min_eig"])
    page.shot("cot1-compare")

    # -- the example's own speed: refused, with what to do -----------------------------------------
    speed(page, 0.01)
    page.click_on(".tab", 2)
    page.pump(0.6)
    cut(page, 128, 2)
    wait_pieces(page, "cooled-block at 1 cm/s cut")
    s = page.js("window.atlas.store.pieces.summary")
    refused = [r for p in s["pieces"] for r in p["envelope"]["refusals"]]
    page.check("the fast coolant pieces are refused: cut the channel shorter",
               refused and all("cut the channel into shorter pieces" in r for r in refused if "cap" in r), refused[:2])
    p = next(p for p in s["pieces"] if p["material"] == "water")
    x, y = page.js(f"window.atlas.pieces.world({p['centre'][0]}, {p['centre'][1]})")
    page.click(x, y)
    page.pump(0.6)
    page.check("the inspector marks it outside the envelope", card(page, "in CO-T envelope") == "✗", card(page, "in CO-T envelope"))
    page.shot("cot1-refused")

    page.js("document.documentElement.dataset.theme = 'dark'")
    page.pump(1.0)
    page.shot("cot1-refused-dark")
    fit = page.js("(() => { const o = []; for (const e of document.querySelectorAll('.view:not([hidden]) .panel')) if (e.scrollWidth > e.clientWidth + 1) o.push(e.className); return o; })()")
    page.check("nothing overflows sideways at 1090 x 620", not fit, fit)
