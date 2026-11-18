"""P5's page check (integrated-ui-plan §8): runs on the pieces, walked with real input.

Done-when, checked in the page: a compare run of TL-s0 against the classical full domain
on bend-3 shows the same accuracy numbers the two-level record holds for that problem
(`evaluation-twolevel-20261111-004223-accuracy.json`, ``benchmarks_seen``, bend-3 steady,
the registered frame, run with 8 torch threads on this machine).  Then the views, Single,
and a short transient march with the step slider.

    python scripts/studio_drive.py scripts/studio_walks/p5.py --out out/studio/p5
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORD = os.path.join(ROOT, "out", "chartop", "evaluation-twolevel-20261111-004223-accuracy.json")

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


def example(page, key):
    page.js(f"window.atlas.net.request('case.example', {{key: '{key}'}})")
    page.wait_for(f"window.atlas.store.state.meta.key === '{key}'", 60, key)
    page.pump(1.0)


def card(page, label):
    return page.js(f"(() => {{ const r = [...document.querySelectorAll('.kvrow')].find(e => e.children[0].textContent.trim() === {json.dumps(label)}); return r ? r.children[1].textContent.trim() : null; }})()")


def walk(page):
    want = next(b for b in json.load(open(RECORD, encoding="utf-8"))["benchmarks_seen"]
                if b["name"] == "bend-3" and b["problem"] == "steady")["frames"]["registered"]["runs"]["TL-s0"]
    example(page, "bend-3")
    page.click_on(".tab", 2)
    page.wait_for("window.atlas.store.pieces && window.atlas.store.pieces.experts", 60, "the experts")
    offered = [e["key"] for e in page.js("window.atlas.store.pieces.experts") if e["offered"]]
    page.check("TL-s0 is offered here (its weights match its record)", "TL-s0" in offered, offered)
    page.press("Cut automatically", "aside.panel")
    wait_pieces(page, "bend-3 cut")
    page.js("window.atlas.net.request('state')")
    page.click_on(".tab", 3)
    page.pump(0.6)
    page.check("Run opens on the pieces", page.js("!!document.querySelector('.cowins')"))

    # -- Compare: A full domain, B TL-s0 on every piece ----------------------------------
    page.press("Compare", "main.panel")
    page.press("Full", ".cowin:nth-of-type(1)")
    page.press("Chart op.", ".cowin:nth-of-type(2)")
    page.js("(() => { const s = document.querySelector('select[aria-label=\"Solver, B\"]'); s.value = 'TL-s0'; s.dispatchEvent(new Event('change')); })()")
    page.press("Run", "main.panel")
    wait_run(page, "TL-s0 against the full domain on bend-3")
    acc = page.js("window.atlas.store.corun.result.accuracy")
    page.check("the energy error is the record's", abs(acc["energy_rel"] - want["energy_rel"]) <= 1e-9 * want["energy_rel"],
               {"page": acc["energy_rel"], "record": want["energy_rel"]})
    page.check("the max error is the record's", abs(acc["max_err"] - want["max_err"]) <= 1e-9 * want["max_err"],
               {"page": acc["max_err"], "record": want["max_err"]})
    page.check("the interface matrix's smallest eigenvalue is the record's", abs(acc["min_eig"] - want["min_eig"]) <= 1e-9 * want["min_eig"],
               {"page": acc["min_eig"], "record": want["min_eig"]})
    page.check("the cards show them", card(page, "energy, all") == "2.0e-3" and card(page, "max error") == "7.7e-3",
               {"energy": card(page, "energy, all"), "max": card(page, "max error")})
    page.check("the run records its torch threads and machine (the record ran with 8)",
               page.js("!!window.atlas.store.corun.result.machine"), page.js("window.atlas.store.corun.result.threads"))
    price = page.js("window.atlas.store.corun.result.price")
    page.check("the price is a ratio with its A/A spread, no seconds", price["ratio"] > 0 and price["aa_spread"] is not None
               and "seconds" not in json.dumps(page.js("window.atlas.store.corun.result")), price)
    chips = page.js("[...document.querySelectorAll('.chipk')].map(e => e.textContent)")
    page.check("the agree chip applies E4's bar and passes", any(c.startswith("agree ✓") for c in chips), chips)
    page.shot("p5-compare-field")

    # -- the views ---------------------------------------------------------------------------
    for view, check in (("Difference", "document.querySelectorAll('.cofield canvas').length === 2"),
                        ("Per piece", "document.querySelectorAll('.ppr').length === 15"),
                        ("Seams", "document.querySelector('.cofield h2')?.textContent.startsWith('largest jump')")):
        page.press(view, ".cotop")
        page.pump(0.5)
        page.check(f"the {view} view draws", page.js(check))
        if view == "Per piece":
            page.shot("p5-per-piece")
    page.press("Field", ".cotop")

    # -- Single: the mixed experts against the full domain -------------------------------------
    page.press("Single", "main.panel")
    page.press("Chart op.", ".cowin")
    page.js("(() => { const s = document.querySelector('select[aria-label=\"Solver, A\"]'); s.value = 'mixed'; s.dispatchEvent(new Event('change')); })()")
    page.press("Run", "main.panel")
    wait_run(page, "a single run of the mixed experts")
    r = page.js("window.atlas.store.corun.result")
    page.check("single: the window against the full domain, no price", r["accuracy"]["energy_rel"] < 1e-9 and "price" not in r, r["accuracy"]["energy_rel"])

    # -- a transient march and the step slider -------------------------------------------------
    example(page, "wall-2")
    page.js("window.atlas.net.request('edit.apply', {ops: [{op: 'set', path: ['run', 'mode'], value: 'transient'}, "
            "{op: 'set', path: ['run', 'steps'], value: 4}, {op: 'set', path: ['run', 'macro_dt'], value: 100}], label: 'a short march'})")
    page.pump(1.5)
    page.click_on(".tab", 2)
    page.pump(0.5)
    page.js("(() => { const s = document.querySelector('select[aria-label=\"Resolution n\"]'); s.value = '32'; s.dispatchEvent(new Event('change')); })()")
    page.press("Cut automatically", "aside.panel")
    wait_pieces(page, "wall-2 cut")
    page.click_on(".tab", 3)
    page.pump(0.5)
    page.press("Compare", "main.panel")
    page.press("Full", ".cowin:nth-of-type(1)")
    page.press("Chart op.", ".cowin:nth-of-type(2)")
    page.js("(() => { const s = document.querySelector('select[aria-label=\"Solver, B\"]'); s.value = 'classical'; s.dispatchEvent(new Event('change')); })()")
    page.press("Run", "main.panel")
    wait_run(page, "a four-step march")
    page.check("four steps marched", page.js("window.atlas.store.corun.frames") == 4)
    slider = page.at('input[aria-label="Step, both windows"]')
    left = page.js("(() => { const b = document.querySelector('input[aria-label=\"Step, both windows\"]').getBoundingClientRect(); return [b.left + 4, b.top + b.height / 2]; })()")
    page.drag([slider, left], steps=6)
    page.pump(1.0)
    page.check("the slider shows step 1 of 4", page.js("document.querySelector('.cofoot .mono').textContent") == "1 / 4",
               page.js("document.querySelector('.cofoot .mono').textContent"))
    page.shot("p5-transient")

    # -- the dark theme ----------------------------------------------------------------------------
    page.js("document.documentElement.dataset.theme = 'dark'")
    page.pump(1.0)
    page.shot("p5-transient-dark")
    fit = page.js("(() => { const o = []; for (const e of document.querySelectorAll('.view:not([hidden]) .panel')) if (e.scrollWidth > e.clientWidth + 1) o.push(e.className); return o; })()")
    page.check("nothing overflows sideways at 1090 x 620", not fit, fit)
