"""P6's page check (integrated-ui-plan §8): the records, walked with real input.

Done-when, checked in the page: each existing evaluation's gates and prices on screen
match its JSON record (read here by explicit paths, not through records.py), and the
Stage 0b row says it is not registered.  Then Open: a held-out problem into Run's
compare mode, run, its numbers the record's.

    python scripts/studio_drive.py scripts/studio_walks/p6.py --out out/studio/p6
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CH = os.path.join(ROOT, "out", "chartop")


def _met(v):
    return v.get("met") if isinstance(v, dict) else None


WHERE = {
    "evaluation-twolevel-20261111-004223.json": lambda r, k: _met(r["gates"].get(k)),
    "evaluation-forms-20261019-205923.json": lambda r, k: _met(r["gates"].get(k)),
    "evaluation-20261018-203833.json": lambda r, k: _met(r["gates"].get(k)),
    "evaluation-prior-20261107-172219.json": lambda r, k: _met(r["gates"]["P"].get(k)),
    "evaluation-squares-20261108-141654.json": lambda r, k: _met(r.get(k)),
    "evaluation-hybrid-20261107-231105.json": lambda r, k: _met(r.get(k)),
}


def fmt3(x):
    s = f"{x:.3g}"
    return s


def walk(page):
    page.click_on(".tab", 4)
    page.wait_for("window.atlas.records && window.atlas.records.data()", 60, "the records")
    page.pump(0.6)
    rows = page.js("[...document.querySelectorAll('.recrow')].map(e => e.innerText.replace(/\\s+/g, ' ').trim())")
    page.check("Stage 0b is listed, not registered", rows and rows[0].startswith("Stage 0b") and "not registered" in rows[0], rows[:2])
    data = page.js("window.atlas.records.data()")
    files = [e["file"] for e in data if e.get("status") == "evaluated"]
    page.check("every evaluation on this machine is listed", set(files) == {f for f in WHERE if os.path.isfile(os.path.join(CH, f))}, files)
    for i, f in enumerate([e.get("file") for e in data]):
        if f not in WHERE:
            continue
        page.click_on(".recrow", i)
        page.pump(0.4)
        rec = json.load(open(os.path.join(CH, f), encoding="utf-8"))
        chips = page.js("[...document.querySelectorAll('.chipk[data-gate]')].map(e => [e.dataset.gate, e.classList.contains('ok') ? true : (e.classList.contains('bad') ? false : null)])")
        want = [[k, WHERE[f](rec, k)] for k, _ in chips]
        page.check(f"{f}: the gates on screen are the record's", chips == want and len(chips) > 0, chips)
        prices = page.js("[...document.querySelectorAll('.pgroup')].map(g => [g.querySelector('.label').textContent, [...g.querySelectorAll('.prow')].map(r => [r.children[0].textContent, r.children[2].textContent])])")
        if f.startswith("evaluation-twolevel"):
            gm = rec["gates"]["PR"]["geometric_mean"]
            label = {"all": "all priced", "held-out six": "held-out 6", "R2's sixteen": "squares (R2's 16)"}
            want = sorted([label[s], a, fmt3(v)] for a, sets in gm.items() for s, v in sets.items())
            got = sorted([g, a, v] for g, rs in prices for a, v in rs)
            page.check("the two-level prices on screen are the record's, each under its problem set", got == want, got[:4])
            page.check("rho's median and 10th percentile are the record's NC numbers",
                       f"median {fmt3(rec['gates']['NC']['median'])}" in page.js("document.querySelector('.recrow2 svg')?.textContent || ''")
                       and f"p10 {fmt3(rec['gates']['NC']['p10'])}" in page.js("document.querySelector('.recrow2 svg')?.textContent || ''"))
            page.shot("p6-twolevel")
        if f.startswith("evaluation-squares"):
            page.check("the squares price is the record's PR2", prices == [["squares (R2's 16)", [["PR2", fmt3(rec["PR2"]["geometric_mean"])]]]], prices)
    # a gate's hover text is its registered title and bar
    i = next(k for k, e in enumerate(data) if (e.get("file") or "").startswith("evaluation-twolevel"))
    page.click_on(".recrow", i)
    page.pump(0.4)
    x, y = page.at(".chipk[data-gate='E4']")
    page.mouse("mouseMoved", x, y, button="none", buttons=0)
    page.wait_for("document.getElementById('tip').classList.contains('on')", 5, "the hover box")
    tip = page.js("document.getElementById('tip').textContent")
    page.check("a gate's hover text is its registered title and bar", "coupled answer against the full-domain solve" in tip and "0.02" in tip, tip[:120])
    page.mouse("mouseMoved", 5, 600, button="none", buttons=0)

    # -- Open: bend-3 into Run's compare, TL-s0 against the full domain --------------------------
    page.press("bend-3", "main.panel")
    page.wait_for("window.atlas.store.tab === 'run' && !!document.querySelector('.cowins')", 60, "Run, compare")
    page.wait_for("window.atlas.store.pieces && window.atlas.store.pieces.summary && window.atlas.store.pieces.summary.pieces.length === 15 && "
                  "window.atlas.store.pieces.job.status === 'done'", 600, "the bend-3 cut")
    sel = page.js("document.querySelector('select[aria-label=\"Solver, B\"]').value")
    page.check("Open set window B to the record's network", sel == "TL-s0", sel)
    page.press("Run", "main.panel")
    page.pump(1.0)
    page.wait_for("window.atlas.store.corun && window.atlas.store.corun.status === 'done' && window.atlas.store.corun.result", 600, "the compare run")
    acc = page.js("window.atlas.store.corun.result.accuracy")
    seen = next(b for b in json.load(open(os.path.join(CH, "evaluation-twolevel-20261111-004223-accuracy.json"), encoding="utf-8"))["benchmarks_seen"]
                if b["name"] == "bend-3" and b["problem"] == "steady")["frames"]["registered"]["runs"]["TL-s0"]
    page.check("opened from Records, the run gives the record's numbers",
               abs(acc["energy_rel"] - seen["energy_rel"]) <= 1e-9 * seen["energy_rel"], {"page": acc["energy_rel"], "record": seen["energy_rel"]})
    page.shot("p6-opened")

    # -- the dark theme --------------------------------------------------------------------------
    page.click_on(".tab", 4)
    page.js("document.documentElement.dataset.theme = 'dark'")
    page.pump(1.0)
    page.shot("p6-records-dark")
    fit = page.js("(() => { const o = []; for (const e of document.querySelectorAll('.view:not([hidden]) .panel')) if (e.scrollWidth > e.clientWidth + 1) o.push(e.className); return o; })()")
    page.check("nothing overflows sideways at 1090 x 620", not fit, fit)
