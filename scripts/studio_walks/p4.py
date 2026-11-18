"""P4's page check (integrated-ui-plan §8): pieces and experts, walked with real input.

Done-when, checked in the page:
* the three held-out benchmarks, cut in the page, reproduce `benchmarks.py`'s pieces
  (each piece's chart centre against the module's own, and the chart methods);
* a coupled classical solve matches `host.monolithic` to round-off (the page's check);
* a piece made to fold is refused with its det J shown (a corner dragged in the page).

    python scripts/studio_drive.py scripts/studio_walks/p4.py --out out/studio/p4
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def summary(page):
    return page.js("window.atlas.store.pieces && window.atlas.store.pieces.summary")


LAST = {"id": 0}


def wait_job(page, what, timeout=600):
    """Until a job newer than the last one waited for has ended."""
    page.wait_for(f"window.atlas.store.pieces && window.atlas.store.pieces.job && "
                  f"window.atlas.store.pieces.job.id > {LAST['id']} && "
                  f"['done', 'failed'].includes(window.atlas.store.pieces.job.status)", timeout, what)
    LAST["id"] = page.js("window.atlas.store.pieces.job.id")
    job = page.js("window.atlas.store.pieces.job")
    page.check(f"{what}: the job ended", job["status"] == "done", job.get("error"))
    page.pump(0.6)


def example(page, key):
    page.js(f"window.atlas.net.request('case.example', {{key: '{key}'}})")
    page.wait_for(f"window.atlas.store.state.meta.key === '{key}' || (window.atlas.store.state.meta.file || '').startsWith('{key}')", 60, key)
    page.pump(1.0)


def resolution(page, n):
    page.js(f"(() => {{ const s = document.querySelector('select[aria-label=\"Resolution n\"]'); s.value = '{n}'; s.dispatchEvent(new Event('change')); }})()")


def centres_match(page, name, n, **kw):
    from atlas.chartop import benchmarks as bm
    b = bm.build(name, n, "steady", **kw)
    s = summary(page)
    mine = [(p["centre"], p["chart"]) for p in s["pieces"]]
    theirs = [(b_.chart.X[n // 2, n // 2].tolist(), b_.chart.method) for b_ in b.problems]
    worst = max(max(abs(a[0][0] - c[0][0]), abs(a[0][1] - c[0][1])) for a, c in zip(mine, theirs)) if len(mine) == len(theirs) else None
    same = len(mine) == len(theirs) and worst is not None and worst < 2e-6 and all(a[1] == c[1] for a, c in zip(mine, theirs))
    page.check(f"{name}, cut in the page, is benchmarks.py's layout", same,
               {"pieces": [len(mine), len(theirs)], "worst centre difference": worst})


def walk(page):
    # -- bend-3: polar strips -------------------------------------------------------------
    example(page, "bend-3")
    page.click_on(".tab", 2)
    page.wait_for("window.atlas.store.pieces && window.atlas.store.pieces.experts && window.atlas.store.pieces.experts.length > 0", 60, "the experts")
    page.pump(0.5)
    experts = page.js("window.atlas.store.pieces.experts")
    offered = [e["key"] for e in experts if e["offered"]]
    page.check("experts are listed; learned ones only with matching weights", "classical" in offered, offered)
    page.press("Cut automatically", "aside.panel")
    wait_job(page, "bend-3 cut")
    s = summary(page)
    seams = [x for x in s["seams"] if len(x["pieces"]) == 2]
    page.check("bend-3: 15 pieces, every chart certified, every seam admitted",
               len(s["pieces"]) == 15 and s["certified"] == [15, 15] and {x["verdict"] for x in seams} == {"admit"}, (len(s["pieces"]), s["certified"], len(seams)))
    centres_match(page, "bend-3", 64)
    page.shot("p4-bend3")

    # a piece chosen with a click, its expert set with a click
    c = s["pieces"][7]["centre"]
    x, y = page.js(f"window.atlas.pieces.world({c[0]}, {c[1]})")
    page.click(x, y)
    page.pump(0.4)
    page.check("clicking a piece selects it", page.js("document.querySelector('aside.panel abbr')?.textContent") is not None
               and "P8" in page.js("document.body.innerText"))
    page.press("TFI", "aside.panel")
    wait_job(page, "charts by TFI")
    s = summary(page)
    page.check("TFI on every piece: all still certified", all(p["chart"] == "transfinite" for p in s["pieces"]) and s["certified"] == [15, 15], s["certified"])
    page.press("The cut's own charts", "aside.panel")
    wait_job(page, "the cut's own charts")
    page.press("Best admitted for all", "aside.panel")
    wait_job(page, "best admitted")
    s = summary(page)
    picked = sorted({p["expert"] for p in s["pieces"]})
    page.check("best admitted: the trained network where its card admits it", picked == (["TL-s0"] if "TL-s0" in offered else ["classical"]), picked)
    page.press("Classical", "aside.panel")
    wait_job(page, "one piece classical")
    s = summary(page)
    page.check("one piece's expert set from its panel", s["pieces"][7]["expert"] == "classical", s["pieces"][7]["expert"])
    page.press("Check the coupling", "aside.panel")
    wait_job(page, "the coupling check", 300)
    s = summary(page)
    full = s["check"]["full"]["max_diff_rel"]
    page.check("a coupled classical solve matches host.monolithic to round-off", full < 1e-10, full)
    page.shot("p4-bend3-experts")

    # -- insert-round: the O-H layout, at n = 32 ----------------------------------------------
    example(page, "insert-round")
    page.click_on(".tab", 2)
    page.pump(0.5)
    resolution(page, 32)
    page.press("Cut automatically", "aside.panel")
    wait_job(page, "insert-round cut")
    s = summary(page)
    page.check("insert-round: 39 pieces, all certified", len(s["pieces"]) == 39 and s["certified"] == [39, 39], s["certified"])
    centres_match(page, "insert-round", 32)

    # -- s-channel: the strip, at n = 32 ---------------------------------------------------------
    example(page, "s-channel")
    page.click_on(".tab", 2)
    page.pump(0.5)
    resolution(page, 32)
    page.press("Cut automatically", "aside.panel")
    wait_job(page, "s-channel cut", 900)
    s = summary(page)
    page.check("s-channel: 9 pieces, its end pieces by Winslow", len(s["pieces"]) == 9 and any("Winslow" in n for n in s["notes"]), s["notes"])
    from atlas.chartop.data import ENVELOPE
    centres_match(page, "s-channel", 32, envelope=ENVELOPE)
    page.shot("p4-s-channel")

    # -- a piece made to fold: a corner dragged past its neighbour ------------------------------
    example(page, "wall-2")
    page.click_on(".tab", 2)
    page.pump(0.5)
    resolution(page, 32)
    page.press("Cut automatically", "aside.panel")
    wait_job(page, "wall-2 cut")
    s = summary(page)
    page.check("wall-2: cut at the material interface", [p["material"] for p in s["pieces"]] == ["steel", "steel", "copper"], [p["material"] for p in s["pieces"]])
    page.press("Cut", ".stage .bar")
    x, y = page.js("window.atlas.pieces.world(0.0, 0.05)")
    page.click(x + 3, y)                        # the left side of the first piece, half way up
    wait_job(page, "a cut by hand")
    s = summary(page)
    page.check("the hand cut runs through all three pieces", len(s["pieces"]) == 6 and s["chords"] == ["c1"], (len(s["pieces"]), s["chords"]))
    page.press("Corner", ".stage .bar")
    a = page.js("window.atlas.pieces.world(0.2, 0.05)")
    b = page.js("window.atlas.pieces.world(0.27, 0.05)")
    page.drag([a, b], steps=12)
    wait_job(page, "a corner dragged")
    s = summary(page)
    bad = [(p["label"], p["certificate"]["min_detJ_rel"]) for p in s["pieces"] if not p["certificate"]["ok"]]
    page.check("a piece made to fold is refused with its det J", len(bad) == 2 and all(d < 0 for _, d in bad), bad)
    i = next(k for k, p in enumerate(s["pieces"]) if not p["certificate"]["ok"])
    c = s["pieces"][i]["centre"]
    page.press("Corner", ".stage .bar")         # back to selecting
    x, y = page.js(f"window.atlas.pieces.world({c[0]}, {c[1]})")
    page.click(x, y)
    page.pump(0.5)
    text = page.js("(() => { const a = document.querySelectorAll('.view:not([hidden]) aside.panel'); return a[a.length - 1].innerText; })()")
    page.check("its panel says it is refused, with det J", "Refused" in text and "det J" in text, text[:200])
    page.shot("p4-folded")

    # -- the dark theme ----------------------------------------------------------------------------
    page.js("document.documentElement.dataset.theme = 'dark'")
    page.pump(1.0)
    page.shot("p4-folded-dark")
    fit = page.js("(() => { const o = []; for (const e of document.querySelectorAll('.view:not([hidden]) .panel, .view:not([hidden]) .bar')) if (e.scrollWidth > e.clientWidth + 1) o.push(e.className); return o; })()")
    page.check("nothing overflows sideways at 1090 x 620", not fit, fit)
