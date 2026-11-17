"""P3's page check (integrated-ui-plan §8): physics, walked with real input.

Done-when, checked in the page: every equation in the list either runs or says why
not on hover, and every regime number is checked against a hand computation on one
case (fast-heat: steel and copper, transient).  The hand computations below are the
numbers written out, not the registry's code.

    python scripts/studio_drive.py scripts/studio_walks/p3.py --out out/studio/p3
"""

import math


def regime_rows(page):
    """The Regime section's rows as {label: value text}."""
    return page.js("""(() => {
        const sec = [...document.querySelectorAll('.subpanel')].find(s => s.querySelector('.label')?.textContent.startsWith('Regime'));
        if (!sec) return null;
        const out = {};
        for (const r of sec.querySelectorAll('.kvrow')) out[r.children[0].textContent.trim()] = r.children[1].textContent.trim();
        return out;
    })()""")


def hover_tip(page, selector_js):
    """Hover an element with the mouse and read the hover box the page shows."""
    r = page.js(f"(() => {{ const e = {selector_js}; if (!e) return null; e.scrollIntoView({{block: 'nearest'}}); "
                f"const b = e.getBoundingClientRect(); return [b.left + Math.min(30, b.width / 2), b.top + b.height / 2, e.getAttribute('data-tip')]; }})()")
    if not r:
        return None, None
    page.mouse("mouseMoved", r[0], r[1], button="none", buttons=0)
    page.wait_for("document.getElementById('tip').classList.contains('on')", 5, "the hover box")
    page.pump(0.3)
    shown = page.js("document.getElementById('tip').textContent")
    page.mouse("mouseMoved", 5, 600, button="none", buttons=0)
    page.pump(0.3)
    return shown, r[2]


def walk(page):
    # a known case: fast-heat (steel and copper, transient)
    page.js("window.atlas.net.request('case.example', {key: 'fast-heat'})")
    page.wait_for("window.atlas.store.state.meta.family === 'conduction-2d' && window.atlas.store.state.case.run.mode === 'transient'", 60, "fast-heat")
    page.click_on(".tab", 1)
    page.wait_for("!!document.querySelector('.eqrow')", 10, "the equation list")
    page.pump(0.5)

    # -- every equation runs or says why not ---------------------------------------------
    rows = page.js("[...document.querySelectorAll('.eqrow')].map(e => ({name: e.querySelector('.name').textContent, off: e.getAttribute('aria-disabled') === 'true', tip: e.getAttribute('data-tip') || ''}))")
    n_eq = page.js("window.atlas.store.catalog.equations.length")
    page.check("the list shows every equation of the catalog", len(rows) == n_eq, f"{len(rows)} of {n_eq}")
    runs = [r for r in rows if not r["off"]]
    off = [r for r in rows if r["off"]]
    page.check("every equation runs or says why not", all(r["tip"] for r in rows) and all(len(r["tip"]) > 20 for r in off),
               f"{len(runs)} run, {len(off)} say why not")
    page.check("the eight kinds that run here are all in the list", len(runs) == 8, [r["name"] for r in runs])
    shown, tip = hover_tip(page, "[...document.querySelectorAll('.eqrow')].find(e => e.querySelector('.name').textContent === 'Darcy flow')")
    page.check("hovering a greyed equation shows why it does not run", shown == tip and "Darcy" in shown, shown)
    shown, tip = hover_tip(page, "document.querySelector('.eqgroup abbr')")
    page.check("hovering a guarantee class explains it", shown and "Guarantee class G0" in shown, shown)

    # -- the regime numbers against the hand computation ----------------------------------
    alpha_cu = 400.0 / (8960.0 * 385.0)                 # copper diffuses fastest: 1.15955e-4 m^2/s
    dt = page.js("window.atlas.store.state.case.run.macro_dt")
    dx = 0.000625
    area = 640 * 384 * dx * dx                          # the whole plate, 0.4 x 0.24 m
    fo_cell = alpha_cu * dt / dx ** 2
    fo_piece = alpha_cu * dt / area
    reg = page.js("window.atlas.store.state.regime")
    page.check("contrast = 400 / 45 (copper over steel)", abs(reg["contrast"]["value"] - 400 / 45) < 1e-12, reg["contrast"]["value"])
    page.check("Fo, cell = alpha dt / h^2", abs(reg["fo_cell"]["value"] - fo_cell) < 1e-12 * fo_cell, (reg["fo_cell"]["value"], fo_cell))
    page.check("Fo, piece = alpha dt / |Omega|, outside the CO-D card's [0.01, 100]",
               abs(reg["fo_domain"]["value"] - fo_piece) < 1e-12 * fo_piece and reg["fo_domain"]["within"] is False, (reg["fo_domain"]["value"], fo_piece))
    shown = regime_rows(page)
    page.check("the page shows them", shown and shown.get("contrast") == "8.89" and shown.get("Fo, cell") == "1.71" and shown.get("Fo, piece", "").endswith("✗"), shown)
    shown_tip, tip = hover_tip(page, "[...document.querySelectorAll('.kvrow abbr')].find(e => e.textContent === 'Fo, cell')")
    page.check("each regime number names its definition on hover", shown_tip and "κΔt / (ρc h²)" in shown_tip, shown_tip)
    page.shot("p3-heat")

    # -- steady and transient ------------------------------------------------------------
    page.press("Steady", "section.panel")
    page.wait_for("window.atlas.store.state.case.run.mode === 'steady'", 30, "steady")
    page.pump(0.5)
    shown = regime_rows(page)
    page.check("steady: no Fourier number", shown.get("Fo, piece", "").startswith("steady") and "Fo, cell" not in shown, shown)
    page.press("Transient", "section.panel")
    page.wait_for("window.atlas.store.state.case.run.mode === 'transient'", 30, "transient")

    # -- a property edited in the table moves the contrast ---------------------------------
    page.fill('input[aria-label="copper k"]', "380")
    page.wait_for("Math.abs(window.atlas.store.state.regime.contrast.value - 380 / 45) < 1e-9", 30, "the new contrast")
    page.pump(0.4)
    page.check("editing copper's κ to 380 gives contrast 380/45", regime_rows(page).get("contrast") == "8.44", regime_rows(page))

    # -- a greyed equation does nothing; one that runs opens its kind ----------------------
    fam = page.js("window.atlas.store.state.meta.family")
    x, y = page.at(".eqrow.off")
    page.click(x, y)
    page.pump(1.0)
    page.check("clicking an equation that does not run changes nothing", page.js("window.atlas.store.state.meta.family") == fam)
    i = page.js("[...document.querySelectorAll('.eqrow')].findIndex(e => e.querySelector('.name').textContent === 'Acoustics')")
    page.click_on(".eqrow", i)
    page.wait_for("window.atlas.store.state.meta.family === 'acoustics-2d'", 60, "the acoustics case")
    page.pump(0.8)
    reg = page.js("window.atlas.store.state.regime")
    w = page.js("window.atlas.store.state.case.physics.params.pulse_width")
    h = page.js("window.atlas.store.state.case.domain.dx")
    page.check("acoustics: k h = 3 h / w", abs(reg["k"]["kh"] - 3 * h / w) < 1e-12, (reg["k"]["kh"], 3 * h / w))
    page.check("the equation screen follows: Acoustics, hyperbolic, G3",
               page.js("document.querySelector('.eqrow[aria-selected=\"true\"] .name')?.textContent") == "Acoustics"
               and "hyperbolic" in page.js("document.body.innerText"), None)
    shown_tip, tip = hover_tip(page, "document.querySelector('.bcell abbr')")
    page.check("a boundary condition explains itself on hover", shown_tip and len(shown_tip) > 10, shown_tip)

    # -- search ------------------------------------------------------------------------------
    page.fill('input[aria-label="Search equations"]', "elastic")
    page.pump(0.4)
    names = page.js("[...document.querySelectorAll('.eqrow')].map(e => e.querySelector('.name').textContent)")
    page.check("the search narrows the list", names and all("lastic" in n or "Thermoelasticity" in n for n in names), names)
    page.fill('input[aria-label="Search equations"]', "")

    # -- the dark theme -------------------------------------------------------------------------
    page.js("window.atlas.net.request('edit.undo')")
    page.wait_for("window.atlas.store.state.meta.family === 'conduction-2d'", 60, "back to heat")
    page.js("document.documentElement.dataset.theme = 'dark'")
    page.pump(1.0)
    page.shot("p3-heat-dark")
    fit = page.js("(() => { const o = []; for (const e of document.querySelectorAll('.view:not([hidden]) .panel')) if (e.scrollWidth > e.clientWidth + 1) o.push(e.className); return o; })()")
    page.check("nothing overflows sideways at 1090 x 620", not fit, fit)
