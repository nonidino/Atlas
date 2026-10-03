"""Write every number, chart and generated block into the website's pages.

The pages in ``site/`` are hand-written HTML and are served as they are (GitHub
Pages needs no build step). This script edits them in place, idempotently:

- an element carrying ``data-claim="<id>"`` gets that claim's display text
  (``site/data/claims.json``); ``data-lit="<entry>.<key>"`` a literature value;
  ``data-lit-ref="<entry>"`` its citation; ``data-field="claim:<id>:<field>"`` or
  ``"lit:<id>:<field>"`` that field's text (a machine, a caveat, a quotation);
- the region between ``<!-- gen:NAME -->`` and ``<!-- /gen:NAME -->`` is regenerated:
  the charts (drawn here as SVG from the claims), the literature cards, the proofs'
  dependency graph (read from the built blueprint), the evidence tables, and the
  install commands (from ``site/data/site.json``);
- ``<!-- img:NAME -->`` regions are replaced by the owner's illustration when
  ``site/images/<NAME>.webp`` exists, with its alt text from the sidecar;
- ``data-config="<key>"`` links get their address from ``site/data/site.json``.

It also copies the architecture document (``proposal/architecture/index.html``) to
``site/architecture/`` unchanged, and the Lean sources to ``site/proofs/lean/`` as
readable pages. Run ``scripts/site_claims.py`` first, and ``scripts/site_blueprint.py``
when the blueprint changes.

Run:  python scripts/site_build.py
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
PAGES = ("index.html", "install.html", "evidence.html")
TIMES = "×"


def jload(rel):
    with open(os.path.join(SITE, rel), encoding="utf-8") as fh:
        return json.load(fh)


CLAIMS = {c["id"]: c for c in jload("data/claims.json")["claims"]}
RECORDS = jload("data/claims.json")["records"]
LIT = {e["id"]: e for e in jload("data/literature.json")["entries"]}
CFG = jload("data/site.json")
esc = html.escape


def claim(cid: str, cls: str = "", tag: str = "span") -> str:
    c = CLAIMS[cid]
    k = f' class="{cls}"' if cls else ""
    return f'<{tag}{k} data-claim="{cid}">{c["display_html"]}</{tag}>'


def field(src: str, eid: str, fld: str, tag: str = "span") -> str:
    table = CLAIMS if src == "claim" else LIT
    v = table[eid][fld]
    if isinstance(v, list):
        v = ", ".join(v)
    return f'<{tag} data-field="{src}:{eid}:{fld}">{esc(str(v))}</{tag}>'


# -- charts ----------------------------------------------------------------------------

FAST_ROWS = [("river", "River plume"), ("heat", "Heat conduction"), ("sound", "Sound"), ("farm", "Wind farm"),
             ("bimetal", "Heated structure"), ("circuit", "Current in a plate"), ("structure", "Loaded structure"),
             ("cooled", "Cooled block")]


def chart_fast() -> str:
    rows = sorted(FAST_ROWS, key=lambda r: -CLAIMS[f"fast.{r[0]}.s"]["value"])
    W, L, R, top, rh = 760, 200, 700, 34, 42
    lo, hi = -3, 1
    x = lambda v: L + (math.log10(v) - lo) / (hi - lo) * (R - L)
    H = top + rh * len(rows) + 12
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="fc-t fc-d">',
           '<title id="fc-t">Speed of the pieces against the undivided solve, one example per kind</title>',
           '<desc id="fc-d">A bar per kind of physics on a logarithmic scale; bars to the right of the same-speed line are faster in pieces, bars to the left are faster solved whole. The values are written at the bar ends.</desc>']
    for e in range(lo, hi + 1):
        v = 10 ** e
        xx = x(v)
        lab = f"{v:g}{TIMES}"
        out.append(f'<line class="ax" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top - 8}" y2="{H - 8}"/>')
        out.append(f'<text x="{xx:.1f}" y="{top - 14}" text-anchor="middle"><tspan data-axis="">{lab}</tspan></text>')
    x1 = x(1)
    for i, (k, name) in enumerate(rows):
        c = CLAIMS[f"fast.{k}.s"]
        v = c["value"]
        yc = top + rh * i + rh / 2
        bh, r = 18, 4
        xv = x(v)
        fast = v >= 1
        if fast:
            d = (f"M{x1:.1f},{yc - bh / 2:.1f} H{xv - r:.1f} Q{xv:.1f},{yc - bh / 2:.1f} {xv:.1f},{yc - bh / 2 + r:.1f} "
                 f"V{yc + bh / 2 - r:.1f} Q{xv:.1f},{yc + bh / 2:.1f} {xv - r:.1f},{yc + bh / 2:.1f} H{x1:.1f} Z")
        else:
            d = (f"M{x1:.1f},{yc - bh / 2:.1f} H{xv + r:.1f} Q{xv:.1f},{yc - bh / 2:.1f} {xv:.1f},{yc - bh / 2 + r:.1f} "
                 f"V{yc + bh / 2 - r:.1f} Q{xv:.1f},{yc + bh / 2:.1f} {xv + r:.1f},{yc + bh / 2:.1f} H{x1:.1f} Z")
        tx, anchor = (xv + 8, "start") if fast else (xv - 8, "end")
        out.append(f'<g class="row" tabindex="0">')
        out.append(f'<title data-field="claim:fast.{k}.s:sentence">{esc(c["sentence"])}</title>')
        out.append(f'<rect class="hit" x="0" y="{yc - rh / 2:.1f}" width="{W}" height="{rh}"/>')
        out.append(f'<text class="lbl" x="{L - 14}" y="{yc + 5:.1f}" text-anchor="end">{esc(name)}</text>')
        out.append(f'<path class="bar {"fast" if fast else "slow"}" d="{d}"/>')
        out.append(f'<text class="val" x="{tx:.1f}" y="{yc + 5:.1f}" text-anchor="{anchor}">'
                   f'<tspan data-claim="fast.{k}.s">{esc(c["display"])}</tspan></text>')
        out.append("</g>")
    out.append(f'<line class="one" x1="{x1:.1f}" x2="{x1:.1f}" y1="{top - 8}" y2="{H - 8}"/>')
    out.append("</svg>")
    return "\n".join(out)


def chart_learned() -> str:
    arms = [("F", "The whole domain"), ("Ep", "Classical pieces"), ("L", "Learned pieces"), ("Cc", "Classical pieces, coarser grid")]
    W, H, L, R, T, B = 560, 360, 70, 520, 30, 300
    x = lambda t: L + (math.log10(t) + 1) / 2 * (R - L)
    y = lambda e: B - e / 0.08 * (B - T)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="lc-t lc-d">',
           '<title id="lc-t">The learned case: time per macro-step against error, four ways of running the farm</title>',
           '<desc id="lc-d">The learned pieces sit beside the classical pieces at the same error and a little faster; the classical pieces on a coarser grid are both faster and closer to the reference; the whole domain is closest and slowest.</desc>']
    for e in range(0, 9, 2):
        yy = y(e / 100)
        out.append(f'<line class="ax" x1="{L}" x2="{R}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        out.append(f'<text x="{L - 10}" y="{yy + 4:.1f}" text-anchor="end"><tspan data-axis="">{e}</tspan></text>')
    for p in (0.1, 1, 10):
        xx = x(p)
        out.append(f'<line class="ax" x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" y2="{B}"/>')
        out.append(f'<text x="{xx:.1f}" y="{B + 20}" text-anchor="middle"><tspan data-axis="">{p:g}</tspan></text>')
    out.append(f'<text x="{(L + R) / 2:.0f}" y="{B + 44}" text-anchor="middle">seconds per macro-step (logarithmic)</text>')
    out.append(f'<text x="18" y="{(T + B) / 2:.0f}" text-anchor="middle" transform="rotate(-90 18 {(T + B) / 2:.0f})">farm power off the reference, %</text>')
    place = {"F": (-12, -16, "end"), "Ep": (12, 30, "start"), "L": (-12, -18, "end"), "Cc": (12, -14, "start")}
    for a, name in arms:
        t, e = CLAIMS[f"lc.{a}.t"]["value"], CLAIMS[f"lc.{a}.P"]["value"]
        cx, cy = x(t), y(e)
        dx, dy, anchor = place[a]
        cls = "dot-l" if a == "L" else "dot-o"
        out.append(f'<g class="pt" tabindex="0"><title data-field="claim:lc.{a}.P:sentence">{esc(CLAIMS[f"lc.{a}.P"]["sentence"])}</title>')
        out.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="16"/>')
        out.append(f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{7 if a == "L" else 6}"/>')
        out.append(f'<text class="lbl" x="{cx + dx:.1f}" y="{cy + dy:.1f}" text-anchor="{anchor}">{esc(name)}</text>')
        out.append(f'<text x="{cx + dx:.1f}" y="{cy + dy + 17:.1f}" text-anchor="{anchor}">'
                   f'<tspan data-claim="lc.{a}.P">{esc(CLAIMS[f"lc.{a}.P"]["display"])}</tspan>, '
                   f'<tspan data-claim="lc.{a}.t">{esc(CLAIMS[f"lc.{a}.t"]["display"])}</tspan></text></g>')
    out.append("</svg>")
    return "\n".join(out)


# -- literature cards ----------------------------------------------------------------------

ICONS = {
    "L2": '<path d="M4 4h10v10H4zM18 4h10v10H18zM4 18h10v10H4zM18 18h10v10H18z" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "L4": '<path d="M4 16h24M16 4v24" stroke="currentColor" stroke-width="1.6"/><circle cx="16" cy="16" r="6" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "N1": '<path d="M3 20c5-8 9 4 14-4s9-6 12-2" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M3 26c5-6 9 2 14-3s9-4 12-1" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "N2": '<path d="M3 21h26M6 21l3-6h12l5 6M9 25a2 2 0 1 0 0-.1M23 25a2 2 0 1 0 0-.1" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "N3": '<path d="M16 27C6 20 4 14 7 9c3-4 7-3 9 0 2-3 6-4 9 0 3 5 1 11-9 18z" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "N4": '<path d="M4 4h8v8H4zM12 4h8v8h-8zM20 4h8v8h-8zM4 12h8v8H4zM12 12h8v8h-8zM20 12h8v8h-8zM4 20h8v8H4zM12 20h8v8h-8zM20 20h8v8h-8z" fill="none" stroke="currentColor" stroke-width="1.2"/>',
    "N5": '<path d="M4 6h14v14H4zM14 12h14v14H14z" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "N6": '<path d="M16 4l10 6v12l-10 6-10-6V10z M6 10l10 6 10-6 M16 16v12" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "N7": '<path d="M4 24c4 0 4-16 8-16s4 16 8 16 4-16 8-16" fill="none" stroke="currentColor" stroke-width="1.6"/>',
}


def lit_card(e: dict) -> str:
    big = ""
    if e["values"]:
        v = e["values"][0]
        big = (f'<div class="fig"><span data-lit="{e["id"]}.{v["key"]}">{esc(v["display"])}</span>'
               f'<small>{esc(v["label"])}</small></div>')
    link = e["url"] or ("https://doi.org/" + e["doi"])
    doi = f' · <a href="https://doi.org/{e["doi"]}">DOI</a>' if e.get("doi") else ""
    return ("\n".join([
        '<article class="card reveal">',
        f'<svg class="icon" viewBox="0 0 32 32" aria-hidden="true">{ICONS.get(e["id"], "")}</svg>',
        '<p class="who">Published by other researchers</p>',
        f'<h3>{esc(e["title"])}</h3>',
        big,
        f'<p class="claim" data-field="lit:{e["id"]}:claim">{esc(e["claim"])}</p>',
        f'<p class="cite"><span data-lit-ref="{e["id"]}">{esc(e["citation"])}</span>. '
        f'<a href="{esc(link)}">Read the source</a>{doi} · <a href="evidence.html#{e["id"]}">The quotation</a></p>',
        "</article>"]))


# -- the proofs' dependency graph --------------------------------------------------------------

HEADLINES = ("Atlas.tier0_headline", "Atlas.tier2_headline", "Atlas.defect_correction_headline")


def parse_dot() -> tuple[dict, list]:
    p = os.path.join(SITE, "proofs", "blueprint", "dep_graph_document.html")
    with open(p, encoding="utf-8") as fh:
        t = fh.read()
    dot = re.search(r"renderDot\(`(.*?)`", t, re.S).group(1)
    body = dot[dot.index("{") + 1: dot.rindex("}")]
    nodes, edges = {}, []
    for stmt in re.split(r";", body):
        s = stmt.strip()
        m = re.match(r'"([^"]+)"\s*->\s*"([^"]+)"', s)
        if m:
            edges.append((m.group(1), m.group(2)))
            continue
        m = re.match(r'"([^"]+)"\s*\[(.*)\]$', s, re.S)
        if m:
            attrs = m.group(2)
            lab = re.search(r'label=("([^"]*)"|([^,\s\]]+))', attrs)
            label = lab.group(2) if lab.group(2) is not None else lab.group(3)
            shape = (re.search(r"shape=(\w+)", attrs) or [None, "ellipse"])[1]
            nodes[m.group(1)] = {"label": label, "shape": shape}
    return nodes, edges


def proof_graph() -> str:
    nodes, edges = parse_dot()
    # the headline theorems' blueprint labels, from the chapter that states them
    preds = {n: [] for n in nodes}
    for a, b in edges:
        if a in nodes and b in nodes:
            preds[b].append(a)
    layer: dict[str, int] = {}

    def depth(n, seen=()):
        if n in layer:
            return layer[n]
        d = 0 if not preds[n] else 1 + max(depth(p, seen + (n,)) for p in preds[n] if p not in seen)
        layer[n] = d
        return d
    for n in nodes:
        depth(n)
    heads = {n for n in nodes if "headline" in n.lower() or "headline" in nodes[n]["label"].lower()}
    by_layer: dict[int, list] = {}
    for n, d in layer.items():
        by_layer.setdefault(d, []).append(n)
    cols = max(by_layer) + 1
    per_row = 12
    bw, bh, gx, gy = 92, 22, 104, 30
    rows_in = {d: math.ceil(len(v) / per_row) for d, v in by_layer.items()}
    y0, pos = 10, {}
    ys = {}
    for d in range(cols):
        ys[d] = y0
        y0 += rows_in.get(d, 0) * gy + 26
    W = per_row * gx + 20
    for d, ns in by_layer.items():
        ns.sort(key=lambda n: nodes[n]["label"])
        for i, n in enumerate(ns):
            r, c = divmod(i, per_row)
            pos[n] = (10 + c * gx + (per_row - min(per_row, len(ns) - r * per_row)) * gx / 2, ys[d] + r * gy)
    H = y0
    out = [f'<svg id="dep-graph" viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="The dependency graph of the blueprint: every definition and theorem, in layers from the first results to the three headline theorems, every one checked.">']
    for a, b in edges:
        if a in pos and b in pos:
            (ax, ay), (bx, by) = pos[a], pos[b]
            out.append(f'<path class="dep-edge" d="M{ax + bw / 2:.0f},{ay + bh:.0f} C{ax + bw / 2:.0f},{ay + bh + 14:.0f} {bx + bw / 2:.0f},{by - 14:.0f} {bx + bw / 2:.0f},{by:.0f}"/>')
    for n, (px, py) in pos.items():
        lab = nodes[n]["label"]
        short = lab if len(lab) <= 13 else lab[:12] + "…"
        cls = "dep-node head" if n in heads else "dep-node"
        rx = 3 if nodes[n]["shape"] == "box" else 11
        out.append(f'<g class="{cls}" data-layer="{layer[n]}"><title>{esc(lab)}</title>'
                   f'<rect x="{px:.0f}" y="{py:.0f}" width="{bw}" height="{bh}" rx="{rx}"/>'
                   f'<text x="{px + bw / 2:.0f}" y="{py + 15:.0f}" text-anchor="middle">{esc(short)}</text></g>')
    out.append("</svg>")
    return "\n".join(out)


# -- evidence tables -------------------------------------------------------------------------

GROUPS = [("fast", "The Fast examples", ("fast.", "machine.")), ("gallery", "Agreement and balance", ("agree.", "balance.", "schan.")),
          ("w346", "Speed by farm size", ("w346.",)), ("lc", "The learned case", ("lc.",)),
          ("lean", "The proofs", ("lean.",)), ("arch", "The architecture's measurements", ("arch.",)),
          ("inst", "The installer", ("inst.",))]


def rec_link(c: dict) -> str:
    links = []
    for src, pub in zip(c["source"], c["public"]):
        name = src.rsplit("/", 1)[-1]
        links.append(f'<a href="{esc(pub)}"><code>{esc(name)}</code></a>')
    if len(links) > 3:
        return links[0] + ' and the others <a href="#records">listed below</a>'
    return ", ".join(links)


def evidence_claims() -> str:
    out = []
    for gid, title, prefixes in GROUPS:
        ids = [k for k in CLAIMS if k.startswith(prefixes)]
        if not ids:
            continue
        out.append(f'<h3 id="{gid}" style="margin-top:40px">{esc(title)}</h3><div class="table-wrap"><table class="ev">')
        out.append("<thead><tr><th>value</th><th>what it is</th><th>conditions and caveat</th><th>record</th></tr></thead><tbody>")
        for k in ids:
            c = CLAIMS[k]
            cond = []
            if c["machine"]:
                cond.append(field("claim", k, "machine"))
            if c["caveat"]:
                cond.append(field("claim", k, "caveat"))
            how = field("claim", k, "pointer", "code") if c["pointer"] else field("claim", k, "derived")
            out.append(f'<tr id="{esc(k)}"><td class="v">{claim(k)}</td><td>{field("claim", k, "sentence")}'
                       f'<br><span class="fine">{how}</span></td><td class="small">{"<br>".join(cond)}</td>'
                       f'<td class="small">{rec_link(c)}</td></tr>')
        out.append("</tbody></table></div>")
    t3 = [r for r in RECORDS if r["source"].endswith("t3_counterexample.json")][0]
    out.append(f'<h3 id="t3" style="margin-top:40px">The finding in the proofs</h3><p>The counterexample to T3 as first written, '
               f'with the controls that pass: <a href="{esc(t3["public"])}"><code>t3_counterexample.json</code></a>. '
               f'The corrected statement, which needs a convergent sweep, is in the <a href="proofs/blueprint/index.html">blueprint</a>.</p>')
    return "\n".join(out)


def evidence_literature() -> str:
    out = ['<div class="table-wrap"><table class="ev"><thead><tr><th>on the site</th><th>the source\'s own words</th><th>where</th><th>source</th></tr></thead><tbody>']
    for e in LIT.values():
        vals = "<br>".join(f'<span data-lit="{e["id"]}.{v["key"]}">{esc(v["display"])}</span>' for v in e["values"])
        link = e["url"] or ("https://doi.org/" + e["doi"])
        out.append(f'<tr id="{e["id"]}"><td class="v">{vals or "—"}</td><td><em>{field("lit", e["id"], "quote_display" if "quote_display" in e else "quote")}</em></td>'
                   f'<td class="small">{field("lit", e["id"], "location")}<br><span class="fine">read on {field("lit", e["id"], "read_on")}</span></td>'
                   f'<td class="small"><span data-lit-ref="{e["id"]}">{esc(e["citation"])}</span>. <a href="{esc(link)}">Source</a></td></tr>')
    out.append("</tbody></table></div>")
    out.append('<p class="small">Results we could not read at their source are not shown, whatever their size.</p>')
    return "\n".join(out)


def evidence_records() -> str:
    out = ['<div class="table-wrap"><table class="ev"><thead><tr><th>published copy</th><th>original in the project</th></tr></thead><tbody>']
    for r in RECORDS:
        out.append(f'<tr><td><a href="{esc(r["public"])}"><code>{esc(r["public"])}</code></a></td><td><code>{esc(r["source"])}</code></td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


# -- the vision's agent graphs, drawn over the owner's illustrations -----------------------------
#
# Coordinates are pixels of each image (wide crop, phone crop), so the SVG uses the image's own
# viewBox with "slice", which crops exactly as the image's object-fit: cover does. Nodes sit on the
# parts the scenario's agent graph names (vision-scenarios-and-image-prompts §2-§4); edges carry the
# port type that joins them.

VISION = {
    "1a": {
        "size": {"wide": (1672, 941), "tall": (1122, 1402)},
        "nodes": {
            "air": ("air over the front wing", (330, 530), (90, 800)),
            "tyres": ("tyres", (1470, 470), (1060, 610)),
            "brakes": ("brakes", (860, 500), (865, 725)),
            "rad": ("radiators", (1120, 440), (960, 525)),
            "pu": ("power unit", (1270, 320), (700, 455)),
            "bat": ("battery", (1030, 580), (560, 650)),
        },
        "edges": [("air", "brakes", "THERM"), ("air", "rad", "ADVEC"), ("brakes", "tyres", "ROT"),
                  ("rad", "pu", "THERM"), ("pu", "bat", "ELEC"), ("pu", "tyres", "ROT"), ("air", "tyres", "MECH")],
        "label": "Agent graph over the car: the air over the front wing, the brakes, the tyres, the radiators, the power unit and the battery, linked by typed ports.",
    },
    "2a": {
        "size": {"wide": (1916, 821), "tall": (1122, 1402)},
        "nodes": {
            "ridge": ("ridge", (300, 330), (150, 700)),
            "valley": ("valley", (720, 440), (380, 640)),
            "slope": ("slope", (1130, 340), (700, 505)),
            "break": ("firebreak", (1160, 450), (650, 730)),
            "air": ("atmosphere", (900, 170), (350, 330)),
            "plane": ("aircraft", (1555, 285), (995, 380)),
            "sense": ("sensors", (1760, 100), (960, 130)),
        },
        "edges": [("ridge", "valley", "THERM"), ("valley", "slope", "ADVEC"), ("slope", "break", "THERM"),
                  ("air", "valley", "MECH"), ("air", "slope", "ADVEC"), ("plane", "break", "MECH"), ("sense", "air", "ADVEC")],
        "label": "Agent graph over the landscape: terrain pieces along the ridge, the valley and the slope, the atmosphere above, the firebreak, the aircraft and the sensors, linked by typed ports.",
    },
    "3a": {
        "size": {"wide": (1916, 821), "tall": (1122, 1402)},
        "nodes": {
            "plume": ("lander plume", (310, 300), (185, 330)),
            "reg": ("regolith", (330, 525), (180, 700)),
            "solar": ("solar towers", (1030, 250), (685, 560)),
            "grid": ("microgrid", (1250, 520), (700, 1000)),
            "hab": ("habitat", (830, 525), (420, 945)),
            "power": ("power station", (1730, 470), (1085, 955)),
            "bat": ("batteries", (965, 548), (600, 990)),
            "rover": ("rover", (1540, 600), (900, 1095)),
        },
        "edges": [("plume", "reg", "MECH"), ("reg", "hab", "ADVEC"), ("solar", "grid", "ELEC"), ("power", "grid", "ELEC"),
                  ("bat", "grid", "ELEC"), ("hab", "grid", "ELEC"), ("rover", "grid", "ELEC"), ("reg", "solar", "ADVEC")],
        "label": "Agent graph over the base: the lander's plume, the regolith, the solar towers, the power station, the batteries, the habitat and the rover, joined through a microgrid.",
    },
}


def overlay(key: str) -> str:
    v = VISION[key]
    out = []
    for crop, idx in (("wide", 1), ("tall", 2)):
        w, h = v["size"][crop]
        # scale marks to how wide the image is shown: about 1100 px wide, 360 px on a phone
        s = w / 1000 if crop == "wide" else w / 400
        pos = {k: n[idx] for k, n in v["nodes"].items()}
        g = [f'<svg class="overlay {crop}" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice" role="img" aria-label="{esc(v["label"])}">']
        for i, (a, b, port) in enumerate(v["edges"]):
            (x1, y1), (x2, y2) = pos[a], pos[b]
            d = f"M{x1} {y1} L{x2} {y2}"
            g.append(f'<path class="ov-edge" d="{d}"/><path class="ov-pulse" d="{d}" pathLength="100" style="animation-delay:{0.45 * i:.2f}s"/>')
            g.append(f'<text class="ov-port" x="{(x1 + x2) / 2 + 8 * s:.0f}" y="{(y1 + y2) / 2 - 6 * s:.0f}" font-size="{11 * s:.0f}">{port}</text>')
        for k, n in v["nodes"].items():
            x, y = n[idx]
            g.append(f'<g class="ov-node"><circle cx="{x}" cy="{y}" r="{9 * s:.0f}" stroke-width="{2 * s:.1f}"/>'
                     f'<text x="{x}" y="{y - 16 * s:.0f}" text-anchor="middle" font-size="{13 * s:.0f}" stroke-width="{3 * s:.1f}">{esc(n[0])}</text></g>')
        g.append("</svg>")
        out.append("\n".join(g))
    return "\n".join(out)


# -- install ----------------------------------------------------------------------------------

def commands(os_name: str) -> str:
    url = CFG["public_repo"] + ".git"
    folder = url.rsplit("/", 1)[1][:-4]
    clone = f"git clone --depth 1 --branch {CFG['source_branch']} {url}"
    run = ".\\run.cmd" if os_name == "win" else "./run.sh"
    return f"<pre>{esc(clone)}\ncd {esc(folder)}\n{esc(run)}</pre>"


def launch_banner() -> str:
    if CFG.get("launched"):
        return ""
    return ('<p class="banner"><strong>Before launch.</strong> The demo\'s repository becomes public on the day this site '
            'does. Until then the commands below will not find it.</p>')


GEN = {
    "chart-fast": lambda: f'<div class="chart-scroll">{chart_fast()}</div>',
    "chart-learned": lambda: f'<div class="chart-scroll small">{chart_learned()}</div>',
    "lit-classical": lambda: "\n".join(lit_card(e) for e in LIT.values() if e["group"] == "classical"),
    "lit-neural": lambda: "\n".join(lit_card(e) for e in LIT.values() if e["group"] == "neural"),
    "proof-graph": proof_graph,
    "evidence-claims": evidence_claims, "evidence-literature": evidence_literature, "evidence-records": evidence_records,
    "launch-banner": launch_banner,
    "details-1": lambda: "\n".join(figure(k) for k in ("1c", "1d", "1e")),
    "details-3": lambda: "\n".join(figure(k) for k in ("3c", "3d")),
    "figure-M1": lambda: figure("M1", "m1"),
    "overlay-1a": lambda: overlay("1a"), "overlay-2a": lambda: overlay("2a"), "overlay-3a": lambda: overlay("3a"),
    "cmd-win": lambda: commands("win"), "cmd-mac": lambda: commands("mac"), "cmd-linux": lambda: commands("linux"),
}


# -- images ------------------------------------------------------------------------------------

def image_block(name: str, current: str) -> str:
    """The owner's illustration, if it has arrived, else the placeholder as it is."""
    side = os.path.join(SITE, "images", name + ".json")
    if not os.path.exists(side):
        return current
    with open(side, encoding="utf-8") as fh:
        meta = json.load(fh)
    srcs = meta["files"]
    alt = esc(meta["alt"])
    lazy = 'fetchpriority="high" ' if meta.get("eager") else 'loading="lazy" '
    if "mobile" in srcs:
        return (f'<picture><source media="(max-width: 700px)" srcset="images/{srcs["mobile"]}">'
                f'<img src="images/{srcs["desktop"]}" alt="{alt}" width="{meta["width"]}" height="{meta["height"]}" '
                f'{lazy}decoding="async"></picture>')
    return (f'<img src="images/{srcs["desktop"]}" alt="{alt}" width="{meta["width"]}" height="{meta["height"]}" '
            f'loading="lazy" decoding="async">')


def figure(name: str, cls: str = "detail") -> str:
    """An optional illustration with its tag, for the gen:details-* rows and gen:figure-*."""
    side = os.path.join(SITE, "images", name + ".json")
    if not os.path.exists(side):
        return ""
    return f'<figure class="{cls} reveal">{image_block(name, "")}<span class="tag">Illustration</span></figure>'


# -- stamping ------------------------------------------------------------------------------------

STAMP = re.compile(r'(<(?P<tag>[a-z][a-z0-9]*)\b(?P<attrs>[^>]*?)\bdata-(?P<kind>claim|lit-ref|lit|field)="(?P<key>[^"]+)"(?P<rest>[^>]*)>)(?P<inner>.*?)(</(?P=tag)>)', re.S)


def stamp(page: str) -> str:
    def repl(m):
        kind, key = m.group("kind"), m.group("key")
        if kind == "claim":
            val = CLAIMS[key]["display_html"]
        elif kind == "lit":
            eid, vk = key.split(".", 1)
            val = esc([v for v in LIT[eid]["values"] if v["key"] == vk][0]["display"])
        elif kind == "lit-ref":
            val = esc(LIT[key]["citation"])
        else:
            src, rest = key.split(":", 1)
            eid, fld = rest.rsplit(":", 1)
            v = (CLAIMS if src == "claim" else LIT)[eid][fld]
            val = esc(", ".join(v) if isinstance(v, list) else str(v))
        return m.group(1) + val + m.group(len(m.groups()))
    return STAMP.sub(repl, page)


def regen(page: str) -> str:
    def repl(m):
        name = m.group(1)
        return f"<!-- gen:{name} -->\n{GEN[name]()}\n<!-- /gen:{name} -->"
    page = re.sub(r"<!-- gen:([\w-]+) -->.*?<!-- /gen:\1 -->", repl, page, flags=re.S)

    def img(m):
        return f"<!-- img:{m.group(1)} -->\n{image_block(m.group(1), m.group(2).strip())}\n<!-- /img:{m.group(1)} -->"
    page = re.sub(r"<!-- img:([\w-]+) -->(.*?)<!-- /img:\1 -->", img, page, flags=re.S)

    def cfg(m):
        return f'{m.group(1)}data-config="{m.group(2)}" href="{esc(CFG[m.group(2)])}"'
    page = re.sub(r'(<a\b[^>]*?)data-config="([\w_]+)" href="[^"]*"', cfg, page)
    return page


# -- copies ----------------------------------------------------------------------------------------

LEAN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Atlas proofs</title><link rel="stylesheet" href="{up}assets/css/site.css">
<style>li code{{word-break:break-all}}pre{{font-family:var(--mono);font-size:13.5px;line-height:1.6;background:var(--bg-2);border-radius:14px;padding:20px;overflow-x:auto;white-space:pre}}</style>
</head><body><main class="section tight"><div class="wrap">
<p class="small"><a href="{up}index.html">Atlas</a> · <a href="{up}proofs/blueprint/index.html">the blueprint</a> · <a href="{idx}">all files</a></p>
<h1 class="h3">{title}</h1>
{body}
</div></main></body></html>
"""


def copy_architecture() -> None:
    src = os.path.join(ROOT, "proposal", "architecture", "index.html")
    dst = os.path.join(SITE, "architecture", "index.html")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def copy_lean() -> None:
    base = os.path.join(SITE, "proofs", "lean")
    if os.path.isdir(base):
        shutil.rmtree(base)
    files = ["AtlasProofs.lean", "lakefile.toml", "lean-toolchain", "lake-manifest.json"]
    files += sorted("AtlasProofs/" + f for f in os.listdir(os.path.join(ROOT, "lean", "AtlasProofs")) if f.endswith(".lean"))
    rows = []
    for rel in files:
        with open(os.path.join(ROOT, "lean", rel), encoding="utf-8") as fh:
            txt = fh.read()
        raw = os.path.join(base, rel)
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        with open(raw, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
        depth = rel.count("/") + 2
        up = "../" * depth
        page = LEAN_PAGE.format(title=esc(rel), up=up, idx="../" * (depth - 2) + "index.html",
                                body=f'<p class="small"><a href="{esc(os.path.basename(rel))}">Plain text</a></p><pre>{esc(txt)}</pre>')
        with open(raw + ".html", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(page)
        rows.append(f'<li><a href="{esc(rel)}.html"><code>{esc(rel)}</code></a></li>')
    idx = LEAN_PAGE.format(title="The Lean project", up="../../", idx="index.html",
                           body="<p>The proofs, as Lean 4 sources on Mathlib. Build them with <code>lake build</code> in a copy "
                                "of this folder; the blueprint explains each statement in plain words.</p><ul>" + "".join(rows) + "</ul>")
    with open(os.path.join(base, "index.html"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(idx)


def main() -> int:
    copy_architecture()
    copy_lean()
    for p in PAGES:
        path = os.path.join(SITE, p)
        with open(path, encoding="utf-8") as fh:
            page = fh.read()
        new = stamp(regen(page))
        new = stamp(new)
        if new != page:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(new)
        print(f"{p}: {'updated' if new != page else 'unchanged'}")
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
