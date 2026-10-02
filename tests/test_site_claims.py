"""The website's honesty check: no number on the site that is not in the claims or the
literature file, and no claim that differs from its record.

What it checks:

1. ``site/data/claims.json`` is exactly what ``scripts/site_claims.py`` derives from the
   records now: a claim whose record changed, or a value edited by hand, fails here.
2. Every claim read through a pointer equals its record's value at that pointer, read
   independently of the pipeline, and its displayed text is its value's formatting.
3. Every published record copy is the cleaned record, and nothing private (the
   computer's name, a user name, a local path) or forbidden (NeuberNet, Poseidon's
   weights) is anywhere under ``site/``.
4. Every literature value's words appear in the quotation it comes from.
5. **Every number on the site's own pages** (``index.html``, ``install.html``,
   ``evidence.html``, their SVG included) sits in an element that names its source:
   ``data-claim`` (a claim, whose text must equal the claim's display), ``data-lit``
   (a literature value), ``data-lit-ref`` (a citation, generated from the file), or
   ``data-axis`` (a chart's tick mark, which states no result). The only other digits
   allowed are identifiers (G5, T3, S1, H1, C1...), "2-D" and "3-D", and the stage
   names, listed in `LABELS` below.

Run:  python -m pytest tests/test_site_claims.py -q
"""
from __future__ import annotations

import json
import os
import re
import sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import site_claims as sc  # noqa: E402

PAGES = ("index.html", "install.html", "evidence.html")

#: digits that are names, not results. Each pattern is matched against the token that
#: contains the digit; anything else with a digit must carry its source.
LABELS = [
    r"[A-Z]{1,2}\d{1,2}[a-z]?",      # G5, T3, S12, C1, A11, N7, L2, H1, W354, O3
    r"S1–S12",
    r"[23]-D",                       # 2-D, 3-D
    r"Stage", r"[0-3]",              # only as a stage name: checked in context below
    r"©?",
]
FORBIDDEN = re.compile(r"neuber|poseidon|nauni|\bchamp\b|[a-z]:\\+users|/home/|onedrive", re.I)


def claims_file() -> dict:
    with open(os.path.join(SITE, "data", "claims.json"), encoding="utf-8") as fh:
        return json.load(fh)


def literature() -> dict:
    with open(os.path.join(SITE, "data", "literature.json"), encoding="utf-8") as fh:
        return json.load(fh)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(" ", " ")).strip()


# -- 1 and 2: the claims against their records -----------------------------------------

def test_claims_file_is_what_the_records_give_now():
    sc.WRITE = False
    try:
        fresh = sc.build()
    finally:
        sc.WRITE = True
    on_disk = claims_file()
    assert json.loads(json.dumps(fresh, ensure_ascii=False)) == on_disk, \
        "site/data/claims.json differs from the records: run python scripts/site_claims.py"


def test_each_pointer_claim_equals_its_record():
    n = 0
    for c in claims_file()["claims"]:
        if not c["pointer"]:
            assert c["derived"], f"{c['id']}: a claim needs a pointer or a stated derivation"
            continue
        rec = sc.load(c["source"][0])
        assert sc.resolve(rec, c["pointer"]) == c["value"], c["id"]
        n += 1
    assert n >= 40


def test_display_is_the_values_formatting():
    for c in claims_file()["claims"]:
        text, html = sc.fmt(c["kind"], c["value"], c["nd"])
        assert c["display"] == text and c["display_html"] == html, c["id"]


def test_timed_claims_name_the_machine_they_were_measured_on():
    for c in claims_file()["claims"]:
        if c["kind"] in ("ratio", "sec") and c["id"].split(".")[0] in ("fast", "lc", "w346"):
            if c["id"].endswith((".s", "_over_l", ".lo")):
                assert c["machine"], c["id"]


# -- 3: the published copies -----------------------------------------------------------

def test_published_records_are_the_cleaned_records():
    for r in claims_file()["records"]:
        with open(os.path.join(SITE, r["public"]), encoding="utf-8") as fh:
            got = json.load(fh)
        want = json.loads(json.dumps(sc.published_copy(r["source"]), ensure_ascii=False))
        assert got == want, r["public"]


def test_nothing_private_or_forbidden_reaches_the_site():
    bad = []
    for dirpath, _, files in os.walk(SITE):
        for f in files:
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, SITE)
            if f.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".npy")):
                bad.append(rel + " (a weights or array file)")
                continue
            if FORBIDDEN.search(rel):
                bad.append(rel + " (path)")
            if f.endswith((".html", ".json", ".js", ".css", ".txt", ".md", ".lean", ".svg", ".xml")):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    m = FORBIDDEN.search(fh.read())
                if m:
                    bad.append(f"{rel}: {m.group(0)!r}")
    assert not bad, bad


# -- 4: the literature -----------------------------------------------------------------

def test_literature_values_are_in_their_quotes():
    ids = set()
    for e in literature()["entries"]:
        assert e["id"] not in ids
        ids.add(e["id"])
        assert e["read_on"] and e["quote"] and e["citation"] and (e["url"] or e["doi"])
        for v in e["values"]:
            assert v["fragment"] in e["quote"], (e["id"], v["key"])


# -- 5: every number on the site's pages -----------------------------------------------

class Page(HTMLParser):
    """Collects the text of a page, attributing each text run to the nearest
    enclosing element that carries a source attribute."""

    SOURCES = ("data-claim", "data-lit", "data-lit-ref", "data-axis")
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
            "track", "wbr", "path", "circle", "rect", "line", "polyline", "polygon", "stop", "use"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict]] = []
        self.free: list[str] = []                # text with no source
        self.sourced: list[tuple[str, str, str]] = []   # (attr, key, text)
        self._buf: dict[int, list[str]] = {}
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style", "code", "pre", "kbd", "noscript"):
            self.skip += 1
        if tag in self.VOID:
            for k in ("alt", "aria-label", "title"):
                if a.get(k):
                    self._text(a[k])
            return
        self.stack.append((tag, a))
        for k in ("aria-label", "title"):
            if a.get(k):
                self._text(a[k])
        if any(s in a for s in self.SOURCES):
            self._buf[len(self.stack)] = []

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID and self.stack and self.stack[-1][0] == tag:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "code", "pre", "kbd", "noscript"):
            self.skip = max(0, self.skip - 1)
        if tag in self.VOID:
            return
        while self.stack:
            depth = len(self.stack)
            t, a = self.stack.pop()
            if depth in self._buf:
                text = norm("".join(self._buf.pop(depth)))
                attr = next(s for s in self.SOURCES if s in a)
                self.sourced.append((attr, a[attr], text))
            if t == tag:
                break

    def _text(self, data):
        if self.skip and not any(d in self._buf for d in range(1, len(self.stack) + 1)):
            return
        owners = [d for d in self._buf if d <= len(self.stack)]
        if owners:
            self._buf[max(owners)].append(data)
        else:
            self.free.append(data)

    def handle_data(self, data):
        if self.skip:
            return
        self._text(data)


def parse(page: str) -> Page:
    with open(os.path.join(SITE, page), encoding="utf-8") as fh:
        p = Page()
        p.feed(fh.read())
        p.close()
    return p


def label_ok(token: str, context: str) -> bool:
    if re.fullmatch(r"[0-3]", token):
        return bool(re.search(r"Stage\s+" + token + r"\b", context))
    return any(re.fullmatch(pat, token) for pat in LABELS)


def test_every_number_on_the_pages_has_a_source():
    claims = {c["id"]: c for c in claims_file()["claims"]}
    lit = {e["id"]: e for e in literature()["entries"]}
    lit_vals = {f"{e['id']}.{v['key']}": v for e in lit.values() for v in e["values"]}
    problems = []
    for page in PAGES:
        if not os.path.exists(os.path.join(SITE, page)):
            continue
        p = parse(page)
        for attr, key, text in p.sourced:
            if attr == "data-claim":
                if key not in claims:
                    problems.append(f"{page}: unknown claim {key}")
                elif text != norm(claims[key]["display"]):
                    problems.append(f"{page}: {key} shows {text!r}, the claim is {claims[key]['display']!r}")
            elif attr == "data-lit":
                if key not in lit_vals:
                    problems.append(f"{page}: unknown literature value {key}")
                elif text != norm(lit_vals[key]["display"]):
                    problems.append(f"{page}: {key} shows {text!r}, the file says {lit_vals[key]['display']!r}")
            elif attr == "data-lit-ref":
                if key not in lit or text != norm(lit[key]["citation"]):
                    problems.append(f"{page}: citation {key} is not the file's")
            elif attr == "data-axis":
                if not re.fullmatch(r"[\d.,]+×?|10[−-]?\d+|", text.replace(" ", "")):
                    problems.append(f"{page}: axis label {text!r} is not a tick")
        free = norm(" ".join(p.free))
        for m in re.finditer(r"[\w−–\-]*\d[\w−–\-]*", free):
            tok = m.group(0)
            ctx = free[max(0, m.start() - 12): m.end() + 2]
            if not label_ok(tok, ctx):
                problems.append(f"{page}: unsourced number {tok!r} in ...{free[max(0, m.start() - 40): m.end() + 40]}...")
    assert not problems, "\n".join(problems)
