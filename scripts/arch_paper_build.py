"""arch_paper_build -- number, cross-reference and inline the figures of the proposal page.

``proposal/architecture/index.html`` is the only source of the paper.  This script edits it
in place and is idempotent: running it twice gives the same file.  It

1. numbers the sections (``<h2 id=...><span class="secnum">``), subsections (``h3``) and,
   inside ``<section class="appendix">``, the appendices (A, B, ...);
2. regenerates the table of contents between ``<!--TOC-->`` and ``<!--/TOC-->``;
3. numbers figures (``<figure class="fig" id=...>``), tables (``<figure class="tab" ...>``)
   and statements (``<div class="thm" data-kind="Theorem" id=...>``, one shared counter),
   rewriting their ``<span class="fignum">``, ``tabnum`` and ``thmnum`` labels;
4. fills every cross-reference ``<a class="xref" href="#id">`` with its full label
   ("Figure 3", "Section 4.2") and every ``<a class="xnum" href="#id">`` with the number
   only;
5. numbers citations ``<span class="cite" data-keys="k1,k2">`` by first appearance,
   reorders the bibliography (``<li id="ref-key">`` between ``<!--REFS-->`` and
   ``<!--/REFS-->``) to match, and stops on a key with no entry;
6. inlines each ``out/arch/figures/<name>.svg`` between ``<!--FIG name-->`` and
   ``<!--/FIG name-->`` when the file exists.

    set PYTHONIOENCODING=utf-8
    python scripts/arch_paper_build.py
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(HERE, "proposal", "architecture", "index.html")
FIGS = os.path.join(HERE, "out", "arch", "figures")


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def main() -> None:
    with open(PAGE, encoding="utf-8") as fh:
        html = fh.read()
    labels: dict[str, str] = {}
    numbers: dict[str, str] = {}

    # 6. figures first, so that their ids are present for numbering
    def inline(m: re.Match) -> str:
        name = m.group(1)
        path = os.path.join(FIGS, name + ".svg")
        if not os.path.exists(path):
            return m.group(0)
        with open(path, encoding="utf-8") as fh:
            svg = fh.read().strip()
        return f"<!--FIG {name}-->\n{svg}\n<!--/FIG {name}-->"

    html = re.sub(r"<!--FIG ([\w-]+)-->.*?<!--/FIG \1-->", inline, html, flags=re.S)

    # 0. every numbered display equation sits in a horizontally scrolling block, so that a
    #    wide equation scrolls on a narrow screen instead of widening the page
    html = re.sub(r'(?<!<div class="eqwrap">)(\\begin\{equation\}.*?\\end\{equation\})',
                  r'<div class="eqwrap">\1</div>', html, flags=re.S)

    # 1. sections
    app_start = html.find('<section class="appendix"')
    toc = []
    sec = sub = app = 0
    out, pos = [], 0
    for m in re.finditer(r'<(h2|h3) id="([^"]+)"(?:><span class="secnum">[^<]*</span>| class="unnumbered">)'
                         r'(.*?)</\1>', html, re.S):
        tag, ident, title = m.group(1), m.group(2), m.group(3)
        in_app = app_start >= 0 and m.start() > app_start
        if "unnumbered" in m.group(0)[:80]:
            toc.append((tag, ident, "", strip_tags(title)))
            labels[ident] = strip_tags(title)
            numbers[ident] = ""
            continue
        if tag == "h2":
            sub = 0
            if in_app:
                app += 1
                num = chr(ord("A") + app - 1)
                labels[ident] = f"Appendix {num}"
            else:
                sec += 1
                num = str(sec)
                labels[ident] = f"Section {num}"
        else:
            sub += 1
            num = f"{chr(ord('A') + app - 1) if in_app else sec}.{sub}"
            labels[ident] = f"{'Appendix' if in_app else 'Section'} {num}"
        numbers[ident] = num
        toc.append((tag, ident, num, strip_tags(title)))
        out.append(html[pos:m.start()])
        out.append(f'<{tag} id="{ident}"><span class="secnum">{num}</span>{title}</{tag}>')
        pos = m.end()
    out.append(html[pos:])
    html = "".join(out)

    # 2. table of contents
    lines = ['<ol class="toc-list">']
    open_sub = False
    for tag, ident, num, title in toc:
        if tag == "h2":
            if open_sub:
                lines.append("</ol></li>")
                open_sub = False
            elif len(lines) > 1:
                lines.append("</li>")
            lines.append(f'<li><a href="#{ident}"><span class="tocnum">{num}</span>{title}</a>')
        else:
            if not open_sub:
                lines.append('<ol class="toc-sub">')
                open_sub = True
            lines.append(f'<li><a href="#{ident}"><span class="tocnum">{num}</span>{title}</a></li>')
    lines.append("</ol></li>" if open_sub else "</li>")
    lines.append("</ol>")
    html = re.sub(r"<!--TOC-->.*?<!--/TOC-->", lambda _: "<!--TOC-->\n" + "\n".join(lines) + "\n<!--/TOC-->",
                  html, flags=re.S)

    # 3. figures, tables, statements
    def number(kind_re: str, span: str, noun: str, html: str) -> str:
        count = 0
        out, pos = [], 0
        for m in re.finditer(kind_re, html, re.S):
            count += 1
            ident = m.group("id")
            labels[ident] = f"{noun} {count}"
            numbers[ident] = str(count)
            seg = html[m.end():]
            sm = re.search(rf'<span class="{span}">[^<]*</span>', seg)
            out.append(html[pos:m.end()])
            out.append(seg[:sm.start()])
            out.append(f'<span class="{span}">{noun} {count}:</span>')
            pos = m.end() + sm.end()
        out.append(html[pos:])
        return "".join(out)

    html = number(r'<figure class="fig" id="(?P<id>[^"]+)"[^>]*>', "fignum", "Figure", html)
    html = number(r'<figure class="tab" id="(?P<id>[^"]+)"[^>]*>', "tabnum", "Table", html)
    count = 0
    out, pos = [], 0
    for m in re.finditer(r'<div class="thm" data-kind="(?P<kind>\w+)" id="(?P<id>[^"]+)"[^>]*>\s*'
                         r'<span class="thmnum">[^<]*</span>', html):
        count += 1
        kind, ident = m.group("kind"), m.group("id")
        labels[ident] = f"{kind} {count}"
        numbers[ident] = str(count)
        out.append(html[pos:m.start()])
        out.append(m.group(0)[: m.group(0).rfind('<span class="thmnum">')] +
                   f'<span class="thmnum">{kind} {count}</span>')
        pos = m.end()
    out.append(html[pos:])
    html = "".join(out)

    # 4. cross-references
    missing = []

    def xref(m: re.Match) -> str:
        cls, ident = m.group(1), m.group(2)
        if ident not in labels:
            missing.append(ident)
            return m.group(0)
        text = labels[ident] if cls == "xref" else numbers[ident]
        return f'<a class="{cls}" href="#{ident}">{text}</a>'

    html = re.sub(r'<a class="(xref|xnum)" href="#([^"]+)">[^<]*</a>', xref, html)

    # 5. citations and bibliography
    refs_m = re.search(r"<!--REFS-->(.*?)<!--/REFS-->", html, re.S)
    entries = {m.group(1): m.group(2).strip() for m in
               re.finditer(r'<li id="ref-([^"]+)"[^>]*>(.*?)</li>', refs_m.group(1), re.S)}
    body = html[: refs_m.start()]
    order: list[str] = []
    for m in re.finditer(r'<span class="cite" data-keys="([^"]+)">', body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in order:
                order.append(k)
    unknown = [k for k in order if k not in entries]
    if unknown:
        sys.exit(f"citation keys with no bibliography entry: {unknown}")
    num = {k: i + 1 for i, k in enumerate(order)}

    def cite(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        inner = ", ".join(f'<a href="#ref-{k}">{num[k]}</a>' for k in keys)
        return f'<span class="cite" data-keys="{m.group(1)}">[{inner}]</span>'

    html = re.sub(r'<span class="cite" data-keys="([^"]+)">.*?</span>', cite, html, flags=re.S)
    refs_m = re.search(r"<!--REFS-->(.*?)<!--/REFS-->", html, re.S)   # offsets moved
    uncited = [k for k in entries if k not in num]
    items = []
    for k in order:
        text = re.sub(r'^<span class="refnum">[^<]*</span>\s*', "", entries[k])
        items.append(f'<li id="ref-{k}"><span class="refnum">[{num[k]}]</span> {text}</li>')
    for k in uncited:
        text = re.sub(r'^<span class="refnum">[^<]*</span>\s*', "", entries[k])
        items.append(f'<li id="ref-{k}" class="uncited"><span class="refnum">[?]</span> {text}</li>')
    html = (html[: refs_m.start()] + "<!--REFS-->\n" + "\n".join(items) + "\n<!--/REFS-->" +
            html[refs_m.end():])

    with open(PAGE, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    sys.stdout.write(f"sections {sec}, appendices {app}, statements {count}, "
                     f"citations {len(order)} of {len(entries)} entries\n")
    if missing:
        sys.stdout.write(f"UNRESOLVED cross-references: {sorted(set(missing))}\n")
    if uncited:
        sys.stdout.write(f"uncited bibliography entries: {uncited}\n")


if __name__ == "__main__":
    main()
