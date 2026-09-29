"""The vault's link check: every ``[[wikilink]]`` must name a file in the vault.

Added 2026-09-29, from a lint that found 13 dead links in a vault whose standing
scan (`vault_scan.py`) reported 0 problems -- because that scan reads bytes and
never looks at links at all.  What it found: a renamed spec cited by its old
name, a page that never existed in git history, and two links to notes in the
assistant's working memory, which lives outside the vault.

A target resolves when some file under the root has that basename, with or
without ``.md`` -- which is how Obsidian resolves the bare links this vault uses
(CLAUDE.md: ``[[page-name]]``, never ``[[folder/page-name]]``).  The target is
the text before any ``|`` (alias; ``\\|`` inside a table) or ``#`` (heading).

Skipped, because Obsidian renders neither as a link: fenced code blocks, and
inline code spans -- which is how a page *quotes* a dead link while reporting
it (the gap worklist's W24 row does exactly that).  Math is NOT skipped: a
commutator written ``[[A,B]]`` inside ``$...$`` would be read as a link and
fail loudly, which is the safe direction.

`KNOWN` pins the dead links left on purpose, each with its count in its file
and the reason.  The check fails on a dead link that is not pinned, and on a
pin that no longer matches -- so fixing one forces the pin out, and the list
cannot quietly outlive what it excuses.

Run:  python scripts/link_scan.py [root]        (default: wiki)
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter

LINK = re.compile(r"\[\[([^\]]+)\]\]")
CODE_SPAN = re.compile(r"`[^`]*`")

#: (file relative to the root, with forward slashes; target) -> (count, reason).
KNOWN: dict[tuple[str, str], tuple[int, str]] = {
    ("concepts/Noether 1.1/Noether 1.1 implementation/00-implementation-plan.md",
     "phase1-resume-prompt"): (1, "W24, open: write the page or drop the references"),
    ("concepts/Noether 1.1/Noether 1.1 implementation/implementation-log.md",
     "phase1-resume-prompt"): (2, "W24, open: write the page or drop the references"),
    ("concepts/Noether 1.1/00-noether-1.1-overview.md",
     "pattern-recognizer-vs-solver"): (1, "a placeholder its own line calls 'link seeded'"),
    ("log.md", "spec-wind-farm-wake"): (1, "an old log entry; the log is append-only"),
    ("log.md", "positive-controls-need-a-horizon"): (1, "an old log entry; the log is append-only"),
}


def _say(text: str) -> None:
    # The console is cp1252; a page name is not guaranteed to be.
    print(text.encode("ascii", "backslashreplace").decode("ascii"))


def target_of(inner: str) -> str:
    """The page a link's inner text names: before any alias or heading."""
    t = inner.split("|")[0].split("#")[0].strip()
    return t[:-1].rstrip() if t.endswith("\\") else t      # `[[page\|alias]]` in a table


def names_under(root: str) -> set[str]:
    out: set[str] = set()
    for _dp, _dn, fns in os.walk(root):
        for fn in fns:
            out.add(fn)
            if fn.endswith(".md"):
                out.add(fn[:-3])
    return out


def links_in(text: str) -> list[tuple[int, str]]:
    """(line number, target) for every link outside code."""
    found: list[tuple[int, str]] = []
    fenced = False
    for i, line in enumerate(text.split("\n"), 1):
        st = line.lstrip()
        if st.startswith("```") or st.startswith("~~~"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for m in LINK.finditer(CODE_SPAN.sub("", line)):
            t = target_of(m.group(1))
            if t:
                found.append((i, t))
    return found


def unresolved(root: str) -> tuple[int, int, list[tuple[str, int, str]]]:
    """(files, links, dead) where dead is [(relpath, line, target)]."""
    names = names_under(root)
    n_files = n_links = 0
    dead: list[tuple[str, int, str]] = []
    for dp, _dn, fns in os.walk(root):
        for fn in sorted(fns):
            if not fn.endswith(".md"):
                continue
            path = os.path.join(dp, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, encoding="utf-8", errors="replace") as fh:
                links = links_in(fh.read())
            n_files += 1
            n_links += len(links)
            for line, t in links:
                name = t.rsplit("/", 1)[-1]
                if name not in names and name + ".md" not in names:
                    dead.append((rel, line, t))
    return n_files, n_links, sorted(dead)


def check(root: str, known=None):
    """(files, links, dead, new, stale).  `new`: dead links with no pin.
    `stale`: pins whose count no longer matches -- one more occurrence in a
    pinned file fails here, and so does a fixed one."""
    known = KNOWN if known is None else known
    n_files, n_links, dead = unresolved(root)
    counts = Counter((rel, t) for rel, _line, t in dead)
    new = [d for d in dead if (d[0], d[2]) not in known]
    stale = [(k, v[0], counts.get(k, 0)) for k, v in known.items() if counts.get(k, 0) != v[0]]
    return n_files, n_links, dead, new, stale


def main(argv) -> int:
    root = argv[1] if len(argv) > 1 else "wiki"
    n_files, n_links, dead, new, stale = check(root)
    for rel, line, t in dead:
        pin = KNOWN.get((rel, t))
        _say(f"{rel}:{line}: [[{t}]]  " + (f"known: {pin[1]}" if pin else "NEW dead link"))
    for (rel, t), want, got in stale:
        _say(f"STALE pin: {rel} [[{t}]] pinned at {want}, found {got}; update KNOWN")
    _say(f"{n_files} files, {n_links} links, {len(dead)} dead ({len(dead) - len(new)} known), "
         f"{len(new)} new, {len(stale)} stale")
    return 1 if (new or stale) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
