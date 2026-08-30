r"""W38's byte-level scan: control characters, split tables, orphan continuations.

Text mode cannot see a lone \r -- Python's universal-newline translation turns it
into \n on read -- and \r is exactly what the escape-corruption class produces
(``\rangle``, ``\rho``, ``\rVert``).  So this reads BYTES.

**W72, 2026-08-29: reading bytes was necessary and not sufficient.**  A tab is a
legal markdown character, so ``\t`` -- ``\tau``, ``\theta``, ``\times``,
``\top``, ``\tilde``, ``\text`` -- corrupts into something a control-character
sweep is *right* not to flag.  It was found the way W38 was: by one edit
producing both signatures at once, ``\beta`` into 0x08 (caught) and ``\tau``
into a tab (not).

**W72 again, 2026-08-30: the tab was an instance and the instance is not the
class.**  Adding one byte to a hand-written list leaves a hand-written list,
which is what failed: W38 closed on 2026-08-27 naming ``\t`` among the bytes to
scan for, and this file, written the next day, implemented every byte in that
list except that one.  A rule with several call sites and one that does not
consult it -- in the tooling rather than in the theory.  Three changes make the
list stop being hand-written:

  1. `ESCAPES` is **derived from Python's own escape decoder**, not typed out.
     Every single-character escape Python interprets is discovered by decoding
     it, so the table cannot drift from what actually happens to a string
     literal.
  2. Two of those bytes are legal markdown and cannot be flagged on sight: 0x0A
     from ``\n`` (``\nu``, ``\nabla``, ``\neq``) and 0x09 from ``\t``.  The tab
     is flagged because this vault uses none.  The newline cannot be, so it is
     caught by its **symptom** instead -- see `odd_dollars`, which needs no list
     at all and catches every byte in the table at once.
  3. `tests/test_tier14_locality_and_scope.py` generates a positive control for
     **every** entry in `ESCAPES` and asserts this scanner flags it.  A byte
     added to the table with no detector is a failing test, not a silent hole.

`\u` and `\x` are in the table with no TeX word attached on purpose: ``\upsilon``
and ``\xi`` in a Python literal raise a SyntaxError rather than corrupting, so
they fail loudly and need no detector here.

Run:  python scripts/vault_scan.py [paths...]
"""

from __future__ import annotations

import os
import re
import sys


def _derive_escapes() -> dict[str, int]:
    """What each single-character escape actually decodes to, asked not assumed.

    Kept as a derivation rather than a literal table so the scanner's idea of the
    escape set cannot drift from Python's.  The TeX words are the reason each
    entry matters in a physics vault and are listed beside it in `TEX_WORDS`.
    """
    out: dict[str, int] = {}
    for ch in "abfnrtv0":
        try:
            decoded = ("\\" + ch).encode().decode("unicode_escape")
        except (UnicodeDecodeError, SyntaxError):                     # pragma: no cover
            continue
        if len(decoded) == 1 and decoded != ch:
            out[ch] = ord(decoded)
    return out


#: escape letter -> the byte a string literal turns it into.
ESCAPES = _derive_escapes()

#: escape letter -> the TeX control words that begin with it.  The tails are what
#: survives the corruption and what the orphan-continuation check looks for.
TEX_WORDS = {
    "a": ("alpha", "approx", "angle", "ast", "arg"),
    "b": ("beta", "bar", "big", "boldsymbol", "bmatrix"),
    "f": ("frac", "forall", "frown", "flat"),
    "n": ("nu", "nabla", "neq", "notin", "nonumber"),
    "r": ("rho", "rangle", "right", "rVert", "rbrace"),
    "t": ("tau", "theta", "times", "top", "tilde", "text"),
    "v": ("varepsilon", "vec", "varphi", "vert"),
    "0": (),
}

#: Bytes that are legal markdown and so cannot be flagged on sight.
LEGAL = {0x0A, 0x0D}

#: Control words that survive a corruption WHOLE, because the escape ate a
#: preceding backslash and not this one.  W38's original list, unchanged: these
#: are matched with the backslash on, so they carry no false-positive risk.
ORPHAN_WORDS = ("\\rangle", "\\rho", "\\rVert", "\\rbrace", "\\right")

#: Tails, matched WITHOUT a backslash, so every entry has to be a string no
#: English or code line would begin with.  `nu` -> "u" is the standing reason
#: this list is curated rather than generated: W38's repair script derived tails
#: mechanically, "u" matched seven legitimate line beginnings (`u_t = ...`,
#: `uv pip install`, `unchoked operation`), and the whole edit had to be
#: reverted.  Generating from TEX_WORDS would also admit "ext" (from `\text`),
#: which matches `external_flow.py` in a directory listing three files away from
#: here.  So the generated set is FILTERED, the filter is a literal allow-list,
#: and `odd_dollars` carries the general case that no list can.
ORPHAN_TAILS = (
    "abla", "arepsilon", "arphi", "heta", "ilde", "lpha", "ngle", "oldsymbol",
    "onumber", "orall", "otin", "pprox", "rown", "vert", "Vert",
)


def odd_dollars(lines: list[str]) -> list[int]:
    r"""Lines with an unpaired ``$``, outside display math and fenced code.

    **The general detector for this whole class, W72, 2026-08-30.**  Every escape
    corruption inside inline math does the same thing regardless of which byte it
    produced: it splits ``$\nu$`` into a line ending ``$`` and a line beginning
    ``u$``, leaving each with exactly one delimiter.  So counting ``$`` per line
    catches ``\n`` -- the one byte in the table that is legal markdown, the one
    no byte-level rule can see, and the most common TeX prefix in a physics vault
    after ``\r`` -- without needing to know which byte was produced, and it
    catches the rest as a free consequence.

    Display blocks (``$$``) and fenced code are skipped: both legitimately carry
    unpaired delimiters.
    """
    bad, in_fence, in_display = [], False, False
    for n, line in enumerate(lines, 1):
        st = line.strip()
        if st.startswith("```") or st.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if st.count("$$") == 1:
            in_display = not in_display
            continue
        if in_display or st.startswith("$$"):
            continue
        # A code span is content, not math: `log.md` quotes the fragment
        # `$\langle U_d` while REPORTING an old corruption, and an unpaired
        # delimiter inside backticks is exactly what such a report looks like.
        stripped = re.sub(r"`[^`]*`", "", line)
        # `\$` is an escaped dollar and is content, not a delimiter
        if (stripped.replace(r"\$", "").count("$") % 2) == 1:
            bad.append(n)
    return bad


def split_table_rows(lines: list[str]) -> list[int]:
    r"""Lines opening with ``|`` that do not close with one.  **W72, 2026-08-30.**

    The ragged-table check needs three contiguous pipe-delimited lines before it
    will say anything, and a corruption that splits a row **destroys the
    contiguity it is counting**.  That is how ``$\lvert\mathcal R(t)\rvert$`` in
    the wind-farm spec's W11 row survived: the ``\r`` became a plain LF rather
    than a lone CR, so no byte-level rule could see it; the split left a two-line
    fragment below the ragged check's minimum; and the tail it left was ``vert``,
    which was not on any list.  It had been in the vault since 2026-08-26,
    passing every scan, and the ``$``-parity check is what found it.

    A row that opens a cell and never closes it is unambiguous on its own and
    needs no minimum block size.
    """
    bad, in_fence = [], False
    for n, line in enumerate(lines, 1):
        st = line.strip()
        if st.startswith("```") or st.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence and st.startswith("|") and not st.endswith("|"):
            bad.append(n)
    return bad


def scan(path: str) -> list[str]:
    out = []
    raw = open(path, "rb").read()
    for i, b in enumerate(raw):
        if b < 9 or (13 <= b <= 31 and b != 13) or b == 11 or b == 12:
            out.append(f"{path}: byte {i}: control char 0x{b:02x}")
    if b"\r" in raw.replace(b"\r\n", b""):
        out.append(f"{path}: a lone CR -- the escape-corruption signature")
    if b"\t" in raw:
        n = raw.count(b"\t")
        out.append(f"{path}: {n} TAB byte(s) -- the escape-corruption signature "
                   "for \\tau, \\theta, \\times, \\top, \\tilde, \\text")

    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    block: list[tuple[int, int]] = []
    for n, line in enumerate(lines, 1):
        # an escaped pipe is cell CONTENT, not a column separator -- counting it
        # reports every table carrying `\|` or a LaTeX norm as ragged
        st = line.strip().replace(r"\|", "")
        if st.startswith("|") and st.endswith("|"):
            block.append((n, st.count("|")))
        else:
            if len(block) >= 3:
                w = [c for _, c in block]
                if len(set(w)) > 1:
                    bad = [f"line {ln} has {c} pipes" for ln, c in block
                           if c != max(set(w), key=w.count)]
                    out.append(f"{path}: ragged table near line {block[0][0]}: "
                               + "; ".join(bad[:4]))
            block = []
    # The orphan check must skip fences for the same reason `odd_dollars` does,
    # and the reason is not hypothetical: section 14.7 QUOTES the corrupted
    # wind-farm row it is about, inside a fence, and this check flagged the
    # quotation. A page reporting a corruption looks exactly like a corrupted
    # page unless the reporter can tell content from prose.
    in_fence = False
    for n, line in enumerate(lines, 1):
        st = line.strip()
        if st.startswith("```") or st.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if st.startswith(ORPHAN_WORDS) or st.startswith(ORPHAN_TAILS):
            out.append(f"{path}: line {n}: orphan continuation starting {st[:24]!r}")
    for n in split_table_rows(lines):
        out.append(f"{path}: line {n}: table row opens a cell and never closes it "
                   "-- a split row, which is what a corrupted control word inside "
                   "a table leaves behind")
    for n in odd_dollars(lines):
        out.append(f"{path}: line {n}: unpaired '$' -- inline math split across a "
                   "line, which is what every escape corruption looks like once "
                   "the byte it produced is a legal one")
    return out


def main(argv):
    roots = argv[1:] or ["wiki"]
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        else:
            for dp, _dn, fn in os.walk(r):
                files += [os.path.join(dp, f) for f in fn if f.endswith(".md")]
    problems = []
    for f in sorted(files):
        problems += scan(f)
    for p in problems:
        print(p)
    print(f"{len(files)} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
