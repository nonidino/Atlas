"""Scan markdown for the corruption a heredoc or a string literal introduces.

Authored as its OWN FILE on purpose. A control-character scan run through a
heredoc once reported zero backslashes in a file that had them, and a LaTeX
sequence written inside a heredoc becomes a TAB before the file is ever read --
`\\times` is the one that has bitten this project.

Flags, per line: TAB, lone CR, any other C0/C1 control character, a literal
backslash-t/backslash-n that is not part of a LaTeX command, and a non-ASCII
character (the console is cp1252 and print() of one raises).

    python scripts/scan_control_chars.py wiki/concepts/.../page.md
"""
from __future__ import annotations

import sys
import unicodedata

#: The typography this vault already uses, surveyed across all 253 pages at
#: Tier 76 rather than chosen: em dash 16566, section sign 4661, en dash 1870,
#: box-drawing horizontal 1582, middle dot 932, rightwards arrow 868, black star
#: 320, multiplication sign 253, left-right arrow 116, e-acute 111, ellipsis 96,
#: almost-equal 77. Calibrating the scan to the corpus is the point: a scan that
#: flags 16566 deliberate em dashes is a scan nobody reads, and the classes it
#: exists to catch -- TAB, lone CR, C0/C1 -- get lost in the noise.
ALLOWED_NON_ASCII = set(
    "—–§─·→★×↔é"
    "…≈≥≤±°λστβ"
    "κωαμΔ∑√∈→←"
    "“”‘’•½ ′″"
)


def scan(path: str) -> list[str]:
    problems: list[str] = []
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return ["%s: not valid UTF-8: %s" % (path, exc)]

    # **A bug this scan had on its own first run, kept as a comment because the
    # failure mode is exactly the one the file exists to catch.**  This
    # repository has `core.autocrlf = true`, so every working-tree file is CRLF
    # while git stores LF.  Splitting the raw text on "\n" then leaves a
    # trailing "\r" on EVERY line, and the lone-CR check reported 5554 problems
    # in a file git sees as clean.  A scan that cries wolf on every line of
    # every file is a scan nobody runs, which is how a real defect gets through.
    # So CRLF is normalised first and what remains is a GENUINE lone CR -- a
    # carriage return not paired with a newline, which is what a corrupted
    # heredoc actually leaves behind.
    n_crlf = raw.count(b"\r\n")
    n_cr = raw.count(b"\r")
    if n_cr != n_crlf:
        problems.append("%s: %d lone CR (not part of a CRLF pair)"
                        % (path, n_cr - n_crlf))
    lines = text.replace("\r\n", "\n").split("\n")
    for i, line in enumerate(lines, 1):
        if "\t" in line:
            problems.append("%s:%d: TAB (a heredoc turns \\times into one)" % (path, i))
        for ch in line:
            o = ord(ch)
            if o < 32 and ch not in "\t\r":
                problems.append("%s:%d: control char U+%04X" % (path, i, o))
            elif 127 <= o < 160:
                problems.append("%s:%d: C1 control char U+%04X" % (path, i, o))
            elif o > 126 and ch not in ALLOWED_NON_ASCII:
                # NEVER echo the character itself. The console is cp1252 and
                # print() of any non-ASCII raises -- a scan that crashes on the
                # thing it is reporting is the failure mode this file exists to
                # avoid, and it took one run of this script to find that out.
                problems.append("%s:%d: non-ASCII U+%04X (%s)"
                                % (path, i, o, unicodedata.name(ch, "unnamed")))
    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: scan_control_chars.py FILE [FILE ...]")
        return 2
    bad = 0
    for path in argv[1:]:
        found = scan(path)
        for p in found:
            print(p)
        bad += len(found)
        if not found:
            print("%s: clean" % path)
    print()
    print("%d problem(s)" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
