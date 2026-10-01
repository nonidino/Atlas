"""Demo step 8: write the learned case's registration, BEFORE any data exists.

    python scripts/learned_register.py

Writes ``out/learned-case/registered.txt``: the gate G1-G6 with its bars, the
arms, the measures, the expert's size from step 0, every layout of the split (the
held-out one, 100 for training, 10 for validation) and its hash, and the sha256 of
`atlas/workbench/learned_gate.py` as committed, line endings normalised.
`tests/test_workbench_learned.py` asserts the file and the module still agree, so a
bar changed after registration fails the suite rather than passing silently
("pre-registered criteria drift in code").

It refuses to overwrite an existing registration: re-registering is a decision,
made by deleting the file in a commit that says why.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "learned-case", "registered.txt")
MODULE = os.path.join(ROOT, "atlas", "workbench", "learned_gate.py")
STEP0 = "out/learned-case/step0-20261001-143036.json"


def module_hash(path: str = MODULE) -> str:
    """sha256 of the module's text with CRLF normalised to LF."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    from atlas.workbench import learned_gate as G
    if os.path.exists(OUT):
        raise SystemExit("%s exists; a registration is not overwritten" % OUT)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    lines = [
        "THE LEARNED CASE (demo item 1.5): REGISTRATION",
        "",
        "written            %s" % dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "before             any training data was generated",
        "repository at      %s (the commit adding this file follows it)" % head,
        "gate module        atlas/workbench/learned_gate.py  sha256 %s" % module_hash(),
        "registered as      %s" % G.REGISTERED,
        "",
        "case               %s, held out; horizons %s macro-steps; %d threads"
        % (G.CASE, ", ".join(str(h) for h in G.HORIZONS), G.THREADS),
        "expert             learned_net.WindowUNet width %d depth %d (step 0: %s)"
        % (G.NET_WIDTH, G.NET_DEPTH, STEP0),
        "",
        "THE GATE",
    ]
    for g in G.GATE:
        lines.append("  %s  %-15s %s" % (g.key, g.title, g.bar))
    lines += [
        "",
        "the measures and the arms are defined in the module's docstring; the judgement",
        "is learned_gate.judge(), evaluated once on the held-out case with the weights",
        "chosen on the validation layouts alone.",
        "",
        "THE SPLIT  (sha256 %s)" % G.split_hash(),
        "  random layouts: %d to %d rotors, x in %s D, y in %s D, spacing >= %.1f D, "
        "induction in [%.2f, %.4f]; refused if more than %d of the held-out rotors have a "
        "rotor within %.1f D" % (G.N_ROTORS[0], G.N_ROTORS[1], G.X_RANGE, G.Y_RANGE,
                                 G.MIN_SPACING, G.A_RANGE[0], G.A_RANGE[1], G.MAX_NEAR,
                                 G.NEAR),
        G.split_text(),
        "",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("registered ->", OUT)
    print("gate module sha256", module_hash())
    print("split sha256", G.split_hash())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
