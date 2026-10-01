"""Ask the Lean kernel what each theorem of the project rests on.

For every declaration the blueprint names (``lean/blueprint/lean_decls``, written
by plasTeX from the ``\\lean{...}`` marks; or ``--names FILE``), this runs
``#print axioms`` and reads the answer.  A declaration PASSES when the axioms
it depends on are among Lean's three standard ones,

    propext, Classical.choice, Quot.sound,

and FAILS on anything else: ``sorryAx`` (a hole left in a proof), a new axiom,
or a name Lean does not know.  The check is the kernel's own, so it does not
depend on how a build log words its warnings.

Run a normal build first (``python scripts/lean_build.py``): the check reads
the built project.  Writes ``out/lean/axioms.json`` (``axioms-<tag>.json`` when
ATLAS_LEAN_TAG is set, see lean_build.py).

    python scripts/lean_axioms.py
    python scripts/lean_axioms.py --names lean/decls-batches-1-3.txt
    python scripts/lean_axioms.py --expect-sorry   # the positive control: at
        least one declaration MUST fail on sorryAx, or the instrument is blind

A theorem that USES another theorem whose proof is still a `sorry` shows
`sorryAx` too.  So a chat that proves one batch while another batch is still
open checks its own names with ``--names``, and the full check is run once the
branches are merged.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("lean_build",
                                               os.path.join(REPO, "scripts", "lean_build.py"))
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)

STANDARD = {"propext", "Classical.choice", "Quot.sound"}
DECLS = os.path.join(REPO, "lean", "blueprint", "lean_decls")


def read_names(path: str) -> list[str]:
    names = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def parse(text: str, names: list[str]) -> dict:
    """Map each name to its axiom list, or to None when Lean printed no answer."""
    found: dict[str, list[str]] = {}
    for m in re.finditer(r"'([^'\s]+)' depends on axioms:\s*\[([^\]]*)\]", text, re.S):
        found[m.group(1)] = [a.strip() for a in m.group(2).replace("\n", " ").split(",")
                             if a.strip()]
    for m in re.finditer(r"'([^'\s]+)' does not depend on any axioms", text):
        found[m.group(1)] = []
    return {n: found.get(n) for n in names}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--names", default=DECLS, help="a file with one declaration per line")
    ap.add_argument("--expect-sorry", action="store_true",
                    help="positive control: succeed only if some declaration fails on sorryAx")
    ns = ap.parse_args(argv[1:])

    names = read_names(ns.names)
    build = lb.BUILD
    work = os.path.join(build, ".lake", "axiom_check")
    os.makedirs(work, exist_ok=True)
    src = os.path.join(work, "AxiomCheck.lean")
    with open(src, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("import AtlasProofs\n\n")
        for n in names:
            fh.write("#print axioms %s\n" % n)

    t0 = time.perf_counter()
    proc = subprocess.run([lb._lake(), "env", "lean", src], cwd=build, env=lb._env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - t0
    text = proc.stdout.decode("utf-8", errors="replace")
    answers = parse(text, names)

    rows, bad, sorry = [], 0, 0
    for n in names:
        ax = answers[n]
        if ax is None:
            verdict = "UNKNOWN to Lean (no answer printed)"
            bad += 1
        elif "sorryAx" in ax:
            verdict = "FAIL: sorry"
            bad += 1
            sorry += 1
        elif set(ax) - STANDARD:
            verdict = "FAIL: extra axioms " + ", ".join(sorted(set(ax) - STANDARD))
            bad += 1
        else:
            verdict = "ok"
        rows.append({"name": n, "axioms": ax, "verdict": verdict})

    out = {
        "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "declarations": len(names), "failing": bad, "with_sorry": sorry,
        "seconds": round(seconds, 2), "lean_exit": proc.returncode, "rows": rows,
    }
    out.update(lb._pins())
    os.makedirs(lb.OUT, exist_ok=True)
    with open(lb.AXIOMS, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")

    width = max(len(n) for n in names) if names else 10
    for r in rows:
        ax = "-" if r["axioms"] is None else ("none" if not r["axioms"] else ", ".join(r["axioms"]))
        lb._say("%-*s  %-12s %s" % (width, r["name"], r["verdict"][:12], ax))
    lb._say("")
    lb._say("[lean_axioms] %d declarations, %d failing (%d on sorry), %.1f s, lean exit %d"
            % (len(names), bad, sorry, seconds, proc.returncode))
    if proc.returncode != 0:
        errs = [ln for ln in text.splitlines() if "error" in ln][:10]
        for ln in errs:
            lb._say("    " + ln)
    if ns.expect_sorry:
        ok = sorry > 0
        lb._say("[lean_axioms] positive control (a sorry must be SEEN): %s"
                % ("seen" if ok else "NOT SEEN - the instrument is blind"))
        return 0 if ok else 1
    return 1 if (bad or proc.returncode != 0) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
