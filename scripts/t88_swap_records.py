r"""Tier 88 -- put the re-derived records in place, keeping the ones they replace.

W334's procedure again (`scripts/w334_swap_records.py`), for the Tier 88 fixes
(W337-W340): the Tier 76-85 gas-seam records currently in `out/` (W334's
re-derivation, build repo adc470b) move to `out/pre_w337/`, and the records
re-derived on the fixed tree (`out/t88/`) take their names. The reverted twin
(`out/t88_revert/`, the attribution control) stays where it is.

Only the names passed are swapped: the ones the fixes moved, once
`scripts/w334_compare_records.py --old out --new out/t88_revert` has shown the
control reproduces the records they would replace. Refuses to run twice.

    python scripts/t88_swap_records.py --names w300,w301,...
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
OLD = os.path.join(OUT, "pre_w337")
NEW = os.path.join(OUT, "t88")
RECORDS = ("w300", "w301", "w302", "w303", "w304", "w305", "w306", "w310", "w312",
           "w312_underresolved", "w312_converged", "w312_gf", "w314")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--names", required=True, help="comma-separated record names to swap")
    a = ap.parse_args(argv)
    names = [n for n in a.names.split(",") if n]
    bad = [n for n in names if n not in RECORDS]
    if bad:
        print("not a Tier 76-85 gas-seam record: %s" % bad)
        return 1
    missing = [n for n in names if not os.path.exists(os.path.join(NEW, n + ".json"))]
    if missing:
        print("re-derived records missing from out/t88: %s" % missing)
        return 1
    os.makedirs(OLD, exist_ok=True)
    clash = [n for n in names if os.path.exists(os.path.join(OLD, n + ".json"))]
    if clash:
        print("out/pre_w337 already holds %s -- refusing to run twice" % clash)
        return 1
    for n in names:
        src = os.path.join(OUT, n + ".json")
        before = sha(src)
        shutil.copy2(src, os.path.join(OLD, n + ".json"))
        assert sha(os.path.join(OLD, n + ".json")) == before
        print("  kept   out/pre_w337/%s.json  (%s)" % (n, before))
    for n in names:
        src, dst = os.path.join(NEW, n + ".json"), os.path.join(OUT, n + ".json")
        shutil.copy2(src, dst)
        assert sha(dst) == sha(src)
        print("  placed out/%s.json  (%s)" % (n, sha(dst)))
    print("done: %d re-derived records in place, the old ones kept in out/pre_w337" % len(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
