r"""W334 -- put the re-derived records in place, keeping the old ones.

The Tier 76-85 gas-seam records (`out/w300` ... `out/w314`) were taken with
W301's saturated isothermal channel and W332's leaking wall. The re-derived
records are in `out/w334/`. This moves every old record to `out/pre_w332/`
(kept, not deleted: the wiki's history quotes them and the controls compare
against them) and copies each re-derived one to the canonical name.

Partial records that a full record supersedes (`w300_part1`, `w300_part2`,
`w304_lam`) move to `out/pre_w332/` with no successor.

Refuses to run twice: if `out/pre_w332/` already holds a record, it stops.

    python scripts/w334_swap_records.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
OLD = os.path.join(OUT, "pre_w332")
NEW = os.path.join(OUT, "w334")
REDERIVED = ("w300", "w301", "w302", "w303", "w304", "w305", "w306", "w310", "w312",
             "w312_underresolved", "w312_converged", "w312_gf", "w314")
SUPERSEDED = ("w300_part1", "w300_part2", "w304_lam")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def main():
    missing = [n for n in REDERIVED if not os.path.exists(os.path.join(NEW, n + ".json"))]
    if missing:
        print("re-derived records missing from out/w334: %s" % missing)
        return 1
    os.makedirs(OLD, exist_ok=True)
    clash = [n for n in REDERIVED + SUPERSEDED if os.path.exists(os.path.join(OLD, n + ".json"))]
    if clash:
        print("out/pre_w332 already holds %s -- refusing to run twice" % clash)
        return 1
    for n in REDERIVED + SUPERSEDED:
        src = os.path.join(OUT, n + ".json")
        if os.path.exists(src):
            before = sha(src)
            shutil.copy2(src, os.path.join(OLD, n + ".json"))
            assert sha(os.path.join(OLD, n + ".json")) == before
            if n in SUPERSEDED:
                os.remove(src)
            print("  kept   out/pre_w332/%s.json  (%s)" % (n, before))
    for n in REDERIVED:
        src = os.path.join(NEW, n + ".json")
        dst = os.path.join(OUT, n + ".json")
        shutil.copy2(src, dst)
        assert sha(dst) == sha(src)
        print("  placed out/%s.json  (%s)" % (n, sha(dst)))
    print("done: %d re-derived records in place, %d old ones kept"
          % (len(REDERIVED), len(os.listdir(OLD))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
