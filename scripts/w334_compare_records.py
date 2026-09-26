r"""W334 -- every Tier 76-85 gas-seam record, old against re-derived.

The old records (``out/pre_w332/*.json``, or ``out/*.json`` before the swap)
were taken with W301's saturated isothermal channel and W332's leaking wall;
the new ones (``out/w334/*.json``) with both fixed, on the rented box. This
flattens each pair to its leaves and reports, per record:

* how many numeric leaves moved, bucketed by relative size;
* the largest relative moves, with both values;
* every NON-numeric leaf that changed -- a sign structure, a prediction's
  held/failed, a boolean gate -- because those are where a conclusion flips.

Timing and provenance leaves (wall seconds, dates, machine, identity) are
listed separately: they are expected to move and say nothing about physics.

    python scripts/w334_compare_records.py [--old out/pre_w332] [--new out/w334]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

RECORDS = ("w300", "w301", "w302", "w303", "w304", "w305", "w306", "w310",
           "w312", "w312_underresolved", "w312_converged", "w312_gf", "w314")

#: leaves that are bookkeeping, not physics
NOISE = ("wall", "seconds", "date", "machine", "build_repo", "standby", "rate_",
         "_signature", "script", "elapsed", "per_call", "t_", "cost", "mhz",
         "charge", "battery", "setup_wall", "build_s", "calls_per")


def flatten(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, "%s.%s" % (path, k) if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, "%s[%d]" % (path, i))
    else:
        yield path, o


def is_noise(path):
    p = path.lower()
    return any(n in p for n in NOISE)


def rel(a, b):
    if a == b:
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    return abs(b - a) / max(abs(a), abs(b), 1e-300)


def compare(name, old, new, top):
    fo, fn = dict(flatten(old)), dict(flatten(new))
    num, lab, noise = [], [], []
    for k in sorted(set(fo) & set(fn)):
        a, b = fo[k], fn[k]
        if is_noise(k):
            if a != b:
                noise.append(k)
            continue
        if isinstance(a, bool) or isinstance(b, bool) or not (
                isinstance(a, (int, float)) and isinstance(b, (int, float))):
            if a != b:
                lab.append((k, a, b))
            continue
        num.append((rel(float(a), float(b)), k, a, b))
    only_old = sorted(k for k in set(fo) - set(fn) if not is_noise(k))
    only_new = sorted(k for k in set(fn) - set(fo) if not is_noise(k))
    buckets = [("identical", lambda r: r == 0.0), ("< 1e-9", lambda r: 0 < r < 1e-9),
               ("1e-9..1e-3", lambda r: 1e-9 <= r < 1e-3),
               ("1e-3..1e-1", lambda r: 1e-3 <= r < 1e-1), (">= 10%", lambda r: r >= 1e-1)]
    print("=" * 100)
    print("%s: %d numeric leaves; %s" % (name, len(num), ", ".join(
        "%s %d" % (lbl, sum(1 for r, *_ in num if f(r))) for lbl, f in buckets)))
    for r, k, a, b in sorted(num, reverse=True)[:top]:
        if r == 0.0:
            break
        print("   %8.2e  %-58s %14.7g -> %.7g" % (r, k[:58], a, b))
    for k, a, b in lab[:30]:
        print("   LABEL     %-58s %r -> %r" % (k[:58], a, b))
    if len(lab) > 30:
        print("   ... %d more label changes" % (len(lab) - 30))
    if only_old or only_new:
        print("   leaves only in old: %d, only in new: %d  %s" % (
            len(only_old), len(only_new), (only_old + only_new)[:4]))
    print("   bookkeeping leaves that moved (timing, provenance): %d" % len(noise))
    return dict(numeric=len(num), big=sum(1 for r, *_ in num if r >= 1e-1),
                labels=[(k, a, b) for k, a, b in lab])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="out/pre_w332")
    ap.add_argument("--new", default="out/w334")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    names = [n for n in RECORDS if not a.only or n in a.only.split(",")]
    summary = {}
    for n in names:
        po, pn = os.path.join(a.old, n + ".json"), os.path.join(a.new, n + ".json")
        if not (os.path.exists(po) and os.path.exists(pn)):
            print("%s: missing (%s %s)" % (n, os.path.exists(po), os.path.exists(pn)))
            continue
        with open(po, encoding="utf-8") as fh:
            old = json.load(fh)
        with open(pn, encoding="utf-8") as fh:
            new = json.load(fh)
        summary[n] = compare(n, old, new, a.top)
    print("=" * 100)
    for n, s in summary.items():
        print("  %-20s %5d numeric leaves, %4d moved >= 10%%, %3d label changes"
              % (n, s["numeric"], s["big"], len(s["labels"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
