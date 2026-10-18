"""Stage 0 (W355): pack what the rented box needs, and nothing else.

    python scripts/chartop_box_payload.py [--plan out/chartop/plan.json]
        -> out/chartop/box/payload.tar.gz

``atlas/*.py`` (the composition layer, which ``atlas/__init__`` imports) and
``atlas/chartop/``; the data, training and pricing scripts; the runner
``chartop_box_run.sh`` (LF endings and mode 755, so bash reads it); and, when given,
the run plan.  No data and no weights: the box draws the data from the registered
seeds.  The sha256 is printed and checked on the box.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "chartop", "box", "payload.tar.gz")


def add_text(tar, src, arc, mode=0o644):
    data = open(src, "rb").read().replace(b"\r\n", b"\n")
    info = tarfile.TarInfo(arc)
    info.size, info.mode = len(data), mode
    tar.addfile(info, io.BytesIO(data))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--with-workbench", action="store_true",
                    help="add atlas/workbench (the benchmarks' geometry and materials) and the E5 script")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        for f in sorted(os.listdir(os.path.join(ROOT, "atlas"))):
            if f.endswith(".py"):
                add_text(tar, os.path.join(ROOT, "atlas", f), "chartop/atlas/" + f)
        for f in sorted(os.listdir(os.path.join(ROOT, "atlas", "chartop"))):
            if f.endswith(".py"):
                add_text(tar, os.path.join(ROOT, "atlas", "chartop", f), "chartop/atlas/chartop/" + f)
        if args.with_workbench:
            wb = os.path.join(ROOT, "atlas", "workbench")
            for f in sorted(os.listdir(wb)):
                if f.endswith(".py"):
                    add_text(tar, os.path.join(wb, f), "chartop/atlas/workbench/" + f)
            for f in sorted(os.listdir(os.path.join(wb, "families"))):
                if f.endswith(".py"):
                    add_text(tar, os.path.join(wb, "families", f), "chartop/atlas/workbench/families/" + f)
        for s in ("chartop_data.py", "chartop_train.py", "chartop_price.py", "chartop_e5.py"):
            add_text(tar, os.path.join(ROOT, "scripts", s), "chartop/scripts/" + s)
        add_text(tar, os.path.join(ROOT, "scripts", "chartop_box_run.sh"), "chartop/run.sh", mode=0o755)
        if args.plan:
            add_text(tar, args.plan, "chartop/plan.json")
    h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    names = tarfile.open(OUT).getnames()
    print("%s  %.2f MB  %d files  sha256 %s" % (OUT, os.path.getsize(OUT) / 1e6, len(names), h))
    for must in ("chartop/atlas/__init__.py", "chartop/atlas/chartop/train.py",
                 "chartop/scripts/chartop_train.py", "chartop/run.sh"):
        if must not in names:
            raise SystemExit("the payload is missing " + must)
    if any(n.endswith((".pt", ".npz", ".npy")) for n in names):
        raise SystemExit("the payload carries data or weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
