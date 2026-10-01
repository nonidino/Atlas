"""Demo step 8: pack what the rented box needs, and nothing else.

    python scripts/learned_box_payload.py        # -> out/learned-case/data/box-payload.tar.gz

  * ``atlas/``, as the workbench bundle carries it: without the proofs of
    concept's demo packages and without the launcher templates or caches;
  * ``vendor/src/atlas/cases/windfarm/`` and the build repository's MIT licence
    (`window_ns.load_reference` looks there under ``ATLAS_BUILD_REPO``);
  * ``scripts/learned_data.py``, ``scripts/learned_train.py``, the runner
    ``learned_box_run.sh`` (LF line endings, so bash reads it) and the install's
    ``constraints.txt``, which pins numpy, scipy and pydantic as the workbench was
    verified.

No weights, no data: the box generates the data from the registered seeds.  The
upload from this connection is the slow leg, so the payload is a single small
tarball; its sha256 is printed and checked on the box.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.environ.get("ATLAS_BUILD_REPO", os.path.join(os.path.expanduser("~"),
                                                        "physics-foundation-model"))
OUT = os.path.join(ROOT, "out", "learned-case", "data", "box-payload.tar.gz")
SKIP_DIRS = {"__pycache__", "bundle", "demo", "demo_frontwing", "demo_racelab",
             ".pytest_cache"}


def add_tree(tar, src, arc):
    for dp, dn, fn in os.walk(src):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        for f in sorted(fn):
            if f.endswith(".pyc"):
                continue
            full = os.path.join(dp, f)
            tar.add(full, arcname=os.path.join(arc, os.path.relpath(full, src)).replace(
                os.sep, "/"))


def add_text(tar, src, arc, mode=0o644):
    data = open(src, "rb").read().replace(b"\r\n", b"\n")
    info = tarfile.TarInfo(arc)
    info.size, info.mode = len(data), mode
    tar.addfile(info, io.BytesIO(data))


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        add_tree(tar, os.path.join(ROOT, "atlas"), "learned/atlas")
        add_tree(tar, os.path.join(BUILD, "src", "atlas", "cases", "windfarm"),
                 "learned/vendor/src/atlas/cases/windfarm")
        add_text(tar, os.path.join(BUILD, "LICENSE"), "learned/vendor/LICENSE")
        for s in ("learned_data.py", "learned_train.py"):
            add_text(tar, os.path.join(ROOT, "scripts", s), "learned/scripts/" + s)
        add_text(tar, os.path.join(ROOT, "scripts", "learned_box_run.sh"),
                 "learned/learned_box_run.sh", mode=0o755)
        add_text(tar, os.path.join(ROOT, "atlas", "workbench", "bundle", "constraints.txt"),
                 "learned/constraints.txt")
    h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
    names = tarfile.open(OUT).getnames()
    print("%s  %.2f MB  %d files  sha256 %s" % (OUT, os.path.getsize(OUT) / 1e6, len(names), h))
    for must in ("learned/atlas/workbench/learned_arms.py",
                 "learned/vendor/src/atlas/cases/windfarm/reference.py",
                 "learned/scripts/learned_train.py", "learned/learned_box_run.sh"):
        if must not in names:
            raise SystemExit("the payload is missing " + must)
    if any("/demo_racelab/" in n or n.endswith(".pt") or n.endswith(".npy") for n in names):
        raise SystemExit("the payload carries something it must not")
    return 0


if __name__ == "__main__":
    sys.exit(main())
