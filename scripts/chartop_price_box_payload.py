"""Pack what a rented box needs for the squares price protocol (`gates_squares.REVISED`), from
one commit, and nothing else.

    python scripts/chartop_price_box_payload.py [--commit HEAD]
        -> out/chartop/box/price-squares-<commit>.tar.gz

From ``git archive`` of the commit, so the box runs exactly what was committed, whatever the
working tree holds: ``atlas/``, ``scripts/`` and ``vendor/``, and the ``out/chartop`` records
the price script reads (`RECORDS`).  Beside them the runner ``chartop_price_box_run.sh`` as
``run.sh`` (LF endings and mode 755, so bash reads it) and ``IDENTITY``, the commit.  No data,
no weights, and no record of the registered layouts: before the one evaluation only the
classical solver reads those.  The sha256 is printed; check it on the box after the upload.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "out", "chartop", "box")
TREES = ("atlas", "scripts", "vendor")
#: the out/ records a price run reads: none.  An audit hook on every file a price run opened,
#: over section 11.17's twelve problems (2026-11-08), found 59 repo files, all in `TREES`.
RECORDS: tuple = ()
RUNNER = "scripts/chartop_price_box_run.sh"
#: never shipped: weights, data, and the registered layouts' record
REFUSED_SUFFIXES = (".pt", ".pth", ".npz", ".npy", ".ckpt")
REFUSED_NAMES = ("registration-squares.json",)
TOP = "chartop"


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", ROOT, "-c", "core.autocrlf=false", *args], check=True,
                          capture_output=True).stdout


def build(commit: str = "HEAD", out_dir: str = OUT_DIR) -> dict:
    full = git("rev-parse", "--verify", commit + "^{commit}").decode().strip()
    raw = git("archive", "--format=tar", f"--prefix={TOP}/", full, *TREES, *RECORDS)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"price-squares-{full[:7]}.tar.gz")
    tmp = out + ".tmp"
    with tarfile.open(fileobj=io.BytesIO(raw)) as src, tarfile.open(tmp, "w:gz") as tar:
        for m in src.getmembers():
            tar.addfile(m, src.extractfile(m) if m.isfile() else None)
        runner = git("show", f"{full}:{RUNNER}").replace(b"\r\n", b"\n")
        for name, data, mode in ((f"{TOP}/run.sh", runner, 0o755), (f"{TOP}/IDENTITY", (full + "\n").encode(), 0o644)):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    os.replace(tmp, out)
    names = tarfile.open(out).getnames()
    for must in (f"{TOP}/atlas/__init__.py", f"{TOP}/atlas/chartop/gates_squares.py",
                 f"{TOP}/scripts/chartop_price_squares.py", f"{TOP}/scripts/chartop_cpus.py",
                 f"{TOP}/run.sh", f"{TOP}/IDENTITY"):
        if must not in names:
            raise SystemExit("the payload is missing " + must)
    bad = [n for n in names if n.endswith(REFUSED_SUFFIXES) or os.path.basename(n) in REFUSED_NAMES]
    if bad:
        raise SystemExit(f"the payload carries what it must not: {bad[:5]}")
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    return {"path": out, "commit": full, "files": len(names), "mb": os.path.getsize(out) / 1e6, "sha256": sha}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", default="HEAD")
    args = ap.parse_args(argv)
    r = build(args.commit)
    print(f"{os.path.relpath(r['path'], ROOT)}  commit {r['commit'][:7]}  {r['files']} files  "
          f"{r['mb']:.2f} MB  sha256 {r['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
