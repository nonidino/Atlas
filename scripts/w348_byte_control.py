"""W348's byte control: W189's capture of every `atlas/cases` graph, compiled by
HEAD's compiler and by the working tree's, compared byte for byte.

    python scripts/w348_byte_control.py head      # capability.py and compiler.py as at HEAD
    python scripts/w348_byte_control.py after     # the working tree
    python scripts/w348_byte_control.py compare

W348 moves R10's case for a cut piece that declares ``elliptic_data_from_ports``.
No case graph declares it, so every one of W189's artifacts must be identical.
"HEAD" is served without touching the working tree (other chats' uncommitted
work lives in it): HEAD's two files are written under the output folder and a
meta-path finder hands them to the importer in place of the tree's.  Run both
with the same ``PYTHONHASHSEED`` (`verdict._jsonable` lists sets in hash order).
Output: the artifacts and HEAD's sources in ``out/w348/w189/`` (ignored, 10 MB); the
two captures' digests and the comparison copied to ``out/workbench/records/w348/w189/``.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "w348", "w189")
RECORDS = os.path.join(ROOT, "out", "workbench", "records", "w348", "w189")
SWAPPED = ("capability", "compiler")


class _HeadFinder(importlib.abc.MetaPathFinder):
    def __init__(self, folder: str):
        self.folder = folder

    def find_spec(self, name, path, target=None):
        if name.startswith("atlas.") and name.split(".", 1)[1] in SWAPPED:
            return importlib.util.spec_from_file_location(
                name, os.path.join(self.folder, name.split(".", 1)[1] + ".py"))
        return None


def _head_sources() -> str:
    folder = os.path.join(OUT, "head_src")
    os.makedirs(folder, exist_ok=True)
    for mod in SWAPPED:
        blob = subprocess.run(["git", "show", f"HEAD:atlas/{mod}.py"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        with open(os.path.join(folder, mod + ".py"), "wb") as fh:
            fh.write(blob)
    return folder


def _w189():
    spec = importlib.util.spec_from_file_location(
        "w189_artifact_control", os.path.join(ROOT, "scripts", "w189_artifact_control.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT = OUT
    return mod


def main(argv) -> int:
    os.makedirs(OUT, exist_ok=True)
    cmd = argv[0] if argv else ""
    if cmd == "head":
        sys.meta_path.insert(0, _HeadFinder(_head_sources()))
    if cmd in ("head", "after"):
        w = _w189()
        import atlas.compiler as C
        where = os.path.relpath(C.__file__, ROOT)
        has = hasattr(C, "_r10_scheme")
        print(f"atlas.compiler from {where}; _r10_scheme present: {has}", flush=True)
        assert has == (cmd == "after"), "the wrong compiler was imported"
        res = w.capture(f"w348_{cmd}")
        errs = {k: r["error"] for k, r in res["rows"].items() if "error" in r}
        print(f"captured {sum(1 for r in res['rows'].values() if 'sha256' in r)}, "
              f"errors {len(errs)}, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")
        for k, e in errs.items():
            print(f"  ERROR {k}: {e[:200]}")
        return 0
    if cmd == "compare":
        import shutil
        out = _w189().compare("w348_head", "w348_after")
        os.makedirs(RECORDS, exist_ok=True)
        for f in ("control_w348_head.json", "control_w348_after.json",
                  "compare_w348_head_vs_w348_after.json"):
            shutil.copy2(os.path.join(OUT, f), os.path.join(RECORDS, f))
        print(f"identical {len(out['identical'])}, differ {len(out['differ'])}, "
              f"not compared {len(out['not_compared'])}")
        for k in out["differ"]:
            print("  DIFFER", k)
        for k in out["not_compared"]:
            print("  NOT COMPARED", k)
        return 0 if not out["differ"] and not out["not_compared"] else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
