"""PoC 3 phase 5 -- assemble the self-contained `poc3-racelab-demo` bundle.

    python scripts/build_racelab_bundle.py                  # into out/bundle3
    python scripts/build_racelab_bundle.py --out C:/tmp/poc3
    python scripts/build_racelab_bundle.py --commit         # ... and refresh the
                                                            # orphan branch locally

In the shape of `w146_build_frontwing_bundle.py` (PoC 2) and
`w112_build_bundle.py` (PoC 1), which it combines: PoC 2's whole build-repo
solver package, because the car's block and wing load the structural and
conduction solvers; PoC 1's checkpoint in the hub's own cache layout, because
the car's windows can be switched to Poseidon-T.

The bundle is a COPY of what normally lives in four places -- this framework,
the build repository's solvers, a Hugging Face cache, and this repository's
recorded runs -- laid out so that each resolves **with no change to any source
file**:

  * the solvers at ``vendor/src/atlas/``, because `window_ns.load_reference`
    looks for ``$ATLAS_BUILD_REPO/src/atlas/cases/windfarm`` and
    `cooling_loop.load_solvers` binds ``$ATLAS_BUILD_REPO/src/atlas`` whole
    (``grid.py``'s ``..config`` imports have to resolve);
  * the checkpoint at ``vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/``,
    because the adapter asks the hub for a repository id and the offline answer
    to that is the hub's own cache;
  * the settled field at ``out/racelab5/cache/settled.npz``, where
    `demo_racelab.engine` looks, **refused if it belongs to a different car**;
  * the case-study pages at ``wiki/concepts/Atlas 0.1/common/``, where the tests
    that quote them look -- so those tests RUN in the bundle instead of skipping
    as they did in PoC 1's and PoC 2's, whose pages were copied flat into
    ``docs/``;
  * everything else where it already is.

That is what lets the tests in the bundle be the same tests as the ones here,
character for character -- which is the only reason passing them there means
anything.  `scripts/verify_racelab_bundle.py` checks it on a fresh clone.

**What must NOT be in it, and this script refuses to finish if it is:**
the unlicensed structural checkpoint this project keeps outside the repository
(no file of it, no path into its cache, no import of it -- its name in the
prose that explains why it is absent is reported, not refused, which is the
rule `test_tier51_racelab_graph.py` already applies); `scOT`, which has no
licence to redistribute and is installed by the launchers instead; any weight
file other than Poseidon-T's; and any three-dimensional field -- the bundle
ships the 3-D generator (`racelab3d.py`), not data from it.

**This script does not push.**  It builds a directory, and with ``--commit`` it
refreshes the local orphan branch inside that directory.  Pushing is a
deliberate, separate act, because the branch is force-replaced wholesale every
time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BRANCH = "poc3-racelab-demo"
TEMPLATES = os.path.join(ROOT, "atlas", "demo_racelab", "bundle")

#: The case-study pages the bundled tests quote, and the ones a reader of the
#: demo needs.  Placed at their OWN paths, not flattened, so a test's `PAGE`
#: constant finds them.  Wiki filenames are unique across the vault, so they
#: are found by name.
WIKI_PAGES = [
    "case-study-racelab-graph-atlas-0.1.md",
    "case-study-racelab-switch-atlas-0.1.md",
    "poc3-racelab-demo.md",
    "poc3-racelab-traced-car.md",
    "poc3-racelab-3d.md",
    "poc3-racelab-drawn-car.md",
    # the body-fitted column, Tiers 58-73.  Each of the first fourteen is
    # OPENED by the test of its tier, so leaving one out turns a passing test
    # into a skip; the last two are not opened by any test and are here because
    # the certified mode is what the demo is for.
    "poc3-racelab-car-fixes.md",
    "poc3-racelab-outlet-and-start.md",
    "poc3-racelab-body-fitted-grids.md",
    "poc3-racelab-overset-flow.md",
    "poc3-racelab-car-solids.md",
    "poc3-racelab-duct-openings.md",
    "poc3-racelab-car-union.md",
    "poc3-racelab-car-graph.md",
    "poc3-racelab-car-render.md",
    "poc3-racelab-car-windows.md",
    "poc3-racelab-bodyfitted-demo.md",
    "poc3-racelab-knobs.md",
    "poc3-racelab-knobs-wired.md",
    "poc3-racelab-dashboard.md",
    "poc3-racelab-certified-step.md",
    "poc3-racelab-certified-screen.md",
]
WIKI_DIR = ("wiki", "concepts", "Atlas 0.1", "common")

#: The batch drivers behind the recorded runs, and the scripts the tests import.
SCRIPTS = [
    "tier51_racelab_graph.py",
    "tier52_racelab_switch.py",
    "tier53_racelab_rerun.py",
    "tier54_traced_car.py",
    "car_editor.py",
    "car_editor.html",
    "CAR_EDITOR.md",
    # Tiers 58-73, the body-fitted column.  These are not optional extras: the
    # demo IMPORTS `tier62_car_solids` and `tier63_duct_openings` at build time
    # (`bodyfitted._devices`, `BodyFittedColumn.build`), so without them the
    # body-fitted page does not start at all.  The rest are what their tests
    # import.  The dependency graph is closed inside this list plus
    # `tier53_racelab_rerun.py`, which is already above.
    "tier58_car_fixes.py",
    "tier59_outflow_and_start.py",
    "tier60_body_fitted_grids.py",
    "tier61_overset_flow.py",
    "tier62_car_solids.py",
    "tier63_duct_openings.py",
    "tier64_car_union.py",
    "tier65_car_graph.py",
    "tier66_car_render.py",
    "tier67_car_windows.py",
    "tier68_bodyfitted_demo.py",
    "tier69_car_knobs.py",
    "tier70_knobs_wired.py",
    "tier71_dashboard.py",
    "tier72_certified_step.py",
    "tier73_certified_screen.py",
]
#: Every PoC 3 test file, unchanged.
TESTS = [
    "test_tier51_racelab_graph.py",
    "test_tier52_racelab_switch.py",
    "test_tier53_racelab_demo.py",
    "test_tier54_racelab_3d.py",
    "test_tier55_car_geometry.py",
    "test_tier56_racelab_drawn_car.py",
    "test_tier57_racelab_bundle.py",
    #: records OF the bundle, which it does not carry, so every test in it
    #: skips there and says why -- carried so the bundle has every PoC 3 test
    "test_tier57_racelab_bundle_records.py",
    # Tiers 58-73, the body-fitted column and the certified mode
    "test_tier58_car_fixes.py",
    "test_tier59_outflow.py",
    "test_tier60_overset_grids.py",
    "test_tier61_overset_flow.py",
    "test_tier62_car_solids.py",
    "test_tier63_duct_openings.py",
    "test_tier64_car_union.py",
    "test_tier65_car_graph.py",
    "test_tier66_car_render.py",
    "test_tier67_car_windows.py",
    "test_tier68_bodyfitted_demo.py",
    "test_tier69_car_knobs.py",
    "test_tier70_knobs_wired.py",
    "test_tier71_dashboard.py",
    "test_tier72_certified_step.py",
    "test_tier73_certified_screen.py",
    #: about the BUILDER and the launcher templates, which live upstream --
    #: those tests skip here and say why, the way the bundle-records ones do,
    #: so the bundle still carries every PoC 3 test
    "test_tier74_showable.py",
]

#: The recorded runs.  Each is what a test asserts against or what the demo
#: reads; `settled.npz` is the release state and is checked against the car.
ARTIFACTS = [
    ("out/racelab5/cache/settled.npz",
     "the settled field the demo releases from, settled around THIS car"),
    ("out/racelab5/racelab5.json",
     "Tier 56's record of the drawn car -- the demo's machine sizing is read "
     "against it"),
    ("out/racelab4/racelab4.json", "Tier 54's record, the traced car"),
    ("out/racelab3/racelab3.json", "Tier 53's record"),
    ("out/racelab2/racelab2.json", "Tier 52's record"),
    ("out/racelab/racelab.json", "Tier 51's record, CS-19"),
    # -- the body-fitted column, Tiers 58-73 -------------------------------
    ("out/racelab6/racelab6.json", "Tier 58's record, the car's fixes"),
    ("out/racelab7/racelab7.json", "Tier 59's record, the outlet and the start"),
    ("out/racelab8/racelab8.json", "Tier 59's arms"),
    ("out/racelab8/cache/settled.npz",
     "Tier 59's settled arm; without it one test in test_tier59 skips"),
    ("out/racelab9/racelab9.json", "Tier 60's record, the overset grids"),
    ("out/racelab10/racelab10.json", "Tier 61's record, the flow solver"),
    ("out/racelab11/racelab11.json", "Tier 62's record, the car as solids"),
    ("out/racelab12/racelab12.json", "Tier 63's record, the duct openings"),
    ("out/racelab13/racelab13.json",
     "Tier 64's record -- `bodyfitted._sizing` READS it for the machine's "
     "sizing inflow, so the body-fitted demo needs it, not only its test"),
    ("out/racelab14/racelab14.json", "Tier 65's record, the graph"),
    ("out/racelab15/racelab15.json", "Tier 66's record, the overlay"),
    ("out/racelab16/racelab16.json", "Tier 67's record, the learned territory"),
    ("out/racelab17/racelab17.json", "Tier 68's record, the demo engine"),
    ("out/racelab18/racelab18.json", "Tier 69's record, the knobs"),
    ("out/racelab19/racelab19.json", "Tier 70's record, the knobs wired"),
    ("out/racelab20/racelab20.json", "Tier 71's record, the dashboard"),
    ("out/racelab21/racelab21.json", "Tier 72's record, the certified step"),
    ("out/racelab22/racelab22.json", "Tier 73's record, the certified screen"),
]

#: The body-fitted column's settled release state.  Its name carries the car's
#: fingerprint, so the path is not a constant -- it is asked for at build time
#: and checked, exactly as `BodyFittedColumn.settled_path` computes it.
#:
#: **The demo does not start without it.**  With no settled field for this car
#: the column falls back to Tier 64's device-free prefix and settles from t = 8
#: to t = 12, which is 320 steps -- about four minutes before the first frame --
#: and if the prefix belongs to another car it refuses to march at all (W293).
def _bodyfitted_settled() -> tuple[str, str]:
    from atlas.demo_racelab.bodyfitted import BodyFittedColumn

    path = BodyFittedColumn().settled_path(ROOT)
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    return rel, os.path.basename(path)
#: The only binary files the bundle may carry, by exact path pattern.
BINARY_ALLOWED = (
    re.compile(r"^vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/"
               r"snapshots/[0-9a-f]{40}/model\.safetensors$"),
    re.compile(r"^out/racelab5/cache/settled\.npz$"),
    re.compile(r"^out/racelab8/cache/settled\.npz$"),
    #: the body-fitted release state, named for the car it was settled around
    re.compile(r"^out/cache/bodyfitted_t\d+(\.\d+)?_[0-9a-f]{12}\.npz$"),
)
#: What a weight or field file looks like, so an unexpected one is caught by
#: its extension before anything else.
DATA_EXT = (".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle",
            ".npz", ".npy", ".h5", ".hdf5", ".nc", ".vtk", ".vtu", ".zarr",
            ".onnx", ".msgpack", ".joblib")

HUB_DIR = "models--camlab-ethz--Poseidon-T"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache",
                                "bundle", ".git", "*.ipynb_checkpoints")

#: The unlicensed structural checkpoint's name, assembled at run time so this
#: file does not trip its own scan -- the same device the tier 51 test uses.
_NB = "neuber" + "net"
#: USE, not the word: an import, an attribute access, or a path into its cache.
_NB_USE = re.compile(r"import\s+%s|from\s+%s|%s\s*\.\w|%s[/\\]|cache[/\\]%s"
                     % (_NB, _NB, _NB, _NB, _NB), re.IGNORECASE)
_NB_WORD = re.compile(_NB, re.IGNORECASE)


def _git(repo: str, *args: str, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                         text=True, check=check)
    return out.stdout.strip()


def _commit_of(path: str) -> str:
    try:
        return _git(path, "rev-parse", "HEAD")
    except Exception:                                       # pragma: no cover
        return "unknown"


def _size_mb(path: str) -> float:
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _dn, fn in os.walk(path) for f in fn) / 1e6


def _hf_cache() -> str:
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _rmtree(path: str) -> None:
    """`shutil.rmtree` that survives a previous build on Windows.

    Found by rebuilding in place: `copytree` copies directory attributes, the
    source `atlas/` lives in OneDrive, which marks folders READ-ONLY, so the
    copy's folders are read-only too and `os.rmdir` answers WinError 5.  Git's
    own objects are read-only as well, and a scanner can hold a fresh file for a
    moment.  So: clear the attribute and retry, a few times, then give up
    loudly.
    """
    import stat
    import time

    def retry(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        except OSError:
            pass
        for k in range(6):
            try:
                func(p)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(0.3 * (k + 1))
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:                                                   # pragma: no cover
        shutil.rmtree(path, onerror=retry)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the scans -- each returns what it found, and `build` refuses on a finding
# ---------------------------------------------------------------------------


def _walk(root: str):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", ".venv", "__pycache__")]
        for f in fn:
            full = os.path.join(dp, f)
            yield full, os.path.relpath(full, root).replace(os.sep, "/")


def _is_text(path: str) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(8192)
    return b"\x00" not in head


def scan_unlicensed_checkpoint(root: str) -> dict:
    """Any file of it, any path into its cache, any import of it.

    Mentions of the name in prose -- the on-screen reason a structural window
    has no learned option, a docstring saying it was not loaded, the tests'
    own guards -- are COUNTED and returned, not refused.  That is the rule
    `test_tier51_racelab_graph.py` states: a substring test on the name flags
    the honesty rather than the violation.
    """
    paths, uses, words = [], [], {}
    for full, rel in _walk(root):
        if _NB_WORD.search(rel):
            paths.append(rel)
        if not _is_text(full):
            continue
        with open(full, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for m in _NB_USE.finditer(text):
            uses.append({"file": rel, "match": m.group(0)})
        n = len(_NB_WORD.findall(text))
        if n:
            words[rel] = n
    return {"files_or_dirs_named_for_it": paths, "uses": uses,
            "prose_mentions_by_file": words,
            "prose_mentions_total": sum(words.values())}


def scan_data(root: str) -> dict:
    """Every binary file and every weight- or field-shaped file."""
    unexpected, allowed = [], []
    for full, rel in _walk(root):
        data_shaped = rel.lower().endswith(DATA_EXT)
        binary = not _is_text(full)
        if not (data_shaped or binary):
            continue
        if any(p.match(rel) for p in BINARY_ALLOWED):
            allowed.append({"file": rel, "bytes": os.path.getsize(full)})
        else:
            unexpected.append({"file": rel, "bytes": os.path.getsize(full)})
    return {"allowed": allowed, "unexpected": unexpected}


def scan_scot(root: str) -> list:
    """`scOT` must be installed, never vendored: no package directory of it."""
    hits = []
    for dp, dn, _fn in os.walk(root):
        for d in dn:
            if d.lower() == "scot":
                hits.append(os.path.relpath(os.path.join(dp, d), root))
    return hits


# ---------------------------------------------------------------------------


def _dirty(paths) -> list:
    """Uncommitted changes under any path the bundle copies from this repo."""
    out = _git(ROOT, "status", "--porcelain", "--", *paths)
    return [ln for ln in out.splitlines() if ln.strip()]


def build(out: str, build_repo: str, allow_dirty: bool = False) -> dict:
    copied_from_here = (["atlas"] + ["scripts/" + s for s in SCRIPTS]
                        + ["tests/" + t for t in TESTS]
                        + ["/".join(WIKI_DIR) + "/" + p for p in WIKI_PAGES]
                        + [a for a, _w in ARTIFACTS if a.endswith(".json")])
    dirty = _dirty(copied_from_here)
    if dirty and not allow_dirty:
        raise SystemExit(
            "these paths the bundle copies have uncommitted changes, so no "
            "commit would name what was copied:\n  " + "\n  ".join(dirty)
            + "\ncommit them, or pass --allow-dirty for a scratch build")

    if os.path.isdir(out):
        _rmtree(out)
    os.makedirs(out)

    # 1. the framework and the demo -----------------------------------------
    shutil.copytree(os.path.join(ROOT, "atlas"), os.path.join(out, "atlas"),
                    ignore=IGNORE)

    # 2. the build repository's solvers, in the shape their loaders expect --
    src = os.path.join(build_repo, "src", "atlas")
    if not os.path.isfile(os.path.join(src, "__init__.py")):
        raise SystemExit("the solvers are not at %r; set ATLAS_BUILD_REPO or "
                         "pass --build-repo" % src)
    shutil.copytree(src, os.path.join(out, "vendor", "src", "atlas"),
                    ignore=IGNORE)
    #: assert the entry points the loaders actually open are present, rather
    #: than trusting that copying a tree copied the right tree
    for probe in ("vendor/src/atlas/__init__.py",
                  "vendor/src/atlas/cases/windfarm/__init__.py",
                  "vendor/src/atlas/cases/windfarm/reference.py",
                  "vendor/src/atlas/cases/windfarm/adapters.py",
                  "vendor/src/atlas/solvers/thermostruct2d.py",
                  "vendor/src/atlas/config/atlas_0_1.yaml"):
        if not os.path.isfile(os.path.join(out, *probe.split("/"))):
            raise SystemExit("the bundle is missing %s -- the build repo layout "
                             "is not what the loaders expect" % probe)

    # 3. the checkpoint, in the shape the hub expects ------------------------
    cache = os.path.join(_hf_cache(), HUB_DIR)
    ref = os.path.join(cache, "refs", "main")
    if not os.path.isfile(ref):
        raise SystemExit("no Poseidon-T in the Hugging Face cache at %r. Load "
                         "the expert once with a network connection, or set "
                         "HF_HOME." % cache)
    snap = open(ref).read().strip()
    snap_dir = os.path.join(cache, "snapshots", snap)
    want = ["config.json", "model.safetensors"]
    missing = [f for f in want if not os.path.isfile(os.path.join(snap_dir, f))]
    if missing:
        raise SystemExit("the cached snapshot %s is missing %s" % (snap, missing))
    hub_out = os.path.join(out, "vendor", "hf-cache", "hub", HUB_DIR)
    os.makedirs(os.path.join(hub_out, "refs"))
    os.makedirs(os.path.join(hub_out, "snapshots", snap))
    # transformers reads `<hub>/version.txt` to decide whether the cache is a
    # pre-4.22 one it should migrate; without it, offline, every load prints a
    # warning that every call will fail -- false here, and the first thing a
    # new user would read (PoC 1's finding).
    with open(os.path.join(os.path.dirname(hub_out), "version.txt"), "w",
              newline="") as fh:
        fh.write("1")
    with open(os.path.join(hub_out, "refs", "main"), "w", newline="\n") as fh:
        fh.write(snap)
    for f in want:
        # copy, never link: a symlink into the developer's cache is a bundle
        # that works on exactly one machine and reports success on every other
        shutil.copyfile(os.path.join(snap_dir, f),
                        os.path.join(hub_out, "snapshots", snap, f))
    weights = os.path.join(hub_out, "snapshots", snap, "model.safetensors")
    weights_sha = _sha256(weights)
    shutil.copyfile(os.path.join(TEMPLATES, "POSEIDON-T-LICENCE.md"),
                    os.path.join(out, "vendor", "POSEIDON-T-LICENCE.md"))

    # 4. the recorded runs, and the field checked against the car ------------
    for rel, why in ARTIFACTS:
        s = os.path.join(ROOT, *rel.split("/"))
        if not os.path.isfile(s):
            raise SystemExit("%s is missing (%s)" % (rel, why))
        d = os.path.join(out, *rel.split("/"))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
    import numpy as np
    from atlas.cases import racelab as RL
    car = RL.geometry_fingerprint()
    with np.load(os.path.join(out, "out", "racelab5", "cache",
                              "settled.npz")) as z:
        field_fp = str(z["geometry"]) if "geometry" in z.files else None
    if field_fp != car:
        raise SystemExit(
            "out/racelab5/cache/settled.npz was settled around a different car "
            "(%s) from the one this repository builds (%s); the demo would "
            "release the car from another car's flow. Re-run `python "
            "scripts/tier54_traced_car.py --out out/racelab5 --stages spinup`."
            % ((field_fp or "no fingerprint")[:16], car[:16]))

    # 4b. the body-fitted column's settled field ----------------------------
    # Its name carries the car's fingerprint, so the check that it belongs to
    # this car IS the check that the file exists under that name -- which is
    # what `settled_path` computes and what `load_state` refuses a mismatch of.
    bf_rel, bf_name = _bodyfitted_settled()
    bf_src = os.path.join(ROOT, *bf_rel.split("/"))
    if not os.path.isfile(bf_src):
        raise SystemExit(
            "the body-fitted column's settled field %s is missing, so the "
            "body-fitted page would settle for about four minutes before its "
            "first frame, or refuse to march. Build it once with `python "
            "scripts/tier68_bodyfitted_demo.py --out out/racelab17 "
            "--stages build,settle`." % bf_rel)
    bf_fp = re.search(r"_([0-9a-f]{12})\.npz$", bf_name)
    if bf_fp is None or bf_fp.group(1) != car[:12]:
        raise SystemExit(
            "%s does not carry this repository's car (%s): the body-fitted "
            "demo would release from another car's flow" % (bf_name, car[:12]))
    bf_dst = os.path.join(out, *bf_rel.split("/"))
    os.makedirs(os.path.dirname(bf_dst), exist_ok=True)
    shutil.copyfile(bf_src, bf_dst)

    # 5. scripts, tests, the pages the tests quote --------------------------
    os.makedirs(os.path.join(out, "scripts"))
    for f in SCRIPTS:
        shutil.copyfile(os.path.join(ROOT, "scripts", f),
                        os.path.join(out, "scripts", f))
    os.makedirs(os.path.join(out, "tests"))
    for f in TESTS:
        s = os.path.join(ROOT, "tests", f)
        if not os.path.isfile(s):
            raise SystemExit("tests/%s does not exist" % f)
        shutil.copyfile(s, os.path.join(out, "tests", f))
    wiki = os.path.join(ROOT, "wiki")
    found = {f: os.path.join(dp, f) for dp, _dn, fn in os.walk(wiki)
             for f in fn if f in set(WIKI_PAGES)}
    lost = [f for f in WIKI_PAGES if f not in found]
    if lost:
        raise SystemExit("these wiki pages are not in %r: %s" % (wiki, lost))
    page_dir = os.path.join(out, *WIKI_DIR)
    os.makedirs(page_dir)
    for f in WIKI_PAGES:
        shutil.copyfile(found[f], os.path.join(page_dir, f))

    # 6. the launchers and the paperwork -------------------------------------
    for f in ("run.py", "run.sh", "run.cmd", "requirements.txt", "conftest.py",
              "README.md", "PROVENANCE.md"):
        shutil.copyfile(os.path.join(TEMPLATES, f), os.path.join(out, f))
    if os.path.isfile(os.path.join(TEMPLATES, "constraints.txt")):
        shutil.copyfile(os.path.join(TEMPLATES, "constraints.txt"),
                        os.path.join(out, "constraints.txt"))
    # **The allowlist is GENERATED from ARTIFACTS, not written by hand.**  The
    # template's hand-written version stopped at racelab5 while ARTIFACTS grew
    # to racelab22, so `--commit` would have dropped every record added after
    # Tier 57 from the orphan branch -- silently, because `git add` on an
    # ignored path says nothing.  One list, two readers.
    keep = [a for a, _w in ARTIFACTS] + [_bodyfitted_settled()[0]]
    rules = []
    for rel in keep:
        parts = rel.split("/")[1:]          # everything under out/
        for i in range(len(parts) - 1):
            d = "/".join(parts[:i + 1])
            rules += ["!out/%s/" % d, "out/%s/*" % d]
        rules.append("!out/%s" % "/".join(parts))
    seen, ordered = set(), []
    for r in rules:                          # stable, de-duplicated
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    with open(os.path.join(TEMPLATES, "gitignore"), encoding="utf-8") as fh:
        base = fh.read()
    with open(os.path.join(out, ".gitignore"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(base.rstrip("\n") + "\n" + "\n".join(ordered) + "\n")
    shutil.copyfile(os.path.join(TEMPLATES, "gitattributes"),
                    os.path.join(out, ".gitattributes"))
    os.chmod(os.path.join(out, "run.sh"), 0o755)

    # 7. what this copy was taken from ---------------------------------------
    today = dt.date.today().isoformat()
    scot = re.search(r"poseidon/archive/([0-9a-f]{40})\.zip",
                     open(os.path.join(TEMPLATES, "run.sh")).read()).group(1)
    lines = [
        "assembled          %s" % today,
        "",
        "nonidino/Atlas                       %s%s"
        % (_commit_of(ROOT), "   (DIRTY: %d paths uncommitted)" % len(dirty)
           if dirty else ""),
        "nonidino/physics-foundation-model    %s" % _commit_of(build_repo),
        "camlab-ethz/Poseidon-T (weights)     %s   (%.1f MB, CC-BY-NC-4.0)"
        % (snap, os.path.getsize(weights) / 1e6),
        "   model.safetensors sha256          %s" % weights_sha,
        "camlab-ethz/poseidon   (scOT code)    %s   (installed by the "
        "launchers, not vendored: no licence upstream)" % scot,
        "",
        "the car                              %s" % car,
        "   (racelab.geometry_fingerprint; out/racelab5/cache/settled.npz "
        "carries the same one)",
        "",
    ]
    with open(os.path.join(out, "SOURCE_COMMITS"), "w", newline="\n") as fh:
        fh.write("\n".join(lines))

    # 8. what must not be here -----------------------------------------------
    nb = scan_unlicensed_checkpoint(out)
    data = scan_data(out)
    scot_dirs = scan_scot(out)
    problems = []
    if nb["files_or_dirs_named_for_it"] or nb["uses"]:
        problems.append("the unlicensed structural checkpoint is reachable: %s"
                        % (nb["files_or_dirs_named_for_it"] + nb["uses"]))
    if data["unexpected"]:
        problems.append("binary or weight-shaped files that are not allowed: %s"
                        % data["unexpected"])
    if scot_dirs:
        problems.append("scOT is vendored, and it has no licence: %s" % scot_dirs)
    report = {
        "out": out, "branch": BRANCH, "snapshot": snap,
        "weights_sha256": weights_sha, "car": car,
        "atlas_commit": _commit_of(ROOT), "dirty": dirty,
        "build_repo_commit": _commit_of(build_repo),
        "sizes_mb": {"total": _size_mb(out),
                     "atlas": _size_mb(os.path.join(out, "atlas")),
                     "vendor_src": _size_mb(os.path.join(out, "vendor", "src")),
                     "hf_cache": _size_mb(os.path.join(out, "vendor",
                                                       "hf-cache")),
                     "out": _size_mb(os.path.join(out, "out"))},
        "scan_unlicensed_checkpoint": nb, "scan_data": data,
        "scan_scot": scot_dirs, "problems": problems,
    }
    print("  built %s" % out)
    print("  %.1f MB total: framework %.1f, solvers %.1f, checkpoint %.1f, "
          "recorded runs %.1f" % tuple(report["sizes_mb"][k] for k in
                                       ("total", "atlas", "vendor_src",
                                        "hf_cache", "out")))
    print("  unlicensed checkpoint: %d files, %d uses; its name in prose %d "
          "times in %d files (reported, not refused)"
          % (len(nb["files_or_dirs_named_for_it"]), len(nb["uses"]),
             nb["prose_mentions_total"], len(nb["prose_mentions_by_file"])))
    print("  binary files: %d allowed, %d unexpected; scOT dirs: %d"
          % (len(data["allowed"]), len(data["unexpected"]), len(scot_dirs)))
    if problems:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(problems))
    return report


def commit(out: str, message: str) -> str:
    """Refresh the local orphan branch from the built directory.

    The branch shares no history with `atlas-0.1` by design, so it is replaced
    wholesale: a copy that accumulated its own history would invite someone to
    fix something in it.
    """
    def git(*a):
        return subprocess.run(["git", "-C", out, *a], check=True,
                              capture_output=True, text=True).stdout.strip()

    def _cfg(key: str, fallback: str) -> str:
        v = subprocess.run(["git", "-C", ROOT, "config", key],
                           capture_output=True, text=True).stdout.strip()
        return v or fallback

    git("init", "-q", "-b", BRANCH)
    # the checkpoint's own directory name is long; past Windows MAX_PATH from
    # a deep build directory without this
    git("config", "core.longpaths", "true")
    git("config", "user.email", _cfg("user.email", "noreply@example.com"))
    git("config", "user.name", _cfg("user.name", "atlas"))
    git("add", "-A")
    #: the recorded runs sit behind an `out/*` ignore rule so that a user's own
    #: runs do not get committed; the ones that ARE the bundle are force-added.
    #: The body-fitted column's settled field is here too and was NOT before:
    #: it is not in `ARTIFACTS` (its name carries the car's fingerprint, so it
    #: is resolved at build time), so `out/*` would have dropped it from the
    #: branch and the body-fitted page would have had nothing to release from
    #: on a fresh clone -- while every test still passed here.
    must_track = [rel for rel, _why in ARTIFACTS] + [_bodyfitted_settled()[0]]
    for rel in must_track:
        git("add", "-f", rel)
    # `os.chmod` above does nothing on Windows -- NTFS has no execute bit and
    # git records 100644 -- so the mode is set in the INDEX.  Without this
    # `./run.sh` on the branch is not executable and a Linux or macOS user's
    # first command fails "Permission denied" (exit 126).
    git("update-index", "--chmod=+x", "run.sh")
    git("commit", "-q", "-m", message)
    head = git("rev-parse", "HEAD")
    mode = git("ls-files", "-s", "run.sh").split()
    print("  committed %s on %s in %s" % (head[:12], BRANCH, out))
    print("  run.sh in the index: %s (must be 100755)" % (mode[0] if mode else "??"))
    if not mode or mode[0] != "100755":
        raise SystemExit("run.sh is not 100755 in the index; a Linux clone "
                         "would fail with exit 126")
    # **What a fresh clone would actually get.**  `git add` on an ignored path
    # says nothing, so a rule that quietly drops a file leaves a branch that
    # builds, tests and demos here and is missing a file there.  Asked of the
    # index rather than of the ignore rules, because the index is what clones.
    tracked = set(git("ls-files").splitlines())
    lost = [p for p in must_track if p not in tracked]
    if lost:
        raise SystemExit(
            "these are in the bundle directory but NOT in the branch, so a "
            "clone would not get them:\n  " + "\n  ".join(lost))
    print("  %d recorded runs tracked, including %s"
          % (len(must_track), os.path.basename(must_track[-1])))
    print("  to publish:  git -C %s push --force <remote> %s" % (out, BRANCH))
    return head


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "bundle3"))
    ap.add_argument("--build-repo",
                    default=os.environ.get(
                        "ATLAS_BUILD_REPO",
                        os.path.join(os.path.expanduser("~"),
                                     "physics-foundation-model")))
    ap.add_argument("--commit", action="store_true",
                    help="also make the orphan branch commit inside --out")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build even if copied paths have uncommitted changes "
                         "(recorded in SOURCE_COMMITS as DIRTY)")
    ap.add_argument("--report", default=None,
                    help="also write the build report as JSON here")
    ap.add_argument("-m", "--message",
                    default="PoC 3 demo bundle: RaceLab, a car as a graph of "
                            "interchangeable physics experts")
    args = ap.parse_args(argv)
    rep = build(os.path.abspath(args.out), args.build_repo, args.allow_dirty)
    if args.commit:
        rep["commit"] = commit(os.path.abspath(args.out), args.message)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
