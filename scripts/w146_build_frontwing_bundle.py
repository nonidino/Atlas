"""W146 -- assemble the self-contained `poc2-frontwing-demo` bundle.

    python scripts/w146_build_frontwing_bundle.py                  # into out/bundle2
    python scripts/w146_build_frontwing_bundle.py --out /tmp/poc2
    python scripts/w146_build_frontwing_bundle.py --commit         # ... and refresh
                                                                   # the orphan branch

The bundle is a COPY of four things that normally live in three places -- this
framework, two solvers in the build repository, and the recorded run's own
artifact -- laid out so that each resolves **with no change to any source file**:

  * the fluid expert at ``vendor/src/atlas/cases/windfarm/``, because
    `window_ns.load_reference` looks for
    ``$ATLAS_BUILD_REPO/src/atlas/cases/windfarm``;
  * the structural expert at ``vendor/src/atlas/solvers/``, because
    `thermal_strain.load_solvers` looks for ``$ATLAS_BUILD_REPO/src/atlas`` and
    binds the whole package under a private module name so its relative imports
    resolve;
  * the settled field and the artifact at ``out/w141/``, where
    `demo_frontwing.engine` and the driver both already look;
  * everything else where it already is.

That is what lets the tests in the bundle be the same tests as the ones here,
character for character -- which is the only reason passing them there means
anything.

**Nothing is fetched from the network, at build time or at run time.** The PoC 1a
bundle carried 83 MB of frozen Poseidon-T weights and installed `scOT` from a
pinned upstream archive; this one carries neither, because every expert in this
demo is a classical solver and the beat that offers a neural operator hands the
compiler its *declaration* rather than its weights. Total: about five megabytes.

**This script does not push.** It builds a directory, and with ``--commit`` it
refreshes the local orphan branch from that directory. Pushing is a deliberate,
separate act, because the branch is force-replaced wholesale every time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BRANCH = "poc2-frontwing-demo"

#: Wiki pages the results actually rest on. Copied flat into `docs/`, which is
#: why every wiki filename in this vault is unique.
DOCS = [
    "poc2-frontwing-results.md",
    "poc2-demo-and-novelty.md",
    "case-study-ground-effect-atlas-0.1.md",
    "case-study-wing-fsi-atlas-0.1.md",
    "prior-art-and-novelty-atlas-0.1.md",
    "case-study-ladder-to-f1.md",
    "f1-pathmap-and-end-goal.md",
    "port-algebra-atlas-0.1.md",
    "poc1a-frozen-expert-results.md",
]
SCRIPTS = ["w141_poc2_frontwing.py"]
TESTS = ["test_tier30_front_wing.py", "test_tier31_frontwing_demo.py"]

#: The recorded run. `settled.npz` is the state the demo releases from and
#: `w141.json` is every "recorded" number on screen. Both are small and both are
#: the difference between a demo that shows one live number and a demo that
#: shows a live number beside the one it should be compared with.
ARTIFACTS = [
    ("out/w141/settled.npz", "the settled flow field the demo releases from"),
    ("out/w141/w141.json", "the full-scale run's own artifact"),
]

#: The two solver trees, and what each is needed for. The whole build-repo
#: `src/atlas` is copied rather than only `solvers/`, because `load_solvers`
#: binds the package and `grid.py`'s ``..config`` imports have to resolve.
VENDOR = [
    ("src/atlas", "the structural expert (solvers/thermostruct2d.py) and the "
                  "fluid expert (cases/windfarm/reference.py)"),
]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache",
                                "bundle", ".git", "*.ipynb_checkpoints")


def _commit_of(path: str) -> str:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                       # pragma: no cover
        return "unknown"


def _size(path: str) -> float:
    return sum(os.path.getsize(os.path.join(dp, f))
               for dp, _dn, fn in os.walk(path) for f in fn) / 1e6


def build(out: str, build_repo: str) -> str:
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # 1. the framework and the demo -----------------------------------------
    shutil.copytree(os.path.join(ROOT, "atlas"), os.path.join(out, "atlas"),
                    ignore=IGNORE)

    # 2. both solvers, in the shape their loaders expect ---------------------
    for rel, _why in VENDOR:
        src = os.path.join(build_repo, *rel.split("/"))
        if not os.path.isfile(os.path.join(src, "__init__.py")):
            raise SystemExit(
                f"the solvers are not at {src!r}; set ATLAS_BUILD_REPO or pass "
                "--build-repo")
        shutil.copytree(src, os.path.join(out, "vendor", *rel.split("/")),
                        ignore=IGNORE)
    #: assert the two entry points the loaders actually open are present, rather
    #: than trusting that copying a tree copied the right tree
    for probe in ("vendor/src/atlas/solvers/thermostruct2d.py",
                  "vendor/src/atlas/cases/windfarm/reference.py",
                  "vendor/src/atlas/cases/windfarm/__init__.py"):
        if not os.path.isfile(os.path.join(out, *probe.split("/"))):
            raise SystemExit(f"the bundle is missing {probe} -- the build repo "
                             "layout is not what the loaders expect")

    # 3. the recorded run ----------------------------------------------------
    os.makedirs(os.path.join(out, "out", "w141"))
    for rel, why in ARTIFACTS:
        src = os.path.join(ROOT, *rel.split("/"))
        if not os.path.isfile(src):
            raise SystemExit(
                f"{rel} is missing ({why}). Run "
                "`python scripts/w141_poc2_frontwing.py` first.")
        shutil.copyfile(src, os.path.join(out, *rel.split("/")))

    # 4. scripts, tests, docs ------------------------------------------------
    os.makedirs(os.path.join(out, "scripts"))
    for f in SCRIPTS:
        shutil.copyfile(os.path.join(ROOT, "scripts", f),
                        os.path.join(out, "scripts", f))
    os.makedirs(os.path.join(out, "tests"))
    for f in TESTS:
        src = os.path.join(ROOT, "tests", f)
        if not os.path.isfile(src):
            raise SystemExit(f"tests/{f} does not exist")
        shutil.copyfile(src, os.path.join(out, "tests", f))
    os.makedirs(os.path.join(out, "docs"))
    # Wiki filenames are unique across the vault by convention -- that is what
    # makes its bare `[[page-name]]` links reorganisation-proof -- so the pages
    # are found by name rather than by the folder they happen to sit in today.
    wiki = os.path.join(ROOT, "wiki")
    found = {f: os.path.join(dp, f) for dp, _dn, fn in os.walk(wiki)
             for f in fn if f in set(DOCS)}
    missing = [f for f in DOCS if f not in found]
    if missing:
        raise SystemExit(f"these wiki pages are not in {wiki!r}: {missing}")
    for f in DOCS:
        shutil.copyfile(found[f], os.path.join(out, "docs", f))

    # 5. the launchers and the paperwork -------------------------------------
    b = os.path.join(ROOT, "atlas", "demo_frontwing", "bundle")
    for f in ("run.py", "run.sh", "run.cmd", "requirements.txt", "conftest.py",
              "README.md"):
        shutil.copyfile(os.path.join(b, f), os.path.join(out, f))
    shutil.copyfile(os.path.join(b, "gitignore"), os.path.join(out, ".gitignore"))
    shutil.copyfile(os.path.join(b, "gitattributes"),
                    os.path.join(out, ".gitattributes"))
    os.chmod(os.path.join(out, "run.sh"), 0o755)

    # 6. what this copy was taken from ---------------------------------------
    today = dt.date.today().isoformat()
    lines = [
        f"assembled          {today}",
        "",
        f"nonidino/Atlas                       {_commit_of(ROOT)}",
        f"nonidino/physics-foundation-model    {_commit_of(build_repo)}",
        "",
        "No third-party weights, no third-party source, nothing fetched from",
        "the network at build time or at run time. Every solver in this bundle",
        "is first-party unpublished work.",
        "",
    ]
    with open(os.path.join(out, "SOURCE_COMMITS"), "w", newline="\n") as fh:
        fh.write("\n".join(lines))

    print(f"  built {out}")
    print(f"  {_size(out):.1f} MB total: framework "
          f"{_size(os.path.join(out, 'atlas')):.1f}, solvers "
          f"{_size(os.path.join(out, 'vendor')):.1f}, recorded run "
          f"{_size(os.path.join(out, 'out')):.1f}")
    return out


def commit(out: str, message: str) -> None:
    """Refresh the local orphan branch from the built directory.

    The branch shares no history with `atlas-0.1` by design, so it is replaced
    wholesale: a copy that accumulated its own history would invite someone to
    fix something in it.
    """
    def git(*a, **kw):
        return subprocess.run(["git", "-C", out, *a], check=True, **kw)

    def _cfg(key: str, fallback: str) -> str:
        v = subprocess.run(["git", "-C", ROOT, "config", key],
                           capture_output=True, text=True).stdout.strip()
        return v or fallback

    git("init", "-q", "-b", BRANCH)
    git("config", "core.longpaths", "true")
    git("config", "user.email", _cfg("user.email", "noreply@example.com"))
    git("config", "user.name", _cfg("user.name", "atlas"))
    git("add", "-A")
    #: the artifacts are behind an `out/*` ignore rule so that a user's own runs
    #: do not get committed; the two that ARE the bundle are force-added
    for rel, _why in ARTIFACTS:
        git("add", "-f", rel)
    # `os.chmod` above does nothing on Windows -- NTFS has no execute bit and git
    # records 100644 -- so the mode is set in the INDEX instead. Without this
    # `./run.sh` on the branch is not executable and a Linux or macOS user's
    # first command fails with "Permission denied" (exit 126).
    git("update-index", "--chmod=+x", "run.sh")
    git("commit", "-q", "-m", message)
    head = subprocess.run(["git", "-C", out, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    mode = subprocess.run(["git", "-C", out, "ls-files", "-s", "run.sh"],
                          capture_output=True, text=True).stdout.strip()
    print(f"  committed {head[:12]} on {BRANCH} in {out}")
    print(f"  run.sh in the index: {mode.split()[0] if mode else '??'} "
          f"(must be 100755)")
    print(f"  to publish:  git -C {out} push --force <remote> {BRANCH}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "bundle2"))
    ap.add_argument("--build-repo",
                    default=os.environ.get(
                        "ATLAS_BUILD_REPO",
                        os.path.join(os.path.expanduser("~"),
                                     "physics-foundation-model")))
    ap.add_argument("--commit", action="store_true",
                    help="also make the orphan branch commit inside --out")
    ap.add_argument("-m", "--message",
                    default="PoC 2 demo bundle: the compiler refuses a real "
                            "neural operator, and an optimiser that can be "
                            "declined")
    args = ap.parse_args(argv)
    out = build(os.path.abspath(args.out), args.build_repo)
    if args.commit:
        commit(out, args.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
