"""W112 -- assemble the self-contained `poc1-windfarm-demo` bundle.

    python scripts/w112_build_bundle.py                     # build into out/bundle
    python scripts/w112_build_bundle.py --out /tmp/poc1
    python scripts/w112_build_bundle.py --commit            # ... and refresh the
                                                            # orphan branch locally

The bundle is a COPY of three things that normally live in three places -- this
framework, the fluid expert in the build repository, and 83 MB of frozen weights
in a Hugging Face cache -- laid out so that each resolves **with no change to any
source file**:

  * the expert at ``vendor/src/atlas/cases/windfarm/``, because
    `window_ns.load_reference` looks for ``$ATLAS_BUILD_REPO/src/atlas/cases/windfarm``;
  * the checkpoint at ``vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/``,
    because `adapters.FrozenFluidExpert` asks the hub for a repository id and the
    way to answer that offline is to hand the hub its own cache;
  * everything else where it already is.

That is what lets the tests in the bundle be the same tests as the ones here,
character for character -- which is the only reason passing them there means
anything.

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

BRANCH = "poc1-windfarm-demo"

#: Wiki pages the results actually rest on. Copied flat into `docs/`, which is
#: why every wiki filename in this vault is unique.
DOCS = [
    "atlas-proof-of-concept-1.md",
    "poc1-results-differentiable-design.md",
    "poc1a-frozen-expert-results.md",
    "poc1-retrospective-and-hybrid-roadmap.md",
    "prior-art-and-novelty-atlas-0.1.md",
    "case-study-wake-array-atlas-0.1.md",
]
SCRIPTS = ["w111_wind_farm_design.py", "w112_farm_demo.py"]
TESTS = ["test_tier21_wind_farm_design.py", "test_tier22_demo.py",
         "test_tier26_poseidon_design.py"]

#: The checkpoint, by hub id. The snapshot hash is read off the cache rather
#: than written down here: a hash in two places is a hash that will disagree.
HUB_DIR = "models--camlab-ethz--Poseidon-T"

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "bundle")


def _hf_cache() -> str:
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _commit_of(path: str) -> str:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:                                       # pragma: no cover
        return "unknown"


def build(out: str, build_repo: str) -> str:
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # 1. the framework -------------------------------------------------------
    shutil.copytree(os.path.join(ROOT, "atlas"), os.path.join(out, "atlas"),
                    ignore=IGNORE)

    # 2. the expert, in the shape its loader expects -------------------------
    src = os.path.join(build_repo, "src", "atlas", "cases", "windfarm")
    if not os.path.isfile(os.path.join(src, "__init__.py")):
        raise SystemExit(f"the expert is not at {src!r}; set ATLAS_BUILD_REPO")
    dst = os.path.join(out, "vendor", "src", "atlas", "cases", "windfarm")
    shutil.copytree(src, dst, ignore=IGNORE)

    # 3. the checkpoint, in the shape the hub expects ------------------------
    cache = os.path.join(_hf_cache(), HUB_DIR)
    ref = os.path.join(cache, "refs", "main")
    if not os.path.isfile(ref):
        raise SystemExit(
            f"no Poseidon-T in the Hugging Face cache at {cache!r}. Load the "
            "expert once with a network connection, or set HF_HOME.")
    snap = open(ref).read().strip()
    snap_dir = os.path.join(cache, "snapshots", snap)
    want = ["config.json", "model.safetensors"]
    missing = [f for f in want if not os.path.isfile(os.path.join(snap_dir, f))]
    if missing:
        raise SystemExit(f"the cached snapshot {snap} is missing {missing}")
    hub_out = os.path.join(out, "vendor", "hf-cache", "hub", HUB_DIR)
    # transformers reads `<hub>/version.txt` to decide whether the cache is a
    # pre-4.22 one it should migrate.  Without it, and offline, every load
    # prints "It is very likely that all your calls to from_pretrained will
    # fail" -- which is false here and is the first thing a new user would see.
    os.makedirs(os.path.join(hub_out, "refs"))
    os.makedirs(os.path.join(hub_out, "snapshots", snap))
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
    mb = os.path.getsize(os.path.join(hub_out, "snapshots", snap,
                                      "model.safetensors")) / 1e6

    # 4. scripts, tests, docs ------------------------------------------------
    os.makedirs(os.path.join(out, "scripts"))
    for f in SCRIPTS:
        shutil.copyfile(os.path.join(ROOT, "scripts", f),
                        os.path.join(out, "scripts", f))
    os.makedirs(os.path.join(out, "tests"))
    for f in TESTS:
        shutil.copyfile(os.path.join(ROOT, "tests", f),
                        os.path.join(out, "tests", f))
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
    b = os.path.join(ROOT, "atlas", "demo", "bundle")
    for f in ("run.py", "run.sh", "run.cmd", "requirements.txt", "conftest.py",
              "README.md", "PROVENANCE.md"):
        shutil.copyfile(os.path.join(b, f), os.path.join(out, f))
    shutil.copyfile(os.path.join(b, "gitignore"), os.path.join(out, ".gitignore"))
    shutil.copyfile(os.path.join(b, "gitattributes"),
                    os.path.join(out, ".gitattributes"))
    shutil.copyfile(os.path.join(b, "POSEIDON-T-LICENCE.md"),
                    os.path.join(out, "vendor", "POSEIDON-T-LICENCE.md"))
    os.chmod(os.path.join(out, "run.sh"), 0o755)

    # 6. what this copy was taken from ---------------------------------------
    today = dt.date.today().isoformat()
    lines = [
        f"assembled          {today}",
        "",
        f"nonidino/Atlas                       {_commit_of(ROOT)}",
        f"nonidino/physics-foundation-model    {_commit_of(build_repo)}",
        f"camlab-ethz/Poseidon-T (weights)     {snap}   "
        f"({mb:.1f} MB, CC-BY-NC-4.0)",
        "camlab-ethz/poseidon   (scOT code)    "
        "b8fa28f59bd7f7673323f28d11a12c6f3a215c61   "
        "(installed by run.sh, not vendored: no licence upstream)",
        "",
    ]
    with open(os.path.join(out, "SOURCE_COMMITS"), "w", newline="\n") as fh:
        fh.write("\n".join(lines))

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _dn, fn in os.walk(out) for f in fn)
    print(f"  built {out}")
    print(f"  {total / 1e6:.1f} MB total, of which the checkpoint is {mb:.1f} MB")
    return out


def commit(out: str, message: str) -> None:
    """Refresh the local orphan branch from the built directory.

    The branch shares no history with `atlas-0.1` by design, so it is replaced
    wholesale: a copy that accumulated its own history would invite someone to
    fix something in it.
    """
    def git(*a, **kw):
        return subprocess.run(["git", "-C", out, *a], check=True, **kw)

    git("init", "-q", "-b", BRANCH)
    git("config", "user.email", subprocess.run(
        ["git", "-C", ROOT, "config", "user.email"], capture_output=True,
        text=True).stdout.strip() or "noreply@example.com")
    git("config", "user.name", subprocess.run(
        ["git", "-C", ROOT, "config", "user.name"], capture_output=True,
        text=True).stdout.strip() or "atlas")
    git("add", "-A")
    git("commit", "-q", "-m", message)
    head = subprocess.run(["git", "-C", out, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"  committed {head[:12]} on {BRANCH} in {out}")
    print(f"  to publish:  git -C {out} push --force <remote> {BRANCH}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "bundle"))
    ap.add_argument("--build-repo",
                    default=os.environ.get("ATLAS_BUILD_REPO",
                                           r"C:\Users\Nauni\physics-foundation-model"))
    ap.add_argument("--commit", action="store_true",
                    help="also make the orphan branch commit inside --out")
    ap.add_argument("-m", "--message",
                    default="PoC 1a demo bundle: frozen-expert column, "
                            "self-measured speed, bundled checkpoint")
    args = ap.parse_args(argv)
    out = build(os.path.abspath(args.out), args.build_repo)
    if args.commit:
        commit(out, args.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
