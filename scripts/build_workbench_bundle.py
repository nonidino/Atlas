"""Demo step 7 (O2) -- assemble the Atlas Workbench's one-command install.

    python scripts/build_workbench_bundle.py                # into out/workbench-bundle
    python scripts/build_workbench_bundle.py --commit       # ... and refresh the
                                                            # orphan branch inside it

In the shape of PoC 3's builder (`build_racelab_bundle.py`), whose scans and
helpers it imports rather than copies.  The bundle is a COPY of what the
workbench needs from two repositories, laid out so that each part resolves
**with no change to any source file**:

  * the framework and the workbench, ``atlas/``, without the three proofs of
    concept's demo packages (``atlas/demo``, ``atlas/demo_frontwing``,
    ``atlas/demo_racelab``: nothing outside them imports them) and without the
    launcher templates (``*/bundle/``);
  * the ONE package of the build repository the workbench loads, the wind
    farm's fluid expert, at ``vendor/src/atlas/cases/windfarm/``: where
    `window_ns.load_reference` looks under ``$ATLAS_BUILD_REPO``, which the
    bundle's `run.py` and `conftest.py` SET to ``vendor/``.  Nothing else of the
    build repository is copied: the one test that cross-checks `fe.py` against
    its ThermoStruct2D skips in the bundle and says why;
  * every workbench test, unchanged, with the helper module they share and the
    scripts they import -- found by reading their imports, not listed by hand,
    so a new import cannot leave the bundle a script short;
  * the launchers and the paperwork, from ``atlas/workbench/bundle/``.

That is what lets the bundle's tests be the same files as the ones here,
character for character, which is the only reason passing them there means
anything.  `scripts/verify_workbench_bundle.py` checks it on a fresh clone.

**What must NOT be in it, and this script refuses to finish if it is:**
NeuberNet in any form (no file of it, no path into its cache, no import of it;
its name in prose is counted and reported, not refused -- PoC 3's rule);
Poseidon's weights, and with them every other weight or field file: no binary
file is allowed at all until the learned case (demo step 8) registers its own
weights in `BINARY_ALLOWED`; a vendored `scOT`; and anything under ``out/``.

**This script does not push.**  It builds a directory, and with ``--commit``
it makes the local orphan branch inside that directory.  The branch stays
private until the website launches (O9); publishing it is a separate act.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

import build_racelab_bundle as B                                      # noqa: E402

BRANCH = "atlas-workbench"
TEMPLATES = os.path.join(ROOT, "atlas", "workbench", "bundle")
OUT = os.path.join(ROOT, "out", "workbench-bundle")
RECORDS = os.path.join(ROOT, "out", "workbench", "records", "installer")

#: atlas/ subpackages the workbench does not import, with the reason.
EXCLUDE_ATLAS = {
    "demo": "PoC 1's dashboard; nothing outside it imports it",
    "demo_frontwing": "PoC 2's dashboard; nothing outside it imports it",
    "demo_racelab": "PoC 3's dashboard and its own bundle templates",
}
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "bundle", ".git",
                                "*.ipynb_checkpoints")

#: The build repository's one package the workbench loads, and the files its
#: loader actually opens -- asserted present rather than trusted to a copytree.
VENDOR_PACKAGE = ("src", "atlas", "cases", "windfarm")
VENDOR_PROBES = ("__init__.py", "reference.py", "adapters.py", "backend.py", "disk.py")

TEST_PATTERN = re.compile(r"^test_workbench_.*\.py$")
TEST_HELPERS = ("workbench_ui.py",)

#: template file -> its path in the bundle.  constraints.txt is optional until it
#: has been generated from a verified install.
LAUNCHERS = {
    "run.py": "run.py",
    "run.sh": "run.sh",
    "run.cmd": "run.cmd",
    "requirements.txt": "requirements.txt",
    "constraints.txt": "constraints.txt",
    "conftest.py": "conftest.py",
    "README.md": "README.md",
    "gitignore": ".gitignore",
    "gitattributes": ".gitattributes",
}
OPTIONAL_TEMPLATES = ("constraints.txt",)

#: The only binary files the bundle may carry, by exact path pattern.  None yet:
#: the learned case (demo step 8) adds its weights here, by name.
BINARY_ALLOWED: tuple = ()

_POSEIDON = re.compile(r"poseidon", re.IGNORECASE)


def tests_to_carry() -> list[str]:
    tests = sorted(f for f in os.listdir(os.path.join(ROOT, "tests")) if TEST_PATTERN.match(f))
    return tests + list(TEST_HELPERS)


def script_closure(files: list[str]) -> list[str]:
    """Every ``scripts/*.py`` the given files import, transitively.

    Read from the source, at any indentation: a test that imports a script
    inside a function (W346's `Rung`) needs it as much as one that imports it at
    the top.  A script that imports another script brings that one too.
    """
    have = {f[:-3] for f in os.listdir(SCRIPTS) if f.endswith(".py")}
    pat = re.compile(r"^[ \t]*(?:import|from)[ \t]+([A-Za-z_][A-Za-z0-9_]*)", re.M)
    seen: set[str] = set()
    todo = list(files)
    while todo:
        with open(todo.pop(), encoding="utf-8") as fh:
            text = fh.read()
        for m in pat.finditer(text):
            name = m.group(1)
            if name in have and name not in seen:
                seen.add(name)
                todo.append(os.path.join(SCRIPTS, name + ".py"))
    return sorted(n + ".py" for n in seen)


def scan_poseidon(root: str) -> dict:
    """Poseidon's weights by NAME as well as by shape: any path named for it that is
    a directory or a data file, any Hugging Face cache layout.  Its name in prose
    (the wind farm's learned windows are Poseidon-T in the case studies) is
    counted, not refused."""
    paths, words = [], {}
    for full, rel in B._walk(root):
        low = rel.lower()
        if ("hf-cache" in low or "/models--" in "/" + low
                or (_POSEIDON.search(rel) and low.endswith(B.DATA_EXT))):
            paths.append(rel)
        if B._is_text(full):
            with open(full, encoding="utf-8", errors="replace") as fh:
                n = len(_POSEIDON.findall(fh.read()))
            if n:
                words[rel] = n
    for dp, dn, _fn in os.walk(root):
        dn[:] = [d for d in dn if d not in (".git", ".venv")]
        for d in dn:
            if _POSEIDON.search(d) or d.startswith("models--") or d == "hf-cache":
                paths.append(os.path.relpath(os.path.join(dp, d), root).replace(os.sep, "/"))
    return {"paths": sorted(set(paths)), "prose_mentions_by_file": words,
            "prose_mentions_total": sum(words.values())}


def scan_data(root: str) -> dict:
    """Every binary or weight- or field-shaped file, against `BINARY_ALLOWED`."""
    unexpected, allowed = [], []
    for full, rel in B._walk(root):
        if not (rel.lower().endswith(B.DATA_EXT) or not B._is_text(full)):
            continue
        row = {"file": rel, "bytes": os.path.getsize(full)}
        (allowed if any(p.match(rel) for p in BINARY_ALLOWED) else unexpected).append(row)
    return {"allowed": allowed, "unexpected": unexpected}


def _dirty(repo: str, paths: list[str], keep=lambda line: True) -> list[str]:
    """`git status --porcelain` lines for these paths.  Read without stripping the
    output: a status line's first column can be a space, and `B._git` strips."""
    r = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=all",
                        "--", *paths], capture_output=True, text=True, check=True)
    return [ln for ln in r.stdout.splitlines() if ln.strip() and keep(ln)]


def _copied_from_atlas(line: str) -> bool:
    """A `git status` line under atlas/ that the bundle actually copies."""
    path = line[3:].strip().strip('"').replace("\\", "/")
    parts = path.split("/")
    if len(parts) > 1 and parts[1] in EXCLUDE_ATLAS:
        return False
    return "bundle" not in parts[:-1] and "__pycache__" not in parts


def build(out: str, build_repo: str, allow_dirty: bool = False) -> dict:
    tests = tests_to_carry()
    scripts = script_closure([os.path.join(ROOT, "tests", t) for t in tests])
    templates = ["atlas/workbench/bundle/" + t for t in LAUNCHERS
                 if os.path.isfile(os.path.join(TEMPLATES, t))]
    dirty = (_dirty(ROOT, ["atlas"], _copied_from_atlas)
             + _dirty(ROOT, ["tests/" + t for t in tests] + ["scripts/" + s for s in scripts]
                      + templates))
    vendor_rel = "/".join(VENDOR_PACKAGE)
    dirty_build = _dirty(build_repo, [vendor_rel, "LICENSE"])
    if (dirty or dirty_build) and not allow_dirty:
        raise SystemExit(
            "these paths the bundle copies have uncommitted changes, so no commit "
            "would name what was copied:\n  " + "\n  ".join(dirty + dirty_build)
            + "\ncommit them, or pass --allow-dirty for a scratch build")

    if os.path.isdir(out):
        B._rmtree(out)
    os.makedirs(out)

    # 1. the framework and the workbench -------------------------------------
    os.makedirs(os.path.join(out, "atlas"))
    src_atlas = os.path.join(ROOT, "atlas")
    for name in sorted(os.listdir(src_atlas)):
        s = os.path.join(src_atlas, name)
        if name in EXCLUDE_ATLAS or name in ("__pycache__", "bundle"):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(out, "atlas", name), ignore=IGNORE)
        elif not name.endswith(".pyc"):
            shutil.copyfile(s, os.path.join(out, "atlas", name))
    for probe in ("atlas/__init__.py", "atlas/workbench/__main__.py",
                  "atlas/workbench/app.py", "atlas/cases/window_ns.py",
                  "atlas/cases/wake_array.py"):
        if not os.path.isfile(os.path.join(out, *probe.split("/"))):
            raise SystemExit("the bundle is missing %s" % probe)

    # 2. the build repository's one package, where its loader looks --------------
    src = os.path.join(build_repo, *VENDOR_PACKAGE)
    if not os.path.isfile(os.path.join(src, "__init__.py")):
        raise SystemExit("the wind farm's solver package is not at %r; set ATLAS_BUILD_REPO "
                         "or pass --build-repo" % src)
    dst = os.path.join(out, "vendor", *VENDOR_PACKAGE)
    shutil.copytree(src, dst, ignore=IGNORE)
    for probe in VENDOR_PROBES:
        if not os.path.isfile(os.path.join(dst, probe)):
            raise SystemExit("the bundle is missing vendor/%s/%s -- the build repo's layout is "
                             "not what the loader expects" % (vendor_rel, probe))
    # the build repository is MIT-licensed, and MIT asks that its notice travel
    # with every copy of its code
    lic = os.path.join(build_repo, "LICENSE")
    if not os.path.isfile(lic):
        raise SystemExit("the build repository has no LICENSE at %r to carry with its code"
                         % lic)
    shutil.copyfile(lic, os.path.join(out, "vendor", "LICENSE"))

    # 3. the tests, their helper, and the scripts they import ----------------------
    os.makedirs(os.path.join(out, "tests"))
    for t in tests:
        shutil.copyfile(os.path.join(ROOT, "tests", t), os.path.join(out, "tests", t))
    os.makedirs(os.path.join(out, "scripts"))
    for s in scripts:
        shutil.copyfile(os.path.join(SCRIPTS, s), os.path.join(out, "scripts", s))

    # 4. the launchers and the paperwork ----------------------------------------------
    for t, d in LAUNCHERS.items():
        s = os.path.join(TEMPLATES, t)
        if not os.path.isfile(s):
            if t in OPTIONAL_TEMPLATES:
                continue
            raise SystemExit("the template %s is missing" % s)
        shutil.copyfile(s, os.path.join(out, d))
    os.chmod(os.path.join(out, "run.sh"), 0o755)

    # 5. what this copy was taken from ------------------------------------------------
    text = open(os.path.join(TEMPLATES, "run.sh"), encoding="utf-8").read()

    def pin(name):
        m = re.search(r'^%s="([^"]+)"' % name, text, re.M)
        return m.group(1) if m else "?"

    lines = [
        "assembled          %s" % dt.date.today().isoformat(),
        "",
        "nonidino/Atlas                       %s%s"
        % (B._commit_of(ROOT), "   (DIRTY: %d paths uncommitted)" % len(dirty) if dirty else ""),
        "nonidino/physics-foundation-model    %s%s   (vendor/%s only)"
        % (B._commit_of(build_repo),
           "   (DIRTY: %d paths uncommitted)" % len(dirty_build) if dirty_build else "",
           vendor_rel),
        "",
        "python             3.10, 3.11 or 3.12",
        "pip                %s" % pin("PIP"),
        "torch (optional)   %s   CPU-only wheel on Windows and Linux; PyPI on Apple-silicon "
        "macOS; none for Intel macOS" % pin("TORCH"),
        "gmsh (optional)    %s   from PyPI, GPL-2.0-or-later: installed on the machine, "
        "never shipped here" % pin("GMSH"),
        "",
    ]
    with open(os.path.join(out, "SOURCE_COMMITS"), "w", newline="\n") as fh:
        fh.write("\n".join(lines))

    # 6. what must not be here -------------------------------------------------------
    nb = B.scan_unlicensed_checkpoint(out)
    pos = scan_poseidon(out)
    data = scan_data(out)
    scot = B.scan_scot(out)
    problems = []
    if nb["files_or_dirs_named_for_it"] or nb["uses"]:
        problems.append("NeuberNet is reachable: %s"
                        % (nb["files_or_dirs_named_for_it"] + nb["uses"]))
    if pos["paths"]:
        problems.append("Poseidon's weights or a model cache: %s" % pos["paths"])
    if data["unexpected"]:
        problems.append("binary or weight-shaped files, and none is allowed: %s"
                        % data["unexpected"])
    if scot:
        problems.append("scOT is vendored, and it has no licence: %s" % scot)
    if os.path.exists(os.path.join(out, "out")):
        problems.append("out/ is in the bundle; it ships nothing there")
    report = {
        "out": out, "branch": BRANCH, "built": dt.datetime.now().isoformat(timespec="seconds"),
        "atlas_commit": B._commit_of(ROOT), "dirty": dirty,
        "build_repo": build_repo, "build_repo_commit": B._commit_of(build_repo),
        "build_repo_dirty": dirty_build, "vendored": "vendor/" + vendor_rel,
        "excluded_from_atlas": EXCLUDE_ATLAS, "tests": tests, "scripts": scripts,
        "sizes_mb": {"total": B._size_mb(out),
                     "atlas": B._size_mb(os.path.join(out, "atlas")),
                     "vendor": B._size_mb(os.path.join(out, "vendor")),
                     "tests": B._size_mb(os.path.join(out, "tests")),
                     "scripts": B._size_mb(os.path.join(out, "scripts"))},
        "files": sum(1 for _ in B._walk(out)),
        "scan_neubernet": nb, "scan_poseidon": pos, "scan_data": data, "scan_scot": scot,
        "problems": problems,
    }
    print("  built %s" % out)
    print("  %d files, %.1f MB: atlas %.1f, vendor %.2f, tests %.2f, scripts %.2f"
          % ((report["files"],) + tuple(report["sizes_mb"][k] for k in
                                       ("total", "atlas", "vendor", "tests", "scripts"))))
    print("  %d tests; scripts they import: %s" % (len(tests), ", ".join(scripts) or "none"))
    print("  NeuberNet: %d files, %d uses; its name in prose %d times in %d files (reported, "
          "not refused)" % (len(nb["files_or_dirs_named_for_it"]), len(nb["uses"]),
                            nb["prose_mentions_total"], len(nb["prose_mentions_by_file"])))
    print("  Poseidon: %d weight or cache paths; its name in prose %d times in %d files"
          % (len(pos["paths"]), pos["prose_mentions_total"],
             len(pos["prose_mentions_by_file"])))
    print("  binary files: %d allowed, %d unexpected; scOT dirs: %d"
          % (len(data["allowed"]), len(data["unexpected"]), len(scot)))
    if problems:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(problems))
    return report


def commit(out: str, message: str) -> str:
    """Make the local orphan branch from the built directory, replaced wholesale.

    It shares no history with `atlas-0.1` by design: a copy that accumulated its
    own history would invite someone to fix something in it.
    """
    def git(*a):
        return subprocess.run(["git", "-C", out, *a], check=True, capture_output=True,
                              text=True).stdout.strip()

    def cfg(key: str, fallback: str) -> str:
        v = subprocess.run(["git", "-C", ROOT, "config", key], capture_output=True,
                           text=True).stdout.strip()
        return v or fallback

    git("init", "-q", "-b", BRANCH)
    git("config", "core.longpaths", "true")
    # Line endings normalised exactly as atlas-0.1's are, so a file's blob here is
    # its blob there: the verifier's identity check compares blob hashes.
    git("config", "core.autocrlf", cfg("core.autocrlf", "false"))
    git("config", "user.email", cfg("user.email", "noreply@example.com"))
    git("config", "user.name", cfg("user.name", "atlas"))
    git("add", "-A")
    # `os.chmod` does nothing on Windows -- NTFS has no execute bit -- so the mode
    # is set in the INDEX, or `./run.sh` on a Mac fails "Permission denied" (126).
    git("update-index", "--chmod=+x", "run.sh")
    git("commit", "-q", "-m", message)
    head = git("rev-parse", "HEAD")
    mode = git("ls-files", "-s", "run.sh").split()
    print("  committed %s on %s in %s" % (head[:12], BRANCH, out))
    print("  run.sh in the index: %s (must be 100755)" % (mode[0] if mode else "??"))
    if not mode or mode[0] != "100755":
        raise SystemExit("run.sh is not 100755 in the index; a Mac or Linux clone would "
                         "fail with exit 126")
    # **What a clone would actually get.**  `git add` on an ignored path says
    # nothing, so an ignore rule that drops a carried file leaves a branch that
    # works here and is missing that file there.  Asked of the index.
    tracked = set(git("ls-files").splitlines())
    on_disk = {rel for _full, rel in B._walk(out)}
    lost = sorted(on_disk - tracked)
    if lost:
        raise SystemExit("these are in the bundle directory but NOT in the branch, so a "
                         "clone would not get them:\n  " + "\n  ".join(lost[:40]))
    print("  all %d files tracked" % len(tracked))
    print("  to publish (not before the website launches, O9):  git -C %s push <remote> %s"
          % (out, BRANCH))
    return head


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--build-repo",
                    default=os.environ.get("ATLAS_BUILD_REPO",
                                           os.path.join(os.path.expanduser("~"),
                                                        "physics-foundation-model")))
    ap.add_argument("--commit", action="store_true",
                    help="also make the orphan branch commit inside --out")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="build even if copied paths have uncommitted changes (recorded in "
                         "SOURCE_COMMITS as DIRTY)")
    ap.add_argument("--report", default=os.path.join(RECORDS, "build.json"),
                    help="where the build report is written")
    ap.add_argument("-m", "--message",
                    default="Atlas Workbench: one-command install (Windows, macOS, Linux)")
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
