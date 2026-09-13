"""PoC 3 phase 5 -- verify the `poc3-racelab-demo` bundle ON A FRESH CLONE.

    python scripts/verify_racelab_bundle.py --bundle out/bundle3 --clone C:/poc3v
    python scripts/verify_racelab_bundle.py ... --launch     # also run the launcher

Inspecting the build directory proves nothing about what a user gets: the
index, not the working tree, decides the mode `run.sh` is checked out with,
and `.gitattributes`, not the author's editor, decides its line endings.  So
every check here is made on a `git clone` of the branch into a directory that
did not exist:

  1. **the mode** -- `run.sh` is ``100755`` in the clone's index (a Windows
     build records ``100644`` unless told, and the first command on Linux then
     dies with exit 126);
  2. **the line endings as checked out** -- no carriage return anywhere in
     `run.sh`, which bash would read as part of the interpreter's name;
  3. **character identity** -- every file the bundle copied from `atlas-0.1`,
     compared by git BLOB HASH against the commit `SOURCE_COMMITS` names, and
     every vendored solver file against the build repository's commit.  Equal
     blob hashes are equal bytes; this is the check that makes the bundle's
     tests the same tests;
  4. **the scans** -- the builder's own: nothing of the unlicensed structural
     checkpoint, no stray weights or fields, no vendored `scOT`;
  5. with ``--launch``, **the launcher itself**: ``--check`` in the clone, with
     its exit code and log kept.

Writes ``<record>`` (default ``out/racelab_bundle/verify_<platform>.json``).
It never pushes and it never deletes a directory it did not create: ``--clone``
must not exist.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BRANCH = "poc3-racelab-demo"

#: Where each launcher file in the bundle comes from in atlas-0.1.
LAUNCHER_SOURCES = {
    "run.py": "atlas/demo_racelab/bundle/run.py",
    "run.sh": "atlas/demo_racelab/bundle/run.sh",
    "run.cmd": "atlas/demo_racelab/bundle/run.cmd",
    "requirements.txt": "atlas/demo_racelab/bundle/requirements.txt",
    "constraints.txt": "atlas/demo_racelab/bundle/constraints.txt",
    "conftest.py": "atlas/demo_racelab/bundle/conftest.py",
    "README.md": "atlas/demo_racelab/bundle/README.md",
    "PROVENANCE.md": "atlas/demo_racelab/bundle/PROVENANCE.md",
    ".gitignore": "atlas/demo_racelab/bundle/gitignore",
    ".gitattributes": "atlas/demo_racelab/bundle/gitattributes",
    "vendor/POSEIDON-T-LICENCE.md":
        "atlas/demo_racelab/bundle/POSEIDON-T-LICENCE.md",
}


def git(repo: str, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, *args],
                       capture_output=True, text=True, check=False)
    if check and r.returncode:
        raise RuntimeError("git %s failed in %s: %s" % (" ".join(args), repo,
                                                         r.stderr.strip()))
    return r.stdout


def tree(repo: str, rev: str) -> dict:
    """``path -> (mode, blob)`` for every file at ``rev``."""
    out = {}
    for ln in git(repo, "ls-tree", "-r", "-z", rev).split("\0"):
        if not ln:
            continue
        meta, path = ln.split("\t", 1)
        mode, kind, blob = meta.split()
        if kind == "blob":
            out[path] = (mode, blob)
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_commits(path: str) -> dict:
    text = open(path, encoding="utf-8").read()

    def after(label):
        m = re.search(re.escape(label) + r"\s+([0-9a-f]{40})", text)
        return m.group(1) if m else None

    return {"atlas": after("nonidino/Atlas"),
            "build_repo": after("nonidino/physics-foundation-model"),
            "snapshot": after("camlab-ethz/Poseidon-T (weights)"),
            "weights_sha256": (re.search(r"model\.safetensors sha256\s+([0-9a-f]{64})",
                                         text) or [None, None])[1],
            "car": (re.search(r"the car\s+([0-9a-f]{64})", text) or [None, None])[1],
            "dirty": "DIRTY" in text,
            "text": text}


def identity(clone: str, build_repo: str, sc: dict) -> dict:
    """Every copied file against the commit it was copied from, by blob hash."""
    mine = tree(clone, "HEAD")
    up = tree(ROOT, sc["atlas"])
    ours = {}
    try:
        br = tree(build_repo, sc["build_repo"])
    except Exception as exc:                                 # pragma: no cover
        br, ours["build_repo_unreadable"] = {}, str(exc)
    same, differ, unmatched = [], [], []
    groups = {"tests": [0, 0], "atlas": [0, 0], "scripts": [0, 0],
              "wiki": [0, 0], "out": [0, 0], "vendor/src": [0, 0],
              "launchers": [0, 0]}

    def tally(group, ok):
        groups[group][0 if ok else 1] += 1

    for path, (_mode, blob) in sorted(mine.items()):
        if path in LAUNCHER_SOURCES:
            src = LAUNCHER_SOURCES[path]
            if src in up:
                ok = up[src][1] == blob
                (same if ok else differ).append(path)
                tally("launchers", ok)
            else:
                unmatched.append(path)
            continue
        if path.startswith("vendor/src/"):
            src = path[len("vendor/"):]
            if src in br:
                ok = br[src][1] == blob
                (same if ok else differ).append(path)
                tally("vendor/src", ok)
            else:
                unmatched.append(path)
            continue
        head = path.split("/", 1)[0]
        if head in ("tests", "atlas", "scripts", "wiki", "out") and path in up:
            ok = up[path][1] == blob
            (same if ok else differ).append(path)
            tally(head, ok)
        else:
            unmatched.append(path)
    # the two binaries atlas-0.1 does not track are compared by content
    extra = {}
    field = os.path.join(clone, "out", "racelab5", "cache", "settled.npz")
    local = os.path.join(ROOT, "out", "racelab5", "cache", "settled.npz")
    if os.path.isfile(field) and os.path.isfile(local):
        extra["settled.npz_identical_to_atlas_0.1_s_cache"] = (
            sha256(field) == sha256(local))
    weights = [p for p in mine if p.endswith("model.safetensors")]
    if weights and sc.get("weights_sha256"):
        extra["weights_sha256_matches_SOURCE_COMMITS"] = (
            sha256(os.path.join(clone, *weights[0].split("/")))
            == sc["weights_sha256"])
    tests_up = sorted(p for p in up if re.match(r"tests/test_tier5[1-7]_", p))
    tests_here = sorted(p for p in mine if p.startswith("tests/"))
    return {"identical": len(same), "different": differ,
            "not_from_either_repo": unmatched, "by_group": groups,
            "tests_in_the_bundle": tests_here,
            "poc3_tests_upstream": tests_up,
            "every_poc3_test_is_carried": tests_up == tests_here,
            **extra, **ours}


def launch(clone: str, log: str, args: list, env_extra: dict | None = None,
           timeout: int = 7200) -> dict:
    env = dict(os.environ)
    for k in ("ATLAS_BUILD_REPO", "HF_HOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    if os.name == "nt":
        cmd = ["cmd", "/c", os.path.join(clone, "run.cmd"), *args]
    else:
        cmd = ["bash", "-c", "cd %s && ./run.sh %s" % (clone, " ".join(args))]
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        r = subprocess.run(cmd, cwd=clone, env=env, stdout=fh,
                           stderr=subprocess.STDOUT, timeout=timeout)
    text = open(log, encoding="utf-8", errors="replace").read()
    return {"argv": args, "exit_code": r.returncode,
            "wall_s": time.time() - t0, "log": log,
            "tail": text[-4000:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default=os.path.join(ROOT, "out", "bundle3"))
    ap.add_argument("--clone", required=True,
                    help="a directory that does not exist yet")
    ap.add_argument("--build-repo",
                    default=os.environ.get(
                        "ATLAS_BUILD_REPO",
                        os.path.join(os.path.expanduser("~"),
                                     "physics-foundation-model")))
    ap.add_argument("--launch", action="store_true",
                    help="also run the launcher's --check in the clone")
    ap.add_argument("--record", default=None)
    args = ap.parse_args(argv)

    clone = os.path.abspath(args.clone)
    if os.path.exists(clone):
        raise SystemExit("%s exists; this script only verifies a clone it made "
                         "itself" % clone)
    plat = "%s-%s" % (platform.system().lower(), platform.machine().lower())
    rec = args.record or os.path.join(ROOT, "out", "racelab_bundle",
                                      "verify_%s.json" % plat)
    out: dict = {"at": dt.datetime.now().isoformat(timespec="seconds"),
                 "platform": plat, "python": sys.version.split()[0],
                 "bundle": os.path.abspath(args.bundle), "clone": clone}

    t0 = time.time()
    r = subprocess.run(["git", "-c", "safe.directory=*", "clone", "--branch",
                        BRANCH, "--single-branch", os.path.abspath(args.bundle),
                        clone], capture_output=True, text=True)
    out["clone_exit_code"] = r.returncode
    out["clone_s"] = time.time() - t0
    if r.returncode:
        out["clone_stderr"] = r.stderr[-2000:]
        _write(rec, out)
        raise SystemExit("git clone failed: %s" % r.stderr)
    out["head"] = git(clone, "rev-parse", "HEAD").strip()

    ls = git(clone, "ls-files", "-s", "run.sh").split()
    out["run_sh_mode_in_the_index"] = ls[0] if ls else None
    with open(os.path.join(clone, "run.sh"), "rb") as fh:
        sh = fh.read()
    with open(os.path.join(clone, "run.cmd"), "rb") as fh:
        cmd = fh.read()
    out["run_sh_has_a_carriage_return"] = b"\r" in sh
    out["run_sh_shebang"] = sh.split(b"\n", 1)[0].decode("ascii", "replace")
    out["run_cmd_lines_end_crlf"] = (cmd.count(b"\r\n") == cmd.count(b"\n")
                                     and cmd.count(b"\n") > 0)
    if os.name != "nt":
        out["run_sh_is_executable_on_disk"] = os.access(
            os.path.join(clone, "run.sh"), os.X_OK)

    sc = source_commits(os.path.join(clone, "SOURCE_COMMITS"))
    out["source_commits"] = {k: v for k, v in sc.items() if k != "text"}
    out["identity"] = identity(clone, args.build_repo, sc)

    import build_racelab_bundle as B
    out["scan_unlicensed_checkpoint"] = B.scan_unlicensed_checkpoint(clone)
    out["scan_data"] = B.scan_data(clone)
    out["scan_scot"] = B.scan_scot(clone)

    if args.launch:
        out["launch_check"] = launch(
            clone, os.path.join(os.path.dirname(rec), "launch_check_%s.log"
                                % plat), ["--check"])
    ok = (out["run_sh_mode_in_the_index"] == "100755"
          and not out["run_sh_has_a_carriage_return"]
          and not out["identity"]["different"]
          and out["identity"]["every_poc3_test_is_carried"]
          and not out["scan_unlicensed_checkpoint"]["uses"]
          and not out["scan_unlicensed_checkpoint"]["files_or_dirs_named_for_it"]
          and not out["scan_data"]["unexpected"]
          and not out["scan_scot"])
    out["static_checks_pass"] = ok
    _write(rec, out)
    print("  clone %s (%s)" % (clone, out["head"][:12]))
    print("  run.sh mode %s, carriage return %s, run.cmd CRLF %s"
          % (out["run_sh_mode_in_the_index"], out["run_sh_has_a_carriage_return"],
             out["run_cmd_lines_end_crlf"]))
    idn = out["identity"]
    print("  identical %d, different %d, from neither repo %d; tests carried %s"
          % (idn["identical"], len(idn["different"]),
             len(idn["not_from_either_repo"]), idn["every_poc3_test_is_carried"]))
    for g, (a, b) in idn["by_group"].items():
        print("    %-10s %4d identical, %d different" % (g, a, b))
    nb = out["scan_unlicensed_checkpoint"]
    print("  unlicensed checkpoint: %d uses, %d paths; prose mentions %d"
          % (len(nb["uses"]), len(nb["files_or_dirs_named_for_it"]),
             nb["prose_mentions_total"]))
    if args.launch:
        lc = out["launch_check"]
        print("  launcher --check: exit %s in %.0f s (log %s)"
              % (lc["exit_code"], lc["wall_s"], lc["log"]))
    print("  static checks pass: %s  ->  %s" % (ok, rec))
    return 0 if ok else 1


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    for k in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                 # OneDrive holds fresh files
            time.sleep(0.4 * (k + 1))
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
