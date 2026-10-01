"""Demo step 7 (O2) -- verify the Atlas Workbench's install ON A FRESH CLONE.

    python scripts/verify_workbench_bundle.py --clone C:/Users/Nauni/wbv          # static
    python scripts/verify_workbench_bundle.py --clone ... --launch --tests        # and run it
    python scripts/verify_workbench_bundle.py --existing C:/Users/Nauni/wbv \
        --no-torch --offline --freeze                                             # an installed clone

Inspecting the build directory proves nothing about what a stranger gets: the
index, not the working tree, decides the mode `run.sh` is checked out with, and
`.gitattributes`, not an editor, decides its line endings.  So every check is
made on a `git clone` of the branch into a directory that did not exist
(PoC 3's verifier, whose checks this repeats for the workbench):

  1. **the mode**: `run.sh` is ``100755`` in the clone's index;
  2. **the line endings as checked out**: no carriage return in `run.sh`,
     CRLF throughout `run.cmd`;
  3. **character identity**: every file copied from `atlas-0.1`, compared by
     git blob hash with the commit `SOURCE_COMMITS` names, and the vendored
     package with the build repository's commit; every workbench test at that
     commit is carried;
  4. **the scans**: the builder's own (NeuberNet, Poseidon's weights, any
     binary file, a vendored scOT);
  5. ``--launch``: the launcher's ``--check`` in the clone, as a stranger would
     run it, with none of this machine's atlas or Python settings leaking in;
  6. ``--tests``: the bundle's tests, in the clone's own environment;
  7. ``--no-torch``: the self-test with torch made unimportable (the classical
     arms must all run: exit 3, every type ok);
  8. ``--offline``: the self-test with every connection off this machine
     refused and counted (loopback is allowed: the page check talks to the
     workbench's own server); zero attempts, and it must still pass;
  9. ``--freeze``: ``pip freeze --all`` of the clone's environment, the size of
     `.venv` and its deepest path;
 10. ``--optional-fails`` (last, since it changes the environment): torch and
     Gmsh uninstalled and the launcher run with pip pointed at a closed port, so
     both optional installs fail for real; the launcher must say so for each,
     write neither stamp, and its self-test still pass every type (exit 3).

Writes ``out/workbench/records/installer/verify_<platform>[_<tag>].json``.  It
never pushes, and it never deletes a directory it did not create.
"""
from __future__ import annotations

import argparse
import datetime as dt
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

import build_workbench_bundle as W                                     # noqa: E402

BRANCH = W.BRANCH
RECORDS = W.RECORDS
LOGS = os.path.join(RECORDS, "logs")

#: bundle path -> its source in atlas-0.1 (the launcher templates)
LAUNCHER_SOURCES = {d: "atlas/workbench/bundle/" + t for t, d in W.LAUNCHERS.items()}
#: written by the builder, so from neither repository
GENERATED = {"SOURCE_COMMITS"}

#: environment variables of this machine that must not reach the clone
_LEAKS = ("ATLAS_BUILD_REPO", "PYTHONPATH", "VIRTUAL_ENV", "PYTHON", "KMP_DUPLICATE_LIB_OK",
          "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV")


def git(repo: str, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", repo, *args],
                       capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError("git %s failed in %s: %s" % (" ".join(args), repo, r.stderr.strip()))
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


def source_commits(path: str) -> dict:
    text = open(path, encoding="utf-8").read()

    def after(label):
        m = re.search(re.escape(label) + r"\s+([0-9a-f]{40})", text)
        return m.group(1) if m else None

    return {"atlas": after("nonidino/Atlas"),
            "build_repo": after("nonidino/physics-foundation-model"),
            "dirty": "DIRTY" in text, "text": text}


def identity(clone: str, build_repo: str, sc: dict) -> dict:
    """Every copied file against the commit it was copied from, by blob hash."""
    mine = tree(clone, "HEAD")
    up = tree(ROOT, sc["atlas"])
    br = tree(build_repo, sc["build_repo"])
    same, differ, unmatched = [], [], []
    groups: dict[str, list[int]] = {}

    def tally(g, ok, path):
        groups.setdefault(g, [0, 0])[0 if ok else 1] += 1
        (same if ok else differ).append(path)

    for path, (_mode, blob) in sorted(mine.items()):
        if path in LAUNCHER_SOURCES:
            src = LAUNCHER_SOURCES[path]
            if src in up:
                tally("launchers", up[src][1] == blob, path)
            else:
                unmatched.append(path)
        elif path == "vendor/LICENSE":
            tally("vendor", "LICENSE" in br and br["LICENSE"][1] == blob, path)
        elif path.startswith("vendor/"):
            src = path[len("vendor/"):]
            if src in br:
                tally("vendor", br[src][1] == blob, path)
            else:
                unmatched.append(path)
        elif path.split("/", 1)[0] in ("atlas", "tests", "scripts") and path in up:
            tally(path.split("/", 1)[0], up[path][1] == blob, path)
        elif path in GENERATED:
            continue
        else:
            unmatched.append(path)
    tests_up = sorted(p for p in up if p.startswith("tests/")
                      and W.TEST_PATTERN.match(p.split("/", 1)[1]))
    tests_here = sorted(p for p in mine if p.startswith("tests/")
                        and W.TEST_PATTERN.match(p.split("/", 1)[1]))
    return {"identical": len(same), "different": differ,
            "not_from_either_repo": unmatched, "by_group": groups,
            "workbench_tests_upstream": len(tests_up),
            "workbench_tests_carried": len(tests_here),
            "every_workbench_test_is_carried": tests_up == tests_here}


def clean_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    for k in _LEAKS:
        env.pop(k, None)
    # an activated conda environment puts its own python first on PATH, which is
    # a choice a stranger's shell would not have made: leave PATH as it is, but
    # say which python the launcher found (it prints it)
    env.update(extra or {})
    return env


def venv_python(clone: str) -> str:
    win = os.name == "nt"
    return os.path.join(clone, ".venv", "Scripts" if win else "bin",
                        "python.exe" if win else "python")


def run_logged(cmd, cwd, env, log, timeout=7200) -> dict:
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        try:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    text = open(log, encoding="utf-8", errors="replace").read()
    return {"argv": cmd, "exit_code": code, "wall_s": round(time.time() - t0, 1),
            "log": log, "tail": text[-5000:]}


def launch(clone: str, log: str, args: list, python: str | None) -> dict:
    env = clean_env({"PYTHON": python} if python else None)
    if os.name == "nt":
        cmd = ["cmd", "/c", os.path.join(clone, "run.cmd"), *args]
    else:
        cmd = ["bash", "-c", "cd '%s' && ./run.sh %s" % (clone, " ".join(args))]
    out = run_logged(cmd, clone, env, log)
    out["python_given"] = python
    return out


#: Run in the CLONE's interpreter with the clone as the working directory: the
#: self-test with torch (and anything under it) made unimportable.
_NO_TORCH = r'''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked by verify_workbench_bundle --no-torch")
        return None
sys.meta_path.insert(0, Block())
sys.argv = ["run.py", "--check"]
try:
    runpy.run_path("run.py", run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 1
    print("\nTORCH_IMPORTED", "torch" in sys.modules)
    sys.exit(code)
'''

#: The self-test with every connection off this machine refused and counted.
_OFFLINE = r'''
import json, os, runpy, socket, sys
attempts, loopback = [], []
LOCAL = ("127.0.0.1", "localhost", "::1")
_connect, _connect_ex, _create = (socket.socket.connect, socket.socket.connect_ex,
                                  socket.create_connection)
def _host(address):
    return address[0] if isinstance(address, tuple) and address else str(address)
def guard(real, self_or_addr, address, *a, **k):
    if _host(address) in LOCAL:
        loopback.append(repr(address)[:100])
        return real(self_or_addr, address, *a, **k)
    attempts.append(repr(address)[:200])
    raise OSError("network disabled by verify_workbench_bundle --offline")
socket.socket.connect = lambda self, address: guard(_connect, self, address)
socket.socket.connect_ex = lambda self, address: guard(_connect_ex, self, address)
socket.create_connection = lambda address, *a, **k: (
    loopback.append(repr(address)[:100]) or _create(address, *a, **k)
    if _host(address) in LOCAL else guard(None, None, address))
sys.argv = ["run.py", "--check"]
code = 0
try:
    runpy.run_path("run.py", run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 1
with open(os.environ["ATTEMPTS_OUT"], "w") as fh:
    json.dump({"exit_code": code, "connection_attempts": attempts,
               "loopback_connections": len(loopback)}, fh)
sys.exit(code)
'''


def self_test_examples(clone: str) -> list[str]:
    """The example each type's self-test marches, read from the clone's own run.py."""
    text = open(os.path.join(clone, "run.py"), encoding="utf-8").read()
    return re.findall(r'^\s*\("[a-z0-9-]+-2d", "([a-z0-9-]+)", \d+\),', text, re.M)


def _type_lines(text: str, keys: list[str]) -> dict:
    """The self-test's line for each type, by its example's key: key -> passed?
    A type whose line is missing (an exception, a crash) is reported as False."""
    out = {k: False for k in keys}
    for ln in text.splitlines():
        for k in keys:
            if re.search(r"\s%s\s+compile\s" % re.escape(k), ln):
                out[k] = ln.startswith("    ok ")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default=W.OUT)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--clone", help="a directory that does not exist yet: clone into it")
    g.add_argument("--existing", help="a clone this script made earlier, already launched")
    ap.add_argument("--build-repo",
                    default=os.environ.get("ATLAS_BUILD_REPO",
                                           os.path.join(os.path.expanduser("~"),
                                                        "physics-foundation-model")))
    ap.add_argument("--launch", action="store_true", help="run the launcher's --check")
    ap.add_argument("--tests", action="store_true", help="run the bundle's tests")
    ap.add_argument("--no-torch", action="store_true", help="the self-test without torch")
    ap.add_argument("--offline", action="store_true", help="the self-test with no network")
    ap.add_argument("--freeze", action="store_true", help="record pip freeze --all")
    ap.add_argument("--optional-fails", action="store_true",
                    help="LAST: uninstall torch and gmsh from the clone's venv and run the "
                         "launcher with no network, so both optional installs fail for real")
    ap.add_argument("--python", default=None, help="PYTHON= for the launcher")
    ap.add_argument("--tag", default="", help="names this record and its logs")
    ap.add_argument("--note", default="", help="the machine's state, in words")
    args = ap.parse_args(argv)

    plat = "%s-%s" % (platform.system().lower(), platform.machine().lower())
    tag = ("_" + args.tag) if args.tag else ""
    os.makedirs(RECORDS, exist_ok=True)
    rec_path = os.path.join(RECORDS, "verify_%s%s.json" % (plat, tag))
    out: dict = {"at": dt.datetime.now().isoformat(timespec="seconds"), "platform": plat,
                 "verifier_python": sys.version.split()[0], "note": args.note,
                 "bundle": os.path.abspath(args.bundle)}

    if args.clone:
        clone = os.path.abspath(args.clone)
        if os.path.exists(clone):
            raise SystemExit("%s exists; this script only verifies a clone it makes itself"
                             % clone)
        t0 = time.time()
        r = subprocess.run(["git", "-c", "safe.directory=*", "clone", "--branch", BRANCH,
                            "--single-branch", os.path.abspath(args.bundle), clone],
                           capture_output=True, text=True)
        out["clone_exit_code"], out["clone_s"] = r.returncode, round(time.time() - t0, 1)
        if r.returncode:
            out["clone_stderr"] = r.stderr[-2000:]
            _write(rec_path, out)
            raise SystemExit("git clone failed: %s" % r.stderr)
    else:
        clone = os.path.abspath(args.existing)
        if not os.path.isfile(os.path.join(clone, "SOURCE_COMMITS")):
            raise SystemExit("%s is not a clone of the bundle" % clone)
    out["clone"] = clone
    out["head"] = git(clone, "rev-parse", "HEAD").strip()

    # 1-2. what the checkout looks like --------------------------------------------
    ls = git(clone, "ls-files", "-s", "run.sh").split()
    out["run_sh_mode_in_the_index"] = ls[0] if ls else None
    sh = open(os.path.join(clone, "run.sh"), "rb").read()
    cmd = open(os.path.join(clone, "run.cmd"), "rb").read()
    out["run_sh_has_a_carriage_return"] = b"\r" in sh
    out["run_sh_shebang"] = sh.split(b"\n", 1)[0].decode("ascii", "replace")
    out["run_cmd_lines_end_crlf"] = (cmd.count(b"\r\n") == cmd.count(b"\n")
                                     and cmd.count(b"\n") > 0)
    if os.name != "nt":
        out["run_sh_is_executable_on_disk"] = os.access(os.path.join(clone, "run.sh"), os.X_OK)
    files = [p for p in git(clone, "ls-files").splitlines() if p]
    out["files_in_branch"] = len(files)
    deepest = max(files, key=len)
    out["deepest_path_in_branch"] = {"path": deepest, "chars": len(deepest)}

    # 3-4. identity and the scans ----------------------------------------------------
    sc = source_commits(os.path.join(clone, "SOURCE_COMMITS"))
    out["source_commits"] = {k: v for k, v in sc.items() if k != "text"}
    out["identity"] = identity(clone, args.build_repo, sc)
    nb = W.B.scan_unlicensed_checkpoint(clone)
    pos = W.scan_poseidon(clone)
    data = W.scan_data(clone)
    scot = W.B.scan_scot(clone)
    out["scan_neubernet"] = {"named": nb["files_or_dirs_named_for_it"], "uses": nb["uses"],
                             "prose_mentions": nb["prose_mentions_total"]}
    out["scan_poseidon"] = {"paths": pos["paths"],
                            "prose_mentions": pos["prose_mentions_total"]}
    out["scan_data"] = data
    out["scan_scot"] = scot
    idn = out["identity"]
    static_ok = (out["run_sh_mode_in_the_index"] == "100755"
                 and not out["run_sh_has_a_carriage_return"]
                 and out["run_cmd_lines_end_crlf"]
                 and not idn["different"] and not idn["not_from_either_repo"]
                 and idn["every_workbench_test_is_carried"]
                 and not nb["uses"] and not nb["files_or_dirs_named_for_it"]
                 and not pos["paths"] and not data["unexpected"] and not scot)
    out["static_checks_pass"] = static_ok
    print("  clone %s (%s), %d files, deepest path %d chars"
          % (clone, out["head"][:12], len(files), len(deepest)))
    print("  run.sh mode %s, carriage return %s; run.cmd CRLF %s"
          % (out["run_sh_mode_in_the_index"], out["run_sh_has_a_carriage_return"],
             out["run_cmd_lines_end_crlf"]))
    print("  identical %d, different %d, from neither repo %d; workbench tests carried "
          "%d of %d" % (idn["identical"], len(idn["different"]),
                        len(idn["not_from_either_repo"]), idn["workbench_tests_carried"],
                        idn["workbench_tests_upstream"]))
    for gname, (a, b) in sorted(idn["by_group"].items()):
        print("    %-10s %4d identical, %d different" % (gname, a, b))
    print("  NeuberNet %d uses, %d paths; Poseidon %d paths; binaries %d; scOT %d"
          % (len(nb["uses"]), len(nb["files_or_dirs_named_for_it"]), len(pos["paths"]),
             len(data["unexpected"]), len(scot)))
    print("  static checks pass: %s" % static_ok)
    _write(rec_path, out)

    ok = static_ok
    #: whole logs stay local (.gitignore); each record keeps the tail of each,
    #: which is where the launcher's self-test and the summaries are
    logs = LOGS
    os.makedirs(logs, exist_ok=True)
    keys = self_test_examples(clone)
    out["self_test_examples"] = keys
    if len(keys) != 8:
        raise SystemExit("the clone's run.py names %d self-test examples, not 8: %s"
                         % (len(keys), keys))

    # 5. the launcher ----------------------------------------------------------------
    if args.launch:
        lc = launch(clone, os.path.join(logs, "launch_check_%s%s.log" % (plat, tag)),
                    ["--check"], args.python)
        lc["types"] = _type_lines(_read(lc["log"]), keys)
        out["launch_check"] = lc
        print("  launcher --check: exit %s in %.0f s; types ok %d of %d"
              % (lc["exit_code"], lc["wall_s"], sum(lc["types"].values()), len(lc["types"])))
        ok = ok and lc["exit_code"] in (0, 3) and len(lc["types"]) == 8 \
            and all(lc["types"].values())
        _write(rec_path, out)

    vpy = venv_python(clone)
    if (args.tests or args.no_torch or args.offline or args.freeze
            or args.optional_fails) and not os.path.isfile(vpy):
        raise SystemExit("%s has no installed venv; run with --launch first" % clone)

    # 6. the tests --------------------------------------------------------------------
    if args.tests:
        t = run_logged([vpy, "-m", "pytest", "tests", "-q", "-rs", "-p", "no:cacheprovider"],
                       clone, clean_env(), os.path.join(logs, "tests_%s%s.log" % (plat, tag)))
        # `-q` prints the summary bare ("318 passed, 2 skipped in 234.27s"), and
        # without -q inside a rule of "=": both are read
        m = re.findall(r"^=*\s*(\d+ (?:passed|failed|errors?|skipped)[^\n]*?) in ([\d.]+)s",
                       _read(t["log"]), re.M)
        t["summary"] = m[-1][0] if m else None
        t["skipped_because"] = re.findall(r"^SKIPPED \[\d+\] ([^\n]+)", _read(t["log"]), re.M)
        out["tests"] = t
        print("  tests: exit %s, %s (%.0f s)" % (t["exit_code"], t["summary"], t["wall_s"]))
        ok = ok and t["exit_code"] == 0
        _write(rec_path, out)

    # 7. without torch --------------------------------------------------------------
    if args.no_torch:
        n = run_logged([vpy, "-c", _NO_TORCH], clone, clean_env(),
                       os.path.join(logs, "no_torch_%s%s.log" % (plat, tag)))
        n["types"] = _type_lines(_read(n["log"]), keys)
        n["torch_imported"] = "TORCH_IMPORTED True" in n["tail"]
        n["said_torch_absent"] = bool(re.search(r"absent\s+torch", n["tail"]))
        out["no_torch"] = n
        print("  without torch: exit %s, types ok %d of %d, torch reported absent %s"
              % (n["exit_code"], sum(n["types"].values()), len(n["types"]),
                 n["said_torch_absent"]))
        ok = (ok and n["exit_code"] == 3 and len(n["types"]) == 8 and all(n["types"].values())
              and n["said_torch_absent"] and not n["torch_imported"])
        _write(rec_path, out)

    # 8. with no network --------------------------------------------------------------
    if args.offline:
        att = os.path.join(logs, "offline_attempts_%s%s.json" % (plat, tag))
        o = run_logged([vpy, "-c", _OFFLINE], clone, clean_env({"ATTEMPTS_OUT": att}),
                       os.path.join(logs, "offline_%s%s.log" % (plat, tag)))
        got = json.load(open(att)) if os.path.isfile(att) else {}
        o["connection_attempts"] = got.get("connection_attempts")
        o["loopback_connections"] = got.get("loopback_connections")
        o["types"] = _type_lines(_read(o["log"]), keys)
        out["offline"] = o
        print("  offline: exit %s, off-machine connection attempts %s, loopback %s, types "
              "ok %d of %d" % (o["exit_code"], o["connection_attempts"],
                               o["loopback_connections"], sum(o["types"].values()),
                               len(o["types"])))
        ok = (ok and o["exit_code"] in (0, 3) and o["connection_attempts"] == []
              and len(o["types"]) == 8 and all(o["types"].values()))
        _write(rec_path, out)

    # 9. the environment as installed --------------------------------------------------
    if args.freeze:
        r = subprocess.run([vpy, "-m", "pip", "freeze", "--all"], capture_output=True,
                           text=True, env=clean_env())
        out["pip_freeze_all"] = sorted(r.stdout.split())
        r2 = subprocess.run([vpy, "-c", "import sys; print(sys.version)"],
                            capture_output=True, text=True, env=clean_env())
        out["venv_python"] = r2.stdout.strip()
        size = 0
        longest = ("", 0)
        venv = os.path.join(clone, ".venv")
        for dp, _dn, fn in os.walk(venv):
            for f in fn:
                p = os.path.join(dp, f)
                try:
                    size += os.path.getsize(p)
                except OSError:
                    pass
                rel = os.path.relpath(p, clone)
                if len(rel) > longest[1]:
                    longest = (rel, len(rel))
        out["venv_mb"] = round(size / 1e6, 1)
        out["deepest_path_in_venv"] = {"path": longest[0].replace(os.sep, "/"),
                                       "chars": longest[1]}
        print("  freeze: %d packages; .venv %.0f MB; deepest path in it %d chars"
              % (len(out["pip_freeze_all"]), out["venv_mb"], longest[1]))
        _write(rec_path, out)

    # 10. the optional installs failing for real ----------------------------------
    # Last, because it takes torch and Gmsh out of the clone's environment.  pip is
    # pointed at a closed local port, which is what no network looks like to it;
    # the packages step is already stamped, so only the two optional steps try.
    if args.optional_fails:
        un = subprocess.run([vpy, "-m", "pip", "uninstall", "-y", "torch", "gmsh"],
                            capture_output=True, text=True, env=clean_env())
        for stamp in (".torch-ok", ".gmsh-ok", ".selftest-ok"):
            p = os.path.join(clone, ".venv", stamp)
            if os.path.exists(p):
                os.remove(p)
        dead = "http://127.0.0.1:9"
        proxies = {k: dead for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
                                     "https_proxy", "all_proxy")}
        proxies.update(NO_PROXY="", no_proxy="")
        if os.name == "nt":
            cmd_ = ["cmd", "/c", os.path.join(clone, "run.cmd"), "--check"]
        else:
            cmd_ = ["bash", "-c", "cd '%s' && ./run.sh --check" % clone]
        env = clean_env(dict(proxies, **({"PYTHON": args.python} if args.python else {})))
        f = run_logged(cmd_, clone, env,
                       os.path.join(logs, "optional_fails_%s%s.log" % (plat, tag)))
        text = _read(f["log"])
        f["uninstall_exit_code"] = un.returncode
        f["types"] = _type_lines(text, keys)
        f["said_torch_failed"] = "torch could not be installed" in text or \
            "torch is not installed" in text or "torch is skipped" in text
        f["said_gmsh_failed"] = "Gmsh could not be installed" in text
        f["stamps_written"] = [s for s in (".torch-ok", ".gmsh-ok")
                               if os.path.exists(os.path.join(clone, ".venv", s))]
        out["optional_fails"] = f
        print("  optional installs failing: exit %s, torch failure said %s, Gmsh failure "
              "said %s, stamps %s, types ok %d of %d"
              % (f["exit_code"], f["said_torch_failed"], f["said_gmsh_failed"],
                 f["stamps_written"] or "none", sum(f["types"].values()), len(f["types"])))
        ok = (ok and f["exit_code"] == 3 and f["said_torch_failed"] and f["said_gmsh_failed"]
              and not f["stamps_written"] and len(f["types"]) == 8 and all(f["types"].values()))
        _write(rec_path, out)

    out["passes"] = ok
    _write(rec_path, out)
    print("  passes: %s  ->  %s" % (ok, rec_path))
    return 0 if ok else 1


def _read(path: str) -> str:
    return open(path, encoding="utf-8", errors="replace").read()


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
