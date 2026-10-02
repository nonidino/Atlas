"""Build the Lean project outside OneDrive, time it, and record the timing.

The Lean sources live in ``lean/`` and are tracked by git.  The BUILD does not
run there, for three reasons that are all about this machine:

* the repository sits inside OneDrive, and Lake's ``.lake`` folder holds the
  unpacked Mathlib cache: several gigabytes in tens of thousands of files.
  Built in place, OneDrive would upload all of it;
* OneDrive holds freshly written files for a moment, and a build writes
  thousands of them (a PermissionError here has cost this project before);
* Mathlib's file paths are long, and the build folder's prefix is shorter.

So this script MIRRORS the sources into a build folder (default
``~/.cache/atlas-lean``, override with ATLAS_LEAN_BUILD_DIR), runs ``lake``
there, and appends one line per run to ``out/lean/builds.jsonl``: the command,
the wall time measured here, the exit code, how many jobs Lake ran, how many
``sorry`` warnings and errors the log holds, and whether the laptop was on
battery (it runs several times slower on battery, so a time without that fact
is not comparable with another).  The full log goes to ``out/lean/logs/``.

Elsewhere (CI, a clone outside OneDrive) plain ``lake build`` in ``lean/``
works: nothing in the project depends on this script.

    python scripts/lean_build.py                 # mirror, then `lake build`
    python scripts/lean_build.py --clean         # re-check every project file
    python scripts/lean_build.py --label t1 build AtlasProofs.PerturbedContraction
    python scripts/lean_build.py update          # resolve dependencies (first run)
    python scripts/lean_build.py exe cache get   # fetch prebuilt Mathlib
    python scripts/lean_build.py --profile       # per-file times, slow steps

Mathlib is never compiled from source: if a run is about to build Mathlib
modules (the log shows them being BUILT rather than replayed), stop it and run
``exe cache get``.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "lean")
OUT = os.path.join(REPO, "out", "lean")
LOGS = os.path.join(OUT, "logs")
#: Two chats may prove different batches at the same time, on different clones.
#: Each sets ATLAS_LEAN_TAG (say "batches-1-3"), and then writes its own record
#: files (builds-<tag>.jsonl, profile-<tag>.json, axioms-<tag>.json), so their
#: branches never both append to one file and merge without a conflict.
TAG = os.environ.get("ATLAS_LEAN_TAG", "").strip()
_SUFFIX = ("-" + TAG) if TAG else ""
RECORD = os.path.join(OUT, "builds%s.jsonl" % _SUFFIX)
PROFILE = os.path.join(OUT, "profile%s.json" % _SUFFIX)
AXIOMS = os.path.join(OUT, "axioms%s.json" % _SUFFIX)
HOME = os.path.expanduser("~")
BUILD = os.environ.get("ATLAS_LEAN_BUILD_DIR", os.path.join(HOME, ".cache", "atlas-lean"))
ELAN_BIN = os.path.join(HOME, ".elan", "bin")

#: Never mirrored: Lake's own folder, and generated blueprint pages.
SKIP_DIRS = {".lake", "__pycache__", "web", "print"}
#: Marks a build folder as this script's, so the mirror may delete inside it.
MARKER = ".atlas-lean-build"
#: A slow elaboration step, in milliseconds (formal-proofs-plan section 4.1:
#: "a tactic call over ten seconds is replaced by an explicit proof step").
SLOW_MS = 10_000


def _retry(fn, *args, tries: int = 8, wait: float = 0.5):
    """OneDrive holds a fresh file for a moment; a read or a replace then raises."""
    for k in range(tries):
        try:
            return fn(*args)
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _source_files(root: str) -> list[str]:
    """Every file under ``root`` that belongs to the project, as relative paths."""
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), root)
            found.append(rel.replace("\\", "/"))
    return sorted(found)


def mirror(build: str) -> dict:
    """Copy changed sources from ``lean/`` into the build folder.

    Only a file whose bytes differ is rewritten, so Lake re-checks only what
    changed.  A source file deleted from ``lean/`` is deleted from the build
    folder too.  ``lake-manifest.json`` is the one file that travels the other
    way (see `manifest_back`): Lake writes it during ``lake update``.
    """
    # The mirror DELETES files the sources no longer have, so it only ever runs
    # in a folder this script made: a new or empty one gets a marker, and a
    # non-empty folder without the marker is refused.
    marker = os.path.join(build, MARKER)
    if os.path.isdir(build) and os.listdir(build) and not os.path.exists(marker):
        raise SystemExit("refusing to mirror into %s: it is not empty and was not made "
                         "by this script (no %s)" % (build, MARKER))
    os.makedirs(build, exist_ok=True)
    if not os.path.exists(marker):
        _write(marker, b"made by scripts/lean_build.py; safe to delete this whole folder\n")
    src = _source_files(SRC)
    copied, removed = [], []
    for rel in src:
        data = _retry(_read, os.path.join(SRC, rel))
        # git stores LF and this checkout is CRLF; Lean hashes file contents,
        # so mirror LF to make the build independent of the checkout's setting.
        if rel.endswith((".lean", ".toml", ".json", ".tex", ".md")) or rel == "lean-toolchain":
            data = data.replace(b"\r\n", b"\n")
        dst = os.path.join(build, rel)
        if not os.path.exists(dst) or _read(dst) != data:
            _write(dst, data)
            copied.append(rel)
    keep = set(src)
    for rel in _source_files(build):
        if rel in keep:
            continue
        if rel in ("lake-manifest.json", MARKER):
            continue  # the manifest is Lake's (copied back by manifest_back())
        os.remove(os.path.join(build, rel))
        removed.append(rel)
    return {"copied": copied, "removed": removed, "files": len(src)}


def manifest_back(build: str) -> bool:
    """Copy Lake's manifest into ``lean/`` when ``lake update`` changed it."""
    src = os.path.join(build, "lake-manifest.json")
    dst = os.path.join(SRC, "lake-manifest.json")
    if not os.path.exists(src):
        return False
    data = _read(src)
    if os.path.exists(dst) and _retry(_read, dst).replace(b"\r\n", b"\n") == data:
        return False
    _retry(_write, dst, data)
    return True


def power() -> dict:
    """AC or battery, from Windows itself.  Empty elsewhere."""
    if os.name != "nt":
        return {}

    class _Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    st = _Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        return {}
    line = {0: "battery", 1: "ac"}.get(st.ACLineStatus, "unknown")
    return {"power": line, "charge_percent": int(st.BatteryLifePercent) & 0xFF}


def _pins() -> dict:
    toolchain = _retry(_read, os.path.join(SRC, "lean-toolchain")).decode("utf-8").strip()
    lakefile = _retry(_read, os.path.join(SRC, "lakefile.toml")).decode("utf-8")
    m = re.search(r'name\s*=\s*"mathlib".*?rev\s*=\s*"([^"]+)"', lakefile, re.S)
    return {"lean": toolchain, "mathlib": m.group(1) if m else None}


def _env() -> dict:
    env = dict(os.environ)
    env["PATH"] = ELAN_BIN + os.pathsep + env.get("PATH", "")
    return env


def _lake() -> str:
    exe = os.path.join(ELAN_BIN, "lake.exe" if os.name == "nt" else "lake")
    return exe if os.path.exists(exe) else "lake"


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _say(text: str) -> None:
    """Print through a console that may be cp1252 (Lean's output is not ASCII)."""
    enc = sys.stdout.encoding or "ascii"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_log(text: str) -> dict:
    """Count what a Lake log reports.

    Lake prints one line per message, `warning: File.lean:L:C: text` or
    `error: File.lean:L:C: text`, and Lean words a hole as "declaration uses
    `sorry`" with backticks.  **The first version of these patterns matched
    neither** (it looked for `: warning:` and for straight quotes) and reported
    0 sorries on a log that held 34.  So the patterns are anchored on the line's
    first word and on the bare word `sorry`, and `--reparse` re-reads a saved
    log: a log known to hold sorries is the positive control.
    """
    jobs = re.search(r"\((\d+) jobs?\)", text)
    return {
        "jobs": int(jobs.group(1)) if jobs else None,
        "errors": len(re.findall(r"(?m)^error: ", text)),
        "warnings": len(re.findall(r"(?m)^warning: ", text)),
        "sorry_warnings": len(re.findall(r"(?m)^warning: .*\bsorry\b", text)),
        "mathlib_modules_compiled": len(re.findall(r"(?m)^\S+ \[\d+/\d+\] Built Mathlib\.", text)),
    }


def run_lake(args: list[str], build: str, label: str, note: str = "") -> dict:
    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, "%s-%s.log" % (_stamp(), label))
    cmd = [_lake()] + args
    before = power()
    t0 = time.perf_counter()
    with open(log_path, "wb") as log:
        proc = subprocess.run(cmd, cwd=build, env=_env(), stdout=log, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - t0
    text = _read(log_path).decode("utf-8", errors="replace")
    counts = parse_log(text)
    built_mathlib = counts["mathlib_modules_compiled"]
    rec = {
        "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": label,
        "cmd": "lake " + " ".join(args),
        "seconds": round(seconds, 2),
        "exit": proc.returncode,
        "log": os.path.relpath(log_path, REPO).replace("\\", "/"),
        "build_dir": build,
    }
    rec.update(counts)
    rec.update(before)
    rec.update(_pins())
    if note:
        rec["note"] = note
    with open(RECORD, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    tail = text.strip().splitlines()[-25:]
    _say("\n".join(tail))
    _say("")
    _say("[lean_build] %s: %.1f s, exit %d, jobs %s, errors %d, sorry %d, power %s"
         % (rec["cmd"], seconds, proc.returncode, rec["jobs"], rec["errors"],
            rec["sorry_warnings"], rec.get("power", "?")))
    if built_mathlib:
        _say("[lean_build] WARNING: %d Mathlib modules were COMPILED here. "
             "Mathlib must come from the cache: run `exe cache get`." % built_mathlib)
    _say("[lean_build] log: %s" % rec["log"])
    return rec


def project_modules() -> list[str]:
    mods = []
    for rel in _source_files(os.path.join(SRC, "AtlasProofs")):
        if rel.endswith(".lean"):
            mods.append("AtlasProofs/" + rel)
    return mods


def split_steps(lines: list[str]) -> tuple[list[str], list[str]]:
    """Separate the profiler's slow lines into elaboration steps and imports.

    The ten-second rule is about a tactic call in THIS project's proofs.  The
    profiler also prints "import took 14.9s" when loading Mathlib's compiled
    files crosses the threshold, which on this laptop on battery it does for
    most files (2026-10-01: 18 of 26, 10.2 to 15 s each, with no proof step
    anywhere near).  Counted as a slow step, that read as 18 violations of a
    rule no proof had broken.  So an import is recorded beside the steps, never
    among them.
    """
    slow = [ln.strip() for ln in lines if re.search(r"\btook \d", ln)]
    imports = [ln for ln in slow if ln.startswith("import took")]
    steps = [ln for ln in slow if not ln.startswith("import took")]
    return steps, imports


def profile(build: str) -> int:
    """Check each project file alone, with Lean's profiler on.

    Reports each file's wall time and every elaboration step the profiler
    prints at or above `SLOW_MS`.  A file is checked against the ALREADY BUILT
    project, so run a normal build first.
    """
    os.makedirs(OUT, exist_ok=True)
    rows, slow_total, import_total = [], 0, 0
    for rel in project_modules():
        cmd = [_lake(), "env", "lean", "-Dprofiler=true",
               "-Dprofiler.threshold=%d" % SLOW_MS, rel]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=build, env=_env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
        seconds = time.perf_counter() - t0
        text = proc.stdout.decode("utf-8", errors="replace")
        # The profiler prints "<what> took <n>s" (or ms) for each step over the threshold.
        slow, imports = split_steps(text.splitlines())
        slow_total += len(slow)
        import_total += len(imports)
        rows.append({"file": rel, "seconds": round(seconds, 2), "exit": proc.returncode,
                     "slow_steps": slow, "slow_imports": imports})
        _say("%-52s %7.2f s  exit %d  slow steps %d%s"
             % (rel, seconds, proc.returncode, len(slow),
                "  (%s)" % imports[0] if imports else ""))
        for ln in slow:
            _say("    " + ln)
    out = {
        "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "slow_ms": SLOW_MS, "files": rows, "slow_steps_total": slow_total,
        "slow_imports_total": import_total,
        "seconds_total": round(sum(r["seconds"] for r in rows), 2),
    }
    out.update(power())
    out.update(_pins())
    with open(PROFILE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    _say("")
    _say("[lean_build] %d files, %.1f s in all, %d step(s) at or over %d ms "
         "(and %d Mathlib import(s) over it, not counted), power %s"
         % (len(rows), out["seconds_total"], slow_total, SLOW_MS, import_total,
            out.get("power", "?")))
    bad = slow_total + sum(1 for r in rows if r["exit"] != 0)
    return 1 if bad else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", default=None, help="a short name for the record line")
    ap.add_argument("--note", default="", help="free text stored with the record line")
    ap.add_argument("--clean", action="store_true",
                    help="delete the project's own build output first (Mathlib stays)")
    ap.add_argument("--profile", action="store_true",
                    help="check each file alone with the profiler; no record line")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--reparse", metavar="LOG", default=None,
                    help="count errors, warnings and sorries in a saved log, and stop")
    ap.add_argument("lake_args", nargs=argparse.REMAINDER,
                    help="arguments for lake (default: build)")
    ns = ap.parse_args(argv[1:])

    if ns.reparse:
        counts = parse_log(_read(ns.reparse).decode("utf-8", errors="replace"))
        _say(json.dumps(counts, sort_keys=True))
        return 0

    if not ns.no_mirror:
        m = mirror(BUILD)
        _say("[lean_build] mirrored %d files to %s (%d changed, %d removed)"
             % (m["files"], BUILD, len(m["copied"]), len(m["removed"])))
        for rel in m["copied"]:
            _say("    + " + rel)
        for rel in m["removed"]:
            _say("    - " + rel)

    if ns.profile:
        return profile(BUILD)

    args = [a for a in ns.lake_args if a != "--"] or ["build"]
    if ns.clean:
        own = os.path.join(BUILD, ".lake", "build")
        if os.path.isdir(own):
            shutil.rmtree(own)
            _say("[lean_build] removed the project's own build output: " + own)
    label = ns.label or ("-".join(args)[:40].replace("/", "_").replace(".", "_"))
    if ns.clean:
        label += "-clean"
    rec = run_lake(args, BUILD, label, ns.note)
    if manifest_back(BUILD):
        _say("[lean_build] lake-manifest.json changed: copied back to lean/")
    return rec["exit"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
