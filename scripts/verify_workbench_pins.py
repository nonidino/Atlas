"""Demo step 7 -- resolve the workbench install's pins for every platform it claims,
with pip's own resolver, installing nothing.

    <a venv's python> scripts/verify_workbench_pins.py

There is no Mac to install on (the owner's answer, 2026-09-30), so this is the
part of a Mac install that can be checked from here: for each of Windows x86-64,
macOS arm64, macOS x86-64 and Linux x86-64, and each of CPython 3.10, 3.11 and
3.12, ``pip install --dry-run --report`` of ``requirements.txt`` with
``constraints.txt``, **binary wheels only**; then torch from the index the
launchers use (the CPU index on Windows and Linux, PyPI on Apple silicon; no
Intel-macOS wheel exists, and `run.sh` skips it there with a message) and Gmsh,
each with the constraints.  A target passes when every package resolves to a
wheel at its pin and nothing the install pulls in is unpinned.

Run it with the pip the launchers pin (24.3.1), i.e. a `.venv` they built.  It
fetches package metadata (and, where PyPI has no separate metadata file, the
wheel into pip's cache); it installs nothing.  Writes
``out/workbench/records/installer/resolve_matrix.json``.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "atlas", "workbench", "bundle")
OUT = os.path.join(ROOT, "out", "workbench", "records", "installer", "resolve_matrix.json")

TARGETS = {
    "win_amd64": ["win_amd64"],
    # pip expands a macOS tag to every older one ...
    "macos-arm64": ["macosx_14_0_arm64"],
    "macos-x86_64": ["macosx_14_0_x86_64"],
    # ... and the legacy manylinux2014 alias to 2010 and 1, but not a PEP 600
    # tag, so each glibc level is named
    "linux-x86_64": ["manylinux_2_28_x86_64", "manylinux_2_27_x86_64",
                     "manylinux_2_26_x86_64", "manylinux_2_24_x86_64",
                     "manylinux_2_17_x86_64", "manylinux2014_x86_64"],
}
PYS = ["3.10", "3.11", "3.12"]
CPU = ["--index-url", "https://download.pytorch.org/whl/cpu",
       "--extra-index-url", "https://pypi.org/simple"]


def norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def pins(path: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.split("#", 1)[0].strip()
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==(\S+)", ln)
            if m:
                out[norm(m.group(1))] = m.group(2)
    return out


def launcher_pins() -> dict[str, str]:
    text = open(os.path.join(TPL, "run.sh"), encoding="utf-8").read()
    return {norm(n): v for n, v in re.findall(r'^(?:TORCH|GMSH)="([a-z]+)==([^"]+)"', text,
                                               re.M)}


def resolve(plats, pyv, args):
    with tempfile.TemporaryDirectory() as tmp:
        rep = os.path.join(tmp, "report.json")
        cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
               "--quiet", "--report", rep, "--target", os.path.join(tmp, "t"),
               "--python-version", pyv, "--implementation", "cp", "--only-binary=:all:",
               "--disable-pip-version-check"]
        for p in plats:
            cmd += ["--platform", p]
        r = subprocess.run(cmd + args, capture_output=True, text=True)
        if r.returncode:
            return None, (r.stderr or r.stdout)[-800:]
        with open(rep, encoding="utf-8") as fh:
            d = json.load(fh)
    return {norm(i["metadata"]["name"]): i["metadata"]["version"] for i in d["install"]}, None


def judge(got: dict, pinned: dict) -> dict:
    return {"n": len(got), "installs": dict(sorted(got.items())),
            "unpinned": sorted("%s==%s" % (n, v) for n, v in got.items() if n not in pinned),
            "off_pin": sorted("%s==%s (pin %s)" % (n, v, pinned[n]) for n, v in got.items()
                              if n in pinned and v.split("+")[0] != pinned[n])}


def main() -> int:
    req = pins(os.path.join(TPL, "requirements.txt"))
    cons = pins(os.path.join(TPL, "constraints.txt"))
    lp = launcher_pins()
    pinned = {**cons, **req, **lp}
    torch_pin = "torch==%s" % lp["torch"]
    gmsh_pin = "gmsh==%s" % lp["gmsh"]
    pip_v = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True,
                           text=True).stdout.split()[1]
    rec = {"at": dt.datetime.now().isoformat(timespec="seconds"), "pip": pip_v,
           "targets": {}, "passes": True}
    for tname, plats in TARGETS.items():
        for pyv in PYS:
            key = "%s py%s" % (tname, pyv)
            row: dict = {"platform_tags": plats}
            got, err = resolve(plats, pyv, ["-r", os.path.join(TPL, "requirements.txt"),
                                             "-c", os.path.join(TPL, "constraints.txt")])
            row["requirements"] = {"error": err} if err else judge(got, pinned)
            if tname == "macos-x86_64":
                row["torch"] = {"skipped": "no Intel-macOS wheel; run.sh skips it, saying so"}
            else:
                extra = [] if tname.startswith("macos") else CPU
                got, err = resolve(plats, pyv, [torch_pin, "-c",
                                                 os.path.join(TPL, "constraints.txt")] + extra)
                row["torch"] = {"error": err} if err else judge(got, pinned)
            got, err = resolve(plats, pyv, [gmsh_pin, "-c",
                                             os.path.join(TPL, "constraints.txt")])
            row["gmsh"] = {"error": err} if err else judge(got, pinned)
            ok = all("error" not in row[k] and not row[k].get("unpinned")
                     and not row[k].get("off_pin")
                     for k in ("requirements", "torch", "gmsh") if "skipped" not in row[k])
            row["passes"] = ok
            rec["passes"] = rec["passes"] and ok
            rec["targets"][key] = row

            def short(k):
                v = row[k]
                if "skipped" in v:
                    return "skipped"
                if "error" in v:
                    return "ERROR " + " ".join(v["error"].split())[-140:]
                return "%d pkgs%s%s" % (v["n"], ", unpinned %s" % v["unpinned"]
                                        if v["unpinned"] else "",
                                        ", off-pin %s" % v["off_pin"] if v["off_pin"] else "")
            print("  %-20s %s   requirements %s; torch %s; gmsh %s"
                  % (key, "ok  " if ok else "FAIL", short("requirements"), short("torch"),
                     short("gmsh")), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("  passes: %s  ->  %s" % (rec["passes"], OUT))
    return 0 if rec["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
