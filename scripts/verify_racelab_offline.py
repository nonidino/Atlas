"""PoC 3 phase 5 -- with no network, the classical column must still run.

    python scripts/verify_racelab_offline.py --clone C:/Users/Nauni/poc3x

§2's third constraint, checked on a clone the launcher has ALREADY installed
(this script installs nothing and downloads nothing):

  (a) **the launcher, offline.**  `scOT` is uninstalled from the clone's venv
      and its stamp removed, so the launcher tries to fetch it again -- with
      every proxy variable pointed at a closed local port, which is what no
      network looks like to pip.  The fetch must fail, the launcher must say so
      and carry on, and ``--check`` must exit **3**: the classical column
      marched, the learned expert is not here.
  (b) **no connection off this machine at all.**  The self-test is run again
      in-process with `socket.socket.connect` (and `connect_ex`, and
      `create_connection`) replaced by a function that records the address
      and refuses anything that is not loopback.  A library that ignored the
      proxy variables would still be caught here.  Loopback is allowed and
      recorded, because the self-test asks the demo's own server on 127.0.0.1
      for a frame (W251).  It must exit 3 with ZERO off-machine attempts.

Writes ``out/racelab_bundle/verify_<platform>_offline.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: run in the CLONE's interpreter, with the clone as the working directory
_PROBE = r'''
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
    raise OSError("network disabled by verify_racelab_offline")
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
               "loopback_connections": loopback}, fh)
sys.exit(code)
'''


def _clean_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    for k in ("ATLAS_BUILD_REPO", "HF_HOME", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE", "PYTHONPATH", "VIRTUAL_ENV", "PYTHON",
              "KMP_DUPLICATE_LIB_OK", "PYTHONIOENCODING", "PYTHONUTF8"):
        env.pop(k, None)
    env.update(extra or {})
    return env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clone", required=True)
    args = ap.parse_args(argv)
    clone = os.path.abspath(args.clone)
    win = os.name == "nt"
    vpy = os.path.join(clone, ".venv", "Scripts" if win else "bin",
                       "python.exe" if win else "python")
    if not os.path.isfile(vpy):
        raise SystemExit("%s has no installed venv; run the launcher first" % clone)
    plat = "%s-%s" % (platform.system().lower(), platform.machine().lower())
    outdir = os.path.join(ROOT, "out", "racelab_bundle")
    rec = {"at": dt.datetime.now().isoformat(timespec="seconds"),
           "platform": plat, "clone": clone}

    # -- (a) the launcher with scOT gone and nowhere to fetch it from --------
    r = subprocess.run([vpy, "-m", "pip", "uninstall", "-y", "scOT"],
                       capture_output=True, text=True)
    rec["uninstall_scot"] = {"exit_code": r.returncode,
                             "tail": (r.stdout + r.stderr)[-600:]}
    stamp = os.path.join(clone, ".venv", ".scot-ok")
    if os.path.exists(stamp):
        os.remove(stamp)
    dead = "http://127.0.0.1:9"
    env = _clean_env({"HTTP_PROXY": dead, "HTTPS_PROXY": dead,
                      "ALL_PROXY": dead, "http_proxy": dead,
                      "https_proxy": dead, "all_proxy": dead, "NO_PROXY": "",
                      "no_proxy": ""})
    cmd = (["cmd", "/c", os.path.join(clone, "run.cmd"), "--check"] if win
           else ["bash", "-c", "cd %s && ./run.sh --check" % clone])
    log = os.path.join(outdir, "offline_launcher_%s.log" % plat)
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(cmd, cwd=clone, env=env, stdout=fh,
                           stderr=subprocess.STDOUT, shell=False)
    text = open(log, encoding="utf-8", errors="replace").read()
    rec["launcher_offline"] = {
        "exit_code": p.returncode, "wall_s": time.time() - t0, "log": log,
        "said_the_fetch_failed": "Could not install scOT" in text,
        "said_partial": "PARTIAL" in text,
        "classical_marched": "the classical column" in text
                             and "rms against the referent 0.0e+00" in text,
        "scot_stamp_written": os.path.exists(stamp),
        "tail": text[-2500:]}

    # -- (b) the self-test with every connection refused and counted ---------
    attempts = os.path.join(outdir, "offline_socket_attempts_%s.json" % plat)
    log2 = os.path.join(outdir, "offline_socket_%s.log" % plat)
    t0 = time.time()
    with open(log2, "w", encoding="utf-8", errors="replace") as fh:
        q = subprocess.run([vpy, "-c", _PROBE], cwd=clone,
                           env=_clean_env({"ATTEMPTS_OUT": attempts}),
                           stdout=fh, stderr=subprocess.STDOUT)
    got = json.load(open(attempts)) if os.path.isfile(attempts) else {}
    rec["socket_refused"] = {
        "exit_code": q.returncode, "wall_s": time.time() - t0, "log": log2,
        "connection_attempts": got.get("connection_attempts"),
        "loopback_connections_allowed": got.get("loopback_connections"),
        "why_loopback_is_allowed": "the self-test asks the demo's own server, "
                                   "on 127.0.0.1, for a frame over the page's "
                                   "WebSocket (W251); only connections off "
                                   "this machine are refused and counted",
        "tail": open(log2, encoding="utf-8", errors="replace").read()[-1500:]}

    rec["passes"] = (rec["launcher_offline"]["exit_code"] == 3
                     and rec["launcher_offline"]["said_the_fetch_failed"]
                     and rec["launcher_offline"]["classical_marched"]
                     and not rec["launcher_offline"]["scot_stamp_written"]
                     and rec["socket_refused"]["exit_code"] == 3
                     and rec["socket_refused"]["connection_attempts"] == [])
    path = os.path.join(outdir, "verify_%s_offline.json" % plat)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("  (a) launcher offline: exit %s, fetch failure reported %s, classical "
          "marched %s" % (rec["launcher_offline"]["exit_code"],
                          rec["launcher_offline"]["said_the_fetch_failed"],
                          rec["launcher_offline"]["classical_marched"]))
    print("  (b) sockets refused: exit %s, connection attempts %s"
          % (rec["socket_refused"]["exit_code"],
             rec["socket_refused"]["connection_attempts"]))
    print("  passes: %s -> %s" % (rec["passes"], path))
    return 0 if rec["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
