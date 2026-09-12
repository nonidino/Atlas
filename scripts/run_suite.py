"""Run the whole test suite: four parallel groups, then the serial files.

    python scripts/run_suite.py            # everything
    python scripts/run_suite.py --groups 4

Five files run SERIALLY and ALONE at the end -- tier21, tier22, tier31, tier32 and
tier33 -- because each drives a server, an optimiser or a long march with its own
wall-clock deadline, and they read as failures when they share the machine.  The
offline variables are exported here rather than being assumed: without them a
Poseidon load resolves the hub id online, and no test sets them.

Every group writes its own log under ``out/suite/``.  The summary at the end
parses those logs; a group whose log is missing is reported as such rather than
being silently dropped from the count.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "suite")
SERIAL = ("test_tier21_", "test_tier22_", "test_tier31_", "test_tier32_", "test_tier33_")

ENV = dict(os.environ)
ENV.update({"KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1"})


def _pytest(files, log):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files]
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)


def _tally(log):
    if not os.path.exists(log):
        return None
    text = open(log, encoding="utf-8", errors="replace").read()
    m = re.findall(r"(\d+) (passed|failed|error|errors|skipped|xfailed|deselected)", text)
    out = {}
    for n, what in m:
        out[what] = out.get(what, 0) + int(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tests = sorted(f for f in os.listdir(os.path.join(ROOT, "tests"))
                   if f.startswith("test_") and f.endswith(".py"))
    serial = [f for f in tests if f.startswith(SERIAL)]
    parallel = [f for f in tests if f not in serial]
    groups = [parallel[i::args.groups] for i in range(args.groups)]
    print(f"{len(tests)} files: {len(parallel)} in {args.groups} parallel groups, "
          f"{len(serial)} serial", flush=True)

    t0 = time.time()
    procs = []
    for i, g in enumerate(groups):
        log = os.path.join(OUT, f"group{i}.log")
        procs.append((f"group{i}", log, _pytest([os.path.join("tests", f) for f in g], log)))
    time.sleep(5)
    for name, log, _p in procs:
        print(f"  {name}: log {'exists' if os.path.exists(log) else 'MISSING'}", flush=True)
    for name, log, p in procs:
        p.wait()
        print(f"  {name} done in {time.time()-t0:.0f}s: {_tally(log)}", flush=True)

    for f in serial:
        log = os.path.join(OUT, f"{f[:-3]}.log")
        print(f"  {f} (serial, alone) ...", flush=True)
        p = _pytest([os.path.join("tests", f)], log)
        p.wait()
        print(f"    {f}: {_tally(log)}  [{time.time()-t0:.0f}s]", flush=True)

    total = {}
    missing = []
    for name in [f"group{i}" for i in range(args.groups)] + [f[:-3] for f in serial]:
        log = os.path.join(OUT, f"{name}.log")
        t = _tally(log)
        if t is None:
            missing.append(name)
            continue
        for k, v in t.items():
            total[k] = total.get(k, 0) + v
    print(f"TOTAL {total}  missing logs: {missing or 'none'}  "
          f"wall {time.time()-t0:.0f}s", flush=True)
    return 1 if total.get("failed") or total.get("error") or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
