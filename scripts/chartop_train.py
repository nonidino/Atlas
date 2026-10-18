"""Stage 0 (W355): train the registered arms of a run plan, in order, resuming.

    python scripts/chartop_train.py --plan plan.json --data DIR --runs DIR [--only B,Bp]

``plan.json`` (written after pricing, before training, and copied into the record as
``out/chartop/plan.json``) lists the arms of `atlas/chartop/gates.py` ``ARMS`` with
their budgets: ``{"arms": [{"name": ..., <train.Arm fields>}, ...], "minutes": {...}}``.
An arm whose ``summary.json`` exists is skipped, and one interrupted resumes from its
``last.pt``, so the same command continues a run after a crash.  Every arm writes
``<runs>/<name>/`` (``arm.json``, ``log.jsonl``, ``best.pt``, ``last.pt``,
``summary.json``) and ``<runs>/<name>.done``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atlas.chartop import gates  # noqa: E402
from atlas.chartop import train as tr  # noqa: E402


def arm_from(d: dict) -> tr.Arm:
    d = dict(d)
    if "batch" in d:
        d["batch"] = {int(k): int(v) for k, v in d["batch"].items()}
    return tr.Arm(**d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    plan = json.load(open(args.plan, encoding="utf-8"))
    only = set(filter(None, args.only.split(",")))
    os.makedirs(args.runs, exist_ok=True)
    status = 0
    for d in plan["arms"]:
        arm = arm_from(d)
        if arm.name not in gates.ARMS:
            raise SystemExit(f"arm {arm.name!r} is not registered in gates.ARMS")
        if only and arm.name not in only:
            continue
        out = os.path.join(args.runs, arm.name)
        if os.path.exists(os.path.join(out, "summary.json")):
            print(f"{arm.name}: done already", flush=True)
            continue
        minutes = plan.get("minutes", {}).get(arm.name)
        t0 = time.time()
        print(f"{arm.name}: training {arm.steps} steps (wall-clock guard {minutes} min)", flush=True)
        try:
            s = tr.train(arm, args.data, out, device=args.device, minutes=minutes)
            print(f"{arm.name}: {json.dumps(s)}", flush=True)
            code = 0
        except Exception:  # noqa: BLE001 -- persist the failure and go on to the next arm
            traceback.print_exc()
            with open(os.path.join(args.runs, arm.name + ".error.txt"), "w", encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
            code = 1
            status = 1
        with open(os.path.join(args.runs, arm.name + ".done"), "w", encoding="utf-8") as fh:
            fh.write(f"{code} {time.time() - t0:.1f}\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
