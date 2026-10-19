"""Stage 0 (W355): train the registered arms of a run plan, in order, resuming.

    python scripts/chartop_train.py --plan plan.json --data DIR --runs DIR [--only B,Bp]

``plan.json`` (written after pricing, before training, and copied into the record as
``out/chartop/plan.json``) lists the arms of `atlas/chartop/gates.py` ``ARMS`` with
their budgets: ``{"arms": [{"name": ..., <train.Arm fields>}, ...], "minutes": {...}}``.
An arm whose ``summary.json`` exists is skipped, and one interrupted resumes from its
``last.pt``, so the same command continues a run after a crash.  Every arm writes
``<runs>/<name>/`` (``arm.json``, ``log.jsonl``, ``best.pt``, ``last.pt``,
``summary.json``) and ``<runs>/<name>.done``.

A plan with ``"registration": "gates_forms"`` trains the registration after Stage 0
(`atlas/chartop/gates_forms.py`, pinned).  Each arm then names its registered arm
(``"registered": "S"``, ``"Sp"`` or ``"W"``), and the whole plan is checked against
`gates_forms.ARMS` before anything trains: the form, networks, ports, objective, data,
bubble, strips, steps and seed of every run, and every registered seed run exactly
once.  ``strips_in_training`` is a training option, not a registered field: it does
not change the weights (`train.Arm`).
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
    d.pop("registered", None)
    if "batch" in d:
        d["batch"] = {int(k): int(v) for k, v in d["batch"].items()}
    return tr.Arm(**d)


def check_forms_plan(plan: dict) -> None:
    """Every run of a ``gates_forms`` plan is one registered arm and seed, and every
    registered seed is run once; a mismatch refuses the whole plan."""
    from atlas.chartop import gates_forms as GF
    want_common = {k: v for k, v in GF.COMMON.items() if k != "m"}   # `train.Arm` keeps m = 16
    seen = []
    for d in plan["arms"]:
        reg = d.get("registered")
        if reg not in GF.ARMS:
            raise SystemExit(f"run {d.get('name')!r} names no registered arm of gates_forms: {reg!r}")
        arm, r = arm_from(d), GF.ARMS[reg]
        got = {"form": arm.form, "backbone": arm.backbone, "size": arm.size, "p_size": arm.p_size,
               "kind": arm.kind, "objective": arm.objective, "data": arm.data,
               "bubble": arm.bubble, "strips": arm.strips, "steps": arm.steps}
        want = dict(want_common, bubble=r["bubble"], strips=r["strips"], steps=r["steps"])
        if got != want or arm.seed not in r["seeds"] or arm.limit is not None:
            raise SystemExit(f"run {arm.name!r} is not gates_forms arm {reg!r}: {got} against {want}, "
                             f"seed {arm.seed} of {r['seeds']}, limit {arm.limit}")
        seen.append((reg, arm.seed))
    need = sorted((k, s) for k, r in GF.ARMS.items() for s in r["seeds"])
    if sorted(seen) != need:
        raise SystemExit(f"the plan runs {sorted(seen)}, the registration {need}")


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
    forms_plan = plan.get("registration") == "gates_forms"
    if forms_plan:
        check_forms_plan(plan)
    os.makedirs(args.runs, exist_ok=True)
    status = 0
    for d in plan["arms"]:
        arm = arm_from(d)
        if not forms_plan and arm.name not in gates.ARMS:
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
