"""Demo item 1.4: sweep a family's Fast example candidates, then confirm the choice.

    python scripts/fast_examples.py sweep <example> [k=v ...] [k=v ...] ...
    python scripts/fast_examples.py confirm <example>

``sweep`` runs the example once per configuration, each a set of overrides
(``threads=8``, ``steps=12``, or an example parameter such as ``nx=640``), with
every arm the page's *Run and compare* runs, timed by the workbench's own runner
(`runner.CaseRun`: the arms take turns, only the family's ``step`` is timed).  It
judges each run against the family's registered bars (``FAST``, `fast.py`) and the
two-minute rule, and writes ``out/workbench/records/fast/sweep-<example>-<time>.json``.
**Selection is not measurement** (`demo-fast-examples-plan` section 1): the number
a card shows comes from ``confirm``, a fresh process running the example exactly as
it is defined, after the choice was made.

Every record carries the machine's state before and after (power, other Python
processes), and the note the owner gave on 2026-09-30 about the other chats.  A run
on battery is refused here: not a result.
"""

from __future__ import annotations

import datetime
import importlib
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "out", "workbench", "records", "fast")
TWO_MINUTES = 120.0
NOTE = ("the owner, 2026-09-30: the other two chats are not idle but are writing "
        "documentation and theory, nothing on the demo")


def _spec(key: str, overrides: dict):
    from atlas.workbench import spec as S
    ex = S.EXAMPLES[key]
    params = dict(ex.params)
    run_over = {k: v for k, v in overrides.items() if k in ("threads", "steps", "macro_dt")}
    params.update({k: v for k, v in overrides.items() if k not in run_over})
    S.EXAMPLES[key] = S.Example(ex.key, ex.label, ex.description, ex.family, ex.style,
                                tuple(params.items()))
    try:
        spec = S.example_case(key)
    finally:
        S.EXAMPLES[key] = ex
    for k, v in run_over.items():
        setattr(spec.run, k, v)
    return spec


def run_once(key: str, overrides: dict) -> dict:
    from atlas.workbench import fast, machine
    from atlas.workbench.runner import CaseRun, adapter_for, arms_for
    from atlas.workbench.spec import EXAMPLES
    spec = _spec(key, overrides)
    module = adapter_for(spec.physics.family)
    arms, _why = arms_for(module, spec)
    declared = dict(EXAMPLES[key].params).get("arms")
    if declared:                       # the arms the page runs for this example
        arms = tuple(a for a in arms if a in str(declared).split())
    t0 = time.perf_counter()
    run = CaseRun(spec, arms=arms, steps=spec.run.steps, threads=spec.run.threads,
                  results_dir=None, case_label=key).run_blocking()
    wall = time.perf_counter() - t0
    rec = run.results or {}
    timing = rec.get("timing", {})
    dec = {a: t.get("speedup_vs_full") for a, t in timing.items() if a != "full"}
    best_arm = max((a for a in dec if dec[a]), key=lambda a: dec[a], default=None)
    checks = rec.get("checks", [])
    agreement = fast.agreement(checks)
    judged = []
    for bars in getattr(module, "FAST", ()):
        judged.append(fast.judge(bars, dec.get(best_arm) if best_arm else None, agreement))
    return {"example": key, "overrides": overrides, "family": spec.physics.family,
            "style": spec.coupling.style, "cells": int(spec.domain.nx * spec.domain.ny),
            "windows": len(spec.windows), "threads": spec.run.threads,
            "steps": spec.run.steps, "macro_dt": spec.run.macro_dt,
            "wall_seconds": wall, "two_minutes": wall <= TWO_MINUTES,
            "speedup": dec, "fastest_arm": best_arm,
            "mean_step_s": {a: t.get("mean") for a, t in timing.items()},
            "checks": [{k: c.get(k) for k in ("key", "kind", "value", "tolerance", "passed",
                                               "detail")} for c in checks],
            "all_checks_pass": all(c.get("passed") is not False for c in checks),
            "bars": judged, "status": run.status, "error": rec.get("error"),
            "notes": rec.get("notes"), "describe": rec.get("problem"),
            "machine_before": rec.get("machine"),
            "machine_after": rec.get("machine_after")}


def _parse(args) -> list[dict]:
    configs = []
    for a in args:
        cfg = {}
        for kv in a.split(","):
            if not kv:
                continue
            k, v = kv.split("=", 1)
            cfg[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        configs.append(cfg)
    return configs or [{}]


def main(argv) -> int:
    from atlas.workbench import machine
    if len(argv) < 2 or argv[0] not in ("sweep", "confirm"):
        print(__doc__)
        return 2
    cmd, key = argv[0], argv[1]
    if not machine.power_status().get("ac"):
        print("on battery: no timing taken (not a result)")
        return 3
    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(OUT, f"{cmd}-{key}-{stamp}.json")
    rec = {"command": cmd, "example": key, "note": NOTE,
           "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "machine": machine.machine_record(), "runs": []}
    configs = _parse(argv[2:]) if cmd == "sweep" else [{}]
    for cfg in configs:
        r = run_once(key, cfg)
        rec["runs"].append(r)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, default=str)
        sp = r["speedup"].get(r["fastest_arm"]) if r["fastest_arm"] else None
        print(f"{key} {cfg or '(as defined)'}: {r['cells']} cells, {r['windows']} pieces, "
              f"T={r['threads']}, {r['steps']} steps; fastest {r['fastest_arm']} "
              f"{sp if sp is None else round(sp, 3)}x; wall {r['wall_seconds']:.1f} s; "
              f"checks pass {r['all_checks_pass']}; bars "
              + "; ".join(f"{b['mechanism']}: met {b['met']} (fast {b['fast']}, close "
                          f"{b['close']}, agreement {b['agreement']})" for b in r["bars"]),
              flush=True)
    print(f"written {os.path.relpath(path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
