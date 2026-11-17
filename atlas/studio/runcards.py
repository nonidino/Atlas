"""A run's record as the page shows it: ratios and verdicts, never raw seconds.

The plan's rule (section 6): **price is a ratio only**, measured on the machine in
front of the viewer, beside the spread an arm has against itself.  The record on
disk keeps every per-step time (`runner.CaseRun._record`); the page is sent only
what this module derives from it:

* ``speed``: the whole domain's mean step time over each decomposed arm's (above 1
  is faster than the whole domain), the retired workbench's speed card;
* ``spread``: each arm's own A/A spread, the 90th over the 10th percentile of its
  per-step times, so a ratio can be read against the noise it sits in;
* ``agreement`` and ``balance``: the family's registered checks of those kinds;
* ``checks``: every registered check, each with its tolerance and verdict;
* the Fast example's card and the learned case's card, when the case is one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from atlas.workbench.runner import ARM_LABELS


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def spread(seconds: list[float]) -> float | None:
    """An arm against itself: its 90th percentile step time over its 10th."""
    s = [v for v in seconds if v and v > 0]
    if len(s) < 3:
        return None
    lo, hi = np.percentile(s, [10, 90])
    return float(hi / lo) if lo > 0 else None


def cards(res: dict[str, Any]) -> dict[str, Any]:
    checks = res.get("checks") or []
    timing = res.get("timing") or {}
    labels = {**ARM_LABELS, **(res.get("arm_labels") or {})}
    ratios = {a: _num(t.get("speedup_vs_full")) for a, t in timing.items()
              if a != "full" and _num(t.get("speedup_vs_full"))}
    speed = None
    if ratios:
        best = "learned" if "learned" in ratios else max(ratios, key=lambda a: ratios[a])
        speed = {"arm": best, "label": labels.get(best, best), "ratio": ratios[best],
                 "ratios": {a: {"label": labels.get(a, a), "ratio": r}
                            for a, r in ratios.items()}}
    spreads = {a: spread(t.get("seconds_per_step") or []) for a, t in timing.items()}

    def pick(kind):
        cs = [c for c in checks if c.get("kind") == kind]
        cs = [c for c in cs if c.get("passed") is not None] or cs
        if not cs:
            return None
        c = cs[0]
        return {"title": c.get("title"), "value": _num(c.get("value")),
                "tolerance": _num(c.get("tolerance")), "passed": c.get("passed"),
                "verdict": c.get("verdict")}
    rows = [{"key": c.get("key"), "title": c.get("title"), "kind": c.get("kind"),
             "value": _num(c.get("value")), "tolerance": _num(c.get("tolerance")),
             "passed": c.get("passed"), "verdict": c.get("verdict"),
             "detail": c.get("detail"), "registered": c.get("registered")}
            for c in checks]
    return {
        "family": res.get("family"), "case_name": res.get("case_name"),
        "arms": list(res.get("arms") or []),
        "labels": {a: labels.get(a, a) for a in res.get("arms") or []},
        "dropped": res.get("dropped_arms") or {},
        "steps": res.get("steps_done"), "steps_requested": res.get("steps_requested"),
        "stopped": bool(res.get("stopped")), "step_label": res.get("step_label"),
        "threads": res.get("threads"),
        "speed": speed, "spread": spreads,
        "agreement": pick("reference"), "balance": pick("balance"),
        "checks": rows,
        "passed": sum(1 for c in checks if c.get("passed") is True),
        "failed": sum(1 for c in checks if c.get("passed") is False),
        "unmeasured": sum(1 for c in checks if c.get("passed") is None),
        "bitwise": res.get("bitwise"),
        "field_difference": {a: {k: _num(v) for k, v in (d or {}).items()}
                             for a, d in (res.get("field_difference") or {}).items()},
        "field_label": res.get("field_label"),
        "notes": [str(n) for n in res.get("notes") or []],
        "record": res.get("record_path"),
        "machine": {"power": (res.get("machine") or {}).get("power"),
                    "cpus": (res.get("machine") or {}).get("cpu_count"),
                    "others": len((res.get("machine") or {}).get("other_python_processes")
                                  or [])},
        "fast": fast_card(res), "learned": learned_card(res),
    }


def fast_card(res: dict[str, Any]) -> dict | None:
    """The Fast example's card (demo item 1.4): the mechanism line and the verdict
    against the bars the kind registered before its first timed run."""
    from atlas.workbench import fast
    from atlas.workbench.runner import adapter_for
    from atlas.workbench.spec import FAST_EXAMPLES
    fid = res.get("family", "")
    if not fid or FAST_EXAMPLES.get(fid) != res.get("case_name"):
        return None
    module = adapter_for(fid)
    bars_all = getattr(module, "FAST", ())
    if not bars_all:
        return None
    style = (res.get("case") or {}).get("coupling", {}).get("style")
    mech = "M" if style == "M" else ("X" if fid == "thermoelastic-2d" else "P")
    bars = next((b for b in bars_all if b.mechanism == mech), bars_all[0])
    timing = res.get("timing") or {}
    ratios = {a: t.get("speedup_vs_full") for a, t in timing.items()
              if a != "full" and t.get("speedup_vs_full")}
    best = max(ratios, key=lambda a: ratios[a]) if ratios else None
    speed = ratios.get(best) if best else None
    j = fast.judge(bars, speed, fast.agreement(res.get("checks", [])))
    prob = res.get("problem") or {}
    limit = getattr(module, "FAST_LIMIT", None)
    try:
        if limit:
            line = limit
        elif mech == "M":
            line = fast.mechanism_line("M", slow=", ".join(prob.get("slow_pieces", [])),
                                       fast=", ".join(prob.get("fast_pieces", [])),
                                       ratio=prob.get("rate", 1))
        elif mech == "X":
            line = fast.mechanism_line("X", fraction=1.0 / speed if speed else float("nan"))
        else:
            n = prob.get("windows") or len((res.get("case") or {}).get("windows", []))
            line = fast.mechanism_line("P", pieces=n, threads=res.get("threads", 1))
    except (KeyError, TypeError, ZeroDivisionError):
        line = fast.MECHANISMS.get(mech, "")
    if j["met"]:
        verdict = (f"It meets both bars this kind registered before its first timed run: at "
                   f"least {bars.speed:g}× faster, and within {bars.agreement:.0e} of the whole "
                   f"domain ({bars.agreement_of}).")
    else:
        verdict = (f"It is below the {bars.speed:g}× bar this kind registered before its "
                   f"first timed run" + (f": {bars.ceiling}." if bars.ceiling else "."))
    return {"mechanism": fast.MECHANISMS[mech], "ratio": _num(speed), "line": line,
            "verdict": verdict, "met": bool(j["met"]), "fixed": fast.FIXED_LINE}


def learned_card(res: dict[str, Any]) -> dict | None:
    """The learned case's card (demo item 1.5): its arms against the truth, the
    registered evaluation's verdict, and the caveat.  Ratios only."""
    from atlas.workbench import learned_case as LC
    from atlas.workbench import learned_gate as G
    params = ((res.get("case") or {}).get("physics") or {}).get("params") or {}
    if not params.get(LC.MARKER):
        return None
    labels = {**ARM_LABELS, **(res.get("arm_labels") or {})}
    timing = res.get("timing") or {}
    mean = {a: (t or {}).get("mean") for a, t in timing.items()}
    speed = {}
    if mean.get("learned"):
        for a in ("parallel", "full"):
            if mean.get(a):
                speed[a] = {"label": labels.get(a, a), "ratio": mean[a] / mean["learned"]}
    metrics = res.get("metrics") or {}
    errs = metrics.get("errors_vs_truth") or {}
    gate = LC.gate_record()
    verdicts = None
    if gate is not None:
        v = gate.get("verdict") or {}
        verdicts = [{"key": g.key, "title": g.title,
                     "passed": (v.get(g.key) or {}).get("passed")} for g in G.GATE]
    return {"speed_vs_learned": speed,
            "errors": {a: {"label": labels.get(a, a), **{k: _num(x) for k, x in e.items()}}
                       for a, e in errs.items()},
            "not_measured": metrics.get("not_measured"),
            "gate": verdicts, "gate_all": bool(((gate or {}).get("verdict") or {}).get("all")),
            "gate_at": str((gate or {}).get("at", ""))[:10] or None,
            "caveat": LC.CAVEAT}


__all__ = ["cards", "fast_card", "learned_card", "spread"]
