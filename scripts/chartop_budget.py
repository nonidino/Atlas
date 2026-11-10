"""The registration after Stage 0 on a rented box: project the cost from this box's own
price record, and refuse to train past the ceiling.

    python scripts/chartop_budget.py --plan plan.json --price-dir out --dph 0.27
        [--only S] [--ceiling 10] [--spent 0.02] [--out budget.json]

Reads the newest ``price-forms-*.json`` in ``--price-dir`` (`scripts/chartop_price.py
--forms`, run on this box first).  Each run of the plan is costed at its own measured
step: S's form with the strips skipped in training when ``strips_in_training`` is False,
otherwise with them, and W's form.  Its validations are costed from the forward pass
with the strips, over the plan's validation split; the data from the generator's rate
on this box, a labelled validation example counted as five plain ones; and setup,
pricing and downloads as a fixed allowance per box.

**The rule** (the owner's words: "stop if the projection exceeds $10").  The whole
registration, all five runs on two boxes like this one at this box's rate, plus what
is already spent, must not exceed ``--ceiling``.  Otherwise the script exits 3 and
nothing trains.  It writes ``budget.json`` whatever it decides.  This box's own share
is recorded beside the rule.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

FIXED_HOURS_PER_BOX = 0.4      # setup, pricing, downloads
LABELLED_COST = 5.0            # a labelled validation example, in plain examples
VALIDATION_MARGIN = 1.5        # the labelled objective and bookkeeping on top of the forward
BOXES = 2


def _label(run: dict) -> str:
    if not run.get("strips", False):
        return "W"
    return "S" if run.get("strips_in_training", True) else "S, strips only at validation"


def project_twolevel(plan: dict, price: dict, dph: float, only: set, spent: float) -> dict:
    """A ``gates_twolevel`` plan, from a ``price-twolevel`` record (`chartop_price.py
    --twolevel`): each run of TL at the measured step; each validation at twice TL's forward
    (TL and its twin, TL0) over the validation split; the data from the generator's rate, every
    training example labelled light and every validation example labelled, each costed as a
    labelled one; both runs on this one box."""
    part = price["twolevel"]
    mix = {int(k): v for k, v in part["projection"]["step_mix"].items()}
    step = {r["n"]: r["step_s"] for r in part["rows"]}
    fwd = {r["n"]: r["forward_s"] / r["batch"] for r in part["rows"]}
    val_n = {int(k): v for k, v in plan["val_examples"].items()}
    runs = []
    for d in plan["arms"]:
        if d.get("registered") != "TL":
            raise SystemExit(f"run {d.get('name')!r} is not gates_twolevel's TL")
        mean = sum(mix[n] * step[n] for n in mix)
        per_val = VALIDATION_MARGIN * 2.0 * sum(val_n[n] * fwd[n] for n in val_n)
        nval = math.ceil(d["steps"] / d["val_every"])
        hours = (mean * d["steps"] + per_val * nval) / 3600.0
        runs.append({"name": d["name"], "priced_as": "TL", "mean_step_s": mean, "validation_s": per_val,
                     "validations": nval, "hours": hours, "on_this_box": (not only) or d["name"] in only})
    rate = price["generator"]["examples_per_s"]
    data_h = LABELLED_COST * (plan["data"]["train"] + plan["data"]["val"]) / rate / 3600.0
    per_box = data_h + FIXED_HOURS_PER_BOX
    reg_hours = sum(r["hours"] for r in runs) + per_box
    box_hours = sum(r["hours"] for r in runs if r["on_this_box"]) + per_box
    return {"price_record": price.get("_path"), "gpu": price.get("gpu"), "dph": dph, "runs": runs,
            "data_hours_per_box": data_h, "fixed_hours_per_box": FIXED_HOURS_PER_BOX,
            "this_box": {"hours": box_hours, "usd": box_hours * dph},
            "registration": {"hours": reg_hours, "usd": reg_hours * dph + spent, "spent_usd": spent,
                             "rule": "both runs of TL on one box like this one, at this box's rate"}}


def project(plan: dict, price: dict, dph: float, only: set, spent: float) -> dict:
    """``plan["registration"] == "gates_prior"`` reads the ``"prior"`` part of a
    ``price-prior`` record, each run costed as its registered arm (P or PG), forward and
    step alike; any other plan is costed as before, from a ``price-forms`` record."""
    prior = plan.get("registration") == "gates_prior"
    part = price["prior"] if prior else price["forms"]
    rows = part["rows"]
    mix = {int(k): v for k, v in part["projection"]["step_mix"].items()}
    step = {}
    fwd = {}
    for r in rows:
        step.setdefault(r["arm"], {})[r["n"]] = r["step_s"]
        fwd.setdefault(r["arm"], {})[r["n"]] = r["forward_s"] / r["batch"]
    val_n = {int(k): v for k, v in plan["val_examples"].items()}
    runs = []
    for d in plan["arms"]:
        lab = d["registered"] if prior else _label(d)
        if lab not in step:
            raise SystemExit(f"the price record has no step for {lab!r}")
        mean = sum(mix[n] * step[lab][n] for n in mix)
        # validations solve the strips whatever the training does
        vfwd = fwd[lab] if prior else fwd["S" if d.get("strips") else "W"]
        per_val = VALIDATION_MARGIN * sum(val_n[n] * vfwd[n] for n in val_n)
        nval = math.ceil(d["steps"] / d["val_every"])
        hours = (mean * d["steps"] + per_val * nval) / 3600.0
        runs.append({"name": d["name"], "priced_as": lab, "mean_step_s": mean,
                     "validation_s": per_val, "validations": nval, "hours": hours,
                     "on_this_box": (not only) or d["name"] in only})
    rate = price["generator"]["examples_per_s"]
    data_h = (plan["data"]["train"] + LABELLED_COST * plan["data"]["val"]) / rate / 3600.0
    per_box = data_h + FIXED_HOURS_PER_BOX
    reg_hours = sum(r["hours"] for r in runs) + BOXES * per_box
    box_hours = sum(r["hours"] for r in runs if r["on_this_box"]) + per_box
    return {"price_record": price.get("_path"), "gpu": price.get("gpu"), "dph": dph,
            "runs": runs, "data_hours_per_box": data_h, "fixed_hours_per_box": FIXED_HOURS_PER_BOX,
            "this_box": {"hours": box_hours, "usd": box_hours * dph},
            "registration": {"hours": reg_hours, "usd": reg_hours * dph + spent, "spent_usd": spent,
                             "rule": f"all {'four' if prior else 'five'} runs on {BOXES} boxes like this one, "
                                     f"at this box's rate"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--price-dir", required=True)
    ap.add_argument("--dph", type=float, required=True, help="this box's dph_total, $ an hour")
    ap.add_argument("--only", default="")
    ap.add_argument("--ceiling", type=float, default=10.0)
    ap.add_argument("--spent", type=float, default=0.0)
    ap.add_argument("--out", default="budget.json")
    args = ap.parse_args()
    plan = json.load(open(args.plan, encoding="utf-8"))
    kind = {"gates_prior": "prior", "gates_twolevel": "twolevel"}.get(plan.get("registration"), "forms")
    files = sorted(glob.glob(os.path.join(args.price_dir, f"price-{kind}-*.json")), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no price-{kind} record: price this box first")
    price = json.load(open(files[-1], encoding="utf-8"))
    price["_path"] = os.path.basename(files[-1])
    proj = project_twolevel if kind == "twolevel" else project
    rec = proj(plan, price, args.dph, set(filter(None, args.only.split(","))), args.spent)
    rec["ceiling_usd"] = args.ceiling
    rec["within"] = bool(rec["registration"]["usd"] <= args.ceiling)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    for r in rec["runs"]:
        print(f"{r['name']:7s} as {r['priced_as']:30s} {r['mean_step_s']:.4f} s/step, "
              f"{r['validations']} validations of {r['validation_s']:.1f} s: {r['hours']:.2f} h"
              + ("  (this box)" if r["on_this_box"] else ""))
    print(f"data {rec['data_hours_per_box']:.2f} h and fixed {FIXED_HOURS_PER_BOX} h per box")
    print(f"this box: {rec['this_box']['hours']:.2f} h, ${rec['this_box']['usd']:.2f}")
    print(f"registration: {rec['registration']['hours']:.2f} h, ${rec['registration']['usd']:.2f} "
          f"against ${args.ceiling:.2f}: {'within' if rec['within'] else 'OVER: nothing trains'}")
    return 0 if rec["within"] else 3


if __name__ == "__main__":
    sys.exit(main())
