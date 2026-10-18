"""Stage 0 (W355): write the Expert Card of arm B from the one evaluation's record.

    python scripts/chartop_card.py --evaluation out/chartop/evaluation-<time>.json

Writes ``out/chartop/card-B.json`` (`atlas.chartop.card`): the identity, the chart
methods, the envelope and Fourier range the host enforces, the output format, and the
certificates measured by the evaluation (SE1's identity residual, E1's cells, the dual
gap); a certificate the evaluation did not measure is written as missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atlas.chartop import card  # noqa: E402
from atlas.chartop.train import load_expert  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--weights", default=os.path.join(ROOT, "out", "chartop", "runs", "B", "best.pt"))
    ap.add_argument("--plan", default=os.path.join(ROOT, "out", "chartop", "plan.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "chartop", "card-B.json"))
    args = ap.parse_args()
    ev = json.load(open(args.evaluation, encoding="utf-8"))
    plan = json.load(open(args.plan, encoding="utf-8"))
    cfg = load_expert(args.weights, "cpu").config()
    c = card.card(ev, args.weights, plan, cfg)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(card.to_json(c) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
