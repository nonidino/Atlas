"""Argument parsing for the PoC 2 demo server."""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _default_threads() -> int:
    """The composed step saturates near 8 cores; more than that is contention."""
    try:
        return max(1, min(8, os.cpu_count() or 4))
    except Exception:                                          # pragma: no cover
        return 4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Live front-wing design demo over the PoC 2 assembled graph")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--threads", type=int, default=_default_threads())
    ap.add_argument("--tiling", default="six", choices=("six", "single"),
                    help="six = the composed column; single = the REFERENT, "
                         "one window over the whole domain, which differs from "
                         "the composed one by the cut and by nothing else")
    ap.add_argument("--coupling", default="tight",
                    choices=("tight", "lagged", "split"),
                    help="tight = the 33-equation interface system solved every "
                         "exchange; lagged = once per macro-step; split = the "
                         "two seams solved in TURN, which is the ablation")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=12,
                    help="macro-steps differentiated per optimiser iteration. "
                         "This is a WARM-STARTED SHORT-HORIZON estimator and "
                         "not the one the results page measures with -- the "
                         "screen says so")
    ap.add_argument("--penalty", type=float, default=40.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ablation-steps", type=int, default=60,
                    help="macro-steps per column in beat 3. The recorded "
                         "ablation is at 120 and is shown beside the live one")
    ap.add_argument("--race-steps", type=int, default=16,
                    help="macro-steps per objective evaluation in beat 4")
    ap.add_argument("--race-grad", type=int, default=14,
                    help="Adam steps in beat 4's gradient column. The two "
                         "budgets are SCALED from the recorded 30:200 rather "
                         "than chosen -- see DemoConfig")
    ap.add_argument("--race-pop", type=int, default=93,
                    help="objective evaluations in beat 4's CMA-ES column")
    ap.add_argument("--no-enforce", action="store_true",
                    help="start with beat 2's envelope check OFF. Everything "
                         "the demo then produces is stamped OUTSIDE THE MODEL; "
                         "this exists so the pre-W145 behaviour can be shown, "
                         "and it is not a mode anything should be measured in")
    ap.add_argument("--open", action="store_true", dest="open_browser")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(max(1, int(a.threads)))

    from .engine import DemoConfig
    from .server import run
    cfg = DemoConfig(tiling=a.tiling, coupling=a.coupling, lag=a.lag,
                     horizon=a.horizon, penalty=a.penalty,
                     stride=a.stride, enforce=not a.no_enforce,
                     ablation_steps=a.ablation_steps,
                     race_steps=a.race_steps, race_grad=a.race_grad,
                     race_pop=a.race_pop).clamped()
    run(host=a.host, port=a.port, cfg=cfg, open_browser=a.open_browser)
    return 0
