"""Argument parsing for the demo server, shared by `-m atlas.demo` and scripts/."""
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
        description="Live wind-farm design demo over the Atlas composed graph")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--domain", default="medium",
                    help="small | medium | large (default medium, which is the "
                         "K12 rung the published 3.15x speed ratio was measured "
                         "at; at small the composed column has fewer windows to "
                         "amortise over and the ratio is smaller or inverted)")
    ap.add_argument("--turbines", type=int, default=12)
    ap.add_argument("--threads", type=int, default=_default_threads(),
                    help="torch CPU threads; the composed step saturates near 8")
    ap.add_argument("--horizon", type=int, default=5,
                    help="macro-steps differentiated per optimiser iteration; a "
                         "gradient macro-step costs three to four and a half "
                         "forward ones, so this sets how long a viewer waits "
                         "between one layout and the next")
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda. The composed columns move; the classical "
                         "monolith is numpy and stays on the CPU either way, "
                         "which the speed panel says out loud.")
    ap.add_argument("--expert", default="poseidon",
                    help="poseidon | reference_exposed. Which fluid expert runs "
                         "in every window. The default is the frozen "
                         "20.8M-parameter Poseidon-T checkpoint (CC-BY-NC-4.0, "
                         "research use only); if it cannot be built here the "
                         "demo says so and runs the classical column rather "
                         "than claiming a frozen expert and showing a solver.")
    ap.add_argument("--steps", type=int, default=40,
                    help="macro-steps per column per measurement region; below "
                         "about 30 the unoptimised layout's wakes have not "
                         "developed and the verified gain is invisible")
    ap.add_argument("--no-third-column", dest="three_way", action="store_false",
                    help="do not march the classical composed column. It is the "
                         "row that separates 'the cut pays' from 'the expert "
                         "pays', and it is a third of the cost of a measurement.")
    ap.add_argument("--no-measure-on-start", dest="measure_on_start",
                    action="store_false",
                    help="do not time the columns on the starting layout at boot")
    ap.add_argument("--open", action="store_true",
                    help="open a browser at the demo once the server is up")
    ap.add_argument("--log-level", default="warning")
    ap.set_defaults(three_way=True, measure_on_start=True)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)

    from .engine import EXPERT_NOTE, DemoConfig
    from .server import run
    cfg = DemoConfig(domain=args.domain, k=args.turbines, horizon=args.horizon,
                     verify_steps=args.steps, device=args.device,
                     expert=args.expert, three_way=args.three_way,
                     measure_on_start=args.measure_on_start).clamped()
    if cfg.device != args.device:
        print(f"  [!] --device {args.device} is unavailable here; running on CPU")
    if cfg.expert != args.expert:
        print(f"  [!] --expert {args.expert} is unavailable here "
              "(the checkpoint or the build repo is missing); running the "
              "classical column instead. The speed claim on the screen is "
              "about the frozen expert and will not reproduce without it.")
    print(f"  fluid expert in every window: {cfg.expert}")
    print(f"  {EXPERT_NOTE.get(cfg.expert, '')}")
    run(host=args.host, port=args.port, cfg=cfg, log_level=args.log_level,
        open_browser=args.open)
    return 0
