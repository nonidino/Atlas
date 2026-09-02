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
    ap.add_argument("--domain", default="small",
                    help="small | medium | large (default small)")
    ap.add_argument("--turbines", type=int, default=6)
    ap.add_argument("--threads", type=int, default=_default_threads(),
                    help="torch CPU threads; the composed step saturates near 8")
    ap.add_argument("--horizon", type=int, default=7,
                    help="macro-steps differentiated per optimiser iteration")
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda. The coupled graph moves; the classical "
                         "monolith is numpy and stays on the CPU either way, "
                         "which the head-to-head panel says out loud.")
    ap.add_argument("--expert", default="reference_exposed",
                    help="reference_exposed | poseidon. Which fluid expert runs "
                         "in every window. `poseidon` is the frozen "
                         "20.8M-parameter checkpoint and needs the build repo "
                         "and the scOT loader; if it cannot be built here the "
                         "demo says so and runs the classical column rather "
                         "than claiming a frozen expert and showing a solver.")
    ap.add_argument("--steps", type=int, default=30,
                    help="macro-steps per classical-vs-coupled comparison")
    ap.add_argument("--open", action="store_true",
                    help="open a browser at the demo once the server is up")
    ap.add_argument("--log-level", default="warning")
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)

    from .engine import DemoConfig
    from .server import run
    cfg = DemoConfig(domain=args.domain, k=args.turbines, horizon=args.horizon,
                     verify_steps=args.steps, device=args.device,
                     expert=args.expert).clamped()
    if cfg.device != args.device:
        print(f"  [!] --device {args.device} is unavailable here; running on CPU")
    if cfg.expert != args.expert:
        print(f"  [!] --expert {args.expert} is unavailable here "
              "(the checkpoint or the build repo is missing); running the "
              "classical column")
    print(f"  fluid expert in every window: {cfg.expert}")
    run(host=args.host, port=args.port, cfg=cfg, log_level=args.log_level,
        open_browser=args.open)
    return 0
