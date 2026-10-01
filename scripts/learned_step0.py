"""Demo step 8, step 0 (demo-learned-case-plan section 4): price the learned window
stepper before any data exists.

    python scripts/learned_step0.py                  # farm-12, widths 8..32
    python scripts/learned_step0.py --case fast-farm

**Speed does not depend on the weights.**  So a randomly initialised network of
each candidate width is timed in the workbench's own process, beside the classical
arms it must beat, from a developed state of the case the page will show:

  t_F     the full domain's macro-step (the workbench's ``full`` arm)
  t_Ep    the classical decomposition's macro-step on threads (``parallel``)
  t_win   the part of t_Ep spent stepping the windows (the classical exposed
          WindowNS, every window, on the pool), timed inside the same steps
  t_rest  t_Ep - t_win: the forcing, the cut, the blend, the global projection
          and the band.  The learned arm keeps all of it, unchanged
  t_net   the network's forward over every window, batched, torch at the same
          thread count as Ep, including the copies numpy -> torch -> numpy
  t_L     t_rest + t_net

**The registered budget (plan section 4):** t_L <= t_Ep / 1.5; the gate's G1 also
asks t_F / t_L >= 3.  If no width meets it, the case moves to design L-B, or to the
5-rotor rung, or stops, before anything is generated.

Medians over the timed macro-steps, each arm stepped from the same developed state
(the classical decomposition marched ``--warm`` macro-steps from the freestream),
on the machine as it stands, whose state is recorded beside the numbers.  Writes
``out/learned-case/step0-<stamp>.json``.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

import numpy as np                                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "learned-case")

from atlas.workbench import machine                                     # noqa: E402
from atlas.workbench.families import windfarm as wf                     # noqa: E402
from atlas.workbench.spec import example_case                           # noqa: E402

BUDGET_EP = 1.5          # t_L <= t_Ep / 1.5  (plan section 4, registered)
BUDGET_F = 3.0           # and t_F / t_L >= 3 (G1's second half)


class TimedPool:
    """The run's thread pool, with the time spent in its ``map`` (the windows'
    classical step) recorded per call."""

    def __init__(self, inner):
        self.inner = inner
        self.last = None

    def map(self, fn, it):
        t0 = time.perf_counter()
        out = list(self.inner.map(fn, it))
        self.last = time.perf_counter() - t0
        return out

    def shutdown(self, *a, **k):
        return self.inner.shutdown(*a, **k)


def window_inputs(run, s):
    """What the learned arm hands the network: every window's (u - U, v, f)."""
    t = run.tiling
    fx, _rec = run.forcing(s.u)
    xs = []
    for shape, idx in run.groups.items():
        xs.append(np.stack([t.cut(s.u, idx) - run.u_inf, t.cut(s.v, idx), t.cut(fx, idx)],
                           axis=1))
    return np.concatenate(xs)


def time_net(torch, net, x, threads, repeats):
    torch.set_num_threads(threads)
    out = []
    with torch.inference_mode():
        for k in range(repeats + 2):
            t0 = time.perf_counter()
            xt = torch.from_numpy(x.astype(np.float32))
            y = net(xt)
            back = y.numpy().astype(np.float64)
            dt_ = time.perf_counter() - t0
            if k >= 2:                                   # two warm-up calls
                out.append(dt_)
    assert back.shape[0] == x.shape[0]
    return float(np.median(out)), out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="farm-12")
    ap.add_argument("--warm", type=int, default=10, help="macro-steps before timing")
    ap.add_argument("--rounds", type=int, default=5, help="timed macro-steps per arm")
    ap.add_argument("--widths", default="8,12,16,24,32")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    import torch
    from atlas.workbench.learned_net import WindowUNet, n_parameters

    spec = example_case(args.case)
    threads = int(spec.run.threads)
    before = machine.machine_record()
    run = wf.build(spec, arms=("parallel", "full"), threads=threads)
    run.pool = TimedPool(run.pool)
    print("%s: %d x %d cells, %d windows, %d rotors, macro-step %.3g, %d threads"
          % (args.case, run.nx, run.ny, run.tiling.n_windows, len(run.rotors), run.dt,
             threads), flush=True)

    s = run.initial("parallel")
    t0 = time.perf_counter()
    for _ in range(args.warm):
        s = run.step("parallel", s)
    print("  warmed %d macro-steps of the decomposition in %.1f s (sub-steps now %d)"
          % (args.warm, time.perf_counter() - t0, s.substeps), flush=True)

    t_ep, t_win, t_f, subs = [], [], [], []
    for r in range(args.rounds):
        for arm in (("parallel", "full") if r % 2 == 0 else ("full", "parallel")):
            t1 = time.perf_counter()
            nxt = run.step(arm, s)
            d = time.perf_counter() - t1
            if arm == "parallel":
                t_ep.append(d)
                t_win.append(run.pool.last)
                subs.append(nxt.substeps)
                s_next = nxt
            else:
                t_f.append(d)
        s = s_next
    med = {k: float(np.median(v)) for k, v in
           (("t_Ep", t_ep), ("t_win", t_win), ("t_F", t_f))}
    med["t_rest"] = med["t_Ep"] - med["t_win"]
    print("  t_F %.3f s   t_Ep %.3f s (windows %.3f, rest %.3f)   F/Ep %.2fx   sub-steps %s"
          % (med["t_F"], med["t_Ep"], med["t_win"], med["t_rest"], med["t_F"] / med["t_Ep"],
             subs), flush=True)

    x = window_inputs(run, s)
    nets = []
    for w in [int(v) for v in args.widths.split(",")]:
        torch.manual_seed(0)
        net = WindowUNet(width=w, depth=args.depth).eval()
        t_net, all_ = time_net(torch, net, x, threads, args.rounds)
        t_l = med["t_rest"] + t_net
        row = {"width": w, "depth": args.depth, "parameters": n_parameters(net),
               "t_net": t_net, "t_net_all": all_, "t_L": t_l,
               "Ep_over_L": med["t_Ep"] / t_l, "F_over_L": med["t_F"] / t_l}
        row["meets_budget"] = bool(row["Ep_over_L"] >= BUDGET_EP and row["F_over_L"] >= BUDGET_F)
        nets.append(row)
        print("  width %3d: %8d parameters, forward %.3f s -> t_L %.3f s, Ep/L %.2fx, F/L "
              "%.2fx  %s" % (w, row["parameters"], t_net, t_l, row["Ep_over_L"],
                             row["F_over_L"], "meets" if row["meets_budget"] else "MISSES"),
              flush=True)

    # the thread count, for the widest network that meets the budget (information only)
    ok = [r for r in nets if r["meets_budget"]]
    threads_scan = {}
    if ok:
        w = ok[-1]["width"]
        torch.manual_seed(0)
        net = WindowUNet(width=w, depth=args.depth).eval()
        for th in sorted({1, 2, threads, 8}):
            threads_scan[th] = time_net(torch, net, x, th, args.rounds)[0]
        torch.set_num_threads(threads)
        print("  width %d forward by torch threads: %s" % (w, {k: round(v, 3) for k, v in
                                                            threads_scan.items()}), flush=True)

    rec = {
        "what": "demo-learned-case-plan section 4, step 0: a random network priced before "
                "any data exists",
        "case": args.case, "cells": run.nx * run.ny, "windows": run.tiling.n_windows,
        "rotors": len(run.rotors), "macro_dt": run.dt, "threads": threads,
        "warm_macro_steps": args.warm, "rounds": args.rounds, "substeps": subs,
        "medians_s": med, "t_Ep_all": t_ep, "t_win_all": t_win, "t_F_all": t_f,
        "budget": {"Ep_over_L_at_least": BUDGET_EP, "F_over_L_at_least": BUDGET_F,
                   "source": "demo-learned-case-plan section 4 (registered); G1 of section 5"},
        "networks": nets, "torch_threads_scan": threads_scan,
        "torch": torch.__version__, "numpy": np.__version__,
        "machine_before": before,
        "machine_after": {"power": machine.power_status()},
        "note": args.note,
        "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "step0-%s.json" % dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    run.close()
    print("  ->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
