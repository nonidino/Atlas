"""Demo step 8: the learned case's truth arm T, run once and stored.

    python scripts/learned_truth.py              # farm-12 at D/64, 40 macro-steps

`learned_gate`'s ``T``: the full domain at twice the resolution (cells of D/64,
1376 x 928), the classical solver as every farm runs it, from the uniform
freestream.  It is a reference, not an arm the page times, so it may run on
battery and beside other work.

Stores what the gate's measures read, on the grid they are read on:
``out/learned-case/truth-farm-12.npz`` with the farm power and the fluctuation
energy at every macro-step, and the velocity at the registered horizons
block-averaged to cells of D/16; the record beside it says how it was run.
It saves its full state every five macro-steps (``out/learned-case/data/``, not
committed), so a run that fails resumes from the last save instead of starting
over (the long-run rule).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

import numpy as np                                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "learned-case")
SAVE = os.path.join(OUT, "data", "truth-state.npz")

from atlas.workbench import learned_arms as LA                          # noqa: E402
from atlas.workbench import learned_gate as G                           # noqa: E402
from atlas.workbench import machine                                     # noqa: E402
from atlas.workbench.spec import example_case                           # noqa: E402


def main() -> int:
    spec = LA.scaled_spec(example_case(G.CASE), 2.0)
    run = LA.FarmRun(spec, arms=("full",), threads=G.THREADS)
    k = int(round(1.0 / run.dx)) // G.COMPARE_CELLS_PER_D           # 64 / 16 = 4
    before = machine.machine_record()
    s = run.initial("full")
    power, energy, subs, secs, fields = [], [], [], [], {}
    start = 0
    if os.path.isfile(SAVE):
        z = np.load(SAVE)
        start = int(z["step"])
        s.u, s.v = z["u"].copy(), z["v"].copy()
        power, energy = list(z["power"]), list(z["energy"])
        subs, secs = list(z["subs"]), list(z["secs"])
        for h in G.HORIZONS:
            if "u%d" % h in z.files:
                fields[h] = (z["u%d" % h], z["v%d" % h])
        print("resuming after macro-step %d" % start, flush=True)
    for n in range(start + 1, G.STEPS + 1):
        t0 = time.perf_counter()
        s = run.step("full", s)
        secs.append(time.perf_counter() - t0)
        power.append(run.power(s.rec))
        energy.append(G.fluctuation_energy(s.u, s.v, run.dx, run.u_inf))
        subs.append(s.substeps)
        if n in G.HORIZONS:
            fields[n] = (G.block_mean(s.u, k), G.block_mean(s.v, k))
        print("  macro-step %2d  %5.1f s  sub-steps %d  power %.5f" % (n, secs[-1], subs[-1],
                                                                       power[-1]), flush=True)
        if n % 5 == 0 or n == G.STEPS:
            os.makedirs(os.path.dirname(SAVE), exist_ok=True)
            extra = {}
            for h, (fu, fv) in fields.items():
                extra["u%d" % h], extra["v%d" % h] = fu, fv
            tmp = SAVE[:-4] + ".tmp.npz"
            np.savez(tmp, step=n, u=s.u, v=s.v, power=power, energy=energy, subs=subs,
                     secs=secs, **extra)
            os.replace(tmp, SAVE)
    arrays = {"power": np.array(power), "energy": np.array(energy)}
    for h, (fu, fv) in fields.items():
        arrays["u%d" % h], arrays["v%d" % h] = fu, fv
    path = os.path.join(OUT, "truth-%s.npz" % G.CASE)
    np.savez(path, **arrays)
    rec = {"what": "learned_gate's T: farm-12 at cells of D/64, the classical full domain, "
                   "from the uniform freestream, %d macro-steps" % G.STEPS,
           "cells": [run.nx, run.ny], "dx": run.dx, "band_cells": run.band_cells,
           "stored": "power and energy per macro-step; u, v at %s block-averaged %dx%d to "
                     "cells of D/16" % (list(G.HORIZONS), k, k),
           "substeps": subs, "seconds_per_step": secs, "machine_before": before,
           "machine_after": {"power": machine.power_status()},
           "file": os.path.relpath(path, ROOT).replace(os.sep, "/"),
           "sha256": __import__("hashlib").sha256(open(path, "rb").read()).hexdigest(),
           "at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    with open(os.path.join(OUT, "truth-%s.json" % G.CASE), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("  ->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
