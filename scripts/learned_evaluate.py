"""Demo step 8: the learned case's gate, evaluated ONCE on the held-out case.

    python scripts/learned_evaluate.py --weights out/learned-case/window-net.pt

The registered evaluation (`atlas/workbench/learned_gate.py`, registered before any
data existed): farm-12, held out of training, from the uniform freestream for 40
macro-steps, the arms ``F`` (full domain), ``Ep`` (the classical decomposition on 4
threads), ``L`` (the same decomposition with the trained network in every window,
torch on 4 threads) and ``Cc`` (the classical decomposition on a grid twice as
coarse) taking turns every macro-step in a rotating order, as the workbench's
runner does; only each arm's step is timed, and every measure is read after the
timer.  ``T`` is read from its stored run (`scripts/learned_truth.py`).

**Evaluated once.**  The script refuses a second evaluation of the same weights
(by their sha256) unless ``--again`` gives a reason, which the record keeps.  G1 is
judged only if every timed macro-step ran on AC power; the record says either way.
G6 is verified from the training data's manifest: every layout in it is a
registered seed and passes `held_out_ok`.

Writes ``out/learned-case/gate-<stamp>.json``: the verdict of each of G1-G6, every
arm's per-step times, farm power, fluctuation energy and divergence, the errors
against T, the machine's state, and the hashes of the weights, the truth and the
registration.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import glob                                                             # noqa: E402
import hashlib                                                          # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

import numpy as np                                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "learned-case")


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def held_out_verified(manifest_path: str) -> tuple[bool, str]:
    from atlas.workbench import learned_gate as G
    if not os.path.isfile(manifest_path):
        return False, "no data manifest at %s" % manifest_path
    man = json.load(open(manifest_path))
    seeds = sorted(int(s) for s in man)
    registered = set(G.TRAIN_SEEDS) | set(G.VALIDATION_SEEDS)
    stray = [s for s in seeds if s not in registered]
    close = [s for s in seeds if not G.held_out_ok(G.sample_layout(s).rotors)]
    if stray or close:
        return False, "seeds outside the registration %s, too close %s" % (stray, close)
    return True, "%d layouts, all registered seeds, none within the held-out rule" % len(seeds)


def main(argv=None) -> int:
    import torch
    from atlas.workbench import learned_arms as LA
    from atlas.workbench import learned_gate as G
    from atlas.workbench import machine
    from atlas.workbench.learned_net import load_window_net
    from atlas.workbench.spec import example_case

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=os.path.join(OUT, "window-net.pt"))
    ap.add_argument("--truth", default=os.path.join(OUT, "truth-%s.npz" % "farm-12"))
    ap.add_argument("--manifest", default=os.path.join(OUT, "data-manifest.json"))
    ap.add_argument("--steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--again", default=None, help="the reason for a second evaluation")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    steps = args.steps or G.STEPS
    smoke = args.steps is not None and args.steps < G.STEPS

    w_sha = sha(args.weights)
    for old in glob.glob(os.path.join(args.out, "gate-*.json")):
        prev = json.load(open(old))
        if prev.get("weights_sha256") == w_sha and not prev.get("smoke") and not smoke \
                and not args.again:
            raise SystemExit("these weights were evaluated already (%s): the gate is "
                             "evaluated once; pass --again with a reason to override"
                             % os.path.basename(old))

    spec = example_case(G.CASE)
    net = load_window_net(args.weights)
    T = np.load(args.truth)
    before = machine.machine_record()
    runs = {
        "F": LA.FarmRun(spec, arms=("full",), threads=G.THREADS),
        "Ep": LA.FarmRun(spec, arms=("parallel",), threads=G.THREADS),
        "L": LA.LearnedFarmRun(spec, net, arms=("learned",), threads=G.THREADS),
        "Cc": LA.FarmRun(LA.scaled_spec(spec, 0.5), arms=("parallel",), threads=G.THREADS),
    }
    arm_of = {"F": "full", "Ep": "parallel", "L": "learned", "Cc": "parallel"}
    cells_per_d = {a: int(round(1.0 / r.dx)) for a, r in runs.items()}
    states = {a: r.initial(arm_of[a]) for a, r in runs.items()}
    rec = {a: {"step_seconds": [], "power": [], "energy": [], "div": [], "substeps": []}
           for a in runs}
    fields: dict = {a: {} for a in runs}
    order = list(runs)
    power_samples = [before["power"].get("ac")]
    t_all = time.perf_counter()
    for n in range(1, steps + 1):
        rot = order[(n - 1) % len(order):] + order[:(n - 1) % len(order)]
        for a in rot:
            r = runs[a]
            t0 = time.perf_counter()
            states[a] = r.step(arm_of[a], states[a])
            rec[a]["step_seconds"].append(time.perf_counter() - t0)
        for a, r in runs.items():                      # the instruments, after the timer
            s = states[a]
            rec[a]["power"].append(r.power(s.rec))
            rec[a]["energy"].append(G.fluctuation_energy(s.u, s.v, r.dx, r.u_inf))
            rec[a]["div"].append(r.mass_measure(arm_of[a], s))
            rec[a]["substeps"].append(int(s.substeps))
            if n in G.HORIZONS:
                fields[a][n] = (s.u.copy(), s.v.copy())
        power_samples.append(machine.power_status().get("ac"))
        print("  macro-step %2d: %s  (%.0f s)" % (n, "  ".join(
            "%s %.2fs" % (a, rec[a]["step_seconds"][-1]) for a in order),
            time.perf_counter() - t_all), flush=True)
    after = machine.machine_record()
    power_samples.append(after["power"].get("ac"))

    horizons = [h for h in G.HORIZONS if h <= steps]
    for a in runs:
        rec[a]["errors"] = {}
        for h in horizons:
            eP = G.error_power(rec[a]["power"], T["power"], h)
            eV = G.error_velocity(fields[a][h], cells_per_d[a],
                                  (T["u%d" % h], T["v%d" % h]), G.COMPARE_CELLS_PER_D)
            rec[a]["errors"][str(h)] = {"P": eP, "V": eV}
    ok6, why6 = held_out_verified(args.manifest)
    res = {"arms": rec, "on_ac": all(p is True for p in power_samples),
           "held_out_verified": ok6}
    verdict = G.judge(res) if not smoke else {"smoke": "a smoke test is not judged"}
    out = {
        "what": "the learned case's gate (atlas/workbench/learned_gate.py), evaluated once "
                "on the held-out case",
        "smoke": smoke, "steps": steps, "case": G.CASE,
        "verdict": verdict, "held_out": why6, "power_samples_ac": power_samples,
        "means_s": {a: float(np.mean(rec[a]["step_seconds"])) for a in runs},
        "arms": rec, "weights": os.path.relpath(args.weights, ROOT).replace(os.sep, "/"),
        "weights_sha256": w_sha, "truth_sha256": sha(args.truth),
        "registration_sha256": sha(os.path.join(OUT, "registered.txt")),
        "torch": torch.__version__, "numpy": np.__version__,
        "machine_before": before, "machine_after": after, "note": args.note,
        "again": args.again,
        "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "gate-%s%s.json" % (
        dt.datetime.now().strftime("%Y%m%d-%H%M%S"), "-smoke" if smoke else ""))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for a in runs:
        runs[a].close()
    print("  means: %s" % {a: round(v, 3) for a, v in out["means_s"].items()})
    for a in runs:
        print("  %-3s errors %s" % (a, {h: {m: "%.3e" % v for m, v in e.items()}
                                       for h, e in rec[a]["errors"].items()}))
    if not smoke:
        for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
            print("  %s %s" % (g, verdict[g]))
        print("  ALL PASS" if verdict["all"] else "  NOT ALL PASS")
    print("  ->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
