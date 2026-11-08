"""The price protocol of the registration after section 11.17 (`atlas/chartop/gates_squares.py`),
and its dry run before the pin.

    python scripts/chartop_price_squares.py --dry-run

The dry run is `DRY_RUNS` price runs on section 11.17's twelve problems, which have been
read, by the owner, outside the Claude app.  The protocol is pinned only if at least
`DRY_RUNS_MIN` count.  It writes ``out/chartop/dryrun-squares-<time>.json`` and prints ten
lines or fewer.

**A price run** (``--price-run``, in its own process): the experts and the problems are built
first; then a sampler records, every `SAMPLE_S` seconds through the timing, the time, AC and
the other processes' share of a core; the arms take turns in `PRICE_ORDER` within each of
`PRICE_REPS` repetitions (the classical expert's responses on every piece; H2's, the classical
solve on R2's picks and U's form without its networks on the squares; H2r's, the reuse
expert's on every piece; and the coupled solve every arm shares); and the run is judged by
`run_counts2`: the classical expert's best time per piece at most `MAX_MS_PER_PIECE`, AC at
every sample, no gap between samples over `MAX_GAP_S`.  The load and the Kernel-Power events
are recorded, not judged.

The evaluation will use `price_runs` from here: `PRICE_RUNS` counted runs, a void run run
again at most `PRICE_RERUNS_MAX` more times, every run kept.  ``--problems``, ``--n`` and
``--reps`` exist for tests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_evaluate as ev0  # noqa: E402
import chartop_evaluate_hybrid as EH  # noqa: E402
import chartop_forms_diagnose as fd  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks_squares as BS, benchmarks_workbench as BW, data  # noqa: E402
from atlas.chartop import gates_squares as GS, host as hs  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402
from atlas.chartop.reuse import ReuseExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
log = ev0.log


# ---------------------------------------------------------------------------
# the problems and the arms
# ---------------------------------------------------------------------------


def dry_run_specs() -> list:
    """Section 11.17's twelve problems, which have been read."""
    return [("workbench", name, problem) for name in BW.NAMES for problem in BW.PROBLEMS]


def build_problem(spec: tuple, n: int):
    """``spec``: ("squares", seed, problem) for a generated layout, ("workbench", name,
    problem) for one of section 11.17's."""
    kind, what, problem = spec
    if kind == "squares":
        return BS.build(BS.draw(int(what)), n, problem)
    if kind == "workbench":
        return BW.build(what, n, problem)
    raise ValueError(f"unknown problem kind {kind!r}")


def spec_key(spec: tuple) -> str:
    return f"{spec[0]} {spec[1]} {spec[2]}"


def make_arms() -> dict:
    """U's form without its networks (as section 11.13's stand-in) and the reuse expert."""
    u = EH.build_u()
    return {"Z": LearnedExpert(fd.without_networks(u, EH.probe_piece()), envelope=data.ENVELOPE),
            "reuse": ReuseExpert(GS.is_square)}


# ---------------------------------------------------------------------------
# a price run
# ---------------------------------------------------------------------------


class Sampler:
    """Every `SAMPLE_S` seconds until stopped: the wall time, AC, and the largest share of
    one core any other process used (over one second).  No process is named."""

    def __init__(self, exclude: set):
        self.exclude = exclude
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        import psutil
        bat = psutil.sensors_battery()
        self.samples.append({"time": time.time(),
                             "plugged": True if bat is None else bool(bat.power_plugged),
                             "other_core_max": EH.other_core_max(self.exclude)})

    def _loop(self):
        while not self._stop.wait(max(0.0, GS.SAMPLE_S - 1.0)):
            self.sample()

    def __enter__(self):
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.sample()


def price_problem2(b, picks: list, arms: dict, reps: int) -> dict:
    L = b.layout(GS.KIND, GS.M_PORTS)
    ps, C = L.problems, hs.ClassicalExpert()
    hard = set(picks)
    picked, squares = [ps[i] for i in range(L.P) if i in hard], [ps[i] for i in range(L.P) if i not in hard]
    resp_C = C.respond(ps, L.pb)

    def h2():
        if picked:
            C.respond(picked, L.pb)
        if squares:
            arms["Z"].respond(squares, L.pb)
    run = {"classical": lambda: C.respond(ps, L.pb), "H2": h2,
           "H2r": lambda: arms["reuse"].respond(ps, L.pb), "coupled solve": lambda: L.couple(resp_C)}
    times = {k: [] for k in GS.PRICE_ORDER}
    for _ in range(reps):
        for k in GS.PRICE_ORDER:
            t0 = time.perf_counter()
            run[k]()
            times[k].append(time.perf_counter() - t0)
    best = {k: min(v) for k, v in times.items()}
    return {"pieces": L.P, "squares": L.P - len(hard), "seconds": best, "every_repetition": times,
            "price_H2": GS.price(best["H2"], best["classical"], best["coupled solve"]),
            "price_H2r": GS.price(best["H2r"], best["classical"], best["coupled solve"]),
            "classical_ms_per_piece": 1000.0 * best["classical"] / L.P}


def price_run2(k: int, threads: int, reps: int, specs: list, n: int, out_dir: str) -> dict:
    torch.set_num_threads(threads)
    arms = make_arms()
    built = []
    for spec in specs:
        b = build_problem(spec, n)
        built.append((spec, b, GS.rule_pieces2(b.layout(GS.KIND, GS.M_PORTS))))
    prices = {}
    with Sampler({os.getpid(), os.getppid()}) as smp:
        for spec, b, picks in built:
            prices[spec_key(spec)] = price_problem2(b, picks, arms, reps)
    s = smp.samples
    start = dt.datetime.fromtimestamp(s[0]["time"]).isoformat(timespec="seconds")
    end = dt.datetime.fromtimestamp(s[-1]["time"]).isoformat(timespec="seconds")
    state = {"classical_ms_per_piece": {key: v["classical_ms_per_piece"] for key, v in prices.items()},
             "plugged": [x["plugged"] for x in s],
             "sample_gaps_s": [b_["time"] - a["time"] for a, b_ in zip(s, s[1:])]}
    rec = {"run": k, "torch_threads": threads, "reps": reps, "n": n, "start": start, "end": end,
           "samples": len(s), "load_mean": float(np.mean([x["other_core_max"] for x in s])),
           "load_max": float(max(x["other_core_max"] for x in s)),
           "max_gap_s": max(state["sample_gaps_s"], default=0.0),
           "power_events": EH.power_events(start, end), "problems": prices,
           "counts": GS.run_counts2(state)}
    registry.write_atomic(os.path.join(out_dir, f"price-run-{k}-threads{threads}.json"),
                          json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    return rec


def spawn(k: int, threads: int, specs: list, n: int, reps: int, out_dir: str) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--price-run", str(k), "--threads", str(threads),
           "--price-dir", out_dir, "--n", str(n), "--reps", str(reps),
           "--problems", *[":".join(str(x) for x in s) for s in specs]]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    path = os.path.join(out_dir, f"price-run-{k}-threads{threads}.json")
    if p.returncode != 0 or not os.path.exists(path):
        raise SystemExit(f"price run {k} failed (exit {p.returncode}):\n{p.stderr[-2000:]}")
    return json.loads(registry.read_bytes(path).decode("utf-8"))


def geomean(xs) -> float:
    return float(math.exp(np.mean(np.log(np.asarray(xs, float)))))


def price_runs(specs: list, n: int, reps: int, out_dir: str, runs: int = GS.PRICE_RUNS,
               reruns: int = GS.PRICE_RERUNS_MAX, checkpoint=None) -> dict:
    """`runs` counted runs, a void run run again at most `reruns` more times; every run
    kept.  ``checkpoint(name, compute)`` keeps each run whole when given."""
    keep = checkpoint or (lambda name, compute: compute())
    all_, counted, k = [], [], 0
    while len(counted) < runs and k < runs + reruns:
        k += 1
        rec = keep(f"price-run-{k}-threads{GS.PRICE_THREADS}",
                   lambda: spawn(k, GS.PRICE_THREADS, specs, n, reps, out_dir))
        all_.append(rec)
        log(f"price run {k}: {'counts' if rec['counts']['counts'] else 'void: ' + '; '.join(rec['counts']['failed'])}")
        if rec["counts"]["counts"]:
            counted.append(rec)
    return {"runs": all_, "counted": [r["run"] for r in counted]}


# ---------------------------------------------------------------------------
# the dry run
# ---------------------------------------------------------------------------


def dry_run(specs: list, n: int, reps: int, runs: int, out: str) -> dict:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(out, f"dryrun-squares-{stamp}-runs")
    os.makedirs(d)
    t0 = time.perf_counter()
    recs = []
    for k in range(1, runs + 1):
        recs.append(spawn(k, GS.PRICE_THREADS, specs, n, reps, d))
        log(f"dry run {k}: {'counts' if recs[-1]['counts']['counts'] else 'void: ' + '; '.join(recs[-1]['counts']['failed'])}")
    counted = [r for r in recs if r["counts"]["counts"]]
    gm = {arm: [geomean([v[f"price_{arm}"] for v in r["problems"].values()]) for r in recs] for arm in ("H2", "H2r")}
    gm_c = [geomean([v["price_H2"] for v in r["problems"].values()]) for r in counted]
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "registration": GS.REGISTERED,
           "problems": [spec_key(s) for s in specs], "n": n, "reps": reps, "runs": recs,
           "counted": [r["run"] for r in counted],
           "passes": GS.dry_run_passes(len(counted), runs) if (n, reps) == (GS.BENCHMARK_N, GS.PRICE_REPS) else None,
           "geometric_mean_by_run": gm,
           "counted_spread_H2": [min(gm_c), max(gm_c)] if gm_c else None,
           "elapsed_s": time.perf_counter() - t0}
    path = os.path.join(out, f"dryrun-squares-{stamp}.json")
    registry.write_atomic(path, json.dumps(ev0.strip(res), indent=1).encode("utf-8"))
    print("DRY RUN   run  counted  H2     H2r    load mean  why not")
    for r, g2, gr in zip(recs, gm["H2"], gm["H2r"]):
        print(f"          {r['run']:3d}  {'yes' if r['counts']['counts'] else 'no ':7s}  {g2:5.3f}  {gr:5.3f}  "
              f"{r['load_mean']:9.2f}  {'; '.join(r['counts']['failed'])[:60]}")
    print(f"          {len(counted)} of {runs} counted; the protocol may be pinned: {res['passes']}; counted runs' "
          f"H2 geometric mean {'-' if not gm_c else f'{min(gm_c):.3f}-{max(gm_c):.3f}'}")
    print(f"          wrote {os.path.relpath(path, ROOT)} ({res['elapsed_s']:.0f} s)")
    return res


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--problems", nargs="+", default=None, help="kind:what:problem, for tests")
    ap.add_argument("--n", type=int, default=GS.BENCHMARK_N)
    ap.add_argument("--reps", type=int, default=GS.PRICE_REPS)
    ap.add_argument("--runs", type=int, default=GS.DRY_RUNS)
    ap.add_argument("--price-run", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, default=GS.PRICE_THREADS, help=argparse.SUPPRESS)
    ap.add_argument("--price-dir", default=None, help=argparse.SUPPRESS)
    return ap


def specs_of(args) -> list:
    if not args.problems:
        return dry_run_specs()
    return [tuple(s.split(":")) for s in args.problems]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.price_run is not None:
        price_run2(args.price_run, args.threads, args.reps, specs_of(args), args.n, args.price_dir)
        return 0
    if args.dry_run:
        dry_run(specs_of(args), args.n, args.reps, args.runs, args.out)
        return 0
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
