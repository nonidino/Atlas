"""The price protocol of the registration after section 11.17 (`atlas/chartop/gates_squares.py`),
and its dry run before the pin.

    python scripts/chartop_price_squares.py --dry-run

The dry run is `DRY_RUNS` price runs on section 11.17's twelve problems, which have been
read, on a rented whole machine (`gates_squares.REVISED`).  It passes if every problem has
`PRICE_RUNS` counted measurements among them (`dry_run_passes`).  It writes
``out/chartop/dryrun-squares-<time>.json`` and prints ten lines or fewer.

**A price run** (``--price-run``, in its own process): the experts and the problems are built
first; then a sampler records, every `SAMPLE_S` seconds through the timing, the time, AC
(and whether a battery is reported at all) and the other processes' share of a core; the
arms take turns in `PRICE_ORDER` within each of `PRICE_REPS` repetitions (the classical
expert's responses on every piece; H2's, the classical solve on R2's picks and U's form
without its networks on the squares; H2r's, the reuse expert's on every piece; and the
coupled solve every arm shares); and `run_counts2` says which of its measurements count:
those of a run that held AC at every sample and had no gap between samples over
`MAX_GAP_S`, on a problem whose classical best time per piece is at most
`MAX_MS_PER_PIECE`.  The load, the Kernel-Power events and the machine are recorded, not
judged; no host and no process is named.

The evaluation will use `price_runs` from here: runs until every problem has `PRICE_RUNS`
counted measurements, at most `PRICE_RUNS_MAX` in all, every run kept; each problem's price
the median of its first `PRICE_RUNS` counted.  ``--problems``, ``--n``, ``--reps`` and
``--runs`` exist for tests.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
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

import chartop_cpus  # noqa: E402
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


def cpu_model() -> str:
    """The CPU's model name: what the one evaluation's price runs must match."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def code_identity():
    """The commit the code came from: the payload's ``IDENTITY`` on a box
    (`chartop_price_box_payload`), else git's HEAD here, marked ``+dirty`` when ``atlas/`` or
    ``scripts/`` differ from it; None when neither can say."""
    path = os.path.join(ROOT, "IDENTITY")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read().strip()
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no", "--",
                                "atlas", "scripts"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if head.returncode != 0 or dirty.returncode != 0:
        return None
    return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def machine() -> dict:
    """What a price run ran on.  No host name and no process: the record is public."""
    import scipy
    return {"system": platform.system(), "release": platform.release(), "cpu": cpu_model(),
            "cpu_sources": chartop_cpus.sources(), "available_cpus": chartop_cpus.available_cpus(),
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__}


class Sampler:
    """Every `SAMPLE_S` seconds until stopped: the wall time, AC, whether a battery is reported
    at all (with none, AC holds vacuously), and the largest share of one core any other process
    used (over one second).  No process is named."""

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
                             "battery": bat is not None,
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
           "machine": machine(), "battery_reported": any(x["battery"] for x in s),
           "samples": len(s), "load_mean": float(np.mean([x["other_core_max"] for x in s])),
           "load_max": float(max(x["other_core_max"] for x in s)),
           "max_gap_s": max(state["sample_gaps_s"], default=0.0),
           "power_events": EH.power_events(start, end), "problems": prices,
           "counts": GS.run_counts2(state), "code": code_identity()}
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


def describe(c: dict) -> str:
    """One line for a run's `run_counts2`."""
    if not c["run_holds"]:
        return "void: " + "; ".join(c["failed"])
    void = [k for k, v in c["problems"].items() if not v["counts"]]
    return "every measurement counts" if not void else f"{len(void)} void: " + ", ".join(void)


def prices_of(recs: list, arm: str) -> dict:
    """Each problem's first `PRICE_RUNS` counted prices of ``arm`` ("H2" or "H2r")."""
    return GS.first_counted([r["counts"] for r in recs],
                            [{key: v[f"price_{arm}"] for key, v in r["problems"].items()} for r in recs])


def price_runs(specs: list, n: int, reps: int, out_dir: str, max_runs: int = GS.PRICE_RUNS_MAX,
               checkpoint=None) -> dict:
    """Runs until every problem has `PRICE_RUNS` counted measurements, at most ``max_runs`` in
    all; every run kept.  ``checkpoint(name, compute)`` keeps each run whole when given."""
    keep = checkpoint or (lambda name, compute: compute())
    recs = []
    while len(recs) < max_runs and not GS.runs_suffice([r["counts"] for r in recs]):
        k = len(recs) + 1
        rec = keep(f"price-run-{k}-threads{GS.PRICE_THREADS}",
                   lambda: spawn(k, GS.PRICE_THREADS, specs, n, reps, out_dir))
        recs.append(rec)
        log(f"price run {k}: {describe(rec['counts'])}")
    counts = [r["counts"] for r in recs]
    return {"runs": recs, "tally": GS.tally(counts), "suffice": GS.runs_suffice(counts),
            "prices_H2": prices_of(recs, "H2"), "prices_H2r": prices_of(recs, "H2r")}


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
        log(f"dry run {k}: {describe(recs[-1]['counts'])}")
    counts = [r["counts"] for r in recs]
    t = GS.tally(counts)
    third = {key: next((i + 1 for i in range(len(counts)) if GS.tally(counts[:i + 1])[key] >= GS.PRICE_RUNS),
                       None) for key in t}
    first = {arm: prices_of(recs, arm) for arm in ("H2", "H2r")}
    pr = {arm: (geomean([float(np.median(v)) for v in first[arm].values()]) if GS.runs_suffice(counts) else None)
          for arm in ("H2", "H2r")}
    gm = {arm: [geomean([v[f"price_{arm}"] for v in r["problems"].values()]) for r in recs] for arm in ("H2", "H2r")}
    registered = (n, reps, runs) == (GS.BENCHMARK_N, GS.PRICE_REPS, GS.DRY_RUNS)
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "registration": GS.REGISTERED,
           "machine": recs[0]["machine"] if recs else None, "code": code_identity(),
           "problems": [spec_key(s) for s in specs], "n": n, "reps": reps, "runs": recs,
           "tally": t, "run_of_third_count": third,
           "passes": GS.dry_run_passes(counts) if registered else None,
           "first_counted": first, "geometric_mean_of_medians": pr,
           "geometric_mean_by_run": gm,
           "elapsed_s": time.perf_counter() - t0}
    path = os.path.join(out, f"dryrun-squares-{stamp}.json")
    registry.write_atomic(path, json.dumps(ev0.strip(res), indent=1).encode("utf-8"))
    fewest = min(t, key=t.get) if t else None
    print("DRY RUN   run  holds  void  H2 (all)  H2r (all)  load mean")
    for r, g2, gr in zip(recs, gm["H2"], gm["H2r"]):
        c = r["counts"]
        void = sum(not v["counts"] for v in c["problems"].values())
        print(f"          {r['run']:3d}  {'yes' if c['run_holds'] else 'no ':5s}  {void:4d}  {g2:8.3f}  {gr:9.3f}  "
              f"{r['load_mean']:9.2f}")
    print(f"          every problem has {GS.PRICE_RUNS} counted: {GS.runs_suffice(counts)} (fewest {t.get(fewest)}, "
          f"{fewest}); may be pinned: {res['passes']}")
    print(f"          median-of-first-{GS.PRICE_RUNS} geometric mean: H2 {pr['H2'] if pr['H2'] is None else round(pr['H2'], 3)}, "
          f"H2r {pr['H2r'] if pr['H2r'] is None else round(pr['H2r'], 3)}; wrote {os.path.relpath(path, ROOT)} "
          f"({res['elapsed_s']:.0f} s)")
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
