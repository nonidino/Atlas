"""The price protocol of the registration after the Stage 0 rework (`atlas/chartop/gates_twolevel.py`,
gate PR), and its dry run before the pin.

    python scripts/chartop_price_twolevel.py --dry-run --weights PATH/S/best.pt   # on the rented machine

**R2's protocol, unchanged** (`gates_squares`, whose helpers this reads):
- each price run in its own process, its BLAS threads fixed through `BLAS_VARIABLES` and
  checked, `PRICE_THREADS` torch threads;
- a sampler for AC, sleep (the gaps between samples) and the other processes' load;
- the arms in turn (`PRICE_ORDER`) within each of `PRICE_REPS` repetitions, best of them;
- per-problem counting (`run_counts2`: AC, no gap over `MAX_GAP_S`, the classical time per
  piece at most `MAX_MS_PER_PIECE` on `PRICE_CPU`);
- runs until every problem has `PRICE_RUNS` counted measurements, at most `PRICE_RUNS_MAX`;
  each problem's price the median of its first `PRICE_RUNS` counted (`first_counted`).

**The arms** (`PRICE_ORDER`): the classical expert's responses on every piece; TL, the
two-level expert through the host (`twolevel.host_expert`: its networks and strips, then each
piece's enlarged coarse level, built from the piece's own matrix and chart); TL0, U's form
without its networks (section 11.13's stand-in) inside `coarse.CoarseCorrected` with the same
coarse level, the classical expert TL is measured against; H2r, R2's reuse expert; the
classical expert again, for the A/A check; and the coupled solve every arm shares.  An arm's
price is ``(t_arm + t_coupled) / (t_classical + t_coupled)`` (`gates_hybrid.price`).

**The problems**: the held-out six at n = 64 (PR's own) and R2's sixteen (where H2r's
registered price, 0.306, was measured), so TL and H2r are priced on the same problems.
**The A/A check**: on every problem the classical expert's second timing, priced against its
first, must lie within `AA_TOLERANCE` of 1 (`gates_squares.aa_check`).

**The dry run**, before the pin: `DRY_RUNS` runs on section 11.17's twelve problems, which
have been read, on `PRICE_CPU`, with TL's networks at S's weights: TL's own do not exist yet,
and a forward's cost does not depend on the weights' values.  It passes if every problem has
`PRICE_RUNS` counted measurements and the A/A check holds (`gates_squares.dry_run_passes`).
It writes ``out/chartop/dryrun-twolevel-<time>.json`` and prints ten lines or fewer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
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
import chartop_price_squares as PS  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks as bm, coarse as co, data, gates_squares as GS  # noqa: E402
from atlas.chartop import gates_twolevel as GT, host as hs, twolevel as tl  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402
from atlas.chartop.reuse import ReuseExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
log = ev0.log
PRICE_ORDER = ("classical", "TL", "TL0", "H2r", "classical again", "coupled solve")
PRICED = ("TL", "TL0", "H2r", "AA")
#: the weights a dry run may time TL's networks with: S's (`gates_twolevel.INIT_SHA256`)
DRY_RUN_WEIGHTS_SHA256 = GT.INIT_SHA256


# ---------------------------------------------------------------------------
# the problems and the arms
# ---------------------------------------------------------------------------


def pr_specs() -> list:
    """PR's problems: the held-out six, then R2's sixteen."""
    return ([("held-out", name, problem) for name, problem in GT.HELD_OUT]
            + [("squares", str(seed), problem) for seed, problem in GS.new_problems()])


def dry_run_specs() -> list:
    return PS.dry_run_specs()


def build_problem(spec: tuple, n: int):
    """``("held-out", name, problem)``, ``("squares", seed, problem)`` or ``("workbench", name,
    problem)``."""
    kind, what, problem = spec
    if kind == "held-out":
        return fd.bench(what, "registered", GT.HELD_OUT_VARIANT, n, problem)
    return PS.build_problem(spec, n)


def make_arms(weights: str | None, seed: int = 0) -> dict:
    """TL from ``weights`` (a `train` checkpoint of S's form; ``None``: S's form with seeded
    random weights, for tests), at `gates_twolevel.N_C` with the corner treatment; TL0, U's form
    without its networks behind the same coarse level; H2r."""
    from atlas.chartop import train as trn
    if weights is None:
        torch.manual_seed(seed)
        S = tl.TwoLevelExpert(**_s_form(), n_c=None)
        with torch.no_grad():
            for prm in S.parameters():
                prm.copy_(0.02 * torch.randn(prm.shape, dtype=prm.dtype))
    else:
        S = trn.load_expert(weights, "cpu")
    T = tl.from_kernel(S, GT.N_C, corners=GT.CORNERS).eval()
    corners = co.CornerTreatment("B", frac=GT.CORNERS)
    z = LearnedExpert(fd.without_networks(EH.build_u(), EH.probe_piece()), envelope=data.ENVELOPE)
    return {"TL": tl.host_expert(T, envelope=data.ENVELOPE),
            "TL0": co.CoarseCorrected(z, GT.N_C, corners=corners),
            "reuse": ReuseExpert(GS.is_square)}


def _s_form() -> dict:
    c = dict(GT.COMMON)
    return {"form": c["form"], "backbone": c["backbone"], "size": c["size"], "kind": c["kind"],
            "m": c["m"], "p_size": c["p_size"], "bubble": c["bubble"], "strips": c["strips"],
            "hats": c["hats"]}


# ---------------------------------------------------------------------------
# a price run
# ---------------------------------------------------------------------------


def price_problem(b, arms: dict, reps: int) -> dict:
    L = b.layout(GT.KIND, GT.M_PORTS)
    ps, C = L.problems, hs.ClassicalExpert()
    resp_C = C.respond(ps, L.pb)
    run = {"classical": lambda: C.respond(ps, L.pb), "TL": lambda: arms["TL"].respond(ps, L.pb),
           "TL0": lambda: arms["TL0"].respond(ps, L.pb), "H2r": lambda: arms["reuse"].respond(ps, L.pb),
           "classical again": lambda: C.respond(ps, L.pb), "coupled solve": lambda: L.couple(resp_C)}
    times = {k: [] for k in PRICE_ORDER}
    for _ in range(reps):
        for k in PRICE_ORDER:
            t0 = time.perf_counter()
            run[k]()
            times[k].append(time.perf_counter() - t0)
    best = {k: min(v) for k, v in times.items()}
    price = {a: GS.price(best["classical again" if a == "AA" else a], best["classical"], best["coupled solve"])
             for a in PRICED}
    return {"pieces": L.P, "squares": sum(GS.is_square(p) for p in ps), "seconds": best,
            "every_repetition": times, **{f"price_{a}": v for a, v in price.items()},
            "classical_ms_per_piece": 1000.0 * best["classical"] / L.P}


def price_run(k: int, threads: int, reps: int, specs: list, n: int, out_dir: str, weights: str | None) -> dict:
    torch.set_num_threads(threads)
    pools = PS.thread_pools()
    if not GS.blas_threads_hold(pools):
        raise SystemExit(f"price run {k} refuses to time: the BLAS pools are not at {GS.BLAS_THREADS} threads "
                         f"each ({pools}); set {', '.join(GS.BLAS_VARIABLES)} before numpy loads")
    arms = make_arms(weights)
    built = [(spec, build_problem(spec, n)) for spec in specs]
    prices = {}
    with PS.Sampler({os.getpid(), os.getppid()}) as smp:
        for spec, b in built:
            prices[PS.spec_key(spec)] = price_problem(b, arms, reps)
    s = smp.samples
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s[0]["time"]))
    end = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(s[-1]["time"]))
    state = {"classical_ms_per_piece": {key: v["classical_ms_per_piece"] for key, v in prices.items()},
             "plugged": [x["plugged"] for x in s],
             "sample_gaps_s": [b_["time"] - a["time"] for a, b_ in zip(s, s[1:])]}
    rec = {"run": k, "torch_threads": threads, "reps": reps, "n": n, "start": start, "end": end,
           "machine": PS.machine(), "thread_pools": pools, "battery_reported": any(x["battery"] for x in s),
           "samples": len(s), "load_mean": float(np.mean([x["other_core_max"] for x in s])),
           "load_max": float(max(x["other_core_max"] for x in s)),
           "max_gap_s": max(state["sample_gaps_s"], default=0.0),
           "power_events": EH.power_events(start, end), "problems": prices,
           "counts": GS.run_counts2(state), "code": PS.code_identity(),
           "weights_sha256": None if weights is None else registry.sha256_file(weights)}
    registry.write_atomic(os.path.join(out_dir, f"price-run-{k}-threads{threads}.json"),
                          json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    return rec


def spawn(k: int, threads: int, specs: list, n: int, reps: int, out_dir: str, weights: str | None) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--price-run", str(k), "--threads", str(threads),
           "--price-dir", out_dir, "--n", str(n), "--reps", str(reps),
           "--problems", *[":".join(str(x) for x in s) for s in specs]]
    if weights is not None:
        cmd += ["--weights", weights]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE", **GS.blas_env())
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    path = os.path.join(out_dir, f"price-run-{k}-threads{threads}.json")
    if p.returncode != 0 or not os.path.exists(path):
        raise SystemExit(f"price run {k} failed (exit {p.returncode}):\n{p.stderr[-2000:]}")
    return json.loads(registry.read_bytes(path).decode("utf-8"))


def prices_of(recs: list, arm: str) -> dict:
    """Each problem's first `PRICE_RUNS` counted prices of ``arm``."""
    return GS.first_counted([r["counts"] for r in recs],
                            [{key: v[f"price_{arm}"] for key, v in r["problems"].items()} for r in recs])


def geomean(xs) -> float:
    return float(math.exp(np.mean(np.log(np.asarray(xs, float)))))


def price_runs(specs: list, n: int, reps: int, out_dir: str, weights: str, max_runs: int = GS.PRICE_RUNS_MAX,
               checkpoint=None) -> dict:
    """The one evaluation's runs: until every problem has `PRICE_RUNS` counted, at most
    ``max_runs``, every run kept; the A/A check on every problem."""
    keep = checkpoint or (lambda name, compute: compute())
    recs = []
    while len(recs) < max_runs and not GS.runs_suffice([r["counts"] for r in recs]):
        k = len(recs) + 1
        recs.append(keep(f"price-run-{k}-threads{GS.PRICE_THREADS}",
                         lambda: spawn(k, GS.PRICE_THREADS, specs, n, reps, out_dir, weights)))
        log(f"price run {k}: {PS.describe(recs[-1]['counts'])}")
    counts = [r["counts"] for r in recs]
    first = {a: prices_of(recs, a) for a in PRICED}
    return {"runs": recs, "tally": GS.tally(counts), "suffice": GS.runs_suffice(counts),
            "first_counted": first, "aa": GS.aa_check(first["AA"], [PS.spec_key(s) for s in specs])}


# ---------------------------------------------------------------------------
# the dry run
# ---------------------------------------------------------------------------


def dry_run(specs: list, n: int, reps: int, runs: int, out: str, weights: str | None) -> dict:
    if weights is not None and registry.sha256_file(weights) != DRY_RUN_WEIGHTS_SHA256:
        raise SystemExit(f"{weights} is not S's recorded weights ({DRY_RUN_WEIGHTS_SHA256[:16]}...)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(out, f"dryrun-twolevel-{stamp}-runs")
    os.makedirs(d)
    t0 = time.perf_counter()
    recs = []
    for k in range(1, runs + 1):
        recs.append(spawn(k, GS.PRICE_THREADS, specs, n, reps, d, weights))
        log(f"dry run {k}: {PS.describe(recs[-1]['counts'])}")
    counts = [r["counts"] for r in recs]
    t = GS.tally(counts)
    first = {a: prices_of(recs, a) for a in PRICED}
    keys = [PS.spec_key(s) for s in specs]
    aa = GS.aa_check(first["AA"], keys) if recs else None
    suffice = GS.runs_suffice(counts)
    gm = {a: (geomean([float(np.median(first[a][k])) for k in keys]) if suffice else None)
          for a in ("TL", "TL0", "H2r")}
    on_cpu = bool(recs) and all(GS.on_price_cpu(r["machine"]) for r in recs)
    registered = (n, reps, runs, weights is not None) == (GT.BENCHMARK_N, GS.PRICE_REPS, GS.DRY_RUNS, True)
    res = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "registration": GT.REGISTERED,
           "machine": recs[0]["machine"] if recs else None, "code": PS.code_identity(),
           "problems": keys, "n": n, "reps": reps, "runs": recs, "tally": t, "aa": aa,
           "on_price_cpu": on_cpu, "passes": (GS.dry_run_passes(counts, aa) and on_cpu) if registered else None,
           "first_counted": first, "geometric_mean_of_medians": gm,
           "weights_sha256": None if weights is None else DRY_RUN_WEIGHTS_SHA256,
           "elapsed_s": time.perf_counter() - t0}
    path = os.path.join(out, f"dryrun-twolevel-{stamp}.json")
    registry.write_atomic(path, json.dumps(ev0.strip(res), indent=1).encode("utf-8"))
    worst = "-" if not aa or aa["worst"] is None else f"{aa['worst']:.3f}"
    print("DRY RUN   run  holds  void  load mean")
    for r in recs:
        c = r["counts"]
        print(f"          {r['run']:3d}  {'yes' if c['run_holds'] else 'no ':5s}  "
              f"{sum(not v['counts'] for v in c['problems'].values()):4d}  {r['load_mean']:9.2f}")
    print(f"          every problem has {GS.PRICE_RUNS} counted: {suffice}; A/A: {aa['holds'] if aa else None} "
          f"(worst {worst}); on {GS.PRICE_CPU}: {on_cpu}; may be pinned: {res['passes']}")
    print("          geometric mean of medians: " + ", ".join(
        f"{a} {'-' if v is None else round(v, 3)}" for a, v in gm.items())
          + f"; wrote {os.path.relpath(path, ROOT)} ({res['elapsed_s']:.0f} s)")
    return res


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--weights", default=None, help="S's best.pt (the dry run) or TL's (the evaluation)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--problems", nargs="+", default=None, help="kind:what:problem, for tests")
    ap.add_argument("--n", type=int, default=GT.BENCHMARK_N)
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
        price_run(args.price_run, args.threads, args.reps, specs_of(args), args.n, args.price_dir, args.weights)
        return 0
    if args.dry_run:
        dry_run(specs_of(args), args.n, args.reps, args.runs, args.out, args.weights)
        return 0
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
