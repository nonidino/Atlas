"""The coolant registration's price protocol (`atlas/chartop/gates_cot.py`, PR: reported, not
a gate since the owner's amendment of 2026-11-19), and its dry run before the pin.

    python scripts/cot_price.py --dry-run                    # on the rented whole machine
    python scripts/cot_price.py --evaluate --weights W.pt --form free --size 1M

**R2's protocol** (`gates_squares`, whose helpers this reads), adapted to CO-T:
- each price run in its own process (``--run K``), its BLAS threads fixed through
  `gates_squares.BLAS_VARIABLES` before numpy loads and checked, `gates_squares.PRICE_THREADS`
  torch threads;
- a sampler for AC and sleep (`chartop_price_squares.Sampler`);
- the arms in turn (`ORDER`) within each of `REPS` repetitions, best of them;
- a run's measurements count when the machine stayed on AC with no gap over
  `gates_squares.MAX_GAP_S` (`gates_squares.run_counts2`).  **No per-piece time bound**: CO-T's
  problems run at n = 32 to 128, whose classical piece solves differ fortyfold, so one bound
  does not fit; the A/A check carries the validity;
- runs until every problem has `gates_squares.PRICE_RUNS` counted measurements, at most
  `gates_squares.PRICE_RUNS_MAX`; a problem's price is the median of its first counted ones.

**The arms** (`ORDER`): the classical CO-T expert's responses on every piece; TL0 at the
network's coarse size (`gates_cot.n_c`); TL0 at 32 where n >= 64; finer cutting (the classical
expert on twice the columns and two rows at half the grid, its own coupled solve included);
the learned arm through the host (`transport_torch.HostExpert`), and the same with the safety
switch (each piece's exact dual residual for its answer and for TL0's,
`transport_switch.piece_eta`); the classical expert again, for the A/A check; and the coupled
solve every arm on the registered cut shares.  An arm's price is
``(t_arm + t_coupled) / (t_classical + t_coupled)``; finer cutting's is its whole time over the
same denominator.

**The A/A check**: on every problem, the classical expert's second timing over its first, the
median of the counted runs, within `gates_squares.AA_TOLERANCE` of 1.

**The dry run**, before the pin: `gates_squares.DRY_RUNS` runs on the six development problems
(read already), with the learned arm untrained (a forward's cost does not depend on its weights'
values).  It passes if every problem has its counted measurements and the A/A check holds.
Writes ``out/chartop/cot/price-<dry|eval>-<time>.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUT = os.path.join(ROOT, "out", "chartop", "cot")
ORDER = ("classical", "TL0", "TL0-32", "finer", "learned", "learned+switch", "classical again", "coupled solve")
REPS = 3


def _problems(which: str) -> list:
    if which == "dev":
        from atlas.studio import cot_layouts as CL
        return [("dev", n, t) for n in CL.NAMES for t in CL.TIERS]
    from atlas.studio import cot_eval_layouts as EL
    return [("eval", n, t) for n in EL.NAMES for t in EL.TIERS]


def _layout(kind: str, name: str, tier: str):
    from atlas.studio import cot
    from atlas.studio import cot_eval_layouts as EL
    from atlas.studio import cot_layouts as CL
    mod = CL if kind == "dev" else EL
    spec, _ = mod.build(name, tier)
    base = mod.cut(spec, name, tier)
    cuts = CL.CUT if kind == "dev" else EL.CUT
    fine = CL.cut(spec, name, tier, cols_factor=2, rows=2, cuts=cuts)
    return cot.layout(base, spec), cot.layout(fine, spec)


def _arm(args):
    import torch
    from atlas.chartop import transport_torch as tt
    e = tt.ArmR(args.size, args.form, seed=0)
    if args.weights:
        st = torch.load(args.weights, map_location="cpu", weights_only=False)
        e.load_state_dict(st.get("model", st))
    else:                                   # the dry run: untrained, its head not zero
        with torch.no_grad():
            for prm in e.net.inner.head.parameters():
                prm.normal_(0.0, 0.02)
    e.eval()
    return e


def price_problem(kind, name, tier, arm, reps: int) -> dict:
    from atlas.chartop import gates_cot as GC
    from atlas.chartop import transport as tr
    from atlas.chartop import transport_coarse as tco
    from atlas.chartop import transport_switch as sw
    from atlas.chartop import transport_torch as tt
    Lb, Lf = _layout(kind, name, tier)
    n = Lb.problems[0].n
    C = tr.ClassicalTransportExpert()
    resp_C = C.respond(Lb.problems, Lb.pb)
    nc = GC.n_c(n)
    learned = tt.HostExpert(arm, n_c=nc)

    def switched():
        resp = learned.respond(Lb.problems, Lb.pb)
        t0 = tco.TL0TransportExpert(n_c=nc)
        for p, r in zip(Lb.problems, resp):
            Q = Lb.pb(p.n).Q
            sw.piece_eta(p, r, Q, "exact", n_c=nc)
            sw.piece_eta(p, t0.response(p, Q), Q, "exact", n_c=nc)
        return resp

    run = {"classical": lambda: C.respond(Lb.problems, Lb.pb),
           "TL0": lambda: tco.TL0TransportExpert(n_c=nc).respond(Lb.problems, Lb.pb),
           "TL0-32": (lambda: tco.TL0TransportExpert(n_c=32).respond(Lb.problems, Lb.pb)) if n >= 64 else None,
           "finer": lambda: Lf.couple(C.respond(Lf.problems, Lf.pb)),
           "learned": lambda: learned.respond(Lb.problems, Lb.pb),
           "learned+switch": switched,
           "classical again": lambda: C.respond(Lb.problems, Lb.pb),
           "coupled solve": lambda: Lb.couple(resp_C)}
    times = {k: [] for k in ORDER if run[k] is not None}
    for _ in range(reps):
        for k in ORDER:
            if run[k] is None:
                continue
            t0 = time.perf_counter()
            run[k]()
            times[k].append(time.perf_counter() - t0)
    best = {k: min(v) for k, v in times.items()}
    den = best["classical"] + best["coupled solve"]
    price = {k: (best[k] + best["coupled solve"]) / den for k in best if k not in ("coupled solve", "finer")}
    price["finer"] = best["finer"] / den
    return {"problem": f"{name} {tier}", "n": n, "pieces": len(Lb.problems), "seconds": best,
            "every_repetition": times, "price": price, "aa": best["classical again"] / best["classical"],
            "classical_ms_per_piece": 1000.0 * best["classical"] / len(Lb.problems)}


def price_run(k: int, args) -> dict:
    import numpy as np  # noqa: F401  (after the BLAS variables)
    import torch
    import chartop_price_squares as PS
    from atlas.chartop import gates_squares as GS
    torch.set_num_threads(GS.PRICE_THREADS)
    pools = PS.thread_pools()
    if not GS.blas_threads_hold(pools):
        raise SystemExit(f"price run {k} refuses to time: the BLAS pools are not at {GS.BLAS_THREADS} threads")
    arm = _arm(args)
    rows = {}
    with PS.Sampler({os.getpid()}) as smp:
        for kind, name, tier in _problems(args.problems):
            if args.only and f"{name}:{tier}" not in args.only:
                continue
            rows[f"{name} {tier}"] = price_problem(kind, name, tier, arm, args.reps)
    gaps = [b["time"] - a["time"] for a, b in zip(smp.samples, smp.samples[1:])]
    state = {"classical_ms_per_piece": {key: r["classical_ms_per_piece"] for key, r in rows.items()},
             "plugged": [s["plugged"] for s in smp.samples], "sample_gaps_s": gaps}
    counts = GS.run_counts2(state, max_ms=float("inf"))
    return {"run": k, "machine": PS.machine(), "pools": pools, "torch_threads": torch.get_num_threads(),
            "problems": rows, "counts": counts, "samples": len(smp.samples)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--run", type=int, help="one price run, in this process (called by the others)")
    ap.add_argument("--problems", choices=("dev", "eval"), default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--form", default="free")
    ap.add_argument("--size", default="1M")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="NAME:TIER problems only (a smoke test)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if a.run is not None:
        print(json.dumps(price_run(a.run, a)))
        return 0
    from atlas.chartop import gates_squares as GS
    if a.evaluate and not a.weights:
        ap.error("--evaluate needs --weights")
    problems = a.problems or ("dev" if a.dry_run else "eval")
    runs_max = a.runs or (GS.DRY_RUNS if a.dry_run else GS.PRICE_RUNS_MAX)
    env = dict(os.environ, **GS.blas_env())
    results, counts = [], []
    for k in range(runs_max):
        cmd = [sys.executable, os.path.abspath(__file__), "--run", str(k), "--problems", problems,
               "--form", a.form, "--size", a.size, "--reps", str(a.reps)] + (["--weights", a.weights] if a.weights else []) \
            + (["--only", *a.only] if a.only else [])
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"price run {k} failed: {out.stderr[-2000:]}")
        r = json.loads(out.stdout.strip().splitlines()[-1])
        results.append(r)
        counts.append(r["counts"])
        print(f"run {k}: holds {r['counts']['run_holds']}", flush=True)
        if a.evaluate and GS.runs_suffice(counts):
            break
    keys = list(results[0]["problems"])
    vals = [{key: r["problems"][key]["price"] for key in keys} for r in results]
    aas = [{key: r["problems"][key]["aa"] for key in keys} for r in results]
    first = GS.first_counted(counts, vals)
    first_aa = GS.first_counted(counts, aas)
    import numpy as np
    med = {key: {arm: float(np.median([v[arm] for v in got])) for arm in got[0]} for key, got in first.items() if got}
    aa = {key: {"median": float(np.median(v)) if len(v) == GS.PRICE_RUNS else None} for key, v in first_aa.items()}
    for v in aa.values():
        v["holds"] = v["median"] is not None and abs(v["median"] - 1.0) <= GS.AA_TOLERANCE
    aa_all = {"holds": all(v["holds"] for v in aa.values()), "problems": aa}
    rec = {"what": "the coolant registration's price protocol" + (" (dry run)" if a.dry_run else ""),
           "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "problems_set": problems, "reps": a.reps, "runs": len(results), "arm": {"form": a.form, "size": a.size,
           "weights": a.weights or "untrained (dry run)"}, "prices": med, "aa": aa_all, "tally": GS.tally(counts),
           "results": results}
    if a.dry_run:
        rec["dry_run_passes"] = len(counts) == GS.DRY_RUNS and GS.runs_suffice(counts) and aa_all["holds"]
    os.makedirs(a.out, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(a.out, f"price-{'dry' if a.dry_run else 'eval'}-{stamp}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    print(f"{os.path.relpath(path, ROOT)}: A/A {'holds' if aa_all['holds'] else 'FAILS'}"
          + (f"; dry run {'passes' if rec['dry_run_passes'] else 'FAILS'}" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
