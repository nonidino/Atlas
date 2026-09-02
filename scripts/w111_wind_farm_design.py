"""W111 -- PoC 1a: differentiable wind-farm design through the composed graph.

    python scripts/w111_wind_farm_design.py --stage timing
    python scripts/w111_wind_farm_design.py --expert poseidon --stage grad --case K12
    python scripts/w111_wind_farm_design.py --stage confirm --case K25 --steps 60
    python scripts/w111_wind_farm_design.py --stage grad --case K12 --opt-steps 30
    python scripts/w111_wind_farm_design.py --stage cma  --case K12 --budget 900
    python scripts/w111_wind_farm_design.py --stage fd   --case K12
    python scripts/w111_wind_farm_design.py --stage sensitivity --case K12
    python scripts/w111_wind_farm_design.py --stage verify --case K12
    python scripts/w111_wind_farm_design.py --expert poseidon --stage crosseval --case K12
    python scripts/w111_wind_farm_design.py --stage figures
    python scripts/w111_wind_farm_design.py --stage merge

`--expert` chooses the fluid agent in every window, and it is the whole of the
substitution: `reference_exposed` (the default) is `reference.WindowNS` with its
elliptic part handed to the composition layer, i.e. the 2026-09-01 column of
[[poc1-results-differentiable-design]]; `poseidon` is the frozen 20.8M-parameter
checkpoint, i.e. the half of the novelty claim that column does not exercise
([[poc1a-frozen-expert-results]]).  Artefacts land in `out/w111/` and `out/w118/`
respectively -- separate directories rather than a filename suffix, so the merge
scan needs no filtering and the two columns cannot contaminate each other's
record.

Two stages exist to compare columns rather than to measure one.  `verify` is the
classical verification panel -- the composed column against
`scaling_ladder.reference_monolith`, the undivided classical solver, alternating
one macro-step each so neither is timed against the other's load.  `crosseval`
scores three layouts (the grid, each column's own optimum) by all three, which is
the deployment story of [[prior-art-and-novelty-atlas-0.1]] section 5 measured
end to end.

Every stage is its own process and writes its own artefact the moment it has one,
and the two optimiser stages append to theirs after **every** step.  That is not
tidiness: an Adam step at K = 25 is ten minutes on a GPU-less desktop and a
CMA-ES budget is hours, and a run that only writes at the end is a run whose
failure costs everything it had already measured.

The stages are separable on purpose, so the long ones can be launched
side by side.  Where they were, `timing` is the uncontended measurement the
per-evaluation costs are quoted from and the contended wall-clock is reported
beside it rather than instead of it.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                  # noqa: E402

from atlas.cases import wake_array as wa                            # noqa: E402
from atlas.cases import wind_farm_design as wd                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "w111")

#: Which fluid expert every stage marches, and where its artefacts land.  Both
#: are set from ``--expert`` in `main` and are module globals rather than
#: arguments threaded through nine stages, because every stage needs them and
#: none of them chooses them.  ``reference_exposed`` is the 2026-09-01 run and
#: its artefact paths are unchanged to the character, so this file still
#: reproduces `poc1-results-differentiable-design` exactly as it did.
EXPERT = "reference_exposed"

#: What the merged record is called, per expert.  W118 is the `gap-worklist`
#: row this column closes; W113-W117 are Tier 23/24's and are taken.
MERGED = {"reference_exposed": "w111.json", "poseidon": "w118.json"}

#: Where each expert's artefacts live.  A separate directory rather than a file
#: suffix, so `stage_merge`'s directory scan and `stage_figures`' lookups need no
#: filtering and the two columns cannot contaminate each other's record.
OUT_FOR = {"reference_exposed": os.path.join(ROOT, "out", "w111"),
           "poseidon": os.path.join(ROOT, "out", "w118")}

#: The band `|u|` must stay inside for the column to count as stable.  Exactly
#: `w100_scaling_ladder.stage_long_march`'s 3.0, so the confirmation march at
#: N24 is read against the same rule that passed N12 to 120 macro-steps.
BAND = 3.0


def _f(x):
    """JSON-safe."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_f(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def write(name: str, payload: dict) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_f(payload), fh, indent=1)
    os.replace(tmp, path)
    return path


def banner(s: str) -> None:
    print()
    print("=" * 78)
    print(s)
    print("=" * 78, flush=True)


def machine() -> dict:
    import torch
    return {"platform": platform.platform(), "processor": platform.processor(),
            "cores": os.cpu_count(), "torch": torch.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "torch_threads": torch.get_num_threads(),
            "numpy": np.__version__, "dtype": "float64",
            "fluid_expert": EXPERT}


def build(case_name: str, steps: int | None, threads: int, device: str = "cpu",
          expert: str | None = None):
    """The case and its rollout, for whichever fluid expert is selected.

    `wd.rollout_for` is the whole swap -- `atlas-proof-of-concept-1` section 9
    says the frozen-checkpoint run "is a ``kind=`` argument", and this is the
    line where that is true.
    """
    import torch
    torch.set_num_threads(threads)
    case = wd.case_for(case_name, steps)
    return case, wd.rollout_for(case, expert or EXPERT, device=device,
                                threads=threads)


# ---------------------------------------------------------------------------
# the worker pool -- the derivative-free baseline's own parallelism
# ---------------------------------------------------------------------------

_POOL_STATE: dict = {}


def _pool_init(case_name: str, steps: int | None, threads: int,
               device: str = "cpu", expert: str = "reference_exposed") -> None:
    # `expert` is passed rather than read off the module global: the pool spawns
    # rather than forks (see `Pool`), so a child re-imports this module fresh and
    # would get the DEFAULT expert while the parent marched the other one.  A
    # baseline that silently optimised a different objective than the gradient
    # method is the one failure this comparison could not survive.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    torch.set_num_threads(threads)
    case = wd.case_for(case_name, steps)
    _POOL_STATE["rollout"] = wd.rollout_for(case, expert, device=device,
                                            threads=threads)
    _POOL_STATE["case"] = case


def _pool_eval(theta_list: list) -> list:
    ro = _POOL_STATE["rollout"]
    return [wd.value_only(np.asarray(t, dtype=float), ro)[0] for t in theta_list]


class Pool:
    """`evaluate(list_of_theta) -> list_of_J`, over `workers` processes.

    Measured on the development box: one process at 8 threads does 0.95
    macro-steps/s and eight processes at 2 threads do 4.80 between them, so the
    composed step is bound by per-operation overhead rather than by memory
    bandwidth and the population scales almost linearly.  The derivative-free
    baseline is given that speedup -- it is the honest version of the
    comparison, and the evaluation COUNT the headline quotes does not move.
    """

    def __init__(self, case_name: str, steps: int | None, workers: int,
                 threads: int, device: str = "cpu",
                 expert: str | None = None) -> None:
        self.workers = workers
        self.threads = threads
        self.expert = expert or EXPERT
        # **spawn, not fork.**  A CUDA context does not survive `fork`, and on
        # Linux `ProcessPoolExecutor` forks by default, so a pool that worked on
        # CPU would produce workers that cannot see the GPU -- or, worse, one
        # that half-works.  Spawn re-imports this module in each child, which is
        # why everything below the stage functions is import-safe.
        self.ex = ProcessPoolExecutor(
            max_workers=workers, initializer=_pool_init,
            initargs=(case_name, steps, threads, device, self.expert),
            mp_context=multiprocessing.get_context("spawn"))
        self.n = 0

    def evaluate(self, batch: list) -> list:
        chunks = [[] for _ in range(self.workers)]
        for i, t in enumerate(batch):
            chunks[i % self.workers].append(np.asarray(t, dtype=float).tolist())
        futs = [self.ex.submit(_pool_eval, c) for c in chunks if c]
        got = [f.result() for f in futs]
        # back into the caller's order: candidate i went to worker i % workers,
        # and each worker kept its chunk's order, so one cursor per worker
        # rebuilds the generation exactly.  Order matters -- CMA-ES ranks the
        # population and a shuffled ranking is a different algorithm.
        cursor = [0] * len(got)
        out = []
        for i in range(len(batch)):
            w = i % len(got)
            out.append(got[w][cursor[w]])
            cursor[w] += 1
        self.n += len(batch)
        return out

    def close(self) -> None:
        self.ex.shutdown(wait=True)


# ---------------------------------------------------------------------------
# 1. timing -- uncontended, and it is what every wall-clock claim is quoted from
# ---------------------------------------------------------------------------


def stage_timing(args) -> dict:
    banner("1. timing -- the per-macro-step cost of the composed column")
    out = {"machine": machine(), "cases": {}}
    for name in ("K12", "K25"):
        case, ro = build(name, args.steps, args.threads, args.device)
        th = wd.initial_design(case)
        probe = 4
        wd.value_only(th, ro, steps=probe)                        # warm
        t0 = time.perf_counter()
        wd.value_only(th, ro, steps=probe)
        t_fwd = (time.perf_counter() - t0) / probe
        t0 = time.perf_counter()
        wd.value_and_grad(th, ro, steps=probe)
        t_grad = (time.perf_counter() - t0) / probe
        # the numpy column the case studies actually ran, for the same step:
        # `exposed_reference_solver.step_batch` for the classical expert and
        # `FrozenFluidExpert.step_many` for the checkpoint.  Same batch, same
        # window count, so the two rows are the agent's own cost with the
        # composition layer taken out of both.
        us = np.ones((ro.n_win, wa.N, wa.N))
        vs = np.zeros_like(us)
        fs = np.zeros_like(us)
        if EXPERT == "poseidon":
            ex = wa.scaled_expert()
            call = lambda: ex.step_many(us - wa.U_INF, vs, wa.MACRO_DT,
                                        galilean=False, project=False)
        else:
            ex = wa.exposed_reference_solver(wa.NU_REF)
            call = lambda: ex.step_batch(us, vs, wa.MACRO_DT, bc0=None,
                                         force=(fs, fs))
        call()
        t0 = time.perf_counter()
        call()
        t_numpy = time.perf_counter() - t0
        row = {
            "case": case.as_dict(), "threads": args.threads,
            "s_per_macro_step_forward": t_fwd,
            "s_per_macro_step_gradient": t_grad,
            "gradient_over_forward": t_grad / t_fwd,
            "s_per_macro_step_numpy_agent_only": t_numpy,
            "s_per_macro_step_numpy_solver_only": t_numpy,   # legacy key
            "fluid_expert": EXPERT,
            "s_per_rollout_forward": t_fwd * case.steps,
            "s_per_rollout_gradient": t_grad * case.steps,
            "n_windows": ro.n_win,
        }
        out["cases"][name] = row
        print(f"  {name}: forward {t_fwd:.3f} s/macro-step, gradient "
              f"{t_grad:.3f} s/macro-step ({t_grad / t_fwd:.2f}x), "
              f"numpy solver alone {t_numpy:.3f} s", flush=True)
        print(f"        one {case.steps}-step rollout: forward "
              f"{t_fwd * case.steps:.1f} s, gradient {t_grad * case.steps:.1f} s",
              flush=True)
    write("timing.json", out)
    return out


# ---------------------------------------------------------------------------
# 2. confirm -- is the exposed column stable at this rung over this horizon?
# ---------------------------------------------------------------------------


def stage_confirm(args) -> dict:
    """The march W100 never ran: N24, with the disks live, past 20 macro-steps.

    `w100_scaling_ladder.stage_long_march` took the exposed agents plus the
    projected assembly to 120 macro-steps at N1, N2, N6 and N12 and stopped
    there.  This PoC optimises at N24 and every objective evaluation is a march
    on that rung, so the band has to be confirmed BEFORE any optimiser runs and
    the result is reported whatever it says.

    Read against `stage_long_march`'s own rule: `|u|` inside a band of 3.0, the
    state finite throughout, and the assembled divergence flat or falling over
    the developed part of the trace rather than accumulating.
    """
    banner(f"2. confirmation march -- {args.case} at {args.steps or 'default'} "
           f"macro-steps, disks live")
    import torch
    case, ro = build(args.case, args.steps, args.threads, args.device)
    th = wd.initial_design(case)
    row = {"case": case.as_dict(), "band": BAND, "u_max": [], "div_rms": [],
           "farm_power": [], "wall_s": [], "diverged_at": None, "finite_to": 0,
           "theta0": wd.design_to_dict(th)}
    print(f"  rung N{case.n_windows}, {case.shape[1]}x{case.shape[0]} cells, "
          f"K={case.k}, {case.steps} macro-steps", flush=True)
    u = torch.full(case.shape, wa.U_INF, dtype=wd.TORCH_DTYPE,
                   device=args.device)
    v = torch.zeros_like(u)
    theta = torch.as_tensor(th, dtype=wd.TORCH_DTYPE, device=args.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for k in range(case.steps):
            u, v, power, _un = ro.macro_step(u, v, theta)
            fin = bool(torch.isfinite(u).all() and torch.isfinite(v).all())
            if not fin:
                row["not_finite_at"] = k + 1
                row["diverged_at"] = row["diverged_at"] or k + 1
                break
            row["finite_to"] = k + 1
            umax = float(u.abs().max())
            un_, vn_ = u.cpu().numpy(), v.cpu().numpy()
            row["u_max"].append(umax)
            row["div_rms"].append(wa.divergence_rms(un_, vn_))
            row["farm_power"].append(float(power.sum()))
            row["wall_s"].append(time.perf_counter() - t0)
            if row["diverged_at"] is None and umax > BAND:
                row["diverged_at"] = k + 1
            if (k + 1) % 5 == 0 or k == 0:
                print(f"    step {k + 1:4d}/{case.steps}  u_max={umax:.4f}  "
                      f"div={row['div_rms'][-1]:.4e}  P={row['farm_power'][-1]:.4f}"
                      f"  {time.perf_counter() - t0:6.0f}s", flush=True)
                write(f"confirm_{args.case}.json", row)
            if umax > 10.0 * BAND:
                row["stopped_at"] = k + 1
                break
    fin = row["div_rms"]
    q = len(fin) // 4
    if q >= 2:
        third = float(np.mean(fin[2 * q:3 * q]))
        fourth = float(np.mean(fin[3 * q:]))
        row["div_tail_trend"] = fourth / third if third > 0 else None
        row["div_flat_or_falling"] = bool(row["div_tail_trend"] <= 1.0)
    row["u_max_overall"] = max(row["u_max"]) if row["u_max"] else None
    row["stable"] = bool(row["diverged_at"] is None
                         and row["finite_to"] == case.steps
                         and "stopped_at" not in row)
    row["wall_total_s"] = time.perf_counter() - t0
    pt = row["farm_power"]
    if len(pt) >= 10:
        row["power_settling_last5"] = float(
            abs(np.mean(pt[-5:]) - np.mean(pt[-10:-5])) / max(abs(np.mean(pt[-5:])), 1e-12))
    print(f"  -> stable={row['stable']}  finite_to={row['finite_to']}/{case.steps}"
          f"  u_max={row['u_max_overall']}  div trend={row.get('div_tail_trend')}"
          f"  ({row['wall_total_s']:.0f}s)", flush=True)
    write(f"confirm_{args.case}.json", row)
    return row


# ---------------------------------------------------------------------------
# 3. grad -- the thing this whole PoC exists for
# ---------------------------------------------------------------------------


def stage_grad(args) -> dict:
    banner(f"3. gradient optimisation -- {args.case}, Adam through the composed "
           f"rollout")
    case, ro = build(args.case, args.steps, args.threads, args.device)
    th0 = wd.initial_design(case)
    name = f"grad_{args.case}.json"
    meta = {"case": case.as_dict(), "threads": args.threads,
            "opt_steps": args.opt_steps, "lr_pos": args.lr_pos,
            "lr_yaw": args.lr_yaw, "theta0": wd.design_to_dict(th0),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def on_step(it, tr):
        meta["trace"] = tr.as_dict()
        meta["theta_current"] = wd.design_to_dict(np.array(tr.theta[-1]))
        meta["min_spacing"] = wd.min_spacing(np.array(tr.theta[-1]))
        write(name, meta)
        print(f"    step {it + 1:3d}/{args.opt_steps}  J={tr.j[-1]:.6f}  "
              f"best={tr.best_j[-1]:.6f}  |g|={tr.grad_norm[-1]:.4f}  "
              f"settle={tr.extra['settling'][-1]:.2e}  "
              f"{tr.wall_s[-1]:7.0f}s", flush=True)

    j0, res0 = wd.value_only(th0, ro, keep_fields=[-1])
    meta["J0"] = j0
    meta["farm_power_0"] = res0.farm_power
    meta["settling_0"] = res0.settling()
    meta["power_trace_0"] = res0.power_trace.sum(axis=1).tolist()
    np.savez_compressed(os.path.join(OUT, f"field0_{args.case}.npz"),
                        u=res0.fields[case.steps][0], v=res0.fields[case.steps][1],
                        theta=th0)
    print(f"  start: J={j0:.6f}  farm power={res0.farm_power:.6f}  "
          f"settling={res0.settling():.3e}  ({res0.wall_s:.0f}s)", flush=True)
    write(name, meta)

    tr = wd.adam_optimise(th0, ro, steps=args.opt_steps, lr_pos=args.lr_pos,
                          lr_yaw=args.lr_yaw, on_step=on_step)
    j_best, th_best = tr.best()
    j_end, res_end = wd.value_only(th_best, ro, keep_fields=[-1])
    np.savez_compressed(os.path.join(OUT, f"field1_{args.case}.npz"),
                        u=res_end.fields[case.steps][0],
                        v=res_end.fields[case.steps][1], theta=th_best)
    meta.update({
        "trace": tr.as_dict(),
        "theta_best": wd.design_to_dict(th_best),
        "J_best": j_best, "J_best_recheck": j_end,
        "farm_power_best": res_end.farm_power,
        "settling_best": res_end.settling(),
        "power_trace_best": res_end.power_trace.sum(axis=1).tolist(),
        "gain_percent": 100.0 * (res_end.farm_power - res0.farm_power)
                        / res0.farm_power,
        "min_spacing_best": wd.min_spacing(th_best),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"  -> J {j0:.6f} -> {j_best:.6f};  farm power {res0.farm_power:.6f} "
          f"-> {res_end.farm_power:.6f}  ({meta['gain_percent']:+.2f}% as measured "
          f"by the composed model)", flush=True)
    write(name, meta)
    return meta


# ---------------------------------------------------------------------------
# 4. cma -- the derivative-free baseline, same objective, same start
# ---------------------------------------------------------------------------


def stage_cma(args) -> dict:
    banner(f"4. derivative-free baseline -- {args.case}, CMA-ES, "
           f"{args.workers} workers x {args.threads} threads")
    case = wd.case_for(args.case, args.steps)
    th0 = wd.initial_design(case)
    name = f"cma_{args.case}.json"
    target = args.target
    meta = {"case": case.as_dict(), "workers": args.workers,
            "threads": args.threads, "budget": args.budget,
            "wall_budget_s": args.wall_budget, "target_J": target,
            "theta0": wd.design_to_dict(th0),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write(name, meta)
    pool = Pool(args.case, args.steps, args.workers, args.threads, args.device,
                expert=EXPERT)
    last = [time.time()]

    def on_eval(n, j, best):
        if target is not None and best >= target and "evals_to_target" not in meta:
            meta["evals_to_target"] = n
        if n % 5 == 0 or time.time() - last[0] > 120:
            last[0] = time.time()
            print(f"    eval {n:5d}/{args.budget}  J={j:.6f}  best={best:.6f}",
                  flush=True)

    try:
        tr = wd.cma_es(th0, None, budget=args.budget, sigma0=args.sigma0,
                       seed=args.seed, on_eval=on_eval,
                       wall_budget_s=args.wall_budget, case=case,
                       evaluate=pool.evaluate)
    finally:
        pool.close()
    j_best, th_best = tr.best()
    meta.update({"trace": tr.as_dict(), "J_best": j_best,
                 "theta_best": wd.design_to_dict(th_best),
                 "n_eval": tr.evals[-1] if tr.evals else 0,
                 "wall_total_s": tr.wall_s[-1] if tr.wall_s else 0.0,
                 "min_spacing_best": wd.min_spacing(th_best),
                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if target is not None:
        meta["evals_to_target"] = wd.evals_to_tolerance(tr, target)
    write(name, meta)
    print(f"  -> {meta['n_eval']} evaluations, best J = {j_best:.6f}, "
          f"evals to target = {meta.get('evals_to_target')}", flush=True)
    return meta


# ---------------------------------------------------------------------------
# 5. fd -- OP-6's check, and the batch layout it is pinned against
# ---------------------------------------------------------------------------


def stage_fd(args) -> dict:
    banner(f"5. OP-6 -- the adjoint against central finite differences, {args.case}")
    case, ro = build(args.case, args.steps, args.threads, args.device)
    th = wd.initial_design(case)
    if args.theta_from:
        with open(args.theta_from, encoding="utf-8") as fh:
            th = wd.dict_to_design(json.load(fh)["theta_best"])
        print(f"  at the optimised layout from {args.theta_from}", flush=True)
    out = {"case": case.as_dict(), "at": wd.design_to_dict(th), "checks": {}}
    # determinism first: the claim "the batch layout is pinned" is only worth
    # anything if the same theta gives the same number twice, bit for bit.
    ja, _ = wd.value_only(th, ro)
    jb, _ = wd.value_only(th, ro)
    out["determinism"] = {"J_first": ja, "J_second": jb,
                          "bitwise_identical": bool(ja == jb),
                          "abs_diff": abs(ja - jb)}
    print(f"  determinism: {ja!r} vs {jb!r} -> "
          f"{'BITWISE IDENTICAL' if ja == jb else 'DIFFERS'}", flush=True)
    write(f"fd_{args.case}.json", out)
    for h in args.fd_h:
        chk = wd.fd_check(th, ro, n_probe=args.fd_probes, h=h)
        out["checks"][f"{h:g}"] = chk
        print(f"  h={h:g}: max rel err {chk['max_rel_err']:.3e}, median "
              f"{chk['median_rel_err']:.3e}, cosine {chk['cosine']:.10f}",
              flush=True)
        for r in chk["components"]:
            print(f"      T{r['turbine']:02d} {r['kind']:3s} adjoint="
                  f"{r['adjoint']: .6e}  fd={r['fd']: .6e}  rel={r['rel_err']:.2e}",
                  flush=True)
        write(f"fd_{args.case}.json", out)
    return out


# ---------------------------------------------------------------------------
# 5b. ablation -- how much of the gradient is the COUPLING?
# ---------------------------------------------------------------------------


def stage_ablation(args) -> dict:
    """Full adjoint against the frozen-rollout gradient, at the same point.

    The PoC's claim is not "we have a gradient"; a wake surrogate has one of
    those.  It is that the gradient goes THROUGH the composed rollout, so this
    stage measures the difference the rollout makes: the same objective at the
    same design vector, differentiated once with the march live and once with
    the march frozen and only the disks' own closure differentiated.

    If the two agreed, every fluid-disk seam in the differentiated path would be
    decorative and a static wake model would give the same search direction.
    """
    banner(f"5b. ablation -- what the coupled path contributes, {args.case}")
    case, ro = build(args.case, args.steps, args.threads, args.device)
    th = wd.initial_design(case)
    if args.theta_from and os.path.isfile(args.theta_from):
        with open(args.theta_from, encoding="utf-8") as fh:
            th = wd.dict_to_design(json.load(fh)["theta_best"])
        print(f"  at the optimised layout from {args.theta_from}", flush=True)
    t0 = time.perf_counter()
    j_full, g_full, _ = wd.value_and_grad(th, ro)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    j_loc, g_loc, _ = wd.value_and_grad_local(th, ro)
    t_loc = time.perf_counter() - t0
    nf, nl = float(np.linalg.norm(g_full)), float(np.linalg.norm(g_loc))
    cos = float(g_full @ g_loc / (nf * nl)) if nf > 0 and nl > 0 else None
    out = {"case": case.as_dict(), "at": wd.design_to_dict(th),
           "J_full": j_full, "J_local": j_loc,
           "wall_full_s": t_full, "wall_local_s": t_loc,
           "grad_norm_full": nf, "grad_norm_local": nl,
           "cosine_full_local": cos,
           "coupled_fraction_of_norm": float(
               np.linalg.norm(g_full - g_loc) / nf) if nf > 0 else None,
           "by_kind": {}}
    for k, name in enumerate(("x", "y", "yaw")):
        a, b = g_full[k::3], g_loc[k::3]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        out["by_kind"][name] = {
            "norm_full": na, "norm_local": nb,
            "cosine": float(a @ b / (na * nb)) if na > 0 and nb > 0 else None,
            "coupled_fraction": float(np.linalg.norm(a - b) / na) if na > 0 else None,
            "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
        }
    print(f"  |g| full {nf:.5f}  frozen-rollout {nl:.5f}  cosine {cos:.6f}",
          flush=True)
    print(f"  the coupled path is {out['coupled_fraction_of_norm'] * 100:.1f}% "
          f"of the full gradient's norm", flush=True)
    for name, r in out["by_kind"].items():
        print(f"    {name:3s}: |full|={r['norm_full']:.5f} "
              f"|frozen|={r['norm_local']:.5f} cos={r['cosine']:.4f} "
              f"coupled={r['coupled_fraction'] * 100:5.1f}%  "
              f"signs agree {r['sign_agreement'] * 100:.0f}%", flush=True)
    write(f"ablation_{args.case}.json", out)
    return out


# ---------------------------------------------------------------------------
# 6. sensitivity -- how much of the answer is the march length?
# ---------------------------------------------------------------------------


def stage_sensitivity(args) -> dict:
    banner(f"6. march-length sensitivity -- {args.case}")
    case = wd.case_for(args.case, None)
    th0 = wd.initial_design(case)
    thb = None
    gp = os.path.join(OUT, f"grad_{args.case}.json")
    if os.path.isfile(gp):
        with open(gp, encoding="utf-8") as fh:
            g = json.load(fh)
        if "theta_best" in g:
            thb = wd.dict_to_design(g["theta_best"])
    out = {"case": case.as_dict(), "lengths": args.lengths, "rows": []}
    for n in args.lengths:
        _c, ro = build(args.case, int(n), args.threads, args.device)
        j0, r0 = wd.value_only(th0, ro)
        row = {"steps": int(n), "J_start": j0, "power_start": r0.farm_power,
               "settling_start": r0.settling(), "wall_s": r0.wall_s}
        if thb is not None:
            j1, r1 = wd.value_only(thb, ro)
            row.update({"J_opt": j1, "power_opt": r1.farm_power,
                        "settling_opt": r1.settling(),
                        "gain_percent": 100.0 * (r1.farm_power - r0.farm_power)
                                        / r0.farm_power})
        out["rows"].append(row)
        print(f"  {n:3d} steps: start P={row['power_start']:.6f} "
              f"(settle {row['settling_start']:.2e})"
              + (f"  opt P={row['power_opt']:.6f}  gain "
                 f"{row['gain_percent']:+.2f}%" if thb is not None else ""),
              flush=True)
        write(f"sensitivity_{args.case}.json", out)
    return out




# ---------------------------------------------------------------------------
# 6b. verify -- the classical verification panel, as a stage rather than a demo
# ---------------------------------------------------------------------------


def stage_verify(args) -> dict:
    """The composed column against the UNDIVIDED classical solver, same layout.

    This is `atlas-proof-of-concept-1` section 10's head-to-head, lifted out of
    the demo so it can be run headless on either expert and quoted from an
    artefact rather than from a screenshot.  `composition-error-theory`'s
    standing position is the reason it exists at all: **the composed model is a
    search instrument, not a verifier**, so the deployment story
    (`prior-art-and-novelty-atlas-0.1` section 5) is search wide and cheap with
    the composed model and verify the shortlist with the classical stack.  This
    stage IS that verification step, and it is what converts a power figure
    "as measured by the composed model" into a number with a referent.

    The referent is `scaling_ladder.reference_monolith` -- `RectangularNS` on the
    whole domain at the same cell size, same viscosity, same transmission, no
    cut.  Against the classical composed column it isolates the cost of the cut,
    holding the expert fixed.  Against the Poseidon column it measures **the
    expert and the cut together**, and no arrangement of this stage can separate
    them, because there is no monolithic Poseidon: the checkpoint is fixed at
    128x128 and cannot be asked to solve the whole domain at any resolution
    (`poseidon.py`'s module docstring).  That asymmetry is reported rather than
    papered over.

    **The two solvers alternate, one macro-step each**, so every timed region has
    the machine to itself; running them concurrently would make a better
    animation and a worthless measurement.  Because both sides sit at step *i* at
    the same moment, the fields are directly comparable and the worst single-cell
    velocity difference is reported beside the difference in farm power.
    """
    import torch
    from atlas.cases import scaling_ladder as sl

    banner(f"6b. classical verification panel -- {args.case}, {EXPERT} composed "
           f"column vs the undivided monolith")
    case, ro = build(args.case, args.steps, args.threads, args.device)
    th_np = wd.initial_design(case)
    label = "grid"
    if args.theta_from and os.path.isfile(args.theta_from):
        with open(args.theta_from, encoding="utf-8") as fh:
            th_np = wd.dict_to_design(json.load(fh)["theta_best"])
        label = os.path.basename(args.theta_from)
        print(f"  at the optimised layout from {args.theta_from}", flush=True)
    th = torch.as_tensor(th_np, dtype=wd.TORCH_DTYPE, device=args.device)
    th_cpu = th if args.device == "cpu" else torch.as_tensor(th_np,
                                                             dtype=wd.TORCH_DTYPE)
    ro_cpu = ro if args.device == "cpu" else build(args.case, args.steps,
                                                   args.threads, "cpu")[1]

    n = case.steps
    u = torch.full(case.shape, wa.U_INF, dtype=wd.TORCH_DTYPE, device=args.device)
    v = torch.zeros_like(u)
    mono = sl.reference_monolith(case.tiling.nx, case.tiling.ny, wa.NU_REF)
    uu = np.full(case.shape, wa.U_INF)
    vv = np.zeros(case.shape)
    band = ro_cpu._band.cpu().numpy()

    row = {"case": case.as_dict(), "expert": EXPERT, "at": wd.design_to_dict(th_np),
           "layout": label, "steps": n, "composed_power": [], "classical_power": [],
           "composed_ms": [], "classical_ms": [], "linf": [],
           "composed_u_max": [], "classical_u_max": []}
    composed_s = classical_s = 0.0
    with torch.no_grad():
        for i in range(n):
            t0 = time.perf_counter()
            u, v, power, _ = ro.macro_step(u, v, th)
            if args.device != "cpu":
                torch.cuda.synchronize()
            dt_c = time.perf_counter() - t0
            composed_s += dt_c

            t0 = time.perf_counter()
            tu = torch.as_tensor(uu, dtype=wd.TORCH_DTYPE)
            tv = torch.as_tensor(vv, dtype=wd.TORCH_DTYPE)
            fx, fy, _un, _T, p = ro_cpu.disks.forcing(tu, tv, th_cpu)
            u1, v1 = mono.step_batch(uu[None], vv[None], wa.MACRO_DT, bc0=None,
                                     force=(fx.numpy()[None], fy.numpy()[None]))
            uu, vv = u1[0], v1[0]
            uu = np.where(band, wa.U_INF, uu)
            vv = np.where(band, 0.0, vv)
            dt_k = time.perf_counter() - t0
            classical_s += dt_k

            un = u.detach().cpu().numpy()
            row["composed_power"].append(float(power.sum()))
            row["classical_power"].append(float(p.sum()))
            row["composed_ms"].append(dt_c * 1000.0)
            row["classical_ms"].append(dt_k * 1000.0)
            row["linf"].append(float(np.abs(un - uu).max()))
            row["composed_u_max"].append(float(np.abs(un).max()))
            row["classical_u_max"].append(float(np.abs(uu).max()))
            if not np.all(np.isfinite(uu)):
                row["classical_not_finite_at"] = i + 1
                break
            if (i + 1) % 5 == 0 or i == 0:
                print(f"    step {i+1:3d}/{n}  composed P={row['composed_power'][-1]:.4f}"
                      f"  classical P={row['classical_power'][-1]:.4f}"
                      f"  Linf={row['linf'][-1]:.4f}"
                      f"  ({dt_c*1000:.0f} / {dt_k*1000:.0f} ms)", flush=True)
                write(f"verify_{args.case}.json", row)

    avg = min(case.avg_window, len(row["composed_power"]))
    comp = float(np.mean(row["composed_power"][-avg:]))
    clas = float(np.mean(row["classical_power"][-avg:]))
    # the first step is held out of both medians: it pays for FFT plans and for
    # every allocation neither solver has made yet, on both sides.
    cm = float(np.median(row["composed_ms"][1:])) if len(row["composed_ms"]) > 1 else None
    km = float(np.median(row["classical_ms"][1:])) if len(row["classical_ms"]) > 1 else None
    row.update({
        "composed_power_final": comp, "classical_power_final": clas,
        "delta": comp - clas,
        "delta_pct": 100.0 * (comp - clas) / clas if abs(clas) > 1e-12 else None,
        "composed_ms_step": cm, "classical_ms_step": km,
        "ms_ratio": (cm / km) if km else None,
        "composed_wall_s": composed_s, "classical_wall_s": classical_s,
        "linf_final": row["linf"][-1] if row["linf"] else None,
        "note": ("Both marched from the freestream for the same number of "
                 "macro-steps, alternating so neither competed with the other "
                 "for cores. The classical side is "
                 "scaling_ladder.reference_monolith: the same discretization "
                 "with no cut. Against the classical composed column the "
                 "difference is the cost of the cut alone; against the Poseidon "
                 "column it is the expert AND the cut together, and there is no "
                 "monolithic Poseidon that could separate them."),
    })
    print(f"  -> composed {comp:.4f} vs classical {clas:.4f} "
          f"({row['delta_pct']:+.1f}%), worst cell {row['linf_final']:.4f}, "
          f"{cm:.0f} vs {km:.0f} ms/step", flush=True)
    write(f"verify_{args.case}.json", row)
    return row




# ---------------------------------------------------------------------------
# 6c. crosseval -- each column's optimum, scored by the other and by the monolith
# ---------------------------------------------------------------------------


def _march_monolith(case, theta_np, ro_cpu, steps: int | None = None) -> dict:
    """`scaling_ladder.reference_monolith` on one layout, from the freestream.

    The undivided classical solver: same discretization, same cell, same
    viscosity, no cut.  It is the referent the deployment story
    (`prior-art-and-novelty-atlas-0.1` section 5) names -- *search wide and cheap
    with the composed model, verify the shortlist with the classical stack* --
    and this is the verifier half of that sentence.
    """
    import torch
    from atlas.cases import scaling_ladder as sl

    n = steps or case.steps
    th = torch.as_tensor(theta_np, dtype=wd.TORCH_DTYPE)
    mono = sl.reference_monolith(case.tiling.nx, case.tiling.ny, wa.NU_REF)
    uu = np.full(case.shape, wa.U_INF)
    vv = np.zeros(case.shape)
    band = ro_cpu._band.cpu().numpy()
    trace = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            fx, fy, _un, _T, p = ro_cpu.disks.forcing(
                torch.as_tensor(uu, dtype=wd.TORCH_DTYPE),
                torch.as_tensor(vv, dtype=wd.TORCH_DTYPE), th)
            u1, v1 = mono.step_batch(uu[None], vv[None], wa.MACRO_DT, bc0=None,
                                     force=(fx.numpy()[None], fy.numpy()[None]))
            uu, vv = u1[0], v1[0]
            uu = np.where(band, wa.U_INF, uu)
            vv = np.where(band, 0.0, vv)
            trace.append(float(p.sum()))
            if not np.all(np.isfinite(uu)):
                return {"farm_power": None, "not_finite_at": len(trace),
                        "trace": trace}
    avg = min(case.avg_window, len(trace))
    return {"farm_power": float(np.mean(trace[-avg:])), "trace": trace,
            "wall_s": time.perf_counter() - t0, "steps": n}


def stage_crosseval(args) -> dict:
    """**The deployment story, measured.**

    Three layouts -- the unoptimised grid, the layout the CLASSICAL column's
    optimiser converged to, and the layout the POSEIDON column's optimiser
    converged to -- each scored three ways: by the Poseidon composed column, by
    the classical composed column, and by the undivided classical monolith.

    This is the only measurement here that can answer the question the swap
    actually raises.  OP-3 says the checkpoint destroys roughly three quarters
    of a wake before the second turbine, so the two columns are optimising
    genuinely different objectives and will not agree on a layout.  Whether that
    matters depends on a question no norm on the gradient can answer:
    **does the layout the cheap column found still beat the starting grid when
    the expensive one scores it?**  If it does, the checkpoint is a usable
    pre-screen and `composition-error-theory`'s "search instrument, not a
    verifier" is a description of a working pipeline.  If it does not, the
    checkpoint's over-dissipation has moved the optimum somewhere that only
    exists inside the checkpoint, and the pre-screen is worthless at this
    fidelity.  Either answer is a result; neither is available from one column.

    The classical column's layout is read from `out/w111/grad_<case>.json`, the
    2026-09-01 artefact, so the comparison is against the run
    [[poc1-results-differentiable-design]] reports rather than a re-run of it.
    """
    banner(f"6c. crosseval -- three layouts, three scorers, {args.case}")
    case, ro_pos = build(args.case, args.steps, args.threads, args.device,
                         expert="poseidon")
    _c, ro_ref = build(args.case, args.steps, args.threads, args.device,
                       expert="reference_exposed")
    ro_cpu = ro_ref if args.device == "cpu" else build(
        args.case, args.steps, args.threads, "cpu", expert="reference_exposed")[1]

    layouts = {"grid": wd.initial_design(case)}
    for tag, path in (("classical_optimum",
                       os.path.join(ROOT, "out", "w111", f"grad_{args.case}.json")),
                      ("poseidon_optimum",
                       os.path.join(ROOT, "out", "w118", f"grad_{args.case}.json"))):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            if "theta_best" in d:
                layouts[tag] = wd.dict_to_design(d["theta_best"])
        else:
            print(f"  {tag}: {path} is not here, skipping that row", flush=True)

    out = {"case": case.as_dict(), "layouts": {}, "scorers":
           ["poseidon_composed", "classical_composed", "classical_monolith"]}
    for tag, th in layouts.items():
        row = {"theta": wd.design_to_dict(th), "min_spacing": wd.min_spacing(th)}
        jp, rp = wd.value_only(th, ro_pos)
        row["poseidon_composed"] = rp.farm_power
        jc, rc = wd.value_only(th, ro_ref)
        row["classical_composed"] = rc.farm_power
        m = _march_monolith(case, th, ro_cpu)
        row["classical_monolith"] = m["farm_power"]
        row["monolith_wall_s"] = m.get("wall_s")
        out["layouts"][tag] = row
        print(f"  {tag:20s}  poseidon {row['poseidon_composed']:8.4f}   "
              f"classical-composed {row['classical_composed']:8.4f}   "
              f"monolith {row['classical_monolith'] if row['classical_monolith'] is None else format(row['classical_monolith'], '8.4f')}",
              flush=True)
        write(f"crosseval_{args.case}.json", out)

    base = out["layouts"].get("grid", {})
    for tag, row in out["layouts"].items():
        if tag == "grid":
            continue
        row["gain_vs_grid"] = {}
        for sc in out["scorers"]:
            b, a = base.get(sc), row.get(sc)
            row["gain_vs_grid"][sc] = (100.0 * (a - b) / b
                                       if (a is not None and b) else None)
        print(f"  {tag:20s}  gain vs grid: "
              + "   ".join(f"{sc.split('_')[0]}/{sc.split('_')[1]} "
                           f"{row['gain_vs_grid'][sc]:+.1f}%"
                           for sc in out["scorers"]
                           if row["gain_vs_grid"][sc] is not None), flush=True)
    out["note"] = ("The row that matters is `poseidon_optimum` scored by "
                   "`classical_monolith`: it is what the deployment story "
                   "claims -- a layout found cheaply by the composed "
                   "checkpoint column and then verified by the classical "
                   "stack. Every column here is a farm power in the same "
                   "units, marched from the same freestream for the same "
                   "number of macro-steps with the same disks.")
    write(f"crosseval_{args.case}.json", out)
    return out


# ---------------------------------------------------------------------------
# 7. figures
# ---------------------------------------------------------------------------


def stage_figures(args) -> dict:
    banner("7. figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    for cname in args.figure_cases:
        f0 = os.path.join(OUT, f"field0_{cname}.npz")
        f1 = os.path.join(OUT, f"field1_{cname}.npz")
        gp = os.path.join(OUT, f"grad_{cname}.json")
        if not (os.path.isfile(f0) and os.path.isfile(f1)):
            print(f"  {cname}: no fields yet, skipping", flush=True)
            continue
        a, b = np.load(f0), np.load(f1)
        case = wd.case_for(cname, None)
        with open(gp, encoding="utf-8") as fh:
            g = json.load(fh)
        lx, ly = case.extent
        # A shared ramp, clipped to the band the WAKES live in.  The unclipped
        # range is set by the thin speed-up strips against the lateral walls,
        # which are the least interesting part of the picture and would take
        # half the colour map with them.
        both = np.concatenate([a["u"].ravel(), b["u"].ravel()])
        vmin = float(np.percentile(both, 0.2))
        vmax = float(np.percentile(both, 99.8))

        def draw(ax, u, theta, title):
            im = ax.imshow(u, origin="lower", extent=(0, lx, 0, ly),
                           vmin=vmin, vmax=vmax, cmap="turbo",
                           interpolation="nearest", aspect="equal")
            t = np.asarray(theta).reshape(-1, 3)
            for x, y, gam in t:
                dx_, dy_ = -0.5 * math.sin(gam), 0.5 * math.cos(gam)
                ax.plot([x - dx_, x + dx_], [y - dy_, y + dy_], "k-", lw=2.2,
                        solid_capstyle="butt")
                ax.plot([x, x + 0.40 * math.cos(gam)],
                        [y, y + 0.40 * math.sin(gam)], "w-", lw=1.0)
            ax.set_title(title, fontsize=10.5, pad=6)
            ax.set_ylabel("y / D")
            return im

        h = 2 * (5.6 * ly / lx) + 1.9
        fig, axes = plt.subplots(2, 1, figsize=(9.0, h), constrained_layout=True)
        im = draw(axes[0], a["u"], a["theta"],
                  f"before: regular {case.k_col}x{case.k_row} grid, no yaw"
                  f"   |   farm power {g['farm_power_0']:.3f}")
        draw(axes[1], b["u"], b["theta"],
             f"after: {len(g['trace']['j'])} gradient steps"
             f"   |   farm power {g['farm_power_best']:.3f}"
             f"   ({g['gain_percent']:+.0f}%)")
        axes[1].set_xlabel("x / D")
        cb = fig.colorbar(im, ax=axes, fraction=0.040, pad=0.015)
        cb.set_label("streamwise velocity  u / U_inf   (shared ramp)")
        fig.suptitle(f"PoC 1a - {case.k} turbines over {case.n_windows} coupled "
                     f"128-cell windows, {case.steps} composed macro-steps"
                     + "\n"
                     + "gradient taken through every fluid-disk seam; "
                     + "power gain as measured by the composed model",
                     fontsize=11)
        p = os.path.join(OUT, f"fields_{cname}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)
        print(f"  wrote {p}", flush=True)

        # -- the optimiser traces -------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2),
                                 constrained_layout=True)
        tr = g["trace"]
        axes[0].plot(np.arange(1, len(tr["j"]) + 1), tr["j"], "o-", ms=3,
                     label="gradient (Adam through the composed rollout)")
        cp = os.path.join(OUT, f"cma_{cname}.json")
        if os.path.isfile(cp):
            with open(cp, encoding="utf-8") as fh:
                c = json.load(fh)
            cb = c["trace"].get("best_j")
            if not isinstance(cb, list):
                cb = list(np.maximum.accumulate(c["trace"]["j"]))
            gb = tr.get("best_j")
            if not isinstance(gb, list):
                gb = list(np.maximum.accumulate(tr["j"]))
            axes[1].plot(c["trace"]["evals"], cb, "-", color="tab:orange",
                         label=f"CMA-ES (popsize {c['trace']['popsize']})")
            axes[1].plot(tr["evals"], gb, "o-", ms=3, color="tab:blue",
                         label="gradient through the composed graph")
            axes[1].axhline(g["J_best"] - 0.02 * (g["J_best"] - g["J0"]),
                            color="k", ls="--", lw=0.8,
                            label="tolerance (2% of the gradient's gain)")
            axes[1].set_xscale("log")
            axes[1].set_xlabel("composed rollouts (forward evaluations)")
            axes[1].set_ylabel("best J so far")
            axes[1].legend(fontsize=8)
            axes[1].set_title("evaluations to the same objective")
        axes[0].set_xlabel("optimiser step")
        axes[0].set_ylabel("J = farm power - spacing penalty")
        axes[0].set_title("farm power over optimiser steps")
        axes[0].legend(fontsize=8)
        if isinstance(tr.get("theta"), list) and tr["theta"] and                 isinstance(tr["theta"][0], list):
            th = np.array(tr["theta"])
            yaw = np.degrees(th[:, 2::3])
            for k in range(yaw.shape[1]):
                axes[2].plot(np.arange(1, yaw.shape[0] + 1), yaw[:, k], lw=1.0)
            axes[2].set_xlabel("optimiser step")
            axes[2].set_title("yaw angles converging")
        else:
            # this run predates `OptTrace.as_dict` carrying the trajectory, so
            # the panel shows where the yaws ENDED rather than how they got
            # there. Said here rather than left for a reader to infer.
            yb = np.degrees(np.array([r["yaw_deg"] for r in g["theta_best"]])
                            * np.pi / 180.0)
            axes[2].bar(np.arange(len(yb)), yb, color="tab:blue")
            axes[2].set_xlabel("turbine")
            axes[2].set_title("yaw at the optimum (history not serialised)")
        axes[2].set_ylabel("yaw / degrees")
        axes[2].axhline(0.0, color="k", lw=0.5)
        p = os.path.join(OUT, f"traces_{cname}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(p)
        print(f"  wrote {p}", flush=True)
    return {"figures": made}


# ---------------------------------------------------------------------------
# 8. merge
# ---------------------------------------------------------------------------


def stage_merge(args) -> dict:
    banner("8. merge")
    out = {"case": "PoC 1a -- differentiable wind-farm design",
           "expert": EXPERT,
           "date": time.strftime("%Y-%m-%d"), "machine": machine(), "parts": {}}
    merged = MERGED[EXPERT]
    out["fluid_expert"] = EXPERT
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".json") or fn in MERGED.values():
            continue
        with open(os.path.join(OUT, fn), encoding="utf-8") as fh:
            out["parts"][fn[:-5]] = json.load(fh)
    out["headline"] = headline(out["parts"])
    p = write(merged, out)
    print(json.dumps(_f(out["headline"]), indent=1))
    print(f"  merged into {p}", flush=True)
    return out


def headline(parts: dict) -> dict:
    """The one number, assembled from whatever stages actually ran."""
    rows = {}
    for cname in ("K12", "K25"):
        g = parts.get(f"grad_{cname}")
        c = parts.get(f"cma_{cname}")
        t = (parts.get("timing") or {}).get("cases", {}).get(cname, {})
        if not g or "trace" not in g:
            continue
        j0 = g.get("J0")
        jg = g.get("J_best")
        row = {"K": g["case"]["K"], "n_design": g["case"]["n_design"],
               "n_windows": g["case"]["n_windows"],
               "rollout_macro_steps": g["case"]["steps"],
               "J_start": j0, "J_gradient": jg,
               "farm_power_start": g.get("farm_power_0"),
               "farm_power_gradient": g.get("farm_power_best"),
               "gain_percent_composed_model": g.get("gain_percent"),
               "gradient_rollouts": g["trace"]["evals"][-1],
               "gradient_wall_s": g["trace"]["wall_s"][-1],
               "s_per_forward_rollout": t.get("s_per_rollout_forward"),
               "s_per_gradient_rollout": t.get("s_per_rollout_gradient"),
               "gradient_cost_in_forward_equivalents":
                   t.get("gradient_over_forward")}
        if j0 is not None and jg is not None:
            eps = 0.02 * (jg - j0)
            target = jg - eps
            row["tolerance_J"] = target
            row["tolerance_rule"] = ("within 2% of the gradient method's own "
                                     "improvement over the start")
            bj = g["trace"]["best_j"]
            if not isinstance(bj, list):
                bj = list(np.maximum.accumulate(g["trace"]["j"]))
            ev = g["trace"]["evals"]
            row["gradient_rollouts_to_tolerance"] = next(
                (e for e, b in zip(ev, bj) if b >= target), None)
            if c and "trace" in c:
                ce = c["trace"]["evals"]
                cb = c["trace"].get("best_j")
                if not isinstance(cb, list):
                    # an artefact written before the key collision was fixed:
                    # the running best is the cumulative max of the raw trace
                    cb = list(np.maximum.accumulate(c["trace"]["j"]))
                hit = next((e for e, b in zip(ce, cb) if b >= target), None)
                row["cma_rollouts_to_tolerance"] = hit
                row["cma_rollouts_spent"] = ce[-1] if ce else 0
                row["cma_best_J"] = c.get("J_best")
                row["cma_reached_tolerance"] = hit is not None
                gr = row["gradient_rollouts_to_tolerance"]
                if hit is not None and gr:
                    row["ratio_rollouts"] = hit / gr
                elif gr:
                    row["ratio_rollouts_lower_bound"] = (ce[-1] / gr) if ce else None
        n = row["n_design"]
        row["coordinate_fd_rollouts_per_gradient"] = 2 * n
        gr = row.get("gradient_rollouts_to_tolerance")
        if gr:
            row["coordinate_fd_rollouts_to_tolerance_if_same_path"] = (2 * n + 1) * gr
            row["ratio_rollouts_vs_coordinate_fd"] = (2 * n + 1)
        rows[cname] = row
    return rows


# ---------------------------------------------------------------------------


STAGES = {"timing": stage_timing, "confirm": stage_confirm, "grad": stage_grad,
          "cma": stage_cma, "fd": stage_fd, "ablation": stage_ablation,
          "sensitivity": stage_sensitivity, "verify": stage_verify,
          "crosseval": stage_crosseval,
          "figures": stage_figures, "merge": stage_merge}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--case", default="K12", choices=("K12", "K25"))
    ap.add_argument("--steps", type=int, default=None,
                    help="macro-steps per rollout (default: the case's own)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; the composed column runs wherever it is put")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--lr-pos", type=float, default=0.12)
    ap.add_argument("--lr-yaw", type=float, default=0.035)
    ap.add_argument("--budget", type=int, default=900)
    ap.add_argument("--wall-budget", type=float, default=None,
                    help="seconds; the derivative-free run stops at it")
    ap.add_argument("--sigma0", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=None,
                    help="J the baseline is run to; the gradient optimum minus "
                         "the tolerance")
    ap.add_argument("--fd-h", type=float, nargs="+",
                    default=[1e-2, 1e-3, 1e-4, 1e-5])
    ap.add_argument("--fd-probes", type=int, default=9)
    ap.add_argument("--theta-from", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=[40, 50, 60])
    ap.add_argument("--figure-cases", nargs="+", default=["K12", "K25"])
    ap.add_argument("--expert", default="reference_exposed",
                    choices=("reference_exposed", "poseidon"),
                    help="which fluid expert every window runs. "
                         "`reference_exposed` is `reference.WindowNS` with its "
                         "elliptic part handed to the composition layer, i.e. "
                         "the 2026-09-01 column; `poseidon` is the frozen "
                         "20.8M-parameter checkpoint, i.e. the half of the "
                         "novelty claim that column does not exercise. "
                         "Artefacts go to out/w111 and out/w118 respectively.")
    args = ap.parse_args(argv)
    global EXPERT, OUT
    EXPERT = args.expert
    OUT = OUT_FOR[EXPERT]
    os.makedirs(OUT, exist_ok=True)
    print(f"fluid expert: {EXPERT}   artefacts: {OUT}", flush=True)
    t0 = time.time()
    try:
        STAGES[args.stage](args)
    except Exception:
        import traceback
        write(f"FAILED_{args.stage}_{args.case}.json",
              {"stage": args.stage, "case": args.case,
               "traceback": traceback.format_exc(),
               "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
        raise
    print(f"\nstage {args.stage} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
