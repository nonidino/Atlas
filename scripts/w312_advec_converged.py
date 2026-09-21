r"""W312 -- the four ADVEC seams probed at their DECLARED clocks, in parallel.

Tier 79 labelled every ADVEC number in CS-21 section 13.3 as *"the operator the
probe could see in 8 sub-steps"* and paid for one anchor to show what the label
costs: on `e-f` at the same declared dim M, beta moved 42.8x and agent `e`'s
share of the seam went 6.4% to 36%.  Tier 83 priced the rest at 6.48 hours of
single-core time, after removing `g-f` by argument -- its band normal is
n_z = -0.000e0 exactly, so its declared flow is the transverse mass flux and no
cadence changes a normal.

**Why this is parallel, and why that is sound.**  `probe_block` is a sequential
loop, but the calls inside it are INDEPENDENT: the traces are `base` and
`base + step * direction_k`, all fixed before any response is computed, and each
`respond` restarts the agent from its own `_U0`.  So the framework is left as the
authority and only the arithmetic is moved:

  pass 1   run the real `seam_operator` with every `respond` replaced by a
           RECORDER that returns zeros and logs what it was asked for.  Costs
           nothing; the result is discarded.  Done TWICE at a cheap cadence as a
           control that the request set is deterministic -- if it were not, a
           record-then-fill cache would be unsound and this script would be
           wrong in a way no output would show.
  pass 2   compute every recorded trace across a process pool, checkpointing as
           they land.
  pass 3   run `seam_operator` again with the cache serving the responses.  The
           framework does its own assembly, diagnostics and bookkeeping, so
           every number is the one the compiler would have produced.

Measured on this machine (16 physical cores, hybrid P/E): aggregate throughput
peaks near 12 workers at about 5x, and 16 workers is SLOWER than 8.  The default
leaves headroom rather than chasing the peak.

    python scripts/w312_advec_converged.py --json out/w312_converged.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from atlas.cases import rocket_experts as RE
from atlas.probe import ProbeBudget

#: `g-f` is excluded on Tier 83's geometric argument, not on cost.
ADVEC_SEAMS = ("a-b", "e-b", "e-f", "d-g")


def _key(aid: str, port: str, trace) -> str:
    v = np.ascontiguousarray(np.asarray(trace, dtype=float))
    return "%s|%s|%s" % (aid, port, hashlib.sha1(v.tobytes()).hexdigest())


class _Respond:
    """Records what it is asked for, or serves what has been computed."""

    def __init__(self, aid, real, seen, cache, mode):
        self.aid, self.real = aid, real
        self.seen, self.cache, self.mode = seen, cache, mode

    def __call__(self, port_name, trace):
        k = _key(self.aid, port_name, trace)
        if self.mode == "record":
            self.seen[k] = (self.aid, port_name,
                            np.array(trace, dtype=float, copy=True))
            #: The response has the same length as the trace: `probe_block`
            #: forms ``R @ flux_V`` with ``R`` the prolongation adjoint, whose
            #: column count is the trace dimension.
            return np.zeros(np.size(trace), dtype=float)
        if k not in self.cache:
            raise KeyError("uncached trace for %s %s -- the request set was not "
                           "deterministic, which invalidates this run" % (self.aid, port_name))
        return self.cache[k]


def _wrapped_graph(dt_scale, seen, cache, mode, experts=None):
    experts = experts or RE.make_rocket_experts(dt_scale=dt_scale)
    for aid, ex in experts.items():
        real = getattr(ex, "_w312_real", None) or ex.respond
        ex._w312_real = real
        ex.respond = _Respond(aid, real, seen, cache, mode)
    graph, _ = RE.build_rocket_real(experts=experts, dt_scale=dt_scale)
    return graph, experts


def _sub(graph, seam_id):
    from atlas.graph import CaseGraph
    conn = [c for c in graph.connections if c.seam_id == seam_id][0]
    x, y = conn.a[0], conn.b[0]
    return CaseGraph(name="%s:%s" % (graph.name, seam_id),
                     agents=[g for g in graph.agents if g.agent_id in (x, y)],
                     connections=[conn], decomposition=graph.decomposition,
                     macro_dt=graph.macro_dt,
                     flux_matching=graph.flux_matching), x, y


# --- the pool ---------------------------------------------------------------

_POOL_AGENTS: dict = {}


def _pool_init(dt_scale):
    global _DT_SCALE
    _DT_SCALE = dt_scale


def _pool_call(job):
    aid, port, trace = job
    cfg = RE.rocket_config()
    ex = _POOL_AGENTS.get(aid)
    if ex is None:
        ex = RE.GasAgent(aid, dt=float(cfg.dt_model[aid]) * _DT_SCALE)
        _POOL_AGENTS[aid] = ex
    t0 = time.perf_counter()
    out = np.asarray(ex.respond(port, trace), dtype=float)
    return _key(aid, port, trace), out, time.perf_counter() - t0


# --- stages -----------------------------------------------------------------


def record(dt_scale, seams):
    """Every trace the framework will ask for, at no solver cost."""
    seen: dict = {}
    graph, _ = _wrapped_graph(dt_scale, seen, {}, "record")
    per = {}
    for sid in seams:
        before = set(seen)
        sub, x, y = _sub(graph, sid)
        RE.seam_operator(sub, budget=ProbeBudget(),
                         probe_state="w312 record dt_scale=%g" % dt_scale)
        per[sid] = len(set(seen) - before)
    return seen, per, graph


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    ap.add_argument("--seams", default=",".join(ADVEC_SEAMS))
    ap.add_argument("--dt-scale", type=float, default=1.0,
                    help="1.0 is the config's own declared clock")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--cheap-scale", type=float, default=1.0e-3,
                    help="the reduced cadence the A/B is against")
    a = ap.parse_args(argv)
    seams = [s.strip() for s in a.seams.split(",") if s.strip()]

    print("=" * 84)
    print("W312 -- ADVEC at the declared clock, %d seams, %d workers"
          % (len(seams), a.workers))
    print("=" * 84)
    t_start = time.perf_counter()
    out: dict = {"seams": seams, "dt_scale": a.dt_scale, "workers": a.workers}

    # -- the control: is the request set deterministic? ---------------------
    print()
    print("=== control: the recorded request set must be deterministic ===")
    s1, per1, _ = record(a.cheap_scale, seams)
    s2, per2, _ = record(a.cheap_scale, seams)
    same = set(s1) == set(s2)
    print("  two independent recordings at dt_scale=%g: %d and %d traces, "
          "identical sets: %s" % (a.cheap_scale, len(s1), len(s2), same))
    if not same:
        print("  REFUSING to continue: a record-then-fill cache is only sound if")
        print("  the framework asks for the same traces every time.")
        return 1
    out["determinism_control"] = dict(n1=len(s1), n2=len(s2), identical=bool(same),
                                      per_seam=per1)

    # -- the cheap arm, for the A/B -----------------------------------------
    print()
    print("=== the reduced cadence, for the comparison (dt_scale=%g) ===" % a.cheap_scale)
    cheap = {}
    t0 = time.perf_counter()
    experts_c = RE.make_rocket_experts(dt_scale=a.cheap_scale)
    graph_c, _ = RE.build_rocket_real(experts=experts_c, dt_scale=a.cheap_scale)
    for sid in seams:
        sub, x, y = _sub(graph_c, sid)
        op, _tr = RE.seam_operator(sub, budget=ProbeBudget(),
                                   probe_state="w312 cheap dt_scale=%g" % a.cheap_scale)
        cheap[sid] = _summary(op)
        print("  %-5s beta %-12.6g kappa %-10.4g omega %-10.4g dim M %-4s shares %s"
              % (sid, op.beta, op.kappa, op.operator_content, op.dim_M,
                 {k: round(b.share, 4) for k, b in op.blocks.items()
                  if b.share is not None}))
    print("  reduced-cadence arm: %.0f s" % (time.perf_counter() - t0))
    out["cheap"] = cheap

    # -- record at the declared clock ---------------------------------------
    print()
    print("=== recording the declared-clock request set ===")
    seen, per, _graph = record(a.dt_scale, seams)
    jobs = [seen[k] for k in seen]
    jobs.sort(key=lambda j: (j[0], j[1]))          # group by agent for the pool
    print("  %d distinct traces: %s" % (len(jobs), per))
    by_agent = {}
    for aid, _p, _t in jobs:
        by_agent[aid] = by_agent.get(aid, 0) + 1
    print("  by agent: %s" % by_agent)
    out["n_traces"] = len(jobs)
    out["per_seam_traces"] = per
    out["by_agent"] = by_agent

    # -- fill them in parallel ----------------------------------------------
    print()
    print("=== filling %d responses across %d workers ===" % (len(jobs), a.workers))
    cache: dict = {}
    times = []
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(a.workers, initializer=_pool_init, initargs=(a.dt_scale,)) as pool:
        for k, vec, dt in pool.imap_unordered(_pool_call, jobs, chunksize=1):
            cache[k] = vec
            times.append(dt)
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.perf_counter() - t0
                rate = done / el
                print("    %4d/%4d  %6.0f s elapsed  %5.2f calls/s  eta %5.0f s"
                      % (done, len(jobs), el, rate, (len(jobs) - done) / max(rate, 1e-9)),
                      flush=True)
                if a.json:
                    _persist(out, a, partial=dict(filled=done, total=len(jobs),
                                                  elapsed=el))
    fill = time.perf_counter() - t0
    serial = float(np.sum(times))
    print("  filled in %.0f s; serial equivalent %.0f s; speedup %.2fx"
          % (fill, serial, serial / max(fill, 1e-9)))
    out["fill"] = dict(wall_s=fill, serial_s=serial, speedup=serial / max(fill, 1e-9),
                       per_call_mean=float(np.mean(times)),
                       per_call_max=float(np.max(times)))

    # -- assemble through the framework -------------------------------------
    print()
    print("=== assembling, with the framework doing its own arithmetic ===")
    seen2: dict = {}
    graph, _ = _wrapped_graph(a.dt_scale, seen2, cache, "serve")
    conv = {}
    for sid in seams:
        sub, x, y = _sub(graph, sid)
        op, _tr = RE.seam_operator(sub, budget=ProbeBudget(),
                                   probe_state="w312 declared dt_scale=%g" % a.dt_scale)
        conv[sid] = _summary(op)
        print("  %-5s beta %-12.6g kappa %-10.4g omega %-10.4g dim M %-4s shares %s"
              % (sid, op.beta, op.kappa, op.operator_content, op.dim_M,
                 {k: round(b.share, 4) for k, b in op.blocks.items()
                  if b.share is not None}))
    out["converged"] = conv

    # -- the comparison -----------------------------------------------------
    print()
    print("=== what the reduced cadence was hiding ===")
    print("  %-6s %-14s %-14s %-10s %-10s" % ("seam", "beta reduced", "beta declared",
                                              "ratio", "sign same?"))
    moves = {}
    for sid in seams:
        c, d = cheap[sid], conv[sid]
        r = (d["beta"] / c["beta"]) if c["beta"] else float("inf")
        moves[sid] = r
        print("  %-6s %-14.6g %-14.6g %-10.4g %-10s"
              % (sid, c["beta"], d["beta"], r,
                 c["sign_structure"] == d["sign_structure"]))
    out["beta_movement"] = moves
    print()
    print("  Tier 79's anchor on e-f moved beta by 42.8x at dim M = 8.")
    out["wall_seconds"] = time.perf_counter() - t_start
    print()
    print("total %.0f s (%.2f h)" % (out["wall_seconds"], out["wall_seconds"] / 3600.0))
    _persist(out, a)
    return 0


def _summary(op):
    return dict(dim_M=op.dim_M, beta=op.beta, kappa=op.kappa,
                omega=op.operator_content, sign_structure=op.sign_structure,
                null_dim=op.null_dim, expected_null_dim=op.expected_null_dim,
                is_empty=bool(op.is_empty), one_sided=op.one_sided,
                effort_normal=op.effort_normal,
                blocks={k: dict(share=b.share, n_solves=b.n_solves,
                                norm_F=float(np.linalg.norm(b.S, "fro")))
                        for k, b in op.blocks.items()})


def _persist(out, a, partial=None):
    if not a.json:
        return
    body = dict(out)
    if partial:
        body["_progress"] = partial
    path = os.path.abspath(a.json)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    for k in range(5):
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(body, fh, indent=2, default=float)
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (k + 1))


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
