"""Stage 0 (W355): gate E5, the cost -- reported whatever it says.

    python scripts/chartop_e5.py --weights out/chartop/runs/B/best.pt [--device cpu|cuda]

On each benchmark (n = 64 per piece) the wall time of every arm the gate names, the
median of ``gates.E5_REPEATS`` runs:

- **monolithic**: one bilinear mesh of every piece, assembled, factorised (SuperLU) and
  solved -- *cold*; and with the factorisation kept, a new load solved -- *warm*;
- **classical superelement**: per piece a factorisation and 4m + 1 pairs of triangular
  solves, the Grams, one interface solve, the reconstruction -- *cold*; with the modes
  and the interface factorisation stored, a new load (one particular solve per piece) --
  *warm*;
- **learned superelement**: the network's one call per piece (batched), the Grams, one
  interface solve, the reconstruction -- *cold*; with the Grams stored, a new load (one
  call per piece for the particular field) -- *warm*;
- **coarse classical**: the monolithic solve on every piece's grid halved (n = 32);
- **the design loop**: ``gates.E5_DESIGN_LOOP`` variants of bend-3 with the inner radius
  changed, so no factorisation or mode survives from one solve to the next: monolithic,
  classical superelement and learned superelement, all cold, with the learned answer's
  error against the monolithic one beside the times.

On the fixed benchmarks every piece's energy matrix is assembled once and shared by all
arms (assembly is the same work for each of them); in the design loop every arm pays
its own assembly, since the geometry is new each time.

The network runs in float32 on a CPU (decision D4) or in bfloat16 on a GPU, and its
share of every learned time is reported separately.  Records the power state: on the
laptop, numbers count only on AC power (the vault's memory "standby suspends a suite").
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import socket
import statistics
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
import torch  # noqa: E402

from atlas.chartop import benchmarks as bm  # noqa: E402
from atlas.chartop import data, fe, gates as G, host as hs  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402
from atlas.chartop.train import load_expert  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from chartop_price import power_state  # noqa: E402


def med(fn, reps):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


def mono_factor(L: hs.Layout):
    """Assemble and factorise the monolithic matrix once; return a solver for loads."""
    from scipy.spatial import cKDTree
    pts = [p.chart.X.reshape(-1, 2) for p in L.problems]
    allp = np.concatenate(pts)
    tree = cKDTree(allp)
    gid = -np.ones(len(allp), int)
    ng = 0
    for k in range(len(allp)):
        if gid[k] >= 0:
            continue
        for q in tree.query_ball_point(allp[k], L.tol):
            gid[q] = ng
        ng += 1
    offs = np.cumsum([0] + [len(x) for x in pts])
    Gi = [gid[offs[i]:offs[i + 1]] for i in range(L.P)]
    rows, cols, vals = [], [], []
    for i, p in enumerate(L.problems):
        M = p.matrix().tocoo()
        rows.append(Gi[i][M.row]); cols.append(Gi[i][M.col]); vals.append(p.kappa_ref * M.data)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ng, ng))
    dval = {}
    for e in L.edges:
        if e.other is None and e.bc is not None and e.bc[0] == "dirichlet":
            i, s = e.owner
            from atlas.chartop import ports as pt
            pos = pt.side_positions(L.problems[i].n, s)
            loop = fe.grid(L.problems[i].n)[0]
            for q, v in zip(loop[pos], e.bc[1]):
                dval[int(Gi[i][q])] = float(v)
    dn = np.array(sorted(dval), int)
    fn = np.setdiff1d(np.arange(ng), dn)
    lu = spla.splu(K[fn][:, fn].tocsc())
    Kfd = K[fn][:, dn]
    ud = np.array([dval[d] for d in dn])

    def solve(loads):
        rhs = np.zeros(ng)
        for i, p in enumerate(L.problems):
            np.add.at(rhs, Gi[i], p.kappa_ref * loads[i])
        u = np.zeros(ng)
        u[dn] = ud
        u[fn] = lu.solve(rhs[fn] - Kfd @ ud)
        return [u[Gi[i]] for i in range(L.P)]
    return solve


def superelement_parts(L: hs.Layout, responses):
    """The host's assembly and interface factorisation, kept for warm solves."""
    cp = L.couple(responses)
    fixed = np.array(sorted(L.fixed), int)
    free = cp.free
    Sff = cp.S[np.ix_(free, free)]
    cho = sla.cho_factor(Sff)
    return cp, cho, fixed, free


def warm_superelement(L: hs.Layout, cp, cho, fixed, free, ups):
    """New particular fields ups (one per piece), stored modes and Grams."""
    N = L.N
    chi = np.zeros(N)
    for i, (p, r) in enumerate(zip(L.problems, cp.responses)):
        pbi = L.pb(p.n)
        M = p.matrix() if not hasattr(p, "_M") else p._M
        V = fe.full_field(p.n, r.H, pbi.Q)
        b = p.load()
        Up = fe.full_field(p.n, ups[i], np.zeros(4 * p.n))
        chi += p.kappa_ref * (L.R[i].T @ (V.T @ b - V.T @ (M @ Up)))
    c = np.zeros(N)
    c[fixed] = [L.fixed[d] for d in fixed]
    c[free] = sla.cho_solve(cho, chi[free] - cp.S[np.ix_(free, fixed)] @ c[fixed])
    return [fe.full_field(p.n, ups[i] + cp.responses[i].H @ (L.R[i] @ c), L.pb(p.n).Q @ (L.R[i] @ c))
            for i, p in enumerate(L.problems)]


def bench_times(name: str, problem: str, ex: LearnedExpert, reps: int) -> dict:
    kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
    b = bm.build(name, 64, problem, **kw)
    L = b.layout("sine-hat", 16)
    for p in L.problems:                      # assembly is shared by every arm: cache it
        p._M = p.matrix()
        p.matrix = (lambda q=p: q._M)
    out = {"name": name, "problem": problem, "pieces": len(b.problems)}
    t, mono = med(lambda: hs.monolithic(L), reps)
    out["monolithic_cold_s"] = t
    solver = mono_factor(L)
    loads = [p.load() for p in L.problems]
    t, _ = med(lambda: solver(loads), reps)
    out["monolithic_warm_s"] = t
    t, cpc = med(lambda: L.solve(hs.ClassicalExpert()), reps)
    out["classical_superelement_cold_s"] = t
    cp, cho, fixed, free = superelement_parts(L, cpc.responses)
    exacts = [fe.Exact(p._M, p.n) for p in L.problems]
    inner = [fe.grid(p.n)[1] for p in L.problems]

    def warm_classical():
        ups = [exs.particular(p.load()[inn]) for exs, p, inn in zip(exacts, L.problems, inner)]
        return warm_superelement(L, cp, cho, fixed, free, ups)
    t, _ = med(warm_classical, reps)
    out["classical_superelement_warm_s"] = t
    t_fw, _ = med(lambda: ex.forward(L.problems), reps)
    out["learned_forward_s"] = t_fw
    t, cpl = med(lambda: L.solve(ex), reps)
    out["learned_superelement_cold_s"] = t
    cpl_parts = superelement_parts(L, cpl.responses)

    def warm_learned():
        fw = ex.forward(L.problems)
        return warm_superelement(L, cpl_parts[0], cpl_parts[1], cpl_parts[2], cpl_parts[3],
                                 [up for _H, up in fw])
    t, _ = med(warm_learned, reps)
    out["learned_superelement_warm_s"] = t
    num, den = cpl.energy_error(mono.fields)
    out["learned_energy_rel"] = num / den
    bc = bm.build(name, 32, problem, **kw)
    Lc = bc.layout("sine-hat", 16)
    t, monoc = med(lambda: hs.monolithic(Lc), reps)
    out["coarse_monolithic_cold_s"] = t
    # coarse error at the coarse nodes against the fine solution at the matching reference nodes
    err = 0.0
    span = max(float(np.max(u)) for u in mono.fields) - min(float(np.min(u)) for u in mono.fields)
    for uf, uc, pf in zip(mono.fields, monoc.fields, b.problems):
        n = pf.n
        Uf = uf.reshape(n + 1, n + 1)[::2, ::2].ravel()
        err = max(err, float(np.max(np.abs(Uf - uc))))
    out["coarse_max_over_range_approx"] = err / span
    return out


def design_loop(ex: LearnedExpert, reps: int) -> dict:
    """bend-3 with the inner radius changed per variant (cells of 5 mm)."""
    import atlas.chartop.benchmarks as B
    rows = []
    radii = np.linspace(34.0, 48.0, G.E5_DESIGN_LOOP)
    orig = B.bend3

    def variant(r0_cells):
        from atlas.workbench import spec as wspec
        dx = float(wspec.example_case("bend-3").domain.dx)
        cx = cy = 4.0 * dx
        r0, r1 = r0_cells * dx, 100.0 * dx
        n_r, n_th = 3, 5
        rho0, rho1 = math.log(r0), math.log(r1)
        dr, dth = (rho1 - rho0) / n_r, (math.pi / 2) / n_th
        mat = B._material("steel")
        probs = []
        for bb in range(n_th):
            for a in range(n_r):
                def F(xi, eta, a=a, bb=bb):
                    rr = np.exp(rho0 + (a + xi) * dr)
                    th = (bb + eta) * dth
                    return cx + rr * np.cos(th), cy + rr * np.sin(th)
                c = B.ch.Chart(B.ch.from_map(F, 64), "conformal-log-polar")
                probs.append(B._finish(c, mat, "steady", B.DT["bend-3"], f"bend {a},{bb}"))
        ref = orig(64, "steady")
        return B.Bench("bend-3-variant", "steady", probs, ref.bc)
    for r0 in radii:
        bv = variant(r0)
        L = bv.layout("sine-hat", 16)
        t_m, mono = med(lambda: hs.monolithic(L), reps)
        t_c, _ = med(lambda: L.solve(hs.ClassicalExpert()), reps)
        t_l, cpl = med(lambda: L.solve(ex), reps)
        num, den = cpl.energy_error(mono.fields)
        rows.append({"inner_radius_cells": float(r0), "monolithic_s": t_m,
                     "classical_superelement_s": t_c, "learned_superelement_s": t_l,
                     "learned_energy_rel": num / den})
    tot = {k: sum(r[k] for r in rows) for k in ("monolithic_s", "classical_superelement_s",
                                                  "learned_superelement_s")}
    return {"variants": rows, "totals_s": tot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=os.path.join(ROOT, "out", "chartop", "runs", "B", "best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "chartop"))
    ap.add_argument("--reps", type=int, default=G.E5_REPEATS)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    net = load_expert(args.weights, args.device)
    ex = LearnedExpert(net, envelope=data.ENVELOPE, device=args.device)
    rec = {"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "platform": platform.platform(), "device": args.device,
           "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
           "torch_threads": args.threads, "power_before": power_state(), "reps": args.reps,
           "weights": args.weights}
    rec["benchmarks"] = []
    for name in G.HELD_OUT_BENCHMARKS:
        for problem in G.BENCHMARK_PROBLEMS:
            r = bench_times(name, problem, ex, args.reps)
            print(json.dumps(r), flush=True)
            rec["benchmarks"].append(r)
    rec["design_loop"] = design_loop(ex, args.reps)
    print(json.dumps(rec["design_loop"]["totals_s"]), flush=True)
    rec["power_after"] = power_state()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "e5-%s-%s-%s.json" % (socket.gethostname().split(".")[0][:24],
                                                        args.device, dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
