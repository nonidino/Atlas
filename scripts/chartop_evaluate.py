"""Stage 0 (W355): the one evaluation of the registered arms, against the registered gates.

    python scripts/chartop_evaluate.py --runs out/chartop/runs [--workers 16]

What it does, in order (`atlas/chartop/gates.py` defines every measure and bar):

1. refuses to run if any of the weights has been evaluated before
   (``out/chartop/evaluated.json`` holds their sha256), then records them;
2. draws the held-out test split (``gates.N_TEST`` seeds from ``SEED_BASE['test']``, with
   exact labels in both port bases) unless it is already on disk;
3. **E1** for arm B, and every arm's median ``e_H`` on the test split for C1-C7, with the
   network in float32 on this machine's CPU (``gates.EVAL_PRECISION``);
4. on the three benchmarks, steady and one step (``n = 64``): **E3, E4, SE1, SE2, SE3** for
   B, and E4 for every arm (C6's comparison reads B' and C6);
5. **DUAL** for B on the steady benchmark pieces and 100 steady test pieces;
6. **KT**, nonlinear conduction by frozen coefficients, for B on the steady benchmarks;
7. the **rollout** for S11 (the s-channel marched ``gates.ROLLOUT_STEPS`` steps);
8. writes ``out/chartop/evaluation-<time>.json`` and ``.txt``, whatever they say.

E5 (cost) is ``scripts/chartop_e5.py``: it needs AC power and repeats.  ``--out`` and
``--test-count`` exist for a smoke run of this script on throwaway weights in a scratch
folder; the registered run uses neither.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atlas.chartop import benchmarks as bm  # noqa: E402
from atlas.chartop import data, dual, fe, gates as G, host as hs, model as md  # noqa: E402
from atlas.chartop import nonlinear as nl, ports  # noqa: E402
from atlas.chartop import train as tr  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

KIND_OF = {"C6": "cosine"}


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------
# the test split
# ---------------------------------------------------------------------------


def ensure_test(test_dir: str, count: int, workers: int) -> None:
    if glob.glob(os.path.join(test_dir, "test_n*_*.npz")):
        return
    os.makedirs(test_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "chartop_data.py"), "--split", "test",
           "--count", str(count), "--workers", str(workers), "--labels", "sine-hat,cosine",
           "--out", test_dir]
    log("drawing the test split: " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)


def e1_rows(expert: md.Expert, test: dict) -> dict:
    """Per-example e_H and e_p on the labelled test split (float32 network on CPU,
    float64 objective)."""
    rows = {k: [] for k in ("e_H", "e_p", "sup_mu", "freq", "n", "seed", "fo", "family")}
    expert.eval()
    with torch.no_grad():
        for n, t in sorted(test.items()):
            N = t["inputs"].shape[0]
            bs = 16 if n <= 64 else 4
            for i0 in range(0, N, bs):
                idx = torch.arange(i0, min(N, i0 + bs))
                b, lab = tr._batch(t, idx, True)
                H, up = expert(b["inputs"], b["scalars"], b["s0"])
                o = md.objectives(b, H, up, expert.kind, expert.m, lab, "label-free")
                rows["e_H"] += o["e_H"].tolist()
                rows["e_p"] += o["e_p"].tolist()
            for k in ("sup_mu", "freq", "seed", "fo", "family"):
                rows[k] += list(t["meta_" + k])
            rows["n"] += [n] * N
    return {k: np.asarray(v) for k, v in rows.items()}


def e1_streamed(nets: dict, test_dir: str) -> dict:
    """e1_rows for every arm, one test shard at a time (the labels of 2400 examples in
    both port bases would not fit in this laptop's memory at once), without copies."""
    rows = {}
    for kind in ("sine-hat", "cosine"):
        arms = [a for a, n_ in nets.items() if n_.kind == kind]
        if not arms:
            continue
        pre = "labels_" if kind == "sine-hat" else f"labels-{kind}_"
        for f in sorted(glob.glob(os.path.join(test_dir, "test_n*_*.npz"))):
            z = np.load(f, allow_pickle=False)
            zl = np.load(os.path.join(test_dir, pre + os.path.basename(f)), allow_pickle=False)
            if not np.array_equal(z["seed"], zl["lseed"]):
                raise ValueError("test labels and examples disagree on seeds in " + f)
            t = {k: torch.from_numpy(np.ascontiguousarray(z[k], dtype=np.float32)) for k in tr.KEYS}
            t["H"] = torch.from_numpy(np.ascontiguousarray(zl["H"]))
            t["up"] = torch.from_numpy(np.ascontiguousarray(zl["up"]))
            t["trace_Lambda"] = torch.from_numpy(np.asarray(zl["trace_Lambda"], np.float64))
            t["up_energy"] = torch.from_numpy(np.asarray(zl["up_energy"], np.float64))
            for k in ("sup_mu", "freq", "seed", "fo", "family"):
                t["meta_" + k] = z[k]
            n = int(t["inputs"].shape[-1]) - 1
            for a in arms:
                r = e1_rows(nets[a], {n: t})
                acc = rows.setdefault(a, {k: [] for k in r})
                for k, v in r.items():
                    acc[k].append(v)
            del t, z, zl
    return {a: {k: np.concatenate(v) for k, v in r.items()} for a, r in rows.items()}


# ---------------------------------------------------------------------------
# the benchmarks
# ---------------------------------------------------------------------------


def network_term(layout: hs.Layout, learned: hs.Coupled, classical: hs.Coupled) -> float:
    """(sum_i kappa_i ||e_p,i + E_i R_i c*||^2_{A_I,i})^(1/2), c* the classical
    superelement's interface coefficients (Theorem 2's bracket)."""
    tot = 0.0
    for i, p in enumerate(layout.problems):
        rL, rC = learned.responses[i], classical.responses[i]
        ci = layout.R[i] @ classical.c
        w = (rL.up - rC.up) + (rL.H - rC.H) @ ci
        M = p.matrix()
        loop, inner = fe.grid(p.n)
        A_I = M.tocsr()[inner][:, inner]
        tot += p.kappa_ref * float(w @ (A_I @ w))
    return math.sqrt(max(tot, 0.0))


def field_errors(mono_fields, fields) -> tuple[float, float]:
    mx = max(float(np.max(np.abs(u - v))) for u, v in zip(mono_fields, fields))
    span = (max(float(np.max(u)) for u in mono_fields) - min(float(np.min(u)) for u in mono_fields))
    return mx, span


def gram_identity(layout: hs.Layout, learned: hs.Coupled, classical: hs.Coupled) -> float:
    worst = 0.0
    for i, p in enumerate(layout.problems):
        E = learned.responses[i].H - classical.responses[i].H
        M = p.matrix().tocsr()
        loop, inner = fe.grid(p.n)
        A_I = M[inner][:, inner]
        L, L0 = learned.grams[i], classical.grams[i]
        res = np.max(np.abs(L - L0 - E.T @ (A_I @ E))) / np.max(np.abs(L0))
        worst = max(worst, float(res))
    return worst


def benchmark_problem(name: str, problem: str, experts: dict) -> dict:
    t0 = time.perf_counter()
    b = bm.build(name, G.BENCHMARK_N, problem, **({"envelope": data.ENVELOPE} if name == "s-channel" else {}))
    out = {"name": name, "problem": problem, "pieces": len(b.problems),
           "charts": sorted({p.chart.method for p in b.problems})}
    L_sh = b.layout("sine-hat", 16)
    mono = hs.monolithic(L_sh)
    cl = L_sh.solve(hs.ClassicalExpert())
    un, ud = cl.energy_error(mono.fields)
    out["truncation_energy"] = un
    out["truncation_energy_rel"] = un / ud
    out["reference_energy"] = ud
    mx0, span = field_errors(mono.fields, cl.fields)
    out["truncation_max_over_range"] = mx0 / span
    out["span"] = span
    out["refusals"] = L_sh.check_envelope(data.ENVELOPE)
    cos_layout, cos_floor = None, None
    for arm, ex in experts.items():
        kind = ex.kind
        if kind == "cosine":
            if cos_layout is None:
                cos_layout = b.layout("cosine", 16)
                cc = cos_layout.solve(hs.ClassicalExpert())
                cos_floor = field_errors(mono.fields, cc.fields)[0]
                cn, cd = cc.energy_error(mono.fields)
                out["truncation_cosine"] = {"energy_rel": cn / cd, "max_over_range": cos_floor / span}
            L = cos_layout
            floor = cos_floor
        else:
            L = L_sh
            floor = mx0
        try:
            cp = L.solve(ex)
        except hs.Refused as r:
            out.setdefault("arms", {})[arm] = {"refused": str(r)}
            continue
        num, den = cp.energy_error(mono.fields)
        mx, _ = field_errors(mono.fields, cp.fields)
        rec = {"energy_rel": num / den, "max_over_range": mx / span, "calls": cp.calls,
               "calls_per_piece": cp.calls / len(b.problems), "min_eig": cp.min_eig(),
               "e4": G.judge_e4(num / den, mx, span, floor)}
        if kind == "sine-hat":
            nt = network_term(L, cp, cl)
            rec["network_term"] = nt
            rec["e3"] = G.judge_e3(num, un, nt)
            rec["se1"] = G.judge_se1(gram_identity(L, cp, cl), cp.min_eig(), cl.min_eig())
            rec["se3"] = G.judge_se3(un, nt)
            if arm == "B":
                out["B_solution"] = {"c": cp.c.tolist()}
                out["_coupled_B"] = cp
        out.setdefault("arms", {})[arm] = rec
    out["_bench"] = b
    out["_layout"] = L_sh
    out["_mono"] = mono
    out["elapsed_s"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# the dual bracket, kappa(T), the rollout
# ---------------------------------------------------------------------------


def dual_records(ex: LearnedExpert, problems: list, traces: list | None) -> list:
    pb = ports.basis(ex.kind, problems[0].n, 16)
    prim = ex.forward(problems)
    duals = ex.forward([dual.dual_problem(p) for p in problems])
    out = []
    for k, p in enumerate(problems):
        pbk = ports.basis(ex.kind, p.n, 16)
        tr_ = [traces[k]] if traces is not None else [smooth_coefficients(pbk, k)]
        out.append(dual.bracket(p, pbk, prim[k][0], duals[k][0], tr_))
    return out


def smooth_coefficients(pb, seed: int) -> np.ndarray:
    """Port data with a decaying spectrum (the catalog's 'smooth data'): vertex values
    N(0, 1), side amplitude j ~ N(0, 1) / (1 + j)^2."""
    rng = np.random.default_rng(10_000 + seed)
    c = np.empty(pb.size)
    for k, col in enumerate(pb.columns):
        c[k] = rng.normal() if col.what == "vertex" else rng.normal() / (1.0 + col.j) ** 2
    return c


def kt_record(name: str, ex: LearnedExpert) -> dict:
    b = bm.build(name, G.BENCHMARK_N, "steady", **({"envelope": data.ENVELOPE} if name == "s-channel" else {}))
    mat = {45.0: "steel", 400.0: "copper"}
    betas = [G.KAPPA_T_BETA[mat[p.kappa_ref]] for p in b.problems]
    cl = nl.iterate(b, betas, None, tol=G.KAPPA_T_TOL, maxit=G.KAPPA_T_MAX_ITERATIONS)
    le = nl.iterate(b, betas, ex, tol=G.KAPPA_T_TOL, maxit=G.KAPPA_T_MAX_ITERATIONS)
    cs = nl.iterate(b, betas, hs.ClassicalExpert(), tol=G.KAPPA_T_TOL, maxit=G.KAPPA_T_MAX_ITERATIONS)
    final = nl.frozen_problems(b, cl["fields"], betas, [p.kappa_ref for p in b.problems])
    num = den = 0.0
    for p, u, v in zip(final, cl["fields"], le["fields"]):
        M = p.matrix()
        e = u - v
        num += p.kappa_ref * float(e @ (M @ e))
        den += p.kappa_ref * float(u @ (M @ u))
    mx, span = field_errors(cl["fields"], le["fields"])
    floor = field_errors(cl["fields"], cs["fields"])[0]
    rec = {"classical_iterations": cl["iterations"], "classical_converged": cl["converged"],
           "superelement16_iterations": cs["iterations"], "floor_max_over_range": floor / span,
           "learned_iterations": le["iterations"], "learned_converged": le["converged"],
           "learned_history": le["history"], "network_calls": le["network_calls"],
           "kt": G.judge_kt(le["iterations"], le["converged"], math.sqrt(num / den), mx, span, floor)}
    lin = hs.monolithic(b.layout("sine-hat", 16)).fields
    if len(set(betas)) == 1:
        kh = nl.kirchhoff(b, betas[0], lin, bm.T_HOT, bm.T_COLD)
        mxk, _ = field_errors(cl["fields"], kh)
        rec["kirchhoff_vs_frozen_classical_max_over_range"] = mxk / span
        lin_learned = b.layout("sine-hat", 16).solve(ex).fields
        khl = nl.kirchhoff(b, betas[0], lin_learned, bm.T_HOT, bm.T_COLD)
        mxl, _ = field_errors(kh, khl)
        rec["kirchhoff_learned_vs_classical_max_over_range"] = mxl / span
    rec["nonlinearity_max_over_range"] = field_errors(cl["fields"], lin)[0] / span
    return rec


def rollout_record(ex: LearnedExpert) -> dict:
    name = G.ROLLOUT_BENCHMARK
    b = bm.build(name, G.BENCHMARK_N, "step", envelope=data.ENVELOPE)
    fos = [p.fo for p in b.problems]
    T_c = [np.full((p.n + 1) ** 2, bm.T_OLD) for p in b.problems]
    T_l = [x.copy() for x in T_c]
    T_s = [x.copy() for x in T_c]
    first = None
    recs = []
    calls0 = ex.calls
    for step in range(1, G.ROLLOUT_STEPS + 1):
        def problems_from(T):
            return [hs.PieceProblem(p.chart, fo=fo, g=(t / fo).reshape(p.n + 1, p.n + 1),
                                    kappa_ref=p.kappa_ref, label=p.label)
                    for p, fo, t in zip(b.problems, fos, T)]
        Pc = problems_from(T_c)
        L = hs.Layout(Pc, b.bc, "sine-hat", 16)
        T_c = hs.monolithic(L).fields
        Pl = problems_from(T_l)
        L2 = hs.Layout(Pl, b.bc, "sine-hat", 16)
        resp = ex.respond(Pl, L2.pb, modes_from=first)
        if first is None:
            first = resp
        cp = L2.couple(resp)
        T_l = cp.fields
        Ps = problems_from(T_s)
        T_s = hs.Layout(Ps, b.bc, "sine-hat", 16).solve(hs.ClassicalExpert()).fields
        if step in (1, 5, 10, 20) or step == G.ROLLOUT_STEPS:
            num = den = 0.0
            for p, u, v in zip(Pc, T_c, T_l):
                M = p.matrix()
                e = u - v
                num += p.kappa_ref * float(e @ (M @ e))
                den += p.kappa_ref * float(u @ (M @ u))
            mx, span = field_errors(T_c, T_l)
            span = max(span, 1e-12)
            floor = field_errors(T_c, T_s)[0]
            recs.append({"step": step, "energy_rel": math.sqrt(num / den), "max_over_range": mx / span,
                         "floor_max_over_range": floor / span,
                         "e4": G.judge_e4(math.sqrt(num / den), mx, span, floor)})
    return {"benchmark": name, "steps": G.ROLLOUT_STEPS, "records": recs,
            "network_calls": ex.calls - calls0, "pieces": len(b.problems)}


# ---------------------------------------------------------------------------


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [strip(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def summarise(e: dict, arm: str) -> dict:
    return {"e_H_median": float(np.median(e["e_H"])), "e_H_p90": float(np.quantile(e["e_H"], 0.9)),
            "e_p_median": float(np.median(e["e_p"])), "e_p_p90": float(np.quantile(e["e_p"], 0.9)),
            "count": int(len(e["e_H"]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=os.path.join(ROOT, "out", "chartop", "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "chartop"))
    ap.add_argument("--test-dir", default=os.path.join(ROOT, "out", "chartop", "data", "test"))
    ap.add_argument("--test-count", type=int, default=G.N_TEST)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dual-test", type=int, default=100)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    arms = {}
    for a in G.ARMS:
        p = os.path.join(args.runs, a, "best.pt")
        if os.path.exists(p):
            arms[a] = p
    if "B" not in arms:
        raise SystemExit("no weights for arm B in " + args.runs)
    reg = os.path.join(args.out, "evaluated.json")
    seen = json.load(open(reg, encoding="utf-8")) if os.path.exists(reg) else {}
    hashes = {a: sha256(p) for a, p in arms.items()}
    again = [a for a, h in hashes.items() if h in seen.values()]
    if again:
        raise SystemExit(f"weights already evaluated: {again}; the evaluation runs once")
    seen.update({f"{dt.date.today().isoformat()}:{a}": h for a, h in hashes.items()})
    with open(reg, "w", encoding="utf-8") as fh:
        json.dump(seen, fh, indent=1)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    res = {"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "gates_sha256": hashlib.sha256(open(os.path.join(ROOT, "atlas", "chartop", "gates.py"), "rb")
                                          .read().replace(b"\r\n", b"\n")).hexdigest(),
           "weights": hashes, "precision": G.EVAL_PRECISION, "torch_threads": args.threads,
           "platform": platform.platform(), "torch": torch.__version__,
           "test_count": args.test_count}
    nets = {a: tr.load_expert(p, "cpu") for a, p in arms.items()}
    res["configs"] = {a: n.config() for a, n in nets.items()}
    # 2-3: E1 and the comparisons' medians
    ensure_test(args.test_dir, args.test_count, args.workers)
    t0 = time.perf_counter()
    e1 = e1_streamed(nets, args.test_dir)
    for a in nets:
        log(f"E1 rows for {a}: median e_H {np.median(e1[a]['e_H']):.4g}, "
            f"median e_p {np.median(e1[a]['e_p']):.4g}")
    log(f"E1 rows for {len(nets)} arms in {time.perf_counter() - t0:.0f} s")
    eB = e1["B"]
    res["E1"] = G.judge_e1(eB["e_H"], eB["e_p"], eB["sup_mu"], eB["freq"])
    res["test_summary"] = {a: summarise(e, a) for a, e in e1.items()}
    res["E1_by_resolution"] = {str(n): summarise({k: v[eB["n"] == n] for k, v in eB.items()}, "B")
                               for n in sorted(set(eB["n"].tolist()))}
    res["E1_steady_vs_step"] = {
        "steady": summarise({k: v[eB["fo"] < 0] for k, v in eB.items()}, "B"),
        "step": summarise({k: v[eB["fo"] > 0] for k, v in eB.items()}, "B")}
    np.savez(os.path.join(args.out, f"evaluation-{stamp}-test-rows.npz"),
             **{f"{a}__{k}": v for a, e in e1.items() for k, v in e.items() if k != "family"})
    cmp_ = {}
    med = {a: float(np.median(e["e_H"])) for a, e in e1.items()}
    if "C1" in e1 and "Bp" in e1:
        s1 = e1["C1"]["sup_mu"] < 0.15
        s0 = e1["Bp"]["sup_mu"] < 0.15
        cmp_["C1"] = dict(G.compare(float(np.median(e1["C1"]["e_H"][s1])),
                                    float(np.median(e1["Bp"]["e_H"][s0]))),
                          near_conformal_test_pieces=int(s1.sum()))
    if "C2-sup" in med and "C2-free" in med:
        cmp_["C2"] = G.compare(med["C2-sup"], med["C2-free"])
    for c, a in (("C3", "C3"), ("C4", "C4"), ("C7", "C7")):
        if a in med and "Bp" in med:
            cmp_[c] = G.compare(med[a], med["Bp"])
    sizes = {s: med[a] for s, a in (("0.3M", "C5-0.3M"), ("1M", "Bp"), ("3M", "C5-3M")) if a in med}
    if len(sizes) == 3:
        cmp_["C5"] = {"medians": sizes, "choice": G.c5_choice(sizes)}
    # 4: the benchmarks
    experts = {a: LearnedExpert(n_, envelope=data.ENVELOPE) for a, n_ in nets.items()}
    bench = []
    for name in G.HELD_OUT_BENCHMARKS:
        for problem in G.BENCHMARK_PROBLEMS:
            r = benchmark_problem(name, problem, experts)
            log(f"benchmark {name} {problem}: B energy {r['arms']['B'].get('energy_rel')}, "
                f"truncation {r['truncation_energy_rel']:.3g} ({r['elapsed_s']:.0f} s)")
            bench.append(r)
    res["benchmarks"] = strip(bench)
    gate = {}
    for key, sub in (("E3", "e3"), ("E4", "e4"), ("SE1", "se1"), ("SE3", "se3")):
        rows = [r["arms"]["B"][sub] for r in bench if sub in r["arms"].get("B", {})]
        gate[key] = {"met": bool(rows and all(x["met"] for x in rows) and len(rows) == len(bench)),
                     "problems": len(rows)}
    gate["SE2"] = {"met": all(r["arms"]["B"].get("calls_per_piece") == G.SE2_CALLS_PER_PIECE
                              for r in bench)}
    res["gates"] = gate
    if "C6" in nets and "Bp" in nets:
        e_c6 = [r["arms"]["C6"]["energy_rel"] for r in bench if "energy_rel" in r["arms"].get("C6", {})]
        e_bp = [r["arms"]["Bp"]["energy_rel"] for r in bench if "energy_rel" in r["arms"].get("Bp", {})]
        if len(e_c6) == len(bench) and len(e_bp) == len(bench):
            cmp_["C6"] = dict(G.compare(float(np.mean(e_c6)), float(np.mean(e_bp))),
                              e_H_median_C6=med.get("C6"), e_H_median_Bp=med.get("Bp"))
    res["comparisons"] = cmp_
    # 5: the dual bracket on steady pieces
    exB = experts["B"]
    drec = []
    for r in bench:
        if r["problem"] != "steady":
            continue
        L, cp = r["_layout"], r["_coupled_B"]
        traces = [L.R[i] @ cp.c for i in range(L.P)]
        drec += dual_records(exB, r["_bench"].problems, traces)
    steady_seeds = [int(s) for s, f in zip(eB["seed"], eB["fo"]) if f < 0][:args.dual_test]
    test_problems = [data.draw(s).problem for s in steady_seeds]
    dtest = []
    for p in test_problems:
        dtest += dual_records(exB, [p], None)
    allmin = [min(d["min_eig_rel"].values()) for d in drec + dtest]
    res["DUAL"] = {**G.judge_dual(allmin), "benchmark_pieces": len(drec), "test_pieces": len(dtest),
                   "benchmark_records": drec, "test_records": dtest}
    # 6: kappa(T)
    res["KT"] = {name: kt_record(name, exB) for name in G.HELD_OUT_BENCHMARKS}
    res["gates"]["KT"] = {"met": all(v["kt"]["met"] for v in res["KT"].values())}
    res["gates"]["DUAL"] = {"met": res["DUAL"]["met"]}
    res["gates"]["E1"] = {"met": res["E1"]["met"]}
    # 7: the rollout
    res["rollout"] = rollout_record(exB)
    path = os.path.join(args.out, f"evaluation-{stamp}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(strip(res), fh, indent=1)
    log("wrote " + path)
    lines = [f"STAGE 0 EVALUATION {res['written']}  (gates sha256 {res['gates_sha256'][:16]}...)", ""]
    for k, v in res["gates"].items():
        lines.append(f"  {k:5s} {'met' if v['met'] else 'NOT MET'}")
    lines.append("")
    for k, v in cmp_.items():
        lines.append(f"  {k}: {json.dumps(strip(v))}")
    with open(path.replace(".json", ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
