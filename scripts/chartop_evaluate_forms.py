"""The registration after Stage 0 (`atlas/chartop/gates_forms.py`, pinned): the one evaluation.

    python scripts/chartop_evaluate_forms.py [--workers 16]
    python scripts/chartop_evaluate_forms.py --resume

Every measure and bar is `gates_forms`'s; the bars it shares with Stage 0 are `gates`'s,
read through it.  What it does, in order:

0. **The weights.**  Every run of the plan (``out/chartop/plan-forms.json``) must have its
   ``best.pt`` with the sha256 recorded when it was downloaded
   (``out/chartop/runs/forms/weights-sha256.txt``), and a configuration that is its
   registered arm's (`gates_forms.ARMS`: form, networks, ports, bubble, strips).  Otherwise
   it refuses.
1. **The registry.**  It refuses weights evaluated before (`scripts/chartop_registry.py`, the
   registry Stage 0's evaluation keeps), records these and opens
   ``evaluation-forms-<time>-checkpoints/``.
2. **The held-out test split:** Stage 0's own, the same registered seeds and labels
   (`chartop_evaluate.ensure_test`).  It is reused when it is whole on disk and drawn
   otherwise, and its sha256 is taken.
3. **E1** for S: ``e_H,L`` on the 36 learned columns against their own energy, and ``e_p``,
   in Stage 0's bins and bars.  Beside it, ``e_H`` over all 64 columns.  **K** on every test
   piece (``e_K``, ``e_K,inf``).  The network runs in float32 on this machine's CPU
   (`gates_forms.EVAL_PRECISION`), and every call solves the strips.
4. **L** for S on the first `gates_forms.L_TEST_PIECES` test pieces.
5. **The six benchmark problems** at n = 64, each as registered and with every temperature
   lowered by 300 K, for all five runs:
   - **E4** in both frames, **D**, and **K** and **L** on every piece;
   - ``e_H`` against the exact modes on every piece (``e_H,L`` too);
   - for S, also **E3, SE1, SE2, SE3**, and **DUAL** on the steady problems' pieces.
6. **CW**, W over S', seed with seed (`gates_forms.compare_pairings`), on
   `gates_forms.cw_measure`.  Beside it go the median ``e_H`` over the benchmark pieces of each
   run, and CW0 from the untrained record.
7. **DUAL** for S on `--dual-test` steady test pieces.
8. **KT** for S.
9. **S11**, S's rollout.
10. It writes ``evaluation-forms-<time>.json`` and ``.txt``, whatever they say.

The comparison arms (S' and W) are evaluated on the benchmarks, as `gates_forms` says;
only S runs on the test split.  E5 was priced on the boxes before training (the
``price-forms-*.json`` records); the evaluation names those records and does not price.

**Resuming** is Stage 0's (`chartop_evaluate`).  Every unit is a checkpoint, written whole
the moment it is done: one test shard, L's test pieces, one benchmark problem, a chunk of
the dual bracket's test pieces, one benchmark's KT, the rollout.  ``--resume`` continues an
interrupted evaluation of the same weights, the same identity and the same test split by
hash, or it refuses.  A finished evaluation is refused, with or without ``--resume``.
``--out``, ``--runs``, ``--plan``, ``--test-dir`` and ``--test-count`` exist for tests on
throwaway weights; the registered run uses none of them.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import replace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_cpus  # noqa: E402
import chartop_evaluate as ev0  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks as bm  # noqa: E402
from atlas.chartop import data, fe, forms as fm, gates as G, gates_forms as GF, host as hs  # noqa: E402
from atlas.chartop import model as md, train as tr  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

FORMS = os.path.join(ROOT, "out", "chartop")
RUNS_DIR = os.path.join(FORMS, "runs", "forms")
PLAN = os.path.join(FORMS, "plan-forms.json")
UNTRAINED = os.path.join(FORMS, "forms-controls-corner-strips.json")
BENCH_N = GF.BENCHMARK_N
DUAL_CHUNK = 20
CONFIG_KEYS = ("form", "backbone", "size", "p_size", "kind", "m", "bubble", "strips")
LEARNED = list(fm.learned_columns(GF.KIND, GF.M_PORTS))
log = ev0.log


def module_sha256(name: str) -> str:
    with open(os.path.join(ROOT, "atlas", "chartop", name), "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def script_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------------------
# 0: the weights are the registered runs'
# ---------------------------------------------------------------------------


def registered_runs(plan: dict, runs_dir: str) -> dict:
    """{run: {"path", "sha256", "registered", "seed"}} for every run of the plan, or SystemExit."""
    manifest = {}
    with open(os.path.join(runs_dir, "weights-sha256.txt"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                h, rel = line.split()
                manifest[rel.lstrip("*").split("/")[0]] = h
    out = {}
    for d in plan["arms"]:
        name, reg = d["name"], d["registered"]
        path = os.path.join(runs_dir, name, "best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"no weights for run {name!r} in {runs_dir}")
        h = registry.sha256_file(path)
        if manifest.get(name) != h:
            raise SystemExit(f"{path}: sha256 {h[:16]}... is not the one recorded at download "
                             f"({str(manifest.get(name))[:16]}...)")
        cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
        r = GF.ARMS[reg]
        want = dict(GF.COMMON, bubble=r["bubble"], strips=r["strips"])
        want = {k: want[k] for k in CONFIG_KEYS if k in want}
        got = {k: cfg.get(k) for k in want}
        if got != want:
            raise SystemExit(f"run {name!r}: configuration {got} is not registered arm {reg!r}'s {want}")
        out[name] = {"path": path, "sha256": h, "registered": reg, "seed": int(d["seed"])}
    if sum(1 for v in out.values() if v["registered"] == "S") != 1:
        raise SystemExit("the plan must hold exactly one run of the main arm S")
    return out


# ---------------------------------------------------------------------------
# 3-4: E1, K and L on the test split
# ---------------------------------------------------------------------------


def e1_rows(expert, test: dict) -> dict:
    """Per test example: ``e_H`` (all 64 columns), ``e_H,L`` (the learned columns, against
    their own energy), ``e_p``, ``e_K`` and ``e_K,inf``, float32 network on the CPU and the
    float64 objective.  ``Lambda_kk`` is the energy of the exact mode ``k``; the record keeps
    the largest relative gap between their sum and the labels' ``tr Lambda``."""
    keys = ("e_H", "e_H_L", "e_p", "e_K", "e_K_inf", "trace_gap", "sup_mu", "freq", "n", "seed", "fo",
            "family")
    rows = {k: [] for k in keys}
    expert.eval()
    with torch.no_grad():
        for n, t in sorted(test.items()):
            ps = md.plain_square(n, expert.kind, expert.m, "cpu")
            ks = fm.kernel_square(n, expert.kind, expert.m, "cpu")
            N = t["inputs"].shape[0]
            bs = 16 if n <= 64 else 4
            for i0 in range(0, N, bs):
                idx = torch.arange(i0, min(N, i0 + bs))
                b, lab = tr._batch(t, idx, True)
                H, up = fm.call(expert, b)
                o = md.objectives(b, H, up, expert.kind, expert.m, lab, "label-free")
                coef = (b["a11"], b["a12"], b["a22"], b["sigma"])
                err = md.energy(*coef, md.full(ps, H - lab["H"]) - ps.QB[None]).numpy()
                lam = md.energy(*coef, md.full(ps, lab["H"])).numpy()
                r = fm.residual(H, expert.last_level, b["s0"], ks)
                rows["e_H"] += o["e_H"].tolist()
                rows["e_p"] += o["e_p"].tolist()
                rows["e_H_L"] += [GF.e_h_learned(err[i], lam[i], LEARNED) for i in range(len(idx))]
                rows["e_K"] += fm.kernel_error_torch(b, r, lab["trace_Lambda"]).tolist()
                rows["e_K_inf"] += r.abs().amax(dim=(-2, -1)).tolist()
                rows["trace_gap"] += (np.abs(lam.sum(-1) - lab["trace_Lambda"].numpy())
                                      / lab["trace_Lambda"].numpy()).tolist()
            for k in ("sup_mu", "freq", "seed", "fo", "family"):
                rows[k] += list(t["meta_" + k])
            rows["n"] += [n] * N
    return {k: np.asarray(v) for k, v in rows.items()}


def e1_shard(expert, shard: str, test_dir: str, ev=None) -> dict:
    """`e1_rows` for one test shard (sine-hat labels), a checkpoint when ``ev`` is given."""
    name = "e1-forms-" + os.path.basename(shard)
    if ev is not None:
        blob = ev.load(name)
        if blob is not None:
            log(f"{name}: from the checkpoint")
            return ev0._npz_load(blob)
    labels = os.path.join(test_dir, ev0.label_prefix(GF.KIND) + os.path.basename(shard))
    with np.load(shard, allow_pickle=False) as z, np.load(labels, allow_pickle=False) as zl:
        if not np.array_equal(z["seed"], zl["lseed"]):
            raise ValueError("test labels and examples disagree on seeds in " + shard)
        t = {k: torch.from_numpy(np.ascontiguousarray(z[k], dtype=np.float32)) for k in tr.KEYS}
        t["H"] = torch.from_numpy(np.ascontiguousarray(zl["H"]))
        t["up"] = torch.from_numpy(np.ascontiguousarray(zl["up"]))
        t["trace_Lambda"] = torch.from_numpy(np.asarray(zl["trace_Lambda"], np.float64))
        t["up_energy"] = torch.from_numpy(np.asarray(zl["up_energy"], np.float64))
        for k in ("sup_mu", "freq", "seed", "fo", "family"):
            t["meta_" + k] = z[k]
    n = int(t["inputs"].shape[-1]) - 1
    rows = e1_rows(expert, {n: t})
    if ev is not None:
        ev.save(name, ev0._npz_bytes({k: v for k, v in rows.items()}))
    return rows


def e1_streamed(expert, test_dir: str, ev=None) -> dict:
    parts = [e1_shard(expert, f, test_dir, ev) for f in ev0.split_shards(test_dir)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load_blind(ex: LearnedExpert, problems: list, seed: int = 11) -> list:
    """Gate L per piece: the modes and the level field from calls with the piece's own source,
    no source and a unit-RMS random source (`gates_forms.L_SOURCES`), bit for bit."""
    rnd = []
    for k, p in enumerate(problems):
        t = np.arange(p.n + 1) / p.n
        XI, ET = np.meshgrid(t, t)
        f = data.Field(np.random.default_rng(seed + k), 0.25)(XI, ET)
        rnd.append(replace(p, g=f / float(np.sqrt(np.mean(f ** 2)))))
    calls = (list(problems), [replace(p, g=np.zeros_like(p.g)) for p in problems], rnd)
    modes, levels = [], []
    for probs in calls:
        modes.append([h for h, _up in ex.forward(probs)])
        levels.append(list(ex.last_levels))
    return [GF.judge_l([m[i] for m in modes], [v[i] for v in levels])["met"] for i in range(len(problems))]


def test_pieces(count: int) -> list:
    """The first ``count`` test pieces, drawn again from their registered seeds."""
    return [data.draw(GF.SEED_BASE["test"] + i).problem for i in range(count)]


# ---------------------------------------------------------------------------
# 5: the benchmarks, in both frames
# ---------------------------------------------------------------------------


def piece_measures(L: hs.Layout, cp: hs.Coupled, cl: hs.Coupled, ex: LearnedExpert) -> dict:
    """Per piece: ``e_H`` and ``e_H,L`` against the exact modes, and the uniform state's residual."""
    eh, ehl, ek, eki = [], [], [], []
    ups = [fm.uniform_state(p) for p in L.problems]
    res = ex.respond(ups, L.pb)
    for i, p in enumerate(L.problems):
        pb = L.pb(p.n)
        _loop, inner = fe.grid(p.n)
        A_I = p.matrix().tocsr()[inner][:, inner]
        E = cp.responses[i].H - cl.responses[i].H
        err = np.einsum("ik,ik->k", E, np.asarray(A_I @ E))
        lam = np.diag(cl.grams[i])
        eh.append(math.sqrt(float(err.sum()) / float(lam.sum())))
        ehl.append(GF.e_h_learned(err, lam, LEARNED))
        r = res[i].up + res[i].H @ fm.kernel_coefficients(pb.kind, pb.n, pb.m) - 1.0
        km = fm.kernel_measures(r, A_I, float(lam.sum()))
        ek.append(km["e_K"])
        eki.append(km["e_K_inf"])
    return {"e_H": eh, "e_H_L": ehl, "e_K": ek, "e_K_inf": eki}


def benchmark_problem(name: str, problem: str, experts: dict, main: str) -> dict:
    """One benchmark problem for every run, as registered and lowered by 300 K."""
    t0 = time.perf_counter()
    kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
    b = bm.build(name, BENCH_N, problem, **kw)
    frames = {"registered": b, "lowered": bm.datum_shift(b, GF.DATUM_SHIFT_K)}
    out = {"name": name, "problem": problem, "pieces": len(b.problems), "frames": {}, "runs": {}}
    keep = {}
    for frame, bb in frames.items():
        L = bb.layout(GF.KIND, GF.M_PORTS)
        mono = hs.monolithic(L)
        cl = L.solve(hs.ClassicalExpert())
        un, ud = cl.energy_error(mono.fields)
        mx0, span = ev0.field_errors(mono.fields, cl.fields)
        fr = {"truncation_energy": un, "reference_energy": ud, "truncation_energy_rel": un / ud,
              "floor_max": mx0, "span": span, "refusals": L.check_envelope(data.ENVELOPE), "runs": {}}
        for run, ex in experts.items():
            try:
                cp = L.solve(ex)
            except hs.Refused as r:
                fr["runs"][run] = {"refused": str(r)}
                continue
            num, den = cp.energy_error(mono.fields)
            mx, _ = ev0.field_errors(mono.fields, cp.fields)
            rec = {"energy_abs": num, "energy_rel": num / den, "max_err": mx, "max_over_range": mx / span,
                   "calls": cp.calls, "calls_per_piece": cp.calls / len(L.problems), "min_eig": cp.min_eig()}
            if frame == "registered":
                rec["pieces"] = piece_measures(L, cp, cl, ex)
                if run == main:
                    nt = ev0.network_term(L, cp, cl)
                    rec["network_term"] = nt
                    rec["e3"] = G.judge_e3(num, un, nt)
                    rec["se1"] = G.judge_se1(ev0.gram_identity(L, cp, cl), cp.min_eig(), cl.min_eig())
                    rec["se3"] = G.judge_se3(un, nt)
                    keep = {"layout": L, "coupled": cp}
            fr["runs"][run] = rec
        out["frames"][frame] = fr
    reg, low = out["frames"]["registered"], out["frames"]["lowered"]
    for run, ex in experts.items():
        a, z = reg["runs"].get(run, {}), low["runs"].get(run, {})
        if "energy_rel" not in a or "energy_rel" not in z:
            out["runs"][run] = {"refused": True}
            continue
        out["runs"][run] = {
            "e4": GF.judge_e4_both_frames(
                {"energy_rel": a["energy_rel"], "max_err": a["max_err"], "span": reg["span"],
                 "floor_max": reg["floor_max"]},
                {"energy_rel": z["energy_rel"], "max_err": z["max_err"], "span": low["span"],
                 "floor_max": low["floor_max"]}),
            "d": GF.judge_d(a["energy_abs"], z["energy_abs"], reg["reference_energy"],
                            a["max_err"], z["max_err"], reg["span"]),
            "k": GF.judge_k(a["pieces"]["e_K"], a["pieces"]["e_K_inf"]),
            "l_pieces_met": load_blind(ex, b.problems),
            "se2": {"met": bool(a["calls_per_piece"] == GF.SE2_CALLS_PER_PIECE
                                and z["calls_per_piece"] == GF.SE2_CALLS_PER_PIECE)}}
    if problem == "steady" and keep:
        L, cp = keep["layout"], keep["coupled"]
        out["dual"] = ev0.dual_records(experts[main], b.problems, [L.R[i] @ cp.c for i in range(L.P)])
    out["elapsed_s"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# 6: CW
# ---------------------------------------------------------------------------


def cw_record(bench: list, runs: dict, untrained: str | None = UNTRAINED) -> dict:
    """W over S' on `gates_forms.cw_measure`, seed with seed, with each run's median ``e_H``
    over the benchmark pieces and CW0 (the untrained forms' ratio) beside it."""
    measure, med = {}, {}
    for run in runs:
        er = {(r["name"], r["problem"], frame): r["frames"][frame]["runs"][run]["energy_rel"]
              for r in bench for frame in ("registered", "lowered")}
        measure[run] = GF.cw_measure(er)
        med[run] = float(np.median([e for r in bench for e in r["frames"]["registered"]["runs"][run]["pieces"]["e_H"]]))
    by = {(v["registered"], v["seed"]): k for k, v in runs.items()}
    alt = {s: measure[by[("W", s)]] for s in GF.ARMS["W"]["seeds"]}
    dflt = {s: measure[by[("Sp", s)]] for s in GF.ARMS["Sp"]["seeds"]}
    out = dict(GF.compare_pairings(alt, dflt), measure=measure, e_H_median_benchmark_pieces=med,
               pairs={str(s): [by[("W", s)], by[("Sp", s)]] for s in dflt})
    if untrained and os.path.exists(untrained):
        ju = json.load(open(untrained, encoding="utf-8")).get("judged_untrained", {})
        z = ju.get("split, corner bubble, strips, zero heads", {})
        out["CW0"] = z.get("CW0_W_start_over_this")
    return out


# ---------------------------------------------------------------------------


def dual_test_records(ex: LearnedExpert, seeds: list) -> list:
    out = []
    for s in seeds:
        out += ev0.dual_records(ex, [data.draw(s).problem], None)
    return out


def gates_summary(res: dict, main: str) -> dict:
    """The gates of the main arm, and K, D and L of every run (`gates_forms.GATES`)."""
    bench = res["benchmarks"]
    g = {}
    k_test = res["K_test"]
    per = {}
    for run in res["runs"]:
        rows = [r["runs"][run] for r in bench if "refused" not in r["runs"].get(run, {"refused": 1})]
        per[run] = {"K": bool(len(rows) == len(bench) and all(x["k"]["met"] for x in rows)),
                    "D": bool(len(rows) == len(bench) and all(x["d"]["met"] for x in rows)),
                    "L": bool(len(rows) == len(bench) and all(all(x["l_pieces_met"]) for x in rows)),
                    "E4": bool(len(rows) == len(bench) and all(x["e4"]["met"] for x in rows))}
    per[main]["K"] = per[main]["K"] and k_test["met"]
    per[main]["L"] = per[main]["L"] and res["L_test"]["met"]
    g["K"] = {"met": all(v["K"] for v in per.values()), "runs": {r: v["K"] for r, v in per.items()}}
    g["D"] = {"met": all(v["D"] for v in per.values()), "runs": {r: v["D"] for r, v in per.items()}}
    g["L"] = {"met": all(v["L"] for v in per.values()), "runs": {r: v["L"] for r, v in per.items()}}
    g["E1"] = {"met": res["E1"]["met"]}
    reg = [r["frames"]["registered"]["runs"].get(main, {}) for r in bench]
    for key, sub in (("E3", "e3"), ("SE1", "se1"), ("SE3", "se3")):
        rows = [x[sub] for x in reg if sub in x]
        g[key] = {"met": bool(len(rows) == len(bench) and all(x["met"] for x in rows)), "problems": len(rows)}
    g["E4"] = {"met": per[main]["E4"], "both_frames": True,
               "runs": {r: v["E4"] for r, v in per.items()}}
    g["SE2"] = {"met": all(r["runs"][main]["se2"]["met"] for r in bench)}
    g["DUAL"] = {"met": res["DUAL"]["met"]}
    g["KT"] = {"met": all(v["kt"]["met"] for v in res["KT"].values())}
    g["S11"] = {"reported": True,
                "e4_met_at": [x["step"] for x in res["rollout"]["records"] if x["e4"]["met"]]}
    return g


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=RUNS_DIR)
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--out", default=FORMS)
    ap.add_argument("--test-dir", default=os.path.join(FORMS, "data", "test"))
    ap.add_argument("--test-count", type=int, default=GF.N_TEST)
    ap.add_argument("--l-test", type=int, default=GF.L_TEST_PIECES)
    ap.add_argument("--dual-test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=chartop_cpus.default_workers(2),
                    help="the test split's pool, if it must be drawn")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--untrained", default=UNTRAINED)
    ap.add_argument("--resume", action="store_true")
    return ap


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    plan = json.load(open(args.plan, encoding="utf-8"))
    if plan.get("registration") != "gates_forms":
        raise SystemExit(f"{args.plan} is not a gates_forms plan")
    runs = registered_runs(plan, args.runs)
    hashes = {r: v["sha256"] for r, v in runs.items()}
    identity = {"gates_forms_sha256": module_sha256("gates_forms.py"), "gates_sha256": module_sha256("gates.py"),
                "test_count": args.test_count, "l_test": args.l_test, "dual_test": args.dual_test}
    try:
        ev = registry.begin(args.out, hashes, resume=args.resume, identity=identity,
                            session={"script_sha256": script_sha256(), "torch": torch.__version__,
                                     "torch_threads": args.threads, "platform": platform.platform()},
                            stamp=None if args.resume else "forms-" + time.strftime("%Y%m%d-%H%M%S"))
    except registry.Refused as e:
        raise SystemExit(str(e))
    if ev.resumed:
        log(f"resuming evaluation {ev.stamp}: {len(ev.state['checkpoints'])} checkpoints")
    try:
        return run(args, ev, runs, identity)
    except (Exception, KeyboardInterrupt):
        log(f"stopped: {len(ev.state['checkpoints'])} checkpoints are kept in {ev.dir}; "
            f"run the same command with --resume to continue")
        raise


def run(args, ev, runs: dict, identity: dict) -> int:
    first = ev.state["sessions"][0]
    main_ = next(r for r, v in runs.items() if v["registered"] == "S")
    res = {"written": ev.state["started"], "registration": GF.REGISTERED, "identity": identity,
           "weights": {r: v["sha256"] for r, v in runs.items()},
           "runs": {r: {k: v[k] for k in ("registered", "seed")} for r, v in runs.items()},
           "main": main_, "precision": GF.EVAL_PRECISION, "torch_threads": first["torch_threads"],
           "platform": first["platform"], "torch": first["torch"]}
    nets = {r: tr.load_expert(v["path"], "cpu") for r, v in runs.items()}
    res["configs"] = {r: n.config() for r, n in nets.items()}
    # 2-3: E1 and K on the test split, S only
    ev0.ensure_test(args.test_dir, args.test_count, args.workers)
    try:
        ev.confirm_test_split(registry.npz_content_sha256(ev0.split_files(args.test_dir)))
    except registry.Refused as e:
        raise SystemExit(str(e))
    res["test_sha256"] = ev.state["test"]["sha256"]
    t0 = time.perf_counter()
    e1 = e1_streamed(nets[main_], args.test_dir, ev)
    log(f"E1 rows for {main_}: median e_H,L {np.median(e1['e_H_L']):.4g}, median e_H {np.median(e1['e_H']):.4g}, "
        f"median e_p {np.median(e1['e_p']):.4g} ({time.perf_counter() - t0:.0f} s)")
    res["E1"] = GF.judge_e1(e1["e_H_L"], e1["e_p"], e1["sup_mu"], e1["freq"])
    beside = ev0.summarise({"e_H": e1["e_H"], "e_p": e1["e_p"]}, main_)
    res["E1_beside"] = {"e_H_all_64": beside, "e_H_L": ev0.summarise({"e_H": e1["e_H_L"], "e_p": e1["e_p"]}, main_),
                        "e_K_median": float(np.median(e1["e_K"])), "e_K_max": float(np.max(e1["e_K"])),
                        "trace_gap_max": float(np.max(e1["trace_gap"]))}
    res["E1_by_resolution"] = {str(n): ev0.summarise({"e_H": e1["e_H_L"][e1["n"] == n], "e_p": e1["e_p"][e1["n"] == n]},
                                                     main_) for n in sorted(set(e1["n"].tolist()))}
    res["K_test"] = GF.judge_k(e1["e_K"], e1["e_K_inf"])
    registry.write_atomic(os.path.join(args.out, f"evaluation-{ev.stamp}-test-rows.npz"),
                          ev0._npz_bytes({k: v for k, v in e1.items() if k != "family"}))
    experts = {r: LearnedExpert(n_, envelope=data.ENVELOPE) for r, n_ in nets.items()}
    # 4: L on the first test pieces
    lt = ev0.unit(ev, "l-test.json", lambda: load_blind(experts[main_], test_pieces(args.l_test)))
    res["L_test"] = {"met": bool(lt and all(lt)), "pieces": len(lt)}
    # 5: the benchmarks
    bench, drec = [], []
    for name in GF.HELD_OUT_BENCHMARKS:
        for problem in GF.BENCHMARK_PROBLEMS:
            r = ev0.unit(ev, f"bench-forms-{name}-{problem}.json",
                         lambda name=name, problem=problem: benchmark_problem(name, problem, experts, main_))
            log(f"benchmark {name} {problem}: {main_} energy {r['frames']['registered']['runs'][main_].get('energy_rel')} "
                f"registered, {r['frames']['lowered']['runs'][main_].get('energy_rel')} lowered ({r['elapsed_s']:.0f} s)")
            bench.append(r)
            drec += r.get("dual", [])
    res["benchmarks"] = bench
    # 6: CW
    res["CW"] = cw_record(bench, runs, args.untrained)
    # 7: the dual bracket on steady test pieces
    steady = [int(s) for s, f in zip(e1["seed"], e1["fo"]) if f < 0][:args.dual_test]
    dtest = []
    for i in range(0, len(steady), DUAL_CHUNK):
        chunk = steady[i:i + DUAL_CHUNK]
        dtest += ev0.unit(ev, f"dual-forms-test-{chunk[0]}-{chunk[-1]}.json",
                          lambda chunk=chunk: dual_test_records(experts[main_], chunk))
    allmin = [min(d["min_eig_rel"].values()) for d in drec + dtest]
    res["DUAL"] = {**G.judge_dual(allmin), "benchmark_pieces": len(drec), "test_pieces": len(dtest),
                   "benchmark_records": drec, "test_records": dtest}
    # 8-9: kappa(T) and the rollout
    res["KT"] = {name: ev0.unit(ev, f"kt-forms-{name}.json", lambda name=name: ev0.kt_record(name, experts[main_]))
                 for name in GF.HELD_OUT_BENCHMARKS}
    res["rollout"] = ev0.unit(ev, "rollout-forms.json", lambda: ev0.rollout_record(experts[main_]))
    res["E5"] = {"reported": "priced on the boxes before training; not re-run here",
                 "records": sorted(os.path.basename(p) for p in glob.glob(os.path.join(FORMS, "price-forms-*.json")))}
    res["gates"] = gates_summary(res, main_)
    provenance = ev.provenance()
    if provenance is not None:
        res["resumed"] = provenance
    path = os.path.join(args.out, f"evaluation-{ev.stamp}.json")
    lines = [f"THE REGISTRATION AFTER STAGE 0: EVALUATION {res['written']}  "
             f"(gates_forms sha256 {identity['gates_forms_sha256'][:16]}...)", ""]
    for k, v in res["gates"].items():
        state = "reported" if "reported" in v else ("met" if v["met"] else "NOT MET")
        lines.append(f"  {k:5s} {state}")
    cw = res["CW"]
    lines += ["", f"  CW: {cw['verdict']}; " + ", ".join(
        f"seed {s}: {p['ratio']:.4g} ({p['verdict']})" for s, p in cw["pairings"].items())
        + f"; CW0 {cw.get('CW0')}"]
    registry.write_atomic(os.path.splitext(path)[0] + ".txt", ("\n".join(lines) + "\n").encode("utf-8"))
    registry.write_atomic(path, json.dumps(ev0.strip(res), indent=1).encode("utf-8"))
    ev.complete(path)
    log("wrote " + path)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
