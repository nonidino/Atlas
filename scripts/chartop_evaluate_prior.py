"""The registration after the forms' evaluation (`atlas/chartop/gates_prior.py`, pinned): the
one evaluation.

    python scripts/chartop_evaluate_prior.py [--workers 16]
    python scripts/chartop_evaluate_prior.py --resume

Every measure and bar is `gates_prior`'s, which reads them all from `gates_forms`.  The
pieces of `scripts/chartop_evaluate_forms.py` that the two registrations share are used
from it unchanged: E1's rows, gate L's three calls, the per-piece measures and the dual
bracket's test records.  What it does, in order:

0. **The pin.**  Every file `out/chartop/registered-prior.txt` pins (`gates_prior`,
   `gates_forms`, `gates`, `benchmarks` and the classical record of the held-out problems)
   must hash as recorded, or it refuses.
1. **The weights.**  Every run of ``out/chartop/plan-prior.json`` must have its ``best.pt``
   with the sha256 recorded at download (``out/chartop/runs/prior/weights-sha256.txt``), the
   configuration of its registered arm (`gates_prior.arm_config`: form, backbone, networks,
   ports, bubble, strips, hats) and, in its ``arm.json``, the arm's training options (the
   prior, the strips skipped).  S''s two runs, which CP reuses (`gates_prior.REUSED`), must
   hash as the registration names them.  Otherwise it refuses.
2. **The registry.**  It refuses the four runs' weights if they were evaluated before.  S''s
   were evaluated once already (the forms' evaluation) and the registration reuses them by
   name, so they are not registered again: they are recorded beside, as reused.
3. **The fresh test split:** `gates_prior.N_TEST` pieces at seeds `gates_prior.TEST_SEEDS`,
   sine-hat labels, drawn (``chartop_data.py --split test-prior --name test``) unless whole
   on disk; its sha256 is taken.
4. **E1** and **K** for the seed-0 run of each arm (P-s0, PG-s0) on the fresh split.
5. **L** for every run on the first `gates_prior.L_TEST_PIECES` test pieces.
6. **The six held-out problems** (`benchmarks`' variant ``"held-out"``) at n = 64, as
   registered and lowered by 300 K, for the four runs and S''s two: **E4** in both frames,
   **D**, **K** and **L** on every piece, and every piece's ``e_H`` and ``e_H,L``; for the
   seed-0 runs also **E3, SE1, SE3**, and **DUAL** on the steady problems' pieces.
   **The six seen problems** go beside, E4's energy error only, in both frames.
7. **CP**, P over S', and **CG**, PG over P, seed with seed (`gates_prior.compare_pairings`)
   on `gates_prior.cw_measure` over the held-out problems; the seen problems' ratios beside.
8. **DUAL** for the seed-0 runs on ``--dual-test`` steady test pieces.
9. **KT** and **S11** for the seed-0 runs, on `gates_forms`' problems (section 11 read neither).
10. It writes ``evaluation-prior-<time>.json`` and ``.txt``, whatever they say.

E5 was priced on the boxes before training; the evaluation names those records and does
not price.  Resuming is `chartop_evaluate`'s: every unit is a checkpoint written whole, and
``--resume`` continues the same weights, identity and test split, or refuses.  ``--out``,
``--runs``, ``--plan``, ``--forms-runs``, ``--test-dir``, ``--test-count`` and
``--registration`` exist for tests on throwaway weights; the registered run uses none of them.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_cpus  # noqa: E402
import chartop_evaluate as ev0  # noqa: E402
import chartop_evaluate_forms as ef  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks as bm, data, gates as G, gates_prior as GP, host as hs  # noqa: E402
from atlas.chartop import train as tr  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
RUNS_DIR = os.path.join(OUT, "runs", "prior")
FORMS_RUNS = os.path.join(OUT, "runs", "forms")
PLAN = os.path.join(OUT, "plan-prior.json")
REGISTRATION = os.path.join(OUT, "registered-prior.txt")
PINNED = ("atlas/chartop/gates_prior.py", "atlas/chartop/gates_forms.py", "atlas/chartop/gates.py",
          "atlas/chartop/benchmarks.py", "out/chartop/registration-prior-heldout.json")
BENCH_N = GP.BENCHMARK_N
DUAL_CHUNK = 20
CONFIG_KEYS = ("form", "backbone", "size", "p_size", "kind", "m", "bubble", "strips", "hats")
TRAINING_KEYS = ("mode_prior", "strips_in_training", "steps", "seed")
log = ev0.log


def file_hash(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def check_pin(path: str = REGISTRATION) -> dict:
    """{file: sha256} for every pinned file, or SystemExit if one is missing or changed."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: the registration is not pinned")
    text = open(path, encoding="ascii").read()
    out = {}
    for rel in PINNED:
        m = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", text)
        if not m:
            raise SystemExit(f"{path} does not pin {rel}")
        h = file_hash(os.path.join(ROOT, rel))
        if h != m.group(1):
            raise SystemExit(f"{rel} is not the registered file: sha256 {h[:16]}... against {m.group(1)[:16]}...")
        out[rel] = h
    return out


def _manifest(runs_dir: str) -> dict:
    out = {}
    with open(os.path.join(runs_dir, "weights-sha256.txt"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                h, rel = line.split()
                out[rel.lstrip("*").split("/")[0]] = h
    return out


def registered_runs(plan: dict, runs_dir: str) -> dict:
    """{run: {"path", "sha256", "registered", "seed"}} for every run of the plan, or SystemExit."""
    if plan.get("registration") != "gates_prior":
        raise SystemExit("the plan is not a gates_prior plan")
    manifest = _manifest(runs_dir)
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
        want = GP.arm_config(reg)
        cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
        got = {k: cfg.get(k) for k in CONFIG_KEYS}
        if got != {k: want[k] for k in CONFIG_KEYS}:
            raise SystemExit(f"run {name!r}: configuration {got} is not registered arm {reg!r}'s")
        arm = json.load(open(os.path.join(runs_dir, name, "arm.json"), encoding="utf-8"))["arm"]
        trained = {k: arm.get(k) for k in TRAINING_KEYS}
        if (trained["mode_prior"], trained["strips_in_training"], trained["steps"]) != (
                want["mode_prior"], want["strips_in_training"], want["steps"]) \
                or trained["seed"] not in GP.ARMS[reg]["seeds"] or trained["seed"] != d["seed"]:
            raise SystemExit(f"run {name!r} was not trained as registered arm {reg!r}: {trained}")
        out[name] = {"path": path, "sha256": h, "registered": reg, "seed": int(d["seed"])}
    seen = sorted((v["registered"], v["seed"]) for v in out.values())
    if seen != sorted((k, s) for k, r in GP.ARMS.items() for s in r["seeds"]):
        raise SystemExit(f"the plan's runs {seen} are not the registration's")
    return out


def reused_runs(forms_runs: str) -> dict:
    """S''s two runs as `gates_prior.REUSED` names them, or SystemExit."""
    manifest = _manifest(forms_runs)
    out = {}
    for seed, (name, sha) in GP.REUSED.items():
        path = os.path.join(forms_runs, name, "best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"no weights for the reused run {name!r} in {forms_runs}")
        h = registry.sha256_file(path)
        if h != sha or manifest.get(name) != sha:
            raise SystemExit(f"{path}: sha256 {h[:16]}... is not the reused run the registration names "
                             f"({sha[:16]}...)")
        out[name] = {"path": path, "sha256": h, "registered": "Sp", "seed": int(seed)}
    return out


# ---------------------------------------------------------------------------
# the fresh test split
# ---------------------------------------------------------------------------


def split_complete(test_dir: str, count: int) -> bool:
    """The shards hold exactly seeds TEST_SEED_BASE .. + count - 1, each once, and each shard
    has its sine-hat labels for the same seeds."""
    shards = ev0.split_shards(test_dir)
    if not shards:
        return False
    seeds = []
    for f in shards:
        lf = os.path.join(test_dir, ev0.label_prefix(GP.KIND) + os.path.basename(f))
        if not os.path.exists(lf):
            return False
        with np.load(f, allow_pickle=False) as z, np.load(lf, allow_pickle=False) as zl:
            if not np.array_equal(z["seed"], zl["lseed"]):
                return False
            seeds += np.asarray(z["seed"]).tolist()
    return sorted(seeds) == list(range(GP.TEST_SEED_BASE, GP.TEST_SEED_BASE + count))


def ensure_test(test_dir: str, count: int, workers: int) -> None:
    if split_complete(test_dir, count):
        return
    os.makedirs(test_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "chartop_data.py"), "--split", "test-prior",
           "--name", "test", "--count", str(count), "--workers", str(workers), "--labels", GP.KIND,
           "--out", test_dir]
    log("drawing the fresh test split: " + " ".join(cmd[1:]))
    subprocess.run(cmd, check=True)
    if not split_complete(test_dir, count):
        raise SystemExit(f"{test_dir} does not hold exactly the {count} registered test pieces (seeds "
                         f"{GP.TEST_SEEDS[0]}..{GP.TEST_SEEDS[0] + count - 1}) after drawing them")


def split_files(test_dir: str) -> list:
    out = []
    for f in ev0.split_shards(test_dir):
        out += [f, os.path.join(test_dir, ev0.label_prefix(GP.KIND) + os.path.basename(f))]
    return out


def e1_streamed(expert, run: str, test_dir: str, ev=None) -> dict:
    """`chartop_evaluate_forms.e1_rows` over the split, one checkpoint per shard and run."""
    parts = []
    for shard in ev0.split_shards(test_dir):
        name = f"e1-{run}-" + os.path.basename(shard)
        blob = ev.load(name) if ev is not None else None
        if blob is not None:
            parts.append(ev0._npz_load(blob))
            continue
        lf = os.path.join(test_dir, ev0.label_prefix(GP.KIND) + os.path.basename(shard))
        with np.load(shard, allow_pickle=False) as z, np.load(lf, allow_pickle=False) as zl:
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
        rows = ef.e1_rows(expert, {n: t})
        if ev is not None:
            ev.save(name, ev0._npz_bytes({k: v for k, v in rows.items()}))
        parts.append(rows)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def test_pieces(count: int) -> list:
    """The first ``count`` pieces of the fresh split, drawn again from their seeds."""
    return [data.draw(GP.TEST_SEED_BASE + i).problem for i in range(count)]


# ---------------------------------------------------------------------------
# the benchmark problems
# ---------------------------------------------------------------------------


def benchmark_problem(name: str, problem: str, experts: dict, mains: set,
                      variant: str | None = GP.HELD_OUT_VARIANT) -> dict:
    """One problem for every run, as registered and lowered by 300 K.  On the held-out
    variant, every gate's measure; on the seen problems (``variant=None``), E4's energy
    error alone, which the comparisons' beside-ratios need."""
    t0 = time.perf_counter()
    kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
    if variant:
        kw["variant"] = variant
    b = bm.build(name, BENCH_N, problem, **kw)
    frames = {"registered": b, "lowered": bm.datum_shift(b, GP.DATUM_SHIFT_K)}
    full = variant is not None
    out = {"name": name, "problem": problem, "variant": variant, "pieces": len(b.problems),
           "frames": {}, "runs": {}}
    keep = {}
    for frame, bb in frames.items():
        L = bb.layout(GP.KIND, GP.M_PORTS)
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
            if full and frame == "registered":
                rec["pieces"] = ef.piece_measures(L, cp, cl, ex)
                if run in mains:
                    nt = ev0.network_term(L, cp, cl)
                    rec["network_term"] = nt
                    rec["e3"] = G.judge_e3(num, un, nt)
                    rec["se1"] = G.judge_se1(ev0.gram_identity(L, cp, cl), cp.min_eig(), cl.min_eig())
                    rec["se3"] = G.judge_se3(un, nt)
                    keep[run] = cp
            fr["runs"][run] = rec
        out["frames"][frame] = fr
    reg, low = out["frames"]["registered"], out["frames"]["lowered"]
    for run, ex in experts.items():
        a, z = reg["runs"].get(run, {}), low["runs"].get(run, {})
        if "energy_rel" not in a or "energy_rel" not in z:
            out["runs"][run] = {"refused": True}
            continue
        r = {"e4": GP.judge_e4_both_frames(
            {"energy_rel": a["energy_rel"], "max_err": a["max_err"], "span": reg["span"],
             "floor_max": reg["floor_max"]},
            {"energy_rel": z["energy_rel"], "max_err": z["max_err"], "span": low["span"],
             "floor_max": low["floor_max"]})}
        if full:
            r.update(d=GP.judge_d(a["energy_abs"], z["energy_abs"], reg["reference_energy"],
                                  a["max_err"], z["max_err"], reg["span"]),
                     k=GP.judge_k(a["pieces"]["e_K"], a["pieces"]["e_K_inf"]),
                     l_pieces_met=ef.load_blind(ex, b.problems),
                     se2={"met": bool(a["calls_per_piece"] == GP.SE2_CALLS_PER_PIECE
                                      and z["calls_per_piece"] == GP.SE2_CALLS_PER_PIECE)})
        out["runs"][run] = r
    if full and problem == "steady":
        out["dual"] = {}
        for run, cp in keep.items():
            L = cp.layout
            out["dual"][run] = ev0.dual_records(experts[run], b.problems, [L.R[i] @ cp.c for i in range(L.P)])
    out["elapsed_s"] = time.perf_counter() - t0
    return out


def comparisons(held: list, seen: list, mains: dict, runs: dict) -> dict:
    """CP (P over S') and CG (PG over P), seed with seed, on the held-out problems, with the
    seen problems' ratios beside."""
    def measure(records, run):
        er = {(r["name"], r["problem"], f): r["frames"][f]["runs"][run]["energy_rel"]
              for r in records for f in ("registered", "lowered")}
        return GP.cw_measure(er)
    by = {(v["registered"], v["seed"]): k for k, v in runs.items()}
    out = {}
    for key, alt, dflt in (("CP", "P", "Sp"), ("CG", "PG", "P")):
        seeds = GP.ARMS[alt]["seeds"]
        a = {s: measure(held, by[(alt, s)]) for s in seeds}
        d = {s: measure(held, by[(dflt, s)]) for s in seeds}
        a_seen = {s: measure(seen, by[(alt, s)]) for s in seeds}
        d_seen = {s: measure(seen, by[(dflt, s)]) for s in seeds}
        out[key] = dict(GP.compare_pairings(a, d), measure_alternative=a, measure_default=d,
                        pairs={str(s): [by[(alt, s)], by[(dflt, s)]] for s in seeds},
                        seen_beside={str(s): a_seen[s] / d_seen[s] for s in seeds})
    return out


def gates_summary(res: dict, mains: dict, runs: dict) -> dict:
    """Per arm: every gate on its seed-0 run, and K, D and L on both of its runs."""
    held = res["benchmarks"]
    out = {}
    for arm, main in mains.items():
        mine = [r for r, v in runs.items() if v["registered"] == arm]
        rows = {r: [x["runs"][r] for x in held] for r in mine}
        ok = {r: all("refused" not in x for x in rows[r]) for r in mine}
        g = {"K": {"met": bool(all(ok[r] and all(x["k"]["met"] for x in rows[r]) for r in mine)
                               and res["K_test"][main]["met"])},
             "D": {"met": bool(all(ok[r] and all(x["d"]["met"] for x in rows[r]) for r in mine))},
             "L": {"met": bool(all(ok[r] and all(all(x["l_pieces_met"]) for x in rows[r]) for r in mine)
                               and all(res["L_test"][r]["met"] for r in mine))},
             "E1": {"met": res["E1"][main]["met"]},
             "E4": {"met": bool(ok[main] and all(x["e4"]["met"] for x in rows[main])), "both_frames": True,
                    "per_problem": {f"{x['name']} {x['problem']}": bool(y.get("e4", {}).get("met", False))
                                    for x, y in zip(held, rows[main])}},
             "SE2": {"met": bool(ok[main] and all(x["se2"]["met"] for x in rows[main]))}}
        reg = [x["frames"]["registered"]["runs"].get(main, {}) for x in held]
        for key, sub in (("E3", "e3"), ("SE1", "se1"), ("SE3", "se3")):
            rr = [x[sub] for x in reg if sub in x]
            g[key] = {"met": bool(len(rr) == len(held) and all(x["met"] for x in rr))}
        g["DUAL"] = {"met": res["DUAL"][main]["met"]}
        g["KT"] = {"met": all(v["kt"]["met"] for v in res["KT"][main].values())}
        g["S11"] = {"reported": True,
                    "e4_met_at": [x["step"] for x in res["rollout"][main]["records"] if x["e4"]["met"]]}
        out[arm] = g
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=RUNS_DIR)
    ap.add_argument("--forms-runs", default=FORMS_RUNS)
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--registration", default=REGISTRATION)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--test-dir", default=os.path.join(OUT, "data", "test-prior"))
    ap.add_argument("--test-count", type=int, default=GP.N_TEST)
    ap.add_argument("--l-test", type=int, default=GP.L_TEST_PIECES)
    ap.add_argument("--dual-test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=chartop_cpus.default_workers(2))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    return ap


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    pins = check_pin(args.registration)
    plan = json.load(open(args.plan, encoding="utf-8"))
    runs = registered_runs(plan, args.runs)
    reused = reused_runs(args.forms_runs)
    hashes = {r: v["sha256"] for r, v in runs.items()}
    identity = {"pinned": pins, "reused": {r: v["sha256"] for r, v in reused.items()},
                "test_count": args.test_count, "l_test": args.l_test, "dual_test": args.dual_test}
    try:
        ev = registry.begin(args.out, hashes, resume=args.resume, identity=identity,
                            session={"script_sha256": file_hash(os.path.abspath(__file__)),
                                     "torch": torch.__version__, "torch_threads": args.threads,
                                     "platform": platform.platform()},
                            stamp=None if args.resume else "prior-" + time.strftime("%Y%m%d-%H%M%S"))
    except registry.Refused as e:
        raise SystemExit(str(e))
    if ev.resumed:
        log(f"resuming evaluation {ev.stamp}: {len(ev.state['checkpoints'])} checkpoints")
    try:
        return run(args, ev, runs, reused, identity)
    except (Exception, KeyboardInterrupt):
        log(f"stopped: {len(ev.state['checkpoints'])} checkpoints are kept in {ev.dir}; "
            f"run the same command with --resume to continue")
        raise


def run(args, ev, runs: dict, reused: dict, identity: dict) -> int:
    first = ev.state["sessions"][0]
    mains = {arm: next(r for r, v in runs.items() if v["registered"] == arm and v["seed"] == 0)
             for arm in GP.ARMS}
    res = {"written": ev.state["started"], "registration": GP.REGISTERED, "identity": identity,
           "weights": {r: v["sha256"] for r, v in runs.items()},
           "reused": {r: v["sha256"] for r, v in reused.items()},
           "runs": {r: {k: v[k] for k in ("registered", "seed")} for r, v in {**runs, **reused}.items()},
           "mains": mains, "precision": GP.EVAL_PRECISION, "torch_threads": first["torch_threads"],
           "platform": first["platform"], "torch": first["torch"]}
    nets = {r: tr.load_expert(v["path"], "cpu") for r, v in {**runs, **reused}.items()}
    res["configs"] = {r: n.config() for r, n in nets.items()}
    # 3-4: the fresh split, E1 and K for each arm's seed-0 run
    ensure_test(args.test_dir, args.test_count, args.workers)
    try:
        ev.confirm_test_split(registry.npz_content_sha256(split_files(args.test_dir)))
    except registry.Refused as e:
        raise SystemExit(str(e))
    res["test_sha256"] = ev.state["test"]["sha256"]
    res["E1"], res["E1_beside"], res["E1_by_resolution"], res["K_test"] = {}, {}, {}, {}
    e1 = {}
    for arm, main in mains.items():
        t0 = time.perf_counter()
        e = e1_streamed(nets[main], main, args.test_dir, ev)
        e1[main] = e
        log(f"E1 rows for {main}: median e_H,L {np.median(e['e_H_L']):.4g}, median e_p "
            f"{np.median(e['e_p']):.4g} ({time.perf_counter() - t0:.0f} s)")
        res["E1"][main] = GP.judge_e1(e["e_H_L"], e["e_p"], e["sup_mu"], e["freq"])
        res["E1_beside"][main] = {"e_H_all_64": ev0.summarise({"e_H": e["e_H"], "e_p": e["e_p"]}, main),
                                  "e_K_median": float(np.median(e["e_K"])), "e_K_max": float(np.max(e["e_K"]))}
        res["E1_by_resolution"][main] = {
            str(n): ev0.summarise({"e_H": e["e_H_L"][e["n"] == n], "e_p": e["e_p"][e["n"] == n]}, main)
            for n in sorted(set(e["n"].tolist()))}
        res["K_test"][main] = GP.judge_k(e["e_K"], e["e_K_inf"])
        registry.write_atomic(os.path.join(args.out, f"evaluation-{ev.stamp}-test-rows-{main}.npz"),
                              ev0._npz_bytes({k: v for k, v in e.items() if k != "family"}))
    experts = {r: LearnedExpert(n_, envelope=data.ENVELOPE) for r, n_ in nets.items()}
    ours = {r: experts[r] for r in runs}
    # 5: L for every run on the first test pieces
    res["L_test"] = {}
    for r in runs:
        lt = ev0.unit(ev, f"l-test-{r}.json", lambda r=r: ef.load_blind(experts[r], test_pieces(args.l_test)))
        res["L_test"][r] = {"met": bool(lt and all(lt)), "pieces": len(lt)}
    # 6: the held-out problems, every run; the seen problems beside, E4's energy error only
    held, seen = [], []
    for name in GP.BENCHMARK_NAMES:
        for problem in GP.BENCHMARK_PROBLEMS:
            r = ev0.unit(ev, f"bench-held-out-{name}-{problem}.json",
                         lambda name=name, problem=problem: benchmark_problem(name, problem, experts,
                                                                              set(mains.values())))
            log(f"held-out {name} {problem}: " + ", ".join(
                f"{m} {r['frames']['registered']['runs'][m].get('energy_rel')}" for m in mains.values()))
            held.append(r)
            s = ev0.unit(ev, f"bench-seen-{name}-{problem}.json",
                         lambda name=name, problem=problem: benchmark_problem(name, problem, experts,
                                                                              set(), variant=None))
            seen.append(s)
    res["benchmarks"], res["benchmarks_seen"] = held, seen
    # 7: CP and CG
    res.update(comparisons(held, seen, mains, {**runs, **reused}))
    # 8: the dual bracket, on the held-out steady pieces and steady test pieces, per seed-0 run
    res["DUAL"] = {}
    for arm, main in mains.items():
        drec = [d for r in held if r["problem"] == "steady" for d in r.get("dual", {}).get(main, [])]
        steady = [int(s) for s, f in zip(e1[main]["seed"], e1[main]["fo"]) if f < 0][:args.dual_test]
        dtest = []
        for i in range(0, len(steady), DUAL_CHUNK):
            chunk = steady[i:i + DUAL_CHUNK]
            dtest += ev0.unit(ev, f"dual-test-{main}-{chunk[0]}-{chunk[-1]}.json",
                              lambda chunk=chunk, main=main: ef.dual_test_records(experts[main], chunk))
        allmin = [min(d["min_eig_rel"].values()) for d in drec + dtest]
        res["DUAL"][main] = {**G.judge_dual(allmin), "benchmark_pieces": len(drec), "test_pieces": len(dtest),
                             "benchmark_records": drec, "test_records": dtest}
    # 9: kappa(T) and the rollout, on gates_forms' problems
    res["KT"], res["rollout"] = {}, {}
    for arm, main in mains.items():
        res["KT"][main] = {name: ev0.unit(ev, f"kt-{main}-{name}.json",
                                          lambda name=name, main=main: ev0.kt_record(name, experts[main]))
                           for name in GP.BENCHMARK_NAMES}
        res["rollout"][main] = ev0.unit(ev, f"rollout-{main}.json",
                                        lambda main=main: ev0.rollout_record(experts[main]))
    res["E5"] = {"reported": "priced on the boxes before training; not re-run here",
                 "records": sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "price-prior-*.json")))}
    res["gates"] = gates_summary(res, mains, runs)
    provenance = ev.provenance()
    if provenance is not None:
        res["resumed"] = provenance
    path = os.path.join(args.out, f"evaluation-{ev.stamp}.json")
    lines = [f"THE REGISTRATION AFTER THE FORMS' EVALUATION: EVALUATION {res['written']}  "
             f"(gates_prior sha256 {identity['pinned']['atlas/chartop/gates_prior.py'][:16]}...)", ""]
    for arm, g in res["gates"].items():
        lines.append(f"  arm {arm} ({mains[arm]}):")
        for k, v in g.items():
            state = "reported" if "reported" in v else ("met" if v["met"] else "NOT MET")
            lines.append(f"    {k:5s} {state}")
    for key in ("CP", "CG"):
        c = res[key]
        lines.append(f"  {key}: {c['verdict']}; " + ", ".join(
            f"seed {s}: {p['ratio']:.4g} ({p['verdict']})" for s, p in c["pairings"].items())
            + "; on the seen problems beside: " + ", ".join(f"{v:.4g}" for v in c["seen_beside"].values()))
    registry.write_atomic(os.path.splitext(path)[0] + ".txt", ("\n".join(lines) + "\n").encode("utf-8"))
    registry.write_atomic(path, json.dumps(ev0.strip(res), indent=1).encode("utf-8"))
    ev.complete(path)
    log("wrote " + path)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
