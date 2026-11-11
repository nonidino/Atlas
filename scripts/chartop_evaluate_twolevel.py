"""The registration after the Stage 0 rework (`atlas/chartop/gates_twolevel.py`, hash-pinned with
the owner's approval on 2026-11-10): the one evaluation.  Three stages, as R2's: the accuracy on
the laptop, the price on a rented whole machine with `gates_squares.PRICE_CPU`, the verdicts on
the laptop.

    python scripts/chartop_evaluate_twolevel.py                                 A, on the laptop, on AC
    python scripts/chartop_evaluate_twolevel.py --resume
    python scripts/chartop_price_twolevel_payload.py                            B's payload, A's commit
    python scripts/chartop_evaluate_twolevel.py --verify-payload <tar> --stamp <stamp>
    STAMP=<stamp> bash scripts/chartop_evaluate_twolevel_box.sh                 B, on the box
    python scripts/chartop_evaluate_twolevel.py --judge --stamp <stamp>         C, on the laptop

Every bar is `gates_twolevel`'s, which reads the earlier registrations' from `gates_forms` and
`gates_prior`.  The helpers that measured those gates before are used unchanged: gate L's three
calls, the per-piece measures and the dual bracket's test records (`chartop_evaluate_forms`), the
network term, the Gram identity, the dual bracket, kappa(T) and the rollout (`chartop_evaluate`),
and the pinned price protocol (`chartop_price_twolevel.price_runs`).

**A. The accuracy, on the laptop**, on AC (it refuses to start on battery):
0. **The pin.**  Every file ``out/chartop/registered-twolevel.txt`` records must hash as recorded
   (`chartop_register_twolevel.check_pin`), and ``atlas/`` and ``scripts/`` must be the commit (no
   ``+dirty``), so that stage B runs this code.  It refuses once the one evaluation's record exists.
1. **The weights.**  Both runs of ``out/chartop/plan-twolevel.json`` (TL-s0, TL-s1) must have their
   ``best.pt`` (the checkpoint the validation split chose) with the sha256 recorded at download
   (``out/chartop/runs/twolevel/weights-sha256.txt``), TL's configuration (S's form, ``n_c = 16``,
   corners 8) and, in ``arm.json``, `gates_twolevel.arm_config`'s training options and the plan's
   seed.  S's weights (TL's start, the frozen control) and S''s two runs (`REUSED`, the reference)
   must hash as the registration names them.
2. **The registry** refuses TL's weights if they were evaluated before: the evaluation runs once.
3. **The fresh test split**: seeds 6,000,000 to 6,002,399 with sine-hat labels and each piece's
   corner flags, drawn by the pinned generator unless whole on disk.  ``chartop_data.py`` has no
   split at that base, so it is drawn as the ``test-prior`` split with ``--start 1000000``: the same
   seeds, each drawn with `data.draw` (`draw_args`).  Before any expert reads it, its content sha256
   -- each example's stored arrays, hashed as `chartop_register_twolevel.example_digest` hashes the
   generator's draw (`split_digest`) -- must be the one the registration recorded, or it stops.
4. **The test rows**: TL's seed-0 run on every test piece, its network in float32 on the CPU, the
   objective and the coarse level in float64, the arrays as the training pipeline feeds them
   (`chartop_evaluate_forms.e1_shard`'s float32 storage):
   - ``e_H``, ``e_H,L``, ``e_p``, ``e_K`` and ``e_K,inf`` (E1 reported, K judged);
   - ``rho`` over the 36 learned columns against TL0 (`twolevel.zero_heads_twin`: the same coarse
     level and corner treatment), by column group and for the particular field, with
     `twolevel.rho`'s guard, as the training's validation and Phase 3's probe computed it (NC, the
     ledger); and ``kappa_kill`` of the network's raw contribution;
   - CX on the piece's own enlarged space (`coarse.coarse_exactness`), the modes and, where the
     piece has a load, the particular field.  The operator is the one the expert was given: the
     stored coefficients as the pipeline feeds them, assembled in float64;
   - the handoff's usage controls: the same weights fed the next piece's input channels and scalars
     (a cyclic shift within the shard; every operator, load and corner flag the piece's own), whose
     ``rho`` should fall to 0 or below; and the network frozen at its initial weights, S's (TL's
     start), through the same coarse level.  From S's weights that ``rho`` need not be 0: it says
     how much of TL's comes from its training.
   One checkpoint per shard.
5. **L** for both runs on the first `gates_forms.L_TEST_PIECES` test pieces through the host, with
   CX on them.
6. **The held-out problems** (`benchmarks`' variant ``"held-out"``) at n = 64, as registered and
   lowered by 300 K, through the host: TL-s0 and TL-s1 (`twolevel.host_expert`); TL0 at each ``n_c``
   of `TL0_CURVE` with the corner treatment (U's form without its networks inside
   `coarse.CoarseCorrected`, as the price protocol builds it); U itself; S''s two runs.  Per run, E4
   in both frames; for TL's runs D, K, L, SE2 and CX on every piece in both frames; for TL-s0 also
   E3, SE1, SE3 and the dual bracket on the steady problems; CX of TL0 at 16 beside.  The seen six
   go beside, E4's errors of TL-s0 and TL0 at 16 only.
7. **DUAL** for TL-s0 on the held-out steady pieces and ``--dual-test`` steady test pieces.
8. **KT** and **S11** for TL-s0 on `gates_forms`' problems, as `gates_prior` kept them.  The
   rollout asks for the modes formed once, and `coarse.CoarseCorrected` keeps none.  So the adapter
   (`HostArm`) forms them at every step and checks them and the level field against the first
   step's, bit for bit (the network's modes do not read the load), keeping the first step's where
   they differ, and the record says which.
9. **The ledger's coupled parts**, per held-out problem and frame: the attribution, coarse first
   and then the network (U, TL0 at 16, TL-s0, the 16-mode floor: `twolevel.attribution`); ``n_c*``
   from TL0's curve (`twolevel.equivalent_nc`); and the cost per piece on the laptop (the expert's
   answer and its coarse level, in laptop seconds: not the price).
It writes ``evaluation-twolevel-<time>-accuracy.json`` and the test rows beside it.

**B. The price, on the rented machine** (``--price``): the pinned protocol's `price_runs` on PR's
problems (`chartop_price_twolevel.pr_specs`: the held-out six and R2's sixteen) with TL-s0's
weights, on `PRICE_CPU` only, each run checkpointed whole.  It writes
``evaluation-twolevel-<time>-price.json``.

**C. The verdicts, on the laptop** (``--judge``): the price record must be of A's commit, on
`PRICE_CPU`, every run's BLAS pools held and every run timing TL-s0's weights.  PR is measured only
if every problem has `PRICE_RUNS` counted measurements and the A/A check holds.  Then every gate,
the reported measures and the five predictions (`judge_predictions`, read as its docstring says).
It writes ``evaluation-twolevel-<time>.json`` and ``.txt``, whatever they say, and prints ten lines
or fewer.

``--runs``, ``--plan``, ``--forms-runs``, ``--out``, ``--test-dir``, ``--test-seed-base``,
``--test-count``, ``--l-test``, ``--dual-test``, ``--bench-variant``, ``--bench-names``,
``--bench-n``, ``--skip``, ``--problems``, ``--n``, ``--reps``, ``--max-runs``, ``--any-cpu`` and
``--any-power`` exist for tests on throwaway weights, on pieces and problems that have been read;
the registered run uses none of them, and none may draw the registered test seeds or build the
held-out problems (`refuse_tests_on_registered`).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

import chartop_cpus  # noqa: E402
import chartop_data as CD  # noqa: E402
import chartop_evaluate as ev0  # noqa: E402
import chartop_evaluate_forms as ef  # noqa: E402
import chartop_evaluate_hybrid as EH  # noqa: E402
import chartop_forms_diagnose as fd  # noqa: E402
import chartop_price_squares as PS  # noqa: E402
import chartop_price_twolevel as CPT  # noqa: E402
import chartop_register_twolevel as RT  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks as bm, coarse as co, data, fe, forms as fm, gates as G  # noqa: E402
from atlas.chartop import gates_forms as GF, gates_squares as GS, gates_twolevel as GT  # noqa: E402
from atlas.chartop import host as hs, model as md, ports as pt, train as tr, twolevel as tl  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
RUNS_DIR = os.path.join(OUT, "runs", "twolevel")
FORMS_RUNS = os.path.join(OUT, "runs", "forms")
PLAN = os.path.join(OUT, "plan-twolevel.json")
TEST_DIR = os.path.join(OUT, "data", "test-twolevel")
PREFIX = "twolevel-"
KIND, M = GT.KIND, GT.M_PORTS
BENCH_N = GT.BENCHMARK_N
FRAMES = ("registered", "lowered")
LEARNED = sorted(fm.learned_columns(KIND, M))
GROUPS = tl.learned_groups(KIND, M)
CONFIG_KEYS = ("form", "backbone", "size", "p_size", "kind", "m", "bubble", "strips", "hats", "n_c", "corners")
DUAL_CHUNK = 20
#: the arms on the held-out problems: TL's runs, TL0's curve, U and S''s runs
TL0_KEYS = tuple(f"TL0-{nc}" for nc in GT.TL0_CURVE)
TL0_MAIN = f"TL0-{GT.N_C}"
#: the parts ``--skip`` may leave out of a test run (never the registered run)
SKIPPABLE = ("dual-test", "kt", "rollout", "seen")
#: the payload's top folder (`chartop_price_twolevel_payload.TOP`)
CPT_PAYLOAD_TOP = "chartop"
#: TL0's curve is flat when its error falls by at most this fraction of itself from n_c = 4 to 32
FLAT_CURVE = 1e-9
log = ev0.log


def path_of(out: str, stamp: str, part: str = "") -> str:
    return os.path.join(out, f"evaluation-{stamp}{part}")


def file_hash(path: str) -> str:
    return RT.file_hash(path)


def nan_to_none(xs) -> list:
    return [None if x is None or not np.isfinite(x) else float(x) for x in np.asarray(xs, float)]


def none_to_nan(xs) -> np.ndarray:
    return np.asarray([np.nan if x is None else x for x in xs], float)


# ---------------------------------------------------------------------------
# the guards: test flags, the power, the code
# ---------------------------------------------------------------------------


def test_flags(args) -> list:
    """The test flags this run uses (stage A's and B's)."""
    used = []
    for flag, default in (("runs", RUNS_DIR), ("plan", PLAN), ("forms_runs", FORMS_RUNS), ("out", OUT),
                          ("test_dir", TEST_DIR), ("test_seed_base", GT.TEST_SEED_BASE),
                          ("test_count", GT.N_TEST), ("l_test", GF.L_TEST_PIECES), ("dual_test", 100),
                          ("bench_variant", GT.HELD_OUT_VARIANT), ("bench_names", None), ("bench_n", BENCH_N),
                          ("problems", None), ("n", GT.BENCHMARK_N), ("reps", GS.PRICE_REPS),
                          ("max_runs", GS.PRICE_RUNS_MAX)):
        v = getattr(args, flag, default)
        if v != default and not (isinstance(v, str) and isinstance(default, str)
                                 and os.path.abspath(v) == os.path.abspath(default)):
            used.append("--" + flag.replace("_", "-"))
    for flag in ("skip",):
        if getattr(args, flag, None):
            used.append("--" + flag)
    for flag in ("any_cpu", "any_power"):
        if getattr(args, flag, False):
            used.append("--" + flag.replace("_", "-"))
    return used


def refuse_tests_on_registered(args, stage: str) -> None:
    """The registered test seeds and the held-out problems are read only by the registered run:
    with any test flag, stage A must draw its pieces (the split, gate L's and the dual bracket's,
    all from ``--test-seed-base``) outside the registered seeds and build the seen problems, and
    stage B must price named problems, none of them held out."""
    flags = test_flags(args)
    if not flags:
        return
    if stage == "A":
        lo, hi = GT.TEST_SEEDS
        a = args.test_seed_base
        b = a + max(args.test_count, args.l_test)
        if a < hi and b > lo:
            raise SystemExit(f"test flags ({', '.join(flags)}) may not draw the registered test seeds "
                             f"{lo:,}..{hi - 1:,}")
        if args.bench_variant == GT.HELD_OUT_VARIANT:
            raise SystemExit(f"test flags ({', '.join(flags)}) may not build the held-out problems: "
                             f"use --bench-variant seen")
    else:
        if not args.problems or any(s.startswith("held-out:") for s in args.problems):
            raise SystemExit(f"test flags ({', '.join(flags)}) need --problems, none of them held out")


def on_ac():
    """True on AC, False on battery, None where the machine reports no battery."""
    try:
        import psutil
        b = psutil.sensors_battery()
    except Exception:                                   # noqa: BLE001
        return None
    return None if b is None else bool(b.power_plugged)


# ---------------------------------------------------------------------------
# 1. the weights
# ---------------------------------------------------------------------------


def _manifest(runs_dir: str) -> dict:
    out = {}
    with open(os.path.join(runs_dir, "weights-sha256.txt"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                h, rel = line.split()
                out[rel.lstrip("*").replace("\\", "/").split("/")[0]] = h
    return out


def want_config() -> dict:
    c = {k: GT.COMMON[k] for k in CONFIG_KEYS if k in GT.COMMON}
    c.update(n_c=GT.N_C, corners=GT.CORNERS)
    return c


def registered_runs(plan: dict, runs_dir: str) -> dict:
    """{run: {"path", "sha256", "registered", "seed", ...}} for both runs of the plan, or SystemExit."""
    if plan.get("registration") != "gates_twolevel":
        raise SystemExit("the plan is not a gates_twolevel plan")
    manifest = _manifest(runs_dir)
    cfg_want = want_config()
    arm_want = json.loads(json.dumps(GT.arm_config("TL")))
    out = {}
    for d in plan["arms"]:
        name = d["name"]
        if d.get("registered") != "TL":
            raise SystemExit(f"run {name!r} is not gates_twolevel's TL")
        path = os.path.join(runs_dir, name, "best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"no weights for run {name!r} in {runs_dir}")
        h = registry.sha256_file(path)
        if manifest.get(name) != h:
            raise SystemExit(f"{path}: sha256 {h[:16]}... is not the one recorded at download "
                             f"({str(manifest.get(name))[:16]}...)")
        st = torch.load(path, map_location="cpu", weights_only=False)
        got = {k: st["config"].get(k) for k in CONFIG_KEYS}
        if got != cfg_want:
            raise SystemExit(f"run {name!r}: configuration {got} is not TL's {cfg_want}")
        arm = json.load(open(os.path.join(runs_dir, name, "arm.json"), encoding="utf-8"))["arm"]
        diff = {k: (arm.get(k), v) for k, v in arm_want.items() if k in arm and arm[k] != v}
        missing = sorted(k for k in arm_want if k not in arm and k != "m")
        if diff or missing or arm.get("seed") != d["seed"] or arm.get("seed") not in GT.SEEDS:
            raise SystemExit(f"run {name!r} was not trained as TL: differs {diff}, missing {missing}, "
                             f"seed {arm.get('seed')} (plan {d['seed']})")
        summary = {}
        sp_ = os.path.join(runs_dir, name, "summary.json")
        if os.path.exists(sp_):
            summary = json.load(open(sp_, encoding="utf-8"))
        stopped = None
        lp = os.path.join(runs_dir, name, "log.jsonl")
        if os.path.exists(lp):
            for line in open(lp, encoding="utf-8"):
                if '"stopped"' in line:
                    stopped = json.loads(line)
        out[name] = {"path": path, "sha256": h, "registered": "TL", "seed": int(d["seed"]),
                     "best_step": st.get("step"), "best_val_e_H_median": st.get("val_e_H_median"),
                     "steps_trained": summary.get("steps"), "stopped": stopped}
    seeds = sorted(v["seed"] for v in out.values())
    if seeds != sorted(GT.SEEDS):
        raise SystemExit(f"the plan's runs have seeds {seeds}, not the registration's {sorted(GT.SEEDS)}")
    return out


def named_weights(forms_runs: str) -> dict:
    """S's weights (TL's start) and S''s two runs (`REUSED`), as the registration names them."""
    manifest = _manifest(forms_runs)
    need = {GT.INIT_RUN: GT.INIT_SHA256, **{name: sha for name, sha in GT.REUSED.values()}}
    out = {}
    for name, sha in need.items():
        path = os.path.join(forms_runs, name, "best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"no weights for {name!r} in {forms_runs}")
        h = registry.sha256_file(path)
        if h != sha or manifest.get(name) != sha:
            raise SystemExit(f"{path}: sha256 {h[:16]}... is not the one the registration names ({sha[:16]}...)")
        out[name] = {"path": path, "sha256": h}
    return out


# ---------------------------------------------------------------------------
# 3. the fresh test split
# ---------------------------------------------------------------------------


def draw_args(seed_base: int, count: int) -> list:
    """The pinned generator's arguments for seeds ``seed_base .. seed_base + count - 1``, each drawn
    with `data.draw`: the split (not ``train-conformal``, whose draw differs) with the largest seed
    base at or below ``seed_base``, and ``--start`` the offset from it."""
    plain = {k: v for k, v in CD.SPLIT_SEED.items() if k != "train-conformal"}
    below = [k for k, v in plain.items() if v <= seed_base]
    if not below:
        raise ValueError(f"no split of the generator has its seed base at or below {seed_base}")
    split = max(below, key=lambda k: plain[k])
    return ["--split", split, "--start", str(seed_base - plain[split]), "--count", str(count),
            "--name", "test", "--labels", KIND, "--corners"]


def _side(test_dir: str, prefix: str, shard: str) -> str:
    return os.path.join(test_dir, prefix + os.path.basename(shard))


def split_complete(test_dir: str, seed_base: int, count: int) -> bool:
    """The shards hold exactly seeds ``seed_base .. + count - 1``, each once, each shard with its
    sine-hat labels and its corner flags for the same seeds."""
    shards = ev0.split_shards(test_dir)
    if not shards:
        return False
    seeds = []
    for f in shards:
        lf, cf = _side(test_dir, ev0.label_prefix(KIND), f), _side(test_dir, "corners_", f)
        if not (os.path.exists(lf) and os.path.exists(cf)):
            return False
        with np.load(f, allow_pickle=False) as z, np.load(lf, allow_pickle=False) as zl, \
                np.load(cf, allow_pickle=False) as zc:
            if not (np.array_equal(z["seed"], zl["lseed"]) and np.array_equal(z["seed"], zc["cseed"])):
                return False
            seeds += np.asarray(z["seed"]).tolist()
    return sorted(seeds) == list(range(seed_base, seed_base + count))


def ensure_test(test_dir: str, seed_base: int, count: int, workers: int) -> None:
    if split_complete(test_dir, seed_base, count):
        return
    os.makedirs(test_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "chartop_data.py"), *draw_args(seed_base, count),
           "--workers", str(workers), "--out", test_dir]
    log("drawing the test split: " + " ".join(cmd[2:]))
    subprocess.run(cmd, check=True)
    if not split_complete(test_dir, seed_base, count):
        raise SystemExit(f"{test_dir} does not hold exactly the test pieces (seeds {seed_base}.."
                         f"{seed_base + count - 1}) after drawing them")


def split_files(test_dir: str) -> list:
    out = []
    for f in ev0.split_shards(test_dir):
        out += [f, _side(test_dir, ev0.label_prefix(KIND), f), _side(test_dir, "corners_", f)]
    return out


def stored_digest(arrays: dict, i: int) -> str:
    """`chartop_register_twolevel.example_digest`'s hash of example ``i`` of a shard: its stored
    arrays, the keys of `train.KEYS` in order, each array's dtype, shape and bytes."""
    h = hashlib.sha256()
    for k in tr.KEYS:
        v = np.ascontiguousarray(np.asarray(arrays[k][i]))
        h.update(f"{k}|{v.dtype.str}|{v.shape}|".encode())
        h.update(v.tobytes())
    return h.hexdigest()


def split_digest(test_dir: str) -> dict:
    """The split's content sha256 as the registration took it (`test_split_sha256`): each
    example's digest in seed order, hashed together; from the stored arrays, so it shows the
    shards hold exactly the registered pieces."""
    got = {}
    for f in ev0.split_shards(test_dir):
        with np.load(f, allow_pickle=False) as z:
            arrays = {k: z[k] for k in tr.KEYS}
            seeds = np.asarray(z["seed"]).tolist()
        n = int(arrays["inputs"].shape[-1]) - 1
        for i, s in enumerate(seeds):
            got[int(s)] = (n, stored_digest(arrays, i))
    h = hashlib.sha256()
    counts = {}
    seeds = sorted(got)
    for s in seeds:
        n, d = got[s]
        h.update(f"{s}:{n}:{d}\n".encode())
        counts[n] = counts.get(n, 0) + 1
    return {"seeds": [seeds[0], seeds[-1]] if seeds else [], "count": len(seeds),
            "by_n": {str(k): v for k, v in sorted(counts.items())}, "sha256": h.hexdigest()}


def registered_split_sha256() -> str:
    rec = json.load(open(RT.RECORD, encoding="utf-8"))
    return rec["test_split"]["sha256"]


# ---------------------------------------------------------------------------
# 4. the test rows
# ---------------------------------------------------------------------------


def load_shard(shard: str, test_dir: str) -> tuple:
    """The shard as `train._batch` reads it (`chartop_evaluate_forms.e1_shard`'s float32 storage),
    its labels, its corner flags and its bins; and the coefficients as the expert is given them,
    in float64, for CX."""
    lf, cf = _side(test_dir, ev0.label_prefix(KIND), shard), _side(test_dir, "corners_", shard)
    with np.load(shard, allow_pickle=False) as z, np.load(lf, allow_pickle=False) as zl, \
            np.load(cf, allow_pickle=False) as zc:
        if not (np.array_equal(z["seed"], zl["lseed"]) and np.array_equal(z["seed"], zc["cseed"])):
            raise ValueError("test labels, corner flags and examples disagree on seeds in " + shard)
        t = {k: torch.from_numpy(np.ascontiguousarray(z[k], dtype=np.float32)) for k in tr.KEYS}
        t["H"] = torch.from_numpy(np.ascontiguousarray(zl["H"]))
        t["up"] = torch.from_numpy(np.ascontiguousarray(zl["up"]))
        t["trace_Lambda"] = torch.from_numpy(np.asarray(zl["trace_Lambda"], np.float64))
        t["up_energy"] = torch.from_numpy(np.asarray(zl["up_energy"], np.float64))
        t["corners"] = torch.from_numpy(np.asarray(zc["corners"], bool))
        for k in ("sup_mu", "freq", "seed", "fo", "family"):
            t["meta_" + k] = z[k]
        given = {k: np.asarray(z[k], np.float32).astype(np.float64) for k in ("a11", "a12", "a22", "sigma", "w", "g")}
    return t, given


def space_P(n: int, n_c: int, S):
    """The arm's ``P_e``: the coarse grid's hats, with the corner columns ``S`` beside them."""
    P = co.prolongation(n, n_c)
    if S is not None and S.shape[1]:
        P = sp.hstack([P, S], format="csr")
    return P


def cx_of(M, n: int, P, Q, H, up, b) -> tuple:
    """CX of one piece's answer on ``P``: the modes (``H`` [(n-1)^2, K]) against ``B Q`` and the
    particular field against the load ``b`` (``None`` without a load)."""
    loop, inner = fe.grid(n)
    BQ = -np.asarray(M[inner][:, loop] @ Q)
    cx = co.coarse_exactness(P, co.residual(M, n, H, Q), BQ)["rel"]
    cxp = co.coarse_exactness(P, co.residual(M, n, up, rhs=b), b)["rel"]
    return cx, cxp


ROW_KEYS = ("e_H", "e_H_L", "e_p", "e_K", "e_K_inf", "trace_gap", "eps_L", "eps0", "rho",
            *[f"rho_{g}" for g in GROUPS], "eps_p", "eps_p0", "rho_p", "kappa_kill",
            "rho_shuffled", "rho_p_shuffled", "rho_frozen", "rho_p_frozen", "cx", "cx_p", "extra",
            "non_right", "sup_mu", "freq", "n", "seed", "fo", "family")


def shard_rows(net: tl.TwoLevelExpert, twin: tl.TwoLevelExpert, frozen: tl.TwoLevelExpert, t: dict,
               given: dict) -> dict:
    """Section 4's measures on one shard (see the module's docstring)."""
    rows = {k: [] for k in ROW_KEYS}
    n = int(t["inputs"].shape[-1]) - 1
    ps = md.plain_square(n, KIND, M, "cpu")
    ks = fm.kernel_square(n, KIND, M, "cpu")
    Q = pt.basis(KIND, n, M).Q
    loop, inner = fe.grid(n)
    N = int(t["inputs"].shape[0])
    nxt = torch.roll(torch.arange(N), -1)               # piece i is fed piece i+1's inputs
    shuffled = {"inputs": t["inputs"][nxt], "scalars": t["scalars"][nxt]}
    bs = 16 if n <= 64 else 4
    for e in (net, twin, frozen):
        e.eval()
    with torch.no_grad():
        for i0 in range(0, N, bs):
            idx = torch.arange(i0, min(N, i0 + bs))
            b, lab = tr._batch(t, idx, True)
            ref = tl.reference(twin, b, lab, strip_solve=True, easy=tl.EASY)
            raw0 = twin.last_uncorrected[0]
            H, up = tl.call(net, b, strip_solve=True)
            raw, level = net.last_uncorrected[0], net.last_level
            kap = tl.kappa_kill(net, (raw - raw0)[:, LEARNED])
            o = md.objectives(b, H, up, KIND, M, lab, "label-free")
            coef = (b["a11"], b["a12"], b["a22"], b["sigma"])
            err = md.energy(*coef, md.full(ps, H - lab["H"]) - ps.QB[None]).numpy()
            lam = md.energy(*coef, md.full(ps, lab["H"])).numpy()
            r = fm.residual(H, level, b["s0"], ks)
            lam_t = tl.column_energies(b, H, KIND, M)
            eps_p = tl.particular_energy(b, up) + lab["up_energy"]
            bsh = dict(b, inputs=shuffled["inputs"][idx], scalars=shuffled["scalars"][idx])
            Hs, ups = tl.call(net, bsh, strip_solve=False)
            Hf, upf = tl.call(frozen, b, strip_solve=False)
            tr_ = lab["trace_Lambda"].numpy()
            upE = lab["up_energy"].numpy()
            lamr = ref["lam"]

            def eps_L(Hx):
                return (tl.column_energies(b, Hx, KIND, M) - lamr)[:, LEARNED].sum(-1).numpy()

            eL = (lam_t - lamr)[:, LEARNED].sum(-1).numpy()
            rows["eps_L"] += eL.tolist()
            rows["eps0"] += ref["eps0"].numpy().tolist()
            rows["rho"] += tl.rho(eL, ref["eps0"].numpy(), tr_).tolist()
            for g, cols in GROUPS.items():
                eg = (lam_t - lamr)[:, cols].sum(-1).numpy()
                rows[f"rho_{g}"] += tl.rho(eg, ref["eps0_groups"][g].numpy(), lamr[:, cols].sum(-1).numpy()).tolist()
            rows["eps_p"] += eps_p.numpy().tolist()
            rows["eps_p0"] += ref["eps_p0"].numpy().tolist()
            rows["rho_p"] += tl.rho(eps_p.numpy(), ref["eps_p0"].numpy(), upE).tolist()
            rows["kappa_kill"] += kap.numpy().tolist()
            rows["rho_shuffled"] += tl.rho(eps_L(Hs), ref["eps0"].numpy(), tr_).tolist()
            rows["rho_p_shuffled"] += tl.rho((tl.particular_energy(b, ups) + lab["up_energy"]).numpy(),
                                             ref["eps_p0"].numpy(), upE).tolist()
            rows["rho_frozen"] += tl.rho(eps_L(Hf), ref["eps0"].numpy(), tr_).tolist()
            rows["rho_p_frozen"] += tl.rho((tl.particular_energy(b, upf) + lab["up_energy"]).numpy(),
                                           ref["eps_p0"].numpy(), upE).tolist()
            rows["e_H"] += o["e_H"].tolist()
            rows["e_p"] += o["e_p"].tolist()
            rows["e_H_L"] += [GF.e_h_learned(err[i], lam[i], LEARNED) for i in range(len(idx))]
            rows["e_K"] += fm.kernel_error_torch(b, r, lab["trace_Lambda"]).tolist()
            rows["e_K_inf"] += r.abs().amax(dim=(-2, -1)).tolist()
            rows["trace_gap"] += (np.abs(lam.sum(-1) - tr_) / tr_).tolist()
            for i in range(len(idx)):
                k = int(idx[i])
                flags = t["corners"][k].numpy()
                corners = tuple(c for c in range(4) if flags[c])
                S = co.corner_patch_columns(n, GT.N_C, corners, GT.CORNERS) if corners else None
                Mi = fe.assemble(n, given["a11"][k], given["a12"][k], given["a22"][k], given["sigma"][k]).tocsr()
                bI = (fe.mass(n, given["w"][k]) @ given["g"][k].ravel())[inner]
                cx, cxp = cx_of(Mi, n, space_P(n, GT.N_C, S), Q, H[i].numpy().reshape(H.shape[1], -1).T,
                                up[i].numpy().reshape(-1), bI)
                rows["cx"].append(cx)
                rows["cx_p"].append(np.nan if cxp is None else cxp)
                rows["extra"].append(0 if S is None else int(S.shape[1]))
                rows["non_right"].append(len(corners))
    for k in ("sup_mu", "freq", "seed", "fo", "family"):
        rows[k] += list(t["meta_" + k])
    rows["n"] += [n] * N
    return {k: np.asarray(v) for k, v in rows.items()}


def rows_streamed(net, twin, frozen, run: str, test_dir: str, ev) -> dict:
    parts = []
    for shard in ev0.split_shards(test_dir):
        name = f"rows-{run}-" + os.path.basename(shard)
        blob = ev.load(name)
        if blob is not None:
            log(f"{name}: from the checkpoint")
            parts.append(ev0._npz_load(blob))
            continue
        t0 = time.perf_counter()
        t, given = load_shard(shard, test_dir)
        rows = shard_rows(net, twin, frozen, t, given)
        del t, given
        ev.save(name, ev0._npz_bytes(rows))
        parts.append(rows)
        log(f"{name}: {len(rows['rho'])} pieces, median rho {np.nanmedian(rows['rho']):.4g} "
            f"({time.perf_counter() - t0:.0f} s)")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summarise(x) -> dict:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"median": None, "p10": None, "p90": None, "count": 0}
    return {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "count": int(a.size)}


def rho_ledger(rows: dict) -> dict:
    """The ledger's per-piece part on the test split: rho over the learned columns, by group, of
    the particular field, by n, input frequency, sup|mu| bin and piece type; kappa_kill; the
    usage controls."""
    def by(sel_of, values):
        return {str(k): tl.summarise_rho(np.asarray(rows["rho"], float)[sel]) for k, sel in sel_of(values)}
    mu = np.array([G.mu_bin_of(float(m)) for m in rows["sup_mu"]])
    out = {"rho": tl.summarise_rho(rows["rho"]),
           "groups": {g: tl.summarise_rho(rows[f"rho_{g}"]) for g in GROUPS},
           "particular": tl.summarise_rho(rows["rho_p"]),
           "kappa_kill": summarise(rows["kappa_kill"]),
           "controls": {"shuffled inputs": tl.summarise_rho(rows["rho_shuffled"]),
                        "shuffled inputs, particular": tl.summarise_rho(rows["rho_p_shuffled"]),
                        "frozen at the initial weights (S's)": tl.summarise_rho(rows["rho_frozen"]),
                        "frozen, particular": tl.summarise_rho(rows["rho_p_frozen"])}}
    out["by_n"] = by(lambda v: [(n, v == n) for n in sorted(set(v.tolist()))], np.asarray(rows["n"]))
    out["by_freq"] = by(lambda v: [(f, v == f) for f in sorted(set(v.tolist()))], np.asarray(rows["freq"]))
    out["by_mu_bin"] = {str(list(G.MU_BINS[k])): tl.summarise_rho(np.asarray(rows["rho"], float)[mu == k])
                        for k in range(len(G.MU_BINS))}
    out["by_family"] = by(lambda v: [(f, v == f) for f in sorted(set(v.tolist()))], np.asarray(rows["family"]))
    out["by_non_right_corners"] = by(lambda v: [(c, v == c) for c in sorted(set(v.tolist()))],
                                     np.asarray(rows["non_right"]))
    return out


def test_pieces(seed_base: int, count: int) -> list:
    """The first ``count`` test pieces, drawn again from their seeds."""
    return [data.draw(seed_base + i).problem for i in range(count)]


# ---------------------------------------------------------------------------
# the host: an adapter for the shared helpers
# ---------------------------------------------------------------------------


class HostArm:
    """A host expert of the two-level kind (`twolevel.host_expert`, or TL0's `coarse.CoarseCorrected`)
    as the shared helpers read an expert: ``respond`` (the host's call), ``forward`` and
    ``last_levels`` (gate L's calls and the dual bracket's), ``calls``, ``kind`` and ``envelope``.

    ``modes_from`` (the rollout's): `coarse.CoarseCorrected` keeps no modes, so they are formed again
    at every step and checked against the kept ones, the level field with them, bit for bit.  Where
    they differ, the kept ones are used, as a march that formed them once would; ``kept`` records,
    per step, whether they agreed."""

    name = "two-level"

    def __init__(self, cc):
        self.cc = cc
        self.kind = getattr(cc, "kind", None) or KIND
        self.m = getattr(cc, "m", None) or M
        self.envelope = getattr(cc, "envelope", None)
        self.last_levels = None
        self.kept = []

    @property
    def calls(self) -> int:
        return self.cc.calls

    @property
    def coarse_seconds(self) -> float:
        return self.cc.coarse_seconds

    def pb_of(self, n: int):
        return pt.basis(self.kind, n, self.m)

    def respond(self, problems, pb_of, modes_from=None):
        resp = self.cc.respond(problems, pb_of)
        if modes_from is None:
            return resp
        out, same = [], True
        for r, k in zip(resp, modes_from):
            s = bool(np.array_equal(r.H, k.H) and ((r.level is None and k.level is None)
                                                   or (r.level is not None and k.level is not None
                                                       and np.array_equal(r.level, k.level))))
            same = same and s
            out.append(r if s else hs.Response(k.H, r.up, k.level))
        self.kept.append(same)
        return out

    def forward(self, problems):
        resp = self.cc.respond(problems, self.pb_of)
        self.last_levels = [r.level for r in resp]
        return [(r.H, r.up) for r in resp]


def host_cx(problems, responses, pb_of) -> dict:
    """CX of each piece's answer on its own enlarged space (B, ``n/8``, at ``n_c = 16``)."""
    treat = co.CornerTreatment("B", frac=GT.CORNERS)
    out = {"cx": [], "cx_p": [], "extra": []}
    for p, r in zip(problems, responses):
        n = p.n
        if GT.N_C >= n:
            continue
        M_ = p.matrix().tocsr()
        _loop, inner = fe.grid(n)
        S = treat.of_problem(p, GT.N_C)
        cx, cxp = cx_of(M_, n, space_P(n, GT.N_C, S), pb_of(n).Q, r.H, r.up, p.load()[inner])
        out["cx"].append(cx)
        out["cx_p"].append(cxp)
        out["extra"].append(0 if S is None else int(S.shape[1]))
    return out


def solve_arm(L, ex) -> dict:
    """`host.Layout.solve`, timed: the expert's answer (and its coarse level) per piece."""
    refusals = L.check_envelope(getattr(ex, "envelope", None))
    if refusals:
        return {"refused": "; ".join(refusals)}
    calls0 = getattr(ex, "calls", 0)
    c0 = getattr(ex, "coarse_seconds", None)
    t0 = time.perf_counter()
    resp = ex.respond(L.problems, L.pb)
    secs = time.perf_counter() - t0
    cp = L.couple(resp, calls=getattr(ex, "calls", 0) - calls0, refusals=refusals)
    rec = {"_coupled": cp, "respond_s_per_piece": secs / len(L.problems)}
    if c0 is not None:
        rec["coarse_s_per_piece"] = (ex.coarse_seconds - c0) / len(L.problems)
    return rec


# ---------------------------------------------------------------------------
# 6. the benchmark problems
# ---------------------------------------------------------------------------


def benchmark_problem(name: str, problem: str, arms: dict, main: str, tl_runs: list, variant, n: int,
                      beside: bool = False) -> dict:
    """One problem, as registered and lowered by 300 K, for every arm (section 6)."""
    t0 = time.perf_counter()
    kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
    if variant:
        kw["variant"] = variant
    b = bm.build(name, n, problem, **kw)
    frames = {"registered": b, "lowered": bm.datum_shift(b, GT.DATUM_SHIFT_K)}
    out = {"name": name, "problem": problem, "variant": variant, "n": n, "pieces": len(b.problems),
           "frames": {}, "runs": {}}
    keep = None
    for frame, bb in frames.items():
        L = bb.layout(KIND, M)
        mono = hs.monolithic(L)
        cl = L.solve(hs.ClassicalExpert())
        un, ud = cl.energy_error(mono.fields)
        mx0, span = ev0.field_errors(mono.fields, cl.fields)
        fr = {"truncation_energy": un, "reference_energy": ud, "truncation_energy_rel": un / ud,
              "floor_max": mx0, "span": span, "refusals": L.check_envelope(data.ENVELOPE),
              "non_right": [list(co.non_right_corners(p.chart.X)) for p in L.problems], "runs": {}}
        for key, ex in arms.items():
            rec = solve_arm(L, ex)
            if "refused" in rec:
                fr["runs"][key] = rec
                continue
            cp = rec.pop("_coupled")
            num, den = cp.energy_error(mono.fields)
            mx, _ = ev0.field_errors(mono.fields, cp.fields)
            rec.update(energy_abs=num, energy_rel=num / den, max_err=mx, max_over_range=mx / span,
                       calls=cp.calls, calls_per_piece=cp.calls / len(L.problems), min_eig=cp.min_eig())
            if not beside and (key in tl_runs or key == TL0_MAIN):
                rec["cx"] = host_cx(L.problems, cp.responses, L.pb)
            if not beside and key in tl_runs and frame == "registered":
                rec["pieces"] = ef.piece_measures(L, cp, cl, ex)
                if key == main:
                    nt = ev0.network_term(L, cp, cl)
                    rec["network_term"] = nt
                    rec["e3"] = G.judge_e3(num, un, nt)
                    rec["se1"] = G.judge_se1(ev0.gram_identity(L, cp, cl), cp.min_eig(), cl.min_eig())
                    rec["se3"] = G.judge_se3(un, nt)
                    keep = cp
            fr["runs"][key] = rec
        out["frames"][frame] = fr
    reg, low = out["frames"]["registered"], out["frames"]["lowered"]
    for key, ex in arms.items():
        a, z = reg["runs"].get(key, {}), low["runs"].get(key, {})
        if "energy_rel" not in a or "energy_rel" not in z:
            out["runs"][key] = {"refused": True}
            continue
        r = {"e4": GT.judge_e4_both_frames(
            {"energy_rel": a["energy_rel"], "max_err": a["max_err"], "span": reg["span"], "floor_max": reg["floor_max"]},
            {"energy_rel": z["energy_rel"], "max_err": z["max_err"], "span": low["span"], "floor_max": low["floor_max"]})}
        if not beside and key in tl_runs:
            r.update(d=GT.judge_d(a["energy_abs"], z["energy_abs"], reg["reference_energy"], a["max_err"],
                                  z["max_err"], reg["span"]),
                     k=GT.judge_k(a["pieces"]["e_K"], a["pieces"]["e_K_inf"]),
                     l_pieces_met=ef.load_blind(ex, b.problems),
                     se2={"met": bool(a["calls_per_piece"] == GF.SE2_CALLS_PER_PIECE
                                      and z["calls_per_piece"] == GF.SE2_CALLS_PER_PIECE)})
        out["runs"][key] = r
    if not beside and problem == "steady" and keep is not None:
        Lk = keep.layout
        out["dual"] = ev0.dual_records(arms[main], b.problems, [Lk.R[i] @ keep.c for i in range(Lk.P)])
    out["elapsed_s"] = time.perf_counter() - t0
    return out


def coupled_ledger(held: list, main: str) -> dict:
    """Per held-out problem and frame: the attribution (coarse first, then the network), ``n_c*`` from
    TL0's curve, and the laptop's cost per piece of TL0 at ``n_c*`` against TL's."""
    out = {}
    for r in held:
        for f in FRAMES:
            fr = r["frames"][f]
            e = {k: v.get("energy_rel") for k, v in fr["runs"].items()}
            key = f"{r['name']} {r['problem']} {f}"
            if any(e.get(k) is None for k in ("U", TL0_MAIN, main, *TL0_KEYS)):
                out[key] = {"refused": True}
                continue
            att = tl.attribution(e["U"], e[TL0_MAIN], e[main], fr["truncation_energy_rel"])
            curve = [e[k] for k in TL0_KEYS]
            if (curve[0] - curve[-1]) <= FLAT_CURVE * curve[0]:
                # a curve that falls by rounding only passes equivalent_nc's check and extrapolates
                # to n_c* = 0: there is no coarse size to match, so none is formed
                ncs = {"refused": f"TL0's error falls by at most {FLAT_CURVE:g} of itself from n_c = "
                                  f"{GT.TL0_CURVE[0]} to {GT.TL0_CURVE[-1]}: no n_c* is formed"}
            else:
                try:
                    ncs = tl.equivalent_nc(GT.TL0_CURVE, curve, e[main])
                except ValueError as err:
                    ncs = {"refused": str(err)}
            if "n_c_star" in ncs:
                ncs["over_n_c"] = ncs["n_c_star"] / GT.N_C
                cost = [fr["runs"][k]["respond_s_per_piece"] for k in TL0_KEYS]
                x = np.log(np.asarray(GT.TL0_CURVE, float))
                ncs["laptop_tl0_s_per_piece_at_n_c_star"] = float(np.exp(np.interp(
                    math.log(ncs["n_c_star"]), x, np.log(np.asarray(cost, float)))))
                ncs["laptop_tl_s_per_piece"] = fr["runs"][main]["respond_s_per_piece"]
            out[key] = {"attribution": att, "n_c_star": ncs, "tl0_curve": dict(zip(TL0_KEYS, curve)),
                        "e_TL": e[main], "e_floor": fr["truncation_energy_rel"]}
    return out


# ---------------------------------------------------------------------------
# A. the accuracy
# ---------------------------------------------------------------------------


def stage_accuracy(args) -> int:
    torch.set_num_threads(args.threads)
    flags = test_flags(args)
    refuse_tests_on_registered(args, "A")
    if not args.any_power and on_ac() is False:
        raise SystemExit("stage A runs on AC: the laptop is on battery")
    pins = RT.check_pin(args.registration)
    code = PS.code_identity()
    if not flags and (code is None or code.endswith("+dirty")):
        raise SystemExit(f"atlas/ and scripts/ must be the commit, so that stage B runs this code ({code})")
    done = sorted(glob.glob(os.path.join(glob.escape(args.out), "evaluation-" + PREFIX + "????????-??????.json")))
    if done:
        raise SystemExit(f"{os.path.basename(done[0])} exists: the one evaluation has run")
    plan = json.load(open(args.plan, encoding="utf-8"))
    runs = registered_runs(plan, args.runs)
    named = named_weights(args.forms_runs)
    hashes = {r: v["sha256"] for r, v in runs.items()}
    names = list(args.bench_names or GT.BENCHMARK_NAMES)
    identity = {"pinned": pins, "named": {k: v["sha256"] for k, v in named.items()}, "code": code,
                "test_seed_base": args.test_seed_base, "test_count": args.test_count, "l_test": args.l_test,
                "dual_test": args.dual_test, "bench": [args.bench_variant, names, args.bench_n],
                "skip": sorted(args.skip or []), "test_flags": flags}
    try:
        ev = registry.begin(args.out, hashes, resume=args.resume, identity=identity,
                            session={"script_sha256": file_hash(os.path.abspath(__file__)),
                                     "torch": torch.__version__, "torch_threads": args.threads,
                                     "platform": platform.platform(), "on_ac": on_ac()},
                            stamp=None if args.resume else PREFIX + time.strftime("%Y%m%d-%H%M%S"))
    except registry.Refused as e:
        raise SystemExit(str(e))
    if ev.resumed:
        log(f"resuming evaluation {ev.stamp}: {len(ev.state['checkpoints'])} checkpoints")
    try:
        return run_accuracy(args, ev, runs, named, identity, names)
    except (Exception, KeyboardInterrupt):
        log(f"stopped: {len(ev.state['checkpoints'])} checkpoints are kept in {ev.dir}; "
            f"run the same command with --resume to continue")
        raise


def run_accuracy(args, ev, runs: dict, named: dict, identity: dict, names: list) -> int:
    t_start = time.time()
    first = ev.state["sessions"][0]
    main = next(r for r, v in runs.items() if v["seed"] == 0)
    tl_runs = sorted(runs)
    skip = set(args.skip or [])
    res = {"stage": "A, the accuracy", "written": ev.state["started"], "stamp": ev.stamp,
           "registration": GT.REGISTERED, "identity": identity, "code": identity["code"],
           "registered_run": not identity["test_flags"], "test_flags": identity["test_flags"],
           "weights": {r: v["sha256"] for r, v in runs.items()},
           "runs": {r: {k: v[k] for k in ("seed", "best_step", "best_val_e_H_median", "steps_trained", "stopped")}
                    for r, v in runs.items()},
           "named": identity["named"], "main": main, "torch_threads": first["torch_threads"],
           "platform": first["platform"], "torch": first["torch"]}
    nets = {r: tl.load(v["path"], "cpu") for r, v in runs.items()}
    res["configs"] = {r: e.config() for r, e in nets.items()}
    frozen = tl.from_kernel(tr.load_expert(named[GT.INIT_RUN]["path"], "cpu"), GT.N_C, corners=GT.CORNERS).eval()
    twin = tl.zero_heads_twin(nets[main])
    z = LearnedExpert(fd.without_networks(EH.build_u(), EH.probe_piece()), envelope=data.ENVELOPE)
    treat = co.CornerTreatment("B", frac=GT.CORNERS)
    hosts = {r: HostArm(tl.host_expert(nets[r], envelope=data.ENVELOPE)) for r in tl_runs}
    tl0 = {f"TL0-{nc}": HostArm(co.CoarseCorrected(z, nc, corners=treat)) for nc in GT.TL0_CURVE}
    sprime = {name: LearnedExpert(tr.load_expert(named[name]["path"], "cpu"), envelope=data.ENVELOPE)
              for name, _sha in GT.REUSED.values()}
    # 3: the fresh test split, checked against the registration before any expert reads it
    ensure_test(args.test_dir, args.test_seed_base, args.test_count, args.workers)
    dig = split_digest(args.test_dir)
    want = registered_split_sha256()
    res["test_split"] = dict(dig, registered_sha256=want, matches_registration=dig["sha256"] == want)
    if res["registered_run"] and dig["sha256"] != want:
        raise SystemExit(f"the test split on disk (content sha256 {dig['sha256'][:16]}...) is not the registered "
                         f"one ({want[:16]}...): nothing has read it; draw it again")
    try:
        ev.confirm_test_split(registry.npz_content_sha256(split_files(args.test_dir)))
    except registry.Refused as e:
        raise SystemExit(str(e))
    res["test_files_sha256"] = ev.state["test"]["sha256"]
    log(f"test split: {dig['count']} pieces {dig['by_n']}, content sha256 {dig['sha256'][:16]}... "
        f"(registered: {dig['sha256'] == want})")
    # 4: the test rows, TL's seed-0 run
    t0 = time.perf_counter()
    rows = rows_streamed(nets[main], twin, frozen, main, args.test_dir, ev)
    res["test_rows_s"] = time.perf_counter() - t0
    registry.write_atomic(path_of(args.out, ev.stamp, f"-test-rows-{main}.npz"),
                          ev0._npz_bytes({k: v for k, v in rows.items() if k != "family"}))
    res["E1"] = GT.judge_e1(rows["e_H_L"], rows["e_p"], rows["sup_mu"], rows["freq"])
    res["E1_beside"] = {"e_H_all_64": summarise(rows["e_H"]), "e_H_L": summarise(rows["e_H_L"]),
                        "e_p": summarise(rows["e_p"]), "trace_gap_max": float(np.max(rows["trace_gap"])),
                        "by_n": {str(n): {"e_H_L": summarise(rows["e_H_L"][rows["n"] == n]),
                                          "e_p": summarise(rows["e_p"][rows["n"] == n])}
                                 for n in sorted(set(rows["n"].tolist()))}}
    res["K_test"] = GT.judge_k(rows["e_K"], rows["e_K_inf"])
    res["CX_test"] = GT.judge_cx(rows["cx"], nan_to_none(rows["cx_p"]))
    res["rho_test"] = nan_to_none(rows["rho"])
    res["ledger"] = {"test": rho_ledger(rows)}
    log(f"test rows: median rho {res['ledger']['test']['rho']['median']}, p10 {res['ledger']['test']['rho']['p10']}; "
        f"max CX {res['CX_test']['max']:.3g}; K {res['K_test']['met']} ({res['test_rows_s']:.0f} s)")
    # 5: L on the first test pieces, both runs, with CX on them
    res["L_test"], res["CX_l_test"] = {}, {}
    pieces = test_pieces(args.test_seed_base, args.l_test)
    for r in tl_runs:
        lt = ev0.unit(ev, f"l-test-{r}.json", lambda r=r: ef.load_blind(hosts[r], pieces))
        res["L_test"][r] = {"met": bool(lt and all(lt)), "pieces": len(lt)}
        res["CX_l_test"][r] = ev0.unit(ev, f"cx-l-test-{r}.json",
                                       lambda r=r: host_cx(pieces, hosts[r].cc.respond(pieces, hosts[r].pb_of),
                                                           hosts[r].pb_of))
    # 6: the held-out problems; the seen ones beside
    arms = {**{r: hosts[r] for r in tl_runs}, **tl0, "U": z, **sprime}
    held, seen = [], []
    for name in names:
        for problem in GT.BENCHMARK_PROBLEMS:
            rec = ev0.unit(ev, f"bench-{args.bench_variant}-{name}-{problem}.json",
                           lambda name=name, problem=problem: benchmark_problem(
                               name, problem, arms, main, tl_runs, None if args.bench_variant == "seen"
                               else args.bench_variant, args.bench_n))
            fr = rec["frames"]["registered"]["runs"]
            log(f"{args.bench_variant} {name} {problem}: TL {fr[main].get('energy_rel')}, TL0 "
                f"{fr[TL0_MAIN].get('energy_rel')}, floor {rec['frames']['registered']['truncation_energy_rel']:.4g} "
                f"({rec['elapsed_s']:.0f} s)")
            held.append(rec)
            if "seen" not in skip and args.bench_variant != "seen":
                seen.append(ev0.unit(ev, f"bench-seen-{name}-{problem}.json",
                                     lambda name=name, problem=problem: benchmark_problem(
                                         name, problem, {main: hosts[main], TL0_MAIN: tl0[TL0_MAIN]}, main,
                                         tl_runs, None, args.bench_n, beside=True)))
    res["benchmarks"], res["benchmarks_seen"] = held, seen
    res["ledger"]["coupled"] = coupled_ledger(held, main)
    # 7: the dual bracket, on the held-out steady pieces and steady test pieces
    drec = [d for r in held if r["problem"] == "steady" for d in r.get("dual", [])]
    steady = [int(s) for s, f in zip(rows["seed"], rows["fo"]) if f < 0][:args.dual_test]
    dtest = []
    if "dual-test" not in skip:
        for i in range(0, len(steady), DUAL_CHUNK):
            chunk = steady[i:i + DUAL_CHUNK]
            dtest += ev0.unit(ev, f"dual-test-{main}-{chunk[0]}-{chunk[-1]}.json",
                              lambda chunk=chunk: ef.dual_test_records(hosts[main], chunk))
    allmin = [min(d["min_eig_rel"].values()) for d in drec + dtest]
    res["DUAL"] = {**G.judge_dual(allmin), "benchmark_pieces": len(drec), "test_pieces": len(dtest),
                   "benchmark_records": drec, "test_records": dtest}
    # 8: kappa(T) and the rollout, on gates_forms' problems
    res["KT"] = {} if "kt" in skip else {
        name: ev0.unit(ev, f"kt-{main}-{name}.json", lambda name=name: ev0.kt_record(name, hosts[main]))
        for name in GT.BENCHMARK_NAMES}
    if "rollout" not in skip:
        hosts[main].kept = []
        roll = ev0.unit(ev, f"rollout-{main}.json",
                        lambda: dict(ev0.rollout_record(hosts[main]), modes_kept_bit_for_bit=list(hosts[main].kept)))
        res["rollout"] = roll
    res["power"] = {"on_ac_at_end": on_ac(),
                    "events": EH.power_events(time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
                                              time.strftime("%Y-%m-%dT%H:%M:%S"))}
    provenance = ev.provenance()
    if provenance is not None:
        res["resumed"] = provenance
    path = path_of(args.out, ev.stamp, "-accuracy.json")
    registry.write_atomic(path, json.dumps(ev0.strip(res), indent=1).encode("utf-8"))
    ev.complete(path)
    e4 = [x["runs"][main]["e4"]["met"] if "e4" in x["runs"][main] else False for x in held]
    print(f"TWO-LEVEL A  {main}: E4 on {sum(e4)} of {len(e4)} {args.bench_variant} problems; median test rho "
          f"{res['ledger']['test']['rho']['median']}; max CX {res['CX_test']['max']:.3g}")
    print(f"             wrote {os.path.relpath(path, ROOT)}")
    print("             next:  python scripts/chartop_price_twolevel_payload.py")
    print(f"                    python scripts/chartop_evaluate_twolevel.py --verify-payload "
          f"out/chartop/box/price-twolevel-<commit>.tar.gz --stamp {ev.stamp}")
    print(f"                    on a whole {GS.PRICE_CPU}: STAMP={ev.stamp} bash scripts/chartop_evaluate_twolevel_box.sh")
    return 0


# ---------------------------------------------------------------------------
# the payload, checked against the pin before it goes to the box
# ---------------------------------------------------------------------------


def verify_payload(tar_path: str, stamp: str, out: str = OUT, registration: str = RT.REGISTRATION) -> dict:
    """The payload must be stage A's commit, every pinned file it carries must hash as the pin
    records (the price box runner also as ``run.sh``), and it must carry this script and its box
    driver as they are here."""
    acc = json.load(open(path_of(out, stamp, "-accuracy.json"), encoding="utf-8"))
    pin = open(registration, encoding="ascii").read()
    top = CPT_PAYLOAD_TOP
    with tarfile.open(tar_path) as t:
        names = set(t.getnames())
        read = lambda arc: t.extractfile(arc).read().replace(b"\r\n", b"\n")  # noqa: E731
        identity = read(f"{top}/IDENTITY").decode().strip()
        files = {}
        for rel in RT.every_path():
            if rel.startswith("out/"):
                continue                                  # records stay on the laptop
            m = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", pin)
            arcs = [f"{top}/{rel}"] + ([f"{top}/run.sh"] if rel == RT.PINNED["price box runner"] else [])
            files[rel] = bool(m) and all(a in names and hashlib.sha256(read(a)).hexdigest() == m.group(1) for a in arcs)
        mine = {}
        for rel in (os.path.relpath(os.path.abspath(__file__), ROOT).replace("\\", "/"),
                    "scripts/chartop_evaluate_twolevel_box.sh"):
            here = open(os.path.join(ROOT, *rel.split("/")), "rb").read().replace(b"\r\n", b"\n")
            mine[rel] = f"{top}/{rel}" in names and read(f"{top}/{rel}") == here
    ok = identity == acc["code"] and all(files.values()) and all(mine.values())
    return {"ok": ok, "identity": identity, "stage A's commit": acc["code"], "files": files, "this script": mine}


# ---------------------------------------------------------------------------
# B. the price, on the rented machine
# ---------------------------------------------------------------------------


def specs_of(args) -> list:
    if not args.problems:
        return CPT.pr_specs()
    return [tuple(s.split(":")) for s in args.problems]


def stage_price(args) -> int:
    if not args.stamp:
        raise SystemExit("--price needs --stamp, stage A's")
    if not args.weights or not os.path.exists(args.weights):
        raise SystemExit("--price needs --weights, TL-s0's best.pt")
    refuse_tests_on_registered(args, "B")
    specs = specs_of(args)
    m = PS.machine()
    if not GS.on_price_cpu(m) and not args.any_cpu:
        raise SystemExit(f"the price is measured on {GS.PRICE_CPU} only; this is {m['cpu']}")
    path = path_of(args.out, args.stamp, "-price.json")
    if os.path.exists(path):
        raise SystemExit(f"{os.path.basename(path)} exists: the price stage has run")
    d = path_of(args.out, args.stamp, "-price-checkpoints")
    runs_dir = os.path.join(d, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    t0 = time.perf_counter()
    weights_sha = registry.sha256_file(args.weights)
    pr = CPT.price_runs(specs, args.n, args.reps, runs_dir, args.weights, max_runs=args.max_runs,
                        checkpoint=lambda name, compute: EH.checkpoint(d, name, compute))
    flags = test_flags(args)
    rec = {"stage": "B, the price", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "stamp": args.stamp,
           "registered_run": specs == CPT.pr_specs() and not flags, "test_flags": flags,
           "code": PS.code_identity(), "machine": m, "on_price_cpu": GS.on_price_cpu(m),
           "weights_sha256": weights_sha, "n": args.n, "reps": args.reps,
           "problems": [PS.spec_key(s) for s in specs], **pr, "elapsed_s": time.perf_counter() - t0}
    registry.write_atomic(path, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    t = pr["tally"]
    print(f"TWO-LEVEL B  {len(pr['runs'])} runs on {m['cpu']}; every problem has {GS.PRICE_RUNS} counted: "
          f"{pr['suffice']} (fewest {min(t.values()) if t else '-'})")
    print(f"             A/A: {pr['aa']['holds'] if pr['aa'] else None} (worst "
          f"{'-' if not pr['aa'] or pr['aa']['worst'] is None else round(pr['aa']['worst'], 4)})")
    print(f"             wrote {os.path.relpath(path, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


# ---------------------------------------------------------------------------
# C. the verdicts
# ---------------------------------------------------------------------------


def geomean_of_medians(first: dict, keys: list):
    meds = [float(np.median(first[k])) for k in keys if len(first.get(k, ())) == GS.PRICE_RUNS]
    return float(math.exp(np.mean(np.log(meds)))) if keys and len(meds) == len(keys) else None


def judge_pr(price, acc: dict) -> dict:
    """PR: TL's price against the classical solve (1), H2r (registered 0.306, and measured beside on
    the same runs) and TL0, from stage B's record.  Measured only if the record is A's commit and
    weights, on `PRICE_CPU` with every run's BLAS pools held, every problem has `PRICE_RUNS` counted
    measurements and the A/A check holds.  "Faster than X" only where the geometric means say so."""
    if price is None:
        return {"measured": False, "why": "no price record"}
    main = acc["main"]
    checks = {"A's commit": price.get("code") == acc["code"],
              "on PRICE_CPU": bool(price.get("on_price_cpu")),
              "BLAS pools held in every run": all(GS.blas_threads_hold(r["thread_pools"]) for r in price["runs"]),
              "TL-s0's weights in every run": all(r.get("weights_sha256") == acc["weights"][main] for r in price["runs"]),
              "PR's problems, no test flag": bool(price.get("registered_run"))}
    aa = price.get("aa") or {}
    measured = bool(price.get("suffice") and aa.get("holds") and all(checks.values()))
    keys = list(price["problems"])
    held = [k for k in keys if k.startswith("held-out ")]
    sq = [k for k in keys if k.startswith("squares ")]
    first = price["first_counted"]
    gm = {a: {"all": geomean_of_medians(first[a], keys), "held-out six": geomean_of_medians(first[a], held),
              "R2's sixteen": geomean_of_medians(first[a], sq)} for a in ("TL", "TL0", "H2r")}
    out = {"measured": measured, "checks": checks, "suffice": price.get("suffice"), "aa": aa,
           "runs": len(price["runs"]), "geometric_mean": gm,
           "per_problem_median": {a: {k: float(np.median(v)) for k, v in first[a].items() if v}
                                  for a in ("TL", "TL0", "H2r")}}
    if measured:
        t_all, t_sq = gm["TL"]["all"], gm["TL"]["R2's sixteen"]
        out["faster_than"] = {
            "the classical solve": bool(t_all < 1.0),
            "H2r as registered (0.306, on R2's sixteen)": bool(t_sq is not None and t_sq < GT.H2R_PRICE),
            "H2r measured beside (R2's sixteen)": bool(t_sq is not None and gm["H2r"]["R2's sixteen"] is not None
                                                      and t_sq < gm["H2r"]["R2's sixteen"]),
            "TL0": bool(gm["TL0"]["all"] is not None and t_all < gm["TL0"]["all"])}
    return out


def gates_summary(acc: dict, pr: dict) -> dict:
    """Every gate of `gates_twolevel.GATES` from stage A's record and PR, and the reported measures."""
    main = acc["main"]
    runs = sorted(acc["weights"])
    held = acc["benchmarks"]
    rows = {r: [x["runs"].get(r, {"refused": True}) for x in held] for r in runs}
    ok = {r: all("refused" not in y for y in rows[r]) for r in runs}
    reg = [x["frames"]["registered"]["runs"].get(main, {}) for x in held]
    g = {}
    g["K"] = {"met": bool(acc["K_test"]["met"] and all(ok[r] and all(y["k"]["met"] for y in rows[r]) for r in runs)),
              "test": acc["K_test"]}
    g["D"] = {"met": bool(all(ok[r] and all(y["d"]["met"] for y in rows[r]) for r in runs))}
    g["L"] = {"met": bool(all(ok[r] and all(all(y["l_pieces_met"]) for y in rows[r]) for r in runs)
                          and all(acc["L_test"][r]["met"] for r in runs))}
    for key, sub in (("E3", "e3"), ("SE1", "se1")):
        rr = [x[sub] for x in reg if sub in x]
        g[key] = {"met": bool(len(rr) == len(held) and all(x["met"] for x in rr))}
    g["E4"] = {"met": bool(ok[main] and all(y["e4"]["met"] for y in rows[main])), "both_frames": True,
               "per_problem": {f"{x['name']} {x['problem']}": bool(y.get("e4", {}).get("met", False))
                               for x, y in zip(held, rows[main])}}
    cx_modes, cx_part = [], []
    for r in runs:
        for x in held:
            for f in FRAMES:
                c = x["frames"][f]["runs"].get(r, {}).get("cx")
                if c:
                    cx_modes += c["cx"]
                    cx_part += c["cx_p"]
        cl = acc["CX_l_test"][r]
        cx_modes += cl["cx"]
        cx_part += cl["cx_p"]
    held_cx = GT.judge_cx(cx_modes, cx_part)
    test_cx = acc["CX_test"]
    g["CX"] = {"met": bool(test_cx["met"] and held_cx["met"]), "max": max(test_cx["max"], held_cx["max"]),
               "test pieces (TL-s0)": test_cx, "held-out and L pieces (both runs)": held_cx}
    e4_tl = {(x["name"], x["problem"], f): x["frames"][f]["runs"][main]["energy_rel"] for x in held for f in FRAMES
             if "energy_rel" in x["frames"][f]["runs"].get(main, {})}
    e4_tl0 = {(x["name"], x["problem"], f): x["frames"][f]["runs"][TL0_MAIN]["energy_rel"] for x in held
              for f in FRAMES if "energy_rel" in x["frames"][f]["runs"].get(TL0_MAIN, {})}
    try:
        nc = GT.judge_nc(none_to_nan(acc["rho_test"]), e4_tl, e4_tl0)
        nc["worse_than_TL0"] = [list(k) for k in nc["worse_than_TL0"]]
    except ValueError as err:
        nc = {"met": False, "refused": str(err)}
    g["NC"] = nc
    g["SE2"] = {"met": bool(ok[main] and all(y["se2"]["met"] for y in rows[main]))}
    g["DUAL"] = {"met": bool(acc["DUAL"]["met"]), "benchmark_pieces": acc["DUAL"]["benchmark_pieces"],
                 "test_pieces": acc["DUAL"]["test_pieces"]}
    g["KT"] = {"met": bool(acc["KT"] and all(v["kt"]["met"] for v in acc["KT"].values())),
               "problems": {k: v["kt"] for k, v in acc["KT"].items()}}
    g["PR"] = pr
    rep = {"E1": acc["E1"], "SE3": {f"{h['name']} {h['problem']}": r_.get("se3") for h, r_ in zip(held, reg)}}
    roll = acc.get("rollout")
    rep["S11"] = None if roll is None else {
        "e4_met_at": [x["step"] for x in roll["records"] if x["e4"]["met"]],
        "records": roll["records"], "network_calls": roll["network_calls"], "pieces": roll["pieces"],
        "modes_kept_bit_for_bit": roll.get("modes_kept_bit_for_bit")}
    rep["LEDGER"] = acc["ledger"]
    return {"gates": g, "reported": rep}


def judge_predictions(gates: dict, acc: dict) -> dict:
    """The five predictions of `gates_twolevel.PREDICTIONS`, read as follows (written into this
    script before the one evaluation ran).
    1. Held if CX, K, D and L are each met on every run where this script measures them: CX on
       TL-s0's test pieces and both runs' L pieces and held-out pieces in both frames; K on TL-s0's
       test pieces and both runs' held-out pieces; D and L on both runs.
    2. Held if NC is met, every clause: TL-s0's median test rho at least 0.3, its 10th percentile at
       least 0, and on no held-out problem, in either frame, an energy error above TL0's by more than
       the rounding allowance.  The median is quoted beside.
    3. Held if TL-s0 meets E4 on each of the six held-out problems, in both frames.
    4. Held if, over the test pieces with something to learn in each group, the median rho of the
       hats is below both the median of j = 1-3 and that of j = 4-8.
    5. Held if TL-s0's price, the geometric mean over PR's problems of each problem's median of its
       first PRICE_RUNS counted prices, is at least 1.  Not judged if PR was not measured."""
    g = gates["gates"]
    out = {"words": list(GT.PREDICTIONS)}
    out["1"] = {"held": bool(g["CX"]["met"] and g["K"]["met"] and g["D"]["met"] and g["L"]["met"]),
                "CX": g["CX"]["met"], "K": g["K"]["met"], "D": g["D"]["met"], "L": g["L"]["met"]}
    out["2"] = {"held": bool(g["NC"]["met"]), "median": g["NC"].get("median"), "p10": g["NC"].get("p10")}
    out["3"] = {"held": bool(g["E4"]["met"]), "per_problem": g["E4"]["per_problem"]}
    grp = acc["ledger"]["test"]["groups"]
    h, a, b = grp["hats"]["median"], grp["j1-3"]["median"], grp["j4-8"]["median"]
    out["4"] = {"held": bool(None not in (h, a, b) and h < a and h < b),
                "medians": {"hats": h, "j1-3": a, "j4-8": b}}
    pr = g["PR"]
    if pr.get("measured"):
        gm = pr["geometric_mean"]["TL"]["all"]
        out["5"] = {"held": bool(gm is not None and gm >= 1.0), "geometric_mean": gm}
    else:
        out["5"] = {"held": None, "why": "PR was not measured: " + str(pr.get("why") or pr.get("checks"))}
    return out


def stage_judge(args) -> int:
    if not args.stamp:
        raise SystemExit("--judge needs --stamp, stage A's")
    acc_path = path_of(args.out, args.stamp, "-accuracy.json")
    acc = json.loads(registry.read_bytes(acc_path).decode("utf-8"))
    price_path = path_of(args.out, args.stamp, "-price.json")
    price = json.loads(registry.read_bytes(price_path).decode("utf-8")) if os.path.exists(price_path) else None
    path = path_of(args.out, args.stamp, ".json")
    if os.path.exists(path):
        raise SystemExit(f"{os.path.basename(path)} exists: the verdicts are written")
    pr = judge_pr(price, acc)
    gs = gates_summary(acc, pr)
    preds = judge_predictions(gs, acc)
    rec = {"stage": "C, the verdicts", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "stamp": args.stamp,
           "registration": GT.REGISTERED, "registered_run": bool(acc["registered_run"] and (price or {}).get("registered_run")),
           "accuracy_record": {"file": os.path.basename(acc_path), "sha256": registry.sha256_file(acc_path)},
           "price_record": None if price is None else {"file": os.path.basename(price_path),
                                                        "sha256": registry.sha256_file(price_path)},
           "main": acc["main"], "weights": acc["weights"], "runs": acc["runs"], "code": acc["code"],
           **gs, "predictions": preds}
    registry.write_atomic(path, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    g = gs["gates"]
    met = [k for k, v in g.items() if k != "PR" and v.get("met")]
    missed = [k for k, v in g.items() if k != "PR" and not v.get("met")]
    pr_line = ("PR not measured" if not pr.get("measured") else
               "PR " + ", ".join(f"{a} {pr['geometric_mean'][a]['all']:.3f}" for a in ("TL", "TL0", "H2r")))
    nc = g["NC"]
    lines = [f"TWO-LEVEL C  {acc['main']}: met {', '.join(met) or '-'}; not met {', '.join(missed) or '-'}",
             f"             NC median rho {nc.get('median')}, p10 {nc.get('p10')}; E4 "
             f"{sum(g['E4']['per_problem'].values())} of {len(g['E4']['per_problem'])}; max CX {g['CX']['max']:.3g}",
             f"             {pr_line}",
             "             predictions held: " + ", ".join(f"{k} {preds[k]['held']}" for k in ("1", "2", "3", "4", "5")),
             f"             wrote {os.path.relpath(path, ROOT)}"]
    registry.write_atomic(os.path.splitext(path)[0] + ".txt", ("\n".join(lines) + "\n").encode("utf-8"))
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--price", action="store_true", help="stage B, on the rented machine")
    ap.add_argument("--judge", action="store_true", help="stage C, on the laptop")
    ap.add_argument("--verify-payload", default=None, metavar="TAR")
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--weights", default=None, help="stage B: TL-s0's best.pt")
    ap.add_argument("--registration", default=RT.REGISTRATION)
    ap.add_argument("--runs", default=RUNS_DIR)
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--forms-runs", default=FORMS_RUNS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--test-dir", default=TEST_DIR)
    ap.add_argument("--test-seed-base", type=int, default=GT.TEST_SEED_BASE)
    ap.add_argument("--test-count", type=int, default=GT.N_TEST)
    ap.add_argument("--l-test", type=int, default=GF.L_TEST_PIECES)
    ap.add_argument("--dual-test", type=int, default=100)
    ap.add_argument("--bench-variant", default=GT.HELD_OUT_VARIANT, choices=(GT.HELD_OUT_VARIANT, "seen"))
    ap.add_argument("--bench-names", nargs="+", default=None, choices=GT.BENCHMARK_NAMES)
    ap.add_argument("--bench-n", type=int, default=BENCH_N)
    ap.add_argument("--skip", nargs="+", default=None, choices=SKIPPABLE)
    ap.add_argument("--problems", nargs="+", default=None, help="stage B: kind:what:problem, for tests")
    ap.add_argument("--n", type=int, default=GT.BENCHMARK_N)
    ap.add_argument("--reps", type=int, default=GS.PRICE_REPS)
    ap.add_argument("--max-runs", type=int, default=GS.PRICE_RUNS_MAX)
    ap.add_argument("--any-cpu", action="store_true")
    ap.add_argument("--any-power", action="store_true")
    ap.add_argument("--workers", type=int, default=chartop_cpus.default_workers(2))
    ap.add_argument("--threads", type=int, default=8)
    return ap


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_payload:
        if not args.stamp:
            raise SystemExit("--verify-payload needs --stamp, stage A's")
        v = verify_payload(args.verify_payload, args.stamp, args.out, args.registration)
        bad = [k for k, ok in v["files"].items() if not ok] + [k for k, ok in v["this script"].items() if not ok]
        commit = str(v["stage A's commit"])
        print(f"payload {'OK' if v['ok'] else 'REFUSED'}: identity {v['identity'][:12]} (stage A's "
              f"{commit[:12]})" + (f"; differs: {bad}" if bad else ""))
        return 0 if v["ok"] else 1
    if args.price:
        return stage_price(args)
    if args.judge:
        return stage_judge(args)
    return stage_accuracy(args)


if __name__ == "__main__":
    raise SystemExit(main())
