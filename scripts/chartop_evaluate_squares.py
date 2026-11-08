"""The registration after section 11.17 (`atlas/chartop/gates_squares.py`, hash-pinned): the
one evaluation.  It has three stages: the accuracy and its control run on the laptop, and the
price on a rented machine of `PRICE_CPU` (`gates_squares.REVISED`).

    python scripts/chartop_evaluate_squares.py                          A, on the laptop
    python scripts/chartop_evaluate_squares.py --resume
    python scripts/chartop_price_box_payload.py                         the pinned payload
    python scripts/chartop_evaluate_squares.py --verify-payload <tar> --stamp <stamp>
    STAMP=<stamp> bash scripts/chartop_evaluate_squares_box.sh          B, on the box
    python scripts/chartop_evaluate_squares.py --judge --stamp <stamp>   C, on the laptop

Every measure and bar is `gates_squares`', which reads E4's from `gates_forms`.

**A. The accuracy, on the laptop.**
0. **The pin.**  Every file ``out/chartop/registered-squares.txt`` records must hash as
   recorded (`chartop_register_squares.check_pin`), and ``atlas/`` and ``scripts/`` must be
   the commit, so that the price stage runs the same code; or it refuses.  It refuses, too,
   once the one evaluation's record exists.
1. **The arms**, built as section 11.17 built them (`chartop_evaluate_hybrid.make_experts`):
   U from `gates_squares.U_FORM`; H2's stand-in for U's networks, which must answer U's bits
   on every piece of every problem in both frames; P-s0, its weights' sha256 checked; and H2r,
   `reuse.ReuseExpert` on `is_square`.
2. **The control.**  On section 11.17's twelve problems, which have been read, at the
   registered resolution, the floor and U's and P-s0's energy errors in both frames must equal
   section 11.17's record bit for bit (`CONTROL_RECORD`).  H2's must equal U's where R2 hands U
   every piece (heated-strip), and the floor where it hands the classical solve every piece.
   Otherwise it stops.
3. **The picks.**  On each new problem, `rule_pieces2` must pick the pieces the classical
   record holds, or it stops.
4. **Accuracy, computed once**, on each new problem, as registered and lowered by 300 K: the
   full-domain reference, the classical floor, and the coupled answers of H2, H2r, U on every
   piece and P-s0 (`chartop_evaluate_hybrid.coupled`), each judged by `judge_e4_both_frames`;
   then E4-H2 (`judge_e4h2`).  The accuracy runs at `PRICE_THREADS` torch threads and the
   laptop's default BLAS threads, as section 11.17's did; only the price runs fix the BLAS.
   It writes ``evaluation-squares-<stamp>-accuracy.json``.

**B. The price, on the rented machine** (``--price``).  The payload is built from the same
commit and checked against the pin first (``--verify-payload``).  On `PRICE_CPU` only, the
pinned protocol's `price_runs` prices the sixteen new problems with `AA_CONTROLS` beside, each
run checkpointed whole.  It writes ``evaluation-squares-<stamp>-price.json``.

**C. The verdicts, on the laptop** (``--judge``).  The price record must be of the accuracy's
commit and on `PRICE_CPU`, with every run's BLAS pools held.  PR2 (`judge_pr2`) is judged only
if the A/A controls hold and every new problem has `PRICE_RUNS` counted measurements.  Beside
it: H2r's E4 and price, and the ratio of H2r's price to H2's.  Then the predictions
(`judge_predictions`, read as its docstring says) and the outcome.  It writes
``evaluation-squares-<stamp>.json`` and ``.txt``, whatever they say, and prints ten lines or
fewer.

``--problems``, ``--control``, ``--n``, ``--reps``, ``--any-cpu``, ``--registration`` and
``--out`` exist for tests, on problems the registration never draws or that have been read;
the registered run uses none of them, and none of them may touch a registered layout
(`refuse_tests_on_registered`).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_evaluate as ev0  # noqa: E402
import chartop_evaluate_hybrid as EH  # noqa: E402
import chartop_price_squares as PS  # noqa: E402
import chartop_register_prior as RP  # noqa: E402
import chartop_register_squares as RS  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks_workbench as BW, gates_squares as GS  # noqa: E402
from atlas.chartop.reuse import ReuseExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
STAMP_PREFIX = "evaluation-squares-"
FRAMES = EH.FRAMES
SQUARES, WORKBENCH = "squares", "workbench"
#: section 11.17's one evaluation, which the control reproduces
CONTROL_RECORD = "out/chartop/evaluation-hybrid-20261107-231105.json"
CONTROL = tuple((name, problem) for name in BW.NAMES for problem in BW.PROBLEMS)
#: prediction 4's round features: the layouts with a ring
ROUND_FEATURES = ("hole", "round-inclusion")
REGISTERED_SPECS = [(SQUARES, str(seed), problem) for seed, problem in GS.new_problems()]
log = ev0.log


def spec_key(spec: tuple) -> str:
    return PS.spec_key(spec)


def path_of(out: str, stamp: str, part: str = "") -> str:
    return os.path.join(out, f"{STAMP_PREFIX}{stamp}{part}")


# ---------------------------------------------------------------------------
# the guards: the one evaluation's record, the test flags, the begin
# ---------------------------------------------------------------------------


def test_flags(args) -> list:
    """The test flags this run uses."""
    used = []
    if args.problems:
        used.append("--problems")
    if args.control:
        used.append("--control")
    if args.n != GS.BENCHMARK_N:
        used.append("--n")
    if args.reps != GS.PRICE_REPS:
        used.append("--reps")
    if args.any_cpu:
        used.append("--any-cpu")
    return used


def refuse_tests_on_registered(args, specs: list) -> None:
    """The registered layouts are read only by the registered run: no test flag may touch one."""
    registered = [spec_key(s) for s in specs if s[0] == SQUARES and int(s[1]) in GS.SEEDS]
    if registered and test_flags(args):
        raise SystemExit(f"the registered layouts are read only by the registered run, with no test flag "
                         f"({', '.join(test_flags(args))}): {registered[:3]}")


def begin(out: str, resume: bool) -> tuple:
    """(stamp, checkpoint directory).  Refuses once the one evaluation's record exists, or while
    an unfinished stage A exists and ``resume`` is not set; resumes the newest unfinished one."""
    done = sorted(glob.glob(os.path.join(out, STAMP_PREFIX + "????????-??????.json")))
    if done:
        raise SystemExit(f"the one evaluation has run: {os.path.basename(done[-1])}")
    open_ = sorted(glob.glob(os.path.join(out, STAMP_PREFIX + "????????-??????-checkpoints")))
    if open_ and not resume:
        raise SystemExit(f"an unfinished evaluation exists ({os.path.basename(open_[-1])}); run with --resume")
    if resume:
        if not open_:
            raise SystemExit("nothing to resume")
        d = open_[-1]
        return os.path.basename(d)[len(STAMP_PREFIX):-len("-checkpoints")], d
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = path_of(out, stamp, "-checkpoints")
    os.makedirs(d)
    return stamp, d


# ---------------------------------------------------------------------------
# A. the arms, the control, the picks and the accuracy
# ---------------------------------------------------------------------------


def make_experts() -> dict:
    """Section 11.17's arms (U, its stand-in Z, P-s0 with its weights' sha256), and H2r's expert."""
    ex = EH.make_experts()
    ex["H2r"] = ReuseExpert(GS.is_square)
    return ex


def control(ex: dict, specs=None) -> dict:
    """Section 11.17's problems (or ``specs``) at the registered resolution: the floor and U's and
    P-s0's energy errors in both frames against its record, bit for bit; H2's against U's where
    R2 hands U every piece, and against the floor where it hands the classical solve every piece."""
    rec = json.load(open(os.path.join(ROOT, *CONTROL_RECORD.split("/")), encoding="utf-8"))["accuracy"]
    out = {}
    for name, problem in (specs or CONTROL):
        b = PS.build_problem((WORKBENCH, name, problem), GS.BENCHMARK_N)
        L = b.layout(GS.KIND, GS.M_PORTS)
        picks = GS.rule_pieces2(L)
        res = EH.coupled(b, {"H": (ex["Z"], picks), "U": (ex["U"], []), "P-s0": (ex["P-s0"], [])})
        want, got = rec[f"{name} {problem}"]["frames"], res["frames"]
        same = all(got[f]["floor"] == want[f]["floor"]
                   and got[f]["arms"]["U"]["energy_rel"] == want[f]["arms"]["U"]["energy_rel"]
                   and got[f]["arms"]["P-s0"]["energy_rel"] == want[f]["arms"]["P-s0"]["energy_rel"] for f in FRAMES)
        every_u, every_classical = len(picks) == 0, len(picks) == L.P
        ref = {f: got[f]["arms"]["U"]["energy_rel"] if every_u else got[f]["floor"] for f in FRAMES}
        h2 = ((every_u and res["H answers U's bits"]) or every_classical) and all(
            got[f]["arms"]["H"]["energy_rel"] == ref[f] for f in FRAMES)
        out[f"{name} {problem}"] = {"pieces": L.P, "classical pieces": len(picks),
                                    "equals section 11.17's": bool(same), "H2 equals U or the floor": bool(h2)}
    return out


def expected_picks(spec: tuple):
    """The picks the classical record holds for a new problem; None for any other."""
    if spec[0] != SQUARES:
        return None
    rec = json.load(open(RS.RECORD, encoding="utf-8"))
    hit = [lay["picks"] for lay in rec["layouts"] if lay["seed"] == int(spec[1])]
    return hit[0] if hit else None


def picks_of(spec: tuple, b) -> list:
    """`rule_pieces2` on ``b``, refused unless it is what the classical record holds."""
    picks = GS.rule_pieces2(b.layout(GS.KIND, GS.M_PORTS))
    want = expected_picks(spec)
    if want is not None and picks != want:
        raise SystemExit(f"{spec_key(spec)}: rule R2 picks {picks}, the classical record holds {want}")
    return picks


def rename(d: dict) -> dict:
    return {("H2" if k == "H" else k): v for k, v in d.items()}


def accuracy(spec: tuple, ex: dict, n: int) -> dict:
    """H2 (the stand-in, the classical solve on R2's picks), H2r (reuse, on the picks), U on every
    piece and P-s0 on every piece, coupled, in both frames (`chartop_evaluate_hybrid.coupled`)."""
    b = PS.build_problem(spec, n)
    picks = picks_of(spec, b)
    res = EH.coupled(b, {"H": (ex["Z"], picks), "H2r": (ex["H2r"], picks), "U": (ex["U"], []),
                         "P-s0": (ex["P-s0"], [])})
    if not res["H answers U's bits"]:
        raise SystemExit(f"{spec_key(spec)}: H2's stand-in does not answer U's bits")
    for f in FRAMES:
        res["frames"][f]["arms"] = rename(res["frames"][f]["arms"])
    return {"frames": res["frames"], "verdicts": rename(res["verdicts"]), "H2 answers U's bits": True,
            "pieces": len(b.problems), "picks": picks}


def stage_accuracy(args) -> int:
    t0 = time.perf_counter()
    specs = specs_of(args)
    refuse_tests_on_registered(args, specs)
    pins = RS.check_pin(args.registration)
    code = PS.code_identity()
    registered_run = specs == REGISTERED_SPECS and not test_flags(args)
    if registered_run and (code is None or code.endswith("+dirty")):
        raise SystemExit(f"atlas/ and scripts/ must be the commit, so that the price stage runs the same code ({code})")
    stamp, d = begin(args.out, args.resume)
    if os.path.exists(path_of(args.out, stamp, "-accuracy.json")):
        raise SystemExit(f"stage A has run ({stamp}); the price stage is next")
    torch.set_num_threads(GS.PRICE_THREADS)
    log(f"evaluation {stamp}: the pin holds ({len(pins)} files), commit {str(code)[:7]}")
    ex = make_experts()
    ctl_specs = [tuple(s.split(":")) for s in args.control] if args.control else None
    ctl = EH.checkpoint(d, "control", lambda: control(ex, ctl_specs))
    bad = [k for k, v in ctl.items() if not (v["equals section 11.17's"] and v["H2 equals U or the floor"])]
    if bad:
        raise SystemExit(f"the control does not reproduce section 11.17 on {bad}; checkpoints kept in {d}")
    log(f"the control reproduces section 11.17 bit for bit on {len(ctl)} problems")
    acc = {}
    for spec in specs:
        key = spec_key(spec)
        acc[key] = EH.checkpoint(d, "accuracy-" + key.replace(" ", "-"), lambda spec=spec: accuracy(spec, ex, args.n))
        log(f"{key}: H2 {100 * acc[key]['frames']['registered']['arms']['H2']['energy_rel']:.4f}% / "
            f"{100 * acc[key]['frames']['lowered']['arms']['H2']['energy_rel']:.4f}%")
    verdicts = {(int(s[1]), s[2]): acc[spec_key(s)]["verdicts"]["H2"] for s in specs if s[0] == SQUARES}
    e4h2 = (GS.judge_e4h2(verdicts) if registered_run else
            {"met": all(v["met"] for v in verdicts.values()),
             "count": f"{sum(v['met'] for v in verdicts.values())} of {len(verdicts)}",
             "note": "not the registered problems"})
    rec = {"stage": "A, the accuracy", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "stamp": stamp,
           "registration": GS.REGISTERED, "registered_run": registered_run, "test_flags": test_flags(args),
           "pins": pins, "code": code, "script_sha256": RP.file_hash(os.path.abspath(__file__)),
           "torch": torch.__version__, "torch_threads": GS.PRICE_THREADS, "machine": PS.machine(),
           "P-s0 sha256": ex["P-s0 sha256"], "control_record": CONTROL_RECORD,
           "control_record_sha256": RP.file_hash(os.path.join(ROOT, *CONTROL_RECORD.split("/"))),
           "n": args.n, "control": ctl, "accuracy": acc, "E4-H2": e4h2, "elapsed_s": time.perf_counter() - t0}
    path = path_of(args.out, stamp, "-accuracy.json")
    registry.write_atomic(path, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    worst = max(acc[k]["frames"][f]["arms"]["H2"]["energy_rel"] - acc[k]["frames"][f]["floor"] for k in acc for f in FRAMES)
    c7 = str(code)[:7]
    print(f"SQUARES A  the pin holds ({len(pins)} files); the control reproduces section 11.17 on {len(ctl)} problems")
    print(f"           E4-H2 {'met' if e4h2['met'] else 'not met'} ({e4h2['count']}); H2's largest excess over "
          f"the floor {worst:.2e}")
    print(f"           wrote {os.path.relpath(path, ROOT)} ({rec['elapsed_s']:.0f} s), commit {c7}")
    print("           next:  python scripts/chartop_price_box_payload.py")
    print(f"                  python scripts/chartop_evaluate_squares.py --verify-payload "
          f"out/chartop/box/price-squares-{c7}.tar.gz --stamp {stamp}")
    print(f"                  on a whole {GS.PRICE_CPU}: STAMP={stamp} bash scripts/chartop_evaluate_squares_box.sh")
    print(f"                  then fetch {os.path.basename(path_of(args.out, stamp, '-price.json'))} and run "
          f"--judge --stamp {stamp}")
    return 0


# ---------------------------------------------------------------------------
# the payload, checked against the pin before it goes to the box
# ---------------------------------------------------------------------------


def verify_payload(tar_path: str, stamp: str, out: str = OUT, registration: str = RS.REGISTRATION) -> dict:
    """The payload must be stage A's commit, every pinned file it carries must hash as the pin
    records, and it must carry this script as it is here."""
    acc = json.load(open(path_of(out, stamp, "-accuracy.json"), encoding="utf-8"))
    pin = open(registration, encoding="ascii").read()
    top = "chartop"
    with tarfile.open(tar_path) as t:
        names = set(t.getnames())
        read = lambda arc: t.extractfile(arc).read().replace(b"\r\n", b"\n")  # noqa: E731
        identity = read(f"{top}/IDENTITY").decode().strip()
        files = {}
        for rel in RS.every_path():
            arc = f"{top}/run.sh" if rel == RS.PINNED["box runner"] else f"{top}/{rel}"
            if rel.startswith("out/"):
                continue                                  # records stay on the laptop
            m = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", pin)
            files[rel] = bool(m) and arc in names and hashlib.sha256(read(arc)).hexdigest() == m.group(1)
        here = open(os.path.abspath(__file__), "rb").read().replace(b"\r\n", b"\n")
        script = f"{top}/scripts/{os.path.basename(__file__)}"
        same_script = script in names and read(script) == here
        box = f"{top}/scripts/chartop_evaluate_squares_box.sh"
        has_box = box in names
    ok = identity == acc["code"] and all(files.values()) and same_script and has_box
    return {"ok": ok, "identity": identity, "stage A's commit": acc["code"], "files": files,
            "this script": same_script, "the box driver": has_box}


# ---------------------------------------------------------------------------
# B. the price, on the rented machine
# ---------------------------------------------------------------------------


def stage_price(args) -> int:
    if not args.stamp:
        raise SystemExit("--price needs --stamp, stage A's")
    specs = specs_of(args)
    refuse_tests_on_registered(args, specs)
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
    pr = PS.price_runs(specs, args.n, args.reps, runs_dir,
                       checkpoint=lambda name, compute: EH.checkpoint(d, name, compute))
    rec = {"stage": "B, the price", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "stamp": args.stamp,
           "registered_run": specs == REGISTERED_SPECS and not test_flags(args), "test_flags": test_flags(args),
           "code": PS.code_identity(), "machine": m, "on_price_cpu": GS.on_price_cpu(m),
           "n": args.n, "reps": args.reps, "problems": [spec_key(s) for s in specs],
           "controls": [spec_key(c) for c in GS.AA_CONTROLS], **pr, "elapsed_s": time.perf_counter() - t0}
    registry.write_atomic(path, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    t = pr["tally"]
    print(f"SQUARES B  {len(pr['runs'])} runs on {m['cpu']}; every problem has {GS.PRICE_RUNS} counted: "
          f"{pr['suffice']} (fewest {min(t.values()) if t else '-'})")
    print(f"           A/A: {pr['aa']['holds'] if pr['aa'] else None} (worst "
          f"{'-' if not pr['aa'] or pr['aa']['worst'] is None else round(pr['aa']['worst'], 4)})")
    print(f"           wrote {os.path.relpath(path, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


# ---------------------------------------------------------------------------
# C. the verdicts
# ---------------------------------------------------------------------------


def geomean_of_medians(prices: dict, keys: list):
    meds = [float(np.median(prices[k])) for k in keys if len(prices.get(k, ())) == GS.PRICE_RUNS]
    return float(math.exp(np.mean(np.log(meds)))) if len(meds) == len(keys) and meds else None


def judge_predictions(acc: dict, pr2: dict, price: dict, features: dict) -> dict:
    """The five predictions of `gates_squares.PREDICTIONS`, read as follows.  An excess is H2's
    energy error less the classical floor in the same frame, as a fraction (0.01 percentage
    points is 1e-4).  A price is the median of a problem's first `PRICE_RUNS` counted.
    1. H2's excess is below 1e-4 on every new problem, in both frames.
    2. E4-H2 is met: H2 meets E4 on every new problem.
    3. PR2 is met; H2's geometric mean is 0.65-0.85, H2r's 0.30-0.60; and H2r's price is below
       H2's on every new problem.
    4. U meets E4 on both problems of every layout without a round feature (a hole or a round
       inclusion: `ROUND_FEATURES`), and misses it on at least one problem of a layout with one.
    5. P-s0's energy error is above H2's on every new problem, in both frames."""
    out = {}
    ex = {k: max(v["frames"][f]["arms"]["H2"]["energy_rel"] - v["frames"][f]["floor"] for f in FRAMES)
          for k, v in acc.items()}
    out["1"] = {"held": all(e < 1e-4 for e in ex.values()), "largest excess": max(ex.values())}
    out["2"] = {"held": all(v["verdicts"]["H2"]["met"] for v in acc.values())}
    keys = list(acc)
    gm2, gmr = geomean_of_medians(price["prices_H2"], keys), geomean_of_medians(price["prices_H2r"], keys)
    cheaper = {k: (float(np.median(price["prices_H2r"][k])) < float(np.median(price["prices_H2"][k])))
               if len(price["prices_H2"].get(k, ())) == GS.PRICE_RUNS == len(price["prices_H2r"].get(k, ()))
               else None for k in keys}
    out["3"] = {"held": bool(pr2.get("met") and gm2 is not None and 0.65 <= gm2 <= 0.85 and gmr is not None
                             and 0.30 <= gmr <= 0.60 and all(cheaper.values())),
                "H2's geometric mean": gm2, "H2r's": gmr, "H2r cheaper on": sum(bool(c) for c in cheaper.values())}
    plain = {k: v for k, v in acc.items() if features[k.split()[1]] not in ROUND_FEATURES}
    ring = {k: v for k, v in acc.items() if features[k.split()[1]] in ROUND_FEATURES}
    out["4"] = {"held": bool(plain) and bool(ring) and all(v["verdicts"]["U"]["met"] for v in plain.values())
                and not all(v["verdicts"]["U"]["met"] for v in ring.values()),
                "U meets E4 without a round feature": sum(v["verdicts"]["U"]["met"] for v in plain.values()),
                "of": len(plain), "U misses E4 with one": sum(not v["verdicts"]["U"]["met"] for v in ring.values())}
    above = {k: all(v["frames"][f]["arms"]["P-s0"]["energy_rel"] > v["frames"][f]["arms"]["H2"]["energy_rel"]
                    for f in FRAMES) for k, v in acc.items()}
    out["5"] = {"held": all(above.values()), "P-s0 above H2 on": sum(above.values()), "of": len(above)}
    return out


def outcome(e4h2: dict, pr2: dict) -> str:
    if not e4h2["met"]:
        return "E4-H2 not met"
    if pr2.get("met") is None:
        return "E4-H2 met, PR2 not judged"
    return "both gates met" if pr2["met"] else "E4-H2 met, PR2 not met"


def price_checks(acc_rec: dict, price: dict) -> list:
    """Why the price record may not be judged; empty when it may."""
    why = []
    if price.get("stamp") != acc_rec.get("stamp"):
        why.append("the price record is of another evaluation")
    if price.get("code") != acc_rec.get("code"):
        why.append(f"the price ran {str(price.get('code'))[:7]}, the accuracy {str(acc_rec.get('code'))[:7]}")
    if not (price.get("on_price_cpu") and GS.on_price_cpu(price.get("machine"))):
        why.append(f"the price was not measured on {GS.PRICE_CPU}")
    if not all(GS.blas_threads_hold(r.get("thread_pools", [])) for r in price.get("runs", [])):
        why.append("a run's BLAS pools were not at the fixed count")
    return why


def table(rec: dict, features: dict) -> list:
    """One row a layout, its steady problem as registered and its step lowered by 300 K (as E4 is
    reported), then the gates, the outcome and the predictions: ten lines or fewer."""
    acc, P = rec["accuracy"], rec["price"]
    lines = ["SQUARES   seed     feature          E% steady/step: H2            floor         U             "
             "P-s0          E4 H2 U P    price H2 / H2r"]
    seeds = list(dict.fromkeys(k.split()[1] for k in acc))
    for seed in seeds:
        cells = {arm: [] for arm in ("H2", "floor", "U", "P-s0")}
        ok, pz = {arm: "" for arm in ("H2", "U", "P-s0")}, []
        for problem in ("steady", "step"):
            v = acc.get(f"squares {seed} {problem}")
            if v is None:
                continue
            f = v["frames"]["registered" if problem == "steady" else "lowered"]
            cells["floor"].append(100 * f["floor"])
            for arm in ("H2", "U", "P-s0"):
                cells[arm].append(100 * f["arms"][arm]["energy_rel"])
                ok[arm] += "Y" if v["verdicts"][arm]["met"] else "n"
            key = f"squares {seed} {problem}"
            got = [P["prices_H2"].get(key, ()), P["prices_H2r"].get(key, ())]
            pz.append("/".join(f"{np.median(x):.2f}" if len(x) == GS.PRICE_RUNS else "-" for x in got))
        row = "  ".join(" / ".join(f"{x:5.3f}" for x in cells[arm]).ljust(13) for arm in ("H2", "floor", "U", "P-s0"))
        lines.append(f"          {seed}  {features.get(seed, '?')[:15]:15s}  {row}  {ok['H2']:2s} {ok['U']:2s} "
                     f"{ok['P-s0']:2s}  {'  '.join(pz)}")
    e4, pr2 = rec["E4-H2"], rec["PR2"]
    gm = pr2.get("geometric_mean")
    lines.append(f"          E4-H2 {'met' if e4['met'] else 'not met'} ({e4['count']}); PR2 "
                 f"{'met' if pr2.get('met') else ('not judged' if pr2.get('met') is None else 'not met')}"
                 f"{'' if gm is None else f' ({gm:.3f})'}; {rec['outcome']}; predictions "
                 + ", ".join(f"{k} {'Y' if v['held'] else 'n'}" for k, v in rec["predictions"].items()))
    return lines[:10]


def layout_features() -> dict:
    rec = json.load(open(RS.RECORD, encoding="utf-8"))
    return {str(lay["seed"]): lay["spec"]["feature"] for lay in rec["layouts"]}


def stage_judge(args) -> int:
    if not args.stamp:
        raise SystemExit("--judge needs --stamp")
    final = path_of(args.out, args.stamp, ".json")
    if os.path.exists(final):
        raise SystemExit(f"{os.path.basename(final)} exists: the one evaluation has been judged")
    acc_path, price_path = path_of(args.out, args.stamp, "-accuracy.json"), path_of(args.out, args.stamp, "-price.json")
    A = json.load(open(acc_path, encoding="utf-8"))
    P = json.load(open(price_path, encoding="utf-8"))
    registered_run = bool(A["registered_run"] and P["registered_run"])
    why = price_checks(A, P)
    keys = list(A["accuracy"])
    if why:
        pr2 = {"met": None, "why": why}
    else:
        try:
            prices = {(int(k.split()[1]), k.split()[2]): P["prices_H2"].get(k, []) for k in keys}
            pr2 = GS.judge_pr2(prices, list(prices), aa=P["aa"])
        except ValueError as e:
            pr2 = {"met": None, "why": [str(e)]}
    gm2, gmr = geomean_of_medians(P["prices_H2"], keys), geomean_of_medians(P["prices_H2r"], keys)
    features = layout_features() if registered_run else {k.split()[1]: "?" for k in keys}
    rec = {"stage": "C, the one evaluation", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "stamp": args.stamp,
           "registration": GS.REGISTERED, "registered_run": registered_run, "code": A["code"],
           "accuracy_record": os.path.basename(acc_path), "accuracy_sha256": RP.file_hash(acc_path),
           "price_record": os.path.basename(price_path), "price_sha256": RP.file_hash(price_path),
           "price_checks": why, "machine": P["machine"], "E4-H2": A["E4-H2"], "PR2": pr2,
           "H2r": {"verdicts": {k: v["verdicts"]["H2r"] for k, v in A["accuracy"].items()},
                   "geometric_mean": gmr, "ratio_to_H2": (gmr / gm2) if gm2 and gmr else None},
           "accuracy": A["accuracy"],
           "price": {"tally": P["tally"], "suffice": P["suffice"], "aa": P["aa"], "runs": len(P["runs"]),
                     "prices_H2": P["prices_H2"], "prices_H2r": P["prices_H2r"]}}
    rec["predictions"] = judge_predictions(A["accuracy"], pr2, P, features) if registered_run else {}
    rec["outcome"] = outcome(A["E4-H2"], pr2)
    lines = table(rec, features)
    registry.write_atomic(final, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    registry.write_atomic(final[:-5] + ".txt", ("\n".join(lines) + "\n").encode("utf-8"))
    for line in lines:
        print(line)
    return 0


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--price", action="store_true", help="stage B, on the rented machine")
    ap.add_argument("--judge", action="store_true", help="stage C")
    ap.add_argument("--verify-payload", default=None, metavar="TAR")
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--registration", default=RS.REGISTRATION)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--problems", nargs="+", default=None,
                    help="kind:what:problem, for tests on problems never drawn or already read")
    ap.add_argument("--control", nargs="+", default=None, help="name:problem, for tests")
    ap.add_argument("--n", type=int, default=GS.BENCHMARK_N)
    ap.add_argument("--reps", type=int, default=GS.PRICE_REPS)
    ap.add_argument("--any-cpu", action="store_true", help="for tests: price off the pricing CPU")
    return ap


def specs_of(args) -> list:
    if not args.problems:
        return list(REGISTERED_SPECS)
    return [tuple(s.split(":")) for s in args.problems]


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_payload:
        if not args.stamp:
            raise SystemExit("--verify-payload needs --stamp")
        v = verify_payload(args.verify_payload, args.stamp, args.out, args.registration)
        bad = [k for k, ok in v["files"].items() if not ok]
        stage_a = str(v["stage A's commit"])[:7]
        print(f"PAYLOAD   {'OK' if v['ok'] else 'REFUSED'}: commit {v['identity'][:7]} (stage A's {stage_a}); "
              f"pinned files {len(v['files']) - len(bad)} of {len(v['files'])}; this script {v['this script']}; "
              f"the box driver {v['the box driver']}" + (f"; differs: {bad}" if bad else ""))
        return 0 if v["ok"] else 1
    if args.price:
        return stage_price(args)
    if args.judge:
        return stage_judge(args)
    return stage_accuracy(args)


if __name__ == "__main__":
    sys.exit(main())
