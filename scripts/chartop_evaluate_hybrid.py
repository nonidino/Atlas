"""The registration after section 11.14 (`atlas/chartop/gates_hybrid.py`, pinned): the one
evaluation.

    python scripts/chartop_evaluate_hybrid.py
    python scripts/chartop_evaluate_hybrid.py --resume

Every measure and bar is `gates_hybrid`'s, which reads E4's from `gates_forms`.  What it
does, in order:

0. **The pin.**  Every file ``out/chartop/registered-hybrid.txt`` records (the drafted scope,
   the files the earlier registration pinned, the control's record) must hash as recorded,
   or it refuses.  It refuses, too, if the one evaluation's record exists already.
1. **The arms.**  U is built from `gates_hybrid.U_FORM`.  H's stand-in for U's networks
   (``chartop_forms_diagnose.without_networks``) must answer U's bits on every piece of every
   problem, in both frames, or it stops; so H is U's form, whatever the stand-in's code.
   P-s0's weights must have the sha256 recorded at download.
2. **The control.**  On section 11.14's six problems, H's energy errors in both frames must
   equal section 11.14's record bit for bit, and P-s0's the one evaluation's, or it stops.
3. **The picks.**  On each new problem, `rule_pieces` must pick the pieces the classical
   record holds, or it stops.
4. **Accuracy, computed once.**  On each new problem, as registered and lowered by 300 K:
   the full-domain reference, the classical floor, and the coupled answers of H, of U on
   every piece and of P-s0, each judged by `judge_e4_both_frames`; then E4-H.
5. **The price.**  `PRICE_RUNS` runs at `PRICE_THREADS` torch threads, each in its own
   process (``--price-run``), the arms taking turns in `PRICE_ORDER` within each of
   `PRICE_REPS` repetitions.  Each run is judged by `run_counts` from its machine state at
   the start and end of its timing (AC, the other processes' use of a core, the power
   events in its window, the classical expert's time per piece); a void run is run again,
   at most `PRICE_RERUNS_MAX` more times, and every run is kept.  Before each run starts,
   it waits (at most ``--quiet-limit`` seconds) until the machine reads quiet; that decides
   only when a run starts, never whether it counts.  PR is `judge_pr` on the counted runs.
   One run at `PRICE_THREADS_REPORTED` thread is reported beside.
6. **The predictions** (`judge_predictions`, the readings in its docstring) and the outcome.
7. It writes ``evaluation-hybrid-<time>.json`` and ``.txt``, whatever they say, and prints
   ten lines or fewer.

Every unit is a checkpoint written whole in ``evaluation-hybrid-<time>-checkpoints/``;
``--resume`` continues the newest unfinished evaluation.  The accuracy runs at
`PRICE_THREADS` torch threads, and the control at the registered resolution, as section
11.14 did.  ``--problems``, ``--control``, ``--n``, ``--reps``, ``--runs``, ``--reruns``,
``--registration`` and ``--out`` exist for tests on the control's problems; the registered
run uses none of them.
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

import chartop_evaluate as ev0  # noqa: E402
import chartop_forms_diagnose as fd  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks as bm, benchmarks_workbench as BW, data  # noqa: E402
from atlas.chartop import forms as fm, gates_hybrid as GH, host as hs, train as tr  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
REGISTRATION = os.path.join(OUT, "registered-hybrid.txt")
PINNED = ("atlas/chartop/gates_hybrid.py", "atlas/chartop/benchmarks_workbench.py", "atlas/chartop/forms.py",
          "atlas/chartop/host.py", "out/chartop/registration-hybrid.json", "atlas/chartop/gates_forms.py",
          "atlas/chartop/gates.py", "atlas/chartop/gates_prior.py", "atlas/chartop/benchmarks.py",
          "out/chartop/forms-diagnose-h3.json")
STAMP_PREFIX = "evaluation-hybrid-"
FRAMES = ("registered", "lowered")
WORKBENCH = "workbench"
#: the problems prediction 2 names
P2_NAMES = ("wall-2", "heated-strip", "bimetal-arc", "plate-hole")
log = ev0.log


# ---------------------------------------------------------------------------
# 0. the pin, and the one evaluation's record
# ---------------------------------------------------------------------------


def file_hash(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def check_pin(path: str = REGISTRATION) -> dict:
    """{file: sha256} for every file the pin records, or SystemExit if one is missing or changed."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: the registration is not pinned")
    text = open(path, encoding="ascii").read()
    out = {}
    for rel in PINNED:
        m = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", text)
        if not m:
            raise SystemExit(f"{path} does not pin {rel}")
        h = file_hash(os.path.join(ROOT, *rel.split("/")))
        if h != m.group(1):
            raise SystemExit(f"{rel} is not the registered file: sha256 {h[:16]}... against {m.group(1)[:16]}...")
        out[rel] = h
    return out


def begin(out: str, resume: bool) -> tuple[str, str]:
    """(stamp, checkpoint directory).  Refuses if the one evaluation's record exists, or if an
    unfinished one exists and ``resume`` is not set; resumes the newest unfinished one."""
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
    d = os.path.join(out, f"{STAMP_PREFIX}{stamp}-checkpoints")
    os.makedirs(d)
    return stamp, d


def checkpoint(d: str, name: str, compute):
    """The unit ``name``: read whole from ``d`` if it is there, else computed and written whole."""
    path = os.path.join(d, name + ".json")
    if os.path.exists(path):
        return json.loads(registry.read_bytes(path).decode("utf-8"))
    val = ev0.strip(compute())
    registry.write_atomic(path, json.dumps(val, indent=1).encode("utf-8"))
    return val


# ---------------------------------------------------------------------------
# 1. the arms
# ---------------------------------------------------------------------------


def build_u() -> fm.KernelExpert:
    """U: `forms.KernelExpert` with `gates_hybrid.U_FORM`'s arguments, after its seed."""
    cfg = dict(GH.U_FORM)
    torch.manual_seed(cfg.pop("seed"))
    return fm.KernelExpert(**cfg).eval()


def probe_piece():
    """The piece section 11.14 read the stand-in's output shapes from: the first piece of the
    held-out bend-3's steady problem at the registered resolution."""
    return fd.bench("bend-3", "registered", GH.CONTROL_VARIANT, GH.BENCHMARK_N, "steady").problems[0]


def make_experts() -> dict:
    u = build_u()
    path = fd.checked_weights(GH.REPORTED_TRAINED)
    return {"U": LearnedExpert(u, envelope=data.ENVELOPE),
            "Z": LearnedExpert(fd.without_networks(u, probe_piece()), envelope=data.ENVELOPE),
            "P-s0": LearnedExpert(tr.load_expert(path, "cpu"), envelope=data.ENVELOPE),
            "P-s0 sha256": registry.sha256_file(path)}


def same_bits(a, z) -> bool:
    return all(np.array_equal(x.H, y.H) and np.array_equal(x.up, y.up)
               and ((x.level is None and y.level is None) or np.array_equal(x.level, y.level))
               for x, y in zip(a, z))


# ---------------------------------------------------------------------------
# 2-4. the problems, the control, the picks and the coupled answers
# ---------------------------------------------------------------------------


def build_problem(spec: tuple, n: int) -> bm.Bench:
    """``spec`` = (name, problem, variant): variant ``"workbench"`` is a new problem
    (`benchmarks_workbench`); any other is `benchmarks`' (the control's ``"held-out"``)."""
    name, problem, variant = spec
    if variant == WORKBENCH:
        return BW.build(name, n, problem)
    return fd.bench(name, "registered", variant, n, problem)


def coupled(b: bm.Bench, arms: dict) -> dict:
    """Each arm's coupled answer on ``b``, as registered and lowered by 300 K, as section
    11.14's ``coupled_errors`` computes it: an arm is (expert, hard pieces), the classical
    solve's responses on the hard pieces and the expert's (answered for every piece)
    elsewhere.  Records the floor and each arm's energy and max error in both frames, and
    E4's verdict on each; and whether ``"H"``'s expert answers ``"U"``'s bits."""
    fr, bits = {}, []
    for frame in FRAMES:
        Lf = (b if frame == "registered" else bm.datum_shift(b, GH.DATUM_SHIFT_K)).layout(GH.KIND, GH.M_PORTS)
        mono = hs.monolithic(Lf)
        cl = Lf.solve(hs.ClassicalExpert())
        un, ud = cl.energy_error(mono.fields)
        mx0, span = ev0.field_errors(mono.fields, cl.fields)
        out = {"floor": un / ud, "floor_max": mx0, "span": span, "arms": {}}
        resp = {}
        for arm, (ex, hard) in arms.items():
            h = set(hard)
            r = ex.respond(Lf.problems, Lf.pb)
            resp[arm] = r
            cp = Lf.couple([cl.responses[i] if i in h else r[i] for i in range(Lf.P)])
            num, den = cp.energy_error(mono.fields)
            mx, _ = ev0.field_errors(mono.fields, cp.fields)
            out["arms"][arm] = {"energy_rel": num / den, "max_err": mx, "span": span, "floor_max": mx0}
        if "H" in resp and "U" in resp:
            bits.append(same_bits(resp["U"], resp["H"]))
        fr[frame] = out
    res = {"frames": fr, "verdicts": {}}
    for arm in arms:
        a, z = fr["registered"]["arms"][arm], fr["lowered"]["arms"][arm]
        res["verdicts"][arm] = GH.judge_e4_both_frames(a, z)
    if bits:
        res["H answers U's bits"] = bool(all(bits))
    return res


def control(ex: dict, n: int, specs=None) -> dict:
    """Section 11.14's six problems (or ``specs``): H's energy errors in both frames against
    section 11.14's record and P-s0's against the one evaluation's, both bit for bit."""
    h3 = json.load(open(os.path.join(ROOT, *GH.CONTROL_RECORD.split("/")), encoding="utf-8"))["problems"]
    rows = fd.evaluation_rows()
    out = {}
    for name, problem in (specs or GH.CONTROL):
        b = build_problem((name, problem, GH.CONTROL_VARIANT), n)
        picks = GH.rule_pieces(b.layout(GH.KIND, GH.M_PORTS))
        res = coupled(b, {"H": (ex["Z"], picks), "P-s0": (ex["P-s0"], [])})
        want_h = h3[f"{name} {problem} held-out"]["hybrid"]
        fr = rows[("held-out", name, problem)]["frames"]
        got = {f: res["frames"][f]["arms"] for f in FRAMES}
        out[f"{name} {problem}"] = {
            "H": [got["registered"]["H"]["energy_rel"], got["lowered"]["H"]["energy_rel"]],
            "H equals section 11.14's": (got["registered"]["H"]["energy_rel"] == want_h["energy_rel"]
                                         and got["lowered"]["H"]["energy_rel"] == want_h["energy_rel_lowered"]),
            "P-s0 equals the one evaluation's": all(
                got[f]["P-s0"]["energy_rel"] == fr[f]["runs"]["P-s0"]["energy_rel"] for f in FRAMES)}
    return out


def expected_picks(spec: tuple, b: bm.Bench):
    """The picks the registration recorded: the classical record's for a new problem,
    section 11.14's labels for the control's; None if neither holds the problem."""
    name, problem, variant = spec
    if variant == WORKBENCH:
        rec = json.load(open(os.path.join(ROOT, *GH.CLASSICAL_RECORD.split("/")), encoding="utf-8"))
        hit = [r["picks"] for r in rec["problems"] if (r["name"], r["problem"]) == (name, problem)]
        return hit[0] if hit else None
    if variant != GH.CONTROL_VARIANT:
        return None
    h3 = json.load(open(os.path.join(ROOT, *GH.CONTROL_RECORD.split("/")), encoding="utf-8"))["problems"]
    labels = h3.get(f"{name} {problem} held-out", {}).get("hard")
    if labels is None:
        return None
    index = {str(p.label): i for i, p in enumerate(b.problems)}
    return sorted(index[lab] for lab in labels)


def picks_of(spec: tuple, b: bm.Bench) -> list:
    """`rule_pieces` on ``b``, refused unless it is what the registration recorded."""
    picks = GH.rule_pieces(b.layout(GH.KIND, GH.M_PORTS))
    want = expected_picks(spec, b)
    if want is not None and picks != want:
        raise SystemExit(f"{spec[0]} {spec[1]}: the rule picks {picks}, the registration recorded {want}")
    return picks


def accuracy(spec: tuple, ex: dict, n: int) -> dict:
    b = build_problem(spec, n)
    picks = picks_of(spec, b)
    res = coupled(b, {"H": (ex["Z"], picks), "U": (ex["U"], []), "P-s0": (ex["P-s0"], [])})
    if not res["H answers U's bits"]:
        raise SystemExit(f"{spec[0]} {spec[1]}: H's stand-in does not answer U's bits")
    res.update(pieces=len(b.problems), picks=picks)
    return res


# ---------------------------------------------------------------------------
# 5. the price
# ---------------------------------------------------------------------------


def price_problem(b: bm.Bench, picks: list, ex: dict, reps: int) -> dict:
    """The best of ``reps`` wall-clock times of each arm, the arms taking turns in
    `PRICE_ORDER` within each repetition: the classical expert's responses for every piece;
    H's (the classical solve on the picks, the stand-in elsewhere); P-s0's; and the coupled
    solve that every arm shares."""
    L = b.layout(GH.KIND, GH.M_PORTS)
    ps, C = L.problems, hs.ClassicalExpert()
    hard = set(picks)
    picked, easy = [ps[i] for i in range(L.P) if i in hard], [ps[i] for i in range(L.P) if i not in hard]
    resp_C = C.respond(ps, L.pb)

    def h():
        if picked:
            C.respond(picked, L.pb)
        if easy:
            ex["Z"].respond(easy, L.pb)
    arms = {"classical": lambda: C.respond(ps, L.pb), "H": h,
            "P-s0": lambda: ex["P-s0"].respond(ps, L.pb), "coupled solve": lambda: L.couple(resp_C)}
    times = {k: [] for k in GH.PRICE_ORDER}
    for _ in range(reps):
        for k in GH.PRICE_ORDER:
            t0 = time.perf_counter()
            arms[k]()
            times[k].append(time.perf_counter() - t0)
    best = {k: min(v) for k, v in times.items()}
    return {"pieces": L.P, "seconds": best, "every_repetition": times,
            "price": GH.price(best["H"], best["classical"], best["coupled solve"]),
            "price_P-s0": GH.price(best["P-s0"], best["classical"], best["coupled solve"]),
            "classical_ms_per_piece": 1000.0 * best["classical"] / L.P}


def other_core_max(exclude: set, interval: float = 1.0) -> float:
    """The largest share of one core any other process used over ``interval`` seconds.  No
    process is named: the record is public."""
    import psutil
    procs = []
    for p in psutil.process_iter():
        try:
            if p.pid in exclude or p.pid == 0:
                continue
            p.cpu_percent(None)
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(interval)
    top = 0.0
    for p in procs:
        try:
            top = max(top, p.cpu_percent(None) / 100.0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return top


def machine_state(exclude: set) -> dict:
    import psutil
    bat = psutil.sensors_battery()
    return {"time": dt.datetime.now().isoformat(timespec="seconds"),
            "plugged": bool(bat.power_plugged) if bat is not None else None,
            "percent": float(bat.percent) if bat is not None else None,
            "other_core_max": other_core_max(exclude)}


def power_events(start: str, end: str):
    """The Ids of the Kernel-Power events `POWER_EVENT_IDS` (a power-source change, entering
    and leaving modern standby) in the System log between ``start`` and ``end`` (local ISO
    times); None if the log cannot be read, which `run_counts` then treats as an event."""
    ids = ",".join(str(i) for i in GH.POWER_EVENT_IDS)
    cmd = (f"$s=[datetime]::Parse('{start}'); $e=[datetime]::Parse('{end}'); "
           f"try {{ Get-WinEvent -FilterHashtable @{{LogName='System'; "
           f"ProviderName='Microsoft-Windows-Kernel-Power'; Id={ids}; StartTime=$s; EndTime=$e}} "
           f"-ErrorAction Stop | ForEach-Object {{ $_.Id }} }} catch [Exception] {{ "
           f"if ($_.Exception.Message -match 'No events') {{ 'NONE' }} else {{ 'ERROR' }} }}")
    try:
        p = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                           text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    words = p.stdout.split()
    if "ERROR" in words or p.returncode != 0:
        return None
    return [int(w) for w in words if w.isdigit()]


def price_run(k: int, threads: int, reps: int, specs: list, n: int, out_dir: str) -> dict:
    """One price run, in this process: the experts and the problems built first, then the
    machine's state at the start of the timing, each problem priced, the state at its end,
    and `run_counts`' verdict.  Written to ``out_dir``."""
    torch.set_num_threads(threads)
    exclude = {os.getpid(), os.getppid()}
    ex = make_experts()
    built = []
    for spec in specs:
        b = build_problem(spec, n)
        built.append((spec, b, picks_of(spec, b)))
    start = machine_state(exclude)
    prices = {}
    for spec, b, picks in built:
        prices[f"{spec[0]} {spec[1]}"] = price_problem(b, picks, ex, reps)
    end = machine_state(exclude)
    events = power_events(start["time"], end["time"])
    state = {"plugged_start": start["plugged"] is not False, "plugged_end": end["plugged"] is not False,
             "power_events": list(GH.POWER_EVENT_IDS[:1]) if events is None else events,
             "other_core_max": max(start["other_core_max"], end["other_core_max"]),
             "classical_ms_per_piece": {key: v["classical_ms_per_piece"] for key, v in prices.items()}}
    rec = {"run": k, "torch_threads": threads, "reps": reps, "n": n, "start": start, "end": end,
           "power_events": events, "problems": prices, "counts": GH.run_counts(state)}
    name = f"price-run-{k}-threads{threads}.json"
    registry.write_atomic(os.path.join(out_dir, name), json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    return rec


def wait_for_quiet(limit_s: float = 600.0, every_s: float = 10.0) -> dict:
    """Before a price run starts: wait, at most ``limit_s``, until no other process uses more
    than `VALID_OTHER_CORE` of a core.  It decides only when a run starts, on the machine's
    state alone; whether the run counts is still judged at the run's own start and end."""
    t0, readings = time.perf_counter(), []
    while True:
        v = other_core_max({os.getpid()})
        readings.append(round(v, 3))
        if v <= GH.VALID_OTHER_CORE or time.perf_counter() - t0 >= limit_s:
            return {"waited_s": time.perf_counter() - t0, "readings": readings,
                    "quiet": v <= GH.VALID_OTHER_CORE}
        time.sleep(every_s)


def spawn_price_run(k: int, threads: int, args, out_dir: str) -> dict:
    """A price run in its own process, started once the machine is quiet (`wait_for_quiet`,
    recorded with the run); returns its record."""
    quiet = wait_for_quiet(args.quiet_limit)
    if not quiet["quiet"]:
        log(f"price run {k}: still busy after {quiet['waited_s']:.0f} s; starting it anyway")
    cmd = [sys.executable, os.path.abspath(__file__), "--price-run", str(k), "--threads", str(threads),
           "--price-dir", out_dir, "--n", str(args.n), "--reps", str(args.reps)]
    if args.problems:
        cmd += ["--problems", *args.problems]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    path = os.path.join(out_dir, f"price-run-{k}-threads{threads}.json")
    if p.returncode != 0 or not os.path.exists(path):
        raise SystemExit(f"price run {k} failed (exit {p.returncode}):\n{p.stderr[-2000:]}")
    rec = json.loads(registry.read_bytes(path).decode("utf-8"))
    rec["before_start"] = quiet
    return rec


def price(args, d: str, specs: list) -> dict:
    """`PRICE_RUNS` counted runs (a void run run again, at most `PRICE_RERUNS_MAX` more
    times), every run kept; PR on the counted runs' prices; one run at one thread beside."""
    runs, counted = [], []
    k = 0
    while len(counted) < args.runs and k < args.runs + args.reruns:
        k += 1
        rec = checkpoint(d, f"price-run-{k}-threads{GH.PRICE_THREADS}",
                         lambda: spawn_price_run(k, GH.PRICE_THREADS, args, d))
        runs.append(rec)
        log(f"price run {k}: {'counts' if rec['counts']['counts'] else 'void: ' + '; '.join(rec['counts']['failed'])}")
        if rec["counts"]["counts"]:
            counted.append(rec)
    out = {"runs": runs, "counted": [r["run"] for r in counted]}
    keys = [f"{s[0]} {s[1]}" for s in specs]
    if len(counted) == args.runs:
        prices = {tuple(key.split(" ", 1)): [r["problems"][key]["price"] for r in counted] for key in keys}
        if [tuple(k_.split(" ", 1)) for k_ in keys] == list(GH.NEW_PROBLEMS) and args.runs == GH.PRICE_RUNS:
            out["PR"] = GH.judge_pr(prices)
        else:
            meds = [float(np.median(v)) for v in prices.values()]
            out["PR"] = {"met": None, "geometric_mean": float(math.exp(np.mean(np.log(meds)))),
                         "note": "not the registered problems: PR not judged"}
    else:
        out["PR"] = {"met": None, "note": f"only {len(counted)} of {args.runs} runs counted after {k}: "
                                          "PR not judged"}
    out["reported_one_thread"] = checkpoint(
        d, f"price-run-0-threads{GH.PRICE_THREADS_REPORTED}",
        lambda: spawn_price_run(0, GH.PRICE_THREADS_REPORTED, args, d))
    return out


# ---------------------------------------------------------------------------
# 6. the predictions and the outcome
# ---------------------------------------------------------------------------


def judge_predictions(acc: dict, pr: dict) -> dict:
    """The five predictions of `gates_hybrid.PREDICTIONS`, read as follows.  An excess is an
    arm's energy error less the classical floor in the same frame, as fractions (0.01
    percentage points is 1e-4).
    1. U's excess on heated-strip is below 1e-4, steady and step, in both frames.
    2. H's excess is below 1e-2 on wall-2, heated-strip, bimetal-arc and plate-hole, steady
       and step, in both frames.
    3. Of the six steady problems (whose frames agree), H's largest excess as registered is
       on plate-insert or fast-heat.
    4. PR is met, and the geometric mean of H's prices is 0.70-0.88.
    5. P-s0's energy error is above H's on every new problem in both frames, and P-s0 meets
       E4 on fewer new problems than H."""
    def excess(arm, name, problem, frame):
        f = acc[f"{name} {problem}"]["frames"][frame]
        return f["arms"][arm]["energy_rel"] - f["floor"]
    out = {}
    e1 = {f"{p} {f}": excess("U", "heated-strip", p, f) for p in BW.PROBLEMS for f in FRAMES}
    out["1"] = {"held": all(v < 1e-4 for v in e1.values()), "U's excess on heated-strip": e1}
    e2 = {f"{n} {p} {f}": excess("H", n, p, f) for n in P2_NAMES for p in BW.PROBLEMS for f in FRAMES}
    out["2"] = {"held": all(v < 1e-2 for v in e2.values()), "largest": max(e2.values())}
    e3 = {n: excess("H", n, "steady", "registered") for n in BW.NAMES}
    top = max(e3, key=e3.get)
    out["3"] = {"held": top in ("plate-insert", "fast-heat"), "largest on": top, "H's steady excess": e3}
    gm = (pr or {}).get("geometric_mean")
    out["4"] = {"held": bool(pr and pr.get("met") and gm is not None and 0.70 <= gm <= 0.88),
                "geometric_mean": gm}
    above = {k: all(v["frames"][f]["arms"]["P-s0"]["energy_rel"] > v["frames"][f]["arms"]["H"]["energy_rel"]
                    for f in FRAMES) for k, v in acc.items()}
    n_p = sum(v["verdicts"]["P-s0"]["met"] for v in acc.values())
    n_h = sum(v["verdicts"]["H"]["met"] for v in acc.values())
    out["5"] = {"held": all(above.values()) and n_p < n_h, "P-s0 above H": above,
                "E4 met by P-s0, by H": [n_p, n_h]}
    return out


def outcome(e4h: dict, pr: dict) -> str:
    if not e4h["met"]:
        return "E4-H not met"
    if pr.get("met") is None:
        return "E4-H met, PR not judged"
    return "both gates met" if pr["met"] else "E4-H met, PR not met"


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--registration", default=REGISTRATION)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--problems", nargs="+", default=None,
                    help="name:problem:variant, for tests on the control's problems only")
    ap.add_argument("--control", nargs="+", default=None, help="name:problem, for tests")
    ap.add_argument("--n", type=int, default=GH.BENCHMARK_N)
    ap.add_argument("--reps", type=int, default=GH.PRICE_REPS)
    ap.add_argument("--runs", type=int, default=GH.PRICE_RUNS)
    ap.add_argument("--reruns", type=int, default=GH.PRICE_RERUNS_MAX)
    ap.add_argument("--quiet-limit", type=float, default=600.0,
                    help="seconds to wait for a quiet machine before each price run")
    ap.add_argument("--price-run", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, default=GH.PRICE_THREADS, help=argparse.SUPPRESS)
    ap.add_argument("--price-dir", default=None, help=argparse.SUPPRESS)
    return ap


def specs_of(args) -> list:
    if not args.problems:
        return [(name, problem, WORKBENCH) for name, problem in GH.NEW_PROBLEMS]
    return [tuple(s.split(":")) for s in args.problems]


def table(rec: dict) -> list:
    """One row a scenario, its steady problem as registered and its step lowered by 300 K
    (as E4 is reported), then the gates and the predictions: ten lines or fewer."""
    acc, P = rec["accuracy"], rec["price"]
    med = {}
    for r in P["runs"]:
        if r["counts"]["counts"]:
            for key, v in r["problems"].items():
                med.setdefault(key, []).append(v["price"])
    names = list(dict.fromkeys(k.split(" ", 1)[0] for k in acc))
    lines = ["HYBRID    scenario       E% steady / step:  H              floor          U              "
             "P-s0           E4 H U P    price"]
    for name in names:
        cells = {arm: [] for arm in ("H", "floor", "U", "P-s0")}
        ok, pz = {arm: "" for arm in ("H", "U", "P-s0")}, []
        for problem in BW.PROBLEMS:
            v = acc.get(f"{name} {problem}")
            if v is None:
                continue
            f = v["frames"]["registered" if problem == "steady" else "lowered"]
            cells["floor"].append(100 * f["floor"])
            for arm in ("H", "U", "P-s0"):
                cells[arm].append(100 * f["arms"][arm]["energy_rel"])
                ok[arm] += "Y" if v["verdicts"][arm]["met"] else "n"
            pz.append(f"{np.median(med[f'{name} {problem}']):.2f}" if f"{name} {problem}" in med else "-")
        row = "  ".join(" / ".join(f"{x:5.3f}" for x in cells[arm]).ljust(13) for arm in ("H", "floor", "U", "P-s0"))
        lines.append(f"          {name:14s} {row}  {ok['H']:2s} {ok['U']:2s} {ok['P-s0']:2s}  {' / '.join(pz)}")
    e4h, pr = rec["E4-H"], P["PR"]
    gm = pr.get("geometric_mean")
    lines.append(f"          E4-H {'met' if e4h['met'] else 'not met'} ({e4h['count']}); PR "
                 f"{'met' if pr.get('met') else ('not judged' if pr.get('met') is None else 'not met')}"
                 f"{'' if gm is None else f' (geometric mean {gm:.3f})'}; outcome: {rec['outcome']}")
    lines.append("          predictions held: " + ", ".join(
        f"{k} {'Y' if v['held'] else 'n'}" for k, v in rec["predictions"].items()))
    return lines


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.price_run is not None:
        price_run(args.price_run, args.threads, args.reps, specs_of(args), args.n, args.price_dir)
        return 0
    t0 = time.perf_counter()
    pins = check_pin(args.registration)
    stamp, d = begin(args.out, args.resume)
    torch.set_num_threads(GH.PRICE_THREADS)
    log(f"evaluation {stamp}: the pin holds ({len(pins)} files)")
    ex = make_experts()
    specs = specs_of(args)
    control_specs = [tuple(s.split(":")) for s in args.control] if args.control else None
    ctl = checkpoint(d, "control", lambda: control(ex, GH.BENCHMARK_N, control_specs))
    bad = [k for k, v in ctl.items() if not (v["H equals section 11.14's"] and v["P-s0 equals the one evaluation's"])]
    if bad:
        raise SystemExit(f"the control does not reproduce on {bad}; checkpoints kept in {d}")
    log("the control reproduces bit for bit")
    acc = {}
    for spec in specs:
        key = f"{spec[0]} {spec[1]}"
        acc[key] = checkpoint(d, "accuracy-" + key.replace(" ", "-"), lambda spec=spec: accuracy(spec, ex, args.n))
        log(f"{key}: H {100 * acc[key]['frames']['registered']['arms']['H']['energy_rel']:.3f}% / "
            f"{100 * acc[key]['frames']['lowered']['arms']['H']['energy_rel']:.3f}%")
    registered_run = [s[:2] for s in specs] == list(GH.NEW_PROBLEMS) and all(s[2] == WORKBENCH for s in specs)
    verdicts = {tuple(k.split(" ", 1)): v["verdicts"]["H"] for k, v in acc.items()}
    e4h = (GH.judge_e4h(verdicts) if registered_run else
           {"met": all(v["met"] for v in verdicts.values()), "count": f"{sum(v['met'] for v in verdicts.values())} "
            f"of {len(verdicts)}", "note": "not the registered problems"})
    pz = price(args, d, specs)
    rec = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "stamp": stamp, "registration": GH.REGISTERED,
           "registered_run": registered_run, "pins": pins, "script_sha256": file_hash(os.path.abspath(__file__)),
           "diagnose_script_sha256": file_hash(fd.__file__), "torch": torch.__version__,
           "torch_threads_accuracy": GH.PRICE_THREADS, "platform": platform.platform(),
           "P-s0 sha256": ex["P-s0 sha256"], "n": args.n, "control": ctl, "accuracy": acc,
           "E4-H": e4h, "price": pz}
    rec["predictions"] = (judge_predictions(acc, pz["PR"]) if registered_run else {})
    rec["outcome"] = outcome(e4h, pz["PR"])
    rec["elapsed_s"] = time.perf_counter() - t0
    path = os.path.join(args.out, f"{STAMP_PREFIX}{stamp}.json")
    lines = table(rec)
    registry.write_atomic(path, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    registry.write_atomic(path[:-5] + ".txt", ("\n".join(lines) + "\n").encode("utf-8"))
    for line in lines:
        print(line)
    print(f"          wrote {os.path.relpath(path, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
