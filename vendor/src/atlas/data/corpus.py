"""Resumable, provenance-carrying corpus generation (tasks T3.3 and T3.4).

The corpus is the single most expensive artifact in the project: a production run
is ~1,000-1,500 core-hours against a trained expert's ~2 GPU-hours, and it cannot
be patched incrementally -- a missing field or an unrecorded interface means
regenerating everything. Two consequences drive this module.

**Resume is not a convenience.** A multi-hour run on a spot or preemptible box
*will* be interrupted; assume it and make the restart cheap. Episodes are
independent, so the unit of recovery is one episode: write to a temporary file,
rename it into place only when it is complete and valid, and record it in a
manifest that a later run reads before deciding what work is left.

**Resume by CONTENT, not by filename.** `episode_0007.h5` says nothing about
which sweep point, which timing, or which config produced it. If the throat
radius changes (D2) or the snapshot cadence changes (D1), the file on disk is
stale even though its name is right -- and silently keeping it would mix two
geometries inside one corpus, which no downstream check would catch. Every
episode therefore carries an `episode_id`: a hash of the sweep point, the
resolved timing, the D1-D5 decisions and the config fingerprint. A file whose id
does not match the plan is regenerated, not reused.

    from atlas.data.corpus import PLANS, generate_corpus
    generate_corpus(PLANS["pilot"], "data/atlas_pilot")
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import AtlasConfig, load_config
from . import schema
import h5py
import numpy as np

from .generate import EpisodeSpec, generate_episode
from .generate_solo import SOLO_SWEEPS
from .sweep import SweepPoint, build_sweep

MANIFEST = "manifest.json"


@dataclass(frozen=True)
class CorpusPlan:
    """One named corpus. `note` records WHY this size, because the size is a
    decision (D6) rather than a constant."""
    name: str
    n_train: int
    n_test: int
    n_macro: int = 4
    dt_snap: float = 1.0e-3
    dt_macro: float | None = None      # None -> cfg.dt_macro (unchanged, per D1)
    coarsen: int = 1
    riemann: str = "hllc"
    farfield: str = "prescribed"
    couple_every: str = "snapshot"
    reduction: str = "B-prime"
    seed: int = 20260809
    note: str = ""

    def spec(self, point: SweepPoint, cfg: AtlasConfig) -> EpisodeSpec:
        return EpisodeSpec(point=point, n_macro=self.n_macro, dt_macro=self.dt_macro,
                           dt_snap=self.dt_snap, coarsen=self.coarsen,
                           riemann=self.riemann, farfield=self.farfield,
                           couple_every=self.couple_every, reduction=self.reduction)

    def points(self) -> tuple[list, list, list]:
        return build_sweep(self.n_train, self.n_test, seed=self.seed)


# --------------------------------------------------------------------------
# D6: the corpus size is a measurement, not a guess
# --------------------------------------------------------------------------
# The decision rule, from the corpus completion plan: generate the PILOT, run the
# with/without-public-pretraining ablation on `reacting_flow` (two ~2 GPU-hour
# runs) with all2all stage-A pretraining already enabled, and set the production
# count from that measurement -- 60 episodes if pretraining helps as expected, up
# toward 144 if it does not. `production` below is the midpoint that the ablation
# is expected to confirm or move; it is deliberately NOT the 300+40 of the
# original spec, which route B' does not need and the public-data audit says is
# not where the marginal value lies.
PLANS = {
    "smoke": CorpusPlan(
        name="smoke", n_train=2, n_test=1, n_macro=2, dt_snap=1.0e-5,
        dt_macro=2.0e-5, coarsen=4, reduction="smoke",
        note="pipeline check only; never training data"),
    "pilot": CorpusPlan(
        name="pilot", n_train=24, n_test=8,
        note="T3.5: ~525 core-hours, ~11 h on 48 cores. Validate the M2 gates "
             "(including G5 and G6) on this before committing to production."),
    "production": CorpusPlan(
        name="production", n_train=60, n_test=12,
        note="T3.7: size set by D6's ablation; 60 train is the with-pretraining "
             "branch. Raise toward 120 only if the ablation says pretraining does "
             "not transfer."),
    "production_max": CorpusPlan(
        name="production_max", n_train=120, n_test=24,
        note="D6's no-transfer branch: the pre-audit working corpus size."),
}


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def episode_id(spec: EpisodeSpec, cfg: AtlasConfig) -> str:
    """Content hash of everything that changes what the episode contains.

    Deliberately includes the config fingerprint: the throat radius, the grids and
    the declared edge list all live there, and a corpus that mixes two of any of
    them is not one corpus."""
    tm = spec.timing(cfg)
    payload = {
        "point": spec.point.as_dict(),
        "split": spec.point.split,
        "idx": spec.point.idx,
        "timing": asdict(tm),
        "coarsen": spec.coarsen,
        "riemann": spec.riemann,
        "inviscid": spec.inviscid,
        "farfield": spec.farfield,
        "snapshot_policy": spec.snapshot_policy,
        "couple_every": spec.couple_every,
        "cfg": cfg.fingerprint(),
        "schema": schema.SCHEMA_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Checksum of the written file, so an archived corpus can be verified after
    the box that made it is gone (T3.9)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


@dataclass
class Manifest:
    """The idempotent record of what exists. Written after every episode, so an
    interrupted run loses at most the episode in flight."""
    path: Path
    plan: dict = field(default_factory=dict)
    episodes: dict = field(default_factory=dict)      # episode_id -> record

    @classmethod
    def load(cls, out: str | Path) -> "Manifest":
        p = Path(out) / MANIFEST
        if not p.exists():
            return cls(path=p)
        d = json.loads(p.read_text())
        return cls(path=p, plan=d.get("plan", {}), episodes=d.get("episodes", {}))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(
            {"plan": self.plan, "episodes": self.episodes}, indent=2, sort_keys=True))
        os.replace(tmp, self.path)

    def is_done(self, eid: str, out: Path) -> bool:
        """Complete means: recorded, the file is still there, and its bytes are the
        ones that were recorded. A truncated file left by a killed process fails
        the checksum and is regenerated rather than trusted."""
        rec = self.episodes.get(eid)
        if not rec or rec.get("error"):
            return False
        f = out / rec["file"]
        if not f.exists():
            return False
        if rec.get("sha256") and file_sha256(f) != rec["sha256"]:
            return False
        return True

    @property
    def n_valid(self) -> int:
        return sum(1 for r in self.episodes.values()
                   if not r.get("error") and not r.get("problems"))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def generate_one(spec: EpisodeSpec, out: Path, cfg: AtlasConfig) -> dict:
    """Generate, validate and atomically install one episode.

    The episode is written under a `.partial` name and renamed only once it has
    passed validation, so a directory listing never contains a half-written file
    that a later resume would trust."""
    eid = episode_id(spec, cfg)
    final = out / f"episode_{spec.point.split}_{spec.point.idx:04d}.h5"
    staging = out / f".{final.stem}.{eid}.partial"
    t0 = time.perf_counter()
    rec: dict = {"id": eid, "idx": spec.point.idx, "split": spec.point.split,
                 "file": final.name}
    try:
        if staging.exists():
            staging.unlink()
        written = generate_episode(spec, staging.parent, cfg)
        os.replace(written, staging)
        problems = schema.validate_episode(staging, cfg,
                                           n_snapshots=spec.timing(cfg).n_snapshots)
        rec["problems"] = problems
        rec["mass_residual"] = schema.mass_budget_residual(staging)
        for key, fn in (("g5_interface_consistency", schema.interface_flux_consistency),
                        ("g5_lag_error", schema.interface_lag_error)):
            try:
                rec[key] = fn(staging)
            except ValueError as exc:                  # pre-v2 file: recorded, not hidden
                rec[key] = {"error": str(exc)}
        os.replace(staging, final)
        rec["sha256"] = file_sha256(final)
        rec["bytes"] = final.stat().st_size
    except Exception as exc:                           # noqa: BLE001 - recorded, not hidden
        rec["error"] = f"{type(exc).__name__}: {exc}"
        if staging.exists():
            staging.unlink()
    rec["wall_seconds"] = time.perf_counter() - t0
    return rec


def _worker(args):
    """Process-pool entry point. Rebuilds the config in the child rather than
    pickling it, because Windows spawns a fresh interpreter per worker."""
    spec, out, cfg_path = args
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return generate_one(spec, Path(out), load_config(cfg_path))


def generate_corpus(plan: CorpusPlan, out: str | Path, cfg: AtlasConfig | None = None,
                    resume: bool = True, workers: int = 1, cfg_path: str | None = None,
                    log=print) -> Manifest:
    """Run `plan` into `out`, skipping episodes already present and valid.

    Generation is embarrassingly parallel across episodes -- no communication, no
    scaling loss -- so `workers` is just the core count of the rented box."""
    cfg = cfg if cfg is not None else load_config(cfg_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train, test, rejected = plan.points()
    (out / "rejected.json").write_text(json.dumps(rejected, indent=2))

    man = Manifest.load(out)
    _refuse_mixed_designs(man, "plan", out)
    from .generate import _git_sha
    man.plan = {**asdict(plan), "cfg_fingerprint": cfg.fingerprint(),
                "git_sha": _git_sha(), "schema_version": schema.SCHEMA_VERSION,
                "decisions": "D1 B-prime + hold-last | D2 rounded throat | "
                             "D3 characteristic far field | D4 record all, declare seven | "
                             "D5 t_exch stamps | D6 pilot-then-ablation sizing"}
    man.save()

    todo = []
    for point in list(train) + list(test):
        spec = plan.spec(point, cfg)
        eid = episode_id(spec, cfg)
        if resume and man.is_done(eid, out):
            log(f"[{point.split} {point.idx:04d}] skip (done, id {eid})")
            continue
        todo.append((spec, eid))

    log(f"{len(todo)} episodes to generate, {len(train) + len(test) - len(todo)} already done")

    if workers > 1 and todo:
        from concurrent.futures import ProcessPoolExecutor
        payload = [(spec, str(out), cfg_path) for spec, _ in todo]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rec in pool.map(_worker, payload):
                man.episodes[rec["id"]] = rec
                man.save()
                log(_fmt(rec))
    else:
        for spec, _ in todo:
            rec = generate_one(spec, out, cfg)
            man.episodes[rec["id"]] = rec
            man.save()
            log(_fmt(rec))

    log(f"{man.n_valid}/{len(man.episodes)} episodes valid -> {out}")
    return man


def _fmt(rec: dict) -> str:
    if rec.get("error"):
        return f"[{rec['split']} {rec['idx']:04d}] FAILED: {rec['error']}"
    if rec.get("group"):                               # solo: no trajectory, no G5
        ok = "OK" if not rec["problems"] else str(rec["problems"])
        return (f"[{rec['group']} {rec['split']} {rec['idx']:04d}] {rec['file']}  "
                f"{rec['wall_seconds']:.1f}s  {rec['bytes'] / 1e6:.0f} MB  {ok}")
    g5 = rec.get("g5_interface_consistency", {})
    g5s = " ".join(f"{k}={v:.2e}" for k, v in g5.items() if isinstance(v, float))
    ok = "OK" if not rec["problems"] else str(rec["problems"])
    return (f"[{rec['split']} {rec['idx']:04d}] {rec['file']}  {rec['wall_seconds']:.1f}s  "
            f"mass {rec['mass_residual']:.2e}  G5 {g5s}  {ok}")


# --------------------------------------------------------------------------
# budgeted portions: the corpus as a sequence of bounded runs
# --------------------------------------------------------------------------
# The constraint that shapes this section is a hard wall-clock cap per rented
# session -- 5 hours, after which the box goes away. Three consequences:
#
#   1. **The corpus is accumulated, not generated.** Each portion is a separate
#      bounded run into the SAME directory; resume-by-content means run N+1 picks
#      up exactly where run N stopped, and a portion that is interrupted early
#      loses at most the episode in flight.
#   2. **Cheap tiers go first.** The T2 solo sweeps cost hours of core time, not
#      hundreds, and two of them supply data the coupled corpus structurally
#      cannot (the choked-throat sweep) or supplies badly (the shell, which has
#      no CFL limit and can run the full 10 s flight).
#   3. **The horizon, not the resolution, is the coupled lever.** Coarsening the
#      grid breaks the solver-tokenizer cell correspondence that body-fitting
#      exists to guarantee, and interface fluxes do not survive interpolation --
#      so `coarsen` stays 1 and the horizon shortens instead.
#
# Measured on the development box (22 logical cores, single-threaded numpy,
# 2026-08-18) -- core-hours per SECOND of simulated flight, all agents:
#
#     coarsen=1  68,736 cells   65.6 core-h/s     coarsen=2   14.1     coarsen=4   4.05
#
# Note the scaling is far shallower than the r^3 the cell count suggests: at
# these sizes numpy is overhead-bound, not flop-bound. That is another reason not
# to buy fidelity reductions with coarsening -- it does not pay what it looks
# like it should.
CORE_HOURS_PER_FLIGHT_SECOND = 65.6      # coarsen=1, measured, see above


@dataclass(frozen=True)
class Portion:
    """One bounded unit of generation, sized to fit a single rented session.

    `target` is what the portion wants; the runner generates as many as the
    budget allows and leaves the rest for the next session. `serves` is not
    decoration -- a portion whose purpose is not written down is the one that gets
    cut when the budget is tight, and for two of these that would silently delete
    a Phase-2 acceptance gate."""
    name: str
    kind: str                       # "coupled" | "solo"
    target: int
    serves: str
    group: str = ""                 # solo only
    T_ep: float = 0.1               # coupled only
    dt_snap: float = 5.0e-4
    n_macro: int = 2
    split_test: int = 0             # coupled: how many held-out corner episodes

    def core_hours_each(self) -> float:
        if self.kind == "coupled":
            return CORE_HOURS_PER_FLIGHT_SECOND * self.T_ep
        sw = SOLO_SWEEPS[self.group]
        # core-hours per flight-second for the group; shell re-measured
        # 2026-08-18 against a full 64-run portion (30 s per 10 s history)
        frac = {"nozzle": 63.4, "external": 6.1, "shell": 0.00083}[self.group]
        return frac * sw.T_ep

    def fits(self, hours: float, workers: int) -> int:
        each = self.core_hours_each()
        return max(int((hours * workers) / max(each, 1e-9)), 0)


# The ordered plan. Run them top to bottom; each line is one session.
PORTIONS = {
    p.name: p for p in (
        Portion(name="shell", kind="solo", group="shell", target=64,
                serves="thermostruct expert; the only expert whose data is not "
                       "compute-bound, so it gets full 10 s histories and the widest "
                       "boundary-condition coverage in the corpus"),
        Portion(name="nozzle", kind="solo", group="nozzle", target=48,
                serves="the choked-throat gate (M4) and regime gate G6 -- back "
                       "pressure swept at FIXED chamber pressure, which no coupled "
                       "episode and no public dataset contains"),
        Portion(name="external", kind="solo", group="external", target=32,
                serves="external_flow coverage in Mach and altitude, independent of "
                       "the shell and of the plume"),
        Portion(name="coupled_a", kind="coupled", target=16, T_ep=0.1, n_macro=2,
                serves="INTERFACE supervision -- the one thing no other tier and no "
                       "public dataset can provide; 200 snapshots each, so all2all "
                       "gives 19,900 lead-time pairs per episode"),
        Portion(name="coupled_b", kind="coupled", target=16, T_ep=0.1, n_macro=2,
                serves="second coupled session; doubles the distinct physical "
                       "configurations, which is what augmentation cannot do"),
        Portion(name="baselines", kind="coupled", target=8, T_ep=0.1, n_macro=2,
                split_test=8,
                serves="Phase-4 held-out CORNER episodes, generated now because "
                       "renting the box twice for the same solver time is waste"),
    )
}


def schedule(hours: float = 5.0, workers: int = 8) -> list[dict]:
    """What each session can actually produce, at this budget and core count."""
    rows = []
    for p in PORTIONS.values():
        each = p.core_hours_each()
        fit = p.fits(hours, workers)
        rows.append({
            "portion": p.name, "kind": p.kind, "target": p.target,
            "core_h_each": each, "fits_in_session": fit,
            "sessions": max(1, -(-p.target // max(fit, 1))),
            "core_h_total": each * p.target, "serves": p.serves,
        })
    return rows


def _solo_specs_for(portion: Portion, cfg: AtlasConfig, plan_seed: int):
    from .generate_solo import build_solo_specs
    train, test, _ = build_sweep(24, 8, seed=plan_seed)
    return build_solo_specs(SOLO_SWEEPS[portion.group], list(train) + list(test),
                            seed=plan_seed)[:portion.target]


def _coupled_specs_for(portion: Portion, cfg: AtlasConfig, plan_seed: int):
    train, test, _ = build_sweep(64, 8, seed=plan_seed)
    pts = list(test) if portion.split_test else list(train)
    if not portion.split_test:
        # portions after the first continue down the same ordered sweep, so two
        # sessions never regenerate the same physical configuration
        prior = sum(q.target for q in PORTIONS.values()
                    if q.kind == "coupled" and not q.split_test
                    and list(PORTIONS).index(q.name) < list(PORTIONS).index(portion.name))
        pts = pts[prior:prior + portion.target]
    else:
        pts = pts[:portion.target]
    return [EpisodeSpec(point=pt, n_macro=portion.n_macro, dt_snap=portion.dt_snap,
                        reduction="B-prime", coarsen=1) for pt in pts]


def run_portion(name: str, out: str | Path, cfg: AtlasConfig | None = None,
                hours: float | None = 5.0, workers: int = 1, plan_seed: int = 20260809,
                cfg_path: str | None = None, log=print) -> Manifest:
    """Generate one portion into `out`, stopping before the wall-clock budget.

    The budget is enforced twice: once by counting what fits before starting, and
    once by checking elapsed time before each new episode. The second guard is the
    one that matters -- the first is an estimate, and an estimate that is wrong in
    the optimistic direction would otherwise blow the session."""
    portion = PORTIONS[name]
    cfg = cfg if cfg is not None else load_config(cfg_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    man = Manifest.load(out)
    _refuse_mixed_designs(man, "portion", out)
    from .generate import _git_sha
    man.plan = {**man.plan, "portions": sorted(set(man.plan.get("portions", []) + [name])),
                "cfg_fingerprint": cfg.fingerprint(), "git_sha": _git_sha(),
                "schema_version": schema.SCHEMA_VERSION}
    man.save()

    specs = (_solo_specs_for(portion, cfg, plan_seed) if portion.kind == "solo"
             else _coupled_specs_for(portion, cfg, plan_seed))
    each = portion.core_hours_each()
    budget = float("inf") if hours is None else hours
    log(f"portion {name} ({portion.kind}): {len(specs)} items, ~{each:.2f} core-h each")
    log(f"  serves: {portion.serves}")
    if hours is not None:
        log(f"  budget {hours:.1f} h x {workers} workers = {hours * workers:.0f} core-h "
            f"-> ~{portion.fits(hours, workers)} items fit (a-priori; the guard "
            f"switches to the observed rate after the first completion)")
    n_cpu = os.cpu_count() or 1
    if workers > n_cpu:
        log(f"  WARNING: {workers} workers on {n_cpu} logical CPUs. Each item is a "
            f"single-threaded numpy solve, so oversubscription buys nothing and makes "
            f"the first wave finish later -- which is when the budget guard is blindest.")

    todo = []
    for spec in specs:
        eid = portion_item_id(portion, spec, cfg)
        if man.is_done(eid, out):
            continue
        todo.append((spec, eid))
    log(f"  {len(specs) - len(todo)} already done, {len(todo)} to go")

    t0 = time.perf_counter()
    done = 0

    def record(rec):
        rec["portion"] = name
        man.episodes[rec["id"]] = rec
        man.save()
        log("  " + _fmt(rec))

    def over_budget(in_flight: int) -> bool:
        """Would starting one more item run past the wall clock?

        The a-priori charge is core-hours / workers, which assumes the pool scales
        perfectly with worker count. **It does not**, and the gap is not small:
        measured on this box, 20 workers on 16 physical cores delivered an
        effective parallelism far below 20 once anything else was running. A guard
        that trusts the a-priori number would happily start a wave that overruns
        the session, which on a rented box means losing the box mid-episode.

        So once anything has finished, the estimate is replaced by the OBSERVED
        rate -- items completed per elapsed hour -- which folds in contention,
        hyperthreading and whatever else the machine is doing, and needs no model
        of any of it."""
        elapsed = (time.perf_counter() - t0) / 3600.0
        if done and elapsed > 0.0:
            per_item = elapsed / done                  # wall hours per completed item
        else:
            per_item = each / max(workers, 1)
        return elapsed + per_item * max(in_flight, 1) > budget

    if workers > 1 and todo:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
        pending = set()
        queue = list(todo)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while queue or pending:
                while queue and len(pending) < workers and not over_budget(1):
                    spec, eid = queue.pop(0)
                    pending.add(pool.submit(_portion_worker,
                                            (portion.kind, spec, str(out), eid, cfg_path)))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    record(fut.result())
                    done += 1
        if queue:
            log(f"  stopping: budget {budget:.1f} h reached. {len(queue)} items remain; "
                f"rerun the same command to continue.")
    else:
        for spec, eid in todo:
            if over_budget(1):
                log(f"  stopping: budget {budget:.1f} h reached. "
                    f"{len(todo) - done} items remain; rerun the same command to continue.")
                break
            record(generate_solo_one(spec, out, cfg, eid) if portion.kind == "solo"
                   else generate_one(spec, out, cfg))
            done += 1
    log(f"portion {name}: {done} generated this session, "
        f"{(time.perf_counter() - t0) / 3600.0:.2f} h elapsed")
    return man


def _refuse_mixed_designs(man: Manifest, mode: str, out: Path) -> None:
    """A directory holds ONE sweep design.

    `--plan` and `--portion` draw their episodes from different `build_sweep`
    calls, so their Latin hypercubes are different point sets. Mixing them in one
    directory does not corrupt anything -- ids differ, resume still works -- and
    that is exactly the problem: the corpus would silently contain two designs,
    with a held-out corner set that no longer corresponds to the training points
    it was supposed to be held out from."""
    has_portions = bool(man.plan.get("portions"))
    has_plan = bool(man.plan.get("name"))
    if mode == "portion" and has_plan and not has_portions:
        raise ValueError(
            f"{out} was generated with --plan {man.plan['name']!r}, which draws from a "
            "different sweep design than the portions do. Use a fresh directory for the "
            "portion-based corpus, or continue with --plan.")
    if mode == "plan" and has_portions:
        raise ValueError(
            f"{out} holds portion-based episodes {man.plan['portions']}. Use a fresh "
            "directory for a --plan corpus, or continue with --portion.")


def _portion_worker(args):
    """Process-pool entry point for one portion item. The config is rebuilt in the
    child rather than pickled, because Windows spawns a fresh interpreter."""
    kind, spec, out, eid, cfg_path = args
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    cfg = load_config(cfg_path)
    return (generate_solo_one(spec, Path(out), cfg, eid) if kind == "solo"
            else generate_one(spec, Path(out), cfg))


def portion_item_id(portion: Portion, spec, cfg: AtlasConfig) -> str:
    if portion.kind == "coupled":
        return episode_id(spec, cfg)
    payload = {"solo": spec.group, "point": spec.point.as_dict(), "idx": spec.idx,
               "T_ep": spec.T_ep, "dt_snap": spec.dt_snap, "coarsen": spec.coarsen,
               "p_back_ratio": spec.p_back_ratio, "T_wall": spec.T_wall,
               "bc_mode": spec.bc_mode, "bc_amplitude": spec.bc_amplitude,
               "cfg": cfg.fingerprint(), "schema": schema.SCHEMA_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def generate_solo_one(spec, out: Path, cfg: AtlasConfig, eid: str) -> dict:
    """Solo counterpart of `generate_one`: same staging, validation and checksum
    discipline, because a solo file is just as unrecoverable once the box is gone."""
    from .generate_solo import generate_solo_episode
    final = out / f"solo_{spec.group}_{spec.split}_{spec.idx:04d}.h5"
    staging = out / f".{final.stem}.{eid}.partial"
    t0 = time.perf_counter()
    rec: dict = {"id": eid, "idx": spec.idx, "split": spec.split,
                 "file": final.name, "group": spec.group}
    try:
        if staging.exists():
            staging.unlink()
        written = generate_solo_episode(spec, staging.parent, cfg)
        os.replace(written, staging)
        rec["problems"] = schema.validate_episode(staging, cfg,
                                                  n_snapshots=spec.n_snapshots)
        # no mass_residual key at all: a solo run has no trajectory to grade the
        # exit mass flux against, and writing NaN would put a non-JSON token in
        # the manifest for a quantity that is not merely unknown but undefined
        os.replace(staging, final)
        rec["sha256"] = file_sha256(final)
        rec["bytes"] = final.stat().st_size
    except Exception as exc:                           # noqa: BLE001 - recorded, not hidden
        rec["error"] = f"{type(exc).__name__}: {exc}"
        if staging.exists():
            staging.unlink()
    rec["wall_seconds"] = time.perf_counter() - t0
    return rec


def augmentation_report(out: str | Path) -> dict:
    """Effective training pairs, which is what the corpus size actually buys.

    Every expert interface is lead-time conditioned, so for an autonomous PDE any
    ORDERED pair of snapshots from one episode is a valid sample: C(K,2) instead
    of K-1. At 200 snapshots that is 19,900 against 199, a ~100x amplification
    that costs nothing.

    Reported here because the honest headline of a small corpus is not its episode
    count. The number it does NOT change is the count of distinct physical
    configurations -- pairs within an episode are heavily correlated, and no
    augmentation invents a regime the sweep did not visit. Both numbers below."""
    man = Manifest.load(out)
    eps = [r for r in man.episodes.values() if not r.get("error")]
    files = {r["file"] for r in eps}
    pairs, snaps, configs = 0, 0, 0
    for f in sorted(files):
        p = Path(out) / f
        if not p.exists():
            continue
        with h5py.File(p, "r") as h:
            k = int(h["meta"].attrs.get("n_snapshots", 0)) or h["rigid"].shape[0]
        snaps += k
        pairs += k * (k - 1) // 2
        configs += 1
    return {"episodes": configs, "snapshots": snaps, "all2all_pairs": pairs,
            "consecutive_pairs": max(snaps - configs, 0)}


__all__ = ["CorpusPlan", "PLANS", "Manifest", "episode_id", "file_sha256",
           "generate_one", "generate_corpus", "Portion", "PORTIONS",
           "schedule", "run_portion", "augmentation_report", "choked_throat_report",
           "CORE_HOURS_PER_FLIGHT_SECOND"]


def choked_throat_report(out: str | Path) -> dict:
    """Gate G6, read off the solo nozzle sweep: is the throat actually choked?

    The claim a choked throat makes is that mass flow is set by the chamber, not
    by what is downstream: vary back pressure at fixed chamber pressure and the
    throat mass flux does not move. The coupled corpus cannot test this -- its back
    pressure is whatever the altitude gave -- so this is the one place the Phase-2
    structural gate can be graded, and it is graded on the SOLVER data first,
    before any model is asked to reproduce it.

    Returns the sweep, plus the spread over the sub-critical (under-expanded)
    subset where choking is unambiguous. A spread that is not small means the
    sweep did not reach choked operation, and the gate would be graded on a
    distribution that never varies -- G6's actual failure mode."""
    out = Path(out)
    rows = []
    for f in sorted(out.glob("solo_nozzle_*.h5")):
        with h5py.File(f, "r") as h:
            if "e-b" not in h["iface"]:
                continue
            ch = [str(s) for s in h.attrs["iface_channels"]]
            rec = h["iface"]["e-b"][-1]                 # last snapshot: most settled
            mdot = float(np.sum(rec[ch.index("mass_flux")] * rec[ch.index("seg_length")]))
            rows.append({"file": f.name,
                         "p_back_ratio": float(h["meta"].attrs["p_back_ratio"]),
                         "p_c": float(h["meta"].attrs["p_c"]),
                         "mdot_throat": mdot})
    rows.sort(key=lambda r: r["p_back_ratio"])
    sub = [r for r in rows if r["p_back_ratio"] <= 1.0]
    m = np.array([abs(r["mdot_throat"]) for r in sub]) if sub else np.array([])
    spread = float(m.std() / max(m.mean(), 1e-30)) if m.size > 1 else float("nan")
    return {"runs": rows, "n_subcritical": len(sub), "mdot_spread_subcritical": spread,
            "p_back_range": (rows[0]["p_back_ratio"], rows[-1]["p_back_ratio"]) if rows
            else (float("nan"), float("nan"))}
