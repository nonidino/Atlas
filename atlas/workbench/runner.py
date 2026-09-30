"""Run a committed case: the arms take turns, are timed, streamed, stoppable, recorded.

The workbench's runner, plan step 3 of [[outcome-c4-path-to-declarative-cases]].
It knows nothing about physics.  A family's adapter (`families/`) prepares the
case and steps one arm at a time; this module decides the order, holds the
timer, publishes what the page draws, honours Stop, and writes the record.

**Taking turns.**  Every macro-step, each arm takes one step, and the order
rotates by one arm per macro-step, so no arm always runs first (a warm cache or
a thermal state favours whoever follows the same predecessor).  All arms share
the machine and the moment, which is what makes their ratio a measurement.

**What the timer holds.**  Only the adapter's ``step``: the diagnostics
(``observe``), the bitwise comparison of the two decomposed arms, the snapshot
the page draws and the record are all taken after the timer stops, so no arm is
charged for its instrument (the vault's memory "An instrument must not run the
arm it prices").

**One run at a time per server process.**  Two runs on one machine would time
each other, so a second Run waits for nothing and is refused with its reason.

**The record.**  A JSON file per run, beside the case file (``<case>.results/``),
with the case as it marched, the machine's state before and after (power source,
other Python processes), every per-step time, and each check with its registered
tolerance.  Numbers in the page are read from this object, never retyped.
"""

from __future__ import annotations

import datetime
import importlib
import json
import math
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import machine
from . import registry

RESULTS_SCHEMA = "atlas-workbench/results@1"

ARM_LABELS = {"serial": "Decomposed, serial", "parallel": "Decomposed, parallel",
              "full": "Full domain"}

#: The page draws fields at most this many cells across.
DISPLAY_CELLS = 320

_ACTIVE = threading.Lock()


class RunRefused(RuntimeError):
    """The case cannot be run as it stands; the message says why, for a person."""


def adapter_for(family: str):
    fam = registry.family(family)
    if not fam.adapter:
        raise RunRefused(f"the {fam.label} family has no runner yet: {fam.note}")
    return importlib.import_module(fam.adapter)


def arms_for(module, spec) -> tuple[tuple[str, ...], dict[str, str]]:
    """The arms a case can run, in the family's order, and why the others cannot
    (an adapter says so through ``available_arms``; otherwise all of its ARMS)."""
    if hasattr(module, "available_arms"):
        return module.available_arms(spec)
    return tuple(module.ARMS), {}


def arm_labels_for(module) -> dict[str, str]:
    """The arms' names on the page: the runner's, unless the family names its own
    (a split by physics has no windows: its arms are a synchronous split, a
    lagged split and the unsplit solver)."""
    return {**ARM_LABELS, **dict(getattr(module, "ARM_LABELS", None) or {})}


def step_label_for(module, spec) -> str:
    """What one step of a run is: a macro-step, or a timed repeat of a steady solve."""
    if hasattr(module, "step_label"):
        return module.step_label(spec)
    return "macro-step"


def downsample(f: np.ndarray, cells: int = DISPLAY_CELLS) -> tuple[np.ndarray, int]:
    s = max(1, int(math.ceil(max(f.shape) / cells)))
    return np.ascontiguousarray(f[::s, ::s], dtype=np.float32), s


@dataclass
class Progress:
    """A copy of what the page reads, taken under the run's lock."""

    status: str
    step: int
    steps: int
    arm: str
    seconds: dict[str, list[float]]
    series: dict[str, dict[str, list[float]]]
    fields: dict[str, np.ndarray]
    field_version: int
    stride: int
    message: str
    error: str | None
    results: dict[str, Any] | None
    #: per decomposed arm, the latest step's iteration history (styles B, C, D)
    convergence: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class CaseRun:
    """One run of one committed case, in a background thread or the caller's."""

    spec: Any
    arms: tuple[str, ...]
    steps: int
    threads: int = 4
    #: where the record goes; None writes nothing (tests)
    results_dir: str | None = None
    case_path: str | None = None
    #: publish the fields this often, in macro-steps (and always at the end)
    snapshot_every: int = 1
    on_finish: Callable[["CaseRun"], None] | None = None
    #: what the record calls the case: the workbench's key for it (a case file has
    #: no name since case@0.6), else its kind
    case_label: str = ""

    status: str = field(default="created", init=False)
    committed_at: str = field(default="", init=False)
    committed_json: str = field(default="", init=False)

    def __post_init__(self):
        self.spec = self.spec.copy_deep()                 # nothing edited later reaches it
        self.committed_json = self.spec.to_json()
        self.committed_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self.module = adapter_for(self.spec.physics.family)
        order, why = arms_for(self.module, self.spec)
        asked = tuple(self.arms)
        self.arms = tuple(a for a in order if a in asked)
        #: arms asked for that this case cannot run, with the reason
        self.dropped = {a: why.get(a, "not an arm of this family") for a in asked
                        if a not in self.arms}
        if not self.arms:
            raise RunRefused("none of the chosen arms can run this case"
                             + ("".join(f"; {a}: {w}" for a, w in self.dropped.items())))
        self.steps = int(self.steps)
        self.step_label = step_label_for(self.module, self.spec)
        self.arm_labels = arm_labels_for(self.module)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._step = 0
        self._arm = ""
        self._seconds: dict[str, list[float]] = {a: [] for a in self.arms}
        self._series: dict[str, dict[str, list[float]]] = {a: {} for a in self.arms}
        self._convergence: dict[str, list[float]] = {}
        self._fields: dict[str, np.ndarray] = {}
        self._field_version = 0
        self._stride = 1
        self._message = "waiting to start"
        self._error: str | None = None
        self.results: dict[str, Any] | None = None
        self.record_path: str | None = None
        self.error_trace: str | None = None

    # -- control ------------------------------------------------------------

    def start(self) -> None:
        #: active from this moment, not from when the thread first runs, so a page
        #: built right after Run already shows Stop enabled
        self._set(status="starting", _message="starting")
        self._thread = threading.Thread(target=self._run, name="workbench-run", daemon=True)
        self._thread.start()

    def run_blocking(self) -> "CaseRun":
        self._run()
        return self

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            if self.status == "running":
                self.status = "stopping"
                self._message = ("stopping after the arm-step in progress (a full-domain "
                                 "step cannot be interrupted part-way)")

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running", "stopping")

    def progress(self) -> Progress:
        with self._lock:
            return Progress(self.status, self._step, self.steps, self._arm,
                            {a: list(v) for a, v in self._seconds.items()},
                            {a: {k: list(x) for k, x in d.items()}
                             for a, d in self._series.items()},
                            dict(self._fields), self._field_version, self._stride,
                            self._message, self._error, self.results,
                            {a: list(v) for a, v in self._convergence.items()})

    def label(self) -> str:
        s = self.spec
        bits = [f"{len(s.windows)} window{'s' * (len(s.windows) != 1)}"]
        if s.devices:
            bits.append(f"{len(s.devices)} rotors")
        bits.append(f"{self.steps} {self.step_label}{'s' * (self.steps != 1)}")
        return (f"the {registry.short_label(s.physics.family)} case as committed at "
                f"{self.committed_at[11:19]} ("
                + ", ".join(bits) + f"; arms: {', '.join(self.arms)}; "
                + f"{self.threads} thread{'s' * (self.threads != 1)})")

    # -- the march ------------------------------------------------------------

    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _run(self) -> None:
        if not _ACTIVE.acquire(blocking=False):
            self._set(status="failed", _error=("another case is running in this workbench "
                                               "server; two runs on one machine would time "
                                               "each other. Stop it or wait for it."),
                      _message="not started")
            self._finish()
            return
        problem = None
        awake = machine.keep_awake(True)
        started = time.time()
        try:
            self._set(status="starting", _message="reading the machine's state")
            before = machine.machine_record()
            self._set(_message="building the solvers")
            t0 = time.perf_counter()
            problem = self.module.build(self.spec, arms=self.arms, threads=self.threads)
            build_s = time.perf_counter() - t0
            states = {a: problem.initial(a) for a in self.arms}
            history: dict[str, list[dict]] = {a: [] for a in self.arms}
            bitwise = {"steps": 0, "first_difference": None}
            self._set(status="running", _message="marching")
            done = 0
            for k in range(self.steps):
                order = self.arms[k % len(self.arms):] + self.arms[:k % len(self.arms)]
                #: steps return new states, so the round's start is kept by reference
                #: and restored if Stop lands part-way through the round
                round_start = dict(states)
                stepped = []
                for a in order:
                    if self._stop.is_set():
                        break
                    self._set(_arm=a)
                    t1 = time.perf_counter()
                    st = problem.step(a, states[a])
                    dt = time.perf_counter() - t1
                    states[a] = st
                    stepped.append(a)
                    with self._lock:
                        self._seconds[a].append(dt)
                if len(stepped) < len(self.arms):
                    states = round_start                   # a partial macro-step is dropped
                    break
                done = k + 1
                same = None
                if "serial" in states and "parallel" in states:
                    same = problem.bitwise_equal(states["serial"], states["parallel"])
                    bitwise["steps"] = done
                    if not same and bitwise["first_difference"] is None:
                        bitwise["first_difference"] = done
                # the instruments, after the round and outside every timer.  Two
                # states just shown equal to the bit are observed once: the same
                # arithmetic on the same numbers cannot read differently
                for a in self.arms:
                    if a == "serial" and same and "parallel" in states:
                        continue
                    obs = problem.observe(a, states[a], round_start[a])
                    history[a].append(obs)
                if same and "parallel" in states:
                    history["serial"].append(dict(history["parallel"][-1],
                                                  observed_as="parallel"))
                with self._lock:
                    for a in self.arms:
                        for key, val in history[a][-1].items():
                            if isinstance(val, bool):
                                val = float(val)
                            if isinstance(val, (int, float)):
                                self._series[a].setdefault(key, []).append(float(val))
                            elif key == "convergence":
                                #: the latest step's iteration history, for the page
                                self._convergence[a] = [float(x) for x in val]
                if done % self.snapshot_every == 0 or done == self.steps:
                    self._publish_fields(problem, states, done)
                with self._lock:
                    self._step = done
                if self._stop.is_set():
                    break
            # a stopped run keeps only whole macro-steps: every arm the same length
            for a in self.arms:
                del history[a][done:]
                with self._lock:
                    del self._seconds[a][done:]
            stopped = self._stop.is_set() and done < self.steps
            self._set(_message="comparing")
            metrics, checks = problem.compare(states, history, bitwise) if done else ({}, [])
            diff = field_differences(problem, states) if done else {}
            self.results = self._record(problem, before, build_s, done, stopped, metrics,
                                        checks, bitwise, started, diff)
            self._publish_fields(problem, states, done)
            unit = self.step_label + "s"
            self._set(status="stopped" if stopped else "done",
                      _message=(f"stopped by the user after {done} of {self.steps} "
                                f"{unit}" if stopped else f"finished {done} {unit}"))
        except Exception as exc:                          # the page says what broke
            self.error_trace = traceback.format_exc()
            self._set(status="failed", _error=f"{type(exc).__name__}: {exc}",
                      _message="the run failed")
        finally:
            if problem is not None:
                try:
                    problem.close()
                except Exception:                         # pragma: no cover
                    pass
            if awake:
                machine.keep_awake(False)
            _ACTIVE.release()
            self._finish()

    def _finish(self) -> None:
        if self.on_finish is not None:
            try:
                self.on_finish(self)
            except Exception:                             # pragma: no cover
                pass

    def _publish_fields(self, problem, states, done: int) -> None:
        if done == 0:
            return
        f: dict[str, np.ndarray] = {}
        stride = 1
        for a in self.arms:
            f[a], stride = downsample(problem.field(states[a]))
        dec = "parallel" if "parallel" in f else ("serial" if "serial" in f else None)
        if dec and "full" in f:
            full_d = problem.field(states[dec]) - problem.field(states["full"])
            f["difference"], _ = downsample(full_d)
        with self._lock:
            self._fields = f
            self._field_version += 1
            self._stride = stride

    # -- the record -------------------------------------------------------------

    def _record(self, problem, before, build_s, done, stopped, metrics, checks, bitwise,
                started, field_difference=None) -> dict[str, Any]:
        timing: dict[str, Any] = {}
        for a in self.arms:
            s = self._seconds[a][:done]
            timing[a] = {"seconds_per_step": s,
                         "mean": float(np.mean(s)) if s else None,
                         "median": float(np.median(s)) if s else None,
                         "min": float(np.min(s)) if s else None}
        if "full" in timing and timing["full"]["mean"]:
            for a in self.arms:
                if timing[a]["mean"]:
                    timing[a]["speedup_vs_full"] = timing["full"]["mean"] / timing[a]["mean"]
        rec = {
            "schema": RESULTS_SCHEMA,
            "case_name": self.case_label or registry.short_label(self.spec.physics.family),
            "case_path": self.case_path,
            "family": self.spec.physics.family,
            "style": self.spec.coupling.style,
            "committed_at": self.committed_at,
            "started": datetime.datetime.fromtimestamp(started).astimezone().isoformat(
                timespec="seconds"),
            "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "wall_seconds": time.time() - started,
            "build_seconds": build_s,
            "arms": list(self.arms),
            "arm_labels": {a: self.arm_labels[a] for a in ARM_LABELS},
            "dropped_arms": dict(self.dropped),
            "step_label": self.step_label,
            "steps_requested": self.steps,
            "steps_done": done,
            "stopped": stopped,
            "threads": self.threads,
            "timing": timing,
            "timing_note": ("mean of the per-macro-step wall time of the adapter's step "
                            "alone, arms taking turns in a rotating order, inside the "
                            "workbench server process"),
            "bitwise": bitwise,
            "metrics": metrics,
            #: every arm's final field against the full domain's (rms, largest)
            "field_difference": field_difference or {},
            "field_label": getattr(self.module, "FIELD_LABEL", "field"),
            "checks": [c.as_dict() for c in checks],
            "notes": problem.notes(done) if hasattr(problem, "notes") else [],
            "problem": problem.describe(),
            "machine": before,
            "machine_after": {"power": machine.power_status(),
                              "time": datetime.datetime.now().astimezone().isoformat(
                                  timespec="seconds")},
            "case": json.loads(self.committed_json),
        }
        if self.results_dir:
            self.record_path = write_record(rec, self.results_dir)
            rec["record_path"] = self.record_path
        return rec


# ---------------------------------------------------------------------------
# where records go
# ---------------------------------------------------------------------------


def results_dir_for(case_path: str | None, cases_dir: str, name: str) -> str:
    """``<case>.results/`` beside a saved case; beside where it would be saved otherwise
    (``name`` is the workbench's key for the case: `app.Workbench.case_key`)."""
    from .spec import slug
    if case_path:
        stem = os.path.splitext(os.path.abspath(case_path))[0]
        return stem + ".results"
    return os.path.join(cases_dir, slug(name) + ".results")


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def field_differences(problem, states) -> dict[str, dict[str, float]]:
    """Each other arm's final field against the full domain's: the rms and the largest
    difference, and the rms over the full field's own rms (the family's displayed
    field, `FIELD_LABEL`).  Empty when the full domain did not run."""
    if "full" not in states:
        return {}
    ref = np.asarray(problem.field(states["full"]), dtype=float)
    #: a drawn domain's field is NaN outside it; the statistics are the domain's
    ok = np.isfinite(ref)
    whole = bool(ok.all())
    r = ref if whole else ref[ok]
    scale = float(np.sqrt(np.mean(r * r)))
    out = {}
    for a, s in states.items():
        if a == "full":
            continue
        dlt = np.asarray(problem.field(s), dtype=float) - ref
        if not whole:
            dlt = dlt[ok]
        rms = float(np.sqrt(np.mean(dlt * dlt)))
        out[a] = {"rms": rms, "max": float(np.max(np.abs(dlt))),
                  "rms_relative": rms / scale if scale > 0.0 else None}
    return out


def write_record(rec: dict, folder: str, prefix: str = "") -> str:
    """Atomically, with OneDrive's retry (it can hold a just-written file)."""
    os.makedirs(folder, exist_ok=True)
    stamp = prefix + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(folder, f"{stamp}.json")
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(folder, f"{stamp}-{n}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(rec), fh, indent=1)
    for k in range(40):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if k == 39:
                raise
            time.sleep(0.25)
    return path


__all__ = ["RESULTS_SCHEMA", "ARM_LABELS", "RunRefused", "CaseRun", "Progress",
           "adapter_for", "arm_labels_for", "downsample", "results_dir_for", "write_record"]
