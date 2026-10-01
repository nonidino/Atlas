"""The learned case in the page (demo item 1.5): one fixed wind farm, four arms.

The case is farm-12, the layout the gate held out of training (`learned_gate`),
opened read-only: its experts were trained for this farm, so nothing about it can
be changed.  Run marches the arms the gate names, in turns, as every workbench run
does:

  ``parallel``  the classical decomposition, 24 windows on threads (the gate's Ep)
  ``full``      the full domain (F)
  ``learned``   the same decomposition with every window stepped by the trained
                network (L); the blend, the global projection and the band classical
  ``coarse``    the classical decomposition on a grid twice as coarse (Cc)

and reads the truth ``T`` (the full domain at twice the resolution) from its stored
run, `out/learned-case/truth-farm-12.npz`.  After the march each arm is measured
against T at the horizon the page runs to (12 macro-steps, the demo's; T is stored
at 12 and 40), exactly as the gate measures it, and the card shows the registered
evaluation's verdict beside this run's numbers (`runview.learned_card`).

**What needs what.**  The classical arms need nothing new.  The learned arm needs
torch and the trained weights, `out/learned-case/window-net.pt`; without either it
is not offered, and the reason is said.  The measures need T's stored run.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np

from . import learned_gate as G
from .checks import CheckResult, CheckSpec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.path.join(ROOT, "out", "learned-case")
WEIGHTS = os.path.join(DIR, "window-net.pt")
TRUTH = os.path.join(DIR, "truth-%s.npz" % G.CASE)

#: the example's key, and the case file's marker (a physics parameter: the case
#: file has no field for it, and a parameter the family ignores is harmless)
KEY = "learned-farm"
MARKER = "learned_case"

ARMS = ("parallel", "full", "learned", "coarse")
ARM_LABELS = {"learned": "Learned experts, parallel",
              "coarse": "Classical, grid twice as coarse"}

#: the caveat the card always shows (demo-learned-case-plan section 7)
CAVEAT = ("Trained on this farm's family of layouts. Experts that work on any shape are "
          "what the proposal builds.")
READ_ONLY = "This case is fixed: its experts were trained for this farm."

REGISTERED = "2026-10-01, the gate's registration (out/learned-case/registered.txt)"
CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(key="accuracy_power", title="Learned: farm power as accurate as the classical",
              kind="reference", tolerance=G.G2_FACTOR, registered=REGISTERED,
              measure="e_P(L) / max(e_P(F), e_P(Ep)), each against T, farm power averaged "
                      "over the last five macro-steps",
              why="G2 of the gate: within 10% of the worse classical arm"),
    CheckSpec(key="accuracy_velocity", title="Learned: velocity as accurate as the classical",
              kind="reference", tolerance=G.G2_FACTOR, registered=REGISTERED,
              measure="e_V(L) / max(e_V(F), e_V(Ep)), rms against T on the D/16 grid",
              why="G2 of the gate"),
    CheckSpec(key="mass", title="Learned: incompressibility closes", kind="balance",
              tolerance=G.G3_DIV, registered=REGISTERED,
              measure="the largest divergence after the global projection, times dx / U",
              why="G3 of the gate: the projection is classical, so round-off"),
    CheckSpec(key="competitor", title="The coarse classical decomposition does not match it",
              kind="control", tolerance=None, registered=REGISTERED,
              measure="Cc's error against T exceeds L's in farm power or in velocity",
              why="G5 of the gate: otherwise a cheaper classical solver does the job"),
)


def is_learned(spec) -> bool:
    return bool(float(spec.physics.params.get(MARKER, 0.0)))


def torch_available() -> str | None:
    """None if torch imports, else why not."""
    try:
        import torch  # noqa: F401
        return None
    except Exception as exc:                          # ImportError, or a DLL that will not load
        return "%s: %s" % (type(exc).__name__, exc)


def available_arms(spec) -> tuple[tuple[str, ...], dict[str, str]]:
    why: dict[str, str] = {}
    t = torch_available()
    if t is not None:
        why["learned"] = "the learned arm needs torch, which is not installed here (%s)" % t
    elif not os.path.isfile(WEIGHTS):
        why["learned"] = "the trained weights are not in this copy (%s)" % os.path.relpath(
            WEIGHTS, ROOT)
    return tuple(a for a in ARMS if a not in why), why


def gate_record() -> dict | None:
    """The registered evaluation's record: the newest gate record that is not a smoke
    test, or None."""
    best = None
    for p in sorted(glob.glob(os.path.join(DIR, "gate-*.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not r.get("smoke"):
            best = dict(r, path=os.path.relpath(p, ROOT).replace(os.sep, "/"))
    return best


class LearnedCase:
    """The runner's adapter for the learned case: one run object per arm."""

    family = "incompressible-2d"
    style = "A"

    def __init__(self, spec, arms=ARMS, threads: int = 4):
        from . import learned_arms as LA
        self.spec = spec
        self.arms = tuple(a for a in ARMS if a in arms)
        self.threads = int(threads)
        self.runs: dict[str, Any] = {}
        classical = tuple(a for a in ("parallel", "full") if a in self.arms)
        if classical:
            fine = LA.FarmRun(spec, arms=classical, threads=self.threads)
            for a in classical:
                self.runs[a] = (fine, a)
        if "learned" in self.arms:
            from .learned_net import load_window_net
            net = load_window_net(WEIGHTS)
            self.runs["learned"] = (LA.LearnedFarmRun(spec, net, arms=("learned",),
                                                      threads=self.threads), "learned")
        if "coarse" in self.arms:
            self.runs["coarse"] = (LA.FarmRun(LA.scaled_spec(spec, 0.5), arms=("parallel",),
                                              threads=self.threads), "parallel")
        any_run = next(iter(self.runs.values()))[0]
        self.u_inf, self.dx = any_run.u_inf, float(spec.domain.dx)
        self.truth = np.load(TRUTH) if os.path.isfile(TRUTH) else None

    # -- the runner's interface ---------------------------------------------------

    def initial(self, arm: str):
        run, a = self.runs[arm]
        return run.initial(a)

    def step(self, arm: str, s):
        run, a = self.runs[arm]
        return run.step(a, s)

    def observe(self, arm: str, s, prev=None) -> dict[str, float]:
        run, a = self.runs[arm]
        return {"power": run.power(s.rec), "mass": run.mass_measure(a, s)}

    def bitwise_equal(self, a, b) -> bool:              # pragma: no cover - never asked
        return bool(np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v))

    def field(self, s) -> np.ndarray:
        """The streamwise velocity on the page's grid: the coarse arm's is shown on it
        cell by cell (each coarse cell over the four fine cells it covers)."""
        u = s.u
        if u.shape[0] * 2 == self.spec.domain.ny:
            u = np.kron(u, np.ones((2, 2)))
        return u

    def field_label(self) -> str:
        return "streamwise velocity u / U"

    def compare(self, states, history, bitwise):
        done = min(len(history[a]) for a in self.arms) if self.arms else 0
        errors: dict[str, dict[str, float]] = {}
        why = None
        if self.truth is None:
            why = "the truth T is not in this copy"
        elif done not in G.HORIZONS:
            why = ("T is stored at %s macro-steps and this run stopped at %d"
                   % (" and ".join(str(h) for h in G.HORIZONS), done))
        else:
            T = self.truth
            for a in self.arms:
                run, _ = self.runs[a]
                cells = int(round(1.0 / run.dx))
                power = [h["power"] for h in history[a]]
                errors[a] = {"P": G.error_power(power, T["power"], done),
                             "V": G.error_velocity((states[a].u, states[a].v), cells,
                                                   (T["u%d" % done], T["v%d" % done]),
                                                   G.COMPARE_CELLS_PER_D)}
        metrics: dict[str, Any] = {"errors_vs_truth": errors, "horizon": done,
                                   "not_measured": why}
        for a in self.arms:
            metrics[a] = {"power_last": history[a][-1]["power"] if history[a] else None}
        checks = []
        have = set(errors)
        if {"learned", "parallel", "full"} <= have:
            for m, spec_ in (("P", CHECKS[0]), ("V", CHECKS[1])):
                ref = max(errors["full"][m], errors["parallel"][m])
                v = errors["learned"][m] / ref if ref > 0 else None
                checks.append(CheckResult(spec_, v, None if v is None else bool(v <= spec_.tolerance),
                                          "L %.3e against the worse classical %.3e"
                                          % (errors["learned"][m], ref)))
        else:
            for spec_ in CHECKS[:2]:
                checks.append(CheckResult(spec_, None, None, why or "needs L, F and Ep"))
        if "learned" in self.arms and history["learned"]:
            mx = max(h["mass"] for h in history["learned"])
            checks.append(CheckResult(CHECKS[2], mx, bool(mx <= CHECKS[2].tolerance)))
        else:
            checks.append(CheckResult(CHECKS[2], None, None, "the learned arm did not run"))
        if {"learned", "coarse"} <= have:
            better = [m for m in ("P", "V") if errors["coarse"][m] > errors["learned"][m]]
            checks.append(CheckResult(CHECKS[3], None, bool(better),
                                      "L is closer to T in %s" % (" and ".join(better))
                                      if better else "Cc is as close to T as L, or closer, "
                                      "in both measures: a classical solver on a grid twice "
                                      "as coarse matches it"))
        else:
            checks.append(CheckResult(CHECKS[3], None, None, why or "needs L and Cc"))
        return metrics, checks

    def notes(self, done: int) -> list[str]:
        return [
            "T is the full domain at cells of D/64, run once and stored; every arm is "
            "measured against it on the grid of D/16, as the gate measures.",
            "The learned arm's network (learned_net.WindowUNet, width %d, depth %d) was "
            "trained on %d layouts of %d to %d rotors that are not this one, and its "
            "weights chosen by %d-macro-step rollouts on 3 of %d validation layouts; this "
            "farm's layout was held out (the gate's G6)."
            % (G.NET_WIDTH, G.NET_DEPTH, len(G.TRAIN_SEEDS), G.N_ROTORS[0], G.N_ROTORS[1],
               G.STEPS, len(G.VALIDATION_SEEDS)),
            CAVEAT,
        ]

    def describe(self) -> dict[str, Any]:
        return {"case": KEY, "arms": list(self.arms), "weights": os.path.relpath(
            WEIGHTS, ROOT).replace(os.sep, "/"), "truth": os.path.relpath(TRUTH, ROOT).replace(
            os.sep, "/")}

    def close(self) -> None:
        seen = set()
        for run, _a in self.runs.values():
            if id(run) not in seen:
                seen.add(id(run))
                run.close()


__all__ = ["KEY", "MARKER", "ARMS", "ARM_LABELS", "CAVEAT", "READ_ONLY", "CHECKS",
           "is_learned", "available_arms", "gate_record", "LearnedCase", "WEIGHTS", "TRUTH"]
