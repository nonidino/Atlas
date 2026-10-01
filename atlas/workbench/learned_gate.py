"""The learned case's gate (demo item 1.5), REGISTERED BEFORE ANY DATA EXISTS.

`demo-learned-case-plan` section 5 drafted the gate; this module is its
registration, committed before the first training sample was generated, so the
commit that adds it is the proof of the order.  `out/learned-case/registered.txt`
is its human-readable copy, written by ``scripts/learned_register.py`` with this
file's hash.

**What is being tested.**  On the wind farm, a small network trained for the slot
(`learned_net.WindowUNet`) replaces the classical window's step inside the
decomposed march; the blend, the one global projection and the freestream band
stay classical.  The owner asked for one case where learned experts are as
accurate as the classical decomposition AND the classical full domain, and
faster.  The gate below asks exactly that, against a truth run at twice the
resolution, and asks one more thing: that a classical solver on a grid twice as
coarse, which is also cheaper, does NOT do the job as well (G5).  Any of the six
can fail, and the page reports whatever the evaluation measures.

**The arms** (all from the uniform freestream, 40 macro-steps, the page's case):

  ``F``   the full domain, cells of D/32, classical (the workbench's ``full``)
  ``Ep``  the classical decomposition, 24 windows on 4 threads (``parallel``)
  ``L``   the same decomposition with every window stepped by the network,
          torch at the same 4 threads; blend, projection and band classical
  ``Cc``  the classical decomposition on a grid twice as coarse (cells of D/16,
          windows of 64^2, the same tiling halved, 4 threads)
  ``T``   the full domain at cells of D/64, run once and stored, not timed

**The measures**, at the two registered horizons (the demo's 12 macro-steps and
W346's 40):

  e_P(arm, h)  |P_arm - P_T| / P_T, farm power averaged over macro-steps h-4..h
  e_V(arm, h)  the rms over the domain of the velocity difference to T at step h,
               both components, over U, on the grid of cells of D/16 (every arm
               block-averaged to it; T by 4 x 4, F, Ep and L by 2 x 2, Cc as it is)
  E(arm, n)    the fluctuation energy (1/2) sum((u - U)^2 + v^2) dx^2 at step n
  div(arm, n)  the workbench's own mass measure: the divergence after the global
               projection, times dx / U, in the projection's own operator

**The split (G6).**  The page's layout, farm-12's twelve rotors with induction
1/3, is held out.  Training uses `TRAIN_SEEDS` and model selection uses
`VALIDATION_SEEDS`, each a random layout drawn by `sample_layout`, and every one
is refused if it is too close to the held-out layout (`held_out_ok`).  The draft
said the page's "layout and inflow" are held out: this family's freestream band
holds the inflow at (U, 0) in every case, so the inflow cannot be varied, and it
is the page's in every trajectory.  The layout is held out; the inflow is not;
the card says so.

**Evaluated once.**  The held-out case is marched with the final weights once,
after training ends; the weights are chosen on the validation layouts alone.  If
G2 or G4 fails, design L-B (a learned correction on a coarse window) is tried
once, under this same gate, and its result is reported whatever it is.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

REGISTERED = ("2026-10-01, before any training data was generated (the commit adding "
              "this file precedes the first sample)")

#: the case the page shows, held out of training
CASE = "farm-12"
HORIZONS = (12, 40)
STEPS = max(HORIZONS)
THREADS = 4

#: the expert, sized by step 0 before any data existed
#: (out/learned-case/step0-20261001-143036.json): widths 8, 12 and 16 met the
#: registered budget t_L <= t_Ep / 1.5 (2.60x, 1.93x, 1.58x) and 20 and 24 did
#: not; 12 is chosen, leaving margin for the gate run's own noise
NET_WIDTH = 12
NET_DEPTH = 3

# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

G1_EP = 1.5          # t_Ep / t_L at least
G1_F = 3.0           # t_F / t_L at least
G2_FACTOR = 1.1      # e_L <= 1.1 max(e_F, e_Ep), every measure, both horizons
G3_DIV = 1e-9        # L's mass measure at every macro-step
G4_REL = 0.10        # |E_L - E_Ep| <= 0.1 E_Ep at every macro-step 1..40
POWER_WINDOW = 5     # farm power averaged over the last five macro-steps to h
COMPARE_CELLS_PER_D = 16


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    bar: str


GATE: tuple[Gate, ...] = (
    Gate("G1", "speed",
         "t_Ep / t_L >= 1.5 and t_F / t_L >= 3: mean macro-step times over the 40-step "
         "evaluation, arms taking turns, measured on AC power (a run on battery does not "
         "count for G1)"),
    Gate("G2", "accuracy",
         "e_L <= 1.1 max(e_F, e_Ep), in farm power and in rms velocity, at 12 and at 40 "
         "macro-steps: four comparisons, all must hold"),
    Gate("G3", "conservation",
         "L's divergence after the global projection, times dx / U, <= 1e-9 at every "
         "macro-step (the workbench's mass measure)"),
    Gate("G4", "stability",
         "|E_L(n) - E_Ep(n)| <= 0.1 E_Ep(n) at every macro-step n = 1..40"),
    Gate("G5", "the competitor",
         "at 12 and at 40 macro-steps, Cc's error against T exceeds L's in at least one "
         "of the two measures; otherwise a classical solver on a grid twice as coarse "
         "does the job"),
    Gate("G6", "held out",
         "farm-12's layout is in no training or validation trajectory (held_out_ok); the "
         "inflow is the family's fixed (U, 0) in every trajectory, and is not held out"),
)

# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

#: the domain of farm-12, in D: 688 x 464 cells of D/32
DOMAIN_D = (688 / 32.0, 464 / 32.0)
#: the page's rotors: (x, y, induction a), all a = 1/3 (the disk's default)
HELD_OUT: tuple[tuple[float, float, float], ...] = tuple(
    (x, y, 1.0 / 3.0) for x, y in (
        (3.75, 1.75), (7.25, 1.75), (14.25, 1.75), (17.75, 1.75), (7.25, 5.25),
        (17.75, 5.25), (3.75, 8.75), (7.25, 8.75), (14.25, 8.75), (17.75, 8.75),
        (7.25, 12.25), (17.75, 12.25)))
TRAIN_SEEDS: tuple[int, ...] = tuple(range(1000, 1100))
VALIDATION_SEEDS: tuple[int, ...] = tuple(range(2000, 2010))

#: what a random layout may be
N_ROTORS = (8, 16)                 # inclusive
X_RANGE = (2.0, 18.0)              # rotor plane, in D: 2 D from the inlet band, 3.5 D
Y_RANGE = (1.5, 13.0)              #   from the outlet; disk centres 1.5 D from the laterals
MIN_SPACING = 2.5                  # centre to centre, in D
A_RANGE = (0.20, 1.0 / 3.0)        # induction: C_T' from 1.0 to 2.0
#: a training layout is too close to the held-out one if more than this many of
#: the held-out rotors have one of its rotors within NEAR of them
NEAR, MAX_NEAR = 0.5, 3


@dataclass(frozen=True)
class Layout:
    seed: int
    #: (x, y, a) per rotor, in D, in drawing order
    rotors: tuple[tuple[float, float, float], ...]


def held_out_ok(rotors) -> bool:
    """At most `MAX_NEAR` of the held-out rotors have a rotor of ``rotors`` within
    `NEAR` D of them."""
    pts = np.array([(x, y) for x, y, _a in rotors], dtype=float)
    near = 0
    for hx, hy, _ in HELD_OUT:
        if pts.size and np.min(np.hypot(pts[:, 0] - hx, pts[:, 1] - hy)) <= NEAR:
            near += 1
    return near <= MAX_NEAR


def sample_layout(seed: int) -> Layout:
    """A random layout, deterministic in ``seed``: the rotor count, then each rotor
    by rejection (spacing), then each rotor's induction; a whole draw too close to
    the held-out layout is redrawn from the same generator."""
    rng = np.random.default_rng(int(seed))
    while True:
        n = int(rng.integers(N_ROTORS[0], N_ROTORS[1] + 1))
        pts: list[tuple[float, float]] = []
        tries = 0
        while len(pts) < n and tries < 10000:
            tries += 1
            x = float(rng.uniform(*X_RANGE))
            y = float(rng.uniform(*Y_RANGE))
            if all(math.hypot(x - px, y - py) >= MIN_SPACING for px, py in pts):
                pts.append((x, y))
        a = rng.uniform(*A_RANGE, size=len(pts))
        rotors = tuple((round(x, 4), round(y, 4), round(float(ak), 4))
                       for (x, y), ak in zip(pts, a))
        if len(rotors) == n and held_out_ok(rotors):
            return Layout(int(seed), rotors)


def split_text() -> str:
    """Every layout of the split, one per line: what registered.txt records."""
    lines = ["held-out  farm-12  " + "  ".join("(%.2f,%.2f,%.4f)" % r for r in HELD_OUT)]
    for name, seeds in (("train", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS)):
        for s in seeds:
            lay = sample_layout(s)
            lines.append("%-10s %d  %d rotors  %s" % (
                name, s, len(lay.rotors),
                "  ".join("(%.4f,%.4f,%.4f)" % r for r in lay.rotors)))
    return "\n".join(lines)


def split_hash() -> str:
    return hashlib.sha256(split_text().encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# the measures and the judgement
# ---------------------------------------------------------------------------


def block_mean(f: np.ndarray, k: int) -> np.ndarray:
    """``f`` averaged over ``k x k`` blocks (the shape must divide)."""
    ny, nx = f.shape
    if ny % k or nx % k:
        raise ValueError("a %d x %d field does not divide into %d x %d blocks" % (ny, nx, k, k))
    return f.reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))


def power_at(series, h: int) -> float:
    """Farm power averaged over macro-steps h-4..h (1-based)."""
    s = np.asarray(series, dtype=float)
    return float(np.mean(s[max(0, h - POWER_WINDOW):h]))


def error_power(arm_power, t_power, h: int) -> float:
    p, t = power_at(arm_power, h), power_at(t_power, h)
    return abs(p - t) / abs(t)


def error_velocity(arm_uv, arm_cells_per_d: int, t_uv, t_cells_per_d: int,
                   u_inf: float = 1.0) -> float:
    """The rms velocity difference over the domain on the D/16 grid, over U."""
    ka = arm_cells_per_d // COMPARE_CELLS_PER_D
    kt = t_cells_per_d // COMPARE_CELLS_PER_D
    ua, va = (block_mean(np.asarray(f, float), ka) for f in arm_uv)
    ut, vt = (block_mean(np.asarray(f, float), kt) for f in t_uv)
    return float(np.sqrt(np.mean((ua - ut) ** 2 + (va - vt) ** 2))) / u_inf


def fluctuation_energy(u: np.ndarray, v: np.ndarray, dx: float, u_inf: float = 1.0) -> float:
    return float(0.5 * np.sum((u - u_inf) ** 2 + v ** 2) * dx * dx)


def judge(res: dict) -> dict:
    """The gate against an evaluation's results.

    ``res["arms"][a]`` for a in F, Ep, L, Cc holds ``step_seconds``, ``power``,
    ``energy`` (per macro-step) and ``errors`` = {h: {"P": e_P, "V": e_V}}; L also
    ``div``.  ``res["on_ac"]`` says whether every timed step ran on AC power.
    """
    A = res["arms"]
    t = {a: float(np.mean(A[a]["step_seconds"])) for a in ("F", "Ep", "L")}
    out: dict = {}
    g1_ep, g1_f = t["Ep"] / t["L"], t["F"] / t["L"]
    out["G1"] = {"Ep_over_L": g1_ep, "F_over_L": g1_f, "on_ac": bool(res.get("on_ac")),
                 "passed": (bool(g1_ep >= G1_EP and g1_f >= G1_F)
                            if res.get("on_ac") else None)}
    rows, ok2 = [], True
    for h in HORIZONS:
        for m in ("P", "V"):
            eL = A["L"]["errors"][str(h)][m]
            bar = G2_FACTOR * max(A["F"]["errors"][str(h)][m], A["Ep"]["errors"][str(h)][m])
            rows.append({"h": h, "measure": m, "e_L": eL, "bar": bar, "held": eL <= bar})
            ok2 = ok2 and eL <= bar
    out["G2"] = {"comparisons": rows, "passed": ok2}
    div = [float(d) for d in A["L"]["div"]]
    out["G3"] = {"max": max(div), "passed": bool(max(div) <= G3_DIV)}
    eL, eE = np.asarray(A["L"]["energy"], float), np.asarray(A["Ep"]["energy"], float)
    rel = np.abs(eL - eE) / eE
    out["G4"] = {"max_relative": float(rel.max()), "worst_step": int(np.argmax(rel)) + 1,
                 "passed": bool(np.all(rel <= G4_REL))}
    rows5, ok5 = [], True
    for h in HORIZONS:
        e_cc = A["Cc"]["errors"][str(h)]
        e_l = A["L"]["errors"][str(h)]
        beats = [m for m in ("P", "V") if e_cc[m] > e_l[m]]
        rows5.append({"h": h, "Cc": e_cc, "L": e_l, "L_better_in": beats})
        ok5 = ok5 and bool(beats)
    out["G5"] = {"comparisons": rows5, "passed": ok5}
    out["G6"] = {"passed": bool(res.get("held_out_verified"))}
    out["all"] = all(out[g]["passed"] is True for g in ("G1", "G2", "G3", "G4", "G5", "G6"))
    return out


__all__ = ["REGISTERED", "CASE", "HORIZONS", "STEPS", "THREADS", "NET_WIDTH", "NET_DEPTH",
           "Gate", "GATE", "HELD_OUT", "TRAIN_SEEDS", "VALIDATION_SEEDS", "Layout",
           "held_out_ok", "sample_layout", "split_text", "split_hash", "block_mean",
           "power_at", "error_power", "error_velocity", "fluctuation_energy", "judge"]
