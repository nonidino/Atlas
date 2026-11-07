"""The registration after the forms' one evaluation: S's form with the objective weighted
by a prior on a piece's traces (arm P), and the same with a global path in the U-Net
(arm PG).  ACCEPTED BY THE OWNER ON 2026-11-07.  NOT YET HASH-PINNED; NOTHING TRAINED.

The owner accepted the draft of [[chart-operator-kernel-exact-forms]] section 11.7
(commit 2a75176) in chat on 2026-11-07, and settled its two open choices the same day,
quoted: the test split "5,000,000 (Recommended)", and the benchmarks "New held-out
problems (Recommended)".  So:
- E1 is judged on a fresh test split of `gates_forms.N_TEST` pieces at seed base
  5,000,000, drawn only at the one evaluation (the split at 3,000,000 has been
  evaluated once);
- E4 and the other benchmark gates are judged on new problems: the same three layouts
  with new materials and boundary data (`benchmarks`' ``variant="held-out"``), built and
  measured with the classical solver only (`scripts/chartop_register_prior.py`), never
  read with a trained network before the one evaluation.  The six problems of
  `gates_forms` were read post hoc (section 11) and are reported beside, as seen.

`gates_forms` and `gates` are not changed.  Every bar here is read from `gates_forms`,
so the registrations cannot drift apart; ``tests/test_chartop_gates_prior.py`` checks
that and the arms against section 11.7.

**The arms** (`ARMS`): S's form (`gates_forms.ARMS["S"]`: split, corner bubble, strips
for j = 9-15, learned hats), at the comparison budget of S' (37,500 steps), the strips
skipped in training (as S' was trained), training seeds 0 and 1:
- P: the U-Net, with `train.Arm.mode_prior` ``"generator-field"`` (section 11.4's B2);
- PG: P on the ``"unet-spectral"`` backbone (B3).
S' is not trained again: its two runs (`REUSED`, sha256 as downloaded) are the first
comparison's default.

**Which runs the gates judge.**  As `gates_forms` judged its main arm, the seed-0 run of
each arm is judged on every gate (E1, E3, E4, SE1-SE3, DUAL, KT; S11 reported); K, D and
L are judged on every run, as there.  The benchmark gates name the held-out problems.
KT (the frozen-coefficient nonlinear problems) and S11 (the rollout) stay on
`gates_forms`' problems: section 11 read neither.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gates as G
from . import gates_forms as GF

REGISTERED = ("accepted by the owner on 2026-11-07, in chat, as drafted in "
              "chart-operator-kernel-exact-forms section 11.7 (commit 2a75176), with the test "
              "split at seed base 5,000,000 and new held-out benchmark problems; not yet "
              "hash-pinned; before any network of these arms was trained")

# ---------------------------------------------------------------------------
# what is kept, read from gates_forms (and through it from Stage 0's gates)
# ---------------------------------------------------------------------------

KIND, M_PORTS, N_COLUMNS, N_LEARNED, STRIP_SINES = (GF.KIND, GF.M_PORTS, GF.N_COLUMNS,
                                                    GF.N_LEARNED, GF.STRIP_SINES)
N_VAL, N_TEST = GF.N_VAL, GF.N_TEST
BENCHMARK_NAMES, BENCHMARK_PROBLEMS, BENCHMARK_N = (GF.HELD_OUT_BENCHMARKS, GF.BENCHMARK_PROBLEMS,
                                                    GF.BENCHMARK_N)
E1_MODES_MEDIAN, E1_MODES_P90 = GF.E1_MODES_MEDIAN, GF.E1_MODES_P90
E1_PARTICULAR_MEDIAN, E1_PARTICULAR_P90 = GF.E1_PARTICULAR_MEDIAN, GF.E1_PARTICULAR_P90
E1_MIN_BIN, MU_BINS, FREQ_BINS = GF.E1_MIN_BIN, GF.MU_BINS, GF.FREQ_BINS
E3_ROUNDING = GF.E3_ROUNDING
E4_ENERGY, E4_MAX_OF_RANGE = GF.E4_ENERGY, GF.E4_MAX_OF_RANGE
SE1_IDENTITY, SE1_LAMBDA_MIN = GF.SE1_IDENTITY, GF.SE1_LAMBDA_MIN
SE2_CALLS_PER_PIECE = GF.SE2_CALLS_PER_PIECE
DUAL_CHAIN_ROUNDING = GF.DUAL_CHAIN_ROUNDING
KAPPA_T_BETA, KAPPA_T_TOL, KAPPA_T_MAX_ITERATIONS = (GF.KAPPA_T_BETA, GF.KAPPA_T_TOL,
                                                     GF.KAPPA_T_MAX_ITERATIONS)
ROLLOUT_BENCHMARK, ROLLOUT_STEPS = GF.ROLLOUT_BENCHMARK, GF.ROLLOUT_STEPS
WIN, LOSE = GF.WIN, GF.LOSE
EVAL_PRECISION = GF.EVAL_PRECISION
K_E_K, K_E_K_INF = GF.K_E_K, GF.K_E_K_INF
DATUM_SHIFT_K, D_RATIO, D_FLOOR, D_MAX_OF_RANGE = (GF.DATUM_SHIFT_K, GF.D_RATIO, GF.D_FLOOR,
                                                   GF.D_MAX_OF_RANGE)
L_TEST_PIECES, L_SOURCES = GF.L_TEST_PIECES, GF.L_SOURCES
E5_PRICED = GF.E5_PRICED
CEILING_USD, BOX_MAX_USD_PER_HOUR = GF.CEILING_USD, GF.BOX_MAX_USD_PER_HOUR
judge_k, judge_d, judge_l, judge_e1 = GF.judge_k, GF.judge_d, GF.judge_l, GF.judge_e1
judge_e4_both_frames, compare_pairings, e_h_learned = (GF.judge_e4_both_frames, GF.compare_pairings,
                                                       GF.e_h_learned)

# ---------------------------------------------------------------------------
# what is new: the test split, the held-out problems, the arms
# ---------------------------------------------------------------------------

#: the fresh test split: example i has seed TEST_SEED_BASE + i, drawn only at the one
#: evaluation; no other split uses this range (`gates.SEED_BASE`)
TEST_SEED_BASE = 5_000_000
TEST_SEEDS = (TEST_SEED_BASE, TEST_SEED_BASE + N_TEST)

#: the benchmark problems E3, E4, K, D, L, SE1-SE3 and DUAL are judged on: `benchmarks`'
#: held-out variant of each layout, steady and step
HELD_OUT_VARIANT = "held-out"
HELD_OUT = tuple((name, problem) for name in BENCHMARK_NAMES for problem in BENCHMARK_PROBLEMS)
#: reported beside, not judged: the six problems `gates_forms` judged, read post hoc since
SEEN = HELD_OUT

#: S' (`gates_forms.ARMS["Sp"]`), trained 2026-10-19 and evaluated once; the first
#: comparison's default, not trained again (sha256 of best.pt as downloaded)
REUSED = {0: ("Sp-s0", "2e7bcf22d5db38a420003852b38a12aee38bc220f309dade2b56c01f1a106cc3"),
          1: ("Sp-s1", "70b7c3d8f041df9483a4e7f8340eb26d20665e7365b7a86a00a4764fb6627538")}

COMMON = dict(GF.COMMON, bubble=GF.ARMS["S"]["bubble"], strips=GF.ARMS["S"]["strips"],
              hats="learned", strips_in_training=False, mode_prior="generator-field")

ARMS = {
    "P":  {"what": "S's form with the objective weighted by the generator's trace prior (B2)",
           "backbone": "unet", "budget": "comparison", "steps": GF.ARMS["Sp"]["steps"],
           "seeds": (0, 1)},
    "PG": {"what": "P with a cosine-basis path at the U-Net's coarsest level (B3)",
           "backbone": "unet-spectral", "budget": "comparison", "steps": GF.ARMS["Sp"]["steps"],
           "seeds": (0, 1)},
}


def arm_config(arm: str) -> dict:
    """One arm's configuration: `COMMON` with the arm's backbone and steps.  Every key but
    ``m`` (the port basis's modes per side, which `train.Arm` leaves to the expert's
    default of 16) is a `train.Arm` field."""
    return dict(COMMON, backbone=ARMS[arm]["backbone"], steps=ARMS[arm]["steps"])


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    bar: str


_FORMS = {g.key: g for g in GF.GATES}
_ON_THE_PROBLEMS = "every held-out benchmark problem (benchmarks' variant 'held-out')"


def _swap(text: str, *pairs) -> str:
    """``text`` with each ``(old, new)`` replaced, refusing an ``old`` that is not there (a
    replacement that silently matched nothing would leave gates_forms' wording in place)."""
    for old, new in pairs:
        if old not in text:
            raise ValueError(f"{old!r} is not in the bar {text!r}")
        text = text.replace(old, new)
    return text


GATES = (
    Gate("K", _FORMS["K"].title, _swap(_FORMS["K"].bar, ("every held-out test piece",
                                                         "every piece of the fresh test split"),
                                       ("every benchmark piece", "every held-out benchmark piece"))),
    Gate("D", _FORMS["D"].title, _swap(_FORMS["D"].bar, ("every benchmark problem", _ON_THE_PROBLEMS))),
    Gate("L", _FORMS["L"].title, _swap(_FORMS["L"].bar, ("both arms", "every run"),
                                       ("every benchmark piece", "every held-out benchmark piece"),
                                       (f"the first {L_TEST_PIECES} test pieces",
                                        f"the first {L_TEST_PIECES} pieces of the fresh test split"))),
    Gate("E1", _FORMS["E1"].title, _swap(_FORMS["E1"].bar, (
        "arm S, on the held-out test split",
        f"the seed-0 run of each arm, on the fresh test split (seeds {TEST_SEEDS[0]:,} to "
        f"{TEST_SEEDS[1] - 1:,})"))),
    Gate("E3", _FORMS["E3"].title, _swap(_FORMS["E3"].bar, ("every benchmark problem", _ON_THE_PROBLEMS))),
    Gate("E4", _FORMS["E4"].title, _swap(_FORMS["E4"].bar, ("every benchmark problem", _ON_THE_PROBLEMS))),
    Gate("E5", _FORMS["E5"].title, _FORMS["E5"].bar),
    Gate("SE1", _FORMS["SE1"].title, _swap(_FORMS["SE1"].bar, ("every benchmark piece and problem",
                                                               "every held-out benchmark piece and problem"))),
    Gate("SE2", _FORMS["SE2"].title, _FORMS["SE2"].bar),
    Gate("SE3", _FORMS["SE3"].title, _swap(_FORMS["SE3"].bar, ("every benchmark problem", _ON_THE_PROBLEMS))),
    Gate("DUAL", _FORMS["DUAL"].title, _FORMS["DUAL"].bar),
    Gate("KT", _FORMS["KT"].title, _FORMS["KT"].bar),
    Gate("S11", _FORMS["S11"].title, _FORMS["S11"].bar),
)

COMPARISONS = (
    Gate("CP", "the prior: P against S'",
         f"ratio = mean E4 energy error over the six held-out problems (the steps lowered by "
         f"{-DATUM_SHIFT_K:g} K), P over S', seed with seed; P wins at <= {WIN:g} and loses at >= "
         f"{LOSE:g}, otherwise a tie and S' stands; a verdict stands only if both pairings give it, "
         f"otherwise unresolved; the same ratio on the six seen problems reported beside it"),
    Gate("CG", "the global path: PG against P",
         "as CP, PG over P"),
)

PREDICTIONS = (
    "1. P's steady energy errors are below the S' run of the same seed on all three held-out "
    "steady problems.",
    "2. P meets E4 on the held-out bend-3 steady problem.",
    "3. Neither P nor PG meets E4 on the held-out s-channel steady problem.",
    "4. Neither meets E1: the median e_H,L stays above 0.1 (section 11.6: four pieces fitted "
    "alone stayed above 0.15 for 3,000 steps).",
    "5. PG's median e_H,L at n = 128 is no higher than at n = 32, while P's rises with n.",
    "6. K, D and L are met, by structure.",
)


def cw_measure(energy_rel: dict) -> float:
    """``energy_rel[(benchmark, problem, frame)]`` for one variant's six problems (the
    held-out ones for the verdict, the seen ones beside it): the mean, the steady ones as
    registered and the steps lowered by 300 K (`gates_forms.cw_measure`'s rule)."""
    vals = [energy_rel[(b, p, "registered" if p == "steady" else "lowered")] for b, p in HELD_OUT]
    return float(np.mean(vals))


def text() -> str:
    lines = [f"REGISTERED: {REGISTERED}", "", "THE GATES"]
    for g in GATES:
        lines.append(f"  {g.key:5s} {g.title:58s} {g.bar}")
    lines += ["", "THE COMPARISONS (ratio = alternative / default)"]
    for g in COMPARISONS:
        lines.append(f"  {g.key:5s} {g.title:58s} {g.bar}")
    lines += ["", "THE ARMS (all: " + ", ".join(f"{k} {v}" for k, v in COMMON.items()) + ")"]
    for k, v in ARMS.items():
        lines.append(f"  {k:3s} {v['what']}; backbone {v['backbone']}; {v['steps']:,} steps, "
                     f"training seeds {', '.join(str(s) for s in v['seeds'])}")
    lines.append("  S'  reused, not trained: " + ", ".join(f"{r} {h[:16]}..." for r, h in REUSED.values()))
    lines += ["", "THE PREDICTIONS"] + [f"  {p}" for p in PREDICTIONS]
    lines += ["", f"COST: at most ${CEILING_USD:.2f} in all, each box at most "
                  f"${BOX_MAX_USD_PER_HOUR:.2f} an hour; nothing long runs before E5 is priced"]
    return "\n".join(lines)
