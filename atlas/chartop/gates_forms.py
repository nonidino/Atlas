"""The registration after Stage 0: the split form with W358's corner bubble and strips
(arm S), and W358's control (arm W).  ACCEPTED BY THE OWNER BEFORE ANY NETWORK IN THESE
FORMS WAS TRAINED; NOT YET HASH-PINNED.

The owner accepted the draft of [[chart-operator-kernel-exact-forms]] section 8 (commit
0f1b4ae) as written, in chat on 2026-10-19.  Its choices, quoted there: the output form
"Bubble + strips", a particular-field network of "1 M (as drafted)", and the
registration shape "Main + W358 control".  Stage 0's module `gates` is not changed.
Every bar this registration shares with Stage 0 is read from it, so the two cannot
drift apart.  ``tests/test_chartop_gates_forms.py`` checks the bars and arms stated in
section 8 against the ones here.  Pinning this file's hash, as
`scripts/chartop_register.py` pinned Stage 0's, is the owner's separate decision.

**The arms** (`ARMS`).  All three use the split form (`forms.KernelExpert("split")`):
two U-Nets of 1 M, one for the modes and one for the particular field, sine-hat ports
with 16 modes per side, the label-free objective with D4 augmentation, and Stage 0's
generator, data seeds, splits, benchmarks and envelope.
- S, the main arm: the corner bubble and the strips, 150,000 steps (Stage 0's main
  budget, arm B's in ``out/chartop/plan.json``), training seed 0.
- S': S at 37,500 steps (Stage 0's comparison budget, that of arm B'), seeds 0 and 1.
- W, W358's control: as S', with Stage 0's square bubble and no strips.
A training seed is `train.Arm.seed`, the weights' initialisation and the batch order.
The data are the registered seeds' in every arm.

**The measures.**
- ``e_H,L``, E1's measure on the columns the network learns.  ``L`` holds the four hats
  and the sines j = 1-8 of each side (`forms.learned_columns`, 36 of 64), and
  ``e_H,L = sqrt(sum_{k in L} ||H_k - H_k^exact||^2_{A_I} / sum_{k in L} Lambda_kk)``.
  The denominator is those columns' own energy: on the plain square the strip columns
  carry 62-70% of tr Lambda, so dividing by tr Lambda would make the bar 1.6-1.8 times
  easier.  Arm W has no strips, so its learned columns are all 64.
- Gate K: ``e_K`` and ``e_K,inf``, the uniform state's residual (`forms.kernel_measures`).
- Gate D: every temperature is lowered by 300 K (`benchmarks.datum_shift`: the
  Dirichlet data and T_old together).  The ratio ``||u - u~||_M,lowered / ||u - u~||_M``
  is within `D_RATIO` of one wherever both errors exceed `D_FLOOR` times ``||u||_M``.
  That norm is the registered frame's, as section 8 writes it.  The maximum errors
  over the range agree to within `D_MAX_OF_RANGE` of the range.
- Gate L: the modes and the level field are the same, bit for bit, with the piece's
  own source, with no source, and with a unit-RMS random source.
- E4 in two frames: as registered, and with every temperature lowered by 300 K, where
  a step's reference energy no longer contains the level.
- CW's measure: the mean E4 energy error over the six benchmark problems.  The steady
  problems count as registered (gate D makes the two frames agree there); the steps
  count lowered by 300 K.

**What is judged, once.**  As in Stage 0 (`gates`), the main arm's weights are chosen on
the validation split alone and evaluated once on the held-out test split and the three
benchmarks.  The comparison arms are evaluated on the benchmarks, and every verdict
is recorded whatever it says.  E1 is judged in float32 on the laptop's CPU (D4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import gates as G

REGISTERED = ("accepted by the owner on 2026-10-19, in chat, as drafted in "
              "chart-operator-kernel-exact-forms section 8 (commit 0f1b4ae), before any "
              "network in these forms was trained")

# ---------------------------------------------------------------------------
# what is kept from Stage 0, read from Stage 0's module
# ---------------------------------------------------------------------------

SEED_BASE, N_VAL, N_TEST, TEST_SEEDS = G.SEED_BASE, G.N_VAL, G.N_TEST, G.TEST_SEEDS
HELD_OUT_BENCHMARKS, BENCHMARK_PROBLEMS, BENCHMARK_N = (G.HELD_OUT_BENCHMARKS,
                                                        G.BENCHMARK_PROBLEMS, G.BENCHMARK_N)
E1_MODES_MEDIAN, E1_MODES_P90 = G.E1_MODES_MEDIAN, G.E1_MODES_P90
E1_PARTICULAR_MEDIAN, E1_PARTICULAR_P90 = G.E1_PARTICULAR_MEDIAN, G.E1_PARTICULAR_P90
E1_MIN_BIN, MU_BINS, FREQ_BINS = G.E1_MIN_BIN, G.MU_BINS, G.FREQ_BINS
E3_ROUNDING = G.E3_ROUNDING
E4_ENERGY, E4_MAX_OF_RANGE = G.E4_ENERGY, G.E4_MAX_OF_RANGE
SE1_IDENTITY, SE1_LAMBDA_MIN = G.SE1_IDENTITY, G.SE1_LAMBDA_MIN
SE2_CALLS_PER_PIECE = G.SE2_CALLS_PER_PIECE
DUAL_CHAIN_ROUNDING = G.DUAL_CHAIN_ROUNDING
KAPPA_T_BETA, KAPPA_T_TOL, KAPPA_T_MAX_ITERATIONS = (G.KAPPA_T_BETA, G.KAPPA_T_TOL,
                                                     G.KAPPA_T_MAX_ITERATIONS)
ROLLOUT_BENCHMARK, ROLLOUT_STEPS = G.ROLLOUT_BENCHMARK, G.ROLLOUT_STEPS
WIN, LOSE = G.WIN, G.LOSE
EVAL_PRECISION = G.EVAL_PRECISION

# ---------------------------------------------------------------------------
# the new bars
# ---------------------------------------------------------------------------

KIND, M_PORTS = "sine-hat", 16
N_COLUMNS = 4 * M_PORTS
STRIP_SINES = (9, 15)          # the side sines the strips compute, on every side
N_LEARNED = N_COLUMNS - 4 * (STRIP_SINES[1] - STRIP_SINES[0] + 1)   # 36

K_E_K = 1e-12                  # e_K on every test and benchmark piece, steady and step
K_E_K_INF = 1e-12              # e_K,inf likewise
DATUM_SHIFT_K = -300.0         # gate D and E4's second frame: every temperature lowered by 300 K
D_RATIO = 1e-8                 # | ||u - u~||_M,lowered / ||u - u~||_M - 1 |
D_FLOOR = 1e-9                 # ... judged wherever both errors exceed this times ||u||_M
D_MAX_OF_RANGE = 1e-9          # the maximum errors over the range agree to this, of the range
L_TEST_PIECES = 100            # gate L: every benchmark piece and the first 100 test pieces
L_SOURCES = ("own", "none", "random unit-RMS")

#: E5: priced on the box before any long run, the strips and their operator's assembly
#: included: the forward pass and one training step at these (n, batch)
E5_PRICED = ((32, 32), (64, 32), (128, 8))

CEILING_USD = 10.0             # the rented-GPU ceiling for the whole registration
BOX_MAX_USD_PER_HOUR = 0.70    # each box


_STAGE0 = {g.key: g for g in G.GATES}


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    bar: str


GATES = (
    Gate("K", "the uniform state",
         f"every held-out test piece and every benchmark piece, steady and step: "
         f"e_K <= {K_E_K:g} and e_K,inf <= {K_E_K_INF:g}"),
    Gate("D", "the datum",
         f"every benchmark problem, steady and step, every temperature lowered by "
         f"{-DATUM_SHIFT_K:g} K (the data and T_old together): | ||u - u~||_M,lowered / "
         f"||u - u~||_M - 1 | <= {D_RATIO:g} wherever both errors exceed {D_FLOOR:g} ||u||_M, "
         f"and the maximum error over the range within {D_MAX_OF_RANGE:g} of the range"),
    Gate("L", "the load",
         f"both arms: the modes and the level field bit for bit the same with the piece's own "
         f"source, no source and a unit-RMS random source, on every benchmark piece and the "
         f"first {L_TEST_PIECES} test pieces"),
    Gate("E1", "accuracy by bin, on the learned columns",
         f"arm S, on the held-out test split, in every (sup|mu|, frequency) bin with >= "
         f"{E1_MIN_BIN} examples: median e_H,L <= {E1_MODES_MEDIAN:g} and 90th percentile <= "
         f"{E1_MODES_P90:g} on the {N_LEARNED} learned columns, against their own energy; "
         f"median e_p <= {E1_PARTICULAR_MEDIAN:g} and 90th percentile <= {E1_PARTICULAR_P90:g}; "
         f"e_H over all {N_COLUMNS} columns and e_K reported beside it, not judged"),
    Gate("E3", "Theorem 2 on the benchmarks", _STAGE0["E3"].bar),
    Gate("E4", "coupled answer against the full-domain solve, in both frames",
         f"every benchmark problem, as registered and with every temperature lowered by "
         f"{-DATUM_SHIFT_K:g} K: energy error <= {E4_ENERGY:g} and max error <= the 16-mode "
         f"classical superelement's own max error + {E4_MAX_OF_RANGE:g} of the range"),
    Gate("E5", "cost",
         "priced on the box before any long run: the forward pass and one training step at "
         + ", ".join(f"n = {n} (batch {b})" for n, b in E5_PRICED)
         + ", the strips and their operator's assembly included; then reported whatever it "
           "says, 'faster' only where measured faster"),
    Gate("SE1", "Gram identity and lambda_min", _STAGE0["SE1"].bar),
    Gate("SE2", "network calls", _STAGE0["SE2"].bar),
    Gate("SE3", "port truncation", _STAGE0["SE3"].bar),
    Gate("DUAL", "the 2-D dual bracket", _STAGE0["DUAL"].bar),
    Gate("KT", "nonlinear conduction, frozen coefficients", _STAGE0["KT"].bar),
    Gate("S11", "the rollout",
         f"reported: the {ROLLOUT_BENCHMARK} marched {ROLLOUT_STEPS} steps, the modes, the level "
         f"field and the strip columns formed once, one particular-field call per piece per step"),
)

COMPARISONS = (
    Gate("CW", "W358: the corner bubble and the strips",
         f"ratio = mean E4 energy error over the six benchmark problems (the steps lowered by "
         f"{-DATUM_SHIFT_K:g} K), W over S'; median e_H over all {N_COLUMNS} columns beside it; "
         f"W wins at <= {WIN:g} and loses at >= {LOSE:g}, otherwise a tie and S' stands; with "
         f"two seeds, a verdict stands only if both pairings (seed with seed) give it, "
         f"otherwise unresolved; the ratio at step 0 (zero heads, CW0) reported beside it"),
)

# ---------------------------------------------------------------------------
# the arms (batches, learning rate and validation cadence as Stage 0's plan;
# the plan for these arms is written after E5's pricing, before training)
# ---------------------------------------------------------------------------

COMMON = {"form": "split", "backbone": "unet", "size": "1M", "p_size": "1M",
          "kind": KIND, "m": M_PORTS, "objective": "label-free", "data": "train"}

ARMS = {
    "S":  {"what": "main: split, corner bubble, strips for j = 9-15", "bubble": "corner",
           "strips": True, "budget": "main", "steps": 150_000, "seeds": (0,)},
    "Sp": {"what": "S at the comparison budget (CW's default)", "bubble": "corner",
           "strips": True, "budget": "comparison", "steps": 37_500, "seeds": (0, 1)},
    "W":  {"what": "W358's control: split, Stage 0's square bubble, no strips (CW's alternative)",
           "bubble": "square", "strips": False, "budget": "comparison", "steps": 37_500,
           "seeds": (0, 1)},
}


# ---------------------------------------------------------------------------
# the judgements (pure functions of measured numbers)
# ---------------------------------------------------------------------------


def e_h_learned(err_energy, lambda_diag, columns) -> float:
    """``e_H,L`` of one piece: the columns' error energies ``||H_k - H_k^exact||^2_{A_I}``
    over their own exact energies ``Lambda_kk``, summed over ``columns`` only."""
    cols = list(columns)
    err = np.asarray(err_energy, float)
    lam = np.asarray(lambda_diag, float)
    if err.shape != (N_COLUMNS,) or lam.shape != (N_COLUMNS,):
        raise ValueError(f"need {N_COLUMNS} columns' energies, not {err.shape} and {lam.shape}")
    if len(set(cols)) != len(cols):
        raise ValueError("a column is named twice")
    return math.sqrt(float(err[cols].sum()) / float(lam[cols].sum()))


def judge_k(e_K, e_K_inf) -> dict:
    eK, eKi = np.asarray(e_K, float), np.asarray(e_K_inf, float)
    return {"met": bool(eK.max() <= K_E_K and eKi.max() <= K_E_K_INF),
            "e_K_max": float(eK.max()), "e_K_inf_max": float(eKi.max()), "pieces": int(eK.size)}


def judge_d(err_registered: float, err_lowered: float, norm_u: float,
            max_registered: float, max_lowered: float, span: float) -> dict:
    """Errors in the energy norm (absolute, each in its own frame), ``norm_u`` the
    registered frame's ``||u||_M``; maximum nodal errors absolute, ``span`` the range."""
    judged = bool(err_registered > D_FLOOR * norm_u and err_lowered > D_FLOOR * norm_u)
    dev = abs(err_lowered / err_registered - 1.0) if judged else None
    dmax = abs(max_lowered - max_registered) / span
    return {"met": bool((dev <= D_RATIO if judged else True) and dmax <= D_MAX_OF_RANGE),
            "ratio_judged": judged, "ratio_deviation": dev, "max_deviation_of_range": dmax}


def judge_l(modes: list, levels: list) -> dict:
    """``modes`` and ``levels``: one array per source in `L_SOURCES`, for one piece."""
    if len(modes) != len(L_SOURCES) or len(levels) != len(L_SOURCES):
        raise ValueError(f"need one call per source: {L_SOURCES}")
    same = (all(np.array_equal(modes[0], h) for h in modes[1:])
            and all(np.array_equal(levels[0], v) for v in levels[1:]))
    return {"met": bool(same)}


def judge_e1(e_HL, e_p, sup_mu, freq) -> dict:
    """Stage 0's bins and bars (`gates.judge_e1`), on ``e_H,L``."""
    out = G.judge_e1(e_HL, e_p, sup_mu, freq)
    out["measure"] = f"e_H,L on the {N_LEARNED} learned columns, against their own energy"
    return out


def judge_e4_both_frames(registered: dict, lowered: dict) -> dict:
    """Each frame: ``energy_rel``, ``max_err``, ``span``, ``floor_max`` as `gates.judge_e4`."""
    r, lo = G.judge_e4(**registered), G.judge_e4(**lowered)
    return {"met": bool(r["met"] and lo["met"]), "as registered": r, "lowered by 300 K": lo}


def cw_measure(energy_rel: dict) -> float:
    """``energy_rel[(benchmark, problem, frame)]``, frame ``"registered"`` or
    ``"lowered"``: the mean over the six problems, the steady ones as registered and the
    steps lowered by 300 K."""
    vals = [energy_rel[(b, p, "registered" if p == "steady" else "lowered")]
            for b in HELD_OUT_BENCHMARKS for p in BENCHMARK_PROBLEMS]
    return float(np.mean(vals))


def compare_pairings(alternative: dict, default: dict) -> dict:
    """``{seed: measure}`` for each arm, paired seed with seed: each pairing judged by
    `gates.compare`, and the verdict stands only if every pairing gives it."""
    if sorted(alternative) != sorted(default):
        raise ValueError(f"the seeds do not pair: {sorted(alternative)} and {sorted(default)}")
    rows = {s: G.compare(alternative[s], default[s]) for s in sorted(default)}
    verdicts = {r["verdict"] for r in rows.values()}
    return {"pairings": rows, "verdict": verdicts.pop() if len(verdicts) == 1 else "unresolved"}


def text() -> str:
    lines = ["THE GATES"]
    for g in GATES:
        lines.append(f"  {g.key:5s} {g.title:58s} {g.bar}")
    lines += ["", "THE COMPARISON (ratio = alternative / default)"]
    for g in COMPARISONS:
        lines.append(f"  {g.key:5s} {g.title:58s} {g.bar}")
    lines += ["", "THE ARMS (all: " + ", ".join(f"{k} {v}" for k, v in COMMON.items()) + ")"]
    for k, v in ARMS.items():
        lines.append(f"  {k:3s} {v['what']}; {v['steps']:,} steps, training seeds "
                     f"{', '.join(str(s) for s in v['seeds'])}")
    lines += ["", f"COST: at most ${CEILING_USD:.2f} in all, each box at most "
                  f"${BOX_MAX_USD_PER_HOUR:.2f} an hour; nothing long runs before E5 is priced"]
    return "\n".join(lines)
