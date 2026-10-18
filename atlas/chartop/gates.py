"""Stage 0's gates and comparison bars, REGISTERED BEFORE THE FIRST TRAINING RUN.

The owner chose these bars on 2026-10-18 (decision D2, "as proposed"), before any
network was trained; `scripts/chartop_register.py` writes their readable copy to
``out/chartop/registered.txt`` with this file's sha256, and
``tests/test_chartop_gates.py`` fails if this file changes after that, or if the bars
stated in the wiki page `chart-operator-stage-0` drift from the ones here (a
registered criterion drifts in code, and always in the permissive direction, unless
something checks it).  [[chart-operator-training-and-cost]] section 6 and Tables 27-28
of the public document are the source of the gate list; this module gives them
numbers.

**What is judged, once.**  After training, `scripts/chartop_evaluate.py` evaluates
the main arm B's weights -- chosen on the validation split alone -- on the held-out
test split and on the three benchmarks, and the comparison arms on the test split
and the benchmarks, and records every verdict below whatever it says.  It refuses to
evaluate the same weights twice.  Nothing here may be changed after that run.

**The measures.**
- ``e_H`` of a piece: the relative energy error of its mode set,
  ``sqrt(sum_k ||H_k - H_k^exact||^2_{A_I} / tr Lambda)``, which is
  ``sqrt((tr Lambda~ - tr Lambda) / tr Lambda)`` by T25 (ii).
- ``e_p``: ``||u_p - u_p^exact||_{A_I} / ||u_p^exact||_{A_I}``.
- On a benchmark: the coupled tier-0 field ``u~`` against the full-domain solve ``u``
  in the same discretisation (every piece's grid, one mesh): the energy-norm error
  ``||u - u~||_M / ||u||_M`` and the maximum nodal error over the range of ``u``.
- Theorem 2's bound: ``||u - u~||_M <= ||u - u_m||_M + (sum_i ||e_p,i + E_i R_i c*||^2_{A_I})^(1/2)``,
  with ``u_m`` the classical superelement with the same 16 modes per side and ``c*`` its
  interface coefficients; the bracket is the **network term**.

**The bins of E1.**  ``sup |mu|`` of the piece's chart in [0, 0.15), [0.15, 0.30),
[0.30, 0.45), [0.45, 0.65] (`data.mu_bin`), crossed with the input frequency's class
(low, mid, high: the correlation length of the material and source fields).  A bin
with fewer than ``E1_MIN_BIN`` test examples is named as not judged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

REGISTERED = ("2026-10-18, by decision D2 of the owner, before any training run; E4's "
              "maximum-error bar re-registered the same day by decision D9, before any "
              "training result was seen")

# ---------------------------------------------------------------------------
# the splits (seeds) -- the test split is drawn only at the one evaluation
# ---------------------------------------------------------------------------

SEED_BASE = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000, "train-conformal": 4_000_000}
N_VAL = 600
N_TEST = 2400
TEST_SEEDS = (SEED_BASE["test"], SEED_BASE["test"] + N_TEST - 1)
HELD_OUT_BENCHMARKS = ("bend-3", "s-channel", "insert-round")
BENCHMARK_PROBLEMS = ("steady", "step")
BENCHMARK_N = 64

# ---------------------------------------------------------------------------
# the bars
# ---------------------------------------------------------------------------

E1_MODES_MEDIAN = 0.01
E1_MODES_P90 = 0.03
E1_PARTICULAR_MEDIAN = 0.02
E1_PARTICULAR_P90 = 0.05
E1_MIN_BIN = 30
MU_BINS = ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.65))
FREQ_BINS = ("low", "mid", "high")

E3_ROUNDING = 1e-9            # relative slack on Theorem 2's bound (rounding only)
E4_ENERGY = 0.02              # ||u - u~||_M / ||u||_M on every benchmark problem
E4_MAX_OF_RANGE = 0.01        # the network's allowance on the max nodal error, of the range
#: D9 (2026-10-18, before any training result): the max nodal error is judged against
#: the classical superelement with the same 16 modes per side, plus the allowance:
#: ||u - u~||_inf <= ||u - u_m||_inf + E4_MAX_OF_RANGE (max u - min u).  The 1% bar of D2
#: was below that floor on the s-channel step (2.81% of the range with exact modes,
#: out/chartop/host_validation.json), so no network could have met it there.
SE1_IDENTITY = 1e-10          # ||Lambda~ - Lambda - E^T A_I E|| / ||Lambda||
SE1_LAMBDA_MIN = 1e-10        # lambda_min(S~) >= lambda_min(S) (1 - this)
SE2_CALLS_PER_PIECE = 1
DUAL_CHAIN_ROUNDING = 1e-10   # each Loewner difference's min eigenvalue >= -this |Lambda|

#: comparisons: ratio = the alternative arm's median e_H (or the named quantity) over the
#: default arm's; the alternative wins at <= WIN, loses at >= LOSE, otherwise a tie and
#: the default stands
WIN, LOSE = 0.8, 1.25

#: E1 is judged with the network in float32 on the laptop's CPU (decision D4); the GPU's
#: bfloat16 numbers are reported beside it, not judged
EVAL_PRECISION = "float32, laptop CPU"

#: the nonlinear-conduction demonstration (v2 addition (a)): kappa(T) = kappa_ref
#: (1 + beta (T - 300 K)) per material, every steady benchmark, frozen-coefficient
#: (Kacanov) iteration from the linear solution, no new network
KAPPA_T_BETA = {"steel": -3.0e-3, "copper": -1.0e-3}
KAPPA_T_TOL = 1e-6            # outer stop: max |T^{k+1} - T^k| / range (a float32 network
                              # resolves the coupled field to about 1e-7 of the range)
KAPPA_T_MAX_ITERATIONS = 50

#: the rollout check for S11: the s-channel marched STEPS backward-Euler steps of the
#: step benchmark's dt, modes formed once, one particular-field call per piece per step
ROLLOUT_BENCHMARK = "s-channel"
ROLLOUT_STEPS = 20

#: E5's protocol: medians of REPEATS, ratios reported, the laptop on AC power
E5_REPEATS = 5
E5_DESIGN_LOOP = 8            # geometry variants of bend-3 (inner radius), refactor per variant


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    bar: str


GATES = (
    Gate("E1", "accuracy by bin",
         f"on the held-out test split, in every (sup|mu|, frequency) bin with >= {E1_MIN_BIN} "
         f"examples: median e_H <= {E1_MODES_MEDIAN:g} and 90th percentile <= {E1_MODES_P90:g}; "
         f"median e_p <= {E1_PARTICULAR_MEDIAN:g} and 90th percentile <= {E1_PARTICULAR_P90:g}"),
    Gate("E2", "tier-2 contraction", "not applicable at Stage 0 (tier 0 only)"),
    Gate("E3", "Theorem 2 on the benchmarks",
         f"every benchmark problem: ||u - u~||_M <= (||u - u_m||_M + network term) (1 + {E3_ROUNDING:g})"),
    Gate("E4", "coupled answer against the full-domain solve",
         f"every benchmark problem: energy error <= {E4_ENERGY:g} and max error <= the "
         f"16-mode classical superelement's own max error + {E4_MAX_OF_RANGE:g} of the range (D9)"),
    Gate("E5", "cost", "reported whatever it says; 'faster' only where measured faster"),
    Gate("E6", "S1-S12", "each answered from a measurement or named as not met"),
    Gate("SE1", "Gram identity and lambda_min",
         f"every benchmark piece and problem: identity residual <= {SE1_IDENTITY:g}; "
         f"lambda_min(S~) >= lambda_min(S) (1 - {SE1_LAMBDA_MIN:g})"),
    Gate("SE2", "network calls", f"{SE2_CALLS_PER_PIECE} per piece per linear solve"),
    Gate("SE3", "port truncation",
         "every benchmark problem: the 16-mode truncation error ||u - u_m||_M below the network term"),
    Gate("DUAL", "the 2-D dual bracket",
         f"steady pieces: L~_low <= L_low <= L <= L~ with every min eigenvalue >= "
         f"-{DUAL_CHAIN_ROUNDING:g} |L|; the gap against the true error reported"),
    Gate("KT", "nonlinear conduction, frozen coefficients",
         f"every steady benchmark with kappa(T): converges within {KAPPA_T_MAX_ITERATIONS} outer "
         f"iterations, and meets E4's bars against the classical frozen-coefficient solution, the "
         f"floor being the same iteration with the 16-mode classical superelement"),
)

COMPARISONS = (
    Gate("C1", "general charts against near-conformal pieces only",
         "on the test pieces with sup|mu| < 0.15: ratio = median e_H of C1 (trained on "
         "near-conformal pieces only) over B' (general); a tie or a loss for C1 means the "
         "general class costs nothing measurable on near-conformal pieces"),
    Gate("C2", "label-free against supervised, equal compute",
         "ratio = median e_H of C2-sup over C2-free (the default objective, D8), both on train-c2"),
    Gate("C3", "principal part against direct prediction",
         "ratio = median e_H of C3 (direct) over B' (principal part plus bubble)"),
    Gate("C4", "U-shaped trunk against the sine-cosine spectral operator",
         "ratio = median e_H of C4 (spectral) over B' (U-Net); forward cost reported beside it"),
    Gate("C5", "0.3, 1 and 3 M parameters",
         "the smallest size whose median e_H is within 1.25 of the best; forward cost reported"),
    Gate("C6", "cosine port modes against sines and corner hats",
         "ratio = mean E4 energy error over the six benchmark problems, C6 (cosine) over B' "
         "(sine-hat, D3); e_H reported beside it, not compared (different port spaces)"),
    Gate("C7", "bubble-weighted correction against V0 + L0^-1 N",
         "ratio = median e_H of C7 (fast-solve form) over B' (bubble)"),
)


# ---------------------------------------------------------------------------
# the arms (their budgets are registered after pricing, in out/chartop/plan.json)
# ---------------------------------------------------------------------------

ARMS = {
    "B":       {"what": "main: U-Net 1M, bubble, sine-hat, label-free, general data", "budget": "main"},
    "Bp":      {"what": "B at the comparison budget (C1, C3-C7's reference)", "budget": "comparison"},
    "C1":      {"what": "B' on near-conformal data only (train-conformal)", "budget": "comparison",
                "data": "train-conformal"},
    "C2-sup":  {"what": "supervised objective on train-c2", "budget": "comparison",
                "objective": "supervised", "data": "train-c2"},
    "C2-free": {"what": "label-free twin on train-c2", "budget": "comparison", "data": "train-c2"},
    "C3":      {"what": "direct prediction of the modes", "budget": "comparison", "form": "direct"},
    "C4":      {"what": "sine-cosine spectral operator", "budget": "comparison", "backbone": "spectral"},
    "C5-0.3M": {"what": "U-Net 0.3M", "budget": "comparison", "size": "0.3M"},
    "C5-3M":   {"what": "U-Net 3M", "budget": "comparison", "size": "3M"},
    "C6":      {"what": "cosine port basis", "budget": "comparison", "kind": "cosine"},
    "C7":      {"what": "fast-solve output form", "budget": "comparison", "form": "fastsolve"},
}


# ---------------------------------------------------------------------------
# the judgements (pure functions of measured numbers)
# ---------------------------------------------------------------------------


def mu_bin_of(sup_mu: float) -> int:
    for k, (lo, hi) in enumerate(MU_BINS):
        if lo <= sup_mu < hi:
            return k
    return len(MU_BINS) - 1 if sup_mu <= MU_BINS[-1][1] + 1e-12 else -1


def judge_e1(e_H, e_p, sup_mu, freq) -> dict:
    e_H, e_p = np.asarray(e_H, float), np.asarray(e_p, float)
    mb = np.array([mu_bin_of(m) for m in sup_mu])
    fq = np.asarray(freq)
    cells, ok_all, judged = [], True, 0
    for k in range(len(MU_BINS)):
        for f in FREQ_BINS:
            sel = (mb == k) & (fq == f)
            cnt = int(sel.sum())
            row = {"mu_bin": list(MU_BINS[k]), "freq": f, "count": cnt}
            if cnt < E1_MIN_BIN:
                row["verdict"] = "not judged (too few test examples)"
                cells.append(row)
                continue
            judged += 1
            qH = (float(np.median(e_H[sel])), float(np.quantile(e_H[sel], 0.9)))
            qp = (float(np.median(e_p[sel])), float(np.quantile(e_p[sel], 0.9)))
            ok = (qH[0] <= E1_MODES_MEDIAN and qH[1] <= E1_MODES_P90
                  and qp[0] <= E1_PARTICULAR_MEDIAN and qp[1] <= E1_PARTICULAR_P90)
            row.update(e_H_median=qH[0], e_H_p90=qH[1], e_p_median=qp[0], e_p_p90=qp[1],
                       verdict="met" if ok else "not met")
            ok_all &= ok
            cells.append(row)
    return {"gate": "E1", "met": bool(ok_all and judged > 0), "bins_judged": judged, "cells": cells}


def judge_e3(err_learned: float, err_classical: float, network_term: float) -> dict:
    bound = (err_classical + network_term) * (1.0 + E3_ROUNDING) + 1e-300
    return {"met": bool(err_learned <= bound), "error": err_learned, "bound": bound,
            "classical_m": err_classical, "network_term": network_term}


def judge_e4(energy_rel: float, max_err: float, span: float, floor_max: float) -> dict:
    """D9: max_err against the 16-mode classical superelement's own max error floor_max."""
    bar = floor_max + E4_MAX_OF_RANGE * span
    return {"met": bool(energy_rel <= E4_ENERGY and max_err <= bar),
            "energy_rel": energy_rel, "max_over_range": max_err / span,
            "floor_max_over_range": floor_max / span, "bar_max_over_range": bar / span}


def judge_se1(identity_rel: float, lam_learned: float, lam_exact: float) -> dict:
    return {"met": bool(identity_rel <= SE1_IDENTITY
                        and lam_learned >= lam_exact * (1.0 - SE1_LAMBDA_MIN)),
            "identity_rel": identity_rel, "lambda_min_learned": lam_learned,
            "lambda_min_exact": lam_exact}


def judge_se3(truncation: float, network_term: float) -> dict:
    return {"met": bool(truncation < network_term), "truncation": truncation,
            "network_term": network_term}


def judge_dual(min_eigs_rel: list) -> dict:
    return {"met": bool(min(min_eigs_rel) >= -DUAL_CHAIN_ROUNDING),
            "worst_min_eig_rel": float(min(min_eigs_rel))}


def judge_kt(iterations: int, converged: bool, energy_rel: float, max_err: float,
             span: float, floor_max: float) -> dict:
    e4 = judge_e4(energy_rel, max_err, span, floor_max)
    return {"met": bool(converged and iterations <= KAPPA_T_MAX_ITERATIONS and e4["met"]),
            "iterations": iterations, **e4}


def compare(alternative: float, default: float) -> dict:
    """ratio = alternative / default: the alternative wins at <= WIN, loses at >= LOSE,
    otherwise a tie and the default stands."""
    r = alternative / default
    verdict = "alternative wins" if r <= WIN else ("alternative loses" if r >= LOSE else "tie")
    return {"ratio": r, "verdict": verdict}


def c5_choice(medians: dict) -> str:
    """The smallest size whose median e_H is within LOSE of the best."""
    best = min(medians.values())
    for size in ("0.3M", "1M", "3M"):
        if size in medians and medians[size] <= LOSE * best:
            return size
    return min(medians, key=medians.get)


def text() -> str:
    lines = ["THE GATES"]
    for g in GATES:
        lines.append(f"  {g.key:5s} {g.title:45s} {g.bar}")
    lines += ["", "THE COMPARISONS (ratio = alternative / default: the alternative wins at <= %.2f,"
              " loses at >= %.2f, otherwise a tie and the default stands)" % (WIN, LOSE)]
    for g in COMPARISONS:
        lines.append(f"  {g.key:5s} {g.title:58s} {g.bar}")
    lines += ["", "THE ARMS"]
    for k, v in ARMS.items():
        lines.append(f"  {k:8s} {v['what']}  [{v['budget']} budget]")
    return "\n".join(lines)
