"""The registration after section 11.17: rule R2's hybrid H2, U's form only on squares of one
material, and beside it H2r, the classical response reused on the squares; judged on E4 and
on a price bar on square-first layouts from a registered generator.  ACCEPTED BY THE OWNER
ON 2026-11-08, AS DRAFTED, ITS PRICE PROTOCOL REVISED THE SAME DAY (`REVISED`); NOT YET
HASH-PINNED.

The owner accepted the draft of [[chart-operator-kernel-exact-forms]] section 11.18 (commit
3f945f2) in chat on 2026-11-08 ("Accept as drafted"), with its four recommended choices
(`SETTLED`): H2 with H2r beside; the generated layouts (`benchmarks_squares`); PR2's bar at
0.90; the load recorded, not judged.  After the first dry run, in which 4 of 6 runs counted,
the owner adopted the per-problem count ("Adopt it (Recommended)") and moved the price to a
rented whole machine ("Box for both (Recommended)"), in chat the same day (`REVISED`).
Pinning is a separate step, run only with the owner's approval, and only after a dry run
passes (`dry_run_passes`).

**The arm, H2.**  No weights; nothing is trained.  U's form without its networks (U as
`gates_hybrid.U_FORM`, the stand-in as section 11.13's) on every piece `is_square` accepts:
an affine chart onto a square, four equal sides and four right angles to `SQUARE_TOL` of the
side, of one material.  The classical solve on every other piece (`rule_pieces2`), then the
coupled solve.  The rule has no parameter and no Dirichlet criterion: on heated-strip, U alone
was at the floor to 1e-9, its Dirichlet owners included (section 11.17).

**Beside it, H2r** (`reuse.ReuseExpert`): H2's picks, the squares answered by one classical
response per class of resolution, Fourier number and source.

**Reported beside, not judged**: the classical expert; U on every piece; P-s0, for E4 only.

**The gates** (`GATES`).  E4-H2: H2 meets E4 on every new problem, in both frames
(`judge_e4_both_frames`, read from `gates_forms`).  PR2: the geometric mean over the new
problems of H2's price is at most `PRICE_BAR`, each problem's price the median of its first
`PRICE_RUNS` counted measurements (`first_counted`).  Beside them: H2r's E4 and price, and the
ratio of H2r's price to H2's.

**The price protocol** (`run_counts2`), counted per problem, and judging whether the timing
itself was disturbed.  A run's measurement of a problem counts if the run's own checks hold
(AC at every sample, taken every `SAMPLE_S` seconds through the run, and no gap between
samples longer than `MAX_GAP_S`, a sleep) and that problem's classical best time per piece
is at most `MAX_MS_PER_PIECE`, with no lower edge.  The other processes' load and the
Kernel-Power events are recorded, not judged.  `PRICE_THREADS` threads, best of `PRICE_REPS`
with the arms in turn (`PRICE_ORDER`), each run in its own process, every run kept.  Runs
continue until every problem has `PRICE_RUNS` counted measurements (`runs_suffice`), at most
`PRICE_RUNS_MAX` runs in all.

**Where** (`REVISED`): a rented whole machine, every GPU of its host, so that no other
tenant shares its CPU.  The one evaluation's price runs are on the dry run's CPU model, and
its accuracy on the laptop.  A machine with no battery holds the AC check vacuously; each
run records whether a battery was reported, and the machine it ran on.

**The dry run**, before pinning: `DRY_RUNS` runs of the protocol on section 11.17's twelve
problems, which have been read.  It passes (`dry_run_passes`) if every problem has
`PRICE_RUNS` counted measurements among them, the evaluation's own need, within six runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import benchmarks_squares as BS
from . import gates as G
from . import gates_forms as GF
from . import gates_hybrid as GH

REGISTERED = ("accepted by the owner on 2026-11-08, in chat ('Accept as drafted'), as drafted "
              "in chart-operator-kernel-exact-forms section 11.18 (commit 3f945f2), with its "
              "four recommended choices; its price protocol revised the same day, in chat, to the "
              "per-problem count ('Adopt it (Recommended)') on a rented whole machine ('Box for "
              "both (Recommended)'); not yet hash-pinned")

#: the owner's revision of the price protocol after the first dry run (2026-11-08, in chat)
REVISED = {
    "the count": "per problem: a run's measurement of a problem counts if the run holds AC and "
                 "sleep and that problem's classical time is at most 60 ms a piece; each "
                 "problem's price the median of its first three counted; at most nine runs "
                 "('Adopt it (Recommended)')",
    "the machine": "a rented whole machine for the dry run and the one evaluation's price runs, "
                   "the evaluation's accuracy on the laptop ('Rent a temporary box', then 'Box "
                   "for both (Recommended)')",
}

SETTLED = {
    "the square path": "H2, with H2r beside",
    "the problems": "the generated square-first layouts of benchmarks_squares",
    "PR2's bar": "0.90, as before",
    "the load": "recorded, not judged; time, power and sleep judged",
}

KIND, M_PORTS, BENCHMARK_N, DATUM_SHIFT_K = GF.KIND, GF.M_PORTS, GF.BENCHMARK_N, GF.DATUM_SHIFT_K
E4_ENERGY, E4_MAX_OF_RANGE = G.E4_ENERGY, G.E4_MAX_OF_RANGE
judge_e4_both_frames = GF.judge_e4_both_frames
U_FORM = GH.U_FORM
REPORTED = ("the classical expert", "U, with its networks, on every piece", "P-s0, for E4 only")

# ---------------------------------------------------------------------------
# the problems
# ---------------------------------------------------------------------------

SEED_BASE, N_LAYOUTS = BS.SEED_BASE, BS.N_LAYOUTS
CLASSICAL_RECORD = "out/chartop/registration-squares.json"
#: the registered layouts: the first N_LAYOUTS seeds from SEED_BASE that are built, are not
#: refused by the envelope and whose classical floors meet E4 in both frames, as the classical
#: record found them (checked by a test).  Seed 7,000,004, the one round inclusion drawn, was
#: skipped: in its step the copper disc's core squares have Fourier number 116, above the
#: envelope's 100.
SEEDS: tuple = (7_000_000, 7_000_001, 7_000_002, 7_000_003, 7_000_005, 7_000_006, 7_000_007, 7_000_008)


def new_problems(seeds=None) -> tuple:
    return tuple((int(s), p) for s in (SEEDS if seeds is None else seeds) for p in BS.PROBLEMS)


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

SQUARE_TOL = 1e-9


def is_square(p) -> bool:
    """An affine chart onto a square of one material: four equal sides and four right
    angles, to `SQUARE_TOL` of the side, and log kappa_hat zero everywhere."""
    if p.chart.method != "affine":
        return False
    C = np.asarray(p.chart.corners(), float)
    e = [C[(k + 1) % 4] - C[k] for k in range(4)]
    sides = [float(np.hypot(*v)) for v in e]
    side = max(sides)
    if side <= 0.0 or side - min(sides) > SQUARE_TOL * side:
        return False
    if any(abs(float(np.dot(e[k], e[(k + 1) % 4]))) > SQUARE_TOL * side * side for k in range(4)):
        return False
    return bool(np.all(np.asarray(p.log_kappa_hat) == 0.0))


def rule_pieces2(L) -> list:
    """Rule R2: the classical solve on every piece that is not a square of one material."""
    return [i for i, p in enumerate(L.problems) if not is_square(p)]


# ---------------------------------------------------------------------------
# the price
# ---------------------------------------------------------------------------

PRICE_BAR = 0.90
PRICE_THREADS = 4
PRICE_REPS = 5
#: counted measurements per problem; its price is the median of its first `PRICE_RUNS`
PRICE_RUNS = 3
#: runs in all, at most: they continue until every problem has `PRICE_RUNS` counted
PRICE_RUNS_MAX = 9
PRICE_ORDER = ("classical", "H2", "H2r", "coupled solve")
MAX_MS_PER_PIECE = 60.0
SAMPLE_S = 5.0
MAX_GAP_S = 30.0
#: the dry run's runs; it passes if every problem has `PRICE_RUNS` counted among them
DRY_RUNS = 6
price = GH.price


def run_counts2(state: dict) -> dict:
    """Which of a price run's measurements count, problem by problem (`REVISED`).  ``state``:
    ``classical_ms_per_piece`` (the classical expert's best time per piece, per problem),
    ``plugged`` (AC at each sample) and ``sample_gaps_s`` (the gaps between consecutive
    samples).  The run's own checks, AC and sleep, void every measurement in it; the time
    check voids only its own problem's.  Load is not judged."""
    failed = []
    if not state["plugged"] or not all(state["plugged"]):
        failed.append("off AC at a sample" if state["plugged"] else "no power samples")
    gap = max(state["sample_gaps_s"], default=float("inf"))
    if gap > MAX_GAP_S:
        failed.append(f"a gap of {gap:.0f} s between samples")
    problems = {}
    for key, ms in state["classical_ms_per_piece"].items():
        why = list(failed)
        if ms > MAX_MS_PER_PIECE:
            why.append(f"the classical expert at {ms:.1f} ms a piece, above {MAX_MS_PER_PIECE:g}")
        problems[key] = {"counts": not why, "failed": why}
    return {"run_holds": not failed, "failed": failed, "problems": problems}


def tally(counts: list) -> dict:
    """Each problem's number of counted measurements over ``counts``, a list of `run_counts2`
    results in run order."""
    out = {}
    for c in counts:
        for key, verdict in c["problems"].items():
            out[key] = out.get(key, 0) + int(verdict["counts"])
    return out


def runs_suffice(counts: list) -> bool:
    """Whether every problem has `PRICE_RUNS` counted measurements: the runs may stop."""
    t = tally(counts)
    return bool(t) and all(v >= PRICE_RUNS for v in t.values())


def first_counted(counts: list, values: list, n: int = PRICE_RUNS) -> dict:
    """Each problem's first ``n`` counted values, in run order.  ``counts[i]`` is run i's
    `run_counts2`, ``values[i][key]`` its measurement of the problem.  A problem short of ``n``
    keeps what it has, which `judge_pr2` refuses."""
    out = {}
    for c, v in zip(counts, values):
        for key, verdict in c["problems"].items():
            got = out.setdefault(key, [])
            if verdict["counts"] and len(got) < n:
                got.append(v[key])
    return out


def dry_run_passes(counts: list) -> bool:
    """The dry run's bar, the evaluation's own need: `DRY_RUNS` runs, and among them every
    problem with `PRICE_RUNS` counted measurements."""
    return len(counts) == DRY_RUNS and runs_suffice(counts)


def judge_e4h2(verdicts: dict, problems=None) -> dict:
    keys = list(problems or new_problems())
    missing = [k for k in keys if k not in verdicts]
    if missing:
        raise ValueError(f"no E4 verdict for {missing}")
    met = [k for k in keys if verdicts[k]["met"]]
    return {"met": len(met) == len(keys), "count": f"{len(met)} of {len(keys)}",
            "not_met": [f"{a} {b}" for a, b in keys if not verdicts[(a, b)]["met"]]}


def judge_pr2(prices: dict, problems=None) -> dict:
    keys = list(problems or new_problems())
    short = [k for k in keys if len(prices.get(k, ())) != PRICE_RUNS]
    if short:
        raise ValueError(f"PR2 needs {PRICE_RUNS} counted measurements per problem; short on {short}")
    meds = [float(np.median(prices[k])) for k in keys]
    m = float(math.exp(np.mean(np.log(meds))))
    return {"met": bool(m <= PRICE_BAR), "geometric_mean": m, "bar": PRICE_BAR}


# ---------------------------------------------------------------------------
# the gates, the predictions and what each outcome would mean
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    bar: str


GATES = (
    Gate("E4-H2", "rule R2's hybrid's coupled accuracy on the new problems",
         f"H2 meets E4 on every new problem, steady and step: energy error <= {E4_ENERGY:g} and "
         f"max error <= the classical floor's + {E4_MAX_OF_RANGE:g} of the range, as registered "
         f"and lowered by {-DATUM_SHIFT_K:g} K (judge_e4_both_frames, unchanged)"),
    Gate("PR2", "rule R2's hybrid's price over the classical expert",
         f"the geometric mean over the new problems of H2's price, (t_H2 + t_c) / (t_C + t_c), each "
         f"the median of its first {PRICE_RUNS} counted measurements, is <= {PRICE_BAR:g}"),
)

PREDICTIONS = (
    "1. H2's energy error exceeds the floor by less than 1e-4 (0.01 percentage points) on every "
    "new problem, in both frames.",
    "2. So E4-H2 is met on every new problem.",
    "3. PR2 is met: H2's geometric mean is 0.65-0.85. H2r's is 0.30-0.60, and H2r is cheaper "
    "than H2 on every problem.",
    "4. U alone meets E4 on every layout without a round feature, and misses it on at least one "
    "with a ring.",
    "5. P-s0 errs above H2 on every new problem.",
)

OUTCOMES = {
    "both gates met": "on square-first layouts the hybrid is exact and cheaper; H2r's price says "
                      "how much of the saving needs no network at all",
    "E4-H2 not met": "U's form is not exact on some square; the record says where, and the "
                     "square path has a defect",
    "E4-H2 met, PR2 not met": "U's form costs too much even on squares; reuse (H2r) is the "
                              "square path",
}


def text() -> str:
    lines = [f"Registered: {REGISTERED}.", "Settled choices:"]
    lines += [f"  {k}: {v}" for k, v in SETTLED.items()]
    lines += ["Revised after the first dry run:"] + [f"  {k}: {v}" for k, v in REVISED.items()]
    lines += ["Gates:"] + [f"  {g.key} ({g.title}): {g.bar}" for g in GATES]
    lines += ["Predictions:"] + [f"  {p}" for p in PREDICTIONS]
    lines += ["What each outcome would mean:"] + [f"  {k}: {v}" for k, v in OUTCOMES.items()]
    return "\n".join(lines)
