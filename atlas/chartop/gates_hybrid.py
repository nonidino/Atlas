"""The registration after section 11.14: the fixed rule's hybrid H, judged on E4 and on a
price bar on six workbench scenarios that no chart-operator code had used.  ACCEPTED BY THE
OWNER, AS DRAFTED, AND HASH-PINNED ON 2026-11-07, BEFORE ANY EXPERT BUT THE CLASSICAL ONE
READ THE NEW PROBLEMS.

The owner accepted the draft of [[chart-operator-kernel-exact-forms]] section 11.15 (commit
6d1b9ca) in chat on 2026-11-07 ("Accept as drafted"), with its three recommended choices
(`SETTLED`):
- the problems: the six scenarios of `benchmarks_workbench`, each steady and one step, 12
  problems;
- PR's bar: 0.90;
- what is pinned: this file, `benchmarks_workbench`, the code H runs (`forms`, `host`) and
  the classical record, beside `gates`, `gates_forms` and `gates_prior`, which are pinned
  already.  H has no weights, so its code is what defines it.
The owner approved the pin the same day, in chat ("Pin it").
``scripts/chartop_register_hybrid.py --pin`` writes ``out/chartop/registered-hybrid.txt`` with
the sha256 of those files, line endings normalised, so the test fails if any of them
changes.  It records beside them, by hash, `benchmarks` (whose helpers build the new
problems; pinned by the earlier registration) and the control's record, and by value the
arm's form, the rule's picks, the time steps and the price's protocol.

**The arm, H.**  It has no weights, and nothing is trained.
- The classical solve on every piece `rule_pieces` names: every piece whose chart is a
  Winslow chart, and every piece that owns a Dirichlet side.  Both functions are moved here
  verbatim from ``scripts/chartop_forms_diagnose.py``, where section 11.14 fixed the rule
  before measuring it; a test checks the two copies are the same text.
- On every other piece, U's form without its networks: S's form (split, corner bubble,
  strips for j = 9-15) with zero heads (`U_FORM`), its networks replaced by a stand-in that
  answers U's zeros, bit for bit (section 11.13).
- The coupled solve, as every arm has it.

**Reported beside, not judged** (`REPORTED`): the classical expert, which gives the floor and
the price's denominator; U, with its networks, on every piece; the trained P-s0, its
weights checked against ``weights-sha256.txt`` as everywhere; the rule's picks; and H's
price at one torch thread.

**The gates** (`GATES`).
- E4-H: H meets E4 on every new problem, steady and step.  E4 is `judge_e4_both_frames`,
  read from `gates_forms` unchanged: an energy error of at most `E4_ENERGY` and a max error
  of at most the classical floor's plus `E4_MAX_OF_RANGE` of the range, as registered and
  lowered by 300 K.
- PR: the geometric mean over the new problems of H's price is at most `PRICE_BAR`.  H's
  price on a problem (`price`) is (t_H + t_c) / (t_C + t_c): t_H the time of H's responses on
  every piece, t_C the classical expert's and t_c the coupled solve's, which both share.  It
  is the median of `PRICE_RUNS` runs.
The registration's claim, that the rule makes a usable expert on new layouts, holds only if
both gates are met.

**How the price is measured** (as section 11.14, with three changes that reduce how much one
run's drift can matter): `PRICE_THREADS` torch threads; each time the best of `PRICE_REPS`
repetitions, the arms taking turns within each repetition in the order `PRICE_ORDER`; three
runs, each in its own process, and the median of the three per problem.  A run counts only
if (`run_counts`) the laptop is on AC at its start and end, no sleep or power event
(`POWER_EVENT_IDS`) falls in its window, no other process uses more than `VALID_OTHER_CORE`
of a core at its start or end, and the classical expert takes `VALID_MS_PER_PIECE` per
piece.  A void run is run again, at most `PRICE_RERUNS_MAX` more times, and the record
keeps every run.  The accuracy verdicts are computed once, and no timing touches them.

**The problems and the control.**  The new problems (`NEW_PROBLEMS`) are measured with the
classical solver only, before the one evaluation (``scripts/chartop_register_hybrid.py``):
a problem whose classical floor fails E4 in either frame is dropped, not redesigned, and
the record says so.  The rule's picks on each are recorded there, and the evaluation stops
if they differ.  As the control, the six held-out problems of section 11.14 (`CONTROL`) are
run beside: H must reproduce section 11.14's errors bit for bit (`CONTROL_RECORD`), and
P-s0 the one evaluation's, or the evaluation stops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import benchmarks_workbench as BW
from . import gates as G
from . import gates_forms as GF
from . import gates_prior as GP
from . import host as hs

REGISTERED = ("accepted by the owner on 2026-11-07, in chat ('Accept as drafted'), as drafted "
              "in chart-operator-kernel-exact-forms section 11.15 (commit 6d1b9ca), with its "
              "three recommended choices; hash-pinned with the owner's approval the same day "
              "('Pin it'), before any expert but the classical one read the new problems")

#: the draft's three open choices, as the owner settled them (the recommended options)
SETTLED = {
    "the problems": "the six scenarios of benchmarks_workbench, steady and one step: 12 problems",
    "PR's bar": "0.90, a saving of at least 10%",
    "what is pinned": "this file, benchmarks_workbench, the code H runs (forms, host) and the "
                      "classical record, beside gates, gates_forms and gates_prior",
}

# ---------------------------------------------------------------------------
# what is kept, read from gates_forms and gates_prior
# ---------------------------------------------------------------------------

KIND, M_PORTS, BENCHMARK_N, DATUM_SHIFT_K = GF.KIND, GF.M_PORTS, GF.BENCHMARK_N, GF.DATUM_SHIFT_K
E4_ENERGY, E4_MAX_OF_RANGE = G.E4_ENERGY, G.E4_MAX_OF_RANGE
judge_e4_both_frames = GF.judge_e4_both_frames

# ---------------------------------------------------------------------------
# the problems
# ---------------------------------------------------------------------------

NEW_NAMES = BW.NAMES
NEW_PROBLEMS = tuple((name, problem) for name in BW.NAMES for problem in BW.PROBLEMS)
#: the control: section 11.14's six problems, `gates_prior`' held-out ones
CONTROL_VARIANT = GP.HELD_OUT_VARIANT
CONTROL = GP.HELD_OUT
CONTROL_RECORD = "out/chartop/forms-diagnose-h3.json"
#: the classical measurement of the new problems, written before the one evaluation
CLASSICAL_RECORD = "out/chartop/registration-hybrid.json"

# ---------------------------------------------------------------------------
# the arm
# ---------------------------------------------------------------------------

#: U: `forms.KernelExpert`'s arguments for S's form with zero heads, built after
#: ``torch.manual_seed(seed)`` (``scripts/chartop_forms_diagnose.zero_heads``)
U_FORM = {"form": "split", "backbone": "unet", "size": "1M", "kind": KIND, "m": M_PORTS,
          "p_size": None, "bubble": "corner", "strips": True, "seed": 0}
REPORTED_TRAINED = "P-s0"
REPORTED = ("the classical expert (the floor, and the price's denominator)",
            "U, with its networks, on every piece",
            "the trained P-s0 (weights checked against weights-sha256.txt)",
            "the rule's picks",
            "H's price at one torch thread")


def dirichlet_sides(L: hs.Layout) -> list:
    """[(piece, side)] for every boundary side that holds Dirichlet data."""
    return sorted((int(e.owner[0]), int(e.owner[1])) for e in L.edges
                  if e.other is None and e.bc is not None and e.bc[0] == "dirichlet")


def rule_pieces(L) -> list:
    """The fixed rule, set before it was measured (the owner's go, 2026-11-07): the
    classical solve on every piece whose chart is a Winslow chart, and on every piece that
    owns a Dirichlet side.  The chart says where the plain square's principal part is
    inexact; the boundary data say where the low sines are excited."""
    winslow = {i for i, p in enumerate(L.problems) if p.chart.method.startswith("winslow")}
    return sorted(winslow | {i for i, _ in dirichlet_sides(L)})


# ---------------------------------------------------------------------------
# the price
# ---------------------------------------------------------------------------

PRICE_BAR = 0.90
PRICE_THREADS = 4
PRICE_THREADS_REPORTED = 1
PRICE_REPS = 5
PRICE_RUNS = 3
PRICE_RERUNS_MAX = 3
PRICE_ORDER = ("classical", "H", "P-s0", "coupled solve")
VALID_MS_PER_PIECE = (30.0, 60.0)
VALID_OTHER_CORE = 0.10
POWER_EVENT_IDS = (105, 506, 507)


def price(t_h: float, t_classical: float, t_coupled: float) -> float:
    """H's price on one problem in one run: (t_H + t_c) / (t_C + t_c)."""
    return (t_h + t_coupled) / (t_classical + t_coupled)


def run_counts(state: dict) -> dict:
    """Whether a price run counts.  ``state``: ``plugged_start``, ``plugged_end`` (bool),
    ``power_events`` (the Ids of the events in the run's window), ``other_core_max`` (the
    largest share of one core any other process used at the run's start or end) and
    ``classical_ms_per_piece`` (the classical expert's best time per piece, per problem)."""
    failed = []
    if not (state["plugged_start"] and state["plugged_end"]):
        failed.append("not on AC at its start and end")
    events = sorted({int(i) for i in state["power_events"]} & set(POWER_EVENT_IDS))
    if events:
        failed.append(f"power events {events} in its window")
    if state["other_core_max"] > VALID_OTHER_CORE:
        failed.append(f"another process used {state['other_core_max']:.0%} of a core")
    lo, hi = VALID_MS_PER_PIECE
    out = [k for k, v in state["classical_ms_per_piece"].items() if not lo <= v <= hi]
    if out:
        failed.append(f"the classical expert outside {lo:g}-{hi:g} ms a piece on {sorted(out)}")
    return {"counts": not failed, "failed": failed}


def pr_measure(prices: dict) -> float:
    """``prices[(name, problem)]``: H's price in each counted run.  The geometric mean,
    over the new problems, of each problem's median."""
    missing = [k for k in NEW_PROBLEMS if k not in prices]
    if missing:
        raise ValueError(f"no price for {missing}")
    meds = [float(np.median(prices[k])) for k in NEW_PROBLEMS]
    return float(math.exp(np.mean(np.log(meds))))


def judge_pr(prices: dict) -> dict:
    short = [k for k in NEW_PROBLEMS if len(prices.get(k, ())) != PRICE_RUNS]
    if short:
        raise ValueError(f"PR needs {PRICE_RUNS} counted runs per problem; short on {short}")
    m = pr_measure(prices)
    return {"met": bool(m <= PRICE_BAR), "geometric_mean": m, "bar": PRICE_BAR,
            "medians": {f"{a} {b}": float(np.median(prices[(a, b)])) for a, b in NEW_PROBLEMS}}


def judge_e4h(verdicts: dict) -> dict:
    """``verdicts[(name, problem)]``: `judge_e4_both_frames`' result for H on that problem.
    Met only if H meets E4 on every new problem judged."""
    missing = [k for k in NEW_PROBLEMS if k not in verdicts]
    if missing:
        raise ValueError(f"no E4 verdict for {missing}")
    met = [k for k in NEW_PROBLEMS if verdicts[k]["met"]]
    return {"met": len(met) == len(NEW_PROBLEMS), "count": f"{len(met)} of {len(NEW_PROBLEMS)}",
            "not_met": [f"{a} {b}" for a, b in NEW_PROBLEMS if not verdicts[(a, b)]["met"]]}


# ---------------------------------------------------------------------------
# the gates, the predictions and what each outcome would mean
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    bar: str


GATES = (
    Gate("E4-H", "the hybrid's coupled accuracy on the new problems",
         f"H meets E4 on every one of the {len(NEW_PROBLEMS)} new problems, steady and step: "
         f"energy error <= {E4_ENERGY:g} and max error <= the classical floor's + "
         f"{E4_MAX_OF_RANGE:g} of the range, as registered and lowered by {-DATUM_SHIFT_K:g} K "
         f"(judge_e4_both_frames, unchanged)"),
    Gate("PR", "the hybrid's price over the classical expert",
         f"the geometric mean over the new problems of H's price, (t_H + t_c) / (t_C + t_c), each "
         f"the median of {PRICE_RUNS} counted runs, is <= {PRICE_BAR:g}"),
)

PREDICTIONS = (
    "1. On heated-strip, whose 40 pieces are squares of one material, U alone, with no classical "
    "piece at all, is within 0.01 percentage points of the classical floor, steady and step, in "
    "both frames.",
    "2. H's energy error exceeds the classical floor by less than 1 percentage point on wall-2, "
    "heated-strip, bimetal-arc and plate-hole, steady and step, in both frames; so H meets E4's "
    "energy bar on those eight wherever the floor is below 1%.",
    "3. Of the six steady problems, H exceeds the floor most on plate-insert or fast-heat. No "
    "verdict is predicted for those two.",
    "4. PR is met: the geometric mean of H's prices is 0.70-0.88.",
    "5. P-s0 errs above H on every new problem, and meets E4 on fewer of them.",
)

OUTCOMES = {
    "both gates met": "the hybrid is the CO-D expert to beat on such problems; a learned "
                      "correction is worth training only if it meets E4 at a lower price",
    "E4-H not met": "the rule is not used as it stands; where it fails, read against "
                    "predictions 2 and 3, says which criterion it lacks",
    "E4-H met, PR not met": "the hybrid is accurate but saves too little on this laptop; the "
                            "classical solve stays the CO-D expert here",
}


def text() -> str:
    lines = [f"Registered: {REGISTERED}.", "Settled choices:"]
    lines += [f"  {k}: {v}" for k, v in SETTLED.items()]
    lines += ["Gates:"] + [f"  {g.key} ({g.title}): {g.bar}" for g in GATES]
    lines += ["Predictions:"] + [f"  {p}" for p in PREDICTIONS]
    lines += ["What each outcome would mean:"] + [f"  {k}: {v}" for k, v in OUTCOMES.items()]
    return "\n".join(lines)
