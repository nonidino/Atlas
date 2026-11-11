"""The registration after the Stage 0 rework (Phase 4 of [[chart-operator-stage-0-rework]],
section 18): TL, the two-level coarse-exact CO-D expert, trained on the relative objective,
judged with its untrained floor TL0 and with S' reused.  ACCEPTED BY THE OWNER AS DRAFTED ON
2026-11-10 (`REGISTERED`), AND HASH-PINNED THE SAME DAY WITH THE OWNER'S APPROVAL, AFTER THE PRICE
PROTOCOL'S DRY RUN PASSED (``out/chartop/registered-twolevel.txt``,
`scripts/chartop_register_twolevel.py`), BEFORE ANY NETWORK OF TL WAS TRAINED.

**Decided by the owner already** (in chat, 2026-11-09 and 2026-11-10): the rework's
formulation; ``n_c in {8, 16, 32}`` with 16 the default (4 added for the curve); E4 the only
accuracy gate, CX added, E1 and SE3 reported; branch 2c's corner treatment B, blocks of
``n/8``, in TL and TL0 ("Adopt B, n/8 (Recommended)"); the NC gate's bar as proposed ("As
proposed (Recommended)"); the comparison budget of 37,500 steps ("37,500 steps
(Recommended)").

**The arms** (`ARMS`):
- TL: `twolevel.TwoLevelExpert` with S's form (`gates_forms.ARMS["S"]`: split, corner bubble,
  strips for j = 9-15, learned hats), ``n_c = 16`` and the corner treatment (``corners = 8``),
  on the relative objective with methods 1-3 (`twolevel`), the start of Phase 3's probe's
  winner (`TL_START`), 37,500 steps, training seeds 0 and 1;
- TL0: the same with zero heads, untrained, at ``n_c`` in `TL0_CURVE` (the floor and the
  ``n_c*`` curve); at 16 it is TL's own floor, with the corner treatment;
- S': reused, not trained again (`gates_prior.REUSED`).

**What is judged, once**, as `gates_forms` judged its main arm: the seed-0 run of TL is chosen
on the validation split and evaluated once on the fresh test split (`TEST_SEEDS`) and the
held-out benchmark problems (`HELD_OUT`, `benchmarks`' ``variant="held-out"``).  The six
problems Phase 2 and the corner check read (`SEEN`) are reported beside, as seen.  K, D and L
are judged on every run.

**The gates** (`GATES`): K, D, L, E3, E4, SE1, SE2, DUAL and KT as `gates_prior` worded them
on its problems (bars read from `gates_forms`); **CX**, coarse exactness on the arm's own
space, at most `CX_BAR` on every test and benchmark piece; **NC**, the network's
contribution (`twolevel.rho` on the test split, TL against TL0 at its coarse size): median at
least `NC_MEDIAN`, 10th percentile at least `NC_P10`, and on no held-out benchmark problem a
coupled energy error above TL0's by more than `NC_ROUNDING` of it, in either frame; **PR**,
the price under R2's protocol against the classical solve, H2r (`H2R_PRICE`) and TL0.  E1,
SE3 and S11 are reported, as the rework's section 3.5 settled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gates_forms as GF
from . import gates_prior as GP

REGISTERED = ("accepted by the owner on 2026-11-10, in chat ('Accept as drafted (Recommended)'), as "
              "drafted in chart-operator-stage-0-rework section 18 (commit 2fca5aa), with its five "
              "predictions, before any network of TL was trained; hash-pinned with the owner's approval "
              "the same day ('Pin it (Recommended)'), after the price protocol's dry run passed on the "
              "pricing CPU (commit 345055c)")

#: the owner's decisions so far, quoted from chat
SETTLED = {
    "the corner treatment": "B with blocks of n/8 in TL and TL0 ('Adopt B, n/8 (Recommended)', 2026-11-10)",
    "the NC bar": "as proposed ('As proposed (Recommended)', 2026-11-10)",
    "the budget": "the comparison budget, 37,500 steps ('37,500 steps (Recommended)', 2026-11-10)",
}

# ---------------------------------------------------------------------------
# kept, read from gates_forms and gates_prior
# ---------------------------------------------------------------------------

KIND, M_PORTS, N_COLUMNS, N_LEARNED = GF.KIND, GF.M_PORTS, GF.N_COLUMNS, GF.N_LEARNED
N_VAL, N_TEST = GF.N_VAL, GF.N_TEST
BENCHMARK_NAMES, BENCHMARK_PROBLEMS, BENCHMARK_N = GP.BENCHMARK_NAMES, GP.BENCHMARK_PROBLEMS, GP.BENCHMARK_N
HELD_OUT_VARIANT, HELD_OUT = GP.HELD_OUT_VARIANT, GP.HELD_OUT
SEEN = tuple((name, problem) for name in BENCHMARK_NAMES for problem in BENCHMARK_PROBLEMS)
E4_ENERGY, E4_MAX_OF_RANGE = GF.E4_ENERGY, GF.E4_MAX_OF_RANGE
DATUM_SHIFT_K = GF.DATUM_SHIFT_K
CEILING_USD, BOX_MAX_USD_PER_HOUR = GF.CEILING_USD, GF.BOX_MAX_USD_PER_HOUR
REUSED = GP.REUSED
judge_k, judge_d, judge_l, judge_e1 = GF.judge_k, GF.judge_d, GF.judge_l, GF.judge_e1
judge_e4_both_frames, e_h_learned = GF.judge_e4_both_frames, GF.e_h_learned

# ---------------------------------------------------------------------------
# what is new
# ---------------------------------------------------------------------------

N_C = 16                             # TL's coarse size (the owner's default)
CORNERS = 8                          # branch 2c's B: blocks of n/8 (`coarse_patches.FRAC`)
TL0_CURVE = (4, 8, 16, 32)           # TL0 untrained, for the floor and n_c*
STEPS = 37_500                       # the comparison budget (S' was trained at it)
SEEDS = (0, 1)
#: Phase 3's probe's winner (section 14): S's weights with their heads, or zero heads
TL_START = "S"
#: S's weights (`gates_forms.ARMS["S"]`, 150,000 steps), sha256 as downloaded
#: (``out/chartop/runs/forms/weights-sha256.txt``), loaded with their heads
INIT_RUN, INIT_SHA256, HEAD_INIT = "S", "e17f881b2b5bdb90145254cb57e9c2f0912c01b3957a3f6b20c01e3e8a8fa35d", "loaded"
#: the probe's rate (AdamW); the rest of the optimiser, the schedule and the batches as S'
LR = 3e-4
#: the fresh test split: example i has seed TEST_SEED_BASE + i, drawn only at the one
#: evaluation; no other split uses this range (3,000,000 and 5,000,000 have been evaluated)
TEST_SEED_BASE = 6_000_000
TEST_SEEDS = (TEST_SEED_BASE, TEST_SEED_BASE + N_TEST)
CX_BAR = 1e-10
NC_MEDIAN = 0.3
NC_P10 = 0.0
#: NC's benchmark clause: TL's coupled energy error may exceed TL0's (n_c = 16, corners) by at
#: most this fraction of it on a held-out problem, in either frame (the rounding of Theorem 2's
#: coupled solve, Phase 2's controls: within 1e-13 of the floor)
NC_ROUNDING = 1e-9
#: H2r's registered price (`gates_squares`' evaluation, section 11.19)
H2R_PRICE = 0.306

COMMON = dict(GF.COMMON, bubble=GF.ARMS["S"]["bubble"], strips=GF.ARMS["S"]["strips"],
              hats="learned", strips_in_training=False, objective="relative", coarse=N_C,
              corners=CORNERS, drop_easy=1e-8, lr=LR, head_init=HEAD_INIT)

ARMS = {
    "TL":  {"what": "S's form, then the coarse correction at n_c = 16 with B's corner patches "
                    "(n/8), trained on the relative objective (methods 1-3)",
            "start": TL_START, "steps": STEPS, "seeds": SEEDS},
    "TL0": {"what": "TL with zero heads, untrained: the floor (n_c = 16, corners) and the curve",
            "n_c": TL0_CURVE, "steps": 0, "seeds": ()},
}


def arm_config(arm: str = "TL") -> dict:
    """TL's training configuration: every key a `train.Arm` field but ``m``.  ``init_from``
    is the box's own path to S's ``best.pt``, checked against `INIT_SHA256` before the run."""
    if arm != "TL":
        raise ValueError("only TL is trained")
    return dict(COMMON, steps=STEPS)


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    bar: str


_PRIOR = {g.key: g for g in GP.GATES}


def _swap(text: str, *pairs) -> str:
    """``text`` with each ``(old, new)`` replaced, refusing an ``old`` that is not there."""
    for old, new in pairs:
        if old not in text:
            raise ValueError(f"{old!r} is not in the bar {text!r}")
        text = text.replace(old, new)
    return text


_TEST = f"the fresh test split (seeds {TEST_SEEDS[0]:,} to {TEST_SEEDS[1] - 1:,})"

GATES = (
    Gate("K", _PRIOR["K"].title, _PRIOR["K"].bar),
    Gate("D", _PRIOR["D"].title, _PRIOR["D"].bar),
    Gate("L", _PRIOR["L"].title, _PRIOR["L"].bar),
    Gate("E3", _PRIOR["E3"].title, _PRIOR["E3"].bar),
    Gate("E4", _PRIOR["E4"].title, _PRIOR["E4"].bar),
    Gate("CX", "coarse exactness, on the arm's own coarse space",
         f"every piece of {_TEST} and every held-out benchmark piece: ||P_e^T (B Q - A_I H)|| / "
         f"||P_e^T B Q|| <= {CX_BAR:g}, and likewise the particular field wherever the piece has a load"),
    Gate("NC", "the network's contribution",
         f"TL's seed-0 run on {_TEST}, against TL0 at n_c = {N_C} with the corner treatment: "
         f"rho = 1 - eps_TL / eps_TL0 over the {N_LEARNED} learned columns (pieces where TL0 leaves "
         f"nothing to learn counted, not divided by): median >= {NC_MEDIAN:g} and 10th percentile >= "
         f"{NC_P10:g}; and on every held-out benchmark problem, in both frames, TL's energy error "
         f"<= TL0's x (1 + {NC_ROUNDING:g})"),
    Gate("SE1", _PRIOR["SE1"].title, _PRIOR["SE1"].bar),
    Gate("SE2", _PRIOR["SE2"].title, _PRIOR["SE2"].bar),
    Gate("DUAL", _PRIOR["DUAL"].title, _PRIOR["DUAL"].bar),
    Gate("KT", _PRIOR["KT"].title, _PRIOR["KT"].bar),
    Gate("PR", "the price, under R2's protocol",
         f"TL's price (its time over the classical solve's, the geometric mean over the problems) "
         f"measured under R2's protocol on a rented whole machine and reported against the classical "
         f"solve (1), H2r ({H2R_PRICE:g}) and TL0; TL is 'faster than X' only where measured so"),
)

REPORTED = (
    Gate("E1", _PRIOR["E1"].title, _swap(_PRIOR["E1"].bar, ("the seed-0 run of each arm", "TL's seed-0 run"),
                                                           ("the fresh test split (seeds 5,000,000 to "
                                                            "5,002,399)", _TEST)) + "; reported, not judged"),
    Gate("SE3", _PRIOR["SE3"].title, _PRIOR["SE3"].bar + "; reported"),
    Gate("S11", _PRIOR["S11"].title, _PRIOR["S11"].bar),
    Gate("LEDGER", "the network ledger",
         "reported: rho per piece and by column group (hats, j = 1-3, j = 4-8), rho of the particular "
         "field, the coupled attribution (coarse first, then the network) against the floor, the "
         "equivalent coarse size n_c* from TL0's curve, kappa_kill, and the usage controls (shuffled "
         "inputs, frozen weights)"),
)


#: section 18's predictions, accepted with the draft; their readings are written into the one
#: evaluation's script before it runs, and checked against these words
PREDICTIONS = (
    "1. CX, K, D and L hold on every run, by structure: Proposition A and Lemma B on V_e, and the "
    "split form.",
    "2. [AI Inference] NC is met: TL's median test rho is at least 0.3.",
    "3. [AI Inference] TL meets E4 on all six held-out problems.",
    "4. [AI Inference] By group, rho is lowest on the hats: on the probe's pieces the hats gave back "
    "what they had gained.",
    "5. [AI Inference] PR: TL is not faster than the classical solve. U's forward with its strips "
    "already cost 1.48 and 0.78 of an exact solve at n = 64 and 128 (section 13).",
)

# ---------------------------------------------------------------------------
# the judgements (pure functions of measured numbers)
# ---------------------------------------------------------------------------


def judge_cx(cx_modes, cx_particular) -> dict:
    """``cx_particular`` may hold ``None`` for a piece without a load (no ratio)."""
    m = np.asarray(cx_modes, float)
    p = np.asarray([x for x in cx_particular if x is not None], float)
    worst = max(float(m.max()) if m.size else 0.0, float(p.max()) if p.size else 0.0)
    return {"met": bool(m.size and worst <= CX_BAR), "max": worst, "pieces": int(m.size)}


def judge_nc(rho, e4_tl: dict, e4_tl0: dict) -> dict:
    """``rho`` per test piece (``nan`` where TL0 left nothing to learn); ``e4_tl[(name,
    problem, frame)]`` and ``e4_tl0[...]`` the coupled energy errors on the held-out problems."""
    r = np.asarray(rho, float)
    ok = r[np.isfinite(r)]
    med = float(np.median(ok)) if ok.size else None
    p10 = float(np.percentile(ok, 10)) if ok.size else None
    keys = [(b, p, f) for b, p in HELD_OUT for f in ("registered", "lowered")]
    if sorted(e4_tl) != sorted(keys) or sorted(e4_tl0) != sorted(keys):
        raise ValueError("NC reads every held-out problem in both frames")
    worse = [k for k in keys if e4_tl[k] > e4_tl0[k] * (1.0 + NC_ROUNDING)]
    return {"met": bool(med is not None and med >= NC_MEDIAN and p10 >= NC_P10 and not worse),
            "median": med, "p10": p10, "pieces": int(ok.size), "nothing_to_learn": int(r.size - ok.size),
            "worse_than_TL0": worse}


def text() -> str:
    lines = [f"REGISTERED: {REGISTERED}", "", "SETTLED"] + [f"  {k}: {v}" for k, v in SETTLED.items()]
    lines += ["", "THE GATES"] + [f"  {g.key:6s} {g.title:58s} {g.bar}" for g in GATES]
    lines += ["", "REPORTED"] + [f"  {g.key:6s} {g.title:58s} {g.bar}" for g in REPORTED]
    lines += ["", "THE ARMS (TL: " + ", ".join(f"{k} {v}" for k, v in COMMON.items()) + ")"]
    lines.append(f"  TL   {ARMS['TL']['what']}; from {INIT_RUN}'s weights ({INIT_SHA256[:16]}..., heads "
                 f"{HEAD_INIT}); {STEPS:,} steps at lr {LR:g}, training seeds {', '.join(str(s) for s in SEEDS)}")
    lines.append(f"  TL0  {ARMS['TL0']['what']}; n_c {', '.join(str(c) for c in TL0_CURVE)}")
    lines.append("  S'   reused, not trained: " + ", ".join(f"{r} {h[:16]}..." for r, h in REUSED.values()))
    lines += ["", f"THE PROBLEMS: held-out {', '.join(f'{b} {p}' for b, p in HELD_OUT)}; the seen six reported"]
    lines += ["", "THE PREDICTIONS"] + [f"  {p}" for p in PREDICTIONS]
    lines += ["", f"COST: at most ${CEILING_USD:.2f} in all, each box at most ${BOX_MAX_USD_PER_HOUR:.2f} an hour"]
    return "\n".join(lines)
