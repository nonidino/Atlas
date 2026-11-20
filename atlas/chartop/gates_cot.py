"""The coolant plan's registration (CO-T on laminar cold plates; [[chart-operator-stage-0b-and-1-plan]]
sections 7 and 15).  **A DRAFT THAT WAS NEVER REGISTERED.**  On 2026-11-19 the owner stopped the
coolant's learned track in Phase 3 ("Stop learned track"), before any dataset, after the capacity
gate CAP failed at 1.1M and 3M parameters (plan section 16.8).  `REGISTERED` stays None and
`STOPPED` says why; the draft is kept as the record of what would have been judged.

**Decided by the owner already** (`SETTLED`, quoted from chat): the case and its regime,
model (a), the Peclet cap, the order of the phases, the frozen dataset, a rented box with
37,500 steps per run in $5 segments, cards for every model, and the amendment of 2026-11-19:
price is a reading, the gates are accuracy and capacity.

**The arms** (`ARMS`):
- **arm R**: TL0 for CO-T at the network's coarse size plus the network's fine corrections,
  trial and adjoint from one network (b -> -b), each followed by the coarse correction
  (`transport_torch.ArmR`), in the correction form CAP chose (`CAP_FORM`);
- **arm P**: arm R with a learned coarse basis P_theta (`transport_model`'s design: four
  corrections per fine node to the bilinear functions of its coarse cell, times a cell
  bubble), built in Phase 5;
- **TL0**: arm R with zero heads, untrained: the control;
- **classical**: the exact piece solves, the floor E4 is set from.
Three seeds each for P and R, 37,500 steps, on the frozen training split
(`data-manifest-train.json`), the label-free relative objective (`transport_torch.objective`).

**What is judged, once** (Phase 6): for each arm, its seed-0 run chosen as the last checkpoint
(no early stopping on the evaluation layouts), on the six evaluation problems
(`EVAL_PROBLEMS`: three layouts at two speeds, `out/chartop/cot/eval-layouts/manifest.json`),
whose classical floors were measured in Phase 3 and nothing else.  The development layouts
are reported beside, as seen.
"""

from __future__ import annotations

from dataclasses import dataclass

#: never set: the draft was not registered
REGISTERED = None
#: why the draft stops here (the owner, in chat)
STOPPED = ("the owner, 2026-11-19, 'Stop learned track': CAP failed at 1.1M (both forms) and 3M (free form) "
           "on the probe's six pieces; no dataset, no training, no registration (plan section 16.8)")

SETTLED = {
    "the case": "the coolant (CO-T) first, laminar cold plates, Pe at most 200 per piece (2026-11-18)",
    "the model": "(a), side walls only ('(a) Side walls only (Recommended)', 2026-11-18)",
    "training": "a rented box under $0.70/h, 37,500 steps per run, in segments of about $5 (2026-11-18)",
    "price": "not a gate: measured and reported; no 'faster' claim without a measurement (2026-11-19)",
    "the gates": "accuracy (E4, CL-A) and capacity (CAP), the safety switch kept unpriced (2026-11-19)",
    "E4 in Phase 2": "'Floor + CO-D margins' (2026-11-19)",
    "CAP": "the bar 0.30 on every group's median, 3,000 steps, both forms at 1M ('0.30 (Recommended)', "
           "'Both forms, 3,000 steps (Recommended)', 2026-11-19); after check 5, the trial groups and the "
           "particular field judged, the adjoint reported ('Trial and particular (Recommended)', 2026-11-19)",
    "the loss": "realigned after check 5 failed: trial columns weighted by the training layouts' port traces, the "
                "adjoint at a small weight, terms TL0 solves left out; re-checked against the same 0.80 bar "
                "('Realign loss, re-check (Recommended)', 2026-11-19)",
}

N_PORT_MODES = 16
#: the network's coarse size on a piece of n (TL0 at the same size is the control)
def n_c(n: int) -> int:
    return min(16, n // 2)


#: E4, as Phase 2 registered it and the draft keeps it: against the full-domain solve on the same mesh
E4_ENERGY_MARGIN, E4_MAX_MARGIN, E4_OUTLET_MARGIN = 0.02, 0.01, 0.01
#: CAP (fixed before the capacity probe ran)
CAP_FRACTION, CAP_STEPS = 0.30, 3000
#: the correction form (set by the capacity probe's result; None until it has run)
CAP_FORM = None
#: structural bars
CX_BAR = 1e-10
K_BAR = 1e-10
SEEDS = (0, 1, 2)
STEPS = 37_500

EVAL_PROBLEMS = tuple((name, tier) for name in ("eval-two-devices", "eval-three-channels", "eval-arch")
                      for tier in ("slow", "fast"))


@dataclass(frozen=True)
class Gate:
    key: str
    title: str
    bar: str
    kind: str


GATES = (
    Gate("E4", "the coupled answer against the full-domain solve on the same mesh",
         f"on every evaluation problem: the coupled field's relative error in the diffusion norm at most the "
         f"classical 16-mode expert's own + {E4_ENERGY_MARGIN}, its largest nodal error at most the classical's "
         f"+ {E4_MAX_MARGIN} of the temperature range, and its outlet mixing-cup temperature error at most the "
         f"classical's + {E4_OUTLET_MARGIN} of the range", "accuracy (primary)"),
    Gate("CL-A", "the learned expert meets E4 where TL0 at its coarse size misses it",
         "on every evaluation problem where TL0 at n_c = min(16, n/2) misses E4, the arm's seed-0 run meets "
         "it; with no such problem CL-A is untestable and reported so. Arm P against arm R: whether "
         "learning the coarse basis is what pays", "accuracy (decisive)"),
    Gate("CAP", "capacity: the network fits the hardest pieces",
         f"on the capacity probe's six pieces (n = 128, Pe >= 180) and again in Phase 4 on six of the frozen "
         f"split's hardest pieces: the median error in the diffusion norm of each of the four trial groups "
         f"(hats, j1-3, j4-8, j9-15) and of the particular field at most {CAP_FRACTION} of TL0's within "
         f"{CAP_STEPS:,} steps; the adjoint groups reported (the owner, 2026-11-19, after check 5)", "capacity"),
    Gate("SEEDS", "robustness", "CL-A holds for every seed (0, 1, 2) of the winning arm", "robustness"),
    Gate("CX", "coarse exactness",
         f"||P^T r|| / ||P^T F|| at most {CX_BAR} for the trial, adjoint and particular outputs on every "
         f"evaluation piece", "structural"),
    Gate("K", "the uniform temperature reproduced on steady pieces",
         f"on every steady evaluation piece, the hat modes' sum within {K_BAR} of 1 (the hat corrections made "
         f"zero-sum on steady pieces, and Lemma B)", "structural"),
    Gate("D", "no flux from a uniform temperature",
         "on every steady evaluation piece, the port matrix applied to the uniform temperature's port "
         "coefficients is zero to rounding (follows from K)", "structural"),
    Gate("SAFE", "the safety switch",
         "on every piece, the learned modes are kept only if their residual in the exact dual norm is at "
         "most TL0's, else TL0's are used; E4 and CL-A are judged with the switch on", "structural"),
    Gate("PR", "price (reported, not a gate)",
         "the whole coupled solve's time over the classical solve's, on a rented whole machine with fixed BLAS "
         "threads and an A/A check, beside TL0 at n_c, TL0 at 32, finer cutting and the classical solve; with "
         "and without the switch", "reported"),
    Gate("AMORT", "amortisation (reported)", "piece solves needed to repay the training's cost", "reported"),
)

__all__ = ["CAP_FORM", "CAP_FRACTION", "CAP_STEPS", "CX_BAR", "E4_ENERGY_MARGIN", "E4_MAX_MARGIN",
           "E4_OUTLET_MARGIN", "EVAL_PROBLEMS", "GATES", "Gate", "K_BAR", "REGISTERED", "SEEDS", "SETTLED",
           "STEPS", "STOPPED", "n_c"]
