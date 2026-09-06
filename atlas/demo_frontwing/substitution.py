"""Beat 1 -- swap an expert into a seam and watch the compiler answer.

This is the sharpest thing the framework does and the cheapest to show, because
**the answer is a property of the DECLARATION and not of the weights.**  A
capability record says what an expert is: how far one step propagates
information, whether it embeds a global elliptic solve, whether its time step is
explicit, implicit, or neither.  `compile_scheme` reads those fields and either
certifies the decomposition or declines to.  Nothing in that path touches a
parameter, so a 20.8 M-parameter checkpoint and a hand-written record with the
same fields get the *same verdict* -- which is what makes this demo runnable on
a laptop with no network and no 83 MB of weights.

That is not a shortcut around the measurement.  The measurement already happened
and it is `W93`: `probe.support_reach` poked a delta at a real wake seam and
found Poseidon-T's response nonzero in **every one of the 128 seam cells**, 64
cells from the poke, where its record declares a stencil radius of 2.  A factor
of 32.  The declaration below carries the fields that measurement licensed, and
`atlas/cases/poseidon.py` -- which needs the actual checkpoint -- builds the same
record from the live object.  A test asserts the two agree field for field.

The three states, and why there are three
-----------------------------------------

| candidate | what changes | what the compiler says |
|---|---|---|
| `windowns` | nothing; the classical incumbent | **amber** -- no seam-specific complaint. The only decertifications reaching a fluid seam are the graph-wide unmeasured constants (W1, W3, W49, W56) |
| `poseidon_declared` | the record as the checkpoint's own wrapper declares it: `time_discretization=unknown`, radius 2, one sub-step | **amber, with three new reasons.** `R10/halo` can no longer decide the overlap; `R10/W60` says the elliptic content is undeclarable; `R2b/W46` holds the transmission rung down |
| `poseidon_probed` | the same, plus `elliptic_subsolve=embedded` -- what a projection-method-shaped operator actually is | **red.** `L2/R10` refuses: six agents of one governing family, so the family's region has been cut, and each agent now solves a global problem on its own piece of it |

The middle row is the honest one to sit on, because `elliptic_subsolve` for a
learned operator is genuinely undeclarable (**W60**: `elliptic_signature`
measures non-normality and was read as measuring globality, and the two come
apart for a self-adjoint operator).  The third row is what you get by declaring
the thing the architecture is.  **Both are shown, and the screen never presents
the third as measured.**

What this beat must not be read as saying
-----------------------------------------

It is **not** "neural operators do not work."  Poseidon-T is a good model.  It is
that *this framework cannot certify a domain decomposition whose agents are
globally receptive*, which is nearly every pretrained operator, because a global
receptive field is a fact about the architecture rather than about the physics.
The consequence for the programme is scheduled rather than hidden: climb the
ladder with classical experts, and price each learned substitution one seam at a
time (`case-study-ladder-to-f1` section 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..capability import (
    Differentiable,
    EllipticSubsolve,
    TimeDiscretization,
)
from ..compiler import compile_scheme
from ..cases import front_wing as F


# ---------------------------------------------------------------------------
# the candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One expert offered for the fluid windows, as a record patch."""

    key: str
    name: str
    #: what a non-specialist needs to know about this expert in one sentence
    plain: str
    #: where each declared field comes from -- a citation, not a claim
    provenance: str
    #: the fields that differ from the incumbent's record
    fields: dict[str, Any] = field(default_factory=dict)
    #: shown beside the verdict; empty for the incumbent
    caveat: str = ""

    @property
    def is_incumbent(self) -> bool:
        return not self.fields


#: The declaration `atlas/cases/poseidon.py` builds from the live checkpoint,
#: minus the two fields that need the object itself (`storage`, `dt_native`).
#: Those two reach no rule in this beat, and a test asserts the rest agree.
_POSEIDON = dict(
    time_discretization=TimeDiscretization.UNKNOWN,
    stencil_radius=2,
    substeps_per_macro_step=1,
    differentiable=Differentiable.NONE,
)

CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        key="windowns",
        name="WindowNS (classical, the incumbent)",
        plain="The solver that is actually running: a finite-volume "
              "Navier-Stokes step with its pressure solve lifted out and given "
              "to the composition layer. Every number on the rest of this "
              "screen comes from it.",
        provenance="atlas/cases/wing_fsi.py flow_capabilities: exposed, "
                   "explicit, stencil radius 2, 4 sub-steps per macro-step",
    ),
    Candidate(
        key="poseidon_declared",
        name="Poseidon-T, as its own wrapper declares it",
        plain="A frozen 20.8-million-parameter neural operator trained by "
              "someone else on a lot of fluid dynamics -- exactly the kind of "
              "model this whole approach is supposed to run on. Nothing about "
              "it is changed; only the description of it is handed to the "
              "compiler.",
        provenance="atlas/cases/poseidon.py poseidon_capabilities, which is "
                   "built from the live checkpoint. A learned one-shot map is "
                   "neither explicit nor implicit, so it declares unknown "
                   "(R2b); the elliptic content is undeclarable (W60).",
        fields=dict(_POSEIDON, elliptic_subsolve=EllipticSubsolve.UNKNOWN),
        caveat="The record is honest and the compiler still cannot decide the "
               "overlap: radius x sub-steps is an EXPLICIT agent's domain of "
               "dependence and this agent has not said its step is explicit.",
    ),
    Candidate(
        key="poseidon_probed",
        name="Poseidon-T, with the elliptic part declared",
        plain="The same model, described as what its architecture makes it: "
              "an operator whose output at any point depends on the input "
              "everywhere. That is what a global attention window buys you, "
              "and it is the one property a domain decomposition cannot "
              "tolerate.",
        provenance="the same record with elliptic_subsolve=embedded. NOT a "
                   "measurement: W60 records that the probe cannot settle this "
                   "field, because elliptic_signature measures non-normality "
                   "and globality is a different thing. It is the declaration "
                   "the architecture implies, offered so the refusal can be "
                   "seen rather than described.",
        fields=dict(_POSEIDON, elliptic_subsolve=EllipticSubsolve.EMBEDDED),
        caveat="DECLARED, not measured. W60 says this field is undeclarable "
               "for a learned operator; this row shows what follows if it is "
               "declared at the value the architecture implies.",
    ),
)

BY_KEY = {c.key: c for c in CANDIDATES}
DEFAULT = CANDIDATES[0].key


#: The measurement the middle row rests on, quoted rather than re-run: it needs
#: the checkpoint, a wake seam and about a minute, and it is already on the
#: record. The screen shows it as a citation with its own provenance.
W93_MEASUREMENT = dict(
    what="probe.support_reach on Poseidon-T at a real wake seam",
    declared_cells=2,
    measured_cells=128,
    distance_cells=64,
    factor=32,
    amplitudes="every amplitude from 1 to 1e-3",
    where="case-study-scaling-ladder-atlas-0.1, gap-worklist W93",
    sentence="the response is nonzero in every one of the 128 seam cells, 64 "
             "cells from the poke, where the record declares 2. A neural "
             "operator's receptive field is global by construction.",
)


# ---------------------------------------------------------------------------
# the swap
# ---------------------------------------------------------------------------


def patch_fluid(graph, fields: dict[str, Any], tiling=None):
    """Return `graph` with every fluid window's capability record patched.

    The structure and the suspension are untouched: this beat substitutes the
    expert in the *decomposed* agents, which is where a learned surrogate would
    actually go, and leaving the other two alone is what makes the verdict
    change attributable.
    """
    if not fields:
        return graph
    names = set((tiling or F.DEFAULT_TILING).names)
    return replace(graph, agents=[
        replace(a, capabilities=replace(a.capabilities, **fields))
        if a.agent_id in names else a
        for a in graph.agents])


def _rules_of(seams: dict) -> dict[str, dict[str, list]]:
    return {s: dict(refusals=list(m["refusals"]),
                    decertifications=list(m["decertifications"]))
            for s, m in seams.items()}


def evaluate(u, v, design: dict, h: float, tiling=None, motion: bool = True,
             candidates: tuple[Candidate, ...] = CANDIDATES) -> dict:
    """Compile the graph once per candidate and diff the answers.

    Returns one row per candidate carrying the per-seam verdict map, the
    graph-level verdict, and -- the part worth looking at -- the rules that
    appear or disappear at each seam **relative to the incumbent**.
    """
    from .engine import seam_verdicts

    tiling = tiling or F.DEFAULT_TILING
    base, _experts = F.build(u, v, motion=motion, design=design, h=h,
                             tiling=tiling)
    rows: dict[str, dict] = {}
    incumbent_rules: dict[str, dict] | None = None
    for c in candidates:
        g = patch_fluid(base, c.fields, tiling)
        r = compile_scheme(g)
        seams = seam_verdicts(g, r)
        rules = _rules_of(seams)
        if incumbent_rules is None:
            incumbent_rules = rules
        added: dict[str, dict] = {}
        for s, cur in rules.items():
            was = incumbent_rules.get(s, {"refusals": [], "decertifications": []})
            new_ref = [x for x in cur["refusals"] if x not in was["refusals"]]
            new_dec = [x for x in cur["decertifications"]
                       if x not in was["decertifications"]]
            if new_ref or new_dec:
                added[s] = dict(refusals=new_ref, decertifications=new_dec)
        rows[c.key] = dict(
            key=c.key, name=c.name, plain=c.plain, provenance=c.provenance,
            caveat=c.caveat, is_incumbent=c.is_incumbent,
            verdict=r.verdict.value,
            seams=seams,
            added=added,
            colours={s: m["colour"] for s, m in seams.items()},
            n_red=sum(1 for m in seams.values() if m["colour"] == "red"),
            n_amber=sum(1 for m in seams.values() if m["colour"] == "amber"),
            n_green=sum(1 for m in seams.values() if m["colour"] == "green"),
            declaration={k: getattr(v_, "value", v_)
                         for k, v_ in (c.fields or {}).items()},
        )
    #: **The refusal needs a decomposition, and with one window there is not
    #: one.** `L2/R10` fires only when the graph holds another agent of the same
    #: governing family -- the compile-time signature of "a larger region of this
    #: physics, of which this agent has been given a piece". On the single-window
    #: referent the fluid family has exactly one member, so R10's premise does
    #: not hold and the rule correctly stays quiet. That is **W114**, the premise
    #: check, working: before it, R10 refused every graph containing an embedded
    #: agent whether or not anything had been decomposed. So the panel says which
    #: column it is in, rather than letting a viewer read a quiet rule as a
    #: passing one.
    n_fluid = len(tiling.names)
    return dict(rows=rows, order=[c.key for c in candidates],
                measurement=dict(W93_MEASUREMENT),
                n_fluid=n_fluid,
                decomposed=n_fluid > 1,
                premise=(
                    f"{n_fluid} fluid windows: the family's region IS cut, so "
                    "R10's premise holds and the refusal can fire."
                    if n_fluid > 1 else
                    "ONE fluid window: nothing has been decomposed, so R10's "
                    "premise does not hold and the rule correctly stays quiet "
                    "however the expert is declared. That is W114's premise "
                    "check working, not the substitution passing. Switch the "
                    "tiling to six windows to see the refusal."),
                seam_order=list(rows[candidates[0].key]["seams"]))
