"""The plain-language layer: what this is, why it is new, and what it is not.

Kept as **data in a Python module rather than prose in the HTML** for one
reason: it is the part of this demo most likely to drift into an overclaim, and
a claim that lives in a file can be reviewed, diffed and tested.
`tests/test_tier31_frontwing_demo.py` asserts that every number quoted here
matches the artifact it came from, and that no sentence here contains a word from
`FORBIDDEN` -- the vocabulary this project has decided it has not earned.

Two audiences, and the page shows both at once rather than choosing:

  * **`plain`** -- an educated reader who is not a specialist. No term is used
    before it is unpacked, every claim is concrete, and the analogies are
    load-bearing rather than decorative.
  * **`technical`** -- someone who knows numerical analysis or machine learning
    and wants the actual mechanism, the citation, and the measurement.

The rule the plain column follows: **say the smaller true thing.**  There is a
version of this project that sounds much more impressive and it is not true, so
every panel here is written to survive somebody checking it.
"""
from __future__ import annotations

#: Words this project has decided it has not earned in a user-facing claim.
#: Asserted absent from every `plain` and `technical` string below.
FORBIDDEN = (
    "revolutionary", "breakthrough", "unprecedented", "solves the problem",
    "certified accurate", "replaces cfd", "guaranteed",
)


# ---------------------------------------------------------------------------
# the top of the page: what this is, in four beats
# ---------------------------------------------------------------------------

HEADLINE = dict(
    what="A front wing on a real car has two things happening at once: it rides "
         "up and down on the suspension, and it bends under its own load. Each "
         "changes the other, and both change the downforce. This is both of "
         "them, in one simulation, differentiated end to end.",
    novel="The new part is not the physics. It is that the software reads a "
          "description of every component and every joint and then tells you, "
          "joint by joint, whether it is willing to certify the result -- and "
          "when it is not, exactly which rule it failed and what nobody has "
          "measured.",
    why="A fast simulator that is confidently wrong is worse than a slow one "
        "that is right, because an optimiser will find and exploit precisely "
        "the places where it is confidently wrong. Everything on this screen is "
        "about making the software say so out loud.",
)


# ---------------------------------------------------------------------------
# why it is novel -- the argument, in the order it has to be made
# ---------------------------------------------------------------------------

WHY_NOVEL = (
    dict(
        n=1,
        title="The problem is not that we cannot simulate a car",
        plain="Coupled aerodynamics, structures and heat for a race car is "
              "routine industrial practice. It is just slow -- hours to days "
              "for one design. So engineering teams evaluate tens or hundreds "
              "of designs, chosen by a human, rather than millions chosen by a "
              "search. That is the actual constraint: not what can be "
              "simulated, but how many designs can be looked at.",
        technical="preCICE, ANSYS System Coupling and Modelica already do "
                  "partitioned multiphysics with interface conservation and "
                  "multi-rate stepping. The gap is the cost and character of "
                  "one coupled evaluation: hours, and non-differentiable, "
                  "because per-solver adjoints exist but the coupling breaks "
                  "the chain.",
    ),
    dict(
        n=2,
        title="The usual fix has a hole in it",
        plain="The obvious answer is to replace the slow simulator with an AI "
              "model trained on a lot of physics. Two things go wrong. First, a "
              "car is not only air and metal -- it is also a motor's torque "
              "curve, a tyre, a gearbox, a battery. A model that predicts "
              "fields on a grid has no way to represent a torque curve, and no "
              "amount of extra training gives it one, because the thing is not "
              "a field. Second, when you glue several trained models together "
              "at their boundaries, nobody can tell you whether the result is "
              "still physics.",
        technical="Poseidon, Walrus, MPP and GPhyT all scale the same way -- "
                  "pretrain one model on more physics. None of them has a "
                  "representation for a lumped subsystem. And classical "
                  "co-simulation's stability theory assumes each subsolver has "
                  "a consistency order, a controllable step, and a stateable "
                  "Lipschitz bound. A frozen neural surrogate has none of "
                  "those, and the universal remedy for coupling instability -- "
                  "shrink the communication step -- is unavailable, because it "
                  "was trained at a native dt and shrinking takes it out of "
                  "distribution.",
    ),
    dict(
        n=3,
        title="A shared connector, so the joints do not multiply",
        plain="Every joint here is described the same way: as a pair of "
              "quantities whose product is power. A push and a speed. A "
              "pressure and a flow rate. A temperature and a heat flow. Any two "
              "components that expose the same kind of connector can be joined "
              "without anyone writing glue code for that specific pair. This "
              "matters for one unglamorous reason: with per-pair glue, twenty "
              "components need up to two hundred and ten hand-built adapters, "
              "and the project collapses under its own plumbing long before the "
              "physics fails.",
        technical="Bond graphs (Paynter, 1959) and port-Hamiltonian systems. "
                  "Five closed port types -- MECH, ROT, THERM, ELEC, ADVEC -- "
                  "each an effort-flow pair. O(K) port declarations replace "
                  "O(K^2) pairwise contracts, and composition of "
                  "port-Hamiltonian systems is port-Hamiltonian, so passivity "
                  "composes. Not novel: this is exactly why it was adopted.",
    ),
    dict(
        n=4,
        title="One question you can ask at every joint",
        plain="Because every joint carries power, there is a single check that "
              "means the same thing everywhere: is energy appearing or "
              "disappearing at the seams? One number, the same definition "
              "whether the assembly has two parts or twenty. You can watch it "
              "on this screen, and you can break it deliberately and watch it "
              "move.",
        technical="The global power residual R(t). On this graph, over 480 "
                  "macro-steps: a static accounting reads 1.0000 (the "
                  "normalising level), the motion terms take it to 5.93e-2, the "
                  "half-step correction to 3.30e-2, and once settled it is "
                  "3.32e-8. Identical in definition at CS-2's two-agent wind "
                  "farm and here.",
    ),
    dict(
        n=5,
        title="And the part that is actually new: it can say no",
        plain="Before anything runs, the software reads a written description "
              "of each component -- how far one step spreads information, "
              "whether it solves a whole-domain equation inside itself, whether "
              "its time-stepping is of a kind the theory covers -- and decides "
              "whether it is willing to vouch for the assembly. There are three "
              "answers: certified; runs but not certified, and here is which "
              "constant nobody has measured; or refused, and here is the rule "
              "and the reason. Every joint gets its own answer. That is the "
              "thing no other framework in this space does: the existing tools "
              "couple, and they do not judge.",
        technical="Nine layers, a seven-hypothesis envelope, three verdicts "
                  "(admit / admit-uncertified / refuse), each attached to a "
                  "named subject -- an agent, a port or a seam. The verdict is "
                  "a property of the declaration, not of the design point: "
                  "measured, zero of sixteen design-box corners change the "
                  "per-seam map. What moves with the design is the quantity "
                  "under each light.",
    ),
)


# ---------------------------------------------------------------------------
# the beats, each with what a viewer should take from it
# ---------------------------------------------------------------------------

BEATS = (
    dict(
        key="substitution",
        n=1,
        title="Offer it a real AI model and watch it refuse",
        plain="Poseidon-T is a frozen twenty-million-parameter neural operator "
              "trained on a great deal of fluid dynamics -- exactly the sort of "
              "model this whole approach is meant to run on. Hand the compiler "
              "an honest description of it and seven of the nine joints turn "
              "amber with three new complaints. Describe it as what its "
              "architecture actually makes it -- a model whose output anywhere "
              "depends on the input everywhere -- and every joint turns red. "
              "The reason is simple enough to say in one line: you cannot cut a "
              "problem into pieces if each piece secretly depends on all the "
              "others.",
        technical="W93. probe.support_reach measured Poseidon-T's domain of "
                  "dependence at a real wake seam as the whole 128-cell window "
                  "where its record declares 2 -- a factor of 32, at every "
                  "amplitude from 1 to 1e-3. required_halo() returns None, "
                  "L2/R10/halo decertifies, and L2/R10 refuses outright once "
                  "elliptic_subsolve is declared at the value the architecture "
                  "implies. A neural operator's receptive field is global by "
                  "construction: a fact about the architecture, not the physics.",
        matters="This is a negative result about the whole "
                "compose-pretrained-operators programme, and it was found by a "
                "machine built to look rather than by someone reading a paper. "
                "It is also why the programme's schedule changed: climb the "
                "ladder with classical solvers and price each learned "
                "substitution one seam at a time.",
        watch="Switch between the three descriptions. Nothing about the model "
              "changes -- only what is declared about it. The colours are "
              "reading the declaration.",
    ),
    dict(
        key="envelope",
        n=2,
        title="Turn off one check and watch the optimiser lie",
        plain="Both components say, in writing, what they are valid for: the "
              "wing is a small-deflection model and stops being one past a "
              "stated bend; the suspension stops being one below a stated ride "
              "height. Until this was found, the check ran only on the plain "
              "march and not on the path the optimiser takes -- so the search "
              "walked the car through its own floor and reported a 27.5% gain "
              "for a design the model does not stand behind. With the check "
              "live it is 19.3%, the optimiser is refused eight times out of "
              "thirty, and the screen names which component objected and at "
              "which step.",
        technical="W145. run() enforced both experts' declared validity "
                  "envelopes and objective() -- the path every search marches "
                  "-- enforced neither. The same asymmetry was in wing_fsi, "
                  "ground_effect and this demo: four paths, one shared "
                  "check_envelopes reading detached values. Found by extending "
                  "an ablation to the optimum purely to remove an unsupported "
                  "inference; it crashed at macro-step 4. Worth a fictitious "
                  "6.9 percentage points.",
        matters="With 15-20 components, the chance that at least one is outside "
                "its training regime on a novel design approaches one -- and a "
                "novel design is the entire point. Knowing when to abstain is "
                "the load-bearing safety property of the whole idea, not a "
                "nicety. This is a small working instance of it, on the path "
                "where it matters.",
        watch="Drag the free ride height and the spring rate both to the bottom "
              "of their range. That design settles below the suspension's "
              "declared floor, and with the check on the march stops and the "
              "screen names the expert and the macro-step. Now untick 'enforce "
              "declared envelopes' and press run: the same design marches, "
              "reports a downforce, and every number it produces is stamped "
              "OUTSIDE THE MODEL. Note that the live optimiser will NOT take "
              "you there on its own -- it is a warm-started twelve-step "
              "estimator and it climbs in a different direction from the "
              "hundred-and-twenty-step one the recorded columns used, which is "
              "itself worth knowing and is why the recorded numbers are the "
              "ones beside it.",
    ),
    dict(
        key="ablation",
        n=3,
        title="Freeze one joint and see what it was worth",
        plain="The claim of the whole approach is that the connections between "
              "parts are the physics, not a correction to it. That is cheap to "
              "check: make the spring infinitely stiff and the wing is bolted "
              "in place; make the wing infinitely stiff and it cannot bend. "
              "March each and read the downforce. At the starting design the "
              "suspension joint is worth half a percent. At the optimised "
              "design it is worth eight -- because the optimum rides much "
              "closer to the ground, where a small change in height is a large "
              "change in force.",
        technical="Each frozen limit is a parent case study: k -> infinity is "
                  "CS-12, E* -> infinity is CS-10. Statically, those limits "
                  "reproduce their parents to 3.37e-10 and 6.36e-8, which is "
                  "what earns the word assembly. The frozen suspension column "
                  "must be pinned at the height the LIVE design settles to: "
                  "using the reference-load release height agrees to 4e-4 at "
                  "the reference design and differs by 37% at the optimum.",
        matters="It also falsified a claim this project had written down. The "
                "prediction was that a search which stiffens the wing shrinks "
                "what the ELASTIC joint is worth. Measured, it went the other "
                "way and on the other joint: 0.49% to 8.16%, a factor of "
                "seventeen, while the elastic column could not be measured at "
                "all because a rigid wing there makes more downforce and drives "
                "itself through the floor.",
        watch="Run it at the starting design, then optimise and run it again. "
              "The seam that barely mattered is now the one doing the work.",
    ),
    dict(
        key="race",
        n=4,
        title="Race the gradient against the population, on one clock",
        plain="Because the whole coupled simulation is differentiable, one "
              "backward pass returns how much the downforce changes for each of "
              "the four design knobs at once. The alternative -- what you do "
              "without gradients -- is to try many designs and keep the good "
              "ones. Both are running here on the same problem and the same "
              "clock. The honest result is smaller than this project used to "
              "quote: the gradient does not find a better design, it finds an "
              "equally good one sooner, and 'sooner' is about two to four "
              "times, not twenty.",
        technical="W143. Two errors compounded: the denominator was CMA-ES's "
                  "whole 200-evaluation budget rather than the 191 at which it "
                  "peaked, and nobody divided by the adjoint premium -- 4.9 "
                  "forward evaluations per gradient iterate on this graph. "
                  "Corrected: 1.8x-4.3x in wall-clock against 18x-22x in "
                  "rollouts, with the two columns landing within 1.003x-1.008x "
                  "of each other in the objective. Opened against PoC 1 and PoC "
                  "1a as well, not only fixed here.",
        matters="Differentiability, not speed, is the property the whole vision "
                "rests on. So the number that matters is whether having a "
                "gradient changes the search -- and it is worth quoting "
                "correctly, because the flattering version of this number was "
                "in three of this project's own documents.",
        watch="The live race is short so it finishes while you watch, and a "
              "short race can go either way -- CMA-ES's first random sample "
              "sometimes beats three Adam steps. The recorded full-scale run is "
              "beside it and labelled.",
    ),
    dict(
        key="balance",
        n=5,
        title="One needle for energy at the seams",
        plain="If a coupling is quietly creating or destroying energy, "
              "everything downstream of it is wrong in a way no plot will show "
              "you. This needle is that check, computed from the joint "
              "quantities alone. Watch it drop as the simulation settles, and "
              "switch the accounting to see it peg when the motion terms are "
              "left out.",
        technical="R(t), the global power residual. Two receivers here, so the "
                  "energy is the plate's strain energy plus the spring's and "
                  "the power is the interface power both seams share. Both "
                  "energies at release are needed: the plate starts flat so its "
                  "strain energy is zero, and forgetting the spring's -- which "
                  "is the largest single number in the balance -- put the first "
                  "step at 1.81 against a static accounting's 1.00.",
        matters="This is the one diagnostic defined identically at every rung "
                "of the roadmap, so a regression on a twenty-agent graph is "
                "comparable with a measurement on a two-agent one. No per-rung "
                "metric engineering.",
        watch="Press reset. Once a march has settled all three accountings "
              "agree, because there is nothing left moving for the motion terms "
              "to account for -- the gap between them only exists during the "
              "transient, which is what the reset replays. The live level is a "
              "running maximum and is still growing over the first frames, so "
              "those are not a reading.",
    ),
)


# ---------------------------------------------------------------------------
# what this does NOT claim -- shown, not buried
# ---------------------------------------------------------------------------

NOT_CLAIMED = (
    dict(
        title="Nothing here is certified",
        plain="Not one joint on this screen is green, and none ever has been in "
              "this project. Three constants the bounds rest on have never been "
              "measured, and while that is true every graph comes back 'runs, "
              "not certified'. The machine is honest about that rather than "
              "rounding it up.",
        technical="L, sigma and C_mu are unmeasured (W1, W3, W49), and a "
                  "non-empty unmeasured list has forced admit-uncertified on "
                  "every graph in this package since W56.",
    ),
    dict(
        title="This is not faster than the solver it replaces",
        plain="Every component here is a classical solver, not a trained model. "
              "The four-to-six orders of magnitude that make the whole idea "
              "worth having only arrive with learned components -- and beat 1 "
              "is the framework refusing the most obvious candidate. That "
              "tension is the honest state of the programme, not a detail.",
        technical="Climb classically, substitute one certified seam at a time "
                  "(case-study-ladder-to-f1 section 2). The composed column is "
                  "measured against itself here; no speed claim is made.",
    ),
    dict(
        title="The physics is a demonstration, not an engineering model",
        plain="One Reynolds number, one grid, one angle of attack. The wing is "
              "a porous inclined plate with no boundary layer of its own and no "
              "circulation condition. The floor is a rolling road, so there is "
              "no viscous choking of the gap. No stress number here is a "
              "statement about an alloy.",
        technical="Re_c = 125, 208x144 cells, 20 degrees. The plate is a 32x2 "
                  "Q1 plane-stress cantilever and Q1 elements lock in bending; "
                  "E* is a design knob in flow units calibrated to a deflection "
                  "rather than taken from a material. No grid-convergence study.",
    ),
    dict(
        title="The big question is still unanswered",
        plain="Does the error grow faster than the number of joints? Nobody has "
              "measured it, here or anywhere. If it does, a library of "
              "components is a dead end no matter how good the components are. "
              "This assembly has nine joints; the target has fifteen to twenty "
              "subsystems.",
        technical="Claim B, and rung 9 of the ladder. Composition error versus "
                  "interface count has never been measured at any N. Also "
                  "untouched: rung 4, whether a learned expert trained for one "
                  "scenario transfers to another.",
    ),
)


def payload() -> dict:
    """Everything the page renders, in one object."""
    return dict(headline=dict(HEADLINE),
                why_novel=[dict(x) for x in WHY_NOVEL],
                beats=[dict(x) for x in BEATS],
                not_claimed=[dict(x) for x in NOT_CLAIMED])


def _all_prose() -> list[str]:
    """Every user-facing sentence, for the FORBIDDEN check in the tests."""
    out: list[str] = list(HEADLINE.values())
    for group in (WHY_NOVEL, BEATS, NOT_CLAIMED):
        for row in group:
            out += [v for k, v in row.items() if isinstance(v, str)
                    and k != "key"]
    return out
