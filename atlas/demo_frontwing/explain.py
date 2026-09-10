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
#
# **The order is the argument.**  It was re-cut on 2026-09-08 and the reason is
# worth writing down beside it, because the previous order was defensible and
# still wrong for the audience.
#
# It used to open on the substitution refusal -- the compiler handed a real
# pretrained neural operator and declining it at every seam.  That is the
# project's strongest *claim* and it is what a specialist should look at first.
# It is also a **negative result**, and opening on it means the first thing a
# reader who has not read the vault sees is the machine saying no.
#
# The positive result was already here and already measured; it was third and
# fourth.  So the order now runs: the search wins on the clock (1), the win is
# inside the model rather than bought by walking through a constraint (2), here
# is what each joint was worth (3), and here is the check that tells you when
# not to believe any of it -- correctly declining a case it should decline (4),
# with the ledger last (5).
#
# **Nothing was re-measured to do this and no figure moved.**  The refusal is
# not softened, it is sequenced: it stops being the headline and becomes the
# reason to trust the headline.  `NOT_CLAIMED` still carries the tension whole.

BEATS = (
    dict(
        key="race",
        n=1,
        tab="a better wing, sooner",
        title="A better wing, and it gets there four times sooner",
        plain="This is what the whole framework is for. Four knobs -- how stiff "
              "the wing is, how thick it is, how stiff the spring is, how high "
              "it rides -- and one number to push up: downforce. The search "
              "finds a wing with 19.3% more of it than the one it started "
              "from, inside both the stress and the deflection limits. What "
              "lets it is that the whole coupled simulation, air and metal and "
              "suspension together, can be differentiated: one backward pass "
              "says how the downforce moves for all four knobs at once. The "
              "alternative -- what you do without a gradient -- is to try many "
              "designs and keep the good ones. Both are running here, on the "
              "same problem and the same clock, and the gradient reaches the "
              "population method's best answer 4.31 times sooner in wall-clock. "
              "It does not find a better design than the population does. It "
              "finds an equally good one sooner, and that is the honest shape "
              "of the result.",
        technical="W143. Adam on the adjoint against CMA-ES on the same "
                  "objective, one evaluation each in turn, with the live march "
                  "held for the duration -- a 12-frames-per-second march in the "
                  "background would be measured instead of the search. "
                  "Recorded: 4.31x in wall-clock, then 22.2x in rollouts, with "
                  "the two columns landing within 1.003x of each other in the "
                  "objective. The wall-clock figure is quoted first and the "
                  "rollout figure is labelled flattering, because the rollout "
                  "ratio prices a gradient iterate at one forward evaluation "
                  "and it is 4.9 of them on this graph. Both columns are warmed "
                  "before either clock starts: unwarmed, a three-step race "
                  "reads a 25x adjoint premium where a warmed one reads 5.1 -- "
                  "the same error stage_cost made when it published 9.47 "
                  "against a warmed re-measurement's 5.4 to 5.8.",
        matters="Differentiability, not raw speed, is the property the whole "
                "approach rests on, and this is the measurement of what having "
                "it buys. The figure is also smaller than this project used to "
                "quote: the flattering version was in three of its own "
                "documents before W143 corrected it, and the corrected one is "
                "what is on the screen.",
        watch="Press 'race them, on one clock'. The live race is short so it "
              "finishes while you watch, and a short race can go either way -- "
              "CMA-ES's first random sample sometimes beats three Adam steps, "
              "and below about nine gradient iterates Adam has not turned yet. "
              "The recorded full-scale run sits beside it and is labelled. The "
              "two live budgets are scaled from the recorded run's 30 against "
              "200 rather than picked, because picking them independently would "
              "mean picking them until the answer came out the way this panel "
              "would prefer.",
    ),
    dict(
        key="envelope",
        n=2,
        tab="the win is real",
        title="And the win is real, not a constraint quietly walked through",
        plain="The first thing worth asking about a number like 19.3% is "
              "whether the search cheated to get it. Both components say, in "
              "writing, what they are valid for: the wing is a "
              "small-deflection model and stops being one past a stated bend; "
              "the suspension stops being one below a stated ride height. "
              "Until this was found, that check ran on the plain march and not "
              "on the path the optimiser takes -- so the search walked the car "
              "through its own floor and reported a 27.5% gain for a design the "
              "model does not stand behind. With the check live it is 19.3%, "
              "the optimiser is refused eight times out of thirty on the way, "
              "and the screen names which component objected and at which step. "
              "The smaller number is the one that is inside the model. You can "
              "turn the check off on this screen and watch the larger one come "
              "back.",
        technical="W145. run() enforced both experts' declared validity "
                  "envelopes and objective() -- the path every search marches "
                  "-- enforced neither. The same asymmetry was in wing_fsi, "
                  "ground_effect and this demo: four paths, one shared "
                  "check_envelopes reading detached values. Found by extending "
                  "an ablation to the optimum purely to remove an unsupported "
                  "inference; it crashed at macro-step 4. Worth a fictitious "
                  "6.9 percentage points. declined and infeasible are separate "
                  "rows on purpose: 39 of the population's 200 evaluations were "
                  "declined by an expert and 66 were infeasible, and merging "
                  "them would overstate this by a factor of 1.7.",
        matters="With 15-20 components, the chance that at least one is outside "
                "its training regime on a novel design approaches one -- and a "
                "novel design is the entire point. Knowing when to abstain is "
                "the load-bearing safety property of the whole idea, not a "
                "nicety, and an optimiser is the worst adversary it will ever "
                "have, because a search finds and exploits precisely the places "
                "where a model is confidently wrong. This is a small working "
                "instance of it, on the path where it matters.",
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
        tab="where the gain came from",
        title="Freeze one joint and see what it was worth",
        plain="Where did that gain come from? The claim of the whole approach "
              "is that the connections between parts are the physics, not a "
              "correction to it -- and that is cheap to check. Make the spring "
              "infinitely stiff and the wing is bolted in place; make the wing "
              "infinitely stiff and it cannot bend. March each and read the "
              "downforce. At the starting design the suspension joint is worth "
              "half a percent. At the optimised design it is worth eight -- "
              "because the optimum rides much closer to the ground, where a "
              "small change in height is a large change in force.",
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
        key="substitution",
        n=4,
        tab="when not to trust it",
        title="How you know when not to trust the fast answer",
        plain="A search that wins on the clock is only worth having if you know "
              "when not to believe it, and that is what this panel is. The same "
              "graph is compiled three times with three written descriptions of "
              "the fluid expert, and the compiler answers at each of the nine "
              "joints. Poseidon-T is a frozen twenty-million-parameter neural "
              "operator trained on a great deal of fluid dynamics -- exactly "
              "the sort of model this approach is meant to run on, and the one "
              "that would make the search above orders of magnitude cheaper. "
              "Hand the compiler an honest description of it and seven of the "
              "nine joints gain three new complaints. Describe it as what its "
              "architecture makes it -- a model whose output anywhere depends "
              "on the input everywhere -- and every joint is refused. The "
              "reason fits in a line: you cannot cut a problem into pieces if "
              "each piece secretly depends on all the others. This is the same "
              "check that was watching the search, correctly declining a case "
              "it should decline.",
        technical="W93. probe.support_reach measured Poseidon-T's domain of "
                  "dependence at a real wake seam as the whole 128-cell window "
                  "where its record declares 2 -- a factor of 32, at every "
                  "amplitude from 1 to 1e-3. required_halo() returns None, "
                  "L2/R10/halo decertifies, and L2/R10 refuses outright once "
                  "elliptic_subsolve is declared at the value the architecture "
                  "implies. A neural operator's receptive field is global by "
                  "construction: a fact about the architecture, not the physics.",
        #: **The Tier 33 addition, folded into this beat rather than given its
        #: own.**  A sixth tab would split one trust-layer story in two; the
        #: point of the measurement is that it sits UNDER these three verdicts
        #: and says why they are what they are.  Every figure is read from
        #: `out/w153/w153.json` by `engine.load_horizon`, never transcribed.
        reach="There is something underneath those three verdicts that is newer "
              "than the verdicts. The rule behind them used to rest on a "
              "yes-or-no: does a component's influence reach past the cut, or "
              "not? Asked that way it answers yes for every component in this "
              "project, the plainest classical solver included -- which is the "
              "check being unavailable rather than a fact about any of them. It "
              "has since been replaced by a measurement of how far influence "
              "actually reaches, and the three candidates separate cleanly. The "
              "local classical solver's influence is exactly zero outside its "
              "declared reach -- not small, zero, over all 13,924 cells beyond "
              "it. The same solver carrying its own pressure solve spreads "
              "across the whole window. Poseidon-T spreads furthest. On a scale "
              "where 1 is 'everywhere', that is 0.186, 0.496 and 0.790. The "
              "verdicts do not move -- Poseidon-T is still refused -- but the "
              "reason is better than it was: the compiler is not "
              "pattern-matching 'neural network, therefore no'. It measured a "
              "graded property and put the three in the order the physics says "
              "they belong.",
        reach_technical="W153, Tier 33, read from out/w153/w153.json and not "
                        "re-run here. Pi in the master error bound is a 0/1 "
                        "indicator standing in for the sensitivity R_i(j), the "
                        "norm of d E_i(j) / d g_i: tight for a local explicit "
                        "agent and loose for influence that is dense but "
                        "decaying, which is the only kind a learned operator "
                        "has. At one exchange per macro-step it reads Pi = "
                        "1.0000 for every agent in this vault. Pi_w substitutes "
                        "the normalised sensitivity itself and ranks WindowNS "
                        "exposed 0.186, WindowNS embedded 0.496 and Poseidon-T "
                        "0.790 -- and Poseidon-T's bound is non-vacuous for the "
                        "first time, by 1.27x, which is small and is said so. "
                        "The instrument reduces exactly: the local agent's far "
                        "field beyond b = 20 cells, its declared radius times "
                        "sub-steps of 2 x 10, is bitwise zero over all 13,924 "
                        "far cells, and at halo 1 the ratio Pi_w/Pi is "
                        "1.000000, so no past verdict can move silently.",
        matters="Two things at once. It is a negative result about the whole "
                "compose-pretrained-operators programme, found by a machine "
                "built to look rather than by someone reading a paper -- and it "
                "is the reason to believe anything else on this screen, because "
                "a check that never says no is not a check. It is also why the "
                "programme's schedule changed: climb the ladder with classical "
                "solvers and price each learned substitution one seam at a "
                "time.",
        watch="Switch between the three descriptions. Nothing about the model "
              "changes -- only what is declared about it -- and the colours are "
              "reading the declaration. Then look at the measured-reach bars "
              "underneath: those are reading the models themselves rather than "
              "their descriptions, which is why both Poseidon rows sit on one "
              "bar. It is the same model twice, described two ways.",
    ),
    dict(
        key="balance",
        n=5,
        tab="energy at the seams",
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
        title="Certified at a fixed shape -- and not certified when it rides",
        plain="This wing is the first assembly this project has ever "
              "certified, and the qualification matters more than the "
              "headline. Held at a FIXED shape, all nine joints are green and "
              "the software raises no objection. Told that the wing MOVES -- "
              "that its own deflection changes where the joint is -- the same "
              "software refuses the two joints on the wing's surface and "
              "keeps the seven inside the air green. That refusal is not "
              "bookkeeping and it is not a bug: nothing here certifies a "
              "joint whose geometry is a function of the answer, and the "
              "riding wing is the case the framework still cannot certify. "
              "A racing wing rides. So the honest reading is that the "
              "machinery works on the problem it can state, and the problem "
              "it cannot state is the one the sport actually has.",
        technical="Every constant is measured on THIS graph -- W157 for L, "
                  "sigma and C_mu, W159 for cut_defect_bound against a "
                  "single-window monolith (5.622e-5, chi-weighted, tight to "
                  "0.05%). Five rules stood after that. Three fell in Tier 35 "
                  "as checker defects or missing measurements (W136 scoped the "
                  "halo rule to the agents the decomposition cuts; W138 made "
                  "the seam operator ORIENTED, taking both surface seams' "
                  "passivity defect to exactly zero; W159 supplied L2/C2's "
                  "value). The last two fell in Tier 39: W160 gave L2/R10's "
                  "undeclared branch the stencil_radius check its own sentence "
                  "presumed, so a two-line spring is no longer read as hiding "
                  "a pressure solve; W161 scoped _enforcing_agents to the "
                  "family the constraint comes from, so R12 no longer names a "
                  "plane-stress elasticity solver as enforcing "
                  "incompressibility. Neither moved a threshold or softened a "
                  "declaration, both have controls that put the objection "
                  "straight back, and both reproduce on graphs they were not "
                  "diagnosed on -- W160 on cooling_loop's four lumped legs and "
                  "powertrain's actuator disk, W161 on wing_fsi, which reaches "
                  "admit for the same repair. With motion declared, "
                  "L2/InterfaceMotion refuses `wet` and `mount` and nothing "
                  "else.",
    ),
    dict(
        title="This is not faster than the solver it replaces",
        plain="Every component here is a classical solver, not a trained model. "
              "The 4.31x on the first panel is one search method against "
              "another on the same solvers; it is not a claim about the solvers "
              "themselves. The four-to-six orders of magnitude that make the "
              "whole idea worth having only arrive with learned components -- "
              "and the fourth panel is the framework refusing the most obvious "
              "candidate. That tension is the honest state of the programme, "
              "not a detail, and moving that panel from first to fourth does "
              "not soften it.",
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
