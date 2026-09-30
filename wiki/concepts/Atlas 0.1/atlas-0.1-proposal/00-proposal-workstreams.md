# The proposal's four workstreams — evaluation, order, and the decisions left to the owner

**Type:** Hub — **plan for finishing the proposal** (folder: `Atlas 0.1/atlas-0.1-proposal/`)
**Status:** written 2026-09-30 in a planning session, on the owner's brief: *"iron out everything — decide specifics, outline architecture and documentation/plan in the wiki, so that future chats can figure everything out on top of it."* **No code was changed, nothing was run, and no number here is new.** The session ran in a cloud container without numpy, so every reading of the workbench is from its source. Its network blocked arXiv and the three reference websites, so every literature entry is from search summaries and is marked for a full read.
**Parent:** [[00-atlas-0.1-outcome]] (the evidence: C1–C5) · **Tracker rows:** [[gap-worklist]] W349–W354

---

## 1. The four workstreams, and a verdict on each

| # | workstream | its purpose (the owner) | verdict | plan |
|---|---|---|---|---|
| **1** | **finish the demo workbench** | *"a way for people to actually interact and become convinced"* | **Items 1.1–1.3 are ordinary engineering.** **Item 1.4 runs into the record**: at showcase sizes the undivided solve beats every decomposed arm except the wind farm's, so for most families a speed mechanism must be built, and **two families cannot reach "much faster" by construction**. **Item 1.5 is a research result, not a feature**: plausible with an expert trained for the slot, with a gate that may fail | [[demo-finish-plan]] · [[demo-fast-examples-plan]] · [[demo-learned-case-plan]] |
| **2** | **the proposal website** | *"an advertisement, a mini-pitch … flashy, unique, cool, and convincing to the non-technical person"* | **Feasible, and the easiest to do well, but it must come last**: its best content (the fast examples, the learned case, the proofs, the images) does not exist yet. Its main risk is overclaiming; the plan answers it with a claims pipeline in which no number is typed by hand | [[proposal-website-plan]] · [[website-evidence-and-citations]] · [[vision-scenarios-and-image-prompts]] |
| **3** | **the learned-expert architecture** | *"the actual ask … fairly technical/mathy"* | **The owner's idea holds up, and is largely adopted**: diffeomorphic charts with the generalised Jacobian passed in, harmonic and transfinite maps, optimized Schwarz, a boundary-to-domain lifting. **But its core is published** (DIMON and DNO for the maps; SNI, NEST and L-DDM for neural Schwarz on any geometry), **so the novelty must move**: to a multiphysics port contract, **a convergence certificate that holds however the expert was trained**, and training for the iteration count | [[chart-operator-architecture]] · [[chart-operator-training-and-cost]] · [[dd-neural-prior-art-2026]] |
| **4** | **formal proofs, in Lean where possible** | *"to show that beyond the 'AI-magic' framework, this is … mathematically secure"* | **The core the proposal needs is finite-dimensional and can be machine-checked in weeks**: Banach with perturbation, the wave-variable Cayley bound, the master error bound, defect correction's consistency, fixed-point consistency (which settles **W348**). The continuous PDE theory, chart injectivity and universal approximation are cited, not formalised. **No formalised domain decomposition theory exists anywhere that was found**, so even the classical parts are a first | [[formal-proofs-plan]] |

---

## 2. The idea that ties them together

**[AI Inference]**, and the thesis the proposal should lead with:

> **Charts for geometry. Waves for agreement. Operators for physics.** Cut any domain into four-sided pieces, chart each onto a square with a map that provably does not fold, let a learned operator solve on the square given the chart's metric, and let the pieces exchange impedance-weighted waves through typed, power-conserving ports. Then **convergence is a checkable property of each expert**, its Lipschitz constant, and **accuracy is a separate, bounded quantity**.

- It answers the vault's hardest lesson: frozen experts failed because nothing in the coupling could be guaranteed about them ([[outcome-c5-requirements-for-dd-native-experts]]).
- It is what the demo's learned case previews ([[demo-learned-case-plan]]).
- It is what the headline theorem proves ([[formal-proofs-plan]] §3).
- It is what the website sells ([[proposal-website-plan]] §1).

---

## 3. Dependencies

```
   demo 1.1-1.3 (UI, fork, holes) ------------------------------+
                                                                 |
   proofs: T1, T4, T7, then T3 --> W348 settled --> demo 1.4 ----+
                                                                 |
   architecture document (reads prior art in full) --> demo 1.5 -+
                                                                 |
   proofs: T8, T9, T22 (the headline) --> blueprint public ------+
                                                                 v
   owner generates the images (from the prompts, any time) --> website
```

- **W348 before 1.4**: most fast examples are substructured or iterated one-physics problems, and R10 refuses that class today, so each would show a red `refuse` beside a correct, faster answer ([[demo-finish-plan]] §5). Theorem T3 is the proof W348 needs.
- **The architecture before 1.5**: the learned case is a rectangular-chart instance of the chart operator, and its choices (inputs, loss, gate) come from there.
- **Everything before the website**, except the images, which the owner can make now.

---

## 4. The recommended order of chats

The owner planned one chat per workstream. **Seven are recommended**, because the demo and the proofs each contain parts of very different kinds:

| chat | work | needs from the owner | plan |
|---|---|---|---|
| **A** | demo 1.1–1.3: the header, the fork, the holes; reproduce first | O1 | [[demo-finish-plan]] |
| **B** | proofs, steps 1–3 (T1, T7, T4, T3, T5, T10, T11), then the W348 compiler change | O5 (the Lean download) | [[formal-proofs-plan]] §5; [[gap-worklist]] W348 |
| **C** | demo 1.4: the micro-benchmarks, then the fast example per type; on AC power | O3 | [[demo-fast-examples-plan]] |
| **D** | the architecture document: read the prior art in full, check the conformal-chart claim, write the proposal | O8 | [[chart-operator-architecture]] §9 |
| **E** | demo 1.5: price the network, register the gate, generate, train, evaluate | O6 | [[demo-learned-case-plan]] |
| **F** | proofs, steps 4–5: the headline theorem, the blueprint, public | O9 | [[formal-proofs-plan]] §5 |
| **G** | the website | O4, O7, the images | [[proposal-website-plan]] §6 |

**A, B and D are independent** and can run in any order. C waits for B's W348 change, E for D, and G for everything.

---

## 5. Decisions left to the owner

| # | decision | the recommendation | where |
|---|---|---|---|
| **O1** | once the examples list leaves the UI, are the other examples still reachable? | a small *More examples* list under *Fast example*, one line each saying what it shows, with no names | [[demo-finish-plan]] §1.3 |
| **O2** | how strangers reach the demo | recorded replays on the website plus a one-command local install; a hosted instance only if labelled as not for speed | [[demo-finish-plan]] §4 |
| **O3** | what *Fast example* shows for a family that cannot reach the bar | its fastest honest configuration, with the limit named on the card | [[demo-fast-examples-plan]] §3 |
| **O4** | the website's hosting, repository and domain | a static host; the site in its own public repository once public | [[proposal-website-plan]] §3 |
| **O5** | installing Lean 4, Mathlib's cache (several GB) and `leanblueprint` | yes, for chat B | [[formal-proofs-plan]] §4 |
| **O6** | for the learned case: torch (probably already installed), a long data-generation run, a GPU rental under the \$0.70/h cap | yes, each after Step 0's price | [[demo-learned-case-plan]] §8 |
| **O7** | the website's headline | *Physics, assembled.* or *Every piece an expert. Every seam a guarantee.* | [[proposal-website-plan]] §5 |
| **O8** | the architecture document's form | a paper-style PDF and an HTML version of it, both linked from the site | [[chart-operator-architecture]] §9 |
| **O9** | are the proofs and the site public? | yes when linked; until then, private | [[formal-proofs-plan]] §4 |

---

## 6. What was not done in this session, named

1. **Nothing was reproduced.** The fork's and the holes' failure causes are hypotheses from the code ([[demo-finish-plan]] §2.2, §3.2).
2. **No timing was taken**, so every ceiling in [[demo-fast-examples-plan]] and every cost in [[chart-operator-training-and-cost]] is arithmetic, labelled so.
3. **No paper was read in full**, and no reference site was opened. Every literature row carries its status ([[website-evidence-and-citations]]; [[dd-neural-prior-art-2026]]).
4. **No Lean was written**, and the Mathlib names in [[formal-proofs-plan]] are to be confirmed.
5. **The image prompts are a first draft**, untested on any generator.

---

## 7. Pages in this folder

| page | what it holds |
|---|---|
| `demo/` [[demo-finish-plan]] | items 1.1–1.3: the header's type selector, the branching river, holes that cross the edge; hosting; why W348 comes first |
| `demo/` [[demo-fast-examples-plan]] | item 1.4: the four mechanisms that make a decomposition genuinely faster, each family's ceiling, the registration protocol |
| `demo/` [[demo-learned-case-plan]] | item 1.5: the wind farm, the arms including the coarse competitor and a truth run, the expert, the gate |
| `website/` [[proposal-website-plan]] | the story, the sections, the design language, the build, the claims pipeline, the honesty rules |
| `website/` [[website-evidence-and-citations]] | every number the site may show, with its source and status |
| `website/` [[vision-scenarios-and-image-prompts]] | the three scenarios, their agent graphs, the style bible and the prompts |
| `expert-architecture/` [[chart-operator-architecture]] | the chart operator: charts, pull-back, operator, wave coupling, the contraction contract, the Expert Card |
| `expert-architecture/` [[chart-operator-training-and-cost]] | sources, stages, the loss, fine-tuning against training from scratch, the cost model, the gate |
| `expert-architecture/` [[dd-neural-prior-art-2026]] | what is published (2021–2026) and what is left to claim |
| `formal-proofs/` [[formal-proofs-plan]] | the theorem inventory T1–T23, Lean tiers, the headline theorem, the project and its order |

---

## See Also

- [[00-atlas-0.1-outcome]] — the five claims this folder turns into a proposal
- [[outcome-c5-requirements-for-dd-native-experts]] — the measured specification the architecture answers
- [[showcase-gallery]] — the demo's measured record
- [[gap-worklist]] — W348, and this folder's rows W349–W354
