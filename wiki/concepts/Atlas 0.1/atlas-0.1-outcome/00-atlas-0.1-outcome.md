# Atlas 0.1 — Outcome: what the framework and the proof-of-concept suite have established

**Type:** Hub — **outcome record and proposal evidence map** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** compiled 2026-09-26, after Tier 88 (W343). **Revised 2026-09-27 after Tier 89**, which measured three things this folder had listed as not done: the matched shrink (W214), the coarse competitor across coupled seams (W345), and decomposition's speed by rotor count (W346). Their records are [[matched-shrink-and-coarse-competitor]] and [[decomposition-speed-by-rotor-count]]. This folder is still a synthesis: every number is quoted from a page that records it, and each claim page names its source page and artifact. Where the record is silent, the page says so; it does not fill the silence.
**Purpose:** the evidence base for a funding proposal to build learned experts (neural operators) that are native to domain decomposition: experts that iterate to convergence cheaply at a seam, and that accept non-rectangular geometry.
**Related:** [[00-atlas-0.1-overview]] · [[f1-pathmap-and-end-goal]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[prior-art-and-novelty-atlas-0.1]] · [[atlas-implementation]]

---

## 1. The pitch, in five claims, and what the record says about each

The proposal wants to make five claims. They are held here against 89 tiers of measurement. The verdict column is deliberately blunt, because a reviewer will check.

| # | the claim as the proposal wants it | verdict on the record | page |
|---|---|---|---|
| **C1** | Classical domain decomposition solves engineering problems **faster** than a full-domain solve **when there are many rotors** (the owner's target, 2026-09-26), and more accurately (the rocket). | **Faster: demonstrated at Tier 89, for one mechanism.** From 5 to 21 rotors, decomposition run on several cores is $3.4$–$5.7\times$ faster than the undivided solver of the same discretization (two timing draws), and $2.1$–$3.7\times$ at 36 rotors, with farm power within $2.3$–$8.3\%$. Run serially it is never faster. **More accurate: not demonstrated.** Decomposition is never more accurate than a same-discretization monolith, and the rocket has no full-domain run, because no single solver covers its families. What the rocket demonstrates is a **correct, conserving, multi-solver coupled episode**. | [[outcome-c1-decomposition-vs-monolith]] |
| **C2** | Learned experts **couple and converge with** classical simulations inside domain decomposition. | **Partly demonstrated.** Frozen experts couple stably (70 macro-steps, 12–24 windows) and pass a teacher-forced momentum go/no-go. Inside **defect correction**, the iteration provably and measurably converges to the **classical** answer whatever the learned map is. But the frozen expert's own answer is biased: it inverts the array efficiency in the wind farm, and it captures $71\%$ of the verified design gain. | [[outcome-c2-learned-experts-in-the-loop]] |
| **C3** | In complex scenarios (many rotors and windows), learned experts run **faster** than classical ones. | **Demonstrated on one case, with conditions attached.** Per agent, the checkpoint goes from $4.90\times$ a classical window's cost at one window to $0.253\times$ at twenty-four. A composed checkpoint column ran $3.15\times$ and $3.64\times$ faster than the undivided classical solver. **Against it:** on single-family flow a $2\times$-coarse classical solver is cheaper still, by $9$–$17\times$. On the car, at the cadence the composition layer runs at, the learned column is $0.067\times$ the classical speed and leaves its envelope. **At coupled seams (Tier 89)** the coarse competitor is correct but dearer than the cold march, so the bar there is the cold march itself. | [[outcome-c3-learned-speed-at-scale]] |
| **C4** | The framework is **modular multiphysics**: declare geometry, equations and experts, set a few parameters, and it runs; experts swap between classical and learned anywhere. | **Largely demonstrated, classically.** A compiler builds graphs from declarations with **zero hand-written coupling code**. Five port types cover flow, heat, mechanics, rotation and electricity. A union of 18 agents across five families marches. A car dashboard flips each fluid window between classical, learned and certified. **Limits:** only one of five families has a working learned option, the vehicle's clocks span $400\times$, and "fast *and* accurate" has not been shown at once. | [[outcome-c4-modular-multiphysics]] |
| **C5** | An architecture and method to **fine-tune or create** learned experts that handle non-rectangular geometry and converge fast under domain decomposition. | **Not done.** No such expert has been designed, trained or fine-tuned. What *does* exist, and is the proposal's strongest asset, is a **measured specification** of what such an expert must satisfy: the geometry ceiling, the time-step ceiling, the boundary input, the Jacobian that sets the convergence rate, and the cost bar. | [[outcome-c5-requirements-for-dd-native-experts]] |

**Supporting pages:** [[outcome-evidence-ledger]] has every quotable number with its source and caveat. [[outcome-demos-and-artifacts]] lists every interactive page, runnable demo and figure. **The evidence board**, one interactive page with the verdicts and the key charts, is `interactive/atlas-outcome-board.html`, published privately at [claude.ai/artifact/VLuXJcetzkYuwFA2eChmw3](https://claude.ai/artifact/VLuXJcetzkYuwFA2eChmw3).

---

## 2. The one-paragraph honest version

**What Atlas 0.1 has built** is a coupling framework. It composes independently written physics solvers, classical or learned, through five typed ports whose product is power. It decides from declarations alone whether a composition is admissible, and it marches, audits and differentiates the result. It has been exercised on wind farms, a front wing, a closed coolant loop, a powertrain, an 18-agent vehicle union, a body-fitted car and a seven-agent rocket ascent. Its conservation bookkeeping closes to round-off in every audit that measured it (per-block content, loop and energy balances), and the coupling residuals that do not close are measured and traced to named causes. Run on several cores, its decomposition of a wind farm beats the undivided solver by $3.4$–$5.7\times$ from 5 to 21 rotors, at a cost of a few percent of farm power. **What it has not built** is a learned expert that *earns its place*. The one frozen neural operator the programme could use (Poseidon-T) couples stably. It is faster per agent than a classical window from twelve windows up, and inside defect correction it cannot corrupt the answer. But it is biased, it accepts only a uniform $128\times128$ grid at one native time step, and in the one slot where its contribution was certified, its learned content was worth $13$ classical calls against a $69$-call noise floor. **That gap is the proposal.** The framework is ready to host a better expert. The record says exactly what "better" means, as numbers ([[outcome-c5-requirements-for-dd-native-experts]]).

---

## 3. How the claims chain together

$$
\underbrace{\text{C4: a framework that composes anything declared}}_{\text{built}}
\;\Longrightarrow\;
\underbrace{\text{C2: learned experts can sit in it without corrupting the answer}}_{\text{built, in the defect-correction slot}}
\;\Longrightarrow\;
\underbrace{\text{C3: and at scale they are cheaper per agent}}_{\text{measured on one family}}
\;\Longrightarrow\;
\underbrace{\text{C5: so an expert built FOR the slot is worth funding}}_{\text{not built — the ask}}
$$

C1 sits beside this chain. It is now two measured claims:
- **Speed at scale**, from parallelism over cache-sized windows once the undivided farm no longer fits in cache ([[decomposition-speed-by-rotor-count]]).
- **Composability**: coupling solvers of different families that no single monolith covers.

**[AI Inference]:** parallelism is the textbook reason domain decomposition exists. The proposal should present the speed claim as this framework reproducing a known effect, not as a new one. See [[outcome-c1-decomposition-vs-monolith]] §4.

---

## 4. What is NOT done, named

The full list, with what each item would take, is in each claim page's last section. In short:

1. **C1 — speed at scale is measured on one family, one laptop and Python threads** (W346). No comparison exists on the rocket or any multiphysics case. None exists at equal cost either, and that is the only kind that could show decomposition winning on accuracy.
2. **C2 — no learned expert has been admitted with a meaningful contribution** in any certified slot. W214 ran at Tier 89. It found that stability along the approach, not the rate at the settled state, limits the shrink ([[matched-shrink-and-coarse-competitor]] §1). A residual-scheduled shrink (W347) is untested.
3. **C3 — the speed advantage has one governing family and one host class behind it.** At the coupled seams measured at Tier 89, the coarse classical competitor is correct and not cheap, so the bar there is the cold march ([[matched-shrink-and-coarse-competitor]] §4).
4. **C4 — no learned expert in any family but 2-D incompressible flow.** No march spans the vehicle's own clocks. No 3-D beyond a classical half-car box, and the union march is not differentiable through the devices (W209).
5. **C5 — no architecture, no training data, no fine-tuning run, and no new expert.** F9 (light adaptation) was priced at $0.35$ h per $2000$ iterations and deliberately not run.

---

## 5. Pages in this folder

| page | what it holds |
|---|---|
| [[outcome-c1-decomposition-vs-monolith]] | C1: classical decomposition against the undivided solve, and the rocket episode's correctness record |
| [[outcome-c2-learned-experts-in-the-loop]] | C2: frozen experts coupled to classical ones; defect correction's certified convergence; the bias |
| [[outcome-c3-learned-speed-at-scale]] | C3: the per-agent cost crossover, the composed-column timings, and the competitors that beat it |
| [[outcome-c4-modular-multiphysics]] | C4: the compiler, the port algebra, and every multiphysics graph built |
| [[outcome-c5-requirements-for-dd-native-experts]] | C5: the measured specification a DD-native learned expert must meet |
| [[outcome-evidence-ledger]] | every quotable number, with source, conditions and caveat |
| [[outcome-demos-and-artifacts]] | the interactive pages, runnable demos, figures and how to open each |
| [[outcome-c4-path-to-declarative-cases]] | the plan for case studies without a Python module, and a decomposition GUI compared with the full-domain solve |
| [[showcase-library-plan]] | the classical showcase library: coupling styles, eight fast cases, the build order, the geometry section, and the survey of existing editors |
| *Tier 89 records* | [[decomposition-speed-by-rotor-count]] (C1, W346) · [[matched-shrink-and-coarse-competitor]] (C2, C3, C5; W214, W345) |
| `interactive/` | local copies of the self-contained HTML viewers (rocket episode, scaling ladder, wake array) and the outcome evidence board |

---

## See Also

- [[00-atlas-0.1-overview]] — the architecture this folder reports the outcome of
- [[f1-pathmap-and-end-goal]] — the end goal, Claims A and B, and the rung status these verdicts sit against
- [[case-study-ladder-to-f1]] — the tier-by-tier record
- [[gap-worklist]] — every open W-number named here
- [[prior-art-and-novelty-atlas-0.1]] — what is and is not new, which a proposal must respect
- [[poc2-novelty-audit]] — the adversarial reading the proposal will face
