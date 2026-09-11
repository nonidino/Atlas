# The End Goal and the Path to It: Autonomous Multiphysics Design

**Type:** Core Concept — Program Roadmap and Vision (folder: `Atlas 0.1/common/`)
**Status:** Living roadmap, written 2026-08-19. **Every case study, every expert, and every architectural decision should be checkable against §3's ladder.** If a piece of work does not advance a rung, it needs a reason. **2026-09-10: §3.3 added** — the rungs recounted against what is built; the schedule and its evidence are [[case-study-ladder-to-f1]] §14.
**Related Concepts:** [[port-algebra-atlas-0.1]], [[prior-art-and-novelty-atlas-0.1]], [[pfm-purpose-and-direction]], [[00-atlas-0.1-overview]], [[expert-library-atlas-0.1]], [[spec-wind-farm-wake-atlas-0.1]], [[incremental-transfer-roadmap]], [[physics-foundation-models]]

---

# 1. The end goal

> **A complete, coupled, differentiable physical model of a real machine — every regime that matters, composed from a library of specialized experts — fast enough and smooth enough that an optimizer, and eventually an autonomous agent, can search the design space rather than sample it.**

The concrete target is a **Formula 1 car**: external aerodynamics, structural response, tyre mechanics, powertrain and energy recovery, brake and cooling thermal management, and every place they couple — all in one graph, evaluated end-to-end in seconds, with gradients of any performance objective with respect to any design parameter.

Then, on top of that: **an agent layer that proposes, evaluates, and refines designs autonomously**, using the physics stack as its evaluator.

## 1.1 Why an F1 car and not something easier

It is the densest possible multiphysics coupling in a single object that a small team can reason about completely:

- **Every major regime in one artifact** — compressible and incompressible flow, boundary layers, structural dynamics, contact and friction, heat transfer in solids and fluids, electromagnetics, electrochemistry, control logic.
- **The couplings are the physics, not a detail.** Aero load changes ride height, ride height changes aero. Brake heat changes tyre temperature, tyre temperature changes grip, grip changes the load path. A model that gets each subsystem right and the couplings wrong is worthless — which is precisely the claim Atlas exists to test.
- **Field *and* lumped subsystems, inseparably.** Aerodynamics is a field problem; a motor torque map is not. This is the structural argument of [[prior-art-and-novelty-atlas-0.1]] §2.3 and the reason a monolithic PDE foundation model cannot reach this target at any scale of pretraining.
- **Design search is genuinely compute-bound**, and in F1 unusually literally so, since aerodynamic development operates under regulated limits on CFD and wind-tunnel usage. Every engineering organization is compute-rationed; this one is rationed by rule.

## 1.2 What the goal is *not*

**Not** replacing CFD or FEA. Coupled aero–thermal–structural simulation of a race car is existing industrial practice, and the classical stack will remain the verification authority for as long as anything is certified. **The deployment story is fixed and should never drift: search wide and cheap with the composed model; verify the shortlist with the classical stack.** Any framing that implies replacement is an overclaim.

What changes is **the number of designs that can be evaluated, and whether gradients exist** — hours to seconds, and derivative-free to differentiable. That is a 4–6 order-of-magnitude shift in the unit economics of search, and it is enough.

---

# 2. The two claims the whole program rests on

Everything below is an attempt to establish exactly two things:

**Claim A — Composition preserves physical validity.** Independently trained experts, connected through [[port-algebra-atlas-0.1]]'s ports, produce a coupled rollout that conserves what it should and stays plausible, without joint retraining.

**Claim B — Composition scales sub-linearly.** Adding the $N$th agent costs $O(1)$ integration work and adds sub-linear composition error, so a 20-expert machine is buildable by the same process that built the 2-expert one.

**If A fails**, the approach is a monolith with extra steps and pretraining breadth ([[walrus-paper]], [[poseidon-pde-foundation-model]]) is simply the better strategy. **If B fails**, the approach works but not at machine scale, and the honest outcome is a coupling framework rather than a foundation model. Both are measurable, and §5 says by what.

---

# 3. The ladder

Each rung adds **exactly one new capability**. A rung is complete when its gate passes; rungs are not skipped, and a failed gate stops the ladder rather than being worked around.

| # | Rung | New capability | Experts added | Gate |
|---|---|---|---|---|
| **0** | *(done)* Scaffold | agent graph, declared edges, message passing | none — identity placeholders | M0/M3 green ✔ |
| **1** | **RBC decomposition** ([[case-study-rbc-decomposition-atlas-0.1]]) | can frozen experts be coupled **at all** | 0 (one reused 3×) | flux residual; decomposed = monolithic |
| **2** | **Wind farm** ([[spec-wind-farm-wake-atlas-0.1]]) | **field ↔ lumped coupling**; a closed-form expert as a peer | 0 learned, 1 algebraic | thrust agreement $r_T$; Betz canary; $\mathcal R(t)$ |
| **3** | **Wind farm + stratification** | a **second learned expert** joins an existing graph; first true multi-family coupling | 1 (reuse [[noether-1.0-rbc]]) | non-regression on rung 2; `THERM` port residual |
| **4** | **Expert reuse test** | the **foundation-model claim itself**: does an expert trained for one scenario work in another? | 0 — reuse only | rung-2 expert transfers with ≤ light adaptation |
| **5** | **Aerofoil + structure** | **structural expert**; two-way FSI; the first genuinely new governing family | 1 (structural) | aeroelastic response vs. reference; $\mathcal R$ closes |
| **6** | **Thermal management loop** | conjugate heat transfer; **`THERM` at scale**; a lumped coolant circuit | 1 (conduction) + lumped | closed thermal balance around a loop |
| **7** | **Powertrain** | **`ELEC` and `ROT` ports**; motor map, battery, ERS as lumped experts | 2–3 lumped | energy balance across domains |
| **8** | **Tyre / contact** | contact mechanics, friction, thermal–mechanical–grip coupling | 1 | load transfer; temperature-dependent grip |
| **9** | **Full-vehicle graph** | $\sim$15–20 agents; **Claim B under real load** | integration only | $\mathcal R$ closes; composition error sub-linear in $N$ |
| **10** | **Design-parameter gradients** | differentiate performance w.r.t. geometry and setup | none | gradient agrees with finite differences |
| **11** | **Optimizer in the loop** | gradient-based design search | none | recovers a known-good design from a bad start |
| **12** | **Agent layer** | autonomous proposal and evaluation of design changes | none — *different research program* | see §6 |

**Rungs 1–2 are specified and buildable now with zero new training data and zero new experts.** Rungs 3–4 need no new experts either. **The first genuinely new expert is rung 5.**

## 3.1 Where the current work sits

Rung 0 is built. Rungs 1 and 2 are fully specified. Rung 4 — the reuse test — is the one most likely to be skipped by accident and is the one that actually tests the foundation-model claim, so it is scheduled explicitly rather than folded into "expansion."

> **2026-09-10:** this paragraph is the count as of 2026-08-19 and stays as written. §3.3 is the count now.

## 3.2 The rung that decides everything

**Rung 9.** Everything before it can succeed while the program still fails, because composition error might grow super-linearly and only show at scale. The $N=2,3,5,8$ sweep in [[case-study-wind-farm-wake-2d-atlas-0.1]] Phase F is the **early warning** for rung 9 and should be treated as such: it is a cheap look at the expensive question, and a bad result there is a reason to stop and rethink, not to press on.

## 3.3 Where the work sits, recounted 2026-09-10

§3.1 is this page's count as of 2026-08-19 and stays as written. The schedule has since been owned by [[case-study-ladder-to-f1]], which reordered this ladder on 2026-08-30 — *climb it classically, substitute learned experts afterwards* — without changing a rung, a gate or a claim; its §14.1 carries the evidence for each row below.

| rung | status on 2026-09-10 |
|---|---|
| **0** | built |
| **1** | answered on rung 2's graphs — frozen experts couple, and stably ([[case-study-ladder-to-f1]] §7, F2); the RBC case study named for it was never built |
| **2** | built — the wind farm, then the wake array |
| **3** | **built 2026-09-10 as a COMPILE, out of order — [[case-study-learned-pair-atlas-0.1]] (CS-17), the first graph holding two learned experts of different families.** Poseidon-T against NeuberNet across a `MECH` seam: `admit-uncertified`, **zero refusals**, because `L2/R10`'s W114 premise has no subject in a two-agent multiphysics graph (W186) — with R10's positive control refusing in the same compile, so the admit is the premise clearing and not a weakened rule. The §3.3 inference below is **confirmed in its strongest form**: $\tau$ is defined in the classical control and UNDEFINED in all three cells holding a checkpoint, and with *two* there is nothing on the seam to appeal to, which a campaign that swaps one side at a time cannot exhibit. **Not complete**: this rung's gate has no subject on this graph and is restated as three clauses in CS-17 §9.1, of which only the first is met (**W183**) |
| **4** | measured, and qualified: a certificate is the expert's *at a probe state*, so a library carries one per expert per regime |
| **5**, **6**, **7** | built **classically** — CS-12 `wing_fsi`, CS-13 `cooling_loop`, CS-14 `powertrain` |
| **8** | **first step done, rung open** — its admissibility question is answered and the answer is a **named hole**: an inequality-constrained seam compiles to the same verdict under the same decisions as an equality one, so the repair is a refusal on a declared condition class rather than a widened rule ([[inequality-seam-admissibility]], W182). The module reshapes from *couple a tyre* to *emit the measurement the missing rule would constrain* and no longer needs a contact expert; **§8's clause 4 — the passivity pair — still does**, and the rung's gate below is one this page must restate rather than quietly retire |
| **9** | blocked on a schema change |
| **10**, **11** | exercised on small graphs only, not on the vehicle graph; rung 11's gate as written has not been posed |
| **12** | a different research programme (§6) |

**The two halves of the programme now stand in different places.** §3.2's early warning was run (CS-7) and did not fire, and the coupling half has climbed to rung 7 with every expert classical. The foundation-model half has no learned expert admitted with a nonzero contribution ([[substitution-campaign-checkpoint]]): what has been demonstrated is §7's floor — a coupling framework plus classical verification — and the route that would lift it, a constrained expert class, is open and unattempted.

**[AI Inference]:** rung 3's absence matters more than its position suggests. The ladder assigned the foundation-model claim a configuration — two learned experts meeting in one graph — that no graph has ever held, and the substitution campaign tests a different one: a learned expert replacing a classical one, seam by seam. A negative from the campaign therefore does not answer rung 3's question, and a positive would not either.

---

# 4. Why the ladder is affordable: ports, not adapters

The ladder adds roughly a dozen experts. Under the old per-pair interface vocabulary that meant up to $\sim\!200$ bespoke contracts — an integration burden that ends projects. Under [[port-algebra-atlas-0.1]]:

- Each new expert declares **2–3 ports** from a **closed set of five** (`MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC`).
- Connection requires only that both sides expose the same port at coincident geometry. **No pairwise work.**
- **Rung $n+1$ costs what rung $n$ cost.** That flatness is the entire reason the ladder is finishable.

Two structural properties of the port formulation matter specifically for this roadmap:

**Unconnected ports make the roadmap visible in the model.** The wind-farm rotor at rung 2 declares a `ROT` port with nothing attached — the extracted power leaves through it. Rung 7 does not *integrate a powertrain*; it **connects that port**. Every rung above is, structurally, a port connection. The incomplete model states its own incompleteness.

**One diagnostic covers every rung.** The global power residual $\mathcal R(t)$ ([[port-algebra-atlas-0.1]] §6) is defined identically at rung 2 and rung 9. There is no per-rung metric engineering, and a regression at rung 9 is comparable to a measurement at rung 2.

---

# 5. Falsification, scheduled in advance

Recorded so the program cannot quietly move its goalposts:

| Criterion | Fails if | Measured at |
|---|---|---|
| **F1** | composition error grows **super-linearly** in interface count | rung 2 Phase F sweep (early), rung 9 (decisive) |
| **F2** | frozen experts cannot couple stably **without joint fine-tuning** | rungs 1–2 fixed-point convergence gate |
| **F3** | **integration hours per added expert** trend upward rather than flat | tracked from rung 2 onward — *log it as it happens, do not reconstruct it* |
| **F4** | a monolithic FM reaches the required coverage first | continuous; defended only by the field+lumped argument |
| **F5** | gradients through the composed stack are **too noisy to optimize with** | rung 10 |

**F5 deserves more weight than it usually gets.** Differentiability is the enabling property of the whole vision (§1.2), but a differentiable model is not automatically an *optimizable* one: frozen surrogates can have gradients that are well-defined and useless — dominated by high-frequency artefacts of the learned representation rather than by physics. **[AI Inference]:** the projection steps that enforce conservation ([[spec-wind-farm-wake-atlas-0.1]] §7.3) may help here, by pinning the composed output to a physically meaningful manifold and removing a class of spurious gradient directions — but this is speculation and rung 10 is where it gets tested. If F5 fails, the fallback is derivative-free search over a fast model, which is still valuable but is a materially weaker claim.

---

# 6. The agent layer, scoped honestly

Rung 12 — agents autonomously proposing and refining designs — is **a different research program**: design optimization and search, with its own literature, failure modes, and evaluation criteria. It is downstream of the physics stack and depends on it, which is a good reason to build the stack first and a bad reason to merge the claims.

Three things worth fixing now, because they change what the physics layer should expose:

1. **An agent needs to know when the model is out of its depth.** With 15–20 experts, the probability that at least one is outside its training regime approaches 1 on any novel design — and a novel design is the entire point. **Abstention and uncertainty (Gap 11, [[pfm-purpose-and-direction]]) stop being a nice-to-have and become the load-bearing safety property of the whole vision.** An optimizer will find and exploit exactly the regions where the model is confidently wrong. This is the single largest unsolved problem on the path and it is not on the ladder yet.
2. **The physics layer should expose a design parameterization, not just a state.** Rung 10's gradients are only useful if there is a differentiable path from *design parameters* (a geometry knob, a material choice, a control gain) to the objective. That path has to be designed in, and the natural time is rung 5, when geometry first becomes a variable rather than a constant.
3. **Verification stays in the loop.** The agent proposes; the composed model ranks; **the classical stack confirms**. An autonomous loop with no classical verification step is not a research goal, it is a way to generate confident nonsense at scale.

---

# 7. Honest assessment of the odds

Worth writing down while it is still cheap to be objective.

**Most likely to succeed:** rungs 1–4. They need no new experts, they have closed-form or monolithic gold standards to grade against, and their failure modes are diagnosable.

**Most likely to fail:** rung 9 (Claim B at scale) and F5 (usable gradients). Both are unmeasured and neither has a strong prior from adjacent literature.

**Most likely to be underestimated:** rung 5's structural expert. [[physics-simulation-datasets]] §3.4 found **no public corpus pairing transient conduction with quasi-static thermoelastic stress** — the structural family is the thinnest in the entire landscape, and it is the first genuinely new expert the ladder requires. Rung 5 is where the project stops being free.

**The most valuable outcome that is not the stated goal:** even if rung 9 fails and full-vehicle composition proves out of reach, a validated answer to *"does composition of frozen neural experts preserve physical validity, and how does error scale with interface count?"* is a real contribution. That question is currently unanswered, the answer is useful either way, and rungs 1–4 answer it. **The program has a floor.**

---

## See Also

- [[port-algebra-atlas-0.1]] — the interface vocabulary that makes §4's flat cost possible
- [[prior-art-and-novelty-atlas-0.1]] — what is and is not novel here; the source of Claims A and B
- [[pfm-purpose-and-direction]] — the earlier, broader vision statement this sharpens onto one target
- [[spec-wind-farm-wake-atlas-0.1]] — rung 2, specified in full
- [[case-study-rbc-decomposition-atlas-0.1]] — rung 1
- [[expert-library-atlas-0.1]] — how experts are cut; what each rung adds
- [[physics-simulation-datasets]] — why rung 5 is where data cost returns
- [[incremental-transfer-roadmap]] — the bootstrap strategy the ladder's expert additions follow
