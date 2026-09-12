# The Case-Study Ladder to the F1 Car — a revised build plan

**Type:** Core Concept — Program Plan (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-08-30, after the sixth real case study; **CS-7 run 2026-08-31 and section 7 updated with its measured values**; **CS-8 run 2026-08-31, Phase A complete, section 7's rung-4 row and section 8 updated** ([[tier0-measurements]] section 19, [[case-study-scaling-ladder-atlas-0.1]]). **2026-09-01: section 9 added** — a cross-machine re-read of the PoC 1a demo, the hybrid-library framing made standing rather than transitional, and one insertion (**CS-9★**, adaptive seam placement) — see [[poc1-retrospective-and-hybrid-roadmap]] for the full argument; this page carries only the resulting change to the schedule. Phase A is one case study in and the schedule below is confirmed rather than stopped. **2026-09-03: CS-10 run, section 11 added, and section 7's F5 row rewritten from a schedule entry into a measurement** ([[case-study-ground-effect-atlas-0.1]], [[gap-worklist]] Tier 24) — W97 closes, `InterfaceMotion` has all three of its numbers, and F5 is not falsified but fails in a way the criterion does not describe. Opened after the sixth real case study ([[case-study-wake-array-atlas-0.1]]). This page **revises** [[f1-pathmap-and-end-goal]] rather than replacing it: the end goal, the two claims and the falsification criteria are unchanged, and the *order and the currency* of the work are what move. Where the two disagree, the pathmap owns the goal and this page owns the schedule. **2026-09-10: §14 added** — CS-13, CS-14, CS-S1, Tier 39's two certified graphs and Tier 40's checkpoint, recorded against this schedule, with the rungs recounted; §1's count, §2.3's and §5's **[AI Inference]** and §7's stopping rule are each resolved where they stand rather than rewritten. **2026-09-10 (Tier 41): §15 added** — branch (a)'s first probe, CS-S2, comes back not compact with W95 binding; the certificate reports its margin with no verdict moved; W137 reverses; the critical path recounted.
**Related:** [[f1-pathmap-and-end-goal]] · [[gap-worklist]] · [[case-study-wake-array-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[atlas-implementation]] · [[end-to-end-architecture-spec]] · [[tier0-measurements]] · [[master-error-bound]] · [[incremental-transfer-roadmap]] · [[poc1-retrospective-and-hybrid-roadmap]]

> **Read §2 first if you read nothing else.** It is one strategic change, it is worth more than the rest of the page, and everything in §4 is scheduled by it.

---

# 1. Where the program actually stands

Eighteen tiers of measurement, six real case studies, one compiler with nine layers, and — measured against [[f1-pathmap-and-end-goal]]'s own ladder — **rung 1 to 2 of 12**.

> **Recounted 2026-09-10.** The sentence above is the count as of 2026-08-30 and stays as written; §14.1 is the count now. Rungs 0 and 2 are built, and rung 1's question — *can frozen experts be coupled at all* — was answered on rung 2's graphs, while the RBC case study named for it was never built. Rung 3 was never built and has no row in §4. Rung 4 is measured and the answer is qualified. Rungs 5, 6 and 7 are built, classically. Rung 8 is scoped and not started, rung 9 is blocked, rungs 10 and 11 are exercised on small graphs only, and rung 12 is a separate programme.

That is not a slow result; the last five days built the *certification machinery*, which is the part nobody else has. But it is worth stating plainly what the machinery has and has not been pointed at:

| | |
|---|---|
| real case studies | **six**, and **all six are 2-D incompressible Navier–Stokes**, plus one thermoelastic seam |
| governing families ever coupled | **two**, at exactly one seam ([[gap-worklist]] Tier 13) |
| agents in the largest real graph | **nine** ([[case-study-wake-array-atlas-0.1]]), against rung 9's $15$–$20$ |
| composition error vs. interface count | **never measured**, at any $N$ |
| gradients through a composed stack | **never measured** |
| a learned expert reused in a second scenario | **never done** |

The last three rows are [[f1-pathmap-and-end-goal]] §5's criteria **F1**, **F5** and rung **4** — its own three most-likely-to-fail items — and each is *cheaper* than the case study just finished. That inversion is the schedule error this page corrects.

## 1.1 The one result that reframes everything

**W93.** `probe.support_reach` measures Poseidon-T's domain of dependence as the **whole $128$-cell window** where the record declares $2$ cells; `required_halo()` now returns `None`; `L2/R10` decertifies, and **refuses** outright once `elliptic_subsolve` is declared at the value the same probe measures.

A neural operator's receptive field is global by construction. So, stated as sharply as it deserves:

> **The framework cannot today certify a decomposition whose agents are globally-receptive pretrained operators — and that is nearly all of them.**

Every learned expert added to an F1 graph inherits that. §2 is the answer, and it is not "fix R10."

---

# 2. The revision: climb the ladder classically, substitute learned experts afterwards

## 2.1 The observation

`atlas/` does not know what an expert *is*. An `ExpertCapabilities` record needs a `boundary_response` callable, a `dt_native`, a `governing_family` and a port list. A finite-volume solver satisfies that. So does a $20.8$ M-parameter checkpoint. Four of the six real case studies are **classical solvers wearing the expert interface**, and the compiler treats them identically to the checkpoint.

And `composition.certify_substitution` exists for exactly one purpose: **swapping one expert for another at a seam, and saying whether the swap is admissible.** It has been exercised on a real swap once — WindowNS $\to$ Poseidon-T at the wake seam, where it correctly returned `refuse`.

## 2.2 The consequence

The pathmap's ladder tacitly assumes learned experts throughout, and therefore inherits their costs at every rung: training data, a receptive field nobody declared, a referent that cannot be built. **None of that is required to establish Claims A and B.** Splitting the program in two:

$$\underbrace{\textbf{Does composition preserve validity and scale?}}_{\text{Claims A and B — classical experts suffice}} \quad\perp\quad \underbrace{\textbf{Is it fast?}}_{\text{learned experts, one certified substitution at a time}}$$

Four things fall out, and each one removes a blocker from the critical path:

| blocker | under "learned throughout" | under "classical first" |
|---|---|---|
| **W93 / R10** — global receptive field refuses the graph | blocks every rung | a property of the **substitution step**, priced per seam |
| **W95** — a learned expert cannot be its own referent | blocks attribution everywhere | *the reason* the classical expert had to exist anyway; it **is** the referent |
| **rung 5's data cost** — no public corpus pairs transient conduction with thermoelastic stress ([[f1-pathmap-and-end-goal]] §7) | *"where the project stops being free"* | **deferred**, not incurred — see §2.3 |
| **F3** — integration hours per added expert | measured on the hardest possible expert | measured on the cheap one first, so the trend is visible before it is expensive |

## 2.3 And the structural expert already exists

The pathmap names rung 5 as the first genuinely new expert and the point at which the program stops being free. Checked against the build repo, that is **true of a learned structural expert and false of a classical one**:

`src/atlas/solvers/thermostruct2d.py` — Q1 bilinear finite elements, **backward-Euler conduction** and **quasi-static plane-stress elasticity**, with `solve_mechanical(T, p_in, p_out, ...)` taking **pressure loads from both sides** and a temperature field for thermal strain. It is graded against closed-form oracles by the build repo's own M1 suite, and [[gap-worklist]] Tier 13 already imports it **unmodified** into `cases/thermal_seam.py`.

Its own docstring says what it was built for: *"exactly the b-c and c-d coupling Atlas will later have to reproduce through its typed edges."*

So rung 5's gate — *aeroelastic response vs. reference; $\mathcal R$ closes* — is reachable with **zero new training data**. What is deferred to the substitution campaign is only the *speed*, which nothing gates on until rung 10.

**[AI Inference]:** the same argument probably holds for rungs 6 and 7. Conduction is in `thermostruct2d`, a compressible gas is in `compressible2d`, and a motor map, a battery and a coolant circuit are **lumped algebraic experts with zero fitted parameters** — the class `disk.ActuatorDisk` already belongs to, which cost an afternoon. On this reading the first rung that genuinely requires a new *learned* expert is not rung 5 but the substitution campaign, and the first that requires new *physics code* is rung 8's contact model. This is a claim about the build repo's contents and it should be checked against each solver before a rung is scheduled on it.

> **Checked 2026-09-10, and it held on both halves.** CS-13 (rung 6) was built from one imported solver — `ThermoStruct2D.step_thermal`, unmodified — and four closed-form coolant legs of the `disk.ActuatorDisk` class, and CS-14 (rung 7) from the build repo's actuator disk and closed-form circuit elements: no new field solver in either. And [[gap-worklist]] W165 found the build repo has **no contact solver** — its one match for *contact* is HLLC's contact discontinuity — so rung 8 is the first rung that needs new physics code, as the inference said. What it did not foresee is that a contact seam's interface condition is a **complementarity** condition rather than an equation, so whether such a seam is admissible at all is a question for a page before it is one for a module.

## 2.4 What this costs, stated honestly

Three things, and the third is the real one:

1. **The end goal needs the learned experts.** A classical F1 graph is a coupled simulation, which exists already and is slow. The composed-model speedup — [[f1-pathmap-and-end-goal]] §1.2's four-to-six orders in the unit economics of search — arrives only with substitution. Climbing classically proves Claims A and B and proves **nothing about the vision**.
2. **Claim B might be expert-dependent.** Sub-linear composition error for a family of classical solvers does not imply it for checkpoints. CS-7 (§4) therefore runs **both columns at every $N$**, which is the whole reason it is scheduled first.
3. **It defers the hardest question rather than answering it.** R10 does not go away. It becomes a scheduled, priced experiment (§5's substitution campaign) instead of a wall standing in front of every rung — but if the answer there is *no admissible substitution exists at any seam*, the program's floor is a coupling framework, exactly as §7 of the pathmap says.

---

# 3. Reordering by coupling kind, not by subsystem

An F1 car is not twenty physics problems; it is four **coupling kinds** repeated. The port algebra has surface bonds for one of them.

| coupling kind | example in the car | does a bond exist? | row |
|---|---|---|---|
| field ↔ field, surface | wing wake into the floor's boundary layer | **yes** — `MECH`, and it is the only one exercised | — |
| field ↔ lumped, surface | aero load into a suspension spring; a motor's torque into a shaft | **yes, and the certificate is blind** by the cell Reynolds number | **W97** |
| **two-way volumetric** | thermal strain in a brake disc; tyre temperature into grip; Joule heating in a battery | **no port and no bond** | **W70**, **W94** |
| **moving / deforming interface** | ride height under aero load; a deflecting wing; a rotating wheel | **no** — `motion_class` is `static` and anything else is refused | **W22 / W30** |

Add the timescale axis — control at kHz, aero at kHz, brake thermal at Hz, a lap at $10^{-2}$ Hz — and the multirate defect has **no bounding rule at all** (**W90**: R9 is aimed at a term $62\times$ smaller than the one that matters).

**Every F1 subsystem is a combination of these four.** Building a subsystem before its bond exists means each subsystem case study rediscovers the same missing bond — which is the per-pair integration burden [[port-algebra-atlas-0.1]] was written to avoid, reappearing one level up. So the ladder below builds **bonds before subsystems**, which is §4 of the pathmap applied to the schedule rather than to the vocabulary.

---

# 4. The ladder

Six real case studies exist, so these are numbered from seven. Each row states the thing `atlas/CASE-STUDY-GUIDE.md` demands of a seventh: **the question none of the existing ones can answer.**

Three phases. **Phase A needs no new experts, no new physics code, and no data.** Phase B needs no data. Phase C is where the subsystems are, and it stays cheap only if §2.3 holds.

| # | case study | the question only it can answer | new experts | rung | cost |
|---|---|---|---|---|---|
| **CS-7** | `scaling_ladder.py` | does composition error grow **sub-linearly in interface count** — for a classical expert *and* for a frozen checkpoint? | none | **9, early** | days |
| **CS-8** | `reuse_probe.py` | does an expert's **certificate travel with the expert**, or is it a property of the state it was probed at? | none | **4** | days |
| **CS-9** | `thermal_strain.py` | what is the **bond** for a two-way volumetric coupling? | none (splits an existing one) | — | ~1 week |
| **CS-10** | `ground_effect.py` | can a **moving interface** carry a design parameter, and are the gradients through it usable? | one lumped (spring) | 5-adj. | ~1 week |
| **CS-11** | `brake_thermal.py` | what **bounds** the multirate lag defect at a real clock ratio? | none | 6-adj. | ~1 week |
| **CS-12** | `wing_fsi.py` | two-way FSI: does $\mathcal R$ close across a **genuinely new governing family**? | none (classical) | **5** | ~1–2 weeks |
| **CS-13** | `cooling_loop.py` | does a **cyclic** port graph close its own thermal balance? | 1–2 lumped | **6** | ~2 weeks |
| **CS-14** | `powertrain.py` | `ELEC` and `ROT`: does energy balance across **domains** rather than across a seam? | 2–3 lumped | **7** | ~2 weeks |
| **CS-15** | `tyre_contact.py` | thermal → grip → load path, on CS-9's bond | 1 | **8** | ~3 weeks |
| **CS-16** | `vehicle.py` | **Claim B under real load** — $\sim18$ agents, all four coupling kinds | integration only | **9** | ~1 month |

Running alongside from Phase B onward: **the substitution campaign** (§5), which is where the learned experts, R10, and every data cost live.

## Phase A — decide the forks before spending anything

### CS-7 · `scaling_ladder.py` — Claim B, cheaply, now

**The question.** [[f1-pathmap-and-end-goal]] §3.2 names rung 9 as *the rung that decides everything* and §5's **F1** as the criterion that fails if composition error grows super-linearly in interface count. It has never been measured at any $N$, and the sweep the pathmap itself calls the *early warning* has never been run.

**The construction.** The wake array's tiling is already parameterized by `N_COL`, `N_ROW`, `STRIDE` and `HALO`; what is not parameterized is the rotor layout, the controls and the referent, which is why this is a new file rather than a flag. Grow a $1\times1$ tiling to $6\times4$ — $N \in \{1, 2, 6, 12, 24\}$ windows, $\{0,1,3,6,12\}$ rotors — holding $\Delta t$, $\mathrm{d}x$, the ramp and the disk model fixed, and measure at every $N$:

- the composed defect against the **same expert's monolith**, which exists for `WindowNS` at any resolution and **does not exist for Poseidon-T at any resolution** — so the checkpoint's column is measured against a `WindowNS` pair, per `lambda_ref`
- $\lVert\sum_i \chi_i\lvert D_i\rvert\rVert$, the L2/C2 cut-defect bound, whose tightness is known at $N=4$ ($0.2\%$) and unknown at $N=24$
- $\Pi$ and $C_\mu$, which have transferred as a **bound** and never as a calibration across experts (**W55**)
- $\tau$, $\sigma$, $\gamma$ per seam, depth-tagged (**W34**) and harness-tagged (**W54**)
- wall time per macro-step, which is Claim B's *other* half — $O(1)$ integration work per added agent

**The gate.** Composed defect grows **no faster than linearly** in interface count, in both columns, over $24\times$ in $N$. Wall time per agent flat.

**Why it is first.** It is measurement on machinery that already exists, it needs no physics that does not, and a bad result is the pathmap's own stated reason to stop and rethink rather than press on. Everything below it is conditioned on it.

**W-rows it must close or price:** **W58** (which of two definitions `cut_defect_bound` was measured by — a provenance field and a decidable geometric check), **W54** (an emitted defect must carry the harness parameters it is a function of, not only its depth), **W81** ($\beta_{\min}$ is undefined framework-wide and is *the whole discriminating power* of the plug-in guarantee).

### CS-8 · `reuse_probe.py` — the foundation-model claim itself

**The question.** [[f1-pathmap-and-end-goal]] §3.1 flags rung 4 as *the one most likely to be skipped by accident*, and it is the only rung that tests the foundation-model claim rather than the coupling claim. Sharpened by everything Tiers 14–18 measured, it is not *"does the expert work in a new scenario"* but:

> **Is a certificate a property of the expert, or of the state it was probed at?**

This matters more for the F1 goal than it looks. A $20$-agent car carries $\sim20$ conformance certificates and $\sim40$ seam certificates. If each is state-specific, **the plug-in claim is per-design rather than per-expert** and the whole economic argument — search wide and cheap — collapses back to re-certifying every design.

**The construction.** One frozen Poseidon-T window, three unrelated flows on identical geometry: the turbine wake it was measured in, a bluff-body wake, and a decaying shear layer with no body force at all. Per flow, re-run the *existing* instruments — `support_reach`, `operator_content`, `elliptic_signature`, `assemble_seam`, `conformance.certify`, $\Xi$, and $\tau$ against a `WindowNS` pair — and compare against the wake array's own recorded values.

**The gate.** Either the certified fields reproduce within their measured reproducibility floor ($\sim10^{-4}$ for this checkpoint, [[gap-worklist]] Tier 11), or the run **names which fields are state-dependent and by how much**. The second outcome is as useful as the first and is the more likely one: `probe_state` is already derived from the base rather than declared (**W77**), so the machinery to say so exists.

**Also here, and not a case study:** the **F5 gradient probe** on CS-7's smallest graph — $\mathrm{d}P/\mathrm{d}a$ by finite difference against the same derivative taken through the composition — and **W13**, a `validity` predicate on a learned expert with a measured declination rate $p$, which is what $T_{\text{abs}}$ and every abstention claim wait on.

## Phase B — build the bonds an F1 car needs

### CS-9 · `thermal_strain.py` — a bond for two-way volumetric coupling

**The question.** **W70** closed by *reframing*: the shell's conduction–elasticity coupling is one expert's internals and one-way, so nothing needed a bond. **W94** then found the case that does not have that escape: an actuator disk against a frozen operator is two-way volumetric, and the port algebra has five surface bonds and nothing else. The ledger row has stood open since Tier 16.

Three of the four F1 subsystems in Phase C need it — brake (thermal $\to$ strain $\to$ contact), tyre (thermal $\to$ grip), battery (Joule $\to$ thermal $\to$ ageing). **It is the single highest-leverage missing piece of vocabulary.**

**The construction.** Split `ThermoStruct2D` into two agents — a conduction agent and an elasticity agent — coupled through thermal strain, which is a **volume** term, and demand a declared bond. Exercise `PortAmendment`'s six-field procedure for the first time (**W32**), which is the defined operation for extending a closed port set. The control is that the unsplit solver is right there to grade against.

**The gate.** The split graph reproduces the monolithic solver's stress field, and $\mathcal R$ closes with the volumetric term in it. The amendment is refused if the object turns out not to be a bond, which is a legitimate outcome and the one W70 already argued for once.

### CS-10 · `ground_effect.py` — a moving interface, and the first design parameter

**The question.** Aero load changes ride height, ride height changes aero. [[f1-pathmap-and-end-goal]] §1.1 names this as the coupling that *is* the physics, and every interface in the vault is `motion_class=STATIC` with anything else refused.

**The construction.** A 2-D wing section over a **moving floor** at ride height $h$, with the integrated aero load fed to a lumped spring — $k(h_0 - h) = L(h)$ — whose solution moves the interface. Flow expert: `WindowNS` first, Poseidon-T under substitution. The suspension is a two-line algebraic expert with zero fitted parameters, the `disk.ActuatorDisk` pattern exactly.

**Three things only this case study gets:**

1. **`InterfaceMotion`** stops being a named hole with a declared interface and becomes a measured one; the operator drift $\lVert S(t + K\Delta t) - S(t)\rVert$ already exists as a number on *static* runs (**W30**) and now has something to be compared against.
2. **W97's blind certificate** has to be closed here or admitted permanently — a spring against a flow field is a field-to-lumped `MECH` seam, and if its substitution certificate is blind then **no lumped subsystem in the car can be certified**. The candidate repair is §4.1's conservative co-normal, which **W47 scoped out** at fluid–fluid seams because the advective term cancels there; at this seam it does not.
3. **The first real design parameter.** $h_0$ is a *knob*, not a state, so this is where F5 gets its honest test: is $\mathrm{d}(\text{downforce})/\mathrm{d}h_0$ through the composed stack physics, or is it the high-frequency artefact of a learned representation? [[f1-pathmap-and-end-goal]] §6.2 says the differentiable path from *design parameters* has to be designed in and that rung 5 is the natural time. This is that moment.

### CS-11 · `brake_thermal.py` — a bound for the multirate defect

**The question.** **W90**: R9 covers the flux transient and is $62.4\times$ smaller than the stale-trace term, which has no rule at all. R4 forbids shrinking the exchange interval below $\max_i \Delta t_i$, and a frozen checkpoint's `dt_native` is not a dial — so the one obvious remedy is the one axis a learned expert cannot move. The candidate is **W17**'s $W>1$ waveform relaxation: a trace carried as a waveform is exactly a trace that is not stale.

**The construction.** `ThermoStruct2D` conduction in a disc (slow, backward Euler, `EMBEDDED`) against a convective duct flow (fast), at the ratio the physics actually gives. `thermal_seam` already carries a $500{:}1$ mismatch and compiles; what it does not have is a **bound**.

**The gate.** A $\sigma$ term that is a function of the exchange interval and holds over a swept clock ratio — or an explicit statement that multirate composition is uncertified and by how much, which is the honest fallback and is what `L7/R9/lag` currently decertifies with.

## Phase C — the subsystems

Each is a pathmap rung, each is scheduled only after the bond it needs exists, and each is built **classically first**.

- **CS-12 `wing_fsi.py`** — rung 5. `ThermoStruct2D.solve_mechanical` against the flow, two-way. The first genuinely new governing family in a *two-way* coupling rather than at one seam. Builds on CS-10's moving interface, since a deflecting wing is one.
- **CS-13 `cooling_loop.py`** — rung 6. Conjugate heat transfer plus a **closed** lumped coolant circuit: the first *cyclic* port graph, where path-dependence of message passing is real ([[spec-wind-farm-wake-atlas-0.1]] §5.1 raised it and no built graph has tested it). Gate: thermal balance closes around the loop.
- **CS-14 `powertrain.py`** — rung 7. **This is the rung that connects the wake array's own open `ROT` port**, which is what [[f1-pathmap-and-end-goal]] §4 means by the incomplete model stating its own incompleteness. Adds `ELEC`; a motor map and a battery are lumped algebraic experts.
- **CS-15 `tyre_contact.py`** — rung 8. Needs CS-9's volumetric bond and CS-11's multirate bound. The first genuinely new *physics code* rather than a new composition.
- **CS-16 `vehicle.py`** — rung 9. $\sim18$ agents, all four coupling kinds, Claim B under real load, graded against CS-7's prediction. **If CS-7 said super-linear, this is not built.**

---

# 5. The substitution campaign — where the learned experts and the data cost live

Runs in parallel from Phase B. For each classical expert in a compiled graph:

1. Obtain a learned expert for that family — pretrained where one exists ([[incremental-transfer-roadmap]] confirms only Poseidon and Walrus as available), fine-tuned or trained where none does.
2. Probe it: `support_reach`, `operator_content`, `conformance.certify`, and a `lambda_ref` naming the classical expert it replaces — which by **W95** it cannot supply for itself.
3. `certify_substitution` at every seam it touches, at a declared $\beta_{\min}$ (**W81**).
4. Record `refuse` / `admit` / `blind` per seam, and the wall-time ratio bought.

**This is where R10 is decided, and it is decided per seam rather than per program.** Three outcomes and all three are results:

| outcome | what it means | what the program becomes |
|---|---|---|
| some architectures pass the halo rule | bounded-receptive-field operators exist and are usable | the vision, on a constrained expert class |
| none pass, but a **tolerance-halo** rule is derivable | a global response with a decaying tail is bounded, not unbounded | the vision, with a new theory branch |
| none pass and no rule is derivable | learned experts are composable and **not certifiable** | a fast uncertified searcher plus classical verification — [[f1-pathmap-and-end-goal]] §1.2's deployment story, unchanged |

**[AI Inference]:** the middle row is the one worth attacking first and no page in this vault has proposed it. `support_reach` currently asks whether the response is *nonzero* past the declared radius; the physically meaningful question is whether it is *larger than the defect the composition already tolerates*. An $\varepsilon$-halo — the radius beyond which the response falls below $\varepsilon_{\text{tol}}$, with the truncated tail carried as an explicit term in $\sigma$ — would be a bounded halo for an operator whose support is formally global, and $\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ already exists in the compiler. Whether the tail is summable is an empirical question about the architecture and is measurable with the probe that exists. **This is a hypothesis, it is not derived, and it should be attacked at CS-7 where the machinery is already in hand.**

> **Resolved 2026-09-10 (Tier 40), by measurement: the middle row does not exist for the scOT class.** [[epsilon-halo-measurement]] band-truncates the dense seam operator, $E(r)=\lVert S-S_r\rVert$, which is exactly the tail term the paragraph above asks to carry in $\sigma$. On Poseidon-T and Poseidon-B, $E(r)/\lVert S\rVert_2$ plateaus at $0.27$–$0.75$ from $r\approx16$ to $r=64$ on a $128$-cell seam, and the transverse response never falls to $1\%$ of its face value at any depth, while the positive control — split-step `WindowNS` — reads exactly zero past $r=13$. Two further kills need no decay at all: $\Pi=1$ for every halo of $16$ cells or more, and $\varepsilon_{\text{tol}}$ has no value on the graph, because $\tau$ and $\sigma$ are unmeasurable for a checkpoint fixed at one resolution. So the checkpoints give the table's **third** row, the classical embedded-elliptic solver gives its **second** — $r^\star(10^{-2}) = 46$–$70$ cells, real and worth nothing at the widths a tiling can afford — and no rule was written. The paragraph above is left as written; this is the answer to it. **The first row — *some architectures pass the halo rule* — is unmeasured**, and §14.5 schedules its first probe.

---

# 6. The W-rows on the critical path

Everything else on [[gap-worklist]] is real and none of it blocks this schedule. These eleven do:

| row | what it blocks | scheduled at |
|---|---|---|
| **W81** — $\beta_{\min}$ undefined | every substitution verdict in the program | **CS-7** |
| **W58** — which `cut_defect_bound` was measured | Claim B's own number is ambiguous | **CS-7** |
| **W54** — defects carry depth but not harness parameters | three composition defects have worn an agent's label; a fourth is a matter of time | **CS-7** |
| **W13** — abstention | $T_{\text{usable}}$, W36, and the *load-bearing safety property* of the agent layer | **CS-8** |
| **W93 / R10** — global receptive field | every learned substitution | **substitution campaign** |
| **W95** — no referent from a checkpoint alone | attribution at any learned seam | closed *by* classical-first |
| **W94 / W70** — no volumetric bond | brake, tyre, battery | **CS-9** |
| **W32** — `PortAmendment` never exercised | any new bond at all | **CS-9** |
| **W97** — field-to-lumped certificate blind | every lumped subsystem in the car | **CS-10** |
| **W30 / W22** — interface motion | ride height, deflection, rotation | **CS-10** |
| **W114** — `L2/R10` refuses a graph its own derivation does not reach | every graph with a quasi-static structural agent, which is every FSI seam in the car | **CS-12 — CLOSED 2026-09-04.** R10 now checks the premise its own sentence names, by asking whether another agent shares the `EMBEDDED` agent's `governing_family`. Four graphs' verdicts move; no measurement moves with them; `window_ns` `as-built` keeps its refusal |
| **W90** — no bound on the multirate lag | every pair of subsystems on different clocks | **CS-11 — CLOSED 2026-09-04.** [[master-error-bound]] §4.2: first order in the exchange interval, two probe constants and one run rate, $1.43\times$–$2.18\times$ loose over $2000\times$ of clock ratio, and the clock ratio turns out not to be the variable |

Two more that are not blockers and are getting worse with every added expert: **W69**'s *three unverifiable declarations* (`validity`, `response_half`, `governing_family`) scale linearly in expert count, and **W99** is a live silent bug in the build repo's own adapter.

> **2026-09-10: one row in this table did not survive, and the path has moved.** **W95** reads *closed by classical-first*, and at the substitution certificate it is what binds: $\beta_{\min}$ derives from $\varepsilon_{\text{tol}}=\min(\tau,\sigma)$, and neither exists for an expert with no same-class reference pair ([[substitution-campaign-checkpoint]] §3.3, [[gap-worklist]] W177). Classical-first routed around it for the ladder and left it standing for the campaign. The rows that now stand on the critical path are in §14.4.

---

# 7. Gates, and what would stop the program

[[f1-pathmap-and-end-goal]] §5's five criteria, restated against this schedule so they cannot be quietly moved:

| criterion | fails if | now measured at | previously |
|---|---|---|---|
| **F1** | composition error super-linear in interface count | **MEASURED 2026-08-31, not falsified.** Over $1 \to 24$ coupled windows: $+0.804\ [+0.641, +0.967]$ classical and $+0.477\ [+0.362, +0.593]$ on the frozen checkpoint, sub-linear in both. **With the physics held fixed the interface-only exponent is $+0.14$ to $+0.20$** — most of the headline is the flow getting harder, not the cut getting longer (**W102**). Read it as F1 asks and no further: five rungs, one geometry, one Reynolds number, $24$ agents against rung 9's $15$–$20$ | rung 9, last |
| **F2** | frozen experts cannot couple stably without joint fine-tuning | **MEASURED 2026-08-31, and the answer inverts the expectation.** The *frozen checkpoint* couples stably at every rung to $110$ macro-steps with its assembled divergence flat; the *classical solver* is the one whose composed rollout goes unstable, at $N \ge 6$, earlier as the graph grows (**W100**). Not fine-tuning — the difference is where the elliptic part lives, and one global projection in the composition layer fixes every rung | rungs 1–2 |
| **F3** | integration hours per added expert trend upward | **MEASURED 2026-08-31, and the columns answer oppositely.** Per agent, $N=24$ against $N=2$: **Poseidon-T $0.32\times$** — it gets *cheaper* per agent as the graph grows, which is better than $O(1)$ — and **WindowNS $3.84\times$**, whose CFL sub-step count is held at $21$ throughout so its *work* per agent is constant and the rise is the memory hierarchy. Neither transfers off this host (**W104**) | *"do not reconstruct it"* — and it had not been logged |
| **F5** | gradients too noisy to optimize with | **MEASURED 2026-09-03 at CS-10, not falsified, and it fails in a way this criterion does not describe.** $\mathrm d(\text{downforce})/\mathrm dh_0$ by reverse-mode adjoint through the composed stack — every sub-exchange of every window's solve, the blend, the global Leray projection, and the seam's own Newton solve at both ends — agrees with a central finite difference to $\mathbf{1.32\times10^{-6}}$ at $\delta = 10^{-7}$, on a truncation branch monotone over five decades with the cancellation branch not reached, at $4.3$–$6.7$ forward evaluations on a bit-reproducible objective. F5's stated risk is *high-frequency artefacts of a learned representation*, and there is no learned representation in this column. **What bites is the horizon**: $\mathrm dJ/\mathrm dh_0$ is $+0.411$ at $40$ macro-steps and $-0.095$, $-0.152$, $-0.055$ at $80$, $160$, $240$ — a **sign change**, with every value confirmed by its own finite difference, so a design search run at the short horizon moves the knob the wrong way. Opened as **W128** ([[case-study-ground-effect-atlas-0.1]] §6) | rung 10 |
| **rung 4** — *is a certificate the expert's or the state's?* | the plug-in claim is per-design rather than per-expert | **MEASURED 2026-08-31, and neither noun is right.** Over $55$ probes of one swap: the norms move $2$–$6\%$ with the probe state, $\Xi$ moves $55.9\%$, and **both $\beta_{\min}$ thresholds move $155\%$ — more than their own size** — while *doubling the graph* moves everything under $4\%$. The verdict is `refuse` at every cell and every admissible tolerance, but only because $\lVert\Delta\rVert/\beta > 1$ everywhere and its minimum is $0.25\%$ from flipping. **A regime label computable from the layout with no run captures $5$–$53\times$ of the seam-to-seam variation**, so a library carries one certificate *per expert per regime* ([[tier0-measurements]] §20, [[case-study-reuse-probe-atlas-0.1]]) | rung 4, *"most likely to be skipped by accident"* |
| **F4** | a monolithic FM reaches the coverage first | continuous | continuous |

**One new stopping rule, which the pathmap does not have.** If the substitution campaign returns `refuse` or `blind` at **every** seam of **every** case study, the composed model cannot be certified with learned experts at all, and the honest outcome is the framework rather than the foundation model. That is [[f1-pathmap-and-end-goal]] §7's floor, and it is worth naming a checkpoint at which it is declared rather than approached asymptotically: **after CS-12**, by which point three expert families and all four coupling kinds have been tried.

> **Stood on 2026-09-10 (Tier 40): the rule does not fire.** [[substitution-campaign-checkpoint]] counted what had been quoted from memory. *Nine of nine seams refused* was one graph-level `L2/R10` decision painted across nine tiles, and conditional on a declaration — `elliptic_subsolve=embedded` — that the vault records as undeclarable; compiled as `poseidon_capabilities` declares it, no seam of any graph refuses the learned expert. And the campaign's own instrument, `certify_substitution`, run at all eight agent-sides of the graph holding the live weights, returns an informative `admit` at three — with $\lVert\Delta\rVert/\lVert S_i\rVert = 0.95$–$1.01$ on every side, so what it admits is within $5\%$ of deleting the block. The declaration is three-part rather than one letter: **(c)** a coupling framework plus classical verification is *demonstrated*; **(a)** a foundation model on a constrained expert class is the **one live route, and unattempted**; **(b)** the theory branch — §5's $\varepsilon$-halo — is **closed**. The paragraph below stands unchanged, and the checkpoint leaned on it.

**And one thing that is not a stopping rule.** `admit-uncertified` is not failure. Six real case studies have landed there and the certificates they could not issue were named, not assumed. A car that composes, runs fast, and reports honestly which of its seams are uncertified is worth building; a car that reports `admit` because nobody measured the constant is the silent-wrongness class this whole framework exists to refuse.

---

# 8. Where the schedule stands after CS-7

**CS-7 is run** ([[tier0-measurements]] §19, [[case-study-scaling-ladder-atlas-0.1]]),
with W58, W54 and W81 closed along the way. §4's schedule is **confirmed rather
than stopped**: F1 was not falsified, so CS-16 is still on the board and Phase C
is still worth building toward.

Three things it changed on this page rather than merely filling in.

**§4's gate for CS-16 should be stated against the fixed-physics exponent, not
the headline.** §19.3's control shows that growing the farm grows the physics as
well as the interface count, and the two contribute $+0.14$ and $+0.7$
respectively to a headline of $+0.85$. A gate written against the headline would
pass a graph whose per-interface error was climbing, as long as its physics
happened to get easier. **W102.**

**§2.2's table gets a fourth row, and it is the one nobody had priced.** The
classical-first strategy assumed the classical column is the safe one — it is the
referent, it has a monolith, its rules are known. Measured, **it is the column
whose composed rollout goes unstable**, and the frozen checkpoint is the stable
one, because W98's repair had already moved the checkpoint's elliptic part into
the composition layer and the classical column's is still inside the agent.
Climbing classically is still right; it is not *free*, and the first thing every
classical case study below needs is the global projection of §19.6. **W100.**

**F3 has numbers and they point in opposite directions**, so the *"integration
hours per added expert"* row cannot be read as one trend. On this problem the
learned expert amortizes and the classical one does not, which is the reverse of
the assumption in §2.4's cost argument.

## Where the schedule stands after CS-8

**CS-8 is run** ([[tier0-measurements]] §20, [[case-study-reuse-probe-atlas-0.1]]).
W81's `visible_above` / `fails_above` pair was exactly the instrument the row
predicted it would be, and it answered: **a certificate is a property of the
expert *at a probe state*, and the state half dominates by twenty to forty
times.** Phase A is complete and §4's schedule stands.

Three things it changes on this page rather than merely filling in.

**§2's economic argument needs a third noun.** The classical-first split assumes
certification is a per-expert cost paid once. Measured, it is neither per-expert
nor per-design: **a regime label computed from the layout with no run at all
captures $5$ to $53$ times the seam-to-seam variation**, so the unit that
amortizes is *expert × regime*. That cost grows with the vocabulary of flow
situations rather than with the number of designs searched — weaker than the
pathmap's claim, and very much cheaper than re-certification.

**§5's substitution campaign should record a probe state beside every verdict.**
It currently specifies `refuse` / `admit` / `blind` per seam at a declared
$\beta_{\min}$. On this evidence that row is incomplete: the verdict is stable
only because it is saturated, and the margin deciding it ranges $153\times$
across probe states. **W107** and **W109**.

**And $\Xi$ cannot rank a library until its reproducibility is quoted.**
[[expert-library-atlas-0.1]]'s composability axis disagrees by up to $49\%$
between two seams the taxonomy calls the same regime. The axis is real; the
ranking built on it is not yet safe.

## The next step

**CS-9, `thermal_strain.py`** — §4's Phase B first row, and the single
highest-leverage missing piece of vocabulary: **a bond for two-way volumetric
coupling**, which three of the four Phase C subsystems need and which **W94** has
had open since Tier 16. It splits `ThermoStruct2D` into a conduction agent and an
elasticity agent coupled through thermal strain, exercises `PortAmendment`'s
six-field procedure for the first time (**W32**), and has the unsplit solver
right there to grade against.

**One thing CS-8 hands it, and one warning.** The **exposed-agent plus
`ProjectedAssembly` column now compiles with zero refusals at both sizes**, so
the classical column CS-9 will build on is the first in this vault `R10` does not
refuse — start there and not from `wake_array.reference_solver`. The warning is
**W106**: CS-9 will want to say a split reproduces the monolith *within
reproducibility*, and this pipeline's measured floor is **exactly zero**, so that
phrase does not bound anything. Quote a movement against the quantity's own level
or against a replicate, not against a floor.

**Also still outstanding from Phase A, and now explicitly unscheduled:** the
**F5 gradient probe** on CS-7's smallest graph, and **W13**'s abstention
predicate. CS-8 was budgeted to the reuse question and did neither; §7's F5 row
records that rather than letting the plan imply it was covered.

---

# 9. Revision after PoC 1a, on a second machine (2026-09-01)

Full argument: [[poc1-retrospective-and-hybrid-roadmap]]. Summary of what it changes here.

**A second machine confirms §8's F3 finding, and sharpens it.** PoC 1a's demo — an all-classical comparison, `wake_array` windows vs. `scaling_ladder.reference_monolith`, no learned expert anywhere in it — measured $10$–$25\%$ **slower** on the decomposed column on a Mac, against $-16.5\%$ on the original dev box. Same sign, same order of magnitude, two different memory hierarchies. §8 already found that the classical column's per-agent cost *rises* with $N$ while a frozen checkpoint's *falls*, and flagged that neither number transfers off its host (**W104**); this is the first evidence that the *qualitative* direction does transfer, even though the ratio does not. PoC 1a's own demo runs at $N=6$, inside the range on both machines where decomposition has not yet been paid for — so the result is exactly what §2.4's cost table already predicted, not a new finding about the demo.

**§2's split gets a corrected destination, not a new mechanism.** §2 already separates *"does composition work"* (classical suffices) from *"is it fast"* (learned experts, substituted per seam) and already runs the substitution campaign (§5) in parallel rather than at the end. What was missing is naming where that campaign is *supposed* to arrive: not "as learned as certification allows," but **a per-seam library where the expert running at each seam is the fastest admissible one — which today, given W93's certification asymmetry and the boundary-condition-input gap in [[expert-donor-survey]], means many seams stay classical permanently.** This changes no case study's scope; it changes how a `admit`-but-slow-to-certify substitution should be read against a classical alternative that is already fast and free to certify.

**One insertion: CS-9★ · `seam_placement.py`, between CS-9 and CS-10.** [[interface-transfer-theory]] §9's cut-quality score $\mathcal Q(\Gamma)$ has existed since 2026-08-27, is built from quantities the probe already returns, and has never been used to choose a decomposition — every case study through CS-16 places its windows by hand. CS-9★ needs no new experts and no new physics: search over candidate tilings of CS-9's two-family domain, scored by $\mathcal Q$ under a cost budget, and gate on whether the search matches or beats the ladder's own hand-chosen $N=6$/$N=12$ rungs. This is the one item in [[f1-pathmap-and-end-goal]]'s four end-goal capabilities — multiple experts, multiphysics, multi-formulation decomposition, **automatic adaptive seam placement** — that is not reachable by continuing to build Phase B/C by hand, and it is also the one capability [[prior-art-and-novelty-atlas-0.1]]'s own verdict table cannot find prior art for.

**Two rows opened: W111** (iterations-to-converge at a seam as a function of which expert sits there — untested, and distinct from either expert's per-call cost) **and W112** (elevate $\mathcal Q(\Gamma)$ from a flagged-weak [AI Inference] to a scheduled measurement, which CS-9★ is the vehicle for).

**Nothing else moves.** CS-9 remains next, exactly as §8 already concluded. CS-10 through CS-16 keep their numbers, scope, and gates.

---

# 10. CS-9 is run (2026-09-01)

Full record: [[case-study-thermal-strain-atlas-0.1]]; measurements in [[gap-worklist]] Tier 22; artifact `out/w94/w94.json`.

**The row's gate is met and its amendment refuses, and the refusal is the more useful half.** `ThermoStruct2D` split into a conduction agent and an elasticity agent, both owning *the same cells* — the first co-located graph in this vault. Carried in full the volume term reproduces the monolith's stress **bit for bit**; lagged one macro-step it costs $7.116\times10^{-3}$, first order in $\Delta t$. `PortAmendment`, exercised for the first time (**W32**), **refused** on fields 1 and 4 and turned out to need a seventh.

**What §4's Phase B row asked for, answered.** *"What is the bond for a two-way volumetric coupling?"* — it is $(\Delta T,\ \beta\,\mathrm{tr}\,\dot{\boldsymbol\varepsilon})$, with $G$ forward and $G^{\!\top}$ back, and **it is a bond and not a port**. A port is a bond on an interface of co-dimension $\ge1$ between agents whose free energies *add*; this one has co-dimension $0$ between agents whose free energies do not, with a bilinear cross term measured at $2680\times$ the elastic energy. **W94 closes, W70 is confirmed from the other side, and the port list stays at five.**

## What this changes for Phase C

**The three subsystems that were waiting for a sixth port type are not getting one, and they do not need one.** The instrument the measurement points at is an **operator splitting with a splitting-error bound**, not a transmission condition, and CS-9 supplies the first such number: $7.116\times10^{-3}$ per macro-step of lag, first order, quoted with its lag exactly as $\sigma$ is (**W86**'s discipline). Three consequences, each concrete:

- **CS-15 `tyre_contact.py`** — §4 schedules it *"on CS-9's bond"*. Re-read: on CS-9's *splitting*, with the thermal-to-grip path inside one agent's own step or across a declared co-located split whose splitting error is measured, and **not** through a port. That removes a blocker rather than adding one.
- **CS-11 `brake_thermal.py`** — unchanged in scope, and it now has a sibling number. R9's flux transient was measured at $62.4\times$ smaller than the stale-trace term (**W90**); CS-9's splitting error is the same *shape* of defect on the volumetric axis, and both are first order in the exchange interval.
- **CS-12 `wing_fsi.py`** — hits **W114** first. A quasi-static structural agent is `EMBEDDED` and has **no `split-step` variant** (no time derivative, nothing to sub-step), so `L2/R10` refuses every graph containing one — including a co-located split, where R10's own derivation does not reach. CS-8's advice to start from the exposed-agent plus `ProjectedAssembly` column **does not transfer to a structural agent**, and that should be settled before CS-12 rather than during it.

## And one warning for every remaining case study

`GlobalField` bypasses L3 entirely. Measured on CS-9's identical physics, routing a genuine two-agent coupling through it is **numerically exact** and **improves the envelope stamp** — E3 from `fails` to `holds`, E7 from `fails` to `unchecked`, and four seam-level decertifications (`L1/E3`, `L4/E7/passivity`, `L4/block-share`, `L4/operator-content`) traded for one line saying it is not a composition, against a surface-`MECH` route that gets the answer wrong by $1279\times$. Nothing in a compile says so. Any graph from here that reaches for it must say in the declaration that it is doing so and why; [[port-algebra-atlas-0.1]] §10.4 is the standing statement.

## The next step

**CS-9★, `seam_placement.py`** — §9's insertion, unchanged and now unblocked, since CS-9 supplies the two-family domain it searches over. Then **CS-10 `ground_effect.py`**, §4's Phase B second row, whose scope and gate are untouched by any of the above.

---

# 11. CS-10 is run (2026-09-03)

Full record: [[case-study-ground-effect-atlas-0.1]]; measurements in [[gap-worklist]] Tier 24; artifact `out/w127/w127.json`.

**All three of §4's CS-10 deliverables are met, and the row's own gate passes.** A 2-D wing over a rolling road, six exposed `WindowNS` windows and a `ProjectedAssembly` against a two-line algebraic suspension, with the unsplit tightly-coupled solve as the referent — the *same code path* at a one-window tiling, so the zero-cut control is bitwise. Over $240$ macro-steps the split reproduces the referent's ride-height history to $2.5\times10^{-3}$ at the peak of the transient and $1.3\times10^{-4}$ at the end; $\mathcal R$ closes with the motion term at $4.097\times10^{-3}$ and reads $1.0$ without it; and the gradient is usable.

**1. `InterfaceMotion` is measured, and the economics invert.** Operator drift on the moving seam is $3.77\times10^{-1}$ over $K = 10$, against $2.51\times10^{-2}$ on the same seam held still and Tier 0's $2.4\times10^{-4}$–$3.5\times10^{-3}$ static band — which the fluid–fluid seam beside it stays inside whether the wing moves or not, so the drift belongs to the interface's motion and not to the flow developing around it. A cached $S$ survives **one macro-step** at a $2\%$ staleness tolerance, so re-probing costs $\mathbf{75.0\%}$ of the march for **one seam of eight**, and the moving interface carries $\mathbf{43.4\%}$ of the seam's power exchange, which a static accounting reports as zero. **W30**'s Tier 0 number existed for exactly this comparison and had waited twenty-three tiers for something to compare against.

**2. W97 closes, by the candidate repair §4 names.** §2.2's normative effort measures $19.4\times$ one-sidedness at the exchange interval ($77.4$ at the macro-step — the ratio needs a *cadence*, W86's discipline), and §4.1's conservative co-normal takes it to $0.381$, below one, with $\beta$ up $21\times$ and the informative window $85\times$ wider. **W47 scoped that form out at fluid–fluid seams because the advective term cancels across a shared ring cell; there is no second fluid side at a wing.** So *"no lumped subsystem in the car can be certified"* is retired — and twice over, because a **rigid** lumped partner turns out not to be blind in the first place (rank $1$ against $\dim M = 17$ collapses $\beta$ below the fluid block's own norm).

**3. F5 is measured and §7's row is rewritten.** See the table above; the short version is that the gradient is exact, cheap and confirmed, and the thing that can still send an optimiser the wrong way is the **horizon** rather than the expert.

## What this changes for the rest of the schedule

**§3's four-coupling-kind table loses its fourth blank, and the entry that replaces it is a price rather than a bond.** *Moving / deforming interface* had *"no — `motion_class` is `static` and anything else is refused"*. It still is refused, and correctly: no rule exists, `L2/InterfaceMotion` fires on both ports and `E2` fails. What is new is that the refusal is now **priced** — a drift, a re-probe count, and an unaccounted power fraction — so a rung that needs a moving interface knows what it costs instead of knowing only that it is not allowed. **CS-12 `wing_fsi.py` inherits this directly**: a deflecting wing is a moving interface, and its seam will re-probe every macro-step or carry a stale operator.

**The lumped seam costs $97\times$ the domain cut, and that reorders where effort goes.** On this graph the field-to-lumped exchange lag is worth $2.5\times10^{-3}$ and the whole six-window decomposition $2.6\times10^{-5}$. Every Phase C subsystem is field-plus-lumped — a motor map, a battery, a coolant circuit, a spring — and §2.3's *"lumped algebraic experts with zero fitted parameters ... cost an afternoon"* is right about the *expert* and wrong about the *seam*. The afternoon is the agent; the seam is where the error is.

**And the obvious lumped coupling diverges.** Read as an update, $h \leftarrow h_0 - L/k$ implies a plate velocity of $-5.77\,U_\infty$ in one macro-step and blows up at step 1 — the partitioned-FSI added-mass instability. What is well posed is the `MECH` port's own condition in the port's own flow variable, solved in three Newton steps. **Any Phase C subsystem that reads its lumped agent's constitutive law as an update rather than as one half of an interface equation will meet this**, and the fix is free once the seam is written in the port algebra's variables rather than in the subsystem's.

## The next step

**CS-11, `brake_thermal.py`** — §4's Phase B third row, and the last one before Phase C. It is the only remaining coupling kind without a bound: **W90** measured R9's flux transient at $62.4\times$ smaller than the stale-trace term, which has no rule at all, and CS-10 has just added a second instance of the same shape — a seam whose lag defect is first order in the exchange interval, worth $97\times$ the cut, and with no field on `MeasuredConstants` to declare it in (**W116**, recurring).

**Two things CS-10 hands it.** The lag sweep is the instrument: three lags, an order in $\log_2$, and the crossing looked for rather than assumed — which at a $500{:}1$ clock ratio is the only way the answer means anything. And the warning is **W128**: CS-11 will want to quote a bound and a bound taken from a rollout is a function of its horizon, exactly as this study's gradient is.

---


# 12. CS-11 is run (2026-09-04)

Full record: [[case-study-brake-thermal-atlas-0.1]]; measurements in [[gap-worklist]] Tier 25; artifact `out/w131/w131.json`. Measured 2026-09-03, written 2026-09-04.

**§4's CS-11 row is met on the first branch of its own gate, and Phase B is complete.** A brake disc against its cooling duct — `ThermoStruct2D` conducting through a $2.5\,\mathrm{mm}$ cast-iron wall, backward Euler, `EMBEDDED`, against a low-Mach convecting duct at the **$500{:}1$ the physics gives** — with the **single-rate** solve as the referent, which is the same code path with the exchange interval set to the duct's own clock.

**1. W90 closes with a function, not a value.** $\sigma$ is first order in the exchange interval and bounded by

$$\sigma(\Delta t_{\text{ex}}) \;\le\; s_\Gamma\,\dot\lambda_{\max}\,\Delta t_{\text{ex}} \;+\; C_2\bigl(\dot\lambda_{\max}\,\Delta t_{\text{ex}}\bigr)^2$$

with $s_\Gamma$ and $C_2$ from **one probe** and $\dot\lambda_{\max}$ from the run. It holds at every graded interval and is loose by $1.43\times$ to $2.18\times$ over $400\times$ of interval and $2000\times$ of clock ratio, with nothing fitted to the table it is checked against, and ratio $1$ — the single-rate control — carries a defect of **exactly zero**.

**2. And the clock ratio is not the variable, which is what makes the bound usable.** Pinning the interval and refining the fast agent moves the ratio $8\times$ and $\sigma$ by $\mathbf{1.00016\times}$. A real conjugate seam runs at $10^4$–$10^5$ and no single-rate referent can be built there — that is why W90 had no number for three tiers — so a bound written in the ratio would be uncheckable where it is needed and a bound written in the interval is not. **Scoped**: it holds for a fast agent that sub-steps internally at its own stability limit, and a frozen checkpoint cannot, which is R4's asymmetry restated as an experiment the substitution campaign owns.

**3. W17's $W>1$ is exercised, and R3 admits it here.** A trace carried as a linear waveform built causally from the previous interval is worth $3.6\times$ to $56.6\times$ per interval and $1.84\times$ over a $200$-interval rollout — non-monotonically, because the ratio between a first difference and a second is the trace's *curvature*. **A waveform buys an order, not a factor.**

**4. W128 closes as a definition, and the second measurement is on different physics.** [[end-to-end-architecture-spec]] §0.4 is a third numbered convention, binding on L7, L8 and L9, covering all three temporal parameters that silently move a reported quantity — exchange interval, clock ratio, rollout horizon — and it makes the gradient case a **refusal** rather than a decertification. Measured here on a duct's inlet velocity, $\mathrm dJ/\mathrm dU$ peaks at $+0.186$ and reaches $-2.402$, **changing sign between $N=150$ and $N=200$**, with a closed-form lumped prediction written before the run that its long-horizon limit matches within $10\%$ and whose sign condition — *the disc released colder than the air cooling it* — is a declared property of the release state. $N_{\text{sign}} = 200$ (one thermal time constant); $N_{\text{valid}}(10\%)$ is **not determined** over $1000$ macro-steps. **The sign settles after one time constant and the magnitude does not settle after five.**

## What this changes for the rest of the schedule

**§3's four coupling kinds now all have a number.** Volumetric has CS-9's splitting error, moving-interface has CS-10's price and refusal, field-to-lumped has W97's closure, and multirate has a bound. **Phase B set out to build the bonds an F1 car needs and the honest summary is that it built three prices and one bound** — which is what the phase was for, and it means every Phase C subsystem now knows what its couplings cost rather than only that some of them are refused.

**Three case studies, three coupling kinds, one order.** CS-9's volumetric splitting error, CS-10's field-to-lumped lag and CS-11's multirate lag are all **first order in the exchange lag** ($\log_2$ ratios $1.003$; $1.258$, $1.165$; order column $1.011$–$1.037$). That is now enough instances to plan against: a Phase C subsystem that halves an exchange interval should expect to halve its coupling defect, and one that cannot halve it — R4 — has a waveform and nothing else.

**And what a rollout accumulates is still nobody's constant.** W123 recurs here on its third quantity: the per-interval defect is first order in the lag and the $200$-interval accumulation is order $0.55$–$0.71$. Every constant in this vault is measured on the one-interval quantity, and CS-16's Claim B is a rollout claim.

## The next step

**Phase C, and CS-12 `wing_fsi.py` is the row.** Two things it inherits and one it should settle first:

> **All three are settled, and §13 below is the record.** W114 closed rather than being worked around; the moving-interface refusal is inherited and priced; and the multirate bound is not what this seam needed, because both agents run at one clock.

- **W114 is still the blocker §10 named.** A quasi-static structural agent is `EMBEDDED` with no `split-step` variant, so `L2/R10` refuses every graph containing one. CS-11 met R10's refusal on the *conduction* side and had the escape — a `split-step` mode where the composition layer supplies the implicit part — precisely because conduction has a time derivative to sub-step. **Elasticity does not, and that is the whole of W114.** It should be settled before CS-12 rather than during it.
- **A deflecting wing is a moving interface**, so CS-10's re-probe economics apply unchanged: at a solution-dependent seam a cached $S$ does not survive one exchange.
- **And it is a multirate seam**, so it is the first graph that gets to *use* §4.2's bound rather than produce it — which is the test of whether $s_\Gamma$ and $C_2$ are seam properties in any sense broader than one seam.

---
## See Also

- [[f1-pathmap-and-end-goal]] — the end goal, the two claims, and the 12-rung ladder this page reschedules
- [[gap-worklist]] — the W-rows §6 draws from, with the reasoning and the definitions of done
- [[case-study-wake-array-atlas-0.1]] — the sixth real case study, and the measurements §1 counts
- `atlas/CASE-STUDY-GUIDE.md` — what a case study is, the five things you declare, and the bar a seventh has to clear
- [[expert-library-atlas-0.1]] — the cut rule, and where a classical expert and a learned one sit relative to it
- [[port-algebra-atlas-0.1]] — the five surface bonds, and the amendment procedure CS-9 exercises
- [[incremental-transfer-roadmap]] — the bootstrap strategy the substitution campaign follows
- [[tier0-measurements]] — the measurement record every number in §1 comes from
- [[cs7-scaling-ladder-pickup]] — the self-contained brief for the next build
- [[poc1-retrospective-and-hybrid-roadmap]] — section 9's full argument: the cross-machine timing read, the hybrid-library reframing, and CS-9★


# 13. CS-12 is run (2026-09-04)

Full record: [[case-study-wing-fsi-atlas-0.1]]; measurements in [[gap-worklist]] Tier 26; artifact `out/w136/w136.json`.

**§4's CS-12 row is met, Phase C has its first row, and the blocker §10 and §12 both named is closed rather than worked around.** A 2-D wing section: CS-10's fluid column **unchanged** — six exposed `WindowNS` windows and a `ProjectedAssembly` at the same $\mathrm dx$, the same domain, the same plate — against **quasi-static plane-stress elasticity** from `ThermoStruct2D.solve_mechanical`, meeting at the wetted surface as a co-dimension-1 `MECH` seam, two-way. The referent is the same code path at a one-window tiling with the same structural solve, so the zero-cut control is bitwise.

**1. W114 closes, and the fix is a premise check rather than an exception.** `L2/R10`'s own sentence names a hypothesis — *"the graph decomposes the domain"* — that the rule never tested. It now refuses an `EMBEDDED` agent only when the graph contains **another agent of the same `governing_family`**. A proxy, and conservative. **Four graphs' verdicts move and no measurement in the vault moves with them** — `thermal_strain` `surface-mech`, `brake_thermal` `as-built`, `thermal_seam` `as-built`, `wing_fsi` — while `window_ns` `as-built` keeps its refusal, which is the row that says the narrowing is right. **This is the first classical graph in the vault with an `EMBEDDED` agent that compiles with zero refusals.**

**2. The gate is met, and the seam beats the cut for a third time.** Over $240$ macro-steps against a fixed scale: the seam's lag alone $2.700\times10^{-2}$, the six-window cut alone $1.139\times10^{-4}$ — a factor of $\mathbf{237}$, against CS-10's $97\times$ at a field-to-lumped seam. The lag defect is first order at its peak and on the settled half and *third* order at the last macro-step, because the columns re-converge; the lagged columns cross the referent six times and the pure-cut column not at all, and the crossing was looked for rather than assumed.

**3. The added-mass answer is not the one the row's question presumes.** A quasi-static structure has no mass, so the fluid/structure mass ratio is infinite and is not the variable. **Loose Gauss–Seidel does not diverge anywhere the structural model admits** — eighteen of eighteen $(E^*,\text{lag})$ cells, out to a lag of $32$ macro-steps — and the boundary it would have to cross ($\mu = \rho(S_e^{-1}K_{\text{aero}}) = 1$ at $E^* = 3861$) is one a linear small-strain law cannot reach, because at divergence the equilibrium deflection is unbounded by definition. **What diverges is the update form**, at $\mu = 0.077$, and it is priced: an implied surface velocity of $2.43\,U_\infty$ and a load $120.9\times$ the one being balanced.

## What this changes for the rest of the schedule

**[[general-coupling-scheme]] §4.2's scope widens from field↔lumped to any massless partner.** It was written from CS-10 as a rule about lumped agents; the mechanism is the *partner having no mass*, and a quasi-static field has none. Every Phase C subsystem with a quasi-static or algebraic partner — which is all of them — inherits it, and the repair is the same: write the residual in the port's conjugate variables and solve it there. At a field↔field seam that is an $N$-square SPD Newton system rather than a scalar division, and it costs three fixed steps.

**And the substitution campaign gets its sharpest obstacle yet, at the seam class the car is made of.** **W137**: the fluid's block is $1.44\times10^{-5}$ of the assembled operator at this seam, so **every** replacement of the flow expert passes its certificate whatever it is. W97's repair — the conservative co-normal, which took a field-to-lumped seam from $19.4$ to $0.381$ — buys $190\times$ here and leaves $6.9\times10^{4}$, because the one-sidedness is the **structure's stiffness** rather than an effort convention, uniformly across every mode of the declared interface space. §5's stopping rule says the campaign is declared after CS-12; the honest reading is that it is not yet decided, and that the first thing it has to answer is whether a certificate can be *scaled* to a seam whose two sides differ by four orders.

## The next step

**CS-13, `cooling_loop.py`** — §4's Phase C second row, rung 6: conjugate heat transfer plus a **closed** lumped coolant circuit, the first *cyclic* port graph. Three things it inherits:

- **§4.2 in its widened form.** Every lumped agent in the loop is massless and its constitutive law is one half of an interface equation.
- **W137's question at a THERM seam.** CS-12 measured the disparity at a `MECH` seam between a stiff solid and a soft fluid; a coolant circuit against a wall is the same shape one port type along, and whether the certificate is blind there is a measurement the loop can make cheaply.
- **And the tier's own mechanism.** Two of the three defects CS-12 found are rules reading a field that is not the one their sentence names. A cyclic graph is where path-dependence of message passing becomes real, and it is worth asking *before* the build which rule states a premise about acyclicity that nothing checks.

---

# 14. Where the schedule stands after CS-14, CS-S1 and the checkpoint (2026-09-10)

Full records: [[case-study-cooling-loop-atlas-0.1]] (Tier 36), [[case-study-powertrain-atlas-0.1]] (Tier 37), [[case-study-neural-interface-atlas-0.1]] (Tier 38), [[gap-worklist]] Tier 39, and Tier 40's two pages, [[epsilon-halo-measurement]] and [[substitution-campaign-checkpoint]].

**§13's next step was CS-13. Two more ladder rows and one off-ladder study were built after it, and then the checkpoint §7 scheduled *after CS-12* was stood on.** This section does for them what §8–§13 did for their rows — it records what each changed on this page — and adds one thing none of those had to: a recount of the rungs, because §1's *rung 1 to 2 of 12* stayed on this page after it stopped being true.

## 14.1 The rungs, recounted

| rung | what [[f1-pathmap-and-end-goal]] §3 asks | status | evidence |
|---|---|---|---|
| **0** | scaffold | **built** | M0/M3 green |
| **1** | can frozen experts be coupled at all | **answered, on rung 2's graphs** | §7's F2 row: the frozen checkpoint couples stably at every rung of CS-7 to $110$ macro-steps. The RBC case study named for this rung was never built |
| **2** | field ↔ lumped, with a closed-form expert as a peer | **built** | the wind farm, then the wake array |
| **3** | a second **learned** expert joins a graph | **never built** | no row in §4, and no graph in `atlas/cases` holds two learned experts |
| **4** | does an expert's certificate travel | **measured, qualified** | CS-8: a certificate is the expert's *at a probe state*, so a library carries one per expert per regime |
| **5** | a structural expert, two-way FSI | **built classically, and certified** | CS-12 `wing_fsi`, `admit` since Tier 39 |
| **6** | a closed thermal loop | **built classically** | CS-13 `cooling_loop`: the block's first law closes to $3.6\times10^{-7}$ W against $1787$ W |
| **7** | `ELEC` and `ROT`, energy across domains | **built classically** | CS-14 `powertrain`: $T\omega = \sum_i I^2R_i + V_{oc}I$ to $1.9\times10^{-15}$, breaking in exact proportion when $k_e \ne k_t$ |
| **8** | tyre and contact | **scoped, not started** | W165 |
| **9** | the full-vehicle graph | **blocked** | W171, then W172 |
| **10** | design-parameter gradients | **exercised on a small graph** | CS-10: a reverse-mode adjoint against a central difference to $1.32\times10^{-6}$ — and a sign change with the rollout horizon, now a convention: a reported sensitivity carries its horizon or is refused ([[end-to-end-architecture-spec]] §0.4) |
| **11** | an optimizer in the loop | **exercised on small graphs; its gate not posed** | on PoC 1a's classical column the gradient search needs $17.9\times$ and $48.3\times$ fewer rollouts than CMA-ES to reach $75\%$ of its gain, at $36$ and $75$ design variables; on the frozen column at $36$ the advantage falls to $8.0\times$; PoC 2's is $1.8$–$4.3\times$ in wall-clock (W143). *Recovers a known-good design from a bad start* has not been asked |
| **12** | the agent layer | **a separate programme** | [[f1-pathmap-and-end-goal]] §6 |

## 14.2 What the three built rows changed on this page

**CS-13 and CS-14 checked §2.3's inference** — the note under §2.3 — **and found the compiler blind to loop topology, on two unrelated graphs.** Compiled with the return seam and without it: same verdict, same rule set, same per-agent decisions (W163). Tier 39 closed it as `L5/R13`, which reports the cycle on both graphs and binds the loop gain only for a sweeping accelerator — and the two gains say why no single number could have been the rule: CS-13's coolant loop composes to $0.9238$, a contraction, and CS-14's circuit to exactly $1$, a constraint.

**CS-S1 is off the ladder. It moved the learned expert to where it cannot do harm, and found it cannot pay there.** On the interface as a predictor, Poseidon-T cannot introduce a silent error and needs no certificate — a start that is pure noise converges to the same answer. But a field predictor's whole budget is **one sweep**, and the checkpoint costs $61\times$ that in the admitted arrangement. The lever that does pay was already declared: exposing the elliptic part is worth $149\times$ in coupling sweeps.

**Tier 39 certified the first two graphs on merit, and both are classical**: the front wing — CS-10 and CS-12 assembled — at a fixed shape, and `wing_fsi`. Declared to ride, the front wing is refused at `L2/InterfaceMotion`, which fires on the classical incumbent too: §11's moving-interface price arriving at an assembly, not a substitution result.

## 14.3 The checkpoint, and where each half of the programme stands

§5's middle row and §7's stopping rule are resolved where they stand, in the notes under each. Read together, they separate the programme's two claims further than §2.2's table did:

- **The coupling half** has climbed to rung 7 with every expert classical, holds two certified graphs, and has one refusal checked against a coupling that really diverges: `L2/R10/halo` is right on $11$ of $11$ halos in the one arrangement that diverges, and over-fires on $5$ of $11$ in each of the three that do not — arrangements the compiler cannot tell apart (W174, W175).
- **The foundation-model half** has no learned expert admitted with a nonzero contribution. The three informative admits the certificate returns are within $5\%$ of the null replacement, on a $\beta_{\min}$ window $0.2\%$–$1.1\%$ of $\beta$ wide that nothing in the framework can locate.

**And the binding constraint is not the one nine tiers attacked.** At the compile it is `L2/R10`'s halo. At the certificate it is the missing **reference pair** — no same-class monolith, so no $\tau$, no $\sigma$, no $\varepsilon_{\text{tol}}$ and no $\beta_{\min}$. That is W95, whose worklist row has stood `open` since Tier 18 on the rollout referent and which §6 filed as *closed by classical-first*; the note under §6 says why that filing did not survive.

## 14.4 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W165** — contact is a complementarity condition, not an equation | rung 8, and any seam with a kink in its response | `open`, large. Its own first step, marked **[AI Inference]** on the worklist, is a page — *is an inequality-constrained seam admissible at all* — before any module |
| **W171** — `decomposition` is one field per graph | rung 9's integration graph | `open`, **scoped** in Tier 40: the carrier is the connection, not the agent. Part 1 fits a tier; `partition_of_unity`, `overlap` and `overlap_cells` do not |
| **W172** — the seams that would join the subsystems do not exist | rung 9's $O(K)$ integration cost | `open`, after W171 |
| **W95** — no same-class reference pair | every learned substitution verdict | `open` on the rollout referent; **binding** at the certificate (W177), and not yet re-scoped against it |
| **W109**, **W137**, **W177** — an `admit` does not say what it rests on | the campaign reading an absent expert as a faithful one | `open` |
| **W175** — the halo branch is one variable short of its cause | the attribution of the one validated refusal | `open`, pinned by two defect-asserting tests |

**Not on it**: W176's banded-plus-low-rank reading, an **[AI Inference]** waiting on a second architecture family; and every row on [[gap-worklist]] that is real and blocks nothing above.

## 14.5 The next step

**Tier 41 probes branch (a) with the instrument that already exists** — one bounded-receptive-field donor through Tier 40's scripts unchanged, split-step `WindowNS` re-run as the positive control, and a classical elliptic solve on the donor's own patch, so that *global because of the physics* and *global because of the architecture* can be told apart. It is registered off the ladder as **CS-S2**. Beside it, the substitution certificate learns to report the two ratios Tier 40 computed by hand — $\lVert\Delta\rVert/\lVert S_i\rVert$, and $\lVert S_i\rVert$ against $\beta$ — as reporting, not as a verdict. Rung 8's module and rung 9 are not started.

---

# 15. CS-S2 is run, and the certificate says what an admit rests on (2026-09-10)

Full records: [[case-study-bounded-donor-atlas-0.1]] (CS-S2, off the ladder) and [[gap-worklist]] Tier 41.

**§14.5's next step was branch (a)'s first probe, and it came back negative.** This section records what that and the tier's two smaller results change on this page. No rung moves.

## 15.1 Branch (a), probed once: not compact

The donor was **NeuberNet** — the one bounded donor in [[expert-donor-survey]] with a continuum boundary port — read through [[epsilon-halo-measurement]]'s instrument unchanged, beside linear elasticity solved classically on the donor's own disc, with split-step `WindowNS` re-run as the positive control (exactly zero from $r=14$ again, to the digit). Of §14.5's three outcomes it is the third:

- **Not compact, and less compact than the physics it approximates.** On its $29$-sensor ring, band-truncating the $u_x$ port leaves $0.69$ of the operator outside $r=4$ where the elastic physics leaves $0.14$; poked at one sensor, its response keeps $95$–$139\%$ of itself from $0.5$ to $2\,R_n$ deep where the physics keeps $12$–$17\%$. The classical control is global along the ring too — a Dirichlet-to-Neumann operator is — so *global along the seam* is physics here, and *global into the patch* is the donor's.
- **W95 binds for this donor as well.** NeuberNet is not resolution-fixed and still has no same-class reference pair: it is defined on one disc inside its training ranges.
- **What it admits would be W76's null replacement again.** On its in-plane ports its operator differs from linear elasticity's by $100$–$110\%$ of the classical norm — [[substitution-campaign-checkpoint]]'s signature on Poseidon-T, arriving on a second architecture family. It is faithful along the far fields it was trained on, tension to $8\%$ and torsion to $12\%$ where both loads are signed, and absent along anything else.

**[AI Inference]:** the survey's *bounded* described the donor's **domain**, and what a halo rule or a certificate consumes is a bounded **response**. Branch (a) therefore narrows to architectures whose locality is a property of the operator — MACE's finite cutoff, DeepFlame's point — and both are unmeasured.

Two things CS-S2 found on the way are framework findings rather than donor ones:

- **A convention check has to excite every component it vouches for (W179).** The adapter passed a stress comparison at a tension-only base with its hoop input the wrong way round, because both subjects' hoop shears are identically zero there. At a base carrying torsion the same comparison reads cosine $-1.000$.
- **A learned forward can switch branches where its own classifier changes its call, and a one-sided probe cannot see it (W180).** At a zero-torsion base NeuberNet's torsion derivative taken one way and the other way differ in magnitude by a factor near $50$ — $0.18$ against $8.7$ of the physical response — because its sign network's call changes between the two steps.

## 15.2 The certificate reports its margin; no verdict moves

`SubstitutionCertificate` now reports what [[substitution-campaign-checkpoint]] computed by hand: `decision_margin`, $(\beta-\beta_{\min})-\lVert\Delta\rVert$, closing **W109**; `null_replacement_ratio`, $\lVert\Delta\rVert/\lVert S_i\rVert$; and `block_over_beta`, $\lVert S_i\rVert/\beta$. A test restates the pre-Tier-41 truth table and every verdict of W177's census, so a rule that moved one would have to rewrite it first.

**No verdict-changing rule was written, because the two graphs do not support one threshold.** On `poseidon-t-2x2` real swaps read $\lVert\Delta\rVert/\lVert S_i\rVert = 0.95$–$1.01$ and six of eight sides have a blind band; on `wing_fsi`'s fluid side there is no learned swap to read, and no blind band at the declared effort.

**And W137 reversed.** It said the fluid's certificate at an aero-structure seam is blind by five orders, reading the fluid's **share** of the operator, $1.44\times10^{-5}$. The certificate reads $\lVert S_F\rVert/\beta$, and under the effort the graph declares that is $1.39$: a replacement that ignores its boundary data is refused at every tolerance. W137 closes on its own done-when. The scaled certificate it proposed is not derived, and moves to **W178** at `poseidon-t-2x2`, where its premise holds.

## 15.3 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W165** — contact is a complementarity condition, not an equation | rung 8 | `open`, large; its first step is a page, not a module |
| **W171** — `decomposition` is one field per graph | rung 9's integration graph | `open`, scoped; part 1 did not fit beside CS-S2 — $7$ rule sites, and a decomposition field that must become a set reaching the emitted artifact |
| **W172** — the seams that would join the subsystems do not exist | rung 9's $O(K)$ integration cost | `open`, after W171 |
| **W95** — no same-class reference pair | every learned substitution verdict | `open`; **binding for bounded donors too** (CS-S2), not yet re-scoped against the certificate |
| **W178** — a scaled certificate, $\beta$ restricted to what agent $i$ can move | an informative verdict at a seam with a real blind band | `open`, a derivation |
| **W175** — the halo branch is one variable short of its cause | the attribution of the one validated refusal | `open`, pinned by two defect-asserting tests |

**Closed off it**: W109 (the margin is reported) and W137 (its premise is false at the declared effort). **W177** has its first half met and stays open on W95. **Not on it**: W176 — the second architecture family's far field is recorded and does not decide the row; and W179 and W180, which are real and block nothing above until a learned expert is placed at a seam.

## 15.4 The next step

**Not started, and the choice is the user's.** In line: rung 8's admissibility page, W165's own first step — can an inequality-constrained seam compile at all; W171 part 1, then part 2; W175, after deciding whether W174's 2$\times$2 counts as a second graph; W95, re-scoped against the certificate; W178; and whether to fetch NeuberNet's CC BY 4.0 Zenodo analyses as a plastic-branch referent, or a finite-cutoff donor for branch (a) — both downloads, so both asked before.

---

# 16. Rung 8's admissibility question is answered, and it reshapes the module (2026-09-10)

Full record: [[inequality-seam-admissibility]] (the decision, with one measurement) and [[gap-worklist]] Tier 42.

**§15.4's list put rung 8's admissibility page first and the choice was taken.** W165's own first step — *can an inequality-constrained seam be compiled at all* — is done. This section does for it what §8–§15 did for their rows: what it settles, what moves on this page, the critical path restated, and the next step. **No rung is marked complete.** Rung 8's status line moves and its module changes shape.

## 16.1 It compiles, and that is the defect

**The answer is no, and the reason it looks like yes is that nothing asks.** Three two-agent graphs differing only in the shape of one side's `boundary_response` — linear, kinked at the probe base, kinked behind a gap of $10^{-3}$ — compile to the **same verdict under the same eighteen decisions in the same order**, with the same envelope stamp and $\beta$ spanning $0.13\%$. At the solved interface trace:

| fixture | reported residual | the condition it stands for | ratio |
|---|---|---|---|
| smooth **(control)** | $8.02\times10^{-16}$ | $9.66\times10^{-16}$ | $1.204$ |
| kink-at-base | $4.75\times10^{-16}$ | $4.420\times10^{-2}$ | $9.30\times10^{13}$ |
| kink-at-gap | $5.45\times10^{-16}$ | $4.031\times10^{-2}$ | $7.39\times10^{13}$ |

For an affine seam $S\lambda-\chi$ **is** the port-matching condition; for a kinked one the two part company, and only the first is ever reported. That is the **silent-wrongness class** measured rather than argued, which is what makes the verdict a **refusal** — `L3/C10` on a declared condition class (**W182**) — and not a decertification.

Three things worth carrying off the page:

- **No rule can be widened into it.** Enumerated layer by layer at the code, every rule that reads a declaration reads one that exists and is correct. The equality is written down in exactly one place, C5's docstring, and the check there is that an orientation string is non-empty. **L1, where a declaration would have to live, has no field for it** — which is the whole verdict in one row.
- **The refusal a contact graph gets today is about the wrong property.** `MotionClass.SOLUTION_DEPENDENT` is the only field a contact seam can truthfully set, and it refuses at `L2/InterfaceMotion` — *and it refuses the linear control just as hard*. So the question is not merely unanswered, it is **unasked**: the compile stops at L2 and never reaches L3.
- **L9 is the one layer whose theory survives intact**, because the master bound needs Lipschitz continuity and not differentiability. The top of the error theory is fine and the instrument feeding it is not, which is [[tier0-measurements]]'s shape on a fixture.

## 16.2 What moves on this page

**§14.1's rung-8 row.** *scoped, not started* becomes **first step done; the module reshaped; the rung open**. §14.1 stays as written and this is the count now.

**And the module is a different and much smaller object than §4's Phase C row implies.** W165 carried an `[AI Inference]` that if the answer were a named hole the module's job would change *from coupling a tyre to emitting the measurement the missing rule would constrain*. It is a named hole, so it does. Rung 8's first module is: a declared condition class with `EQUALITY` as the default so every existing graph is byte-identical; `L3/C10`; a `NamedHole` emitting three numbers — the branch gap $\lVert J_+-J_-\rVert/\lVert J_-\rVert$, the active-set switch rate per macro-step, and the violated-condition residual; and a probe that returns **both** one-sided blocks at a declared switch, or declines. **None of that needs a contact expert.**

**What still does.** The passivity pair — a genuinely non-associated Coulomb tangent beside W138's orientation artefact, on one unmodified instrument — is where W165 priced most of the rung's value, and a fixture cannot supply it: the question it answers is whether anything *other than the analyst's prior knowledge* separates a repairable convention error from irreducible physics. That is clause 4 of the page's done-when and rung 8 cannot be finished without it. **The row's budget — more than rungs 6 and 7 together — stands for the rung; it never priced the page.**

**One prediction is confirmed and its own diagnostic refuted**, which is the correction most worth carrying. W165 said the probed block would depend on probe amplitude *"in a way no `epsilon` sweep will settle"*. Over nine decades the smooth control is stable to $2.49\times10^{-10}$, the **kink at the base to $2.33\times10^{-10}$** — nine digits, the control's nine — and only the offset kink drifts, at $1.37\times10^{-1}$. $\mathrm{relu}$ is positively homogeneous, so the one-sided quotient is exactly step-independent, and the sweep returns $J_+$ to fifteen digits while $J_-$ sits $26.1\%$ away unmentioned. **A stable sweep is not evidence of differentiability.** §5's negative had the same shape — *the proposal's own wording names the wrong test* — and the instrument that covers all three cases is a two-sided probe, which is **W180**'s done-when arrived at from the other side.

## 16.3 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W182** — an inequality seam compiles, and nothing can declare otherwise | rung 8's first module, and any seam with a discrete component in its condition | `open`, **medium**, scoped: a default-valued declaration, one L3 condition, one `NamedHole` and three numbers. It is the whole of rung 8 that is buildable without an expert |
| **W165** — rung 8 itself | rung 8's gate | `open`, **large**, and **reshaped**. Its first step is done. Clause 4 of its done-when needs a contact expert nobody here has, and the build repo has none |
| **W171** — `decomposition` is one field per graph | rung 9's integration graph | `open`, scoped in Tier 40, **not started**. Part 1 is $7$ rule sites and $81$ references across $17$ test files; `partition_of_unity`, `overlap` and `overlap_cells` are part 2 |
| **W172** — the seams that would join the subsystems do not exist | rung 9's $O(K)$ integration cost | `open`, after W171 |
| **rung 3** — no graph holds two learned experts | the foundation-model claim's own configuration | `open`, **never built**, and the substitution campaign tests a different arrangement ([[f1-pathmap-and-end-goal]] §3.3). Not attempted in Tier 42 |
| **W95** — no same-class reference pair | every learned substitution verdict | `open`; binding at the certificate and for bounded donors, still not re-scoped against it |
| **W181** — R7 states a hypothesis it does not check | the auditability of every probed block | `open`, small-to-medium, pinned by two defect-asserting tests |
| **W178** — a scaled certificate | an informative verdict at a seam with a real blind band | `open`, a derivation |
| **W175** — the halo branch is one variable short of its cause | the attribution of the one validated refusal | `open`, pinned by two defect-asserting tests |

**Closed off it**: nothing. **Not on it**: W176, W179, W180 — though W180 is now *coupled* to W182 by a shared instrument, and closing either half-closes the other.

## 16.4 The next step

**Not started, and the choice is the user's.** In line, in the order the work actually unblocks:

- **W182, rung 8's first module.** The only piece of rung 8 buildable today, scoped to a default-valued declaration and one refusal, with a byte-identical-artifact test as its own control. It also gives **W180** its instrument.
- **Rung 3** — a second learned expert joining a graph, which no graph has ever held and which the substitution campaign does not test. Poseidon-T and NeuberNet are both local and are different families; `thermal_seam` is the natural template and its coherence is unchecked. **Expect a refusal and treat it as the result**: `L2/R10` has a live reason against one and W95 against both, so the finding would be the first time the framework has been asked this exact question.
- **W171 part 1, then part 2, then W172** — rung 9, in that order and not otherwise.
- Then W175, W95 re-scoped against the certificate, and W178.

**Rung 8's module, rung 3 and rung 9 are each a tier.** Tier 42 did the first of §15.4's list and named the rest as not done.

---

# 17. Rung 3 is built, out of order, and it compiles (2026-09-10)

Full record: [[case-study-learned-pair-atlas-0.1]] (CS-17) and [[gap-worklist]] Tier 43.

**§16.4's second item was taken.** Rung 3 — *a second learned expert joins an existing graph; first true multi-family coupling* — had never been built, has no row in §4, and is the one configuration the substitution campaign structurally cannot reach. It is built now, as **CS-17**, out of numeric order because rung 3 is a back-fill: CS-15 and CS-16 stay reserved for rungs 8 and 9.

## 17.1 The result, and it is not the one that was expected

**The graph compiles.** Poseidon-T against NeuberNet across a `MECH` seam reaches `admit-uncertified` with **zero refusals** — and so does every cell of a $2\times2$ against the classical incumbents on the identical geometry. The brief that commissioned it said to expect a refusal and to treat a refusal as the result; there is none, anywhere, and the reason is a rule this vault narrowed on purpose.

`L2/R10` refuses an agent with an embedded elliptic solve **only when the graph holds another agent of the same `governing_family`** — W114, closed at CS-12 so that multiphysics graphs would stop being refused for a reason that never applied to them. In a two-agent multiphysics graph each agent is the sole one of its family, so R10 admits, explicitly, as `L2/R10/sole-family`. **It admits even when Poseidon-T declares the `embedded` that W93 measured**, and declaring that measured value is *cleaner* than the package default, because W160's undeclared-pressure-solve branch then stops firing as well.

**And the rule still works**: its positive control, run in the same compile on the same two-agent shape with one declaration changed, refuses the moment a second same-family agent is present. So the admit is the premise clearing and **not** a rule being weakened to get one. **W186** is the consequence, and it re-scopes nine tiers of work: *the substitution campaign's blocker is a property of tilings.*

## 17.2 What the second learned expert costs, attributed

| cell | fluid | solid | decerts | extra over CC | $\tau$ | E7 |
|---|---|---|---|---|---|---|
| CC | `WindowNS` | classical patch | 10 | — | **defined** | holds |
| CL | `WindowNS` | **NeuberNet** | 11 | `L4/E7/passivity` | UNDEFINED | **fails** |
| LC | **Poseidon-T** | classical patch | 12 | `L2/R10`, `L4/R2b/W46` | UNDEFINED | holds |
| LL | **Poseidon-T** | **NeuberNet** | 13 | all three | UNDEFINED | **fails** |

**Exactly additive as sets, with empty intersection.** At the level of which rules fire, two learned experts cost the union of what each costs alone and nothing more — and each cost is charged to one expert by a control rather than by inspection. That is rung 3's own result and no single-substitution experiment could produce it. ($\beta$ is *not* additive — $3.2916 \to 0.8878$ for the pair against $2.9227$ and $0.5199$ for the singles — so the rules are a union and the constants they name are not.)

**The sharpest single finding is §6.2's.** A reference pair is a **pair**. With one checkpoint the missing half has a classical peer that still declares `lambda_ref`, so the repair is a measurement somebody could make; with two there is nothing on the seam to appeal to, because NeuberNet is defined on one disc and Poseidon-T is fixed at $128^2$. **That is W95 on both halves of one seam at once**, and it confirms [[f1-pathmap-and-end-goal]] §3.3's `[AI Inference]` in its strongest form.

Two more that travel: **`L4/E7/passivity` fails on NeuberNet and reads exactly zero on the classical patch on the same ring at the same base** — a *third* kind of non-symmetry beside W138's convention artefact and rung 8's non-associated physics (W185). And **a learned expert imports its own expressiveness into every seam it touches**: NeuberNet's measured effective rank on its own port is $3$ against the classical patch's $25$, so $\dim M=\min_i m_i^{\text{eff}}$ puts a $128$-cell fluid face and a $29$-sensor ring through a **three-dimensional** interface space.

## 17.3 What this changes on this page

**§14.1's rung-3 row.** *never built* becomes **built as a compile, not as a run; the gate restated**. §14.1 stays as written and this is the count now.

**And the ladder's currency changes for one row.** §4's Phase A/B/C schedule has no rung-3 row because the revised ladder assumed the substitution campaign would carry the foundation-model claim. CS-17 shows it cannot: the campaign's arrangement always leaves one side able to answer, and the configuration the pathmap actually assigned rung 3 behaves differently in the one way that matters. **A row is not added retroactively** — §4 stays as written — but any future reading of *"the campaign is the foundation-model half"* should carry §17.1's re-scoping beside it.

**The gate is restated, not substituted.** [[f1-pathmap-and-end-goal]] §3's rung-3 gate — *non-regression on rung 2; `THERM` port residual* — has no subject on the only graph rung 3 can be built on: this graph shares nothing with the wind farm, and neither available learned expert has a `THERM` port or can have one. The replacement is three clauses, **G1** met and **G2**, **G3** open, and **rung 3 may not be marked complete on G1 alone**. That is **W183**.

## 17.4 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W182** — an inequality seam compiles, and nothing can declare otherwise | rung 8's first module | `open`, medium, scoped in Tier 42 |
| **W183** — rung 3's gate has no subject | marking rung 3 complete | `open`. G2 needs a seam that is not a fiction, which needs a donor nobody has |
| **W184** — a port cannot say what kind of boundary it is | G2, and `_decomposition_cuts`' proxy | `open`, medium, a schema change two rules want |
| **W171** — `decomposition` is one field per graph | rung 9's integration graph | `open`, scoped in Tier 40, **not started** |
| **W172** — the seams that would join the subsystems do not exist | rung 9's $O(K)$ integration cost | `open`, after W171 |
| **W186** — R10 has no subject in a two-agent multi-family graph | the campaign's reading of its own blocker | `open`, a re-scoping rather than a repair |
| **W95** — no same-class reference pair | every learned substitution verdict | `open`, and CS-17 shows it binding on **both halves at once** |
| **W165** — rung 8 itself | rung 8's gate | `open`, large, reshaped in Tier 42; clause 4 needs an expert nobody here has |
| **W187**, **W185**, **W181**, **W178**, **W175** | attribution and auditability, each real | `open`, small to medium |

**Closed off it**: nothing. **Not on it**: W176, W179, W180 — W180 still coupled to W182 by a shared instrument.

## 17.5 The next step

**Not started, and the choice is the user's.** In line:

- **W182, rung 8's first module** — the only piece of rung 8 buildable without a contact expert, and it gives W180 its instrument.
- **W171 part 1, then part 2, then W172** — rung 9, in that order and not otherwise. Untouched through two tiers now, and it is the longest-standing blocker on this list.
- **W186's re-census** — re-run [[substitution-campaign-checkpoint]]'s verdict count in the two-agent multiphysics arrangement, where `L2/R10` does not bind, and name what is left. Cheap, and it changes how the campaign reads its own nine tiers.
- Then W184, W183's G2, W95 re-scoped against the certificate, W185, W187, W178, W175.

**Rung 8's module, rung 9 and the re-census are each a tier.** Tier 43 did §16.4's second item and named the rest as not done.

---

# 18. Rung 9's blocker is removed, and the carrier had to be declared (2026-09-10)

Full record: [[per-region-decomposition-axis]] and [[gap-worklist]] Tier 44.

**§17.5's second item was taken.** W171 — *`decomposition` is one field per graph, and rung 9's graph does not have one axis* — was scoped at Tier 40, priced at Tier 41 and untouched through two tiers since. Part 1 is built. **W172 was not started**, and §18.4 says so.

## 18.1 The axis is per region, and the seven rules read it

`CaseGraph.decomposition` was one field; a vehicle is a fluid tiling and two circuits at once. Now every seam has an axis, every cut **region** has one, and the rules ask the region their subject is in.

**The three rules W171 is actually about** — `_halo_rule`, `_r12_conservative_assembly`, `_w49_sigma_branch` — each opened with `if <the graph's axis> is not OVERLAPPING: return`, emitting nothing. Measured on a union graph carrying a coolant circuit beside a fluid tiling: **all three said nothing at all about the tiling**. Each now asks which regions are overlapping, scopes to their agents, and **says so when there are none** — `_halo_rule`'s own stated discipline about the agents it excludes, applied to the axis.

**The four that were loud** are scoped where scoping is the right answer and left alone where it is not. `_cut_policy` now emits once per axis *present*, so a union gets `L2/C2` for its tiling **and** `L2/C3` for its circuit rather than one of the two; each cross-point is judged by the axis of the region its agents are in, with a vertex spanning two regions treated as non-overlapping — the direction that can be inspected. `_decide_rung` and `_l5_l7_scheme` stay graph-global **on purpose**: R2's lift to the non-overlapping view is a property of the *interface problem*, which is one problem however many regions a graph has.

**The control is the claim.** Seven constructible graphs, **zero disagreements**: one axis each, equal to the one declared, `D_decomposition` unchanged in the artifact, and the new `D_decomposition_axes` a one-element list containing it. A change that moved an axis on a graph nobody was asking about would be a behaviour change wearing a schema change's name.

## 18.2 And the carrier had to be declared, which is the tier's correction

Tier 40 found the row's `[AI Inference]` false — the agent is the wrong carrier, because a cut is a relation among several agents — and proposed the **connection**, *deriving* the axis from `geometrically_coincident`, measured on four graphs with no exception. **Censused over all eleven constructible graphs, per seam, the derivation does not separate the axis:**

| same family | coincident | declared axes seen |
|---|---|---|
| True | False | `{overlapping}` |
| **True** | **True** | **`{overlapping, non-overlapping}`** |
| **False** | **True** | **`{overlapping, non-overlapping}`** |

Two of three populated cells ambiguous, so no ordering of the clauses rescues it. The breakers are the two the sample did not contain — `window_ns` and `wind_farm_real`, overlapping tilings whose every seam declares coincidence, **and they are right to**: coincidence is a property of the two sides' *discretizations* and the axis is a property of their *domains*. The obvious fallbacks fail on a third: `rocket` is overlapping with no partition of unity and no `overlap_cells`.

So the axis is **declared** — `Connection.cut_axis`, `None` = inherit the graph's — which is what every prior graph means and is why all seven reproduce.

> **This row's carrier has now been proposed twice, measured twice and falsified twice by widening the sample**, and both times the measurement was right on what it was measured on. **W188** turns that into a standing instruction: two graphs is enough to open a *rule* and is not enough to conclude that an existing field *decides* something, because a derivation is a claim about the whole population. The census that falsified this one cost under a second and the proposal had stood for two tiers.

## 18.3 What moves on this page

**§14.1's rung-9 row.** *blocked* becomes **unblocked on the schema; the integration graph still unbuilt**. §14.1 stays as written and this is the count now.

Rung 9 is not built and is not close. What has changed is that the thing that made it *unexpressible* is gone: a graph can now say it is a tiling and a circuit at once, and the rules will read the region rather than one field.

## 18.4 What this tier did not do

**W172 was not started**, and it was W171's dependent. The union built here is **disjoint** — the three seams that would *join* the subsystems do not exist — so nothing re-runs the $O(K)$ count across them, and the caveat stands exactly as written: **the zero-per-pair count is strong evidence about the algebra and no evidence about the integration cost.**

**Part 2 (W189) was not built.** `partition_of_unity`, `overlap` and `overlap_cells` are still one object per graph, and that boundary is now a measurement: with a partition present all three formerly-silent rules speak, so part 1 is sufficient for the *rules*, and what remains is that a partition covering only the tiling must be declared as the whole graph's.

## 18.5 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W172** — the seams that would join the subsystems do not exist | rung 9's $O(K)$ integration cost | `open`, **and its prerequisite is now met** |
| **W189** — `partition_of_unity`, `overlap`, `overlap_cells` are one per graph | rung 9's schema, the rest of it | `open`, medium, W171 part 2 |
| **W182** — an inequality seam compiles, and nothing can declare otherwise | rung 8's first module | `open`, medium, scoped in Tier 42 |
| **W183** — rung 3's gate has no subject | marking rung 3 complete | `open`; G2 needs a seam that is not a fiction |
| **W184** — a port cannot say what kind of boundary it is | rung 3's G2, and `_decomposition_cuts`' proxy | `open`, medium |
| **W186** — R10 has no subject in a two-agent multi-family graph | the campaign's reading of its own blocker | `open`, a re-scoping |
| **W95** — no same-class reference pair | every learned substitution verdict | `open`; CS-17 shows it binding on both halves at once |
| **W165** — rung 8 itself | rung 8's gate | `open`, large; clause 4 needs an expert nobody here has |
| **W188**, **W187**, **W185**, **W181**, **W178**, **W175** | how this vault reasons, and attribution | `open`, small to medium |

**Closed off it**: **W171 part 1**, the first row closed in three tiers.

## 18.6 The next step

**Not started, and the choice is the user's.** In line:

- **W172**, whose prerequisite is now met — build the three joining seams on a real union and re-run the count across them. It is the one row whose answer changes what rung 9 costs.
- **W189**, part 2, if the union is to carry a real tiling rather than a two-window fixture.
- **W182**, rung 8's first module — still the only piece of rung 8 buildable without a contact expert.
- **W186's re-census**, cheap, and it changes how the substitution campaign reads its own nine tiers.
- Then W184, W183's G2, W95, W188's rider, W185, W187, W178, W175.

**W172, W189 and rung 8's module are each a tier.** Tier 44 did §17.5's second item and named the rest as not done.

---

# 19. Rung 9's schema is complete, and the partition is one per overlapping region (2026-09-10)

Full record: [[per-region-assembly]] and [[gap-worklist]] Tier 45.

**§18.6's second item was taken ahead of its first**, and the reason is concrete: W172's honest version needs a union carrying a **real** tiling beside the circuits, and until this tier a real tiling's partition of unity could only be declared as the whole graph's. W189 is closed; **W172 was not started**, and §19.4 says so.

## 19.1 The shape was counted before it was built

The row's open question was whether one partition per region is even the right object. **Counted over all forty constructible graphs before any code changed**: every one of the 22 that carries a partition has exactly one overlapping region, and the only graph with two overlapping regions carries none. So the object is **one partition per OVERLAPPING region** — a sparse map, which is the vehicle's shape — and its key is **declared**, because neither derivation survives the count: in 12 of the 22 the family region holds lumped closures the partition does not blend, and in 2 the partition's subdomain names are no agent's. Both derivations failed on the census before either was built, where Tier 40's failed two tiers after it was proposed — [[gap-worklist]] W188's instruction, working.

## 19.2 The control, and the union

**Forty artifacts byte-identical before and after**, against a repeat floor of forty of forty under two hash seeds. Every scoped rule runs its unchanged body per region and re-issues its decisions through one mechanism, `DecisionRecord.rescope`, so a region-scoped decision is the graph-scoped decision with its region named — and `window_ns` declared both ways reaches the identical rule set.

On `window_ns`'s real four-window tiling beside **both** circuits — fourteen agents, three regions — every assembly and halo decision names the fluid region and **none names a circuit**, the substructuring criterion names each circuit, and the same union declared the old way reaches the same verdict and **names no region at all**. Given two overlapping regions, `rocket` gets every rule twice, once per region.

## 19.3 What moves on this page

**§18.3's rung-9 row.** *Unblocked on the schema; the integration graph still unbuilt* becomes **the schema is complete; the integration graph still unbuilt**. §14.1 and §18.3 stay as written and this is the count now.

## 19.4 What this tier did not do

**W172 was not started.** The union is **disjoint**, so the $O(K)$ count was not re-run across joining seams and the caveat stands exactly as written. What the tier produced for W172 is **input**, measured rather than counted: of the union's graph-global fields the parts disagree on three — `decomposition`, `macro_dt` and `measured` — and two had to be scoped; its one refusal is `powertrain`'s clock and not the assembly; and **one agent's region was re-derived by the union itself** — `powertrain`'s rotor, sole in its own graph and cut in the fluid region beside the tiling (**W192**).

## 19.5 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W172** — the seams that would join the subsystems do not exist | rung 9's $O(K)$ integration cost | `open`; **both prerequisites met**, and a real-tiling union exists to start from |
| **W192** — joining graphs re-derives an agent's region | how W172 counts what adding a subsystem changes | `open`, medium, **new** |
| **W190** — measured constants are one record per graph | any bound on a graph with two overlapping regions | `open`, medium, **new**; guarded where it would be quoted |
| **W193** — a named cross-point has no region | union verdicts on a tiling's cross-points | `open`, small to medium, **new** |
| **W182** — an inequality seam compiles, and nothing can declare otherwise | rung 8's first module | `open`, medium |
| **W183** — rung 3's gate has no subject | marking rung 3 complete | `open` |
| **W184** — a port cannot say what kind of boundary it is | rung 3's G2, and W192's cousin one object down | `open`, medium |
| **W186** — R10 has no subject in a two-agent multi-family graph | the campaign's reading of its own blocker | `open`, a re-scoping |
| **W95** — no same-class reference pair | every learned substitution verdict | `open` |
| **W165** — rung 8 itself | rung 8's gate | `open`, large |
| **W191**, **W188**, **W187**, **W185**, **W181**, **W178**, **W175** | how this vault reasons, one mislabelled partition, and attribution | `open`, small to medium |

**Closed off it**: **W189**, the rest of rung 9's schema.

## 19.6 The next step

**Not started, and the choice is the user's.** In line:

- **W172** — build the three joining seams on the union `scripts/w189_region_assembly.py` already constructs, and re-run the count across them at more than one family count. It is still the one row whose answer changes what rung 9 costs, and it now has W192 to count with it.
- **W186's re-census**, cheap, and it changes how the substitution campaign reads its own nine tiers.
- **W182**, rung 8's first module.
- **W191** and **W193**, each small; W191 moves one artifact on purpose and must say so against a fresh control.
- Then W190, W184, W183's G2, W95, W188's rider, W185, W187, W178, W175.

**W172 and rung 8's module are each a tier.** Tier 45 did §18.6's second item and named the rest as not done.

---

# 20. Rung 9's joining seams exist, and a join is O(1) in declarations but not in state (2026-09-11)

Full record: [[joining-seam-cost]] and [[gap-worklist]] Tier 46.

**§19.6's first item was taken.** W172 — *the seams that would join the subsystems do not exist* — is closed on its own done-when: the joining seams exist on a union that compiles, the count was re-run across them at three, four and five families, and existing structure and adding-a-subsystem cost are stated separately.

## 20.1 What joins what

Checked against the port algebra before anything was built, two of the three joins this page's schedule assumed are **not constructible**: the tiling carries no temperature, so there is no `THERM` bond from the aero side to the cooling loop, and the clamped structure offers the powertrain neither a rotational degree of freedom nor a second loaded face. **The replacements are recorded as decisions**: the radiator core stands in the airflow (`MECH`), and the rotor — the only open ports in the three graphs — stands in the wing's wake (`MECH`), beside the machine's heat conducted into the cooled block (`THERM`).

## 20.2 The count, across the joins

- **Existing structure is unchanged by joining**: 48 ports, 48 prolongations, five scale sets and **zero per-pair declarations** after all three joins, as before them.
- **A join's declaration cost does not grow with the family count** — measured at 3, 4 and 5 — and it is a small constant: two to four ports, one split seam or two re-declared ports, at most one unit reconciliation.
- **It is not "two new ports and nothing else."** One join splits an existing seam, one moves an existing response, one re-declares an open port for its new host.
- **What a join re-derives is not local.** The rotor in the wake moves the powertrain's operating point, and with the heat path present that reaches six ports on agents off its own seams instead of three — at the same agents and families. No rule sees it (**W196**).
- **Adding a subsystem adds exactly its own structure**, in both orders, plus nine graph-level fields reconciled once per union.

## 20.3 What moves on this page

**§19.3's rung-9 row.** *The schema is complete; the integration graph still unbuilt* becomes **the joining seams exist and the joined union compiles; nothing marches it**. And this page's affordability premise — rung $n+1$ costs what rung $n$ cost — is now **measured for declarations and unmeasured for state**.

## 20.4 What this tier did not do

**Nothing marches the joined union**, and rung 9's gate is about a run. The union compiles to `refuse` on `L7/R9` alone and to `admit-uncertified` with its clocks reconciled — and R9 turns out to refuse a union with **no** multirate seam, because its subject is the graph (**W194**). A per-pair form of the rotor's join in two cell measures is admitted by L3 (**W195**), and every actuator down seam in the package reads non-passive, `wake_array` included (**W197**). **Task 3, rung 9's integration graph, was not started.**

## 20.5 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W196** — a join's re-derived state is not local, and no rule reads it | a march of the joined union, and whether rung 9's integration cost is $O(K)$ | `open`, medium, **new** |
| **W194** — R9's subject is the graph, not the multirate seam | any multirate union, joined or not | `open`, medium, **new** |
| **W192** — joining graphs re-derives an agent's region | how a union counts what adding a subsystem changes | `open`, medium, **priced**: a subsystem moves a region, a join does not |
| **W195** — L3 never compares the two sides' measures | any per-pair transfer | `open`, small to medium, **new** |
| **W197** — every actuator down seam reads non-passive | E7 on every rotor graph | `open`, small to medium, **new** |
| **W190** — measured constants are one record per graph | any bound on the union | `open`, medium |
| **W193** — a named cross-point has no region | union verdicts on a tiling's cross-points | `open`, small to medium |
| **W182** — an inequality seam compiles, and nothing can declare otherwise | rung 8's first module | `open`, medium |
| **W183**, **W184**, **W186** | rung 3's gate, a port's kind of boundary, R10's subject | `open` |
| **W95**, **W165** | every learned substitution verdict; rung 8 itself | `open` |
| **W191**, **W188**, **W187**, **W185**, **W181**, **W178**, **W175** | how this vault reasons, one mislabelled partition, and attribution | `open`, small to medium |

**Closed off it**: **W172**, the joining seams.

## 20.6 The next step

**Not started, and the choice is the user's.** In line:

- **Rung 9's integration graph (the brief's third task).** The joined union exists and admits with its clocks reconciled. A march needs a decision this tier exposed and did not make — whether the shared operating point is solved union-wide or declared per subsystem (**W196**) — and rung 9's gate is about composition error at scale, decided by a run, so the gate should be re-read against what a march of this graph can actually produce before it is attempted.
- **W194**, cheap and well-controlled: re-scope R9 to the multirate seam, with the disjoint union and `thermal_seam` as the two controls.
- **W197**'s diagnosis on `wake_array`, and **W195**'s measure check — each small.
- Then W190, W193, W182, W186's re-census, W191, W184, W183's G2, W95, W188's rider, W185, W187, W178, W175.

**Rung 9's march and rung 8's module are each a tier.** Tier 46 did §19.6's first item and named the rest as not done.

---

# 21. Rung 9's gate, restated: neither clause fits the vehicle as written (2026-09-11)

Full record: [[rung9-gate-restated]] and [[gap-worklist]] Tier 47.

**§20.6's first item was taken — and read before it was built.** Rung 9's gate is *"$\mathcal R$ closes; composition error sub-linear in $N$"*, and the graph now exists, so each clause was held against it first.

## 21.1 Why neither clause fits

- **$N$ is not a dial on a vehicle.** The unions three subsystems form sit at $N = 8, 10, 13, 18$; $N = 13$ is two different graphs; every step up adds a subsystem's physics. CS-7's ladder could hold the physics still and this graph cannot — and holding it still is what cut CS-7's classical exponent from $+0.80$ to $+0.14$.
- **$\mathcal R$ cannot be written from the records.** No field carries dissipation, the tiling's domain boundaries are not ports, and the three clocks are in three unrelated unit systems.

## 21.2 What was measured instead

This vault's own practice since CS-9 — each join's receiving subsystem closes its own balance with the join's term in it and fails without it:

- **J2 passes**: the block's first law closes to $3\times10^{-8}$ with the mount term and fails by seven orders without it.
- **J3 fails by exactly a factor of two**: the rotor is still one diameter wide in a host whose face is half that. Re-sized, the balance closes — and the powertrain has **no operating point**, the machine demanding $31\times$ the torque the disk can give (**W199**).
- **J1 carries no power**; its coupling moves the loop, $-1.35$ K for $10\%$ more air.

## 21.3 The gate, proposed

**G1**, $O(1)$ declarations per join — **met**. **G2**, each join's receiver balance with and without its term, over a march — **J2 passes and J3 fails at the release state; no march**. **G3**, per-join composition error against a tightly coupled referent, and whether the joins' errors add — **not built**. The $N$ clause is withdrawn to CS-7, where $N$ is a dial. **Rung 9 is not complete** (**W198**).

## 21.4 What moves on this page

§20.3's rung-9 row becomes **the gate is restated; J3 fails its receiver balance and cannot pass it with the powertrain as sized**. And the schedule's CS-16 line — *graded against CS-7's prediction* — has no subject on the graph CS-16 turned out to be.

## 21.5 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W199** — the rotor keeps its native size, and the machine was sized for it | G2 at J3, and any march | `open`, medium, **new** — a design decision |
| **W201** — no unit of time or length | any march of the union | `open`, medium, **new** |
| **W200** — $\mathcal R$ cannot be assembled from the records | G2 by declaration rather than by hand | `open`, medium, **new** |
| **W198** — the gate itself | marking rung 9 complete | `open`, **new** |
| **W196** — a join's re-derived state is not local | a march, and whether rung 9's integration cost is $O(K)$ | `open`, medium |
| **W194** — R9's subject is the graph, not the multirate seam | any multirate union | `open`, medium |
| **W197**, **W195** | E7 and the power path on rotor seams; a per-pair measure | `open`, small to medium |
| **W190**, **W193**, **W182**, **W183**, **W184**, **W186**, **W95**, **W165** | as §20.5 | `open` |
| **W191**, **W188**, **W187**, **W185**, **W181**, **W178**, **W175** | how this vault reasons, one mislabelled partition, and attribution | `open`, small to medium |

**Closed off it**: nothing. This tier restated a gate and closed no row.

## 21.6 The next step

**Not started, and the choice is the user's.** Rung 9's march cannot begin until two things are decided that are not framework work:

- **W199 and W201 together — which vehicle this is**: the speed and the wing chord that relate the tiling's units to seconds and metres, and a rotor sized for its host with a powertrain sized for that rotor.

In line after that:

- **W194**, cheap: R9 scoped to the multirate seams.
- **W200**: declared dissipation and boundary power, or §6 restated as a per-receiver residual.
- **W197**'s orientation, then the march itself — the devices as body forces (W94), the shared operating point (W196), G2 over the march and G3 against a tight referent.
- Then W190, W193, W182, W186's re-census, W191, W184, W183's G2, W95, W188's rider, W185, W187, W178, W175.

**Rung 9's march is at least a tier, and its two design decisions come first.** Tier 47 did §20.6's first item as a re-read and named the rest as not done.

---

# 22. What a learned expert can contribute, asked as fourteen formulations (2026-09-12)

Full records: [[learned-contribution-kill-tests]] (the divergence and the kill tests), [[defect-correction-learned-operator]] (the formulation, its pre-registered gate and CS-S3) and [[gap-worklist]] Tier 48.

**§21.6's list was not taken.** Rung 9's march waits on two decisions that are not framework work, and the question asked instead is the one §5's campaign and §14.3's recount leave standing: **can a learned expert contribute at all** — with its error bounded, or with a guarantee of another kind. **No rung moves**, and this section records what came back.

## 22.1 The shape of the answer

Fourteen formulations, each dropping one assumption under the W95 chain; kill tests on `wake_array`'s own graphs at two windows, five minutes of compute; **seven dead, two not chosen, three standing and unmeasured, one detector, one converged on**. The one converged on is a slot where a learned map **cannot** change the answer: defect correction with the classical composed macro-step as the target operator, whose limit is the classical settled state whatever the cheap map is, and whose returned state carries the classical residual times a measured constant — a constant §22.2 records holding at two rungs and **failing at the third**.

## 22.2 What was measured

- **In sample, at two windows, the checkpoint looks like a contribution** — shrunk halfway to the null element it reaches the certified state in $25$ classical calls against a detuned copy's $32$, the composition layer without it at $35$, and the cold classical march's $47$.
- **Out of sample, at six windows, it is not** — $99$ classical calls where the gate asked for at most $0.8\times\min(141,112)$; a cost of $314$ classical-equivalents against the cold march's $141$ and against a $2\times$-coarse classical solver's $\mathbf{51.9}$; and **a checkpoint with Gaussian noise on every weight tensor needs $71$, fewer than the clean one**.
- **At twelve windows it is fifth of seven arms by cost.** $222$ classical calls against the gate's $0.8\times\min(267,240)=192$; $258.4$ classical-equivalents against the cold march's $267$ and the coarse classical solver's $\mathbf{38.1}$ — a $7.0\times$ saving the checkpoint does not approach. Ahead of it sit that coarse solver, **the corrupted checkpoint at $175.0$**, the coarse solver shrunk, and the composition layer with no checkpoint in it at all; behind it, the detuned checkpoint and the unshrunk one, which costs **more than not iterating**. The checkpoint's per-call price has fallen below a classical step here ($0.296$) and it still does not pay.
- **The certificate holds at two and six windows and fails at twelve.** On the first two rungs every arm returns a state inside its own $\Theta r$, the checkpoint's bound is $1.14\times$ its measured error, and $50$ further classical steps move the certified answer less than $2\Theta r$ ($30$ steps at twelve, also inside). At twelve windows **two arms of seven land outside their own bound** — the coarse classical solver by $3\%$, the corrupted checkpoint by $14\%$ — because $\Theta$ is read off one approach to the settled state and then used on every other, and the worst arm needs $3.81$ where the cold march reports $3.35$ (**W208**). Theorem 1 is untouched; what fails is the finite-residual bound built on an estimated constant.
- **Two findings belong to no candidate.** CS-S1's $61\times$ cost ratio is an *arrangement*: measured against the monolith in one process, the checkpoint is $3.43$, $2.01$ and $0.296$ of a classical step at two, six and twelve windows — while $2\times$ coarsening reaches $0.032$. And **the classical monolith's settled state is not isolated**: it carries its outflow ring as state, so a march from the checkpoint's settled state converges *exactly*, residual $1.6\times10^{-16}$, a third of the cold distance from the reference (**W203**).

## 22.3 What moves on this page

**§5's campaign table gains a row it did not have.** Its three outcomes are all about certifying the *expert*; this tier measures a fourth arrangement, in which nothing certifies the expert and the answer is certified instead — and the checkpoint still fails, on cost and on Jacobian fidelity rather than on a missing referent.

**§14.3's sentence stands and is now priced.** *The foundation-model half has no learned expert admitted with a nonzero contribution* remains true; what is new is an **entry condition** with numbers in it, and the measurement that the in-sample saving is not attributable to what the checkpoint learned (**W205**).

**CS-S1's ceiling is scoped rather than repealed.** It binds on a *start*, and this tier confirms it one level up for a different reason: a start is flushed out in one transit of an open domain whatever its size, so the checkpoint's settled state costs $75$ classical steps where the freestream costs $68$. It does **not** bind on a cheap map inside the iteration operator, which is why the slot in §22.1 exists at all.

## 22.4 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W204** — the entry condition for a learned cheap operator | any learned contribution in the one certified slot | `open`, medium, **new** |
| **W205** — a corrupted checkpoint is the better cheap operator | attributing the in-sample saving to the checkpoint | `open`, medium, **new** |
| **W203** — boundary data carried as state | every settled-state claim and every residual certificate | `open`, small to medium, **new** |
| **W208** — $\Theta$ from a march under-bounds other approaches | the residual certificate this tier proposed, at scale | `open`, medium, **new** |
| **W206**, **W207** — the answer-side certificate's horizon, and the referent-free tripwire | a runtime guarantee and W13's falsifiable half | `open`, **new** |
| **W199**, **W201**, **W200**, **W198** | rung 9's march and its gate | `open`, as §21.5 |
| **W196**, **W194**, **W197**, **W195** | a march of the joined union | `open`, as §21.5 |
| **W190**, **W193**, **W182**, **W183**, **W184**, **W186**, **W95**, **W165** | as §21.5 | `open` |

## 22.5 The next step

**Not started, and the choice is the user's.** In line:

- **W205 first**, because it decides how to read everything else here: a corruption sweep over seeds and magnitudes at two rungs, against the property of the column that is doing the work.
- **W204 where the condition could hold** — three dimensions or an accelerator, both of which move the same lever that the classical competitor uses.
- **W203**, small and structural: boundary data declared as problem data.
- **W208**, because it is the one clause of this tier's own gate that the theory was supposed to guarantee: $\Theta$ from a power iteration on $I-D\Phi$ rather than from one march, and the three rungs re-judged against it.
- Then rung 9's two design decisions (W199, W201) and its march, as §21.6 lists them.

## 22.6 What this tier did NOT do, named

- **No rung moved, and no rule was written or narrowed.** `atlas/` gains one module and nothing in the compiler changed.
- **No expert was trained**, though one iteration is priced at $0.630$ s and $2000$ of them at $0.35$ h.
- **No trajectory is certified** — the certified object is a settled state — and **no design family was built**, so multifidelity Monte Carlo stays unkilled and unrun.
- **The Jacobians were never assembled**: both mechanism readings are error structure, marked **[AI Inference]**.
- **The certificate that failed at twelve windows was not repaired, and not withdrawn.** $\Theta$ still comes from one classical march; the two arms outside their bound stay outside, and each arm now reports the constant it would have needed beside the constant it was given (**W208**).
- **Nothing was downloaded, no machine was rented, and NeuberNet was not loaded.**

---

# 23. Rung 9 is marched: the seven decisions, the gate, and the clock the union cannot span (2026-09-12)

Full records: [[case-study-vehicle-march-atlas-0.1]] (**CS-18**), [[vehicle-scale-and-sizing]] (the two decisions that are not framework work) and [[gap-worklist]] Tier 49.

**§21.6's list was taken.** Rung 9's march waited on two decisions the vault could not derive — which vehicle this is, and a rotor that fits its host — and this tier makes them, makes the other five, and marches the union: eighteen agents, five families, three joins, three clocks, seven arms at $1200$ fluid macro-steps, with the gate written as numbers before any arm ran.

## 23.1 The two decisions §21.6 was waiting on

**Which vehicle this is (W201).** $L_0 = 0.50$ m — a $0.25$ m front-wing chord — and $U_0 = 50$ m/s. The finding is in the arithmetic: the three clocks `L7/R9` compares as the bare numbers $0.0125$, $0.05$, $0.2$, spread $\mathbf{16}$, are in seconds $1.25\times10^{-4}$, $10^{-3}$, $5\times10^{-2}$, spread $\mathbf{400}$ — and the *ordering* changes, the coolant circuit going from the middle clock to the slowest by $50\times$. Tier 46's "reconciled clocks" equalised numbers, not times. The ratio is $4U_0/L_0$, never below $100$ over any plausible F1 choice, so **the decision moves the number and not the conclusion**.

**A rotor for its host and a machine for the rotor (W199).** The disk takes its host face, $D = 0.5$; the machine is re-sized by a **similarity** rather than a fit — $k_e = k_t \to Ak$, every resistance $\to R/A$ — which returns the control's induction and demand-over-supply ratio to $\le10^{-12}$ with shaft power and machine heat each exactly halved. Both controls land: width $1$ reproduces Tier 46's operating point to every published digit, and width $0.5$ un-resized reproduces Tier 47's $31.15\times$.

## 23.2 What the gate says

**G2 is met over a march on all three joins.** J3's receiving balance closes to $\mathbf{0.0399}$ against a pre-registered $0.075$, with both controls failing; J2's block closes its first law to $\mathbf{4.07\times10^{-9}}$ with the mount term and $\mathbf{0.998}$ without; J1's conductance tracks its declared exponent to $5.57\times10^{-12}$ while its null holds $UA$ at exactly $42.000$ on a **bitwise-identical** fluid.

**G3 is built, and its answer is stronger than the clause could express.** The pre-registered clause compares norms and returns **NEITHER**. The per-component data show J1's and J3's error vectors **anti-parallel at a cosine of $-0.9999$**, summing to the both-lagged arm's error with a relative residual of $\mathbf{0.0039}$: **the joins' composition errors superpose to four parts in a thousand and partially cancel.** A norm cannot see a sign, and the triangle inequality is slack exactly when two errors are anti-parallel — which is the case the clause most wanted to find.

**And they are below the noise.** All three per-join errors sit near $10^{-6}$ against a flow whose own unsteadiness over the settle window has a norm of $1.008\times10^{-3}$. The repeat floor is bitwise **zero**, which makes G4's *ten times the floor* vacuous; against the floor that binds, the joins are **not resolved** at this horizon.

**The negative is the clock.** The block's thermal time constant is $\mathbf{50.1}$ s $= 1002$ coolant steps $= \mathbf{400{,}800}$ fluid macro-steps — about $21$ hours of this host at the lagged price, four days at the tight one. **The union marches and no single march spans the vehicle's own time scales**, and no choice of $L_0$ and $U_0$ removes that.

## 23.3 What moves on this page

**§21.3's row becomes: G2 met over a march on all three joins, G3 built and measured, rung 9 marched and not complete.** §21.5's critical path loses W199, W201 and W194 and gains W209–W213.

**§14.3's sentence is untouched.** Nothing learned was run in this tier; the coupling half climbed, the foundation-model half did not move.

**And §5's campaign gains nothing, because this tier has no expert in it** — which is worth saying explicitly, since rung 9 is the rung [[f1-pathmap-and-end-goal]] §3.2 calls the one that decides everything, and it has now been marched with **every expert classical**.

## 23.4 The critical path now

| row | what it blocks | status |
|---|---|---|
| **W209** — the union that marches and the union that compiles differ | the gate being measured on a graph the compiler has seen | `open`, medium, **new** |
| **W213** — nothing can declare a subsystem quasi-steady | a march that spans the vehicle's clocks | `open`, medium, **new** |
| **W211** — the shared operating point's stiffness is $63.3$ and no rule reads it | any claim that a stale operating point is harmless | `open`, medium, **new** |
| **W212**, **W210** — the anti-parallel errors, and the body force against the claimed power | G3's mechanism, and `wake_array`'s own accounting | `open`, **new** |
| **W198** — rung 9's gate | marking rung 9 complete | `open` |
| **W200**, **W201** as schema holes, **W197**, **W195**, **W196** | declared dissipation, units on a record, the port orientation, the per-pair measure | `open`, as §21.5 |
| **W204**, **W205**, **W203**, **W208**, **W206**, **W207** | the foundation-model half | `open`, as §22.4 |
| **W190**, **W193**, **W182**, **W183**, **W184**, **W186**, **W95**, **W165** | as §22.4 | `open` |

**Closed off it**: **W194** (R9 scoped to the multirate seam, with the two Tier 46 tests rewritten and the diagnosis kept), **W199** and **W201** as decisions.

## 23.5 The next step

- **W205 first**, as §22.5 says — a corruption sweep over seeds and magnitudes at two rungs. It is the cheapest live row in the vault and it decides how to read all of Tier 48.
- **W209**, small and structural: put the re-sized device and machine on the union itself so the graph that marches is the graph that compiles.
- **W213**: a quasi-steady declaration with a verdict, which is what a vehicle-scale march actually needs.
- **W204** where the condition could hold, and **W203**, **W208**, as §22.5 lists them.
- Then G3 above a floor — a second graph, or a longer horizon, or a device count that makes the joins' errors bigger than the flow's own wander.

## 23.6 What this tier did NOT do, named

- **No rung is marked complete.** G3 is one horizon on one graph, below the flow's own noise, with the additivity question well posed over two joins of three.
- **The march is not differentiable through the devices**, so rung 10 gains nothing.
- **`integration_union` and every subsystem module are unchanged**, which is exactly why W209 exists.
- **No dissipation field, no boundary port, no unit on any record**, and `L7/R9` still compares bare numbers.
- **Tasks 2 and 3 of the brief were not started** — the corruption sweep and the coarse competitor across a coupled seam.
- **Nothing was downloaded, no checkpoint was loaded, no machine was rented, NeuberNet was not loaded, and nothing was pushed.**
