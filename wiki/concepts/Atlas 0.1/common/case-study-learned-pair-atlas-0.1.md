# CS-17 — The Learned Pair: rung 3, and the graph compiles

**Type:** Concept page — **case study**, rung 3, built out of numeric order (folder: `Atlas 0.1/common/`)
**Status:** measured 2026-09-10, Tier 43. `scripts/rung3_learned_pair.py`, `out/rung3/rung3.json`, `tests/test_tier43_rung3_learned_pair.py`. Worklist rows **W183**–**W187**.
**Related:** [[f1-pathmap-and-end-goal]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[case-study-bounded-donor-atlas-0.1]] · [[substitution-campaign-checkpoint]] · [[case-study-wing-fsi-atlas-0.1]] · [[case-study-neural-interface-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[expert-donor-survey]] · [[master-error-bound]]

---

# 0. The result, in one paragraph

**Rung 3 is built — the first Atlas graph ever to hold two learned experts of different governing families — and it compiles.** Poseidon-T against NeuberNet across a `MECH` seam reaches `admit-uncertified` with **zero refusals**, and so does every other cell of a $2\times2$ against the classical incumbents on the identical geometry. The reason it is not refused is `L2/R10`'s **W114 premise**: a rule that refuses an embedded elliptic solve only when a *second agent of the same family* is present has no subject in a two-agent multiphysics graph — and R10's positive control, run in the same compile, refuses the moment that premise is met, so this is the premise clearing rather than the rule being absent. **What the second learned expert costs is exactly additive**: the decertification ladder runs $10 \to 11 \to 12 \to 13$ and the pair's extra set is the *union* of what each costs alone, with no interaction term. Three costs are attributed by control: `L4/E7/passivity` **fails** wherever NeuberNet is and holds where the classical patch on the same ring is; `L2/R10` and `L4/R2b/W46` belong to Poseidon; and $\tau$ is **UNDEFINED** in all three cells holding a checkpoint and *defined* in the classical one — because a reference pair is a **pair**, and two learned experts break both halves at once, which is the thing the substitution campaign structurally cannot exhibit.

**And rung 3's gate does not fit the graph.** §9 says why, and proposes the replacement rather than substituting one quietly. **Rung 3 is built as a compile, not as a run.**

---

# 1. What rung 3 asks, and why nine tiers of substitution do not answer it

[[f1-pathmap-and-end-goal]] §3 defines rung 3 as *"a second **learned** expert joins an existing graph; first true multi-family coupling"*, with the note that rungs 3–4 need no new experts. §3.3 records the state plainly: **never built**, no row on the revised ladder, and **no graph in `atlas/cases` holds two learned experts**. It carries an `[AI Inference]` that is the reason this study exists:

> rung 3's absence matters more than its position suggests. The ladder assigned the foundation-model claim a configuration — two learned experts meeting in one graph — that no graph has ever held, and the substitution campaign tests a different one: a learned expert replacing a classical one, seam by seam. A negative from the campaign therefore does not answer rung 3's question, and a positive would not either.

**That inference is confirmed here, and §6.2 is the sharpest form of it.** The campaign swaps one side and leaves the other classical, so the classical side always still declares `lambda_ref` and the seam always still has *half* a reference pair. Only a graph with two learned experts can show what happens when neither half exists — and what happens is not twice as bad, it is **categorically different**: with one checkpoint the missing referent has a peer to appeal to; with two there is nothing on the seam to appeal to at all.

---

# 2. The design decision, recorded

## 2.1 The suggested template is not coherent, and it is decidable before anything is loaded

The natural template looked like [[case-study-wing-fsi-atlas-0.1]]'s sibling `thermal_seam` — a compressible gas against a thermoelastic shell — with Poseidon-T on the gas side and NeuberNet on the shell side. **It is not constructible, and the check costs nothing:**

| | declares | returns | port types |
|---|---|---|---|
| `thermal_seam`'s `cht` seam | — | — | **`THERM`** ($T$ against $q_n/T$) |
| **Poseidon-T** | four `face:MECH` ports | $\nu\,\partial u/\partial n$ | **`MECH` only** |
| **NeuberNet** | `ring:ux`, `ring:uy`, `ring:ut`, `ring:all` | hat-weighted traction | **`MECH` only** |

Neither learned expert carries a temperature field at all. Poseidon-T is incompressible Navier–Stokes; NeuberNet is quasi-static elastoplasticity. **`MECH` is the only bond both can declare**, so the rung-3 graph is a learned *fluid–structure* seam — `wing_fsi`'s shape, not `thermal_seam`'s. This is asserted live in the suite off `poseidon_capabilities`, so a port added to the checkpoint's record fails the test rather than sliding past it.

## 2.2 And the seam is a geometric fiction, declared in the open

NeuberNet's port is a ring of $29$ material sensors at $5R_n$ around a V-notch. **That ring is an internal cut in a solid** — the rest of the body imposes displacement there, which is exactly a domain-decomposition interface. A fluid face is a **wetted surface**. Handing fluid traction to an internal material cut is physically wrong, and no choice of prolongation repairs it.

**It is declared anyway, and the reason is the finding.** `PortDecl.geometry` is **free text**. Nothing in the schema distinguishes a wetted surface from a cut, so no rule can object — which is `R10`'s own recorded problem (*"what would replace it is a structured domain declaration on `Agent`, which does not exist — `Agent.domain` is free text"*) **one object down**, on the port rather than the agent. The graph is built with the fiction stated on the port's `geometry` string, in the artifact, in this section and in the test that pins it, so that the compiler's silence is read as **the schema's silence** and never as the fixture's virtue.

> **What this costs the study, stated so it cannot be forgotten.** Every number below is a number about *the framework's handling of two checkpoints at one seam*. None is a number about fluid–structure physics, and §10 says so again.

## 2.3 Where the code lives, and why it is not in `atlas/cases/`

NeuberNet is **unlicensed and local-only** ([[case-study-bounded-donor-atlas-0.1]] §2.1). CS-S2 set the precedent: the adapter lives in `scripts/`, the weights live outside the repository in `~/.cache/neubernet`, and only measured numbers are committed. This study follows it exactly — `scripts/rung3_learned_pair.py` builds the graph, `atlas/` gains nothing, and the suite passes on a machine that has never had the weights, because **no test loads NeuberNet**.

---

# 3. The $2\times2$

Four graphs, one declared interface space, differing **only** in which side is a checkpoint:

| | solid: **NeuberNet** | solid: classical patch |
|---|---|---|
| fluid: **Poseidon-T** | **LL** — rung 3 | **LC** |
| fluid: `WindowNS` | **CL** | **CC** — the classical incumbent |

The two halves of each pair are genuinely comparable rather than merely similar:

- **The fluid pair sits on the checkpoint's own geometry.** [[case-study-neural-interface-atlas-0.1]] built `WindowNS` windows on Poseidon-T's $128$-cell tiling precisely so a learned and a classical column could be compared — *"same domain, same overlap, same ramp, same four seams, same prolongations, a different expert"*. One window of each, same tile, same state, same $16$-mode face.
- **The solid pair sits on CS-S2's ring**, with the identical $29$ sensors, the identical trace (hat interpolant of the sensor displacements) and the identical Galerkin flux $f_j=\int_\Gamma t\,\phi_j\rho\,\mathrm ds\big/\int_\Gamma\phi_j\rho\,\mathrm ds$, at CS-S2's own tension-elastic base.

**One declared $\dim M$ for every cell, and that is a deliberate control.** The rule is $\dim M=\min_i m_i^{\text{eff}}$, and §7 shows that would give $3$ in the cells holding NeuberNet and $16$ in the others — so a diff would conflate *the solid is learned* with *the interface space is smaller*. Every cell is declared at $\dim M = 3$, the smallest honest value, so the only thing that varies is which expert answers. §7 reports what each cell's own $\dim M$ would have been.

**The seam is non-conforming and says so.** The two sides' effective resolutions differ, so `derive_space` is unavailable — correctly: the conforming special case does not apply, and the 2026-08-27 amendment requires a declared $M$ with one prolongation per side. Both are declared: a $3$-mode real Fourier basis into the $128$ face cells with $G_V=h\,I$, and the same into the $29$ sensors with $G_V=\mathrm{diag}(\text{hat mass})$ — the $\rho$-weighted mass that makes the forced adjoint the Galerkin reduction rather than a hand-written one. `L3/C2/C3/C6` admits it and names the price: *"adds a consistency term $\sigma_{nc}$ to $\sigma$, which is not reduced by iteration and belongs to the coupling rather than to the agents"*.

---

# 4. It compiles, and R10's premise is why

| cell | verdict | refusals | decertifications | envelope E1–E7 | $\beta$ | $\lVert S\rVert_2$ |
|---|---|---|---|---|---|---|
| **LL** | `admit-uncertified` | **0** | 13 | `h h f h u u f` | $0.887797$ | $207.20$ |
| **LC** | `admit-uncertified` | **0** | 12 | `h h f h u u h` | $2.922679$ | $696.47$ |
| **CL** | `admit-uncertified` | **0** | 11 | `h h f h u u f` | $0.519852$ | $207.57$ |
| **CC** | `admit-uncertified` | **0** | 10 | `h h f h u u h` | $3.291599$ | $696.84$ |

**The brief expected a refusal. There is none, anywhere.** The load-bearing decision is this one, emitted on the solid agent:

> `L2/R10/sole-family` **admit** — *"1 agents declare `elliptic_subsolve=embedded` and are the ONLY agent of their `governing_family` in this graph, so the decomposition does not cut them: the elliptic solve runs over exactly the region a monolith would run it over, and R10's premise — that the decomposition changes the operator rather than restricting it — does not hold."*

That is **W114**, closed at CS-12 on 2026-09-04 so that a multiphysics graph would stop being refused for a reason that never applied to it. It does exactly what it was written to do, and the consequence for the substitution campaign is worth saying out loud: **`L2/R10`, the rule that has stood between every learned expert and every seam, has no subject in a two-agent multi-family graph.** The campaign's blocker is a property of *tilings*.

## 4.1 The positive control, in the same compile

*The rule did not fire* is worth nothing without a cell where it does. The same two-agent shape, the same rule, one declaration changed:

| control | `governing_family` of the second agent | verdict | refusing rules |
|---|---|---|---|
| same family | equal to the embedded agent's | **`refuse`** | `L2/R10` |
| different family | different | `admit-uncertified` | none |

Declaration-only, no checkpoint, under a second. **R10 still refuses when its premise is met**, so rung 3's zero refusals are the premise clearing and not the rule being weakened. No rule was touched in this tier.

## 4.2 The elliptic axis, compiled all three ways rather than argued

**W60** records `elliptic_subsolve` as undeclarable for a checkpoint, and `poseidon.build` exposes it as a parameter *"so a caller can compile the same graph both ways and see that the verdict flips"*. **W93 measured it** — `support_reach` $64$ of $128$, with `elliptic_signature` independently agreeing — so `embedded` is the measured value and `none` is the package default.

| Poseidon declares | verdict | refusals | R10's decision |
|---|---|---|---|
| `none` (the default) | `admit-uncertified` | 0 | `L2/R10` **decertifies** — the undeclared-pressure-solve branch (W160) |
| `embedded` (**measured**, W93) | `admit-uncertified` | 0 | R10 says **nothing**: the sole-family premise clears it |
| `unknown` | `admit-uncertified` | 0 | `L2/R10/W60` decertifies |

**Declaring the measured value is not merely harmless — it is cleaner than the default.** The graph that tells the truth about the checkpoint carries *fewer* decertifications than the one that declares `none`, because R10's undeclared branch stops firing and its premise check clears the agent explicitly. That inverts the expectation `poseidon_capabilities`'s own docstring sets, and it is a direct consequence of W114 plus W160 — two independently-motivated narrowings meeting on one graph.

---

# 5. What a second learned expert costs: exactly additive

| cell | decertifications | extra over CC |
|---|---|---|
| **CC** | 10 | — |
| **CL** | 11 | `L4/E7/passivity` |
| **LC** | 12 | `L2/R10`, `L4/R2b/W46` |
| **LL** | 13 | `L2/R10`, `L4/E7/passivity`, `L4/R2b/W46` |

$$\text{extra}(\mathrm{LL}) \;=\; \text{extra}(\mathrm{CL}) \;\cup\; \text{extra}(\mathrm{LC}), \qquad \text{extra}(\mathrm{CL}) \cap \text{extra}(\mathrm{LC}) = \varnothing .$$

**Set equality, not a count.** At the level of *which rules fire*, two learned experts cost the union of what each costs alone and nothing more — there is no interaction term, and each cost is charged to one expert by a control rather than by inspection. That is rung 3's own result and no single-substitution experiment could have produced it.

> **[AI Inference]:** additivity at the *rule* level is not additivity in the *bound*. $\beta$ is not additive — it moves $3.2916 \to 0.8878$ when both experts are learned, and the two single swaps give $2.9227$ and $0.5199$, so the pair is **not** the product or the sum of the singles. The rules that fire are a set union; the constants they name are not. Nothing here measures a composed error, so this is a reading of the table and not a theorem.

---

# 6. The three costs, one at a time

## 6.1 `L4/E7/passivity` fails on the learned solid, and holds on its classical control

| cell | passivity defect | E7 |
|---|---|---|
| LL | $1.062$ | **fails** |
| CL | $0.6932$ | **fails** |
| LC | $0.0$ | holds |
| CC | $0.0$ | holds |

**The classical elastic patch on the identical ring at the identical base has a passivity defect of exactly zero, and NeuberNet does not.** The symmetric part of the assembled seam operator has a negative mode, so the $L\le1$ branch of [[master-error-bound]] is unavailable and $L$ falls back to fitted.

**This is a third case for that rule, and it is worth distinguishing from the two already recorded.** **W138** is non-symmetry as a **convention artefact** — each side reporting traction against its own outward normal — which one declaration took to exactly zero. [[inequality-seam-admissibility]] §4b names the second, non-symmetry as **physics**: Coulomb friction is non-associated and its tangent is genuinely asymmetric. This is neither. It is non-symmetry as an **architecture artefact**: a learned operator that is not passive because nothing in its training made it so, standing beside the classical operator it approximates, which is passive by construction on the same data.

> **[AI Inference]:** if that reading is right, the three cases are separable by their *controls* rather than by the defect: a convention artefact goes to zero under a re-declaration, a physics one does not and its classical peer is also non-symmetric, and an architecture one does not and its classical peer **is** passive. That is a decision procedure and it is one control short — nobody has run a re-declaration sweep on this seam. Not built.

## 6.2 The reference pair is a **pair**, and two learned experts break both halves

| cell | `tau_undefined_seams` |
|---|---|
| **CC** | **`[]`** — a reference pair exists |
| LC, CL, LL | `['fsi']` |

`L1/E3` on the rung-3 seam:

> *"multiphysics seam: FLUID and SOLID declare different governing families (`incompressible-navier-stokes-2d` against `elastic-plastic-notch-2d`) and **carry no `lambda_ref`**, so there is no reference trajectory either side can be scored against and $\tau$ is emitted as UNDEFINED for both. The composition still RUNS correctly — probing never mentions a governing equation, so the transmission layer is robust to this failure. It is the error ATTRIBUTION that has no referent."*

**Why CC is the control that makes this mean something.** `window_ns.window_capabilities` declares `lambda_ref=None`, which is right for a **tiling** — there the referent is the monolith the windows were cut from, not the window. At a **multiphysics** seam the same solver is its own referent, and `wing_fsi` declares exactly that on exactly this expert (*"itself: `reference.WindowNS`, exposed, at this cell"*). Declared here on the same precedent, CC's $\tau$ becomes defined and the three cells holding a checkpoint stay undefined. Without that correction all four cells read UNDEFINED and nothing is attributable.

**And the asymmetry between one checkpoint and two is not a matter of degree.** In LC and CL the missing half is one side's, and the seam has a classical peer that *does* declare a referent — so the repair is a measurement somebody could make. In LL there is nothing on the seam to appeal to: NeuberNet is defined on **one disc** and cannot be its own referent ([[case-study-bounded-donor-atlas-0.1]] §9); Poseidon-T is fixed at $128\times128$ and cannot be evaluated on a larger domain at any resolution. **That is W95 arriving on both halves of one seam simultaneously**, and it is the configuration [[substitution-campaign-checkpoint]] could not reach, because a campaign that swaps one side always leaves the other able to answer.

## 6.3 `L2/R10` and `L4/R2b/W46` belong to the learned fluid

`L4/R2b/W46`, in LL and LC and not in the classical cells:

> *"2 agents declare `time_discretization=unknown`, so whether the probed-DtN interface condition applies to them is undecided — and an undecided premise is not a favourable one. The rung is held at `dirichlet` rather than lifted: lifting it would pick the scheme that is measured $8.9\times$ worse than doing nothing whenever the agents turn out to be explicit."*

Both learned experts declare `UNKNOWN` and both are right to: a one-shot learned map is neither explicit nor implicit. The consequence is that **the transmission rung is held down for the whole graph by either checkpoint**, so the probed-DtN construction the framework is built around is unavailable at a learned seam — and, by W61's default, an undecided premise is not a favourable one. It is a decertification and not a refusal, which is the right verdict and the reason the graph still runs.

---

# 7. The learned expert sets the interface space, and it is small

`effective_resolution` is *"the expert's own measured spectral cutoff, not a chosen number"*. Nobody had measured NeuberNet's, so this study does — one column per sensor, CS-S2's amplitude $a=10^{-2}$, CS-S2's Galerkin flux, the $29\times29$ `ring:ux` block:

| subject | $n$ | $\lVert S\rVert_2$ | leading singular values | rank at $99\%$ | at $99.9\%$ |
|---|---|---|---|---|---|
| **NeuberNet** | 29 | $0.4618$ | $0.4618,\ 0.0682,\ 0.0417,\ 0.0381,\ 0.0105,\ \dots$ | **3** | 4 |
| classical patch | 29 | $0.8328$ | $0.8328,\ 0.8156,\ 0.8107,\ 0.8052,\ 0.7969,\ \dots$ | **25** | 27 |

**A factor of $8.3$ in effective rank, on one ring, at one base, through one instrument.** The classical operator's spectrum is nearly flat — a Dirichlet-to-Neumann map on a disc excites every mode the discretization carries — and the checkpoint's collapses after one. That is CS-S2's *"effective rank $6$–$7$ of $87$ against $72$"* reproduced per port and sharper, and it reaches the compiler directly through the rule $\dim M=\min_i m_i^{\text{eff}}$:

| cell | $\dim M$ each side would give | $\dim M$ the rule sets |
|---|---|---|
| LL, CL | Poseidon $16$ / WindowNS $16$, **NeuberNet 3** | **3** |
| LC, CC | $16$ / classical $25$ | **16** |

**A $128$-cell fluid face meets a $29$-sensor ring through a three-dimensional interface space**, and the number three is the checkpoint's. The rule is not wrong — representing the trace more finely than the coarser expert can respond to *does* buy nothing, which is `L3/C2/C3/C6`'s own sentence — and the consequence is that **a learned expert imports its own expressiveness into every seam it touches**, for every peer, however capable.

---

# 8. The fluid block the certificate cannot see

`L4/block-share`, on the rung-3 seam:

> *"disclosure, not a defect of this compile: the substitution certificate is **BLIND** at this seam. FLUID becomes visible only once $\beta_{\min} > 0.8847$ (assembled $\beta = 0.8878$, block norms FLUID $0.003093$, SOLID $207.2$)."*

| cell | $\lVert S_{\text{fluid}}\rVert$ | $\lVert S_{\text{solid}}\rVert$ | fluid share of $\lVert S\rVert$ |
|---|---|---|---|
| LL | $0.003093$ | $207.20$ | $1.49\times10^{-5}$ |
| LC | $0.003093$ | $696.47$ | $4.44\times10^{-6}$ |
| CL | $0.380484$ | $207.20$ | $1.83\times10^{-3}$ |
| CC | $0.380484$ | $696.47$ | $5.46\times10^{-4}$ |

Two things, and the first is corroboration rather than novelty:

- **$1.49\times10^{-5}$ is W137's number on a third graph.** W137 recorded the fluid's share at `wing_fsi`'s aero-structure seam as $1.44\times10^{-5}$; the same order arrives here on unrelated geometry with a different fluid and a different solid. A fluid–structure `MECH` seam is **the solid's operator**, near enough, whoever computes it — which is W97's one-sidedness promoted from a port-type pairing to a physics pairing.
- **And the learned fluid is $123\times$ more invisible than the classical one.** Poseidon-T's block on the identical geometry, state and face is $0.8\%$ of `WindowNS`'s. That is [[substitution-campaign-checkpoint]]'s null-replacement signature again: the checkpoint barely responds to its boundary data, and at this seam its contribution is below anything a certificate could catch at any tolerance the graph could carry.

---

# 9. The gate rung 3 is being held to

**The pathmap's gate does not fit this graph, and it is restated rather than quietly substituted.** [[f1-pathmap-and-end-goal]] §3's rung-3 row reads:

> Gate: **non-regression on rung 2; `THERM` port residual.**

It was written when rung 3 meant *wind farm + stratification* — a thermal expert joining the wind farm's own graph. Neither clause has a subject here:

- **"non-regression on rung 2"** presumes the second expert joins *rung 2's graph*. This graph shares no agent, no geometry and no port with the wind farm, so there is nothing to regress against. The reason is the donor survey's, not a choice: no second learned expert of a different family exists that couples to the wind farm's flow.
- **"`THERM` port residual"** presumes a `THERM` port. §2.1 shows neither available learned expert has one, and cannot.

## 9.1 The replacement, proposed

Rung 3's *capability* is **a second learned expert joins a graph; first true multi-family coupling**. A gate should test that and nothing else. Three clauses:

| # | clause | status |
|---|---|---|
| **G1** | **The graph compiles and every decision it costs is attributable** — a $2\times2$ against classical incumbents on the identical geometry, so each extra decision is charged to one expert by a control rather than by inspection | **met** (§4, §5) |
| **G2** | **The composition runs, and $\mathcal R(t)$ closes across the learned seam — or the framework says why it cannot** | **not met, and not attempted** (§9.2) |
| **G3** | **At least one of the two experts carries a certificate at this seam** | **not met, structurally** — W95 binds on both halves at once (§6.2), and the fluid's block is below visibility at every tolerance (§8) |

**Rung 3 is therefore built as a compile and not as a run, and it is not complete.** G1 is a real result and is what this page reports. G2 and G3 are open, and G3 is open for a reason no amount of work on this graph removes.

## 9.2 Why no rollout was attempted, stated so it is not mistaken for an omission

The verdict is `admit-uncertified`, so the framework calls the graph `runnable`. It was not run, for two reasons that are about the fixture and not about the framework:

1. **The seam is a geometric fiction** (§2.2). A power residual across it would be a number about the fixture, and this vault's own practice is not to produce those.
2. **The two experts have no common time base.** Poseidon-T's macro-step is a *lead time* into a one-shot map; NeuberNet is quasi-static and has no time at all. The solid's `dt_native` is declared equal to the fluid's, on `wing_fsi`'s precedent for a quasi-static structural agent — which makes `E4` and `R9` pass trivially and is a **declaration of convenience** wherever a real rollout would need it to mean something.

**G2 is what an honest rung-3 graph needs next, and it needs a seam that is not a fiction** — which means a second learned expert whose port is a surface the first one's physics actually touches. [[expert-donor-survey]]'s open question 3 is where that search lives.

---

# 10. What this does NOT claim

- **It is not fluid–structure physics.** §2.2: the seam pairs a fluid face with an internal material cut. Every number here is about the framework's handling of two checkpoints at one seam.
- **It does not claim two learned experts are safe to couple.** It claims the compiler does not object, and separately that $\tau$ is unavailable, E7 fails, the transmission rung is held down and the fluid block is invisible — four reasons a careful reader should not run it.
- **It does not measure composition error.** No $\tau$, no $\sigma$, no $L$, no rollout. §5's additivity is about *which rules fire*, and the `[AI Inference]` under it says explicitly that $\beta$ is not additive.
- **It does not claim `L4/E7/passivity`'s three-case reading is established.** §6.1's separation of convention / physics / architecture artefacts is an `[AI Inference]` and is one control short.
- **It does not settle W60.** §4.2 reports that declaring the measured `embedded` refuses nothing *on this graph*, where the sole-family premise clears it. On a tiling it still refuses, which is W114's whole design.
- **It does not put NeuberNet in the repository.** Weights stay in `~/.cache/neubernet`, nothing but measured numbers is committed, and no test loads them.

---

## See Also

- [[f1-pathmap-and-end-goal]] — §3's rung-3 row and its gate, which §9 restates; §3.3's `[AI Inference]`, which §1 confirms
- [[case-study-ladder-to-f1]] — the revised schedule, which has no rung-3 row
- [[case-study-bounded-donor-atlas-0.1]] — CS-S2: the donor, the ring, the port, and §9's W95 verdict this study inherits
- [[substitution-campaign-checkpoint]] — the configuration this one is not, and the null-replacement signature §8 sees again
- [[case-study-wing-fsi-atlas-0.1]] — W114's closure, the `lambda_ref` precedent §6.2 uses, and W137's block share §8 reproduces
- [[case-study-neural-interface-atlas-0.1]] — the classical column on the checkpoint's own geometry that makes the fluid pair comparable
- [[inequality-seam-admissibility]] — §4b's passivity pair, which §6.1 adds a third case to
- [[port-algebra-atlas-0.1]] — §3.1's `MECH` bond, the only one these two experts share
- [[expert-donor-survey]] — where a second learned expert with a real shared surface would have to come from
- [[gap-worklist]] — W183 (the gate), W184 (a port cannot say what kind of boundary it is), W185 (passivity's third case), W186 (R10 has no subject here), W187 (`lambda_ref` is per-expert and its value is per-graph)
