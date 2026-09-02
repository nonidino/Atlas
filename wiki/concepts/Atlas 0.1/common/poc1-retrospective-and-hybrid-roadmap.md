# PoC 1a, re-read on a second machine — and the ladder revised toward hybrid experts and adaptive seams

**Type:** Core Concept — Retrospective and Program Revision (folder: `Atlas 0.1/common/`)
**Status:** written 2026-09-01, in response to a cross-machine re-run of [[poc1-results-differentiable-design]] and a direct question about the program's next phase.
**Related:** [[atlas-proof-of-concept-1]] · [[poc1-results-differentiable-design]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[tier0-measurements]] · [[expert-donor-survey]] · [[expert-library-atlas-0.1]] · [[interface-transfer-theory]] · [[schwarz-iteration-atlas-0.1]] · [[general-coupling-scheme]] · [[gap-worklist]]

---

## 0. The question this answers

A second machine (a Mac, CPU-only) ran the PoC 1a demo and measured $630\,\mathrm{ms}$ per macro-step on the coupled column against $520\,\mathrm{ms}$ on the classical monolith — the coupled side **slower**, by $10$–$25\%$ depending on the layout, not the $-16.5\%$ measured on the original dev box but the same sign and the same order of magnitude. Four things followed from that observation: what did the PoC actually demonstrate, given that number; can classical solvers stand in the expert library permanently rather than as a stepping stone; should the case-study ladder become hybrid-by-design; and how many more classical rungs before the program earns the right to attempt the genuinely new things — automatic seam placement chief among them, with the F1 car as the reason any of this matters.

---

## 1. What PoC 1a actually demonstrated, read against both machines

**Say what it is not, first, because it is easy to over-read.** The PoC 1a demo has **no neural network in it anywhere.** Its "coupled graph" is `wake_array`'s windows — plural instances of the *same* classical solver, `reference.WindowNS`, stitched by [[case-study-scaling-ladder-atlas-0.1|the projected-assembly fix]] — and its "classical solver" is `scaling_ladder.reference_monolith`, the identical discretization, undivided. Both columns are 2-D incompressible Navier–Stokes, both are the same expert. The comparison is not classical-vs-learned; it is **decomposed-vs-monolithic, holding the expert fixed** — the cost of the cut, isolated.

Read that way, the Mac number is not a disappointing repeat of the dev-box result — it is **the second independent machine to find the same sign**. [[tier0-measurements]] §19.10 measured, on the Windows dev box, that the classical column's per-agent cost *rises* $3.84\times$ from $N=2$ to $N=24$ windows (a memory-hierarchy effect: past $\sim84$k cells the solver's temporaries stop fitting where they did at $6$ windows) while a frozen checkpoint's per-agent cost *falls* $0.32\times$ over the same range, because one batched forward pass amortizes fixed overhead across more windows. The PoC 1a demo runs at $6$ windows on both machines — **inside the range where the classical column has not yet paid for its own decomposition on either host.** Two machines, two different memory hierarchies, the same qualitative finding: **splitting an all-classical NS solve into windows costs more than solving it whole, at this graph size**, and the number that changes between hosts ($-16.5\%$ vs. $-10$ to $-25\%$) is exactly the part [[tier0-measurements]] already flagged as host-specific (**W104**, "neither statement transfers off this host"). The *qualitative* finding now has evidence it is a property of the decomposition, not an artifact of one desktop.

**So, plainly, what the PoC proved and what it did not:**

| | |
|---|---|
| **Proved** | A composed graph of classical windows, reassembled through the frozen [[case-study-scaling-ladder-atlas-0.1|projected-assembly]] internals, **can be differentiated end to end** — every window solve, every seam blend, every actuator disk, every macro-step of a checkpointed rollout — without touching the assembly fix, and a gradient optimizer using that differentiability moves a real design (turbine layout) toward higher power, verified against central finite differences ([[poc1-results-differentiable-design]] §5, OP-6). |
| **Proved, now on two machines** | Decomposing an all-classical solve is not free, and at $6$–$24$ windows with no learned expert in the loop it is a net cost, not a saving. This *is* the architecture's own prediction, not a surprise: the entire argument for decomposition ([[f1-pathmap-and-end-goal]] §1.2, [[case-study-ladder-to-f1]] §2.4) is that it buys **certified reuse and eventual learned-expert speed**, never that cutting a homogeneous domain into pieces is intrinsically faster to solve. |
| **Not attempted** | Multiphysics. Every cell in the demo obeys the same governing equation; the "seams" are computational partitions of one continuum problem, placed where the tiling code put them (128-cell windows, 16-cell halo — [[case-study-wake-array-atlas-0.1]]'s own parameters), not where two different physical regimes meet. This is the user's own critique and it is correct: **a seam that separates identical physics demonstrates the plumbing, not the interesting claim.** |
| **Not attempted** | A learned expert anywhere in the loop, so the demo says nothing about the Poseidon-vs-classical speed question directly — §2 below answers that from measurements that already exist. |
| **Not attempted** | Anything about *where* the seams should go. The window boundaries are a construction parameter, not a decision any algorithm made — §4 below is precisely this gap, and it is already named and unbuilt elsewhere in this vault. |

**One clarification worth being exact about, because "the classical solver is slower" is easy to state backwards.** The Mac numbers say the *coupled column* is slower than the *classical monolith* — i.e., **the classical monolith is the fast one** in this comparison, consistent with the dev-box result. This is not "classical solvers are slow"; it is "decomposing a classical solver is, at this scale, slower than not decomposing it." Keep those two claims separate — §2 shows the opposite ordering (classical faster than a learned expert) holds at small $N$ for an entirely different reason.

**[AI Inference]:** the PoC's honest contribution, restated without the demo framing, is a single sentence: *the reverse-mode graph through the wake-array composition is correct and usable for optimization.* That is worth having — it is the mechanism [[f1-pathmap-and-end-goal]] rung 10 needs (design-parameter gradients) and it is exercised here two rungs early, on the cheapest graph available, which is exactly [[case-study-ladder-to-f1]] §2's own "climb classically, cheaply, first" strategy applied to differentiability rather than to Claims A/B. It should be filed as evidence toward rung 10 and F5, not toward speed.

---

## 2. Classical solvers as experts: already true, and here is how fast

### 2.1 Is it possible?

Yes, and it already is the majority of what is built. [[case-study-ladder-to-f1]] §2.1: *"Four of the six real case studies are classical solvers wearing the expert interface, and the compiler treats them identically to the checkpoint."* `atlas/capability.py`'s `ExpertCapabilities` record asks for a `boundary_response` callable, a `dt_native`, a `governing_family` and a port list — nothing in that contract inspects whether the callable is a finite-volume stencil or a transformer forward pass. `RectangularNS`, `WindowNS`, `ThermoStruct2D` are all first-class experts today, no adapter required.

### 2.2 How fast, compared to a neural operator — the numbers that already exist

[[tier0-measurements]] §19.10 measured exactly this, milliseconds per macro-step **per agent**, `WindowNS` (classical) against Poseidon-T (a $20.8$M-parameter frozen checkpoint), across the same scaling ladder PoC 1a's demo sits on the small end of:

| $N$ (windows) | classical (`WindowNS`) | Poseidon-T |
|---|---|---|
| $1$ | $81.5$ | $399.7$ |
| $2$ | $88.8$ | $270.1$ |
| $6$ | $125.5$ | $154.4$ |
| $12$ | $329.1$ | $\mathbf{109.8}$ |
| $24$ | $340.8$ | $\mathbf{86.1}$ |

**There is a crossover between $N=6$ and $N=12$, and it has a mechanism, not just a sign.** At small $N$ the classical solver wins by up to $4.9\times$ ($N=1$) because Poseidon-T pays a large, nearly-fixed per-call overhead that a handful of windows cannot amortize. At large $N$ the ordering **inverts** — Poseidon-T wins by up to $3.96\times$ ($N=24$) — because one batched forward pass spreads that fixed cost over more windows while the classical solver's per-agent cost is *rising* over the same range, from a memory-hierarchy effect (§1). **Neither number is a property of "neural nets" or "classical solvers" in general — one is fixed-overhead amortization under batching, the other is cache behavior at a specific problem size on a specific host — and W104 already says so:** neither curve is asserted to transfer off the machine it was measured on. What should transfer is the *shape*: a crossover exists, its location is set by (call overhead, batchability) vs (per-window solve cost, cache footprint), and it should be expected to move with hardware, batch implementation, and solver.

**PoC 1a's demo sits at $N=6$, inside the region where classical wins in this table** — one more reason the Mac's $10$–$25\%$ classical advantage is not a surprising result once the two measurements are read side by side.

### 2.3 The customizability and certifiability argument, which the framework already half-makes

The user's intuition — that classical solvers might be *better* for this formulation, not merely acceptable — has direct support already in the vault, from a different angle than speed:

- **Certifiability is asymmetric, structurally, not by degree.** [[case-study-ladder-to-f1]] §1.1's **W93** result: a neural operator's receptive field is global by construction, so `required_halo()` returns `None` and `L2/R10` **refuses** the decomposition outright for nearly every learned donor in [[expert-donor-survey]]. A classical finite-difference/finite-volume stencil has an *analytically known, exact* halo width — it does not need to be measured, and it cannot silently be wrong the way a probed bound can be. This is not a speed argument; it is that **the entire certification machinery this vault has spent eighteen tiers building is close to free for a classical expert and remains an open research question for a learned one.**
- **Physics-encoding spectrum.** [[expert-library-atlas-0.1]]'s own placement rule: a classical solver sits at **level 5 (hard architectural constraint)** — exact governing equations, exact boundary conditions, exact conservation to machine precision — while every donor in [[expert-donor-survey]] sits at levels 1–2 (data-driven, at best a soft loss). A classical expert can have a boundary condition *added* — literally, a new term in a discretized PDE — where [[expert-donor-survey]] finding (2) records that **almost no pretrained donor accepts a boundary condition as an input at all** (two exceptions out of roughly twenty surveyed, and one is MindSpore-only). "More customizable" is not merely plausible, it is close to the current state of the field as surveyed here.
- **The cost is genuinely deferred, not avoided.** [[case-study-ladder-to-f1]] §2.3 already found the structural expert (`thermostruct2d.py` — conduction plus quasi-static elasticity) sitting in the build repo, graded against closed-form oracles, needing **zero new training data**. The [AI Inference] in that section — that the same likely holds for compressible flow and every lumped subsystem — means the multiphysics rungs (5–9) may be **cheaper classically than the pathmap assumed when it wrote rung 5 as "where the project stops being free."**

**Conclusion:** classical experts are not a stopgap to be substituted away as soon as possible. For any seam where certification, exact conservation, or a boundary-condition channel matters more than raw forward-pass speed, a classical expert may be the *permanently correct* choice, and the substitution campaign ([[case-study-ladder-to-f1]] §5) should be read as "find where a learned expert is cheap **and** admissible," not "replace everything eventually." §5 below makes this the standing architecture rather than a waypoint.

### 2.4 The convergence-rate axis — new, and genuinely unmeasured

The user's third point is sharper than a speed comparison: **not** "which expert is faster per call" but "which expert's *seam* converges in fewer fixed-point iterations." This is a real, distinct quantity and nothing in the vault currently measures it.

The nearest existing result, [[schwarz-iteration-atlas-0.1]], measured the opposite corner: on a **periodic** window the additive Schwarz sweep is provably the identity map (two passes agree bitwise), so the "iteration count" question is degenerate — zero effective iterations are needed because there is nothing to converge. `WindowNS`'s own **Dirichlet** sweep (needed once a real boundary channel exists) costs $14\times$ the periodic one *per iteration*, but that page never asked, and nothing else in the vault asks, **how many iterations of a Schwarz/waveform-relaxation loop a classical expert needs to reach $\varepsilon_{\text{tol}}$ at a seam, compared to how many a frozen learned expert needs at the identical seam.** These could differ for reasons that have nothing to do with per-call cost:

- A classical expert's boundary response is a **local, low-order operator** — its Steklov–Poincaré map $\Lambda_i$ is well-approximated by a few Fourier modes, which is exactly what makes [[probed-dtn-coupling]]'s $m{+}1$-solve probe cheap for it. A tight local operator should need *few* Richardson/Schwarz sweeps to agree with a neighbor, because the two sides are both easy to model well with a low-order boundary condition.
- A learned expert's response is diffuse over its whole (global) receptive field ([[case-study-ladder-to-f1]] §1.1's W93 finding), so a boundary condition change at one edge may propagate through the *entire* window before the next exchange sees a consistent state — plausibly requiring **more** sweeps to reach the same tolerance, independent of how fast any single sweep is.

**[AI Inference]:** if that asymmetry is real, it changes the calculus of §2.3 again: a classical expert might win doubly at a genuinely iterative interface (Dirichlet, non-periodic, multi-way) — cheaper *per* sweep and needing *fewer* sweeps — while a learned expert's forward-pass speed advantage (§2.2, large $N$) would be partially or wholly eaten by needing more of them. This is a hypothesis, not a measurement, and it is answerable with machinery that already exists: [[general-coupling-scheme]]'s accelerator parameter $\mathcal K$ and window parameter $W$ are already compiled fields; what is missing is a run that holds the seam and the tolerance fixed and swaps only the expert, counting iterations to convergence rather than wall-clock. **§5 opens this as W111.**

---

## 3. Toward a hybrid scheme, and whether the ladder needs revising

**The short answer: the ladder does not need new case studies for this, because it was already built classical-first for exactly this reason — it needs its framing corrected from "temporary bridge" to "standing architecture."**

[[case-study-ladder-to-f1]] §2 already splits the program: *"Does composition preserve validity and scale?"* (Claims A/B, classical experts suffice) is set apart from *"Is it fast?"* (learned experts, substituted one certified seam at a time). §5's substitution campaign runs **in parallel from Phase B onward**, and its own three-outcome table already allows for the case that some architectures are certifiable and others are not — it does not assume every classical expert is eventually replaced.

What that framing is missing, and what promotes it from *incremental strategy* to *the answer to the user's question*, is naming the destination correctly. As written, §5 reads as a campaign that *ends* somewhere — implicitly, "as much substitution as the certification results allow." Read against §2.3 and §2.2 above, the correct destination is not "as learned as possible" but:

> **A per-seam library in which the expert running at each seam is the fastest one that is admissible there — which, given today's certification and donor landscape, means many multiphysics seams run classical *permanently*, not provisionally.**

This is not a new theory or a new mechanism. It is the existing `certify_substitution` / capability-record machinery, plus the observation (new in §2.3–2.4) that "admissible and fast" does not converge on "learned" as a rule. Concretely, this reframing changes three things on the existing ladder without renumbering anything:

1. **Phase C's subsystems (CS-12 through CS-15) should be built and left classical by default**, exactly as scheduled, but their entries in [[case-study-ladder-to-f1]] §4 should stop reading as "classical, pending substitution" and start reading as "classical, substitution attempted opportunistically and adopted only where §5's three-outcome table returns a genuine win on the combined (speed × admissibility × iteration-count) criterion" — the third factor being new, from §2.4.
2. **The substitution campaign's report (§5) should carry the W111 iteration-count measurement beside the existing wall-clock and admissibility columns**, so a `admit` verdict with a slow convergent seam is distinguishable from an `admit` verdict with a fast one.
3. **Nothing about Phase A/B/C's ordering or content changes.** CS-9 (the volumetric bond) remains the single highest-leverage next step exactly as [[case-study-ladder-to-f1]] §8 already concludes — it is a prerequisite for the adaptive-seam work below as well as for the FSI/thermal rungs, so it does not move.

---

## 4. How many case studies before the genuinely new work — and the F1 north star

The end goal ([[f1-pathmap-and-end-goal]] §1) names four capabilities the program is ultimately for: **multiple experts, classical or learned, in one graph; multiphysics; domain decomposition in more than one formulation, including on a shared spatial domain; and automatic, adaptive seam placement.** Checked against the current ladder:

| capability | where it is on the ladder today | status |
|---|---|---|
| multiple experts (mixed classical/NN) in one graph | the substitution campaign, run per-seam from Phase B onward | **scheduled**, reframed by §3 above as the standing state rather than a transition |
| multiphysics | CS-9 (volumetric bond) → CS-12 (FSI, new governing family) → CS-13 (cyclic thermal) → CS-14 (`ELEC`/`ROT`) → CS-15 (contact) | **scheduled**, unchanged |
| domain decomposition, several formulations, sometimes on one spatial domain | CS-9 is already exactly this — splitting one PDE, defined on one domain, into two agents that share it (conduction/elasticity via thermal strain) | **one instance built**; the *other* formulations (non-overlapping mortar/FETI-DP, overset/chimera, multi-resolution) are catalogued as **importable** in [[theory-closure-audit]]'s nine-item list but **none has been instantiated as code** |
| automatic, adaptive seam generation | a cut-quality score $\mathcal Q(\Gamma)$ exists ([[interface-transfer-theory]] §9), built from quantities the probe already returns ($\beta$, off-diagonal mass of $\tilde\Lambda^M$) — but it has **never been used to choose a cut**, is explicitly flagged **"the weakest claim on the page"** and **"kept at the status it has"** (an [AI Inference], unpromoted, since the day it was proposed) | **not built at all** — the single largest gap between the current program and its own stated end goal |

**The honest reading of that table is that three of the four capabilities are on a path that reaches them through the existing Phase B/C schedule, and the fourth — adaptive seam placement — is not on the ladder as a case study anywhere, despite being named in [[interface-transfer-theory]] since 2026-08-27 and despite [[prior-art-and-novelty-atlas-0.1]] identifying it (implicitly, by its absence from the prior-art table) as one of the few things here that is not already in the classical DD literature.** Schwarz decomposition (1870), partitioned co-simulation (preCICE), flux continuity at interfaces (cPINN/XPINN, 2020) — all have prior art, per that page's own verdict table. *Where to place the cut, decided by an algorithm rather than by the person writing the tiling code*, does not.

### 4.1 The recommendation: one more case study, inserted after CS-9, before Phase C

**Not a renumbering of the ladder.** [[case-study-ladder-to-f1]] §4's CS-7 through CS-16 stay exactly as scheduled — insert a new case study, provisionally **CS-9★ · `seam_placement.py`**, between CS-9 (the volumetric bond) and CS-10 (the moving interface), for three reasons specific to sequencing:

1. **It needs a nontrivial two-way seam to place, and CS-9 is the first one the ladder builds** (the wake array's seams are all one governing family; there is nothing for a *placement* algorithm to be interesting about until two different physics can sit on either side of a candidate cut).
2. **It needs no new experts and no new physics code** — the same Phase-A property that made CS-7/CS-8 cheap. The $\mathcal Q$ score, $\beta$, and the off-diagonal mass of $\tilde\Lambda^M$ are already computed by the probe machinery CS-7 and CS-8 built; what is missing is a search loop over candidate decompositions, scored by $\mathcal Q$ under a cost budget (e.g. total window count, or total halo cells), rather than a person choosing `N_COL`/`N_ROW` by hand as every case study so far has done.
3. **Its gate is genuinely falsifiable and cheap to state:** given a domain and a library of experts with declared capability records, does an automatic search find a decomposition whose $\mathcal Q$-weighted defect is within some factor of the best hand-chosen tiling this vault has already measured (the CS-7 ladder's own $N=6$ or $N=12$ rung)? If it cannot beat, or come close to, five days of a human choosing window boundaries, the $\mathcal Q$ score itself is the thing that needs revisiting — which is exactly the falsification [[interface-transfer-theory]] §9 says the criterion has never had.

**What this case study would also settle, as a side effect:** [[interface-transfer-theory]] §9's flagged weakness — that $\mathcal Q$'s off-diagonal-mass definition is basis-dependent and the Fourier basis is a locality measure only by assumption — gets its first real stress test, because a placement search will actively seek out the geometries where that assumption is weakest.

### 4.2 The other three "formulations of domain decomposition" — smaller, and can ride along

CS-9★ is not the only unbuilt decomposition formulation. [[theory-closure-audit]]'s IMPORTABLE list already names **primal cross-point DOFs (FETI-DP/BDDC)** and **mortar projection for non-conforming interfaces** as solved-elsewhere-and-absent-here. These do not need their own case studies on the critical path to F1 — [[interface-transfer-theory]] §4 already derives the mortar machinery in the abstract ("the multiplier space and the probe basis are the same space, so mortar costs no extra probes") — but CS-13's cyclic thermal loop (the first case study with a genuine graph cycle) and CS-16's full vehicle (rung 9, $\sim18$ agents) are the natural places to actually exercise a non-conforming interface for the first time, since a real F1 car's cooling loop and its brake/wheel assembly are not conforming meshes by construction. **No new case study number is needed for this — flag it as a requirement on CS-13 and CS-16's build, not as a new rung.**

### 4.3 Answering "how many, then?" directly

**Ten more classical case studies (CS-9 through CS-16, as already scheduled) plus one insertion (CS-9★) — eleven — before the ladder reaches rung 9 (the full-vehicle graph) and the program has, for the first time, all four end-goal capabilities represented at least once: multiphysics (from CS-12 onward), a mixed classical/learned library (from the parallel substitution campaign, mandatory to have attempted by CS-16 per [[case-study-ladder-to-f1]] §7's own stopping rule after CS-12), more than one decomposition formulation (CS-9's shared-domain split plus CS-13/CS-16's non-conforming interfaces), and automatic seam placement (CS-9★, standing alone as the one genuinely novel algorithm in the set).** That is the earliest point at which "start doing the genuinely novel things" is not premature — before it, each additional case study is still retiring a named, dated gap (W94, W97, W90, ...) rather than exploring past the map. After CS-16, per [[case-study-ladder-to-f1]] §7, the program already has a declared checkpoint for admitting the floor outcome (a coupling framework rather than a foundation model) — the same point is the natural one to decide whether to keep climbing toward the literal F1 car or to generalize sideways from what CS-9★ found.

---

## 5. New rows

Opened against [[gap-worklist]]'s numbering (next free: **W111**):

| # | where | what | status |
|---|---|---|---|
| **W111** | new instrumentation on `atlas/scheme.py`'s accelerator ($\mathcal K$) and window ($W$) fields | **Iterations to convergence at a seam has never been measured as a function of which expert sits there.** Hold one seam, one tolerance $\varepsilon_{\text{tol}}$, one accelerator fixed; swap only the expert (classical vs. a frozen checkpoint) and count Schwarz/waveform-relaxation sweeps to convergence, separately from each sweep's wall-clock cost. §2.4's hypothesis — a local classical operator converges in fewer sweeps than a globally-receptive one, independent of per-sweep speed — is the first thing this would test. | `open`, proposed 2026-09-01 |
| **W112** | [[interface-transfer-theory]] §9, `compiler.py`'s L2 layer | **Elevate $\mathcal Q(\Gamma)$'s status from a named-but-unused [AI Inference] to a scheduled measurement.** §9 built the score and immediately flagged it as the page's weakest claim; nothing since has used it to choose a cut, only to describe one already chosen by hand. §4.1 above proposes the case study (CS-9★) that would exercise it for the first time. | `open`, proposed 2026-09-01, blocks the adaptive-seam-generation capability entirely until closed |

---

## 6. What did not change

Worth stating so this page is not mistaken for a bigger revision than it is. **Claims A and B are untouched** — nothing here bears on whether composition preserves validity or scales sub-linearly, both already measured (F1, F2 in [[case-study-ladder-to-f1]] §7). **The Phase A/B/C schedule and every existing case study's number, scope and gate are unchanged**, except for the one insertion in §4.1. **The substitution campaign's mechanism (§5 of the ladder page) is unchanged** — only its stated destination is corrected, from "as learned as certification allows" to "whichever expert is fastest *and* admissible *and* fast-converging, per seam, which today is often classical and may remain so." **PoC 1a's own results page and demo need no correction** — [[poc1-results-differentiable-design]] already states, in its own §0 and §9, that the demo is not a speed claim and that quoted numbers are "as measured by the composed model." What this page adds is the second-machine confirmation and the connective tissue to the ladder's existing F3 measurement, which the demo's own documentation did not have reason to cite.

---

## See Also

- [[atlas-proof-of-concept-1]], [[poc1-results-differentiable-design]] — the PoC this page re-reads
- [[case-study-ladder-to-f1]] — the ladder this page revises in framing (§3) and content (§4.1's insertion)
- [[f1-pathmap-and-end-goal]] — the four end-goal capabilities §4 checks the ladder against
- [[tier0-measurements]] §19.10 — the classical-vs-Poseidon timing table §2.2 reads
- [[expert-donor-survey]] — the boundary-condition-as-input finding behind §2.3
- [[expert-library-atlas-0.1]] — the physics-encoding-spectrum placement §2.3 draws on
- [[interface-transfer-theory]] §9 — $\mathcal Q(\Gamma)$, unused since it was proposed, the subject of W112
- [[schwarz-iteration-atlas-0.1]] — the periodic-sweep-is-the-identity result §2.4 contrasts with the open Dirichlet question
- [[general-coupling-scheme]] — the $\mathcal K$/$W$ fields W111 would instrument
- [[theory-closure-audit]] — the IMPORTABLE decomposition formulations named in §4.2
- [[gap-worklist]] — W111 and W112 filed here
