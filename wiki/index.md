# Physics Foundation Model Wiki — Index

**Last Updated:** 2026-11-18 (lint and compaction: Atlas rows regrouped by purpose)
**Sources Ingested:** 36 documents · **Pages:** 306

> 🤖 An LLM session? Read **[[context-router]]** first: it names the few pages each task needs, so this catalog is rarely needed whole. For Atlas, the current state is **[[00-atlas-current-state]]**.
> 🧭 New here? Read **[[00-start-here]]** for the guided reading path.
> 🔗 Code home: **[github.com/nonidino/Atlas](https://github.com/nonidino/Atlas)** (branch `atlas-0.1`; website [nonidino.github.io/Atlas](https://nonidino.github.io/Atlas/)). The earlier build repository is `nonidino/physics-foundation-model` (private). See [[code-and-papers]] for both + per-paper implementations.
> 🕑 Build history: **[[log]]** (October 2026 onward; earlier months archived verbatim: [[log-2026-04]] · [[log-2026-05]] · [[log-2026-06]] · [[log-2026-07]] · [[log-2026-08]] · [[log-2026-09]]).
> 📚 Long descriptions: rows marked ↗ are shortened here; their full text is in **[[index-digest]]**.

This wiki is organized into two layers, each split into themed subfolders:
- **`summaries/`** — one page per source paper (overview, key equations in LaTeX, results, PFM relevance, AI inferences, cross-links).
- **`concepts/`** — synthesized topic pages (reference explanations, LaTeX math, cross-references, AI inferences).

All page links use bare `[[page-name]]` form and are folder-independent.

---

# Concept Pages (`concepts/`)

## ★ PFM Core — `concepts/00-pfm-core/`
The heart of the project: what a PFM is, the goal, the interface, and the architecture decision.

| Concept | Page | Description |
|---|---|---|
| Physics Foundation Models | [[physics-foundation-models]] | Core goal; challenges; current model landscape; open questions |
| PFM Concept Overview | [[pfm-concept-overview]] | What a PFM is; 7 target capabilities; 8 gaps; gap table |
| PFM Interface Design | [[pfm-interface-design]] | Communicating with a PFM without text; 5-input interface; timescales; regimes |
| PFM Architecture Approaches | [[pfm-architecture-approaches]] | **Key comparison** — 7+ paradigms with pros/cons; hybrid roadmap |
| Possible Architectures | [[possible-architectures]] | 5-architecture benchmark plan; Burgers' benchmark spec |
| **Regime-Conditioned MoE (proposal)** | [[regime-moe-architecture]] | **Competing architecture, not yet reconciled with 1.1**: a survey of about 20 physical regimes; dimensionless numbers drive the MoE gate directly, and smooth and threshold crossovers need separate mechanisms ↗ |
| **Purpose & direction (vision)** | [[pfm-purpose-and-direction]] | **North star** — multi-agent deployment target (moon/mars base + 4 worked examples: wildfire, EV battery runaway, atmospheric reentry, offshore wind farm); reframes the 8 existing gaps against it; 3 new gaps (cross-boundary consistency, composability, verified fallback) |
| ★★★ **Mathematical compendium** *(cross-track reference, new 2026-09-03)* | [[mathematical-compendium]] | **Every piece of formalism the project relies on, stated once in one notation**, each with its symbols, hypotheses and a link to its derivation. The reference to check before quoting any result ↗ |
| **Incremental transfer roadmap** | [[incremental-transfer-roadmap]] | **Build strategy**: bootstrap [[regime-moe-architecture]] from pretrained weights (Poseidon and Walrus) in four stages, frozen wrap, LoRA, structural distillation, shared trunk ↗ |

## ★★ Noether 1.0 (initial PFM model) — `concepts/Noether 1.0/`
*Dormant since July 2026 (Noether 1.0 and 1.1 alike); Atlas is the active track.* **The first concrete model.** One coherent architecture (not the design space): broad multiphysics, direct state prediction, symmetric attention, unified graph tokenizer, inbuilt conservation. The bridge from design layer → build layer. *(Folder renamed from `initial-pfm-model` 2026-07-04; bare links unchanged.)*

| Concept | Page | Description |
|---|---|---|
| ★ Overview / commitments | [[00-initial-model-overview]] | The 5 design commitments and how they resolve into components |
| Unified graph tokenizer | [[graph-tokenizer]] | Nodes = particles OR field patch-centroids; query-compression encoder; resolution-free; continuum–particle bridge |
| Intelligent patching | [[intelligent-patching]] | Adaptive/AMR-style tokenization; small patches where detail is high; refinement tree = supernode hierarchy |
| Multi-head attention | [[multihead-attention]] | Is MHA needed? Heads = basis of physical coupling operators; conservation-preserving combination |
| Normalization scheme | [[normalization-scheme]] | The LayerNorm decision: nondimensionalize + structure-preserving gain; no mean-subtraction in conservative blocks |
| **Full architecture + diagram** | [[initial-model-architecture]] | **End-to-end pipeline**; how Re/Ma enter; inbuilt conservation mechanism; training plan; ablations |
| **Open architectural problems** | [[open-architectural-problems]] | **Critical self-audit** — 25 problems, **all now DECIDED** (2026-07-01) with scored candidate solutions; Tier 1 core, Tier 2 spec gaps, Tier 3 general 2D/3D geometry; the design-completeness ledger |
| **Pipeline contract** | [[pipeline-contract]] | **Binding stage order, per-stage interfaces, latent-vs-physical space split**; three-tier (enforce/regress/generate) message bus; load-bearing ordering invariants |
| **Training curriculum** | [[training-curriculum]] | **Binding loss composition**: 4 dependency-ordered loss groups; within-group uncertainty weighting; staged curriculum; SFA-style conservation discovery; phase-separated diffusion training |
| ★ **Noether 1.0** | [[noether-1.0]] | **The named, sized model** — nano/base/large param configs, full backbone sizing formula ($12Ld^2$), graph/hierarchy/backbone/optimizer hyperparameters, eval protocol, honest inference-cost analysis |
| ★ **Noether 1.0 — RBC (as-built)** | [[noether-1.0-rbc]] | **First multiphysics instantiation, as built**: 2-D Rayleigh–Bénard with a multi-field encoder, a divergence-free stream function and wall BCs; the rollout-divergence cause and fix, and the full results ↗ |
| **Attractor-energy projection** | [[attractor-energy-projection]] | **Driven-dissipative analog of S1.1** — project onto the attractor's known mean-energy shell $E^\*(\mathrm{Ra})$ instead of a conserved total; the rollout-stability fix for the residual slow energy drift (RBC, build-validated); div-free-preserving soft relaxation |
| **Smoke-test bring-up** | [[smoke-test-bringup]] | **First training batch, corpus expansion ladder, implementation-readiness checklist** — Batch 0 (expressivity sanity), G0/Phase-1 first real batches, 5-level corpus ladder, resolved/deferred/open component audit |

## ★★ Noether 1.1 — `concepts/Noether 1.1/`
**The second-generation architecture.** Revised after the 1.0-RBC as-built report exposed a smoothed-near-copy failure mode and a loss of generality. Design thesis: **a pattern recognizer, not a solver** — over-complete representation, separated field streams, learned edges, and conservation the model *discovers* rather than has hard-coded. One page per model portion (intuition → math → physics → constants), hub at the overview.

| Concept | Page | Description |
|---|---|---|
| ★ Architecture overview / hub | [[00-noether-1.1-overview]] | The 1.1 thesis, end-to-end forward pass, per-portion map, design invariants |
| Conditioning & fundamental constants | [[conditioning-and-constants-1.1]] | Unified contract for how *all* physical constants enter — ruler (nondimensionalize) / dial (conditioning token) / arrow (covariant vector channel); range- and symmetry-constants reach into structure |
| Graph tokenizer | [[graph-tokenizer-1.1]] | Resolution-free node encoder with **redundant derived features** (gradients, vorticity); dual smooth/high-freq path; constant-light by design |
| Field token streams | [[field-token-streams-1.1]] | **NEW** — one token per field per site on a shared graph; no summation; field coupling = explicit learned edge types |
| ↳ Field descriptors & conditioned queries | [[field-descriptors-1.1]] | **NEW substep** — learned per-field descriptor $\theta_f$ (type + force affordances) concatenated into each token; **generates simulation-conditioned query banks** $Z_q^f$ (Problem 8 / S8.4; instantiates [[physics-conditioned-query-tokens]]) |
| **Edge generation** | [[edge-generation-1.1]] | **NEW dedicated page** — same-field KNN + cross-field KNN (smaller K) + a **learned long-range edge model**; conservative-geometric vs. non-conservative-learned edge split |
| Backbone | [[backbone-1.1]] | Port-Hamiltonian symmetric+skew core on the typed multigraph; momentum-antisymmetry now a **gateable channel** |
| Decoder | [[decoder-1.1]] | **General increment head is default** (particle-compatible); div-free **not** hard-wired; structure-preserving decode is discovered-gated & optional |
| Discovered conservation | [[discovered-conservation-1.1]] | **NEW** — learned invariants $\{C_k\}$; **threshold-gated-hard + learned-soft** hybrid; local (divergence) discovery first; constraint-type cost hierarchy |
| **Post-decoder diffusion** | [[post-decoder-diffusion-1.1]] | **NEW dedicated page** — generative fill-in of the high-k chaotic residual band the deterministic decode averages away; targets attractor statistics, re-projected onto discovered constraints |
| **Training scheme** | [[training-scheme-1.1]] | **Master spec for what trains in what order**: six module phases P0–P5 and a six-rung physics curriculum from Burgers to MHD; learned edges deferred until a loss demands motion ↗ |

## ★★ Noether 1.1 Implementation — `concepts/Noether 1.1/Noether 1.1 implementation/`
**The executable build layer for Noether 1.1.** Translates the design into an ordered, agent-runnable plan (GUI → data → model), with per-portion spec files and two sized configs. Code home: the `noether-1.1` branch of [github.com/nonidino/physics-foundation-model](https://github.com/nonidino/physics-foundation-model); `noether-1.0` branch is the frozen 1.0 reference.

| Doc | Page | Description |
|---|---|---|
| ★ Master plan | [[00-implementation-plan]] | 3 macro-phases (GUI/data/scaffold → deterministic core → discovery/generation/eval); dependency graph; milestones M0–M7; branch strategy |
| Progress log | [[implementation-log]] | Append-only build tracker + portion status board (distinct from the vault-wide [[log]]) |
| Training runbook | [[vast-ai-training-runbook]] | Copy-pasteable procedure for training Noether 1.1 on a rented vast.ai GPU, phase by phase, with checkpoints synced to GitHub Releases so a box that dies is recoverable |
| Medium config | [[noether-1.1-medium]] | ~44M; $d{=}384,L{=}12$; param budget, hyperparams; **answers "deep stacked-attention like an LLM?" — yes** |
| Large config | [[noether-1.1-large]] | ~327M; $d{=}768,L{=}24$; scaled budget; carries rungs 4–6 (compressible/particles/MHD) |
| Fineness & token density | [[noether-1.1-fineness-and-token-density]] | **Diagnosis + fix** for the "pixelated" output — blocks = the 8×4=32 patches at `patch_domain_size=0.25`; finer default + multi-scale `patch_size_jitter` + inference override; **token count is a runtime dim, not a weight shape → warm-start, no from-scratch restart** |
| Portion specs (0–10) | [[impl-repo-scaffold]], [[impl-gui-2d]], [[impl-data-generation]], [[impl-tokenizer-descriptors]], [[impl-edge-generation]], [[impl-backbone]], [[impl-decoder]], [[impl-training-harness]], [[impl-discovered-conservation]], [[impl-diffusion]], [[impl-eval-benchmarks]] | Standalone agent-executable task specs — objective, interface contract, build steps, acceptance tests — one per implementation portion, linked from the plan |

## ★★ Atlas 0.1 (the current track) — `concepts/Atlas 0.1/`
**Start at [[00-atlas-current-state]]**: what is built, measured, open and decided now, and which pages are history. Atlas splits a simulation into agents joined by typed power ports, an expert per governing family, and a compiler that builds or refuses each coupling. The framework built is classical domain decomposition with a certifying compiler; the learned expert (the chart operator) is designed and tested in Stage 0. Rows are grouped by purpose, not folder (page paths: `common/` unless the subsection says otherwise). Rows marked ↗ have a longer description in [[index-digest]].

### Start here

| Page | Description |
|---|---|
| [[00-atlas-current-state]] | **The current state** (2026-11-18): what exists, the five claims' verdicts, the chart operator's standing, open fronts, rules in force, what is history |
| [[00-atlas-0.1-overview]] | The original thesis (2026-08-08, top level): the manifold-atlas metaphor, the forward pass, 7 design invariants, the contrast with Noether 1.1 |
| [[f1-pathmap-and-end-goal]] | **The end goal**: a coupled, differentiable F1-car model fast enough to search the design space, and the rungs to it ↗ |
| [[case-study-ladder-to-f1]] | The case-study ladder rescheduled after six case studies; the programme sat at rung 1–2 of 12 ↗ |

### The binding framework: specification, ports, implementation

| Page | Description |
|---|---|
| [[end-to-end-architecture-spec]] | **Binding spec**: nine layers in dependency order, from expert declaration to emit ↗ |
| [[port-algebra-atlas-0.1]] | **Binding interface contract**: five effort–flow ports (`MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC`) whose product is power ↗ |
| [[atlas-implementation]] | The spec executed: `atlas/`, the seven-hypothesis envelope, the three-verdict compiler; a new case is declarations only ↗ |
| [[interface-transfer-theory]] | **Binding**: one interface space per seam with a prolongation per side; unifies six of nine importable mechanisms ↗ |
| [[plug-in-composition-theorems]] | **Binding**: closure by the Schur complement, how the capability record composes, substitution, attribution across nesting ↗ |
| [[per-region-decomposition-axis]] | W171 part 1: the decomposition axis made per region ↗ |
| [[per-region-assembly]] | W189 and W171 part 2: one partition of unity per overlapping region ↗ |

### Theory: errors, coupling, closure

| Page | Description |
|---|---|
| [[master-error-bound]] | **The error formulation**: per-step defect = agent infidelity + transmission infidelity + solve incompleteness ↗ |
| [[composition-error-theory]] | Interface agreement alone buys nothing; the worst-agent conjecture holds only under three hypotheses ↗ |
| [[temporal-error-accumulation]] | The rollout bound: the monolith's recursion plus one term; exponential only when the step's Lipschitz constant exceeds one ↗ |
| [[general-coupling-scheme]] | Schwarz generalised to a seven-parameter scheme compiled from declarations ↗ |
| [[probed-dtn-coupling]] | Probe the interface (Steklov–Poincaré) operator directly instead of a Robin ladder frozen experts cannot climb ↗ |
| [[schwarz-iteration-atlas-0.1]] | A frozen expert's field of view enlarged by forward passes; S1 refused as unbuildable ↗ |
| [[symmetry-averaging-atlas-0.1]] | Group averaging makes any operator exactly equivariant, without training ↗ |
| [[control-observability]] | Tier 32: for a globally receptive agent the bound's indicator is 1, so the bound is vacuous; observability measured ↗ |
| [[interaction-horizon]] | Tier 33: the certificate's Π was a 0/1 placeholder for an unmeasured sensitivity ↗ |
| [[inequality-seam-admissibility]] | A contact seam compiles, and that is the defect: it poses a variational inequality the solver does not ↗ |
| [[theory-closure-audit]] | Is the theory closed? No; holes sorted into closed, importable and research ↗ |
| [[generalization-requirements]] | Twenty gaps sorted by which goal each blocks: accurate or generic composition ↗ |
| [[atlas-and-standard-dd-theory]] | Where Atlas sits among the four classical domain-decomposition families (a Toselli & Widlund companion) ↗ |
| [[coupling-cost-and-complexity]] | Coupling cost with exact local solves; a monolithic direct solve wins at these 2-D sizes (`atlas-0.1-proposal/expert-architecture/`) ↗ |

### Tracking, planning, positioning

| Page | Description |
|---|---|
| [[gap-worklist]] | **The tracker**: a status board of every W-item (348 at 2026-10-17) and Tiers 76 onward; grep it for a W-number ↗ |
| [[gap-worklist-tiers-0-37]] | Archive (2026-10-17): Tiers 0–37 word for word; history only |
| [[gap-worklist-tiers-38-75]] | Archive (2026-10-17): Tiers 38–75 word for word; history only |
| [[open-problems-atlas-0.1]] | Problems found by measurement, diagnosed and deliberately deferred (OP-1 on), each with its fix ↗ |
| [[tier0-measurements]] | The measurement record from Tier 0 on, the first numbers from a real expert ↗ |
| [[prior-art-and-novelty-atlas-0.1]] | **Read before claiming novelty**: what is and is not new, checked against sources ↗ |
| [[expert-donor-survey]] | What is downloadable per governing family: licence, resolution, BCs, gradients, a frozen-or-transfer verdict ↗ |
| [[learned-contribution-kill-tests]] | Fourteen formulations of what a learned expert could contribute, and the kill tests that answer most negatively ↗ |
| [[poc1-retrospective-and-hybrid-roadmap]] | PoC 1a on a second machine, and the ladder's hybrid revision ↗ |

### The 2026-08-08 thesis design (history: the learned-model plan before the classical framework)

| Page | Description |
|---|---|
| [[agent-definition-atlas-0.1]] | What counts as an agent; granularity; per-agent conditioning |
| [[graph-tokenizer-atlas-0.1]] | Reuses [[graph-tokenizer-1.1]] per agent, with boundary tokens for edges |
| [[edge-generation-atlas-0.1]] | Declared and RL-based edge instantiation; the rocket's typed edge list |
| [[expert-library-atlas-0.1]] | Experts cut by governing family, not edge label; the rocket's four experts |
| [[unet-hierarchy-atlas-0.1]] | Pool/unpool over the agent graph ([[multiscale-hierarchical-gnn]], [[intelligent-patching]]); MLP vs RL gating |
| [[conservation-as-constraint-atlas-0.1]] | Flux matching at declared edges, against [[discovered-conservation-1.1]] |
| [[global-fields-and-topology-atlas-0.1]] | Gravity as a uniform field; staging as graph-topology mutation (requirement only) |
| [[training-and-bootstrap-atlas-0.1]] | The four-stage transfer plan of [[incremental-transfer-roadmap]] applied to the rocket's experts |

### Case studies: wind farm and PoC 1a (`case-study-wind-farm-wake/` and `common/`)

| Page | Description |
|---|---|
| [[case-study-wind-farm-wake-2d-atlas-0.1]] | The 2-D wind-farm scenario that survives the no-new-experts, no-new-data screen ↗ |
| [[spec-wind-farm-wake-atlas-0.1]] | The binding wind-farm spec: 8 agents, 15 typed edges, equations, coupling rules ↗ |
| [[wind-farm-agent-graph-figure]] | The agent graph as a figure, with both schedules ↗ |
| [[impl-wind-farm-guide]] | Implementation guide, phases W0–W6 with gates ↗ |
| [[wind-farm-implementation-log]] | Append-only build record from 2026-08-19 |
| [[results-w0-w3-wind-farm]] | W0–W3: the go/no-go passed; frozen Poseidon-T keeps momentum closed across a rotor ↗ |
| [[results-w6-w11-wind-farm]] | W6–W11, negative with a cause: the frozen checkpoint inverts the array efficiency ↗ |
| [[results-n-sweep-wind-farm]] | 8 to 26 agents, turbine ratio moves under 1% ↗ |
| [[case-study-wake-array-atlas-0.1]] | CS-6: three turbines over six frozen Poseidon-T windows ↗ |
| [[vast-ai-windfarm-runbook]] | Running the coupled rollouts on a rented vast.ai GPU ↗ |
| [[atlas-proof-of-concept-1]] | PoC 1a spec: layout and yaw by exact gradients through a graph of experts ↗ |
| [[poc1-results-differentiable-design]] | PoC 1a results: 24–26 rollouts to tolerance against CMA-ES's 700 and 928 ↗ |
| [[poc1a-frozen-expert-results]] | PoC 1a, every window Poseidon-T, adjoint through 20.8 M frozen parameters ↗ |
| `atlas/demo/` | PoC 1a's interactive demo (`python scripts/w112_farm_demo.py`) ↗ |
| [[cs7-scaling-ladder-pickup]] | CS-7 build brief ↗ |
| [[case-study-scaling-ladder-atlas-0.1]] | CS-7: one geometry from 1 to 24 coupled windows ↗ |
| [[case-study-reuse-probe-atlas-0.1]] | CS-8: is a certificate the expert's or the state's? ↗ |
| [[decomposition-speed-by-rotor-count]] | W346, Tier 89: parallel decomposition beats the monolith from 5 to 36 rotors; serial never does ↗ |
| [[matched-shrink-and-coarse-competitor]] | W214 and W345, Tier 89: stability along the approach decides; a coarse competitor exists but is not cheap ↗ |

### Case studies: the multiphysics ladder (CS-9 to CS-18, CS-S1 to CS-S5)

| Page | Description |
|---|---|
| [[case-study-thermal-strain-atlas-0.1]] | CS-9: the first co-located split, and the port-extension procedure it failed ↗ |
| [[case-study-seam-placement-atlas-0.1]] | CS-9★: can an algorithm choose where to cut? ↗ |
| [[case-study-ground-effect-atlas-0.1]] | CS-10 (`case-study-ground-effect/`): a moving seam and the first design parameter ↗ |
| [[case-study-brake-thermal-atlas-0.1]] | CS-11 (`case-study-brake-thermal/`): a bound for the multirate lag, a horizon for a gradient ↗ |
| [[case-study-wing-fsi-atlas-0.1]] | CS-12 (`case-study-wing-fsi/`): two-way fluid–structure coupling ↗ |
| [[case-study-rbc-decomposition-atlas-0.1]] | Domain-decomposed Rayleigh–Bénard (`case-study-rbc-decomposition/`), the single-expert control ↗ |
| [[case-study-cooling-loop-atlas-0.1]] | CS-13: the first graph whose ports form a directed cycle ↗ |
| [[case-study-powertrain-atlas-0.1]] | CS-14: the rotor's shaft port to a motor-generator, DC bus and battery ↗ |
| [[case-study-learned-pair-atlas-0.1]] | CS-17: two learned experts of different families across a `MECH` seam ↗ |
| [[joining-seam-cost]] | W172: rung 9's three joining seams built; the union compiles ↗ |
| [[rung9-gate-restated]] | Rung 9's gate re-read and a replacement proposed openly ↗ |
| [[vehicle-scale-and-sizing]] | W201: the tiling's units and the vehicle's three clocks ↗ |
| [[case-study-vehicle-march-atlas-0.1]] | CS-18: eighteen agents, five families, rung 9 marched ↗ |
| [[case-study-neural-interface-atlas-0.1]] | CS-S1: a neural interface predictor, the first scheme study ↗ |
| [[epsilon-halo-measurement]] | Negative: the ε-halo past `L2/R10` does not exist ↗ |
| [[substitution-campaign-checkpoint]] | The substitution campaign's stopping rule: no seam refuses a learned expert ↗ |
| [[case-study-bounded-donor-atlas-0.1]] | CS-S2: NeuberNet, bounded patch, unbounded response; negative ↗ |
| [[defect-correction-learned-operator]] | CS-S3: defect correction, where a learned expert changes the rate, not the answer ↗ |
| [[corrupted-checkpoint-and-jacobian-fidelity]] | CS-S5: Tier 48's number was the best of twelve draws; the mechanism measured ↗ |

### PoC 2: the front wing

| Page | Description |
|---|---|
| [[poc2-frontwing-results]] | Two surface seams live at once; a constrained design search through both ↗ |
| [[poc2-demo-and-novelty]] | The certification demo (`atlas/demo_frontwing/`) and its novelty argument in one screen ↗ |
| [[poc2-novelty-audit]] | The adversarial reading: one narrow claim survives; §3.1 the front wing's own constants (`out/w157/w157.json`) ↗ |

### PoC 3: RaceLab, a car as a graph (Tiers 51–74)

| Page | Description |
|---|---|
| [[case-study-racelab-graph-atlas-0.1]] | CS-19, phase 1: thirteen bodies in fourteen windows ↗ |
| [[case-study-racelab-switch-atlas-0.1]] | CS-20, phase 2: the classical/learned switch ↗ |
| [[poc3-racelab-demo]] | Phase 3: the demo marches one vehicle ↗ |
| [[poc3-racelab-traced-car]] | The car traced as a silhouette ↗ |
| [[poc3-racelab-3d]] | Phase 4: a half-car in 3-D, classical only ↗ |
| [[poc3-racelab-drawn-car]] | Tier 56: the owner's drawn car; the gate refused ↗ |
| [[poc3-racelab-bundle]] | Tier 57, phase 5: the bundle self-tests on Windows and Linux ↗ |
| [[poc3-racelab-car-fixes]] | Tier 58: the drawn car's two faults on five copies ↗ |
| [[poc3-racelab-outlet-and-start]] | Tier 59: outlet opened, start settled, five arms inside every envelope ↗ |
| [[poc3-racelab-body-fitted-grids]] | Tier 60: body-fitted overlapping grids ↗ |
| [[poc3-racelab-overset-flow]] | Tier 61: incompressible flow on them, verified ↗ |
| [[poc3-racelab-car-solids]] | Tier 62: the car as welded solids, 381,698 unknowns ↗ |
| [[poc3-racelab-duct-openings]] | Tier 63: the duct cut into the solids ↗ |
| [[poc3-racelab-car-union]] | Tier 64: radiator and turbine in the solid duct ↗ |
| [[poc3-racelab-car-graph]] | Tier 65: the body-fitted column compiles; a volumetric device gets a port ↗ |
| [[poc3-racelab-car-render]] | Tier 66: the composite on one raster ↗ |
| [[poc3-racelab-car-windows]] | Tier 67: a learned expert can sit on at most a third of the column ↗ |
| [[poc3-racelab-bodyfitted-demo]] | Tier 68: the body-fitted car as a demo column ↗ |
| [[poc3-racelab-knobs]] | Tier 69: eleven parameters moved; four reach their subsystem ↗ |
| [[poc3-racelab-knobs-wired]] | Tier 70: rake and duct area wired ↗ |
| [[poc3-racelab-dashboard]] | Tier 71: the dashboard; the coolant knob now reaches the march ↗ |
| [[poc3-racelab-certified-step]] | Tier 72: certified mode on the implicit step's fixed point ↗ |
| [[poc3-racelab-certified-screen]] | Tier 73: certified mode on screen, its cost over a march ↗ |
| [[poc3-racelab-showable]] | Tier 74: the bundle carries the column it claims ↗ |

### PoC 4: the rocket (`case-study-rocket-ascent/` and `common/`)

| Page | Description |
|---|---|
| [[rocket-episode-pickup]] | **Start here for the rocket**: the self-contained pickup, both repositories and the four code fixes ↗ |
| [[case-study-rocket-bc-seam-atlas-0.1]] | CS-21, Tiers 76–88 in one page: the b–c seam (§1–§9), two errors that nearly cancelled (§10), E7's sign structure (§12), R0 complete (§13), R2 at 50:1 (§14), the trajectory agent (§15), W308/W312 (§16), the seams rebuilt and the wall that passed mass (§18), W334's re-derivation (§18.8), the residuals (§19), the four fixes and re-march (§20), W343 (§20.8) ↗ |
| [[rocket-episode-seam-audit]] | Audit of the marched rocket's seams (W321): picture, mesh or coupler? ↗ |
| [[case-study-rocket-ascent-2d-atlas-0.1]] | History: the first rocket plan (agent graph, four experts, Phases 0–5), never built as planned |

### Outcome: the evidence for the proposal (`atlas-0.1-outcome/`)

| Page | Description |
|---|---|
| [[00-atlas-0.1-outcome]] | **Five claims with blunt verdicts**, revised after Tier 89 ↗ |
| [[outcome-c1-decomposition-vs-monolith]] | C1: parallel decomposition $3.4$–$5.7\times$ faster from 5 to 21 rotors; serial never ↗ |
| [[outcome-c2-learned-experts-in-the-loop]] | C2: stable coupling, adjoint through the checkpoint, defect correction converging to the classical answer; the bias |
| [[outcome-c3-learned-speed-at-scale]] | C3: per-window crossover $4.90 \to 0.253$; a $2\times$-coarse classical solver cheaper still |
| [[outcome-c4-modular-multiphysics]] | C4: the compiler, five ports, zero coupling code; every multiphysics graph built; the limits |
| [[outcome-c4-path-to-declarative-cases]] | C4's plan: four steps to declarative cases and a decomposition GUI (now the studio) ↗ |
| [[outcome-c5-requirements-for-dd-native-experts]] | C5, not done: the measured specification S1–S12 for DD-native experts |
| [[outcome-evidence-ledger]] | Every quotable number with its conditions and source |
| [[outcome-demos-and-artifacts]] | Published pages, viewers, the runnable PoC demos, figures, caveats |
| [[showcase-library-plan]] | The classical case library's plan, built 2026-09-29 for the Panel workbench (retired; cases live on in the studio) ↗ |
| [[showcase-gallery]] | Every example compiled and run, one row each; every check passed ↗ |

### Proposal: workstreams, studio, website (`atlas-0.1-proposal/`)

| Page | Description |
|---|---|
| [[00-proposal-workstreams]] | The proposal hub (2026-09-30): four workstreams, verdicts, owner decisions O1–O9 ↗ |
| [[integrated-ui-plan]] | **The studio** (`atlas/studio/`), Atlas's one UI: P0–P6 built 2026-11-16/17; §12 how every future feature joins it (`demo/`) |
| [[demo-finish-plan]] | Demo items 1.1–1.3 (header, forked river, holes) and the installer; done 2026-10-01 ↗ |
| [[demo-fast-examples-plan]] | Demo item 1.4: a fast example per type, and each family's ceiling ↗ |
| [[demo-learned-case-plan]] | Demo item 1.5: the 12-rotor learned wind farm; failed G5 ↗ |
| [[proposal-chat-prompts]] | The prompts for the three chats run side by side ↗ |
| [[proposal-website-plan]] | The website plan: claims pipeline with no number typed by hand (`website/`) ↗ |
| [[website-evidence-and-citations]] | Every number the site may show, Atlas's (A1–A10) and published (L1–L6, N1–N12), with status |
| [[website-outline]] | The site section by section after the owner's decisions ↗ |
| [[vision-scenarios-and-image-prompts]] | Race car, wildfire, Moon base: stories, agent graphs, image prompts |
| [[website-design-notes]] | Design notes from three reference sites ↗ |
| [[website-image-prompts]] | The illustration brief and style bible ↗ |
| [[website-launch-checklist]] | What waited on the owner; live 2026-10-02 ↗ |

### Proposal: the chart operator, the learned expert (`atlas-0.1-proposal/expert-architecture/`)

| Page | Description |
|---|---|
| [[chart-operator-architecture]] | **The design**, versions 0–2 (v2 public since 2026-10-12): charts for geometry, waves for agreement, an energy Gram PSD for any network output ↗ |
| [[chart-operator-general-form]] | The general form: the guarantee rests on exact traces and the host holding the physics; classes G0–G3 ↗ |
| [[chart-operator-family-catalog]] | Twelve families over about forty PDEs, an architecture per family ↗ |
| [[chart-operator-design-decisions]] | Running record of what the owner decided, when, on what evidence ↗ |
| [[chart-operator-training-and-cost]] | Data, stages 0–3, objective, gates, a labelled cost model ↗ |
| [[dd-neural-prior-art-2026]] | Neural Schwarz on any geometry is published; what is left to claim, N1–N6 ↗ |
| [[chart-operator-stage-0]] | Stage 0 (2026-10-18): guarantees held, modes did not learn; E1, E4, KT fail ↗ |
| [[chart-operator-kernel-exact-forms]] | W356–W358: kernel-exact output forms; R2's hybrid met E4 and price with nothing learned |
| [[chart-operator-stage-0-rework]] | W363: the two-level coarse-exact expert; untrained removes 87–96% of coupled error, training adds nothing coupled; §20 what carries over |

### Proposal: formal proofs (`atlas-0.1-proposal/formal-proofs/`)

| Page | Description |
|---|---|
| [[formal-proofs-plan]] | Theorem inventory T1–T23 with novelty and Lean tier ↗ |
| [[formal-proofs-record]] | **What is machine-checked**: 105 theorems in 26 Lean files, no `sorry`; T3 false as first written ↗ |
| [[formal-proofs-implementation-plan]] | The route to Lean: batches, toolchain, what stays a paper proof; batches 1–6 done, 7 not started |

### History: the rocket's first implementation plan (archived 2026-11-18 to `archive/rocket-ascent-first-plan/`)
Written 2026-08-09 for the old build repository's `atlas-0.1` branch; Phase 1 built, M2 blocked on compute; replaced by CS-21 (PoC 4) from Tier 76.

| Page | Description |
|---|---|
| [[00-atlas-0.1-implementation-plan]] | The rocket master plan: layout, phase graph, milestones M0–M7 ↗ |
| [[impl-atlas-0.1-phase0-scope-and-data]] | Phase 0: classical solvers as teacher and judge; M2 blocked on compute ↗ |
| [[impl-atlas-0.1-phase1-scaffold]] | Phase 1, built 2026-08-09: the native scaffold ↗ |
| [[impl-atlas-0.1-phase2-experts]] | Phase 2 spec: the experts ↗ |
| [[impl-atlas-0.1-phase3-integration]] | Phase 3 spec: dispatch, subcycling, conservation projection ↗ |
| [[impl-atlas-0.1-phase4-validation]] | Phase 4 spec: falsification table, milestone checks, seven ablations |
| [[impl-atlas-0.1-phase5-expansion]] | Phase 5 spec: scenario 2, reuse levels, versioning to 0.2 |
| [[impl-atlas-0.1-corpus-completion-plan]] | The M2 work plan: six decisions, three modules, two gates ↗ |
| [[impl-atlas-0.1-compute-and-training-budget]] | Cross-phase resource plan and route B′ ↗ |
| [[atlas-0.1-implementation-log]] | Append-only rocket build tracker from 2026-08-09 |

## Adjusted Attention Mechanisms — `concepts/adjusted-attention-mechanism/`
How tokens interact. The complement to the tokenizer folder: symmetric (reciprocal + conservative) attention as the core, plus large-context variants.

| Concept | Page | Description |
|---|---|---|
| ★ Overview / hub | [[00-attention-overview]] | Why attention needs fixing for physics; reciprocal ≠ conservative; the catalog |
| Symmetric attention (core) | [[symmetric-attention-physics]] | Reciprocity = Newton's 3rd law; the dissipative-vs-conservative resolution; attention = learnable graph Laplacian |
| Skew-symmetric dynamics & attention | [[skew-symmetric-attention]] | The conservative channel; rotation vs. stretching; math + pros/cons; survey of models using skew-symmetric structure |
| Index Share sparse attention | [[index-share-sparse-attention]] | GLM-derived; local+global+landmark indices = PDE locality; large-domain scaling |
| Hierarchical query attention | [[hierarchical-query-attention]] | Q-Former/HierarQ; resolution-free compression; local+global query streams; bounded long-rollout memory |

*See also the prior synthesis [[antisymmetric-signed-attention-transformer]] in `extension-architectures/`.*

## Physics Foundations — `concepts/physics-foundations/`
| Concept | Page | Description |
|---|---|---|
| Partial Differential Equations | [[partial-differential-equations]] | PDE taxonomy (NS, Euler, MHD, Stokes, Burgers) |
| Navier-Stokes Equations | [[navier-stokes-equations]] | NS in depth; turbulence; Reynolds number; PPE; ML approaches |
| Message Passing / Belief Propagation | [[message-passing-belief-propagation]] | BP algorithms; cyclicity reinterpretation; GNN connections |

## ML Building Blocks — `concepts/ml-building-blocks/`
| Concept | Page | Description |
|---|---|---|
| Multimodal Tokenization | [[multimodal-tokenization]] | Patch/VQ/Q-Former/Perceiver/Cosmos; physics tokenization gaps |
| Iterative Refinement Architecture | [[iterative-refinement-pfm]] | Progressive per-step constraint enforcement; diffusion analog |
| In-Context Learning for Physics | [[in-context-learning-physics]] | Physics ICL; enabling conditions; GP$_{\text{hy}}$T evidence |
| Transformer Architectures for Physics | [[transformer-architectures]] | ViT/DiT/axial; PDE-discretization theory; token clustering |
| Neural Operators | [[neural-operators]] | DeepONet, FNO, variants; operator-learning framework |
| Neural Surrogate Models | [[neural-surrogates]] | Taxonomy; accuracy-speed tradeoff; hybrid methods |
| ★ **Data augmentation for surrogates** | [[data-augmentation-physics-surrogates]] | **Which augmentations survive engineering geometry**: symmetry dies on fixed domains, scaling is already quotiented out, and all2all pair amplification pays ↗ |
| ★ **Physics simulation datasets** | [[physics-simulation-datasets]] | **Datasets mapped by governing family, not scenario**, and a four-axis domain-gap discount for using them ↗ |
| Diffusion Models for Physics | [[diffusion-models-physics]] | Score/flow matching; latent diffusion; advantages |
| Autoregressive Rollout Stability | [[autoregressive-rollout-stability]] | Error accumulation; patch jittering; diffusion fixes |
| Transfer Learning and Fine-Tuning | [[transfer-learning-fine-tuning]] | Pretraining; LoRA; PI fine-tuning; diversity principle |
| Mixture of Experts | [[mixture-of-experts]] | MoE for physics; hard vs. soft routing |
| Memory-Augmented Physics Models | [[memory-augmented-physics-models]] | Static + dynamic computation; U-shaped scaling law |
| Gaussian Processes | [[gaussian-processes]] | Kernels; posterior; NTK; connection to attention |
| Equivariant Graph Neural Networks | [[equivariant-gnns]] | Spatial → SE(3) → conservation-constrained taxonomy |
| World Models in Physics AI | [[world-models-physics-ai]] | Newtonian vs. Keplerian; context-length phase transition |
| ★ **RL for flow control & coordination** | [[rl-for-flow-control-and-coordination]] | **HydroGym assessed against the ladder**: not an expert but a ready-made reuse battery, and RL's third role formalised ↗ |

## Benchmark Architecture Specs — `concepts/benchmark-architectures/`
Implementation-ready specs for the 5-way Burgers' benchmark. Each covers full internal architecture, equations, and implementation notes.

| Concept | Page | Description |
|---|---|---|
| Arch 1: Autoregressive Transformer | [[arch-autoregressive-transformer]] | Tubelet patching, patch jittering, axial RoPE, factorized attention, rollout |
| Arch 2: Diffusion Backbone | [[arch-diffusion-backbone]] | Latent spectral diffusion; DiT denoiser; DPS+Adam guidance |
| Arch 3: Neural Differentiator | [[arch-neural-differentiator]] | Derivative prediction + RK4 integrator; ODE error analysis |
| Arch 4: Physics-Mamba (SSM) | [[arch-physics-mamba]] | Selective SSM; Δt-as-input; axial 2D scan; optional global attention |
| Arch 5: GNN-PB | [[arch-gnn-physics-bottleneck]] | Antisymmetric edge frames (exact momentum); energy bottleneck; ghost-node BCs |

## Extension Architecture Proposals — `concepts/extension-architectures/`
Ideas beyond the benchmark five, each grounded in canonical literature.

| Concept | Page | Description |
|---|---|---|
| Multi-Scale Hierarchical GNN | [[multiscale-hierarchical-gnn]] | FMM-analog V-cycle; long-range coupling at $O(N)$; closes GNN-PB gap |
| Hamiltonian + Dissipative Message Passing | [[hamiltonian-message-passing]] | Helmholtz decomposition per edge; exact energy conservation; level 4 → 5 |
| Action-Based / Noether Enforcement | [[action-based-noether-enforcement]] | Lagrangian functional + symmetries → automatic conservation; subsumes equivariance |
| Antisymmetric / Signed-Edge Attention | [[antisymmetric-signed-attention-transformer]] | Newton's 3rd law at attention level; fixes clustering pathology |

## Unified Token Representation — `concepts/unified-token-representation/`
How heterogeneous physical states become token sequences — **one deep-dive page per representation** (intuition, mathematics, pros/cons). The physics-specific complement to [[multimodal-tokenization]]. Start at the hub.

| Concept | Page | Description |
|---|---|---|
| ★ Overview / hub | [[00-token-representation-overview]] | Why tokenization is the PFM bottleneck; 8-criterion scorecard; 3 trade-off axes; folder map |
| Patch-embedding tokens | [[patch-embedding-tokens]] | Linear ViT patches — default of Walrus/GP$_{\text{hy}}$T/Poseidon/MPP/DPOT |
| Spatiotemporal tubelet tokens | [[spatiotemporal-tubelet-tokens]] | 3D/video patches encoding short-time dynamics (GP$_{\text{hy}}$T, Cosmos) |
| Hierarchical windowed tokens | [[hierarchical-windowed-tokens]] | Multiscale Swin/scOT pyramid (Poseidon) |
| Adaptive-compute tokens | [[adaptive-compute-tokens]] | Resolution-adaptive patching / CSM (Walrus) |
| Vector-quantized tokens | [[vector-quantized-tokens]] | Discrete codebooks VQ/FSQ/LFQ (AION-1, DALL-E) |
| Scalar CDF tokens | [[scalar-cdf-tokens]] | Distribution-aware scalar/parameter encoding (AION-1) |
| Spectral / Fourier tokens | [[spectral-fourier-tokens]] | Mode coefficients; resolution-free, global (FNO/SFNO/PISD) |
| Branch–trunk operator tokens | [[branch-trunk-operator-tokens]] | DeepONet function/query split; resolution-free output |
| Graph / mesh tokens | [[graph-mesh-tokens]] | Node/edge tokens; geometry-native (GNS, Dynami-CAL, MGNO) |
| Coordinate / implicit tokens | [[coordinate-implicit-tokens]] | Token = coordinate; INR/PINN; exact autodiff derivatives |
| Learned-query compression | [[learned-query-compression-tokens]] | Q-Former/Perceiver fixed-budget compression |
| *Alt:* Physics-conditioned queries | [[physics-conditioned-query-tokens]] | Q-Former queries = physical operators (resolution-inv. + interpretable + scale-aware) |
| *Alt:* Wave–particle dual tokens | [[wave-particle-dual-tokens]] | Continuous + discrete simultaneously; Eulerian/Lagrangian duality |
| *Alt:* Structure-preserving tokens | [[structure-preserving-tokens]] | Equivariant / conservation-aware / coordinate-&-scale-aware (encoding-spectrum level 5) |
| **Axis deep-dive** | [[tokenization-tradeoff-axes]] | Full pros/cons of continuous/discrete, local/global, grid-tied/resolution-free — PDE-type mapping, design roadmap |

---

# Miscellaneous (`misc/`)

| Page | Description |
|---|---|
| [[prereqs-and-resources]] | Tiered learning roadmap — PDEs, spectral methods, numerical methods, Hamiltonian mechanics, functional analysis, fluids — for someone entering with ML + calculus + ODEs |
| [[context-router]] | **For LLM sessions**: which pages to read for each kind of task (build, reference, design, decision, brainstorm, ingest, lint), and which never to read whole (at the wiki root) |
| [[handoff-atlas-0.1-corpus-generation]] | *History (2026-08, M2 never run; archived 2026-11-18 to `archive/`):* **Session handoff prompt** for executing M2 on the code repo: thesis, build state, the unmet oracles and their causes, settled decisions and costs ↗ |

---

# Source Summaries (`summaries/`)

## Foundation Models — `summaries/foundation-models/`
| Title | Page | Key Topic |
|---|---|---|
| Poseidon: Efficient Foundation Models for PDEs | [[poseidon-pde-foundation-model]] | scOT operator transformer + all2all semi-group training; 15 OOD tasks; ~20 samples = FNO@1024 |
| Towards a Physics Foundation Model (GP$_{\text{hy}}$T) | [[gphyt-physics-foundation-model]] | Neural differentiator + in-context physics; 7× SOTA |
| WALRUS: Cross-Domain Foundation Model (full paper) | [[walrus-paper]] | 1.3B-param transformer; patch jittering; 19 scenarios |
| Walrus Overview (Polymathic AI blog) | [[walrus-overview]] | Accessible companion to the Walrus paper |
| AION-1: Omnimodal Foundation Model for Astronomy | [[aion-1-astronomy]] | 39 modalities; 200M objects; masked multimodal modeling |
| Multimodal Learning With Transformers: A Survey | [[multimodal-transformers-survey]] | 4 phases; tokenization; Q-Former; fusion; PFM relevance |

## Transformers / Theory — `summaries/transformers-theory/`
| Title | Page | Key Topic |
|---|---|---|
| PDE-Transformer (ICML 2025 full paper) | [[pde-transformer-paper]] | Diffusion transformer for PDEs; separate-channel representation |
| PDE-Transformer (Landing Page) | [[pde-transformer-landing]] | Capability comparison table; accessible overview |
| Mathematical Framework for Transformers | [[transformer-mathematical-framework]] | Transformer = discretization of an integro-differential equation |
| Transformers as Interacting Particle Systems | [[transformers-particle-systems-clustering]] | Mean-field clustering; Wasserstein gradient flow |
| From Kepler to Newton: Inductive Biases | [[kepler-newton-inductive-biases]] | Context length controls Newtonian vs. Keplerian world model |
| PhysicsFormer: Transformer-PINN for NS | [[physicsformer-pinn-ns]] | Encoder-decoder PINN; w·sin(t) activation; MSE ~10⁻⁶ |
| Conditional Memory via Scalable Lookup (Engram) | [[conditional-memory-engram]] | Static memory lookup + MoE; U-shaped sparsity law |
| GLM 5.2: Index Share Sparse Attention + MTP | [[glm-index-share-attention]] | Group-shared local/global/landmark indices; multi-token prediction; KV redesign; long-context |
| Q-Former: Learned-Query Multimodal Alignment | [[q-former-architecture]] | BLIP-2/HierarQ/DisenQ; query compression; local+global query streams; resolution-free tokenizer |

## Neural Operators — `summaries/neural-operators/`
| Title | Page | Key Topic |
|---|---|---|
| DeepONet as Multi-Operator Extrapolation | [[deeponet-multi-operator]] | D2NO pretraining + physics-informed zero-shot fine-tuning; LoRA |
| Physics-Constrained DeepONet for CFD | [[pc-deeponet-cfd]] | Divergence-free velocity via skew-symmetric Jacobian |
| Variationally Mimetic Operator Network | [[varmion-viscous-flows]] | Branch architecture from variational (weak) form of Stokes |

## Graph Networks / Equivariant GNNs — `summaries/graph-networks/`
| Title | Page | Key Topic |
|---|---|---|
| Learning to Simulate Complex Physics (GNS) | [[gns-graph-network-simulators]] | Encode-process-decode GNN; 34× generalization |
| Dynami-CAL GraphNet | [[dynami-cal-graphnet]] | Exact linear + angular momentum via antisymmetric edge frames |
| EquiformerV3 | [[equiformer-v3]] | SE(3)-equivariant graph attention; 1.75× speedup; SOTA |
| Equivariant Quantum NNs for HEP at LHC | [[eqnn-hep-lhc]] | EQNN with p4m symmetry; quantum ML; HEP classification |
| Multipole Graph Neural Operator | [[multipole-graph-neural-operator]] | FMM-inspired multi-level GNN; $O(N)$ all-range interactions |
| Anti-Symmetric DGN | [[anti-symmetric-dgn]] | Skew-symmetric weights; non-dissipative ODE flow; ICLR 2023 |

## Hamiltonian / Lagrangian — `summaries/hamiltonian-lagrangian/`
| Title | Page | Key Topic |
|---|---|---|
| Hamiltonian Neural Networks | [[hamiltonian-neural-networks]] | Parameterize $H_\theta(q,p)$; exact energy conservation |
| Dissipative Hamiltonian Neural Networks | [[dissipative-hamiltonian-neural-networks]] | Helmholtz $H + D$ split; conservative + dissipative |
| Lagrangian Neural Networks | [[lagrangian-neural-networks]] | Parameterize $L_\theta(q,\dot q)$; works without canonical coords |
| Noether Networks | [[noether-networks]] | Meta-learn conserved quantity; test-time tailoring |

## Diffusion / Generative — `summaries/diffusion-generative/`
| Title | Page | Key Topic |
|---|---|---|
| Lost in Latent Space: Latent Diffusion for Physics | [[latent-diffusion-physics]] | 1000× compression robustness; diffusion > deterministic |
| Multiscale Diffusion for Solar Systems | [[multiscale-diffusion-solar]] | Multiscale temporal inference; 8.5 TB solar dataset |
| Physics-Informed Diffusion in Spectral Space (PISD) | [[pisd-physics-informed-spectral-diffusion]] | Spectral latent diffusion; Adam-DPS; 100–10,000× lower residuals |

## Solvers & Simulation — `summaries/solvers-and-simulation/`
| Title | Page | Key Topic |
|---|---|---|
| Deep Learning for NS on Non-Uniform Grids | [[navier-stokes-nonuniform-grids]] | MConv-based HyDEA; hybrid DL + CG |
| Genesis: Universal Physics Engine | [[genesis-physics-engine]] | 10–80× faster; unified solver; generative data engine for PFM |

## Other Methods — `summaries/other-methods/`
| Title | Page | Key Topic |
|---|---|---|
| Message Passing and Cyclicity Transition | [[message-passing-cyclicity]] | BP identifies cyclicity, not giant component |
| Persistent Memory via Triple-Loop Consolidation | [[deep-memory-dissipative]] | Non-gradient DM; MoE routing; hippocampal analogy |
| Gaussian Process Regression Tutorial | [[gaussian-process-regression]] | GP definition, RBF kernel, posterior, $O(N^3)$ scaling |

---

## Key Cross-Cutting Themes

### "Train Once, Deploy Anywhere"
A single pretrained model that handles any physical system via in-context learning or few-shot finetuning. Tracked through [[poseidon-pde-foundation-model]] (629M params; operator learning; 15 OOD tasks from a 6-operator pretraining set), [[gphyt-physics-foundation-model]] (385M params; in-context), [[walrus-paper]] (1.3B params; autoregressive), [[aion-1-astronomy]] (39 modalities; masked). These four span the four training paradigms (operator / in-context-derivative / autoregressive / masked).

### Three PFM Training Paradigms (compared in [[pfm-architecture-approaches]])
- **Operator learning** — full trajectory from the initial datum: [[poseidon-pde-foundation-model]].
- **Autoregressive / in-context next-step** — [[walrus-paper]], [[gphyt-physics-foundation-model]].
- **Masked multimodal modeling** — [[aion-1-astronomy]].
Poseidon argues operator learning ≠ next-step prediction and shows it scales + generalizes.

### Unified Token Representation
The "no canonical data template" obstacle — how heterogeneous fields/grids/dimensions become tokens. Full per-representation treatment in [[00-token-representation-overview]]; broader taxonomy in [[multimodal-tokenization]].

### Physics Encoding Spectrum (weak → strong)
1. Data-driven only ([[walrus-paper]], [[poseidon-pde-foundation-model]], GP$_{\text{hy}}$T backbone)
2. Soft loss constraints ([[deeponet-multi-operator]])
3. Test-time physics guidance ([[pisd-physics-informed-spectral-diffusion]], [[noether-networks]])
4. Architectural soft biases ([[varmion-viscous-flows]], [[arch-gnn-physics-bottleneck]], [[multipole-graph-neural-operator]], [[anti-symmetric-dgn]])
5. Hard architectural constraints ([[pc-deeponet-cfd]], [[dynami-cal-graphnet]], [[hamiltonian-neural-networks]], [[lagrangian-neural-networks]])

*Full discussion: [[00-start-here]].*

### Generative vs. Deterministic Emulators
Strong evidence from [[latent-diffusion-physics]] that diffusion emulators outperform deterministic surrogates — critical for chaotic/turbulent systems.

### Polymathic AI Cluster
A coherent research program toward universal scientific foundation models: [[walrus-paper]], [[walrus-overview]], [[latent-diffusion-physics]], [[multiscale-diffusion-solar]], [[aion-1-astronomy]].

---

## Navigation Tips
- All `[[page-name]]` links are Obsidian-style and folder-independent.
- **[AI Inference]** sections contain clearly-marked speculative connections.
- Concept pages contain LaTeX for all mathematics.
- LLM session → [[context-router]]. New reader → [[00-start-here]]. Looking for code → [[code-and-papers]]. Build history → [[log]].

---

*This index is maintained by the wiki system. Update after each new source ingestion.*
