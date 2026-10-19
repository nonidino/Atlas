# Physics Foundation Model Wiki — Index

**Last Updated:** 2026-10-18
**Sources Ingested:** 36 documents

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
**The first concrete model.** One coherent architecture (not the design space): broad multiphysics, direct state prediction, symmetric attention, unified graph tokenizer, inbuilt conservation. The bridge from design layer → build layer. *(Folder renamed from `initial-pfm-model` 2026-07-04; bare links unchanged.)*

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

## ★★ Atlas 0.1 (new architecture track) — `concepts/Atlas 0.1/`
*(Folder restructured 2026-08-19: **`common/`** holds the case-study-agnostic architecture; each case study now owns a folder — `case-study-rocket-ascent/` (with its `implementation/` subfolder), `case-study-wind-farm-wake/`, `case-study-rbc-decomposition/`. All links are bare `[[page-name]]` and were unaffected.)*
**A parallel track to Noether 1.1, not a replacement.** Design thesis: physical objects as **agents** (local coordinate charts), typed **edges** (transition maps) declaring what must be continuous where charts meet, and specialized **experts** organized by governing-equation family rather than by field-quantity label. Conservation is an *imposed* interface constraint, not a *discovered* invariant — the opposite emphasis from Noether 1.1, hence the different name. Emerged from [[regime-moe-architecture]] → [[pfm-purpose-and-direction]] → [[incremental-transfer-roadmap]] → the rocket-ascent worked example. Status: design complete; **Phases 0 and 1 built 2026-08-09** on the `atlas-0.1` branch (M0, M3 green). **2026-08-18: the six M2 decisions D1–D6 are settled in code, the T3 generator modules and the T2 solo tier are built, and generation is restructured into 5-hour portions** (the `shell` portion is generated: 64 full 10 s thermostruct histories for 0.5 core-hours) — route B′ timing, a rounded throat (M1 now 11 of 12), 10 recorded interfaces against 7 declared, exchange stamps, and resumable content-addressed generation; **M2 generation itself still blocked on compute**. **2026-08-20: the wind-farm case study is under build on its own branch `atlas-0.1-windfarm` — gates W0–W3 pass, including the go/no-go ([[results-w0-w3-wind-farm]]), with nothing trained.** Phases 2–5 below are still spec, not build reports.

*Rows marked ↗ have a longer description, kept word for word in [[index-digest]]. The table is split by subfolder (2026-10-17); within each, rows keep their earlier order.*

### Top level: the thesis

| Concept | Page | Description |
|---|---|---|
| ★ Overview / thesis / hub | [[00-atlas-0.1-overview]] | The manifold-atlas metaphor; end-to-end forward pass; per-portion map; 7 design invariants; explicit contrast with Noether 1.1 |

### `common/`: the framework, its theory, and the records of every case study

| Concept | Page | Description |
|---|---|---|
| Agent definition | [[agent-definition-atlas-0.1]] | What counts as an agent; granularity heuristic; per-agent conditioning; worked across all [[pfm-purpose-and-direction]] examples |
| Tokenizer | [[graph-tokenizer-atlas-0.1]] | Reuses [[graph-tokenizer-1.1]] directly per-agent; adds boundary-token exposure for edge attachment |
| **Edge generation** | [[edge-generation-atlas-0.1]] | **Two live mechanisms** — declared/geometric instantiation (Stage 1 default) *and* RL-based instantiation (deferred until ≥2 scenarios exist); the rocket's typed edge list, $b\!-\!g$ removed |
| **Expert library** | [[expert-library-atlas-0.1]] | **Central design page** — experts cut by governing-equation family, not edge-type label; the rocket's 4-expert table; physics-encoding-spectrum placement |
| U-Net expert hierarchy | [[unet-hierarchy-atlas-0.1]] | Pool/unpool over the agent graph (reuses [[multiscale-hierarchical-gnn]] + [[intelligent-patching]]); MLP vs. RL gating mapped onto smooth/threshold; resolves [[regime-moe-architecture]]'s shared-trunk-vs-disjoint question; v0 uses 2 levels not 4 |
| Conservation | [[conservation-as-constraint-atlas-0.1]] | Flux-matching constraint at declared conservation-typed edges — direct contrast with [[discovered-conservation-1.1]]; the "conservation isn't its forte" naming rationale, formalized |
| Global fields & topology | [[global-fields-and-topology-atlas-0.1]] | Gravity as a uniform field bypassing the edge mechanism entirely; staging/docking as first-class graph-topology mutation (not designed yet, requirement only) |
| Training & bootstrap | [[training-and-bootstrap-atlas-0.1]] | Applies [[incremental-transfer-roadmap]]'s 4-stage plan to the 4 rocket experts; the v0 scope-reduction decisions (declared edges, MLP gating, 2-level U-Net, direct flux-matching) and their trigger conditions for revisiting |
| ★★ **Expert donor survey** *(new 2026-08-31)* | [[expert-donor-survey]] | **What is actually downloadable, per governing family**: seven families checked for licence, resolution, BCs as input, gradients, native time step and training distribution, with a frozen-or-transfer verdict per model ↗ |
| ★★ **Prior art, novelty & scaling** | [[prior-art-and-novelty-atlas-0.1]] | **The positioning audit, to read before claiming novelty anywhere**: what is and is not new, checked against sources (Schwarz, preCICE, cPINN/XPINN, CompNO, bond graphs and more) ↗ |
| ★★★ **Port algebra** *(binding)* | [[port-algebra-atlas-0.1]] | **The binding interface contract**, rebuilt 2026-08-19: five effort–flow ports whose product is power, `MECH`, `ROT`, `THERM`, `ELEC` and the `ADVEC` multiport ↗ |
| ★★ **Symmetry averaging** *(first exact enforcement)* | [[symmetry-averaging-atlas-0.1]] | **An invariance the frozen expert lacks, supplied exactly without training**: averaging over a closed symmetry group makes any operator equivariant at one forward pass per group element ↗ |
| ★★ **Schwarz iteration** *(built 2026-08-23; S1 refused as unbuildable)* | [[schwarz-iteration-atlas-0.1]] | **Giving a frozen expert a field of view larger than its input, in forward passes rather than parameters**: the planned answer to OP-5, and why its fix is the minor term ↗ |
| ★★ **Open problems** *(deferred, with causes)* | [[open-problems-atlas-0.1]] | **Problems found by measurement, diagnosed, then deliberately deferred**, each with a fix that is a design decision (OP-1 onward), so picking one up later costs a read rather than a rediscovery ↗ |
| ★★★ **Composition error theory** *(new 2026-08-26)* | [[composition-error-theory]] | **What interface agreement buys on its own: nothing.** The worst-agent conjecture is false as stated and true under three hypotheses; the page separates the three ways it fails ↗ |
| ★★★ **Probed DtN coupling** *(new 2026-08-26)* | [[probed-dtn-coupling]] | **A way to make agents agree that needs nothing from the expert**: probe the interface operator (Steklov–Poincaré) directly instead of climbing a Robin or Ventcell ladder frozen experts cannot climb ↗ |
| ★★★ **Temporal error accumulation** *(new 2026-08-26)* | [[temporal-error-accumulation]] | **The rollout bound**: composition adds no new error law, the recursion is the monolith's with one extra term, and growth is exponential only when the step's Lipschitz constant exceeds one ↗ |
| ★★★ **Master error bound** *(new 2026-08-26)* | [[master-error-bound]] | **The complete error formulation, derived once**: the per-step defect is three terms, agent infidelity, transmission infidelity and solve incompleteness ↗ |
| ★★★ **Control observability** *(measured result, new 2026-09-06)* | [[control-observability]] | **Tier 32, one measurement with five controls**: for a globally receptive agent the master bound's indicator is identically 1, so the bound is vacuous rather than violated; the lever left is observability, measured here ↗ |
| ★★★ **The interaction horizon** *(measured result, new 2026-09-07)* | [[interaction-horizon]] | **Tier 33: the framework's own certificate was a placeholder.** The quantity Π is a 0/1 envelope standing in for a sensitivity nobody had measured ↗ |
| ★★★ **General coupling scheme** *(new 2026-08-26)* | [[general-coupling-scheme]] | **Schwarz generalised into a seven-parameter scheme compiled from what the agents declare**; every coupling method the vault has argued about is one point in it ↗ |
| ★★★ **Generalization requirements** *(new 2026-08-26; restructured same day)* | [[generalization-requirements]] | **Twenty gaps, sorted by which of two goals each blocks**: accurate composition and generic composition share only four items, so progress on one does not buy the other ↗ |
| ★★★ **Gap worklist** *(tracker, opened 2026-08-26)* | [[gap-worklist]] | **The actionable tracker**: every work item (W-number) with its order and a definition of done precise enough that it cannot be quietly declared finished. Opens with a status board of all 348 items; holds Tiers 76 onward ↗ |
| Gap worklist, Tiers 0–37 *(archive, 2026-10-17)* | [[gap-worklist-tiers-0-37]] | The worklist's sections for Tiers 0–37, moved word for word; history only, current status is on the board in [[gap-worklist]] |
| Gap worklist, Tiers 38–75 *(archive, 2026-10-17)* | [[gap-worklist-tiers-38-75]] | The worklist's sections for Tiers 38–75, moved word for word; history only, current status is on the board in [[gap-worklist]] |
| ★★★ **CS-13 — the cooling loop** *(case study, new 2026-09-09)* | [[case-study-cooling-loop-atlas-0.1]] | **CS-13, rung 6: the first graph whose ports form a directed cycle**, a block cooled by a closed coolant loop with one real expert and four lumped legs ↗ |
| ★★★ **CS-14 — the powertrain** *(case study, new 2026-09-09)* | [[case-study-powertrain-atlas-0.1]] | **CS-14, rung 7: the rotor's open shaft port connected** to a motor-generator, a DC bus and a battery, closing the power that previously left the system ↗ |
| ★★★ **CS-S1 — the neural interface predictor** *(scheme study, new 2026-09-09)* | [[case-study-neural-interface-atlas-0.1]] | **CS-S1: the first study whose variable is the coupling scheme**, a neural interface predictor in the one arrangement the R10 refusal does not reach ↗ |
| ★★★ **The epsilon-halo, measured** *(negative result, new 2026-09-10)* | [[epsilon-halo-measurement]] | **Negative result: the ε-halo past `L2/R10` does not exist.** The band-truncation error plateaus, so six of six learned configurations on two donors reach no target at any non-trivial radius ↗ |
| ★★★ **The substitution-campaign checkpoint** *(declaration, new 2026-09-10)* | [[substitution-campaign-checkpoint]] | **The substitution campaign's stopping rule, applied**: the quoted "nine of nine seams refused Poseidon-T" was wrong three ways, and compiled honestly no seam refuses a learned expert ↗ |
| ★★★ **CS-S2 — a bounded donor, measured** *(off-ladder study, new 2026-09-10)* | [[case-study-bounded-donor-atlas-0.1]] | **CS-S2: a bounded donor (NeuberNet) measured, and negative**: the patch is bounded but its response is not ↗ |
| ★★★ **Can an inequality seam compile?** *(decision, new 2026-09-10)* | [[inequality-seam-admissibility]] | **Rung 8's admissibility question: a contact seam compiles, and that is the defect.** It poses a variational inequality the solver does not; layer by layer, two layers break silently and E1 is stamped true where it is false ↗ |
| ★★★ **CS-17 — the learned pair** *(case study, rung 3, new 2026-09-10)* | [[case-study-learned-pair-atlas-0.1]] | **CS-17, rung 3: the first graph with two learned experts of different families**, Poseidon-T across a `MECH` seam from NeuberNet, and it compiles ↗ |
| ★★★ **The decomposition axis, per region** *(schema change, new 2026-09-10)* | [[per-region-decomposition-axis]] | **W171 part 1: the decomposition axis made per region**, after three functions returned early on the wrong axis for a union graph; the carrier had to be declared after all ↗ |
| ★★★ **The overlapping mechanism, per region** *(schema change, new 2026-09-10)* | [[per-region-assembly]] | **W189 and W171 part 2: rung 9's schema completed**, one partition of unity per overlapping region rather than one per region ↗ |
| ★★★ **The joining seams** *(measurement, new 2026-09-11)* | [[joining-seam-cost]] | **W172 closed: rung 9's three joining seams built and the union compiles**; two assumed joins were not constructible, and the prediction they test is half right ↗ |
| ★★★ **Rung 9's gate, restated** *(gate re-read, new 2026-09-11)* | [[rung9-gate-restated]] | **Neither clause of rung 9's gate fits the full-vehicle graph as written**, so a replacement gate is proposed openly rather than substituted ↗ |
| ★★★ **Fourteen formulations, and the kill tests** *(divergence and kill record, new 2026-09-12)* | [[learned-contribution-kill-tests]] | **What a learned expert could contribute at all**: fourteen formulations and the cheap kill tests that answer most of them in the negative ↗ |
| ★★★ **Defect correction with a learned operator** *(theory, pre-registered gate, CS-S3, new 2026-09-12)* | [[defect-correction-learned-operator]] | **CS-S3: defect correction with a learned operator**, the one slot where a learned expert changes the rate and cannot change the answer; the pre-registered gate was failed by the checkpoint ↗ |
| ★★★ **Which vehicle this is** *(two declarations, new 2026-09-12)* | [[vehicle-scale-and-sizing]] | **The two non-framework decisions rung 9 needed (W201)**: the tiling's length and speed units, and what follows from them for the vehicle's three clocks ↗ |
| ★★★ **CS-18 — rung 9, marched** *(case study, rung 9, pre-registered gate, new 2026-09-12)* | [[case-study-vehicle-march-atlas-0.1]] | **CS-18, rung 9 marched**: eighteen agents, five families, three joins and three clocks; the gate is measurable, and no single march spans the vehicle's time scales ↗ |
| ★★★ **The corrupted checkpoint, swept** *(measurement, mechanism, CS-S5, new 2026-09-12)* | [[corrupted-checkpoint-and-jacobian-fidelity]] | **CS-S5 / W205: Tier 48's most quoted number was the best of twelve draws**, and the mechanism behind the corrupted checkpoint's saving is measured ↗ |
| ★★★ **CS-19 — RaceLab phase 1, a car as a graph** *(case study, PoC 3 phase 1, pre-registered gate, new 2026-09-12)* | [[case-study-racelab-graph-atlas-0.1]] | **CS-19, PoC 3 phase 1: the car marches as a graph** of thirteen bodies in fourteen windows, and the shape breaks two of the framework's disciplines and one expert ↗ |
| ★★★ **CS-20 — RaceLab phase 2, the switch** *(case study, PoC 3 phase 2, W222 closed, new 2026-09-12)* | [[case-study-racelab-switch-atlas-0.1]] | **CS-20, PoC 3 phase 2: the classical/learned switch works**, but the learned column cannot be asked for the step the composition layer runs at ↗ |
| ★★★ **PoC 3 — the demo, and the arms re-run on the column it marches** *(case study, PoC 3 phase 3, new 2026-09-13)* | [[poc3-racelab-demo]] | **PoC 3 phase 3: the demo marches one vehicle**, and CS-19's five arms re-run on its column all land inside the envelope ↗ |
| ★★★ **The traced car, and the arms re-run on it** *(case study, PoC 3, new 2026-09-13)* | [[poc3-racelab-traced-car]] | **The car traced as a real silhouette** in place of disconnected plates, and the arms re-run on it ↗ |
| ★★★ **PoC 3 phase 4 — the half-car in three dimensions** *(case study, PoC 3 phase 4, new 2026-09-13)* | [[poc3-racelab-3d]] | **PoC 3 phase 4: a half-car in 3-D** with a symmetry plane and a 3-D window solver, classical only; W236's staggering defect found by measurement ↗ |
| ★★★ **The drawn car, and the arms that were not run** *(case study, PoC 3, Tier 56, new 2026-09-13)* | [[poc3-racelab-drawn-car]] | **Tier 56: the car the owner drew replaced the traced one, and the gate refused**; two envelopes decline (W244), so the five arms were not run ↗ |
| ★★★ **PoC 3 phase 5 — the bundle** *(case study, PoC 3 phase 5, Tier 57, new 2026-09-13)* | [[poc3-racelab-bundle]] | **PoC 3 phase 5: the bundle installs and self-tests from a fresh clone on Windows and Linux**, after three verification rounds that each hid a defect (W250) ↗ |
| ★★★ **The drawn car's two faults, and the fixes measured on copies** *(case study, PoC 3, Tier 58, new 2026-09-13)* | [[poc3-racelab-car-fixes]] | **Tier 58: the drawn car's two faults re-read on five copies**, and neither is what it looked like ↗ |
| ★★★ **The outlet opened, the start settled, and the drawn car's arms** *(case study, PoC 3, Tier 59, new 2026-09-14)* | [[poc3-racelab-outlet-and-start]] | **Tier 59: the outlet opened and the start settled**; the drawn car runs its five arms inside every envelope for 600 macro-steps ↗ |
| ★★★ **Body-fitted grids for RaceLab: the foundation** *(case study, PoC 3, Tier 60, new 2026-09-14)* | [[poc3-racelab-body-fitted-grids]] | **Tier 60: body-fitted overlapping grids for RaceLab**, the foundation built and verified before any car part goes on it ↗ |
| ★★★ **Navier–Stokes on body-fitted grids: the flow solver, verified** *(case study, PoC 3, Tier 61, new 2026-09-15)* | [[poc3-racelab-overset-flow]] | **Tier 61: incompressible flow on the body-fitted grids**, verified against two known answers including the cylinder at Re = 100 ↗ |
| ★★★ **The drawn car as solids, on body-fitted grids** *(case study, PoC 3, Tier 62, new 2026-09-15)* | [[poc3-racelab-car-solids]] | **Tier 62: the drawn car as welded solids on body-fitted grids**, 381,698 unknowns on ten grids, stable from the free stream ↗ |
| ★★ **The duct opened: the car's cooling air on solid walls** *(case study, PoC 3, Tier 63, new 2026-09-15)* | [[poc3-racelab-duct-openings]] | **Tier 63: the duct's inlet and outlet cut into the solids** and the cooling flow measured before the devices go in ↗ |
| ★★★ **The devices in the body-fitted duct: the union's three joins on solid walls** *(case study, PoC 3, Tier 64, new 2026-09-15)* | [[poc3-racelab-car-union]] | **Tier 64: the radiator core and turbine back in the solid duct**, where the union's three joins close better than on the porous column ↗ |
| ★★★ **The body-fitted column declared: a volumetric device gets a port** *(case study, PoC 3, Tier 65, new 2026-09-15)* | [[poc3-racelab-car-graph]] | **Tier 65: the body-fitted column compiles** and is refused only at `L7/R9`; a volumetric device gets a port ↗ |
| ★★ **The body-fitted composite on one raster: the demo's field overlay** *(case study, PoC 3, Tier 66, new 2026-09-16)* | [[poc3-racelab-car-render]] | **Tier 66: the body-fitted composite drawn on one raster**, using the solver's own interpolation so the picture can be trusted ↗ |
| ★★★ **Where a learned expert can sit on the body-fitted column** *(case study, PoC 3, Tier 67, new 2026-09-16)* | [[poc3-racelab-car-windows]] | **Tier 67: where a learned expert can sit on the body-fitted column**: at most a third of it, none near the car ↗ |
| ★★ **The body-fitted car as a demo column, and the cache that starts it** *(case study, PoC 3, Tier 68, new 2026-09-16)* | [[poc3-racelab-bodyfitted-demo]] | **Tier 68: the body-fitted car as a second demo column**, and the cache that starts it ↗ |
| ★★ **Section 3.3's parameters, and what each one actually reaches** *(case study, PoC 3, Tier 69, new 2026-09-16)* | [[poc3-racelab-knobs]] | **Tier 69: every one of the eleven parameters moved end to end**; only four reach the subsystem they claim to ↗ |
| ★★ **Rake and the duct area wired, and the knob layer criterion 2 needs** *(case study, PoC 3, Tier 70, new 2026-09-16)* | [[poc3-racelab-knobs-wired]] | **Tier 70: rake and duct area wired**, and the knob layer criterion 2 needs ↗ |
| ★★★ **The dashboard, and the coolant knob that finally reaches the march** *(case study, PoC 3, Tier 71, new 2026-09-16)* | [[poc3-racelab-dashboard]] | **Tier 71: the dashboard**, which showed at once that the coolant knob never reached the march; now it does ↗ |
| ★★★ **The fixed point the implicit step has, and the certified mode on it** *(case study, PoC 3, Tier 72, new 2026-09-16)* | [[poc3-racelab-certified-step]] | **Tier 72: the implicit step already existed**, so the certified mode (defect correction) runs on its fixed point ↗ |
| ★★★ **The certified mode on the screen, and what it costs over a march** *(case study, PoC 3, Tier 73, new 2026-09-16)* | [[poc3-racelab-certified-screen]] | **Tier 73: criterion 4 met**, the certified mode on the dashboard and what it costs over a march, with the learned mode shown refused and why ↗ |
| ★★★ **Showable: the bundle carrying the column it claims to** *(case study, PoC 3, Tier 74, new 2026-09-16)* | [[poc3-racelab-showable]] | **Tier 74: the bundle now carries the column it claims**, Tiers 58–73 added to a clone that had stopped at Tier 57 ↗ |
| ★★★ **Theory closure audit** *(new 2026-08-27)* | [[theory-closure-audit]] | **Is the theory closed? No**, but the holes sort into closed, importable and research, and only the last is research ↗ |
| ★★★ **End-to-end architecture spec** *(binding; new 2026-08-27)* | [[end-to-end-architecture-spec]] | **The binding specification of the whole system**: nine layers in dependency order, from expert declaration to emit ↗ |
| ★★★ **Interface transfer theory** *(binding; new 2026-08-27)* | [[interface-transfer-theory]] | **The interface transfer rules the spec needed**: one common interface space per seam with a prolongation per side, which unifies six of nine importable mechanisms ↗ |
| ★★★ **Plug-in composition theorems** *(binding; new 2026-08-27)* | [[plug-in-composition-theorems]] | **The three properties a plug-in architecture asserts**: closure (a matched subgraph is a legal expert, by the Schur complement), how the capability record composes, and substitution; plus an attribution theorem across nesting depth ↗ |
| ★★★ **The Atlas implementation** *(implementation record; new 2026-08-27)* | [[atlas-implementation]] | **The specification executed**: `atlas/`, nine layers, the seven-hypothesis envelope and the three-verdict compiler; a new case is declarations only ↗ |
| ★★★ **Tier 0 measured** *(measurement record; new 2026-08-27)* | [[tier0-measurements]] | **The measurement record from Tier 0 on**: the first numbers from a real expert, `reference.WindowNS`, and the first page refuted by one ↗ |
| ★★ **Atlas and standard DD theory** *(reading companion, new 2026-08-31)* | [[atlas-and-standard-dd-theory]] | **Where Atlas sits in the four classical domain-decomposition families**, for a reader working through Toselli & Widlund ↗ |
| ★★★ **End goal & the ladder to it** | [[f1-pathmap-and-end-goal]] | **The programme roadmap**: a coupled, differentiable model of an F1 car fast enough to search the design space, and the rungs to it ↗ |
| ★★★ **The case-study ladder to F1** *(revised plan, new 2026-08-30)* | [[case-study-ladder-to-f1]] | **The ladder rescheduled after six case studies**: the goal and claims unchanged, the order and the currency moved; the programme sat at rung 1–2 of 12 ↗ |
| ★★ **PoC 1a re-read on a second machine, and the ladder's hybrid revision** *(new 2026-09-01)* | [[poc1-retrospective-and-hybrid-roadmap]] | **PoC 1a re-run on a second machine**: two machines agree on the sign of the timing; the PoC showed end-to-end differentiability, not speed or multiphysics; and the ladder's hybrid revision ↗ |
| ★★ **CS-7 pickup — the scaling ladder** *(build brief, new 2026-08-30)* | [[cs7-scaling-ladder-pickup]] | **CS-7 build brief**: the self-contained pickup page for the scaling ladder ↗ |
| ★★★ **Scaling ladder — the summary** *(new 2026-08-31)* | [[case-study-scaling-ladder-atlas-0.1]] | **CS-7: one geometry grown from 1 to 24 coupled windows**, the first case study whose variable is the size of the graph ↗ |
| ★★★ **Reuse probe — is a certificate the expert's or the state's?** *(new 2026-08-31)* | [[case-study-reuse-probe-atlas-0.1]] | **CS-8: is a certificate the expert's or the state's?** The test of the foundation-model claim rather than the coupling claim ↗ |
| ★★★ **Thermal strain — is a volumetric coupling a bond?** *(new 2026-09-01)* | [[case-study-thermal-strain-atlas-0.1]] | **CS-9: the first co-located split**, and the port algebra's own extension procedure, which it failed ↗ |
| ★★★ **Seam placement — can an algorithm choose the cut?** *(new 2026-09-02)* | [[case-study-seam-placement-atlas-0.1]] | **CS-9★: can an algorithm choose where to cut?** The first test of automatic seam placement, an end-goal capability with no prior art found ↗ |
| ★★★ **Rocket b–c seam — the first rocket physics through Atlas, and the wall that was never listening** *(case study CS-21, PoC 4 rung R0, Tier 76, new 2026-09-17)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21, Tier 76: the first real rocket physics through Atlas**, the b–c seam wired to the build repo's solvers, and the wall that was never listening ↗ |
| ★★★ **The correction — two errors that nearly cancelled** *(case study CS-21 §10, PoC 4 rung R0, Tier 77, new 2026-09-17)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21 §10, Tier 77: two errors that nearly cancelled** in the published seam constant (later re-derived at Tier 86) ↗ |
| ★★★ **The sign structure E7 never read** *(case study CS-21 §12, framework change, Tier 78, new 2026-09-17)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21 §12, Tier 78: the passivity check read only one sign**, so two different failures reported the same number; now a rule ↗ |
| ★★★ **R0 complete — and what the other six agents cost to believe** *(case study CS-21 §13, PoC 4 rung R0 closed, Tier 79, new 2026-09-17)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21 §13: R0 complete**, all seven rocket agents stepping real solvers on their own meshes ↗ |
| ★★★ **R2's remaining half — the composed defect at 50:1, and the bound that breaks at it** *(case study CS-21 §14, PoC 4 rung R2, Tier 80, new 2026-09-17)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21: R2's second half**, the composed defect at 50:1 and the bound that breaks there ↗ |
| ★★★ **The trajectory agent, and the MECH bond with no null space to measure** *(case study CS-21 §15, PoC 4, Tier 81, new 2026-09-21)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21 §15: the trajectory agent**, and a `MECH` bond with no null space to measure ↗ |
| ★★ **W308 closed and W312 priced — an estimate wrong in both directions, and a seam removed by argument** *(case study CS-21 §16 and §12.4, PoC 3 audit, Tiers 82–83, new 2026-09-21)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **W308 closed and W312 priced**: an estimate wrong in both directions, and a seam removed by argument ↗ |
| ★★★ **The seams rebuilt, and a wall that passed mass** *(case study CS-21 §18, PoC 4, Tier 86, new 2026-09-23)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21 §18, Tier 86: the seams rebuilt (W323–W331)**, and a wall that passed mass ↗ |
| ★★★ **W334 — Tiers 76–85 re-derived, and what the old wall had been saying** *(case study CS-21 §18.8, PoC 4, Tier 86, new 2026-09-23)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **W334: Tiers 76–85 re-derived with the fixed wall**, and which earlier findings were the old wall's ↗ |
| ★★★ **The episode's residuals — what "correct" can mean here** *(case study CS-21 §19, PoC 4, Tier 87, new 2026-09-24)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21 §19, Tier 87: what "correct" can mean for the marched episode**: the solver's own verification, discrete balances, and distance from a finer run ↗ |
| ★★★ **The four fixes, the re-march, and a floor that was not what it looked like** *(case study CS-21 §20, PoC 4, Tier 88, new 2026-09-26)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **CS-21 §20: the four fixes (W337–W340), the re-march**, and a floor that was not what it looked like ↗ |
| ★★ **W343 — the d–g seam made two-way, and the floor it leaves** *(case study CS-21 §20.8, PoC 4, Tier 88 continued, new 2026-09-26)* | [[case-study-rocket-bc-seam-atlas-0.1]] | **W343: the d–g seam made two-way**, halving its loss, and the floor it leaves ↗ |
| ★★★ **The matched shrink, and a coarse competitor across coupled seams** *(measurement, W214 + W345, Tier 89, new 2026-09-26)* | [[matched-shrink-and-coarse-competitor]] | **W214, the matched shrink**: stability along the approach, not the rate at the fixed point, decides the calls. **W345**: a coarse classical competitor across coupled seams exists and is correct, but is not cheap ↗ |
| ★★★ **Classical decomposition against the monolith, by rotor count** *(measurement, W346, Tier 89, new 2026-09-26)* | [[decomposition-speed-by-rotor-count]] | **W346: classical decomposition against the monolith from 1 to 36 rotors**; only parallelism over cache-sized windows makes it faster ↗ |
| ★★ **PoC 1a — the spec** *(scoped demonstration target, opened 2026-08-31)* | [[atlas-proof-of-concept-1]] | **PoC 1a spec**: re-optimise a wind farm's layout and yaw with exact gradients through a graph of experts, and how it can fail ↗ |
| ★★★ **PoC 1a — the results** *(measurement record, new 2026-09-01)* | [[poc1-results-differentiable-design]] | **PoC 1a results**: the gradient reaches its tolerance in 24–26 rollouts, where CMA-ES from the same start spent 700 and 928 and never reached it; one gradient costs 6–9 forward evaluations ↗ |
| ★★★ **PoC 1a — the frozen-expert column** *(measurement record, new 2026-09-02)* | [[poc1a-frozen-expert-results]] | **PoC 1a, frozen-expert column**: every window Poseidon-T, with the adjoint through all 20.8 M frozen parameters ↗ |
| ★★ **PoC 1a — the interactive demo** *(runnable, new 2026-09-01)* | `atlas/demo/` | **The composed march, live in a browser** (`python scripts/w112_farm_demo.py`, or the `poc1-windfarm-demo` branch): optimise the layout by gradients through the live march, and check it against the undivided solve side by side ↗ |
| ★★★ **PoC 2 — the front-wing assembly** *(measurement record, new 2026-09-05)* | [[poc2-frontwing-results]] | **PoC 2: two surface seams live at once on the front wing**, ride height and bending, and a constrained design search through both ↗ |
| ★★★ **PoC 2 — the certification demo** *(runnable and packaged, new 2026-09-05)* | `atlas/demo_frontwing/` · [[poc2-demo-and-novelty]] | **PoC 2 demo: the compiler's own verdict on screen**, a one-command bundle for a reader who has not read this vault ↗ |
| ★★ **What is actually novel here, in one screen** *(positioning, new 2026-09-05)* | [[poc2-demo-and-novelty]] | **The demo's novelty argument in one screen**: four of five points are not novel, and say so with citations ↗ |
| ★★★ **What PoC 2 actually shows — the audit** *(critical assessment, new 2026-09-08)* | [[poc2-novelty-audit]] | **The adversarial reading of PoC 2**: one claim survives, and it is narrower than the demo implies ↗ |
| ★★ **The front wing's own constants, and what they did not buy** *(measurement, new 2026-09-08)* | [[gap-worklist]] Tier 34 · [[poc2-novelty-audit]] §3.1 · `out/w157/w157.json` | **The front wing's own constants measured**: the provenance objection removed, and the compiler still refused until Tier 39's checker fixes ↗ |

### `case-study-wind-farm-wake/` (with `implementation/`)

| Concept | Page | Description |
|---|---|---|
| ★★ **Buildable substitute case study** | [[case-study-wind-farm-wake-2d-atlas-0.1]] | **2-D wind-farm wake interaction**: the engineering scenario that survives the no-new-experts, no-new-data screen ↗ |
| ★★ **Wind farm — full specification** | [[spec-wind-farm-wake-atlas-0.1]] | **The binding wind-farm spec**: domain, 8 agents, 15 typed edges, the governing equations, and §7's coupling and conservation rules ↗ |
| ★★ **Agent graph — figure** | [[wind-farm-agent-graph-figure]] | **The agent graph as a figure**, with both schedules; drawing it exposed overlapping regions and a contradictory agent table ↗ |
| ★★★ **Implementation guide** | [[impl-wind-farm-guide]] | **The wind-farm implementation guide**: phases W0–W6 with gates, executable by an agent holding only it and the spec ↗ |
| ★★ **Implementation log** | [[wind-farm-implementation-log]] | **The wind farm's append-only build record** from 2026-08-19, phase by phase from W0, each entry marked built, fixed, measured or blocked |
| ★★★ **Results — W0–W3** | [[results-w0-w3-wind-farm]] | **Results W0–W3, built 2026-08-20: the go/no-go passed**; a frozen Poseidon-T keeps the momentum budget closed across a rotor ↗ |
| ★★★ **Results — W6–W11** | [[results-w6-w11-wind-farm]] | **Results W6–W11, negative with a cause**: the frozen checkpoint inverts the array efficiency, turbine 2 out-producing turbine 1 ↗ |
| ★★★ **Results — the $N$-sweep** | [[results-n-sweep-wind-farm]] | **The N-sweep: 8 to 26 agents and the turbine ratio moves under 1%**, composition error bounded in graph size ↗ |
| ★★★ **Wake array — the summary** *(new 2026-08-30)* | [[case-study-wake-array-atlas-0.1]] | **CS-6, the wake array in one page**: three turbines over six frozen Poseidon-T windows, the first real geometry ↗ |
| ★★ **GPU rollout runbook** | [[vast-ai-windfarm-runbook]] | **How to run the coupled rollouts on a rented vast.ai GPU** and how to pick the box: the wind farm is memory-bandwidth bound in float64 ↗ |

### `case-study-rocket-ascent/` (its `implementation/` has its own section below)

| Concept | Page | Description |
|---|---|---|
| ★★ **The marched rocket's seams — picture, mesh, or coupler?** *(audit note on W321, not a tier, new 2026-09-23)* | [[rocket-episode-seam-audit]] | **Audit of the marched rocket's seams (W321)**: picture, mesh or coupler? Three layers, only the first presentation ↗ |
| ★★★ **Rocket ascent episode — self-contained pickup** *(build brief, new 2026-09-26)* | [[rocket-episode-pickup]] | **Rocket episode pickup**: the one page a session with no context needs to continue the rocket, including the two repositories and the four code fixes ↗ |
| ★ **First instantiation — plan of action** | [[case-study-rocket-ascent-2d-atlas-0.1]] | **2D rocket ascent** — full agent/edge graph, 4-expert table, Phase 0–5 plan (scope/data → scaffold → experts → wiring → validation → expansion); explicit non-goals; status: planned, not built |

### `case-study-ground-effect/`

| Concept | Page | Description |
|---|---|---|
| ★★★ **Ground effect — a moving interface, and the first design parameter** *(new 2026-09-03)* | [[case-study-ground-effect-atlas-0.1]] | **CS-10: the first interface whose geometry depends on the solution**, a moving seam and the first design parameter ↗ |

### `case-study-brake-thermal/`

| Concept | Page | Description |
|---|---|---|
| ★★★ **Brake thermal — a bound for the multirate lag, and a horizon for a gradient** *(new 2026-09-04)* | [[case-study-brake-thermal-atlas-0.1]] | **CS-11: a bound for the multirate lag**, the first σ term that is a function rather than a value, and a horizon for a gradient ↗ |

### `case-study-wing-fsi/`

| Concept | Page | Description |
|---|---|---|
| ★★★ **Wing FSI — two-way coupling across a new governing family** *(new 2026-09-04; **verified 2026-09-04**)* | [[case-study-wing-fsi-atlas-0.1]] | **CS-12: two-way fluid–structure coupling across a new governing family**, two agents solving different continuum problems on a surface ↗ |

### `case-study-rbc-decomposition/`

| Concept | Page | Description |
|---|---|---|
| Synthetic control for the same test | [[case-study-rbc-decomposition-atlas-0.1]] | **Domain-decomposed 2-D Rayleigh–Bénard**: the interface layer tested with no new data or experts, the single-expert control ↗ |


## ★★ Atlas 0.1 Implementation (rocket case study) — `concepts/Atlas 0.1/case-study-rocket-ascent/implementation/`
**The executable build layer for Atlas 0.1.** One self-contained spec per phase, each with **intuition → theory → implementation** sections plus acceptance tests and pitfalls, written so an agent holding only that page can build the phase correctly. Code home: the **`atlas-0.1` branch** of [github.com/nonidino/physics-foundation-model](https://github.com/nonidino/physics-foundation-model) (cut from `noether-1.1` 2026-08-09), as the `src/atlas/` package parallel to `src/noether11/`. **Phase 1 is built** — M0 + M3 green, 1,114 tokens, 3,832 typed edges, 8.35 M scaffold params.

| Doc | Page | Description |
|---|---|---|
| ★ Master plan | [[00-atlas-0.1-implementation-plan]] | **The rocket master plan**: repo layout, phase graph, milestones M0–M7, and the binding conventions (frame, agents, edges, time steps) ↗ |
| Phase 0 — Scope & data | [[impl-atlas-0.1-phase0-scope-and-data]] | **Phase 0, built 2026-08-09**: classical solvers as teacher and judge, analytic oracles, and the sweep; M1 10 of 11 green, M2 blocked on compute ↗ |
| Phase 1 — Native scaffold | [[impl-atlas-0.1-phase1-scaffold]] | **Phase 1, built 2026-08-09**: the native scaffold, plumbing before physics; the build falsified the agent-isolation test's premise ↗ |
| Phase 2 — Experts | [[impl-atlas-0.1-phase2-experts]] | **Phase 2, the experts**: the family cut, `reacting_flow` from scratch, two experts bootstrapped from Poseidon, and per-expert gates ↗ |
| Phase 3 — Integration | [[impl-atlas-0.1-phase3-integration]] | **Phase 3, integration**: dispatch and blending, multi-rate subcycling with the exchange-cadence rule, and a characteristic-upwinded conservation projection ↗ |
| Phase 4 — Validation | [[impl-atlas-0.1-phase4-validation]] | **The experiment.** Falsification table stated up front; horizon-resolved metrics; four physical-milestone checks (thrust-vs-altitude, max-Q timing, nozzle separation, shock cells); 7 ablations incl. **A4 declared-vs-dense graph** and **A2 conservation-off**; post-phase decision rule |
| Phase 5 — Expansion | [[impl-atlas-0.1-phase5-expansion]] | Scenario 2 = offshore wind farm; reuse levels R1/R2/R3; **regression suite frozen first**; the long-range direction-dependent wake edge; fast/slow split for decade-scale fatigue; RL edge instantiation unlocked (with its stability-before-pooling ordering constraint); versioning to Atlas 0.2 |
| ★ **Corpus completion plan** | [[impl-atlas-0.1-corpus-completion-plan]] | **The ordered work plan for M2**: six irreversible decisions, three new modules and two new acceptance gates ↗ |
| ★ **Compute & training budget** | [[impl-atlas-0.1-compute-and-training-budget]] | **Cross-phase resource plan**: Atlas 0.1 is a data-generation problem (about 2,400 CPU-core-hours against 160 GPU-hours), and route B′ past the M2 wall ↗ |
| Progress log | [[atlas-0.1-implementation-log]] | Append-only build tracker + portion status board + action items. **First build entries 2026-08-09**: Phase 1 as-built report, the token-graph-reachability finding, the $d\!-\!g$ / undeclared $d\!-\!f$ interface finding |

## ★★ Atlas 0.1 Outcome (proposal evidence) — `concepts/Atlas 0.1/atlas-0.1-outcome/`
**What Atlas 0.1 has established, compiled 2026-09-26 after Tier 88 and revised after Tier 89, as the evidence base for a funding proposal** to build learned experts native to domain decomposition (fast to converge at a seam, accepting non-rectangular geometry). A synthesis: no new measurement, every number sourced. Holds five proposal claims against the record and names what is not done. Also holds `interactive/` (the rocket episode, scaling ladder and wake-array viewers, and the evidence board) and `figures/`.

| Page | Link | Description |
|---|---|---|
| ★★★ **Outcome hub** | [[00-atlas-0.1-outcome]] | **The five claims with blunt verdicts, revised after Tier 89**: C1 classical decomposition faster only through parallelism; C2 learned experts partly couple; C3 to C5 as recorded ↗ |
| C1 — decomposition vs monolith | [[outcome-c1-decomposition-vs-monolith]] | **C1, revised at Tier 89**: parallel decomposition beats the undivided solve 3.4–5.7× from 5 to 21 rotors; serial decomposition never does ↗ |
| C2 — learned experts in the loop | [[outcome-c2-learned-experts-in-the-loop]] | Stable coupling, adjoint through the checkpoint, rung 3's compile; defect correction's certified convergence (an adversarial map lands $6.15\times10^{-9}$ from the classical answer); the bias (OP-3, $P_2/P_1 = 1.0162$) and the 13-vs-69-call contribution |
| C3 — learned speed at scale | [[outcome-c3-learned-speed-at-scale]] | Per-window crossover $4.90 \to 0.253$; composed column $3.15$/$3.64\times$ faster than the monolith (hardware-dependent); the $2\times$-coarse classical solver $9$–$17\times$ cheaper still; off-lead on the car, $0.067\times$ |
| C4 — modular multiphysics | [[outcome-c4-modular-multiphysics]] | The compiler, five ports, zero coupling code; every multiphysics graph built (CS-12 to CS-21, PoC 2, RaceLab); the per-window switch and live knobs; the limits |
| C5 — requirements for DD-native experts | [[outcome-c5-requirements-for-dd-native-experts]] | **Not done**, and the measured specification S1–S12: geometry ($\le 33.4\%$ of the body-fitted car reachable), step (error $4.3\times$ below native), interface input, bounded dependence, **slow-mode Jacobian** ($\rho = +0.83$ vs accuracy's $+0.69$), cost, reference class, licence |
| Evidence ledger | [[outcome-evidence-ledger]] | Every quotable number with its conditions and source, by claim |
| C4 — the path to declarative cases and a decomposition GUI *(plan, Tier 89)* | [[outcome-c4-path-to-declarative-cases]] | **C4, a plan**: what a case module does today layer by layer, and a four-step path to declarative cases and a decomposition GUI ↗ |
| Showcase library — the plan *(plan, 2026-09-28; all six steps built by 2026-09-29: the geometry section, the style-A runner, styles B–D, all eight cases, the compile and the gallery; the page rebuilt 2026-09-29; starting values 2026-09-30)* | [[showcase-library-plan]] | **The workbench's classical case library**, all six steps built by 2026-09-29: drawn geometry and every family running on it ↗ |
| Showcase gallery *(showcase, not research record; measured 2026-09-29)* | [[showcase-gallery]] | **Every workbench example compiled and run, one row each**, with its verdict, checks and runtime; every check passed ↗ |
| Demos & artifacts | [[outcome-demos-and-artifacts]] | Published pages (evidence board, rocket episode, scaling ladder), local viewer copies, the three runnable PoC demos with commands and branches, figures, and each item's caveat |

## ★★ Atlas 0.1 Proposal (the four workstreams) — `concepts/Atlas 0.1/atlas-0.1-proposal/`
**The plan for finishing the proposal, written 2026-09-30 in a planning session: no code changed, nothing run, no new number.** Four workstreams from the owner's brief, each evaluated, with the owner's decisions O1–O9 (made the same day), draft checks to register, the website's outline, and the prompts for the three chats that run at the same time. Subfolders `demo/`, `website/`, `expert-architecture/` and `formal-proofs/`. Tracker rows W349–W354 in [[gap-worklist]].

| Page | Link | Description |
|---|---|---|
| ★★★ **Proposal hub** | [[00-proposal-workstreams]] | **The proposal hub**: a verdict per workstream, the idea that ties them together, the order of the chats, and the owner's decisions ↗ |
| Demo: header, fork, holes *(items 1.1–1.3)* | [[demo-finish-plan]] | **Demo items 1.1–1.3**: the type selector, the forked river and holes ↗ |
| Demo: a fast example per type *(item 1.4)* | [[demo-fast-examples-plan]] | **Demo item 1.4, a fast example per type**: the four genuine mechanisms for speed, and each family's ceiling ↗ |
| Demo: the one learned case *(item 1.5)* | [[demo-learned-case-plan]] | **Demo item 1.5, the one learned case**: the 12-rotor wind farm, its five arms, and pricing a random network before any data exist ↗ |
| Website: the plan *(task 2)* | [[proposal-website-plan]] | **The website plan**: two readers, five beats, nine sections, and a claims pipeline with no number typed by hand ↗ |
| Website: the evidence register | [[website-evidence-and-citations]] | Every number the site may show: Atlas's own (A1–A10, measured) and published results (classical L1–L6, neural N1–N12), each with its status (measured, search summary, or unverified) and wording |
| Website: the outline *(after the owner's decisions, 2026-09-30)* | [[website-outline]] | **The site section by section** after the owner's decisions: topics, the statistics to show, voice, style, motion and image prompts ↗ |
| The three chat prompts | [[proposal-chat-prompts]] | **The prompts for the three chats run side by side**: the demo, the architecture and the proofs, and how they share one folder ↗ |
| Website: the vision | [[vision-scenarios-and-image-prompts]] | The whole race car, a wildfire forecast and steered, the agentic Moon base: each story, its agent graph by port type, what exists today toward it, and **first-draft image prompts** (a clean plate, an assembled variant, details) under one style bible; overlays drawn in code |
| Website: the design notes *(website chat, 2026-10-02)* | [[website-design-notes]] | **The three reference sites read from their source** (HTML, CSS, scripts), and the design notes drawn from them ↗ |
| Website: the image prompts *(website chat, 2026-10-02)* | [[website-image-prompts]] | **The illustration brief**: four required images in two crops each, thirteen optional, one style bible ↗ |
| Website: the launch checklist *(website chat, 2026-10-02)* | [[website-launch-checklist]] | **What waited on the owner before launch**, in order; the site went live on 2026-10-02 ↗ |
| ★★★ **The chart operator** *(task 3, architecture v0)* | [[chart-operator-architecture]] | **Version 0 of the chart operator**: topology to the decomposition, geometry to a chart, physics to the operator, agreement by wave variables ↗ |
| The chart operator: training and cost | [[chart-operator-training-and-cost]] | **Data, stages 0–3, objective, gates and a labelled cost model** for the chart operator; from scratch first, fine-tuning as a Stage 2 comparison ↗ |
| Prior art, 2021–2026 | [[dd-neural-prior-art-2026]] | **Neural Schwarz on any geometry is already published**; what is left to claim, N1–N6, each to confirm against the full papers ↗ |
| ★★ **The chart operator: decisions** *(architecture chat, from 2026-10-01)* | [[chart-operator-design-decisions]] | **The running record of the chart-operator design conversation**: what was decided, when, and on what evidence ↗ |
| ★★★ **The chart operator, version 1** *(confirmed by the owner 2026-10-01; supersedes the v0 row above)* | [[chart-operator-architecture]] | **Version 1, confirmed 2026-10-01**: experts return constraint modes and the host forms their energy Gram, positive semidefinite for every network output, with a label-free objective ↗ |
| ★★ **The chart operator in general form** *(under discussion, 2026-10-10)* | [[chart-operator-general-form]] | **The general form, under discussion**: the guarantee rests on exact traces and the host holding the physics, not on linearity; guarantee classes G0–G3 ↗ |
| ★★ **The chart-operator family catalog** *(under discussion, 2026-10-10)* | [[chart-operator-family-catalog]] | **The family catalog, under discussion**: twelve families covering about forty PDEs, eight kinds of interaction, unit-square speedups and an architecture per family ↗ |
| ★★★ **The public document, version 2** *(2026-10-12; replaces version 1 on the website)* | [[chart-operator-architecture]] · [[chart-operator-general-form]] · [[chart-operator-family-catalog]] | **Version 2, public since 2026-10-12**: version 1, the general form and the family catalog in one document written intuition first; version 1 archived beside it ↗ |
| ★★★ **The chart operator, Stage 0** *(built, trained and judged once, 2026-10-18)* | [[chart-operator-stage-0]] | **CO-D built and judged once against gates registered before training**: the coupling's guarantees held to rounding, the modes did not learn in eleven arms (\$4.06 of rented GPU), so E1, E4 and KT fail; the hat modes' broken sum dominates the error ↗ |
| Coupling cost and complexity | [[coupling-cost-and-complexity]] | **Coupling cost measured with exact local solves**: rounds by method on 8 × 8 pieces (none for the superelement), and a monolithic direct solve winning at these 2-D sizes ↗ |
| ★★★ **Formal proofs** *(task 4)* | [[formal-proofs-plan]] | **The theorem inventory T1–T23** with novelty and Lean tier: a finite-dimensional core machine-checkable in weeks, and what is cited instead ↗ |
| Formal proofs: the record | [[formal-proofs-record]] | **What is machine-checked**: 105 theorems in 26 Lean files, no `sorry`, standard axioms only, one row per theorem; T3 for restricted additive Schwarz is false as first written ↗ |
| Formal proofs: the implementation plan | [[formal-proofs-implementation-plan]] | **The route to machine-checking**: which statements go to Lean and in what batches, the toolchain and its measured set-up cost, what stays a paper proof or a citation and why (§5), and effort estimates. Batches 1 to 6 are done; batch 7 is not started |

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
| [[handoff-atlas-0.1-corpus-generation]] | **Session handoff prompt** for executing M2 on the code repo: thesis, build state, the unmet oracles and their causes, settled decisions and costs ↗ |

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
- New reader → [[00-start-here]]. Looking for code → [[code-and-papers]]. Build history → [[log]].

---

*This index is maintained by the wiki system. Update after each new source ingestion.*
