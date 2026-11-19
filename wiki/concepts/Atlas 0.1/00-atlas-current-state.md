# Atlas 0.1: current state

**Type:** Hub — **current-state snapshot** of the Atlas 0.1 track (folder: `Atlas 0.1/`)
**Status:** written 2026-11-18 in the wiki lint and compaction pass, from the pages linked below; no new measurement. Updated the same day for the chart operator's plan after Stage 0 (§4, §6), and for its Phase 1, built and measured (§4, §6); on 2026-11-19 for its Phase 2 and the expert store (§4, §6). **Update this page whenever a hub's status line changes.** Where it and a linked page disagree, the linked page wins and this page is stale.
**Purpose:** the one page to read before any Atlas build, design, decision or brainstorm, so the history (89 tiers, five PoCs, the archived worklist) need not be read to learn what is true now. For which other pages to read per task, see [[context-router]].
**Related:** [[00-atlas-0.1-overview]] · [[00-atlas-0.1-outcome]] · [[00-proposal-workstreams]] · [[gap-worklist]] · [[integrated-ui-plan]]

---

## 1. Atlas in three sentences

Atlas splits a simulation into **agents** (pieces of the domain), each run by an **expert** chosen by governing-equation family, joined by typed **ports** (effort–flow pairs whose product is power, [[port-algebra-atlas-0.1]]). A **compiler** reads the agents' declarations and either builds the coupling or refuses it with a named reason, and every run is checked against a **master error bound** ([[master-error-bound]]): per-step defect = agent infidelity + transmission infidelity + solve incompleteness. The thesis changed by measurement: the framework built is **classical domain decomposition with a certifying compiler**; a learned expert native to it (the **chart operator**) is the proposal's ask, designed and tested in Stage 0, not yet earning its place.

## 2. What exists, and where

| piece | state | read |
|---|---|---|
| **Framework** `atlas/` | built: nine layers, five ports, the three-verdict compiler, a case = declarations only | [[atlas-implementation]] · [[end-to-end-architecture-spec]] (binding) |
| **Studio** `atlas/studio/` | **the default and only UI** (owner, 2026-11-18). P0–P6 built 2026-11-16/17: draw, choose physics, cut into pieces, run, read guarantees and records. The Panel workbench page is retired (its tests drive the studio's protocol) | [[integrated-ui-plan]] §12 · `atlas/studio/README.md` |
| **Case library** | eight showcase cases, all compiled and run | [[showcase-gallery]] |
| **Proofs** `lean/` | 105 theorems in 26 Lean files, no `sorry`, standard axioms only; batches 1–6 done, batch 7 not started. T3 (restricted additive Schwarz) false as first written | [[formal-proofs-record]] |
| **Website** `site/` | live since 2026-10-02 at nonidino.github.io/Atlas; every number read from a record by `scripts/site_claims.py` | [[proposal-website-plan]] · [[website-evidence-and-citations]] |
| **Chart operator** | design v2 public (2026-10-12); Stage 0 and its rework run and judged (below) | [[chart-operator-architecture]] · [[chart-operator-general-form]] · [[chart-operator-family-catalog]] |
| **Tests** | `python scripts/run_suite.py`; full suite 3021 passed, none failed, after the studio's P6 (2026-11-18) | [[log]] |

## 3. What has been measured: the five claims

From [[00-atlas-0.1-outcome]] (revised after Tier 89); numbers there, with conditions, in [[outcome-evidence-ledger]].

| claim | verdict |
|---|---|
| **C1** decomposition faster than the monolith | **only through parallelism**: $3.4$–$5.7\times$ from 5 to 21 rotors on several cores; serial decomposition never faster; never more accurate than a same-discretization monolith ([[outcome-c1-decomposition-vs-monolith]]) |
| **C2** learned experts couple with classical ones | **partly**: frozen experts couple stably; defect correction converges to the classical answer whatever the learned map; the frozen expert's own answer is biased ([[outcome-c2-learned-experts-in-the-loop]]) |
| **C3** learned faster at scale | **one case, with conditions**: per-window cost $4.90\to0.253$ of classical from 1 to 24 windows; a $2\times$-coarse classical solver is cheaper still ([[outcome-c3-learned-speed-at-scale]]) |
| **C4** modular multiphysics | **demonstrated**: the compiler, five ports, zero coupling code across CS-12 to CS-21, PoC 2 and RaceLab ([[outcome-c4-modular-multiphysics]]) |
| **C5** requirements for DD-native experts | **not done**; specification S1–S12 measured ([[outcome-c5-requirements-for-dd-native-experts]]) |

## 4. The chart operator: where the learned expert stands

- **Stage 0** (2026-10-18, W355): CO-D (2-D diffusion on curved four-sided pieces) built and judged once. The coupling's guarantees held to rounding; the modes did not learn; E1, E4 and KT failed ([[chart-operator-stage-0]]).
- **Kernel-exact forms** (W356–W358, to 2026-11-08): output forms that keep a uniform temperature exact; trained, E4 still failed. **Rule R2's hybrid met E4 and the price bar with nothing learned** on square-first layouts ([[chart-operator-kernel-exact-forms]]).
- **Stage 0 reworked** (W363, done 2026-11-12): a two-level, coarse-exact expert. Untrained (TL0) the coarse level removes 87–96% of the coupled error above the floor; trained (TL) the network adds nothing coupled: E4 on 4 of 6, NC not met, 1.05 of a classical solve against TL0's 0.84 ([[chart-operator-stage-0-rework]] §19–§20).
- **So, today:** the CO-D expert stays the classical hybrid H2r; **W362 (pieces that are not squares) is open**. Design rationale and owner decisions: [[chart-operator-design-decisions]].
- **The plan after Stage 0** (agreed 2026-11-18): **the coolant case comes next**, CO-T on laminar cold plates (Péclet at most 200 per piece), in phases that stop before training unless the classical options fail and a perfect network would beat them at equal price. The dataset is frozen before training; training is on a rented box, 3 seeds, 37,500 steps; every trained model gets a card in an expert store the studio reads. Then CO-E, the bond and the biharmonic network classically on TL0; **Stage 0b deferred**; Helmholtz out ([[chart-operator-stage-0b-and-1-plan]]).
- **Its Phase 1 is built** (2026-11-18, W364; plan §12). The owner chose model (a), side walls only. The classical CO-T expert (Petrov–Galerkin, adjoint modes by reversing the flow), `ADVEC` seams and the studio's CO-T path are in place. The seams are **stable at Pe 200** by six criteria fixed before the run: whole traces equal the monolithic solve to $5\times10^{-11}$, every piece is passive under the `ADVEC` split (the `THERM`-only pairing is not, about $-0.18$ at Pe 200), the coupled error equals the 16-mode oracle's (ratio 0.95–1.00, no growth along 8 pieces), and the modes' errors enter quadratically. The 16-mode floor is 0.1–0.3% on a channel and a bend, but 6–13% on the steady plate. Nothing is learned yet.
- **Its Phase 2 ran** (2026-11-19, plan §13): **the stop rule is not met.** On six development problems (three plates with Channel-tool channels, Pe 20 and 110–180), no classical option meets E4 (the owner's bar: the classical floor plus CO-D's margins) on all six under the 0.90 price bar. Finer cutting meets E4 on all six at 1.02 of the classical solve. TL0 at $n_c=32$ meets it at 0.34–0.60 on the fast tier, but no coarse size below the grid works at $n=32$. The multiscale bases fail at high Péclet (29–42%). Built on the way: the channel cut (curved channels cut along their walls, charted by the existing TFI, Laplacian and conformal tools), the flow on any channel, and Danckwerts' inlet (a held inlet had pinned the plate and inflated a floor from 0.3% to 31%). **The safety switch as built costs about one classical piece solve**, so Phase 3 runs the oracle ceiling first.
- **The expert store exists** (`out/experts/`, plan §8): 22 backfilled cards (Stage 0's eleven arms, the forms and prior runs, TL-s0/s1), every one's weights present and matching on this machine (checked live: a clone without the git-ignored weights offers none); the studio reads the catalog instead of its hand-written list.

## 5. Case studies and PoCs, one line each

| study | what it settled | page |
|---|---|---|
| PoC 1a, wind farm | exact gradients through a graph of experts: 24–26 rollouts against CMA-ES's 700+ | [[poc1-results-differentiable-design]] |
| CS-6/7, wake array, scaling ladder | frozen Poseidon-T windows couple stably, 1 to 24 windows | [[case-study-wake-array-atlas-0.1]] · [[case-study-scaling-ladder-atlas-0.1]] |
| CS-9 to CS-14 | thermal strain, seam placement, ground effect, brake thermal, wing FSI, cooling loop, powertrain | [[case-study-ladder-to-f1]] |
| PoC 2, front wing | two surface seams live, constrained design search; one narrow claim survives audit | [[poc2-frontwing-results]] · [[poc2-novelty-audit]] |
| PoC 3, RaceLab (Tiers 51–74) | a car as a graph on body-fitted overset grids, a classical/learned switch, certified mode, a bundle | [[case-study-racelab-graph-atlas-0.1]] → [[poc3-racelab-showable]] |
| PoC 4, rocket (CS-21, Tiers 76–88) | first real rocket physics through Atlas; a leaking wall found and fixed; Tiers 76–85 re-derived | [[case-study-rocket-bc-seam-atlas-0.1]] · [[rocket-episode-pickup]] |
| Tier 89 | parallel decomposition beats the monolith by rotor count; matched shrink; coarse competitor | [[decomposition-speed-by-rotor-count]] · [[matched-shrink-and-coarse-competitor]] |

## 6. Open fronts

- **The coolant case** (CO-T, W364) per [[chart-operator-stage-0b-and-1-plan]]: Phases 1 and 2 done (§12, §13), the stop rule not met; **Phase 3's oracle ceiling fails** (§14, 2026-11-19): a perfect network with the switch costs 3.35 times the classical solve, 1.19 with none, against a bar of 0.90. On $n=32$ pieces the forward passes alone cost more than the classical solve. **The owner amended the plan the same day (§15):** price becomes a reading, not a gate; the gates are accuracy (E4; CL-A, meeting E4 where TL0 at the same coarse size misses it) and capacity (an overfit probe on the hardest pieces, Pe near 200 at $n=128$, before any dataset). The safety switch stays, unpriced until an $O(N)$ estimator exists. **Phase 3 resumes** with the capacity probe. Stage 0b deferred; **W362** (non-square pieces).
- **Formal proofs batch 7** ([[formal-proofs-implementation-plan]]).
- The remaining open W-items: 173 open at the board's generation (2026-10-17). Read the board in [[gap-worklist]], never the archives, and grep for a W-number instead of reading the page (§7).
- Deferred design problems with their causes: [[open-problems-atlas-0.1]].
- The 2026-11-18 lint brought five stale board rows up to date with the owner's approval: W24, W25 and W349 done, W351 closed (the learned case failed G5), W353 in progress (proof batches 1–6 done). See the last section of [[gap-worklist]].

## 7. Rules in force (decisions a new session must not undo)

- **The studio is the only UI**: any feature with a page goes into `atlas/studio/` with a protocol handler, a view, a `tests/test_studio_*.py` test and a page walk ([[integrated-ui-plan]] §12).
- **Registered evaluations are pinned**: modules hashed by a registered evaluation are called, never edited; gates are registered before any run and judged once ([[chart-operator-stage-0-rework]]).
- **Records, not copies**: numbers on the site and in the studio are read from `out/`, never retyped ([[proposal-website-plan]]).
- **Lean**: the owner approves every statement before it is proved; build with `scripts/lean_build.py`, never `lake` inside OneDrive ([[formal-proofs-implementation-plan]]).
- **Public repository**: no secrets, no unlicensed weights; NeuberNet only when present locally ([[case-study-bounded-donor-atlas-0.1]]).
- **Many wiki pages are read by tests** (`tests/test_tier*.py`, `tests/test_chartop_*.py`, `scripts/build_racelab_bundle.py` join their paths and assert their text): do not move or rename a page under `Atlas 0.1/` without grepping `tests/` and `scripts/` for its file name.

## 8. What is history (read only for provenance)

- **The 2026-08-08 thesis pages** (learned tokenizer, edge generation, U-Net of experts, training-and-bootstrap, global fields): the original learned-model design; the as-built framework is classical composition ([[end-to-end-architecture-spec]]). Hub: [[00-atlas-0.1-overview]].
- **The rocket's first plan** (archived to `wiki/archive/rocket-ascent-first-plan/`, Phases 0–5 on the old build repository, M2 blocked on compute): replaced by CS-21 from Tier 76.
- **The Panel workbench and its demo plans** ([[showcase-library-plan]], [[demo-finish-plan]], [[demo-fast-examples-plan]]): their cases and installer survive in the studio; the page does not.
- **Chart operator v0 and v1**: folded into version 2 ([[chart-operator-architecture]]).
- **Worklist Tiers 0–75**: [[gap-worklist-tiers-0-37]], [[gap-worklist-tiers-38-75]]; the board in [[gap-worklist]] carries every item's latest status.
- **Noether 1.0 and 1.1**: dormant since 2026-07-19 (the last Noether 1.1 build entry); Atlas began as a parallel track on 2026-08-08 and has been the only active track since. Hubs: [[00-initial-model-overview]], [[00-noether-1.1-overview]].
