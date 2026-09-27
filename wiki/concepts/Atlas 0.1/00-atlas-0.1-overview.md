# Atlas 0.1: Overview & Thesis

**Type:** Core Concept — Model Overview / Hub
**Related Concepts:** [[pfm-purpose-and-direction]], [[regime-moe-architecture]], [[incremental-transfer-roadmap]], [[00-noether-1.1-overview]], [[mixture-of-experts]], [[pfm-interface-design]], [[case-study-rocket-ascent-2d-atlas-0.1]]

---

## Why "Atlas"

A differentiable manifold is covered by an **atlas**: a collection of local coordinate charts, each simple on its own, glued together by **transition maps** that keep the whole thing globally consistent even though no single chart describes the whole space. That is a near-literal description of this architecture: **agents are charts** (each one a locally simple physical domain, handled by a regime-appropriate expert), **typed edges are transition maps** (the declared interface contract — five **ports**, each an effort–flow pair whose product is power, per [[port-algebra-atlas-0.1]] — saying how two charts must agree where they overlap), and **conservation constraints are the consistency condition** that keeps the composition from drifting apart at the seams.

The name is also a deliberate contrast with [[00-noether-1.1-overview]]. Noether 1.1's thesis is a single backbone that *discovers* conservation as an emergent structural property ([[discovered-conservation-1.1]]). Atlas's thesis is a graph of specialized local experts where conservation is *imposed* at declared boundaries, not discovered — composition and interface consistency are Atlas's forte, not conservation discovery. See [[conservation-as-constraint-atlas-0.1]] for the direct comparison.

**This does not deprecate or change the status of Noether 1.1.** Both are documented, parallel architecture tracks — see [[implementation-log]] for Noether 1.1's active build. Atlas is a new track, starting from a concrete first instantiation ([[case-study-rocket-ascent-2d-atlas-0.1]]) rather than a from-scratch general build.

---

## Provenance

Atlas is the concrete architecture that emerged from the 2026-08-07/08-08 design conversation chain: [[regime-moe-architecture]] (the first-principles regime survey and the case for expert partitioning) → [[pfm-purpose-and-direction]] (the multi-agent deployment target and worked examples) → [[incremental-transfer-roadmap]] (bootstrap-from-existing-weights build strategy) → the rocket-ascent worked example → the agent/typed-edge graph formalization → the U-Net-of-experts hierarchy → this page. Every design decision below has a specific worked-example justification traceable through that chain; none of it is abstract architecture-astronomy.

---

## End-to-end forward pass

$$
\underbrace{(\text{geometry}, \{\text{agents}\}, \{\text{typed edges}\}, \Delta t)}_{\text{declared input, Stage 1}} \;\longrightarrow\; \text{Tokenizer} \;\longrightarrow\; \text{Edge Instantiation} \;\longrightarrow\; \text{Typed Experts} \;\longrightarrow\; \text{U-Net pool/unpool} \;\longrightarrow\; \text{Decoder} \;\longrightarrow\; \text{Conservation constraint} \;\longrightarrow\; \hat{x}_{t+\Delta t}
$$

1. **Per-agent tokenization** ([[graph-tokenizer-atlas-0.1]]) — each agent's field state, over its own declared geometry, is encoded independently.
2. **Edge instantiation** ([[edge-generation-atlas-0.1]]) — declared typed edges (heat/pressure/stress/fluid/conservation) are materialized into token-level connections at shared interface geometry.
3. **Typed expert routing** ([[expert-library-atlas-0.1]]) — each agent/region is processed by the expert matching its governing-equation family, not by one expert per edge-type label.
4. **U-Net pooling hierarchy** ([[unet-hierarchy-atlas-0.1]]) — fine agent-local computation is progressively coarsened toward a global/bulk representation and then refined back down with skip connections, with expert choice made at each level.
5. **Decode** — per-agent field reconstruction.
6. **Conservation enforcement** ([[conservation-as-constraint-atlas-0.1]]) — every port conserves by construction; the residual is reported everywhere and *enforced* where one side is closed-form. The global power residual $\mathcal R(t)$ ([[port-algebra-atlas-0.1]] §6) is the single cross-case-study diagnostic.
7. **Global fields and topology** ([[global-fields-and-topology-atlas-0.1]]) — gravity-like uniform fields are injected directly, outside the edge mechanism; agent set membership can change mid-rollout (staging, docking).

---

## Per-portion map

| Portion                  | Page                                      | What it covers                                                                                     |
| ------------------------ | ----------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Agent definition         | [[agent-definition-atlas-0.1]]            | What counts as an agent, granularity, per-agent conditioning                                       |
| Tokenizer                | [[graph-tokenizer-atlas-0.1]]             | Per-agent encoding, reused from Noether 1.1, plus boundary-token exposure for edges                |
| **Port algebra** *(binding)* | [[port-algebra-atlas-0.1]] | The five closed interface types — `MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC` — as effort–flow pairs; why $O(K)$ ports replace $O(K^2)$ adapters |
| **Program roadmap**      | [[f1-pathmap-and-end-goal]] | The end goal (autonomous multiphysics design, F1 as target) and the 12-rung ladder to it; falsification criteria |
| **Prior art & novelty**  | [[prior-art-and-novelty-atlas-0.1]] | What is and is not new here, checked against sources; the $O(K^2)$ problem and its fix |
| Edge generation          | [[edge-generation-atlas-0.1]]             | Declared/geometric instantiation (Stage 1 default) *and* RL-based instantiation (deferred option)  |
| Expert library           | [[expert-library-atlas-0.1]]              | Organizing experts by governing-equation family; the rocket case study's 4-expert breakdown        |
| U-Net hierarchy          | [[unet-hierarchy-atlas-0.1]]              | Pooling/unpooling, per-level MLP/RL gating, resolves the shared-trunk-vs-disjoint-experts question |
| Conservation             | [[conservation-as-constraint-atlas-0.1]]  | Imposed flux-matching vs. Noether 1.1's discovered invariants                                      |
| Global fields & topology | [[global-fields-and-topology-atlas-0.1]]  | Gravity as a non-edge; staging/docking as graph mutation                                           |
| Training & bootstrap     | [[training-and-bootstrap-atlas-0.1]]      | Applies [[incremental-transfer-roadmap]]'s 4-stage plan to Atlas's specific experts                |
| First instantiation *(blocked)* | [[case-study-rocket-ascent-2d-atlas-0.1]] | The 2D rocket ascent case study — agents, edges, experts, phased plan. **Blocked at M2** on corpus generation and three from-scratch experts |
| **Active case study**    | [[case-study-wind-farm-wake-2d-atlas-0.1]] | **2D wind-farm wake** — the engineering scenario needing *no new experts and no new data*; the screen that eliminated wildfire/aircraft/car/battery/reentry, plus the phased plan |
| ↳ its full specification | [[spec-wind-farm-wake-atlas-0.1]] | 8 agents, 15 typed edges, governing equations, the four conservation laws and the mechanism enforcing each, time stepping, gates W0–W10 |
| Single-expert control    | [[case-study-rbc-decomposition-atlas-0.1]] | Domain-decomposed 2D Rayleigh–Bénard — one family, one frozen expert, monolithic gold standard; the cleanest isolation of communication-layer error |
| **Outcome** *(2026-09-26)* | [[00-atlas-0.1-outcome]] | What 88 tiers established, held against five proposal claims; the evidence ledger, the interactive materials, and what is not done |

---

## Design invariants

1. **In Stage 1, agents, edges, and edge types are declared inputs, not discovered.** This is a deliberate scope reduction against the full [[regime-moe-architecture]] vision (dimensionless-number-driven coarse gate, learned edge scoring) — see [[edge-generation-atlas-0.1]] for what's deferred and why.
2. **Experts are organized by governing-equation family, not by field-quantity label.** "Heat," "pressure," "stress," "fluid" are edge-type *labels* describing an interface contract — they are not, each, a separate expert. See [[expert-library-atlas-0.1]].
3. **Conservation is an interface constraint, not a discovery target.** See [[conservation-as-constraint-atlas-0.1]] for the direct contrast with [[discovered-conservation-1.1]].
4. **Global uniform fields (gravity) bypass the edge mechanism entirely.** See [[global-fields-and-topology-atlas-0.1]].
5. **Graph topology can mutate mid-rollout** (an agent leaves at staging, a new agent joins at docking) as a first-class event, not an edge-type. See [[global-fields-and-topology-atlas-0.1]].
6. **Bootstrap via transfer before native training.** Experts are seeded from existing pretrained models where a donor exists, trained from scratch only where none does — see [[training-and-bootstrap-atlas-0.1]] and [[incremental-transfer-roadmap]].
7. **Minimize simultaneous unknowns.** RL edge instantiation and RL expert-gating are real, documented options ([[edge-generation-atlas-0.1]], [[unet-hierarchy-atlas-0.1]]) but are deliberately deferred until more than one scenario exists to learn across — introducing them alongside a from-scratch pipeline would make failures undiagnosable.

---

## Folder layout

*(Restructured 2026-08-19.)* This hub sits at the top of `concepts/Atlas 0.1/`. Beneath it:

- **`common/`** — the case-study-agnostic architecture: [[agent-definition-atlas-0.1]], [[graph-tokenizer-atlas-0.1]], [[edge-generation-atlas-0.1]], [[expert-library-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[global-fields-and-topology-atlas-0.1]], [[training-and-bootstrap-atlas-0.1]]. **Anything that would survive swapping the case study lives here.**
- **`case-study-rocket-ascent/`** — the rocket scenario plus its `implementation/` subfolder (master plan, phases 0–5, budget, build log). All of it is rocket-specific: geometry, agent table, corpus plan, expert set.
- **`case-study-wind-farm-wake/`** — the active scenario: [[case-study-wind-farm-wake-2d-atlas-0.1]] (rationale + plan) and [[spec-wind-farm-wake-atlas-0.1]] (the binding spec).
- **`case-study-rbc-decomposition/`** — the synthetic single-expert control.

Links are bare `[[page-name]]` throughout, so the move broke nothing and future moves will not either.

---

## Status

Design-stage — **nothing is built yet.** [[case-study-rocket-ascent-2d-atlas-0.1]] is a plan of action, not an as-built report (contrast [[noether-1.0-rbc]]'s as-built status for Noether 1.0).

The `Atlas 0.1 implementation` folder alongside this one now holds the **executable build layer**: [[00-atlas-0.1-implementation-plan]] (master plan, global conventions, milestones M0–M7) plus one self-contained spec per phase — [[impl-atlas-0.1-phase0-scope-and-data]], [[impl-atlas-0.1-phase1-scaffold]], [[impl-atlas-0.1-phase2-experts]], [[impl-atlas-0.1-phase3-integration]], [[impl-atlas-0.1-phase4-validation]], [[impl-atlas-0.1-phase5-expansion]] — with [[atlas-0.1-implementation-log]] as the append-only build tracker. Each phase spec has intuition / theory / implementation sections and is written to be executable by an agent holding only that page.

---

## See Also

- [[pfm-purpose-and-direction]]
- [[regime-moe-architecture]]
- [[incremental-transfer-roadmap]]
- [[00-noether-1.1-overview]]
- [[mixture-of-experts]]
- [[pfm-interface-design]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
