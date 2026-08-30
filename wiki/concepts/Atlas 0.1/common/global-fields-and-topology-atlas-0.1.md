# Atlas 0.1: Global Fields and Graph Topology Mutation

**Type:** Core Concept — Model Portion
**Related Concepts:** [[00-atlas-0.1-overview]], [[pfm-purpose-and-direction]], [[edge-generation-atlas-0.1]], [[expert-library-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]]

---

## Two mechanisms that don't fit the agent/edge/expert framing

Both surfaced directly by working through the rocket-ascent case study — neither is an edge, and neither is handled by an expert in [[expert-library-atlas-0.1]]'s sense.

## 1. Global uniform fields (gravity)

Gravity acts identically on every agent in the rocket graph — it is not an interaction *between* any two agents, has no proximity condition, and requires no routing. Modeling it as an edge (e.g., a "gravity agent" that every other agent connects to) would be structurally dishonest: it would imply gravity is discovered or routed the way heat/pressure/stress/fluid interactions are, when it is simply an input every agent's update equation receives directly.

**Atlas treats gravity, and any field like it, as a third category alongside agent-to-agent edges and per-agent internal state: a global scalar/vector field injected uniformly into every agent's update.**

$$
\frac{d}{dt}\big(\text{momentum}_a\big) = \sum_{b \in \mathcal{N}(a)} F_{a \gets b}^{\text{edge}} + F^{\text{global}}, \qquad F^{\text{global}} = m_a \, \mathbf{g} \ \ \forall a
$$

where the edge-sourced forces route through [[expert-library-atlas-0.1]]'s typed experts as usual, but $F^{\text{global}}$ bypasses that machinery entirely. This is the mechanism that feeds the rigid-body trajectory expert's net-force integration ([[expert-library-atlas-0.1]]) alongside the forces produced by every other expert.

**No other page in this vault names this as a distinct architectural category before this one** — [[pfm-interface-design]]'s conditioning contract handles per-agent dimensionless parameters, but a uniform external field applying identically to every agent regardless of its own state is a different thing, and worth keeping distinct rather than folding into either the edge mechanism or the conditioning contract.

## 2. Graph topology mutation (staging, docking)

Stage separation is the sharpest instance in the rocket case: the spent first stage **leaves the agent graph mid-rollout**, and the remaining stage's mass, thrust, and aerodynamic characteristics change discontinuously the instant it happens. This is structurally the same category of event as a new module docking at a moon base (Gap 10, composability, in [[pfm-purpose-and-direction]]) — just membership loss instead of membership gain.

Neither case is a regime crossover in [[regime-moe-architecture]]'s finding-2 sense (smooth vs. threshold *field* transitions) — it's a discontinuity in **which agents exist**, not in how an existing agent's physics behaves. No amount of FiLM conditioning or RL-gated expert switching (see [[unet-hierarchy-atlas-0.1]]) handles a node disappearing from the graph; it requires the graph itself to be mutable at inference time.

### What this requires, concretely

- The [[edge-generation-atlas-0.1]] mechanism (whichever variant is active) must be able to **re-instantiate edges after a topology change** — a docked module's edges to its neighbors, or the removal of every edge touching a jettisoned stage.
- The [[unet-hierarchy-atlas-0.1]] pooling structure must tolerate a **change in the number of leaves feeding the hierarchy** — the supernode clustering computed before a staging event is not valid after it.
- This is the concrete case for [[pfm-purpose-and-direction]]'s Gap 10 (composability as inference-time generalization) rather than a training-time property: the model has to handle the post-staging graph correctly the instant it happens, with no retraining window.

**Not designed in Atlas's current scope.** [[case-study-rocket-ascent-2d-atlas-0.1]]'s Phase 0–5 plan does not include a staging event — the 2D case study is scoped to powered ascent without a stage separation. This page documents the requirement for when a scenario needs it, not a solved mechanism.

---

## See Also

- [[00-atlas-0.1-overview]]
- [[pfm-purpose-and-direction]]
- [[edge-generation-atlas-0.1]]
- [[expert-library-atlas-0.1]]
- [[unet-hierarchy-atlas-0.1]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
