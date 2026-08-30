# Atlas 0.1: U-Net Expert Hierarchy

**Type:** Core Concept — Model Portion
**Related Concepts:** [[00-atlas-0.1-overview]], [[regime-moe-architecture]], [[multiscale-hierarchical-gnn]], [[intelligent-patching]], [[poseidon-pde-foundation-model]], [[expert-library-atlas-0.1]], [[edge-generation-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]]

---

## Why a hierarchy at all

A single flat GNN over every agent's tokens has no mechanism for long-range or bulk-scale coupling without dense all-pairs attention. Poseidon's scOT addresses this on a regular grid via SwinV2-style hierarchical windowed attention with a U-Net encoder/decoder shape. Atlas needs the graph-native equivalent: **pool the agent graph into progressively coarser supernodes, process each level with the expert appropriate to that scale, then unpool back to token resolution with skip connections preserving fine detail.**

This is not a new mechanism invented for Atlas — it composes two things already in this wiki:

- **[[multiscale-hierarchical-gnn]]**'s FMM-analog V-cycle, which already solves the "how do you get $O(N)$ long-range coupling in a GNN" problem this hierarchy needs.
- **[[intelligent-patching]]**'s refinement tree / supernode hierarchy, reused **in reverse** — that page builds a tree for adaptive refinement (fine where detail is high); Atlas's pooling path walks the same kind of tree the other direction, coarsening rather than refining.

What *is* new: **a different expert at each level**, not a generic pooling operator throughout.

## Structure

$$
\text{Token-level GNN} \xrightarrow{\text{pool}} \text{Agent-region supernodes} \xrightarrow{\text{pool}} \text{Per-agent supernodes} \xrightarrow{\text{pool}} \text{Whole-vehicle bottleneck}
$$
$$
\text{Whole-vehicle bottleneck} \xrightarrow{\text{unpool} + \text{skip}} \text{Per-agent} \xrightarrow{\text{unpool} + \text{skip}} \text{Agent-region} \xrightarrow{\text{unpool} + \text{skip}} \text{Token-level output}
$$

Each pooling step must be a **physically meaningful coarsening**, not generic feature averaging — a chamber-wall supernode has to aggregate heat flux and pressure consistently, which is exactly the constraint [[intelligent-patching]]'s hierarchy was already designed to respect.

## Expert gating per level: MLP vs. RL, mapped onto smooth vs. threshold

This is the piece that ties the hierarchy back to [[regime-moe-architecture]]'s finding 2 (smooth vs. threshold crossovers need separate mechanisms), rather than treating "how do I pick an expert at each level" as an unconstrained design choice:

- **Fine, local levels** (token-level, agent-region) mostly see continuous variation — a **differentiable MLP gate** blending experts is the right tool, the same argument as FiLM/adaLN conditioning in [[regime-moe-architecture]].
- **Coarser levels** (per-agent, whole-vehicle bottleneck) are where genuinely discrete regime decisions live — "is this agent now dominated by combustion instability or nominal thrust," "has staging happened" — closer to what RL-style reward-driven discrete switching is suited for (same argument as [[edge-generation-atlas-0.1]]'s case for RL edge instantiation).

This reconciles what could otherwise look like two competing proposals (MLP gating vs. RL gating) into one design: **MLP where the transition is smooth, RL where it's a discrete regime-dominance decision**, matching level to mechanism rather than picking one gate type for the whole hierarchy.

## This is the concrete answer to [[regime-moe-architecture]]'s open trunk-vs-disjoint question

That page left unresolved whether Atlas-like architectures should use a shared trunk with late specialization or fully disjoint experts from layer one. The U-Net hierarchy answers this directly: **the pooling/unpooling structure itself is the shared trunk** (every agent's tokens pass through the same coarsening/refinement machinery), and **per-level expert selection is the disjoint-specialization axis** (different physics at different levels gets genuinely different experts, per [[expert-library-atlas-0.1]]). These coexist rather than compete — the open question was a false dichotomy once the hierarchy separates "how information flows" (shared) from "what computes on it" (specialized).

## Stage 1 scope reduction: 2 levels, not 4

For [[case-study-rocket-ascent-2d-atlas-0.1]]'s first build, the full 4-level hierarchy above is more machinery than 7 agents justifies, and it adds failure surface that makes a first end-to-end test hard to diagnose (if the pipeline doesn't work, is it the experts, the gating, or the pooling levels?). **Stage 1 uses 2 levels only: token-level GNN + a single whole-vehicle bottleneck**, skipping agent-region and per-agent pooling entirely. The intermediate levels are added back only when a future scenario's agent count actually demands them — the moon-base scenario ([[pfm-purpose-and-direction]]), with dozens of modules, is the most likely trigger.

## Skip connections

As in any U-Net, skip connections carry fine-scale detail (the exact per-token state at each encoder level) directly to the matching decoder level, so the bottleneck's global/bulk correction doesn't overwrite local detail it was never meant to represent — the whole-vehicle trajectory expert should inform local aerodynamic loads without erasing the boundary-layer-scale structure the token-level expert already computed.

---

## See Also

- [[00-atlas-0.1-overview]]
- [[regime-moe-architecture]]
- [[multiscale-hierarchical-gnn]]
- [[intelligent-patching]]
- [[poseidon-pde-foundation-model]]
- [[expert-library-atlas-0.1]]
- [[edge-generation-atlas-0.1]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
