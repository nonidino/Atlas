# Mixture of Experts (MoE)

**Type:** Core Concept  
**Related Sources:** Deep Memory / Dissipative Cognitive Architecture, implicitly in PFM architectures

---

## Overview

**Mixture of Experts (MoE)** is an architecture where a collection of specialized sub-networks ("experts") each handle a different subset of inputs, and a **routing mechanism** decides which expert(s) process each input. The key benefit: **total model capacity scales independently of per-input computational cost**, enabling very large models with efficient inference.

---

## Standard MoE Architecture

For input $x$, a gating network $G(x)$ selects $k$ of $N$ experts $\{E_1, \ldots, E_N\}$:
$$y = \sum_{i\in \text{TopK}(G(x))} G_i(x) \cdot E_i(x)$$

**Soft routing (Top-$k$ sparse):** Select the $k$ experts with highest gating score; compute weighted sum. Standard in LLMs (e.g., Mixtral).

**Hard routing (discrete):** Select a single expert $k^* = \arg\max G(x)$; no mixing.

---

## Hard Routing in Dissipative Cognitive Architecture

The Deep Memory paper uses **hard discrete routing** as a causal prerequisite for effective expert specialization:

$$k^* = \arg\max_k \operatorname{sim}(x_t, \mu_k)$$

This enforces strict separation between expert domains. Without hard routing, centroid convergence collapses all experts to the same representation, destroying the capacity for context-specific memory.

**Mutual information result:**
- With DM + discrete binding: $\text{MI}(\text{context}, \text{expert}) = 1.10$
- With binding destroyed: $\text{MI} = 0.001$

---

## Why MoE for Physics?

Physical systems naturally partition into distinct regimes and equation families. MoE is appealing for a PFM because:

1. **Expert specialization:** One expert per equation family (fluid, elastic, electromagnetic, …).
2. **Scalable capacity:** A 10× increase in expert count increases capacity by 10× but inference cost only by the routing overhead.
3. **Modular update:** New physical domains can be added by introducing new experts without disrupting existing expert knowledge (no catastrophic forgetting of old domains).
4. **Interpretability:** Expert activations reveal which physics regime the model is currently processing.

---

## Challenges for Physics MoE

1. **Routing smoothness:** Physical systems can transition between regimes continuously (laminar → turbulent). Hard routing creates discontinuities at regime boundaries.
2. **Load balancing:** If one physical regime is overrepresented in training data, that expert receives most queries, hindering generalization.
3. **Cross-physics transfer:** Important physical principles (conservation laws, symmetries) are shared across all regimes. Pure expert isolation may prevent beneficial cross-regime generalization.
4. **Communication between experts:** Multi-physics problems require interaction between experts (e.g., fluid-structure interaction requires both fluid and solid mechanics experts communicating).

---

## MoE vs. Dense Models for PFM

| Aspect | MoE | Dense |
|---|---|---|
| Capacity at fixed compute | Higher (many experts) | Fixed |
| Cross-domain transfer | Less natural (isolated experts) | More natural (shared weights) |
| Inference cost | Fixed (routing overhead only) | Scales with model size |
| Catastrophic forgetting | Lower (experts protect each other) | Higher (shared weights update for all tasks) |
| Interpretability | Higher (expert assignment reveals regime) | Lower (distributed representation) |

---

## Connection to In-Context Learning

A hybrid approach: use **expert routing for coarse-grained domain identification** (fluid vs. elastic vs. electromagnetic) while using **in-context learning within each expert** for fine-grained adaptation to specific parameters and boundary conditions.

This two-level hierarchy — MoE for domain routing + ICL for parameter adaptation — may be the most efficient path to a PFM that handles both broad coverage and fine-grained adaptability.

**[AI Inference]:** The Deep Memory paper's finding that hard routing is a **causal prerequisite** for stable expert memory (rather than just a useful component) suggests that soft routing may not be sufficient for a physics PFM where different equation families genuinely require different internal representations. A physics PFM should use hard routing between major equation families (governed by structural mathematical differences) and soft routing within families (governed by parameter variations).

**[[regime-moe-architecture]] is a concrete instantiation of this two-level hierarchy**, worked out from a first-principles survey of ~20 physics regimes: it specifies the routing signal (dimensionless numbers — Re, Kn, Ma, $v/c$ — already established as conditioning tokens in [[pfm-interface-design]]) and, critically, splits what this page's "routing smoothness" challenge conflates into two separate mechanisms — continuous FiLM/adaLN conditioning for *smooth* regime crossovers (Kn, Ma) vs. a dedicated discontinuity channel for *threshold* crossovers (shock formation, yield surfaces, fracture) that are genuinely discontinuous in the physics and should not be smoothed by the router.

---

## See Also

- [[physics-foundation-models]]
- [[in-context-learning-physics]]
- [[transfer-learning-fine-tuning]]
- [[deep-memory-dissipative]]
- [[regime-moe-architecture]]
- [[incremental-transfer-roadmap]] — how to bootstrap this MoE structure from existing pretrained weights rather than training experts from scratch
- [[expert-library-atlas-0.1]] — a concrete worked instantiation: experts cut by governing-equation family (not field-quantity label), with a physics-encoding-spectrum placement per expert
