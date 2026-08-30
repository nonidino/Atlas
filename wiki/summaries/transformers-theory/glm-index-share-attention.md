# GLM 5.2: Index Share Sparse Attention, Multi-Token Prediction, KV Redesign

**Source:** MindStudio blog — "GLM 5.2 Architecture Deep Dive: Index Share, Sparse Attention, and Multi-Token Prediction" (2026-06-25). Vendor/secondary write-up of Zhipu AI + Tsinghua's GLM 5.2; not a peer-reviewed paper.
**Type:** Source summary (folder: transformers-theory)
**Related Concepts:** [[index-share-sparse-attention]], [[00-attention-overview]], [[symmetric-attention-physics]], [[transformer-architectures]], [[mixture-of-experts]], [[autoregressive-rollout-stability]]
**Related Summaries:** [[q-former-architecture]], [[conditional-memory-engram]]

---

## Overview

GLM 5.2 targets the **long-context efficiency wall**: standard attention is $O(n^2)$ in sequence length $n$, which is operationally impractical at $n \sim 10^6$ tokens. The model ships three coupled mechanisms — **Index Share sparse attention**, **multi-token prediction (MTP)**, and an aggressive **KV-cache redesign** — that together claim $\sim 2.9\times$ fewer effective compute operations at 1M-token context and $\sim 20\times$ lower KV-cache memory.

For the PFM project the relevance is direct: a physics foundation model on large 2D/3D domains or long trajectories produces **enormous token counts** ($64^3 \approx 2.6\times10^5$ spatial tokens before any temporal axis), so the same $O(n^2)$ wall applies. Index Share is the wiki's first concrete *content-adaptive sparse-attention* mechanism for handling that regime, and its local + global + landmark index structure maps cleanly onto the **local-vs-global** PDE axis ([[tokenization-tradeoff-axes]]).

---

## Mechanism 1: Index Share Sparse Attention

### The sparse-attention problem

Sparse attention constrains each query to attend to a selected subset of size $k \ll n$ rather than all $n$ keys. If $k = O(\sqrt n)$ or $O(\log n)$, total cost becomes near-linear. The hard question is *which* $k$ positions — picking wrong drops critical long-range dependencies.

### The Index Share insight

In standard multi-head attention with $h$ heads, each head independently computes its own sparse index set — $h$ separate selection passes. **Index Share groups heads into $g$ clusters that share one index set per cluster.** With 32 heads in 4 groups of 8, you compute 4 index sets instead of 32; all 8 heads in a group attend to the same positions but with independent $Q/K/V$ projections.

Benefits compound:
- **Compute:** index-generation overhead drops by the group size factor.
- **Memory coherence:** KV entries for shared positions are loaded once and reused across the group.
- **Batch parallelism:** shared indices let the group's attention batch efficiently on-GPU.

### How indices are selected (content-adaptive)

Not fixed patterns — a lightweight routing network scores positions. Each group maintains three index types:

1. **Local indices** — a sliding window of recent tokens (short-range context).
2. **Global indices** — tokens chosen by the routing network from attention-score estimation (long-range context).
3. **Landmark indices** — periodically spaced fixed anchors (structural backbone).

This is spiritually a once-per-group version of Routing Transformers / online top-$k$ attention (arXiv:2003.05997). The local+global+landmark triple is the key structural idea.

$$\text{Attn}(q_i) = \text{softmax}\!\Big(\tfrac{q_i K_{\mathcal{I}(g)}^\top}{\sqrt{d}}\Big) V_{\mathcal{I}(g)}, \qquad \mathcal{I}(g) = \underbrace{\mathcal{L}}_{\text{local}} \cup \underbrace{\mathcal{G}(q)}_{\text{routed}} \cup \underbrace{\mathcal{A}}_{\text{landmarks}}$$

where $\mathcal{I}(g)$ is shared across all heads in group $g$.

---

## Mechanism 2: Multi-Token Prediction (MTP)

Standard autoregressive training gives one gradient signal per forward pass (predict the next token only). MTP predicts $k$ future tokens simultaneously from one forward pass ($k=4$ in GLM 5.2), via $k$ **shallow** prediction heads (one or two layers each) that share the main hidden states but have independent output projections — roughly 5–10% parameter overhead.

Two effects:
1. **Richer training signal** — the model must form representations predictive of a *horizon*, not just one step, improving internal structure (especially syntax/logic).
2. **Speculative decoding** — generate $k$ candidates in parallel, verify in one pass; accepted tokens skip full generation (1.5–2.5× throughput on structured tasks).

Cost: loss-weighting across the $k$ heads is delicate; gains are task-dependent (higher on predictable/structured sequences).

---

## Mechanism 3: KV-Cache Redesign

At 1M tokens the KV cache dominates memory (a naive 40-layer/32-head/128-dim FP16 cache $\approx 655$ GB). Three combined fixes give $\sim 20\times$ reduction:
- **Grouped-Query Attention (GQA)** — 8 query heads share 1 KV head → 8× smaller cache.
- **Sparse cache eviction** — positions never in any current Index Share set are offloaded/evicted.
- **INT8 quantized KV** — per-head INT8 storage with scale factors → 2× over FP16.

---

## Relevance to the PFM Goal

1. **Large-domain / long-trajectory feasibility.** The PFM's token count explodes with spatial resolution and rollout length. Index Share is a concrete route to keep attention near-linear without abandoning content-adaptivity — unlike pure sliding-window (Mistral) which loses long-range elliptic coupling, the failure mode flagged in [[tokenization-tradeoff-axes]].

2. **Local + global + landmark ≈ the PDE locality spectrum.** Hyperbolic physics needs local reach; elliptic physics (Poisson pressure solve, Stokes) needs global coupling. The three index types are exactly the structural ingredients to serve both — a sparse-attention analog of the multiscale hierarchy in [[multiscale-hierarchical-gnn]]. The landmark anchors are a cheap stand-in for the coarse-grid solve in multigrid.

3. **MTP ↔ rollout stability.** Predicting $k$ steps ahead in one pass forces horizon-aware representations — directly relevant to [[autoregressive-rollout-stability]]. Multi-step training is one of the few proven mitigations for compounding error; MTP is its architectural form, and pairs naturally with the initial model's direct-state prediction.

4. **Index sharing ↔ shared physical sparsity.** Heads in a group attending to the same positions parallels physical reality: many fields at a location are governed by the *same* local stencil/neighborhood. Sharing indices across field-channels (the physics analog of heads) is physically motivated, not just an efficiency hack.

---

## [AI Inference]

**[AI Inference]:** Index Share's group-shared index set is the natural attention-side partner of the **graph tokenizer** ([[graph-tokenizer]]). If each physics token is a graph node, the "index set" is just the node's neighbor list (radius graph for local, hierarchical supernodes for global, fixed landmark nodes for structural anchoring). Computing one neighbor list per *node-group* and reusing it across all field-channel heads is precisely Index Share, recast in graph terms — the sparse-attention and message-passing views coincide. This unifies GLM's efficiency mechanism with the wiki's [[graph-mesh-tokens]] / [[symmetric-attention-physics]] program.

**[AI Inference]:** MTP is the discrete-token version of **derivative / multi-step prediction** in physics ([[arch-neural-differentiator]], GP$_{\text{hy}}$T). Predicting $\hat{x}_{t+1},\dots,\hat{x}_{t+k}$ from one pass and enforcing consistency among them (e.g., that they lie on a smooth trajectory) is a learned, integrator-free analog of multi-step time-stepping — a candidate inbuilt stabilizer for the initial model that does not require committing to a classical integrator.

---

## Cross-Links

- [[index-share-sparse-attention]] — concept page formalizing the mechanism for physics
- [[00-attention-overview]] — attention design space for the PFM
- [[symmetric-attention-physics]] — the chosen core mechanism; Index Share is the large-context complement
- [[tokenization-tradeoff-axes]] — local-vs-global axis = local/global/landmark indices
- [[multiscale-hierarchical-gnn]] — multigrid analog of the landmark/global structure
- [[autoregressive-rollout-stability]] — MTP as a stability mechanism
- [[q-former-architecture]] — the other long-context source in this batch (compression, not sparsity)
- [[mixture-of-experts]] — routing networks select indices (shared routing idea)
