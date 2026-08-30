# Index Share Sparse Attention for Physics (large-context mechanism)

**Type:** Concept (folder: adjusted-attention-mechanism)
**Status:** Adaptation of GLM-5.2's Index Share ([[glm-index-share-attention]]) to the PFM setting.
**Related Concepts:** [[00-attention-overview]], [[symmetric-attention-physics]], [[hierarchical-query-attention]], [[multiscale-hierarchical-gnn]], [[graph-tokenizer]], [[tokenization-tradeoff-axes]]
**Related Summaries:** [[glm-index-share-attention]], [[multipole-graph-neural-operator]]

---

## The problem it solves

A PFM on a large domain has a huge token count: a $256^2$ field is $\sim 6.5\times10^4$ patch tokens; a $64^3$ volume is $\sim 2.6\times10^5$; multiply by trajectory length for temporal context. Full $O(n^2)$ attention is intractable. But the naive fixes break physics:
- **Sliding-window only** (Mistral) → loses long-range *elliptic* coupling (Poisson pressure, Stokes, gravity), the canonical failure in [[tokenization-tradeoff-axes]].
- **Pure linear attention** → weak at the long-range retrieval physics needs for global constraints.

**Index Share** ([[glm-index-share-attention]]) is a content-adaptive sparse attention that keeps long-range reach while being near-linear, and its index structure happens to match PDE locality.

---

## Mechanism

Each query attends to a selected subset $\mathcal I$ of size $k \ll n$ rather than all keys. Index Share's two ideas:

1. **Group-shared indices.** Heads are grouped; all heads in a group share one index set $\mathcal I(g)$ (computed once per group, not per head). With $h$ heads in $g$ groups, index-selection cost drops $\sim g\times$; shared KV loads cut memory bandwidth; the group batches efficiently.

2. **Three index types per group** — the physically meaningful part:

$$\mathcal I(g) = \underbrace{\mathcal L}_{\text{local window}} \;\cup\; \underbrace{\mathcal G(q)}_{\text{routed global}} \;\cup\; \underbrace{\mathcal A}_{\text{fixed landmarks}}$$

- **Local** $\mathcal L$ — a window of nearby tokens (recent in time / adjacent in space).
- **Global** $\mathcal G(q)$ — content-routed: a lightweight network scores all positions and selects the top relevant ones.
- **Landmark** $\mathcal A$ — periodically spaced fixed anchors that always carry domain-global structure.

$$\text{Attn}(q_i) = \mathrm{softmax}\!\Big(\tfrac{q_i K_{\mathcal I(g)}^\top}{\sqrt d}\Big) V_{\mathcal I(g)}.$$

---

## Why this maps onto physics

The local/global/landmark triple is a near-perfect match for the **local-vs-global PDE axis** ([[tokenization-tradeoff-axes]]):

| Index type | Physics it serves | PDE class |
|---|---|---|
| Local window $\mathcal L$ | finite-speed propagation; shocks, fronts, characteristics | **hyperbolic** (Euler, wave, advection) |
| Routed global $\mathcal G(q)$ | data-dependent long-range coupling | turbulent non-local transport, MHD |
| Landmarks $\mathcal A$ | domain-wide instantaneous equilibrium | **elliptic** (Poisson pressure, Stokes, gravity, Coulomb) |

The landmark anchors play the role of the **coarse-grid solve in multigrid** / the supernodes in the multipole hierarchy ([[multiscale-hierarchical-gnn]], [[multipole-graph-neural-operator]]): a cheap channel through which global constraints propagate in $O(1)$ hops instead of $O(\text{diameter})$ layers. This is how a sparse-attention PFM can do a pressure projection without going fully dense.

---

## Preserving symmetry under sparsity

The initial model's core is **symmetric** attention ([[symmetric-attention-physics]]). Sparsity must not break $A_{ij}=A_{ji}$. The condition is **mutual membership**:

$$j \in \mathcal I(i) \iff i \in \mathcal I(j).$$

- **Local windows** and **radius graphs** satisfy this automatically (neighborhoods are symmetric).
- **Landmarks** satisfy it if every node attends to all landmarks and landmarks attend back.
- **Routed global indices** need care: top-$k$ routing is generally *not* mutual. The fix is to **symmetrize the selected edge set** ($\mathcal E \leftarrow \mathcal E \cup \mathcal E^\top$) before computing scores, or to route on a symmetric score so $i$ selecting $j$ implies $j$ ranks $i$ comparably. On the graph tokenizer this is just maintaining an undirected edge set.

---

## Relationship to the graph tokenizer

On [[graph-tokenizer]], "index set" = "neighbor list." Index Share is then literally: **compute one neighbor list per node-group and reuse it across all field-channel heads.** Local indices = radius-graph edges; landmarks = hierarchical supernodes; global routing = learned long-range edges. The sparse-attention and message-passing views coincide exactly — Index Share is the efficiency mechanism that makes symmetric graph attention scale to foundation-model token counts.

---

## When to use it

- **Use** when token count exceeds the dense-attention budget (high-resolution 2D/3D, long trajectories) — GLM switches to sparse above $\sim 32$k tokens; below that, full symmetric attention is cheaper and strictly better.
- **Complement**, not replace, [[hierarchical-query-attention]]: Index Share keeps $n$ tokens but sparsifies interaction; query compression reduces $n$. Compose them — compress intra-patch, sparsify inter-node.

---

## [AI Inference]

**[AI Inference]:** Index Share's "group of heads shares one index set" has a clean physics reading: the field channels at a spatial location (velocity, pressure, temperature, …) are governed by the **same local stencil** — they couple to the same neighbors, just with different coefficients. So sharing the neighbor list across field-channel "heads" is not merely an efficiency hack; it is the architectural statement that *co-located physical quantities share a spatial coupling structure.* This predicts the index groups should be organized by spatial locality, not by abstract head index.

**[AI Inference]:** The landmark indices are the attention-side dual of the **memory-augmented** static library in [[memory-augmented-physics-models]]: a small fixed set of always-visible anchors carrying global state, analogous to a coarse canonical-solution cache. Co-designing landmarks with a learned coarse representation (a few global "summary" nodes updated each step) would merge Index Share's landmark channel with the multipole coarse level — one structure serving both efficiency and elliptic coupling.

---

## See Also

- [[glm-index-share-attention]] — source summary
- [[00-attention-overview]] — attention design space
- [[symmetric-attention-physics]] — the core mechanism this scales; mutual-membership symmetry condition
- [[hierarchical-query-attention]] — the complementary compression-based long-context mechanism
- [[multiscale-hierarchical-gnn]] / [[multipole-graph-neural-operator]] — landmarks as the multigrid coarse level
- [[graph-tokenizer]] — index set = neighbor list
- [[tokenization-tradeoff-axes]] — local-vs-global PDE axis
- [[memory-augmented-physics-models]] — landmarks as a global-state cache
