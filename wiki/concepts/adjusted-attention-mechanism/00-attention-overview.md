# Adjusted Attention Mechanisms for the PFM — Overview / Hub

**Type:** Concept hub (folder: adjusted-attention-mechanism)
**Purpose:** Catalog the attention mechanisms considered for the physics foundation model, the physical reasoning behind each, and when to use which. The tokenizer's complement: [[graph-tokenizer]] decides *what a token is*; this folder decides *how tokens interact*.
**Related Concepts:** [[symmetric-attention-physics]], [[index-share-sparse-attention]], [[hierarchical-query-attention]], [[antisymmetric-signed-attention-transformer]], [[transformer-architectures]], [[initial-model-architecture]]
**Related Summaries:** [[glm-index-share-attention]], [[q-former-architecture]], [[transformers-particle-systems-clustering]], [[anti-symmetric-dgn]]

---

## Why attention needs adjusting for physics

Standard softmax self-attention was designed for language, and the mathematical-physics analysis of transformers ([[transformers-particle-systems-clustering]], [[transformer-mathematical-framework]]) shows it is **structurally wrong for physics** in three ways:

1. **Asymmetric** — $A_{ij} \neq A_{ji}$ because softmax normalizes row-wise. This violates Newton's third law, the reciprocity at the heart of every pairwise physical interaction.
2. **Purely attractive** — softmax weights are non-negative, so the induced "force" $\sum_j A_{ij}(V_j - V_i)$ only ever pulls tokens together. Real interactions have repulsion too.
3. **Dissipative / clustering** — for $d \ge 3$ the dynamics provably drive all tokens to a single Dirac mass (over-smoothing / rank collapse). Information from early layers is destroyed.

For language these are tolerable (clustering ≈ semantic abstraction). For physics — where tokens are particles, mesh nodes, or field patches and attention *is* the interaction — they are direct obstacles to conservation and long-horizon stability, the two properties the initial model is built around.

This folder collects the fixes. The design decision for the initial model: **symmetric attention as the core interaction, with large-context variants layered on as the token count grows.**

---

## The central distinction: reciprocal ≠ conservative

A subtlety that organizes everything here. "Symmetric attention" can mean two different things with opposite stability behavior — getting this right is the crux of the design:

| Property | What it gives | Dynamical character |
|---|---|---|
| **Symmetric score** $A_{ij}=A_{ji}$ + standard aggregation $\sum_j A_{ij}V_j$ | Reciprocity (each pair influences the other equally) | **Dissipative** — a symmetric, row-normalized $A$ is a graph-diffusion / heat operator; it smooths and decays. Conserves the *mean*, dissipates *energy*. |
| **Symmetric score** + **force-style aggregation** $\sum_j A_{ij}(V_j-V_i)$ | Newton's 3rd law; exact conservation of the aggregate (total "momentum") | Still **dissipative in energy** — this is exactly the graph Laplacian $-LV$, i.e. diffusion. Smooths variance while conserving the sum. |
| **Skew-symmetric linear part** $W - W^\top - \gamma I$ (A-DGN) | Norm/energy preservation across depth | **Conservative** — purely imaginary Jacobian eigenvalues → oscillatory, non-dissipative, information preserved indefinitely. |

The lesson, derived in full in [[symmetric-attention-physics]]: **a symmetric attention score restores reciprocity but is intrinsically dissipative (diffusive).** To also achieve *energy conservation and long-horizon stability* — the initial model's headline goal — symmetric reciprocal attention must be paired with a **conservative (skew-symmetric / symplectic) update channel**. Reciprocity and conservation are complementary ingredients, not the same thing. The existing [[antisymmetric-signed-attention-transformer]] page already combines them correctly; this folder makes the reasoning explicit and chooses the configuration for the first model.

---

## The catalog

### 1. Symmetric attention (core mechanism) — [[symmetric-attention-physics]]
The chosen default interaction for the initial model. Symmetric score $A_{ij}=A_{ji}$ for reciprocity, optionally signed ($\tanh$) for repulsion, paired with a skew-symmetric linear channel for non-dissipative depth. On a graph tokenizer this *is* a learnable, dynamic graph-Laplacian / message-passing operator — the unification of "attention" and "message passing." Encoding-spectrum level 4–5.

### 2. Index Share sparse attention (large context) — [[index-share-sparse-attention]]
GLM-5.2-derived ([[glm-index-share-attention]]). When the domain is large (high-resolution 2D/3D) or the trajectory is long, $O(n^2)$ attention is intractable. Index Share gives content-adaptive sparsity with a **local + global + landmark** index structure (mapping onto the hyperbolic/elliptic locality spectrum) and shares index sets across head groups for efficiency.

### 3. Hierarchical query attention (large context, resolution-free) — [[hierarchical-query-attention]]
Q-Former / HierarQ-derived ([[q-former-architecture]]). Learned queries cross-attend into the field to produce a **fixed-budget**, resolution-free token set; parallel local (entity) and global (scene) query streams with a compressing memory bank serve both PDE locality regimes and bound long-rollout context cost.

---

## How they compose in the initial model

These are not competitors — they occupy different layers of the stack:

```
Intra-patch  : hierarchical-query compression  → resolution-free node latents  ([[graph-tokenizer]])
Inter-node   : symmetric attention / message passing on the radius graph        (core dynamics)
Long-range   : landmark / supernode global indices (Index Share + multiscale)   (elliptic coupling)
Long-horizon : compressing scene-memory (HierarQ) bounds rollout context cost
Conservation : skew-symmetric linear channel + force-style aggregation          (stability)
```

Symmetric attention is the **mechanism**; Index Share and hierarchical queries are **how it scales** to large domains and long rollouts. See [[initial-model-architecture]] for the assembled pipeline.

---

## Position on the physics encoding spectrum

| Mechanism | Level | Notes |
|---|---|---|
| Symmetric score | 4 | architectural soft bias toward reciprocal dynamics |
| Force-style aggregation | 5 | exact conservation of aggregate momentum |
| Skew-symmetric linear part | 4–5 | architectural non-dissipative information flow |
| Signed coupling | 4 | architectural support for repulsion |
| Index Share / hierarchical queries | 1–2 | efficiency mechanisms, physics-neutral (locality is a soft bias) |

---

## See Also

- [[symmetric-attention-physics]] — the core mechanism, in depth
- [[index-share-sparse-attention]] — large-context sparse attention
- [[hierarchical-query-attention]] — resolution-free local+global query attention
- [[antisymmetric-signed-attention-transformer]] — the prior synthesis this folder builds on
- [[anti-symmetric-dgn]] — rigorous stability foundation (skew-symmetric ODE)
- [[transformers-particle-systems-clustering]] — proof that standard attention is dissipative
- [[graph-tokenizer]] — the token representation attention operates on
- [[initial-model-architecture]] — how these assemble into the first model
