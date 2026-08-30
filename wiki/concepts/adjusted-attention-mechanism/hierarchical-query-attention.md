# Hierarchical Query Attention for Physics (resolution-free local + global)

**Type:** Concept (folder: adjusted-attention-mechanism)
**Status:** Adaptation of Q-Former / HierarQ ([[q-former-architecture]]) to the PFM setting.
**Related Concepts:** [[00-attention-overview]], [[symmetric-attention-physics]], [[index-share-sparse-attention]], [[graph-tokenizer]], [[learned-query-compression-tokens]], [[physics-conditioned-query-tokens]]
**Related Summaries:** [[q-former-architecture]], [[multimodal-transformers-survey]]

---

## The problem it solves

Two PFM requirements that patch attention cannot meet simultaneously:
1. **Resolution-freedom** — a model trained at $128^2$ should run at $512^2$ without retraining or token-count blow-up (criterion 1, [[00-token-representation-overview]]).
2. **Local *and* global reach at bounded cost** — serve hyperbolic (local) and elliptic (global) physics without $O(n^2)$ attention.

**Learned-query attention** ([[learned-query-compression-tokens]]) addresses (1); **HierarQ's dual local/global query streams** ([[q-former-architecture]]) address (2).

---

## Mechanism 1: query compression → resolution-freedom

Introduce $N$ learnable query vectors $Z \in \mathbb R^{N\times D}$, independent of input size. They cross-attend into the raw field features $F \in \mathbb R^{n\times D}$ (where $n$ grows with resolution) and emit a fixed-length summary:

$$Z'' = \mathrm{CrossAttn}(Z, F) = \mathrm{softmax}\!\Big(\tfrac{(ZW_Q)(FW_K)^\top}{\sqrt d}\Big)(FW_V) \in \mathbb R^{N\times D}.$$

Only the $N$ summary tokens go downstream — **fixed count regardless of $n$** — at $O(Nn)$ cost (linear in input size) versus $O(n^2)$ self-attention. Interleave with self-attention among the queries (BLIP-2 pattern) so queries specialize and de-duplicate.

In the initial model this is the **per-patch / per-node set encoder** of the [[graph-tokenizer]]: the variable number of sample points inside a patch is compressed by $N$ queries into a fixed-dim node latent — the source of the tokenizer's resolution-freedom.

## Mechanism 2: dual local + global streams (HierarQ)

Run two query streams in parallel ([[q-former-architecture]]):
- **Entity stream (local)** — small queries focused on fine, short-range structure (shocks, boundary layers, vortex cores).
- **Scene stream (global)** — queries that aggregate domain-wide / long-range structure (pressure field, mean flow, global invariants).

$$Z_{\text{local}} = \mathrm{CrossAttn}(Z_{\text{loc}}, F_{\text{window}}), \qquad Z_{\text{global}} = \mathrm{CrossAttn}(Z_{\text{sce}}, F_{\text{all}}).$$

This is a learned-query realization of the **local-vs-global PDE axis** — the compression-side dual of [[index-share-sparse-attention]]'s sparsity-side realization.

## Mechanism 3: compressing memory bank → bounded long-horizon context

HierarQ's two memory banks address rollout context growth ([[autoregressive-rollout-stability]]):
- **Entity memory** — FIFO of recent local detail (short window).
- **Scene memory** — compresses history via similarity-merging, aggregating redundant global state.

For a PFM this bounds the cost of long rollouts: keep a fixed-size compressed memory of the global flow state plus a FIFO of recent local detail, instead of an ever-growing trajectory window — directly relevant to [[memory-augmented-physics-models]] and the "context helps but the window grows" tension in [[iterative-refinement-pfm]].

---

## Steerable queries = the non-text conditioning interface

Q-Former queries can be **conditioned** (text-conditioned in BLIP-2). For physics, condition them on the [[pfm-interface-design]] inputs instead of text — inject $\hat{\Delta t}$, dimensionless numbers $(\mathrm{Re},\mathrm{Ma},\mathrm{Pr})$, field-type codes, and boundary-condition codes as query conditioning. The same field then yields different token sets under different physical regimes, realizing [[physics-conditioned-query-tokens]] at the tokenizer. Disentangled query groups (DisenQ, with an orthogonality penalty) can be assigned per field type or per conserved quantity to keep them non-overlapping.

---

## Interaction with the symmetric core

Query attention is **cross-attention** (queries ≠ keys), so it is not symmetric in the [[symmetric-attention-physics]] sense — and it does not need to be: it is an *encoder/compressor*, not the *dynamics*. The division of labor:

```
Hierarchical query attention  → encode field → fixed-budget, resolution-free node latents
Symmetric attention            → evolve those nodes → reciprocal, conservative dynamics
```

Compression happens once at tokenize/detokenize; the conservative symmetric mechanism carries the actual time evolution among the compressed tokens. This keeps the expensive symmetric/skew-symmetric machinery on a *small, fixed* token set.

---

## Cost of compression

- **Lossy** — too small $N$ discards fine-scale turbulence/shocks, and the error compounds over rollout. Mitigations: adaptive $N$ (more queries for turbulent inputs), or a residual high-detail path ([[wave-particle-dual-tokens]]).
- **Opaque** — vanilla queries lack physical meaning; physics conditioning + disentanglement ([[physics-conditioned-query-tokens]]) restores interpretability and steers extraction toward conserved/relevant quantities.

---

## [AI Inference]

**[AI Inference]:** Hierarchical query attention and [[index-share-sparse-attention]] are dual solutions and should be used at different stages: **compression** for the *intra-patch* problem (variable sample points → fixed node latent, resolution-freedom) and **sparsity** for the *inter-node* problem (many nodes, sparse symmetric interaction, efficiency). A PFM that uses Q-Former compression to build resolution-free graph nodes and Index Share / symmetric graph attention to evolve them gets resolution-freedom *and* near-linear scaling *and* conservative dynamics — three otherwise-competing requirements, each handled by the mechanism suited to it.

**[AI Inference]:** The scene-memory bank reframes long-horizon rollout from "feed a growing trajectory" to "maintain a fixed-dim global state estimate." This is effectively a **learned, compressed Eulerian summary** of the flow's slow modes (mean flow, large eddies, global pressure), updated each step — a tokenizer-level analog of the slow/fast decomposition that multiscale methods exploit. Pairing a slow compressed scene memory with a fast local FIFO is the attention-side version of multiscale time-stepping.

---

## See Also

- [[q-former-architecture]] — source summary (BLIP-2 / HierarQ / DisenQ)
- [[00-attention-overview]] — attention design space
- [[learned-query-compression-tokens]] / [[physics-conditioned-query-tokens]] — the token-representation pages
- [[index-share-sparse-attention]] — the dual long-context mechanism (sparsity)
- [[symmetric-attention-physics]] — the dynamics mechanism this feeds
- [[graph-tokenizer]] — query compression as the resolution-free node encoder
- [[pfm-interface-design]] — steerable queries as the conditioning interface
- [[memory-augmented-physics-models]] / [[iterative-refinement-pfm]] — compressing memory for long rollouts
