# Q-Former: Learned-Query Multimodal Alignment (BLIP-2 / HierarQ / DisenQ)

**Source:** EmergentMind topic page — "Q-Former Architecture" (2025-07-11). Secondary review synthesizing BLIP-2 (arXiv:2301.12597), the Q-Former PEFT study (Kim et al., arXiv:2410.09489), HierarQ (Azad et al., arXiv:2503.08585), and DisenQ (Azad et al., arXiv:2507.07262).
**Type:** Source summary (folder: transformers-theory)
**Related Concepts:** [[hierarchical-query-attention]], [[learned-query-compression-tokens]], [[physics-conditioned-query-tokens]], [[00-attention-overview]], [[graph-tokenizer]], [[multimodal-tokenization]]
**Related Summaries:** [[glm-index-share-attention]], [[multimodal-transformers-survey]]

---

## Overview

The **Q-Former** (Querying Transformer) is a modular transformer that aligns a perceptual encoder (vision, video, audio, 3D) with a downstream model using a small set of **learnable query tokens** that *cross-attend into* the raw features and extract a fixed-length, information-rich summary. It is the alignment bridge in BLIP-2 / InstructBLIP and has been extended to hierarchical video (HierarQ) and disentangled representation (DisenQ).

For the PFM the Q-Former matters on two fronts already tracked in the wiki: (1) it is the mechanism behind [[learned-query-compression-tokens]] — fixed token budget independent of input resolution (criteria 1 & 7 in [[00-token-representation-overview]]); and (2) its **interleaved self-attention / cross-attention** and **hierarchical local+global query streams** (HierarQ) are concrete attention designs for the "local *and* global, resolution-free" tokenizer the initial model needs.

---

## Core Architecture

A set of $N$ learnable query tokens $Z \in \mathbb{R}^{N \times D}$ (independent of input size) is processed by alternating:

- **Self-attention** among the queries: $Z' = \text{SelfAttn}(Z)$ — queries coordinate, specialize, de-duplicate.
- **Cross-attention** into fixed input features $F$ (from an upstream encoder): $Z'' = \text{CrossAttn}(Z', F)$ — queries pull task-relevant content out of the perceptual stream.

The output is $N$ compact tokens passed downstream via a linear/MLP projection. Crucially the query count $N$ is fixed (BLIP-2 uses $N=32$), so a $4096^2$ input and a $128^2$ input both emit 32 tokens — **resolution-decoupled** at $O(NF)$ cost (linear in input size $F$), versus $O(F^2)$ self-attention over raw features.

---

## Parameter-Efficient Fine-Tuning

Adaptation uses LoRA / AdaLoRA: reparameterize weight updates as a low-rank product

$$\Delta W = BA, \qquad B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \ll \min(d,k),$$

training <2% of parameters at accuracy comparable to full fine-tuning. AdaLoRA reallocates rank per-layer using SVD-derived importance scores — empirically, **self-attention layers dominate for perceptual alignment**, FFN layers for richer language-grounded reasoning, cross-attention consistently important for fusion.

---

## Extension 1: HierarQ — Hierarchical Local + Global Querying

HierarQ (arXiv:2503.08585) runs **two query streams in parallel**:
- **Entity-level (short-term/local)** queries focus on fine object detail within short context windows.
- **Scene-level (long-term/global)** queries capture long-range temporal/contextual dependencies.

Each stream has a dedicated **memory bank**: entity memory is FIFO (immediate context); scene memory compresses via cosine-similarity merging (aggregates redundant information over time). This lets HierarQ process *all* frames sequentially rather than sparsely sampling, within normal context limits — SOTA on medium-to-long video understanding.

This dual-stream local/global structure is the directly transferable idea for physics: **one query stream for local (hyperbolic/shock) structure, one for global (elliptic/pressure) structure**, with a compressing memory for long rollouts.

---

## Extension 2: DisenQ — Disentangled Query Groups

DisenQ (arXiv:2507.07262) uses three independent query sets to separate identity / motion / appearance, with an **orthogonality constraint** penalizing overlap between feature groups. The transferable idea: **distinct query groups can be forced to encode distinct, non-overlapping factors** — for physics, separate query groups per field type or per conserved quantity, kept disentangled by an orthogonality penalty.

---

## Relevance to the PFM Goal

1. **Resolution-free node/token encoder.** The single highest-value transfer: use Q-Former-style learned-query cross-attention as the **per-patch (per-node) set encoder** in the graph tokenizer. A variable number of sample points inside a patch is compressed by $N$ learned queries into a fixed-dim node latent — resolution-free *by construction*, the property patch embeddings lack ([[graph-tokenizer]]).

2. **Local + global without all-pairs cost.** HierarQ's parallel entity/scene streams are a learned-query realization of the local-vs-global axis ([[tokenization-tradeoff-axes]]) — complementary to GLM's sparse-index realization ([[glm-index-share-attention]]). Two different routes to the same physical requirement.

3. **Steerable conditioning interface.** Because queries can be conditioned (text-conditioned in BLIP-2), they are a natural home for the PFM's non-text conditioning — inject $\hat{\Delta t}$, dimensionless numbers, field-type codes, or boundary-condition codes as query conditioning, realizing the [[pfm-interface-design]] inputs at the tokenizer ([[physics-conditioned-query-tokens]]).

4. **PEFT for cross-equation transfer.** LoRA/AdaLoRA on a Q-Former front-end is the cheapest known route to adapting a frozen physics backbone to a new equation family — the BLIP-2/Poseidon frozen-backbone transfer pattern ([[transfer-learning-fine-tuning]]).

---

## Limitations

- **Lossy compression** — squashing a high-resolution field to $N$ queries discards fine-scale structure; too small $N$ loses turbulence/shocks, and errors compound over rollout ([[autoregressive-rollout-stability]]). Argues for adaptive $N$ or a residual high-detail path ([[wave-particle-dual-tokens]]).
- **Opaque queries** — vanilla learned queries carry no physical meaning; nothing guarantees they extract conserved quantities (the gap [[physics-conditioned-query-tokens]] addresses).
- **Inference latency** — the modular self/cross-attention stages add stages of computation.
- **Unproven for dynamical physics** — validated on vision-language understanding, not yet as the primary tokenizer of a physics emulator.

---

## [AI Inference]

**[AI Inference]:** The Q-Former and GLM Index Share ([[glm-index-share-attention]]) are dual solutions to the same problem. GLM keeps all $n$ tokens but **sparsifies which interact** (compute-side); Q-Former **reduces the token count** to $N$ summaries (representation-side). For the PFM these compose: a Q-Former front-end compresses each patch's interior to a node latent (resolution-freedom), and Index Share / symmetric graph attention then handles sparse interaction among the resulting nodes (efficiency at scale). Compression for the *intra-patch* problem, sparsity for the *inter-patch* problem.

**[AI Inference]:** HierarQ's compressing **scene-memory bank** is a candidate solution to long-horizon rollout context management ([[autoregressive-rollout-stability]], [[memory-augmented-physics-models]]). Rather than feeding a growing trajectory window, maintain a fixed-size compressed memory of the global flow state (scene) plus a FIFO of recent local detail (entity) — bounding context cost while retaining long-range history, exactly the "context helps but window grows" tension noted in [[iterative-refinement-pfm]].

---

## Cross-Links

- [[hierarchical-query-attention]] — concept page formalizing HierarQ-style local+global queries for physics
- [[learned-query-compression-tokens]] — the token-representation page this source grounds
- [[physics-conditioned-query-tokens]] — physics-meaningful, steerable upgrade
- [[graph-tokenizer]] — Q-Former as the resolution-free per-node set encoder
- [[00-attention-overview]] — attention design space
- [[glm-index-share-attention]] — dual long-context mechanism (sparsity vs. compression)
- [[multimodal-transformers-survey]] — original Q-Former/Perceiver source
- [[pfm-interface-design]] — steerable queries as the non-text conditioning interface
- [[transfer-learning-fine-tuning]] — LoRA/AdaLoRA frozen-backbone transfer
