# Summary: WALRUS — A Cross-Domain Foundation Model for Continuum Dynamics

**Source:** `raw/WALRUS, A CROSS-DOMAIN FOUNDATION MODEL FOR CONTINUUM DYNAMICS.md`
**Authors:** Michael McCabe et al. — The Polymathic AI Collaboration (Flatiron Institute, NYU, Cambridge, …)
**arXiv:** 2511.15684v1
**Date Ingested:** 2026-04-11 · **Deepened:** 2026-06-29

---

## Overview

**Walrus** is a **1.3 billion-parameter** transformer foundation model for continuum-level physical simulation — the largest and most broadly-pretrained open-source physics emulator in the wiki. It is pretrained on **19 diverse physical scenarios** spanning astrophysics, geoscience, rheology, plasma physics, acoustics, active matter, and classical fluids, covering **63 distinct state variables** across both 2D and 3D. Its contribution is less a single new idea than a **systematic engineering attack on the four obstacles** that make cross-domain physics foundation models hard (catalogued accessibly in [[walrus-overview]]): harmonic-analysis-grounded **patch jittering** for rollout stability, **2D-into-3D augmentation** for dimensional unification, **adaptive-compute tokenization** for mixed resolution, and **topology-aware sampling** for training throughput. The overarching empirical lesson is **diversity-first pretraining**: optimizing pretraining loss on a narrow distribution produces *worse* general-purpose models.

Walrus is the autoregressive counterpart to [[poseidon-pde-foundation-model]]'s operator-learning approach, and the large-scale sibling of the smaller [[gphyt-physics-foundation-model]].

---

## Problem Setting

Learn a single model $M$ such that for *any* physical system $S$:

$$u^S_{t+\Delta t} \approx u^S_t + M(U^S_t),\qquad U^S_t = [\,u^S_{t-\tau\Delta t},\ldots, u^S_t\,].$$

Two design commitments are encoded here:
- **Residual prediction** ($u_{t+\Delta t}=u_t+M(\cdot)$) rather than absolute next-state — the same conditioning argument as GP$_{\text{hy}}$T, and the basis of [[autoregressive-rollout-stability]].
- **History-as-context.** The model infers relative timescales and the operative dynamics from a window $U^S_t$ of past states, *not* from explicit PDE coefficients — so it can run on **experimental data with no known governing equation**. This is the in-context premise it shares with [[gphyt-physics-foundation-model]] and the property that distinguishes a foundation model from a PDE-specific surrogate.

---

## Architecture: Space–Time Factorized Transformer

Walrus uses a transformer with **axial (space–time factorized) attention**: alternating blocks attend over spatial axes and over the temporal axis separately, reducing the $O((N_xN_yN_t)^2)$ cost of full attention to a sum of cheaper per-axis attentions. Key components:

- **Axial RoPE (Rotary Position Encoding)** for spatial attention — relative, resolution-portable spatial positions (rotations in feature space encode displacement).
- **T5-style relative position bias** for **causal** temporal attention — preserves arrow-of-time causality in the history window.
- **QK normalization** (Dehghani et al., 2023) — normalizes query/key before the dot product to bound attention logits and stabilize large-model training (the same role scaled-cosine attention plays in [[poseidon-pde-foundation-model]]'s SwinV2).
- **Convolutional Stride Modulation (CSM) encoder/decoder** — the tokenizer adapts its downsampling **stride per input resolution**, natively handling fields sampled at different resolutions. This is Walrus's distinctive token representation; see [[adaptive-compute-tokens]].

The base tokenization is **spatiotemporal tube/patch** tokenization over the history window (see [[spatiotemporal-tubelet-tokens]], [[patch-embedding-tokens]]), wrapped by CSM for resolution adaptivity.

---

## The Four Novel Contributions

### 1. Patch Jittering (harmonic-analysis stabilization)
Before tokenization, the input is randomly **jittered** — shifted by a small random spatial offset. Fixed patch grids alias high-frequency content and **break translation equivariance** (a feature at a patch boundary is encoded differently than the same feature mid-patch), and these artifacts compound catastrophically over autoregressive rollouts. Jittering averages out the grid-locked aliasing. Derived from a harmonic analysis of patch-compression aliasing, it **reduces long-horizon error in 89% of pretraining scenarios**. This is one of the wiki's most-cited stabilization tricks (also in [[arch-autoregressive-transformer]], [[multimodal-tokenization]]).

### 2. 2D-into-3D Augmentation (dimensional unification)
All 2D data is embedded as a **randomly oriented 2D plane inside a 3D volume** ("a sheet in a thin box"), with tensor-law-aware transformation of vector/tensor field components under the embedding. This prevents the model from learning **dimensionality-specific shortcuts** that would let it cheat on 2D and fail to transfer to 3D, forcing a genuinely dimension-agnostic representation. It is the architectural expression of "one model, all dimensionalities."

### 3. Adaptive-Compute Tokenization (mixed resolution)
CSM-based compute-adaptive patching allocates **different compression levels to different inputs** by resolution/complexity. During pretraining a **fixed token count per axis** is maintained by varying the compression factor — so a 3D snapshot (orders of magnitude more grid points than a 2D one) and a 2D snapshot yield comparable token budgets, making batch construction and compute balancing tractable. See [[adaptive-compute-tokens]].

### 4. Topology-Aware Sampling (throughput)
A distributed-training strategy that ties data sampling across GPU ranks to **minimize task variance within sharding groups** (so a sharding group isn't bottlenecked by one rank drawing a far heavier 3D task). **+262% training throughput** — an infrastructure contribution that is itself a prerequisite for scaling a heterogeneous-data PFM.

---

## Pretraining Data

19 scenarios from **The Well** (Ohana et al., 2025) and **Flowbench** (Tali et al., 2024):
- Domains: astrophysics, geoscience, rheology, plasma physics, acoustics, active matter, classical fluids.
- Both 2D and 3D; diverse boundary conditions and physical parameterizations.
- **63 distinct state-variable types** — the breadth that makes Walrus genuinely cross-domain.

---

## Results

- **State of the art on downstream prediction** (short- and long-horizon) across The Well, Flowbench, PDEArena, PDEGym, PDEBench — outperforming prior physics foundation models.
- **Diversity-first, empirically proven:** restricting pretraining to a narrower distribution *lowers pretraining loss* but *weakens downstream transfer*. Broad diversity is essential for generalization — the same lesson [[poseidon-pde-foundation-model]] later explains mechanistically (diverse operators supply recombinable physical primitives).
- **Ablations confirm each contribution** (patch jittering, 2D→3D augmentation, topology-aware sampling) adds value individually.

---

## Relevance to the Physics Foundation Model Goal

Walrus is the wiki's reference point for **what it actually takes to engineer a large cross-domain physics FM**, as opposed to the conceptual feasibility shown by smaller models. Its analysis of *training-stability obstacles* is the most concrete guidance available for building the wiki's own benchmark architectures ([[arch-autoregressive-transformer]] in particular inherits patch jittering, axial RoPE, QK-norm, residual prediction directly).

It anchors the **autoregressive-prediction pole** of [[pfm-architecture-approaches]], complementary to Poseidon's operator-learning pole and to [[aion-1-astronomy]]'s masked-modeling pole. The three together (Walrus / Poseidon / AION-1) are the wiki's "train once, deploy anywhere" triad.

The **diversity-first** principle and the **tokenization-as-the-hard-part** observation (CSM, jittering) are the throughline connecting Walrus to the [[00-token-representation-overview]] folder — Walrus's token representation choices (adaptive compression + jittered patches) are concrete, validated answers to the resolution-invariance and stability gaps in [[multimodal-tokenization]].

**[AI Inference]:** Walrus's **2D-into-3D augmentation** could be a *general* principle for any multi-dimensional foundation model that must unify 1D spectra, 2D images, and 3D volumes (cf. [[aion-1-astronomy]]'s 39-modality tokenization). Embedding lower-dimensional data into a common higher-dimensional space with covariance-aware transforms is a clean way to force a shared representation across dimensionalities.

**[AI Inference]:** Patch jittering is, in effect, a **Monte-Carlo approximation of translation-equivariant pooling**. A principled alternative would be an *exactly* translation-equivariant tokenizer (e.g. group-convolution patch embedding, or the steerable/coordinate-aware tokens of [[structure-preserving-tokens]]) — which would remove the need to average over random shifts at all, trading data augmentation for an architectural guarantee (encoding-spectrum level 1 → 5 for the symmetry).

**[AI Inference]:** Walrus's autoregressive residual prediction and Poseidon's continuous-time operator learning are not mutually exclusive: a model could use **CSM-adaptive jittered tokens** (Walrus's tokenizer) feeding a **lead-time-conditioned operator head** (Poseidon's output), combining Walrus's resolution/stability engineering with Poseidon's any-time inference and all2all data amplification.

---

## Links

- [[walrus-overview]] — accessible blog-post version (the four-obstacles framing)
- [[poseidon-pde-foundation-model]] — operator-learning PFM (contrast: autoregressive vs. operator)
- [[gphyt-physics-foundation-model]] — smaller sibling; in-context derivative prediction
- [[aion-1-astronomy]] — masked-modeling multimodal sibling from Polymathic AI
- [[adaptive-compute-tokens]] — CSM resolution-adaptive token representation (detail)
- [[spatiotemporal-tubelet-tokens]] / [[patch-embedding-tokens]] — base tokenization
- [[pde-transformer-paper]] — competing architecture
- [[latent-diffusion-physics]] / [[multiscale-diffusion-solar]] — related Polymathic AI work
- [[physics-foundation-models]] — broader context
- [[transformer-architectures]] — axial/factorized attention backbone
- [[autoregressive-rollout-stability]] — the stability challenge Walrus's tricks address
