# Unified Token Representation for Physics Foundation Models — Overview

**Type:** Concept hub (folder index)
**Related Sources:** [[poseidon-pde-foundation-model]], [[walrus-paper]], [[gphyt-physics-foundation-model]], [[aion-1-astronomy]], [[multimodal-transformers-survey]]
**Related Concepts:** [[multimodal-tokenization]], [[neural-operators]], [[transformer-architectures]], [[physics-foundation-models]], [[pfm-interface-design]]

---

## Why this folder exists

The single hardest unsolved problem for a physics foundation model is **the lack of a canonical data template** — obstacle 2 of [[walrus-overview]]. Text has Byte-Pair Encoding: every string becomes one sequence over one vocabulary, so a transformer sees uniform input regardless of language or topic. Physics has no equivalent. A physical state is a function

$$u:\Omega\subset\mathbb{R}^d \to \mathbb{R}^m,$$

sampled on grids that differ in **dimensionality** ($d=1,2,3$), **resolution**, **geometry** (Cartesian, mesh, point cloud), **field content** ($m$ = which of velocity/pressure/temperature/density/B-field/… are present), and **characteristic scales** (Reynolds, Mach, Péclet, …). A *token representation* is the map from this heterogeneous mess to a sequence (or set, or graph) of vectors the backbone can process.

> **The tokenizer is the PFM's bottleneck.** The METER finding ([[multimodal-transformers-survey]]) — visual tokenization matters more than text tokenization — applies *a fortiori* to physics, where no pretrained vocabulary exists. Every empirical PFM result in the wiki is downstream of a tokenization choice, and most failure modes (rollout instability, resolution-locking, lost conservation, missing field coupling) trace back to it.

This folder collects **one page per token representation** — those actually used by past foundation models, plus better alternatives — each with **intuition, mathematics, and pros/cons**. It is the physics-specific, per-representation deep-dive complementing the broader visual/LLM taxonomy in [[multimodal-tokenization]].

---

## What a good physics tokenizer must deliver

A scorecard used across the pages in this folder:

1. **Resolution invariance** — same model on $128^2$ and $1024^2$ without retraining (a neural-operator property; patch tokens lack it).
2. **Dimensional / geometric generality** — 1D/2D/3D, Cartesian and irregular, on one representation.
3. **Field/modality unification** — heterogeneous field sets and scalar parameters in one stream.
4. **Scale awareness** — dimensionless governing numbers encoded alongside field values.
5. **Structure preservation** — conservation laws, symmetries, BCs survive tokenization.
6. **Stability under rollout** — no aliasing/grid-locking that compounds over autoregressive steps.
7. **Compute tractability** — token count controllable independent of grid size.
8. **Invertibility / generation** — tokens can be decoded back to a field (needed for emulation).

No existing representation scores well on all eight — which is why this is an open design frontier.

---

## Map of the folder

### Representations used by past foundation models

| Page | Representation | Used by | One-line |
|---|---|---|---|
| [[patch-embedding-tokens]] | Linear patch (ViT) | Walrus, GP$_{\text{hy}}$T, Poseidon, MPP, DPOT | Continuous linear projection of $p\times p$ patches — the default |
| [[spatiotemporal-tubelet-tokens]] | Tubelet (3D patch) | GP$_{\text{hy}}$T, Walrus, Cosmos | A patch over $\tau$ frames — encodes short-time dynamics in the token |
| [[hierarchical-windowed-tokens]] | Multiscale Swin tokens | Poseidon (scOT) | Patch-merging hierarchy + windowed attention — multiresolution |
| [[adaptive-compute-tokens]] | Resolution-adaptive | Walrus (CSM) | Variable compression → fixed token budget across resolutions |
| [[vector-quantized-tokens]] | Discrete codebook (VQ/FSQ/LFQ) | AION-1, DALL-E | Quantize to a learned vocabulary — shared discrete tokens |
| [[scalar-cdf-tokens]] | CDF-binned scalars | AION-1 | Encode scalars/parameters via their empirical CDF |
| [[spectral-fourier-tokens]] | Fourier / spectral modes | FNO, SFNO, PISD | Tokens are spectral coefficients — global, resolution-free |
| [[branch-trunk-operator-tokens]] | Branch/trunk (operator) | DeepONet family | Function and query-point encoded separately |
| [[graph-mesh-tokens]] | Node/edge tokens | GNS, Dynami-CAL, MGNO | Tokens are mesh nodes/particles; geometry-native |
| [[coordinate-implicit-tokens]] | Coordinate / INR | PINNs, implicit nets | Token = coordinate; field is a function of position |
| [[learned-query-compression-tokens]] | Q-Former / Perceiver | BLIP-2, Flamingo (LLM); proposed for physics | Fixed learned queries compress any input |

### Better / proposed alternatives

| Page                                 | Idea                                                                                                                                         |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [[physics-conditioned-query-tokens]] | Q-Former whose queries are *physical operators* (vorticity, divergence, energy) → resolution-invariant + physically meaningful + scale-aware |
| [[wave-particle-dual-tokens]]        | Maintain continuous (wave/spectral) **and** discrete (particle/codebook) tokens simultaneously — bridge the continuum/particle gap           |
| [[structure-preserving-tokens]]      | Tokenizers that are equivariant / conservation-respecting / coordinate-and-scale-aware by construction (encoding-spectrum level 5)           |

### Deep dives into the organizing principles

| Page | Idea |
|---|---|
| [[tokenization-tradeoff-axes]] | In-depth treatment of the three axes — continuous/discrete, local/global, grid-tied/resolution-free — with pros/cons per option, PDE-type mapping, and a design roadmap |

---

## The central trade-offs (read this before the detail pages)

Three axes organize almost every choice:

**(a) Continuous vs. discrete.** Continuous tokens (patch, coordinate, spectral) preserve information and suit regression/generation but break LLM-style shared-vocabulary generation and have no natural compositional symbol. Discrete tokens (VQ/FSQ/CDF) give a shared vocabulary, masked-modeling, and any-to-any inference (AION-1) but impose a lossy bottleneck fatal to fine-scale turbulence. → [[wave-particle-dual-tokens]] refuses the choice.

**(b) Local vs. global.** Patch/tubelet/graph tokens are *local* (each token is a region) — cheap, geometry-flexible, but blind to long-range elliptic coupling (Poisson, gravity, Coulomb) unless many layers stack. Spectral/operator tokens are *global* — each token sees the whole domain, capturing long-range coupling for free, but tied to nice geometries. → [[hierarchical-windowed-tokens]] and [[multiscale-hierarchical-gnn]] interpolate via multiscale.

**(c) Grid-tied vs. resolution-free.** Patch tokens scale $N\propto$ resolution and lock the model to a discretization; neural-operator (spectral, branch/trunk) and compression (Q-Former) tokens decouple token count from resolution. Resolution-freedom is the property that separates a true *operator* from a fixed-grid emulator ([[neural-operators]]).

**[AI Inference]:** The empirically strongest PFMs to date ([[walrus-paper]], [[poseidon-pde-foundation-model]], [[gphyt-physics-foundation-model]]) all use **continuous, local, grid-tied** patch tokens — the *weakest* corner of all three axes on the scorecard — and succeed by *engineering around* the weaknesses (jittering for stability, multiscale for locality, padding for field unification, CSM for resolution). This suggests the field is at a local optimum: patch tokens are easy to train and "good enough," so the harder representations (operator-valued, structure-preserving, dual) remain under-explored despite scoring better on the eight criteria. The next qualitative jump in PFM capability may come from the tokenizer, not the backbone.

---

## See also

- [[multimodal-tokenization]] — broader visual/LLM tokenization taxonomy (parent concept)
- [[neural-operators]] — resolution-invariance as a function-space property
- [[pfm-interface-design]] — how tokens encode lead time, parameters, BCs, field codes
- [[physics-foundation-models]] / [[pfm-architecture-approaches]] — why these choices matter
- [[equivariant-gnns]] — graph "tokenization" for irregular geometry
