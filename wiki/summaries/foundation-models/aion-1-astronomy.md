# Summary: AION-1 — Omnimodal Foundation Model for Astronomical Sciences

**Source:** `raw/13724_AION_1_Omnimodal_Foundat/13724_AION_1_Omnimodal_Foundat.md`
**Authors:** Liam Parker, François Lanusse, Jeff Shen et al. — The Polymathic AI Collaboration
**Venue:** NeurIPS 2025
**Date Ingested:** 2026-04-11 · **Deepened:** 2026-06-29

---

## Overview

**AION-1** is the first large-scale **multimodal (omnimodal) foundation model for astronomy**, trained on **200M+ astronomical objects** across **39 data modalities** (multiband images, optical spectra, photometry, astrometry, scalar measurements, segmentation/property maps). It spans 300M–3.1B parameters. Its architecture is the cleanest existing template for the hardest part of a physics foundation model — **unifying genuinely heterogeneous observational data into one token stream** — via a two-stage design: (1) **modality-specific tokenization** into a *shared discrete vocabulary*, then (2) **multimodal masked modeling** (4M-style) over the unified token sequence.

Where [[walrus-paper]] and [[poseidon-pde-foundation-model]] unify *fields of the same continuum type*, AION-1 unifies *qualitatively different instruments and data structures*. This makes it the wiki's reference for the **unified token representation** problem ([[00-token-representation-overview]]) and the **masked-modeling** training pole of [[pfm-architecture-approaches]] (vs. autoregressive Walrus and operator-learning Poseidon).

---

## Architecture

### Stage 1: Universal Tokenization (modality-specific → shared codebook)

AION-1's strategy: each modality gets its *own encoder + quantizer* that maps it into **discrete tokens drawn from a shared vocabulary**, so that downstream a single transformer sees one homogeneous sequence regardless of source instrument. Four tokenizer families:

**Multiband images** — ResNet encoder + **Finite-Scale Quantization (FSQ)**, trained with an **inverse-variance-weighted NLL** so the loss respects per-pixel measurement noise (essential for scientific data where uncertainty varies across the field):
$$\mathcal{L}_\text{NLL} = \sum_i \tfrac{1}{2}\big\|\boldsymbol{\Sigma}_i^{-1/2}\big(\mathbf{x}_i - \text{Dec}_\theta(\text{Enc}_\phi(\mathbf{x}_i))\big)\big\|_2^2.$$

**Spectra** — ConvNeXt V2 encoder + **Look-up-Free Quantization (LFQ)**; spectra are normalized and resampled to a **shared latent wavelength grid** so instruments with different spectral coverage become comparable. (See [[vector-quantized-tokens]] for FSQ/LFQ mechanics.)

**Scalars** — **uniform binning in the cumulative distribution function (CDF)**: each scalar is mapped through its empirical CDF and discretized, which handles arbitrary dynamic ranges and skewed distributions without distribution mismatch, and yields tokens directly compatible with the shared vocabulary. (See [[scalar-cdf-tokens]] — directly relevant to encoding dimensionless physical parameters like Reynolds/Mach number in a PFM.)

**Segmentation / property maps** — convolutional autoencoder + FSQ for normalized $[0,1]$ maps.

### Stage 2: Multimodal Masked Modeling (4M-style)

Given token sequences $\mathbf{X} = \{\mathbf{x}_1,\ldots,\mathbf{x}_M\}$ for the $M$ available modalities of an object, two **disjoint random subsets** are drawn — $\mathbf{x}^\text{obs}$ (observed/conditioning) and $\mathbf{x}^\text{tgt}$ (to predict) — *across all modality tokens jointly*:
$$\mathcal{L}_\text{4M}(\theta) = -\sum_{t=1}^N \log p_\theta\big(\mathbf{x}_t^\text{tgt} \mid \mathbf{x}_t^\text{obs}\big).$$

A **transformer encoder–decoder** performs cross-modal masked-token prediction, learning **intra-modal *and* cross-modal** structure simultaneously (e.g. predict a spectrum's tokens from image tokens, or scalars from both). Because the observed/target split is random over modalities, the *same* trained model supports **any-to-any** inference: any available subset of modalities can predict any other.

---

## Why This Design Matters (the unification mechanism)

The two-stage design solves the "no canonical data template" obstacle ([[walrus-overview]] obstacle 2) by **pushing all heterogeneity into the tokenizers** and keeping the backbone modality-agnostic. Three properties follow:
- **Additive modality support:** a new instrument needs only a new tokenizer into the shared vocabulary; the backbone is untouched (contrast Walrus, where new physics can require architectural change).
- **Graceful missingness:** since training masks arbitrary modality subsets, missing modalities at inference are the *normal case*, not a failure mode.
- **Discrete shared vocabulary** makes masked cross-entropy the single training objective across all data types — the analog of BPE making all text one vocabulary.

---

## Emergent Capabilities

1. **Emergent physical understanding** — non-trivial scientific tasks solved by *simple linear probing* of frozen representations (the representation already encodes the physics).
2. **Low-data-regime superiority** — competitive with supervised baselines trained on orders of magnitude more labeled data.
3. **Flexible data fusion** — arbitrary modality combinations at inference; missing modalities handled natively.
4. **Physically structured latent space** — embeddings organize objects along physically meaningful directions; enables **rare-object retrieval** that outperforms SOTA retrieval methods.

These mirror, in the static/observational setting, the in-context generalization of [[gphyt-physics-foundation-model]] and the compositional transfer of [[poseidon-pde-foundation-model]] — strong evidence that *masked multimodal pretraining produces physically meaningful representations without explicit physics supervision*.

---

## Training Data

200M+ objects (galaxies, stars, quasars) from major surveys (Gaia, SDSS, Pan-STARRS, …): 39 modalities including multiband images (many filters), optical spectra, photometry, astrometry, and scalar properties.

---

## Relevance to the Physics Foundation Model Goal

AION-1 is the closest existing model to a truly **omnimodal scientific foundation model**, and it contributes the two ingredients a continuum PFM most lacks:

1. **A blueprint for unifying heterogeneous physical data** — modality-specific tokenizers into a shared vocabulary, then a modality-agnostic backbone. Directly applicable to multi-physics fields (velocity, pressure, temperature, magnetic field, density) and to mixing field data with scalar parameters. This is the AION row throughout [[00-token-representation-overview]].
2. **Masked modeling as an alternative training objective** to next-step / operator learning — it could replace or complement the autoregressive objective of [[walrus-paper]] / [[gphyt-physics-foundation-model]], giving rich static representations and any-to-any inference (forward, inverse, and data-fusion problems in one model — see [[pfm-interface-design]]).

The **scalar CDF tokenization** is the single most directly transferable idea: encoding $\mathrm{Re}, \mathrm{Ma}, \mathrm{Pr}$, viscosity, etc. as CDF-binned tokens would let a PFM condition on dimensionless governing parameters in the *same* vocabulary as field data, closing the scale-awareness gap flagged in [[multimodal-tokenization]] and [[structure-preserving-tokens]].

**[AI Inference]:** AION-1 and Walrus are **complementary paradigms** for a multimodal physics FM:
- **AION-1 (masked modeling):** rich static representations, arbitrary modality combinations, any-to-any inference.
- **Walrus / Poseidon (autoregressive / operator):** dynamical simulation, time evolution.

A truly general PFM might **share a tokenizer** (AION-1 style) feeding *both* a static representation head (masked) and a dynamical prediction head (autoregressive/operator), jointly trained — so the same discrete physical vocabulary supports both "what is this state?" and "what happens next?".

**[AI Inference]:** AION-1's **any-to-any masked objective is a natural framing for inverse problems**. Forward simulation (IC → solution) and inverse inference (observations → parameters/source) become the *same* operation — predict the masked tokens — differing only in which modalities are observed vs. target. This is a cleaner route to the unified forward/inverse PFM interface ([[pfm-interface-design]]) than bolting an inverse solver onto a forward emulator.

**[AI Inference]:** Discrete VQ tokenization (FSQ/LFQ) imposes a representational bottleneck that is *lossy* for the fine-scale fields turbulent physics needs ([[vector-quantized-tokens]]). For a continuum PFM, AION-1's discrete-token elegance may need to be hybridized with continuous tokens for the dynamical head — a tension the [[wave-particle-dual-tokens]] proposal addresses directly.

---

## Links

- [[walrus-paper]] — sister Polymathic AI project for continuum dynamics (autoregressive)
- [[poseidon-pde-foundation-model]] — operator-learning PFM (third training pole)
- [[gphyt-physics-foundation-model]] — in-context dynamical PFM
- [[00-token-representation-overview]] — AION-1's tokenizers are central exhibits
- [[vector-quantized-tokens]] — FSQ/LFQ discrete tokenization (detail)
- [[scalar-cdf-tokens]] — CDF binning for scalars/parameters (detail)
- [[multimodal-tokenization]] — broader tokenization taxonomy
- [[multimodal-transformers-survey]] — masked modeling objectives (MLM/MVM)
- [[multiscale-diffusion-solar]] — Polymathic AI solar work
- [[physics-foundation-models]] / [[pfm-interface-design]] — broader goal + interface
- [[transformer-architectures]] — shared backbone
- [[transfer-learning-fine-tuning]] — linear probing / downstream use
