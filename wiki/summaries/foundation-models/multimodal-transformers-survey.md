# Multimodal Learning With Transformers: A Survey (Wu et al., 2023)

**Source:** Medium blog post reviewing Wu, J., Gan, W., Chen, Z., Wan, S., & Yu, P. S. (2023). *Multimodal Large Language Models: A Survey.* arXiv:2311.13165v1.
**Type:** Survey summary (blog review of a survey paper — not the primary paper)
**Related Concepts:** [[multimodal-tokenization]], [[00-token-representation-overview]], [[transformer-architectures]], [[transfer-learning-fine-tuning]], [[in-context-learning-physics]], [[physics-foundation-models]]
**Date Ingested:** 2026-06-26 · **Deepened:** 2026-06-29

---

## Overview

Surveys the landscape of multimodal transformer models — systems that process and integrate multiple data types (images, text, audio, sensor data) simultaneously. Covers four historical phases, five core technical components, notable models, and key challenges. Its relevance to the PFM goal is direct and structural: **physical simulation data is inherently multimodal** — velocity, pressure, temperature, density, magnetic field, particle positions, boundary geometry, and scalar parameters are distinct "modalities" that a PFM must unify in a shared representation. Every design axis the survey enumerates (tokenization, objective, structure, fusion, prompting) is a design axis for a PFM. This page is the bridge between the mature multimodal-LLM literature and the physics-specific token-representation work in [[00-token-representation-overview]].

---

## Four Phases of Multimodal Research

| Phase | Period | Key development | Relevance to PFM |
|---|---|---|---|
| Single modality | 1980–2000 | HMMs (speech), Eigenfaces (vision); separate per-modality processing | Baseline: classical PDE solvers are single-physics |
| Modality conversion | 2000–2010 | HCI, social-signal processing, cross-modal translation | Physics solvers translated between domains ad hoc |
| Modality fusion | 2010–2020 | Deep Boltzmann machines, neural captioning, joint latent spaces | Neural surrogates begin coupling physics fields |
| Large-scale multimodal | 2020+ | CLIP, DALL-E 2, BEiT-3, KOSMOS-1, PaLM-E | Direct PFM analog — "one model, many modalities" |

The trajectory — from siloed processing → translation → fusion → a single joint model — is **exactly the trajectory physics ML is on now**: per-PDE surrogates → transfer learning → cross-domain foundation models ([[walrus-paper]], [[poseidon-pde-foundation-model]], [[aion-1-astronomy]]). The survey's "phase 4" is where physics ML currently sits.

---

## Core Technical Components

### 1. Knowledge Representation (Tokenization)

Three image-tokenization approaches:
- **Region-based** — extract bounding-box object features (Faster R-CNN style). Expensive, detection-dependent. *No clean physics analog* (physical fields have no "objects" to detect — though coherent structures like vortices are a loose analog).
- **Grid-based** — fixed spatial grid cells with pooled features; uniform resolution.
- **Patch-based** — non-overlapping square patches, each linearly projected (ViT paradigm). **Currently dominant**, and the basis of every continuum PFM tokenizer ([[patch-embedding-tokens]]).

Text tokenization evolved Word2Vec (fixed vocab) → **Byte-Pair Encoding / subword** (open vocab, handles rare words; current standard).

> **Key empirical finding (METER model):** improving *visual* tokenization quality has greater downstream impact than improving *text* tokenization — the visual representation is the primary bottleneck. **For PFMs the analog is stronger still:** there is no pretrained "physics vocabulary," so the *field tokenizer is the whole ballgame.* This is the motivating claim behind the [[00-token-representation-overview]] folder.

### 2. Learning Objectives

| Objective | Abbrev. | Description | PFM analog |
|---|---|---|---|
| Image–text contrastive | ITC | Align image/text embeddings in shared space (CLIP) | Align field data with parameter/equation embeddings |
| Masked language modeling | MLM | Reconstruct masked text from visual + partial text | — |
| Masked visual modeling | MVM | Reconstruct masked patches from remaining visual + text | **AION-1's 4M objective**; masked-field pretraining ([[aion-1-astronomy]]) |
| Image–text matching | ITM | Binary: does this pair match? | Consistency check between state and governing parameters |

Combining objectives helps (UNITER: MLM+ITC) but too many simultaneously can hurt (METER). For physics, the MVM family maps directly onto **masked-field modeling** — the alternative to next-step/operator training, validated by [[aion-1-astronomy]].

### 3. Model Structure

- **Encoder-only** — retrieval/classification (CLIP, ALBEF); cannot generate. PFM analog: representation/diagnosis models, linear-probing tasks (cf. AION-1's emergent understanding).
- **Encoder–decoder** — full generation (T5, SimVLM, GPT-4V). **Necessary for physics simulation** (generating future states / solution trajectories). Both [[poseidon-pde-foundation-model]] (U-Net encoder–decoder) and [[walrus-paper]] are encoder–decoder in spirit.

### 4. Information Fusion

- **Single-stream (fusion encoder):** all modality tokens concatenated, processed by shared self-attention → rich cross-modal interaction; cost $O(N_\text{total}^2)$.
- **Dual-stream (dual encoder):** per-modality encoders interacting only late (or via contrastive loss) → cheaper, better for retrieval, but misses fine cross-modal coupling.

**[AI Inference]:** This distinction maps directly onto PFM design. **Single-stream** (shared self-attention over all physics fields at once) captures **field coupling** — Bernoulli (velocity↔pressure), buoyancy (temperature↔velocity), Lorentz force (velocity↔magnetic field) — but is quadratic in total tokens. **Dual-stream** (field-specific encoders, late fusion) is cheaper but blind to inter-field interaction. For genuinely coupled multi-physics (MHD, thermal convection), single-stream or explicit cross-attention coupling is almost certainly required; for weakly-coupled or single-physics regimes, dual-stream suffices. A PFM may need **routed fusion** — single-stream where coupling is strong, dual-stream elsewhere (cf. [[mixture-of-experts]]).

### 5. Prompts

Prompting bridges pretraining and fine-tuning. **For a PFM**, "prompting" is the *context window of physical states* itself — the model reads the regime, boundary conditions, and timescale off the prompt ([[in-context-learning-physics]], [[gphyt-physics-foundation-model]]). The survey's prompt-engineering insights thus translate into **how to construct the conditioning context** for a PFM — which fields, how many snapshots, what scalar parameters — rather than text instructions ([[pfm-interface-design]]).

---

## Notable Models

| Model | Key innovation | PFM relevance |
|---|---|---|
| **BLIP-2** | Q-Former: fixed 32 learnable query tokens extract features from a frozen encoder | Compress dense field data into a fixed-length latent — resolution-invariance ([[learned-query-compression-tokens]]) |
| **Flamingo** | Perceiver Resampler: fixed 64 output tokens from variable-length input; cross-attention with frozen LLM | Same fixed-compression idea; cross-attention field coupling |
| **LLaMA-Adapter** | Adapter modules for efficient fine-tuning; multiscale visual features | Efficient PFM fine-tuning (cf. LoRA in [[deeponet-multi-operator]]) |
| **MiniGPT-4** | ~15M trainable params (linear projection only) | Extreme efficiency via frozen backbones — echoes Poseidon's <0.5% frozen-latent finetune |
| **LLaVA** | GPT-4-generated instruction data; simple linear projector | Continuous-token projection ([[patch-embedding-tokens]]); data-generation strategy |
| **Visual ChatGPT** | Prompt manager chaining visual foundation models | Multi-model routing — analogous to MoE over physics specializations |

The **frozen-backbone + small-bridge** pattern (BLIP-2, MiniGPT-4, LLaMA-Adapter) is independently validated in physics by [[poseidon-pde-foundation-model]]'s frozen-latent finetuning (adapt only embedding/recovery, freeze the physics backbone) — strong convergent evidence that **the backbone holds general physics and adaptation is mostly a tokenizer-alignment problem.**

---

## Challenges Identified

1. **Modality expansion** — each new sensor type needs architectural changes. PFM analog: adding a new physics regime (plasma, EM) may need new tokenization. *(AION-1's additive-tokenizer design is the cleanest mitigation.)*
2. **Training cost** — large multimodal models need distributed compute at scale (cf. Walrus's topology-aware sampling, +262% throughput).
3. **Lifelong / continual learning** — incorporate new knowledge without full retraining. PFM analog: add physics domains without forgetting old ones.
4. **Catastrophic forgetting** — finetuning degrades prior capabilities. Mitigations: small networks + data replay, or a large *frozen* backbone. *(Poseidon's case studies show it does **not** forget pretraining physics during finetuning — it reuses it — suggesting physics representations may be unusually robust to forgetting.)*

---

## Relevance to PFM

1. **Tokenization is the primary bottleneck.** The METER finding (visual tokenization dominates) generalizes a fortiori to physics, where there is no pretrained vocabulary. Hence [[00-token-representation-overview]].
2. **Masked modeling is a viable PFM objective.** The MVM family (→ AION-1's 4M) is an alternative to next-step/operator training, giving any-to-any inference (forward + inverse + fusion).
3. **Frozen-backbone fine-tuning works** in both LLMs (BLIP-2) and physics (Poseidon frozen-latent) — a robust transfer recipe.
4. **Fixed-token compression (Q-Former)** gives resolution-invariance — a primary PFM requirement that patch tokenization alone does not satisfy ([[learned-query-compression-tokens]], [[physics-conditioned-query-tokens]]).
5. **Fusion design (single vs. dual stream)** determines whether multi-physics coupling is captured — a first-order architectural choice for any multi-field PFM.

---

## See Also

- [[multimodal-tokenization]] — detailed tokenization concept (this survey + web research)
- [[00-token-representation-overview]] — physics-specific unified-token-representation hub
- [[learned-query-compression-tokens]] — Q-Former / Perceiver for physics (detail)
- [[aion-1-astronomy]] — masked multimodal modeling realized for science (39 modalities)
- [[poseidon-pde-foundation-model]] — frozen-latent transfer; encoder–decoder physics FM
- [[walrus-paper]] — cross-domain continuum FM; distributed-training at scale
- [[transformer-architectures]] — architecture foundations
- [[in-context-learning-physics]] — context-as-prompt for physics
- [[transfer-learning-fine-tuning]] — frozen-backbone fine-tuning paradigm
- [[mixture-of-experts]] — routed fusion / per-regime specialization
- [[physics-foundation-models]] / [[pfm-interface-design]] — the PFM goal and interface
