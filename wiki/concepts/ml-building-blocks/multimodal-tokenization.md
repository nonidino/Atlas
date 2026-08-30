# Multimodal Tokenization

**Type:** Core Concept  
**Related Sources:** [[multimodal-transformers-survey]], [[aion-1-astronomy]], [[walrus-paper]]  
**Related Concepts:** [[transformer-architectures]], [[physics-foundation-models]], [[in-context-learning-physics]], [[transfer-learning-fine-tuning]], [[neural-operators]]

---

## The Core Problem

Text tokenization is solved: Byte Pair Encoding (BPE) maps any string to a sequence of subword tokens over a vocabulary of ~32K–100K items. Every token is a discrete integer; the transformer sees a uniform sequence.

**Visual and physical data lack a canonical equivalent.** An image is not a string — it is a 2D continuous signal sampled on a grid. A physical field is a function $u: \Omega \subset \mathbb{R}^d \to \mathbb{R}^m$ defined on a spatial domain. The tokenization question is:

> *How do you convert a continuous spatial signal into a sequence of tokens that a transformer can process?*

The answer has implications for:
- What information is preserved vs. discarded
- Whether the token space supports generation vs. understanding only
- How efficiently variable-resolution inputs can be handled
- Whether physical structure (conservation laws, symmetries) survives tokenization

---

## Taxonomy of Visual / Physical Tokenization

### 1. Patch-Based Continuous Tokenization (ViT)

**Mechanism:** Divide the spatial domain $\Omega$ into non-overlapping patches $\{P_i\}_{i=1}^N$ of size $p \times p$ pixels (or grid cells). Each patch is flattened and linearly projected to a $d$-dimensional embedding:

$$z_i = W_E \cdot \text{flatten}(P_i) + b_E, \quad z_i \in \mathbb{R}^d$$

The resulting sequence $[z_1, z_2, \ldots, z_N]$ is the token input to the transformer.

**Properties:**
- **Continuous:** Each token is a real-valued vector — no information discretization at this stage.
- **Lossless within patch:** Exact up to the patch size; sub-patch detail is lost.
- **Resolution-dependent:** The number of tokens $N = (H/p)(W/p)$ scales quadratically with resolution, which scales the attention cost as $O(N^2)$.
- **No semantic compression:** The projection is learned but the tokens carry raw pixel statistics, not semantic or physical concepts.

**Physics analog:** Walrus and GP$_{\text{hy}}$T both use patch-based tokenization on physical fields. Walrus applies "patch jittering" — randomly shifting patch boundaries during training — to reduce alignment artifacts at patch boundaries, which otherwise cause error spikes.

---

### 2. Discrete Tokenization via Vector Quantization

**Mechanism:** A learned encoder maps the input to a continuous latent, which is then quantized to the nearest entry in a learned codebook $\mathcal{C} = \{e_k\}_{k=1}^K$:

$$z = \text{Encoder}(x), \qquad \hat{z} = \arg\min_{e_k \in \mathcal{C}} \|z - e_k\|_2, \qquad \text{token} = k$$

A decoder then reconstructs from the quantized code: $\hat{x} = \text{Decoder}(\hat{z})$.

**Key models:**
- **VQ-VAE** (van den Oord et al., 2017) — introduced the discrete bottleneck. Codebook updated via exponential moving average; straight-through estimator for backpropagation through the non-differentiable argmin.
- **VQ-GAN** — adds adversarial training for higher-fidelity reconstruction.
- **DALL-E** — uses 8192-code codebook; images as discrete token sequences for autoregressive generation.

**Properties:**
- **Discrete tokens:** Each image patch becomes an integer index. Enables treating image generation identically to text generation (shared vocabulary, shared cross-entropy loss).
- **Lossy:** The codebook imposes a representational bottleneck. High-frequency fine detail may be lost.
- **Compatible with LLMs:** Discrete image tokens can be appended to text token sequences in a shared embedding space.
- **Understanding gap:** Discrete tokens are better for generation than for semantic understanding — the quantization loses the smooth semantic structure that continuous embeddings preserve.

---

### 3. Hierarchical / Dual-Codebook Tokenization

**Problem with single-codebook VQ:** Understanding and generation need different information granularities. Understanding needs high-level semantic features (is this a vortex or a shock?). Generation needs pixel-level texture and detail.

**SemHiTok** addresses this with a two-level codebook:
1. **Semantic codebook:** Clusters patches by high-level content (one of $K_s$ semantic codes per patch).
2. **Per-semantics pixel sub-codebook:** For each semantic code, a separate sub-codebook quantizes the low-level texture residual.

$$\text{token} = (k_s, k_p), \quad k_s \in [K_s],\ k_p \in [K_p^{(k_s)}]$$

**TokenFlow (CVPR 2025)** introduces a **dual-codebook architecture** — one codebook optimized for semantic understanding, one for pixel reconstruction — with a shared mapping mechanism that keeps both aligned. For the first time, discrete visual input surpassed LLaVA-1.5 13B in understanding benchmarks (+7.2%), while achieving FID 0.63 for reconstruction.

**[AI Inference]:** For physics, this hierarchy maps onto a natural physical hierarchy: a semantic codebook at the level of flow regimes (laminar, turbulent, shock-dominated, boundary layer), and a pixel/field sub-codebook at the level of local field values. The semantic code tells the model *what kind of physics is happening*; the pixel code tells it the precise values. This is analogous to how multigrid methods work: coarse level captures structure, fine level captures detail.

---

### 4. Compression Modules: Q-Former and Perceiver Resampler

**Core problem:** A high-resolution image produces hundreds of patch tokens ($14 \times 14 = 196$ for ViT-L at 224px, scaling to $\sim 1000$+ for larger images). Feeding this many tokens into an LLM is computationally expensive and sequence-length limited.

**Solution: Fixed-output-length compression** via a learned attention bottleneck.

#### Q-Former (BLIP-2)

A lightweight transformer with $Q$ learnable query vectors (independent of image size):

$$z_q = \text{CrossAttn}(Q, \text{ViT}(x)), \quad |z_q| = Q = 32 \text{ (fixed)}$$

The queries attend over the ViT features and extract a fixed-length representation. Only the 32 vectors $z_q$ are passed to the LLM. The Q-Former is text-conditioned: the query vectors attend to both the image and a text prefix, so the extracted visual information is guided by the query.

**Properties:**
- Fixed-output tokens regardless of image resolution — efficient and LLM-compatible.
- Semantic compression: queries learn to extract task-relevant features, not just patch averages.
- Text-conditional: the same image can yield different Q-Former outputs depending on the text query.

#### Perceiver Resampler (Flamingo)

Similar principle: a transformer with $M$ learned latent vectors attends over variable-length image features:

$$z_l = \text{CrossAttn}(L, \text{ViT}(x)), \quad |z_l| = M = 64 \text{ (fixed)}$$

Less conditioned on text than Q-Former; the resampled tokens are then injected into the frozen LLM via cross-attention layers.

**[AI Inference]:** The Q-Former / Perceiver Resampler pattern directly addresses a key PFM challenge. A physical field at high resolution (e.g., a $1024 \times 1024$ velocity field = 65,536 patches at $1 \times 1$ resolution) would produce an unmanageable token count. A physics-adapted Q-Former could compress this to $Q$ "physics tokens" — each capturing a physically meaningful feature (average vorticity, maximum pressure gradient, kinetic energy spectrum peak, etc.) — regardless of spatial resolution. This would enable **resolution-invariant processing at fixed computational cost**, which is a primary requirement for a PFM. The connection to neural operator ideas (resolution invariance) is direct.

---

### 5. Continuous Token Regression

Rather than quantizing to a codebook, the model predicts continuous token vectors directly. The downstream LLM receives real-valued visual embeddings aligned to the text embedding space via a projector:

$$z = W_{\text{proj}} \cdot \text{ViT}(x) + b_{\text{proj}}$$

**LLaVA** uses this approach: a simple linear projector between a frozen ViT and a frozen LLM. Despite its simplicity, achieves strong instruction-following performance, demonstrating that the LLM can adapt to continuous visual inputs through alignment training.

**InternVL** uses a more sophisticated projector (pixel shuffle + MLP) to reduce token count while preserving spatial resolution.

**Properties:**
- Higher information density than discrete tokens (no quantization loss).
- Requires additional loss functions beyond cross-entropy (e.g., regression loss for continuous predictions).
- Incompatible with pure autoregressive language modeling — cannot generate image tokens from text using a shared vocabulary.
- Superior for understanding tasks; inferior for generation without a separate decoder.

---

### 6. Video / Temporal Tokenization

Extending image tokenization to video adds a temporal dimension. Key approaches:

**3D patch (tube) tokenization:** Each token covers a $p \times p$ spatial patch over $\tau$ consecutive frames — a spatiotemporal "tube":
$$z_i = W_E \cdot \text{flatten}(P_i^{x,y,t}), \quad P_i^{x,y,t} \in \mathbb{R}^{p \times p \times \tau}$$

Walrus uses this approach for physical fields, treating a short window of past physical states as a spatial-temporal volume.

**NVIDIA Cosmos Tokenizer** (2025): A dedicated video tokenizer achieving **8× more total compression** and **12× faster processing** than prior tokenizers, trained on 20 million hours of physical/robotic video data. Converts raw video frames to discrete tokens preserving motion structure at high compression rates. Cosmos was trained on 9,000 trillion tokens from autonomous driving, robotics, and synthetic environments — the largest physical world model training dataset to date.

---

### 7. Wave-Particle Dual Tokenization

A recent proposal ("Wave-Particle Continuous-Discrete Dualistic Visual Tokenization", arXiv:2511.01593) explicitly frames the continuous/discrete trade-off as a duality:
- **Wave (continuous) tokens:** Preserve phase information, smooth gradients, spatial correlations — analogous to a wave in physics.
- **Particle (discrete) tokens:** Localized, symbolic, discrete indices — analogous to particles.

The model maintains both representations simultaneously and uses them for different purposes: continuous tokens for generation quality, discrete tokens for LLM compatibility and semantic reasoning.

**[AI Inference]:** This duality has an exact physics interpretation for PFMs. Physical fields have both a wave-like description (Fourier modes, PDEs) and a particle-like description (Lagrangian tracers, SPH particles). A PFM tokenizer that maintains both representations simultaneously could bridge the continuum/particle gap identified as a major open problem in [[pfm-architecture-approaches]].

---

## Summary Comparison

| Approach | Token Type | Resolution Invariant | LLM Compatible | Information Density | Best For |
|---|---|---|---|---|---|
| Patch (ViT) | Continuous | No (tokens ∝ resolution) | With projector | High (lossless within patch) | Understanding; current PFMs |
| VQ-VAE / VQ-GAN | Discrete (index) | No | Natively (shared vocab) | Lower (codebook bottleneck) | Generation; unified models |
| Dual codebook (TokenFlow) | Discrete (dual index) | No | Natively | Moderate (semantic + pixel) | Understanding + generation |
| Q-Former | Continuous (compressed) | **Yes** (fixed $Q$ tokens) | With projector | Depends on $Q$ | Efficiency; multi-resolution |
| Perceiver Resampler | Continuous (compressed) | **Yes** (fixed $M$ tokens) | With projector | Depends on $M$ | Efficiency; video |
| Continuous projector (LLaVA) | Continuous | No | With projector | Highest | Understanding only |
| Video tube (Walrus, Cosmos) | Continuous or discrete | No | With projector | High | Temporal dynamics |

---

## Implications for Physics Foundation Model Tokenization

### What current methods get right
- **Patch tokenization** is a natural starting point and has been validated in current PFMs (Walrus, GP$_{\text{hy}}$T).
- **Q-Former / Perceiver Resampler** compression enables resolution-invariant processing — essential for a PFM handling simulations at different resolutions.
- **Dual-codebook** approaches show that understanding and generation can be unified, which a PFM needs (predict next state AND generate inverse-problem solutions).

### What is missing for physics
1. **No coordinate-aware tokenization:** Standard patch tokenization ignores the physical meaning of spatial position. A physics tokenizer should embed spatial coordinates alongside field values, enabling the model to know where in $\Omega$ each token comes from.

2. **No scale-aware tokenization:** Physical fields have characteristic scales (Reynolds number, Mach number, Prandtl number). Standard tokenization loses this. A tokenizer that embeds dimensionless numbers alongside field data would let the model reason across physical parameter regimes.

3. **No conservation-respecting tokenization:** Standard VQ codebooks do not enforce that quantized representations satisfy physical conservation laws. A physics-aware codebook could include basis vectors that correspond to known conserved quantities, ensuring that tokenized representations lie in a physically consistent latent space.

4. **No multi-field coupling in tokenization:** For multi-physics problems (velocity + pressure + temperature + magnetic field), current tokenizers handle each field independently. A joint tokenizer that encodes inter-field correlations (e.g., Bernoulli: high velocity ↔ low pressure) would preserve more physical structure.

5. **No irregular-geometry tokenization at scale:** Patch tokenization requires a regular grid. Physical domains are often irregular (turbine blades, biological tissue, terrain). This connects to the GNN-based approaches in [[equivariant-gnns]] — graph-based "tokenization" that operates on unstructured meshes.

**[AI Inference]:** The most promising near-term physics tokenizer may be a **physics-conditioned Q-Former**: $Q$ learnable "physics query tokens" attend over raw field patches, where each query vector is initialized to correspond to a specific physical concept (vorticity, divergence, energy, boundary layer thickness). Training with physics-informed objectives (PDE residual as auxiliary loss) would bias the queries to extract physically meaningful latent representations, achieving both resolution invariance and physical structure preservation.

---

## See Also

- [[00-token-representation-overview]] — **the physics-specific unified-token-representation folder**: one deep-dive page per representation (intuition, math, pros/cons), covering every approach above plus better alternatives. This page is the broad visual/LLM taxonomy; that folder is the per-representation physics treatment.
- [[multimodal-transformers-survey]] — primary source for this page
- [[aion-1-astronomy]] — 39-modality tokenization; most analogous to multi-physics PFM
- [[walrus-paper]] — patch + tube tokenization for physics fields; patch jittering
- [[pisd-physics-informed-spectral-diffusion]] — spectral latent space as alternative to patch tokenization
- [[transformer-architectures]] — attention mechanism that processes token sequences
- [[neural-operators]] — resolution-invariant function-space learning (complementary approach)
- [[physics-foundation-models]] — why tokenization choices matter for the PFM goal
- [[pfm-architecture-approaches]] — tokenization in the context of full architecture choices
- [[equivariant-gnns]] — graph-based alternative to patch tokenization for irregular geometry
