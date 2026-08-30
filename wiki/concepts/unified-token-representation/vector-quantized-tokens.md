# Vector-Quantized Tokens (Discrete Codebooks: VQ / FSQ / LFQ)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** [[aion-1-astronomy]] (FSQ for images/maps, LFQ for spectra), VQ-VAE / VQ-GAN / DALL-E (vision)
**Related:** [[00-token-representation-overview]], [[scalar-cdf-tokens]], [[wave-particle-dual-tokens]], [[multimodal-tokenization]], [[multimodal-transformers-survey]]

---

## Intuition

Give physics a *vocabulary*. Instead of a continuous vector per patch, learn a finite **codebook** of prototype vectors and snap each encoded patch to its nearest codebook entry; the token is then just the **integer index** of that entry. Now a physical field is a string of symbols over a fixed alphabet — exactly like text. This unlocks everything the LLM stack assumes: a shared cross-entropy loss, masked-token prediction, any-to-any inference, and one common vocabulary across wildly different modalities (images, spectra, scalars). This is how [[aion-1-astronomy]] puts 39 astronomical modalities into a single token stream. The price is a **lossy bottleneck**: continuous detail is rounded to the nearest symbol.

---

## Mathematics

**Vanilla VQ-VAE.** Encoder $z=\text{Enc}(x)$; quantize to the nearest of $K$ codebook vectors $\mathcal C=\{e_k\}_{k=1}^K$:

$$\hat z = e_{k^\ast},\quad k^\ast=\arg\min_k \|z-e_k\|_2,\qquad \text{token}=k^\ast,\qquad \hat x=\text{Dec}(\hat z).$$

The $\arg\min$ is non-differentiable → **straight-through estimator** (copy gradients past the quantizer) plus codebook/commitment losses:

$$\mathcal L = \underbrace{\|x-\hat x\|^2}_{\text{recon}} + \underbrace{\|\text{sg}[z]-e_{k^\ast}\|^2}_{\text{codebook}} + \beta\underbrace{\|z-\text{sg}[e_{k^\ast}]\|^2}_{\text{commitment}},$$

with $\text{sg}[\cdot]$ the stop-gradient. Codebooks are often updated by EMA. Failure mode: **codebook collapse** (few codes used).

**FSQ (Finite-Scale Quantization)** — used by AION-1 for images/maps. Drop the learned codebook entirely: project to a low-dim space, then round each coordinate to a fixed set of levels $L$:

$$\hat z_i = \text{round}\!\big(\tfrac{L-1}{2}\,\tanh(z_i)\big)\cdot\tfrac{2}{L-1},\qquad |\mathcal C|=\prod_i L_i.$$

The implied codebook is the grid $\{$levels$\}^{d}$ — no collapse, no commitment loss, trivially stable. AION-1 trains image FSQ with an **inverse-variance-weighted NLL** $\sum_i \tfrac12\|\Sigma_i^{-1/2}(x_i-\text{Dec}(\text{Enc}(x_i)))\|^2$ so the bottleneck respects per-pixel measurement noise.

**LFQ (Look-up-Free Quantization)** — used by AION-1 for spectra. Quantize each latent dimension independently to a sign bit, giving a binary code of size $2^d$ with no explicit codebook lookup — extreme, collapse-free, large vocabularies.

---

## Pros

- **Shared discrete vocabulary across modalities** — heterogeneous fields/instruments become one token stream; the basis of AION-1's omnimodal unification.
- **Enables masked modeling & any-to-any inference** — cross-entropy over tokens; predict any masked modality from any observed subset (forward *and* inverse in one model).
- **LLM-stack compatible** — append to text tokens, reuse autoregressive/masked machinery, shared embedding table.
- **Compact & quantization-regularized** — discrete bottleneck can denoise and compress.
- **FSQ/LFQ remove codebook collapse** — stable training, very large effective vocabularies, no commitment loss.
- **Generation-friendly** — discrete tokens are natural for autoregressive/masked generative emulation.

## Cons

- **Lossy — fatal for fine-scale physics.** Rounding to a finite alphabet discards exactly the high-frequency detail that turbulence, shocks, and thin interfaces depend on. Acceptable for AION-1's *static* observational data; dangerous for *dynamical* PDE emulation where small errors compound ([[autoregressive-rollout-stability]]).
- **No smooth gradient structure** — discrete tokens lose the smooth latent geometry that continuous embeddings preserve, hurting regression/understanding (the gen-vs-understand tension; [[multimodal-tokenization]]).
- **No physics structure** — codebook entries are statistical prototypes, not conserved quantities or modes; nothing guarantees a quantized state is physically admissible (e.g. divergence-free).
- **Reconstruction ceiling** — error is lower-bounded by codebook granularity; raising $K$/levels raises cost and can still miss tails.
- **Training intricacy (vanilla VQ)** — straight-through gradients, commitment weight, EMA, collapse — FSQ/LFQ mitigate but constrain the latent geometry.

---

## Relationship to other representations

- vs. [[patch-embedding-tokens]]: same patch encoder, then a *quantizer* — continuous → discrete. Trades information for vocabulary-compatibility.
- vs. [[scalar-cdf-tokens]]: CDF binning is the *scalar* special case of discrete tokenization (1D, distribution-aware binning instead of nearest-codebook).
- vs. [[wave-particle-dual-tokens]]: the dual proposal keeps a continuous token *alongside* the discrete one precisely to recover what VQ throws away.

**[AI Inference]:** For a *dynamical* PFM, pure VQ is likely too lossy, but a **residual/hierarchical VQ** (RVQ: quantize, then quantize the residual, repeatedly) could give a discrete vocabulary whose precision is tunable per scale — coarse codes for regime/structure, fine residual codes for detail. This maps onto the dual-codebook idea ([[multimodal-tokenization]]) and onto a physical hierarchy: a "semantic" code for flow regime (laminar/turbulent/shock) + residual codes for local field values.

**[AI Inference]:** A **physics-constrained codebook** — codebook vectors restricted to a basis of physically admissible states (e.g. divergence-free modes, or eigenfunctions of the relevant operator) — would make every quantized state automatically satisfy a constraint, pushing VQ from encoding-spectrum level 1 toward level 5. This is the codebook instantiation of [[structure-preserving-tokens]].

---

## See also

- [[00-token-representation-overview]] — hub; the continuous-vs-discrete trade-off axis
- [[aion-1-astronomy]] — FSQ/LFQ realized across 39 modalities
- [[scalar-cdf-tokens]] — discrete tokenization of scalars/parameters
- [[wave-particle-dual-tokens]] — keep continuous + discrete together
- [[structure-preserving-tokens]] — physics-constrained codebooks
- [[multimodal-tokenization]] — VQ-VAE/GAN/dual-codebook lineage
- [[autoregressive-rollout-stability]] — why lossiness is dangerous for dynamics
