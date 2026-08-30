# Adaptive-Compute Tokens (Resolution-Adaptive Patching / CSM)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** [[walrus-paper]] (Convolutional Stride Modulation, CSM)
**Related:** [[00-token-representation-overview]], [[patch-embedding-tokens]], [[hierarchical-windowed-tokens]], [[neural-operators]], [[mixture-of-experts]]

---

## Intuition

Different inputs deserve different amounts of compute. A smooth, low-resolution 2D field needs few tokens; a turbulent, high-resolution 3D field needs many — but if you let the token count float freely, batches become impossible to balance (a 3D snapshot can have *orders of magnitude* more grid points than a 2D one; [[walrus-overview]] obstacle 3). Adaptive-compute tokenization resolves the tension by **varying the patch size / downsampling stride per input so that the *token count stays fixed* regardless of the grid**. High-resolution inputs are compressed harder; low-resolution inputs less. The model always sees a comparable-length sequence, so heterogeneous-resolution data trains stably in one batch.

---

## Mathematics

Let an input have resolution $R$ (grid points per axis). Standard patching with fixed $p$ gives $N(R)=(R/p)^2$ tokens — grid-dependent. CSM instead chooses a **resolution-dependent stride** $s(R)$ (equivalently patch size) to hold the per-axis token count $N_\text{axis}$ fixed:

$$s(R) = \frac{R}{N_\text{axis}}\quad\Longrightarrow\quad N(R)=N_\text{axis}^2\ \ \text{(constant)}.$$

The compression factor $c(R)=s(R)^2 n$ (cells aggregated per token) therefore *increases with resolution*. Concretely Walrus uses a **convolutional** encoder whose stride is modulated by the input resolution (a strided/depthwise conv stack), and a matching decoder that expands tokens back to the native grid. The effective tokenizer is a learned, resolution-conditioned downsampler:

$$\mathbf v = \text{CSM-Enc}_{s(R)}(a),\qquad \hat a = \text{CSM-Dec}_{s(R)}(\mathbf v),\qquad |\mathbf v| = N_\text{axis}^2\ \forall R.$$

This is the tokenizer-level mechanism that lets a **fixed token budget** span 2D and 3D, low and high resolution — the precondition for Walrus's topology-aware batch construction.

---

## Pros

- **Fixed token budget across resolutions/dimensions** — makes mixed-resolution, mixed-dimensionality batches trainable; the key enabler for cross-domain pretraining.
- **Compute allocated by need** — coarse inputs cheap, fine inputs compressed but still represented; aligns FLOPs with information content.
- **Partial step toward resolution invariance** — the *same* model runs on many resolutions (unlike fixed-$p$ patches), though the representation is still grid-derived, not function-space (cf. [[neural-operators]]).
- **Convolutional encoder adds locality smoothing** — strided convs alias less abruptly than hard patch averaging, modestly helping stability.
- **Composable with jittering & multiscale** — CSM is orthogonal to [[hierarchical-windowed-tokens]] and patch jittering and can stack with them.

## Cons

- **High compression loses fine detail** — at high resolution the large stride aggregates many cells per token, discarding sub-token structure exactly where (turbulence, shocks) it matters most. Resolution invariance of *outputs* is only approximate.
- **Not truly discretization-invariant** — unlike spectral/operator tokens, CSM still learns grid-tied conv kernels; behavior can drift across very different resolutions.
- **Stride scheduling is a design burden** — choosing $s(R)$ / token budget trades accuracy against cost and must be tuned.
- **Inherits patch-token semantics gap** — no physics, conservation, or scale awareness in the tokens.
- **Decoder must reconstruct fine grid from few tokens** — a hard upsampling problem at high compression; artifacts possible.

---

## Relationship to other representations

- vs. [[patch-embedding-tokens]]: CSM = patch tokens with a *resolution-conditioned* patch size and a convolutional (not purely linear) embedding.
- vs. [[learned-query-compression-tokens]]: both fix the token count independent of input size. CSM does it by *geometric* stride modulation (spatially local, structure-preserving order); Q-Former does it by *attention* into learned queries (content-adaptive, permutation-mixing). Q-Former can attend to physically meaningful global features; CSM keeps a spatial grid of tokens.
- vs. [[neural-operators]]: CSM approximates resolution invariance discretely; FNO/DeepONet achieve it in function space exactly — CSM is the pragmatic, transformer-friendly middle ground.

**[AI Inference]:** CSM modulates stride by resolution; the natural generalization is to modulate it by **local complexity** — fine tokens where vorticity/gradient magnitude is high, coarse tokens in quiescent regions (a learned, content-adaptive, *spatially varying* stride). This is an attention-free cousin of [[mixture-of-experts]] routing applied to tokenization: spend tokens where the physics is, an "adaptive mesh refinement" for tokens.

**[AI Inference]:** Combining CSM with a function-space decoder (a small neural operator head, [[branch-trunk-operator-tokens]]) could give *true* resolution invariance on output while keeping CSM's training-stability benefits on input — decoupling the (grid-tied) encoder convenience from the (resolution-free) decoder requirement.

---

## See also

- [[00-token-representation-overview]] — hub; criterion 1 (resolution invariance) and 7 (compute tractability)
- [[walrus-paper]] — CSM in context; mixed-resolution training obstacle
- [[walrus-overview]] — the mixed-resolution-instability obstacle in plain terms
- [[patch-embedding-tokens]] — the base it adapts
- [[learned-query-compression-tokens]] — attention-based fixed-budget alternative
- [[neural-operators]] / [[spectral-fourier-tokens]] — true resolution invariance
- [[mixture-of-experts]] — content-adaptive compute allocation
