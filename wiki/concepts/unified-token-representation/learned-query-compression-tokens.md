# Learned-Query Compression Tokens (Q-Former / Perceiver)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** BLIP-2 Q-Former, Flamingo Perceiver Resampler ([[multimodal-transformers-survey]]) — *proposed* for physics; precursor to [[physics-conditioned-query-tokens]]
**Related:** [[00-token-representation-overview]], [[adaptive-compute-tokens]], [[physics-conditioned-query-tokens]], [[branch-trunk-operator-tokens]], [[multimodal-tokenization]]

---

## Intuition

Fix the number of tokens regardless of input size by letting a small set of **learned query vectors** *attend into* the raw input and extract a fixed-length summary. Whatever the resolution — $128^2$ or $4096^2$, 2D or 3D — the model emits exactly $Q$ tokens, because $Q$ learnable queries pull out the information they were trained to pull. The queries act like a fixed roster of "questions" asked of every input ("how much vorticity? where is the strongest gradient? what is the energy spectrum peak?"). This decouples token count from grid size — the single cleanest route to **resolution invariance with a fixed compute budget** (criteria 1 and 7 in [[00-token-representation-overview]]) — and is the mechanism behind BLIP-2 and Flamingo. For physics it is so far a *proposal*, the seed of [[physics-conditioned-query-tokens]].

---

## Mathematics

Let $X=\text{Enc}(a)\in\mathbb{R}^{N\times d}$ be raw input features ($N$ = many patches, varies with resolution). Introduce $Q$ learnable query vectors $\mathbf q\in\mathbb{R}^{Q\times d}$ (independent of $N$). Cross-attention compresses:

$$\mathbf z = \text{CrossAttn}(\underbrace{\mathbf q}_{\text{query}},\ \underbrace{X}_{\text{key,value}}) = \text{Softmax}\!\Big(\tfrac{(\mathbf q W_Q)(X W_K)^\top}{\sqrt{d}}\Big)(X W_V)\in\mathbb{R}^{Q\times d}.$$

Only the $Q$ vectors $\mathbf z$ pass downstream — **fixed length** regardless of $N$.
- **Q-Former (BLIP-2):** $Q=32$; queries can be *text-conditioned* (attend jointly to input + a prompt), so extraction is steerable.
- **Perceiver Resampler (Flamingo):** $M=64$ learned latents; resampled tokens injected into a frozen backbone via cross-attention.

Cost is $O(QN)$ (linear in input size) versus $O(N^2)$ for self-attention over the raw tokens — a large saving when $N\gg Q$.

---

## Pros

- **Resolution-invariant token count** — $Q$ fixed for any input size/dimension; the property patch tokens fundamentally lack (criterion 1).
- **Fixed, controllable compute downstream** — backbone always processes $Q$ tokens; decouples model cost from grid (criterion 7).
- **Content-adaptive, semantic compression** — queries learn to extract *task-relevant* features, not blind patch averages; can capture global properties in a single token.
- **Steerable** — conditioning the queries (on a prompt, a parameter, a task) changes what is extracted from the *same* input — a natural conditioning interface ([[pfm-interface-design]]).
- **Frozen-backbone friendly** — the compression bridge is small; pairs with the BLIP-2/Poseidon frozen-backbone transfer pattern ([[multimodal-transformers-survey]], [[transfer-learning-fine-tuning]]).

## Cons

- **Lossy bottleneck** — compressing a high-resolution field to $Q$ tokens discards detail; too small a $Q$ loses fine-scale turbulence/shocks. The accuracy–$Q$ trade-off must be tuned.
- **Queries are opaque (as used in vision)** — vanilla learned queries have no physical meaning; nothing guarantees they extract conserved or physically relevant quantities (the gap [[physics-conditioned-query-tokens]] closes).
- **Spatial localization weakened** — compressed tokens are global summaries; reconstructing a full field or applying local BCs from $Q$ tokens needs a capable decoder.
- **Unproven for dynamical physics** — successful in vision-language understanding; not yet demonstrated as the primary tokenizer of a physics emulator (compression may hurt where small errors compound, [[autoregressive-rollout-stability]]).
- **Cross-attention training can be unstable** if $Q$ is too small or queries collapse to redundancy.

---

## Relationship to other representations

- vs. [[adaptive-compute-tokens]]: both fix the token budget independent of input size. CSM does it **geometrically** (resolution-conditioned stride, keeps a spatial grid of tokens, order-preserving); Q-Former does it by **attention** (content-adaptive, permutation-mixing, can extract global non-local features). Q-Former is more flexible/semantic; CSM preserves spatial locality and is cheaper.
- vs. [[branch-trunk-operator-tokens]]: the branch coefficient vector is a fixed-length compression of the input function — Q-Former is the attention-based generalization (a set of learned summaries instead of one MLP code).
- vs. [[physics-conditioned-query-tokens]]: that page is *this representation with the queries given physical meaning and physics-informed training* — the proposed upgrade.

**[AI Inference]:** Q-Former compression is the most direct fix for the resolution-invariance gap that limits [[gphyt-physics-foundation-model]] (fixed 256×128) and [[poseidon-pde-foundation-model]] (grid-tied patches). A physics Q-Former front-end would let a single PFM ingest any resolution/dimension at fixed downstream cost — but only if the lossiness is controlled, which argues for **adaptive $Q$** (more queries for turbulent inputs, cf. the content-adaptive CSM idea) or for pairing it with a residual high-detail path ([[wave-particle-dual-tokens]]).

**[AI Inference]:** Because the queries are steerable, a Q-Former is also a **task/conditioning interface**: the same field can yield different token sets when queried with different parameter or task embeddings — unifying the tokenizer with the "how do you tell a PFM what to do without text" question in [[pfm-interface-design]].

---

## See also

- [[00-token-representation-overview]] — hub; criteria 1 (resolution) and 7 (compute)
- [[physics-conditioned-query-tokens]] — the physics-meaningful upgrade (proposed)
- [[adaptive-compute-tokens]] — geometric fixed-budget alternative (Walrus CSM)
- [[branch-trunk-operator-tokens]] — branch code as input compression
- [[multimodal-transformers-survey]] — Q-Former / Perceiver source
- [[multimodal-tokenization]] — broader compression-module discussion
- [[wave-particle-dual-tokens]] — compression + residual detail path
- [[pfm-interface-design]] — steerable queries as a conditioning interface
