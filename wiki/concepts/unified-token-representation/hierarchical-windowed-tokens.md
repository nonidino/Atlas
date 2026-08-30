# Hierarchical / Multiscale Windowed Tokens (Swin / scOT)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** [[poseidon-pde-foundation-model]] (scOT — SwinV2 U-Net), Swin-Transformer-based emulators
**Related:** [[00-token-representation-overview]], [[patch-embedding-tokens]], [[transformer-architectures]], [[multiscale-hierarchical-gnn]], [[neural-operators]]

---

## Intuition

PDE solutions are **multiscale**: a turbulent flow has a thin boundary layer, mid-scale vortices, and domain-spanning pressure fields, all at once. A flat patch-token sequence at one resolution either resolves the fine scales (huge token count) or the coarse scales (loses detail) — not both. Hierarchical windowed tokens fix this the way a U-Net / multigrid does: start with fine patch tokens, then **repeatedly merge neighboring tokens** to build coarser, higher-channel tokens, processing each scale with cheap *local windowed attention*, and bridging scales with skip connections. The token representation is thus a **pyramid**, not a sequence — each level a different physical scale. Information crosses windows because the windows **shift** every block, so over a few layers every point can influence every other.

This is the token representation behind [[poseidon-pde-foundation-model]], and the paper shows it — not model size — is decisive: a multiscale CNN trained on identical data (CNO-FM) is markedly worse.

---

## Mathematics

**Windowed multi-head self-attention (W-MSA).** Attention acts only inside $M\times M$ windows; a discrete SwinV2 head (Poseidon SM eq. 21):

$$\mathcal A_l(\mathbf v)=\text{Softmax}\!\left(\mathbf B_l(\mathbf v)+\frac{\cos\big((\mathbf v\mathbf W_Q^l+\mathbf 1\mathbf b_Q^{l\top})^\top,(\mathbf v\mathbf W_K^l)^\top\big)}{\tau_l}\right)\big(\mathbf v\mathbf W_V^l+\mathbf 1\mathbf b_V^{l\top}\big).$$

Cost drops from global $O(N^2)$ to $O(N\cdot M^2)$ — **linear in token count** for fixed window size. Two SwinV2 features matter physically: **scaled cosine attention** $\cos(Q,K)/\tau_l$ (bounded logits, stable at scale) and a **continuous relative-log-position bias** from a shared MLP

$$\mathcal P(\Delta x,\Delta y)=\text{ReLU}\big([\,\text{sign}(\Delta x)\log(1+|\Delta x|),\ \text{sign}(\Delta y)\log(1+|\Delta y|)\,]\,\mathbf W_{B,1}+b\big)\,\mathbf W_{B,2}.$$

**Window shifting:** between consecutive blocks the window partition is cyclically displaced by $M/2$, so cross-window information propagates over depth.

**Patch merging (the hierarchy):** at level $i$, four neighboring tokens $\mathbf v\in\mathbb{R}^{4\cdot C\cdot 2^i}$ are linearly fused, halving spatial resolution and doubling channels:

$$\mathcal D_i(\mathbf v,t)=\mathcal N\big(\mathbf W_{\mathcal D_i}\mathbf v,\,t\big),\qquad \mathbf W_{\mathcal D_i}\in\mathbb{R}^{C\cdot 2^{i+1}\times 4\,C\cdot 2^i}.$$

**Patch expansion** inverts this in the decoder; same-scale encoder↔decoder levels are bridged by **ConvNeXt** blocks (U-Net skips). Poseidon fixes $p=4$, $M=16$, $L=4$ levels, heads $[3,6,12,24]$.

---

## Pros

- **Genuinely multiscale** — the token pyramid represents boundary layers, vortices, and global fields simultaneously; matches the multiscale structure of PDE solutions.
- **Linear-cost attention** — windowing makes attention $O(N)$, enabling high effective resolution.
- **Long-range coupling via hierarchy** — coarse levels carry near-global context cheaply (analogous to multigrid / FMM; see [[multiscale-hierarchical-gnn]]), partly mitigating patch tokens' locality weakness.
- **U-Net skips preserve fine detail** — coarse semantics + fine detail recombined at decode.
- **Empirically decisive** — Poseidon's advantage over the matched-data CNO-FM is attributed to this backbone, and even 21M-param Poseidon-T beats larger flat models.
- **Continuous relative position bias** is resolution-portable.

## Cons

- **Still built on linear patch tokens** at the finest level → inherits aliasing / grid-locking / non-resolution-invariance ([[patch-embedding-tokens]]).
- **Window/merge structure assumes a regular grid** — not directly applicable to meshes/point clouds ([[graph-mesh-tokens]]).
- **Fixed hierarchy depth/window size** are hyperparameters tied to expected scale separation; a flow with scales outside the designed range is under-served.
- **Shifted-window bookkeeping** (cyclic shifts, masking) adds implementation complexity.
- **No explicit physics semantics or conservation** in the tokens themselves.

---

## Relationship to other representations

- vs. [[patch-embedding-tokens]]: this *is* patch tokens, arranged in a learned multiscale pyramid with local attention.
- vs. [[spectral-fourier-tokens]]: both target multiscale/long-range structure — spectral does it globally in frequency, Swin does it hierarchically in space. Spectral is resolution-free but geometry-restricted; Swin handles arbitrary Cartesian fields and skip-connects fine detail.
- vs. [[multiscale-hierarchical-gnn]]: the same multigrid/FMM idea on *irregular* geometry — the graph analog of Swin's pyramid.

**[AI Inference]:** scOT's token hierarchy is a **continuous-grid V-cycle**. Pairing it with the *graph* V-cycle of [[multiscale-hierarchical-gnn]] under a shared latent would yield a tokenizer that is multiscale on *both* structured and unstructured domains — covering grids, meshes, and point clouds with one multiresolution representation, a major step toward criterion 2 (dimensional/geometric generality) in [[00-token-representation-overview]].

**[AI Inference]:** Because patch merging is linear and invertible-ish, the merge weights $\mathbf W_{\mathcal D_i}$ could be **constrained to conserve a coarse-grained invariant** (e.g. enforce that the merged token's mean equals the average of its children for a conserved density). That would make the multiscale tokenizer *conservation-respecting across scales* — a fusion of this page with [[structure-preserving-tokens]].

---

## See also

- [[00-token-representation-overview]] — hub
- [[poseidon-pde-foundation-model]] — scOT, the realized example
- [[patch-embedding-tokens]] — the finest-level base tokens
- [[multiscale-hierarchical-gnn]] — the irregular-geometry analog (FMM V-cycle)
- [[spectral-fourier-tokens]] — global alternative for multiscale/long-range
- [[transformer-architectures]] — windowed/factorized attention
- [[structure-preserving-tokens]] — conservation-aware merging
