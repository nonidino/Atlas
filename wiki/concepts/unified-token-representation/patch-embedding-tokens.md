1# Patch-Embedding Tokens (Linear ViT Patches)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** [[walrus-paper]], [[gphyt-physics-foundation-model]], [[poseidon-pde-foundation-model]], MPP, DPOT, PDE-Transformer — *the default tokenizer of essentially every continuum PFM.*
**Related:** [[00-token-representation-overview]], [[spatiotemporal-tubelet-tokens]], [[hierarchical-windowed-tokens]], [[adaptive-compute-tokens]], [[multimodal-tokenization]], [[transformer-architectures]]

---

## Intuition

Take the physical field, cut it into a grid of small square tiles, and turn each tile into one vector by a single learned linear layer. The sequence of tile-vectors is the token sequence. It is the **direct import of the Vision Transformer (ViT)** recipe into physics: a $p\times p$ patch of grid cells plays the role of a "visual word." Nothing about physics is built in — the patch is just a local averaging-and-projection. Its dominance is a story of convenience: it is trivial to implement, plays perfectly with standard transformers, and — with enough data and a few stabilization tricks — works well enough that no one has been forced off it.

---

## Mathematics

Partition $D=[0,1]^2$ (discretized on a $J\times J$ grid) into $P^2=(J/p)^2$ non-overlapping $p\times p$ patches. Stack the $n$ input field channels; patch $j$ is $\mathbf a^p_j\in\mathbb{R}^{p\times p\times n}$. A shared learnable $\mathbf W_{\mathcal E}\in\mathbb{R}^{C\times n\times p\times p}$ and bias $\mathbf b_{\mathcal E}\in\mathbb{R}^C$ embed it to a $C$-dim token ($C>n$):

$$(\mathbf v_j)_i = (\mathbf b_{\mathcal E})_i + \sum_{k=1}^{n}\sum_{u,v=1}^{p}(\mathbf W_{\mathcal E})_{i,k,u,v}\,(\mathbf a^p_j)_{k,u,v},\qquad i=1,\ldots,C.$$

The continuous (function-space) view used by [[poseidon-pde-foundation-model]] (its SM eq. 12) makes the physics explicit — the token is a **weighted patch average** lifted to latent space:

$$\mathbf v(x)=\hat{\mathbf E}(a)(x)=\sum_{\rho=1}^{P^2}\mathbf F\!\left(\int_{D_\rho} W(x)\,a(x)\,dx\right)\mathbb{I}_{D_\rho}(x),\qquad \mathbf F\in\mathbb{R}^{C\times n}.$$

Token count $N=(H/p)(W/p)$ scales **quadratically with resolution**, so full self-attention costs $O(N^2)=O(\text{res}^4)$ — the reason windowed ([[hierarchical-windowed-tokens]]) or factorized attention is needed at scale. Positional information must be added separately (learned, RoPE, or relative-log bias), since the linear projection is position-blind.

---

## Pros

- **Trivially simple & general** — one matmul; works for any field count $n$ by stacking channels (Poseidon pads to a common $n$ across PDEs).
- **Information-preserving within a patch** — continuous (no quantization loss), exact up to sub-patch detail.
- **Backbone-agnostic** — produces a standard token sequence; any transformer consumes it.
- **Empirically validated** — underlies every strong PFM result in the wiki.
- **Invertible** — a linear "patch recovery" decoder reconstructs the field (Poseidon's $\mathbf W_{\mathcal R}$).

## Cons

- **Not resolution-invariant** — token count and the learned $\mathbf W_{\mathcal E}$ are tied to the grid; a model trained at one resolution does not transfer to another without interpolation/retraining (the core gap vs. [[neural-operators]] and [[spectral-fourier-tokens]]).
- **Aliasing & translation-equivariance breaking** — fixed patch boundaries alias high frequencies and encode the same feature differently depending on its position within a patch; these artifacts **compound over autoregressive rollouts** ([[autoregressive-rollout-stability]]). Walrus's **patch jittering** is the standard band-aid; [[structure-preserving-tokens]] is the principled fix.
- **Sub-patch detail is lost** — averaging discards within-patch structure (bad for shocks, thin interfaces unless $p$ is small, which explodes $N$).
- **No physics semantics** — tokens carry raw field statistics, not vorticity/divergence/energy; no conservation or symmetry structure survives.
- **No scale awareness** — patch values give no information about $\mathrm{Re}$, $\mathrm{Ma}$, etc.
- **Quadratic attention cost** in resolution unless windowed/factorized.
- **Regular grid required** — fails on meshes/point clouds (→ [[graph-mesh-tokens]]).

---

## Variants & how PFMs patch the weaknesses

- **Patch jittering** ([[walrus-paper]]): random shift before patching → averages out aliasing; −error in 89% of scenarios.
- **Derivative-augmented channels** ([[gphyt-physics-foundation-model]]): append $\partial_x,\partial_y,\partial_t$ as extra channels so the linear projection sees gradients → ~10× better sharp-gradient resolution.
- **Multiscale / patch merging** ([[poseidon-pde-foundation-model]], [[hierarchical-windowed-tokens]]): hierarchy of patch resolutions recovers locality at multiple scales.
- **Adaptive compression** ([[adaptive-compute-tokens]]): vary $p$/stride to fix the token budget across resolutions.

**[AI Inference]:** Patch tokens are the *local, continuous, grid-tied* corner of all three trade-off axes in [[00-token-representation-overview]] — the weakest on the scorecard — yet they dominate because they are the easiest to train. The accumulated patch-fixes (jitter + derivative channels + multiscale + CSM) are effectively reinventing the properties that operator-valued and structure-preserving tokenizers would provide natively; a from-scratch redesign may eventually beat the patched-patch approach.

---

## See also

- [[00-token-representation-overview]] — the hub and trade-off framing
- [[spatiotemporal-tubelet-tokens]] — temporal extension of patch tokens
- [[hierarchical-windowed-tokens]] — multiscale patch hierarchy (Poseidon)
- [[adaptive-compute-tokens]] — resolution-adaptive patching (Walrus)
- [[spectral-fourier-tokens]] / [[branch-trunk-operator-tokens]] — resolution-free alternatives
- [[structure-preserving-tokens]] — equivariant fix for aliasing
- [[autoregressive-rollout-stability]] — why patch artifacts matter over rollouts
