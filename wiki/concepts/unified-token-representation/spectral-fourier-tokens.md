# Spectral / Fourier Tokens (Mode Coefficients)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** FNO / SFNO (Fourier Neural Operator family), [[pisd-physics-informed-spectral-diffusion]] (spectral latent diffusion), [[latent-diffusion-physics]]
**Related:** [[00-token-representation-overview]], [[neural-operators]], [[spectral-fourier-tokens]], [[diffusion-models-physics]], [[arch-diffusion-backbone]]

---

## Intuition

Instead of chopping the field into spatial tiles, **transform it into frequency space and let each token be a spectral mode**. A few low-frequency Fourier coefficients already capture the large-scale structure of most PDE solutions; high modes carry fine detail. Because a Fourier coefficient is a *global* property of the field (every grid point contributes to every mode), spectral tokens see the whole domain at once — long-range elliptic coupling (Poisson, gravity, Coulomb) that local patch tokens need many layers to approximate is **native** here. And because the continuous Fourier transform is defined independent of the grid, spectral tokens are **resolution-free**: the same low modes mean the same thing at $128^2$ or $1024^2$. This is the representational heart of the neural-operator program ([[neural-operators]]).

---

## Mathematics

For a field $u(x)$ on a periodic domain, the (truncated) Fourier representation keeps modes $|k|\le k_{\max}$:

$$\hat u(k)=\int_D u(x)\,e^{-2\pi i k\cdot x}\,dx,\qquad u(x)\approx\sum_{|k|\le k_{\max}}\hat u(k)\,e^{2\pi i k\cdot x}.$$

The **FNO spectral convolution layer** parameterizes a learnable complex multiplier $R_\phi(k)$ per retained mode (a global convolution by the convolution theorem):

$$(\mathcal K u)(x)=\mathcal F^{-1}\big(R_\phi(k)\cdot \mathcal F u(k)\big)(x),\qquad R_\phi(k)\in\mathbb{C}^{C\times C},\ |k|\le k_{\max}.$$

So the "tokens" are the retained mode coefficients $\{\hat u(k)\}_{|k|\le k_{\max}}$, and the layer mixes channels per mode. **SFNO** replaces the torus FFT with the **spherical harmonic transform** for data on the sphere (climate/astro). **PISD** ([[pisd-physics-informed-spectral-diffusion]]) runs *diffusion in spectral latent space*, exploiting a Sobolev-regularity lemma: physical fields have rapidly decaying spectra, so a compact set of modes is a faithful, low-dimensional latent — and physics residuals (derivatives) are cheap in spectral space ($\partial_x \to 2\pi i k_x$).

---

## Pros

- **Resolution / discretization invariant** — modes are grid-independent; train at one resolution, evaluate at another (the defining neural-operator property; criterion 1 in [[00-token-representation-overview]]).
- **Global receptive field for free** — each token couples the whole domain; long-range elliptic interactions captured in one layer (where patch tokens need depth).
- **Compact for smooth fields** — fast spectral decay (Sobolev) means few modes suffice; natural compression.
- **Cheap derivatives & physics residuals** — differentiation is multiplication by $ik$; PDE-residual losses and divergence/curl operators are trivial in spectral space (great for physics-informed objectives, [[arch-diffusion-backbone]]).
- **$O(N\log N)$ transform** via FFT.

## Cons

- **Geometry-restricted** — vanilla FFT assumes periodic, rectangular, uniform grids. Non-periodic BCs, irregular domains, and complex geometry break it (partly fixed by Geo-FNO, but with effort).
- **Poor for sharp/local features** — truncating modes causes **Gibbs ringing** at shocks and discontinuities; spectral tokens are bad exactly where patch tokens with derivative channels ([[gphyt-physics-foundation-model]]) shine. Multiscale-spatial methods ([[hierarchical-windowed-tokens]]) handle shocks better.
- **Mode truncation = information loss** — fine-scale turbulence in the high tail is discarded if $k_{\max}$ is modest.
- **Global tokens lose locality** — a single mode says nothing about *where*; hard to apply spatially-local boundary conditions or masks.
- **Complex-valued bookkeeping** and channel-mixing per mode add implementation overhead.
- **Less natural for multi-field coupling at a point** — Bernoulli-type local couplings are spread across all modes.

---

## Relationship to other representations

- vs. [[patch-embedding-tokens]]: spectral is the **global/frequency** dual of local/spatial patches — opposite ends of the local-vs-global axis in [[00-token-representation-overview]]. Patches: sharp features, any geometry, grid-tied. Spectral: smooth/long-range, periodic geometry, resolution-free.
- vs. [[hierarchical-windowed-tokens]]: both target multiscale/long-range; Swin builds a *spatial* pyramid (handles shocks, any Cartesian field), spectral uses a *frequency* decomposition (resolution-free, periodic only).
- vs. [[branch-trunk-operator-tokens]]: both are neural-operator tokenizations achieving resolution invariance; DeepONet via a learned basis (trunk), FNO via the Fourier basis — DeepONet is more geometry-flexible, FNO more efficient on grids.

**[AI Inference]:** Spectral and patch tokens are **complementary, not competing**. A hybrid tokenizer that carries *both* a low-mode spectral token set (global structure, long-range coupling, resolution-free) and a sparse set of spatial patch tokens at shock/interface locations (sharp local features) would cover both ends of the local-global axis — a concrete instance of the [[wave-particle-dual-tokens]] philosophy (wave = spectral, particle = localized).

**[AI Inference]:** Because differentiation is diagonal in spectral space, a **spectral tokenizer makes hard physics constraints cheap**: enforcing $\nabla\cdot u=0$ is a linear projection $\hat u(k)\mapsto \hat u(k)-\frac{k(k\cdot\hat u)}{|k|^2}$ applied per token. A spectral token representation is therefore the most natural host for divergence-free / conservation-constrained tokens ([[structure-preserving-tokens]], [[pc-deeponet-cfd]]).

---

## See also

- [[00-token-representation-overview]] — hub; local-vs-global and resolution axes
- [[neural-operators]] — FNO/SFNO/DeepONet operator-learning framework
- [[pisd-physics-informed-spectral-diffusion]] — spectral latent diffusion + cheap physics residuals
- [[latent-diffusion-physics]] / [[arch-diffusion-backbone]] — spectral latent for generative emulation
- [[patch-embedding-tokens]] / [[hierarchical-windowed-tokens]] — spatial alternatives
- [[branch-trunk-operator-tokens]] — learned-basis operator tokenization
- [[wave-particle-dual-tokens]] — spectral (wave) + localized (particle) hybrid
- [[structure-preserving-tokens]] — divergence-free projection in spectral space
