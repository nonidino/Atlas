# Physics-Informed Diffusion Models in Spectral Space (PISD)

**Type:** Summary  
**Source:** "Physics-informed diffusion models in spectral space" (arXiv 2602.09708v1, Gallon et al., 2026)  
**Related Concepts:** [[diffusion-models-physics]], [[partial-differential-equations]], [[autoregressive-rollout-stability]], [[transformer-architectures]]  
**Related Summaries:** [[latent-diffusion-physics]], [[pde-transformer-paper]]

---

## Overview

PISD introduces a framework for physics-informed generation/inverse-problem solving using diffusion models that operate in **scaled Fourier (spectral) latent space** rather than physical space. The core insight is that representing PDE solutions in their natural spectral basis (1) preserves the Sobolev regularity of the solution throughout the diffusion process, ensuring PDE operators remain well-defined even at intermediate noise levels, and (2) enables efficient physics guidance via Diffusion Posterior Sampling (DPS) with Adam optimization to enforce both PDE residuals and measurement constraints. Results on Poisson, Helmholtz, and Navier-Stokes equations show 100–10,000× lower PDE residuals than DiffusionPDE at 3–15× faster inference.

---

## Method

### Spectral Encoding

Given a PDE solution field $f: \Omega \to \mathbb{R}$, the encoder computes the discrete Fourier transform and normalizes frequency-wise by the empirical standard deviation of the training data at each wavenumber:

$$\mathcal{E}(f)(k) = \frac{\hat{f}(k)}{s_k}, \qquad s_k^2 = \text{Var}\!\left(\widehat{X_\text{data}}(k)\right)$$

This normalization is key: it whitens the spectrum so that all frequency components have unit variance in the latent space, preventing the diffusion model from ignoring high-frequency components that carry physically important small-scale structure.

The decoder inverts this:

$$\mathcal{D}(z)(k) = s_k \cdot z(k), \qquad f = \mathcal{F}^{-1}(\mathcal{D}(z))$$

### Sobolev Regularity Preservation

**Lemma (Sobolev Regularity):** If the data distribution satisfies $X_\text{data} \in H^s(\Omega)$ (Sobolev space of order $s$), then the spectral normalization $\mathcal{E}$ maps $X_\text{data}$ to a distribution in $H^s(\Omega)$ with bounded Sobolev norm. Crucially, the **noisy intermediate states** during diffusion $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ also remain in $H^s$, because the noise $\epsilon$ added in the normalized spectral domain respects the frequency decay of the original data.

This lemma is the theoretical foundation: standard pixel-space diffusion adds noise uniformly across all frequencies, which can destroy the regularity of PDE solutions (e.g., making a smooth velocity field non-differentiable at intermediate timesteps), causing PDE differential operators to be ill-defined during guidance. Spectral normalization prevents this.

### Physics Guidance via DPS + Adam

PISD enforces physics constraints during the reverse diffusion process using Diffusion Posterior Sampling (DPS). At each reverse step $t$, after computing the denoised estimate $\hat{z}_0 = D_\theta(z_t, t)$, the guidance gradient is computed:

$$\nabla_{z_t} \mathcal{L}_\text{physics}(\hat{z}_0) = \nabla_{z_t}\!\left[\lambda_\text{PDE} \cdot \|\mathcal{R}(\mathcal{D}(\hat{z}_0))\|^2 + \lambda_\text{obs} \cdot \|\mathcal{A}(\mathcal{D}(\hat{z}_0)) - \mathbf{y}\|^2\right]$$

where $\mathcal{R}$ is the PDE residual operator and $\mathcal{A}$ is the observation operator (e.g., point measurements, integral constraints).

**Key departure from standard DPS:** Rather than using a single gradient step with a fixed step size, PISD applies **Adam optimization** for each guidance step. Adam's adaptive learning rates per-parameter prevent gradient imbalance between low- and high-frequency components, which would otherwise cause the guidance to over-correct low-frequency modes while under-correcting high-frequency modes.

The full reverse update is:

$$z_{t-1} \leftarrow \tilde{\mu}(z_t, \hat{z}_0) - \text{Adam-step}\!\left(\nabla_{z_t} \mathcal{L}_\text{physics}\right)$$

### Architectural Details

- **Backbone:** U-Net or DiT-style denoiser operating in spectral latent space
- **Guidance schedule:** Physics guidance applied throughout the entire reverse process (timesteps $T \to 1$), not just the final refinement steps (unlike methods that apply guidance only at $t < 0.1T$)
- **Training:** Purely unconditional — the diffusion model is trained without any physics information. Physics is injected only at inference time via DPS. This means the same trained model can enforce different PDEs without retraining.

---

## Results

### Benchmark Tasks

| Task | Equation | Domain |
|---|---|---|
| Poisson | $-\Delta u = f$ | 2D square, Dirichlet BC |
| Helmholtz | $-\Delta u - k^2 u = f$ | 2D square, various $k$ |
| NS Vorticity | $\partial_t \omega + (u \cdot \nabla)\omega = \nu \Delta \omega$ | 2D periodic box |

### Quantitative Comparison vs. DiffusionPDE

| Method | PDE Residual $\downarrow$ | Inference Time $\downarrow$ | Measurement Error $\downarrow$ |
|---|---|---|---|
| DiffusionPDE (pixel-space) | Baseline | Baseline | Baseline |
| **PISD (spectral)** | **100–10,000× lower** | **3–15× faster** | Comparable or better |

The dramatic improvement in PDE residual at lower inference cost is the headline result. The simultaneous speedup comes from the spectral latent space being more compact (fewer modes needed to represent smooth solutions) and from the Adam-based guidance converging in fewer iterations.

### Ablations

- **Without spectral normalization:** Guidance diverges for high-wavenumber components; irregular artifacts in reconstructed fields
- **Without Adam (single gradient step):** Frequency imbalance causes over-smoothing; high-frequency physics structure lost
- **Guidance in last 10% of steps only:** PDE residual increases ~50× vs. full-process guidance; physics constraints insufficiently enforced

---

## Relevance to Physics Foundation Models

### Spectral Latent Space as Universal Physics Representation
Fourier modes are the natural basis for homogeneous PDEs on periodic domains. PISD demonstrates that operating in this space (rather than physical space) gives both theoretical guarantees (Sobolev regularity) and practical benefits (efficiency, guidance stability). A PFM could use spectral latent space as a universal field representation, with modality-specific encoders mapping heterogeneous physical fields to a common spectral basis.

### Training-Free Physics Conditioning
PISD's key architectural decision — unconditional training, physics at inference — enables a single model to solve multiple different PDEs without retraining. This directly aligns with the PFM goal. The model learns the geometry of physically plausible fields; the specific equation is enforced at test time. This is the diffusion analog of in-context learning.

### Complementarity with Autoregressive Methods
PISD is a generative solver for static or slowly-varying PDEs (Poisson, Helmholtz, forward/inverse NS problems). It is not designed for long-horizon autoregressive rollouts. The ideal PFM architecture might combine autoregressive dynamics (Walrus/GP$_{\text{hy}}$T style) for time evolution with PISD-style spectral diffusion for physics-constrained refinement or inverse problem solving.

---

## AI Inference

**[AI Inference]:** The Sobolev regularity lemma in PISD provides a theoretical justification for why existing latent diffusion physics models (e.g., "Lost in Latent Space") work better than pixel-space diffusion even without explicit physics normalization: the autoencoder's latent space implicitly learns a smoother representation of physical fields, partially replicating the effect of spectral normalization. PISD makes this implicit property explicit and theoretically grounded, suggesting that future PFMs using latent diffusion should explicitly design their latent spaces to have controlled Sobolev regularity.

**[AI Inference]:** The Adam-based guidance for DPS is likely to be adopted broadly beyond physics. The frequency-imbalance problem PISD solves (gradient guidance over-correcting low-frequency modes) is a general problem for any diffusion-based inverse solver working with signals that have structured frequency spectra (images, audio, scientific fields). PISD's solution — adaptive per-frequency step sizes — is the correct general fix.

**[AI Inference]:** PISD's test-time-only physics enforcement could enable a compelling hybrid: pretrain a large diffusion model on diverse unlabeled physics data (cheap), then use DPS with equation-specific residuals to solve any target PDE at inference time (zero-shot). Combined with AION-1-style universal tokenization, this could be the generative backbone of a true PFM for inverse problems and data assimilation tasks.

---

## See Also

- [[diffusion-models-physics]]
- [[partial-differential-equations]]
- [[autoregressive-rollout-stability]]
- [[gaussian-processes]]
- [[latent-diffusion-physics]]
- [[pde-transformer-paper]]
- [[aion-1-astronomy]]
- [[gaussian-process-regression]]
