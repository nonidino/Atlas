# Architecture: Diffusion / Generative Backbone

**Type:** Architecture Specification  
**Status:** Best-supported (from existing research)  
**Date:** 2026-05-14  
**Related Concepts:** [[diffusion-models-physics]], [[autoregressive-rollout-stability]], [[transformer-architectures]], [[physics-foundation-models]], [[possible-architectures]]  
**Related Summaries:** [[pisd-physics-informed-spectral-diffusion]], [[latent-diffusion-physics]], [[pde-transformer-paper]]

---

## Conceptual Overview

Where the Autoregressive Transformer predicts a single deterministic next state, the Diffusion Backbone samples from a **distribution** over physically plausible next states. This distinction is not cosmetic — for chaotic and turbulent systems, there is no single "correct" next state from an observation. The true dynamics are intrinsically uncertain, and a model that forces a point estimate will produce unphysical over-smoothed outputs that violate the actual statistical properties of the flow.

The diffusion model learns the geometry of physically plausible fields. It denoises random noise into a coherent physical field, conditioned on the prior trajectory. At inference time, this same trained model can be steered to satisfy arbitrary PDE constraints without retraining — via Diffusion Posterior Sampling (DPS). This "training-free physics conditioning" is the diffusion analog of in-context learning: the governing equation is not part of training, it is injected at test time.

**Key empirical result:** Diffusion-based emulators consistently outperform deterministic surrogates for chaotic PDEs, even at 1000× spatial compression ("Lost in Latent Space"). This is the generative approach's central advantage.

---

## Internal Architecture

The architecture has four separable components:

1. **Physics Autoencoder** — compresses physical fields to a compact latent representation
2. **Spectral Normalization** — maps the latent space into a frequency-whitened spectral representation
3. **Diffusion Denoiser** — the core neural network; a U-Net or DiT operating in spectral latent space
4. **DPS Physics Guidance** — test-time enforcement of PDE constraints (optional, without retraining)

---

### Component 1: Physics Autoencoder

**Purpose:** Compress high-resolution physical fields to a tractable latent space for the diffusion model.

**Architecture:**  
A convolutional encoder-decoder pair trained with reconstruction loss:

$$\text{Encoder}: \mathbf{u} \in \mathbb{R}^{F \times H \times W} \to \mathbf{z} \in \mathbb{R}^{C_\text{latent} \times (H/r) \times (W/r)}$$

$$\text{Decoder}: \mathbf{z} \in \mathbb{R}^{C_\text{latent} \times (H/r) \times (W/r)} \to \hat{\mathbf{u}} \in \mathbb{R}^{F \times H \times W}$$

where $r$ is the spatial downsampling factor. Empirically: $r = 32$ for 2D, $r = 8$ for 3D; compression robustness holds even at $r = 32$ ("Lost in Latent Space").

**Key design choices:**
- No KL divergence penalty on the latent. Instead, bound latents with a saturating nonlinearity:
$$\mathbf{z}_\text{bounded} = \frac{\mathbf{z}}{\sqrt{1 + \mathbf{z}^2 / B^2}}, \quad B = 5$$
This prevents runaway latent magnitudes without forcing a particular distribution shape.
- **Optimizer:** PSGD (preconditioned stochastic gradient descent) dramatically accelerates autoencoder training vs. Adam — a practical implementation detail from "Lost in Latent Space."

**Training objective:**
$$\mathcal{L}_\text{AE}(\phi, \psi) = \mathbb{E}_{\mathbf{u}}\!\left[\|\mathbf{u} - \mathcal{D}_\psi(\mathcal{E}_\phi(\mathbf{u}))\|_2^2\right]$$

The autoencoder is trained independently of the diffusion model and fixed during diffusion training.

---

### Component 2: Spectral Normalization

**Purpose:** Map latent representations into a frequency-whitened spectral space where PDE regularity is preserved during diffusion.

**Spectral Encoder:**  
Given a latent field $z: \Omega \to \mathbb{R}$, compute the discrete Fourier transform and normalize frequency-wise by the empirical standard deviation:

$$\mathcal{E}_\text{spectral}(z)(k) = \frac{\hat{z}(k)}{s_k}, \quad s_k^2 = \text{Var}\!\left(\widehat{Z_\text{data}}(k)\right)$$

This whitens the spectrum: all wavenumber components have unit variance in spectral latent space. Without this, the diffusion model would learn to ignore high-frequency components (which have small variance but carry physically critical small-scale structure).

**Spectral Decoder:**  
$$\mathcal{D}_\text{spectral}(\tilde{z})(k) = s_k \cdot \tilde{z}(k), \quad z = \mathcal{F}^{-1}(\mathcal{D}_\text{spectral}(\tilde{z}))$$

**Why this matters — Sobolev Regularity Lemma (PISD):**  
If the data distribution lies in Sobolev space $H^s(\Omega)$ (the correct regularity class for solutions to elliptic/parabolic PDEs), then the spectral normalization $\mathcal{E}_\text{spectral}$ preserves this:

$$\mathbb{E}[\|z\|_{H^s}^2] = \sum_k (1 + |k|^2)^s \cdot s_k^2 < \infty$$

Crucially, the **noisy intermediate states** during diffusion $\tilde{z}_t = \sqrt{\bar{\alpha}_t} \tilde{z}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ also lie in $H^s$, because the noise $\epsilon$ is added in the whitened space and thus has the same frequency decay as the data. Standard pixel-space diffusion adds noise uniformly, potentially destroying differentiability and making PDE guidance operators ill-defined at intermediate noise levels.

---

### Component 3: Diffusion Denoiser

**Forward process (rectified flow, preferred):**  
$$z_t = (1-t) z_0 + t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad t \in [0, 1]$$

This is a linear interpolation from data $z_0$ to noise $\epsilon$ — simpler than DDPM's variance schedule and with straight-line probability flow ODE paths.

**Training objective (denoising score matching):**  
$$\mathcal{L}_\text{diff}(\theta) = \mathbb{E}_{t, z_0, \epsilon}\!\left[\left\|D_\theta(\underbrace{z_0 + t(\epsilon - z_0)}_{z_t}, t, \mathbf{c}) - z_0\right\|_2^2\right]$$

where $\mathbf{c}$ is the conditioning signal (prior trajectory in spectral latent space), and $D_\theta$ is the denoiser.

**Conditioning on prior trajectory:**  
The prior $C$-frame trajectory $[\tilde{z}^{t-C+1}, \ldots, \tilde{z}^t]$ in spectral latent space is concatenated to the noisy input along the channel dimension:

$$D_\theta\!\left(\text{concat}(z_t, \tilde{z}^{t-C+1}, \ldots, \tilde{z}^t),\ t,\ \mathbf{c}\right)$$

**Alternatively:** Use a binary mask $b \in \{0,1\}^{C+1}$ to indicate which frames are known vs. noised (MCVD approach from "Lost in Latent Space"). This unifies conditioning and generation in a single framework.

**Denoiser backbone — DiT (Diffusion Transformer):**  
A transformer operating on spectral latent tokens. Preferred over U-Net for physics because:
- No inductive locality bias (physics has global spatial coupling)
- Scales more predictably with parameter count
- Conditioning via AdaLN (adaptive layer norm) from timestep $t$ and context:

$$\text{AdaLN}(\mathbf{z}, \mathbf{c}) = \mathbf{c}_\text{scale} \odot \frac{\mathbf{z} - \mu}{\sigma} + \mathbf{c}_\text{shift}$$

where $(\mathbf{c}_\text{scale}, \mathbf{c}_\text{shift})$ are learned linear functions of the timestep/context embedding.

**Temporal bundling:**  
Rather than predicting one step per diffusion call, predict $n = 4$ consecutive steps jointly. This reduces the number of autoregressive calls by $4\times$ and substantially reduces error accumulation ("Lost in Latent Space").

---

### Component 4: Reverse Process and DPS Physics Guidance

**Standard reverse process (inference without physics guidance):**  
Starting from $z_T \sim \mathcal{N}(0, I)$, iterate:

$$z_{t-1} = \tilde{\mu}(z_t, \hat{z}_0), \quad \hat{z}_0 = D_\theta(z_t, t, \mathbf{c})$$

using 3rd-order Adams-Bashforth ODE integration for high-quality samples with few neural function evaluations (NFEs < 20 sufficient).

**DPS Physics Guidance (test-time PDE enforcement):**  
At each reverse step, after computing the denoised estimate $\hat{z}_0$, compute the guidance gradient:

$$\mathbf{g}_t = \nabla_{z_t}\!\left[\lambda_\text{PDE} \|\mathcal{R}(\mathcal{D}(\hat{z}_0))\|_2^2 + \lambda_\text{obs} \|\mathcal{A}(\mathcal{D}(\hat{z}_0)) - \mathbf{y}\|_2^2\right]$$

where:
- $\mathcal{R}$ is the PDE residual operator (e.g., $\mathcal{R}(u) = \partial_t u + u \cdot \nabla u - \nu \Delta u$ for NS)
- $\mathcal{A}$ is an observation operator (point measurements, boundary values, etc.)
- $\mathbf{y}$ is the observation data

**Adam-based guidance step (PISD key innovation):**  
Instead of a fixed step-size gradient update, use Adam's adaptive per-parameter learning rates:

$$z_{t-1} \leftarrow \tilde{\mu}(z_t, \hat{z}_0) - \text{Adam-step}(\mathbf{g}_t)$$

This prevents frequency imbalance in the guidance: without adaptive rates, the gradient updates over-correct low-frequency modes (large magnitude) while under-correcting high-frequency modes (small magnitude), destroying physical small-scale structure.

**Physics guidance schedule:**  
Apply guidance throughout the **entire** reverse process ($t: T \to 1$), not just final refinement steps. Restricting guidance to the last 10% of steps increases PDE residual by ~50× (PISD ablation).

**Critical capability: training-free multi-PDE generalization.**  
The diffusion model is trained purely unconditionally (no physics information in training). Different physics constraints are injected only at inference time via $\mathcal{R}$. The same trained model can solve Poisson, Helmholtz, and Navier-Stokes without retraining — by changing only $\mathcal{R}$ and $\lambda_\text{PDE}$.

---

## Full Forward Pass (Inference)

```
Given: C prior snapshots u^{t-C+1}, ..., u^t
Given: (optional) PDE residual operator R, observations y

1. Encode to latent: z^i = E_phi(u^i) for i = t-C+1, ..., t
2. Apply spectral normalization: z̃^i = E_spectral(z^i)
3. Sample noise: z_T ~ N(0, I)  [same shape as z̃^t]
4. For t' = T, T-1, ..., 1:
   a. Denoise: ẑ_0 = D_theta(concat(z_T, z̃^{t-C+1}, ..., z̃^t), t', c)
   b. (Optional DPS): g = ∇_{z_{t'}} [λ_PDE ||R(D_spectral(ẑ_0))||² + λ_obs ||A(D_spectral(ẑ_0)) - y||²]
   c. Update: z_{t'-1} = μ̃(z_{t'}, ẑ_0) - Adam-step(g)
5. Decode from spectral: z^{t+1} = D_spectral(z_0)  [final denoised]
6. Decode to field: û^{t+1} = D_psi(z^{t+1})
```

For $n$-step temporal bundling, steps 1–6 output $[\hat{u}^{t+1}, \ldots, \hat{u}^{t+n}]$ simultaneously.

---

## Uncertainty Quantification

A unique capability absent from architectures 1 and 3: draw $K$ samples from the same conditioned diffusion model to form a physics ensemble:

$$\{\hat{u}^{t+1}_k\}_{k=1}^K \sim p_\theta(\cdot \mid u^{t-C+1}, \ldots, u^t)$$

The ensemble variance is a natural uncertainty estimate:
$$\sigma^2(x) = \frac{1}{K-1}\sum_{k=1}^K (\hat{u}^{t+1}_k(x) - \bar{\hat{u}}^{t+1}(x))^2$$

For chaotic systems, this variance correctly grows over time — the diffusion model doesn't collapse to the mean trajectory that deterministic models are forced to predict.

---

## Key Design Parameters

| Component | Parameter | Recommended Value |
|---|---|---|
| Autoencoder | Spatial compression $r$ | 4–8 (benchmark scale) |
| Autoencoder | Latent channels $C_\text{latent}$ | 4–16 |
| Autoencoder | Optimizer | PSGD |
| Diffusion | Noise schedule | Rectified flow ($t \in [0,1]$) |
| Diffusion | Denoiser | DiT (prefer over U-Net) |
| Diffusion | Context frames $C$ | 4 |
| Diffusion | Temporal bundling $n$ | 4 |
| Diffusion | Inference NFEs | 20 (Adams-Bashforth) |
| DPS | $\lambda_\text{PDE}$ | Sweep $[0.01, 1.0]$ |
| DPS | Guidance timing | Full process ($T \to 1$) |

---

## Benchmark Adaptation (Burgers')

For 1D Burgers' on 64 grid points:

**Autoencoder:** 1D convolutional; $r = 4$ compression to 16 latent points; $C_\text{latent} = 8$.

**Spectral normalization:** 1D FFT over the 16-point latent; normalize per-wavenumber.

**DiT denoiser:** Treat the 16 latent points as a sequence of tokens; 4 transformer layers, hidden dim 128, 4 heads. ~200K parameters.

**DPS for Burgers':**  
$$\mathcal{R}(u) = \partial_t u + u \partial_x u - \nu \partial_x^2 u \approx \frac{u^{t+1} - u^t}{\Delta t} + u^t \frac{\partial u^t}{\partial x} - \nu \frac{\partial^2 u^t}{\partial x^2}$$

Finite-difference approximation; computable at inference time without any PDE information in training.

---

## [AI Inference]

**[AI Inference]:** The PISD Sobolev regularity lemma provides a theoretical justification for why existing latent diffusion models work for physics even without explicit spectral normalization: the autoencoder's latent space implicitly learns smoother representations of physical fields, partially replicating the spectral normalization effect. Explicitly designing the latent space for Sobolev regularity (as in PISD) should be the default for any physics diffusion model.

**[AI Inference]:** The combination of temporal bundling ($n = 4$ steps) with diffusion sampling could resolve the core tension between diffusion (high per-sample cost) and autoregression (many sequential steps). If $n = 4$ bundling holds quality with only 20 NFEs per bundle, the effective cost is 20/4 = 5 NFEs per physical timestep — competitive with or better than a large transformer forward pass. This deserves direct measurement in the benchmark.

**[AI Inference]:** DPS with Adam guidance is essentially performing a mini-optimization at each reverse diffusion step. The number of Adam steps per guidance iteration is an underexplored hyperparameter. For stiff PDEs (large $\|\mathcal{R}\|$), more Adam steps per guidance step may be needed to converge the residual — analogous to Newton's method convergence for stiff ODEs.

---

## See Also

- [[possible-architectures]] — benchmark plan and comparison table
- [[diffusion-models-physics]] — theoretical foundations
- [[autoregressive-rollout-stability]] — the stability problem diffusion addresses
- [[pisd-physics-informed-spectral-diffusion]] — spectral diffusion + DPS
- [[latent-diffusion-physics]] — latent diffusion for physics emulation
- [[pde-transformer-paper]] — diffusion transformer architecture
- [[arch-autoregressive-transformer]] — Architecture 1: deterministic alternative
- [[arch-neural-differentiator]] — Architecture 3: derivative prediction alternative
