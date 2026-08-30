# Diffusion Models for Physics Emulation

**Type:** Core Concept  
**Related Sources:** Lost in Latent Space, Multiscale Diffusion Solar, PDE-Transformer

---

## Overview

**Diffusion models** (also called score-based generative models or denoising diffusion probabilistic models) are generative models that learn to denoise data progressively, enabling sampling from complex distributions. Applied to physics, they serve as **probabilistic emulators** that model the full distribution of possible future states rather than point predictions.

---

## Continuous-Time Diffusion

Define a forward noising process that progressively corrupts data $x \in \mathbb{R}^N$:
$$p(x_t) = \int \mathcal{N}(x_t \mid \alpha_t x, \sigma_t^2 I)\, p(x)\, dx$$

where $\alpha_t/\sigma_t$ is monotonically decreasing. A family of reverse-time SDEs exists:
$$dx_t = \left[f_t x_t - \frac{1+\eta^2}{2}g_t^2 \nabla_{x_t}\log p(x_t)\right]dt + \eta g_t\, dw_t$$

Setting $\eta=0$ gives the probability flow ODE.

---

## Training: Denoising Score Matching

A neural network $d_\phi(x_t, t)$ is trained to denoise:
$$\arg\min_\phi\; \mathbb{E}_{p(x)p(t)p(x_t\mid x)}\left[\lambda_t \|d_\phi(x_t,t) - x\|_2^2\right]$$

The optimal denoiser is the posterior mean $\mathbb{E}[x\mid x_t]$, related to the score via Tweedie's formula:
$$\mathbb{E}[x\mid x_t] = \frac{x_t + \sigma_t^2 \nabla_{x_t}\log p(x_t)}{\alpha_t}$$

---

## Flow Matching (Rectified Flow)

A cleaner alternative to DDPM noise schedules (used in PDE-Transformer):
$$\alpha_t = 1-t, \quad \sigma_t = t \quad (t \in [0,1])$$

The velocity field $v_\theta$ is learned to transport noise to data:
$$\mathcal{L}_\text{FM} = \mathbb{E}\left[\|v_\theta(\mathbf{x}_t,t) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2\right]$$

where $\mathbf{x}_t = t\mathbf{x}_1 + (1-t)\mathbf{x}_0$, $\mathbf{x}_0 \sim p_\text{noise}$, $\mathbf{x}_1 \sim p_\text{data}$.

---

## Latent Diffusion for Physics

Key finding from "Lost in Latent Space":
- Physics state $x \in \mathbb{R}^{H\times W\times C}$ is compressed by autoencoder to $z \in \mathbb{R}^{H/r\times W/r\times C_\text{lat}}$.
- Diffusion operates in latent space: much fewer tokens → faster training and inference.
- **Robust to compression:** emulation accuracy is surprisingly stable up to $1000\times$ compression.
- **Latent ≥ pixel:** latent emulators match or exceed pixel-space accuracy with fewer parameters.

---

## Conditional Emulation

For physics time series, the diffusion model generates the next $n$ states $z^{i+1:i+n}$ conditioned on the current state $z^i$ and simulation parameters $\theta$:
$$p(z^{i+1:i+n} \mid z^i, \theta)$$

**Temporal bundling** (predict $n>1$ future steps in one call) reduces the number of autoregressive steps and mitigates error accumulation.

**Binary mask conditioning** (MCVD-style): a mask $b \in \{0,1\}^{n+1}$ indicates which elements are known (conditioning) vs. generated:
$$d_\phi(\underbrace{z^{i:i+n}\odot b}_\text{clean} + \underbrace{z_t^{i:i+n}\odot(1-b)}_\text{noisy}, b, \theta, t) \to z^{i:i+n}$$

---

## Advantages of Diffusion for Physics

| Advantage | Mechanism |
|---|---|
| **Captures uncertainty** | Generative model samples from full posterior, not point estimate |
| **Avoids mean prediction** | Does not produce unphysical "blurred" averages for chaotic systems |
| **Rollout stability** | Stochastic sampling avoids deterministic error accumulation |
| **Long-range dependencies** | Multiscale inference schemes enable conditioning on distant past |
| **Quality vs. speed tradeoff** | Fewer NFEs at slight quality cost (Adams-Bashforth integration) |

---

## Disadvantages

| Disadvantage | Impact |
|---|---|
| **Inference cost** | Many neural function evaluations (NFEs) per generation step |
| **Training complexity** | Requires careful noise schedule, sampler, and hyperparameter tuning |
| **Mode dropping** | May not cover tail of distribution in low-data regimes |

Latent diffusion partially addresses the inference cost disadvantage.

---

## Connection to Score Functions and Physics

The score function $\nabla_{x_t}\log p(x_t)$ is related to the Fokker-Planck equation governing probability transport. This connects diffusion models to:
- **Langevin dynamics** in statistical mechanics.
- **Stochastic differential equations** governing physical systems.
- The **fluctuation-dissipation theorem** relating noise and dissipation.

This deep connection suggests that diffusion models may be especially well-suited to modeling physical systems with thermodynamic fluctuations.

---

## See Also

- [[physics-foundation-models]]
- [[autoregressive-rollout-stability]]
- [[transformer-architectures]]
- [[latent-diffusion-physics]]
- [[multiscale-diffusion-solar]]
- [[pde-transformer-paper]]
