# Gaussian Process Regression — Tutorial

**Type:** Summary  
**Source:** "An Intuitive Tutorial to Gaussian Process Regression" (arXiv 2009.10862v5, Jie Wang)  
**Related Concepts:** [[gaussian-processes]], [[neural-operators]], [[transfer-learning-fine-tuning]]

---

## Overview

A pedagogical introduction to Gaussian Process Regression (GPR) — a non-parametric Bayesian framework for regression that places a prior directly over functions. Unlike parametric models (neural networks, polynomial regression), a GP is characterized entirely by its mean function and covariance kernel, enabling principled uncertainty quantification without tuning a fixed model architecture. The tutorial covers the mathematical foundations, practical algorithms (Cholesky decomposition), kernel design, hyperparameter learning, and computational limitations.

---

## Key Equations

### Gaussian Process Definition

A GP is a collection of random variables, any finite subset of which is jointly Gaussian:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}'))$$

where $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$ is the mean function (often set to zero) and $k(\mathbf{x}, \mathbf{x}')$ is the covariance kernel encoding prior beliefs about function smoothness.

For $N$ training points $\mathbf{X} = \{x_1,\ldots,x_N\}$, the function values form a multivariate normal:

$$\mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu},\, \mathbf{K})$$

where $K_{ij} = k(x_i, x_j)$ is the Gram matrix.

### Noiseless Posterior

Given observed function values $\mathbf{f}$ at $\mathbf{X}$ and test points $\mathbf{X}_*$, the posterior over $\mathbf{f}_*$ is:

$$\mathbf{f}_* \mid \mathbf{f}, \mathbf{X}, \mathbf{X}_* \sim \mathcal{N}(\boldsymbol{\mu}_*, \boldsymbol{\Sigma}_*)$$

$$\boldsymbol{\mu}_* = \mathbf{K}_*^\top \mathbf{K}^{-1} \mathbf{f}$$

$$\boldsymbol{\Sigma}_* = \mathbf{K}_{**} - \mathbf{K}_*^\top \mathbf{K}^{-1} \mathbf{K}_*$$

where $\mathbf{K}_* = k(\mathbf{X}, \mathbf{X}_*)$ and $\mathbf{K}_{**} = k(\mathbf{X}_*, \mathbf{X}_*)$.

### Noisy Observations

With i.i.d. Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$, observations $y_i = f(x_i) + \epsilon_i$ modify the posterior:

$$\boldsymbol{\mu}_* = \mathbf{K}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$

$$\boldsymbol{\Sigma}_* = \mathbf{K}_{**} - \mathbf{K}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{K}_*$$

### RBF (Squared Exponential) Kernel

The most commonly used kernel, encoding smooth functions:

$$k(x_i, x_j) = \sigma_f^2 \exp\!\left(-\frac{\|x_i - x_j\|^2}{2l^2}\right)$$

where $\sigma_f^2$ is the signal variance and $l$ is the length scale. The length scale controls the characteristic distance over which the function can change appreciably.

### Hyperparameter Optimization

Kernel hyperparameters $\boldsymbol{\Theta} = \{l, \sigma_f, \sigma_n, \ldots\}$ are learned by maximizing the log marginal likelihood:

$$\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\Theta}) = -\frac{1}{2}\mathbf{y}^\top C^{-1} \mathbf{y} - \frac{1}{2}\log|C| - \frac{N}{2}\log 2\pi$$

where $C = \mathbf{K} + \sigma_n^2 \mathbf{I}$. Optimized via gradient ascent.

---

## Computational Algorithm

Direct matrix inversion is $O(N^3)$ via standard methods. The practical algorithm uses Cholesky decomposition:

1. Compute $\mathbf{L} = \text{chol}(\mathbf{K} + \sigma_n^2 \mathbf{I})$ — lower triangular, $O(N^3)$
2. Solve $\mathbf{L}\boldsymbol{\alpha}' = \mathbf{y}$ via forward substitution
3. Solve $\mathbf{L}^\top \boldsymbol{\alpha} = \boldsymbol{\alpha}'$ via backward substitution — gives $\boldsymbol{\alpha} = C^{-1}\mathbf{y}$
4. Posterior mean: $\boldsymbol{\mu}_* = \mathbf{K}_*^\top \boldsymbol{\alpha}$
5. Solve $\mathbf{L}\mathbf{v} = \mathbf{K}_*$ and compute $\boldsymbol{\Sigma}_* = \mathbf{K}_{**} - \mathbf{v}^\top \mathbf{v}$

This requires $O(N^3)$ training and $O(N^2)$ prediction per test point. Memory: $O(N^2)$.

---

## Common Kernels

| Kernel | Expression | Properties |
|---|---|---|
| **RBF / Squared Exponential** | $\sigma_f^2 \exp(-\|x-x'\|^2 / 2l^2)$ | Infinitely differentiable; very smooth |
| **Matérn 3/2** | $(1 + \frac{\sqrt{3}\|x-x'\|}{l})\exp(-\frac{\sqrt{3}\|x-x'\|}{l})$ | Once differentiable; more realistic roughness |
| **Matérn 5/2** | $(1 + \frac{\sqrt{5}\|x-x'\|}{l} + \frac{5\|x-x'\|^2}{3l^2})\exp(-\frac{\sqrt{5}\|x-x'\|}{l})$ | Twice differentiable; popular for physical data |
| **Periodic** | $\exp(-\frac{2\sin^2(\pi\|x-x'\|/p)}{l^2})$ | Encodes periodicity with period $p$ |
| **Linear** | $\sigma_b^2 + \sigma_v^2(x - c)(x' - c)$ | Bayesian linear regression as special case |

Kernels can be combined via addition (modeling superposition of independent processes) or multiplication (modeling modulated processes).

---

## Main Results / Properties

1. **Exact posterior is analytic** — no approximate inference needed for standard GPR (unlike deep generative models)
2. **Uncertainty is calibrated** — predictive variance is highest far from training data, properly reflecting epistemic uncertainty
3. **Kernel encodes inductive bias** — length scale, smoothness, periodicity all expressible through kernel choice
4. **Marginal likelihood avoids overfitting** — automatically penalizes overly complex models (Occam's razor via log-det term)
5. **Scalability limitation** — exact GPR fails beyond $N \approx 10{,}000$ due to $O(N^3)$ cost and $O(N^2)$ memory

### Scalable Approximate GP Methods (mentioned in tutorial)

- **Inducing point methods (SVGP):** Select $M \ll N$ inducing points; reduce to $O(NM^2)$
- **Sparse approximations:** Nyström approximation of Gram matrix
- **GPy / GPflow / GPyTorch:** Software packages; GPyTorch uses CG instead of Cholesky for GPU-friendly inference

---

## Relevance to Physics Foundation Models

### Uncertainty Quantification
GPR provides the principled Bayesian baseline for uncertainty quantification in physics emulation. Neural operators (DeepONet, FNO) are deterministic; GPs offer calibrated predictive intervals — critical for scientific applications where knowing prediction confidence matters as much as the prediction itself.

### Kernel as Physics Prior
The choice of kernel encodes domain knowledge about solution regularity. For PDEs with known smoothness (e.g., Stokes flows), a Matérn kernel with appropriate order captures the Sobolev regularity of the solution. This is directly analogous to the Sobolev regularity preservation approach in PISD's spectral encoding.

### Connection to Neural Operators
In the infinite-width limit, many neural network architectures (including deep networks) converge to GPs (Neural Tangent Kernel / NTK theory). Understanding GPR provides intuition for what neural operators implicitly learn about function-space structure.

### Limitations at Scale
The $O(N^3)$ scaling is why GPs alone cannot replace neural surrogates for high-resolution PDE fields. A PFM processing $256^3$ spatial grids has $\sim 1.7 \times 10^7$ points — far beyond exact GP tractability. However, GP ideas (kernel structure, marginal likelihood, inducing points) inform the design of scalable physics-aware priors.

---

## AI Inference

**[AI Inference]:** The GP posterior mean formula $\boldsymbol{\mu}_* = \mathbf{K}_*^\top \mathbf{K}^{-1} \mathbf{f}$ is structurally identical to the attention mechanism $\text{Attn}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d})V$ when the kernel matrix is replaced by the (normalized) dot-product similarity. This is not a coincidence — transformer attention can be interpreted as performing kernel regression with a learned kernel. A PFM could exploit this connection explicitly by parameterizing attention kernels as physics-informed kernels (e.g., Matérn kernels adapted to the local PDE operator), providing stronger inductive bias than pure data-driven attention while remaining scalable.

**[AI Inference]:** Gaussian processes serve as an ideal evaluation baseline for PFMs: if a PFM trained on diverse physics cannot outperform a GP fitted with a carefully chosen physics-informed kernel on a specific downstream task, the PFM has not successfully transferred meaningful prior knowledge. This suggests a principled evaluation protocol: compare PFM fine-tuning against GP with PDE-adapted kernel on held-out benchmarks.

---

## See Also

- [[gaussian-processes]]
- [[neural-operators]]
- [[transfer-learning-fine-tuning]]
- [[pisd-physics-informed-spectral-diffusion]]
- [[deeponet-multi-operator]]
- [[varmion-viscous-flows]]
