# Gaussian Processes

**Type:** Core Concept  
**Related Sources:** GPR Tutorial, PISD  
**Related Concepts:** [[neural-operators]], [[diffusion-models-physics]], [[transfer-learning-fine-tuning]]

---

## Definition

A **Gaussian Process (GP)** is a probability distribution over functions such that any finite collection of function evaluations is jointly Gaussian. Formally:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}'))$$

where:
- $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$ is the **mean function** (typically set to zero)
- $k(\mathbf{x}, \mathbf{x}') = \mathbb{E}[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}'))]$ is the **covariance kernel**

For any finite set $\mathbf{X} = \{x_1,\ldots,x_N\}$:

$$\mathbf{f} = [f(x_1),\ldots,f(x_N)]^\top \sim \mathcal{N}(\boldsymbol{\mu},\, \mathbf{K})$$

where $K_{ij} = k(x_i, x_j)$ is the Gram (kernel) matrix.

---

## Gaussian Process Regression (GPR)

GPR uses the GP as a prior over functions and conditions on observed data to obtain an analytic posterior.

### Posterior Derivation

Given training data $\mathcal{D} = \{(\mathbf{X}, \mathbf{y})\}$ with $y_i = f(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, the joint distribution over training and test outputs is:

$$\begin{pmatrix} \mathbf{y} \\ \mathbf{f}_* \end{pmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\, \begin{pmatrix} \mathbf{K} + \sigma_n^2 \mathbf{I} & \mathbf{K}_* \\ \mathbf{K}_*^\top & \mathbf{K}_{**} \end{pmatrix}\right)$$

Conditioning on observations (using the Schur complement of the block matrix):

$$\mathbf{f}_* \mid \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_*, \boldsymbol{\Sigma}_*)$$

$$\boldsymbol{\mu}_* = \mathbf{K}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$

$$\boldsymbol{\Sigma}_* = \mathbf{K}_{**} - \mathbf{K}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{K}_*$$

This is exact — no approximations required.

---

## Common Kernels

### Radial Basis Function (RBF) / Squared Exponential

$$k_\text{RBF}(x, x') = \sigma_f^2 \exp\!\left(-\frac{\|x - x'\|^2}{2l^2}\right)$$

Stationary (depends only on $\|x - x'\|$); corresponds to infinitely differentiable functions; characterized by length scale $l$ and signal variance $\sigma_f^2$.

### Matérn Family

$$k_{\nu}(x, x') = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\|x-x'\|}{l}\right)^\nu K_\nu\!\left(\frac{\sqrt{2\nu}\|x-x'\|}{l}\right)$$

where $K_\nu$ is the modified Bessel function. Key special cases:

- **Matérn 1/2:** $\exp(-\|x-x'\|/l)$ — Ornstein-Uhlenbeck process; continuous but not differentiable
- **Matérn 3/2:** $(1 + \frac{\sqrt{3}r}{l})\exp(-\frac{\sqrt{3}r}{l})$ — once differentiable; popular for physical data
- **Matérn 5/2:** $(1 + \frac{\sqrt{5}r}{l} + \frac{5r^2}{3l^2})\exp(-\frac{\sqrt{5}r}{l})$ — twice differentiable

For PDE solutions in $H^s(\Omega)$ (Sobolev space), the Matérn-$(s - d/2)$ kernel is theoretically optimal.

### Kernel Composition

Kernels can be combined to encode more complex prior structure:
- **Sum:** $k_{1+2}(x,x') = k_1(x,x') + k_2(x,x')$ — models superposition of independent processes
- **Product:** $k_{1\times2}(x,x') = k_1(x,x') \cdot k_2(x,x')$ — modulated process
- **Periodic:** $k_\text{per}(x,x') = \sigma^2\exp\!\left(-\frac{2\sin^2(\pi|x-x'|/p)}{l^2}\right)$

---

## Hyperparameter Learning

Kernel hyperparameters $\boldsymbol{\Theta} = \{l, \sigma_f^2, \sigma_n^2, \ldots\}$ are optimized by maximizing the log marginal likelihood:

$$\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\Theta}) = -\frac{1}{2}\mathbf{y}^\top C^{-1} \mathbf{y} - \frac{1}{2}\log\det C - \frac{N}{2}\log 2\pi$$

where $C = \mathbf{K} + \sigma_n^2 \mathbf{I}$. The three terms represent:
1. **Data fit:** $-\frac{1}{2}\mathbf{y}^\top C^{-1} \mathbf{y}$ — penalizes misfit with training data
2. **Complexity penalty:** $-\frac{1}{2}\log\det C$ — penalizes overly complex models (Occam's razor)
3. **Normalization:** constant

The balance between terms provides automatic regularization — no explicit regularization parameter is needed.

---

## Computational Complexity

| Operation | Complexity |
|---|---|
| Training (Cholesky) | $O(N^3)$ |
| Prediction (mean) | $O(N)$ per test point |
| Prediction (variance) | $O(N^2)$ per test point |
| Memory | $O(N^2)$ |

**Practical limit:** Exact GPR is tractable up to $N \approx 10{,}000$. Beyond this, approximations are required.

### Scalable Approximations

| Method | Complexity | Description |
|---|---|---|
| **Sparse GP / FITC** | $O(NM^2)$ | $M \ll N$ inducing points; Nyström approximation |
| **SVGP** | $O(NM^2)$ | Stochastic variational inference with inducing points |
| **SKI / KISS-GP** | $O(N)$ | Structured kernel interpolation on grid |
| **GPyTorch CG** | $O(N^2)$ | Conjugate gradients; GPU-parallelizable |

---

## Connection to Neural Networks

### Neural Tangent Kernel (NTK)

In the infinite-width limit, the training dynamics of a neural network with random initialization are equivalent to kernel regression with the Neural Tangent Kernel:

$$k_\text{NTK}(\mathbf{x}, \mathbf{x}') = \mathbb{E}_{\theta \sim \mathcal{N}}\!\left[\nabla_\theta f_\theta(\mathbf{x}) \cdot \nabla_\theta f_\theta(\mathbf{x}')\right]$$

This establishes a formal equivalence between (infinite) neural networks and GPs — providing theoretical intuition for why neural surrogates generalize.

### Attention as Kernel Regression

The attention mechanism can be interpreted as kernel regression:

$$\text{Attn}(Q, K, V)_i = \sum_j \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{j'}\exp(q_i^\top k_{j'}/\sqrt{d})} v_j$$

This is equivalent to GP posterior mean prediction with a softmax-normalized dot-product kernel, where queries $Q$ play the role of test inputs, keys $K$ are training inputs, and values $V$ are training outputs.

---

## GPs for Physics / PDEs

### Physics-Informed Kernels

The kernel can encode known PDE structure:

- **Derivative kernels:** If $f \sim \mathcal{GP}(0, k)$, then $\partial f / \partial x \sim \mathcal{GP}(0, \partial^2 k / \partial x \partial x')$ — differentiating the GP preserves the GP structure. This allows placing GP priors directly on PDE solutions and conditioning on both function values and derivative observations.

- **Green's function kernels:** For a linear operator $\mathcal{L}$, if $\mathcal{L}u = f$ and $f \sim \mathcal{GP}(0, k_f)$, then $u \sim \mathcal{GP}(0, k_u)$ where $k_u = \mathcal{L}^{-1} k_f (\mathcal{L}^{-1})^\top$ — the solution GP has a kernel defined by the Green's function of $\mathcal{L}$.

### Connection to Neural Operators

Neural operators (DeepONet, FNO) learn mappings between function spaces. GPs with function-valued observations correspond to Gaussian process regression in function space — the operator-level generalization of standard GPR. This connection is exploited in approaches like the **Neural Process** family.

### Relationship to PISD

PISD's spectral normalization $s_k^2 = \text{Var}(\widehat{X_\text{data}}(k))$ implicitly defines a GP prior in spectral space: the normalized coefficients $\hat{f}(k)/s_k$ are treated as approximately i.i.d. $\mathcal{N}(0,1)$. This is equivalent to a GP prior with a stationary kernel whose spectral density is $s_k^2$ (by Bochner's theorem). PISD can therefore be viewed as a diffusion model that samples from this GP prior while enforcing physics constraints via DPS guidance.

---

## Advantages and Disadvantages

| Aspect | GPs | Neural Surrogates |
|---|---|---|
| Uncertainty quantification | Exact, calibrated | Requires ensembles or BNNs |
| Data efficiency | High (strong priors) | Low-to-moderate |
| Scalability | $O(N^3)$ — poor | $O(N)$ — excellent |
| Kernel design | Manual or learned | Implicit via architecture |
| Multi-output | Structured covariance | Standard |
| High dimensions | Curse of dimensionality | Better via deep representations |

---

## AI Inference

**[AI Inference]:** The formal equivalence between GP posterior means and attention mechanisms suggests that transformer-based PFMs are implicitly performing kernel regression with learned, input-dependent kernels. Making this connection explicit — by designing attention patterns to mimic known physics-informed kernels (e.g., Green's functions for the Laplacian or wave operator) during the early layers — could provide a principled initialization strategy that dramatically reduces the data needed to train a PFM.

**[AI Inference]:** The marginal likelihood's automatic Occam's razor (complex term $-\frac{1}{2}\log\det C$) has no obvious analog in neural network training. PFMs optimized by standard empirical risk minimization (ERM) lack this automatic complexity penalty, which may explain why large PFMs tend to overfit to the statistical patterns in their training distribution at the expense of generalization to genuinely new physical regimes. Adding a GP-style marginal likelihood term (or its variational bound) to PFM pretraining could improve out-of-distribution generalization.

---

## See Also

- [[neural-operators]]
- [[diffusion-models-physics]]
- [[transfer-learning-fine-tuning]]
- [[partial-differential-equations]]
- [[gaussian-process-regression]]
- [[pisd-physics-informed-spectral-diffusion]]
- [[deeponet-multi-operator]]
