# A Mathematical Perspective on Transformers: Interacting Particle Systems and Clustering

**Source:** arXiv:2312.10794v5  
**Authors:** Borjan Geshkovski (MIT), Cyril Letrouit (Paris-Saclay / CNRS), Yury Polyanskiy (MIT), Philippe Rigollet (MIT)  
**Affiliation:** MIT Department of Mathematics; Laboratoire de mathématiques d'Orsay  
**Related Concepts:** [[transformer-architectures]], [[partial-differential-equations]], [[world-models-physics-ai]], [[diffusion-models-physics]]  
**Related Summaries:** [[transformer-mathematical-framework]], [[kepler-newton-inductive-biases]], [[physicsformer-pinn-ns]]

---

## Overview

This paper develops a rigorous **mathematical framework for analyzing Transformers as mean-field interacting particle systems**, with a focus on long-time clustering behavior. Unlike the PDE discretization perspective of Tai et al. ([[transformer-mathematical-framework]]), this work uses **statistical mechanics and dynamical systems** to characterize what happens to tokens as they propagate through Transformer depth. The central result: tokens (particles) inexorably cluster into a single point under the Transformer dynamics — a phenomenon consistent with empirically observed "token uniformity," "over-smoothing," and "rank collapse" in trained models.

The framework connects Transformers to:
- **Nonlinear transport equations** (continuity equation on the sphere)
- **Wasserstein gradient flows** (energy functional on probability measures)
- **Collective behavior models** (Kuramoto oscillators, Krause/Cucker-Smale models)
- **Optimal sphere configurations** (Tammes problem, Gegenbauer polynomials)

---

## The Interacting Particle System Model

A Transformer operates on $n$ tokens (particles) $x_i(0) \in \mathbb{S}^{d-1}$. Layer normalization constrains all tokens to the unit sphere throughout. The continuous-time dynamics are:

$$\dot{x}_i(t) = \mathbf{P}^\perp_{x_i(t)}\!\left(\frac{1}{Z_{\beta,i}(t)}\sum_{j=1}^{n} e^{\beta\langle Q(t)x_i(t),\, K(t)x_j(t)\rangle} V(t)x_j(t)\right) \tag{SA}$$

where:
- $\mathbf{P}^\perp_x y = y - \langle x, y\rangle x$ is the **tangential projection** onto $T_x\mathbb{S}^{d-1}$ (from layer normalization)
- $Z_{\beta,i}(t) = \sum_k e^{\beta\langle Q(t)x_i(t), K(t)x_k(t)\rangle}$ is the **partition function** (softmax denominator)
- $Q, K, V$ are the learnable query/key/value weight matrices
- $\beta > 0$ is the **inverse temperature** (sharpness of attention; $\beta = 1/\sqrt{d_k}$ in standard transformers)

The **self-attention matrix** is $A_{ij}(t) = e^{\beta\langle Qx_i, Kx_j\rangle} / Z_{\beta,i}$ — an $n\times n$ row-stochastic matrix capturing inter-token attraction weighted by semantic similarity.

### Simplified Model (Q = K = V = I)

The core physics emerges in the simplified case:

$$\dot{x}_i(t) = \mathbf{P}^\perp_{x_i(t)}\!\left(\frac{\sum_{j=1}^{n} e^{\beta\langle x_i(t), x_j(t)\rangle} x_j(t)}{\sum_{k=1}^n e^{\beta\langle x_i(t), x_k(t)\rangle}}\right) \tag{SA, simplified}$$

Qualitative behavior (clustering) is the same for generic random $(Q, K, V)$.

---

## Measure-Theoretic Formulation: Flow on Probability Measures

A Transformer is a **flow map on $\mathcal{P}(\mathbb{S}^{d-1})$** — the space of probability measures on the unit sphere. The input sequence is the empirical measure $\mu(0) = \frac{1}{n}\sum_i \delta_{x_i(0)}$, and the dynamics are governed by the **continuity equation**:

$$\partial_t \mu + \mathrm{div}(\mathcal{X}[\mu]\,\mu) = 0, \quad \mathcal{X}[\mu](x) = \mathbf{P}^\perp_x\!\left(\frac{\int e^{\beta\langle x,y\rangle} y\,d\mu(y)}{Z_{\beta,\mu}(x)}\right)$$

where $Z_{\beta,\mu}(x) = \int e^{\beta\langle x,y\rangle} d\mu(y)$. This is a **mean-field interacting particle system**: each token's velocity depends on the empirical measure of all other tokens.

---

## The Interaction Energy

The **interaction energy** functional

$$\mathsf{E}_\beta[\mu] = \frac{1}{2\beta}\iint e^{\beta\langle x,x'\rangle}\,d\mu(x)\,d\mu(x')$$

is **monotonically increasing** along the Transformer flow:

$$\frac{d}{dt}\mathsf{E}_\beta[\mu(t)] = \int \|\mathcal{X}[\mu(t)](x)\|^2\, Z_{\beta,\mu(t)}(x)\,d\mu(t,x) \geq 0$$

**Key extrema:** (Proposition 3.4)
- **Global minimizer:** the uniform measure $\sigma_d$ on $\mathbb{S}^{d-1}$ (tokens spread uniformly — maximum entropy)
- **Global maximizer:** a Dirac mass $\delta_{x^*}$ at some $x^* \in \mathbb{S}^{d-1}$ (all tokens collapsed to one point)

Since the Transformer dynamics increase $\mathsf{E}_\beta$, they drive the measure toward a **Dirac mass** — this is the mathematical explanation for clustering.

### Wasserstein Gradient Flow Structure

A symmetrized surrogate model (replacing $Z_{\beta,i}$ by $n$):

$$\dot{x}_i(t) = \mathbf{P}^\perp_{x_i(t)}\!\left(\frac{1}{n}\sum_{j=1}^{n} e^{\beta\langle x_i(t), x_j(t)\rangle} x_j(t)\right) \tag{USA}$$

is exactly the **Wasserstein gradient flow** of $\mathsf{E}_\beta$ on $\mathcal{P}(\mathbb{S}^{d-1})$. The true model (SA) is a gradient flow of $\mathsf{E}_\beta$ with respect to a **modified (weighted) Wasserstein metric**:

$$\langle \nabla\psi_1, \nabla\psi_2\rangle_{\mu,\mathsf{E}_\beta} := \int \langle \nabla\psi_1(x), \nabla\psi_2(x)\rangle\,\delta\mathsf{E}_\beta[\mu](x)\,d\mu(x)$$

---

## Main Clustering Theorems

All results establish convergence to a single point $x^* \in \mathbb{S}^{d-1}$ for a.e. initial conditions.

| Regime | Condition | Result |
|--------|-----------|--------|
| Zero temperature | $\beta = 0$ | Theorem 4.1: consensus for a.e. initial $(x_i(0))$ |
| Small $\beta$ | $\beta \leq C/n$ | Theorem 4.3: consensus for a.e. initial conditions |
| Large $\beta$ | $\beta \geq Cn^2$ | Theorem 5.1: consensus for a.e. initial conditions |
| High-dimensional | $d \geq 3$, any $\beta$ | **Theorem 6.1:** consensus always (Geshkovski et al. + MTG17, CRMB24) |
| $d \geq n$ | any $\beta > 0$ | **Theorem 6.3:** exponential convergence $\|x_i(t) - x^*\| \leq Ce^{-\lambda t}$ |

**Cone Collapse (Lemma 6.4):** If all initial tokens lie in an open hemisphere (i.e., $\exists\, w$ s.t. $\langle x_i(0), w\rangle > 0$), then exponential convergence to a single cluster holds for any $\beta, d, n$. Since random initial conditions satisfy this almost surely when $d \geq n$, Theorem 6.3 follows.

### The Two-Phase Metastable Dynamics

In practice (intermediate $\beta$), the dynamics exhibit **two timescales**:
1. **Fast phase:** tokens quickly coalesce into a small number of clusters (distinguishable groups)
2. **Slow phase:** clusters merge pairwise until all tokens collapse to a single point

This explains the apparent paradox: trained models produce diverse outputs despite theoretical convergence to a point mass. The model operates in the **metastable intermediate state** (clustered into a few groups) rather than the infinite-time limit (single point). The metastable clusters correspond to semantically distinct token groups.

---

## Connections to Physics and Mathematics

### Kuramoto Oscillators ($d=2$)

When $d=2$, the Transformer dynamics on $\mathbb{S}^1$ are analogous to the **Kuramoto model** of synchronization of coupled oscillators:

$$\dot{\theta}_i = \frac{1}{n}\sum_{j=1}^{n}\sin(\theta_j - \theta_i)$$

Both systems exhibit phase transitions between synchronized and incoherent states controlled by coupling strength ($\beta$).

### Collective Behavior Models

The Transformer attention mechanism resembles the **Krause opinion dynamics model**:

$$\dot{x}_i = \sum_j a_{ij}(x_j - x_i), \quad a_{ij} = \frac{\phi(\|x_i - x_j\|^2)}{\sum_k \phi(\|x_i - x_k\|^2)}$$

Transformers are more general: coupling is via inner products on the sphere with learnable kernel matrices, with no compact-support restriction.

### Mean-Field Limit

The particle system has a well-posed mean-field limit as $n \to \infty$: the Wasserstein-1 distance between the empirical measure ($n$ particles) and continuous solution grows at most as $e^{O(t)} W_1(\mu_n(0), \mu(0))$.

### Optimal Sphere Configurations

The energy-minimization viewpoint connects to the **Tammes problem** (packing points on a sphere with maximum separation) — the saddle points of $\mathsf{E}_\beta$ correspond to optimal sphere configurations.

---

## Implications for Token Uniformity and Rank Collapse

The theory provides rigorous grounding for empirically observed phenomena:
- **Token uniformity / over-smoothing:** Tokens becoming too similar after many layers — this is early-stage clustering.
- **Rank collapse:** The token embedding matrix losing rank — this is the Dirac mass limit; all tokens converge to the same vector.
- **Depth = integration time:** More layers = more time in the continuous dynamical system = more clustering. This suggests a depth-expressivity tradeoff: too many layers push all tokens toward a single representation.

---

## Relevance to Physics Foundation Models

1. **Depth vs. expressivity tradeoff:** The clustering result implies that very deep transformers may lose information about distinct tokens/features. For physics, where one must distinguish different spatial locations and physical fields, this is a design constraint. Architectures with skip connections that bypass deep layers could mitigate over-clustering.

2. **High-dimensional token embeddings as mitigation:** Theorem 6.1 holds for $d \geq 3$, and metastability is more pronounced at large $d$ — high embedding dimension gives richer intermediate cluster structure, potentially beneficial for representing multiple physical quantities simultaneously.

3. **Temperature $\beta$ as a hyperparameter:** The inverse temperature controls clustering rate. For physics problems requiring long-range correlations (large-scale turbulence, global circulation), smaller $\beta$ might be preferable to preserve multiple semantic clusters. For problems requiring sharp localization (shock waves, boundaries), larger $\beta$ accelerates clustering to dominant features.

4. **The two-phase picture and physics:** The metastable clustered state (fast phase) is where physics modeling happens. The slow second phase (merging clusters) is analogous to information loss in deep models — an argument for **residual connections** that preserve early-layer cluster structure.

**[AI Inference]:** The interaction energy $\mathsf{E}_\beta[\mu]$ is formally analogous to a **mean-field free energy** in statistical mechanics — the same mathematical object that appears in the McKean-Vlasov equation and Landau theory of phase transitions. The Transformer's drive toward maximizing $\mathsf{E}_\beta$ is thus formally equivalent to a **ferromagnetic alignment** of tokens. For physics modeling, this suggests transformers are biased toward representing dominant coherent structures (large-scale features) rather than fine-grained detail — which may explain why physics transformers require explicit multi-scale processing (e.g., patch jittering, hierarchical architectures) to handle both large- and small-scale features.

**[AI Inference]:** The **modified Wasserstein metric** $W_{2,\mathsf{E}_\beta}$ that makes the Transformer a gradient flow is an **energy-weighted transport distance** — tokens that are harder to move (higher interaction energy) contribute less to the metric. This is structurally analogous to how inertia-weighted transport appears in compressible fluid dynamics. There may be a formal connection between the Transformer's token dynamics and the dynamics of tracer particles in turbulent flow — both are mean-field systems with energy-based weighting.

---

## Cross-Links

- [[transformer-architectures]] — extends with dynamical systems / clustering perspective
- [[world-models-physics-ai]] — context length and phase transitions in transformer behavior
- [[partial-differential-equations]] — continuity equation on the sphere; aggregation equations
- [[diffusion-models-physics]] — Wasserstein gradient flow connection
- [[transformer-mathematical-framework]] — complementary mathematical view (IDE / operator splitting)
- [[kepler-newton-inductive-biases]] — context length / world model phase transitions
- [[physicsformer-pinn-ns]] — practical physics transformer architecture
- [[walrus-paper]] — large-scale physics transformer (clustering implications for pretraining)
