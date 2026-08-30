# A Mathematical Explanation of Transformers for Large Language Models and GPTs

**Source:** arxiv:2510.03989  
**Authors:** Xue-Cheng Tai, Hao Liu, Lingfeng Li, Raymond H. Chan  
**Affiliation:** NORCE Norwegian Research Centre; Hong Kong Baptist University; Lingnan University  
**Related Concepts:** [[transformer-architectures]], [[partial-differential-equations]], [[neural-operators]]  
**Related Summaries:** [[pde-transformer-paper]], [[varmion-viscous-flows]]

---

## Overview

This paper provides a rigorous mathematical derivation proving that the standard Transformer architecture (Vaswani et al., 2017) is exactly the discretization of a continuous **integro-differential equation** via operator splitting. Each transformer component — self-attention, layer normalization, feedforward layers, skip connections — emerges from specific substeps of the Lie splitting scheme applied to this continuous equation. The framework unifies Transformers, CNNs, and UNets under the common lens of PDE discretization.

---

## The Continuous Transformer Equation

The full transformer operation is captured by a single structured evolution equation:

$$\begin{cases}
u_t = \underbrace{\langle \gamma(\mathbf{x},\cdot,t;u),\, V(\cdot,\mathbf{y},t;u)\rangle_{\Omega_x}}_{\text{I: attention}} + \underbrace{\partial I_{S_1(\sigma_1(t),\sigma_2(t))}(u)}_{\text{II: layer normalization}} \\[6pt]
\quad\quad\quad + \underbrace{\sum_{j=1}^J \left(\langle W_j(\cdot,\mathbf{y},t), u(\mathbf{x},\cdot,t)\rangle_{\Omega_y} + b_j(\mathbf{x},t)\right) + \partial I_{S_2}(u)}_{\text{III: feedforward network}}, \quad t \in (0,T], \\[6pt]
u(\mathbf{x},\mathbf{y},0) = f(\mathbf{x},\mathbf{y}).
\end{cases}$$

Here:
- $\mathbf{x} \in \Omega_x$ is the **token index** (continuous)
- $\mathbf{y} \in \Omega_y$ is the **feature/embedding dimension** (continuous)
- $u(\mathbf{x},\mathbf{y},t)$: evolving feature representation
- $T$: depth of the transformer (number of layers)

---

## Mathematical Interpretation of Each Component

### Self-Attention as a Non-Local Integral Operator

The QKV projections are integral transformations with learnable kernels $W^Q, W^K, W^V$:

$$Q(\mathbf{x},\mathbf{y},t;u) = \int_{\Omega_y} W^Q(\xi,\mathbf{y},t)\,u(\mathbf{x},\xi,t)\,d\xi$$

$$K(\mathbf{x},\mathbf{y},t;u) = \int_{\Omega_y} W^K(\xi,\mathbf{y},t)\,u(\mathbf{x},\xi,t)\,d\xi$$

$$V(\mathbf{x},\mathbf{y},t;u) = \int_{\Omega_y} W^V(\xi,\mathbf{y},t)\,u(\mathbf{x},\xi,t)\,d\xi$$

The attention score and output are:

$$\gamma(\mathbf{x},\widetilde{\mathbf{x}},t;u) = \text{Softmax}_2\!\left(\frac{1}{\sqrt{|\Omega_y|}}\langle Q(\mathbf{x},\cdot), K(\widetilde{\mathbf{x}},\cdot)\rangle_{\Omega_y}\right)$$

Attention output: $\langle \gamma(\mathbf{x},\cdot), V(\cdot,\mathbf{y})\rangle_{\Omega_x}$

Upon spatial discretization ($N_x$ tokens, $N_y$ features), the integrals become matrix multiplications, recovering the standard $\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}$ scaled dot-product attention.

### Layer Normalization as Projection to a Constraint Set

Define the constraint set:
$$S_1(\sigma_1, \sigma_2) = \left\{u : \frac{1}{|\Omega_y|}\int_{\Omega_y} u\,d\xi = \sigma_1,\ \frac{1}{|\Omega_y|}\int_{\Omega_y}(u-\sigma_1)^2 d\xi = \sigma_2^2\right\}$$

Layer normalization is the **projection** $u = \arg\min_{\bar{u}\in S_1} \|\bar{u} - v\|^2$, with closed-form solution:
$$u(\mathbf{x},\mathbf{y}) = \frac{v(\mathbf{x},\mathbf{y}) - \alpha(\mathbf{x})}{\sqrt{\beta(\mathbf{x})}}\,\sigma_2 + \sigma_1$$

where $\alpha$ and $\beta$ are the mean and variance along $\mathbf{y}$. This exactly recovers standard LayerNorm.

### ReLU as Projection to Non-Negativity Cone

$$S_2 = \{u : u \geq 0\}$$

The projection $u = \arg\min_{v \in S_2} \|v - \bar{u}\|^2$ solves pointwise to $u = \max\{\bar{u}, 0\} = \text{ReLU}(\bar{u})$.

### Feedforward Network as Linear + ReLU Substeps

Each feedforward layer corresponds to a linear operator substep (the integral inner product with $W_j$ and bias $b_j$) followed by a projection to $S_2$.

---

## Operator Splitting Scheme

Using the Lie splitting scheme with time step $\Delta t = 1$ (one layer per time step), the continuous equation is discretized into $M = 4 + J$ sequential substeps:

| Substep | Operation | Transformer Component |
|---|---|---|
| 1 | Attention integral + skip connection | Self-attention + residual |
| 2 | Project to $S_1(\sigma_1, \sigma_2)$ | Layer normalization (post-attention) |
| 3 to $2+J$ | Linear transform + project to $S_2$ | Feedforward layers with ReLU |
| $3+J$ | Averaging of skip connection | Residual merge |
| $4+J$ | Project to $S_1(\sigma_1, \sigma_2)$ | Layer normalization (post-FFN) |

For the standard Transformer with $J=2$: $M=6$ substeps per layer. Each full transformer layer = one time step from $t^{n-1}$ to $t^n$.

**Theorem:** After spatial discretization to a $N_x \times N_y$ grid, the Lie splitting scheme on the Continuous Transformer exactly recovers the Transformer architecture of Vaswani et al. (2017).

---

## Extensions

- **Multi-head attention:** Each head processes a subspace of $\Omega_y$; the full attention pools over all heads (parallel integral operators).
- **Vision Transformer (ViT):** The same framework applies; 2D patch indexing corresponds to a 2D discretization of $\Omega_x$.
- **Convolutional Transformer (CvT):** Replacing the QKV integral kernels $W^Q, W^K, W^V$ with convolutional kernels (local support) recovers CvT-style architectures.
- **UNet:** Previously derived by the same authors as a discretization of a different PDE (convection-diffusion type), unifying the two architectures.

---

## Key Conceptual Insights

1. **Transformer depth = time integration.** Each layer advances the continuous state by one time step. More layers = finer temporal resolution of the underlying continuous dynamics.
2. **Non-locality of attention.** The attention layer is intrinsically an **integral operator** over the token index domain $\Omega_x$ — capturing global dependencies at each "time step."
3. **Variational structure.** Layer norm and activation functions are projections, arising from subdifferentials of indicator functions of constraint sets — a variational (energy minimization) interpretation.
4. **Architecture design from PDEs.** This framework provides a principled pathway: choose a continuous evolution equation with domain knowledge, then derive the architecture via operator splitting.

---

## Relevance to Physics Foundation Models

1. **Principled design of physics-aware transformers:** The integro-differential equation framework allows incorporating physical operators directly into the attention kernel $W^Q, W^K, W^V$. For example, using divergence operators or Fourier kernels as the integral kernel would yield physically constrained attention.
2. **Connection to neural operators:** When the integral kernels are data-independent (fixed operator), the framework reduces to classical operator theory — connecting to FNO and DeepONet.
3. **Architecture modifications justified by theory:** Changing the splitting order or adding physics-informed terms to the continuous equation provides a rigorous basis for novel architectures.

**[AI Inference]:** The integro-differential equation framework suggests that **physics-informed transformers** could be derived by augmenting the continuous equation (I) with known physical operators (e.g., divergence-free projection, energy dissipation) as additional terms. This would embed physics not in the loss but directly in the attention mechanism — corresponding to level 5 of the physics encoding spectrum (hard architectural constraints).

**[AI Inference]:** The Transformer-as-PDE-solver interpretation connects to GP$_{\text{hy}}$T's "neural differentiator" paradigm: if each layer computes a time step of a continuous equation, and the continuous equation encodes physics, then a sufficiently deep transformer could implicitly simulate PDE dynamics without explicit numerical integration.

---

## Cross-Links

- [[transformer-architectures]] — extends with continuous mathematical framework
- [[neural-operators]] — connection: fixed integral kernels = classical operators
- [[partial-differential-equations]] — the integro-differential equation formulation
- [[pde-transformer-paper]] — another physically-motivated transformer architecture
- [[varmion-viscous-flows]] — variational formulation connection
- [[gphyt-physics-foundation-model]] — neural differentiator interpretation
