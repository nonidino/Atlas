# Architecture: Neural Differentiator + Numerical Integrator

**Type:** Architecture Specification  
**Status:** Best-supported (from existing research)  
**Date:** 2026-05-14  
**Related Concepts:** [[transformer-architectures]], [[autoregressive-rollout-stability]], [[partial-differential-equations]], [[physics-foundation-models]], [[possible-architectures]]  
**Related Summaries:** [[gphyt-physics-foundation-model]], [[walrus-paper]]

---

## Conceptual Overview

The Neural Differentiator is an architectural modification to the standard AR Transformer (Architecture 1) that targets its central weakness: error accumulation over long rollouts. The key change is in the **prediction target**: instead of predicting the next state $u^{t+\Delta t}$ directly, the model predicts the **temporal derivative** $\frac{\partial u}{\partial t}$, and a classical numerical integrator advances the state forward.

This distinction is physically motivated. The governing equation of any continuum physics system can be written as:

$$\frac{\partial u}{\partial t} = \mathcal{F}[u, \nabla u, \nabla^2 u, \ldots]$$

where $\mathcal{F}$ is the (possibly unknown) physics operator. Learning $\frac{\partial u}{\partial t}$ is **structurally aligned with learning $\mathcal{F}$** — the model is approximating the right-hand side of the governing PDE, not some function that happens to map states one step forward.

The 7× SOTA improvement of GP$_{\text{hy}}$T over the next-best baseline is attributed primarily to this design choice. It is the strongest single empirical signal in the wiki.

---

## Why Derivative Prediction Helps

**Mathematical argument:**  
Direct state prediction asks the model to learn:
$$\hat{u}^{t+\Delta t} = f_\theta(u^t, u^{t-1}, \ldots)$$

This is a map between full physical fields — potentially large amplitude, non-smooth, complex.

Derivative prediction asks the model to learn:
$$\hat{\partial}_t u^t = g_\theta(u^t, u^{t-1}, \ldots)$$

For smooth PDEs, $\partial_t u$ is significantly **smoother** than $u$ itself. A velocity field in turbulence can have $O(10^3)$ spatial variation; its time derivative has $O(10^2)$ variation (the acceleration is smoother than the velocity). Smoother targets are easier to learn accurately, and small errors in a smooth target produce smaller downstream rollout errors than small errors in a full-state target.

**Temporal consistency:**  
When the numerical integrator steps forward using $\hat{\partial}_t u^t$, it enforces the chain rule of calculus: consecutive predictions must be consistent with a single smooth trajectory. Pure AR prediction has no such constraint — successive predictions are independent forward-pass outputs and can be mutually inconsistent.

**Scale generalization:**  
Derivative features normalize out the temporal scale: $\frac{\partial u}{\partial t} \approx \frac{u^{t+\Delta t} - u^t}{\Delta t}$ has magnitude $O(\Delta t^{-1})$ smaller than the state itself. Training with variable $\Delta t$ and derivative targets forces the model to learn the **instantaneous physics** independently of the simulation step size — enabling genuine temporal scale generalization.

---

## Internal Architecture

This architecture shares its backbone with Architecture 1 (AR Transformer) and differs in the input augmentation and output head.

---

### Stage 1: Input Augmentation with Explicit Derivative Features

**Raw input:** $C$ snapshots of the physical field:

$$\mathbf{U}^t = [u^{t-C+1}, \ldots, u^t] \in \mathbb{R}^{C \times F \times H \times W}$$

**Explicit spatial derivative features:**  
For each snapshot $u^{t_i}$ and each field $f$, compute central-difference approximations of spatial derivatives:

$$\partial_x u_f^{t_i}(x, y) \approx \frac{u_f^{t_i}(x+1, y) - u_f^{t_i}(x-1, y)}{2\Delta x}$$

$$\partial_y u_f^{t_i}(x, y) \approx \frac{u_f^{t_i}(x, y+1) - u_f^{t_i}(x, y-1)}{2\Delta y}$$

$$\partial_{xx} u_f^{t_i}(x, y) \approx \frac{u_f^{t_i}(x+1, y) - 2u_f^{t_i}(x, y) + u_f^{t_i}(x-1, y)}{\Delta x^2}$$

$$\partial_{yy} u_f^{t_i}(x, y) \approx \frac{u_f^{t_i}(x, y+1) - 2u_f^{t_i}(x, y) + u_f^{t_i}(x, y-1)}{\Delta y^2}$$

**Temporal derivative feature:**  
$$\partial_t u_f^{t_i} \approx \frac{u_f^{t_i} - u_f^{t_i - \Delta t}}{{\Delta t}}$$

(First-order backward difference; can be improved with central difference using additional context frame.)

**Augmented input channels:**  
Concatenate all derivative features with the raw field values along the channel dimension:

$$\tilde{u}_f^{t_i} = \left[u_f^{t_i},\ \partial_x u_f^{t_i},\ \partial_y u_f^{t_i},\ \partial_{xx} u_f^{t_i},\ \partial_{yy} u_f^{t_i},\ \partial_t u_f^{t_i}\right] \in \mathbb{R}^{6 \times H \times W}$$

For $F$ fields, the augmented state has $6F$ channels per snapshot. GP$_{\text{hy}}$T uses the spatial and temporal first derivatives, improving sharp-gradient resolution by ~1 order of magnitude in long rollouts.

**[AI Inference]:** Including second-order spatial derivatives ($\partial_{xx}, \partial_{yy}$) would give the model direct access to the Laplacian $\nabla^2 u$ — the dominant term in diffusion-type PDEs (NS viscous term, heat equation, Poisson). This may be especially impactful for high-viscosity or low-Re flows where diffusion dominates.

---

### Stage 2: Tokenization

Identical to Architecture 1: spatiotemporal patch embedding with patch jittering.

**Modified patch shape:** Because of the $6F$ channel augmentation, the patch projection matrix has input dimension $6F \cdot p_h \cdot p_w$ (vs. $F \cdot p_h \cdot p_w$ for Architecture 1). This increases the input to the linear patch embedder but does not change the token sequence structure.

---

### Stage 3: Transformer Backbone

Identical to Architecture 1: space-time factorized transformer with axial RoPE, T5 temporal bias, QK normalization, alternating spatial and temporal attention blocks.

---

### Stage 4: Output Head — Derivative Prediction

**Modified output head:**  
Instead of predicting $\Delta u^t = u^{t+\Delta t} - u^t$ (residual prediction in Architecture 1), the model predicts the temporal derivative:

$$\hat{\partial}_t \mathbf{u}^t = \text{Decoder}(\mathbf{Z}^L) \in \mathbb{R}^{F \times H \times W}$$

The decoder is a linear unpatch projection (same as Architecture 1), but the target is now $\partial_t u$ rather than $u^{t+\Delta t}$.

**Training target normalization:**  
The derivative target $\frac{u^{t+\Delta t} - u^t}{\Delta t}$ must be independently normalized per dataset because its magnitude depends on $\Delta t$ and the physics. Per-dataset mean and variance are computed from the training trajectories and applied before the MSE loss.

---

### Stage 5: Numerical Integration

The predicted derivative is advanced using a classical numerical integrator.

**First-Order Forward Euler (baseline):**  
$$\hat{u}^{t+\Delta t} = u^t + \Delta t \cdot \hat{\partial}_t u^t$$

Euler is cheap (one derivative evaluation) but first-order accurate. Error is $O(\Delta t^2)$ per step.

**Fourth-Order Runge-Kutta (RK4, recommended for stiff PDEs):**  
$$k_1 = f_\theta(u^t)$$
$$k_2 = f_\theta\!\left(u^t + \frac{\Delta t}{2} k_1\right)$$
$$k_3 = f_\theta\!\left(u^t + \frac{\Delta t}{2} k_2\right)$$
$$k_4 = f_\theta\!\left(u^t + \Delta t \cdot k_3\right)$$
$$\hat{u}^{t+\Delta t} = u^t + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

RK4 is fourth-order accurate ($O(\Delta t^5)$ per step) but requires 4 neural network forward passes per integration step. For stiff PDEs (large highest-frequency eigenvalue / lowest-frequency eigenvalue ratio), adaptive-step integrators (RK45, Dormand-Prince) are preferred.

**Variable $\Delta t$ handling:**  
$\Delta t$ is passed as an explicit scalar input token concatenated with the context sequence. The numerical integrator uses this same $\Delta t$ value. This makes the full prediction pipeline naturally adaptive to variable-step-size inference, which is critical for stiff systems where the optimal $\Delta t$ varies by orders of magnitude over the trajectory.

---

### Full Prediction Pipeline

$$\hat{u}^{t+\Delta t} = \underbrace{\text{NumericalIntegrate}}_{\text{RK4 or Euler}}\!\left(\underbrace{f_\theta\!\left(\tilde{u}^{t-C+1}, \ldots, \tilde{u}^t,\ \Delta t\right)}_{\text{neural derivative prediction}},\ \Delta t\right)$$

where $\tilde{u}$ denotes the derivative-augmented input.

---

## Training Objective

Regression on temporal derivative targets:

$$\mathcal{L}(\theta) = \mathbb{E}_{(u^t, u^{t+\Delta t}), \Delta t}\!\left[\sum_{f=1}^F \left\|\hat{\partial}_t u_f^t - \frac{u_f^{t+\Delta t} - u_f^t}{\Delta t}\right\|_2^2\right]$$

The finite-difference approximation $\frac{u^{t+\Delta t} - u^t}{\Delta t}$ is the training target. For higher-order temporal accuracy, use the 4th-order central-difference approximation over 5 frames:

$$\partial_t u^t \approx \frac{-u^{t+2\Delta t} + 8u^{t+\Delta t} - 8u^{t-\Delta t} + u^{t-2\Delta t}}{12\Delta t}$$

This requires a context window of at least $C = 5$ frames but provides $O(\Delta t^4)$ accuracy in the target.

---

## Error Accumulation Analysis

**Why this architecture does better:**  
Let $\epsilon_t = \|\hat{u}^t - u^t\|$ be the state error at step $t$. For pure AR prediction, errors accumulate via a feedback loop — each step's error depends on the previous step's error, yielding roughly exponential growth.

For the Neural Differentiator, the numerical integrator acts as a **filter**: the Euler update $\hat{u}^{t+\Delta t} = \hat{u}^t + \Delta t \hat{\partial}_t \hat{u}^t$ only adds an increment proportional to $\Delta t$, bounding the per-step error contribution. If the derivative prediction error is $\|\hat{\partial}_t u - \partial_t u\| \leq \varepsilon_\partial$, the state error growth rate is bounded by:

$$\frac{d\epsilon}{dt} \leq L_\mathcal{F} \epsilon + \varepsilon_\partial$$

where $L_\mathcal{F}$ is the Lipschitz constant of the physics operator $\mathcal{F}$. This is the familiar ODE error bound — the neural derivative error $\varepsilon_\partial$ contributes additively rather than multiplicatively.

**Remaining limitation:** The model still has no mechanism to **detect** when it has drifted far from the training manifold. On novel physics or very long rollouts, the derivative prediction itself degrades, and the integrator faithfully propagates the error forward. Error detection via uncertainty estimation requires Architecture 2 (diffusion).

---

## Implementation Notes for Benchmark

**Burgers' adaptation:**  
For 1D Burgers' on 64 grid points, the exact Burgers' RHS is:

$$\partial_t u = -u \partial_x u + \nu \partial_{xx} u$$

The neural differentiator should learn to approximate this. The derivative features $\partial_x u$ and $\partial_{xx} u$ are therefore the most relevant inputs — the model only needs to learn the coefficients $(−u, \nu)$ from context, not the functional form itself.

Recommended augmented input for 1D case: $[u, \partial_x u, \partial_{xx} u, \partial_t u]$ — 4 channels, computed by finite differences on the 64-point grid.

**Architecture:**  
Same as Architecture 1 except:
- Input channels: $4 \times C$ (4 derivative-augmented channels per frame, $C$ frames)
- Output: $\partial_t u \in \mathbb{R}^{64}$ (not $u^{t+\Delta t}$)
- Post-processing: Euler integration: $\hat{u}^{t+\Delta t} = u^t + \Delta t \cdot \hat{\partial}_t u^t$

**Training note:** Normalize derivative targets by dataset statistics. For Burgers', $|\partial_t u| \sim O(\nu \cdot u_\text{max} / \Delta x^2)$ — can be $10-100\times$ larger than $|u|$ at high resolution. Without normalization, the loss will be dominated by high-gradient regions.

---

## Comparison with Architecture 1 (AR Transformer)

| Property | AR Transformer | Neural Differentiator |
|---|---|---|
| Prediction target | $u^{t+\Delta t}$ | $\partial_t u^t$ |
| Target smoothness | Moderate | Higher (smoother than state) |
| Temporal consistency | None enforced | Enforced by integrator |
| Error accumulation | Exponential (feedback) | Additive (ODE bound) |
| Variable $\Delta t$ | Yes (trained in) | Yes + integrator uses exact $\Delta t$ |
| Inference cost | 1 forward pass per step | 1–4 forward passes per step (Euler/RK4) |
| Stiff PDE handling | No | Yes (adaptive integrators available) |

---

## [AI Inference]

**[AI Inference]:** The derivative-feature approach could be extended to encode higher-order curvature information ($\partial^2 u / \partial x^2$, mixed partials $\partial^2 u / \partial x \partial y$) and even PDE-residual features computed from candidate equations. If the model is given $[\mathcal{R}_1(u), \mathcal{R}_2(u), \ldots]$ for several candidate PDEs as input channels, it could learn to identify which PDE is governing the current trajectory — enabling in-context PDE identification as an emergent capability.

**[AI Inference]:** Combining this architecture with the Mamba SSM backbone (Architecture 4) could be especially powerful: Mamba's recurrence naturally tracks the running state $u^t$ and could maintain an implicit derivative estimate $h_t \approx \partial_t u^t$ in its hidden state, rather than recomputing it from scratch at each step. This would make the Neural Differentiator + Mamba hybrid the most structurally aligned architecture with the true ODE/PDE structure.

---

## See Also

- [[possible-architectures]] — benchmark plan and comparison table
- [[autoregressive-rollout-stability]] — error accumulation problem this addresses
- [[partial-differential-equations]] — governing equation structure
- [[gphyt-physics-foundation-model]] — primary empirical evidence (7× SOTA)
- [[arch-autoregressive-transformer]] — Architecture 1: base architecture this modifies
- [[arch-physics-mamba]] — Architecture 4: alternative backbone with recurrence
