# Architecture: Physics-Mamba (Selective State Space Model)

**Type:** Architecture Specification  
**Status:** New Proposal  
**Date:** 2026-05-14  
**Related Concepts:** [[autoregressive-rollout-stability]], [[transformer-architectures]], [[partial-differential-equations]], [[physics-foundation-models]], [[possible-architectures]]  
**Related Summaries:** [[gphyt-physics-foundation-model]], [[walrus-paper]]

---

## Conceptual Overview

The AR Transformer (Architecture 1) is powerful but computationally expensive for long-rollout inference: self-attention is $O(N^2)$ in the sequence length, so doubling the rollout length quadruples the cost. For long-horizon physics simulation (thousands of timesteps), this is prohibitive.

Physics-Mamba replaces the transformer backbone with a **Selective State Space Model (SSM)** — specifically the Mamba architecture — which processes sequences in $O(N)$ time and $O(1)$ memory via recurrence. The key physics adaptation: $\Delta t$ (the integration step size) is passed as an explicit input token, making the model's effective time resolution naturally adaptive. The recurrence directly models the ODE structure of the underlying physics: the hidden state tracks the evolving physical system, and the transition matrices adapt to the current dynamics.

**Benchmark hypothesis:** Physics-Mamba should match AR Transformer quality on Burgers' while being 3–5× faster at inference for rollout lengths > 200 steps.

---

## Background: The Mamba SSM

### Continuous-Time Linear State Space

The theoretical foundation is the continuous-time linear SSM:

$$\frac{dh(t)}{dt} = A\,h(t) + B\,u(t)$$
$$y(t) = C\,h(t) + D\,u(t)$$

where:
- $h(t) \in \mathbb{R}^N$ is the hidden state (the "memory" of the system)
- $u(t) \in \mathbb{R}$ is the input
- $y(t) \in \mathbb{R}$ is the output
- $A \in \mathbb{R}^{N \times N}$ is the state transition matrix (diagonal in practice)
- $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$, $D \in \mathbb{R}$ are input/output projection matrices

### Discrete-Time Recurrence (ZOH Discretization)

For discretization with step size $\Delta t$ (zero-order hold):

$$\bar{A} = e^{A\Delta t}, \quad \bar{B} = (e^{A\Delta t} - I) A^{-1} B \approx \Delta t \cdot B \text{ (for small } \Delta t \text{)}$$

Recurrence:
$$h_{t+1} = \bar{A}\, h_t + \bar{B}\, u_t, \quad y_t = C\, h_t + D\, u_t$$

This is $O(N)$ per step, not $O(N^2)$ like attention.

### Selectivity (The Mamba Innovation)

Classic SSMs use fixed $A, B, C$ — equivalent to a linear time-invariant (LTI) filter. Mamba makes these **input-dependent** ("selective"):

$$B_t = s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = \text{softplus}(s_\Delta(x_t))$$

where $s_B, s_C, s_\Delta$ are learned linear projections of the current input $x_t$. The state transition $\bar{A}_t = e^{A \Delta_t}$ is also effectively input-dependent through $\Delta_t$.

This selectivity allows the model to:
- **Filter** irrelevant information (small $\Delta_t \to \bar{A} \approx I$, no integration)
- **Integrate** important signals (large $\Delta_t \to \bar{A} \approx 0$, full update to new state)

In language: the model learns to "pay attention" to relevant tokens by controlling how quickly the hidden state forgets.

---

## Physics-Mamba Internal Architecture

The physics adaptation adds four components on top of vanilla Mamba:
1. **Explicit $\Delta t$ input encoding** — makes temporal adaptivity explicit
2. **2D/Axial SSM scan patterns** — handles spatial coupling in 2D/3D fields
3. **Derivative prediction head** — combined with Architecture 3's output strategy
4. **Optional global attention layer** — for elliptic PDEs requiring global coupling

---

### Stage 1: Input Preparation

**Physical input:** Same as Architecture 1 — a sequence of $C$ snapshots, each with $F$ physical fields over a spatial grid.

For 2D: $\mathbf{U}^t \in \mathbb{R}^{C \times F \times H \times W}$.  
For 1D Burgers': $\mathbf{U}^t \in \mathbb{R}^{C \times 1 \times N_x}$.

**Derivative augmentation (optional, recommended):**  
Augment with spatial and temporal derivatives as in Architecture 3:
$$\tilde{u}_f^{t_i} = [u_f^{t_i},\ \partial_x u_f^{t_i},\ \partial_{xx} u_f^{t_i},\ \partial_t u_f^{t_i}] \in \mathbb{R}^{4}$$

per grid point. This is particularly natural for Mamba because the derivative features can be interpreted as the "velocity" and "acceleration" of the hidden state — directly encoding the ODE structure.

---

### Stage 2: Spatial Tokenization and Flattening

**Patch embedding:**  
Divide the spatial domain into $p_h \times p_w$ patches and embed each patch to a $d$-dimensional token (same as Architecture 1). This gives $N_\text{space} = (H/p_h)(W/p_w)$ spatial tokens per timestep.

**Flattening for SSM scan:**  
SSMs process 1D sequences. For 2D spatial fields, the spatial tokens must be serialized into a 1D sequence before the SSM. The scan order determines what spatial relationships the SSM captures easily:

| Scan Pattern | Captures Well | Misses |
|---|---|---|
| Row-major (raster) | Horizontal correlations | Diagonal, vertical |
| Column-major | Vertical correlations | Horizontal, diagonal |
| Hilbert curve | Local 2D neighbors | Long-range global |
| Z-order (Morton) | Hierarchical spatial | Diagonal at fine scale |

**Axial SSM (recommended):**  
Rather than scanning all spatial dimensions together, apply separate 1D Mamba scans along each spatial axis and add the results:

$$h^\text{out}_{x,y} = \text{Mamba}_x([h_{1,y}, h_{2,y}, \ldots, h_{H,y}])_x + \text{Mamba}_y([h_{x,1}, h_{x,2}, \ldots, h_{x,W}])_y$$

This is $O(H + W)$ per feature vs. $O(HW)$ for a full 2D scan, and captures correlations in both spatial directions independently.

**Multi-directional scanning (bidirectional Mamba):**  
Run four scans (forward/backward in each spatial dimension) and concatenate:

$$h^\text{out}_{x,y} = \text{concat}[\text{Mamba}_{x+}, \text{Mamba}_{x-}, \text{Mamba}_{y+}, \text{Mamba}_{y-}]$$

This is the standard approach for image Mamba models and should be the baseline.

---

### Stage 3: Temporal SSM Processing (Physics-Adapted)

**Core physics innovation — $\Delta t$ as explicit input token:**  

Rather than treating time as implicit (fixed steps in training), $\Delta t$ is passed as an explicit input that modulates the SSM discretization:

$$\Delta_t^{\text{physics}} = \text{softplus}(s_\Delta(x_t) + \log(\Delta t_{\text{input}}))$$

where $\Delta t_\text{input}$ is the physical time step provided as input. This is a **log-linear** combination: the learned $s_\Delta(x_t)$ provides data-dependent adaptation, and $\log(\Delta t_\text{input})$ shifts the overall timescale.

**Physical interpretation:**  
The SSM's $\Delta t$ parameter controls how quickly the hidden state updates. A large physical $\Delta t$ corresponds to "more physics happening" and requires a larger effective step — $e^{A \Delta_t^\text{physics}}$ is farther from identity, giving the new input more weight. A small $\Delta t$ keeps the state nearly unchanged, correctly modeling a nearly static system.

This is the Mamba analog of GP$_{\text{hy}}$T's variable-$\Delta t$ training: both force the model to represent physics at multiple temporal scales, but Physics-Mamba does this **through the SSM's own structural parameter** rather than through the input encoding alone.

**Full Physics-Mamba block:**

Per Mamba block, applied to each spatial position's temporal sequence:

```
Input: x ∈ R^{C × d}  (C timesteps, d-dim features at one spatial position)

1. Norm: x = LayerNorm(x)
2. Branch 1: x₁ = Linear_in(x) ∈ R^{C × d_expand}   [expansion factor 2]
3. Branch 2: x₂ = Linear_in2(x) ∈ R^{C × d_expand}
4. Conv1D(x₁)  [short local context, kernel size 4]
5. SiLU activation on x₁
6. SSM parameters:
   B_t = Linear_B(x₁_t)           [input-dependent input projection]
   C_t = Linear_C(x₁_t)           [input-dependent output projection]
   Δ_t = softplus(Linear_Δ(x₁_t) + log(Δt_input))  [physics-adapted step]
   Ā_t = exp(A · Δ_t)              [A is diagonal, learned, initialized from HiPPO]
7. Recurrence:
   h_{t+1} = Ā_t · h_t + Δ_t · B_t · x₁_t
   y_t = C_t · h_t
8. Gate: out = y * SiLU(x₂)
9. Linear_out(out) + residual
```

The `Linear_B`, `Linear_C`, `Linear_Δ` operations make the model selective (Mamba's key innovation over vanilla SSMs).

**A matrix initialization (HiPPO):**  
The diagonal $A$ matrix is initialized using the HiPPO (High-order Polynomial Projection Operator) construction, which gives the SSM optimal long-range memory by initializing it to project the input history onto Legendre polynomials. This significantly improves training convergence compared to random initialization.

---

### Stage 4: Spatial–Temporal Interleaving

Similar to Architecture 1's factorized attention, alternate between:
1. **Temporal Mamba blocks** — process the sequence of timesteps at each spatial position
2. **Spatial Mamba blocks (axial)** — process the sequence of spatial positions at each timestep

A full Physics-Mamba layer:
```
x ← TemporalMambaBlock(x)   # processes time dimension
x ← SpatialMambaBlock_x(x)  # processes x-dimension (axial)
x ← SpatialMambaBlock_y(x)  # processes y-dimension (axial)
x ← FFN(LayerNorm(x)) + x   # standard feed-forward
```

This factorization keeps the total complexity $O(C \cdot N_\text{space})$ — linear in both sequence dimensions, vs. quadratic for full attention.

---

### Stage 5: Global Attention Layer (Optional — for Elliptic PDEs)

**The receptive field problem:**  
SSMs scanning in one direction have a limited effective receptive field for 2D domains. For elliptic PDEs (Poisson, Stokes), a perturbation at point $A$ affects the entire domain simultaneously — the solution is globally coupled. A purely local SSM scan will miss this.

**Mitigation: one global attention layer per $k$ Mamba layers:**  
Insert a standard full self-attention layer every $k = 4$ Mamba layers:

$$\mathbf{Z} \leftarrow \text{MultiHeadAttention}(\mathbf{Z}) + \mathbf{Z}$$

The attention layer has full global receptive field ($O(N_\text{space}^2)$ but applied only 1/4 as often, so total cost is $O(N_\text{space}^2 / k)$. For $k = 4$ and moderate $N_\text{space}$ (64–256 tokens), this is manageable.

For purely parabolic/hyperbolic PDEs (heat equation, Burgers', wave equation) — where information propagates at finite speed — the global attention layer can be omitted entirely.

---

### Stage 6: Output Head

**Derivative prediction (recommended, from Architecture 3):**  
The output head predicts $\partial_t u^t$, and Euler/RK4 advances the state:

$$\hat{u}^{t+\Delta t} = u^t + \Delta t \cdot f_\theta^\text{Mamba}(\tilde{u}^{t-C+1}, \ldots, \tilde{u}^t,\ \Delta t)$$

This combination — Mamba backbone + derivative prediction head — may be the strongest version of this architecture because:
- Mamba's recurrence naturally tracks the integrated state
- The derivative head aligns with the SSM's own internal dynamics
- $\Delta t$ modulates both the SSM step and the integrator

**Direct state prediction:**  
Alternatively, predict $u^{t+\Delta t}$ directly (Architecture 1 style). This avoids the integrator coupling but loses the temporal consistency benefits.

---

### Temporal Recurrence at Inference (Key Efficiency Gain)

At inference time, Mamba is purely recurrent: the hidden state $h_T$ summarizes the entire history up to step $T$ in $O(N_\text{hidden})$ memory. Each new step costs $O(N_\text{hidden})$ — constant, independent of trajectory length.

For rollout of length $T$:
- **Transformer (Architecture 1):** $O(T \cdot N_\text{space}^2)$ total cost (quadratic in $T$)
- **Physics-Mamba:** $O(T \cdot N_\text{space} \cdot N_\text{hidden})$ total cost (linear in $T$)

For $N_\text{space} = 256$ spatial tokens and $T = 500$ rollout steps: Physics-Mamba is ~256× less total FLOPs for the temporal processing. In practice, additional constant-factor costs (convolutions, FFN) reduce this speedup to the claimed 3–5×.

---

## Training

**Training mode (parallel scan):**  
At training time, SSMs can be computed in parallel using the parallel prefix scan algorithm — not sequentially. This gives the same wall-clock training time as a transformer while enabling efficient inference recurrence.

**Loss:**  
Same as Architecture 1 (next-state MSE) or Architecture 3 (derivative MSE). Derivative loss is recommended for the same reasons.

**Noise injection for rollout stability:**  
Add small Gaussian noise to intermediate states during training (GNS technique, also applicable here):

$$\tilde{u}_t = u_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I)$$

This closes the gap between the training distribution (ground-truth inputs) and the inference distribution (predicted inputs), exactly as in Architecture 1's push-forward training.

---

## Key Design Parameters

| Parameter | Recommended Value |
|---|---|
| SSM hidden dim $N$ | 64–128 |
| Model dim $d$ | 256 |
| Expansion factor | 2 |
| Conv1D kernel size | 4 |
| Number of layers | 12–24 |
| Spatial scan | Bidirectional axial (4 directions) |
| Global attention | Every 4 layers (elliptic PDEs only) |
| $\Delta t$ input | Log-linear shift on $\Delta_t$ |
| Output head | Derivative prediction + Euler |
| A initialization | HiPPO diagonal |

---

## Benchmark Adaptation (1D Burgers')

For 1D Burgers' on 64 grid points: no 2D spatial scanning needed. The model processes:
- **Temporal dimension:** SSM over $C = 4$ timesteps at each spatial position
- **Spatial dimension:** SSM over 64 spatial positions at each timestep

```
Architecture:
  Spatial patch size: 1 (no patching for 64 points; one token per grid point)
  Temporal context C: 4
  Mamba layers: 6 (alternating temporal/spatial)
  SSM hidden dim: 32
  Model dim: 64
  Total params: ~500K
```

**Expected advantage:** At rollout lengths > 200 steps, Physics-Mamba should outperform the AR Transformer in inference speed while matching its accuracy. For rollout lengths < 50 steps, the transformer may be faster due to lower constant-factor costs.

---

## [AI Inference]

**[AI Inference]:** The $\Delta t$-as-input design in Physics-Mamba is precisely analogous to GP$_{\text{hy}}$T's derivative prediction — both are attempts to make the temporal structure of the PDE explicit in the architecture rather than leaving it entirely to learned weights. The strongest possible combination may be: **Physics-Mamba backbone + derivative prediction head + $\Delta t$ explicit input**, where all three mechanisms reinforce the same temporal physics structure.

**[AI Inference]:** The HiPPO initialization for the $A$ matrix was derived to optimally remember a history by polynomial projection. For physics, a better initialization might be derived from the Green's function of the target PDE — initializing $A$ to represent the solution operator of the linearized PDE. This "physics-informed HiPPO" initialization could dramatically accelerate convergence for specific PDE families.

**[AI Inference]:** The limited spatial receptive field of SSMs may be less of a problem for Burgers' than feared: the Burgers' equation is hyperbolic (information propagates at the advection speed), so the relevant spatial coupling at each timestep is local to the characteristic cone. An SSM with receptive field $R_\text{SSM} > u_\text{max} \cdot \Delta t / \Delta x$ should be sufficient — suggesting that for the benchmark, the SSM receptive field is adequate without any global attention.

---

## See Also

- [[possible-architectures]] — benchmark plan; this is Architecture 4
- [[autoregressive-rollout-stability]] — efficiency problem this addresses
- [[transformer-architectures]] — the attention mechanism this replaces
- [[partial-differential-equations]] — PDE structure driving the $\Delta t$ design
- [[arch-autoregressive-transformer]] — Architecture 1: the transformer baseline to compare against
- [[arch-neural-differentiator]] — Architecture 3: derivative prediction head (compatible with this backbone)
- [[arch-gnn-physics-bottleneck]] — Architecture 5: alternative for irregular geometry
