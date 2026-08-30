# PhysicsFormer: An Efficient and Fast Attention-Based Physics-Informed Neural Network for Solving Incompressible Navier–Stokes Equations

**Source:** arxiv:2601.03613  
**Authors:** Biswanath Barman, Debdeep Chatterjee, Rajendra K. Ray  
**Affiliation:** Indian Institute of Technology Mandi; Sikkim Manipal Institute of Technology  
**Related Concepts:** [[navier-stokes-equations]], [[transformer-architectures]], [[partial-differential-equations]]  
**Related Summaries:** [[navier-stokes-nonuniform-grids]], [[pde-transformer-paper]], [[gns-graph-network-simulators]]

---

## Overview

PhysicsFormer is a transformer-based physics-informed neural network (PINN) that addresses two critical failure modes of standard PINNs: (1) inability to capture high-frequency and temporally varying solutions, and (2) neglect of temporal dependencies. The architecture reframes pointwise PINN predictions as a **sequential data learning** problem using an encoder-decoder transformer with multi-head cross-attention. The result is ~3× faster than PINNsFormer with equivalent or better accuracy, and uses significantly less GPU memory.

---

## Motivation: PINN Failure Modes

Standard MLP-based PINNs have known failure modes:

1. **Spectral bias:** Neural networks prefer low-frequency solutions; PINNs trained on high-frequency or chaotic PDEs produce overly smooth, incorrect approximations.
2. **Temporal dependency neglect:** PINNs predict $u(\mathbf{x},t)$ pointwise, independently — no explicit temporal correlation structure. Initial conditions propagate poorly through the network.
3. **Convergence issues:** Sequential training schemes (Seq2Seq) and NTK-based methods have high computational cost or scaling difficulties.

---

## Architecture

PhysicsFormer consists of four components:

### 1. Data-Embedder: Pseudo-Sequence Generation

Converts a single spatiotemporal input $[\mathbf{x}, t] \in \mathbb{R}^d$ into a temporal pseudo-sequence of $k$ steps:

$$[\mathbf{x}, t] \xrightarrow{\text{Embedder}} \{[\mathbf{x},t],\, [\mathbf{x},t+\Delta t],\, \ldots,\, [\mathbf{x},t+(k-1)\Delta t]\} \in \mathbb{R}^{k\times d}$$

This explicit sequencing provides temporal context to the transformer without requiring separate time-series data collection.

### 2. High-Dimensional Space Generation

A fully-connected MLP linearly projects the pseudo-sequence into a higher-dimensional embedding space. This enriches sparse low-dimensional $(x, t)$ inputs with richer representations before the attention layers.

### 3. Encoder-Decoder with Multi-Head Cross-Attention

- **Encoder:** Multiple identical layers, each with self-attention + feedforward. Builds dependencies across all pseudo-sequence time steps.
- **Decoder:** Each layer has encoder-decoder **cross-attention** + feedforward. Uses the same spatiotemporal embeddings as the encoder (since the goal is to approximate the current state, not predict a future state).

The shared encoder-decoder embeddings distinguish PhysicsFormer from language transformers: there is no autoregressive generation, only parallel function approximation.

### 4. Output Layer

A final MLP maps from the decoder hidden state to the PDE solution $\hat{\boldsymbol{u}}(\mathbf{x},t)$.

---

## Novel Activation Function: $w\sin(t)$

Standard PINNs use $\tanh$ or ReLU. PhysicsFormer introduces:
$$\phi(t) = w\sin(t), \quad w \in \mathbb{R} \text{ (trainable)}$$

**Why:** Oscillatory activations capture high-frequency solutions more naturally. The trainable $w$ adapts the frequency scale to the problem (crucial for vortex shedding). Proved to be a universal approximator via the Universal Approximation Theorem (continuous + bounded + non-constant).

This activation is significantly cheaper than Wavelet activations while achieving equivalent accuracy.

---

## Physics-Informed Loss (Sequential Version)

The standard PINN loss is extended to the pseudo-sequence:

$$\mathcal{L}_{\text{residual}} = \frac{1}{k\mathcal{N}_{res}}\sum_{i=1}^{\mathcal{N}_{res}}\sum_{\gamma=0}^{k-1}\left|\mathcal{D}\left[\hat{\boldsymbol{u}}\left(\mathbf{x}_i, t_i + \gamma\Delta t\right)\right] - f(\mathbf{x}_i, t_i+\gamma\Delta t)\right|^2$$

Similarly for $\mathcal{L}_{bc}$ (all sequence steps at boundary) and $\mathcal{L}_{ic}$ (first sequence element only, since only $t=0$ satisfies initial condition). Total loss:

$$\mathcal{L}_{\text{PhysicsFormer}} = \lambda_{res}\mathcal{L}_{res} + \lambda_{bc}\mathcal{L}_{bc} + \lambda_{ic}\mathcal{L}_{ic} + \lambda_{data}\mathcal{L}_{data}$$

Dynamic loss weighting adaptively adjusts $\lambda$-values during training to speed convergence and avoid local minima.

---

## Benchmark Results

### Benchmark 1: 1D Burgers' Equation (Forward)

$$u_t + uu_x - \frac{0.01}{\pi}u_{xx} = 0, \quad x\in[-1,1],\ t\in[0,1]$$

| Model | MSE / $L_2$ Relative Error |
|---|---|
| Standard PINNs | $L_2 = 6.7 \times 10^{-4}$ |
| **PhysicsFormer** | **$L_2 = 2.4 \times 10^{-4}$**, MSE $\approx 10^{-6}$ |

Architecture: $d_\text{model}=32$, $d_\text{hidden}=512$, $N=1$ layer, 2 attention heads. GPU memory: ~500MB (fits any modern GPU). Training time: ~20min on Google Colab T4.

### Benchmark 2: 2D Incompressible Navier-Stokes (Forward + Inverse)

**Setup:** Cylinder wake at $\text{Re}=100$ (Kármán vortex street). Domain $[1,8]\times[-2,2]$, 1500 training velocity points (0.15% of total data), validation on full velocity + pressure field.

**Forward problem** (flow reconstruction — velocity, pressure, vorticity, streamlines):

| Model | Pressure Field |
|---|---|
| Standard PINNs | Qualitative errors |
| QRes, FLS | Moderate |
| **PhysicsFormer** | **Best qualitative and quantitative** |

MSE $\approx 10^{-6}$ for velocity.

**Inverse problem** (identify unknown $\lambda_1$ (convection), $\lambda_2$ (diffusion) in NS):

| Noise Level | $\lambda_1$ Error | $\lambda_2$ Error |
|---|---|---|
| Clean data | **0%** | **0%** |
| 1% Gaussian noise | **0.07%** | **0%** |

### Benchmark 3: Convection PDE (PINN Failure Mode)

$$u_t + \beta u_x = 0, \quad \beta = 50$$

Standard PINNs: $\approx 100\%$ relative error at $\beta=50$ (high-frequency failure).  
PhysicsFormer: $\approx 10^{-5}$ relative error — captures high-frequency solution accurately.

---

## Computational Efficiency

| Model | GPU Memory | Speed |
|---|---|---|
| PINNsFormer | High (requires high-memory GPU) | Baseline |
| **PhysicsFormer** | ~500MB (standard GPU) | **~3× faster** |

Efficiency gains come from:
1. Parallel (not sequential) learning via encoder-decoder
2. Lightweight $w\sin(t)$ activation vs. Wavelet
3. Fewer training data points needed (1500 vs. 2500 for NS)

---

## Relevance to Physics Foundation Models

1. **PINN + transformer synthesis:** PhysicsFormer demonstrates that combining sequential data representation with physics-informed losses resolves core PINN failure modes without sacrificing computational efficiency. This hybrid approach is directly applicable to the PFM architecture Arch 2 (Diffusion Backbone) and Arch 3 (Neural Differentiator).
2. **High-frequency solution capture:** The $w\sin(t)$ activation addresses the spectral bias problem — relevant for turbulent flows and any PFM targeting high-Reynolds-number physics.
3. **Inverse problem capability:** 0% parameter identification error is exceptional — suggests attention-based PINN hybrids could serve as physics parameter estimators in a PFM pipeline.
4. **Physics encoding level:** PhysicsFormer is at level 2 (soft loss constraints via PINN residuals) in the physics encoding spectrum. The sequential embedder adds level 1 (data-driven temporal context).

**[AI Inference]:** The data-embedder's pseudo-sequence strategy — converting a pointwise $(x,t)$ input into a temporal sequence — is a lightweight form of temporal context injection without requiring full autoregressive rollout. This could be combined with the neural differentiator architecture (Arch 3) to provide temporal context for derivative estimation, potentially improving stability without the full cost of long-context transformers.

**[AI Inference]:** The 0% inverse problem identification error suggests that cross-attention mechanisms are particularly good at integrating sparse data with physical constraints. For a PFM that needs to infer parameters from sparse observational data (e.g., experimental measurements), an encoder-decoder PINN structure like PhysicsFormer could serve as the "physics parameter decoder" module.

---

## Cross-Links

- [[navier-stokes-equations]] — primary application domain
- [[partial-differential-equations]] — general PDE formulation
- [[transformer-architectures]] — encoder-decoder architecture
- [[navier-stokes-nonuniform-grids]] — another NS DL solver (HyDEA)
- [[pde-transformer-paper]] — physics-specific transformer architecture
- [[varmion-viscous-flows]] — variational approach to viscous flow
