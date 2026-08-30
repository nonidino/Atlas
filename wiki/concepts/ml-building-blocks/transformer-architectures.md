# Transformer Architectures for Physics

**Type:** Core Concept  
**Related Sources:** GP$_{\text{hy}}$T, Walrus, PDE-Transformer, Lost in Latent Space, AION-1, Transformer Mathematical Framework, Kepler-Newton Inductive Biases, PhysicsFormer

---

## Foundational Transformer

The transformer (Vaswani et al., 2017) uses **self-attention** to model global dependencies in sequences:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$ are linear projections of the input $X$.

For a sequence of $N$ tokens, self-attention is $O(N^2)$ in time and space. Multi-head attention parallelizes this across $h$ attention heads with dimension $d_k = d_\text{model}/h$.

---

## Vision Transformer (ViT) Adaptation to Physics

Physics data (spatial fields) is handled by patching:
- Divide the 2D spatial domain into $p\times p$ patches.
- Linearly embed each patch into a token $\mathbf{z}_i \in \mathbb{R}^d$.
- Add positional encoding; apply transformer layers.

For physics, the **number of tokens** grows as $(H/p)^2 \cdot (W/p)^2$ — making global attention expensive for high-resolution fields.

---

## Key Architectural Innovations for Physics

### Spatiotemporal Tubelet Tokens (GP$_{\text{hy}}$T)
Input is a 4D tensor (time, height, width, channels). Patches extend across time (tubelets), enabling the model to encode temporal dynamics directly into token structure. Each token represents a spatiotemporal subdomain.

### Space-Time Factorized Attention (Walrus)
Instead of full 4D attention, alternate between:
- **Spatial attention:** within each timestep across spatial locations.
- **Temporal attention:** across timesteps at fixed spatial positions.

Reduces $O(N^2)$ to $O(N_s^2) + O(N_t^2)$ where $N_s$ and $N_t$ are spatial and temporal token counts.

### Shifted Window Attention (PDE-Transformer)
Restrict self-attention to local windows of size $w\times w$. Alternate between shifted and non-shifted windows between consecutive blocks to allow cross-window communication. Cost: $O(N \cdot w^2)$ rather than $O(N^2)$.

### Axial Attention (Walrus, PDE-Transformer SC)
Apply attention only along one axis at a time (rows, then columns, then channels). Effective for 2D and 3D structured grids with strong axis-aligned correlations.

### Compute-Adaptive Patching (Walrus)
Vary patch size (compression level) dynamically based on input resolution. Higher resolution data gets higher compression to maintain a fixed token budget. Uses Convolutional Stride Modulation (CSM).

---

## Diffusion Transformers (DiT)

Diffusion transformers adapt ViT to denoising diffusion models:
- Input is noised data $\mathbf{x}_t$ plus noise level conditioning $t$.
- **adaLN-Zero conditioning:** scale and shift token representations based on conditioning via:
$$\mathbf{z} \leftarrow \gamma(c) \cdot \text{LayerNorm}(\mathbf{z}) + \beta(c)$$
where $\gamma, \beta$ are learned from conditioning $c$.
- Enables conditioning on PDE type, physical channel type, Reynolds number, etc.

---

## Position Encoding for Physics

| Encoding | Used In | Properties |
|---|---|---|
| **Absolute (learned)** | Standard ViT | Fixed resolution; doesn't generalize |
| **RoPE (Rotary)** | Walrus | Resolution-invariant; relative positions |
| **T5 Relative** | Walrus (temporal) | Generalizes to different sequence lengths |
| **Log-spaced relative + FFN** | PDE-Transformer | Translation-invariant for PDEs |
| **None** | Shifted-window models | Relies on local window structure |

---

## Stabilization Techniques

Transformers for physics can suffer from training instabilities due to large dynamic range of physical fields:

- **QK Normalization** (Walrus, Lost in Latent Space): Normalize query and key tensors before dot product → prevents attention entropy blowup.
- **RMSNorm on Q,K** (PDE-Transformer): Equivalent to QK normalization.
- **Gradient clipping via EMA** (PDE-Transformer): Clips based on exponential moving average of gradient norms → prevents loss spikes.
- **Weight decay** (PDE-Transformer): Small AdamW weight decay ($10^{-15}$) for bf16 training.
- **Low learning rate** (PDE-Transformer): $4\times10^{-5}$ vs. typical $10^{-4}$.

---

## Channel Representation

**Mixed Channel (MC):** Multiple physical fields packed into the same token — efficient but less flexible.

**Separate Channel (SC, PDE-Transformer):** Each physical field (velocity, pressure, etc.) embedded independently as its own token sequence. Channels interact only via channel-axis axial attention. Better for:
- Transfer learning (new channels can be added without retraining spatial representation).
- Physics interpretability (each token represents one physical observable).

---

## Relevance to PFM

Transformers are the natural backbone for a PFM because:
1. They scale with data and compute (empirically demonstrated in LLMs).
2. Self-attention can capture long-range spatial correlations in physical fields.
3. In-context learning emerges at scale (GP$_{\text{hy}}$T uses this for physics inference).
4. Conditioning mechanisms (adaLN, cross-attention) can encode PDE parameters, equation types, physical units.
5. The architecture is modality-agnostic — the same backbone works for images, spectra, fields.

**[AI Inference]:** The remaining gap between physics transformers and LLMs is primarily **data at scale**. LLMs train on internet-scale text; physics models train on simulation data that requires expensive compute to generate. A PFM development strategy should prioritize: (1) identifying the cheapest simulation regimes that cover the most physics diversity, (2) data augmentation strategies (like Walrus's 2D→3D) that multiply effective data scale, (3) transfer across physical domains to reduce data requirements.

---

---

## Mathematical Foundation: Transformer as PDE Discretization

Tai et al. ([[transformer-mathematical-framework]]) prove rigorously that the standard Transformer (Vaswani 2017) is the discretization of a **continuous integro-differential equation** via the Lie operator-splitting scheme:

$$\begin{cases}
u_t = \underbrace{\langle \gamma(\mathbf{x},\cdot,t;u),\, V(\cdot,\mathbf{y},t;u)\rangle_{\Omega_x}}_{\text{attention}} + \underbrace{\partial I_{S_1(\sigma_1,\sigma_2)}(u)}_{\text{layer norm}} + \underbrace{\sum_j \langle W_j, u\rangle + b_j + \partial I_{S_2}(u)}_{\text{feedforward + ReLU}}, \\
u(\mathbf{x},\mathbf{y},0) = f(\mathbf{x},\mathbf{y}).
\end{cases}$$

Key identifications:
- **Attention** = non-local integral operator (kernel $W^Q, W^K, W^V$ are integral transform kernels)
- **Layer normalization** = projection onto the constraint set $S_1 = \{u : \text{mean}=\sigma_1, \text{var}=\sigma_2^2\}$
- **ReLU** = projection onto $S_2 = \{u \geq 0\}$
- **Each transformer layer** = one time step of the operator splitting scheme
- **Transformer depth** = integration time $T$ of the continuous equation

**Implication for PFM design:** Physical operators (divergence-free projection, Fourier kernels, spectral mixing) can be embedded directly into the integral kernel structure of attention, yielding architecturally-constrained physics-aware transformers derived from first principles.

---

## Context Length and World Model Type

From Liu et al. (ICML 2025, [[kepler-newton-inductive-biases]]):

Context length controls what kind of "world model" a transformer learns for physical dynamics:

| Context Length | Learned Model | Internal Representation | OOD Generalization |
|---|---|---|---|
| Short (= 2) | **Newtonian** (local, causal) | Gravitational forces | Good |
| Long (100+) | **Keplerian** (global, geometric) | Orbital parameters | Poor |

Three inductive biases are needed for a transformer to learn true world models:
1. **Spatial smoothness:** continuous coordinates or small vocabulary size
2. **Spatial stability:** noisy context learning ($\sigma \approx 0.1$)
3. **Temporal locality:** short context window (= order of the governing ODE)

**Design implication:** For prediction accuracy, use long context. For physical law discovery, restrict to short context. Multi-head attention with mixed window sizes could achieve both simultaneously.

---

## Dynamical Systems Perspective: Token Clustering

Geshkovski et al. ([[transformers-particle-systems-clustering]]) model Transformers as **mean-field interacting particle systems** on the unit sphere $\mathbb{S}^{d-1}$. Each token $x_i \in \mathbb{S}^{d-1}$ follows:

$$\dot{x}_i(t) = \mathbf{P}^\perp_{x_i(t)}\!\left(\frac{\sum_j e^{\beta\langle Qx_i, Kx_j\rangle} Vx_j}{\sum_k e^{\beta\langle Qx_i, Kx_k\rangle}}\right)$$

Layer normalization constrains tokens to the sphere; attention provides exponentially weighted mean-field coupling. The **interaction energy**

$$\mathsf{E}_\beta[\mu] = \frac{1}{2\beta}\iint e^{\beta\langle x,x'\rangle}\,d\mu(x)\,d\mu(x')$$

is maximized at a Dirac mass and increases monotonically under the dynamics — driving all tokens toward a single point. The Transformer is a **Wasserstein gradient flow** (with energy-weighted metric) of $\mathsf{E}_\beta$.

**Key clustering results:**
- $d \geq 3$ (practically always true): all tokens converge to consensus for any $\beta$, any $n$
- $d \geq n$: exponential convergence $\|x_i(t) - x^*\| \leq Ce^{-\lambda t}$
- Two-phase metastable dynamics: **fast** coalescing into a few clusters, then **slow** merging to one point

**Implication for physics:** The metastable clustered state (where trained models operate) supports a small number of distinct token representations — analogous to dominant coherent structures in fluid flow. Residual connections and multi-scale architectures counteract over-clustering (information loss).

---

## See Also

- [[physics-foundation-models]]
- [[in-context-learning-physics]]
- [[world-models-physics-ai]]
- [[diffusion-models-physics]]
- [[poseidon-pde-foundation-model]] — SwinV2 multiscale windowed/hierarchical attention as a physics backbone
- [[hierarchical-windowed-tokens]] — multiscale Swin token representation
- [[walrus-paper]]
- [[gphyt-physics-foundation-model]]
- [[pde-transformer-paper]]
- [[transformer-mathematical-framework]]
- [[transformers-particle-systems-clustering]]
- [[kepler-newton-inductive-biases]]
- [[physicsformer-pinn-ns]]
- [[aion-1-astronomy]]
