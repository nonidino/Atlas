# Architecture: Autoregressive Next-Step Transformer

**Type:** Architecture Specification  
**Status:** Best-supported (from existing research)  
**Date:** 2026-05-14  
**Related Concepts:** [[transformer-architectures]], [[autoregressive-rollout-stability]], [[in-context-learning-physics]], [[physics-foundation-models]], [[possible-architectures]]  
**Related Summaries:** [[walrus-paper]], [[gphyt-physics-foundation-model]], [[pde-transformer-paper]]

---

## Conceptual Overview

The Autoregressive (AR) Transformer is the direct physics analog of a large language model. Just as a language model predicts the next token from prior tokens, the AR physics transformer predicts the next physical state from a sequence of prior physical states. The "prompt" is a window of recent snapshots; the model outputs the next snapshot, which is fed back as input for the next step.

The key insight that makes this powerful: **the model never sees the governing PDE explicitly**. It infers dynamics from the trajectory alone — the same way GPT infers grammar from text. A trajectory of a turbulent flow "contains" the Navier-Stokes equations implicitly, just as a body of English prose "contains" grammatical rules implicitly.

This is the most battle-tested paradigm at scale (385M–1.3B parameters demonstrated), with the strongest empirical foundation of any approach in the benchmark suite.

---

## Internal Architecture

### Stage 1: Input Representation

**Multi-field physical state:**  
At each timestep $t$, the physical state is a collection of $F$ spatial fields:

$$\mathbf{u}^t = \{u_f^t : \Omega \to \mathbb{R}\}_{f=1}^F, \quad \Omega \subset \mathbb{R}^d$$

For a 2D velocity-pressure system: $F = 3$ (velocity components $u_x, u_y$, pressure $p$). For Walrus: $F$ up to 63.

**Context window:**  
The input is a temporal sequence of $C$ snapshots:

$$\mathbf{U}^t = [u^{t-C+1}, u^{t-C+2}, \ldots, u^t] \in \mathbb{R}^{C \times F \times H \times W}$$

Empirically, $C = 4$ is the sweet spot (GP$_{\text{hy}}$T: log-linear diminishing returns beyond 2 snapshots).

---

### Stage 2: Tokenization

**Spatiotemporal Tubelet Patching (GP$_{\text{hy}}$T):**  
The 4D input $(C, F, H, W)$ is divided into non-overlapping 4D tubes of size $(c_t, F, p_h, p_w)$, where $p_h, p_w$ are spatial patch sizes and $c_t$ is temporal patch depth. Each tube is linearly projected to a $d$-dimensional token:

$$\mathbf{z}_{i,j,k} = W_\text{proj} \cdot \text{flatten}(\mathbf{U}^t_{c_t \cdot i,\, :,\, p_h \cdot j,\, p_w \cdot k}) + \mathbf{b}$$

This produces $N = \lceil C/c_t \rceil \times \lceil H/p_h \rceil \times \lceil W/p_w \rceil$ tokens.

**Patch Jittering (Walrus):**  
Before patching, the spatial domain is randomly shifted by $(\delta_x, \delta_y) \sim \text{Uniform}(0, p_h) \times \text{Uniform}(0, p_w)$ and wrapped (periodic BC) or padded (other BCs). This breaks the aliasing artifacts from fixed patch boundaries:

$$\tilde{u}(x, y) = u\!\left(x - \delta_x,\ y - \delta_y\right)$$

The harmonic analysis justification: fixed patching introduces a spatial aliasing frequency $f_\text{alias} = 1/p$; jittering dithers this, spreading the artifact across frequencies and dramatically improving long-horizon rollout stability (89% of pretraining scenarios improved in Walrus).

**Adaptive-Resolution Tokenization (Walrus CSM):**  
For varying-resolution inputs, a Convolutional Stride Modulation (CSM) encoder/decoder adjusts the downsampling stride per input to maintain a fixed token count $N_\text{fixed}$ per axis regardless of resolution. This enables multi-resolution pretraining without padding artifacts.

---

### Stage 3: Positional Encoding

Two encoding strategies, both applied:

**Axial Rotary Position Encoding (RoPE) — spatial:**  
For 2D spatial coordinates $(x, y)$, RoPE applies axis-specific rotation matrices to query/key vectors:

$$\mathbf{q}'_{x,y} = R_x(\theta_x \cdot x) \cdot R_y(\theta_y \cdot y) \cdot \mathbf{q}_{x,y}$$

where $R_x(\phi) = \begin{pmatrix}\cos\phi & -\sin\phi \\ \sin\phi & \cos\phi\end{pmatrix}$ (in 2D; generalized to $d/2$ rotation planes in $d$ dimensions). Axial factorization separates $x$ and $y$ position encoding, making spatial attention computationally $O(N)$ instead of $O(N^2)$ in the spatial dimension.

**T5-Style Relative Position Bias — temporal:**  
The temporal attention uses learned relative position biases $b(i-j)$ added to attention logits before softmax. This is purely relative (captures "2 steps ago" not "timestep 14"), enabling generalization across different trajectory lengths and variable $\Delta t$.

---

### Stage 4: Transformer Backbone

**Space-Time Factorized Architecture (Walrus):**  
Rather than full 3D attention (cost $O((HW \cdot C)^2)$), attention is factorized into alternating spatial and temporal blocks:

$$\begin{aligned}
\mathbf{Z}'_t &= \text{SpatialAttn}(\mathbf{Z}_t) + \mathbf{Z}_t \quad \forall t \in \{1,\ldots,C\} \\
\mathbf{Z}''_{x,y} &= \text{TemporalAttn}(\mathbf{Z}'_{x,y}) + \mathbf{Z}'_{x,y} \quad \forall (x,y) \in \text{spatial positions}
\end{aligned}$$

Cost reduced to $O(N_\text{space}^2 \cdot C + N_\text{space} \cdot C^2)$, manageable for $C \ll N_\text{space}$.

**Block structure (per layer):**
1. Layer norm (pre-norm convention)
2. Multi-head self-attention (spatial OR temporal, alternating)
3. Residual connection
4. Layer norm
5. Feed-forward MLP (typically 4× expansion, GELU activation)
6. Residual connection

**QK Normalization (Walrus):**  
Applies layer normalization to $Q$ and $K$ before the attention dot product:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{\text{LN}(Q)\,\text{LN}(K)^\top}{\sqrt{d_k}}\right)V$$

This stabilizes training by preventing the attention logits from exploding as $\|Q\|, \|K\| \to \infty$ — a critical issue at scale with physical data (large magnitude variation across fields).

---

### Stage 5: Output Head and Prediction

**Residual prediction:**  
The model predicts the **increment** to the current state rather than the next state directly:

$$\hat{u}^{t+\Delta t} = u^t + \text{Decoder}(\mathbf{Z}^L)$$

This is easier to learn than the full state (the increment is small for small $\Delta t$) and inherits the current state as a strong prior.

**Unpatch projection:**  
The $L$-layer transformer output $\mathbf{Z}^L \in \mathbb{R}^{N \times d}$ is reshaped and projected back to field space via a learned linear layer:

$$\hat{u}^{t+\Delta t} \in \mathbb{R}^{F \times H \times W}$$

**Per-field normalization:**  
Each physical field $f$ is independently normalized to zero mean and unit variance during training. This prevents high-amplitude fields (e.g., pressure in high-Re flows) from dominating the loss.

---

### Stage 6: Autoregressive Rollout

At inference time, the model rolls out predictions iteratively:

$$\hat{u}^{t_k+1} = f_\theta\!\left(\hat{u}^{t_k-C+1}, \ldots, \hat{u}^{t_k}\right)$$

The context window slides forward: at each step, the oldest snapshot is dropped and the newest prediction is appended. For rollout length $T$ starting from $C$ seed frames, this requires $T$ forward passes.

**Variable $\Delta t$ generalization:**  
Training with randomly sub-sampled $\Delta t$ (e.g., uniform over $[\Delta t_\text{min}, \Delta t_\text{max}]$) forces the model to infer temporal scale from context rather than assuming a fixed step. $\Delta t$ is passed as an additional scalar input token concatenated with the sequence.

---

### Stage 7: Training Objective

Standard next-step supervised regression on physical fields:

$$\mathcal{L}(\theta) = \mathbb{E}_{(u^t, u^{t+\Delta t})}\!\left[\sum_{f=1}^F \left\|\hat{u}_f^{t+\Delta t} - u_f^{t+\Delta t}\right\|_2^2\right]$$

**Optional push-forward trick:**  
Train on multi-step rollouts with gradient propagated through all steps, directly penalizing long-rollout error:

$$\mathcal{L}_\text{PF}(\theta) = \sum_{k=1}^{K} \left\|f_\theta^{\circ k}(u^t) - u^{t+k\Delta t}\right\|_2^2$$

Push-forward training increases training cost but substantially reduces error accumulation at test time.

---

## Key Design Parameters

| Parameter | GP$_{\text{hy}}$T | Walrus | Recommended for Benchmark |
|---|---|---|---|
| Model size | 9M → 385M | 1.3B | 1M → 10M |
| Context length $C$ | 4 | 4–8 | 4 |
| Patch size $(p_h, p_w)$ | Not specified | Adaptive | 8×8 |
| Layers $L$ | Not specified | ~24 | 6–12 |
| Attention heads $h$ | Not specified | ~16 | 4–8 |
| Hidden dim $d$ | Not specified | ~1024 | 128–512 |
| Patch jittering | No | Yes | **Yes** |
| QK norm | No | Yes | **Yes** |
| Variable $\Delta t$ | Yes | Yes | **Yes** |

---

## In-Context Learning Mechanism

A critical emergent property at sufficient scale: the model infers different governing dynamics from the trajectory prompt alone, without retraining.

Mechanistically, this works because the transformer attention can identify that "these initial snapshots look like high-Re turbulence" vs. "these look like Stokes flow" and route to different effective computation pathways — analogous to how LLMs detect language style from context.

The GP$_{\text{hy}}$T evidence: a 385M-parameter model successfully infers new boundary conditions and remains below NMSE = 1 for entirely novel physics (supersonic flow, turbulent radiative layer) that was never in the training distribution.

**Enabling conditions** (from [[in-context-learning-physics]]):
1. Diverse multi-physics pretraining (at least 7 distinct physical regimes)
2. Large enough model (evidence suggests >100M parameters)
3. Context window contains enough snapshots to distinguish governing dynamics (4 snapshots minimum)

---

## Error Accumulation and Mitigations

The central weakness of this architecture. See [[autoregressive-rollout-stability]] for full treatment.

**Root cause:** The training distribution is over one-step predictions from ground-truth inputs; inference is over multi-step predictions from model outputs. Small per-step errors shift the input distribution away from the training distribution, causing subsequent errors to grow.

**Mitigations implemented in this architecture:**
1. **Patch jittering** — reduces aliasing artifacts that seed error growth
2. **QK normalization** — stabilizes training, reducing initialization sensitivity
3. **Residual prediction** — reduces the magnitude of errors (predicting small increment not full state)
4. **Variable $\Delta t$** training — improves temporal generalization
5. **Push-forward training** (optional) — directly penalizes multi-step error

**Mitigations NOT in this architecture (available in Architectures 2–3):**
- Derivative prediction (Architecture 3)
- Stochastic ensembling (Architecture 2)

---

## Implementation Notes for Benchmark

**Burgers' equation adaptation:**  
For 1D Burgers' on 64 grid points with $C = 4$ context frames:
- Input shape: $(C, 1, 64) = (4, 1, 64)$
- Patch along $x$: patch size 8 → 8 spatial tokens
- Temporal attention over $C = 4$ frames: 4 temporal tokens
- Total: 32 tokens per forward pass

**Recommended minimal architecture:**
```
Embedding: 64 → 256 (patch projection)
Layers: 6 (alternating spatial/temporal, 3 each)
Attention heads: 4
Hidden dim: 256
FFN dim: 1024
Total params: ~1.5M
```

**Training:**
- Optimizer: AdamW, $\beta_1=0.9$, $\beta_2=0.95$, weight decay $10^{-4}$
- LR schedule: cosine decay with warmup (1000 steps)
- Batch size: 64
- Training steps: 50K

---

## [AI Inference]

**[AI Inference]:** The log-linear diminishing returns beyond 2–4 context frames (GP$_{\text{hy}}$T) suggests the transformer is not effectively using longer context for physics — possibly due to the factorized attention losing cross-temporal correlations. A hybrid architecture with full 3D attention for the temporal dimension (keeping spatial factorized) might extract more signal from longer context, at moderate extra cost.

**[AI Inference]:** Walrus's 2D-into-3D augmentation (embedding 2D data as a plane in 3D space) could be generalized to a dimension-agnostic tokenization scheme: represent 1D fields (spectra, 1D PDEs) as degenerate 3D volumes. This would enable joint pretraining across 1D/2D/3D data without architecture changes — directly applicable to the Burgers' benchmark data (1D) and eventual 3D physics pretraining.

---

## See Also

- [[possible-architectures]] — benchmark plan and comparison table
- [[autoregressive-rollout-stability]] — error accumulation problem in depth
- [[in-context-learning-physics]] — emergent capability at scale
- [[transformer-architectures]] — architectural building blocks
- [[walrus-paper]] — 1.3B parameter implementation
- [[gphyt-physics-foundation-model]] — 385M parameter implementation with derivative features
- [[arch-neural-differentiator]] — Architecture 3: derivative prediction variant
- [[arch-diffusion-backbone]] — Architecture 2: generative alternative
