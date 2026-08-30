# Learning to Simulate Complex Physics with Graph Networks (GNS)

**Source:** Sanchez-Gonzalez et al., ICML 2020  
**File:** `new/sanchez-gonzalez20a.md`  
**Related Concepts:** [[equivariant-gnns]], [[neural-surrogates]], [[autoregressive-rollout-stability]], [[pfm-architecture-approaches]], [[possible-architectures]]  
**Related Summaries:** [[dynami-cal-graphnet]], [[navier-stokes-nonuniform-grids]]

---

## Overview

Graph Network-based Simulators (GNS) is the foundational framework for learning particle-based physical simulation via message-passing GNNs. The key idea: represent the physical state as a particle graph, learn dynamics via an encode-process-decode pipeline, and roll out autoregressively using an Euler integrator. A single GNS architecture trained on multi-material data (water, sand, goop) generalizes to 10–34× more particles and 8× longer trajectories than those seen during training.

---

## Architecture

The GNS implements the **encode-process-decode** scheme:

$$\hat{X}^{t_{k+1}} = s_\theta(\hat{X}^{t_k}) = \text{Update}\!\left(\hat{X}^{t_k},\ d_\theta(\hat{X}^{t_k})\right)$$

where the dynamics model $d_\theta: \mathcal{X} \to \mathcal{Y}$ predicts per-particle **accelerations** $\bar{\mathbf{p}}_i$, and an Euler integrator provides the update.

### Encoder
Maps particle state $X$ to a latent graph $G^0$:
$$\mathbf{v}_i = \varepsilon^v(\mathbf{x}_i), \quad \mathbf{e}_{ij} = \varepsilon^e(\mathbf{r}_{ij})$$
where $\mathbf{x}_i = [\mathbf{p}_i, \dot{\mathbf{p}}_{i}^{t-C+1:t}, \mathbf{f}_i]$ (position, $C=5$ prior velocities, material features) and $\mathbf{r}_{ij} = [\mathbf{p}_i - \mathbf{p}_j,\ \|\mathbf{p}_i - \mathbf{p}_j\|]$ (relative displacement + magnitude). Latent size 128, encoded by MLPs.

### Processor
$M$ rounds of learned message-passing over the latent graph:
$$G^{m+1} = \text{GN}^{m+1}(G^m), \quad G^M = \text{Processor}(G^0)$$
The authors use $M = 10$ unshared GNs with residual connections on node and edge attributes. Performance is strongly sensitive to $M$.

### Decoder
$$\mathbf{y}_i = \delta^v(\mathbf{v}_i^M) \approx \bar{\mathbf{p}}_i \quad \text{(predicted acceleration)}$$

### Training
Supervised $L_2$ loss on predicted accelerations:
$$L(\mathbf{x}_i^{t_k}, \mathbf{x}_i^{t_{k+1}}; \theta) = \|d_\theta(\mathbf{x}_i^{t_k}) - \bar{\mathbf{p}}_i^{t_k}\|^2$$

**Noise injection for rollout stability:** Training inputs are corrupted with random-walk noise $\mathcal{N}(0, \sigma_v = 3 \times 10^{-4})$, closing the gap between one-step training distribution and the accumulated-error inference distribution.

---

## Key Results

| Domain | Particles | Rollout Steps | Rollout MSE (×10⁻³) |
|---|---|---|---|
| WATER-3D (SPH) | 13k | 800 | 10.1 |
| SAND-3D | 20k | 350 | 0.554 |
| GOOP-3D | 14k | 300 | 0.618 |
| MULTIMATERIAL | 2k | 1000 | 16.9 |

**Generalization highlights:**
- Trained on 2.5k-particle WATERRAMPS; tested on 85k particles over 5,000 steps (34× more particles, 8× longer) — rollout remains plausible
- Single MULTIMATERIAL model learns water-water, sand-sand, and cross-material interactions simultaneously
- Continuous friction-angle interpolation: model trained on $[0°, 30°] \cup [55°, 90°]$ generalizes to held-out $[30°, 55°]$

**Key architectural finding:** Performance is primarily determined by (1) number of message-passing steps $M$ and (2) training noise scale $\sigma_v$. Other hyperparameters (MLP depth, linear encoders/decoders, global latent) have minimal effect.

---

## Relevance to PFM Goal

GNS establishes that a **single learned model** can simulate qualitatively distinct physical materials via message passing on particle graphs — without hand-engineering material-specific equations. This is the particle-based analog of the PFM autoregressive paradigm.

**Complementary to existing paradigms:**
- Unlike neural operators (FNO, DeepONet), GNS operates on **irregular particle configurations** — no fixed grid required
- Unlike continuum field transformers (Walrus, GP$_{\text{hy}}$T), GNS handles discrete interacting bodies naturally
- Noise injection is the GNS analog of Walrus's patch jittering — both mitigate distribution shift in AR rollout

**Critical gap:** GNS conserves no physical quantities by design. The model learns to approximately satisfy conservation laws from data alone. Dynami-CAL GraphNet ([[dynami-cal-graphnet]]) directly addresses this limitation.

---

## **[AI Inference]**

**[AI Inference]:** The GNS encode-process-decode architecture is structurally isomorphic to a transformer with: Encoder → embedding layer, Processor ($M$ GN steps) → $M$ transformer blocks, Decoder → output projection. The key difference is that GNS message passing is **local** (connectivity radius $R$), while transformer attention is **global**. This suggests a natural hybrid: a GNS-style particle encoder + a global attention layer in the processor for long-range pressure propagation (Stokes, elliptic PDEs), with local GN layers for short-range contact and viscous interactions.

**[AI Inference]:** The MULTIMATERIAL result — a single model trained on water, sand, goop, and their interactions — is the particle-based analog of Walrus's multi-physics pretraining across 19 scenarios. Both show that **multi-task pretraining on diverse physical regimes generalizes better than single-task training**. The key open question is whether particle-based (GNS) and field-based (Walrus) representations can be unified into a single architecture that handles both continuum PDEs and discrete many-body systems.

---

## See Also

- [[dynami-cal-graphnet]] — extends GNS with exact conservation of linear + angular momentum
- [[equivariant-gnns]] — broader context on equivariant and physics-constrained GNNs
- [[autoregressive-rollout-stability]] — error accumulation problem addressed by noise injection
- [[neural-surrogates]] — GNS as a learned surrogate simulator
- [[possible-architectures]] — GNN-PB proposal builds on GNS as empirical foundation
