# Neural Surrogate Models

**Type:** Core Concept  
**Related Sources:** All simulation papers

---

## Definition

A **neural surrogate model** (or ML emulator) is a neural network trained to approximate the input-output behavior of a computationally expensive simulation, enabling fast approximate prediction at a fraction of the cost.

$$\text{Surrogate}: \theta_\text{sim}, \text{IC}, \text{BC} \;\longrightarrow\; \text{solution fields} \quad (\text{fast, approximate})$$
$$\text{Solver}: \theta_\text{sim}, \text{IC}, \text{BC} \;\longrightarrow\; \text{solution fields} \quad (\text{slow, accurate})$$

---

## Taxonomy

| Category | Architecture | Physics Encoding | Resolution |
|---|---|---|---|
| **PINNs** | MLP / ResNet | PDE residual in loss | Mesh-free |
| **Neural Operators (FNO, DeepONet)** | Trunk + Branch / Spectral | Operator approximation | Discretization-invariant |
| **Transformer Surrogates** | ViT / Axial attention | Conditioning only | Grid-based |
| **Physics-Constrained** | DeepONet + divergence-free | Hard physical constraints | Grid-free |
| **Hybrid Methods** | DNN + iterative solver | Accelerates bottleneck | Grid-based |
| **Diffusion Emulators** | Score-based generative | Distribution modeling | Grid-based |

---

## The Accuracy vs. Speed Tradeoff

Numerical solvers provide near-perfect accuracy but are expensive:
- Fine-grained NS solver: hours to months per simulation.
- FEM solver: minutes to hours for engineering flows.

Neural surrogates are fast but sacrifice accuracy:
- Inference: milliseconds to seconds.
- Typical error: $10^{-2}$–$10^{-1}$ relative $L_2$ norm.

For practical engineering applications, errors below $10^{-3}$ are typically required — a 1-2 order of magnitude gap that current surrogates cannot close.

---

## Hybrid Approaches

Rather than replacing solvers entirely, hybrid methods accelerate specific bottlenecks:

**HyDEA (this wiki):** Uses DeepONet to provide efficient line-search directions for the Pressure Poisson Equation, complementing CG iteration:
- Neural network: eliminates global/low-frequency error (spectral bias advantage).
- CG iteration: eliminates local/high-frequency error.
- Combined: faster convergence than either alone.

**Key insight:** Neural networks and classical iterative methods have complementary strengths — neural networks are good at global patterns; iterative methods are good at local refinement.

---

## Generalization Challenge

Traditional surrogates suffer from the **generalization gap**: a model trained on flow around a specific geometry fails for a slightly different geometry. This is the primary barrier to practical deployment.

Solutions being explored:
1. **Neural operators:** Discretization-invariant; some geometry generalization.
2. **Physics constraints:** Hard-enforcement of conservation laws (PC-DeepONet) improves data efficiency.
3. **Foundation model pretraining:** Broad pretraining creates universal representations that generalize (Walrus, GP$_{\text{hy}}$T).
4. **Zero-shot PI fine-tuning:** D2NO pretraining + physics-informed loss enables new geometry adaptation without data (DeepONet multi-operator paper).

---

## From Surrogate to Foundation Model

The trajectory of the field:
1. **Narrow surrogates (2010s):** Single system, single geometry, single resolution.
2. **Neural operators (2019–2022):** Multi-resolution, some geometry generalization.
3. **Multi-task surrogates (2022–2024):** Multiple PDE types, but still require fine-tuning.
4. **Foundation models (2024–):** In-context inference, zero-shot transfer, broad pretraining.

A PFM is the logical endpoint: one model that subsumes all narrower surrogates.

---

## Data Generation

Training surrogates requires large amounts of high-quality simulation data:
- **The Well** (Ohana et al., 2025): benchmark dataset of diverse PDE simulations used by Walrus, GP$_{\text{hy}}$T, PDE-Transformer, Lost in Latent Space.
- **FlowBench** (Tali et al., 2024): additional flow scenarios used by Walrus.
- **PDEBench, PDEArena:** standard evaluation benchmarks.
- **Custom simulations:** GP$_{\text{hy}}$T generates 4 custom datasets to cover engineering scenarios (obstacles, two-phase flows, thermal flows) absent from The Well.

---

## See Also

- [[neural-operators]]
- [[physics-foundation-models]]
- [[autoregressive-rollout-stability]]
- [[navier-stokes-equations]]
- [[navier-stokes-nonuniform-grids]]
- [[gphyt-physics-foundation-model]]
- [[walrus-paper]]
