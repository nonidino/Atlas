# Physics Foundation Models (PFM)

**Type:** Core Concept  
**Related Sources:** GP$_{\text{hy}}$T, Walrus, PDE-Transformer, DeepONet, AION-1

---

## Definition

A **Physics Foundation Model (PFM)** is a single large model pretrained on diverse physical simulation data that can be adapted to a wide range of downstream physics tasks without retraining — the physics analog of LLMs' "train once, deploy anywhere" paradigm. The goal is to democratize access to high-fidelity simulations and accelerate scientific discovery by eliminating the need for task-specific solver development.

---

## Core Capability Requirements

A true PFM must demonstrate:

1. **Multi-physics learning** — simultaneously model fluid dynamics, heat transfer, plasma physics, elasticity, etc.
2. **In-context generalization** — infer governing dynamics from context (input snapshots) without being told the equations.
3. **Zero-shot transfer** — produce physically plausible predictions for entirely new physical systems never seen during training.
4. **Long-horizon stability** — maintain physical consistency during autoregressive rollouts over many timesteps.
5. **Variable-resolution handling** — operate on data with different spatial resolutions, dimensionalities, and domain shapes.

---

## Key Challenges

### 1. Data Heterogeneity
Physical systems have different numbers and types of fields (velocity, pressure, temperature, magnetic field…), defined over different geometrical domains with varying resolutions and boundary conditions. No canonical template exists.

### 2. Error Amplification
Learned models can amplify small per-step errors dramatically in autoregressive rollouts. This is exacerbated by:
- Tokenization/patching artifacts (Walrus: patch jittering addresses this).
- Distribution shift between training and inference (latent diffusion helps).

### 3. Multi-Scale Dynamics
Physical systems span enormously different spatial and temporal scales (microfluidics: micrometers/milliseconds; weather: kilometers/days). A PFM must either be told scales or infer them from context.

### 4. Data Scarcity
High-fidelity simulations are expensive to generate. A PFM must extract maximum value from available data, leveraging transfer across domains.

### 5. Equation Diversity
Small changes in equations yield categorically different physics (Euler → Navier-Stokes → MHD). A PFM trained on one equation family may fail to generalize across equation types.

---

## Current Approaches

| Model                 | Approach                                     | Scale            | Key Innovation                            |
| --------------------- | -------------------------------------------- | ---------------- | ----------------------------------------- |
| **GP$_{\text{hy}}$T** | Neural differentiator + numerical integrator | 385M params      | Derivative features + in-context learning |
| **Walrus**            | Autoregressive transformer                   | 1.3B params      | Patch jittering, 2D→3D augmentation       |
| **PDE-Transformer**   | Diffusion transformer with separate channels | ~100M params     | SC channel representation, flow matching  |
| **AION-1**            | Masked modeling across modalities            | 300M–3.1B params | Universal tokenization for 39 modalities  |

---

## The In-Context Learning Paradigm

The most promising path to a true PFM is **in-context learning**: the model receives a short "prompt" of prior physical states and infers the governing dynamics purely from that prompt, analogous to few-shot prompting of LLMs. This requires:

$$\text{next state} = f(\text{prompt: } [x_{t-n},\ldots,x_{t-1},x_t] \mid \text{no equation info})$$

The model must learn a universal dynamics inferencer — understanding that the same visual pattern of shock waves always implies compressible flow, regardless of the specific parameters.

---

## Relationship to This Project

The explicit goal of this wiki is to support the development of a physics foundation model. Key open questions:

1. **Architecture:** Should a PFM be autoregressive (next-step prediction like Walrus/GP$_{\text{hy}}$T) or masked/generative (like AION-1)?
2. **Scope:** Should it start with continuum dynamics (tractable) and expand, or target the broadest possible coverage from the start?
3. **Physics encoding:** Should physics knowledge be soft (loss penalties) or hard (architectural constraints like PC-DeepONet, VarMiON)?
4. **Scale:** Current models are 300M–1.3B params. LLMs at comparable capability had ~175B. What's the right scale for physics?

**[AI Inference]:** A hybrid architecture combining Walrus-style dynamics modeling with AION-1-style universal tokenization may be the most tractable near-term path to a general PFM. The separate-channel approach in PDE-Transformer and the modality-specific tokenization in AION-1 are convergent ideas from different communities that should be synthesized.

---

## See Also

- [[in-context-learning-physics]]
- [[neural-operators]]
- [[transformer-architectures]]
- [[autoregressive-rollout-stability]]
- [[diffusion-models-physics]]
- [[poseidon-pde-foundation-model]]
- [[gphyt-physics-foundation-model]]
- [[walrus-paper]]
- [[aion-1-astronomy]]
- [[00-token-representation-overview]]
