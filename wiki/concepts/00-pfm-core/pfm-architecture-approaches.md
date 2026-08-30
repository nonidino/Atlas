# Physics Foundation Model: Architecture Approaches

**Type:** Core Concept  
**Related Concepts:** [[physics-foundation-models]], [[autoregressive-rollout-stability]], [[diffusion-models-physics]], [[neural-operators]], [[transformer-architectures]], [[in-context-learning-physics]], [[transfer-learning-fine-tuning]]

---

## Overview

This page catalogs the major architectural paradigms currently tested as candidates for a Physics Foundation Model (PFM), with pros, cons, and current scale. No single approach is sufficient — the open question is how to synthesize them.

---

## 1. Autoregressive Next-Step Prediction

**Examples:** [[walrus-paper]] (1.3B params), [[gphyt-physics-foundation-model]] (385M params)

The model takes a sequence of prior physical states as a prompt and predicts the next state, then rolls out autoregressively. The governing physics is inferred entirely from the trajectory — no equation specification required.

### Advantages
- Directly mirrors LLM pretraining — the most battle-tested paradigm at scale
- Natural in-context learning: model infers governing dynamics from the trajectory prompt
- Walrus demonstrated generalization across 19 physical scenarios and 63 fields simultaneously
- Straightforward to scale with more data

### Disadvantages
- **Error accumulation:** small per-step errors compound over long rollouts
- Distribution shift between training and inference degrades stability
- Patch jittering (Walrus) and derivative prediction (GP$_{\text{hy}}$T) partially mitigate but do not solve the problem
- No principled uncertainty quantification — single deterministic trajectory

**Current scale:** 385M–1.3B params

---

## 2. Neural Differentiator + Numerical Integrator

**Example:** [[gphyt-physics-foundation-model]]

Instead of predicting the next state directly, the model predicts **derivatives** of physical fields; a classical numerical solver integrates these forward.

$$\hat{u}_{t+\Delta t} = \text{NumericalIntegrate}\!\left(f_\theta\!\left(\frac{\partial u}{\partial t}, \frac{\partial^2 u}{\partial t^2}, \ldots\right)\right)$$

### Advantages
- Derivative targets are smoother and more stable than full-state targets
- Inherits conservation properties from the numerical integrator
- 7× SOTA — derivative features are a powerful inductive bias
- Errors do not accumulate in the same pathological way as pure autoregressive prediction

### Disadvantages
- Tightly coupled to a specific numerical integration scheme — reduces flexibility
- Still requires knowing the appropriate integrator family (Euler, RK4, etc.)
- Harder to apply to discrete or irregular domains where differentiation is ill-defined
- Not fully data-driven — the classical solver is a hardcoded architectural component

**Current scale:** 385M params

---

## 3. Diffusion / Generative Modeling

**Examples:** [[pde-transformer-paper]], [[pisd-physics-informed-spectral-diffusion]], [[latent-diffusion-physics]]

A diffusion model learns to sample from the distribution over physically plausible states conditioned on observations or prior states, rather than predicting a single deterministic outcome.

### Advantages
- Naturally handles intrinsic uncertainty in chaotic/turbulent systems
- Consistently outperforms deterministic surrogates for chaotic PDEs (demonstrated in "Lost in Latent Space")
- PISD's DPS guidance enforces PDE constraints at inference time — the same trained model solves different PDEs without retraining
- Latent diffusion is robust to 1,000× spatial compression without accuracy loss
- Stochastic output enables ensemble-style uncertainty estimation

### Disadvantages
- Inference is expensive — requires many network evaluations per sample (though PISD claims 3–15× speedup over prior spectral-space methods)
- Not designed for long-horizon autoregressive rollouts — fundamentally a snapshot model
- Stochastic output is a disadvantage for deterministic engineering applications requiring a single best prediction
- Training objective (denoising) is not directly aligned with multi-step dynamics

**Current scale:** ~100M–1B params

---

## 4. Masked Multimodal Modeling

**Example:** [[aion-1-astronomy]] (300M–3.1B params, 39 modalities)

Following BERT/MAE: randomly mask portions of multi-modal input (images, spectra, time series, metadata) and train to reconstruct them. Applied to astronomy but structurally analogous to multi-physics.

### Advantages
- Universal tokenization handles extreme heterogeneity — different field types, resolutions, modalities in one model
- 39 modalities in AION-1 is the broadest coverage of any current approach
- Bidirectional attention captures long-range dependencies better than causal autoregressive models
- Natural for inverse problems: reconstruct missing modalities from available ones

### Disadvantages
- Not inherently causal — does not naturally produce time-forward predictions
- Training objective (reconstruction) is less aligned with the PFM goal of predicting dynamics
- No physics constraint in the loss — purely data-driven; may violate conservation laws
- Evaluated on astronomy, not continuum mechanics — generalization to PDE dynamics unproven

**Current scale:** 300M–3.1B params

---

## 5. Neural Operator Learning

**Examples:** [[poseidon-pde-foundation-model]] (operator learning *at foundation-model scale*), [[deeponet-multi-operator]], [[varmion-viscous-flows]], [[pc-deeponet-cfd]]

Learn a mapping between function spaces — e.g., from initial conditions to solution trajectories — rather than discretization-specific predictions. The learned operator is resolution-invariant by construction. **[[poseidon-pde-foundation-model]] is the landmark result here:** it scaled operator learning to 629M params via a SwinV2 multiscale-transformer (scOT) with continuous lead-time conditioning and **all2all (semi-group) training**, and demonstrated generalization from a 6-operator fluid pretraining set to 15 OOD downstream PDEs (waves, reaction–diffusion, Poisson, Helmholtz) — directly refuting the "operator learning doesn't scale / doesn't generalize across equation families" objection below.

$$\mathcal{G}_\theta: \mathcal{U} \to \mathcal{V}, \quad u_0 \mapsto u(\cdot, T)$$

A key conceptual distinction Poseidon stresses: **operator learning (full trajectory from the initial datum) is *not* the same task as next-step-with-context prediction** (paradigm 1). MPP/DPOT do the latter and underperform Poseidon substantially; the operator framing is what enables continuous-in-time evaluation at any lead time.

### Advantages
- **Discretization-invariant:** train at low resolution, infer at high resolution (DeepONet/FNO); Poseidon shows multiscale transformers approximate this while scaling
- Physics can be built directly into the architecture (PC-DeepONet: exact divergence-free output; VarMiON: branch network derived from the PDE's variational weak form)
- D2NO enables multi-operator pretraining with LoRA fine-tuning — closest current analog to parameter-efficient physics transfer learning
- Physics-informed zero-shot fine-tuning via PDE residual loss requires no labeled solution data for new equations
- **all2all / semi-group data amplification** ([[poseidon-pde-foundation-model]]) turns $K$ snapshots into $O(K^2)$ training pairs — architecture-agnostic and applicable to any time-dependent-PDE PFM

### Disadvantages
- DeepONet/FNO models remain small (10M–100M params); **Poseidon broke this barrier** (21M–629M) but is still patch-token grid-tied (its resolution-invariance is approximate, not function-space exact like FNO)
- Branch/trunk decomposition in DeepONet is a structural constraint that may limit representational capacity
- FNO assumes periodic boundary conditions and regular grids — severe limitation for real geometries
- Cross-equation-family generalization was thought poor — **Poseidon weakened this objection** (Euler/NS → Allen–Cahn, Poisson) but only from a curated, diverse pretraining set

**Current scale:** ~10M–100M params (DeepONet/FNO); **up to 629M (Poseidon)**

---

## 6. Hard Physics-Constrained Architectures

**Examples:** [[pc-deeponet-cfd]], [[varmion-viscous-flows]]

The architecture is designed so that its outputs **exactly** satisfy physical constraints by algebraic construction — not as a loss penalty but as an inviolable property.

**PC-DeepONet:** Divergence-free velocity enforced via skew-symmetric Jacobian:
$$\mathbf{v} = \nabla \times (J_b - J_b^\top)$$

**VarMiON:** Branch network architecture determined by the discrete weak form of the governing PDE, ensuring the output lies in the correct solution space.

### Advantages
- Physical laws are guaranteed — no training needed to enforce them, no risk of violation
- PC-DeepONet achieves 7× accuracy improvement over unconstrained baseline purely from the divergence-free constraint
- Architectures derived from variational form encode deep mathematical structure of the PDE

### Disadvantages
- Each constraint must be hand-derived for each specific PDE — requires expert knowledge per equation
- The constraint narrows the model's hypothesis class — may be unable to represent solutions to structurally different equations
- Scaling to 3D multiphysics adds algebraic complexity that may become intractable
- Maximally task-specific — fundamentally opposed to the "general model" philosophy

**Current scale:** ~10M–100M params

---

## 7. Spectral / Fourier Space Methods

**Examples:** [[pisd-physics-informed-spectral-diffusion]], and FNO (discussed in [[neural-operators]])

Operations are performed in frequency (wavenumber) space rather than physical space, exploiting the structured spectral properties of PDE solutions.

$$\mathcal{E}(f)(k) = \frac{\hat{f}(k)}{s_k}, \qquad s_k^2 = \text{Var}\!\left(\widehat{X_\text{data}}(k)\right)$$

### Advantages
- Global receptive field in a single convolution (FNO) — no need for deep stacks to propagate information across the domain
- Sobolev regularity is naturally preserved (PISD) — PDE operators remain well-defined at every noise level during diffusion
- Efficient: PISD is 3–15× faster than pixel-space diffusion; FNO outperforms standard convolution on large periodic grids
- Frequency-wise normalization is theoretically grounded via Bochner's theorem

### Disadvantages
- Assumes periodic or homogeneous boundary conditions — real geometry requires workarounds
- Non-uniform grids and irregular boundaries break the FFT assumption
- FNO truncates high-frequency modes, which may discard physically important small-scale structure
- Spectral representations are less interpretable than physical-space outputs

**Current scale:** ~10M–100M params (standalone); incorporated into larger models (PISD)

---

## 8. Graph Network-Based Simulators (Particle GNNs)

**Examples:** [[gns-graph-network-simulators]] (Sanchez-Gonzalez et al., 2020), [[dynami-cal-graphnet]] (Dynami-CAL GraphNet)

Rather than operating on continuum fields on grids, these models represent physical systems as **particle graphs** — each particle is a node, interactions are edges — and compute dynamics via learned message-passing. An Euler integrator updates positions from predicted accelerations.

$$\hat{X}^{t+1} = \text{Update}\!\left(\hat{X}^t,\ d_\theta(\hat{X}^t)\right), \quad d_\theta = \text{Decode} \circ \underbrace{\text{GN}^M \circ \cdots \circ \text{GN}^1}_{\text{Processor}} \circ \text{Encode}$$

**Conservation-constrained variant (Dynami-CAL GraphNet):** Exact conservation of linear and angular momentum is enforced architecturally via antisymmetric edge-local reference frames:
$$\vec{F}_{ij} = \sum_k f_k(m_{ij})\, \hat{e}_k^{ij}, \quad \hat{e}_k^{ij} = -\hat{e}_k^{ji},\ m_{ij} = m_{ji} \implies \vec{F}_{ij} = -\vec{F}_{ji}$$

### Advantages
- **Irregular geometry natively:** operates on arbitrary particle configurations — no fixed grid, no FFT assumptions, handles complex boundaries via ghost-node reflection
- **Multi-material generalization:** a single GNS model trained on water, sand, and goop simultaneously learns cross-material interactions
- **Extreme generalization:** GNS trained on 2.5k particles generalizes to 85k particles over 34× longer trajectories at test time
- **Exact conservation possible:** Dynami-CAL proves that Newton's third law can be hard-wired into the architecture, holding even under dissipation and external forces
- **Spatiotemporal memory:** edge memory across message-passing sub-steps enables fine-grained temporal reasoning within a single prediction step

### Disadvantages
- **Scale ceiling:** current models are small (~1M–10M params); have not been scaled to foundation model size due to over-smoothing and optimization difficulties
- **No long-range interactions by default:** message-passing radius $R$ limits the receptive field; elliptic PDEs (Poisson, Stokes) require global coupling that local GNNs miss
- **Particle-only:** does not naturally handle continuum field quantities (velocity fields, pressure) — requires reformulation as particle approximations (SPH, MPM)
- **Conservation constraints are mechanics-specific:** Dynami-CAL's conservation laws apply to classical mechanics; deriving analogous constraints for electromagnetism or general relativity requires separate derivations

**Current scale:** ~1M–10M params (particle GNNs); ~5M–500M params (SE(3)-equivariant atomistic GNNs like EquiformerV3 [[equiformer-v3]])

---

## Summary Comparison

| Approach | Best For | Critical Weakness | Current Scale |
|---|---|---|---|
| Autoregressive AR | Time evolution, in-context learning | Error accumulation | 385M–1.3B |
| Neural differentiator | Stable long-rollout dynamics | Coupled to numerical integrator | 385M |
| Diffusion / generative | Chaotic systems, inverse problems | Inference cost; not AR-native | ~100M–1B |
| Masked multimodal | Heterogeneous data, inverse problems | Not causal; dynamics unproven | 300M–3.1B |
| Neural operators | Discretization-invariant mapping | Not yet scaled; equation-specific | ~10M–100M |
| Hard physics constraints | High-accuracy specialized tasks | Equation-specific; no generality | ~10M–100M |
| Spectral methods | Periodic domains, efficiency | Irregular geometry fails | ~10M–100M |
| Particle GNNs (GNS / Dynami-CAL) | Irregular geometry, many-body, exact conservation | Not yet scaled; limited long-range | ~1M–10M |

---

## Toward a Hybrid Architecture

No current approach satisfies all five PFM capability requirements (see [[physics-foundation-models]]). The most plausible near-term path is a synthesis:

1. **Autoregressive backbone** (Walrus/GP$_{\text{hy}}$T style) for time evolution and in-context dynamics learning
2. **Universal tokenization** (AION-1 style) to handle field heterogeneity across modalities and resolutions
3. **Test-time physics guidance** (PISD style) for constraint enforcement and inverse problem solving — without retraining
4. **Neural operator ideas** (discretization invariance, LoRA fine-tuning) for efficient adaptation to new equations
5. **Spectral latent space** as the common field representation, with Sobolev regularity guarantees
6. **Particle GNN module** (GNS / Dynami-CAL style) for discrete many-body subsystems (granular flows, molecular dynamics, structural mechanics) with exact conservation constraints

The separate-channel field representation in PDE-Transformer and the modality-specific tokenization in AION-1 are convergent ideas from different communities pointing at the same solution: heterogeneous fields need heterogeneous representations, unified in a shared latent space.

A key architectural gap revealed by the GNS and Dynami-CAL literature: the largest current PFMs (Walrus, GP$_{\text{hy}}$T) operate on **regular grids** and handle **continuum fields**. Particle-based many-body physics — granular flows, molecular dynamics, structural collisions — requires a fundamentally different representation. A complete PFM likely needs a hybrid continuum-particle backbone, with a shared latent space bridging both representations.

**[[regime-moe-architecture]]** arrives at a structurally similar conclusion — heterogeneous representations unified by a shared conditioning contract — from a different starting point: a first-principles survey of physics regimes rather than a survey of published models. It goes further by arguing the *single dense backbone* framing above (paradigms 1–8, synthesized into "one hybrid model") may itself be the wrong shape, proposing instead a large MoE partitioned by equation family (elliptic/hyperbolic/parabolic, per its finding 1), with continuous FiLM-style conditioning within each expert and hard routing only across genuinely different theory types. This has not been reconciled against the roadmap below.

---

## Open Questions

1. Can autoregressive error accumulation be solved at scale, or is a generative (diffusion) backbone ultimately necessary for long rollouts?
2. Is physics encoding better as a soft loss, a test-time constraint (PISD), or a hard architectural constraint — and does the answer depend on the application?
3. At what parameter scale do physics foundation models exhibit the qualitative capability jumps seen in LLMs?
4. Can masked multimodal pretraining (AION-1 style) be extended to causal dynamics — or is there a fundamental tension between bidirectional reconstruction and forward prediction?
5. Can particle GNNs (GNS paradigm) be scaled to 100M+ parameters without over-smoothing, and how do they compare to continuum field models at equal compute on the same physics?
6. Can a single architecture bridge continuum fields (PDEs on grids) and discrete many-body physics (particles, molecules) in a unified representation?
7. Is a single conditioned dense backbone (this page's hybrid roadmap, and [[00-noether-1.1-overview]]'s concrete instantiation of it) sufficient, or does breadth across regimes fundamentally require expert partitioning ([[regime-moe-architecture]])? Unresolved — no head-to-head evaluation exists yet.

---

## See Also

- [[physics-foundation-models]]
- [[autoregressive-rollout-stability]]
- [[diffusion-models-physics]]
- [[neural-operators]]
- [[transformer-architectures]]
- [[in-context-learning-physics]]
- [[gaussian-processes]]
- [[poseidon-pde-foundation-model]]
- [[walrus-paper]]
- [[gphyt-physics-foundation-model]]
- [[pde-transformer-paper]]
- [[aion-1-astronomy]]
- [[00-token-representation-overview]]
- [[pisd-physics-informed-spectral-diffusion]]
- [[deeponet-multi-operator]]
- [[gns-graph-network-simulators]]
- [[dynami-cal-graphnet]]
- [[equiformer-v3]]
- [[equivariant-gnns]]
- [[regime-moe-architecture]]
- [[pfm-purpose-and-direction]]
- [[incremental-transfer-roadmap]]
