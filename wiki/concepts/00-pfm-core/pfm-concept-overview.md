# What Is a Physics Foundation Model?

**Type:** Core Concept — Vision & Gaps  
**Related Concepts:** [[physics-foundation-models]], [[pfm-architecture-approaches]], [[possible-architectures]], [[in-context-learning-physics]], [[world-models-physics-ai]], [[autoregressive-rollout-stability]], [[neural-operators]], [[diffusion-models-physics]]

---

## The Idea in One Paragraph

A **Physics Foundation Model (PFM)** is a single large neural network — pretrained once on diverse physical simulation data — that can be pointed at any new physical system and immediately produce useful predictions, without retraining, without re-specifying the governing equations, and without a domain expert writing a custom solver. It is the physics analog of what GPT-4 is to language: a general-purpose reasoner that works across domains because it has internalized deep regularities of the subject, not just memorized specific examples.

The analogy to LLMs is exact and intentional. LLMs have a "train once, deploy anywhere" property because natural language shares deep structural regularities across domains — grammar, syntax, semantics — that transfer. The PFM hypothesis is that **physical systems share analogous deep regularities** — conservation laws, symmetries, scale separation, smooth field evolution — that a sufficiently large and well-trained model can internalize and transfer.

---

## What a PFM Would Be Able to Do

### 1. Simulate Any Physical System via In-Context Learning

Given a short sequence of snapshots of a physical state — no equation specification, no parameter tuning — a PFM would infer the governing dynamics from the trajectory itself and roll forward in time:

$$\hat{x}_{t+1} = f_\theta\!\left(\underbrace{[x_{t-n}, \ldots, x_{t-1}, x_t]}_{\text{context: what the physics is doing}}\right)$$

This is **in-context learning for physics**: the model treats the input trajectory as a "prompt" describing the physics, just as an LLM treats a few-shot example as a description of the task. Demonstrated at 385M–1.3B parameters by GP$_{\text{hy}}$T and Walrus across up to 19 physical scenarios and 63 simultaneous fields.

### 2. Generalize Zero-Shot to Unseen Systems

A PFM would produce physically plausible predictions for systems it was never trained on — new geometries, new materials, new parameter regimes — because it has learned the structure of physical dynamics, not a lookup table of trained cases.

The benchmark for this is **out-of-distribution generalization**: train on water, sand, and elastic solids; test on glass or plasma. GNS demonstrated 34× particle count generalization and 8× longer rollouts than training. The PFM target is generalization across *equation families*, not just parameter ranges.

### 3. Handle Multi-Physics Simultaneously

A PFM would natively couple:
- **Fluid dynamics** (Navier-Stokes, Euler, MHD)
- **Heat and mass transfer** (advection-diffusion)
- **Structural mechanics** (elasticity, fracture)
- **Electromagnetics** (Maxwell's equations)
- **Many-body particle physics** (molecular dynamics, granular flows)

…within a single forward pass, without separate solvers for each domain. The fields may couple (fluid-structure interaction, magnetohydrodynamics) and the model handles this implicitly by learning the joint dynamics. Walrus demonstrated this at the continuum field level; AION-1 extended the paradigm to 39 observational modalities in astronomy.

### 4. Solve Inverse Problems

Given an observed physical state and knowledge of the goal, a PFM would work *backward* — inferring initial conditions, boundary conditions, material parameters, or governing equation parameters that produced the observation. This is scientifically important because experiments typically measure effects, not causes:

$$p(\theta \mid x_{\text{obs}}) \propto p(x_{\text{obs}} \mid \theta)\, p(\theta)$$

PISD's diffusion posterior sampling (DPS) demonstrates this: the trained model is steered at inference time by a PDE residual gradient, solving the inverse problem without retraining.

### 5. Quantify Uncertainty for Chaotic Systems

Physical systems exhibit intrinsic unpredictability (turbulence, weather beyond ~10 days, granular flows). A PFM with a generative backbone would produce **ensemble predictions** — a distribution over futures, not a single deterministic trajectory — enabling calibrated uncertainty quantification:

$$p(x_{t+1:T} \mid x_{0:t}) \quad \text{(distribution over futures, not a point estimate)}$$

Latent diffusion models consistently outperform deterministic surrogates for chaotic PDEs by modeling this distribution rather than predicting its mean.

### 6. Discover Scientific Laws (World Model Mode)

In its most ambitious form, a PFM would not just predict — it would **understand**. Given trajectory data, it would recover the underlying governing law in symbolic form: the force law $F \propto 1/r^2$ from orbital trajectories, or the correct scaling exponent from turbulence data.

This requires a **world model** (Newtonian mode) rather than a **curve fitter** (Keplerian mode). The distinction: a world model encodes local dynamical laws that generalize out-of-distribution; a curve fitter memorizes trajectory statistics that fail outside the training domain. Short context windows ($n = 2$ for second-order ODEs) force the model toward the Newtonian regime.

### 7. Run Faster Than Classical Solvers

A trained PFM amortizes computation at inference time. Classical solvers for turbulent Navier-Stokes at high Reynolds number are prohibitively expensive ($\sim 10^{10}$ operations per timestep for direct numerical simulation). A PFM predicts the same output via a single forward pass of learned computation — trading offline training cost for online inference speed. Current surrogates achieve $10^2$–$10^4\times$ speedups over classical methods.

---

## What Is Currently Lacking

### Gap 1: No Unified Representation for Heterogeneous Physics

Physical fields differ in type (scalar, vector, tensor), dimensionality (1D, 2D, 3D), resolution (uniform/irregular/adaptive), and domain geometry (periodic box, arbitrary curved boundaries). No current model handles all of these in a single representation.

- Grid-based transformers (Walrus, GP$_{\text{hy}}$T) require uniform grids and assume periodic or structured boundary conditions.
- Particle GNNs (GNS, Dynami-CAL) handle irregular geometry natively but cannot process continuum fields.
- AION-1's modality-specific tokenization handles 39 astronomical modalities but is not designed for PDE dynamics.

**What is needed:** A universal tokenization scheme that encodes both grid-based field data and unstructured particle/mesh data in a shared latent space, with built-in resolution invariance. The separate-channel representation in PDE-Transformer and the modality-specific tokens in AION-1 are convergent ideas pointing toward this, but neither has been implemented at general continuum-particle scope.

### Gap 2: Error Accumulation in Autoregressive Rollouts

Current autoregressive models amplify small per-step errors exponentially over long rollouts. This is the most studied unsolved problem in the literature:

$$\|e_T\| \approx \|e_1\| \cdot \lambda^T \quad (\lambda > 1 \text{ for unstable modes})$$

Partial mitigations exist — patch jittering (Walrus), derivative prediction (GP$_{\text{hy}}$T), noisy context training (Kepler/Newton) — but none solves the problem in general. Diffusion models avoid the issue by modeling the full distribution, but at the cost of inference speed and natural autoregressibility.

**What is needed:** A training or architectural strategy that provably bounds error growth over long rollouts without sacrificing single-step accuracy. The most promising theoretical candidate is combining derivative prediction (smooth targets) with a generative backbone (distributional stability), but this has not been implemented at scale.

### Gap 3: Scale Gap Between Current PFMs and Demonstrated LLM Capability

Current PFMs are 300M–3.1B parameters. LLMs demonstrated qualitative capability jumps (emergent reasoning, few-shot learning) around 10B–175B parameters. The scaling laws for physics models are unknown:

- Does a 10B-parameter PFM exhibit qualitatively new physics capabilities?
- Is the critical parameter count lower for physics (because physical laws are more structured than language) or higher (because physical simulation is a harder computational problem)?
- Do the same power-law scaling laws that govern language model loss apply to physics emulation error?

**What is needed:** Systematic scaling experiments on a common benchmark, sweeping model size from 1M to 10B parameters and measuring whether physics generalization improves smoothly or in phase transitions. No such study exists.

### Gap 4: Physics Enforcement Without Sacrificing Generality

Hard physics constraints (exact conservation laws, divergence-free output, variational structure) dramatically improve accuracy when applied correctly — PC-DeepONet achieves 7× improvement from a divergence-free constraint alone; Dynami-CAL achieves exact momentum conservation architecturally. But these constraints are **equation-specific**: each requires custom mathematical derivation for each PDE.

The tension is fundamental: the strongest physics encodings (Level 5: hard architectural constraints) are the most equation-specific and therefore the least general. The most general approaches (Level 1: data-driven only) provide no physics guarantees.

| Physics Encoding Level | Example | Generality | Accuracy |
|---|---|---|---|
| 1. Data-driven only | Walrus, GP$_{\text{hy}}$T | Universal | Baseline |
| 2. Soft loss constraints | PI-DeepONet | Wide | Moderate improvement |
| 3. Test-time guidance | PISD (DPS) | Wide | Large improvement, costly |
| 4. Architectural soft bias | GNN-PB energy bottleneck, A-DGN | Moderate | Strong improvement |
| 5. Hard architectural constraints | PC-DeepONet, Dynami-CAL, HNN | Narrow (equation-specific) | Strongest |

**What is needed:** A framework for automatically deriving hard architectural constraints from a symbolic specification of the governing physics — effectively a compiler from equations to architecture. This does not exist. Current hard constraints require manual mathematical derivation per equation.

### Gap 5: Irregular Geometry at Foundation Model Scale

The largest current PFMs (Walrus at 1.3B params, GP$_{\text{hy}}$T at 385M params) operate on **regular grids**. Real engineering geometry — turbine blades, ship hulls, human vasculature, aircraft wings — is irregular. The Fourier Neural Operator assumes periodic grids; transformer patch methods are approximate on curved boundaries.

GNN-based models (GNS, EquiformerV3) handle irregular geometry natively, but have not been scaled past ~10M–500M parameters due to over-smoothing and optimization difficulties.

**What is needed:** Either (a) a GNN architecture that scales to 1B+ parameters without over-smoothing, or (b) a learned mesh parameterization that maps arbitrary geometries onto regular grids while preserving physical structure, or (c) a hybrid continuum-particle backbone operating in a shared latent space.

### Gap 6: Multi-Scale Dynamics Without Explicit Supervision

Physical systems span enormous scale ranges. Turbulence cascades energy from kilometer-scale eddies to millimeter-scale dissipation. Weather involves planetary-scale circulation and local convective cells simultaneously. Current models either:
- Process everything at a single fixed resolution (missing small-scale structure), or
- Use explicit multi-resolution architectures that must be designed for each scale range.

The multipole GNN operator achieves $O(N)$ complexity for all-range interactions via a learned FMM-analog hierarchy, but this has not been validated at PFM scale. Spectral methods (FNO, PISD) capture multi-scale structure efficiently but only on periodic domains.

**What is needed:** A resolution-adaptive architecture that automatically identifies the relevant scales for each physical system at inference time, without prior specification of the scale hierarchy.

### Gap 7: No Benchmark Infrastructure

Unlike language models (BIG-Bench, HELM, MMLU) or image models (ImageNet, COCO), there is no standard benchmark suite for measuring PFM progress. Papers report results on different PDEs, different resolutions, different time horizons, different metrics, making direct comparison impossible.

**What is needed:** A physics equivalent of BIG-Bench — a standardized multi-PDE benchmark covering:
- 1D benchmarks (Burgers', Allen-Cahn, KdV)
- 2D benchmarks (2D Navier-Stokes, wave equation, diffusion)
- 3D benchmarks (3D NS, MHD)
- Multi-physics benchmarks (fluid-structure interaction)
- Irregular geometry benchmarks
- Long-rollout stability benchmarks (1000+ timesteps)

The wiki's proposed Burgers' benchmark (see [[possible-architectures]]) is a minimal starting point for the 1D case only.

### Gap 8: Training Data at Scale

High-fidelity physics simulations are expensive. A single 3D turbulent flow simulation at DNS resolution can require millions of CPU-hours. LLMs trained on the entire internet; the equivalent for a PFM would require a library of high-fidelity simulations covering the full diversity of physical systems — which does not exist.

Current datasets:
- The Well (Polymathic AI): ~15TB of multi-physics PDE data — the largest known PFM training set, but still orders of magnitude smaller than language pretraining corpora.
- Genesis physics engine: capable of generating simulation data at 10–80× faster than Isaac/MuJoCo, but focused on robotics (rigid body, deformables) rather than continuum fields.

**What is needed:** A standardized, large-scale, diverse multi-physics simulation dataset — the "Common Crawl of physics" — covering fluid dynamics, electromagnetics, structural mechanics, thermodynamics, and many-body particle physics at sufficient resolution for foundation model training.

---

## The Distance to a True PFM: A Summary

| Requirement            | Current State                                            | Gap                                                |
| ---------------------- | -------------------------------------------------------- | -------------------------------------------------- |
| Multi-physics learning | Demonstrated for continuum fields (Walrus: 19 scenarios) | Particle/many-body not unified with continuum      |
| In-context learning    | Demonstrated at 385M–1.3B params                         | Unknown if qualitative jumps occur at larger scale |
| Zero-shot transfer     | Demonstrated within equation families                    | Fails across equation families (Euler → NS → MHD)  |
| Long-horizon stability | Partially mitigated (patch jitter, derivative pred.)     | No general solution; error accumulation unsolved   |
| Variable resolution    | Operator learning achieves this at small scale           | Not demonstrated at foundation model scale         |
| Irregular geometry     | GNNs handle this at small scale                          | Not scaled to foundation model size                |
| Physics guarantees     | Hard constraints exist but are equation-specific         | No general auto-derivation of constraints          |
| Scientific discovery   | Demonstrated for simple systems (planetary orbits)       | Not demonstrated for PDEs or multi-physics         |
| Scale                  | 300M–3.1B params                                         | Scaling laws unknown; qualitative jumps uncertain  |
| Training data          | ~15TB (The Well)                                         | Orders of magnitude below language pretraining     |
| Benchmark              | No standard suite                                        | No way to track progress across papers             |

---

## **[AI Inference]:** The Most Tractable Near-Term Path

The strongest synthesis from the current literature:

1. **Autoregressive backbone** (Walrus/GP$_{\text{hy}}$T style) as the foundation — battle-tested, natural in-context learning, most scalable paradigm.
2. **Universal modality-specific tokenization** (AION-1 style) to handle field heterogeneity — separate channels per field type, shared latent space.
3. **Derivative prediction heads** (GP$_{\text{hy}}$T style) to mitigate error accumulation — smoother targets, integrator-enforced consistency.
4. **Test-time physics guidance** (PISD style) for constraint enforcement and inverse problems — no retraining required for new equations.
5. **Particle GNN module** (Dynami-CAL style) with exact momentum conservation for many-body subsystems — hard-wired Newton's third law.
6. **Spectral latent space** (PISD style) as the shared field representation — Sobolev regularity guarantees, global receptive field.

The critical missing piece that no current paper addresses: a **learned bridge between the continuum-field representation** (items 1–4 above) **and the particle representation** (item 5). A complete PFM likely requires both, unified in a shared latent space.

---

## See Also

- [[physics-foundation-models]] — formal definition and capability requirements
- [[pfm-architecture-approaches]] — 8 paradigms compared; hybrid synthesis roadmap
- [[possible-architectures]] — 5-architecture benchmark plan; Burgers' benchmark spec
- [[in-context-learning-physics]] — the in-context learning mechanism for physics
- [[world-models-physics-ai]] — Newtonian vs. Keplerian world models; OOD generalization
- [[autoregressive-rollout-stability]] — error accumulation problem in depth
- [[neural-operators]] — discretization-invariant operator learning
- [[diffusion-models-physics]] — stochastic emulation for chaotic systems
- [[equivariant-gnns]] — GNNs with conservation law constraints
- [[walrus-paper]], [[gphyt-physics-foundation-model]] — leading AR approaches
- [[aion-1-astronomy]] — universal tokenization for heterogeneous modalities
- [[pisd-physics-informed-spectral-diffusion]] — test-time physics guidance
- [[gns-graph-network-simulators]], [[dynami-cal-graphnet]] — particle GNN paradigm
- [[pfm-purpose-and-direction]] — reframes these 8 gaps against a concrete multi-agent deployment target, adds 3 more (cross-boundary consistency, composability, verified fallback)
- [[incremental-transfer-roadmap]] — build strategy for closing these gaps via transfer rather than from-scratch training
