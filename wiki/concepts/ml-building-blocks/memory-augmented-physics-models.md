# Memory-Augmented Physics Models: Hybrid Static-Dynamic Architectures

**Type:** Architecture Concept  
**Related Concepts:** [[transformer-architectures]], [[mixture-of-experts]], [[neural-surrogates]], [[pfm-architecture-approaches]], [[possible-architectures]]  
**Related Summaries:** [[conditional-memory-engram]], [[walrus-paper]], [[deep-memory-dissipative]], [[gphyt-physics-foundation-model]]

---

## Overview

A **memory-augmented physics model** pairs dynamic neural computation (transformer, MoE) with static pre-computed **physics memory** — a lookup table of canonical solutions, basis functions, Green's functions, or operator matrices. Rather than learning all physics from data, the model retrieves known solutions when applicable and uses sparse neural experts for:
- Heterogeneous interactions (material-specific, boundary-dependent)
- Nonlinear corrections to canonical solutions
- Regimes outside the pre-computed library

The key insight from Engram ([[conditional-memory-engram]]) is the **U-shaped scaling law**: the optimal allocation between memory size and compute size exhibits a characteristic tradeoff, neither memory-heavy nor compute-heavy, but balanced.

---

## Conceptual Framework

### The Static-Dynamic Split

Total model capacity $N = N_{\text{mem}} + N_{\text{compute}}$ is divided between:

**Static Memory (Physics Library):**
$$\mathcal{M} = \{\text{Green's functions}, \text{canonical solutions}, \text{operator bases}, \text{discretization matrices}\}$$

Accessed via deterministic addressing (no learned routing). Examples:
- $\mathcal{M}_{\text{Poisson}}(k, L, BC) \to$ solution for Poisson equation with wavenumber $k$, domain size $L$, boundary condition $BC$
- $\mathcal{M}_{\text{basis}}(n, L_{\max}) \to$ orthogonal polynomial basis of degree $n$, up to $L_{\max}$
- $\mathcal{M}_{\text{kernel}}(\nu, d) \to$ viscous kernel matrix for viscosity $\nu$, discretization $d$

**Dynamic Compute (Neural Backbone):**
$$\mathcal{C}_\theta: \text{MoE or sparse transformer predicting} \quad \Delta u_\theta = u_{\text{true}} - u_{\text{memory}}$$

The neural part learns to **correct** the static library, not replicate it.

### Router and Sparsity Allocation

A router decides per-region (or per-query) whether the problem is "canonical" (retrieve from memory) or "heterogeneous" (use MoE):

$$\hat{u}_i = \begin{cases} \mathcal{M}(p_i) & \text{if } d(p_i, \text{canonical}) < \tau \\ \mathcal{M}(p_i) + \mathcal{C}_\theta(p_i) & \text{otherwise} \end{cases}$$

where $p_i$ is a problem descriptor (equation parameters, boundary type, material) and $d$ is a distance metric to the canonical regime.

---

## Physics Encoding Spectrum Placement

Memory-augmented models occupy **level 2–3** on the physics encoding spectrum (from CLAUDE.md):

1. **Data-driven only** — pure neural (Walrus, GNS)
2. **Soft loss constraints** — loss term enforces physics (PI-DeepONet)
3. **Test-time physics guidance** — inference-time constraints (PISD)
4. **Architectural soft biases** — bottleneck, MoE structure
5. **Hard architectural constraints** — exact divergence-free, conservation

Memory-augmented models sit at **level 2.5**: the static library encodes physics at the architectural level (hard bias), but the neural correction allows flexibility (soft learning).

---

## Multi-Scale Application

The key advantage for physics: **canonical solutions typically operate at one scale or regime; memory allows scaling across many**.

### Example: Navier-Stokes with heterogeneous materials

| Scale / Regime | Memory | Compute |
|---|---|---|
| Low-Re laminar flow | Pre-computed Stokes operator (analytical) | Skip (zero correction) |
| Moderate-Re smooth flow | Basis projection of standard NS solver | Learn Re-dependent corrections |
| High-Re / turbulence | Low-rank POD basis | Full MoE for nonlinear closure |
| Shock / material interface | None (heterogeneous) | Full neural computation |

Memory handles the "smooth, frequently-encountered" regimes; MoE handles the "rare, heterogeneous" ones. Result: smaller total model than pure neural.

---

## U-Shaped Scaling Law for Physics

Engram's empirical finding: given a fixed total parameter budget $N$, performance is non-monotonic in $N_{\text{mem}}$:

$$\text{Loss}(N_{\text{mem}}) = \alpha (N_{\text{mem}} - N_{\text{opt}})^2 + \beta$$

**Physical interpretation:**
- **Too little memory** ($N_{\text{mem}} \ll N_{\text{opt}}$): neural backbone must recompute frequent patterns (Poisson solves, Gaussian kernels, polynomial projections) — wasteful
- **Too much memory** ($N_{\text{mem}} \gg N_{\text{opt}}$): most memory entries are unused; compute budget is underutilized
- **Optimal balance** ($N_{\text{mem}} \approx N_{\text{opt}}$): memory handles frequent canonical solutions, MoE focuses on corrections

For a PFM, this suggests:
- **Small PFM (1M–10M params):** allocate ~50–70% to memory (basis functions, kernels)
- **Large PFM (100M–1B params):** allocate ~30–50% to memory; MoE scales faster
- **Empirical tuning:** search for $N_{\text{opt}}$ via scaling experiments

---

## Variants and Design Choices

### 1. Analytical vs. Data-Driven Memory

**Analytical memory:** Pre-computed via classical solvers (finite difference, spectral methods). Guaranteed accuracy for canonical regime; requires equation-specific effort per library entry.

**Data-driven memory:** Pre-trained neural networks on curated datasets; more generalizable across equations but less interpretable.

### 2. Basis-Function Memory vs. Solution Memory

**Basis-function approach:** Library stores orthogonal polynomial bases, spherical harmonics, wavelets. Router computes $u_{\text{mem}} = \sum_n c_n \phi_n$, where $c_n$ are learned coefficients or retrieved analytically.

**Solution approach:** Library stores pre-computed solutions for specific parameter combinations (e.g., PDE solutions at $\nu = 0.001, 0.01, 0.1$). Router interpolates or retrieves nearest neighbor.

### 3. Addressing Scheme

**Deterministic:** Problem parameters → memory address via hash or learned encoder. Enables prefetching, deterministic runtime.

**Learned:** Router network predicts which memory entry is relevant. More flexible but less infrastructure-efficient.

---

## Connections to Existing PFM Work

**Neural Differentiator** ([[gphyt-physics-foundation-model]]): The GPhyT model predicts time derivatives; memory-augmented variant could store pre-computed derivative kernels for linear/canonical PDEs.

**Neural Operators** ([[deeponet-multi-operator]]): DeepONet's branch/trunk structure is analogous — branch network (memory-like, computes $u_0$) vs. trunk (dynamic, computes corrections). Memory-augmented models generalize this to **explicit lookup** instead of branch network.

**PC-DeepONet** ([[pc-deeponet-cfd]]): Hard divergence-free constraint is a form of physics memory — the constraint matrix is pre-computed, neural part learns residuals. Memory-augmented models extend this to soft memory (retrieve, don't enforce).

**Mixture of Experts** ([[mixture-of-experts]]): Conditional computation via expert routing. Engram adds a second dimension: memory (deterministic) vs. compute (routed). A PFM could combine both: route to experts AND optionally retrieve from memory if relevant.

---

## Open Questions

1. **What should the physics library contain for a general PFM?** Green's functions for canonical PDEs? Spectral bases? Pre-trained surrogates for specific material classes?

2. **How to design the router?** Learned or analytical? Should it be differentiable for end-to-end training, or frozen?

3. **Can memory-augmented models achieve the "train once, deploy anywhere" goal better than pure neural models?** Memory provides a fixed reference (canonical solutions); neural part learns variations. Does this architecture generalize better?

4. **Infrastructure:** What is the penalty for prefetching large operator matrices from host memory vs. keeping them in VRAM? Does this limit library size on resource-constrained devices?

5. **Multi-physics:** For coupled systems (fluid-structure, thermo-mechanical), how do you organize memory across physics domains? Separate lookup tables per domain, or a unified multi-physics library?

---

## See Also

- [[conditional-memory-engram]] — empirical grounding in LLMs; sparsity allocation problem
- [[gphyt-physics-foundation-model]] — neural differentiator; potential memory module for derivatives
- [[deeponet-multi-operator]] — branch/trunk analogy to memory-augmented design
- [[transformer-architectures]] — backbone architecture
- [[mixture-of-experts]] — conditional routing; pairs naturally with memory
- [[pfm-architecture-approaches]] — broader PFM design space
