# Multi-Scale / FMM-Analog Hierarchical GNN

**Type:** Architecture Concept / Proposal
**Status:** New synthesis page (2026-05-18)
**Related Concepts:** [[arch-gnn-physics-bottleneck]], [[equivariant-gnns]], [[neural-operators]], [[partial-differential-equations]], [[possible-architectures]]
**Related Summaries:** [[multipole-graph-neural-operator]], [[gns-graph-network-simulators]], [[dynami-cal-graphnet]], [[equiformer-v3]]

---

## Motivation

The single largest unaddressed gap in the current PFM architecture proposals (AR Transformer, Diffusion Backbone, Neural Differentiator, Physics-Mamba, GNN-PB; see [[possible-architectures]]) is **multi-scale coupling on irregular geometry**. All five architectures use one of three connectivity patterns:

1. **Local radius graph** (GNN-PB, GNS): $O(N k)$ cost; captures short-range interactions but misses long-range elliptic coupling (Poisson pressure, gravitational, Coulomb, viscous dissipation across the domain).

2. **Fully-connected / all-pairs attention** (AR Transformer, Diffusion): $O(N^2)$ cost; captures all interactions but does not scale to fine resolutions.

3. **Fourier / spectral** (PISD, neural operators on regular grids): $O(N \log N)$ cost; only works on regular periodic grids — fails on the irregular geometries that make GNN-based methods valuable in the first place.

Real physics has multi-scale interactions everywhere:
- **Turbulence:** energy cascades from large eddies to dissipative scales.
- **Gravitational N-body:** every body interacts with every other; close encounters require fine resolution.
- **Plasma physics:** Debye-screened short-range + long-range electromagnetic coupling.
- **Solid mechanics:** boundary layers + global stress equilibrium.
- **Atmospheric / oceanic flow:** local convection + planetary-scale circulation.

Classical numerical methods solved this problem decades ago. This page formalizes the neural analog.

---

## Background: How Classical Methods Solve Multi-Scale Coupling

| Method | Year | Cost | Idea |
|---|---|---|---|
| Direct N-body | – | $O(N^2)$ | Compute all pairs |
| Particle-mesh / Ewald summation | 1921 | $O(N \log N)$ | Long-range via FFT on a coarse grid; short-range direct |
| Barnes-Hut treecode | 1986 | $O(N \log N)$ | Hierarchical center-of-mass approximation; quadtree/octree |
| Fast Multipole Method (FMM) | 1987 | $O(N)$ | Multipole expansion of far field; recursive tree-based |
| Multigrid | 1977 | $O(N)$ | V-cycle of relaxation on hierarchy of grids |
| $\mathcal{H}$-matrices | 1999 | $O(N \log N)$ | Low-rank factorization of off-diagonal kernel blocks |

The key structural insight in all of these methods is **scale separation**: short-range interactions are computed directly on the fine scale; long-range interactions are computed cheaply on coarse representations (multipole moments, coarsened grids, low-rank factorizations). Both contributions are then combined to recover the full interaction.

A neural network architecture that does the same thing inherits the same scaling properties.

---

## The Neural Multipole / Multigrid Pattern

### Hierarchy of Graphs

Build $L$ graphs $G_1 \supset G_2 \supset \cdots \supset G_L$ where $G_\ell$ has $N_\ell$ nodes with $N_{\ell+1} = N_\ell / r$ for a fixed coarsening ratio $r$ (typically $r = 2$ or $4$). $G_1$ has the full $N$ nodes; $G_L$ has $O(1)$ or $O(\log N)$ nodes.

The coarsening is **adaptive**: in regions of high gradient (shocks, boundary layers, vortices) the coarsening ratio should be smaller; in regions of slow variation the coarsening can be more aggressive.

### V-Cycle Message Passing

At each prediction step, run a V-cycle:

```
Down-pass (fine → coarse):
  for ℓ = 1, ..., L-1:
    h_{ℓ+1} = Restriction_{ℓ→ℓ+1}(h_ℓ)

Inter-scale message passing (each level):
  for ℓ = 1, ..., L:
    h_ℓ = MessagePass(h_ℓ, G_ℓ, edges_ℓ; learned_kernel_ℓ)

Up-pass (coarse → fine):
  for ℓ = L-1, ..., 1:
    h_ℓ = h_ℓ + Prolongation_{ℓ+1→ℓ}(h_{ℓ+1})
```

Each level has its own learned message-passing kernel $\kappa_\ell$. The coarse levels have few nodes and few edges, so message-passing cost per level is small. Total cost is $O(N)$ because work per node is constant across the hierarchy.

### Learned Restriction and Prolongation

Both the restriction operator $V_{\ell \to \ell+1}$ (fine → coarse aggregation) and prolongation $V_{\ell+1 \to \ell}$ (coarse → fine broadcasting) are learned. The simplest implementations:

- **Restriction:** Each coarse node $i' \in G_{\ell+1}$ aggregates from a cluster of fine nodes $\{i \in G_\ell\}$ via learned attention or a fixed cluster-mean.
- **Prolongation:** Each fine node receives a weighted broadcast from its coarse parent + learned residual.

For physical interpretability, the restriction should compute **multipole moments**: zeroth moment (total mass / scalar quantity), first moment (center of mass / dipole), second moment (moment of inertia / quadrupole), etc. The prolongation broadcasts these back as low-order Taylor expansions evaluated at fine-node positions.

This is the *physical* parameterization — the [[multipole-graph-neural-operator]] uses learned restriction/prolongation without explicit moment structure, but EquiformerV3-style irreps make the moment structure native.

---

## Architectural Design

### Layer Structure

```
Input: irregular point cloud {p_i, x_i} with per-node features

1. Build hierarchy: G_1 ⊃ G_2 ⊃ ... ⊃ G_L
   - G_1 = original radius graph
   - G_{ℓ+1} = coarsened version of G_ℓ (k-means clustering, learned, or geometric subsampling)

2. Encode: h_i^0 = MLP_encode(x_i) for i ∈ G_1

3. V-cycle (M iterations):
   For each iteration m = 1, ..., M:
     # Down-pass
     for ℓ = 1, ..., L-1:
       h_{ℓ+1} = Restriction_ℓ(h_ℓ, cluster_assignments_ℓ)

     # Message passing at each level
     for ℓ = 1, ..., L:
       h_ℓ = MessagePass_ℓ(h_ℓ, G_ℓ, edges_ℓ)

     # Up-pass with residual
     for ℓ = L-1, ..., 1:
       h_ℓ = h_ℓ + Prolongation_ℓ(h_{ℓ+1})

4. Decode: y_i = MLP_decode(h_i^M) for i ∈ G_1
```

### Combination with Dynami-CAL Antisymmetric Conservation

At each level $\ell$, the message-passing kernel uses Dynami-CAL's antisymmetric edge frame. The fine-level conservation laws are exactly enforced; the coarse-level frames apply to aggregated multipole moments (which themselves transform consistently under SE(3)).

Practically: at fine levels, the antisymmetric edges produce node forces $\vec{F}_{ij} = -\vec{F}_{ji}$ that conserve momentum. At coarse levels, the antisymmetric edges produce aggregated quadrupole/dipole interactions that respect the same antisymmetry, conserving the moments. This is consistent with classical multipole theory: total momentum is the zeroth moment, and is conserved by any antisymmetric pairwise interaction at any level.

### Combination with EquiformerV3 Irreps

The multipole moments **are** irreps:

- Monopole ($L = 0$) = total mass / scalar charge
- Dipole ($L = 1$) = center of mass / dipole moment vector
- Quadrupole ($L = 2$) = moment of inertia tensor / charge quadrupole
- Higher $L$ = higher multipole moments

EquiformerV3-style features with $L_{\max} = 2$ or higher naturally store all relevant moments at coarse levels. The restriction operator computes the moments; the message passing at coarse levels operates on irreps; the prolongation evaluates the moment fields at fine-node positions.

This is the **principled combination**: multipole hierarchy gives the multi-scale structure, irreps give the moment representation, Dynami-CAL gives the conservation law at every level.

---

## Computational Properties

| Method | Cost | Long-range | Geometry | Memory |
|---|---|---|---|---|
| Local radius GNN | $O(Nk)$ | No | Irregular ok | $O(N)$ |
| All-pairs attention | $O(N^2)$ | Yes | Irregular ok | $O(N^2)$ |
| FNO | $O(N \log N)$ | Yes | Regular only | $O(N)$ |
| **Multipole GNN** | $O(N)$ | **Yes** | **Irregular ok** | $O(N \log N)$ |

The multipole GNN is the only known architecture that achieves all three desired properties: linear complexity, long-range coupling, and arbitrary geometry.

---

## Mathematical Foundation: Why This Works

The key theorem underlying both FMM and the neural multipole framework: **far-field expansions converge rapidly**.

For two well-separated clusters of points centered at $c_1, c_2$ with separation $r = \|c_1 - c_2\|$, the kernel $\kappa(x, y)$ for $x$ near $c_1$ and $y$ near $c_2$ admits the expansion

$$\kappa(x, y) = \sum_{|\alpha| \leq P} \sum_{|\beta| \leq P} \kappa_{\alpha\beta}(c_1, c_2)\, (x - c_1)^\alpha (y - c_2)^\beta + O\!\left(\frac{1}{r^{P+1}}\right)$$

with $P$ the multipole truncation order. The number of terms is $O(P^d)$ where $d$ is the spatial dimension; in 3D with $P = 4$, this is $\binom{4+3}{3} = 35$ terms per cluster pair. For a hierarchy of $\log N$ levels and constant work per level, the total cost is $O(N)$.

[[multipole-graph-neural-operator]] proves rigorously that the learned-kernel version inherits these scaling properties and demonstrates discretization invariance.

---

## Discretization Invariance

A multipole graph network learns a **continuous integral operator**, not a discretization-specific approximation. The same trained model can be evaluated on different mesh resolutions, with accuracy improving (rather than degrading) under mesh refinement. This is the defining property of a **neural operator** in the sense of [[neural-operators]] and is the reason FNO works across resolutions.

For a PFM intended to deploy at varying resolutions (low-res for screening, high-res for production), this property is essential. The multipole GNN is the only known architecture with this property on irregular geometry.

---

## Benchmark Adaptation

For the 1D Burgers' benchmark of [[possible-architectures]], the multipole structure is overkill — 1D periodic Burgers' has no long-range coupling worth modeling at this scale. The multipole architecture should be benchmarked separately on:

1. **2D Darcy flow** (elliptic, long-range pressure-velocity coupling) — the [[multipole-graph-neural-operator]]'s primary demonstration.
2. **2D Navier-Stokes vorticity** (long-range biot-savart coupling between vortex elements).
3. **3D N-body gravity** (textbook FMM test case).
4. **Irregular-geometry Poisson** (elliptic, irregular mesh) — where neither FNO nor local GNNs apply cleanly.

Expected performance: comparable to or better than baselines on all four, with the most pronounced advantage on long-range coupled tasks and irregular geometry simultaneously.

---

## Open Problems

1. **Adaptive coarsening.** How to choose the hierarchy adaptively per task? Fixed geometric subsampling is simple but suboptimal for shocks and boundary layers. Learned clustering (DiffPool, MinCutPool) helps but is itself a hard optimization problem. **[AI Inference]:** Multigrid-style algebraic coarsening (Ruge-Stüben) based on the residual of an auxiliary equation may be the principled solution.

2. **Translating exact conservation to coarse levels.** Dynami-CAL gives exact momentum conservation at the fine level. The coarse multipole moments transform consistently, so their *invariance* is automatic, but whether *exact conservation* extends across levels under arbitrary message passing is not fully established. Worth a careful proof.

3. **Combination with diffusion / generative backbones.** Can the multipole structure be added to the DiT denoiser of Arch 2 ([[arch-diffusion-backbone]])? At first glance the answer is yes — the denoiser can use multipole attention instead of standard attention — but the implications for the score function structure need analysis.

4. **Scaling to billion-parameter regime.** None of the existing multipole-GNN papers have been trained at PFM scale. Whether the multipole structure helps or hurts at 1B+ parameters remains untested.

---

## [AI Inference]

**[AI Inference]:** The combination of multipole hierarchy + Dynami-CAL antisymmetric frames + EquiformerV3 irreps + Hamiltonian message passing (see [[hamiltonian-message-passing]]) is the architectural endpoint of the "physics GNN" line of reasoning. No published work combines all four. Implementing this composition would test the limits of how much physical structure can be embedded in a GNN before expressive capacity becomes the bottleneck.

**[AI Inference]:** The multipole framework is naturally compatible with **multi-resolution time integration**. Different levels of the hierarchy correspond to different spatial scales, which have different characteristic time scales (small eddies evolve fast; large eddies evolve slow). A multi-rate integrator that takes small time steps on the fine levels and large time steps on the coarse levels — controlled by the Δt-as-input mechanism of [[arch-physics-mamba]] — would be the spatial-temporal hierarchical generalization.

**[AI Inference]:** Engram's static memory ([[conditional-memory-engram]]) and the multipole framework are spatial duals. Engram precomputes lookup tables for canonical cases; multipole precomputes coarse-level representations for far-field interactions. Both bifurcate computation across an axis that becomes regular at coarse scales (canonical solutions / multipole moments). A unified hybrid — static multipole moments for the most common cluster configurations + dynamic neural correction — would be the most aggressive computation-saving design point.

---

## Position on the Physics Encoding Spectrum

| Level | Description | Multipole GNN |
|---|---|---|
| 1 | Data-driven only | Adds spatial inductive bias of hierarchical locality |
| 2 | Soft loss | Optional |
| 3 | Test-time guidance (DPS) | Compatible |
| 4 | Architectural soft bias | Hierarchical scale separation is itself a level-4 bias |
| 5 | Hard architectural constraint | When combined with Dynami-CAL, exact conservation at each level |

Multipole hierarchy alone is a level-4 inductive bias (architectural soft bias for scale separation). Combined with Dynami-CAL or HNN-style components it climbs to level 5.

---

## Cross-Links

- [[multipole-graph-neural-operator]] — rigorous foundation
- [[arch-gnn-physics-bottleneck]] — the architecture this proposal upgrades
- [[possible-architectures]] — broader benchmark plan
- [[equivariant-gnns]] — irreps formalism
- [[neural-operators]] — discretization invariance
- [[hamiltonian-message-passing]] — complementary Hamiltonian design
- [[antisymmetric-signed-attention-transformer]] — complementary attention design
- [[action-based-noether-enforcement]] — complementary action-based design
- [[gns-graph-network-simulators]] — single-scale baseline
- [[dynami-cal-graphnet]] — conservation backbone
- [[equiformer-v3]] — irreps machinery
- [[unet-hierarchy-atlas-0.1]] — reuses this V-cycle as the pooling half of a graph U-Net, with a different expert at each level rather than a uniform message-passing operator
