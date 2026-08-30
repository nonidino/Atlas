# Equivariant Graph Neural Networks for Physics

**Type:** Core Concept  
**Related Concepts:** [[pfm-architecture-approaches]], [[neural-operators]], [[transformer-architectures]], [[physics-foundation-models]], [[possible-architectures]]  
**Related Summaries:** [[gns-graph-network-simulators]], [[dynami-cal-graphnet]], [[equiformer-v3]], [[eqnn-hep-lhc]], [[pc-deeponet-cfd]]

---

## Overview

Graph Neural Networks (GNNs) are a natural architecture for physical simulation: particles or mesh nodes become graph nodes, interactions become edges, and message-passing computes how each node influences its neighbors. The question is **what inductive biases to embed**. This page catalogs the three major levels of physical bias in GNN design:

1. **Spatial inductive bias only** — GNS-style: particles as nodes, local connectivity radius, message passing computes accelerations
2. **Symmetry inductive bias** — equivariant GNNs: features and outputs respect physical symmetry groups (SE(3), E(3))
3. **Conservation law inductive bias** — hard architectural constraints: outputs exactly satisfy conservation laws by construction

Each level is strictly stronger than the previous in terms of physical consistency.

---

## Level 1: Spatial Inductive Bias — Graph Network Simulators

**Reference:** [[gns-graph-network-simulators]] (Sanchez-Gonzalez et al., 2020)

The GNS framework treats physical simulation as message-passing on a particle graph. State representation:
$$X = \{\mathbf{x}_i\}_{i=1}^N, \quad \mathbf{x}_i = [\mathbf{p}_i,\ \dot{\mathbf{p}}_i^{t-C+1:t},\ \mathbf{f}_i]$$

The model follows **encode-process-decode**:
- **Encoder:** embeds particle states and pairwise displacements into a latent graph
- **Processor:** $M$ rounds of learned message-passing (graph network steps)
- **Decoder:** extracts per-particle accelerations; Euler integrator updates state

**Key finding:** Performance is determined primarily by (1) number of message-passing steps $M$ and (2) training noise injection for rollout stability. Other architectural details are secondary.

**Limitation:** No conservation laws guaranteed. The model approximates physics from data — conservation of momentum and energy is learned, not enforced, and fails under out-of-distribution conditions (e.g., high-momentum extrapolation).

---

## Level 2: Symmetry Inductive Bias — SE(3)-Equivariant GNNs

Equivariant GNNs embed the symmetry group of the physical law directly into the architecture. The dominant symmetry for 3D physics is **SE(3)** — the special Euclidean group of 3D rotations and translations:

$$f(R\mathbf{x} + \mathbf{t}) = R \cdot f(\mathbf{x}) \quad \forall R \in SO(3),\ \mathbf{t} \in \mathbb{R}^3$$

### Irreducible Representations (Irreps)

Features are represented as type-$L$ vectors transforming under Wigner-$D$ matrices:
$$f^{(L)} \mapsto D^{(L)}(R)\, f^{(L)}$$

Equivariant operations couple irreps via Clebsch-Gordan tensor products:
$$h^{(L_3)}_{m_3} = \sum_{m_1, m_2} C^{(L_3,m_3)}_{(L_1,m_1)(L_2,m_2)} f^{(L_1)}_{m_1} g^{(L_2)}_{m_2}, \quad |L_1 - L_2| \leq L_3 \leq L_1 + L_2$$

Higher $L_{\max}$ encodes finer angular structure (essential for forces). Tensor products cost $O(L_{\max}^6)$ naively; eSCN decomposition reduces this to $O(L_{\max}^4)$.

### Scalarization-Vectorization Paradigm (Dynami-CAL / EGNN / GMN / ClofNet)

An efficient alternative to full irreps: project 3D vectors onto a local frame (scalarization), process with standard MLPs, then reconstruct 3D vectors from learned scalar coefficients (vectorization).

EGNN computes:
$$\mathbf{v}_i \mathrel{+}= \sum_j \phi(m_{ij}) \cdot (\mathbf{x}_i - \mathbf{x}_j)$$

where $m_{ij}$ is a scalar edge embedding. This achieves E(n)-equivariance with low computational cost but limited expressivity — it can only produce forces along relative position vectors (central forces), not general non-central forces.

### EquiformerV3

The state-of-the-art SE(3)-equivariant architecture for 3D atomistic systems ([[equiformer-v3]]). Key advances:
- **Merged layer normalization:** shared RMS across all degrees, preserving relative magnitudes
- **SwiGLU-S² activations:** fast tensor products via sphere projection at $O(L_{\max}^4)$
- **Smooth radius cutoff:** envelope functions in attention ensuring continuity of learned PES

**Domain:** Molecular/materials science (DFT acceleration). Benchmarks: OC20, OMat24, Matbench Discovery.

---

## Level 3: Conservation Law Inductive Bias — Hard Architectural Constraints

**Reference:** [[dynami-cal-graphnet]] (Sharma & Fink, 2025)

The strongest form: outputs exactly satisfy conservation laws by algebraic construction. No training needed; no risk of violation under distribution shift.

### Antisymmetric Edge-Local Frame
An orthonormal basis $\{\hat{e}_k\}_{ij}$ is constructed so that $\hat{e}_k^{ij} = -\hat{e}_k^{ji}$. Edge embeddings are constructed to be node-symmetric: $m_{ij} = m_{ji}$.

**Linear momentum conservation (Newton's 3rd Law):**
$$\vec{F}_{ij} = \sum_k f_k(m_{ij}) \cdot \hat{e}_k^{ij} = -\vec{F}_{ji}$$

because coefficients are symmetric ($f_k(m_{ij}) = f_k(m_{ji})$) and basis vectors antisymmetric.

**Angular momentum conservation:**
$$\boldsymbol{\tau}_{ij} = \vec{A}_{ij} - (\vec{r}_i - \vec{x}_{0,ij}) \times \vec{F}_{ij}, \quad \vec{x}_{0,ij} = \vec{x}_{0,ji}$$

where $\vec{x}_{0,ij}$ is a shared predicted force application point. Provably ensures global angular momentum conservation via symmetry-preserving aggregation.

**Key advantage over energy-conserving Hamiltonian/Lagrangian GNNs:** Conservation of linear and angular momentum holds even in **dissipative systems with external forces** — energy conservation does not. This makes Dynami-CAL applicable to the full range of realistic physics (friction, inelastic collisions, external fields).

---

## Taxonomy of GNN Approaches for Physics

| Model | Level | Conservation | Domain | Scale |
|---|---|---|---|---|
| GNS | 1 (spatial) | None (learned) | Particle simulation | ~5M params |
| EGNN, GMN | 2 (SE(3) equivariant) | None | Molecular dynamics | ~1M params |
| ClofNet | 2 (equivariant + local frame) | Approx. linear only | Molecular dynamics | ~1M params |
| Dynami-CAL GraphNet | 2+3 (equivariant + conserving) | Exact linear + angular | Granular, biomolecular, robotic | ~1M params |
| NequIP, EquiformerV2 | 2 (SE(3) irreps) | None (approx. via smooth PES) | Atomistic DFT | 5M–30M params |
| EquiformerV3 | 2 (SE(3) irreps + smooth PES) | Approx. (smooth cutoff) | Atomistic DFT | 5M–500M params |
| PC-DeepONet | 3 (hard divergence-free) | Exact mass conservation | CFD (curved BFS) | ~10M params |

---

## Equivariant GNNs vs. Other PFM Paradigms

| Dimension | Equivariant GNNs | Autoregressive Transformers | Neural Operators |
|---|---|---|---|
| Representation | Particles/nodes | Grid patches or tokens | Function space (branch/trunk) |
| Geometry | Irregular, adaptive | Regular grid or patch | Regular or resolution-invariant |
| Conservation | Hard (with design) or none | None | None (soft loss only) |
| Scaling | ~1M–100M params | 100M–1.3B params | ~10M–100M params |
| Generalization | Particle count, geometry | In-context dynamics | Operator family (limited) |

---

## Toward a GNN-Based PFM

The current GNN approaches are **small** (1M–100M params) compared to transformer-based PFMs (385M–1.3B). The path to GNN-scale PFMs involves:

1. **Scale message-passing depth:** GNS shows $M$ steps ↔ longer-range interactions. Scaling $M$ and hidden width is the direct analog of scaling transformer depth/width.
2. **Combine global attention with local GNS:** Use a global attention layer for long-range elliptic coupling (Poisson, Stokes) and local GN steps for short-range contact/viscous interactions.
3. **Embed conservation laws from Level 3:** Dynami-CAL proves exact conservation is compatible with generalization and expressivity. This should be the default in any particle-based PFM.
4. **Equivariant mesh GNNs:** Apply SE(3)-equivariant message passing on adaptive meshes for continuum PDEs on irregular geometries — the single largest gap in the current PFM landscape.

**[AI Inference]:** The GNN-PB proposal in [[possible-architectures]] can be strengthened by combining: (a) GNS-style encode-process-decode as the backbone, (b) Dynami-CAL's antisymmetric frame for conservation, and (c) EquiformerV3's SwiGLU-S² activation for higher-order angular interactions. This produces a model that is simultaneously scalable, conservation-preserving, and expressively equivariant.

---

## Open Questions

1. Can GNNs scale to 1B+ parameters without over-smoothing? What regularization techniques (normalization, residual connections) are required?
2. Is SE(3) equivariance necessary for continuum PDEs, or is approximate equivariance (via data augmentation) sufficient at scale?
3. Can exact conservation laws (Level 3) be extended to continuum field quantities — not just per-particle momentum, but divergence-free velocity fields, energy-conserving pressure?
4. How does a particle-based GNN PFM compare to a continuum field transformer (Walrus) on the same physical system at the same computational budget?

---

## See Also

- [[pfm-architecture-approaches]] — where GNN-based simulation fits among 7 paradigms
- [[neural-operators]] — function-space learning (complementary paradigm)
- [[transformer-architectures]] — attention-based approaches for physics
- [[possible-architectures]] — GNN-PB architecture proposal
- [[gns-graph-network-simulators]] — foundational GNS framework
- [[dynami-cal-graphnet]] — conservation-constrained GNN
- [[equiformer-v3]] — SE(3)-equivariant Transformer for atomistic systems
- [[pc-deeponet-cfd]] — hard divergence-free constraint in neural operators
