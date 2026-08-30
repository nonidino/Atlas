# Dynami-CAL GraphNet: Physics-Informed GNN Conserving Linear and Angular Momentum

**Source:** Sharma & Fink, EPFL, arXiv 2501.07373v2  
**File:** `new/Dynami-CAL GraphNet...md`  
**Related Concepts:** [[equivariant-gnns]], [[pfm-architecture-approaches]], [[possible-architectures]], [[neural-surrogates]]  
**Related Summaries:** [[gns-graph-network-simulators]], [[pc-deeponet-cfd]], [[varmion-viscous-flows]]

---

## Overview

Dynami-CAL GraphNet is a physics-informed GNN for six-degree-of-freedom (6-DoF) multi-body dynamics. It is the first GNN architecture to **exactly** conserve both linear and angular momentum through hard architectural constraints — without requiring these as loss penalties. The core insight is that by constructing an **antisymmetric edge-local reference frame** and decoding forces/torques through it, Newton's third law ($\vec{F}_{ij} = -\vec{F}_{ji}$) and angular momentum conservation are enforced by algebraic construction, not training.

Benchmarked on 3D granular collisions, human motion capture, and molecular dynamics — outperforming GNS and equivariant GNNs (EGNN, GMN, ClofNet) across all tasks.

---

## Key Equations

### Edge-Local Reference Frame
Each edge $(i \to j)$ is assigned an orthonormal basis $\{\hat{e}_1, \hat{e}_2, \hat{e}_3\}_{ij}$ that is:
- **SO(3)-equivariant** — rotates correctly under coordinate changes
- **T(3)-invariant** — independent of absolute position
- **Antisymmetric under node interchange**: $\hat{e}_k^{ij} = -\hat{e}_k^{ji}$ for all $k$

This antisymmetry is the structural guarantee. Scalar edge embeddings $m_{ij}$ are constructed to be **symmetric** under node interchange: $m_{ij} = m_{ji}$.

### Force Vectorization (Linear Momentum Conservation)
$$\vec{F}_{ij} = \sum_{k=1}^{3} f_k(m_{ij}) \cdot \hat{e}_k^{ij}$$

Because $f_k(m_{ij}) = f_k(m_{ji})$ (symmetric coefficients) and $\hat{e}_k^{ij} = -\hat{e}_k^{ji}$ (antisymmetric basis):
$$\vec{F}_{ij} = -\vec{F}_{ji} \quad \Rightarrow \quad \text{linear momentum conserved at every edge}$$

### Angular Momentum Conservation
Total angular momentum exchange per edge is decoded as:
$$\vec{A}_{ij} = -\vec{A}_{ji}$$

The spin torque (affecting angular velocity $\vec{\omega}_i$) is isolated by subtracting the orbital contribution:
$$\boldsymbol{\tau}_{ij} = \vec{A}_{ij} - (\vec{r}_i - \vec{x}_{0,ij}) \times \vec{F}_{ij}$$

where $\vec{x}_{0,ij} = \vec{x}_{0,ji}$ is a predicted, shared force application point. This ensures both spin and orbital contributions to angular momentum are globally conserved across all edges.

### Spatiotemporal Message Passing with Edge Memory
Iterates $K$ sub-steps within each prediction interval. At each sub-step, edge embeddings are updated from current node states **and** the previous edge embedding (Edge Memory):
$$\mathbf{e}_{ij}^{(k+1)} = \phi\!\left(\mathbf{v}_i^{(k)}, \mathbf{v}_j^{(k)},\ \mathbf{e}_{ij}^{(k)}\right)$$

This gives the model spatiotemporal memory — capturing both spatial neighbor interactions and temporal dynamics within a single prediction step.

### Ghost-Node Boundary Modeling
Walls are modeled by reflecting each physical node $i$ across the wall's outward normal to create a ghost node $i'$. Edges connect $i$ to $i'$ only within a distance threshold, making wall interactions structurally identical to body-body interactions. For a stationary wall, the ghost node inherits zero velocity.

---

## Key Results

**Granular 6-DoF Collisions (60 spheres, 500-step rollout):**
- Dynami-CAL GraphNet retains all particles and accurately tracks kinetic energy decay, linear momentum, and angular momentum evolution
- GNS diverges in the extrapolation regime (3× training kinetic energy), failing to confine particles
- EGNN, GMN, and ClofNet all underperform GNS despite their equivariance — GNS's expressiveness for impulse-driven dynamics outweighs equivariant inductive biases without conservation

**Oblique Collision Conservation Test (2-sphere closed system):**
- Dynami-CAL GraphNet conserves total linear and angular momentum while correctly predicting kinetic energy dissipation
- GNS violates both conservation laws

**Extrapolation to Rotating Hopper Mixer:**
- Trained on 60 spheres in stationary box; tested on 2,073 spheres in a rotating cylindrical hopper with curved walls and non-uniform rotational acceleration
- Demonstrates generalization to 34× more particles, unseen boundary geometry, and dynamic boundary conditions

---

## Relevance to PFM Goal

Dynami-CAL GraphNet is **level 5** on the physics encoding spectrum (hard architectural constraints) applied to GNNs. Unlike PC-DeepONet ([[pc-deeponet-cfd]]) which enforces divergence-free constraints for a specific PDE, Dynami-CAL's conservation constraints are universal — they hold for any system obeying Newton's laws, regardless of interaction type (elastic, inelastic, frictional, dissipative).

This is a critical advance for the GNN-PB architecture proposal ([[possible-architectures]]): it provides an existence proof that exact conservation can be built into a GNN without sacrificing generalization or expressivity.

**Key contrast with VarMiON ([[varmion-viscous-flows]]):** VarMiON derives its architecture from the weak-form variational structure of a specific PDE (Stokes). Dynami-CAL derives constraints from universal mechanical laws (Newton's third law). The latter is more general.

---

## **[AI Inference]**

**[AI Inference]:** The antisymmetric edge-local frame design is mathematically analogous to the skew-symmetric Jacobian trick in PC-DeepONet ($\mathbf{v} = \nabla \times (J - J^\top)$) — both use algebraic antisymmetry to encode a conservation law. The GNN version is more powerful because it applies pairwise, making the conservation local and composable rather than requiring a global divergence-free field.

**[AI Inference]:** Edge Memory (latent state carried across sub-steps) is the GNN analog of the KV-cache in transformer inference. This raises the question of whether a transformer-GNN hybrid — using attention to set global context and GNS-style local message passing with edge memory for fine-grained particle interactions — could achieve both global pressure propagation (a weakness of pure GNNs) and exact local conservation (a weakness of transformers).

**[AI Inference]:** The ghost-node boundary method is remarkably elegant: it makes wall interactions algebraically identical to particle-particle interactions, requiring no special-case code. This suggests a general design principle: whenever a physical constraint (boundary condition, symmetry, external force) can be recast as a fictitious particle with defined properties, it can be handled by the same message-passing mechanism — significantly simplifying the architecture for multi-physics systems.

---

## See Also

- [[gns-graph-network-simulators]] — foundational GNS framework this extends
- [[pc-deeponet-cfd]] — analogous hard constraint approach for continuum PDEs
- [[varmion-viscous-flows]] — variational-form architectural constraints
- [[equivariant-gnns]] — SE(3) equivariance and conservation in GNNs
- [[possible-architectures]] — GNN-PB proposal; Dynami-CAL provides empirical grounding
- [[pfm-architecture-approaches]] — where particle-based GNN simulation fits in the landscape
