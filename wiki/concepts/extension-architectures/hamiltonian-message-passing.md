# Hamiltonian + Dissipative Message Passing

**Type:** Architecture Concept / Proposal
**Status:** New synthesis page (2026-05-18)
**Related Concepts:** [[arch-gnn-physics-bottleneck]], [[action-based-noether-enforcement]], [[multiscale-hierarchical-gnn]], [[equivariant-gnns]], [[possible-architectures]]
**Related Summaries:** [[hamiltonian-neural-networks]], [[dissipative-hamiltonian-neural-networks]], [[lagrangian-neural-networks]], [[dynami-cal-graphnet]]

---

## Motivation

[[arch-gnn-physics-bottleneck]] (GNN-PB) uses a **soft energy bottleneck** to encode energy conservation as a level-4 inductive bias (architectural soft bias). The motivation is that real physics is dissipative, so hard energy conservation is wrong, and the bottleneck merely *routes* information through energy coordinates without enforcing exact conservation.

This is a conservative-friendly compromise but it leaves an obvious upgrade unrealized: **the conservative part of the dynamics can be hard-conserved by architecture, and the dissipative part can be learned separately**. This is exactly the Helmholtz decomposition that Dissipative HNNs ([[dissipative-hamiltonian-neural-networks]]) formalize for non-graph settings.

This page formalizes the graph-message-passing extension: a GNN whose message-passing kernel decomposes into a Hamiltonian (conservative) part parameterized as a potential function and a Rayleigh (dissipative) part parameterized as a velocity-dependent damping. Combined with Dynami-CAL's antisymmetric edge frame for momentum conservation, the resulting architecture has **exact conservation of linear momentum, angular momentum, and energy in the conservative limit**, with an explicit interpretable dissipation budget.

---

## The Three Constraints to Combine

| Constraint | Mechanism | Level |
|---|---|---|
| Linear momentum (Newton's 3rd) | Dynami-CAL antisymmetric edge frame: $\vec{F}_{ij} = -\vec{F}_{ji}$ | 5 (hard) |
| Angular momentum | Dynami-CAL angular torque + force-application point | 5 (hard) |
| Energy (conservative part) | Force from potential: $\vec{F}_{ij} = -\nabla V_{ij}$ + symplectic integrator | 5 (hard, conservative limit) |
| Energy dissipation | Rayleigh function $D_\phi$ acting on relative velocities | 5 (decomposed explicitly) |

The combination is **strictly stronger** than any of the three components alone and stronger than the GNN-PB's level-4 energy bottleneck.

---

## Architecture

### Per-Edge Parameterization

For each edge $(i, j)$, parameterize **two scalar functions**:

1. **Pairwise potential** $V_{ij}(\|r_{ij}\|, \theta) \in \mathbb{R}$: scalar function of the inter-node distance (and possibly node features) representing the conservative interaction energy.

2. **Pairwise Rayleigh dissipation** $D_{ij}(\vec{v}_{ij}, \|r_{ij}\|, \phi) \in \mathbb{R}_{\geq 0}$: non-negative scalar function of the relative velocity $\vec{v}_{ij} = \vec{v}_i - \vec{v}_j$ representing the dissipative power loss along this edge.

The total system Hamiltonian and Rayleigh function are sums over edges + per-node kinetic and external-potential terms:

$$H_\theta = \sum_i \frac{\|\vec{p}_i\|^2}{2 m_i} + \sum_i V^{\text{ext}}_i(q_i) + \sum_{(i,j) \in E} V_{ij}(\|r_{ij}\|)$$

$$D_\phi = \sum_{(i,j) \in E} D_{ij}(\vec{v}_{ij}, \|r_{ij}\|)$$

### Conservative Force from Potential

The conservative force on node $i$ from neighbor $j$ is derived by autodiff:

$$\vec{F}^{\text{cons}}_{ij} = -\nabla_{r_i} V_{ij}(\|r_{ij}\|) = -V'_{ij}(\|r_{ij}\|)\, \hat{r}_{ij}$$

where $\hat{r}_{ij}$ is the unit vector from $j$ to $i$. By construction:

1. **Newton's 3rd law:** $\vec{F}^{\text{cons}}_{ij} = -\vec{F}^{\text{cons}}_{ji}$ (because $V_{ij}$ depends symmetrically on $\|r_{ij}\| = \|r_{ji}\|$ and the gradient flips sign under $i \leftrightarrow j$).

2. **Conservativity:** $\nabla \times \vec{F}^{\text{cons}} = 0$ trivially (gradient of a scalar). This is **stronger than Dynami-CAL alone**: Dynami-CAL allows any antisymmetric force, including non-conservative ones. The potential parameterization restricts to conservative forces.

3. **Central:** the force is along $\hat{r}_{ij}$. For non-central forces (e.g., spin-orbit coupling, magnetic dipole-dipole), use a more general $V_{ij}(\vec{r}_{ij})$ that depends on the full relative vector — still conservative but no longer central.

### Dissipative Force from Rayleigh

The dissipative force on node $i$ from neighbor $j$ is:

$$\vec{F}^{\text{diss}}_{ij} = -\frac{\partial D_{ij}}{\partial \vec{v}_i}$$

For a quadratic Rayleigh function $D_{ij} = \frac{1}{2} c_{ij} \|\vec{v}_{ij}\|^2$ (linear damping), this gives $\vec{F}^{\text{diss}}_{ij} = -c_{ij} \vec{v}_{ij}$ — viscous drag along the relative velocity, antisymmetric under $i \leftrightarrow j$. More general Rayleigh functions give nonlinear damping (e.g., quadratic drag $\propto v^2$).

**Key property:** $D_\phi \geq 0$, so total energy $H_\theta$ decreases at rate $-2D_\phi \leq 0$. The dissipation budget is explicit and trackable.

### Total Force

$$\vec{F}_{ij} = \vec{F}^{\text{cons}}_{ij} + \vec{F}^{\text{diss}}_{ij}$$

Both contributions are antisymmetric under $i \leftrightarrow j$, so total linear momentum is exactly conserved (Newton's 3rd law applies to the sum).

### Symplectic Integrator at the Outer Loop

For the conservative part to *exactly* conserve energy under integration (not just in the continuous limit), a **symplectic integrator** is required:

- **Störmer-Verlet / leapfrog:** simplest symplectic integrator for separable Hamiltonians $H = T(p) + V(q)$.
- **Implicit midpoint:** symplectic, second-order, time-reversible.
- **Higher-order symplectic methods:** Yoshida composition, Forest-Ruth.

Standard Euler / RK4 are *not* symplectic and accumulate energy error linearly / quadratically. With symplectic integration, energy is bounded for all time (in the conservative limit) rather than drifting.

### Dynami-CAL Antisymmetric Frame for Angular Momentum

Angular momentum conservation requires the additional structure from Dynami-CAL ([[dynami-cal-graphnet]]): the antisymmetric edge-local orthonormal frame $\{\hat{e}_k^{ij}\}$ and the symmetric force-application point $\vec{x}_{0,ij}$.

In the Hamiltonian-message-passing setting, the force is fixed by the gradient of the potential — but the *torque* contribution from non-central effects needs explicit modeling. This is parameterized as an antisymmetric vector $\vec{A}_{ij} = -\vec{A}_{ji}$ projected on the edge-local frame, exactly as in Dynami-CAL.

---

## Full Architectural Pipeline

```
Input: nodes (positions, velocities, materials), edges (radius graph or hierarchy)

1. Per-edge Hamiltonian + Rayleigh parameterization
   For each edge (i, j):
     V_ij = MLP_V(||r_ij||, node features)         # scalar potential
     D_ij = softplus(MLP_D(v_ij, ||r_ij||))         # non-negative Rayleigh

2. Conservative force from autodiff
   F_cons_ij = -∂V_ij / ∂r_i

3. Dissipative force from autodiff
   F_diss_ij = -∂D_ij / ∂v_i

4. Antisymmetric edge frame (Dynami-CAL)
   {ê_k^ij}: orthonormal basis with ê_k^ij = -ê_k^ji

5. Angular momentum exchange (non-central effects)
   A_ij = Σ_k a_k(m_ij) ê_k^ij,  with a_k symmetric in (i,j) → A_ij = -A_ji
   x_{0,ij} = symmetric force-application point
   τ_ij = A_ij - (r_i - x_{0,ij}) × (F_cons_ij + F_diss_ij)

6. Aggregate to node-level acceleration
   F_i = Σ_{j∈N(i)} (F_cons_ij + F_diss_ij)
   τ_i = Σ_{j∈N(i)} τ_ij

7. Symplectic time integration (leapfrog or Störmer-Verlet)
   v_i^{t+Δt/2} = v_i^t + (Δt/2) · F_i / m_i           # half-kick
   r_i^{t+Δt}   = r_i^t + Δt · v_i^{t+Δt/2}           # drift
   v_i^{t+Δt}   = v_i^{t+Δt/2} + (Δt/2) · F_i^{t+Δt} / m_i  # half-kick

   (Or use a Lie-Trotter splitting: full conservative leapfrog + dissipative implicit step)
```

The conservative and dissipative integrations may be **split** in time: a full leapfrog step on the conservative dynamics, followed by an implicit step on the dissipative term. This preserves the symplectic property of the conservative part while handling stiff dissipation stably.

---

## Why This Beats GNN-PB's Energy Bottleneck

| Property | GNN-PB (energy bottleneck) | Hamiltonian + Dissipative MP |
|---|---|---|
| Linear momentum | Exact (Dynami-CAL) | Exact (Dynami-CAL) |
| Angular momentum | Exact (Dynami-CAL) | Exact (Dynami-CAL) |
| Energy (conservative limit) | Soft (level 4) | Exact (level 5) |
| Energy dissipation | Implicit, not explicit | Explicit Rayleigh $D_\phi$ |
| Energy budget tracking | No | Yes, via $dH/dt = -2D$ |
| Interpretability | Bottleneck projection | $V$ and $D$ separately inspectable |
| Symbolic discovery | Hard | Possible via sparse regression on $V, D$ |
| Reversibility | No | Yes (in conservative limit) |

The Hamiltonian + Dissipative message passing is **strictly stronger** for every desideratum.

---

## Composition with Other Wiki Architectures

### With Multipole Hierarchy ([[multiscale-hierarchical-gnn]])

Each level of the multipole hierarchy uses Hamiltonian + Rayleigh message passing. Coarse-level potentials $V^{(\ell)}_{IJ}$ represent inter-cluster interactions at scale $r_\ell$; coarse-level Rayleigh $D^{(\ell)}_{IJ}$ represents scale-$\ell$ dissipation (e.g., eddy viscosity at scale $\ell$ for turbulence).

This is the principled way to represent **multi-scale dissipation**, e.g., the Kolmogorov turbulent cascade where energy flows from large scales (small $D$) to small scales (large $D$, high dissipation).

### With Equivariant Irreps ([[equiformer-v3]])

The pairwise potential can take irreps-valued inputs: $V_{ij}(\vec{r}_{ij}, \vec{S}_i, \vec{S}_j, ...)$ where $\vec{S}_i$ are tensor features (e.g., spins, polarizations, orientations). Equivariance is preserved as long as $V_{ij}$ is a scalar function of equivariant inputs — which any sum of irreps tensor products satisfies.

### With Lagrangian Formulation ([[action-based-noether-enforcement]])

The Hamiltonian formulation has a Lagrangian counterpart. For systems where canonical momenta are awkward (relativistic, constrained, fields), use:

$$L_\theta = \sum_i T_i(q_i, \dot{q}_i) - \sum_i V^{\text{ext}}_i(q_i) - \sum_{(i,j) \in E} U_{ij}(q_i, q_j, \dot{q}_i, \dot{q}_j)$$

with Rayleigh dissipation handled via Lagrange-d'Alembert. This is the action-based dual of Hamiltonian message passing.

---

## Open Problems

1. **Convexity of Rayleigh.** For the Helmholtz decomposition to be unique (up to gauge), $D$ should be convex in $\dot{q}$. Enforcing convexity in a learned $D_\phi$ requires careful parameterization — e.g., input-convex neural networks (Amos et al. 2017).

2. **Coulomb friction / dry friction.** Coulomb friction $F = -\mu N \text{sgn}(\dot{q})$ is non-smooth at zero velocity and is not a Rayleigh-derivable damping. Handling it requires extending the framework with subgradient-style updates or complementarity conditions.

3. **Non-local dissipation.** Turbulent eddy viscosity is non-local in scale. The single-edge Rayleigh formulation must be extended to scale-coupled dissipation — natural in the multipole hierarchy setting.

4. **Symplectic integrator + adaptive timestep.** Variable-timestep symplectic integrators exist (e.g., Hairer's adaptive integrators) but are more complex than fixed-step leapfrog. The integration with $\Delta t$-as-input (from [[arch-physics-mamba]]) needs careful treatment.

5. **Compatibility with diffusion backbones.** Can a Hamiltonian + dissipative message passing serve as the score function $\nabla \log p_t$ in a diffusion model? The Helmholtz decomposition of the score has a known structure (Stein operator); whether the learned $H, D$ recover the right decomposition for physics distributions is an open question.

---

## Benchmark Adaptation

For the 1D Burgers' benchmark of [[possible-architectures]]:

- **Conservative part:** $V_{ij}$ represents the conservative advection (in 1D, this is the nonlinear $u \partial_x u$ term). Parameterize as a learned function of $u_i$ and $\|x_i - x_j\|$.
- **Dissipative part:** $D_{ij}$ represents the viscous diffusion $\nu \partial_{xx} u$. Parameterize as $D_{ij} = \frac{1}{2} \nu (u_i - u_j)^2 / \|x_i - x_j\|^2$ as an initialization, learned thereafter.
- **Symplectic integration:** leapfrog with $\Delta t$ chosen for stability.

Expected behavior: exact conservation of $\int u\, dx$ (mass, level-5) and explicit dissipation rate matching the viscous decay. Performance on Burgers' should match or beat GNN-PB with cleaner interpretation.

---

## [AI Inference]

**[AI Inference]:** The Hamiltonian + Dissipative split is the GNN realization of **port-Hamiltonian systems** (van der Schaft 2000): open systems with conservative dynamics, dissipative dissipation, and external ports. For PFM applications with external forcing (boundary conditions, body forces, time-varying parameters), the port-Hamiltonian extension is the natural generalization — and would integrate cleanly with the architecture above by adding an external port $g(x) u$ to the dynamics.

**[AI Inference]:** The combination of Hamiltonian message passing + Noether Networks ([[noether-networks]]) gives the strongest known conservation framework. The Hamiltonian backbone provides exact conservation of energy via $H_\theta$, momentum via Dynami-CAL, and angular momentum via the symmetric force-application point. Noether Networks then meta-learn additional approximately-conserved quantities $g_\phi$ on top of these — capturing system-specific structure (e.g., particle number, vorticity, helicity) without hardcoding them. This is the unifying architectural pattern that subsumes the wiki's current best designs.

**[AI Inference]:** For a PFM trained on diverse physical regimes, the choice between Hamiltonian and Lagrangian formulations should be **regime-dependent**. Classical mechanics with simple Cartesian coordinates is fine in either form; relativistic and field-theoretic regimes prefer Lagrangian (no canonical $p$); strongly dissipative regimes prefer the explicit H + D split. A regime-conditioned mixture (e.g., MoE-routed between Hamiltonian and Lagrangian experts) could give the best of both — connecting to [[mixture-of-experts]].

---

## Position on the Physics Encoding Spectrum

| Component | Level |
|---|---|
| Hamiltonian-derived conservative force | 5 (hard energy conservation in conservative limit) |
| Rayleigh-derived dissipative force | 5 (explicit, interpretable dissipation budget) |
| Dynami-CAL antisymmetric frame | 5 (hard momentum conservation) |
| Symplectic integrator | Required for level-5 in discrete time |

The full architecture sits at **level 5 across all three primary conservation laws** — the strongest position on the physics encoding spectrum currently achievable for a GNN simulator. The only level-5+ aspiration beyond this would be exact gauge / symmetry preservation for field theories, which requires the Lagrangian extension (see [[action-based-noether-enforcement]]).

---

## Cross-Links

- [[hamiltonian-neural-networks]] — rigorous HNN foundation
- [[dissipative-hamiltonian-neural-networks]] — Helmholtz decomposition foundation
- [[lagrangian-neural-networks]] — Lagrangian alternative
- [[dynami-cal-graphnet]] — momentum conservation backbone
- [[arch-gnn-physics-bottleneck]] — soft-energy-bottleneck predecessor (this concept upgrades)
- [[multiscale-hierarchical-gnn]] — multi-scale composition partner
- [[action-based-noether-enforcement]] — Lagrangian sibling
- [[antisymmetric-signed-attention-transformer]] — attention-side sibling
- [[equivariant-gnns]] — physics inductive bias taxonomy
- [[possible-architectures]] — broader benchmark plan
