# Action-Based Learning and Noether-Style Conservation Enforcement

**Type:** Architecture Concept / Proposal
**Status:** New synthesis page (2026-05-18)
**Related Concepts:** [[hamiltonian-message-passing]], [[equivariant-gnns]], [[world-models-physics-ai]], [[in-context-learning-physics]], [[physics-foundation-models]], [[possible-architectures]]
**Related Summaries:** [[lagrangian-neural-networks]], [[noether-networks]], [[hamiltonian-neural-networks]], [[dynami-cal-graphnet]], [[varmion-viscous-flows]]

---

## Motivation

The wiki's existing physics-encoding approaches enforce conservation laws **one at a time**:

- Dynami-CAL hardcodes Newton's 3rd law (linear + angular momentum) via antisymmetric edge frames
- HNN hardcodes energy conservation via Hamilton's equations
- PC-DeepONet hardcodes divergence-free flow via skew-symmetric Jacobians
- Soft-loss PINN approaches penalize PDE residuals without architectural enforcement

This is a combinatorial explosion: each new conservation law (charge, particle number, vorticity for 2D Euler, helicity for 3D MHD, fermion number, ...) needs its own bespoke architectural trick. A physics foundation model intended to cover every relevant regime cannot hardcode every relevant conservation law.

**Noether's theorem (1918) gives the underlying principle:** every continuous symmetry of the action functional corresponds to a conserved quantity. If the model is parameterized by an action $S[q]$, then enforcing a symmetry of $S$ automatically produces the corresponding conservation law. Rather than enforce each conservation law separately, enforce each symmetry once at the action level.

This page formalizes the resulting architecture: **action-based learning with Noether-style conservation discovery**. The architecture has three layers:

1. **Action functional** $S_\theta[q] = \int L_\theta(q, \dot{q})\, dt$ — parameterizes the dynamics.
2. **Explicit symmetries** of $L_\theta$ — give exact conservation of energy, momentum, angular momentum, etc. by construction.
3. **Meta-learned auxiliary conserved quantities** $g_\phi$ — discover regime-specific conservation laws not arising from explicit symmetries.

The combination subsumes Lagrangian Neural Networks ([[lagrangian-neural-networks]]) and Noether Networks ([[noether-networks]]) into a single end-to-end-trainable architecture.

---

## Noether's Theorem in Practical Form

**Theorem (Noether, 1918):** Let $S[q] = \int_{t_1}^{t_2} L(q, \dot{q}, t)\, dt$ be an action functional. Suppose $L$ is invariant under a continuous one-parameter transformation

$$q \to q + \epsilon \delta q(q, \dot{q}, t)$$

(up to a total time derivative). Then along solutions of the Euler-Lagrange equation, the quantity

$$Q = \frac{\partial L}{\partial \dot{q}} \cdot \delta q$$

is conserved: $\frac{dQ}{dt} = 0$.

### Examples

| Symmetry $\delta q$ | Noether quantity $Q$ |
|---|---|
| Time translation $t \to t + \epsilon$ | Energy $E = p\dot{q} - L$ |
| Spatial translation $q \to q + \epsilon$ | Linear momentum $p = \partial L / \partial \dot{q}$ |
| Spatial rotation $q \to R(\epsilon) q$ | Angular momentum $L = q \times p$ |
| Galilean boost | Center-of-mass position |
| Global phase $\psi \to e^{i\epsilon}\psi$ (complex fields) | Particle number / charge |

### Practical Consequence for Neural Networks

If a neural network parameterizes $L_\theta$ such that $L_\theta$ is exactly invariant under a symmetry, then the trajectories derived from $L_\theta$ exactly conserve the corresponding Noether quantity — **without any soft loss penalty**. The conservation is enforced by the architectural symmetry, not by training.

This is the action-level analog of Dynami-CAL's antisymmetric edge frame, but generalized: instead of enforcing one specific conservation law, we enforce one specific symmetry and get the corresponding conservation as a bonus.

---

## Architecture

### Layer 1: Action Functional

Parameterize the Lagrangian as a neural network $L_\theta(q, \dot{q})$. For many-body / continuum systems, decompose as a sum over local and pairwise contributions:

$$L_\theta(\{q_i, \dot{q}_i\}) = \sum_i T_i(q_i, \dot{q}_i) - \sum_i V^{\text{ext}}_i(q_i) - \sum_{(i,j)\in E} U_{ij}(q_i, q_j, \dot{q}_i, \dot{q}_j)$$

This is the **Lagrangian Graph Network** decomposition from [[lagrangian-neural-networks]] §"Lagrangian Graph Networks". Each $T_i, V_i, U_{ij}$ is a small MLP.

### Layer 2: Explicit Symmetry Enforcement

The functional forms of $T_i, V^{\text{ext}}_i, U_{ij}$ are restricted to manifest the desired symmetries:

| Symmetry | Functional restriction |
|---|---|
| Spatial translation $q \to q + a$ | $U_{ij}(q_i, q_j, \dot{q}_i, \dot{q}_j) = U_{ij}(q_i - q_j, \dot{q}_i, \dot{q}_j)$ — depends only on relative position |
| Spatial rotation $q \to R q$ | All vector inputs combined via rotation-invariant operations (inner products, norms, irreps tensor products) |
| Time translation | $L_\theta$ has no explicit $t$ dependence |
| Galilean boost $\dot{q} \to \dot{q} + v$ | $U_{ij}$ depends only on relative velocities $\dot{q}_i - \dot{q}_j$ |
| Permutation of identical particles | Per-node MLPs are shared (weight tying); aggregation is permutation-invariant |

These restrictions are familiar from equivariant network design (see [[equivariant-gnns]]). The action-based framing **rederives** equivariance from a more fundamental principle (action symmetry) and **extends** it to non-spatial symmetries (time translation, gauge symmetries, Galilean boosts).

By construction, $L_\theta$ obeys these symmetries exactly. By Noether, the corresponding quantities are exactly conserved along Euler-Lagrange trajectories.

### Layer 3: Dynamics via Euler-Lagrange

Derive trajectories by solving the Euler-Lagrange equation through automatic differentiation:

$$\ddot{q} = (\nabla_{\dot{q}}\nabla_{\dot{q}}^\top L_\theta)^{-1} \left[\nabla_q L_\theta - (\nabla_q \nabla_{\dot{q}}^\top L_\theta)\dot{q}\right]$$

(see [[lagrangian-neural-networks]] for derivation.) The mass-matrix inversion is the only non-autodiff operation; for typical mechanical systems it is $O(d^3)$ in the system dimension.

### Layer 4: Noether-Networks for Discovered Symmetries

The explicit symmetries above are the "known" Noether quantities. For physical regimes with additional approximate or unknown conserved structure (e.g., vorticity in 2D Euler, baryon number in particle physics, enstrophy in 2D NS), use [[noether-networks]]'s meta-learning framework:

- Train an auxiliary network $g_\phi: q \to \mathbb{R}^k$.
- At test time, tailor $\theta$ to minimize the conservation loss $\sum_t \|g_\phi(\hat{q}_{t+1}) - g_\phi(\hat{q}_t)\|^2$.
- In the outer loop, meta-learn $\phi$ so that this tailoring improves prediction.

This adds *soft* discovered conservation on top of the hard explicit-symmetry conservation. The architecture supports both simultaneously.

### Layer 5: Dissipation via Lagrange-d'Alembert

For dissipative systems, add a Rayleigh function $D_\eta(\dot{q})$ (or pairwise Rayleigh $D_{ij}$ for graph systems). The dynamics become the Lagrange-d'Alembert equation:

$$\frac{d}{dt}\frac{\partial L_\theta}{\partial \dot{q}} - \frac{\partial L_\theta}{\partial q} + \frac{\partial D_\eta}{\partial \dot{q}} = 0$$

This is the action-based analog of Dissipative HNN ([[dissipative-hamiltonian-neural-networks]]).

---

## Pipeline

```
Input: trajectory data {q_t}, optionally with derivative information

1. Define Lagrangian network L_θ(q, q̇):
   - Per-node MLPs (shared across nodes for permutation symmetry)
   - Per-edge MLP U_{ij}(r_ij, v_ij)  [translation/rotation/boost-invariant]
   - No explicit time dependence  [time-translation symmetry]

2. Optional: Rayleigh dissipation D_η(q̇) for dissipative systems

3. Optional: Noether-discovery network g_φ(q) for additional conservation laws

4. Compute dynamics:
   q̈ = (∇_q̇∇_q̇ᵀ L_θ)⁻¹ [∇_q L_θ - (∇_q ∇_q̇ᵀ L_θ) q̇] - ∇_q̇ D_η

5. Integrate (variational integrator for symplectic + Lagrange-d'Alembert):
   q^{t+Δt} = integrate(q^t, q̇^t, q̈^t)

6. Test-time tailoring (Noether):
   θ → θ - η ∇_θ Σ_t ||g_φ(q̂_{t+1}) - g_φ(q̂_t)||²

7. Training:
   Outer loop: minimize MSE on q̈ or q (full trajectory) + meta-train φ
```

---

## Conservation Properties (Summary)

| Conservation | Mechanism | Conditions |
|---|---|---|
| Linear momentum | Translation-invariant $U_{ij}$ | Exact, no dissipation |
| Angular momentum | Rotation-invariant $U_{ij}$ | Exact, no dissipation |
| Energy | Time-invariant $L_\theta$, no $D_\eta$ | Exact, conservative limit |
| Galilean covariance | Boost-invariant $U_{ij}$ | Exact |
| Particle number / charge | Phase-invariant $L_\theta$ | Exact for complex fields |
| Other (discovered) | Noether Networks $g_\phi$ | Approximate, meta-learned |
| Energy budget | $dH/dt = -2 D_\eta$ | Exact when dissipation is Rayleigh-type |

This is a substantially stronger conservation profile than any architecture in the wiki's current 5-architecture benchmark plan.

---

## Position Within the PFM Architecture Landscape

The action-based / Noether architecture is structurally distinct from the 5 architectures in [[possible-architectures]]:

| Architecture | Primary representation | Symmetries / Conservation |
|---|---|---|
| Arch 1: AR Transformer | Token sequence | None (data-driven) |
| Arch 2: Diffusion Backbone | Latent + score | None (data-driven + DPS guidance optional) |
| Arch 3: Neural Differentiator | Derivatives + integrator | None |
| Arch 4: Physics-Mamba | State-space recurrence | None |
| Arch 5: GNN-PB | Graph forces | Hard momentum (Dynami-CAL), soft energy bottleneck |
| **Action-based / Noether** | **Lagrangian functional** | **All Noether laws from explicit symmetries + meta-learned discovery** |

This is the **action-functional** paradigm — fundamentally different from "predict next state" or "predict derivatives." The model learns *what should be extremized*, not what should happen. Trajectories are derived by extremization.

For physical regimes where the action principle is the natural formulation (almost all of fundamental physics: mechanics, electromagnetism, GR, QFT, classical field theories), this architecture is the most principled choice.

---

## Strengths

1. **All Noether conservation laws automatic** from the corresponding symmetries.
2. **Field theory native.** Continuous Lagrangians for waves, electromagnetism, fluids extend the architecture directly.
3. **Relativistic regimes** handled correctly (LNN works where HNN fails).
4. **Constrained mechanics** via Lagrange multipliers added to $L_\theta$.
5. **Approximate conservation discovery** via Noether Networks.
6. **Symbolic interpretability** — the learned $L_\theta$ can be sparse-regressed to symbolic form (Cranmer et al.'s symbolic distillation, AI Feynman, PySR).
7. **Unified framework.** Action + Rayleigh + meta-learned conservation in one architecture.

---

## Limitations

1. **Mass-matrix inversion cost.** $O(d^3)$ in system dimension per forward pass. Mitigations: exploit sparsity (block-diagonal mass matrices for systems with locally-coupled coordinates), iterative solvers, second-order optimizers.

2. **Second-order autodiff.** Hessian of $L_\theta$ requires double backprop. ~2-3× more expensive than HNN's single backprop.

3. **Variational integrators are less standard.** Most ML / physics code uses Euler / RK4. Symplectic variational integrators (e.g., DEL — discrete Euler-Lagrange) are mathematically required for level-5 conservation but require careful implementation.

4. **Noether tailoring is computationally expensive.** The bilevel optimization adds test-time cost.

5. **No closed-form for the learned $L_\theta$.** Interpretability requires post-hoc symbolic distillation.

6. **Black-box symmetry enforcement.** Architectural symmetries must be chosen at design time — the framework does not automatically discover which symmetries to enforce. (Noether Networks discover *conserved quantities*, not *symmetries*; the two are dual but not identical.)

---

## Connections and Compositions

### With Hamiltonian Message Passing

The Hamiltonian formulation is the Legendre transform of the Lagrangian. For systems where $p = \partial L / \partial \dot{q}$ is invertible, the two formulations are equivalent. The Lagrangian is preferred when:

- Canonical momenta are awkward (relativistic, gauge theories)
- Constraints are present (rigid bodies, articulated systems)
- Fields are the natural variables

The Hamiltonian is preferred when:

- Symplectic integrators are critical (long-horizon orbital mechanics)
- The structure of $H = T + V$ is known a priori

A PFM might support both as parallel pathways with regime-conditioned routing — see the MoE inference in [[hamiltonian-message-passing]] §"Open Problems".

### With Multipole Hierarchy

The Lagrangian decomposes naturally into multipole contributions: at each scale $\ell$, the pairwise interaction $U^{(\ell)}_{IJ}$ between coarse clusters captures scale-$\ell$ physics. The Noether quantities (energy, momentum, angular momentum) decompose into contributions at each scale, with cross-scale exchange captured by the level-prolongation/restriction operators. This is the **action-based multi-scale architecture** — the cleanest version of the "physics GNN" line of reasoning.

### With Equivariant Networks

The rotation-invariant $U_{ij}$ requirement is equivalent to SE(3)-equivariant feature processing. EquiformerV3-style irreps with $L_{\max} > 0$ give a more expressive way to construct rotation-invariant $U_{ij}$ from rotation-equivariant features. The action framing **subsumes** equivariant networks: equivariance is a *consequence* of the symmetry requirement on $L_\theta$, not an independent property.

### With In-Context Learning

[[noether-networks]]' test-time tailoring is structurally identical to **in-context learning**: model behavior adapts at inference using unlabeled context. For a physics foundation model, the conservation-loss signal from $g_\phi$ is the natural "in-context" signal: the model should respect whatever quasi-conservation the test trajectory exhibits. This makes the action-based / Noether architecture especially natural for the ICL-style deployment of [[gphyt-physics-foundation-model]].

---

## Benchmark Adaptation

For the 1D Burgers' benchmark of [[possible-architectures]]:

- **Lagrangian density:** $\mathcal{L}(u, u_t, u_x) = \frac{1}{2}\phi_t^2 - V(u, u_x)$ where $u = \partial_x \phi$ (Clebsch-style potential). Parameterize $V$ as a learned scalar function of $u$ and $u_x$.
- **Translation symmetry:** $\mathcal{L}$ depends on $u_x$, not on $u$ at absolute position $x$ — gives conservation of $\int u\, dx$ (mass / first integral of the Burgers' equation).
- **Rayleigh dissipation:** $D = \frac{1}{2}\nu (\partial_x u)^2$ — viscous damping. With $\nu = 0$, recovers inviscid Burgers' (conservation of energy $\int u^2/2\, dx$).
- **Noether tailoring:** $g_\phi$ discovers any task-specific conservation (e.g., enstrophy for very-viscous regime).

Expected results: exact mass conservation; explicit dissipation rate $-2D = -\nu \int (\partial_x u)^2 dx$ matching the viscous decay of energy; comparable or better predictive accuracy than baselines with cleaner interpretability.

---

## Open Problems

1. **Automated symmetry discovery.** The architectural symmetry choices currently rely on human knowledge. Can we learn which symmetries to enforce? (Noether Networks discover *quantities*; discovering *symmetries* is dual but harder.) Recent work on symmetry-discovery networks (e.g., LieGAN, Yang et al. 2023) gives a path.

2. **Field-theoretic extensions.** The Lagrangian density $\mathcal{L}(\phi, \partial_\mu \phi)$ for continuous fields requires careful discretization. Discrete exterior calculus and finite-element exterior calculus give principled discretizations; integrating these with neural Lagrangians is open.

3. **Gauge theories.** For systems with local gauge symmetry (electromagnetism, Yang-Mills, hydrodynamics in Eulerian frame), the Lagrangian has redundant degrees of freedom that must be gauge-fixed or treated via Lagrange multipliers. The architectural choices for neural gauge theory are mostly unexplored.

4. **Stiff systems.** Systems with fast modes (e.g., elastic constraints, high-frequency oscillations) make the mass matrix near-singular, making the Euler-Lagrange acceleration calculation unstable. Constrained variational integrators (DEL with constraints) provide a path.

5. **Symbolic distillation pipeline.** Once $L_\theta$ is trained, applying sparse regression (AI Feynman, PySR) to recover a symbolic Lagrangian would give a fully interpretable model. The end-to-end pipeline from data → neural $L_\theta$ → symbolic $L$ is mostly a research opportunity.

---

## [AI Inference]

**[AI Inference]:** The deepest reason the action-based formulation should be the PFM architecture is that **fundamental physics is action-based**. Mechanics, electromagnetism, general relativity, quantum field theory — all are formulated via stationary-action principles. A neural network whose architecture matches this formulation can in principle represent any of these theories with appropriate Lagrangian forms. No other architecture in the wiki has this property. The "train once, deploy anywhere" PFM aspiration is strongest in the action-based framing because the language of fundamental physics is the language the architecture is written in.

**[AI Inference]:** The combination Lagrangian + Noether + Multipole + Antisymmetric attention (the four concept pages in this set) is the architectural composition that maximizes the wiki's stated desiderata. Lagrangian gives the symmetry/conservation foundation; Noether discovers additional approximate conservation; multipole gives multi-scale coupling; antisymmetric attention gives stable deep stacking. None of these have been combined in published work. A serious attempt at building such a model is a major research undertaking but is the natural endpoint of the wiki's current trajectory.

**[AI Inference]:** The action-based framing also suggests a principled approach to the [[world-models-physics-ai]] dichotomy. A Newtonian (causal, local) world model corresponds to a Lagrangian with explicit time evolution and short-range $U_{ij}$. A Keplerian (geometric, global) world model corresponds to an action principle treating the entire trajectory holistically (least-action principle with boundary conditions at both endpoints). The transition between the two is controlled by *how the Euler-Lagrange equation is solved* — initial-value problem (Newtonian) vs. boundary-value problem (Keplerian). The action-based architecture supports both regimes; the choice is a deployment decision, not an architectural one.

---

## Position on the Physics Encoding Spectrum

| Component | Level |
|---|---|
| Lagrangian functional with explicit symmetries | 5 (hard: Noether conservation from symmetry) |
| Rayleigh dissipation function | 5 (explicit dissipation budget) |
| Noether-discovered conservation $g_\phi$ | 3-4 (test-time soft enforcement) |
| Variational symplectic integrator | Required for level-5 in discrete time |

The full architecture sits at **level 5 for symmetry-derived conservation laws + level 3-4 for discovered conservation**. The discovered-conservation layer is intentionally soft — it captures *approximate* conservation that real-world data exhibits without forcing exact enforcement.

---

## Cross-Links

- [[lagrangian-neural-networks]] — foundation (LNN)
- [[noether-networks]] — foundation (Noether discovery)
- [[hamiltonian-neural-networks]] — sibling formulation (HNN)
- [[dissipative-hamiltonian-neural-networks]] — dissipation in Hamiltonian form
- [[dynami-cal-graphnet]] — alternative force-based momentum conservation
- [[varmion-viscous-flows]] — variational neural operator (related variational ML approach)
- [[hamiltonian-message-passing]] — Hamiltonian sibling concept page
- [[multiscale-hierarchical-gnn]] — multi-scale composition partner
- [[antisymmetric-signed-attention-transformer]] — attention-side sibling
- [[equivariant-gnns]] — equivariance taxonomy (subsumed by Noether framing)
- [[world-models-physics-ai]] — Newtonian vs. Keplerian
- [[in-context-learning-physics]] — ICL connection
- [[possible-architectures]] — broader benchmark plan
