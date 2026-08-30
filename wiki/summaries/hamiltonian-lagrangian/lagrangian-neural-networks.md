# Lagrangian Neural Networks

**Source:** arXiv:2003.04630 (ICLR 2020 Workshop on Integration of Deep Neural Models and Differential Equations)
**Authors:** Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, Shirley Ho
**Affiliation:** Princeton, Google Brain, Google Research, DeepMind, Flatiron Institute
**Related Concepts:** [[action-based-noether-enforcement]], [[hamiltonian-message-passing]], [[equivariant-gnns]], [[physics-foundation-models]]
**Related Summaries:** [[hamiltonian-neural-networks]], [[noether-networks]], [[dynami-cal-graphnet]], [[varmion-viscous-flows]]

---

## Overview

Lagrangian Neural Networks (LNNs) parameterize the Lagrangian $L_\theta(q, \dot{q})$ — a scalar function of generalized coordinates and their time derivatives — with a neural network. The dynamics are recovered by applying the **Euler-Lagrange equation** through automatic differentiation. This is a structurally cleaner alternative to Hamiltonian Neural Networks because it works directly on the observed generalized coordinates without requiring canonical momenta. The Lagrangian formulation also extends naturally to relativistic systems, constrained mechanics, and classical field theory — domains where HNN runs into the canonical-coordinate problem.

For physics foundation models, LNN is the foundational architecture for **action-based learning** — predicting trajectories by extremizing a learned action functional. This is the closest neural-network analog of how fundamental physics is actually formulated, and the natural setting for embedding Noether's theorem and gauge symmetries.

---

## The Lagrangian Formulation

The action of a classical system is the time-integral of its Lagrangian:

$$S[q] = \int_{t_1}^{t_2} L(q(t), \dot{q}(t), t)\, dt$$

**Hamilton's principle of stationary action:** physical trajectories are stationary points of $S$. The Euler-Lagrange equation is the necessary condition for stationarity:

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0$$

For typical mechanical systems, $L = T(q, \dot{q}) - V(q)$ (kinetic minus potential), and the Euler-Lagrange equation reproduces Newton's second law. But $L$ can be much more general — relativistic dynamics, electromagnetic minimal coupling $L = \frac{1}{2}m\dot{q}^2 + q A(q)\cdot \dot{q} - \phi(q)$, constrained systems via Lagrange multipliers $L + \lambda \cdot \text{constraint}$, classical fields $\mathcal{L}(\phi, \partial_\mu \phi)$ — all fit naturally.

**Crucially, $L$ is a function of $(q, \dot{q})$.** The observed generalized coordinates and their direct time derivatives. No canonical momentum required.

---

## LNN Method

### Solving the Euler-Lagrange Equation for $\ddot{q}$

Expanding the total time derivative in the Euler-Lagrange equation:

$$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}} = \left(\nabla_{\dot{q}} \nabla_{\dot{q}}^\top L\right) \ddot{q} + \left(\nabla_q \nabla_{\dot{q}}^\top L\right) \dot{q}$$

Setting this equal to $\nabla_q L$ and solving for $\ddot{q}$:

$$\boxed{\ddot{q} = \left(\nabla_{\dot{q}} \nabla_{\dot{q}}^\top L\right)^{-1} \left[ \nabla_q L - \left(\nabla_q \nabla_{\dot{q}}^\top L\right) \dot{q} \right]}$$

This is the closed-form expression for accelerations given the Lagrangian. All four ingredients on the right are computed by automatic differentiation:

1. $\nabla_q L$ — first derivative of $L_\theta$ w.r.t. $q$
2. $\nabla_{\dot{q}} L$ — first derivative w.r.t. $\dot{q}$
3. $\nabla_q \nabla_{\dot{q}}^\top L$ — mixed Hessian (used in time derivative)
4. $\nabla_{\dot{q}} \nabla_{\dot{q}}^\top L$ — Hessian w.r.t. $\dot{q}$, the **mass matrix**

The mass matrix is positive-definite for physically reasonable Lagrangians and well-conditioned for typical mechanical systems.

### Training Loss

Standard L2 on accelerations (or on first-derivative observations if available):

$$\mathcal{L}_{\text{LNN}}(\theta) = \|\ddot{q}_{\text{obs}} - \ddot{q}_\theta\|^2$$

Trajectories can be rolled out by integrating $\ddot{q}_\theta$ with any standard ODE integrator.

---

## Why Lagrangian Over Hamiltonian

| Scenario | HNN | LNN |
|---|---|---|
| Cartesian mass-spring | Works | Works |
| Relativistic free particle | Fails ($p$ not directly observable) | Works |
| Constrained mechanics (holonomic) | Needs reduction to constraint manifold | Works via Lagrange multipliers |
| Velocity-dependent potentials (EM) | Awkward | Direct |
| Classical fields (Klein-Gordon, EM, GR) | Limited | Native formulation |
| Gauge theories | Difficult | Native |

The Lagrangian is closer to fundamental physics; the Hamiltonian is closer to numerical practice. For a PFM intended to cover diverse physical regimes, the Lagrangian is the more flexible choice.

---

## Empirical Results

The paper demonstrates LNNs on:

1. **Double pendulum** (nonlinear, non-separable): LNN conserves energy over long horizons; vanilla MLP baseline drifts. HNN also works here, but requires care in identifying canonical $p$ from data.

2. **Relativistic particle** ($L = -mc^2\sqrt{1 - v^2/c^2}$): LNN correctly learns the relativistic Lagrangian and conserves relativistic energy. HNN **fails** because the canonical momentum $p = mv/\sqrt{1-v^2/c^2}$ is not directly observable from simulated $(q, v)$ trajectories — they would have to be transformed first.

3. **Wave equation on a string** (Lagrangian Graph Network extension): A discretized field theory where each node carries a Lagrangian density contribution; the total Lagrangian is the sum. Demonstrates that LNN extends to spatially extended systems via graph architecture.

In every case, LNN preserves energy where unconstrained baselines drift, with comparable or better accuracy.

---

## Strengths

1. **No canonical coordinates required** — works with whatever generalized coordinates are convenient/observable.
2. **Energy conservation by structure** in the absence of dissipation.
3. **Direct support for constraints** via Lagrange multipliers.
4. **Field theory native** — Lagrangian densities $\mathcal{L}(\phi, \partial_\mu \phi)$ are the natural starting point for classical field theory; LNN extends directly to continuum systems.
5. **Noether's theorem applies directly** — symmetries of $L_\theta$ yield conserved quantities automatically (see [[noether-networks]] for the meta-learned variant).

---

## Limitations

1. **Mass matrix inversion required.** Cost $O(d^3)$ per forward pass for $d$ generalized coordinates. For high-DoF systems this becomes expensive; iterative solvers (CG) or factorizations help.

2. **Second-order autodiff.** Computing Hessians of $L_\theta$ requires double backprop — slower and more memory-intensive than HNN's single backprop.

3. **Dissipation not handled in vanilla form.** Pure LNN assumes conservative dynamics. Lagrange-d'Alembert with Rayleigh dissipation extends LNN to dissipative systems analogously to D-HNN.

4. **Black-box Lagrangian.** The learned $L_\theta$ is opaque. Recovering symbolic form requires sparse regression or symbolic distillation (e.g., AI Feynman, PySR, the Cranmer follow-up work on symbolic models).

5. **No graph structure in vanilla form.** For many-body systems, Lagrangian Graph Networks (LGN, Cranmer et al. 2020 follow-up) are needed.

---

## Connection to Variational Methods in Physics

The LNN is the neural analog of the most general formulation of classical physics: the **principle of stationary action**. Several PFM-relevant architectures already use variational formulations:

- **VarMiON** ([[varmion-viscous-flows]]) — branch network derived from the variational (weak) form of Stokes equations. Same family.
- **Variational integrators** — symplectic integrators derived from a discrete Lagrangian. The natural pairing for LNN at the time-stepping level.
- **Galerkin methods** — variational discretization of PDEs. Conceptually consistent with action-based learning.

This puts LNN at the center of a growing literature on **variational machine learning for physics**.

---

## Lagrangian Graph Networks (LGN)

The paper introduces (and follow-up work extends) Lagrangian Graph Networks for many-body systems. The total Lagrangian is expressed as a sum:

$$L_\theta(\{q_i, \dot{q}_i\}) = \sum_i T_i(q_i, \dot{q}_i) - \sum_i V_i(q_i) - \sum_{(i,j) \in E} U_{ij}(q_i, q_j, \dot{q}_i, \dot{q}_j)$$

where $T_i, V_i$ are per-node kinetic and potential contributions and $U_{ij}$ are pairwise interaction Lagrangians on edges. Each term is a small MLP. The Euler-Lagrange equation is solved on the full system, giving per-node accelerations.

This is the action-based analog of Dynami-CAL: where Dynami-CAL hardcodes Newton's 3rd law via antisymmetric edge frames, LGN derives Newton's 3rd law from translation invariance of the pairwise interaction Lagrangian. The latter is more general (handles non-central forces, velocity-dependent interactions) but provides only soft conservation (requires the model to respect translation invariance, which is enforced by the choice of $U_{ij}$ inputs).

---

## [AI Inference]

**[AI Inference]:** The Lagrangian formulation is the natural setting for **Noether-driven** physics foundation models. Combining LNN with [[noether-networks]] gives a complete pipeline: LNN parameterizes the action, Noether Networks meta-learn auxiliary conserved quantities $g_\phi$ on top, and tailoring at test time adapts $\theta$ to enforce conservation of $g_\phi$. This combines hard structural conservation (via Lagrangian symmetries) with soft discovered conservation (via meta-learning) — the most expressive symmetry/conservation framework currently available.

**[AI Inference]:** For a physics foundation model intended to span classical mechanics, field theory, and relativistic regimes uniformly, the Lagrangian is the natural unification target. A LNN-based PFM would represent the kinematic regime (Newton: $L = \frac{1}{2}m\dot{q}^2 - V$), the field regime ($\mathcal{L}(\phi, \partial_\mu \phi)$ for waves, electromagnetism), and the relativistic regime ($L = -mc^2\sqrt{1-v^2/c^2} - V$) within the **same architecture**, just with different learned functionals. This is a stronger unification than is provided by any current transformer or GNN PFM.

**[AI Inference]:** The mass matrix inversion at each forward pass is a hidden architectural choice. For typical mechanical Lagrangians the mass matrix is sparse (each coordinate's mass entry depends on a few nearby coordinates) — exploiting this sparsity gives $O(d)$ rather than $O(d^3)$ inversion cost. This is unaddressed in the original paper and is a substantial scalability lever for large-system LNNs.

---

## Cross-Links

- [[action-based-noether-enforcement]] — action-based / Noether enforcement concept page
- [[hamiltonian-message-passing]] — sibling page on Hamiltonian formulation
- [[hamiltonian-neural-networks]] — HNN base
- [[noether-networks]] — auxiliary conserved-quantity discovery
- [[dynami-cal-graphnet]] — force-based hard conservation; the dual of LGN
- [[varmion-viscous-flows]] — variational method for fluids
- [[equivariant-gnns]] — physics inductive bias taxonomy
