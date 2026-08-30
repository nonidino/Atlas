# Hamiltonian Neural Networks

**Source:** arXiv:1906.01563 (NeurIPS 2019)
**Authors:** Sam Greydanus, Misko Dzamba, Jason Yosinski
**Affiliation:** Google Brain, Uber AI Labs
**Related Concepts:** [[hamiltonian-message-passing]], [[equivariant-gnns]], [[physics-foundation-models]], [[pfm-architecture-approaches]]
**Related Summaries:** [[lagrangian-neural-networks]], [[dissipative-hamiltonian-neural-networks]], [[dynami-cal-graphnet]]

---

## Overview

Hamiltonian Neural Networks (HNNs) parameterize the scalar Hamiltonian $H_\theta(q, p)$ of a physical system with a neural network and recover dynamics via Hamilton's equations through automatic differentiation. The architectural trick is that the model never predicts $\dot{q}, \dot{p}$ directly — it predicts a scalar $H_\theta$, and the dynamics are derived. This embeds **exact energy conservation** and **time-reversibility** into the learned dynamics, properties no unstructured neural ODE provides.

HNN is the canonical paper in a now-large literature on physics-structured neural dynamics, alongside Lagrangian Neural Networks (Cranmer et al. 2020), Symplectic ODE-Net (Zhong et al. 2020), Dissipative HNN (Sosanya & Greydanus 2022), and Hamiltonian Graph Networks (Sanchez-Gonzalez et al. 2019).

---

## The Hamiltonian Formulation

A classical Hamiltonian system is specified by a scalar function $H: \mathbb{R}^{2d} \to \mathbb{R}$ on phase space $(q, p)$, with dynamics

$$\dot{q} = \frac{\partial H}{\partial p}, \qquad \dot{p} = -\frac{\partial H}{\partial q}$$

Three structural consequences:

1. **Energy conservation:** $\frac{dH}{dt} = \partial_q H \cdot \dot{q} + \partial_p H \cdot \dot{p} = \partial_q H \cdot \partial_p H - \partial_p H \cdot \partial_q H = 0$. Exact, along any trajectory.
2. **Symplecticity:** Phase-space 2-form $\omega = dq \wedge dp$ is preserved; phase-space volume is preserved (Liouville).
3. **Time-reversibility:** $(q, p, t) \to (q, -p, -t)$ leaves the equations invariant.

The Hamiltonian is the most fundamental representation of a conservative system; for a typical mechanical system, $H = T(p) + V(q)$ (kinetic + potential energy), but many systems have more complex $H$ (relativistic kinematics, coupled coordinates, etc.).

---

## HNN Method

### Architecture

A neural network $H_\theta: \mathbb{R}^{2d} \to \mathbb{R}$ outputs a scalar. The predicted dynamics are recovered by autodiff:

$$\hat{\dot{q}} = \frac{\partial H_\theta}{\partial p}, \qquad \hat{\dot{p}} = -\frac{\partial H_\theta}{\partial q}$$

### Training Loss

L2 error between predicted and observed time derivatives:

$$\mathcal{L}_{\text{HNN}}(\theta) = \left\|\dot{q}_{\text{obs}} - \frac{\partial H_\theta}{\partial p}\right\|_2^2 + \left\|\dot{p}_{\text{obs}} + \frac{\partial H_\theta}{\partial q}\right\|_2^2$$

Time derivatives come from finite differences or direct measurement.

**Crucially, no supervision on $H$ itself.** The model never sees a label "this is the energy"; it just learns a scalar whose autodiff gradients match the dynamics.

### Inductive Bias Strength

HNN constrains the predicted vector field to lie on a $d$-dimensional manifold of "Hamiltonian vector fields" within the $2d$-dimensional space of general vector fields. This is a substantial restriction:

| Approach | Vector-field dimension | Energy conserved? |
|---|---|---|
| Vanilla MLP predicting $\dot{q}, \dot{p}$ | $2d$ (unrestricted) | No |
| HNN ($\nabla H_\theta$) | $d$ (gradient of scalar) | Yes (exactly) |

The reduction in expressivity is roughly a factor of 2 — but in exchange the model is structurally correct for the class of systems it targets.

---

## Empirical Results

The original paper evaluates HNN on:

1. **Mass-spring ($H = \frac{1}{2}(p^2 + q^2)$):** HNN recovers $H$ exactly up to additive constant; baseline drifts in total energy on long rollouts.
2. **Ideal pendulum ($H = 2mg\ell(1 - \cos q) + \frac{p^2}{2m\ell^2}$):** HNN trajectories close periodically; baseline trajectories spiral outward over time.
3. **Two-body / three-body gravitational system:** HNN maintains bound orbits over long horizons; baseline gradually unbinds and ejects.
4. **Pixel pendulum:** With an autoencoder front-end, HNN learns a *latent* Hamiltonian whose value is conserved in latent space — the first demonstration of learning Hamiltonians from raw observations rather than from $(q, p)$ measurements.

In every case: faster training, better long-horizon rollout accuracy, exact energy conservation.

---

## Strengths

1. **Exact energy conservation** (continuous-time) or to integrator accuracy (discrete-time). Unmatched by any soft-loss-based approach.
2. **Time-reversibility** is automatic.
3. **Symplectic structure** — phase-space volume preservation — emerges naturally.
4. **Faster training, fewer parameters** for the same accuracy on conservative systems.
5. **Composable.** HNNs work as inner modules: Hamiltonian Graph Networks (HGN) use an HNN as the local interaction module of a GNN.

---

## Limitations and Extensions

### 1. Requires canonical coordinates

The Hamiltonian formulation assumes phase space $(q, p)$ where $p$ is the canonical momentum. For mechanical systems with simple kinetic energy, $p = m\dot{q}$ and this is fine. For relativistic, constrained, or field-theoretic systems, the canonical momentum can be difficult to obtain.

**Extension:** Lagrangian Neural Networks ([[lagrangian-neural-networks]]) parameterize $L(q, \dot{q})$ instead and derive dynamics via the Euler-Lagrange equation — no canonical coordinates required.

### 2. Conservative systems only

Friction, viscosity, inelastic collisions all dissipate energy and pure HNNs cannot represent them.

**Extension:** Dissipative HNNs ([[dissipative-hamiltonian-neural-networks]]) parameterize $H$ and a Rayleigh dissipation function $D$ jointly, giving a Helmholtz decomposition into conservative + dissipative parts.

### 3. Time-independent Hamiltonian

The basic formulation assumes $H = H(q, p)$ without explicit time-dependence. External time-varying forcing (e.g., a driven pendulum) is not naturally accommodated.

**Extension:** Port-Hamiltonian neural networks allow external "ports" through which energy can flow in/out.

### 4. No graph structure in vanilla HNN

The original HNN is for a single system with global state $(q, p)$. For many-body systems, this is impractical.

**Extension:** Hamiltonian Graph Networks (Sanchez-Gonzalez et al. 2019, arXiv:1909.12790) combine HNN with GNS-style message passing — the model predicts a per-system $H$ from particle configurations.

---

## Relevance to Physics Foundation Models

1. **Hard energy conservation in the conservative limit.** For physical regimes where energy is genuinely conserved (orbital mechanics, molecular dynamics in isolated systems, ideal fluids), HNN gives exact conservation. This is **stronger than the soft energy bottleneck** of GNN-PB ([[arch-gnn-physics-bottleneck]]).

2. **Compatible with Dynami-CAL.** A Hamiltonian Graph Network that uses Dynami-CAL antisymmetric edge frames gets:
   - Exact linear + angular momentum conservation (from Dynami-CAL)
   - Exact energy conservation (from HNN, in the conservative limit)
   This combination achieves level 5 of the physics encoding spectrum across all three primary conservation laws — currently unrealized in the wiki literature.

3. **Foundation for Hamiltonian + dissipative split.** Combined with Dissipative HNN, the architecture supports realistic dissipative physics while preserving the conservative-part structure. See [[hamiltonian-message-passing]].

4. **Symplectic integrators required at the outer loop.** To preserve energy conservation under discretization, a symplectic integrator (leapfrog, Störmer-Verlet, implicit midpoint) should be used instead of Euler or RK4. This complements the neural-differentiator paradigm of [[arch-neural-differentiator]].

---

## [AI Inference]

**[AI Inference]:** The HNN's scalar parameterization $H_\theta$ is structurally equivalent to a learned **potential function** in the GNN-PB setting. If we parameterize the pairwise interaction as $V_{ij}(\|r_{ij}\|)$ — a scalar — and compute the force as $\vec{F}_{ij} = -\nabla_{r_i} V_{ij}$, we get both Newton's 3rd law (antisymmetry, $\vec{F}_{ij} = -\vec{F}_{ji}$) and exact conservativity (curl-free force field) by construction. This is the GNN translation of the HNN idea, and it is more constrained than the Dynami-CAL antisymmetric edge frame (which allows non-conservative forces). For the conservative part of the dynamics, the potential parameterization is the right choice.

**[AI Inference]:** The HNN's energy conservation can be understood as a special case of [[noether-networks]]'s framework, with the conserved quantity being the Hamiltonian itself and the symmetry being time-translation. The Noether Network framework discovers this conserved quantity automatically; HNN hardcodes it. This suggests a hybrid: meta-learn auxiliary conserved quantities $g_\phi$ on top of an HNN that already conserves $H$ exactly — Noether providing soft auxiliary biases, HNN providing the hard energy conservation backbone.

---

## Cross-Links

- [[hamiltonian-message-passing]] — concept page formalizing Hamiltonian+dissipative GNN architectures
- [[action-based-noether-enforcement]] — sibling page on action-functional and Noether enforcement
- [[equivariant-gnns]] — physics inductive bias taxonomy
- [[lagrangian-neural-networks]] — Lagrangian variant
- [[dissipative-hamiltonian-neural-networks]] — dissipative extension
- [[dynami-cal-graphnet]] — complementary momentum conservation
- [[noether-networks]] — auxiliary discovered conserved quantities
