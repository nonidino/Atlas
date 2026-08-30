# Navier-Stokes Equations

**Type:** Core Concept  
**Related Sources:** HyDEA (non-uniform grids), PC-DeepONet, VarMiON, GP$_{\text{hy}}$T, Walrus

---

## Overview

The **Navier-Stokes (NS) equations** are the fundamental governing equations of viscous fluid flow. They express conservation of momentum and mass for a Newtonian fluid. Despite their deceptively simple form, they encode extraordinarily complex behavior — including turbulence — that remains one of the major unsolved problems in physics and mathematics (Clay Millennium Prize Problem).

---

## Incompressible Navier-Stokes

For an incompressible ($\nabla\cdot\mathbf{u}=0$) Newtonian fluid with constant density $\rho$ and dynamic viscosity $\mu$:

**Momentum equation:**
$$\rho\underbrace{\left(\frac{\partial\mathbf{u}}{\partial t} + \underbrace{(\mathbf{u}\cdot\nabla)\mathbf{u}}_\text{advection}\right)}_\text{material derivative} = \underbrace{-\nabla p}_\text{pressure} + \underbrace{\mu\nabla^2\mathbf{u}}_\text{diffusion} + \underbrace{\mathbf{f}}_\text{body forces}$$

**Continuity (incompressibility):**
$$\nabla\cdot\mathbf{u} = 0$$

**Dimensionless form** (Re = Reynolds number $= \rho U L / \mu$):
$$\frac{\partial\mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \frac{1}{Re}\nabla^2\mathbf{u} + \mathbf{f}$$

---

## Compressible Navier-Stokes

When density $\rho$ varies (high-speed flows, shockwaves):
$$\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\mathbf{u}) = 0 \quad \text{(mass)}$$
$$\frac{\partial(\rho\mathbf{u})}{\partial t} + \nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u}) = -\nabla p + \nabla\cdot\boldsymbol{\tau} + \mathbf{f} \quad \text{(momentum)}$$
$$\frac{\partial E}{\partial t} + \nabla\cdot((E+p)\mathbf{u}) = \nabla\cdot(\boldsymbol{\tau}\cdot\mathbf{u}) + \mathbf{q} \quad \text{(energy)}$$

The **Euler equations** are the inviscid limit ($\mu \to 0$, $\boldsymbol{\tau} = 0$).

---

## The Reynolds Number and Flow Regimes

$$Re = \frac{\rho U L}{\mu} = \frac{\text{inertial forces}}{\text{viscous forces}}$$

| $Re$ | Regime |
|---|---|
| $\ll 1$ | Stokes flow (linear, fully viscosity-dominated) |
| $\sim 1$–$100$ | Laminar flow with weak inertia |
| $\sim 1000$ | Transitional; onset of instabilities |
| $\gg 10^4$ | Turbulent flow (chaotic, multi-scale) |

---

## Turbulence

At high $Re$, the nonlinear advection term $(\mathbf{u}\cdot\nabla)\mathbf{u}$ generates an energy cascade from large to small scales (Kolmogorov cascade). Turbulence is:
- Chaotic (sensitive to initial conditions).
- Multi-scale (energy at all spatial scales $\eta \leq \ell \leq L$, where $\eta$ is the Kolmogorov scale).
- Dissipative (kinetic energy converted to heat at small scales by viscosity).

The Kolmogorov microscale: $\eta = (\nu^3/\varepsilon)^{1/4}$ where $\nu = \mu/\rho$ is kinematic viscosity and $\varepsilon$ is the energy dissipation rate.

---

## Pressure Poisson Equation

In fractional-step methods, the incompressibility constraint leads to a **Pressure Poisson Equation** (PPE) at each timestep:
$$\nabla^2 p = \frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^*$$
Solving this large linear system is the primary computational bottleneck in most incompressible flow codes. HyDEA accelerates this step using DeepONet-based line-search directions.

---

## Immersed Boundary Method (IBM)

For flows with solid obstacles (fluid-structure interaction), the forcing term $\mathbf{f}$ in the NS equations encodes no-slip boundary conditions on immersed surfaces without requiring body-fitted meshes:
$$\frac{\partial\mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u} = -\nabla p + \frac{1}{Re}\nabla^2\mathbf{u} + \mathbf{f}$$
The MConv-based HyDEA framework handles this setting on non-uniform Cartesian grids.

---

## Machine Learning for Navier-Stokes

Multiple ML approaches exist for NS simulation:

| Approach | Examples | Trade-offs |
|---|---|---|
| **Physics-Informed NNs (PINNs)** | Raissi et al. 2019 | Data-free but slow convergence |
| **Transformer-PINN hybrids** | PhysicsFormer | 3× faster than PINNsFormer; 0% inverse error; handles high-frequency solutions |
| **Neural Operators (FNO, DeepONet)** | Li et al. 2020, Lu et al. 2019 | Fast inference, discretization-invariant |
| **Hybrid solvers** | HyDEA | Accelerates bottleneck, preserves CFD accuracy |
| **End-to-end surrogates** | Walrus, GP$_{\text{hy}}$T | Fastest inference, lower accuracy |
| **Diffusion emulators** | Lost in Latent Space | Captures uncertainty, stable rollouts |

### PhysicsFormer PINN (2025)

PhysicsFormer ([[physicsformer-pinn-ns]]) addresses core PINN failure modes for NS:
- Converts pointwise $(x,t)$ inputs into temporal pseudo-sequences → encoder-decoder cross-attention captures temporal dependencies
- Novel $w\sin(t)$ activation (trainable $w$) handles high-frequency vortex shedding
- Achieves MSE $\approx 10^{-6}$ for 2D cylinder wake at $\text{Re}=100$
- **Inverse problem** (parameter identification $\lambda_1$, $\lambda_2$): 0% error on clean data, 0.07% under 1% Gaussian noise
- GPU memory: ~500MB; ~3× faster than PINNsFormer

---

## See Also

- [[partial-differential-equations]]
- [[neural-operators]]
- [[autoregressive-rollout-stability]]
- [[navier-stokes-nonuniform-grids]]
- [[pc-deeponet-cfd]]
- [[varmion-viscous-flows]]
- [[physicsformer-pinn-ns]]
- [[poseidon-pde-foundation-model]] — pretrained on incompressible NS + compressible Euler operators
- [[gphyt-physics-foundation-model]]
- [[walrus-paper]]
