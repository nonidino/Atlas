# Partial Differential Equations (PDEs)

**Type:** Core Concept  
**Related Sources:** All physics simulation papers

---

## Definition

A **Partial Differential Equation (PDE)** relates a function $u(x, t)$ of multiple independent variables to its partial derivatives. Physical systems are typically governed by PDEs that encode conservation laws, constitutive relations, and symmetry principles.

The general form of a $k$-th order PDE:
$$F\!\left(x, t, u, \frac{\partial u}{\partial x}, \frac{\partial u}{\partial t}, \frac{\partial^2 u}{\partial x^2}, \ldots\right) = 0$$

---

## PDEs Prominent in This Wiki

### Navier-Stokes Equations (Incompressible)
The governing equations of viscous incompressible flow:
$$\rho\left(\frac{\partial\mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right) = -\nabla p + \mu\nabla^2\mathbf{u} + \mathbf{f}$$
$$\nabla\cdot\mathbf{u} = 0$$

where $\mathbf{u}$ is velocity, $p$ is pressure, $\rho$ is density, $\mu$ is dynamic viscosity, $Re = \rho U L/\mu$ is the Reynolds number.

### Euler Equations (Compressible)
Inviscid compressible flow — describes shockwaves and rarefaction waves:
$$\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\mathbf{u}) = 0$$
$$\frac{\partial(\rho\mathbf{u})}{\partial t} + \nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u} + p\mathbf{I}) = 0$$
$$\frac{\partial E}{\partial t} + \nabla\cdot((E+p)\mathbf{u}) = 0$$

### Magnetohydrodynamics (MHD)
Navier-Stokes coupled with Maxwell's equations; describes plasma dynamics in fusion reactors and astrophysical systems:
$$\rho\left(\frac{\partial\mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\right) = -\nabla p + \mathbf{J}\times\mathbf{B} + \mu\nabla^2\mathbf{u}$$
$$\frac{\partial\mathbf{B}}{\partial t} = \nabla\times(\mathbf{u}\times\mathbf{B}) + \eta\nabla^2\mathbf{B}$$

### Stokes Equations (Linearized)
Low-Reynolds limit of Navier-Stokes (inertia negligible):
$$\rho\frac{\partial\mathbf{u}}{\partial t} = -\nabla p + \mu\Delta\mathbf{u} + \mathbf{f}, \quad \nabla\cdot\mathbf{u} = 0$$

### Rayleigh-Bénard Convection
Buoyancy-driven thermal convection; Boussinesq approximation adds temperature equation:
$$\nabla\cdot\mathbf{u} = 0, \quad \frac{D\mathbf{u}}{Dt} = -\nabla p + \nu\nabla^2\mathbf{u} + \alpha g T\hat{e}_z$$
$$\frac{DT}{Dt} = \kappa\nabla^2 T$$

### Burgers' Equation
1D or 2D nonlinear PDE; canonical test case for shock formation and diffusion:
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}$$

### Pressure Poisson Equation (PPE)
Derived from incompressibility in fractional-step methods; the main computational bottleneck:
$$\nabla^2 p = \frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^*$$

---

## Numerical Methods for PDEs

Classical approaches that ML surrogate models compete with / complement:

| Method | Description |
|---|---|
| **Finite Difference (FD)** | Approximate derivatives on structured grids using Taylor expansion |
| **Finite Element (FE)** | Variational formulation on unstructured meshes; weak form |
| **Finite Volume (FV)** | Conserves fluxes across cell faces; robust for conservation laws |
| **Spectral Methods** | Expand solution in global basis (Fourier, Chebyshev); exponential convergence for smooth solutions |

Fractional-step (projection) methods split NS into advection + diffusion + pressure correction steps, solving the PPE at each timestep.

---

## PDE Classification

| Type | Example | Key Property |
|---|---|---|
| **Elliptic** | Laplace, Poisson | Smoothing, instantaneous action-at-a-distance |
| **Parabolic** | Heat equation | Diffusion, finite propagation speed |
| **Hyperbolic** | Wave equation, Euler | Wave propagation, characteristic curves |
| **Nonlinear** | Navier-Stokes, Burgers | Turbulence, chaos, shocks |

---

## The Operator Learning Perspective

A PDE defines an operator $G$ mapping:
- Input functions (initial conditions, boundary conditions, parameters) → Solution function

Neural operators learn this mapping directly without solving the PDE. The key advantage: once trained, evaluation is fast (microseconds vs. hours for CFD). The key challenge: accuracy, generalization to out-of-distribution inputs, and long-horizon stability.

---

## See Also

- [[neural-operators]]
- [[navier-stokes-equations]]
- [[physics-foundation-models]]
- [[autoregressive-rollout-stability]]
