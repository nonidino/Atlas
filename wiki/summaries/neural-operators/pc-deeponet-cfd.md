# Summary: Physics-Constrained DeepONet for Surrogate CFD Models

**Source:** `raw/Physics-constrained DeepONet for Surrogate CFD models a curved backward-facing step case.md`  
**Authors:** Anas Jnini, Harshinee Goordoyal, Sujal Dave, Flavio Vella, Katharine H. Fraser, Artem Korobenko  
**arXiv:** 2503.11196v1  
**Date Ingested:** 2026-04-11

---

## Overview

Introduces **PC-DeepONet** (Physics-Constrained DeepONet) — a DeepONet architecture that **hard-enforces the divergence-free constraint** (continuity equation) by construction, applied to surrogate modeling of incompressible flow over parameterized curved backward-facing step (BFS) geometries. Achieves ~7× lower relative $L_2$ error vs. standard DeepONet with only 50 training samples.

---

## Physical Problem

Incompressible Navier-Stokes equations:
$$\nabla\cdot\mathbf{u} = 0 \quad \text{(continuity)}$$
$$\rho\frac{\partial\mathbf{u}}{\partial t} + \rho(\mathbf{u}\cdot\nabla)\mathbf{u} = -\nabla p + \mu\nabla^2\mathbf{u} + \mathbf{f}_b \quad \text{(momentum)}$$

The continuity equation imposes the **divergence-free condition** on velocity: $\nabla\cdot\mathbf{u} = 0$.

---

## PC-DeepONet Architecture

Standard DeepONet output:
$$G^{\xi_q}(x,y) = \sum_{i=1}^{N_\alpha} \alpha_i(\xi_g;\theta_\alpha)\,\phi_i(x,y;\theta_\phi)$$

For PC-DeepONet, raw velocity outputs $b_u(x,y)$ and $b_v(x,y)$ are combined into a vector field $\mathbf{b}$, and a **skew-symmetric matrix $A$** is constructed from the Jacobian:
$$A = J_b - J_b^\top = \begin{bmatrix} 0 & \frac{\partial b_u}{\partial y} - \frac{\partial b_v}{\partial x} \\ \frac{\partial b_v}{\partial x} - \frac{\partial b_u}{\partial y} & 0 \end{bmatrix}$$

The divergence-free velocity field is:
$$\mathbf{v} = \text{div}(A)$$

By construction, $\nabla\cdot\mathbf{v} = 0$ is exactly satisfied (divergence of a skew-symmetric Jacobian is identically zero). This is the approach of Richter-Powell et al. (2022) applied to DeepONets.

---

## Geometry Parameterization

Curved BFS slope parameterized via **NURBS** (Non-Uniform Rational B-Splines) with 4 control points (2 fixed, 2 varied). The x-coordinates of the 2 free control points serve as DeepONet branch inputs. 50 geometries generated with FreeFEM++ at $Re = 1000$.

---

## Results

| Model | Training Loss | Validation Loss | Relative $L_2$ |
|---|---|---|---|
| PC-DeepONet | $1.77\times10^{-6}$ | $1.82\times10^{-6}$ | $4.45\times10^{-3}$ |
| DeepONet | $1.24\times10^{-5}$ | $1.23\times10^{-5}$ | $1.16\times10^{-2}$ |

Both converge in only 50 L-BFGS iterations. PC-DeepONet is ~7× more accurate. Errors cluster near boundary layers due to undersampling near the slope.

---

## Relevance to Physics Foundation Model Goal

Hard-encoding physical conservation laws (divergence-free, mass conservation, energy conservation) into the network architecture rather than soft-penalizing them in the loss function is a powerful paradigm. For a PFM, architectures that **exactly satisfy physical symmetries and conservation laws by construction** will be more reliable and data-efficient than those that learn approximate compliance.

**[AI Inference]:** The skew-symmetric Jacobian trick is an instance of a broader principle: **use differential geometry to build conservation laws into the network architecture**. This could be extended to enforce other conservation laws: incompressibility via stream functions in 2D, solenoidal magnetic fields in MHD via vector potentials, energy conservation via Hamiltonian structure. A PFM might incorporate a library of such physics-constrained output heads that users can select based on the governing equations of their problem.

---

## Links

- [[neural-operators]] — DeepONet fundamentals
- [[navier-stokes-equations]] — the governing equations
- [[deeponet-multi-operator]] — multi-operator extension of DeepONet
- [[varmion-viscous-flows]] — another operator approach for viscous flows
- [[navier-stokes-nonuniform-grids]] — HyDEA for CFD on non-uniform grids
- [[neural-surrogates]] — surrogate modeling context
