# Summary: Variationally Mimetic Operator Network Approach to Transient Viscous Flows

**Source:** `raw/Variationally mimetic operator network approach to transient viscous flows.md`  
**Authors:** Laura Rinaldi, Giulio G. Giusteri (CNR-IMATI Pavia, Università di Padova)  
**arXiv:** 2604.02124v1  
**Date Ingested:** 2026-04-11

---

## Overview

Extends the **Variationally Mimetic Operator Network (VarMiON)** — a physics-informed operator network whose branch architecture is determined by the **weak (variational) formulation of the governing PDE** — to **transient viscous flows** (time-dependent Stokes problem). Demonstrates very good agreement with reference finite-element solutions on three paradigmatic flow geometries.

---

## Physical Problem

The incompressible Navier-Stokes equations for a Newtonian fluid:
$$\rho\left(\frac{\partial\mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right) = \nabla\cdot\boldsymbol{\sigma}(\mathbf{u},p) + \mathbf{f}$$
$$\nabla\cdot\mathbf{u} = 0$$

with stress tensor $\boldsymbol{\sigma}(\mathbf{u},p) = 2\mu\dot{\boldsymbol{\varepsilon}}(\mathbf{u}) - p\mathbf{I}$ and strain-rate tensor:
$$\dot{\boldsymbol{\varepsilon}}(\mathbf{u}) = \frac{1}{2}\left(\nabla\mathbf{u} + (\nabla\mathbf{u})^\top\right)$$

At low-to-moderate Reynolds number (inertial forces $\ll$ viscous forces), this linearizes to the **time-dependent Stokes problem**:
$$\rho\frac{\partial\mathbf{u}}{\partial t} = -\nabla p + \mu\Delta\mathbf{u} + \mathbf{f}, \quad \nabla\cdot\mathbf{u} = 0$$

---

## VarMiON Architecture

Like DeepONet, VarMiON has **trunk networks** (build basis functions from spatial-temporal coordinates) and **branch networks** (build coefficients from input data — initial/boundary conditions, parameters, forcing). 

The key distinction: the **branch architecture is precisely determined by the discrete weak form** of the PDE. This is analogous to how Galerkin finite element methods build basis functions from the variational principle — hence "variationally mimetic."

For the Stokes problem, the weak form involves:
$$a(\mathbf{u},\mathbf{v}) = \int_\Omega 2\mu\,\dot{\boldsymbol{\varepsilon}}(\mathbf{u}):\dot{\boldsymbol{\varepsilon}}(\mathbf{v})\,d\Omega - \int_\Omega p\,\nabla\cdot\mathbf{v}\,d\Omega$$

The branch network structure mirrors this bilinear form.

---

## Test Cases

Three paradigmatic geometries:
1. **Lid-driven cavity flow** — classical benchmark for wall-driven recirculation.
2. **Flow past a cylinder** — vortex shedding, time-dependent wake.
3. **Contraction flow** — channel narrowing, pressure-driven acceleration.

VarMiON predictions show very good agreement with reference finite-element solutions across all three cases, particularly in capturing **transient behavior**.

---

## Relationship to DeepONet

VarMiON is an upgrade of DeepONet where the variational formulation of the PDE is **exploited in the architecture itself** (not just as a loss function penalty as in PINNs / PI-DeepONets). This means:
- The network respects the mathematical structure of the problem by construction.
- It can leverage existing variational FEM theory for convergence guarantees.
- Related to VarNet and VPINNs — but those modify the loss function, not the architecture.

---

## Relevance to Physics Foundation Model Goal

VarMiON represents a principled approach to encoding physics into operator network architectures using the mathematical structure of variational formulations. For a PFM spanning multiple PDEs, a key question is how to encode different physical laws efficiently. VarMiON shows that the **discrete weak form itself can serve as an architectural blueprint** — potentially allowing automatic derivation of physics-consistent network architectures from equation specifications.

**[AI Inference]:** The variational formulation framework could be combined with symbolic computation: given a new PDE, automatically derive the weak form and generate the corresponding VarMiON branch architecture. This would constitute a form of **meta-learning at the architectural level** — an alternative to the in-context learning approach of GP$_{\text{hy}}$T, WALRUS. The two approaches are complementary: one uses data-driven context inference, the other uses mathematical structure.

---

## Links

- [[neural-operators]] — DeepONet and VarMiON family
- [[navier-stokes-equations]] — governing equations
- [[partial-differential-equations]] — variational formulations
- [[pc-deeponet-cfd]] — related physics-constrained DeepONet
- [[deeponet-multi-operator]] — multi-operator DeepONet pretraining
- [[navier-stokes-nonuniform-grids]] — hybrid approach to viscous CFD
