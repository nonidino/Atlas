# Prerequisites & Learning Resources

**Type:** Reference / learning guide
**Audience:** Someone with ML, multivariable calculus, linear algebra, ODEs, and basic mechanics + fluids who wants to understand the PFM wiki deeply.

---

## What You Already Have

- Machine learning (training, backprop, transformers)
- Multivariable calculus ($\nabla$, $\nabla^2$, line/surface integrals, chain rule in multiple variables)
- Linear algebra (eigendecomposition, matrix norms, vector spaces)
- ODEs (existence/uniqueness, phase portraits, numerical integration)
- Basic classical mechanics (Newton's laws, energy/momentum conservation)
- Basic fluid mechanics (Bernoulli, continuity, pressure, viscosity)

---

## Tier 1 — Do These First (Direct Gaps)

### 1. Partial Differential Equations

The language of the entire project. You know ODEs; PDEs are their 2D/3D/space-time extension. The classification into **elliptic / parabolic / hyperbolic** (Laplace, heat, wave) maps almost perfectly onto the local-vs-global tokenization axis — understanding it early makes every architecture decision clearer.

**Resource:** Strauss — *Partial Differential Equations: An Introduction* (Wiley)
- Best undergraduate intro; physical motivation throughout
- Priority chapters: 1–5 (classification, separation of variables, Fourier series as solutions, Green's functions)

**Backup:** MIT OCW 18.152 — free lecture notes, same scope

**Priority topics:**
- Separation of variables
- Fourier series as a function-space basis (connects directly to spectral tokens)
- The three canonical PDE types and their Green's functions
- Boundary value problems vs. initial value problems
- Maximum principle (elliptic), characteristics (hyperbolic), diffusion kernel (parabolic)

---

### 2. Fourier Analysis and Spectral Methods

Understand the Fourier transform not just as signal processing but as a way of *representing functions in a different basis*. The FNO / spectral token architecture only makes sense once you see that a Fourier mode is a global basis function and that differentiation becomes multiplication ($\partial_x \to ik$).

**Resource:** Trefethen — *Spectral Methods in MATLAB* (SIAM; free PDF at people.maths.ox.ac.uk/trefethen/)
- Very applied and code-first; shows directly how spectral methods solve PDEs
- Short book, high insight per page

**Priority topics:**
- DFT/FFT — definition, periodicity assumption, computational cost $O(N \log N)$
- Spectral differentiation: $\partial_x u \leftrightarrow ik \hat u(k)$ — this is why physics constraints are cheap in spectral space
- Sobolev decay: smooth functions have rapidly decaying Fourier coefficients (this is why few modes suffice for smooth PDE solutions)
- Aliasing and the Nyquist criterion — directly causes rollout instability in patch tokenizers

---

### 3. Numerical Methods for PDEs

Classical numerical solvers (FDM, FEM, spectral) are what neural operators are trying to generalize. Without knowing their limitations — resolution-dependent cost, solver-specific code, no cross-domain transfer — the PFM motivation feels abstract. This course gives the "why does this matter" context.

**Resource:** Leveque — *Finite Difference Methods for Ordinary and Partial Differential Equations* (SIAM; free PDF at faculty.washington.edu/rjl/fdmbook/)
- Best intro; covers stability, consistency, convergence rigorously but accessibly

**Supplement:** Trefethen Chapters 1–4 (spectral methods as an alternative)

**Priority topics:**
- Finite differences: forward/backward/central stencils, truncation error
- CFL stability condition — why numerical solvers have a maximum timestep; understanding this explains why neural emulators are useful (they can take large timesteps)
- Condition numbers of discretized elliptic operators
- What "resolution" means and why doubling it makes classical solvers ~8× more expensive in 3D

---

## Tier 2 — Fill These In Parallel

### 4. Lagrangian and Hamiltonian Mechanics

Prerequisite for the structure-preserving architecture side of the wiki: Hamiltonian neural networks, Lagrangian neural networks, action-based Noether enforcement, Hamiltonian message passing. The key insight — parameterizing $H_\theta(q,p)$ and taking gradients gives exact energy conservation by construction — only lands once you understand Hamilton's equations.

**Resource (undergraduate entry):** Taylor — *Classical Mechanics* (University Science Books)
- Clearer exposition than Goldstein; good for building intuition first

**Resource (graduate depth):** Goldstein — *Classical Mechanics* (Addison-Wesley)
- Chapters 1–3 (Lagrangian) and 7–9 (Hamiltonian formalism, canonical transformations)

**Priority topics:**
- Generalized coordinates and constraints
- Lagrangian $L = T - V$, Euler-Lagrange equations
- Hamiltonian $H = T + V$, Hamilton's equations $\dot q = \partial H/\partial p$, $\dot p = -\partial H/\partial q$
- Poisson brackets (the structure that antisymmetric architectures preserve)
- **Noether's theorem** — symmetry of the action → conserved quantity. This is central to the action-based Noether enforcement architecture page.

---

### 5. Functional Analysis (Light)

Neural operators learn maps between *function spaces*, not between $\mathbb{R}^n$ and $\mathbb{R}^m$. The formal distinction — a function maps numbers to numbers; an operator maps functions to functions — is what makes "resolution-invariant" a precise statement rather than a vague claim. You don't need a full graduate course; Chapters 1–3 of one text gives you the necessary vocabulary.

**Resource:** Kreyszig — *Introductory Functional Analysis with Applications* (Wiley), Chapters 1–3 only

**Supplement:** First 10 pages of the DeepONet paper background (Li et al., arXiv:1910.03193) — states the universal approximation theorem for operators in plain language

**Priority topics:**
- Normed vector spaces and Banach spaces (extend "vector" from $\mathbb{R}^n$ to spaces of functions)
- Sobolev spaces $H^s(\Omega)$ — what it means for a function to have $s$ square-integrable derivatives (this is the regularity that makes spectral truncation lossless for smooth fields)
- Bounded linear operators between function spaces, operator norms
- Why the FNO output is "resolution-invariant" formally: it approximates a continuous operator $\mathcal G: L^2(\Omega) \to L^2(\Omega)$, not a matrix $\mathbb{R}^{J \times J} \to \mathbb{R}^{J \times J}$

---

### 6. Fluid Mechanics (Deeper)

The primary testbed of the whole project. You know basic fluids; what you need is the Navier-Stokes equations in their full incompressible form, the Reynolds number as a bifurcation parameter, turbulence phenomenology, and boundary layer theory.

**Resource:** Batchelor — *An Introduction to Fluid Dynamics* (Cambridge)
- The classic reference; dense but complete

**Lighter alternative:** MIT OCW 2.20 (Marine Hydrodynamics) or 2.29 (Numerical Fluid Mechanics) — both free, more applied

**Video supplement:** Steve Brunton's YouTube lectures on fluid mechanics and data-driven methods — intuition-first, directly relevant

**Priority topics:**
- Full incompressible Navier-Stokes derivation from conservation laws
- Dimensionless numbers: Reynolds (Re), Mach (Ma), Prandtl (Pr), Péclet (Pe) — these are the governing parameters the PFM is conditioned on
- Laminar-turbulent transition; what Re controls
- Kolmogorov microscale and energy cascade — why turbulence is a multiscale problem and why single-resolution tokenizers fail at high Re
- Boundary layers — thin regions of rapid variation that require fine local resolution (the case for local tokens)
- Pressure-velocity coupling (the PPE: $\nabla^2 p = -\nabla \cdot (\mathbf u \cdot \nabla \mathbf u)$) — the canonical elliptic problem inside NS, and why global tokens matter

---

## Tier 3 — When the Project Matures

### 7. Statistical Mechanics

Needed when the project gets into generative PFMs, diffusion models, partition functions, and thermodynamic limits. Also gives intuition for the "energy landscape" language used in Hamiltonian architectures.

**Resource:** MIT OCW 8.333/8.334 (Statistical Mechanics I & II) — Kardar's lectures, freely available; excellent

---

### 8. Electrodynamics

Needed when the project extends beyond fluids to MHD or plasma physics. Also gives a second complete set of PDE examples: Poisson ($\nabla^2 \phi = -\rho/\epsilon_0$, elliptic), wave equation for EM fields (hyperbolic), diffusion in conductors (parabolic).

**Resource:** Griffiths — *Introduction to Electrodynamics* (Cambridge) — the standard undergraduate text

---

## Project-Specific Resources (Use Throughout)

These are not courses but map directly onto wiki content and can be read alongside the above.

| Resource | What it covers | When to use |
|---|---|---|
| Brunton & Kutz — *Data-Driven Science and Engineering* (free at databookuw.com) | DMD, SINDy, POD/ROM, neural ODEs — ML + physics from an applied math perspective | Start early; read Ch. 6–7 first |
| Steve Brunton's YouTube channel | Video companion to above; fluid mechanics + data-driven methods | Ongoing; great for intuition |
| FNO paper — Li et al. (arXiv:2010.08895) | The original Fourier Neural Operator; once you have spectral methods (Tier 1 above) this paper is readable and gives the full spectral token picture | After Tier 1 item 2 |
| Poseidon paper — Herde et al. (arXiv:2405.19101) | The most complete current PFM; once you have PDEs + numerical methods the methods section is clear | After Tier 1 items 1 and 3 |
| DeepONet paper — Lu et al. (arXiv:1910.03193) | The branch/trunk operator tokenization; readable with basic calculus + Tier 2 item 5 | After Tier 2 item 5 |

---

## Suggested Order and Timeline

| Week | Topic |
|---|---|
| 1–5 | Strauss PDEs, Chapters 1–5 |
| 3–6 | Trefethen Spectral Methods, Chapters 1–4 (overlap with PDEs) |
| 6–8 | Leveque Finite Differences, Chapters 1–3 |
| 1–ongoing | Brunton & Kutz (reference style, read chapters as they become relevant) |
| 9–12 | Goldstein / Taylor Classical Mechanics, Lagrangian + Hamiltonian chapters |
| 12–14 | Kreyszig Functional Analysis, Chapters 1–3 only |
| 14–20 | Batchelor or MIT OCW fluids |
