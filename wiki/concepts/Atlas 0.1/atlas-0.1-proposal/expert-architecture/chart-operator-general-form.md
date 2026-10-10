# The chart operator in general form — nonlinear physics, multi-field seams, and co-located fields

**Type:** Concept page — **design note under discussion** (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-10-10 in answer to the owner's three questions of that day: why so much of the design relies on the PDE being linear, and what a generalized form would be; how fields that differ across a seam couple (the rocket's air against its skin); and how two separately trained single-field experts on one piece (temperature and stress) can work in one frame. **This is a proposal for discussion, not a decision.** [[chart-operator-architecture]] version 1, and the public paper, stand until the owner decides. The sanity checks in §2 and §4 are measured by `scripts/arch_general_form_checks.py` (`out/arch/general_form_checks.txt`), with classical solves plus perturbations standing in for networks, as in the Gram probe; they check algebra and convergence claims and say nothing about any trained network. Everything else is a derivation, a cited result, or **[AI Inference]**.
**Hub:** [[00-proposal-workstreams]] · **Extends:** [[chart-operator-architecture]] · [[chart-operator-design-decisions]] · **Builds on the record:** [[port-algebra-atlas-0.1]] · [[case-study-thermal-strain-atlas-0.1]] · [[case-study-wing-fsi-atlas-0.1]] · [[coupling-cost-and-complexity]] · [[formal-proofs-plan]]

---

## 0. The answer in one paragraph

Version 1 relies on linearity in three places: **superposition** (a piece's response to any boundary data is a fixed combination of modes formed once), **positivity by construction** (the port matrix is the Gram of a fixed quadratic energy), and **one solve with no iteration**. None of the three is what actually carries the guarantee. What carries it is two facts that do not need linearity: **the network only proposes fields whose boundary values are exact, and the host evaluates the true discrete physics on them.** The general form keeps those two facts and drops the rest. The network becomes a **trial-space generator**: at the current state it returns constraint modes, a few interior enrichment modes and a predicted field, all with exact traces. The host then **projects the true physics onto that space**: it minimises the energy where the physics has one, and projects the residual (Petrov–Galerkin or least squares) where it does not. The guarantee becomes a hierarchy set by the structure of the physics (§2.2), with version 1's tier 0 as its linear, symmetric corner. The same principle, *the network proposes and the host holds the physics*, answers the two coupling questions. Across a seam, the fields need not match; only the typed power-conjugate ports of [[port-algebra-atlas-0.1]] must (§3). On one piece, two single-field experts cooperate through a **host-owned coupling form over the product of their trial spaces**. For thermoelasticity that form is symmetric quasi-definite, so it stays well-conditioned and quasi-optimal for any expert outputs (§4).

---

## 1. Where version 1 is linear, and what its guarantee really rests on

| what version 1 does | why it needs linearity | what breaks for a nonlinear PDE |
|---|---|---|
| forms each piece's constraint modes once and reuses them for every boundary datum and every step | superposition: the response to $\sum_k c_kq_k$ is $\sum_kc_kH_k+u_p$ | the response depends nonlinearly on the boundary data and on the state, so no fixed basis reproduces it |
| forms the port matrix as $\Lambda=V^\top MV$ | the discrete problem is the minimiser of a fixed convex **quadratic** | the energy (if any) is not quadratic; for convection-dominated flow, reacting flow or shocks there is no energy to minimise at all |
| couples with one interface solve | both of the above | a nonlinear interface equation needs iteration (Newton or a fixed point) |

**Why it was built that way.** Stage 0 is linear diffusion (decision 18), and only there does a guarantee hold *a priori, for every network output*. But version 1 then treated nonlinearity as an exception (tiers 1 and 2), which is backwards for a physics foundation model: most of the target physics (compressible and reacting flow, large deformation, temperature-dependent properties, radiation) is nonlinear.

**What the guarantee actually rests on** (read off the proof of version 1's Theorem 5, [[chart-operator-architecture]] §3.6):

- **(E) exact traces.** Every field the network returns carries exactly the prescribed boundary values, so the pieces are conforming by construction.
- **(H) the host holds the physics.** The host never asks the network for a number it cannot check (a port matrix, a flux). It evaluates its own energy matrix $M$ on the network's fields.

Positivity and Céa-optimality then follow from the **Dirichlet principle** (the exact solution minimises the energy among fields with the same boundary values). Linearity only made that minimum a quadratic form. Both (E) and (H) make sense for any PDE, and the general form is built from them.

---

## 2. The general form

### 2.1 Definition

**Intuition.** Do not ask the network for *the* answer; ask it for a small set of good **directions** to look in, each with the right boundary values. The host, which knows the equations exactly, finds the best answer in that set. For a linear problem the right set contains the exact answer, and one search suffices. For a nonlinear problem the host searches, moves, asks the network for new directions at the new point, and searches again.

**Rigorously.** On a piece with chart $F$, the physics family is given **to the host** as a discrete residual on the reference grid,
$$R(w;z)=0,\qquad w=(u,\lambda),$$
with $u$ the cell values of every field of the family, $\lambda=(\lambda_P)_P$ the boundary values per **typed port component** $P$ (§3), and $z$ the inputs (geometry fields, materials, sources, the previous step, scalars). The host can evaluate $R$ and its Jacobian–vector products without solving anything. A **generalized chart operator** of rank $r$ is a network
$$\mathcal N_\theta:\ (z,x)\ \longmapsto\ \bigl(\hat H(x),\ \hat B(x),\ \hat u(x)\bigr),$$
where $x$ is the current state (absent for a linear family):

- $\hat H\in\mathbb R^{n_u\times p}$ are **state-conditioned constraint modes**, one per port coordinate ($p=\sum_P 4\,m\,\dim P$);
- $\hat B\in\mathbb R^{n_u\times r}$ are **interior enrichment modes** with zero trace;
- $\hat u$ is the network's predicted interior field.

Together they define the **affine trial set**
$$\widetilde W(x)=\Bigl\{\bigl(\hat u+\hat Hc+\hat Ba,\ Qc\bigr):\ c\in\mathbb R^{p},\ a\in\mathbb R^{r}\Bigr\},$$
every member of which has the exact trace $Qc$. Pieces share the port coordinates $c$ of their common sides. The host solves the **projected problem** on the assembled trial set (§2.2), sets $x$ to the result, and calls the network again if the physics is nonlinear.

**Version 1 is the linear corner.** If the modes do not depend on $x$, $\hat u$ is the particular field and $r=0$, this is exactly tier 0: one call per piece, one solve, no outer iteration.

**Affine, not curved, on purpose.** One could let the network return a field as a nonlinear function of the boundary data (a "manifold" trial set). The affine form keeps the host's projected problem **convex whenever the physics' energy is convex** (a convex function composed with an affine map is convex). That makes the projected problem uniquely solvable and its Newton iteration globally convergent. The manifold form gives up both in exchange for fewer degrees of freedom.

### 2.2 The guarantee is set by the structure of the physics

| class | examples | the host's projection | the guarantee, for **any** network output | status |
|---|---|---|---|---|
| **G0** linear, symmetric, coercive | diffusion; linear elasticity; Stokes in stream-function form | the energy Gram (version 1) | PSD port matrices; error $=E^\top A_IE$, quadratic and one-sided; Céa with constant 1 | **machine-checked** (`Atlas.tier0_headline`) |
| **G0′** linear, non-symmetric | advection–diffusion; Oseen | Petrov–Galerkin with **adjoint** constraint modes as test functions | port error $=E_W^\top A_{II}E_V$, quadratic; **no positivity**, even for the exact piece; stability checked a posteriori | derived; checked (§2.3) |
| **G1** convex energy | nonlinear conduction (after Kirchhoff's transform); radiation boundaries; $p$-Laplacian; implicit steps of gradient flows; incremental plasticity | minimise the **true** energy over $\widetilde W$ | energy upper bound; quasi-optimality with constant $\sqrt{L/m}$; a learned subspace correction that never does worse than one classical step (§2.4) | derived; checked (§2.4) |
| **G2** general residual | compressible and reacting Navier–Stokes; non-potential constitutive laws | least-squares Petrov–Galerkin of the true residual over $\widetilde W$, then classical defect correction if the residual is above tolerance | a computable residual certificate at every call; local quasi-optimality; with the fallback, **the classical answer** | cited ingredients; not checked here |
| **G3** explicit hyperbolic | shock-capturing steps | the network proposes face fluxes; the host limits them toward a classical monotone flux | exact conservation and positivity of density and pressure (an invariant domain) | cited ingredients; not checked here |

### 2.3 G0′ — non-symmetric physics needs adjoint modes

**Intuition.** With advection, the response to "push here" is not the mirror image of the response to "measure here". The Gram trick used one set of fields for both roles; the non-symmetric version needs two sets: the network's **primal** modes (how the piece responds) and its **adjoint** modes (how a boundary measurement weights the interior).

**Rigorously.** With $M=\begin{pmatrix}A_{II}&A_{I\Gamma}\\A_{\Gamma I}&A_{\Gamma\Gamma}\end{pmatrix}$ non-symmetric, define primal modes $V=(H;Q)$ by $A_{II}H+A_{I\Gamma}Q=0$ and adjoint modes $W=(G;Q)$ by $A_{II}^\top G+A_{\Gamma I}^\top Q=0$. Then $W^\top MV=Q^\top(A_{\Gamma\Gamma}-A_{\Gamma I}A_{II}^{-1}A_{I\Gamma})Q=\Lambda$. For learned $\tilde V=V+(E_V;0)$ and $\tilde W=W+(E_W;0)$, the two cross terms vanish, because the interior rows of $MV$ and of $M^\top W$ are zero:
$$\tilde W^\top M\tilde V-\Lambda=E_W^\top A_{II}E_V .$$
The error is quadratic: the product of the primal and the adjoint errors. With primal modes alone (Galerkin, $\tilde V^\top M\tilde V$) the cross term $(A_{II}^\top H+A_{\Gamma I}^\top Q)^\top E_V$ survives, and the error is first order.

**Checked** (`arch_general_form_checks.py` A: upwind advection–diffusion on one $32\times32$ piece, 16 cosine modes per side):

| Péclet (cell Péclet) | identity residual | port-error slope, Petrov–Galerkin | slope, Galerkin | error at $\varepsilon=0.01$, PG / Galerkin | min eigenvalue of the exact $\tfrac12(\Lambda+\Lambda^\top)$, relative |
|---|---|---|---|---|---|
| 0 | $4\times10^{-16}$ | 1.97 | 1.96 | $9.1\times10^{-5}$ / $1.2\times10^{-4}$ | $+1\times10^{-16}$ |
| 20 (0.63) | $4\times10^{-16}$ | 1.98 | 1.12 | $9.1\times10^{-5}$ / $3.0\times10^{-3}$ | $-4.4\times10^{-2}$ |
| 200 (6.25) | $3\times10^{-16}$ | 2.00 | 0.99 | $7.9\times10^{-5}$ / $7.3\times10^{-3}$ | $-1.5\times10^{-1}$ |

**The finding to keep.** Under advection, the *exact* piece's port matrix has an indefinite symmetric part in these port variables: an inflow boundary feeds the piece with whatever it is given. So **positivity is not lost by the network; it was never there.** For non-symmetric physics the tier-0 guarantee splits in two. Quadratic accuracy survives (with adjoint modes). Unique solvability must be checked a posteriori, on the assembled system's smallest singular value, which the host can compute. **[AI Inference]:** passivity can be restored by choosing port variables that count the advected energy at inflow ([[port-algebra-atlas-0.1]]'s `ADVEC` multibond is that bookkeeping). Not tried here.

### 2.4 G1 — convex energies: a nonlinear Ritz method over learned modes

**Intuition.** For many nonlinear problems the solution still minimises an energy; it is just not quadratic. Every field with the right boundary values then has *at least* the minimum energy. So the host can always score a network's proposal honestly, and the best proposal in the network's span is the best that span allows.

**Rigorously.** Let $E$ be the discrete energy of the whole problem, $m$-strongly convex and $L$-smooth on the relevant set, with minimiser $u^\star$ over the conforming fields $U$. For the piece energy $E_i$ and boundary data $g$, let $\Psi_i(g)=\min\{E_i(u):\text{trace}=g\}$.

1. **Upper bound.** $\tilde\Psi_i(g):=\min_a E_i\bigl(\hat u+\hat Hc+\hat Ba\bigr)\ \ge\ \Psi_i(g)$, since every trial field has the exact trace. This is the nonlinear form of $\tilde\Lambda\succeq\Lambda$: a learned piece never stores less energy than the true one at the same boundary state.
2. **Quasi-optimality.** If $\tilde u=\arg\min_{\widetilde W}E$ and $\widetilde W\subset U$, then, because $\nabla E(u^\star)$ vanishes on directions in $U_0$,
$$\tfrac m2\lVert\tilde u-u^\star\rVert^2\le E(\tilde u)-E(u^\star)=\min_{w\in\widetilde W}\bigl[E(w)-E(u^\star)\bigr]\le\tfrac L2\min_{w\in\widetilde W}\lVert w-u^\star\rVert^2,$$
so $\lVert\tilde u-u^\star\rVert\le\sqrt{L/m}\,\min_{w\in\widetilde W}\lVert w-u^\star\rVert$. For a quadratic energy $L=m$ in the energy norm, which gives version 1's Céa constant of one.
3. **Learned subspace correction.** At each outer iteration, let the trial set contain the current iterate and one classical descent direction (here a Jacobi-preconditioned gradient), besides the network's proposals. Then the host's minimiser is at least as good as one classical step from the same point. So the energy decreases monotonically, and the iteration converges at no worse than the classical worst-case rate, **whatever the network returns**. Good proposals make it as fast as Newton's method. This is the nonlinear counterpart of defect correction's "the network changes the speed, not the answer": here the network cannot even slow a step down. It is subspace correction in the sense of Xu (*SIAM Rev.* 34, 1992).

**Checked** (C: $-\Delta u+100\,u^3=0$ on one $32\times32$ piece with Dirichlet data; the "network" proposes Newton's direction with a smooth error of relative size $\varepsilon$ in the Hessian norm; target: relative energy-norm error $10^{-10}$):

| proposal | outer iterations (network calls) | energy monotone | steps worse than one classical step from the same point |
|---|---|---|---|
| Newton's direction, exact | 6 | yes | 0 |
| relative error 0.01 / 0.1 / 0.5 | 7 / 11 / 25 | yes | 0 |
| relative error 1 / 3 | 48 / 154 | yes | 0 |
| a random smooth field (no information) | not converged in 3000 ($2.5\times10^{-8}$) | yes | 0 |
| none (the classical direction alone) | 923 | yes | — |

**The correction this check forced.** The guarantee is per step, against the classical step from the **same** point. It is *not* a promise to beat the classical *trajectory*. The uninformative run never did worse than a classical step at any of its points, yet it ended slower than the plain classical run, whose own trajectory happened to be faster. The honest statement is "never worse than the classical worst case", not "never slower than classical".

**A route that needs no new network at all.** For quasilinear problems whose nonlinearity sits in a coefficient, such as conduction with $\kappa(T)$ or the Kacanov iteration for $-\nabla\cdot(\mu(|\nabla u|)\nabla u)$, each fixed-point step is a *linear* problem with a frozen coefficient field. The Stage 0 expert is trained for exactly that: anisotropic, heterogeneous coefficient fields. So **the linear chart operator already serves this nonlinear class through its coefficient channel**. Each inner solve keeps version 1's machine-checked guarantee, and the outer convergence is classical (iterative linearisations of this kind, the Kacanov method among them, are analysed by Heid & Wihler, *Math. Comp.* 89, 2020). **[AI Inference]:** this is the cheapest nonlinear demonstration available, and the obvious next probe.

### 2.5 G2 — no energy: project the residual, certify it, fall back

When no energy exists, the host minimises the true residual over the trial set in a chosen dual norm. This is the **least-squares Petrov–Galerkin** (LSPG) projection of nonlinear model reduction (Carlberg, Barone & Antil, *J. Comput. Phys.* 330, 2017). Near a regular solution, with $C_s=\lVert DR(u^\star)^{-1}\rVert$ and $C_L$ a local Lipschitz bound,
$$\lVert\tilde u-u^\star\rVert\ \le\ C_s\,\lVert R(\tilde u)\rVert_*\ \le\ C_sC_L\min_{w\in\widetilde W}\lVert w-u^\star\rVert .$$
The middle quantity is **computed by the host at every call**: a certificate of each expert's accuracy at that call, in the expert's own slot. If it exceeds the tolerance, classical Newton–Krylov continues from $\tilde u$ with the network as the initial guess, and the answer is the classical one (defect correction, [[defect-correction-learned-operator]]). For incompressible flow the velocity modes should be stream-function modes, so that every trial field is divergence-free by construction.

**[AI Inference], a measured warning from the record:** at a fluid–structure seam the structure dominates the assembled interface operator (the fluid's block is $1.44\times10^{-5}$ of it on the wing; [[case-study-wing-fsi-atlas-0.1]] §4.2, W137). An interface-level certificate therefore cannot see a fluid expert's error. **The fluid expert's guarantee has to be its own residual certificate**, and G2 provides exactly that.

### 2.6 G3 — explicit hyperbolic steps

The network proposes face fluxes (version 1's conservative format), so conservation is exact. The host blends each proposed flux toward a classical monotone flux, with the largest weight that keeps every cell's update inside the invariant domain (positive density and pressure, a maximum principle). This is the flux-corrected transport and convex-limiting idea (Zalesak, *J. Comput. Phys.* 31, 1979; Guermond & Popov, *SIAM J. Numer. Anal.* 54, 2016). It gives **positivity and conservation for any network output**. Entropy stability is not guaranteed by this alone.

### 2.7 What the network looks like now

| | version 1 | general form |
|---|---|---|
| inputs | $(\log J,\operatorname{Re}\mu,\operatorname{Im}\mu,\log\hat\kappa,\hat f)$, Fo | the same, plus **the current state** $x$ (and fields derived from it, such as a frozen coefficient), **coupling-induced sources** (§4), and, for vector physics, the chart's rotation $\theta$ |
| outputs | $4m$ constraint modes and a particular field | constraint modes **per typed port component**, $r$ interior enrichment modes, a predicted field; optionally adjoint modes (G0′) or face fluxes (G3) |
| calls per solve | one per piece | one per piece **per outer iteration**; outer iterations about as many as Newton's method needs when the proposals are good, never worse than the classical safeguard |
| what the host owns | the energy matrix $M$ | the family's full discrete residual or energy, its Jacobian–vector products, the projection, the certificate and the fallback |

### 2.8 Prior art, and what is left

- **Classical nonlinear decomposition:** nonlinearly preconditioned inexact Newton (ASPIN; Cai & Keyes, *SIAM J. Sci. Comput.* 24, 2002); nonlinear FETI-DP and BDDC (Klawonn, Lanser & Rheinbach, *SIAM J. Sci. Comput.* 36, 2014).
- **Projection-based nonlinear reduced models with decomposition:** DD-LSPG (Hoang, Choi & Carlberg, *CMAME* 384, 2021, arXiv 2007.11835); a domain-decomposition nonlinear-manifold reduced model (Diaz, Choi & Heinkenschloss, *CMAME* 425, 2024, arXiv 2305.15163); LSPG on manifolds learned by convolutional autoencoders (Lee & Carlberg, *J. Comput. Phys.* 404, 2020).
- **Learned local DtN maps of nonlinear problems:** Boutilier, Brenner & Miguez (arXiv 2405.04433), already cited by the paper.
- **Variational structure:** minimising movements for gradient flows (Jordan, Kinderlehrer & Otto, *SIAM J. Math. Anal.* 29, 1998); variational constitutive updates (Ortiz & Stainier, *CMAME* 171, 1999); the Deep Ritz method.

**[AI Inference], what is left after this reading:** the reduced-model works train per problem and per geometry from snapshots. Not found:
- a trial-space generator reusable **across geometries through certified charts**;
- with **exact traces** and **typed multi-field ports**;
- whose **guarantee class is declared from the physics' structure** (G0–G3) and enforced by the host.

The ingredients are published; the combination and its guarantee hierarchy are what the proposal would claim. A fuller search should come before any such claim reaches the paper.

---

## 3. Different fields across one seam: the rocket's air against its skin

### 3.1 What must match is the ports, not the fields

**Intuition.** The air and the skin solve for completely different things: density, velocity, pressure and temperature on one side; displacement, stress and temperature on the other. They need not agree on any of that. They must agree only on what crosses the skin: **mechanical power** (the wall moves the air at the speed the air pushes back on it) and **heat** (the temperature is continuous and the heat leaving one side enters the other). Think of each side of a piece as having plugs, one per kind of energy exchange. Two pieces connect where their plugs match.

**Rigorously, in the vault's vocabulary** ([[port-algebra-atlas-0.1]] §3): a seam carries a direct sum of typed power-conjugate pairs. At the rocket skin these are `MECH` (effort: traction $\mathbf t=\boldsymbol\sigma\mathbf n$; flow: velocity $\mathbf v$) and `THERM` (effort: $T$; flow: entropy flux $q_n/T$), plus `ADVEC` with its passenger list where mass crosses (ablation, transpiration cooling). The interface conditions are a power-conserving junction:
- equal velocities, $\mathbf v_{\text{air}}=\dot{\mathbf d}_{\text{skin}}$;
- opposite tractions;
- equal temperatures;
- opposite heat fluxes.

So the total power through the seam vanishes. The connection rule is unchanged: two experts may share a side for port type $P$ if both declare $P$, with the same passenger list, on that side.

### 3.2 What the chart adds

| item | rule |
|---|---|
| **declarations per side** | each side of a chart expert declares its port list. The skin's outer side: `MECH`, `THERM`. Its inner side, against the tank: `MECH`, `THERM`. A cut inside the skin: the same. The air's wall side: `MECH`, `THERM` |
| **modes per component** | constraint modes per port component: a 2-D thermoelastic skin piece has $(2+1)\times4\times m$ modes, 192 at $m=16$ (fewer where a side carries fewer ports) |
| **one frame** | vectors (velocity, traction) are exchanged in the **seam's physical frame** $(\mathbf n,\boldsymbol\tau)$, never in either chart's reference frame. The host pushes velocities forward with $DF$ and maps tractions with Nanson's formula $\mathbf n\,ds=J\,DF^{-\top}\hat{\mathbf n}\,d\hat s$ |
| **one parametrisation** | port coordinates are transferred between the two charts by an $L^2$ projection in physical arc length (the mortar of version 1), component by component. Efforts map *consistently* and flows *conservatively*, the adjoint pair of [[interface-transfer-theory]] §2.2 |
| **one set of units** | each port's reference scales are declared on the Expert Card, and the host nondimensionalises |

### 3.3 Who imposes what: the seam is one-sided

**Intuition.** At a stiff-against-soft seam, let the stiff side receive the *push* and the soft side receive the *position*. The steel skin is far stiffer, mechanically and thermally, than the air. So the air should be told where the wall is and how hot it is, and the skin should be told the traction and the heat flux the air returns.

**Rigorously.** Per port, with Dirichlet–Neumann exchange the soft side (the smaller DtN map) should receive the Dirichlet data.
- **Thermally** this is Giles's stability result for conjugate heat transfer (*Int. J. Numer. Meth. Fluids* 25, 1997). Optimised Robin coefficients for the same seam are derived by Errera & Chemin (*J. Comput. Phys.* 245, 2013).
- **Mechanically**, the danger is the added-mass instability (Causin, Gerbeau & Nobile, *CMAME* 194, 2005). It is negligible for air against metal, and serious for dense propellant against a thin tank wall. There, Robin exchange (Badia, Nobile & Vergara, *J. Comput. Phys.* 227, 2008) or added-mass partitioned schemes (Banks, Henshaw & Schwendeman, *J. Comput. Phys.* 269, 2014) are the cure.
- **The record measured the one-sidedness** on the wing: the structure's interface block is about $5300\times$ the fluid's in every mode ([[case-study-wing-fsi-atlas-0.1]] §4.2).
- **In wave variables** the impedance does this choice automatically per port: a small $Z$ is Dirichlet-like, a large $Z$ Neumann-like.

### 3.4 Mixed tiers: the condensed skin as the air's boundary condition

**[AI Inference], with classical support.** The skin is linear(ised) thermoelastic, so it is a **tier-0 superelement**: the host has its exact condensed port matrix, block by block in `MECH` and `THERM`. The air is nonlinear (G2). The best transmission condition for a subdomain is its neighbour's DtN map; this is the basis of optimised Schwarz theory (Gander, *SIAM J. Numer. Anal.* 44, 2006). So the host can impose the skin's own condensed response on the air expert as an impedance (Robin) condition per port and mode. Then only the air side iterates, and few exchanges should be needed. Because the seam is one-sided, the skin's block also dominates the assembled interface operator, so it is the natural preconditioner. Not measured.

### 3.5 A moving seam, and two time scales

- **Small deflection** (the usual case for a rocket body): keep the air's geometry fixed and impose the wall motion as a transpiration velocity.
- **Large deflection:** re-chart the air piece from the moved boundary (transfinite interpolation plus the discrete certificate, the fast path of the chart rule). The air expert then needs a **mesh-velocity input channel**, and the update must respect the geometric conservation law.
- **Time scales:** the air's are milliseconds and the skin's thermal ones seconds. Exchange **time-integrated** heat and impulse over each macro step, so that multirate coupling conserves energy exactly, and the air sub-steps inside.

---

## 4. Two single-field experts on one piece: temperature and stress

### 4.1 What the record already established

[[case-study-thermal-strain-atlas-0.1]] (CS-9) split one thermoelastic solver for a rocket shell segment (Al–Li, heated by a gas streak) into a conduction agent and an elasticity agent that **both own the whole region**. Its findings, all measured:

- thermal strain is a **bond of co-dimension 0** between agents whose free energies **do not add**: the elastic energy carries a cross term $-\mathbf u^\top G\Delta T$ that belongs to neither agent. So it is a bond and **not a port**;
- the reverse coupling (the Biot heating term) is exactly $G^\top$, the same matrix transposed, which is what makes the pair a bond;
- the surface-only equivalent load is wrong by three orders of magnitude, because the body part $G_{\text{body}}$ (about 10% of the load) cancels a large bending component;
- a lagged split costs $7.1\times10^{-3}$ relative stress per macro step, first order in the step;
- the two-way fixed point converges in 7 iterations at coupling number $\delta=9.075\times10^{-3}$;
- routing one agent's state into the other as a `GlobalField` is exact but bypasses every seam check, and the compiler now **refuses** it (W117).

So the question is not whether the two experts can be coupled, but **what object owns the coupling, and with what guarantee**.

### 4.2 The proposal: a host-owned co-located coupling form

**Intuition.** Both experts sit on the same chart and grid, the same sheet of graph paper, so a field from one is already on the other's grid: no interpolation, no seam. Neither expert knows the other exists. The **host** owns the coupling law (thermal expansion, thermoelastic heating) exactly as it owns boundary conditions, and translates one expert's output into the other's *ordinary* inputs. A temperature becomes a body force plus a traction on the mechanical expert, and a strain rate becomes a heat source on the thermal expert. Then the host solves the coupled problem in the space the two experts' fields span.

**Rigorously** (linear thermoelasticity, one backward-Euler step, $\theta=T-T_0$). With $K_u$ the stiffness, $G$ the thermal-load operator (body *and* surface parts, as CS-9 requires), $C_\theta$ and $K_\theta$ the heat capacity and conduction matrices, scale the energy equation by $-\Delta t/T_0$:
$$\begin{pmatrix}K_u&-G\\-G^\top&-N\end{pmatrix}\begin{pmatrix}\mathbf u\\\theta\end{pmatrix}=\begin{pmatrix}\mathbf f\\ \mathbf r\end{pmatrix},\qquad N=\frac{C_\theta+\Delta t\,K_\theta}{T_0}\succ0 .$$
This matrix is **symmetric quasi-definite**: a positive definite block, a negative definite block, and reciprocal coupling. Every such matrix is nonsingular (Vanderbei, *SIAM J. Optim.* 5, 1995). Let the mechanical expert supply a trial space $\tilde U$ and the thermal expert $\tilde\Theta$. The host orthonormalises each in its own energy ($K_u$, $N$) and solves the Galerkin problem on $\tilde U\times\tilde\Theta$. Define the **dimensionless coupling strength**
$$\gamma=\bigl\lVert N^{-1/2}G^\top K_u^{-1/2}\bigr\rVert_2 .$$

> **Proposition (two experts, one piece) — ours, elementary.** For any trial spaces $\tilde U$, $\tilde\Theta$:
> 1. the reduced matrix has exactly $\dim\tilde U$ positive and $\dim\tilde\Theta$ negative eigenvalues, all of modulus in $[1,\sqrt{1+\gamma^2}]$: its condition number is at most $\sqrt{1+\gamma^2}$;
> 2. the Galerkin solution is within $1+\gamma$ of the best approximation in $\tilde U\times\tilde\Theta$, in the block energy norm $\lVert(\mathbf u,\theta)\rVert^2=\mathbf u^\top K_u\mathbf u+\theta^\top N\theta$;
> 3. a staggered (block Gauss–Seidel) sweep with exact single-field solves contracts the error by $\gamma^2$;
> 4. $\gamma^2\le\delta=\beta^2T_0/\bigl((\lambda+2\mu)\rho c\bigr)$, the classical thermoelastic coupling number.
>
> *Proof sketch.*
> 1. In energy-orthonormal bases the reduced matrix is $\begin{pmatrix}I&C^\top\\C&-I\end{pmatrix}$, with $\lVert C\rVert\le\gamma$. Its square is $\operatorname{diag}(I+C^\top C,\ I+CC^\top)$.
> 2. The form is $T$-coercive with $T=\operatorname{diag}(I,-I)$, which maps the product trial space to itself (Ciarlet, *Comput. Math. Appl.* 64, 2012). Its inf-sup constant is 1 and its continuity constant $1+\gamma$, so Babuska's theorem with Xu & Zikatanov's sharpening (*Numer. Math.* 94, 2003) gives the constant $1+\gamma$.
> 3. The block Gauss–Seidel iteration matrix has spectral radius $\rho(N^{-1}G^\top K_u^{-1}G)=\gamma^2$.
> 4. Fourier analysis of the continuum operators: the bound is attained at long wavelengths, where conduction is negligible over one step.

**Checked** (B: a 1-D steel bar, 200 cells, one step at $\mathrm{Fo}=0.05$, surface heating $10^5\,\mathrm{W\,m^{-2}}$; each "expert" is an exact single-field solve plus a smooth error of relative size $\varepsilon$ in that field's energy norm):

| quantity | measured | the statement |
|---|---|---|
| $\gamma$, $\gamma^2$; $\delta$ | 0.1042, 0.0108; 0.0108 | $\gamma^2\le\delta$, attained |
| staggered-sweep error ratios | 0.0087, 0.0099, 0.0104, 0.0106, 0.0107 | $\to\gamma^2$ |
| eigenvalue moduli of the reduced matrix, over every row including $\varepsilon=1$ | between 1.0000 and 1.0050 | in $[1,\sqrt{1+\gamma^2}]=[1,1.0054]$ |
| inertia | $(\dim\tilde U,\dim\tilde\Theta)$ in every row | quasi-definite |
| Galerkin error / best approximation | at most 1.003 | at most $1+\gamma=1.104$ |

For the record's Al–Li shell, $\delta=9.1\times10^{-3}$ (with the plane dilatational modulus). If part 4 carries over to plane stress by the same Fourier argument, which is expected but not checked here, the bound gives a condition number of at most $1.0045$ and a quasi-optimality constant of at most $1.095$. **For metals the coupled problem is nearly as well posed as two separate ones**, whatever the experts return.

### 4.3 Cross-enrichment: what the two trial spaces must contain

The product of the two experts' *uncoupled* answers cannot represent the coupled solution. With no mechanical load, the mechanical expert's uncoupled answer is zero, and the thermal-expansion displacement is missing entirely. So each expert is also asked for its response to the coupling load of the other's latest field:
- round $k$ adds $K_u^{-1}G\,\theta_{k-1}$ (the mechanical response to a thermal load);
- and $-N^{-1}G^\top\mathbf u_k$ (the thermal response to mechanical heating).

This is the Neumann series of the coupling, and each round gains a factor $\gamma^2$. **Checked**, with exact experts:

| rounds | displacement error (energy norm) | temperature error |
|---|---|---|
| 0 | 100% (missing) | $6.7\times10^{-3}$ |
| 1 | $2.8\times10^{-3}$ | $1.6\times10^{-5}$ |
| 2 | $8.5\times10^{-6}$ | $4.6\times10^{-8}$ |

With experts of 1% error the floor is about 1% after two rounds, which is the experts' own accuracy. Each round costs one extra batched call per expert per piece, and at a fixed step the result is reused.

### 4.4 Staggering, the fallback, and when it is stable

Black-box or nonlinear experts can still be **staggered**: thermal step, then mechanical, and so on, iterated to convergence at the rate $\gamma^2$ per sweep. CS-9 measured the classical version. A **non-iterated** split, one sweep per time step, adds the splitting error CS-9 measured, first order in the step. Its stability depends on *which* split is used. The isothermal split (mechanics at frozen temperature, then conduction) is only conditionally stable. The adiabatic split (mechanics at frozen entropy, then conduction) is unconditionally stable (Armero & Simo, *Int. J. Numer. Meth. Eng.* 35, 1992). **[AI Inference]:** the adiabatic split's mechanical phase uses the adiabatic moduli, $\lambda+\beta^2T_0/(\rho c)$ in place of $\lambda$. The host passes these through the mechanical expert's coefficient channel, so a provably stable split needs **no retraining**, provided the expert's envelope covers those moduli.

### 4.5 Which couplings have this structure

**[AI Inference], a generalisation to check.** The quasi-definite form came from a **reciprocal** coupling: the forward operator is $G$, the reverse $G^\top$. Reciprocity is Onsager's, and it holds whenever the coupled problem derives from a joint thermodynamic potential (Biot, *J. Appl. Phys.* 27, 1956, for thermoelasticity). The same block structure is standard for:
- **quasi-static poroelasticity:** displacement against pore pressure;
- **piezoelectricity:** displacement against electric potential, from the electric enthalpy.

So the proposition should extend to those families with their own $\gamma$. **Non-reciprocal** couplings, such as temperature carried by a flow, are one-way or non-symmetric. One-way couplings make the system block-triangular: solve in sequence, and each block keeps its own guarantee. General non-symmetric couplings fall back to G2's a-posteriori certificate.

### 4.6 What the standard must require, and what separately trained experts cannot do

- **Coupling channels in training.** Each single-field expert must have been trained over the **source and coefficient fields that coupling will feed it**: thermal-load body forces and tractions with the boundary layers of a heated wall; Biot heat sources; temperature-dependent moduli. The Expert Card declares this **coupling envelope**, and the host refuses an input outside it.
- **The coupling is declared, not smuggled.** The host-owned form is a third kind of object, beside a port and a global field: a declared **co-located bond** naming its two experts, its law ($\beta$, $T_0$, $G$ in chart coordinates) and its form (quasi-definite, block-triangular, or general). It answers CS-9's *what to build instead* and W116's missing slot. The splitting error stays declared when the fallback of §4.4 is used.
- **Not reachable by two separate experts:** phenomena that live in the *joint* nonlinearity, such as thermal buckling (a bifurcation of the coupled system) or strongly coupled phase change. Newton's method with both experts' Jacobian–vector products can track these in principle, but only if their tangents are accurate in the coupled regime. For these a joint expert may be needed.

---

## 5. One principle under both questions: passivity and power-conserving interconnection

**Intuition.** A system cannot run away if every part only *stores or dissipates* energy and every connection only *passes energy on*.

**Rigorously.** A system is passive if its stored energy grows by no more than the power supplied at its ports (Willems, *Arch. Rational Mech. Anal.* 45, 1972). Passive systems interconnected through power-conserving junctions form a passive system; this is the basic composition theorem of port-Hamiltonian theory (van der Schaft & Jeltsema, *Found. Trends Syst. Control* 1, 2014). In this design:
- **a tier-0 piece is passive by construction**, because its Gram port matrix is PSD;
- **a seam junction is power-conserving by the connection rule** (§3.1);
- **the thermoelastic bond is lossless** in the right variables, because it is reciprocal.

The quasi-definite proposition is this principle made quantitative for one bond. **[AI Inference]:** for nonlinear dissipative experts the corresponding architectural route is a network whose dynamics are written in port-Hamiltonian or GENERIC form (Grmela & Oettinger, *Phys. Rev. E* 56, 1997), which builds the first and second laws in. Structure-preserving networks of this kind exist (Hernandez et al., *J. Comput. Phys.* 426, 2021). That is the research frontier of the general form, not a design ready to build.

---

## 6. What this would change, and the decisions for the owner

| item | the proposal |
|---|---|
| the core of the design | the general form (§2), with version 1's tier 0 as its linear corner; the tiers become guarantee classes G0–G3 declared from the physics' structure |
| the network | add the current state and coupling-induced sources as inputs; add interior enrichment modes, optional adjoint modes, and modes per typed port component as outputs |
| the host | owns the family's residual or energy, the projection, the certificate and the fallback |
| seams | per-side typed port lists, the seam's physical frame, mortar per component, impedance per port |
| co-located fields | a new declared object, the co-located bond, with the quasi-definite guarantee where the coupling is reciprocal and staggering as the declared fallback |
| the first nonlinear demonstration | conduction with $\kappa(T)$ and radiating boundaries, through the frozen-coefficient route (§2.4): no new network, version 1's checked guarantee per inner solve |
| the rocket demonstration | the skin as tier-0 thermoelastic superelements with a co-located bond; the air as a G2 expert (or a classical solver at first) behind `MECH` and `THERM` with the skin's condensed impedance |

**To decide, in discussion:** whether the general form becomes the paper's core (the paper is public since the launch, so a revision goes through the website chat as well); the affine trial set against a learned manifold; whether the co-located bond enters the port algebra's vocabulary; and which nonlinear family is Stage 1's first.

---

## 7. For the proofs chat

**The owner relays this.** Candidates, all finite-dimensional, in the style of the existing `GramPort.lean`:

1. **G0′:** $\tilde W^\top M\tilde V-\Lambda=E_W^\top A_{II}E_V$, under the defining equations of the primal and adjoint modes. Tier A, a direct generalisation of `gram_sub_gram_posSemidef` without symmetry.
2. **The two-experts proposition** of §4.2, parts 1–3, for a symmetric quasi-definite matrix and arbitrary subspaces. Tier A: linear algebra only. Part 4 is a continuum estimate and should be **cited**, not formalised.
3. **G1, parts 1 and 3:** the energy upper bound (immediate), and the subspace-correction step (the decrease is at least the classical step's). Tier B: it needs Mathlib's convexity library.
4. **Nothing proved changes.** `Atlas.tier0_headline` and `Atlas.tier2_headline` keep their statements; G0 *is* tier 0.

---

## 8. What this page does not do

1. No network was trained or run. Every "learned" output in the checks is an exact solve plus a perturbation.
2. G2 and G3 are assembled from cited results; nothing here checks them.
3. The thermoelastic check is one-dimensional. The 2-D shell of CS-9 is the next place to run it.
4. §3.4's condensed-impedance coupling of a tier-0 skin to a nonlinear air expert is not measured.

---

## See Also

- [[chart-operator-architecture]] — version 1, whose tier 0 is G0 here
- [[chart-operator-design-decisions]] — the decisions this note would revisit
- [[port-algebra-atlas-0.1]] — the port types and the connection rule of §3
- [[case-study-thermal-strain-atlas-0.1]] — the co-located split §4 builds on
- [[case-study-wing-fsi-atlas-0.1]] — the one-sided fluid–structure seam of §3.3
- [[interface-transfer-theory]] — consistent and conservative transfer as one adjoint pair
- [[defect-correction-learned-operator]] — the fallback of G2
- [[formal-proofs-plan]] — where the candidates of §7 would go
