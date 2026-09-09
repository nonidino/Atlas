# Atlas and Standard Domain-Decomposition Theory

**Type:** Concept page — framework theory / reading companion, **version-independent** (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-30, as a map from Atlas's coupling machinery onto the four classical domain-decomposition (DD) families; **extended 2026-09-07 (Tier 33) with a fifth -- nonlocal DD -- and with the measured verdict on the fourth**, which section 5 had left open and [[interaction-horizon]] closes, for a reader working through Toselli & Widlund, *Domain Decomposition Methods — Algorithms and Theory*. It states where Atlas sits in each family, which classical convergence and optimality theorems stop applying when a subdomain solver is a frozen neural operator, and why. Cross-linked from [[prior-art-and-novelty-atlas-0.1]] and [[general-coupling-scheme]].
**Related:** [[general-coupling-scheme]] · [[probed-dtn-coupling]] · [[composition-error-theory]] · [[master-error-bound]] · [[schwarz-iteration-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[prior-art-and-novelty-atlas-0.1]] · [[interface-transfer-theory]] · [[temporal-error-accumulation]] · [[gap-worklist]]

> **The one-line version.** Atlas is not a new coupling paradigm — every mechanism in it has a classical home, and this page names the home. Its as-built wind-farm coupling is **one-level restricted-additive overlapping Schwarz** with a single sweep (Schwarz waveform relaxation, window $W=1$); its target is **non-overlapping substructuring** with the Steklov–Poincaré operator assembled by black-box probing ([[probed-dtn-coupling]]); its interface-matching move toward a declared common space with prolongations ([[interface-transfer-theory]]) is a **mortar method** with a measured rather than proven inf-sup constant; the fourth classical family — **optimization-based / virtual-control DD** (Lions, Glowinski, Gunzburger) — was flagged here as a likely gap and has since been **measured and refused** ([[interaction-horizon]] §6: minimising an agreement objective moves two biased agents' errors into cancellation, not toward the truth); and §6 now carries the family the original four all exclude by assumption — **nonlocal DD**, where a finite interaction *horizon* replaces the stencil throughout the estimates, which is the classical home of an operator whose receptive field is dense. The reason none of the classical *theorems* transfer unchanged is a single property of the subdomain solver: a frozen neural operator has no consistency order, no CFL condition, no statable Lipschitz bound, and an error that does not shrink when the step shrinks ([[prior-art-and-novelty-atlas-0.1]] §2.1).

---

# 1. The abstract Schwarz framework, and the hypothesis that fails

Toselli & Widlund organise the whole theory around an **abstract Schwarz framework** (their Ch. 2): a preconditioner assembled from a coarse space $V_0$ and local subspaces $\{V_i\}$, with a condition-number bound of the shape

$$\kappa\bigl(M^{-1}A\bigr) \;\le\; C\,\bigl(1+\rho(\mathcal E)\bigr)\,\omega,$$

where $\omega$ bounds the local solvers in the energy norm, $\rho(\mathcal E)$ is a strengthened-Cauchy–Schwarz constant coupling the subspaces (a colouring/overlap quantity), and $C$ comes from the existence of a **stable decomposition** of any function into subspace pieces with controlled total energy. Every classical result quoted below — the $H^{-2}$ of one-level Schwarz, the polylogarithmic FETI-DP/BDDC bound, mortar optimality — is this framework specialised to a particular $V_i$, $\omega$ and $C$.

**Two of the three inputs assume the local solver is an exact solve of a coercive, variationally-posed discrete problem.** That is the assumption a frozen neural expert breaks:

- $\omega$ is a bound on the local-solver error measured in an energy norm the operator is exact in. A neural operator is not known to be exact in any norm; it has an *independent accuracy* $\varepsilon_i$ on its own validation distribution and a *consistency defect* $\tau_i$ under composition-imposed data, and [[composition-error-theory]] §3.2 is the argument that these two numbers are pulled apart by composition. There is no $\omega$.
- $\rho(\mathcal E)$ is geometric — overlap and colouring — and **survives** unchanged.
- $C$ needs a bounded, stable partition of unity. Atlas has one, with the identity $\sum_i R_i^{\!\top}\chi_i R_i = I$ ([[master-error-bound]] §1), and *adds* a condition the classical theory does not need (§2.1).

So the honest reading of Toselli & Widlund for Atlas is: **the framework's geometry carries over; its analysis of the local solver does not.** §2–§5 place Atlas in each of the four families and §6 in a fifth the text does not cover; §7 is the list of theorems that go with the analysis rather than the geometry.

**One hypothesis is common to all four classical families and it is the one Atlas breaks**: the subdomain operator is *local* — its influence is carried by a stencil of finite radius, so an overlap wider than that radius decouples the pieces. §6's family is the one that drops it.

---

# 2. Overlapping Schwarz — where Atlas sits

**Atlas's as-built wind-farm coupling is one-level restricted-additive overlapping Schwarz, run for a single sweep.** In the seven-parameter scheme of [[general-coupling-scheme]] §1.1 it is

$$\Sigma_{\text{as built}} = \bigl(\ \texttt{overlapping}(0.5D),\ \ \texttt{dirichlet},\ \ \texttt{additive},\ \ \texttt{richardson},\ \ \texttt{one},\ \ W{=}1,\ \ k{=}1\ \bigr).$$

Mapping term by term onto Toselli & Widlund:

| Atlas | classical name |
|---|---|
| `overlapping(0.5D)` + assembly $\mathcal A$ with $\sum_i R_i^{\!\top}\chi_i R_i = I$ | **restricted additive Schwarz** (RAS, Cai–Sarkis): the partition of unity $\chi_i$ is the RAS weighting that makes the operator non-symmetric but removes the overlap double-count |
| `additive` ordering | additive Schwarz (all subdomains from the same iterate); **never multiplicative** — R5 of [[general-coupling-scheme]], structural: multiplicative ordering reinstates graph-cycle path dependence and breaks the exactness of [[symmetry-averaging-atlas-0.1]] |
| `dirichlet` ring | first-order (Dirichlet) transmission — the far-left rung of [[composition-error-theory]] §4.0's ladder |
| `one` level | one-level method: $\kappa\bigl(M_{\text{AS},1}^{-1}A\bigr)\le C\,H^{-2}(1+H/\delta)$, information advancing one overlap-connected neighbour per iteration |
| $W=1$, $k=1$ | a **single sweep**, not an iterated fixed point |

## 2.1 The single sweep is Schwarz waveform relaxation, converged in one step by geometry

Within one macro-step the subdomain problem is an evolution over $\Delta t$, so the relevant classical theory is **Schwarz waveform relaxation** (SWR; Gander, Gander–Halpern, Gander–Stuart), not elliptic Schwarz. Its governing result for advection-dominated problems is favourable and sharp: if no signal can cross the overlap within the time window,

$$U_\infty\,\Delta t_{\text{macro}} \;<\; \delta,$$

classical SWR converges in a **finite** number of iterations, and in the limiting case **one** ([[schwarz-iteration-atlas-0.1]] §5.1). Atlas's tile overlap is $0.5D$ against an advective distance $U_\infty\Delta t = 0.25D$ — a factor of two inside the condition — and the one instantaneous mechanism that would defeat a single sweep, elliptic pressure coupling, is solved globally rather than per window. **That is why the as-built coupling exchanges once and does not iterate**: for this configuration a single sweep already is (nearly) the SWR fixed point. [[schwarz-iteration-atlas-0.1]] refused an explicit iteration stage (S1) for the separate reason that under *periodic* windows the sweep is the identity in the iteration index; with a real boundary channel (`reference.WindowNS`) iterating is well-defined SWR and buys the finite-step result above.

## 2.2 No classical coarse space, and why the usual fix is unavailable

The one-level bound carries an $H^{-2}$ that degrades as subdomains are added, and the classical remedy is a **coarse space** ([[schwarz-iteration-atlas-0.1]] §3.2). Atlas cannot use the textbook construction: a geometric coarse space means enlarging the subdomain size $L$, and the frozen expert's regime law $\mathrm{Re}_{\text{eff}} = T_s/(\nu_p L^2)$ makes that *change the physics rather than the resolution* (§S3 of that page). Spectrally-adapted coarse spaces (GenEO, in Dolean–Jolivet–Nataf) assume access to local eigenproblems of the subdomain stiffness matrix, which a neural operator does not expose. **The substitute is the probed Schur complement of §3**, which is dense across the whole interface and therefore couples inlet to outlet in one solve — it belongs to family 2 below, not to overlapping Schwarz.

## 2.3 The transmission ladder is the optimized-Schwarz ladder

[[composition-error-theory]] §4.0's ladder

$$\text{Dirichlet}\prec\text{Neumann}\prec\text{Dirichlet–Neumann}\prec\text{Robin}(\alpha)\prec\text{optimized Robin}\prec\text{Ventcell}\prec\Lambda_i$$

is exactly the **optimized-Schwarz** hierarchy (Gander 2006; Dolean–Jolivet–Nataf Ch. 2). The optimal Robin coefficient is the symbol of the exact DtN operator, classically obtained by Fourier-analysing the PDE. Atlas sits on the far-left rung and, crucially, **can only climb it by measurement** — a frozen checkpoint has no symbol to analyse (§7, item 3; [[probed-dtn-coupling]] §4.4). **Climbed one rung 2026-09-09** ([[case-study-neural-interface-atlas-0.1]] §7): `probe.alpha_star` is the measured symbol mode by mode and had never been used as a transmission condition; used as one it must be **damped**, because it cuts $\kappa$ from $22.7$ to $7.3$ and still diverges — Richardson needs the preconditioned spectrum inside $(0,2)$ and $\rho(D^{-1}S)=2.98$. At $\omega=1/\rho$ it is worth $3.5\times$.

## 2.4 What the overlapping branch of the error bound looks like

On the overlapping branch there is no interface solve, so the transmission-infidelity term $\sigma$ of [[master-error-bound]] carries **no $1/\beta$ amplifier**. Instead ([[master-error-bound]] §4.1)

$$\sigma \;\le\; C_\mu\,\Pi\,\lVert\delta\lambda\rVert, \qquad \Pi = \max_j\sum_i \chi_{ij}\,\mathbf 1\!\left[\,b_i(j)\le d_i\,\right],$$

the **contaminated partition-of-unity weight**: the assembly weight given to cells a stale artificial-boundary datum can reach within one exchange. Overlap width is a *precondition* ($\Pi\to1$ once $\delta<$ the domain of dependence); the partition of unity is the *mechanism* ($\Pi\to0$, and then $\sigma$ is zero, not small, because the blend never reads a contaminated cell). This is a quantitative form of the classical intuition that a wider overlap and a smoother PoU both help — with the measured finding that the PoU ramp does far more work than the halo width.

## 2.5 One condition Atlas needs that Toselli & Widlund do not state

Classical overlapping Schwarz iterates a partition-of-unity blend to a fixed point, and the blend weights only need to be a partition of unity. Atlas **blends after a single sweep**, and [[master-error-bound]] §4.1 / [[gap-worklist]] W50 (rule L6/C1) records a condition that is then necessary *and* sufficient: the partition of unity must be **non-negative and sum to one**, $\chi_i\ge0$, $\sum_i\chi_i = 1$. It comes from a cellwise bias–variance identity

$$\bigl|\mathcal A(u) - u^\star\bigr|^2 \;=\; \sum_i \chi_i\,\bigl|u_i - u^\star\bigr|^2 \;-\; V_\chi,$$

whose variance term $V_\chi$ is non-negative for all data exactly when $\chi\ge0$; a signed partition of unity that still sums to one can make the blend *less* accurate than the worst local solve it blends. This is a genuine addition, forced by the single-sweep design rather than imported.

**Where Atlas sits (overlapping Schwarz):** one-level, restricted-additive, Dirichlet-transmission, single-sweep SWR; no classical coarse space (unavailable for a frozen expert); optimized-Schwarz transmission conditions reachable only by probing; the error contribution set by a partition-of-unity contamination weight, with a convexity condition the iterated method does not require.

---

# 3. Steklov–Poincaré / substructuring (Schur complement) — where Atlas sits

The exact non-overlapping interface condition is not "match the values" — it is the **Steklov–Poincaré operator** $\Lambda_i:\ g\mapsto \partial_n\mathcal S_i[g]\big|_{\partial\Omega_i}$, and the coupled solution is the interface trace $\lambda$ at which the fluxes balance:

$$\mathcal G(\lambda) \;:=\; \sum_i \Lambda_i\lambda - \chi \;=\; 0, \qquad S := \sum_i \Lambda_i \ \ \text{(the Schur complement)}.$$

Solving $S\lambda = \chi$ reproduces the monolithic solution **exactly** — this is Toselli & Widlund Ch. 4, and Quarteroni & Valli Ch. 1–2, and it is the one case in which [[composition-error-theory]]'s worst-agent conjecture is a theorem ([[master-error-bound]] §8, row 2).

**Atlas's `probed-DtN` rung is substructuring with the Schur complement built by black-box probing** ([[probed-dtn-coupling]] §2). Classically $S$ is formed by eliminating interior degrees of freedom from a sparse stiffness matrix; Atlas forms each block $S_i$ by $m+1$ Dirichlet solves — impose a boundary basis function $\psi_k$, read the returned flux, subtract the zero-probe. This is the only route to $S$ when the interior stiffness matrix does not exist as an object, and it is what rule R2 of [[general-coupling-scheme]] is about: `probed-DtN` needs only that the expert *accepts a Dirichlet ring*, because the higher rung is assembled **outside** the expert.

Placing it more finely against Ch. 4–6 of Toselli & Widlund:

- **Primal substructuring.** Atlas assembles $S$ and solves it directly (`direct-schur` accelerator), which is a primal method — closest to the classical Schur-complement / iterative-substructuring family, not to the dual (FETI) family. The Neumann–Neumann / balancing preconditioners of Ch. 6.2 are the classical way to precondition $S$ iteratively; Atlas's $S$ is small enough (a few hundred DOF) to invert directly, so it needs no preconditioner.
- **The probed $S$ *is* the missing coarse space.** Because $S$ is dense across the entire interface, solving $S\lambda=\chi$ transmits information inlet-to-outlet in one solve, not in the $k_{\min}=5$ sweeps (16 to cross the domain) that the one-level $H^{-2}$ mechanism would need ([[probed-dtn-coupling]] §3). Atlas's 8 declared agents over 124 fine tiles were always a two-level $H/h$ hierarchy; the probed Schur complement is the coarse *solve* it lacked, and it never enlarges $L$. **Measured 2026-09-09** ([[case-study-neural-interface-atlas-0.1]] §7): priced in expert calls *including its own 34-call probe*, solving $S\lambda=\chi$ costs **44 calls against one-level Richardson's 594** at a tolerance of $10^{-6}\lVert r(0)\rVert$ — $13.5\times$, on a seam where the one-level contraction is $0.97$ per sweep.
- **Nonlinear substructuring.** For Navier–Stokes and for a neural operator $\mathcal E_i$ is not affine in $\lambda$, so a probe at state $\lambda^{(k)}$ recovers the **tangent map** $D\Lambda_i(\lambda^{(k)})$ and the method is **Newton–Krylov–Schur** — the family the nonlinear variants of FETI-DP and BDDC sit in ([[probed-dtn-coupling]] §7). Every quantity read off $S$ ($\beta$, $\kappa$, the passivity spectrum, the composability index $\Xi$) is then local to that linearization.
- **The compatibility / null-space condition.** Incompressibility leaves a rank-one null space in $S$ at a fluid–fluid seam ($\oint_\Gamma\lambda\cdot n = 0$) — this is the compatibility condition of every Neumann-based substructuring method (Toselli & Widlund Ch. 6.2.1). Atlas turns it into a free correctness check, and the measurement showed the predicted dimension depends on where the elliptic solve lives ($n_0=1$ with incompressibility inside $\Lambda_i$, $n_0=0$ with it in the composition layer; [[probed-dtn-coupling]] §2.1).
- **Cross-points are an open gap.** A non-overlapping decomposition creates points where three or more subdomains meet, where the transmission conditions are not independent and naive Robin conditions are ill-posed. FETI-DP and BDDC exist precisely to handle this, by making corner degrees of freedom **primal** (continuous by construction). Atlas has no cross-point rule yet — gap G1, tracked as [[gap-worklist]] W6 / W64 — and its as-built *overlapping*-with-partition-of-unity tiling avoids the difficulty structurally, which is why 124 tiles meeting four-at-a-corner have never triggered it. **Moving to `probed-DtN` moves to non-overlapping and therefore introduces a difficulty the present scheme does not have** ([[general-coupling-scheme]] §5).

> **The measured caveat that scopes this whole family for Atlas.** For an **explicit one-step map**, the flux-balance equation $S\lambda=\chi$ is a *steady* condition that the true trace does not satisfy: substituting the reference solution's own trace leaves the interface residual essentially unchanged, and driving it to zero overshoots the true trace, worsening as $\Delta t$ is refined ([[probed-dtn-coupling]] §2.3; [[tier0-measurements]]). Rule R2b now gates `probed-DtN` on `time_discretization`. So Atlas *aspires* to substructuring at this rung, and it is the right family for an **implicit** expert, but as of the Tier 0 measurements the explicit wind-farm checkpoint stays on the overlapping halo of §2. The Schur route's value proposition — one solve couples the whole graph — is real; its interface *equation* is only correct when the local solver poses a boundary-value problem rather than an initial–boundary-value one.

**Where Atlas sits (substructuring):** primal iterative substructuring with a probed (black-box) Schur complement; direct interface solve; Newton–Krylov–Schur under nonlinearity; the Neumann compatibility condition used as a correctness check; **no cross-point rule** (the FETI-DP/BDDC corner-primal construction is an acknowledged missing piece); valid for implicit experts, refused for explicit one-step maps.

---

# 4. Lagrange-multiplier / mortar methods — where Atlas sits

Mortar element methods (Bernardi–Maday–Patera; Toselli & Widlund Ch. 1.3.5 and the constrained/FETI material of Ch. 6) enforce interface continuity **weakly**, through a Lagrange multiplier $\lambda$ on an interface space $M_\Gamma$:

$$\int_\Gamma \lambda\,(u_A - u_B)\,ds = 0 \qquad \forall\,\lambda\in M_\Gamma,$$

and when $M_\Gamma$ satisfies a discrete **inf-sup (LBB) condition** with constant $\beta>0$, the method is **optimal**: the global error is bounded by the sum of the subdomains' own best-approximation errors, with a constant independent of the decomposition. **That is the worst-agent bound, and $\beta$ is the hypothesis it needs** ([[composition-error-theory]] §4.2).

- **Atlas today enforces the interface strongly and pointwise.** The port algebra's connection rule is $e_A = e_B$, $f_A = -f_B$ *pointwise* ([[port-algebra-atlas-0.1]] §4). Requiring two inexact operators to match pointwise in both value and flux is the **over-constrained end** of the ladder; locking (too many constraints, $\beta$ collapses) and spurious interface modes (too few) are the classical symptoms. Weak enforcement is the well-posed version of the same idea, not a relaxation of it.
- **Atlas's non-conforming-transfer construction is a mortar method.** [[interface-transfer-theory]] replaces the connection rule's *"interface geometries coincide"* with: declare a **common interface space $M$** per seam and a stable prolongation $P_i:M\to V_i$ per side, and the reduction is *forced* to be $P_i^\ast$ because the port is a power bond and any other choice leaks power. $M$ is a mortar multiplier space; $\tilde\Lambda_i^M = P_i^\ast\Lambda_i P_i$ is a Galerkin (mortar) compression of the DtN operator; and the probe basis and the multiplier space are **the same space**, so mortar costs no extra probes ([[probed-dtn-coupling]] §2.2). Geometric coincidence is the special case $M = V_A = V_B$, $P_i = \mathrm{id}$.
- **Primal or dual?** FETI is the *dual* (Lagrange-multiplier) substructuring method; Atlas's construction is a hybrid — primal in that it assembles and inverts $S$, mortar in that the interface unknowns are a declared coarse space $M$ with $\dim M < \dim V_i$, chosen by the expert's effective spectral resolution rather than by a finite-element trace space.
- **The inf-sup constant is measured, not proven.** $\beta = \sigma_{\min}\bigl(\tilde\Lambda\big|_{\text{constrained subspace}}\bigr)$ and $\kappa(S) = \sigma_{\max}/\sigma_{\min}$ are read off the assembled matrix ([[master-error-bound]] §4; [[probed-dtn-coupling]] §4.3). The classical routes to a *provable* $\beta$ — a macroelement argument, a Fortin operator — need a finite-element trial space with an approximation theory, which a frozen operator does not have. [[composition-error-theory]] §4.2 marks the multiplier space for a neural expert as a genuinely open design question; the tractable answer adopted in [[probed-dtn-coupling]] §2.2 is a Fourier interface basis truncated at the expert's measured effective resolution (`[AI Inference]` there), with $\beta$ then obtained numerically.

**Where Atlas sits (mortar):** pointwise (strong) matching today, at the over-constrained end of the ladder; a specified generalization to a declared common interface space that *is* a mortar formulation, with the inf-sup constant obtained by measurement because the frozen operator has no approximation theory; the multiplier-space choice is an open design parameter.

---

# 5. Optimization-based / virtual-control DD — where Atlas sits: measured, and refused for this class of agent

The fourth classical family poses the coupled problem as a **minimization**. Instead of imposing a transmission condition, one introduces the interface traces (or fluxes) as **virtual control variables** and minimises a functional measuring the interface mismatch:

$$\min_{\lambda}\ J(\lambda), \qquad J(\lambda) = \tfrac12\bigl\lVert u_A(\lambda) - u_B(\lambda)\bigr\rVert_\Gamma^2 \quad\text{(or a least-squares interface-residual functional)},$$

solved through the optimality system — i.e. via the **adjoint** of the subdomain solves. The minimiser makes the transmission conditions hold. The lineage: J.-L. Lions on the Schwarz method as a control problem; Glowinski and co-workers on least-squares / virtual-control formulations; Gunzburger, Lee & Peterson, "An optimization-based domain decomposition method"; and, for coupling subdomains that solve *different* equations, Gervasio, Lions & Quarteroni, "Heterogeneous coupling by virtual control methods" (*Numer. Math.* 2001).

**Atlas had no page on this family until 2026-09-06**, and the reason it was worth writing one is specific:

- Atlas already claims **end-to-end differentiability through the coupling** — the adjoint — as a distinguishing property against preCICE ([[prior-art-and-novelty-atlas-0.1]] §2 verdict table and §1 comparison table). A virtual-control formulation is built out of exactly that adjoint.
- Both [[composition-error-theory]] §4.4 and [[probed-dtn-coupling]] §5.2 note the differentiability claim is currently **spent on nothing**: the adjoint is invoked only for dual-weighted-residual error *localization*, never for the coupling solve itself.

**[AI Inference]:** the interface residual $\mathcal G(\lambda)$ of [[general-coupling-scheme]] §4.1 is already a jump functional. Minimising $\tfrac12\lVert\mathcal G(\lambda)\rVert^2$ over $\lambda$ by gradient descent, with $\partial\mathcal G/\partial\lambda$ supplied by the end-to-end adjoint, is a virtual-control DD scheme that needs **no probe assembly and no Newton Jacobian**. It is a distinct point in the $\Sigma$-space of [[general-coupling-scheme]]: the accelerator axis there is $\mathcal K\in\{\texttt{richardson},\ \texttt{krylov},\ \texttt{newton-krylov},\ \texttt{direct-schur}\}$, and "adjoint gradient descent on a control functional" is none of them. It is also the natural formulation when $\dim M$ is too small for a Newton solve — the obstruction [[gap-worklist]] W85 records, where a $\dim M$ operator handed to a $\dim V$ Newton solve stalls in the untouched directions — and when the expert is differentiable but not cheaply probeable ([[probed-dtn-coupling]] §5.3).

> **~~[AI Inference]:~~ measured 2026-09-06/07 and it is the opposite — Tiers 32 and 33.** The inference was that a control formulation should degrade *more* gracefully than substructuring when the subdomain solver is inexact, because it never asserts the transmission condition has an exact solution. **The inexactness is precisely what breaks it.** Write each agent as exact-plus-defect, $\mathcal E_i=\mathcal S_i+e_i$, and linearise the jump about the true datum $g^\star$: $\ \text{jump}(g)=T(g-g^\star)+(e_i-e_j)+\text{h.o.t.}$ The exact parts cancel at $g^\star$ by construction and **the defects do not**, so $\arg\min\lVert\text{jump}\rVert^2 = g^\star - T^{+}(e_i-e_j)$. The minimiser is the datum at which the two agents' **errors cancel on the overlap**, and making two biased models agree moves both further from the truth in a correlated direction. $J$ measures consistency, $\sigma$ measures correctness, and for inexact agents those have different minimisers.
>
> **Measured, on one seam:** the map is observable — $T$ is full rank $32/32$ at $\kappa=4.48$ even through a frozen checkpoint — and minimising $J$ still drives $\sigma$ **up** by $6.6\times$ to $21\times$, surviving Tikhonov over ten decades, truncated SVD over six ranks and a step-length sweep. A **richer** control space makes it worse rather than better ($3.21\times$ worse on the full ring), which is the signature the derivation predicts: more room to absorb $e_i-e_j$. The scheme comparison has a sign — $\sigma_{\text{ctrl}}\lesssim C_\mu\lVert\Delta_e\rVert/\beta_{\text{ctrl}}$ against $\sigma_{\text{lag}}\le C_\mu\Pi\lVert\delta\lambda\rVert$ — so **virtual control wins iff the agents are accurate relative to the lag**, and $\tau\gg\varepsilon$ is this project's founding premise about frozen operators. Full record: [[control-observability]] and [[interaction-horizon]] §6.

**Where Atlas sits (optimization-based DD): measured, and refused for this class of agent.** The family is a natural fit on paper — the one ingredient a control formulation needs, the adjoint, is already a headline claim and is unused — and the reason it does not transfer is not implementation effort but the same property that breaks every other classical hypothesis on this page: the subdomain solver is **biased**, and an objective built on agreement between biased models has its minimum in the wrong place. That is a transferable negative result about output-matching coupling of inexact surrogates, and it is recorded as such ([[gap-worklist]] W150, `refused`).

---

# 6. Nonlocal domain decomposition — the fifth family, and the one Atlas actually belongs to

Sections 2–5 cover the four families Toselli & Widlund organise, and **all four assume the same thing about the subdomain solver: that it is local.** Influence travels by a stencil of finite radius, so an overlap wider than that radius decouples the pieces, and every estimate on this page — the $H^{-2}$ of one-level Schwarz, the optimized-Schwarz symbol, the mortar inf-sup argument — is written against that assumption. A frozen neural operator is not local, and the classical literature has a family for operators that are not: it is filed under **nonlocal PDEs**, not under machine learning.

**The nonlocal diffusion / peridynamics programme.** Du, Gunzburger, Lehoucq & Zhou (*SIAM Review* 54, 2012; *M3AS* 23, 2013) build an analysis for operators of the form

$$\mathcal L u(x) \;=\; \int_{B_\delta(x)} \bigl(u(y)-u(x)\bigr)\,\gamma(x,y)\,dy ,$$

where a **finite interaction horizon** $\delta$ replaces the stencil radius throughout — well-posedness, a nonlocal vector calculus with its own Green's identities, nonlocal balance laws, and the local limit as $\delta\to0$. Silling's peridynamics (*JMPS* 48, 2000) is the mechanics instance. **Two structural features are exactly Atlas's:**

- **Boundary conditions become *volume* constraints.** A nonlocal operator cannot be given data on a surface; it needs data on a **collar of thickness $\delta$** surrounding the domain, because that is what its integral reads. **That is a halo.** Atlas's overlapping exchange, which the classical families treat as an implementation choice to be minimised away, is in this literature the *only* well-posed way to give a nonlocal operator its data — and the halo rule's $d_i = \rho_i s_i$ is a discrete interaction horizon, declared rather than measured.
- **The horizon is a *parameter of the operator*, not of the discretization.** Refining the mesh does not shrink it. That is precisely the property [[prior-art-and-novelty-atlas-0.1]] §2.1 names as the frozen surrogate's fourth missing property — error that does not shrink when the step shrinks — restated as a feature of a well-understood class rather than as a pathology.

**Off-diagonal decay, and why it is the theorem Atlas wants.** The adjacent operator-algebra literature — Jaffard's class of matrices with off-diagonal decay (1990), and Gröchenig & Leinert's symmetry and **inverse-closedness** results (*J. Amer. Math. Soc.* 19, 2006) — proves that matrices whose entries decay fast enough away from the diagonal form an algebra that is *closed under inversion*: the inverse decays at the same rate. **That is the theorem that would let a pseudo-local agent inherit a local agent's estimates**, because every interface construction on this page ultimately inverts something — $S$, $M_{\text{AS}}^{-1}A$, the optimality system.

**Measured, and the three agents land in three different classes** ([[interaction-horizon]] §3–§5):

| agent | decay of the influence away from the artificial face | class |
|---|---|---|
| `WindowNS`, elliptic exposed | **exactly zero** beyond $b=20=\rho s$, bitwise | compactly supported — the strongest case, and the classical hypothesis |
| `WindowNS`, elliptic embedded | one mode family $e^{-ck}$, another $\sim 1/k$; the profile itself is flat in distance | algebraic at best — marginal for an off-diagonal-decay class |
| **Poseidon-T** | none at any threshold from $10^{-1}$ to $10^{-10}$; far-field effective rank $26$ of $32$ | outside every decay class tried |

**Where Atlas sits (nonlocal DD): this is its home family, and it has been coupling nonlocal operators with local-model machinery the whole time.** $\Pi_w$ is the horizon-weighted contamination — the nonlocal analogue of an indicator that assumes a stencil — and $d_{\text{eff}}(\theta)$ is the interaction horizon, **measured rather than declared**, which is the one thing the classical nonlocal literature takes as given and a frozen checkpoint cannot supply. The refusal Atlas issues against a globally receptive agent is, in this language, the statement that the agent has **no finite horizon**, so the volume constraint it would need is the whole domain and there is nothing left to decompose.

**And the fourth family reappears here, with the same verdict.** Optimization-based coupling is the standard tool for joining a nonlocal model to a local one — D'Elia, Perego, Bochev & Littlewood, "A coupling strategy for nonlocal and local diffusion models with mixed volume constraints and boundary conditions" (*Comput. Math. Appl.* 71, 2016) minimises exactly a mismatch functional over virtual controls in the overlap. §5's measured refusal applies to it unchanged whenever the two models being joined are **biased relative to each other**, which is the frozen-expert case and is not the case that literature is written for: there, one of the two models is a trusted local reference.

> **[AI Inference]:** the productive reframing is that Atlas should declare an **interaction horizon** per expert rather than a stencil radius, with `required_halo` reading $d_{\text{eff}}(\theta)$ at a declared tolerance and the residual influence beyond it charged explicitly as a **tail term** in $\sigma$ rather than assumed to be zero. That is what the nonlocal literature does with $\delta$ and what an off-diagonal-decay argument would license. It is not implemented, it is [[gap-worklist]] W155, and the measurement it needs — the profile — now exists.

---

# 7. Which classical theorems break under a frozen neural operator, and why

The root causes are the four properties [[prior-art-and-novelty-atlas-0.1]] §2.1 lists that a frozen neural surrogate lacks:

- **(a)** no **consistency order** — the truncation error does not go to zero under mesh/step refinement;
- **(b)** no **CFL condition** and no amplification factor from which a stability region could be read;
- **(c)** no **statable Lipschitz bound** on the input–output map, and generally no monotonicity in the coupling variable;
- **(d)** **error not reducible by shrinking the step** — the operator was trained at a native $\Delta t$, and stepping below it takes the operator out of distribution.

| # | Classical theorem | Family | Hypotheses it needs | Why it breaks | What survives |
|---|---|---|---|---|---|
| 1 | One- / two-level Schwarz condition-number bounds, $\kappa\le C H^{-2}(1+H/\delta)$ and $C(1+H/\delta)$ | overlapping | SPD, variationally-posed $A$; **exact** local solves (a, c) | no energy norm the local solve is exact in ⇒ no $\omega$ in the abstract framework of §1 | the **sparsity** argument: a dense assembled interface operator couples the whole graph in one solve regardless of SPD ([[probed-dtn-coupling]] §3 "under-claiming note"). $\kappa(S)$ is then **measured**, not bounded a priori |
| 2 | FETI-DP / BDDC polylogarithmic bound $\kappa\le C\bigl(1+\log(H/h)\bigr)^2$ | substructuring | SPD; exact subdomain solves; a well-defined $h$ (a) | a neural operator has no $h$ and no SPD stiffness matrix; the bound has no analogue | the **construction** — primal corner constraints at cross-points still make sense as a way to make the interface problem well-posed; only its *optimality* fails to transfer |
| 3 | Optimized-Schwarz optimal transmission coefficient = symbol of the exact DtN | overlapping | a known PDE, known symbol, known viscosity (a, c) | OP-4: the checkpoint has no documented $\nu$ and 81% of its dissipation below the scale where a single $\nu$ exists; every analytic route to $\alpha^\star$ is closed | the coefficient can still be **measured** — it is the diagonal of the probed $S_i$ in a Fourier basis ([[probed-dtn-coupling]] §4.4) |
| 4 | Lax–Richtmyer equivalence: consistency + stability $\Leftrightarrow$ convergence | all | a consistency **order** — truncation error $\to0$ as $\Delta t, h\to0$ (a, d) | the per-step agent defect $\tau_i$ ([[master-error-bound]] §3) has **no small parameter**; refining $\Delta t$ takes the expert OOD, so $\tau_i\not\to0$ | you **measure** $\tau_i$ against a validated reference on the true solution's restriction; you cannot make it small by refinement |
| 5 | Partitioned-coupling stability theory — Jacobi/Gauss–Seidel coupling, waveform-relaxation convergence, quasi-Newton interface acceleration (preCICE IQN-ILS), energy-residual step control | all (co-simulation) | consistent discretizations of known order; error controllable by step size; Lipschitz, usually monotone I/O maps; known subsolver amplification factor (a, b, c, d) | all four hypotheses fail at once; a coupling iteration can diverge because the expert's response to a boundary perturbation is **non-monotone**, not because the physics is stiff ([[spec-wind-farm-wake-atlas-0.1]] §8.3 makes this an explicit gate) | the *structure* — additive vs. multiplicative, Dirac interconnection — but **not** the universal remedy: shrinking the communication step is unavailable (d) |
| 6 | Mortar optimality: global error $\le$ sum of subdomain best-approximation errors, constant independent of $H$ | mortar | each subdomain space has an approximation theory (best-approx error $\to0$ under refinement); a **provable** discrete inf-sup constant (a, c) | a frozen operator has no trial space and no best-approximation error, and $\beta$ can only be measured | the **shape** of the estimate: [[master-error-bound]] §4's $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$ is the mortar bound with every factor measured — a bound with numbers only after a probe, not a theorem |
| 7 | Substructuring exactness: $\tilde\Lambda=\Lambda \Rightarrow$ composed solution exact (the worst-agent conjecture as a theorem, [[master-error-bound]] §8 row 2) | substructuring | $\Lambda_i$ a genuine **linear** operator, i.e. $\mathcal E_i$ affine in $\lambda$ (c) | for Navier–Stokes and for a neural operator $\mathcal E_i$ is nonlinear; a probe recovers a tangent map at a state, and $\beta$, $\kappa$, $\Xi$, the passivity spectrum are all local to that linearization ([[probed-dtn-coupling]] §7). Measured further: for an explicit one-step map the flux-balance equation is not the right equation at all ([[probed-dtn-coupling]] §2.3) | the theorem still holds for a genuinely linear, implicit, boundary-value subdomain solve — the classical case — and it is the target the ladder climbs toward |
| 8 | SWR contraction-rate estimates — superlinear on bounded windows (parabolic), linear with computable factor (hyperbolic) | overlapping (SWR) | an **exact** local evolution operator (a) | the rate constants assume the subdomain problem is solved exactly over the window | the **geometric** finite-step / one-step result under $U_\infty\Delta t<\delta$ survives, because it is a statement about the PDE's transport speed and the overlap width, not about the solver ([[schwarz-iteration-atlas-0.1]] §5.1) |

**What carries over completely unchanged**, worth stating so the losses above are not read as "the theory is useless":

- the **additive-vs-multiplicative** ordering argument (structural, R5 of [[general-coupling-scheme]]);
- the **global-coupling / sparsity** argument for a dense interface operator (item 1, right column);
- the SWR **geometric condition** $U_\infty\Delta t < \delta$ (item 8, right column);
- the **partition-of-unity identity** $\sum_i R_i^{\!\top}\chi_i R_i = I$ and its use in bounding assembly ([[master-error-bound]] §6);
- and — *added* by Atlas rather than imported — the **convexity condition** $\chi_i\ge0$ (§2.5), which a single-sweep blend needs and an iterated Schwarz method does not.

**The through-line.** Every broken theorem breaks at the same joint: the classical proof bounds the *local* solve in an energy norm and lets refinement drive that bound to zero. A frozen expert gives you neither the norm nor the refinement, so each classical *number* becomes an Atlas *measurement* — $\kappa(S)$, $\beta$, $\tau_i$, $\lVert\Lambda-\tilde\Lambda\rVert$, the passivity spectrum — and the master bound of [[master-error-bound]] is the bookkeeping that says which measurement stands in for which theorem.

---

# 8. Annotated reading list

- **Toselli & Widlund, *Domain Decomposition Methods — Algorithms and Theory* (Springer, 2005).** The reference text. Ch. 1–2 for the abstract Schwarz framework $\kappa\le C(1+\rho(\mathcal E))\omega$ that §1 here is built on — every "theorem that breaks" in §7 is a hypothesis of that framework failing. Ch. 2–3 overlapping Schwarz and coarse spaces; Ch. 4 iterative substructuring and the Schur complement; Ch. 5 Neumann–Neumann and balancing; Ch. 6 FETI-1, FETI-DP and BDDC, including the corner-primal construction Atlas needs for cross-points ([[gap-worklist]] W6). Read the local-solver hypotheses in each chapter as the thing a neural expert removes.
- **Quarteroni & Valli, *Domain Decomposition Methods for Partial Differential Equations* (Oxford, 1999).** The Steklov–Poincaré operator treated directly (Ch. 1–2), and — the reason it is on this list — **heterogeneous / multiphysics coupling** where the two subdomains solve *different* equations (advection–diffusion to pure advection, Stokes–Darcy). That is the closest classical analogue to Atlas's field↔lumped seams and to the $\tau$-`UNDEFINED` multiphysics seam of [[master-error-bound]] §7 and [[gap-worklist]] W88. Pair with Gervasio–Lions–Quarteroni (below) for the virtual-control treatment of the same problem.
- **Dolean, Jolivet & Nataf, *An Introduction to Domain Decomposition Methods — Algorithms, Theory, and Parallel Implementation* (SIAM, 2015).** Optimized Schwarz — optimal Robin and Ventcell coefficients from the DtN symbol (Ch. 2), which §7 item 3 is about — and **spectral coarse spaces (GenEO, Ch. 5)** that adapt to the operator rather than to geometry. Directly relevant to why [[probed-dtn-coupling]] §3 needs a probed coarse space and cannot use either the geometric or the GenEO construction.
- **Gander, "Optimized Schwarz Methods" (*SIAM J. Numer. Anal.* 44, 2006)** and **"Schwarz methods over the course of time" (*ETNA* 31, 2008)**; **Gander & Halpern / Gander & Stuart on Schwarz waveform relaxation.** The definitive account of why optimized transmission conditions beat Dirichlet–Neumann, and the **finite-step / one-step SWR convergence result** for advection-dominated problems when no signal crosses the overlap in the time window ($U_\infty\Delta t<\delta$) — [[schwarz-iteration-atlas-0.1]] §5.1 is a direct application, and §7 item 8 here is what does and does not survive from it.
- **Du, Gunzburger, Lehoucq & Zhou, "Analysis and Approximation of Nonlocal Diffusion Problems with Volume Constraints" (*SIAM Review* 54, 2012)** and **"A nonlocal vector calculus, nonlocal volume-constrained problems, and nonlocal balance laws" (*M3AS* 23, 2013)**; **Silling, "Reformulation of elasticity theory for discontinuities and long-range forces" (*JMPS* 48, 2000)**. §6's family. Read for two things Atlas needs and the other four families do not provide: the **interaction horizon** $\delta$ carried through every estimate in place of a stencil radius, and **volume constraints** — data on a collar of thickness $\delta$ rather than on a surface, which is what Atlas's halo actually is. The local limit $\delta\to0$ is the bridge back to §§2–5.
- **Jaffard, "Propriétés des matrices bien localisées près de leur diagonale" (*Ann. IHP* 7, 1990)**; **Gröchenig & Leinert, "Symmetry and inverse-closedness of matrix algebras and functional calculus for infinite matrices" (*JAMS* 19, 2006)**. Off-diagonal decay as an algebra closed under inversion — the theorem that would let a **pseudo-local** agent inherit a local one's estimates, since every construction on this page inverts something. [[interaction-horizon]] measures which decay class each of Atlas's three agents is in, and the checkpoint is in none of them.
- **D'Elia, Perego, Bochev & Littlewood, "A coupling strategy for nonlocal and local diffusion models with mixed volume constraints and boundary conditions" (*Comput. Math. Appl.* 71, 2016).** §5's family applied to §6's operators, and the closest published thing to what Atlas tried and refused. The difference that matters: there, one of the two coupled models is a **trusted local reference**, so the optimisation has an unbiased anchor. Atlas has two biased agents and no anchor, which is exactly the case [[interaction-horizon]] §6 shows the objective mis-minimises.
- **van der Schaft & Jeltsema, *Port-Hamiltonian Systems Theory: An Introductory Overview* (*Foundations and Trends in Systems and Control*, 2014).** Dirac structures, power-preserving interconnection, and the theorem that an interconnection of passive port-Hamiltonian systems is passive — the structural backing for [[port-algebra-atlas-0.1]]'s connection rule and for [[master-error-bound]] §6.1's $L\le1$ result. Read for **incremental (shifted) passivity**, the property that bounds the *error* between two trajectories rather than the state, which is what [[composition-error-theory]] §4.1 insists on.
- **The preCICE papers: Bungartz et al., "preCICE — A fully parallel library for multi-physics surface coupling" (*Computers & Fluids* 141, 2016)**, and **Chourdakis et al., "preCICE v2: a sustainable and user-friendly coupling library" (*Open Research Europe* 2, 2021).** The mature partitioned-coupling implementation Atlas is positioned against ([[prior-art-and-novelty-atlas-0.1]] §3, §8): quasi-Newton interface acceleration (IQN-ILS), the **conservative-vs-consistent mapping** distinction ([[port-algebra-atlas-0.1]] §9.3, derived in [[interface-transfer-theory]]), and serial/parallel explicit/implicit coupling. Read for what Atlas should borrow rather than reinvent, and as the concrete instance of the stability theory §7 item 5 says does not extend to neural subsolvers.

---

## See Also

- [[general-coupling-scheme]] — the seven-parameter scheme $\Sigma$ this page reads as coordinates in classical DD space; §2's `overlapping` vs §3's `non-overlapping` axis, and the accelerator axis §5 says has no virtual-control entry
- [[case-study-neural-interface-atlas-0.1]] — §2.2's coarse space and §7's Robin coefficient **measured**, in expert calls; and the one-level $H^{-2}$ story seen from the other side, where the overlap turns out to be load-bearing only while the elliptic solve is inside the agent
- [[probed-dtn-coupling]] — the substructuring construction of §3: the Steklov–Poincaré operator assembled by black-box probing, the coarse space of §2.2, the measured Robin coefficient of §7 item 3
- [[composition-error-theory]] — §4.0's transmission ladder is the optimized-Schwarz ladder of §2.3; §4.2's weak/mortar argument is §4 here; §4.4's adjoint is the ingredient §5 says is unused
- [[master-error-bound]] — the bookkeeping that replaces each broken theorem of §7 with a measured term; §4.1 is §2.4's overlapping-branch $\sigma$, §4 is §4's mortar-shaped bound
- [[interface-transfer-theory]] — the declared common interface space $M$ and prolongations $P_i$ that make Atlas's non-conforming transfer a mortar method (§4)
- [[schwarz-iteration-atlas-0.1]] — the SWR reading of §2.1, the $H^{-2}$ / $k_{\min}=5$ arithmetic of §2.2, and the faithfulness precondition behind §7 item 8
- [[port-algebra-atlas-0.1]] — the pointwise (strong) connection rule of §4, and the power-bond structure van der Schaft–Jeltsema underpins
- [[prior-art-and-novelty-atlas-0.1]] — §2.1's four missing properties of a frozen surrogate, which are the root causes tabulated in §7
- [[temporal-error-accumulation]] — where the SWR / multiple-shooting window $W$ becomes a design variable rather than a consequence of the native $\Delta t$
- [[gap-worklist]] — carries W150 (the optimization-based-DD family, measured and `refused`), W153–W156 from §6's nonlocal reading, and W6/W64 (cross-points) and W85 (the $\dim M$ Newton obstruction) referenced above
- [[interaction-horizon]] — §6's measurements: $d_{\text{eff}}$, the horizon-weighted envelope $\Pi_w$, and the three-way decay control that puts the three agents in three different classes
- [[control-observability]] — §5's measured verdict: the control-to-jump map is observable through a frozen checkpoint and its minimiser is still not the truth
