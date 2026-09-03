# Mathematical Compendium — every result the project relies on, in one notation

**Type:** Concept page — **synthesis hub, cross-track** (folder: `concepts/00-pfm-core/`)
**Status:** written 2026-09-03. A *reference*, not a new result: every statement below is derived on another page and links to it. Nothing here is original except (a) the reconciled notation of §2 and (b) the cross-reading findings of §17, which are lint output rather than theory.
**Related:** [[port-algebra-atlas-0.1]] · [[master-error-bound]] · [[composition-error-theory]] · [[probed-dtn-coupling]] · [[temporal-error-accumulation]] · [[schwarz-iteration-atlas-0.1]] · [[symmetry-averaging-atlas-0.1]] · [[general-coupling-scheme]] · [[interface-transfer-theory]] · [[conservation-as-constraint-atlas-0.1]] · [[theory-closure-audit]] · [[plug-in-composition-theorems]] · [[backbone-1.1]] · [[discovered-conservation-1.1]] · [[post-decoder-diffusion-1.1]]

---

# 0. What this page is for, and how to read it

**Intuitively.** The project has accumulated about a dozen pages of real mathematics, each written to answer the question in front of it. They are individually careful and collectively unreadable as a body of theory: the same letter means four things, the same result is stated in three notations, and the hypotheses of one page are the measured failures of another. This page is the single place where all of it is stated once, in one alphabet, with each result's *assumptions* written next to it rather than three sections away.

**Rigorously.** For each result: the statement in LaTeX, a gloss of every symbol, the hypotheses under which it holds, and a link to the page that derives it. Where a result is known to be false, vacuous, or scope-limited outside its stated hypotheses, that is recorded with the result rather than in a footnote.

**The one thing to take away if you read nothing else.** The whole Atlas theory is one inequality with four constants,

$$\lVert e^N\rVert \;\le\; L^N\lVert e^0\rVert \;+\; \sum_{n=1}^{N} L^{\,N-n}\bigl(\tau^n+\sigma^n+\gamma^n\bigr),$$

and every construction in the vault is an attempt to control exactly one of $L$, $\tau$, $\sigma$, $\gamma$. Knowing which one a given page is about is most of understanding it.

**Scope.** Atlas 0.1 composition theory (§3–§12), the Noether 1.1 dynamics core (§13), the governing-equation families the experts carry (§14), the hypothesis envelope (§15), the constants and what is measured (§16), and the inconsistencies found while cross-reading (§17). Noether 1.0 appears only where 1.1 inherits from it.

---

# 1. The objects, once

These are the nouns every later section uses. Established across [[master-error-bound]] §1, [[composition-error-theory]] §1 and [[probed-dtn-coupling]] §2.

| symbol | name | meaning |
|---|---|---|
| $\Omega$, $\{\Omega_i\}_{i=1}^{K}$ | global domain, agent domains | the decomposition; $K$ agents |
| $R_i$ | restriction | $\Omega\to\Omega_i$; $R_i^{\!\top}$ is its transpose (extension by zero) |
| $\chi_i$ | partition of unity | $\sum_i R_i^{\!\top}\chi_i R_i = I$; the assembly weights |
| $\Gamma$ | interface | $\bigcup_{i\neq j}\partial\Omega_i\cap\overline{\Omega_j}$ |
| $\mathcal S_{\Delta t}$ | exact evolution | the true solution operator on $\Omega$ over one macro-step |
| $\mathcal S_i(v;\mu)$ | exact local evolution | on $\Omega_i$, from state $v$, with interface datum $\mu$ |
| $\mathcal E_i(v;\mu)$ | **agent** | the expert's approximation to $\mathcal S_i$ — the frozen checkpoint or classical solver |
| $\Lambda_i$ | exact interface operator | Steklov–Poincaré / Dirichlet-to-Neumann for $\Omega_i$; global and nonlocal |
| $\tilde\Lambda$ | transmission condition | the rung actually used: Dirichlet ring, Robin, probed Schur… |
| $\mathcal G(\lambda;u^n)$ | interface residual | the equation the coupling poses; $\mathcal G=0$ is *its* consistency condition |
| $\mathcal A$ | assembly | combines local outputs into a global state (partition of unity, or single-valued fluxes) |
| $\Phi$ | composed macro-step | $u^n\mapsto u^{n+1}$; the whole coupled solve, $\Phi=\mathcal A\circ\bigl(\bigoplus_i\mathcal E_i\bigr)$ |
| $e^n$ | error | $u^n - u^{n,\star}$ against the true solution sampled at $t^n$ |

**Three interface traces, and keeping them apart is the content of §4.**

$$\lambda^\star=\operatorname{tr}_\Gamma u^\star \quad\text{(what the \emph{true} solution puts on }\Gamma)$$
$$\lambda^\dagger:\ \mathcal G(\lambda^\dagger)=0 \quad\text{(the \emph{exact root} of the problem you \emph{posed})}$$
$$\lambda^{(k)} \quad\text{(what the solver \emph{returned} after }k\text{ iterations)}$$

---

# 2. Global notation — the reconciliation

This is the section the compendium exists for. Sixteen symbols carry more than one meaning across the vault, and several of the collisions are between a *quantity of the error theory* and a *quantity of continuum mechanics* appearing in the same worked example. Each row below states the convention this page uses, then names every page that departs from it. **Departures are not errors** — each page's usage is standard within its own field — but a reader moving between pages must know where the meaning changes.

## 2.1 The collisions that matter most

| symbol | **this page's convention** | competing meanings, and where |
|---|---|---|
| $L$ | **amplification constant**, $L=\operatorname{Lip}(\Phi)$ on the relevant set | (a) the **physical size of an expert's window** in $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p L^2)$ — [[schwarz-iteration-atlas-0.1]] §S3, [[probed-dtn-coupling]] §3, and the `L_native` field of [[general-coupling-scheme]] §2, where it appears **within three lines of the fitted $\hat L$**; (b) the **backbone depth** in Noether's $12Ld^2$ parameter formula — [[noether-1.0]], [[noether-1.1-medium]]; (c) $\mathcal L$, the **spatial operator** $\partial_t u=\mathcal L(u)$ — [[composition-error-theory]] §1. **Here:** window size is $\ell_w$, depth is $n_L$, spatial operator stays $\mathcal L$ (script) |
| $\beta$ | **inf-sup / interface conditioning constant**, $\beta=\sigma_{\min}\bigl(\tilde\Lambda\vert_{\text{constrained}}\bigr)$ | the **thermoelastic modulus** $\beta=E\alpha/(1-\nu)=2.403\times10^{6}\,\mathrm{Pa\,K^{-1}}$ — [[case-study-thermal-strain-atlas-0.1]] §§6–10, which also *probes seams* and so carries both meanings on one page. **Here:** the thermoelastic modulus is $\beta_{\text{th}}$ |
| $\tau$ | **agent infidelity** (the first defect term) | (a) **torque**, the `ROT` effort — [[port-algebra-atlas-0.1]] §3.1; (b) the **deviatoric stress tensor** in $\boldsymbol\sigma=-p\mathbf I+\boldsymbol\tau$ — [[port-algebra-atlas-0.1]] §3.3 and the compressible-NS viscous fluxes; (c) an **edge-type index** and its gate $g_\tau$ — [[backbone-1.1]]; (d) the **diffusion noise index** — [[post-decoder-diffusion-1.1]], which flags its clash with the physical clock $t$ but not with the defect. **Here:** torque is $\tau_{\text{rot}}$, deviatoric stress is $\boldsymbol\sigma_{\text{dev}}$, edge type is $\theta$, diffusion index stays $\tau$ **only inside §13.3** |
| $\sigma$ | **transmission infidelity** (the second defect term) | (a) the **Cauchy stress tensor** $\boldsymbol\sigma$ — every mechanics page; (b) $\sigma_{\min},\sigma_{\max}$ **singular values** — [[probed-dtn-coupling]] §4.3; (c) $\sigma_\tau$ the diffusion **noise scale**; (d) $\sigma_{\text{SB}}$ **Stefan–Boltzmann**; (e) the **CFL number** $\sigma=0.4$ — [[impl-atlas-0.1-phase0-scope-and-data]] §2.1. **Here:** stress is always **bold** $\boldsymbol\sigma$, singular values always subscripted, CFL is $\mathrm{CFL}$ |
| $R$ | $\mathcal R(t)$ (script) is the **global power residual**; $R_i$ (roman) is the **restriction operator** | (a) $R_i=P_i^{\ast}$, the **reduction** of [[interface-transfer-theory]] §1 — *this is a different operator from the restriction, on a different space*; (b) $\mathcal R_i$, restriction **to a halo** — [[schwarz-iteration-atlas-0.1]] §3; (c) `R1`–`R10`, the **compiler admissibility rules** — [[general-coupling-scheme]] §3; (d) `R1`/`R2`, the two **interface-data variants** (halo tokens vs. declared flux) — [[schwarz-iteration-atlas-0.1]] §2.1 and [[impl-wind-farm-guide]] §6.2. **(c) and (d) collide exactly**, within one folder. **Here:** rules are always code-font `R1`…`R10`; the interface variants are written `I-R1`/`I-R2` |
| $W$ | **window length in macro-steps** | (a) `W1`, `W2`, … the **[[gap-worklist]] rows**, cited beside window arithmetic on several pages; (b) `W0`–`W11`, the **wind-farm gates**; (c) $W_{\text{skew}},W_O,W_V$, backbone **weight matrices**. **Here:** worklist rows and gates are always code-font |
| $\gamma$ | **solve incompleteness** (the third defect term) | (a) $\gamma_{\text{gas}}$, the **ratio of specific heats**; (b) $\gamma_\ell$, the per-layer **dissipation dial** — [[backbone-1.1]]; (c) $\gamma_{\text{vec}}$, a **vector constant** on the covariant channel. **Here:** all three carry their subscripts |
| $\varepsilon$ | $\varepsilon_i$ is an agent's **benchmark accuracy** on its own distribution | (a) the **probe step** in a finite difference — [[probed-dtn-coupling]] §5.1; (b) $\varepsilon_i=c\,\varepsilon_{\text{mach}}\lVert S_i\rVert$, the **passivity noise floor** — [[probed-dtn-coupling]] §4.2, *three sections from meaning (a) and using the same subscript as the benchmark accuracy*; (c) $\boldsymbol\varepsilon$, the **strain tensor**, and $\boldsymbol\varepsilon_0$ the **eigenstrain**; (d) $\varepsilon_{\text{tol}}$; (e) the nozzle **expansion ratio** $A_e/A_t$; (f) the surface **emissivity**. **Here:** probe step is $\epsilon_{\text{p}}$, noise floor is $\eta_i$, strain is bold |
| $\mathcal D$ | the **decomposition axis** of the scheme tuple | (a) $\mathcal D_i$, an agent's **declared dissipation** — [[port-algebra-atlas-0.1]] §6 and [[composition-error-theory]] §4.1; (b) $\mathcal D_i$, an agent's **validation distribution** — [[composition-error-theory]] §1, *on the same page as (a)*; (c) $D$, the **rotor diameter**; (d) $D_k$, an invariant's **drift** — [[discovered-conservation-1.1]]; (e) $\mathsf D$, the **elasticity matrix**; (f) $d_i$, the **domain-of-dependence reach**. **Here:** dissipation is $\mathcal D_i^{\text{diss}}$, validation distribution is $\mathfrak D_i$ |
| $H$ | $H_i$ is a **storage function** | (a) $H$, the **subdomain diameter** in the Schwarz condition number — [[schwarz-iteration-atlas-0.1]] §3.2; (b) $H$, the **number of attention heads**; (c) $H$, the **thermal stiffness matrix** in $\tfrac12\Delta T^{\!\top}H\Delta T$. **Here:** diameter is $H_\Omega$, heads are $n_H$, thermal matrix is $H_{\text{th}}$ |
| $\rho$ | the **density** | (a) $\rho_S$, the **Schwarz contraction factor** — [[master-error-bound]] §5; (b) $\rho_i$, the **stencil radius** — [[master-error-bound]] §4.1; (c) $\rho_i$, the **local residual** of agent $i$ in dual-weighted-residual attribution — [[composition-error-theory]] §4.4. **(b) and (c) share both letter and subscript.** **Here:** stencil radius is $r_i^{\text{sten}}$, DWR residual is $\varrho_i$ |
| $E$ | $\mathcal E_i$ (script) is the **agent operator** | (a) $E$, the frozen **expert operator** in the Reynolds average — [[symmetry-averaging-atlas-0.1]] §2 (*identical object to $\mathcal E_i$, different letter*); (b) $E$, the **total energy** in the compressible-NS state vector; (c) $E_i$, an agent's **stored energy** in $\mathcal R(t)$; (d) $E_s$, **Young's modulus**. **Here:** the agent is always $\mathcal E_i$, stored energy $E_i$, Young's modulus $E_s$ |
| $\mu$ | $\mu_i=\lambda_{\min}\bigl(\tfrac12(S_i+S_i^{\!\top})\bigr)$, the **passivity margin** | (a) the **dynamic viscosity**; (b) $\mu$, the **interface datum** in $\Phi_\mu$ and the probe basis element $\mu_k\in M$ — [[master-error-bound]] §3, [[interface-transfer-theory]] §3; (c) $\mu_k$, a **chemical potential** — [[port-algebra-atlas-0.1]] §3.2. **(b) and (c) collide as $\mu_k$.** **Here:** viscosity $\mu_{\text{visc}}$, chemical potential $\mu^{\text{chem}}_k$ |
| $\Pi$ | the **contaminated weight** of the overlapping $\sigma$ bound | $\Pi_M$, the **$M$-projection**, and $\Pi_W$, the **temporal projection** — [[interface-transfer-theory]] §§4.2, 5.3. **Here:** the projections always carry their subscript; $\Pi$ bare is the contaminated weight |
| $\lambda$ | the **interface trace** | (a) $\lambda_1$, the leading **Lyapunov exponent**; (b) $\lambda_{\min}$, an **eigenvalue**; (c) $\lambda=-\tfrac23\mu_{\text{visc}}$, the **second viscosity** (Stokes' hypothesis); (d) $\lambda_k$, a **loss weight** — [[discovered-conservation-1.1]]. **Here:** all four carry subscripts or are named in place |
| $k$ | the **iteration index** | (a) the **thermal conductivity** $k_s$; (b) the **mode index** $\psi_k,\mu_k$; (c) the invariant index $C_k$. **Here:** conductivity always $k_s$ |

## 2.2 Reserved symbols — the ones with exactly one meaning

Kept here so a reader can trust them without checking.

| symbol | meaning | first stated |
|---|---|---|
| $e,f$ | port **effort** and **flow**; $e\cdot f$ is a power | [[port-algebra-atlas-0.1]] §2 |
| $\mathcal R(t)$ | the **global power residual** | [[port-algebra-atlas-0.1]] §6 |
| $\tau,\sigma,\gamma$ | agent / transmission / solve defects | [[master-error-bound]] §3 |
| $C_\mu$ | $\operatorname{Lip}_\mu(\Phi_\mu)$ — sensitivity of the composed local solve to its interface datum | [[master-error-bound]] §4 |
| $\Xi_i$ | **composability index** $\lVert\Lambda_i^{\text{expert}}\rVert/\lVert\Lambda_i^{\text{ref}}\rVert$ | [[probed-dtn-coupling]] §4.5 |
| $S_i$, $S=\sum_i S_i$ | the **probed** blocks and the assembled **Schur complement** | [[probed-dtn-coupling]] §2.2 |
| $\pi_i$ | **passivity defect**, $\mu_i$ clipped at the noise floor | [[probed-dtn-coupling]] §4.2 |
| $\kappa$ | $\sigma_{\max}/\sigma_{\min}$ of the assembled operator | [[probed-dtn-coupling]] §4.3 |
| $P_i$ | **prolongation** $M\to V_i$; $P_i^{\ast}$ its adjoint (the reduction) | [[interface-transfer-theory]] §1 |
| $M$ | the **common interface space** of a seam | [[interface-transfer-theory]] §1 |
| $n_0(\Gamma)$ | expected **null-space dimension** at a seam | [[interface-transfer-theory]] §7 |
| $\Sigma$ | the seven-parameter **coupling scheme** tuple | [[general-coupling-scheme]] §1 |
| $T_{\text{pred}}$ | the **predictability horizon** past which the bound is vacuous | [[master-error-bound]] §9 |
| $\mathrm{E1}$–$\mathrm{E7}$ | the **seven-hypothesis envelope** | [[theory-closure-audit]] §3 |

---

# 3. The port algebra

Derived on [[port-algebra-atlas-0.1]]; the conservation reading is on [[conservation-as-constraint-atlas-0.1]].

## 3.1 The principle

**Intuitively.** Every way two physical systems can touch is a way for energy to cross between them, and energy crossing is always a product of two things — a "push" and a "how much flows". Naming the pair instead of the phenomenon is what stops the vocabulary growing with the number of case studies.

**Statement.**

$$P = e\cdot f,\qquad P_\Gamma=\int_\Gamma e\,f\;\mathrm ds .$$

$e$ is the **effort** (an intensity: traction, temperature, potential), $f$ the **flow** (a flux density: velocity, entropy flux, current density), $P_\Gamma$ the power crossing interface $\Gamma$.

**Hypotheses.** The pairing must be **unique** — if two readings of the same coupling give different powers, their difference is a stored energy and the residual of §3.4 has nowhere to put it. This is field 1 of the amendment procedure (§3.5) and it is what refused the first candidate sixth port type.

## 3.2 The five port types — the closed vocabulary

| Port | effort $e$ | flow $f$ | $e\cdot f$ |
|---|---|---|---|
| `MECH` | traction $\mathbf t=\boldsymbol\sigma\!\cdot\!\mathbf n$ [Pa] | velocity $\mathbf v$ [m s$^{-1}$] | W m$^{-2}$ |
| `ROT` | torque $\tau_{\text{rot}}$ [N m] | angular velocity $\omega$ [s$^{-1}$] | W |
| `THERM` | temperature $T$ [K] | entropy flux $\dot s_n=q_n/T$ [W m$^{-2}$K$^{-1}$] | W m$^{-2}$ |
| `ELEC` | potential $\phi$ [V] | current density $\mathbf j\!\cdot\!\mathbf n$ [A m$^{-2}$] | W m$^{-2}$ |
| `ADVEC` | *multiport*: carrier $\dot m=\rho\,\mathbf u\!\cdot\!\mathbf n$, passengers $h_0=h+\tfrac12\lVert\mathbf u\rVert^2$, $Y_k$ | — | $h_0\dot m$ [W m$^{-2}$] |

**`MECH` absorbs four earlier labels exactly**, and the identity is the Cauchy decomposition:

$$\boldsymbol\sigma=\underbrace{-p\,\mathbf I}_{\text{"pressure"}}+\underbrace{\boldsymbol\sigma_{\text{dev}}}_{\text{"stress", "shear"}},\qquad \mathbf t=\boldsymbol\sigma\!\cdot\!\mathbf n .$$

Pressure is the isotropic part, viscous and elastic stress the deviatoric part, "shear" the tangential component of the same traction. **This is an identity, not a naming convention.**

**Assumptions carried by `THERM`.** Atlas uses the *true bond* $(T,q_n/T)$, whose product $T\cdot(q_n/T)=q_n$ is a power, rather than the *pseudo-bond* $(T,q_n)$ that most co-simulation codes exchange and whose product is not. The choice is what lets `THERM` enter the single residual of §3.4. It presumes entropy generation occurs **inside** agents and never at the crossing itself.

**Assumptions carried by `ADVEC`.** It is deliberately a multibond: a mass flux carries several conserved quantities at once. **The passenger list is part of the type and is declared per face**, not per agent — an agent whose two faces carry different passenger lists has no single correct per-agent declaration.

## 3.3 The scaling argument — $O(K)$ against $O(K^2)$

**Statement.** With $K$ expert families and a vocabulary indexed by *phenomenon*, the interface contract is defined per expert pair and the integration cost is

$$N_{\text{contracts}} \;=\; \binom{K}{2}+K \;=\;\frac{K(K+1)}{2}\;=\;O(K^2),$$

which is $10$ at $K=4$ but $120$–$210$ at the $K\approx15$–$20$ the end goal needs ([[f1-pathmap-and-end-goal]]). With a vocabulary indexed by *port type*, each expert publishes one declaration and any two experts exposing the same type connect without a bespoke adapter:

$$N_{\text{declarations}} \;=\; K \;=\; O(K).$$

**Hypotheses, and they are the whole content.** (i) The port vocabulary is **closed** — it may not grow with case studies, or the $O(K)$ collapses back. (ii) The **connection rule** is satisfiable: two agents may share an edge for port $P$ iff both declare $P$ with the same passenger list and a common interface space is declarable (§7.1). (iii) Nondimensionalization is a residual per-expert cost — real, mechanical, and $O(K)$, not $O(K^2)$.

**Departure to note.** The $O(K)$ claim is about *declaration count*, never about accuracy. [[port-algebra-atlas-0.1]] §8 is explicit that a clean contract makes more connections **legal** without making any of them **valid**, and that with $K$ experts the probability at least one is out of distribution approaches $1$ — so this axis gets *worse* as the vocabulary gets cleaner.

## 3.4 The global power residual $\mathcal R(t)$

$$\boxed{\;\mathcal R(t)\;=\;\sum_{i\in\mathcal A}\frac{\mathrm dE_i}{\mathrm dt}\;+\;\sum_{\Gamma\in\mathcal P}\int_\Gamma\bigl(e\,f\bigr)_\Gamma\,\mathrm ds\;-\;\sum_i\mathcal D_i^{\text{diss}}\;-\;P_{\text{ext}}\;}$$

$E_i$ is agent $i$'s stored energy, $\mathcal D_i^{\text{diss}}$ its declared dissipation, $P_{\text{ext}}$ the power through unconnected (open) ports. $\mathcal R\neq0$ means the coupling is creating or destroying energy; the per-port breakdown localizes where.

**Per-port residual**, reported everywhere rather than only at edges once labelled conservative:

$$r_\Gamma^{(\text{port})}=\frac{\bigl\lVert f_A+f_B\bigr\rVert_\Gamma}{\tfrac12\bigl(\lVert f_A\rVert+\lVert f_B\rVert\bigr)_\Gamma}.$$

**Hypotheses — three, and two of them are usually left implicit.**
1. **Additivity of stored energy**: $\Psi=\sum_i\Psi_i(u_i)$. This is the premise of the first sum, and it **fails** for a co-located coupling whose free energy has a bilinear cross term belonging to neither agent — measured at $2680\times$ the elastic energy in [[case-study-thermal-strain-atlas-0.1]]. It is field 7 (`support`) of the amendment procedure.
2. **Every port is a genuine power bond**, hence the `THERM` convention of §3.2 and the scale identity $s_e s_f = s_P$ of §7.4. Converting effort and flow with independently chosen scales is dimensionally plausible and silently destroys the bond, after which $\mathcal R$ reports a unit error in watts.
3. **$\mathcal R=0$ is necessary, never sufficient.** $\mathcal R\neq0$ is an excellent falsifier; $\mathcal R=0$ bounds nothing (§5.2). Derived on [[composition-error-theory]] §2.2.

**Known blind spot.** On a co-located graph $\mathcal R$ was measured blind to the coupling under test by six orders of magnitude ($1.04\times10^{-6}$ of the conducted power). The right residual there is the *receiving subsystem's own balance*.

## 3.5 Conservation is not a port type

$$e_A\big|_\Gamma=e_B\big|_\Gamma,\qquad f_A\big|_\Gamma=-f_B\big|_\Gamma .$$

Every port carries this by construction, so `conservation` was never a quantity — it is the definition of a connection. What survives is a per-port residual (above) plus a `conservative` flag deciding whether it is *enforced* or merely *measured*.

**The enforce-or-measure rule** ([[conservation-as-constraint-atlas-0.1]]), extended by [[composition-error-theory]] §4.5 to *enforce, measure, or decline*:

| situation | mechanism | encoding-spectrum level |
|---|---|---|
| one side closed-form | hard minimum-norm projection onto the exact side | 5 |
| the quantity has a potential representation (mass via a stream function) | exact by construction | 5 |
| both sides learned and frozen | **measure only** — no exact side, no loss | 2 |
| both sides learned and trainable | soft interface loss (the cPINN regime) | 2–3 |

**The flux-matching constraint** at a port declared conservative:

$$\Phi_{e\to f}(\hat x_e)\;=\;\Phi_{f\leftarrow e}(\hat x_f).$$

**Conservation is free and exact when the numerical flux is single-valued** — compute it once, hand the same number to both sides with opposite signs. The wind farm's $4.3\times10^{-14}$ mass closure is machine epsilon, i.e. *structural*, not the output of an enforcement step ([[temporal-error-accumulation]] §4.3).

## 3.6 The amendment procedure — seven fields

A sixth port type enters only by satisfying a **conjunction**: `bond` (a unique conjugate pair whose product is a power), `mapping` (derived, never declared — §7.2), `transfer` (an adjoint reduction/prolongation pair with $\dim M<\dim V_i$), `dtn_reading` (defined *and affordable*: a probe is $\dim M+1$ solves), `distinctness` (preferably an exact decomposition with a positive control), `exercise` (one built graph), `support` (the co-dimension of the carrier, and at $0$ an argument that free energies add).

**Exercised once, verdict `refuse`.** Thermal strain: fields 1, 4 and 7 fail. The constructive residue is a distinction the vocabulary did not have —

> A **port** is a bond on an interface of co-dimension $\ge1$ between agents whose free energies add. Thermal strain is a bond of co-dimension $0$ between agents whose free energies do **not** add: a bond, and not a port.

The instrument for such a coupling is an operator splitting with a splitting-error bound, measured first order in $\Delta t$ (§14.5), not a transmission condition.

---

# 4. The master error bound

Derived once on [[master-error-bound]]; the recursion is on [[temporal-error-accumulation]] §1, the $\tau\neq\varepsilon$ distinction on [[composition-error-theory]] §3.2.

## 4.1 The exact recursion

**Intuitively.** Two things happen every step: yesterday's error gets stretched or shrunk by the dynamics, and today's step adds a fresh mistake. Nothing else. The whole theory is about how big the stretch factor is and where the fresh mistake comes from.

With $e^n=u^n-u^{n,\star}$, add and subtract $\Phi(u^{n,\star})$ — an identity, no approximation:

$$e^{n+1}=\underbrace{\bigl[\Phi(u^n)-\Phi(u^{n,\star})\bigr]}_{\text{propagation}}+\underbrace{\bigl[\Phi(u^{n,\star})-\mathcal S_{\Delta t}(u^{n,\star})\bigr]}_{\text{one-step defect }d^{n+1}}$$

$$\lVert e^{n+1}\rVert\le L\lVert e^n\rVert+\lVert d^{n+1}\rVert \qquad\Longrightarrow\qquad \lVert e^N\rVert\le L^N\lVert e^0\rVert+\sum_{n=1}^{N}L^{\,N-n}\lVert d^n\rVert .$$

**Hypotheses.** $L=\operatorname{Lip}(\Phi)$ must exist on the set the trajectory visits. For a frozen neural surrogate this is exactly one of the four properties [[prior-art-and-novelty-atlas-0.1]] §2.1 records that partitioned-coupling stability theory assumes and a checkpoint does not supply.

## 4.2 The three-way defect split

Write $\Phi_\mu$ for the composed step *forced to use interface datum* $\mu$, so $\Phi=\Phi_{\lambda^{(k)}}$. Telescoping through the two intermediate traces of §1:

$$\boxed{\;d^{n+1}=\underbrace{\bigl[\Phi_{\lambda^\star}-\mathcal S_{\Delta t}\bigr](u^{n,\star})}_{\tau^{n+1}}+\underbrace{\bigl[\Phi_{\lambda^\dagger}-\Phi_{\lambda^\star}\bigr](u^{n,\star})}_{\sigma^{n+1}}+\underbrace{\bigl[\Phi_{\lambda^{(k)}}-\Phi_{\lambda^\dagger}\bigr](u^{n,\star})}_{\gamma^{n+1}}\;}$$

| term | name | what is wrong | reduced by |
|---|---|---|---|
| $\tau$ | **agent infidelity** | the agents are wrong *even handed the true interface data*: $\mathcal E_i\neq\mathcal S_i$ on the true solution's restriction | a better expert; a faithful (non-periodic) local solve |
| $\sigma$ | **transmission infidelity** | the interface problem you posed has the **wrong solution**, because $\tilde\Lambda\neq\Lambda$ | climbing the DtN ladder — **and nothing else** |
| $\gamma$ | **solve incompleteness** | you did not converge to your own problem's root | iterating; or one direct interface solve |

**Three consequences, each one line of algebra.**
1. **Schwarz iteration reduces $\gamma$ and nothing else.** It drives $\lambda^{(k)}\to\lambda^\dagger$, the root of *your* problem. If $\tilde\Lambda$ is wrong, iterating converges harder onto the wrong trace.
2. **$\sigma$ is invisible to every diagnostic the framework reports.** The interface residual measures $\gamma$; the conservation residual measures finitely many functionals.
3. **The vault's sharpest failure is a pure-$\sigma$ failure.** Under periodic windows $\Lambda\equiv0$, so $\tilde\Lambda$ is maximally wrong; $\gamma\equiv0$ *bitwise*; the answer was wrong by $2\times$. The two-term split cannot express that configuration.

## 4.3 Bounding $\sigma$ — branch A, substructuring

$$\lVert\lambda^\dagger-\lambda^\star\rVert\le\lVert\tilde\Lambda^{-1}\rVert\,\lVert(\tilde\Lambda-\Lambda)\lambda^\star\rVert\le\frac{1}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert,\qquad \beta=\sigma_{\min}\bigl(\tilde\Lambda\vert_{\text{constrained}}\bigr)$$

$$\boxed{\;\sigma\;\le\;\frac{C_\mu}{\beta}\,\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert\;}$$

**This is the theorem "agreement is an operator" was reaching for**: accuracy is set by the operator gap, amplified by $1/\beta$. Both factors are exactly what a probe returns (§6).

**Hypotheses — and one of them is stronger than the page states.**
- The scheme actually **solves an interface equation**, so that $\lambda^\dagger$ exists and an operator mismatch is amplified by $1/\beta$. Applied to an *overlapping* scheme the form is not conservative but **uninformative**: measured overestimate $4.6\times10^{7}$ (implied $C_\mu=2.2\times10^{-8}$).
- $\lambda^\star$ must **satisfy the exact interface equation** $\Lambda\lambda^\star=\chi$, or the first-order perturbation has no reference point. For a one-step map of an *initial*-boundary-value problem no such $\Lambda$ exists — flux balance is a steady condition. This was measured (§6.3) and is recorded on [[probed-dtn-coupling]] §2.3; it is **not** carried back into this bound's hypothesis list. See §17.

## 4.4 Bounding $\sigma$ — branch B, overlapping

An overlapping scheme poses no interface equation. What plays $\lambda^\dagger$'s part is the **stale artificial-boundary datum**, $\delta\lambda:=\lambda^{\text{lagged}}-\lambda^\star\big|_{\partial\Omega_i\setminus\partial\Omega}$. Two declared quantities stand between it and the assembled field — **reach** and **weight**:

$$\boxed{\;\Pi:=\max_j\;\sum_i\chi_{ij}\,\mathbf 1\!\left[\,b_i(j)\le d_i\,\right],\qquad \sigma\le C_\mu\,\Pi\,\lVert\delta\lambda\rVert\;}$$

$d_i=r_i^{\text{sten}}s_i$ is the reach (stencil radius $\times$ sub-steps per exchange) — the number of cells an explicit agent's boundary datum can propagate inward in one exchange interval; $b_i(j)$ is the distance from cell $j$ to $\Omega_i$'s nearest **artificial** face (a face on the real boundary carries no stale datum and does not enter); $\chi_{ij}$ is the partition-of-unity weight.

**The two limits are forced, and they are why the factorization is trusted.** $\Pi=1$ when the overlap is narrower than the domain of dependence — nothing in the blend is clean. $\Pi=0$ when the partition of unity vanishes over the whole contaminated band, and then $\sigma$ is **zero, not small**.

**Measured**: over 16 configurations spanning $\sigma$ by $4.3\times10^{4}$, $C_\mu^{\text{halo}}\in[0.228,1.178]$ — a constant moving $5\times$ while the quantity it relates moves $4\times10^{4}$. $C_\mu=1.2$ is quoted as *measured with a stated scope*: one expert, one governing family, one time discretization.

**The design lesson, stated as a ratio.** Widening the halo from 11 to 61 cells at fixed ramp leaves $\Pi$ unchanged and $\sigma$ within $40\%$; widening the *ramp* at fixed halo takes $\Pi$ from $0.75$ to $0.020$ and $\sigma$ by $1.9\times10^{4}$. **Overlap width is a precondition; the partition of unity is the mechanism.**

## 4.5 Bounding $\gamma$ — the accelerator is a bound, not a preference

$\gamma\le C_\mu\lVert\lambda^{(k)}-\lambda^\dagger\rVert$, and the second factor is whatever the solver guarantees:

| accelerator | $\lVert\lambda^{(k)}-\lambda^\dagger\rVert$ | governed by |
|---|---|---|
| Richardson / classical Schwarz | $\rho_S^{\,k}\lVert\lambda^{(0)}-\lambda^\dagger\rVert$ | the contraction $\rho_S$; one-level $\rho_S\to1$ as subdomains shrink (§8.2) |
| Krylov on the interface | $\sim\bigl((\sqrt\kappa-1)/(\sqrt\kappa+1)\bigr)^{k}$ | $\kappa(\tilde\Lambda)$ — **conditioning**, not subdomain count |
| direct Schur solve | $\sim\kappa\cdot\epsilon_{\text{mach}}$ | **negligible** |

## 4.6 The stability constant, and the one theorem that controls it

$$L\;\le\;\lVert\mathcal A\rVert\cdot\max_i\operatorname{Lip}_v(\mathcal E_i)\cdot\Bigl(1+\frac{C_\mu C_{\mathcal G}}{\beta}\Bigr),\qquad C_{\mathcal G}=\Bigl\lVert\frac{\partial\mathcal G}{\partial u^n}\Bigr\rVert$$

> **$\beta$ sits in the denominator of both $\sigma$ (§4.3) and $L$ (here). An ill-conditioned interface hurts twice — additively per step, multiplicatively across steps.**

$\lVert\mathcal A\rVert=1$ **by proof** for any partition of unity; $\operatorname{Lip}_v(\mathcal E_i)$ and $C_{\mathcal G}$ remain unmeasured.

### The passivity theorem

**Hypothesis.** Each agent is **incrementally passive** at its ports: for two trajectories under the same forcing there is a storage $H_i\ge0$ with

$$\frac{\mathrm d}{\mathrm dt}H_i(\Delta u_i)\;\le\;\int_{\partial\Omega_i}\Delta e_i\,\Delta f_i\,\mathrm ds ,$$

and the interconnection is **power-preserving** — which the port rule $e_A=e_B,\ f_A=-f_B$ *is*, by construction (a Dirac interconnection).

**Conclusion.** Port powers cancel in pairs, $V=\sum_i H_i(\Delta u_i)$ is non-increasing, and in the $V$-norm

$$\boxed{\;L\le1\quad\text{for any graph, at any }N,\ \text{with no global analysis}\;}$$

**Why the word *incremental* is load-bearing.** Plain passivity bounds the **state** — the coupling cannot manufacture energy, which is stability, not accuracy. Incremental passivity (equivalently contraction / monotonicity) is what bounds the **error**. [[composition-error-theory]] §4.1 makes this distinction and it is easy to lose.

**The defect, when it fails.** $\pi_i=\sup_u\bigl[\Delta H_i(u)-\int e_if_i\,\mathrm dt\bigr]^{+}$ **composes additively**, which is precisely the property a flux-matching residual lacks.

## 4.7 The master bound

$$\boxed{\;\lVert e^N\rVert\;\le\;\underbrace{L^N\lVert e^0\rVert}_{\text{initial}}\;+\;\sum_{n=1}^{N}L^{\,N-n}\Bigl[\;\underbrace{\tau^n}_{\text{agent}}+\underbrace{\sigma^n}_{\text{transmission}}+\underbrace{\gamma^n}_{\text{solve}}\;\Bigr]\;}$$

With windows of $W$ macro-steps the $\gamma$ sum runs over $J=N/W$ windows rather than $N$ steps, while $\tau$ and $\sigma$ stay per-step.

**Specializations — the bound is a family.**

| setting | reduction | reads |
|---|---|---|
| monolithic autoregressive model | one subdomain, $\sigma=\gamma=0$ | $\lVert e^N\rVert\le L^N\lVert e^0\rVert+\sum L^{N-n}\tau^n$ |
| exact interface operator, converged | $\tilde\Lambda=\Lambda\Rightarrow\sigma=0$; $\gamma=0$ | $\lVert e^N\rVert\le\sum L^{N-n}\tau^n$ — **the worst-agent conjecture, here a theorem** |
| passive agents | $L\le1$ | $\lVert e^N\rVert\le\lVert e^0\rVert+\sum_n(\tau+\sigma+\gamma)$ — linear in $N$, never exponential |
| contractive | $L<1$ | $\lVert e\rVert\le\dfrac{\max(\tau+\sigma+\gamma)}{1-L}$ — **bounded uniformly in time** |
| periodic windows (measured) | $\Lambda\equiv\tilde\Lambda\equiv0$, $\gamma\equiv0$ bitwise | controlled entirely by $\tau$ and $\sigma$, with $\sigma$ maximal and **every reported diagnostic reading zero** |

**Caveat on row 2 that the table does not carry.** $\tilde\Lambda=\Lambda$ requires the exact Steklov–Poincaré operator, and computing it means solving the problem you decomposed ([[composition-error-theory]] §4.0). The row is a theorem and not a construction; §6 is the construction that approximates it from outside.

## 4.8 When the bound goes vacuous

For a chaotic system $L\approx e^{\lambda_1\Delta t}>1$ with $\lambda_1$ the leading Lyapunov exponent, and the bound exceeds any tolerance at the **predictability horizon**

$$T_{\text{pred}}\;\approx\;\frac{1}{\lambda_1}\,\ln\frac{\delta_{\text{tol}}}{\tau+\sigma+\gamma}.$$

**Past $T_{\text{pred}}$ nothing on this page proves anything, and no construction changes that** — the divergence is the physics, not the method. Beyond it the deliverable must change from a **trajectory** claim to a **statistical** one (invariant measure, spectra, structure functions), which needs a different theory (shadowing under hyperbolicity, or ergodic-average estimates) that the vault does not have. This is envelope hypothesis $\mathrm{E5}$ and gap G6.

**Three branches** ([[end-to-end-architecture-spec]] §11.3): $L>1$ exponential horizon, $L=1$ linear, $L<1$ unbounded. The first measured instance landed on the third — $L=0.948$, so $T_{\text{pred}}$ never arrives and the measured error **saturates**.

---

# 5. Composition error theory

Derived on [[composition-error-theory]].

## 5.1 The conjecture, stated so it can be graded

Let $\varepsilon_i=\lVert\mathcal E_i-\mathcal S_i\rVert_{\mathfrak D_i}$ be agent $i$'s **independent accuracy** — the number a benchmark reports, on the agent's own validation distribution $\mathfrak D_i$. Let $u^\star$ be the composed solution with interface agreement driven to zero and every declared port conservative.

$$\textbf{Conjecture:}\qquad \lVert u-u^\star\rVert = O\!\bigl(\max_i\varepsilon_i\bigr).$$

**Verdict: false as stated; true under three hypotheses; the content is in which one breaks.**

## 5.2 Three distinct failure channels

**(1) Agreement is a consistency condition, not a correctness one.** The Schwarz fixed point satisfies $u_i^\star=\mathcal E_i[u_i^\star;\mathcal R_iu^\star]$ for every $i$, and this is the global solution **only if each $\mathcal E_i$ is the exact restriction of the global operator to $\Omega_i$**. Drop that and the iteration still converges — confidently — to the fixed point of a *different* map. In backward-error language: a perfectly agreeing composition does not approximately solve your problem, it **exactly solves a nearby problem**, and the distance between the two is not bounded by $\max_i\varepsilon_i$.

> **Perfect agreement among unfaithful local solves is not a good state. It is the worst state — the wrong answer with the alarm disconnected.**

**(2) Conservation constrains a codimension-few subspace of the error.** A conservation law says finitely many *linear functionals* vanish:

$$\langle\ell_k,\,e\rangle=0,\qquad k=1,\dots,m,\qquad m=O(\text{ports}\times\text{conserved quantities}).$$

The error lives in a $10^5$–$10^7$-dimensional space. Killing $m\sim10$–$100$ linear functionals leaves $\lVert e\rVert$ unbounded, and **every flux-neutral redistribution is in the kernel of every conservation constraint you can write**. Measured: mass closed at $4.3\times10^{-14}$ while the array-efficiency answer was off by $\approx2\times$.

**(3) Local errors are amplified and transported, not aggregated.** Amplification is §4.1's recursion. Transport is that agent $i$'s output is agent $j$'s boundary data, so along an advective chain errors compose rather than take a maximum. The honest aggregator is adjoint-weighted:

$$e_{\mathcal J}\;\approx\;\sum_i\bigl\langle\phi_i,\ \tau_i\bigr\rangle,$$

with $\phi_i$ the adjoint solution restricted to $\Omega_i$. For $\varepsilon=10^{-2}$ per agent: $\max_i$ says $10^{-2}$; a sum over an 8-agent chain says $8\times10^{-2}$; at $L=1.05$ over 80 steps the same defect says $\approx0.5$. **Three answers, two orders of magnitude, identical $\varepsilon$.**

## 5.3 The correct statement, and the load-bearing substitution

$$\boxed{\;\lVert u-u^\star\rVert\;\le\;C_S\Bigl(\max_i\tau_i+\gamma\Bigr)\;}$$

| symbol | definition |
|---|---|
| $\tau_i$ | **local consistency defect**: $\bigl\lVert\mathcal S_i[u\vert_{\Omega_i};\mathcal R_iu]-\mathcal E_i[u\vert_{\Omega_i};\mathcal R_iu]\bigr\rVert$ — the agent's error **on the restriction of the true global solution, under the boundary data the true solution imposes** |
| $\gamma$ | coupling defect: the interface residual at convergence |
| $C_S$ | stability constant $\lVert(\mathrm I-D\Phi)^{-1}\rVert$ for the linearized composed map |

The conjecture is exactly this bound with three silent substitutions: $\gamma=0$ (fair — that *is* what agreement means), $C_S=O(1)$ (a hypothesis), and $\tau_i=\varepsilon_i$ — **false, and this is the crux.**

**Why $\tau_i\neq\varepsilon_i$.** $\varepsilon_i$ is measured on the agent's own distribution — interior states it trained on, boundary conditions it trained under. $\tau_i$ is measured on the true global solution's restriction, with the boundary data the *other agents* impose. These coincide only if composition presents the agent with data it was validated on, and **composition is a distribution-shift machine by construction**: boundary data is out of distribution before anything else is (a checkpoint trained on periodic boxes has never seen an inflow ring, and may have no channel to accept one); the agent is fed its neighbours' errors, not the truth; and the regime drifts, with $P(\text{at least one of }K\text{ experts OOD})\to1$.

Measured hint: composition error $0.3293$ against a single-step $0.0055$ — a $60\times$ gap for the same operator.

## 5.4 The three hypotheses, graded

| | hypothesis | requires | status |
|---|---|---|---|
| **H1** | **Faithfulness** — each $\mathcal E_i$ approximates the *restriction* of the global operator, including its boundary behaviour, on the data composition presents | $\tau_i\approx\varepsilon_i$ | **violated, badly, measured**: periodic windows are not restrictions of an open channel; interface residual identically zero while the answer was $2\times$ wrong |
| **H2** | **Stability** — the composed map is non-expansive, uniformly in graph size and rollout length | $C_S=O(1)$ | **half-cleared**: bounded in $N$ (under $1\%$ over a $3\times$ graph); **never established in $t$** until the first $L$ fit |
| **H3** | **Exact agreement** at the interfaces | $\gamma=0$ | achievable, and **the least valuable of the three** — enforced to $10^{-14}$ with no effect on the answer |

> **The conjecture is wrong because H1 and H2 are the entire content, and H3 — the one the conjecture names — carries the least weight.**

## 5.5 The reframing: agreement is an operator

The exact non-overlapping interface condition is not "match the values" but the **Steklov–Poincaré (Dirichlet-to-Neumann) operator**

$$\Lambda_i:\;g\longmapsto\partial_n\mathcal S_i[g]\big|_{\partial\Omega_i},$$

which encodes everything the rest of the subdomain does in response to boundary data. Composing with the exact $\Lambda_i$ reproduces the global solution exactly. Every practical transmission condition is an approximation to it, and they form a ladder:

$$\text{Dirichlet}\prec\text{Neumann}\prec\text{Dirichlet–Neumann}\prec\text{Robin}(\alpha)\prec\text{optimized Robin}\prec\text{Ventcell}\prec\Lambda_i$$

$$\boxed{\;\text{composition accuracy is set by }\lVert\Lambda_i-\tilde\Lambda_i\rVert,\ \textbf{not}\ \text{by how tightly }\tilde\Lambda_i\text{ is satisfied}\;}$$

**Do not tighten the agreement; improve the operator being agreed on.** The formal version of this sentence is §4.3's $\sigma$ bound.

## 5.6 Weak matching and the inf-sup constant — the theorem the conjecture wants

Mortar methods enforce interface continuity **weakly** through a multiplier $\lambda$ on an interface space $M_\Gamma$:

$$\int_\Gamma\lambda\,(u_A-u_B)\,\mathrm ds=0\qquad\forall\lambda\in M_\Gamma,$$

and when $M_\Gamma$ satisfies a discrete **inf-sup (LBB) condition**

$$\inf_{\lambda\in M_\Gamma}\ \sup_{v}\ \frac{\int_\Gamma\lambda\,[v]\,\mathrm ds}{\lVert\lambda\rVert\,\lVert v\rVert}\;\ge\;\beta>0,$$

the method is **optimal**: the global error is bounded by the sum of the subdomains' own best-approximation errors, with a constant independent of the decomposition. **That is the worst-agent bound, and $\beta$ is the hypothesis the conjecture is missing.**

**Two immediate consequences.** Pointwise matching of *both* $e_A=e_B$ and $f_A=-f_B$ between two inexact operators is the **over-constrained** rung — generically infeasible, with locking ($\beta$ collapses) and spurious interface modes as the classical symptoms. And $\beta$ is a reportable per-port number, giving the enforce-or-measure rule the vocabulary it lacked for saying *this port is badly conditioned*.

**Hypothesis a frozen expert cannot supply.** The classical $\beta$ is *proved* from approximation theory of the trial spaces. A frozen operator has no trial space and no approximation theory, so here $\beta$ is **measured** as $\sigma_{\min}$ of an assembled matrix (§6.4, §7.3). An inf-sup condition that is measured rather than proved is the only version available to a black box.

---

# 6. Probed Dirichlet-to-Neumann coupling

Derived on [[probed-dtn-coupling]]. The construction that makes §5.5's ladder available to an expert that cannot be told anything.

## 6.1 The observation

**Intuitively.** You cannot open the box, and you cannot ask it to accept a fancier boundary condition. But a Dirichlet-to-Neumann map is *defined* by what it takes and what it returns — a trace in, a flux out — so a box that merely accepts a ring **already is** one. You build the operator by asking it questions, from outside.

$$\Lambda_i:\;\underbrace{g}_{\text{trace on }\partial\Omega_i}\longmapsto\underbrace{\partial_n\mathcal S_i[g]\big|_{\partial\Omega_i}}_{\text{flux returned}}$$

> **The agreement operator does not have to live inside the agent. It has to live between them.**

This is [[symmetry-averaging-atlas-0.1]]'s move — the composition layer supplying a property the expert lacks — applied to the interface condition.

## 6.2 The probe

Choose an interface basis $\{\psi_k\}_{k=1}^{m}$ on each seam. Holding everything else fixed,

$$S_i\,\psi_k=\partial_n\Bigl(\mathcal E_i[\psi_k]-\mathcal E_i[0]\Bigr)\Big|_\Gamma,\qquad k=1,\dots,m .$$

**$m+1$ Dirichlet solves and the block is assembled.** The zero-probe $\mathcal E_i[0]$ is what removes every $\lambda$-independent bias in the agent, and that subtraction is the whole feasibility argument of §6.6.

**The basis dimension is measured, not chosen.** Interface content below the scale at which the expert stops representing flow is not worth probing:

$$m\approx\frac{\text{seam length}}{\text{expert's effective cutoff}}\quad\text{— for the wind farm, } \frac{2D}{0.125\,D}=16\ \text{modes per component}.$$

## 6.3 The interface equation, and the case in which it is wrong

$$\boxed{\;\mathcal G(\lambda):=\sum_i\Lambda_i\lambda-\chi=0\;}\qquad \chi=-\sum_i\partial_n\mathcal E_i[0]\big|_\Gamma$$

For a **linear** problem $\mathcal E_i$ is affine in $\lambda$, $\Lambda_i$ is a genuine linear operator, and this is the **Steklov–Poincaré (Schur complement) equation**; solving it reproduces the undivided solution exactly.

> **Measured refutation, and it is the sharpest scope limit in the vault.** Every expert this framework can probe is the **one-step map of an initial–boundary-value problem**, not the solution operator of a boundary-value problem. Solving $S\lambda=\chi$ exactly made one composed macro-step $8.9\times$ **worse**; substituting the reference's own post-step trace left the residual essentially unchanged, so the true trace **does not satisfy the condition**; and refining $\Delta t$ made the overshoot *grow* ($41\times$, $75\times$, $150\times$), which names it — **flux balance is a steady condition.**
>
> The mechanism is sharper still and it is spatial, not temporal. At a shared-layer seam both sides pin the same cell, so with outward normals
> $$F_A+F_B=\nu\,\frac{2w_\Gamma-w_{A,\text{in}}-w_{B,\text{in}}}{h}=-\,\nu h\,\partial_{nn}w,$$
> verified bit-exactly against the monolith. **The flux-balance residual at a shared-layer seam is a discrete second derivative, not a jump** — $O(h)$ on the exact solution, vanishing only where that solution is linear across the seam. Driving it to zero asks the solution to be straight at every cut, and no amount of integrating in time removes it. A time-integrated ($W=2$ waveform) version was built and is *worse*, not neutral.
>
> **What replaces it for an explicit one-step map: the overlapping halo update, bounded by §4.4's $\Pi$.**

## 6.4 What falls out of the same matrix

Four items from [[composition-error-theory]] §4's list, obtained from one assembly.

**(a) The port algebra already specifies the probe.** Reading `MECH` in the order the probe uses it,

$$\underbrace{\mathbf v}_{\text{flow — the Dirichlet trace imposed}}\;\longmapsto\;\underbrace{\boldsymbol\sigma\!\cdot\!\mathbf n}_{\text{effort — the flux returned}}\qquad\textbf{is }\Lambda_i .$$

The port algebra had specified both sides of the DtN interface, in the right units with the right sign convention, and never treated the pair as an **operator to be identified**.

**(b) Passivity is positive-realness.**

$$\text{agent }i\text{ incrementally passive at its ports}\iff\Lambda_i\text{ positive real}\iff\tfrac12\bigl(S_i+S_i^{\!\top}\bigr)\succeq0$$

so the passivity certificate is an **eigenvalue computation on a matrix you already assembled**, not a statistic gathered over rollouts. With $\mu_i=\lambda_{\min}\bigl(\tfrac12(S_i+S_i^{\!\top})\bigr)$, the defect takes a **scale-relative noise floor**:

$$\pi_i=\begin{cases}\lvert\mu_i\rvert,&\mu_i<-\eta_i\\[2pt]0,&\text{otherwise}\end{cases}\qquad \eta_i=c\,\varepsilon_{\text{mach}}\lVert S_i\rVert,\quad c=\dim S_i .$$

**The floor is not fastidiousness**: without it a matrix positive semidefinite *by construction* returns $\mu_i\approx-10^{-15}$ from an eigensolver, enough to decertify the $L\le1$ branch on a clean graph. $\mu_i$ is emitted raw alongside $\pi_i$ so the clipping is visible. The **eigenvector names which interface mode is being amplified** — nothing else in the framework localizes an instability to a mode.

**Hypothesis this equivalence needs and does not state.** It holds *once the flux convention is pinned*, and the page proposes two incompatible ones (§17.4). The per-block verdict flips between them; **the verdict belongs to the seam, not the block**.

**(c) $\beta$ and $\kappa$ are singular values.**

$$\beta=\sigma_{\min}\bigl(S\vert_{\text{constrained subspace}}\bigr),\qquad \kappa(S)=\sigma_{\max}/\sigma_{\min}$$

Both printable per port. This generalizes the hard-won conditioning lesson — *a projection whose sensitivity is not $O(1)$ is not a projection* — from one thrust constraint to every interface.

**(d) The optimal Robin coefficient is read off, not derived.** Optimized Schwarz chooses $\alpha$ in $\partial_nu_i+\alpha u_i=\partial_nu_j+\alpha u_j$ to minimize the convergence factor, and the optimum is the **symbol of the exact DtN**. You cannot Fourier-analyze a checkpoint whose PDE you do not know; in a Fourier interface basis the diagonal of $S_i$ **is** the measured symbol:

$$\alpha^\star_k=\bigl(S_i\bigr)_{kk}\ \text{(measured)}.$$

## 6.5 The composability index

$$\Xi_i:=\frac{\bigl\lVert\Lambda_i^{\text{expert}}\bigr\rVert}{\bigl\lVert\Lambda_i^{\text{ref}}\bigr\rVert}$$

— how much of the true boundary response the agent actually reproduces. It turns a yes/no eligibility question into a **norm**, and it recovers the vault's sharpest existing finding as an exact value:

$$\text{frozen periodic expert}\Longrightarrow\Lambda_i\equiv0\Longrightarrow S\equiv0\Longrightarrow\mathcal G(\lambda)=-\chi\ \text{for every }\lambda .$$

**The interface problem is not badly conditioned; it is empty.** Every trace is equally consistent, because no agent's output depends on any agent's input. Measured on a real periodic solver: every entry of $S$ bitwise zero, $\beta=0$, $\Xi=0$, with 34 solver calls confirming the probe genuinely ran. The compiler refuses the direct and Krylov solvers on an empty operator (rule `R6`) and refuses the word "coupled" on any output involving that seam.

**Scope limit measured later and worth carrying with the definition.** $\Xi$ is a function of the **probe state**, not of the expert alone: same-regime replicates disagree by up to $49\%$ ([[case-study-reuse-probe-atlas-0.1]]). It is proposed as a *ranking* axis for a library, and its reproducibility had never been quoted when it was proposed.

## 6.6 The probe floor

The obvious objection is that a checkpoint's per-call noise floor is $3\times10^{-2}$ and a finite difference divides by $\epsilon_{\text{p}}$. **The objection is wrong, and precisely why matters.** That figure is a *systematic, state-dependent model error* — the same input returns the same output bitwise — and a finite difference **subtracts it off**:

$$\frac{\mathcal E[\lambda+\epsilon_{\text{p}}\psi]-\mathcal E[\lambda]}{\epsilon_{\text{p}}}\qquad\text{— any }\lambda\text{-independent bias cancels in the numerator.}$$

What does not cancel is genuine non-reproducibility, of which exactly one source is measured: the expert's forward pass depends on a window's **position in the batch** at the $10^{-6}$ level. So probe error $\sim10^{-6}/\epsilon_{\text{p}}$, and at $\epsilon_{\text{p}}=10^{-2}U_\infty$ that is $\sim10^{-4}$, four orders below the signal — **and it is removable**, by pinning the batch layout.

**Three probe classes**, a finer selection axis than the ladder rung and invisible to any accuracy benchmark: *probe-cheap* (differentiable → exact JVP, no $\epsilon_{\text{p}}$ at all), *probe-affordable* (deterministic and smooth → finite differences at the reproducibility floor), *probe-expensive* (opaque → regularized regression over $m'>m$ probes, error averaging down as $\eta/\sqrt{m'}$ instead of amplifying as $\eta/\epsilon_{\text{p}}$).

**Cost, stated at its worst.** Naive assembly is $\approx200\times$ the macro-step and unaffordable as stated; $2$–$3\times$ after equivalence classes and refreshing every $K$ steps. And every metric is **local to a state** — this is Newton–Krylov–Schur under nonlinearity.

---

# 7. Interface transfer

Derived on [[interface-transfer-theory]]. Six previously separate importable mechanisms turn out to be one declared object.

## 7.1 The object

Three spaces per seam, not two: $V_A$, $V_B$ (what each side can accept and return on its own discretization) and $M$, a **common interface space** chosen once per seam. Each side declares **one** operator, the prolongation

$$P_i:M\longrightarrow V_i .$$

**The reduction is forced.** Requiring that the transfer neither creates nor destroys interface power gives $\langle R_ie_i,\mu\rangle_M=\langle e_i,P_i\mu\rangle_{V_i}$ for all $\mu$, which is the definition of an adjoint:

$$\boxed{\;R_i=P_i^{\ast}\;}$$

**One declared operator per side per seam; the other is its adjoint, and any other choice leaks power at the interface.**

**Revised connection rule** (superseding the geometric-coincidence clause of §3.3): two agents may share an edge for port type $P$ iff both declare $P$, a common space $M$ is declared, and each side declares a $P_i$ that is **stable** ($\lVert P_i\rVert$ bounded) and **implementable** (the agent can accept $P_i\mu$ as boundary data and return a flux). Geometric coincidence is the special case $M=V_A=V_B$, $P_i=\mathrm{id}$.

## 7.2 Conservative vs. consistent mapping is derived

A **consistent** map is an interpolation: it reproduces constants (rows sum to one). Efforts — intensities: temperature, traction, pressure — map this way. A **conservative** map preserves integrals: it is the transpose of a consistent map in the opposite direction. Flows — fluxes — map this way. Since the effort map is $P_i$ and the flow map is $P_i^{\ast}$:

$$\text{effort}\ \xrightarrow{\ P_i\ }\ \text{consistent},\qquad \text{flow}\ \xrightarrow{\ P_i^{\ast}\ }\ \text{conservative}.$$

**Getting it backwards stops being expressible.** The classic partitioned-coupling bug requires two independent declarations to disagree; with one declaration there is nothing to disagree with. **Field↔lumped adjointness is the same condition at $\dim M=1$** — a lumped port is a one-dimensional common space, and the actuator disk was not missing a check, it was missing a declaration.

**What is still declared and can still be wrong.** $P_i$ itself. A wrong interpolation order, support or quadrature is a real error and is **not** caught by adjointness, because $P_i^{\ast}$ is wrong consistently with it. Adjointness buys **no power leak**, not accuracy.

## 7.3 The composite interface operator on $M$

$$\bigl(\tilde\Lambda_i^{M}\bigr)_{lk}=\bigl\langle\mu_l,\ P_i^{\ast}\,\partial_n\bigl(\mathcal E_i[P_i\mu_k]-\mathcal E_i[0]\bigr)\bigr\rangle_M,\qquad\text{i.e.}\qquad \tilde\Lambda_i^{M}=P_i^{\ast}\,\Lambda_i\,P_i .$$

1. **It is a Galerkin projection** — the compression of the true DtN onto $M$; classical theory of Galerkin compressions applies unmodified.
2. **Positive-realness survives**, because $P_i^{\ast}\Lambda_iP_i$ is a **congruence**: $\tfrac12\bigl(P_i^{\ast}\Lambda_iP_i+(\cdot)^{\!\top}\bigr)=P_i^{\ast}\,\tfrac12(\Lambda_i+\Lambda_i^{\!\top})\,P_i\succeq0$. **If the pair is not adjoint** the expression is $R_i\Lambda_iP_i$ with $R_i\neq P_i^{\ast}$, the congruence structure is destroyed, and positive-realness is no longer inherited. That single line is the mechanism by which a missing adjointness declaration **silently disables the passivity theorem**.
3. $\tilde\Lambda^M=\sum_iP_i^{\ast}\Lambda_iP_i$, so $\beta$ and $\kappa$ read off exactly as in §6.4, on $M$ rather than on a shared grid.

**Choosing $M$, and the dimension rule.** The inf-sup condition requires $M$ **no finer than the coarser of the two sides**. Representing the trace more finely than the coarser expert can respond to buys nothing, because $\Lambda_i$ returns noise there — the same argument that sets the probe basis:

$$\boxed{\;\dim M=\min_i\ m_i^{\text{eff}}\;}$$

**The multiplier space and the probe basis are the same space**, so mortar costs no probes beyond those already budgeted.

**Two new $\sigma$ components, neither reduced by iteration.**

$$\sigma_{\text{nc}}\le\frac{C_\mu}{\beta}\bigl\lVert(I-\Pi_M)\lambda^\star\bigr\rVert,\qquad \sigma_{\text{time}}\le\frac{C_\mu}{\beta}\bigl\lVert(I-\Pi_W)\lambda^\star(\cdot)\bigr\rVert_{[t^n,t^{n+1}]}$$

with $\Pi_M$ the $M$-projection and $\Pi_W$ the projection onto the temporal representation used across the window. Both belong in $\sigma$, not in $\tau$ — the agents are innocent of them.

## 7.4 The time axis, and the nondimensionalization identity

The transfer is a tensor product of a space rule and a time rule, **each with a conservative and a consistent member**:

| | conservative (integral preserved) | consistent (pointwise preserved) |
|---|---|---|
| **space** ($\Gamma$) | flow map $P_i^{\ast}$ | effort map $P_i$ |
| **time** ($\Delta t$) | time-integrated flux matching (`R9`) | higher-order $\lambda(t)$ representation (waveform relaxation) |

**Rule `R9`, stated.** Over a macro-step $[t^n,t^{n+1}]$ the coupling condition on a flow variable is

$$\int_{t^n}^{t^{n+1}}\!\!\int_\Gamma f_A\,\mathrm ds\,\mathrm dt\;+\;\int_{t^n}^{t^{n+1}}\!\!\int_\Gamma f_B\,\mathrm ds\,\mathrm dt\;=\;0,$$

evaluated with each side's **own** substep quadrature. Pointwise-in-time matching is **refused** whenever the substep counts differ, because its residual converges and means nothing — the same silent-wrongness signature as a bitwise-zero interface residual.

**Two consequences it did not previously carry.** *Interface representation order caps the scheme order*: a second-order expert coupled with a zeroth-order interface representation is a first-order method. And *stability is set by coupling stiffness, not by either agent's step* — the admissible macro-step can be **shorter** than $\min_i\Delta t_{\text{native}}$, which is the opposite direction from `R4`.

**Nondimensionalization** factors the prolongation as $P_i=D_i^{-1}\hat P_i$, with $\hat P_i$ the geometric transfer and $D_i$ the unit conversion, subject to the power identity

$$\boxed{\;s_e^{(i)}\cdot s_f^{(i)}=s_P^{(i)}\;}$$

for every declared port type. A scale set violating it is incomplete or inconsistent, and either is a compile-time refusal. Completeness is decidable per port type from the port list alone.

## 7.5 Cross-points and the null-space check

At a vertex where three or more subdomains meet the transmission conditions of the incident seams are **not independent**. The classical fix (FETI-DP, BDDC) makes cross-point degrees of freedom **primal** — single-valued, solved in a small coarse problem — and keeps the rest of the seam dual.

**The decision is conditional on the decomposition axis.** Overlapping with a partition of unity: the vertex is handled, **no action** — which is why 124 tiles meeting four-at-a-corner have never caused trouble. Non-overlapping (which probed-DtN requires): cross-point DOFs are primal, removed from $M$ and appended as constraints; a non-overlapping decomposition with a cross-point and no primal set is refused.

**The detector is free**, riding on the probe's null-space check:

$$\dim\ker\bigl(\tilde\Lambda^M_\Gamma\bigr)=\underbrace{n_0(\Gamma)}_{\text{per seam, declared}}+\underbrace{\#\{\text{untreated cross-points on }\Gamma\}}_{\text{should be }0}$$

**$n_0(\Gamma)$ has three inputs, all declared**: which agents meet there, the vertex valence, and **where the elliptic solve lives**:

$$n_0(\Gamma)=1\ \text{if the elliptic solve is inside }\Lambda_i,\qquad 0\ \text{if it is in the composition layer}.$$

Known values: $1$ at a fluid–fluid `MECH` seam with incompressibility inside $\Lambda_i$ (rank-one from $\oint_\Gamma\lambda\cdot n=0$); $0$ at a fluid–fluid seam with the projection exposed; $0$ at a field↔lumped seam (the lumped side responds in the direction incompressibility leaves open); and a fourth row for a free solid–solid `MECH` seam's rigid translation normal to the face. **Carrying $n_0=1$ as a per-port-type constant produces a false defect report on every rotor face**, which is how the per-seam rule was found.

**Both directions get a verdict**: an **excess** is `refuse` (the interface system is singular in a direction nothing handles); a **deficit** is `admit-uncertified` (either the probe is not resolving a constraint the physics has, or $n_0(\Gamma)$ is wrong — not decidable from the number alone).

---

# 8. Schwarz iteration

Derived on [[schwarz-iteration-atlas-0.1]].

## 8.1 The iteration and the faithfulness hypothesis

**Additive Schwarz** — every subdomain solves from the previous iterate's boundary data, all updated together:

$$\mathbf u_i^{(k+1)}=\mathcal E_i\!\left[\mathbf u_i^{(k)};\ \mathcal R_i\mathbf u^{(k)}\right],\qquad i=1,\dots,K\ \text{simultaneously.}$$

**Multiplicative Schwarz** solves in sequence with the latest neighbour data — faster per iteration and **order-dependent**.

**The load-bearing theorem.** The fixed point $\mathbf u^\star$ satisfies $\mathbf u_i^\star=\mathcal E_i[\mathbf u_i^\star;\mathcal R_i\mathbf u^\star]$ for every $i$, **and if each $\mathcal E_i$ is the exact restriction of the global operator to $\Omega_i$**, the pieces agree on overlaps and $\mathbf u^\star$ solves the global problem. **Drop that hypothesis and the theorem gives nothing**: the iteration still converges, to the fixed point of the wrong map, and reports no distress.

## 8.2 The periodic-window identity result

**Measured, and stronger than "converges to the wrong answer".**

$$\text{periodic window}\;\Longrightarrow\;\text{no boundary channel}\;\Longrightarrow\;\text{additive Schwarz is the \textbf{identity}}$$

Because `gather` reads each tile's whole window out of the global field at $t^n$ and the operator returns one field, the map $\mathbf u^{(k+1)}=\mathrm{scatter}\bigl(\mathcal E(\mathrm{gather}(\mathbf u^n))\bigr)$ **does not depend on $k$**. Two sweeps agree **bitwise**. So per-window periodicity is not merely a precondition for the iteration converging to the right thing — it is a precondition for the iteration **existing**, and `schwarz > 1` is now refused at configuration time.

This is the same fact §6.5 states as an operator: $\Lambda\equiv0$, $\Xi=0$, the interface problem **empty**.

## 8.3 One level does not scale

For the model SPD elliptic problem, the classical two-sided estimate (Toselli–Widlund):

$$\kappa\bigl(M_{\text{AS},1}^{-1}A\bigr)\le C\,H_\Omega^{-2}\bigl(1+H_\Omega/\delta\bigr),\qquad \kappa\bigl(M_{\text{AS},2}^{-1}A\bigr)\le C\bigl(1+H_\Omega/\delta\bigr)$$

with $H_\Omega$ the subdomain diameter and $\delta$ the overlap width. **The one-level method carries an $H_\Omega^{-2}$ — it degrades as subdomains are added — and a coarse space removes it entirely.** The mechanism is transparent: one-level additive Schwarz advances information exactly **one overlap-connected neighbour per iteration**, so with $2D$ windows at $1.5D$ stride and $7D$ turbine spacing,

$$k_{\min}=\left\lceil\frac{7D}{1.5D}\right\rceil=5\ \text{iterations before turbine 1 can reach turbine 2},\qquad \left\lceil\frac{24}{1.5}\right\rceil=16\ \text{to cross the domain},$$

against a loop that runs three.

**Hypotheses, and they are not small.** The estimate is for a **linear, SPD, variationally-posed** problem with **exact** local solves. A frozen neural operator is none of the three. What survives without the theorem is a statement about the *sparsity pattern* — a dense assembled interface operator couples the whole graph in one solve whereas a local exchange does not (§6.4) — and that is the form the vault claims.

## 8.4 The waveform-relaxation condition

Within one macro-step the subdomain problem is an evolution, not an elliptic solve — the setting of **Schwarz waveform relaxation**. For advection-dominated problems, if no signal can cross the overlap within the time window, classical SWR converges in a **finite** number of iterations, and in the limiting case **one**:

$$U_\infty\,\Delta t_{\text{macro}}\;<\;\delta .$$

## 8.5 Why additive, and the first recorded conflict between two guarantees

Multiplicative converges in fewer iterations; Atlas is additive **by rule** (`R5`), for three compounding reasons: order-independence (a sequential sweep reinstalls graph-cycle path dependence); hardware (all $K$ windows independent within an iteration, which is the batched call already built, worth $\sim5\times$); and — the durable one — **a multiplicative sweep would break the exactness of §9's symmetry averaging**, because mirror-paired tiles would be visited at different points in the iteration and would no longer see equivalent data. That is the vault's first case of two composition-layer guarantees *conflicting*, and it exposed a gap in the port algebra, which defines what a port carries and says nothing about the **order in which ports may be resolved**. `R5` is the answer: *an ordering is admissible iff it preserves the power-preserving interconnection*.

---

# 9. Symmetry averaging

Derived on [[symmetry-averaging-atlas-0.1]]. The composition layer supplying an invariance the frozen expert provably lacks — exactly, without training.

## 9.1 The construction and its proof

**Intuitively.** The box does not know your problem is left–right symmetric, so it returns something lopsided. Ask it twice — once as posed, once with the picture flipped — flip the second answer back, and average. Whatever private preference it had appears with a plus sign in one answer and a minus in the other.

Let $G$ be a **finite group** acting on the field space and $\mathcal E$ any operator. The **Reynolds average**:

$$\tilde{\mathcal E}(\mathbf u)\;=\;\frac{1}{\lvert G\rvert}\sum_{g\in G}g^{-1}\,\mathcal E(g\,\mathbf u)$$

**Claim.** $\tilde{\mathcal E}$ is exactly $G$-equivariant: $\tilde{\mathcal E}(h\mathbf u)=h\,\tilde{\mathcal E}(\mathbf u)$ for every $h\in G$.

**Proof.**

$$\tilde{\mathcal E}(h\mathbf u)=\frac{1}{\lvert G\rvert}\sum_{g\in G}g^{-1}\mathcal E(gh\,\mathbf u)$$

Substitute $g'=gh$. As $g$ ranges over $G$ so does $g'$ — this is precisely the statement that $G$ is **closed under composition**, and it is where the group axioms earn their place. Then $g^{-1}=hg'^{-1}$, so

$$\tilde{\mathcal E}(h\mathbf u)=\frac{1}{\lvert G\rvert}\sum_{g'\in G}h\,g'^{-1}\mathcal E(g'\mathbf u)=h\,\frac{1}{\lvert G\rvert}\sum_{g'\in G}g'^{-1}\mathcal E(g'\mathbf u)=h\,\tilde{\mathcal E}(\mathbf u).\qquad\blacksquare$$

**The proof requires nothing about $\mathcal E$.** It may be nonlinear, learned, discontinuous, or actively hostile to the symmetry. Equivariance follows from the sum's structure alone.

## 9.2 Hypotheses, and why closure is load-bearing

- **$G$ must be a group, not a set.** If $G$ is merely a set of operations the substitution $g'=gh$ does not permute the sum and the conclusion fails. The result is a **smoothed** operator that is not equivariant and **nothing in its output announces the difference**. Concretely $\{e,\mathcal M_x,\mathcal M_y\}$ is not a group — the point reflection $\mathcal M_x\circ\mathcal M_y$ is absent. Closure is therefore **verified numerically at construction** (compose every pair on a random field, require a match to machine precision), which also catches two errors a symbolic check would miss: an operation whose action on fields and on the frame velocity disagree, and one that reindexes correctly while getting a component's sign wrong.
- **The group elements act on tensors, not on sample indices.** A reflection re-signs the component normal to the mirror,
$$\mathcal M_y:\quad u(x,y)\mapsto u(x,-y),\qquad v(x,y)\mapsto -v(x,-y),$$
and the same transformation must be applied to every *spatially constant* vector travelling with the field — the Galilean frame velocity, any uniform forcing. A reindex-without-re-sign passes an involution test either way.
- **For the coupled system, the tiling must be mirror-paired.** Per-window equivariance gives global equivariance only then; verified as zero mirror-unpaired tiles out of 124 before the measurement was trusted.

## 9.3 What it preserves, and what it cannot repair

Averaging is **linear**, so every linear property of the output survives it: divergence-free output, the field's mean, any linear conservation statement. **Accuracy does not.** $\tilde{\mathcal E}$ commits the same errors $\mathcal E$ does; it merely commits them symmetrically. Group averaging repairs a *symmetry* defect and nothing else.

**Contrast with projection, and this is the reusable lesson.** A minimum-norm projection needs a *correction direction*, and one such projection failed because its chosen direction lay in the null space of the constraint it was meant to enforce ($\mathrm dg/\mathrm d\lambda=10^{-6}$ against a residual of $10^{-1}$, reported at post-projection $7.5\times10^{-11}$). **An average over a group cannot be degenerate in that way, because it chooses no direction.** Prefer enforcement by averaging over a declared group to enforcement by projection along a chosen direction, wherever the invariant admits both.

**Cost.** Exactly $\lvert G\rvert$ forward passes. Measured system-level cost $1.7\times$ at $\lvert G\rvert=2$, because the expert calls double and the rest of the macro-step does not.

**Measured.** Isolated, the averaged operator is symmetric **bitwise** — on a lattice whose cell centres are symmetric about the axis the reflection *is* the array reversal, the average is $\tfrac12(a+\mathrm{rev}(a))$, and floating-point addition is commutative, so there is no residue to be small. Coupled, the residual floor is $2.90\times10^{-7}$ and **flat**, against a baseline $2.77\times10^{-2}$ and **growing**; the floor is float32 batch-position dependence, not the construction.

**The Noether caveat.** Noether's theorem applies to continuous symmetries of an action; $\{e,\mathcal M_y\}$ is discrete and yields **no conserved current**. Enforcing the mirror does not by itself enforce a conservation law. Whether a continuous group averaged this way would enforce the corresponding conserved quantity is open and testable.

---

# 10. Temporal error accumulation

Derived on [[temporal-error-accumulation]].

## 10.1 The three growth regimes

**The correction to the usual framing:** growth is exponential **only when $L>1$**. With $\tau+\sigma+\gamma=\delta$ constant the geometric sum closes:

$$\lVert e^N\rVert\le\frac{L^N-1}{L-1}\,\delta=\begin{cases}\dfrac{\delta}{1-L},&L<1\quad\textbf{contractive — bounded uniformly in }N\\[8pt]N\,\delta,&L=1\quad\textbf{non-expansive — linear}\\[8pt]\dfrac{L^N}{L-1}\,\delta,&L>1\quad\textbf{expansive — exponential}\end{cases}$$

(The middle branch is an identity; the top is the $N\to\infty$ limit and the bottom an upper bound.)

> **The error does not accumulate because time passes; it accumulates because the map is expansive.** And the $L<1$ branch is not a curiosity — dissipative Navier–Stokes at moderate $\mathrm{Re}$ *is* contractive on its attractor in the energy norm, so a local solve inheriting the physics' dissipativity inherits $L\le1$ with it.

## 10.2 A monolith obeys the identical law

$$\text{monolithic:}\ \lVert e^N\rVert\le\sum_nL_{\text{mono}}^{\,N-n}\tau_{\text{mono}}^n \qquad \text{composed:}\ \lVert e^N\rVert\le\sum_nL_{\text{comp}}^{\,N-n}\bigl(\tau^n_{\text{comp}}+\sigma^n+\gamma^n\bigr)$$

**Decomposition buys nothing in the *form* of the bound.** Any argument for the architecture resting on composed systems having a fundamentally better error law is wrong.

**Where they differ — and this is the architecture's whole thesis.** Favourably: $L$ **factors**, $L_{\text{comp}}\lesssim L_{\mathcal A}\cdot\max_iL_i$, with the coupling half *designed rather than trained*; and $L\le1$ **composes structurally** through the Dirac interconnection (§4.6). Unfavourably: $\tau$ can be worse and there are $K$ chances for it to be, and $\gamma$ exists at all.

> **The thesis, stated as the inequality it is:** decomposition is worth its coupling cost precisely when $L_{\text{comp}}<L_{\text{mono}}$ by enough to pay for $\gamma$. **A monolithic autoregressive network has no per-part handle on its Lipschitz constant; a port-connected system has one per agent, and they compose.**

## 10.3 Windowing — three mechanisms and the design rule

Enforcing agreement across a window of $W$ macro-steps is **Schwarz waveform relaxation** / **multiple shooting**. The interface unknown stops being a trace and becomes a trajectory: $\lambda\in\Gamma\longrightarrow\lambda(t)\in\Gamma\times[T_j,T_j+W\Delta t]$.

**Mechanism A — fewer injections, a factor of $W$.** With $J=N/W$ windows,

$$\lVert e^N\rVert\le\underbrace{\sum_{n=1}^{N}L^{\,N-n}\tau^n}_{\text{still per-step}}+\underbrace{\sum_{j=1}^{J}L^{\,N-jW}\gamma_j}_{\textbf{count reduced by }W}$$

*provided* the window problem still converges to the same tolerance, so $\gamma_W\approx\gamma$ rather than growing with $W$.

**Mechanism B — superlinear, not linear, convergence on a bounded window.** For parabolic problems SWR's kernel estimates give superlinear convergence in the iteration index; for advection-dominated problems with finite propagation speed convergence is **finite** (§8.4). **But the constant degrades as the window lengthens**, so B pushes toward short windows while A pushes toward long ones.

**Mechanism C — the real payoff: marching becomes a simultaneous solve.** Multiple shooting makes the unknowns the states at the $J$ window boundaries; the Jacobian is block-bidiagonal with blocks $\Phi^{(W)}$ and $I$, so its conditioning is governed by

$$\bigl\lVert\Phi^{(W)}\bigr\rVert\approx L^{W}\qquad\textbf{not}\qquad L^{N}.$$

**It converts an exponential in the rollout length into an exponential in a length you choose.** It does not repeal §10.1 — a genuinely expansive system is genuinely unpredictable — it removes the *method's* own amplification on top of the physics'.

**The design rule.**

$$\boxed{\;W^{\star}\approx\min\Bigl(\underbrace{1/\ln L}_{\text{conditioning, C}},\ \underbrace{W_{\text{SWR}}}_{\text{convergence rate, B}}\Bigr)\;}$$

Worked at $L=1.05$ over $80$ steps: $1/\ln(1.05)\approx20.5$, so windows of $\approx20$ macro-steps, each block conditioned at $1.05^{20}\approx2.7$ against $1.05^{80}\approx49$ for the undivided march — **a factor of $18$ in the amplification the method contributes.** The two mechanisms are in genuine tension and the optimum is measurable rather than derivable for a neural expert.

**And windowing is the one axis a frozen expert can move along.** It cannot refine below its native $\Delta t$, but it can be called $W$ times — `R4`'s asymmetry.

## 10.4 Conservation and agreement are separate constraints

- **Conservation** is a statement about **sums**: $\oint_\Gamma(f_A+f_B)\,\mathrm ds=0$.
- **Agreement** is a statement about **values**: $e_A=e_B$ pointwise on $\Gamma$.

You can have exact conservation with no agreement and agreement with no conservation. The port rule asks for **both pointwise**, which is §5.6's over-constrained rung.

| mechanism | guarantees | cost | limitation |
|---|---|---|---|
| single-valued numerical flux | **conservation exactly, structurally** — an identity of the data structure | **zero** | conserves only what the flux form conserves; **bounds nothing** |
| min-norm projection | the constraint holds to solver tolerance | one linear solve | a degenerate correction direction makes the projection a lie *reported as a tiny residual* |
| weak / mortar enforcement | global error bounded by the subdomains' best-approximation errors, constant $1/\beta$ | one saddle-point solve | needs a well-conditioned $M$; $\beta$ must be **measured** |
| dissipation inequality / incremental passivity | **bounds the error itself**, via $L\le1$ | a storage per expert; one probe | requires the expert to *be* incrementally passive |
| abstention | nothing, and that is the point | a validity declaration | the precondition all four need |

**Only the fourth row constrains $\lVert e\rVert$ rather than finitely many functionals of it, and it does so by acting on $L$ rather than on $\delta$.** That is the asymmetry worth internalizing: conservation and agreement act on the per-step defect, which the bound multiplies by $L^{N-n}$; **passivity acts on $L^{N-n}$ itself.**

> **Effort ordering comes out inverted from intuition: stability ($L$), then faithfulness ($\tau$), then agreement ($\gamma$)** — and $\gamma$ is the cheapest constant to kill and the least valuable to have killed, because driving it to zero removes the only symptom of an unfaithful local solve.

---

# 11. The general coupling scheme

Derived on [[general-coupling-scheme]].

## 11.1 The tuple

$$\boxed{\;\Sigma=\bigl(\ \mathcal D,\ \tilde\Lambda,\ \mathcal O,\ \mathcal K,\ \mathcal C,\ W,\ \varepsilon_{\text{tol}}\ \bigr)\;}$$

| axis | choices | what it sets | bound term |
|---|---|---|---|
| $\mathcal D$ **decomposition** | `overlapping(δ)` · `non-overlapping` | whether a partition of unity or a trace is the primitive; whether cross-points exist | $\rho_S$ |
| $\tilde\Lambda$ **transmission** | `dirichlet` ≺ `neumann` ≺ `dirichlet-neumann` ≺ `robin(α)` ≺ `optimized-robin` ≺ `ventcell` ≺ `probed-DtN` | the rung — *what agreement means* | $\boldsymbol\sigma$ |
| $\mathcal O$ **ordering** | `additive` · `multiplicative` · `restricted-additive` | order-dependence; composability with other guarantees | $\rho_S$ |
| $\mathcal K$ **accelerator** | `richardson` · `krylov` · `newton-krylov` · `direct-schur` | how $\lambda^{(k)}\to\lambda^\dagger$ | $\boldsymbol\gamma$ |
| $\mathcal C$ **levels** | `one` · `two(coarse)` | global information transfer per iteration | $\rho_S$ at large $K$ |
| $W$ **window** | $1,2,\dots$ macro-steps | step-wise coupling vs. waveform relaxation | $\gamma$ count; the method's share of $L$ |
| $\varepsilon_{\text{tol}}$ | residual target on $\lVert\mathcal G\rVert$ | when to stop | $\gamma$ |

**$\tau$ appears in no row.** It is a property of the agents, not of the coupling — the formal statement that faithfulness is a *precondition* of coupling helping, not a peer of it. **No choice of $\Sigma$ can reduce $\tau$.**

**Every coupling method the vault has argued about is one point in this space** — one-pass exchange, the halo variant, the thrust fixed point (a mis-specified $\varepsilon_{\text{tol}}$, not a different method), the refused Schwarz stage, `WindowNS`, the probed Schur solve, waveform relaxation. Six design debates were six coordinate settings.

## 11.2 The capability record

An expert publishes, alongside its port list: `bc_channel` (`none`|`dirichlet`|`neumann`|`robin`|`ventcell`), `bc_time_varying`, `differentiable` (`none`|`jvp`|`vjp`|`both`), `dt_native`, `L_native` ($\ell_w$), `regime_law` (e.g. $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p\ell_w^2)$), `ports`, `storage` ($H_i$ or none), `equivariances`, `validity`. Later additions include `elliptic_subsolve` (§7.5) and `time_discretization` (§6.3).

**Two fields carry disproportionate weight and are invisible to every accuracy benchmark:** `storage`, which turns §4.6's passivity theorem from a hope into a check, and `validity`, the abstention precondition. **The frozen wind-farm checkpoint declares `bc_channel: none`, and that single field is the formal cause of every coupling limitation recorded against it.**

## 11.3 The nine admissibility rules

| rule | statement |
|---|---|
| `R1` | **the interface is as weak as its weakest agent**: $\operatorname{rung}(\tilde\Lambda)\le\min_i\operatorname{rung}(\texttt{bc\_channel}_i)$ |
| `R2` | **probing is the exception that makes `R1` survivable**: `probed-DtN` needs only `bc_channel ≥ dirichlet`, because the higher rung is built *outside* the expert. **It decouples the achievable rung from the expert's own interface** |
| `R3` | $W>1$ requires `bc_time_varying` on every agent at that interface |
| `R4` | the exchange interval cannot go below $\max_i$ `dt_native`. **A frozen expert can coarsen but never refine** |
| `R5` | `multiplicative` is forbidden when any composition-layer guarantee depends on equivalent treatment of agents (§8.5). A `direct-schur` solve satisfies this automatically, being order-free |
| `R6` | `newton-krylov`/`direct-schur` require $\tilde\Lambda\not\equiv0$ — otherwise the interface system is not ill-conditioned but **empty** |
| `R7` | probing method follows `differentiable` (§6.6's three classes) |
| `R8` | `two(coarse)` via a classical coarse solve needs an agent that can run at a different $\ell_w$ without changing its regime; **via a probed Schur complement it requires nothing extra**, since every call stays at $\ell_w^{\text{native}}$ |
| `R9` | under multirate, conservation must be enforced on the **time-integrated** flux (§7.4) |

Later additions: `R10` refuses to decompose an agent whose `elliptic_subsolve` is `embedded` (decomposing it changes the operator rather than restricting it); `R10b` requires the exchange cadence to equal the agent's own sub-step cadence for a split-step agent; `R2b` gates probed-DtN on `time_discretization` (§6.3).

## 11.4 The compiler, and two lines that deserve emphasis

The scheme is **derived**, not debated: the rung is the highest admissible under `R1` lifted by `R2` where a probe is affordable; ordering is always additive; the accelerator is `direct-schur` if `R6`/`R7` are satisfiable and the budget allows, else `krylov` if $\tilde\Lambda\neq0$, else `richardson`; levels are `two(probed-schur)` if `R8` via probing; the window is $\mathrm{clamp}(\mathrm{round}(1/\ln\hat L),1,W_{\max})$; the decomposition is non-overlapping iff the rung is `probed-DtN`, else `overlapping(δ ≥ U_∞Δt)`.

**$\varepsilon_{\text{tol}}$ is set relative to the other terms, and tighter is refused:**

$$\varepsilon_{\text{tol}}\ \text{such that}\ \gamma\lll\min(\hat\tau,\hat\sigma).$$

Converging the interface far below $\tau$ and $\sigma$ is the precise form of spending compute to shrink the negligible term while removing the only alarm.

**$W$ needs $\hat L$, which was long unmeasured. Until it is, $W=1$ is the honest default and the compiler should say so rather than guess.**

**The `emit` line is the contract, not optional instrumentation**: every coupled macro-step emits $\gamma=\lVert\mathcal G(\lambda^\star)\rVert$, $\beta=\sigma_{\min}(S)$, $\kappa(S)$, $\pi$, $\mathcal R(t)$ and the bound estimate — every one a by-product of machinery the solve already built.

**Two structural traps.** *Cross-points*, which overlapping-with-partition-of-unity avoids and which moving to `probed-DtN` therefore **introduces** (§7.5). And *the degenerate-axis trap*: **an axis whose value cannot affect the answer must be refused at config time, not measured**, citing the rule it failed.

**Three verdicts, not two** ([[end-to-end-architecture-spec]]): `admit`, `admit-uncertified`, `refuse` — split by the rule *refuse the silent-wrongness class, decertify the unverified-hypothesis class*.

---

# 12. Composition of compositions

Derived on [[plug-in-composition-theorems]]. Three properties a plug-in architecture asserts and had never stated.

**Closure.** A connected subgraph with its internal ports matched **is** a legal expert, its transmission operator being the Schur complement

$$\tilde\Lambda_{EE}-\tilde\Lambda_{EI}\tilde\Lambda_{II}^{-1}\tilde\Lambda_{IE},$$

so nesting is not a new mechanism — it is **stopping the existing elimination early**. Legal exactly when $\beta_{\text{int}}>0$, and **grouping-invariant** by the quotient property, so a build need not commit to a nesting depth. The capability record composes in 8 of 13 fields and fails in 3 (`L_native`, `regime_law`, `validity` — all *operating-point* fields).

**Attribution.** From the master bound at $N=1$:

$$\tau_{\mathcal C}\;\le\;\tau^{\text{int}}+\sigma^{\text{int}}+\gamma^{\text{int}}$$

— **a subassembly's transmission and solve error *become* agent error one level up, so the three-way split is not invariant under regrouping.** Every emitted defect must carry its depth, and any expert/architecture error share is depth-relative. This was then *observed*: a $\tau$ measured at $99.8\%$ of the defect turned out to be a per-window pressure projection, i.e. a $\sigma$ one level down wearing a $\tau$ label.

**Substitution.** A certificate from **two probes and no rollout**, by Weyl's inequality on $\beta$:

$$\lVert\Delta\rVert<\beta-\beta_{\min},\qquad \Delta=\Lambda^{\text{new}}-\Lambda^{\text{old}} .$$

**Passivity is the only property in the framework that survives a swap uncertified**, which promotes the storage declaration from an accuracy nicety to the enabling condition of plug-in. **There is no monotonicity theorem: a strictly better expert can make a composition worse**, through the $\beta$ that sits in the denominator of both $\sigma$ and $L$.

**Measured scope limit.** Each threshold is a difference of two nearly equal norms ($0.21$ against $0.22$ at an interesting seam), so **any certificate decided near $\lVert\Delta\rVert\approx\beta$ inherits the amplification — which is the regime a *good* substitution is in.** Certificate thresholds moved $155\%$ with the probe state while $\lVert\Delta\rVert$ moved $3.63\%$ and $\beta$ moved $6.46\%$.

**Abstention gains a second horizon**, degrading **linearly in library size**:

$$T_{\text{abs}}\approx\frac{\Delta t}{Kp},$$

with $p$ the per-call probability an expert is out of distribution. For a large library the binding limit may be abstention, not chaos.

**`validity` is falsifiable but not verifiable**, making it the architecture's single unverifiable axiom — and every bound in §4–§10 is conditioned on it.

---

# 13. The Noether 1.1 dynamics core

The other architecture track. Derived on [[backbone-1.1]], [[discovered-conservation-1.1]], [[post-decoder-diffusion-1.1]]. Included because §3.1's port-Hamiltonian structure and §13.1's are the same structure, reached from opposite directions.

## 13.1 The port-Hamiltonian backbone

**Intuitively.** Each layer is one small, stability-controlled Euler step of a learned flow with two parts, matching the two things physics does: a **conservative** part that rotates latent state without gaining or losing energy, and a **dissipative** part that can bleed it. **Depth is latent time.**

Per layer $\ell$, over the typed multigraph, with ReZero gains $\alpha_\ell,\alpha'_\ell$ and step $\epsilon$:

$$h_i^{\ell+1}=h_i^{\ell}+\alpha_\ell\,\epsilon\,\tanh\!\Big[\underbrace{(W_{\text{skew}}-W_{\text{skew}}^{\!\top}-\gamma_\ell I)h_i^\ell}_{\text{conservative rotation}\;+\;\text{learnable dissipation}}+\sum_{\theta}\sum_{j\in\mathcal N_\theta(i)}g_\theta\,W_O^{(\theta)}A_{ij}^{(\theta)}\bigl(v_j^{(\theta)}-g_\theta v_i^{(\theta)}\bigr)\Big]$$

$$h_i^{\ell+1}\mathrel{+{=}}\alpha'_\ell\,\epsilon\,W_2\,\mathrm{renorm}\bigl(\mathrm{GELU}(W_1h_i^{\ell+1})\bigr)$$

with per-edge-type attention using **tied $Q=K$**:

$$q_i^{(\theta,k)}=k_i^{(\theta,k)}=W^{(\theta,k)}h_i,\qquad A_{ij}^{(\theta,k)}=\tanh\!\Bigl(\frac{q_i^{(\theta,k)}\cdot q_j^{(\theta,k)}}{\sqrt{d_h}}\Bigr).$$

**Newton's third law, split into two independently-enforced mechanisms.** Tying $Q=K$ makes the score **reciprocal** by construction ($q_i\cdot q_j=q_j\cdot q_i$) — the "equal strength" half. The antisymmetric flux $(v_j-g_\theta v_i)$ supplies the "opposite direction" half: at $g_\theta=1$ the aggregation telescopes, $\sum_i\Delta_i=0$, so whatever $i$ gains $j$ loses exactly.

**The gate, and why its parameterization matters.** $g_\theta\in[0,1]$ is a learnable per-type gate on the flux symmetry, default $1$ on same-field transport edges and free elsewhere — because equal-and-opposite is right for transport and **wrong for forcing** (buoyancy, external drive). A sigmoid is the wrong primitive: it reaches $1$ only in the limit, parks near $0.99$, and leaks a little momentum *every layer*, accumulating roughly linearly over a rollout. Use a **clamped gate** $g_\theta=\mathrm{clamp}(a_\theta,0,1)$, which hits exactly $1$ on an open set, exactly $0$ at the pure-forcing endpoint, and has zero gradient at the rails so a genuinely conservative edge **locks in**.

**What the exact gate does not buy — the soft/hard split.** Even at $g_\theta=1$ exactly, the layer is **not** exactly momentum-conserving, because the per-node $\tanh$ wraps the aggregated message: $\sum_i\tanh(\Delta_i)\neq\tanh(\sum_i\Delta_i)$. The telescoping is a property of the *pre-nonlinearity aggregate* — of **depth**, not of the decoded physical trajectory. So the exact gate sharpens the *soft* conservation and shrinks what a corrector must clean up; the **hard** guarantee against rollout drift is the live projection of §13.2.

**Cayley parameterization for stability.** $W_{\text{skew}}=(I-S)(I+S)^{-1}$ of a learned skew generator $S$ is orthogonal by construction, which bounds $\lVert W_{\text{skew}}-W_{\text{skew}}^{\!\top}\rVert_2\le2$ **through training, not just at initialization**, so the Euler-stability condition $\epsilon<2/\lVert\cdot\rVert_2$ holds without empirical policing.

**Heads sum, not average.** Heads are a *basis of additive coupling operators* (buoyancy $+$ pressure $+$ viscous), so their sum is the net force; averaging would systematically shrink it as operators are added — the opposite of language-model heads, which are interchangeable subspaces. A *constant* $1/\sqrt{n_H}$ scale is the correct fix for $\tanh$ saturation at initialization, because being constant it preserves the $g_\theta=1$ telescoping exactly.

**The connection to §3.** Conservative (skew) $+$ dissipative ($-\gamma_\ell I$) $+$ external ports (gated non-conservative edges) **is** the port-Hamiltonian decomposition of an open physical system. Atlas reaches the same structure by declaring ports between frozen experts; Noether 1.1 reaches it inside one network's layers. The passivity theorem of §4.6 is the statement that the first construction inherits what the second builds in.

## 13.2 Discovered conservation

**Intuitively.** Do not tell the model what is conserved — that is domain knowledge that is wrong as often as it is right (energy is not conserved in a driven convection cell; divergence is not zero in a compressible gas). Let it watch the data, keep a shortlist of candidates, and enforce each **as hard as the evidence warrants**.

**Discovery.** For candidate invariant functionals $\{C_k\}$, measure drift along training trajectories

$$D_k=\mathbb E_{\text{traj},t}\bigl[(C_k(\text{state}_{t+1})-C_k(\text{state}_t))^2\bigr],$$

and search for new parameterized $C_k$ minimizing $D_k$ subject to non-triviality ($\mathrm{Var}(C_k)>0$) and mutual orthogonality. **For a linear readout $C_k(h)=w_k^{\!\top}h_{\text{top}}$ this objective *is* a generalized eigenproblem** — minimize $w^{\!\top}\Sigma_\Delta w$ subject to $w^{\!\top}\Sigma w=1$, with $\Sigma=\operatorname{Cov}(h_{\text{top}})$ and $\Sigma_\Delta=\operatorname{Cov}(\Delta h_{\text{top}})$ — whose conserved directions are the *smallest* generalized eigenvectors of $(\Sigma_\Delta,\Sigma)$. This is Slow Feature Analysis.

**Seeds are written against a field's descriptor, never a named field**: "mass-like" $=\sum_f\mathbb 1[\mathrm{rank}(\theta_f)=0]\int f$, "momentum-like" $=\sum_f\mathbb 1[\mathrm{rank}(\theta_f)=1]\int f$, and so on.

**Enforcement — a hybrid keyed to measured drift.** Always-on learned-soft: a penalty $\lambda_kD_k$ with $\lambda_k$ uncertainty-weighted. Conditional threshold-gated-hard: the projection

$$x^{\ast}=\hat x-M^{-1}A^{\!\top}\bigl(AM^{-1}A^{\!\top}\bigr)^{-1}\bigl(A\hat x-b\bigr)$$

applied **only** for $C_k$ whose *held-out* drift $D_k<\varepsilon$ and whose projection is cheap. **A mis-discovered hard constraint injects systematic bias, so hard enforcement is gated on evidence, not on a learned scalar the model could game.**

**Cost hierarchy.** Global scalar integrals (mass, energy, momentum) are cheap — one projection or a soft loss. Local pointwise differential constraints ($\nabla\cdot u=0$) need an expensive elliptic/Leray solve and default to **learned-soft only**. Inequalities (positivity, entropy increase) need different machinery and are deferred.

**The direct contrast with Atlas.** Atlas *imposes* conservation at declared interfaces because the graph and the edge types are declared inputs — there is nothing left to discover. Noether 1.1 *discovers* it because agents, edges and regimes are themselves unknown. Reusing the discovery mechanism inside Atlas would import cost and a failure mode (a discovery mechanism can misidentify what is conserved) for a problem the declared graph has already solved. **A more general mechanism is here the wrong tool, not merely an unnecessary one.**

## 13.3 Post-decoder diffusion

**The problem, stated as a fact about the loss.** A deterministic next-state predictor trained on a chaotic system with MSE is *rewarded for blurring*: when many fine-scale futures are equally likely, their average has the lowest error, so the model outputs the mean and the sharp features die.

Let $\bar v_{t+1}$ be the deterministic decode and $r=v_{t+1}-\bar v_{t+1}$ the residual, dominant in the near-cutoff band. Train a conditional diffusion model $p_\vartheta(r\mid\bar v_{t+1},\{h_i\},z_{\text{cond}})$ by denoising:

$$\mathcal L_{\text{diff}}=\mathbb E_{r,\tau,\epsilon}\bigl\lVert\epsilon-\epsilon_\vartheta(r_\tau,\ \tau,\ \bar v_{t+1},\ \{h_i\},\ z_{\text{cond}})\bigr\rVert^2,\qquad r_\tau=\sqrt{\bar\alpha_\tau}\,r+\sqrt{1-\bar\alpha_\tau}\,\epsilon .$$

**$\tau$ here is a diffusion noise-level index, unrelated to the physical clock $t$ — and, in this compendium's notation, unrelated to the agent-infidelity defect of §4.2.** $\bar\alpha_\tau$ is a fixed (not learned) schedule decreasing from near $1$ (clean) to near $0$ (pure noise).

**Sampling**, run only at inference:

$$r_{T_{\text{diff}}}\sim\mathcal N(0,I);\qquad \hat\epsilon=\epsilon_\vartheta(r_\tau,\tau,\cdot),\qquad \hat r_0=\frac{r_\tau-\sqrt{1-\bar\alpha_\tau}\,\hat\epsilon}{\sqrt{\bar\alpha_\tau}},$$

then $r_{\tau-1}=(\text{schedule-weighted blend of }\hat r_0\text{ and }r_\tau)+\sigma_\tau z$, terminating at $r_0$ and returning $v_{t+1}=\bar v_{t+1}+r_0$.

**Four design commitments, each with a reason.** *Residual band only* — $r$ occupies a narrow band with small dynamic range, so both $\epsilon_\vartheta$ and the reverse-step count stay small. *Runs after conservation* — the generated residual is re-projected onto discovered constraints, which is safe **because** the projection is the minimal-disturbance correction under its metric. *Phase-separated training* — the deterministic stack trains first and is frozen, or the diffusion head chases a residual defined against a nonstationary target. *Optional at inference* — low-temperature scaling of $\sigma_\tau z$ trades diversity for closeness; setting $r=0$ gives a stable reference rollout, at the cost of the attractor statistics full sampling restores.

---

# 14. The governing-equation families the experts carry

Experts are cut along **governing-equation boundaries**, never along edge-type labels ([[expert-library-atlas-0.1]]): types that come from the same governing system share an expert; types from different systems get different experts exchanging the coupling term explicitly. Sources: [[impl-atlas-0.1-phase0-scope-and-data]] §2, [[spec-wind-farm-wake-atlas-0.1]] §4, [[case-study-thermal-strain-atlas-0.1]].

## 14.1 Two-dimensional incompressible Navier–Stokes

The family of every real case study built so far.

$$\boxed{\;\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u=-\nabla p+\frac{1}{\mathrm{Re}_{\text{eff}}}\nabla^2\mathbf u+\mathbf f,\qquad \nabla\cdot\mathbf u=0\;}$$

In vorticity form, which is how the fluid expert internally represents its state,

$$\partial_t\omega+(\mathbf u\cdot\nabla)\omega=\frac{1}{\mathrm{Re}_{\text{eff}}}\nabla^2\omega+\nabla\times\mathbf f,\qquad \omega=\partial_xu_y-\partial_yu_x,$$

and with a stream function $\mathbf u=\nabla^{\perp}\psi=(\partial_y\psi,-\partial_x\psi)$, so $\nabla^2\psi=-\omega$. **The stream-function representation is not incidental** — it makes mass conservation across every interface exact and free (§3.5's second row).

**The nondimensional regime law**, which is why a frozen expert's window size is not a free parameter. With $x=\ell_wx_p$, $u=U_su_p$, $t=T_st_p$ and $T_s=\ell_w/U_s$:

$$\boxed{\;\mathrm{Re}_{\text{eff}}=\frac{T_s}{\nu_p\,\ell_w^{2}}\;}$$

**Doubling the physical size one window covers quarters $\mathrm{Re}_{\text{eff}}$** — which is why coarsening changes the *physics* rather than the resolution, and is the blocker `R8` discharges by probing rather than coarsening.

## 14.2 Two-dimensional compressible Navier–Stokes, conservative form

$$U=\begin{bmatrix}\rho\\\rho u\\\rho v\\E\end{bmatrix},\quad F(U)=\begin{bmatrix}\rho u\\\rho u^2+p\\\rho uv\\(E+p)u\end{bmatrix},\quad G(U)=\begin{bmatrix}\rho v\\\rho uv\\\rho v^2+p\\(E+p)v\end{bmatrix}$$

$$\partial_tU+\partial_zF(U)+\partial_yG(U)=\partial_zF_{\text{visc}}+\partial_yG_{\text{visc}}+S$$

with ideal-gas closure $E=\dfrac{p}{\gamma_{\text{gas}}-1}+\tfrac12\rho(u^2+v^2)$, $p=\rho R T$, $c=\sqrt{\gamma_{\text{gas}}p/\rho}$; Newtonian viscous stresses under Stokes' hypothesis $\lambda_{\text{visc}}=-\tfrac23\mu_{\text{visc}}$; and heat flux $q_i=-k_s\partial_iT$ with $k_s=\mu_{\text{visc}}c_p/\mathrm{Pr}$, $\mathrm{Pr}=0.72$.

**Reaction source** — a single progress variable rather than finite-rate chemistry:

$$\partial_t(\rho Y)+\nabla\!\cdot\!(\rho\mathbf uY)=\rho\,\omega_r(Y,T),\qquad \omega_r=A\,(1-Y)\exp\!\Bigl(-\frac{T_{\text{act}}}{T}\Bigr),\qquad S_E=\rho\,\omega_r\,q_{\text{rxn}} .$$

**Numerics and their assumptions**: MUSCL reconstruction with a minmod limiter (second order in smooth regions, TVD across shocks — required, because the plume genuinely contains shocks); HLLC for the inviscid flux with Rusanov as a robust fallback; central differences for viscous fluxes; SSP-RK2; $\Delta t\le\mathrm{CFL}\min\bigl(\min(\Delta z,\Delta y)/(\lvert\mathbf u\rvert+c)\bigr)$ at $\mathrm{CFL}=0.4$.

## 14.3 Backward-Euler conduction

$$\rho_sc_{p,s}\,\partial_tT=\nabla\!\cdot\!\bigl(k_s\nabla T\bigr)$$

with Robin (convective, radiatively linearized) conditions on both faces:

$$-k_s\partial_nT=h_{\text{in}}\bigl(T-T_{\text{gas,in}}\bigr),\qquad -k_s\partial_nT=h_{\text{out}}\bigl(T-T_{\text{gas,out}}\bigr)+\epsilon_{\text{em}}\sigma_{\text{SB}}\bigl(T^4-T_\infty^4\bigr).$$

**Discretization and why**: Q1 bilinear finite elements; **backward Euler**, chosen for unconditional stability because the shell's thermal timescale is slow and large steps are wanted. Analytic oracle: the 1D semi-infinite slab, $T(x,t)=T_s+(T_0-T_s)\,\mathrm{erf}\bigl(x/(2\sqrt{\alpha_{\text{th}}t})\bigr)$, matched to $<1\%$.

## 14.4 Quasi-static plane-stress elasticity

$$\nabla\!\cdot\!\boldsymbol\sigma+\mathbf b=0,\qquad \boldsymbol\varepsilon=\tfrac12\bigl(\nabla\mathbf u+\nabla\mathbf u^{\!\top}\bigr)$$

$$\boldsymbol\sigma=\frac{E_s}{1-\nu^2}\Bigl[(1-\nu)\boldsymbol\varepsilon+\nu\,\mathrm{tr}(\boldsymbol\varepsilon)\mathbf I\Bigr]-\frac{E_s\alpha_s\,\Delta T}{1-\nu}\mathbf I$$

**The quasi-static hypothesis** — inertia negligible against the ascent timescale, so elasticity is a **solve each step, not a march**. Analytic oracles: a uniformly heated *free* plate must produce zero stress; a fully constrained one must produce $\boldsymbol\sigma=-E_s\alpha_s\Delta T/(1-\nu)\,\mathbf I$.

## 14.5 Thermal strain / eigenstrain — a bond that is not a port

The discrete coupling term, lifted out of the element quadrature so it can be *named*:

$$\boxed{\;\boldsymbol f_{\text{th}}=\int_\Omega\mathsf B^{\!\top}\mathsf D\,\boldsymbol\varepsilon_0\,\mathrm dV=G\,(\boldsymbol T-T_{\text{ref}}),\qquad \boldsymbol\varepsilon_0=\alpha_s\,\Delta T\,(1,1,0)^{\!\top}\;}$$

with $\mathsf B$ the strain–displacement matrix, $\mathsf D$ the plane-stress elasticity matrix, and $G$ the assembled thermal-load operator. **Asserted against the expert rather than assumed**: on its own free-body solve, $\lVert K\boldsymbol u-(I-VV^{\!\top})G\Delta T\rVert/\lVert G\Delta T\rVert=5.3\times10^{-15}$ to $1.3\times10^{-14}$, with $V$ spanning the rigid modes.

**The reverse half is the same matrix transposed**, $G^{\!\top}$ — the Biot / Gough–Joule term $-T_0\beta_{\text{th}}\,\mathrm{tr}\,\dot{\boldsymbol\varepsilon}$ in the energy equation. That reciprocity is what makes the pair a **bond**; it is exhibited, not asserted, by checking $\boldsymbol u\cdot G\Delta T=\Delta T\cdot G^{\!\top}\boldsymbol u$. The two-way fixed point contracts in 7 iterations at coupling number $\delta_c=T_0\beta_{\text{th}}^2/(\rho c_pD_{\text{plane}})=9.075\times10^{-3}$.

**Why it is not a port** (§3.6): the conjugate pair $\bigl(\Delta T,\ \beta_{\text{th}}\,\mathrm{tr}\,\dot{\boldsymbol\varepsilon}\bigr)$ with $\beta_{\text{th}}=E_s\alpha_s/(1-\nu)$ has a product in $\mathrm{W\,m^{-3}}$, but **two readings of the same coupling differ by $3395\times$** and their sum is an exact total derivative of an energy neither agent owns:

$$P_\Omega^{(A)}+P_\Omega^{(B)}=\frac{\mathrm d}{\mathrm dt}\Bigl(\boldsymbol u^{\!\top}G\,\Delta T-\tfrac12\Delta T^{\!\top}H_{\text{th}}\,\Delta T\Bigr).$$

The free energy $\tfrac12\boldsymbol u^{\!\top}K\boldsymbol u-\boldsymbol u^{\!\top}G\Delta T+\tfrac12\Delta T^{\!\top}H_{\text{th}}\Delta T$ has a **bilinear cross term belonging to neither agent**, so $\Psi\neq\Psi_{\text{cond}}+\Psi_{\text{elas}}$ and §3.4's premise fails before any of its arithmetic runs. For a free body the ratio is exact: $E_\times=-2E_{\text{strain}}$ identically.

**The surface reduction, and when it is exact.** The exact split

$$\underbrace{\int_\Omega\beta_{\text{th}}\Delta T\,\nabla N_k\,\mathrm dV}_{G}=\underbrace{\oint_{\partial\Omega}\beta_{\text{th}}\Delta T\,N_k\,\mathbf n\,\mathrm ds}_{G_{\text{surf}}}-\underbrace{\Bigl(-\int_\Omega N_k\,\beta_{\text{th}}\nabla(\Delta T)\,\mathrm dV\Bigr)}_{G_{\text{body}}}$$

holds to $10^{-15}$. $G_{\text{surf}}$ is what a surface `MECH` bond can deliver; $G_{\text{body}}$ — $9.6\%$ of the load — is what it cannot, and dropping it costs $1279\times$ in stress because the two halves nearly cancel. **The classical equivalent-thermal-pressure reduction is exact exactly when $\nabla\Delta T=0$** (measured $6.8\times10^{-17}$ of the load on the uniform-$\Delta T$ control).

**Carried in full, the volumetric term reproduces the monolith's stress bit for bit; lagged one macro-step it costs $7.116\times10^{-3}$ relative, first order in $\Delta t$** ($\log_2$ ratios $1.003,1.002,1.001$). That splitting-error bound is the framework's instrument for a co-located coupling, in place of a transmission condition.

## 14.6 Actuator-disk momentum extraction

Classical freestream-referenced momentum theory:

$$U_d=U_\infty(1-a),\quad U_w=U_\infty(1-2a),\quad C_T=4a(1-a),\quad C_P=4a(1-a)^2,\quad T=\tfrac12\rho AU_\infty^2C_T .$$

**The freestream form is unusable for a downstream turbine**, and this matters more than it looks: $U_\infty$ is not defined inside a wake, and using the domain inlet value would make turbine 2's thrust independent of the wake it sits in — **silently destroying the coupling the case study exists to test**. The local form:

$$\boxed{\;C_T'=\frac{C_T}{(1-a)^2}=\frac{4a}{1-a},\qquad T=\tfrac12\rho AC_T'\,\langle U_d\rangle^2,\qquad P=T\,\langle U_d\rangle\;}$$

with $\langle U_d\rangle$ the disk-averaged streamwise velocity **measured by the fluid expert on the rotor's upstream face**. At $a=1/3$, $C_T'=2$ and $C_P^{\max}=16/27$ (Betz).

**The body force handed back to the fluid:**

$$\mathbf f_{\text{disk}}(\mathbf x)=-\frac{T}{A\,\Delta_d}\,\hat{\mathbf x}\,\mathbb 1[\mathbf x\in R_i]=-\frac{C_T'\langle U_d\rangle^2}{2\Delta_d}\,\hat{\mathbf x}\,\mathbb 1[\mathbf x\in R_i].$$

**The circular dependence and its resolution.** $T$ depends on $\langle U_d\rangle$, which depends on $\mathbf f_{\text{disk}}$, which depends on $T$. Resolved by fixed-point iteration under relaxation:

$$T^{(k+1)}=(1-\theta)\,T^{(k)}+\theta\cdot\tfrac12\rho AC_T'\bigl\langle U_d\bigr\rangle^2\bigl[\mathbf f(T^{(k)})\bigr],\qquad\theta=0.5 ,$$

measured converging in $3.02$ iterations on average. **In the scheme of §11 this is not a distinct method — it is one point in $\Sigma$ with a mis-specified $\varepsilon_{\text{tol}}$**, because the stopping test is a scalar on one rotor and says nothing about whether the interfaces agree.

**Validity limit, declared rather than modelled**: momentum theory breaks down for $a\gtrsim0.4$ (the turbulent-wake state, needing an empirical correction). The spec stays inside $a\le0.35$ and treats $a>0.4$ as out of scope. This is the `validity` field of §11.2 in its clearest form.

**Two structural findings about this expert as a *port*.** A field-to-lumped `MECH` seam is **one-sided by $\mathrm{Re}_h$** — the field side's effort carries viscosity and a disk's traction does not — so its substitution certificate is blind by construction. And the disk has **no expressible bond at all** by two independent routes: `R2b` refuses the non-overlapping cut a surface bond needs, and the overlapping alternative is two-way volumetric, which §3.6 has just refused as a port.

---

# 15. The seven-hypothesis envelope

The precise class of systems on which §4.7's bound is a theorem. Stated on [[theory-closure-audit]] §3, made a per-run stamp by [[end-to-end-architecture-spec]].

| # | hypothesis | removed by | status on the flagship case study |
|---|---|---|---|
| $\mathrm{E1}$ | the interaction graph is **fixed** for the whole rollout | topology mutation | **holds** |
| $\mathrm{E2}$ | interfaces are **static** and geometrically coincident | moving $\Gamma$; non-conforming transfer (§7) | **holds** |
| $\mathrm{E3}$ | a **single global evolution operator** exists whose restrictions the agents approximate — so that $\tau$ is *defined* | heterogeneous physics at a multiphysics seam | **holds** (one Navier–Stokes) |
| $\mathrm{E4}$ | a **single macro-step clock**, or a multirate scheme whose *time-integrated* flux matching is proved conservative | multirate (`R9`) | **holds** |
| $\mathrm{E5}$ | the claim is a **trajectory** claim inside $T_{\text{pred}}$ | chaos | **unverified** |
| $\mathrm{E6}$ | the assembly $\mathcal A$ is bounded with $\lVert\mathcal A\rVert$ known, and the blend is at least as accurate as the local solves it blends | assembly consistency | **unverified** |
| $\mathrm{E7}$ | the interconnection is **power-preserving** and every reduction/prolongation pair is **adjoint** — required for the $L\le1$ branch | adjointness; storage declaration | **unverified** |

**$\mathrm{E1}$–$\mathrm{E4}$ are decidable at compile time before any compute is spent.** The three unverified ones are exactly the three that need a measurement. **The flagship case study satisfies four of seven and has never checked the other three**, so it is not known to sit inside the envelope of its own error bound.

**$\mathrm{E3}$ is the one worth dwelling on**, because it is the hypothesis a genuinely multiphysics graph breaks: $\tau$ is *defined* as the agent's error against the restriction of a single global operator, and at a seam between two different governing families there is no such operator to restrict. That is a hole in the definition, not in a measurement, and it has a named slot rather than a fix.

---

# 16. What is measured, and what is not

The bound has four constants and the framework's instrumentation history is lopsided.

| constant | controlled by | how measured | status |
|---|---|---|---|
| $L$ | incremental passivity (§4.6); windowing caps the *method's* share at $L^W$ | fit from one paired perturbed rollout; or $\lambda_{\min}$ of the probed symmetric part | first fitted 2026-08-27: $0.948$–$0.980$, **below one**, and matching the monolith's to five decimals for a correctly built decomposition |
| $\tau$ | a faithful local solve — a real boundary channel, non-periodic windows | against a validated classical reference on the true restriction | measured and dominant; but **misattributed once** (a decomposed elliptic solve wearing a $\tau$ label — §12's attribution theorem observed) |
| $\sigma$ | the transmission rung, amplified by $1/\beta$ | assemble $\tilde\Lambda$, compare to a reference assembly, report $\beta$ | measured on the overlapping branch via $\Pi$; **maximal in the known failure** |
| $\gamma$ | the accelerator (§4.5) | the interface residual at convergence | **the only one routinely reported, and negligible by eleven orders** |

> **The framework measured the smallest term, controlled the second-largest, and for a long time reported neither of the other two.** That single sentence is the practical content of the bound.

**The durable methodological lesson, from the run that first measured all four**: every diagnostic the construction proposed for itself came back green while it was solving the wrong equation, and the missing test was free — *check that the reference's own answer reduces your residual.* The control that would have redirected the whole search — four windows each covering the whole domain, which must return exactly $0.0$ — was the one not run.

---

# 17. Mathematical inconsistencies found

The lint value of the exercise. Everything below was found by reading two or more pages against each other; each item names the pages and says what a reader should do. Items are ordered by how much they change a conclusion.

## 17.1 Two incompatible $\sigma$ bounds share one constant

[[master-error-bound]] §4 gives $\sigma\le(C_\mu/\beta)\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$ and §4.1 gives $\sigma\le C_\mu\Pi\lVert\delta\lambda\rVert$. Both define $C_\mu$ identically as $\operatorname{Lip}_\mu(\Phi_\mu)$, yet the same measurement implies $C_\mu=2.2\times10^{-8}$ on the first form and $C_\mu\in[0.228,1.178]$ on the second — **eight orders apart for one defined quantity.** The page diagnoses the factorization rather than the constant, correctly, but leaves both branches writing $C_\mu$ unqualified. **A reader must not quote "$C_\mu=1.2$, measured" into §4's form.** Write $C_\mu^{\text{sub}}$ and $C_\mu^{\text{halo}}$.

## 17.2 The boxed master bound is not the bound that was evaluated

[[master-error-bound]] §7 boxes the master bound with $(C_\mu/\beta)\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$ substituted for $\sigma^n$ — the substructuring form. Every evaluation quoted immediately below the box is an **overlapping** `WindowNS` tiling, for which the same page's §4 says that form overestimates by $4.6\times10^{7}$. The *generic* bound (§4.7 here) is fine; the *specialized box* as printed is inconsistent with §4.1 of its own page. **The box should carry $\sigma^n$ unexpanded, with the two branch estimates below it.**

## 17.3 $L$ is reported with two different values for the same run, on one page

[[master-error-bound]] §7's table gives the as-built scheme $L=0.972776$, $\tau=2.935\times10^{-4}$, $\sigma=2.253\times10^{-5}$, asymptotic bound $1.16\times10^{-2}$ against a plateau of $2.39\times10^{-3}$ — **$5\times$ loose**. [[tier0-measurements]] §4.2 evaluates what it calls the same bound with $L=0.948441$, $\tau=3.7014\times10^{-4}$, $\sigma=2.6995\times10^{-5}$, asymptotic $7.70\times10^{-3}$ — **$14\times$ loose**. And [[master-error-bound]] §9's own box quotes $L=0.948441$ for "the composed map", i.e. **both values appear on one page without reconciliation.** Either two different runs are being described under one label, or one table is stale. **Neither looseness figure should be quoted without saying which run it is from.**

## 17.4 One page defines the probe's output flux twice, incompatibly

[[probed-dtn-coupling]] §2.2 defines the probe response as $\partial_n(\mathcal E_i[\psi_k]-\mathcal E_i[0])$; §4.1 proposes instead the conservative momentum flux — the `MECH` effort — and argues the refinement matters because one-sided differencing is noisy. Measured: on the **assembled** seam the two agree to $1.11\times10^{-9}$, so §4.1's refinement is **a no-op exactly where it claims to matter**; per *block* they differ by $10^3$ in $\beta$ and flip the passivity verdict ($\pi=0$ against $\pi=2.39$). The passivity equivalence of §6.4(b) is true only for §2.2's convention. **The page pins the convention in a later box while §4.1's body text still argues the other way**, and the "noisy" objection did not materialize (the finite difference is stable to seven digits across four decades of $\epsilon_{\text{p}}$).

## 17.5 The $\sigma$ bound presumes a boundary-value problem, and says so on a different page

[[master-error-bound]] §4's derivation needs $\lambda^\star$ to satisfy the exact interface equation $\Lambda\lambda^\star=\chi$; the first-order perturbation estimate has no reference point otherwise. [[probed-dtn-coupling]] §2.3 measured that **the true trace does not satisfy the flux-balance condition at all** for a one-step map of an initial-boundary-value problem — substituting the reference's trace leaves the residual unchanged, and the overshoot grows as $\Delta t$ is refined. **That is a hypothesis of the $\sigma$ bound and it is recorded only as a refutation of the construction**, three pages away, never added to the bound's hypothesis list. §4.3 above carries it.

## 17.6 On the overlapping branch, $\gamma$'s defining trace does not exist

The three-way split defines $\gamma=[\Phi_{\lambda^{(k)}}-\Phi_{\lambda^\dagger}](u^{n,\star})$, and [[master-error-bound]] §4.1 states that an overlapping scheme "poses no interface equation, so there is no $\lambda^\dagger$ in §1's sense." Yet §7's table reports $\gamma=0$ for two overlapping schemes. The split survives only if $\lambda^\dagger$ is **redefined** as the lagged datum — which §4.1 implies and never writes. Under that reading $\lambda^{(k)}=\lambda^\dagger$ identically for any one-pass overlapping scheme, so **the reported $\gamma=0$ is a tautology rather than a measurement.** The conclusion ("the framework reports the term that does not matter") is if anything strengthened by this, but the number should not be quoted as evidence.

## 17.7 A page argues `max` is the wrong aggregator, then boxes a bound that uses it

[[composition-error-theory]] §2.3 argues at length that $\max_i$ is the wrong aggregator on two counts and that the honest one is an adjoint-weighted sum, giving $10^{-2}$, $8\times10^{-2}$ and $0.5$ for identical $\varepsilon$. §3.1 then boxes $\lVert u-u^\star\rVert\le C_S(\max_i\tau_i+\gamma)$. The two are reconcilable — the max is a valid upper bound with the transport absorbed into $C_S$ — but **the page never says so**, and a reader quoting the box loses §2.3's entire point. **Read $C_S$ in that box as carrying the chain length.**

## 17.8 $C_S$ and $L$ are used interchangeably and are not the same object

[[composition-error-theory]] §3.1 defines $C_S=\lVert(\mathrm I-D\Phi)^{-1}\rVert$; [[temporal-error-accumulation]] and [[master-error-bound]] use $L=\operatorname{Lip}(\Phi)$ with the geometric sum. These coincide only for $L<1$ (where $C_S\le1/(1-L)$); at $L=1$ the resolvent is unbounded while the geometric form gives the perfectly finite $N\delta$, and for $L>1$ the resolvent form is not the relevant object at all. **Hypothesis H2 is stated as "$C_S=O(1)$" and is then graded against measurements of $L$** — a different constant. The grading is morally right and formally loose.

## 17.9 $n_0(\Gamma)=1$ contradicted $n_0(\Gamma)=0$ within one subsection

[[probed-dtn-coupling]] §2.1 point 1 prescribes keeping the elliptic solve **outside** $\Lambda_i$; point 2 of the same subsection asserts incompressibility gives $\Lambda_i$ a rank-one null space. **These cannot both hold** — if incompressibility is excluded from $\Lambda_i$, the constraint that produces the null space is excluded with it. Measured $\dim\ker=0$ on four seams in every control. Resolved, correctly, by making $n_0$ conditional on `elliptic_subsolve` (§7.5) — but the resolution introduces a **third declared input to a quantity that two other pages describe as a property of the seam's agents and vertex valence alone**. The free correctness check caught a real error on its first use; it caught one in the theory rather than in the probe.

## 17.10 $\mathcal R(t)$'s additivity premise is unstated on the page that defines it

[[port-algebra-atlas-0.1]] §6 writes $\mathcal R(t)$ with $\sum_i\mathrm dE_i/\mathrm dt$ as its first term and does not state that this presumes $\Psi=\sum_i\Psi_i(u_i)$. The premise **fails** for a co-located pair, measured, with a bilinear cross term $2680\times$ the elastic energy. The gap was closed by adding field 7 (`support`) to the amendment procedure — but the additivity premise still does not appear in §6 itself, only in §10.3's later commentary. **Anyone reading §6 alone gets a residual whose premise is invisible.**

## 17.11 $\mathcal R(t)$ has no stated rule for the `ADVEC` multiport

§6's residual integrates $(e\,f)_\Gamma$ per port connection. `ADVEC` is explicitly **not** a simple effort–flow pair — it is a multibond with a carrier and a variable passenger list, whose conjugate effort is $h_0$ for the energy channel and $\mu_k^{\text{chem}}$ for species $k$. **No page says how it enters $\mathcal R(t)$**: one term for the carrier, or one per passenger. For an isothermal incompressible case study this never bit, because `ADVEC` degenerates to volumetric flux; it will bite on the first reacting or multi-species graph. Unstated assumption, not yet an error.

## 17.12 `R1`/`R2` name two different things in one folder

[[schwarz-iteration-atlas-0.1]] §2.1 and [[impl-wind-farm-guide]] §6.2 use `R1` and `R2` for the two **interface-data variants** (halo tokens vs. declared flux). [[general-coupling-scheme]] §3 uses `R1` and `R2` for the first two **compiler admissibility rules**, and `R2` there is described as "the single most consequential rule on the page." Both usages are live, in one folder, in pages that cite each other. §2.1 of this compendium adopts `I-R1`/`I-R2` for the variants; the vault should pick one and rename.

## 17.13 `R9`'s status was promoted and then retracted

[[general-coupling-scheme]] `R9` is a **multirate** rule. [[tier0-measurements]] promoted it to a single-rate requirement on the strength of the steady-condition diagnosis, then the 2026-08-28 measurement showed the defect is **spatial** ($-\nu h\,\partial_{nn}w$), a time-integrated fix is *worse* than pointwise, and "W7's promotion is retracted and `R9` reverts to being a multirate rule." **[[interface-transfer-theory]] §5.2 states `R9` as a general rule with a refusal, without the retraction.** Read `R9` as multirate-only; the single-rate case is answered by §4.4's $\Pi$, not by time integration.

## 17.14 "Multiplicative ordering breaks symmetry averaging's exactness" mislocates the guarantee

[[schwarz-iteration-atlas-0.1]] §4 and rule `R5` say a sequential sweep breaks the exactness of the Reynolds average. **The theorem of §9.1 holds for any operator $\mathcal E$ and any ordering** — it is a property of the average, not of the sweep. What a multiplicative sweep breaks is the *hypothesis* that mirror-paired tiles are handed mirror-image inputs. The conclusion (`R5`: additive only) is right; the reason as stated attacks the theorem instead of its input. **`R5` is better justified as "an ordering is admissible iff it preserves the power-preserving interconnection"**, which [[composition-error-theory]] §5 already supplies.

## 17.15 $\Xi$ is proposed as a per-expert scalar and is a per-state one

[[probed-dtn-coupling]] §4.5 defines $\Xi_i$ and proposes it as a selection axis; [[plug-in-composition-theorems]] promotes it to the **routing criterion** for a library. [[case-study-reuse-probe-atlas-0.1]] then measured same-regime replicates disagreeing by up to $49\%$ and records that its reproducibility had never been quoted. **A library ranked by $\Xi$ is ranked by a number with a $49\%$ error bar**, and neither of the two pages that propose the ranking says so.

## 17.16 The worst-agent conjecture "as a theorem" is stated without its constructive caveat

[[master-error-bound]] §8's specialization table lists $\tilde\Lambda=\Lambda\Rightarrow\sigma=0$ as recovering the worst-agent conjecture "and here it is a theorem." True — and $\tilde\Lambda=\Lambda$ requires the exact Steklov–Poincaré operator, which [[composition-error-theory]] §4.0 states means *solving the problem you decomposed*. The row is a theorem and not an achievable configuration, and the table does not say so. §4.7 above carries the caveat.

## 17.17 Two stale cross-references

- [[port-algebra-atlas-0.1]] §5's YAML declaration example still shows no prolongation, while §10.1 field 2 records that the mapping class is "derived from the declared prolongation, never declared." The example predates [[interface-transfer-theory]] and should gain a $P_i$ line.
- [[gap-worklist]]'s own lint records a **dangling `[[phase1-resume-prompt]]` referenced from three pages as if it exists.** Still dangling at the time of writing.

## 17.18 A symbol audit finding, stated once because it is systemic

The two most-used symbols of the composition error theory, $\tau$ and $\sigma$, are the two most-used symbols of continuum mechanics — and in [[port-algebra-atlas-0.1]] §3.3 they appear **in the same equation with the opposite emphasis**: $\boldsymbol\sigma=-p\mathbf I+\boldsymbol\tau$, where $\boldsymbol\sigma$ is the total stress and $\boldsymbol\tau$ the deviatoric part, against $\sigma=$ transmission infidelity and $\tau=$ agent infidelity three pages away. Bold-versus-italic is the only thing separating them, and several pages set stress unbolded in prose. **This is the single highest-frequency collision in the vault** and §2.1 is the convention proposed for it: mechanics quantities always bold, error terms always italic and scalar.

---

# 18. Reading order

For someone coming to the theory cold, in the order that minimizes back-references:

1. **[[port-algebra-atlas-0.1]]** — what an interface *is*. §3 here.
2. **[[master-error-bound]]** — the one inequality everything else is about. §4 here.
3. **[[composition-error-theory]]** — why the natural conjecture fails, and the three hypotheses. §5 here.
4. **[[probed-dtn-coupling]]** — the construction that makes $\sigma$ measurable. §6 here.
5. **[[temporal-error-accumulation]]** — what happens over a rollout, and the window rule. §10 here.
6. **[[general-coupling-scheme]]** — all of it as a compiled declaration. §11 here.
7. **[[theory-closure-audit]]** — the envelope, and what is still open. §15 here.

[[schwarz-iteration-atlas-0.1]], [[symmetry-averaging-atlas-0.1]] and [[interface-transfer-theory]] are best read as the three worked constructions behind steps 4–6 — respectively the negative result that motivated the probe, the positive result that established the "build it outside the expert" pattern, and the transfer rules the pattern needs to survive two experts that were never built to meet.

---

## See Also

- [[port-algebra-atlas-0.1]] — the five ports, the $O(K)$ argument, $\mathcal R(t)$, and the amendment procedure (§3)
- [[master-error-bound]] — the three-way split and the master bound (§4)
- [[composition-error-theory]] — the worst-agent conjecture graded, and the four constructions (§5)
- [[probed-dtn-coupling]] — the probe, $\beta$, $\kappa$, $\pi$, $\Xi$, and the interface equation's scope limit (§6)
- [[interface-transfer-theory]] — the common interface space, $R_i=P_i^{\ast}$, and the null-space check (§7)
- [[schwarz-iteration-atlas-0.1]] — the faithfulness hypothesis and the periodic-window identity (§8)
- [[symmetry-averaging-atlas-0.1]] — the Reynolds average and its equivariance proof (§9)
- [[temporal-error-accumulation]] — the three regimes, windowing, and the enforcement ladder (§10)
- [[general-coupling-scheme]] — the seven-parameter scheme and the nine rules (§11)
- [[plug-in-composition-theorems]] — closure, attribution, substitution (§12)
- [[conservation-as-constraint-atlas-0.1]] — enforce, measure, or decline (§3.5)
- [[backbone-1.1]], [[discovered-conservation-1.1]], [[post-decoder-diffusion-1.1]] — the Noether 1.1 core (§13)
- [[expert-library-atlas-0.1]] — why experts are cut by governing family (§14)
- [[theory-closure-audit]] — the seven-hypothesis envelope, and the closed / importable / open cut (§15)
- [[end-to-end-architecture-spec]] — the nine layers and the three verdicts the scheme compiles into
- [[tier0-measurements]] — where most of §16's numbers come from
- [[gap-worklist]] — what to do about §17
- [[atlas-and-standard-dd-theory]] — where all of this sits in classical domain-decomposition theory
- [[physics-foundation-models]], [[pfm-architecture-approaches]] — the goal this formalism serves
