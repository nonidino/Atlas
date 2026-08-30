# Probed DtN Coupling — building the agreement operator outside the expert

**Type:** Concept page — framework theory, **version-independent** (folder: `Atlas 0.1/common/`; no Atlas revision suffix, for the same reason as [[composition-error-theory]] — nothing here depends on the case study or the revision)
**Status:** written 2026-08-26, as the constructive continuation of [[composition-error-theory]] §4.0. That page established *what* to fix; this one is *how*, for the one class of agent Atlas is actually built around — a black box that cannot be told anything. **Measured 2026-08-27 ([[tier0-measurements]]), and the result splits the page in two: the probe is sound and four of its claims about $\tilde\Lambda$ hold as written, while §2.1's premises and §2.3's interface condition do not.** Four inline boxes below mark what changed; the unmarked text stands. **Extended 2026-08-28 (§9 there): §4.5's positive control is run and returns exactly zero, and §2.3's proposed remedy was built and does not work — the mechanism is an identity, and it is spatial rather than temporal.**
**Related:** [[tier0-measurements]] · [[composition-error-theory]] · [[schwarz-iteration-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[conservation-as-constraint-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[open-problems-atlas-0.1]] · [[symmetry-averaging-atlas-0.1]] · [[prior-art-and-novelty-atlas-0.1]] · [[results-n-sweep-wind-farm]] · [[f1-pathmap-and-end-goal]]

> **The one-line version.** [[composition-error-theory]] says composition accuracy is set by how well the interface operator approximates Steklov–Poincaré, and prescribes climbing a ladder — Robin, Ventcell — that **this vault has twice recorded frozen experts cannot climb**. The way out is that you never needed the expert's cooperation: **Dirichlet-to-Neumann is by definition Dirichlet-in, Neumann-out**, so an expert that merely *accepts* a boundary ring already exposes everything needed to build its DtN operator **from outside**, by probing. Do that and the agreement operator becomes an assembled matrix — which is simultaneously the coarse space [[schwarz-iteration-atlas-0.1]] §S3 could not construct, the passivity certificate [[composition-error-theory]] §4.1 wants, the inf-sup constant its §4.2 wants, and the optimal Robin coefficient its §4.3 wants. **Four constructions, one measurement.**

---

# 1. The contradiction this page exists to resolve

[[composition-error-theory]] §4.0 is right and its prescription is unusable as written. Set the two next to each other:

| It says | The vault has measured |
|---|---|
| Climb from Dirichlet toward Robin and Ventcell; accuracy is set by $\lVert\Lambda_i-\tilde\Lambda_i\rVert$ | A Robin condition needs the expert to accept a *flux-plus-value* combination on its boundary. A field-in/field-out operator **has nowhere to put a boundary ring at all** — recorded as the reason a frozen expert was *structurally ineligible* for the fix that moved $P_2/P_1$ from $1.02$ to $0.77$ |
| Improve the operator being agreed on | [[schwarz-iteration-atlas-0.1]] §3.1: under periodic windows the sweep is a **constant map in the iteration index**, two passes agreeing *bitwise* — so there is no operator to improve |
| A coarse space removes the $H^{-2}$ that makes one-level Schwarz fail | §S3: coarsening means enlarging $L$, and $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p L^2)$ means that **changes the physics rather than the resolution**. Marked *do not build without settling* |

So the framework's own diagnosis prescribes three fixes, and the agents it is built around are ineligible for all three. That is not a gap in the theory; it is a gap in how the theory was applied. **Every one of those fixes was aimed at the inside of the expert.**

## 1.1 The observation that dissolves it

The Dirichlet-to-Neumann map is *defined* by what it takes and what it returns:

$$\Lambda_i:\; \underbrace{g}_{\text{trace on }\partial\Omega_i}\;\longmapsto\;\underbrace{\partial_n \mathcal S_i[g]\big|_{\partial\Omega_i}}_{\text{flux the subdomain returns}}$$

An expert that accepts a Dirichlet ring and returns a field **already implements $\Lambda_i$** — you have simply never asked it for the answer in that form. You cannot *feed* it a Robin condition, but you do not need to: given $\Lambda_i$, every rung of the ladder above Dirichlet is a linear-algebraic rearrangement of the same matrix, performed **in the composition layer**, on data the expert has already returned.

> **The agreement operator does not have to live inside the agent. It has to live between them.**

That is not a new pattern in this vault — it is precisely [[symmetry-averaging-atlas-0.1]]'s shape, where the composition layer supplied mirror equivariance that OP-6 proved the checkpoint does not have. This page applies the same move to the interface condition.

## 1.2 And it is immediately runnable

`reference.WindowNS` already exists, already accepts a one-cell Dirichlet ring, and was validated before wiring: uniform flow preserved **exactly**, Taylor–Green second-order, projection divergence $7.6\times10^{-16}$, a Gaussian wake diffusing within **$0.44\%$ of analytic**. That is a boundary channel that has been shown not to lie. Nothing below needs training, a new expert, or a change to the adopted R2 interface contract.

---

# 2. The construction

## 2.1 Substructuring, stated for this setting

Take the **non-overlapping** view: agent domains $\{\Omega_i\}_{i=1}^N$ meeting on an interface $\Gamma=\bigcup_{i\neq j}\partial\Omega_i\cap\partial\Omega_j$. Let $\lambda = u|_\Gamma$ be the unknown interface trace. Each agent solves a Dirichlet problem on its own domain with trace $\lambda$ and returns its flux. The composed solution is the $\lambda$ at which **the fluxes balance**:

$$\boxed{\;\mathcal G(\lambda)\;:=\;\sum_i \Lambda_i\lambda \;-\; \chi \;=\;0\;}\qquad \chi = -\sum_i\partial_n \mathcal E_i[0]\big|_\Gamma$$

For a linear problem $\mathcal E_i$ is affine in $\lambda$, so $\Lambda_i$ is a genuine linear operator and $\mathcal G$ is the **Steklov–Poincaré (Schur complement) equation**. Solving it reproduces the undivided solution *exactly* — this is the case [[composition-error-theory]] §4.0 identifies as the one where the worst-agent bound is a theorem rather than a conjecture.

**Two structural facts make this the right formulation for Atlas specifically, and they are not generic.**

1. **The elliptic part is already global and stays where it is.** [[schwarz-iteration-atlas-0.1]] §5.1's table records that pressure is solved by a **whole-domain DCT Poisson solve**, at $0.08$ s against $2.7$ s for a 124-window sweep. So the operator being substructured is the *advection–diffusion* part — exactly the part with finite propagation speed, hence exactly the part where a transmission condition is the binding constraint. The construction does not fight the existing architecture; it slots into the gap the architecture left.
> **Measured 2026-08-27, and both numbered facts below failed.** [[tier0-measurements]] ran this construction against `reference.WindowNS` for the first time. **(1) is violated by the very build this page names as runnable**: `WindowNS` projects **per window**, not globally, so the elliptic part does *not* stay where it is — and the price is a per-window divergence of $2.76\times10^{-4}$ against a monolith's $8\times10^{-15}$, which turned out to be the dominant defect of the whole decomposition (§4.1 of that page). **(2) is not observed**: the measured null dimension is $\mathbf{0}$ on four seams, in every control, with no spectral gap at the bottom. **And the two failures are one fact seen twice** — an operator that *would* be singular in the constant mode is instead made nonsingular by the solver silently absorbing the mass imbalance the constant mode carries. **(1) and (2) cannot both hold in any case**: if pressure is excluded from $\Lambda_i$, the constraint that produces the null space is excluded with it.
>
> **Resolved the same day, and the page keeps both — conditionally.** $n_0(\Gamma)$ is **not a property of the seam alone; it depends on where the elliptic solve lives.** With incompressibility inside $\Lambda_i$ the trace is constrained to zero net flux and $n_0=1$; with the elliptic part in the composition layer — which is what (1) prescribes — $\Lambda_i$ is the pure advection–diffusion DtN and $n_0=0$. The field that decides it is `capability.elliptic_subsolve`, added by this measurement. Measured $0$ on four seams under both agents, $\kappa$ between $1.10$ and $1.36$, no spectral gap.
>
> **And (1) is now enforced rather than assumed.** **R10** refuses a decomposition of an agent that embeds a global elliptic solve, because decomposing it changes the operator rather than restricting it. Taking the projection out of `WindowNS` and running it in the composition layer — which is (1), implemented — improved $\beta$ by $69\times$, $\kappa$ by $18\times$ and the operator's asymmetry by $76\times$, to $0.002$: **with the elliptic part where this page says it belongs, the interface operator is nearly self-adjoint.** The pressure coupling was what made it non-normal.

2. **Incompressibility leaves a rank-one null space — at a fluid–fluid seam.** $\lambda$ must satisfy $\oint_\Gamma\lambda\cdot n = 0$, so $S=\sum_i\Lambda_i$ is singular in one direction. Solve in the constrained subspace (or by pseudoinverse); this is standard and it is also a **free correctness check** — a probed $S$ whose measured null space is not the dimension *that seam* predicts is reporting a defect in the probe, not in the physics. **The predicted dimension is a property of the assembled seam, not of a port type**: at a field↔lumped seam it is $0$, because the lumped side responds in the direction incompressibility leaves open. [[interface-transfer-theory]] §7 gives the count per seam and the verdict for each direction of a mismatch.

## 2.2 The probe

Choose an interface basis $\{\psi_k\}_{k=1}^{m}$ on each seam. Then, holding everything else fixed:

$$S_i\,\psi_k \;=\; \partial_n\Bigl(\mathcal E_i[\psi_k]\;-\;\mathcal E_i[0]\Bigr)\Big|_\Gamma ,\qquad k=1,\dots,m$$

**$m+1$ Dirichlet solves and the block is assembled.** The $\mathcal E_i[0]$ term is the zero-probe; subtracting it is what removes every $\lambda$-independent bias in the agent — which matters more than it looks, and §5.1 is why.

**The basis dimension is set by a measured property of the expert, not chosen.** A tile seam is $2D$ long at $dx=0.0156\,D$, i.e. $128$ cells, $256$ raw DOF for two velocity components. But OP-4 records $81\%$ of the field's dissipation living below $0.125\,D$, and OP-6 records the checkpoint's artefact comb at $32/16/8$ cells. Interface content below $\sim8$ cells is therefore inside the range where the expert is not representing flow. Truncating there gives

$$m\;\approx\;\frac{2D}{0.125\,D}\;=\;16 \ \text{ modes per component},\qquad m_\Gamma^{\text{seam}}\approx 32 .$$

**[AI Inference]:** using the expert's own spectral cutoff to set the multiplier dimension is the natural resolution of the open question [[composition-error-theory]] §4.2 raises — *what is the multiplier space for a frozen operator, which has no finite-element trial space and no approximation theory?* The answer proposed here is that it does have an effective resolution, it has been measured twice for other reasons, and that number is the right one. Untested, and cheap to falsify: enrich $m$ past $16$ and the added rows of $S_i$ should be indistinguishable from probe noise.

## 2.3 What you do with the assembled matrix

> **This subsection is the one the 2026-08-27 measurement refutes, and it is worth stating before it is read rather than after.** Solving $S\lambda=\chi$ exactly made one composed macro-step **$8.9\times$ worse** than not solving it at all. The reason is not the operator — $\tilde\Lambda$ assembles cleanly, $\kappa\approx20$, $\beta$ and $\mu$ are real numbers — but the **equation**: the true trace does not satisfy $\sum_i\Lambda_i\lambda=\chi$. Substituting the monolith's own post-step seam values leaves the residual essentially unchanged ($2.6421\times10^{-5}\to2.6520\times10^{-5}$, i.e. marginally *worse*), so the condition is not an approximation of a true statement about $\lambda$ but a different statement, and driving it to zero moves $\lambda$ **$41\times$ past** the truth. **Refining $\Delta t$ makes the overshoot grow** — $41\times$, $75\times$, $150\times$ at $\Delta t=0.05,\ 0.01,\ 0.002$ — which names the mechanism: a condition whose solution is $\Delta t$-independent is a **steady-state** condition.
>
> $\mathcal E_i$ is written throughout this section as the solution operator of a *boundary-value problem*. **Every expert this vault can actually probe is the one-step map of an initial–boundary-value problem**, and the page never states the interface condition for that case. Flux balance is what the trace satisfies at a fixed point, not after one explicit macro-step, where the interface's own unsteady term is a first-order part of its equation and $\Lambda_i$ contains none of it. The correct condition is a statement about the **time-integrated** flux across the macro-step — [[general-coupling-scheme]]'s **R9**, so far written only for multirate — with $\lambda$ carried as a function of time over the window, which is waveform relaxation ([[gap-worklist]] W17), until now filed as a deferred refinement.
>
> **The amplification is this page's own arithmetic and that is the part worth keeping.** $\delta=-S^{-1}r$ multiplies the residual by $1/\beta\approx190$, so a residual that is a modelling artifact rather than a physical imbalance arrives as a large trace error. §4.3's $\beta$ is doing exactly what [[master-error-bound]] §4 says it does; what neither page says is that **the numerator has to be a real imbalance for the bound to be about anything**. Full record: [[tier0-measurements]] §5.

> **2026-08-28 — the remedy the box above names was built, and it is not the remedy. The mechanism is sharper than "a steady condition", and it is an identity.**
>
> The time-integrated condition was implemented as a $W=2$ waveform, $\lambda(t)=(1-s)\lambda^n+s\lambda^{n+1}$, with the sub-step-averaged flux read from the same run as the pointwise one so no convention could drift between them. **It fails both tests the pointwise condition fails, by the same margins.** Reference-trace ratios: $1.002$ (exposed agent) and $1.028$ (embedded) against pointwise's $1.002$ and $1.040$, at $\Delta t=0.05$, $0.01$ and $0.002$. Solving it degrades the composed macro-step by $2.00\times$ against pointwise's $1.59\times$ — it is *worse*, not neutral.
>
> **Why.** At a shared-layer seam both sides pin the *same* cell, so with outward normals
>
> $$F_A+F_B \;=\; \nu\,\frac{2w_\Gamma-w_{A,\text{in}}-w_{B,\text{in}}}{h} \;=\; -\,\nu h\,\partial_{nn}w$$
>
> — verified **bit-exactly** against the monolith's own solution on all four seams. **The flux-balance residual at a shared-layer seam is a discrete second derivative, not a jump.** It is $O(h)$ on the exact solution and vanishes only where that solution is *linear across the seam*, so driving it to zero asks the solution to be straight at every cut. The defect is **spatial**, which is why it is $\Delta t$-independent — and why no amount of integrating in time can remove it. The steady-condition reading predicts the $\Delta t$-scaling; this one predicts that *and* the failed reference-trace check *and* the failure of the time-integrated fix.
>
> **And §5's $8.9\times$ is largely an R10 effect.** With the elliptic part in the composition layer the same wrong trace is nearly harmless ($0.988$, a $1.2\%$ improvement) while still overshooting the true trace by $66\times$: the composed step is simply insensitive to the seam trace there. The $8.9\times$ was the per-window pressure solve amplifying it.
>
> **What replaces it.** Not a corrected flux condition — **an overlapping decomposition has no common interface at which a flux jump is defined**, each subdomain's artificial boundary being somewhere else. That is the measured reason behind the compiler's existing rule that probed-DtN forces the non-overlapping axis. The construction that *is* valid for an explicit one-step map is the **overlapping halo update itself**, and it now has its own error bound: [[master-error-bound]] §4.1's $\Pi$. [[gap-worklist]] W46 closes on that; W7's promotion is retracted and R9 reverts to being a multirate rule.

Solve $S\lambda=\chi$ **directly**. $S$ is dense of size $m_\Gamma^{\text{global}}$ — at $N=2$'s 15 declared interfaces and $m_\Gamma^{\text{seam}}=32$, that is a $480\times480$ dense solve, which is **microseconds** and not a cost line at all.

This is the step that changes the character of the method: it replaces *iterating an exchange until the pieces agree* with *solving once for the trace at which they must agree*.

---

# 3. This is the coarse space, and it kills §S3's arithmetic

The strongest argument for the construction is that it removes a disqualification [[schwarz-iteration-atlas-0.1]] states in full and has no answer to.

That page's §3.2 records the classical two-sided estimate

$$\kappa\bigl(M_{\text{AS},1}^{-1}A\bigr)\le C\,H^{-2}\bigl(1+H/\delta\bigr),\qquad \kappa\bigl(M_{\text{AS},2}^{-1}A\bigr)\le C\bigl(1+H/\delta\bigr)$$

and the transparent mechanism behind the $H^{-2}$: **one-level additive Schwarz advances information exactly one overlap-connected neighbour per iteration.** Applied to this build — $2D$ windows, $0.5D$ overlap, $1.5D$ stride, $7D$ turbine spacing, $24D$ domain:

$$k_{\min}=\Bigl\lceil\tfrac{7D}{1.5D}\Bigr\rceil=5\ \text{iterations before turbine 1 can reach turbine 2},\qquad \Bigl\lceil\tfrac{24}{1.5}\Bigr\rceil=16\ \text{to cross the domain}$$

against a loop that runs three. The page calls this "disqualifying for the metric in dispute," and it is right.

**A Schur complement has no such limit, and the reason is structural rather than a matter of degree.** $S=\sum_i\Lambda_i$ is **dense across the whole interface**: every seam DOF is coupled to every other, because $\Lambda_i$ is the nonlocal operator that encodes what all of $\Omega_i$ does in response to boundary data. Solving $S\lambda=\chi$ transmits information from the inlet to the outlet **in one solve**, not in $16$ sweeps, and it does so at any $N$.

$$\text{one-level Schwarz: } H^{-2},\ \text{one subdomain per iteration}\quad\longrightarrow\quad\text{substructuring: global coupling, by construction}$$

**And the coarse operator is built entirely from fine-scale expert calls at the expert's native $L$.** §S3's blocker was that a coarse solve requires enlarging $L$, which changes $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p L^2)$ and therefore changes the physics rather than the resolution. Probing never changes $L$. Every call is a native-resolution call on a native-sized window; the global coupling comes from *assembling* those calls, not from coarsening them.

> **This is the two-level structure the vault noticed it already had — 8 declared agents over 124 fine tiles — and could not use. The missing coarse solve is the probed Schur complement, and it is available to exactly the experts a coarse solve was not.**

**Under-claiming note, and it is required.** The $\kappa$ estimates above are for *linear, SPD, variationally-posed* problems with *exact* local solves. [[prior-art-and-novelty-atlas-0.1]] §2 already names that a frozen neural surrogate is none of the three. The claim here is narrower than the classical theorem and does not depend on it: it is that a **dense assembled interface operator couples the whole graph in one solve whereas a local exchange does not**, which is a statement about the sparsity pattern and holds regardless of whether $S$ is SPD. What the classical theory would additionally buy — a conditioning bound — is *not* claimed and is instead **measured**, per §4.3.

---

# 4. What falls out of the same matrix

This is the part that makes the construction worth its cost. Each subsection below is a separate item on [[composition-error-theory]] §4's list, obtained from one assembly.

## 4.1 The port algebra is already a DtN interface — it has just never been used as an operator

[[port-algebra-atlas-0.1]] defines every port as an **effort–flow pair whose product is power**: `MECH` is $(\boldsymbol\sigma\!\cdot\!\mathbf n,\ \mathbf v)$, `THERM` is $(T,\ q_n/T)$, and so on. Read the `MECH` pair in the order the probe uses it:

$$\underbrace{\mathbf v}_{\text{flow — the Dirichlet trace you impose}}\;\longmapsto\;\underbrace{\boldsymbol\sigma\!\cdot\!\mathbf n}_{\text{effort — the flux the agent returns}}$$

**That map is $\Lambda_i$.** The port algebra already specifies both sides of the DtN interface, in the right units, with the right sign convention, for every port type in the closed vocabulary. What it has never done is treat the pair as an **operator to be identified** rather than a pair of numbers to be matched.

This also settles a detail that would otherwise be the construction's weakest point. Reading $\partial_n u$ off a one-cell ring by one-sided differencing is noisy. Reading the **conservative momentum flux through the seam** is not, and it is what the port already declares. So the output side of the probe is not a new quantity requiring a new convention — it is the existing `MECH` effort, already implemented, already sign-checked in §3.4's thermal-convention discipline.

> **Measured 2026-08-27: this paragraph proposes a *second* flux, §2.2 having proposed a first, and the page never compares them. They were compared.** On the **assembled** seam operator they agree to $1.11\times10^{-9}$ relative — the advective term $-(\mathbf u\!\cdot\!\mathbf n)w$ is evaluated at the shared ring cell where the two sides carry the same value and opposite normals, so it cancels identically. **§4.1's refinement is a no-op exactly where this paragraph claims it matters.** Per *block* it is a factor of $10^3$: at one seam the upstream window reads $\beta=3.42\times10^{-3},\ \kappa=24,\ \pi=0$ under §2.2's flux and $\beta=1.51,\ \kappa=1.6,\ \pi=\mathbf{2.39}$ under §4.1's. **The "noisy" objection also did not materialize** — the finite difference is stable to seven digits across four decades of $\epsilon$, because a deterministic expert's probe floor is machine epsilon, not OP-6's $10^{-6}$. See [[tier0-measurements]] §2.2.

## 4.2 Passivity is positive-realness of the probed matrix — §4.1 and §4.0 are one object

This is the strongest theoretical consequence on the page.

[[composition-error-theory]] §4.1 wants a **storage function** $H_i$ and a **dissipation inequality** per expert, and calls that the highest-value single change to the framework. It also — correctly, and this is the subtle part — insists that plain passivity bounds only the *state*, while **incremental** passivity is what bounds the *error*.

Incremental passivity is a statement about how the agent responds to *differences* in boundary data. The probed operator is exactly that response. With the port pairing $\langle e,f\rangle$ being power by construction:

$$\text{agent }i\text{ incrementally passive at its ports}\quad\Longleftrightarrow\quad \Lambda_i \ \text{positive real}\quad\Longleftrightarrow\quad \tfrac12\bigl(S_i+S_i^{\!\top}\bigr)\succeq 0$$

So the passivity certificate is **an eigenvalue computation on a matrix you already assembled**, not a statistic gathered over rollouts. Three consequences, in increasing order of usefulness:

- **The passivity defect becomes a spectrum, not a scalar — and it takes a noise floor.** Write $\mu_i=\lambda_{\min}\bigl(\tfrac12(S_i+S_i^{\!\top})\bigr)$. The defect is $\mu_i$ clipped against a **scale-relative** floor,

$$\pi_i \;=\; \begin{cases} \lvert\mu_i\rvert, & \mu_i < -\varepsilon_i \\[2pt] 0, & \text{otherwise} \end{cases} \qquad\qquad \varepsilon_i \;=\; c\,\varepsilon_{\text{mach}}\,\lVert S_i\rVert ,\quad c=\dim S_i$$

  which recovers §4.1's additive defect. **The floor is not fastidiousness.** Taken as $\lvert\min(0,\mu_i)\rvert$ the defect reports **arithmetic noise as physics**: a matrix positive semidefinite by construction returns $\mu_i\approx-10^{-15}$ from `eigh`, and that is enough to make $\pi_i>0$, decertify the $L\le1$ branch, and stamp E7 `fails` on a clean graph — measured at $1.3\times10^{-15}$ on the first configuration that had no other problem. A diagnostic that fires on every symmetric operator it is ever shown stops being read. $\pi_i$ is discontinuous at the floor by construction, which is the right trade when the discontinuity sits at machine noise, and **$\mu_i$ is emitted raw alongside $\pi_i$ so the clipping is visible rather than silent.** The *eigenvector* is untouched by any of this and names **which interface mode** is being amplified; nothing currently in the vault localizes an instability to a mode.
- **It needs no viscosity, and now for a second reason.** §4.1 already observed that a storage function sidesteps OP-4's missing single $\nu$. Positive-realness sidesteps it more completely: it is a property of the input–output map, and never mentions the interior at all.
- **It composes.** Atlas's connection rule $e_A=e_B,\ f_A=-f_B$ is a power-preserving (Dirac) interconnection, so positive-real blocks compose to a positive-real $S$. **That is $C_S=O(1)$ obtained structurally**, which is [[composition-error-theory]]'s hypothesis **H2** — the one that is "half-cleared: bounded in $N$, never established in $t$."

> **Measured 2026-08-27, and the equivalence needs one more hypothesis than it states.** All four assembled seams came back **passive** — $\mu\in[+2.05,+6.39]\times10^{-3}$, so $\pi=0$ and the floor never fired. But the **per-block** verdict flips with the flux convention this page leaves open (§4.1 above), and it flips all the way: the upstream window is passive under §2.2's flux and $\pi=2.39$ under §4.1's, purely because of a *transport* term that carries no dissipation and cancels one level up. **The block-level equivalence is only about dissipation once the convention is pinned, and this page pins it in two incompatible places.** Two consequences. First, the convention that makes the port pairing a genuine power — §2.2's — is the one the equivalence is true for. Second, **the verdict belongs to the seam, not the block**: a per-block gate would have refused a graph whose assembled operator is provably passive, and W33's "refuse at admission on a contradicted declaration" is exactly such a gate. §4.3's "both are printable per port" needs to say *which one the verdict is taken on*. [[tier0-measurements]] §2.2–2.3.
>
> **A second block-level trap, from the same data.** The **downstream** side of a seam has an almost rank-deficient DtN — $\beta=2.67\times10^{-5}$, $\kappa=1529$ — while its own assembled seam is $\kappa=23$. A uniform-flow control reproduces it within $8\%$ on a field with no structure at all, so it is a property of *being downstream*, not of the wake: an outflow window's response to inflow data is nearly a pure translation. **Two orders of magnitude between a block's conditioning and its seam's**, and only the seam's means anything.

**[AI Inference]:** the sharpest testable form is that OP-2's drift should be visible as $\pi>0$ — a negative mode **below the floor** — **in a single probe, without a rollout at all.** OP-2's signature is a *monotone accumulation* rather than a wrong steady state, which is what a mildly non-passive composed map looks like. If a probe at $t=5$ already shows such a mode and OP-2's drift direction projects onto its eigenvector, that is a diagnosis obtained in minutes for a problem that has resisted rollout-based analysis. If $\mu\ge-\varepsilon$, passivity is eliminated as OP-2's cause — equally valuable and equally cheap. **The floor matters to this test specifically**: without it the answer is *"a negative mode"* every time and the experiment cannot distinguish its two outcomes.

> **Run 2026-08-27, and it came out the second way — which this paragraph correctly priced as equally valuable.** $\mu>0$ on every assembled seam, so **passivity is eliminated as a cause of OP-2's drift in this configuration**, in minutes, with no rollout. A paired rollout run alongside agrees independently: the fitted $L=0.948\pm0.005$ is **below** one, where OP-2's recorded drift implies $L=1.0143$. The structural route and the empirical route reach the same place, which is the strongest thing that can be said for either. **The caveat is that OP-2 was measured on the periodic frozen-checkpoint configuration and this is a Dirichlet-ring exact solver at a different $\Delta t$** — three variables differ, so this eliminates passivity *here* and does not close OP-2. See [[tier0-measurements]] §3.1.

## 4.3 $\beta$, conditioning, and the falsifier for over-constrained matching

[[composition-error-theory]] §4.2 identifies the inf-sup constant $\beta$ as *the hypothesis the worst-agent conjecture is missing*, and leaves it as an open design question. On an assembled $S$ it is a singular value:

$$\beta \;=\; \sigma_{\min}\bigl(S\big|_{\text{constrained subspace}}\bigr),\qquad \kappa(S)=\sigma_{\max}/\sigma_{\min}$$

Both are printable per port. This does two things at once:

- It generalizes **OP-1's expensively-learned conditioning lesson** — *a projection whose sensitivity is not $O(1)$ is not a projection* — from the thrust constraint to every interface, which §4.2 called for and had no mechanism to deliver. OP-1's pathology was $dg/d\lambda=10^{-6}$ against a residual of $10^{-1}$; that is a $\sigma_{\min}$ diagnosis, and on the interface it would now be caught before being trusted rather than after.
- It gives [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule the vocabulary it lacks for saying **a port is badly conditioned**. A port with $\beta$ at the probe floor is one where pointwise matching is meaningless: the agents' responses do not distinguish the modes being matched, so agreement there is unconstrained rather than achieved. That is the over-constrained-rung symptom §4.2 predicts, now with a number attached.

## 4.4 The optimal Robin coefficient is read off, not derived — and for a neural expert it can only be read off

Optimized Schwarz chooses $\alpha$ in $\partial_n u_i+\alpha u_i = \partial_n u_j + \alpha u_j$ to minimize the convergence factor, and the optimum is **the symbol of the exact DtN**. Classically you obtain it by Fourier-analyzing the PDE.

**You cannot do that for a frozen checkpoint, because you do not know what PDE it solves.** OP-4 is the same obstruction in a different costume: the operator has no documented $\nu$, no documented symbol, and 81% of its dissipation below the scale where a single $\nu$ exists. Every analytic route to $\alpha^\star$ is closed.

Probing opens the only remaining one. In a Fourier interface basis the diagonal of $S_i$ **is** the measured symbol, mode by mode:

$$\alpha^\star_k \;=\; \bigl(S_i\bigr)_{kk}\quad\text{(measured)}\qquad\text{rather than}\qquad \alpha^\star=\text{the symbol of an operator you do not have}$$

So §4.3's construction — *the principled form of the Orlanski outflow OP-2 already names as its remedy* — becomes available to neural experts specifically, and the sweep over $\alpha$ that §4.3 proposes as measurement #4 is replaced by reading one diagonal.

**[AI Inference]:** this suggests a cheap intermediate deployment that does not require the full Schur solve. Probe once, extract the per-mode diagonal, install it as a **mode-dependent Robin condition** in the existing sweep, and keep everything else. That is a one-rung climb of §4.0's ladder at the cost of a single assembly, and it is the natural fallback if §6's budget for the full method does not close.

## 4.5 A composability index — and the frozen checkpoint scores exactly zero

[[schwarz-iteration-atlas-0.1]] §8 ends on a wish: that boundary-condition flexibility, not accuracy, should be the primary expert-selection criterion, and [[composition-error-theory]] §4.3 sharpens it to *how far up the DtN ladder an expert's interface can go*. Both are yes/no. The probe makes it a norm:

$$\Xi_i \;:=\; \frac{\bigl\lVert \Lambda_i^{\text{expert}}\bigr\rVert}{\bigl\lVert \Lambda_i^{\text{ref}}\bigr\rVert}\qquad\text{— how much of the true boundary response the agent actually reproduces}$$

**And it recovers the vault's sharpest existing finding as an exact value.** A periodic window has no boundary channel, so $\partial\mathcal E_i/\partial\lambda = 0$ identically — which is precisely the bitwise result [[schwarz-iteration-atlas-0.1]] §3.1 measured, that two sweeps agree under `np.array_equal`. In this language:

$$\text{frozen periodic expert}\;\Longrightarrow\;\Lambda_i\equiv0\;\Longrightarrow\;S\equiv0\;\Longrightarrow\;\mathcal G(\lambda)=-\chi\ \text{for every }\lambda$$

**The interface problem is not badly conditioned; it is empty.** Every trace is equally consistent, because no agent's output depends on any agent's input. That is a strictly stronger statement than "the sweep is the identity" — it says the composed system has no interface problem *to* solve, and it explains without further argument why iterating it converges instantly and means nothing.

$\Xi=0$ for the frozen checkpoint; $\Xi$ is measurable and positive for `WindowNS`. **A composability index that returns exactly zero on the case the vault already proved uncomposable is the strongest available evidence that it is measuring the right thing.**

> **Run 2026-08-28, and it returns exactly zero — [[tier0-measurements]] §9.5.** The expert is `reference.SpectralNS`, the periodic vorticity–streamfunction solver the frozen-checkpoint harness itself uses, whose `step` takes no boundary argument at all. The agent declares `bc_channel = NONE` and is otherwise identical to the Dirichlet one: same ports, same declared 16-mode prolongation, same flux, same ring indices, same macro-step. **The whole graph goes through `compile_scheme`**, not a hand-rolled probe.
>
> | | measured |
> |---|---|
> | $\lVert\Lambda\rVert_2$ | $0.000000\times10^{0}$ |
> | $\max_{ij}\lvert S_{ij}\rvert$ | $0$ — **every entry bitwise zero** |
> | per block | $0$ / $0$ |
> | $\beta$ | $0$ |
> | null dim (declared $0$) | $16$, the whole space |
> | solver calls | $34$ — the probe genuinely ran |
> | $\Xi$ | $\mathbf 0$ |
>
> **And the compiler does the right thing with it.** `L5/R6` refuses the direct and Krylov solvers on an empty operator, and the run carries **`refused claims: ["the word 'coupled' on any output involving seam sx0"]`** — spec §6.4(a), reached from a real periodic solver rather than a fixture. §9's row 3 is discharged as predicted, and the $\Xi=0.19$–$0.53$ measured for the two `WindowNS` variants is therefore not an artifact of a probe that manufactures small operators out of roundoff or an off-by-one.

## 4.6 $\tau$ against $\varepsilon$, in operator norm, without a rollout

[[composition-error-theory]]'s falsifiable prediction #3 — *is the consistency defect $\tau_i$ actually much larger than benchmark accuracy $\varepsilon_i$?* — is its load-bearing claim, and the proposed test is a rollout against a classical reference. Probing gives a cleaner instrument. Assemble $\Lambda_i^{\text{expert}}$ and $\Lambda_i^{\text{ref}}$ on the same basis and compare directly:

$$\bigl\lVert \Lambda_i^{\text{expert}} - \Lambda_i^{\text{ref}}\bigr\rVert$$

This is a **boundary-data-aware** comparison by construction — it measures exactly the response to imposed neighbour data that $\varepsilon_i$ never probes and $\tau_i$ is defined by — and it involves no rollout, so it cannot be contaminated by the accumulated-error confound. It also does not repeat OP-5's methodological failure, where a subtraction bucket turned out to contain a Reynolds-number mismatch: two matrices assembled on the same basis at the same state differ in one thing.

---

# 5. Differentiating a black box, and what actually sets the probe floor

The construction stands or falls on getting an accurate directional derivative out of an operator you cannot open. There are three routes and they are not equivalent.

## 5.1 The floor is OP-6's $10^{-6}$, not W0's $3\times10^{-2}$ — and the difference is the whole feasibility argument

The obvious objection is that the checkpoint's per-call noise floor is $0.031\,U_\infty$ (W0: peak emitted from an *exactly uniform* field), and a finite difference divides by $\epsilon$, so probing should be hopeless.

**That objection is wrong, and it is worth being precise about why.** W0's figure is a *systematic, state-dependent model error*, not call-to-call randomness — the same input returns the same output bitwise. A finite difference **subtracts it off**:

$$\frac{\mathcal E[\lambda+\epsilon\psi]-\mathcal E[\lambda]}{\epsilon}\qquad\text{— any }\lambda\text{-independent bias cancels in the numerator}$$

What does *not* cancel is genuine non-reproducibility, and the vault has measured exactly one source of it: **OP-6's finding that the expert's forward pass depends on a window's position in the batch at the $10^{-6}$ level** — the float32 residual that sets W7's floor. That is the number that divides by $\epsilon$:

$$\text{probe error}\;\sim\;\frac{10^{-6}}{\epsilon}\qquad\Longrightarrow\qquad \epsilon=10^{-2}U_\infty\ \text{gives}\ \sim10^{-4},\ \text{four orders below the signal}$$

**And it is removable rather than merely tolerable.** OP-6's dependence is on *batch position*; pin the batch layout across a probe pair and the two calls differ only in the ring. So a finding previously logged as an irritating floor on a symmetry gate turns out to be **the quantity that sets the probe step, and the one lever that controls it.**

## 5.2 The right route is a JVP, and it is the second unspent use of a headline claim

[[prior-art-and-novelty-atlas-0.1]] claims end-to-end differentiability through the coupling as a distinguishing property against preCICE. [[composition-error-theory]] §4.4 already observed that this claim is spent on nothing, because end-to-end differentiability **is** the adjoint and the adjoint is never taken.

The probe is the second unspent use, and it is cheaper than the first. $D\mathcal E_i[\lambda]\cdot\psi_k$ is a **forward-mode JVP**: exact to floating point, no $\epsilon$, no OP-6 contamination, one pass per basis vector. If the expert is differentiable — and the framework's own novelty claim asserts it is — this is strictly the right route and §5.1's analysis is a fallback rather than the plan.

## 5.3 The fallback for a genuinely opaque expert

If an expert is neither differentiable nor smooth at the probe scale, **fit** rather than difference: take $m'>m$ random probes and solve a regularized least-squares problem for $S_i$. Reproducibility error then averages down as $\eta/\sqrt{m'}$ instead of amplifying as $\eta/\epsilon$, at the cost of more calls and a Tikhonov parameter that has to be justified.

**[AI Inference]:** this three-way split is itself a useful addition to [[expert-library-atlas-0.1]]'s selection axes, and it is finer than the ladder rung of §4.5. An expert is *probe-cheap* (differentiable → JVP), *probe-affordable* (deterministic and smooth → finite differences with $\epsilon$ set by its reproducibility floor), or *probe-expensive* (opaque → regression). That distinction is invisible to any accuracy benchmark and it directly prices composability.

---

# 6. Cost, stated at its worst first

The honest headline is that **naive assembly is unaffordable and the method depends on amortization.** Working from timings already in the vault — $0.022$ s per window at batch 32, $124$ windows, and a Dirichlet sweep at $\mathbf{14\times}$ the periodic one, the whole factor being the Galilean frame:

| step | arithmetic | cost |
|---|---|---|
| one global Dirichlet sweep = one matvec $Sv$ | $124\times0.022\times14$ | $\approx38$ s |
| full assembly, agents probed concurrently | $(\max_i m_i+1)\approx129$ sweeps | $\approx\mathbf{82}$ **min per macro-step** |
| against the current macro-step | $\sim24$ s | $\approx200\times$ — **unaffordable as stated** |

Three reductions, each independently justified, and they multiply:

1. **Equivalence classes.** An aligned row with uniform inflow and identical turbines has interior agents that are translates of one another, and the mirror symmetry $|G|=2$ is *already exploited* by [[symmetry-averaging-atlas-0.1]]. Probe one representative per class — plausibly inlet / interior / outlet — and each sweep activates $\sim3$ agents rather than 8. **$\approx30$ min.**
2. **Refresh, do not rebuild.** $S$ depends on state, but [[results-n-sweep-wind-farm]] is direct evidence that this coupling is *insensitive*: tripling the graph moved the answer under $1\%$ with the fixed point converging in 3 iterations throughout. Re-probe every $K$ macro-steps and use the standing $S^{-1}$ in between. At $K=50$: **$\approx36$ s per step amortized.**
3. **Matrix-free where the full matrix is not needed.** For the *solve* alone, GMRES on $\mathcal G$ needs $r$ matvecs and not $m_\Gamma$ of them. The full assembly is needed only for §4's diagnostics — which are the point, so this trades against them rather than replacing them.

**Budget with (1) and (2): roughly one extra Dirichlet sweep per macro-step plus $\sim36$ s amortized re-probing — call it $2$–$3\times$ the current step.** That is expensive and it is not disqualifying, and it buys a global coupling that no number of cheap sweeps can buy at all.

**These are estimates assembled from recorded timings, not measurements**, and the two soft spots are named: the $14\times$ Dirichlet factor was measured on `WindowNS` and may differ per expert, and the refresh interval $K$ is an assumption about how fast $S$ drifts that §9's measurement 2 exists to test.

---

# 7. Nonlinearity — what this actually is, said plainly

For Navier–Stokes $\mathcal E_i$ is **not** affine in $\lambda$, so $\Lambda_i$ is not an operator and §2.1's clean statement does not hold as written. What a probe recovers at a state $\lambda^{(k)}$ is the **tangent** map $D\Lambda_i(\lambda^{(k)})$, and the method is therefore:

$$\text{solve }\ \mathcal G(\lambda)=0\ \text{ by Newton:}\qquad D\mathcal G\bigl(\lambda^{(k)}\bigr)\,\delta\lambda \;=\;-\,\mathcal G\bigl(\lambda^{(k)}\bigr),\qquad \lambda^{(k+1)}=\lambda^{(k)}+\delta\lambda$$

which is **Newton–Krylov–Schur** — nonlinear substructuring, the family FETI-DP and BDDC sit in. Three consequences that must not be glossed:

- **Every §4 quantity is local to a state.** $\beta$, $\kappa$, $\Xi$ and the passivity spectrum are properties of the linearization at $\lambda^{(k)}$, not global certificates. They should always be reported with the state they were measured at. **[AI Inference]:** the useful form is likely a *trace* of $\lambda_{\min}\bigl(\tfrac12(S+S^\top)\bigr)$ over a rollout rather than a single number — and a trace that crosses zero would be a considerably more informative object than OP-2's drift curve.
- **Convergence is local.** Newton needs a good enough $\lambda^{(0)}$, and the natural one is the previous macro-step's converged trace, which for a quasi-steady wake should be excellent. Damping or a line search on $\lVert\mathcal G\rVert$ is the standard remedy and is cheap relative to a sweep.
- **The additive/multiplicative distinction survives, and the answer is unchanged.** [[schwarz-iteration-atlas-0.1]] §4 requires *additive* ordering on two grounds — order-independence, and preserving [[symmetry-averaging-atlas-0.1]]'s exactness, which a sequential sweep breaks. A Schur solve is **inherently order-free**: all agents are probed and evaluated from the same iterate, and the coupling is in the linear algebra rather than in the visiting order. So the construction satisfies §4's requirement *by construction* rather than by discipline, and it does not reopen the conflict that page flagged as the first case of two composition-layer guarantees disagreeing.

---

# 8. What this does not fix

Stated up front, because the failure modes of this construction are not the ones it removes.

- **It does not make a bad expert good.** $\Xi_i$ measures whether an agent responds *at all*; $\lVert\Lambda^{\text{expert}}-\Lambda^{\text{ref}}\rVert$ measures whether it responds *correctly*, and a large value there is OP-3 arriving at the interface. Probing diagnoses this precisely and repairs none of it.
- **It does not touch interior error.** [[composition-error-theory]] §3.1's bound is $C_S(\max_i\tau_i+\gamma)$; this construction attacks $\gamma$ and $C_S$ and leaves $\tau_i$ where it is. OP-5's finding that the **windowed architecture carries 66%** of the error is largely an interior-faithfulness problem, and nothing here addresses it.
- **It does not remove the abstention requirement.** [[composition-error-theory]] §4.5 stands unchanged: every bound here is in terms of local defects, and all of them are vacuous where an agent is operating outside its validity domain. *Enforce, measure, or decline* is not superseded.
- **It is unavailable to the frozen checkpoint**, and §4.5 is the proof rather than an inconvenience — $\Lambda\equiv0$ means there is nothing to probe. The construction applies to `WindowNS`/`SolverExpert` today and to any future expert with a boundary channel, which is the same eligibility boundary [[schwarz-iteration-atlas-0.1]] §S2 already drew.

---

# 9. Falsifiable measurements, cheapest first

Each can end a line of enquiry on its own. **(1) is a script, and it decides whether the rest is worth building.**

> **Run 2026-08-27 against `reference.WindowNS` — (1), (2) and (5) are done; (3), (4) and (6) are not.** Full record on [[tier0-measurements]]. **(1)** assembled cleanly, $\kappa\in[18.8,38.5]$, $\beta\in[3.10,6.44]\times10^{-3}$, entries far above the probe floor — **but the null-space dimension came back $0$, not $1$**, on four seams and in every control. The row's own instruction is then correct and it points at the declaration: see the box in §2.1. **(2)** the drift is slow, $0.02\%$–$0.35\%$ over ten macro-steps, so §6's amortization survives. **(5)** $\mu>0$ everywhere, so passivity is **eliminated** as OP-2's cause in this configuration. **The cost column was wrong by two orders**: (1) is seconds, not hours, because a `WindowNS` macro-step is $0.069$ s — the estimate had been made against a neural checkpoint.
>
> **And a seventh measurement, which this table does not contain and should have.** All six rows measure the *operator*. None measures whether the **equation the operator is plugged into** is the right one — and that is what failed. The one that would have caught it costs nothing extra: **substitute the reference solution's own trace into the interface residual and check that it goes down.** It does not (§2.3's box). Adding it here is the durable lesson: a construction can pass every diagnostic it proposes for itself while being wrong about what it is solving.
>
> **2026-08-28: it is now a named measurement in the package** — `probe.reference_trace_check`, three calls and no rollout — and it has been run on both flux conventions at three macro-steps. **Every ratio is above one**: $1.002$ with the exposed agent, $1.028$–$1.040$ with the embedded one, at $\Delta t = 0.05$, $0.01$ and $0.002$. See §2.3's second box for the identity that explains it, and [[tier0-measurements]] §9.3 for the full table.

| # | Measurement | Cost | Prediction | What a failure would mean |
|---|---|---|---|---|
| 1 | **Probe one `WindowNS` agent** — assemble $S_i$ on a 16-mode Fourier basis, report $\kappa$, $\beta$, the null-space dimension, and $\mu=\lambda_{\min}$ of the symmetric part beside the floored $\pi$ | $\sim m{+}1$ window solves; hours | $S_i$ assembles cleanly, null space of dimension $n_0(\Gamma)=1$ — **this is a fluid–fluid seam**; see [[interface-transfer-theory]] §7 — and entries well above the $10^{-6}$ probe floor | A null space of the wrong dimension means the probe or the declaration is broken, not the physics — and it is caught before anything is built on it |
| 2 | **How fast does $S$ drift?** Re-probe the same agent at $t$ and $t+K\Delta t$ for a few $K$ | one extra assembly | slow — the $N$-sweep's insensitivity says the coupling is not delicate | Fast drift kills §6's amortization and with it the budget; the §4.4 Robin fallback survives it |
| 3 | **$\Xi$ for the frozen checkpoint** | minutes | **exactly $0$**, reproducing the bitwise result as an operator norm | Anything nonzero would mean the periodic window has a boundary channel nobody has found, which would be a larger finding than this page. **Run 2026-08-28 on `SpectralNS`: $\lVert\Lambda\rVert$ bitwise zero, 34 solver calls, the word *coupled* refused. Prediction met.** |
| 4 | **$\lVert\Lambda^{\text{expert}}-\Lambda^{\text{ref}}\rVert$ for one boundary-capable expert** | two assemblies | large — this is [[composition-error-theory]]'s $\tau\gg\varepsilon$ claim, and W9's $0.3293$ against a single-step $0.0055$ is a $60\times$ hint | $\approx0$ would put H1 back in play and redirect attention to $C_S$ |
| 5 | **Does OP-2's drift project onto a negative passivity mode?** | one assembly + the existing drift trace | yes — §4.2's [AI Inference], and it would diagnose OP-2 without a rollout | $\lambda_{\min}\ge0$ eliminates passivity as OP-2's cause, cheaply |
| 6 | **Full $S\lambda=\chi$ against the iterated sweep**, $P_2/P_1$ at $N=2$ | one instrumented rollout | moves toward the classical reference; §3 says the $7D$ signal now crosses in one solve | No movement with a well-conditioned $S$ would indict the interior, i.e. confirm OP-5's 66% and retire this line |

**[AI Inference]:** (1), (3) and (5) are independent and all cheap, and (3) is nearly free. Running (3) first is the right order despite it being the least informative, because it is the construction's **positive control** — a method that does not return zero on the case already proved degenerate should not be trusted on cases that are not.

---

# 10. What this changes in the framework

1. **[[port-algebra-atlas-0.1]] gains an operator view of its own ports.** The effort–flow pair is not just a pair of quantities to match; it is the input and output of $\Lambda_i$. This is a reinterpretation of what is already written rather than a new declaration, and it makes the port list the *specification of the probe*.
2. **[[expert-library-atlas-0.1]] gains two measured axes** — the composability index $\Xi_i$ (§4.5) and the probe class *cheap / affordable / expensive* (§5.3). Both are invisible to accuracy benchmarks and both price composability directly.
3. **[[schwarz-iteration-atlas-0.1]] §S3 has an answer.** The coarse space it could not build because coarsening changes $\mathrm{Re}_{\text{eff}}$ is the probed Schur complement, which never changes $L$. §S3's "do not build without settling" is settled.
4. **[[composition-error-theory]] §4.1–4.3 collapse into one measurement.** Passivity, inf-sup, and the optimal Robin coefficient are all read off the same assembled matrix. That is a considerable simplification of a four-item programme, and it changes the ordering: **assemble first, then all three are reports rather than projects.**
5. **[[conservation-as-constraint-atlas-0.1]] gains a third state for a port.** Beyond *enforce* and *measure*: a port can be **ill-conditioned** ($\beta$ at the probe floor), which means matching there is neither enforced nor measured but undefined.

---

## See Also

- [[master-error-bound]] — **where this construction's outputs land in the bound.** $\lVert\Lambda-\tilde\Lambda\rVert$ and $\beta$ are the two factors of the transmission defect $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$, and $\beta$ also enters $L$ — so an ill-conditioned interface costs twice. The bound and this construction were built for each other without either page noticing
- [[general-coupling-scheme]] — **rule R2**, the exception that makes this construction matter: `probed-DtN` needs only `dirichlet`, so the achievable rung is decoupled from the expert's own interface. **§5 adds the caveat this page omits** — going non-overlapping *introduces* cross-points
- [[generalization-requirements]] — **G1**, the cross-point gap this construction creates; and **G5**, where its probe supplies a measurable criterion for *where to cut*
- [[composition-error-theory]] — the page this continues; §4.0's ladder, §4.1's passivity, §4.2's $\beta$, §4.3's Robin, all obtained here from one assembly
- [[temporal-error-accumulation]] — what the probed matrix is *for*: §4.2's positive-realness is the structural route to $L\le1$, the one constant a rollout bound multiplies rather than adds. Its §3.7 promotes $\Lambda_i$ from a matrix to a **transfer function** $\Lambda_i(s)$ over a time window, which is the honest form of incremental passivity — so a probe sweeping window length is a *frequency sweep of the expert*
- [[schwarz-iteration-atlas-0.1]] — §3.2's $H^{-2}$ and $k_{\min}=5$ arithmetic that §3 removes; §S3's coarse-space blocker that §10.3 settles; `reference.WindowNS`, the boundary channel this needs
- [[port-algebra-atlas-0.1]] — the effort–flow pair that §4.1 reinterprets as the DtN interface; the Dirac interconnection that makes §4.2's positive-realness compose
- [[open-problems-atlas-0.1]] — OP-6 ($10^{-6}$ batch dependence → the probe floor, §5.1), OP-4 (no $\nu$ → §4.4's unavailable symbol), OP-2 (drift → §4.2's negative mode), OP-1 (conditioning → §4.3's $\sigma_{\min}$), OP-5 (the 66% interior share this does *not* fix, §8)
- [[symmetry-averaging-atlas-0.1]] — the precedent for the whole move: the composition layer supplying what the expert lacks; and the $|G|=2$ reuse §6 leans on
- [[expert-library-atlas-0.1]] — where $\Xi_i$ and the probe class belong
- [[results-n-sweep-wind-farm]] — the insensitivity result §6's amortization argument rests on
- [[prior-art-and-novelty-atlas-0.1]] — §2's named gap for frozen neural subsolvers; the differentiability claim §5.2 finally spends
- [[conservation-as-constraint-atlas-0.1]] — enforce / measure / decline, extended with *ill-conditioned*
- [[interface-transfer-theory]] — re-expresses §2's construction on a **common interface space**, $\tilde\Lambda_i^M=P_i^\ast\Lambda_iP_i$. The congruence is what carries §4.2's positive-realness through a non-conforming transfer, §2.2's basis-dimension argument becomes the rule for $\dim M$, and §2.1's null-space check becomes the cross-point detector
- [[plug-in-composition-theorems]] — a **partial** elimination of this page's $S\lambda=\chi$ leaves an operator of exactly the type a single expert exposes, which is what makes nesting legal. §4.5's $\Xi=0$ is reinterpreted there as a **failed conformance test** rather than a property of one checkpoint, and §4.3's $\beta$ becomes the substitution certificate's threshold
- [[f1-pathmap-and-end-goal]] — F2, composability without fine-tuning, is what this bears on
