# Composition Error Theory — what "agreement" actually buys

**Type:** Concept page — framework theory, **version-independent** (folder: `Atlas 0.1/common/`; the filename deliberately carries no version suffix, because nothing here depends on the case study or the Atlas revision)
**Status:** written 2026-08-26 in answer to the question *"if the agents agree perfectly and everything is conserved across the interaction graph, is the simulation error on the order of the worst agent's independent accuracy?"*
**Related:** [[schwarz-iteration-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[conservation-as-constraint-atlas-0.1]] · [[open-problems-atlas-0.1]] · [[prior-art-and-novelty-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[results-n-sweep-wind-farm]] · [[symmetry-averaging-atlas-0.1]] · [[f1-pathmap-and-end-goal]]

> **The one-line version.** No — and the way it fails is the most useful thing the framework can learn from it. The worst-agent bound is a **theorem with three hypotheses**. Atlas satisfies one of them (measured), violates one badly (measured), and has never stated the third. "Agreement" is not a scalar condition to be tightened; it is an **operator to be approximated**, and *which* operator decides whether the bound exists at all.

---

# 1. The claim, stated so it can be graded

Let the global problem be $\partial_t u=\mathcal L(u)$ on $\Omega$, and let $\Omega$ be covered by agent domains $\{\Omega_i\}_{i=1}^N$. Write $\mathcal S_i$ for the **exact** local evolution on $\Omega_i$ given boundary data $g$ on $\partial\Omega_i$, and $\mathcal E_i$ for the agent's approximation to it. Let

$$\varepsilon_i \;=\; \bigl\lVert \mathcal E_i - \mathcal S_i\bigr\rVert_{\mathcal D_i}$$

be the agent's **independent accuracy** — the number a benchmark reports, measured over the agent's own validation distribution $\mathcal D_i$.

Let $u^\star$ be the composed solution: the fixed point of the coupled sweep, with interface agreement driven to zero and every declared port conservative.

**The conjecture.** $\;\lVert u-u^\star\rVert \;=\; O\!\bigl(\max_i \varepsilon_i\bigr)$.

**Verdict: false as stated, true under three hypotheses, and the whole content is in which hypothesis breaks.** §2–3 separate them; §4 is the constructive part — what you would build to make each one hold.

---

# 2. Three distinct reasons the conjecture fails

These are genuinely different failure channels. Conflating them is the mistake [[open-problems-atlas-0.1]]'s OP-5 already had to correct once, when a subtraction bucket labelled "windowing" turned out to contain a Reynolds-number mismatch.

## 2.1 Agreement is a *consistency* condition, not a *correctness* condition

This is already proved in this vault, and the proof is unusually clean. [[schwarz-iteration-atlas-0.1]] §3.1: the Schwarz fixed point $u^\star$ satisfies

$$u_i^\star=\mathcal E_i\bigl[u_i^\star;\ \mathcal R_i u^\star\bigr]\quad\forall i$$

and this is the global solution **only if each $\mathcal E_i$ is the exact restriction of the global operator to $\Omega_i$.** Drop that and the iteration still converges — confidently, quickly — to the fixed point of a *different* map.

The precise way to say this is **backward error**: a perfectly agreeing composition does not approximately solve your problem, it **exactly solves a nearby problem**, and the distance between the two problems is not bounded by $\max_i\varepsilon_i$. It is bounded by how far each $\mathcal E_i$ is from being a *restriction* — a different quantity entirely (§3.2).

And the failure is silent **by construction**. Perfect agreement means the interface residual is zero, so the diagnostic that would have reported the trouble is precisely the thing you drove to zero. This vault has the extreme case measured: under periodic windows the sweep is a **constant map in the iteration index**, two passes agreeing *bitwise*, so $r_{\text{iface}}^{(k)}\equiv 0$ identically while $P_2/P_1$ was wrong by a factor of two.

> **Perfect agreement among unfaithful local solves is not a good state. It is the worst state — the wrong answer with the alarm disconnected.**

## 2.2 Conservation constrains a codimension-few subspace of the error

Conservation feels stronger than it is. Written as a constraint on the error $e=u-u^\star$, a conservation law says that finitely many **linear functionals** vanish:

$$\langle \ell_k,\,e\rangle=0,\qquad k=1,\dots,m,\qquad m=O(\text{ports}\times\text{conserved quantities}).$$

The error lives in an infinite-dimensional — in practice $10^5$–$10^7$-dimensional — space. Killing $m\sim10$–$100$ linear functionals of it leaves $\lVert e\rVert$ completely unbounded. Every error mode that **redistributes without net flux** is in the kernel of every conservation constraint you can write: a wake that decays too fast while mass is conserved exactly, a compensating pair of errors, a displaced profile with the right integral.

The wind farm supplies the counterexample at full strength: mass closed at $4.3\times10^{-14}$ — *enforced*, not measured — and the array-efficiency answer was still off by roughly $2\times$. Conservation held to fourteen digits and bought nothing on the quantity of interest.

**Corollary for [[port-algebra-atlas-0.1]] §6.** $\mathcal R(t)$ is a *necessary* condition and an excellent falsifier: $\mathcal R\neq0$ proves the coupling is manufacturing energy. $\mathcal R=0$ proves nothing about accuracy. The page is right to call it the cheapest possible physical-plausibility monitor; it must never be read as an error bound. §4.1 is how to upgrade it into one.

## 2.3 Local errors are amplified and transported, not merely aggregated

Even granting faithful local solves and exact agreement, "$\max_i$" is the wrong aggregator, on both counts.

**Amplification.** Over $n$ macro-steps, with $\Phi$ the composed one-step map and $L=\mathrm{Lip}(\Phi)$ in the relevant norm, a per-step defect $\delta$ accumulates as

$$\lVert e^n\rVert\;\le\;\frac{L^n-1}{L-1}\,\delta\;=\;\begin{cases} n\,\delta, & L=1 \quad(\text{non-expansive})\\[6pt] \dfrac{L^n}{L-1}\,\delta, & L>1\quad(\text{exponential}) \end{cases}$$

Nothing about agreement or conservation controls $L$. This is the entire stability half of the bound, and it is what [[prior-art-and-novelty-atlas-0.1]] §2 points at when it says partitioned-coupling stability theory assumes a Lipschitz input–output map a frozen surrogate does not provide. It is also the shape of OP-2's signature: a *monotone accumulation* rather than a wrong steady state is what $L\gtrsim1$ with a small per-step bias looks like.

**Transport.** Agent $i$'s output is agent $j$'s boundary data. Along an advective chain of length $\ell$ — which is exactly what a wind-farm row, a rocket gas path, or an F1 airflow *is* — errors compose along the path rather than taking a maximum over it. The honest aggregator is a **sensitivity-weighted sum**,

$$e_{\mathcal J}\;\approx\;\sum_i \bigl\langle \phi_i,\ \tau_i\bigr\rangle,$$

with $\phi_i$ the adjoint solution restricted to $\Omega_i$ (§4.4). For $\varepsilon=10^{-2}$ per agent and $O(1)$ weights: $\max_i$ says $10^{-2}$; the sum over an 8-agent chain says $8\times10^{-2}$; at $L=1.05$ per step over 80 steps the same defect says $\approx0.5$. **Three answers, two orders of magnitude, identical $\varepsilon$.**

**One measured piece of good news, and it is Atlas's strongest result to date.** [[results-n-sweep-wind-farm]] took the graph from 8 agents / 15 interfaces to 26 / 63 and moved the answer by under $1\%$, with the fixed point converging in 3 iterations throughout. That is direct evidence that for this problem the amplification factor is **$O(1)$ in graph size** — composition error bounded in $N$ rather than accumulating in it. §2.3's amplification channel is therefore the one channel Atlas has empirically cleared, and it should be quoted that specifically rather than as a general claim about composition.

---

# 3. The correct statement

> **Superseded in one respect, 2026-08-26 — read [[master-error-bound]] alongside this section.** The bound below is correct but **incomplete**: it carries two defect terms where there are three. Telescoping through the interface trace that the *approximate* transmission condition selects splits off $\sigma$ — **transmission infidelity**, the error from posing the interface problem with the wrong operator — from $\gamma$, which is only *failure to converge*. The distinction is load-bearing here, because **iteration reduces $\gamma$ and nothing else**, and §2.1's silent-failure result is precisely the case $\gamma=0$ with $\sigma$ maximal. §4.0's "agreement is an operator" is the informal version of that missing term; its formal version is $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$, which also places §4.2's $\beta$ in a formula for the first time.

## 3.1 A Lax-equivalence-shaped bound for composed agents

Define three quantities, none of which is $\varepsilon_i$:

| symbol | name | definition |
|---|---|---|
| $\tau_i$ | **local consistency defect** | $\bigl\lVert \mathcal S_i[u\rvert_{\Omega_i};\,\mathcal R_i u] - \mathcal E_i[u\rvert_{\Omega_i};\,\mathcal R_i u]\bigr\rVert$ — the agent's error **evaluated on the restriction of the true global solution**, under the boundary data the true solution imposes |
| $\gamma$ | **coupling defect** | the interface residual at convergence of the sweep |
| $C_S$ | **stability constant** | $\bigl\lVert(\mathrm I-D\Phi)^{-1}\bigr\rVert$ for the linearized composed map — §2.3's amplification |

Then, for a well-posed problem and a convergent sweep,

$$\boxed{\;\lVert u-u^\star\rVert\;\le\;C_S\Bigl(\max_i \tau_i \;+\; \gamma\Bigr)\;}$$

**Consistency plus stability gives convergence; agreement alone gives neither.** The conjecture of §1 is exactly this bound with three substitutions silently made: $\gamma=0$ (fair — that *is* what agreement means), $C_S=O(1)$ (a hypothesis, measured favourably in $N$ and *not* measured in $t$), and $\tau_i=\varepsilon_i$ — **false, and this is the crux.**

## 3.2 Why $\tau_i \neq \varepsilon_i$ — the load-bearing paragraph

$\varepsilon_i$ is measured on the agent's own distribution $\mathcal D_i$: interior states it was trained on, boundary conditions it was trained under. $\tau_i$ is measured on the true global solution restricted to $\Omega_i$, with the boundary data the *other agents* impose. These coincide only if composition presents the agent with data it was validated on — and **composition is a distribution-shift machine by construction:**

1. **Boundary data is out of distribution before anything else is.** A checkpoint trained on periodic boxes has never seen an inflow ring. Its $\varepsilon$ on periodic boxes can be $10^{-3}$ while its $\tau$ under imposed boundary data is $O(1)$ — or *undefined*, because it has no channel to accept the data at all. Not hypothetical: this vault recorded a frozen expert being **structurally ineligible** for the boundary fix that moved $P_2/P_1$ from $1.02$ to $0.77$, because a field-in/field-out operator has nowhere to put a boundary ring.
2. **The agent is fed its neighbours' errors, not the truth.** $\tau_i$ is defined on exact data; in the actual sweep the agent sees perturbed data, and the sensitivity of $\mathcal E_i$ to that perturbation is a second unmeasured quantity — the local Lipschitz constant from which $C_S$ is built.
3. **The regime drifts.** A composed rollout visits states no single-step benchmark sampled. With $K$ experts the probability that at least one is out of distribution approaches 1 — [[port-algebra-atlas-0.1]] §8 already flags this and correctly notes it gets *worse* as the interface vocabulary gets cleaner, because a clean contract makes more connections **legal** without making any of them **valid**.

> **The sharp form.** *Independent accuracy is measured against the agent's own benchmark; composition consistency is measured against the true global solution's restriction. The conjecture assumes these are the same number, and every mechanism in a composed system pushes them apart.*

## 3.3 The three hypotheses, graded against what has been measured

| | Hypothesis | What it requires | Atlas status |
|---|---|---|---|
| **H1** | **Faithfulness.** Each $\mathcal E_i$ approximates the *restriction* of the global operator — including its boundary behaviour — on the data composition actually presents | $\tau_i \approx \varepsilon_i$ | **Violated, badly, and measured.** Periodic windows are not restrictions of an open channel; the interface residual was *identically zero* while the answer was $2\times$ wrong |
| **H2** | **Stability.** The composed map is non-expansive, uniformly in graph size and rollout length | $C_S=O(1)$ | **Half-cleared.** Bounded in $N$ (under $1\%$ over a $3\times$ graph). **Never established in $t$** — OP-2's monotone drift is the counter-signal |
| **H3** | **Exact agreement** at the interfaces | $\gamma=0$ | Achievable, and **the least valuable of the three.** Enforced to $10^{-14}$ on mass with no effect on the answer |

The conjecture is not wrong because Atlas built it badly. **It is wrong because H1 and H2 are the entire content, and H3 — the one the conjecture names — carries the least weight.**

---

# 4. Four constructions that would actually deliver the bound

Ranked by how much of §3.3 each buys. The underlying theorems are established literature; **the inferences are about their applicability to frozen neural experts**, and are marked as such.

## 4.0 The reframing that organizes all four: agreement is an operator

The single most useful shift available here. In non-overlapping domain decomposition the *exact* interface condition is not "match the values" — it is the **Steklov–Poincaré (Dirichlet-to-Neumann) operator** $\Lambda_i:\,g\mapsto \partial_n \mathcal S_i[g]$, which encodes everything the rest of the domain does in response to boundary data. Composing with the exact $\Lambda_i$ reproduces the global solution *exactly*, and the composed error is then local error only — **the conjecture is a theorem in that case.**

The catch is the reason the whole field exists: $\Lambda_i$ is **global and nonlocal**, so computing it means solving the problem you decomposed. Every practical transmission condition is an approximation to it, and they form a ladder:

$$\text{Dirichlet}\;\prec\;\text{Neumann}\;\prec\;\text{Dirichlet–Neumann}\;\prec\;\text{Robin}(\alpha)\;\prec\;\text{optimized Robin}\;\prec\;\text{Ventcell (2nd order)}\;\prec\;\Lambda_i$$

$$\boxed{\;\text{composition accuracy is set by }\ \lVert\Lambda_i-\tilde\Lambda_i\rVert,\ \ \text{not by how tightly }\tilde\Lambda_i\ \text{is satisfied}\;}$$

This is the answer to "a better way to make the agents agree." **Do not tighten the agreement; improve the operator being agreed on.** Atlas currently sits at the far-left rung — a one-cell Dirichlet ring — and then iterates it to convergence. That is precision applied to the wrong quantity.

> **Continued at [[probed-dtn-coupling]], and it corrects a prescription made below.** §4.3 tells the expert to accept a Robin condition, and §4.1's storage function likewise assumes something can be asked of the operator's interior — but this vault has twice recorded that frozen experts are *structurally* ineligible for exactly that. The resolution is that $\Lambda_i$ can be built **from outside the expert**, because DtN is by definition Dirichlet-in/Neumann-out and a ring-accepting operator therefore already implements it. Probing it turns §4.1's passivity, §4.2's $\beta$ and §4.3's $\alpha^\star$ into three readings of **one assembled matrix**, which reorders this section: **assemble first, and the rest become reports rather than projects.**

## 4.1 Passivity, not conservation — the upgrade to $\mathcal R(t)$ (highest value)

**The theorem.** Give each agent a **storage function** $H_i(u_i)\ge0$ and require a **dissipation inequality** against the port power [[port-algebra-atlas-0.1]] already defines:

$$\frac{d}{dt}H_i \;\le\; \int_{\partial\Omega_i} e_i f_i\,ds \;-\; \mathcal D_i,\qquad \mathcal D_i\ge0 .$$

Atlas's port connection rule $e_A=e_B,\ f_A=-f_B$ is precisely a **power-preserving (Dirac) interconnection**, so port powers cancel in pairs and the storages add:

$$\frac{d}{dt}\sum_i H_i \;\le\; -\sum_i\mathcal D_i \;+\; P_{\text{ext}}\;\le\;P_{\text{ext}} .$$

**Passivity composes.** The composed system inherits a Lyapunov function from its parts — for any graph, at any $N$, with no global analysis. [[port-algebra-atlas-0.1]] §2 already asserts this in one line and then never uses it; [[backbone-1.1]]'s symmetric-plus-skew core is the same structure.

**The precision that matters, and it is easy to get wrong.** Plain passivity bounds the *state*: it says the coupling cannot manufacture energy, which is stability, not accuracy. What bounds the **error** is **incremental** passivity (equivalently contraction / monotonicity): for two trajectories $u,\tilde u$ of the same agent,

$$\frac{d}{dt}H_i(u_i-\tilde u_i)\;\le\;\int_{\partial\Omega_i}(e_i-\tilde e_i)(f_i-\tilde f_i)\,ds .$$

Incremental passivity of every agent plus a Dirac interconnection gives $L\le1$ for the composed map, hence bounded $C_S$ and **error growth linear in $n$ rather than exponential** — which is H2 obtained *structurally* instead of hoped for.

**Why this is the one to build.** It converts a diagnostic into a bound, it is checkable on a **frozen** checkpoint with no training, and its failure mode is a *number* rather than a yes/no. Define the **passivity defect**

$$\pi_i \;=\; \sup_{u}\ \Bigl[\Delta H_i(u) \;-\; \textstyle\int e_i f_i\,dt\Bigr]^{+},$$

measured by feeding the expert fields and reading what it actually did to the storage. The composed bound then degrades gracefully as $\sum_i\pi_i$ — **a defect that composes additively**, which is exactly the property flux-matching residuals lack.

**[AI Inference]:** this *derives* OP-4's recommended fix instead of choosing it. OP-4 is stuck because $\mathcal R(t)$'s dissipation term needs a single $\nu$ and the checkpoint has none — $81\%$ of its dissipation lives below the scale where one exists. A storage function needs **no viscosity**: you measure the energy the operator removed per call. OP-4's option (1) and this construction are the same object, and the passivity framing says why it generalizes — *every* frozen expert has some dissipation and none of them come with a documented $\nu$.

**[AI Inference]:** and it predicts OP-2. A per-call passivity defect $\pi>0$ integrates to a monotone accumulation, not a wrong steady state — which is the signature OP-2 records and has never explained. If OP-2's drift rate tracks $\pi\times(\text{calls per unit time})$, that is a diagnosis; if it does not, $\pi$ is eliminated cheaply.

**The measurement, in full, because it is a morning's work:** sample fields spanning the operating range; for each, compute $H$ before and after one expert call; compute the port power integral over the same call from the boundary ring; report the histogram of the defect. A negative-definite result would be the strongest positive statement the expert-library programme has produced. A large positive $\pi$ would explain three open problems at once.

## 4.2 Weak interface agreement with an inf-sup condition — the theorem you actually want

**This is the closest thing in the literature to the conjecture, and there it is true.** Mortar element methods enforce interface continuity **weakly**, through a Lagrange multiplier $\lambda$ on an interface space $M_\Gamma$:

$$\int_\Gamma \lambda\,(u_A-u_B)\,ds \;=\; 0\qquad \forall\lambda\in M_\Gamma,$$

and when $M_\Gamma$ satisfies a discrete **inf-sup (LBB) condition**

$$\inf_{\lambda\in M_\Gamma}\ \sup_{v}\ \frac{\int_\Gamma \lambda\,[v]\,ds}{\lVert\lambda\rVert\,\lVert v\rVert}\;\ge\;\beta>0,$$

the method is **optimal**: the global error is bounded by the sum of the subdomains' own best-approximation errors, with a constant independent of the decomposition. *That is the worst-agent bound, and $\beta$ is the hypothesis the conjecture is missing.*

**Two things this immediately explains about Atlas.**

- **Pointwise matching is the over-constrained end of the ladder.** Requiring $e_A=e_B$ *and* $f_A=-f_B$ pointwise between two inexact operators is generically infeasible; the classic symptoms are locking (too many constraints, $\beta$ collapses) and spurious interface modes (too few). Weak enforcement is not a relaxation of a good idea — it is the well-posed version of it.
- **$\beta$ is a reportable per-port number.** [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule currently has no way to say a port is *badly conditioned*. $\beta$ is that quantity, and it is the interface analogue of OP-1's conditioning lesson — *a projection whose sensitivity is not $O(1)$ is not a projection* — learned the expensive way on the thrust constraint and never generalized to the ports.

**[AI Inference]:** the multiplier space for a neural expert is a genuinely open design question, because a frozen operator's "trial space" is not a finite element space and carries no approximation theory. The tractable version: define $M_\Gamma$ over a **declared** interface basis (low-order polynomial or Fourier on $\Gamma$, dimension set by the existing band rule), treat the expert as a black-box solution operator, and measure $\beta$ numerically from the assembled interface system. Untested; cheap; decisive either way.

## 4.3 Optimized transmission conditions — the cheapest real improvement

Straight down §4.0's ladder. Replace the Dirichlet ring with a **Robin** condition

$$\partial_n u_i + \alpha\,u_i \;=\; \partial_n u_j + \alpha\,u_j \qquad\text{on }\Gamma_{ij},$$

and choose $\alpha$ to minimize the Schwarz convergence factor. Classical optimized-Schwarz results give order-of-magnitude better contraction than Dirichlet–Neumann for advection–diffusion, and the *optimal* $\alpha$ is the symbol of the exact DtN — so tuning $\alpha$ is literally climbing the ladder one rung.

**This is also the principled version of a fix OP-2 already wants.** OP-2 concluded the build has **no usable open boundary** — Dirichlet is stable but traps the wake, characteristic is inconsistent and gets *worse* under refinement — and named the remedy as a convective (Orlanski) outflow $\partial_t u+U_c\partial_x u=0$. That is a first-order transmission condition; Robin/Ventcell is the same family with the coefficient chosen by analysis rather than by a convection-speed estimate. **One construction closes OP-2's remedy and improves H1 at every seam.**

**[AI Inference]:** the frozen-expert asymmetry recurs here, and it is the one [[schwarz-iteration-atlas-0.1]] §8 predicted. A Robin condition requires the expert to accept a *flux-plus-value* combination on its boundary. An operator that cannot accept a Dirichlet ring certainly cannot accept a Robin one — so this construction, like the last, is available only to experts with boundary-condition flexibility, and it sharpens that criterion from "can it take boundary data" to "**how far up the DtN ladder can its interface go.**" That is a strictly more informative selection axis for [[expert-library-atlas-0.1]] than accuracy.

## 4.4 Adjoint / dual-weighted-residual localization — the tool that ranks agents

The conjecture asks for a bound in terms of $\max_i$. Goal-oriented a-posteriori theory (dual-weighted residual, Becker–Rannacher) gives the *right* aggregator for a quantity of interest $\mathcal J$:

$$\mathcal J(u)-\mathcal J(u^\star)\;\approx\;\sum_i \underbrace{\rho_i}_{\text{local residual of agent }i}\cdot\underbrace{\omega_i}_{\text{adjoint weight}},$$

where $\omega_i$ comes from solving the **adjoint problem backwards through the same graph**. This converts "assume the worst agent dominates" into a computed ranking of which agent's error actually reaches the answer — and it prices the interfaces too, since the adjoint has its own transmission conditions and its own jump terms.

**Atlas is unusually well placed to do this and has never mentioned it.** [[prior-art-and-novelty-atlas-0.1]] already claims end-to-end differentiability through the coupling as a distinguishing property against preCICE — and end-to-end differentiability *is* the adjoint. The machinery for the strongest available error-localization method is claimed as a headline feature and used for nothing.

**[AI Inference]:** run on the wind farm, this is a direct test of a claim the case study currently makes by argument. OP-5 attributes shares to windowing / assembly / checkpoint **by subtraction**, and the entry itself records that the subtraction bucket contained a Reynolds-number mismatch. An adjoint-weighted residual attributes them **by construction**, per agent and per seam, in one backward pass — with no differencing of runs that differ in more than one variable, which is the exact failure mode that page is a monument to.

## 4.5 The precondition none of the four removes: abstention

All four constructions bound error in terms of *local* defects. Every one is vacuous if an agent is asked to operate where it is invalid, because there the defect is unbounded and unmeasured. §3.2 point 3 is a structural feature of composition, not a data problem, and it *worsens* as the port vocabulary gets cleaner.

**The framework consequence:** an expert declaration should carry a **validity domain** alongside its ports — conditioning ranges plus an in-distribution check on the incoming state — and the coupled solve should be permitted to **abstain** rather than return a number. This extends [[conservation-as-constraint-atlas-0.1]]'s rule by one step: *enforce, measure, or decline*.

---

# 5. What this changes in the framework

Concrete, and each is small on its own:

1. **[[port-algebra-atlas-0.1]] gains a storage function per expert**, alongside its port list, and $\mathcal R(t)$ is reported as an **inequality with a defect $\pi_i$** rather than as a residual. Highest-value single change on that page.
2. **Transmission condition becomes a declared, graded property of a port** — which rung of §4.0's ladder it sits on — not an implementation detail of the coupling loop. Two experts may share a port type and still be uncomposable at the rung the problem needs.
3. **Report $\tau_i$ separately from $\varepsilon_i$** for every expert. A library entry listing benchmark accuracy and not consistency-under-imposed-boundary-data is listing the number that does not enter the bound.
4. **$\beta$ (interface inf-sup) joins the per-port metric set**, generalizing OP-1's hard-won conditioning lesson from the thrust constraint to every interface.
5. **[[schwarz-iteration-atlas-0.1]] §4's open question gets an answer.** That page found two composition-layer guarantees *conflicting* — multiplicative ordering breaks symmetry averaging's exactness — and called the missing port-resolution-order rule a genuine gap. Passivity supplies the rule: **an ordering is admissible iff it preserves the power-preserving interconnection**, which additive sweeps do and sequential sweeps do not, for the same structural reason.

---

# 6. Falsifiable predictions, cheapest first

Each can end a line of enquiry on its own.

| # | Measurement | Cost | Prediction | What a failure would mean |
|---|---|---|---|---|
| 1 | **Passivity defect $\pi$ of the frozen checkpoint** — $\Delta H$ per call vs. port power over the boundary ring | hours, no training | $\pi>0$ and systematic: the operator is dissipative but not *incrementally* passive | If $\pi\approx0$, H2 holds structurally and OP-2's drift needs a different cause |
| 2 | **Does OP-2's drift rate track $\pi\times$ calls-per-unit-time?** | one sweep | yes — a per-call bias, not a wrong steady state | If it tracks the clock instead, the bias is in the transmission condition, not the operator |
| 3 | **$\tau$ vs. $\varepsilon$ for one expert** — benchmark accuracy against accuracy on the classical reference's restriction, under imposed boundary data | one rollout; the reference already exists | $\tau/\varepsilon\gg1$. This is §3.2's whole argument, and W9's $0.3293$ against a single-step $0.0055$ is a $60\times$ hint of it | $\tau\approx\varepsilon$ would mean H1 holds and the error is elsewhere, reopening H2 |
| 4 | **Robin transmission at the tile seams**, $\alpha$ swept | a solver-expert sweep | measurable improvement where Dirichlet traps and characteristic diverges — OP-2's remedy, obtained by analysis | No improvement across $\alpha$ indicates the defect is not the transmission condition |
| 5 | **Adjoint-weighted attribution of the OP-5 gap** | one backward pass per rollout | agrees with the subtraction shares in *ranking*, disagrees in *magnitude* | Disagreement in ranking retires the subtraction decomposition outright |

**[AI Inference]:** (1) and (3) should be done first and they are independent. (3) is the direct test of §3.2, the load-bearing claim of this page — and the vault has both a validated classical reference and a working harness, so it is a script rather than a project.

---

# 7. The answer in one paragraph

Perfect agreement plus perfect conservation does **not** bound the error by the worst agent's independent accuracy. On its own it bounds nothing, because agreement is a *consistency* condition that a self-consistently wrong composition satisfies exactly, and conservation constrains only finitely many linear functionals of an infinite-dimensional error. What is true is $\lVert u-u^\star\rVert\le C_S(\max_i\tau_i+\gamma)$, and the conjecture is that bound with the stability constant assumed $O(1)$ and — the substitution that actually matters — with the agent's *benchmark* accuracy $\varepsilon_i$ put in place of its *consistency defect* $\tau_i$: the error it commits on the true global solution's restriction, under the boundary data its neighbours impose. Composition drives those two numbers apart by construction. The constructive route is therefore not tighter agreement but a **better operator to agree on** — climb from a Dirichlet ring toward the Steklov–Poincaré operator (§4.0, §4.3), enforce the match **weakly** through a conditioned multiplier space so the bound has a constant (§4.2), and replace the conservation *residual* with a per-expert **dissipation inequality** so stability composes structurally rather than being hoped for (§4.1). Do those and the conjecture becomes a theorem whose hypotheses you can measure — a considerably more valuable object than a conjecture that happens to be true.

---

## See Also

- [[master-error-bound]] — **this page's claims, derived as one bound, plus the term it was missing.** The defect splits *three* ways, $\tau+\sigma+\gamma$, and $\sigma$ — transmission infidelity — is what §4.0's "agreement is an operator" was about: $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$, with §4.2's $\beta$ finally placed in a formula. It also proves §2.1 in one line (iteration reduces $\gamma$ and nothing else) and shows the periodic-window failure is a **pure-$\sigma$** failure the two-term split cannot express
- [[general-coupling-scheme]] — §4's four constructions as coordinates of a seven-parameter scheme, compiled from the agents' declared capabilities rather than chosen
- [[generalization-requirements]] — what is still missing for arbitrary systems; the "perfect agents" premise discharges $\tau$ and leaves $\sigma$, $\gamma$ and $L$ untouched
- [[temporal-error-accumulation]] — **§2.3's amplification channel, assembled into a rollout bound.** Shows a monolithic autoregressive model obeys the *identical* law, so the case for decomposition is an inequality about $L$ rather than a claim about error growth; adds multi-step agreement (Schwarz waveform relaxation / multiple shooting, conditioning $L^W$ not $L^N$); and separates conservation from agreement, with only the dissipation inequality bounding $\lVert e\rVert$ rather than functionals of it
- [[probed-dtn-coupling]] — **the constructive continuation of §4.0.** Builds $\Lambda_i$ by probing a black-box expert, which makes §4.1–§4.3 three readings of one matrix, supplies the coarse space [[schwarz-iteration-atlas-0.1]] §S3 could not, and turns §4.5's yes/no eligibility into a measured norm that scores the frozen checkpoint at exactly zero
- [[schwarz-iteration-atlas-0.1]] — §3.1's faithfulness hypothesis, and the bitwise proof that a converged sweep can be a constant map
- [[port-algebra-atlas-0.1]] — the Dirac interconnection this needs; §6's $\mathcal R(t)$, and why it is a falsifier rather than a bound
- [[conservation-as-constraint-atlas-0.1]] — enforce-or-measure, extended here to *enforce, measure, or decline*
- [[open-problems-atlas-0.1]] — OP-2 (drift → passivity defect), OP-4 (no $\nu$ → storage function), OP-5 (subtraction bucket → adjoint attribution), OP-1 (conditioning → inf-sup)
- [[results-n-sweep-wind-farm]] — the measurement that clears H2 in graph size
- [[prior-art-and-novelty-atlas-0.1]] — §2's named gap, partitioned-coupling stability theory not covering neural subsolvers; and the differentiability claim §4.4 would finally spend
- [[expert-library-atlas-0.1]] — boundary-condition flexibility, sharpened into *how far up the DtN ladder an expert's interface can go*
- [[symmetry-averaging-atlas-0.1]] — the precedent: the composition layer supplying a property the expert lacks
- [[plug-in-composition-theorems]] — **§4.5's abstention precondition, named as the architecture's single axiom.** `validity` is falsifiable but not verifiable, so it is the one declaration no conformance test can certify — and every bound on this page is conditioned on it
- [[interface-transfer-theory]] — §4.2's inf-sup constant, obtained as a **measurement rather than a proof**, which is the only version available to a frozen expert with no approximation theory
- [[f1-pathmap-and-end-goal]] — F2, composability without fine-tuning, is what all of this bears on
