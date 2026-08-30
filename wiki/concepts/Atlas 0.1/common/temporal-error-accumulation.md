# Temporal Error Accumulation — the rollout bound, and why decomposition changes the constants rather than the law

**Type:** Concept page — framework theory, **version-independent** (folder: `Atlas 0.1/common/`; no Atlas revision suffix, following [[composition-error-theory]] and [[probed-dtn-coupling]])
**Status:** written 2026-08-26 in answer to three questions: *(1) per-step error is agent error plus coupling error and grows exponentially — but doesn't an autoregressive model do the same? (2) can agreement be enforced across several steps rather than one? (3) how is conservation/communication error actually enforced?*
**Related:** [[composition-error-theory]] · [[probed-dtn-coupling]] · [[schwarz-iteration-atlas-0.1]] · [[autoregressive-rollout-stability]] · [[port-algebra-atlas-0.1]] · [[conservation-as-constraint-atlas-0.1]] · [[open-problems-atlas-0.1]] · [[prior-art-and-novelty-atlas-0.1]] · [[results-n-sweep-wind-farm]] · [[expert-library-atlas-0.1]]

> **The one-line version.** **(1) Yes — an autoregressive model obeys the identical recursion, and that is the most important fact on this page.** Composition does not introduce a new error law; it introduces a new *handle* on the one constant that matters. Growth is exponential only when $L>1$, and $L\le1$ is a property that **composes structurally** for a port-connected system and has no per-part analogue in a monolithic network. **(2) Yes, and it is the strongest of the three ideas** — it is Schwarz waveform relaxation / multiple shooting, and its real payoff is not fewer coupling injections but that it **converts a forward march into a simultaneous solve, so conditioning goes like $L^W$ instead of $L^N$**. **(3) Conservation and agreement are different things**: conservation is free and exact via single-valued fluxes and **bounds nothing**; agreement is what you actually solve for; and the only mechanism on the list that bounds the *error* rather than a functional of it is the dissipation inequality.

---

# 1. The recursion, and two corrections to the premise

## 1.1 The premise is right in form

Let $\Phi$ be the composed one-macro-step map and $u^{n,\star}$ the true solution sampled at $t^n$. The one-step defect splits exactly as the question says:

$$d^{n} \;=\; \underbrace{\tau^{n}}_{\text{agent term}} \;+\; \underbrace{\gamma^{n}}_{\text{coupling / communication term}}$$

and with $L=\mathrm{Lip}(\Phi)$ in the relevant norm,

$$\lVert e^{n+1}\rVert \;\le\; L\,\lVert e^{n}\rVert + \tau^{n+1} + \gamma^{n+1} \qquad\Longrightarrow\qquad \boxed{\;\lVert e^{N}\rVert \;\le\; L^{N}\lVert e^{0}\rVert \;+\; \sum_{n=1}^{N} L^{\,N-n}\bigl(\tau^{n}+\gamma^{n}\bigr)\;}$$

That is the whole object. Everything below is about its three constants.

## 1.2 Correction one: it is exponential only in one of three regimes

With $\tau+\gamma=\delta$ constant, the geometric sum closes:

$$\lVert e^{N}\rVert\;\le\;\frac{L^{N}-1}{L-1}\,\delta\;=\;
\begin{cases}
\dfrac{\delta}{1-L}, & L<1 \quad\textbf{contractive — bounded uniformly in } N\\[8pt]
N\,\delta, & L=1 \quad\textbf{non-expansive — linear}\\[8pt]
\dfrac{L^{N}}{L-1}\,\delta, & L>1 \quad\textbf{expansive — exponential}
\end{cases}$$

**Calling the growth "exponential" builds the bad case into the premise, and the bad case is the one thing on this page that is a design choice rather than a fact.** The $L<1$ branch is not a theoretical curiosity: dissipative Navier–Stokes at moderate $\mathrm{Re}$ *is* contractive on its attractor in the energy norm, so a local solve that inherits the physics' own dissipativity inherits $L\le1$ with it. **The error does not accumulate because time passes; it accumulates because the map is expansive.**

**[AI Inference]:** this reframes OP-2 usefully. A *monotone accumulation* rather than a wrong steady state is the signature of $L\gtrsim1$ with a small persistent bias — which is $L=1+\eta$ for small $\eta$, i.e. the boundary between the second and third branches. OP-2 is therefore best read as a measurement that the composed map is **marginally expansive**, and the quantity to report is $\eta$, not the drift magnitude at some chosen $t$.

## 1.3 Correction two: the agent term is $\tau$, not the benchmark number

Carried over from [[composition-error-theory]] §3.2 and worth restating because everything here is built on it. $\tau^n$ is the **consistency defect** — the agent's error evaluated on the *true global solution's restriction to its domain, under the boundary data its neighbours impose*. It is not $\varepsilon_i$, the accuracy a benchmark reports on the agent's own validation distribution. Substituting one for the other is the error that makes the worst-agent conjecture look true. The vault's sharpest instance: **W9 put composition error at $0.3293$ against a single-step $0.0055$ — a $60\times$ gap** between the two numbers for the same operator.

---

# 2. The comparison the question is really asking

## 2.1 A monolithic autoregressive model obeys the same recursion

Yes. Exactly the same, with $\gamma\equiv0$ and $\tau$ the single network's one-step defect:

$$\text{monolithic:}\quad \lVert e^{N}\rVert \le \sum_{n} L_{\text{mono}}^{\,N-n}\,\tau_{\text{mono}}^{n} \qquad\qquad \text{composed:}\quad \lVert e^{N}\rVert \le \sum_{n} L_{\text{comp}}^{\,N-n}\bigl(\tau^{n}_{\text{comp}}+\gamma^{n}\bigr)$$

[[autoregressive-rollout-stability]] is the vault's page on the monolithic side, and its §1 formula was **corrected on 2026-08-26** — it previously claimed error grows as $\epsilon^{k}$, which for $\epsilon<1$ *decays*. The correct statement is the one above, and the correction matters here because it is the exact point of comparison.

**So decomposition buys nothing in the *form* of the bound.** Any argument for Atlas that rests on composed systems having a fundamentally better error law is wrong, and this page exists partly to stop that argument being made.

## 2.2 Where they actually differ — and this is the architecture's whole thesis

Composition is worth doing iff it moves the constants. Four places it does, two favourable and two not:

**Favourable — $L$ factors, and the factors are separately controllable.** For a port-connected system the composed map is $\Phi=\mathcal A\circ\bigl(\bigoplus_i \mathcal E_i\bigr)$, so

$$L_{\text{comp}} \;\lesssim\; L_{\mathcal A}\cdot\max_i L_i$$

where $L_{\mathcal A}$ is the coupling layer's contribution — **designed, not trained**. For a monolithic network $L_{\text{mono}}$ is a single unconstrained property of the weights: unknown, unmeasurable except empirically, and with no structural reason to be $\le1$.

**Favourable — and this is the load-bearing one — $L\le1$ composes.** [[composition-error-theory]] §4.1 and [[probed-dtn-coupling]] §4.2: if every agent is **incrementally passive at its ports** and the connection rule is $e_A=e_B,\ f_A=-f_B$ (a power-preserving Dirac interconnection), storages add and the composed map is non-expansive **for any graph, at any $N$, with no global analysis**. That is $L_{\text{comp}}\le1$ obtained *structurally*.

> **The thesis, stated as the inequality it is:** decomposition is worth its coupling cost precisely when $\;L_{\text{comp}}<L_{\text{mono}}\;$ by enough to pay for $\gamma$. **A monolithic autoregressive network has no per-part handle on its Lipschitz constant; a port-connected system has one per agent, and they compose.** That is the entire argument, and it is checkable rather than rhetorical.

**Unfavourable — $\tau$ can be worse, and there are $N$ chances for it to be.** Composition is a distribution-shift machine ([[composition-error-theory]] §3.2): every agent is fed its neighbours' errors as boundary data, and with $K$ experts the probability that at least one is out of distribution approaches 1. The monolithic model has one distribution-shift problem; the composed system has $N$ plus the interfaces.

**Unfavourable — $\gamma$ exists at all.** It is a term the monolithic model simply does not have. §4 is about how cheaply it can be killed, and the answer is: very cheaply, which is also why it is the least interesting term.

## 2.3 The one measurement the vault has on this comparison

**The monolithic run gives $P_2/P_1 = 0.907$ against the partitioned $0.889$ — removing every agent made it slightly *worse*.** With `SolverExpert` inside every window and no agents at all, OP-5 still finds the answer wrong by $2.2\times$, and attributes $66\%$ to the **windowed architecture**, $19\%$ to assembly, and only $15\%$ to the checkpoint.

Read through this page: on the wind farm, $\tau$ dominates and $\gamma$ is not the binding term. **Agent decomposition is not the defect — the local solve is.** And [[results-n-sweep-wind-farm]] supplies the other half: tripling the graph ($8\to26$ agents, $15\to63$ interfaces) moved the answer under $1\%$, so the amplification factor is $O(1)$ **in graph size**. Composition error is bounded in $N$; what has never been established is that it is bounded in $t$, which is $L$ and is exactly what §1.2 says to measure.

---

# 3. Multi-step agreement — the second question, and the best of the three

## 3.1 It has a name, and the vault has already touched it

Enforcing agreement across a window of $W$ macro-steps instead of one is **Schwarz waveform relaxation** (SWR) in the domain-decomposition literature and **multiple shooting** in the time-integration literature. The interface unknown stops being a trace and becomes a *trajectory*:

$$\lambda \;\in\; \Gamma \quad\longrightarrow\quad \lambda(t) \;\in\; \Gamma\times[T_j,\,T_j+W\Delta t]$$

Each agent solves its own space-time problem over the whole window given $\lambda(t)$, and the iteration is over whole trajectories. [[schwarz-iteration-atlas-0.1]] §5.1 already invokes SWR for its finite-step convergence condition $U_\infty\Delta t_{\text{macro}}<\delta$, but only ever as a one-step argument; the window itself was never made a design variable.

## 3.2 Mechanism A — fewer coupling injections, a factor of $W$

Solving the window as one coupled problem means the coupling residual is driven to tolerance **over the whole window simultaneously**, rather than being injected at each step and then amplified by every subsequent step. With $J=N/W$ windows:

$$\lVert e^{N}\rVert\;\le\;\underbrace{\sum_{n=1}^{N} L^{\,N-n}\tau^{n}}_{\text{still per-step}} \;+\; \underbrace{\sum_{j=1}^{J} L^{\,N-jW}\gamma_j}_{\textbf{count reduced by } W}$$

At $L=1$ this reads $N\tau + (N/W)\,\gamma_W$. **The $\gamma$ contribution falls by a factor of $W$ — provided the window problem can still be converged to the same tolerance, so that $\gamma_W\approx\gamma$ rather than growing with $W$.** That proviso is §3.4's tension and is not free.

## 3.3 Mechanism B — convergence on a bounded window is superlinear, not linear

Steady Schwarz converges linearly with a contraction factor set by overlap and subdomain size. SWR on a **bounded time window** does better: for parabolic problems the classical kernel estimates give **superlinear** convergence in the iteration index, and for advection-dominated problems with finite propagation speed the convergence is **finite** — exactly [[schwarz-iteration-atlas-0.1]] §5.1's condition, where no signal crosses the overlap within the window.

**But the constant degrades as the window lengthens.** Superlinear convergence on $[0,T]$ has a rate that worsens with $T$, so this mechanism **pushes toward short windows** while §3.2 pushes toward long ones. There is an interior optimum, and it is measurable rather than derivable for a neural expert.

## 3.4 Mechanism C — the real payoff: marching becomes a simultaneous solve

This is the mechanism the question is reaching for and it is stronger than the other two.

Enforcing agreement across a window with continuity constraints at window boundaries is **multiple shooting**: the unknowns are the states at the $J$ window boundaries, and the constraints are that each window's propagation lands on the next window's start. The Jacobian is block-bidiagonal with blocks $\Phi^{(W)}$ — the $W$-step propagator — and $I$. Its conditioning is therefore governed by

$$\bigl\lVert\Phi^{(W)}\bigr\rVert \;\approx\; L^{W}\qquad\textbf{not}\qquad L^{N}$$

**That is why multiple shooting is the standard tool for unstable trajectories, and it applies here unchanged.** A forward march over $N$ steps carries the full $L^{N}$ amplification with no recourse. A windowed formulation caps the amplification any single block can contribute at $L^{W}$, and the global coupling is handled by a Newton solve over the window boundaries rather than by transport through $N$ compositions.

**This does not repeal §1.2** — a genuinely expansive system is genuinely unpredictable, and no formulation changes that. What it changes is whether the *numerical method* adds its own amplification on top of the physics'. It converts an exponential in the rollout length into an exponential in a **length you choose**.

## 3.5 The design rule, and a worked number

Choose $W$ so that a single block is well-conditioned — say $L^{W}\le e$:

$$\boxed{\;W^{\star}\;\approx\;\min\Bigl(\underbrace{1/\ln L}_{\text{conditioning, §3.4}},\ \underbrace{W_{\text{SWR}}}_{\text{convergence rate, §3.3}}\Bigr)\;}$$

Using [[composition-error-theory]] §2.3's own worked example, $L=1.05$ over $80$ steps: $1/\ln(1.05)\approx 20.5$, so windows of $\approx20$ macro-steps, each block conditioned at $1.05^{20}\approx2.7$ — against $1.05^{80}\approx49$ for the undivided march. **Same $L$, same defect, a factor of $18$ in the amplification the method contributes.**

$L$ is not currently measured for this build. **It is the single most valuable unmeasured number on this page**, and §6 measurement 1 is how to get it.

## 3.6 When it helps — and the honest prediction for the current build

Mechanism A helps iff $\gamma$ is a meaningful share of $\tau+\gamma$. §2.3 says that on the wind farm **it is not**: OP-5 puts $66\%$ on the windowed architecture, which is a $\tau$ problem, not a $\gamma$ problem.

**Prediction, recorded before running: multi-step agreement will buy little on the current build, and a great deal after the per-window periodicity is fixed.** This is the same ordering [[schwarz-iteration-atlas-0.1]] §5.3 established for iteration — *faithfulness is a precondition, not a peer* — and it applies to windowing for the identical reason. Mechanism C is the exception: it acts on $L$ rather than $\gamma$, so it would help even where $\tau$ dominates, provided $L>1$.

**One structural point in windowing's favour that is specific to frozen experts.** [[schwarz-iteration-atlas-0.1]] §5.1 records that *a frozen expert's native $\Delta t$ imposes an exchange interval it cannot refine* — it cannot go below its native step. Multi-step agreement asks for the **opposite** direction: a *coarser* exchange interval, which a frozen expert can supply by simply being called $W$ times. **Windowing is the one axis on this page that a frozen expert can move along**, which makes it unusually well matched to the constraint that has blocked most other constructions.

## 3.7 In the probed-DtN language, the window is what makes passivity honest

[[probed-dtn-coupling]] assembles $\Lambda_i$ as a matrix from a one-step probe. Over a window, the same probe returns an operator that is a **convolution in time** — a transfer function $\Lambda_i(s)$ rather than a matrix. Two things follow:

- **The optimal transmission condition is nonlocal in time**, which is precisely why optimized SWR uses Robin-in-time (Ventcell) approximations. The one-step Robin coefficient of §4.4 there is the $W=1$ degeneration of a richer object.
- **Passivity takes its correct form.** Incremental passivity is *defined* as positive-realness of the transfer function for $\mathrm{Re}(s)>0$ — the classical positive-real lemma. The matrix condition $\tfrac12(S_i+S_i^{\top})\succeq0$ is that statement at a single frequency. **[AI Inference]:** so the multi-step version does not complicate the passivity story, it is the version the passivity story was always about; and a probe that sweeps window length is a **frequency sweep of the expert**, which would say *at which timescales* an expert is non-passive. Nothing in the vault currently distinguishes that from a scalar defect.

---

# 4. Enforcing conservation and agreement — the third question

## 4.1 They are two different constraints, and conflating them is the trap

- **Conservation** is a statement about **sums**: what leaves $A$ through the seam enters $B$, $\ \oint_\Gamma (f_A+f_B)\,ds = 0$.
- **Agreement** is a statement about **values**: the two agents' traces coincide, $e_A=e_B$ pointwise on $\Gamma$.

You can have exact conservation with no agreement, and agreement with no conservation. [[port-algebra-atlas-0.1]]'s connection rule asks for **both** — $e_A=e_B$ *and* $f_A=-f_B$ — and requiring both **pointwise** between two inexact operators is the over-constrained rung of the ladder ([[composition-error-theory]] §4.2): generically infeasible, with locking and spurious interface modes as the classical symptoms.

**Only one of the two is enforceable for free, and it is not the one that matters.**

## 4.2 The ladder of mechanisms, with what each actually guarantees

| mechanism | guarantees | cost | limitation |
|---|---|---|---|
| **Single-valued numerical flux** — one number per face, used with opposite signs by both sides | **Conservation exactly, structurally.** An identity of the data structure, not a constraint that is solved | **zero** | Conserves only what the flux form conserves. **Bounds nothing about accuracy** |
| **Min-norm projection** onto the constraint manifold | The constraint holds to solver tolerance | one linear solve | **OP-1.** If the correction direction has degenerate sensitivity the projection is a lie *reported as a tiny residual* — $dg/d\lambda=10^{-6}$ against a residual of $10^{-1}$, reported at $7.5\times10^{-11}$ |
| **Weak / mortar enforcement** with a multiplier space | The constraint holds in a *tested* sense, and the global error is bounded by the subdomains' own best-approximation errors **with constant $1/\beta$** | one saddle-point solve | Needs a well-conditioned multiplier space; $\beta$ must be **measured**, not assumed |
| **Dissipation inequality / incremental passivity** | **Bounds the error itself**, by giving $L\le1$ | a storage function per expert; one probe | Requires the expert to *be* incrementally passive — a property to measure, not assume |
| **Abstention** | Nothing, and that is the point | a validity-domain declaration | The precondition all four above need: every bound is vacuous where an agent is invalid |

## 4.3 Conservation is free and exact — and the wind farm proved it buys nothing

The first row is the real answer to "how do we enforce conservation," and it is almost anticlimactic: **make the interface flux single-valued.** Compute it once, hand the same number to both sides with opposite signs. Conservation is then true by construction to roundoff, with no residual to monitor and no constraint to solve. That is how the wind farm reached mass closure at $4.3\times10^{-14}$ — that figure is *machine epsilon*, i.e. it is structural, not the output of an enforcement step.

**And the array-efficiency answer was still wrong by roughly $2\times$.** [[composition-error-theory]] §2.2 is the reason: a conservation law kills $m\sim10$–$100$ **linear functionals** of an error living in $10^{5}$–$10^{7}$ dimensions. Every flux-neutral redistribution — a wake that decays too fast while mass is conserved exactly, a displaced profile with the right integral — is in the kernel of every conservation constraint you can write.

> **Enforce conservation because it is free and because $\mathcal R\neq0$ is an excellent falsifier. Never read $\mathcal R=0$ as an error bound.**

## 4.4 Agreement is not "enforced" — it is solved for, and that is $\gamma$

There is no free structural trick for agreement, because agreement is the unknown. The mechanisms are:

1. **Iterate the exchange** (Schwarz) — converges linearly, and [[schwarz-iteration-atlas-0.1]] §3.2's arithmetic shows one-level iteration cannot transmit a $7D$ signal in the 3 iterations available.
2. **Solve the interface problem** ([[probed-dtn-coupling]]) — one dense solve, global coupling, no iteration count to run out of.
3. **Solve it over a window** (§3) — the same, with the trace promoted to a trajectory.

And the conclusion that ought to govern effort allocation: **$\gamma$ is the cheapest of the three constants to drive to zero and the least valuable to have driven there.** [[composition-error-theory]] §2.1's warning is the sharp version — driving the interface residual to zero **removes the only symptom** of an unfaithful local solve, and the vault holds the extreme case where $\gamma\equiv0$ *bitwise* while the answer was $2\times$ wrong. **Perfect agreement among unfaithful agents is the worst state, not the best.**

## 4.5 The one that bounds the error

Only the fourth row of §4.2's table constrains $\lVert e\rVert$ rather than finitely many functionals of it, and it does so by acting on $L$ rather than on $\delta$. That is the asymmetry worth internalizing: **conservation and agreement both act on the per-step defect, which the bound multiplies by $L^{N-n}$; passivity acts on $L^{N-n}$ itself.** Over a long rollout the second is worth more than any achievable improvement in the first.

---

# 5. The bound, assembled

Collecting §1–§4, with windows of length $W$ and $J=N/W$:

$$\boxed{\;\lVert e^{N}\rVert \;\le\; \underbrace{L^{N}\lVert e^{0}\rVert}_{\text{initial}} \;+\; \underbrace{\sum_{n=1}^{N} L^{\,N-n}\,\tau^{n}}_{\substack{\text{local faithfulness}\\ \text{OP-5's }66\%,\ \textbf{dominant today}}} \;+\; \underbrace{\sum_{j=1}^{J} L^{\,N-jW}\,\gamma_j}_{\substack{\text{coupling}\\ \text{cheapest, least valuable}}}\;}$$

with the three constants owned by three different constructions:

| constant | what controls it | status |
|---|---|---|
| $L$ | **incremental passivity** of each agent + a Dirac interconnection ⟹ $L\le1$ structurally; and §3.4's windowing caps the method's own contribution at $L^{W}$ | **never measured.** The most valuable missing number in the vault |
| $\tau$ | faithfulness of the local solve — a non-periodic window, a real boundary channel, an expert whose $\Lambda$ is close to $\Lambda^{\text{ref}}$ | **measured and bad.** OP-5: $66\%$; W9: $60\times$ its own single-step error |
| $\gamma$ | interface solve — Schwarz, or a Schur/DtN solve, or a windowed version of either | **cheap to kill, and killing it hides $\tau$** |

**The ordering this implies is the opposite of the intuitive one.** The intuitive priority is agreement, then conservation, then stability. The bound's priority is **stability ($L$), then faithfulness ($\tau$), then agreement ($\gamma$)** — and the vault's measurements agree with the bound on every point where they overlap.

---

# 6. Falsifiable measurements

| # | Measurement | Cost | Prediction | What a failure would mean |
|---|---|---|---|---|
| 1 | **Measure $L$.** Perturb a converged state by $\Delta$, roll both trajectories, fit $\lVert e^{n}\rVert$ against $L^{n}$ | one paired rollout | $L=1+\eta$ with small $\eta>0$ — marginally expansive, matching OP-2's monotone drift | $L<1$ would mean the drift is a *bias*, not amplification, and redirects OP-2 entirely |
| 2 | **Does OP-2's drift rate match the fitted $\eta$?** | arithmetic on (1) | yes | A mismatch means a per-call bias rather than an expansive map |
| 3 | **$\tau$ vs $\gamma$ decomposition of the per-step defect** — measure the interface residual and the local defect against the classical reference separately, at the same step | one instrumented rollout | $\tau\gg\gamma$, consistent with OP-5's $66\%$ | $\gamma\gtrsim\tau$ would make §3.2's windowing the top priority instead of a deferred one |
| 4 | **Window sweep**, $W\in\{1,2,5,10,20\}$ at fixed total horizon | $\sim5$ rollouts | little movement now (§3.6); a clear optimum after the periodicity fix | A strong effect *now* would falsify §3.6 and mean $\gamma$ was underestimated |
| 5 | **Is the composed map incrementally passive?** $\lambda_{\min}\bigl(\tfrac12(S+S^{\top})\bigr)$ from [[probed-dtn-coupling]]'s probe | one assembly | negative, and its eigenvector aligns with OP-2's drift direction | $\lambda_{\min}\ge0$ eliminates passivity as OP-2's cause |

**[AI Inference]:** (1) is the highest value-per-hour measurement currently available anywhere in the Atlas programme. It is a paired rollout and a linear fit; it decides which of §5's three terms deserves the next session; and every stability claim the framework makes is currently unfalsifiable without it.

---

# 7. What this changes in the framework

1. **$L$ joins the reported metric set**, alongside the conservation residual and the interface residual. It is the only one of the three that multiplies rather than adds, and it is the only one not currently reported.
2. **Window length $W$ becomes a declared coupling parameter**, not an implicit consequence of the expert's native $\Delta t$. §3.5 gives the rule that sets it.
3. **The claim "composition scales better than a monolith" must be stated as §2.2's inequality**, never as a claim about error laws — they are the same law. [[prior-art-and-novelty-atlas-0.1]]'s scaling argument should be read against this.
4. **[[conservation-as-constraint-atlas-0.1]] gains the §4.1 distinction explicitly**: conservation and agreement are separate constraints with separate mechanisms and separate guarantees, and the port rule asks for both.
5. **Effort ordering is inverted** relative to intuition: stability, then faithfulness, then agreement — §5.

---

## See Also

- [[master-error-bound]] — **this page's recursion, completed.** The per-step defect $\tau+\gamma$ used here is really $\tau+\sigma+\gamma$; the missing middle term is transmission infidelity, and $\beta$ turns out to sit in the denominator of both it and $L$. Also states the **vacuity condition** §1.2's three regimes imply: past $T_{\text{pred}}$ the trajectory bound proves nothing
- [[general-coupling-scheme]] — where $W$ becomes a compiled parameter with rule **R3** (windows need time-varying interface data) and **R4** (a frozen expert can coarsen but never refine)
- [[generalization-requirements]] — **G6**, the chaotic claim class that must replace the trajectory bound past $T_{\text{pred}}$, and **G4**, the multirate stability theory subcycling needs
- [[composition-error-theory]] — where $\tau\neq\varepsilon$ is established, and §2.3's amplification/transport analysis that §1 assembles into a rollout bound
- [[probed-dtn-coupling]] — supplies $L$'s structural handle (positive-realness of the probed matrix) and the interface solve that kills $\gamma$; §3.7 promotes its matrix to a transfer function
- [[schwarz-iteration-atlas-0.1]] — §5.1's SWR condition, generalized here into a window design variable; §5.3's faithfulness-is-a-precondition ordering, which §3.6 reuses
- [[autoregressive-rollout-stability]] — the monolithic side of §2.1's comparison; its error-growth formula was **corrected 2026-08-26** to match §1.2
- [[port-algebra-atlas-0.1]] — the Dirac interconnection that makes $L\le1$ compose; the connection rule §4.1 shows is two constraints, not one
- [[conservation-as-constraint-atlas-0.1]] — enforce / measure / decline, and §4.3's warning against reading $\mathcal R=0$ as a bound
- [[open-problems-atlas-0.1]] — OP-2 (drift → $L=1+\eta$), OP-5 ($\tau$ dominance), OP-1 (projection conditioning, §4.2's second row)
- [[results-n-sweep-wind-farm]] — bounded in $N$, and the reason $t$ is the remaining direction
- [[expert-library-atlas-0.1]] — incremental passivity as a selection criterion, which §2.2 makes the decisive one
