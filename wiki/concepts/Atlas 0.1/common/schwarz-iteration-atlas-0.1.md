# Schwarz Iteration — the composition layer buying field-of-view the expert does not have

**Type:** Concept page — **built 2026-08-23; the plan's central prediction was upgraded to a proof, and stage S1 was refused as a result** (folder: `Atlas 0.1/common/`)
**Related:** [[open-problems-atlas-0.1]] · [[symmetry-averaging-atlas-0.1]] · [[edge-generation-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[expert-library-atlas-0.1]] · [[conservation-as-constraint-atlas-0.1]] · [[prior-art-and-novelty-atlas-0.1]] · [[wind-farm-implementation-log]] · [[impl-wind-farm-guide]] · [[spec-wind-farm-wake-atlas-0.1]] · [[f1-pathmap-and-end-goal]]
**Code:** `reference.WindowNS`, `SolverExpert.window_bc`/`.force_mode`/`step_many(bc=)`, `CoupleConfig.window_bc`/`.schwarz`/`.schwarz_tol`/`.force_mode`, `CoupledSystem._sweep`/`._sweep_once`/`._overlap_masks`, `StepRecord.schwarz_*`, `invariants.symmetry` ring forwarding, `scripts/windfarm_schwarz.py`, `tests/atlas/windfarm/test_schwarz.py` (11 tests; existing 175 unaffected).

> **The one-line version.** A frozen expert sees a $2D$ window. The farm is $24D$. Classical domain decomposition buys global consistency from local solves by *iterating* the interface exchange instead of exchanging once — and it costs expert calls, not training. **But the theory that makes it work has a precondition this build currently violates, and finding that out is worth more than the fix.**

---

## 1. The intuitive picture

Twelve people are each given a small window onto a long river and asked to predict the next thirty seconds. Each can see a little way past their own stretch — that overlap is the only thing connecting them.

**One-pass exchange** is: everyone looks at their neighbour's *current* sketch once, then draws. Whatever a neighbour discovers while drawing arrives too late.

**Schwarz iteration** is: everyone draws, shows their neighbours, and draws again with the updated edges — repeating until nobody's drawing changes. The fixed point of that loop, in classical numerical analysis, *is* the answer you would have got from one person with a view of the whole river.

That last sentence is a theorem, and like all theorems it has hypotheses. The one that matters here is that **each person must be solving the real problem on their own stretch.** If one of them has been told their stretch wraps around — that water leaving the right edge re-enters at the left — then no amount of comparing notes fixes it. The group will converge, confidently and quickly, to a consistent picture of the wrong river.

That is the shape of the finding this page exists to record before any code is written.

---

## 2. What already exists, and what is actually missing

Two things must be separated before "add Schwarz iteration" means anything, because **the word already has a different meaning in this case study.**

### 2.1 "Schwarz halo" (R1) is not what this page proposes

[[impl-wind-farm-guide]] §6.2's **R1** is the *overlapping halo* interface variant — agents overlap and the edge layer supplies halo tokens from the neighbour's interior. It was built, tested at W3 ($r_T = 6.2\times10^{-4}$), and **not adopted**; **R2** (declared flux-BC tokens) passed too, so by the guide's own outcome table the declared-edge contract was validated and R2 is in use.

R1 vs R2 asks **what data crosses an interface.** This page asks **how many times per macro-step it crosses.** The two are orthogonal: R2 can be iterated, and the adopted interface contract needs no change. Nothing here reopens the W3 decision.

### 2.2 The iteration machinery exists; the convergence test is on the wrong quantity

The current macro-step is *not* a single sweep. [[spec-wind-farm-wake-atlas-0.1]] §8.3's fixed point already re-enters the fluid solve:

$$T^{(k+1)}=(1-\theta)\,T^{(k)}+\theta\cdot\tfrac12\rho A\,C_T'\,\bigl\langle U_d\bigr\rangle^2\bigl[\mathbf f(T^{(k)})\bigr],\qquad \theta=0.5$$

and W5 measured it converging in **3.02 iterations on average, 5 at worst, every step**. Since $\langle U_d\rangle[\mathbf f(T^{(k)})]$ is the disk-averaged velocity *produced by* applying the body force, each iteration re-runs the fluid sweep.

**So the field exchange is already being iterated roughly three times per macro-step — and the loop is stopped by a scalar test that says nothing about whether the interfaces agree.** The convergence criterion is $|T^{(k+1)}-T^{(k)}|/T<10^{-4}$: a single number describing one rotor.

**Verified 2026-08-23, and the answer is worse than the inference allowed for.** The fixed point *does* re-run the full 124-window sweep — `_fixed_point` calls `_sweep(fx)` inside its loop — so the sweep really is executed three times a macro-step. But it resets the field to `base` each time, so **every one of those sweeps is byte-for-byte the same computation with a different thrust scalar.** Adding an interface-residual term to the convergence test would therefore have measured a residual that is identically zero by construction. §3.1 below is where that leads.

---

## 3. The rigorous statement

Partition $\Omega$ into overlapping subdomains $\{\Omega_i\}_{i=1}^{N}$ with overlap width $\delta$ and subdomain diameter $H$. Let $\mathcal E_i$ denote the local solve on $\Omega_i$ and $\mathcal R_i$ the restriction of the global state to $\Omega_i$'s halo.

**Additive Schwarz.** Every subdomain solves from the *previous* iterate's boundary data, and all are updated together:

$$\mathbf u_i^{(k+1)} \;=\; \mathcal E_i\!\left[\mathbf u_i^{(k)};\ \mathcal R_i\,\mathbf u^{(k)}\right],\qquad i=1,\dots,N \ \text{ simultaneously}$$

**Multiplicative Schwarz.** Subdomains solve in sequence, each using the latest available neighbour data — faster per iteration, and **order-dependent**.

### 3.1 The fixed point is the global solution *only if the local solves are faithful*

The property that makes Schwarz worth doing is that its fixed point $\mathbf u^\star$ satisfies $\mathbf u_i^\star = \mathcal E_i[\mathbf u_i^\star; \mathcal R_i \mathbf u^\star]$ for every $i$ — and if each $\mathcal E_i$ is the **exact restriction of the global operator to $\Omega_i$**, then the pieces agree on overlaps and $\mathbf u^\star$ solves the global problem.

Drop that hypothesis and the theorem gives nothing. If $\mathcal E_i$ solves a *different* problem — a periodic box rather than a piece of an open channel — the iteration still converges, to the fixed point of the wrong map. **Convergence is not correctness, and a converged Schwarz loop reports no distress.** This is §1's wrapped river, and §5 is where it lands on this build.

**Measured 2026-08-23, and it is stronger than "the iteration converges to the wrong answer".** On this build the iteration does not move *at all*. `gather` reads each tile's whole $128^2$ window out of the global field at $t^n$ and the operator returns one field, so the map $\mathbf u^{(k+1)} = \text{scatter}\bigl(E(\text{gather}(\mathbf u^{n}))\bigr)$ **does not depend on $k$**. Two sweeps from the same base with the same forcing agree to `np.array_equal` — bitwise, not to a tolerance, and a tolerance is exactly what would have left room to believe the iteration was doing something small.

$$\text{periodic window} \;\Longrightarrow\; \text{no boundary channel} \;\Longrightarrow\; \text{additive Schwarz is the identity}$$

So per-window periodicity is not merely a *precondition* for the iteration converging to the right thing; it is a precondition for the iteration **existing**. `CoupleConfig` now refuses `schwarz > 1` unless `window_bc='dirichlet'`, with the measurement quoted in the error.

### 3.2 One level does not scale, and the arithmetic is unforgiving

For the model SPD elliptic problem, the classical two-sided estimate (Toselli–Widlund) is

$$\kappa\bigl(M_{\text{AS,1}}^{-1}A\bigr)\;\le\;C\,H^{-2}\bigl(1+H/\delta\bigr), \qquad \kappa\bigl(M_{\text{AS,2}}^{-1}A\bigr)\;\le\;C\bigl(1+H/\delta\bigr)$$

The one-level method carries an $H^{-2}$ — it **degrades as subdomains are added** — and a **coarse space** removes it entirely. The mechanism is transparent: in one-level additive Schwarz, information advances exactly **one overlap-connected neighbour per iteration.**

Applied here: windows are $2D$ with $0.5D$ overlap, so the stride is $1.5D$, and turbine 2 sits $7D$ downstream of turbine 1.

$$k_{\min} \;=\; \left\lceil \frac{7D}{1.5D} \right\rceil \;=\; 5 \ \text{ iterations before turbine 1's wake can influence turbine 2 at all}$$

and $\lceil 24/1.5\rceil = 16$ to cross the domain. **Against a loop that currently runs three iterations, that alone is disqualifying for the metric in dispute** — $P_2/P_1$ is a $7D$-range quantity, and a one-level three-iteration loop cannot transmit a $7D$ signal.

**Caveat, and it is not small.** That estimate is for a *linear, SPD, variationally-posed* problem with exact local solves. A frozen neural operator is none of those. [[prior-art-and-novelty-atlas-0.1]] §2 already names this as the open gap — *partitioned-coupling stability theory assumes properties frozen neural experts lack* — so the theory is a guide to what to measure, never a guarantee to quote.

---

## 4. Why additive, and why that is a framework argument rather than a preference

Multiplicative Schwarz converges in fewer iterations. **Atlas should still use additive**, for three reasons that compound:

1. **Order-independence.** A multiplicative sweep depends on the order subdomains are visited. That is precisely the *path-dependent message passing around the graph cycle* that [[impl-wind-farm-guide]] §10 predicted for W7 — the prediction the 2026-08-23 reattribution showed was a false confirmation, and which "has never been shown to contribute anything." Adopting multiplicative would deliberately install the mechanism the case study just spent a session ruling out.
2. **It would break the exactness of [[symmetry-averaging-atlas-0.1]].** The group average is exact because mirror-paired tiles are treated identically. A sequential sweep visits them at different points in the iteration, so paired tiles no longer see equivalent data, and W7's guarantee degrades from *exact by construction* to *approximate and untested*. **An additive sweep composes with symmetry averaging; a multiplicative one does not.**
3. **Additive is what the hardware wants.** All $N$ windows are independent within an iteration, which is exactly the batched expert call already built — $0.117$ s for one window against $0.022$ s per window at batch 32. Multiplicative serializes and loses the $5\times$.

**[AI Inference]:** point 2 is the durable one. It says composition-layer guarantees can *conflict*, and that the framework needs a rule for which constructions may be combined. That is a genuine gap in [[port-algebra-atlas-0.1]], which currently defines what a port carries but says nothing about the *order* in which ports may be resolved.

---

## 5. The prediction — and why it argues the fix is candidate (1), not candidate (3)

This is the part worth having before the build, because it is falsifiable and it points somewhere other than where **OP-5** currently points.

### 5.1 For evolution problems, overlap buys finite-step convergence

Within one macro-step the subdomain problem is not elliptic but an evolution over $\Delta t$ — the setting of **Schwarz waveform relaxation**. Its governing result is favourable and sharp: for advection-dominated problems, if no signal can cross the overlap within the time window, classical SWR converges in a **finite** number of iterations, and in the limiting case **one**. The condition is

$$U_\infty\,\Delta t_{\text{macro}} \;<\; \delta$$

Every transport mechanism in this build sits **inside** the overlap at the adopted step:

| mechanism | distance travelled in $\Delta t=0.25$ | vs. tile overlap $\delta = 0.5\,D$ |
|---|---|---|
| advection | $U_\infty\Delta t = 0.25\,D$ | $\mathbf{2\times}$ inside |
| diffusion | $\sqrt{\nu\Delta t}\approx0.016\,D$ at $\nu=10^{-3}$ | far inside |
| pressure (elliptic, infinite speed) | global | **already handled globally** — the 2026-08-22 DCT Poisson solve is over the whole domain, not per window |

**So the one mechanism that would defeat one-pass exchange — instantaneous elliptic coupling — is the one already solved globally**, and the two that remain are both contained by the existing overlap.

**[AI Inference]:** this predicts that **Schwarz iteration of the tile seams will buy very little**, and that OP-5's candidate (3) is *not* the dominant term. If the sweep is iterated to a tight interface residual and $P_2/P_1$ barely moves, that is not a failed experiment — it is a measurement that eliminates a third of OP-5's hypothesis space at the cost of a few sweeps.

**Confirmed from the source 2026-08-23:** `SpectralNS.step` computes `n_sub = ceil(dt / (cfl·dx/umax))` and sub-steps internally. So for a *solver* expert $\Delta t = 0.25$ is an **exchange interval, not an integration step**, and OP-5's candidates (2) and (3) are **one knob rather than two** — exactly the collapse §6's closing paragraph said to settle before designing a three-legged sweep. Only the frozen expert genuinely conflates them, because its native lead forces the integration step and the exchange interval to be equal. The general statement therefore sharpens as predicted: *a frozen expert's native $\Delta t$ imposes an exchange interval it cannot refine* — a constraint on **coupling**, not on integration.

### 5.2 The declared ports are marginal where the tile seams are comfortable

The two interface widths in this build are not the same number, and the distinction discriminates:

* **tile seams** — overlap $0.5\,D$, i.e. $2\times$ the advective distance per step. Comfortably inside.
* **declared ports** — the band rule recorded at W0-revisit is $\text{band}\ge U_\infty\Delta t_{\text{macro}}$, adopted as $0.25\,D$. That is **exactly** $U_\infty\Delta t$, sitting on the boundary of the one-iteration condition rather than inside it.

**[AI Inference]:** iterating should therefore help measurably at declared ports and negligibly at tile seams. That is a discriminating prediction — the two seam families are instrumented separately already, and if the effect appears at seams rather than ports the reasoning above is wrong in a way worth knowing.

### 5.3 The precondition this build violates

§3.1's hypothesis is that each local solve is the exact restriction of the global operator. **A periodic window is not.** Each window wraps its own outflow into its own inflow, so imposing distinct neighbour data on its two streamwise edges contradicts the operator's own structure — the edges are *identified* inside the solve.

The consequence is exactly §1's river: **the iteration will converge, and it will converge to the fixed point of the periodic map, not to the global solution.** Iterating a locally-inconsistent solve makes the pieces agree with each other more precisely while leaving them all wrong in the same way — and it removes the interface residual that was the only visible symptom.

**This reorders OP-5's candidates.** Candidate (1), per-window periodicity, is not one of three peers: it is a **precondition** of candidate (3) being able to help at all. Fixing exchange frequency first can only produce a tighter, quieter version of the present error.

**[AI Inference]:** and it explains a number already on the table. The 2026-08-23 harness found the monolithic run at $P_2/P_1 = 0.907$ against the partitioned $0.889$ — **removing every agent made it slightly worse.** Under this reading that is unsurprising: agent decomposition is not the defect, the periodic local solve is, and the monolithic build has just as much of it.

---

## 6. The build plan

Staged so that each stage can end the sequence. **Nothing here trains anything**, per the case study's standing rule.

**Outcome, 2026-08-23.** S0 was answered without being built, S1 was **refused as unbuildable**, S2 was built and is the only stage that turned out to exist. The staging did its job: the sequence ended early, and it ended early because a stage was found to be vacuous rather than because one failed.

### Stage S0 — instrument, before changing any behaviour

Report an **interface residual** per iteration, separately for declared ports and tile seams:

$$r_{\text{iface}}^{(k)} \;=\; \max_{\text{seams}}\ \frac{\bigl\lVert \mathbf u^{(k+1)} - \mathbf u^{(k)}\bigr\rVert_{\infty,\ \text{overlap}}}{U_\infty}$$

Run it against the *existing* three-iteration loop and change nothing else. **This is the cheapest decisive measurement on the page.** If $r_{\text{iface}}$ is already at the expert's per-call noise floor ($\sim3\times10^{-2}\,U_\infty$, W0) after three iterations, then the interfaces already agree as well as the expert can resolve, **iteration cannot help, and stages S1–S3 should not be built.**

*Gate S0: report $r_{\text{iface}}$ at $k=1,2,3$ for both seam families. Proceed only if it is still falling at $k=3$.*

**Answered without instrumenting anything (§3.1).** Under periodic windows $r_{\text{iface}}^{(k)} \equiv 0$ for every $k\ge1$ **identically**, because the sweep is a constant map. The gate is not *passed* or *failed*; it is degenerate, and building the reporter first would have produced a column of exact zeros and an argument about what they meant. The masks were built anyway (53.6% of cells are same-agent tile seams, 25.5% are multi-agent port neighbourhoods, disjoint) because S2 needs them.

### Stage S1 — decouple the convergence test from the thrust scalar

Add $r_{\text{iface}}$ to the existing fixed-point criterion, keep additive ordering, cap iterations at $k_{\max}=8$, and log the full trace. Re-run the $2\times2$ of {frozen, solver} $\times$ {nearest, PoU}.

*Gate S1: does $P_2/P_1$ move? §5.1 predicts it barely does. **Record the prediction before running.***

**Not built — refused at the config, which is the honest form of this result.** `schwarz > 1` under `window_bc='periodic'` now raises, quoting the bitwise measurement. The prediction in §5.1 was that iterating would buy *little*; the truth is that it buys *exactly nothing*, and a prediction of 'small' that resolves to 'identically zero' is worth recording as a miss in the right direction rather than as a hit.

### Stage S2 — the precondition, and it is the real experiment

Give `SolverExpert` a **non-periodic** window solve (the BC machinery of `pressure.py` exists), then repeat S1. This is the only configuration in which §3.1's hypothesis holds, so it is the only one where iterating is *supposed* to converge to the right answer.

*Gate S2: with a faithful local solve and a converged interface residual, does $P_2/P_1$ approach the undivided $0.404$? This is OP-5's decisive measurement, and Schwarz iteration is the instrument rather than the fix.*

**Built as `reference.WindowNS`** — Dirichlet ring one cell wide, ramped from $t^n$ to the neighbours' iterate, interior initial condition pinned at $t^n$. Validated before wiring: uniform flow preserved **exactly** ($0.0$), Taylor-Green second-order convergent, projection divergence $7.6\times10^{-16}$, and a Gaussian wake diffusing within **0.44% of the analytic** rate — the last being the one that matters, since a local solve that damps would be worse than the defect it was built to remove.

**And the gate's own target turned out to be wrong.** $0.404$ was measured at $\nu = 0.03$ ($\mathrm{Re}_D = 33$) while the coupled system runs at $\nu = 1/255$ ($\mathrm{Re}_D = 255$). Re-running the baseline with $dx$ refined in step to hold the cell Reynolds number at 2.08: $P_2/P_1 = 0.4044$ at $\mathrm{Re}_D=33$ (reproducing the published number exactly) and $\mathbf{0.2663}$ at $\mathrm{Re}_D=67$. The target is **not a constant**, and at the coupled system's own Reynolds number it is materially lower than $0.404$. See OP-5's qualification.

### Stage S3 — the coarse space, only if S2 succeeds and $k_{\max}$ binds

§3.2 says a one-level method needs $\ge5$ iterations to carry a signal from turbine 1 to turbine 2 and 16 to cross the domain. If S2 works but needs an impractical $k$, the classical remedy is a coarse solve.

**[AI Inference]:** Atlas already has the two-level structure and has never used it as one. **The 8 declared agents are a coarse decomposition over 124 fine tiles** — that is precisely the $H$/$h$ hierarchy of a two-level method, arrived at for an unrelated reason (the expert's fixed input shape). What is missing is a coarse *solve*, and the honest difficulty is that the frozen expert cannot supply one: coarsening means enlarging $L$, and $\mathrm{Re}_{\text{eff}} = T_s/(\nu_p L^2)$ means that changes the physics rather than the resolution. A classical coarse solve would work but imports a non-expert solver into the loop, which is a question for [[expert-library-atlas-0.1]] and [[f1-pathmap-and-end-goal]] rather than a coding decision. **Do not build S3 without settling that.**

> **Settled 2026-08-26 — the coarse solve exists and imports nothing.** [[probed-dtn-coupling]] assembles a **Schur complement by probing the expert at its native resolution**: every call is a native-sized window at $L_{\text{native}}$, so $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p L^2)$ never moves, and the global coupling comes from *assembling* the calls rather than from coarsening them. That is a two-level method whose coarse operator is built entirely out of fine-scale expert calls — which is exactly the object this paragraph says the frozen expert cannot supply, obtained without asking it for anything it does not already do. The blocker was real; it was a blocker on *coarsening*, and the Schur route does not coarsen. Formalized as rule **R8** in [[general-coupling-scheme]] §3.

---

## 7. Cost

Each additional iteration is one full batched sweep. W5 measured a 124-window sweep at $2.7$ s (batch 32) inside a $\sim24$ s macro-step, so **iterations are cheap relative to the step** — the pressure solve, assembly and metrics dominate. Going from $k=3$ to $k=8$ adds roughly $5\times2.7 \approx 14$ s, under $60\%$ on the step.

Composing with [[symmetry-averaging-atlas-0.1]] multiplies expert calls by $|G|=2$, measured at $1.7\times$ on the full step. The two are **independent and commuting** under additive ordering (§4, point 2), so the combined cost is the product and the combined guarantee is the conjunction.

---

## 8. What this would and would not change

**Would:** give a fixed-window operator an effective field of view larger than its input shape, paid for in forward passes rather than parameters — the same *shape* of result as [[symmetry-averaging-atlas-0.1]], where the composition layer supplied a property the expert lacked.

**Would not:** fix OP-5 on its own. §5.3 is the load-bearing paragraph: iteration converges to the fixed point of whatever the local solve *is*, and the current local solve is periodic. **Built first, this makes a wrong answer more self-consistent and removes the symptom.**

**The honest framing for the log.** The value of this plan is mostly in S0 and S2. S0 can end the line for the price of one instrumented rollout; S2 tests the precondition, and is the same experiment OP-5's candidate (1) already calls for, run with a convergence criterion that makes its result interpretable.

**[AI Inference]:** if S2 succeeds, the general statement available to [[expert-library-atlas-0.1]] is sharper than "iterate the coupling": *a frozen expert can be composed beyond its native field of view if and only if its local solve is a faithful restriction of the global problem* — which makes **boundary-condition flexibility, not accuracy, the primary expert-selection criterion.** A checkpoint pretrained only on periodic data may be uncomposable at any accuracy, and that would be the strongest negative result the expert-library programme has produced.

---

## See Also

- [[composition-error-theory]] — §3.1's faithfulness hypothesis generalized into an error bound; §4 answers this page's missing port-*resolution-order* rule via passivity
- [[general-coupling-scheme]] — **this page generalized.** Every method argued about here is one point in a seven-parameter scheme $\Sigma$, compiled from the agents' declared capabilities: §4's additive-only argument becomes rule **R5**, §S3's coarse-space blocker is discharged by **R8**, the refusal of S1 becomes a general *degenerate-axis* rule, and the §8.3 thrust fixed point turns out to be a mis-specified tolerance rather than a different method
- [[master-error-bound]] — proves §3.1's central claim in one line: iteration drives $\lambda^{(k)}\to\lambda^\dagger$, the root of the problem *you posed*, so **it reduces $\gamma$ and nothing else** — and the periodic-window failure is a pure transmission-infidelity failure
- [[temporal-error-accumulation]] — **makes §5.1's SWR condition a design variable.** The window length $W$ is promoted from an implicit consequence of the expert's native $\Delta t$ to a declared coupling parameter, with $W^\star\approx\min(1/\ln L,\,W_{\text{SWR}})$; and its §3.6 reuses this page's *faithfulness-is-a-precondition-not-a-peer* ordering to predict that windowing, like iteration, buys little until the periodic window is fixed
- [[probed-dtn-coupling]] — **answers §S3.** The coarse solve this page could not build — because coarsening enlarges $L$ and $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p L^2)$ changes the physics — is the *probed Schur complement*, assembled from native-resolution calls so $L$ never moves. It also removes §3.2's $k_{\min}=5$ disqualification (a dense interface operator crosses the graph in one solve, not 16 sweeps), satisfies §4's additive requirement *by construction* since a Schur solve is order-free, and restates §3.1's bitwise constant-map result exactly as $\Lambda\equiv0$
- [[open-problems-atlas-0.1]] — **OP-5**, the problem this addresses; §5.3 argues its candidates (1) and (3) are ordered, not parallel
- [[symmetry-averaging-atlas-0.1]] — the same pattern, already measured; §4 explains why additive ordering is required to compose with it
- [[edge-generation-atlas-0.1]] — Mechanism A materializes the ports this iterates; unchanged by this plan
- [[port-algebra-atlas-0.1]] — where a rule for *resolution order* of ports is missing (§4)
- [[prior-art-and-novelty-atlas-0.1]] — Schwarz, 1870, and §2's named gap in partitioned-coupling theory for frozen experts
- [[expert-library-atlas-0.1]] — boundary-condition flexibility as a selection criterion (§8)
- [[wind-farm-implementation-log]] — the fixed-point trace, band rule and sweep timings quoted here
- [[case-study-rbc-decomposition-atlas-0.1]] — the R1 halo variant in its original framing
