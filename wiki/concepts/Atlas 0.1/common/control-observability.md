# Control observability — is a globally receptive operator uncontrollable, or merely unconfined?

**Type:** Concept page — measured result, **version-independent** (folder: `Atlas 0.1/common/`; no Atlas revision suffix, for the same reason as [[probed-dtn-coupling]] — nothing here depends on the case study or the revision)
**Status:** measured 2026-09-06, Tier 32. Artifact `out/w149/w149.json`, driver `scripts/w149_control_observability.py`, suite `tests/test_tier32_control_observability.py`. **This tier is one measurement and its controls.** No optimiser was built, `compiler.py` was not touched, and no `Accelerator` member was added — the rows that would do those things are opened on [[gap-worklist]] and left there.
**Related:** [[probed-dtn-coupling]] · [[master-error-bound]] · [[atlas-and-standard-dd-theory]] · [[poc2-demo-and-novelty]] · [[gap-worklist]] · [[tier0-measurements]] · [[composition-error-theory]] · [[schwarz-iteration-atlas-0.1]] · [[interface-transfer-theory]] · [[port-algebra-atlas-0.1]] · [[open-problems-atlas-0.1]] · [[expert-library-atlas-0.1]]

> **The one-line version.** Global receptivity makes $\Pi$ **exactly one**, which does not violate [[master-error-bound]] §4.1 — it makes it *vacuous*, leaving $\lVert\delta\lambda\rVert$ as the only free factor. Attacking that factor means solving for the artificial-boundary datum instead of lagging it, and whether that is possible is a question about **observability**, which nothing in this vault had measured. It is now: on one seam, **Poseidon-T's control-to-jump map is full rank $32/32$ at $\kappa = 4.48$** — better conditioned than the classical solver carrying its own elliptic solve ($19.4$), against $1.95$ for the one that does not. **The hypothesis holds: confinement and observability are different properties, and global receptivity is fatal to the first and not to the second.** And the second half is negative and belongs in the same sentence: **minimising the objective moves $\sigma$ the wrong way** by $6.6\times$ to $21\times$, while the monolith's own datum, in the same control coordinates, moves it to within $0.7\%$ of the floor. The channel carries the correction; this objective's minimiser is not it.

---

# 1. The argument this tier tests

[[master-error-bound]] §4.1 bounds the transmission error of an **overlapping** scheme:

$$\Pi \;:=\; \max_j\;\sum_i \chi_{ij}\,\mathbf 1\!\left[\,b_i(j)\le d_i\,\right],\qquad \sigma\;\le\;C_\mu\,\Pi\,\lVert\delta\lambda\rVert ,\qquad C_\mu\in[0.228,\,1.178]\ \text{measured}$$

with $d_i$ the agent's domain of dependence over one exchange interval. **A globally receptive agent has $d_i=\infty$, so the indicator is identically $1$, so**

$$\Pi \;=\; \max_j\;\sum_i \chi_{ij} \;=\; 1 \qquad\textbf{exactly}$$

— pinned by the partition-of-unity identity, for every halo width and every ramp. Set that against the lever working normally: at fixed halo, ramp $1\to$ ramp $21$ took $\Pi$ from $0.75$ to $0.020$ and $\sigma$ from $1.20\times10^{-4}$ to $6.26\times10^{-9}$, a factor of $1.9\times10^{4}$.

> **$\Pi = 1$ is not a failure.** The bound is not violated; it is **uninformative**, and the distinction matters because the two have different remedies. A violated bound wants a better bound. A vacuous one wants a different factor. **This is a certificate becoming unavailable, not a defect appearing.**

Exactly one factor is then left free: $\lVert\delta\lambda\rVert$, the **staleness of the artificial-boundary datum**. Atlas has never attacked it, because every case so far had a local operator where attacking $\Pi$ was cheaper. **Virtual control** — solving for the boundary datum rather than lagging it — is the mechanism for the other factor.

The same fact drives the elliptic firing. `compiler.py:1312` already records that an embedded elliptic solve has an infinite domain of dependence whether or not its region was cut, so [[poc2-demo-and-novelty]]'s beat 1 — where Poseidon-T is refused on `L2/R10/halo`, `L2/R10/W60` and `L2/R10` — is **one fact with one home in the bound**, seen through three rule firings.

## 1.1 The hypothesis, stated so it can fail

> **Confinement and observability are different properties, not ordered by strictness. Global receptivity is fatal to the first and neutral-or-favourable to the second.**

`probe.support_reach`'s $128$-of-$128$ result — Poseidon-T's response reaching every cell of a seam where its record declares stencil radius $2$ — is the **numerator** of that argument and not the whole of it. **Nonzero everywhere is not well-conditioned everywhere**, and a response can be dense and nearly rank-deficient at the same time. The quantity that closes the gap is the smallest singular value of the **control-to-jump map**, written $\beta_{\text{ctrl}}$ throughout this page. Nothing on [[probed-dtn-coupling]] establishes it and nothing should be read as though it did.

## 1.2 The object, and the cadence it is measured at

$$J(g) \;=\; \tfrac12\!\!\sum_{i<j}\ \int_{\Omega_i\cap\Omega_j}\!\!\omega_{ij}\,\bigl\lvert\,\mathcal E_i[u^n,g_i] - \mathcal E_j[u^n,g_j]\,\bigr\rvert^2 \;+\; \tfrac{\eta}{2}\lVert g - g_{\text{lag}}\rVert^2$$

$$T \;:=\; \frac{\partial\,(\text{jump})}{\partial g}\Big|_{g_{\text{lag}}},\qquad \beta_{\text{ctrl}} := \sigma_{\min}(T),\qquad \kappa := \sigma_{\max}(T)/\sigma_{\min}(T)$$

$\eta = 0$ for every consistency check on this page, and that is not a simplification: a Tikhonov term centred **on the lagged trace** hands the lagged column a bonus and the reference column a penalty in the one comparison the check exists to make.

**One exchange per macro-step, for all four columns.** It is the only cadence all four share — Poseidon-T is a one-shot map at a fixed lead time and cannot sub-step at all — and it is the cadence `WindowAgent.respond` uses, so every $\Lambda$ in this vault was probed at it. Two consequences, stated here rather than discovered later:

- the EXPOSED column here is **not** the split-step scheme of [[tier0-measurements]], which exchanges every sub-step (**R10b**); its numbers are not comparable to that scheme's $\sigma = 3.73\times10^{-8}$;
- at this cadence $d = \rho s = 20$ cells against a $21$-cell overlap, so the classical agent's **confinement advantage is nearly absent here too**. That is the right way round for this tier: holding confinement roughly equal across the columns is what leaves observability as the thing being compared.

---

# 2. Gate 0 — the objective, before anything else

[[probed-dtn-coupling]] §2.3 killed flux balance for an explicit one-step map, and how it died is the standard this objective had to clear: the reference trace did **not** reduce the residual ($2.6421\times10^{-5}\to2.6520\times10^{-5}$, marginally *worse*), solving overshot $41\times/75\times/150\times$ as $\Delta t$ refined, and the identity turned out to be $F_A+F_B=-\nu h\,\partial_{nn}w$ — a discrete second derivative, not a jump.

`probe.reference_trace_check` was run against $J$ on `reference.WindowNS`, where a monolithic reference trajectory exists, at $\Delta t = 0.05$, $0.01$ and $0.002$. **The result splits cleanly on the elliptic axis, and it splits the way the compiler already does.**

| $\Delta t$ | elliptic **exposed** (L2 admits) | elliptic **embedded** (L2/R10 refuses) |
|---|---|---|
| $0.05$ | $\mathbf{0.9884}$ | $1.0752$ |
| $0.01$ | $\mathbf{0.9907}$ | $1.0199$ |
| $0.002$ | $\mathbf{0.9760}$ | $1.0015$ |

*(ratio $= J(\text{reference})/J(\text{lagged})$; below $1$ passes.)*

And $J$ is **monotone along the whole line** from the lagged trace to the monolith's own — decreasing at every one of five points in the exposed column, increasing at every one in the embedded column. A ratio near $1$ could be noise; a ratio near $1$ at the end of a monotone line is a direction.

> **The literal form of the gate — every row — does not hold, and the artifact keeps saying so.** What holds is the gate on the admissible column. The embedded column is the configuration `L2/R10` already refuses, and its failure carries **R10's own signature rather than flux balance's**: the defect **grows with $\Delta t$** ($1.0015 \to 1.0199 \to 1.0752$), where flux balance's grew as $\Delta t$ **shrank**. A defect that accumulates over more sub-steps is a per-window elliptic solve; a defect that survives refinement is a steady-state condition. These are not the same failure and the $\Delta t$ sweep is what separates them.

**[AI Inference]:** that the objective inherits R10's refusal rather than failing independently of it is the more useful reading — it says $J$ is meaningful exactly where the compiler already says the decomposition is meaningful, which is a consistency between two instruments built for different purposes. Untested beyond this one graph.

---

# 3. Gate 1 — the decisive measurement

$T$ assembled on one seam (`sx0`, `W00`'s `xhi` against `W10`'s `xlo`), $m+1$ marches per side, no optimiser and no rollout. Three probe states, and the first two columns are controls.

| # | agent | $\beta_{\text{ctrl}}$ | $\sigma_{\max}$ | $\kappa$ | rank | bottom gap | controllable fraction | $\Xi_{\text{ctrl}}$ | spread over 3 states |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `SpectralNS`, periodic | $\mathbf{0}$ *exactly* | $0$ | — | $0/32$ | — | $0$ | $\mathbf 0$ | — |
| 2 | `WindowNS`, elliptic **exposed** | $1.381\times10^{-1}$ | $2.69$–$2.94\times10^{-1}$ | $\mathbf{1.95}$–$2.17$ | $32/32$ | $0.643$ @ $16$ | $0.36$–$0.73$ | $1.00$ | $1.025\times$ |
| 3 | `WindowNS`, elliptic **embedded** | $3.887\times10^{-2}$ | $7.54\times10^{-1}$ | $19.4$ | $32/32$ | $\mathbf{0.166}$ @ $\mathbf{31}$ | $0.92$–$0.98$ | $2.80$ | $1.004\times$ |
| 2b | `WindowNS` exposed, **both ring components** | $1.297\times10^{-1}$ | $2.70$–$2.96\times10^{-1}$ | $2.08$–$2.33$ | $64/64$ | $0.755$ @ $32$ | $\mathbf{0.93}$–$\mathbf{0.95}$ | $1.00$ | $1.019\times$ |
| 4 | **Poseidon-T** | $\mathbf{8.373\times10^{-3}}$ | $3.75$–$3.85\times10^{-2}$ | $\mathbf{4.47}$–$\mathbf{4.61}$ | $\mathbf{32/32}$ | $0.635$ @ $16$ | $0.42$–$0.58$ | $\mathbf{0.139}$ | $1.005\times$ |

$\Xi_{\text{ctrl}} := \lVert T^{\text{expert}}\rVert/\lVert T^{\text{ref}}\rVert$ is [[probed-dtn-coupling]] §4.5's composability index asked of the **control** map rather than the DtN map: how much of the reference agent's control authority this one reproduces.

## 3.1 The answer the tier exists for

**Poseidon-T is not unobservable from a seam.** $T$ is full rank $32/32$ at every probe state, $\kappa = 4.47$–$4.61$, and $\beta_{\text{ctrl}}$ moves $1.005\times$ across states separated by $10.6\%$.

The comparison that carries the argument is with **column 3, not column 2**. Poseidon-T's control map is **better conditioned ($4.48$) than the classical solver's when that solver carries its own elliptic solve ($19.4$)** — the configuration `R10` refuses, and the one whose global receptivity is of exactly the kind Poseidon-T's is. Against the R10-compliant classical agent it is $2.3\times$ worse, which is a real gap and a small one next to the $32\times$ discrepancy `support_reach` measured between the checkpoint's declared stencil radius and its actual reach.

$\beta_{\text{ctrl}}$ itself is $6.1\%$ of the reference agent's — but so is the checkpoint's whole response: $\Xi_{\text{ctrl}} = 0.139$. **Scaled by its own response magnitude, the checkpoint's control authority is $14\%$ of the reference's and its conditioning penalty is $2.3\times$.** Those are two different statements and quoting only the first would be the same error as reading a small $\beta$ off an operator that is uniformly small.

## 3.2 The predicted rank-one null space, found where it was predicted

[[probed-dtn-coupling]] §2.1 records that $n_0(\Gamma)$ **is not a property of the seam alone**: with incompressibility inside $\Lambda_i$ the trace is constrained to zero net flux and $n_0=1$; with the elliptic part in the composition layer, $\Lambda_i$ is the pure advection–diffusion DtN and $n_0=0$. The prediction transfers to $T$ and it is confirmed, three ways at once:

| | elliptic **embedded** | elliptic **exposed** |
|---|---|---|
| bottom spectral gap | $\mathbf{0.166}$, at index $\mathbf{31}$ of $32$ | $0.643$, at index $16$ — mid-spectrum, i.e. not a gap |
| null dimension by that gap | $\mathbf 1$ | — |
| overlap of the smallest right singular vector with the **constant** modes | $\mathbf{0.9974}$ | $0.169$ |
| its split between the two sides | $0.7074\,/\,0.7068$ | $1.000\,/\,0.000$ |

The direction is the constant (net-flux) mode, and it is **carried equally by the two sides** — which is what a net-flux constraint *should* look like, because it is a statement about the seam rather than about either window. All four rows are asserted in the suite rather than described here, including the exposed control: a single small singular value with no constant-mode content would pass a bare `null_dim == 1` check and fails these.

**The predicted dimension is a free correctness check and it passed.** A probed operator whose measured null space is not the dimension that seam predicts is reporting a defect in the probe, not in the physics — and this one reports neither.

## 3.3 A block asymmetry, reproduced

The per-side DtN blocks come off the same marches, so they cost nothing extra:

| seam side | $\beta$ | $\kappa$ |
|---|---|---|
| `W00` (upstream), exposed | $3.463\times10^{-1}$ | $1.13$ |
| `W10` (downstream), exposed | $1.668\times10^{-5}$ | $\mathbf{4171}$ |
| `P00` (upstream), Poseidon-T | $5.746\times10^{-5}$ | $57.1$ |
| `P10` (downstream), Poseidon-T | $3.696\times10^{-5}$ | $179.6$ |

That is [[probed-dtn-coupling]] §4.2's downstream finding reproducing on a different instrument — there $\beta = 2.67\times10^{-5}$, $\kappa = 1529$; here $\beta = 1.67\times10^{-5}$, $\kappa = 4171$, a $20\,760\times$ block asymmetry within one graph.

**And it is the sharpest single piece of evidence for the hypothesis.** The downstream block's **DtN** is nearly rank-deficient ($\kappa = 4171$) while the seam's **control map** is well conditioned ($\kappa = 2.17$) at the same state, from the same solver calls. Two orders of magnitude between the two, on one seam, at one instant: **the response operator's conditioning and the control map's conditioning are not the same number and do not have to move together.** That is the abstract claim of §1.1, measured.

---

# 4. The controls

Five, and one of them is designed to be able to fail.

| control | result |
|---|---|
| the field march against `WindowAgent.respond`'s flux | **bitwise equal**, max difference $0.0$ on a flux of $5.198\times10^{-4}$ |
| `SpectralNS` positive control | $T$ **bitwise zero**, $\beta_{\text{ctrl}}=0$, rank $0/32$, $\Xi_{\text{ctrl}}=0$, at all three states — while the jump itself is $9.9\times10^{-2}$, so the zero operator is informative rather than vacuous |
| $\varepsilon$ sweep, `WindowNS` | $1.0024\times$ (exposed) and $1.0005\times$ (embedded) over **five decades**, $10^{-1}$ to $10^{-5}$ |
| $\varepsilon$ sweep, **Poseidon-T** | $1.0562\times$ over four decades — and $\mathbf{1.0040\times}$ dropping the smallest step |
| $\sigma$ against the exact-datum run, evaluated at the exact datum | $\mathbf 0$ exactly — an identity, not a floor (**W106**) |

> **The checkpoint's sweep is the one that can fail, and it half did — which is the control working.** `window_ns` declares a machine-epsilon probe floor because its expert is a deterministic solver. Poseidon-T's declared floor is $10^{-6}$ (OP-6's batch-position dependence), six orders higher. Its $\beta_{\text{ctrl}}$ reads $8.351/8.390/8.400\times10^{-3}$ at steps $10^{-1}/10^{-2}/10^{-3}$ — converging — and then jumps to $8.820\times10^{-3}$ at $10^{-4}$, a $5\%$ break upward. **That is cancellation against the declared floor arriving exactly where the declaration says it should.** The quoted value is taken in the converged band, and the break is reported rather than trimmed.

**The probe-state horizon is the control that was got wrong first, and the correction is worth recording.** The three states were originally taken by marching the settled state, and $\beta_{\text{ctrl}}$ came back flat to $1.0000\times$. It was flat because **the settled state is a fixed point**: $200$ macro-steps move it $6.03\times10^{-4}$ relative and it saturates by step $40$. A spread quoted along that is a spread across **one state wearing three labels** — [[gap-worklist]]'s *positive controls need a horizon* in its purest form, where the horizon was not short but absent. Spinning up from uniform inflow under the same forcing gives states at $10.5\%$, $7.9\%$ and $0\%$ from the settled one, a separation of $\mathbf{10.6\%}$ and two orders more motion, at cell Reynolds numbers $4.24$, $4.74$ and $4.83$ against `WindowAgent.validity`'s declared $8$ — so **all three are inside the envelope** and the spread is not bought with an out-of-envelope state. The last mark reproduces `s0_state.npz` to $0.0\times10^{0}$, which says that artifact was made by this march and is a free check that the trajectory is the right one.

---

# 5. The negative half, and it belongs in the same sentence as the favourable one

$\sigma$ here is the vault's own definition — the composed macro-step under a candidate datum against **the same composed step under the monolith's own datum**, so $\tau$ cancels exactly. Measured at the settled state:

| | $\sigma$ | relative to lagged |
|---|---|---|
| lagged (what the halo scheme does now) | $5.536\times10^{-7}$ | $1$ |
| **floor**: a perfect datum on this one seam | $3.753\times10^{-7}$ | $\mathbf{0.678}$ |
| the monolith's own datum, **declared port** (normal component) | $5.115\times10^{-7}$ | $0.924$ |
| the monolith's own datum, **both ring components** | $\mathbf{3.791\times10^{-7}}$ | $\mathbf{0.685}$ |
| **solving $J$**, unregularised, both components | $1.171\times10^{-5}$ | $\mathbf{21.1}$ |
| solving $J$, best over three regularisation families | $4.829\times10^{-7}$ | $0.872$ |

**Two things, and they have to be read together.**

**The channel carries the correction.** With the widened control, the monolith's own datum reaches $0.685$ against an available floor of $0.678$ — it captures $\mathbf{97.7\%}$ of what a perfect control on this seam could buy. So the seam is not merely observable in the rank sense; the correction that would remove $32.2\%$ of $\sigma$ is **expressible in the control's own coordinates**. That is a stronger statement than $\beta_{\text{ctrl}} > 0$ and it is the one the tier needed.

**And minimising $J$ does not find it.** The unregularised solve moves $\sigma$ **up** by $21\times$, and it overshoots the true datum by $36.8\times$ in norm — [[probed-dtn-coupling]] §2.3's own diagnostic, on the new objective, at a magnitude between flux balance's $41\times$ and $150\times$. This survives every attempt to regularise it away: **Tikhonov over ten decades of $\eta$, truncated SVD over six retained ranks, and a plain step-length sweep**, with $\sigma$ re-evaluated by *marching* at every point and the best selected on $\sigma$ itself rather than on $J$. The most favourable point any of the three families offers is a heavily damped step ($\alpha = 0.03$, which lowers $J$ by only $5\%$) at $0.872$ — capturing $12.8\%$ of the $32.2\%$ available. Through the **declared** port it is $0.9989$: $0.1\%$ of $32.2\%$, i.e. nothing.

> **The mechanism, and it is not the one flux balance had.** $T$ is a genuine derivative — the change in $J$ it predicts at the reference control matches the change a march produces to $\mathbf{0.1\%}$ ($1.0009$ and $0.9999$). Gate 0 passes. The map is full rank. What is wrong is that **$J$'s minimiser is not the truth**: the jump on the overlap has sources the seam datum does not own — chiefly the other artificial faces, held stale — and the seam control can *imitate* them. Driving the two windows into agreement therefore buys agreement, not correctness. **Agreement and correctness are different objectives, and this is the second time this vault has paid to learn that a residual going to zero is not the same as a solution going to the truth.**

**[AI Inference]:** the natural next form is to stop asking the agents to agree and start asking them to agree *with something* — a defect-correction or trust-region objective anchored on the previous accepted state rather than on mutual agreement, or a $J$ restricted to the sub-region the seam datum actually reaches. Both are cheap to falsify with this instrument, since the expensive part — $T$, the $\sigma$ evaluator and the controls — is built and artifact-backed. Neither is tested and neither should be read as a plan.

---

# 6. The declared port is narrower than the datum, and it costs

A `MECH` ring carries two velocity components. The **declared** port carries one — the normal component, which is the effort–flow pair [[port-algebra-atlas-0.1]] specifies. Same seam, same states, same probe, one more component:

| | declared (normal only) | widened (both components) |
|---|---|---|
| control dimension | $32$ | $64$ |
| controllable fraction of the jump | $0.36$–$0.73$ | $\mathbf{0.93}$–$\mathbf{0.95}$ |
| $\kappa$ | $1.95$–$2.17$ | $2.08$–$2.33$ |
| $\sigma$ at the monolith's own datum | $0.924$ | $\mathbf{0.685}$ |

**Widening the control costs $6\%$ of $\beta_{\text{ctrl}}$ and essentially nothing in conditioning, and it is the difference between reaching $7.6\%$ and $97.7\%$ of the available reduction.** This is a fact about the **port declaration**, not about the physics or the agent — and it is the kind of thing that is invisible until something tries to use the port as an actuator rather than as a matching condition. [[gap-worklist]] **W152**.

---

# 7. What this page may not claim

Four, and they are load-bearing rather than decorative.

1. **This does not make neural operators composable.** It attacks $\sigma$ and $\gamma$. $\tau$ is untouched, and $\Sigma$ has no $\tau$ axis. The best case this line of work can reach is converting a **refusal into `admit-uncertified` with a measured $\sigma$** — never a seam this framework would certify, and nothing here changes that.
2. **$\Pi = 1$ is not a failure.** It is a certificate becoming unavailable. The bound still holds; it stops distinguishing configurations, which is a different problem with a different remedy.
3. **None of this is faster.** A $3\times$ coupling tax is affordable only where the learned expert is much cheaper than the classical one, and at these grid sizes it is **not faster** — [[poc1a-frozen-expert-results]] records the checkpoint at $128^2$ against a solver that is quicker on the same window. Virtual control adds forward calls on top of that.
4. **R12's global projection and this are not the same move.** R12 **imposes** a second elliptic answer on top of the agents' own and made things worse. This **solves for the input under which the agents' own answers agree** — no second answer is imposed, and the agents are never told anything. If that distinction blurs, the idea has not been understood.

---

# 8. Where this leaves the argument

**The hypothesis of §1.1 survives its first test.** On one seam, at one cadence, with three probe states and five controls: a globally receptive learned operator is **observable from its seam** — full rank, $\kappa = 4.48$, stable to $1.005\times$ across a $10.6\%$ state separation and to $1.004\times$ across three decades of probe step. It is better conditioned than the classical agent in the configuration whose global receptivity most resembles its own. **Global receptivity cost it $\Pi$ and did not cost it observability.**

**What is not established is that anything can be built on that.** The objective proposed for exploiting it passes its consistency check and its minimiser still moves $\sigma$ the wrong way, under every regularisation tried. So the tier ends where it should: **the door is open and the key that was cut for it does not turn.** [[gap-worklist]] **W150** carries the `Accelerator = VIRTUAL_CONTROL` axis and states plainly that its precondition is an objective whose minimiser is the truth — which this one is not, and which is now a measured statement rather than a worry.

**Scope, stated because it is narrow.** One seam of one graph, one governing family, one expert pair per column, one cadence, one $\Delta t$ for gate 1. $\beta_{\text{ctrl}}$ is a local number in exactly the sense every probed quantity on [[probed-dtn-coupling]] is local: it exists at a state, and the three states here are on one spin-up of one configuration. Nothing establishes that $\kappa = 4.48$ travels to another checkpoint, another seam of this one, or a multiphysics port.
