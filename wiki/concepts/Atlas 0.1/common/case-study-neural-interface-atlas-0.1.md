# CS-S1 — the learned operator on the interface, and the ceiling on what a guess can buy

**Not a rung on [[case-study-ladder-to-f1]]'s climb — the first case study whose variable is the SCHEME, which is why it carries `S1` rather than a number from the ladder's sequence (CS-16 is `vehicle.py`, rung 9).** `atlas/cases/neural_interface.py`, `scripts/w166_neural_interface.py`, `tests/test_tier38_neural_interface.py`, `out/w166/w166.json`. Built 2026-09-09.

Every substitution certificate in this vault has refused Poseidon-T, and **R10** is why: a neural operator's domain of dependence is the whole window — [[gap-worklist]]'s W93 measured 64 cells against a declared 2, nonzero in all 128 seam cells — so cutting the domain cuts the operator, and no halo repairs it. That refusal is correct and it is **not about accuracy**. Fine-tuning the checkpoint to a one-step error of $10^{-9}$ would not move it.

So put the learned operator where its globality is an asset instead of a liability:

> **Let the neural operator predict the interface trace. Let classical solvers own the subdomains.**

Three things follow, and **only the third is a hypothesis**:

1. **It cannot introduce a silent error.** A starting guess for a *convergent* iteration is a preconditioner: a wrong guess costs iterations, not correctness.
2. **It needs no certificate, because it is not an agent.** The refusal machinery is about experts that *produce* the answer; this one produces a guess.
3. **It should be good at this**, because the interface trace genuinely is determined by the whole domain — the thing Schwarz needs many sweeps to propagate and a global operator has for free. [[control-observability]] measured the supporting half from the other side: Poseidon's control-to-jump map is full rank $32/32$ at $\kappa = 4.48$, *better* conditioned than the classical solver carrying its own elliptic part at $19.4$. **Global receptivity cost it $\Pi$ and did not cost it observability.**

**1 and 2 hold and are measured. 3 is false, for a reason that is not about Poseidon.** And the lever that does work was already in the vault, unquoted.

---

## 1. The unknown, and why it took two arrangements to find it

[[general-coupling-scheme]] and `atlas/solve.py` pose the interface problem in the **multiplier space**: a common 16-mode $\lambda$ imposed on both sides' rings, whose root balances two outward fluxes,

$$r(\lambda) \;=\; \sum_i P_i^{*}\,\mathcal F_i\!\left(P_i \lambda\right) \;=\; 0 .$$

On an **overlapping** decomposition those two rings are 20 cells apart — `C00`'s `xhi` sits at global $x = 127$ and `C10`'s `xlo` at $x = 107$. Measured, $\lVert\lambda^\star\rVert$ comes out **more than 5× the size of the field's own change over the macro-step in the same reduced space** (22× at the first measurement). So $\lambda^\star$ is not a trace that any predictor of the new *field* predicts, and in that arrangement **every field predictor scored negative against a zero start — the classical one at $-0.025$, Poseidon at $-0.103$, amplitude-matched noise at $-0.170$.**

**The classical arm is what settles it.** A prediction built from the very solver that owns the subdomains also scored negative, so the failure is a fact about the unknown and not about the checkpoint. `test_the_multiplier_root_is_far_bigger_than_the_fields_own_change` pins it.

The **overlapping Schwarz transmission condition** is a different object, and it is the one the composed rollout actually runs on. `reference.WindowNS.step_batch(bc0, bc1)` reads the ring at *both* ends of the macro-step and interpolates across the sub-steps; the shipped scheme passes `bc1=None`, which **holds the old ring across the step**. So the unknown is the ring at $t+\Delta t$, and the fixed point is where every window's artificial-face data equals the assembled field there. That *is* a physical trace.

`test_the_shipped_scheme_is_this_iterations_ZEROTH_iterate` asserts the two agree **to the bit**: this iteration's cold start is exactly the scheme the vault ships, so everything below is about a scheme somebody runs.

**One correctness fix came out of building it.** An artificial face's ring can *end* on the domain edge — `C00`'s `yhi` ring runs the full width of the window and its first cell sits at global $x = 0$, the inlet. Writing the iterate there let the domain boundary drift: wrecking the edge by $7.0$ moved the sweep's output by $9.2\times10^{-6}$, a leak of $1.3\times10^{-6}$. Small, and exactly the kind of small a fixed point quietly becomes a property of the tiling through. `_domain_edge_mask` holds those cells; the test that found it is kept, with a control that shows an *interior* seam does move the sweep.

---

## 2. What is safe by construction, and it is measured rather than argued

**The fixed point does not depend on where the iteration starts.** Five arms — hold-the-old-ring, one classical sweep, Poseidon-T, amplitude-matched noise, and the oracle — marched to the same tolerance from starts that differ by more than an order of magnitude:

| arrangement | worst pair disagreement | tolerance | bound |
|---|---|---|---|
| split-step (admitted) | $1.55\times10^{-6}$ | $9.95\times10^{-6}$ | $2\,\text{tol}$ |
| as-built (refused) | $6.95\times10^{-6}$ | $5.20\times10^{-6}$ | $2\,\text{tol}$ |

**The bound to read these against is $2\,\text{tol}$ and not $\text{tol}$**, and the second row is why it matters: each arm stops within $\text{tol}$ of the limit, so two arms sit $2\,\text{tol}$ apart at worst by the triangle inequality. Asserting $\le \text{tol}$ would be asserting something the stopping rule does not promise — and at $1.34\times$ it would have failed.

The strongest form is the noise arm. A start that is **pure noise at ten times the size of the thing being solved for** converges to the same answer. That is what makes the predictor exempt from a substitution certificate: not a convention about what counts as an agent, but a measurement that the predictor's output *cannot reach* the answer.

The structural half is checked mechanically. `predictor_is_not_an_agent` returns `True` for all four predictors on the compiled graph — no agent id, no port, no weight hash, no note mentions them — and **the check can fail**: a predictor named `C00` returns `False`. A check that cannot fail is not a check.

---

## 3. The ceiling, and it is one sweep

A linearly convergent iteration needs

$$m \;=\; \frac{\log\!\left(d_0/\varepsilon\right)}{\log\left(1/\rho\right)}$$

sweeps. **A predictor moves $d_0$. Only the operator moves $\rho$.** So the ceiling is not an opinion, and there is an instrument for it: `OneSweep` — one classical sweep, used as a prediction — is the first iterate, built from the same solver that owns the subdomains, so no prediction of the new field can be much better.

**It saves exactly one sweep, in both arrangements, at every tolerance, because it *is* one sweep** — and it costs exactly the sweep it saves, so its total comes out identical to the incumbent's *to the unit*:

| arrangement | cold | one sweep | saved | cold total | one-sweep total |
|---|---|---|---|---|---|
| split-step | 2 sweeps | 1 sweep | 1 | 8 solves | 8 solves |
| as-built | 35 sweeps | 34 sweeps | 1 | 140 solves | 140 solves |

**A field predictor's entire budget is one sweep**, which on this graph is four subdomain solves. That is the number every candidate has to beat, and it is also the harness's own control: a harness in which `OneSweep` saved two would be double-counting, and one in which it saved none would not be measuring the start at all.

---

## 4. Poseidon-T measured against that ceiling

On the arrangement the compiler admits:

| arm | skill | sweeps | predictor cost | total | final distance |
|---|---|---|---|---|---|
| hold old ring *(the shipped scheme)* | $+0.000$ | 2 | 0 | 8 | $1.22\times10^{-7}$ |
| one classical sweep | $+0.998$ | 1 | 4 | 8 | $1.22\times10^{-7}$ |
| **poseidon-T** | $\mathbf{-1.201}$ | 2 | $\mathbf{244}$ | 252 | $5.76\times10^{-7}$ |
| amplitude-matched noise | $-1.597$ | 2 | 0 | 8 | $1.28\times10^{-6}$ |
| oracle *(the fixed point)* | $+1.000$ | 1 | — | — | $2.10\times10^{-16}$ |

$\text{skill} = 1 - \lVert g_0 - g^\star\rVert / \lVert g_{\text{cold}} - g^\star\rVert$: one for a perfect start, zero for the incumbent, **negative for a start that is worse than doing nothing**.

**Two independent reasons it does not pay, and either alone is enough.**

**Cost.** Poseidon costs **244 subdomain solves** in the admitted arrangement — $61\times$ the four-solve break-even — and 27 in the refused one, $7\times$ it. The two differ because the as-built window step costs about $9\times$ the split-step one (it runs a Poisson solve per sub-step), so the same checkpoint buys more sweeps there. Both are **single-shot timing ratios measured in-process** and neither is quoted more finely than its order; the conclusion is the same at either.

**Skill.** It is negative, and it is barely better than noise: $-1.201$ against $-1.597$ in the admitted arrangement, $-3.253$ against $-3.684$ in the refused one. **The gap between a 20.8M-parameter checkpoint and pure noise of the same amplitude is 0.4 of a skill point, in both.**

**[AI Inference]:** the mechanism is legible and is [[case-study-scaling-ladder-atlas-0.1]]'s $\Pi$ seen from the other side. The checkpoint's error is spread over the whole window, so its *ring* is as wrong as its interior — and the ring is the only part a transmission condition reads. Globality helps the trace to be *determined by* the far field; it does not help the trace to be *accurate*. Nothing here measures that decomposition, and it would take a per-mode skill spectrum to.

---

## 5. The lever that does work: one declared field, worth $149\times$

The same four windows, the same overlap, the same state, the same tolerance — and one word on the capability record:

| arrangement | sweeps to $10^{-13}$ | contraction | sweeps per decade |
|---|---|---|---|
| as-built, `elliptic_subsolve = embedded` | 466 | $0.9697$ | $74.81$ |
| split-step, `exposed` + global projection | 7 | $0.0102$ | $0.50$ |
| split-step, `exposed`, **no projection** *(control)* | 7 | $0.0087$ | $0.49$ |

**$149\times$, and the control attributes it.** W100 added two things at once — take the pressure solve out of the agent, *and* put one global Leray projection in the composition layer. For the composed **error** both are needed ([[case-study-scaling-ladder-atlas-0.1]]). For the iteration's **rate** only the first is: with the projection switched off the rate is the same to within 2%.

**`L2/R10` refuses an agent with an embedded pressure solve because the composed *error* is elliptic. The same refusal is worth $149\times$ in coupling sweeps**, and the compiler picks the fast arrangement out of the declaration **before a single sweep runs**:

```
split-step   admit-uncertified
as-built     refuse -- L2/R10: 4 agents declare elliptic_subsolve=embedded
                       and the graph decomposes the domain
```

That is an argument a practitioner acts on, and R10's message does not make it. **Opened as W168.**

---

## 6. The halo rule, in a regime it was not derived for

R10's halo rule is stated as an **accuracy** condition on a *one-sweep* scheme: a halo narrower than the agent's reach over the exchange interval carries stale data into the interior, and `assembly.py` charges the contaminated cells for it. Nothing said what it does to an *iterated* scheme, because until this study nothing in the vault iterated one.

Swept from halo 4 to 48, window size and everything else held fixed:

| halo | 4 | 8 | 12 | 16 | 18 | **20** | 21 | 22 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **as-built** | 1.565 | 1.220 | 1.108 | 1.035 | 1.006 | **0.981** | 0.904 | 0.956 | 0.934 | 0.850 | 0.701 |
| **split-step** | 0.089 | 0.018 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 | 0.010 | 0.012 |

The agent's declared reach is $\texttt{STENCIL\_RADIUS} \times \texttt{SUBSTEPS} = 20$ cells.

**As built, the contraction crosses 1 between halo 18 and halo 20 — the bracket lands on the declared reach exactly.** With the elliptic part exposed, the iteration converges at **every** halo from 4 up, at 7–8 sweeps.

So R10's two branches are **one fact**: the embedded elliptic solve is what makes the overlap load-bearing. Take it out and the overlap stops deciding whether the coupling converges at all. *It still decides accuracy* — that is `assembly.py`'s contaminated-cell weight and a different measurement, unaffected by anything here.

**The flat split-step row is reported as a floor, not a null.** The iteration reaches round-off in 7–8 sweeps, so no asymptote is resolvable there. Halo 4 is resolvably worse at $0.089$ over 12 sweeps, which is what says the instrument can tell the rows apart at all. The as-built row is non-monotone at halo 21 vs 22 ($0.904$ against $0.956$) and is recorded that way rather than smoothed.

---

## 7. `alpha_star`, used as a transmission condition for the first time

`probe.py` has emitted `alpha_star` since Tier 0 with the comment that **in a Fourier interface basis the diagonal IS the measured symbol, mode by mode** — the optimal Robin coefficient, read off rather than derived, which for a frozen expert is the only available route ([[atlas-and-standard-dd-theory]] §7). One thing in the vault has ever read it: `SeamOperator.mode_shares`, and it reads it as a ratio. **Nothing has used it as a transmission condition, which is what it is.**

Priced in **expert calls, including the 34-call probe each accelerator needs**, on the as-built seam:

| accelerator | $\varepsilon = 10^{-2}$ | $10^{-3}$ | $10^{-6}$ |
|---|---|---|---|
| Richardson, $\theta = 1/\lVert S\rVert$ | 192 | 288 | 594 |
| Richardson, $\alpha^\star$ **undamped** | *diverges* | *diverges* | *diverges* |
| Richardson, $\alpha^\star$ **damped** | 76 | 100 | **172** |
| Newton on the probed $S$ | 40 | 40 | **44** |

**The undamped arm diverges while cutting $\kappa$ from $22.686$ to $7.345$ — a $3.1\times$ improvement — and the reason is in the same probe.** Richardson converges when the preconditioned spectrum sits inside $(0,2)$, and $\rho(D^{-1}S) = 2.976$. **A condition number is not an iteration.** Damped by $\omega = 1/\rho = 0.336$ it converges and beats plain Richardson by $3.5\times$ at the tight tolerance.

$S$ is **not** a near-diagonal operator here — the off-diagonal share is $0.445$ assembled, and $0.922$ on the `xlo` block alone — which is why a diagonal preconditioner buys a factor rather than a solve.

**And the probed $S$ beats every sweep, which is the vault's own claim measured.** [[atlas-and-standard-dd-theory]] §2.2 argues that a geometric coarse space is unavailable for a frozen expert (enlarging $L$ changes $\mathrm{Re}_{\text{eff}}$) and GenEO is unavailable (no local eigenproblems), and §3 names **the probed Schur complement as the substitute** — dense across the interface, coupling it in one solve. Priced against its own probe it wins by $4.8\times$ at $10^{-2}$ and $13.5\times$ at $10^{-6}$.

**In the admitted arrangement all four collapse to 38–52 calls**, because $\kappa$ is $1.2045$ and there is nothing left to accelerate — which corroborates §8.3's calibration pair ($\kappa = 21.73$ embedded, $1.196$ exposed) on a tiling and a state neither was taken at.

---

## 8. What this does not buy

**No constant is measured on this graph.** $\tau$, $\sigma$, $C_\mu$, $L$ and $\lVert\mathcal A\rVert$ are all `unmeasured` and `MeasuredConstants` is `None`. Every one of them is about composition **error**; this study measures the **iteration**, and they are different questions. Carrying `window_ns`'s numbers over would make the compile look better and would be about neither.

**One expert family, one Reynolds number, one geometry, one state.** The ceiling argument in §3 is structural and does not depend on any of that; the skill numbers in §4 do.

**The predictor arms are measured at a single macro-step.** Over a rollout a predictor's skill could improve — the previous step's converged trace is a better prior than the previous step's field. That is [[interaction-horizon]]'s question and is not measured here.

**Nothing here rehabilitates fine-tuning.** The refusal that blocks substitution is R10 and R10 is about the receptive field, not the loss; and this study's negative is about the *ceiling on a starting guess*, which no amount of training raises above one sweep.

**It declares no new capability record.** The agents are `window_ns.window_capabilities` unchanged and the geometry is `poseidon.PoseidonTiling` unchanged — the affordability claim being *used* rather than restated. `test_no_new_capability_record_is_declared` and `test_the_geometry_is_the_checkpoints_own_and_is_not_re_derived` hold it there.

**And it is the first classical graph on the checkpoint's own geometry** — 235 cells, four 128-cell windows, halo 21, ramp 8, the same four seams and the same prolongations as [[case-study-wake-array-atlas-0.1]]'s Poseidon graph, with `WindowNS` in the windows. That is what makes a predictor arm and a subdomain-owner arm comparable at all.

---

## Links

- [[schwarz-iteration-atlas-0.1]] — the iteration this measures, and the page that predicted the fixed point is what matters
- [[atlas-and-standard-dd-theory]] — §2.2's missing coarse space and §7's Robin coefficient, both measured here for the first time
- [[probed-dtn-coupling]] — $\Lambda$, $\alpha^\star$, and §2.1's assumption that the elliptic part "stays where it is"
- [[case-study-scaling-ladder-atlas-0.1]] — W100's split, whose *other* half this study's control separates out
- [[control-observability]] — Tier 32's $32/32$ at $\kappa = 4.48$, the measurement that made the proposal worth trying
- [[case-study-wake-array-atlas-0.1]] — the checkpoint's own graph on this geometry
- [[gap-worklist]] — Tier 38: **W166** closed with a negative, **W167**, **W168**, **W169** opened
- [[general-coupling-scheme]] — the multiplier-space interface problem of §1, and `Accelerator`
