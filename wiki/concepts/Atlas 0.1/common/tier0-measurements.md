# Tier 0 Measured — the bound's constants, against a real expert

**Type:** Concept page — **measurement record** (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-08-27, revised the same day, **and extended four times on 2026-08-28**. §1-§7 record the first run; **§8 records the remediation, and it changes several conclusions above rather than only adding to them**; **§9 closes the last named hole, retracts one of §5.1's promotions, and breaks one of §8's headline numbers**; **§10 closes the last [AI Inference] by deriving one half of it and falsifying the other**; **§11 closes W55 by putting a second discretization and a frozen neural operator through the same stack**; **§12 runs the conformance suite, treats the cross-point, and points the port algebra at real `ADVEC` and `ROT` ports** -- inline notes mark which. **W1, W2 and W3 of [[gap-worklist]] run for the first time against a real forward pass.** Every graph the compiler had seen before this used a hand-declared `boundary_response`; the first of these calls `reference.WindowNS`.
**Read §12.5 first if you are picking this up cold** -- it is the consolidated ledger for the whole 2026-08-28 session; §12.4, §11.8, §10.5 and §9.8 are the per-section ones. **`compile_scheme` now returns `admit` for the `split-step` graph**, which it had never done for anything.
**Three experts, since §11:** `reference.WindowNS` (skew advection, all-Neumann projection), `reference.ChannelNS` (advective form, `project_outflow`, **non-singular**), and **Poseidon-T**, a frozen 20.8M-parameter neural operator. Every rule held; one rule's implementation did not (**W61**), and only the third expert could have shown it.
**Code:** `atlas/cases/window_ns.py`, `atlas/cases/channel_ns.py`, `atlas/cases/poseidon.py` (the three real case studies); `scripts/tier0_window_ns.py`, `scripts/w16_cut_policy.py`, `scripts/w55_second_expert.py` (the drivers); `out/tier0b/`, `out/w16/`, `out/w55/` (the artefacts). Experts: `src/atlas/cases/windfarm/{reference,adapters}.py` in the build repo.
**Related:** [[gap-worklist]] · [[probed-dtn-coupling]] · [[master-error-bound]] · [[interface-transfer-theory]] · [[open-problems-atlas-0.1]] · [[general-coupling-scheme]] · [[plug-in-composition-theorems]] · [[schwarz-iteration-atlas-0.1]] · [[composition-error-theory]]

> **The 2026-08-28 version, in three sentences.** The assembly layer's named hole is closed by a condition on $\chi$ alone -- convexity -- because the cellwise bias-variance identity makes it necessary and sufficient, and `R11` enforces it; $\sigma$ has a bound that applies to a halo scheme, with the first $C_\mu$ this vault has measured; and the composability index scores **exactly zero** on the positive control it was waiting for. **Against that, the time-integrated interface condition §5.1 promoted was built and does not work, and §8's $220\times$ turned out to be a property of a hard-coded sub-step count rather than of the construction** -- with the cadence matched it is $138\times$ to $361\times$ across a $4\times$ range of $\Delta t$. Every one of E1-E7 now stamps `holds` and every bound constant is measured; one unearned inference stands between `split-step` and `admit`.

> **The two-sentence version, after the 2026-08-27 revision.** The probe works exactly as [[probed-dtn-coupling]] says it will, and the equation it was being plugged into was the wrong one -- both diagnosed here and both now fixed in `atlas/`, as rules rather than as notes. **The composed macro-step went from $3.7\times10^{-4}$ to $1.4\times10^{-6}$ against an identical monolith, the interface operator's conditioning from $\kappa=21.7$ to $\kappa=1.20$, and the fitted $L$ from disagreeing with the monolith's to matching it to five decimal places.**

> **The original one-sentence version, kept because the diagnosis is the reusable part.** The probe works exactly as [[probed-dtn-coupling]] says it will — $\tilde\Lambda$ assembles cleanly, $\kappa\approx20$, $\beta$ and $\mu$ are real numbers, the finite difference is stable over six decades — and **the equation that operator is plugged into is wrong**: the true trace does not satisfy the flux-balance condition, so solving it exactly moves the interface $41\times$ further from the truth than leaving it alone.

---

# 1. The configuration, and why each choice is what it is

| | |
|---|---|
| Expert | `reference.WindowNS`, one $128\times128$ window, Dirichlet ring, real forward pass |
| Domain | $255\times255$ at $h=1/64$, tiled by **four** windows that **share their seam layer** ($127+128=255$) |
| Classical reference | the *same class* at $n=255$ — identical stencil, advection form, projection, sub-step rule |
| Physics | $\nu=1/255$, $U_\infty=1$, a frozen actuator strip ($T=0.4444$, halfspan $0.5$, thickness $0.25$) at $(1,1)$ |
| Base state | developed to $t=5$; steady residual $\lVert\dot U\rVert/\lVert U\rVert=1.26\times10^{-3}$ |
| Macro-step | $\Delta t=0.05$ — [[spec-wind-farm-wake-atlas-0.1]] §8.1's value, not the code's off-spec $0.25$ |
| Interface space | $\dim M=16$ real Fourier modes, a **declared** $128\times16$ prolongation with $G_V=hI$; the reduction is forced as $P^\ast$ |
| Probe | finite difference, $\epsilon=10^{-6}$, chosen from a sweep |

**The shared seam layer is the load-bearing geometric choice.** $\Lambda_A+\Lambda_B$ is the Steklov–Poincaré operator only if both blocks act on *the same* trace. A cell-centred tiling with no shared layer gives two cells half a step apart and two different traces, and the sum means nothing. Four squares tiling a square is the smallest configuration with a same-class monolithic reference — two side by side are a $2\!:\!1$ rectangle and `WindowNS` is square-only — and it costs a genuine **cross-point**, which is declared rather than hidden.

**The probe floor is machine epsilon, not OP-6's $10^{-6}$.** `WindowNS` is deterministic to the last bit, so §5.1's argument applies with room to spare:

| $\epsilon$ | $\beta$ | $\kappa$ |
|---|---|---|
| $10^{-2}$ | $5.291321\times10^{-3}$ | $22.7031$ |
| $10^{-4}$ | $5.292135\times10^{-3}$ | $22.6891$ |
| $10^{-6}$ | $5.292143\times10^{-3}$ | $22.6890$ |
| $10^{-8}$ | $5.292143\times10^{-3}$ | $22.6890$ |

Seven-digit agreement across four decades. **The response is linear over at least six decades of trace amplitude**, which is the cheapest available confirmation that a probe of a nonlinear operator is measuring a derivative and not a chord.

---

# 2. W2 — the first real $\tilde\Lambda$

Assembled per seam on the 16-mode basis. $\mu=\lambda_{\min}\bigl(\tfrac12(S+S^\top)\bigr)$ raw; $\pi$ the floored defect.

| seam | orientation | $\beta$ | $\kappa$ | null dim (**declared 1**) | $\mu$ | $\pi$ | cut score $Q$ | $\lVert S-S^\top\rVert/\lVert S\rVert$ |
|---|---|---|---|---|---|---|---|---|
| `sx0` | across the flow | $5.292\times10^{-3}$ | $22.69$ | **0** | $+5.280\times10^{-3}$ | $0$ | $84.1$ | $0.165$ |
| `sx1` | across the flow | $6.435\times10^{-3}$ | $18.81$ | **0** | $+6.391\times10^{-3}$ | $0$ | $60.2$ | $0.157$ |
| `sy0` | along the flow | $3.193\times10^{-3}$ | $37.17$ | **0** | $+2.209\times10^{-3}$ | $0$ | $237.0$ | $1.248$ |
| `sy1` | along the flow | $3.096\times10^{-3}$ | $38.46$ | **0** | $+2.048\times10^{-3}$ | $0$ | $247.1$ | $1.265$ |

**Every assembled seam is passive.** $\mu>0$ on all four, so $\pi=0$ without any clipping being needed — §4.2's floor never fires, and the passivity certificate is earned rather than defaulted.

**Every assembled seam is well conditioned**, $\kappa$ between $19$ and $38$. §4.3's fear — a port with $\beta$ at the probe floor, where matching is undefined — does not materialize: the floor here is $\sim10^{-16}$ and $\beta$ sits three orders above it.

**The cut score orders the seams, and the ordering is physical.** [[gap-worklist]] W16's criterion, measured for the first time: the two seams *across* the flow score $60$–$84$, the two *along* it $237$–$247$. Cutting perpendicular to the flow is $3\times$ better, and $\kappa$ and the asymmetry agree independently. The criterion is [AI Inference] in `probe.cut_score`'s own docstring and it remains so — but it now has data, and the data is not random.

## 2.1 The declared null space is not there, and the declaration is what is wrong

> **Superseded in part by §8.4.** The conclusion below — that the declaration is wrong — stands. What §8.4 adds is *why*, and it rescues [[interface-transfer-theory]] §7 rather than contradicting it: **$n_0(\Gamma)$ depends on where the elliptic solve lives.** Inside $\Lambda_i$ it is $1$; in the composition layer it is $0$. The case study now declares $0$.

`expected_null_dim = 1` on every seam, from [[interface-transfer-theory]] §7 and [[probed-dtn-coupling]] §2.1: incompressibility constrains $\oint_\Gamma\lambda\cdot n=0$, so $S$ should be singular in one direction. **Measured null dimension: $0$, on all four seams, at every probe step, in every control.** The constant mode is not close to null — $\lVert S e_{\text{const}}\rVert = 0.0215$ against $\sigma_{\max}=0.120$ — and there is no spectral gap at the bottom ($\sigma_{14,15,16}=1.05\times10^{-2},\,6.47\times10^{-3},\,5.29\times10^{-3}$).

The theory says a wrong null dimension "indicts the probe or the declaration, not the physics", and that instruction is **correct**: it is the declaration. §2.1 asserts two things that cannot both hold —

1. *"the elliptic part is already global and stays where it is"*, so $\Lambda_i$ is the DtN of the **advection–diffusion** part; and
2. *"incompressibility leaves a rank-one null space"* in $S$.

If pressure is excluded from $\Lambda_i$, the constraint that produces the null space is excluded with it. The two statements are in tension and the measurement resolves which survives. **The null-space check is still worth keeping** — it caught a real error on its first use — but what it caught is a contradiction inside the theory, not a broken probe. See §5.1 for the version of (1) this build also fails.

## 2.2 §2.2's flux and §4.1's flux give the *same* assembled operator

> **Scoped by §8.5.** The $1.11\times10^{-9}$ agreement below is a property of a **shared-layer** seam, where both sides read the same cell and the advective term cancels identically. At a working halo of 21 cells they read cells 20 apart and the assembled operators differ by $3.9\times10^{-2}$. The block-level hazard is unchanged, and it is what settles W47.

[[probed-dtn-coupling]] proposes two different fluxes and never compares them: §2.2 defines $S_i\psi_k=\partial_n(\mathcal E_i[\psi_k]-\mathcal E_i[0])$, and §4.1 argues that reading $\partial_n$ off a one-cell ring is noisy and the **conservative momentum flux** is the right quantity. Both were run.

$$\frac{\lVert S^{\text{mom}}-S^{\text{diff}}\rVert}{\lVert S^{\text{diff}}\rVert}\;=\;1.11\times10^{-9}$$

**They are the same operator.** The advective term $-(\mathbf u\!\cdot\!\mathbf n)\,w$ is evaluated at the shared ring cell, where the two sides carry the same value and opposite normals, so it cancels identically in the sum. §4.1's refinement is a **no-op at the seam** on a shared-layer decomposition.

**Per block it is emphatically not a no-op, and this is the sharper finding.** At `sx0`:

| | $\beta$ | $\kappa$ | $\pi$ |
|---|---|---|---|
| `W00` (upstream), §2.2 flux | $3.424\times10^{-3}$ | $24.07$ | $0$ |
| `W10` (downstream), §2.2 flux | $2.670\times10^{-5}$ | $1529$ | $8.27\times10^{-4}$ |
| `W00` (upstream), §4.1 flux | $1.507$ | $1.59$ | $\mathbf{2.39}$ |
| `W10` (downstream), §4.1 flux | $1.530$ | $1.58$ | $0$ |

**The per-agent passivity verdict flips with a convention the theory leaves open.** Under §4.1's flux the upstream window is declared massively non-passive ($\pi=2.39$) purely because of a *transport* term that carries no dissipation and cancels one level up. Under §2.2's it is passive. [[plug-in-composition-theorems]] §3 makes `storage` the one certificate that survives an expert swap without re-earning, and W33 would **refuse a record at admission** whose measured passivity contradicts its declaration — so this is not academic: with §4.1's flux, W33 refuses a graph whose assembled seam is provably passive.

> **§4.2's equivalence — incrementally passive $\iff$ $\tfrac12(S_i+S_i^\top)\succeq0$ — is only meaningful once the flux convention is pinned, and [[probed-dtn-coupling]] pins it in two incompatible places.** The convention that makes the pairing a genuine power, and the only one under which the block statement is about dissipation rather than transport, is §2.2's.

## 2.3 Two controls, and what they localize

| control | $\mu$ (seam) | $\beta$ | $\kappa$ | blocks |
|---|---|---|---|---|
| zero state (Stokes limit) | $+3.202\times10^{-3}$ | $3.207\times10^{-3}$ | $35.64$ | $+1.601\times10^{-3}$ **each, identical** |
| uniform flow | $+6.185\times10^{-3}$ | $6.188\times10^{-3}$ | $19.26$ | $+3.357\times10^{-3}$ / $-9.163\times10^{-4}$ |

The zero-state control **pins the sign convention**: with $\Lambda_i\lambda=\nu\,\partial_n u$ taken along each domain's *outward* normal, the diffusive DtN is positive definite, which is what $\int_\Gamma\lambda\,\partial_n u=\int_\Omega\lvert\nabla u\rvert^2\ge0$ requires. Both blocks return the identical eigenvalue, as two mirror-image windows with no flow must.

The uniform-flow control **localizes the downstream block's near-singularity to the flow direction, not to the wake**: $\beta=1.81\times10^{-5}$, $\kappa=2260$, $\pi=9.16\times10^{-4}$ on a field with no structure at all, within $8\%$ of the developed-state values. **The downstream side of a seam has an almost rank-deficient DtN because its response to inflow data is nearly a pure translation** — and the assembled seam is $\kappa=19$ while its own block is $\kappa=2260$, two orders apart.

> **Any per-block conditioning gate would have condemned a seam that is in fact well conditioned.** $\beta$ is a property of the assembled operator and reporting it per block invites exactly the wrong refusal. [[probed-dtn-coupling]] §4.3 says "both are printable per port" without saying which one the verdict is taken on; it is the seam.

## 2.4 W30 — the operator drifts slowly, so the cached-$S$ economics survive

$\lVert S(t+K\Delta t)-S(t)\rVert_2$ at $K=10$, relative:

| `sx0` | `sx1` | `sy0` | `sy1` |
|---|---|---|---|
| $2.98\times10^{-3}$ | $3.53\times10^{-3}$ | $6.41\times10^{-4}$ | $2.37\times10^{-4}$ |

$0.02\%$–$0.35\%$ over ten macro-steps. §6's amortization argument — assemble once, refresh every $K$ — **is priced rather than assumed** for the first time, and the price is low: refreshing every $50$ steps holds the operator within about $1.5\%$.

---

# 3. W1 — $L$, fitted, three times

$L$ is a property of a **scheme**, not of a problem, so it was fitted for three maps. Paired rollouts from the developed state, a divergence-free streamfunction blob as $\Delta$, $\lVert e^n\rVert$ the relative $L_2$ gap between perturbed and unperturbed trajectories of the *same* scheme, least squares on $\log\lVert e^n\rVert$ against $n$.

| map | $L$ | std. err. | $\eta=L-1$ | rate $\ln L/\Delta t$ |
|---|---|---|---|---|
| monolith $n=255$ | $0.979649$ | $0.00041$ | $-0.0204$ | $-0.411$ |
| composed, one-pass lagged Dirichlet | $\mathbf{0.948441}$ | $0.0048$ | $\mathbf{-0.0516}$ | $-1.059$ |
| composed, probed-DtN | $0.974593$ | $0.00042$ | $-0.0254$ | $-0.514$ |

**Amplitude independence to six digits.** At $\Delta=10^{-3}$ and $\Delta=10^{-4}$ the fits agree to $L=0.948441$ vs $0.948443$. The fit is measuring the linearized amplification, not a nonlinear excursion.

**$L<1$ on all three, so the master bound's contractive branch applies and $T_{\text{pred}}$ is unbounded** for this configuration. This is the branch [[master-error-bound]] §6.1 could only reach structurally, through a passivity hypothesis nobody had discharged; here it is reached empirically, and §2's $\mu>0$ says the structural route would have reached the same place.

**The composed map is *more* contractive than the monolith** ($0.948$ against $0.980$). Decomposition is stabilizing here, and the mechanism is visible: a pinned Dirichlet ring is a sink for perturbations, so the seam layer damps what crosses it. That is the same property that makes the ring inconsistent for the *mean* flow (§5), and it is worth stating that the two are one mechanism seen twice.

## 3.1 The verdict against OP-2, stated as a verdict

[[open-problems-atlas-0.1]] OP-2 records a centreline deficit rising $9.3\%\to25.9\%$ between $t=2$ and $t=20$ at $\Delta t_{\text{macro}}=0.25$. As a per-macro-step amplification that is

$$L_{\text{OP-2}}=\Bigl(\tfrac{0.259}{0.093}\Bigr)^{1/72}=1.0143,\qquad \eta_{\text{OP-2}}=+1.43\times10^{-2}$$

**The fitted $\eta$ and OP-2's implied $\eta$ have opposite signs.** $-5.16\times10^{-2}$ against $+1.43\times10^{-2}$: this configuration contracts where OP-2 grows.

**This does not refute OP-2, and saying so is not hedging.** OP-2's number is from the *periodic-window frozen-checkpoint* configuration at $\Delta t=0.25$; this one is a Dirichlet-ring exact solver at $\Delta t=0.05$. Three things differ at once. What the measurement does establish is narrower and still useful: **$L>1$ is not a property of windowed decomposition as such**, so OP-2's monotone accumulation has to be charged to the expert, the periodicity, or the step size — and the fitted $L$ now exists as the instrument that will decide it, one variable at a time.

---

# 4. W3 — the three-way split, and $\gamma$ is even smaller than predicted

One macro-step from the developed state, against the $n=255$ monolith. Depth $0$ ([[plug-in-composition-theorems]] §1.4's tag). Relative $L_2$ over the assembled global field.

| term | value | share |
|---|---|---|
| total defect | $3.7071\times10^{-4}$ | — |
| $\tau$ — agent, exact ring supplied | $\mathbf{3.7014\times10^{-4}}$ | $99.8\%$ |
| $\sigma$ — transmission | $2.6995\times10^{-5}$ | $7.3\%$ |
| $\gamma$ — interface solve | $3.48\times10^{-15}$ | machine zero |

$\tau$ and $\sigma$ are norms of partly cancelling errors, so they need not sum to the total.

**The prediction $\gamma\lll\tau,\sigma$ is confirmed by eleven orders of magnitude.** The direct dense solve takes the interface residual from $2.64\times10^{-5}$ to $8.4\times10^{-20}$; $\gamma$ is arithmetic, exactly as [[master-error-bound]] §5 says a direct Schur solve makes it. **The framework's habit of reporting $\gamma$ and nothing else was reporting the term that does not matter, and now that is measured rather than argued.**

## 4.1 $\tau$ is not agent infidelity, and the attribution theorem predicted this

$\tau$ is $99.8\%$ of the defect while the agent **is** the reference solver, which is only possible if $\tau$ is carrying something that is not the agent. Three measurements identify it.

**It does not scale with $\Delta t$.** $\tau=3.7014\times10^{-4},\ 3.6630\times10^{-4},\ 3.5727\times10^{-4}$ at $\Delta t=0.05,\ 0.01,\ 0.002$ — flat across a $25\times$ range, so $\tau/\Delta t$ *diverges* as the step is refined. That is the signature of an **inconsistent** decomposition, not an inaccurate one.

**It does not live at the seams.** By distance from the nearest seam layer: $9.7\%$ within $4$ cells, $48.5\%$ at $5$–$32$ cells, $41.9\%$ beyond $32$. The defect is global.

**It is the per-window pressure projection, and the identity is exact.** `WindowNS` solves its pressure Poisson problem with homogeneous Neumann data, singular in the constant mode, so a solution exists only if the right-hand side integrates to zero — i.e. only if the ring's net flux balances. A subdomain of a through-flow has net flux through it:

| | `W00` | `W10` | `W01` | `W11` |
|---|---|---|---|---|
| net boundary flux | $+1.104\times10^{-3}$ | $-8.279\times10^{-4}$ | $+8.280\times10^{-4}$ | $-1.104\times10^{-3}$ |

The solver subtracts the mismatch uniformly and reports it, and the reported number *is* the residual divergence, to ten significant figures:

$$\texttt{tile.last\_mismatch}=2.7607\times10^{-4}\;=\;\texttt{tile.last\_div}=2.7607\times10^{-4}$$
$$\texttt{mono.last\_mismatch}=1.37\times10^{-20},\qquad \texttt{mono.last\_div}=7.99\times10^{-15}$$

**Eleven orders of magnitude between the monolith's divergence and a window's**, from one cause: incompressibility is a global constraint and a per-window projection under an unbalanced Dirichlet ring cannot honour it.

> **So $\tau$ here is a $\sigma$ from one level down, wearing a $\tau$ label — [[plug-in-composition-theorems]] §1.4's attribution theorem, observed.** Its **[AI Inference]** that OP-5's expert-versus-architecture share is depth-relative is no longer only plausible: the defect that dominates this split is manufactured by decomposing an elliptic solve, and it is charged to the agent purely because of where the harness cuts. **Every emitted defect needs its depth (W34), and this is the measurement that says so.**

## 4.2 The master bound, evaluated

With $L=0.948441$, $\tau=3.7014\times10^{-4}$, $\sigma=2.6995\times10^{-5}$, $\gamma\approx0$ and $e^0=0$, [[master-error-bound]] §7 predicts

$$\lVert e^N\rVert\;\le\;(\tau+\sigma)\,\frac{1-L^N}{1-L}\;\xrightarrow[N\to\infty]{}\;\frac{3.971\times10^{-4}}{0.05156}\;=\;7.70\times10^{-3}$$

Rolled out and compared against the monolith at every step:

| $N$ | measured | bound | ratio |
|---|---|---|---|
| $1$ | $3.707\times10^{-4}$ | $3.971\times10^{-4}$ | $0.93$ |
| $10$ | $4.534\times10^{-4}$ | $3.166\times10^{-3}$ | $0.14$ |
| $40$ | $5.919\times10^{-4}$ | $6.776\times10^{-3}$ | $0.087$ |
| $120$ | $5.413\times10^{-4}$ | $7.689\times10^{-3}$ | $0.070$ |

**The bound holds at every step, is tight at $N=1$, and is loose by $14\times$ asymptotically.** That is the right shape: at one step the bound *is* $\tau+\sigma$, and over a rollout it assumes worst-case alignment of per-step defects while real ones partially cancel.

**And the measured error saturates.** $5.4\times10^{-4}$ from about $N=40$ ($t=2$) through $N=120$ ($t=6$) — a plateau, not a drift. **This is the first time this vault has quoted an error bound with every constant measured rather than named**, and the first coupled configuration whose error is observed to settle.

---

# 5. The finding that costs the most: the interface condition is a steady condition

> **Acted on in §8.2.** This is now **R2b**, a compiler rule: `capability.TimeDiscretization` is a declared field and probed-DtN is gated on it. R2 lifts the rung from what the expert *accepts*; R2b gates it on what the expert *is*. Recorded as an `admit` rather than a refusal, because the compiler corrects the rung and nothing is left silently wrong.

Everything in §2 says the probe is healthy. The scheme built on it is not, and the two facts are independent.

**Solving $\sum_i\Lambda_i\lambda=\chi$ exactly makes one composed macro-step $8.9\times$ worse** than not solving it at all: $3.287\times10^{-3}$ against the one-pass $3.707\times10^{-4}$. The interface residual falls by $500\times$ while the answer degrades — which is precisely [[schwarz-iteration-atlas-0.1]]'s warning about removing the only visible symptom, arriving from a direction that page did not anticipate.

**The decisive measurement.** Substitute the *true* trace — what the monolith actually produces at the seam after one macro-step — into the interface residual:

| | $\lVert r\rVert$ | trace move $\lVert\cdot\rVert_\infty$ |
|---|---|---|
| lagged trace (do nothing) | $2.6421\times10^{-5}$ | $0$ |
| **true trace** | $\mathbf{2.6520\times10^{-5}}$ | $7.249\times10^{-5}$ |
| probed trace (solve exactly) | $5.234\times10^{-8}$ | $2.968\times10^{-3}$ |

**The true solution does not satisfy the condition.** Its residual is *marginally larger* than doing nothing. So the condition is not an approximation of a true statement about $\lambda$ — it is a different statement, and driving it to zero drives $\lambda$ to somewhere the solution is not, $41\times$ past it.

**Refining $\Delta t$ makes it worse, which names the mechanism.** Overshoot $=\lVert\lambda_{\text{probed}}-\lambda_{\text{lag}}\rVert_\infty/\lVert\lambda_{\text{true}}-\lambda_{\text{lag}}\rVert_\infty$:

| $\Delta t$ | true move | probed move | overshoot | $\beta$ |
|---|---|---|---|---|
| $0.05$ | $7.249\times10^{-5}$ | $2.968\times10^{-3}$ | $41\times$ | $5.29\times10^{-3}$ |
| $0.01$ | $1.375\times10^{-5}$ | $1.031\times10^{-3}$ | $75\times$ | $7.94\times10^{-3}$ |
| $0.002$ | $3.706\times10^{-6}$ | $5.546\times10^{-4}$ | $\mathbf{150\times}$ | $1.17\times10^{-2}$ |

As $\Delta t\to0$ the true trace move goes to zero — the seam barely changes in a short step — while the imposed correction does not follow it down. **A condition whose solution is $\Delta t$-independent is a steady-state condition**, and imposing it inside a step that is not steady over-determines the trace by an amount that grows without limit as the step is refined.

> **[[probed-dtn-coupling]] §2.1 writes $\mathcal E_i$ as the solution operator of a boundary-value problem. Every expert this vault can actually probe is the one-step map of an initial–boundary-value problem, and the page never states the interface condition for that case.** Flux balance is what the trace satisfies at a fixed point; it is not what it satisfies after one explicit macro-step, where the interface's own unsteady term is a first-order part of its equation and $\Lambda_i$ contains none of it.

**Why the damage is so large is the bound's own arithmetic, and that is the consoling part.** $\delta=-S^{-1}r$, so a residual $r$ that is a modelling artifact rather than a physical imbalance is amplified by $1/\beta\approx190$. $\beta$ sitting in the denominator of $\sigma$ is exactly what [[master-error-bound]] §4 says; what §4 does not say is that the numerator has to be a *real* imbalance for the bound to be about anything. **The construction's own diagnostics were all green while it was doing this**, which is the strongest argument on this page for measuring the scheme and not only the operator.

## 5.1 The remedy this points at is already two rows on the worklist

The condition that is right for a one-step map is a statement about the **time-integrated** flux across the macro-step, not the pointwise flux at its end — which is [[general-coupling-scheme]]'s **R9**, so far written only for the multirate case ([[gap-worklist]] W7). **The measurement says R9 is needed at a single rate too**, and for a reason R9's multirate framing does not cover. Carrying $\lambda$ as a function of time across the window rather than a value at its end is **W17** (waveform relaxation, $W>1$), which [[gap-worklist]] parks in Tier 4 as deferred. **Both are promoted by this measurement**, and W17 in particular stops being a refinement and becomes the correct formulation.

> **Retracted 2026-08-28 by §9.3, which built it.** The time-integrated flux condition fails *both* of §5's tests, identically to the pointwise one: the reference's own trace raises its residual (ratio $1.002$ exposed, $1.028$ embedded, at three $\Delta t$), and solving it degrades the composed step by $2.00\times$ against pointwise's $1.59\times$. The reason is an identity rather than a subtlety -- at a shared-layer seam $F_A+F_B=-\nu h\,\partial_{nn}w$ **exactly**, a discrete second derivative and not a jump -- so the defect is *spatial* and integrating in time cannot touch it. R9 remains right for multirate, which is what it is about. **The construction that is valid for an explicit one-step map is the overlapping halo update itself**, and its bound is §9.2's $\Pi$.

**And this build violates §2.1's other premise as well.** §2.1 rests on the elliptic part being solved globally — "a whole-domain DCT Poisson solve" — but `WindowNS` projects **per window**, which is what §4.1 measured as the $\tau$-labelled defect. So the two halves of §2.1 both fail here, in different ways, and the second failure was invisible until the first was investigated.

---

# 6. What this changes, itemized

| # | Page | Change |
|---|---|---|
| 1 | [[probed-dtn-coupling]] §2.1 | The rank-one null space and "the elliptic part stays global" cannot both hold. Measured null dim $0$ on four seams |
| 2 | [[probed-dtn-coupling]] §2.2 / §4.1 | The two proposed fluxes agree to $1.1\times10^{-9}$ on the assembled seam and disagree by a factor of $10^{3}$ per block. §4.1's refinement is a no-op where it is claimed and a hazard where it is not |
| 3 | [[probed-dtn-coupling]] §4.2 | The per-block passivity verdict is convention-dependent and flips. The seam-level verdict does not. W33 must certify on the seam |
| 4 | [[probed-dtn-coupling]] §2.1 | **The interface condition is a steady condition.** The true trace does not satisfy it; solving it exactly overshoots by $41\times$, growing to $150\times$ as $\Delta t\to0$ |
| 5 | [[interface-transfer-theory]] §7 | The null-count table's fluid–fluid entry of $1$ is not what a pressure-excluded $\Lambda$ produces |
| 6 | [[master-error-bound]] | Evaluated, with all constants measured, for the first time. Holds; tight at $N=1$; $14\times$ loose asymptotically |
| 7 | [[open-problems-atlas-0.1]] OP-2 | $L<1$ measured here, against OP-2's implied $L=1.0143$. $L>1$ is not a property of decomposition as such |
| 8 | [[plug-in-composition-theorems]] §1.4 | The attribution theorem observed: the dominant $\tau$ is a decomposed elliptic solve, i.e. a $\sigma$ one level down |
| 9 | [[gap-worklist]] W7, W17 | Promoted. R9 is needed at a single rate; waveform relaxation is the correct formulation, not a refinement |
| 10 | `atlas/probe.py` | `SeamOperator.as_dict` dropped `passivity_lambda_min`, so the assembled operator emitted only the clipped $\pi$ — against §4.2's explicit requirement. Fixed |

---

# 7. What stayed open after the first run

> **Read §8 before this list.** Three of the items below have since been closed by the remediation — $\lVert\mathcal A\rVert$ (a theorem, not a measurement), $\Xi$ (measured, $0.19$–$0.53$), and $\lVert\Lambda-\tilde\Lambda\rVert$ (measured, and it made the $\sigma$ factorization's failure visible). The list is kept as the honest statement of where the first run ended.

- **$\lVert\mathcal A\rVert$ is not computed.** W2's second half is untouched. The partition-of-unity identity residual is machine zero by construction on a shared-layer tiling with $\chi_i=1/(\text{owner count})$, which is W28's first field and **not** its second. `AssemblyCertificate` stays a named hole; `condition` stays empty; E6 stays `unchecked`.
- **$\Xi$ was not measured.** The composability index needs a $\Lambda^{\text{ref}}$ on the same basis, which needs a one-sided reference operator this configuration does not have. §4.5's positive control — $\Xi=0$ for the frozen checkpoint — is still unrun.
- **$\lVert\Lambda-\tilde\Lambda\rVert$ was not measured**, so $\sigma$ in §4 is the *observed* field defect, not the bound's product form. The two have not been reconciled.
- **`differentiable` is `NONE`, tested.** `WindowNS.step_batch` returns numpy at the class boundary by deliberate design, so `torch.func.jvp` raises `Cannot access data pointer of Tensor that doesn't have storage` whatever the backend. §5.2's "second unspent use of the differentiability claim" **stays unspent**, and the blocker is one line of a port made for a good unrelated reason.
- **One configuration, one state, one expert.** Every number here is local to a developed wake at $t=5$, $\Delta t=0.05$, $\nu=1/255$. [[probed-dtn-coupling]] §7 says this out loud about probes and it is equally true of $L$.
- **The cross-point was declared, not treated.** L2's refusal is correct and W6 is unchanged by anything here.
- **W33 was not run.** The conformance suite rides on W2's probe and is now cheap, but §2.2's convention question has to be settled before a `storage` certificate means anything.

---

---

# 8. The remediation — what the fixes measured

**Everything in §1–§7 was diagnosis. This section is what happened when the diagnoses were acted on**, and it revises §2.1, §2.2 and §5 rather than only extending them. Two things were wrong and both are now rules in `atlas/` rather than notes beside it.

## 8.1 Three hypotheses, tested in the wrong order first — and that is the lesson

The §4.1 diagnosis said $\tau$ was the per-window pressure projection. Acting on it produced **three candidate fixes, each of which was measured alone and each of which looked like a failure**:

| tried alone | $\tau$ | verdict at the time |
|---|---|---|
| widen the halo, 1 to 31 cells | $3.70\times10^{-4}\to2.66\times10^{-4}$ | 28% for $31\times$ the overlap — "not the domain of dependence" |
| a partition of unity that vanishes at artificial edges | $3.70\times10^{-4}\to3.70\times10^{-4}$ | nothing — "not the assembly" |
| a global projection, applied after assembly | $3.70\times10^{-4}\to3.36\times10^{-4}$ | 9% — "not the pressure either" |

**All three are necessary and none is sufficient, so testing them one at a time refuted the correct hypothesis three times.** Together:

| configuration | $\tau$ |
|---|---|
| as-built: embedded projection, one exchange per macro-step, $\chi=1/(\text{owner count})$ | $2.935\times10^{-4}$ |
| **split-step: projection in the composition layer, exchange every sub-step, ramped PoU, halo 21** | $\mathbf{1.335\times10^{-6}}$ |

**$220\times$.** A halo sweep at the working configuration reaches $1.8\times10^{-7}$ at halo $129$, so the remaining defect is still halo-limited rather than at a floor.

> **The methodological point is worth more than the number.** Three correct ingredients, each individually indistinguishable from noise, and a control — four windows each covering the *whole* domain, which returns exactly $0.0$ — that should have been run before any of them. A composition fix is not separable into independent tests when the ingredients are a boundary condition, the region it contaminates, and the weight that region is given.

## 8.2 The two rules that came out of it

**R10 — an embedded elliptic sub-solve is not decomposable.** `capability.EllipticSubsolve` is a new declared field, and L2 **refuses** a decomposition of an agent that embeds one. The justification is the measurement: a projection method's pressure Poisson solve is global over whatever domain it runs on, so cutting the domain cuts the operator, and the resulting error is elliptic — flat in distance from the cut, flat in $\Delta t$, and untouched by any halo or partition of unity. It is the silent-wrongness class exactly: no exception, no failed gate, and a number that reads as a bad expert.

**R2b — probed-DtN needs an implicit macro-step.** `capability.TimeDiscretization` is the second new field. $\sum_i\Lambda_i\lambda=\chi$ is the interface condition of a *boundary-value problem*; discretize implicitly and each macro-step is one, discretize explicitly and there is none. R2 lifts the rung from what the expert *accepts*; R2b gates it on what the expert *is*. It is recorded as an **admit**, not a refusal — the compiler corrected the rung, so nothing is silently wrong, which is what the three-verdict split rule turns on.

**And a halo rule**, from `required_halo() = stencil_radius * substeps_per_macro_step`: the overlap must outrun the agent's own domain of dependence, or the blended region is contaminated by each window's artificial boundary. The working configuration declares $2\times10=20$ and carries $21$.

## 8.3 W2, re-measured — every diagnostic improves by one to three orders

Same seams, same basis, same probe. The only change is which agent.

| | $\beta$ | $\kappa$ | $\mu$ | $\pi$ | cut $Q$ | $\lVert S-S^\top\rVert/\lVert S\rVert$ |
|---|---|---|---|---|---|---|
| as-built (embedded) | $5.05\times10^{-3}$ | $21.73$ | $+5.04\times10^{-3}$ | $0$ | $85.6$ | $0.153$ |
| **split-step (exposed)** | $\mathbf{0.3486}$ | $\mathbf{1.196}$ | $+0.3486$ | $0$ | $0.15$ | $\mathbf{0.002}$ |

$\beta$ improves $69\times$, $\kappa$ $18\times$, the cut score $570\times$, and **the asymmetry $76\times$ — the operator becomes essentially self-adjoint**. That last one is the physically legible part: the pressure coupling was what made the interface operator non-normal, and with the elliptic part where it belongs the advection–diffusion DtN is nearly symmetric.

All four seams, split-step: $\beta\in[0.277,0.376]$, $\kappa\in[1.10,1.36]$, $\mu>0$ and $\pi=0$ throughout. **An interface problem with $\kappa\approx1.2$ is not a hard problem** — and the first run's $\kappa\approx20$ was measuring the decomposed pressure solve, not the seam.

**$\Xi$ is measured, and it is the first nonzero one in this vault.** Taking the exposed agent as the reference, the embedded agent scores

$$\Xi \;=\; 0.19\text{–}0.53,\qquad \frac{\lVert\Lambda^{\text{emb}}-\Lambda^{\text{exp}}\rVert}{\lVert\Lambda^{\text{exp}}\rVert}\;=\;0.78\text{–}0.92$$

so the as-built agent reproduces between a fifth and a half of the boundary response of the same agent with its elliptic part removed. **R10 in operator norm**, obtained from two assemblies and no rollout, exactly as §4.6 says such a comparison should be. It is *not* §4.5's positive control — that needs a frozen periodic checkpoint and remains unrun.

**W30 drift falls tenfold** with the corrected agent: $4.3\times10^{-5}$ to $3.7\times10^{-4}$ relative over ten macro-steps.

## 8.4 §2.1's null space, resolved rather than merely contradicted

The first run measured $\dim\ker=0$ against a declared $1$ and concluded the declaration was wrong. **The remediation says *why*, and it is not that [[interface-transfer-theory]] §7 is wrong:**

> **$n_0(\Gamma)$ is not a property of the seam alone. It depends on where the elliptic solve lives.** With incompressibility *inside* $\Lambda_i$ the trace is constrained to zero net flux and $n_0=1$; with the elliptic part in the composition layer — which is what [[probed-dtn-coupling]] §2.1 prescribes — $\Lambda_i$ is the pure advection–diffusion DtN and $n_0=0$.

Both halves of §2.1 are now consistent, and the field that decides which applies is `elliptic_subsolve`. Measured $0$ on four seams under both agents, with $\kappa$ between $1.10$ and $1.36$ and no spectral gap: not a near-null direction anywhere. The case study now declares `expected_null_dim = 0`, and the four decertifications the wrong declaration was producing are gone.

## 8.5 §2.2's correction, which the first run got half right

§2.2 above reported the two flux conventions agreeing to $1.11\times10^{-9}$ on the assembled seam. **That is true only for a shared-layer seam**, where both sides read the *same cell* so the advective term cancels identically. At the working halo of 21 cells the two sides read cells 20 apart and the cancellation is only approximate: the assembled operators differ by $3.9\times10^{-2}$ relative.

**The block-level hazard is unchanged and it is what settles W47.** At the same seam, $\pi=0$ under §2.2's flux and $\pi=2.01$ under §4.1's — a transport term reported as a passivity defect. **§2.2's flux is normative; §4.1's is an approximate seam-level equivalent and a block-level trap.**

**And W48 is sharper than the first run made it.** Under the corrected agent the seam is $\kappa=1.196$ while its own downstream block is $\kappa=7061$ — now **5900$\times$ apart**, up from 130. The downstream block's near-singularity survives the fix, so it is genuinely a property of being the outflow side and not an artifact of the pressure. **The verdict belongs to the seam, and a per-block gate would refuse a seam whose conditioning is essentially perfect.**

## 8.6 W1, re-measured — and this is the strongest single result

| map | $L$ | std. err. |
|---|---|---|
| monolith $n=255$ | $0.979650$ | $0.00041$ |
| as-built composition | $0.972776$ | $0.00043$ |
| **split-step composition** | $\mathbf{0.979644}$ | $0.00041$ |

**The fixed composition's stability constant matches the monolith it approximates to five decimal places** — $0.979644$ against $0.979650$, well inside a standard error of $4\times10^{-4}$. The as-built one is off by $7\times10^{-3}$, seventeen standard errors.

That is the cleanest available statement that the composition is now *faithful* rather than merely accurate: it is not just producing a similar answer, it is the same dynamical map to the precision the fit can resolve. Amplitude independence to six digits holds as before.

**The earlier $L=0.948$ was a different scheme**, not a different answer: the one-cell-shared tiling with $\chi=1/(\text{owner count})$, whose frozen seam layer over-damps perturbations. Reporting it as "the composed $L$" would have been reporting a property of a bad assembly as a property of decomposition.

## 8.7 W3, re-measured, and the bound re-evaluated

| | total | $\tau$ | $\sigma$ | $\gamma$ |
|---|---|---|---|---|
| as-built | $2.948\times10^{-4}$ | $2.935\times10^{-4}$ | $2.253\times10^{-5}$ | $0$ |
| **split-step** | $\mathbf{1.360\times10^{-6}}$ | $1.335\times10^{-6}$ | $3.733\times10^{-8}$ | $0$ |

$\gamma$ is now identically zero rather than merely negligible: under the `dirichlet` rung with a lagged trace **there is no interface system to solve**. The first run's $\gamma=3.5\times10^{-15}$ was the refused probed-DtN rung's direct solve, and it remains the best evidence for the prediction $\gamma\lll\tau,\sigma$.

The bound holds for both schemes, and tightly at one step:

| | $N=1$ | $N=120$ | asymptotic bound | measured plateau |
|---|---|---|---|---|
| as-built | ratio $0.93$ | ratio $0.21$ | $1.16\times10^{-2}$ | $2.39\times10^{-3}$ |
| split-step | ratio $\mathbf{0.99}$ | ratio $0.20$ | $6.74\times10^{-5}$ | $\mathbf{1.23\times10^{-5}}$ |

**The composed error over 120 macro-steps is $194\times$ smaller**, and the bound is tight to 1% at $N=1$.

**The assembly certificate is emitted in full.** PoU identity residual $1.11\times10^{-16}$; blend defect $\alpha=5.7\times10^{-15}$, i.e. **the blend is not worse than the worst thing it blends**, which is the property partition-of-unity theory has conditions for and this framework had never checked. And $\lVert\mathcal A\rVert$:

> **Superseded by §9.1, and this is a correction rather than an extension.** The $\alpha=5.7\times10^{-15}$ was measured on inputs that make it zero by construction -- the reference cut into pieces and blended back, which the partition-of-unity identity zeroes for **any** partition, a signed one included. It is the `norm_A` failure repeated one field over. The sentence in bold above was true and unsupported; §9.1 measures it properly ($\alpha=-1.041\times10^{-2}$ against the cellwise max, on real local solves) and shows that **no** inequality on $\alpha$ can be the assembly condition. **$\lVert\mathcal A\rVert=1$ holds for any *convex* partition of unity** -- the word was missing, and a signed one on this same tiling measures $1.7038$.

> $\lVert\mathcal A\rVert=\max_j\sqrt{\sum_i\chi_{i,j}^2}=1$ **for any partition of unity**, attained at every single-owner cell. It is a theorem, not a measurement. The previous implementation computed $\lVert\sum_i R_i^\top\chi_iR_i\rVert$, which is $\lVert I\rVert=1$ whenever the identity holds — **a field that could only ever report a number when the thing it was checking had already failed.** Fixed, and the master bound's $\lVert\mathcal A\rVert$ factor is now known rather than unmeasured.

## 8.8 The $\sigma$ bound's product form is vacuous for a halo scheme

The one measurement that came out badly, and it opens a new gap rather than closing one. [[master-error-bound]] §4 gives

$$\sigma\;\le\;\frac{C_\mu}{\beta}\,\lVert\Lambda-\tilde\Lambda\rVert\,\lVert\lambda^\star\rVert$$

Every factor is now measurable. The scheme uses **no interface operator at all** — the rung is `dirichlet` with a lagged trace — so $\tilde\Lambda=0$ and $\lVert\Lambda-\tilde\Lambda\rVert=\lVert\Lambda\rVert=0.417$. With $\beta=0.349$ and $\lVert\lambda^\star\rVert=1.440$ the bound at $C_\mu=1$ is $\mathbf{1.72}$, against a measured $\sigma$ of $3.73\times10^{-8}$:

$$C_\mu^{\text{implied}}\;=\;2.2\times10^{-8}\qquad\text{(as-built: }1.9\times10^{-7}\text{)}$$

**The product form overestimates by $4.6\times10^{7}$.** The reason is structural: the bound assumes the transmission error enters through an *interface solve*, where an operator mismatch is amplified by $1/\beta$. **A halo scheme has no interface solve** — its accuracy comes from the overlap width, and the bound has no term for overlap. Applied to one, it is not conservative, it is uninformative.

$C_\mu$ is therefore **left unmeasured** on the record. Back-fitting it from the measured $\sigma$ would make the bound tautological, and a constant of $10^{-8}$ is a diagnosis of the factorization rather than a value for it. This is [[gap-worklist]] **W49**.

> **Closed 2026-08-28 by §9.2, and the refusal above is the reason it could be.** §4 is now scoped to the *substructuring* branch and [[master-error-bound]] §4.1 carries the *overlapping* one: $\sigma\le C_\mu\,\Pi\,\lVert\delta\lambda\rVert$, where $\Pi$ is the weight the assembly gives to cells a stale artificial-boundary datum can have reached. Over **16 configurations** the implied $C_\mu$ stays in $[0.228,1.178]$ while $\sigma$ moves by $4.3\times10^{4}$, so $C_\mu=1.2$ is quoted as *measured with a stated scope* rather than back-fitted -- which is a different act from the one refused here, and the difference is that this factorization keeps one constant and §4's needs $2.2\times10^{-8}$.

## 8.9 Where the compile lands

| graph | verdict | refusals | stamp |
|---|---|---|---|
| `as-built` | **`refuse`** | 1 — L2/R10 | E1–E5 hold, E6 unchecked, E7 holds |
| `split-step` | **`admit-uncertified`**, runnable | **0** | E1–E5 hold, E6 unchecked, E7 holds |

**`admit` is now blocked by exactly one thing**, and it is a named hole rather than a missing measurement: **`AssemblyCertificate`'s `condition` — nothing anywhere states that a blend must be at least as accurate as the local solves it blends**, so E6 cannot be stamped `holds` and the run cannot be certified. The blend defect is measured ($5.7\times10^{-15}$, negative) and unconstrained. The only unmeasured constant left is $C_\mu$, for the reason in §8.8.

That is a considerably sharper end state than "no graph reaches `admit`": one hole, named, with the measurement it would constrain already being emitted.

> **Wrong, and §9.6 has the count.** The `split-step` graph carried **three** decertifications, not one: `L2/G5/W16`, `L5/eps_tol` and `L6/AssemblyCertificate`. The middle one was a live bug of the W45 class -- it asserted that $\tau$ and $\sigma$ were unmeasured on a graph whose record declares both. Two are now closed and the true end state is §9.7: **every hole filled, every constant measured, and one unearned [AI Inference] left**.



---
---

# 9. 2026-08-28 — the assembly condition, the $\sigma$ bound, and what varying two knobs broke

**Status of §1–§8 after this session:** every number in them reproduced from scratch before anything was touched — the compile, `L=0.979644`, $\kappa=1.196$, $\tau=1.3353\times10^{-6}$, $\sigma=3.7334\times10^{-8}$, the PoU residual, `norm_A`, the 141 tests. §8's conclusions stand except where a box below says otherwise, and **three of them are corrected rather than extended**: §8.7's blend defect was measuring nothing, §8.9's "blocked by exactly one thing" was wrong, and §5.1's promotion of W7/W17 is retracted by the construction being built and measured.

---

## 9.1 `AssemblyCertificate.condition`, closed — and the condition is on $\chi$, not on $\alpha$

The slot asked for *"the inequality $\alpha$ must satisfy"*, with `must_satisfy` = *"checkable without the exact solution, or the certificate is a diagnostic rather than a condition"*. **The answer is that no inequality on $\alpha$ can be the condition, and the measurement is what says so.**

**The identity that settles it.** For any weights with $\sum_i\chi_{ij}=1$, at every cell,

$$\bigl|A(u)-u^\star\bigr|^2 \;=\; \underbrace{\sum_i\chi_i\bigl|u_i-u^\star\bigr|^2}_{\text{weighted mean of the local errors}} \;-\; \underbrace{V_\chi}_{\textstyle V_\chi=\sum_i\chi_i u_i^2-\bigl(\sum_i\chi_i u_i\bigr)^2}$$

$V_\chi$ is the $\chi$-weighted **variance of the local values**. It needs no exact solution. It is non-negative for all data **exactly when** $\chi\ge0$. So

$$\boxed{\;\textbf{L6/C1:}\quad \chi_{ij}\ge0\ \ \wedge\ \ \textstyle\sum_i\chi_{ij}=1 \quad\Longrightarrow\quad \bigl|A(u)-u^\star\bigr|_j\le\max_i\bigl|u_i-u^\star\bigr|_j\ \text{at every cell}\;}$$

and therefore in **every** $\ell^p$ norm. That is the missing statement, it is a theorem rather than an assumption, and its only hypothesis is read off the declaration.

**Measured on the four-window tiling**, one sub-step of the exposed agent per window against a no-projection monolith of the same class — the same map, the same state, no projection and no assembly in between, so nothing but L6 is in the comparison:

| | identity residual | $\lVert\mathcal A\rVert$ | $\min\chi$ | $\min_j V_\chi$ | hull escape | bias–variance gap |
|---|---|---|---|---|---|---|
| flat $\chi=1/(\text{owner count})$ | $0$ | $1.0000$ | $0$ | $-2.2\times10^{-16}$ | $0$ | $6.1\times10^{-16}$ |
| ramp 1 (hard switch) | $2.2\times10^{-16}$ | $1.0000$ | $0$ | $-2.2\times10^{-16}$ | $0$ | $6.5\times10^{-16}$ |
| ramp 4 | $2.2\times10^{-16}$ | $1.0000$ | $0$ | $-4.4\times10^{-16}$ | $2.2\times10^{-16}$ | $8.0\times10^{-16}$ |
| **ramp 8 (the working one)** | $1.1\times10^{-16}$ | $1.0000$ | $0$ | $-8.9\times10^{-16}$ | $2.2\times10^{-16}$ | $9.4\times10^{-16}$ |
| ramp 21 (full overlap) | $2.2\times10^{-16}$ | $1.0000$ | $0$ | $-1.1\times10^{-15}$ | $4.4\times10^{-16}$ | $1.1\times10^{-15}$ |
| **signed (extrapolatory)** | $2.2\times10^{-16}$ | $\mathbf{1.7038}$ | $\mathbf{-0.598}$ | $\mathbf{-3.4\times10^{-8}}$ | $\mathbf{1.13\times10^{-4}}$ | $1.1\times10^{-15}$ |

**The identity closes to $10^{-15}$ for every partition, signed included** — it is an identity, not an approximation, and that is what lets the condition be about $\chi$ alone.

### The measurement that forces the condition to be on $\chi$

$\alpha$ was the emitted field, in two candidate definitions, and **neither discriminates**:

| partition | $\alpha$ (max-of-norms, the old definition) | $\alpha$ (cellwise max, what convexity bounds) | hull escape | $\tau$, one composed macro-step |
|---|---|---|---|---|
| ramp 8 | $-6.84\times10^{-3}$ | $-1.04\times10^{-2}$ | $2\times10^{-16}$ | $1.34\times10^{-6}$ |
| **signed** | $-6.06\times10^{-3}$ | $-9.63\times10^{-3}$ | $\mathbf{1.13\times10^{-4}}$ | $\mathbf{2.21\times10^{-4}}$ |

**Both $\alpha$s read as *passing* on the partition that costs $166\times$**, and the reason is structural rather than a bad choice of norm: $\alpha$ compares the blend against the *local solves*, and when those are individually bad and largely cancel, a blend can be far better than any of them and still far worse than the convex blend of the same data. `tests/test_tier9_assembly_condition.py::test_NEITHER_blend_defect_detects_a_signed_partition` holds the counterexample in three lines of algebra.

> **§8.7's `blend_defect = 5.7e-15` was measuring nothing, and it is the `norm_A` failure repeated one field over.** The driver passed `tiling.cut(u_ref)` as the local solves and `u_ref` as the reference — it cut the reference into pieces and blended them back, which the partition-of-unity identity makes **exactly zero for any partition**. Measured across the six partitions above: $0$ to $6\times10^{-15}$, including for the hard switch and the signed one. The sentence *"the blend is not worse than the worst thing it blends"* was true and unsupported. Fixed: the driver now supplies real local solves and reports $\alpha=-1.041\times10^{-2}$, $V_\chi^{\min}=-8.9\times10^{-16}$, hull escape $2.2\times10^{-16}$.

### What it costs to get this wrong, measured end to end

One composed macro-step, split-step scheme, the same graph, only $\chi$ changed:

| partition | $\Theta$ | $\tau$ | total | $\sigma$ | after 20 steps |
|---|---|---|---|---|---|
| flat $1/(\text{owner count})$ | $5.80\times10^{-3}$ | $5.01\times10^{-5}$ | $1.66\times10^{-4}$ | $1.20\times10^{-4}$ | $1.27\times10^{-3}$ |
| ramp 1 | $5.57\times10^{-3}$ | $3.32\times10^{-5}$ | $5.99\times10^{-5}$ | $2.75\times10^{-5}$ | $3.86\times10^{-4}$ |
| ramp 4 | $3.60\times10^{-3}$ | $4.88\times10^{-6}$ | $5.17\times10^{-6}$ | $3.24\times10^{-7}$ | $3.27\times10^{-5}$ |
| **ramp 8** | $1.98\times10^{-3}$ | $1.34\times10^{-6}$ | $1.36\times10^{-6}$ | $3.73\times10^{-8}$ | $8.62\times10^{-6}$ |
| **ramp 21** | $2.19\times10^{-3}$ | $\mathbf{2.46\times10^{-7}}$ | $\mathbf{2.49\times10^{-7}}$ | $6.26\times10^{-9}$ | $\mathbf{1.58\times10^{-6}}$ |
| signed | $1.40\times10^{-2}$ | $2.21\times10^{-4}$ | $5.29\times10^{-5}$ | $1.996\times10^{-4}$ | $1.76\times10^{-4}$ |

**The signed partition is the worst on $\tau$ of all six**, worse even than the flat one, and its damage is almost entirely in $\sigma$ — a blend that leaves the hull is a transmission failure, not an agent one.

**And ramp 21 is $5.4\times$ better than the working ramp 8.** That is a real result and it is *not* acted on here: `DEFAULT_TILING` still carries ramp 8, because every constant on this page — $L$, $\tau$, $\sigma$, $\Pi$, $C_\mu$ — is measured at ramp 8, and changing the default would make all of them stale in one commit. Opened as **W53**.

### The rule

**R11 — a partition of unity must be convex**, and it **refuses**. It is the silent-wrongness class exactly: the identity check passes (residual $2\times10^{-16}$), both blend defects read as passing, and $\lVert\mathcal A\rVert$ is the only existing field that notices, at $1.7038$ instead of $1$.

> **A correction to §8.7 and to `assembly.py`'s own docstring:** $\lVert\mathcal A\rVert=1$ is a theorem for any **convex** partition of unity. The word was missing, and a signed partition on this same tiling measures $1.7038$. Given $\sum_i\chi_i=1$, $\chi\ge0$ forces $\chi\le1$ hence $\sum_i\chi_i^2\le1$; drop the sign condition and it is unbounded. So $\lVert\mathcal A\rVert>1$ is a *sufficient* detector of non-convexity and $\chi_{\min}<0$ is the necessary and sufficient one.

**E6 now stamps `holds`** for the `split-step` graph. The certificate carries $\chi_{\min}$, the identity residual, $\lVert\mathcal A\rVert$, $V_\chi^{\min}$, the hull escape and $\Pi$, and it is issued **at compile time from the declaration** — no run, no reference — which is what the slot's `must_satisfy` demanded.

---

## 9.2 W49 — the $\sigma$ bound for the halo branch, and the first measured $C_\mu$

[[master-error-bound]] §4 routes the transmission error through an **interface solve**, and the $1/\beta$ in it is that solve's amplifier. §8.8 measured that applied to a halo scheme it overestimates by $4.6\times10^{7}$. **§4 is now scoped to the substructuring branch and §4.1 carries the overlapping one.**

What carries the transmission error in a halo scheme is the **stale artificial-boundary datum**, and two declared things stand between it and the assembled field: how far it can reach in one exchange ($d_i=\rho_i s_i$, the halo rule's own quantity) and how much weight the assembly gives the cells it reached ($\chi_i$). Composing them,

$$\Pi \;=\; \max_j\sum_i\chi_{ij}\,\mathbf 1\!\left[b_i(j)\le d_i\right],\qquad \sigma\;\le\;C_\mu\,\Pi\,\lVert\delta\lambda\rVert$$

**Eleven configurations, varying the halo from 1 to 61 cells and the partition of unity across five shapes:**

| configuration | $\Pi$ | $\lVert\delta\lambda\rVert$ | $\sigma$ | bound at $C_\mu=1.2$ | ratio |
|---|---|---|---|---|---|
| halo 1, ramp 8 | $1.000$ | $3.343\times10^{-4}$ | $2.683\times10^{-4}$ | $4.012\times10^{-4}$ | $1.50$ |
| halo 11, ramp 8 | $9.541\times10^{-2}$ | $1.040\times10^{-6}$ | $3.612\times10^{-8}$ | $1.191\times10^{-7}$ | $3.30$ |
| halo 21, ramp 8 | $9.541\times10^{-2}$ | $1.099\times10^{-6}$ | $3.733\times10^{-8}$ | $1.258\times10^{-7}$ | $3.37$ |
| halo 31, ramp 8 | $9.541\times10^{-2}$ | $1.190\times10^{-6}$ | $3.940\times10^{-8}$ | $1.363\times10^{-7}$ | $3.46$ |
| halo 41, ramp 8 | $9.541\times10^{-2}$ | $1.363\times10^{-6}$ | $4.223\times10^{-8}$ | $1.561\times10^{-7}$ | $3.70$ |
| halo 61, ramp 8 | $9.541\times10^{-2}$ | $1.797\times10^{-6}$ | $5.097\times10^{-8}$ | $2.057\times10^{-7}$ | $4.04$ |
| halo 21, flat | $0.750$ | $1.354\times10^{-4}$ | $1.196\times10^{-4}$ | $1.219\times10^{-4}$ | $\mathbf{1.02}$ |
| halo 21, ramp 1 | $0.750$ | $4.742\times10^{-5}$ | $2.753\times10^{-5}$ | $4.268\times10^{-5}$ | $1.55$ |
| halo 21, ramp 4 | $0.2967$ | $3.601\times10^{-6}$ | $3.244\times10^{-7}$ | $1.282\times10^{-6}$ | $3.95$ |
| halo 21, ramp 21 | $2.000\times10^{-2}$ | $6.565\times10^{-7}$ | $6.264\times10^{-9}$ | $1.576\times10^{-8}$ | $2.52$ |

**The bound holds on every row at an overestimate of $1.02\times$ to $4.04\times$**, against §4's $4.6\times10^{7}$ on the same scheme. Seven orders.

**The finding inside the finding: W49 asked for "a term for overlap width", and that is not quite the right term.** At fixed ramp, widening the halo from 11 to 61 leaves $\Pi$ unchanged at $9.54\times10^{-2}$ and $\sigma$ within $40\%$ of itself. Widening the *ramp* at fixed halo takes $\Pi$ from $0.75$ to $0.020$ and $\sigma$ from $1.20\times10^{-4}$ to $6.26\times10^{-9}$ — a factor $1.9\times10^{4}$. **Overlap width is a precondition; the partition of unity is the mechanism.** The one place halo width decides on its own is the transition at $\delta<d$, where $\Pi$ jumps to 1 and $\sigma$ jumps four orders.

**$C_\mu$, over all sixteen configurations** (these eleven plus §9.4's five): $[0.228,\,1.178]$, a spread of $5.2\times$ while $\sigma$ moves by $4.3\times10^{4}$. **A constant that moves $5\times$ while the quantity it relates moves $4\times10^{4}$ is behaving like a constant.** $C_\mu=1.2$ is the sampled maximum rounded up, and it is quoted as measured with a stated scope rather than back-fitted — the distinction §8.8 refused to blur, and the reason it can be crossed here is that the halo factorization keeps *one* constant while §4's needs $2.2\times10^{-8}$, which is a diagnosis and not a value.

**`unmeasured` is now empty for the `split-step` graph.** $C_\mu$ was the last entry.

---

## 9.3 W7 / W17 — the positive construction was built, and it does not work

§5.1 promoted W7 (R9's time-integrated flux) and W17 (waveform relaxation) on the grounds that *"the condition that is right for a one-step map is a statement about the time-integrated flux across the macro-step"*. **It was built and put through §5's own two tests, and it fails both, identically to the pointwise condition.**

The construction: carry the interface datum as the $W=2$ waveform $\lambda(t)=(1-s)\lambda^n+s\lambda^{n+1}$ — which `WindowNS.step_batch` already accepts through `bc0`/`bc1` — and match the **sub-step-averaged** flux rather than the flux at the end of the step. Both conventions are read off the *same* run, so no convention drifts between them. Configuration: §1's shared-layer tiling ($n=128$, halo $1$), the only one with a genuine single seam trace.

**Test A — substitute the reference's own trace.** A condition the true solution satisfies must have its residual *fall*:

| agent | $\Delta t$ | convention | $\lVert r\rVert$ lagged | $\lVert r\rVert$ at the true trace | ratio |
|---|---|---|---|---|---|
| exposed | $0.05$ | pointwise | $2.7593\times10^{-3}$ | $2.7639\times10^{-3}$ | $1.002$ |
| exposed | $0.05$ | **time-integrated** | $1.6376\times10^{-3}$ | $1.6403\times10^{-3}$ | $\mathbf{1.002}$ |
| embedded | $0.05$ | pointwise | $2.2053\times10^{-5}$ | $2.2945\times10^{-5}$ | $1.040$ |
| embedded | $0.05$ | **time-integrated** | $2.0316\times10^{-5}$ | $2.0889\times10^{-5}$ | $\mathbf{1.028}$ |
| embedded | $0.01$ | pointwise / integrated | $2.48\times10^{-5}$ / $2.03\times10^{-5}$ | $2.49\times10^{-5}$ / $2.04\times10^{-5}$ | $1.005$ / $1.005$ |
| embedded | $0.002$ | pointwise / integrated | $3.31\times10^{-5}$ / $2.39\times10^{-5}$ | $3.32\times10^{-5}$ / $2.41\times10^{-5}$ | $1.005$ / $1.005$ |

**Every ratio is above one.** The embedded row at $\Delta t=0.05$ reproduces §5's own numbers ($2.6421\times10^{-5}\to2.6520\times10^{-5}$, ratio $1.0037$) and its lagged composed defect is $3.7071\times10^{-4}$ — §4's total, to five figures. **Time-integrating changes the number by under $2\%$ and does not change the verdict anywhere.**

### Why, and this is sharper than §5's diagnosis

§5 concluded *"a condition whose solution is $\Delta t$-independent is a steady-state condition"*. The mechanism is more specific, and it is an identity rather than an inference. At a **shared-layer** seam both sides pin the *same* cell, so with outward normals

$$F_A+F_B \;=\; \nu\,\frac{2w_\Gamma-w_{A,\text{in}}-w_{B,\text{in}}}{h} \;=\; -\,\nu h\,\partial_{nn}w$$

**Verified bit-exactly against the monolith's own solution on all four seams** — `-nu*h*d2 matches: True`, with $\lVert r\rVert$ from $6.16\times10^{-5}$ to $1.73\times10^{-4}$ and $\max|\partial_{nn}w|$ from $0.107$ to $1.014$.

> **The flux-balance residual at a shared-layer seam is a discrete second derivative, not a jump.** It is $O(h)$ on the exact solution and vanishes only where that solution is *linear across the seam*, so driving it to zero asks the solution to be straight at every cut. It is a **spatial** defect, which is why it is $\Delta t$-independent ($2.21\to3.31\times10^{-5}$ across a $25\times$ range of $\Delta t$, while the true trace move falls $7.57\to1.53\times10^{-5}$, i.e. proportionally to $\Delta t$) — and why **no amount of integrating in time can remove it**. §5's $\Delta t$-scaling observation is explained by this and so is Test A, which the steady-condition reading explains only the first of.

**Test B — solve it exactly.** With the *embedded* agent, solving degrades the composed step by $1.59\times$ (pointwise) and $\mathbf{2.00\times}$ (time-integrated) — so the time-integrated version is not merely neutral but *worse*. With the **exposed** agent it is nearly neutral: $0.988$, i.e. a $1.2\%$ improvement, while the solved trace still overshoots the true one by $66\times$. **So §5's $8.9\times$ degradation is largely an R10 effect**: the trace fed a per-window pressure solve that amplified it, and with the elliptic part in the composition layer the same wrong trace is nearly harmless — the composed step is simply insensitive to the seam trace there.

### The retraction, and what replaces it

**§5.1's promotion of W7 and W17 is withdrawn.** R9's time-integration remains right for what R9 is about — *multirate* conservation, where two clocks genuinely need a common integral — and it is not the answer to §5's finding, because §5's finding is not about time.

**The construction that is valid for an explicit one-step map is the overlapping halo update itself.** It is not an approximation to a flux condition; it is a different and correct thing, and it now has its own error bound — §9.2's $\Pi$. That closes W46's "open as a construction" half, and not the way §5.1 predicted: *there is no flux-balance interface condition to fix, because an overlapping decomposition has no common interface at which a flux jump is defined.* Each subdomain has its own artificial boundary in a different place. That is the measured reason behind the compiler's existing `R2/axis` rule — probed-DtN forces the non-overlapping view — which until now asserted the coupling without the mechanism.

**W46's named measurement now exists in the package.** `probe.reference_trace_check` runs Test A on any residual functional in three calls and no rollout, and `probe.shared_layer_flux_sum` makes the identity checkable rather than asserted.

---

## 9.4 Beyond one configuration — and what varying $\Delta t$ broke

§7 lists *"one configuration, one state, one expert"* as the standing caveat, and §8 moved the numbers without removing it. Five configurations, varying $\mathrm{Re}$ by $4\times$ and $\Delta t$ by $4\times$:

| | $\beta$ | $\kappa$ | null | $\mu$ | asym | $L$ monolith | $L$ split-step | gap |
|---|---|---|---|---|---|---|---|---|
| $\mathrm{Re}=255$, $\Delta t=0.05$ | $0.3486$ | $1.196$ | $0$ | $+0.3486$ | $0.0026$ | $0.979930$ | $0.979925$ | $0.01\,\text{se}$ |
| $\mathrm{Re}=510$ | $0.2042$ | $1.110$ | $0$ | $+0.2042$ | $0.0024$ | $0.987482$ | $0.987478$ | $0.02\,\text{se}$ |
| $\mathrm{Re}=128$ | $0.5473$ | $1.348$ | $0$ | $+0.5473$ | $0.0032$ | $0.970675$ | $0.970665$ | $0.01\,\text{se}$ |
| $\Delta t=0.025$ | $0.3754$ | $1.122$ | $0$ | $+0.3754$ | $0.0022$ | $0.987294$ | $0.987305$ | $0.08\,\text{se}$ |
| $\Delta t=0.10$ | $0.3498$ | $1.193$ | $0$ | $+0.3498$ | $0.0024$ | $0.969843$ | $0.969824$ | $0.02\,\text{se}$ |

**Every claim §8 makes survives, and $L$ survives more strongly than §8.6 stated it.** §8.6 reported the composed $L$ matching the monolith's to five decimals at one configuration; across all five the worst disagreement is **$0.08$ standard errors**. $\kappa\in[1.11,1.35]$, $\mu>0$ and $\pi=0$ everywhere, null dim $0$ everywhere, L6/C1 convex everywhere, the W49 bound holds everywhere.

### The one that broke, and it was in the harness

**$220\times$ is not a property of the split-step construction.** At $\Delta t=0.025$ and $\Delta t=0.10$ the improvement over the as-built scheme collapsed to $1.8\times$ and $2.1\times$, with $\tau$ rising from $1.34\times10^{-6}$ to $1.60\times10^{-4}$ and $1.43\times10^{-4}$ — **two orders — while $\sigma$ stayed at $10^{-7}$ and every probe diagnostic stayed healthy.**

The cause: `SUBSTEPS = 10` is a hard-coded constant tuned to $\Delta t=0.05$, where it happens to equal the monolith's own internal sub-step count. The composition layer applies the exposed elliptic part **once per exchange**; the reference applies it once per internal sub-step. When the two cadences differ, the composed step is a *different splitting of the same equations*, and the difference is charged to $\tau$.

| $\Delta t$ | agent's own sub-steps | exchanges | $\tau$ | improvement |
|---|---|---|---|---|
| $0.025$ | $5$ | $10$ | $1.602\times10^{-4}$ | $1.8\times$ |
| $0.025$ | $5$ | **$5$ (matched)** | $\mathbf{8.004\times10^{-7}}$ | $\mathbf{361\times}$ |
| $0.050$ | $10$ | **$10$ (matched)** | $1.335\times10^{-6}$ | $217\times$ |
| $0.100$ | $20$ | $10$ | $1.428\times10^{-4}$ | $2.1\times$ |
| $0.100$ | $20$ | **$20$ (matched)** | $\mathbf{2.122\times10^{-6}}$ | $\mathbf{138\times}$ |

**With the cadence matched the construction is robust: $138\times$ to $361\times$ across a $4\times$ range of $\Delta t$.** The claim survives; the implementation did not.

> **This is [[plug-in-composition-theorems]] §1.4's attribution theorem for the third time on this project, and the third time it was a composition-layer defect wearing an agent's label.** First the decomposed pressure solve (§4.1, 99.8% of the defect, charged to $\tau$); then the partition of unity with full weight at the artificial edge (§8.1); now the exchange cadence. All three were silent, all three read as agent infidelity, and none was visible to any diagnostic the framework reports. **W34 — depth-tag every defect — is not enough on its own: a defect also needs the harness parameter it is a function of.** Opened as **W54.**

**R10b** is the rule: when the composition layer owns an agent's elliptic part, it must apply it at the agent's own sub-step cadence. It is an `admit` that *sets* the exchange interval, a decertification when `substeps_per_macro_step` is undeclared, and a refusal when two exposed agents declare different cadences. It also fixed an emitted number that had been wrong since the split-step scheme was built: the compiler reported `exchange_interval = 0.05`, the macro-step, describing a scheme nobody ran; it now reports $0.005$.

---

## 9.5 §4.5's positive control, run — $\Xi=0$ exactly

§7 and §8.3 both record it as unrun. It is the only measurement that can establish that the probe **reports zero when there is nothing to report**, and every nonzero $\Xi$ rests on that.

The expert is `reference.SpectralNS` — the periodic vorticity–streamfunction solver the frozen-checkpoint harness uses — whose `step` takes no boundary argument at all. The agent declares `bc_channel = NONE` and is otherwise identical to `WindowAgent`: same ports, same declared 16-mode prolongation, same flux, same ring indices, same macro-step. **The whole graph goes through `compile_scheme`**, not a hand-rolled probe.

| | measured |
|---|---|
| $\lVert\Lambda\rVert_2$ | $\mathbf{0.000000\times10^{0}}$ |
| $\max_{ij}\lvert S_{ij}\rvert$ | $\mathbf{0}$ — every entry bitwise zero |
| per block, `P00` / `P10` | $0$ / $0$ |
| $\beta$ | $0$ |
| null dim (declared $0$) | $16$ — the whole space |
| solver calls | $34$, so the probe genuinely ran |
| $\Xi=\lVert\Lambda^{\text{periodic}}\rVert/\lVert\Lambda^{\text{exposed}}\rVert$ | $\mathbf{0}$ |

**And the compiler does the right thing with it**: `L5/R6` refuses the direct and Krylov solvers on an empty operator, and the verdict carries **`refused claims: ["the word 'coupled' on any output involving seam sx0"]`** — spec §6.4(a), reached from a real periodic solver rather than a fixture. The null-space check reports $16$ excess directions, which is the correct reading of "the interface problem is empty" and not a defect in the seam.

**§9.1's falsifiable row 3 of [[probed-dtn-coupling]] is discharged as predicted**, and the $\Xi=0.19$–$0.53$ of §8.3 is not an artifact of a probe that manufactures small operators.

---

## 9.6 Two things that were wrong in `atlas/`, found by looking

**The `eps_tol` decertification ignored the measured constants the graph declares.** `_choose_tolerance` decertified unconditionally with the message *"Both are unmeasured (W3)"* — on a graph whose `MeasuredConstants` carries $\tau=1.3353\times10^{-6}$ and $\sigma=3.7334\times10^{-8}$. **Exactly the W45 class of bug, one call site further on**: W45 taught `unmeasured()` to consult `graph.measured` and stopped there. Fixed: the rule now applies, $\varepsilon_{\text{tol}}=\min(\tau,\sigma)=3.7334\times10^{-8}$, recorded as an `admit` with the provenance, and the decertification survives only when a constant is genuinely absent — naming which one.

**§8.9's "`admit` is now blocked by exactly one thing" was wrong.** It was blocked by three decertifications: `L2/G5/W16`, `L5/eps_tol` and `L6/AssemblyCertificate`. Two are now closed. The true statement is below.

---

## 9.7 Where the compile lands, 2026-08-28

| graph | verdict | refusals | decertifications | unmeasured | stamp |
|---|---|---|---|---|---|
| `as-built` | **`refuse`** | 1 — L2/R10 | 1 | — | E1–E5 hold, **E6 holds**, E7 holds |
| `split-step` | **`admit-uncertified`**, runnable | **0** | **1** | **none** | E1–E5 hold, **E6 holds**, E7 holds |

**Every one of the seven envelope hypotheses now stamps `holds` for the `split-step` graph, and every bound constant is measured.** What stands between it and `admit` is a single decertification:

> **`L2/G5/W16` — the decomposition policy.** *"Cut where the exact operator is closest to local and the conditioning is comfortable"* is adopted as policy with its **[AI Inference]** status preserved: identified, never derived, never tested. §2 gave it data for the first time (the cut score ordered the four seams physically, and $\kappa$ and the asymmetry agreed independently), and §9.4 gives it four more configurations — but **agreeing with two other diagnostics on five configurations is not a derivation**, and adopting a policy does not upgrade it. This is a theory-status item, not a missing measurement, and no amount of running the case study closes it.

That is a different end state from §8.9's: not "one named hole", but **one unearned inference**, with every hole filled and every constant measured. It is also the first time this vault can say what a graph would need to reach `admit` and have the answer be a page rather than a run.

---

## 9.8 Closed this session / still open

**Closed, each with a passing test or a from-scratch reproduction.**

| | what | evidence |
|---|---|---|
| `AssemblyCertificate.condition` | L6/C1: convexity, from the cellwise bias–variance identity. Necessary and sufficient, checkable from the declaration | 42 tests in `test_tier9_assembly_condition.py`; identity closes to $10^{-15}$ on the real tiling |
| **R11** | refuses a non-convex partition. Passes all five convex partitions, refuses the signed one | end-to-end through `compile_scheme` on the real graph and on a synthetic one |
| **W49** | $\sigma\le C_\mu\Pi\lVert\delta\lambda\rVert$ for the overlapping branch; §4 scoped to substructuring | bound holds on 16/16 configurations at $1.02$–$4.04\times$ |
| **$C_\mu$** | measured: $1.2$, from $[0.228,1.178]$ over 16 configurations | the last entry on `unmeasured` for the `split-step` graph |
| **W46** (construction half) | the valid construction for an explicit one-step map is the halo update, with $\Pi$ as its bound — *not* a time-integrated flux condition | the time-integrated condition built and measured to fail both of §5's tests |
| **W7/W17 promotion** | retracted. R9 stays a multirate rule | 8 measured rows, two conventions, three $\Delta t$ |
| **$\Xi=0$ control** | run, exactly zero, through the full compiler, with the word *coupled* refused | `out/xi/xi_control.json` |
| **R10b** | the exchange cadence must match the agent's sub-step cadence | $\tau$ recovers from $1.6\times10^{-4}$ to $8.0\times10^{-7}$ when matched |
| **`eps_tol` ingest** | the W45 bug at a second call site | $\varepsilon_{\text{tol}}$ now set from the declared constants |
| **the vacuous blend defect** | the driver blended the reference against itself | now $-1.041\times10^{-2}$ on real local solves |
| **broadened validation** | 5 configurations, $\mathrm{Re}$ and $\Delta t$ each $\times4$ | every §8 claim survives; $L$ within $0.08$ se everywhere |

**Still broken, or still only one configuration.**

- **One expert.** Everything on this page is `reference.WindowNS` and its no-projection subclass, plus one `SpectralNS` control. **$C_\mu=1.2$, $\Pi$'s calibration, and R10b's cadence rule are untested against a second solver and against any trained checkpoint.** This is the largest remaining caveat and nothing this session did touches it.
- **One topology.** Four windows, one cross-point, `MECH` ports only. The full 8-agent wind farm with real `ADVEC` and rotor ports has not been run against a real expert, so W5, W6 and the port algebra's harder cases remain fixture-tested only.
- **The cross-point is still declared, not treated.** W6 unchanged.
- **`G5/W16` is unearned**, and it is now the only thing between `split-step` and `admit`.
- **W33 has not been run.** The conformance suite rides on a probe that is now cheap and a flux convention that is now pinned (W47), so the blocker is gone and the work is not done.
- **`differentiable` is still `NONE`**, for the same one-line reason as §7.
- **Ramp 21 measures $5.4\times$ better than the working ramp 8** and the default was deliberately not changed. **W53.**
- **$\Delta t$-independence of $\Pi$ is untested at multirate**, where the exchange interval is not one number.
- **Three composition-layer defects have now worn an agent's label.** W34's depth tag does not catch any of them; a defect needs the harness parameter it is a function of. **W54.**

**How to reproduce, cold.**

```
python -m pytest tests/ -q                    # 191 tests
python scripts/tier0_window_ns.py             # the Tier 0 record, all four stages
python scripts/l6_assembly_condition.py       # 9.1's six partitions
python scripts/l6_macro_consequence.py        # 9.1's macro-step table
python scripts/w49_sigma_halo.py              # 9.2's eleven rows
python scripts/w7_time_integrated_interface.py --agent embedded
python scripts/tier0_sweep.py                 # 9.4's five configurations
python scripts/xi_positive_control.py         # 9.5
python scripts/vault_scan.py wiki             # W38's byte-level scan
```

---


---
---

# 10. 2026-08-28 (second session of the day) — the decomposition policy, derived and falsified

**Status of §1–§9 after this session:** every number reproduced cold before anything was touched — the compile, $L=0.979644$, $\kappa=1.196$, $\beta=0.3486$, $\tau=1.3353\times10^{-6}$, $\sigma=3.7334\times10^{-8}$, the PoU residual $1.11\times10^{-16}$, `norm_A` $=1$, $\chi_{\min}=1.9455\times10^{-3}$, the $216.7\times$ improvement, and the 191 tests. **Nothing drifted.** §9's conclusions stand except where a box below says otherwise, and **one of them is retracted**: §9.7's *"one unearned inference"* end state is closed, and that retraction is the subject of this section.

> **The one-paragraph version.** `G5/W16` is closed **both** ways it could be. It is **derived**: the composed defect over one exchange interval is exactly the $\chi$-weighted assembly of the per-subdomain *restriction defects* $D_i=\mathcal E_iR_i-R_i\mathcal E$, and convexity turns that identity into a cellwise bound — so *"the exact operator is closest to local"* acquires the definition it never had, from the same inequality that closed the assembly hole. And the scalarization it was carried by is **falsified**: $Q=\beta^{-1}\lVert S-\operatorname{diag}S\rVert/\lVert S\rVert$ ranks cut placements **backwards** ($-0.853$), is not a function of the decomposition at all (a $361\times$ orbit under re-declaring the same interface space), and is constant to $8\times10^{-11}$ where the measured defect spreads $2.29\times$. **`cases/window_ns.py` in `split-step` mode now compiles to `admit`** — the first in this project's history — and getting there exposed a third instance of the W45 bug class, **W56**.

---

## 10.1 The derivation — L2/C2, and what "closest to local" means

G5/W16's phrase named no object. This one does.

**The restriction defect.** For subdomain $i$ with restriction $R_i$, let $\mathcal E$ be the exact one-**exchange-interval** operator and $\mathcal E_i$ the agent's own, boundary data included:

$$D_i \;:=\; \mathcal E_i R_i - R_i\mathcal E$$

the failure of the exact operator to commute with restriction to $\Omega_i$ — a commutator, and the exact sense in which an operator is or is not *local to a subdomain*. The boundary data sits inside $\mathcal E_i$ deliberately: truncating the domain, and staling the datum that truncation forces, are two consequences of one cut and a criterion must charge for both.

**The identity.** With $\sum_iR_i^\top\chi_iR_i=I$ — the partition of unity L6 already checks —

$$\boxed{\;\mathcal A\bigl(\{\mathcal E_iR_iu\}\bigr)-\mathcal Eu \;=\; \sum_i R_i^\top\chi_i\,D_iu\;}$$

exactly, for *any* local maps, convex or not, PDE or not. **The composed defect over one exchange interval is the $\chi$-weighted assembly of the restriction defects, with nothing left over.**

**The bound.** Add $\chi\ge0$ — L6/C1, which **R11** already enforces — and the triangle inequality gives, at every cell $j$,

$$\bigl|\mathcal A(\{\mathcal E_iR_iu\})-\mathcal Eu\bigr|_j\;\le\;\sum_i\chi_{ij}\bigl|D_iu\bigr|_j\;\le\;\max_i\bigl|D_iu\bigr|_j$$

and therefore in every $\ell^p$ norm. **So the decomposition criterion is: minimize the $\chi$-weighted restriction defect.** It is a theorem on hypotheses the compiler already checks, which is the sense in which this *derives* the locality half of the policy rather than adopting it — and it is the same cellwise argument, on the same convexity hypothesis, that closed `AssemblyCertificate` in §9.1. **L6/C1 read on the solution is the assembly condition; read on the defect it is the cut condition.**

### Measured

| | identity residual (rel) | cellwise bound violation | runs |
|---|---|---|---|
| strip model, 4 cut placements $\times$ 4 states | $1.953\times10^{-12}$ | $2.220\times10^{-16}$ | 16 |
| `reference.WindowNS`, four windows, $u$ | $3.586\times10^{-11}$ | $2.220\times10^{-16}$ | 1 |
| `reference.WindowNS`, four windows, $v$ | $5.091\times10^{-12}$ | $2.776\times10^{-17}$ | 1 |

The violations are machine zero *against the state's scale*, and that floor is real rather than sloppy: where $D_i=0$ the bound is exactly $0$ while the blend computes $\chi_a x+\chi_b x$, which equals $x$ to within one ulp and is not bit-identical to it.

### The two forms are not close, and the gap is what the partition of unity buys

Four-window tiling, one exchange interval, joint $(u,v)$, relative:

| | value | tightness against the measured defect |
|---|---|---|
| measured one-interval defect | $2.6216\times10^{-7}$ | — |
| **$\lVert\sum_i\chi_i\lvert D_i\rvert\rVert$ (the criterion)** | $\mathbf{2.6268\times10^{-7}}$ | $\mathbf{0.9980}$ |
| $\lVert\max_i\lvert D_i\rvert\rVert$ | $5.7525\times10^{-5}$ | $0.0046$ |

**The $\chi$-weighted form is tight to $0.2\%$ and the max form is $220\times$ loose**, and the ratio between them is precisely the ramped partition of unity's contribution: $\chi$ is small exactly where $D$ is large, which is what §8.1 found the ramp was for and this is that finding as a number. The criterion is the $\chi$-weighted one; the max form is reported beside it because it is the one that survives when the weights are known only to be non-negative.

### The reference-free surrogate, and when it is exact

$D_i$ needs $\mathcal Eu$ — the monolithic solve a decomposition exists to avoid — so a criterion resting on it is a diagnostic, not a condition. That is the distinction `AssemblyCertificate`'s `must_satisfy` drew and L6/C1 had to clear. On the overlap $R_i\mathcal Eu=R_j\mathcal Eu$, so

$$\mathcal E_iR_iu-\mathcal E_jR_ju \;=\; D_i-D_j \quad\text{there, exactly}$$

the common mode cancels and what survives is the part of the defect **the cut itself creates**. It costs nothing: both local solves are already computed by the composed step.

**It equals the max form exactly when the agents' contaminated bands are pairwise disjoint** — then at most one $D_i$ is nonzero per cell — and that is decidable at compile time from the `contaminated` geometry the partition already declares for W49's $\Pi$. On two strips it reduces to $\text{halo}\ge2\rho s$:

| halo (reach $=2$) | supports disjoint | $\lVert\hat Q-Q^{\max}\rVert/\lVert Q^{\max}\rVert$ |
|---|---|---|
| $2$ | no | $6.611\times10^{-3}$ |
| $3$ | no | $1.009\times10^{-3}$ |
| $4,\ 5,\ 6,\ 8,\ 10$ | **yes** | $\mathbf{0}$ — exactly, at every one |

**On the four-window tiling the condition fails and the surrogate holds anyway.** The declared contaminated sets share $1120$ cells (measured: $184$ of $65025$ carry two nonzero $D_i$), because in 2-D two neighbouring windows can also share a *parallel* artificial face, which the two-strip argument does not cover. The surrogate agrees to $\mathbf{1.00007}$ regardless. **That is a measured statement and not a licence**: where the disjointness condition fails the surrogate stops being provably exact, and here it stops being provably exact while remaining accurate to four decimal places.

---

## 10.2 The falsification — $Q$ ranks cuts backwards, and is not a function of the cut

> **This retracts §2's *"the criterion now has data, and the data is not random"*, and §9.7's reading of it.** The data was not random; it was **anti-correlated**, and one graph whose cut cannot move could not show that. Both statements were honestly qualified at the time — §2 called $Q$'s scalarization a guess and §9.7 called agreement-on-five-configurations not a derivation — and the thing neither could do is the experiment below, because **the four-window tiling's cut is at the centre by construction and there is nowhere else to put it.**

The instrument is `tests/strip_model.py`: linear advection–diffusion on a torus, two overlapping strips, and a `cut` parameter that rotates the whole decomposition. Strip height, halo, ramp, agent count and agent are all held fixed; **only the cut location moves**. It is neither a fixture nor a case study — it earns nothing on Tier 0 and measures no constant of any real expert — it is a model problem for a *rule*, the way §9.1's algebra was for L6/C1.

### It ranks backwards

Twelve cut placements, four states each, $\nu$ varying across the cut direction:

| criterion | rank correlation vs the measured composed defect |
|---|---|
| **$Q=\beta^{-1}\lVert S-\operatorname{diag}S\rVert/\lVert S\rVert$** | $\mathbf{-0.853}$ |
| &nbsp;&nbsp;its locality factor $\lVert S-\operatorname{diag}S\rVert/\lVert S\rVert$ | $-0.853$ |
| &nbsp;&nbsp;**the same, un-normalized** $\lVert S-\operatorname{diag}S\rVert$ | $\mathbf{+0.853}$ |
| &nbsp;&nbsp;its conditioning factor $1/\beta$ | $-0.853$ |
| $\lVert\sum_i\chi_i\lvert D_i\rvert\rVert$ (L2/C2) | $\mathbf{+1.000}$ |
| $\lVert\max_i\lvert D_i\rvert\rVert$ | $+1.000$ |
| the reference-free surrogate | $+1.000$ |
| $\max_i\lVert D_i\rVert$ | $+0.972$ |

**Following $Q$ costs $1.407\times$ the best cut available — $92\%$ of the whole range on offer**, i.e. it picks very nearly the worst placement. L2/C2 picks the best one exactly.

**And the diagnosis is surgical.** The off-diagonal mass carries the right information: **un-normalized it ranks correctly**. Both things $Q$ does to it invert the sign.

1. **Dividing by $\lVert S\rVert$** removes the magnitude that matters. A ratio says what *shape* the operator has; the defect depends on how much of the evolution actually crosses the cut.
2. **Dividing by $\beta$** imports the wrong branch's amplifier. $1/\beta$ is the *interface solve's* gain — the one §5 measured turning an artifact residual into an $8.9\times$ degradation — and **W49 already scoped `master-error-bound` §4 to the substructuring branch for exactly this reason.** The overlapping bound $\sigma\le C_\mu\Pi\lVert\delta\lambda\rVert$ has no $\beta$ in it at all. The mechanism is legible in one line: $\beta\propto\nu$ for a diffusive DtN, so $Q$ prefers cutting where the coupling across the cut is *strongest* — and strong coupling is precisely what a lagged artificial boundary gets wrong.

> **This is §8.8's error one field over, and the vault has now made it twice.** §8.8 applied §4's product form to a halo scheme and it overestimated by $4.6\times10^{7}$; W49 fixed it by *scoping* §4 and deriving §4.1. `G5/W16` applied §4's amplifier to a halo scheme's cut placement, and the fix has the same shape: scope $Q$ to substructuring, derive L2/C2 for overlapping.

### It is not a function of the decomposition

Replacing a declared prolongation $P$ by $PU$ for orthogonal $U$ declares **the same interface space** — $\operatorname{range}(PU)=\operatorname{range}(P)$ — so the scheme is bit-identical, and under the `dirichlet` rung the composed step never reads the interface basis at all. $S\mapsto U^\top SU$ leaves $\beta$ and $\lVert S\rVert$ invariant and moves the off-diagonal mass:

| frame for the same $M$, same decomposition | $Q$ |
|---|---|
| declared (16-mode Fourier) | $0.5016$ |
| 64 random orthonormal frames | $[0.419,\;0.4895]$ |
| **eigenbasis of the symmetric part** | $\mathbf{0.001388}$ |

**A $361\times$ orbit on one decomposition**, with $\beta$ invariant to $10^{-10}$ throughout. And §8.3 measured the split-step operator's asymmetry at $0.002$ — **essentially self-adjoint** — so every seam of the graph this vault actually runs could be declared into a frame where $Q$ reads zero. `cut_score`'s own docstring worried that "the choice of basis is doing unjustified work"; it was doing all of it.

### It is blind where the answer is not in the operator

With $\nu$ uniform across the cut direction, every placement has a provably identical $S$:

| | $Q$'s relative spread over the scan | measured defect spread |
|---|---|---|
| uniform $\nu$, smooth random states | $3.98\times10^{-11}$ (probe roundoff) | $1.105\times$ |
| uniform $\nu$, one localized feature | $8.39\times10^{-11}$ | $\mathbf{2.29\times}$ |

**$Q$ is constant to eleven digits while the truth moves by a factor of two**, and the reference-free surrogate ranks both scans at $+1.000$ and picks the best cut in each. The decision a wind farm actually faces — cut through the wake or beside it — **is not expressible in the operator alone**, and $Q$ reads only the operator.

---

## 10.3 W56 — `admit` was reachable with an unmeasured constant

Found by the first graph that ever reached `admit`, which is the only way it could have been found.

`_w49_sigma_branch` quoted the module constant `C_MU_HALO` unconditionally, so a graph declaring **no** $C_\mu$ was handed $\sigma\le1.2\,\Pi\lVert\delta\lambda\rVert$ while the same compile listed `C_mu (W3)` on `unmeasured`. **`unmeasured` was reported on every compile and gated nothing.** That was invisible for as long as no graph could reach `admit`: every one carried at least the `G5/W16` decertification, so a missing constant could not change the verdict.

**This is the W45 / W52 bug for the third time**: the ingest path exists and one more call site did not use it. `holes.Unmeasured` was built so that `float(L)` *raises* rather than returning a plausible number; quoting $1.2$ for a graph that never measured it is the same act with the guard bypassed — and $1.2$ is what sixteen configurations of `reference.WindowNS` measured, **not a default for somebody else's solver**, which is the whole content of W55.

Two fixes, and the second is the one that matters:

- `_w49_sigma_branch` now reads $C_\mu$ from the record and decertifies when it is absent, naming it.
- **The backstop:** a non-empty `unmeasured` forces `admit-uncertified`, citing `L8/W56` and listing the constants. Each constant still has its own decision at the site that needs it; this exists so that the *fourth* such call site does not have to be found the hard way.

---

## 10.4 Where the compile lands, 2026-08-28 (second session)

| graph | verdict | refusals | decertifications | unmeasured | stamp |
|---|---|---|---|---|---|
| `as-built` | **`refuse`** | 1 — L2/R10 | 3 — `L2/C2`, `L6/W49`, `L8/W56` | `C_mu` | E1–E7 hold |
| `split-step` | **`admit`** | **0** | **0** | **none** | **E1–E7 all hold** |

> **This supersedes §9.7.** *"One unearned inference stands between `split-step` and `admit`"* is no longer true: the inference is closed, in both directions at once. **`compile_scheme` returns `admit` for the first time**, and the README's standing sentence *"`compile_scheme` still never returns `admit`"* is retired with it.

**What `admit` means here, stated narrowly.** The graph is overlapping with a convex partition of unity, so L2/C2's bound applies as a theorem; its value is measured ($2.6268\times10^{-7}$) and declared on the record with provenance, exactly as $L$, $\tau$, $\sigma$, $C_\mu$ and $\lVert\mathcal A\rVert$ are. Every envelope hypothesis is `holds` on a measurement rather than a default. **It does not mean the cut is optimal** — the criterion ranks cuts and this graph declares one — and it does not extend past this expert, this state, this cadence or this topology, which is what the provenance fields are for and what W55 remains open about.

**What is deliberately still decertified.** A **non-overlapping** graph gets `L2/C2/W57`: the derivation's hypotheses are a partition of unity and an assembly, and a substructuring decomposition has neither. $Q$ is retired rather than re-scoped into a criterion there, so **no criterion is derived for that branch** — a new open row rather than a silent gap.

---

## 10.5 Closed this session / still open

**Closed, each with a passing test or a from-scratch reproduction.**

| | what | evidence |
|---|---|---|
| **`G5/W16`, derivation half** | **L2/C2**: the composed defect is $\sum_iR_i^\top\chi_iD_i$ exactly, bounded cellwise by $\sum_i\chi_i\lvert D_i\rvert$ under L6/C1. *"Closest to local"* is $D_i=\mathcal E_iR_i-R_i\mathcal E$ | identity closes to $1.95\times10^{-12}$ (strip model, 16 runs) and $3.6\times10^{-11}$ (real expert); bound violated by $\le2.2\times10^{-16}$ everywhere; 19 tests in `test_tier10_cut_policy.py` |
| **`G5/W16`, falsification half** | $Q$ ranks backwards ($-0.853$), has a $361\times$ orbit under an admissible re-declaration, and is constant where the truth spreads $2.29\times$. Retired as a criterion, scoped to substructuring | 12-placement scan, 4 states each, plus a uniform-$\nu$ control and a localized-feature scan |
| **the criterion's constant** | `cut_defect_bound` $=2.6268\times10^{-7}$, declared on `MeasuredConstants` with provenance | measured in `scripts/tier0_window_ns.py`; tight to $0.2\%$ against the measured one-interval defect |
| **the reference-free form** | neighbour disagreement $=D_i-D_j$ on the overlap; **exactly** the max form when contaminated sets are pairwise disjoint | gap $0.000$ at halo $\ge2\rho s$, $6.6\times10^{-3}$ below it; $1.00007$ on the real tiling where the condition fails |
| **W56** | `admit` was reachable with an unmeasured constant. $C_\mu$ now read from the record; a non-empty `unmeasured` forces `admit-uncertified` | 3 tests; `as-built` now honestly reports it |
| **the first `admit`** | `split-step`: zero refusals, zero decertifications, zero unmeasured, E1–E7 all `holds` | `python scripts/tier0_window_ns.py`, then `compile_scheme` |

**Still open, and W55 is still the largest.**

- **One expert**, unchanged. $C_\mu=1.2$, $\Pi$'s calibration, R10b's cadence rule and now `cut_defect_bound` are all `reference.WindowNS`. **W55.**
- **No criterion for the substructuring branch.** $Q$ is retired and nothing replaces it there. **W57.**
- **The criterion's value needs a reference to measure**, exactly as $\tau$ does. A graph with no monolith declares the surrogate instead, which is measured to agree to $1.00007$ here and is *provably* equal only under disjointness. The compiler accepts one declared number and does not currently record which of the two it is.
- **One topology.** Four windows, one cross-point, `MECH` only. W5, W6 and the port algebra's harder cases remain fixture-tested.
- **The cross-point is still declared, not treated.** W6 unchanged.
- **W33 has not been run.**
- **W53** — ramp 21 still measures $5.4\times$ better than the working ramp 8 and the default is still ramp 8, for the same reason: every constant on this page, `cut_defect_bound` now included, is measured at ramp 8.
- **`differentiable` is still `NONE`.**

**How to reproduce, cold.**

```
python -m pytest tests/ -q                    # 213 tests
python scripts/tier0_window_ns.py             # Tier 0, now including L2/C2
python scripts/w16_cut_policy.py              # 10.1 and 10.2, the derivation and the falsification
python scripts/l6_assembly_condition.py       # 9.1's six partitions
python scripts/w49_sigma_halo.py              # 9.2's eleven rows
python scripts/tier0_sweep.py                 # 9.4's five configurations
python scripts/xi_positive_control.py         # 9.5
python scripts/vault_scan.py wiki             # W38's byte-level scan
```

---
---

# 11. 2026-08-28 — W55, the second and third experts, and the rule a learned operator broke

**W55 was the largest standing caveat**: every constant on this page — $C_\mu$, $\Pi$'s calibration, R10b's cadence rule, $L$, $\tau$, $\sigma$, $\beta$, $\kappa$, $\mu$, `cut_defect_bound` — came from `reference.WindowNS` and its no-projection subclass, plus one `SpectralNS` control that measures zero by construction. Two more experts now exist and the whole stack has been re-run against both.

| | `reference.WindowNS` | `reference.ChannelNS` | Poseidon-T |
|---|---|---|---|
| what it is | skew-symmetric advection, all-Neumann projection | **advective form**, `np.roll` Laplacian, **`project_outflow`** | **20.8M-parameter frozen neural operator** |
| its pressure solve | Poisson, homogeneous Neumann — **singular in the constant mode** | MAC, Neumann inlet/walls, **Dirichlet outlet — non-singular** | not declarable (**W60**) |
| resolution | free | free | **fixed at $128\times128$** |
| case study | `cases/window_ns.py` | `cases/channel_ns.py` | `cases/poseidon.py` |

> **The one-paragraph version.** Every rule held and one rule's *implementation* did not. **R10, R10b, R11/L6/C1 and L2/C2 all survive a second discretization**, and two of them survive it with numbers that transfer: L2/C2's bound is tight to $0.17\%$ on `ChannelNS` against $0.20\%$ on `WindowNS`, and the composed $L$ matches its monolith to $0.00$ standard errors on both. **$C_\mu$ transfers as a bound and not as a calibration** — implied $[0.088,0.708]$ against $[0.228,1.178]$, overlapping, with $1.2$ bounding all six new configurations. And **the learned operator found a live bug**: R2b left the transmission rung lifted for an agent declaring `time_discretization=unknown`, producing a *runnable* probed-DtN scheme — the construction §5 measured at $8.9\times$ worse than doing nothing with every diagnostic healthy. That is **W61**, and no expert this vault had ever compiled could have found it.

---

## 11.1 R10 survives losing the mechanism it was measured by

This is the sharpest single test on the page, and it was set up to be able to fail.

§4.1 identified the elliptic defect by a specific mechanism: `WindowNS`'s pressure Poisson problem carries homogeneous Neumann data on all four faces, so it is **singular in the constant mode** and solvable only if the ring's net flux balances — which a subdomain of a through-flow violates by construction. That was $99.8\%$ of the first composed defect and it is the measurement R10 came out of.

**`project_outflow` has an outlet Dirichlet, so it has no compatibility condition at all.** The flux imbalance simply leaves. R10's measured mechanism is therefore *absent* on the second expert, while R10's stated reason — an elliptic operator is global over whatever domain it runs on, so cutting the domain cuts the operator — is untouched.

$\tau$ for the embedded agent, across a $25\times$ range of $\Delta t$:

| $\Delta t$ | $\tau$ (`ChannelNS`, embedded) |
|---|---|
| $0.05$ | $2.6977\times10^{-2}$ |
| $0.01$ | $2.6251\times10^{-2}$ |
| $0.002$ | $2.5686\times10^{-2}$ |

**Flat: a ratio of $1.05$ over $25\times$.** That is §4.1's signature exactly — an inconsistent decomposition rather than an inaccurate one — reproduced on a projection that cannot violate a compatibility condition because it does not have one.

> **So R10 is about the elliptic operator being global, and not about that solver's Neumann compatibility.** The mechanism §4.1 measured was the *route* by which one solver expressed it, and the rule generalizes past its own evidence. This is the first rule on this project to be tested against the absence of the thing it was derived from.

**And the magnitude does not transfer at all.** `ChannelNS`'s embedded defect is $2.70\times10^{-2}$ against `WindowNS`'s $2.93\times10^{-4}$ — **$92\times$ worse** — so the split-step improvement is $\mathbf{21257\times}$ here against $217\times$ there. A rule that transfers and a number that moves by two orders is exactly the shape §9.4 warned about when `SUBSTEPS` broke: **the claim survives, the constant does not.**

---

## 11.2 What transferred, measured side by side

All at $\Delta t=0.05$, halo 21, ramp 8, the same tiling, the same state, the same declared 16-mode Fourier prolongation — held identical so that any difference is the expert's.

| | `WindowNS` | `ChannelNS` | transfers? |
|---|---|---|---|
| **L2/C2** identity residual | $3.59\times10^{-11}$ | $3.22\times10^{-11}$ | **yes** |
| **L2/C2** bound, $\chi$-weighted | $2.6268\times10^{-7}$ | $2.5433\times10^{-7}$ | — |
| &nbsp;&nbsp;its **tightness** | $0.9980$ | $\mathbf{0.9983}$ | **yes, to three digits** |
| &nbsp;&nbsp;reference-free surrogate ratio | $1.00007$ | $1.00006$ | **yes** |
| **L6/C1** PoU residual | $1.11\times10^{-16}$ | $1.11\times10^{-16}$ | yes (declaration) |
| &nbsp;&nbsp;$\lVert\mathcal A\rVert$, $\chi_{\min}$ | $1.0$, $1.9455\times10^{-3}$ | $1.0$, $1.9455\times10^{-3}$ | yes (declaration) |
| &nbsp;&nbsp;blend defect, real local solves | $-1.041\times10^{-2}$ | $\mathbf{-1.045\times10^{-2}}$ | **yes** |
| &nbsp;&nbsp;hull escape | $2.220\times10^{-16}$ | $2.220\times10^{-16}$ | yes |
| **W1** composed $L$ vs its monolith | within $0.01$ se | within $\mathbf{0.00}$ se | **yes** |
| **R10b** cost of a cadence mismatch | $\sim200\times$ in $\tau$ | $\mathbf{1231\times}$ in $\tau$ | **yes, harder** |
| $\beta$, split-step, seam `sx0` | $0.3486$ | $0.3212$ | close |
| $\kappa$, split-step | $1.196$ | $1.219$ | **yes** |
| asymmetry, split-step | $0.0026$ | $0.0038$ (`sx`), $\mathbf{0.29}$ (`sy`) | **no — and it is legible** |
| $\tau$, split-step | $1.3353\times10^{-6}$ | $5.8732\times10^{-7}$ | different, as expected |

**The asymmetry is the one clean disagreement and it is attributable.** §8.3 read `WindowNS`'s near-self-adjointness ($0.002$) as the physical signature of the pressure coupling being gone. `ChannelNS` reproduces that on the cross-flow seams ($0.0038$) and **not** on the along-flow ones ($0.29$) — and the reason is in the discretization: the skew-symmetric form $\tfrac12[u\!\cdot\!\nabla f+\nabla\!\cdot\!(uf)]$ is antisymmetric by construction where the advective form $u\partial_x f$ is not. **A property that looked like a property of the exposed elliptic part is partly a property of the advection form**, and only a second discretization could have separated them.

---

## 11.3 $C_\mu$ transfers as a bound, not as a calibration

W55's definition of done asked for *"a second real expert reproducing $\Pi$'s two limits and a $C_\mu$ inside the stated range"*. Six configurations of `ChannelNS`, varying the ramp (which §9.2 identified as the mechanism) and the halo (which it identified as only a precondition):

| halo | ramp | $\Pi$ | $\lVert\delta\lambda\rVert$ | $\sigma$ | implied $C_\mu$ | bound / $\sigma$ |
|---|---|---|---|---|---|---|
| $21$ | $1$ | $\mathbf{0.750}$ | $8.948\times10^{-5}$ | $4.748\times10^{-5}$ | $0.7075$ | $1.70$ |
| $21$ | $4$ | $0.2967$ | $8.637\times10^{-5}$ | $3.368\times10^{-6}$ | $0.1314$ | $9.13$ |
| $21$ | $8$ | $9.541\times10^{-2}$ | $8.620\times10^{-5}$ | $8.662\times10^{-7}$ | $0.1053$ | $11.39$ |
| $21$ | $21$ | $\mathbf{2.000\times10^{-2}}$ | $8.615\times10^{-5}$ | $1.519\times10^{-7}$ | $0.0881$ | $13.61$ |
| $11$ | $8$ | $9.541\times10^{-2}$ | $8.552\times10^{-5}$ | $9.635\times10^{-7}$ | $0.1181$ | $10.16$ |
| $41$ | $8$ | $9.541\times10^{-2}$ | $1.050\times10^{-4}$ | $1.109\times10^{-6}$ | $0.1106$ | $10.85$ |

**Three things reproduce and one does not.**

1. **The bound holds on 6/6**, at $1.70$–$13.61\times$ conservative.
2. **$\Pi$'s two limits reproduce.** $0.750$ at ramp 1 and $2.0\times10^{-2}$ at ramp 21 — the same two numbers §9.2 reports, because $\Pi$ is computed from the declared geometry and is an expert-independent quantity. That is worth saying plainly: **$\Pi$ transfers exactly, because it is not a measurement of the expert at all.**
3. **§9.2's mechanism claim reproduces.** At fixed ramp, moving the halo $11\to41$ leaves $\Pi$ unchanged and $\sigma$ within $15\%$; moving the *ramp* at fixed halo moves $\sigma$ by $\mathbf{313\times}$. *"Overlap width is a precondition; the partition of unity is the mechanism"* — confirmed on a second solver.
4. **$C_\mu$ is not calibrated across experts.** Implied $[0.0881,\,0.7075]$ here against $[0.228,\,1.178]$ there. The ranges **overlap** and $C_\mu=1.2$ bounds every configuration of both, so the *bound* is safe; but this expert's sampled maximum is $0.71$, so quoting $1.2$ for it is conservative by $1.7\times$ and quoting $0.088$ would be wrong for `WindowNS` by $13\times$.

> **So $C_\mu=1.2$ stays, and its status changes.** It was *"measured with a stated scope"* on one expert; it is now **the sampled maximum over two experts and 22 configurations, and a bound rather than a calibration for either**. Within an expert it behaves like a constant — $5.2\times$ spread against $\sigma$ moving $4.3\times10^{4}$ there, $8.0\times$ against $313\times$ here. Across experts it does not, and a page quoting it should say which of those two facts it is relying on.

---

## 11.4 A learned operator, and the first probe floor this vault has measured

Poseidon-T is not a discretization. It is a frozen $20.8$M-parameter neural operator, and what it cannot do is as informative as what it can.

### It cannot supply a monolith, and that is structural

`adapters.EXPERT_RES` is $128$ and the wrapper *raises* if the checkpoint disagrees. **There is no same-class monolith at any resolution, ever.** So $\tau$, $\sigma$, $C_\mu$ and L2/C2's reference form are **structurally unavailable** rather than merely unmeasured — and the reference-free surrogate §10.1 derived is the only form of the cut criterion this expert can carry. *That is the case it was derived for*, arrived at from the other direction.

### The probe floor is real, and OP-6 was about right

| | `WindowNS` | Poseidon-T |
|---|---|---|
| repeat-call determinism | bitwise | **bitwise** ($0.000$) |
| batch-position spread | n/a | $\mathbf{6.618\times10^{-7}}$ |
| probe floor | machine epsilon | $\sim10^{-4}$ |

$\lVert S\rVert$ against the finite-difference step:

| $\epsilon$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-7}$ |
|---|---|---|---|---|---|---|---|
| $\lVert S\rVert$ | $8.980\!\times\!10^{-3}$ | $8.949\!\times\!10^{-3}$ | $8.939\!\times\!10^{-3}$ | $9.122\!\times\!10^{-3}$ | $1.800\!\times\!10^{-2}$ | $1.780\!\times\!10^{-1}$ | $1.762$ |

Below $10^{-4}$, $\lVert S\rVert$ scales as $1/\epsilon$ — the signature of a fixed noise floor being amplified — and the implied noise, $\sim1.8\times10^{-7}$, **agrees with the independently measured batch-position spread of $6.6\times10^{-7}$ to within a factor of $3$**. Two routes to the same number.

> **§1's *"the probe floor is machine epsilon, not OP-6's $10^{-6}$"* is correct and was scoped to an expert that had never been probed.** It is a statement about a float64 deterministic solver. For a checkpoint the floor exists, OP-6's $10^{-6}$ is the right order, and the measured value is $0.66\times$ it.

**And $\beta$ is untrustworthy where $\lVert S\rVert$ is not.** Over $\epsilon\in[10^{-1},10^{-3}]$, $\lVert S\rVert$ is stable to $0.5\%$ while $\beta$ moves from $3.56\times10^{-5}$ to $8.01\times10^{-5}$ — a factor of $2.2$ — and $\kappa$ from $94$ to $41$. That is not a defect in the sweep: **$\beta$ is the smallest singular value and inherits the whole noise floor**, while the norm is dominated by the largest. Since $1/\beta$ is the substructuring branch's amplifier, the practical statement is that **the quantity the bound divides by is the least measurable thing on a learned expert** — one more reason §10.2's retirement of a $1/\beta$ criterion matters.

### What the seam looks like

$\beta=1.778\times10^{-3}$, $\kappa=9.411$, null dim $0$, $\mu=+1.534\times10^{-3}$ so $\pi=0$, asymmetry $0.2717$.

**The seam is passive.** E7's hypothesis holds for a trained checkpoint, measured rather than assumed — and `storage` is [[plug-in-composition-theorems]] §3's one certificate that survives an expert swap without re-earning, so this is the first time that claim has been tested on a swap that is genuinely to a different kind of object.

**$\Xi=0.0061$.** The first *cross-solver* composability index: the checkpoint's boundary response is $0.6\%$ of the `WindowNS` exposed agent's, on the same window and the same state. The vault now has three points — $0$ exactly (`SpectralNS`, no channel at all, §9.5), $0.006$ (a learned map whose ring is an initial condition), and $1$ by construction (a solver with a real Dirichlet ring). **A learned expert's boundary channel is real and two orders weaker than a solver's**, which is a quantitative version of what OP-2 and the wind-farm finding have been circling since the beginning.

---

## 11.5 W61 — R2b left the rung lifted, and only this expert could have shown it

**R2b** (§8.2) exists because §5 measured that solving the flux-balance interface condition makes a composed macro-step $8.9\times$ worse than not solving it, *while $\beta$, $\kappa$, the passivity spectrum and the probe residual all read healthy*. It gates the probed-DtN rung on `time_discretization`: for `explicit` it **corrects the rung** and records an `admit`.

For `unknown` it emitted a decertification ending *"It is not assumed to"* — **and left the rung lifted.** Which is assuming it.

Measured, on two graphs identical but for one field:

| `time_discretization` | rung | axis | runnable |
|---|---|---|---|
| `explicit` | `dirichlet` | overlapping | yes |
| **`unknown` (before)** | **`probed-DtN`** | **non-overlapping** | **yes** |
| `unknown` (after) | `dirichlet` | overlapping | yes |
| `implicit` | `probed-DtN` | non-overlapping | yes |

**This is the silent-wrongness class by the three-verdict split rule's own definition**: the compiler selected, and marked runnable, the scheme measured to be $8.9\times$ worse than doing nothing, on an *unestablished* premise, with a decertification beside it that described behaviour the code did not have.

**Why nothing found it before.** `window_ns` and `channel_ns` both declare `explicit`, so they took the corrected branch. `wind_farm` and `rocket` declared nothing — and `unknown` is the field's **default**, so the fixture whose documented purpose is *"R2 lifts the rung, the axis switches, the cross-point difficulty appears"* was getting its lift **from an omission**. Poseidon-T is the first expert for which `unknown` is the honest answer rather than an unfilled field: a frozen one-shot map is neither explicit nor implicit.

**The fix, and the asymmetry in it.** The rung is held at `dirichlet` and the decertification **stays**. That differs from the `explicit` branch on purpose: there we *know* probed-DtN is wrong, so correcting it is an `admit`; here we know only that it is unestablished, so correcting it is the conservative choice and nothing is certified either way. Both fixtures now **declare** `time_discretization=implicit`, which is what the rung-lift always required and never asked for.

> **The general lesson, and it is the third instance of one pattern.** W45, W52 and W56 were all *"a measured value exists and one call site does not consult it"*. W61 is the neighbouring failure: **a default that reads as permission.** `unknown` should be the most conservative value of its field and it was the most permissive. The other enum with an implicit-permission default is `EllipticSubsolve`, which has no `unknown` at all — **W60**.

---

## 11.6 Two things the record cannot say, found by the third expert

**W59 — a boundary channel that is an initial condition.** `FrozenFluidExpert.step(u, v, dt)` takes no boundary argument: there is nowhere to hold a trace *during* the step because there is no "during". `cases/poseidon.py` writes the ring into the **initial condition** instead, which is what a Dirichlet channel degenerates to for a one-shot map — and `BCChannel` has no value for that distinction, so the record says `DIRICHLET`. It is strictly weaker than what `WindowNS` provides and it is **not the object §2.2's Steklov–Poincaré operator is defined from**. $\Xi=0.0061$ is the size of the difference.

**W60 — undeclarable elliptic content.** `EllipticSubsolve` is `none | embedded | exposed` with no `unknown`, and for a black box the honest answer is `unknown`: declaring `none` asserts something nobody measured *and switches R10 off*, declaring `embedded` asserts something nobody measured *and makes L2 refuse*. **The compiler already half-catches this** — an incompressible `governing_family` with `elliptic_subsolve=none` is challenged, not believed — and that decertification fires on this graph.

`poseidon.elliptic_signature()` is the measurement that would settle it, from the probe alone and with two calibration points already in hand from §8.3 on the same instrument, basis and state:

| | $\kappa$ | asymmetry |
|---|---|---|
| `WindowNS`, elliptic **embedded** | $21.73$ | $0.153$ |
| `WindowNS`, elliptic **exposed** | $1.196$ | $0.002$ |
| **Poseidon-T** | $\mathbf{9.411}$ | $\mathbf{0.2717}$ |

Poseidon-T's asymmetry is **above the embedded calibration point** and its $\kappa$ is between the two. **This is a diagnosis and not a proof, and the difference is load-bearing**: a high asymmetry is *consistent with* an embedded elliptic part, and a learned operator could be non-normal for reasons that have nothing to do with pressure. What the statistic supports is a decertification — *"this record's `elliptic_subsolve` is not corroborated by the probe"* — and never a silent promotion of one value over another.

---

## 11.7 Where the three compiles land

| graph | verdict | refusals | unmeasured |
|---|---|---|---|
| `window_ns` split-step | **`admit`** | 0 | none |
| `window_ns` as-built | `refuse` | 1 — L2/R10 | $C_\mu$ |
| `channel_ns` split-step | `admit-uncertified` | **0** | $L$, $\tau$, $\sigma$, $C_\mu$ |
| `channel_ns` as-built | `refuse` | 1 — **L2/R10** | $L$, $\tau$, $\sigma$, $C_\mu$ |
| `poseidon` | `admit-uncertified` | 0 | $L$, $\tau$, $\sigma$, $C_\mu$, $p$ |

**`channel_ns` and `poseidon` both default `measured=None`, and that is deliberate rather than unfinished.** A constant measured on one expert is not another's, and handing `ChannelNS` the `WindowNS` numbers would be the W56 bug committed on purpose. The measurements to close `channel_ns` exist on this page and are not declared on its record in the same commit that measured them, for §9.1's reason about ramp 21: a number and the page that scopes it should land together.

**For Poseidon-T the decertification is permanent.** $\tau$ and $\sigma$ need a same-class monolith; the checkpoint is $128\times128$; there is no version of this that a better harness fixes. **A fixed-resolution learned expert can be composed, probed, and bounded on the transmission side, and cannot be certified against itself at scale.**

---

## 11.8 Closed this session / still open

**Closed.**

| | what | evidence |
|---|---|---|
| **W55** | three experts, not one. R10, R10b, R11/L6/C1 and L2/C2 all survive a second discretization; L2/C2's tightness and the composed-$L$ match transfer numerically | `scripts/w55_second_expert.py`, `out/w55/w55.json` |
| **R10 generalizes** | it survives the *absence* of the compatibility violation it was measured by: $\tau$ flat to $1.05$ over a $25\times$ range of $\Delta t$ on a **non-singular** projection | §11.1 |
| **$\Pi$** | transfers exactly, both limits — because it is computed from the declaration and is not a measurement of the expert | $0.750$ at ramp 1, $0.020$ at ramp 21, identical to §9.2 |
| **$C_\mu$** | a **bound** across experts, a **calibration** within one. $[0.088,0.708]$ against $[0.228,1.178]$, overlapping; $1.2$ bounds 22/22 configurations | §11.3 |
| **the probe floor** | measured for the first time on an expert that has one: $\sim10^{-4}$, with OP-6's $10^{-6}$ right to a factor of $1.5$ | §11.4 |
| **W61** | R2b left the rung lifted on `unknown`, producing a runnable probed-DtN scheme. Fixed; 5 tests | `test_tier11_second_expert.py` |
| **E7 on a checkpoint** | the learned seam is **passive** ($\mu>0$, $\pi=0$), so the one certificate that survives an expert swap is tested on a real swap | §11.4 |

**Still open.**

- **W57** — no cut criterion for the substructuring branch.
- **W58** — the record does not say which of L2/C2's two measurements `cut_defect_bound` is.
- **W59** — `BCChannel` cannot express "the ring of the initial condition is settable".
- **W60** — `EllipticSubsolve` has no `unknown`; `elliptic_signature` is a diagnosis, not a gate.
- **`channel_ns`'s constants are measured and not declared.** One commit away, deliberately not taken in this one.
- **One topology still.** Four windows, one cross-point, `MECH` only — on all three experts. W5, W6 and the port algebra's harder cases remain fixture-tested.
- **W6** — the cross-point is still declared, not treated.
- **W33** has not been run.
- **W53** — ramp 21 still measures better and the default is still ramp 8.

**How to reproduce, cold.**

```
python -m pytest tests/ -q                    # 224 tests
python scripts/w55_second_expert.py           # both new experts, all of section 11
python scripts/w55_second_expert.py --parts A # ChannelNS only; no torch needed
python scripts/tier0_window_ns.py             # the incumbent, unchanged
python scripts/w16_cut_policy.py              # section 10
```

---
---

# 12. 2026-08-28 — the topology: W33 run, W6 treated, and the port algebra pointed at real ports

Three rows had the same shape and had had it for a long time: **the machinery exists and has never been pointed at anything real.** W33's conformance suite was built with every field's test implemented and had never certified an expert. W6's cross-point rule was written and conditional on the decomposition, with a definition of done — *"a test on a 4-agent junction showing $\beta$ does not collapse"* — that had never been run on any graph. W5's `ADVEC` passengers, `ROT` port and field-to-lumped seam were fixture-tested only.

All three were run. **Every one produced a finding, and none of them needed anything new to be built** — they needed existing machinery aimed at a real object.

> **Three defects and one correction, in one sentence each.** **W63:** the conformance suite refused on the *block* while W48 was closed claiming it certifies on the *seam*, and the first real run committed exactly the error W48 documented. **W66:** an `ADVEC` port whose callable returned the power where its scale set says effort produced a **false E7 failure** of $3.9\times10^{-2}$, and nothing in the port algebra can catch that class. **W64:** the cross-point coupling lives in blocks `boundary_response` **cannot return**, so the vault has been refusing on cross-points it could never have measured. And W6's own stated test **passes** — $\beta$ does not collapse — while the defect a cross-point actually causes turns out to be something else entirely.

---

## 12.1 W33 — the conformance suite, run

Five real records plus one that lies, because a suite that has only ever seen true declarations has not been shown to catch a false one.

| record | verdict | `bc_channel` | `storage` | `validity` |
|---|---|---|---|---|
| `WindowNS` embedded | `admit-uncertified` | admit | admit | decertify |
| `WindowNS` exposed | `admit-uncertified` | admit | admit | decertify |
| `ChannelNS` exposed | `admit-uncertified` | admit | admit | decertify |
| **Poseidon-T** | `admit-uncertified` | admit | admit | decertify |
| `SpectralNS`, honest | `admit-uncertified` | admit | decertify | decertify |
| **`SpectralNS`, lying** | **`refuse`** | **refuse** | decertify | decertify |

Field by field over all six: `ports/schema` 6 admit, `nondim` 6 admit, `bc_channel` 5 admit and **1 refuse**, `storage` 4 admit and 2 decertify, `equivariances` 6 admit, `differentiable` 6 admit, `validity` **6 decertify**.

**The row's definition of done is met.** The lying record is the same expert with the same weights and **one field changed** — `bc_channel` claiming `DIRICHLET` where the expert has no boundary channel at all — and it is **refused at admission**: *"the probe measures $\Xi=0$: the agent's output does not depend on imposed boundary data."* Nothing else in the package can tell the two records apart, which is `plug-in-composition-theorems` §3's whole argument.

**`validity` decertifies on all six, and that is the correct answer rather than a gap.** It is falsifiable and not verifiable: certifying it needs the reference the predicate exists to make unnecessary. The certificate says *"not falsified on suite X"* and never *"valid"*. The architecture rests on exactly one unverifiable declaration and the suite is where that becomes visible.

### W63 — and the suite's first run committed the error its own theory page had closed

`_test_storage` **refused** when the probed **block** had a positive passivity defect. W48 was closed on 2026-08-28 with the statement *"the conformance suite certifies on the seam and reports blocks as diagnostics"*, and the suite did not.

The vault had measured why this matters twice already: §2.3 found an assembled seam at $\kappa=19$ whose own block was $2260$, and §8.5 found $1.196$ against $7061$ — *"any per-block gate would have condemned a seam that is in fact well conditioned."* Then the suite ran, and:

$$\text{Poseidon-T: block defect } 3.218\times10^{-3}, \qquad \text{seam defect } 0 \ \ (\mu = +1.534\times10^{-3})$$

**A record refused for not being passive, by a suite whose theory page says the verdict belongs to the seam, on an expert whose seam is passive.**

Fixed: `seam_defect` supplied $\Rightarrow$ the verdict is the seam's and the block is a diagnostic beside it; absent $\Rightarrow$ the block number is reported and the field **decertifies**, because refusing on a block is the error W48 named. Poseidon-T's certificate goes from `refuse` to `admit-uncertified`.

> **This is the fourth instance of one pattern and the first of a new sub-species.** W45, W52, W56 and W61 were all *"the theory says X and one call site does not do X"*. Those were about **measured constants** and a **default**; this one is about a **level** — the seam versus the block — and the page said which level for a whole session before the code did.

---

## 12.2 W6 — the cross-point, treated

### First, the thing that made it untreatable

The cross-point coupling is the off-diagonal block $\partial(\text{flux on face } B)/\partial(\text{trace on face } A)$ for two faces of **one agent**. The declared interface is

```
BoundaryResponse = Callable[[str, np.ndarray], np.ndarray]      # trace on V_i -> flux on V_i
```

— one port in, **that same port's** flux out. No sequence of single-port calls produces an off-diagonal block, because each call restarts from the base state. Measured: a per-seam assembly of the four-window junction gives a **block-diagonal** operator, and $\beta_{\text{global}}$ equals $\min_s\beta_s$ to every digit *by construction*.

> **So the vault has been declaring cross-points, and refusing on them at L2, without ever being able to measure one.** `scripts/w6_cross_point.py` supplies the generalized probe — `respond_multi`, $\{$face$\}\to\{$flux$\}$ in one solve — in the harness rather than in the package, because putting it on the record is a `PortAmendment`-sized decision. **W64.**

### With the multi-port probe

| | cross-point cell | $\min_s\beta_s$ | $\beta_{\text{global}}$ | collapse | off-diagonal fraction |
|---|---|---|---|---|---|
| shared-layer, halo 1 | $(127,127)$ | $2.8773\times10^{-1}$ | $2.8773\times10^{-1}$ | $\mathbf{1.00\times}$ | $1.84\times10^{-2}$ |
| overlapping, halo 21 | **none exists** | $2.7735\times10^{-1}$ | $2.7735\times10^{-1}$ | $1.00\times$ | $1.75\times10^{-2}$ |

**The seams genuinely see each other** — about $2\%$ of the global operator's norm is off the seam-diagonal, which a per-seam probe misses entirely — **and $\beta$ does not collapse.** W6's stated test passes, on a real 4-agent junction, for the first time. The smallest singular direction carries only $0.0097$ of its weight at the cross-point, so it is not a cross-point direction either.

The primal treatment costs what it should and buys nothing here: $64\to61$ multipliers ($3$ constraints, the four seams agreeing at one cell), and $\beta$ **unchanged to four digits**.

### The defect a cross-point actually causes

$\beta$ was the wrong instrument. Each seam carries its own multipliers over its own ring, so the shared cell gets a value from *each* seam that contains it — and those values differ, because each seam truncates a **different** function to the same 16 modes.

| component | seams | values at the cross-point | exact | spread |
|---|---|---|---|---|
| $u$ | `sx0`, `sx1` | $1.08511$, $1.06876$ | $1.12383$ | $\mathbf{1.62\%}$ of the trace scale |
| $v$ | `sy0`, `sy1` | $0.01154$, $\mathbf{-0.02530}$ | $0.01848$ | $\mathbf{66.7\%}$ — **and the sign disagrees** |

> **The cross-point cell is multi-valued, badly, in the transverse component — and $\beta$ says everything is fine.** That is this project's recurring shape for the fifth time: §5 (every diagnostic green, wrong equation), §9.1 (both blend defects passing, $166\times$ worse), §9.4 (every probe diagnostic healthy, two orders in $\tau$), §12.1 (a block verdict condemning a passive seam), and now this.

**It is a property of the declared interface space, not of the expert.** Each seam's 16-mode truncation is a different truncation; no better probe removes it, and every expert on a given tiling has the same one. That is exactly what a **primal (single-valued) corner degree of freedom** removes, and it is why the rule prescribes one — for a reason the rule did not state, since the rule was written about $\beta$.

**And the rule's conditionality is confirmed geometrically.** Under the overlapping tiling there is **no common cell at all** — each window's artificial boundary is somewhere else — so there is nothing to be multi-valued about. The *"no action under overlapping + partition of unity"* branch is right, and now for a measured reason.

---

## 12.3 The eight-agent topology, with real experts

`cases/wind_farm_real.py` keeps the fixture's topology exactly — eight agents, fifteen typed edges, the same adjacency — and replaces every `boundary_response` with a real one: six `reference.WindowNS` windows cut from the developed state at six *different* places, and two `disk.ActuatorDisk` rotors (algebraic, **zero fitted parameters**). The experts, ports, passengers and responses are real; **the geometry is schematic** and a number measured here is a number about the port algebra, not about wakes.

**What it actually exercises**, counted rather than claimed: **11 `MECH` + 4 `ADVEC` connections**; four `ADVEC` ports carrying a real `h0` passenger, with agent `N` carrying one on **two different faces** — the per-face case `CASE-STUDY-GUIDE` says a per-agent list cannot express; and **2 `ROT` ports, direction `OUT`, effective resolution 1, both unconnected** — genuinely open ports with a measurable power flow.

**And one claim that did not survive being derived.** This section began life asserting *"six cross-points instead of one"*. Derived from the adjacency, the fifteen edges contain exactly **one** 3-clique, $\{N, F, B_p\}$ — so the wind farm's own topology has the **same** cross-point count as the four-window tiling, not more. The interesting structure is the port types.

### W66 — the scale set is checked and the callable is not

The `ADVEC` port's first version returned $(\mathbf u\!\cdot\!\mathbf n)\,h_0$ — the enthalpy **power** — against a normal-velocity trace. The compile then failed **E7** with a passivity defect of $3.915\times10^{-2}$ on the *assembled* seam `e5a`.

The number was real and it was measuring nothing: the pairing was (velocity $\times$ power), which is not a power, so the passivity eigenvalue was not about dissipation. The declared scale set is `h0_effort` $=U^2$, `h0_flow` $=U$, `h0_power` $=U^3$, so the conjugate pair on that port is (specific total enthalpy, mass flux) and a velocity trace demands an **effort** response. Returning $h_0$ instead: **E7 stamps `holds`.**

**`check_scales` validates the declared scale *set* — $s_e s_f = s_P$ on the `nondim` dictionary — and nothing anywhere checks that `boundary_response` returns the declared flow variable.** Its parameters are `(port_type, scales, passengers)`; it never sees the callable, and `PortDecl` has no field naming which half of the conjugate pair the callable returns.

> **This is §2.2 / W47 arriving on a new port type.** There, a *transport* term in the MECH flux convention reported a passivity defect of $2.39$ on a block that was passive, and the fix was to pin the diffusive form as normative. An `ADVEC` port has **no diffusive form to fall back on** — transport is the whole of what it carries — so the only defence is the conjugate pairing, and the pairing is the thing nothing validates.

**It is the silent-wrongness class inverted**: a *false alarm* rather than a false pass. A graph would have been decertified — E7 `fails`, the $L\le1$ branch withdrawn — for a defect that does not exist. That is not better than the other direction, only differently expensive.

### W65 — two rules that are each right, meeting on one record

The rotors declare `elliptic_subsolve = none`, which is **true**: an `ActuatorDisk` is a closed-form zero-parameter law with no solve of any kind. They declare `governing_family = "incompressible-navier-stokes-2d"`, which they **must**, or E3 fails at every rotor face and $\tau$ goes `UNDEFINED` there — `CASE-STUDY-GUIDE` says so explicitly, *"put there after getting it wrong once"*.

R10's plausibility guard challenges exactly that pair: *"an incompressible solver almost always contains a pressure solve"*. It is right about solvers, this is not a solver, and **the guide's own instruction is what walks every lumped closure into it.** It decertifies rather than refuses, which is what keeps it survivable, and the decertification is a false positive on the two agents in the graph that are telling the truth.

### Where it lands

| | verdict | refusals | stamp |
|---|---|---|---|
| `as-built` (embedded) | `refuse` | 1 — L2/R10 | E1–E4 hold, E5/E6 unchecked, E7 holds |
| `split-step` (exposed) | `admit-uncertified` | **0** | E1–E4 hold, E5/E6 unchecked, **E7 holds** |

**The topology is admissible.** Fifteen typed edges across three port types, two open ports, a field-to-lumped seam and a cross-point compile with zero refusals; what is missing is measurements ($L$, $\tau$, $\sigma$, $C_\mu$, $\lVert\mathcal A\rVert$) and an assembly declaration, both of which the graph honestly reports. **W5's port-algebra cases are no longer fixture-only.**

---

## 12.4 Closed this session / still open

**Closed.**

| | what | evidence |
|---|---|---|
| **W33** | the conformance suite, run on five real records. A false `bc_channel` label is **refused at admission**; the honest one is confirmed | `out/w33/w33.json`, 3 tests |
| **W63** | the suite refused on the block where W48 says the seam decides. Poseidon-T: block $3.218\times10^{-3}$, seam $0$ | fixed; 3 tests |
| **W6** | run on a real 4-agent junction with a **multi-port** probe. $\beta$ does **not** collapse ($1.00\times$, off-diagonal $1.8\%$); the real defect is multi-valuedness, $1.6\%$ and $\mathbf{66.7\%}$ | `out/w6/w6.json`, 2 tests |
| **W5** (port algebra) | 11 `MECH` + 4 `ADVEC` edges, per-face passengers, 2 open `ROT` ports, a field-to-lumped seam, all against real experts. Zero refusals in `split-step` | `cases/wind_farm_real.py` |
| **W66** | a non-conjugate `ADVEC` pairing produced a **false** E7 failure of $3.9\times10^{-2}$ and nothing caught it. Fixed in the case study | 2 tests; E7 flips to `holds` |

**Still open.**

- **W64** — the cross-point coupling is not expressible through `boundary_response`. The multi-port probe lives in the harness; putting it on the record is a `PortAmendment`-sized decision.
- **W66's rule** — the case study is fixed and **the gap is not**: nothing validates that a callable returns its declared conjugate half. It needs a declaration (which half the response is), not a cleverer diagnostic — §12.3's second test shows the operator alone cannot say.
- **W65** — R10's guard misfires on an algebraic closure declaring the flow's family, which the guide requires.
- **W6's primal treatment is measured and not enforced.** L2 refuses an untreated cross-point under non-overlapping and does nothing about multi-valuedness under any decomposition, which is the defect that was actually found.
- **The eight-agent graph declares no partition of unity and no measured constants**, so E5 and E6 are `unchecked` there. It is a topology test and it has not been rolled out.
- **W57**, **W58**, **W59**, **W60**, **W62** — unchanged from §11.8.
- **W53** — unchanged.

**How to reproduce, cold.**

> **Superseded 2026-08-29 by §13.9's block, which is the current one.** The test
> count below was right on 2026-08-28 and is now 256; the two scripts §13 adds are
> missing from it. Kept because it is what this session's numbers were produced by.

```
python -m pytest tests/ -q                    # 234 tests
python scripts/w33_conformance.py             # 12.1
python scripts/w6_cross_point.py              # 12.2
python scripts/w55_second_expert.py           # section 11
python scripts/w16_cut_policy.py              # section 10
python scripts/tier0_window_ns.py             # sections 1-11's incumbent
```

---

## 12.5 The whole session, in one table

§10, §11 and §12 were one session and each has its own ledger. **This is the consolidated one**, for a reader picking the project up cold who needs to know what changed and what did not.

**Where the compiler lands now.**

| graph | verdict | refusals | decertifications | unmeasured | stamp |
|---|---|---|---|---|---|
| `window_ns` split-step | **`admit`** | **0** | **0** | **none** | **E1–E7 all `holds`** |
| `window_ns` as-built | `refuse` | 1 — L2/R10 | 3 | $C_\mu$ | E1–E7 hold |
| `channel_ns` split-step | `admit-uncertified` | 0 | 5 | 4 | E1–E4, E6, E7 hold |
| `wind_farm_real` split-step | `admit-uncertified` | 0 | 6 | 5 | E1–E4, E7 hold |
| `poseidon` | `admit-uncertified` | 0 | 18 | 5 | E1–E4, E7 hold |

**234 tests pass**, from 191 at the start of the session.

### Closed

| | what | how it was closed |
|---|---|---|
| **`G5/W16`** | the decomposition policy's **[AI Inference]**, the last thing between any graph and `admit` | **both ways at once**: derived as **L2/C2** from the restriction-defect identity, and its scalarization falsified three separate ways |
| **W55** | one expert | **three experts**: a second discretization and a frozen neural operator, with every rule surviving and $C_\mu$ transferring as a bound but not a calibration |
| **W33** | the conformance suite, never run | run on five real records plus one that lies; the liar is **refused at admission** |
| **W6** | the cross-point, declared not treated | measured on a real 4-agent junction: $\beta$ does **not** collapse, and the defect is **multi-valuedness** ($66.7\%$ in the transverse component) |
| **W5** (port algebra) | `ADVEC`, `ROT`, field-to-lumped: fixture-tested only | the eight-agent topology with real experts, compiling with zero refusals |
| **W56** | `admit` was reachable with an unmeasured constant | $C_\mu$ read from the record; a non-empty `unmeasured` forces `admit-uncertified` |
| **W61** | R2b left the rung lifted on `unknown` — a **runnable** scheme measured $8.9\times$ worse than doing nothing | the rung is held down and the decertification stays |
| **W63** | the conformance suite refused on the **block** where W48 says the **seam** decides | the verdict is the seam's; the block is a diagnostic |

### Opened

| | what |
|---|---|
| **W57** | no cut criterion is derived for the **substructuring** branch; $Q$ is retired and nothing replaces it there |
| **W58** | the record does not say which of L2/C2's two measurements `cut_defect_bound` is |
| **W59** | `BCChannel` cannot express a channel that is an **initial condition** ($\Xi=0.0061$ is the size of the difference) |
| **W60** | `EllipticSubsolve` has no `unknown`, so a black box cannot be declared honestly |
| **W62** | `ChannelNS`'s constants are measured and not yet declared |
| **W64** | cross-point coupling is **not expressible** through `boundary_response` — the vault has been refusing on something it could not measure |
| **W65** | R10's plausibility guard misfires on an algebraic closure declaring the flow's family, which the guide requires |
| **W66** | the scale set is checked and **the callable is not**; a non-conjugate pairing gave a false E7 failure |

### Unchanged, and worth saying so

- **One topology of experts, one governing family.** All four real case studies are incompressible Navier–Stokes in 2-D. Nothing here says anything about a multiphysics seam, and E3's `governing_family` comparison has still never been exercised against a genuine disagreement.

  > **Superseded 2026-08-29 (§13.2).** A fifth real case study puts `compressible2d` against `thermostruct2d` across one `THERM` seam. E3 has now been exercised against a genuine disagreement and does exactly what the spec says: it `fails`, $\tau$ goes `UNDEFINED`, and the probe still runs — $\beta = 4.806$, $\kappa = 1.0002$, zero refusals.
- **W53** — ramp 21 still measures $5.4\times$ better than the working ramp 8, and the default is still ramp 8, because every constant on this page is measured at ramp 8.
- **`differentiable` is still `NONE`** everywhere, for §7's one-line reason.
- **No graph has been rolled out beyond 120 macro-steps**, and the claim-typing horizon has never been tested against a trajectory that exceeds it.
- **W13's declination rate $p$ is still unmeasured**, so spec §3.4's $K$-threshold is still a decertification at every graph size.

> **The methodological result, which outlasts every number above.** Five times this project has now found the same shape: **the reported diagnostic is healthy and the thing that matters is not.** §5 (every probe diagnostic green, the wrong interface equation), §9.1 (both blend defects passing, a partition $166\times$ worse), §9.4 (every diagnostic healthy, two orders in $\tau$ from a hard-coded cadence), §12.1 (a block verdict condemning a passive seam), §12.2 (a cross-point cell multi-valued while $\beta$ reads perfect). And four times it has found the neighbouring shape: **the theory page says X and one call site does not do X** — W45, W52, W56, W63 — with **W61** as its inverse, *a default that reads as permission*. Neither list is closed, and a session that adds to either has done more for this project than one that adds a number.


---

# 13. 2026-08-29 — the fifth expert, the pairing declared, and two rules exercised against a real disagreement

§12.5 ended with a diagnostic rather than a number: every hole found in the previous session came from one of four mechanisms, and only the first is found by auditing. **Mechanisms 2–4 — the declared interface cannot express a phenomenon that exists; validation checks the declaration and never calls the callable; composed diagnostics never proven jointly sufficient — are found by pointing unmodified machinery at an object nobody wrote with this framework in mind.** This session was spent that way, and the result is that three of the four mechanisms fired again, on the first object of that kind the vault has had.

> **The one sentence per item.** **W55's remaining half:** E3's `governing_family` comparison had never met a genuine disagreement, and now it has — the composition **runs** at a seam where E3 `fails`, exactly as the spec claims, with $\beta = 4.806$ and $\kappa = 1.0002$. **W66:** a declared field is the only fix, and this is now a **measurement** rather than a preference — the operator cannot say, the magnitude cannot say, and the power balance cannot say, all three for one reason. **W64:** decided against extending the record, with the L2 refusal rewritten to cite the quantity §12.2 actually measured. **W57:** closed with a theorem *and* its measurement — **L2/C3** ranks cut placements at $+0.579$ to $+1.000$ over eleven configurations while §4's own product form ranks **negative in nine of them**. **W60:** closed by an enum value, because the measurement route that was supposed to close it is now known to be unsound for an entire class of elliptic operators. And **W59** is *not* closed, deliberately.

---

## 13.1 The fifth real case study, and what only it can say

`CASE-STUDY-GUIDE` sets the bar for a fifth real case study at *"it can answer a question none of the other four can"*. All four existing ones — `window_ns`, `channel_ns`, `poseidon`, `wind_farm_real` — are 2-D incompressible Navier–Stokes, which is the caveat §12.5 recorded under *"unchanged, and worth saying so"*.

`cases/thermal_seam.py` is a **conjugate-heat-transfer seam**: `solvers/compressible2d.Compressible2D` against `solvers/thermostruct2d.ThermoStruct2D`, both imported from the build repo **unmodified**, each graded against a closed-form oracle by that repo's own M1 suite (the exact Riemann solution, the isentropic vortex, the erf slab, free thermal expansion). Neither was written with this package in mind.

| | gas | shell |
|---|---|---|
| `governing_family` | `compressible-navier-stokes-2d` | `thermoelastic-shell-2d` |
| `bc_channel` | `DIRICHLET` (isothermal wall) | **`ROBIN`** ($-k\,\partial T/\partial n = h(T-T_\infty)$) |
| `time_discretization` | `EXPLICIT` (MUSCL/HLLC, SSP-RK2) | `IMPLICIT` (backward Euler) |
| `elliptic_subsolve` | `NONE` | **`EMBEDDED`** (a global solve, and a quasi-static elasticity solve besides) |
| `dt_native` | $10^{-4}$ s | $5\times10^{-2}$ s |

**Four rules meet a genuine disagreement here for the first time**, and the geometry is schematic and says so: a $0.20 \times 0.02$ m duct over a $0.20$ m $\times$ 8 mm shell, 48 cells along the seam on both sides. A number measured here is about the port algebra and the envelope stamp, not about a rocket.

> **The shell is also the first real expert in this vault whose boundary channel is *above* `dirichlet`.** Every other real record declares `DIRICHLET`, so R1's *"the interface is as weak as its weakest agent"* has never had two different values to be an inequality over. Measured: gas rung 1, shell rung 3, seam rung 1. The ladder works; it had simply never been used as one.

---

## 13.2 W55's remaining half — E3, against a disagreement that is a fact about the solvers

E3 is checked by string comparison and the spec calls it load-bearing. Until now every graph that failed it was a fixture built to fail it. The claim under test is precise, from `_stamp_E3`'s own decertification text: *"tau is emitted as UNDEFINED for both sides. The composition still RUNS correctly — probing never mentions a governing equation, so the transmission layer is robust to this failure. It is the error ATTRIBUTION that has no meaning."*

> **Superseded by §16, 2026-08-29.** E3's *hypothesis* is unchanged and still fails, but its **consequence** was wrong: `tau` is not undefined at a multiphysics seam whose two sides declare `lambda_ref`. It needs a reference TRAJECTORY, not a shared governing family, and the tightly coupled pair is one. `thermal_seam` now compiles with `tau_undefined_seams` empty and `L1/E3` admitting.

Measured on `thermal_seam` in `split-step` mode at a matched clock — **the variant with zero refusals**, so nothing else is confounding the result:

| quantity | value |
|---|---|
| verdict | `admit-uncertified`, **0 refusals** |
| E3 | **`fails`** — `'compressible-navier-stokes-2d'` against `'thermoelastic-shell-2d'` |
| $\tau$ | **`UNDEFINED`** on seam `cht` |
| the probe | **ran**: $\beta = 4.8062$, $\kappa = 1.000231$, $\dim\ker = 0$, operator non-empty |

> **Superseded 2026-08-30 (W74).** The probe linearized about the **zero trace**, and this port's effort is a temperature in kelvin — so every number in the table above is the operator at $0$ K. At the experts' own operating point the same run gives $\beta = 0.3807$ and $\kappa = 1.000072$; $\beta$ moves by a factor of $12.6$. **Every clause of the claim survives** — the verdict, the $0$ refusals, `E3 fails`, $\tau$ `UNDEFINED` on `cht`, $\dim\ker = 0$ and a non-empty operator are all unchanged — because each is about a sign or a verdict rather than a magnitude. See §14.4.

**Every clause of the claim holds.** E3 fails cleanly rather than throwing; $\tau$ is marked undefined at the seam and nowhere else; and the transmission layer assembles a **better-conditioned operator than any single-physics graph in this vault** — $\kappa = 1.0002$ against `window_ns`'s $1.20$ in `split-step` and $21.7$ as-built.

> **That last number is not a coincidence and it is worth stating.** A conjugate-heat-transfer seam over one exchange interval is dominated by the local Robin term $h(T_{\text{gas}} - T_{\text{wall}})$ on one side and the conduction-limited $k_{\text{gas}}/\delta n$ on the other, both of which are near-multiples of the identity in any basis. The multiphysics seam is the *easiest* interface problem here, not the hardest — which inverts the intuition that a seam between different equations must be worse conditioned than one between the same equations.

**E4 also fails, honestly and separately.** The gas runs at $10^{-4}$ s and the shell at $5\times10^{-2}$ s, a genuine multirate seam, and L7 refuses under **R9** — *"multirate coupling requires conservation on the TIME-INTEGRATED flux over the macro-step"*. The `clocks="matched"` variant brings the shell down to the gas's clock so E3 can be measured with E4 out of the way; bringing the gas *up* to the shell's is 60,000 CFL sub-steps per probe column and is not a measurement anyone runs.

**W55 is now closed in both halves.** §11 closed *"one expert"* with three; this closes *"one governing family"* with two.

---

## 13.3 W66 — the pairing is a declaration, and that is measured rather than preferred

§12.3 opened W66 with an `ADVEC` port whose callable returned the enthalpy *power* against a velocity trace, producing a **false E7 failure** of $3.9\times10^{-2}$. The row asked for *"a declaration of which half the callable returns, and an L3 check"*, and noted the operator alone cannot infer it. The question left open was whether something cheaper exists.

**Three candidate routes were measured. All three fail, and for one reason.**

### Leg 1 — the operator cannot say, and the THERM seam shows it is worse than "cannot"

`PORT_SPECS[THERM]` declares the bond $(T,\ q_n/T)$ — entropy flux — and its own note says $(T, q_n)$ is *"the pseudo-bond most co-simulation codes exchange"*. Every thermal solver in the build repo computes $q_n$: `ThermoStruct2D._robin` implements $-k\,\partial T/\partial n = h(T-T_{\text{gas}})$ and `generate.py::_wall_flux` returns $h$ and $T$. **Nothing anywhere divides by $T$.** So the physically natural callable returns the wrong half, and it arrives that way by itself rather than by construction.

| | verdict | stamp | E7 | passivity defect | $\dim\ker$ | $\lVert S\rVert$ | $\beta$ |
|---|---|---|---|---|---|---|---|
| `entropy` (declared) | `admit-uncertified` | E1–E4, E7 hold | `holds` | $0$ | $0$ | $4.807$ | $4.806$ |
| **`heat`** (pseudo-bond) | `admit-uncertified` | **identical** | `holds` | $0$ | $0$ | $456.5$ | $456.5$ |

**Verdict, stamp, null count and passivity defect are numerically identical.** The only thing that moves is $\beta$, and nothing checks $\beta$. Fitting the two operators, $S_{\text{heat}} = 94.97 (measured at the probe's zero base; **1199.0 at the declared operating point, W74 / §14.4** — the *ratio* is what the leg rests on and it moves with the base, while the conclusion, a positive rescale, does not)\,S_{\text{entropy}}$ to a relative residual of $6.4\times10^{-5}$: a **positive rescale**.

$$\operatorname{sym}(cS) = c\operatorname{sym}(S) \implies \lambda_{\min}\bigl(\operatorname{sym}(cS)\bigr) = c\,\lambda_{\min}\bigl(\operatorname{sym}(S)\bigr), \qquad c>0$$

so every eigenvalue keeps its **sign** and E7's passivity test is *exactly* invariant. And $q_n \big/ (q_n/T) = T > 0$ always. **So on `THERM` the most common wrong convention in all of co-simulation is precisely the one E7 is structurally blind to.**

> **W66 is therefore both directions, not one.** §12.3 characterized it as *"the silent-wrongness class inverted — a false alarm rather than a false pass"*. On `ADVEC` it is a false alarm because that pairing multiplies by a *velocity*, which changes sign across the seam. On `THERM` it is a **false pass**. The row was right about the gap and wrong about which way it cuts.

### Leg 2 — the magnitude cannot say, and this is general

The obvious cheap guard is to nondimensionalize the probed response by its declared scale and complain if it is not $O(1)$. On `THERM` it works overwhelmingly: the declared entropy-flux scale is $11.11$, the correct response nondimensionalizes to $3.0$ and $34.3$ on the two sides, the wrong one to $1.96\times10^{3}$ and $2.24\times10^{4}$.

**On `wind_farm_real` it is vacuous.** There $U_\infty = 1$, so `h0_effort` $=$ `h0_flow` $=$ `h0_power` $= 1$, and the correct `ADVEC` response $h_0 = 0.5418$ and the incorrect $(\mathbf u\!\cdot\!\mathbf n)h_0 = 0.5713$ differ by **5%**.

This is not a quirk of one case. The power identity C4 *enforces* is $s_e s_f = s_P$; set $s_e = s_f = 1$ and $s_P = 1$ follows. **A properly nondimensionalized port declares all three halves at the same scale, so magnitude carries no information about which half a number is — and the port algebra requires the nondimensionalization.** The framework's own hygiene destroys exactly the information the check would need.

### Leg 3 — the power balance cannot say either

The sharpest remaining idea: `storage` is a declared functional $u\mapsto H(u)$, so check $\mathrm{d}H/\mathrm{d}t$ against $\int_\Gamma e\cdot f$. If the response is the wrong half the product is not a power and the balance should fail.

| port | correct pairing | wrong pairing |
|---|---|---|
| `THERM` (shell) | $\int e f \big/ q_{\text{true}} = 1.385$ | $\mathbf{900.0}$ — exactly $T_{\text{hot}}$ |
| `ADVEC` (`wind_farm_real`, trace $10^{-3}$) | $\int e f \big/ \dot H = 0.473$ | $0.499$ |
| `ADVEC` (trace $10^{-1}$) | $2.971$ | $3.422$ |

**It discriminates on `THERM` only because `THERM` is dimensional** — the factor is $T$ in kelvin. On the nondimensional port the separation is 5–15%, indistinguishable from ordinary modelling error, and no threshold exists.

### What all three legs have in common

$$\text{wrong half} = (\text{right half}) \times (\text{an } O(1) \text{ factor, once nondimensionalized})$$

**No dimensionless diagnostic can separate $O(1)$ from $O(1)$.** That is one statement covering all three failures, and it is why the answer is a declaration.

### The fix, and what it can and cannot do

`ports.ResponseHalf` (`EFFORT` / `FLOW` / `UNDECLARED`), a `PortDecl.response_half` field, and **L3/C9**:

- both sides declare the **same** half $\Rightarrow$ `admit`, and the trace is the conjugate half;
- both declare and they **differ** $\Rightarrow$ **refuse**, because $\tilde\Lambda_M = \sum_i P_i^*\Lambda_i P_i$ *adds* the two responses and a traction plus a velocity is not a quantity;
- either **undeclared** $\Rightarrow$ decertify — the pre-existing silence, made audible.

> **The positive control, and it is the sharpest number in this section.** Before C9 existed, `window_ns` in `split-step` mode — **the only graph in this vault that reaches `admit`** — was made to return the *velocity* where its declared pair says *traction*, on one side of every seam. It compiled to **`admit`**: zero refusals, E7 `holds`, every $\beta$ and $\kappa$ healthy. With the halves declared honestly the same graph **refuses at L3/C9**.

**And what the fix does not do, stated plainly.** A record whose declaration is *false* — the callable returns the flow, the record says effort — is still undetected, and by legs 1–3 it is undetectable. **`response_half` is the second unverifiable declaration in this architecture**, after `validity`. §12.1 said *"the architecture rests on exactly one unverifiable declaration and the suite is where that becomes visible."* It now rests on two, and that is the price of closing W66 rather than a reason not to.

---

## 13.4 W64 — decided: cross-point coupling is out of scope, permanently

The row offered two acceptable answers and forbade only silence. **The record is not extended**, and the reasoning is §12.2's own measurements read as a cost-benefit rather than as a gap:

- the off-diagonal coupling is real and **1.8%** of the global operator norm;
- $\beta$ does **not** collapse — $1.00\times$ on a real four-window junction — so the quantity the old refusal cited is not affected by the block it could not see;
- the defect a cross-point actually causes is **multi-valuedness**, fixed by a primal corner degree of freedom, which is a *declaration* and not a measurement.

A `PortAmendment`-sized widening of the one interface every expert must implement, to obtain a 1.8% number no rule consumes, is a cost with no verdict attached. `compiler.CROSS_POINT_COUPLING_SCOPE` states it permanently and both cross-point branches cite it in the artifact.

**L2's refusal stands and its stated reason is replaced**, because §12.2 measured the old one and it was wrong in all three claims — $\beta$ does not collapse, the conditioning is not degraded, and the smallest singular direction carries $0.0097$ at the cross-point. It now cites `cross_point_multivaluedness`, which is **decidable from the declaration** (a cell in two seams under a non-overlapping axis is multi-valued, full stop), where the old one cited $\beta$, which is not.

> **This is a rule that was resting on an unmeasurable quantity and now rests on a decidable one, with the same verdict on every graph.** The refusal did not change; what it can honestly say did.

---

## 13.5 W57 — L2/C3, derived and measured, and §4's own product form ranks backwards

The row asked for *"a criterion for the substructuring branch derived from §4's own factorization — where $1/\beta$ genuinely **is** the amplifier — or an explicit statement that cut placement there is unconstrained."* It closes with the first.

### The derivation

`master-error-bound` §4 chains two steps, and the first is an **identity**:

$$\lVert\lambda^\dagger-\lambda^\star\rVert \;\le\; \bigl\lVert\tilde\Lambda^{-1}\bigr\rVert\,\bigl\lVert(\tilde\Lambda-\Lambda)\lambda^\star\bigr\rVert \;\le\; \frac{1}{\beta}\,\bigl\lVert\Lambda-\tilde\Lambda\bigr\rVert\,\lVert\lambda^\star\rVert$$

Stopping one step earlier removes the operator mismatch entirely. With $\lambda^\dagger$ solving $S_M\lambda=\chi_M$ on the declared interface space and $a^\star$ the exact trace's coordinates there,

$$S_M(\lambda^\dagger - a^\star) = \chi_M - S_M a^\star =: -r \implies \lVert\lambda^\dagger-a^\star\rVert \le \frac{\lVert r\rVert}{\beta}$$

$$\boxed{\textbf{L2/C3:}\quad Q_{\text{sub}}(\Gamma) \;=\; \frac{\lVert S_M a^\star - \chi_M\rVert}{\beta}}$$

— the residual the exact trace leaves in the approximate interface equation, amplified by the interface solve's own conditioning. **Two forms, exactly as L2/C2 has two**: this tight one and §4's loose product form, the same inequality chain used twice. $C_\mu$ is absent because it is the agents' sensitivity to interface data, is not a function of the cut, and a constant cannot change a ranking.

### The instrument

`tests/substructure_model.py` — a steady advection–diffusion torus cut in two by the **exact algebraic Schur complement**, with a freely movable cut and everything else held fixed. A *model problem for a rule* in `strip_model.py`'s third category: neither a fixture nor a case study, earning nothing on Tier 0.

**Steady, and that is forced rather than convenient.** §4's bound assumes the transmission error enters through an interface solve; §4's own 2026-08-28 box says applying it to an overlapping scheme is *"not conservative but uninformative"*. For an unsteady explicit march flux balance is not the right condition at all — that is §5 — so an unsteady model would measure §5 again instead of W57. A steady problem poses a boundary-value problem by construction.

**With the full interface space the model reproduces the monolith to $5.3\times10^{-16}$**, so every defect it reports is the *declared* interface space's truncation and not the model's error — the same source §12.2 found the cross-point defect in.

### Measured, over eleven configurations

| | Spearman rank against the measured composed defect |
|---|---|
| $Q_{\text{sub}} = \lVert r\rVert/\beta$ (tight) | $+0.5794$ to $+\mathbf{1.0000}$ — **positive in 11/11** |
| $\lVert r\rVert$ alone | strictly lower in **11/11** |
| §4's product form $\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert/\beta$ | $-0.7441$ to $+0.1176$ — **negative in 9/11** |

- **Tightness:** $Q_{\text{sub}}/\text{defect} \in [0.965,\,1.707]$, median $1.358$ — a bound within a factor of $1.7$, not merely a ranking.
- **Cost of following it:** at most $\mathbf{1.105\times}$ the best available cut. The falsified overlapping-branch $Q$ cost **92%** of the available range.
- **The defect spreads $4\times$ to $36\times$ across placements**, so there was something to rank.

> **Two findings beyond the criterion.** First, **$1/\beta$ helps here and inverted the ranking there.** §10.2 measured $1/\beta$ turning a $+0.853$ correlation into $-0.853$ on the overlapping branch; §4's box said why that should not transfer, and this is the measurement of the other half — $Q_{\text{sub}}$ beats the bare residual in **11 of 11**. The scoping was correct, and it is no longer only stated.
>
> Second, **§4's own product form is not merely loose, it is anti-correlated.** It must be used as a bound and never as a criterion — and it is *the shape the retired $Q$ had*, which is the cleanest available explanation of why that one ranked at $-0.853$.

L2 now issues `L2/C3` on the substructuring branch: `admit` with the value when a graph declares `cut_defect_bound`, and a decertification naming the criterion when it does not. `C2/W57` — *"no criterion is derived for this branch"* — is retired.

---

## 13.6 W60 — closed by an enum, because the measurement that was to close it is unsound

W60 opened because `EllipticSubsolve` has no `unknown`: for a black box, `none` asserts something unmeasured *and switches R10 off*, `embedded` asserts something unmeasured *and makes L2 refuse*. The row's proposed resolution was to promote `poseidon.elliptic_signature` *"from a diagnosis to a gate once it has more than two calibration points"*.

**The extra calibration points arrived, from physics that is not Navier–Stokes, and they falsify the promotion.** The `thermal_seam` shell is a *known* embedded elliptic part — backward Euler over the whole shell, plus a quasi-static elasticity solve with no time step to shrink at all:

| record | true value | $\kappa$ | asymmetry | the statistic's verdict |
|---|---|---|---|---|
| `WindowNS` (§8.3) | `embedded` | $21.73$ | $0.153$ | consistent with EMBEDDED |
| `WindowNS` (§8.3) | `exposed` | $1.196$ | $0.002$ | consistent with EXPOSED |
| **shell, as-built** | **`embedded`** | $\mathbf{1.0002}$ | $\mathbf{8.8\times10^{-5}}$ | **"consistent with EXPOSED or NONE"** |
| shell, split-step | `exposed` | $1.0002$ | $8.8\times10^{-5}$ | consistent with EXPOSED |

**The statistic reads the same for both, to four decimal places, and calls the embedded one exposed.** It is not a cadence artifact: swept over a $10{,}000\times$ range of exchange interval the asymmetry moves only from $2.7\times10^{-6}$ to $7.1\times10^{-6}$, against an embedded calibration point of $0.153$ — a factor of $5\times10^{4}$ away at every cadence.

> **The diagnosis is exact, and it is a mechanism-4 finding.** `elliptic_signature` measures **non-normality**, and it has been read as measuring **globality**. Those coincide for a pressure projection, where the elliptic part is a *constraint* that makes the interface operator strongly non-normal. They come apart for a **self-adjoint** elliptic operator: $(M/\Delta t + K)$ is SPD, so its Schur complement on a seam is near-normal *however global it is*. There is a whole class of embedded elliptic solves this statistic cannot see, and it is the class containing every diffusion problem.

> **Superseded 2026-08-30 (W68).** The diagnosis above is right and it is not the whole mechanism, and the difference matters because the diagnosis predicts a fix that does not work. If the trouble were non-normality-read-as-globality, measuring the shell's **own block** rather than the assembled seam would help; it does not, because the block reads $\kappa = 1.0001$ too. One level lower: the block *is* $5.0001\,I$ — the shell's film coefficient — with the conduction operator $300\times$ underneath it. The statistic was not blind, it was **swamped**, and no threshold on $\kappa$ or the asymmetry could have recovered anything. The identity defect $\omega$ separates the two situations at $2370\times$, and `elliptic_signature` now returns a third outcome, `NO OPERATOR RESOLVED`, on this input. See §14.1 and §14.3. The conclusion of this section is unchanged: the promotion stays refused and `EllipticSubsolve.UNKNOWN` stays the honest declaration.

So for a black box there is no honest value **and no prospect of measuring one**. `EllipticSubsolve.UNKNOWN` is added, and L2 issues `R10/W60` — a decertification, not a refusal, because an undeclared internal is an unverified hypothesis rather than a measured contradiction, and it is strictly more conservative than `none`, which is what W61 asks of a default.

---

## 13.7 W59 — not closed, and this is the finding rather than the omission

The row needs a `BCChannel` value for *"boundary data enters through the initial condition, not persistently"*, measured on Poseidon-T at $\Xi = 0.0061$ against a solver's Dirichlet ring. **The bar §12.5 set was a second learned expert or a re-reading showing the enum is load-bearing wrong rather than merely imprecise. Neither arrived, so no value is added.**

What *did* arrive is the exact trigger, measured from the compiler's own rung table:

| `bc_channel` | `time_discretization` | rung |
|---|---|---|
| `none` | any | `dirichlet` |
| `dirichlet` | `unknown` | `dirichlet` — **W61's fix holds it down** |
| `dirichlet` | **`implicit`** | **`probed-DtN`** |
| `dirichlet` | `explicit` | `dirichlet` |

**`bc_channel` is load-bearing for the rung only when `time_discretization = implicit`.** Poseidon-T declares `unknown`, so W61's fix masks W59 completely: the rung is `dirichlet` either way and the imprecise declaration costs nothing today.

> **The two errors cancel, and the cancellation is not stable.** The moment any learned expert declares `implicit` honestly, an initial-condition channel declared `DIRICHLET` — because there is nothing else to say — lifts the rung to `probed-DtN` on a premise that is false, and §11.5 measured that construction at $8.9\times$ worse than doing nothing. W59 is a latent defect with a named trigger, which is a more useful thing to record than an enum value nobody has evidence for.

---

## 13.8 Where the compiles land, 2026-08-29

| graph | verdict | refusals | decertifications | unmeasured | stamp |
|---|---|---|---|---|---|
| `window_ns` split-step | **`admit`** | **0** | **0** | **none** | **E1–E7 all `holds`** |
| `window_ns` as-built | `refuse` | 1 — L2/R10 | 3 | $C_\mu$ | E1–E7 hold |
| `channel_ns` split-step | `admit-uncertified` | 0 | 5 | 4 | E1–E4, E6, E7 hold |
| `wind_farm_real` split-step | `admit-uncertified` | 0 | 6 | 5 | E1–E4, E7 hold |
| **`thermal_seam` split-step, matched** | `admit-uncertified` | **0** | 6 | 5 | E1, E2, E4, E7 hold; **E3 `fails`** |
| `thermal_seam` split-step, native | `refuse` | 1 — L7/R9 | 7 | 5 | **E3 and E4 `fail`** |
| `thermal_seam` as-built, native | `refuse` | 2 — L2/R10, L7/R9 | 7 | 5 | E3 and E4 `fail` |

**`window_ns` still reaches `admit` with the new L3/C9 check in place**, which is the point: C9 was added, every shipped record declares the field, and the flagship graph's verdict is unchanged. **256 tests pass**, from 234 at the start of the session. `scripts/tier0_window_ns.py` reproduces **bit-exact** from a cold start — 545 numbers across `w1`, `w2` and `w3`, zero drift.

---

## 13.9 Closed this session / still open

**Closed.**

| | what | evidence |
|---|---|---|
| **W55** (remaining half) | one governing family; E3 never exercised against a real disagreement | a real multiphysics seam: E3 `fails`, $\tau$ `UNDEFINED`, **the probe runs** at $\beta=4.806$, $\kappa=1.0002$, zero refusals. `out/w67/w67.json` |
| **W66** | the scale set is checked and the callable is not | `ResponseHalf` + **L3/C9**, with a **measured** proof that a declaration is the only fix: the operator, the magnitude and the power balance all fail, for one reason. 9 tests |
| **W64** | cross-point coupling not expressible through `boundary_response` | **decided**: not extended, scope stated permanently, and L2's refusal moved from $\beta$ (unmeasurable, and measured not to collapse) to multi-valuedness (decidable). 2 tests |
| **W57** | no cut criterion for the substructuring branch | **L2/C3**, derived from §4's own factorization and measured: $+0.579$ to $+1.000$ over 11 configurations, tight to $1.7\times$, cost $\le 1.105\times$. `out/w57/w57.json`, 6 tests |
| **W60** | `EllipticSubsolve` has no `unknown` | added, **because the measurement route is unsound**: a known-embedded self-adjoint solve reads as exposed at every cadence over a $10^4$ range |

**Opened.**

| | what |
|---|---|
| **W68** | **`elliptic_signature` measures non-normality and has been read as measuring globality.** They coincide for a pressure projection and come apart for any self-adjoint elliptic operator, which is every diffusion problem. The statistic must not gate; §11.6's *"diagnosis and not a proof"* was righter than it knew |
| **W69** | **`response_half` is unverifiable, and is the second such declaration** after `validity`. A record whose declaration is false is undetectable by legs 1–3 of §13.3. The conformance suite can catch a lying `bc_channel` ($\Xi=0$) and cannot catch this |
| **W70** | **A thermoelastic body coupling has no port.** The shell's conduction and elasticity are coupled *volumetrically* through thermal strain, not across a surface, and the port algebra has only surface bonds. `ShellAgent.mechanical` is real, exercised and **on no port** — the first physics in this vault the five types cannot express, and it is not a sixth port type either |
| **W71** | **A multiphysics seam is better conditioned than a single-physics one** ($\kappa=1.0002$ against $1.20$ and $21.7$), which inverts the working intuition. Nothing rests on it yet and nothing has explained it beyond §13.2's one-paragraph argument |

**Still open, unchanged or sharpened.**

- **W59** — *sharpened, not closed*: the trigger is now exact (any learned expert declaring `time_discretization=implicit`), and W61's fix is what currently masks it. §13.7.
- **W65** — R10's plausibility guard still misfires on an algebraic closure declaring the flow's family.
- **W58** — the record still does not say which of L2/C2's two measurements `cut_defect_bound` is, and L2/C3 has now given the substructuring branch a **second** quantity with the same name and a different definition, which makes the row more urgent rather than less.
- **W62** — `ChannelNS`'s constants are measured and still not declared.
- **W53** — ramp 21 against ramp 8, unchanged.
- **W13's $p$** — still unmeasured; spec §3.4's $K$-threshold is still a decertification at every graph size.
- **No graph has been rolled out beyond 120 macro-steps**, and `thermal_seam` has not been rolled out at all — it is a compile-and-probe result, and its E5/E6 are `unchecked` for that reason.

**How to reproduce, cold.**

```
python -m pytest tests/ -q                    # 256 tests
python scripts/w67_thermal_seam.py            # 13.1, 13.2, 13.3, 13.6
python scripts/w57_substructuring.py          # 13.5
python scripts/w33_conformance.py             # 12.1
python scripts/w6_cross_point.py              # 12.2
python scripts/w55_second_expert.py           # section 11
python scripts/w16_cut_policy.py              # section 10
python scripts/tier0_window_ns.py             # sections 1-11 incumbent; bit-exact
python scripts/vault_scan.py wiki             # 190 files, 0 problems
```

> **The methodological result, updated.** The recurring shape now has **six** instances — §5, §9.1, §9.4, §12.1, §12.2, and §13.3's `heat` convention, where verdict, stamp, null count and passivity defect are *bit-identical* between a correct pairing and a wrong one. And a seventh of a new kind arrived in §13.6: **a diagnostic that measures the right thing about the wrong quantity**, correct on the two calibration points it was fitted to and wrong on the first independent object it met. The four-mechanism diagnostic §12.5 proposed held up as a *planning* instrument: mechanism 1 produced nothing this session because there was nothing left to audit, and mechanisms 2, 3 and 4 produced everything above — every one of them from pointing unmodified machinery at `compressible2d` and `thermostruct2d`, which nobody wrote with this framework in mind.


---

---

# 14. Locality, and the price of the boundary condition — 2026-08-30

§13 closed five holes and opened five. Four of the five opened turned out to be **one question asked from four directions**, and it is not the question any of the four rows asked:

> How much of a probed block is the expert's **operator**, and how much is the **boundary condition** sitting on top of it?

`elliptic_signature` misreading a known-embedded solve (W68), the multiphysics seam being the best-conditioned interface in the vault (W71), `stencil_radius` having no test (W69), and the shell's uncertified half (W70) are four views of that. One statistic answers all four, and measuring it exposed a sixth hole (**W74**) that none of them predicted and that moves $\beta$ on a real seam by a factor of $12.6$.

Reproduce with `python scripts/w73_locality_and_scope.py`; the artifact is `out/w73/w73.json`.

## 14.1 The statistic: how much of a block is not a scalar

$$\omega(S) \;=\; \frac{\lVert S - c I\rVert_F}{\lVert S\rVert_F}, \qquad c = \frac{\operatorname{tr} S}{n}$$

Frobenius, because the question is how much of the *whole matrix* is off-scalar, not how much the largest direction is. It is invariant under $S \mapsto cS$ for $c>0$, which matters: §13.3 measured that the wrong THERM half is exactly a positive rescale of the right one, so a statistic that moved under a rescale would be reporting the flux convention.

**Why nothing already in the record could see this.** A multiple of the identity is a perfectly healthy operator by every diagnostic the probe emits: $\beta$ is fine, $\kappa$ is $1$, the null count is $0$, the passivity defect is $0$. All four are then properties of a **declared constant**, and there is no way to tell from any of them.

`thermal_seam`'s seam is that case, and the arithmetic is explicit:

| block | $\lVert S\rVert_F$ | $\omega$ | $\kappa - 1$ | what it actually is |
|---|---|---|---|---|
| gas | $0.3705$ | $1.91\times10^{-5}$ | $7.2\times10^{-5}$ | $-k_{\text{gas}}/\delta n \cdot I$, the conduction-limited film |
| shell | $1.8934$ | $1.97\times10^{-5}$ | $7.2\times10^{-5}$ | $+h_{\text{in}} \cdot I$, the declared Robin coefficient |
| **assembled** | $1.5229$ | $2.00\times10^{-5}$ | $7.2\times10^{-5}$ | **the difference of two film coefficients** |

Against the two graphs that do carry operators, on the same instrument:

| block | $\omega$ | declared `elliptic_subsolve` |
|---|---|---|
| `window_ns` as-built, W00 | $0.5268$ | `embedded` |
| `window_ns` split-step, W00 | $0.0467$ | `exposed` |
| `thermal_seam`, shell | $1.97\times10^{-5}$ | `embedded` |
| `thermal_seam`, gas | $1.91\times10^{-5}$ | `none` |

`OPERATOR_CONTENT_FLOOR` is $10^{-3}$: a **2370$\times$** separation, $47\times$ of margin above and $51\times$ below, and the probe's own noise floor for $\omega$ is $1.9\times10^{-6}$ — so the thin blocks are *resolved small numbers*, not noise. Every $\lVert S\rVert$ above is stable to $1.0003\times$ across $\varepsilon$ from $10^{-2}$ to $10^{-5}$.

### The spatial-domain picture, which is what makes it legible

Poke a delta at one seam cell and watch where the response goes:

| expert | spread | profile out from the pole |
|---|---|---|
| gas | **0 cells** | $9.26\times10^{-2}$, then **bit-zero** at every other cell |
| shell | **5 cells** | $4.71\times10^{-1}$, $1.59\times10^{-3}$, $1.83\times10^{-4}$, $2.88\times10^{-5}$, $5.19\times10^{-6}$, $1.01\times10^{-6}$ |

**The shell's globality is present and resolved.** It is a clean exponential over 5 cells with about twelve decades of dynamic range above the floor. It is simply **300$\times$ below the boundary condition sitting on top of it** — and $\omega$ is that ratio.

> **[AI Inference]:** the same reading explains a block nobody had looked at. On `window_ns`'s seam `sx0` the two sides are wildly unequal — W00 has $\lVert S\rVert_F = 1.474$ and $\omega = 0.047$, W10 has $0.070$ and $\omega = 0.964$ — a $21\times$ asymmetry between two instances of *the same solver* on a geometrically symmetric $2\times2$ tiling. It is not noise: both norms are stable to $1.0003\times$ over four decades of $\varepsilon$. The mental picture of $\tilde\Lambda_M$ as a sum of two comparable DtN blocks is wrong even in the most symmetric real case here, and the assembled $\kappa = 1.196$ shows none of it.

## 14.2 W71 — closed, with a closed form instead of a paragraph

§13.2 asserted that both sides are near-multiples of the identity and called that a sketch. It is now a two-group law. Sweeping the shell block alone over the exchange interval (7 decades) and the film coefficient (7 more):

$$\omega \;\approx\; C \, \Pi, \qquad \Pi \;=\; \underbrace{\frac{h d}{k}}_{\text{Bi}} \cdot \bigl(k_{\max}\,\ell\bigr)^2, \qquad \ell = \sqrt{\alpha\,\Delta t}$$

with $C$ measured in $[0.135,\ 0.307]$ over **11 points spanning $\Pi$ from $1.0\times10^{-5}$ to $1.0$** — a $2.27\times$ spread in $C$ over five decades of $\Pi$, from two independent sweeps that agree with each other. Below $\Pi \approx 10^{-6}$, $\omega$ stops falling and sits at $1.92\times10^{-6}$, which is the probe's own floor; above $\Pi \approx 1$ it saturates at $0.27$, where the expansion stops holding. Here $k_{\max} = 251.3$ rad m$^{-1}$ (a $25$ mm shortest retained wavelength against an $8$ mm shell).

**The two groups are the two independent ways an operator can hide**, and that is the content:

- $\mathrm{Bi} \ll 1$ — the interface resistance $1/h$ swamps the internal resistance $d/k$, so the seam sees a film and not a body.
- $k_{\max}\ell \ll 1$ — heat does not travel one interface wavelength in one exchange interval, so no seam mode can tell it apart from any other.

**Either one alone suffices.** That is why §13.6's $10{,}000\times$ sweep of the exchange interval moved nothing worth seeing: it varied $\ell$ while holding $\mathrm{Bi} = 0.033$ fixed, and at $\mathrm{Bi} = 0.033$ the product stays under $10^{-2}$ across the whole sweep.

So the intuition that a multiphysics seam must couple *worse* than a single-physics one was not merely wrong, it was measuring the wrong thing: **conditioning at a seam is set by locality, not by whether the two sides solve the same equations.** A conjugate-heat-transfer seam at small Biot number is the easiest interface problem here for the same reason it is the least informative one.

## 14.3 W68 — closed, and the 2026-08-29 diagnosis was right and not sufficient

§13.6 diagnosed `elliptic_signature` as measuring **non-normality** where **globality** was wanted, and that stands: they coincide for a pressure projection and come apart for a self-adjoint operator. What that diagnosis predicted, though, is that measuring the shell's own block *per agent* rather than on the assembled seam would help. It does not — the shell's own block reads $\kappa = 1.0001$ too.

**The mechanism is one level lower.** The statistic was not blind, it was **swamped**: it was handed $5.0001\,I$ and asked about its spectrum. No threshold on $\kappa$ or on the asymmetry could have recovered anything, because the information was not in the matrix at the level those statistics look.

The fix is therefore **a precondition and a third outcome, not a third threshold**:

| $\omega$ | outcome |
|---|---|
| $\ge$ floor, non-normal / ill-conditioned | consistent with EMBEDDED |
| $\ge$ floor, near-normal / well-conditioned | consistent with EXPOSED or NONE |
| $<$ floor | **NO OPERATOR RESOLVED** — neither, and not a weaker form of either |

On the object that falsified the promotion, it now says the true thing: the shell reads `NO OPERATOR RESOLVED — identity defect 1.97e-05 is below the 0.001 floor`, in both the `embedded` and the `exposed` variants. The record's `elliptic_subsolve` is *neither corroborated nor contradicted*, which is exactly the epistemic state, and `EllipticSubsolve.UNKNOWN` continues to be the honest declaration for a black box.

**And it now gates something, on a quantity that deserves to gate.** `L4/operator-content` decertifies a seam carrying a thin block, naming $\beta$:

> $\beta$, $\kappa$, the null count and the passivity defect are all well behaved on such a block and all four are then properties of that coefficient — which means **every bound carrying $1/\beta$ is scaled by a declared constant rather than by a measured operator.**

That is a decertification and not a refusal, deliberately. The composition is not wrong and the numbers are not wrong; what is wrong is reading them as statements about the solvers. `window_ns` in `split-step` is unaffected and still reaches `admit` with $0$ refusals and $0$ decertifications.

## 14.4 W74 — the probe was linearizing about 0 K, and nobody had asked

Found while writing W69's `stencil_radius` test, which measured the shell's along-seam spread as $0$ where the standalone measurement said $5$. The difference was the base trace, and the base trace was **hard-wired to zero**:

```python
zero = respond(port_name, np.zeros(n))
cols.append((respond(port_name, step * d_k) - zero) / step)
```

For an **affine** expert this is exactly right and the choice is free: the derivative is the same everywhere, so the zero probe removes every trace-independent bias and costs nothing. The docstring said so, and justified the choice on *bias cancellation* — which is true — while saying nothing about the linearization point, which is the part that is not free.

It stops being free the moment the port variable has an **absolute origin**. `thermal_seam`'s THERM effort is a temperature in kelvin, so the zero trace is $0$ K: the gas was asked for its wall flux against a $0$ K wall, and the shell conducted against a $0$ K gas. Measured against the experts' own operating point:

| quantity | probe base $=0$ K | probe base $=$ operating point | ratio |
|---|---|---|---|
| $\beta$ (seam `cht`) | $4.8062$ | $0.3807$ | $\mathbf{12.6\times}$ |
| $\lVert S\rVert_F$ | $19.227$ | $1.523$ | $12.6\times$ |
| $\kappa - 1$ | $2.31\times10^{-4}$ | $7.16\times10^{-5}$ | $3.2\times$ |
| $\lVert S_{\text{heat}}\rVert / \lVert S_{\text{entropy}}\rVert$ (§13.3) | $94.96$ | $1199.0$ | $12.6\times$ |
| verdict | `admit-uncertified` | `admit-uncertified` | — |
| decertification set | 7 rules | **the same 7 rules** | — |

**The compiler's conclusions were robust and every number the certificate carried was not.** And the certificate's own `probe_state` field said `"duct, T_hot=900 K, T_wall=400 K"` the whole time — so the architecture's principle that *a certificate is valid at a state, never globally* was being recorded **with the wrong state**.

**The cause is not radiation.** Setting `radiate=False` changes the numbers by nothing to six digits. It is the declared bond itself: `PORT_SPECS[THERM]` pairs $(T,\ q_n/T)$ rather than the pseudo-bond $(T,\ q_n)$, chosen in §13.3 precisely so that effort times flow is a power — and dividing by $T$ is what makes the response nonlinear in the effort. **The pseudo-bond would have been affine.** Two independently correct choices with incompatible hidden assumptions, and nothing in the architecture noticed the collision.

The fix is `ExpertCapabilities.probe_base`, a `port -> trace` callable that both probe routes now use. Undeclared means zeros, which is what every record written before it does, so **every measurement taken before this reproduces bit-for-bit** — verified: `scripts/tier0_window_ns.py` from cold gives $6058$ of $6059$ numbers identical, the one difference being wall-clock seconds.

It is not a fourth unverifiable declaration: a base the expert cannot accept raises rather than broadcasting, and a base far from the operating point shows up in $\omega$ and in the $\varepsilon$ window rather than passing silently.

> **All of §13's `thermal_seam` numbers are superseded by this**, and the boxes are on the sections that carry them. The *claims* all survive — E3 still fails, $\tau$ is still `UNDEFINED`, the probe still runs, W66 is still a false pass — because every one of them is about a sign, a verdict or a ratio rather than about a magnitude.

> **Superseded in part by §15.1, 2026-08-29 — the fix was right and the field was on the wrong object.** `probe_base` is per *expert*, so the two sides of one seam name their own base and nothing compared them: `cht`'s gas linearizes at $400$ K and its shell at $900$ K, on one interface variable. Correcting to a *consistent* base moves $\beta$ from the $0.3807$ in the table above to $\mathbf{1.23807}$ at $\lambda^{*} = 371.97$ K, a further $3.30\times$ — and the $0.3757$ this seam reports is attainable at **no** admissible interface state, the whole admissible range being $[0.4200,\ 1.7689]$. The claims in this section all survive; the third correction to the same number does not.

## 14.5 W69 — closed by an audit, which found a third unverifiable declaration and a live rule bug

§12.1 said *"the architecture rests on exactly one unverifiable declaration"*. §13.3 made it two by adding `response_half`. The audit says it was **three the whole time**, and the third is the one with teeth:

| field | can a test falsify it? | what a false value does |
|---|---|---|
| `validity` | falsifiable, never verifiable | **decertifies** — already the correct verdict |
| `response_half` | no (§13.3's three legs) | L3/C9 **refuses** a disagreement; a *matching wrong pair* passes |
| `governing_family` | **no test existed at all** | E3 reads `holds` and $\tau$ is attributed under a hypothesis that does not hold — it **promotes** |

`governing_family` is a free-form string, compared across a seam by E3 and read by nothing else. Two agents sharing one false value make E3 pass. Nothing can test it; what `_test_governing_family` now does is make the gap **visible in the certificate**, because an untested field with no entry looks identical to a field nobody thought about.

**What the audit did close** is the adjacent and fixable set — fields that were untested but perfectly testable. Two are now tested, at one and two solves:

- **`deterministic`** — two calls on one trace. It sets the probe's finite-difference step through `reproducibility_floor`, so a false value contaminates every column of every block on every seam this expert joins. Every record in the vault declares `True` and none had ever been asked. All pass.
- **`stencil_radius`** — a delta on the seam, measuring along-seam spread. This is a **lower bound** on the stencil's reach, so it can falsify an under-declaration and never confirm one — `validity`'s shape, stated that way rather than dressed up.

### And the stencil test found a rule consulting two of the three fields it needs

`required_halo()` returned `stencil_radius × substeps_per_macro_step` and **never consulted `time_discretization`**, which sits in the same record. That product is an *explicit* agent's domain of dependence; one implicit macro-step inverts an operator coupling every cell to every other. The shell declares `stencil_radius=1, substeps=1` — honestly, they describe its Q1 element — so the method returned $1$ against a measured $5$-cell reach, and `_halo_rule` **refuses** an overlap below the number it returns. A too-small value buys a passing halo check for a contaminated blend.

It was latent: the shell is on a non-overlapping graph, and the only implicit agent on an overlapping graph here is `wind_farm_real`'s actuator disk, whose `stencil_radius` is $0$. That is also why the guard is on `radius >= 1` and not on the label — a zero-stencil algebraic closure couples nothing, so there is no dense inverse to be global and its reach is $0$ under either discretization. **The guard follows the derivation rather than preserving a verdict**, and `wind_farm_real` is unchanged at `admit-uncertified`, $0$ refusals.

This is a **mechanism-1** hole — a rule with several inputs and one it does not consult — found where mechanism 1 is supposed to be found, by auditing. But the audit only happened because contact said where to look.

## 14.6 W70 — closed, and reframed: the coupling needs no port, the OUTPUT needs disclosure

The 2026-08-29 framing was *"a thermoelastic body coupling is volumetric and the port algebra has no volumetric bond, and a sixth port type is not the fix."* The first half is right; the second half was aimed at the wrong target. **The coupling is not between two agents at all** — the shell's conduction and its elasticity are one expert's internals, and no bond of any kind belongs between them. Nothing needs a port.

Three measurements, and they do not all point the same way:

1. **The coupling is one-way.** `step_thermal(T, dt, h_in, T_gas_in, h_out, T_gas_out, radiate, T_inf)` takes no displacement, strain or stress. Temperature drives deformation and deformation never returns, so the thermal subsystem is closed and E7's passivity against `storage` is **not** wrong.
2. **The elastic energy is negligible.** $\tfrac12 u^{\mathsf T} K u$ is $2.0\times10^{-5}$ of the declared $\tfrac12 T^{\mathsf T} M T$ at the operating point ($2.8\times10^{4}$ J against $1.4\times10^{9}$ J), reaching only $2.5\times10^{-5}$ at $1200$ K.
3. **And the stress is more sensitive to the seam trace than the certified quantity is.**

| interface mode | $\delta\lVert u\rVert / \lVert u\rVert$ | $\delta\lVert\sigma\rVert / \lVert\sigma\rVert$ | $\delta\lVert\text{flow}\rVert / \lVert\text{flow}\rVert$ |
|---|---|---|---|
| 0 | $3.64\times10^{-3}$ | $8.53\times10^{-4}$ | $2.741\times10^{-3}$ |
| 1 | $7.11\times10^{-4}$ | $4.15\times10^{-3}$ | $2.746\times10^{-3}$ |
| 7 | $1.90\times10^{-5}$ | $5.54\times10^{-3}$ | $2.747\times10^{-3}$ |
| 15 | $3.01\times10^{-6}$ | $4.79\times10^{-3}$ | $2.752\times10^{-3}$ |

**(3) is the finding.** Displacement integrates and stress differentiates, so over the retained band the displacement's sensitivity **falls $1207\times$** while the stress's **rises $5.6\times$**, ending at $1.74\times$ the certified flow's. The high interface modes — exactly the ones a 16-mode truncation discards — drive the stress hardest and the displacement least. **The truncation argument that bounds the error in the certified flow points the wrong way for the uncertified stress.**

So what is missing is smaller than a port type and worse than a missing bond: **the record describes a strict subset of what the expert computes, and nothing in L1 says so.** A reader of the certificate cannot distinguish an expert that computes only what is certified from one that does not. `VOLUMETRIC_COUPLING_SCOPE` states it permanently, in `CROSS_POINT_COUPLING_SCOPE`'s form and for the same reason.

Left as a scope statement rather than a field on purpose: a disclosure field would be a **fourth unverifiable declaration**, since an expert that omits an output from its disclosure is exactly as undetectable as one that mis-declares `response_half`. Buying disclosure with another unverifiable is not obviously a trade worth making, and W69 is the reason to say so out loud.

## 14.7 W72 — the tab was an instance, and the instance is not the class

Adding one byte to a hand-written list leaves a hand-written list, which is what failed: W38 closed on 2026-08-27 naming `\t` among the bytes to scan for, and `vault_scan.py`, written the next day, implemented every byte in that list except that one.

Three changes make the list stop being hand-written:

1. **The escape table is derived from Python's own decoder** rather than typed out, so it cannot drift from what actually happens to a string literal.
2. **A symptom check that needs no list at all.** Two of the eight escape bytes are legal markdown and cannot be flagged on sight: `0x09` from `\t` (flagged anyway, because this vault uses no tabs) and **`0x0A` from `\n`** — which no byte-level rule can ever see, and which leads `\nu`, `\nabla`, `\neq`, `\notin`. So count `$` **per line**: every escape corruption inside inline math splits one delimiter pair across two lines, leaving each line with an odd count, *regardless of which byte was produced*. Fenced code, display blocks and code spans are excluded.
3. **A generated positive control**, one per entry in the derived table, asserting the scanner objects. A byte added with no detector is now a failing test rather than a silent hole.

### It found a live corruption on its first run

`spec-wind-farm-wake-atlas-0.1.md`, the **W11 go/no-go row**:

```
| **W11** | Global power residual | $\lvert\mathcal R(t)
vert$ below 1% of total extracted power, all steps, ... |
```

`\rvert` had been eaten, splitting the row and stopping the table rendering from there down. It had been in the vault since **2026-08-26**, through W37's two fixes and W38's rewrite, passing every scan — and it passed for three independent reasons, each of which is now closed:

- the `\r` arrived as a plain **LF**, not a lone CR, so the byte scan was right to see nothing;
- the split left a **two-line fragment**, below the ragged-table check's three-line minimum — so a corruption that splits a row *destroys the contiguity that check counts*, which is a check disabled by the thing it is looking for;
- the tail it left was **`vert`**, and the orphan list carried `\rVert` but not `\rvert`, a one-character case difference.

The `$`-parity check is what caught it, and `split_table_rows` — a line that opens a cell and never closes it, needing no minimum block size — now catches it a second way. One unescaped currency `$` was fixed in the same pass. The vault is at **190 files, 0 problems**.

> **The shape worth keeping.** This is the third time the standing verification has been the thing at fault rather than the content, and each time the fix was *the specific byte that bit* rather than the class. The class is: **a detector enumerated by hand will be missing an entry, and the entry it is missing is invisible precisely because it is missing.** The two defences that do not have this failure mode are a derivation and a symptom check, and both are now in place.

## 14.8 Where the compiles land, 2026-08-30

| graph | verdict | refusals | decertifications | unmeasured |
|---|---|---|---|---|
| `window_ns` split-step | **`admit`** | 0 | 0 | 0 |
| `window_ns` as-built | `refuse` | 1 | 3 | 1 |
| `channel_ns` | `admit-uncertified` | 0 | 5 | 4 |
| `wind_farm_real` | `admit-uncertified` | 0 | 6 | 5 |
| `thermal_seam` split-step, matched | `admit-uncertified` | 0 | 7 | 5 |

Unchanged from §13.8 except `thermal_seam`, which gains `L4/operator-content`. **285 tests pass** (256 before). `python scripts/vault_scan.py wiki` → 190 files, 0 problems. `python scripts/tier0_window_ns.py` reproduces bit-exact from cold.

> **Superseded by §15.5, 2026-08-29.** `window_ns` split-step is no longer `admit`: `L4/block-share` decertifies `sx0` and `sx1`, whose blocks differ by $3486\times$ in $\kappa$ under an assembled $\kappa$ of $1.196$. It keeps **zero refusals**. `thermal_seam` gains `L4/probe-base`, `wind_farm_real` gains five, and the test count is 314.

## 14.9 Closed this session / still open

### Closed

| hole | how |
|---|---|
| **W68** | not a better threshold — a **precondition** and a third outcome. The statistic was swamped, not blind: it was handed $5.0001\,I$. `operator_content` separates the cases at $2370\times$, and `L4/operator-content` decertifies the seam naming $\beta$ |
| **W71** | $\omega \approx C\,\mathrm{Bi}\,(k_{\max}\ell)^2$, $C \in [0.135, 0.307]$ over 11 points and 5 decades. Conditioning at a seam is set by **locality**, not by whether the two sides solve the same equations |
| **W69** | audited: **three** unverifiable declarations, not one — `governing_family` was the third and had no entry at all. `deterministic` and `stencil_radius` are now tested, and the stencil test found `required_halo()` ignoring `time_discretization` |
| **W70** | reframed and measured: the coupling is one-way and needs no port; the **output** needs disclosure, and its sensitivity to the seam *rises* with mode number where the certified quantity's is flat |
| **W72** | derived escape table, a `$`-parity symptom check that needs no list, a generated positive control per byte — and **one live corruption**, in the vault since 2026-08-26 |

### Opened

| hole | statement |
|---|---|
| **W74** | the probe linearized about the **zero trace**, which on a kelvin port is $0$ K. $\beta$ moves $12.6\times$. Fixed by `probe_base`, but the class is open: **the declared bond $(T, q_n/T)$ that makes the power pairing right is what makes the response nonlinear**, so W66's fix and the probe's affine assumption were in conflict and nothing detected it |
| **W75** | `elliptic_signature` still has no route to a verdict on a **thin** block, and `EllipticSubsolve` therefore still cannot be measured for any expert whose seam is film-dominated. The honest gate would need a probe at a **different exchange interval** — $\Pi \gtrsim 10^{-2}$ — which is a cadence the composition may not be able to run |
| **W76** | the two blocks of `window_ns`'s seam `sx0` differ by $21\times$ in norm and $20\times$ in $\omega$, resolved and stable, between two instances of the same solver on a symmetric tiling. Nothing reads per-block $\beta$ or $\kappa$; the assembled $\kappa = 1.196$ shows none of it |
| **W77** | the certificate's `probe_state` is a **free-form string supplied by the caller** and W74 is the proof it can be wrong while everything else is right. It is a fourth undetectable declaration in everything but name |

> **All four of these were closed on 2026-08-29 — see §15.6 — and two of them not as stated here.** **W75** said the resolving cadence *"may be unreachable"*: it is reachable, three cadences over, and the finding is instead that `elliptic_signature` cannot decide the field at *any* cadence, ranking a known implicit/explicit pair backwards at $\Delta t = 100$. The gate is a delta poke. **W76** is understated above: the two blocks of `sx0` differ by $3486\times$ in $\kappa$ and $22{,}500\times$ in $\beta$, not only $21\times$ in norm, and the consequence is that the substitution certificate cannot see an under-responding replacement.

### Unchanged, and worth saying so

- **W59** stays open with its trigger identified (§13.7); no second learned expert arrived this session either.
- **W53, W58, W62, W65** untouched.
- §13.5's W57 result re-ran unchanged: $+0.5794$ to $+1.0000$ over 11 configurations, product form negative in $9/11$, cost $\le 1.105\times$.

> **The mechanism tally, four sessions in.** §12.5 predicted that mechanism 1 is found by auditing and mechanisms 2–4 only by contact with an independent object. This session ran on §13's leftovers rather than on a new object and the prediction held in a way worth recording: **every hole closed here was reachable by auditing, and not one of them had been found that way.** W69's `required_halo` bug is textbook mechanism 1 and sat through four audits; what made it findable was a measurement taken for another purpose entirely. Auditing finds mechanism 1 **once you know which rule to audit**, and contact is what tells you.

# 15. What the probe assumed about its own base — 2026-08-29

Reproduce with `python scripts/w78_base_blocks_and_reach.py`; artifact `out/w78/w78.json`. Tests: `tests/test_tier15_base_blocks_and_reach.py`.

> **A dating correction, and it is a records defect rather than a typo.** §14 and its log entry are stamped `2026-08-30`. They were written on **2026-08-29** — the system clock, the environment date and the mtimes of `atlas/probe.py`, `scripts/w73_locality_and_scope.py` and `wiki/log.md` all say `2026-08-29 14:37` local. The stamp was one day fast. Nothing is renamed here, because the log is append-only and rewriting it would hide the drift rather than record it; §14 keeps its stamp and this box is the correction. **This section is dated by the clock, so the log now reads 08-30 then 08-29 and the ordering is by append, not by date.**

§14 closed W74's *instance* — the probe linearized about the zero trace, which on a kelvin port is $0$ K — and left the class open in one sentence: *nothing detects that a port spec's own choice of bond has invalidated a probe assumption*. Chasing that closed W77 with it, inverted W75, and found that the field §14 added was **on the wrong object**.

## 15.1 The class, and where the field belongs

The seam operator is a sum,

$$\Lambda_M \;=\; \sum_i P_i^{*}\,\Lambda_i\,P_i ,$$

and each $\Lambda_i$ is a Jacobian. **A sum of Jacobians is a Jacobian only if every term was taken at the same point.** §14 put `probe_base` on `ExpertCapabilities`, which is one record per *expert*, so each side of a seam names its own base and nothing compared them. The interface variable is one variable: both sides see the same trace $\lambda$.

On `thermal_seam` the two records are each honest, each independently correct, and **500 K apart**:

| side | declared `probe_base` | what it is |
|---|---|---|
| gas | $400$ K | `T_WALL_0`, the wall temperature the gas sees |
| shell | $900$ K | `T_HOT`, the gas temperature the shell sees |

Each side declares the temperature *the other side presents to it*, which is the natural thing to write and is right for that side in isolation. Assembled, they are two linearizations of one variable at points 500 K apart.

### The consequence is not a shifted number

Sweeping a **common** base over the physically admissible interval — the interface temperature is trapped between the two reservoirs, $T_{\text{OUT}} = 250$ K and $T_{\text{HOT}} = 900$ K:

| $\lambda$ [K] | $\lVert S_{\text{gas}}\rVert$ | $\lVert S_{\text{shell}}\rVert$ | $\beta$ |
|---|---|---|---|
| $250$ | $0.47346$ | $7.55066$ | $1.76887$ |
| $400$ | $0.37050$ | $4.97125$ | $1.14974$ |
| $650$ | $0.26058$ | $2.87807$ | $0.65395$ |
| $900$ | $0.19307$ | $1.87463$ | $0.42002$ |

$\beta$ runs over $[0.42002,\ 1.76887]$ across the whole admissible interval, and the mismatched-base probe — the one the certificate carries — reports

$$\beta_{\text{reported}} \;=\; 0.37565 ,$$

which is **below the entire range**. So it is not the right quantity at the wrong point; it is a quantity attained at *no* admissible point. `test_the_reported_beta_is_not_attainable_at_any_admissible_base` asserts exactly that.

### The consistent base, and what it costs

The state the coupled step actually reaches is where the two one-step fluxes balance:

$$q_{\text{gas}}(\lambda^{*}) + q_{\text{shell}}(\lambda^{*}) = 0 \qquad\Longrightarrow\qquad \lambda^{*} = 371.97\ \text{K}$$

and there $\beta = 1.23807$ — **$3.30\times$ the reported number**, after §14 had already moved it $12.6\times$. Compounded, the $\beta$ this seam carried on 2026-08-28 was $4.8062$ against a consistent-base $1.238$: a factor of $3.9$, in the opposite direction from §14's correction, because the two errors do not compose monotonically.

$\omega$ at $\lambda^{*}$ is $3.27\times10^{-4}$, still below `OPERATOR_CONTENT_FLOOR`, so **§14's W68 decertification stands unchanged** — the film still dominates, and correcting the base does not rescue the operator content.

### The check whose passing removes the finding

`probe.base_sensitivity` re-probes at a displaced base and reports $\lVert S(b+\Delta) - S(b)\rVert / \lVert S(b)\rVert$. Below `BASE_SENSITIVITY_FLOOR` $=10^{-6}$ the response is **affine**, the base is provably free, and every base-related decertification goes away. It is the one check in this vault whose *passing* promotes.

| response | base moved | relative change | affine? |
|---|---|---|---|
| `linear_response` fixture | $+50\%$ | $5.40\times10^{-11}$ | **yes** |
| `linear_response`, base $0$ | $+50\%$ | $1.13\times10^{-15}$ | **yes** |
| gas | $+1\%$ | $6.13\times10^{-3}$ | no |
| gas | $+10\%$ | $5.88\times10^{-2}$ | no |
| shell | $+1\%$ | $1.37\times10^{-2}$ | no |
| shell | $+10\%$ | $1.26\times10^{-1}$ | no |

Three orders of margin either side of the floor. And the nonlinearity is the bond, exactly as §14.4 said: $(T,\ q_n/T)$ divided by $T$ is not affine in $T$, and `test_a_reciprocal_response_is_base_dependent_and_that_is_the_therm_bond` reduces the whole mechanism to four lines of arithmetic — $q/T$ is base-dependent, $5T$ is not.

**This is distinct from $\varepsilon$-stability, and §14 measured that and drew no conclusion from it.** All four blocks were stable to $1.0003\times$ over four decades of probe step. That measures *curvature at the base*; base sensitivity measures whether the base is the right place. A response can be locally linear to five digits and still be linearized 500 K from anywhere the system goes.

`assemble_seam` now takes `seam_base`, an interface state on $M$ prolonged to both sides. Omitting it keeps the per-expert behaviour — which is what every record written before today does, so **every earlier measurement reproduces bit-for-bit** ($6107$ numbers compared from cold, $0$ drifted) — and `base_disagreement` reports whether the sides agreed. `L4/probe-base` decertifies when they did not.

## 15.2 W75 — my own hypothesis, falsified, and then the gate found elsewhere

§14 opened W75 saying the honest gate *"may be unreachable: the cadence that would resolve the operator is not necessarily one the composition can run"*. Two requirements pull opposite ways in one parameter: resolving the operator wants $\Delta t$ **large** (so $\omega \ge$ floor), keeping the linearization valid wants it **small**.

| $\Delta t$ [s] | $\omega$ | resolves? | $\delta T_{\text{face}}$ | base sensitivity | linearizes? |
|---|---|---|---|---|---|
| $10^{-2}$ | $7.66\times10^{-5}$ | no | $-0.06$ | $1.64\times10^{-4}$ | yes |
| $5\times10^{-2}$ | $3.03\times10^{-4}$ | no | $-0.17$ | $4.47\times10^{-4}$ | yes |
| $1$ | $6.94\times10^{-3}$ | **yes** | $-1.10$ | $2.94\times10^{-3}$ | **yes** |
| $10$ | $6.22\times10^{-2}$ | **yes** | $-7.20$ | $2.03\times10^{-2}$ | **yes** |
| $100$ | $2.16\times10^{-1}$ | **yes** | $-24.25$ | $7.20\times10^{-2}$ | **yes** |

**Three cadences satisfy both, so the hypothesis is falsified.** The windows are not disjoint; they overlap by two orders of magnitude.

### And it does not matter, because the instrument is wrong

Run `elliptic_signature` at those cadences on the *same shell under two solvers* — `ThermoStruct2D` backward Euler against `_explicit_shell_class()` sub-stepped, same mesh, same material, same film coefficient, so the only difference is whether the elliptic part is embedded:

| $\Delta t$ | implicit $\kappa$ | explicit $\kappa$ | asymmetry |
|---|---|---|---|
| $1$ | $1.0214$ | $1.0210$ | $\sim 5\times10^{-6}$ |
| $10$ | $1.2311$ | $1.2649$ | $\sim 5\times10^{-6}$ |
| $100$ | $3.1965$ | $\mathbf{9.6687}$ | $\sim 5\times10^{-6}$ |

Indistinguishable at $\Delta t = 1$ and **ranked backwards** at $\Delta t = 100$ — the explicit solver reads the higher $\kappa$. The asymmetry is flat at $5\times10^{-6}$ throughout and carries no signal at any cadence. **W75 does not close by widening the exchange interval.**

### The reason is the probe basis, not the statistic

A Fourier mode is global by construction. A block assembled from smooth modes has no way to report whether the operator behind it was local, so no function of that block can decide a question about *support*. `EllipticSubsolve` is defined in its own docstring as *"a solve with an INFINITE domain of dependence"* — which is a statement about support and was being chased through the spectrum.

**Poke a delta instead.** One implicit macro-step inverts $(M/\Delta t + K)$, and the inverse of a sparse SPD matrix is **dense**: every seam cell responds to every seam cell. An explicit march's response is *exactly* zero past `radius * substeps` — arithmetic zero, not small. Five known points, two of them one shell under two solvers with everything else fixed:

| case | declared | cells responding | fraction | reading |
|---|---|---|---|---|
| shell implicit, $\Delta t = 0.05$ | `embedded` | $48/48$ | $1.000$ | **global** |
| shell explicit, $\Delta t = 0.05$ | `none` | $9/48$ | $0.188$ | not global |
| gas, explicit CFL march | `none` | $1/48$ | $0.021$ | not global |
| `window_ns` as-built | `embedded` | $136/138$ | $0.986$ | **global** |
| `window_ns` split-step | `exposed` | $29/138$ | $0.210$ | not global |

**Every declaration corroborated, at two solves each** — the same two `_test_support_radius` already spends — with no threshold on any spectral quantity. `conformance._test_elliptic_subsolve` is the gate, and it *resolves* `EllipticSubsolve.UNKNOWN`, which is what W60 asked for on 2026-08-29 and W68 refused to grant from the spectrum. The difference is not strictness: reach measures the property the enum is defined by, where $\kappa$ measures a correlate that comes apart for a self-adjoint operator.

### Its operating range, which is bounded on both sides

Only the **global** reading is a positive measurement. A genuinely dense response whose far tail underflows to zero is indistinguishable from a compact one, and this is not hypothetical — it happened on the first real run:

| the same implicit shell | $\Delta t$ | fraction |
|---|---|---|
| native clock | $5\times10^{-2}$ | $1.000$ |
| **matched clock** | $10^{-4}$ | $\mathbf{0.875}$ |

At $\Delta t = 10^{-4}$, $\sqrt{\alpha\Delta t}$ is $60\times$ below one seam cell and $e^{-d/\ell}$ underflows past about 21 cells. The verdict there is `admit-uncertified`, not a contradiction — the conservative direction, since L2 refuses on `embedded` anyway. At the other end the explicit competitor fills the seam too ($0.771$ at $\Delta t = 1$), so the discriminating window is bounded above as well.

**So the two instruments are complements, not alternatives.** The spectral gate needs a long interval and fails there anyway; the spatial gate needs a short one and works. That is the closing statement on W60, three sessions after it opened.

## 15.3 W76 — what the assembly hides, and a certificate that cannot fail

Per-block $\beta$, $\kappa$, `null_dim` and `alpha_star` have been emitted since Tier 0 and read by **no rule**; `alpha_star` — the measured symbol — by nothing anywhere.

### Non-sufficiency is a fact, not a measurement

$$S_A = \operatorname{diag}(1,\ 10^{-12}), \qquad S_B = \operatorname{diag}(0,\ 1), \qquad S_A + S_B = \operatorname{diag}(1,\ 1 + 10^{-12})$$

The sum's every diagnostic is ideal and $S_A$ is singular to $10^{-12}$. **No function of the assembled matrix can recover that**, so the question of whether assembled diagnostics suffice for the block ones is settled in two lines and needs no case study.

### On the real seams

| seam | assembled $\beta$ | assembled $\kappa$ | worst block $\kappa$ | ratio | min share |
|---|---|---|---|---|---|
| `sx0` | $0.348568$ | $1.1964$ | $\mathbf{4170.92}$ | $\mathbf{3486\times}$ | $0.167$ |
| `sx1` | $0.376225$ | $1.1051$ | $1574.25$ | $1425\times$ | $0.167$ |
| `sy0` | $0.277344$ | $1.3563$ | $1.6098$ | $1.2\times$ | $0.494$ |
| `sy1` | $0.288128$ | $1.3317$ | $1.5179$ | $1.1\times$ | $0.481$ |

§14 recorded that `sx0`'s blocks *"differ by $21\times$ in norm and $20\times$ in $\omega$"*. They also differ by $\mathbf{3486\times}$ in $\kappa$ and $\mathbf{22{,}500\times}$ in $\beta$ — block W10 is $\beta = 1.67\times10^{-5}$ against an assembled $0.349$ — and the assembled $\kappa = 1.196$ shows none of it. Read off `alpha_star`, the dominant side holds a median $0.987$ of the response on `sx0` at **every one of 16 modes above 0.90**, against a median $0.543$ and no mode above 0.90 on `sy0`. The vertical seams are one-sided because the *flow* is: the downstream window barely responds to its own inlet ring.

### The derivable consequence

A substitution certificate compares two **assembled** operators and passes when $\lVert\Delta\rVert < \beta - \beta_{\min}$. Replacing agent $i$ changes only block $i$, so a replacement that **under**-responds moves the seam by at most that agent's own block:

$$\lVert\Delta\rVert \;\le\; \lVert S_i^{\text{old}}\rVert \quad\text{for any replacement with } S_i^{\text{new}} = 0 .$$

If $\lVert S_i\rVert < \beta - \beta_{\min}$, **no such failure can be caught** — including an expert that ignores its boundary data entirely, which is precisely the `bc_channel` failure `conformance.py` calls the foundational hole. Measured, by replacing one agent with exactly that:

| seam | agent | $\lVert\Delta\rVert$ | $\beta_{\min} = 0.05$ | $\beta_{\min} = 0.10$ | $\beta_{\min} = 0.20$ | $\beta$ moves |
|---|---|---|---|---|---|---|
| `sx0` | W10 (16.7%) | $0.0696$ | blind | blind | **blind** | $0.7\%$ |
| `sy0` | W01 (49.4%) | $0.1860$ | blind | `REFUSE` | `REFUSE` | $45.7\%$ |

The one-sided seam is blind at every threshold tried; the balanced one starts working at $0.10$. **And nothing in this framework derives $\beta_{\min}$** — `certify_substitution` takes it from the caller with no default, and the one sibling default beside it is $10^{-12}$, at which every seam in the vault is blind.

The fix goes where the claim is made: `SubstitutionCertificate` gains `block_norm` and a `blind` property, and a pass that could not have been a failure is downgraded from `admit` to `admit-uncertified`. The compiler's `L4/block-share` **discloses** the thresholds without charging them against the verdict, because a compile makes no substitution claim — and it *decertifies* only for what the assembly genuinely hides, a block more than $100\times$ worse conditioned than the operator it sums into.

> A note on the word *share*: it is $\lVert S_i\rVert_2 / \lVert S\rVert_2$, a ratio and not a partition, and it can exceed $1$ when the two blocks partly cancel — `thermal_seam`'s shell reads $1.243$. No seam here cancels strongly enough for that to matter, which is why it is a note and **W82** rather than a result.

## 15.4 W77 — the state, derived instead of declared

`probe_state` was a free-form string the caller handed in, and §14.4 is the proof it can be wrong while everything else is right: it read `"duct, T_hot=900 K, T_wall=400 K"` on every `thermal_seam` certificate ever emitted while the probe was linearizing at $0$ K.

It never needed to be declared. **The probe knows the base it used, and the base *is* the state the linearization is about.** `SeamOperator.derived_probe_state` is built from it:

```
declared : 'duct, T_hot=900 K, T_wall=400 K'
derived  : gas@400; shell@900 INCONSISTENT(spread 223.6)
agree?     False
```

The declared string was not even wrong about the *values* — both numbers are in it. It was wrong about there being **one** of them, which is the defect of §15.1 showing up in the label.

Two details that matter. The base is compared on $M$, because that is the common space, and **reported** in $V$: a uniform $400$ K trace reduces to $11.18$ in a measure-weighted Fourier basis, so $M$ is the right space to compare in and the wrong one to read. And this **removes** a declaration rather than adding a fourth to W69's three — the field it replaces was never checked against anything.

## 15.5 Where the compiles land, 2026-08-29

| graph | verdict | refusals | decertifications | change |
|---|---|---|---|---|
| `window_ns` split-step | `admit-uncertified` | 0 | 2 | **was `admit` 0/0** — `L4/block-share` on `sx0`, `sx1` |
| `window_ns` as-built | `refuse` | 1 | 4 | $+1$, `block-share` on `sx0` |
| `channel_ns` | `admit-uncertified` | 0 | 6 | $+1$ |
| `wind_farm_real` | `admit-uncertified` | 0 | 11 | $+5$, on seams `e2`–`e5`, `e7` |
| `thermal_seam` split-step, matched | `admit-uncertified` | 0 | 8 | $+1$, `L4/probe-base` |

**The vault no longer has a clean `admit`, and that is the honest price of this tier.** What `window_ns` split-step lost is not its composition — $\sigma$ and $L$ rest on the assembled $\beta$, which is sound, and it still has **zero refusals** — but the claim that its per-agent substitutions mean anything. That claim was never true.

`wind_farm_real` firing on five of its seams is corroboration rather than noise: a wake is one-way, so one-sidedness tracks the advection direction, and the case where the mechanism should be strongest is the case where it fires most.

**314 tests pass** (285 before). `python scripts/tier0_window_ns.py` from cold: $6107$ numbers compared, **$0$ drifted**. `python scripts/vault_scan.py wiki` → 0 problems.

## 15.6 Closed this session / still open

### Closed

| hole | how |
|---|---|
| **W74** (class) | the field was on the wrong **object**. A base belongs to a seam, not an expert; the two sides of `cht` are $500$ K apart and the reported $\beta = 0.3757$ is attainable at no admissible base. `seam_base`, `base_disagreement`, `base_sensitivity` and `L4/probe-base` |
| **W75** | my own "may be unreachable" is **falsified** — three cadences satisfy both requirements. The spectral gate fails there anyway and ranks a known pair backwards, because a Fourier basis cannot report support. The gate is a **delta poke**, threshold-free, right on all five known points, and it resolves `UNKNOWN` |
| **W76** | non-sufficiency proved in two lines; measured at $3486\times$ in $\kappa$ on a real seam; and the derivable consequence is that the substitution certificate is **blind** to an under-responding replacement, demonstrated by certifying an expert that ignores its boundary data |
| **W77** | derived from the base rather than declared, so it cannot disagree with the measurement. One fewer declaration, not a fourth |

### Opened

| hole | statement |
|---|---|
| **W79** | the support gate has an **operating range in $\Delta t$** — bounded below by tail underflow ($0.875$ on a known-global shell at the matched clock) and above by an explicit competitor filling the seam ($0.771$ at $\Delta t = 1$) — and **nothing computes whether a given seam is inside it**. The gate is sound and its applicability is undeclared, which is the shape of the thing W68 had to fix for `elliptic_signature` |
| **W80** | `assemble_seam` now *accepts* a seam base and still **defaults to the inconsistent per-expert one**, so W74's class is reported rather than repaired. Deriving $\lambda^{*}$ took a 50-step bisection — order 100 solves — which is the interface solve probing exists to avoid, so the honest fix may be a declaration after all, and that would be a fourth unverifiable |
| **W81** | $\beta_{\min}$ is **undefined anywhere in this framework**. It is the entire discriminating power of the plug-in guarantee and it is a caller's free parameter; the blind-spot check reports the threshold above which a substitution becomes visible and nothing derives what it should be |
| **W82** | block *share* is a ratio, not a partition, and exceeds $1$ under cancellation (shell: $1.243$). Whether strong cancellation between blocks is itself a conditioning defect is **unmeasured** — no seam here cancels hard enough to say |

### Unchanged

- **W53, W58, W59, W62, W65** untouched; W59 still open with its trigger identified (§13.7).
- §14's W68/W71 results stand: correcting the base moves $\omega$ to $3.27\times10^{-4}$, still below the floor, so the film still dominates and `L4/operator-content` still fires.

> **The mechanism tally, and this session inverts §14's own lesson.** §14 concluded that *auditing finds mechanism 1 once you know which rule to audit, and contact is what tells you which*. This session had no new object at all — it ran on §14's own leftovers — and still found three defects by **contact with a measurement taken for a different purpose**: the base disagreement fell out of writing W74's class detector, the spectral gate's backwards ranking fell out of testing W75's reachability, and the substitution blindness fell out of asking what per-block $\kappa$ is *for*. What did not work even once was reading the code and thinking about it. **The generalization is not about new objects; it is that a rule becomes auditable only after some measurement has named it.**

# 16. Multiphysics: what E3 was actually gating — 2026-08-29

Reproduce with `python scripts/w83_multiphysics_attribution.py`; artifact `out/w83/w83.json`. Tests: `tests/test_tier16_multiphysics.py`. New module: `atlas/multiphysics.py`.

**The standing reason no multiphysics graph in this vault could be certified.** `E3` asks whether the two sides of a seam declare the same `governing_family` — a string comparison — and when they do not, the compiler emits $\tau$ as `UNDEFINED` for both sides. `rocket.py`, seven agents with a genuine fluid–structure seam, is kept as a *fixture rather than a target* partly for this. `thermal_seam` has never done better than `admit-uncertified` with $\tau$ undefined at its only seam.

## 16.1 What $\tau$ is actually measured as

From `scripts/tier0_window_ns.py`, unchanged since Tier 0:

$$u_{\text{ref}} = \text{the reference trajectory}, \qquad
\tau = \lVert u_t - u_{\text{ref}}\rVert, \qquad
\sigma = \lVert u_c - u_t\rVert$$

with $u_t$ the composed step given the **true** trace and $u_c$ the composed step with the **lagged** one. **Nothing in that mentions a governing equation.** What it requires is a *reference trajectory*.

At a multiphysics seam one exists and is constructible from the agents themselves: the **tightly coupled** solve, in which the interface condition is converged *inside* the macro-step instead of lagged across it. That is the exact analogue of the single-physics monolith, which is likewise not ground truth but "the same expert applied without the cut".

> So E3 was gating the wrong thing. Sharing a governing family is what lets two agents share a **monolithic** reference; $\tau$ needs only a **reference pair**. And `lambda_ref` — a field on the record since the end-to-end spec, documented in `capability.py` as *"tau at a multiphysics seam"* — was the declaration that one exists. It was checked for presence and **never consumed**.

## 16.2 Does an attribution attribute?

The test is a surrogate carrying a *known* error, so what $\tau$ recovers can be checked against it rather than merely being plausible. The gas is replaced by the same solver with its wall conductivity scaled, the shell is left alone, and the reference pair is the two real solvers:

| gas expert | $\tau_{\text{gas}}$ | $\tau_{\text{shell}}$ | injected |
|---|---|---|---|
| reference (itself) | $0.0000$ | $0.0000$ | $0\%$ |
| $k_{\text{gas}}\times1.05$ | $\mathbf{5.0000\times10^{-2}}$ | $0.0000$ | $5\%$ |
| $k_{\text{gas}}\times1.25$ | $\mathbf{2.5000\times10^{-1}}$ | $0.0000$ | $25\%$ |
| $k_{\text{gas}}\times2.00$ | $\mathbf{1.0000}$ | $0.0000$ | $100\%$ |

Exact, per agent, with the **unswapped** side identically zero. Attribution works and it never needed the families to match. The same three rows in the pure algebra (`test_tau_recovers_an_injected_error_exactly`) agree to $10^{-9}$.

For the real pair $\tau = 0$ exactly — these *are* the actual solvers, so each is its own reference and its infidelity is zero by construction. That is worth measuring rather than assuming, and it is the number the certificate should carry. **[AI Inference]:** the case this matters for is the one this vault exists for — swap a learned expert onto one side and $\tau$ becomes that expert's infidelity, measured at the seam, without any monolith of its class ever existing.

## 16.3 The second problem, and it is the one that needed a module

The master bound adds $\tau + \sigma + \gamma$ as **scalars**, which presumes a single norm. At a multiphysics seam there is none: the gas's state is a field of conserved variables, the shell's is a temperature in kelvin, and there is no scalar sum of the two.

Worse, an agent's own state norm can be **blind**. Measured on the gas, one macro-step of $10^{-4}$ s:

| wall temperature | $\lVert U_{\text{gas}}\rVert$ | $\lVert \text{flow}\rVert$ |
|---|---|---|
| $352.378$ K | $4.9122462491\times10^{6}$ | $2.634\times10^{2}$ |
| $400$ K | $4.9122462491\times10^{6}$ | $2.317\times10^{2}$ |
| $450$ K | $4.9122462491\times10^{6}$ | $2.008\times10^{2}$ |
| $500$ K | $4.9122365770\times10^{6}$ | $1.721\times10^{2}$ |

**Bit-identical across 100 K** while the flow moves 24%: the isothermal wall enters through a ghost state the interior has not felt in one step. A defect measured in that norm reads exactly zero.

The port algebra already carries the common currency. Every entry of `PORT_SPECS` pairs its halves so that **effort times flow is a power** — which is precisely why `THERM`'s bond is $(T,\ q_n/T)$ and not the pseudo-bond $(T,\ q_n)$, the choice §13.3 made deliberately and W66 rests on. Power is a unit neither side owns, both sides agree on, and every port type already has. So the defect is measured in **interface power**:

$$P(\lambda) \;=\; \int_\Gamma e(\lambda)\, f(\lambda)\, \mathrm{d}\Gamma$$

and it is then commensurable and summable. This inherits `response_half` — one of W69's three unverifiable declarations — because nothing distinguishes an effort from a flow in the returned numbers, and `interface_power` **refuses** rather than guessing when it is `UNDECLARED`.

> **The bond that W66 chose for a thermodynamic reason turns out to be the one that makes multiphysics attribution possible at all.** The pseudo-bond $(T, q_n)$ is not a power and would not have been a common norm. That is the second time this choice has paid — and the first time (W74) it cost, by making the response nonlinear in the effort.

## 16.4 The instrument, and what it costs

`multiphysics.tight_couple` converges the interface inside the macro-step. It is a **measurement instrument, not a scheme** — converging the interface is exactly what the composition avoids — and it is run once to buy the referent, the way Tier 0 runs a monolith it would never ship.

**A scalar secant is not enough, and the failure is structural.** Iterating on the mean residual moves only a *uniform* shift of the trace, so the part of the residual that varies along the seam is untouchable:

| iteration scheme | result |
|---|---|
| scalar secant on the mean | **stalls** at $3.04\times10^{-3}$, from $2.32\times10^{2}$ |
| Newton on a finite-difference Jacobian | $1.32\times10^{-7}$ in **9** iterations |

Reproduced in the algebra by a residual with a non-uniform root, where the secant cannot converge at any step size. The Jacobian costs $n+1$ residual evaluations, so $2(n+1)$ solves for a two-sided seam — **W85**.

## 16.5 $\sigma$ is not a number, it is a function of the lag

Measured at the initial condition, $\sigma = 1.0000$ **exactly** — which is a fact about the lag and not about the coupling. The lagged trace is `T_WALL_0`, the shell's own initial temperature, so at that trace the shell sees no gradient and transmits *zero* power while at $\lambda^{*}$ it transmits all of it. That is the degenerate first macro-step from a uniform initial condition.

A real run lags on the **previous step's converged trace**:

| lag [K] | $\lvert \text{lag} - \lambda^{*}\rvert$ | $\sigma$ | ratio |
|---|---|---|---|
| $400.000$ | $27.86$ | $1.0000$ | — |
| $352.138$ | $20.00$ | $6.6872\times10^{-1}$ | — |
| $367.138$ | $5.00$ | $1.7119\times10^{-1}$ | $3.91$ per $4\times$ |
| $371.138$ | $1.00$ | $3.4447\times10^{-2}$ | $4.97$ per $5\times$ |
| $372.038$ | $0.10$ | $3.4494\times10^{-3}$ | $9.99$ per $10\times$ |
| $372.138$ | $0.00$ | $1.0773\times10^{-11}$ | — |

$\sigma \to 0$ at zero lag, which is the check that it measures the lag and nothing else. It is linear **to leading order only**: interface power is *bilinear*, so there is a quadratic correction, and the ratios drift from $9.99$ to $3.91$ as the lag grows. **A $\sigma$ quoted without its lag is not a number** — the interface here drifts $9.83\times10^{-4}$ K per macro-step, and $3.44\times10^{-2}\text{ K}^{-1} \times 9.83\times10^{-4}$ K $= 3.38\times10^{-5}$ predicts the measured $\sigma = 3.392\times10^{-5}$.

## 16.6 W84 — a rule that degenerated the moment $\tau$ could be zero

`eps_tol` $= \min(\tau, \sigma)$, and $\tau$ is now $0$ for a composition of exact solvers. That sets the interface tolerance to **zero**, which no solve can meet.

It had never come up: $\tau$ was `UNDEFINED` across a family boundary and nonzero within one, so no term had ever been exactly zero. The rule's own justification names the fix — it exists so the tolerance is not driven *"far below"* the defects that dominate, and a term that is identically zero names no scale at all. The minimum is now taken over the terms that carry one, with the zeroed term reported; both zero decertifies rather than emitting an unmeetable tolerance.

## 16.7 Where the compiles land, and what it does not fix

| graph | verdict | refusals | decertifications | change |
|---|---|---|---|---|
| `thermal_seam` split-step, matched | `admit-uncertified` | **0** | **6** | was 8; `L1/E3` now **admits**, `L5/eps_tol` clears |
| `thermal_seam` as-built, matched | `refuse` | 1 | 6 | was 8 |
| `window_ns` split-step | `admit-uncertified` | 0 | 2 | unchanged |
| `wind_farm_real` | `admit-uncertified` | 0 | 11 | unchanged |

`tau_undefined_seams` is **empty** for the first time on a multiphysics graph, and `unmeasured` falls from 5 to 3 ($L$, $C_\mu$, `norm_A` remain — ordinary per-case work). E3 itself still **fails**, and always will: the two sides really do solve different equations. What E3's failure costs is the *monolithic* reference and nothing else.

**This closes one of the four multiphysics blockers, and it was the foundational one.** Against `rocket.py`'s own list of five compile failures:

| blocker | status |
|---|---|
| L1/L4 — different governing families, $\tau$ `UNDEFINED` | **closed** — §16.1–16.3 |
| L7 — multirate, R9 refuses unless the flux is matched time-integrated | **open** — `thermal_seam` at native clocks (500:1) still refuses on E4 |
| L2 — the combustion front and plume boundary are not static → `InterfaceMotion` | **open**, and it is research |
| L3 — a lumped trajectory expert coupling to field agents, $\dim M = 1$ | **open** |
| staging → E1 → `TopologyEvent` | **open** |

## 16.8 Opened by this tier

| # | Where | What | Status |
|---|---|---|---|
| **W84** | `atlas/compiler.py` | `eps_tol = min(tau, sigma)` degenerates to $0$ when a term is exactly zero, which first became possible when $\tau$ became measurable at a multiphysics seam. Corrected to a minimum over the scale-bearing terms — but **`min` itself was never derived**, and the right combinator for two defect terms in the same norm is still a choice rather than a theorem | `open` for the derivation; the degeneracy is fixed and tested |
| **W85** | `atlas/multiphysics.py` | the referent costs $2(n+1)$ solves per Newton iteration — $98$ here, and for a learned expert at $128^2$ prohibitive. The probed operator $S$ is the same object compressed to $M$ and is the obvious cheap substitute, but **$M$ is chosen to make transmission cheap, not to converge a Newton solve** | `open` — and it decides whether multiphysics attribution is affordable for the learned experts this vault is for |
| **W86** | `atlas/graph.py` | $\sigma$ is a function of the lag and the certificate carries **one number with the lag in a provenance string**. Nothing checks that the run's actual lag matches the one $\sigma$ was measured at, and $\sigma$ moves $30{,}000\times$ across the measured range | `open` — the same shape as W77's `probe_state` before it was derived, and it may admit the same fix |
| **W87** | `atlas/conformance.py` | **`lambda_ref` is now load-bearing and has no test.** It gates whether $\tau$ is `UNDEFINED`, and declaring it falsely reports $\tau$ as measurable when no referent can be built. It is falsifiable — supply the pair and run — which is `validity`'s shape exactly, so it is a *fourth* declaration of that kind | `open` — the honest test is cheap: `tight_couple` either converges or it does not, and `seam_defect_split` already refuses on it |

> **What this tier did not have to invent.** The reference trajectory is the agents themselves; the norm is the port algebra's own bond; the attribution is Tier 0's split with the state norm swapped out. Every piece was already in the vault, and what was missing was the observation that E3's string comparison had been standing in for a question about *referents* since the spec was written. **[AI Inference]:** that is the same failure mode as W74's — a field on the wrong object — and it suggests the audit worth running next is not over rules but over *what each declaration is a proxy for*.

---

# 17. The referent's three loose ends, and the rule aimed at the smaller term — 2026-08-30

Reproduce with `python scripts/w87_referent_cost_and_lag.py` and `python scripts/w7_multirate_matching.py`; artifacts `out/w87/w87.json` and `out/w7m/w7m.json`. Tests: `tests/test_tier17_referent_and_multirate.py`.

> **Dating.** Every measurement here was taken on **2026-08-29** — the artifact mtimes say so — and this section was written just after midnight, so it is stamped by the clock as **2026-08-30**, which is [[tier0-measurements]] §15's rule and the log's. Section 15 is stamped 08-29 and section 14 08-30, so the sequence reads 30, 29, 30; nothing is renamed, because the log is append-only.

Tier 16 closed the multiphysics blocker and opened four rows about the object it built. Three of them (**W87**, **W85**, **W86**) are debt rather than theory, and all three came back with an answer the row did not anticipate. The fourth item, **multirate**, is the one Tier 16 listed as a foundational blocker, and it turns out **R9 has been refusing every multirate graph over the term that is $62\times$ smaller than the one it does not mention.**

## 17.1 W87 — `lambda_ref` names an experiment, and the experiment sees one thing

`lambda_ref` decides whether $\tau$ is `UNDEFINED` at a multiphysics seam: declared on both sides, `L1/E3` **admits**; absent, $\tau$ is emitted as `UNDEFINED`. So a *false* declaration promotes, which is `governing_family`'s shape — the class with teeth. Tier 16 logged it as a **fourth** unverifiable declaration, on the grounds that it is a free-form string.

**It is not, and the reason generalizes W77.** The string is not what the rule consumes. What the rule consumes is the claim that a **reference pair converges**, and `multiphysics.tight_couple` settles that in one solve. So `conformance._test_lambda_ref` runs it and the count stays at three.

> A declaration that can be *derived* is an unwritten derivation (W77). A declaration that names an **experiment** is an unrun experiment. The question to ask of a new field is what would falsify it, not what it is made of.

**And then the experiment was run against four deliberately wrong reference pairs, and four of five converge.**

| reference pair | verdict | $\lambda^{*}$ [K] | inside $[250, 900]$ |
|---|---|---|---|
| the real pair | `admit` | $372.1377$ | yes |
| shell sign-flipped | `admit` | $425.6904$ | yes |
| shell blind | `admit` | $900.0067$ | **no**, by $0.0067$ K |
| gas blind | `admit` | $374.1082$ | yes |
| both blind | **`refuse`** | — (residual $2.317\times10^{2}$ after 50 iterations) | — |

A monotone residual has a root whichever way its two halves lean, so a reference that ignores its boundary datum — and even one on the *wrong side of its own bond* — still balances, at a different trace. **What converging falsifies is exactly one claim: that the interface problem is not empty.** That is `CASE-STUDY-GUIDE` mistake 6 arriving through a declaration, and it is the claim `L1/E3` promotes on.

**The admissible interval is not a second discriminator, and that was measured rather than assumed.** The one wrong pair it separates lands $0.0067$ K outside a $650$ K reservoir span — a relative margin of $1.0\times10^{-5}$ — because a constant shell flux is balanced only where the gas transmits that constant, which is the gas's own inlet temperature. A rule keyed on a $10^{-5}$ excursion is not a rule. The sign-flipped pair is not separated at all.

**What the miss costs, and the one reassuring thing in it.** Scoring the *real* pair, whose true $\tau$ is zero, against each wrong referent:

| reference | $\tau_{\text{gas}}$ | $\tau_{\text{shell}}$ |
|---|---|---|
| shell sign-flipped | $0$ | $\mathbf{2.0000}$ |
| shell blind | $0$ | $1.3179\times10^{8}$ |
| gas blind | $\mathbf{7.3161\times10^{-2}}$ | $0$ |

Every defect lands on the agent whose referent was corrupted and the other side stays identically zero. So a false `lambda_ref` corrupts an attribution's **magnitude** and not its **localization** — worth knowing, and not a reason to trust the magnitude.

## 17.2 W85 — the seam's operator is not a Jacobian, and no threshold could make it one

The referent costs $2(n+1)$ solves per Newton iteration on a dense finite-difference Jacobian. The seam's probed $S$ is already assembled by the compiler for every seam, so it is the obvious cheap substitute; W85 asked whether `operator_content` predicts when it is good enough.

Five operating points on `thermal_seam`, with $\Delta t$ across $4\times$ and the film coefficient across $100\times$, giving $\omega$ from $2.00\times10^{-5}$ to $3.55\times10^{-4}$ — a $17.8\times$ range:

| $\Delta t$ | $h_{\text{in}}$ | Jacobian | $\omega$ | iters | residual | solves | $\lvert\lambda-\lambda_{\text{FD}}\rvert$ |
|---|---|---|---|---|---|---|---|
| $10^{-4}$ | $500$ | dense FD | — | 9 | $1.32\times10^{-7}$ | 116 | — |
| | | $S$ split-base | $2.000\times10^{-5}$ | 20 | $2.569\times10^{-2}$ | 82 | $3.031\times10^{-3}$ |
| | | $S$ seam-base | $3.339\times10^{-5}$ | 6 | $1.051\times10^{-4}$ | **24** | $5.229\times10^{-5}$ |
| $4\times10^{-4}$ | $500$ | $S$ seam-base | $3.433\times10^{-5}$ | 6 | $5.435\times10^{-4}$ | 32 | $2.495\times10^{-4}$ |
| $10^{-4}$ | $5000$ | $S$ seam-base | $3.596\times10^{-5}$ | 6 | $1.042\times10^{-4}$ | 28 | $5.166\times10^{-6}$ |
| $10^{-4}$ | $50000$ | $S$ seam-base | $1.619\times10^{-4}$ | 6 | $1.041\times10^{-4}$ | 24 | $5.165\times10^{-7}$ |

**Not one of the ten $S$ rows converges**, at any operating point, and the reason is not conditioning. Decomposing the residual at the stopping point:

| | residual in $\operatorname{range}(P)$ | orthogonal to it |
|---|---|---|
| $S$ seam-base, all five points | $8.3\times10^{-14}$ to $1.0\times10^{-11}$ | $1.041\times10^{-4}$ to $5.435\times10^{-4}$ |

> **The step $P S^{-1} R\,r$ lives in $\operatorname{range}(P)$, and $\dim M = 16$ against $\dim V = 48$ leaves 32 directions it cannot reach.** The iteration drives what it can see to machine zero in six iterations and stops; the plateau *is* the orthogonal part, to every digit. **No threshold on a spectral quantity can fix a subspace** — and none is licensed by the data either, since over a $17.8\times$ range of $\omega$ the outcome does not move at all.

This is the third time a statistic has been asked to decide something it is not about (**W68** non-normality for globality, **W75** a Fourier basis for support), and the first where the obstruction is dimensional rather than spectral.

**What $S$ *is* worth is a warm start**, and only at a consistent seam base: 24–32 solves against 106–116, landing $5.2\times10^{-5}$ K from the converged trace. At the per-expert bases the compiler still defaults to (**W80**), the same operator leaves $2.6\times10^{-2}$ *in* $\operatorname{range}(P)$ and lands $3.0\times10^{-3}$ away — $245\times$ worse in residual, $58\times$ in trace. **That is W74's class getting a numerical consequence for the first time** rather than a certificate one: $\Lambda_M = \sum_i P_i^{*}\Lambda_i P_i$ is a Jacobian only if the terms share a point, and using it as one is where that stops being an argument.

`multiphysics.seam_jacobian` exposes the route with those numbers in its docstring; the default stays dense FD, and a $\dim M$ operator handed to a $\dim V$ Newton is now **refused** rather than silently falling through to the scalar secant, which is what it used to do.

## 17.3 W86 — the slope belongs to the seam, the lag belongs to the run, and the declared number is neither

§16.5 established that $\sigma$ is a function of the lag and quoted one with the lag in a provenance *string*. Walking `thermal_seam`'s reference trajectory for five macro-steps, re-converging at each:

| step | $\lambda^{*}$ [K] | drift [K] | $\sigma$ at that lag | slope [K$^{-1}$] | slope $\times$ drift |
|---|---|---|---|---|---|
| 1 | $372.137823$ | $5.1669\times10^{-4}$ | $4.0684\times10^{-6}$ | $3.450014\times10^{-2}$ | $1.7826\times10^{-5}$ |
| 2 | $372.137209$ | $1.1740\times10^{-3}$ | $2.1176\times10^{-5}$ | $3.450060\times10^{-2}$ | $4.0502\times10^{-5}$ |
| 3 | $372.136301$ | $1.0872\times10^{-3}$ | $3.1328\times10^{-5}$ | $3.450068\times10^{-2}$ | $3.7510\times10^{-5}$ |
| 4 | $372.135468$ | $1.7742\times10^{-3}$ | $2.8740\times10^{-5}$ | $3.450084\times10^{-2}$ | $6.1210\times10^{-5}$ |
| 5 | $372.134905$ | $1.7245\times10^{-3}$ | $1.9449\times10^{-5}$ | $3.450133\times10^{-2}$ | $5.9497\times10^{-5}$ |

Three facts, and they decide the row's question in a way neither of its two options anticipated:

1. **The slope is a property of the seam.** $3.45001\times10^{-2}$ to $3.45013\times10^{-2}$ — constant to five digits across every step.
2. **The lag is a property of the run.** It moves $3.43\times$ across five consecutive steps, non-monotone, and **nothing at compile time knows it**, because nothing has stepped yet. So W77's fix does *not* transfer: the probe had already run when `probe_state` was derived from it, and the trajectory has not.
3. **$\sigma$ is not a function of the scalar lag.** At drifts within $8\%$ of each other (steps 2 and 3) it differs by $\mathbf{1.48\times}$, and over the five steps it ranges $7.70\times$. The uniform-shift prediction over-predicts by $1.20\times$ to $4.38\times$ and does not track. **$\sigma$'s argument is the lag profile**, and §16.5's curve — measured against uniform shifts — is a one-dimensional slice of a 48-dimensional argument.

**So the declared constant is a uniform-shift extrapolation, and its provenance named the wrong stage.** `MEASURED_W83.sigma` $=3.392278\times10^{-5}$ reproduces exactly as $3.4501\times10^{-2}\times9.83\times10^{-4}$, and the artifact its `source` field cited carries `measured_sigma` $=1.0773\times10^{-11}$ — the $\sigma$ at *zero* lag, which is the check that $\sigma$ measures the lag and not a value to quote. The run's own $\sigma$ over five steps is $4.07\times10^{-6}$ to $3.13\times10^{-5}$, all *below* the declared value, so it is conservative here — the safe direction for a bound, and not the same as correct.

The fix is therefore split rather than chosen: `MeasuredConstants.sigma_lag` carries the lag as a **number**, `multiphysics.lag_distance` derives the run's own from the two consecutive traces `solve.rollout` was already carrying and nothing read, and `check_sigma_lag` compares them at a provenance tolerance of $2\times$. It can falsify and it cannot confirm, and the $1.48\times$ is why.

## 17.4 W7 / R9 — multirate, and a rule aimed at the smaller of two terms

`thermal_seam` at `clocks="native"` — $0.1$ ms of compressible gas against a $50$ ms backward-Euler shell, $500{:}1$ — has compiled to `refuse` since it was written, on `L7/R9`. The refusal named its own exit: *"until the scheme declares time-integrated matching with each side's own substep quadrature"*, and nothing could declare it.

§9.3 built the time-integrated condition for the **single-rate** case in 2026-08-28 and it failed both tests, because §5's defect was spatial. That retraction ended with a sentence this section discharges: *"R9's time-integration remains right for what R9 is about — multirate — and the two-rate test is still owed."*

**The two-rate test.** Hold the interface at the tightly coupled trace, march the gas over one shell step recording its own sub-step flux, and compare the integral against the pointwise value the slow side would be handed. One march of 500 steps, sliced, so the whole sweep costs one march:

| ratio | $\int F\,\mathrm{d}t$ | $F(T)\,\Delta T$ | relative leak |
|---|---|---|---|
| $1$ | $2.295095$ | $2.295095$ | $\mathbf{0}$ exactly |
| $10$ | $22.94848$ | $22.94597$ | $1.1\times10^{-4}$ |
| $50$ | $114.6753$ | $114.6218$ | $4.7\times10^{-4}$ |
| $500$ | $1146.248$ | $1146.149$ | $8.6\times10^{-5}$ |

Ratio $1$ is the control and it is zero to machine precision, because one sub-step *is* the interval — so the discrepancy is the clocks and not the instrument. **And at $500{:}1$ the pointwise match delivers $0.9999\times$ the heat the gas actually transported.** R9 has been refusing a graph over that.

**The comparison that decides it, both terms in interface power over one $\Delta T = 0.05$ s interval:**

| term | value |
|---|---|
| R9's — pointwise against integrated flux | $5.9972\times10^{-5}$ |
| the lag — a stale trace over the whole interval | $\mathbf{3.7431\times10^{-3}}$ |
| ratio | $\mathbf{62.4}$ |

> **R9 refuses over the fast side's flux *transient*. What a $500{:}1$ exchange interval actually costs is the fast side's trace being *stale* for 500 of its own steps, and that is $62\times$ larger.** Both are dimensionless defects in the same norm — interface power, §16.3's unit — so the master bound would add them as scalars and the smaller one is the one with a rule.

**What was built anyway, and why.** The rule is right in principle: over an interval the conserved quantity *is* the integral, and matching a pointwise value is not conservative at any magnitude. So `graph.FluxMatching` supplies the declaration the refusal asked for, and L7 requires three things of it rather than taking it on trust:

- **the clocks nest** — $\Delta T/\Delta t_i$ a whole number of each agent's own steps, or the two sums are over intervals with different endpoints and there is no common integral;
- **every side can compute its integral** — `boundary_response_integrated`, because `boundary_response` restarts from the agent's own state, so calling it $n$ times recomputes the first sub-step $n$ times rather than marching $n$. A scheme claiming to match an integral it cannot compute is a contradiction inside the declaration, which is `boundary_response_jvp`'s rule applied to the same shape of claim;
- **the composition uses it** — `solve._port_fluxes` reads the scheme and calls it.

And it is **not a fifth unverifiable declaration**: at $n = 1$ the integrated response must equal the plain one exactly, which is two solves and a test.

**And the third of those three requirements was false when it was written.** L7's admission rests on *"`solve.coupled_step` calls it"*, and it did not: `_port_fluxes` gained the branch and the scheme was never passed to it, so the branch was unreachable and the `assert` guarding it could never fire. **The compile was admitting R9 on a promise the run does not keep** — which is exactly the class the three-verdict split exists to separate, arriving in the code that implements the separation. It was found by asking what would happen if the call site were wrong, not by any check: no test exercised a *time-integrated coupled step*, only the compile that authorizes one. Two now do, and the second is the control that the pointwise path never touches the integral.

> **[AI Inference]:** the shape here is worth naming because it is not the usual one. Every other defect this vault has found was a rule whose premise or magnitude was wrong; this one was a rule that was right, stated correctly in the admission message, and **not wired to anything**. A decision message is a claim about the code, and nothing in this architecture checks those the way it checks claims about experts.

`L7/R9/lag` then decertifies immediately beside the admission, carrying the $62.4$. Admitting R9 without it would read as having fixed multirate.

## 17.5 Where the compiles land, 2026-08-30

| graph | verdict | refusals | change |
|---|---|---|---|
| `thermal_seam` split-step, **native** ($500{:}1$) | **`admit-uncertified`** | **0** | was **`refuse`** |
| `thermal_seam` split-step, matched | `admit-uncertified` | 0 | unchanged |
| `thermal_seam` native, `flux_matching=POINTWISE` | `refuse` | 1 | the pre-2026-08-30 compile, kept reachable |
| `window_ns` split-step | `admit-uncertified` | 0 | unchanged |

Tests go $344 \to 374$. The native-clock graph's `unmeasured` list is $L$, $\sigma$, $\tau$, $C_\mu$, `norm_A` — ordinary per-case work, and $\sigma$ is now the one §17.4 says will decide it.

## 17.6 Opened by this tier

| # | Where | What | Status |
|---|---|---|---|
| **W90** | `atlas/compiler.py`, `atlas/scheme.py` | **The multirate defect has no rule.** R9 covers the flux transient and is $62\times$ too small; the lag over a long exchange interval is $\sigma$, and nothing bounds it as a function of the interval. R4 forbids shrinking the exchange interval below $\max_i \Delta t_i$, so the obvious remedy is the one axis a frozen expert cannot move. The candidate is **W17's $W > 1$**: a trace carried as a waveform is exactly a trace that is not stale | `open` — and it is now the *named* multirate blocker rather than an unexamined one |
| **W91** | `atlas/multiphysics.py` | $\sigma$'s argument is the lag **profile** and every measurement of it, §16.5's included, holds the profile uniform. What the profile's shape costs is unmeasured: $1.48\times$ between two steps whose scalar lags agree to $8\%$ is a lower bound on it, from two points | `open` |
| **W92** | `atlas/probe.py`, `atlas/composition.py` | W85 gives **W80** its first numerical consequence — the split-base operator is $245\times$ worse as a Jacobian — and W80's own objection stands: deriving $\lambda^{*}$ took a 50-step bisection. But a **run** has the previous step's converged trace in hand for free, which is exactly the consistent base, and nothing feeds it to `assemble_seam` | `open`, and cheaper than W80 assumed |

## 17.7 What this tier says about the ones before it

| Claim | Where | Now |
|---|---|---|
| `lambda_ref` is a *fourth* declaration of `validity`'s kind | Tier 16, W87 | **no** — it names an experiment, and the count stays at three. W77's lesson generalizes past "derive it" |
| *"the probed $S$ is the obvious cheap substitute, but $M$ is sized for transmission"* | `numerical_jacobian`, W85 | right, and the obstruction is **dimensional**: the in-subspace residual goes to $10^{-13}$ and the orthogonal part is untouched at $1.04\times10^{-4}$. No spectral threshold applies |
| $\sigma = 3.392278\times10^{-5}$ *"at the lag a real run carries"* | §16.5, `MEASURED_W83` | a **uniform-shift** value at an assumed drift, reproducing as slope $\times$ lag; the run's own is $4.07\times10^{-6}$ to $3.13\times10^{-5}$ — W86 |
| §16.5's $\sigma$-versus-lag curve | §16.5 | measured against **uniform** shifts only, so it is a slice. The scalar lag does not determine $\sigma$ (1.48$\times$ at matched lag) — W91 |
| *"multirate — R9 refuses unless the flux is matched time-integrated"* | §16.7's ledger | the declaration exists and the graph compiles; **the ledger row was aimed at the term $62\times$ smaller than the real one** — W90 |
| §9.3's *"R9's time-integration remains right for what R9 is about"* | §9.3 | **right, and now measured**: conservation is exact at ratio 1 and the leak is real above it. Also $8.6\times10^{-5}$ at $500{:}1$, which is not what a foundational blocker looks like |

## The mechanism tally

Tier 15 concluded that *a rule becomes auditable only after some measurement has named it*. This tier is the cleanest case of that yet and it sharpens it in one direction: **every one of the four findings came from measuring the object a rule is stated about, and in three of the four the rule's premise was true and its magnitude was wrong.** W87's check works and sees one thing out of five; $S$ is the right operator and lives in the wrong subspace; R9's integral is the conserved quantity and is $62\times$ too small to matter here. Only W86's declared constant was wrong on its own terms.

> **The pattern to carry forward: a rule can be correct, derived, and aimed at the wrong order of magnitude, and nothing but a measurement in a common norm will say so.** Interface power is that norm and this vault has had it since §16.3 — the multirate ledger row has stood since 2026-08-27 because nobody put the two terms in it.

# 18. Real turbine geometry, and the attribution machinery on a frozen expert — 2026-08-30

Full record here; worklist rows on [[gap-worklist]] Tier 18. Reproduce with
`python scripts/w93_wake_array.py --steps 60`; artifacts `out/w93/w93.json` and
`out/w93/state.npz`. New case study `atlas/cases/wake_array.py`, the **sixth**
real one and the first whose *geometry* is real.

**Provenance.** The script was run from cold and again from the cached state and
the two artifacts compared leaf by leaf: **1321 numeric values, of which 1314 are
identical to the bit.** The seven that differ are all `wall_s` — wall-clock
timings, which are not measurements of anything. The checkpoint is
bit-deterministic (two identical calls differ by exactly $0$), so every number
below traces to one invocation and reproduces from either path.

The superseded pre-W98 artifact is kept as `out/w93/w93.pre-w98.json`, and the
reference solver's column is **unchanged to every digit** between the two — which
is the control that says W98 was in the composition and not in the physics.

Tiers 14–17 built an attribution stack — `probe.support_reach`,
`probe.operator_content`, `probe.assemble_seam` with a seam base,
`multiphysics.seam_defect_split`, `composition.certify_substitution` — and every
number it has ever produced came from `thermal_seam`, a two-agent duct, or from
`window_ns`, one solver tiled against itself. **None of it had been pointed at a
real pretrained checkpoint inside a real composed flow**, which is the case the
whole project exists for. This tier does that, on the problem wind-farm
operators actually pay for: wake-induced power loss across a small array.

`wind_farm_real.py` says of itself that its geometry is schematic — six windows
of one field standing in for a farm — and that *"a number measured here is a
number about the port algebra. It is not a wind-farm result and must not be
quoted as one."* This is the other half.

## 18.1 The geometry, which the checkpoint chose

The checkpoint is fixed at $128\times128$ and `adapters.Scaling` fixes
everything else from it. One macro-step must be one *native* expert lead
($0.1$), the freestream must land inside the trained velocity distribution
($U_s = 2$), so a window spanning $S$ rotor diameters forces
$\Delta t = S/20$ — and

$$\mathrm{Re}_{\text{eff}} \;=\; \frac{1}{2\,\nu_p\,S}\;=\;\frac{1020}{S}.$$

**Choosing the window chooses the macro-step, the Reynolds number, and how much
lateral room a wake has to recover in, and the last two pull against each
other.** §11's note was that with a solver you choose the window to suit the
halo and with a checkpoint you choose the domain to suit the window; here it
goes one step further, because **the turbine spacing is a whole number of window
strides by construction.**

| | value |
|---|---|
| window | $4\,D$ per 128 cells, so $\mathrm{d}x = \tfrac{1}{32}D$ |
| macro-step | $0.2$, lead $0.1$ — exactly native |
| halo | 16 cells $= 0.5\,D$, stride 112 cells $= 3.5\,D$ |
| domain | $352 \times 240$ cells $= 11.0 \times 7.5\,D$, six windows in a $3\times2$ tiling |
| rotors | R1 $(3.75,\,1.75)$, R2 $(7.25,\,1.75)$, R3 $(7.25,\,5.25)$, $D = 1 = 32$ cells |
| $\mathrm{Re}_{\text{eff}}$ | $\mathbf{255}$, against the spec's band $[10^3, 10^4]$ |

R1 and R2 in line, R3 abreast of R2 one row over: an **L**, $3.5\,D$ on both
legs, which is a real closely-spaced layout. R1 and R3 see clean inflow and R2
sees R1's wake, so the array carries its own controls.

**The Reynolds number is the price of the geometry and it is paid deliberately.**
The band and the lateral room a wake needs are in direct conflict for a
fixed-resolution operator: $\mathrm{Re} = 1020/S$, and $S=1$ — where W0 put it —
is one rotor diameter per window, which leaves a wake nowhere to go. W0 §4.2 has
the deeper reason not to mind very much: the checkpoint's viscosity **is not a
number**. It fits $\nu_p = 4.9\times10^{-4}$ at $\lambda = 0.125\,D$ with
$r^2 = 0.998$ and nothing separable from a 2%-per-step bias at $0.25\,D$ or
$0.5\,D$, so any $\mathrm{Re}$ quoted for it describes a spectral cutoff rather
than a fluid.

**Three declarations fall out of that cutoff rather than out of convention.**
`effective_resolution` is a *wavelength*, not a mode count — $m = 2n/8 + 1$ from
$\lambda_{\text{cut}} = 0.25\,D = 8$ cells, giving 33 modes on a 128-cell face,
25 on a 96-cell one and **9 on a 32-cell rotor face**, where carrying
`window_ns`'s 16 would have claimed four times the resolution the checkpoint
has. And the reference solver's viscosity is the same fit converted to these
units, $\nu = \nu_p S U_s = 3.92\times10^{-3}$.

## 18.2 A port is per face SEGMENT, and the seams do not claim a coincidence

`wind_farm_real` established that an `ADVEC` passenger list belongs to a **face**
and not to an agent. Real geometry needs one more step: a rotor spans $1\,D$ of a
$4\,D$ face, so the plane it sits on carries **two ports on one ring** — the disk
and the open flow beside it — and nothing in the port algebra says a port's
$V_i$ has to be a whole face. The bypass segment is the face minus the rotor,
which is two disjoint intervals, and an orthonormal basis on an index set is an
orthonormal basis. Thirteen seams over nine agents come out of the layout with
nothing listed by hand.

**And `geometrically_coincident` is `False` on all thirteen, for the first time
in this vault.** In an overlapping decomposition the two artificial rings a seam
pairs are `halo` cells apart and **no two of them coincide**; `poseidon.py` and
`wind_farm_real.py` both declare `True` on rings 20 cells apart. The README
lists the field as *"a label nothing checks"*, and this is what checking it would
have said. Nothing downstream changes, because every seam declares a transfer.

One consequence of the same geometry reads backwards and is right: **each
window's artificial ring lies inside its neighbour**, so a surface agent sitting
in the overlap meets the *downstream* window's inflow ring on its **upstream**
side. R1's `up:MECH` port therefore connects to F10 and its `down:MECH` port to
F00.

## 18.3 The trace is absolute, and W74's class was hiding inside the state

`window_ns`, `channel_ns`, `poseidon` and `wind_farm_real` all write the trace
into the ring as a **perturbation**, `ring + trace`. Under that convention
`probe_base` is identically zero on every fluid seam in this vault, so
`probe.base_disagreement` reads *consistent* on all of them — while the two sides
are in fact linearized about their own stored states, which nothing compares.
**W74 fixed the case where the base is declared and wrong; this is the case where
the base is not a declaration at all.**

`wake_array` writes the trace **absolutely** and declares `probe_base` as the
ring's own physical value. The arithmetic at that point is identical; what
changes is that the point is on the record. The result, on the compile:

| | seams | base spread |
|---|---|---|
| `L4/probe-base` **fires** | **10 of 13** | up to $\mathbf{123\%}$ of the base norm ($0.01927$ on $M$, seam `y0c0`) |
| agrees | 3 | the three `R*_up` seams, whose rotor `u_ref` is *constructed from* the fluid ring it faces |

The three that agree are exactly the three built to agree, and the ten that do
not are pairs of rings $0.5\,D$ apart across a velocity gradient. **The check has
existed since 2026-08-29 and has never been able to say anything about a fluid
seam.**

Two more things follow, and one of them is required rather than nice:

- `base_sensitivity` reads $2.678\times10^{-2}$ on Poseidon-T and
  $5.124\times10^{-3}$ on the reference at a $1.03\%$ shift, four and three
  orders above `BASE_SENSITIVITY_FLOOR`. **The base is not free here.** W74's
  escape hatch — an affine response makes the base provably irrelevant — is
  unavailable for either expert.
- **`multiphysics.seam_defect_split` needs the absolute convention.** It measures
  the defect in interface power $\int_\Gamma e\,f$, and a perturbation times a
  perturbation is not a power. None of the four perturbation-convention case
  studies could have been measured in that norm.

## 18.4 W93 — the halo rule reads a declaration, and the probe measures the same thing

`capability.required_halo` returns $\text{radius} \times \text{substeps}$. W69
already corrected it once, in exactly these words: *"that product is an EXPLICIT
agent's domain of dependence"*, and it fixed the branch where the record says
`implicit`. **It left the branch where the record says nothing** — and `unknown`
is precisely what a frozen learned one-shot map declares, under R2b, because it
is neither.

`probe.support_reach` measures the same quantity, by poking a delta. On
Poseidon-T, at the developed array state, on the upper-left window's downstream
face — which is upstream of every rotor, so this is the checkpoint's reach in
ordinary flow rather than a property of the wake:

| expert | nonzero | fraction | reach | verdict |
|---|---|---|---|---|
| **Poseidon-T** | $128/128$ | $\mathbf{1.0000}$ | 64 | `('embedded',)` |
| `reference.WindowNS`, embedded projection | $126/128$ | $0.9844$ | 63 | `('embedded',)` |

identically at probe amplitudes $1$, $10^{-1}$, $10^{-2}$ and $10^{-3}$, so it is
not an amplitude artifact — and **the checkpoint is *more* global than a genuine
Poisson solve.**

$$\text{declared } 2 \text{ cells} \qquad \text{measured } 64 \text{ cells}
\qquad \mathbf{32\times}$$

with the graph's overlap at 16, so `L2/R10`'s halo rule was **admitting** on the
strength of the 2. `required_halo` now returns `None` for an undeclared clock
with a nonzero stencil, `_halo_rule` decertifies naming the measurement, and the
wake-array compile carries `L2/R10/halo` over all six checkpoint windows.

> **The reason this is not a fix to Poseidon-T's record.** A neural operator's
> receptive field is global **by construction** — a U-Net with attention over a
> $128^2$ window has no compact domain of dependence at any depth — and that is a
> fact about the *architecture*, not about the physics it was trained on. So the
> honest declaration is not a bigger integer; there is no integer. The halo is
> undecidable from the record and measurable from the expert.

**And the same measurement resolves `elliptic_subsolve`, which is where it gets
uncomfortable.** `support_reach` is W75's gate and `consistent_with` reads
`('embedded',)`; `poseidon.elliptic_signature`, the *other* instrument, reads
$\kappa = 267$, asymmetry $0.788$, $\omega = 0.819$ — above
`OPERATOR_CONTENT_FLOOR`, so its precondition is cleared — and returns
*"consistent with EMBEDDED"* too. **Two independent instruments, one built from
support and one from the spectrum, agree on the first black box both have had
signal on.** Declare the value they measure and `L2/R10` **refuses the whole
graph**, which is the correct behaviour of the rule and a hard result for the
project: *R10 refuses to decompose any expert whose response is global, and a
globally-receptive pretrained operator is nearly all of them.*

## 18.5 The array, marched, and the number an operator is paid in

> **Everything in this section was re-measured after W98 (§18.5.1).** The figures
> first recorded here came from a march that manufactured a wake $3.5\,D$
> upstream of a lone turbine, and roughly two thirds of the headline disagreement
> between the two experts was the *coupling's* rather than the checkpoint's. The
> superseded artifact is kept at `out/w93/w93.pre-w98.json`. The referent's column
> below is unchanged to every digit from that artifact, which is the control: the
> reference solver never had the defect.

Four runs, 60 macro-steps each, at three turbines with the disks' own
zero-parameter closed form. Two controls: a **turbine-free** run, which is the
array's own numerical noise floor, and a **single-turbine** run.

**Two things about the coupling are forced rather than chosen, and both are
consequences of the expert being frozen.** Measured over four macro-steps with
all three disks:

| | min $u$ | max $u$ |
|---|---|---|
| derived thickness, projection **on** (as shipped) | $+0.4313$ | $1.3587$ |
| `disk.py`'s default $\Delta_d = 0.1\,D$, projection on | $+0.2073$ | $1.5989$ |
| derived thickness, projection **off** | $\mathbf{-0.7395}$ | $1.7725$ |

**The thickness is derived, not chosen.** A body-force impulse over one
macro-step is $f\,\Delta t = T\Delta t/(A\,\Delta_d)$ and momentum theory allows a
crossing parcel to lose $T/(A\,\langle U_d\rangle)$; the two agree only at
$\Delta_d = \langle U_d\rangle\,\Delta t$. R4 forbids shrinking the exchange
interval below $\max_i \Delta t_i$ and a frozen checkpoint's
$\Delta t_{\text{native}}$ is not a dial, so **the disk's smearing thickness is a
property of the expert rather than of the rotor** — the row above where it is the
rotor's costs $2.1\times$ in the wake minimum.

**And the projection is not optional: without it the flow through the disk plane
REVERSES**, which no momentum sink may produce. The checkpoint's pressure channel
is a *placeholder its loader pins to $0$* (`adapters` fact 1), so the field that
mediates an actuator disk's momentum sink is not among its outputs. The
composition layer supplies it, and the graph declares it a `GlobalField` — a
declaration §18.5.1 is about the march failing to honour. The reference solver
needs none, because its own projection is embedded: **the property R10 refuses to
decompose is the same property that makes a classical solver usable here without
help.**

| | Poseidon-T | `reference.WindowNS` |
|---|---|---|
| turbine-free control, $\langle U_d\rangle$ drift | $+0.438\%$, $+0.481\%$, $+0.345\%$ | $0$ exactly, spread $0$ |
| isolated turbine $\langle U_d\rangle$ (sampled $0.25\,D$ upstream) | $0.7763$ | $0.8289$ |
| R1, against R1 alone — **blockage** | $+3.60\%$ | $+1.60\%$ |
| **R2, against R1 off — the wake loss** | $\mathbf{+74.98\%}$ | $\mathbf{+79.54\%}$ |
| R3, against R1 off — the offset row | $-8.48\%$ | $+4.29\%$ |
| array, $\sum P / 3P_{\text{iso}}$ | $\mathbf{14.62\%}$ lost | $\mathbf{25.27\%}$ lost |

**Ten and a half percentage points of array loss, on the same geometry, the same
disks, the same controls and the same macro-step, from swapping the fluid
expert** — down from $21.5$ before W98, so the framework had been contributing
more error than the model it was measuring. At farm scale the remainder is still
the whole of the quantity micro-siting and wake steering are bought to move.

Three honesties about that table, none of which the number survives without:

1. **Neither column is validated against data.** The classical solver is the
   *declared referent*, not ground truth — that is exactly what `lambda_ref` is
   and exactly what W87 warns a referent can be wrong about.
2. **The one independent check does not rank them.** $\langle U_d\rangle$ is
   sampled $0.25\,D$ upstream of the disk plane, where momentum theory's
   $U_\infty(1-a) = 0.667$ is a *lower* bound because the induction is still
   building; both must read above it, and $+16.44\%$ against $+24.33\%$ bounds
   how much upstream induction each produces rather than which is right.
3. **The disagreement is now where the physics is weakest, and it changed
   character.** Before W98 the two split the same total loss differently — more
   blockage on R1, less wake on R2 — and the array totals were $21.5$ pp apart.
   After it, R2 sits in a developed wake and the two are $4.6$ pp apart, while R3
   sits in the *bypass* flow beside a wake and there they differ in **sign**: the
   checkpoint has R3 gaining $8.5\%$ from its neighbour's blockage where the
   referent has it losing $4.3\%$. A wake model that is nearly right inside the
   wake and wrong outside it is a specific, reportable failure; it was unreadable
   while the coupling's own error dominated both columns.

R3 is the row that needs its noise floor quoted beside it. Its
$\langle U_d\rangle$ moves $+2.76\%$ between the two runs against a turbine-free
drift of $+0.345\%$ in the same quantity — a margin of $8.0\times$, which is
enough to call the sign real and not enough to quote a second significant figure
of it.

### 18.5.1 W98 — transport and pressure are global, and the march ran both per window

`build` declares `pressure` a `GlobalField`, with the note *"the composition
layer's, and it has to be"*. The march then asked `step_many` for
`project=True`, which runs an exact Leray projection **on each 128-cell window,
separately and periodically**; the same call's `frame` argument translates each
window by $U\,\Delta t/\Delta x = 6.4$ cells with `spectral_shift`, which is
periodic **on the window**. Two consequences, both physical:

* a wake reaching a window's outflow edge re-entered *that window's own* inflow
  edge, completing one circulation every $N/6.4 = 20$ macro-steps — which is why
  the manufactured deficit was flat in $x$: it is the wake's own $y$-profile,
  smeared along the direction it kept going round;
* a disk's momentum sink had its elliptic response spread over its own $4\,D$ box
  and wrapped onto that box's upstream image.

Measured with **R1 the only disk**, so that every cell upstream of its plane and
the whole bottom row have no cause:

| $u$ at $1.25\,D$ upstream, after 10 macro-steps | before | after | referent |
|---|---|---|---|
| the only turbine in the domain is $1.25\,D$ downstream | $0.8244$ | $\mathbf{0.9495}$ | $0.9966$ |

and at $3.5\,D$ upstream after 40 macro-steps, $0.7123 \to \mathbf{0.9063}$
against the referent's $0.9992$. A row containing no turbine at all had drifted
to $0.976$.

**It is not a halo failure and no overlap fixes it.** A partition-of-unity dead
zone giving the wrapped strip *exactly* zero weight moves the deficit by
$0.012$ — because §18.4's measured support reach is the whole window, so the
contamination is not confined to the strip the translation wrapped. The repair is
`wake_array.transport_and_project`: one Leray projection and one translation, on
the domain, which is extended downstream by `PAD_CELLS` $= N$ cells of
fluctuation tapered to zero. That extension is periodic-compatible, so the
projection is exact, the translation is a phase factor rather than an
interpolation, and what wraps onto the inlet is the taper's zero — freestream,
which is what an inlet in an unbounded stream supplies. It converges in the
buffer width: $0.8374$ at no buffer, $0.9279$ at $N$, $0.9377$ at $1.75N$.

**The residual belongs to the checkpoint.** About $5\%$ of spurious upstream
deficit survives the repair at ten macro-steps, and it is §18.4's reach: a window
holding a disk responds to it everywhere, so a decomposition into windows cannot
keep a disk's influence inside the sub-domain that contains it. That is the
concrete cost of W93 rather than its statement.

**On how it was found.** Nothing asserted its way to this and no reasonable
assertion would have: the manufactured deficit is smooth, bounded, physically
shaped and in the right units, and every control in this section passed with it
present — $\tau$ calibrated exactly, the certificate refused, the turbine-free
run sat at its noise floor. What exposed it was rendering the field beside the
referent's and watching, where a wake standing upstream of the only turbine is
obvious in a second. **A rendering of the state is a measurement instrument**,
and on this evidence it belongs in the same tier as the probes.

## 18.6 $\tau$, measured — and the instrument's positive control on this expert

The headline seam is `x1r0_bypass`: the open flow beside R2, on the plane
$3.5\,D$ downstream of R1, **fluid-to-fluid**, so the port's own $\nu$ appears on
both sides and cancels exactly in a relative defect. $\dim V = 96$, trace mean
$1.17952$, and the lag the run itself carries over one macro-step is
$7.913\times10^{-3}$ (`multiphysics.lag_distance`, from two consecutive states).

**The control first, because §16's ladder had only ever run on `thermal_seam`.**
Reference = the WindowNS pair, actual = the same pair with one side scaled:

| injected | $\tau[\text{F10}]$ | $\tau[\text{F20}]$ |
|---|---|---|
| $\times 1.05$ | $\mathbf{0.050000}$ | $0.000\times10^{0}$ |
| $\times 1.25$ | $\mathbf{0.250000}$ | $0.000\times10^{0}$ |
| $\times 2.00$ | $\mathbf{1.000000}$ | $0.000\times10^{0}$ |

Exact recovery, and the unswapped side identically zero — the same result §16.2
got in a different norm, on a different physics, with the trace convention
changed underneath it.

**Then the measurement.** Reference = the WindowNS pair at
$\nu = 3.92\times10^{-3}$; the referent converges, with
$P_{\text{ref}} = 7.6627\times10^{-4}$ at $\lambda^{*} = 1.05694$.

| actual pair | $\tau[\text{F10}]$ | $\tau[\text{F20}]$ | $\sigma$ | total | sub-additive |
|---|---|---|---|---|---|
| the reference pair (control) | $0$ | $0$ | $1.30727$ | $1.30727$ | yes |
| **Poseidon-T on F10 only** | $\mathbf{11.57598}$ | $0$ | $0.84864$ | $12.42463$ | yes |
| **Poseidon-T on F20 only** | $0$ | $\mathbf{1.12281}$ | $1.30727$ | $1.85948$ | yes |
| Poseidon-T on both | $11.57598$ | $1.12281$ | $0.84864$ | $12.42463$ | yes |

**The framework catches it, and loudly.** At the interface state the reference
pair converges to, the checkpoint's interface power is **an order of magnitude
above** the referent's on the upstream side — and the attribution *localizes*:
swapping one side leaves the other's $\tau$ at exactly zero, so the number names
an agent rather than a graph. The master bound's shape survives on every row.

**$\tau$ rose when the coupling was repaired**, $10.74 \to 11.58$, while the
power tables in §18.5 converged. That is not a contradiction and it is the more
interesting direction: $\tau$ is a defect in interface power *at a probed state*,
and W98's error had been degrading both sides of the seam together, which is
partly self-cancelling in a difference. Removing it made the trajectory better
and the attribution sharper at once — the behaviour one wants from an instrument
meant to isolate the expert from the scheme it runs in.

Two corroborating quantities from the same seam, both first measurements:

- **$\Xi = 0.0182$** — the composability index of Poseidon-T against a
  same-geometry classical solver on a real wake seam. §8.3 measured $0.19$–$0.53$
  for the embedded-versus-exposed pair of *one* solver; `poseidon.py` wrote
  `xi_against_window` and it had never been run at a physical state. The
  checkpoint reproduces under $2\%$ of the boundary response the reference does,
  and the mechanism is **W59**: overwriting the ring of an *initial condition*
  and letting a one-shot map smooth it is not holding a Dirichlet trace through a
  step, and the near-wall gradient the Steklov–Poincaré flux reads is what the
  difference shows up in.
- the assembled seam, at the seam base the run supplies:

| | $\beta$ | $\kappa$ | $\pi$ | $\omega$ | shares | `one_sided` |
|---|---|---|---|---|---|---|
| Poseidon pair | $8.403\times10^{-6}$ | $334.2$ | $2.37\times10^{-4}$ | $0.7469$ | $0.532 / 1.107$ | $0.532$ |
| reference pair | $4.543\times10^{-3}$ | $29.43$ | $0$ | $0.5300$ | $0.635 / 0.384$ | $0.384$ |

$\beta$ is **540× smaller** under the checkpoint and the passivity defect is
nonzero where the reference's is exactly zero — so E7 `fails` on this graph, and
it fails because of the learned expert rather than because of the topology.

### 18.6.1 W95 — the referent $\tau$ is defined against cannot be built from the checkpoint

§16's construction is that at a seam with no monolith the reference trajectory is
the **tightly coupled pair**. For a checkpoint fixed at $128\times128$ there is
no monolith at any resolution, so that is the only route — and `wake_array`
declares `lambda_ref` on the checkpoint's record for exactly that reason, which
generalizes W88: **`lambda_ref` was introduced for a multiphysics seam and a
fixed-resolution learned expert needs it at a single-physics one, for the same
reason.**

Then the experiment. Reference = the Poseidon pair itself, dense
finite-difference Jacobian, 194 forward passes:

| | |
|---|---|
| $J$ | $96\times96$, $\sigma_{\max} = 3.0253\times10^{-3}$, $\sigma_{\min} = 2.7449\times10^{-7}$, $\kappa = 1.10\times10^{4}$ |
| effective rank at a $10^{-8}$ cut | $96/96$ — **full** |
| the same $J$ at a $2\times$ smaller probe step | differs by $\mathbf{4.70\times10^{-2}}$ relative |
| directions below that reproducibility | $\mathbf{63/96}$ |
| Newton, damping $1.0$ / $0.5$ / $0.2$ | **diverges**: $3.38\times10^{-1}$, $1.24\times10^{-1}$, $5.76\times10^{-2}$ from $2.49\times10^{-2}$ |
| Newton truncated to the 33 resolved directions | **diverges**: $1.66\times10^{-1}$ |
| the same solve on the **WindowNS** pair, for contrast | converges in **7 iterations**, $9.20\times10^{-3} \to 1.02\times10^{-9}$ |

**It is a third mechanism, and the first two are eliminated by measurement.**
W85's obstruction was *dimensional* — a step in $\operatorname{range}(P)$ cannot
reach $\dim V - \dim M$ directions — and this Jacobian lives in $V$, so that one
does not apply. Conditioning is eliminated by the truncation: keeping only the
33 directions the probe can reproduce still diverges. Step length is eliminated
by the damping sweep, which is the standard remedy for a Jacobian valid only
locally and which does not help at $0.2$. What is left is that the interface
residual of a frozen checkpoint **has no root reachable from the state the run is
in**, and the residual history is non-monotone from the first iteration
($2.49\times10^{-2} \to 6.80 \to 9.00 \to 10.4 \to 9.18 \times10^{-2}$).

**W98 did not change this conclusion, and the sensitivity of its numbers to the
state is itself the finding.** Re-measured on the repaired march, every
mechanism lands in the same place — full rank, most directions below
reproducibility, divergence under truncation *and* damping — while the
individual figures move by tens of percent ($\kappa$ from $2.17\times10^{4}$ to
$1.10\times10^{4}$). A conclusion that survives a substantive change of the
state it was probed at, with its supporting numbers visibly moving underneath
it, is a stronger result than one measured once.

`seam_defect_split` reports this correctly and refuses rather than guessing:
*"the reference pair did not converge... there is no reference trajectory, so tau
and sigma have no referent here."* So the calibration ladder above had to be run
against the classical pair, and the honest reading is the one W87 already stated
in a different setting — **a $\tau$ measured against a declared referent is a
distance from that referent, and for a learned expert the referent cannot be the
expert.**

> **The cost inverts, which is the one piece of good news.** W85 priced the
> referent at $2(n+1)$ solves and called it *"prohibitive for a learned expert at
> $128^2$"*. But `seam_defect_split` converges the **reference** pair and
> evaluates the actual pair exactly four times — so when the referent is the
> classical solver, the $194$ solves are paid in `WindowNS` currency at
> $0.04\,\mathrm{s}$ each and the checkpoint is asked for four forward passes.
> The expensive half of the attribution is not the expensive expert's.

### 18.6.2 The referent's own viscosity, and how little $\tau$ cares

The checkpoint's viscosity is not a number, so the reference solver's is a
*choice*, and $\tau$ inherits it. Measured across W0 §4.2's own fits:

| $\nu_p$ fit | $\nu_{\text{solver}}$ | cell $\mathrm{Re}$ | $\tau[\text{F10}]$ |
|---|---|---|---|
| $\lambda = 0.125\,D$ ($r^2 = 0.998$) | $3.920\times10^{-3}$ | $7.97$ | $11.57598$ |
| geometric mean | $7.513\times10^{-4}$ | $41.59$ | $11.15393$ |
| $\lambda = 0.5\,D$ ($r^2 = 0.27$) | $1.440\times10^{-4}$ | $217.01$ | $11.02736$ |

**The referent is under-determined by $27\times$ in cell Reynolds number and
$\tau$ moves $1.05\times$.** That is a positive result and it was not the
expected one: the attribution is robust to the one degree of freedom the referent
has, which is what lets the $11.6$ above be quoted at all. The ratio is unchanged
by W98 — before the repair the same sweep spanned $10.74$ to $10.19$, also
$1.05\times$ — so the *robustness* of the attribution to the referent's viscosity
is a property of the seam rather than of the state it was probed at.

The row that is *not* free is the reference solver's own validity. Cell
Reynolds is $\mathrm{d}x\,U/\nu = 1/(256\,\nu_p)$ — **independent of the window
size**, so it is a property of the checkpoint — and `wind_farm_real`'s declared
predicate is $\le 8$. Only the grid-scale fit clears it at the freestream
($7.97$), and none of the three clears it at the wake state's own maximum speed
($10.78$, $56.24$, $293.41$). The referent is a solver run just outside its
declared envelope, and it says so.

## 18.7 W97 — a rotor seam is one-sided by the cell Reynolds number

`probed-dtn-coupling` §2.2's normative MECH effort is $\nu\,\partial w/\partial n$
and W47 pinned that form as the block-level one. An actuator disk's traction is
$\tfrac12\rho C_T' u^2$, which carries no $\nu$. **So a field-to-lumped MECH seam
is one-sided by a factor of the cell Reynolds number, structurally, before any
expert is chosen.** Measured on `R2_up`:

| seam | $\lVert\Delta\rVert$ | $\beta$ | $\lVert S_{\text{fluid}}\rVert$ | share | `one_sided` | verdict at $\beta_{\min} \in [0, 0.5]$ |
|---|---|---|---|---|---|---|
| `x1r0_bypass` fluid–fluid | $0.08318$ | $0.004243$ | $0.08341$ | $0.634$ | $0.386$ | **`refuse`**, `blind = False` |
| `R2_up` field–lumped | $0.00683$ | $0.55976$ | $0.00681$ | $\mathbf{0.0042}$ | $\mathbf{4.21\times10^{-3}}$ | `admit-uncertified`, **`blind = True`** |

W76's inequality is that a swap of agent $i$ is invisible whenever
$\lVert S_i\rVert < \beta - \beta_{\min}$, so the range of $\beta_{\min}$ over
which the fluid expert cannot be caught is
$(\beta - \lVert S_i\rVert)/\lVert S_i\rVert$ — **in units of the block being
tested**, which is the only scale-free statement — and for two commuting blocks
that ratio is exactly
$\lVert S_{\text{other}}\rVert / \lVert S_i\rVert$. At a rotor seam that is the
cell Reynolds number:

$$\frac{\lVert S_{\text{rotor}}\rVert}{\lVert S_{\text{fluid}}\rVert}
\;\sim\; \frac{\tfrac12 C_T' u}{\nu/\mathrm{d}x} \;=\; \mathrm{Re}_h.$$

So the fluid expert's replacement at a rotor face is certified blind over a
$\beta_{\min}$ range **81 times its own block norm** as measured
($(0.55976 - 0.00681)/0.00681$), against $1\times$ at a balanced seam, and
$\mathrm{Re}_h = 255$ in the exactly-commuting idealization the test pins. On this graph the certificate says `admit-uncertified` with
`blind = True` at every $\beta_{\min}$ tried, and the measurement it is blind to
is the $\tau = 10.7$ two sections up.

**W76 found this as a property of one asymmetric tiling; here it is a property of
a port TYPE PAIRING**, and it will hold at every field-to-lumped MECH seam any
case study ever writes. The candidate repair is §4.1's conservative co-normal,
which W47 scoped *out* at fluid–fluid seams on the grounds that the advective
term cancels between the two sides — and at a rotor face the advective term is
exactly what does not cancel, because the disk is what makes it jump. That is a
new entry in W47's own scoping box rather than a contradiction of it.

## 18.8 W94 — an actuator disk has no bond, by two routes

An actuator disk is a surface across which the traction jumps, and a **surface
bond** is what the port algebra has. Getting one needs the two fluid subdomains
to *abut* at the disk plane — a non-overlapping cut — and `R2b` will not lift the
transmission rung to probed-DtN for an agent whose `time_discretization` is
`unknown`, which a learned one-shot map's is (**W61**). Under the *overlapping*
decomposition the checkpoint does support, the disk sits inside the overlap,
**both** windows cover it, and the coupling is **two-way volumetric** — which is
W70's shape and the row [[gap-worklist]] §16.7 lists as `open` with the note
*"the two-way one has no port and no bond"*.

So the two available formulations are each blocked, by different rules, and both
blocks are about the frozen checkpoint:

| formulation | blocked by |
|---|---|
| surface bond, non-overlapping cut at the disk plane | **R2b / W61** — a learned one-shot map cannot support probed-DtN, so the axis is illegal |
| volumetric two-way coupling, overlapping | **W70 / §16.7** — no port, no bond |

`wake_array` takes the third option, which is neither: it declares the rotor's
two ports at the two nearest artificial rings, $0.25\,D$ upstream and $0.22\,D$
downstream of the disk plane, declares `geometrically_coincident=False` because
they are not, and lets the compiler speak. **The measurements in §18.7 are taken
on that construction and are about it**; what §18.8 says is that the construction
is a compromise with a name, and that the vault's flagship case study lands
exactly on the open blocker its own ledger has been carrying since 2026-08-29.

> **[AI Inference]:** the actuator disk is the simplest, oldest, zero-parameter
> rotor model there is, and it is the object this project's flagship case study
> was built around. That it has no expressible bond against a frozen operator —
> by two independent routes — is a stronger statement about the port algebra's
> coverage than any of the fixtures could make, and it is only visible with real
> geometry, because a schematic rotor can be hung off any ring at all.

## 18.9 Where the compiles land, 2026-08-30

| graph | verdict | refusals | decertifications |
|---|---|---|---|
| `wake_array` split (`elliptic_subsolve=unknown`) | **`admit-uncertified`** | **0** | 50 across 44 rules |
| `wake_array`, `elliptic_subsolve=embedded` — *what §18.4 measures* | **`refuse`** | 1 (`L2/R10`, all six checkpoint windows) | — |
| `wake_array`, `elliptic_subsolve=none` — *what `poseidon.py` defaults to* | `admit-uncertified` | 0 | — |

Stamp: `E1 holds  E2 holds  E3 holds  E4 holds  E5 unchecked  E6 holds  E7 fails`.
`tau_undefined_seams` is **empty** — every seam has a referent, which is what
declaring `lambda_ref` on a fixed-resolution expert buys. `unmeasured` is
$L$, $\sigma$, $\tau$, $C_\mu$, $p_{\text{decline}}$: ordinary per-case work, and
$\tau$ is now measured *at a seam* without being declared *on the graph*.

The decertification profile is worth reading as a shape rather than a list:
thirteen `L3/C8` and six `L1/C8` for the checkpoint's absent `validity` (it has
no measurable envelope — W13); ten `L4/probe-base` and ten `L4/E7/passivity`, one
per probed fluid seam; one `L2/R10/W60`, one `L2/R10/halo` (**new, W93**), one
`L4/R2b/W46`, and one `L4/block-share` — W76's rule, the first thing in this
vault ever to read `alpha_star`, firing on a graph it was not derived from.

## 18.10 A tenth finding, small and cheap to state

`wind_farm_real.RotorAgent.respond` returns `ActuatorDisk.force_density` — a
force per unit **volume** — against a port whose scale set declares `stress`. At
the default $\Delta_d = 0.1$ the two differ by exactly $10\times$. This is
**W66's class** with one difference that matters: W69 concluded that *"the wrong
half differs from the right one by an $O(1)$ factor once nondimensionalized, and
no dimensionless diagnostic separates $O(1)$ from $O(1)$"*, and here the factor
is a **declared geometric length**, so this instance is separable in principle
even though nothing separates it today. `wake_array.RotorDisk` returns the
traction, $\tfrac12\rho C_T' u^2$, and §12's `ADVEC` and `ROT` numbers are
unaffected — the rotor's `MECH` magnitude is the only thing that inherits it.

## 18.11 Closed this tier / opened by it

| # | verdict |
|---|---|
| **W93** | **`done`.** `required_halo` returns `None` for an undeclared clock with a nonzero stencil, by W69's own derivation; `_halo_rule` decertifies naming `support_reach`; measured $2$ against $64$. §18.5.1 gives it a *consequence*: the reach is why a $5\%$ spurious upstream deficit survives even a correct coupling |
| **W98** | **`done`.** Transport and pressure are global operations and the march ran both per window, each periodic on its window, manufacturing a $25\%$ velocity deficit $3.5\,D$ upstream of a lone turbine. Repaired by `wake_array.transport_and_project`; it was worth roughly two thirds of the headline expert disagreement, and **it was found by rendering the field, not by any assertion** |
| **W94** | `open` — the actuator disk has no bond by two routes, and this is W70 / §16.7's row with a case behind it |
| **W95** | `open` — the referent cannot be built from a frozen checkpoint's own pair; dimensional, spectral and step-length mechanisms all eliminated by measurement, and the conclusion survived re-measurement on the repaired state while its numbers moved by tens of percent |
| **W96** | `open` — `wind_farm_real`'s rotor returns a force density against a `stress` scale key, $10\times$ |
| **W97** | `open` — a field-to-lumped MECH seam is one-sided by $\mathrm{Re}_h$, so every substitution certificate on its fluid side is blind by construction |
| **W99** | `open` — `FrozenFluidExpert.step` and `.step_many` drop `force` silently when `galilean` is false, and pin each window's output to zero mean, deleting the deficit a disk just deposited. Both are one line in the build repo and neither announces itself |

## 18.12 What this tier says about the ones before it

| Claim | Where | Now |
|---|---|---|
| *"that product is an EXPLICIT agent's domain of dependence"* | §14.5, W69 | right, and it fixed one of the two branches it indicts — W93 |
| W75's support gate, *"two solves, no spectral threshold"* | §15.2 | holds on a learned operator, and reads it as **`embedded`** for a reason that is architectural rather than physical |
| W68/W75's *"the two instruments are complements rather than alternatives"* | §14.3, §15.2 | on the first black box where **both** have signal they **agree** — `('embedded',)` and *"consistent with EMBEDDED"* |
| W76, *"the blind spot is the block norm"* | §15.3 | a property of a **port-type pairing**, not of one tiling: every field-to-lumped MECH seam is blind by $\mathrm{Re}_h$ — W97 |
| W85, *"the referent costs $2(n+1)$ solves, prohibitive for a learned expert"* | §17.2 | the cost is paid in the **referent's** currency, not the tested expert's — four forward passes here. And the price was the wrong question: the referent does not exist — W95 |
| W87, *"a false referent corrupts $\tau$'s magnitude while leaving its localization correct"* | §17.1 | measured on a second case: the localization is exact ($0$ on the unswapped side, every row) and the magnitude moves $1.05\times$ over a $27\times$ referent band |
| W74, *"the base belongs to the SEAM"* | §14.4, §15.1 | and it can be **hidden in the state**: four case studies' perturbation convention makes every fluid `probe_base` zero, and `base_disagreement` has been reading *consistent* on all of them |
| §16.7's *"volumetric two-way coupling between two agents — no port and no bond"* | §16.7 | has a case now, and it is the wind farm — W94 |
| §11's *"with a checkpoint you choose the domain to suit the window"* | §11 | one step further: **the turbine spacing is a whole number of window strides**, and the Reynolds band and the wake's lateral room are in direct conflict |

## The mechanism tally

Tier 17 concluded that *a rule can be correct, derived, and aimed at the wrong
order of magnitude, and nothing but a measurement in a common norm will say so*.
This tier's four findings do not share that shape. **Three of the five rows above
are rules that were right and had never been evaluated where their premise
fails** — W69's halo derivation stopped one branch short, W76's blindness was
stated for a tiling and holds for a port pairing, W85's referent price assumed a
referent — and the fourth, W94, is a gap the ledger already carried and that no
fixture could reach.

> **What made all four visible is the same thing: real geometry.** A schematic
> rotor can be hung off any ring, so it never forces the bond question; a
> synthetic seam has a base of zero, so it never forces the linearization
> question; and a solver you can run at any resolution has a monolith, so it
> never forces the referent question. **The three case studies before this one
> were each honest about being a fixture in one dimension, and each of those
> dimensions was hiding a rule.**

# 19. How composition error grows with interface count — 2026-08-31

Full record here; worklist rows on [[gap-worklist]] Tier 19. Reproduce with
`python scripts/w100_scaling_ladder.py --steps 60 --acc-steps 20`, then
`python scripts/w100_timing.py --steps 8 --repeats 6`; artifacts
`out/w100/w100.json`, `out/w100/state_<rung>.npz`, `out/w100/solo_<rung>.npz`.
New case study `atlas/cases/scaling_ladder.py`, the **seventh** real one, and the
first whose *variable* is the size of the graph.

> **Dating.** The marches ran 2026-08-30 and the run finished and was written up
> 2026-08-31, verified against the environment clock before stamping, per Tier
> 15's rule.

[[f1-pathmap-and-end-goal]] §5's **F1** is the criterion that fails if
composition error grows *super-linearly in interface count*, §3.2 names rung 9 as
*the rung that decides everything*, and §5 schedules this sweep as rung 9's early
warning — *"a cheap look at the expensive question, and a bad result there is a
reason to stop and rethink, not to press on."* **It had never been run at any
size.** [[case-study-ladder-to-f1]] §4 then schedules nine further case studies
on the answer and says outright that CS-16 is not built if it comes back
super-linear.

So this tier grows one geometry from $1$ to $24$ coupled windows with everything
else nailed down, in two columns, and reports an exponent with an error bar.

## 19.1 The ladder, and what is held fixed

The parent's geometry, parameterized. `wake_array.ArrayTiling` now carries
`n_col`, `n_row` and `rotors`; `ArrayTiling()` reproduces the $3\times2$ array to
the bit, which `tests/test_tier19_scaling_ladder.py` pins and which the artifact
diff confirms — **1314 of 1321 numeric values in `out/w93/w93.json` are identical
after the refactor, and the seven that differ are wall-clock timings.**

| held fixed | value | why |
|---|---|---|
| $\mathrm{d}x$ | $S_{\text{LEN}}/128 = 1/32\,D$ | the checkpoint's resolution is not a dial |
| $\Delta t_{\text{macro}}$ | $0.2$ | exactly one native expert lead |
| overlap | $16$ cells | changing it changes $\Pi$, which is a measured output |
| ramp | $8$ cells | W53 keeps ramp 8, so every prior constant stays comparable |
| $\nu_{\text{ref}}$ | $3.92\times10^{-3}$ | the referent's own viscosity, W0 4.2's grid-scale fit |
| disk model, induction $a$ | `ActuatorDisk`, $1/3$ | zero fitted parameters |

| $N$ | tiling | domain (cells) | domain ($D$) | rotors | seams | overlapping pairs |
|---|---|---|---|---|---|---|
| $1$ | $1\times1$ | $128\times128$ | $4.0\times4.0$ | $0$ | $0$ | $0$ |
| $2$ | $2\times1$ | $240\times128$ | $7.5\times4.0$ | $1$ | $3$ | $1$ |
| $6$ | $3\times2$ | $352\times240$ | $11.0\times7.5$ | $3$ | $13$ | $11$ |
| $12$ | $4\times3$ | $464\times352$ | $14.5\times11.0$ | $5$ | $27$ | $29$ |
| $24$ | $6\times4$ | $688\times464$ | $21.5\times14.5$ | $12$ | $62$ | $68$ |

**Two counts, reported side by side rather than one being called *the* interface
count.** `n_seams` is what the compile sees. `n_overlaps` is what the composition
defect is created at, and they are not the same thing: two windows meeting
diagonally share a $16\times16$ corner and have no `Connection` at all, while
their local solves still disagree there. Every exponent below is against
`n_overlaps`.

The rotor layout is a **rule**, not a list: `rotor_motif` tiles the parent's L —
two turbines in line and one abreast, $3.5\,D$ both ways — with period ($3$
columns, $2$ rows) and truncates. At $3\times2$ that returns
`wake_array.ROTORS` exactly. At $4\times3$ truncation gives **five** rotors where
the plan sketched six; the rule is kept and the count reported, because a motif
bent to hit a predicted number is not a rule.

### The two controls, and both pass

**$N=1$ returns exactly zero, twice over.** With one window the partition of
unity is identically $1$, `cut` and `assemble` are the identity, and the composed
step *is* the monolith: the composed field and the monolith agree **bit for bit**
(`np.array_equal`, not a tolerance), and the reference-free blend spread is
identically $0$ because there is nothing to spread. This is [[tier0-measurements]]
§8's own lesson — *the control that would have redirected the search immediately
was the one not run* — and it costs one line.

There is a declaration-side half as well: one window has no artificial face, so
it has no port, and `ExpertCapabilities` raises before the compiler is reached.
**The framework will not express a one-agent composition**, and it refuses one
layer *earlier* than L3's "no connections declared: this is not a composition",
which is the message a reader would expect.

**$N=6$ reproduces the parent to the digit.** Array loss $25.27\%$ (WindowNS) and
$14.62\%$ (Poseidon-T) against [[case-study-wake-array-atlas-0.1]]'s $25.273\%$
and $14.625\%$, on a driver that is a transcription of `w93_wake_array.march`
rather than a re-derivation — which is the point: a control against a
re-implementation controls nothing.

| $N$ | array loss, WindowNS | array loss, Poseidon-T |
|---|---|---|
| $2$ | $0.00\%$ | $0.00\%$ |
| $6$ | $\mathbf{25.27\%}$ | $\mathbf{14.62\%}$ |
| $12$ | $31.38\%$ | $21.76\%$ |
| $24$ | $30.37\%$ | $38.97\%$ |

$N=2$ is $0.00\%$ by construction — one turbine is its own control. And the
turbine-free noise floor is **exactly $0.000\%$** for the reference solver at
every rung (a uniform stream is a fixed point of the unforced monolith and of the
unforced composed step alike) against $0.42$–$0.49\%$ for the checkpoint.

## 19.2 The gate — F1 is not falsified

The composed defect over **one exchange interval**, measured at each rung's own
developed state with the disks off, in three forms. The reference-free one is the
chi-weighted variance $\sqrt{V_\chi}$ of the local one-macro-step solutions —
which is exactly the term L6/C1's identity removes, needs no monolith, and is
therefore the only currency the checkpoint's column can carry (**W95**).

| $N$ | interfaces | $\Pi$ | spread, WindowNS | spread, Poseidon-T | composed vs monolith | $\lVert\sum_i\chi_i\lvert D_i\rvert\rVert$ | tightness |
|---|---|---|---|---|---|---|---|
| $1$ | $0$ | $0$ | $0$ | $0$ | $\mathbf{0}$ | $0$ | — |
| $2$ | $1$ | $0.4678$ | $8.121\times10^{-1}$ | $1.364$ | $2.194$ | $2.318$ | $1.057$ |
| $6$ | $11$ | $0.7250$ | $4.780$ | $3.723$ | $15.57$ | $16.19$ | $1.040$ |
| $12$ | $29$ | $0.7250$ | $10.80$ | $6.545$ | $35.29$ | $36.71$ | $1.040$ |
| $24$ | $68$ | $0.7250$ | $25.43$ | $10.34$ | $82.67$ | $85.96$ | $1.040$ |

**The gate.** Ordinary least squares of $\log(\text{defect})$ on
$\log(\text{interfaces})$ over the four rungs that have an interface, $n-2=2$
degrees of freedom, $t_{0.975,2}=4.303$:

| series | exponent $p$ | 95% CI | $r^2$ | reading |
|---|---|---|---|---|
| classical, reference-free | $\mathbf{+0.804}$ | $[+0.641,\ +0.967]$ | $0.996$ | sub-linear |
| checkpoint, reference-free | $\mathbf{+0.477}$ | $[+0.362,\ +0.593]$ | $0.994$ | sub-linear |
| classical, against the monolith | $\mathbf{+0.851}$ | $[+0.748,\ +0.954]$ | $0.998$ | sub-linear |

> **F1 is not falsified.** No column's interval lies above $1$ over the full
> $24\times$ range.

**The test is one-sided and that is not a detail.** F1 fails on *super-linear*
growth, so what would falsify it is an interval lying strictly **above** $1$.
Gating instead on *"upper bound at or below 1"* would report a linear result as a
failure, which is the criterion tightened after the fact; the pickup's own three
readings — sub-linear confirms Claim B, linear is survivable, super-linear stops
the program — are the three the code implements.

The pairwise local slopes are reported beside the fit because a fitted exponent
averages over curvature and curvature is what a super-linear onset would look
like:

| series | $1\to11$ | $11\to29$ | $29\to68$ |
|---|---|---|---|
| classical, reference-free | $+0.739$ | $+0.841$ | $+1.005$ |
| checkpoint, reference-free | $+0.419$ | $+0.582$ | $+0.537$ |
| classical, against the monolith | $+0.817$ | $+0.844$ | $+0.999$ |

**The classical column's local slope reaches $1.00$ at the top of the ladder and
the checkpoint's does not.** That is the one thing in this section a reader
should carry forward with suspicion: the fitted interval clears $1$, and the last
segment does not.

Two supporting series, both sub-linear:

| series | exponent | 95% CI |
|---|---|---|
| per cell (RMS), classical | $+0.534$ | $[+0.49,\ +0.58]$ |
| per cell (RMS), checkpoint | $+0.207$ | $[+0.17,\ +0.24]$ |
| worst cell (sup norm), classical | $+0.553$ | $[+0.42,\ +0.69]$ |
| worst cell (sup norm), checkpoint | $+0.138$ | $[+0.02,\ +0.26]$ |
| per interface, classical | $+0.304$ | $[+0.14,\ +0.47]$ |
| per interface, checkpoint | $\mathbf{-0.023}$ | $[-0.14,\ +0.09]$ |

**Per interface, the checkpoint's composition defect is flat to within its own
error bar.** Errors adding in quadrature over independent interfaces would give
$+0.5$ on the total and $0$ per interface, which is what the checkpoint's column
does; the classical column adds $+0.3$ on top, which is correlation between
interfaces along the streamwise direction a wake actually crosses.

## 19.3 The control that decomposes the exponent, and it is most of the answer

**A bigger array is not only more interfaces.** At $N=24$ the developed field
reaches $u_{\min}=0.183$ against $N=2$'s $0.509$: deeper wakes, stronger
gradients, harder physics. A defect that grows with $N$ could be growing because
the flow got harder rather than because the cut got longer, and §19.2 cannot
separate them.

So the same three numbers are measured again at a **one-turbine state**, where
only the first rotor runs. The flow near it is identical at every rung — same
inflow, same disk, same wake, same position relative to the inlet — and every
window added beyond it sits in near-freestream. A defect that still grows there
is growing on interface count alone.

| $N$ | interfaces | spread, WindowNS | spread, Poseidon-T | composed vs monolith |
|---|---|---|---|---|
| $1$ | $0$ | $0$ | $0$ | $0$ |
| $2$ | $1$ | $0.8121$ | $1.364$ | $2.194$ |
| $6$ | $11$ | $0.8566$ | $2.513$ | $2.217$ |
| $12$ | $29$ | $1.035$ | $2.839$ | $2.817$ |
| $24$ | $68$ | $1.600$ | $3.124$ | $4.878$ |

| series | exponent | 95% CI |
|---|---|---|
| one turbine, classical, reference-free | $+0.138$ | $[-0.16,\ +0.43]$ |
| one turbine, checkpoint, reference-free | $+0.201$ | $[+0.09,\ +0.31]$ |
| one turbine, classical, against the monolith | $+0.160$ | $[-0.23,\ +0.55]$ |

> **With the physics held fixed, composition error is nearly flat in interface
> count: $p \approx 0.14$–$0.20$ against the headline's $0.48$–$0.85$.** A
> $68\times$ increase in interfaces buys a $2.2\times$ increase in composed
> defect. Most of §19.2's growth is the flow, not the cut.

Both are real and only the second is what F1 asks about. Stated as a warning
rather than a comfort: **a sub-linear headline on a ladder whose physics grows
with its size is a weaker result than it looks**, and this control is what makes
the number interpretable. It was not in the brief and should have been.

## 19.4 The bound's tightness does not degrade, and $\Pi$ saturates

$\lVert\sum_i\chi_i\lvert D_i\rvert\rVert$ against the measured one-interval
defect:

| $N$ | $1$ | $2$ | $6$ | $12$ | $24$ |
|---|---|---|---|---|---|
| bound / actual | — | $1.057$ | $1.040$ | $1.040$ | $1.040$ |

**L2/C2's bound is tight to $4\%$ and flat over the whole $24\times$ range.**
§10's $0.2\%$ at four windows of the *other* tiling is not reproduced here and
was never claimed to transfer; what does transfer is that the bound does not
loosen as the graph grows, which is the property a bound needs to be worth
carrying.

$\Pi$ — W49's contaminated weight, the constant the overlapping branch's $\sigma$
bound needs — is $0$ at $N=1$, $0.4678$ at $N=2$ and **$0.7250$ at every rung from
$N=6$ on, to every digit**. It saturates because past $3\times2$ every added
window is an interior one and the worst cell is already as contaminated as it
gets. So $\Pi$ is a property of the *tiling's local pattern* and not of its size,
and $C_\mu = 1.2$ carries across the ladder with no re-measurement.

## 19.5 W58 — which bound was measured, and the hypothesis nobody had stated

W58 asked for two things: a provenance field naming which of `cut_defect_bound`'s
two definitions was measured, and a compile-time check of the disjointness
condition that makes them equal. Both are built —
`MeasuredConstants.cut_defect_bound_form`, `graph.CUT_DEFECT_FORMS`,
`assembly.contaminated_multiplicity`, and `L2/C2/W58` on the compile — and then
the measurement found a **second hypothesis the framework had never stated**, and
it is the one that scales.

`probe.neighbour_disagreement` is a **max over neighbour pairs of a norm**.
L2/C2's max form is a **norm over the whole grid of a cellwise max**. Those
coincide when there is at most one overlapping pair and diverge above it —
whatever the contaminated sets are doing — because a max over pairs saturates as
the tiling grows while the bound keeps accumulating.

| $N$ | contaminated cells | shared by $>1$ | max multiplicity | disjoint? | **aggregated** / max form | **pairwise** / max form |
|---|---|---|---|---|---|---|
| $1$ | $0$ | $0$ | $0$ | yes | — | — |
| $2$ | $2048$ | $0$ | $1$ | **yes** | $0.7653$ | $0.7653$ |
| $6$ | $12800$ | $512$ | $3$ | no | $0.7993$ | $0.4170$ |
| $12$ | $30208$ | $1536$ | $3$ | no | $0.7969$ | $0.3902$ |
| $24$ | $66304$ | $3840$ | $3$ | no | $0.8020$ | $\mathbf{0.2339}$ |

> **The disjointness hypothesis the framework named moves the ratio by four
> percent. The aggregation nobody had named moves it by a factor of three.**

Disjointness holds at $N=2$ and fails from $N=6$ on, with the shared count growing
$512 \to 1536 \to 3840$ — and across that failure the *aggregated* surrogate holds
at $0.765 \to 0.802$, while the *pairwise* one falls $0.765 \to 0.234$. The
framework's stated hypothesis is real and is second-order here; the unstated one
is first-order and grows without bound.

A third gap is visible in the same table and is smaller than either: even at
$N=2$, with disjointness holding exactly, the aggregated surrogate is $0.765$ of
the max form rather than $1$. The theorem's argument assumes $D_i$ is *supported
inside* the declared contaminated set, and it is not — `contaminated` is a
declaration about reach within one exchange interval, and the actual restriction
defect has a tail beyond it. So **disjointness of the declared sets is necessary
and not sufficient**, and the residual $23\%$ is the tail.

`atlas/graph.py` now names four forms rather than two, `atlas/probe.py` gains
`aggregated_neighbour_disagreement`, and the compile decertifies the pairwise form
on any partition with more than two subdomains — decidable from the subdomain
count, with no run.

**And the row has teeth on this run's own compiles.** The classical column can
declare the chi-weighted form because it has a monolith; the checkpoint's column
cannot, at any resolution ever (W95), so the only form it can supply is the
reference-free one. At $N=2$ the contaminated sets are disjoint and `L2/C2`
**admits** it as the bound. **From $N=6$ on the same declaration is decertified**,
purely on geometry:

> `L2/C2/W58` fires on the Poseidon column at $N=6$, $12$ and $24$ and not at
> $N=2$. The checkpoint's column loses the cut criterion at a graph size, without
> anything about the checkpoint changing.

### One thing this row cost, and it is the reason the row exists

`probe.restriction_defect_bound` takes arrays **on the global grid** — the
identity it evaluates is $A - Eu = \sum_i R_i^\top \chi_i D_i$ and the $R_i^\top$
has to have happened already. On a uniform tiling every subdomain-local array is
the same length, so passing the un-lifted ones raises nothing and silently stacks
every window at cell $0$, inflating the bound by cross terms it invents. It was
caught by a tightness ratio that came out at $1.80$ where the same measurement one
rung down gave $1.03$. The function now takes an optional `n_global` that makes it
an error, and the convention is in the docstring rather than in a caller's head.

## 19.6 The classical composed rollout is unstable — and the compiler was already refusing it

**This is the finding the run was not looking for, and it is the largest one.**

Continue the wake array's own march past the $60$ macro-steps Tier 18 published.
The composed WindowNS column leaves the band between step $70$ and $80$ at $N=6$
and is not finite by $82$. The monolith, from the same state under the same
forcing, holds $u_{\max}$ at $1.33$ throughout. The checkpoint's column is stable
to $110$.

Measured from each rung's developed state, $20$ macro-steps, band $\lvert u\rvert = 3$:

| $N$ | monolith $u_{\max}$ | **classical**: diverged at | its $\lVert\nabla\cdot u\rVert_{\text{rms}}$ | **classical + projection** | its divergence | **checkpoint** | its divergence |
|---|---|---|---|---|---|---|---|
| $1$ | $1.000$ | — | $0 \to 0$ | — | $0 \to 0$ | — | $0.0109 \to 0.0129$ |
| $2$ | $1.297$ | — | $0.0132 \to 0.0132$ | — | $0.0259 \to 0.0249$ | — | $0.0680 \to 0.0900$ |
| $6$ | $1.413$ | **step 14** | $0.0957 \to \mathbf{2.269}$ | — | $0.0232 \to 0.0222$ | — | $0.0543 \to 0.0815$ |
| $12$ | $1.424$ | **step 9** | $0.1507 \to \mathbf{2.343}$ | — | $0.0300 \to 0.0163$ | — | $0.0568 \to 0.0641$ |
| $24$ | $1.557$ | **step 8** | $0.2502 \to \mathbf{4.583}$ | — | $0.0336 \to 0.0175$ | — | $0.0555 \to 0.0646$ |

**The mechanism, and it is one line of vector calculus.** Each `WindowNS` window
returns a field that is divergence-free *on its own window*. A
partition-of-unity blend of two divergence-free fields is not divergence-free:
on the overlap, where $\chi_1+\chi_2=1$,

$$\nabla\cdot(\chi_1 u_1 + \chi_2 u_2) \;=\; \nabla\chi_1\cdot(u_1-u_2)$$

so **the assembly creates divergence exactly where the two local solves disagree,
in proportion to how fast $\chi$ is turning over** — and in the classical column
nothing removes it. The divergence at the developed state grows monotonically
with the graph, $0.0132 \to 0.0957 \to 0.1507 \to 0.2502$, and the onset comes
earlier as it does: step $14$, then $9$, then $8$.

**The positive control closes it.** Adding *one global Leray projection to the
assembled field* each macro-step — R10b's own prescription, and exactly what the
checkpoint's column already does — makes every rung stable and holds the
divergence flat or falling ($0.0336 \to 0.0175$ at $N=24$). Nothing else changes.

> **So `L2/R10` is vindicated by measurement for the first time.** The rule
> refuses to decompose an agent whose `elliptic_subsolve` is `embedded`, and it
> has refused the reference column of every graph in this vault since Tier 0 —
> **`refuse`, one refusal, `L2/R10`, at all four rungs here.** What was missing
> was a demonstration that the refusal was about anything. It is: the graph it
> refuses goes unstable, the graph that exposes its elliptic part to the
> composition layer does not, and the difference is a single operator.

Three things follow.

**1. Tier 18's published state is on a diverging trajectory.** The wake array
marched $60$ steps and its reference column blows up between $70$ and $80$. Every
Tier 18 number measured *at* that state stands — the state exists, the numbers
are what the instruments read there — but the trajectory it sits on does not
continue, and nothing in Tier 18 said so because nothing marched past $60$.

**2. The rollout defect had to be re-scoped.** A defect quoted past the onset is a
number about a blow-up. At a short horizon of $10$ macro-steps, where every column
is still inside the band:

| $N$ | classical | classical + projection | checkpoint |
|---|---|---|---|
| $1$ | $0$ | $0$ | $0.7181$ |
| $2$ | $2.644$ | $1.934$ | $13.39$ |
| $6$ | $57.34$ | $9.013$ | $24.67$ |
| $12$ | $147.6$ | $17.91$ | $30.35$ |
| $24$ | $420.5$ | $42.05$ | $53.56$ |

The classical column's series is **void** — it diverged inside the window at three
of five rungs, and the exponential growth is present well before the band is
crossed, which is why the filter drops a rung that diverged *at all* rather than
one that crossed early. The two stable columns:

| series | exponent | 95% CI |
|---|---|---|
| rollout defect, classical + projection | $+0.712$ | $[+0.51,\ +0.92]$ |
| rollout defect, checkpoint | $+0.305$ | $[+0.08,\ +0.53]$ |

Both sub-linear, which is the same answer §19.2 gives on a single step.

**3. The $N=1$ rung gives the checkpoint a zero-composition baseline.** One
window, no interface, no assembly: the Poseidon column's defect against the
monolith over $10$ steps is $0.7181$, which is **pure model infidelity with the
composition removed by construction**. Every larger number in the checkpoint's
column is that plus composition.

### Two rejected constructions, on the record because each is the obvious thing to try

* **Freestream, disks off.** Exactly $0.0000$ at every rung. A uniform field is a
  fixed point of the composed step *and* of the monolith, so this is a correct
  answer to a useless question.
* **Developed state, disks frozen at the force that state implies.** Diverges —
  relative defect $1.285$ at $N=6$, NaN at $N=12$. An actuator disk's thrust must
  track its own inflow; freezing it removes the closure's negative feedback and
  the deceleration runs away. **That instability is the disk model's and would
  have been charged to the composition**, which is W54's shape exactly.

### 2026-08-31 — the positive control does not survive its own length either

**Everything above this line stands as measured, and its repair does not.** The
paragraph above calls one global Leray projection on the assembled field *the
positive control*, on the strength of $20$ macro-steps from each rung's developed
state. Marched to $120$, the same arrangement fails — and it fails **worse than
doing nothing**.

That is Tier 18's own mistake happening again one level down, and it is worth
naming in those words: *Tier 18 published a state on a trajectory nobody had
marched far enough, and §19.6 published a repair on a trajectory nobody had
marched far enough.* The horizon that exposed the first was $80$ steps; the
horizon that exposes the second is $40$.

$N=6$, $120$ macro-steps, disks live, reporting the macro-step at which
$\lvert u\rvert$ leaves the band $3$:

| agents | composition layer | from freestream | from the developed state |
|---|---|---|---|
| embedded | nothing | $74$ (not finite by $82$) | $\mathbf{14}$ — §19.6's own number |
| embedded | global **spectral** Leray | $\mathbf{51}$ | $\mathbf{38}$ |
| embedded | global **Neumann** Leray | $\mathbf{33}$ | $\mathbf{14}$ |
| **exposed** | nothing | never — and $\lVert\nabla\cdot u\rVert = 1.25$ | — |
| **exposed** | global spectral Leray | **never, to $120$** | — |

The $38$ in the second row is the whole story of the retraction: the positive
control **is** stable over the $20$ steps §19.6 measured, and dies at $38$.

**And the divergence is not the proximate cause.** The Neumann variant holds
$\lVert\nabla\cdot u\rVert_{\text{rms}}$ at $\mathbf{0.0089}$ against the bare
column's $\mathbf{0.69}$ — a factor of $78$ cleaner — and leaves the band
**soonest of the three**. L6/C2's residual is real, the assembly does inject it,
and it is not by itself what ends the rollout. §19.6's causal sentence — *"the
assembly creates divergence and nothing removes it"* — describes a true mechanism
and attributes an effect to it that a control now separates.

#### What the measurement selects, and it is the rule that was already there

**Adding a projection to agents that already project applies the pressure twice.**
Each `WindowNS` window has already answered the disk's momentum sink on its own
subdomain, with its own Neumann solve, inside its own sub-steps; a global solve
after assembly answers it again. The arrangement that survives is the one
`L2/R10` has prescribed since Tier 0 and that no classical column in this vault
had ever been built to satisfy: **take the elliptic part out of the agent** and
let the composition layer apply it once, to the assembled field.

`wake_array.exposed_reference_solver` is that agent — `window_ns._no_projection_class`
with `_project` replaced by the identity and the velocity update untouched, which
is the composition layer declining to use *a part of* an expert rather than
editing one. Its record declares `elliptic_subsolve=exposed`, and the consequence
at the compile is the one the rule has been promising for nineteen tiers:

| classical column | `R10` | `R10b` | `R12` | graph verdict |
|---|---|---|---|---|
| embedded, bare blend | **refuse** | — | decertify | `refuse` |
| embedded + projected assembly | **refuse** | — | decertify | `refuse` |
| **exposed + projected assembly** | *clears* | **admit** | **admit** | `admit-uncertified` |

**It is the first classical column in this vault that `R10` does not refuse**, and
it is the one that survives $120$ macro-steps. The residual `admit-uncertified` is
`G5/W16` and the disk agent's own `elliptic_subsolve=none`, neither of which is
about the assembly.

#### The march, at every rung

Entries are the macro-step at which $\lvert u\rvert$ left the band $3$, or **the number of macro-steps survived** where the column never left it.

| $N$ | overlaps | monolith | embedded, bare | embedded **+ projected** | **exposed + projected** | exposed, no projection | checkpoint |
|---|---|---|---|---|---|---|---|
| $1$ | $0$ | **120** | **120** | **120** | **120** | **120** | **120** |
| $2$ | $1$ | **120** | **120** | band $52$ | **120** | **120** | **120** |
| $6$ | $11$ | **120** | band $74$, NaN $82$ | band $51$ | **120** | **120** | **120** |
| $12$ | $29$ | **120** | band $69$, NaN $77$ | band $54$ | **120** | not run | **120** |

The assembled $\lVert\nabla\cdot u\rVert_{\text{rms}}$, first finite macro-step to last:

| $N$ | embedded, bare | embedded + projected | **exposed + projected** | exposed, no projection | checkpoint |
|---|---|---|---|---|---|
| $1$ | $0 \to 0$ | $0 \to 0$ | $0 \to 0$ | $0 \to 0$ | $0.01088 \to 0.01291$ |
| $2$ | $0.01277 \to 0.01322$ | $0.01261 \to 19.1$ | $0.01936 \to 0.05042$ | $0.524 \to 1.328$ | $0.06098 \to 0.09084$ |
| $6$ | $0.01314 \to 2.57$ | $0.01298 \to 13.47$ | $0.02022 \to 0.06556$ | $0.5417 \to 1.25$ | $0.06283 \to 0.07846$ |
| $12$ | $0.01213 \to 2.343$ | $0.01197 \to 10.14$ | $0.01723 \to 0.04011$ | -- | $0.05763 \to 0.06648$ |

**Read the fourth column against the third.** The classical column with its elliptic part left inside the agents is stable at $N=2$ and not at $N=6$; **adding L6/C2's step to it kills it at both**, at macro-step $52$ and $51$ against a bare column that survives all $120$ at $N=2$ and reaches $74$ at $N=6$. There is no rung at which the projection added on top helps, and that is the retraction stated as a measurement.

**And the fifth column is the deliverable.** With the elliptic part moved out of the agents, one global projection after the blend holds every rung run here — $1, 2, 6, 12$ windows — for all $120$ macro-steps, with $u_{\max}$ inside the monolith's own range:

| $N$ | $u_{\max}$, monolith | $u_{\max}$, exposed + projected | divergence tail trend | defect vs monolith at $120$ | checkpoint's, for scale |
|---|---|---|---|---|---|
| $1$ | $1$ | $1$ | $--$ | $0$ | $0.7714$ |
| $2$ | $1.299$ | $1.309$ | $1.0006$ | $15.62$ | $14.4$ |
| $6$ | $1.423$ | $1.336$ | $0.9893$ | $27.22$ | $27.88$ |
| $12$ | $1.478$ | $1.355$ | $1.0441$ | $33.54$ | $49.04$ |

**The trend column is where the claim is weaker than it reads, so read it exactly.** It is the last quarter's mean over the third quarter's, both taken after the flow has developed, because a march from the freestream is a transient and every column's divergence rises while the wake forms. Against the two columns that are known to be well behaved:

| $N$ | monolith | **exposed + projected** | checkpoint |
|---|---|---|---|
| $2$ | $1.0000$ | $1.0006$ | $1.0003$ |
| $6$ | $1.0016$ | $0.9893$ | $0.9998$ |
| $12$ | $0.9678$ | $1.0441$ | $1.0063$ |

**Flat or falling at $N \le 6$ and not yet at $N=12$.** At six windows the repaired column's trend is $0.9893$ — falling, and the only column of the three that is — against $22.39$ for the bare column and $62.65$ for the projected-on-top one. At twelve it is $1.0441$: **still rising**, by $4\%$ over the last thirty macro-steps, on a column whose divergence is nonetheless $0.04011$ against the checkpoint's $0.06648$. So the honest form of the claim is *stable at every rung marched, with the divergence settling by $120$ steps at $N \le 6$ and still settling at $N=12$* — and the $120$-step horizon is now known to be the wrong thing to trust without checking, which is the whole lesson of this subsection.

**The last column of that table is the one to be uncomfortable about.** The exposed classical column's defect against the monolith at $N=6$ is $27.22$ and the frozen checkpoint's is $27.88$ — the classical column and the learned one are, after $120$ macro-steps, the same distance from the right answer. Nothing here says the composed classical column is *accurate*; it says it exists, which is what it did not before.

**The control that says the projection is doing the work.** The same exposed agents with no global projection also survive all $120$ macro-steps — at $\lVert\nabla\cdot u\rVert = 1.25$ against the projected column's $0.06556$, and a defect of $66.85$ against $27.22$. **Stable, and not incompressible**, and $2.46\times$ further from the referent. That is W105.

> **N24 was not marched at this horizon**, and the reason is wall time rather than a result: the top rung is $6.1$ s per macro-step per classical column (W104), so six columns of $120$ steps is about an hour for that rung alone. The claim above is made for the rungs listed and for no others.

#### What `R12` says now, and the row it opens

`R12`'s verdict ladder is the measurement, not a design:

| what the graph declares | `R12` |
|---|---|
| a global, after-assembly projection, **and every agent exposed** | **`admit`** — the measured-stable arrangement |
| a global, after-assembly projection, **and some agent still projecting** | **`decertify`** — the operator runs twice; the message names `R10` as the actual repair |
| no projection, and some agent enforces the constraint internally | `decertify` — the blend breaks it and nothing restores it |
| the projection **per subdomain** | `refuse` — W98's measured failure |
| the projection **before the blend** | `refuse` — that *is* the unrepaired column; L6/C2 is about the order |
| cadence $0$ | `refuse` — declared and not applied |

> **`R12` does not clear `R10`, and `R10` without `R12` is not enough either.**
> `R10` moves the elliptic part out of the agent; `R12` says where the
> composition layer must then put it. Neither alone is the scheme — and the
> control that proves it is the fourth row of the table above: exposed agents
> with **no** global projection run all $120$ macro-steps without leaving the
> band, at $\lVert\nabla\cdot u\rVert = 1.25$. **Stable, and not
> incompressible.** A rollout that never blows up is not the same thing as a
> rollout that solves the equations, and this is the first place in this vault
> where the two come apart.

**W105 is that gap and it is open.** `R10b` fixes the *cadence* at which the
composition layer must apply an exposed agent's elliptic part and assumes it
happens; `R12` is silent on an all-exposed graph because L6/C2's hypothesis
$\mathcal C u_i = 0$ genuinely does not hold there. So a graph can clear both
rules and never apply the operator, which is exactly the $1.25$ above. Declaring
an `assembly.ProjectedAssembly` is what makes it checkable; making that
*required* would move two synthetic fixtures at Tier 9 and Tier 10 from `admit`
and is left as its own row rather than done in passing.

#### The declaration, which is what survives of the original brief

The step is now first-class regardless of which arrangement it sits in, and that
part of the work is unaffected by the retraction above. `assembly.ProjectedAssembly`
carries a partition of unity **and** an `assembly.ConstraintProjection`, declared
together so a case study cannot apply the projection without the record saying so
or declare it without the driver doing it. `L6/C2` states the condition,
`assembly.AssemblyCertificate.conservative` reports it beside `condition`, and
`emit.HarnessParameters.assembly_projection` puts it on every emitted defect —
**W54's fifth instance**, and the one whose consequence is a trajectory that
stops existing rather than a factor in $\tau$.

The statement generalizes past two windows, and written that way it says what the
two-window version only implies. For a linear $\mathcal C$ with
$\mathcal C u_i = 0$ on every subdomain,

$$\mathcal C\Big(\sum_i \chi_i u_i\Big) = \sum_i [\mathcal C, \chi_i]\,u_i
= \sum_i \nabla\chi_i\cdot u_i = \sum_i \nabla\chi_i\cdot(u_i - w)$$

for **any** field $w$, because $\sum_i \nabla\chi_i = \nabla(1) = 0$. **The
residual is a functional of the disagreement**, manufactured inside the overlap
rather than transported into it, which is why no halo width touches it and why it
is nobody's agent defect. Measured on two windows whose fields are *exactly*
discretely divergence-free by construction, the blend's divergence is
$\mathbf{0.42}$ against $10^{-16}$ for each of them, and it returns to $10^{-16}$
when the two are given the same field with the same $\chi$, the same ramp and the
same overlap.

**And $N=1$ is why a ladder was needed.** With one window $\nabla\chi\equiv 0$,
there is no commutator, and the projected assembly and the bare one are *the same
operator* — measured, all four columns at $N=1$ return identical fields. No case
study run at a single size could have found any of this.

#### Two caveats on the operator, both measured

**The instrument and the operator do not share a kernel.**
$\lVert\nabla\cdot u\rVert_{\text{rms}}$ is a wide centred difference on the
unpadded interior; the spectral projection acts on a domain extended downstream
by `PAD_CELLS`. They disagree at the domain edge, and at $N=2$ — where the blend's
commutator is one overlap wide — **that disagreement is the larger of the two**,
so the projected column reads a *higher* divergence than the bare one ($0.0216$
against $0.0132$, both flat). From $N=6$ up the commutator dominates and the
ordering reverses. §19.6's published table already showed this and nothing had
said why.

**The operator has a hypothesis and it is now written down.**
`wake_array.leray_projection`'s note states it: periodic in $y$ and
periodic-compatible in $x$ *only because* the march holds the inlet and both
laterals at $(U_\infty, 0)$ and `_extend` tapers the outflow. On a field that is
large at the boundary it projects onto a different kernel — measured on a random
perturbation field at $N=1$, it moves the state by $6\%$. And because a
declaration cannot assert that an operator is a projection,
`ConstraintProjection.idempotence_defect` measures
$\lVert P(Pu)-Pu\rVert/\lVert Pu\rVert$: $\mathbf{10^{-3}}$ on this case study's
own states, two orders worse outside the hypothesis, with every compile-time
check on the assembly still passing. *A "projection" that is not idempotent is a
smoother, and a smoother applied once per macro-step is a scheme change nobody
declared.* The Neumann alternative — the classical solver's **own** projection,
`RectangularNS._project`, which drives the measured divergence to $10^{-14}$ — is
on the record as tried and worse: band left at $33$ against the spectral one's
$51$.

## 19.7 W54 — the harness on the emitted defect

Four composition-layer defects have now worn an agent's label, and the depth tag
caught none of them: the decomposed pressure solve ($99.8\%$ of the first composed
step's error), the partition of unity at full weight at the artificial edge (a
factor of $200$ in $\tau$), the exchange cadence (two orders in $\tau$), and
**W98** (transport and pressure per window). §19.6's frozen-force blow-up is a
fifth candidate of the same shape.

`emit.HarnessParameters` now carries overlap, $\chi$'s shape, the exchange
interval and cadence, and the elliptic placement; `BoundTerms.harness` holds it;
`RunArtifact.validate` **refuses** an artifact whose $\tau$, $\sigma$ or $\gamma$
is a measured number and whose harness is absent; and `HarnessParameters.from_graph`
derives it from the `CaseGraph` so it cannot disagree with what the compile ran.
`PartitionOfUnity` and `GridPartitionOfUnity` gained `ramp_cells` and `profile`,
because *convex* is admissibility and the **ramp** is accuracy.

The row's own definition of done is *"a test that changing a harness parameter
changes the attribution"*. It is
`test_changing_a_harness_parameter_changes_the_attribution`: one fixed set of
restriction defects, shaped the way a real one is, attributed under two partitions
of unity differing only in $\chi$'s ramp. The chi-weighted constant moves and the
**depth tag is identical on both** — the row in one line.

The harness this run emits, at every rung:

| field | value |
|---|---|
| `overlap_cells` | $16$ |
| `chi_shape` | partition-of-unity, ramp $8$ cells |
| `exchange_dt` | $0.2$ |
| `exchange_cadence` | $1$ |
| `elliptic_placement` | `agent` |

That last row is the one §19.6 is about. The ladder varies exactly one harness
parameter by design — the tiling — so anything else moving in the tag is a bug
the schema now shows.

## 19.8 W81 — $\beta_{\min}$, derived where it can be and reported where it cannot

The row: $\beta_{\min}$ is *the whole discriminating power* of the plug-in
guarantee, `certify_substitution` took it from the caller with no default, and the
one sibling default is $10^{-12}$ at which every seam in the vault is blind. Both
branches the row allows are now built.

**Derived.** `composition.beta_min_from_tolerance` returns
$\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ over the terms that carry a
**positive** scale — W84's caveat is why the positivity filter is there, because a
composition of exact solvers has $\tau=0$ and a plain minimum sets the tolerance
to zero, at which every certificate passes and none of them sees anything. That is
the $10^{-12}$ default arriving from the other direction.

**Reported.** Both `blind` and `passes` are monotone in $\beta_{\min}$, so the
whole verdict function is two numbers, and `SubstitutionCertificate` now exposes
them:

$$\beta_{\min} < \beta - \lVert S_i\rVert \;\Rightarrow\; \textbf{blind};\qquad
\beta_{\min} > \beta - \lVert\Delta\rVert \;\Rightarrow\; \textbf{refuse}$$

With no tolerance supplied the certificate returns `admit-uncertified` and a
message naming both thresholds, which is strictly more than a verdict at a number
nobody derived. And the sibling default is gone: `schur_complement`'s parameter is
renamed `beta_int_floor`, because it guards a **singularity on the internal
block** — where $10^{-12}$ is exactly right, since the question there is whether a
matrix is invertible. Two numbers had one name and only one of them should ever
have had a default.

Measured on one wake seam per rung, WindowNS $\to$ Poseidon-T:

| $N$ | seam | $\beta$ | $\lVert S_i\rVert$ | $\lVert\Delta\rVert$ | visible above | refused above | $\varepsilon_{\text{tol}}$ | derived verdict |
|---|---|---|---|---|---|---|---|---|
| $2$ | `x0r0_bypass` | $4.708\times10^{-3}$ | $0.08108$ | $0.08074$ | $-0.0764$ | $-0.0760$ | $0.8825$ | `refuse` |
| $6$ | `x1r0_bypass` | $4.543\times10^{-3}$ | $0.08487$ | $0.08466$ | $-0.0803$ | $-0.0801$ | $0.8486$ | `refuse` |
| $12$ | `x1r0_bypass` | $4.577\times10^{-3}$ | $0.08507$ | $0.08486$ | $-0.0805$ | $-0.0803$ | $0.6706$ | `refuse` |
| $24$ | `x4r0_bypass` | $4.431\times10^{-3}$ | $0.08480$ | $0.08459$ | $-0.0804$ | $-0.0802$ | $0.7833$ | `refuse` |

**Both thresholds are negative at every rung, and that is the informative case.**
$\lVert\Delta\rVert$ exceeds $\beta$ by a factor of $19$, so the swap is refused at
*every* admissible $\beta_{\min}$ and is **never blind** — the whole non-negative
axis is inside the informative window. The derived $\varepsilon_{\text{tol}}$ and
the swept values $\{0,\ 10^{-12},\ 0.01,\ 0.1,\ 0.25,\ 0.5\}$ agree on `refuse` at
every rung, which is the cross-check that the derivation is not doing something
the sweep would not.

Set against **W97**, where a rotor seam is blind at every $\beta_{\min}$ up to
$81\times$ the fluid block's norm, the two thresholds now separate the two seam
types cleanly: a fluid–fluid seam is maximally informative and a field-to-lumped
one is blind, and both statements are two numbers rather than a verdict.

$\tau$ and $\sigma$ at the same seams, harness-tagged (§19.7) and depth-tagged:

| $N$ | $\tau$ | $\sigma$ | $\Xi$ |
|---|---|---|---|
| $2$ | $4.544$ | $0.8825$ | $0.03128$ |
| $6$ | $12.70$ | $0.8486$ | $0.02101$ |
| $12$ | $12.85$ | $0.6706$ | $0.02190$ |
| $24$ | $8.432$ | $0.7833$ | $0.02230$ |

$\tau$ is **not monotone in $N$** and should not be read as a scaling quantity: it
is the checkpoint's infidelity at one seam in one local flow, and which flow that
seam sits in changes with the layout. $\gamma$ is **not measured** here and is
recorded as absent rather than as zero — a halo scheme poses no interface solve,
so there is no solve infidelity localized at a seam, and what plays $\gamma$'s
part is the assembly, measured globally in §19.2 as the gap between the composed
defect and the bound.

## 19.9 Where the compiles land, and how the cost of reporting grows

| $N$ | seams | agents | WindowNS | Poseidon-T | probed WindowNS | probed Poseidon-T |
|---|---|---|---|---|---|---|
| $1$ | $0$ | $1$ | `RecordError` | `RecordError` | — | — |
| $2$ | $3$ | $2$ | `refuse`, 1 / 4 | `admit-uncertified`, 0 / 13 | 1 / 6 | 0 / 16 |
| $6$ | $13$ | $6$ | `refuse`, 1 / 4 | `admit-uncertified`, 0 / 28 | 1 / 17 | 0 / 50 |
| $12$ | $27$ | $12$ | `refuse`, 1 / 4 | `admit-uncertified`, 0 / 48 | 1 / 36 | 0 / 95 |
| $24$ | $62$ | $24$ | `refuse`, 1 / 4 | `admit-uncertified`, 0 / 95 | not run | not run |

*(refusals / decertifications; the probed compile is skipped above $30$ seams,
and the unprobed one runs at every rung so the counts are comparable.)*

**The classical column refuses at every rung, always on `L2/R10`, always exactly
one refusal** — and §19.6 is what that refusal is about.

**The decertification count is exactly linear in the graph, with slope one per
seam and one per agent.** `L3/C8` fires once per seam ($3, 13, 27, 62$), `L1/C8`
once per agent ($2, 6, 12, 24$), and everything else is a constant $8$ — nine from
$N=6$ on, where `L2/C2/W58` joins. So

$$\#\text{decertifications} \;=\; \#\text{seams} + \#\text{agents} + 9$$

with no cross terms. **The compiler's own reporting is $O(N)$ and not $O(N^2)$**,
which is a small result and worth having: an audit trail that grew quadratically
would be unreadable at rung 9 whatever the physics did.

The envelope stamp is unchanged across the ladder — E1–E4 and E6 `holds`, E5 and
E7 `unchecked` — and `unmeasured` is the same three rows at every rung
(`L (W1)`, `beta (W2)`, `p_decline (W36)`).

## 19.10 Wall time — Claim B's other half, and it does not come out the same way

Claim B has two halves. The error is §19.2; the other is **$O(1)$ integration
work per added agent**, and a per-agent cost that climbs with the graph falsifies
it exactly as a super-linear error would.

**The instrument had to be built twice, and the first version's failure is the
reason to trust the second.** Timing the driver's own marches is provenance and
not measurement: its rungs are marched at different times and some are resumed
from a cache. Running every rung back to back in one process is not enough
either — this is a shared desktop and the load moves *inside* a pass. Two passes
of that version, same code, same inputs, minutes apart, returned $27.0$ and
$103.1\ \mathrm{ms}$ per agent for the same rung, and per-agent ratios of
$4.49\times$ and $3.03\times$. A ratio that moves by half its own value between
two runs is not a measurement of anything.

`scripts/w100_timing.py` therefore **interleaves** — one repeat visits every
(rung, expert) row in turn, so a transient is spread across all rows rather than
charged to whichever was unlucky — takes the minimum over repeats, and **reports
its own reproducibility**. Over six interleaved repeats the worst best-to-second
spread is **$6.0\%$**, against a stated limit of $25\%$, and the two ratios below
are $1.34$ and $1.14$ in log units against a log-noise of $0.058$: **separable by
more than twenty times the noise.** The same script at two repeats reported
$34\%$ and refused to quote a ratio, which is the behaviour that makes the $6\%$
worth believing.

| $N$ | agents | cells | WindowNS | Poseidon-T | sub-steps | WindowNS per sub-step |
|---|---|---|---|---|---|---|
| $1$ | $1$ | $16384$ | $81.5$ | $399.7$ | $17$ | $4.79$ |
| $2$ | $2$ | $30720$ | $88.8$ | $270.1$ | $21$ | $4.23$ |
| $6$ | $6$ | $84480$ | $125.5$ | $154.4$ | $21$ | $5.98$ |
| $12$ | $12$ | $163328$ | $329.1$ | $109.8$ | $21$ | $15.67$ |
| $24$ | $24$ | $319232$ | $340.8$ | $86.1$ | $21$ | $16.23$ |

*(milliseconds per macro-step per agent; minimum of six interleaved repeats.)*

**The two columns answer oppositely.**

$$\text{per-agent cost},\ N{=}24 \text{ against } N{=}2:\qquad
\textbf{WindowNS } 3.84\times,\qquad \textbf{Poseidon-T } 0.32\times$$

- **The frozen checkpoint beats $O(1)$.** Its per-agent cost *falls* by
  $3.1\times$ across the ladder, because one batched forward pass amortizes its
  fixed overhead over more windows. Composition gets **cheaper** per agent as the
  graph grows, which is more than Claim B asks for.
- **The classical column does not.** Its per-agent cost *rises* by $3.84\times$,
  with the jump between $84$k and $163$k cells.

**And it is not the CFL, which is the one confound worth eliminating.**
`WindowNS` chooses its sub-step count from the advective CFL, so a bigger farm
with deeper wakes would sub-step more — and that would be the *flow* getting
faster, not the composition costing more. The instrument records it: **$21$
sub-steps at every rung from $N=2$ up, at an identical $u_{\max} = 1.302$.** So
the per-sub-step ratio is the same $3.84\times$, and the arithmetic per agent is
constant by construction — same operator, same window size, same sub-step count,
and a composed step that is per-window and embarrassingly parallel.

> **So the classical column's $3.84\times$ is the memory hierarchy and not the
> composition.** The work per added agent is $O(1)$; the *time* is not, because
> $12$ windows of $128^2$ in double precision with the solver's temporaries stop
> fitting where $6$ did. That is a real cost of scale on this hardware and it is
> not what F3 is about.

**F3 — integration hours per added expert — is therefore measured for the first
time and the answer is split**: favourable and better than $O(1)$ for the frozen
checkpoint, $O(1)$ in work but not in wall time for the classical solver, and
neither statement transfers off this host. What the earlier failure bought is the
knowledge of how far it does not: a factor of four, between two runs of one
script.

## 19.11 Closed this tier / opened by it

| # | verdict |
|---|---|
| **W58** | **`done`, and larger than the row.** The provenance field, the four named forms and the compile-time disjointness check are built; and the measurement found a **second hypothesis nobody had stated** — the surrogate's *aggregation* — which moves the ratio by $3\times$ where the stated one moves it by $4\%$. A third gap is named and left: disjointness of the *declared* contaminated sets is necessary and not sufficient, because $D_i$ has a tail outside them |
| **W54** | **`done`.** `HarnessParameters` on `BoundTerms`, derived from the graph, refused at emit when a measured defect lacks it, and a test that a changed harness moves the attribution while the depth tag does not |
| **W81** | **`done`, both branches.** $\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ over positive terms is the derived candidate; absent one, the certificate reports `visible_above` and `fails_above` instead of a verdict; and the $10^{-12}$ sibling is renamed `beta_int_floor` because it guards a different quantity |
| **W100** | **`done` 2026-08-31, and the repair is not the one the row proposed.** The mechanism stands; the positive control **does not survive its own length**. At $120$ macro-steps a global projection *added* to agents that already project is worse than none — band left at $51$ against $74$ — because the pressure is applied twice, and the Neumann variant is cleaner by $78\times$ in $\lVert\nabla\cdot u\rVert$ and dies soonest, so the divergence is not the proximate cause. What holds $120$ steps is **R10's own prescription**: the elliptic part *moved* out of the agent, applied once to the assembled field. Built: `L6/C2`, `ProjectedAssembly`, **`R12`**, the certificate's fifth field and the harness's sixth. Opens **W105** |
| **W105** | `open` — **a graph can clear `R10` and `R10b` and never apply the elliptic part at all.** Measured: the exposed classical column with no global projection runs all $120$ macro-steps without leaving the band, at $\lVert\nabla\cdot u\rVert_{\text{rms}} = 1.25$ against the projected column's $0.066$ — **stable, and not incompressible**. `R12` is silent there because L6/C2's hypothesis genuinely fails on an all-exposed graph. The fix is known and not free: requiring a `ProjectedAssembly` moves two synthetic fixtures at Tier 9 and Tier 10 off `admit` |
| **W102** | `open` — a sub-linear exponent measured on a ladder whose **physics grows with its size** is weaker than it looks. §19.3's control puts the interface-only exponent at $0.14$–$0.20$; no other case study in this vault separates the two, and every scaling claim made before this one conflated them |
| **W103** | `open`, small — `restriction_defect_bound` took subdomain-local arrays without complaint because on a uniform tiling they are all the same length. Guarded by an optional `n_global`, which is opt-in, so the silent path still exists for a caller who does not pass it |

## 19.12 What this tier says about the ones before it

| Claim | Where | Now |
|---|---|---|
| `L2/R10` refuses to decompose an `embedded` elliptic subsolve | Tier 0 §8.2 | **vindicated, and more sharply than §19.6 first claimed.** The refused graph goes unstable at $N\ge6$ — and *patching around it with a global projection makes it worse*. The repair is to obey the rule: with the elliptic part actually EXPOSED and one projection after the blend, the column holds $120$ macro-steps and is the first classical column here R10 does not refuse |
| *"assembly is the least examined layer and carries a measured 19% of total error"* | `assembly.py` | it now carries a second **condition**. L6/C1 and L6/C2 are independent — every convex partition here satisfies the first and violates the second — and the residual L6/C2 names is real without being what ends the rollout |
| W98, *"transport and pressure are global operations"* | §18.5.1 | the **pressure** half generalizes past the checkpoint: any composed classical column needs it too, and W98's repair is why the Poseidon column is the stable one here |
| §10's *"the chi-weighted bound is tight to 0.2%"* | §10 | $4\%$ on this geometry, and **flat over $24\times$** — the transferable property is the flatness, not the $0.2\%$ |
| W49's $\Pi$ | §9.2 | **saturates at $0.7250$ from $N=6$ on**, so it is a property of the tiling's local pattern rather than of its size, and $C_\mu$ transfers along the ladder unchanged |
| W58, *"they agreed to 1.00007 anyway, which is luck"* | Tier 10 | right that it was luck, and **wrong about which hypothesis the luck was in**: the agreement survives the disjointness failure and does not survive the aggregation |
| W76 / W97, the blind certificate | §15.3, §18.7 | the two thresholds now separate the seam types: a fluid–fluid seam is informative on the whole admissible axis, a rotor seam is blind on all of it |
| Tier 18's published state | §18.5 | measured at a state on a **diverging trajectory**; the numbers stand, the trajectory does not continue, and $60$ steps is $10$–$20$ short of where it would have shown |
| §19.6's own positive control | §19.6 | **the same mistake, one level down and the same day.** It was measured over $20$ macro-steps and dies at $38$. A repair marched no further than the failure it repairs is not a repair, and this is now twice |

## 19.13 What the picture says, and it is not what the numbers say

[[case-study-wake-array-atlas-0.1]]'s most transferable finding is that **a
rendering of the state is a measurement instrument** — W98 was smooth, bounded,
physically shaped, in the right units, passed every control, and was found by
watching an animation. So `scripts/w100_frames.py` renders each rung's final
state three ways, plus the two difference fields, at full resolution
(`out/w100/look_<rung>.png`), and this was done before any exponent above was
believed.

**Nothing of W98's class is present.** No wake stands upstream of a turbine, no
wake wraps onto its own window, and the freestream band holds. All three fields
at $N=24$ read as a wind farm: twelve turbines in four rows, the L motif visible,
wakes propagating and recovering downstream.

**The difference fields are where the run's own story is written, and the two
columns are not merely different in size — they are different in *shape*.**

| | what the difference against the monolith looks like |
|---|---|
| **WindowNS composed** | a **checkerboard of $\pm0.25$ blobs organized on the window grid** — vertical banding at the $3.5\,D$ stride and horizontal banding at the row seams, saturating the colour range over a large fraction of the domain |
| **Poseidon-T composed** | **smooth wake-shaped lobes hugging each shear layer**, with no window-grid structure anywhere |

That is the two error sources telling each other apart without being asked. The
classical column's error is organized on **the cut**, which is what an
assembly-injected divergence looks like and is §19.6's mechanism seen rather than
inferred. The checkpoint's error is organized on **the physics**, which is model
infidelity and has nothing to do with the partition.

Two consequences worth having.

**The picture corroborates the attribution the numbers make.** §19.2 says the
classical column's defect grows with exponent $0.85$ and the checkpoint's with
$0.48$, and §19.6 says the classical column's error is the assembly's. Neither
statement contains the word *cut*, and the difference panel does: it puts the
classical error on the window lattice at every rung.

**And it shows what an RMS hides.** At $N=24$ the classical composed field
differs from the monolith by up to **$25\%$** in large coherent regions, against a
per-cell RMS of $0.146$. The composed field is visibly blotchier than the
monolith and carries over-speed regions ($u > 1.4$) the monolith does not have —
at macro-step $60$, which §19.6 places about eight steps from the divergence
onset. **The field was already visibly wrong at the step Tier 18 published**, and
looking would have said so.

> The rule the parent case study wrote for itself held again: **look at the
> largest one before believing any number from it.** Here the picture did not
> overturn a number — it named which of two mechanisms each column's number
> belongs to, which nothing in the tables could do.

## The mechanism tally

Tier 18 concluded that *what made all four findings visible is the same thing:
real geometry*. This tier's shape is different and it is worth naming, because
three of its findings arrived the same way.

**The measurement that decides is the one that removes a variable, and in each
case the removed variable was one nobody had noticed was moving.** §19.3 removes
the physics and the exponent falls by a factor of five. §19.6's positive control
removes the elliptic placement and an instability disappears. §19.5 removes the
aggregation and W58's stated hypothesis turns out to be the second-order one. In
all three the *headline* measurement was correct and uninterpretable on its own.

**A fourth arrived after the tier was written up, and it is the one to keep.**
§19.6's repair was a *positive control*, and a positive control is a measurement
of a scheme nobody has to run. Marched to the length of the failure it repairs,
it fails — worse than the thing it repairs. **The horizon a control is measured
over is a parameter of the control**, it was $20$ steps here against a failure at
$38$, and nothing in the framework asks for it. That is the second time this tier
has caught a number that was true exactly as far as it was marched.

> **A ladder measures a derivative, and a derivative taken while two things move
> measures neither.** The brief said to hold $\mathrm{d}x$, $\Delta t$, the
> overlap, the ramp, the viscosity and the disk model fixed, and this run did. It
> did not say to hold the *flow* fixed, because nobody had noticed that growing
> the farm grows the physics — and that control is worth more than the headline
> it qualifies.

---

# 20. Does a certificate travel with the expert? — 2026-08-31

Full record here; worklist rows on [[gap-worklist]] Tier 20. Reproduce with
`python scripts/w106_reuse_probe.py --steps 110`; artifacts
`out/w106/w106.json`, `out/w106/state_N6.npz`, `out/w106/state_N12.npz`.
New case study `atlas/cases/reuse_probe.py`, the **eighth** real one, and the
first whose *variable* is the state a measurement is taken at rather than
anything about the graph.

> **Dating.** Marches, probes and write-up all 2026-08-31, verified against the
> environment clock before stamping, per Tier 15's rule.

[[case-study-ladder-to-f1]] §4 names **CS-8** as the only rung that tests the
*foundation-model* claim rather than the coupling claim, and
[[f1-pathmap-and-end-goal]] §3.1 flags rung 4 as *the one most likely to be
skipped by accident*. Sharpened by Tiers 14–19 the question is not *"does the
expert work in a new scenario"* but:

> **Is a substitution certificate a property of the expert, or of the state it
> was probed at?**

It is not a philosophical question. `probe.probe_block` linearizes about
`probe_base`, so for a nonlinear expert the block it returns is the **tangent
map at that point** — [[atlas-and-standard-dd-theory]]'s nonlinear-substructuring
section says so outright, and adds that *every* quantity read off $S$ is local to
that linearization. Nothing in the record has ever said how far the reading
travels. A twenty-agent car carries roughly twenty conformance certificates and
forty seam certificates; if each is state-specific the plug-in claim is
**per-design rather than per-expert**, and the economic argument for a reusable
expert library collapses back into re-certifying every design.

**The answer, in one line: the verdict travels, the norms nearly travel, and the
two numbers that give the certificate its discriminating power do not — they move
by more than their own size.**

## 20.1 The design, and the two controls

One swap, everywhere: `reference.WindowNS` $\to$ Poseidon-T at one fluid agent of
one seam, $55$ cells in all. Four factors and two controls; nothing else moves.

| factor | what varies | what is held |
|---|---|---|
| **state** | macro-step $\in \{0, 10, 30, 60, 110\}$ of one trajectory | the seam, the graph, both experts |
| **seam** | the flow regime the seam sits in | the state, the graph, both experts, the port declaration |
| **geometry** | $6$ windows against $12$ | the seam ID, the state, both experts |
| **replicate** | two seams the *rule* calls the same regime | everything else |

**The probe states come off the R10-compliant classical column and no other.**
`wake_array.exposed_reference_solver` — elliptic part *out* of the agent — with
one `assembly.ProjectedAssembly` after the blend. §19.6 is the reason: the
embedded classical column is not finite past macro-step $82$ at six windows, and
a probe taken on a diverging trajectory measures the divergence. **Tier 18's
published wake-array state is excluded for exactly that reason**, and
`reuse_probe.probe_state_health` refuses a state rather than a write-up.

Both marches reproduce §19.6's column on an independently written driver:

| $N$ | $u_{\max}$ at step $110$ | $\lVert\nabla\cdot u\rVert_{\text{rms}}$ at $110$ | §19.6's own, at $120$ |
|---|---|---|---|
| $6$ | $1.314$ | $0.06560$ | $0.06556$ |
| $12$ | $1.332$ | $0.04043$ | $0.04011$ |

Every probe state is inside the band, and the divergence sanity-check §19's
$N=12$ caveat asks for is satisfied at the **value** — $0.0312$ to $0.0671$
across all ten snapshots — rather than at the ratio. See **W110** for why the
ratio form was inert.

**The seam taxonomy is a rule over the layout, not a list.** `reuse_probe
.regime_of` counts the rotors of a seam's own window-row lying upstream of it,
which is well defined before the march runs, because `rotor_motif` puts every
rotor of a row at the same $y$ and a row's rotors are therefore in line. Five
regimes fall out, and $N=12$ is the smallest rung carrying all of them:

| seam | regime | upstream rotors | modes |
|---|---|---|---|
| `x0r1_full` | near-freestream | $0$ | $33$ |
| `x2r1_full` | shallow-wake | $1$ | $33$ |
| `x2r0_full` | deep-wake | $2$ | $33$ |
| `x2r2_full` | deep-wake — **the replicate**, another row | $2$ | $33$ |
| `x0r0_bypass` | bypass-clean | $0$ | $25$ |
| `x1r0_bypass` | bypass-wake — CS-7's own headline seam | $1$ | $25$ |
| `x1r2_bypass` | bypass-wake — its replicate | $1$ | $25$ |

**Comparisons are made within a port group and never across one.** A `full` face
is $128$ cells and $33$ interface modes and a `bypass` face is $96$ and $25$; a
spread across the two would be a spread across different spaces.
`SeamChoice.comparable_key` is what keeps them apart, and it is checked rather
than remembered.

### Control ZERO — and it passes exactly

At the freestream every window holds the same uniform field, every port of a
group is the same face with the same modes, and the solver is deterministic. So
seams the taxonomy calls **different regimes** must return the **same operator**.

| group | seams | regimes spanned | worst range over all eight quantities |
|---|---|---|---|
| $N{=}12$, `full` | $4$ | near-freestream, shallow-wake, deep-wake | $\mathbf{0.0}$ |
| $N{=}12$, `bypass` | $3$ | bypass-clean, bypass-wake | $\mathbf{0.0}$ |
| $N{=}6$, `bypass` | $3$ | bypass-clean, bypass-wake | $\mathbf{0.0}$ |

Exactly zero, by `==`, not to a tolerance. Had it not been, the regime labels
would be reading the **port** rather than the **flow** and every number in §20.4
would be uninterpretable. This is [[tier0-measurements]] §8's lesson applied
before the fact rather than after it, and it costs one rung of the grid.

It also has a consequence for the arithmetic below: **the freestream is control
ZERO, not a level of the seam, replicate or geometry factors.** Those cells are
identically zero by construction, and a median that includes them halves itself
— which would make the taxonomy look better than the measurement says it is. The
three cross-sectional factors exclude step $0$; the **state** factor keeps it,
because there the freestream is a genuine level and most of what the factor
measures.

### Control FLOOR — exactly zero too, which costs the framework a test

The identical `(seam, state)` cell probed twice: state reloaded from the `.npz`,
experts constructed fresh, graphs rebuilt, operators re-probed. Four cells, two
rungs, both port groups:

| rung | seam | step | worst disagreement over the eight quantities |
|---|---|---|---|
| $N{=}6$ | `x0r1_full` | $110$ | $0.0$ |
| $N{=}12$ | `x2r0_full` | $110$ | $0.0$ |
| $N{=}12$ | `x1r0_bypass` | $110$ | $0.0$ |
| $N{=}12$ | `x0r1_full` | $0$ | $0.0$ |

**The whole pipeline is bit-reproducible**, which is the good news and also the
problem. `reuse_probe.travel_verdict` asks *is the movement larger than the
floor*, and against a floor of exactly zero **every** quantity that moved at all
answers `state-dependent` — including one that moved by $0.037\%$. The test the
framework brought to this question does not discriminate, and saying so is
**W106**. What replaces it below is the quantity's **own level** — its range over
the mean of its magnitude — and the **replicate**, which prices what the regime
label buys. Both are reported at every cell and neither is a floor.

## 20.2 The answer — the norms travel and the thresholds do not

Median over the factor of each quantity's range, as a percentage of the level the
quantity sits at. Read the columns against each other, not the rows.

| quantity | **state** | **seam** | **replicate** | **geometry** |
|---|---|---|---|---|
| $\beta$ | $6.46\%$ | $3.99\%$ | $0.769\%$ | $0.340\%$ |
| $\lVert S_i\rVert$ | $2.49\%$ | $1.99\%$ | $0.037\%$ | $0.199\%$ |
| $\lVert\Lambda^{\text{expert}} - \Lambda^{\text{ref}}\rVert$ | $3.63\%$ | $2.97\%$ | $0.228\%$ | $0.156\%$ |
| $\Xi$ (block) | $\mathbf{55.9\%}$ | $\mathbf{53.9\%}$ | $\mathbf{20.5\%}$ | $1.61\%$ |
| $\Xi$ (seam) | $69.1\%$ | $95.2\%$ | $36.4\%$ | $11.2\%$ |
| `visible_above` | $\mathbf{155\%}$ | $79.2\%$ | $3.56\%$ | $2.70\%$ |
| `fails_above` | $\mathbf{155\%}$ | $96.0\%$ | $5.32\%$ | $3.66\%$ |
| $\lVert\Delta\rVert/\beta$ | $10.2\%$ | $7.03\%$ | $0.631\%$ | $0.205\%$ |

**Three orders of magnitude separate the most portable quantity from the least,
and they are all fields of one certificate.** $\lVert S_i\rVert$ moves $2.5\%$
with the state; `fails_above` moves $155\%$ — *more than its own size*. A record
that reports "the certificate" as one object, portable or not, is averaging over
that spread. **W107.**

**The mechanism is arithmetic, and it says the result is not an accident of this
expert.** Both thresholds are a *difference of two nearly equal large numbers*:

$$\texttt{visible\_above} = \beta - \lVert S_i\rVert, \qquad
\texttt{fails\_above} = \beta - \lVert\Delta\rVert$$

On this seam $\beta \approx 0.21$ and both norms are $\approx 0.22$, so the
thresholds are $\sim 10^{-3}$ — two orders below the terms they are built from. A
$3\%$ movement in $\lVert\Delta\rVert$ against a $6\%$ movement in $\beta$
therefore lands as a $155\%$ movement in their difference. **Any certificate
whose verdict is decided near $\lVert\Delta\rVert \approx \beta$ inherits that
amplification** — and that is the regime a *good* substitution is in. The closer
a swap is to admissible, the less its certificate travels.

**[AI Inference]:** if that reading is right, portability and admissibility are in
tension by construction, and the quantity a record should carry is not the
threshold but the *pair of norms it is a difference of*, each of which travels to
a few percent. This is derived from the definitions and the measured magnitudes
above rather than measured separately, and it should be checked at a seam where
$\lVert\Delta\rVert \ll \beta$ — which this vault does not yet have.

## 20.3 The state factor, seam by seam

The full group at $N=12$, down one trajectory. $\Delta$ is
$\lVert\Lambda^{\text{expert}} - \Lambda^{\text{ref}}\rVert$ at the target block.

| seam / regime | step | $\beta$ | $\lVert\Delta\rVert$ | $\Xi$ | `visible_above` | `fails_above` | $\lVert\Delta\rVert/\beta$ |
|---|---|---|---|---|---|---|---|
| `x0r1_full` near-freestream | $0$ | $0.2147$ | $0.2154$ | $0.006945$ | $-7.23\times10^{-4}$ | $-7.35\times10^{-4}$ | $1.00342$ |
| | $30$ | $0.2160$ | $0.2165$ | $0.007285$ | $-8.33\times10^{-4}$ | $-5.48\times10^{-4}$ | $1.00254$ |
| | $110$ | $0.2150$ | $0.2164$ | $0.006997$ | $-1.71\times10^{-3}$ | $-1.41\times10^{-3}$ | $1.00654$ |
| `x2r1_full` shallow-wake | $0$ | $0.2147$ | $0.2154$ | $0.006945$ | $-7.23\times10^{-4}$ | $-7.35\times10^{-4}$ | $1.00342$ |
| | $30$ | $0.1739$ | $0.2230$ | $0.01258$ | $-0.04697$ | $-0.04912$ | $1.28245$ |
| | $110$ | $0.1776$ | $0.2239$ | $0.01176$ | $-0.04456$ | $-0.04622$ | $1.26020$ |
| `x2r0_full` deep-wake | $0$ | $0.2147$ | $0.2154$ | $0.006945$ | $-7.23\times10^{-4}$ | $-7.35\times10^{-4}$ | $1.00342$ |
| | $30$ | $0.1889$ | $0.2218$ | $0.01076$ | $-0.03194$ | $-0.03295$ | $1.17443$ |
| | $60$ | $0.1649$ | $0.2249$ | $0.01303$ | $-0.05760$ | $-0.06007$ | $1.36438$ |
| | $110$ | $0.1716$ | $0.2248$ | $0.01225$ | $-0.05076$ | $-0.05322$ | $1.31019$ |

Per-seam totals over the whole trajectory, as a fraction of each quantity's own
level ($N=12$):

| seam | regime | $\beta$ | $\lVert\Delta\rVert$ | $\Xi$ | `fails_above` | $\lVert\Delta\rVert/\beta$ |
|---|---|---|---|---|---|---|
| `x0r1_full` | near-freestream | $0.594\%$ | $0.505\%$ | $4.77\%$ | $107\%$ | $0.398\%$ |
| `x0r0_bypass` | bypass-clean | $2.48\%$ | $0.893\%$ | $6.65\%$ | $117\%$ | $3.30\%$ |
| `x1r0_bypass` | bypass-wake | $6.54\%$ | $3.79\%$ | $72.3\%$ | $157\%$ | $10.2\%$ |
| `x2r1_full` | shallow-wake | $21.3\%$ | $4.01\%$ | $55.9\%$ | $167\%$ | $24.0\%$ |
| `x2r0_full` | deep-wake | $26.1\%$ | $4.31\%$ | $59.4\%$ | $199\%$ | $30.8\%$ |
| `x2r2_full` | deep-wake | $27.6\%$ | $4.55\%$ | $95.0\%$ | $201\%$ | $32.7\%$ |

**Two readings, and the second is the one to keep.**

**The trajectory does most of its damage in the first thirty macro-steps and then
stops.** At `x2r0_full` the thresholds move by a factor of $44.8$ between the
freestream and step $30$, and by $1.6$ between step $30$ and step $110$. So the
state-dependence measured here is overwhelmingly *freestream versus developed*,
not *developed versus developed* — the more hopeful of the two possible shapes,
and the one a library could actually exploit.

**And the seam whose flow does not develop does not move.** `x0r1_full` is
near-freestream by construction: no rotor upstream in its row. Over $110$
macro-steps its $\beta$ moves $0.594\%$ and its $\Xi$ moves $4.77\%$ — against
$26.1\%$ and $59.4\%$ at the deep-wake seam sitting in the same graph, on the
same trajectory, at the same steps. **The certificate is state-dependent exactly
to the extent that its own seam's flow is.** That sounds circular and is not: it
says the dependence is *local*, and a seam in undisturbed flow inherits a
portable certificate even inside a graph whose other seams do not. The ordering
of that table is the rotor count of §20.1's rule, unchanged.

## 20.4 The seam factor, and the replicate that prices the label

At a fixed developed state, seams of one port group in different regimes. Every
row is macro-step $110$, $N=12$:

| seam | regime | $\beta$ | $\Xi$ | `fails_above` | $\lVert\Delta\rVert/\beta$ | $\tau$ | $\sigma$ |
|---|---|---|---|---|---|---|---|
| `x0r1_full` | near-freestream | $0.2150$ | $0.006997$ | $-1.41\times10^{-3}$ | $1.00654$ | $115.0$ | $3.957$ |
| `x2r1_full` | shallow-wake | $0.1776$ | $0.01176$ | $-0.04622$ | $1.26020$ | $27.92$ | $0.3593$ |
| `x2r0_full` | deep-wake | $0.1716$ | $0.01225$ | $-0.05322$ | $1.31019$ | $40.58$ | $0.9776$ |
| `x2r2_full` | deep-wake *(replicate)* | $0.1703$ | $0.01989$ | $-0.05489$ | $1.32236$ | $60.72$ | $1.552$ |
| `x0r0_bypass` | bypass-clean | $0.2095$ | $0.007392$ | $-7.72\times10^{-3}$ | $1.03686$ | $17.85$ | $0.2322$ |
| `x1r0_bypass` | bypass-wake | $0.2012$ | $0.01299$ | $-0.02250$ | $1.11184$ | $8135$ | $63.61$ |
| `x1r2_bypass` | bypass-wake *(replicate)* | $0.2013$ | $0.01349$ | $-0.02288$ | $1.11366$ | $3535$ | $30.91$ |

**The regime label does most of the work, and the replicate is how we know.**
Two seams the rule calls the same regime, in different rows of the same graph at
the same station, against the spread across regimes:

| quantity | seam factor | replicate | what the label buys |
|---|---|---|---|
| $\lVert S_i\rVert$ | $1.99\%$ | $0.037\%$ | $\mathbf{53\times}$ |
| $\lVert\Delta\rVert$ | $2.97\%$ | $0.228\%$ | $13\times$ |
| `visible_above` | $79.2\%$ | $3.56\%$ | $22\times$ |
| `fails_above` | $96.0\%$ | $5.32\%$ | $18\times$ |
| $\lVert\Delta\rVert/\beta$ | $7.03\%$ | $0.631\%$ | $11\times$ |
| $\beta$ | $3.99\%$ | $0.769\%$ | $5.2\times$ |
| $\Xi$ (block) | $53.9\%$ | $20.5\%$ | $\mathbf{2.6\times}$ |

So **a label computable from the layout with no run at all captures between five
and fifty times the seam-to-seam variation** — for every quantity except one.

**The exception is $\Xi$, and it is the interesting one.** The two deep-wake
replicates at step $110$ read $\Xi = 0.01225$ and $0.01989$, which is $47.5\%$ of
their own level; the $N=6$ bypass-clean pair reads $0.007271$ and $0.01199$, or
$49\%$. The composability index is a ratio of two norms that each moved a couple
of percent in *opposite* directions, so it inherits the sum of both — and it is
the quantity [[expert-library-atlas-0.1]] and [[probed-dtn-coupling]] §4.5
propose to **rank a library by**. A ranking axis whose same-label reproducibility
is $20$–$49\%$ can reorder two experts that are genuinely a fifth apart. **W107**
carries this.

## 20.5 The geometry factor — the cheapest axis, by an order of magnitude

The same seam IDs in a $6$-window graph and a $12$-window one, at matched
macro-steps. Same declared port, same face, same station; what differs is how
much graph is around it.

| quantity | median range across the two rungs | as a fraction of its level |
|---|---|---|
| $\beta$ | $7.11\times10^{-4}$ | $0.340\%$ |
| $\lVert S_i\rVert$ | $4.36\times10^{-4}$ | $0.199\%$ |
| $\lVert\Delta\rVert$ | $3.37\times10^{-4}$ | $0.156\%$ |
| $\Xi$ (block) | $1.34\times10^{-4}$ | $1.61\%$ |
| `fails_above` | $4.35\times10^{-4}$ | $3.66\%$ |
| $\lVert\Delta\rVert/\beta$ | $2.18\times10^{-3}$ | $0.205\%$ |

**Doubling the graph costs a twentieth to a fortieth of what developing the flow
costs**, on every quantity. And it is not that graph size is invisible:
`x1r0_bypass` at step $110$ reads $\lVert\Delta\rVert/\beta = 1.11479$ at $N=6$
and $1.11184$ at $N=12$, a difference the pipeline resolves exactly. It is small.

That is the result CS-7 could not have produced, and it is worth naming plainly:
**a certificate is far more portable across designs of different size than across
operating points of one design.** For a library that is the wrong way round from
the convenient answer — reuse across scenarios is what the economic argument
needs, and reuse across sizes is what it gets.

## 20.6 The verdict is invariant, and it is invariant because it is saturated

The certificate as this driver calls it is given no tolerance, so by W81's own
design it returns `admit-uncertified` and the two thresholds instead of a
verdict. **That is a constant by construction and is not a stability result**, so
the artifact labels it as such and answers the real question by sweeping the whole
admissible axis. `blind` and `passes` are each monotone in $\beta_{\min}$, so the
verdict at any tolerance is a function of the two stored thresholds alone
(`reuse_probe.verdict_at`) and needs no re-probe:

$$\beta_{\min} \le \texttt{visible\_above} \Rightarrow \textbf{blind};\qquad
\beta_{\min} > \texttt{fails\_above} \Rightarrow \textbf{refuse};\qquad
\text{otherwise } \textbf{admit}$$

Swept over the admissible axis
$\beta_{\min} \in \{0, 10^{-12}, 10^{-6}, 10^{-3}, 0.01, 0.1, 0.25, 0.5\}$
at all $55$ cells: **`refuse`, everywhere, at every tolerance.** The
verdict travels perfectly.

**It travels because it is saturated, and the margin says how far from failing
that is.** Both thresholds are negative exactly when $\lVert\Delta\rVert > \beta$,
so on the non-negative admissible axis the verdict is `refuse` if and only if that
ratio exceeds $1$. Measured over all $55$ cells:

$$\frac{\lVert\Delta\rVert}{\beta} \in [\,\mathbf{1.00254},\ \mathbf{1.38952}\,],
\qquad \text{never below } 1, \qquad \text{mean } 1.0743,\ \text{sd } 0.1032$$

The minimum is `x0r1_full` at $N=12$, macro-step $30$ — **a quarter of one percent
above the value at which the verdict changes**, and at the seam whose own
state-dependence is the smallest in the study. The maximum is `x2r2_full` at
macro-step $60$, $39\%$ above. **So the invariance of the verdict is the
invariance of a sign, and the distance from that sign change varies by a factor
of $153$ across the grid.** Reporting *"the verdict is a property of the expert"*
on this evidence would be true, and the least informative true thing available.
**W109.**

## 20.7 $\tau$, $\sigma$ and the derived $\beta_{\min}$ — the least portable quantity here

W81's first branch derives the tolerance from the run's own defect terms,
$\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ over the terms carrying a positive
scale. CS-8 asks that number the same question, at `x1r0_bypass`, $N=12$, down
one trajectory:

| step | $\tau$ | $\sigma$ | $\varepsilon_{\text{tol}}$ |
|---|---|---|---|
| $0$ | **UNDEFINED** | **UNDEFINED** | **UNDEFINED** |
| $10$ | $17.49$ | $1.987$ | $1.987$ |
| $30$ | $193.6$ | $28.16$ | $28.16$ |
| $60$ | $4430$ | $69.81$ | $69.81$ |
| $110$ | $8135$ | $63.61$ | $63.61$ |

**$\tau$ moves by a factor of $465$ and the derived tolerance by a factor of
$35$, at one seam, with nothing changing but how long the flow has been
running.** And at the freestream the tolerance does not exist:
`seam_defect_split` refuses a *relative* defect where no power crosses the
reference interface — L4's empty-seam case — which for a uniform stream is the
correct answer, and is recorded as a reason rather than as a zero.

So the two branches of W81 have opposite portability. The **reported** branch —
the thresholds — is a function of two norms that each travel to a few percent,
even though their difference does not. The **derived** branch is a function of
$\tau$, the checkpoint's infidelity in one local flow, which is the single most
state-dependent quantity anywhere in this study. **W81 closed the question *where
does $\beta_{\min}$ come from*; it opened *which state is it the $\beta_{\min}$
for*, and this tier is the answer.** **W108.**

This also re-reads §19.8. That table quotes
$\tau = 4.544, 12.70, 12.85, 8.432$ across four rungs and says $\tau$
*"is not monotone in $N$ and should not
be read as a scaling quantity: it is the checkpoint's infidelity at one seam in
one local flow."* CS-8 measures how much of that is the local flow: a $465\times$
swing at one seam of one graph. **The non-monotonicity in $N$ was not weak
evidence of a scaling law — it was the state, sampled once per rung.**

## 20.8 Where the compiles land, and W105 asserted on every graph

**W105 is that a graph can clear `R10` and `R10b` and never apply the elliptic
part at all.** The rules cannot check it; a case study can, and
`reuse_probe.assert_projected` runs on **every graph this tier builds** — the
$110$ in the grid, the sixteen in the floor control, and the four compiled ones.
All report a `ProjectedAssembly` carrying a `ConstraintProjection` with scope
`global`, stage `after-assembly`, cadence $1$.

| rung | column | verdict | refusals | decertifications | `elliptic_subsolve` |
|---|---|---|---|---|---|
| $N{=}6$ | exposed classical | `admit-uncertified` | $\mathbf{0}$ | $7$ | `exposed` |
| $N{=}6$ | Poseidon-T | `admit-uncertified` | $\mathbf{0}$ | $31$ | `unknown` |
| $N{=}12$ | exposed classical | `admit-uncertified` | $\mathbf{0}$ | $7$ | `exposed` |
| $N{=}12$ | Poseidon-T | `admit-uncertified` | $\mathbf{0}$ | $51$ | `unknown` |

**Zero refusals in either column at either size.** §19.9's table has the
classical column refusing at every rung, always on `L2/R10`, always exactly once;
with the elliptic part actually out of the agent the refusal is gone, and the
seven remaining classical decertifications — `L2/R10` (the *disk* agent's own
`elliptic_subsolve=none`), `L2/C2`, `L4/budget`, `L5/eps_tol`, `L6/W49`,
`L9/E5`, `L8/W56` — are **identical at both sizes**. The checkpoint's count grows
the way §19.9's $\#\text{seams} + \#\text{agents} + 9$ says: $31$ at six agents
against $51$ at twelve, the difference being six more `L1/C8` and fourteen more
`L3/C8`, one per added agent and one per added seam.

## 20.9 What this means for the expert library

Stated against the claim CS-8 exists to test, and no further.

| claim | verdict from this tier |
|---|---|
| *a certificate is a property of the expert* | **false as stated.** Two of its eight quantities move by more than their own size with the probe state |
| *a certificate is a property of the design* | **also false as stated**, and this is the more surprising half: doubling the graph moves everything by under $4\%$, twenty to forty times less than developing the flow |
| *the plug-in claim is per-design* | **not shown.** The verdict is invariant across every cell and every admissible tolerance — but only because it is saturated, at a margin whose distance from the sign change varies $153\times$ |
| *a certificate can be indexed by something cheaper than a re-probe* | **supported, and this is the practical result.** A regime label computed from the layout with no run captures $5$–$53\times$ of the seam-to-seam variation on every quantity but $\Xi$, and the state-dependence is concentrated in the first $30$ macro-steps rather than spread along the trajectory |

**The honest form of the answer is that neither noun in the question is right.**
A certificate is a property of the *expert at a probe state*; the state half
dominates; and the part of it a library can amortize is the part a cheap label
predicts. What a library would have to carry is not one certificate per expert
and not one per design, but **one per expert per regime**, with the regime
derivable from the layout — a cost that grows with the vocabulary of flow
situations rather than with the number of designs searched. That is a weaker
claim than the pathmap's and a very much cheaper one than re-certification.

**[AI Inference]:** the shape of §20.3 — almost all of the movement between the
freestream and step $30$, very little after — suggests the right index is not
*time* but *whether the seam's own flow has developed*, which is a property of the
state a probe already measures. If that holds, a certificate could carry an
admissibility envelope in the probe-base coordinates rather than a scalar, and
`SeamOperator.probe_state_agrees` (**W77**, already derived rather than declared)
is the field it would live on. This is a design proposal fitted to one geometry
at one Reynolds number; nothing here tests it.

## 20.10 Closed this tier / opened by it

| # | verdict |
|---|---|
| **W106** | `open` — **the framework's own "did it move" test does not discriminate on a deterministic pipeline.** The reproducibility floor over four re-probe cells is exactly $0.0$ on all eight quantities, so a movement of $0.037\%$ and a movement of $155\%$ both return `state-dependent`. The yardsticks that do work are the quantity's own level and a same-label replicate, and both are now reported at every cell |
| **W107** | `open` — **the eight quantities of one certificate differ in portability by three orders of magnitude**, and the record reports them as one object. $\lVert S_i\rVert$ moves $2.49\%$ with the state and `fails_above` moves $155\%$. Worse for the library argument: $\Xi$, the axis [[expert-library-atlas-0.1]] proposes to *rank* by, disagrees by $20.5\%$ in the median and up to $49\%$ between two seams the taxonomy calls the same regime |
| **W108** | `open` — **W81's derived $\beta_{\min}$ is the least portable quantity in the study.** $\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ moves by $35\times$ at one seam down one trajectory, $\tau$ by $465\times$, and at the freestream it is *undefined* — `seam_defect_split` correctly refuses a relative defect where no power crosses the reference interface. W81 closed *where does the number come from*; this is *which state is it the number for* |
| **W109** | `open` — **the verdict's invariance is the invariance of a sign.** `refuse` at all $55$ cells and all eight tolerances, because $\lVert\Delta\rVert/\beta > 1$ everywhere — but that ratio ranges $[1.00254, 1.38952]$ and its minimum is a quarter of one percent from flipping. A stability claim read off a saturated verdict is not a stability claim |
| **W110** | `open`, small — **the divergence guard's ratio form is inert where it is armed.** `probe_state_health` compares a state's $\lVert\nabla\cdot u\rVert$ against the first snapshot's, and the first snapshot is the freestream, where the divergence is identically zero — so `div_ratio` is `None` at every cell and only the absolute value was ever checked. The reference should be the first *developed* snapshot. The values themselves are recorded and healthy, so nothing above is affected |

## 20.11 What this tier says about the ones before it

| Claim | Where | Now |
|---|---|---|
| $\tau$ *"is not monotone in $N$ and should not be read as a scaling quantity"* | §19.8 | **right, and the reason is bigger than the sentence.** $\tau$ moves $465\times$ at ONE seam of ONE graph with nothing varying but the macro-step. The four rung values were four states, sampled once each |
| $\Xi$ as the expert library's ranking axis | [[expert-library-atlas-0.1]], [[probed-dtn-coupling]] §4.5 | it moves $55.9\%$ with the probe state and disagrees by up to $49\%$ between same-regime replicates. **The axis is real and its reproducibility had never been quoted**; ranking a library on it needs the state named |
| W81, *"the certificate reports the threshold above which the substitution becomes visible"* | §19.8 | the thresholds are the *right* output and the least portable form of it, because each is a difference of two nearly equal norms. The **norms** travel to a few percent; their difference does not |
| `L2/R10` refuses the classical column at every rung | §19.9 | **not any more.** With the elliptic part exposed and one projection after the blend, both rungs compile with **zero refusals in both columns** — the first real graph in this vault to do so, and the residual seven decertifications are size-independent |
| W93, *"the framework cannot certify a decomposition whose agents are globally-receptive pretrained operators"* | [[case-study-ladder-to-f1]] §1.1 | unchanged, and now priced differently. The halo decertification is size-independent; what a substitution campaign re-pays per design is the **probe**, not the rule |
| Tier 18's published state | §18.5, §19.6 | excluded here by construction, and the exclusion cost something worth recording: every probe state in this tier had to be re-marched, because the only classical trajectory this vault trusts past macro-step $82$ is one day old |
| W77, `probe_state` derived rather than declared | §17 | vindicated as the right design and now load-bearing. Every reading here is indexed by a probe-state string the operator derives, and §20.9's proposal has nowhere else to live |

## The mechanism tally

Tier 19: *the measurement that decides is the one that removes a variable*. This
tier's shape is the mirror image, and it is worth naming as such.

**The measurement that decides here is the one that holds everything still and
varies the thing nobody had called a variable.** Nine tiers have quoted $\beta$,
$\Xi$, $\tau$ and a certificate verdict, each at *one* probe state, and no page
had ever asked what the second reading would be. It is not that the earlier
numbers are wrong — control FLOOR says they reproduce exactly, bit for bit. It is
that **seven of them were a sample of size one from a distribution nobody had
drawn.**

> **A quantity measured once is a quantity whose variability is being asserted to
> be zero.** The floor being exactly $0.0$ is what makes the assertion visible as
> an assertion: this pipeline can tell a $0.037\%$ movement from a $155\%$ one,
> and until this tier it had never been asked to.

And a second one, smaller and cheaper to state. **The control that could have
invalidated the whole design cost one rung of the grid and returned exactly
zero.** Had the four regime seams disagreed at the freestream, §20.4 would have
been measuring the port rather than the flow. Running it took five minutes; not
running it would have been §8's mistake in a new place.

---

## See Also

- [[gap-worklist]] — the rows this page moves, and the ones it deliberately does not
- [[probed-dtn-coupling]] — the construction measured here; §2.1's two failed premises, §4.1's no-op, §4.2's convention dependence
- [[master-error-bound]] — evaluated with measured constants; §4's $1/\beta$ is the amplifier that turns §5's artifact residual into an $8.9\times$ degradation
- [[plug-in-composition-theorems]] — §1.4's attribution theorem, observed rather than argued
- [[interface-transfer-theory]] — §7's null count, contradicted; §4.3's $\dim M$ rule, used as specified
- [[open-problems-atlas-0.1]] — OP-2's drift against a fitted $L<1$; OP-6's probe floor, irrelevant for a deterministic expert
- [[general-coupling-scheme]] — R9, promoted from a multirate rule to a single-rate necessity
- [[schwarz-iteration-atlas-0.1]] — `reference.WindowNS`, the boundary channel this used; §3.1's constant-map result, which the probe reproduces as an operator
- [[atlas-implementation]] — the compiler this ran through
- [[composition-error-theory]] — §9.1's convexity is the condition its "agreement is not correctness" argument needed on the assembly side
- [[generalization-requirements]] --- **G5**, whose W16 row this page's section 10 closes: the locality half derived as L2/C2, the scalarization falsified
- [[end-to-end-architecture-spec]] --- L2's cut rule, and the three-verdict split that makes an `admit` mean something
