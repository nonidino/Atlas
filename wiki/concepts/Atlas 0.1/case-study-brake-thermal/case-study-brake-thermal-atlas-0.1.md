# Brake Thermal — a bound for the multirate lag defect, and a horizon for a gradient

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/case-study-brake-thermal/`)
**Status:** **measured 2026-09-03, written 2026-09-04**, and both dates are on the record because the run finished at 22:01 and the page did not — Tier 15's rule is to verify the environment clock before stamping, and a single date here would have been wrong by one of them. The twelfth real case study, the last row of Phase B, and the first place in this vault where a $\sigma$ term is a **function** rather than a value. Everything here is quoted from `out/w131/w131.json`; nothing is estimated.
**Code:** `atlas/cases/brake_thermal.py` (the two agents, the seam and the interface condition), `scripts/w131_brake_thermal.py` (the driver, ten stages), `tests/test_tier28_brake_thermal.py` (42 tests)
**Related:** [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[gap-worklist]] · [[master-error-bound]] · [[general-coupling-scheme]] · [[end-to-end-architecture-spec]] · [[temporal-error-accumulation]] · [[case-study-ground-effect-atlas-0.1]] · [[case-study-thermal-strain-atlas-0.1]] · [[case-study-seam-placement-atlas-0.1]] · [[tier0-measurements]]

---

## 1. The question, and why it is the last row of Phase B

[[case-study-ladder-to-f1]] §4's CS-11 row states it in the row's own words:

> **W90**: R9 covers the flux transient and is $62.4\times$ smaller than the stale-trace term, which has no rule at all. R4 forbids shrinking the exchange interval below $\max_i \Delta t_i$, and a frozen checkpoint's `dt_native` is not a dial — so the one obvious remedy is the one axis a learned expert cannot move. The candidate is **W17**'s $W>1$ waveform relaxation.

And the gate:

> A $\sigma$ term that is a function of the exchange interval and holds over a swept clock ratio — or an explicit statement that multirate composition is uncertified and by how much, which is the honest fallback and is what `L7/R9/lag` currently decertifies with.

**The first branch is what happened.** §4 gives the function, §5 gives the sweep, and §4.2 gives the control that makes the function usable at a ratio no referent can be built at. The fallback is not needed and is not taken.

**Why this row and not another.** Every pair of subsystems in an F1 car runs on different clocks — a brake disc against its duct, a battery against its coolant, a tyre against the road — and [[case-study-ladder-to-f1]] §6 lists W90 as blocking *"every pair of subsystems on different clocks"*. It is also the only coupling kind in §3's table that had no bound at all: CS-9 supplied the volumetric splitting error, CS-10 the moving-interface price, and multirate had a **refusal** (R9) aimed at the smaller of its two terms.

---

## 2. How it is built

A brake disc against its cooling duct. `ThermoStruct2D` conducts through a $2.5\,\mathrm{mm}$ cast-iron wall — backward Euler over the whole cross-section, `EMBEDDED`, slow — and a low-Mach duct convects past its face, explicit and fast. One `THERM` seam, $32$ cells, and **$500{:}1$ in clock because the physics puts them there**.

| | |
|---|---|
| disc | $80\,\mathrm{mm} \times 2.5\,\mathrm{mm}$ section, $32\times6$ Q1 elements, $231$ nodes, cast iron ($\rho c_p = 3.29\times10^{6}$, $k = 48$) |
| duct | $80 \times 16\,\mathrm{mm}$, $32\times8$ cells, air at $420\,\mathrm{K}$, $U = 40\,\mathrm{m/s}$, $\mathrm{Re} = 20\,677$ |
| friction face | a second Robin at $(h_{\text{pad}}, T_{\text{pad}}) = (250,\ 1150\,\mathrm K)$, standing in for the pad's heat input |
| clocks | disc $\Delta t = 5\times10^{-2}\,\mathrm s$, duct $\Delta t = 1\times10^{-4}\,\mathrm s$ with $5$ sub-steps inside it — **ratio $500$** |
| disc time constant | $\tau \approx 9.6\,\mathrm s$, so $200$ macro-steps is one $\tau$ and $1000$ is five |
| seam | $32$ cells, $M = 32$ (see §2.2), seam base $336.53\,\mathrm K$ |
| release | the disc **cold at $300\,\mathrm K$** in air at $420\,\mathrm K$ — the state §6's sign change turns on |

**One side is a build-repo donor and the other is written here, and saying so is the scope statement.** The disc is `thermostruct2d.ThermoStruct2D` imported verbatim through `thermal_seam.load_solvers`, unmodified, with cast iron passed as a `SolidMaterial` — which is the expert's own parameterization rather than a change to it. The duct is new. It is a *solver* and not a fixture — it marches a PDE, its domain of dependence is finite and declared, its `boundary_response` is a solve — but nobody else has graded it, so `tests/test_tier28_brake_thermal.py` does: against the method of characteristics for pure advection, against the **erf half-space solution** for pure diffusion (to $<15\%$ on eight cells), against its own refinement (first order, ratios $2.09$/$2.14$ against forward Euler's $2$), and for the discrete energy balance its wall flux has to satisfy exactly.

### 2.1 Why the duct is low-Mach, and what that buys

`compressible2d` was the obvious choice — `thermal_seam` uses it, and it carries the same $500{:}1$. It is acoustically limited at $\Delta t = 3.5\times10^{-6}\,\mathrm s$, so a **single-rate referent** over the disc's own transient is $\sim10^{7}$ sub-steps.

> **The one thing this case study cannot do without is the single-rate solve it grades against.** A bound on a lag defect measured against a column that also carries a lag is measuring a difference of two unknowns.

A brake duct runs at $M = 0.12$; the acoustics are not the physics, and dropping them is what makes the control exist. The clock ratio is still what the physics gives — a convective cell transit against a conduction time constant — and it is still $500{:}1$. **The single-rate referent costs $\mathbf{5.2\times}$ to $\mathbf{6.3\times}$ the multirate column over the same $10\,\mathrm s$ of physical time**, which is the economic half of the multirate question and is measured here rather than asserted. It is quoted as a *ratio measured twice on the same box* rather than as a wall time, because the box throttles: an identical duct workload repeated eight times ran $62\,\mathrm{ms}$ per interval for four repetitions and $88$–$101\,\mathrm{ms}$ for the next four, so an absolute second here would be a statement about a laptop (**W104**).

**The turbulence closure is textbook and its exponent is a check.** A molecular-conductivity duct gives a wall coefficient near $15\,\mathrm{W/(m^2 K)}$ and a disc time constant of fifteen minutes, which is not a brake. `eddy_diffusivity` is the classical linear-in-wall-distance mixing model with $\kappa = 0.41$, $\mathrm{Pr}_t = 0.85$ and Blasius' friction law — three constants of the literature, none fitted here. What comes out is $494$ to $1575\,\mathrm{W/(m^2 K)}$ over the sweep and, unprompted, a $U^{0.836}$ scaling against **Dittus–Boelter's $U^{0.8}$**. Nothing told the closure that exponent.

### 2.2 The seam has no spectral cutoff, and `effective_resolution` has no answer to give

`CASE-STUDY-GUIDE` is explicit that `effective_resolution` is *"the expert's own measured spectral cutoff, not a chosen number"*, and `wake_array.modes_for` turns a cutoff **wavelength** into a mode count. Measured here, the question does not apply.

| block | $\lvert\text{diag}\rvert$ | $\max\lvert\text{off-diag}\rvert$ | off/diag | $\kappa$ | $\sigma_{\min}/\sigma_{\max}$ |
|---|---|---|---|---|---|
| duct | $2.513$ | $3.069\times10^{-2}$ | $1.22\times10^{-2}$ | $\mathbf{1.026}$ | $0.9745$ |
| disc | $5.792$ | $5.491\times10^{-2}$ | $9.48\times10^{-3}$ | $\mathbf{1.040}$ | $0.9620$ |

> **A conjugate-heat seam's probed response is the identity to a condition number of $1.03$.** The flux at a cell is set by the temperature *at* that cell; transport along the seam is a $1\%$ correction. There is no spectral decay to cut at.

The consequence is arithmetic: for an operator with no small singular directions, **a rank-$m$ truncation discards exactly the fraction of the modes it drops.**

| $m$ | $5$ | $11$ | $16$ | $21$ | $25$ | $29$ | $31$ | $32$ |
|---|---|---|---|---|---|---|---|---|
| truncation, duct | $0.921$ | $0.815$ | $0.712$ | $0.591$ | $0.472$ | $0.309$ | $0.178$ | $\mathbf{8.2\times10^{-15}}$ |

A cutoff-derived declaration would have said $m = 11$ — `thermal_seam`'s $6.4$-cell cutoff carried onto $32$ cells — which loses **$81\%$ of the operator in norm**. `thermal_seam` itself declares $16$ modes on $48$ cells *"as everywhere else"*, and that is the number this study would have inherited. **This graph declares $M = 32$, the grid.**

**What that choice costs a published quantity is nothing, and the distinction matters.** $\beta$ moves by $0.05\%$ between $m = 11$ and $m = 32$ ($3.2138$ against $3.2124$) and $\kappa$ from $1.005$ to $1.055$ — because an operator with no small singular directions has none to lose. **The truncation costs the interface SPACE, not the conditioning.** So nothing `thermal_seam` published is wrong; what its interface space could not have represented is a seam disagreement above its sixteenth mode.

**And the full-rank basis found a bug the other case studies could not have hit.** At $k = n/2$ the cosine samples at $\cos(\pi(i+\tfrac12))$, which is zero at every cell centre, so the shared `fourier_basis` construction returns a **zero column** at Nyquist and every projection built on it silently drops a direction. `window_ns`, `thermal_seam` and `wake_array` all declare $m$ far below $n$ and never reach it. Fixed here (take the sine, which is the real Nyquist mode) and pinned by a test.

### 2.3 The interface condition, and CS-10's lesson one seam along

CS-10 found that a lumped agent's constitutive law read as an **update** diverges at macro-step 1, and that what is well posed is the port's own condition solved in the port's own variable — three Newton steps against an added-mass instability that no under-relaxation can fix. That finding is now [[general-coupling-scheme]] §4.2, lifted out so Phase C does not meet it four more times.

**At this seam the same reading is cheaper still.** On the states at hand both responses are *affine* in the trace,

$$\underbrace{\frac{k_{\text{eff}}}{\mathrm dy/2}\bigl(T_{\text{gas},0} - \lambda\bigr)}_{\text{the duct's wall flux}} \;=\; \underbrace{h_{\text{in}}\bigl(\lambda - T_{\text{face}}\bigr)}_{\text{the disc's Robin response}} \qquad\Longrightarrow\qquad \lambda^\star \;=\; \frac{(k_{\text{eff}}/\tfrac{\mathrm dy}{2})\,T_{\text{gas},0} \;+\; h_{\text{in}}\,T_{\text{face}}}{(k_{\text{eff}}/\tfrac{\mathrm dy}{2}) \;+\; h_{\text{in}}}$$

so the interface condition has a **closed form**, needs no iteration at all, and its residual is zero to floating point rather than to a tolerance. That is the same rule one step further: the question is never *what does my subsystem's law say the answer is*, it is *where do the two responses balance*, and how expensive that is to answer is a property of the responses rather than of the coupling.

**And the root does not depend on the flux convention**, which is worth stating because W66 is the live question at a `THERM` seam: the `entropy` convention divides both halves by the same face temperature, so it scales the residual and moves no root. W66 is about what the bond *is* and about what `interface_power` measures; it is not about where the interface sits.

### 2.4 The referent, and the control that says the instrument works

`BrakeRollout` is one code path and the referent is **one argument changed**: `exchange = DT_DUCT` instead of `DT_DISC`. Both agents step, the interface condition is re-solved every exchange, and at the duct's own clock the trace is stale for one duct step instead of five hundred. So the composed column and the referent differ by the **exchange cadence** and by nothing else.

Two controls, and both are exact rather than small:

- **the referent's own convergence.** Refining it $4\times$ (exchange $10^{-4} \to 2.5\times10^{-5}\,\mathrm s$) moves the disc face temperature by $1.392\times10^{-4}\,\mathrm K$. That is the floor every $\sigma$ below is read against, it is not a bitwise zero, and nothing is quoted against it (**W106**'s discipline).
- **the single-rate lag defect is exactly zero.** At an exchange interval equal to the fast agent's own step there is nothing to be stale about, and the measured defect is $0$ and not $10^{-16}$. Anything else would mean the instrument was measuring itself.

---

## 3. The bound — $\sigma$ as a function of the exchange interval  (W90)

The measurement isolates the staleness and nothing else. Both columns march the **same** agents from the **same** state over the **same** interval, and the disc takes one step of $\Delta t_{\text{ex}}$ in both; the only difference is the trace the duct is handed — the referent's own path $\lambda^\star(t)$, or its value at the start of the interval held constant. **A comparison against a finer-stepped column instead would have confounded the lag with both agents' time-discretization error**, which is the shape of mistake the vault has been caught by before.

The bound is [[master-error-bound]] §4.2, and its two constants come from a **probe** — no run:

$$\sigma(\Delta t_{\text{ex}}) \;\le\; s_\Gamma\,\lVert\delta\lambda\rVert \;+\; C_2\,\lVert\delta\lambda\rVert^2, \qquad \lVert\delta\lambda\rVert \;\le\; \dot\lambda_{\max}\,\Delta t_{\text{ex}}$$

with $s_\Gamma = 9.4966\times10^{-3}\,\mathrm K^{-1}$, $C_2 = 1.4288\times10^{-5}\,\mathrm K^{-2}$ and the run's own $\dot\lambda = 11.94\,\mathrm{K/s}$.

| $\Delta t_{\text{ex}}$ [s] | ratio | lag [K] | measured $\sigma$ | bound | bound/measured | order |
|---|---|---|---|---|---|---|
| $1\times10^{-4}$ | $1$ | $0$ | $\mathbf{0}$ | $0$ | — **control** | — |
| $5\times10^{-4}$ | $5$ | $5.304\times10^{-3}$ | $3.533\times10^{-5}$ | $5.037\times10^{-5}$ | $1.426$ | — |
| $2\times10^{-3}$ | $20$ | $2.485\times10^{-2}$ | $1.435\times10^{-4}$ | $2.360\times10^{-4}$ | $1.645$ | $1.011$ |
| $1\times10^{-2}$ | $100$ | $1.252\times10^{-1}$ | $6.186\times10^{-4}$ | $1.189\times10^{-3}$ | $1.922$ | $0.908$ |
| $5\times10^{-2}$ | $500$ | $7.524\times10^{-1}$ | $3.285\times10^{-3}$ | $7.154\times10^{-3}$ | $2.178$ | $1.037$ |
| $2\times10^{-1}$ | $2000$ | $4.211$ | $1.853\times10^{-2}$ | $4.024\times10^{-2}$ | $2.172$ | $1.248$ |

> **The bound holds at every graded interval and is loose by between $1.43\times$ and $2.18\times$ over $400\times$ of exchange interval and $2000\times$ of clock ratio. Nothing in that table is fitted to it.**

Three things worth reading off it.

**The defect is first order in the interval.** The order column sits at $1.0$ until the last row, where the quadratic term starts to matter and the bound's own second term is what keeps it above. That matches CS-9's volumetric splitting error ($\log_2$ ratios $1.003$) and CS-10's field-to-lumped lag ($1.258$, $1.165$) — **three case studies, three different coupling kinds, one order.**

**Ratio $1$ is the control and its defect is exactly zero, not small.** One duct step *is* the interval, so there is nothing to be stale about. It carries no tightness ratio, and quoting $0/0$ as one would be **W106**'s mistake in the same run that takes W106's discipline seriously elsewhere.

**And the looseness has a mechanism that is already on the record.** $s_\Gamma$ is measured by displacing the trace **uniformly**, and a run's lag is not uniform — which is exactly **W86**'s finding, that $\sigma$'s argument is the lag *profile* and a scalar summary does not determine it. Here that costs a factor of $1.4$ to $2.2$, which is what a bound is allowed to cost. It is worth saying what it cost before it was noticed: measuring $s_\Gamma$ through the **disc's** flux while measuring $\sigma$ through the **duct's** made the same bound over-predict by $5.7\times$ to $8.6\times$ and look exactly like a lag-profile effect. **Interface power can be read off either side and the two agree only at a converged trace; a bound assembled from two sides is comparing different quantities.**

---

## 4. The clock ratio is not the variable, and that is what makes the bound usable

The interval sweep moves two things at once — the interval and the ratio — so on its own it cannot say which one $\sigma$ is a function of. Pinning the interval at the native $5\times10^{-2}\,\mathrm s$ and **refining the fast agent's own march** moves the ratio $8\times$ with the staleness untouched:

| Courant number | $\Delta t_{\text{stable}}$ | ratio | $\sigma$ | relative |
|---|---|---|---|---|
| $0.40$ | $2.048\times10^{-5}$ | $2442$ | $3.285085\times10^{-3}$ | $1.000000$ |
| $0.20$ | $1.024\times10^{-5}$ | $4883$ | $3.285386\times10^{-3}$ | $1.000092$ |
| $0.10$ | $5.120\times10^{-6}$ | $9767$ | $3.285536\times10^{-3}$ | $1.000137$ |
| $0.05$ | $2.560\times10^{-6}$ | $19533$ | $3.285611\times10^{-3}$ | $1.000160$ |

> **Eight times of clock ratio at a fixed interval moves $\sigma$ by $1.00016\times$.**

**This is what makes the bound worth having rather than merely true.** A real conduction-against-convection seam runs at $10^4$–$10^5$ and no single-rate referent can be built there — that is the whole reason W90 had no number for three tiers. A bound written in the **ratio** would be uncheckable at the ratio it is needed at. A bound written in the **interval** is checked here at ratio $5$ and applies at ratio $10^5$, because the ratio is not in it.

**Scoped, and the scope is the interesting part.** This holds for a fast agent that **sub-steps internally at its own stability limit** — what a classical explicit solver does, and what makes its declared `dt_native` a bookkeeping choice rather than a physical constraint. A **frozen learned expert cannot refine its own march**: `dt_native` is fixed by its weights, so at a fixed interval a coarser ratio means genuinely fewer and genuinely larger steps, and the ratio re-enters through the fast side's own discretization error. That case is **unmeasured here**, it is precisely R4's asymmetry, and it is the one the substitution campaign will meet.

**And it says where R9's term lives.** The ratio moves the fast side's flux transient — R9's own subject — and leaves the staleness alone; the interval moves the staleness and drags the ratio with it only because R4 ties them together. That is the cleanest statement of why the two terms were confused for the life of the compiler.

---

## 5. W17's $W>1$, which R3 admits here

W90 names waveform relaxation as the candidate remedy, and it is **the one axis R4 leaves open**: R4 forbids shrinking the interval, and a waveform does not shrink it — it stops the trace inside it being constant. **R3** requires `bc_time_varying` on every agent at the interface, and both of these declare it, so this is admissible by rule rather than by hope.

The waveform is built from the **previous** interval's rise, $\lambda(t) = \lambda_n + (\lambda_n - \lambda_{n-1})\,s$. That matters: an extrapolation that reads its own interval's endpoint is not a scheme, it is the answer, and measuring one would be measuring nothing.

| $\Delta t_{\text{ex}}$ [s] | held | waveform | reduction |
|---|---|---|---|
| $2\times10^{-3}$ | $1.393\times10^{-4}$ | $1.042\times10^{-5}$ | $13.4\times$ |
| $1\times10^{-2}$ | $6.320\times10^{-4}$ | $1.117\times10^{-5}$ | $\mathbf{56.6\times}$ |
| $5\times10^{-2}$ | $5.036\times10^{-3}$ | $1.399\times10^{-3}$ | $3.6\times$ |
| $2\times10^{-1}$ | $2.531\times10^{-2}$ | $3.488\times10^{-3}$ | $7.3\times$ |

**The reduction is not monotone in the interval, and it should not be.** The held scheme's error is the trace's *first* difference and the waveform's is its *second*, so the ratio between them is the trace's **curvature**, which is a property of where on the transient the interval sits rather than of the interval's length. That is W86's shape once more — the transferable quantity is a slope and the thing that moves is the profile — and the practical reading is that **a waveform buys an order, not a factor**. Over a $200$-interval rollout (§6) it is worth $1.84\times$ on the accumulated defect.

---

## 6. The accumulated defect, marched past the crossing

Tier 23's standing rule, adopted after its third instance: *a comparison between two configurations is not a result until it has been marched past the point where the two curves could cross, and the crossing has to be looked for rather than assumed absent.* CS-10 was the first study to apply it as a gate; this is the second.

The single-rate referent over $10\,\mathrm s$ — $100\,000$ exchanges — against four multirate columns at $200$ macro-steps:

| column | $\Delta T$ at the end [K] | $\max\lvert\Delta T\rvert$ | at step | sign changes | relative at end |
|---|---|---|---|---|---|
| referent (single-rate) | — | — | — | — | $301.64 \to 516.40\,\mathrm K$ |
| multirate, held | $-1.715$ | $1.715$ | $200$ | $\mathbf{1}$ | $3.32\times10^{-3}$ |
| multirate, lag 2 | $-2.512$ | $2.512$ | $200$ | $\mathbf{1}$ | $4.87\times10^{-3}$ |
| multirate, lag 4 | $-4.112$ | $4.112$ | $200$ | $\mathbf{1}$ | $7.96\times10^{-3}$ |
| multirate, waveform | $\mathbf{-0.934}$ | $0.934$ | $200$ | $\mathbf{1}$ | $1.81\times10^{-3}$ |

**The crossing is there and it was looked for.** Every column starts on one side of the referent and finishes on the other. The referent itself rises $215\,\mathrm K$ over the window, so the defect is read against that rise and not against zero.

**Two things this leaves open, and both are stated rather than smoothed.**

- **The defect has not peaked at the horizon marched.** Every column's maximum is at step $200$, the last one, so $1.715\,\mathrm K$ is a *value at $10\,\mathrm s$* and not a bound. CS-10 left the same caveat on its cut defect, for the same reason, and the honest form is the one both take.
- **Accumulation is sub-linear in the lag, not first order.** $1.715 \to 2.512 \to 4.112$ over lags $1 \to 2 \to 4$ gives $\log_2$ ratios of $0.55$ and $0.71$, against §3's per-interval order of $1.0$. The per-interval defect is first order and what a rollout accumulates is not — **which is W123 exactly, on a third quantity**: the one-interval defect is not what a rollout accumulates, and every constant in this vault is measured on the former.

---

## 7. W128: the design sensitivity, and the horizon it needs

CS-10 opened W128 because a design sensitivity taken from a rollout is a function of the horizon and has no field on the record — measured there at $+0.411$ on a $40$-step rollout and $-0.152$ at $160$, a sign change with every value finite-difference-confirmed. **Here it is measured on different physics, with a different knob, against a closed-form prediction written before the run.**

$\theta = U_{\text{duct}}$, the cooling duct's inlet velocity — what a brake duct is actually sized for. It enters **only** the fast agent. $J$ is the settled seam temperature over the last quarter of an $N$-step march, CS-10's convention. The derivative is a central finite difference, which is **R7's own branch** for an agent declaring `NONE` with `deterministic=True` and a one-ulp floor, and CS-10 is what makes that instrument trustworthy: it agreed with an exact reverse-mode adjoint on a composed stack to $1.3\times10^{-6}$ over five decades of step.

| $N$ | $10$ | $25$ | $50$ | $100$ | $150$ | $200$ | $300$ | $400$ | $600$ | $800$ | $1000$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $t$ [s] | $0.5$ | $1.25$ | $2.5$ | $5$ | $7.5$ | $10$ | $15$ | $20$ | $30$ | $40$ | $50$ |
| $J$ [K] | $314.5$ | $333.7$ | $364.1$ | $416.4$ | $459.9$ | $496.6$ | $552.2$ | $590.7$ | $635.7$ | $657.4$ | $667.8$ |
| $\mathrm dJ/\mathrm dU$ | $+0.0616$ | $+0.1125$ | $+0.1682$ | $\mathbf{+0.1863}$ | $+0.1137$ | $-0.0215$ | $-0.3902$ | $-0.8009$ | $-1.538$ | $-2.066$ | $\mathbf{-2.402}$ |

> **The sensitivity changes sign between $N = 150$ and $N = 200$, and it is still growing in magnitude at $N = 1000$.**

**The finite difference is on a truncation branch.** Three steps spanning $4\times$ ($\delta U = 0.5$, $1.0$, $2.0\,\mathrm{m/s}$) agree to $1.77\times10^{-3}$ relative at worst, so the sweep is not in cancellation. The objective is bit-reproducible ($0.0$), asserted twice and quoted against nothing.

**Two controls the convention needs.** Releasing $U = 42$ from its *own* settled duct rather than the shared one moves $J(20)$ by $5.92\times10^{-3}\,\mathrm K$ — the duct re-settles inside one macro-step, so the choice of release field is immaterial *and is measured rather than argued*. And $\theta$ enters no initial condition, so this is a derivative of the composed stack.

### 7.1 The mechanism, in closed form, written before the run

A two-Robin lumped disc, $C\dot T = h_d A(T_g - T) + h_p A(T_p - T)$, gives

$$\frac{\mathrm dT}{\mathrm d\theta}(x) \;=\; \underbrace{T_{\text{eq}}'\bigl(1 - e^{-x/\tau}\bigr)}_{\text{equilibrium shift}} \;-\; \underbrace{(T_0 - T_{\text{eq}})\,e^{-x/\tau}\,\frac{x/\tau}{H}}_{\text{rate shift}}, \qquad H = h_d + h_p$$

whose two limits are

$$\lim_{x\to0}\frac{1}{x}\frac{\mathrm dT}{\mathrm d\theta} \;\propto\; \frac{T_g - T_0}{H}, \qquad\qquad \lim_{x\to\infty}\frac{\mathrm dT}{\mathrm d\theta} \;=\; T_{\text{eq}}' \;=\; \frac{h_p(T_g - T_p)}{H^2}\,\frac{\mathrm dh_d}{\mathrm d\theta}$$

> **The two have opposite signs whenever the disc is released colder than the air cooling it** — because a cold disc is being *warmed* by the duct, so more flow warms it faster, until the friction heat dominates and more flow cools it. That is a brake at the start of a lap in a duct that has already been warmed, and it is this study's declared release state.

Measured against it: $h_{\text{eff}} = 608.9$, $H = 858.9\,\mathrm{W/(m^2K)}$, $T_g - T_0 = +120\,\mathrm K$ (so the short-horizon sign is **positive**, and it is), and $\mathrm dT_{\text{eq}}/\mathrm dU = -2.178\,\mathrm{K/(m/s)}$ against a measured long-horizon $-2.402$ — **within $10\%$, from a lumped model against a two-dimensional conjugate simulation.**

**So the sign change is not an accident of one objective.** CS-10 argued its two regimes from a quasi-static limit after the fact; here the mechanism is a two-term closed form, its sign condition is a declared property of the release state, and the run confirms both limits.

### 7.2 The validity limits — W128's own $T_{\text{pred}}$

[[end-to-end-architecture-spec]] §0.4 makes these binding. Two numbers, and they are not the same number:

| quantity | value | meaning |
|---|---|---|
| $N_{\text{sign}}$ | $\mathbf{200}$ ($t = 10\,\mathrm s \approx 1\,\tau$) | above this the **sign** is settled. A design search run below it moves the knob the wrong way |
| $N_{\text{valid}}(50\%)$ | $600$ ($30\,\mathrm s$) | above this the magnitude is within $50\%$ of its value at $N = 1000$ |
| $N_{\text{valid}}(20\%)$ | $800$ ($40\,\mathrm s$) | |
| $N_{\text{valid}}(10\%)$, $N_{\text{valid}}(5\%)$ | **not determined** | not reached over the horizons marched |

> **The sign is settled after one thermal time constant. The magnitude is not settled after five.**

**A tolerance met only at the final horizon is not met**, and reporting $N = 1000$ for the $10\%$ row would be reporting the sweep's own endpoint as a result — W106's shape, one level up. Those rows return *not determined*, which is the honest reading and is what the record carries.

**[AI Inference]:** this is why $N_{\text{sign}}$ is the field an optimiser should be handed rather than $N_{\text{valid}}$ — a gradient search tolerates a $2\times$ error in magnitude and does not tolerate a sign — and it is $4\times$ cheaper to reach here. Argued from the structure of a descent step, not measured; no optimiser has been run in this loop.

---

## 8. The compile, and what `L7/R9/lag` says now

| graph | verdict | refusals | what decertifies |
|---|---|---|---|
| `as-built`, native clocks | **`refuse`** at the time of writing; **`admit-uncertified` since 2026-09-04** | `L2/R10` | the disc's `EMBEDDED` backward-Euler solve — and the refusal turned out to be **false**, see the box below |
| `split-step`, native clocks | `admit-uncertified` | **0** | `L2/C3/W57`, `L5/eps_tol`, **`L7/R9/lag`**, `L7/R9/order`, `L6/E6`, `L9/E5`, `L8/W56` |
| `split-step`, matched clocks | `admit-uncertified` | 0 | the same, less the two R9 rows |
| `split-step`, pointwise matching | **`refuse`** | `L7/R9` | R9's own refusal, unchanged and untouched by any of this |

**The `as-built` refusal was NOT correct, and CS-12 found out one day later.** This page said it was, on the reasoning that the ladder's CS-11 row asks for a slow `EMBEDDED` conduction agent, backward Euler over a cross-section *is* one, and declaring otherwise to dodge R10 would be the rule working and the declaration lying. All of that still holds — **and R10 should not have fired.**

> **W114, closed 2026-09-04 at CS-12** ([[case-study-wing-fsi-atlas-0.1]] §2). R10's own sentence names a premise — *"the graph decomposes the domain"* — that the rule never checked; it read `elliptic_subsolve` alone. This graph's two agents are a conduction **disc** (`thermoelastic-shell-2d`) and a convecting **duct** (`convection-diffusion-2d`), each the only agent of its family: the disc's cross-section is not a piece of a tiled conduction domain, so nothing was cut and the elliptic error R10 exists to refuse cannot arise. R10 now asks whether another agent shares the family before refusing, and `as-built` compiles at `admit-uncertified`.
>
> **Nothing this page measured moves.** Every number in it comes from the marches, not from a compile, and the $\sigma$ law, the ratio control, the waveform and the horizon are all measured on `split-step` for reasons that have nothing to do with R10. What changes is one cell of §8's table and one sentence of this one.
>
> **And `split-step` is not made pointless by it.** Exposing the elliptic part is still exactly what R10 *prescribes* for an agent that **is** cut, which is every fluid tiling in this vault; what CS-11 did not need was to build it *in order to compile*.

**What changes is what `L7/R9/lag` decertifies *with*.** It has said, since it was written, that R9 is the smaller term and quoted `thermal_seam`'s $62.4\times$. It can now name the larger one: a bound, first order in the interval, with two constants from a probe and one rate from the run. **That is the difference between a decertification that says *this is uncertified* and one that says *this is uncertified by this much*,** and it is what §4's gate asked for.

---

## 9. Three things this case study found without looking for them

### 9.1 A conjugate-heat seam has no spectral cutoff, so `effective_resolution` has no answer

§2.2. The probed seam operator is the identity to $\kappa = 1.03$, every mode carries the same information, and a rank-$m$ truncation discards exactly the fraction of modes it drops — $81\%$ at the $m = 11$ a cutoff-derived declaration would have given. `thermal_seam` declares $16$ modes on $48$ cells *"as everywhere else"* and would have been this study's default. **Nothing it published is wrong** ($\beta$ moves $0.05\%$ between $m=11$ and $m=32$), and what its interface space could not have represented is a seam disagreement above its sixteenth mode.

### 9.2 The shared Fourier basis returns a zero column at Nyquist

§2.2. At $k = n/2$ the cosine samples at $\cos(\pi(i+\tfrac12))$, zero at every cell centre, so `column_stack` hands back a rank-deficient "basis" and every projection built on it silently drops a direction. `window_ns`, `thermal_seam` and `wake_array` all declare $m \ll n$ and could not have reached it. Found by being the first graph to declare $m = n$.

### 9.3 A single-stage re-run truncated a twelve-minute artifact, and the protection was in the same file

The driver writes after every stage — the discipline that makes a long sweep safe to start — and then began each invocation from an empty dictionary, so running `--stages constants` alone destroyed everything the other nine had written. **The protection and the destruction were in the same file.** Fixed by loading the artifact before writing it, and it is worth recording because the failure is not in either half on its own: incremental persistence is only a protection if the next invocation reads what it protected.

---

## 10. What this case study cannot say

- **One side of the seam is not a build-repo donor.** The duct is written here and graded against two closed-form oracles and its own refinement; that makes it a solver rather than a fixture, and it does not make it a validated third-party expert the way `ThermoStruct2D` is.
- **The ratio-independence is measured for an agent that can refine its own march.** A frozen checkpoint cannot, and that is the case R4's asymmetry is about and the one the substitution campaign will meet. §4 states the scope; nothing here tests it.
- **One seam, one pair of experts, one operating point.** $s_\Gamma$ and $C_2$ are seam properties in the sense W86 established for `thermal_seam`'s slope — constant along a trajectory — and nothing here establishes that either travels to another seam.
- **The accumulated defect has not peaked** at $10\,\mathrm s$ (§6), so those are values and not bounds.
- **The gradient is measured on one objective and one knob**, and the magnitude has not converged at $50\,\mathrm s$. The sign has, and the two-regime mechanism is checked against a closed form, but $N_{\text{valid}}(10\%)$ is genuinely unknown.
- **No learned expert ran.** [[case-study-ladder-to-f1]] §2's classical-first split says the substitution campaign prices R10 per seam and this seam has not been priced.
- **Nothing reaches `admit`**: $L$ is unmeasured (**W1**), $C_\mu$ and the graph's own $\sigma$ were unmeasured before this run, and a non-empty `unmeasured` list forces `admit-uncertified` since **W56**.

---

## 11. Rows this closes and opens

| row | outcome |
|---|---|
| **W90** | **closed, by the first branch of its own gate.** $\sigma$ is a function of the exchange interval, first order in it, bounded by two probe-measured seam constants and one run-derived rate, holding at $1.43\times$–$2.18\times$ over $2000\times$ of clock ratio — and the clock **ratio** turns out not to be an independent variable at all, which is what lets a bound checked at ratio $5$ apply at the $10^4$–$10^5$ a real conjugate seam runs at. The candidate remedy the row names, W17's $W>1$, is admissible under R3 here and is worth $3.6\times$–$56.6\times$ per interval and $1.84\times$ over a rollout |
| **W128** | **closed as a definition and measured a second time.** The row asks that *"a declared gradient carry the horizon it was taken at, the way `sigma_lag` carries the lag — one field and a definition."* [[end-to-end-architecture-spec]] §0.4 is that definition, made binding across all three temporal parameters, and this graph supplies the second independent measurement: a sign change between $N=150$ and $N=200$ on different physics, with $N_{\text{sign}} = 200$, $N_{\text{valid}}(20\%) = 800$, and $N_{\text{valid}}(10\%)$ **not determined over $1000$ macro-steps** |
| **W17** | **exercised for the first time.** A trace carried as a waveform, built causally from the previous interval, on a graph where R3 admits it. Not closed: the construction here is a linear extrapolation inside one interval and not a converged waveform-relaxation iteration, which is what the row is ultimately about |
| **W123** | **recurs on a third quantity, and the recurrence is measured.** The per-interval defect is first order in the lag; what a $200$-interval rollout accumulates is order $0.55$–$0.71$. Tier 23 found it for cut criteria and Tier 24 for gradients; §6 finds it for the multirate defect itself |
| **W86** | **generalized and priced.** The slope transfers and the profile does not; here that costs the bound a factor of $1.4$–$2.2$, which is what a bound is allowed to cost. And a second reading: a slope measured through one side's flux does not bound a defect measured through the other's, worth $5.7\times$–$8.6\times$ before it was noticed |
| **W116** | **partly answered.** The row is that a splitting defect has no slot on `MeasuredConstants` because `sigma` is a *transmission* infidelity. The multirate lag is a transmission defect and `sigma` **is** the right field for it — what it needed was the interval beside the lag, which §0.4 now requires. CS-9's co-located splitting error still has no slot |
| **W106** | **applied twice as a discipline and once as a correction.** The single-rate control is exactly zero and carries no tightness ratio; the referent's own refinement floor is $1.39\times10^{-4}\,\mathrm K$ and nothing is quoted against it; and the first version of this study's duct-convergence *test* compared roundoff to roundoff, which is the trap in the tester rather than the tested |
| **W97** | **its mechanism lifted out.** [[general-coupling-scheme]] §4.2 now carries CS-10's interface-solve rule as the default at a field↔lumped `MECH` seam, and §2.3 here is the field↔field instance where the same reading has a closed form |
| **W133** | **opened.** `effective_resolution` is specified as a measured spectral cutoff and a conjugate-heat seam does not have one — §9.1 |
| **W134** | **opened.** The shared `fourier_basis` returns a zero column at the Nyquist mode, so a full-rank interface space is silently rank-deficient — §9.2 |
| **W135** | **opened.** `sigma` carries `sigma_lag` and no field saying which side's response the interface power was evaluated through, and the two differ by $5.7\times$–$8.6\times$ — §3 |
| **W132** | **filed, from CS-10 rather than from here**, as a Tier-1 silent-correctness row: two different physical quantities under one declared `response_half`, invisible to `L3/C9` |

---

## See Also

- [[case-study-ladder-to-f1]] — §4's CS-11 row, which scheduled this and wrote its gate; §6's critical path, where W90 blocked *"every pair of subsystems on different clocks"*
- [[master-error-bound]] — §4.2, the multirate branch of $\sigma$, which this study is the measurement for
- [[general-coupling-scheme]] — R4 and R9, and §4.2's interface-solve default lifted out of CS-10
- [[end-to-end-architecture-spec]] — §0.4, the binding convention on the three temporal parameters; §9's L7 rules
- [[temporal-error-accumulation]] — §3's $W>1$ window, which R3 admits here and §5 measures
- [[case-study-ground-effect-atlas-0.1]] — CS-10, which opened W128 and supplied the interface-solve rule and the finite-difference calibration
- [[case-study-thermal-strain-atlas-0.1]] — CS-9, whose lag construction §3 reuses and whose splitting error is the same order
- [[case-study-seam-placement-atlas-0.1]] — CS-9★, whose standing rule about marching past a crossing §6 applies as a gate
- [[gap-worklist]] — Tier 25: W90 and W128 closed, W133–W135 opened, W132 filed to Tier 1
- [[tier0-measurements]] — the measurement record, and W30's static drift band §2 is read against
