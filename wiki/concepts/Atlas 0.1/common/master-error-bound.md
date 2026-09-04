# The Master Error Bound — the complete formulation, derived once

**Type:** Concept page — framework theory, **version-independent** (folder: `Atlas 0.1/common/`)
**Status:** written 2026-08-26. Assembles [[composition-error-theory]] (what agreement buys), [[probed-dtn-coupling]] (how to build the interface operator) and [[temporal-error-accumulation]] (how defects accumulate) into **one derivation**, and adds the term all three were missing.
**Related:** [[composition-error-theory]] · [[probed-dtn-coupling]] · [[temporal-error-accumulation]] · [[general-coupling-scheme]] · [[generalization-requirements]] · [[schwarz-iteration-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[conservation-as-constraint-atlas-0.1]] · [[open-problems-atlas-0.1]]

> **The one-line version.** The per-step defect is **three** terms, not two. Splitting off the middle one — $\sigma$, the error from posing the interface problem with the *wrong operator* — is what makes the bound complete, because it is the only term that is invisible to every diagnostic the framework currently reports and the only one that iteration cannot reduce. With it the master bound reads $\lVert e^N\rVert \le L^N\lVert e^0\rVert + \sum_n L^{N-n}(\tau^n+\sigma^n+\gamma^n)$, and each term maps to exactly one construction, one measurement, and one failure already in the vault's record.

---

# 1. Objects

| symbol | name | meaning |
|---|---|---|
| $\mathcal S_{\Delta t}$ | exact evolution | the true solution operator on $\Omega$ over one macro-step |
| $\Omega_i,\ R_i,\ \chi_i$ | subdomains, restriction, partition of unity | $\sum_i R_i^{\!\top}\chi_i R_i = I$ |
| $\Gamma$ | interface | $\bigcup_{i\neq j}\partial\Omega_i\cap\overline{\Omega_j}$ |
| $\mathcal S_i(v;\mu)$ | **exact** local evolution | on $\Omega_i$, from state $v$, with interface datum $\mu$ |
| $\mathcal E_i(v;\mu)$ | **agent** | the expert's approximation to $\mathcal S_i$ |
| $\Lambda$ | **exact** interface operator | Steklov–Poincaré / DtN; global and nonlocal |
| $\tilde\Lambda$ | **transmission condition** | the rung actually used — Dirichlet ring, Robin, probed Schur… |
| $\mathcal G(\lambda;u^n)$ | interface residual | $\mathcal G=0$ is the coupled-consistency condition |
| $\mathcal A$ | assembly | combines local outputs into a global state (PoU, single-valued fluxes) |
| $\Phi$ | composed macro-step | $u^n\mapsto u^{n+1}$, the whole coupled solve |

**Three interface traces, and keeping them apart is the entire content of §3.**

$$\lambda^\star \;=\; \operatorname{tr}_\Gamma u^\star \quad\text{(what the \emph{true} solution puts on } \Gamma)$$
$$\lambda^\dagger \;:\; \mathcal G(\lambda^\dagger)=0 \quad\text{(the \emph{exact root} of the interface problem you actually posed)}$$
$$\lambda^{(k)} \quad\text{(what the solver \emph{returned} after } k \text{ iterations)}$$

---

# 2. The exact recursion

With $e^n = u^n - u^{n,\star}$, add and subtract $\Phi(u^{n,\star})$:

$$e^{n+1} \;=\; \underbrace{\bigl[\Phi(u^n)-\Phi(u^{n,\star})\bigr]}_{\text{propagation}} \;+\; \underbrace{\bigl[\Phi(u^{n,\star})-\mathcal S_{\Delta t}(u^{n,\star})\bigr]}_{\text{one-step defect } d^{n+1}}$$

This is an identity — no approximation yet. With $L=\operatorname{Lip}(\Phi)$ on the relevant set,

$$\lVert e^{n+1}\rVert \;\le\; L\lVert e^n\rVert + \lVert d^{n+1}\rVert \qquad\Longrightarrow\qquad \lVert e^N\rVert \;\le\; L^N\lVert e^0\rVert + \sum_{n=1}^{N}L^{\,N-n}\lVert d^{n}\rVert$$

**Everything hard is in $d$ and in $L$.** §3–§5 open $d$; §6 opens $L$.

---

# 3. The defect splits three ways, not two

Write $\Phi_\mu$ for the composed step *forced to use interface datum* $\mu$. Then $\Phi = \Phi_{\lambda^{(k)}}$, and the defect telescopes through the two intermediate traces:

$$\boxed{\;d^{n+1} \;=\; \underbrace{\bigl[\Phi_{\lambda^\star}(u^{n,\star})-\mathcal S_{\Delta t}(u^{n,\star})\bigr]}_{\textstyle \tau^{n+1}} \;+\; \underbrace{\bigl[\Phi_{\lambda^\dagger}-\Phi_{\lambda^\star}\bigr](u^{n,\star})}_{\textstyle \sigma^{n+1}} \;+\; \underbrace{\bigl[\Phi_{\lambda^{(k)}}-\Phi_{\lambda^\dagger}\bigr](u^{n,\star})}_{\textstyle \gamma^{n+1}}\;}$$

Three genuinely different failures:

| term | name | what is wrong | reduced by |
|---|---|---|---|
| $\tau$ | **agent infidelity** | the agents are wrong *even when handed the true interface data*. $\mathcal E_i\neq\mathcal S_i$ on the restriction of the true solution | a better expert; a faithful (non-periodic) local solve |
| $\sigma$ | **transmission infidelity** | the interface problem you posed has the **wrong solution**, because $\tilde\Lambda\neq\Lambda$ | climbing the DtN ladder — **and nothing else** |
| $\gamma$ | **solve incompleteness** | you did not converge to your own problem's root | iterating; or one direct interface solve |

## 3.1 Why splitting $\sigma$ off is the point of this page

[[composition-error-theory]] proved that agreement is a consistency condition rather than a correctness one, and [[probed-dtn-coupling]] argued that accuracy is set by $\lVert\Lambda-\tilde\Lambda\rVert$. **Both statements are about $\sigma$, and neither page carries it as a term in the bound** — it was folded into $\gamma$ or left implicit. Three consequences follow immediately from separating it:

1. **Schwarz iteration reduces $\gamma$ and nothing else.** It drives $\lambda^{(k)}\to\lambda^\dagger$, which is the root of the problem *you posed*. If $\tilde\Lambda$ is wrong, $\lambda^\dagger\neq\lambda^\star$ and iterating converges *harder onto the wrong trace*. This is [[schwarz-iteration-atlas-0.1]] §3.1's wrapped river, now as one line of algebra.
2. **$\sigma$ is invisible to every diagnostic currently reported.** The interface residual measures $\lVert\mathcal G(\lambda^{(k)})\rVert$, i.e. $\gamma$. The conservation residual measures a few functionals. **Nothing in the framework reports $\sigma$**, and §4 shows it is bounded by two quantities that only a probe produces.
3. **The vault's sharpest failure is a pure-$\sigma$ failure.** Under periodic windows $\Lambda\equiv0$, so $\tilde\Lambda$ is maximally wrong; $\gamma\equiv0$ *bitwise* because the sweep is a constant map; and the answer was wrong by $2\times$. **$\gamma=0$, $\sigma$ maximal.** The three-term split predicts exactly that configuration; the two-term version cannot express it.

---

# 4. Bounding $\sigma$ — where $\beta$ and $\lVert\Lambda-\tilde\Lambda\rVert$ enter

Let $C_\mu = \operatorname{Lip}_\mu(\Phi_\mu)$ — the sensitivity of the composed local solve to its interface datum. Then $\sigma \le C_\mu\lVert\lambda^\dagger-\lambda^\star\rVert$, and the trace perturbation is a standard first-order estimate. With $\lambda^\star$ satisfying the exact interface equation and $\lambda^\dagger$ the approximate one, on the constrained subspace:

$$\lVert\lambda^\dagger-\lambda^\star\rVert \;\le\; \bigl\lVert\tilde\Lambda^{-1}\bigr\rVert\,\bigl\lVert(\tilde\Lambda-\Lambda)\lambda^\star\bigr\rVert \;\le\; \frac{1}{\beta}\,\bigl\lVert\Lambda-\tilde\Lambda\bigr\rVert\,\lVert\lambda^\star\rVert, \qquad \beta \;=\; \sigma_{\min}\bigl(\tilde\Lambda\big|_{\text{constrained}}\bigr)$$

giving

$$\boxed{\;\sigma \;\le\; \frac{C_\mu}{\beta}\,\bigl\lVert\Lambda-\tilde\Lambda\bigr\rVert\,\bigl\lVert\lambda^\star\bigr\rVert\;}$$

> **This is the *substructuring* branch, and saying so is a 2026-08-28 correction — [[gap-worklist]] W49, now `done`.** Until then §4 was written as though it were *the* $\sigma$ bound; it is one of two, and §4.1 below carries the other. **The scoping was forced by a measurement.** For a **halo** scheme the rung is `dirichlet` with a lagged trace, so there is no interface operator at all: $\tilde\Lambda=0$ and $\lVert\Lambda-\tilde\Lambda\rVert=\lVert\Lambda\rVert$. With $\lVert\Lambda\rVert=0.417$, $\beta=0.349$ and $\lVert\lambda^\star\rVert=1.440$, the bound at $C_\mu=1$ reads $\sigma\le\mathbf{1.72}$ against a **measured $\sigma=3.73\times10^{-8}$** — an overestimate of $4.6\times10^{7}$, i.e. $C_\mu^{\text{implied}}=2.2\times10^{-8}$.
>
> **The reason is structural rather than a matter of degree.** This bound assumes the transmission error enters through an *interface solve*, where an operator mismatch is amplified by $1/\beta$. An overlapping scheme has no interface solve, so applied to one this form is not conservative but uninformative. **$C_\mu$ *here* stays unmeasured rather than back-fitted**: fitting it from the measured $\sigma$ would make the bound tautological, and a constant of $10^{-8}$ is a diagnosis of the factorization, not a value for it. See [[tier0-measurements]] §8.8.

---

# 4.1 Bounding $\sigma$ on the overlapping branch — where the *partition of unity* enters

Written 2026-08-28, from the measurement in §4's box. **The question §4 answers for a substructuring scheme — what carries the transmission error, and what amplifies it — has a different answer for an overlapping one, and the derivation is short enough that not having it was an oversight rather than a gap.**

An overlapping scheme poses no interface equation, so there is no $\lambda^\dagger$ in §1's sense. What plays its part is the **stale artificial-boundary datum**: each agent is handed Dirichlet data on $\partial\Omega_i\setminus\partial\Omega$ taken from the previous exchange, and

$$\delta\lambda \;:=\; \lambda^{\text{lagged}}-\lambda^\star\big|_{\partial\Omega_i\setminus\partial\Omega}$$

is the whole of the transmission error's source. Two things stand between $\delta\lambda$ and the assembled field, and both are declared:

1. **Reach.** Over one exchange interval an explicit agent's boundary datum propagates exactly $d_i=\rho_i s_i$ cells inward — `stencil_radius` $\times$ sub-steps per exchange — and *no further*. Beyond $d_i$ the local solve does not depend on the stale datum at all. This is the halo rule's own quantity, used here for what it bounds rather than only as a gate.
2. **Weight.** A cell the datum reached still contributes nothing unless the assembly gives it weight. That is $\chi_i$.

Composing the two gives the **contaminated weight**

$$\boxed{\;\Pi \;:=\; \max_j\;\sum_i \chi_{ij}\,\mathbf 1\!\left[\,b_i(j)\le d_i\,\right],\qquad \sigma\;\le\;C_\mu\,\Pi\,\lVert\delta\lambda\rVert\;}$$

with $b_i(j)$ the distance from cell $j$ to $\Omega_i$'s nearest *artificial* face. A face coinciding with the real boundary carries no stale datum and does not enter.

**The two limits are the interesting ones, and neither is available from §4.** $\Pi=1$ when the overlap is narrower than the domain of dependence — nothing in the blend is clean, and the bound degrades to $\lVert\delta\lambda\rVert$, which is right. $\Pi=0$ when the partition of unity vanishes over the whole contaminated band, and then $\sigma$ is **zero, not small**: the assembly never reads a cell the stale datum touched.

> **W49 asked for "a term for overlap width", and the measurement says that is not quite the right term.** $\Pi$ depends on the overlap only through what the overlap *lets you do with $\chi$*. Measured at fixed ramp, widening the halo from 11 to 61 cells leaves $\Pi$ at $9.54\times10^{-2}$ and $\sigma$ within $40\%$ of itself; widening the *ramp* at fixed halo 21 takes $\Pi$ from $0.75$ to $0.020$ and $\sigma$ from $1.20\times10^{-4}$ to $6.26\times10^{-9}$, a factor of $1.9\times10^{4}$. **Overlap width is a precondition; the partition of unity is the mechanism.** The one place halo width is decisive on its own is the transition at $\delta<d$, where $\Pi$ jumps to $1$ — measured $\sigma=2.68\times10^{-4}$ at halo 1 against $3.61\times10^{-8}$ at halo 11.

## 4.1.1 $C_\mu$, measured

$\Pi$ and $\lVert\delta\lambda\rVert$ were measured on eleven configurations of the four-window `WindowNS` tiling — halo $1$ to $61$, five partitions of unity, $\sigma$ spanning a factor $4.3\times10^{4}$:

| configuration | $\Pi$ | $\lVert\delta\lambda\rVert$ | measured $\sigma$ | $C_\mu^{\text{implied}}$ |
|---|---|---|---|---|
| halo 1, ramp 8 | $1.000$ | $3.343\times10^{-4}$ | $2.683\times10^{-4}$ | $0.802$ |
| halo 11, ramp 8 | $9.541\times10^{-2}$ | $1.040\times10^{-6}$ | $3.612\times10^{-8}$ | $0.364$ |
| halo 21, ramp 8 | $9.541\times10^{-2}$ | $1.099\times10^{-6}$ | $3.733\times10^{-8}$ | $0.356$ |
| halo 31, ramp 8 | $9.541\times10^{-2}$ | $1.190\times10^{-6}$ | $3.940\times10^{-8}$ | $0.347$ |
| halo 41, ramp 8 | $9.541\times10^{-2}$ | $1.363\times10^{-6}$ | $4.223\times10^{-8}$ | $0.325$ |
| halo 61, ramp 8 | $9.541\times10^{-2}$ | $1.797\times10^{-6}$ | $5.097\times10^{-8}$ | $0.297$ |
| halo 21, flat $\chi$ | $0.750$ | $1.354\times10^{-4}$ | $1.196\times10^{-4}$ | $\mathbf{1.178}$ |
| halo 21, ramp 1 | $0.750$ | $4.742\times10^{-5}$ | $2.753\times10^{-5}$ | $0.774$ |
| halo 21, ramp 4 | $0.2967$ | $3.601\times10^{-6}$ | $3.244\times10^{-7}$ | $0.304$ |
| halo 21, ramp 21 | $2.000\times10^{-2}$ | $6.565\times10^{-7}$ | $6.264\times10^{-9}$ | $0.477$ |

**A second sweep varies the physics rather than the assembly** — the Reynolds number and the macro-step, at the working partition — and is reported in full on [[tier0-measurements]] §9.4:

| configuration | $\Pi$ | measured $\sigma$ | $C_\mu^{\text{implied}}$ |
|---|---|---|---|
| $\mathrm{Re}=255$, $\Delta t=0.05$ (the baseline) | $9.541\times10^{-2}$ | $3.733\times10^{-8}$ | $0.356$ |
| $\mathrm{Re}=510$, $\Delta t=0.05$ | $9.541\times10^{-2}$ | $3.850\times10^{-8}$ | $0.337$ |
| $\mathrm{Re}=128$, $\Delta t=0.05$ | $9.541\times10^{-2}$ | $4.056\times10^{-8}$ | $0.404$ |
| $\mathrm{Re}=255$, $\Delta t=0.025$ | $9.541\times10^{-2}$ | $1.831\times10^{-8}$ | $\mathbf{0.228}$ |
| $\mathrm{Re}=255$, $\Delta t=0.10$ | $9.541\times10^{-2}$ | $7.487\times10^{-8}$ | $0.526$ |

$$\textbf{16 configurations in all:}\qquad C_\mu^{\text{halo}} \in [0.228,\,1.178],\qquad \text{spread } 5.2\times$$

**A constant that moves by $5\times$ while the quantity it relates moves by $4\times10^{4}$ is behaving like a constant.** That is the evidence for the factorization, and it is exactly what §4's form does *not* do on this branch, where the same exercise returns $2.2\times10^{-8}$. So $C_\mu=1.2$ — the sampled maximum, rounded up — is quoted as **measured with a stated scope**, and with it the bound holds on all sixteen configurations. Against §4's $4.6\times10^{7}$ on the same scheme that is **seven orders of tightening**, and it is the first $C_\mu$ this vault has been able to write down.

**Scope, stated because it is still narrow.** One expert, one governing family, one time discretization. $\mathrm{Re}$ moves by $4\times$ and $\Delta t$ by $4\times$ across the second sweep, which is the part of the "one configuration" caveat this removes; the rest of it stands. $C_\mu$ is a Lipschitz constant of the composed local solve in its interface datum, and nothing here establishes that $1.2$ travels to a different agent or to a multirate exchange. What *does* travel is the factorization: $\Pi$ is derived, not fitted, and its two limits are forced.

> **One thing the $\Delta t$ sweep required first, and it is worth knowing before quoting any $\sigma$ from a split-step run.** The composition layer applies the exposed elliptic part once per exchange, so **the exchange cadence must equal the agent's own sub-step cadence** or the composed step is a different splitting from the reference it is compared against. Hard-coded at ten exchanges, the $\Delta t=0.025$ row measured $C_\mu^{\text{implied}}=0.134$ and a $\tau$ two orders too large — with $\sigma$ and every other diagnostic unchanged. That is now **R10b**; see [[tier0-measurements]] §9.3.

**Where the two branches meet.** A scheme that both overlaps *and* solves an interface problem carries both terms and the bound takes their sum; a scheme with $\Pi=0$ and $\tilde\Lambda=\Lambda$ has $\sigma=0$ from both, which is the "exact interface operator" row of §8. Nothing in §3's telescoping changes — $\sigma$ is still $[\Phi_{\lambda^\dagger}-\Phi_{\lambda^\star}](u^{n,\star})$; only the estimate of $\lambda^\dagger-\lambda^\star$ differs, because in one case that difference is set by an operator mismatch and in the other by a lag.

**$\Pi$ is emitted.** `atlas`'s L6 reports it from the partition of unity on every overlapping compile and **decertifies when the partition does not declare which of its cells a stale datum can reach** — the same discipline the halo rule applies to an undeclared overlap. `assembly.sigma_halo_bound` evaluates the product.

**This is the theorem the "agreement is an operator" claim was reaching for.** It says precisely what [[composition-error-theory]] §4.0 asserted informally — accuracy is set by how well the operator is approximated, not by how tightly the approximation is satisfied — and it adds the amplifier that assertion omitted: $1/\beta$, the inf-sup constant that §4.2 of that page correctly identified as *the missing hypothesis* and could not place in a formula.

**Both factors are exactly what a probe returns.** [[probed-dtn-coupling]] assembles $\tilde\Lambda$ as a matrix; $\beta=\sigma_{\min}$ of it, and $\lVert\Lambda-\tilde\Lambda\rVert$ is the comparison against a reference assembly. **The construction and the bound were built for each other without either page noticing.**

---

# 4.2 Bounding $\sigma$ on the MULTIRATE branch — where the exchange interval enters

Written 2026-09-03 from CS-11 ([[case-study-brake-thermal-atlas-0.1]]), and it closes **W90**, which has stood open since Tier 17 with the sharpest statement on the worklist: *"R9 covers the flux transient and is $62\times$ too small; the lag over a long exchange interval is $\sigma$, and nothing bounds it as a function of that interval."*

§4 bounds $\sigma$ where an interface **operator** is wrong. §4.1 bounds it where a **stale halo datum** reaches into a blend. This section bounds it where the two agents run at **different clocks**, and the mechanism is neither of those: the trace is not wrong and the halo is not contaminated — the trace is *old*, for the whole of an interval the slow agent's step fixes and R4 forbids shrinking.

## The form

Over one exchange interval $[t,\,t+\Delta t_{\text{ex}}]$ the composition hands every agent $\lambda(t)$ and the tightly coupled trace moves to $\lambda^\star(t+s)$. Write the lag

$$\delta\lambda(\Delta t_{\text{ex}}) \;:=\; \lambda^\star(t+\Delta t_{\text{ex}}) - \lambda^\star(t), \qquad \lVert\delta\lambda\rVert \;\le\; \dot\lambda_{\max}\,\Delta t_{\text{ex}}$$

and the bound is

$$\boxed{\;\sigma(\Delta t_{\text{ex}}) \;\le\; s_\Gamma\,\lVert\delta\lambda\rVert \;+\; C_2\,\lVert\delta\lambda\rVert^2 \;\le\; s_\Gamma\,\dot\lambda_{\max}\,\Delta t_{\text{ex}} \;+\; C_2\bigl(\dot\lambda_{\max}\,\Delta t_{\text{ex}}\bigr)^2\;}$$

**Two measured constants and one run-derived rate**, which is the same division of labour `sigma_lag` already carries and is why this belongs beside that field rather than instead of it:

| symbol | what it is | who owns it |
|---|---|---|
| $s_\Gamma = \mathrm d\sigma/\mathrm d(\text{uniform lag})$ | the seam's own slope. **W86** established that this is the part of $\sigma$ that transfers — constant to five digits over five consecutive macro-steps of `thermal_seam` | the **seam**. One probe, no run |
| $C_2$ | the curvature interface power's **bilinearity** forces. $\int_\Gamma e\,f$ with $f$ affine in $e$ is quadratic in the trace, so a first-order law is an approximation and the second term says by how much | the **seam**. The same probe |
| $\dot\lambda_{\max}$ | the trace's own rate of change | the **run**, and nothing at compile time can know it, because nothing has stepped yet |

## Measured

CS-11's brake seam — `ThermoStruct2D` conduction against a convecting cooling duct, $500{:}1$ at native clocks — with $s_\Gamma = 9.4966\times10^{-3}\,\mathrm K^{-1}$ and $C_2 = 1.4288\times10^{-5}\,\mathrm K^{-2}$ from the probe alone, and each interval's own measured lag. **Nothing below is fitted to this table.**

| $\Delta t_{\text{ex}}$ [s] | ratio | lag [K] | measured $\sigma$ | bound | bound / measured | order |
|---|---|---|---|---|---|---|
| $1\times10^{-4}$ | $1$ | $0$ | $\mathbf{0}$ | $0$ | — (control) | — |
| $5\times10^{-4}$ | $5$ | $5.304\times10^{-3}$ | $3.533\times10^{-5}$ | $5.037\times10^{-5}$ | $1.426$ | — |
| $2\times10^{-3}$ | $20$ | $2.485\times10^{-2}$ | $1.435\times10^{-4}$ | $2.360\times10^{-4}$ | $1.645$ | $1.011$ |
| $1\times10^{-2}$ | $100$ | $1.252\times10^{-1}$ | $6.186\times10^{-4}$ | $1.189\times10^{-3}$ | $1.922$ | $0.908$ |
| $5\times10^{-2}$ | $500$ | $7.524\times10^{-1}$ | $3.285\times10^{-3}$ | $7.154\times10^{-3}$ | $2.178$ | $1.037$ |
| $2\times10^{-1}$ | $2000$ | $4.211$ | $1.853\times10^{-2}$ | $4.024\times10^{-2}$ | $2.172$ | $1.248$ |

> **The bound holds at every graded interval and is loose by between $1.43\times$ and $2.18\times$ over $400\times$ of exchange interval and $2000\times$ of clock ratio.** The defect is first order in the interval — the order column sits at $1.0$ until the last row, where the quadratic term starts to matter and the bound's own second term is what keeps it above.

**Ratio $1$ is the control and its defect is exactly zero, not small.** One duct step *is* the interval, so there is nothing to be stale about; anything else would mean the instrument was measuring itself. It carries no tightness ratio and quoting $0/0$ as one would be **W106**'s mistake.

## 4.2.1 The clock ratio is not the variable, and that is what makes the bound usable

The interval sweep above moves two things at once, so on its own it cannot say which one $\sigma$ is a function of. Pinning the interval at the native $5\times10^{-2}\,\mathrm s$ and **refining the fast agent** moves the ratio $8\times$ with the staleness untouched:

| Courant number | $\Delta t_{\text{stable}}$ | ratio | $\sigma$ | relative |
|---|---|---|---|---|
| $0.40$ | $2.048\times10^{-5}$ | $2442$ | $3.285085\times10^{-3}$ | $1.000000$ |
| $0.20$ | $1.024\times10^{-5}$ | $4883$ | $3.285386\times10^{-3}$ | $1.000092$ |
| $0.10$ | $5.120\times10^{-6}$ | $9767$ | $3.285536\times10^{-3}$ | $1.000137$ |
| $0.05$ | $2.560\times10^{-6}$ | $19533$ | $3.285611\times10^{-3}$ | $1.000160$ |

$$\textbf{8}\times \textbf{ of clock ratio at a fixed interval moves } \sigma \textbf{ by } 1.00016\times.$$

**This is what makes the bound worth having.** A real conduction-against-convection seam runs at $10^4$–$10^5$, which no single-rate referent can be built at; a bound written in the *ratio* would be uncheckable there. A bound written in the *interval* is checked here at ratio $5$ and applies at ratio $10^5$, because the ratio is not in it.

**Scoped, and the scope is the interesting part.** This holds for a fast agent that **sub-steps internally at its own stability limit** — what a classical explicit solver does, and what makes its declared `dt_native` a bookkeeping choice rather than a physical constraint. A **frozen learned expert cannot refine its own march**: its `dt_native` is fixed by its weights, so at a fixed interval a coarser ratio means genuinely fewer, genuinely larger steps, and the ratio re-enters through the fast side's own discretization error. That case is **unmeasured**, it is exactly the case R4's asymmetry is about, and it is the one the substitution campaign will meet.

## 4.2.2 The remedy R4 leaves open, and what it is worth

R4 forbids $\Delta t_{\text{ex}} < \max_i \Delta t_i$, so the first term of the bound cannot be reduced by shortening the interval. **W17's $W>1$ is the one axis that remains**, and it does not shorten the interval — it stops the trace inside it being constant. R3 admits it wherever every agent at the seam declares `bc_time_varying`, which both of CS-11's do.

Carrying $\lambda$ as a linear waveform built from the **previous** interval's rise replaces the first difference with the second, and measured over one interval it is worth $3.6\times$ to $56.6\times$; over a $200$-interval rollout it is worth $1.84\times$ on the accumulated defect. **The reduction is not monotone in the interval and should not be**: the ratio between a first difference and a second is the trace's *curvature*, which is a property of where on the transient the interval sits. That is W86's shape again — the transferable quantity is a slope and the thing that moves is the profile — and the practical reading is that **a waveform buys an order, not a factor.**

## 4.2.3 Where this sits against R9

R9 matches the **time-integrated** flux and is the rule that has refused every multirate graph since this compiler existed. Measured on `thermal_seam`, R9's own term is $62.4\times$ smaller than the lag term. Nothing here retracts R9 — a pointwise match across two clocks is still not conservative and CS-11's graph is still refused when it declares one — but the two terms now have different statuses:

| term | rule | status |
|---|---|---|
| flux transient (pointwise vs integrated) | **R9**, a refusal | covered, and it is the smaller term by $62\times$ |
| stale trace over the interval | **this section** | **bounded**, first order in $\Delta t_{\text{ex}}$, with two seam constants and one run rate |
| interpolation order caps the scheme order; coupling stiffness sets stability | `L7/R9/order` | still a decertification. Neither is touched here |

**And the clock ratio turns out to be where R9's term lives and this one does not** — which is the cleanest statement of why the two were confused. The ratio moves the fast side's flux transient and leaves the staleness alone; the interval moves the staleness and drags the ratio with it only because R4 ties them together.

---
# 5. Bounding $\gamma$ — and why the accelerator choice is a bound, not a preference

$\gamma\le C_\mu\lVert\lambda^{(k)}-\lambda^\dagger\rVert$, and the second factor is whatever the solver guarantees:

| accelerator | $\lVert\lambda^{(k)}-\lambda^\dagger\rVert$ | governed by |
|---|---|---|
| **Richardson / classical Schwarz** | $\rho^{k}\lVert\lambda^{(0)}-\lambda^\dagger\rVert$ | the Schwarz contraction $\rho$; one-level $\rho\to1$ as subdomains shrink — the $H^{-2}$ of [[schwarz-iteration-atlas-0.1]] §3.2 |
| **Krylov on the interface** | $\sim\Bigl(\tfrac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\Bigr)^{k}$ | $\kappa(\tilde\Lambda)$ — *conditioning*, not subdomain count |
| **Direct Schur solve** | $\sim\kappa\cdot\epsilon_{\text{mach}}$ | **negligible** |

**So the honest statement of what [[probed-dtn-coupling]] buys is: it makes $\gamma$ vanish, leaving $\tau+\sigma$** — and it *also* reduces $\sigma$ by climbing the ladder, which is the larger of the two effects. The $k_{\min}=5$ arithmetic that disqualifies one-level iteration for a $7D$ signal is a statement about $\rho$; a direct solve has no $\rho$.

---

# 6. The stability constant, and the one theorem that controls it

Perturbing $u^n$ perturbs the interface problem, so $\lambda$ moves too. Chain rule through the coupled solve:

$$L \;\le\; \lVert\mathcal A\rVert\cdot\max_i\operatorname{Lip}_v(\mathcal E_i)\cdot\Bigl(1+\frac{C_\mu\,C_{\mathcal G}}{\beta}\Bigr), \qquad C_{\mathcal G}=\Bigl\lVert\frac{\partial\mathcal G}{\partial u^n}\Bigr\rVert$$

> **$\beta$ sits in the denominator of both $\sigma$ (§4) and $L$ (§6). An ill-conditioned interface hurts twice — once additively in the per-step defect, once multiplicatively in the accumulation.** That is a stronger reason to measure $\beta$ than either page that asked for it gave.

## 6.1 The passivity theorem

**Hypothesis.** Each agent is *incrementally passive* at its ports: for two trajectories with the same forcing, there is a storage $H_i\ge0$ with

$$\frac{d}{dt}H_i(\Delta u_i)\;\le\;\int_{\partial\Omega_i}\Delta e_i\,\Delta f_i\,ds$$

and the interconnection is **power-preserving** — which [[port-algebra-atlas-0.1]]'s rule $e_A=e_B,\ f_A=-f_B$ *is*, by construction.

**Conclusion.** Port powers cancel in pairs, so $V=\sum_i H_i(\Delta u_i)$ is non-increasing, and in the $V$-norm

$$\boxed{\;L\;\le\;1\quad\text{for any graph, at any } N,\ \text{with no global analysis}\;}$$

This is the only route on this page to the $L\le1$ branch, and it is **structural rather than empirical**: it is inherited from a declared property of each part, not fitted from a rollout. It is also the whole of the argument in [[temporal-error-accumulation]] §2.2 for why decomposition is worth its coupling cost — a monolithic network has no per-part handle on its Lipschitz constant.

---

# 7. The master bound

$$\boxed{\;\lVert e^N\rVert \;\le\; \underbrace{L^N\lVert e^0\rVert}_{\text{initial}} \;+\; \sum_{n=1}^{N} L^{\,N-n}\Bigl[\;\underbrace{\tau^n}_{\text{agent}} \;+\; \underbrace{\frac{C_\mu}{\beta_n}\bigl\lVert\Lambda-\tilde\Lambda\bigr\rVert_n\lVert\lambda^{\star,n}\rVert}_{\text{transmission}} \;+\; \underbrace{\gamma^n}_{\text{solve}}\;\Bigr]\;}$$

and with windows of $W$ macro-steps ([[temporal-error-accumulation]] §3), the $\gamma$ sum runs over $J=N/W$ windows rather than $N$ steps, while $\tau$ and $\sigma$ stay per-step.

> **Evaluated 2026-08-27 with every constant measured — the first time this bound has been a number rather than a shape.** Four `reference.WindowNS` windows against an identically-discretized monolith ([[tier0-measurements]]), for two schemes. The `as-built` agent embeds its pressure solve, which **R10** now refuses; the `split-step` agent gives the elliptic part to the composition layer, which is what §2.1 of [[probed-dtn-coupling]] prescribes.
>
> | scheme | $L$ | $\tau$ | $\sigma$ | $\gamma$ | asymptotic bound | measured plateau | ratio at $N=1$ |
> |---|---|---|---|---|---|---|---|
> | as-built | $0.972776$ | $2.935\times10^{-4}$ | $2.253\times10^{-5}$ | $0$ | $1.16\times10^{-2}$ | $2.39\times10^{-3}$ | $0.93$ |
> | **split-step** | $\mathbf{0.979644}$ | $1.335\times10^{-6}$ | $3.733\times10^{-8}$ | $0$ | $6.74\times10^{-5}$ | $\mathbf{1.23\times10^{-5}}$ | $\mathbf{0.99}$ |
>
> **The bound holds at every step for both, is tight to 1% at $N=1$, and is $5\times$ loose asymptotically** — the right shape, since at one step the bound *is* $\tau+\sigma$ while over a rollout it assumes worst-case alignment of defects that partly cancel. **The measured error saturates** rather than drifting, and the corrected scheme's plateau is $194\times$ lower.
>
> **The split-step $L$ matches the monolith's ($0.979650\pm0.00041$) to five decimal places**, which is a stronger statement than accuracy: the composed map is the *same dynamical map* to the precision the fit resolves.
>
> **Three qualifications.** (i) The $\sigma$ above is the *observed field defect*, and §4's product form for it is **measured to be vacuous for this scheme** — see the box in §4. (ii) $\lVert\mathcal A\rVert$ in §6's product is now known to be $1$ for any partition of unity, by proof rather than measurement, so §6's bound on $L$ has one fewer free constant — but it is still untested, because $\operatorname{Lip}_v(\mathcal E_i)$ and $C_{\mathcal G}$ are not measured. (iii) **$\tau$ was misnamed in the first run**: $99.8\%$ of the defect while the agent was the reference solver, because it was a decomposed elliptic solve — a $\sigma$ one level down, per [[plug-in-composition-theorems]] §1.4. Under the corrected agent $\tau$ falls $220\times$ and the label is accurate.

**Each term, its owner, and its status in this vault:**

| term | controlled by | measurement | status |
|---|---|---|---|
| $L$ | incremental passivity (§6.1); windowing caps the *method's* share at $L^W$ | fit from one paired rollout; or $\lambda_{\min}$ of the probed symmetric part | **never measured** |
| $\tau$ | faithful local solve — a real boundary channel, non-periodic windows | vs. a validated classical reference on the true restriction | **measured, dominant**: OP-5's $66\%$; W9's $60\times$ |
| $\sigma$ | the DtN rung, amplified by $1/\beta$ | assemble $\tilde\Lambda$, compare to $\Lambda^{\text{ref}}$, report $\beta$ | **never measured, never reported, and maximal in the known failure** |
| $\gamma$ | accelerator choice (§5) | interface residual at convergence | **the only one routinely reported** |

> **The framework measures the smallest term, controls the second-largest, and has never reported the other two.** That single sentence is the practical content of the bound.

---

# 8. Specializations — the bound is a family, not one inequality

| setting | reduction | reads |
|---|---|---|
| **Monolithic autoregressive model** | $N=1$ subdomain, $\sigma=\gamma=0$ | $\lVert e^N\rVert\le L^N\lVert e^0\rVert+\sum L^{N-n}\tau^n$ — [[autoregressive-rollout-stability]] |
| **Exact interface operator** | $\tilde\Lambda=\Lambda\Rightarrow\sigma=0$; converged $\Rightarrow\gamma=0$ | $\lVert e^N\rVert\le\sum L^{N-n}\tau^n$ — **the worst-agent conjecture, and here it is a theorem** |
| **Passive agents** | $L\le1$ | $\lVert e^N\rVert\le\lVert e^0\rVert+\sum_n(\tau+\sigma+\gamma)$ — linear in $N$, never exponential |
| **Contractive** | $L<1$ | $\lVert e\rVert\le\frac{\max(\tau+\sigma+\gamma)}{1-L}$ — **bounded uniformly in time** |
| **Periodic windows (the measured case)** | $\Lambda\equiv0$, $\tilde\Lambda\equiv0$, $\gamma\equiv0$ bitwise | the bound is controlled entirely by $\tau$ and $\sigma$, with $\sigma$ maximal and **every reported diagnostic reading zero** |

The second row is worth stating plainly: **the conjecture that composed error is of order the worst agent's is exactly the master bound with $\sigma=\gamma=0$ and $L\le1$** — three hypotheses, each now a measurable quantity rather than an assumption.

---

# 9. When the bound goes vacuous — and it must be declared

For a chaotic system $L\approx e^{\lambda_1\Delta t}>1$ with $\lambda_1$ the leading Lyapunov exponent, so the bound grows until it exceeds any tolerance at the **predictability horizon**

$$T_{\text{pred}} \;\approx\; \frac{1}{\lambda_1}\,\ln\frac{\delta_{\text{tol}}}{\tau+\sigma+\gamma}$$

**Past $T_{\text{pred}}$ this page proves nothing, and no construction on it changes that** — the divergence is the physics, not the method. Passivity and windowing buy the *method's* share of the amplification, not the system's.

**The framework consequence, and it is not currently anywhere in the vault.** Beyond $T_{\text{pred}}$ the deliverable must change from a **trajectory** claim to a **statistical** one — invariant measure, spectra, structure functions, long-time averages — and that needs a different theory (shadowing under hyperbolicity, or ergodic-average estimates), a different gate set, and a different notion of what $\tau$ even means. The vault has **zero** mentions of shadowing, ergodicity or long-time statistics; it discusses chaos only when choosing architectures. See [[generalization-requirements]] G6.

**[AI Inference]:** the useful practical form is to report $T_{\text{pred}}$ alongside every rollout, computed from a fitted $L$. A rollout quoted past its own predictability horizon is reporting noise with a decimal point, and nothing currently stops that happening.

> **Measured 2026-08-27, and the first fitted $L$ lands on the branch this section does not cover.** $L=0.948441\pm0.0048$ for the composed map, $0.979649\pm0.00041$ for the monolith it approximates — **both below one**, so $\lambda_1<0$, $T_{\text{pred}}$ is unbounded, and the vacuity condition never arrives. This section is written as though $L>1$ were the case to plan for; a laminar wake at $\mathrm{Re}$ this low is the case where it is not, and the measured error duly **saturates** at $5.4\times10^{-4}$ rather than growing (§7's box). **The three-branch horizon rule [[gap-worklist]] W14 records — exponential at $L>1$, linear at $L=1$, unbounded at $L<1$ — is the right shape and this is its first instance.** It also means the configuration says nothing about the chaotic regime: $T_{\text{pred}}$ is untested where it matters, and a turbulent case remains the experiment that would test it.

---

# 10. Falsifiable measurements

| # | Measurement | Cost | Prediction |
|---|---|---|---|
| 1 | **Fit $L$** from a paired perturbed rollout | one paired rollout | $L=1+\eta$, small $\eta>0$ — OP-2's monotone drift |
| 2 | **Assemble $\tilde\Lambda$; report $\beta$ and $\kappa$** | one probe ([[probed-dtn-coupling]] §9.1) | $\beta$ small at declared ports, comfortable at tile seams |
| 3 | **$\sigma$ directly**: $\lVert\Lambda^{\text{ref}}-\tilde\Lambda\rVert/\beta$ | two assemblies | $\sigma\gtrsim\tau$ once the periodic window is fixed — this is the term nobody has looked at |
| 4 | **Three-way split of $d$** on one step: $\tau$, $\sigma$, $\gamma$ separately against the classical reference | one instrumented step | $\gamma\lll\tau,\sigma$ — confirming the framework reports the smallest term |
| 5 | **$T_{\text{pred}}$** from (1) | arithmetic | shorter than the rollouts currently quoted |

**[AI Inference]:** (4) is the decisive one for this page and it needs no new machinery — the classical reference, the harness and the masks all exist. If the split comes out $\gamma\lll\tau,\sigma$ as predicted, then **every diagnostic the framework currently reports is measuring the negligible term**, which is a finding about the instrumentation rather than the physics, and cheap to act on.

---

## See Also

- [[composition-error-theory]] — the $\tau\neq\varepsilon$ distinction §3 depends on; its §4 constructions are §7's "controlled by" column
- [[probed-dtn-coupling]] — produces both factors of §4's $\sigma$ bound, and collapses $\gamma$ to roundoff (§5)
- [[temporal-error-accumulation]] — §2's recursion and §6's $L$; the windowing that caps the method's share of the amplification
- [[general-coupling-scheme]] — the algorithm this bound grades, parameterized so any case can instantiate it
- [[generalization-requirements]] — what is still missing before the bound applies to arbitrary systems; G6 is §9's gap
- [[schwarz-iteration-atlas-0.1]] — §3.1's faithfulness hypothesis, which §3.1 here restates as "iteration reduces $\gamma$ and nothing else"
- [[port-algebra-atlas-0.1]] — the power-preserving interconnection §6.1's theorem requires
- [[conservation-as-constraint-atlas-0.1]] — why conservation constrains functionals of $e$ rather than $\lVert e\rVert$
- [[open-problems-atlas-0.1]] — OP-2 ($L=1+\eta$), OP-5 ($\tau$ dominance), OP-1 ($\beta$ as the general form of its conditioning lesson)
- [[theory-closure-audit]] — **§3 there states the seven-hypothesis envelope this bound silently assumes**, which no section here delimits; it also certifies §2–§8 as closed and §9 as the one open hole *inside* the envelope
- [[interface-transfer-theory]] — names **two components of $\sigma$ this page's §4 does not separate**: $\sigma_{\text{nc}}$ from a non-conforming interface space and $\sigma_{\text{time}}$ from the temporal representation of $\lambda(t)$. Neither is reduced by iteration, for the same reason §3.1 gives for $\sigma$ itself
- [[plug-in-composition-theorems]] — §7's bound **at $N=1$** is the attribution theorem: a subassembly's $\sigma$ and $\gamma$ become its $\tau$ one level up, so **the three-way split is not invariant under regrouping**. It also shows that $\beta$'s appearance in both §4 and §6 is why a strictly more accurate expert can worsen a composition
- [[end-to-end-architecture-spec]] — the system this bound grades, layer by layer, with the envelope as a per-run stamp. Its §11.3 reads **three horizon branches** off §8's specialization table — exponential at $L>1$, **linear at $L=1$**, unbounded at $L<1$ — where §9 here writes only the first, and it links §6.1's passivity certificate to the claim type
