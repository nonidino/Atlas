# Open Problems — Atlas 0.1

**Type:** Concept page (folder: `Atlas 0.1/common/`)
**Related:** [[conservation-as-constraint-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[spec-wind-farm-wake-atlas-0.1]] · [[wind-farm-implementation-log]] · [[f1-pathmap-and-end-goal]]

> **What this page is for.** Problems that were found by measurement, diagnosed to a cause, and then **deliberately deferred** rather than solved. Each entry records what is broken, what is known about why, what was tried, and what would count as a fix — so that picking it up later costs a read rather than a rediscovery.
>
> This is not a bug list. A bug goes in the log and gets fixed. What lands here is a problem whose fix is a **design decision** that should not be made in a hurry, and which usually touches the spec rather than the code.

---

# OP-1. The thrust projection cannot enforce C2 — the correction direction is degenerate

**Status:** open, deferred 2026-08-22 · **Found at:** wind-farm W5 · **Blocks:** conservation law C2 at level 5, gate W5's post-projection item, and the same construction in every later case study

## The one-paragraph version

[[spec-wind-farm-wake-atlas-0.1]] §7.3 specifies a **min-norm correction**: after each macro-step, add the smallest divergence-free velocity perturbation that makes the fluid side's momentum budget match the thrust the actuator-disk expert computed. The correction *mechanism* works — it is a scalar quadratic with a closed-form root, and it drives the measured residual to $10^{-15}$ in one step. **The direction it corrects along does not.** It is a *uniform* velocity field, the momentum balance is Galilean-invariant under one, and so the constraint barely responds: $dg/d\lambda = 1.02\times10^{-6}$ against a residual of $1.01\times10^{-1}$. The correction required is $\lambda = 9.85\times10^4$, which produces a flow at $770$ times freestream.

## Why the direction is uniform — the part worth not re-deriving

The spec builds the direction as the Leray projection of the control volume's indicator function in $x$:

$$\mathbf w = \mathbb P_{\text{Leray}}\bigl[\mathbb 1[x_l\le x\le x_r]\,\hat{\mathbf x}\bigr]$$

A field that varies only in $x$ and points along $x$ has, for every wavevector, $\hat{\mathbf d} \parallel \mathbf k$ — it is **pure gradient in every mode**. The Leray projection removes exactly the gradient part, so it removes everything except $k=0$:

$$\hat{\mathbf w}(\mathbf k) = \hat{\mathbf d} - \frac{\mathbf k(\mathbf k\cdot\hat{\mathbf d})}{|\mathbf k|^2} = 0 \quad\text{for all }\mathbf k\ne 0,\qquad \hat{\mathbf w}(0) = \overline{\mathbf d}$$

Measured confirmation: $\max|w_u|$ equals its own mean to every printed digit.

## Why it appeared to work before, and why that is the interesting part

Under the **periodic** pressure recovery the projection ran without obviously exploding. That was not because the direction was viable — it was because a periodic pressure **cannot carry a mean gradient**, so it could not perform the cancellation that makes the residual Galilean-invariant, and $\beta$ came out spuriously nonzero.

With the pressure recovered correctly (Neumann data containing $-\partial_t\mathbf u$), the cancellation is exact to five figures:

| term | response to $\lambda\mathbf w$ |
|---|---|
| $dM/dt$ | $+6.19\times10^{-2}\,\lambda$ |
| pressure flux | $-6.19\times10^{-2}\,\lambda$ |
| **net** | $\sim10^{-6}\,\lambda$ |

**Two bugs that concealed each other.** The same missing degree of freedom made the *target* wrong and made the *direction* look viable. This is the general hazard worth carrying: a defect in an instrument can mask a defect in the thing the instrument steers, and fixing the instrument first is what exposes the second.

## What was tried

- **Fixing the pressure** (2026-08-22, `pressure.py`). Necessary and not sufficient. It made the target correct and the residual readable — pre-projection $r_T$ fell from $0.73$ to $0.14$/$0.047$, at $2.5\times$ its null floor instead of $32\times$ — and it is what *revealed* this problem.
- **Enforcing anyway.** Destroys the flow while reporting post-projection $r_T = 7.5\times10^{-11}$, four orders inside the gate. A guard now refuses any correction exceeding half the field's RMS and marks the gate item BLOCKED.

## What would count as a fix

A correction direction that is (i) divergence-free under the domain's real boundary conditions, (ii) **not** in the null space of the momentum constraint, and (iii) local to the rotor rather than spread over the window. Concretely: a direction that is *not* uniform must have $\mathbf k\ne0$ content surviving the Leray projection, which requires it to vary in $y$ as well as $x$ — for instance a divergence-free dipole straddling the disk, or a correction built from a stream function localized to the streamtube.

**The cleanest reframing:** the min-norm problem should be posed over a *subspace* of divergence-free fields chosen so the constraint gradient is bounded away from zero, and the conditioning $|\beta|$ should be reported as a gate item in its own right. A projection whose sensitivity is not $O(1)$ is not a projection.

## Why deferring is defensible

[[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule permits a conserved quantity to be **measured** rather than enforced, provided the claim is stated that way. C2 is now measured at $2.5\times$ its own noise floor with an explicit error bar, which is a reportable scientific result; C1 (mass) *is* enforced, to $5.6\times10^{-14}$. The case study can complete W6–W11 and answer its falsification criteria without C2 enforcement. **What must not happen is C2 being quietly reported as enforced**, which the guard and the tri-state gate now prevent.

## Where the details live

[[symmetry-averaging-atlas-0.1]] (the concept and the proof), [[wind-farm-implementation-log]] entries 2026-08-23 (operator equivariance; group average). Code: `atlas/invariants/symmetry.py`, `tests/atlas/invariants/test_symmetry.py` (15 tests).

---

# OP-2. The coupled system's upstream induction drifts

**Status:** open, **recovering under measurement** · **Found at:** wind-farm W5/W9 · **Blocks:** W7 (symmetry), W10 (array efficiency) · **Diagnosed 2026-08-24** — under a non-periodic window solve this quantity *inverted sign* and diverged; the cause was the window outflow condition being applied to 117 interior tile seams, not the expert. **Updated 2026-08-24 (later):** with the open faces declared geometrically, the induction moves toward the reference at all four upstream stations on every one of eight macro-steps, and at $x=-0.5$ it has **changed sign** ($-0.0546 \to +0.0013$) — the first correctly-signed upstream induction in any coupled run of this case study. $P_2/P_1$ falls monotonically $1.0216 \to 0.8144$. It is **not resolved**: $-0.0418$ against a monolithic $+0.0473$ at $x=-1$, and $0.8144$ against $0.403$–$0.429$. Whether it settles at the reference or stalls short is the $t=20$ rollout's question — and W9's periodic run drifted monotonically to $25.9\%$ rather than settling, so a monotone trend is not by itself an answer here. See the updates below.

Distinct from OP-1 and **not deferred** — recorded here because it is the sibling finding and the two are easy to confuse.

Measured against the monolithic 2D baseline on an *identical* configuration (`reference.ChannelNS`, same domain, boundary conditions, disk and thrust law; only the fluid operator differs):

| $x$ | 2-D NS baseline | coupled, $t=20$ |
|---|---|---|
| $-3$ | $0.7\%$ | $14.1\%$ |
| $-1$ | $\mathbf{5.7\%}$ | $\mathbf{25.9\%}$ |

$4.5\times$, and **drifting rather than settled**: $9.3\%$ at $t=2$ rising monotonically through $13.9$, $17.3$, $22.1$, $25.0$ to $25.9\%$ at $t=20$, passing through the correct answer at about $t=1$. A steady blockage equilibrates within one or two convective times, as the baseline does. *Something is accumulating*, which is a more findable statement than "the number is too big".

**Narrowed 2026-08-23 by gate W9, and the narrowing moves it out of the coupling.** Running the identical system with the *same 124 tiles and the agents removed* — no partition, no ports — gives $P_2/P_1 = 1.220$ against the eight-agent run's $1.143$. **The defect is present, and slightly worse, with no decomposition at all.** So this is not a coupling problem: it belongs to the frozen expert, next to OP-3, and the two are plausibly one phenomenon. Composition error accounts for $0.273$ of a total field error of $1.618$ in relative $L_2$, against the expert's own $1.762$.

**Still listed as active rather than merged into OP-3** because the *drift* is a distinct signature — OP-3 is a steady over-dissipation, this is a monotone accumulation — and the experiment that separates them is unchanged: vary the number of expert calls per unit physical time and see whether the drift rate follows call count or clock.

**[AI Inference]:** the monotone growth points at a per-step bias rather than a wrong steady state — the same shape as W0's spurious-injection finding, where the expert emits structure on every call and it accumulates on a closed window. If so, the diagnostic is to vary the number of expert calls per unit time and see whether the drift rate follows call count rather than physical time, which is exactly the experiment that separated dissipation-per-call from dissipation-per-time at W0.

**Updated 2026-08-24 — under a non-periodic window solve the induction does not merely drift, it *inverts*.** Replacing the periodic windows with `WindowNS` (Dirichlet ring, characteristic outflow) and a solver expert changes the sign of the quantity this entry is about:

| $x$ | 2-D NS baseline | coupled, periodic windows | **windowed, $t=1.0$** |
|---|---|---|---|
| $-3$ | $+0.6\%$ | $+6.9\%$ | $\mathbf{-2.5\%}$ |
| $-1$ | $+4.7\%$ | $+17.3\%$ | $\mathbf{-14.3\%}$ |

A negative deficit is a **jet**: the flow one diameter ahead of the disk is $14\%$ *faster* than freestream, while the bypass at $y=+2$ sits at $0.949$ — the slow-core/fast-bypass pattern of a real disk, exactly reversed. And it grows as a single coherent mode, by $\times6.08$, $\times6.18$, $\times6.33$ at $x=-3,-2,-1.5$ over one convective half-time. **This is the divergence** that killed both non-periodic rollouts, and it is the same quantity as the drift recorded above, driven past zero rather than merely too far.

**This qualifies the W9 attribution.** The paragraph above concludes from W9 that the defect "is not a coupling problem: it belongs to the frozen expert." That reading survives for the *periodic* configuration W9 tested. It does not extend to this one: the expert here is an exact solver, and a **single** forced `WindowNS` window reproduces the correct upstream sign at every station. The inversion is manufactured by the assembly of many windows, not by the local operator — so at least one branch of OP-2 is a composition defect after all, and the entry's central claim is now configuration-dependent rather than settled.

**The diagnostic this entry proposed is available and unrun.** The `[AI Inference]` above predicts a per-step bias rather than a wrong steady state, and names the experiment: *vary the number of expert calls per unit physical time and see whether the drift follows call count or clock*. [[spec-wind-farm-wake-atlas-0.1]] §8.1 prescribes $\Delta t_{\text{macro}} = 0.05$ and the code runs $0.25$ — the knob is already five notches off spec, and under impulse forcing that makes each hit $\Delta u / \langle U_d\rangle = 0.696$ against $0.14$ at the specified step. **OP-2's predicted diagnostic and OP-5's next test are the same run**, which is a reason to prefer it over either alone.

**Diagnosed 2026-08-24 — the cause is the window's outflow condition, and neither available transmission is usable.** A refinement test on one window with the **rotor force switched off**: `dirichlet` transmission returns $1.2182$, $1.2182$, $1.2183$ at 48, 190 and 379 sub-steps — converged. `characteristic` returns $1.1730$, $3.1044$, $11.8699$. **Refining the time step makes it worse**, which is the signature of an inconsistent boundary condition rather than an unstable one, and the outlet velocity reverses from $+0.92$ to $-3.74$ inside a single macro-step. A zero normal gradient is *exact* for an $x$-invariant outflow — a clean synthetic wake is stable under both transmissions to six digits — and wrong in proportion to the streamwise structure crossing the outlet. **The error it commits is itself streamwise structure, so it compounds.** So `dirichlet` is stable and traps the wake, `characteristic` releases the wake and is inconsistent, and the build has no usable open boundary. The remedy is a convective (Orlanski) outflow, $\partial_t u + U_c\,\partial_x u = 0$.

This also **retires the diagnostic proposed above.** The $\Delta t$ knob is real but it is a modulator, not the cause: the ring is refreshed once per macro-step, so a smaller step means fewer sub-steps of accumulated boundary error before the reset. Varying calls-per-unit-time would have produced a clean signal and the wrong attribution.

**Updated 2026-08-27 — $L$ is now a measured number, and in a `WindowNS` decomposition it is *below* one.** [[tier0-measurements]] fits the amplification of the composed macro-step map from a paired rollout, at three schemes:

| map | $L$ | std. err. | $\eta=L-1$ |
|---|---|---|---|
| monolith reference | $0.979650$ | $0.00041$ | $-0.0204$ |
| composed, **corrected** (split-step, halo 21) | $\mathbf{0.979644}$ | $0.00041$ | $\mathbf{-0.0204}$ |
| composed, as-built (embedded pressure solve) | $0.972776$ | $0.00043$ | $-0.0272$ |
| composed, one-cell-shared tiling, $\chi=1/(\text{owner count})$ | $0.948441$ | $0.0048$ | $-0.0516$ |
| composed, probed-DtN (a rung R2b now refuses) | $0.974593$ | $0.00042$ | $-0.0254$ |

**The first three rows are the ones to read.** The corrected composition's $L$ matches the monolith's to five decimal places; the two below it are schemes since diagnosed as defective, kept because the spread across them is the point — **$L$ varies by $3\%$ across schemes on the same physics, so quoting "the composed $L$" without naming the scheme is meaningless.**

Amplitude-independent to six digits, so the fit is the linearized amplification rather than a nonlinear excursion. Against it, this entry's own drift — $9.3\%$ at $t=2$ to $25.9\%$ at $t=20$ at $\Delta t_{\text{macro}}=0.25$ — implies $L=1.0143$, i.e. $\eta=+1.43\times10^{-2}$. **The two have opposite signs.**

**Three things follow, and the third is the useful one.**

1. **This does not close OP-2.** The fitted numbers come from a Dirichlet-ring exact solver at $\Delta t=0.05$; the drift above comes from a periodic-window frozen checkpoint at $\Delta t=0.25$. Three variables differ at once, which is exactly the confound this entry has been narrowing for four days.
2. **$L>1$ is not a property of windowed decomposition as such — and a correctly built decomposition does not change $L$ at all.** The corrected composition reproduces the monolith's $L$ to five decimals. The over-contraction first measured ($0.948$) was a *defective assembly*: a pinned seam layer given full weight, which damps what crosses it. **So decomposition per se neither amplifies nor damps; a bad one damps.** Whatever accumulates in the runs this entry is about, it is not the decomposition.
3. **Passivity is eliminated as the cause, in this configuration, without a rollout** — [[probed-dtn-coupling]] §4.2's [AI Inference] predicted a negative mode below the floor and there is none: $\mu>0$ on all four seams, $\pi=0$. The structural and the empirical routes agree, which is the strongest available check on either.

**The instrument this entry needed now exists.** $L$ is fitted from one paired rollout in under a minute, so the periodic-versus-Dirichlet, checkpoint-versus-solver and $\Delta t$ variables can be moved **one at a time** and each given its own $L$. That is a cleaner separation than the deficit-drift trace this entry has been read from, and it is the next experiment.


---

# OP-3. The frozen expert dissipates wakes ~20× too fast

**Status:** open, characterized, **now quantified against a valid reference** · **Found at:** W0, confirmed at W9 · **Blocks:** W8, W10 · **Measured 2026-08-24** — grading the $t=20$ far wake against the classical 2-D reference rather than a 3-D corridor, the frozen checkpoint retains a mean $0.26\times$ of the reference's centreline deficit (and only $0.12\times$ of it by $x=16$); the solver path retains $0.62\times$. So the checkpoint destroys roughly three quarters of the wake before the second turbine. The earlier '$\sim 20\times$ too fast' was a per-call half-life estimate; this is the end-to-end consequence, and the two are consistent rather than competing. See [[results-w6-w11-wind-farm]].

W0 measured a wake half-life of $1.75$ convective times for Poseidon-T and rejected Poseidon-B for being worse. W9 confirms the consequence in the coupled system: at sixteen diameters the 2D baseline retains a $28.3\%$ deficit — in genuine 2D wakes barely recover, because there is no three-dimensional entrainment to refill them — while the coupled system retains $1.6\%$.

This is a property of the **checkpoint**, not of the composition, and no coupling arrangement fixes it. The options are a different expert, a coarser coupling step (dissipation is per *call*, which is why $\Delta t=0.25$ was adopted), or reporting W8/W10 as bounded by expert quality. It belongs here because the choice among those is a design decision and [[expert-library-atlas-0.1]] is where it should eventually be made.

**Downgraded 2026-08-23 by OP-5.** This is real but it is a *minority* effect. Replacing the checkpoint with an exact solver — same windows, same everything else — moves $P_2/P_1$ only from $1.002$ to $0.889$ against a correct $0.404$: **$15\%$ of the error.** The windowed architecture carries $66\%$. So the expert swap this entry contemplates is worth doing and is not the leverage, and OP-5 should be attacked first. **Read OP-5's 2026-08-23 qualification before quoting the $15\%$ or the $66\%$** — both are shares of a gap whose target moves with the baseline's viscosity, and the windowing share has not yet had a null control.

---

# OP-5. The windowed architecture is the dominant error source, not the expert

**Status:** open, **and it is now the top priority** · **Found at:** the 2026-08-23 composition-layer harness · **Blocks:** W8, W10, and the value of any expert swap · **Qualified 2026-08-23** — the *ranking* stands; the $66\%$ does not yet stand as a measurement of windowing alone. See "The $66\%$ is a subtraction bucket" below, and **run the null control before the three-way sweep.**

## The measurement

A frozen operator with a fixed input shape cannot see the whole domain, so the domain is tiled — here into 124 periodic $2\,D$ windows, stepped at $\Delta t = 0.25$. That is not a neutral implementation detail. Running the full $2\times2$ of {assembly rule} $\times$ {expert} to $t=20$, against a `ChannelNS` solve of the undivided domain:

| configuration | $P_2/P_1$ |
|---|---|
| frozen checkpoint, nearest-tile assembly | $1.143$ |
| frozen checkpoint, partition-of-unity assembly | $1.002$ |
| **exact solver**, partition-of-unity assembly | $0.889$ |
| **exact solver, no agents at all**, still windowed | $0.907$ |
| **`ChannelNS`, undivided** | $\mathbf{0.404}$ |

Of the $0.739$ gap between the starting point and the correct answer:

* assembly rule — $0.141$, **$19\%$**
* checkpoint — $0.113$, **$15\%$**
* **the windowed architecture — $0.485$, $66\%$**

**With a perfect solver inside every window and no agent decomposition whatsoever, $P_2/P_1$ is still wrong by a factor of $2.2$.** Nothing learned is in that loop.

## Why this matters more than the checkpoint question

It reverses the working assumption. [[wind-farm-implementation-log]]'s 2026-08-22 entry read $B-C = 1.762$ as "expert error" and concluded the checkpoint dominates — but $B-C$ compares a *tiled, periodic-windowed* run against an undivided solver, so it conflated the two and charged the whole difference to the expert. Separated, the larger share belongs to Atlas.

**Consequence for OP-3 and for [[expert-library-atlas-0.1]]:** swapping the checkpoint buys at most $\sim15\%$ of the error. A better expert is worth having and is *not the leverage*.

## The $66\%$ is a subtraction bucket — qualification added 2026-08-23

The entry above was written the same day it was measured, and it repeats, in a smaller way, the structure of the error it corrects. It is kept as written because the correction is the reusable part.

**The third factor was never varied at any level.** The $2\times2$ crossed {assembly rule} $\times$ {expert}. Every one of its four cells is windowed — `SolverExpert` was built holding *"same tiling, same window size, same per-window periodicity"* fixed, deliberately and correctly, so that the expert swap would be attributable. The consequence is that **there is no run in the table with windowing off and the rest of the coupled path on.** The only windowing-off leg is `ChannelNS`, a different code path with a different solver, different boundary conditions, a different timestep and a different nondimensionalization.

So "the windowed architecture — $66\%$" is, strictly, *everything that differs between `SolverExpert`-in-124-windows and `ChannelNS`-undivided*, plus every interaction term the additive split has nowhere else to put. That is the same shape as reading $B-C$ as expert error: one leg of the comparison still differs in more than one way.

**Three things are inside the bucket that are not windowing.**

**(a) The target is not a fixed number.** [[wind-farm-implementation-log]]'s 2026-08-22 baseline table sweeps $\nu$, $dx$ and strip thickness and concludes the run is *"insensitive to everything it should be insensitive to."* That is true of the **induction** column — $5.7\%$, flat to $0.2$ points, exactly as a potential-flow quantity should be. It is **not** true of the $P_2/P_1$ column in the same table:

| run | $\nu$ | $dx$ | strip | $P_2/P_1$ |
|---|---|---|---|---|
| coarse | $0.03$ | $0.0625$ | $0.25$ | $0.403$ |
| half $\nu$ | $0.015$ | $0.0625$ | $0.25$ | $\mathbf{0.246}$ |
| fine grid | $0.03$ | $0.03125$ | $0.25$ | $0.421$ |
| thin strip | $0.03$ | $0.0625$ | $0.10$ | $0.429$ |

The baseline's own parameter spread is $0.183$ — about $25\%$ of the $0.739$ gap being decomposed. The verdict was read off the flat column and applied to the sensitive one. **Array efficiency is a wake-*survival* measure and has no business being $\nu$-insensitive**; upstream induction is a blockage measure and has every business being so. One sweep, two quantities, opposite expectations.

**(b) The two legs may not be at the same Reynolds number — check this first.** $\nu=0.03$ at $D=U_\infty=1$ is $\mathrm{Re}_D\approx33$. [[spec-wind-farm-wake-atlas-0.1]] §2.3 declares $\mathrm{Re}_{\text{eff}}\in[10^3,10^4]$ and W0's analytic wake-retention check used $\nu=10^{-3}$ — a gap of roughly $30\times$. And $\nu=0.03$ at $dx=0.0625$ gives a cell Reynolds number $U\,dx/\nu = 2.08$, sitting on the classical central-difference limit, which suggests $\nu$ was **set by the grid rather than chosen as physics**. The "fine grid" run then halved $dx$ while holding $\nu$ fixed, so it never spent the resolution it bought.

What `SpectralNS` uses for $\nu_p$ inside a window, and whether it maps through the $L=2\,D,\;U_s=4\,U_\infty,\;T_s=0.5$ scaling to the same *physical* $\mathrm{Re}_{\text{eff}}$ as `ChannelNS`, is not recorded anywhere in this case study. **If those two viscosities disagree, a real share of the $66\%$ is a Reynolds mismatch and not an architectural cost at all.** The sign is consistent with it: the coupled system behaves like an over-viscous one, and more viscosity raises $P_2/P_1$ — the direction of the defect.

### Answered 2026-08-23, and the mismatch is real

Read off the source: `CoupledSystem.__init__` sets $\nu = 1/255$, so every coupled run — frozen or solver — is at $\mathrm{Re}_D = 255$. The $0.404$ target was produced at $\nu = 0.03$, i.e. $\mathrm{Re}_D = 33$. **The two legs differ by $7.7\times$ in Reynolds number.**

`scripts/windfarm_reynolds_sweep.py` re-runs the baseline with $dx$ refined in step, holding the cell Reynolds number at $2.08$ so each rung is resolved to the same standard rather than on the same grid:

| $\nu$ | $\mathrm{Re}_D$ | grid | $\langle U_d\rangle$ | $P_2/P_1$ |
|---|---|---|---|---|
| $0.03$ | $33$ | $128\times384$ | $(0.7894,\,0.5837)$ | $\mathbf{0.4044}$ |
| $0.015$ | $67$ | $256\times768$ | $(0.7733,\,0.4975)$ | $\mathbf{0.2663}$ |

The first rung reproduces the published number **and its disk velocities** exactly, which is what makes the second trustworthy. **Doubling the Reynolds number moves the target by $34\%$** — and the coupled system sits nearly four times further along that axis again.

**So the $0.739$ gap the shares are computed from is measured against the wrong number.** At the coupled system's own Reynolds number the correct answer is materially below $0.404$, the gap is *larger* than $0.739$, and every share in the decomposition moves. The direction is uncomfortable for the framework rather than flattering: the windowed architecture's absolute error grows.

**Caveat, stated because it bounds the claim.** Both rungs are read at $t=20$ to match the coupled runs, and neither is steady — centreline drift is $2.3\times10^{-2}$ and $3.4\times10^{-2}$, against the $3\times10^{-5}$ the 2026-08-22 entry reports for a longer run. These are like-for-like against the coupled numbers *at the same instant*, and are **not** steady-state answers. The two remaining rungs ($\mathrm{Re}_D=133$ and $255$) were not run: at cell Reynolds $2.08$ they need $512\times1536$ and $1024\times3072$ grids, and the machine was needed for the rollouts.

**(c) The split is additive on a cubed quantity.** $P_2/P_1=(U_2/U_1)^3$ exactly, at every row of the table: $(0.6798/0.6502)^3=1.1429$, $(0.5837/0.7894)^3=0.4042$. Attributing shares linearly in $P$ is therefore not metric-invariant. In velocity space the same three runs give

| | share in $P$ | share in $U$ |
|---|---|---|
| assembly rule | $19\%$ | $15\%$ |
| checkpoint | $15\%$ | $13\%$ |
| **windowing** | $\mathbf{66\%}$ | $\mathbf{73\%}$ |

The ranking is unchanged and the qualitative finding survives — which is the point of checking. But **$66\%$ should not be quoted to two figures as though it were a property of the system rather than of the metric.**

## What has not been separated, and how to separate it

**Run the null control before the three-way sweep.** What the $2\times2$ lacks is a **level-off condition for the windowing factor** — a run where windowing is off and everything else is the coupled path. Sweep window count through the *identical* harness with `SolverExpert`:

$$N \;=\; 124 \;\to\; \sim31 \;\to\; \sim8 \;\to\; 1,\qquad \nu_p \text{ rescaled at each level to hold } \mathrm{Re}_{\text{eff}}=\frac{T_s}{\nu_p L^2}\ \text{fixed}$$

Holding $\mathrm{Re}_{\text{eff}}$ constant across window sizes is the entire reason to own a solver expert: window size is not a free parameter for the frozen checkpoint (W0/§4.3 — doubling $L$ *quarters* $\mathrm{Re}_{\text{eff}}$), but for a solver $\nu_p$ is an explicit dial that cancels it. **This turns the case study's most-quoted confound into a controlled variable.**

Read it as follows. If $P_2/P_1\to0.404$ monotonically as $N\to1$, windowing is confirmed *and* quantified as a function of seam count in a single sweep, and ranking the sub-candidates below is worth doing. If it plateaus well short — say at $0.85$ — then most of the $66\%$ is harness, $\nu$ or BC difference, and ranking sub-effects inside a mislabeled bucket would be a control-shaped mistake of the same family as the first five, rather than their cure.

**Then the three candidates**, each varyable with a perfect interior operator so nothing is confounded by the checkpoint:

1. **Per-window periodicity.** Each window is solved as if it wrapped. Vary by giving `SolverExpert` a non-periodic window solve (the BC machinery of `pressure.py` already exists) and re-measuring. **Argued 2026-08-23 to be a *precondition* of (3) rather than a peer of it** — see [[schwarz-iteration-atlas-0.1]] §5.3: iterating an interface exchange converges to the fixed point of whatever the local solve is, and a periodic window is not a faithful restriction of an open channel. Do this one first.
2. **The macro-step.** $\Delta t = 0.25$ against a CFL-stable $2.03\times10^{-3}$ at the coupled system's own $dx = 0.0156\,D$ — a factor of **123**, not the 40 first written here, which compared against a coarser baseline grid. Vary directly; note the frozen expert *cannot* go below its native lead, but `SolverExpert` can, which is exactly the kind of question the harness was built to answer.
3. **One-pass exchange.** Each macro-step does a single sweep plus a scalar fixed point on thrust; classical overlapping Schwarz iterates the *field* exchange to convergence. Vary by iterating the sweep. **Planned in full at [[schwarz-iteration-atlas-0.1]]**, which also predicts this term is *small*: every transport mechanism in the build travels less than the $0.5\,D$ tile overlap in one macro-step, and the one mechanism that would defeat one-pass exchange — elliptic pressure coupling — is already solved globally. Note also that the existing thrust fixed point already re-runs the sweep $\sim3\times$ per step, so what is missing may be the *convergence criterion* rather than the machinery.

**(2) and (3) may be one knob, not two.** If `SolverExpert` substeps internally to CFL, then $\Delta t=0.25$ is not an integration step at all — it is the **exchange interval**, i.e. how long a window is allowed to run on stale boundary data. Under that reading (2) and (3) are the same factor measured two ways, and only the *frozen* expert genuinely conflates them, because its native lead forces the integration step and the exchange interval to be equal. **Settle whether the solver substeps before designing a three-legged sweep that only has two legs.** If it does, the sharper statement available is that a frozen expert's native $\Delta t$ imposes an exchange interval it cannot refine — a constraint on *coupling*, not on *integration*.

**[AI Inference]:** conditional on the null control landing near $0.404$, (2)/(3) is both the most likely single culprit and the most likely structural one. A factor of 123 in step size is enormous for an explicitly-coupled scheme, and [[prior-art-and-novelty-atlas-0.1]] §2 already names the relevant gap — *partitioned-coupling stability theory assumes error shrinks with the macro step, which fails for an expert trained at a native $\Delta t$*. If it dominates, the finding is that **a frozen expert's native $\Delta t$ sets a floor on composition accuracy, not just on stability**, which is a sharper and more general statement than the case study has yet made and would belong in [[expert-library-atlas-0.1]] as a selection criterion.

## Candidate fixes, none yet attempted

* **Re-baseline at the spec's Reynolds number** — cheapest of all and a prerequisite to the rest, because it decides what "correct" *is*. Refine `ChannelNS`'s grid *and* lower $\nu$ together toward $\mathrm{Re}_{\text{eff}}\in[10^3,10^4]$, keeping the cell Reynolds number in range, and re-read $P_2/P_1$. W10's $[0.4,0.8]$ band was vindicated against $\nu=0.03$; it has **not** been vindicated at the Reynolds number the spec declares.
* **Larger windows** — fewer, bigger patches reduce seam count, but $\mathrm{Re}_{\text{eff}} = T_s/(\nu_p L^2)$ means doubling the physical size *quarters* the effective Reynolds number (W0/§4.3). Window size is not a free parameter; this trades one error for another and the trade must be measured. **For `SolverExpert` the trade is escapable** — rescale $\nu_p$ — and for the frozen checkpoint it is not. That asymmetry is itself a result about frozen experts.
* **Schwarz iteration to convergence** rather than one pass — classical, well-understood, and costs expert calls linearly. This is the fix available to *both* expert kinds, which makes it the one that generalizes. **Planned at [[schwarz-iteration-atlas-0.1]]** in four stages, of which **S0 is the cheapest decisive measurement available to OP-5**: instrument the interface residual against the *existing* loop and change nothing. If it has already reached the expert's per-call noise floor by iteration 3, iterating cannot help and the rest is not worth building. The same page argues the fix is an *instrument* for candidate (1) rather than a fix in its own right.
* **A finer macro-step** — available to a solver, and *not* available to the frozen expert below its native lead, which is why this is a framework/expert co-design question rather than a tuning knob.

---

# OP-6. The frozen expert is not mirror-equivariant — gate W7 is unwinnable with this checkpoint

**Status:** **RESOLVED 2026-08-23** — diagnosed and fixed the same day; the fix is [[symmetry-averaging-atlas-0.1]], and gate W7 now passes. Kept here in full because the *diagnosis* is the reusable part and because the fix is a framework capability rather than a patch. Originally: open, found 2026-08-23 by direct operator test · **Found at:** the W7 narrowing · **Blocks:** W7 outright; implicated in OP-2

## The measurement

Gate W7 asks for a mirror residual below $10^{-6}$ under symmetric inflow. That is a property of the **operator** before it is a property of the composition: if $E(\mathcal M\mathbf u)\ne\mathcal M E(\mathbf u)$, no arrangement of agents, tiles, ports, blending or Schwarz iteration can produce a symmetric field.

Under the $y$-mirror $\mathcal M[u](x,y)=u(x,-y)$, $\mathcal M[v](x,y)=-v(x,-y)$, on an **exactly** mirror-symmetric input, one call at $\Delta t=0.25$:

| symmetric input | $\max\lvert u-\mathcal M u\rvert $ | $\max\lvert v-\mathcal M v\rvert $ |
|---|---|---|
| uniform inflow $u\equiv1$ | $2.42\times10^{-2}$ | $3.59\times10^{-2}$ |
| wake deficit | $3.16\times10^{-2}$ | $3.58\times10^{-2}$ |
| symmetric jet | $2.79\times10^{-2}$ | $3.46\times10^{-2}$ |
| symmetric counter-rotating $v$ | $2.58\times10^{-2}$ | $3.68\times10^{-2}$ |

**A uniform inflow — the most trivial symmetric field there is — comes back asymmetric at $2.4\times10^{-2}$, which is $2.4\times10^{4}$ times the gate, in a single call.** Note this is the same field W4 uses as its null control and the same field W0 certified the checkpoint on: the *mean* is held to $10^{-16}$ and the *symmetry* is not. Neither gate looked.

## It is systematic, not noise, and it accumulates

Two calls on identical input agree to $\max|u_a-u_b| = 0$ exactly — the checkpoint is **deterministic**, so this is a fixed property of the operator, not per-call jitter. On a closed window it grows monotonically:

| calls | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| $\max\lvert u-\mathcal M u\rvert $ | $3.16\times10^{-2}$ | $4.69\times10^{-2}$ | $7.12\times10^{-2}$ | $1.02\times10^{-1}$ |

which brackets W7's measured $2.77\times10^{-2}$ at $t=5$ rising to $7.42\times10^{-2}$ at $t=10$.

## The positive control that makes the convention trustworthy

`SolverExpert` (`SpectralNS`), run through the **identical** harness, the identical mirror convention and the identical input, gives $1.06\times10^{-15}$ relative in $u$ and $1.17\times10^{-17}$ in $v$. So the test is not measuring a sign error in my own reflection operator — a wrong convention would have failed the solver too.

## The mechanism is grid-locked, not flow-locked

Taking the antisymmetric part of the output on uniform inflow, averaging over $x$, and transforming in $y$ (128 cells):

| wavenumber $k$ | 4 | 8 | 16 |
|---|---|---|---|
| wavelength | $32$ cells | $16$ cells | $8$ cells |
| amplitude | $6.00\times10^{-1}$ | $5.19\times10^{-1}$ | $1.41\times10^{-1}$ |

A comb at $32/16/8$ cells, on an input with **no structure at all**. The checkpoint's `patch_size` is $4$ and its `window_size` is $16$ patches $=64$ cells, with a shifted-window stride of $8$ patches $=32$ cells. The dominant wavelength is exactly that stride and the rest are its harmonics.

**[AI Inference]:** this is the shifted-window attention of SwinV2. A patch grid maps to itself under reflection, but the *shifted* partition does not — the shift is applied from one corner, so the mirrored field is partitioned differently and attends over different neighbourhoods. If so, the property is architectural and shared by **every** Swin-based neural operator, which would make it an [[expert-library-atlas-0.1]] selection criterion rather than a fact about this checkpoint. Isolating it would mean disabling the shift and re-running this test — cheap, and not yet done.

## What this corrects

The 2026-08-23 gate table attributed W7 to *"path-dependent message passing around the graph cycle, exactly what [[impl-wind-farm-guide]] §10 predicts."* The guide predicted it, the magnitude matched, and it read as confirmed. **It is wrong.** The operator produces asymmetry of the same order with no graph, no ports, no agents and no tiling — one call on a uniform field. Sweep-order path dependence may also be present; it is not needed to explain W7 and has not been shown to contribute at all.

**Fifth time in this case study a control has overturned a conclusion rather than confirming it** — and the first time the overturned conclusion was one the guide itself predicted in advance.

## What would count as a fix

1. **Symmetrize the operator at composition level:** $\tilde E(\mathbf u) = \tfrac12\bigl[E(\mathbf u)+\mathcal M^{-1}E(\mathcal M\mathbf u)\bigr]$. Costs exactly $2\times$ the expert calls, trains nothing, and makes W7 pass **by construction** on any mirror-symmetric configuration. It is a *framework* fix to an *expert* defect, which is precisely the kind of thing Atlas exists to test — and it generalizes: the same average over any symmetry group the problem declares.
2. **A different checkpoint**, if a non-Swin operator turns out to be equivariant. Unknown, and testable in minutes by this same script.
3. **Restate W7** as a bound on symmetry *error growth* rather than symmetry, reporting the operator's own equivariance defect as a separate measured quantity.

**[AI Inference]:** (1) is the one to build. It is a concrete instance of a general capability — *the composition layer restoring a symmetry the expert lacks* — which is a stronger claim for Atlas than any gate passing, because it is something the frozen expert provably cannot do alone. It also connects directly to [[conservation-as-constraint-atlas-0.1]]: Noether ties symmetries to conserved quantities, so an operator that breaks a discrete symmetry is the same class of defect as one that breaks a conservation law, and the enforce-or-measure rule should cover both.

## Resolution

Option (1) was built: `src/atlas/invariants/symmetry.py`, the Reynolds average over a declared finite group, wired in as `CoupleConfig.symmetry="mirror-y"`. Full treatment in [[symmetry-averaging-atlas-0.1]].

| | raw | symmetrized |
|---|---|---|
| symmetric input, one call | $3.59\times10^{-2}$ | $\mathbf{0}$ (bitwise) |
| general equivariance | $3.73\times10^{-2}$ | $\mathbf{0}$ |
| accumulated over 8 calls | $1.02\times10^{-1}$ | $\mathbf{0}$ |
| **gate W7, coupled system, $t=5$** | $2.77\times10^{-2}$, growing | $\mathbf{2.90\times10^{-7}}$, **flat**, PASS |

Cost $1.7\times$ on the macro-step. Mean and divergence untouched, because averaging is linear.

The coupled residual is $\sim3\times10^{-7}$ rather than $0$, and the cause was traced rather than tolerated: it enters at the **first scatter**, and the expert's float32 forward pass depends on a window's *position in the batch* — the same window at batch sizes $8$ and $40$ differs by $9.6\times10^{-7}$ to $1.5\times10^{-6}$, about $11\varepsilon_{32}$, while an identical batch repeats bitwise. Mirror-paired tiles sit in different batch slots. **The construction is exact; the floor is arithmetic** — and W7's $10^{-6}$ threshold is only $\sim2\times$ above the raw lattice asymmetry, so the gate is graded near the precision of the expert it grades.

**What remains open from this entry** is the mechanism attribution — the shifted-window hypothesis is still an inference, testable by disabling the shift, and it matters for [[expert-library-atlas-0.1]] because it decides whether this is a Poseidon fact or a Swin fact.

## Where the details live

[[symmetry-averaging-atlas-0.1]] (the concept and the proof), [[wind-farm-implementation-log]] entries 2026-08-23 (operator equivariance; group average). Code: `atlas/invariants/symmetry.py`, `tests/atlas/invariants/test_symmetry.py` (15 tests).

---

# OP-4. Gate W11's power residual is not measurable with an LES-like expert

**Status:** **CLOSED 2026-08-24 as not measurable** — the disposition, not a deferral · **Found at:** wind-farm W6 experiments, 2026-08-22 · **Blocks:** W11 (now closed with it) — the residual's dissipation term needs a single $\nu$, and W0 established that $81\%$ of the field's dissipation lives below $0.125,D$, exactly the scale where no single $\nu$ reproduces the checkpoint. Per [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule the honest disposition is to close the gate and say so rather than report a number produced by choosing a $\nu$. **No power-residual figure is quoted in [[results-w6-w11-wind-farm]].**

W11 asks for $|\mathcal R(t)| < 1\%$ of extracted power. Measured on the coupled system at $t=10$: $|\mathcal R| = 1.61$ against an extracted power of $0.67$ — off by a factor of $240$ rather than by a few percent.

**The instrument was checked first, and it is sound.** `CoupledField` reproduces W4's validated identity — the balance with $P_{\text{ext}}$ omitted equals the extracted power — to $10^{-12}$, and on smooth analytic fields it recovers the outflux term to $2\times10^{-4}$ and the dissipation term to $3\times10^{-5}$ relative. So the failure belongs to the field, not the adapter.

**Where it comes from.** The ledger's dissipation term is $\nu\int|\nabla\mathbf u|^2$, and that integral is dominated by the smallest resolved scales. Progressively smoothing the coupled field and recomputing:

| smoothing radius | $0$ | $1$ cell | $2$ | $4$ | $8$ ($0.125\,D$) |
|---|---|---|---|---|---|
| dissipation, % of raw | $100$ | $74$ | $58$ | $39$ | $\mathbf{19}$ |

**$81\%$ of the dissipation lives below $0.125\,D$** — which is exactly the wavelength W0 identified as this checkpoint's dissipation cutoff ($\nu = 4.9\times10^{-4}$, $r^2 = 0.998$, with *no* measurable decay above it). So the ledger is integrating $\nu|\nabla\mathbf u|^2$ with a single nominal $\nu = 1/255$ precisely over the band where W0 established that a single $\nu$ is meaningless.

**This is W0's finding arriving in the power ledger.** The log's own words: *"the checkpoint behaves like an LES with a spectral cutoff, not a Newtonian fluid. Any $\mathrm{Re}_{\text{eff}}$ quoted in this case study describes that cutoff."* A power balance needs a dissipation term; a dissipation term needs a viscosity; this expert does not have one.

## What would count as a fix

Three options, and the choice is a design decision rather than a calculation:

1. **Measure the expert's dissipation instead of assuming it** — apply the expert to a field and compute the energy it actually removed per call, as an operator property, then use that in the ledger in place of $\nu\int|\nabla\mathbf u|^2$. Closest to honest, and it makes $\mathcal R$ a statement about composition rather than about viscosity.
2. **Band-limit the ledger** — evaluate the balance on the field low-passed at the cutoff, where the expert *does* behave like a resolved solver. Changes what W11 claims: conservation of the resolved scales, which is what an LES reports anyway.
3. **Restate W11** as a bound on the terms the framework controls (interface fluxes and $P_{\text{ext}}$) and drop dissipation from the gate, reporting it separately as a checkpoint property.

**[AI Inference]:** option 1 generalizes and the other two do not. Any frozen expert has *some* dissipation, and none of them come with a documented $\nu$ — so a power ledger that measures its experts' dissipation rather than assuming a fluid property is what [[expert-library-atlas-0.1]] would need for every later case study, not just this one. If so, [[port-algebra-atlas-0.1]]'s global residual $\mathcal R(t)$ needs a per-expert dissipation term in its definition, which is a change to the framework rather than to the wind farm.

## Why deferring is defensible

The *other* four gates run from the same rollout and are readable: W6 passes at exactly $16/27$, and W7, W8 and W10 fail with attributable causes. W11 is the one gate whose failure is an artefact of the gate's own definition meeting this checkpoint's nature, and reporting it as a conservation failure of the framework would be wrong.

---

## See Also

- [[composition-error-theory]] — the theory these sit inside: OP-2 as a passivity defect, OP-4's missing $\nu$ as a storage function, OP-5's subtraction buckets as an adjoint attribution, OP-1's conditioning as an interface inf-sup
- [[conservation-as-constraint-atlas-0.1]] — the enforce-or-measure rule OP-1 is deferred under
- [[wind-farm-implementation-log]] — every number here in the context it was measured in
- [[impl-wind-farm-guide]] — the gates these problems block
- [[f1-pathmap-and-end-goal]] — F2 (composability without fine-tuning) is what all three bear on
- [[expert-library-atlas-0.1]] — where OP-3's checkpoint decision belongs
