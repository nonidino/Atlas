# Ground Effect — a moving interface, and the first design parameter

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/case-study-ground-effect/`)
**Status:** opened 2026-09-03. The eleventh real case study, and the first in this vault whose interface geometry is a function of the solution. Everything here is quoted from `out/w127/w127.json`; nothing is estimated.
**Code:** `atlas/cases/ground_effect.py` (the declarations, the wing, the spring and the three couplings), `scripts/w127_ground_effect.py` (the driver, nine stages), `tests/test_tier27_ground_effect.py` (47 tests)
**Related:** [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[gap-worklist]] · [[end-to-end-architecture-spec]] · [[probed-dtn-coupling]] · [[master-error-bound]] · [[port-algebra-atlas-0.1]] · [[case-study-thermal-strain-atlas-0.1]] · [[case-study-seam-placement-atlas-0.1]] · [[poc1-results-differentiable-design]] · [[tier0-measurements]]

---

## 1. The question, and why it is Phase B's second row

[[case-study-ladder-to-f1]] §3 tabulates the four coupling kinds an F1 car is made of and finds that **one of them has no bond at all**:

> | **moving / deforming interface** | ride height under aero load; a deflecting wing; a rotating wheel | **no** — `motion_class` is `static` and anything else is refused | **W22 / W30** |

[[f1-pathmap-and-end-goal]] §1.1 names that coupling as the one that *is* the physics — *"aero load changes ride height, ride height changes aero"* — and every interface built in this vault before today declares `motion_class=STATIC`. §4's CS-10 row asks for three things and this page reports all three:

1. **`InterfaceMotion` stops being a named hole with a declared interface and becomes a measured one.** `holes.INTERFACE_MOTION` has asked for three numbers since it was written — a re-probe count, the operator drift $\lVert S(t+K\Delta t)-S(t)\rVert$, and the fraction of the power residual unaccounted at a sweeping seam — and had **none** of them. W30 emitted the drift on *static* runs in Tier 0 precisely so a moving one would have something to be compared against.
2. **W97 has to be closed here or admitted permanently.** A field-to-lumped `MECH` seam is one-sided by the cell Reynolds number, and if its substitution certificate is blind then *no lumped subsystem in the car can be certified*. The candidate repair is [[probed-dtn-coupling]] §4.1's **conservative co-normal**, which **W47 scoped out** at fluid–fluid seams because the advective term cancels there.
3. **$h_0$ is a genuine design parameter and not a state**, so this is **F5**'s honest test: is $\mathrm{d}(\text{downforce})/\mathrm{d}h_0$ through the composed stack physics, or an artefact?

---

## 2. How it is built

A 2-D wing section over a **moving floor** at ride height $h$. Six `reference.WindowNS` windows tile a $208\times144$ domain at $\mathrm{d}x = 1/64$; the flow they produce carries an integrated load $L$ across a **field-to-lumped `MECH` seam** to a lumped suspension whose whole content is

$$k\,(h_0 - h) \;=\; L(h),$$

two lines of algebra with zero fitted parameters, the `disk.ActuatorDisk` pattern exactly. The suspension's solution moves the interface every macro-step.

| | |
|---|---|
| domain | $208\times144$ cells $= 3.25 \times 2.25$, $\mathrm{d}x = 1/64$ |
| windows | six, $80\times80$, halo $16$ against a required $8$ (a factor of $2$) |
| partition of unity | ramp $8$, identity residual $1.11\times10^{-16}$, $\chi_{\min} = 2.07\times10^{-3}$, $\Pi = 0.725$ |
| cell Reynolds number | $3.906$, just inside the $4$ `WindowNS`'s own docstring names |
| chord Reynolds number | $125$ |
| macro-step | $0.0125$, with **four** composition-layer exchanges inside it |
| plate | chord $0.5$ ($32$ cells) at $20^\circ$, cells $[88, 119)$ |

**The agents are the exposed ones and the assembly is a `ProjectedAssembly`.** [[case-study-scaling-ladder-atlas-0.1]] §8's advice, taken: `_project` is replaced by the identity inside every window and the composition layer applies **one** global spectral Leray projection to the assembled field, once per exchange. That arrangement — `wake_array.exposed_reference_solver` plus one projection after the blend — is the one CS-7 selected by measurement and the one `L2/R10` has prescribed since Tier 0; this is the third case study to build on it, and the graph compiles at `admit-uncertified` with **zero refusals** when the interface is held still.

**And the cuts are placed clear of the wing, before the fact.** [[case-study-seam-placement-atlas-0.1]]'s **W124** found that every x-seam of CS-6 and CS-7 passes exactly through a rotor disc, because `wake_array.Rotor` is declared on an x-*adjacency* and the tiling rule cannot express anything else. Here the plate spans cells $[88,119)$, the x-overlap bands are $[64,80)$ and $[128,144)$, and `GroundTiling.cuts_clear_of_wing()` returns **8 cells** — asserted by a test rather than noticed later.

### 2.1 The wing, stated as the model it is

A **porous inclined plate**: `disk.ActuatorDisk`'s quadratic momentum sink with the disk's axis replaced by a plate normal $\mathbf n$. Per chordwise station $k$,

$$w_k \;=\; \tfrac12\big(\bar{\mathbf u}^{+}_k + \bar{\mathbf u}^{-}_k\big)\!\cdot\!\mathbf n \;-\; v_{\text{plate}}\,n_y, \qquad \mathbf F_k \;=\; \tfrac12\rho\,C_N\,\frac{c}{S}\,\lvert w_k\rvert\,w_k\,\mathbf n,$$

with $C_N = 20$. It is **not** thin-aerofoil theory: the normal force goes as $\sin^2\alpha$ rather than $\sin\alpha$, there is no Kutta condition and no bound circulation. What it does have is *thickness* — it blocks the gap, which is the mechanism ground effect runs on — and a load that is an **exact discrete momentum exchange** rather than a second model: the station kernels are normalized discretely on their stamping box, so the force on the fluid integrates to minus the force on the plate to floating point (asserted; `disk.body_force_field`'s own gate-W2 property).

**The two sides are averaged and that is load-bearing.** A station sampled inside its own smearing reads its own induction and an actuator line that does so drives itself; the self-induced velocity of a sheet is antisymmetric across it, so the mean of the two sides is the external flow at leading order.

**$C_N = 20$ is the one number here chosen to make the coupling measurable rather than derived, and it is declared as such.** At the textbook flat plate's $C_N = 2$ the load rises $8.5\%$ from $h = 0.8$ to $h = 0.09$ and the feedback ratio is $0.02$ — a moving interface with no coupling to measure. At $20$ the load rises $32\%$ and $u_{\max}$ reaches $1.35$. It is bounded above by the explicit stability of its own quadratic sink, $\Delta t\,C_N\lvert w\rvert/(\sigma_n\,\mathrm{d}x) = 0.8$ at $20$ and $1$ at $25$.

### 2.2 The ground-effect curve, measured and not assumed

Held at a fixed floor and marched $40$ macro-steps per height:

| $h$ | $0.075$ | $0.09$ | $0.11$ | $0.14$ | $0.18$ | $0.23$ | $0.30$ | $0.40$ | $0.55$ | $0.75$ |
|---|---|---|---|---|---|---|---|---|---|---|
| $L$ | $0.2810$ | $0.2752$ | $0.2683$ | $0.2595$ | $0.2497$ | $0.2401$ | $0.2304$ | $0.2222$ | $0.2157$ | $0.2123$ |

**Monotone, and no peak.** $L(0.075)/L(0.75) = \mathbf{1.324}$. That is the *inviscid blockage* branch of ground effect and it is all this construction can see: the floor is a **rolling road** at $(U_\infty, 0)$, so there is no floor boundary layer at all — which is why Formula 1 uses one, and which removes the viscous gap choke that ends the real downforce curve. The turnover was **looked for** at every height down to $4.8$ cells of gap and is not there.

**And the unsteadiness is the level everything else is quoted against.** The plate sheds; over the last quarter of each march the settled load moves by a median of $\mathbf{1.256\%}$ peak-to-peak. **W106**'s discipline, taken seriously: the pipeline's bitwise floor is exactly zero and bounds nothing, so no difference below is quoted against it.

### 2.3 The two aero stiffnesses, and why the obvious update diverges

The quasi-static stiffness $\lvert\mathrm dL/\mathrm dh\rvert$ from the table above reaches $0.389$. Measured on **one frozen field** — moving the plate on a field settled at another height, which is what one exchange interval actually does — it reaches $\mathbf{1.833}$.

> **A designer computing the aero stiffness quasi-statically understates what one exchange interval sees by $4.71\times$.**

That is a standing fact about this seam. It is *not*, at this operating point, what makes the obvious scheme fail: $1.833$ is below the spring rate $k = 2.5$ at every height. What fails is the reading of $k(h_0-h) = L$ as an **update**:

$$h^{n+1} \;=\; h_0 - L^n/k .$$

A two-line algebraic suspension has no mass and no damper, so it moves the whole way to the new equilibrium inside one macro-step. From $h = 0.30$ that implies a plate velocity of $\mathbf{-5.77\,U_\infty}$, and the fluid's own response to a plate moving at $5.77$ times the freestream is a load of order $17$ against the $0.25$ the spring was balancing. **It diverges at macro-step 1**, at $h = 14.39$. Under-relaxation does not help and cannot: where the map's derivative is positive and above one the fixed point is repelling, and $(1-\omega) + \omega g' > 1$ for every $\omega > 0$.

**What is well posed is the port's own condition**, and writing it in the port's own variables is the whole of the repair. With $v$ the plate's vertical velocity — which is exactly the `MECH` port's **flow** half — and $h^{n+1} = h + \Delta t\,v$,

$$\mathcal R(v) \;=\; \underbrace{k\,(h_0 - h - \Delta t\,v)}_{\text{\texttt{Suspension.respond}}} \;-\; \underbrace{L(u, h, v)}_{\text{the wing port's effort}} \;=\; 0,$$

which is the two agents' `boundary_response`s summed to zero on a massless plate. Its derivative is analytic and needs no second field evaluation, because the external flow at the stations does not depend on $v$:

$$\frac{\partial\mathcal R}{\partial v} \;=\; -k\,\Delta t \;-\; C_N\sum_k \lvert w_k\rvert\cos^2\!\alpha\,\mathrm ds .$$

Three Newton steps at a fixed count — fixed, because the tape has to have the same shape at every $h_0$ for a reverse-mode gradient to mean anything — and the residual comes back at $1.11\times10^{-16}$.

> **This is the partitioned-FSI added-mass instability, arriving unprompted at the first moving interface in the vault, and the port algebra's own variables are what dissolve it.** The lumped side supplies a force from a *position*; the port's flow half is a *velocity*; and the term that makes the interface problem non-empty is the aerodynamic damping the field expert reports back, $C_d = \partial L/\partial v \approx 3$.

### 2.4 The referent, and a zero-cut control that is bitwise

`GroundRollout` is one code path and the referent is a *tiling* of it. `GroundTiling.single()` puts one window over the whole domain: the partition of unity is identically $1$ (asserted with `atol = 0`), `cut` and `blend` are the identity **to the bit**, and the same global projection is applied to a field nothing was blended into. So the referent and the composed column differ by the **cut** and by nothing else, and CS-7 §2's $N=1$ rung — *the control that would have redirected the parent's whole search was the one nobody ran* — is available here by construction rather than by care.

Every coupled march is **released from a settled fixed-floor field**, never from a uniform stream: the impulsive-start load is $2.0\times$ the settled one and would put the plate through the floor at step $0$. `GroundRollout.run` raises there rather than clamping, because a hard stop is a fitted parameter and this expert has none. And the release height is **below** $h_0$ — above it a linear compression spring is in tension and `Suspension.validity` declines, which is the expert refusing correctly and is how the first version of this study was caught.

---

## 3. The gate: does the split reproduce the referent's load and ride-height history?

**Yes — and the answer reverses if you stop marching at the peak.**

Released from $h = 0.298$ against $h_0 = 0.32$, $k = 2.5$, over $240$ macro-steps. The referent falls to a **minimum of $0.2199$ at step $103$** and comes back to $0.2257$: it undershoots its own equilibrium and recovers, so a march stopped at the minimum reports a different answer.

| column | $h$ at $240$ | $\max\lvert\Delta h\rvert/h$ | at step | $\Delta h/h$ at the end | sign changes |
|---|---|---|---|---|---|
| referent (tight, one window) | $0.225688835$ | — | — | — | — |
| lag 1, no cut | $0.225719178$ | $2.536\times10^{-3}$ | $62$ | $+1.344\times10^{-4}$ | **2** |
| lag 2, no cut | $0.225763347$ | $6.068\times10^{-3}$ | $63$ | $+3.302\times10^{-4}$ | **2** |
| lag 4, no cut | $0.225866244$ | $1.360\times10^{-2}$ | $67$ | $+7.861\times10^{-4}$ | **2** |
| lag 1, six windows | $0.225725105$ | $2.526\times10^{-3}$ | $62$ | $+1.607\times10^{-4}$ | **2** |
| tight, six windows | $0.225694760$ | $2.626\times10^{-5}$ | $239$ | $+2.626\times10^{-5}$ | $0$ |

Three things, and the third is the one Tier 23's standing rule is about.

**The lag defect is first order in the lag.** $2.536\times10^{-3} \to 6.068\times10^{-3} \to 1.360\times10^{-2}$, $\log_2$ ratios $1.258$ and $1.165$ — first order with a $20\%$ excess, the same shape as [[case-study-thermal-strain-atlas-0.1]] §3's $\log_2$ ratios of $1.003$.

**The seam's lag dominates the domain cut by two orders.** The lag alone is worth $2.536\times10^{-3}$ and the cut alone $2.626\times10^{-5}$: a factor of **97**. Every window in the composed column is doing its job; what costs is exchanging with the lumped agent once a macro-step instead of every exchange. On this graph the field-to-lumped seam is $97\times$ the whole six-window decomposition.

**And the difference between the split and the referent changes sign when marched.** Every lagged column peaks at step $\sim62$, on the transient, and **ends on the other side of zero**. Read at step $62$ the split *undershoots* the referent's ride height; read at $240$ it *overshoots* it. Tier 23 adopted the rule after its third instance — *a comparison between two configurations is not a result until it has been marched past the point where the two curves could cross, and the crossing has to be looked for rather than assumed absent* — and this is the first case study to apply it as a gate rather than to discover it. The crossing was looked for, it is there, and both halves are reported.

**One thing this leaves open.** `tight_cut` — the cut alone — has its maximum at step $239$, the last one. The cut's own defect has **not peaked** at the horizon marched, so $2.626\times10^{-5}$ is a value at $240$ macro-steps and not a bound.

---

## 4. `InterfaceMotion`, measured

`holes.INTERFACE_MOTION` names three `measurements_required` and every compile in this vault has reported them **outstanding**. Here they are, at $K = 10$ macro-steps over which the interface travels $-0.0172 = -1.10$ cells.

### 4.1 Operator drift

$\lVert S(t + K\Delta t) - S(t)\rVert$, relative, on the **same seam** in a moving column and in a fixed-floor one released from the same field:

| seam | moving | static | ratio |
|---|---|---|---|
| **`wing`** (field ↔ lumped) | $\mathbf{3.77\times10^{-1}}$ | $2.51\times10^{-2}$ | $\mathbf{15.0\times}$ |
| `x00` (fluid ↔ fluid) | $1.63\times10^{-4}$ | $3.21\times10^{-5}$ | $5.1\times$ |

W30's own static reference, four `window_ns` seams at $K = 10$, is $2.4\times10^{-4}$ to $3.5\times10^{-3}$. So:

> **The fluid–fluid seam sits inside W30's static band whether the wing moves or not. The moving seam drifts $108$ to $1572$ times W30's static reference, and $15.0$ times the same seam held still.** The drift is a property of the *interface's motion*, not of the flow developing around it — which is exactly the separation the static number was emitted to make possible, and it took twenty-three tiers and one moving interface to use it.

### 4.2 The re-probe count, and the economics it prices

`INTERFACE_MOTION`'s own `must_satisfy` says *"the staleness predicate must be cheaper than the re-probe it gates, or the economics of probing collapse."* Priced:

At a $2\%$ staleness tolerance a cached $S$ survives **$1.00$ macro-steps** on the moving seam and $6.0$ on the static one — the moving seam's drift of $3.77\%$ **per macro-step** already exceeds the tolerance, so it is stale before the next exchange. Over the $240$-step march that is $240$ re-probes against $41$; one re-probe is $18$ solves, so **$4320$ extra window solves against the march's own $5760$ — $75.0\%$ of the whole run, for one seam of eight.**

> **The entire economic argument for probed-DtN coupling is a cached $S$, and at a solution-dependent interface the cache does not survive one exchange.** That is not a defect in the construction; it is the number `InterfaceMotion` was written to ask for, and it says what a moving seam costs.

### 4.3 The unaccounted power

The slot asks for *"the fraction of $\mathcal R(t)$ unaccounted at the moving seam"* — the interface does work on the region it sweeps and that term is absent from the residual. Measured at the wing at $K=10$: the port term $-L\,v_{\text{plate}} = +1.479\times10^{-2}$ against the body's own energy exchange $\int \mathbf f\!\cdot\!\mathbf u\,\mathrm dV = -3.411\times10^{-2}$.

> **The moving interface carries $\mathbf{43.4\%}$ of the seam's power exchange, and a static accounting reports exactly zero for that share.**

---

## 5. W97: the field-to-lumped certificate, and the co-normal that closes it

The seam is probed under three efforts on identical arithmetic, so the ratio is a measurement rather than an argument.

| effort | $\lVert S_{\text{fluid}}\rVert$ | $\lVert S_{\text{SUSP}}\rVert$ | lumped / fluid | $\beta$ | $\kappa$ | informative window |
|---|---|---|---|---|---|---|
| §2.2 diffusive, $\nu\,\partial v/\partial n$ | $8.071\times10^{-4}$ | $1.5625\times10^{-2}$ | $\mathbf{19.36}$ | $8.927\times10^{-6}$ | $1.76\times10^{3}$ | $4.735\times10^{-4}$ |
| §4.1 co-normal, $\nu\,\partial v/\partial n - (\mathbf u\!\cdot\!\mathbf n)v$ | $4.105\times10^{-2}$ | $1.5625\times10^{-2}$ | $\mathbf{0.3806}$ | $1.889\times10^{-4}$ | $1.79\times10^{2}$ | $4.045\times10^{-2}$ |
| the momentum exchange itself | $1.4875\times10^{1}$ | $1.5625\times10^{-2}$ | $\mathbf{1.05\times10^{-3}}$ | $2.606$ | $2.34$ | $1.486\times10^{1}$ |

**W97 closes, and by the repair the row itself names.**

- **The one-sidedness reproduces.** W97 reports $81\times$ at a rotor face under §2.2's effort. This seam — a different lumped expert, a different geometry — measures $19.4\times$ **at the exchange interval** and $77.4\times$ at the macro-step, because a spring's block is proportional to the interval it is probed over. The ratio is a property of the **pairing** (W76) *and* of the **cadence**, and both have to be quoted: §5.2 below is what happens when only one is.
- **§4.1's conservative co-normal crosses unity.** $19.36 \to 0.3806$, a factor of $50.9$, and it takes $\beta$ **up $21\times$** and $\kappa$ **down $10\times$** with it. The informative window — the $\beta_{\min}$ interval on which a swap both passes and could have failed — widens **$85\times$**. W47 scoped this form *out* at fluid–fluid seams because the advective term is evaluated at a shared ring cell where the two sides carry the same value and opposite normals and cancels identically; **there is no second fluid side at a wing, and it does not cancel.** [[case-study-ladder-to-f1]] §4 predicted exactly this and it is what happened.
- **And the mechanism is sharper than "the cell Reynolds number".** §2.2's effort is the **viscous traction alone**. At a fluid–fluid seam the pressure and advective parts cancel between the two sides; at a field-to-lumped seam nothing cancels, so §2.2's fluid side is missing most of the traction the lumped side is returning. The exact momentum exchange — the traction the plate actually feels, which is *the same physical quantity* the spring returns — moves the ratio $18\,430\times$. `L3/C9` checks that both sides declare the same `response_half`; **nothing checks that they mean the same effort**, and the $19.4\times$ is the measurement of that gap. This is `CASE-STUDY-GUIDE` mistake 7 one notch along, and W69's *"the wrong half differs from the right one by an $O(1)$ factor and no dimensionless diagnostic separates $O(1)$ from $O(1)$"* has a counter-example: here the two *correct* halves differ by four orders.

### 5.1 And the certificate at this seam was never blind, for a reason nobody had named

Under all three efforts `visible_above` is **negative**, so the fluid swap is informative at every admissible $\beta_{\min}$. That is the opposite of W97's rotor face, and the discriminator is not the ratio:

$$\text{rank}\,S_{\text{SUSP}} = 1 \quad\text{against}\quad \dim M = 17 .$$

> **A rigid lumped partner responds only in the rigid mode, so the assembled operator's smallest singular value is the field expert's own response in the $16$ directions the rigid body cannot excite — and $\beta$ collapses to $90\times$ below the fluid block's own norm.** A rotor disc responds cellwise and is full rank; a spring does not.

So the ladder's worry — *"if its substitution certificate is blind then no lumped subsystem in the car can be certified"* — is answered twice over: the effort mismatch that produces the blindness is repairable, and a **rigid** lumped subsystem is not blind in the first place. **[AI Inference]:** the ratio and the rank are independent axes and the taxonomy should carry both — a full-rank lumped partner with a matched effort is the good case, a rank-deficient one with a mismatched effort is where the $81\times$ came from. Argued from two seams, not measured across a family.

---

## 6. F5: the gradient through the composed stack

$h_0$ enters **only** through the spring, and every column is released from the same field, so $\mathrm dJ/\mathrm dh_0$ is a derivative of the composed stack and not of its initial condition. $J$ is the settled downforce over the last quarter of a lagged, six-window, moving-interface march. The gradient is reverse-mode autograd through every sub-exchange of every window's local solve, through the blend, through the global spectral projection, and through the Newton solve of the seam's own equation at both ends — the tape [[poc1-results-differentiable-design]] built, pointed at a **knob** rather than at a layout.

**It agrees with a finite difference over five decades.**

| central FD step | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-7}$ |
|---|---|---|---|---|---|---|
| relative to the adjoint | $1.48\times10^{-1}$ | $1.51\times10^{-2}$ | $3.57\times10^{-3}$ | $4.95\times10^{-4}$ | $4.85\times10^{-5}$ | $\mathbf{1.32\times10^{-6}}$ |

A clean truncation branch, monotone over five decades, **with the cancellation branch not yet reached at $10^{-7}$** — float64 throughout, and the objective is bit-reproducible ($0.0$ relative between two runs, because `_place` never uses `scatter_add` and no atomic re-ordering enters). PoC 1a's float32 checkpoint column floors at $1.4\times10^{-3}$ (**W121**) and would resolve none of this.

One gradient costs **$4.3$ to $6.7$ forward evaluations** — the same code on the same box, twice, with the forward itself moving $12.2$ s to $18.7$ s between them. That spread is quoted rather than one of its endpoints, because [[atlas-proof-of-concept-1]] §10 measured a factor of $2.9$ in one evening on unchanged code and this page is not going to make the same claim twice.

### 6.1 The sign depends on the horizon, and that is the result

| horizon (macro-steps) | $40$ | $80$ | $160$ | $240$ |
|---|---|---|---|---|
| $J$ | $0.1901$ | $0.2420$ | $0.2470$ | $0.2383$ |
| $\mathrm dJ/\mathrm dh_0$, adjoint | $\mathbf{+0.4108}$ | $-0.0951$ | $-0.1516$ | $-0.0547$ |
| central FD, $\delta = 10^{-5}$ | $+0.4107$ | $-0.0951$ | $-0.1513$ | $-0.0545$ |
| agreement | $2.9\times10^{-4}$ | $4.3\times10^{-4}$ | $1.5\times10^{-3}$ | $2.5\times10^{-3}$ |

> **The gradient changes sign with the horizon, and a design search run at the short one moves $h_0$ the wrong way.**

**It is physics, and the physics has two regimes.** On the transient a higher $h_0$ means a shorter descent, a smaller $\lvert v_{\text{plate}}\rvert$ and therefore *less* aerodynamic unloading — a positive sensitivity. At equilibrium the plate velocity is zero and only the ground effect is left — a negative one. The quasi-static prediction is closed form,

$$\frac{\mathrm dJ}{\mathrm dh_0} \;\longrightarrow\; \frac{\mathrm dL}{\mathrm dh}\cdot\frac{k}{k + \lvert \mathrm dL/\mathrm dh\rvert} \;=\; -0.163 \quad\text{at } h^\star = 0.2257,$$

and the $160$-step adjoint is $-0.152$, within $7\%$ of it. **Nothing here is a high-frequency artefact of a learned representation — there is no learned representation in this column, which is the point of running it classically first** ([[case-study-ladder-to-f1]] §2). What F5 was worried about is not what bit; what bit is that the objective has a transient in it and the horizon is not on anyone's record.

### 6.2 Usable, and the asymmetry that is the argument for the adjoint

A $1\%$ change in $h_0$ moves $J$ by $0.244\%$ against the plate's own settled unsteadiness of $1.256\%$. A finite difference taken on an evaluator carrying that noise would need a **$5\%$ perturbation** to see the signal; the adjoint needs none, because the pipeline is bit-reproducible and the derivative is exact rather than differenced.

> **That asymmetry is the argument for differentiating a composed stack rather than sampling it, and it is measured here rather than asserted.**

---

## 7. $\mathcal R(t)$, with the motion term in it

The gate asks that the residual close with the motion term included. Following [[case-study-thermal-strain-atlas-0.1]] §6's rule — *the right residual is the receiving subsystem's own balance*, because a global $\mathcal R$ is blind to a lumped term by six orders — the spring's own energy rate is compared against the interface power:

$$\frac{\mathrm d}{\mathrm dt}\Big[\tfrac12 k (h_0-h)^2\Big] \quad\text{against}\quad -L\,v_{\text{plate}} .$$

| | relative residual |
|---|---|
| **with** the motion term | $\mathbf{4.097\times10^{-3}}$ |
| **without** it (a static accounting) | $1.0000$ |

A factor of $244$, against a level of $\lvert\mathrm dE/\mathrm dt\rvert_{\max} = 1.870\times10^{-2}$ per unit span. The residual that remains is the finite differencing of $E(t)$ and not the bond. **$\mathcal R$ closes with the motion term in it and does not close without it**, which is the gate.

---

## 8. The compile, and the refusal that is the deliverable

| graph | verdict | envelope | refusals | decertifications |
|---|---|---|---|---|
| fixed floor, either effort | `admit-uncertified` | E(H H H H U H F) | **0** | 7 |
| moving, either effort | **`refuse`** | E(H **F** H H U H F) | 1 rule, 2 ports | 7 |

The moving graph is refused at `L2/InterfaceMotion` on **both** ports of the wing seam and on nothing else new, and `E2` goes from `holds` to `fails`. **That refusal is correct and this case study does not repair it**: no rule exists, and a moving interface silently invalidating a cached operator is the silent-wrongness class. What §4 supplies is the three measurements the slot has asked for since it was written, which is what turns a named hole from a declaration into a priced one.

Nothing here reaches `admit`: $L$ is unmeasured (**W1**), $\sigma$ and $C_\mu$ are unmeasured (**W3**, **W49**), and a non-empty `unmeasured` list forces `admit-uncertified` since **W56**.

---

## 9. Four framework defects the case study found without looking for them

### 9.1 W127 — an expert's declared step and the cadence the composition runs it at

`Suspension.dt` was declared at the macro-step, where **R10b** fixes the exchange interval at `dt_native / substeps_per_macro_step`. A spring's block is proportional to the interval it is probed over, so the lumped block came out **four times too large** and W97's headline ratio read $77.4$ where it should read $19.4$.

`CASE-STUDY-GUIDE` mistake 5 is exactly this on the *field* side — *hard-coding the sub-step count*, measured at two orders in $\tau$ — and R10b was written for it. **Nothing checks the lumped side.** R10b reads `substeps_per_macro_step` only on agents that declare `elliptic_subsolve=exposed`; an algebraic agent declares `NONE`, is skipped, and its own response interval is never compared against the exchange interval the compile derives. It is one comparison and it has no rule.

### 9.2 W128 — a design sensitivity has no horizon on the record

§6.1's sign flip is not a defect in the gradient; it is a defect in what a gradient can be *declared* as. `MeasuredConstants` carries `sigma_lag` precisely because **W86** established that $\sigma$ is a function of the lag and quoting it without one is meaningless. A design sensitivity through a rollout has exactly the same shape — it is a function of the horizon, it changes **sign** over the range this study measured, and there is no field for it. `poc1-results-differentiable-design` quotes gradients at one horizon and says so in prose; this is W86's discipline asking to be applied one level up.

**And it is W123 one level up too.** Tier 23 opened W123 because *the one-exchange-interval defect is not what a rollout accumulates, and every cut criterion in this vault predicts the former*. A gradient taken at a short horizon is the same error with a different consumer.

### 9.3 W129 — `L2/R10` decertifies a zero-parameter closed form

The suspension declares `governing_family = "incompressible-navier-stokes-2d"` — correctly, because an algebraic closure *within* a continuum problem is not a different continuum problem, and declaring otherwise fails E3 at the seam — and `elliptic_subsolve = NONE`, correctly, because two lines of algebra contain no solve. R10's third branch then decertifies it: *"an incompressible solver almost always contains a pressure solve."*

It is not a solver. `wake_array.RotorDisk` carries the same decertification for the same reason and no page has recorded it. The rule reads two fields and cannot see that the agent is lumped; the discriminator it needs is already on the record (`stencil_radius = 0`, `L_native` a body scale rather than a domain), and it is small.

### 9.4 W130 — a probed block that is exactly zero is indistinguishable from an empty interface problem

The plate is declared in **global** coordinates and a window sees it in its own frame. With the x offset unsubtracted the plate sat outside the window's array, `grid_sample`'s border clamp returned a constant, and the probed fluid block came back **exactly zero**. The compile then read $16$ excess null directions and refused at `L4/null-space` — a true statement about the matrix, with nothing pointing at the cause.

That failure mode is byte-identical to the legitimate one spec §6.4(a) describes and [[probed-dtn-coupling]] §4.5 celebrates: a `bc_channel: none` agent whose $\Lambda \equiv 0$, where *"the interface problem is not badly conditioned; it is empty."* **A mis-wired agent and a genuinely uncoupled one produce the same block, the same $\Xi = 0$, the same null count and the same refusal.** The distinguishing evidence is free and nobody collects it: an agent declaring a `bc_channel` above `NONE` and returning an identically zero block is contradicting its own declaration, and `L1/E3`-style *"a measured quantity contradicts a hypothesis outright"* is exactly the refusal class for it. Pinned here by a test that asserts the block is nonzero.

---

## 10. What this case study cannot say

- **One wing model, and it is a model.** The porous inclined plate is `disk.ActuatorDisk` with a rotated axis; it has thickness and an exact momentum exchange, and it has no Kutta condition, no bound circulation and no boundary layer on its own surface. $C_L$ figures here are numbers about the composed model.
- **One branch of ground effect.** A rolling road at $U_\infty$ carries no boundary layer by construction, so the viscous gap choke that ends the real downforce curve is not in this model. $L(h)$ came back monotone and the turnover was looked for; that is a statement about *this* construction, not about wings.
- **One Reynolds number and one grid.** $\mathrm{Re}_c = 125$, $\mathrm{Re}_h = 3.906$, $208\times144$ cells. Nothing here is a grid-convergence study.
- **The cut's own defect has not peaked at the horizon marched** (§3), so $2.626\times10^{-5}$ is a value and not a bound.
- **The gradient is measured at four horizons on one objective and one design parameter.** The sign flip is a fact about this objective; the two-regime *reading* of it is argued from the closed-form quasi-static limit and is not established for a general one.
- **No learned expert ran.** [[case-study-ladder-to-f1]] §2's classical-first split says the substitution campaign prices R10 per seam, and this seam has not been priced. What §5 establishes is that a field-to-lumped substitution certificate *can* be made informative; whether Poseidon-T is admissible at a wing seam is unmeasured.
- **Nothing reaches `admit`**, and §8 names why.

---

## 11. Rows this closes and opens

| row | outcome |
|---|---|
| **W30** | **the comparison the row was built for.** W30 closed in Tier 0 with a drift number on *static* seams, emitted *"even on static runs"* so that a moving one would have something to be compared against. It does now: $2.27\times10^{-1}$ against a static $1.51\times10^{-2}$ on the same seam and against W30's own $2.4\times10^{-4}$–$3.5\times10^{-3}$ band, with the re-probe count and the unaccounted power beside it. `InterfaceMotion`'s three `measurements_required` are supplied for the first time |
| **W22** | **half priced.** The row splits into `TopologyEvent` (W31, untouched) and `InterfaceMotion` (W30). The second half now has numbers and an economic verdict: a cached $S$ does not survive one exchange at a solution-dependent seam, and re-probing costs $75\%$ of the march for one seam of eight |
| **W97** | **closed**, by the candidate repair the row names. §4.1's conservative co-normal — which W47 scoped *out* at fluid–fluid seams because the advective term cancels there — crosses unity at a field-to-lumped seam, $19.4 \to 0.381$, taking $\beta$ up $21\times$, $\kappa$ down $10\times$ and the informative window $85\times$. The mechanism is sharper than the row's own statement: §2.2's effort is the viscous traction alone and the two sides of the seam were returning *different physical quantities under the same declared half* |
| **F5** | **measured** ([[f1-pathmap-and-end-goal]] §5). The adjoint agrees with a central finite difference to $1.3\times10^{-6}$ and costs $4.3$ forward evaluations; the gradient is usable where a finite difference on the same noisy evaluator would need a $5\%$ perturbation. **Not falsified, and it fails in a way the criterion does not describe**: the risk F5 names is a high-frequency artefact of a learned representation, and what bit is the horizon |
| **W123** | **reframed, one level up.** *"A cut criterion stated with a horizon"* generalizes to any quantity consumed from a rollout. A design sensitivity has the same defect and the same fix |
| **W106** | **applied as a discipline rather than discovered.** The bitwise floor here is exactly zero (asserted, twice) and nothing is quoted against it; the level is the plate's own settled unsteadiness, $1.256\%$ |
| **W124** | **honoured before the fact.** The tiling places every cut $8$ cells clear of the plate and a test asserts it, which is what W124's *done when* asks a case study to do while the `ArrayTiling` fix is open |
| **W116** | **recurs.** CS-9 opened it for a co-located splitting error with no slot on `MeasuredConstants`. §3's lag defect is a *seam* splitting error with the same problem: it is first order in the exchange lag, it is not a transmission infidelity, and `sigma` is the wrong field for it |
| **W127** | **opened.** An expert's declared response interval is never compared against the exchange interval the compile derives, on any agent R10b does not read — §9.1 |
| **W128** | **opened.** A design sensitivity through a rollout is a function of the horizon, changes sign over the range measured here, and has no field on the record — §9.2 |
| **W129** | **opened.** `L2/R10`'s *"incompressible family with no elliptic sub-solve"* branch decertifies a zero-parameter closed form; the discriminator it needs is already declared — §9.3 |
| **W130** | **opened.** A mis-wired agent and a genuinely uncoupled one produce an identical probed block, an identical $\Xi = 0$ and an identical refusal, and the contradiction with the agent's own `bc_channel` is free evidence nobody collects — §9.4 |

## See Also

- [[case-study-ladder-to-f1]] — §4's CS-10 row, which scheduled this and named all three deliverables; §6's critical-path table, where W97 and W30/W22 sat
- [[f1-pathmap-and-end-goal]] — §1.1's *"the couplings are the physics"*, §5's F5, and §6.2's *"the physics layer should expose a design parameterization"*
- [[gap-worklist]] — Tier 24: W97 closed, W30's comparison made, W127–W130 opened
- [[end-to-end-architecture-spec]] — §12.2's `InterfaceMotion` slot, whose declared interface this fills in, and §1's E2
- [[probed-dtn-coupling]] — §2.2's normative effort, §4.1's conservative co-normal, §4.3's $\beta$, and §4.5's empty interface problem that §9.4 is about
- [[case-study-thermal-strain-atlas-0.1]] — CS-9, whose lag construction §3 reuses and whose *"the right residual is the receiving subsystem's own balance"* §7 follows
- [[case-study-seam-placement-atlas-0.1]] — CS-9★, whose standing rule about marching past a crossing §3 applies as a gate
- [[case-study-scaling-ladder-atlas-0.1]] — the exposed-agent plus `ProjectedAssembly` column this is built on, and the $N=1$ control it demands
- [[poc1-results-differentiable-design]] — the tape §6 reuses, and W121's float32 finite-difference floor
- [[port-algebra-atlas-0.1]] — the `MECH` bond whose flow half is what makes §2.3's interface problem well posed
