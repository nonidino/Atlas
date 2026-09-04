# Wing FSI — two-way coupling across a genuinely new governing family

**Type:** Concept page — **case-study summary** (folder: `Atlas 0.1/case-study-wing-fsi/`)
**Status:** opened 2026-09-04. The thirteenth real case study, Phase C's first row, and the first graph in this vault whose two agents solve **different continuum problems**, are coupled **both ways**, and meet on a **surface**. Everything here is quoted from `out/w136/w136.json`; nothing is estimated.
**Code:** `atlas/cases/wing_fsi.py` (the declarations, the flexible wing, the structural agent and the three couplings), `scripts/w136_wing_fsi.py` (the driver, eight stages), `tests/test_tier29_wing_fsi.py` (28 tests)
**Related:** [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[gap-worklist]] · [[end-to-end-architecture-spec]] · [[general-coupling-scheme]] · [[port-algebra-atlas-0.1]] · [[probed-dtn-coupling]] · [[master-error-bound]] · [[case-study-ground-effect-atlas-0.1]] · [[case-study-thermal-strain-atlas-0.1]] · [[case-study-brake-thermal-atlas-0.1]] · [[case-study-scaling-ladder-atlas-0.1]]

---

## 1. The question, and why it is Phase C's first row

[[case-study-ladder-to-f1]] §4's CS-12 row asks one thing:

> | **CS-12** | `wing_fsi.py` | two-way FSI: does $\mathcal R$ close across a **genuinely new governing family**? | none (classical) | **5** | ~1–2 weeks |

Nine of the twelve real case studies before it couple incompressible Navier–Stokes to incompressible Navier–Stokes. `thermal_seam`'s gas meets a thermoelastic shell ([[tier0-measurements]] §13–17) at **one** seam and one way; [[case-study-brake-thermal-atlas-0.1]]'s disc meets its duct at **one** seam and one way; [[case-study-thermal-strain-atlas-0.1]] adds elasticity **co-located**, with no interface at all and no port. [[case-study-ground-effect-atlas-0.1]]'s spring is two-way and moving, and it declares the *flow's own* `governing_family` — correctly, because an algebraic closure inside a continuum problem is not a different continuum problem.

**An elastic solid is.** This is the first graph where the two sides of a seam genuinely disagree about what equation they are solving, in both directions, across a surface — which is the shape every aero-structure coupling in a Formula 1 car has, and which is why the pathmap puts a deflecting wing at rung 5.

The row inherits two things from Phase B and had to settle one before it could be built:

1. **W114, settled first and not during.** A quasi-static structural agent is `EMBEDDED` and has **no `split-step` escape** — no time derivative, nothing to sub-step — so `L2/R10` refused *every* graph containing one. [[case-study-ladder-to-f1]] §10 and §12 both name this as the blocker and both say it should be settled before CS-12 rather than during it. §2 below is that settlement.
2. **A deflecting wing is a moving interface**, so CS-10's re-probe economics apply unchanged and `L2/InterfaceMotion` refuses the deforming graph. That refusal is inherited, correct, and already priced.
3. **The seam is the port algebra's own `MECH` bond** — traction against surface velocity — used for the first time between two *fields* rather than between a field and a lumped body.

---

## 2. W114, settled before the build

`L2/R10` refuses any graph containing an agent that declares `elliptic_subsolve=embedded`, and its sentence names a premise:

> *"the graph decomposes the domain, so the decomposition changes the operator rather than restricting it."*

**The rule never checked that premise.** It read `elliptic_subsolve` and nothing else, and it fired on two graphs where the decomposition does not cut the agent at all: [[case-study-thermal-strain-atlas-0.1]]'s co-located split, where both agents own all of $\Omega$, and this one, where the fluid is tiled into six windows and $\Omega_{\text{solid}}$ is **one agent's whole domain**. In both, the elliptic solve runs over exactly the region a monolithic solve would run it over, so the elliptic error R10 exists to refuse — flat in distance from the cut, flat in $\Delta t$, untouched by any halo — cannot arise.

CS-9 recorded that as `thermal_strain.R10_SCOPE` and took the refusal. **CS-12 could not.** A quasi-static structural agent is `EMBEDDED` or it is not an agent: there is no time derivative to sub-step, so the `split-step` escape that `window_ns` and [[case-study-brake-thermal-atlas-0.1]] both take does not exist on this side. Taking the refusal would have meant the whole compile-side content of Phase C's first row was *"R10 refused, again."*

**The handle, and it is decidable from the declarations alone.** An `EMBEDDED` agent is refused only when the graph contains **another agent of the same `governing_family`** — which is the compile-time signature of *"a larger region of the same physics, of which this agent has been given a piece."* Every graph R10 was derived on is a tiling of one family and keeps its refusal unchanged.

Two things about that predicate, both stated in the rule's own message:

- **It is a proxy for *"is this agent's domain cut"*, not a decision procedure.** What would replace it is a structured domain declaration on `Agent`, and there is none — `Agent.domain` is free text.
- **It is conservative in the safe direction.** Two agents of one family on genuinely *disjoint* regions are still refused, which is a false refusal that inspection resolves, where the other direction would be a silent admission. R10's whole reason for existing is that its error class is silent.

**And it is checked by a control rather than trusted.** Declare the structural agent with the *fluid's* `governing_family` — which is what a window tiling looks like from R10's side — and the refusal comes straight back at `L2/R10`. A rule with nothing to fire on is not a rule; this one still fires.

| graph | verdict | envelope | refusals | decertifications |
|---|---|---|---|---|
| fixed shape, six windows | `admit-uncertified` | E(H H **F** H U H **F**) | **0** | 8 |
| fixed shape, one window | `admit-uncertified` | E(H H **F** H U H **F**) | **0** | 8 |
| deforming, six windows | **`refuse`** | E(H **F** **F** H U H **F**) | 1 rule, 2 ports | 8 |
| the control (structure declaring the fluid's family) | **`refuse`** | — | `L2/R10` | — |

**This is the first classical graph in the vault containing an `EMBEDDED` agent that compiles with zero refusals**, and W114 closes. What it costs is recorded in the rule's own admission: the narrowing clears R10 and **does not** clear the agent's *reach* — an embedded elliptic solve has an infinite domain of dependence inside its own region whether or not that region was cut, which §4 measures.

### 2.1 And the same scope defect is one rule along — W136

`L2/R10/halo` decertifies the structural agent for being *implicit with a nonzero stencil*, so that *"stencil_radius $\times$ substeps is an EXPLICIT agent's domain of dependence, not theirs"* and *"the halo the exchange needs is undecidable here."* Every word of that is true and **there is no halo on this seam to be inadequate**: $\Gamma$ is a *physical* boundary of $\Omega_{\text{solid}}$, not an artificial cut; the structure is not tiled; and no overlap exists on that side for anything to outrun.

It is the same premise R10 was missing, in the same shape, on the next rule down — and it is left **open** rather than fixed, because the two failures are not equally dangerous. A refusal that over-fires is terminal; a decertification that over-fires is honest and noisy. Opened as **W136**.

---

## 3. How it is built

**The fluid column is CS-10's, unchanged, and that is the point.** Same $\mathrm{d}x$, same $208\times144$ domain, same six $80\times80$ exposed `reference.WindowNS` windows at halo $16$, same `ProjectedAssembly` with one global spectral Leray projection per exchange, same macro-step with four exchanges inside it, same porous inclined plate at $20^\circ$ with $C_N = 20$. **Nothing about the fluid moves between CS-10 and CS-12**, so the variable is the partner: CS-10's was a two-line algebraic spring with one degree of freedom, and this one is a field with $32$.

| | |
|---|---|
| domain | $208\times144$ cells $= 3.25\times2.25$, $\mathrm dx = 1/64$ |
| windows | six, $80\times80$, halo $16$, ramp $8$ |
| plate | chord $0.5$ ($32$ cells) at $20^\circ$, cells $x\in[88,119)$, $y\in[40,52)$, owned by `F10` |
| cuts clear of the wing | **8 cells in $x$, 12 cells in $y$** — W124's discipline, on both axes |
| structure | $32\times2$ Q1 plane stress, $t/c = 0.04$, **clamped at the leading edge** |
| macro-step | $0.0125$, with four composition-layer exchanges inside it |
| spin-up | $120$ fixed-shape macro-steps; settled load $0.227255$ |
| settled unsteadiness | **$1.503\%$ peak-to-peak over the last quarter** — the LEVEL every difference below is quoted against |

**The structure is `ThermoStruct2D.solve_mechanical`, the real solver, unmodified**, on a mesh of the plate itself, clamped at the leading edge and free at the trailing edge — a cantilevered flap, which is what an F1 front-wing element is. `T = T_ref` everywhere and `alpha = 0`, so the thermal load is identically zero and the solve is pure quasi-static elasticity: CS-9 measured the thermal-strain **bond**, and this case study needs the **elasticity**, so the coupling is switched off by a declared coefficient rather than by reaching into the solver.

### 3.1 The conjugate pair is NORMAL, and getting that wrong costs 12%

The `MECH` bond is traction against velocity. At a plate there are two ways to write it and only one is the interface power:

- **normal** — effort $f_n$, the plate-normal traction; flow $w_n$, the wetted surface's normal velocity. Power $= f_n w_n$, exactly, because the force is along $\mathbf n$ and the surface moves along $\mathbf n$.
- **vertical** — effort $f_n n_y$; flow $w_n n_y$. This is what CS-10's seam carried, correctly, *because a ride height is vertical*. It projects **both** halves, so its product is $f_n w_n n_y^2$ and it under-reports the power by $\cos^2\alpha = \mathbf{0.883}$ at this incidence.

Two correctly-declared `EFFORT` halves whose product is not the power — **W132's family, in a form that is arithmetic rather than physical**, and asserted in the tests rather than argued.

The choice is not only about the power. The structural compliance in the normal pairing is **symmetric by Betti**; in the vertical pairing it is not, because two load systems that push along $\mathbf n$ and are read along $\mathbf y$ do not satisfy Betti's identity in the read variable. Measured, on the same mesh:

| pairing | $\lVert C - C^\top\rVert/\lVert C\rVert$ | work-conjugate vs geometric $\delta$ |
|---|---|---|
| **normal** | $\mathbf{5.08\times10^{-11}}$ | $5.73\times10^{-4}$ |
| vertical | $2.5\times10^{-4}$ | $9.3\times10^{-2}$ |

### 3.2 The surface operator, built from the expert and not re-derived

The seam carries a generalized displacement $\delta$, defined as the work-conjugate of the traction the port carries, $\sum_s p_s\,\delta_s\,\mathrm ds = \mathbf f^\top\mathbf u$. From the expert's own `solve_mechanical` answers $U$ to the unit station tractions and the expert's own assembled stiffness $K_{me}$,

$$C \;=\; \frac{U^\top K_{me} U}{\mathrm ds}, \qquad S_e \;=\; C^{-1},$$

so $C$ is symmetric **by construction** rather than by reciprocity happening to hold, and this function contains **no physics at all** — $U$ comes from the expert and $K_{me}$ comes from the expert. $S_e$ is the structure's Dirichlet-to-Neumann map on $\Gamma$: the traction it takes to hold the surface at a given deflection. It is the field analogue of CS-10's spring rate $k$, and where that was rank one this is rank $32$.

| | measured |
|---|---|
| $\lVert C - C^\top\rVert/\lVert C\rVert$ | $5.08\times10^{-11}$ |
| $\kappa(C)$ | $7.08\times10^{8}$ |
| $\mathrm{eig}(S_e)$ | $31.25$ to $2.212\times10^{10}$ |
| reduced $S_e$ against `solve_mechanical`'s own answer | $2.44\times10^{-11}$ |
| linearity in $E$ | $\mathbf{0}$ — **bitwise** |

The last two are what let the march be differentiable in the stiffness without re-running the FE solver inside it: linear elasticity is exactly linear in $1/E$ at fixed $\nu$, so $S_e(E) = (E/E_{\text{ref}})\,S_e(E_{\text{ref}})$ holds to the bit, and $C\,t$ is `solve_mechanical`'s own answer to $2.4\times10^{-11}$, which is the direct sparse solve's round-off at $\kappa = 7\times10^{8}$. **Each is a control and not a floor** (W106): nothing below is quoted against them.

### 3.3 The stiffness is declared in flow units, and that is a scope statement

`SolidMaterial`'s own $E = 70$ GPa against a traction of order $\rho U_\infty^2 = 1$ deflects the plate by $10^{-11}$ and there is no coupling to measure. $E^* = E/(\rho U_\infty^2)$ is the aeroelastic stiffness parameter and it is declared as **the one number chosen to make the coupling measurable rather than derived**, exactly as CS-10 declares $C_N = 20$. At $E^* = 5\times10^{4}$ the settled trailing edge sits $1.64$ cells below its unloaded position.

Two consequences that bound what any number here means, both in `wing_fsi.STRUCTURE_SCOPE`:

- **Q1 elements lock in bending**, so a two-element-deep plate is stiffer than its own beam theory. $E^*$ is calibrated to a deflection, so the locking is absorbed into the calibration — and **no stress figure here is a statement about an alloy**.
- The seam's generalized displacement is identified with the plate line's position in the fluid. The porous plate has **no thickness in the flow** — it is a line of stations — so there is no second definition to be inconsistent with, and what the identification costs is $5.73\times10^{-4}$ of the compliance norm, a property of the structural model's thickness that the fluid model does not resolve at all.

### 3.4 The referent, and a zero-cut control

`FSIRollout` is one code path and the referent is a *tiling* of it: `WingTiling.single()` puts one window over the whole domain, the partition of unity is identically $1$, cut and blend are the identity, and the same global projection is applied to a field nothing was blended into. The referent's structural solve is the same structural solve. **So the referent and the composed column differ by the cut and by nothing else**, and the composed path at one window reproduces itself bit-for-bit (asserted).

---

## 4. The seam, probed

### 4.1 The structural response reaches every station, and it is the physics

`probe.support_reach` pokes one seam cell and reports which cells move. On the structural agent, at every amplitude from $1$ to $10^{-4}$:

| agent | cells moved | reach | declared `stencil_radius` $\times$ `substeps` |
|---|---|---|---|
| **`STRUCT`** | $\mathbf{32}$ of $32$ | $16$ (the whole seam) | $1$ |
| the fluid window | $11$ of $32$ | $5$ | $2\times4 = 8$ |

**W93 measured exactly this reading on a frozen neural operator** — the response nonzero in all $128$ seam cells against a declared radius of $2$ — and drew the conclusion that *"a neural operator's receptive field is global by construction, which is a fact about the architecture and not about the physics."* Here the same reading has the **opposite cause**: a quasi-static elliptic operator inverts a sparse SPD matrix whose inverse is dense, so a poke at one station moves every station, and that is the physics of an elastic body. It is the second instance in the vault of a globally-supported response and the first whose globality is not an artefact of how the operator was built.

**[AI Inference]:** this is why the halo rule's decertification (§2.1) cannot simply be switched off for structural agents even once R10's premise is fixed. The reach is real; what is absent is an *overlap* for it to be compared against, because the seam is a physical boundary rather than a cut. The two facts are independent and only the second is a scope defect.

### 4.2 W97's repair moves the one-sidedness by $1.8\times10^{4}$ and cannot close it

`FLUX_MODES` runs the three efforts a `MECH` seam can carry on identical arithmetic:

| effort | $\lVert S_{\text{fluid}}\rVert$ | $\lVert S_{\text{STRUCT}}\rVert$ | ratio | $\beta$ | the small block's share |
|---|---|---|---|---|---|
| `diffusive` — §2.2's $\nu\,\partial w/\partial n$ | $3.914\times10^{-4}$ | $4.863\times10^{5}$ | $1.242\times10^{9}$ | $1.452$ | $8.05\times10^{-10}$ |
| `conormal` — §4.1's $\nu\,\partial w/\partial n - (\mathbf u\!\cdot\!\mathbf n)w$ | $7.453\times10^{-2}$ | $4.863\times10^{5}$ | $6.525\times10^{6}$ | $1.474$ | $1.53\times10^{-7}$ |
| `reaction` — the exact momentum exchange | $7.018$ | $4.863\times10^{5}$ | $\mathbf{6.930\times10^{4}}$ | $2.166$ | $1.44\times10^{-5}$ |

**W97 closed at a field-to-lumped seam by taking this ratio from $19.4$ to $0.381$ — below one — with the conservative co-normal. That closure does not transfer here, and the reason is that the one-sidedness is a different quantity.** At CS-10's seam the disparity was an *effort-convention* defect: §2.2's normative effort is the viscous traction alone and a lumped body's traction carries no $\nu$, so the two sides were returning different physical quantities under one declared half. Here the co-normal buys $190\times$ and the exact momentum exchange buys $1.79\times10^{4}$ — reproducing W132's $1.84\times10^{4}$ to $3\%$, which is the same arithmetic on the same plate model rather than an independent confirmation — and the ratio is still $\mathbf{6.9\times10^{4}}$. **What is left is the structure being stiff.**

**And it is not a truncation artefact, which is the reading this measurement was made to falsify.** A bending plate's stiffness goes as the fourth power of the wavenumber, so the obvious story is that the structure dominates the *high* modes and the declared interface space carries the fluid's cutoff. Measured mode by mode on the declared $M = 17$: the structural diagonal runs $2.12\times10^{4}$ to $2.279\times10^{4}$ — flat to $1.08\times$ across $M$ — and the fluid's $3.993$ to $3.975$. The structure is $\sim\!5300\times$ the fluid in **every** mode, the stiff structural modes are not in the declared space at all, and no choice of interface space repairs it.

> **What that costs the substitution campaign, and it is the sharpest thing on this page.** `composition.SubstitutionCertificate` can only see a perturbation bounded by the swapped agent's own block, and the fluid's share of this seam is $\mathbf{1.44\times10^{-5}}$. **The flow expert is the one you want to replace with a learned operator, and at an aero-structure seam its certificate is blind by five orders — worse than W97's lumped case by three.** Every aero-structure coupling in a Formula 1 car has this shape. Opened as **W137**.

### 4.3 $n_0(\Gamma)$ gains a fifth row, and it differs from CS-9's by the boundary condition

| what meets at the seam | $n_0(\Gamma)$ |
|---|---|
| solid–solid `MECH` on a **free** body (CS-9) | $1$ per unconstrained rigid direction — a uniform normal velocity **is a rigid translation**, so no strain, no reaction, and the probed operator cannot see it |
| **fluid–solid `MECH` on a CLAMPED body (here)** | $\mathbf{0}$ — the same trace **bends** it, and the reaction is nonzero in every direction of the trace space |

Measured: `null_dim = 0` against a declared $0$, at $\sigma_{\min}/\sigma_{\max} = 4.45\times10^{-6}$ on the assembled operator and $2.99\times10^{-6}$ on the structural block alone. **It is the boundary condition and not the physics**, and CS-9's own row said as much — *"not incompressibility, not lumpedness — kinematics"* — without a second body to check it against.

### 4.4 The deforming interface drifts $200\times$ its frozen self

$\lVert S(t + K\Delta t) - S(t)\rVert$ relative, $K = 10$, on the same seam in a deforming column and in one held rigid from the same field:

| column | drift | the interface moved |
|---|---|---|
| **deforming** | $\mathbf{8.245\times10^{-1}}$ | $1.18$ cells |
| frozen shape | $4.119\times10^{-3}$ | $0$ |
| CS-10, rigid body translating | $3.77\times10^{-1}$ | $1.10$ cells |
| CS-10, same seam held still | $2.51\times10^{-2}$ | $0$ |
| Tier 0's static band (W30) | $2.4\times10^{-4}$ – $3.5\times10^{-3}$ | — |

**A deforming interface drifts $2.2\times$ more than a translating one over the same travel**, and its frozen control sits at the top of Tier 0's static band rather than above it. So the drift belongs to the *shape change* and not to the flow developing around the plate, and CS-10's re-probe economics get worse rather than better: a cached $S$ does not survive one exchange at this seam either, and there are more directions for it to go stale in.

---

## 5. The gate: does the split reproduce the referent's aeroelastic response?

**Yes, and what order the lag defect has depends on where you read it.**

Released from the settled fixed-shape field at $\delta = 0$ and marched $240$ macro-steps at $E^* = 5\times10^{4}$. The referent — one window, the FSI seam solved at every exchange — falls to a **minimum tip deflection of $-0.030871327$ at step $29$** and recovers to $-0.027220161$: it overshoots its own equilibrium and comes back, so a march stopped at the minimum reports a different answer. Every difference below is quoted against $\lvert\delta_{\text{tip}}\rvert$ **settled**, which is a fixed level; a pointwise relative error divides by a deflection that starts at zero and its maximum lands on the release transient, measuring the denominator.

| column | $\delta_{\text{tip}}$ at $240$ | $\max\lvert\Delta\delta\rvert/\delta_{\text{settled}}$ | at step | over the settled half | at the end | sign changes |
|---|---|---|---|---|---|---|
| referent (tight, one window) | $-0.027220161$ | — | — | — | — | — |
| lag 1, no cut | $-0.027220109$ | $2.700\times10^{-2}$ | $7$ | $5.596\times10^{-4}$ | $+1.898\times10^{-6}$ | **6** |
| lag 2, no cut | $-0.027219757$ | $6.401\times10^{-2}$ | $8$ | $1.300\times10^{-3}$ | $+1.482\times10^{-5}$ | **6** |
| lag 4, no cut | $-0.027218056$ | $1.312\times10^{-1}$ | $9$ | $2.692\times10^{-3}$ | $+7.730\times10^{-5}$ | **6** |
| lag 1, six windows | $-0.027217082$ | $2.701\times10^{-2}$ | $7$ | $4.609\times10^{-4}$ | $+1.131\times10^{-4}$ | **4** |
| tight, six windows | $-0.027217136$ | $\mathbf{1.139\times10^{-4}}$ | $68$ | $1.136\times10^{-4}$ | $+1.111\times10^{-4}$ | $0$ |

Four things.

**The lag defect is first order in the lag, and third order at the end.** $\log_2$ ratios of the **peak**: $1.245$, $1.035$. Of the **settled half**: $1.216$, $1.050$. Of the value **at step 240**: $2.965$, $2.383$. The columns re-converge, so the last point of a rollout is the one place the lag looks like it costs almost nothing. This is [[gap-worklist]] **W123**'s shape again — *what a rollout accumulates is not the per-interval quantity* — with the sign the other way: here the accumulated defect is *smaller* than the per-interval one because the transient is the whole of it.

**The seam's lag dominates the domain cut by $237\times$.** The lag alone is worth $2.700\times10^{-2}$ and the six-window cut alone $1.139\times10^{-4}$. CS-10 measured $97\times$ at a field-to-lumped seam and CS-11 measured the same shape at a multirate one; this is the third coupling kind and the third time **the seam is where the error is and the domain cut is not**. On this graph the ratio is $2.4\times$ worse than CS-10's, on a seam with $32$ degrees of freedom instead of one.

**And the crossing is there -- six times on the lagged columns and not at all on the cut.** The lagged columns cross the referent at steps $30, 70, 97, 156, 171, 207$ (lag 1) and $33, 75, 103, 158, 179, 212$ (lag 4): the plate sheds, and the difference oscillates through zero with it. The six-window **cut** column never crosses at all and sits at a steady $+1.1\times10^{-4}$ from step $50$ onwards, which is why its own defect is a clean number and the lag's is not.

Tier 23's standing rule is that a comparison of two configurations is not a result until it has been marched past the point where the curves could cross, and *the crossing has to be looked for rather than assumed absent*. It was, and the consequence is concrete. The lag-1 column's difference runs

$$+2.700\times10^{-2}\ ( \text{step } 7), \qquad -6.420\times10^{-3}\ (\text{step } 43), \qquad +1.9\times10^{-6}\ (\text{step } 240),$$

so a march stopped anywhere in the first fifty steps reports a different number **and a different sign** from one marched to the end -- and the number it reports at the peak is $1.4\times10^{4}$ times the one at the horizon.

**The cut's own defect HAS peaked**, at step $68$ of $240$, with zero sign changes. CS-10 had to report that its cut defect was still rising at the horizon marched, so $2.626\times10^{-5}$ was a value and not a bound. Here $1.139\times10^{-4}$ is a bound over $240$ macro-steps.

**The controls.** The composed path at one window reproduces itself bit-for-bit, and it is a control and not a floor: the settled unsteadiness of the plate, $1.503\%$ peak to peak, is the level, and nothing above is quoted against zero (**W106**).

---

## 6. $\mathcal R$ across the FSI seam

[[case-study-thermal-strain-atlas-0.1]] §6's rule — *the right residual is the receiving subsystem's own balance, because a global $\mathcal R$ is blind to the coupling under test by six orders* — applied to a structure that receives all its energy through $\Gamma$:

$$\frac{\mathrm d}{\mathrm dt}\Big[\tfrac12\,\boldsymbol\delta^\top S_e\,\boldsymbol\delta\,\mathrm ds\Big] \quad\text{against}\quad \int_\Gamma f_n\,w_n\,\mathrm ds .$$

| accounting | whole march ($120$ macro-steps) | over the settled second half |
|---|---|---|
| **with** the deformation term | $1.0548\times10^{-1}$ | $6.206\times10^{-6}$ |
| **without** it — a static accounting | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ |
| with the deformation term **and the half-step energy term** | $\mathbf{8.658\times10^{-3}}$ | $\mathbf{5.845\times10^{-8}}$ |

against a level of $\lvert\mathrm dE/\mathrm dt\rvert_{\max} = 4.4466\times10^{-3}$ per unit span. **$\mathcal R$ closes with the deformation term in it and does not close without it** — a factor of $115$ over the whole march and $1.7\times10^{7}$ over the settled half — which is the gate.

**And the residual that remains on the transient is the accounting rather than the bond, which is measured and not argued.** At a converged interface solve the fluid's effort is the structure's reaction at the **end** of the exchange, $f = S_e\boldsymbol\delta^{n+1}$, while the structure's energy gain over that exchange is taken at its **middle**, $S_e\bar{\boldsymbol\delta}\cdot\Delta\boldsymbol\delta$. The difference is exactly

$$\tfrac12\,\Delta\boldsymbol\delta^\top S_e\,\Delta\boldsymbol\delta ,$$

the backward-Euler-against-midpoint difference of the energy accounting, second order in $\Delta\boldsymbol\delta$ and therefore invisible once the interface is quasi-steady and worth $10\%$ where it is moving fastest. `FSIRollout.macro_step` accumulates it beside the work so it can be subtracted, and subtracting it removes $\mathbf{92\%}$ of the transient residual and $\mathbf{99\%}$ of the settled one. What is left, $8.7\times10^{-3}$ at the peak of the release, is the seam.

---

## 7. Added mass, at rank 32

A wing under aerodynamic load is the canonical partitioned-FSI added-mass problem, and **a quasi-static structure is its extreme case**: the structure has no mass at all, so the textbook fluid/structure mass ratio is infinite and *is not the variable*. What is the variable is the aeroelastic stiffness ratio

$$\mu \;=\; \rho\!\left(S_e^{-1} K_{\text{aero}}\right), \qquad K_{\text{aero}} = \frac{\partial f_{\text{aero}}}{\partial\boldsymbol\delta}\Big\vert_{\text{frozen field}} ,$$

which is $\mu = 1$ exactly at aeroelastic divergence, and which is also the amplification factor of a fixed-point iteration on the structure's own constitutive law. Measured, $\lVert K_{\text{aero}}\rVert_2 = 7.244$ and it is **exactly diagonal** — a station reads the flow at its own location, so on a frozen field moving one station does not change what another sees. (That is a property of the frozen Jacobian and not of the coupled system, where the flow responds globally.)

| $E^*$ | $\mu$ | equilibrium tip $\delta$ | inside the small-strain envelope |
|---|---|---|---|
| $5\times10^{4}$ (the reference) | $0.0772$ | $-0.02810$ | yes |
| $4\times10^{4}$ | $0.0965$ | $-0.03449$ | yes |
| $3\times10^{4}$ | $0.1287$ | $-0.04465$ | yes |
| $2\times10^{4}$ | $0.1931$ | $-0.06328$ | **no** |
| $1\times10^{4}$ | $0.3861$ | $-0.10851$ | no |
| $5\times10^{3}$ | $0.7722$ | $-0.16846$ | no |
| $4\times10^{3}$ | $0.9653$ | $-0.18917$ | no |
| $3\times10^{3}$ | $1.2870$ | $-0.21539$ | no |
| $2\times10^{3}$ | $1.9306$ | $-0.24903$ | no |

**$\mu = 1$ at $E^* = 3861$, and that boundary is outside the structural model's own validity by construction.** At divergence the equilibrium deflection is unbounded *by definition*, so a linear small-strain constitutive law is violated before the boundary is reached. That is a statement about linear aeroelastic divergence and **not** about this construction: `WingStructure.validity` declines above $\delta = 0.05$, which is $\mu \approx 0.15$, and no reformulation of the coupling moves that.

**Flutter is not reachable at all here, and the reason is structural.** Classical bending–torsion flutter needs structural inertia to have eigenfrequencies to couple; a quasi-static structure has none. Whether the *wake's* own dynamics can destabilize a massless surface is a different question and is **not measured here**.

### 7.1 The update form diverges at step 1, and the mechanism is not $\mu$

`FSIRollout` runs three couplings on identical arithmetic. At $E^* = 5\times10^{4}$, $\mu = 0.0772$ — a factor of thirteen inside the divergence boundary:

| coupling | outcome |
|---|---|
| `tight` — the interface equation solved at every exchange | **stable**, tip $-0.027939$ |
| `lagged` — solved once per macro-step and held | **stable**, tip $-0.027921$ |
| `staggered` — the structure's law read as an UPDATE, $\boldsymbol\delta \leftarrow S_e^{-1} f_{\text{aero}}$ | **left the envelope at macro-step 1**, $\lvert\delta\rvert_{\max} = 1.07$, $21\times$ the bound |

**The update form fails at a stiffness thirteen times inside the divergence boundary, so $\mu$ is not what kills it.** Priced: moving a massless structure the whole way to its own equilibrium inside one macro-step implies a surface velocity of $\mathbf{2.43\,U_\infty}$, and the fluid's response to a surface moving at $2.4$ times the freestream is a load $\mathbf{120.9\times}$ the one it was balancing. CS-10 measured $5.77\,U_\infty$ and a factor of order $70$ at a *lumped* seam.

> **The mechanism is identical and it does not depend on the partner being lumped.** [[general-coupling-scheme]] §4.2 was written from CS-10 as a rule about field↔**lumped** seams. It is a rule about seams whose partner has no mass, and a quasi-static field is one. §4.2's own repair — write the residual in the port's conjugate variables and solve it there — is what makes both surviving columns survive, and at a field↔field seam it is a $32$-square SPD Newton system rather than a scalar division, with the aerodynamic damping $\mathrm{diag}(C_N\lvert w\rvert)$ on its diagonal and $\Delta t\,S_e$ beside it. Three fixed Newton steps, and the residual is reported.

### 7.2 Loose Gauss–Seidel does not diverge anywhere the model admits

The exchange lag is the axis that *can* be pushed without leaving the constitutive envelope, so the stability map is $(E^*, \text{lag})$ rather than $(E^*)$ alone:

| $E^*$ | lag 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| $5\times10^{4}$ | $-0.027921$ | $-0.027900$ | $-0.027882$ | $-0.028049$ | $-0.029319$ | $-0.028665$ |
| $4\times10^{4}$ | $-0.034409$ | $-0.034415$ | $-0.034479$ | $-0.034911$ | $-0.036722$ | $-0.035836$ |
| $3\times10^{4}$ | $-0.045080$ | $-0.045157$ | $-0.045393$ | $-0.046242$ | $-0.048654$ | $-0.047659$ |

**Eighteen of eighteen stable.** Loose Gauss–Seidel — the interface equation solved once and its answer held for up to $32$ macro-steps, which is $128$ exchange intervals — does not diverge anywhere inside the structural expert's own declared envelope, and the tip deflection it converges to drifts by $5.0\%$ at the worst cell rather than blowing up. The answer to *"does loose coupling diverge, and at what ratio"* is therefore:

> **No, not in this construction — and the boundary it would have to cross ($\mu = 1$ at $E^* = 3861$) is one the linear structural model cannot reach, because at divergence its own small-strain hypothesis has already failed. What diverges is the UPDATE form, at the one stiffness it was run at -- thirteen times inside that boundary -- for a reason that has nothing to do with $\mu$.**

**And the two failure modes are recorded separately, because they look identical from outside a `try` block.** A run that leaves the constitutive envelope is the *expert* declining; a run that goes non-finite is the *scheme* diverging. `_try_march` classifies them, and the staggered column is labelled `envelope` — it stopped at the validity check — while the evidence that it is the *scheme* is the $120.9\times$ load, not the label.

---

## 8. The design knob, and its horizon

$E^*$ is a **design parameter and not a state**: it enters only through the structure, and the release field is the same for every value of it, which is what makes $\mathrm dJ/\mathrm dE^*$ a derivative of the composed stack rather than of its initial condition. $J$ is the settled downforce over the last quarter of the march, and the derivative is taken by reverse-mode autograd through every sub-exchange of every window's solve, through the blend, through the global Leray projection, and through the seam's own Newton solve at both ends — the same tape CS-10 pointed at $h_0$.

**The adjoint is exact and cheap.** At $N = 120$ it agrees with a central finite difference to $\mathbf{1.34\times10^{-5}}$ relative, on a truncation branch monotone over an order of step ($+8.1916\times10^{-3}$, $+8.2190\times10^{-3}$, $+8.2996\times10^{-3}$ at relative steps $10^{-2}$, $3\times10^{-2}$, $10^{-1}$) with the cancellation branch not reached, against an adjoint of $+8.1915\times10^{-3}$.

| $N$ | $J$ | $\mathrm dJ/\mathrm d\ln E^*$ |
|---|---|---|
| $20$ | $0.21070282$ | $+3.615\times10^{-2}$ |
| $40$ | $0.22393840$ | $+9.523\times10^{-3}$ |
| $60$ | $0.21881286$ | $+2.913\times10^{-3}$ |
| $80$ | $0.21590846$ | $+7.140\times10^{-3}$ |
| $120$ | $0.21290498$ | $+8.192\times10^{-3}$ |
| $160$ | $0.21021021$ | $+7.754\times10^{-3}$ |
| $240$ | $0.20988938$ | $+7.612\times10^{-3}$ |
| $320$ | $0.20924671$ | $+7.453\times10^{-3}$ |
| $480$ | $0.20835594$ | $+7.484\times10^{-3}$ |

**The three §0.4 fields, and this is the first of the three measurements where the sign does not flip:**

> $N = 480$; $\mathbf{N_{\text{sign}} \le 20}$ — **no crossing over the horizons marched**; $N_{\text{valid}}(10\%) = \mathbf{80}$, $N_{\text{valid}}(5\%) = \mathbf{160}$.

$N_{\text{sign}}$ is reported as $\le 20$ rather than $= 20$ because $20$ is the shortest horizon marched: what is measured is the *absence* of a crossing, not its location. The magnitude is non-monotone over the first three horizons — $+3.6\times10^{-2}$, $+9.5\times10^{-3}$, $+2.9\times10^{-3}$ — while the release transient runs, and settles to within $10\%$ of its long-horizon value from $N = 80$, which is about $2.7$ transient timescales.

**Why this one does not flip, and it is a hypothesis rather than a result.** CS-10's $\mathrm dJ/\mathrm dh_0$ and CS-11's $\mathrm dJ/\mathrm dU$ both change sign, and in both the knob acts on an agent with a **slow internal process** — a ride height relaxing over ninety macro-steps, a disc with a thermal time constant — whose long-horizon behaviour reverses the short-horizon answer. $E^*$ acts on a **quasi-static** structure, which has no internal timescale at all: the only clock in the loop is the fluid's.

> **[AI Inference]:** a design knob acting on an agent with no internal timescale has no horizon-driven sign change, because the only slow process in the loop belongs to the other agent and the knob does not enter it. This is one instance against two of the other kind, it is argued from the mechanism rather than measured across knobs, and the way to test it is a knob on the *fluid* side of this same seam.

**What does not change is that §0.4 still binds.** A sensitivity reported without its horizon is refused, not decertified, and *"the sign did not flip over the range I marched"* is a statement that requires the range to be stated. Read at $N = 20$ this gradient is $4.8\times$ its converged value; an optimiser stepping on it moves the right way and by nearly five times too much.

---

## 9. The compile, and the refusal that is inherited

| graph | verdict | envelope | refusals | decertifications |
|---|---|---|---|---|
| fixed shape, six windows | `admit-uncertified` | E(H H **F** H U H **F**) | **0** | 8 |
| fixed shape, one window | `admit-uncertified` | E(H H **F** H U H **F**) | **0** | 8 |
| deforming, six windows | **`refuse`** | E(H **F** **F** H U H **F**) | `L2/InterfaceMotion` on both ports | 8 |

**The deforming graph is refused at `L2/InterfaceMotion` and on nothing else new, and this case study does not repair it.** CS-10 established that the refusal is correct — no rule exists, and a moving interface silently invalidating a cached operator is the silent-wrongness class — and priced it. §4.4 above is the price on a *deforming* interface rather than a translating one.

**E3 fails at the wetted seam and $\tau$ survives it**, which is `CASE-STUDY-GUIDE`'s multiphysics box exercised for the first time on a two-way surface seam: the two sides declare `incompressible-navier-stokes-2d` and `plane-stress-elasticity-2d`, so they share no monolithic reference, and both declare `lambda_ref` — each is a real solver and its own reference — so the compile reports $\tau = 0$ rather than $\tau = \text{UNDEFINED}$. **E3's failure costs the monolithic reference and nothing else.**

**Nothing reaches `admit`**: the unmeasured list is $L$ (**W1**), $\sigma$ (**W3**) and $C_\mu$ (**W3**), and a non-empty list has forced `admit-uncertified` since **W56**. And **W116 recurs for the third time**: the FSI seam's own lag defect is a *splitting* error measured in relative tip deflection, not a *transmission* infidelity measured in interface power, and `MeasuredConstants` still has a slot for the second and none for the first — so `sigma` stays unmeasured rather than carrying a number that means something else.

---

## 10. Three framework defects the case study found without looking for them

### 10.1 W136 — the halo rule has R10's scope defect, one rule along

§2.1. `L2/R10/halo` decertifies an agent whose domain is not cut, on a seam that is a physical boundary with no overlap for anything to outrun. The predicate that fixes it is the one R10 now uses. Left open rather than fixed: a decertification that over-fires is honest and noisy, where a refusal that over-fires is terminal, and the two deserve different urgencies.

### 10.2 W137 — at a fluid–structure seam the FLUID's certificate is blind, and W97's repair does not transfer

§4.2. The fluid's block is $1.44\times10^{-5}$ of the assembled operator, so `SubstitutionCertificate` cannot see any replacement of the flow expert — and the flow expert is the one the substitution campaign exists to replace. W97 closed the same *reading* at a field-to-lumped seam by finding an effort-convention defect and repairing it; here the co-normal buys $190\times$, the exact momentum exchange buys $1.79\times10^{4}$, and $6.9\times10^{4}$ remains, uniformly across every mode of the declared interface space. **The one-sidedness is the structure's stiffness and no effort convention is wrong.**

### 10.3 W138 — $\Lambda_M$ ADDS two blocks that the interface residual SUBTRACTS

The two sides of this seam both declare `EFFORT` and both are efforts: the fluid returns the load *on* the surface, the structure returns the reaction it takes to *hold* it. The interface residual is therefore their **difference** — `FSIRollout.solve_interface` writes $S_e(\boldsymbol\delta + \Delta t\,w) - f_{\text{aero}} = 0$, and CS-10's scalar version writes $k(h_0 - h - \Delta t v) - L$. But $\tilde\Lambda_M = \sum_i P_i^*\Lambda_i P_i$ **adds** them, and `L4/E7/passivity` reads the symmetric part of that sum.

`Connection.orientation` exists for exactly this — *"normal points from the first-named agent to the second"* — and **nothing reads it**.

**Measured, and it is decisive rather than suggestive.** The symmetric part of the assembled block has $\lambda_{\min} = -2.1661$, a passivity defect of $2.1661$; the same two blocks with the fluid's sign flipped give $\lambda_{\min} = +5.0537$ and a passivity defect of **exactly $0$**. So E7's failure at this seam is *entirely* an orientation artefact, not a mode being amplified — a false alarm of the same class `CASE-STUDY-GUIDE` mistake 7 is about, on the other declaration, and it has stood at CS-10's seam too.

---

## 11. What this case study cannot say

- **One structural model, and it is a model.** A $32\times2$ Q1 plane-stress cantilever two elements through the thickness. Q1 elements lock in bending, so the plate is stiffer than its own beam theory; $E^*$ is calibrated to a deflection rather than taken from a material, which is legitimate because $E^*$ is a *declared design knob in flow units* — and which is why **no stress figure here is a statement about an alloy**.
- **One wing model, and it is CS-10's.** The porous inclined plate has thickness and an exact discrete momentum exchange, and it has no Kutta condition, no bound circulation and no boundary layer on its own surface.
- **One Reynolds number, one grid, one incidence.** $\mathrm{Re}_c = 125$, $\mathrm{Re}_h = 3.906$, $208\times144$ cells, $20^\circ$. Nothing here is a grid-convergence study.
- **The divergence boundary is computed, not marched.** $\mu = 1$ at $E^* = 3861$ comes from the frozen-field Jacobian and the surface stiffness, and the marches that would confirm it are outside the structural expert's declared envelope. Its proportionality to $1/E^*$ is arithmetic rather than a measurement, since $S_e$ is exactly linear in $E^*$; what is *measured* is $K_{\text{aero}}$, and what is *observed* is that the envelope is reached first.
- **Loose coupling is stable over the map that was run**, which is three stiffnesses and six lags at one operating point. **And the update form was run at ONE stiffness**, so the claim that it fails for a reason unrelated to $\mu$ rests on the priced mechanism -- an implied surface velocity of $2.43\,U_\infty$ and a $120.9\times$ load -- and on CS-10's independent instance at a lumped seam, not on a sweep. Neither result is a proof: the map's own worst cell already drifts $5\%$ in the tip deflection.
- **$\mathcal R$ is the structure's own balance and not a global residual.** CS-9 §6 measured a global $\mathcal R$ blind to the coupling under test by six orders, and this page follows that rule rather than re-testing it.
- **No learned expert ran.** The substitution campaign prices R10 per seam and this seam has not been priced — and §4.2 is the reason it will be hard to.
- **Nothing reaches `admit`.** $L$ is unmeasured (**W1**), $\sigma$ and $C_\mu$ are unmeasured (**W3**, **W49**), and a non-empty `unmeasured` list forces `admit-uncertified` since **W56**.

---

## 12. Rows this closes and opens

| row | outcome |
|---|---|
| **W114** | **closed**, by the premise check the row's own *done when* asks for: *"R10 consults whether any agent's own domain is actually cut."* The predicate is a proxy — another agent of the same `governing_family` — and it is conservative, and it is checked by a control that still refuses. **Four graphs' verdicts move and no measurement in the vault moves with them**: `thermal_strain` `surface-mech`, `brake_thermal` `as-built`, `thermal_seam` `as-built` and this one, all multiphysics graphs where nothing was cut. `window_ns` `as-built` — four windows of one family — keeps its refusal, which is the row that says the narrowing is right |
| **W136** | **opened.** `L2/R10/halo` has R10's scope defect one rule along — §2.1, §10.1 |
| **W137** | **opened.** At a fluid–structure `MECH` seam the fluid's block is $1.44\times10^{-5}$ of the assembled operator and W97's repair does not transfer, because the one-sidedness is the structure's stiffness rather than an effort convention — §4.2, §10.2 |
| **W138** | **opened.** $\tilde\Lambda_M$ adds two blocks the interface residual subtracts; `Connection.orientation` exists for it and nothing reads it — §10.3 |
| **W123** | **recurs on a fourth quantity, with the sign the other way.** The lag defect is first order at its peak and on the settled half, and *third* order at the last macro-step, because the columns re-converge. Every previous instance had the rollout accumulating *more* than the per-interval quantity predicts |
| **W106** | **applied as a discipline.** The bitwise floor here is exactly zero (asserted) and nothing is quoted against it; the level is the plate's own settled unsteadiness, $1.503\%$ |
| **W124** | **honoured before the fact, on both axes.** The tiling places every cut $8$ cells clear of the plate in $x$ and $12$ in $y$, and a test asserts both — CS-10 needed only the first |
| **W130** | **pinned again.** A test asserts the probed fluid block at the wetted seam is not identically zero, because a window that cannot see the plate produces a block byte-identical to a genuinely empty interface problem |
| **W132** | **reproduced, not confirmed.** The exact momentum exchange is $1.79\times10^{4}$ times §2.2's viscous effort here against $1.84\times10^{4}$ at CS-10 — the same arithmetic on the same plate model, which is a check on the implementation and not evidence about other seams |

## See Also

- [[case-study-ladder-to-f1]] — §4's CS-12 row, which scheduled this; §10 and §12, which both said W114 had to be settled first
- [[f1-pathmap-and-end-goal]] — rung 5, and §5's F5
- [[gap-worklist]] — Tier 26: W114 closed, W136–W138 opened
- [[case-study-ground-effect-atlas-0.1]] — the fluid column this reuses unchanged, the moving interface, and the field-to-lumped seam this is the field-to-field version of
- [[case-study-thermal-strain-atlas-0.1]] — where W114 was opened, and the free-body row of $n_0(\Gamma)$ this adds a clamped one beside
- [[case-study-brake-thermal-atlas-0.1]] — the other graph whose R10 refusal turns out to have been false
- [[general-coupling-scheme]] — §4.2, whose scope this widens from field↔lumped to any massless partner
- [[end-to-end-architecture-spec]] — §0.4's horizon rule, and L2/R10's own text
- [[port-algebra-atlas-0.1]] — the `MECH` bond, used between two fields for the first time
