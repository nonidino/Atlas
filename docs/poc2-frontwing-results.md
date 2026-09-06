# PoC 2 — Results: the front-wing assembly, and a constrained design search on it

**Type:** Concept page — **results record**, the companion to [[poc1a-frozen-expert-results]] (folder: `Atlas 0.1/common/`)
**Status:** measured **2026-09-04 and 2026-09-05** — the design search, the cost measurement and the ablation were all re-run past midnight after **W145** was found, and both dates are on the record rather than one of them. Every number below is quoted from `out/w141/w141.json`; nothing is estimated. This is a **capability demonstration**, not a rung of the [[case-study-ladder-to-f1]] ladder — and it is an **assembly** of two rungs that already exist rather than a fourteenth case study.
**Code:** `atlas/cases/front_wing.py` (the assembly), `scripts/w141_poc2_frontwing.py` (the driver, twelve stages), `atlas/demo_frontwing/` (the live demo), `tests/test_tier30_front_wing.py`
**Related:** [[case-study-ground-effect-atlas-0.1]] · [[case-study-wing-fsi-atlas-0.1]] · [[poc1a-frozen-expert-results]] · [[poc1-results-differentiable-design]] · [[gap-worklist]] · [[case-study-ladder-to-f1]] · [[general-coupling-scheme]] · [[end-to-end-architecture-spec]] · [[port-algebra-atlas-0.1]] · [[f1-pathmap-and-end-goal]]

---

## 0. What this is, and the one thing it must not be read as

[[case-study-ground-effect-atlas-0.1]] built a wing that **rides** — a ride height on a spring over a moving floor, and a plate that cannot bend. [[case-study-wing-fsi-atlas-0.1]] built the same wing **bending** — a quasi-static elastic cantilever under aero load, on a mount that cannot move. Each is a real case study with its own gate, and each left the other degree of freedom out.

**PoC 2 turns both on at once.** One graph, eight agents, nine seams, and **two surface seams live simultaneously on the same plate**: a field-to-lumped `MECH` carrying the wing's *position* and a field-to-field `MECH` carrying its *shape*. The whole rollout is differentiated in four design knobs and a **constrained** search is run on it — maximise downforce subject to a stress ceiling and a deflection ceiling — against a population baseline, as [[poc1a-frozen-expert-results]] did for the unconstrained layout problem.

**It is an assembly and the word is load-bearing.** Nothing about the fluid column, the wing model, the structural expert or the suspension was rewritten:

| | where it comes from | what changed |
|---|---|---|
| the fluid column | `ground_effect`, through `wing_fsi` | **nothing.** Six exposed `reference.WindowNS` windows of $80\times80$ at halo $16$ over $208\times144$ cells, `ProjectedAssembly`, one global spectral Leray projection per exchange |
| the composition layer | `wing_fsi.FSIRollout` | **nothing.** `FrontWingRollout` *subclasses* it, so `cut`, `blend`, `project` and `band` are inherited rather than copied — the only way to say "unchanged" credibly is to not have a second copy |
| the wing | `wing_fsi.FlexWing` | one **optional** argument: the mount height. Omitting it is asserted **bitwise** identical to passing the declared one |
| the structure | `wing_fsi.WingStructure` | the thickness promoted from a module constant to a field, and a **stress map** added. At the declared thickness the operator is CS-12's to the digit |
| the suspension | `ground_effect.Suspension` | **nothing** |

**What this must not be read as saying.** No stress figure here is a statement about an alloy, no verdict here is an `admit`, and the demo's per-seam lights do not change when you turn a knob — §4 measures that and says so. The one claim being made is that **two independently built couplings compose, the composition reduces to each of them exactly, and a gradient runs through both at once.**

---

## 1. The assembly's own gate: each parent is a limit, measured

A composition that does not reduce to its parts is not an assembly, so this is checked before anything else is measured.

The two seams' `boundary_response`s are balanced in the ports' own conjugate variables — [[general-coupling-scheme]] §4.2's rule with two seams instead of one. The unknowns are the two ports' **flow** halves: $\dot{\boldsymbol\delta}$, the wetted surface's normal velocity relative to the mount ($32$ of them), and $\dot h$, the mount's vertical velocity (one). With $w_{\text{plate}} = \dot{\boldsymbol\delta} + \dot h\,n_y$ the surface's total normal velocity, $w = w_{\text{ext}} - w_{\text{plate}}$ the relative flow and $\mathbf f = \tfrac12 C_N\lvert w\rvert w$ the aero traction,

$$\mathcal R_1 = S_e\big(\boldsymbol\delta + \Delta t\,\dot{\boldsymbol\delta}\big) - \mathbf f, \qquad \mathcal R_2 = k\big(h_0 - h - \Delta t\,\dot h\big) - L, \qquad L = -\textstyle\sum_k f_k\,n_y\,\mathrm ds,$$

both of the form *reaction minus load*. The Jacobian is analytic and **one field evaluation serves the whole system**, for CS-10's reason carried to two seams — $w_{\text{ext}}$ does not depend on either unknown — so with $g = C_N\lvert w\rvert$,

$$\frac{\partial\mathcal R_1}{\partial\dot{\boldsymbol\delta}} = \Delta t\,S_e + \operatorname{diag}(g), \quad \frac{\partial\mathcal R_1}{\partial\dot h} = g\,n_y, \quad \frac{\partial\mathcal R_2}{\partial\dot{\boldsymbol\delta}} = -g\,n_y\,\mathrm ds, \quad \frac{\partial\mathcal R_2}{\partial\dot h} = -k\Delta t - \textstyle\sum_k g_k n_y^2\,\mathrm ds,$$

and the $33$-square system is assembled and solved directly, three Newton steps at a **fixed** count so a reverse-mode tape has the same shape at every design.

**Both limits are measured and both are exact to the solve's own round-off.**

| control | what it must reproduce | measured |
|---|---|---|
| $k \to \infty$ — the mount is held | `wing_fsi.FSIRollout.solve_interface`, the $32$ equations CS-12 published | $\lVert\dot{\boldsymbol\delta} - w_{\text{plate}}^{\text{CS-12}}\rVert_\infty / \lVert w\rVert_\infty = \mathbf{3.37\times10^{-10}}$, with $\dot h = 4.1\times10^{-11}$ |
| $S_e \to \infty$ — the plate is rigid | CS-10's scalar Newton step, **including** its analytic derivative $-k\Delta t - C_N\sum\lvert w\rvert\cos^2\alpha\,\mathrm ds$ | $\lvert \dot h - v_{\text{plate}}^{\text{CS-10}}\rvert / \lvert v\rvert = \mathbf{6.36\times10^{-8}}$, with $\lVert\dot{\boldsymbol\delta}\rVert_\infty = 9.7\times10^{-10}$ |
| the stress map against the expert | `solve_mechanical`'s own answer | $\mathbf{1.79\times10^{-13}}$ relative |
| the zero-cut control | the composed path at one window, twice | **bit-identical** — a control and **not** a floor (**W106**) |

The second control is written out in the driver and in the test rather than shared with `ground_effect`, so what is compared is the **equation** and not a call into the same helper. CS-10's wing carries $33$ stations at the chord's endpoints and CS-12's carries $32$ at element midpoints, so a bitwise comparison against `GroundRollout` is not available and is not claimed; what is asserted is that the assembled system's scalar block *is* CS-10's, on CS-12's stations.

---

## 2. The design box, and where its bounds come from

Four knobs — wing stiffness $E^*$, plate thickness $t/c$, spring rate $k$, free ride height $h_0$ — and **every bound is a measurement rather than a preference.**

$E^*$'s lower bound is where the structural expert stops answering: CS-12 §7's table has an equilibrium tip of $-0.0633$ at $E^* = 2\times10^{4}$ against `WingStructure.validity`'s bound of $0.05$, and CS-12's own verification pass found the $E^* = 3\times10^{4}$ cell reaching $0.049822$ of that bound on its release transient. $h_0$'s box sits inside CS-10's own measured ground-effect curve, which runs $0.075$ to $0.75$ monotone with no turnover.

**The thickness bound is the interesting one, and it is a defect of the element rather than of the wing.**

| $t/c$ | tip deflection | peak von Mises | $\lVert S_e\rVert_2$ | $\lambda_{\min}(S_e)$ |
|---|---|---|---|---|
| $0.020$ | $1.194\times10^{-1}$ | $306.9$ | $1.532\times10^{10}$ | $6.13$ |
| $0.026$ | $9.890\times10^{-2}$ | $\mathbf{325.5}$ | $1.298\times10^{10}$ | $7.40$ |
| $0.032$ | $7.334\times10^{-2}$ | $295.2$ | $1.180\times10^{10}$ | $9.98$ |
| $0.040$ | $4.687\times10^{-2}$ | $234.2$ | $1.106\times10^{10}$ | $15.62$ |
| $0.050$ | $2.738\times10^{-2}$ | $173.7$ | $1.074\times10^{10}$ | $26.74$ |
| $0.064$ | $1.427\times10^{-2}$ | $116.9$ | $1.074\times10^{10}$ | $51.29$ |
| $0.080$ | $7.737\times10^{-3}$ | $78.99$ | $1.102\times10^{10}$ | $94.54$ |

**Three readings, and the first one is a trap the norm sets.**

- **$\lVert S_e\rVert_2$ is non-monotone in the thickness and every response quantity is monotone.** Its power-law exponent runs $-0.630, -0.459, -0.291, -0.134, +0.001, +0.118$ — it *rises* with thickness above $t/c = 0.05$. The norm is set by the **stiffest** mode, which on a two-element-deep Q1 mesh is a locked shear mode; $\lambda_{\min}$, the bending end, rises monotonically by $15.4\times$ across the range. **A design search that read the operator norm would have optimised an element formulation.**
- **The tip deflection has the physical sign everywhere and the wrong magnitude.** Its exponent runs $-0.720$ at the thin end to $-2.743$ at the thick end against beam theory's $-3$: the Q1 elements lock in bending, the locking is worse for more slender elements, and the model approaches beam theory from above as the elements fatten.
- **The peak stress INVERTS below $t/c \approx 0.026$.** Its exponent is $\mathbf{+0.224}$ on $[0.020, 0.026]$ — thicker is *more* stressed, which is the wrong way — and $-1.038$ to $-1.757$ above it against beam theory's $-2$. **The design box's lower thickness bound is placed at $0.032$, above the inversion**, and a search run through it would be optimising the element.

### 2.1 The thickness's upper bound is the flow model's own neglect, and a search found it

**The first run of §9 was given $t/c \le 0.080$ and drove straight to it**, with the optimum on all four box bounds and **both ceilings inactive** — stress $86$ of $600$, deflection $0.0019$ of $0.040$. That is a box search wearing the word "constrained", and the cause is physical: **stiffening and thickening the wing raise the downforce *and* lower both the stress and the deflection**, so the objective and both constraints pull the *same* way on the two structural knobs. There is no trade to find.

The reason the thickness is free is on CS-12's own page: **the porous plate has no thickness in the flow.** It is a line of stations, so a thicker plate blocks no more gap and adds no drag — it only gets stiffer. A knob with a benefit and no cost goes to its bound, and where that bound sits is then the whole answer.

> **So the bound has to come from the flow model rather than from the search.** A plate the grid would have to resolve is a plate the flow model may not neglect. At $\mathrm dx = 1/64$ on a chord of $0.5$, $t/c = 0.046$ is $\mathbf{1.47}$ cells and $t/c = 0.080$ is $\mathbf{2.56}$. The plate may be about a cell thick and not three, so the box closes at $0.046$ — **and the first run's optimum is therefore outside the flow model's scope rather than merely at a box edge.** Both runs are kept in the artifact; the loose one is `design_loose`.

$$\text{box}: \quad E^* \in [3\times10^{4},\, 2\times10^{5}], \quad t/c \in [0.032,\, 0.046], \quad k \in [1.5,\, 6.0], \quad h_0 \in [0.20,\, 0.45].$$

### 2.2 The two ceilings, set where they bind

A stress is set by the **load** and the **geometry**: raising $E$ at a fixed traction shrinks the displacement and the strain in the same proportion and leaves $D\varepsilon$ where it was, so the stress ceiling barely sees $E^*$ at all. A deflection is set by the load and the **bending stiffness**, so the deflection ceiling sees both.

**A ceiling the search never reaches is not a constraint**, and the first run's pair were never reached. Measured over the structural box at the aggressive ride height the search chooses, the *reachable* ranges are $\mathbf{85}$ to $\mathbf{308}$ in the stress and $\mathbf{0.0019}$ to $\mathbf{0.102}$ in the deflection. So:

| | value | why there |
|---|---|---|
| $\sigma_{\max}$ | $\mathbf{200}$ | inside the reachable range and **above** the reference design's own $165.4$, so the start is feasible and the downforce-seeking direction runs into it |
| $\lvert\delta\rvert_{\max}$ | $\mathbf{0.035}$ | **above** the reference design's own $0.0304$ and **below** what the reference *structure* gives at the aggressive ride height ($0.0432$), so it was expected to be what stops the ride height coming down until the structure is stiffened |

> **The second of those two expectations did not survive the search, and the row above is left as it was written.** §9.1 measures the deflection margin at the optimum at $-0.673$ — a third of the ceiling, never approached. The reasoning was right about the *reference structure* at the aggressive ride height and wrong about the *path*: the search stiffens and thickens **before** it drops the ride height, because both of those raise the downforce too, so by the time $h_0$ comes down the deflection has already fallen out of reach of its ceiling. **A ceiling placed by reasoning about the endpoint can be missed by a search that takes a different route to it**, and only one of the two placed here binds.

`DELTA_CEIL` sits **inside** `WingStructure.validity`'s own $0.05$ deliberately: a search driven onto the deflection constraint must not simultaneously drive the expert out of its envelope, or the two failures are indistinguishable from outside a `try` block (CS-12 §7.1's rule).

**None of this is a statement about an alloy**, and `front_wing.THICKNESS_SCOPE` says so in the module.

---

## 3. The spin-up, and the level everything is quoted against

$120$ fixed-shape, fixed-height macro-steps at the reference design — $E^* = 5\times10^{4}$, $t/c = 0.04$ (CS-12's), $k = 2.5$, $h_0 = 0.32$ (CS-10's), released at the quasi-static equilibrium $h = h_0 - L_{\text{ref}}/k = 0.2291$.

| | |
|---|---|
| settled load | $0.238574$ |
| settled unsteadiness over the last quarter | $\mathbf{0.562\%}$ peak-to-peak — **the LEVEL** |
| $u_{\max}$ | $1.3116$ |

**The release height is a refusal rather than a preference.** At $h = h_0$ the spring carries nothing and the wing falls under the whole load from rest; a release *above* $h_0$ puts a linear compression spring in tension, which is what `Suspension.validity` declines and which is how the first version of CS-10 was caught. Releasing where the spring already carries the load it will settle at makes the transient the *coupling* adjusting rather than a body being dropped.

**And it is a consistency check on the assembly for free.** CS-10's own measured ground-effect curve gives $L = 0.2401$ at $h = 0.23$; this assembly settles at $0.2386$ at $h = 0.229$, a difference of $0.6\%$ — on a different station layout, a different spin-up and a bending plate. Not a control and not quoted as one; it is the two constructions agreeing to within the plate's own unsteadiness.

**W124 is honoured on both axes and over the whole design box, not just at the reference design.** As $h_0$ runs $0.20$ to $0.45$ the plate spans rows $12$–$40$; the $y$-overlap band starts at row $64$, clear by $\mathbf{24}$ cells. In $x$ the plate spans cells $[88,119)$ and the overlap bands are clear by $\mathbf{8}$. A test asserts both.

---

## 4. Every seam's own verdict — and the thing the demo must not overstate

This is the part [[poc1a-frozen-expert-results]] has no analogue for, and it is the demo's reason to exist.

`compile_scheme` already produces a per-decision `subject` — an agent, a seam id, or an `AGENT.PORT` pair — and **nothing in this package had ever grouped them by seam.** A decision reaches a seam when its subject *is* the seam, or is one of the two ports the seam pairs, or is one of the two agents it joins; graph-level subjects (`<graph>`, `<assembly>`, `<run>`) reach every seam. One function does the grouping and the driver and the demo share it, asserted equal by a test.

| graph | verdict | `wet` | `mount` | the seven fluid–fluid |
|---|---|---|---|---|
| fixed shape | `admit-uncertified` | **amber** — `L2/R10/halo`, `L4/E7/passivity` | **amber** — `L2/R10`, `L4/E7/passivity` | **amber** (graph-level only) |
| riding | **`refuse`** | **red** — `L2/InterfaceMotion` | **red** — `L2/InterfaceMotion` | **amber** |

**The two surface seams are red for the same rule and amber for different ones**, which is the panel earning its place: `wet` carries `L2/R10/halo`, which **W136** says over-fires there because that seam is a *physical* boundary of the solid with no overlap for anything to outrun; `mount` carries `L2/R10`, CS-10's, unchanged. Both carry `L4/E7/passivity`, which **W138** measured to be an **orientation** artefact on this exact seam pair.

**Nothing is green, and the reason is not about this graph.** $L$ (**W1**), $\sigma$ and $C_\mu$ (**W3**, **W49**) are unmeasured, and a non-empty `unmeasured` list has forced `admit-uncertified` on every graph in this package since **W56**.

### 4.1 W114's narrowing survives the third agent, and that had to be checked

`L2/R10` refuses an `EMBEDDED` agent only when **another agent shares its `governing_family`**. Adding a third agent is exactly the kind of change that could re-trigger a rule that counts agents per family. It does not, because `Suspension` declares the **flow's** family — correctly, since an algebraic closure inside a continuum problem is not a different continuum problem — so `STRUCT` remains the sole `plane-stress-elasticity-2d` agent and the premise check still clears it. **The assembled riding graph compiles with the refusals `L2/InterfaceMotion` and nothing else**, and the control still fires: declare the structure with the fluid's family and `L2/R10` comes straight back.

### 4.2 The colour is a property of the declaration, not of the design point

**Measured, because a live indicator that animates implies otherwise.** Over the **sixteen corners** of the design box, **zero** produce a different per-seam verdict map.

> So the lights change when the interface is *declared* to move, and they do **not** flicker as a knob turns. The demo re-runs the compile on every design change anyway, on its own thread, because the honest way to show that is to run it rather than to assume it — and the page says which of the two the viewer is seeing.

What *does* move with the design is the quantity under each light — each seam's one-sidedness ratio, the interface residual, and the two constraint margins — and that is what the demo puts there. **[AI Inference]:** the reason the verdict is design-independent here is that every rule that fires reads a **declaration** (`motion_class`, `elliptic_subsolve`, `time_discretization`, `stencil_radius`) or a **structural** property of the graph, and none of the four knobs changes any of them. A knob that moved an agent *out of its declared validity envelope* would change the verdict, and none of the sixteen corners does — which is a property of the box being drawn inside the experts' envelopes on purpose (§2), not a general fact about designs.

---

## 5. The gate: does the split reproduce the referent, in **both** degrees of freedom?

**Yes, and the seams beat the cut by two orders on one degree of freedom and by nearly three on the other.**

Released from the settled fixed-shape field and marched $480$ macro-steps — twice CS-12's gate, because §6's own row (**W140**) says a quantity read off a rollout is quoted with its horizon and the cut's defect had not peaked at $240$. The referent is the *same code path* at a one-window tiling, so referent and composed column differ **by the cut and by nothing else**.

| column | $\delta_{\text{tip}}$ at $480$ | $h$ at $480$ | $\max\lvert\Delta\delta\rvert/\delta_{\text{settled}}$ | at | crossings | $\max\lvert\Delta h\rvert/h_{\text{settled}}$ | at | crossings |
|---|---|---|---|---|---|---|---|---|
| referent (tight, one window) | $-0.029101545$ | $0.232592818$ | — | — | — | — | — | — |
| lag 1, no cut | $-0.029101734$ | $0.2325921$ | $5.427\times10^{-2}$ | $3$ | **9** | $2.334\times10^{-3}$ | $4$ | **2** |
| lag 2, no cut | $-0.029102012$ | $0.2325911$ | $1.233\times10^{-1}$ | $4$ | **7** | $5.405\times10^{-3}$ | $4$ | **2** |
| lag 4, no cut | $-0.029102569$ | $0.2325890$ | $2.363\times10^{-1}$ | $5$ | **7** | $1.042\times10^{-2}$ | $5$ | **2** |
| lag 1, six windows | $-0.029099112$ | $0.2325972$ | $5.427\times10^{-2}$ | $3$ | **6** | $2.333\times10^{-3}$ | $4$ | **3** |
| **tight, six windows** | $-0.029098922$ | $0.2325980$ | $\mathbf{9.013\times10^{-5}}$ | $479$ | $1$ | $\mathbf{2.209\times10^{-5}}$ | $479$ | $0$ |

**Four things.**

**Both seams' lag defects are first order in the lag, independently.** $\log_2$ ratios of the peak: $1.184, 0.938$ in the deflection and $1.212, 0.947$ in the ride height. That is the **fourth and fifth** instance of a first-order lag defect in this vault — CS-9's volumetric splitting error, CS-10's field-to-lumped lag, CS-11's multirate lag, and now both of this graph's seams — and the first time two of them are measured *on the same rollout*, so the order is not a property of one seam's physics.

**The seams dominate the cut on both degrees of freedom, and by different factors.** In the deflection the lag alone is worth $5.427\times10^{-2}$ against the whole six-window cut's $9.013\times10^{-5}$ — a factor of $\mathbf{602}$; in the ride height, $\mathbf{106}$. CS-10 measured $97\times$ at its field-to-lumped seam and CS-12 $237\times$ at its field-to-field one.

> **The ride height's $106$ is CS-10's $97$ to within ten per cent, on a graph where the wing now also bends. The deflection's $602$ is $2.5\times$ CS-12's $237$, and that is not the horizon** — at $240$ macro-steps this graph gives $616$, so the difference is the graph and not the march. **[AI Inference]:** the most likely cause is that the wetted seam's lag now also lags the *ride height*, so a held interface answer goes stale in two ways instead of one; the measurement that would separate them is this gate with the mount frozen, which §7.2's `no suspension` column is at one lag only and not across the lag sweep.

> **The fourth coupling kind and the fourth time the seam is where the error is and the domain cut is not** — and the first graph in which two different seam kinds are measured against *one* cut, so the comparison is not confounded by the decomposition changing between them.

**The crossing was looked for and it is there, more of it at the longer horizon.** The lag-1 column crosses the referent's tip **nine** times over $480$ macro-steps against six over $240$ — the plate sheds and the difference oscillates through zero with it — while the pure-cut column crosses once in the tip and **not at all** in the ride height. That asymmetry is the reason the cut's defect is a clean number and the lag's is not.

**And the cut's own defect has NOT peaked, at either horizon.** Its maximum is at step $479$ of $480$ and at step $239$ of $240$: it grew from $8.737\times10^{-5}$ to $9.013\times10^{-5}$, which is $+3.2\%$ for a doubling of the march. **So $9.0\times10^{-5}$ is a value at $480$ macro-steps and not a bound** — CS-10's situation, not CS-12's, whose cut defect peaked at step $68$ of $240$. It is reported as a value, with both horizons, and the growth rate is the evidence about how much a longer march would move it.

---

## 6. $\mathcal{R}$ across **both** seams at once

CS-9 §6's rule — *the right residual is the receiving subsystem's own balance* — with **two** receivers. The energy is the structure's strain energy **plus** the spring's, and the power is the interface power the two seams share:

$$\frac{\mathrm d}{\mathrm dt}\Big[\underbrace{\tfrac12\,\boldsymbol\delta^\top S_e\,\boldsymbol\delta\,\mathrm ds}_{\text{structure}} + \underbrace{\tfrac12 k (h_0-h)^2}_{\text{spring}}\Big] \quad\text{against}\quad \int_\Gamma f_n\,w_n\,\mathrm ds, \qquad w_n = \dot{\boldsymbol\delta} + \dot h\,n_y .$$

| accounting | whole march ($480$ macro-steps) | Q2 | Q3 | **Q4** |
|---|---|---|---|---|
| **without** the motion terms — a static accounting | $\mathbf{1.0000}$ | — | — | — |
| **with** them | $\mathbf{5.931\times10^{-2}}$ | $1.136\times10^{-6}$ | $3.323\times10^{-8}$ | $\mathbf{2.963\times10^{-8}}$ |
| with the half-step terms as well | $3.305\times10^{-2}$ | $1.544\times10^{-8}$ | $1.559\times10^{-8}$ | $\mathbf{1.523\times10^{-8}}$ |

against a level of $\lvert\mathrm dE/\mathrm dt\rvert_{\max} = 2.595\times10^{-2}$ per unit span. **$\mathcal R$ closes with the motion terms in it and does not close without them** — a factor of $\mathbf{17}$ over the whole march and $3.4\times10^{7}$ over the last quarter — **and the ordering was checked step by step: $\lvert\mathrm dE/\mathrm dt - P\rvert \ge \lvert\mathrm dE/\mathrm dt\rvert$ at $\mathbf 0$ of $480$ steps**, worst instantaneous ratio $0.447$ at step $8$. That is the gate.

### 6.1 Two things the assembly got wrong first, and both were found by the residual

**The spring's energy at release was missing, and it is the largest number in the balance.** The plate starts flat, so the structure's strain energy at release is identically zero — which makes it easy to carry one initial value and not the other. The suspension, by construction, is *already carrying the load* at release. Taking the spring's energy **after** the first macro-step as its value **before** put the first step's residual at $\mathbf{1.81}$ against a static accounting's $1.00$: the gate **failed**, on a march whose every other step closed to $5\times10^{-8}$. The tell was that the failure was entirely in the first quartile.

**And the half-step term belongs to both receivers.** At a converged interface solve each agent's effort is its reaction at the **end** of the exchange while its energy gain is taken at the **middle**, and the difference is second order in the increment: $\tfrac12\Delta\boldsymbol\delta^\top S_e\Delta\boldsymbol\delta\,\mathrm ds$ for the structure and $\tfrac12 k\,\Delta h^2$ for the spring, with the **same sign**. CS-12 accumulated the first, because it had one receiver. Carrying it without the second would correct one agent's accounting and not the other's on a balance that adds them — and with both carried the correction halves the settled residual ($2.96\times10^{-8} \to 1.52\times10^{-8}$) where CS-12's own verification pass found a single-receiver correction doing nothing on its settled tail (**W140**).

---

## 7. The ablation: is the coupling *between* the two seams doing any work?

[[poc1a-frozen-expert-results]] §6's question, on the axis this assembly adds. Two ablations, at the reference design over $120$ macro-steps.

### 7.1 Solving the two seams **together** against solving them **in turn**

| scheme | settled load | $h$ | $\delta_{\text{tip}}$ | interface residual |
|---|---|---|---|---|
| **joint** — both seams, one $33$-equation system | $0.22632070$ | $0.2294717$ | $-0.0304010$ | $\mathbf{1.905\times10^{-8}}$ |
| **split** — the two seams solved in turn, Gauss–Seidel over the seams | $0.22631880$ | $0.2294725$ | $-0.0303996$ | $\mathbf{8.119\times10^{-5}}$ |

**The answer is: almost none in the state, and four orders in the residual.** The two schemes' settled loads differ by $8.4\times10^{-6}$ relative — well below the plate's own $0.562\%$ settled unsteadiness, so on this graph *the state cannot tell them apart*. What separates them is the **interface residual**, by $\mathbf{4262\times}$: the joint solve drives both seams' balance to $1.9\times10^{-8}$ and the split one leaves $8.1\times10^{-5}$ after the same three Newton sweeps.

> **So the joint system is not buying accuracy here — it is buying a residual you can quote.** That is worth stating plainly rather than dressing up: at this operating point the two seams are weakly coupled to each other (the structure responds to the traction, the spring to its resultant, and neither sees the other except through the flow), so a Gauss–Seidel sweep over them converges nearly as well as a monolithic solve. **[AI Inference]:** the seams' mutual coupling should strengthen as the deflection becomes a larger fraction of the ride height — a wing that bends by as much as it rides changes its own gap — and this graph's reference design has $\delta_{\text{tip}}/h = 0.13$. The measurement that would settle it is this same ablation at the aggressive end of the design box, and it has not been run.

### 7.2 Each seam frozen in turn — what the assembly buys over either parent

| column | settled load | $h$ | $\delta_{\text{tip}}$ |
|---|---|---|---|
| **no suspension** (CS-12 alone, $k \to \infty$) | $0.22521116$ | $0.2294717$ | $-0.0301902$ |
| **no structure** (CS-10 alone, $E^* \to \infty$) | $\mathbf{0.23556656}$ | $0.2257734$ | $-1.6\times10^{-6}$ |
| **both seams live** | $0.22632070$ | $0.2294717$ | $-0.0304010$ |

**At the reference design the elastic seam is worth $4.1\%$ of the downforce and the suspension seam $0.49\%$ — an eightfold asymmetry.** Freezing the structure ($E^*\to\infty$) *raises* the load by $\mathbf{4.1\%}$ — a rigid wing keeps its incidence and its gap blockage instead of bending away from the load — and drops the ride height by $1.6\%$, because the extra load compresses the spring further. Freezing the suspension ($k\to\infty$) costs $\mathbf{0.49\%}$, and it also changes the deflection, by $0.7\%$: each freeze moves the operating point the *other* seam sits at.

> **The frozen column is pinned at the height the LIVE design settles to**, not at the reference-load release height $h_0 - L_{\text{ref}}/k$. The two agree to $4\times10^{-4}$ here and the first version of this table used the second; at the *optimum* below they differ by $37\%$, and the difference would have been read as the seam's worth. A control that sits at a different operating point is measuring the transit and not the seam.

> **That is the assembly earning its place: a model with the elastic seam frozen is wrong about the downforce by $4.1\%$ against a level of $0.562\%$ — seven times the plate's own unsteadiness — and one with the suspension frozen is wrong by $0.49\%$, which is at the level and is the honest reason CS-12 could leave the ride height out and CS-10 could not leave the structure out.**

### 7.3 The same ablation at the optimum — and the inference it was added to remove was wrong

An ablation at the reference design is a statement about the reference design. An earlier draft of §7.2 said that a search which stiffens the wing shrinks the elastic seam's worth — an **[AI Inference]** about a design point the ablation had not been run at. It costs three rollouts to stop inferring it, so it was measured. **The measurement went the other way, and on the other seam.**

| at $E^* = 9.013\times10^4$, $t/c = 0.04355$, $k = 1.5$, $h_0 = 0.2569$ | settled load | $h$ | $\delta_{\text{tip}}$ |
|---|---|---|---|
| **both seams live** | $0.27005497$ | $0.0768233$ | $-0.0177988$ |
| **no suspension** ($k \to \infty$, pinned at $h = 0.0768233$) | $0.24802409$ | $0.0768233$ | $-0.0161275$ |
| **no structure** ($E^* \to \infty$) | **declined** — the suspension expert, at macro-step $40$, $h = 0.0702$ | — | — |

> **The suspension seam goes from $0.49\%$ of the downforce at the reference design to $\mathbf{8.16\%}$ at the optimum — a factor of $17$.** The optimum rides at $h = 0.0768$ against the reference's $0.2295$, and in deep ground effect $\mathrm dL/\mathrm dh$ is far larger, so freezing the ride height at the same height still costs a great deal: what is frozen is the seam's *response*, and the response is what carries the load fluctuation at a small gap. **The seam that mattered least at the reference design is the one that matters most at the optimum**, which is the opposite of what §7.2's inference asserted and is on a different seam from the one it asserted it about.

**And the other column could not be measured at all**, which is a result too: a rigid wing at the optimum's ride height makes *more* downforce, compresses the spring further and drives itself through the suspension's declared floor at macro-step $40$. **At this operating point the elastic seam is not an accuracy term — it is what keeps the design inside the model's envelope.**

**[AI Inference]:** the general shape is that a seam's worth scales with the local sensitivity of the physics it carries, so an ablation at a reference design is a lower bound on its worth at an optimum that the search has driven into a more sensitive regime. One instance, on one seam, argued from $\mathrm dL/\mathrm dh$ growing as the gap closes.

---

## 8. The gradient: four knobs, and §0.4's three fields for each

The whole rollout is differentiated in all four knobs at once — reverse-mode autograd through every sub-exchange of every window's solve, through the blend, through the global Leray projection, and through **both** seams' Newton solves at both ends.

**Three of the four knobs are exactly differentiable and the fourth is a difference on the operator, and the split is where each is exact.** Linear elasticity is exactly linear in $E$ at fixed geometry, so $S_e(E^*) = (E^*/E_{\text{ref}})\,S_e(E_{\text{ref}})$ holds to the bit (CS-12 §3.2 asserts it) and autograd carries $E^*$, $k$ and $h_0$ with no help. The **thickness moves the mesh**, so there is no such identity: its derivative is a derivative of an FE assembly. It is taken as a central difference **on the operator only** — one scalar's worth of $32$ solves of a $198$-degree-of-freedom system, evaluated away from any rollout — and the exact adjoint is taken through everything else. The step is declared and swept:

| relative step on $t/c$ | $10^{-2}$ | $10^{-3}$ | $\mathbf{10^{-4}}$ | $10^{-5}$ | $10^{-6}$ |
|---|---|---|---|---|---|
| $\mathrm d(\sum S_e + \sum M)/\mathrm d(t/c)$ | $-2.32404\times10^{9}$ | $-2.32336\times10^{9}$ | $\mathbf{-2.32336\times10^{9}}$ | $-2.32316\times10^{9}$ | $-2.32125\times10^{9}$ |

A clean plateau at $10^{-3}$–$10^{-4}$ with cancellation setting in below, so the declared $10^{-4}$ sits on it.

### 8.1 The horizon, per knob — and none of the four changes sign

| $N$ | $J$ | $\mathrm dJ/\mathrm dE^*$ | $\mathrm dJ/\mathrm d(t/c)$ | $\mathrm dJ/\mathrm dk$ | $\mathrm dJ/\mathrm dh_0$ |
|---|---|---|---|---|---|
| $20$ | $0.20810365$ | $+4.411\times10^{-7}$ | $+1.2319$ | $-1.073\times10^{-2}$ | $-1.860\times10^{-1}$ |
| $40$ | $0.21661648$ | $+3.241\times10^{-7}$ | $+0.9053$ | $-9.178\times10^{-3}$ | $-2.573\times10^{-1}$ |
| $80$ | $0.22473065$ | $+2.344\times10^{-7}$ | $+0.6559$ | $-7.411\times10^{-3}$ | $-2.615\times10^{-1}$ |
| $120$ | $0.22631017$ | $+1.881\times10^{-7}$ | $+0.5265$ | $-5.995\times10^{-3}$ | $-1.888\times10^{-1}$ |
| $160$ | $0.22449587$ | $+1.693\times10^{-7}$ | $+0.4739$ | $-4.811\times10^{-3}$ | $-1.231\times10^{-1}$ |
| $240$ | $0.22087545$ | $+1.631\times10^{-7}$ | $+0.4565$ | $-3.389\times10^{-3}$ | $-8.282\times10^{-2}$ |

> $N = 240$; $N_{\text{sign}} \le 20$ on **all four** knobs — no crossing over the horizons marched; $N_{\text{valid}}(10\%) = 160$ for $E^*$ and $t/c$, and $240$ for $k$ and $h_0$.

**Every knob keeps its sign, and that is not what CS-10 found.** CS-10's $\mathrm dJ/\mathrm dh_0$ changes sign between $N=40$ and $N=80$ and CS-11's $\mathrm dJ/\mathrm dU$ changes sign too; CS-12's $\mathrm dJ/\mathrm dE^*$ does not. Here **none of the four does** — including $h_0$, which is CS-10's own knob on CS-10's own seam.

**[AI Inference]:** the difference is the release state. CS-10 releases every column from one field at one ride height, so a change in $h_0$ changes how far the plate has to travel and the short-horizon answer is dominated by that transit; here every design is released at **its own quasi-static equilibrium** $h = h_0 - L_{\text{ref}}/k$, so a change in $h_0$ does not create a transit to be dominated by. That makes this $\mathrm dJ/\mathrm dh_0$ **a different derivative from CS-10's** and not a contradiction of it — and it is the right one for a design search, where each design should be judged from a state consistent with itself rather than from another design's. It is argued from the construction and has not been tested by re-running CS-10's release convention here.

**And §0.4 still binds.** $\mathrm dJ/\mathrm dh_0$ read at $N = 20$ is $2.2\times$ its value at $N=240$ and read at $N=80$ it is $3.2\times$; $\mathrm dJ/\mathrm d(t/c)$ at $N=20$ is $2.7\times$ its $N=240$ value. **An optimiser stepping on the short-horizon gradient moves the right way and by two to three times too much**, which is what a design search's step size then has to absorb.

### 8.2 Is the gradient right? Central differences on all four knobs

Seven relative steps per knob, $10^{-2}$ down to $10^{-5}$, each a pair of $120$-macro-step rollouts from the same settled field — $56$ rollouts, and the reason this is a stage of its own rather than a paragraph.

| knob | adjoint | best central difference | relative agreement | at step | truncation branch monotone? |
|---|---|---|---|---|---|
| $E^*$ | $+1.88049048\times10^{-7}$ | $+1.88048721\times10^{-7}$ | $\mathbf{1.74\times10^{-6}}$ | $10^{-5}$ | **no** |
| $t/c$ | $+5.26542574\times10^{-1}$ | $+5.26541694\times10^{-1}$ | $\mathbf{1.67\times10^{-6}}$ | $3\times10^{-5}$ | **no** |
| $k$ | $-5.99481095\times10^{-3}$ | $-5.99473944\times10^{-3}$ | $\mathbf{1.19\times10^{-5}}$ | $10^{-5}$ | yes |
| $h_0$ | $-1.88831019\times10^{-1}$ | $-1.88845494\times10^{-1}$ | $\mathbf{7.67\times10^{-5}}$ | $10^{-5}$ | yes |

**All four agree, and the worst of them is $7.7\times10^{-5}$** — including $t/c$, whose adjoint is *not* pure autograd but the central difference on the operator spliced into the exact adjoint through everything else. That splice is the one place a sign or a chain-rule factor could be wrong and not show up as a crash, and it is the second-best-agreeing knob of the four.

**And the sweep exists because a single step said the opposite.** `stage_gradient` compares the adjoint against **one** central difference per knob at a relative step of $10^{-3}$, and two of the four came back at $4.4\times10^{-2}$ and $1.1\times10^{-1}$ — against CS-10's $1.3\times10^{-6}$ and CS-12's $1.3\times10^{-5}$. **Either the adjoint is wrong or the difference is, and one step cannot tell them apart.** Seven can, and the answer is that the difference is:

| relative step | $10^{-2}$ | $3\times10^{-3}$ | $10^{-3}$ | $3\times10^{-4}$ | $10^{-4}$ | $3\times10^{-5}$ | $10^{-5}$ |
|---|---|---|---|---|---|---|---|
| $E^*$ | $9.4\times10^{-5}$ | $2.4\times10^{-5}$ | $4.5\times10^{-5}$ | $2.2\times10^{-5}$ | $1.0\times10^{-5}$ | $4.0\times10^{-5}$ | $\mathbf{1.7\times10^{-6}}$ |
| $t/c$ | $1.9\times10^{-3}$ | $8.4\times10^{-6}$ | $3.3\times10^{-5}$ | $3.0\times10^{-5}$ | $2.0\times10^{-5}$ | $\mathbf{1.7\times10^{-6}}$ | $5.4\times10^{-6}$ |
| $k$ | $1.7\times10^{-1}$ | $1.1\times10^{-1}$ | $\mathit{4.4\times10^{-2}}$ | $6.3\times10^{-4}$ | $1.5\times10^{-4}$ | $6.5\times10^{-5}$ | $\mathbf{1.2\times10^{-5}}$ |
| $h_0$ | $2.0\times10^{-1}$ | $1.5\times10^{-1}$ | $\mathit{1.1\times10^{-1}}$ | $4.2\times10^{-2}$ | $1.1\times10^{-3}$ | $1.7\times10^{-4}$ | $\mathbf{7.7\times10^{-5}}$ |

**The two bad rows are exactly the two knobs that move the ride height at release, and that is the discriminator.** Each design is released at its own quasi-static equilibrium $h = h_0 - L_{\text{ref}}/k$, which is a function of $h_0$ and $k$ **and of neither of the structural knobs**. And `FlexWing.forcing` places the plate's Gaussian kernels on an **integer stamping box** whose corner is $\operatorname{round}(h/\mathrm dx - \tfrac12)$, taken on a *detached* value — a window on the lattice rather than a parameter, and `DiskBank.forcing`'s own device. That corner is a **step function of the ride height**. So $J$ is piecewise smooth in $h_0$ and $k$, the adjoint returns the smooth branch's derivative exactly and is **blind to the jump by construction**, and a central difference wide enough to land the two evaluations on different stamping boxes measures the jump instead. **[AI Inference]:** that the mechanism is specifically the stamping box is argued from *which* two knobs misbehave and from the shape of their curves — a quantisation artefact does not improve with a smaller step until the step stops crossing a boundary, which is what the $4.2\times10^{-2} \to 1.1\times10^{-3}$ collapse between $3\times10^{-4}$ and $10^{-4}$ looks like — and it has not been confirmed by instrumenting the corner directly.

**So what is quoted is the best over the sweep and not a converged floor**, and the page does not claim one. The branches for $E^*$ and $t/c$ are **not monotone** — $E^*$'s bounces at the $10^{-5}$ level before dropping — and three of the four are still improving at the smallest step tested, so each figure is an upper bound on the disagreement rather than an estimate of it. Compare [[poc1-results-differentiable-design]] §5, whose classical column did get a textbook curve either side of a minimum at $h = 10^{-5}$ and could quote $5.9\times10^{-9}$ as a floor. **This is a check that the gradient is right; it is not a measurement of how right.**

> **And the convergence ORDER is the sharper reading, because it reaches back to CS-10.** A central difference of a smooth function converges at $O(h^2)$ — one decade of step buys two decades of error. Fitting the log-log slope between consecutive steps: PoC 2's $h_0$ branch runs $0.25,\,0.28,\,0.79,\,3.29,\,1.59,\,0.71$ and its $k$ branch $0.32,\,0.86,\,3.52,\,1.30,\,0.71,\,1.54$ — erratic and nowhere near $2$. **And [[case-study-ground-effect-atlas-0.1]] §6's own published table, on this same knob, converges at $0.99,\,0.63,\,0.86,\,1.01,\,1.57$ — order ONE**, which CS-10 read as *"a clean truncation branch, monotone over five decades."* Monotone it is; a central difference's truncation branch it is not, and a first-order error term in a second-order formula is a statement that the function is not smooth at the scale being probed. Opened as **W144**. **It does not overturn CS-10's F5 claim** — the adjoint still agrees to $1.3\times10^{-6}$ and the *agreement* was the claim — but the characterisation of the branch was wrong, and the same defect was sitting in a published table nobody had fitted a slope to.

The $10^{-4}$ step declared for the thickness *operator* (the table above) comes from a different sweep — the operator's own, evaluated away from any rollout — and the two should not be confused: this table sweeps the step used for the whole objective's central difference, and the operator's step stays at its declared $10^{-4}$ throughout.

---

## 9. The constrained design search

The thing PoC 1a could not run, because it had no constraint: four knobs, one objective, **two ceilings**, and a gradient taken through the whole composed rollout including both seams' Newton solves.

$$
\max_{d\,\in\,\mathcal B}\; L(d)
\quad\text{s.t.}\quad
\underbrace{\max_{n,e}\,\sigma_{\text{vm}}\big(d\big) \le \sigma_{\text{ceil}}}_{\text{stress}},
\qquad
\underbrace{\max_n\,\lvert\delta_n(d)\rvert \le \delta_{\text{ceil}}}_{\text{deflection}},
\qquad
d = \big(E^*,\; t/c,\; k,\; h_0\big).
$$

$L$ is the settled downforce, averaged over the last $30$ of $N = 120$ macro-steps from the shared settled field (§3). Both constraints are **maxima over the whole tail and the whole structure**, so each enters as a margin normalised by its own ceiling — $g_\sigma = \sigma_{\max}/\sigma_{\text{ceil}} - 1$ and $g_\delta = \delta_{\max}/\delta_{\text{ceil}} - 1$ — which makes the two commensurable without a weight nobody measured. They are combined by an exterior quadratic penalty:

$$
J_{\text{pen}}(d) \;=\; L(d) \;-\; w\Big(\max(0,\,g_\sigma)^2 + \max(0,\,g_\delta)^2\Big),
\qquad w = 40 .
$$

**Two margins are computed for each constraint and they do different jobs.** The maximum is smoothed by `logsumexp` at a temperature of $0.5\%$ of the ceiling so the gradient is not a $\delta$-function on whichever station happens to be peak; but a smooth maximum **overshoots the true one** by $\text{temp}\times\log n$, and inside a constraint that overshoot is a constraint error wearing the word "smoothing" (**W142**). So the smooth pair drives the gradient and the **true** pair decides feasibility, and every number below is judged on the true pair.

### 9.1 The result — and what actually binds, which is neither ceiling

| | $J$ | $\sigma$ margin | $\delta$ margin | $E^*$ | $t/c$ | $k$ | $h_0$ |
|---|---|---|---|---|---|---|---|
| start | $0.22631017$ | $-0.1730$ | $-0.1307$ | $5.00\times10^{4}$ | $0.0400$ | $2.50$ | $0.320$ |
| **gradient**, best feasible (it $25$) | $\mathbf{0.27004816}$ | $-0.0230$ | $-0.4773$ | $9.013\times10^{4}$ | $0.04355$ | $1.500$ | $0.2569$ |
| **CMA-ES**, best feasible (eval $191$ of $200$) | $0.26927251$ | $-0.0809$ | $-0.6883$ | $1.347\times10^{5}$ | $0.04574$ | $2.132$ | $0.2107$ |

$+19.3\%$ downforce over the start, and:

> **Neither named ceiling is active. What stops the search is `Suspension.validity`'s declared FLOOR under the ride height.** The gradient optimum sits $2.3\%$ under the stress ceiling and at $52\%$ of the deflection ceiling, and **$8$ of the $30$ gradient iterations and $66$ of the $200$ population samples were declined by the suspension expert** — the search climbs to $J \approx 0.270$, walks off the envelope, backs off, and climbs again. The binding constraint is a **validity bound**, not a design constraint.

**The two optimisers reached the same objective by different routes, and only the envelope explains it.** The gradient rides the floor — a soft spring ($k$ on its lower bound) at a high free height, settling at $h = 0.0768$, which is $4.92$ cells of gap against the floor's $4.5$. CMA-ES landed on a *stiff* spring ($k = 2.13$) at a low free height, settling at $h = 0.0844$, $5.40$ cells — further from the floor, with $8\%$ of stress headroom against the gradient's $2.3\%$. **The same $J$ to $0.3\%$, from two designs that are not near each other**, because the objective is nearly flat along the trade between $k$ and $h_0$ that holds $h$ fixed and the only thing breaking the tie is how close each route passes to a wall neither optimiser can see.

**And that is a result about the model rather than a failure of the search.** In this flow the downforce rises monotonically all the way down — CS-10 §2.2 looked for the turnover at every height down to $4.8$ cells of gap and there is none, because the rolling road removes the viscous gap choke that ends a real downforce curve. So the objective points straight at the floor, and the first thing it meets is the height below which a two-line algebraic suspension is not a model of anything. **The optimiser found the edge of the model before it found the edge of the wing.**

**The envelope binds on the TRANSIENT, not on the settled state.** The declines land at macro-steps $16$, $18$, $24$, $31$, $34$, $38$ and $48$ of $120$ — never at step $0$ and never in the settled tail. The reported optimum settles at $h = 0.0768$, comfortably above the floor of $0.0703$; what is declined is *how a nearby design gets there*, because the algebraic suspension has no mass and no damper and follows the release transient's load overshoot straight down. A design whose settled ride height is legal can still be declined for its path, and that is the envelope behaving correctly.

**Why neither ceiling binds, and why that is the flow model's doing.** Stiffening and thickening the wing raise the downforce *and* lower both the stress and the deflection, because the porous plate has no thickness in the flow — §2.1. So the downforce-seeking direction runs *away* from the deflection ceiling, and it approaches the stress ceiling only through the ride height, which the envelope cuts off first. **§9.6 asks the same question with the stress ceiling placed below what the envelope lets the search reach**, which is where the constrained-search capability is actually demonstrated.

**Two prior placements are kept in the artifact with their outcomes, because a ceiling moved until the answer is agreeable is not a constraint.** `design_loose` ($\sigma \le 600$, $\delta \le 0.040$, $t/c \le 0.080$) ended with both ceilings inactive at an optimum on all four box bounds — a **box search wearing the word "constrained"** — and, it turned out, with its optimum settling *below the rolling road*. `design_unenforced` is this run's box and ceilings without W145's check.

### 9.2 What the trajectory does at a wall it cannot see

$22$ of the $30$ gradient iterations are answered and $8$ are **declined**, and the shape is a limit cycle against the envelope:

| it | $J_{\text{pen}}$ | $\sigma$ margin | $k$ | $h_0$ | outcome |
|---|---|---|---|---|---|
| $0$ | $0.2263102$ | $-0.1730$ | $2.500$ | $0.3200$ | ok |
| $4$ | $0.2681504$ | $-0.0258$ | $1.500$ | $0.2600$ | ok |
| $5$ | — | — | — | — | **declined**, macro-step $24$, $h = 0.0700393$ |
| $8$ | $0.2695510$ | $-0.0277$ | $1.538$ | $0.2542$ | ok |
| $9$ | — | — | — | — | **declined**, macro-step $16$ |
| $\mathbf{25}$ | $\mathbf{0.2700482}$ | $-0.0230$ | $1.500$ | $0.2569$ | **ok — the answer** |
| $26$ | — | — | — | — | **declined**, macro-step $18$ |
| $29$ | — | — | — | — | **declined**, macro-step $31$ |

**The penalty never fires in this column.** Every answered iterate is strictly inside both ceilings, so $J_{\text{pen}} = L$ throughout and the exterior penalty is inert — which is why §9.4's weight sweep reads identically at all three weights. The optimiser is being stopped by something the objective *cannot see*, because a validity envelope is not a term in it.

> **A declination is a legitimate outcome of a design and the search carries on**, which is CS-12 §7.1's rule and was already implemented: `_evaluate` catches the `RuntimeError`, classifies it as `envelope` / `blowup` / `other`, and the loop backs off halfway toward the start along the last good direction. **The machinery had never had anything to catch** until W145 put the check on the path the search uses. What it produces is a limit cycle rather than convergence — Adam's moments are not reset by a back-off, so the next step leaps at the wall again — and the best feasible iterate arrives at $25$ of $30$ rather than at the end.

**[AI Inference]:** the principled repair is to give the search a *smooth* signal for the wall instead of a cliff — a third margin $g_h = h_{\text{floor}}/h_{\min} - 1$ carried through the same penalty, so the gradient turns away from the floor before reaching it rather than discovering it by declining. That is not done here: the run was specified with two ceilings and adding a third constraint would change the problem rather than measure it. It is what **W145**'s open half is about.

### 9.3 Against CMA-ES — three ratios, and the flattering one is not the headline

The baseline is **CMA-ES** (`cma` $4.4.4$), the same optimiser [[poc1a-frozen-expert-results]] used, on the **same** objective, the **same** penalty weight, the **same** box, from the **same** start. An earlier draft of this stage rolled its own Gaussian sampler with a shrinking $\sigma$; that is a weaker baseline than CMA-ES and would have flattered the gradient column, so it was replaced. (`design_loose`'s $20\times$ was measured against that weaker sampler and should not be compared with the numbers here.)

| what is being compared | value |
|---|---|
| **quality** — best feasible $J$, gradient over population | $\mathbf{1.0029\times}$ — a **tie** |
| rollouts to match, against the population's **whole budget** | $200/9 = 22.2\times$ |
| **rollouts to match, against the evaluation the baseline actually peaked at** | $191/9 = \mathbf{21.2\times}$ |
| one gradient iterate, in population evaluations, **in wall-clock** | $18.09\,\mathrm{s} / 3.67\,\mathrm{s} = 4.92$ |
| **wall-clock to match** | $\mathbf{4.31\times}$ |

**The $22\times$ is the flattering form.** It divides by the baseline's *entire* budget rather than by the evaluation at which it peaked. Here CMA-ES peaked at $191$ of $200$, so the correction is worth only $5\%$ — **but this column is the exception**: in the pre-W145 run the correction is worth $59\%$ ($22.2\times$ read for a true $14.0\times$, peak at evaluation $126$) and in §9.6's tight column $54\%$ ($18.2\times$ for $11.8\times$, peak at $130$). **Two of the three runs are badly flattered by the full-budget form and one is not, so which form is used cannot be decided per run.** All of them are recorded in each record's `comparison` block.

**And rollouts are not the currency anyone pays.** One Adam iterate costs a forward march **and** a reverse sweep; one population evaluation costs the forward march alone. Measured end-to-end on this run — $542.6$ s for $30$ Adam steps against $734.5$ s for $200$ evaluations — **a gradient iterate costs $4.92$ population evaluations**, which takes $21.2\times$ down to $\mathbf{4.31\times}$ in seconds. That is the number a practitioner pays, and §10 measures the same premium independently at $5.48$.

> **Both columns got cheaper per evaluation than the pre-W145 run, and for the same reason.** A declined design stops marching where it leaves the envelope — macro-step $16$ to $48$ of $120$ — so it costs a fraction of a full rollout. With $8$ of $30$ gradient iterations and $66$ of $200$ population samples declining, the mean cost per iterate fell on both sides. **A wall-clock ratio taken across a run with declines in it is therefore a ratio between two different mixes of full and truncated rollouts**, and it is quoted as what it is.

> **The denominator here is $30$ and not $31$**, and that is not pedantry. The gradient history carries a final bare evaluation that pays no reverse sweep; dividing the wall-clock by $31$ makes an Adam iterate look $3\%$ cheaper than it is and moves the ratio the flattering way. What `reach` counts is Adam **steps**, so what the seconds are divided by is Adam steps.

> **Which of these is comparable with PoC 1's and PoC 1a's headline ratios, and which is not.** Those pages count *rollouts to a common target* — $18$ against $144$ for $8.0\times$ at $36$ variables — which is the same convention as the $\mathbf{14.0\times}$ here and **not** the same as the $22.2\times$. So $14.0\times$ is the number to set beside $8.0\times$. **Neither of those pages divides by the adjoint premium**, although [[poc1-results-differentiable-design]] measures it on the same page at *"six to nine forward evaluations"* per gradient — so PoC 1's and PoC 1a's wall-clock ratios would be several times below their published rollout ratios too, and nothing there says so. Opened as **W143**. It does not overturn their conclusion, which rests on a *trend* in $n$ that a constant per-iterate factor does not touch.

**Four variables is where a population method is strongest, and this is the expected shape.** PoC 1a measured $8.0\times$ at $36$ variables and $>30.9\times$ at $75$, against the same optimiser, and the advantage of an adjoint over a population grows with $n$ because an adjoint costs what it costs regardless of $n$. A CMA-ES run at $n = 4$ has a search space small enough to cover, and it very nearly did — it landed within $0.3\%$ of the gradient's answer. **What this run demonstrates is not a large win but that the win is still there once the derivative has to be taken through two simultaneous interface solves and the feasible region is bounded by an expert that can refuse to answer.** The claim being tested was never "gradients beat CMA-ES at four variables"; it was that the assembly is differentiable end-to-end and that the resulting gradient is good enough to drive a search to the edge of the model and hold near it. It is, and it did — §9.6 is where it holds a *ceiling*.

> **$n = 1$ on both sides.** One seed, one start, one budget. The quality tie is well inside what a different seed could move, and no variance was measured. What is not seed-dependent is *which* thing binds: eight declines on one column and sixty-six on the other are not a seed's doing.

### 9.4 The penalty weight, and a sweep that measures nothing — which is the finding

Three weights over a factor of $16$, eight Adam steps each, everything else held:

| $w$ | last answered iterate $J_{\text{pen}}$ | $\sigma$ margin | violation | best feasible $J$ |
|---|---|---|---|---|
| $10$ | $0.2526398$ | $-0.09732$ | $0.00\%$ | $0.2681504$ (it $4$) |
| $\mathbf{40}$ | $0.2526398$ | $-0.09732$ | $0.00\%$ | $0.2681504$ (it $4$) |
| $160$ | $0.2526398$ | $-0.09732$ | $0.00\%$ | $0.2681504$ (it $4$) |

**Identical to the digit at all three weights, and that is the point.** The exterior penalty only does anything once an iterate violates a ceiling, and in this column **no iterate ever does** — the envelope declines the trajectory before the stress ceiling is reached (§9.2). So $J_{\text{pen}} = L$ everywhere, the weight multiplies zero, and a sweep designed to measure the penalty's equilibrium measures the *envelope's* instead.

> **A sweep that returns the same number three times is either a bug or a statement, and the way to tell is to have predicted which.** Here the prediction is available and it was wrong: the sweep was written expecting the textbook picture — a light weight buying ascent and paying in violation, a heavy one the reverse — and that is exactly what the **pre-W145** run showed, at $2.76\%$, $1.45\%$ and $0.00\%$ violation for $w = 10, 40, 160$. Those numbers are in the artifact as `penalty_unenforced`. They were measured on a trajectory that had already left the model's declared envelope by iteration $5$, so what they characterise is a penalty acting on designs the suspension expert does not stand behind.

**Where the exterior penalty's behaviour IS visible is §9.6**, where a ceiling that binds inside the envelope gives the penalty something to act on.

### 9.5 The run that had no envelope check, and exactly what it was worth

**W145.** `FrontWingRollout.run` enforced both experts' declared envelopes at every macro-step. `objective` — the method a gradient is taken through and the one `stage_design` calls — enforced neither. So the first constrained run had **no envelope check at all**, and it walked straight out:

| pre-W145 iterate | $J$ | settled $h$ | cells of gap | |
|---|---|---|---|---|
| $0$ | $0.2263102$ | $0.22948$ | $14.69$ | inside |
| $4$ | $0.2681504$ | $0.08126$ | $5.20$ | **inside — the last one that is** |
| $5$ | $0.2770891$ | $0.06060$ | $3.88$ | outside |
| $8$ | $0.2880381$ | $0.03382$ | $2.16$ | outside |
| $\mathbf{14}$ — *the reported answer* | $\mathbf{0.2886323}$ | $\mathbf{0.03501}$ | $\mathbf{2.24}$ | **outside** |

The declared floor is $h > 0.0703125$, which is $4.5$ cells; [[case-study-ground-effect-atlas-0.1]] §2.2 measured its ground-effect curve down to $4.8$. **The pre-W145 answer sat at $2.24$ cells — less than half the lowest ride height either the envelope admits or the parent case study ever measured** — and marching that same design through `run` raises at macro-step $4$ of $120$. `design_loose`, the first run of all, is worse still: its optimum settles at $h = -0.0070$, **the mount below the rolling road**, and nothing anywhere said so.

> **What the missing check was worth: a fictitious $+6.9\%$.** The pre-W145 run reported $J = 0.2886323$; the enforced run reports $0.2700482$. Everything between them was bought outside the model. And the pre-W145 run's own last valid iterate — number $4$, at $J = 0.2681504$ — is within $0.7\%$ of the enforced answer, so **a search with the check would have arrived at essentially the same place in a fifth of the iterations.** The eleven iterations after it were spent optimising a model that had stopped applying.

**The mechanism is that bounding $h_0$ does not bound $h$.** The design box bounds the *free* ride height at $h_0 \ge 0.20$; the *settled* one is $h = h_0 - L/k$, and the search buys ride height by softening the spring. §2's bounds were each argued from a measurement, and this one was argued about the wrong variable.

**Three paths had the same asymmetry and it is not this graph's bug.** `wing_fsi.FSIRollout.objective` and `ground_effect.GroundRollout.objective` are identical in structure — a `run` that checks and an `objective` that does not — and the demo was a third, since its march and its optimiser both drive `macro_step` directly. All four now call one check that reads **detached** values, so it cannot enter a tape or move a number: CS-10's and CS-12's suites are $82$ green and unchanged, and this graph's reference design is bitwise unaffected. PoC 2 is simply where a search walked far enough to find it.

> **A framework that can decline and does not decline on the path a search uses is worse than one that cannot decline at all** — because the search is precisely the thing that goes looking for the edge, and a silent envelope turns "the model does not apply here" into "here is your optimum." The driver's `_evaluate` had carried the catch-and-classify machinery for this since it was written. It had never had anything to catch.

**And how it was found is the part worth keeping.** The ablation was extended to run at the optimum for no reason except to replace an **[AI Inference]** in §7.2 with a measurement. It crashed. **A measurement added to remove a hedge found a defect that no gate, no residual, no crossing search and no certificate had seen** — none of them look at a design the search chose, because all of them run at the reference design.

### 9.6 A ceiling placed where the envelope lets the search reach it — the constrained column

§9.1's search is stopped by a validity bound, which is a result about the model and not a demonstration of constrained optimisation. So the same search is run once more with **one** thing changed: the stress ceiling at $\sigma \le 180$ instead of $200$, below the $\approx195$ the envelope permits. Same box, same start, same weight, same budget, same seed.

| | $J$ | $\sigma$ margin | $\delta$ margin | $E^*$ | $t/c$ | $k$ | $h_0$ |
|---|---|---|---|---|---|---|---|
| start | $0.22631017$ | $-0.0811$ | $-0.1307$ | $5.00\times10^{4}$ | $0.0400$ | $2.50$ | $0.320$ |
| **gradient**, best feasible (it $10$) | $\mathbf{0.26807548}$ | $\mathbf{-0.0037}$ | $-0.6575$ | $1.189\times10^{5}$ | $0.0460$ | $1.565$ | $0.2584$ |
| **CMA-ES**, best feasible (eval $130$ of $200$) | $0.26618891$ | $-0.0107$ | $-0.7896$ | $1.927\times10^{5}$ | $0.04591$ | $1.725$ | $0.2503$ |

> **The stress ceiling is ACTIVE**: the optimum sits $0.37\%$ under it, and the trajectory crosses it seven times ($+0.90\%$, $+0.31\%$, $+0.39\%$, $+1.21\%$, $+1.21\%$, $+1.89\%$, $+1.62\%$ …) and comes back. **One** iterate out of $30$ is declined by the envelope, against eight in §9.1. So the search is riding a *constraint*, the exterior penalty is doing the work it exists for, and the settled ride height at the optimum is $h = 0.0871$ — $5.57$ cells, clear of the floor.

**And this is the exterior penalty's textbook picture, measured on a trajectory that stays inside the model.** $J_{\text{pen}} \ne L$ at every violating iterate, the penalty pulls each one back, and what is reported is the best strictly feasible iterate rather than the converged one. The quality ratio against CMA-ES is $1.0071\times$ — a tie again, at $n = 4$ — and this column is where the flattering-ratio correction bites hardest: **$18.2\times$ against the population's whole budget, $11.8\times$ against the evaluation it actually peaked at ($130$ of $200$), and $\mathbf{1.84\times}$ in wall-clock**, because a gradient iterate here costs $6.41$ population evaluations.

> **All three columns, so the spread is visible rather than a single number being quoted:**
>
> | column | quality | rollouts, full budget | rollouts, to the peak | **wall-clock** |
> |---|---|---|---|---|
> | `design` (§9.1) | $1.0029\times$ | $22.2\times$ | $21.2\times$ | $\mathbf{4.31\times}$ |
> | `design_tight` (§9.6) | $1.0071\times$ | $18.2\times$ | $11.8\times$ | $\mathbf{1.84\times}$ |
> | `design_unenforced` (§9.5) | $1.0066\times$ | $22.2\times$ | $14.0\times$ | $\mathbf{3.82\times}$ |
>
> **The rollout ratio is $18$–$22\times$ and the wall-clock ratio is $1.8$–$4.3\times$**, on the same three runs. Quality is a tie in all three. **The honest summary of §9.3 is therefore that at four design variables, against CMA-ES, on this problem, an adjoint buys between two and four times the wall-clock efficiency and no better answer** — and PoC 1a's $8.0\times$ at $36$ variables is a rollout ratio that has not had this division applied to it either (**W143**).

> **What §9.6 buys, said plainly.** It does not make the wing better: $J = 0.2681$ against §9.1's $0.2700$, because a tighter ceiling is a smaller feasible set. What it buys is the demonstration the run was specified for — a gradient taken through two simultaneous interface solves, driving a design onto a stated constraint and holding it there — on a column where the constraint is what the design meets first. **§9.1 remains the answer to "what does this model say the best front wing is"; §9.6 is the answer to "does the constrained search work".**

**The ceiling has now been placed twice and every placement is in the artifact with its outcome.** $600 \to 200$ after `design_loose` turned out to be a box search; $200 \to 180$ here after the envelope turned out to bind first. §2.2's rule is that a ceiling the search cannot reach is not a constraint, and "cannot reach" now demonstrably includes "the validity envelope stops it first." What would make this indefensible is a ceiling moved until the *answer* was agreeable — so `design_loose`, `design_unenforced`, `design` and `design_tight` all sit in `w141.json` together, and this paragraph is the reason.

---

## 10. What it costs, and why there is no GPU column

**One CPU thread, and that is measured rather than assumed — three times, because the first measurement was wrong.**

| | bare march | objective forward | $+$ reverse sweep |
|---|---|---|---|
| $1$ thread | $\mathbf{33.3}$–$37.1$ ms/macro-step | $33.3$–$35.5$ ms | $\mathbf{182}$–$192$ ms |
| $2$ threads | $45.4$–$48.6$ ms | — | — |
| $4$ threads | $47.9$–$52.0$ ms | — | — |
| $8$ threads | $63.4$–$73.3$ ms | $61.7$–$68.9$ ms | $426$–$483$ ms |

> **More cores make this workload $1.9$–$2.0\times$ SLOWER**, and that is the answer to whether any of it wants an accelerator. Six $80\times80$ windows in float64 with a $33$-square dense solve and one FFT per exchange are tensors small enough that OpenMP's fork/join *per operation* costs more than the operation. A workload that will not use eight CPU cores is bound on per-operation overhead rather than on arithmetic, and a GPU adds launch latency and host synchronisation to the same operation count.

**Checked bitwise before it was adopted**: $12$ macro-steps at $1$ thread and at $8$ agree to the **bit** on the load, the ride height, the tip deflection and the peak stress, in every one of the three runs, so the thread count is a **clock and not a variable**. A threaded reduction can reorder a floating-point sum, and had it done so here every number in this run would have carried that as its level.

**One gradient iterate costs $\mathbf{5.4}$–$\mathbf{5.8}$ objective forwards** — $5.48$, $5.77$, $5.39$ on three runs — which is *inside* CS-10's $4.3$–$6.7$ on its own column. The forward it is measured against is `objective(grad=False)`: margins, stress map and all, which is exactly what a population evaluation pays, so this is the premium §9.3 divides by.

> **The first version of this table was wrong and the way it was wrong is the lesson.** It reported a forward of $45.4$ ms, an adjoint of $473$ ms, a ratio of $\mathbf{9.47}$ and a thread slowdown of $\mathbf{4.8\times}$ — and it got them by timing a **taped, back-propagated objective against a bare march that computes none of the objective's quantities**, unwarmed, over $8$ macro-steps, so the operator cache miss and torch's first-call cost landed entirely on the numerator. **A ratio between two differently-defined quantities is not a ratio.** The $9.47$ was then *explained* on this page — "the assembly's adjoint carries two seams' Newton solves rather than one" — which is the failure mode worth naming: **a wrong number that sits just outside a plausible range invites a mechanism, and the mechanism makes it look measured.** The real figure sits inside CS-10's range and needs no explanation. Both the corrected ratio and the bare-march form ($5.4$–$5.5$) are recorded in `cost`, so the two cannot be confused again.

**The ratio is the claim and the milliseconds are not.** [[poc1a-frozen-expert-results]]'s own demo shipped a table of milliseconds-per-step that was wrong by a factor of four the next time anyone measured it, on unchanged code, because the box was in a different power state. **[AI Inference]:** the one column here that *would* be GPU-shaped is a frozen neural expert in the windows — dense matmul, which is what PoC 1a ran on an A100 — and **W137** is the row that says what certifying such a swap at this graph's aero-structure seam would cost.

---

## 11. What this opens

| # | Row |
|---|---|
| **W141** | **A per-seam verdict is derivable from what the compiler already emits, and nothing derived it.** Fifteen lines of grouping over `Decision.subject` turns "the graph is refused" into "seven of nine seams are not the problem, and the two that are are not the problem for the same reason." Done here in the driver and in the demo and asserted equal; done properly when `CompileResult` exposes the grouping so a caller need not know that `<graph>` reaches everything |
| **W142** | **A `logsumexp` smooth maximum overshoots by $\text{temp}\times\log n$, and inside a constraint that is a constraint error wearing the word "smoothing."** At $2\%$ of the ceiling over a $30$-step tail it is $14\%$ of a stress reading. Caught by writing the arithmetic down rather than by a failed run; fixed here by dropping the temperature to $0.5\%$ and returning **both** the smooth margins (which drive the gradient) and the true ones (which decide feasibility). A standing note rather than a defect: nothing in `atlas/` provides a constrained objective, so the next one will face the same choice |
| **W145** | **A declared envelope was enforced on the march and not on the path the design search uses, so the search optimised its way out of it and nothing fired.** `run` checked both experts' envelopes at every macro-step; `objective` — what `stage_design` calls — checked neither. The first constrained run walked the ride height down through `Suspension.validity`'s floor and reported an optimum that settles at $2.24$ cells of gap against the $4.8$ CS-10 measured down to; the loose run's settles **below the road**. Fixed here as one method called from both paths, and the search re-run with it live. **A framework that can decline and does not decline where a search looks is worse than one that cannot** — and what found it was the ablation being extended to the optimum purely to replace an **[AI Inference]** with a measurement |
| **W144** | **A central difference that converges at order ONE is not a truncation branch, and CS-10 published one as one.** The plate's kernels are stamped on an **integer** box whose corner is a step function of the ride height, so $J$ is piecewise smooth in exactly the two knobs that set the release height, and the adjoint is blind to the jump by construction. §8.2 measures it here and fits CS-10 §6's own published table at order $\approx 1$. The agreement CS-10 claimed still holds; the characterisation of the branch does not |
| **W143** | **Every gradient-vs-population ratio on this project is in rollouts, and one gradient rollout is not one population rollout.** PoC 1 measures the adjoint premium on its own page and never divides by it; PoC 1a inherits the convention. Measured here on three columns it is $3.7$–$6.4$, and it takes rollout ratios of $18$–$22\times$ down to wall-clock ratios of $\mathbf{1.8}$–$\mathbf{4.3\times}$ — §9.3 and §9.6. This PoC also made a second error first, in the other direction: dividing by the baseline's *whole* budget rather than by the evaluation it peaked at, which flatters by $54$–$59\%$ on two of the three columns. Neither overturns PoC 1's conclusion, which rests on a trend in $n$ that a constant per-iterate factor leaves alone |

**And three things this graph is now the place to ask.** Whether the two seams' mutual coupling strengthens where the deflection is a larger fraction of the ride height (§7.1); whether a knob on the **fluid** side of the wetted seam changes sign with the horizon, which is the test CS-12 §8 proposes for its own **[AI Inference]** and which this graph can now run against a knob that does not; and **W137** — what it would cost to certify a learned expert in these windows, at a seam where the fluid's block is $10^{-5}$ of the assembled operator.

---

## 12. What this does not claim

- **No stress figure here is a statement about an alloy.** The plate is a $32\times2$ Q1 plane-stress cantilever, Q1 elements lock in bending, and $E^*$ is a declared design knob in **flow units** calibrated to a deflection rather than taken from a material. The ceilings are levels chosen to be reachable from the reference design in the downforce-seeking direction, which is the only property a ceiling needs for a constrained search to be about the constraint. §2 measures the locking and places the box's lower thickness bound above the point where it inverts the stress trend.
- **The reported optimum rides at $4.92$ cells of gap, and the parent case study measured its curve down to $4.8$.** So the answer sits at the very bottom of the range CS-10 ever validated, on a monotone curve with no turnover because the rolling road removes the viscous gap choke (§9.1). It is inside the declared envelope and it is not somewhere the underlying flow model was ever checked against anything.
- **Nothing reaches `admit`.** $L$ (**W1**), $\sigma$ and $C_\mu$ (**W3**, **W49**) are unmeasured, so no seam can be green and none is.
- **The riding graph is REFUSED and the march is run anyway.** `L2/InterfaceMotion` fires on all four ports of the two surface seams and the refusal is correct — CS-10 established that no rule certifies a cached operator across a solution-dependent interface, and priced it. Everything downstream of that refusal is a **search instrument** and not a verified computation, which is [[composition-error-theory]]'s own distinction.
- **The thickness knob costs the aerodynamics nothing in this model**, because the porous plate has no thickness in the flow. That is why the box's upper thickness bound has to come from the flow model's own resolution rather than from the search finding a trade — §2, and it is the reason the first design run's optimum was out of scope.
- **One Reynolds number, one grid, one incidence.** $\mathrm{Re}_c = 125$, $208\times144$ cells, $20^\circ$, one rolling road with no floor boundary layer and therefore no viscous gap choke. Nothing here is a grid-convergence study, and CS-10's ground-effect curve is monotone with no turnover for that reason.
- **The cut's own defect has not peaked at either horizon marched.** $9.0\times10^{-5}$ is a value at $480$ macro-steps, not a bound (§5).
- **The two seams' mutual coupling is weak at the reference design and that was measured at one operating point** (§7.1). Whether it strengthens where the deflection is a larger fraction of the ride height is an open question and the ablation that would settle it has not been run.
- **$\mathrm dJ/\mathrm dh_0$ here is not CS-10's $\mathrm dJ/\mathrm dh_0$.** Each design is released at its own quasi-static equilibrium rather than from one shared field, so the two derivatives answer different questions (§8.1). The claim that this is why none of the four signs flips is argued from the construction and is **[AI Inference]**.
- **The demo's per-seam lights do not move with the design**, and §4.2 measures that over the box's sixteen corners rather than asserting it.
- **The baseline is CMA-ES and the budget is stated, and neither method was tuned against the other.** The first version of §9 rolled its own Gaussian sampler, which is a weaker baseline and would have flattered the gradient column; it was replaced before anything was published. What is *not* claimed is that either optimiser is well tuned for this landscape — the comparison is at a fixed budget on a fixed objective, which is [[poc1a-frozen-expert-results]]'s protocol and inherits its limits.
- **NEITHER named ceiling is active in the primary design column.** The optimum sits $2.3\%$ under the stress ceiling and at $52\%$ of the deflection ceiling; what stops the search is the suspension expert's declared *validity envelope* (§9.1). So a constrained search riding a **constraint** is demonstrated only in §9.6's tight column, where the stress ceiling is placed below what the envelope lets the search reach. The primary column demonstrates something else — that the search finds the edge of the model — and the two are not conflated.
- **The stress ceiling has been placed twice and both prior placements are in the artifact with their outcomes.** $600 \to 200$ after the first run turned out to be a box search; $200 \to 180$ for §9.6 after the envelope turned out to bind first. §2.2's rule — a ceiling the search cannot reach is not a constraint — is the whole justification, and it is not weakened by being applied twice; what would weaken it is a ceiling moved until the *answer* was agreeable, which is why `design_loose`, `design_unenforced` and `design_tight` all sit in `w141.json` beside `design`.
- **The primary column's exterior penalty never fires**, because no iterate reaches a ceiling. The textbook picture — the equilibrium sitting outside the feasible set by an amount set by $w$ — is measured only in `penalty_unenforced` (§9.4), on a trajectory that had already left the model's envelope, and in §9.6. No projection, filter or augmented-Lagrangian multiplier was used.
- **A validity envelope is a cliff and the optimiser cannot see it.** It is not a term in the objective, so the gradient learns about it only by being declined, and the trajectory limit-cycles against it rather than converging (§9.2). The smooth third margin that would fix this is named and **not implemented**, because the run was specified with two ceilings and adding a third constraint would change the problem rather than measure it.
- **$n = 1$ on both columns of §9.3.** One seed, one start, one budget, no variance measured. The quality tie ($1.0029\times$) is well inside what a different seed could move; *which* thing binds is not.
- **The wall-clock ratio is taken across runs containing declines**, and a declined design stops marching part-way, so both columns' mean cost per evaluation depends on how often each declined. §9.3 says so. **And the three ratios there are three different questions, only one of which is comparable with PoC 1a's**: $4.31\times$ in wall-clock against $21.2\times$ in rollouts, where PoC 1's and PoC 1a's published ratios carry the same un-divided adjoint premium — **W143**, opened against those pages rather than quietly worked around here.
- **§8.2 is a check that the gradient is right, not a measurement of a finite-difference floor.** Two of the four branches are non-monotone, three are still improving at the smallest step tested, and the two ride-height knobs converge at an erratic order well below a central difference's $2$ because the plate's forcing is stamped on an integer box (**W144**). So each agreement figure is an upper bound on the disagreement. No number here is comparable with [[poc1-results-differentiable-design]] §5's $5.9\times10^{-9}$, which *was* a floor.

---

## See Also

- [[case-study-ground-effect-atlas-0.1]] — CS-10, the ride-height half: the moving interface, the field-to-lumped seam, and the ground-effect curve this assembly rides on
- [[case-study-wing-fsi-atlas-0.1]] — CS-12, the bending half: the structural expert, the surface operator, W114, and the FSI seam
- [[poc1a-frozen-expert-results]] — the protocol this borrows: a gradient column against a population baseline at a fixed budget, with an ablation and an honest limitations section
- [[poc1-results-differentiable-design]] — the adjoint through a composed graph, first spent
- [[general-coupling-scheme]] — §4.2, whose rule this applies at two seams at once
- [[end-to-end-architecture-spec]] — §0.4's horizon rule, applied per knob in §8.1
- [[gap-worklist]] — W141 to W144, opened here; W137, which this graph is the place to spend
- [[case-study-ladder-to-f1]] — where CS-10 and CS-12 sit, and what an assembly of two rungs is evidence for
- `atlas/demo_frontwing/README.md` — the live demo, and what its panel must not be read as saying
