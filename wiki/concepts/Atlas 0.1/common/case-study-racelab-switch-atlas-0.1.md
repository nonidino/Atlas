# CS-20 — RaceLab, phase 2: the switch, and the cadence a frozen expert cannot be asked for

**Type:** Case study — **PoC 3's second phase**: per-window classical / learned / certified, the per-window one-step error, the per-call cost, and the envelope check Tier 51 was missing (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-12, Tier 52. `atlas/cases/racelab_switch.py`, the envelope check in `atlas/cases/racelab.py`, `scripts/tier52_racelab_switch.py`, `out/racelab2/racelab2.json`, `tests/test_tier52_racelab_switch.py`. Registered on the ladder as **CS-20**. Worklist row **W222** (closed), **W216**–**W221** (annotated), **W223**–**W226** (opened).
**Related:** [[case-study-racelab-graph-atlas-0.1]] · [[case-study-vehicle-march-atlas-0.1]] · [[defect-correction-learned-operator]] · [[corrupted-checkpoint-and-jacobian-fidelity]] · [[learned-contribution-kill-tests]] · [[prior-art-and-novelty-atlas-0.1]] · [[poc2-frontwing-results]] · [[poc2-demo-and-novelty]] · [[vehicle-scale-and-sizing]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[physics-simulation-datasets]] · [[f1-pathmap-and-end-goal]]

---

# 0. The result, in one paragraph

**The switch works, every number it reports is measurable, and the learned column cannot be asked for the step the composition layer runs at.** Phase 2 puts three experts on every fluid window — `WindowNS`, Poseidon-T, and Poseidon-T inside defect correction — and measures each against the classical one *on the same input state*, which is the one number [[case-study-racelab-graph-atlas-0.1]]'s requirements call the most trustworthy because it needs no global referent. **The scaling is over-determined and that is the whole story.** One 128-cell window spans $2.0$ tiling length units by geometry, so `adapters.Scaling`'s `time` is $1.0$ and the lead follows: a macro-step is $\tfrac18$ of the checkpoint's native step and **one exchange — the cadence the composition layer actually runs at — is $\tfrac{1}{32}$ of it.** Measured across leads, the per-window median error is $0.3607$, $0.5615$, $0.7301$, $\mathbf{0.1708}$, $0.2408$ at $\tfrac18$, $\tfrac14$, $\tfrac12$, $\mathbf{1}$ and $2$ times native: **a sharp minimum AT the native lead, and the error grows by $4.3\times$ as the step SHRINKS below it.** That is [[prior-art-and-novelty-atlas-0.1]] §4's named assumption — *that error shrinks with the macro step, which fails for an expert trained at a native $\Delta t$* — measured false on this graph. **So in the march the learned column is not merely worse, it leaves the fluid expert's own declared bound**: all-learned runs $\mathbf{0.067\times}$ the classical speed with an rms field error of $3.84$ and is outside a declared envelope for $35$ of $40$ macro-steps. **Per call, at the one lead where it works, it is a draw**: with each column at its own best thread count the classical column takes $0.5216$ s and the learned one $0.4526$ s, a ratio of $1.15$ — and two independent draws of that ratio differ by $20\%$, so the honest reading is *even*, not *faster*. **The certified mode does what its theorem says and costs what Tiers 48 and 50 predicted**: both windows reach the classical fixed point, and against W76's null replacement — the same iteration with the checkpoint swapped for the identity — the checkpoint bought $\mathbf{5}$ and $\mathbf{7}$ classical calls out of $270$ and $165$, about $2\%$, while taking $\mathbf{3\times}$ the wall time. **And the envelope check Tier 51 lacked is in, on by default, and found two things immediately**: that tier's machine declines on this field, and *every* arm of that tier released from a uniform freestream whose transient leaves the disk's clamp for $58$ macro-steps. **One of five governing families on this graph has a shippable learned option, and sixteen of twenty-six agents.**

---

# 1. What phase 2 is, and what closes with it

`POC3-RACELAB-REQUIREMENTS.md` §9's phase 2: **per-window mode selection, classical / learned / certified, with per-window one-step error and per-call cost**, and a headless path that runs any mode assignment and reports §5.2 and §5.3. No dashboard — that is phase 3.

**And W222, which CS-19 opened and did not close.** That tier marched 600 macro-steps with the machine motoring and nothing said so. The repair is in `atlas/cases/racelab.py` and it comes first on this page, because every number below is taken on a column the check admits.

---

# 2. The envelope check: the predicates existed and nobody asked them

**The sharpest form of W222 is not that a check was missing.** `powertrain.MachineAgent.validity` already declared the condition, in its own docstring:

> *"A generator can only push current into the battery while its back-EMF exceeds the open-circuit voltage. Below that speed the loop current reverses and the machine MOTORS -- which is a legitimate mode and a different one from the mode this graph declares, so the record declines rather than reporting a negative generated power as if it were generation."*

The disk declares its induction clamp. The fluid window declares a cell-Reynolds bound. **All three existed before Tier 51, all three were computed during it, and none of them was read.**

## 2.1 One funnel, and three states

`RaceRollout._absorb` consults every declared predicate once, at the end of every macro-step, and it is the only path `march` takes. That is PoC 2's `Engine._absorb` and **W145**'s lesson: *a check present on one path and absent from another is worse than no check, because a search goes looking for the edge.*

**Three states per expert and never two.** `True` inside, `False` declined, and `None` **not consultable** — a predicate that cannot be evaluated at this state. A missing check that defaults to `False` is a false alarm; one that defaults to `True` is the hole this row exists to close.

**`enforce` is ON by default**, because a check that is off by default is not a check. OFF does not remove it: the check still runs and what it *would* have said is recorded into `outside`, so a measurement of an out-of-envelope graph can be published **with the stamp** rather than as though it were in. `outside_first` is sticky, because a march that left the envelope at step 4 and came back at step 40 still left it.

## 2.2 The repair, and it is W199's own similarity applied one level on

The machine was sized for a rotor in open flow. The host changed. [[vehicle-scale-and-sizing]] §2.3's two conditions decide the new scaling uniquely — write $x = u_{\text{host}}/u_{\text{ref}}$:

- the back-EMF must not move. $\omega = \lambda u / r$ so $\omega \propto x$, and $k_e\omega$ invariant gives $k_e \to k_e/x$; $k_e = k_t$ is the same air-gap flux linkage, so $k_t$ follows.
- the torque must follow the disk's. $T \propto u^2$ at fixed induction and $\tau_{\text{disk}} = Tr/\lambda$, so $\tau \propto x^2$; with $k_t \propto 1/x$ that needs $I \propto x^3$, and the numerator $k_e\omega - V_{oc}$ is invariant, so $R \to R/x^3$ for every element.

$$\boxed{\;k_e = k_t \;\longrightarrow\; k_e/x,\qquad R_i \;\longrightarrow\; R_i/x^{3},\qquad V_{oc}\ \text{unchanged}.\;}$$

| | value |
|---|---|
| the machine generates only above $u = V_{oc}/(2\lambda k_e/A)$ | $\mathbf{0.8933}$ |
| CS-18's rotor, in open flow | $0.9225$ — $3.3\%$ above it |
| RaceLab's turbine, in the duct | $\mathbf{0.6692}$ — $25\%$ below it |
| $k_e$, before $\to$ after | $0.0500 \to 0.0689$ |
| $R_{\text{total}}$, before $\to$ after | $0.600 \to 1.572$ |

> **Superseded in one respect, 2026-09-13 (Tier 53).** The sizing below is for the host's inflow measured **at one instant**, and that is not enough over a horizon: $u_{\mathrm{rotor}}$ falls $6.3\%$ across 600 macro-steps while this machine's margin over its own crossover is $3.3\%$ of $u$, so the induction reaches its clamp at **macro-step 312**. Sizing for the horizon's **minimum** admits all 600. Everything else here stands — the similarity is still exact, the $x = 1$ control is still the identity bitwise, and the induction still returns to $0.11388706$ at the sizing inflow. What changes is WHICH inflow to size at. See [[poc3-racelab-demo]] §2 and [[gap-worklist]] **W228**.

**Three controls, and all three are the point.**

1. **At $x = 1$ the similarity is the identity**, bitwise, on every element's `resistance`, `k_e`, `k_t` and `r_total`. A similarity about a reference point that is not the identity *at* that point is a fit.
2. **At the duct's inflow it returns the reference induction exactly** — $0.11388706$ against $0.11388706$, a gap of $\mathbf{0}$. The electrical solution is *similar*; only its size changed.
3. **The declined/repaired pair is taken from ONE settled field.** The machine CS-19 declared raises `EnvelopeDeclined` naming `MGU`; the host-sized one marches with all four predicates `True` and zero steps outside. One declaration differs and nothing else.

| at the duct's $u = 0.6692$ | induction | loop current | `rotor_valid` |
|---|---|---|---|
| the machine CS-19 declared | $0.0200$ — the clamp floor | $\mathbf{-0.5605}$, motoring | `False` |
| the machine sized for its host | $\mathbf{0.11388706}$ | $+0.0278$, generating | `True` |

---

# 3. The second thing the check found, immediately

**Every arm of CS-19 released from a uniform freestream, and that is a state the model's own experts decline during the transient.**

It is inherited practice that was not followed. CS-18 releases from `out/w141/settled.npz`; `atlas/demo_frontwing`'s README says that without the settled cache *"the demo releases from the freestream, says so on screen, and shows a transient that no number on the results page was measured at."*

Measured, on the repaired column:

| | |
|---|---|
| macro-steps outside a declared envelope | $\mathbf{58}$, steps $0$ to $57$ |
| which expert declined | `ROTOR` — the disk's induction clamp, at its **upper** edge |
| the release state at step $240$ | **admitted by every predicate** |
| the Picard residual on the machine's sizing | $\mathbf{0.00445}$ |

The spin-up runs with `enforce=False` **by design** — a march that stops at macro-step 3 of a transient reports nothing — and the measured columns release from what it ends at with `enforce=True`. **The transient is declared rather than marched through in silence.**

**The sizing is a fixed point and one Picard step is taken, not converged.** The machine is sized for an inflow that its own thrust then changes: sized for CS-19's $0.66915$, the repaired column settles at $0.67213$. The residual is $0.45\%$ and it is **reported rather than iterated away**, because iterating in silence would hide how sensitive this coupling is — and CS-18 §7.2 measured that sensitivity at $63.3$.

---

# 4. The scaling is over-determined, and it decides everything after it

`adapters.Scaling` declares `length` as **the tiling units one 128-cell window spans**. That is geometry, not a knob: $128$ cells at $\mathrm{d}x = 1/64$ is $\mathbf{2.0}$ length units, and declaring anything else would be declaring a different window. Then $\text{time} = \text{length}/\text{velocity} = 1.0$ and $\text{lead} = \mathrm{d}t/\text{time}$ follows, against a checkpoint whose native lead is $0.1$.

| cadence | $\mathrm{d}t$ | lead | against native |
|---|---|---|---|
| **one exchange** — what the composition layer runs at | $0.003125$ | $0.003125$ | $\mathbf{1/32}$ |
| one macro-step | $0.0125$ | $0.0125$ | $1/8$ |
| eight macro-steps | $0.1$ | $0.1$ | $\mathbf{1}$ |

**`wake_array` gets a native lead and this graph cannot.** That tiling's window spans 4 rotor diameters at a macro-step of $0.2$ and exchanges once per macro-step; `ground_effect`'s spans 2 length units at $0.0125$ and exchanges four times. The ratio is arithmetic, not a tuning choice: **a learned expert dropped into this composition layer is asked for a step thirty-two times shorter than the one it was trained to take.**

---

# 5. Section 5.2 — the per-window numbers, which need no referent

## 5.1 The error does not shrink with the macro step

[[prior-art-and-novelty-atlas-0.1]] §4 names the assumption partitioned-coupling stability theory makes and a frozen neural expert lacks: *"above all that error shrinks with the macro step, which fails for an expert trained at a native $\Delta t$."* **Here it is, measured.**

| macro-steps | lead / native | median error | min | max | classical / learned |
|---|---|---|---|---|---|
| 1 | $1/8$ | $0.3607$ | $0.2987$ | $1.4252$ | $0.135$ |
| 2 | $1/4$ | $0.5615$ | — | — | $0.231$ |
| 4 | $1/2$ | $0.7301$ | — | — | $0.498$ |
| **8** | $\mathbf{1}$ | $\mathbf{0.1708}$ | $0.0865$ | $0.4403$ | $0.857$ |
| 16 | $2$ | $0.2408$ | — | — | $1.619$ |

**A sharp minimum exactly at the native lead**, and the error at $1/8$ of it is $4.3\times$ the minimum. Halving the step makes the frozen expert *worse*, monotonically, down to the shortest step measured — and the composition layer's own cadence is $1/32$, off the bottom of the table.

**[AI Inference]:** the mechanism is not diagnosed here. A plausible reading is that the checkpoint's output has a fixed-magnitude increment it was trained to produce, so asking it for a shorter interval leaves the same increment against a smaller true change, and the relative error scales roughly as the inverse of the lead — which the $0.36$, $0.56$, $0.73$ ordering is *not* monotone enough to confirm. **What would test it is a step-refinement study on a checkpoint whose training lead is known and varied**, and no such checkpoint exists here. Unmeasured, and it is **W223**.

## 5.2 Per window, at the native lead

| window | error | window | error |
|---|---|---|---|
| F00 | $0.3749$ | F01 | $\mathbf{0.4403}$ |
| F10 | $0.1663$ | F11 | $0.1968$ |
| F20 | $0.1559$ | F21 | $0.1314$ |
| F30 | $0.1150$ | F31 | $\mathbf{0.0865}$ |
| F40 | $0.2933$ | F41 | $0.1753$ |
| F50 | $0.1486$ | F51 | $0.1557$ |
| F60 | $0.1974$ | F61 | $0.1869$ |

**The worst window is F01 and the reason is the normalisation, not the physics.** The error is divided by the referent's own **fluctuation** — the field minus the free stream — because dividing by the field itself would put a window of nearly uniform flow at nearly zero error however wrong it was, and most of this domain is nearly uniform. F01 is the upper-left window, almost all free stream, so it has the smallest denominator on the graph and a modest absolute error reads as $0.44$. **The number is right and it is not a statement about that window's physics**; a demo that ranked windows by it would be ranking them by how empty they are, which is **W224**.

## 5.3 Per call, and it is a draw rather than a win

**Each column at its OWN best thread count, because the two do not agree with each other and did not agree with CS-19.**

| threads | classical (8 macro-steps) | learned (1 native call) |
|---|---|---|
| 1 | $0.7161$ | $1.1527$ |
| 2 | $0.5835$ | $0.6715$ |
| **4** | $\mathbf{0.5216}$ | $\mathbf{0.4526}$ |
| 8 | $0.5839$ | $0.5009$ |

Both are fastest at four threads here, and the ratio is $\mathbf{1.153}$ — the learned call is $15\%$ faster than the eight classical macro-steps it replaces.

**It is reported as a draw and not as a win, and the reason is a control.** A second independent draw of the same pair, in a separate process, gave $0.944$ — the other side of one. **Two measurements of one ratio $20\%$ apart do not establish a $15\%$ effect**, which is Tier 50's **W215** exactly: *a randomised — here, a noisy — quantity scored as a deterministic one*. What can be said is that at its native lead the learned call and the classical column it replaces cost about the same on this host, and that the classical column's own thread preference **differs from CS-19's**, where the full march with the car's body force in it was fastest at one thread. A bare window solve and a march carrying thirty-five stamping kernels are different objects and thread differently.

---

# 6. The certified mode: the theorem holds and the price is Tier 48's

`atlas.defect_correction` certifies a **fixed point**: it finds $w$ with $\lVert\phi(w) - w\rVert \le r_{\text{stop}}$. Theorem 1 says the limit is the classical map's own fixed point **whatever the learned map does** — the one accuracy claim in this demo that rests on a proof rather than a measurement. What is measured is only the price.

**The control is W76's null replacement**: the same iteration with $\psi$ the *identity* instead of the checkpoint. Whatever the checkpoint column does that the identity column also does is not the checkpoint's.

| window | $\psi$ | status | classical calls | cheap calls | wall |
|---|---|---|---|---|---|
| F40 | the checkpoint | `fallback_converged` | $270$ | $54$ | $8.5$ s |
| F40 | **the identity** | `fallback_converged` | $275$ | $20$ | $\mathbf{2.9}$ s |
| F01 | the checkpoint | `fallback_converged` | $165$ | $48$ | $6.8$ s |
| F01 | **the identity** | `fallback_converged` | $172$ | $10$ | $\mathbf{1.8}$ s |

**The checkpoint bought $\mathbf{5}$ and $\mathbf{7}$ classical calls out of $270$ and $165$ — about $2\%$ — and cost $\mathbf{3\times}$ the wall time.**

Both columns read `fallback_converged`: the outer iteration **stalled** and the classical march finished the job, which is the mechanism's own declared failure path working. **The answer is still the classical fixed point** — that is what the status means and what the theorem guarantees — so this is a price measurement and never an accuracy one.

**This is smaller than Tier 48's and in the same direction.** That tier measured the checkpoint worth $13$ classical calls against a clean $99$, about $13\%$; here it is $2\%$. [[corrupted-checkpoint-and-jacobian-fidelity]] then measured that the $13$ sat against a $69$-call spread from perturbing the same weights by $3\%$. **Nothing on this page moves [[case-study-ladder-to-f1]] §14.3's sentence**, and phase 2 was never going to: *the foundation-model half has no learned expert admitted with a nonzero contribution.*

---

# 7. Section 4.3 — four of five families have no learned option

The requirements are explicit that this is shown and not hidden: *"a viewer should learn from the dashboard that four of five families have no learned option — that is a true and important fact about the field."*

| family | agents | switch | why not |
|---|---|---|---|
| `incompressible-navier-stokes-2d` | 16 | **classical / learned / certified** | — |
| `plane-stress-elasticity-2d` | 1 | classical only | no shippable learned structural expert exists here; NeuberNet is unlicensed and local-only and must not appear in a bundle |
| `heat-conduction-2d` | 1 | classical only | no learned conduction operator is vendored, and no public checkpoint sits at this geometry |
| `incompressible-thermal-transport-1d` | 4 | classical only | the legs are 1-D advection with a closed-form decay; there is nothing to learn that is cheaper than the closed form |
| `lumped-dc-circuit` | 4 | classical only | the circuit is algebraic — a bisection on two scalar equations — so a surrogate would be replacing microseconds of arithmetic |

**One of five families, and sixteen of twenty-six agents.** [[physics-simulation-datasets]] §3.4 records why the field is shaped this way.

---

# 8. Section 5.3 — the mixed march, where the switch actually lives

Forty macro-steps from the admitted release state, every column against the all-classical one.

| assignment | ledger (cl / le) | s / macro-step | speed | rms field error | steps outside an envelope |
|---|---|---|---|---|---|
| **all classical** | $14$ / $0$ | $\mathbf{0.3135}$ | $1.000$ | $0$ | $\mathbf{0}$ |
| upper row learned | $7$ / $7$ | $1.7398$ | $0.180$ | $2.617$ | $33$ |
| wake learned | $8$ / $6$ | $2.5634$ | $0.122$ | $3.705$ | $35$ |
| **all learned** | $0$ / $14$ | $4.6958$ | $\mathbf{0.067}$ | $3.844$ | $\mathbf{35}$ |

**The learned columns do not merely degrade — they leave the fluid expert's own declared bound**, for $33$ to $35$ of $40$ macro-steps, and the check says so rather than the field quietly going wrong. Those rms errors are measurements of states the model declines to stand behind and are stamped accordingly; the column was run with `enforce=False` **for that purpose**, which is the only honest way to publish a number from outside an envelope.

**And this is the same arithmetic as §4, compounded.** The composition layer asks for $1/32$ of the native lead **four times per macro-step**; §5.1 measured the one-step error at $1/8$ of native at $0.36$, and the table has no row for $1/32$ because it is off the bottom. Applying that map $160$ times is what the last row is.

**The speed is the cadence too.** Poseidon is called once per exchange whatever the lead, so a learned window costs one forward pass per exchange against `WindowNS`'s one sub-step — and the forward pass is sized for a step eight times longer. All-learned is $\mathbf{15\times}$ slower for exactly that reason, and it is **not** a statement that the checkpoint is slow: §5.3 measured that at its own lead it is a draw.

---

# 9. What phase 2 hands phase 3, and what it takes away

**The switch is built and every number §5.2 and §5.3 ask for is measurable.** A per-window mode, a one-step error against the classical expert on the same input state, a per-call cost, a ledger, a speed ratio and an rms field error against the all-classical referent — all of it, headless, for any assignment.

**What phase 3 must put on screen, because phase 2 measured it:**

- **the lead mismatch, as a first-class number.** A dashboard that lets a viewer flip a window to *learned* without showing that the composition layer is asking the checkpoint for $1/32$ of its native step is showing a broken expert and calling it a learned one. The honest screen shows the ratio beside the switch.
- **the envelope stamp.** Every learned column here leaves a declared bound within a handful of macro-steps. PoC 2 already has the pattern — the page turns red, the field gets a red border, the readout flags the stats — and phase 3 needs it from the first frame.
- **four of five families greyed out with their reasons**, which is §7 and is a fact about the field rather than about this code.
- **that the certified mode's claim is a theorem and its cost is a measurement.** The mode returns the classical answer; it bought $2\%$ of the classical calls and cost $3\times$ the wall time.

**[AI Inference]:** the demo the requirements describe — *"flip it to a learned expert and see its error and its speed change"* — will, on this graph, show a viewer a column that is slower and leaves the envelope, every time. **That is a true result and a poor demonstration**, and the repair is not to hide it but to give the switch a lead it can be asked for: run the learned column at eight macro-steps and the classical one beside it at the same horizon, which is the comparison §5.1 shows is fair. That is a phase 3 design decision, it is not taken here, and it is **W225**.

---

# 10. What this tier did NOT do, named

- **No rung moved and no expert was trained.** Poseidon-T is frozen and loaded from the local cache with the hub offline.
- **[[case-study-ladder-to-f1]] §14.3 is unchanged.** The certified slot's learned contribution here is $2\%$ of the classical calls, *smaller* than Tier 48's $13\%$ and in the same direction.
- **The mechanism behind §5.1's curve is not diagnosed** (**W223**). A minimum at the native lead is measured; why the error grows as the step shrinks is an inference, and the study that would settle it needs a checkpoint whose training lead can be varied.
- **The per-window error's normalisation ranks empty windows worst** (**W224**), and no second normalisation is offered.
- **CS-19's arms were NOT re-run on the repaired column** — *done in Tier 53, see [[poc3-racelab-demo]] §4; the sizing this page used needed §2's amendment first.* Every number on that page stands as the record of what the un-repaired graph did, stamped there as outside the envelope; this tier does not restate them and does not correct them. **So the two pages describe two different vehicles**, and that is named rather than reconciled.
- **`CERTIFIED` is not a per-macro-step mode**, and `MixedRollout` raises rather than quietly running something else. Defect correction certifies a fixed point and an explicit time step has none to certify; the requirements' §4.2 describes it as a per-window march mode and **that is not what it is**. **W226.**
- **The second referent of §5.4 was not run** — the single-domain monolith, against which the composed classical answer carries its own composition defect. CS-19 did not run it either and neither page claims it.
- **The mixed marches are 40 macro-steps**, which is a sixth of the spin-up and far short of anything settled; they are a cost-and-divergence measurement, not a settled-state one.
- **The thread ratio is two draws $20\%$ apart** and is reported as a draw rather than a $15\%$ win (**W215**'s class).
- **Nothing was downloaded, no machine was rented, NeuberNet was not loaded, and nothing was pushed.**

## See Also

- [[case-study-racelab-graph-atlas-0.1]] — phase 1: the car, the decomposition, the graph and the march this one switches experts inside, and where W222 was opened.
- [[defect-correction-learned-operator]] — Theorem 1, the certified mode's whole claim, and the pre-registration discipline.
- [[corrupted-checkpoint-and-jacobian-fidelity]] — what the checkpoint's learned content is worth against the noise floor of the object carrying it.
- [[learned-contribution-kill-tests]] — the fourteen formulations, and why the certified slot is the one that can be asked at all.
- [[prior-art-and-novelty-atlas-0.1]] §4 — the assumption §5.1 measures false.
- [[poc2-frontwing-results]] — `Engine._absorb`, W145, and the OUTSIDE THE MODEL stamp this tier adopts.
- [[vehicle-scale-and-sizing]] §2.3 — the similarity §2.2 applies one level on.
- [[case-study-ladder-to-f1]] §26 — where this sits, and what it does not move.
- [[gap-worklist]] Tier 52 — W222 closed, W223 to W226 opened.
