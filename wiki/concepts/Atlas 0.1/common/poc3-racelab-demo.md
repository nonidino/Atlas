# PoC 3 — the demo, and the arms re-run on the column it marches

**Type:** Case study — **PoC 3's phase 3**, and CS-19's arms re-run on the repaired column that the demo marches (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-12, Tier 53. `atlas/demo_racelab/`, `scripts/tier53_racelab_rerun.py`, `out/racelab3/racelab3.json`, `tests/test_tier53_racelab_demo.py`. Worklist rows **W227**–**W229** (opened), **W222** (annotated).
**Related:** [[case-study-racelab-graph-atlas-0.1]] · [[case-study-racelab-switch-atlas-0.1]] · [[case-study-vehicle-march-atlas-0.1]] · [[vehicle-scale-and-sizing]] · [[poc2-frontwing-results]] · [[poc2-demo-and-novelty]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[defect-correction-learned-operator]] · [[prior-art-and-novelty-atlas-0.1]]


# 0. The result, in one paragraph

**The two RaceLab pages now describe one vehicle, the demo marches it, and closing the gap took three attempts because the envelope check kept finding things.** CS-19's five arms are re-run at the same horizon, with the same five arms, the same settle fraction and every threshold inherited unchanged — **all five now inside the envelope, against CS-19's every arm outside for every macro-step.** **P3 passes at $\mathbf{7.954\times10^{-7}}$ where CS-19 failed at $3.482\times10^{-6}$, and it is the confirmation rather than the pass that matters**: CS-20 diagnosed that failure as Jensen's term over an unsettled window and this tier pre-registered *"if P3 still fails, the CS-20 diagnosis is wrong"* — the pointwise residual is $1.877\times10^{-16}$, the averaged one is $7.954\times10^{-7}$, and Jensen's term is $7.956\times10^{-7}$, **a ratio of $0.99973$**, with no threshold touched. P2 passes at $0.002401$ against $0.075$ with its null at exactly $1.0000$; P4 closes to $5.517\times10^{-9}$ against $0.997772$ with the term withheld; P5 is bitwise in every crossing quantity and in the field; P6 splits exactly as CS-19's did, the lagged column at $\mathbf{0.3232}$ s under the $0.5$ s ceiling and within $4\%$ of that tier's number on a different machine sizing, a different release state and with the check running. **The machine's heat falls by a factor of $\mathbf{949}$** — $3982$ W motoring to $4.198$ W generating — **and the block runs $438$ K cooler**, its thermal time constant returning to $50.8$ s, within $1.4\%$ of the $50.1$ s CS-18 measured; the aerodynamics move by under $8\%$, which is the powertrain's stiffness seen from the other side. **The re-run took three attempts and the middle one is the finding.** Sizing the machine for the host's inflow *at one instant* — CS-20's repair — admits the release state and is **declined at macro-step 312**, because $u_{\text{rotor}}$ falls $6.3\%$ over the horizon while the machine's margin over its own crossover is $3.3\%$ of $u$; sizing for the horizon's **minimum** admits all 600, verified on a twelve-minute lagged march before a ninety-minute commitment. **So a declared envelope is a statement about a state, and this package has been checking states and assuming trajectories.** And **phase 3 is built**: `python -m atlas.demo_racelab --open`, driven through its own socket, showing the car, the fourteen windows, the switch, and the three things CS-20 said a dashboard must not hide — the $1/32$ lead ratio beside the switch, the `OUTSIDE THE MODEL` stamp, and four of five families greyed out with their reasons.

---

# 1. Why the re-run and the demo are one tier

The demo marches a column. **If that column's joins do not conserve, nothing on the screen means anything** — so the re-run is not bookkeeping beside the demo, it is the demo's own referent being certified.

CS-19 measured five arms on a graph whose powertrain was outside its declared envelope for every macro-step of every one of them. CS-20 built the envelope check, found that, repaired the machine — and did not re-run the arms, so the two RaceLab pages described two different vehicles. This closes it, and the closing turned out to be more interesting than the re-run.

---

# 2. The sizing is a range, not a value, and the check found that too

## 2.1 What happened

The re-run was launched with CS-20's repair: the machine sized for the inflow its host delivers, $u = 0.66915$, which is CS-19's measured value **at one instant**, at CS-18's own margin of $1.0326$ over the machine's crossover.

**The referent arm was declined at macro-step 312.**

```
EnvelopeDeclined: the march left a declared envelope at macro-step 312 --
ROTOR: the induction 0.02 is AT the clamp `clamp_induction` declares,
so the disk has no operating point here
```

## 2.2 Why, measured

A 600-step lagged march at that sizing, with the check recording rather than stopping:

| macro-step | $u_{\text{rotor}}$ | induction | loop current |
|---|---|---|---|
| 0 | $0.67208$ | $0.12665$ | $+0.03168$ |
| 100 | $0.66717$ | $0.10487$ | $+0.02522$ |
| 200 | $0.65867$ | $0.06271$ | $+0.01404$ |
| 300 | $0.65170$ | $0.02317$ | $+0.00487$ |
| 400 | $0.64361$ | $\mathbf{0.02000}$ | $\mathbf{-0.00578}$ |
| 600 | $0.63000$ | $0.02000$ | $-0.02367$ |

**$u_{\text{rotor}}$ falls $6.3\%$ over the horizon and the machine's margin over its own crossover is $3.3\%$ of $u$.** The margin is consumed at macro-step 312, the induction sits on its clamp floor from there, and the current crosses sign near 300 — the machine goes back to motoring, which is exactly the failure CS-20 repaired, arriving later instead of at step 0.

**The flow is still settling, and this graph's flow does not settle.** [[case-study-racelab-graph-atlas-0.1]] §6.2 measured the fluid band still at $5.8\%$ after 1600 lagged macro-steps. A machine sized at one instant of a flow that never stops moving is sized for a number that is not there any more.

## 2.3 The repair, and it is a different kind of statement

**Size for the horizon's minimum inflow, not its release value.** The similarity puts the induction at its reference value at the sizing inflow and above it everywhere the inflow is higher, so sizing at the minimum admits the whole range.

$$u_{\text{size}} = \min_{\text{horizon}} u_{\text{rotor}} = 0.63, \qquad k_e \to k_e/x, \qquad R \to R/x^3, \qquad x = u_{\text{size}}/u_{\text{ref}}.$$

**Verified on a lagged march before any arm was launched**, which is the order that matters — a twelve-minute check before a ninety-minute commitment:

| | sized for the release value | **sized for the horizon's minimum** |
|---|---|---|
| macro-steps outside, of 600 | $\mathbf{288}$ | $\mathbf{0}$ |
| declined first at | macro-step $312$ | — |
| $u_{\text{rotor}}$, start $\to$ min | $0.67208 \to 0.63000$ | $0.67208 \to 0.61753$ |
| induction, min | $\mathbf{0.02000}$ — the clamp | $\mathbf{0.04767}$ — $2.4\times$ the clamp |
| loop current, min | $-0.02367$ — **motoring** | $\mathbf{+0.008691}$ — generating throughout |

**The sizing is still one Picard step and the page says so.** Sized for $0.63000$, the new trajectory's minimum is $0.61753$ — a $1.98\%$ residual. **The margin the sizing creates absorbs it**: the induction at that minimum is $0.0477$ against a clamp floor of $0.02$. That is what a margin is for, and it is the difference between a sizing that has one and CS-20's, which did not.

## 2.4 What this supersedes

**CS-20 §2.2's repair is measured at the release state and is not a statement about a horizon.** That page reported the Picard residual rather than iterating it away, and said the sizing was a fixed point taken one step. **This is how big that omission was: 288 of 600 macro-steps.**

It does not change anything else on that page — the similarity is still exact, the $x = 1$ control is still the identity bitwise, the induction still returns to $0.11388706$ at the sizing inflow. What changes is **which inflow to size at**, and that is **W228**.

---

# 3. The check has now earned its keep three times

It was built to close W222, which was one finding. It has since produced three more, none of which anyone was looking for:

| # | what it found | where |
|---|---|---|
| 1 | the machine motoring from macro-step 0 on CS-19's column | CS-20 §2 |
| 2 | every CS-19 arm releasing from a freestream whose transient leaves the disk's clamp for 58 macro-steps | CS-20 §3 |
| 3 | **the repaired machine's margin consumed by the inflow's own drift at macro-step 312** | §2 above |

**[AI Inference]:** the pattern in all three is the same and it is not about this vehicle. A declared envelope is a statement about a *state*, and every one of these was a case of a quantity being *checked at one state and assumed over a trajectory*. A framework that evaluates `validity` at a release state and never again is making that assumption everywhere. Unmeasured on any other graph in this package, and it is **W229**.

---

# 4. The arms, re-run

Five arms at 600 macro-steps, released from the settled field, with `enforce=True` and the machine sized for the horizon's inflow range. **Every threshold is `racelab.GATE`'s**, which is CS-18's; the horizon, the settle fraction, the five arms and `vehicle_march.receiver_balances` are CS-19's, unchanged.

## 4.1 Every arm is inside the envelope

| arm | wall | s / macro-step | outside |
|---|---|---|---|
| referent, tight | $3411.6$ s | $5.686$ | $\mathbf{0}$ |
| **repeat**, tight | $1120.3$ s | $\mathbf{1.867}$ | $\mathbf{0}$ |
| all lagged | $193.9$ s | $\mathbf{0.323}$ | $\mathbf{0}$ |
| null J3, tight | $965.2$ s | $1.609$ | $\mathbf{0}$ |
| null J1, tight | $988.9$ s | $1.648$ | $\mathbf{0}$ |

**Zero, against CS-19's every arm outside for every macro-step.** That is what the tier was for.

**The referent's wall time is contaminated and the repeat is the clean reading** — $3411.6$ s against $1120.3$ s for two computations that returned bitwise identical answers, a factor of $\mathbf{3.05}$. This is [[case-study-vehicle-march-atlas-0.1]] §3.4's own experience: *"the $422.6$ s is contaminated... and the repeat is the clean reading."* A five-step probe taken before the run predicted $2.038$ s a macro-step and the clean arm read $1.867$ — **within $9\%$** — so the probe was sound and the arm was not. The cost to quote is the repeat's.

## 4.2 P3 passes, and CS-20's diagnosis is confirmed to four figures

**This is the result the tier was most able to be wrong about.** CS-19 failed P3 at $3.482\times10^{-6}$ against an inherited $10^{-6}$. CS-20 diagnosed that as Jensen's term — the clause averages a *concave* map over the settle window, so it measures the window's variance times the curvature — and this tier's pre-registered prediction said so in terms that could have refuted it:

> **If P3 still fails, the CS-20 diagnosis is wrong** and that is the most informative outcome available here.

| | CS-19 | **this run** |
|---|---|---|
| tracking residual | $3.482\times10^{-6}$ — **fail** | $\mathbf{7.954\times10^{-7}}$ — **pass** |
| the **pointwise** residual | $0.0$ | $\mathbf{1.877\times10^{-16}}$ |
| the window's relative variance in $u$ | $1.24\times10^{-5}$ (at 120 steps) | $9.945\times10^{-6}$ |
| Jensen's $\tfrac12 p(p-1)\mathrm{Var}(u)/\bar u^2$ | $9.931\times10^{-7}$ | $\mathbf{7.956\times10^{-7}}$ |
| **averaged residual over Jensen's term** | $0.9995$ | $\mathbf{0.99973}$ |

**The averaged residual is Jensen's term to four significant figures, and the identity holds pointwise to machine zero.** The clause moved from fail to pass with **no threshold touched**, purely because the window it averages over is steadier — which is precisely what the diagnosis said would happen. **A tier confirming a previous tier's inference with a test that could have refuted it is the strongest thing available here**, and it is worth more than the pass.

## 4.3 P2 passes, and its residual no longer *is* the velocity gap

| | CS-19 | this run |
|---|---|---|
| relative residual with the term | $0.010130$ | $\mathbf{0.002401}$ |
| the null arm | $1.0000$ | $\mathbf{1.0000}$ |
| $u$ on the ring / $u$ in the cell | $1.0101289$ | $0.9959669$ |
| **what the velocity gap alone predicts** | $0.0101289$ | $0.0040331$ |
| measured over predicted | $\mathbf{1.0000}$ | $\mathbf{0.595}$ |
| the ratio's own band over the settle window | $0.00116$ | $0.00468$ |

**Two things changed and both are worth saying.**

**The sign went back.** CS-19 had $u_{\text{plane}} > u_{\text{ring}}$ and this run has $u_{\text{plane}} < u_{\text{ring}}$, which is CS-18's direction — the device's own force slowing the air it is about to work on. The repaired machine extracts $2.5\times$ the thrust ($0.02248$ against $0.00914$), and a device that extracts more slows what is in front of it more.

**And the gap now over-predicts by $1.68\times$**, where on CS-19's column it accounted for the whole residual to five significant figures. So on this column **the ring-to-plane gap is no longer a complete explanation of J3's residual** and something partially offsets it. Not diagnosed. **W231.**

**And the residual is below its own band.** $0.002401$ against a per-macro-step ratio band of $0.004683$ — so P2 **passes comfortably and is not resolved**, where on CS-19's column it sat at $8.8\times$ its band. Reported as not resolved rather than as a number.

## 4.4 P4, P5, P6

| clause | verdict | the number |
|---|---|---|
| **P4 · J2's receiving balance** | **pass** | $5.517\times10^{-9}$ with the mount term against $\mathbf{0.997772}$ with it removed, a factor of $1.8\times10^{8}$ |
| **P5 · the repeat floor** | **pass** | bitwise in every crossing quantity **and in the field**, over 600 macro-steps |
| **P6 · the macro-step cost** | **splits, as before** | lagged $\mathbf{0.3232}$ s **passes** the $0.5$ s ceiling; tight fails it, at $1.867$ s on the clean arm ($3.7\times$) and $5.686$ s on the contaminated one |

P6's split is CS-19's, and the lagged column's $0.3232$ s is within $4\%$ of that tier's $0.3363$ — **on a different machine sizing, a different release state and with the envelope check running**, which is a useful incidental control on all three.

## 4.5 The vehicle, CS-19 beside this run

| settled quantity | CS-19 | repaired | ratio |
|---|---|---|---|
| $u$ at the turbine ring | $0.669153$ | $0.618859$ | $0.925$ |
| $u$ through the core | $0.688000$ | $0.664755$ | $0.966$ |
| $UA$ | $31.140$ | $38.001$ | $1.220$ |
| induction | $0.0200$ — the clamp | $\mathbf{0.05540}$ | $2.77$ |
| loop current | $\mathbf{-0.560451}$ | $\mathbf{+0.010235}$ | **sign** |
| **the machine's heat** | $\mathbf{3981.97}$ W | $\mathbf{4.198}$ W | $\mathbf{1/949}$ |
| downforce | $0.426394$ | $0.395580$ | $0.928$ |
| drag | $0.878171$ | $0.846299$ | $0.964$ |
| rotor thrust | $0.009140$ | $0.022480$ | $2.46$ |
| shaft power | $0.006118$ | $0.013913$ | $2.27$ |

**The machine's heat falls by a factor of $\mathbf{949}$** — from a motoring machine dissipating four kilowatts into the block to a generating one putting in four watts. The prediction recorded before the run said *"about two orders... $I^2$ about $320\times$ smaller"*; the mechanism was right and the number was three times out, because the settled current came in smaller than the short march suggested.

**And the block runs $\mathbf{438}$ K cooler.** On the coolant clock the wall settles at $303.18$ K against CS-19's $741.08$, the block takes $178.3$ W against $2700.5$, and the thermal time constant falls from $56.9$ s to $\mathbf{50.8}$ s — back within $1.4\%$ of the $50.1$ s CS-18 measured on the front wing's graph. **CS-19's block was being cooked by an expert operating outside its declared envelope**, and every thermal number on that page is a number about that.

**The aerodynamics barely moved**: downforce $0.928$, drag $0.964$, $u$ through the core $0.966$. The powertrain's excursion was catastrophic for the powertrain and the thermal loop and nearly invisible in the flow — which is CS-18 §7.2's elasticity of $63.3$ seen from the other side.

---

# 5. The demo

```bash
python -m atlas.demo_racelab --open
```

A single page at <http://127.0.0.1:8013/>, vanilla HTML/CSS/JS in one file, FastAPI and uvicorn behind it, a background engine thread marching continuously and frames pushed over a WebSocket — the pattern [[poc2-frontwing-results]]' demo established and the only one in this project proven to run on a machine that has nothing.

**The layout is the requirements' §6.** The car and the field overlay in the centre with window boundaries and mode tints; the switch, the field selector and the march controls on the left; the window inspector on the right; the global telemetry along the bottom; the expert names and their licences in the header, always visible.

## 5.1 The three things it is built to make impossible to miss

**1. The lead ratio, beside the switch.** The composition layer exchanges four times a macro-step and one window spans $2.0$ length units, so a learned window is asked for $\tfrac{1}{32}$ of the step Poseidon-T was trained to take — and [[case-study-racelab-switch-atlas-0.1]] §5.1 measured the error at a *minimum* at the native lead and $4.3\times$ that minimum at an eighth of it. **A viewer flipping a window to *learned* is watching an expert asked for the thing it is worst at**, and the panel says so before they click. That is **W225**, answered by showing the number rather than by hiding the column.

**2. The envelope stamp.** Every declared predicate is consulted every macro-step. The engine runs with `enforce=False` *by design* — a dashboard that stops reports nothing — and the moment a predicate declines, the stage border turns red and a banner names the expert, the reason and the macro-step, and says that every number below it is a measurement of a state the model declines to stand behind. The all-classical column stays inside; **every learned column leaves within a handful of macro-steps**, which is the demo's most honest frame.

**3. Four of five families have no learned option**, listed with the reason for each. A viewer should leave knowing that, because it is a true and important fact about the field rather than about this code.

## 5.2 What it showed when it was driven

Measured through the socket on this host, all-classical unless stated:

| | |
|---|---|
| march | $0.675$ s a macro-step, with a second all-classical column held beside it at $0.651$ s |
| envelope | **0 macro-steps outside** |
| the machine | $u_{\text{rotor}} = 0.67112$, $I = +0.03041$, $30.7$ W — **generating** |
| compile | `refuse`, `L7/R9` alone, 26 agents, 36 seams |
| **measure windows** | 14 windows, median one-step error $\mathbf{0.351}$ at $0.125\times$ native |
| **three windows flipped to learned** | ledger $11/3$, speed $\mathbf{0.259\times}$, rms field error $0.065$ and climbing |
| **`certified` selected** | refused as a march mode; the window stays classical and the page says why |

## 5.3 What is not wired, and why that is the honest choice

**The requirements' §3.3 lists thirteen parameters and this page exposes none of them as sliders.** §3.3's own rule is that *"a slider that moves a number nothing reads is worse than no slider: if a parameter cannot reach its subsystem, it is omitted and the omission is recorded."*

Phase 1 wired three geometry knobs into `CarParams` and stopped there. A parameter that moves a body can move it across a window boundary, and the decomposition is **static** — that is E1, a fixed graph, and §4.1 says a parameter that would change it is either clamped or the window set is rebuilt with a visible *recompiling* state. **Neither is built**, so no slider is shown and the omission is here. That is **W230**.

---

# 6. What this tier did NOT do, named

- **No rung moved and no expert was trained.** Poseidon-T is frozen and loaded from the local cache with the hub offline.
- **[[case-study-ladder-to-f1]] §14.3 is unchanged**, and the demo's README says in its own words that it is not evidence a learned expert pays.
- **The sizing is still one Picard step** (§2.3). Sized for $0.63000$, the trajectory's minimum is $0.61753$ — a $1.98\%$ residual the margin absorbs. **It is not iterated to convergence** and doing so would need a settled flow, which this graph does not have.
- **CS-19's page is not rewritten.** It stands as the record of what the un-repaired graph measured, with a dated note pointing here — the vault's own practice with [[joining-seam-cost]] §6.4.
- **CS-20 §2.2 is superseded in one respect only** — which inflow to size at — and annotated rather than rewritten, because everything else on it is true of what it measured.
- **No parameter sliders** (§5.3, **W230**), **no three dimensions** (phase 4) and **no bundle** (phase 5).
- **The demo was driven through its own socket, not through a browser.** Nobody has looked at the page with human eyes; what is verified is that every endpoint, message and number the page reads is produced and correct.
- **Section 5.4's second referent, the single-domain monolith, is still not run**, here or on either previous page.
- **Nothing was downloaded, no machine was rented, NeuberNet was not loaded, and nothing was pushed.**

## See Also

- [[case-study-racelab-graph-atlas-0.1]] — phase 1: the car, the decomposition, the graph, and the arms this tier re-runs.
- [[case-study-racelab-switch-atlas-0.1]] — phase 2: the switch, the envelope check, and every number the screen displays.
- [[poc2-frontwing-results]] and [[poc2-demo-and-novelty]] — the demo this one is modelled on, and what a PoC in this project is for.
- [[case-study-vehicle-march-atlas-0.1]] — the union, the three joins, and the gate whose thresholds are still the ones being met.
- [[gap-worklist]] Tier 53 — W227 to W230.
- [[case-study-ladder-to-f1]] §27.
