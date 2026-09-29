# The traced car, and the arms re-run on it

*Case study, PoC 3. New 2026-09-13.*

---

# 0. The result, in one paragraph

**The car was a scatter of disconnected plates that did not read as a car, and it is now a traced silhouette.** A CC0 Formula One side view was rasterised, thresholded, simplified and mapped **isotropically** into the cell box; the shell is one welded chain, the wheels come from their own tyre contact patches, and the radiator duct moved down with the body because a real Formula One body **tapers** behind the cockpit and the old duct band put its roof where a real engine cover has no metal. Every marched number then described a different vehicle, so all five arms were re-run. **All five pass their pre-registered clauses with every threshold inherited unchanged, and every arm is inside the envelope for every macro-step.** But the horizon moved, and **that is the finding**: the traced car is blockier, accelerates the flow more, and breaches the window expert's declared cell-Reynolds bound at macro-step $554$, so CS-19's $600$ macro-steps are not available to it and the arms run at $530$. **The most useful thing in the tier is a failure of this tier's own process**: the sizing probe recorded that the envelope would object on $46$ macro-steps, the gate read a different number, and a twenty-five-minute referent arm was lost to a decline the probe had already predicted.

---

# 1. The car

## 1.1 What was wrong

Plotted from `car_bodies()` rather than from the screen, the old car was eleven plates and two rings with nothing joined to anything:

- **the nose sloped the wrong way** — `alpha = -7` put the trailing edge *lower*, where a modern Formula One nose rises rearward from a low tip to the chassis;
- it **ended in mid-air** at $x = 175$ with the sidepod starting at $x = 220$, a 45-cell hole;
- **the body had no rear** — `POD_UP` ended $(419, 66.6)$ and `POD_LO` $(419.7, 50.5)$, a 16-cell gap never closed;
- **the floor ran straight through both wheels.**

The renderer was faithful; there was simply no car outline in the model to draw.

## 1.2 What it is now

| | |
|---|---|
| source | freesvg.org id 48844, **CC0 / public domain** |
| method | rasterised $1200\times310$, thresholded to a silhouette, upper and lower profiles simplified with Douglas–Peucker and **welded into a chain** |
| map | $x = 94.0 + x_{\text{raster}}\cdot 0.3760$, $y = (309 - y_{\text{raster}})\cdot 0.3760$ — **isotropic**, so the car is not distorted |
| parts | 22 plates and 2 wheels, 46 flat bodies (was 11 and 2, 35 flat) |

The wings are **not** traced: they stay the calibrated aerofoils, because `FW_MAIN` carries the front wing's real chord ($0.25$ m) and the structure CS-12 built at it. The wheels **are** traced — centres from their contact patches, radius from their crowns.

## 1.3 Three compromises, each with its reason in the code

- **The roll hoop's spike is clipped** to a flat crown at $y = 105.7$. The trace reaches $116.2$ and the tiling's horizontal seam is at $y = 112$; two 128-tall rows in a 240-tall box force row offsets of exactly $0$ and $112$, so the seam cannot move and a body crossing it would have its force split between two experts that exchange only a ring. The clipped car clears it by $6.3$ cells.
- **The diffuser is not traced.** On a centreline slice the ground-touching rear tyre occupies the floor's exit — at the floor's height the tyre spans $x = 411.6$ to $463$ — so a traced ramp would sit inside the wheel and double-count its drag. It keeps its own knob and starts behind the tyre.
- **The car is shifted back and scaled so the tyres clear two things**: the calibrated front wing (its trailing edge is $109.9$, the tyre's leading edge $111.3$) and the turbine cut. With $w_x = 128$ and `device_overlap_max` $= 16$, two device cuts are forced exactly $112$ cells apart, which put the turbine cut **inside** the traced rear wheel until the car was sized up.

---

# 2. What the geometry change broke, and what that was worth

## 2.1 W235 — a stamping box that shrank as the body leaned down

`wing_fsi.box_y` was $\tfrac12 c\sin\alpha/\mathrm{d}x + 10$: the **signed** sine. A body leaning nose-down spans just as much of $y$ as one leaning up, but with the signed sine the box *shrank* as $\alpha$ went negative — a plate dropping more than about $11$ cells got `box_y <= 0` and `torch.arange` raised, and one dropping less got a box too small to hold its own kernel, which renormalises over the clipped region, so the force stayed **conserved** but was squeezed into too narrow a band of $y$.

**This is why the old car could not have a realistically tapering rear**: every engine cover that falls is a negative-alpha plate. Only `racelab` had any, so the fix — $\lvert\sin\alpha\rvert$ — has this case as its whole blast radius.

## 2.2 The structural move, and it was the small one

The layout is **derived** from the geometry, so a new car risked a new graph. It did not get one: **14 windows, the same names**, with the cuts moving to $112/174/286/398/496/544$ and the banded force fraction going $26.9\% \to 31.3\%$. What did have to move is `DUCT_Y0`, from $44$ to $18$ — the traced body spans $y = 9.8$ to $60.6$ at the duct's tightest station, and a 32-cell duct at $44$–$76$ poked out through the engine cover by $16$ cells with the turbine plane floating **outside the car**. The device planes' $x$ positions still sit on tiling seams, so only the rows the devices occupy moved.

---

# 3. The re-sizing

`U_DUCT` is measured here, not written down. W228's procedure — size for the horizon's **minimum**, not for one instant — applied to the traced duct:

| | Tier 53 (hand-drawn) | **Tier 54 (traced)** |
|---|---|---|
| $u_{\text{rotor}}$ over the horizon | falls $6.3\%$ | falls $\mathbf{2.01\%}$ |
| sized for | $0.63000$ | $\mathbf{0.654077}$ |

**The traced duct sits in a steadier part of the flow.** It moved down to $y = 18$–$50$, nearer the floor, and the margin W228 found being consumed is three times larger here.

---

# 4. The horizon moved, and the process failed before the physics did

## 4.1 What happened

The first arms run was **declined at macro-step 552** — not by the powertrain, which the re-sizing had fixed, but by the **fluid**: $u_{\max} = 2.04971$ gives a cell Reynolds number of $8.007$ against the window expert's declared bound of $8$.

**The sizing probe had already said so.** It marched all $600$ macro-steps unenforced and recorded `outside_the_envelope_steps: 46`. The tier printed the $u_{\text{rotor}}$ band out of that result, ignored that field, and gated the arms on an **eighty-step** enforced verify against a **six-hundred-step** horizon.

**Two recorded lessons at once.** *Positive controls need a horizon* — march a repair as far as the failure it repairs. And the sharper one: **a gate that reads a different quantity from the one that predicts the failure is not a gate.** The number that would have stopped this was in the same dictionary the tier read its sizing out of.

Fixed in the script and not only in this run: `verify` now marches the **whole** horizon the arms will use, `arms` **refuses** to run at any horizon other than the verified one, and the probe's outside-count and first outside step are recorded and acted on.

## 4.2 The physics underneath

| | |
|---|---|
| first breach | macro-step $\mathbf{554}$, $u_{\max} = 2.05072$, cell Re $\mathbf{8.011}$ vs bound $8$ |
| breached on | $46$ of $600$ macro-steps |
| horizon used | $\mathbf{530}$, with $24$ macro-steps of margin |

The traced car is **blockier** — bigger wheels ($r = 33.5$ against $28$), an airbox, and a floor that runs *between* the wheels instead of through them — so it accelerates the flow more. **CS-19's $600$-macro-step horizon is not available to this car, and the bound was not touched.** Every comparison below is a comparison at a shorter horizon and says so.

**The breach is $0.1\%$ over the bound**, and the envelope caught a geometry change's consequence $554$ macro-steps downstream of the change. That is the check doing exactly what W222 built it for, for the fifth time.

---

# 5. The arms

**Every arm inside the envelope for every macro-step**, at $530$ macro-steps, with every threshold inherited from CS-18 unchanged.

| clause | verdict | the number |
|---|---|---|
| **P2 · J3's receiving balance** | **pass** | $0.036194$ against $0.075$; the null at exactly $1.0000$ |
| **P3 · J1's parametric check** | **pass** | $\mathbf{1.853\times10^{-7}}$ against $10^{-6}$; the null pins $UA$ at $UA_{\text{RAD}} = 42$ |
| **P4 · J2's receiving balance** | **pass** | $1.081\times10^{-8}$ against $10^{-6}$ — **its null arm was not run in this tier** |
| **P5 · the repeat floor** | **pass** | bitwise in every crossing quantity **and in the field** |
| **P6 · the macro-step cost** | **splits, as always** | lagged $\mathbf{0.3955}$ s under the $0.5$ s ceiling; tight $1.9824$ s |

## 5.1 P2's residual is nearly the velocity gap again

This clause has now told three different stories and the third is the interesting one:

| | residual | what the ring-to-plane velocity gap predicts | ratio |
|---|---|---|---|
| CS-19 | $0.010130$ | $0.0101289$ | $1.00001$ — five figures |
| Tier 53 | $0.002401$ | $0.0040331$ | $1.68$ |
| **Tier 54** | $\mathbf{0.036194}$ | $0.039729$ | $\mathbf{1.098}$ |

**W231 asked why the residual stopped being the velocity gap. On the traced car it very nearly is again, to within $10\%$** — and the residual is $15\times$ larger than Tier 53's while still passing at half its tolerance. The ratio's own per-step band is $0.055\%$ of its mean, so $0.036$ sits far above the band: this is a resolved quantity, not noise. **Not diagnosed**, and the three-row table is the useful object rather than any one of its rows.

## 5.2 The vehicle

| settled | Tier 53 | **Tier 54** |
|---|---|---|
| $u_{\text{rotor}}$ | $0.61886$ | $0.64534$ |
| $u_{\text{core}}$ | $0.66476$ | $0.60814$ |
| $UA$ | $38.001$ | $37.589$ |
| induction | $0.055405$ | $0.070859$ |
| loop current | $+0.010235$ | $+0.015013$ |
| machine heat | $4.198$ W | $8.005$ W |

**The machine generates on both cars**, and on the traced one it does so with $28\%$ more induction and half again the current — it is further from its own crossover than the hand-drawn car's was, which is the same fact as §3's steadier duct seen from the electrical side.

## 5.3 The cost

The tight column costs $1.9824$ s a macro-step against Tier 53's clean repeat at $1.867$ — **$6\%$ more for $31\%$ more bodies**, so the solver and not the stamping still dominates this column. The lagged column is $0.3955$ s against $0.3232$, which is $22\%$ more and still under the ceiling.

---

# 6. What this tier did NOT do, named

- **CS-19, CS-20 and Tier 53 are not rewritten.** They are the record of what three earlier vehicles measured. This page is the fourth, and the horizon differs, so the numbers are not interchangeable.
- **P4's null arm was not run.** The clause passes at $1.081\times10^{-8}$ with the term in; the arm with it withheld — CS-18's third column — was not marched here, so P4 passes **without its control** in this tier and Tier 53's $0.997772$ is the last time that control was taken.
- **The horizon is $530$ and not $600$**, so every clause is measured over a shorter settle window than CS-19's, and P3's Jensen term in particular is a function of that window.
- **Nothing was downloaded except the CC0 drawing**, which was authorised; no machine was rented; NeuberNet was not loaded; nothing was pushed.

---

## See Also

- [[case-study-racelab-graph-atlas-0.1]] — phase 1, whose arms these are.
- [[case-study-racelab-switch-atlas-0.1]] — phase 2, and W228's sizing procedure.
- [[poc3-racelab-demo]] — phase 3, the dashboard this car now appears on.
- [[poc3-racelab-3d]] — phase 4, the half-car swept out of this silhouette.
- [[gap-worklist]] — W235 to W239.
