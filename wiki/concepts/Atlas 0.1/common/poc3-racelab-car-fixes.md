# The drawn car's two faults, and the fixes measured on copies

*Case study, PoC 3. New 2026-09-13. Tier 58.*

---

# 0. The result, in one paragraph

**Tier 56 could not run the arms on the car the user drew, for two reasons — the machine (W244) and the fluid at the outflow (W245) — and an ablation there pointed at the closed rear box. The user chose to measure fixes before redrawing anything, so this tier marched five copies of the car, each spun up from the freestream in its own right:** the car as drawn (the control, bitwise Tier 56's), the car as drawn on the layout any fix derives, the rear box removed, the endplate alone removed, and the car in a box $128$ cells longer. **Neither fault is what it looked like.** *The fluid:* in every arm, at both sizings, on every macro-step, the car's own flow stays inside the window expert's cell-Reynolds bound; **only the outflow column breaks it**, and the wake the car sends across that column sets *when* — macro-step $56$ as drawn, $294$ without the endplate, $394$ without the rear box, $265$ in the longer box — and nothing tried removes it within $600$. *The machine:* the duct flow falls $14.7$ to $18.0\%$ over the horizon in every arm — with the box and without it, on either layout, in either box — so **Tier 56's attribution of that fall to the rear box is refuted**: its ablation removed the box from a flow the box had already shaped. And at W228's sizing the machine is outside its window only for the **first $5$ to $12$ macro-steps**, in every arm, and inside for the rest; Tier 56 read that probe by its first decline. **Removing the rear box buys $39\%$ more downforce, $26\%$ less drag and a fluid breach $338$ macro-steps later — and fixes neither fault**, while moving both device planes $14$ cells downstream, which on its own changes the duct flow by $6.2\%$. Four of eleven registered predictions failed, and the failures are the result.

---

# 1. What the user chose, and what was not touched

After Tiers 56 and 57 the user was asked how to proceed with the car, and chose to measure first: *leave the drawing untouched and test candidate fixes on copies, each spun up from the freestream and probed over $600$ steps, so the results are not confounded by the old wake, then choose with numbers.*

- **`atlas/cases/car_geometry.json` is unchanged**; its fingerprint is still `a1f67a8e`. Every arm is a deep copy of the geometry dict, and a test asserts that building one leaves the cached geometry and the fingerprint as they were.
- **No threshold, clamp, bound, sizing rule or source file was changed.** `scripts/tier58_car_fixes.py` is new and imports Tier 56's driver for its record I/O, its stall guard and its summariser of where a cell is — so the two tiers cannot disagree about what "the outflow column" means.
- **The model's FLUID predicate is read exactly as it is.** The two extra readings — the largest speed without the outflow column, and without the last eight columns — are judged by the same expression `RaceRollout.validity_report` evaluates, checked against it on both sides of the bound in a test.

---

# 2. What was read before the run, and how it shaped it

Four things were read out of Tier 56's record before the script was written. The record carries them under `read_before_this_run`, so they are not presented as found here.

1. **Tier 56's five saved snapshots**: no cell over the bound at macro-step $0$; $51$, $81$, $100$ and $113$ at macro-steps $150$, $300$, $450$, $599$ — **every one on the outflow column**, where $v$ reaches $-1.99$ to $-3.00$. Tier 56 §4 had placed the fastest cell there; this sharpened it to every over-bound cell. *(The record's wording, "about $-0.3$ one to two cells upstream", is loose: the next two columns carry a sawtooth of up to about $1$ in magnitude, and the columns beyond it about $-0.3$.)* *Consequence:* every probe here records the column-free speeds every macro-step.
2. **Tier 56's probe 2**, sized for the minimum: the rotor's induction was on a clamp on **$9$ of $600$** macro-steps, all at the start. Most of the $553$ steps that probe counted outside were the fluid's. *Consequence:* every probe here records each expert's declines on every macro-step, not the first decline of any.
3. **The layout is derived from the bodies, and it moves.** Removing the rear box, or the endplate alone, or lengthening the box, moves both device planes $14$ cells downstream. *Consequence:* the arm `drawn_moved_layout` — the intact car on the moved layout — and every fix is compared with it, not with the control, because a comparison that varies the car and the layout at once is not a control.
4. **The machine's window** from a uniform ring on a $0.005$ grid: $[0.975, 1.140]$, which §5 replaces with the exact edges.

---

# 3. The arms

| arm | what changed | columns | device planes | windows | flat bodies |
|---|---|---|---|---|---|
| `drawn` — **the control** | nothing | $0, 112, 160, 272, 384, 496, 544$ | $280$, $392$ | $14$ | $51$ |
| `drawn_moved_layout` — **the reference** | the layout only | $0, 112, 174, 286, 398, 496, 544$ | $294$, $406$ | $14$ | $51$ |
| `no_rear_box` | `RW_ENDPLATE`, `RW_ENDPLATE_LO`, `DIFF_EXIT` removed | $0, 112, 174, 286, 398, 496, 544$ | $294$, $406$ | $14$ | $48$ |
| `no_endplate` | `RW_ENDPLATE` removed | $0, 112, 174, 286, 398, 496, 544$ | $294$, $406$ | $14$ | $50$ |
| `long_box` | the box $800$ cells long, $128$ more behind the car | $0, 112, 174, 286, 398, 510, 560, 672$ | $294$, $406$ | $16$ | $51$ |

Per arm, in one code path: **spin-up**, $240$ macro-steps from the freestream with the declared machine (`racelab.settled_field` with the arm's own tiling and bodies — Tier 56's recipe); **probe 1**, $600$ macro-steps lagged and unenforced, sized for the release state; **probe 2**, $600$ at probe 1's minimum — W228's sizing — whenever probe 1's machine declined at all (the control marches $60$, enough to compare with Tier 56's probe 2). A settled field carries its arm's identity — the car's fingerprint, the box, the columns, the spin-up length — and a field for any other arm is refused.

---

# 4. The prediction, registered before any stage

Written to the record at 19:40:12 with the run's first entry and judged in code (`judge`) under the same ids. A test asserts the texts in the script are, character for character, the texts in the record.

| id | the prediction, shortened | outcome |
|---|---|---|
| C1 | the control reproduces Tier 56 bitwise | **confirmed** — §6 |
| F1 | in every arm the fastest cell is on the outflow column while FLUID declines; without that column the speed is inside on $\ge 90\%$ of steps, without the last eight on all $600$ | **confirmed**, in its strongest form: $100\%$, $600$ of $600$, $600$ of $600$ in all five arms |
| F2 | removing the rear box or the endplate moves the first FLUID decline by at most $15$ macro-steps | **falsified** — $56 \to 394$ and $56 \to 294$ (§7) |
| F3 | the longer box delays the breach by at least $60$ macro-steps, and it breaks on its own outflow column | **confirmed** — $56 \to 265$, on $x = 799$ on every declined step |
| M1 | moving the planes changes $u_{\text{rotor}}$ at release by more than $1\%$; the fall stays within $4$ points | **confirmed** — $-6.2\%$; fall $16.6\% \to 16.0\%$ |
| M2 | **the rear box carries W244**: without it the fall is under half the reference's and the first machine decline is after macro-step $100$ | **falsified** — fall $17.9\%$ against $16.0\%$, first decline at $22$ against $18$ (§8.1) |
| M3 | the endplate alone lies between the reference and the box removed | **falsified**, narrowly — its fall is $18.0\%$, above both ($16.0\%$, $17.9\%$) |
| M4 | the longer box does not help the machine | **confirmed** — fall $14.7\%$, first decline at $17$ |
| W1 | the uniform-ring window agrees with $\ge 98\%$ of Tier 56's recorded verdicts | **confirmed** — $1200$ of $1200$ |
| A1 | removing the rear box lowers the drag | **confirmed** — $1.419$ against $1.928$ |
| K1 | the longer box costs $1.10$–$1.35\times$ a step; removing plates costs within $5\%$ | **falsified, and uninformative** — §10 |

---

# 5. The machine's window, measured exactly

`racelab.machine_for_host` sizes the machine for an inflow $u_{\text{host}}$ by W199's similarity about $u_{\text{ref}} = 0.9225$: with $x = u_{\text{host}}/u_{\text{ref}}$,

$$k_e \to k_e/x, \qquad k_t \to k_t/x, \qquad R \to R/x^3 .$$

Under that scaling the operating point depends on the inflow only through the ratio

$$r = \frac{\bar u_{\text{ring}}}{u_{\text{host}}},$$

so the machine's envelope is a window in $r$. The disk is inside when the operating-point equation has its root $a^\ast$ within the induction clamp $[a_{\min}, a_{\max}] = [0.02, 0.40]$, rather than the induction being pinned to an end of it; the machine generates when its back-EMF exceeds the battery's open-circuit voltage, $k_e\,\omega > V_{oc}$. Stage `band` bisects both predicates, with a uniform ring, to $48$ iterations:

$$\text{rotor inside for } 0.9731 \le r \le 1.1408, \qquad \text{machine generating for } r > 0.9684,$$

so **the machine is inside for $r \in [0.9731, 1.1408]$, a span of $1.1722$**, and the edges agree at three different sizings to $1.3\times10^{-15}$ — the similarity, checked.

**The check is against the model's own verdicts, not against the sweep.** Read against the inflow Tier 56 recorded, the window reproduces **every** per-step ROTOR and MGU verdict of Tier 56's release-sizing trace and every one of its minimum-sizing probe — $1200$ of $1200$ — and in this tier it reproduces every measured machine verdict in all ten probes, with no disagreement. That holds although the real ring is not uniform, because the window is read against the *realised* inflow; it says nothing about the thrust's feedback on that inflow, which is why it is a diagnostic here and never a verdict.

What it explains: **a constant sizing can hold a horizon only if $\max u_{\text{rotor}}/\min u_{\text{rotor}} \le 1.1722$.** The release-sizing probes span $1.17$ to $1.22$ — the longer box's $1.1723$ misses by $10^{-4}$.

---

# 6. The control

The car as drawn, marched through this script's path — its own derived layout, `settled_field` with that tiling and the geometry file's bodies, a hand-stepped probe:

- the settled field is **bitwise** `out/racelab5/cache/settled.npz`;
- probe 1's rows equal Tier 56's stage-trace rows in $u_{\max}$, $u_{\text{rotor}}$, induction, current, load, drag, the declined list and the fastest cell's position on **all $600$ macro-steps**;
- probe 2, sized for exactly Tier 56's $0.44276205468922647$, equals Tier 56's probe 2 on all $60$ macro-steps marched;
- the derived layout is `racelab.layout()`.

So every arm below is comparable with Tier 56's record, and the extra readings demonstrably changed nothing in the march — a test also steps this script's march and Tier 56's `traced_march` side by side and requires every shared field to be equal.

---

# 7. W245 re-read: the car's own flow never leaves the envelope

The bound is $8\nu/h = 2.048$ free-stream speeds.

| arm | FLUID first, the model's reading | declined steps | fastest cell on the outflow column while declining | largest speed without the outflow column | without the last eight columns | $\lvert v\rvert$ on the outflow column at $550$ |
|---|---|---|---|---|---|---|
| as drawn | $56$ | $544$ | $100\%$ | $2.0322$ | $1.8189$ | $3.065$ |
| moved layout | $56$ | $544$ | $100\%$ | $2.0286$ | $1.8186$ | $3.064$ |
| endplate removed | $294$ | $306$ | $100\%$ | $1.5942$ | $1.5849$ | $2.072$ |
| rear box removed | $394$ | $206$ | $100\%$ | $1.5502$ | $1.5502$ | $1.791$ |
| longer box | $265$ | $153$ | $100\%$ | $1.9785$ | $1.9785$ | $1.771$ |

**Read the two column-free readings first.** With the outflow column excluded, no arm exceeds the bound on any macro-step at either sizing — ten probes, $5460$ macro-steps. With the last eight columns excluded, the car as drawn never again reaches even the speed it had at release, $1.8189$; its $2.03$ lies within seven columns of the edge, where $v$ carries a sawtooth. **The model declines correctly — the state it is handed has a boundary column carrying $\lvert v\rvert$ above $3$ — and the car's flow is not what it declines.**

**What the car does control is when the boundary breaks.** The breach moves by $238$ to $338$ macro-steps when a rear plate is removed, and the outflow column's $\lvert v\rvert$ grows more slowly in step, from release to macro-step $550$: $1.07 \to 3.07$ as drawn, $0.60 \to 1.79$ without the rear box. The closed rear box sends a taller, faster flow across the edge — the drawn car's fastest cell at release is above its wake, $x = 630$, $y = 122$, $113$ cells behind the rear wing, at $1.82$, against $1.45$ over the airbox without the box. In the longer box the column is quiet until the wake arrives ($\lvert v\rvert = 0.14$ at release against $1.07$), and the breach is intermittent — $153$ of the $335$ macro-steps after the first. Its fastest interior cells in the five snapshots sit above the wake at $x = 556$ to $758$, $y = 130$ to $144$ — partly in the added cells — and the largest interior speed over its probe is $1.98$, under the bound.

**So Tier 56 §4.2's ablation was right to call its fluid column uninformative, and wrong about the size of what it could not see**: released from the intact car's wake, removing the box moved the breach by three macro-steps; spun up without it, by $338$.

**What the solver says about its own outflow**, read from the code rather than inferred: RaceLab's column builds every window with `transmission="dirichlet"` (`ground_effect.solver_for`), under which `WindowNS` overwrites all four faces of a window's ring from the input on every sub-step — the physical outflow face included — and the solver's own docstring calls that choice *over-constrained* for an advection-dominated flow, because "prescribing velocity on an outflow boundary is ill-posed" and "the wake deficit … cannot leave the window". The same class implements `"characteristic"` (an outflow cell takes its interior neighbour's value) and `"convective"` transmissions and an `open_faces` selector; RaceLab uses none of them.

> **[AI Inference]:** that is very probably W258's mechanism. Nothing a window does updates the last column; the blend passes it through unweighted at the physical edge; only the global Leray projection — which extends the field downstream by $32$ cells of tapered fluctuation — corrects it, once per exchange, and nothing relaxes what it adds. A correction that accumulates where the wake's shear crosses a frozen column would grow at a rate the wake sets, which is the dependence measured above. Not measured: the discriminating test is an open outflow face marched against this tier's column-free reading as its control.

---

# 8. W244 re-read

## 8.1 The rear box does not carry the fall (W256)

| arm | $u_{\text{rotor}}$ at release | its fall over $600$ macro-steps | first machine decline, release sizing |
|---|---|---|---|
| as drawn | $0.5317$ | $16.6\%$ | $20$ |
| moved layout | $0.4986$ | $16.0\%$ | $18$ |
| endplate removed | $0.5465$ | $18.0\%$ | $19$ |
| rear box removed | $0.5939$ | $17.9\%$ | $22$ |
| longer box | $0.4942$ | $14.7\%$ | $17$ |

**Removing the rear box raises the duct flow by $19\%$ and does not make it fall less** — it falls slightly more. Tier 56's ablation found no machine decline in $200$ macro-steps without the box, and that is not reproduced from the freestream: the first decline is at $22$, against the reference's $18$. The ablation removed the plates from a field the box had already shaped.

> **[AI Inference]:** what it measured was the duct's trapped outflow draining after the box vanished — an inflow rising back into the window for a while — rather than a steadier duct. The fall itself is common to every car shape tried here, so it belongs to something every arm shares, and the record does not say what.

## 8.2 At W228's sizing the machine is outside only at the start (W259)

| arm | sized for | machine declined on | longest run inside | $u_{\text{rotor}}/\text{sizing}$ at $599$ |
|---|---|---|---|---|
| as drawn — Tier 56's own $600$, read here | $0.442762$ | macro-steps $0$–$8$ | $591$ | $0.9737$ |
| moved layout | $0.418214$ | $0$–$7$ | $592$ | $0.9750$ |
| endplate removed | $0.447436$ | $0$–$11$ | $588$ | $0.9746$ |
| rear box removed | $0.486930$ | $0$–$10$ | $589$ | $0.9743$ |
| longer box | $0.420920$ | $0$–$4$ | $595$ | $0.9742$ |

The machine never motors at this sizing: MGU declines on no macro-step in any arm. What declines is the rotor, **on its upper clamp**, because the inflow starts $17$ to $22\%$ above what the machine is sized for — above the window's $1.1408$ — and falls into the window within a dozen steps. From there it stays inside for the whole horizon.

**The end is thinner than the start.** Sizing for the minimum lowers the minimum — probe 2's is $2.50$ to $2.63\%$ below probe 1's, W228's Picard step again — and the window's lower margin is $1 - 0.9731 = 2.69\%$. **The machine is inside at macro-step $599$ by the difference between those two numbers, $0.06$ to $0.19\%$** — the car as drawn by the least — which is a coincidence of two measurements and not a margin anyone designed.

**What changes at step $0$ is recorded.** Every arm's spin-up marches the machine `powertrain` declares, and at every arm's release state that machine sits on its lower clamp, induction $0.02$, motoring (current $-0.75$ to $-1.00$). The probe switches in a machine sized for the duct: its induction at step $0$ is $0.110$ at the release sizing and $0.40$, the upper clamp, at W228's. The disk's thrust is $\tfrac12 C_T' A \langle U_d\rangle^2$ with $C_T' = 4a/(1-a)$, which goes from $0.082$ to $0.50$ and to $2.67$ — **six and thirty-three times the thrust, in one step, on a ring whose speed has moved by $0.1\%$.**

> **[AI Inference]:** so the start is a step the pipeline makes, not a property of the car: the duct is released from a flow settled against a disk that barely pushes, and brakes when one that pushes arrives. A spin-up that already carries the sized machine, or a settling march at the sizing before the horizon starts, would test it. Neither was run.

## 8.3 A drawing change moves the machine (W257)

Removing one plate or three from the rear, or lengthening the box, moves both device planes from $280$ and $392$ to $294$ and $406$. **On the unchanged car, that move alone takes $u_{\text{rotor}}$ at release from $0.5317$ to $0.4986$**, $-6.2\%$, while leaving the fall ($16.6\% \to 16.0\%$) and the fluid (first decline $56$, $544$ steps, identical) where they were. The turbine's plane at $406$ is the top of its declared range, $10.5$ cells from the duct's exit.

It is not a near-tie that a rounding flips. In the layout's own objective — the car's occupancy inside the cut bands — the drawn car scores $0.2673$ on its layout and $0.2833$ on the fixes', $6.0\%$ worse; without the rear box the order reverses, $0.2446$ against $0.2720$. The optimum genuinely moves, and it carries the machine with it, because the device planes are wherever the minimising layout puts its two narrow bands. The control that this is the dynamic program's own number: the drawn car on its own layout reproduces the recorded banded fraction to $10^{-16}$.

---

# 9. What each candidate buys, side by side

Against the reference — the car as drawn on the same layout. Loads are means over probe 1's last $150$ macro-steps, of a flow still drifting.

| candidate | fluid: first decline (declined steps) | machine at release sizing | machine at W228's sizing | downforce | drag |
|---|---|---|---|---|---|
| as drawn, moved layout | $56$ ($544$) | out from $18$; fall $16.0\%$ | out on $0$–$7$ | $0.836$ | $1.928$ |
| rear box removed | $394$ ($206$) | out from $22$; fall $17.9\%$ | out on $0$–$10$ | $1.165$ ($+39\%$) | $1.419$ ($-26\%$) |
| endplate removed | $294$ ($306$) | out from $19$; fall $18.0\%$ | out on $0$–$11$ | $0.978$ ($+17\%$) | $1.498$ ($-22\%$) |
| box $128$ cells longer | $265$ ($153$) | out from $17$; fall $14.7\%$ | out on $0$–$4$ | $0.860$ ($+3\%$) | $1.977$ ($+3\%$) |

**No candidate admits the arms under the enforced pipeline, and the reason is the same in every row**: `verify` would decline at macro-step $0$, on the machine's start. Setting the start aside — which would change the trajectory, and was not marched — the first decline in each probe at W228's sizing is the fluid's: $56$ as drawn, $286$ without the endplate, $388$ without the rear box, $265$ in the longer box. So on this record a drawing change could buy a horizon of a few hundred macro-steps at best, and the car as drawn none. **What stands between the car as drawn and all $600$ macro-steps is not in the drawing:** it is the outflow column (§7) and the machine's start (§8.2) — with the end margin of §8.2 still to be re-measured once the start is gone.

The choices the numbers leave, each unmeasured as a repair:

1. **the outflow boundary** — a code change, and the only candidate aimed at W245's mechanism; the solver already carries an open outflow condition RaceLab does not use, and this tier's column-free reading is its ready-made control;
2. **the machine's start** — a procedure change: spin up with the machine the arms use, or settle at the sizing before the horizon; the end margin of $0.06$–$0.19\%$ would then need re-measuring;
3. **the rear box** — a drawing change that improves the aerodynamics and delays the breach, and repairs neither fault;
4. **the device planes** — a declaration, so a drawing change cannot move the machine.

---

# 10. The timings, and why none of them is a claim

K1 predicted that the longer box costs $1.10$–$1.35\times$ a macro-step and that removing plates costs within $5\%$. Measured, probe 1: $0.664$ s as drawn, $0.852$ on the moved layout, $0.721$ without the box, $0.866$ in the longer box, $0.665$ without the endplate — so the longer box is $1.02\times$ the reference and $1.30\times$ the control, and the reference itself ran $28\%$ slower than the control on the same car. The laptop was on mains with one Python process. A counter logged every $30$ s from 19:54 — after the control had finished, so it cannot compare the two — shows the processor at $172$ to $237\%$ of its base frequency, ten-minute means $192$ to $207\%$; OneDrive was seen using a quarter of a core while the cache was being written. **The spread between arms that should cost the same is larger than the effect K1 predicted, so K1's failure says nothing about the box**, and no cost on this page is a claim.

---

# 11. What this tier did NOT do, named

- **No fix was adopted.** The drawing, the demo, the bundle and `out/racelab5` are unchanged; which fix, if any, is the user's decision.
- **None of the four candidates the numbers point at was marched as a repair**: no outflow treatment, no spin-up or settling march with the sized machine, no pinned device planes, no rear box removed *and* a longer box together.
- **No arms and no gate**: P2–P6 remain unmeasured on every car.
- **One longer box only**, $128$ cells; how the breach moves with length is one point.
- **The aerodynamic numbers describe a drifting flow** — the drawn car's downforce falls from $1.48$ to $0.86$ over its probe — so they rank the arms and do not describe a settled car.
- **Nothing was downloaded, no machine was rented, the unlicensed structural checkpoint was not loaded, and nothing was pushed.**

```
python scripts/tier58_car_fixes.py --out out/racelab6
python scripts/tier58_car_fixes.py --out out/racelab6 --stages layouts,summary
```

---

## See Also

- [[poc3-racelab-drawn-car]] — Tier 56, the faults this tier re-reads and the ablation it refutes.
- [[poc3-racelab-traced-car]] — Tier 54, the driver this tier imports.
- [[case-study-racelab-switch-atlas-0.1]] — CS-20, where the machine was first sized for its host (W222, W228).
- [[case-study-racelab-graph-atlas-0.1]] — CS-19, the layout's dynamic program and the arms.
- [[poc3-racelab-demo]] — the dashboard the drawn car runs on, unchanged.
- [[gap-worklist]] — W256 to W259.
