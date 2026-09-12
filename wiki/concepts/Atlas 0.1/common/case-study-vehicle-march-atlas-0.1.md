# CS-18 — Rung 9, Marched: the union under load, the gate it passes, and the clock it cannot span

**Type:** Case study — **rung 9's first march**, with a gate pre-registered before the arms ran (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-12, Tier 49. `atlas/cases/vehicle_march.py`, `scripts/tier49_union_march.py`, `out/tier49/tier49.json`, `tests/test_tier49_union_march.py`, and the compiler change for W194 in `atlas/compiler.py` and `atlas/graph.py`. Registered on the ladder as **CS-18**. Worklist rows **W94**, **W194**, **W199**, **W201** (closed), **W196**, **W197**, **W198**, **W200** (annotated), **W209**–**W213** (opened).
**Related:** [[vehicle-scale-and-sizing]] · [[rung9-gate-restated]] · [[joining-seam-cost]] · [[f1-pathmap-and-end-goal]] · [[case-study-ladder-to-f1]] · [[gap-worklist]] · [[case-study-wing-fsi-atlas-0.1]] · [[case-study-cooling-loop-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] · [[case-study-wake-array-atlas-0.1]] · [[case-study-scaling-ladder-atlas-0.1]] · [[defect-correction-learned-operator]] · [[port-algebra-atlas-0.1]] · [[composition-error-theory]] · [[poc2-frontwing-results]]

---

# 0. The result, in one paragraph

**The union marches.** Eighteen agents, five families, three joins and three clocks: `front_wing`'s six-window tiling with the rotor and the radiator core in it as body forces, the powertrain's operating point solved across the union at every exchange, and the coolant circuit sub-cycled at the ratio [[vehicle-scale-and-sizing]]'s units declaration gives. All seven decisions [[rung9-gate-restated]] §6 listed are made, and a gate was written as numbers after a 40-macro-step instrument run and **before** any of the seven 1200-step arms. **Four of its five measurable clauses pass and the fifth returns neither of its two outcomes.** J3's receiving balance closes to $\mathbf{0.0399}$ against a pre-registered $0.075$ — and the residual is *diagnosed*, not merely small: the disk reads its inflow on a ring $7.5$ cells upstream of the plane where its own force does the work, the two velocities stand at $0.955592$, and that gap alone predicts $0.0444$. J2's block closes its first law to $\mathbf{4.07\times10^{-9}}$ with the mount term and fails at $\mathbf{0.998}$ without it. J1's conductance tracks its declared exponent to $5.6\times10^{-12}$ while the air through the core falls $0.998\to0.907$, and its null holds $UA$ at exactly $42.000$ **on a bitwise-identical fluid**. **G5 — whether the joins' errors add — returns NEITHER, and the clause could not have answered it**: it compares norms, and a norm cannot see a sign. The per-component data show J1's and J3's error vectors **anti-parallel at a cosine of $-0.9999$**, summing to the both-lagged arm's error with a relative residual of $\mathbf{0.0039}$. **The joins' composition errors superpose to four parts in a thousand and partially cancel** — the strongest form of *composition error does not compound*, which the pre-registered norm test was structurally unable to report. And both are **three orders of magnitude below the flow's own unsteadiness** ($\sim10^{-6}$ against a band of $1.01\times10^{-3}$), so the honest reading of G4 is not *resolved* but *unresolvable here*: the repeat floor is bitwise zero, which makes the clause as written vacuous. **The negative is the clock.** The union's three clocks span $\mathbf{400}$ in seconds where the bare numbers `L7/R9` compares span $16$, and the block's own thermal time constant is $\mathbf{50.1}$ s — $1002$ coolant steps, $\mathbf{400{,}800}$ fluid macro-steps, about **21 hours** of this laptop at the lagged price and four days at the tight one. **Rung 9's graph marches, its gate is measurable, and no single march can span the vehicle's own time scales.**

---

# 1. What had never happened, and what this is

Tier 46 built the joined union and compiled it. Tier 47 held rung 9's gate against it, found neither clause fitted, proposed a replacement — G1 integration work, G2 each join's receiving balance, G3 composition error at vehicle scale — and measured G2 **at the release state, one step each**. [[rung9-gate-restated]] §7's first line: *"Nothing marches the union."* §6 then listed seven decisions a march needs first, and said of the first two that they *"are not framework work at all."*

This tier makes the seven and marches.

**Intuitively.** Three subsystems were built by three people who never agreed how big anything is or how fast it goes. Bolting them together on paper is a compile; running them together is a march, and a march forces every question the compile could leave open — what a second is, how wide the rotor is, where the power actually goes, and what to do when one part of the machine settles four hundred times more slowly than another. This page answers those, runs the thing, and reports what the run says including where it says the question was badly posed.

---

# 2. The seven decisions, made

| # | decision | row | what was decided | what stays open |
|---|---|---|---|---|
| 1 | time and length units | **W201** | $L_0 = 0.50$ m, $U_0 = 50$ m/s — [[vehicle-scale-and-sizing]] §1 | no record carries a unit; `L7/R9` still compares bare numbers |
| 2 | a rotor sized for its host, a machine sized for it | **W199** | the disk at its host face $D = 0.5$; the machine by a **similarity**, $k_e = k_t \to Ak$ and $R \to R/A$ — §2 of the same page | `CircuitSolve` and `integration_union` are unchanged (**W209**) |
| 3 | dissipation and domain-boundary power | **W200** | **not declared**: each receiver's balance is written by hand, as CS-9 and CS-12 did | no dissipation field, no boundary ports; the core's unaccounted loss is $0.223919$ (§5.3) |
| 4 | the down faces' orientation | **W197** | **dissolved for a march**: a body force's power path is the force's work, not its ports (§4.3) | still open for a compile |
| 5 | the devices applied as body forces | **W94** | the donor's own exact sink, thrust conserved to $0$, with a declared one-cell thickness floor (§4.2) | the march is not differentiable through the devices |
| 6 | the shared operating point | **W196** | **solved across the union at every exchange**, iterated to a declared residual (§4.4) | its stiffness is $\mathbf{63.3}$ and no rule reads it (§7.2) |
| 7 | `L7/R9` scoped to the multirate seams | **W194** | **closed**: R9's premise is the multirate seam (§4.5) | — |

Decisions 1 and 2 have their own page, [[vehicle-scale-and-sizing]], because they are the choice of which vehicle this is rather than framework work, and because they should be cheap for the user to overrule. The rest are here.

---

# 3. What the march is

## 3.1 The three clocks, and how they are run

At the declared scale ([[vehicle-scale-and-sizing]] §1.3):

| subsystem | bare `dt_native` | in seconds | over the fastest |
|---|---|---|---|
| `front_wing` | $0.0125$ | $1.25\times10^{-4}$ | $1$ |
| `powertrain` | $0.2$ | $1.0\times10^{-3}$ | $8$ |
| `cooling_loop` | $0.05$ | $5.0\times10^{-2}$ | $\mathbf{400}$ |

**spread in the bare numbers: $16$. Spread in seconds: $\mathbf{400}$.**

The march therefore runs the fluid on its own clock, the circuit **algebraically** (`CircuitSolve` contains no time, so it resolves within any step), and the coolant circuit **sub-cycled 400:1**. The clocks nest exactly, which is what `L7/R9`'s quadrature branch asks of a multirate scheme and what makes the sub-cycling well posed rather than a rounding.

## 3.2 The horizon, fixed with the gate

$$1200\ \text{fluid macro-steps} \;=\; 15\ \text{tiling time units} \;=\; 4.6\ \text{transits of the }3.25\text{-unit domain} \;=\; \mathbf{0.15\ s} \;=\; \text{exactly }3\ \text{coolant steps}.$$

The settle window is the last quarter, $300$ macro-steps. The structure is rigid (`motion=False`), matching what `integration_union` compiles, so the only couplings under test are the three joins and the tiling's own window seams.

## 3.3 Tight and lagged, which is what G3 is about

There is no monolith spanning this union — [[rung9-gate-restated]] §2 checked, and the three referents that exist are each inside one subsystem. **The referent is therefore the same union, tightly coupled**, and composition error is the difference between coupling cadences:

- **tight** — a join's device computes its thrust from the ring at the **end** of the exchange, found by a fixed-point iteration of a declared count, and the operating point is re-solved with it. Five solver calls per exchange.
- **lagged** — the thrust is computed from the ring at the **start of the macro-step** and held for all four exchanges. One solver call per exchange.

That is [[case-study-wing-fsi-atlas-0.1]]'s own tight/lagged distinction carried from the FSI seam to the joins. The inner fixed point is defended rather than assumed: its residual falls from $2.62\times10^{-6}$ to $9.06\times10^{-10}$ across the declared three iterations.

## 3.4 The arms

| arm | J1 | J3 | null | wall |
|---|---|---|---|---|
| **referent** | tight | tight | — | $422.6$ s |
| **repeat** | tight | tight | — | $225.2$ s |
| all lagged | lagged | lagged | — | $40.7$ s |
| J1 lagged | lagged | tight | — | $206.7$ s |
| J3 lagged | tight | lagged | — | $199.0$ s |
| null J3 | tight | tight | the rotor's force withheld | $226.9$ s |
| null J1 | tight | tight | $UA$ pinned at `UA_RAD` | $215.6$ s |

**The referent and the repeat are the same computation and returned the same answer to the bit, $1.88\times$ apart in wall time.** The $422.6$ s is contaminated — a window of $200$ macro-steps in it took $209$ s where the neighbouring $200$ took $37$ — and the repeat is the clean reading. So the cost to quote is structural: **five solver calls per exchange against one**, and the measured $225.2 / 40.7 = 5.5$ agrees with it. This is [[atlas-proof-of-concept-1]] §10's standing warning arriving on a tier that had a repeat arm to catch it with.

**Each null withholds a different kind of thing, because each join's term is a different kind of thing.** J3's term is a power path, so its null withholds the rotor's force **while the shaft goes on claiming its power** — withholding the whole device would leave the balance reading $0/0$, and a null arm that cannot fail is not a control. J1 carries no power at all ([[rung9-gate-restated]] §4.4), so its term is the *dependence*: its null pins $UA$ and leaves the core's drag, which is why the null arm's fluid comes out **bitwise identical to the referent's** and the control varies exactly one thing.

---

# 4. The five decisions that live on this page

## 4.1 Decision 3 — the receiver balances, written by hand (W200)

No dissipation field was added and no domain boundary became a port. $\mathcal R(t)$ still cannot be assembled from the records: thirteen of eighteen agents dissipate by their own constants, ten window faces carry the fluid domain's inflow and outflow through nothing declared, and there is no field for either. **Each receiver's balance is written by hand instead** — CS-9's rule, which CS-12 closed rung 5's gate on and the front-wing demo uses. §5 is those balances. **The hole is named and not filled.**

## 4.2 Decision 5 — the devices as body forces (W94)

The plate in `front_wing`'s own march is already applied as a body force through `FlexWing.forcing`; the two devices are added to that same $f_x$ before it is cut. So a device is applied exactly the way the plate already was, and the composition layer — cut, blend, one global spectral Leray projection per exchange, band — is CS-12's, inherited and not re-implemented.

The sink itself is `disk.ActuatorDisk.body_force_field`, the donor's own exact-overlap weighting, which makes

$$\sum_{\text{cells}} f_x\,\mathrm dA \;=\; -T$$

true to floating point. Asserted on **every call**, and measured exactly $0$ for both devices at three thrusts: a discretization losing $3\%$ of the thrust would read as a $3\%$ interface residual in §5 and be blamed on the join.

**One departure, declared.** The donor's smearing thickness is $\Delta_d = \langle U_d\rangle\,\Delta t$ — the distance a parcel travels while the impulse is applied, which is what makes the impulse agree with what momentum theory allows. On `wake_array`'s lattice that is $1.6$ cells. On this host the cell is half as wide and the clock sixteen times finer, so the same rule gives $\mathbf{0.182}$ of a cell. The rule is kept and **a floor of one cell is declared on top of it**. The floor changes no total — the weighting conserves thrust either way — and what it changes is how sharply the sink is presented to the projection.

## 4.3 Decision 4 — the rotor's power path, and why W197 does not block a march

Tier 47 measured the rotor's two declared faces carrying **exactly zero** net power at the rotor's own base, against a shaft power of $0.785$: *"the power the shaft delivers enters through no declared port."*

**A march does not go through the ports.** The device is a body force, so the power path is the force's work on the fluid, $\int f_x u\,\mathrm dA$, and §5.1 reads it there. That **dissolves W197 for a march and leaves it open for a compile** — the orientation question is about what the ports report, and a compile is what reads the ports. The row stays open, with its subject narrowed.

## 4.4 Decision 6 — the operating point, solved across the union (W196)

At every exchange in a tight arm, the rotor's inflow is read from the host ring, `SizedCircuitSolve` re-solves the union's operating point at it, and the thrust that comes back is what the fluid is forced with on the next iteration. That is *solved across the union* rather than *declared per subsystem*, which is the second of W196's two options.

[[joining-seam-cost]] §5.4 measured that re-derivation reaching six ports on agents off J3's own seams and recorded that **no rule reads any of it**. §7.2 gives it a magnitude.

## 4.5 Decision 7 — `L7/R9` scoped to the multirate seam (W194), closed

`FluxMatching`'s own docstring states R9's requirement as every agent **at a multirate seam**; the code read every agent in the graph. `CaseGraph.multirate_seams` and `agents_at_multirate_seams` now exist and the rule reads them.

| graph | multirate by agents | multirate seams | agents at one | verdict | refusals |
|---|---|---|---|---|---|
| joined union, native clocks | yes | $5$ | $8$ of $18$ | `refuse` | `L7/R9` |
| **disjoint union, native clocks** | yes | $\mathbf{0}$ | $0$ | **`admit-uncertified`** | **none** |
| joined union, time-integrated | yes | $5$ | $8$ of $18$ | `refuse` | `L7/R9/quadrature`, naming **eight** |
| *control*: `thermal_seam`, pointwise | yes | $1$ | $2$ | `refuse` | `L7/R9` |
| *control*: `thermal_seam`, its own default | yes | $1$ | $2$ | `admit-uncertified` | none — Tier 17's reason, not this tier's |
| *control*: joined union, clocks reconciled | no | $0$ | $0$ | `admit-uncertified` | none |

**The disjoint union stops being refused for a mismatch no seam carries**, and every graph that has a multirate seam still refuses. **E4 still fails** on the disjoint union: the agents really do run at different steps, and what the narrower premise costs is the refusal, not the hypothesis.

**This closure changes two cells of a published table.** [[joining-seam-cost]] §6.4 records both unions refusing, and the quadrature refusal naming all eighteen agents; `out/w172/w172.json` still carries those, and `tests/test_tier46_joining_seams.py` still asserts them **as the record of what Tier 46 measured**, beside the live compile that now admits. The diagnosis is kept rather than deleted, and that page carries a dated note.

**And decision 7 turned out not to be on this march's critical path**, which is worth saying: the union that marches has five genuine multirate seams, so `L7/R9` fires on it correctly. W194's false refusal was about the *disjoint* union, which is not what marches.

---

# 5. The gate, pre-registered

Written after the instrument run of §6 — a $40$-macro-step march, its bitwise repeat, and the settling report — and **before** any $1200$-step arm, any null arm and the second vehicle scale. That is [[defect-correction-learned-operator]] §7's discipline.

**Its weakness, named.** This graph has one geometry. "Out of sample" here means a longer horizon and a different clock ratio, not a different rung, so this is a **weaker** form of pre-registration than Tier 48's $N = 2 \to 6 \to 12$. Every threshold below is nevertheless justified by a number that existed before this tier, so that none of them is the answer read backwards.

| clause | measured as | passes if |
|---|---|---|
| **G1 · J3's receiving balance, over a march** | the work the rotor's body force does on the fluid over the settle window, against the shaft power the union's operating point claims | relative residual $\le 0.075$ with the term, $\ge 0.5$ both without it **and** when the shaft is charged against the *other* device's work |
| **G2 · J2's receiving balance, over a march** | the block's first law over the coolant circuit's own march, with the mount term and without it | $\le 10^{-6}$ with, $\ge 0.01$ without |
| **G3 · J1's parametric check, over a march** | $UA$ against the declared exponent's prediction from the measured air; the null pins $UA$ | tracking residual $\le 10^{-6}$ with, and $UA$ exactly constant without while the air moves |
| **G4 · composition error, per join** | the crossing vector's relative $\ell_2$ error of each single-join-lagged arm against the all-tight referent | each exceeds $10\times$ the repeat floor, or is reported **not resolved** rather than as a number |
| **G5 · do the joins' errors add** | the all-lagged arm's error against the sum of the single-join-lagged arms' errors | **ADD** if the gap $\le 0.25$ of the sum; **COMPOUND** — Claim B failing at vehicle scale — if the whole exceeds $2$ times the sum; otherwise **NEITHER**, reported as neither |
| **G6 · the repeat floor** | the referent marched twice as two independent constructions in one process | bitwise identical in every crossing quantity, and the floor quoted beside every difference G4 reports |

The thresholds as the code holds them, so the prose and the evaluation cannot drift — `tests/test_tier49_union_march.py` asserts every number below appears on this page, because Tier 48 measured that a criterion written in prose and evaluated in code **drifts permissive**:

```
G1  tol_with 0.075     tol_without 0.5
G2  tol_with 1e-06     tol_without 0.01
G3  tol_with 1e-06     tol_without 0.0
G4  floor_multiple 10.0
G5  add_tol 0.25       compound_factor 2.0
```

**Where each number comes from.** G1's $0.075$ is the inflow non-uniformity [[rung9-gate-restated]] §4.2 measured at the release state and declared *a property of the model, not a leak*. G2's pair brackets Tier 47's release-state $3.0\times10^{-8}$ and $0.269$ with two orders of margin each. G3's is an algebraic identity's tolerance — $UA(u)$ **is** $UA_{\text{rad}}(u/u_{\text{ref}})^{0.8}$ and nothing else. G4's is this vault's standing rule that a difference below the instrument's floor is not a signal. G5's two constants exist so that **NEITHER is a reportable outcome** rather than a default to the convenient one.

## 5.0 The prediction, recorded with the gate

> G1 passes, at a residual **larger** than the $0.030$ the 40-step instrument run showed, because the rotor's own wake deepens over a longer horizon and the whole residual is the gap between the ring the disk reads and the cell its force sits in. G2 and G3 pass. G4 resolves all three joins. G5 returns **ADD** rather than **COMPOUND**, because two of the three joins are one-way and the third is algebraic, so there is no loop for an error to go round — and COMPOUND would have been the more valuable result, since it is Claim B failing at vehicle scale, which [[f1-pathmap-and-end-goal]] §7 calls the most likely failure on the ladder.

**Three of those five predictions were right and two were not.** G1 passed at $0.0399 > 0.030$, for exactly the stated reason. G2 and G3 passed. **G4 did not resolve the joins** — against the floor that binds rather than the one pre-registered. And **G5 returned neither ADD nor COMPOUND**, for a reason the prediction did not anticipate and the clause could not express.

---

# 6. The instrument, before the gate was fixed

A $40$-macro-step tight march and its repeat.

| | |
|---|---|
| cost, tight | $0.232$ s per macro-step |
| the repeat | **bitwise identical in every crossing quantity** |
| J3's balance at $40$ steps | $0.0303$ |
| the join's inner fixed point | $2.62\times10^{-6} \to 9.06\times10^{-10}$ over three iterations |
| the flow's own unsteadiness over the settle window | load $0.33\%$, $u_{\text{rotor}}$ $0.15\%$, $u_{\text{core}}$ $0.69\%$, induction $5.9\%$ peak-to-peak |

That last row is the one that decides how to read G4, and it was measured before the gate was written.

---

# 7. Measured

## 7.1 G1 — J3's receiving balance over a march: **pass**, and diagnosed

| | |
|---|---|
| the join's term: the rotor's body force's work on the fluid, settled | $0.0664$ |
| the shaft's claim, $T\langle U_d\rangle$ | $0.0685$ |
| **relative residual with the term** | $\mathbf{0.0399}$ against a pre-registered $0.075$ |
| **null arm** — the force withheld, the shaft still claiming | $\mathbf{1.000}$ |
| **mis-attribution control** — the shaft charged against the *core's* work | $\mathbf{1.032}$ |
| $u$ on the ring the disk reads | $0.922538$ |
| $u$ in the cell the sink sits in | $0.881570$ |
| **ratio** | $\mathbf{0.955592}$ |
| what the velocity gap alone predicts | $\mathbf{0.0444}$ |

**The residual is not a leak, it is the donor's own convention meeting a body force.** `disk.disk_average`'s rule is *deliberately upstream of the strip: sampling inside it would read a velocity the disk's own body force has already slowed* — the classic actuator-disk double-counting error. So the shaft's claim is $T$ times the velocity $7.5$ cells **upstream**, while the force does its work inside the strip, where that same force has slowed the flow by $4.4\%$. The measured $3.99\%$ against a predicted $4.44\%$ is the whole of it.

**[AI Inference]:** this is a property of the local-induction closure rather than of the composition, and it would appear identically in `wake_array`'s own march — which has never had its body force's work compared with its disks' claimed power. Not checked there. That is **W210**.

## 7.2 J2's clock, and how stiff the shared operating point is

The block's first law lives on the coolant clock, so §7.3 measures it there. What the fluid-clock march contributes is the crossing datum, and the null arm gives its sensitivity:

| | referent | **null J3** (the rotor's force withheld) |
|---|---|---|
| $u$ at the rotor's ring | $0.92254$ | $0.98457$ |
| induction | $0.11401$ | $0.26087$ |
| loop current | $0.073011$ | $0.228092$ |
| **the machine's heat** | $\mathbf{67.47}$ W | $\mathbf{658.45}$ W |

A $6.7\%$ change in one join's inflow moves another subsystem's heat by $\mathbf{9.8\times}$. Exactly, because the machine sits just above the battery's open-circuit voltage:

$$I(u) = \frac{k_e\lambda u/r - V_{oc}}{R_{\text{total}}},\qquad \frac{\mathrm d\ln I}{\mathrm d\ln u} = \frac{k_e\lambda u/r}{k_e\lambda u/r - V_{oc}} = \frac{1.3837}{0.0438} = \mathbf{31.66},$$

and the heat is $I^2R$, so its elasticity is $\mathbf{63.32}$. **The similarity of [[vehicle-scale-and-sizing]] §2.3 preserves this elasticity exactly** — the back-EMF is its invariant — so re-sizing restored the operating point's *existence* and left its *conditioning* precisely where it was. That is **W196 with a magnitude, and W211**: a one-percent error anywhere upstream of this rotor is a sixty-three-percent error in the heat a different subsystem receives, and no rule reads it.

## 7.3 G2 — J2's receiving balance over a march: **pass**

On the coolant circuit's own clock, $1200$ steps of $0.05$ s $= 60$ s:

| | with the mount term | the block with its own declared source | **the mount balance, mount term removed** |
|---|---|---|---|
| block first law, relative | $\mathbf{4.07\times10^{-9}}$ | $8.88\times10^{-11}$ | $\mathbf{0.998}$ |
| settled wall temperature | $324.85$ K | $446.59$ K | — |
| heat into the coolant | $219.66$ W | $1349.72$ W | — |
| return temperature | $306.54$ K | $334.12$ K | — |

**A factor of $2.45\times10^{8}$ between the balance with the join's term and without it**, against a pre-registered $10^{-6}$ and $0.01$. Tier 47 measured the same pair at one state as $3.0\times10^{-8}$ and $0.269$; over a march it is tighter on both sides.

**A physical consequence of decision 2, recorded because it is large.** Sizing the rotor for its host halves the machine's heat, $134.31 \to 67.47$ W, and the block settles $324.85$ K instead of the $446.59$ K its own declared source drives it to. The joined vehicle's block runs $122$ K cooler than the unjoined one, and that is the join doing its job rather than a defect.

## 7.4 G3 — J1's parametric check over a march: **pass**, on a bitwise-identical fluid

| | referent | **null J1** ($UA$ pinned) |
|---|---|---|
| air through the core, release $\to$ settled | $0.997965 \to 0.906618$ | identical, **bitwise** |
| the core's drag work | $-0.20530$ | identical, **bitwise** |
| $UA$, release $\to$ settled | $41.7809 \to 38.6923$ | $\mathbf{42.000}$ throughout |
| tracking residual against the declared exponent | $\mathbf{5.57\times10^{-12}}$ | — |
| the air moved over the march by | $0.0914$ | $0.0914$ |

**This is the cleanest control on the page.** The null differs from the referent in exactly one declaration and in nothing else — the fluid is identical to the bit — and $UA$ is exactly constant while the air moves by $9.1\%$.

**And J1's power still goes nowhere.** The air gives up $0.223919$ of mechanical work across the core, in the tiling's units per unit span, and no record can name where it goes: the flow carries no temperature, which is why this bond is `MECH` at all, and no field can carry the core's dissipation. **That number is W200's hole, measured.**

## 7.5 G4 — composition error at vehicle scale: **pass as written, and the clause is vacuous**

| arm | composition error against the tight referent |
|---|---|
| J1 lagged | $1.677\times10^{-6}$ |
| J3 lagged | $7.301\times10^{-7}$ |
| both lagged | $9.503\times10^{-7}$ |
| J2 lagged, on the coolant clock's own horizon | $1.970\times10^{-3}$ |
| J2 lagged, over the union march's three coolant steps | $4.52\times10^{-11}$ |

**The repeat floor is exactly zero**, so "ten times the floor" is zero and every nonzero difference clears it. The clause as pre-registered is vacuous, and it is reported as vacuous rather than passed quietly.

**The floor that binds** is the flow's own unsteadiness over the settle window — a difference smaller than the band the physics moves through on its own is not a signal however reproducible it is:

| quantity | relative peak-to-peak |
|---|---|
| $u_{\text{core}}$ | $3.03\times10^{-5}$ |
| $UA$ | $2.42\times10^{-5}$ |
| $u_{\text{rotor}}$ | $1.43\times10^{-5}$ |
| loop current | $4.50\times10^{-4}$ |
| the machine's heat | $9.01\times10^{-4}$ |
| **norm** | $\mathbf{1.008\times10^{-3}}$ |

**J1's, J3's and the pair's composition errors are all three orders of magnitude below it.** The honest verdict is **not resolved at this horizon**, not *resolved*. Only J2's, measured on its own clock over $60$ s, clears it.

## 7.6 G5 — do the joins' errors add: **NEITHER**, and the clause could not have said

As pre-registered, over the two joins that share a clock:

$$e_{\text{both}} = 9.503\times10^{-7},\qquad e_{J1} + e_{J3} = 2.407\times10^{-6},\qquad \frac{\lvert e_{\text{both}} - (e_{J1}+e_{J3})\rvert}{e_{J1}+e_{J3}} = 0.605.$$

Not within $0.25$, so not ADD. Not above $2$ times the sum, so not COMPOUND. **NEITHER**, reported as neither and not moved.

**The three-join form has no subject at all.** J1's and J3's errors are measured over $1200$ fluid macro-steps ($0.15$ s) and J2's over $1200$ coolant steps ($60$ s) — a horizon $400$ times longer. Adding them is not an addition of comparable things, and the clause is restated rather than silently substituted, which is [[case-study-learned-pair-atlas-0.1]] §9's and W198's precedent on this ladder.

### 7.6.1 What the clause could not see

**A norm cannot see a sign.** Per component, in the relative crossing space:

| component | $e_{J1}$ | $e_{J3}$ | $e_{J1} + e_{J3}$ | $e_{\text{both lagged}}$ |
|---|---|---|---|---|
| $u_{\text{core}}$ | $+3.533\times10^{-8}$ | $-2.412\times10^{-8}$ | $1.121\times10^{-8}$ | $1.114\times10^{-8}$ |
| $UA$ | $+2.826\times10^{-8}$ | $-1.929\times10^{-8}$ | $8.972\times10^{-9}$ | $8.916\times10^{-9}$ |
| $u_{\text{rotor}}$ | $+2.373\times10^{-8}$ | $-1.033\times10^{-8}$ | $1.340\times10^{-8}$ | $1.345\times10^{-8}$ |
| loop current | $+7.495\times10^{-7}$ | $-3.262\times10^{-7}$ | $4.233\times10^{-7}$ | $4.249\times10^{-7}$ |
| the machine's heat | $+1.499\times10^{-6}$ | $-6.524\times10^{-7}$ | $8.465\times10^{-7}$ | $8.498\times10^{-7}$ |
| block wall temperature | $-3.334\times10^{-9}$ | $-4.482\times10^{-11}$ | $-3.378\times10^{-9}$ | $-3.378\times10^{-9}$ |

$$\cos\angle(e_{J1},\,e_{J3}) = \mathbf{-0.9999},\qquad \frac{\lVert (e_{J1}+e_{J3}) - e_{\text{both}}\rVert}{\lVert e_{\text{both}}\rVert} = \mathbf{0.0039}.$$

**The joins' composition errors superpose to four parts in a thousand, and they point opposite ways.** That is a *stronger* statement than ADD: the errors do not merely fail to compound, they are linear in the set of joins lagged, and because J1's and J3's contributions are anti-parallel the two partially **cancel** — the pair's error is smaller than either join's alone. The pre-registered clause, comparing norms, reported that as a $0.605$ gap and could not have reported anything else: the triangle inequality is slack exactly when the errors are anti-parallel, which is the case the clause most wanted to find.

**[AI Inference]:** the mechanism is not diagnosed. Lagging a device makes it read a ring its own force has not yet slowed, so it over-applies its thrust; the two devices sit in different rows of one domain and the global spectral Leray projection is elliptic, so the sign at the other device's site need not follow. Whether the anti-parallelism is a property of these two sites or of any two devices in one projected tiling is **unmeasured**, and it is **W212**.

## 7.7 G6 — the repeat floor: **pass**

Bitwise identical in every crossing quantity over $1200$ macro-steps, in two independent constructions. Cross-process reproducibility was not tested.

## 7.8 The verdict, clause by clause

| clause | verdict | the number |
|---|---|---|
| **G1 · J3's receiving balance** | **pass** | $0.0399$ against $0.075$; both controls at $1.000$ and $1.032$ |
| **G2 · J2's receiving balance** | **pass** | $4.07\times10^{-9}$ against $0.998$ |
| **G3 · J1's parametric check** | **pass** | $5.57\times10^{-12}$; the null exact on a bitwise-identical fluid |
| **G4 · composition error** | **pass as written, vacuous as written** | the floor is $0$; against the floor that binds, **not resolved** |
| **G5 · do the errors add** | **NEITHER** | and the vector form says **superposition to $0.0039$** |
| **G6 · the repeat floor** | **pass** | bitwise |

**Nothing was re-tuned and no clause was moved.** Two were found to be badly posed after the fact — G4's floor and G5's norm — and both are reported as posed, with the reading that answers the question beside them.

---

# 8. The clock the union cannot span

The block's own thermal time constant, measured rather than quoted: march the coolant circuit until the wall temperature settles and read the time to $1 - 1/e$ of the total change.

| | |
|---|---|
| block wall, release $\to$ settled | $359.47 \to 309.76$ K |
| **thermal time constant** | $\mathbf{50.1}$ s |
| in coolant steps | $1002$ |
| **in fluid macro-steps** | $\mathbf{400{,}800}$ |
| what that costs here, lagged | $\approx 21$ hours |
| what that costs here, tight | $\approx 4$ days |

**This is the tier's negative, and it is structural rather than a budget problem.** The union marches, and a march that resolves the fluid reaches $0.15$ s while the subsystem it is coupled to needs $50$ s. [[vehicle-scale-and-sizing]] §1.4 shows the ratio is $4U_0/L_0$ and never below $100$ over any plausible F1 choice, so **no choice of vehicle removes this**; a lighter block or a faster loop would, and both are design changes rather than framework ones.

**Out of sample on the clock.** The second vehicle scale — $L_0 = 1.0$ m, $U_0 = 30$ m/s, ratio $120$ — is a genuine second cell, and it is free, because the coupling on the coolant side is one-way and the fluid trajectory does not depend on the coolant clock at all (asserted **bitwise** in the tests). Over the same fluid horizon it takes $10$ coolant steps instead of $3$, and the block's wall reaches $357.60$ K against $358.77$ K. **The clock ratio is a real parameter of the answer**, not a bookkeeping choice.

**[AI Inference]:** the standard repair is what every multirate code does — declare the fast subsystem quasi-steady on the slow one's clock, which is licensed here because the fluid settles in far less than one coolant step. This tier uses that closure for the slow march and **does not** build it as a scheme: nothing in `atlas/` declares a quasi-steady subsystem, and `L7/R9` has no verdict for one. That is **W213**.

---

# 9. Where rung 9 stands

| clause | status |
|---|---|
| **G1** — $O(1)$ integration work per join | **met**, Tier 46, qualified by W196 — and W196 now has a magnitude ($63.3$, §7.2) |
| **G2** — each join's receiving balance, **over a march**, with its null | **met on all three joins**: J3 $0.0399$, J2 $4.07\times10^{-9}$ against $0.998$, J1 exact on a bitwise-identical fluid |
| **G3** — composition error at vehicle scale, and whether the joins' errors add | **built and measured.** The errors **superpose to $0.0039$** and partially cancel; they are three orders below the flow's own unsteadiness, so the per-join numbers are **not resolved** at this horizon; and the three-join additivity question **has no subject**, because the joins do not share a clock |

**Rung 9 is marched and it is not complete.** G2 is met. G3 is built and its answer is the strongest available form of *composition error does not compound* — and it is an answer at one horizon, on one graph, below the noise floor, with the additivity question well posed over only two of the three joins. Calling that Claim B established at vehicle scale would be an overclaim of exactly the kind [[f1-pathmap-and-end-goal]] §7 exists to prevent. **What can be said is that nothing compounded, that the composition error is linear in the joins lagged, and that the vehicle's clocks are the thing that actually blocks a full-scale march.**

---

# 10. What this tier did NOT do, named

- **No rung is marked complete.** G3 is measured at one horizon on one graph and below the flow's own noise; the additivity clause has a subject for two joins of three.
- **The union that marches and the union that compiles are not the same object.** `vehicle_march` re-sizes the rotor and the machine; `integration_union` does not, so every number on [[joining-seam-cost]] is bitwise what it was and the compiled graph still carries the rotor Tier 47 found too wide. **W209**, and it is a real gap rather than bookkeeping: the gate is measured on a graph the compiler has not seen.
- **`CircuitSolve` is unchanged** and still builds its disk at the default width, so any other caller has Tier 47's bug.
- **No dissipation field, no boundary ports, no unit on any record.** W200 and W201 are answered as decisions and open as schema holes; `L7/R9` still compares bare numbers.
- **W197 is narrowed, not closed.** The march reads the power path off the body force; what the two declared ports report is untouched.
- **The march is not differentiable through the devices.** The donor's body force is numpy and the operating point is a bisection; both are converted rather than taped. `front_wing`'s own gradient path is unused here, so rung 10 gains nothing from this tier.
- **The Reynolds number is not the vehicle's** — $250$ against $1.6\times10^{6}$ — and no amount of declaring metres changes that.
- **Nothing quasi-steady is declared as a scheme** (W213), and the second vehicle scale is a replay rather than a second fluid march.
- **W210 is opened and unmeasured**: `wake_array`'s own march has never had its body force's work compared with its disks' claimed power.
- **W212 is opened and unmeasured**: whether the two joins' anti-parallel errors are a property of these sites or of any two devices in one projected tiling.
- **Task 2 (W205, the corruption sweep) and Task 3 (W204's coarse competitor across a coupled seam) were not started.** This tier is Task 1, and the brief said Task 1 alone is a good session.
- **Nothing was downloaded, no checkpoint was loaded, no machine was rented, NeuberNet was not loaded, and nothing was pushed.**

## See Also

- [[vehicle-scale-and-sizing]] — decisions 1 and 2, with their controls and what choosing otherwise costs.
- [[rung9-gate-restated]] — the gate this page measures, and the seven decisions it listed.
- [[joining-seam-cost]] — the union that is marched here, and the dated note W194's closure put on its §6.4.
- [[case-study-wing-fsi-atlas-0.1]] — the tight/lagged distinction §3.3 carries from the FSI seam to the joins.
- [[case-study-scaling-ladder-atlas-0.1]] — where composition error was measured with $N$ as a dial, which this graph cannot do.
- [[defect-correction-learned-operator]] §7 — the pre-registration discipline §5 follows, and the drift trap the fenced block above exists to close.
- [[composition-error-theory]] — what a composition error is, and why a referent of the same union is the only one available here.
- [[gap-worklist]] Tier 49 and [[case-study-ladder-to-f1]] §23 — the rows this tier closed, annotated and opened.
