# CS-19 — RaceLab, phase 1: a car as a graph, and the two disciplines a car breaks

**Type:** Case study — **PoC 3's first phase**: the car's geometry, a window decomposition derived from it, the assembled graph and a headless march (folder: `Atlas 0.1/common/`)
**Status:** built 2026-09-12, Tier 51. `atlas/cases/racelab.py`, `scripts/tier51_racelab_graph.py`, `out/racelab/racelab.json`, `tests/test_tier51_racelab_graph.py`. Registered on the ladder as **CS-19**. Worklist rows **W216**–**W222** (opened).
**Related:** [[case-study-vehicle-march-atlas-0.1]] · [[vehicle-scale-and-sizing]] · [[poc2-frontwing-results]] · [[poc2-demo-and-novelty]] · [[joining-seam-cost]] · [[rung9-gate-restated]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[gap-worklist]] · [[case-study-wing-fsi-atlas-0.1]] · [[case-study-ground-effect-atlas-0.1]] · [[composition-error-theory]] · [[physics-simulation-datasets]]



# 0. The result, in one paragraph

**The car marches, and a shape breaks two of the framework's disciplines and one of its experts.** Thirteen immersed bodies on a rolling road in a $672\times240$ box — chosen from a timing measurement against the front-wing case run in the same process, not from the requirements' guess — decomposed into **fourteen $128\times128$ windows laid out by a dynamic program over the car's own occupancy**, assembled into a 26-agent graph across CS-18's five governing families, and marched for 600 macro-steps in five arms with a null per join. **The compile reproduces CS-18's verdicts on a graph that is not CS-18's**: the joined union refuses at `L7/R9` alone, the disjoint union and the clocks-reconciled control refuse nothing — and of the thirty-six seams the map paints red, **not one earns a refusal on its own subject**; twenty-one are locally clean and fifteen locally decertified, because `L7/R9` is a single graph-level decision seen thirty-six times. **Four of seven gate clauses pass, one splits as predicted, one fails and is diagnosed, and the largest finding had no clause at all.** J3's receiving balance closes to $\mathbf{0.010130}$ against a pre-registered $0.075$ with its null at exactly $1.0000$ — and the residual is *the geometry*: the ring the disk reads and the cell its force sits in stand at $1.0101289$, which predicts $0.0101289$, agreeing to five significant figures — with the sign inverted against CS-18's $0.955592$, so the balance overshoots where that one undershot (§7.1 offers a mechanism and labels it an inference, because no profile along the duct was measured). It is also **resolved**, at $8.8\times$ the ratio's own band, which is the control CS-18 could not write. J2's block closes its first law to $\mathbf{5.29\times10^{-11}}$ with the mount term and $\mathbf{1.0000000060}$ with it removed, a factor of $1.9\times10^{10}$. **J1's check fails its inherited $10^{-6}$ at $3.48\times10^{-6}$ — and the pointwise residual is exactly zero.** The clause averages a *concave* map over the settle window, so what it measures is the window's variance times the curvature: measured, the averaged residual is Jensen's term to $\mathbf{0.9995}$, and the threshold is reported failed rather than moved. **Two disciplines break on a car.** W124 — no seam through a body — holds on the front wing with eight cells to spare because there is *one* body in 208 cells; thirteen bodies covering 72 to 501 of 672 admit no clear layout at any stride the halo allows, so the objective becomes *cut where the car is thinnest* and still bands $\mathbf{26.9\%}$ of it. And **ten of the thirteen bodies are invisible to the compiler** — body forces with no agent, no port and no capability record — so that $26.9\%$ has nothing to fire on. **The largest result is an expert walking out of its own envelope unnoticed.** Re-siting the device by one duct dropped its inflow from CS-18's $0.9225$ to $0.6692$, past the $u = 0.8933$ at which this machine stops generating; the loop current crossed **sign**, $+0.0729 \to -0.5605$, the induction pinned at its clamp floor, `rotor_valid` went `False` on **all five arms** — and the march never read it, because `SizedCircuitSolve` returns that flag, `JoinState` records it and **nothing consults it**. That is CS-18's W211 realised rather than predicted, and PoC 2's W145 on a second subsystem. **A car is not a bigger wing.**

---


# 1. What phase 1 is, and what it inherits

`POC3-RACELAB-REQUIREMENTS.md` §9 splits RaceLab into five phases. This is the first and the whole of it: **the car's bodies as immersed body forces, a static window decomposition derived from those bodies, the assembled `CaseGraph`, and a headless march at the declared vehicle scale.** No server, no dashboard, no expert switch, no third dimension.

**What is reused rather than re-implemented, and where.**

| asset | reused as | what it gives |
|---|---|---|
| `ground_effect.GroundTiling` | `RaceTiling` **subclasses** it | the partition of unity, the halo, the cut, the assembly and the contamination mask, unchanged — the subclass overrides the LAYOUT and nothing else |
| `wing_fsi.FlexWing` | every body in the car is one | the porous inclined plate, its stations, its normal traction and its exactly conservative stamping kernel |
| `wing_fsi.FSIRollout` | `RaceRollout` subclasses it | cut, blend, project, band |
| `vehicle_march` | imported | the vehicle scale, the host-sized rotor, the machine sized to it, the devices' body force, the coolant sub-cycling, and `receiver_balances` unchanged |
| `integration_union` | imported | J1, J2 and J3 — the core, the mount and the rotor |

**Intuitively.** CS-18 marched eighteen agents that were not a shape: a wing, a cooling loop and a battery, bolted together by three joins, on a lattice whose only geometry was one plate. RaceLab gives that union a body. The question phase 1 asks is not whether the physics is new — none of it is — but **what happens to the framework's own disciplines when the thing being decomposed stops being one object in a box and becomes thirteen objects filling it.** The answer is that one of them survives, one of them cannot be met at all, and the difference is measurable.

---

# 2. The box, measured before anything was built on it

§10's first named risk is that the domain is too slow to be interactive, and the requirements' own answer to it is *"the exact figure is a measurement, not a guess."* Eight boxes, timed in **one process with the front-wing case as a control in that same process**, so what is reported is a ratio.

| box | windows | window | s / macro-step | against the control | metres |
|---|---|---|---|---|---|
| **control**: the front wing, $208\times144$ | 6 | $80^2$ | $0.0258$ | $\mathbf{1.00}$ | $1.62\times1.12$ |
| the requirements' own $384\times192$ | 18 | $80^2$ | $0.0752$ | $2.92$ | $3.00\times1.50$ |
| $384\times192$, wider windows | 12 | $128\times96$ | $0.0835$ | $3.24$ | $3.00\times1.50$ |
| $464\times240$ | 8 | $128^2$ | $0.0855$ | $3.32$ | $3.62\times1.88$ |
| $576\times240$ | 10 | $128^2$ | $0.1197$ | $4.65$ | $4.50\times1.88$ |
| **chosen**: $672\times240$, the derived layout | **14** | $128^2$ | $\mathbf{0.1558}$ | $\mathbf{6.05}$ | $5.25\times1.88$ |
| $800\times240$ | 14 | $128^2$ | $0.1497$ | $5.81$ | $6.25\times1.88$ |
| a full-length car, $1024\times240$ | 18 | $128^2$ | $0.2191$ | $8.51$ | $8.00\times1.88$ |

The control re-timed at the end of the run reads $0.9647$ of its first reading, so the drift across the table is under $4\%$ — and the $800\times240$ row being *faster* than the $672$ one at identical window work is inside that. **A single-draw timing here resolves nothing below about $5\%$, and the table is read accordingly.**

## 2.1 Why the window is 128 cells and not the front wing's 80

**The reason is phase 2 and it is a licence-and-error argument, not a speed one.** `wake_array`'s own line 24: *"The checkpoint is fixed at 128x128."* Poseidon-T is the learned expert the switch switches to, so a window that is not 128 cells has to be resized before the checkpoint can see it, and **a resize is an error that is not the expert's** — it would be charged to the learned column, and pricing that column honestly is the entire point of phase 2. Measured, the price of $128$ over $80$ at the same domain is $3.24/2.92 = 1.11$ at $384\times192$, and both are far inside the ceiling.

## 2.2 What the full-length car would have cost, measured rather than assumed

At the inherited $L_0 = 0.50$ m a real F1 car is $11.2$ length units, so a box that holds one with a wake is about $1024$ cells. That was timed: $0.2191$ s a macro-step, $8.51\times$ the control, **still under the ceiling**. It was rejected because it is $8.0$ m of domain to hold $5.6$ m of car, it puts the window count at $18$, and §3 records what the shorter car costs instead. **The choice is a trade recorded with both numbers, not a constraint.**

## 2.3 The thread count, and the measurement that was not reproducible

**Three measurements of one configuration gave three different answers, and this is reported rather than resolved.** One thread, four and eight were timed on the composition layer in three separate processes; the fastest was one, then four, then eight, and the eight-thread reading ranged over a factor of $3.5$. Two confounds were found afterwards and both are real: two idle `server.py` processes from an earlier demo were running throughout, and **the machine moved from battery to mains part-way through the session**.

What settles it is measuring the thing that is actually run rather than a proxy. On the **composition layer alone** eight threads is fastest ($0.1026$ s against one thread's $0.1414$); on the **full march with the car in it** eight threads is $0.684$ s a macro-step against one thread's $0.333$ — **one thread is twice as fast, and the proxy gave the opposite answer.** The car's body force is thirty-five small stamping kernels a call, and small tensors thread badly. Every recorded number in this tier is at **one thread**, and `machine_state()` writes the process list and the power state into the artifact beside every timing. That is **W217**.

---

# 3. The car

Thirteen bodies on a rolling road, in a $672\times240$ box at $\mathrm{d}x = 1/64$, which at the inherited scale ([[vehicle-scale-and-sizing]] §1) is $5.25\times1.88$ m.

| body | what it is |
|---|---|
| `FW_MAIN`, `FW_FLAP` | the front wing, main plane and flap, in ground effect |
| `NOSE` | the nose, sloping down to the wing |
| `FLOOR`, `DIFF` | the flat floor and the diffuser ramp |
| `POD_UP`, `POD_LO` | the sidepod's two surfaces |
| `DUCT_UP`, `DUCT_LO` | the radiator duct inside it, 32 cells tall |
| `RW_MAIN`, `RW_FLAP` | the rear wing |
| `WHEEL_F`, `WHEEL_R` | two wheels, each a closed ring of twelve plate segments |
| the moving ground | `ground_effect`'s band condition, **inherited unchanged** |

**Every one of them is `wing_fsi.FlexWing`** at a different $x_{\text{le}}$, $y$, chord and angle, with a rigid zero deflection. `PlateBody` is a name for a `FlexWing` and nothing more, so the discrete conservation identity W94 exists for holds here as it does there: over all thirteen bodies the worst relative residual of $\int f\,\mathrm dA$ against minus the force on the body is $\mathbf{2.1\times10^{-16}}$.

**The car is short, and by a measured factor.** It runs $x = 72$ to $501$ cells and $y = 2$ to $102$, which is $\mathbf{3.35}$ m long and $\mathbf{0.78}$ m tall against an F1 car's $5.6$ m and $0.95$ m — compressed $1.7\times$ in length and $1.2\times$ in height. The front wing's chord is at its own real size, $0.25$ m, inside a body that is not. **It is a recognisable car and not a dimensionally faithful one**, and §2.2 has the price of the alternative.

## 3.1 The wheel's coefficient, derived rather than inherited

Giving each of a wheel's twelve chords the plate's own $C_N = 20$ is not *"the same model applied to a circle"*: it is twelve independent screens in series. Measured at the release state it gave **$11.52$ of drag per wheel against $1.00$ for the whole rest of the car**, a peak force density $16\times$ the next largest body's, and the march blew up at macro-step 2 with $u_{\max} = 16.6$.

So the coefficient is chosen to make the body's **total drag a cylinder's**, which is the one number about a wheel this model can be asked to get right. For a segment whose outward normal is at $\theta$ the relative normal velocity is $U\cos\theta$ and the traction's streamwise component is $\tfrac12 c_n U^2 |\cos\theta|\cos^2\theta$, so

$$\text{drag} \;=\; \tfrac12 c_n U^2 \oint |\cos\theta|\cos^2\theta \, R\,\mathrm d\theta \;=\; \tfrac{4}{3} c_n U^2 R,$$

against a cylinder's $\tfrac12 C_D U^2 (2R) = C_D U^2 R$. Equating gives

$$\boxed{\;c_n \;=\; \tfrac{3}{4} C_D \;=\; 0.9 \ \text{ at } C_D = 1.2.\;}$$

**Measured, the twelve-chord ring returns $0.51856$ against the continuous $0.525$ — $1.23\%$.** The calibration removed a factor of $\mathbf{22.2}$, and with it the blow-up: $u_{\max}$ settles near $1.33$ at one sub-step.

**This is a calibration and it is named as one.** It makes the wheel's drag right and says nothing about its wake, its lift, or its rotation.

## 3.2 The rotation is not physics, and the measurement says so

The requirements' §3.1 asks for wheels *"with a rotating-surface boundary condition"*. **This model cannot express one, and that is measured rather than asserted.**

`WheelBody` **computes** the surface velocity $\mathbf u_s = \omega\,\hat z\times(\mathbf r - \mathbf r_c)$ at every station and projects it onto that station's outward normal, handing it to `FlexWing.station_normal` as the port's FLOW half. In the continuum the answer is exactly zero: $\mathbf u_s$ is tangential to the circle and the closure carries a **normal** traction only. On a polygon a chord's normal is radial only at its midpoint, so what survives is $\omega s$ at a station $s$ along the chord — and that is a discretisation artifact, which is exactly what its convergence says:

| stations per chord (12 chords) | $\max \lvert \mathbf u_s\cdot\hat n \rvert$ |
|---|---|
| 1 | $\mathbf{1.3\times10^{-15}}$ — machine zero, the station IS the midpoint |
| 2 | $0.1294$ |
| 3 (declared) | $0.1725$ |
| 4 | $0.1941$ |

| chords (3 stations each) | $\max \lvert \mathbf u_s\cdot\hat n \rvert$ | ratio |
|---|---|---|
| 12 | $0.17255$ | — |
| 24 | $0.08702$ | $1.983$ |
| 48 | $0.04360$ | $1.996$ |

**Machine zero at one station a chord, and halving exactly with the chord length.** That is a discretisation artifact's signature and not a physical effect's. At the declared geometry it is $\mathbf{17\%}$ of the free stream and it moved the field by $\mathbf{12.9\%}$ over eight macro-steps.

**So the declared car's wheels do not roll.** Declaring them to would inject spurious blowing and suction on the wheel's surface and a dashboard would be showing a discretisation and calling it a rotating wall. A real rotating-surface condition needs a **tangential** closure this project does not have, and adding one would be a new declared constant with no measurement behind it. That is **W219**.

## 3.3 The moving ground was already there

`ground_effect`'s band condition holds row 0 at $u = U_\infty$, $v = 0$ every exchange, which in the car's frame **is** a road moving at road speed. Nothing was written for it. What it costs is [[poc2-frontwing-results]]' own note, unchanged: no floor boundary layer and no viscous gap choke, so the ground-effect curve is monotone with no turnover.

---

# 4. The windows, derived from the car

The requirements' §4.1 asks for **12–20 fluid windows, laid out along the body rather than as a uniform grid**, computed once from the geometry and static thereafter. What that means in practice is decided by two constraints that pull against each other.

## 4.1 The obvious objective is unachievable, and finding that out is the first result

W124's discipline, which CS-10 set and `wing_fsi` inherited, is that **no seam should pass through a body**: a body cut by a seam has its force split between two experts that exchange only a ring. `WingTiling.cuts_clear_of_wing` checks it and the front wing satisfies it with eight cells to spare — because there is **one** body in 208 cells.

Here there are thirteen whose $x$-spans cover $72$ to $501$ of $672$ cells **almost without a gap**, and the largest stride the halo allows is $112$. The longest gap between consecutive non-cuttable bodies is shorter than a band. So **every layout that covers this domain cuts the car**, and the measured clearance is $-1$ by construction. There is no clear-of-the-bodies layout to find.

**So the objective is not "avoid the bodies" but "cut where the car is thinnest":**

$$\min_{\text{layouts}} \; \sum_{\text{bands } B} \; \sum_{x \in B} \Phi(x),$$

with $\Phi$ the car's occupancy per lattice column. That is exact and additive, so it is a shortest-path problem over the offsets and is **solved by dynamic programming rather than searched** — no layout is missed and none is sampled. 273 nodes on a two-cell grid, strides in $[48, 112]$.

## 4.2 The two constraints that are not free

**The devices have to sit ON a cut.** `integration_union.DeviceSite` declares a device on the **overlap** of a tiling $x$-seam and nowhere else: the inflow ring is the downstream window's `xlo` and the outflow ring the upstream window's `xhi`. So the radiator core and the recovery turbine each need a cut inside the duct — which is why the duct is 168 cells long rather than the 92 a radiator alone would need, and why the box is 672 cells rather than 608.

**And that cut has to be narrow.** The band's width **is** the distance between the two rings a device sees. CS-18 §7.1 diagnosed J3's whole $4\%$ receiving residual as this gap at $7.5$ cells; **the first layout this tier searched put it at forty**, and the balance would have been reporting the layout rather than the join. `device_overlap_max = 16` fixes it, and the box grew rather than the error.

**The front wing must be clear of every cut.** The `wet` and `mount` seams pair STRUCT and SUSP with **one** fluid window, so a plate inside an overlap would be a declaration that does not describe the march. What the constraint costs is measured: without it the optimum bands $19.2\%$ of the car, with it $\mathbf{26.9\%}$ — a factor of $\mathbf{1.40}$.

**The rows are not a choice at all.** $n_y = 240$ with $w_y = 128$ admits exactly one row offset above zero, $112$: lower leaves the top uncovered, higher is not a row. So the $y$-cut is $[112, 128)$ whatever the car looks like, and what the car has to do is stay below it.

## 4.3 The layout, and its controls

| | |
|---|---|
| columns | $0, 110, 164, 276, 388, 496, 544$ |
| rows | $0, 112$ |
| windows | $\mathbf{14}$, of $128\times128$ |
| overlaps | $18, 74, 16, 16, 20, 80$ in $x$; $16$ in $y$ |
| declared halo | $\mathbf{16}$ — the smallest, which is the one a bound is evaluated at |
| overlap ratio | $1.422$ |
| device planes | $x = 284$ (the core) and $x = 396$ (the turbine), both on 16-cell bands |
| ring to plane | $\mathbf{7.5}$ cells for both — **the front-wing case's, to the digit** |
| occupancy inside a cut | $\mathbf{26.9\%}$ |
| bodies a cut crosses | 16 of 35 segments; 12 of them not declared cuttable |
| $x$-cut clearance | $\mathbf{-1}$ — §4.1's finding |
| $y$-cut clearance | $\mathbf{+10.46}$ cells — **the discipline IS met in $y$** |

**Three controls.**

1. **The two profiles agree.** The objective was run twice, once on the purely geometric occupancy and once on the actual release-state body force. The release-state force is **blind to the floor** — at uniform inflow the normal traction is $\tfrac12 c_n |w| w$ with $w = -U\sin\alpha$, so a body at zero incidence carries exactly zero force and the car's floor is at zero incidence. The two objectives return **the same column count and differ at one offset by two cells, which is the search grid's own step.** So the layout does not turn on that blindness.
2. **The window-count frontier.** 14 windows band $26.9\%$ of the car; 16 band $36.5\%$. More windows is not free and the curve says what it costs.
3. **A uniform `RaceTiling` reproduces a `GroundTiling` bitwise** — same offsets, same names, same halo, same weights array for array. That is what makes "the partition of unity is the parent's" a check rather than a claim, and the partition sums to $1$ over every one of the $672\times240$ cells to $10^{-12}$.

---

# 5. The graph, compiled

26 agents, 36 seams, five governing families — CS-18's five, on a graph that is not CS-18's.

| family | agents |
|---|---|
| `incompressible-navier-stokes-2d` | 16 — 14 fluid windows, SUSP and ROTOR |
| `plane-stress-elasticity-2d` | 1 — STRUCT |
| `heat-conduction-2d` | 1 — BLOCK |
| `incompressible-thermal-transport-1d` | 4 — the coolant legs, RAD among them |
| `lumped-dc-circuit` | 4 — MGU, BUS, BATT, INV |

| graph | verdict | refusals |
|---|---|---|
| **the joined union, native clocks** | `refuse` | $\mathbf{L7/R9}$, and nothing else |
| **control**: the disjoint union | `admit-uncertified` | **none** |
| **control**: the joined union, clocks reconciled | `admit-uncertified` | none |

**This reproduces CS-18 §4.5's table on a different graph**, and that is the point of running it. The car's graph has 26 agents against 18, 14 fluid windows against 6, and a tiling derived from geometry rather than a $3\times2$ grid. Agreeing on the verdict says the refusal is a statement about **the clocks** and not about the front wing's layout. The disjoint union has no multirate **seam**, so W194's narrowing leaves it alone here exactly as it does there.

## 5.1 Thirty-six red tiles are one refusal, and no seam earns one

The per-seam map comes back $36$ of $36$ red. **That is one verdict seen thirty-six times, not thirty-six verdicts.** `L7/R9`'s subject is `<graph>`: the compiler emits a single decision and a per-seam map paints it across every seam so it can be seen. PoC 2 published the other reading once and **W177** corrected it, so the artifact here carries `local_verdict` beside `verdict` — what a seam earned on **its own subject, its two ports and its two agents** — and the two readings are recorded side by side:

| reading | `refuse` | `admit-uncertified` | `admit` |
|---|---|---|---|
| everything that reaches the seam | $\mathbf{36}$ | $0$ | $0$ |
| what the seam **earned** | $\mathbf{0}$ | $15$ | $21$ |

**Not one of the thirty-six seams earns a refusal on its own subject.** The graph is refused, once, for its clocks; every seam in it is locally clean or locally decertified. A panel that showed only the first row would be telling a viewer that thirty-six things are wrong with this car, and thirty-six things are not.

## 5.2 Ten of the car's thirteen bodies are invisible to the compiler

**The graph does not contain the car.** `FW_MAIN` is declared, through STRUCT and SUSP and the `wet` and `mount` seams. The other ten bodies — the nose, the floor, the diffuser, both pod surfaces, both duct walls, the rear wing's two elements and the two wheels — are **body forces with no agent, no port and no capability record.** The two devices have agents because `integration_union` gives them ports; the passive bodies have nothing.

So the compiler cannot see that $26.9\%$ of the car sits inside a cut. There is no rule for it to fire, because there is no declaration for a rule to read. **That is W220, and it is the largest gap this tier opens**: a decomposition derived from geometry is checked against a graph that does not carry the geometry.

---

# 6. The gate, pre-registered

Written after the instrument run of §6.1 and **before** any arm. Every threshold is **CS-18's own**, inherited rather than chosen here, so that none of them is this tier's answer read backwards; the one new clause is P6 and its number is the requirements' §10 ceiling, written before this tier existed.

| clause | measured as | passes if |
|---|---|---|
| **P1 · the compile** | the joined union's refusals, and the disjoint union's | joined refuses at `L7/R9` and nothing else; disjoint refuses nothing |
| **P2 · J3's receiving balance** | the work the turbine's body force does on the fluid over the settle window, against the shaft power the operating point claims | relative residual $\le 0.075$ with the term, $\ge 0.5$ with it withheld |
| **P3 · J1's parametric check** | $UA$ against the declared exponent's prediction from the measured air | tracking residual $\le 10^{-6}$; the null pins $UA$ exactly while the air moves |
| **P4 · J2's receiving balance** | the block's first law on the coolant clock, with the mount term and without | $\le 10^{-6}$ with, $\ge 0.01$ without |
| **P5 · the repeat floor** | the referent marched twice as two independent constructions | bitwise identical in every crossing quantity |
| **P6 · the macro-step cost** | one macro-step of the WHOLE march at the chosen box | the lagged column is under $0.5$ s |
| **P7 · the body force conserves** | $\int f\,\mathrm dA$ against minus the force on each body | worst relative residual under $10^{-12}$ |

The thresholds as the code holds them, so the prose and the evaluation cannot drift — `tests/test_tier51_racelab_graph.py` asserts every number below appears on this page, because Tier 48 measured that a criterion written in prose and evaluated in code **drifts permissive**:

```
P2  tol_with 0.075   tol_without 0.5
P3  tol_with 1e-06   tol_without 0.0
P4  tol_with 1e-06   tol_without 0.01
P6  ceiling_s 0.5
P7  tol 1e-12
```

**Its weakness, named.** The car is one geometry and the horizon is one horizon; *out of sample* here means a different graph from CS-18's, not a different rung. What makes that worth something is that the graph really is different.

## 6.1 The instrument, before the gate was fixed

A 40-macro-step tight march, its repeat, a lagged one, and the settling report.

| | |
|---|---|
| cost, tight | $1.687$ s per macro-step |
| cost, the repeat | $1.629$ s — $3.4\%$ apart, which is this host's timing floor |
| cost, lagged | $\mathbf{0.333}$ s |
| tight over lagged | $\mathbf{5.07}$, against a structural 5 solver calls an exchange against 1 |
| the repeat | **bitwise identical in every crossing quantity** |
| the join's inner fixed point | $5.54\times10^{-5} \to 2.58\times10^{-8}$ over three iterations |
| sub-steps | $1$ throughout; $u_{\max} = 1.328$ against a declared band of $4$ |
| J3's balance at 40 steps | $0.0221$ |

## 6.2 The settling curve, and the horizon it fixes

CS-18 marched $1200$ steps of a $3.25$-unit domain with **one** body in it — $4.6$ transits — and its settle band had a norm of $1.008\times10^{-3}$. This domain is $10.5$ units and holds thirteen bluff bodies whose wakes shed. A 1600-step lagged march, with the band read over four trailing windows:

| through macro-step | band norm | fluid-only norm |
|---|---|---|
| 160 | $1.663$ | $0.0873$ |
| 400 | $0.3404$ | $0.1873$ |
| 800 | $0.1987$ | $0.1139$ |
| 1600 | $\mathbf{0.0878}$ | $\mathbf{0.0583}$ |

**The car does not settle to CS-18's standard at any affordable horizon.** At 1600 lagged steps the fluid quantities still move by $\mathbf{5.8\%}$ over the last quarter, against CS-18's $0.1\%$.

**The two columns settle in opposite directions and that is the point of keeping both.** The full norm falls monotonically, $1.663 \to 0.340 \to 0.199 \to 0.088$, and it is dominated at $160$ steps by the machine's heat — the circuit is still in its opening transient. The fluid-only norm *rises* first, $0.087 \to 0.187$, because at $160$ steps **the flow has not reached the rear of the car yet** and a window over a quantity that has not started moving is quiet for the wrong reason. The curve is what says where the transient actually begins, and a single trailing window would have hidden it.

**So the horizon is $\mathbf{600}$ macro-steps** — $7.5$ tiling time units, $0.075$ s at the declared scale, and **$0.71$ transits of the domain against CS-18's $4.6$**. Four tight arms at 600 steps is about an hour; at CS-18's $4.6$ transits it would be nine. What the longer horizon would have bought is on the table above: the fluid band falls from $18.7\%$ at 400 steps to $5.8\%$ at 1600, **a factor of three and not an order**. Every residual in §7 is read against that band and not against zero — and a balance is a *ratio*, so what it is read against is **the ratio's own band**, measured per macro-step over the settle window, and not the band of either of the two co-varying quantities in it. That is **W221**.

---

# 7. Measured

Five arms at 600 macro-steps: the tight referent, its repeat, all-lagged, and one null per join. J2's receiver lives on the coolant clock and is measured there.

| arm | wall | s / macro-step |
|---|---|---|
| referent, tight | $1028.0$ s | $1.713$ |
| **repeat**, tight | $1028.3$ s | $1.714$ |
| all lagged | $201.8$ s | $0.336$ |
| null J3 — the turbine's force withheld | $1009.5$ s | $1.682$ |
| null J1 — $UA$ pinned at `UA_RAD` | $1020.4$ s | $1.701$ |

**The referent and the repeat are $\mathbf{0.03\%}$ apart in wall time and bitwise identical in answer.** The cost to quote is structural — five solver calls an exchange against one — and the measured $1.713/0.336 = \mathbf{5.10}$ agrees with it.

## 7.1 P2 — J3's receiving balance: **pass**, and the residual is the geometry

| | |
|---|---|
| the join's term: the turbine's body force's work on the fluid | $-0.006180$ |
| the shaft's claim, $T\langle U_d\rangle$ | $0.006118$ |
| **relative residual with the term** | $\mathbf{0.010130}$ against a pre-registered $0.075$ |
| **null arm** — the force withheld, the shaft still claiming | $\mathbf{1.0000}$ |
| the null arm's work on the fluid | exactly $-0.0$ |
| the null arm's shaft claim | $0.006363$, still there |
| $u$ on the ring the disk reads | $0.669153$ |
| $u$ in the cell the sink sits in | $0.675931$ |
| **ratio** | $\mathbf{1.0101289}$ |
| **what the velocity gap alone predicts** | $\mathbf{0.0101289}$ |

**The residual and the velocity gap agree to five significant figures.** This is CS-18 §7.1's diagnosis reproduced exactly: `disk.disk_average` reads its inflow deliberately upstream of the strip, while the force does its work inside it, and the whole residual is the difference between those two velocities.

**The sign is the other way round here, and that much is measured**: CS-18's ratio is $0.955592$ and this one is $1.0101289$, so the balance overshoots where that one undershot.

> **[AI Inference]:** the mechanism is *offered and not measured*. In open flow the disk's own force slows the air it is about to work on, which is why $u_{\text{plane}} < u_{\text{ring}}$ there; the plausible reading here is that the turbine sits inside a duct downstream of a radiator core whose drag is $0.1426$, so the streamwise profile across those $7.5$ cells is set by the duct and the core rather than by the device. **No velocity profile along the duct was measured and no control separates the duct's geometry from the core's drag** — the two would be separated by re-siting the same device into open flow on this lattice, which is one arm and was not run.

**And it is resolved, which CS-18's could not claim.** A balance is a ratio of two co-varying quantities, so what it must be read against is **the ratio's own band**, not the band of either component. Over the settle window the per-macro-step ratio moves by $\mathbf{0.00116}$ relative, while $u_{\text{rotor}}$ moves by $0.047$ and the loop current by $0.141$. The residual is $\mathbf{8.8\times}$ the ratio's own band. **Reading it against the components' band instead would have charged the join with unsteadiness that cancels inside it** — which is exactly the trap CS-18 §7.5 fell into from the other side.

## 7.2 P3 — J1's parametric check: **fail**, and the clause is measuring the settling

| | referent | **null J1** ($UA$ pinned) |
|---|---|---|
| air through the core, release $\to$ settled | $0.992288 \to 0.688000$ | identical, **bitwise** |
| $UA$, release $\to$ settled | $41.7407 \to 31.1400$ | $\mathbf{42.000}$ throughout |
| tracking residual against the declared exponent | $\mathbf{3.482\times10^{-6}}$ against $10^{-6}$ | — |
| the air moved over the march by | $30.7\%$ | $30.7\%$ |

**The null arm is the cleanest control on the page and it is clean.** $UA$ is exactly $42.000$ while the air moves by $30.7\%$, and the fluid is **bitwise identical** to the referent's over 600 macro-steps — one declaration changed and nothing else, exactly as CS-18 measured.

**But the clause fails, and the failure is not the join's.** `receiver_balances` averages $UA$ and $u$ over the settle window and *then* compares $\overline{UA}$ against $UA_0(\bar u/u_0)^{0.8}$. The map is **concave**, so averaging first and mapping second is not mapping first and averaging second. Measured on a 120-step march:

| | |
|---|---|
| the **pointwise** residual, $\max_t$ | $\mathbf{0.0}$ — exactly, at every macro-step |
| the **averaged** residual | $9.9257\times10^{-7}$ |
| the window's relative variance in $u$ | $1.2413\times10^{-5}$ |
| Jensen's term, $\tfrac12 p(p-1)\,\mathrm{Var}(u)/\bar u^2$ at $p = 0.8$ | $9.9306\times10^{-7}$ |
| **averaged over Jensen** | $\mathbf{0.9995}$ |

**The identity holds pointwise to machine zero, and the averaged residual IS the Jensen term to four significant figures.** So the clause as inherited measures *how settled the flow is*, quadratically — CS-18's window sat at $10^{-3}$ and got $5.57\times10^{-12}$; this one sits at $2\times10^{-2}$, four hundred times wider in variance, and gets $3.48\times10^{-6}$.

**The threshold is not moved.** P3 is reported as a **fail** with the mechanism attached, exactly as CS-18 reported G4 vacuous and G5 NEITHER rather than adjusting either. What the fail says is that a tolerance inherited from a settled flow is not a tolerance for an unsettled one, and the repair is to write the clause pointwise — which is **W221**.

## 7.3 P4 — J2's receiving balance, on the coolant clock: **pass**

$1200$ steps of $0.05$ s $= 60$ s, with the machine's heat and $UA$ held at what the fluid march settled on.

| | with the mount term | the block on its own declared source | **the mount balance, term removed** |
|---|---|---|---|
| block first law, relative | $\mathbf{5.289\times10^{-11}}$ | $1.004\times10^{-10}$ | $\mathbf{1.0000000060}$ |
| settled wall temperature | $608.53$ K | $451.04$ K | — |
| return temperature | $383.52$ K | $341.43$ K | — |

**A factor of $\mathbf{1.89\times10^{10}}$ between the balance with the join's term and without it**, against a pre-registered $10^{-6}$ and $0.01$.

**Three columns and not two, because the obvious null closes.** A block with no mount term carries `cooling_loop`'s own declared gas temperature on its dry face, so its first law still shuts — $1.004\times10^{-10}$, the middle column. What has to fail is the balance *written with the mount term in it* and the term withheld. Asking the wrong object was this tier's first attempt and it reported a pass as a fail.

## 7.4 P5, P6, P7

| clause | verdict | the number |
|---|---|---|
| **P5 · the repeat floor** | **pass** | bitwise in every crossing quantity and in the field, over 600 macro-steps in two independent constructions |
| **P6 · the macro-step cost** | **splits, as predicted** | lagged $\mathbf{0.3363}$ s — **passes** the $0.5$ s ceiling; tight $1.7133$ s — **fails** it by $3.4\times$ |
| **P7 · the body force conserves** | **pass** | worst relative residual $\mathbf{2.14\times10^{-16}}$ over thirteen bodies |

P6's split was written into the prediction before the arms ran, and it is the answer to §10's first risk rather than a caveat on it: **the interactive column is the lagged one and phase 3 has to say so.**

## 7.5 The composition error between the two cadences

| | |
|---|---|
| all-lagged against the tight referent, relative $\ell_2$ over the six crossing quantities | $\mathbf{1.246\times10^{-3}}$ |
| $u_{\text{core}}$ | $+8.03\times10^{-5}$ |
| $UA$ | $+6.42\times10^{-5}$ |
| $u_{\text{rotor}}$ | $+1.86\times10^{-4}$ |
| loop current | $-5.54\times10^{-4}$ |
| the machine's heat | $\mathbf{-1.096\times10^{-3}}$ |
| block wall temperature | $+2.28\times10^{-14}$ |

**The prediction recorded with the gate was right about this**: the composition error is dominated by the circuit and not by anything in the fluid. The machine's heat carries $88\%$ of the norm while every fluid quantity is an order smaller, because the machine sits just above the battery's open-circuit voltage and CS-18 §7.2 measured that elasticity at $63.3$.

**And it is not resolved at this horizon.** $1.246\times10^{-3}$ against a flow whose own fluid quantities move by $9.67\times10^{-2}$ over the settle window is two orders below the band that binds. The repeat floor is bitwise zero, so *ten times the floor* is vacuous here exactly as it was in CS-18 — and it is reported as vacuous rather than passed quietly.

## 7.6 The finding the gate had no clause for: the march ran outside the envelope

**Re-siting the device by one duct took the powertrain out of its operating envelope, and nothing in the framework noticed.**

The machine's loop current is $I = (k_e\omega - V_{oc})/R_{\text{total}}$ with $\omega = \lambda u / r$, so on this rotor $k_e\omega = 1.5\,u$ against $V_{oc} = 1.34$: **the machine generates only above $u = V_{oc}/1.5 = \mathbf{0.8933}$.**

| | CS-18, the rotor in open flow | **RaceLab, the turbine in the radiator duct** |
|---|---|---|
| inflow at the ring | $0.9225$ | $\mathbf{0.6692}$ |
| margin over the crossover | $+3.3\%$ | $\mathbf{-25\%}$ |
| induction | $0.1140$ | $\mathbf{0.0200}$ — the clamp's floor |
| loop current | $+0.0729$ | $\mathbf{-0.5605}$ — **motoring** |
| `rotor_valid` | `True` | $\mathbf{False}$ |
| the machine's heat | $67.5$ W | $\mathbf{3982}$ W |

**All five arms are outside it**, re-derived from each arm's own settled inflow; every one is motoring with the induction pinned at its clamp floor.

**This is W196 and W211 realised rather than predicted.** CS-18 §7.2 measured $\mathrm{d}\ln q/\mathrm{d}\ln u = 63.3$ and wrote: *"a one-percent error anywhere upstream of this rotor is a sixty-three-percent error in the heat a different subsystem receives, and no rule reads it."* Phase 1 moved the rotor by one duct — a $27\%$ change in inflow — and the operating point crossed **sign**.

**And the march said nothing, which is the part that is a defect rather than a result.** `SizedCircuitSolve.solve` returns `rotor_valid`; `RaceRollout.refresh` records it into `JoinState`; **nothing reads it.** This is PoC 2's **W145** on a different subsystem — *the wing could be walked through the floor and the screen would have shown a downforce for a design the model declines* — and every number in §7.1 to §7.5 is therefore stamped **outside the rotor's declared envelope**. They remain measurements of the graph as declared: the balances are statements about whether the work matches the claim, and they do, whether the machine is generating or motoring.

**The repair is priced and not taken.** The same similarity [[vehicle-scale-and-sizing]] §2.3 used to size the machine for a host-sized rotor, applied one level on: size it for the inflow its host actually delivers. Scaling $k_e = k_t$ by $\mathbf{1.3786}$ — from $0.05$ to $0.0689$ — puts the machine at the same $1.0326$ margin over its crossover that CS-18's sat at, and the control confirms it restores `rotor_valid` at the duct's own inflow. **It is not taken here because it is a second vehicle decision of exactly the kind that page's §0.1 says is the user's call and cheap to overrule**, and taking it silently would be making that call.

## 7.7 The clock, again, and worse

The block's own thermal time constant, measured rather than quoted:

| | |
|---|---|
| block wall, release $\to$ settled | $359.47 \to 741.08$ K |
| **thermal time constant** | $\mathbf{56.9}$ s |
| in coolant steps | $1138$ |
| **in fluid macro-steps** | $\mathbf{455{,}200}$ |
| what the march's 600 steps are, in coolant steps | $\mathbf{1.5}$ |

CS-18 measured $50.1$ s and $400{,}800$ fluid macro-steps on the front wing's graph. The car's block runs hotter, and that follows from measured quantities rather than being attributed: the machine's heat is $I^2R$ at the operating point, the loop current is $-0.5605$ against CS-18's $+0.0729$, and $3982$ W against $67.5$ W is exactly that ratio squared. **The heat is a consequence of §7.6's envelope excursion, so this row inherits its stamp.** **The conclusion is unchanged and is not a budget problem**: [[vehicle-scale-and-sizing]] §1.4 shows the clock ratio is $4U_0/L_0$ and never below $100$ over any plausible F1 choice, so no choice of vehicle removes it.

## 7.8 The verdict, clause by clause

| clause | verdict | the number |
|---|---|---|
| **P1 · the compile** | **pass** | joined refuses `L7/R9` alone; disjoint and clocks-reconciled refuse nothing |
| **P2 · J3's receiving balance** | **pass** | $0.010130$ against $0.075$; the null at exactly $1.0000$; and the residual IS the $1.0101289$ velocity gap |
| **P3 · J1's parametric check** | **fail**, and diagnosed | $3.482\times10^{-6}$ against $10^{-6}$ — **Jensen's term to four significant figures**, with the pointwise residual exactly $0$ |
| **P4 · J2's receiving balance** | **pass** | $5.289\times10^{-11}$ against $1.0000000060$, a factor of $1.89\times10^{10}$ |
| **P5 · the repeat floor** | **pass** | bitwise |
| **P6 · the macro-step cost** | **splits** | lagged $0.3363$ s passes, tight $1.7133$ s fails |
| **P7 · the body force conserves** | **pass** | $2.14\times10^{-16}$ |

**Nothing was re-tuned and no threshold was moved.** One clause failed and one split; both were predicted to behave as they did — P6 explicitly, P3 not at all — and the fail is reported with its mechanism rather than with a wider tolerance.

---

# 8. What phase 1 costs, and what that tells phase 3

§10's first risk asks whether the domain is too slow to be interactive. It is not, **on one of the two columns**, and the distinction is the answer rather than a caveat.

| column | s / macro-step | macro-steps / s | against the $0.5$ s ceiling |
|---|---|---|---|
| **lagged** | $\mathbf{0.333}$ | $3.0$ | **passes**, $1.5\times$ of headroom |
| **tight** | $1.687$ | $0.59$ | **fails**, by $3.4\times$ |

**A dashboard would march the lagged column**, and phase 3 has to say so on screen. The tight column is five solver calls an exchange against one — it exists so that the composition error between the two cadences is measurable, which is what CS-18's G3 is about, and it was never an interactive column on any domain.

**Where the lagged macro-step actually goes, and it is not where it was expected.** The composition layer alone at this layout — cut, fourteen window solves, blend, one global spectral Leray projection, band — is $0.1558$ s. The full lagged march is $0.333$ s. So **the car's thirteen bodies cost more than the fluid solve they are immersed in**: about $0.18$ s of a $0.33$ s macro-step, $53\%$.

The reason is arithmetic rather than deep. A lagged macro-step calls `FlexWing.forcing` $35 \times 4 = 140$ times, each building its own stamping box and placing it into the full field, and **24 of the 35 are wheel segments** — twelve chords a wheel, three stations a chord. A tight macro-step calls it $700$ times. Batching the wheel's chords into one kernel evaluation is the obvious repair and it is not made here. That is **W216**, and it is the cheapest row this tier opens: it is the difference between $3$ and perhaps $6$ frames a second for the dashboard phase 3 has to build.

**[AI Inference]:** the same arithmetic says the 3-D phase's cost is dominated by the same term, not by the solver, because a surface in three dimensions has $O(n^2)$ stations where a curve has $O(n)$. Unmeasured, and phase 4 is the place to measure it.

---

# 9. Where RaceLab stands

| phase | deliverable | status |
|---|---|---|
| **1 — the graph** | geometry, the decomposition, the `CaseGraph`, a headless march | **this page** |
| 2 — the switch | per-window classical / learned / certified, with per-window error and cost | not started |
| 3 — the dashboard | the server, the page, the overlay, the inspector, the sliders | not started |
| 4 — three dimensions | the half-car, a 3-D window solver, classical only | not started, optional to ship |
| 5 — the bundle | the branch, the launchers, the self-test, the licences | not started |

**What phase 1 hands phase 2**, and it is more than a graph:

- **a window that Poseidon-T can see without a resize** — $128\times128$, which is the checkpoint's own resolution and was chosen for that reason;
- **fourteen windows**, inside the requirements' $12$–$20$ target, each a `reference.WindowNS` the switch can replace;
- **a cost decomposition** that says the learned column has to beat $0.156$ s of composition layer, not $0.333$ s of march, because the car's body force is charged to every column equally;
- **a warning**: $26.9\%$ of the car sits inside a cut, so a window flipped to a learned expert is in many cases carrying part of a body whose other part is carried by its neighbour;
- **and a second warning**: the powertrain is outside its envelope at the duct's inflow (§7.6). Phase 2 measures per-window error against the all-classical referent, and **that referent is the march measured here** — so either the envelope is repaired first, at the price §7.6 gives, or every number phase 2 produces inherits the stamp.

**And what it does not hand it.** Phase 1 has **no learned expert of any kind**. Nothing on this page is evidence that a learned expert pays, and [[case-study-ladder-to-f1]] §14.3's sentence is unchanged: *the foundation-model half has no learned expert admitted with a nonzero contribution.* Tiers 48 and 50 measured that in the one slot where a theorem certifies the answer, the checkpoint's learned content is worth about $13$ classical calls against a $69$-call spread from perturbing its own weights by $3\%$. **RaceLab must not imply otherwise and phase 1 does not.**

---

# 10. What this tier did NOT do, named

- **No rung moved.** This is PoC 3's phase 1, not a rung of the ladder, and nothing here is a new coupling result: the three joins, their balances and their nulls are CS-18's, re-sited.
- **Every arm ran outside the rotor's declared envelope and the march did not read the flag that says so** (§7.6, **W222**). The check exists, is computed and is recorded; nothing consults it. **No envelope check was added in this tier** — the repair is stated, PoC 2's `Engine._absorb` is the pattern, and it is not built. The vehicle-side repair — scaling $k_e$ by $1.3786$ — is priced and **not taken**, because it is a second vehicle decision and [[vehicle-scale-and-sizing]] §0.1 makes that the user's.
- **Ten of the car's thirteen bodies are invisible to the compiler** (§5.2, **W220**). They are body forces with no agent, no port and no capability record, so no rule can read that a quarter of the car sits in a cut. **This is the largest schema gap the tier opens** and it is the one a decomposition-derived-from-geometry most wants closed.
- **P3 is reported as a failure and its threshold was not moved** (§7.2, **W221**). The clause as inherited averages a concave map over an unsettled window; the repair is to write it pointwise and that is not done here.
- **The structure is rigid.** `motion=False`, which is what CS-18 marched and what `integration_union` compiles, so the `wet` and `mount` seams are **declared and not solved**: the car's shape does not move, the front wing does not bend, the suspension does not ride. Every coupling under test is a join or a window seam.
- **The wheels do not rotate** (§3.2, **W219**), and the rotating-surface condition the requirements ask for is not expressible in a normal-only closure.
- **W124's discipline is not met in $x$** (§4.1, **W218**), and the layout minimises the breach rather than avoiding it. In $y$ it is met with $10.5$ cells to spare.
- **The horizon is $0.71$ transits of the domain** against CS-18's $4.6$, and the flow does not settle to CS-18's band at any horizon this host can afford (§6.2, **W221**). Every residual is reported against the band that binds.
- **The march is not differentiable through the devices or the bodies.** The stamping kernel is torch but the operating point is a bisection in numpy, exactly as in CS-18; rung 10 gains nothing.
- **The Reynolds number is the tiling's $250$**, not a vehicle's $1.61\times10^{6}$, and no amount of declaring metres changes that ([[vehicle-scale-and-sizing]] §1.5).
- **The car is short** — $3.35$ m against $5.6$ m — and §2.2 records that the full-length box was measured and rejected rather than found unaffordable.
- **`integration_union`, `vehicle_march`, `wing_fsi`, `front_wing`, `ground_effect` and every subsystem module are unchanged**, so every number on [[case-study-vehicle-march-atlas-0.1]] and [[joining-seam-cost]] is bitwise what it was.
- **Nothing was downloaded, no checkpoint was loaded, no machine was rented, NeuberNet was not loaded, and nothing was pushed.**

## See Also

- [[case-study-vehicle-march-atlas-0.1]] — the union this re-sites, its three joins and the gate whose thresholds §6 inherits.
- [[vehicle-scale-and-sizing]] — the $L_0 = 0.50$ m, $U_0 = 50$ m/s declaration and the $400{:}1$ clock ratio this march sub-cycles at.
- [[poc2-frontwing-results]] — the tiling, the plate, the rolling road and the per-seam panel this page's §5 is the car-sized version of.
- [[poc2-demo-and-novelty]] — what a PoC in this project is for, and the novelty argument phase 2 has to carry.
- [[case-study-wing-fsi-atlas-0.1]] — the composition layer, the halo and the tight/lagged distinction, inherited unchanged.
- [[joining-seam-cost]] — J1, J2 and J3 as declarations, and the ports the devices sit on.
- [[case-study-ladder-to-f1]] §25 — where this sits on the ladder, and what it does not move.
- [[gap-worklist]] Tier 51 — W216 to W221.
