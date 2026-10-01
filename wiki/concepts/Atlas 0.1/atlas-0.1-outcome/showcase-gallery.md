# The showcase gallery

**Type:** Outcome page — **gallery of measured showcase runs** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** written 2026-09-29 from one pass through the served workbench on this laptop; two drawn rows added that day, and eight more on 2026-09-29 and 30, compiled and run headless (§2). Every example was opened from *File > New from example*, checked, compiled with the Atlas compiler, and run decomposed and on the full domain. Every number below is read from that pass's records in `out/workbench/records/step-f/` (§6). **Every row is a showcase, not a research record** ([[showcase-library-plan]] §2): one pass, checks registered before the family's first run, and a record file. There are no predictions, audits or follow-up tiers. **§8, the Fast examples**, was added 2026-10-01: one per kind, swept, then confirmed from a fresh start.
**Hub:** [[00-atlas-0.1-outcome]] · **Plan:** [[showcase-library-plan]] · **Parent claim:** [[outcome-c4-modular-multiphysics]] · **Code:** `atlas/workbench/` (`python -m atlas.workbench --open`, then the header's *Fast example*, whose arrow lists the chosen kind's other examples (*More examples* until 2026-10-01); from 2026-09-29 to 30 the whole list was *Case > Examples...*, and before that *File > Showcase gallery*) · **Updated 2026-09-30 by the demo chat:** the forked river's row (§2); the demo's plans are in [[demo-finish-plan]]

---

## 1. How to read a row

Each case is a small classical problem cut into pieces: overlapping windows, or materials meeting at an interface. The pieces trade data at **seams**. The same problem solved whole, on the same grid, is the **full domain**, and it is the reference. A row answers four questions.

1. **Will the compiler accept the way the pieces are joined?** The *compile* column holds its verdict: `admit`, `admit-uncertified` or `refuse`. The case becomes a `CaseGraph` through its family (one agent per window or piece, one connection per seam, the cross-points declared), and `atlas/compiler.py` judges it. A seam's verdict is the worst of the compiler's decisions about that seam.
2. **Does the joined run keep what physics says must be kept?** The *balance* check: mass, energy, charge or force, closed to a tolerance.
3. **Does it reproduce the undivided solve, or a textbook answer?** The *reference* check.
4. **How long does it take, and which is faster?** The run's wall time, and the speed ratio

$$
s \;=\; \frac{t_{\text{full}}}{t_{\text{arm}}},
$$

where $t$ is the mean time of the family's step alone. Diagnostics, the bitwise comparison and the snapshots are timed out. $s>1$ means the decomposed arm is faster than the full domain.

The reference checks are relative. For a field $u$ they read

$$
\frac{\max_i \lvert u^{\text{arm}}_i - u^{\text{full}}_i \rvert}{\text{a stated scale}},
$$

where the scale is the temperature span, the peak concentration, the largest displacement, or the heating's rise. The results' details also show the rms difference of the displayed field, $\big(\tfrac{1}{n}\sum_i (u^{\text{arm}}_i-u^{\text{full}}_i)^2\big)^{1/2}$, in the field's own units. Every tolerance was registered in its family's module before that family's first run, and none was changed afterwards.

---

## 2. The gallery

Measured 2026-09-29, 05:14–05:41 EDT, in the served page on AC power (§5). Every check that could be measured passed.

| case | style | port type | compile (time) | balance check (tolerance) | reference check (tolerance) | run: wall time (steps) | label |
|---|---|---|---|---|---|---|---|
| `wake-array-3`: 3 rotors, 6 windows | A | `MECH` | admit-uncertified (7.3 s; 9 agents, 13 seams, 2 cross-points) | incompressibility $1.65\times10^{-16}$ ($10^{-9}$) | farm power $-4.07\%$ against the full domain ($25\%$) | 29.1 s (40 macro-steps) | showcase, not research record |
| `farm-12`: 12 rotors, 24 windows | A | `MECH` | admit-uncertified (45.3 s; 36 agents, 62 seams, 15 cross-points) | $2.22\times10^{-16}$ ($10^{-9}$) | $-8.00\%$ ($25\%$) | 69.9 s (12 macro-steps) | showcase, not research record |
| `farm-21`: 21 rotors, 48 windows (W346's farm) | A | `MECH` | admit-uncertified (95.1 s; 69 agents, 124 seams, 35 cross-points) | $2.57\times10^{-16}$ ($10^{-9}$) | $-4.03\%$ ($25\%$) | 70.0 s (6 macro-steps) | showcase, not research record |
| `wall-2`: a steel layer and a copper layer, steady | C | `THERM` | **admit-uncertified** (W348; refused at R10 before it, 0.06 s); its one seam admitted | energy $1.67\times10^{-11}$ ($10^{-6}$) | closed form $q=\Delta T/\sum_i L_i/k_i$: $8.64\times10^{-13}$ ($10^{-6}$); full domain $2.69\times10^{-13}$ of the 100 K span ($10^{-6}$) | 0.7 s (5 timed repeats) | showcase, not research record |
| `plate-insert`: a copper insert in a steel plate, transient | B | `THERM` | **admit-uncertified** (W348; refused at R10 before it, 0.09 s); its one seam admitted | energy $1.64\times10^{-8}$ ($10^{-6}$) | full domain $3.58\times10^{-9}$ of the 100 K span ($10^{-6}$) | 4.5 s (40 steps) | showcase, not research record |
| `plate-circuit`: a graphite film on a battery and a resistor | D | `ELEC` | admit-uncertified (0.08 s; 2 seams, one per electrode) | Kirchhoff $1.43\times10^{-12}$, energy $3.16\times10^{-13}$ ($10^{-6}$ each) | the film and circuit as one sparse solve: $1.07\times10^{-13}$ of the current ($10^{-6}$) | 0.6 s (5 timed repeats) | showcase, not research record |
| `plume-2`: a pollutant plume, a shallow reach into a deep one | A, one explicit step per exchange | `ADVEC` | admit-uncertified (0.14 s) | mass $8.12\times10^{-14}$ ($10^{-9}$) | full domain $8.76\times10^{-17}$ of the peak ($10^{-10}$) | 7.9 s (6,000 steps) | showcase, not research record |
| `sound-air-water`: sound from air into water | C, explicit | `MECH` | admit-uncertified (0.02 s) | energy $8.43\times10^{-16}$ ($10^{-10}$) | reflection $R$ within $3.10\times10^{-10}$ of $(Z_2-Z_1)/(Z_2+Z_1)$ ($10^{-6}$) | 2.2 s (2,000 steps) | showcase, not research record |
| `bracket-2`: steel and aluminium under a 5 kN load | B, on a structure | `MECH` | **admit-uncertified** (W348; refused at R10 before it, 0.33 s); its one seam admitted | forces $1.28\times10^{-10}$ ($10^{-6}$) | full domain $3.66\times10^{-11}$ of the largest displacement, 0.770 mm ($10^{-6}$) | 31.5 s (5 timed repeats) | showcase, not research record |
| `heated-strip`: a steel-and-copper strip, heated | split by physics | none: the bond is volumetric | **refused before the compiler** by the port vocabulary (0.02 s) | heat in = heat stored, $9.95\times10^{-11}$ ($10^{-9}$) | the unsplit solver, equal bit for bit over 60 steps (exact) | 1.5 s (60 steps) | showcase, not research record |
| `cooled-block`: a chip on a copper block under a water channel | C, at a seam between two physics | `THERM` | admit-uncertified (0.08 s) | energy $6.95\times10^{-11}$ ($10^{-6}$), 2 kW per metre of depth made and carried out | full domain $2.56\times10^{-10}$ of the 29.5 K rise ($10^{-6}$) | 1.8 s (5 timed repeats) | showcase, not research record |
| `bend-3`: a quarter ring of steel drawn with arcs, on three curved windows | B, drawn | `THERM` | **admit-uncertified** (W348; refused at R10 before it, 0.17 s); its two seams admitted | energy $1.24\times10^{-9}$ ($10^{-6}$) | full domain $3.07\times10^{-10}$ of the 100 K span ($10^{-6}$) | 1.98 s (5 timed repeats) | showcase, not research record |
| `insert-round`: a round copper insert in a steel plate, cut along the circle | C, drawn | `THERM` | **admit-uncertified** (W348; refused at R10 before it, 0.14 s); its one seam admitted | energy $9.86\times10^{-12}$ ($10^{-6}$) | full domain $1.01\times10^{-10}$ of the 100 K span ($10^{-6}$) | 1.85 s (5 timed repeats) | showcase, not research record |

**The last two rows are drawn geometry** (case file 0.4, added 2026-09-29 on the owner's request: irregular domains and windows that are not rectangles; see [[showcase-library-plan]] §6). They were not taken in the page pass above. They were compiled and run at 11:59 EDT through the workbench's own `Workbench` object, the code the page drives, headless and **on battery**, because the served page could not be drawn for screenshots at the time. So their wall times are not speed figures. Their checks are the conduction family's, registered on 2026-09-28 before its first run, and unchanged. The bend's staircase moves its heat flow by $-0.97\%$ from the ring's continuum value $k\,\Delta T\ln(r_o/r_i)/\theta$: information, not a check.

**Eight more rows, added 2026-09-29 and 30**: the S-channel, on windows generated from its shape (case file 0.5), and one drawn example for each family that had none ([[showcase-library-plan]] §6). Each was compiled and run with the workbench's own `CompileJob` and `CaseRun`, the code the page drives, headless, on AC power, with no other Python process running. Seven ran at 21:33–21:37 EDT on 2026-09-29. `farm-hill` was run again at 01:48 EDT on 2026-09-30, after the fix below. The machine was running slower then: the full domain's unchanged step took 0.85 s against 0.30 s four hours before, while the owner's screensaver was running. So that row's wall time is not a speed figure. The records are in `out/workbench/records/step-j/` (§6).

| case | style | port type | compile (time) | balance check (tolerance) | reference check (tolerance) | run: wall time (steps) | label |
|---|---|---|---|---|---|---|---|
| `s-channel`: a steel channel drawn with splines, on four windows generated from its shape | B, windows that follow the shape | `THERM` | **admit-uncertified** (W348; refused at R10 before it, 0.10 s); its three seams admitted | energy $1.13\times10^{-8}$ ($10^{-6}$) | full domain $1.80\times10^{-9}$ of the 100 K span ($10^{-6}$) | 1.2 s (5 timed repeats) | showcase, not research record |
| `ring-film`: a quarter ring of graphite film on a battery and a resistor | D, drawn | `ELEC` | admit-uncertified (0.05 s; 2 seams, one per electrode) | Kirchhoff $5.36\times10^{-13}$, energy $9.59\times10^{-14}$ ($10^{-6}$ each) | the film and circuit as one sparse solve: $5.96\times10^{-14}$ of the largest branch current, 5.78 A ($10^{-6}$) | 0.2 s (5 timed repeats) | showcase, not research record |
| `river-bend`: a plume round a winding river, its flow solved along the banks | A, drawn, one explicit step per exchange | `ADVEC` | admit-uncertified (0.24 s) | mass $3.25\times10^{-14}$ ($10^{-9}$) | full domain $1.84\times10^{-16}$ of the 1.285 mg/L peak ($10^{-10}$) | 3.6 s (2,000 steps) | showcase, not research record |
| `sound-lens`: a pulse in a round-ended air duct meeting a curved water surface | C, drawn, explicit | `MECH` | admit-uncertified (0.02 s) | energy $4.22\times10^{-16}$ ($10^{-10}$) | *not measured*: a curved surface has no textbook reflection; the two pieces equal the full domain bit for bit over 1,500 steps | 3.4 s (1,500 steps) | showcase, not research record |
| `plate-hole`: a steel plate with a round hole, clamped and pulled | B, drawn, on a structure | `MECH` | **admit-uncertified** (W348; refused at R10 before it, 0.82 s); its one seam admitted | forces $7.68\times10^{-12}$ ($10^{-6}$) | full domain $3.18\times10^{-12}$ of the largest displacement, 0.0225 mm ($10^{-6}$) | 24.6 s (5 timed repeats) | showcase, not research record |
| `bimetal-arc`: a quarter ring of steel inside copper, one end held hot | split by physics, drawn | none: the bond is volumetric | **refused before the compiler** by the port vocabulary (0.00 s) | heat in = heat stored, $2.16\times10^{-10}$ ($10^{-9}$) | the unsplit solver, equal bit for bit over 60 steps (exact) | 2.8 s (60 steps) | showcase, not research record |
| `cooled-winding`: a copper block cooled by a winding water channel | C, drawn, at a seam between two physics | `THERM` | admit-uncertified (0.11 s) | energy $4.71\times10^{-11}$ ($10^{-6}$), 2 kW per metre of depth made and carried out | full domain $1.53\times10^{-9}$ of the 18 K rise ($10^{-6}$) | 1.9 s (5 timed repeats) | showcase, not research record |
| `farm-hill`: the three-rotor array over terrain drawn with splines | A, drawn | none compiled | **refused before the compiler**: a drawn farm has no declared graph (0.02 s) | incompressibility $2.22\times10^{-16}$ ($10^{-9}$) | farm power $-19.7\%$ against the full domain ($25\%$) | 101.3 s (40 macro-steps), on the slower machine: not a speed figure | showcase, not research record |

**`farm-hill` passes, and does not agree the way the plain farms do.** Its farm power is $19.7\%$ from the full domain, inside the $25\%$ registered for the farms, where the plain farms measured $2.3$–$8.3\%$. Its velocity differs from the full domain's by $0.20\,U$ rms, against $0.047\,U$ for the plain three-rotor farm on the same windows, and its windows take 26 sub-steps a macro-step against the full domain's 32. Two controls hold bit for bit, and they place the departure in the hill, not in the drawn machinery:
- one window over the drawn farm is its full domain, in the arrangement where one window is the full domain's arithmetic;
- a drawn farm that is the whole rectangle is the plain farm, both arms, in both arrangements.

**[AI Inference]:** the terrain turns the flow, and style A exchanges once per macro-step, so seams that a turned flow crosses may carry a larger lag than seams along a flat wake. Where the lag concentrates was not measured; the per-seam difference, windows that follow the terrain, or an iterated coupling would be the tests of it, and none was run.

**Found while recording these rows, and fixed before them** (2026-09-30):
- **A drawn farm's windows did not hold their ground.** Each window marched the box of its fluid cells alone, so a window over the terrain had its box's edge where the solid began, and one window over the whole drawn farm missed its full domain by $0.61\,U$ after one macro-step. A window's box is now the window as drawn, with the solid in it held at rest. `farm-hill`'s power moved by $0.1$ point, $19.6\%$ to $19.7\%$: the departure above was the hill's all along.
- **The drawn river did not compile**: the compile's velocity scale divided by the river's shallowest depth, which is zero past a drawn river's banks.
- **A drawn farm compiled as the plain farm.** The wind farm's graph is the scaling-ladder rung's, a full rectangle of fluid, and the terrain was nowhere in it. A drawn farm is now refused before the compiler with that reason, as an irregular tiling already was.

**One more row, added 2026-09-30 by the demo chat**: the forked river, the owner's request of that day ([[demo-finish-plan]] §2). It was compiled and run in the served page on AC power, with no other Python process at its start. **Its times are not quoted.** Two other chats shared the laptop that afternoon, and the owner was not asked to idle them for this run.

| case | style | port type | compile (time) | balance check (tolerance) | reference check (tolerance) | run: wall time (steps) | label |
|---|---|---|---|---|---|---|---|
| `river-fork`: a river that forks into two branches, each with its own outlet, on six windows along its own flow | A, drawn, one explicit step per exchange | `ADVEC` | admit-uncertified (6 agents, 5 seams, one from the fork to each branch) | mass $2.72\times10^{-14}$ ($10^{-9}$); the flow's continuity $2.25\times10^{-14}$ of the discharge ($10^{-12}$, registered before its first run) | full domain $2.10\times10^{-16}$ of the peak ($10^{-10}$) | not quoted (2,000 steps) | showcase, not research record |

The wider branch takes 53.6% of the water and, at the run's end, 57.3% of the pollutant leaving. A potential flow splits by the branches' shapes alone, which is a stated guess.

**The third check, bit for bit, where a case has one:**
- Threaded equals serial in the three farms (over 40, 12 and 6 macro-steps), the insert (40 steps), the plume (6,000) and the bracket (5 repeats).
- The sound's two pieces equal the full domain over 2,000 steps.
- The strip's lagged split equals the synchronous one a step late over 60 steps. Lagging costs $8.3\times10^{-3}$ of the stress.
- Two checks are *not measured*, and the page says why for each: the wall's threaded check (style C has no threaded arm), and the insert's closed form (an insert is not a wall of layers, so it has none).

**Coverage:** all four coupling styles and the split by physics. Four of the five port types are compiled here: `MECH`, `THERM`, `ELEC` and `ADVEC`. `ROT` does not appear. The farms' graphs bond each rotor to its windows through `MECH` seams, and the plan's lumped cases that carry `ROT` (the CS-13 and CS-14 adapters) were optional and are not built (§7).

---

## 3. Speed: the full domain wins except at farm scale

$s = t_{\text{full}}/t_{\text{arm}}$, measured in the same runs.

| case | cells | decomposed, serial | decomposed, threaded (threads) |
|---|---|---|---|
| `wake-array-3` | 84,480 | 0.716 | 1.08 (4) |
| `farm-12` | 319,232 | 0.937 | **4.83** (4) |
| `farm-21` | 627,456 | 0.899 | **4.55** (8) |
| `wall-2` | 6,400 | 0.317 | — (style C is sequential) |
| `plate-insert` | 15,360 | 0.0279 | 0.0383 (2) |
| `plate-circuit` | 12,800 | 0.249 | — |
| `plume-2` | 28,800 | 0.450 | 0.307 (2) |
| `sound-air-water` | 29,600 | 0.702 | — |
| `bracket-2` | 6,400 | 0.00205 | 0.00233 (2) |
| `heated-strip` | 4,000 | synchronous split 1.15 | lagged split 1.08 (2) |
| `cooled-block` | 12,000 | 0.00475 | — |
| `bend-3` (on battery) | 6,601 of a 108 x 108 grid | 0.0179 | 0.0200 (2) |
| `insert-round` (on battery) | 16,000 | 0.0276 | — |
| `s-channel` | 5,208 of a 200 x 110 grid | 0.00488 | 0.00485 (2) |
| `ring-film` | 6,601 of a 108 x 108 grid | 0.222 | — (style D is one piece) |
| `river-bend` | 9,774 of a 240 x 100 grid | 0.745 | 0.313 (2) |
| `sound-lens` | 19,424 of a 300 x 120 grid | 0.419 | — (style C is sequential) |
| `plate-hole` | 12,352 of a 160 x 80 grid | 0.00586 | 0.00760 (2) |
| `bimetal-arc` | 5,031 of a 108 x 108 grid | synchronous split 1.02 | lagged split 1.41 (2) |
| `cooled-winding` | 11,968 of a 200 x 60 grid | 0.00344 | — |
| `farm-hill` | 77,290 of a 352 x 240 grid | not quoted (the slower machine, §2) | not quoted |

- **Threads pay only at farm scale.** At 12 and 21 rotors the threaded arm is $4.83\times$ and $4.55\times$ faster than the full domain. Both fall inside W346's measured ranges ($4.0$–$5.7\times$ and $4.3$–$4.7\times$, [[decomposition-speed-by-rotor-count]] §5). The serial arm stays below 1, as W346 found. W346 timed the developed wake at step 40, and these runs time the start-up. So this is a live reproduction of the ratio, not a new measurement of it.
- **Below about 30,000 cells, one direct solve or one explicit step on the whole grid beats every spatially decomposed arm.** The margin runs from $1.3\times$ (the drawn river's serial windows) to about $490\times$ (the bracket's serial Schwarz, 616 iterations to $10^{-12}$). The eight rows added on 2026-09-29 and 30 keep the rule: every spatial arm is below 1. The heated strip is split by physics, not space, and its three arms tie within the noise of a 3 ms step. The drawn arc's lagged split ran $1.41\times$ faster than its unsplit solver, in one run of 60 steps (means of 5.0 ms against 7.1 ms); that was not investigated.
- **[AI Inference]:** at these sizes the per-window bookkeeping and the iteration count cost more than the smaller solves save. A coarse space for Schwarz, or larger grids, would move the crossover, but neither was measured here.
- **Quote a ratio with its record, not as a constant.** The same 40-step `wake-array-3` march took 46.6 s in step A's record and 29.1 s here, and its threaded ratio moved from 1.30 to 1.08. At 21 rotors the ratios held across sessions: serial 0.90 in all three runs, threaded 4.0–4.6 ([[showcase-library-plan]] §5).

---

## 4. What the compile verdicts say

**No case earns a plain `admit`, and none could.** Every graph leaves the master bound's constants unmeasured ([[master-error-bound]]): $L$ (W1); $\sigma$, $\tau$ and $C_\mu$ (W3); and $\lVert A\rVert$ (W28) where no assembly is declared. Rule W56 forbids `admit` while a bound would carry a constant nobody measured. Every row is also *untyped* (L9/E5), because $L$ has never been measured. These are the compiler's standing conditions, not properties of the showcase.

**What decertifies each seam, beyond that:**

| where | rule | what the compiler read |
|---|---|---|
| the three farms' window seams | L4 block-share; E7 passivity | one window's block is far worse conditioned than the assembled seam ($\kappa = 1254$ against $1.09$). The assembled seam has a passivity defect of 1.807, with a spectrum that is one-signed negative ($\lambda_{\max}=-1.776$): a global sign, not an amplified mode (W306, [[case-study-rocket-bc-seam-atlas-0.1]] §10.3) |
| the plume, the sound, the circuit | L4 operator-content | each probed block is a multiple of the identity. An explicit step, or a one-number electrode port, answers a trace with a boundary coefficient, not an operator, so every bound carrying $1/\beta$ is a property of that coefficient |
| the cooled block, the circuit | L1 E3 | the seam joins two governing families (a coolant and a solid; a film and a lumped circuit), so there is no single monolith in the compiler's sense. Both sides declare a $\lambda_{\text{ref}}$, so $\tau$ is measurable against the tightly coupled pair |

**The eight rows added on 2026-09-29 and 30 follow their rectangular counterparts.** The S-channel and the plate with a hole compile as their rectangles do, every seam admitted, and R10 decertifies them after the scheme since W348 (below). The drawn arc is refused before the compiler, as the heated strip is, and the drawn farm because it has no declared graph (§2). The film, the sound and the winding are admitted uncertified on the same rules as their rectangles, above. The one difference: the drawn river's seam is admitted outright, where the plain river's reads L4 operator-content. **[AI Inference]:** the drawn river's solved flow varies along its seam, so the probed block is not a multiple of the identity, as it is for the plain river's uniform flow.

**Until W348, the three refusals were one rule: R10**, and so were the two drawn examples', the bend and the round insert, whose pieces are curved. The wall, the insert and the bracket each have two agents that declare an embedded elliptic sub-solve and share one governing family. R10 reads that as the family's region cut in two, each agent solving a global problem on its own piece, "the decomposition changes the operator". Each of their seams is admitted; the refusal is on the graph. The runs measure how far the iterated pieces are from the full domain: $2.69\times10^{-13}$, $3.58\times10^{-9}$ and $3.66\times10^{-11}$ of their scales, all far inside $10^{-6}$. The same three compile records derive `K_accelerator = direct-schur` for the interface, a scheme the compiler itself says *"makes the solve-incompleteness term vanish"*. R10 cannot read it: it runs at L2, before the scheme is derived.

**[AI Inference]:** R10's reason fits a fixed number of exchanges, as in style A. Styles B and C iterate until the pieces agree. For these linear problems the fixed point of that iteration is the undivided solution, and the measured agreement is consistent with that. R10 does not read whether a scheme iterates. Whether it should was a question for the compiler, not the workbench, opened as **W348** in [[gap-worklist]] and settled below.

**Settled by W348, 2026-09-30.** A piece now declares when the coupling supplies every datum its embedded solve uses on its cut (`elliptic_data_from_ports`), and the workbench's style-B and style-C pieces declare it. R10 then decides the cut piece after the scheme, at L5. A direct interface solve, or a sweep to a stated tolerance, is **decertified** with W168's cost ($149\times$ in coupling sweeps) instead of refused, citing the fixed-point consistency of restricted additive Schwarz (Frommer & Szyld, *SIAM J. Numer. Anal.* 39, 2001); T3 in [[formal-proofs-plan]] is not yet machine-checked. The seven rows above that read `refuse` now compile `admit-uncertified`. The graphs R10 was measured on keep their refusal at L2, measured rather than assumed: Tier 0's four windows as built and CS-S1 with its pressure solve embedded. Across every compile the full suite makes, 307 before and 311 after, the only rows that moved are the workbench's 9 compiles of these graphs. The W189 control's 40 artifacts are byte-identical ([[demo-finish-plan]] §5).

**The heated strip never reaches the compiler.** Its two agents share one mesh, and the whole temperature field crosses between them. The port algebra has no bond for that: `VOLUMETRIC` is not one of the five port types, and a sixth enters only through the amendment procedure, which has never been exercised (`NamedHoleError: PortAmendment`, gap O6/G16). This is the finding of [[case-study-thermal-strain-atlas-0.1]], and the same gap as W94's disk ([[case-study-wake-array-atlas-0.1]] §3). The page shows it as a refusal before the compiler, with the package's own message.

**How the graphs are built.**
- The farms compile through the vault's own graph for their tiling (`atlas/cases/scaling_ladder.py`'s `build`, with W346's exposed windows). A hand-drawn tiling that is not one of that ladder's rungs is refused before the compiler, with that reason (a test moves one window and sees it).
- Every other family builds its graph with `atlas/workbench/compile.py`. The seams come from the geometry: two windows meet where a cut face of one opens into the other. Each port declares a real Fourier prolongation over its faces, of up to 8 modes. An electrode's port has one mode, because its potential is one number.
- Every agent's `boundary_response` is the family's own arithmetic: a window's Dirichlet-to-Neumann map through its own sparse solve, the flux an explicit step would send, or a structure's reaction.
- The first draft returned the flow *out* of the window, and the compiler read a passivity defect of 299 on the wall. The flow *into* the window, as in the Steklov–Poincaré convention, is what is now declared.
- Cross-points are always declared (W162): named where three windows overlap, and `()` where there are none.

---

## 5. Conditions, and the two-minute rule

- **Where and how:** the served page (`python -m atlas.workbench`, port 8020), driven by clicks, on 2026-09-29, 05:14–05:41 EDT. The machine ("champ", 22 logical cores) was on AC power before and after every run. At every start the only other Python processes were the owner's phone-remote server (pids 31624 and 33492). The server was restarted three times during the pass, after code fixes (§6). The dark OS colour scheme was checked too: the page stays one consistent light theme.
- **Every run finished under two minutes.** The longest were `farm-21` at 70.0 s and `farm-12` at 69.9 s.
- **Every compile finished under two minutes.** The longest was `farm-21` at 95.1 s: 124 seams, each probed through its agents' own solves.
- **`farm-21`'s compile plus its run is 165 s, and the owner accepted it** on 2026-09-29. The plan's rule is per run ("each opens, runs and compares in under 2 minutes"), and the compile is its own step. The alternatives, caching the farm's compile or shrinking the example, were not taken.
- **The farms' step counts are the owner's, recalibrated on measurement.** The owner chose 14 and 8 macro-steps for 12 and 21 rotors. Measured in step A, 8 steps at 21 rotors took 122.1 s and 7 took 120.8 s, and 14 steps at 12 rotors took 112.6 s. So the examples run 12 and 6 ([[showcase-library-plan]] §5). In so few steps a wake has not reached the next rotor, so farm power covers the start-up only, and the page says so. W346's 40-step comparison remains the record for the developed wake.

---

## 6. The records, and what the pass fixed

**Records**, one compile and one run per case, in `out/workbench/records/step-f/`, named `<case>-compile-<time>.json` and `<case>-<time>.json`:

| case | compile | run |
|---|---|---|
| `wake-array-3` | `20260929-051456` | `20260929-054032` |
| `farm-12` | `20260929-052121` | `20260929-053611` |
| `farm-21` | `20260929-053019` | `20260929-053206` |
| `wall-2` | `20260929-051618` | `20260929-053805` |
| `plate-insert` | `20260929-051653` | `20260929-051702` |
| `plate-circuit` | `20260929-051721` | `20260929-051727` |
| `plume-2` | `20260929-053856` | `20260929-053918` |
| `sound-air-water` | `20260929-051821` | `20260929-051828` |
| `bracket-2` | `20260929-051848` | `20260929-051924` |
| `heated-strip` | `20260929-051943` | `20260929-051949` |
| `cooled-block` | `20260929-052009` | `20260929-052016` |

The two drawn examples' records are in `out/workbench/records/step-g/`: `bend-3` (compile `20260929-115906`, run `20260929-115908`) and `insert-round` (compile `20260929-115909`, run `20260929-115911`).

The eight rows added on 2026-09-29 and 30 are in `out/workbench/records/step-j/<case>/`, named `compile-<time>.json` and `<time>.json`, with `summary.json` beside them:

| case | compile | run |
|---|---|---|
| `s-channel` | `20260929-213314` | `20260929-213316` |
| `ring-film` | `20260929-213316` | `20260929-213316` |
| `river-bend` | `20260929-213656` | `20260929-213700` |
| `sound-lens` | `20260929-213321` | `20260929-213324` |
| `plate-hole` | `20260929-213326` | `20260929-213350` |
| `bimetal-arc` | `20260929-213351` | `20260929-213353` |
| `cooled-winding` | `20260929-213354` | `20260929-213356` |
| `farm-hill` | `20260930-014634` | `20260930-014815` |

The forked river's are in `out/workbench/records/demo-fork/`:
- `river-fork-compile-20260930-172156.json` and `river-fork-20260930-172211.json`;
- `drawn-y-20260930-172123.json`, the owner's scenario walked in the page (a smooth Y drawn with its stem on the left, then Run).

A run record holds the case as it marched, every per-step time, the metrics, each check with its tolerance and registration date, the field differences, and the machine's state before and after. A compile record holds every decision, the compiler's report, and the case as compiled.

**Found in this pass, and fixed before the records above:**
- **The farm's Results table had two rms columns that disagreed.** The family's own column (both velocity components) was filled for the threaded arm only. The new generic column (the streamwise component alone) was filled for both arms. Now every decomposed arm carries the family's measures, and the column is labelled "(u and v)". `farm-12` and `wake-array-3` were re-run so their records carry both arms.
- **Stale labels.** After another case was opened, the Check and Results labels still described the last compile and run in the session: "compiled: admit-uncertified" stood beside `farm-12` when `farm-21` was the case compiled. They now read "before a change", and the Results step says whose results it shows. The rebuilt page keeps the rule: its Run & results tab says whose results it shows, and a compile of another case reads "not compiled since this case was opened", naming the case that was.

---

## 7. Not in the gallery

- **The optional Gray–Scott case, and the CS-13 and CS-14 adapters.** Both were optional in the plan and were not built, so `ROT` has no showcase case.
- **The rocket and the vehicle union.** They stay research records ([[showcase-library-plan]] §4).
- **A speedup below farm scale.** None was measured in the showcase pass (§3). The Fast examples of §8 are larger cases built for it.

---

## 8. The Fast examples (demo item 1.4, 2026-09-30 to 10-01)

**What they are.** The header's *Fast example* opens, for the chosen kind, one complete case chosen to show the pieces running faster than the whole; its arrow lists the kind's other examples ([[demo-fast-examples-plan]]). Each kind registered its bars in its family module before its first timed run (`FAST`, `atlas/workbench/fast.py`):
- the fastest decomposed arm at $s=t_{\text{full}}/t_{\text{arm}}\ge3$;
- agreement with the full domain within $10^{-9}$ of the field's scale where the arm is the same algebra, $10^{-3}$ for multirate, and 3% in farm power for the wind farm.

**Selection is not measurement.** A sweep chose each configuration (`scripts/fast_examples.py sweep`), and a confirmation run from a fresh start measured it (`confirm`). Every timed run was on AC power with no other Python process running. The other two chats were writing documentation and theory, with nothing on the demo (the owner, 2026-09-30).

**Step 0, the micro-benchmarks** (`scripts/fast_step0.py`; each rule registered in the script before the first run):
- **P scales on large windows.** A numpy five-point update reached 3.56–3.81× on 8–16 threads at windows of $256^2$ and $512^2$ cells, and at most 1.53× at $128^2$. The river's sparse step reached 4.2–8.9×.
- **SuperLU releases the GIL.** Its factorization ran 3.1–3.7× faster on 4 threads for pieces of up to 16,384 unknowns, and only 1.2× at 65,536.
- **Where the explicit update leaves cache.** Its cost per cell rose from 8.4 to 19.7 ns between $256^2$ and $512^2$ cells; the registered rule (1.5 times the smallest size's cost) puts the cliff at $1024^2$.

**The table**, from the confirmation runs (`out/workbench/records/fast/confirm-*.json`):

| kind | example | mechanism | grid cells | $s$ | agreement with the full domain (bar) | wall | bars |
|---|---|---|---|---|---|---|---|
| wind farm | `fast-farm`: 5 rotors on 12 windows, 4 threads, 30 macro-steps | P | 163,328 | **4.62** | farm power 1.84% (3%) | 48.9 s | met |
| heat conduction | `fast-heat`: a copper spreader in a steel plate, style M | M | 245,760 | **5.58** | $1.22\times10^{-4}$ of the span ($10^{-3}$) | 35.3 s | met |
| river plume | `fast-river`: 6 windows, 6 threads, 24 steps per exchange | P | 460,800 | **5.85** | $4.7\times10^{-17}$ of the peak ($10^{-9}$) | 42.4 s | met |
| sound | `fast-sound`: 4 windows, 4 threads, 16 steps per exchange | P | 296,000 | **4.67** | the full domain bit for bit ($10^{-9}$) | 26.6 s | met |
| loaded structure | `plate-hole` | O3 | 12,800 | 0.008 | $3.2\times10^{-12}$ | 19.3 s | below |
| current in a plate | `plate-circuit` | O3 | 12,800 | 0.275 | $1.1\times10^{-13}$ | 0.2 s | below |
| heated structure | `bimetal-arc` | X: at most 2 | 11,664 | 1.17 | the split's own controls | 2.8 s | below, by construction |
| cooled block | `cooled-block` | O3 | 12,000 | 0.004 | $2.6\times10^{-10}$ | 1.5 s | below |

**What the sweeps found:**
- **One exchange a step is memory-bound.** With one step per exchange, the river's windows ran 0.84–1.02× the full domain on 4–16 threads. Each window's sparse matrix streams from memory every step, as the whole domain's does, so the threads share one memory bus. Several steps per exchange on a halo as deep (`run.exchange_every`) keep a window in a core's cache across its steps, and the windows' own cells stay the full domain's: the river's to $10^{-16}$, the sound's bit for bit.
- **Multirate does not suit the river.** Explicit upwind advection adds a numerical diffusion $\tfrac{u\,\Delta x}{2}(1-\mathrm{CFL})$ that depends on the step. Here it is about as large as the physical mixing, so a reach stepping five times longer would differ from the full domain by percents, not $10^{-3}$. Conduction has no such term: its multirate arm agrees to $1.2\times10^{-4}$ after 1000 steps, and the time interpolation's error falls roughly as the inverse of the number of steps.
- **Subnormal floats.** The river's first confirmation measured 2.74×, not the sweep's 4.35×. Its plume's far tail had decayed into subnormal floats (4,268 cells by macro-step 300), whose arithmetic is many times slower, and they sat unevenly among the windows. Every arm now sets a concentration under $10^{-30}$ g/m$^3$ to zero once a macro-step, and the river measured 5.85×.
- **The farm needs 28 macro-steps** for its power to come within 3%: 8.8% at step 22, as the first wake reaches the next row, and 1.84% at 30. All three arms take about 4 s a step, so its example runs the threaded windows and the whole domain. The serial arm, the threads' bit-for-bit control, stays in Run settings and in the tests.
- **Four to six threads** were best, the laptop's six performance cores; 12–16 were slower.
- **The first confirmations are superseded and kept:** the farm 4.25× in 131.6 s (all three arms at 28 steps, over two minutes), the river 2.74× (the subnormals), the sound 2.93×, and conduction 5.40×. Each configuration changed for the reasons above, and each was confirmed again.

**What limits the other four (O3).** A steady structure, a film on its circuit, and a block with its coolant are each one direct solve here, factored once and reused. Their pieces iterate until they agree, so the whole is faster, by 3.6× to 250×. Substructuring would factor the pieces at once and solve their interface directly; it is what could win, and it is not built. The heated structure is split by physics, which can at most halve the time; it measured 1.17×.

**The compile.** Five of the eight compile `admit-uncertified`.
- `fast-river` was first refused at `L2/R10/halo`: each window takes 24 steps per exchange, and the windows overlapped by only 16 cells. The compiler was right about the declaration, so the windows now overlap by as many cells as their steps, and the graph is admitted.
- `fast-sound`'s style A has no declared graph yet, so it is refused before the compiler, as `bimetal-arc` is by the port vocabulary (§4).

**[AI Inference]:** the four ratios belong to these 2-D cases on this laptop. Every card's fixed line says what the plan expects to grow them: complicated shapes, three dimensions, and learned experts. No 3-D ratio has been measured in the workbench.

---

## See Also

- [[showcase-library-plan]] — the plan this gallery completes, with each step's "Built" notes
- [[outcome-c4-path-to-declarative-cases]] — the workbench's original plan: the loader, the executor and the GUI
- [[outcome-c4-modular-multiphysics]] — the compiler and the port algebra these verdicts come from
- [[decomposition-speed-by-rotor-count]] — W346, the farm's speed record that the farm rows reproduce
- [[case-study-thermal-strain-atlas-0.1]] — why the heated strip has no port
- [[master-error-bound]] — the constants every verdict here leaves unmeasured
- [[port-algebra-atlas-0.1]] — the five port types, and why a volumetric bond is not one
- [[outcome-c1-decomposition-vs-monolith]] — decomposition against the undivided solve, as a research claim
- [[00-atlas-0.1-outcome]]
