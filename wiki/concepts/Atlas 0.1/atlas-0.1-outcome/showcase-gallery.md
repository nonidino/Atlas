# The showcase gallery

**Type:** Outcome page — **gallery of measured showcase runs** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** written 2026-09-29 from one pass through the served workbench on this laptop. Every example was opened from *File > New from example*, checked, compiled with the Atlas compiler, and run decomposed and on the full domain. Every number below is read from that pass's records in `out/workbench/records/step-f/` (§6). **Every row is a showcase, not a research record** ([[showcase-library-plan]] §2): one pass, checks registered before the family's first run, and a record file. There are no predictions, audits or follow-up tiers.
**Hub:** [[00-atlas-0.1-outcome]] · **Plan:** [[showcase-library-plan]] · **Parent claim:** [[outcome-c4-modular-multiphysics]] · **Code:** `atlas/workbench/` (`python -m atlas.workbench --open`, then *File > Showcase gallery*)

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

where the scale is the temperature span, the peak concentration, the largest displacement, or the heating's rise. The Results step also shows the rms difference of the displayed field, $\big(\tfrac{1}{n}\sum_i (u^{\text{arm}}_i-u^{\text{full}}_i)^2\big)^{1/2}$, in the field's own units. Every tolerance was registered in its family's module before that family's first run, and none was changed afterwards.

---

## 2. The gallery

Measured 2026-09-29, 05:14–05:41 EDT, in the served page on AC power (§5). Every check that could be measured passed.

| case | style | port type | compile (time) | balance check (tolerance) | reference check (tolerance) | run: wall time (steps) | label |
|---|---|---|---|---|---|---|---|
| `wake-array-3`: 3 rotors, 6 windows | A | `MECH` | admit-uncertified (7.3 s; 9 agents, 13 seams, 2 cross-points) | incompressibility $1.65\times10^{-16}$ ($10^{-9}$) | farm power $-4.07\%$ against the full domain ($25\%$) | 29.1 s (40 macro-steps) | showcase, not research record |
| `farm-12`: 12 rotors, 24 windows | A | `MECH` | admit-uncertified (45.3 s; 36 agents, 62 seams, 15 cross-points) | $2.22\times10^{-16}$ ($10^{-9}$) | $-8.00\%$ ($25\%$) | 69.9 s (12 macro-steps) | showcase, not research record |
| `farm-21`: 21 rotors, 48 windows (W346's farm) | A | `MECH` | admit-uncertified (95.1 s; 69 agents, 124 seams, 35 cross-points) | $2.57\times10^{-16}$ ($10^{-9}$) | $-4.03\%$ ($25\%$) | 70.0 s (6 macro-steps) | showcase, not research record |
| `wall-2`: a steel layer and a copper layer, steady | C | `THERM` | **refuse**, R10 (0.06 s); its one seam admitted | energy $1.67\times10^{-11}$ ($10^{-6}$) | closed form $q=\Delta T/\sum_i L_i/k_i$: $8.64\times10^{-13}$ ($10^{-6}$); full domain $2.69\times10^{-13}$ of the 100 K span ($10^{-6}$) | 0.7 s (5 timed repeats) | showcase, not research record |
| `plate-insert`: a copper insert in a steel plate, transient | B | `THERM` | **refuse**, R10 (0.09 s); its one seam admitted | energy $1.64\times10^{-8}$ ($10^{-6}$) | full domain $3.58\times10^{-9}$ of the 100 K span ($10^{-6}$) | 4.5 s (40 steps) | showcase, not research record |
| `plate-circuit`: a graphite film on a battery and a resistor | D | `ELEC` | admit-uncertified (0.08 s; 2 seams, one per electrode) | Kirchhoff $1.43\times10^{-12}$, energy $3.16\times10^{-13}$ ($10^{-6}$ each) | the film and circuit as one sparse solve: $1.07\times10^{-13}$ of the current ($10^{-6}$) | 0.6 s (5 timed repeats) | showcase, not research record |
| `plume-2`: a pollutant plume, a shallow reach into a deep one | A, one explicit step per exchange | `ADVEC` | admit-uncertified (0.14 s) | mass $8.12\times10^{-14}$ ($10^{-9}$) | full domain $8.76\times10^{-17}$ of the peak ($10^{-10}$) | 7.9 s (6,000 steps) | showcase, not research record |
| `sound-air-water`: sound from air into water | C, explicit | `MECH` | admit-uncertified (0.02 s) | energy $8.43\times10^{-16}$ ($10^{-10}$) | reflection $R$ within $3.10\times10^{-10}$ of $(Z_2-Z_1)/(Z_2+Z_1)$ ($10^{-6}$) | 2.2 s (2,000 steps) | showcase, not research record |
| `bracket-2`: steel and aluminium under a 5 kN load | B, on a structure | `MECH` | **refuse**, R10 (0.33 s); its one seam admitted | forces $1.28\times10^{-10}$ ($10^{-6}$) | full domain $3.66\times10^{-11}$ of the largest displacement, 0.770 mm ($10^{-6}$) | 31.5 s (5 timed repeats) | showcase, not research record |
| `heated-strip`: a steel-and-copper strip, heated | split by physics | none: the bond is volumetric | **refused before the compiler** by the port vocabulary (0.02 s) | heat in = heat stored, $9.95\times10^{-11}$ ($10^{-9}$) | the unsplit solver, equal bit for bit over 60 steps (exact) | 1.5 s (60 steps) | showcase, not research record |
| `cooled-block`: a chip on a copper block under a water channel | C, at a seam between two physics | `THERM` | admit-uncertified (0.08 s) | energy $6.95\times10^{-11}$ ($10^{-6}$), 2 kW per metre of depth made and carried out | full domain $2.56\times10^{-10}$ of the 29.5 K rise ($10^{-6}$) | 1.8 s (5 timed repeats) | showcase, not research record |
| `bend-3`: a quarter ring of steel drawn with arcs, on three curved windows | B, drawn | `THERM` | **refuse**, R10 (0.17 s); its two seams admitted | energy $1.24\times10^{-9}$ ($10^{-6}$) | full domain $3.07\times10^{-10}$ of the 100 K span ($10^{-6}$) | 1.98 s (5 timed repeats) | showcase, not research record |
| `insert-round`: a round copper insert in a steel plate, cut along the circle | C, drawn | `THERM` | **refuse**, R10 (0.14 s); its one seam admitted | energy $9.86\times10^{-12}$ ($10^{-6}$) | full domain $1.01\times10^{-10}$ of the 100 K span ($10^{-6}$) | 1.85 s (5 timed repeats) | showcase, not research record |

**The last two rows are drawn geometry** (case file 0.4, added 2026-09-29 on the owner's request: irregular domains and windows that are not rectangles; see [[showcase-library-plan]] §6). They were not taken in the page pass above. They were compiled and run at 11:59 EDT through the workbench's own `Workbench` object, the code the page drives, headless and **on battery**, because the served page could not be drawn for screenshots at the time. So their wall times are not speed figures. Their checks are the conduction family's, registered on 2026-09-28 before its first run, and unchanged. The bend's staircase moves its heat flow by $-0.97\%$ from the ring's continuum value $k\,\Delta T\ln(r_o/r_i)/\theta$: information, not a check.

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

- **Threads pay only at farm scale.** At 12 and 21 rotors the threaded arm is $4.83\times$ and $4.55\times$ faster than the full domain. Both fall inside W346's measured ranges ($4.0$–$5.7\times$ and $4.3$–$4.7\times$, [[decomposition-speed-by-rotor-count]] §5). The serial arm stays below 1, as W346 found. W346 timed the developed wake at step 40, and these runs time the start-up. So this is a live reproduction of the ratio, not a new measurement of it.
- **Below about 30,000 cells, one direct solve or one explicit step on the whole grid beats every spatially decomposed arm.** The margin runs from $1.4\times$ (the sound's two pieces) to about $490\times$ (the bracket's serial Schwarz, 616 iterations to $10^{-12}$). The heated strip is split by physics, not space, and its three arms tie within the noise of a 3 ms step.
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

**The three refusals are one rule: R10**, and so are the two drawn examples', the bend and the round insert, whose pieces are curved. The wall, the insert and the bracket each have two agents that declare an embedded elliptic sub-solve and share one governing family. R10 reads that as the family's region cut in two, each agent solving a global problem on its own piece, "the decomposition changes the operator". Each of their seams is admitted; the refusal is on the graph. The runs measure how far the iterated pieces are from the full domain: $2.69\times10^{-13}$, $3.58\times10^{-9}$ and $3.66\times10^{-11}$ of their scales, all far inside $10^{-6}$. The same three compile records derive `K_accelerator = direct-schur` for the interface, a scheme the compiler itself says *"makes the solve-incompleteness term vanish"*. R10 cannot read it: it runs at L2, before the scheme is derived.

**[AI Inference]:** R10's reason fits a fixed number of exchanges, as in style A. Styles B and C iterate until the pieces agree. For these linear problems the fixed point of that iteration is the undivided solution, and the measured agreement is consistent with that. R10 does not read whether a scheme iterates. Whether it should is a question for the compiler, not the workbench, and it is open as **W348** in [[gap-worklist]]. The Check step shows the refusal as the compiler gave it, with a note saying so.

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

A run record holds the case as it marched, every per-step time, the metrics, each check with its tolerance and registration date, the field differences, and the machine's state before and after. A compile record holds every decision, the compiler's report, and the case as compiled.

**Found in this pass, and fixed before the records above:**
- **The farm's Results table had two rms columns that disagreed.** The family's own column (both velocity components) was filled for the threaded arm only. The new generic column (the streamwise component alone) was filled for both arms. Now every decomposed arm carries the family's measures, and the column is labelled "(u and v)". `farm-12` and `wake-array-3` were re-run so their records carry both arms.
- **Stale labels.** After another case was opened, the Check and Results labels still described the last compile and run in the session: "compiled: admit-uncertified" stood beside `farm-12` when `farm-21` was the case compiled. They now read "before a change", and the Results step says whose results it shows.

---

## 7. Not in the gallery

- **The optional Gray–Scott case, and the CS-13 and CS-14 adapters.** Both were optional in the plan and were not built, so `ROT` has no showcase case.
- **The rocket and the vehicle union.** They stay research records ([[showcase-library-plan]] §4).
- **A speedup below farm scale.** None was measured (§3).

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
