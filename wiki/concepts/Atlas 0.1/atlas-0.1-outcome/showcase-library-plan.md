# The classical showcase library: plan

**Type:** Outcome page — **build plan** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** written 2026-09-28 on the owner's request. **Step 1 (the geometry section) was built the same day** (§6), and so was **step 2, the general runner for style A** (the wind-farm family). **Step 3, the solver-family interface with styles B, C and D, was built on 2026-09-29**, with cases 2 and 5 as its two new families, and **step 4, cases 3, 4, 6, 7 and 8, the same day**. **Steps 5 (the gallery) and 6 (the compile) were built the same day too**, and every example was compiled and run in the served page: [[showcase-gallery]]. Details are in §5's "Built" notes. Only the plan's optional items are not built: the Gray–Scott case and the CS-13/CS-14 adapters. It extends [[outcome-c4-path-to-declarative-cases]] from one physics family to a library, and plans the workbench's geometry section to fit it. Runtimes and effort in §4–§5's tables are estimates, marked **[AI Inference]**; measured numbers are in the "Built" notes, each with its record.
**Hub:** [[00-atlas-0.1-outcome]] · **Parent claim:** [[outcome-c4-modular-multiphysics]] · **Code:** `atlas/workbench/`

---

## 1. Why a library, and why it has been small

**The question (owner, 2026-09-28):** can't Atlas be used with any classical decomposed simulation, and why not expand the library with easy cases that show its diversity and run live in a couple of minutes?

**In principle, yes.** The compiler reads capability records, not solvers ([[atlas-implementation]]): what an expert declares about its ports, clock and boundary behaviour. Three things kept the library small, and none of them is a limit of the framework:

1. **No general runner.** Every real case study has its own hand-written time loop: `FSIRollout`, `GroundRollout`, `BrakeRollout`, `FrontWingRollout`, `RaceRollout`, `UnionRollout`, `CarUnion`, `wind_farm_design.Rollout`, and the rocket's `CoupledEpisode`. The generic `atlas/solve.py` runtime is used only by `thermal_seam`. So a new case has cost hundreds of lines of bespoke code.
2. **Every case so far was a research study,** with registered predictions, audits and follow-up tiers.
3. **Several cases were chosen to probe learned experts or theory gaps, not to show breadth.** The rocket is the extreme: combustion on millisecond scales, flight on minute scales, and reacting compressible flow ([[case-study-rocket-bc-seam-atlas-0.1]]).

**[AI Inference]:** once a general runner exists, a simple classical case costs a solver adapter (often 50–150 lines of numpy), a case file, and two sanity checks.

---

## 2. What qualifies as a showcase case

| criterion | why |
|---|---|
| **simple regimes:** one clock, or a small integer clock ratio; no shocks, no combustion | keeps the multi-clock problems out ([[case-study-vehicle-march-atlas-0.1]]: a 400× clock span) |
| **runs in under 2 minutes** on the laptop, ideally seconds, at about 100k cells or fewer | computed live in the showcase |
| **shows something the others do not:** a port type, a coupling style, a kind of decomposition | breadth is the point |
| **has a reference:** the same solver on the whole domain, or a textbook answer | the Run & compare step needs one |
| **carries two checks:** a balance that closes (mass, energy, charge), and agreement with the reference within a stated tolerance | a live demo must not show a wrong number |
| **labelled "showcase", not "research record"** | the depth of the evidence differs, and the label says so |

---

## 3. The four coupling styles the runner must support

| style | used for | mathematics | status |
|---|---|---|---|
| **A. Overlapping windows, one exchange per step** (time-dependent) | wind farm, waves, plumes | blend by a partition of unity $\sum_i \chi_i = 1$, then the global operators; the Schwarz waveform relaxation run for one sweep ([[atlas-and-standard-dd-theory]]) | built (wake array; W346's harness, serial and threaded) |
| **B. Overlapping windows, iterate until they agree** (steady problems) | heat, elasticity, groundwater | alternating or additive Schwarz: solve each window with its neighbours' latest trace until $\lVert u^{k+1}-u^{k}\rVert < \varepsilon$; the convergence curve is itself a display | planned as new and small; built in the workbench (step 3) |
| **C. Material interfaces without overlap** | copper against steel; air against water | Dirichlet–Neumann (or Robin) iteration: continuity of the field and of the normal flux, $u_1 = u_2$ and $k_1\,\partial_n u_1 = k_2\,\partial_n u_2$ on $\Gamma$ | planned as new; built in the workbench (step 3), iterated and explicit |
| **D. Field joined to lumped parts** | circuits, coolant loops, rotors | the field's boundary integral is the lumped part's port variable (current, heat, torque), exchanged per step or solved tightly | built (CS-13 [[case-study-cooling-loop-atlas-0.1]], CS-14 [[case-study-powertrain-atlas-0.1]], actuator disks); in the workbench, the plate on a circuit (step 3) |

---

## 4. The cases

All 2-D. Runtimes and code sizes are **[AI Inference]** estimates.

| # | case | physics and port | style | what only it shows | reference and checks | runtime | new code |
|---|---|---|---|---|---|---|---|
| 1 | **Wind farm, 1–21 rotors** | incompressible flow + actuator disks; `MECH`, `ROT` | A | the measured parallel speedup, live | full domain ([[decomposition-speed-by-rotor-count]]) | 10 s – 2 min | none: the runner only |
| 2 | **Two-material plate** (copper insert in steel), steady and transient | heat conduction; `THERM` | C, then B | materials and resolution per region; the convergence plot | full domain; the two-layer wall in closed form, $q = \Delta T / \sum_i L_i/k_i$; energy balance | seconds | ~100 lines |
| 3 | **Pollutant plume down a river** in two reaches (different speed and mixing) | advection–diffusion; `ADVEC` | A, one-way | transport across a seam | full domain; mass balance | seconds | ~80 lines |
| 4 | **Sound through two media** (air-like into water-like) | acoustic wave equation; pressure–velocity (`MECH`) | C, explicit | reflection and transmission at a seam | textbook $R = (Z_2 - Z_1)/(Z_2 + Z_1)$ with $Z = \rho c$; energy balance | seconds | ~100 lines |
| 5 | **Current spreading in a resistive plate** fed by a battery and a resistor | $\nabla\cdot(\sigma\nabla\phi) = 0$ joined to a lumped circuit; `ELEC` | D | the electrical port on a spatial field | full domain; Kirchhoff and energy balance, $\sum I = 0$, $P = I^2R$ | seconds | ~120 lines |
| 6 | **Two-material bracket under load** | plane-stress elasticity (`ThermoStruct2D.solve_mechanical`); `MECH` | B | structures, not only flows | full domain; force balance | seconds | ~80 lines of wiring |
| 7 | **Heated plate that expands** | conduction → elasticity, same mesh | co-located split | the split by physics rather than space | full domain ([[case-study-thermal-strain-atlas-0.1]]) | seconds | wiring |
| 8 | **Heated block cooled by channel flow** | advection–diffusion fluid + conduction solid; `THERM` | C at a two-family seam | two physics families meeting at one seam | full domain; energy balance | < 1 min | ~120 lines |
| + | **Coolant loop, powertrain** | CS-13, CS-14 | D | loop and circuit topologies | their existing balances ($2\times10^{-14}$, $1.9\times10^{-15}$) | seconds | adapters |
| opt. | **Pattern-forming reaction–diffusion** (Gray–Scott) | nonlinear reaction–diffusion | A | nonlinearity crossing seams; visual | full domain | seconds | ~60 lines |

**Coverage:** all five port types (`MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC`), all four coupling styles, and the co-located split. **Not in the showcase:** the rocket and the vehicle union, which stay research records.

---

## 5. What to build, in order

| step | what | effort **[AI Inference]** | depends on |
|---|---|---|---|
| **1** | **Geometry section** of the workbench (§6) | 1–2 sessions | — |
| **2** | **General runner for style A**, the wind-farm family: march a case file, compare with the full domain, live fields (plan step 3 of [[outcome-c4-path-to-declarative-cases]]) | 1–2 sessions | 1 |
| **3** | **A solver-family interface** (state, step given boundary traces, the full-domain equivalent, restriction and assembly) **plus styles B, C, D** | ~1 session | 2 |
| **4** | **Cases 2–8**, about half a session each; 2 and 3 first, then 5 | ~3–4 sessions | 3 |
| **5** | **Showcase gallery:** File > New from example lists every case; each opens, runs and compares in under 2 minutes | small | 4 |
| **6** | **Compile verdict per case** (plan step 2's loader), so Check shows the compiler's per-seam output | ~1 session | 3 |

**Risks.**
- The general runner is where heterogeneous cases will expose design gaps. Keep a per-family adapter rather than one universal stepper.
- Cross-points (three or more windows meeting) break naive blending. The compiler has rules for them (`L2/I2/G1`), and those rules have been wrong once, on a circuit ([[case-study-cooling-loop-atlas-0.1]]).
- Keep every showcase case to one clock.

> **Built, 2026-09-28: step 2, the general runner for style A, for the wind-farm family** (`atlas/workbench/runner.py`, `families/windfarm.py`, `tiling.py`, `runview.py`, `checks.py`, `machine.py`; `tests/test_workbench_runner.py`). A person can open a wind-farm case, press **Run**, and watch it march decomposed (serially and on threads) and on the full domain in turns: the fields, their difference, each arm's time per step and the farm power are drawn live, **Run > Stop** ends it at a whole macro-step, and the Results step shows each check against a tolerance registered before any run. Each run's record is saved beside the case file.
> - **It is W346's march, lifted, not re-derived.** On the six-window example the serial, threaded and full-domain arms equal W346's own `E`, `Ep4` and `F` to the bit (a test). Windows may be any rectangles. The partition of unity (`tiling.RectangleTiling`) is `ArrayTiling.weights` to the bit on the three measured tilings, and is certified by `GridPartitionOfUnity`'s identity residual ($1.1\times10^{-16}$ on every example).
> - **One window equals the full domain to the bit where the discretization is the same**: the elliptic part inside the window, and the blend alone. The measured arrangement (exposed windows, one global projection) is not the full domain's discretization even at one window, because it moves the projection out of the window. A test pins both.
> - **The three checks, registered before the first march:**
>   - incompressibility closes to $10^{-9}$ in the operator that enforced it;
>   - farm power within 25% of the full domain (W346's own B5);
>   - threaded equal to serial, bit for bit (W346's B1).
> - **W346's result, reproduced in the page on this laptop**, on AC power, with the owner's phone-remote server running. Records: `out/workbench/records/step-a/`. Speed is $t_{\text{full}}/t_{\text{arm}}$, so above 1 is faster than the full domain.
>
> | rotors | macro-steps | serial | threaded | farm power vs full | rms velocity difference | wall time |
> |---|---|---|---|---|---|---|
> | 3 | 40 | 0.854 | 1.298 (4 threads) | $-4.07\%$ | 0.0470 | 46.6 s |
> | 12 | 14 | 0.937 | 3.955 (4) | $-8.27\%$ | 0.0401 | 112.6 s |
> | 21 | 8 | 0.898 | 4.622 (8) | $-6.30\%$ | 0.0328 | 122.1 s |
> | 21 | 7 | 0.907 | 4.015 (8) | $-5.32\%$ | 0.0309 | 120.8 s |
>
> - **At 21 rotors the threaded decomposition beats the full domain and the serial one does not**, in both runs. Threaded equalled serial bit for bit at every step of every run.
> - The 3-rotor, 40-step run is **W346's accuracy row exactly**: full-domain farm power 1.3580 against W346's 1.358, $-4.07\%$ against $-4.1\%$, rms 0.0470 against 0.047. This is the same arithmetic, so it reproduces.
> - **The step counts are the owner's decision, recalibrated on measurement.** W346 marched 40 macro-steps. With all three arms that is about 8 minutes at 21 rotors, against the 2-minute rule. The owner chose short marches, proposed as 14 and 8 macro-steps ("about 100 s each"). Measured, 8 steps at 21 rotors took 122.1 s. Then 7 took 120.8 s, because the laptop's cost per step rose about 15% between back-to-back runs while the ratios held. So the examples now run **12 and 6 macro-steps**, sized to the slowest step measured, with room for the compile. In so few macro-steps a wake has not reached the next rotor: 3.5 D at the freestream speed takes 17.5 steps, which the adapter computes from the case. So farm power covers the start-up only. The page says so, and W346's 40-step comparison remains the record for the developed wake.
> - **Not built:** the solver-family interface and styles B–D (step 3), the other families (step 4), the gallery (step 5), the compile (step 6). A run of a family without an adapter is refused, with the registry's reason.

> **Built, 2026-09-29: step 3, the solver-family interface and coupling styles B, C and D.** Code: `atlas/workbench/fv.py`, `styles.py`, `families/conduction.py`, `families/electric.py`; the case file `atlas-workbench/case@0.3`. Tests: `tests/test_workbench_fv.py`, `tests/test_workbench_families.py`.
> - **The interface was derived from two real families, as §7 required.** The wind farm and conduction each fill the same five slots: the state, restriction, a step given boundary traces, the full-domain equivalent, and assembly with a balance. `styles.py` tabulates what fills each slot in each family. The two share no code at the solver level, and should not: the wind farm's step is W346's to the bit.
> - **One finite-volume core** (`fv.py`) serves conduction and the electric potential, and will serve the plume and the cooled block. It has harmonic-mean faces, upwind advection, and any set of cells. It closes a face that leaves the set in one of three ways, and those three are what the styles differ in. The full domain is the same call on every cell, so a one-window decomposition is the full-domain system to the bit **by construction**, and a test checks it, steady and transient.
> - **The three styles:**
>   - **B:** restricted additive Schwarz iterated to a tolerance. Threads change no bit.
>   - **C:** Dirichlet–Neumann with stated relaxation. The page shows the first factor, Aitken's rule if chosen, and the 1-D contraction factor $\rho = k_D L_N/(k_N L_D)$. A test shows the unrelaxed iteration diverging when $\rho = 14.8$, and the optimal factor $1/(1+\rho)$ rescuing it.
>   - **D:** a field's boundary integral is a lumped part's port variable. The electrodes' currents drive a circuit solved by nodal analysis; ideal and Norton batteries are both allowed.
> - **Case 2, the two-material plate**, is two examples.
>   - The two-layer wall (steel, then copper; style C, steady). Its heat flow of 1744.186 W/m is within $4.3\times10^{-13}$ of the closed form $q = \Delta T/\sum_i L_i/k_i$ (the full domain: $8.6\times10^{-13}$). It takes 3 Dirichlet–Neumann iterations, at $\rho = 0.075$.
>   - The copper insert in a steel plate (style B, transient, 40 steps of 120 s). The energy balance closes to $1.6\times10^{-8}$, the iteration's floor (the full domain: $5.2\times10^{-12}$). It is within $3.6\times10^{-7}$ K of the full domain, and threaded equals serial bit for bit over all 40 steps.
> - **Case 5, current spreading in a resistive film** (style D): a 30 $\mu$m graphite film on a 12 V battery ($r = 0.5\,\Omega$ inside) and a resistor ($R = 1\,\Omega$).
>   - It carries 5.001792 A, within $1.1\times10^{-13}$ of the film and circuit solved as one sparse system.
>   - The film's resistance between its electrodes is $0.899\,\Omega$, and 22.495 W of the 60.022 W delivered is heat in the film.
>   - Kirchhoff closes to $1.4\times10^{-12}$ and the energy balance to $3.2\times10^{-13}$.
>   - The unrelaxed iteration's factor is $G(r+R) = 1.67$, where $G$ is the film's conductance: over 1, so without relaxation it would diverge. Aitken's rule converges in 3 iterations.
>   - **Its control is round-off, not bits.** The partition is physical (a field against a lumped network), so its full-domain reference is different linear algebra on the same discretization.
> - **What the showcase does not show: a speedup.** At these sizes (6,400–15,360 cells) the full domain's sparse direct solve beat every decomposed arm, measured in the page on AC power: 2.6× faster than Dirichlet–Neumann on the wall, 3.6× than style D, and 24–35× than Schwarz on the insert. These cases show the coupling, and the page says so with the ratio. Records: `out/workbench/records/step-b/`.
> - **Found in the served page:** the family selector disabled none of the planned families. It had passed family ids to a list whose values in the page are labels; this dates from the shell and is now pinned by a test. Also found: the tolerance box displayed $10^{-10}$ as "+0.000", and a plot autoscaled two equal answers into two lines.
> - **Not built yet:** the attachments' canvas and table layer (the circuit is set in the case file and drawn nowhere), cases 3, 4, 6, 7 and 8, the gallery, and the compile.

> **Built, 2026-09-29: step 4, cases 3, 4, 6, 7 and 8**, as five new families: `atlas/workbench/families/{plume,acoustics,elasticity,thermoelastic,cooling}.py`, and `fe.py` for the two structural ones. Tests: `tests/test_workbench_cases.py`. Every check below was registered in its family's module before that family's first run. The figures are from page runs on AC power, records in `out/workbench/records/step-c/`.
>
> | case | style | the checks, measured | wall time |
> |---|---|---|---|
> | 3, a pollutant plume down a river (a shallow reach into a deep one) | A with one explicit step per exchange | mass: $8.1\times10^{-14}$ (tolerance $10^{-9}$); the full domain: $8.8\times10^{-17}$ of the peak ($10^{-10}$); threaded equal to serial over 6,000 steps | 8.2 s |
> | 4, sound from air into water | C, explicit | reflection: 0.999443938429 against the textbook $R=(Z_2-Z_1)/(Z_2+Z_1)=0.999443938429$, largest departure $3.1\times10^{-10}$ ($10^{-6}$); energy: $8.4\times10^{-16}$ ($10^{-10}$); the two pieces equal the full domain over 2,000 steps | 2.4 s |
> | 6, a steel-and-aluminium bracket under a 5 kN load | B on a structure | forces: $1.3\times10^{-10}$ ($10^{-6}$); displacements: $3.7\times10^{-11}$ of the largest, 0.770 mm ($10^{-6}$); threaded equal to serial | 29.8 s |
> | 7, a heated steel-and-copper strip | split by physics | heat: $9.9\times10^{-11}$ ($10^{-9}$); the synchronous split equals the unsplit solver over 60 steps; the lagged split equals the synchronous one a step late over 60 steps | 1.7 s |
> | 8, a chip on a copper block under a water channel | C at a seam between two physics | energy: $7.0\times10^{-11}$ ($10^{-6}$), 2 kW per metre of depth generated and carried out; the full domain: $2.6\times10^{-10}$ of the 29.5 K rise ($10^{-6}$) | 1.8 s |
>
> - **What each shows that the others do not.**
>   - **The plume** is style A where it is exact. One explicit step needs only the neighbours' old values, so each window takes the full-domain step on its own cells. The wind farm's windows march several sub-steps on one exchange and are not exact (W346: 2–8% in farm power). The page draws the plume on a log scale, three decades deep.
>   - **The sound** is style C with nothing to iterate. Its measure is the reflected pulse's integral, the zero-wavenumber part, which the discrete interface reflects with the textbook $R$ exactly, whatever the dispersion does to the pulse's shape. With air on both sides, the positive control, it reads $R=0$ (a test). The pressure that enters the water is doubled ($T=1+R=1.9994$) while 0.111% of the energy does.
>   - **The bracket** is the first structure. Its Schwarz windows are the nodes of their cells, and it converges slowly: plain Schwarz has no coarse level, so a bending mode takes 616 iterations to $10^{-12}$.
>   - **The strip** is the split by physics. A conduction agent and an elasticity agent share one mesh, and the whole temperature field crosses between them (the bond [[case-study-thermal-strain-atlas-0.1]] found is not a port). Lagging elasticity a step costs $8.3\times10^{-3}$ of the stress at the last step, the same order as that case study's $7.1\times10^{-3}$ on its shell. Uniform heating, the positive control, stresses a bimetal and leaves a plain strip unstressed (a test).
>   - **The cooled block** couples two physics at one seam: the coolant's advection and the block's conduction. Its bulk rise is the closed form of its balance, $P/(\dot m c_p) = 2000/418 = 4.785$ K.
> - **`fe.py` is `ThermoStruct2D`'s element with a material per element.** On one material its stiffness, conduction and mass matrices are the build repo's own to $10^{-12}$, and its thermal load gives that solver's clamped displacement to $10^{-10}$ (a test, which skips without the build repo). The plan named `ThermoStruct2D.solve_mechanical` for case 6, but that solver holds one material for the whole mesh, and the bracket needs two.
> - **Two failed runs, both of the cooled block's reference check**, before the pass above. The tolerance was not changed.
>   - First run: 0.018 of the rise. The "auto" rule gave the Dirichlet side to the lower conductivity (the water), which left the block, insulated all round, as the Neumann side: a floating problem with a singular matrix.
>   - Second run: $1.4\times10^{-3}$. The block was now the Dirichlet side, but 200 iterations were not enough: copper against water puts the 1-D factor at 132.
>   - The fixes: a floating piece now takes the Dirichlet side, and the check refuses a case that makes it the Neumann side (a method fix, pinned by a test). The example allows 2,000 iterations; Aitken converged in 330.
> - **No speedup at these sizes, except the wind farm's.** On 4,000–29,600 cells, one direct solve or one explicit step on the whole grid beat every decomposed arm: by 1.5× (the sound's two pieces) to 520× (the bracket's serial Schwarz). The strip's arms tie within about 10%.
> - **Also built:**
>   - **the circuit layer** for case 5: a schematic of the attachments on the Geometry canvas, and a table with *Add battery* and *Add resistor*;
>   - **regions that feed materials**: a region drawn in a library material brings its properties in the same edit;
>   - **arm names per family**.
>
>   Found in the served page and fixed: electrode nodes cut in half by the canvas's edge, add-part buttons offered to families that ignore a circuit, and the split's panels titled "Decomposed".
> - **Not built:** the optional Gray–Scott case and CS-13/CS-14 adapters (both optional in this plan); the gallery (step 5); the compile (step 6).

> **Built, 2026-09-29: step 6, the compile, and step 5, the gallery.** Code: `atlas/workbench/compile.py`, a `case_graph(spec)` in every family module, and the Check, Results and gallery views in `app.py` and `runview.py`. Tests: in `tests/test_workbench_cases.py`, `test_workbench_shell.py` and `test_workbench_runner.py`. Every example was opened, compiled and run in the served page, and the figures are on [[showcase-gallery]], with the records in `out/workbench/records/step-f/`.
> - **The compile is the loader's graph half** ([[outcome-c4-path-to-declarative-cases]] step 2), built per family rather than through a registry of experts.
>   - Each family turns a case into a `CaseGraph`: one agent per window or piece, a connection wherever a cut face of one opens into another, and the cross-points declared (W162: named, or `()` for none).
>   - `atlas/compiler.py` compiles it in a background thread. The Check step shows the verdict per seam, the worst of the compiler's decisions about that seam, with its rules, and the record goes beside the case file.
>   - Every agent's `boundary_response` is the family's own arithmetic: a window's Dirichlet-to-Neumann map through its own sparse solve, an explicit step's flux, or a structure's reaction.
>   - **One sign was wrong on first contact.** The first draft returned the flow out of the window, and the compiler read a passivity defect of 299 on the wall's seam. The flow *into* the window (the Steklov–Poincaré convention) is what is now declared.
> - **The verdicts, per example:**
>   - `admit-uncertified`: the three farms, the plume, the circuit, the sound and the cooled block.
>   - `refuse` at R10: the wall, the insert and the bracket. Each of their seams is admitted.
>   - Refused before the compiler: the heated strip. Its temperature field is a volumetric bond, and `VOLUMETRIC` is not a port type (`PortAmendment`, as [[case-study-thermal-strain-atlas-0.1]] found).
>   - No case can earn a plain `admit` while the master bound's constants are unmeasured (W56).
> - **R10 is a conservative proxy.** It refuses two embedded agents of one family on a cut region, and it does not read whether the coupling iterates. The three refused cases iterate to agreement and are measured within $3.6\times10^{-9}$ of the full domain or closer. The page shows the refusal as it is, with a note. Whether R10 should read iteration is left to the compiler, and it is open as **W348** in [[gap-worklist]] (opened 2026-09-29). The same compile records derive a direct Schur solve for the interface, and R10, at L2, runs before that scheme exists.
> - **The farms compile through the vault's own graph** (`scaling_ladder.build`, W346's exposed windows): 13, 62 and 124 seams in 7.3 s, 45.3 s and 95.1 s. A drawn tiling that is not a ladder rung is refused before the compiler, with that reason.
> - **Run & compare and Results, finished.**
>   - The Results table carries each arm's seconds per step, $t_{\text{full}}/t_{\text{arm}}$, the family's metrics, the displayed field's rms difference from the full domain, and the checks against their registered tolerances.
>   - The run's JSON record is saved beside the case file.
>   - `NOT_BUILT` is gone: every menu item works, and a test says so.
> - **The gallery:** *File > Showcase gallery* lists every example with its description, family and style, and an *Open* button.
> - **In the gallery pass, every run finished under 2 minutes**, the longest in 70.0 s (`farm-21`, 6 macro-steps on 8 threads). **`farm-21`'s compile takes 95.1 s, so its compile plus its run is 165 s.** Each step is under 2 minutes, and the pair is not. **The owner accepted this on 2026-09-29**: the compile is its own step, and the rule is per run. Caching the farm's compile and shrinking the example were not taken.
> - **Found while building it or in the served page, and fixed:**
>   - two farm rms columns that disagreed, one filled for the threaded arm only;
>   - the Check and Results labels still describing a case no longer open;
>   - the Check label not refreshed when a compile finished with no run in progress;
>   - a compile job with no summary yet crashing the view;
>   - an apostrophe escaped in the gallery's text.
> - **Not built:** the optional Gray–Scott case and the CS-13/CS-14 adapters, so no showcase case carries `ROT`.

---

## 6. The geometry section, planned to fit the library

The library changes what geometry must express. Windows alone are not enough, so the case file moves to **version 0.2** (version 0.1 files still load), with four layers on one canvas, each with an editable table:

| layer | what it holds | needed from |
|---|---|---|
| **Domain** | size and cell size | built |
| **Regions** | rectangles (polygons later) assigned a material or physics | case 2 |
| **Windows** | the decomposition | case 1 |
| **Attachments and boundaries** | rotors; lumped components wired to a region's edge; boundary conditions on the domain edges (inlet, outlet, wall, fixed temperature, fixed potential) | cases 1, 3, 5 |

**Interaction, as proposed (Bokeh, installed):**
- a mode selector (Select/move · Draw window · Draw region · Place device · Set boundary);
- `BoxEditTool` to draw, drag and delete; corner handles from `PointDrawTool` to resize;
- tables kept in sync with the canvas both ways;
- snapping to a selectable step (1, 8 or 16 cells);
- a tiling generator (rows × columns, window size, overlap defaulting to twice the ramp width);
- live warnings on the canvas (uncovered cells, thin overlaps, cross-points);
- everything through the existing undo, redo and save.

**Built on Bokeh in the page, with a Gmsh import path beside it** (the owner chose the hybrid on 2026-09-28; see §8).

**Decisions for the owner** (recommendation first):
1. **Snapping default:** 8 cells, or single cells?
2. **Creating windows:** tiling generator plus hand editing, or free drawing only?
3. **Resizing:** corner handles plus editable table, or table only?
4. **Overlap rule:** warn on the canvas and let Check refuse, or block the edit?
5. **Layers:** all four now (case 2 needs regions), or windows only first?
6. **Which cases first after the wind farm:** 2 and 3, then 5?

> **Built, 2026-09-28** (`atlas/workbench/geometry.py`, `editor.py`, `gmsh_import.py`; schema `atlas-workbench/case@0.2`). The owner asked for the build without answering 1–5 one by one, so **the recommendation was taken in each**, and each is a small change: snapping 8 cells (selectable 1, 8, 16), a tiling generator plus hand editing, corner handles plus an editable table, warnings on the canvas with Check refusing, and all four layers. Decision 6 belongs to step 4 and stays open. Two rules were settled while building:
> - **The overlap rule is a cell rule, not a window rule.** Every covered cell must have one window at full weight, with the weight built exactly as `wake_array.ArrayTiling.weights` builds it (a test checks the two agree cell for cell). Every measured tiling passes it, and touching windows and overlaps under twice the ramp fail it. It is not a pairwise overlap rule: a dense tiling of narrow windows passes it where one pair's overlap alone looks thin, because a third window covers that overlap at full weight (a test pins such a case).
> - **Regions stack in list order**: a later region takes the cells it covers. So a plate with an insert is the plate and then the insert; the alternative (regions may not overlap) would have forced the plate to be cut around every insert.
>
> Built and checked in a served page: move, corner-resize, click-to-add a rotor, the tiling dialog, undo, and a Gmsh upload of a steel plate with a copper disc and an aluminium hexagon. 48 tests. What the section does not do yet: lumped attachments (case 5), and any family but the wind farm can be chosen, so regions and editable boundaries are ready for families that do not exist yet.

---

## 7. What this plan does not decide

- The solver-family interface's exact shape. It is decided by building style A first and generalizing from two real families, not in advance.
- Whether showcase cases get pages of their own in the wiki. The proposal: one row each in a gallery page, with the two checks' numbers.
- 3-D. Out of scope for the showcase.

---

## 8. Existing editors that could replace a hand-built geometry section

The owner's question (2026-09-28): is there an existing 2-D visual editor or CAD tool that could be integrated instead of building from scratch? Surveyed that day, with each tool's licence and whether it is already installed checked.

**The short answer:** many tools draw geometry, and **none knows what a window, an overlap ramp, a seam or a port is.** Whatever draws the shapes, the Atlas-specific part (mapping shapes to layers, snapping to the grid, validating overlaps and cross-points, feeding the compiler) has to be written either way. For rectangles on a Cartesian grid, the drawing is the smaller part.

| tool | licence | installed? | what it gives | what it lacks here |
|---|---|---|---|---|
| **Gmsh 4.15** (desktop GUI + Python API) | GPL-2+; its linking exception covers only Netgen, METIS, OpenCASCADE and ParaView | **yes** | a real geometry kernel (OpenCASCADE: curves, booleans); **named physical groups** for regions and boundaries (checked: `region:steel`, `region:copper`); mesh **partitioning** into subdomains, with an option for ghost (overlap) cells; meshing for body-fitted work later | a separate desktop window, not embeddable in the page; its partitions are graph cuts of an unstructured mesh, not rectangles on a grid. meshio could not read a *partitioned* file (checked), so import goes through Gmsh's own API. GPL matters only if the code is ever distributed |
| **draw.io / diagrams.net** (web, embed mode) | Apache-2.0 | no | a polished editor in an iframe: resize handles, grid snapping, layers, per-shape data, undo; returns XML to the host page | embed mode is officially supported only on `embed.diagrams.net` (online), so offline use means downloading the web app. It is a diagram tool, with no units |
| **Excalidraw** | MIT | no | an embeddable React canvas | freehand whiteboard with no units or precision; needs a React build |
| **tldraw** | proprietary SDK licence (production needs a licence key) | no | a capable React canvas | the licence; excluded |
| **FloorspaceJS** (NREL) | open source (licence not checked) | no | the closest in spirit: a web geometry editor aware of zones and boundary conditions | built for buildings (stories, spaces); adapting it is heavier than building |
| **Any 2-D CAD via DXF** (LibreCAD GPL-2, QCAD, FreeCAD, AutoCAD) + **ezdxf** | ezdxf MIT | ezdxf no; no CAD installed | precise drafting in a familiar tool; layers map to windows, regions, boundaries | a separate app and a re-import after every edit; ezdxf needs installing |
| **SVG from Inkscape** (the geomIO pattern) | Inkscape GPL | no | layers of drawn shapes, parsed as geometry | same round-trip; no units |
| **Bokeh / HoloViews drawing tools** (`BoxEdit`, `PolyDraw`, `PolyEdit`, `PointDraw`) | BSD-3 | **yes** | draw, drag and delete rectangles and polygons in the page, wired straight to Python | resize handles have to be composed; it is building, but from high-level parts |

**Recommendation [AI Inference]: a hybrid, not a single tool.**
1. **In-page editing on Bokeh/HoloViews** for the showcase's geometry: rectangles on a grid. It is the only option that can show Atlas's live warnings (thin overlap, cross-point, uncovered cells) while you draw, and it needs nothing new installed.
2. **A Gmsh import path** for anything curved or non-rectangular: draw in Gmsh's GUI, name physical groups by convention (`region:<material>`, `bc:<kind>`, `window:<id>`), and import into the case file. Gmsh is already installed, it is the one surveyed tool that understands regions and boundaries, and it is also the road to body-fitted grids ([[poc3-racelab-body-fitted-grids]]).
3. **Not a diagram tool** (draw.io, Excalidraw, tldraw): no units, and they need network access, a large download or a licence key.

**Owner's decision (2026-09-28): the hybrid.** In-page Bokeh editing for rectangles on the grid, and a Gmsh import path by physical-group naming for anything else.

---

## See Also

- [[showcase-gallery]] — every case compiled and run in the served page: the verdicts, the checks' measured values, the runtimes
- [[outcome-c4-path-to-declarative-cases]] — the four-step path this extends
- [[outcome-c4-modular-multiphysics]] — every graph built so far
- [[decomposition-speed-by-rotor-count]] — case 1's measured answer
- [[atlas-and-standard-dd-theory]] — where each coupling style sits in classical domain decomposition
- [[port-algebra-atlas-0.1]] — the five port types the cases cover
