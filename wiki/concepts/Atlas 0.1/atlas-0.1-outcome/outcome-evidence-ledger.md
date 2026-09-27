# Outcome — the evidence ledger: every quotable number, with its conditions

**Type:** Outcome page — **reference ledger** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** compiled 2026-09-26 from existing records; no new measurement.
**Hub:** [[00-atlas-0.1-outcome]]

**How to use this page.** A number for a proposal should be taken from here with its **conditions** column attached. Every number was measured on 2-D problems. Timings are ratios measured in one process on one machine unless stated; the vault's rule is that wall-clock numbers do not transfer between machines, and ratios transfer only approximately. Where a later tier changed a number, the current value is given.

---

## 1. The framework

| quantity | value | conditions | source |
|---|---|---|---|
| port types | 5: `MECH`, `ROT`, `THERM`, `ELEC`, `ADVEC` | effort–flow pairs whose product is power | [[port-algebra-atlas-0.1]] |
| coupling code written per new case | 0 lines | a graph compiles from capability records and port connections | [[atlas-implementation]] |
| declarations per join | constant at 3, 4 and 5 families; 0 per-pair | the vehicle union's three joins | [[joining-seam-cost]] |
| largest graph marched | 18 agents, 5 families, 3 joins, 3 clocks | the vehicle union, CS-18 | [[case-study-vehicle-march-atlas-0.1]] |
| largest graph by interface count | 26 agents, 63 interfaces | wind farm $N=8$, classical expert | [[results-n-sweep-wind-farm]] |
| case modules; test files | 36; 86 | `atlas/cases/`, `tests/`, 2026-09-27 | the repository |
| last full suite in the log | 1950 passed, 0 failed | Tier 89, 2026-09-27, 418 s | [[log]] |

## 2. Classical decomposition against the monolith (C1)

| quantity | value | conditions | source |
|---|---|---|---|
| farm power, decomposed vs monolith | $-9.4\%$ (K12), $-18.0\%$ (K25) | classical composed column, unoptimised layout | [[poc1a-frozen-expert-results]] §7.1 |
| composed-defect exponent in interfaces | $+0.851$ [0.748, 0.954]; $+0.160$ with physics fixed | $N = 1 \to 24$ windows, against the monolith | [[case-study-scaling-ladder-atlas-0.1]] |
| cost per step, decomposed / monolith | $0.932$, $1.31$, $1.02$ at 2, 6, 12 windows | same process, interleaved | [[learned-contribution-kill-tests]] §2.3 |
| same comparison on other occasions | $1.54\times$ slower (6 windows); $10$–$25\%$ slower (a Mac); $3.98\times$ faster (K12, exposed classical column, one box) | hardware-dependent | [[atlas-proof-of-concept-1]] §10–§11 · [[poc1-retrospective-and-hybrid-roadmap]] |
| embedded-elliptic decomposed column | leaves its band at step 70–80, not finite by 82 | the monolith holds; exposing the elliptic part fixes it to 120 steps | [[case-study-scaling-ladder-atlas-0.1]] §5 |
| **parallel decomposition vs monolith, by rotor count** | $3.4$–$5.7\times$ faster at 5–21 rotors; $2.1$–$3.7\times$ at 36; slower below 5 | Python threads on the dev laptop, on mains; two timing draws; bit for bit the serial column | [[decomposition-speed-by-rotor-count]] |
| serial decomposition vs monolith, same study | $0.95$–$1.25\times$ the monolith's cost | never faster | same |
| the mechanism | monolith cost per cell $\approx 2 \to 8.7$ µs between 85k and 163k cells | a cache cliff; threads keep window chunks under it | same, §4 |
| farm power, decomposition vs monolith | $-2.3$ to $-8.3\%$ at 1–21 rotors | 40 macro-steps from the freestream | same, §3 |
| local time stepping | $5$–$8\%$ fewer sub-steps; farm power unchanged to the fifth digit | the windows all see near-freestream velocities | same |

## 3. The rocket episode (C1, C4)

| quantity | value | conditions | source |
|---|---|---|---|
| block content vs boundary inflows | $1.2\times10^{-13}$ | every block, every step; mass, momenta, energy | [[case-study-rocket-bc-seam-atlas-0.1]] §19.3, §20.3 |
| mass through walls | $3.5\times10^{-17}$ of $\dot m\,\Delta t$ | after W332 | same, §20.3 |
| thrust over ideal planar theory | $0.992$ | given the nozzle's actual inflow | same, §20.3 |
| injector over declared flow; chamber over declared pressure | $1.000000$; $0.980$ | after W337 | same |
| engine heat, gas lost vs shell gained | $600.35$ vs $600.49$ J/m per side per step | after W339 | same |
| plume seams d\|f, g\|f | $-38\%$, $+23\%$ mass created | W341: one gas model in the plume block; open | same, §20.3 |
| d\|g seam floor | $1.13\%$ | coupled both ways; W344 open | same, §20.8 |
| wall fluxes | double per grid halving | W342: no wall model; open | same, §19.7, §20.4 |
| reproducibility | bitwise in altitude, attitude, $\omega$, mass | Windows/MKL vs Linux/OpenBLAS | same, §20.3 |
| registered gates | 17 of 17 | Tier 86's gates on the current march | same, §20.8 |
| march cost | 2879 s (one-way d–g); 2384 s (two-way) | 29 steps, rented 64-core / 128-thread boxes, coarsened $4\times$ | same, §20.3, §20.8 |

## 4. Learned experts coupled to classical ones (C2)

| quantity | value | conditions | source |
|---|---|---|---|
| momentum budget across a rotor | $r_T = 2.6\times10^{-3}$ (flux BC), $6.2\times10^{-4}$ (halo) | teacher-forced, pre-projection, $5\%$ gate | [[results-w0-w3-wind-farm]] |
| composed frozen column stability | finite 70 of 70 steps; $u_{\max}$ 1.45 / 1.44 | 12 and 24 windows | [[poc1a-frozen-expert-results]] §3 |
| adjoint accuracy | median $1.4\times10^{-3}$ vs central differences | float32 floor | same, §5 |
| defect correction, in sample | 25 classical calls vs the cold march's 47 | 2 windows, $\alpha^\star = 0.5$ | [[defect-correction-learned-operator]] §8.1 |
| defect correction, out of sample | 99 calls vs a gate of $\le 90$; 314 classical-equivalents vs the coarse solver's 52 | 6 windows | same, §8.2 |
| learned content in that slot | 13 classical calls vs a 69-call noise spread | 12 corrupted copies at $3\%$ weight noise | [[corrupted-checkpoint-and-jacobian-fidelity]] |
| what decides the rate | Jacobian fidelity $\rho = +0.83$; accuracy $+0.69$ | rank correlation over 20 cheap maps | same |
| slow-mode response | learned $0.5578$, classical $0.4780$, after the shrink $0.2789$ | settled state, 6 windows | same |
| adversarial cheap map | lands $6.15\times10^{-9}$ from the classical answer | body-fitted car, implicit step | [[poc3-racelab-certified-step]] |
| certified mode price | $1.21\times$ the classical step; $1.83\times$ with verification | body-fitted car, on screen | [[poc3-racelab-certified-screen]] |
| array efficiency $P_2/P_1$ | learned $1.0162$; classical solver in the same graph $0.7663$; 2-D reference $0.404$ | wind farm, $N=2$ | [[results-w6-w11-wind-farm]] |
| verified design gain captured | $71\%$ at both K12 and K25 | learned column's optimum scored by the monolith | [[poc1a-frozen-expert-results]] §7.2 |

## 5. Speed at scale (C3)

| quantity | value | conditions | source |
|---|---|---|---|
| learned / classical cost per window | $4.90$, $3.04$, $1.23$, $0.334$, $0.253$ at $N = 1, 2, 6, 12, 24$ | dev laptop, one process | [[learned-contribution-kill-tests]] §2.1 |
| learned composed step / monolith step | $3.43$, $2.01$, $0.296$ at 2, 6, 12 windows | same process | same, §2.3 |
| $2\times$-coarse classical / monolith | $0.197$, $0.132$, $0.0320$ | same process | same |
| composed learned column vs monolith | $3.15\times$ (K12), $3.64\times$ (K25) faster | dev box, CPU; $0.43\times$–$3.4\times$ across machines | [[poc1a-frozen-expert-results]] §8 · [[atlas-proof-of-concept-1]] §11 |
| gradient cost | $3$–$4.5$ forward evaluations | frozen column; A100 $12.8$–$19.5\times$ the desktop | [[poc1a-frozen-expert-results]] §8 |
| rollouts to design tolerance, gradient vs CMA-ES | $18$ vs $144$ ($8.0\times$); $30$ vs $>928$ ($>30.9\times$) | 36 and 75 design variables | same, §4 |
| learned car at the coupling cadence | $0.067\times$ classical speed; outside envelope 35 of 40 steps | RaceLab porous column | [[case-study-racelab-switch-atlas-0.1]] |
| learned car per call at native lead | $1.15\times$, read as even | two draws differ by $20\%$ | same |

## 6. The specification for a DD-native expert (C5)

| quantity | value | conditions | source |
|---|---|---|---|
| reachable share of the body-fitted car | $\le 33.4\%$ (ceiling); $17.4\%$ (a real tiling) | uniform $128\times128$ windows; body grids hold $61.0\%$ | [[poc3-racelab-car-windows]] |
| error growth below the native step | $4.3\times$ at half the native lead | per-window median | [[case-study-racelab-switch-atlas-0.1]] |
| seam operator beyond any radius | $27$–$75\%$ of its norm, flat from $r = 16$ to $64$ | two scOT checkpoints; a classical operator reads 0 past $r = 13$ | [[epsilon-halo-measurement]] |
| donors taking boundary data as input | 2 of twenty-odd (PDEformer-2, NeuberNet) | survey of public weights | [[expert-donor-survey]] |
| price of light adaptation | $0.35$ h per 2000 iterations | Poseidon-T, forward+backward on 2 windows; not run | [[learned-contribution-kill-tests]] §4.7 |

## 7. Tier 89: the matched shrink and the coarse competitor (C2, C3, C5)

| quantity | value | conditions | source |
|---|---|---|---|
| matched shrink $\alpha$ | $0.143$ (6 windows); $-0.020$ (12 windows) | $1 - \phi_{\text{slow}}/\psi_{\text{slow}}$; negative means the column under-responds | [[matched-shrink-and-coarse-competitor]] §1.1 |
| classical calls, matched vs $\alpha^\star = 0.5$ | $82$ vs $99$ (6); $270$ vs $222$ (12) | one draw each; an $\alpha$ scan at 6 windows swings $38$ calls | same, §1.2 |
| why | weakly shrunk iterations diverge within 1–2 outer iterations from the freestream | the linear rate at $w^\star$ ordered every pair backwards | same, §1.3 |
| coarse competitor, CS-13 | classical calls $7411 \to 62$–$71$; total cost at best $1.11\times$ the cold march | coarse call $1.26\times$ a fine call (solve alone $0.96\times$) | same, §2, §4 |
| coarse competitor, CS-12 | $745$ vs $901$ classical calls; total $1.24\times$ the cold march | coarse call $0.445\times$; coarse settled load within $0.42\%$ | same, §3, §4 |
| seam vs frozen controls | same calls within 4 (CS-12) and 8 (CS-13, $m \ge 400$) | the seam is not what removes the competitor | same, §4 |
| CS-12 settles | residual $2.9\times10^{-3} \to 1.3\times10^{-9}$ over 2000 steps | its case page's "1.5% unsteadiness" was the transient at step 120 | same, §3.1 |

---

## See Also

- [[00-atlas-0.1-outcome]] · [[outcome-c1-decomposition-vs-monolith]] · [[outcome-c2-learned-experts-in-the-loop]] · [[outcome-c3-learned-speed-at-scale]] · [[outcome-c4-modular-multiphysics]] · [[outcome-c5-requirements-for-dd-native-experts]] · [[outcome-demos-and-artifacts]]
