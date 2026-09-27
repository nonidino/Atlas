# The matched shrink (W214) and a coarse competitor across coupled seams (W345)

**Type:** Concept page — **measurement record, two rows** (folder: `Atlas 0.1/common/`)
**Status:** measured 2026-09-26, Tier 89. `scripts/w214_matched_shrink.py` → `out/w214/w214.json`; `scripts/w345_coarse_competitor.py` and `scripts/w345_fsi_competitor.py` → `out/w345/w345.json`, `out/w345/fsi_long.json`; `tests/test_tier89_competitor_and_speed.py`. Predictions were registered in the scripts before any arm ran (times in `out/w214/registered.txt`, `out/w345/registered.txt`). Worklist: **W214** closed, **W345** opened and closed, **W347** opened.
**Related:** [[defect-correction-learned-operator]] · [[corrupted-checkpoint-and-jacobian-fidelity]] · [[learned-contribution-kill-tests]] · [[case-study-cooling-loop-atlas-0.1]] · [[case-study-wing-fsi-atlas-0.1]] · [[case-study-powertrain-atlas-0.1]] · [[outcome-c2-learned-experts-in-the-loop]] · [[outcome-c5-requirements-for-dd-native-experts]] · [[gap-worklist]]

---

## 0. The results, in one paragraph each

**W214.** Tier 50 measured that the composition layer's shrink $\alpha^\star = 0.5$ halves the learned column's slow-band response to $0.2789$, far below the classical map's $0.4780$. It proposed choosing $\alpha$ per rung to match them. **Matched, the rule gives $\alpha = 0.143$ at six windows and $\alpha = -0.020$ at twelve.** At twelve windows the clean column already under-responds, and a shrink can only damp. At six windows the matched arm needs $82$ classical calls against the control's $99$. The scan beside it reads $120$ at $\alpha = 0.25$ and $84$ at $0.35$, so the gain sits inside a $38$-call swing between neighbouring $\alpha$. At twelve windows the rule prescribes no shrink, and the arm needs $270$ against $222$. **The mechanism is not the one the rule was built on.** Every weakly shrunk arm **diverges within one or two outer iterations** from the freestream start, and it finishes on the classical fallback. Only $\alpha = 0.5$ decreases monotonically. Tier 50's linear probe at the settled state predicted the opposite ordering (worst-band rate $0.09$ matched against $0.28$ at $0.5$), and it cannot see the region where these runs fail. **G5 does not move**: $176$ and $294$ classical-equivalents against the coarse solver's $52$ and $38$.

**W345.** Tier 48's cheapest competitor to a learned expert is a $2\times$-coarse classical solver used inside defect correction. The open question since Tier 51 was whether one exists at all across a **coupled** seam. **On the two coupled graphs that have a field to coarsen, it exists and stays correct, and it is not cheap.** Every arm is inside its certificate. On CS-13 (a conduction block on a closed coolant circuit) it cuts classical calls from $7411$ to $62$–$71$. But a coarse call costs $1.26\times$ a fine one there, so it is at best $1.11\times$ **dearer** than the cold march. On CS-12 (a flexible wing in two-way flow) it saves $17\%$ of the classical calls and is $1.24\times$ dearer. **So G5's bar at these seams collapses to the cold classical march, as the brief anticipated, and not for the reason it gave.** The seam-frozen controls cost the same as the coupled arms, so the seam is not the cause. The cases are: a solver whose cost does not fall with its cell count, and a coarse map that stalls the outer iteration. CS-14 has no field, so there is nothing to coarsen. **A premise was wrong on the way.** CS-12 was believed to have no settled state; a 2000-step march shows it converging geometrically to $10^{-9}$. The "1.5% unsteadiness" on its case page was the transient 120 steps in.

---

## 1. W214 — the matched shrink

### 1.1 The rule and the probe

The shrink scales the cheap map, $\Psi_\alpha = (1-\alpha)\,\Psi$, and matching its slow-band response to the classical map's gives

$$(1-\alpha)\,\psi_{\text{slow}} = \phi_{\text{slow}} \quad\Longrightarrow\quad \alpha = 1 - \frac{\phi_{\text{slow}}}{\psi_{\text{slow}}}.$$

The per-band signed Rayleigh quotients come from Tier 50's own perturbations: same bands, $\varepsilon$, draws and seeds. **At six windows they reproduce Tier 50's cache exactly** (difference $0.0$ for all three maps). Twelve windows had never been probed.

| rung | $\phi_{\text{slow}}$ (classical) | $\psi_{\text{slow}}$ (checkpoint) | $\psi_{\text{slow}}$ (no checkpoint, Z) | $\alpha_P$ | $\alpha_Z$ |
|---|---|---|---|---|---|
| 6 windows | $0.4780$ | $0.5578$ | $0.5624$ | $\mathbf{0.1431}$ | $0.1501$ |
| 12 windows | $0.5148$ | $0.5049$ | $0.5159$ | $\mathbf{-0.0196}$ | $0.0022$ |

**At twelve windows the clean column under-responds on the slow band.** Tier 50's overshoot was a six-window reading, and the sign of $\psi_{\text{slow}} - \phi_{\text{slow}}$ changes with the graph. A negative $\alpha$ is an amplification that `shrink` cannot represent. `Rung.arm` applies a shrink only for $\alpha > 0$, so the arm ran at $\alpha = 0$; the record keeps the requested value beside the one applied.

### 1.2 The arms

Classical calls to the certified state (`DEC_SETTINGS` unchanged from Tier 48), with the cost in classical-equivalents $\phi + c_P\,\psi$:

| arm | 6 windows | 12 windows |
|---|---|---|
| cold classical march | $141$ | $267$ |
| P at $\alpha^\star = 0.5$ (control) | $\mathbf{99}$ (reproduces Tier 48) | $\mathbf{222}$ (reproduces Tier 48) |
| **P at matched $\alpha$** | $\mathbf{82}$ ($176$ equiv.) | $\mathbf{270}$ ($294$ equiv.; $\alpha$ clipped to $0$) |
| Z at its matched $\alpha$ | $126$ | $270$ |
| P at $\alpha = 0.25$ / $0.35$ (scan) | $120$ / $84$ | — |
| best corrupted direction (Tier 50) | $71$ | $138$ |
| coarse classical C, $\alpha = 0$ (G5's competitor) | $51.9$ equiv. | $38.1$ equiv. |

Every arm returns a state inside its certificate.

### 1.3 Why: the iteration diverges before it gets near $w^\star$

The outer residuals, from the freestream start ($r_0 = 0.033$; the stopping residual is about $9\times10^{-5}$):

| arm | first outer residuals | stop |
|---|---|---|
| 6 windows, $\alpha = 0.5$ | $0.0332, 0.0240, 0.0201, 0.0178, 0.0164, \dots$ monotone | `max_outer` at 24, then fallback |
| 6 windows, $\alpha = 0.143$ | $0.0332, 0.0180, 0.0137, \mathbf{0.0167, 0.0358}$ | stalled at 4 |
| 6 windows, Z, $\alpha = 0.150$ | $0.0332, 0.0165, \mathbf{0.0214, 0.0680}$ | stalled at 3 |
| 12 windows, $\alpha = 0$ | $0.0308, \mathbf{0.0371, 0.1294}$ | stalled at 2 |

The linear reading at $w^\star$ (Theorem 2's worst band, $\max_b \lambda_b$) put the matched arm at $0.090$ and $\alpha = 0.5$ at $0.276$ at six windows. At twelve it put $\alpha = 0$ at $0.020$ and $0.5$ at $0.351$. **It ordered every pair backwards.** The weakly shrunk iteration amplifies its residual in the first outer iterations, where the state is a freestream nowhere near $w^\star$. The stall rule catches it, and the classical march finishes the job. Tier 48 chose $\alpha^\star = 0.5$ as the one value that converged without falling back, and that was the right criterion for the quantity that binds.

**[AI Inference]:** two consequences follow, neither tested here.
- A shrink **scheduled** on the residual (large far from $w^\star$, relaxing toward the matched value as the residual falls) would test whether the matched rate can be reached once the approach is stable. That is **W347**.
- For [[outcome-c5-requirements-for-dd-native-experts]]: a learned map's Jacobian must be well behaved **along the approach**, not only at the settled state. A probe at $w^\star$ alone is not an admission test.

### 1.4 The registered predictions

| | prediction | got | |
|---|---|---|---|
| M1 | the $\alpha^\star = 0.5$ control reproduces $99$ and $222$ | $99$, $222$ | held |
| M2 | matched beats $0.5$ at both rungs | $82 < 99$; $270 > 222$ | **FAILED** |
| M3 | matched does not beat the best corrupted direction | $82 \ge 71$; $270 \ge 138$ | held |
| M4 | G5 unmoved: matched costs more than the coarse arm | $176 > 52$; $294 > 38$ | held |
| M5 | matched recovers half the 28-call gap at six windows ($\le 85$) | $82$ | held |
| M6 | the checkpoint is not the variable at the matched rule ($\lvert Z - P\rvert \le 13$) | $126$ vs $82$ | **FAILED** |
| M7 | every six-window arm inside its certificate | 5 of 5 | held |

**M6's failure reads in the checkpoint's favour, and it does not stand up alone.** At the matched rule, the no-checkpoint control needs $44$ more calls than the checkpoint, where Tier 50 measured $13$ at $\alpha = 0.5$. But both arms stalled within four iterations, and the scan shows a $38$-call swing between neighbouring $\alpha$. So one pair of single draws cannot say whether the checkpoint's contribution grew. That is W215's point, and it applies to both numbers in the comparison.

**So W214 closes with its question answered and its premise changed.** The composition layer's knob was not the whole story, and the checkpoint was not shown to be the variable. At these rungs **stability during the approach** decides where the classical calls go, and a probe at the settled state does not measure it.

---

## 2. W345 — CS-13, the cooling loop

The block is `thermostruct2d.step_thermal` (backward Euler, conjugate-gradient solve), $48 \times 6$ elements. Its wetted face is coupled through the `PASS` leg to a closed four-leg circuit, whose return temperature is solved in closed form every macro-step.
- **The competitor** restricts the block's nodal field by injection, takes one coarse coupled step on a $24 \times 3$ block with the same circuit, and prolongs bilinearly.
- **The control** freezes the coolant at the settled return temperature. It shares the coupled fixed point, because at the fixed point the return temperature is that value.

| quantity | value |
|---|---|
| reproduction control: the script's map against `LoopSolve.solve`'s loop body, 50 steps | **bitwise** ($0.0$) |
| settled state | $15\,884$ classical steps to the solver's floor; return $341.217771$ K |
| cold classical calls to the stopping rule ($10^{-3}$ of the cold distance) | $\mathbf{7411}$ (frozen control: $5928$) |
| $\Theta$ (from the march) | $1.07\times10^{3}$: a slow thermal mode, time constant about 50 s against a 0.05 s step |
| the competitor's own settled state against the fine one | heat delivered to $4\times10^{-10}$; return temperature to $1.6\times10^{-8}$ K |

| inner march $m$ | coupled C: classical / coarse calls | frozen C0: classical / coarse calls | inside certificate |
|---|---|---|---|
| $40$ | $5107$ / $2460$ | $3649$ / $2460$ | yes |
| $400$ | $\mathbf{71}$ / $9026$ | $63$ / $7750$ | yes |
| $4000$ | $\mathbf{62}$ / $9732$ | $60$ / $13\,481$ | yes |

**Read the bias line with care, because it is exact for a reason.** The coolant presents one uniform temperature along the wetted face, so the settled temperature is uniform along the seam and linear through the thickness. A bilinear element represents that exactly at any resolution. **This is the easy case for a coarse model, and it was measured as the easy case.** A seam carrying a profile (the passage heating along its length) would not be. The circuit's lumped `PASS` leg declares no profile (`CoolantLeg.wall_temperature` is uniform by construction).

**Cost** is §4.

---

## 3. W345 — CS-12, the wing in two-way flow

### 3.1 The premise that was wrong

The first version of this measurement assumed CS-12 had no settled state, citing [[case-study-wing-fsi-atlas-0.1]] §4's *"settled unsteadiness 1.503% peak to peak"*. Its registered floor check (Q6) measured the one-step residual at $2.0$–$4.3\times10^{-4}$ of the field over 40 steps: below the $10^{-3}$ registered, so Q6 **failed** as written. The residual was also **falling**, so the march was extended (`out/w345/fsi_long.json`):

| macro-step | 0 | 200 | 400 | 800 | 1200 | 1800 |
|---|---|---|---|---|---|---|
| one-step residual / field rms | $2.9\times10^{-3}$ | $2.0\times10^{-4}$ | $2.1\times10^{-5}$ | $9.8\times10^{-7}$ | $6.8\times10^{-8}$ | $1.3\times10^{-9}$ |

The load's peak-to-peak, per quarter of the 2000 steps, is $40\%$, $0.17\%$, $6.1\times10^{-3}\%$ and $2.2\times10^{-4}\%$. **CS-12 settles.** Its case page measured the transient at 120 macro-steps and called it the level. That page's gates were quoted against that level, and they still hold against the smaller true one.

### 3.2 The coarse wing

The case's resolution is set by module constants in `ground_effect` and `wing_fsi`. So the competitor loads **private copies of both modules with every resolution constant halved in cells and held in physical units**: cell $1/32$, windows $40$, overlap $8$, ramp $4$, band $4$, padding $16$, leading edge at cell $44$, mount at cell $20$, sample offset $1.25$ cells, force width $0.75$ cells. Each replacement is asserted to match exactly once, and nothing in `atlas/` is edited. **The loader's control:** the same loader with no replacement reproduces the real module's macro-step **bit for bit**. The coarse wing is $72\times104$ against $144\times208$, with the same 33 structural stations.

| quantity | value |
|---|---|
| settled state (fine) | $3495$ macro-steps to the floor; load $0.208017$, tip $-0.026949$ |
| cold classical calls to the stopping rule | $\mathbf{901}$ |
| $\Theta$ | $152$ |
| the coarse wing marched alone: settled load | $0.208885$: $\mathbf{0.42\%}$ above the fine wing's; tip $-0.026996$ ($0.17\%$) |

### 3.3 The arms

The seam-frozen control holds the deflection at the settled one (`motion=False`). Its own fixed point lies $1.05\times10^{-10}$ from the coupled one's fluid, and its cold march needs the same $901$ calls.

| inner march $m$ | coupled C: classical / coarse calls | frozen C0: classical / coarse calls | stop | inside certificate |
|---|---|---|---|---|
| $40$ | $745$ / $844$ | $749$ / $829$ | `max_outer` at 59, then fallback | yes |
| $400$ | $745$ / $900$ | $746$ / $890$ | `max_outer` at 59, then fallback | yes |

The corrected iteration makes slow progress for 60 outer iterations, and the classical march finishes. It saves $156$ of the cold march's $901$ classical calls.

---

## 4. W345 — what the competitor costs, and the verdict

Per-call cost, interleaved in one process, the minimum over five repeats of the mean:

| | CS-13 cooling loop | CS-12 wing |
|---|---|---|
| a classical call $\Phi$ | $0.66$ ms | $221$ ms |
| **a coarse call over a classical call** | $\mathbf{1.255}$ | $\mathbf{0.445}$ |
| of which: the coarse solve over the fine solve | $0.962$ | — |

**On CS-13 the coarse solve costs what the fine one does.** At 343 nodes `step_thermal` spends its time assembling the Robin matrices and setting up the conjugate-gradient solve, not on cells. The remaining $0.29$ is this script's prolongation ($114\ \mu$s of Python interpolation), which could be made nearly free and would not change the verdict.

**The competitor, in classical-equivalents** ($\phi + c_C\,\psi$), against the cold march:

| | cold march | best competitor arm | ratio |
|---|---|---|---|
| CS-13 | $7411$ | $5107 + 1.255 \times 2460 = 8194$ ($m = 40$); $71 + 1.255 \times 9026 = 11\,399$ ($m = 400$) | $\mathbf{1.11\times}$ **dearer** at best |
| CS-12 | $901$ | $745 + 0.445 \times 844 = 1121$ ($m = 40$) | $\mathbf{1.24\times}$ **dearer** |
| *wake array, Tier 48 (single family, for contrast)* | *$141$* | *$51.9$* | *$2.7\times$ cheaper* |

**The answer to the Tier 51 brief's question.** On both coupled 2-D graphs that have a field to coarsen, the coarse competitor **exists and is correct**. Theorem 1 holds under measurement, and every arm is inside its certificate. It is **not cheap**: counting its own cost it loses to the cold classical march. So **G5's bar at these seams collapses to the cold march itself**, as the brief anticipated. But the brief's mechanism was wrong: **the seam is not the cause.**
- **The seam costs nothing:** the frozen controls behave identically. CS-12's coupled and frozen arms differ by at most 4 calls; CS-13's differ by at most 8 at $m \ge 400$.
- **What removes the competitor is the case:** a solver whose cost does not fall with its cell count (CS-13), and a coarse map that slows the outer iteration more than it saves (CS-12).

The bar a learned expert must clear there is $7411$ and $901$ classical calls, not $52$.

**CS-14 has nothing to coarsen.** Every agent is lumped algebra, and one full circuit solve costs $0.33$ ms. The cheapest classical alternative is the solve.

**[AI Inference]:** the easy-case caveat of §2 cuts the other way for cost. A larger block, or a seam carrying a profile, would make a coarse solve cheaper per call than a fine one, and could bring the competitor back. **"The coarse competitor collapses at a coupled seam" is measured at these sizes and is not a law.**

### 4.1 The registered predictions

| | prediction | got | |
|---|---|---|---|
| Q1 | the classical map reproduces `LoopSolve` bit for bit | $0.0$ | held |
| Q2 | CS-13's competitor stays correct at every $m$ | 3 of 3 inside | held |
| Q3 | its own settled heat within 1% | $4\times10^{-10}$ | held, and exact for the reason in §2 |
| Q4 | coupled no more than $1.5\times$ the frozen control's classical calls | $5107/3649 = 1.40$; $1.13$; $1.03$ | held |
| Q5 | a coarse call costs more than $0.25$ of a classical one | $1.255$ | held |
| Q6 | CS-12's one-step residual stays above $10^{-3}$ over 40 steps | $2.0\times10^{-4}$ | **FAILED**, and the premise with it (§3.1) |
| F0 | the unpatched loader copy reproduces `wing_fsi` bit for bit | yes | held |
| F1 | the coarse wing's settled load within 10% | $0.42\%$ | held |
| F2 | CS-12's competitor stays correct | 2 of 2 inside | held |
| F3 | coupled no more than $1.5\times$ frozen | $745/749$, $745/746$ | held |
| F4 | a coarse wing call costs at most $0.35$ of a fine one | $0.445$ | **FAILED** |

---

## 5. What this tier did NOT do, named

- **No scheduled shrink** (W347). §1.3's inference, that the matched rate is reachable once the approach is stable, is untested.
- **No second draw of any W214 arm.** The $38$-call swing across the $\alpha$ scan is the only noise estimate, and W215 (randomised arms scored as distributions) is still open.
- **No efficient transfer operators for CS-13**, and no larger block. §4's cost verdict is at the case's declared size.
- **No coarse competitor on CS-18's joined union**, where the rotor's fluid meets the powertrain and the radiator core meets the coolant.
- **CS-12's case page is not rewritten.** Its settled numbers were taken 120 steps into a transient that ends near step 2000. §3.1 is the correction, and the page's gates are unaffected.

---

## See Also

- [[defect-correction-learned-operator]] — the slot, the gate G5 belongs to, and §8's arms this reproduces
- [[corrupted-checkpoint-and-jacobian-fidelity]] — the probe W214 reuses, and the six-window reading it generalised from
- [[outcome-c2-learned-experts-in-the-loop]] · [[outcome-c3-learned-speed-at-scale]] · [[outcome-c5-requirements-for-dd-native-experts]]
- [[gap-worklist]] — W214, W345, W347
