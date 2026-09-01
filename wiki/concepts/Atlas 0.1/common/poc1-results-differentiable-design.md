# PoC 1a — Results: Differentiable Wind-Farm Design

**Type:** Concept page — **results record** for [[atlas-proof-of-concept-1]] (folder: `Atlas 0.1/common/`)
**Status:** run 2026-09-01. Every number below is quoted from `out/w111/w111.json`; nothing is estimated. This is a **capability demonstration**, not a rung of the [[case-study-ladder-to-f1]] ladder.
**Code:** `atlas/cases/wind_farm_design.py`, `scripts/w111_wind_farm_design.py`, `tests/test_tier21_wind_farm_design.py`
**Related:** [[atlas-proof-of-concept-1]] · [[prior-art-and-novelty-atlas-0.1]] · [[case-study-wake-array-atlas-0.1]] · [[case-study-scaling-ladder-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[composition-error-theory]] · [[open-problems-atlas-0.1]]

---

## 0. Read this first: what was and was not exercised

The spec's headline sentence is about *frozen, independently-pretrained* experts. **This run does not exercise that half.** Every rollout used `wake_array.exposed_reference_solver` — `reference.WindowNS` with its elliptic part removed — as the fluid agent in every window, because that is the only arrangement of the classical column that survives a long march ([[case-study-scaling-ladder-atlas-0.1]] §5, W100) and because it is what the build brief specified.

So what is measured here is:

> **gradient-based design optimisation through a spatially-partitioned graph of independently-declared experts — twelve to twenty-four fluid windows coupled at overlapping seams, plus one zero-parameter lumped actuator-disk expert per rotor — with the adjoint taken through every seam and every disk closure.**

The field/lumped heterogeneity ([[prior-art-and-novelty-atlas-0.1]] §2.3) *is* exercised, and so is the partitioning. What is **not** exercised is the frozen-checkpoint half of §2.1. Poseidon-T is a PyTorch module and is differentiable, so the substitution is a `kind="poseidon"` swap rather than new machinery — but it has not been run, and no sentence on this page should be read as if it had.

Two further scope statements, stated before any number:

- **This is not accuracy.** $\mathrm{Re}_D = 255$ at the checkpoint-derived grid-scale viscosity, the wake physics is the reference solver's, and no column here is validated against data. Every power figure is quoted **"as measured by the composed model"**, which is [[composition-error-theory]]'s standing position: the composed model is a search instrument, not a verifier.
- **The optimum sits on its constraints.** At both sizes the minimum-spacing constraint is *active* ($\min_{jk} d_{jk} = 2.000\,D$ exactly, against a $2.0\,D$ limit) and turbines reach the domain box. The optimised layout is therefore partly a statement about the box that was drawn, and the box is declared in `FarmCase`.

---

## 1. The setup

| | K = 12 | K = 25 |
|---|---|---|
| ladder rung | N12 — $4\times3$ windows | N24 — $6\times4$ windows |
| domain | $464\times352$ cells $=14.5\times11.0\,D$ | $688\times464$ cells $=21.5\times14.5\,D$ |
| turbines | $4$ streamwise $\times\ 3$ lateral | $5\times5$ |
| design vector $\theta$ | $(x_k,y_k,\gamma_k)$, **36** variables | **75** variables |
| macro-steps per rollout | 50 | 40 |

Held at CS-7's values throughout: $\mathrm{d}x = D/32$, $\Delta t = 0.2$, $16$-cell overlap, $8$-cell partition-of-unity ramp, $\nu = 3.92\times10^{-3}$. The assembly is `wake_array.projected_assembly` — an `assembly.ProjectedAssembly`, i.e. the blend **and** one global Leray projection on the assembled field, once, after the blend.

**The objective.** $J(\theta) = \big\langle \sum_k P_k \big\rangle_{\text{last 5 macro-steps}} - \lambda \sum_{j<k}\big[\max(0,\,s_{\min}-d_{jk})\big]^2$, with $P_k = T_k U_{n,k}$ read from each disk's `ROT` port, $T_k = \tfrac12\rho A C_T' U_{n,k}^2$ and $C_T' = 4a/(1-a) = 2$ — `disk.ActuatorDisk`'s own closure, zero fitted parameters.

**Yaw is a closure and this page says so once more.** A yawed disk responds to the axis-normal inflow $U_n = u\cos\gamma + v\sin\gamma$ and pushes back along its own axis; the lateral component $-T\sin\gamma$ is what steers the wake. Nothing in `WindowNS` knows about a yawed rotor. The $\cos^3\gamma$ power law is a *consequence* of the rotation rather than an imposed law (asserted to $10^{-9}$ in the tests), but **the deflection itself is a model**, exactly as FLORIS's is.

---

## 2. Provenance — is this the column CS-7 and CS-8 marched?

Yes, and to the bit. The PoC runs in torch (and on CUDA) rather than numpy, so this had to be established rather than assumed:

| check | result |
|---|---|
| torch agent vs `wake_array.exposed_reference_solver`, real `step_batch` **on CPU** | $\max\lvert\Delta u\rvert = \mathbf{0.0}$ — **bitwise** |
| the same **on CUDA** | $\max\lvert\Delta u\rvert = \mathbf{0.0}$ — **bitwise** |
| torch blend + global Leray vs `wake_array.assemble_conservative` through the declared `ProjectedAssembly` | $2.2\times10^{-16}$ |
| $J(\theta)$ evaluated twice, same $\theta$ | **bit-identical**, both devices |
| $J(\theta)$ on A100 vs on CPU | K12 **exactly equal**; K25 $3.6\times10^{-16}$ relative |
| K25 confirmation march $u_{\max}$, CPU vs A100 | $1.5192356689116893$ vs $1.5192356689116908$ |

**One thing had to change to get there, and it is worth recording.** `Tensor.scatter_add` — the obvious way to accumulate windows into the global field — is implemented with `atomicAdd` on CUDA, so the summation order of overlapping contributions varies between runs and $J$ stops being reproducible. A finite difference of a function that moves in its last digits measures the noise, and CMA-ES ranking a population by such a number is a different algorithm. The assembly now accumulates in a **fixed order** (`_place`), which is deterministic on both devices and is the operational content of OP-6's *"pin the batch layout"*.

### 2.1 The dependency the PoC was blocked on

**N24 had never been marched past 20 macro-steps.** It has now: 60 macro-steps at the N24 rung with all 25 disks live, exposed agents plus the projected assembly.

> `stable = True`, finite to $60/60$, $u_{\max} = 1.5192$ against a band of $3.0$, assembled divergence $1.30\times10^{-2}\to5.11\times10^{-2}$ with a **developed-tail trend of $0.963$** (flat-to-falling).

The optimiser's 40-step horizon at K=25 sits well inside that.

---

## 3. The headline number

**The tolerance rule.** $\varepsilon$-target $= J_{\text{grad}} - 0.02\,(J_{\text{grad}} - J_0)$: within 2 % of the gradient method's *own* improvement over the unoptimised grid. Both methods start from the identical layout, optimise the identical objective, and get the identical constraint handling — the box clip **and** the spacing projection, so no part of the ratio is a comparison of constraint machinery.

| | K = 12 (36 vars) | K = 25 (75 vars) |
|---|---|---|
| $J$: start $\to$ gradient optimum | $2.3142 \to \mathbf{9.6080}$ | $6.4379 \to \mathbf{19.5199}$ |
| farm power gain, *as measured by the composed model* | $\mathbf{+315\,\%}$ | $\mathbf{+203\,\%}$ |
| **gradient rollouts to tolerance** | $\mathbf{26}$ | $\mathbf{24}$ |
| CMA-ES evaluations spent | $700$ (50 generations, popsize 14) | $928$ (58 generations, popsize 16) |
| CMA-ES best $J$ reached | $8.011$ — **never reached the tolerance** | $16.642$ — **never reached the tolerance** |
| **ratio, rollouts** | $\mathbf{> 26.9\times}$ | $\mathbf{> 38.7\times}$ |
| coordinate finite difference, $2n+1$ per gradient | $\mathbf{73\times}$ per step $\to 1898$ rollouts | $\mathbf{151\times}$ per step $\to 3624$ rollouts |

Because CMA-ES never reached the target inside its budget, **both ratios are lower bounds.** Read at intermediate tolerances, where CMA-ES did arrive, the comparison is measured rather than bounded:

| fraction of the gradient's gain | K12 gradient | K12 CMA-ES | ratio | K25 gradient | K25 CMA-ES | ratio |
|---|---|---|---|---|---|---|
| 50 % | 8 | 33 | $4.1\times$ | 9 | 322 | $\mathbf{35.8\times}$ |
| 75 % | 11 | 197 | $17.9\times$ | 13 | 628 | $\mathbf{48.3\times}$ |
| 90 % | 22 | — | $>31.8\times$ | 18 | — | $>51.6\times$ |
| 100 % | 30 | — | $>23.3\times$ | 25 | — | $>37.1\times$ |

**The ratio grows with the dimension of the design vector, and that is the result that matters.** From 36 variables to 75 the measured ratio at 75 % of the gain goes $17.9\times \to 48.3\times$, and the coordinate-finite-difference ratio goes $73\times \to 151\times$ *by construction*, because it is $2n+1$. An adjoint costs what it costs regardless of $n$; every derivative-free method pays in $n$. The spec's "two to three orders of magnitude" is **not** reached at these sizes — one to nearly two is — but the trend that would take it there is measured, and it is the dimension.

### 3.1 What one gradient costs

Measured, not assumed. The rollout is checkpointed one macro-step at a time (`torch.utils.checkpoint`), because the un-checkpointed tape for a 50-step rollout at 12 windows is $\sim\!10^2$ GB and neither machine has that. Checkpointing pays two forwards plus one backward:

| | forward rollout | gradient rollout | gradient in forward-equivalents |
|---|---|---|---|
| commodity desktop, 22 cores, **no GPU** — K12 | $21.5$ s | $140$ s | $6.4\times$ |
| the same — K25 | $21.2$ s | $116$ s | $5.5\times$ |
| one rented **A100 SXM4** — K12 | $4.34$ s | $38.2$ s | $8.8\times$ |
| the same — K25 | $3.61$ s | $21.5$ s | $5.9\times$ |

So **a gradient costs six to nine forward evaluations**, against $73$ and $151$ for a coordinate finite difference — and that factor is the honest per-step version of the headline. The A100 is $5$–$6\times$ the desktop, not the $50\times$ a bandwidth argument would predict, because a $[24,128,128]$ float64 stencil is bound by per-operation overhead rather than by memory bandwidth. The whole GPU experiment — both gradient runs, both CMA-ES runs, the confirmation march, the ablation, the finite-difference check and the sensitivity sweep — cost **\$0.69 and 55 minutes of wall-clock**.

The desktop CPU figures are quoted from **three interleaved repeats agreeing within 5 %**. An earlier single reading of $1.01$ s per macro-step was contaminated by concurrent load and is superseded.

---

## 4. The visual

`out/w111/fields_K12.png`, `out/w111/fields_K25.png` — before/after streamwise velocity for the whole farm on a **shared colour ramp**, rotor chords drawn to scale with a white axis whisker showing yaw. `out/w111/traces_K12.png`, `out/w111/traces_K25.png` — the farm-power trace over optimiser steps, the evaluations-to-tolerance comparison on a log axis, and the yaw angles at the optimum.

What the field pair shows, in one sentence: **the unoptimised grid is three (or five) continuous wake corridors with every downstream turbine sitting in the shadow of the one in front, and the optimised layout is a stagger in which almost every rotor has found clean inflow.**

One defect in the artefact, stated rather than left for a reader to notice: the third trace panel shows the yaw angles **at** the optimum rather than converging to it. `OptTrace.as_dict` serialised only the endpoints of the design trajectory, and although the record was written after every optimiser step, each write overwrote the file with a record that kept no history — so the trajectory existed at no point in time. The serialisation is fixed in the code; the panel will be a convergence plot on the next run of it.

---

## 5. Is the gradient right? (OP-6)

The check the spec asks for: the adjoint against central differences on a real composed rollout, at 9 components spanning all three variable kinds, with the batch layout pinned.

| $h$ (rotor diameters) | median relative error | cosine |
|---|---|---|
| $10^{-2}$ | $7.13\times10^{-4}$ | $0.9999407$ |
| $10^{-3}$ | $4.16\times10^{-6}$ | $0.99999999$ |
| $10^{-4}$ | $4.18\times10^{-8}$ | $1.0000000000$ |
| $\mathbf{10^{-5}}$ | $\mathbf{5.85\times10^{-9}}$ | $1.0000000000$ |
| $10^{-6}$ | $4.04\times10^{-8}$ | $1.0000000000$ |

A textbook finite-difference error curve with a minimum at $h \approx 10^{-5}$: truncation falling as $h^2$ above it, cancellation rising below it. **The adjoint and the differences agree to nine significant figures**, and $J$ is bit-reproducible, so there is no probe floor to fight — OP-6's $\sim\!10^{-6}$ concern is a *float32 checkpoint* phenomenon and this column does not have it.

**One component reads a relative error of exactly 1.0 and it is the most informative entry in the table.** Turbine 6's yaw derivative is $-1.04\times10^{-16}$ by adjoint and $-1.11\times10^{-10}$ by difference: both are numerically zero. Turbine 6 sits in the **centre row** of the $4\times3$ layout, where the array is laterally symmetric and yaw has no preferred sign. The instrument is dividing two zeros.

**And the check caught a real defect before it caught anything else.** The disk's smearing thickness $\Delta_d = \langle U_d\rangle\,\Delta t$ was initially `detach()`-ed as "a statement about the discretization". The finite difference immediately reported a systematic $0.2\,\%$ deficit in *every* component — the exact size of the path that had been cut. $\Delta_d$ is CS-7's own derivation from momentum theory, so it is part of the model and the adjoint carries it. That is what the check is for, and it is on the record because it found something.

---

## 6. Is the *coupling* doing the work? — the ablation

The claim is not "we have a gradient"; a static wake surrogate has one of those. It is that the gradient goes **through the composed rollout**. So the same objective was differentiated twice at the same design point: once with the march live (the full adjoint), and once with the march frozen and only the disks' own closure differentiated on the states the live march produced.

| | $\lVert g_{\text{full}}\rVert$ | $\lVert g_{\text{frozen}}\rVert$ | cosine | sign agreement |
|---|---|---|---|---|
| **K12**, all | $0.1064$ | $1.1304$ | $\mathbf{0.238}$ | — |
| — streamwise $x$ | $0.0764$ | $1.1266$ | $0.294$ | $\mathbf{50\,\%}$ |
| — lateral $y$ | $0.0716$ | $0.0802$ | $0.473$ | $75\,\%$ |
| — yaw $\gamma$ | $0.0193$ | $0.0462$ | $0.754$ | $83\,\%$ |
| **K25**, all | $0.3650$ | $1.5750$ | $\mathbf{0.054}$ | — |
| — streamwise $x$ | $0.2385$ | $1.5100$ | $\mathbf{-0.115}$ | $\mathbf{40\,\%}$ |
| — lateral $y$ | $0.2525$ | $0.4094$ | $0.513$ | $84\,\%$ |
| — yaw $\gamma$ | $0.1121$ | $0.1810$ | $\mathbf{0.953}$ | $68\,\%$ |

**The frozen-field gradient is three to eleven times larger and nearly orthogonal to the true one.** At $K=25$ its streamwise component is *anti*-correlated with the truth ($\cos = -0.115$) and its signs agree $40\,\%$ of the time — worse than a coin flip. The reason is legible: on a frozen field the only way to raise a turbine's power is to move it upstream into faster flow, which is a large and consistent signal; the true derivative knows that moving it upstream deepens the wake it casts on everything behind it, and the two nearly cancel.

**Yaw is the transferable component** ($\cos = 0.95$ at K25), because a yawed rotor's own $\cos^3\gamma$ loss is local and dominates. So a static wake model would get yaw roughly right and **site turbines in the wrong direction** — which is precisely the case for putting the seam inside the differentiated path, and it is measured rather than argued.

---

## 7. How converged is any of this?

**Not very, and the spec said so in advance.** Three numbers:

- **Quasi-steady, not converged.** The relative change in farm power over the last five macro-steps at the optimum is $1.9\times10^{-2}$ (K12, 50 steps) and $3.0\times10^{-2}$ (K25, 40 steps). The objective is a five-step time average precisely because a single snapshot of this state is not a fixed point.
- **March-length sensitivity**, K12, at the same two layouts:

  | macro-steps | 30 | 40 | 50 | 60 | 70 |
  |---|---|---|---|---|---|
  | gain | $+236\,\%$ | $+331\,\%$ | $+315\,\%$ | $+329\,\%$ | $+336\,\%$ |
  | settling at the optimum | $4.2\times10^{-2}$ | $9.7\times10^{-3}$ | $1.9\times10^{-2}$ | $7.6\times10^{-3}$ | $5.1\times10^{-3}$ |

  From 40 macro-steps on, the gain is stable within its own transient scatter; **at 30 it is 24 % lower**, so the horizon is load-bearing below 40 and the 40–60 band the spec names is the right one. K25's 40 is the low end of that band and was chosen for cost.
- **Neither optimiser converged.** The gradient traces were still rising when their fixed budgets ran out (last increments $+0.13$ at K12, $+0.24$ at K25). Because the tolerance is defined from the gradient method's *own* final $J$, a longer gradient run would raise the bar and the ratios would grow, not shrink — so the reported ratios are conservative in that direction too.

**Where the gain comes from, stated plainly.** $+315\,\%$ is a large number because the *starting* layout is pathological: four in-line turbines at $2.83\,D$ in a slowly-recovering $\mathrm{Re}_D = 255$ wake produce a farm power of $2.31$ against an unwaked ideal of $12.0$ — $19\,\%$ of ideal. The optimiser is escaping a bad initial condition, not finding $315\,\%$ that a real farm has on the table. **The defensible claim is the ratio and the wall-clock, not the percentage**, which is exactly what [[atlas-proof-of-concept-1]] §4 committed to in advance.

---

## 8. What the optimiser actually did

| | K = 12 | K = 25 |
|---|---|---|
| min spacing at the optimum | $2.000\,D$ — constraint **active** | $2.000\,D$ — **active** |
| yaw, mean $\lvert\gamma\rvert$ | $3.9^\circ$ | $6.3^\circ$ |
| yaw range | $-5.8^\circ$ to $+8.1^\circ$ | $-16.6^\circ$ to $+11.9^\circ$ |
| yaw at the $\pm30^\circ$ bound | $0/12$ | $0/25$ |
| position range reached | $x\in[1.50,10.51]$, $y\in[1.50,9.50]$ | $x\in[2.33,17.18]$, $y\in[1.50,13.00]$ |

**Re-siting dominates wake steering here.** No turbine reached the yaw bound, and mean yaw is under $7^\circ$, while positions moved to the box edges and packed to the spacing limit. That is a property of *this* problem — the box gives several diameters of lateral freedom, which is cheaper than yawing — and it should not be read as a general statement about wake steering, where the layout is usually fixed and yaw is the only free variable. The stronger test of the yaw half would fix positions and optimise $\gamma$ alone.

---

## 9. What this does not claim

- **Not the frozen-checkpoint result.** §0. The fluid agent is the classical reference solver.
- **Not accuracy.** Not validated against data or against a classical stack, at either size.
- **Not a scaling result and not multiphysics.** One governing family; the ladder's F1 criterion is [[case-study-scaling-ladder-atlas-0.1]]'s business, not this page's.
- **Not two to three orders of magnitude.** One to nearly two, at 36 and 75 design variables, with the dimensional trend that points at more.
- **Not a converged optimum**, at either size.

What it does claim, and this is the whole of it: **reverse-mode differentiation of a design objective back through a 40–50 macro-step rollout of a spatially-partitioned expert graph — across every fluid–fluid seam and every fluid–disk closure — is implementable, is correct to nine significant figures against finite differences, costs six to nine forward evaluations, and finds in 24–26 rollouts a layout that CMA-ES had not reached in 700–928.**

---

## 10. What this opens

Candidates for [[gap-worklist]] rather than rows added here, since the tier structure belongs to the case-study ladder and this is not on it:

1. **The frozen-checkpoint run.** `kind="poseidon"` through the same optimiser. It is a swap, and until it is run §0 stands. It also puts OP-3's over-dissipation and OP-6's float32 probe floor back in play, both of which this column does not have.
2. **The constraint set is doing visible work.** Both optima sit on the spacing bound and the box. A result whose optimiser stops at its constraints is partly a result about the constraints.
3. **Yaw alone, positions fixed** — the experiment that would actually test the deflection closure, which this run's re-siting freedom made unnecessary.
4. **The ablation's $\cos = -0.115$ deserves its own measurement.** That a static-field gradient is *anti*-correlated with the true one in the streamwise direction is the sharpest single argument in this vault for differentiating through the coupling, and it rests on one design point at one rung.
5. **PoC 1b, the stretch of [[atlas-proof-of-concept-1]] §7**: connect the open `ROT` ports to a lumped drivetrain and co-design $(\theta,\phi)$. The ports are already open and already carry $\tau\omega = P$ (asserted in the tests).

---

## See Also

- [[atlas-proof-of-concept-1]] — the specification this executes; §9 there carries the summary
- [[prior-art-and-novelty-atlas-0.1]] — §1's differentiability-as-value-proposition, §2.3's field + lumped peers; §2.1 is the half **not** exercised
- [[case-study-scaling-ladder-atlas-0.1]] — §5's projected assembly, without which none of these rollouts is finite
- [[case-study-wake-array-atlas-0.1]] — the graph and the geometry this optimises over
- [[port-algebra-atlas-0.1]] — §5.1's open `ROT` port, which is the objective read here
- [[composition-error-theory]] — why a composed model is a search instrument and not a verifier
- [[open-problems-atlas-0.1]] — OP-3, OP-6
