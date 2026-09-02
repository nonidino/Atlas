# Atlas Proof of Concept 1 — Differentiable Wind-Farm Design

**Type:** Concept page — proof-of-concept specification (folder: `Atlas 0.1/common/`)
**Status:** opened 2026-08-31. A scoped demonstration target, not a case study on the [[case-study-ladder-to-f1]] ladder — its purpose is external: one headline number and one headline visual that are *both* genuinely novel and legible to a non-specialist.
**Related:** [[prior-art-and-novelty-atlas-0.1]] · [[case-study-wake-array-atlas-0.1]] · [[case-study-scaling-ladder-atlas-0.1]] · [[port-algebra-atlas-0.1]] · [[composition-error-theory]] · [[f1-pathmap-and-end-goal]] · [[general-coupling-scheme]] · [[expert-donor-survey]]

---

## 1. The claim, in one sentence

> **Re-optimize a wind-farm layout — turbine positions and yaw angles — for total power in seconds on one machine, using exact gradients taken straight through a graph of frozen, independently-pretrained physics experts coupled at typed interfaces, and beat a derivative-free search by two to three orders of magnitude in evaluations.**

Everything after this section is how, why it is defensible, and how it fails.

## 2. Why this is the right first PoC

**It spends the one capability that is actually unique.** [[prior-art-and-novelty-atlas-0.1]] §2 is candid that almost every *mechanism* in Atlas has been done before — domain decomposition since 1870, co-simulation, flux-matching interfaces (cPINN), expert libraries (CompNO). The three things it lists as genuinely open all bear on one property: **end-to-end differentiability through a composed graph of frozen heterogeneous experts.** That same page also notes the adjoint this implies "is claimed as a headline feature and used for nothing." This PoC uses it.

**Nobody has done design optimization through a spatially-partitioned graph of independently-pretrained frozen experts.** CompNO composes operators *functionally* inside one domain; classical shape optimization uses a monolithic adjoint of a single solver; co-simulation frameworks (preCICE, System Coupling) generally break the gradient chain at the coupling. Optimizing *across* a multi-expert coupled graph, with the fluid–turbine seams inside the differentiated path, is unpublished.

**The domain is funded and the demo is visceral.** Wake steering and layout optimization are active industrial problems (NREL FLORIS, Ørsted, Vestas, DTU). The visual — wakes visibly re-curving as an optimizer runs, a farm-power number climbing — needs no explanation. Run it interactively (CS-7 clocked $86\ \text{ms}$ per macro-step at $N{=}24$, i.e. $\sim\!12\ \text{Hz}$) and a viewer can drag a turbine and watch the coupled field respond.

**It needs no new experts.** The wake-array graph ([[case-study-wake-array-atlas-0.1]]) already exists: a frozen Poseidon-T fluid expert per window, zero-parameter actuator disks as lumped experts, `MECH` and `ADVEC` ports at the seams, `ROT` ports open at the rotors. It is already a PyTorch graph and already differentiable. The only hard dependency is the seam-divergence fix (§6).

## 3. What gets built

1. **A design vector.** $\theta = (\{x_k, y_k\}, \{\gamma_k\})$ — turbine positions and yaw angles for $k = 1 \dots K$, $K \in \{12, 25\}$ (the CS-7 ladder's $N{=}24$ tiling and one larger). Positions enter the graph through the disk-expert placement; yaw enters through the actuator-disk thrust/deflection model.
2. **A composed forward pass** to a quasi-steady state: march the coupled fluid + disk graph $\sim\!40$–$60$ macro-steps with the **projected assembly** of §6 so the rollout is stable.
3. **An objective.** Total array power $J(\theta) = \sum_k P_k$ read from the disk experts' `ROT` port power, with an optional wake-loss regulariser and a minimum-spacing penalty.
4. **The gradient.** $\nabla_\theta J$ by reverse-mode autograd through the *entire* rollout, including every fluid–disk seam. This is the artefact that does not exist elsewhere.
5. **The optimiser.** Adam or L-BFGS on $\theta$, projected onto the domain box and spacing constraints, $\sim\!30$–$100$ steps.
6. **The baseline.** The identical objective optimised by a derivative-free method — CMA-ES and/or coordinate finite-difference — from the same start, to the same tolerance, counting forward evaluations.

## 4. The headline results

**One number.** Forward evaluations (equivalently wall-clock) to reach a layout within $\varepsilon$ of the gradient method's optimum:

| method | evaluations to optimum | wall-clock (1 machine) |
|---|---|---|
| gradient through the composed graph | $\sim\!K$ per step $\times\ 30$–$100$ steps | target: **seconds to a few minutes** |
| CMA-ES / finite-difference | $10^2$–$10^4\times$ more | minutes to hours |
| classical coupled-CFD adjoint | one gradient = one HPC job | hours, and not set up for this coupling |

The defensible statement is the **ratio** and the **wall-clock on commodity hardware**, not an absolute FLOP count.

**One visual.** Before/after streamwise-velocity fields for the full farm on a shared colour ramp, wakes re-steering between the two; the farm-power trace rising over optimiser steps; the yaw angles converging. Interactive version: drag a turbine, the coupled field and the power readout update live.

**One power figure, quoted carefully.** "$+X\%$ total power over an unoptimised grid layout, *as measured by the composed model*." Verified against a classical reference on the final shortlist (§5), not claimed as ground truth.

## 5. How it fails, and the honest framing

- **The checkpoint over-dissipates wakes (OP-3, $\sim\!20\times$ too fast per call).** The absolute $+X\%$ may shrink or move under classical verification. **Mitigation:** state the deployment story [[prior-art-and-novelty-atlas-0.1]] §5 already commits to — *search wide and cheap with the composed model, verify the shortlist with the classical stack.* The speed-and-gradient claim is independent of the checkpoint's absolute accuracy.
- **Quasi-steady, not a converged rollout.** Wake steering is a slowly-varying problem, so a short march to a developed state is defensible, but it must be stated and the sensitivity to march length reported.
- **The gradient could be noisy** through a frozen operator with a float32 forward pass whose output depends on batch position ([[open-problems-atlas-0.1]] OP-6, $\sim\!10^{-6}$). **Mitigation:** pin the batch layout across the rollout; report a finite-difference check of a few components of $\nabla_\theta J$.
- **Yaw–wake deflection is a model, not the checkpoint's physics.** The actuator-disk deflection law is a closure. Say so; it is standard practice (FLORIS does the same) and it is exactly the kind of lumped expert Atlas is built to carry as a first-class peer.
- **It does not prove F1 or Claim B.** This is a capability demo, not a scaling result. Keep it off the [[case-study-ladder-to-f1]] ladder in every writeup.

## 6. Dependencies

- **Hard: the divergence-free assembly fix ([[case-study-scaling-ladder-atlas-0.1]] §5, [[gap-worklist]] W100).** CS-7 found that a partition-of-unity blend of per-window divergence-free fields is not divergence-free on the overlap — $\nabla\cdot(\chi_1 u_1 + \chi_2 u_2) = \nabla\chi_1\cdot(u_1 - u_2)$ — and the classical composed rollout goes unstable by macro-step $\sim\!80$. The fix is one global Leray projection on the *assembled* field each macro-step (what the Poseidon column already does, what `R10b` prescribes). The optimiser marches many rollouts; every one must be stable. This should land as a first-class *projected assembly* step before the PoC is built.
- **Soft: a batched multi-turbine yaw model** in the actuator-disk expert. Small.
- **None:** no new trained experts, no new data, no new port types.

## 7. Stretch — co-design the controller through the fluid

The wake array's `ROT` ports are open ([[port-algebra-atlas-0.1]] §5.1: an open port is measurable incompleteness, and "add the generator" is *connect a port*). Wire them to a lumped drivetrain/generator expert and a yaw controller with parameters $\phi$, then optimise $J$ over $(\theta, \phi)$ jointly — **layout, generator, and control law co-designed, with gradients through the coupled fluid.** A monolithic PDE foundation model structurally cannot represent the torque map or the controller; this is [[prior-art-and-novelty-atlas-0.1]] §2.3 made concrete and is the strongest version of the story. Schedule it as PoC 1b once 1a lands.

## 8. What this does not claim

- Not more accurate than CFD — it is not, and should never say so.
- Not a replacement for the classical stack — it is a pre-screen.
- Not a scaling result, not a multiphysics result (one governing family), not a certified composition (Poseidon-T is globally receptive, [[expert-donor-survey]]).
- The novelty is precisely: **gradient-based design optimisation through a coupled graph of frozen, independently-pretrained experts, fast, on one machine.** That sentence is the whole marketing claim and it is true.

---

## 9. Results (run 2026-09-01) — built, and it works, with one large asterisk

Full record: [[poc1-results-differentiable-design]]. Artefact `out/w111/w111.json`; code `atlas/cases/wind_farm_design.py`, `scripts/w111_wind_farm_design.py`, `tests/test_tier21_wind_farm_design.py` (36 assertions, all passing).

**The asterisk first, because §1's sentence overstates what was run.** Every rollout used `wake_array.exposed_reference_solver` — the *classical* reference solver — as the fluid agent, per the build brief and because it is the only classical arrangement that survives a long march. So the **partitioning** half of the novelty claim and the **field + lumped peers** half (§2.3 of [[prior-art-and-novelty-atlas-0.1]]) are exercised; the **frozen-pretrained** half (§2.1) is not. Poseidon-T is differentiable and the swap is a `kind=` argument, but it has not been run, and §1's sentence should not be quoted as if it had.

> **The asterisk is retired as of 2026-09-02 — §11 below, [[poc1a-frozen-expert-results]].** The swap was a `kind=` argument, as this paragraph predicted, and it has now been run at both sizes. §1's sentence may be quoted; §11 is the list of the things it still may not be quoted *for*.

**The number.** Same start, same objective, same box-and-spacing projection for both methods; tolerance = within 2% of the gradient method's own gain.

| | K = 12 (36 design vars) | K = 25 (75 design vars) |
|---|---|---|
| gradient rollouts to tolerance | **26** | **24** |
| CMA-ES evaluations spent / best reached | 700 / never reached | 928 / never reached |
| ratio (lower bound) | **> 26.9x** | **> 38.7x** |
| measured ratio at 75% of the gain | 17.9x | **48.3x** |
| coordinate finite difference, 2n+1 per gradient | **73x** | **151x** |
| farm power, *as measured by the composed model* | +315% | +203% |

**The ratio grows with the design dimension** — 17.9x to 48.3x from 36 to 75 variables at matched tolerance — which is the trend that would carry §4's "two to three orders of magnitude". At these sizes it reaches one to nearly two, not three; that is stated rather than rounded up.

**One gradient costs 6 to 9 forward evaluations** (checkpointed, measured), against 73 and 151 for a coordinate finite difference. On a 22-core desktop with no GPU a K12 gradient rollout is 140 s; on one rented A100 it is 38 s. The entire GPU experiment cost \$0.69 and 55 minutes.

**The gradient is right.** Adjoint against central differences at 9 components: median relative error **5.9e-9** at h = 1e-5, cosine 1.0000000000, with a textbook truncation/cancellation curve either side. J is bit-reproducible, and bitwise identical between CPU and CUDA. The check earned its keep: it caught a detached path in the disk's smearing thickness as a systematic 0.2% deficit in every component.

**The coupling is doing the work, and this is the sharpest result.** Differentiating the same objective with the rollout *frozen* — the gradient a static wake surrogate would give — yields a vector 3 to 11 times larger and nearly orthogonal to the truth (cosine 0.238 at K12, **0.054** at K25). Its streamwise component at K25 is *anti*-correlated (cosine **-0.115**, signs agreeing 40% of the time). Yaw is the transferable part (cosine 0.95). So a static model would get yaw roughly right and site turbines in the wrong direction.

**The blocking dependency is closed**: N24, never marched past 20 macro-steps, is confirmed stable to 60 with all 25 disks live (u_max 1.52 against a band of 3.0, divergence tail trend 0.963).

**What §5's honest framing looks like once measured.** Quasi-steady and not converged: settling 1.9e-2 (K12) and 3.0e-2 (K25) at the optimum, and the gain moves from +236% at 30 macro-steps to +315-336% across 40-70, so the 40-60 band is load-bearing. Neither optimiser converged inside its budget, which makes the ratios conservative. Both optima sit *on* the spacing bound and the domain box. And the +315% is large mostly because the starting grid is pathological — 19% of the unwaked ideal — so **the ratio and the wall-clock are the claim, not the percentage**, exactly as §4 committed.

---

## 10. Interactive demo (2026-09-01)

`python scripts/w112_farm_demo.py`, then <http://127.0.0.1:8011/>. Code `atlas/demo/`, docs `atlas/demo/README.md`, tests `tests/test_tier22_demo.py` (29 assertions). §4's "interactive version: drag a turbine, the coupled field and the power readout update live" now exists. A self-contained copy that runs on any machine, macOS included, is on the `poc1-windfarm-demo` branch — see `atlas/demo/PACKAGING.md`.

**The demo makes two claims and keeps them apart on screen.** ① **Optimise**, which only the coupled graph can do: one backward pass through the entire coupled simulation returns the derivative of farm power with respect to every position and every yaw at once, so the optimiser moves all of them together. ② **Check**, which either solver can do: a single layout marched by the coupled graph and by the undivided classical solver, both fields drawn side by side on the same fixed colour scale, both timed per step. The classical solver can produce ② and cannot produce ①, and that is the whole argument for the composed graph — stated on the page rather than in a caption.

**The head-to-head is a second window, and it is timed properly.** The two solvers **alternate, one macro-step each**, so every timed region has the machine to itself; running them concurrently would make a better animation and a worthless measurement. For the same reason the **live march parks itself** while a comparison runs, and the window says so while it is doing it — a per-step cost measured against a moving load is a measurement of the load. Because both sides sit at step *i* at the same moment, their fields are directly comparable, so the panel also reports the worst single-cell velocity difference alongside the difference in farm power. Results are cached by layout hash; the main window's *Check this layout* button is instant for a layout already checked.

**A measured run at 6 windows, 6 turbines, 30 steps, on an optimised layout:** coupled 943 ms per macro-step and 2.73 farm power; classical 611 ms and 3.27; difference **-16.5%**, worst single cell 0.475 of freestream. So the classical column is the **cheaper** one per step, by 1.54x. Splitting a domain into windows and re-assembling them costs more than solving it in one piece, and the panel says so rather than implying a speedup that is not there. The coupled graph's advantage is the gradient, not forward speed.

**A number this section used to give was wrong, and the fix is structural.** The 2026-09-01 first version of §10 said one composed macro-step is 212 ms at 6 windows (4.7 Hz). The same code on the same box, unchanged, later measured **~890 ms** — the machine is a Core Ultra 7 155H and it had clocked itself down from 3.8 GHz to 1.4 GHz. Two measurements taken 40 minutes apart on that box differ by a factor of two. **No wall-clock constant is quoted in the demo any more**: `Engine.eta_s` reports an exponential moving average of what iterations have actually cost on the machine in front of you, the screen marks it *(estimate)* only until the first real iteration lands, and the head-to-head times both solvers per step and holds out the first (which pays for FFT plans and allocation). What is stable is the ratio: a gradient macro-step costs about 5.5 forward ones, and both scale with the window count. Every ms figure above is a measurement on one machine in one power state, including the ones two paragraphs up.

**The estimator the live number uses is still not §9's, and the demo still says so.** §9 marches from the freestream on every evaluation, which makes J a function of the layout alone; the demo is warm-started and short-horizon, so it carries the history of every layout the user has dragged through and sees less downstream interaction. **The head-to-head is what converts it back into a defensible number** — which is why it is a first-class window rather than a panel at the bottom of the page.

**Replay plays and pauses.** Every optimiser iteration records its design vector, and the replay walks back through them with play/pause and a scrubber. Only the layouts are recorded, not the fields — a stored field per iteration would be about 1.3 MB a step — so the replay **re-marches**: the turbines jump to where they were and the wake re-forms around them, which is why the cursor advances once every two macro-steps rather than once a frame. Pausing does not stop the fluid, so pausing on a layout lets its wakes settle, which is the only way to see what that layout was doing rather than what it looked like in passing.

**The validity panel is a citation.** Each row is a predicate something already declared, with its number and its provenance: `reference.WindowNS`'s own cell-Reynolds limit of 8 (the case study sits at 7.97, so the wind-speed slider has almost no headroom above 1.0), W100's stability band, and the Leray projection's declared hypothesis that the march holds the inlet and both laterals at (U_INF, 0) — which makes **wind direction the knob that leaves the regime**, red past 2 degrees. One row is deliberately a **diagnostic rather than a gate**: the same predicate on the *state* is ~11-13 in any developed wake, above 8 at every inflow the demo allows, which §5 already records. Gating on it would paint the panel red permanently for a condition nobody can act on.

**Nothing in the optimiser was modified.** `engine.DemoRollout` overrides `band` and `project` so the freestream can be pointed and scaled, and at the default inflow it is asserted **bitwise identical** to `wind_farm_design.Rollout` on a real macro-step. That assertion is what says the animation is the column §9 measured.
---

## 11. The frozen-expert column (run 2026-09-02) — §9's asterisk, retired

Full record: [[poc1a-frozen-expert-results]]. Artefact `out/w118/w118.json`; code `atlas/cases/wind_farm_design.py` (`TapedPoseidon`, `PoseidonRollout`, `rollout_for`), `scripts/w111_wind_farm_design.py --expert poseidon`, `tests/test_tier26_poseidon_design.py` (17 assertions, passing on CPU and CUDA).

**Every window is now Poseidon-T** — a frozen $20.8$M-parameter checkpoint this project did not train — and the adjoint runs through all of it. The tiling, the disks, the blend, the declared `ProjectedAssembly`, the objective, the optimiser and the constraint projection are the same objects §9's column used, unchanged. So all three of [[prior-art-and-novelty-atlas-0.1]] §2's halves are exercised at once for the first time, and §1's sentence is now a description of something that exists.

**The five things worth knowing before quoting it.**

1. **It is stable.** The projected assembly holds a **globally receptive** expert — W93 measured this checkpoint's domain of dependence as the whole window — for 70 macro-steps at both rungs, $u_{\max} = 1.45$ against a band of $3.0$, divergence tail flat. §6's hard dependency did not need re-opening.
2. **The gradient is right, to the precision the checkpoint allows.** Median relative error $1.4\times10^{-3}$ against central differences at $h = 10^{-2}$, textbook curve either side. §5's OP-6 caveat lands exactly where it was aimed: the float32 forward pass puts the finite-difference floor six orders above the classical column's, and *pinning the batch layout buys reproducibility, not precision*.
3. **The headline ratio is real but three times smaller, and at $K=12$ the baseline wins.** $8.0\times$ at 36 design variables (CMA-ES reached the tolerance in 144 evaluations and finished $15.6\,\%$ **above** the gradient optimum) and $>30.9\times$ at 75 (CMA-ES never reached it). §4's "two to three orders of magnitude" is further away on this column than on the classical one, and the dimensional trend that would close the gap is still there.
4. **§5's honest framing was right about OP-3 and can now be priced.** A layout found entirely inside the frozen column, scored by the undivided classical solver, is worth $+266\,\%$ (K12) and $+170\,\%$ (K25) over the starting grid — against $+374\,\%$ and $+240\,\%$ for the classical column's layout under the same verifier. **The pre-screen works and captures $71\,\%$ of the verified gain, the same fraction at both sizes.** That is §5's *"search wide and cheap with the composed model, verify the shortlist with the classical stack"* turned into a measurement.
5. **And the composed graph is finally the fast one.** §10 measured the classical composed column at $1.54\times$ **slower** than the undivided monolith. With the checkpoint in the windows, on the same box, the composed column is **$3.15\times$ (K12) and $3.64\times$ (K25) faster** than the monolith. The cut still costs what it costs; what changed is what is inside each window.

**The demo carries the toggle:** `python scripts/w112_farm_demo.py --expert poseidon`, and the panel names the expert on screen rather than leaving a viewer to infer which column produced the power number.

---

## See Also

- [[poc1a-frozen-expert-results]] — §11's record; the frozen-expert column, measurement for measurement against §9's
- [[prior-art-and-novelty-atlas-0.1]] — §1 (speed/differentiability as the value proposition), §2.3 (field + lumped peers), the adjoint claim this spends
- [[case-study-wake-array-atlas-0.1]] — the graph this optimises over
- [[case-study-scaling-ladder-atlas-0.1]] — §5's instability and the projected-assembly fix this depends on; §6's timing numbers behind the interactive claim
- [[port-algebra-atlas-0.1]] — the `ROT` ports §7 connects; the power read as the objective
- [[composition-error-theory]] — why the composed model is a search instrument and not a verifier
- [[expert-donor-survey]] — why Poseidon-T cannot be certified, and what a certifiable replacement would be
- [[f1-pathmap-and-end-goal]] — the long-range target this is explicitly *not* on the path to yet
