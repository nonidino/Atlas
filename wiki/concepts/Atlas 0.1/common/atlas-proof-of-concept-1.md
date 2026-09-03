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

## 10. Interactive demo — rebuilt 2026-09-02 around the frozen column

`python scripts/w112_farm_demo.py`, then <http://127.0.0.1:8011/>. Code `atlas/demo/`, docs `atlas/demo/README.md` and `atlas/demo/PACKAGING.md`, bundle sources `atlas/demo/bundle/`, builder `scripts/w112_build_bundle.py`, tests `tests/test_tier22_demo.py` (**38 tests**, passing). A self-contained copy that runs on a machine with nothing set up — no repositories, no environment, no downloaded weights — is on the `poc1-windfarm-demo` branch.

**What changed, and why.** The 2026-09-01 version made two claims and kept them apart: **①** *optimise*, which only the coupled graph can do, and **②** *check*, which either solver can do — and its own head-to-head found the composed column $1.54\times$ **slower** than the undivided monolith, which it reported rather than hid. §11 changed the second claim: with Poseidon-T in the windows the composed column is *faster*. So the head-to-head stopped being a caveat in a second window and became **the top half of the main screen** — the two streamwise-velocity fields side by side on the one fixed ramp, turbines drawn to scale with their yaw as their rotation — and every number under them is produced on the machine in front of the viewer.

**One measurement region feeds three panels, and nothing else feeds them.** A region parks the live march and marches one layout from the freestream with **every column taking one macro-step in turn**: the composed graph with Poseidon-T in every window, the *same cut* with `reference.WindowNS` in every window, and `scaling_ladder.reference_monolith`, undivided. Alternating is what gives each timed region the machine to itself, and it is also what makes the fields comparable cell by cell, because every column sits at step $i$ at the same moment. Off that one march the screen reads **speed** (ms per macro-step per column, exponentially averaged over regions, the first step of each held out), **accuracy** (worst single-cell $|\Delta u|$ against the monolith, and the difference in farm power), and the **three-way scoring** of §7.2. One region runs automatically at boot, so the headline is on screen without a viewer having to know to ask for it, and the panel fills in from the region *in flight* after two macro-steps rather than staying empty for two minutes.

**The headline, measured five times on the development box in one evening.** At the default rung — `medium`, $464\times352$ cells, 12 windows, 12 turbines, the K12 rung §11 measured:

| where the region ran | composed (Poseidon-T) | same cut, classical windows | undivided monolith | ratio |
|---|---|---|---|---|
| boot region, 40 steps/column | $688$ ms | $460$ ms | $1832$ ms | $\mathbf{2.66\times}$ |
| a region at 20 steps/column | $437$ | $285$ | $1414$ | $3.24\times$ |
| a region at 10 steps/column | $1265$ | $1052$ | $4276$ | $3.38\times$ |
| the packaged bundle's own self-test, in its own venv | $787$ | — | $1658$ | $2.11\times$ |
| a standalone script, 6 steps | $1179$ | — | $3248$ | $2.75\times$ |

**The same composed macro-step cost between $437$ and $1265$ ms — a factor of $2.9$ — on one box, on one evening, on unchanged code.** That is this section's standing lesson arriving again, unprompted, and it is why the demo quotes nothing it has not just measured. The **ratio** moved less on this box, $2.1\times$ to $3.4\times$, and §11's $3.15\times$ sits inside that — **but the ratio is not a constant either, and on one machine it changed sign.** Verifying the packaged branch (below) measured the same comparison at $1.07\times$ on one rented Linux box and at $\mathbf{0.43\times}$ — the monolith *faster* — on another, whose CPU is a share of a 192-core EPYC. So the honest statement of §11's headline is: **the composed column is faster on the hardware the result was measured on, by a margin that is a property of that hardware, and the demo's job is to measure it where it is standing rather than to promise it.** What no machine changes is that only the composed column can be differentiated. The published $3.15\times$ is re-derived on the machine in front of the viewer rather than quoted at them, and appears on screen only in a field of its own, labelled as measured elsewhere.

**A third column, and a finding the results page does not contain.** §7.1 *scored* the classical composed column and §8 *timed* only the Poseidon one, so nothing in this vault had timed **the same cut with the classical solver in it** at the K12 rung. The demo does, because the three-way scoring needs that column anyway — and on this box it runs at $3.98\times$ the monolith, **faster than the Poseidon column**. Read carefully that says the *cut itself* has already paid for itself at twelve windows on this host, and the expert is not the whole reason the composed column wins here. It contradicts nothing published: §10's old $1.54\times$-slower was measured at **six** windows, and §11's "the reason is the expert" compares against a monolith whose cost grows with the domain. But it is a second variable that the published pair of numbers cannot separate. **[AI Inference]:** the likeliest mechanism is that the demo's classical column is the **exposed** solver — its elliptic part removed and handed to the composition layer, so a window solve is a stencil sweep with no per-window pressure solve — while [[tier0-measurements]] §19.10's per-agent classical numbers are the as-built agent with its own Poisson solve. If that is it, this row measures "cut plus exposed elliptic" against "cut plus checkpoint", which is a third comparison and not either published one. It is one machine, one rung, and it deserves a worklist row; it does not have one yet.

**The scoring table, and the horizon that decides whether it can see anything.** Rows are layouts — the one the demo started from and the one the optimiser reached — and columns are the three solvers, with the **monolith's** gain as the headline because it is the only column with no composition error in it. A complete run on the development box, at the demo's own defaults, 14 optimiser iterations in $127$ s at $10.5$ s an iteration:

| layout | composed (Poseidon-T) | same cut, classical | **undivided monolith** |
|---|---|---|---|
| the starting grid | $3.065$ | $2.956$ | $3.393$ |
| the layout at optimiser step 14 | $3.783$ | $3.588$ | $\mathbf{4.328}$ |
| **gain over the start** | $+23.4\,\%$ | $+21.4\,\%$ | $\mathbf{+27.5\,\%}$ |

**A layout found entirely inside the frozen-checkpoint column, by an optimiser that never called a classical solver, is worth $+27.5\,\%$ when the undivided classical solver scores it.** That is §7.2's pipeline, live, in about six minutes, on a laptop-class CPU — and it is a *demonstration* rather than a re-measurement of §7.2: fourteen iterations is not convergence, and §11's $+266\,\%$ is a converged run under the PoC's own protocol.

**How far a region marches decides whether that number exists at all.** The same 14-iteration layout, scored by the monolith at four horizons: $\mathbf{-1.4\,\%}$ at 20 macro-steps, $+29.3\,\%$ at 30, $+27.7\,\%$ at 40, $+34.5\,\%$ at 50. **At 20 steps the wakes of the unoptimised grid have not developed, so it has not yet paid for being a grid and there is nothing for the verifier to see.** §9 already found the 40–60 band load-bearing for exactly this reason; the demo's default is now 40 steps per column, and the settled power is averaged over the last quarter of the march rather than its last five samples, because a developed wake meanders and five samples of an oscillation moved a measured gain by 15 percentage points. The start layout is still settling at step 50, so every one of these numbers is a lower bound.

**The capture fraction is shown, and it is withheld until it is earned.** §7.2's $71\,\%$ compares an optimum found on the frozen column against one found on the classical column, under the same verifier. The demo can compute exactly that — the expert is a live toggle and swapping it rebuilds every solver — but only if a session has actually optimised and scored on both. Until then the panel shows the gain it *did* verify and cites $71.2\,\%$ / $70.9\,\%$ with the page and section they came from; once both optima exist it replaces the citation with that machine's own number. Keeping those two apart on screen is what W98's lesson looks like in a user interface.

**No wall-clock constant survives anywhere in the demo.** The 2026-09-01 version still carried a per-domain table of cold-start estimates for the seconds-per-optimiser-step readout. It is gone: `Engine.speed` starts empty and is filled only by measurement regions, `Engine.eta_s()` returns `None` rather than a guess, the screen says *not measured yet* until an iteration has been timed and marks the interim figure *(estimate)*, and `tests/test_tier22_demo.py` asserts that the deleted table has not come back.

**Packaging: one command, from a bare checkout, with the checkpoint in it.** The previous bundle listed six unpinned packages, left `torch` to pip — which on Linux fetches a 2.5 GB CUDA wheel whether or not there is a GPU — and did not carry the checkpoint at all, so the frozen column would have silently downloaded 83 MB on first use or failed. Now every package is pinned to an exact version; `torch` is pinned too and installed by a launcher that looks for an answering `nvidia-smi` and takes the CUDA wheel if there is one and the CPU-only wheel otherwise; the Python version is checked up front against the range the pins have wheels for; `run.py` imports every dependency by name and says which is missing before the server starts; and **the checkpoint is in the branch**, at `vendor/hf-cache/` in the hub's own cache layout with `HF_HOME` and `HF_HUB_OFFLINE` pointed at it, so `ScOT.from_pretrained("camlab-ethz/Poseidon-T")` resolves it offline with **no source file patched** — which is what lets the tests on that branch be the same files, character for character, as the ones here. The one thing still fetched at install time is `scOT` itself, from a pinned upstream source archive, because `github.com/camlab-ethz/poseidon` **publishes no licence file** and is therefore not ours to redistribute; the weights, which are CC-BY-NC-4.0, are.

**Verified from the branch itself, on three machines with nothing set up, and it earned its keep on the first command.** On **Windows 11, Python 3.12, no NVIDIA driver**: a fresh copy of the branch and `run.cmd`, which built the venv, installed every pin and the CPU-only torch wheel, installed `scOT` from its pinned archive, loaded the bundled checkpoint **offline**, and marched both columns — $787$ ms composed against $1658$ ms undivided. On a rented **RTX 3090 host running Ubuntu 22.04 with Python 3.10** and nothing else installed, a CPU-only environment built the same way: $636$ ms against $681$ ms. On a rented **RTX 4060 Ti host** (a share of a 192-core EPYC 7K62), `./run.sh` detected the driver, took the CUDA wheel, and the self-test marched the composed column at $838$ ms on that CPU against $363$ ms for the monolith — *the monolith faster* — and at $\mathbf{64}$ ms on the GPU. **The farm power agreed to four digits across all three machines and both devices** ($8.870$ composed, $8.884$ undivided, after three steps from still air), which is the check that the three installs are running the same model.

Then the whole demo, on the GPU box: `tests/test_tier22_demo.py` **38 passed**, `tests/test_tier26_poseidon_design.py` **17 passed** — the same files as this repository's, character for character — and the server with `--device cuda` ran a full measurement region: composed $61$ ms, the classical windows $93$ ms, the monolith $390$ ms on the CPU, worst single cell $0.606$, farm power $-9.2\,\%$. That last ratio is **not like-for-like** — the composed columns are on the card and the monolith is numpy on the CPU — and the screen now says so in a box of its own, which it did not before this run.

**Three faults the verification found, which is what it was for.** ① `./run.sh` returned **exit 126** on the first Linux machine: the branch carried it as mode `644`, because `os.chmod` does nothing on Windows, where the builder runs — a Linux user's first command would have been *Permission denied*. The builder now sets the mode in the git index. ② `pytest tests/test_tier22_demo.py` printed `38 passed` and then **aborted at interpreter exit** with *"terminate called without an active exception"*, exit 134: `Engine.stop()` set a flag and did not wait, so a daemon worker was killed inside a torch call. It now joins, and the same command on the same box returns 0. ③ The Linux GPU path pulled its 3 GB of wheels from PyPI at $0.6$ MB/s where `download.pytorch.org` served the same wheel at $2.1$ MB/s from that host, so both launchers now name the CUDA index — which pins the CUDA version as a side effect. Neither GPU box was given a `git clone`: the repository is private and a GitHub token does not belong on a rented machine, so each received a `git archive` of the pushed commit, byte-identical to the branch.

**The licence is on the screen, not only in a file.** The header names the expert — *Poseidon-T (frozen, 20.8M parameters)* — and carries **CC-BY-NC-4.0, research use only** beside it, because the speed claim is a property of what is inside the windows and a viewer who cannot tell which expert is running cannot read the number. `--expert reference_exposed` swaps it and the whole screen re-labels itself. [[expert-donor-survey]]'s consequence is **W120**, and this demo is now the most visible place the non-commercial term bites.

**What did not change, and is load-bearing.** `DemoRollout` and `DemoPoseidonRollout` still override only the freestream — and, for the checkpoint, the translation kernel that follows from it — and each is still asserted **bitwise identical** to its parent in `wind_farm_design` on a real macro-step at the default inflow. That assertion is the licence for the subclasses to exist and is what says the animation is the column §9 and §11 measured. The validity panel is still a citation with a number, a limit and a provenance per row, still with one row that is a diagnostic rather than a gate. The replay still re-marches rather than storing fields. Nothing in the optimiser, in `wake_array.exposed_reference_solver`, or in the projected-assembly internals was touched.

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
