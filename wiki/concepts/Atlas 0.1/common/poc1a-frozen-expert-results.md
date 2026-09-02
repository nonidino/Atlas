# PoC 1a — Results: the frozen-expert column

**Type:** Concept page — **results record**, the companion to [[poc1-results-differentiable-design]] (folder: `Atlas 0.1/common/`)
**Status:** run 2026-09-02. Every number below is quoted from `out/w118/w118.json` and the artefacts merged into it; nothing is estimated. This is a **capability demonstration**, not a rung of the [[case-study-ladder-to-f1]] ladder.
**Code:** `atlas/cases/wind_farm_design.py` (`TapedPoseidon`, `PoseidonRollout`, `rollout_for`), `scripts/w111_wind_farm_design.py --expert poseidon`, `tests/test_tier26_poseidon_design.py` (17 assertions, all passing on CPU and on CUDA)
**Related:** [[poc1-results-differentiable-design]] · [[atlas-proof-of-concept-1]] · [[prior-art-and-novelty-atlas-0.1]] · [[case-study-wake-array-atlas-0.1]] · [[open-problems-atlas-0.1]] · [[expert-donor-survey]] · [[composition-error-theory]] · [[poc1-retrospective-and-hybrid-roadmap]]

---

## 0. What this closes

[[poc1-results-differentiable-design]] §0 opens with a disclaimer:

> "The spec's headline sentence is about *frozen, independently-pretrained* experts. **This run does not exercise that half.** … Poseidon-T is a PyTorch module and is differentiable, so the substitution is a `kind="poseidon"` swap rather than new machinery — but it has not been run, and no sentence on this page should be read as if it had."

**It has now been run.** The same optimiser, the same objective, the same tiling, the same disks, the same partition-of-unity blend and the same declared `assembly.ProjectedAssembly` — with `reference.WindowNS` replaced in every window by **Poseidon-T**, a frozen $20.8$M-parameter scOT neural operator that this project did not train and cannot retrain. Both sizes, $K=12$ and $K=25$. Every headline measurement of that page reproduced.

**So, against [[prior-art-and-novelty-atlas-0.1]] §2, which of the three halves does each run exercise?**

| | §2.1 — partitioned coupling around a **frozen neural** subsolver | §2.2 — **partitioning** as the composition axis | §2.3 — **field + lumped** experts as peers |
|---|---|---|---|
| the 2026-09-01 classical column | **no** — every window is `reference.WindowNS`; nothing learned is in the loop | yes — 12 or 24 overlapping windows, adjoint through every seam | yes — one zero-parameter actuator disk per rotor, `ROT` port open |
| **this run** | **yes** — every window is the frozen checkpoint, and the adjoint goes through all 20.8 M parameters' activations | yes — identical tiling, identical assembly | yes — identical disks, identical closure |

That is the whole of what §0 was waiting for. It is stated as a table rather than a sentence because the temptation is to write "now the claim is fully exercised" and stop, and §9 below is a list of the things that are still not true.

**One thing this page must not be read as saying.** The frozen column is **not more accurate**, and §7 measures how much less accurate it is. What it demonstrates is that the differentiable-composition machinery is indifferent to what is inside a window — which is precisely the property [[expert-library-atlas-0.1]] asserts and nothing had yet spent.

---

## 1. The swap, and what had to be built to make it possible

`rollout_for(case, kind="poseidon")` is four lines. Two things behind it were not.

**The checkpoint arrives with its differentiability deliberately removed.** `adapters.FrozenFluidExpert` cuts the tape in three separate places — numpy in, `with torch.no_grad()` around the forward, `.detach().cpu().numpy()` on the way out — and its own docstring says why: *"so that a gradient cannot be taken by accident."* Taking one **on purpose** needed a taped twin, `TapedPoseidon`, written in the vault rather than in the build repo for the same reason `window_ns._no_projection_class` and `_TapeBackend` are: the agent is not ours to change, and what the composition layer may do is decline to use one part of it and supply that part itself. §2 asserts the twin is the wrapper's arithmetic to the bit.

**The checkpoint needs more from the composition layer than the solver does, and this is a real asymmetry rather than plumbing.** `reference.WindowNS` is a Navier–Stokes solver: it advects, and the composition layer owes it only the global Leray projection (the `ProjectedAssembly` of [[case-study-scaling-ladder-atlas-0.1]] §5). Poseidon-T, run the way W98 settled — `galilean=False`, `project=False` — advances the *fluctuation* and does **not** transport it, so the composition layer owes it the advection as well. One macro-step is therefore

$$\text{cut} \to \text{expert} \to \text{mean restore} \to \text{blend} \to +\,\mathbf{f}\,\Delta t \to \text{Leray} \to \text{translate by } U_\infty \Delta t \to \text{band},$$

which is `scripts/w93_wake_array.py`'s march exactly, in torch, with the adjoint through all of it. Three of those steps are W-numbered findings and none is a choice made here: the mean restore is W0's E1 (fed a uniform flow the checkpoint returns a mean of $0.969$ after one call and $0.281$ after forty); the global transport and the global projection are **W98**; and the body force is applied *after* the blend because `step_many` **drops** `force` silently when `galilean` is off, which is **W99**.

**Nothing that the expert is composed *by* was touched.** `assembly.ProjectedAssembly`'s internals, `wake_array.exposed_reference_solver`, `wake_array.project_assembled`, `_place`'s fixed accumulation order and the whole of `wake_array.build`'s declaration are unchanged, and `tests/test_tier26` asserts that the classical `Rollout` still constructs the classical agent and holds no checkpoint.

---

## 2. Provenance — is this the checkpoint the case studies marched?

Yes, and the first row is bitwise.

| check | result |
|---|---|
| `TapedPoseidon.step_batch` vs `FrozenFluidExpert.step_many(galilean=False, project=False)`, real batch of 12 windows | $\max\lvert\Delta u\rvert = \mathbf{0.0}$, $\max\lvert\Delta v\rvert = \mathbf{0.0}$ — **bitwise** |
| torch `transport_project` vs `wake_array.transport_and_project` | $2.22\times10^{-16}$ / $1.67\times10^{-16}$ |
| one whole composed macro-step vs `scripts/w93_wake_array.py`'s numpy march, written out line for line | $2.22\times10^{-16}$ / $1.11\times10^{-16}$ |
| parameters, and parameters requiring a gradient | $20\,774\,444$ and **$0$** |
| $J(\theta)$ evaluated twice, same $\theta$ | **bit-identical**, on CPU and on CUDA |
| expert lead time at $\Delta t = 0.2$ under `Scaling(4.0, 2.0)` | $0.1$ — exactly the checkpoint's native lead |

**"Frozen" is asserted, not assumed.** `n_grad_params` is zero and the tests check that after a backward pass no parameter holds a gradient. The adjoint reaches $\theta$ through 20.8 M parameters' *activations* and accumulates nothing on the weights, which is the difference between differentiating through a frozen expert and fine-tuning one.

### 2.1 Two things the classical column never had to worry about

**TF32 is turned off, in code, and the state is recorded.** On an Ampere card torch runs float32 matmuls in TF32 by default — ten mantissa bits, so roughly $10^{-3}$ relative. The classical column is float64 and never met this; the checkpoint is float32 and *every* matmul is on that path. Left on, it would move $J$ in its fourth significant figure between CPU and GPU, and a finite difference of a number that moves in its fourth digit measures the arithmetic. `TapedPoseidon._pin_precision` disables it where the expert is built, and `tests/test_tier26` asserts the flag rather than trusting it, because it is a global that anything in the process can turn back on.

**Cross-machine agreement is $10^{-6}$, not $10^{-16}$, and that is the checkpoint.** The classical column reported $J$ *exactly equal* between a 22-core Windows desktop and an A100. Here:

| quantity | 22-core desktop (CPU, float64 column) | A100 SXM4 (CUDA) | relative |
|---|---|---|---|
| $J$ at the K12 start layout | $2.4781570027046085$ | $2.4781639622569047$ | $2.8\times10^{-6}$ |
| $u_{\max}$, K25, 70-macro-step march | $1.4397067614816792$ | $1.4397076419518902$ | $6.1\times10^{-7}$ |
| ablation cosine, K12 | $0.413272$ | $0.413268$ | $10^{-5}$ |

$6\times10^{-7}$ is the same order as **OP-6's own measured batch-position spread of $6.618\times10^{-7}$** on this checkpoint, and it has the same cause: a float32 forward pass whose reduction order differs between kernels. Within one machine the layout is pinned and $J$ is bit-reproducible; across machines it is not, and no pinning available to the composition layer would make it so. §5 is where that costs something.

---

## 3. Does a globally receptive expert break the projected assembly?

**No — and this was the open question the swap was most likely to fail on.** W93 measured this checkpoint's domain of dependence as the *whole window*: it declares a 2-cell halo and a delta poke returns nonzero response in all 128 seam cells, which is why `L2/R10` refuses to certify the decomposition at all ([[case-study-wake-array-atlas-0.1]] §3). A partition-of-unity blend whose ramp is 8 cells is, for such an expert, a blend of two fields neither of which respects the region it is being blended over. The classical column's stability at 60–120 macro-steps carried no implication for it.

Marched from the freestream with all disks live, the same rule `w100_scaling_ladder.stage_long_march` applies — $\lvert u\rvert$ inside a band of $3.0$, finite throughout, assembled divergence flat or falling over the developed tail:

| | K12 (N12 rung, 12 windows) | K25 (N24 rung, 24 windows) |
|---|---|---|
| finite to | $70/70$ | $70/70$ |
| $u_{\max}$ against a band of $3.0$ | $\mathbf{1.4519}$ | $\mathbf{1.4397}$ |
| assembled divergence, first $\to$ last | $1.75\times10^{-2} \to 8.58\times10^{-2}$ | $1.72\times10^{-2} \to 8.53\times10^{-2}$ |
| developed-tail trend | $\mathbf{1.020}$ | $\mathbf{1.008}$ |
| farm-power settling over the last 5 steps | $1.36\times10^{-2}$ | $1.01\times10^{-2}$ |

Compare the classical column at the same N24 rung: $u_{\max} = 1.5192$, tail trend $0.963$. **The two columns sit in the same band and neither leaves it.** The checkpoint's divergence tail is flat-to-very-slightly-rising rather than falling, which is consistent with the assembled field carrying a residual the global Leray projection removes each step and the expert re-introduces — but at $8.5\times10^{-2}$ after seventy steps it is not accumulating, and the optimiser's 40–50-step horizon sits well inside it.

**[AI Inference]:** the reason this works despite W93 is worth stating because it is not luck. The projected assembly's job is to make the *assembled* field divergence-free, and it does that globally, on the whole domain, after the blend — so it never needs the per-window fields to be individually valid outside their own supports. A globally receptive expert breaks the *certification* argument (R10 refuses, and rightly) and does not break the *arithmetic*. That is exactly the gap [[composition-error-theory]] describes when it calls the composed model a search instrument rather than a verifier: it runs, and running is not the same as being certified to run.

---

## 4. The headline number

Same rule as [[poc1-results-differentiable-design]] §3, so the two pages can be read side by side: both methods start from the identical layout, optimise the identical objective, get the identical box clip and spacing projection, and the tolerance is $J_{\text{grad}} - 0.02\,(J_{\text{grad}} - J_0)$ — within 2 % of the gradient method's *own* improvement. Adam gets 30 steps, CMA-ES gets the same budget the classical run gave it.

| | K = 12 (36 vars) | K = 25 (75 vars) |
|---|---|---|
| $J$: start $\to$ gradient optimum | $2.4782 \to \mathbf{7.0243}$ | $7.0997 \to \mathbf{14.6951}$ |
| farm power gain, *as measured by the composed model* | $\mathbf{+183.5\,\%}$ | $\mathbf{+107.0\,\%}$ |
| **gradient rollouts to tolerance** | $\mathbf{18}$ | $\mathbf{30}$ |
| CMA-ES evaluations spent | $700$ (50 generations, popsize 14) | $928$ (58 generations, popsize 16) |
| CMA-ES best $J$ reached | $\mathbf{8.1200}$ — **reached the tolerance, at evaluation 144** | $13.1890$ — **never reached the tolerance** |
| **ratio, rollouts** | $\mathbf{8.0\times}$ | $\mathbf{> 30.9\times}$ |
| coordinate finite difference, $2n+1$ per gradient | $73\times$ | $151\times$ |

**The K12 row is a genuine reversal and it goes first.** On the classical column CMA-ES spent 700 evaluations and never reached the target; on this one it reached it in 144 and then kept going, finishing at $J = 8.1200$ — **$15.6\,\%$ above the gradient method's own optimum.** The derivative-free baseline did not merely become competitive at $K=12$; it won on final quality inside the same budget. The ratio falls from a lower bound of $>26.9\times$ to a measured $8.0\times$.

At intermediate tolerances, where both methods arrive:

| fraction of the gradient's gain | K12 gradient | K12 CMA-ES | ratio | K25 gradient | K25 CMA-ES | ratio |
|---|---|---|---|---|---|---|
| 50 % | 6 | 11 | $1.8\times$ | 7 | 203 | $\mathbf{29.0\times}$ |
| 75 % | 10 | 57 | $5.7\times$ | 19 | 308 | $16.2\times$ |
| 90 % | 15 | 96 | $6.4\times$ | 27 | — | $>34.4\times$ |
| 98 % (the tolerance) | 18 | 144 | $\mathbf{8.0\times}$ | 30 | — | $\mathbf{>30.9\times}$ |

**The dimensional trend survives, and it is the load-bearing part of the classical page's argument.** $8.0\times$ at 36 variables against $>30.9\times$ at 75 is the same shape as the classical column's $17.9\times \to 48.3\times$ — the advantage of an adjoint over a population grows with $n$, because an adjoint costs what it costs regardless of $n$ and every derivative-free method pays in $n$. What changed is the **level**, and it dropped by roughly a factor of three at both sizes.

### 4.1 Why the baseline does better here, stated as a hypothesis and not as a result

**[AI Inference]:** the mechanism most consistent with everything else on this page is OP-3. The checkpoint retains about a quarter of the reference solver's centreline wake deficit and only $0.12\times$ of it by sixteen diameters — so on this column a turbine's wake reaches its downstream neighbours much weaker than it should. Turbine–turbine coupling is what makes the layout objective non-separable, and weakening it makes the landscape closer to separable, flatter, and easier for a population method to search. The same mechanism predicts §6's ablation result and §7's, so it is one hypothesis rather than three. **It is not a measurement**, and the experiment that would settle it is a viscosity or wake-strength sweep holding everything else fixed — opened as W119 below.

Two things it is **not**. It is not gradient error: §5 puts the adjoint at $1.4\times10^{-3}$ median relative error against central differences with a textbook curve either side, and a wrong gradient would not produce a monotone 30-step ascent from $2.48$ to $7.02$. And it is not a constraint-handling artefact: both methods get the same projection, and the CMA optimum sits at a minimum spacing of $2.212\,D$ — *off* the packing bound the gradient optimum sits exactly on.

---

## 5. Is the gradient right? — OP-6, and what a float32 expert costs

The check [[atlas-proof-of-concept-1]] §5 asks for, on a real composed rollout, 9 components spanning all three variable kinds, batch layout pinned. **The step size is the finding here, so the sweep is the measurement rather than a single row.**

**K12, 50 macro-steps, 12 windows, at the start layout:**

| $h$ (rotor diameters) | median relative error | cosine |
|---|---|---|
| $3\times10^{-1}$ | $1.291\times10^{-1}$ | $0.98384$ |
| $10^{-1}$ | $6.163\times10^{-2}$ | $0.996833$ |
| $3\times10^{-2}$ | $1.083\times10^{-2}$ | $0.9999418$ |
| $\mathbf{10^{-2}}$ | $\mathbf{1.442\times10^{-3}}$ | $0.9999983$ |
| $\mathbf{3\times10^{-3}}$ | $1.998\times10^{-3}$ | $\mathbf{0.99999947}$ |
| $10^{-3}$ | $3.374\times10^{-3}$ | $0.9999956$ |
| $3\times10^{-4}$ | $1.408\times10^{-2}$ | $0.999958$ |
| $10^{-4}$ | $2.049\times10^{-2}$ | $0.999878$ |

**K25, 40 macro-steps, 24 windows:** minimum at $h = 3\times10^{-3}$, median $1.697\times10^{-3}$, cosine $0.9999958$; $8.181\times10^{-2}$ and $0.99513$ by $h = 10^{-4}$.

**The curve is textbook and the floor is six orders of magnitude higher than the classical column's.** Truncation falls as $h^2$ from $3\times10^{-1}$ down to about $10^{-2}$; cancellation rises below $3\times10^{-3}$. The classical column's minimum is $5.85\times10^{-9}$ at $h = 10^{-5}$; this one's is $1.4\times10^{-3}$ at $h = 10^{-2}$, and the cancellation branch arrives **two and a half decades earlier**. That gap is exactly the float32 forward pass: $J$ carries a relative noise floor near $10^{-7}$, so a central difference divides it by $h$ and a step of $10^{-5}$ would return noise.

**This is the operational meaning of OP-6 for a design loop, and it is worth separating from the version the vault has been carrying.** OP-6 has been quoted as "$\sim10^{-6}$ batch-position dependence, removable by pinning the batch layout." Pinning does remove it *within a machine*: $J$ here is bit-reproducible on both devices, and there is no run-to-run jitter to fight. What pinning cannot remove is that the number itself is only good to about seven digits, because the arithmetic that produced it was. **A pinned batch layout buys reproducibility, not precision**, and the finite-difference instrument needs the second. The classical column could not have discovered this: it had four spare digits at its own optimum.

**Cross-machine, the curve is the same curve** — the desktop CPU run and the A100 run agree on where the minimum is and on its depth:

| $h$ | desktop, median / cosine | A100, median / cosine |
|---|---|---|
| $10^{-1}$ | $6.183\times10^{-2}$ / $0.9968343$ | $6.163\times10^{-2}$ / $0.9968333$ |
| $3\times10^{-2}$ | $1.058\times10^{-2}$ / $0.9999417$ | $1.083\times10^{-2}$ / $0.9999418$ |
| $10^{-2}$ | $1.353\times10^{-3}$ / $0.9999984$ | $1.442\times10^{-3}$ / $0.9999983$ |
| $3\times10^{-3}$ | $1.908\times10^{-3}$ / $0.9999991$ | $1.998\times10^{-3}$ / $0.9999995$ |

---

## 6. Is the *coupling* doing the work? — the ablation

The same experiment as [[poc1-results-differentiable-design]] §6: the same objective at the same design point, differentiated once with the march live and once with the march **frozen** and only the disks' own closure differentiated on the states the live march produced. The second is the gradient a static wake surrogate would give.

| | $\lVert g_{\text{full}}\rVert$ | $\lVert g_{\text{frozen}}\rVert$ | cosine | sign agreement |
|---|---|---|---|---|
| **K12**, all | $0.4432$ | $0.8636$ | $0.4133$ | — |
| — streamwise $x$ | $0.0707$ | $0.7226$ | $\mathbf{0.0410}$ | $58\,\%$ |
| — lateral $y$ | $0.4050$ | $0.4600$ | $0.8211$ | $83\,\%$ |
| — yaw $\gamma$ | $0.1656$ | $0.1103$ | $0.1718$ | $\mathbf{50\,\%}$ |
| **K25**, all | $0.8587$ | $1.2521$ | $0.4107$ | — |
| — streamwise $x$ | $0.1892$ | $0.9337$ | $\mathbf{-0.4583}$ | $\mathbf{12\,\%}$ |
| — lateral $y$ | $0.8093$ | $0.8100$ | $0.7472$ | $88\,\%$ |
| — yaw $\gamma$ | $0.2156$ | $0.1992$ | $0.7606$ | $80\,\%$ |

**Read the aggregate cosine first, and then stop reading it.** At $0.41$ it looks like the frozen-rollout gradient is a much better approximation here than on the classical column ($0.238$ at K12, $0.054$ at K25) — and the direction of that difference is what OP-3 predicts, since a column whose wakes barely reach the next turbine has less for the coupling to contribute. But the aggregate is dominated by the lateral component, whose norm is $4$–$6\times$ the streamwise one and which is well correlated on both columns. **The streamwise component is where the argument lives, and there this column is *worse*, not better:**

> at $K=25$, the frozen-rollout gradient's streamwise part has cosine $\mathbf{-0.458}$ against the truth and its signs agree $\mathbf{12\,\%}$ of the time.

The classical column's sharpest single number was $-0.115$ and $40\,\%$. **This is a four-fold stronger anti-correlation and a sign agreement that is worse than a coin flip by more than a factor of four.** A static wake model given this checkpoint's fields would not merely fail to know which way to move a turbine upstream–downstream; it would be confidently wrong about it seven-eighths of the time. That is the case for putting the seam inside the differentiated path, and it is stronger on the frozen column than on the classical one.

**Yaw behaves differently on the two columns and it is worth flagging rather than smoothing.** On the classical column yaw was the transferable component ($\cos = 0.95$ at K25); here it is $0.76$ at K25 and $0.17$ at K12, with signs agreeing exactly half the time at K12. **[AI Inference]:** at K12 the full gradient's yaw component is small ($0.166$) and the array is laterally symmetric at the start layout, so much of what is being compared is two nearly-zero vectors — the same situation [[poc1-results-differentiable-design]] §5 identified when one component read a relative error of exactly $1.0$. It should not be read as a claim that yaw is *less* transferable on a learned expert without a measurement at an asymmetric design point.

---

## 7. The classical verification panel, and the deployment story measured end to end

[[prior-art-and-novelty-atlas-0.1]] §5's commitment is: **search wide and cheap with the composed model, verify the shortlist with the classical stack.** Until now that has been a plan. This section is the measurement.

### 7.1 The panel: each composed column against the undivided classical solver

`scaling_ladder.reference_monolith` — `RectangularNS` on the whole domain, same cell size, same viscosity, same transmission, **no cut** — marched from the freestream on the same layout, alternating one macro-step each so neither solver is timed against the other's load. Both columns, at both sizes, on the start layout:

| | composed | monolith | difference | worst single cell |
|---|---|---|---|---|
| **K12**, Poseidon column | $2.4782$ | $2.5544$ | $\mathbf{-3.0\,\%}$ | $0.734$ |
| **K12**, classical composed column | $2.3142$ | $2.5544$ | $-9.4\,\%$ | $0.698$ |
| **K25**, Poseidon column | $7.0997$ | $7.8536$ | $\mathbf{-9.6\,\%}$ | $0.575$ |
| **K25**, classical composed column | $6.4379$ | $7.8536$ | $-18.0\,\%$ | $0.437$ |

**At the unoptimised grid the frozen column is *closer* to the undivided solver than the classical composed column is, at both sizes.** That reads backwards and it is not a mistake: the classical composed column pays the full cost of the cut on an expert whose per-window solve is exact, while the checkpoint's own error and the cut's error happen to partially cancel here. It is a coincidence of one operating point, not a property, and §7.2 shows it does not survive optimisation. It is reported because the alternative — quoting only the number that fits the narrative — is what [[case-study-wake-array-atlas-0.1]] §4's W98 lesson is about.

**One asymmetry cannot be removed and is stated rather than worked around.** Against the classical composed column the monolith isolates *the cost of the cut*, because the expert is held fixed. Against the Poseidon column it measures **the expert and the cut together**, and no arrangement of this panel separates them, because **there is no monolithic Poseidon**: the checkpoint is fixed at $128\times128$ and cannot be asked to solve the whole domain at any resolution. That is `poseidon.py`'s founding observation — *"everything Tier 0 measures against a monolithic reference is structurally unavailable, not merely unmeasured"* — arriving in a results table.

### 7.2 The pre-screen, scored by the verifier

Three layouts — the unoptimised grid, the layout the **classical** column's optimiser converged to (the 2026-09-01 artefact), and the layout **this** column's optimiser converged to — each scored three ways. Farm power, same units, same freestream, same horizon, same disks.

**K = 12:**

| layout | Poseidon composed | classical composed | **classical monolith** |
|---|---|---|---|
| unoptimised grid | $2.4782$ | $2.3142$ | $2.5544$ |
| classical column's optimum | $8.0062$ | $9.6080$ | $\mathbf{12.1120}$ — $\mathbf{+374.2\,\%}$ |
| **Poseidon column's optimum** | $7.0243$ | $7.2760$ | $\mathbf{9.3617}$ — $\mathbf{+266.5\,\%}$ |

**K = 25:**

| layout | Poseidon composed | classical composed | **classical monolith** |
|---|---|---|---|
| unoptimised grid | $7.0997$ | $6.4379$ | $7.8536$ |
| classical column's optimum | $14.8735$ | $19.5199$ | $\mathbf{26.6808}$ — $\mathbf{+239.7\,\%}$ |
| **Poseidon column's optimum** | $14.6951$ | $15.2755$ | $\mathbf{21.2022}$ — $\mathbf{+170.0\,\%}$ |

**This is the sharpest result on the page, and it has two halves.**

> **The pre-screen works.** A layout found entirely inside the frozen-checkpoint column, by an optimiser that never once called a classical solver, is worth **$+266\,\%$ at K12 and $+170\,\%$ at K25 when the undivided classical solver scores it.** Those are real gains against a real verifier, not gains measured by the model that proposed them.
>
> **And it leaves about $30\,\%$ of the gain on the table.** The classical column's layout is worth $+374\,\%$ and $+240\,\%$ under the same verifier. The frozen column captures $\mathbf{71.2\,\%}$ of the classical column's verified gain at K12 and $\mathbf{70.9\,\%}$ at K25 — the same fraction at both sizes, to within a fifth of a percentage point.

That two independent problem sizes give $71\,\%$ and $71\,\%$ is more informative than either number alone. **[AI Inference]:** a constant fraction across a factor of two in design dimension is what one would expect if the deficit comes from a systematic property of the expert — OP-3's over-dissipation moving the optimum by a roughly fixed relative amount — rather than from search failure, which would be expected to worsen with dimension. If that reading holds, the quantity to improve is the checkpoint, not the optimiser, and [[expert-donor-survey]]'s permissively-licensed alternatives (Walrus, DPOT, GPhyT) become a measurable rather than a preference. The experiment is to re-run §7.2 with a second donor; opened as W120.

**And the honest framing that comes with it.** [[composition-error-theory]]'s position — the composed model is a search instrument, not a verifier — is now a measured pipeline rather than a slogan: the instrument found something real, the verifier priced it, and the price was 29 % of the available gain. Whether that trade is worth taking depends on the cost ratio, which is §8.

---

## 8. What it costs

**The forward comparison, on one machine, uncontended, alternating one macro-step each** (22-core Core Ultra 7 155H, CPU only, no GPU):

| | composed Poseidon column | undivided classical monolith | ratio |
|---|---|---|---|
| K12, $464\times352$ cells, 12 windows | $\mathbf{763}$ ms / macro-step | $2404$ ms | $\mathbf{3.15\times}$ **faster** |
| K25, $688\times464$ cells, 24 windows | $\mathbf{1774}$ ms | $6453$ ms | $\mathbf{3.64\times}$ **faster** |

**This is the first measurement in this vault in which the composed graph is faster forward than the monolith, and the reason is the expert.** [[atlas-proof-of-concept-1]] §10 measured the *classical* composed column at $1.54\times$ **slower** than the monolith on this box, and [[poc1-retrospective-and-hybrid-roadmap]] §1 found a second machine agreeing on the sign: *"splitting an all-classical NS solve into windows costs more than solving it whole, at this graph size."* That finding stands, unchanged — it is about the cut, with the expert held fixed. What this row adds is the other term: **once the thing inside each window is a batched forward pass instead of a stencil sweep, the decomposition stops being a tax and starts paying.** It is [[tier0-measurements]] §19.10's crossover — classical wins at $N=1$, Poseidon wins by $3.96\times$ at $N=24$ — arriving at the level of a whole composed rollout rather than a per-agent microbenchmark, and it is the first time the two halves of [[poc1-retrospective-and-hybrid-roadmap]]'s §2 argument have been measured in the same experiment.

Every millisecond above is one machine in one power state. What should transfer is the **ratio**, and §10 of [[atlas-proof-of-concept-1]] is the standing warning about why.

**The adjoint's own cost, measured on both machines:**

| | forward, s/macro-step | gradient, s/macro-step | gradient in forward-equivalents |
|---|---|---|---|
| desktop, 22 cores, no GPU — K12 | $0.679$ | $2.790$ | $\mathbf{4.11\times}$ |
| the same — K25 | $1.067$ | $3.187$ | $2.99\times$ |
| one rented **A100 SXM4** — K12 | $0.0530$ | $0.2360$ | $4.45\times$ |
| the same — K25 | $0.0546$ | $0.2228$ | $4.08\times$ |

**A gradient costs three to four and a half forward evaluations** on the frozen column, against six to nine on the classical one, and against $73$ and $151$ for a coordinate finite difference. The A100 is $12.8\times$ the desktop at K12 and $19.5\times$ at K25 — against the classical column's $5$–$6\times$, and for the obvious reason: a float32 transformer forward pass is what a tensor-core GPU is for, where a float64 stencil is not. One K12 gradient rollout is $11.8$ s on the A100 and $140$ s on the desktop.

**The whole experiment** — both gradient runs, both CMA-ES runs, two 70-step confirmation marches, both finite-difference sweeps, both ablations, four verification panels, both cross-evaluations and a sensitivity sweep — cost **about 65 minutes on one A100 and roughly \$1.15**.

---

## 9. How converged is any of this?

Same three questions as [[poc1-results-differentiable-design]] §7, same honesty.

- **Quasi-steady, not converged.** Relative change in farm power over the last five macro-steps at the optimum: $2.10\times10^{-2}$ (K12, 50 steps) and $2.44\times10^{-2}$ (K25, 40 steps).
- **March-length sensitivity**, K12, at the same two layouts:

  | macro-steps | 40 | 50 | 60 |
  |---|---|---|---|
  | gain | $+166.7\,\%$ | $+183.4\,\%$ | $+182.1\,\%$ |
  | settling at the optimum | $5.9\times10^{-3}$ | $2.1\times10^{-2}$ | $2.3\times10^{-2}$ |

  Stable within its own transient scatter from 50 on, and $9\,\%$ lower at 40 — a shallower dependence than the classical column's $24\,\%$ drop at 30, which is again what a column with weaker wakes would give.
- **Neither optimiser converged, and at K12 the gradient one is the further from converged of the two.** The gradient trace was still rising at its budget ($+0.036$ at K12, $+0.178$ at K25), and CMA-ES overtook it. On the classical column the same fact made the ratios *conservative*; here it does not, and §4.1 says so rather than reusing the sentence.

**What the optimiser did:**

| | K = 12 | K = 25 |
|---|---|---|
| min spacing at the optimum | $2.000\,D$ — constraint **active** | $2.000\,D$ — **active** |
| yaw, mean $\lvert\gamma\rvert$ | $5.3^\circ$ | $6.5^\circ$ |
| yaw range | $-5.5^\circ$ to $+20.6^\circ$ | $-20.8^\circ$ to $+14.8^\circ$ |
| yaw at the $\pm30^\circ$ bound | $0/12$ | $0/25$ |
| position range reached | $x\in[1.50,12.50]$, $y\in[1.50,9.50]$ | $x\in[2.81,19.33]$, $y\in[1.50,13.00]$ |

Re-siting dominates wake steering here too, and both optima sit on the spacing bound and the box — the same caveat, for the same reason, as the classical column's.

**The visual.** `out/w118/fields_K12.png`, `out/w118/fields_K25.png` — before/after streamwise velocity on a shared colour ramp; `out/w118/traces_K12.png`, `out/w118/traces_K25.png` — the farm-power trace, the evaluations-to-tolerance comparison on a log axis, and the yaw angles converging. The yaw panel is a genuine convergence plot on this run: the serialisation defect [[poc1-results-differentiable-design]] §4 records is fixed and the whole design trajectory is in the artefact.

---

## 10. What this does not claim

- **Not accuracy, and now with a number attached.** §7.2 prices the frozen column's optimum at $71\,\%$ of the classical column's verified gain. It is a pre-screen.
- **Not a certified composition.** `L2/R10` still refuses this decomposition, for W93's reason, and §3's stability result does not touch that. A column that runs is not a column that is certified to run.
- **Not two to three orders of magnitude.** $8.0\times$ at 36 design variables and $>30.9\times$ at 75 — one order, with the dimensional trend that points at more, and a level about three times below the classical column's.
- **Not a scaling result and not multiphysics.** One governing family, and the seams still separate identical physics — [[poc1-retrospective-and-hybrid-roadmap]] §1's critique applies to this run word for word.
- **Not a converged optimum**, at either size, and at K12 the derivative-free baseline found a better one.
- **Not a commercial artefact.** Poseidon-T's weights are **CC-BY-NC-4.0** ([[expert-donor-survey]]). Every number on this page is research use, and the frozen-expert claim as demonstrated here cannot be shipped with this checkpoint in it.

What it does claim, and this is the whole of it: **reverse-mode differentiation of a design objective back through a 40–50 macro-step rollout of a spatially-partitioned graph of frozen, independently-pretrained neural experts — 20.8 M parameters per window, twelve to twenty-four windows, the adjoint through every seam and every fluid–disk closure — is implementable without touching the assembly, is stable over the horizon it is used at, is correct against central differences to the precision a float32 checkpoint allows, costs three to four forward evaluations, and finds in 18–30 rollouts a layout that a classical verifier scores $+266\,\%$ and $+170\,\%$ over the unoptimised grid.**

---

## 11. What this opens

New rows for [[gap-worklist]] (next free: **W119**):

| # | what | why |
|---|---|---|
| **W119** | **Is the baseline's advantage at K12 caused by OP-3?** §4.1's hypothesis is that weaker wakes make the layout objective closer to separable and therefore easier for a population method. Sweep the wake strength — the reference solver's viscosity, or the checkpoint against a coarser macro-step — holding the optimiser, the budget and the constraints fixed, and see whether the gradient/CMA ratio tracks it. | It is the only [AI Inference] on this page that three separate measurements depend on (§4.1, §6, §7.2), and it is cheap. |
| **W120** | **Re-run §7.2 with a second donor.** The $71\,\%$ capture fraction is constant across a factor of two in design dimension, which points at the expert rather than the search. Walrus (MIT), DPOT-Ti (Apache-2.0) and GPhyT (MIT) are permissively licensed, so this also removes the CC-BY-NC problem from the result. | Turns "a better expert is worth having" into a number, and [[expert-donor-survey]]'s licence finding into a decision. |
| **W121** | **A pinned batch layout buys reproducibility, not precision.** §5 measures the finite-difference floor at $1.4\times10^{-3}$ against the classical column's $5.9\times10^{-9}$, and traces it to the float32 forward pass rather than to batch position. OP-6's entry should carry that distinction, and any future probe-step derivation that cites OP-6's $10^{-6}$ should cite this instead. | The probe-step reasoning in [[probed-dtn-coupling]] §5 rests on the removable version of OP-6; half of it is not removable. |
| **W122** | **Yaw's transferability differs between the two columns** ($\cos = 0.95$ classical vs $0.76$ / $0.17$ here) and at K12 the comparison is between two near-zero vectors on a laterally symmetric layout. Re-run the ablation at an asymmetric design point. | §6 flags it as unsafe to read; a measurement would make it readable. |

Already closed by this run: [[poc1-results-differentiable-design]] §10's item 1, *"the frozen-checkpoint run"*.

---

## See Also

- [[poc1-results-differentiable-design]] — the classical column this reproduces, number for number; its §0 is what this page closes
- [[atlas-proof-of-concept-1]] — the specification both runs execute
- [[prior-art-and-novelty-atlas-0.1]] — §2.1's frozen-neural-subsolver half, exercised here for the first time; §5's deployment story, measured in §7.2
- [[case-study-wake-array-atlas-0.1]] — W93, W98 and W99, all three of which shape §1's macro-step
- [[open-problems-atlas-0.1]] — OP-3 behind §4.1, §6 and §7.2; OP-6 behind §5
- [[expert-donor-survey]] — why this checkpoint is a research-only donor, and what W120 would swap in
- [[composition-error-theory]] — the search-instrument-not-verifier position, now a measured pipeline
- [[poc1-retrospective-and-hybrid-roadmap]] — §1's "no neural network in it anywhere", which no longer applies to this column; §2.2's crossover, which §8 measures end to end
