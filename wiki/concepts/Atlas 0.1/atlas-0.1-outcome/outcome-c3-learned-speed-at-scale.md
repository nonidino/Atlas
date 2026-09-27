# Outcome C3 — are learned experts faster than classical ones when the graph gets large?

**Type:** Outcome page — **claim audit** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** compiled 2026-09-26 from existing records; no new measurement.
**Verdict:** **Demonstrated on one governing family, with conditions.** Per agent, the learned expert becomes cheaper than a classical window from about twelve windows up, and a composed learned column beat the undivided classical solver on the machine it was measured on. **Two competitors beat it:** a coarse classical solver is cheaper still, and on the car the learned column cannot be run at the step the composition layer asks for.
**Hub:** [[00-atlas-0.1-outcome]]
**Sources:** [[learned-contribution-kill-tests]] §2 · [[case-study-scaling-ladder-atlas-0.1]] §7 · [[tier0-measurements]] §19.10 · [[poc1a-frozen-expert-results]] §4, §8 · [[atlas-proof-of-concept-1]] §11 · [[case-study-racelab-switch-atlas-0.1]] · [[defect-correction-learned-operator]] §8

---

## 1. Why the learned side should win at scale

**Intuition.** A classical window is a stencil sweep. Its work per window is constant, but many windows spill out of cache, so its time per window *rises* as the graph grows. A neural operator runs many windows as one batched forward pass on dense matrix hardware, so its time per window *falls* as the batch fills. Somewhere the lines cross.

**Formally**, with $c_C(N)$ and $c_L(N)$ the per-agent, per-macro-step cost of the classical and learned window in a graph of $N$ windows, the crossover is the $N^\ast$ with

$$\frac{c_L(N^\ast)}{c_C(N^\ast)} = 1 .$$

What matters for a whole model is the cost of a composed macro-step against the cheapest alternative that gives an answer of acceptable quality. That second clause is where the result gets conditional.

---

## 2. The crossover, measured

**Per agent, one process, the dev laptop** ([[learned-contribution-kill-tests]] §2.1, on the wake-array geometry with $128$-cell windows):

| windows $N$ | classical `WindowNS`, ms/agent/step | Poseidon-T, ms/agent/step | learned / classical |
|---|---|---|---|
| $1$ | $81.5$ | $399.7$ | $4.90$ |
| $2$ | $88.8$ | $270.1$ | $3.04$ |
| $6$ | $125.5$ | $154.4$ | $1.23$ |
| $12$ | $329.1$ | $109.8$ | $\mathbf{0.334}$ |
| $24$ | $340.8$ | $86.1$ | $\mathbf{0.253}$ |

From $N=2$ to $N=24$ the checkpoint gets $3.1\times$ **cheaper** per agent and the classical window $3.84\times$ **dearer**. The CFL sub-step count is identical at every rung, so the classical rise is the memory hierarchy, not extra work ([[case-study-scaling-ladder-atlas-0.1]] §7). The crossover $N^\ast$ lies between $6$ and $12$ windows.

**Against one monolith macro-step, same process** ([[learned-contribution-kill-tests]] §2.3):

| windows | learned composed column | classical composed column | composition layer alone | $2\times$-coarse classical monolith |
|---|---|---|---|---|
| $2$ | $3.43$ | $0.932$ | $0.110$ | $0.197$ |
| $6$ | $2.01$ | $1.31$ | $0.149$ | $0.132$ |
| $12$ | $\mathbf{0.296}$ | $1.02$ | $0.0365$ | $\mathbf{0.0320}$ |

---

## 3. A whole composed rollout, faster than the undivided solver

**PoC 1a, the frozen-expert wind farm** ([[poc1a-frozen-expert-results]] §8), CPU only, uncontended, alternating one macro-step each:

| | composed learned column | undivided classical monolith | ratio |
|---|---|---|---|
| K12: $464\times352$ cells, 12 windows | $763$ ms/step | $2404$ ms/step | **$3.15\times$ faster** |
| K25: $688\times464$ cells, 24 windows | $1774$ ms/step | $6453$ ms/step | **$3.64\times$ faster** |

This was the first measurement in the vault in which a composed graph was faster forward than the monolith.

**The ratio is a property of the hardware.** The same comparison read $2.1\times$ to $3.4\times$ on the same box on one evening. It read $1.07\times$ on one rented Linux box and $\mathbf{0.43\times}$ (the monolith faster) on another, a share of a 192-core EPYC ([[atlas-proof-of-concept-1]] §11). That page also found the **classical** composed column, with its elliptic part exposed, running at $3.98\times$ the monolith on the dev box at K12, faster than the learned column. So "the reason is the expert" is not separated from "the reason is the cut" by the published pair.

**The design loop, where speed compounds.** Gradients through the learned column cost $3$–$4.5$ forward evaluations. On one A100 they ran $12.8\times$ (K12) to $19.5\times$ (K25) faster than the desktop CPU, where the classical column gained only $5$–$6\times$ from the same card. Reaching the tolerance took the gradient $18$ and $30$ rollouts against CMA-ES's $144$ and more than $928$: **$8.0\times$ and more than $30.9\times$**. The whole experiment cost about 65 A100-minutes, roughly \$1.15.

---

## 4. The conditions, and the competitors that win

### 4.1 A coarse classical solver is cheaper still

In the same ledger, a classical monolith on a grid twice as coarse costs $0.197$, $0.132$ and $0.0320$ of a full step: **$17.4\times$, $15.2\times$ and $9.2\times$ cheaper than the learned column** at 2, 6 and 12 windows. The gap narrows as the graph grows, and at twelve windows it is still an order of magnitude. Inside defect correction at six windows, this coarse solver reached the certified state at $52$ classical-equivalents where the learned column cost $314$ ([[defect-correction-learned-operator]] §8.2).

**Accuracy per unit cost.** At twelve windows, 120 steps took the learned column $112$ s against $419$ s (classical composed) and $438$ s (monolith), about a quarter of the wall time. It ended $1.5\times$ further from the monolith ($0.121$ against $0.0830$ rms) ([[learned-contribution-kill-tests]] §2.2).

### 4.2 On the car, the learned column cannot be asked for the step it is given

RaceLab's porous column has 14 fluid windows ([[case-study-racelab-switch-atlas-0.1]]). One $128$-cell window spans $2.0$ length units by geometry, which fixes the checkpoint's time scaling. The exchange the composition layer runs at is then $\tfrac{1}{32}$ of the checkpoint's native lead, and its one-step error grows as the step **shrinks**:

| lead / native | $\tfrac18$ | $\tfrac14$ | $\tfrac12$ | $1$ | $2$ |
|---|---|---|---|---|---|
| per-window median error | $0.3607$ | $0.5615$ | $0.7301$ | $\mathbf{0.1708}$ | $0.2408$ |

- **In the march**, an all-learned car runs at **$0.067\times$** the classical speed, with an rms field error of $3.84$. It is outside a declared envelope for $35$ of $40$ macro-steps.
- **Per call, at the native lead**, each column at its own best thread count takes $0.5216$ s (classical) and $0.4526$ s (learned): a ratio of $1.15$. Two independent draws of that ratio differ by $20\%$, so the page reads it as **even**, not faster.

---

## 5. What C3 can honestly claim

> *On 2-D incompressible flow, a frozen neural-operator window becomes cheaper per agent than a classical window between six and twelve windows, and $4\times$ cheaper at twenty-four. A composed rollout of such windows ran $3.2$–$3.6\times$ faster than the undivided classical solver on the machine it was measured on. Differentiating through it for design costs 3–4.5 forward passes, and gains most from GPUs. The advantage depends on hardware, is matched or beaten by a coarse classical solver, and disappears when the learned expert is run away from its native time step.*

**[AI Inference]:** the first two sentences are the scaling argument the proposal wants. The last sentence says what a better expert must fix: its native step must be set by the composition layer's cadence, and it must beat a $2\times$-coarse classical solver per unit accuracy, not merely a full-resolution one. Both are requirements in [[outcome-c5-requirements-for-dd-native-experts]].

---

## 6. What was NOT done, named

1. **One family, one host class.** Every crossover number is 2-D incompressible flow with Poseidon-T and `WindowNS`. None is multiphysics, none 3-D.
2. **The coarse competitor across a coupled seam, measured at Tier 89** ([[matched-shrink-and-coarse-competitor]] §2–§4). On CS-13 and CS-12 it is correct, with every arm inside its certificate. It is **not cheap**: counting its own cost it is $1.11\times$ and $1.24\times$ dearer than the cold classical march. On CS-13 the coarse solve costs $0.96$ of a fine one, because the solver is assembly-bound at 343 nodes; on CS-12 the coarse map stalls the outer iteration. The seam-frozen controls behave the same, so the seam is not the cause. **So at these seams the bar a learned expert must clear is the cold march, $7411$ and $901$ classical calls, not the $52$ it faced on the wake array.** CS-14 has no field to coarsen.
3. **The exposed classical column's $3.98\times$ has no worklist row**, and no controlled comparison separates the cut from the expert.
4. **No timing on accelerators for the classical side at matched effort.** The A100 numbers compare each column against its own CPU run, not against a GPU-tuned classical solver.
5. **The rocket and the body-fitted car have no learned column at all**, so no speed comparison exists there.

---

## See Also

- [[00-atlas-0.1-outcome]] · [[outcome-c2-learned-experts-in-the-loop]] · [[outcome-c5-requirements-for-dd-native-experts]] · [[outcome-evidence-ledger]]
- [[atlas-proof-of-concept-1]] §11 — why the demo re-measures its speed ratio on the viewer's own machine rather than quoting one
- [[poc1-retrospective-and-hybrid-roadmap]] — the cross-machine sign of the classical cut's cost
- [[vast-ai-windfarm-runbook]] — why the classical column is memory-bandwidth-bound in float64
