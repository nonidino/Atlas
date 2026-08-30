# Wind Farm — the $N$-Turbine Scaling Sweep

**Type:** Results page (folder: Atlas 0.1 / case-study-wind-farm-wake)
**Status:** Measured 2026-08-25 on a rented RTX 5090. $N \in \{2,3,5,8\}$, solver expert + open-face fix, $t=20$, $\Delta t=0.25$.
**Code:** branch `atlas-0.1-windfarm`. `scripts/windfarm_n_sweep.py`, `src/atlas/cases/windfarm/geometry.py`. Raw: `results/windfarm/n_sweep/`.
**Related:** [[results-w6-w11-wind-farm]] (the $N=2$ answer this extends), [[impl-wind-farm-guide]], [[open-problems-atlas-0.1]], [[vast-ai-windfarm-runbook]]

> This is the item [[results-w6-w11-wind-farm]] §6 called "the largest open piece and the one most directly about the *framework* rather than this case study." It is now closed.

---

## 0. The headline

**The composition layer does not degrade as the agent graph grows.** Going from $N=2$ to $N=8$ takes the graph from 8 agents / 15 declared interfaces to **26 agents / 63 interfaces**, and the second turbine's power ratio moves by less than $1\%$:

| $N$ | agents | interfaces | tiles | $P_2/P_1$ | wall |
|---|---|---|---|---|---|
| 2 | 8 | 15 | 124 | $0.7663$ | $3.3$ min |
| 3 | 11 | 23 | 158 | $0.7588$ | $3.8$ min |
| 5 | 17 | 39 | 234 | $0.7581$ | $5.0$ min |
| 8 | 26 | 63 | 344 | $0.7584$ | $8.0$ min |

Every run finite, every geometry gate W1 passing, no divergence anywhere. **Composition error is bounded in graph size, not accumulating in it** — the single most load-bearing framework claim this case study can make, and it is now measured rather than hoped.

**And the farm-level error still grows with $N$.** Those two statements are not in tension, and §3 is about why.

---

## 1. The two curves

Per-turbine $P_i/P_1$, framework (solver expert + open-face fix) against the classical undivided 2-D reference (`ChannelNS`) on the *same* domain with the *same* disks:

| | $T_1$ | $T_2$ | $T_3$ | $T_4$ | $T_5$ | $T_6$ | $T_7$ | $T_8$ |
|---|---|---|---|---|---|---|---|---|
| **framework** $N{=}8$ | $1.0$ | $0.7584$ | $0.7396$ | $0.7416$ | $0.7408$ | $0.7407$ | $0.7406$ | $0.7512$ |
| **classical** $N{=}8$ | $1.0$ | $0.4033$ | $0.3422$ | $0.3438$ | $0.3437$ | $0.3437$ | $0.3437$ | $0.3446$ |

**Both curves have the same three-part shape**, and this is worth stating before the disagreement: a large drop at $T_2$, a **plateau** from $T_3$ onward, and a small upturn at the last turbine. That structure — the "deep-array" asymptote, where a turbine deep in a farm sees an equilibrium wake rather than a progressively worse one — is real wind-farm phenomenology, and the framework reproduces it qualitatively without being told to.

**The trailing upturn is not a framework artefact.** It appears in the classical reference too ($0.3437 \to 0.3446$ at $T_8$), which was the cheap check worth running before attributing it to the composition layer. It is a property of the shared domain construction — the last turbine is the one whose downstream neighbour is the outlet rather than another rotor.

---

## 2. Where they disagree, and by how much

$$\text{array efficiency} \quad \eta_N \;=\; \frac{1}{N}\sum_{i=1}^{N} \frac{P_i}{P_1}$$

| $N$ | $\eta$ framework | $\eta$ classical | overprediction | plateau (framework) | plateau (classical) |
|---|---|---|---|---|---|
| 2 | $0.8832$ | $0.7022$ | $1.258\times$ | $0.7663$ | $0.4043$ |
| 3 | $0.8364$ | $0.5821$ | $1.437\times$ | $0.7546$ | $0.3732$ |
| 5 | $0.7978$ | $0.4868$ | $1.639\times$ | $0.7401$ | $0.3430$ |
| 8 | $\mathbf{0.7766}$ | $\mathbf{0.4331}$ | $\mathbf{1.793\times}$ | $0.7407$ | $0.3434$ |

At $N=8$ the framework **overpredicts farm output by 79%**. In deficit terms, at the plateau it retains

$$\frac{1 - 0.7407}{1 - 0.3434} \;=\; \frac{0.2593}{0.6566} \;=\; \mathbf{0.395}$$

— **under 40% of the momentum deficit the reference sustains.** This is [[open-problems-atlas-0.1]] OP-3 measured at farm scale, and it agrees in direction and rough magnitude with W8's centreline figure of $0.62\times$ for the solver at $N=2$; the deep-array number is worse because the plateau is precisely where a too-fast-recovering wake has the most room to be wrong.

---

## 3. Why per-turbine error is flat but farm error grows

These are the same measurement read two ways, and conflating them would be the easy mistake.

**Per-turbine, the error is $N$-independent.** $T_2$ reads $0.7663 / 0.7588 / 0.7581 / 0.7584$ across $N = 2/3/5/8$ while the classical reference reads $0.4043 / 0.4033 / 0.4033 / 0.4033$. Both are flat; the offset between them is flat. Adding eleven agents and forty-eight interfaces does not make any individual turbine's answer worse. **Nothing compounds through the graph.**

**At farm level, the error grows anyway** — $1.258\times \to 1.793\times$ — because the *plateau*, where the framework is badly wrong, occupies a growing fraction of the array. At $N=2$ exactly one turbine sits in the wrong regime; at $N=8$, seven do. The farm-level divergence is **composition of a fixed per-unit error over more units**, not degradation of the composition itself.

This is the distinction the sweep existed to draw, and it lands on the useful side for the framework and the damaging side for the expert:

> The interface machinery scales. The physics inside it does not improve, and at farm scale that is what dominates.

---

## 4. What was verified before any of the above was believed

The sweep required parametrising a graph that had been hand-written for exactly two turbines — `geometry.py`'s eight `AgentRegion`s, fifteen `Interface`s, priority ordering and domain extent. A generalisation that quietly changed $N=2$ would have invalidated [[results-w6-w11-wind-farm]] rather than extended it, so $N=2$ was pinned three independent ways:

1. **Structural.** The generated $N=2$ graph is the original graph — 8 agents, 15 interfaces, domain $[-6,18]\times[-4,4]$ — and `verify()` (gate W1) passes at import, as does W1 for every $N$ in the sweep.
2. **Bit-for-bit.** Old code at `456ec7e` against new code, same short rollout, same arguments: `max|diff| = 0.0` on every saved array including the full $512\times1536$ $u$ and $v$ fields and `thrust`. **Zero numerical drift**, not "within tolerance."
3. **End-to-end at thesis scale.** $t=20$ on GPU through the refactored path returns $P_2/P_1 = \mathbf{0.7663}$ — the committed W10 number to four decimals.

Full suite: **606 passed, 0 failed.**

**One real defect was caught by insisting on (3) rather than trusting (1) and (2).** `CoupleConfig.force_mode` defaults to `'impulse'`, but the run that produced $0.7663$ was launched with `--force-mode rhs`. The sweep script did not set it. Every $N$ would still have produced a plausible, internally consistent, monotone-looking curve — of a configuration that is not the one the case study's result is about. It was found because the $N=2$ row is a *guard* and not merely the first data point, and it is the reason that row is run every time rather than assumed.

---

## 5. Compute

| item | where | wall |
|---|---|---|
| $N=2$ regression, $t=20$ | RTX 5090 | $4.3$ min |
| sweep $N = 2,3,5,8$, $t=20$ | RTX 5090 | $20.1$ min |
| classical references, all four $N$ | CPU (same box, parallel) | $0.6$ min total |

Total rental $\approx 40$ min at \$0.35/hr, **about \$0.23**. The classical reference is a single explicit-projection grid and turns out to be nearly free — it was run *concurrently* with the GPU sweep rather than after it, which is worth remembering: the reference is not the expensive half of any comparison in this case study.

Source reached the box as a `git archive` over `scp`; no GitHub credential was placed on rented hardware. Instance destroyed on completion.

---

## 6. What is not done

- **The frozen-expert path was not swept.** Only the solver + open-face-fix configuration was run. The frozen checkpoint is a known separate failure mode ($P_2/P_1 = 1.0162$, W10 fail) whose cause is already attributed in [[results-w6-w11-wind-farm]], and re-confirming it per $N$ would buy nothing. **So this page says nothing about how a frozen expert scales**, and the §0 headline claim is about the composition layer *carrying a solver*.
- **Spacing is fixed at $7D$.** The generator takes `turbine_spacing`, but only the default was measured.
- **Single row, aligned, uniform inflow.** No lateral offset, no yaw, no staggering — all of which are where real array-efficiency questions live.
- **[AI Inference]:** the flat per-turbine error suggests the port exchange is behaving as a *local* operation whose cost does not propagate — consistent with W9's finding that composition error ($0.3293$) is dominated by expert-plus-tiling error ($1.2445$). If that reading is right, the same graph carrying a *better* expert should move the farm-level number substantially while leaving the scaling flatness intact. Untested: no such expert was run, and the one available frozen checkpoint cannot accept the boundary treatment this configuration depends on.

---

## See Also
- [[results-w6-w11-wind-farm]] — the $N=2$ answer, and the open item this page closes
- [[results-w0-w3-wind-farm]] — the earlier gates
- [[impl-wind-farm-guide]] — the gate definitions and §9's reporting rules
- [[open-problems-atlas-0.1]] — OP-3 (dissipation), now measured at farm scale
- [[vast-ai-windfarm-runbook]] — how GPU runs are produced
- [[case-study-wind-farm-wake-2d-atlas-0.1]] — the case study
- [[schwarz-iteration-atlas-0.1]] — the fixed-point layer that stayed at 3 iterations throughout
