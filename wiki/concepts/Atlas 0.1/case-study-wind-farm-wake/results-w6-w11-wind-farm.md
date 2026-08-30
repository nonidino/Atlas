# Wind Farm — Results W6–W11

**Type:** Results page (folder: Atlas 0.1 / case-study-wind-farm-wake)
**Status:** Measured 2026-08-24 on a rented RTX 5090. Every number below is from a $t=20$ rollout at $\Delta t = 0.25$, 124 tiles, unless stated.
**Code:** branch `atlas-0.1-windfarm`. `scripts/windfarm_schwarz.py`, `windfarm_w9_compare.py`, `windfarm_w8_gate.py`. Raw: `results/windfarm/gpu/`.
**Related:** [[impl-wind-farm-guide]] (the gates), [[wind-farm-implementation-log]] (how each was reached), [[open-problems-atlas-0.1]], [[results-w0-w3-wind-farm]], [[vast-ai-windfarm-runbook]]

> Reporting follows [[impl-wind-farm-guide]] §9. **§9.2 in particular:** 2-D turbulence has an inverse cascade and no vortex stretching, so W8 and W10 are *corridors and sanity bands, never wake-model accuracy*. §9.2 is not a disclaimer here — it changed a verdict. See W8.

---

## 0. The headline

**The question this case study exists to answer** is whether composing frozen pretrained experts through declared ports preserves physical validity. The answer, measured:

| configuration | $P_2/P_1$ | induction at $x{=}{-}1$ | wake retained vs 2-D reference |
|---|---|---|---|
| classical 2-D reference (`ChannelNS`, undivided) | $0.404$ | $+0.0473$ | $1.00\times$ |
| **solver expert + open-face fix** | $\mathbf{0.7663}$ | $\mathbf{+0.0389}$ | $0.62\times$ |
| solver expert, periodic windows | $0.8373$ | — | — |
| **frozen checkpoint** (best available config) | $\mathbf{1.0162}$ | $+0.2629$ | $0.26\times$ |

**The frozen checkpoint inverts the array efficiency**: turbine 2 out-produces turbine 1, against a reference where it produces 40% as much. That is a negative answer, and it is the result.

**The architecture is not what breaks it.** With the expert replaced by a solver and the 2026-08-24 open-face fix applied, $P_2/P_1$ lands at $0.7663$ — inside gate W10's band — and the upstream induction comes within 18% of the reference with the correct sign.

**The two rows cannot be made to differ in one variable, and that is itself the finding.** The frozen row is also periodic-windowed and impulse-forced. Those are not choices: a one-shot field-in/field-out operator has no channel through which a boundary ring could be imposed, so the fix that produced the middle row is *structurally unavailable* to a frozen expert. `CoupleConfig` refuses the combination rather than silently ignoring it.

---

## 1. Gate table

| ID | Criterion | Verdict | Number |
|---|---|---|---|
| W6 | $C_P \le 16/27$ at both turbines, every step | **pass** | canary never fired |
| W7 | mirror residual $< 10^{-6}$ | **pass** | $2.9\times10^{-7}$ via [[symmetry-averaging-atlas-0.1]] |
| W8 | self-similar Gaussian far wake | **fail (shape)** | collapse spread $0.127$ solver, $0.350$ frozen; reference $0.033$ |
| W8 | inside the Jensen/BPA corridor | **not a valid test in 2-D** | the *reference itself* scores $0/6$ — see below |
| W9 | composition error below the expert's single-step error | **fail** | $0.3293$ vs $0.0055$ |
| W10 | $P_2/P_1 \in [0.4, 0.8]$ | **pass** (solver + fix) / **fail** (frozen) | $0.7663$ / $1.0162$ |
| W11 | power residual $< 1\%$ | **not measurable** | see [[open-problems-atlas-0.1]] OP-4 |

---

## 2. W8 — and the check that changed the verdict

The gate has two halves and they fail independently, so they are reported independently.

**The corridor half does not discriminate in 2-D, and this was measured rather than argued.** Running the same gate on the classical 2-D reference — the known-correct answer — puts it **$0/6$ stations inside the Jensen corridor**, because a 2-D wake recovers *slower* than any 3-D model allows. Grading the system against a corridor the right answer also fails would have produced a "W8 FAIL" that carried no information. §9.2 predicted exactly this; the cheap check confirmed it rather than trusting the prediction.

Replacing the 3-D corridor with the 2-D reference gives a test that does discriminate:

| $x$ | reference deficit | solver + fix | frozen |
|---|---|---|---|
| $9$ | $0.4957$ | $0.5485$ ($1.11\times$) | $0.2144$ ($0.43\times$) |
| $12$ | $0.3754$ | $0.1977$ ($0.53\times$) | $0.0958$ ($0.26\times$) |
| $16$ | $0.2513$ | $0.0765$ ($0.30\times$) | $0.0290$ ($0.12\times$) |
| **mean** | $1.00\times$ | $\mathbf{0.62\times}$ | $\mathbf{0.26\times}$ |

**The frozen checkpoint destroys roughly three quarters of the wake the reference keeps**, and by $x=16$ it has kept an eighth of it. That is OP-3, now quantified against a valid reference instead of a per-call estimate.

**The shape half does discriminate**, and the reference passes it: collapsed lateral profiles lie on one Gaussian to $0.033$, against $0.127$ (solver) and $0.350$ (frozen). So the far wake in both system configurations is not merely too shallow — it is the wrong shape.

---

## 3. W9 — the decomposition, and what it does and does not separate

Three runs (guide §8's gold standard), all frozen expert, $t=20$:

| run | what it is | $\langle U_d\rangle_1$ | $\langle U_d\rangle_2$ | $P_2/P_1$ |
|---|---|---|---|---|
| A | 8 agents, 15 declared ports | $0.6307$ | $0.6341$ | $1.016$ |
| B | 1 agent, same tiles, no ports | $0.6505$ | $0.6400$ | $0.952$ |
| C | classical 2-D NS, undivided | $0.7894$ | $0.5837$ | $0.404$ |

Relative $L^2$ in $u$, normalised by the perturbation from freestream:

$$A-B = 0.3293 \quad\text{(composition)}\qquad B-C = 1.2445 \quad\text{(expert + tiling)}\qquad A-C = 1.3875 \quad\text{(total)}$$

**Gate W9 fails**: composition error $0.3293$ against the expert's own single-step error of $0.0055$ — declaring agents and exchanging ports costs sixty times what one expert call costs. That is a result about the framework, not a bug to tune away.

**What this does not separate.** $B$ keeps the tiling, so $B-C$ is *expert and windowing together*. It is not evidence for or against OP-5's earlier $66\%$ attribution to windowing, and it should not be quoted as such — that attribution remains qualified. What the three rows *do* show unambiguously is that $B-C \gg A-B$: whatever the split between checkpoint and tiling, the pair of them dominates the port exchange by roughly $4\times$.

---

## 4. W11 — closed as not measurable

Gate W11 asks for a power residual under 1% with a per-port breakdown. It is **not measurable with this checkpoint**, and the reason is structural rather than a shortfall of effort: the residual's dissipation term needs a single $\nu$, and W0 established that **81% of the field's dissipation lives below $0.125\,D$** — exactly the scale where no single $\nu$ reproduces the checkpoint's behaviour. Recorded in full as OP-4.

Per [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule, the honest disposition is to close the gate as unmeasurable and say so, rather than report a number produced by choosing a $\nu$. **No power-residual figure is quoted anywhere in this page.**

---

## 5. Compute

The whole set above is about five minutes of GPU. It was six hours of CPU this morning.

| run | wall |
|---|---|
| frozen $t=20$ | $191$ s |
| solver $t=20$ (torch/CUDA backend) | $111$ s |
| W9 (three runs) | $\approx 6$ min |

The window solve was ported to a torch backend after profiling showed **95.2%** of a macro-step in batched finite-difference stencils and a CPU thread pool saturating at $2.81\times$ — a memory-bandwidth wall, not a core shortage. On the 5090 the same `step_batch` runs **$33.8\times$** faster than numpy and agrees with it to $4\times10^{-13}$ on a field of scale $7.4$; at small sizes the agreement is $10^{-15}$ across all three transmission conditions. The full coupled system reproduced the CPU result exactly. See [[vast-ai-windfarm-runbook]].

---

## 6. What is not done

- ~~**The $N = 2,3,5,8$ scaling curve does not exist.**~~ **Closed 2026-08-25 — see [[results-n-sweep-wind-farm]].** `geometry.py` was parametrised (the graph is now generated: $3N+2$ agents, $8N-1$ interfaces) and the curve measured. The headline: **composition error is bounded in graph size, not accumulating in it** — $P_2/P_1$ moves under $1\%$ between $N=2$ and $N=8$ while the graph triples. Farm-level array efficiency still degrades ($1.258\times \to 1.793\times$ overprediction) because the plateau where the expert is wrong covers more of the array, which is a different statement and the page keeps them apart. The $N=2$ row of that sweep reproduces the $\mathbf{0.7663}$ on this page to four decimals, so nothing here is superseded.
- R1-vs-R2 write-up, and scaffold action item 4 (expert/MP interleaving), still open from the definition of done.
- Integration hours are logged per session in [[wind-farm-implementation-log]] rather than aggregated here.

---

## 7. The answer, stated plainly

Composing frozen pretrained experts through declared ports **did not preserve physical validity** on this problem. Array efficiency inverts, upstream induction is $5.6\times$ the reference, and three quarters of the wake is gone by the second turbine.

The failure is **attributable rather than diffuse**, which is the part worth keeping. With the expert swapped for a solver and the boundary treatment corrected, the same graph, the same tiles, the same ports and the same assembly produce $P_2/P_1 = 0.7663$ and a correctly-signed induction within 18% of the reference. The composition machinery is sound enough to pass W10; the frozen one-shot operator inside it is not, and the specific repair that fixes the architecture is one a frozen expert cannot accept.

**[AI Inference]:** the sharpest reading is that the interface is the constraint, not the weights. What a frozen checkpoint lacks here is not accuracy but a *channel* — somewhere to put a boundary condition. An expert exposing even a one-cell ring as an input would be eligible for the fix that moved $P_2/P_1$ from $1.02$ to $0.77$. That is a claim about what to require of a pretrained operator before composing it, and it is untested — no such expert was run.

---

## See Also
- [[impl-wind-farm-guide]] — the gates and §9's reporting rules
- [[wind-farm-implementation-log]] — the measurement history behind each number
- [[open-problems-atlas-0.1]] — OP-2 (induction), OP-3 (dissipation), OP-4 (W11), OP-5 (windowing)
- [[results-w0-w3-wind-farm]] — the earlier half
- [[vast-ai-windfarm-runbook]] — how the GPU runs were produced
- [[case-study-wind-farm-wake-2d-atlas-0.1]] — the case study
