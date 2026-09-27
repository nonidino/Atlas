# Outcome C2 — learned experts coupled to classical ones, and converging to the classical answer

**Type:** Outcome page — **claim audit** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** compiled 2026-09-26 from existing records; no new measurement.
**Verdict:** **Partly demonstrated.** Coupling is stable, and convergence to the classical answer is *certified* inside defect correction. The frozen expert's own answer is biased, and its contribution to the rate is small.
**Hub:** [[00-atlas-0.1-outcome]]
**Sources:** [[results-w0-w3-wind-farm]] · [[results-w6-w11-wind-farm]] · [[poc1a-frozen-expert-results]] · [[case-study-learned-pair-atlas-0.1]] · [[case-study-neural-interface-atlas-0.1]] · [[defect-correction-learned-operator]] · [[corrupted-checkpoint-and-jacobian-fidelity]] · [[case-study-racelab-switch-atlas-0.1]] · [[poc3-racelab-certified-step]] · [[poc3-racelab-certified-screen]]

---

## 1. The claim, split into the three things it could mean

*"Learned experts speak with, and converge with, classical simulations in domain decomposition"* can mean three different things. They have different answers.

| reading | question | answer on the record |
|---|---|---|
| **(a) they couple** | Does a learned expert exchange data across a seam with a classical one without breaking conservation or blowing up? | **Yes.** §2 |
| **(b) the iteration converges to the classical answer** | Can a learned expert sit inside a coupling iteration whose limit is the classical solution? | **Yes, by theorem and by measurement**, in defect correction. §3 |
| **(c) the learned expert's own answer agrees with the classical one** | Is the learned column accurate? | **No.** It is biased, by a known mechanism. §4 |

**The one expert behind almost every row** is **Poseidon-T**: a frozen $20.8$M-parameter scOT neural operator for 2-D incompressible flow, pretrained by others on periodic $128\times128$ boxes, licensed CC-BY-NC-4.0. This project did not train it and never fine-tuned it. Rung 3 adds **NeuberNet** (a DeepONet for elastic-plastic notch stress, unlicensed, local-only).

---

## 2. (a) They couple

**A frozen checkpoint closes a momentum budget it was never trained on.** The wind farm's go/no-go gate W3 hands Poseidon-T a rotor's body force from an exact actuator-disk expert. Teacher-forced and before any projection, the streamwise momentum budget across the rotor closes to

$$r_T = \frac{\lvert \text{momentum deficit} - T \rvert}{T} = 2.6\times10^{-3}\ \text{(flux-BC exchange)},\qquad 6.2\times10^{-4}\ \text{(halo exchange)},$$

against a $5\%$ threshold. That is the level a purpose-built spectral solver reaches on the same problem ([[results-w0-w3-wind-farm]]). Its own page calls the gate weak and names the rollout it cannot survive.

**A composed column of frozen experts marches stably.** With every window Poseidon-T and one zero-parameter actuator disk per rotor, the composed column stays finite and inside its velocity band for $70$ of $70$ macro-steps at 12 and at 24 windows. Its $u_{\max}$ is $1.45$ and $1.44$ against a band of $3.0$. The classical column at the same rung reads $1.52$ ([[poc1a-frozen-expert-results]] §3). The page's own inference is why: the global projection after the blend makes the *assembled* field valid, so it never needs each window's field to be valid outside its support.

**The adjoint goes through it.** Reverse-mode gradients of farm power pass through all $20.8$M parameters' activations and every seam. They agree with central differences to a median relative error of $1.4\times10^{-3}$, which is the float32 floor ([[poc1a-frozen-expert-results]] §5).

**Two learned experts of different families compile together.** Poseidon-T meets NeuberNet across a `MECH` seam and reaches `admit-uncertified` with zero refusals, as does every cell of a $2\times2$ against the classical incumbents. The cost of the second learned expert is exactly additive: decertifications run $10\to11\to12\to13$ with no interaction term ([[case-study-learned-pair-atlas-0.1]]). **This is a compile, not a march.**

---

## 3. (b) The iteration converges to the classical answer: defect correction

This is the programme's cleanest positive result about learned experts, and it is a theorem before it is a measurement ([[defect-correction-learned-operator]]).

### 3.1 The slot

Let $\Phi$ be the classical composed macro-step, $w^\star$ its settled state ($\Phi(w^\star)=w^\star$), and $\Psi$ any cheap map on the same state space, such as the learned composed column. Stetter's defect correction iterates

$$G_\Psi(w_{k+1}) \;=\; G_\Psi(w_k) - G_\Phi(w_k),\qquad G_\Phi = I-\Phi,\quad G_\Psi = I-\Psi,$$

solved for $w_{k+1}$ by the inner fixed-point march

$$x^{(0)}=\Phi(w_k),\qquad x^{(m+1)} = \Phi(w_k) + \bigl[\Psi(x^{(m)})-\Psi(w_k)\bigr].$$

**Intuition.** The classical solver is called once per outer iteration and always has the last word. The learned map proposes corrections, and it enters **only through a difference**, so a constant bias in it cancels before it is added.

### 3.2 What is guaranteed

- **Theorem 1 (consistency).** Any limit of the iteration is the classical settled state $w^\star$, **whatever $\Psi$ is.** The learned map's errors cost classical calls, never correctness.
- **The null element.** Replacing $\Psi$ by a constant makes the iteration the classical march, bit for bit.
- **The rate** is set by $\Psi$'s Jacobian against $\Phi$'s, not by $\Psi$'s accuracy. Per band, Theorem 2's contraction is $\lambda = \lvert\phi-\psi\rvert/\lvert 1-\psi\rvert$.
- **A certificate.** The returned state carries its own classical residual times a constant $\Theta$ of the classical map.

### 3.3 Measured

| where | result | source |
|---|---|---|
| wind array, 2 windows, in sample | Poseidon-T column, shrunk halfway to the null element, reaches the certified state in **$25$ classical calls against the cold march's $47$**. The detuned column needs $32$. The composition layer with the checkpoint replaced by the identity falls back after $14$ outer iterations, $35$ classical calls in all. | [[defect-correction-learned-operator]] §8.1 |
| the car's body-fitted column, implicit step | **every cheap map lands on the classical answer**, including a deliberately wrong one (a random rescaling), which lands $6.15\times10^{-9}$ from it against a state of order one | [[poc3-racelab-certified-step]] |
| the car, on screen | a three-way switch (classical, certified, learned-refused) on the live dashboard. The certified step costs $1.21\times$ the classical step, $1.83\times$ with verification on | [[poc3-racelab-certified-screen]] |
| RaceLab's porous column, per window, as a steady-state solve | both windows reach the classical fixed point, as `fallback_converged`: the outer iteration stalled and the classical march finished the job, which is the mechanism's declared failure path | [[case-study-racelab-switch-atlas-0.1]] |

**So reading (b) holds, in the strongest form available: the answer is classical by construction, and an adversarial learned map could not move it.**

### 3.4 But the learned map barely speeds the iteration up

| where | what the learned content bought | source |
|---|---|---|
| wind array, 6 windows, out of sample | $99$ classical calls where the pre-registered gate asked for $\le 90$. $314$ classical-equivalents against a $2\times$-coarse classical solver's $52$ | [[defect-correction-learned-operator]] §8.2 |
| wind array, 12 windows | fifth of seven arms by cost. The certificate failed for 2 of 7 arms, because $\Theta$ was read off one approach to the settled state (W208) | same |
| 12 corrupted copies of the checkpoint | the learned content is worth **$13$ classical calls** over the identity, against a **$69$-call** spread from $3\%$ weight noise | [[corrupted-checkpoint-and-jacobian-fidelity]] |
| RaceLab, per window, steady-state solve | against the identity, the checkpoint bought $5$ and $7$ classical calls out of $270$ and $165$ (about $2\%$) at $3\times$ the wall time | [[case-study-racelab-switch-atlas-0.1]] |

**Why, measured.** On the slow band, the classical map's response is $\phi_{\text{slow}} = 0.4780$ and the clean learned column's is $\psi_{\text{slow}} = 0.5578$: it **preserves what the classical map damps**. The composition layer's shrink $\alpha^\star=0.5$ then halves it to $0.2789$, overshooting below the classical map by $2.5\times$. Over twenty cheap maps, Theorem 2's per-band rate orders the classical-call counts at a rank correlation of $\mathbf{+0.83}$, against $+0.69$ for accuracy. **Jacobian fidelity, not accuracy, decides the rate.** This is the single most important input to [[outcome-c5-requirements-for-dd-native-experts]].

---

## 4. (c) The learned expert's own answer is biased

| measurement | learned column | reference | source |
|---|---|---|---|
| wind farm, turbine 2 over turbine 1 power, $P_2/P_1$ | $\mathbf{1.0162}$ (turbine 2 out-produces turbine 1) | classical 2-D reference $0.404$; a classical solver in the **same** graph $0.7663$ | [[results-w6-w11-wind-farm]] |
| wake retained by turbine 2 | $0.26\times$ the reference | the classical expert in the same graph: $0.62\times$ | same |
| farm power at the unoptimised layout, vs the undivided classical solver | $-3.0\%$ (K12), $-9.6\%$ (K25) | the classical composed column: $-9.4\%$, $-18.0\%$ | [[poc1a-frozen-expert-results]] §7.1 |
| verified design gain, layout found in the learned column | $+266\%$ (K12), $+170\%$ (K25) | the classical column's layout: $+374\%$, $+240\%$. The learned column captures **$71\%$** of it at both sizes | same, §7.2 |
| rms velocity difference from the monolith after 120 steps, 2 / 6 / 12 windows | $0.0822$ / $0.0959$ / $0.121$ | classical composed: $0.0891$ / $0.0937$ / $0.0830$ | [[learned-contribution-kill-tests]] §2.2 |
| one-step error per window at the composition layer's lead ($\tfrac18$ native) | median $0.3607$ | at the native lead: $0.1708$ | [[case-study-racelab-switch-atlas-0.1]] |

**The mechanism is named (OP-3).** The checkpoint over-dissipates wakes: it keeps about a quarter of the classical centreline deficit, and $0.12\times$ of it by sixteen diameters. The results page's own conclusion is the one to quote: **the architecture is not what breaks it.** Swap the expert for a solver in the identical graph and $P_2/P_1$ lands at $0.7663$, inside the gate's band.

The learned error also **saturates**: $0.074$, $0.093$, $0.097$, $0.096$, $0.096$, $0.096$ at steps 20 to 120. It is a bias, not a drift.

---

## 5. What C2 can honestly claim

> *A frozen, independently pretrained neural operator couples stably with classical and closed-form experts through declared ports, and can be differentiated through. Placed inside defect correction, the coupled iteration provably converges to the classical solution whatever the learned operator does. This holds under measurement, including an adversarial map. The frozen operator available today is too biased to replace the classical answer, and too poorly matched on its slow-mode Jacobian to accelerate the iteration by more than a few percent.*

**[AI Inference]:** the second half of that sentence describes the frozen checkpoint, not learned experts in general. The defect-correction theory says precisely which property an expert would need to change it (§3.4). That makes C2's negative half the motivation for C5, not a refutation of the approach.

---

## 6. What was NOT done, named

1. **W214, run at Tier 89** ([[matched-shrink-and-coarse-competitor]] §1). Matching the shrink to the slow-mode response gives $\alpha = 0.143$ at six windows and $-0.020$ at twelve, where the clean column already under-responds. It improves six windows from $99$ to $82$ calls, inside a $38$-call swing across neighbouring $\alpha$, and worsens twelve windows to $270$. **The finding is the mechanism.** Weakly shrunk iterations diverge within one or two outer iterations from the freestream start, so stability along the approach, not the rate at the settled state, decides the calls. The linear probe at $w^\star$ cannot see it. G5 did not move. **W347** (a shrink scheduled on the residual) is the next test and is cheap.
2. **W215:** a randomised arm is still scored as if it were deterministic.
3. **W208:** $\Theta$ from a power iteration on $I - D\Phi$ at $w^\star$, rather than from one march. The certificate failed for 2 of 7 arms without it.
4. **No second donor.** W120 (re-run the verified pre-screen with Walrus, DPOT or GPhyT, all permissively licensed) was opened and not run. Every learned result here is one checkpoint.
5. **No learned expert has run in any family except 2-D incompressible flow.** NeuberNet compiled but never marched in a coupled rollout.
6. **No learned expert has run on the body-fitted car**, the rocket, or any non-rectangular geometry. See [[outcome-c5-requirements-for-dd-native-experts]] for why.

---

## See Also

- [[00-atlas-0.1-outcome]] · [[outcome-c3-learned-speed-at-scale]] · [[outcome-c5-requirements-for-dd-native-experts]] · [[outcome-evidence-ledger]]
- [[learned-contribution-kill-tests]] — the fourteen formulations, and why defect correction was the one kept
- [[substitution-campaign-checkpoint]] — why forty-seven tiers of substitution certificates could not admit a learned expert (W95)
- [[open-problems-atlas-0.1]] — OP-3, the wake over-dissipation
- [[expert-donor-survey]] — the checkpoints that exist, and their licences
