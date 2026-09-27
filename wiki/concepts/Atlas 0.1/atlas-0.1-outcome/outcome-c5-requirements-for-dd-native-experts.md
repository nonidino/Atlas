# Outcome C5 — what a learned expert native to domain decomposition must satisfy, as measured

**Type:** Outcome page — **claim audit and measured specification** (folder: `Atlas 0.1/atlas-0.1-outcome/`)
**Status:** compiled 2026-09-26 from existing records; no new measurement.
**Verdict:** **Not done.** No architecture has been designed, and no learned expert has been created, trained or fine-tuned for non-rectangular geometry or for fast convergence under domain decomposition. **What exists** is the proposal's strongest asset: 88 tiers of measurement that say, as numbers, **why the available frozen expert fails in each slot and what an expert would need in order not to.** This page collects that specification. It proposes no design.
**Hub:** [[00-atlas-0.1-outcome]]
**Sources:** [[poc3-racelab-car-windows]] · [[poc3-racelab-certified-step]] · [[case-study-racelab-switch-atlas-0.1]] · [[defect-correction-learned-operator]] · [[corrupted-checkpoint-and-jacobian-fidelity]] · [[learned-contribution-kill-tests]] · [[epsilon-halo-measurement]] · [[substitution-campaign-checkpoint]] · [[expert-donor-survey]] · [[poc1a-frozen-expert-results]] · [[results-w0-w3-wind-farm]] · [[case-study-neural-interface-atlas-0.1]]

---

## 1. What "native to domain decomposition" means here

**Intuition.** A learned expert inside a decomposed model does not solve a whole problem. It advances **its own piece**, given what its neighbours tell it across the seams, and it is called again and again as the pieces iterate toward agreement. Being *accurate on its own* is not enough. It must accept the piece's real shape, run at the step the coupling runs at, take its neighbours' data as input, and respond to a change in that data the way the true physics would, because that response sets how fast the iteration converges.

**Formally.** Write the expert as a map

$$E:\ (u_\Omega,\ g_{\partial\Omega},\ \mathcal G_\Omega,\ \Delta t)\ \longmapsto\ u_\Omega(t+\Delta t),$$

with $u_\Omega$ the state on its subdomain $\Omega$, $g_{\partial\Omega}$ the interface data from its neighbours, $\mathcal G_\Omega$ the subdomain's geometry and $\Delta t$ the step. In the one slot where a learned map provably cannot corrupt the answer, defect correction ([[outcome-c2-learned-experts-in-the-loop]] §3), the outer iteration's per-band contraction is

$$\lambda \;=\; \frac{\lvert\phi-\psi\rvert}{\lvert 1-\psi\rvert},$$

with $\phi$ and $\psi$ the classical and learned maps' responses on that band. The rate depends on the learned map's **derivative** along perturbations of its input, not on its value. That single fact is why the specification below is about responses, inputs and cadence, and only secondarily about accuracy.

---

## 2. The specification, measured on the frozen checkpoint

Each row is a property the available expert (Poseidon-T) was measured to lack, and the consequence measured in a named slot.

| # | property | measured on the frozen checkpoint | consequence, measured | source |
|---|---|---|---|---|
| **S1** | **accepts the subdomain's geometry** | accepts only a uniform $128\times128$ grid | on the body-fitted car, the curved grids hold $61.0\%$ of the $377{,}267$ unknowns and cannot be accepted at all. The ceiling is $33.4\%$ of the composite; a real greedy tiling covers $17.4\%$. **None of the reachable third is near the car**: the boundary layers, wheel clearances and cooling duct are all in the unreachable $61\%$ | [[poc3-racelab-car-windows]] |
| **S2** | **runs at the coupling's step** | one native lead, $0.1$; error is minimal there and grows $4.3\times$ as the step **shrinks** ($0.1708$ at native, $0.7301$ at $\tfrac12$) | the porous car exchanges at $\tfrac{1}{32}$ of native: all-learned runs at $0.067\times$ classical speed and leaves its envelope for 35 of 40 steps. The body-fitted car steps at $0.025$, a quarter of native, so the wrapper **refuses the call** | [[case-study-racelab-switch-atlas-0.1]] · [[poc3-racelab-certified-step]] §6 |
| **S3** | **takes interface data as an input** | no boundary-data channel: pretrained on periodic boxes, driven only by overwriting a halo. Of twenty-odd surveyed donors, only PDEformer-2 and NeuberNet take boundary data as a declared argument. Fed a uniform flow, Poseidon-T returns a mean of $0.969$ after one call and $0.281$ after forty unless the composition layer restores it | the composition layer must supply the mean, the transport and the projection itself | [[expert-donor-survey]] finding 2 · [[poc1a-frozen-expert-results]] §1 |
| **S4** | **a bounded, measurable domain of dependence** | declares a 2-cell halo; a one-cell poke moves all 128 seam cells. Band-truncating the seam operator leaves $27$–$75\%$ of its norm behind, **flat** from radius $16$ to $64$, on two scOT checkpoints | `L2/R10` refuses to certify the decomposition; no $\varepsilon$-halo exists to carry the tail as an error term | [[epsilon-halo-measurement]] |
| **S5** | **the right slow-mode Jacobian** | slow-band response $\psi_{\text{slow}} = 0.5578$ against the classical $\phi_{\text{slow}} = 0.4780$: it **preserves what the classical map damps** | in defect correction, per-band rate orders 20 cheap maps' classical-call counts at rank correlation $+0.83$, accuracy only $+0.69$. The learned content is worth $13$ calls against a $69$-call noise floor | [[corrupted-checkpoint-and-jacobian-fidelity]] |
| **S6** | **not closer to singular than the classical map on any mode** | the entry condition defect correction needs (W204) | stated as a gate, failed out of sample at 6 and 12 windows | [[defect-correction-learned-operator]] §4, §8 |
| **S7** | **cheaper than the cheapest classical alternative in the same slot** | $0.296$ of a monolith step at 12 windows | a $2\times$-coarse classical solver costs $0.0320$: $9.2\times$ cheaper, and $17.4\times$ at 2 windows | [[learned-contribution-kill-tests]] §2.3 |
| **S8** | **a reference of its own class** | fixed at one resolution, so no same-class monolith exists at any resolution (W95) | no $\tau$, $\sigma$, $\varepsilon_{\text{tol}}$ or $\beta_{\min}$, so no substitution certificate can be issued; forty-seven tiers traced to this chain | [[substitution-campaign-checkpoint]] |
| **S9** | **accurate enough to be worth the slot** | over-dissipates wakes (OP-3): keeps about a quarter of the centreline deficit | inverts the wind farm's array efficiency ($P_2/P_1 = 1.0162$ against $0.404$). Captures $71\%$ of the verified design gain | [[results-w6-w11-wind-farm]] · [[poc1a-frozen-expert-results]] §7 |
| **S10** | **precision the probes can use** | float32: finite-difference floor $1.4\times10^{-3}$ against the classical column's $5.9\times10^{-9}$; TF32 silently on by default on Ampere | probe steps and certificates built on the classical floor do not transfer | [[poc1a-frozen-expert-results]] §2.1, §5 |
| **S11** | **a stable free rollout** | unforced fluctuation energy passes its initial value at $t=42.5$ and reaches $2.47\times$ by $t=60$ | the wind farm's spec rollout cannot be survived without the composition layer's projection | [[results-w0-w3-wind-farm]] |
| **S5a** | **a stable approach, not only a good settled-state Jacobian** (Tier 89) | the linear rate at $w^\star$ predicts the weakly shrunk column converging $3\times$ faster than the $\alpha = 0.5$ one | every weakly shrunk arm **diverges within one or two outer iterations** from the freestream start and falls back to the classical march; only $\alpha = 0.5$ decreases monotonically. A probe at the settled state ordered every pair backwards | [[matched-shrink-and-coarse-competitor]] §1.3 |
| **S7a** | **the bar at a coupled seam** (Tier 89) | — | the coarse classical competitor is correct but loses to the cold march on CS-13 and CS-12, so the bar a learned expert must clear there is the cold march ($7411$ and $901$ classical calls), not a coarse solver | [[matched-shrink-and-coarse-competitor]] §4 |
| **S12** | **a licence that allows use** | Poseidon's weights are CC-BY-NC-4.0, and the term propagates to anything fine-tuned from them | every learned result in the vault is research use only | [[expert-donor-survey]] finding 1 |

**One more, for implicit steps.** On the body-fitted car the certified fixed point is a linear-solve sweep with Jacobian $I - PM$, not a time advance. The certified-step page's own inference: *"a learned expert that helps this fixed point would be a learned preconditioner, which is a different artefact from any checkpoint this project holds"* ([[poc3-racelab-certified-step]] §6, **[AI Inference]** there). The frozen checkpoint as a preconditioner was already killed: its block is $0.8$–$9\%$ of the classical one, so it acts as the zero operator (F8, [[learned-contribution-kill-tests]]).

---

## 3. What the record already provides toward a method

None of this is an architecture. It is what a funded project would start from instead of from zero.

| asset | what it is | where |
|---|---|---|
| **the slot, with theorems** | defect correction with the learned map as the approximate operator: consistency whatever the map is, the null element bitwise, the rate through the Jacobian, and a residual certificate | [[defect-correction-learned-operator]] |
| **a pre-registered gate** | classical calls against the cold march, cost against the coarse competitor, the entry condition. Written before out-of-sample runs, and failed honestly by the frozen checkpoint | same, §7 |
| **the instruments** | per-band Jacobian probes at the settled state (`scripts/w205_corruption_sweep.py --stages probe`); support-reach and band-truncation probes for the domain of dependence; the seam ledger that reads a solver's own face fluxes | [[corrupted-checkpoint-and-jacobian-fidelity]] · [[epsilon-halo-measurement]] · [[case-study-rocket-bc-seam-atlas-0.1]] §19.2 |
| **classical teachers on non-rectangular geometry** | body-fitted overset Navier–Stokes, verified on a cylinder at $\mathrm{Re}=100$ (St $0.169$) and marched on the car; a curvilinear compressible nozzle validated to $0.992$ of ideal thrust | [[poc3-racelab-overset-flow]] · [[outcome-c1-decomposition-vs-monolith]] §2 |
| **the price of adaptation** | Poseidon-T forward on two windows $0.173$ s; forward and backward $0.630$ s; $2000$ iterations $\approx 0.35$ h before data and optimiser. **Priced, never run** (F9) | [[learned-contribution-kill-tests]] §4.7 |
| **donor candidates** | permissively licensed: Walrus (MIT), DPOT (Apache-2.0), GPhyT (MIT). For irregular meshes: Transolver (MIT; only aero checkpoints confirmed). For boundary data as input: PDEformer-2 (MindSpore only) | [[expert-donor-survey]] |
| **earlier training plans** | the rocket's Phase 2 spec (adapter → LoRA bootstrap from Poseidon, stage-A all-to-all operator pretraining) and its compute budget. Written 2026-08-09, before any of §2 was measured, and not executed | [[impl-atlas-0.1-phase2-experts]] · [[impl-atlas-0.1-compute-and-training-budget]] |
| **rented-compute workflow** | vast.ai CPU runs costing \$0.17–\$2.34 per rocket tier (W334, Tier 88, W343), and a whole frozen-expert design study for about \$1.15 on one A100, hash-checked and destroyed after use | [[vast-ai-windfarm-runbook]] |

---

## 4. The requirements this implies

**[AI Inference]:** read directly off §2. **This is not a design and has not been checked against any build.**

1. **Geometry as input** (S1, S4): an operator defined on the subdomain's own grid or point set, curvilinear and overset included, with a locality structure the compiler can probe.
2. **Step as input, valid at the coupling cadence** (S2): trained across the range of leads the composition layer will ask for, down to a fraction of any native step, and checked for step-doubling self-consistency (F6).
3. **Interface data as a declared input** (S3): the neighbour's traces enter as an argument, not as an overwritten halo.
4. **Trained on responses, not only states** (S5, S6): the loss must see the slow-mode response to interface perturbations, because that, measured at $\rho=+0.83$ against accuracy's $+0.69$, is what decides convergence. **And along the whole approach** (S5a): the response must be checked away from the settled state too, because Tier 89's weakly shrunk iterations failed there, not near $w^\star$.
5. **Priced against the cheapest classical alternative in the same slot** (S7, S7a). That is a coarse solver on single-family flow, and the cold classical march at the coupled seams measured so far. The expert must also be resolution-flexible enough that a same-class reference exists (S8).
6. **A licence that permits the proposal's intended use** (S12).

---

## 5. What was NOT done, named

Everything the claim asks for:

1. **No architecture** has been chosen or specified for a DD-native expert.
2. **No training data** has been generated. The corpus plan written for the rocket priced its original corpus at 30.6 core-years ([[impl-atlas-0.1-phase0-scope-and-data]]); nothing was generated for learned experts on non-rectangular subdomains.
3. **No training or fine-tuning run** of any size. F9 was priced and deliberately not run, because the theory says what a Jacobian must do and nothing indicated fine-tuning on one-step errors would teach it that.
4. **No evaluation protocol** beyond the defect-correction gate, which is 2-D, single-family and uniform-grid.
5. **W214 was run at Tier 89** and answered a different question from the one it asked (S5a): the composition layer's shrink is limited by stability along the approach, not by fidelity at the settled state. A shrink scheduled on the residual (W347) is untested.
6. **No second donor was ever loaded** (W120), so every row of §2 is one checkpoint.
7. **No 3-D expert** and no expert for any family but 2-D incompressible flow.

---

## See Also

- [[00-atlas-0.1-outcome]] · [[outcome-c2-learned-experts-in-the-loop]] · [[outcome-c3-learned-speed-at-scale]] · [[outcome-evidence-ledger]]
- [[prior-art-and-novelty-atlas-0.1]] — §4 names the assumption S2 falsifies: that error shrinks with the macro step
- [[probed-dtn-coupling]] — why the interface operator should approximate Steklov–Poincaré, and what probing costs
- [[generalization-requirements]] — the earlier cut of gaps between accurate and generic composition
- [[case-study-neural-interface-atlas-0.1]] — why a learned *first guess* cannot beat one sweep, which is why S5 is about the operator
