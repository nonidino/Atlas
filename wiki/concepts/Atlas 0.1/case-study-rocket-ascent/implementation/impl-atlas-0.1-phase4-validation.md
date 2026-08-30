# Phase 4 — End-to-End Validation

**Type:** Implementation spec — agent task (folder: Atlas 0.1 / Atlas 0.1 implementation)
**Phase of:** [[00-atlas-0.1-implementation-plan]]. **Depends on:** [[impl-atlas-0.1-phase3-integration]] (M5 passed). **Unblocks:** [[impl-atlas-0.1-phase5-expansion]].
**Design pages:** [[case-study-rocket-ascent-2d-atlas-0.1]], [[00-atlas-0.1-overview]], [[training-and-bootstrap-atlas-0.1]].
**Milestones:** M6 (stable full-ascent rollout), M7 (validation report).

---

# 1. Intuition

**This phase is the experiment. Everything before it was setup.**

Atlas 0.1 exists to test one proposition, stated in [[00-atlas-0.1-overview]]: *does composing independently-trained, regime-specialized experts through declared typed interfaces produce a stable, physically plausible coupled rollout?* Phases 0–3 built the apparatus. Phase 4 runs it and reports honestly what happened.

Three things make this phase easy to do badly:

**The temptation to measure only accuracy.** A low relative $L^2$ against the classical solver is necessary but shallow. The interesting failure mode of a composed architecture is not "each expert is a bit off" — it is "the experts disagree at the seams and the disagreement compounds." That shows up in *conservation residuals* and *long-horizon drift*, not in one-step error. Design the metrics around the failure mode you actually expect.

**The temptation to skip ablations.** If Atlas 0.1 works, the immediate follow-up question is *which part made it work?* Without the ablations in §3.4, a success is uninterpretable — it could be that the declared graph structure is doing nothing and a single monolithic network on the same data would have done as well. That would be a genuinely important negative result about the architecture, and it is cheap to discover here and expensive to discover later.

**The temptation to over-claim.** Per [[transfer-learning-fine-tuning]]'s diversity principle, narrow pretraining reliably produces models that look good in-distribution and generalize badly. Success on one scenario is evidence that **the architecture is soundly wired**. It is *not* evidence that **the experts generalize**. These are different claims and the report must keep them separate. The second claim cannot be tested until Phase 5 introduces a second scenario.

---

# 2. Theory

## 2.1 What would falsify the proposition

Stating this before running is what makes the result meaningful. The proposition fails if any of the following hold on the held-out corner configurations:

| Failure | Signature | What it would mean |
|---|---|---|
| **Seam divergence** | conservation residual grows monotonically with rollout step | Independently-trained experts cannot be composed through declared interfaces without joint training that defeats the point |
| **Rollout blow-up** | NaN, or rel-$L^2$ exceeding 1.0 before burnout | Multi-rate coupling is unstable at the chosen cadence |
| **Structural failure** | choked throat lost, plume shock cells absent, or nozzle separation appearing at the wrong altitude | Experts interpolated the corpus without learning regime structure |
| **No speedup** | Atlas macro-step wall-clock ≥ classical coupled solver | The amortization premise fails at this scale; a surrogate that is not faster has no reason to exist |
| **Architecture does nothing** | monolithic-expert ablation matches or beats full Atlas | The agent/edge/expert decomposition is not carrying its weight |

The last row is the one most likely to be quietly ignored and is the most valuable outcome if it occurs.

## 2.2 Metrics

**Horizon-resolved relative error**, per agent, per field:

$$\varepsilon_a^{(k)}=\frac{\left\|\hat x_a(t_0+k\Delta t_{\text{macro}})-x_a(t_0+k\Delta t_{\text{macro}})\right\|_2}{\left\|x_a(t_0+k\Delta t_{\text{macro}})\right\|_2}$$

Reported as a curve over $k$, never as a single number — the *shape* of the curve distinguishes bounded error from compounding error, which is the whole question.

**Conservation residuals**, per conservation edge, per step:

$$R_{\dot m}^{(k)}=\frac{\left|\dot m_\Gamma^{\text{src}}-\dot m_\Gamma^{\text{dst}}\right|}{\left|\dot m_\Gamma^{\text{src}}\right|},\qquad R_E^{(k)}\ \text{analogously}$$

$R_{\dot m}$ should sit at machine-epsilon (it is hard-projected); $R_E$ is the informative one, since energy is only softly constrained.

**Global budgets** over the full ascent:

$$\Delta_m=\frac{\left|m(T)-m(0)+\int_0^T\dot m_{\text{exit}}\,dt\right|}{m(0)},\qquad \Delta_E=\frac{\left|E(T)-E(0)-\int_0^T\left(\dot W_{\text{thrust}}-\dot Q_{\text{loss}}\right)dt\right|}{E(0)}$$

**Trajectory error:** altitude, velocity, and pitch angle against the classical coupled run.

**Speedup:** wall-clock seconds per simulated second, Atlas vs. the Phase 0 coupled solver, on the same hardware, both warm.

## 2.3 Physical-milestone checks

These test whether the model reproduces *phenomena*, not just numbers, and they are the strongest available evidence that the experts learned regime structure rather than the corpus mean. Each is a qualitative behaviour with a known direction:

1. **Thrust rises with altitude.** For fixed chamber conditions, $F=\dot m u_e+(p_e-p_\infty)A_e$ increases as $p_\infty$ falls. Atlas's thrust curve must show this monotone rise, with the sea-level-to-vacuum ratio within 10% of the solver's.
2. **Max-Q exists and is correctly timed.** $q_\infty=\tfrac12\rho_\infty v^2$ rises as velocity grows, then falls as density drops — a genuine interior maximum. Atlas's max-Q *time* must land within 5% of the solver's, and its magnitude within 10%. This is a demanding test because it depends on the whole coupled loop (thrust → acceleration → altitude → density → drag) being right, not on any single agent.
3. **Nozzle flow separation at low altitude.** At sea level with the design expansion ratio $\varepsilon=3$, the nozzle is over-expanded and the solver should show separation; at high altitude it should not. Atlas must reproduce the presence/absence of separation at both ends, and the transition altitude within 20%.
4. **Plume shock-cell structure.** An under-expanded plume at altitude produces a quasi-periodic shock-cell pattern. Cell spacing within 10% of the solver's.

Checks 2 and 3 are the ones that cannot be passed by a model that merely regressed the training distribution.

---

# 3. Implementation

## 3.1 Rollout protocol (`eval/rollout.py`)

```python
def run_ascent(model, cfg, episode_ic, horizon_s: float = 60.0) -> RolloutRecord:
    """Free-running: no teacher forcing, no ground-truth injection at any point."""
```

Binding rules:
- **Initial condition only.** The model receives $t{=}0$ state and the atmosphere model. Nothing else is supplied during the rollout. Any ground-truth injection invalidates the result.
- **Held-out corner configs only** for the headline numbers (from [[impl-atlas-0.1-phase0-scope-and-data]] §3.5: highest $p_c$/highest altitude, lowest $p_c$/sea level).
- **Horizon:** 60 s of simulated ascent = 1,200 macro-steps = 60,000 combustion substeps. Report the full curve; do not truncate at the point where error becomes unflattering.
- **Matched comparison:** the classical coupled solver runs the identical configuration with the identical macro-step and coupling scheme, so the comparison isolates the surrogate, not the coupling scheme.
- **Five seeds** per config; report median and inter-quartile range. Single-seed results from a stochastic training pipeline are not a measurement.

## 3.2 Baselines

| Baseline | Purpose |
|---|---|
| Classical coupled solver (Phase 0) | ground truth and speed reference |
| Persistence ($\hat x_{t+\Delta t}=x_t$) | the floor — any model must beat this, and it is surprisingly strong at short horizons |
| Monolithic expert (ablation A1) | tests whether decomposition helps |
| Per-agent experts, no edges (ablation A3) | tests whether coupling helps |

Persistence is included because it catches a specific embarrassing failure: an increment-predicting model whose increments have collapsed toward zero will look stable and score reasonably at short horizons while having learned nothing.

## 3.3 Report contents (`eval/report.py`, → M7)

1. Horizon-resolved $\varepsilon_a^{(k)}$ curves, all 7 agents, log-$y$.
2. Conservation residual traces for $e\!-\!f$ and $d\!-\!g$.
3. Global mass and energy budget closure.
4. Trajectory overlay (altitude, velocity, pitch) vs. classical.
5. All four physical-milestone checks, pass/fail with measured values.
6. Speedup table.
7. Ablation table (§3.4).
8. **A "what failed" section.** Mandatory, and it must be written even if everything passed — in which case it records what was *not tested* (staging, composability, abstention, 3D, turbulence closure), so the result is not later read as a broader claim than it is.

## 3.4 Ablations (the interpretive core of the phase)

| ID | Ablation | Question it answers |
|---|---|---|
| **A1** | One monolithic expert for all 7 agents, same total parameter count | Does regime specialization help, or is it decoration? |
| **A2** | Conservation constraint disabled | Does imposed flux matching actually prevent seam drift? |
| **A3** | Declared edges removed (agents evolve independently) | Is the coupling doing work, or is each agent nearly autonomous? |
| **A4** | Fully-connected agent graph (all 21 pairs, all types) | Is the *declared sparsity* helping, or would a dense graph do better? |
| **A5** | Bottleneck/rigid-body coupling removed | Does the global loop matter, or is local physics sufficient over 60 s? |
| **A6** | Frozen donor (stage 2a) vs. LoRA (stage 2b) | Was the bootstrap fine-tuning worth its cost? |
| **A7** | Single-rate stepping (everything at $10^{-3}$ s) | What does multi-rate cost in accuracy, and what does it buy in time? |

**A4 deserves emphasis.** It is the direct test of [[edge-generation-atlas-0.1]]'s declared-graph premise. If a fully-connected agent graph performs *better*, the declared edge list is over-constraining the model and the RL-edge-discovery direction (Mechanism B, currently deferred) becomes considerably more urgent than "after scenario 2." That would be a substantive redirection of the project, discoverable cheaply here.

**A2 deserves emphasis for the opposite reason.** If disabling the conservation constraint changes nothing, then the constraint is not load-bearing and [[conservation-as-constraint-atlas-0.1]]'s central claim — that imposed interface conservation is what holds the composition together — is not supported by evidence. That is worth knowing regardless of which way it falls.

## 3.5 Decision rule after Phase 4

| Outcome | Action |
|---|---|
| All gates pass, ablations show A1/A3/A4 favour the full architecture | Proceed to [[impl-atlas-0.1-phase5-expansion]] |
| Gates pass but A1 matches full Atlas | **Stop and reconsider.** The decomposition is not earning its complexity; investigate before adding scenarios |
| Seam divergence (A2 shows the constraint is necessary but insufficient) | Escalate the constraint: hard-project momentum and energy too, re-run |
| Rollout blow-up | Reduce exchange cadence (more frequent coupling) before touching the experts; instability is more often a coupling problem than a model problem |
| No speedup | Profile before redesigning — the likely cause is Python-level per-substep overhead in the multi-rate loop, not the networks |

---

# 4. Acceptance tests (M6, M7)

**M6 — stable rollout:**
- 60 s free-running ascent completes on all held-out configs, 5 seeds, no NaN.
- $\varepsilon$ for every agent stays below 0.30 at the 60 s horizon.
- $\Delta_m<10^{-2}$; $R_{\dot m}$ at machine epsilon throughout.

**M7 — report:**
- All four physical-milestone checks executed and reported (pass or fail).
- All seven ablations run.
- Speedup measured, both directions of the comparison warm.
- The "what failed / what was not tested" section written.

---

# 5. Pitfalls

- **Injecting ground truth mid-rollout.** Invalidates the entire result; easy to do accidentally when reusing training-loop code for evaluation.
- **Reporting a single scalar error.** Hides the compounding behaviour that is the actual question.
- **Evaluating on training configs.** Corner-case held-out configs exist for this reason.
- **Single seed.** Not a measurement.
- **Truncating the horizon where the curve turns bad.** Report the whole curve.
- **Reading success as evidence of generality.** It is evidence of correct wiring. [[transfer-learning-fine-tuning]]'s diversity principle is explicit that in-distribution success overstates generalization.
- **Skipping A4.** It is the cheapest test of the architecture's central premise.

---

## See Also

- [[case-study-rocket-ascent-2d-atlas-0.1]] — the design-level statement of this experiment
- [[00-atlas-0.1-overview]] — the proposition being tested
- [[impl-atlas-0.1-phase5-expansion]] — gated on this phase passing
- [[atlas-0.1-implementation-log]] — where results get recorded
- [[transfer-learning-fine-tuning]] — the diversity principle and why single-scenario success is limited evidence
