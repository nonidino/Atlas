# Training Curriculum & Loss Composition (initial model)

**Type:** Concept — initial model master spec (folder: Noether 1.0)
**Status:** Binding spec for [[open-architectural-problems]] Problem 6 (loss function). Training-time companion to [[pipeline-contract]] (forward pass): this page fixes which loss terms exist, how they're weighted, and in what order they're introduced.
**Related Concepts:** [[open-architectural-problems]], [[initial-model-architecture]], [[pipeline-contract]], [[graph-tokenizer]], [[intelligent-patching]], [[normalization-scheme]]
**Related Summaries:** [[poseidon-pde-foundation-model]], [[gns-graph-network-simulators]], [[walrus-paper]], [[pisd-physics-informed-spectral-diffusion]]

---

## Why loss composition is hard here

Five separate design decisions each demand their own loss term, in different spaces, maturing at different points in training:

- **S1.1** wants tier-2 (discovered) invariants found and validated — a *latent* signal.
- **S3.1** wants the high-frequency residual channel to reconstruct sharp features — a *fidelity* signal (but this is the separate G0 pretraining gate, not a per-step term).
- **S5.1** wants the diffusion core to match small-scale *statistics* — a *generative* signal.
- The base regression wants per-field-accurate residuals — a *physical-space* signal spanning many orders of magnitude across channels.
- Rollout stability (S1.5) wants multi-step coherence — a *temporal* signal.

A flat sum of these from step 0, with fixed weights, is fragile: it conflicts with the breadth-from-day-one commitment (weights tuned for one field set don't transfer to a new one), and several terms are only *meaningful* once an earlier one has partly converged (you can't usefully penalize multi-step drift before single-step prediction is decent). The design below (adopted [[open-architectural-problems]] S6.1/S6.3/S6.4/S6.5) resolves both problems: **automatic per-channel weighting within a group**, and **staged introduction across groups**.

---

## The four loss groups

| Group | Terms | Space | Introduced after |
|---|---|---|---|
| **A — Core state fit** | residual regression + spectral/statistical loss, per-channel | physical + Fourier | G0 tokenizer gate passes |
| **B — Temporal consistency** | MTP horizon loss, push-forward curriculum | physical (rollout) | Group A stable |
| **C — Conservation discovery (tier 2)** | SFA-style slowness + non-degeneracy penalty | latent | Group B; trajectories meaningful |
| **D — Generative core** | diffusion denoising (score-matching) on the residual channel | latent (residual) | Groups A–C converged |

**Not in this table, and not curriculum-staged:** tier-1 **known** invariants (mass, momentum, energy) are enforced by the **architectural projection** ([[pipeline-contract]] Stage 7/10, [[open-architectural-problems]] S1.1) — a forward-pass operation, not a loss term. They need no weight and no phase; they are always on from the first training step. Only tier-2 **discovered** invariants need Group C's loss. The G0 reconstruction fidelity floor ([[graph-tokenizer]] S3.4) is a separate pretraining-stage objective, resolved before any of the groups below begin.

---

## Within-group weighting: homoscedastic uncertainty (Group A)

Per field channel $c$ (velocity, pressure, vorticity, …) and per domain $d\in\{\text{physical},\text{spectral}\}$, learn a log-variance $\log\sigma_{c,d}^2$ and weight:

$$\mathcal L_A = \sum_{c,d} \left[\frac{1}{2\sigma_{c,d}^2}\big\|\hat y_{c,d}-y_{c,d}\big\|^2 + \log\sigma_{c,d}\right].$$

The $\log\sigma$ term is what prevents the trivial collapse (inflate $\sigma\to\infty$ to zero out a hard channel's loss). This automatically balances channels that differ by orders of magnitude post-nondimensionalization, without hand-tuning per field type — the property that makes it compatible with the multiphysics-breadth commitment (a new field type just gets its own learned $\sigma$, no re-tuning).

---

## Cross-group weighting: staged curriculum, GradNorm in reserve

Fixed weights are avoided *within* a group (above); *across* groups, the chosen mechanism is **order**, not a learned weight, because the groups are not simultaneously meaningful:

| Phase | Groups active | Gate to advance |
|---|---|---|
| 0 (pretraining) | G0 tokenizer fidelity (S3.4) | encode→decode error below discretization floor |
| 1 | A only | single-step residual + spectral loss converged |
| 2 | A + B | multi-step rollout ($1\to2\to4\to8$, S1.5) stable |
| 3 | A + B + C | trajectories accurate enough to probe for invariants |
| 4 | A + B + C, backbone frozen/lightly fine-tuned; **D** trains separately | — (terminal phase) |

The tier-1 projection ([[pipeline-contract]] Stage 7/10) runs in the forward pass throughout **all** phases, including 0 — it is architecture, not curriculum.

**GradNorm** (dynamic gradient-magnitude rebalancing across groups) is held in **reserve**: if staged introduction alone leaves one group's gradients dominating after it comes online, use GradNorm to rebalance rather than hand-tuning a fixed cross-group weight — but it is a diagnostic fallback, not the default mechanism (extra backward passes, extra hyperparameters).

---

## Group C: discovering hidden invariants (SFA-style)

**[AI Inference]:** "discover a conserved quantity" is a **slowness-learning** problem: a genuine invariant is a probe $\hat C_\phi(h)$ that is *constant along a given trajectory* but *not a trivial global constant* (it must vary with which trajectory/initial condition you're in, or it's measuring nothing). This is exactly the classic **Slow Feature Analysis** framing, extended with a discriminative term for the non-degeneracy the [[open-architectural-problems]] second-pass audit flagged as missing:

$$\mathcal L_C = \underbrace{\mathbb E_{\text{traj}}\big[\mathrm{Var}_t(\hat C_\phi(h_t))\big]}_{\text{slowness: constant within a trajectory}} \;-\; \beta\,\underbrace{\mathrm{Var}_{\text{traj}}\big(\mathbb E_t[\hat C_\phi(h_t)]\big)}_{\text{discriminative: varies across trajectories}}, \qquad \|W_C\|=1\ \text{(fixes scale)}.$$

Minimizing $\mathcal L_C$ finds probes that are slow *and* informative — avoiding collapse onto a boring constant. This naturally surfaces **multiple** candidate invariants (the top-$k$ slowest, most-discriminative probes), matching how real systems carry more than the textbook three (2D turbulence: enstrophy; ideal MHD: magnetic helicity; integrable systems: infinitely many). A probe is **promoted to hard projection** (joining S1.1's tier-1 mechanism) only once validated: low enough variance, over a long enough horizon, across held-out trajectories.

---

## Group D: phase-separated generative training

Diffusion training does **not** start until Phase 4, after the deterministic backbone (Groups A–C) has converged — mirroring standard latent-diffusion practice (train the autoencoder/backbone, *then* train a diffusion model in its latent space, rather than jointly optimizing a regression head and a stochastic generative head on a shared, still-shifting representation). The backbone is **lightly fine-tuned**, not frozen outright, during Phase 4 — a hard freeze would prevent the diffusion head's gradients from ever correcting an upstream representation deficiency.

The diffusion model trains on the S3.1 high-frequency residual channel across **all** training trajectories (not just ones labeled "chaotic") — the S5.2 regime router is an **inference-time** cost-saving decision about *whether to invoke* diffusion, not a training-time data filter, which sidesteps needing ground-truth chaos labels to train the generative core.

---

## Decision (2026-07-01)

**Adopted:** S6.1 (within-group uncertainty weighting) + S6.3 (staged cross-group curriculum) + S6.4 (SFA-style Group C discovery loss) + S6.5 (phase-separated Group D training). **S6.2 (GradNorm)** held in reserve as a diagnostic. **S6.6 (fixed hand-tuned weights)** rejected as the sole mechanism — too brittle for the breadth goal — but implicitly present as each phase's initial constant before its within-group weighting stabilizes.

---

## [AI Inference]

**[AI Inference]:** The curriculum's phase order (A→B→C→D) is the training-time mirror of the pipeline contract's "projection sandwich" framing: just as the forward pass learns freely in the interior and enforces exactly at the boundary, training learns the *easy, well-posed* objective first (single-step fit) and only *later* asks the model to discover structure (invariants) or generate under uncertainty (diffusion) — deferring the hardest optimization problems until the representation underneath them is trustworthy. Both are instances of the same principle: don't ask a component to do a hard job on top of an unstable foundation.

---

## See Also

- [[open-architectural-problems]] — Problem 6, the audit this page resolves
- [[pipeline-contract]] — the forward-pass contract; tier-1 projection runs independently of this curriculum
- [[initial-model-architecture]] — Training plan section, Stage 5b (S1.1 projection)
- [[normalization-scheme]] — the per-channel structure this page's Group-A weighting respects
- [[graph-tokenizer]] — the S3.4 reconstruction/fidelity gate at Phase 0
- [[poseidon-pde-foundation-model]] — all2all data amplification, complementary to this curriculum
- [[gns-graph-network-simulators]] / [[walrus-paper]] — push-forward training precedent (Group B)
- [[pisd-physics-informed-spectral-diffusion]] — guided-diffusion precedent (Group D)
