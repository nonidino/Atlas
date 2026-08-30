# Discovered Conservation (Noether 1.1)

**Type:** Concept — model portion, NEW in 1.1 (folder: Noether 1.1)
**Status:** Design. The heart of the 1.1 generality thesis: **the model learns which quantities are conserved and enforces each in proportion to how invariant it actually is in data** — rather than hard-coding domain-specific conservation laws. Promotes the S6.4 SFA-style discovery loss ([[noether-1.0]] §"Why Noether") and the learned readout $\hat C(h)$ from auxiliaries to the *primary* conservation mechanism, and demotes 1.0's hand-specified hard projections (mass, div-free) to "the confident end of a discovered spectrum."
**Related Concepts:** [[00-noether-1.1-overview]], [[backbone-1.1]], [[decoder-1.1]], [[field-descriptors-1.1]], [[action-based-noether-enforcement]], [[attractor-energy-projection]], [[autoregressive-rollout-stability]], [[in-context-learning-physics]], [[open-architectural-problems]], [[training-curriculum]]
**Related Summaries:** [[noether-networks]], [[hamiltonian-neural-networks]], [[dynami-cal-graphnet]]

---

## What it does — intuitively

Do not tell the model that mass, momentum, or energy is conserved — that is domain knowledge that is *wrong* as often as it is right (energy is not conserved in a driven convection cell; divergence is not zero in a compressible gas). Instead, let the model **watch the data and measure what stays constant**, keep a shortlist of candidate conserved quantities, and enforce each **as hard as the evidence warrants**: something that never drifts in the data gets projected exactly; something that drifts a little gets a soft nudge; something that clearly is not conserved gets left alone.

This is conservation as **pattern recognition applied to invariants** — the model *recognizes* what is conserved instead of being told. It is what lets the same architecture be exactly-conservative for an isolated N-body system, driven-dissipative for convection, and correct for niche or newly-invented physics where the conservation laws are not known in advance.

## What it does — mathematically

Maintain candidate invariant functionals $\{C_k(\text{state})\}$ — some parameterized (a readout $\hat C_k(h)=w_k^\top h_{\text{top}}$ on the global supernode, or a small net), some symbolic seeds. **Seeds are written generically against a field's descriptor $\theta_f$ ([[field-descriptors-1.1]]), never against a named field:** "mass-like" $=\sum_f\mathbb 1[\mathrm{rank}(\theta_f)=0]\int f$, "momentum-like" $=\sum_f\mathbb 1[\mathrm{rank}(\theta_f)=1]\int f$, "divergence-like" $=\nabla\cdot(\text{the rank-1 field, if any})$. This genericity — templated on tensor rank, not on field name — is what lets discovery **transfer across system types** rather than being re-specified per simulation; see "Generalization to new systems" below.

**1. Discovery — measure invariance on trajectories.** For each $C_k$, measure its drift along training trajectories,
$$D_k = \mathbb E_{\text{traj},t}\big[(C_k(\text{state}_{t+1}) - C_k(\text{state}_t))^2\big],$$
and *search* for new parameterized $C_k$ minimizing $D_k$ subject to a non-triviality constraint ($\mathrm{Var}(C_k)>0$, so $C_k\neq\text{const}$) and mutual orthogonality (so distinct $C_k$ are not the same invariant rediscovered) — the SFA / "slowness = conservation" objective. Route 1 (invariance-on-trajectories) ships first; route 2 ([[action-based-noether-enforcement]]: discover a symmetry, let Noether hand back the current) is the deeper, later target.

**Condition the search — whiten first.** For a linear/quadratic readout $C_k(h)=w_k^\top h_{\text{top}}$ this objective *is* a generalized eigenproblem — minimize $w^\top\Sigma_\Delta w$ subject to $w^\top\Sigma w=1$ and mutual $\Sigma$-orthogonality, with $\Sigma=\operatorname{Cov}(h_{\text{top}})$, $\Sigma_\Delta=\operatorname{Cov}(\Delta h_{\text{top}})$ — whose conserved directions are the *smallest* generalized eigenvectors of $(\Sigma_\Delta,\Sigma)$. This is Slow Feature Analysis, and it wants a normalization step not as hygiene but as a correctness condition, for three reasons: **(1)** whitening $\Sigma\to I$ turns the ill-conditioned *generalized* eigenproblem into a well-conditioned *ordinary* symmetric one and **implements the non-triviality constraint for free** (unit variance $=$ the $\mathrm{Var}(C_k)>0$ requirement above); **(2)** the physics latent spans enormous dynamic range ([[normalization-scheme]]'s 30-orders-of-magnitude point), so an unwhitened $\Sigma$ corrupts precisely the *smallest* eigenvectors — the conserved directions we are trying to recover; **(3)** this design's deliberate feature **redundancy** (design invariant 3, [[00-noether-1.1-overview]]) *guarantees* $\Sigma$ is near-rank-deficient (redundant channels are collinear), so plain inversion is unstable — the step must be **shrinkage / reduced-rank whitening** $(\Sigma+\varepsilon I)^{-1}$ or a null-space-dropping whitening, which is how the redundancy thesis is reconciled with the conditioning the eigenproblem needs. For a nonlinear readout the same role is played by a decorrelation/whitening layer on the penultimate features (BatchNorm-without-shift), with orthogonality enforced as a decorrelation penalty — standard deep-SFA. The whitening statistics are themselves regime-dependent, so they are computed conditioned on $z_{\text{cond}}$ (or per-batch within a regime).

**2. Enforcement — a hybrid keyed to measured drift.** Each $C_k$ gets a **conservation strength** and two channels:
- **Learned-soft (always on):** a penalty $\lambda_k\, D_k$ in the loss, with $\lambda_k$ learned (uncertainty-weighted, [[training-curriculum]]). Safe even if $C_k$ is only approximately conserved — a wrong soft weight costs a little accuracy, nothing catastrophic.
- **Threshold-gated-hard (conditional):** the S1.1-style projection $x^*=\hat x - M^{-1}A^\top(AM^{-1}A^\top)^{-1}(A\hat x - b)$ is applied **only** for $C_k$ whose *held-out* drift $D_k<\varepsilon$ (genuinely conserved) and whose projection is cheap. A mis-discovered hard constraint injects systematic bias, so hard enforcement is gated on evidence, not on a learned scalar the model could game.

**3. Constraint-type cost hierarchy** (determines *how* $C_k$ is enforced):

| Type | Example | Enforcement |
|---|---|---|
| Global scalar integral | mass, energy, momentum | cheap — one-line projection or soft loss |
| Local / pointwise differential | $\nabla\cdot u=0$ | **expensive** hard projection (Leray/elliptic solve); default **learned-soft-only** |
| Inequality | positivity, entropy $\uparrow$ | different machinery — deferred |

**First-cut scope (decided):** attempt **local (divergence) discovery first**, on NS velocity fields — a learned-weight soft penalty on $\nabla\cdot u$ whose $\lambda$ goes to zero on compressible data by itself — then extend to RBC's global invariants. This front-loads the compressible-vs-incompressible generality demo.

## Physics relevance

- **The right sibling for driven-dissipative physics.** Energy in RBC drifts, so $\lambda_{\text{energy}}\to0$ correctly — no wall fights the physics. Where a true invariant statistic exists instead (an attractor energy shell), that is the [[attractor-energy-projection]] mechanism, now one discovered $C_k$ among many rather than a bolted-on special case.
- **Divergence as a discovered, not assumed, law.** $\nabla\cdot u$ is just one $C_k$: incompressible → strength high → enforced; compressible → strength zero → ignored. The compressible/incompressible fork disappears from the code and becomes a measurement ([[decoder-1.1]] can then switch on a structure-preserving decode mode when it is confidently zero).
- **Niche / invented physics.** For systems whose conservation laws are unknown, discovery is the *only* way to get both accuracy and adaptability — the model finds tier-2 invariants with no closed form, exactly the case [[noether-1.0]]'s AI-inference predicted the learned readout becomes load-bearing.

## How fundamental constants are included

- **Constants gate which invariants are even candidates.** A symmetry a constant respects (or breaks) tells the search where to look — an isotropy-breaking gravity vector rules total-vertical-momentum out of the conserved set while leaving horizontal momentum in ([[conditioning-and-constants-1.1]]). Constants act on the *conservation set*, not just the dynamics.
- **Strengths are conditioned.** $\lambda_k$ and the gate depend on $z_{\text{cond}}$, so the same discovery machinery enforces div-free at low Mach and relaxes it at high Mach without a code change.
- **Momentum-antisymmetry is the one default-on structural prior** ([[backbone-1.1]]'s gate $g_\tau$), kept because it is cheap and central to the particle case — but it too is *gateable*, so it is a strong discovered-conservation prior, not an inviolable law.

## How this behaves during rollout — soft (weight-shaping) vs. hard (live correction)

At rollout there is no ground truth and no backward pass, so "the soft loss helps" cannot mean the loss executes during inference. It means one of two distinct mechanisms, which act differently and should not be conflated.

**Soft — indirect, always present, nothing computed at inference.** Minimizing $\sum_k D_k/2\sigma_k^2+\log\sigma_k$ over training is gradient pressure on every weight that touches the state, so the converged backbone/decoder weights are ones whose *own single-step predictions* already tend not to move the low-$\sigma_k$ quantities. At rollout you run the model, not the loss — but the model **is** what the loss shaped. This is ordinary implicit regularization: the penalty never runs at test time, but the function it produced carries the effect forward.

**The single-step gap, and why push-forward closes it.** $D_k$ is measured on single steps. A soft loss trained only on one-step drift guarantees the model is well-behaved *locally* on the training distribution — it says nothing about $K$ compounding autoregressive steps. Closing this needs $D_k$ measured **through the push-forward unroll** (roll $K$ steps with noise injection, per [[autoregressive-rollout-stability]] / [[training-curriculum]]'s existing curriculum, and backprop through the whole unroll), not single-step training alone. This is not hypothetical: [[noether-1.0-rbc]] §5.2 already measured exactly this pattern for one hand-picked quantity (kinetic energy) — noise-injected push-forward "trains stability directly into the weights," confirmed by the raw (uncorrected) rollout staying flat at 200 steps. Discovered conservation generalizes that one validated case to the whole discovered $\{C_k\}$ set, rather than inventing a new mechanism.

**Hard — direct, re-executed every rollout step.** For any $C_k$ past the held-out drift gate, the projection $x_t^*=\hat x_t-M^{-1}A^\top(AM^{-1}A^\top)^{-1}(A\hat x_t-b)$ is **literally recomputed at every step**: decode $\hat x_t$, evaluate $C_k(\hat x_t)$, correct, feed $x_t^*$ — not $\hat x_t$ — into the next autoregressive step. This is a live, mechanical correction, not a training artifact, and it targets a different failure mode than push-forward. [[noether-1.0-rbc]] §5.2's table shows the two are **complementary, not redundant**: push-forward flattened short/medium-horizon drift; the explicit attractor-energy projection flattened a *slower residual* drift only visible past ~300 steps that push-forward alone missed. Expect the same split here — soft gets most of the way cheaply, hard cleans up what a bounded training horizon can't see.

**A third, passive use.** For $C_k$ not hard-gated, $C_k(x_t)$ can still be *measured* through a rollout purely as a diagnostic — no correction, just logging. This is how the held-out gate itself gets re-evaluated over time, and (below) how a new system's trustworthiness gets assessed without committing to correcting it.

## Generalization to new systems — three tiers, decreasing confidence

**Tier A — a new parameter regime of a known system type** (an untrained Ra; crossing from low to high Mach). The easy case: $\sigma_k$ (hence enforcement strength) is a **function of $z_{\text{cond}}$**, not a fixed scalar per invariant — "$\lambda_k$ and the gate depend on $z_{\text{cond}}$" (above). A never-seen constant value is just a new point in the same continuous conditioning space, interpolated/extrapolated the way the rest of the conditioned backbone already is ([[conditioning-and-constants-1.1]]). No new mechanism required.

**Tier B — a new system type, from a known field/edge vocabulary in a new combination** (fluids and particles seen separately, now coupled; a field set in an unseen arrangement). This is why the symbolic seeds must be written generically against $\theta_f$'s rank/parity tags (amended above), never against a named field: since $z_{\text{sim}}=\operatorname{pool}_f\theta_f\Vert z_{\text{cond}}$ already generalizes "what kind of simulation is this" across arbitrary field sets ([[field-descriptors-1.1]]), a generic seed re-instantiates automatically on whatever rank-0/rank-1 streams the new system exposes — no new code path, because the seed was never field-specific to begin with. The learned readout $\hat C_k(h)=w_k^\top h_{\text{top}}$ inherits the same generality, since $h_{\text{top}}$ already pools over whatever structure is present.

**Tier C — genuinely novel-form physics, zero training exposure.** Honest limit: route 1's search (the SFA-style minimize-$D_k$ search) runs **during training**, across the training distribution — there is no mechanism specified for discovering a new invariant *form*, live, on a single unseen rollout with no gradient step. A truly alien system at pure zero-shot inference gets only Tier A/B's extrapolation, with no transfer guarantee.

A cheap, currently-unspecified extension is worth flagging as a concrete next step: since the symbolic seeds are pure function evaluations (no gradient needed), $D_k$ for every already-known candidate can be **measured directly on whatever short context trajectory is available for the new system, at inference time**, and used to recalibrate the trained prior $\sigma_k(z_{\text{cond}})$ for *this rollout specifically* (empirical-Bayes-style: trust the trajectory in front of you a bit more than the generic training-time prior) — zero new architecture, just running the existing measurement online instead of only offline, in the same spirit as [[in-context-learning-physics]]. What this **cannot** do is discover an invariant whose form was never on the candidate list — that needs live, in-context tailoring, which is exactly what [[noether-networks]] was built for ("meta-learn a space of conserved quantities during training, then tailor to the specific one at test time") and is what route 2 ([[action-based-noether-enforcement]]) is the deeper path toward. Both remain unbuilt — Tier C should be read as **partially open**, not solved.

## Open items / risks
- **Online recalibration of $\sigma_k$ from a new system's context trajectory (Tier B/C generalization) is proposed but unspecified** — cheap in principle (pure function evaluation, no gradient), not yet part of the training/inference contract; needs a concrete combination rule (e.g. empirical-Bayes shrinkage) before it's implementable.
- **Discovery is underdetermined** — trivial ($C=\text{const}$) and degenerate (many near-conserved combos) solutions; needs the non-triviality + orthogonality constraints **and** trajectory diversity. The off-attractor, energy-diverse corpus already built for RBC ([[noether-1.0-rbc]] §5.1) is exactly the data diversity this needs. Numerically those constraints are realized by whitening + shrinkage of the state covariance (see "Condition the search" above); without it the redundant-feature latent makes the eigenproblem's smallest — i.e. conserved — directions the least trustworthy.
- **Hard-projecting a mis-discovered invariant is dangerous** — hence the held-out-drift gate and default-to-soft.
- **Route 2 (symmetry-first Noether) is unbuilt** — the principled, most-general path; scheduled after route 1 validates.

## See Also
- [[action-based-noether-enforcement]] — route 2: symmetry discovery → Noether current
- [[attractor-energy-projection]] — a single discovered $C_k$ for driven-dissipative attractors
- [[backbone-1.1]] — the gated antisymmetric-flux channel this coordinates with
- [[training-curriculum]] — where the $\lambda_k$ live in the loss composition
- [[noether-networks]] — meta-learned conserved quantity, test-time tailoring; the literature answer to Tier C
- [[field-descriptors-1.1]] — the rank/parity tags generic seeds are templated on; $z_{\text{sim}}$, used in Tier B generalization
- [[autoregressive-rollout-stability]] — push-forward / noise injection, the mechanism behind the soft channel's rollout effect
- [[in-context-learning-physics]] — the online-recalibration proposal's closest existing precedent
- [[conservation-as-constraint-atlas-0.1]] — the direct contrast in the [[00-atlas-0.1-overview]] track: conservation *imposed* at declared interfaces instead of *discovered*, appropriate when the graph structure is given rather than inferred
