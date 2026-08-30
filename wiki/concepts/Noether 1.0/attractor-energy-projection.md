# Attractor-Energy Projection — the driven-dissipative analog of S1.1

**Type:** Concept — initial model, rollout-stability mechanism (folder: Noether 1.0)
**Status:** Build-layer-validated (Phase 2 / RBC, 2026-07-03). Extends the S1.1 conservation projection ([[open-architectural-problems]], [[initial-model-architecture]] stage 5b) from *conservative* to *driven-dissipative* systems. Ships in the build layer as `NoetherRBC.project_energy`.
**Related Concepts:** [[initial-model-architecture]], [[noether-1.0-rbc]], [[open-architectural-problems]], [[normalization-scheme]], [[symmetric-attention-physics]], [[skew-symmetric-attention]], [[autoregressive-rollout-stability]]
**Related Summaries:** [[dissipative-hamiltonian-neural-networks]], [[deep-memory-dissipative]], [[walrus-paper]], [[gphyt-physics-foundation-model]]

---

## One-sentence summary

For a **driven-dissipative** system (no conserved energy), the correct analog of the S1.1 "project onto the conserved manifold" step is to **softly project the state onto the known mean-energy shell of its statistical attractor** — removing the slow autoregressive energy drift without conserving anything.

---

## The problem S1.1 does not solve

The architecture's S1.1 projection ([[open-architectural-problems]], [[initial-model-architecture]] §5b) converts *depth*-conservation into *time*-conservation by projecting the predicted state onto $\{h : \hat C(h) = C_t + \text{flux}\}$ for the known invariants $C$ (mass, momentum, energy). It is the load-bearing rollout-stability mechanism for **conservative** physics (Hamiltonian, Euler, weakly-viscous Burgers).

**Driven-dissipative systems break its premise.** Rayleigh–Bénard convection ([[noether-1.0-rbc]]) has *no* conserved momentum or energy: buoyancy injects energy, viscosity and the wall boundary layers remove it, and the attractor is the *balance* of the two. So $C_{\text{energy}}$ is not conserved, the projection has nothing to enforce, and [[noether-1.0-rbc]] §4.9 correctly notes S1.1 is not wired into the RBC forward pass at all.

Yet long-horizon rollout still needs an energy anchor. Empirically (build layer, RBC-base), after the two other stabilizers are in place — the Ra-aware frozen normalization ([[normalization-scheme]]) and noise-injected push-forward ([[autoregressive-rollout-stability]]) — the autoregressive kinetic energy still **drifts slowly upward**: a per-step energy injection far *below* the single-step error floor ($\sim0.01$–$0.04\%$/step, vs a $5.7\%$ single-step VRMSE) that push-forward cannot resolve, because the trainable horizon ($\sim8$ steps) is two orders of magnitude shorter than the drift's manifestation horizon ($\sim$hundreds of steps). Over 1000 steps this compounds to a $1.4$–$3.5\times$ energy growth.

## The insight: project onto the attractor, not onto a conservation law

A driven-dissipative system does not conserve energy, but its **statistically-stationary attractor has a well-defined mean energy** $E^\*$. For RBC this is a known function of the control parameters — indeed the same nondimensionalization ([[normalization-scheme]]) that makes the fields $O(1)$ *already estimates it*: the fitted velocity scale is a power law
$$\mathrm{rms}\,|\mathbf u| \;\approx\; a\,\mathrm{Ra}^{b}\qquad (\text{RBC: } a\approx0.084,\ b\approx0.57),$$
so the mean kinetic energy of the attractor is
$$E^\*(\mathrm{Ra}) \;=\; \langle |\mathbf u|^2\rangle \;=\; 2\,\big(a\,\mathrm{Ra}^{b}\big)^2$$
(the factor $2$ because the fitted scale is a *per-component* rms). The stabilizer is then a **soft projection onto this shell**, applied each rollout step:
$$\mathbf u \;\leftarrow\; \mathbf u\cdot\Big(\frac{E^\*}{\langle|\mathbf u|^2\rangle}\Big)^{\lambda/2},\qquad \lambda\in(0,1].$$

- $\lambda = 1$ hard-clamps the energy to $E^\*$ every step; small $\lambda$ (0.05–0.1) is a **gentle relaxation** that corrects the slow drift while leaving the fast turbulent energy fluctuations of the attractor intact.
- It is a **per-sample scalar rescale**, so it commutes with the differential operators: if $\nabla\!\cdot\!\mathbf u = 0$ then $\nabla\!\cdot\!(c\,\mathbf u)=0$, and the no-slip walls $\mathbf u=0$ map to $0$. **Divergence-free and boundary conditions are preserved exactly** — the same structural-safety property that made the stream-function decode (P2.4) and S1.1 composable.

## Why this is the right analog

| | **S1.1 (conservative)** | **Attractor-energy projection (driven-dissipative)** |
|---|---|---|
| Invariant | conserved total $C_t + \text{flux}$ | stationary attractor statistic $E^\*$ |
| Manifold | $\{\hat C(h)=C_{\text{target}}\}$ | energy shell $\{\langle\lvert \mathbf u\rvert ^2\rangle = E^\*\}$ |
| Target source | boundary-flux bookkeeping | known $E^\*(\text{regime})$ from nondimensionalization |
| Strength | hard (exact) | soft relaxation $\lambda$ (attractor fluctuates) |
| Physics-encoding level | 5 (hard constraint) | 3 (test-time guidance) / 5 if trained in |

Both are the *same move* — "project the prediction onto the manifold the physics is known to live on" — differing only in *which* manifold the regime supplies. This is the projection-side counterpart to the observation ([[symmetric-attention-physics]], [[skew-symmetric-attention]]) that RBC's physics lives in *dissipation and forcing balance*, not conservation: the conservation backbone is inert for RBC, and the attractor-energy shell is what actually anchors the long-horizon dynamics.

## Build-layer evidence (RBC, 2026-07-03)

On the RBC-base checkpoint, a gentle projection ($\lambda=0.05$) flattens the 1000-step kinetic-energy growth from $\mathbf{4.4\times \to \sim1.0\times}$ (in-range Ra) and $4.05\times\to1.13\times$ (OOD $\mathrm{Ra}=10^6$), while the 9-step rollout VRMSE is **unchanged** ($0.031\to0.032$). Because it is a scalar rescale, the normalized divergence stays at the float32 floor throughout. It requires **no retraining** — it works on existing checkpoints at inference.

This landed only after two prerequisites, which are worth recording as the full driven-dissipative rollout-stability stack:
1. **Ra-aware frozen normalization** ([[normalization-scheme]]) — replaces per-instance normalization, which is *scale-equivariant* and therefore chains the output energy to the input energy, removing any restoring force and causing a strong multiplicative blow-up (the root cause of the original RBC divergence).
2. **Noise-injected push-forward** ([[autoregressive-rollout-stability]], S1.5) — the trained-stability complement; flattens short-to-medium horizons but has diminishing returns on the slow drift.
3. **Attractor-energy projection** (this page) — removes the residual slow energy mode that (1) and (2) leave.

---

## [AI Inference]

**[AI Inference]:** This generalizes beyond energy and beyond RBC. *Any* driven-dissipative attractor with a **known stationary statistic** — kinetic energy, enstrophy, the Nusselt number, a spectral slope, a passive-scalar variance — admits the same soft-projection anchor onto that statistic's expected value. The general principle is **prefer projecting onto the attractor's known invariants of the statistics over enforcing conservation laws that do not hold**, the driven-dissipative sibling of the tiered-invariant idea in S1.1. The natural next form is **flux-balance projection**: rather than pinning total energy, softly enforce the stationary *budget* (buoyant production $=$ viscous dissipation; the exact Grossmann–Lohse / Nusselt identities), which is the true conservation-law of a driven-dissipative steady state.

**[AI Inference]:** The projection is also a minimal, *correct* answer to the recurring "driven systems need dissipation" critique of a conservation-first backbone. One does **not** need to rewrite the backbone into a state- and scale-dependent dissipation operator $R(h)$: the observed instability is a single slow global mode (the total-energy drift), so a **global rank-1 relaxation onto the known energy shell** is the matched, lowest-cost fix. A full $R(h)$ operator would be over-engineering for the mode that actually misbehaves — evidence, empirically, that the conservation backbone plus a targeted attractor projection covers both conservative and driven-dissipative regimes without an architectural fork. This is consistent with the port-Hamiltonian reading in [[symmetric-attention-physics]]: the projection supplies exactly the dissipative $R$ part, but as a rank-1 shell relaxation rather than a dense operator.

**[AI Inference]:** Because $E^\*$ comes from the *nondimensionalization* rather than from data seen during dynamics training, the projection degrades gracefully out of distribution: at $\mathrm{Ra}=10^6$ (well above the training range) the extrapolated shell is $\sim\!12\%$ low, so the energy settles at $1.13\times$ rather than $1.0\times$ — still flat, not drifting. Where OOD accuracy matters, $E^\*$ should be taken from the true attractor (a short reference solve or an in-context estimate) rather than the extrapolated power law.

---

## See Also
- [[initial-model-architecture]] — S1.1 conservation projection (stage 5b), the conservative-regime original
- [[open-architectural-problems]] — S1.1 decision; tiered invariants (known/discovered)
- [[noether-1.0-rbc]] — the as-built RBC model report; §4.8–4.9 (why conserved-total S1.1 is not wired in, and where this projection slots in as §4.8.3), §5.2 (the fix in context)
- [[normalization-scheme]] — the Ra-aware frozen scale that both fixes the blow-up and *supplies* $E^\*(\mathrm{Ra})$
- [[autoregressive-rollout-stability]] — push-forward / noise injection, the trained-stability complement
- [[symmetric-attention-physics]] / [[skew-symmetric-attention]] — port-Hamiltonian $J-R$ reading; why the conservation core is inert for RBC
- [[dissipative-hamiltonian-neural-networks]] — learned dissipation as a contrast to this projection-based approach
