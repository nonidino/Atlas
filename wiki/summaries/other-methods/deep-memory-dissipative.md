# Summary: Persistent Memory Through Triple-Loop Consolidation in a Non-Gradient Dissipative Cognitive Architecture

**Source:** `raw/Persistent Memory Through Triple-Loop Consolidation in a Non-Gradient Dissipative Cognitive Architecture.md`  
**Author:** Jianwei Lou (RailMind Systems, Germany)  
**arXiv:** 2603.27188v1  
**Date Ingested:** 2026-04-11

---

## Overview

Introduces **Deep Memory (DM)** — a non-gradient persistent memory mechanism for **dissipative cognitive architectures** where computational units are stochastically replaced when they exhaust their energy budget. DM maintains persistent context-specific representations through a **triple-loop consolidation cycle** (recording → seeding → stabilization), requiring **discrete expert routing via Mixture-of-Experts (MoE) gating** as a causal prerequisite.

---

## Dissipative Cognitive Grid

The system consists of $N$ units on a lattice, each with content vector $z_i \in \mathbb{R}^D$ and energy $E_i$. At each cycle:
1. **Energy allocation:** Input $x_t$ provides energy based on local activity.
2. **Activation:** Units with $E_i > E_\text{min}$ activate; output $a_i$ modulated by homeostatic threshold $\varphi_i$.
3. **Content update:** $z_i \leftarrow (1-\alpha)z_i + \alpha\bar{x}_i$ (EMA toward neighborhood-weighted input).
4. **Metabolic cost:** $E_i \leftarrow E_i - c(a_i)$.
5. **Stochastic replacement:** Units with $E_i < E_\text{min}$ are replaced with random state.

No gradients, no backpropagation — all learning is local Hebbian-type dynamics.

---

## Discrete Expert Routing (MoE)

Units are partitioned into $K$ expert groups. At each step, expert $k^*$ is selected by:
$$k^* = \arg\max_k \operatorname{sim}(x_t, \mu_k)$$
where $\mu_k$ is the running centroid for expert $k$. Only group $k^*$ activates. This is **hard routing** — no soft mixing.

**Causal necessity:** Without discrete routing, centroids converge to the same value, making stored memories identical. Mutual information between context and expert binding: $\text{MI} = 1.10$ (with DM + binding) vs. $0.001$ (with binding destroyed).

---

## Deep Memory: Triple-Loop Consolidation

**Loop 1 — Recording:** For each expert group $k$, maintain centroid $m_k \in \mathbb{R}^D$:
$$m_{k^*} \leftarrow (1-\gamma)m_{k^*} + \gamma\bar{z}_{k^*}$$

**Loop 2 — Seeding:** When a unit $i$ in group $k$ is replaced (energy depleted), with probability $\rho$:
$$z_i \leftarrow m_k$$
This seeds the new unit with the expert's stored representation rather than random noise.

**Loop 3 — Stabilization:** Continuous seeding re-enters stored representations back into the active pool, counteracting dissipative drift. One-shot seeding (without continuous re-entry) fails — stabilization loop is necessary.

---

## Key Results (~970 simulation runs)

- **Representation quality:** $R = 0.984$ (DM) vs. $R = 0.385$ (no memory), across $n=16$ runs.
- **Reconstruction after interference:** $R_\text{recon} = 0.978$ with continuous seeding; one-shot seeding fails.
- **Operating envelope:** Characterized in $(K, \rho)$ space with identifiable phase boundaries (pass / degraded / failure), $n=350$ runs.
- **Scheduling invariance:** DM quality invariant across 5 qualitatively different context-scheduling patterns.
- **Minimal mechanism:** Recording × seeding is the irreducible critical dyad (factorial ablation, $n=40$).
- **Comparison with non-gradient baselines:** Outperforms Hopfield networks and Echo State Networks (ESN) under matched turnover conditions.

---

## Parallels to Biological Memory

The triple-loop mechanism has **functional parallels to hippocampal consolidation**:
- Recording ↔ rapid hippocampal encoding.
- Seeding ↔ hippocampal-to-neocortical memory transfer (slow consolidation).
- Stabilization ↔ memory reconsolidation / ongoing neocortical re-entry.

The Complementary Learning Systems (CLS) theory (McClelland et al.) is cited as the functional blueprint; DM provides a computational implementation in non-gradient systems.

---

## Relevance to Physics Foundation Model Goal

This work is somewhat peripheral to the core physics simulation focus, but has important implications for **cognitive architecture** aspects of a PFM:

1. **Continual learning without catastrophic forgetting** — a PFM must be incrementally updated as new physics domains are encountered. DM provides a non-gradient mechanism for this.
2. **Expert specialization** — the MoE + hard routing structure ensures different experts genuinely specialize in different input contexts, which parallels how physics experts might specialize in different equation families.
3. **Metastability** — the architecture demonstrates that productive computation can occur in systems with continuous stochastic perturbation, which is an interesting model for robustness.

**[AI Inference]:** The DM framework could inspire a **continual learning module** for a PFM that maintains a persistent memory of previously encountered physical systems without requiring full retraining. The seeding mechanism — initializing new units from stored expert centroids — is analogous to warm-starting new task heads in a foundation model from related task embeddings. The operating envelope characterization (phase boundaries in parameter space) provides a principled way to design the memory capacity vs. turnover rate tradeoff.

---

## Links

- [[mixture-of-experts]] — discrete expert routing mechanism
- [[transfer-learning-fine-tuning]] — adjacent to continual learning
- [[physics-foundation-models]] — target application context
