# Possible Architectures for Benchmarking

**Type:** Concept / Planning Note  
**Date:** 2026-04-13  
**Related Concepts:** [[pfm-architecture-approaches]], [[physics-foundation-models]], [[transformer-architectures]], [[neural-operators]], [[diffusion-models-physics]], [[autoregressive-rollout-stability]], [[message-passing-belief-propagation]], [[equivariant-gnns]]

---

## Overview

This page proposes a set of five architectures for direct empirical comparison — a benchmark suite for the Physics Foundation Model (PFM) research program. Three are drawn from the best-supported paradigms already documented in the wiki. Two are new proposals not yet explored in the ingested literature, chosen to fill specific gaps in the current landscape.

All five are to be evaluated on a common, minimal benchmark (see §Benchmark below) before scaling.

---

## The Three Best-Supported Architectures (from existing research)

### 1. Autoregressive Next-Step Transformer

The canonical paradigm: a transformer backbone takes a sequence of prior physical states as a prompt and predicts the next state, rolled out autoregressively.

**Best evidence:** [[walrus-paper]] (1.3B params, 19 physical scenarios), [[gphyt-physics-foundation-model]] (385M params, 7× SOTA, in-context learning demonstrated)

**Why include:** The most battle-tested architecture at scale. Natural in-context learning — the model infers governing dynamics from the trajectory prompt without any equation specification. The direct LLM analog.

**Key weaknesses to probe:** Error accumulation over long rollouts; no principled uncertainty quantification.

**Current scale:** 385M–1.3B params

---

### 2. Diffusion / Generative Backbone

A diffusion model learns to sample from the distribution over physically plausible next states, rather than predicting a single deterministic output. Can be conditioned on prior trajectory, boundary conditions, or PDE residuals at inference time (DPS-style).

**Best evidence:** [[pisd-physics-informed-spectral-diffusion]] (100–10,000× lower PDE residuals; 3–15× faster than pixel-space diffusion), [[latent-diffusion-physics]] (diffusion consistently outperforms deterministic surrogates for chaotic PDEs), [[pde-transformer-paper]]

**Why include:** Consistently outperforms deterministic surrogates for chaotic and turbulent systems. Stochastic output enables ensemble-style uncertainty estimation. PISD demonstrates that the same trained model can enforce different physics constraints at inference time without retraining — a critical PFM capability.

**Key weaknesses to probe:** Inference cost (many forward passes per sample); not natively designed for long autoregressive rollouts.

**Current scale:** ~100M–1B params

---

### 3. Neural Differentiator + Numerical Integrator

Instead of predicting the next state directly, the model predicts **derivatives** of physical fields; a fixed classical solver (e.g., RK4) integrates these forward in time.

$$\hat{u}_{t+\Delta t} = \text{RK4}\!\left(f_\theta\!\left(\frac{\partial u}{\partial t},\ \frac{\partial^2 u}{\partial t^2},\ \ldots\right),\ \Delta t\right)$$

**Best evidence:** [[gphyt-physics-foundation-model]] (7× SOTA; derivative targets are smoother and more stable than full-state targets)

**Why include:** Addresses error accumulation — the numerical integrator enforces consistency between timesteps in a way pure AR prediction cannot. The 7× SOTA improvement from this design choice alone is the strongest single empirical signal in the wiki.

**Key weaknesses to probe:** Coupling to a fixed integrator family reduces flexibility; ill-defined on discrete or irregular domains.

**Current scale:** 385M params

---

## Two New Architecture Proposals

### 4. Selective State Space Model (Physics-Mamba)

A Mamba-style selective state space model (SSM) adapted for physical field rollout. The core recurrence is:

$$h_{t+1} = e^{A \Delta t}\, h_t + B \Delta t \cdot u_t, \qquad y_t = C h_t$$

where $A$, $B$, $C$ are input-dependent (selective), and $\Delta t$ is treated as an explicit architectural parameter rather than a fixed hyperparameter. The key physics adaptation: $\Delta t$ is passed as an input token, making the model naturally multi-resolution and adaptive to stiff PDEs with widely varying characteristic timescales.

**Why this fills a gap:**
- **Linear-time inference** vs. quadratic attention — directly addresses the computational cost of long-rollout inference, which no current paradigm in the wiki explicitly targets
- $\Delta t$ as an input enables the model to handle adaptive timestepping without architectural changes
- No positional encoding required — time is handled entirely by the recurrence, which is physically natural
- Mamba has matched transformer quality on long sequences at a fraction of the inference cost in NLP; this needs to be tested for physics

**Key risk:** SSMs have a limited spatial receptive field. Elliptic PDEs (e.g., Poisson, Stokes) require global spatial coupling — an SSM scanning in one spatial direction may miss long-range dependencies that attention handles naturally. Mitigation: use a 2D/axial SSM scan order, or pair with a single global attention layer.

**[AI Inference]:** The $\Delta t$-as-input design is analogous to the way GP$_{\text{hy}}$T uses derivative targets — both are attempts to make the temporal structure of the PDE explicit in the architecture rather than leaving it entirely to learned weights. The combination (Mamba backbone + derivative prediction head) may be the strongest possible version of this idea.

**Benchmark hypothesis:** Physics-Mamba should match AR transformer quality on Burgers' while being 3–5× faster at inference for rollout lengths > 200 steps.

---

### 5. Graph Neural Network with Physics Bottleneck (GNN-PB)

A message-passing GNN operating on a learned adaptive graph (not a fixed grid), with a **physics bottleneck layer** that compresses the latent state through a low-dimensional subspace spanned by known conserved quantities before expanding back to the full hidden dimension.

The bottleneck projection:
$$z_{\text{bn}} = \Pi_{\text{conserved}}\, h_\ell, \qquad h_{\ell+1} = \text{MLP}(z_{\text{bn}})$$

where $\Pi_{\text{conserved}}$ is a learned projection onto a basis that includes mass, momentum, and energy coordinates. This is a **soft architectural bias** (level 4 on the physics encoding spectrum from CLAUDE.md) — it does not enforce conservation laws exactly, but forces the information bottleneck to route through physically meaningful coordinates.

**New evidence (since original proposal):**

Two newly ingested papers substantially strengthen this proposal:

- **GNS** ([[gns-graph-network-simulators]]): Proves that a single GNN can simulate qualitatively distinct multi-material physics (water, sand, goop) and generalize to 34× more particles and 8× longer rollouts than training. This is the empirical foundation for the GNN-PB backbone — encode-process-decode with local connectivity is the correct starting point.

- **Dynami-CAL GraphNet** ([[dynami-cal-graphnet]]): Proves that **exact** conservation of both linear and angular momentum can be hard-wired into the GNN architecture via antisymmetric edge-local frames, holding even under dissipation and external forces. This upgrades the GNN-PB proposal: rather than a soft bottleneck, we can embed **hard conservation** using the Dynami-CAL technique for the momentum/angular-momentum constraints, and use a soft bottleneck only for energy (which is not conserved in dissipative systems).

**Revised architecture:**

$$\vec{F}_{ij} = \sum_k f_k(m_{ij})\, \hat{e}_k^{ij}, \quad \hat{e}_k^{ij} = -\hat{e}_k^{ji}, \quad m_{ij} = m_{ji}$$

with an additional energy bottleneck at the node level:
$$z_{\text{bn},i} = \Pi_E\, h_i^{\ell}, \quad h_i^{\ell+1} = \text{MLP}(z_{\text{bn},i})$$

**Why this fills a gap:**
- **Irregular geometry** — the single largest unresolved gap in the current paradigms. FNO and spectral methods assume periodic grids; transformer patch methods are approximate on curved boundaries. A GNN on a learned mesh is the most principled approach.
- **Hard + soft conservation:** Exact momentum conservation (Dynami-CAL) + soft energy bottleneck — combining levels 3 and 4 of the physics encoding spectrum
- Connects directly to the message passing literature: [[message-passing-cyclicity]], [[message-passing-belief-propagation]]
- Learned mesh refinement allows the model to allocate resolution adaptively (shocks, boundary layers, stagnation points)
- Ghost-node boundary treatment (Dynami-CAL) unifies body-body and body-wall interactions in a single message-passing framework

**Key risk:** GNNs have historically failed to scale past ~100M parameters due to over-smoothing and optimization difficulties. The learned mesh adds a second optimization problem on top of the physics one. GNS performance is dominated by the number of message-passing steps $M$ — scaling $M$ linearly increases inference cost.

**[AI Inference]:** The physics bottleneck idea is a structural analog to the MoE routing in [[deep-memory-dissipative]] — both force information to flow through a low-dimensional structured subspace before expanding. The physics bottleneck just replaces learned routing gates with physically-motivated projection axes. This connection suggests the bottleneck layer could be implemented as a physics-aware MoE where each expert corresponds to a conserved quantity.

**[AI Inference]:** EquiformerV3 ([[equiformer-v3]]) demonstrates that SE(3)-equivariant GNNs with SwiGLU-S² activations achieve SOTA on atomistic benchmarks at 5–500M params. Applying EquiformerV3-style irreps features (high-$L_{\max}$ equivariant representations) to the GNN-PB processor could give both conservation and high-order angular expressivity — at the cost of $O(L_{\max}^4)$ tensor product complexity per edge.

**Benchmark hypothesis:** GNN-PB should outperform transformer baselines on irregular-geometry Burgers' (non-uniform grid), while underperforming on the standard uniform-grid benchmark due to overhead from the graph construction.

---

## Benchmark: 1D Burgers' Equation

The minimal benchmark for comparing all five architectures — the direct analog of training a small transformer on Shakespeare to test scaling behavior before committing resources.

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

**Why Burgers':**
- 1D, single field $u(x,t)$, single parameter $\nu$ — minimal setup cost
- Develops shocks and nonlinear structure — non-trivial, not solvable by linear methods
- Universal benchmark — every physics ML paper reports results here, enabling direct comparison with literature
- 10,000+ trajectories generatable in minutes on CPU; tiny models (1M–10M params) train in minutes on a single GPU

**Setup:**

| Parameter | Value |
|---|---|
| Domain | $x \in [0,1]$, periodic BC |
| Time horizon | $t \in [0, 2]$ |
| Resolution | 64 or 128 grid points, $\Delta t = 0.01$ |
| Training data | Random sinusoidal ICs; solved with RK45 (`scipy.integrate.solve_ivp`) |
| Task | Given first 10 timesteps, autoregressively predict the remaining 190 (to $t=2$; an earlier draft said "next 90," which is inconsistent with the $t\in[0,2]$ horizon at $\Delta t=0.01$ and the $t=2$ pass bar below — fixed 2026-07-01, build-layer audit) |
| Metric | Relative $L^2$ error at $t = T$: $\frac{\|u_\theta - u_{\text{true}}\|_2}{\|u_{\text{true}}\|_2}$ |
| Scale sweep | 100K → 1M → 10M parameters per architecture |

**Secondary benchmark (for GNN-PB):** Same Burgers' equation on a non-uniform grid (random point spacing with 5× refinement near $x = 0.5$) — this is the regime where GNN-PB has a structural advantage and the transformer/SSM baselines are weakest.

**What "passing" looks like:** Relative $L^2 < 0.05$ at $t = 2$ with a 1M-parameter model, training in under 30 minutes on a single GPU. This matches published results for small FNO and AR transformer baselines on this task.

---

## Summary Table

| Architecture | Paradigm | Key Advantage | Key Risk | New? |
|---|---|---|---|---|
| AR Transformer | Autoregressive | Battle-tested; in-context learning | Error accumulation | No |
| Diffusion Backbone | Generative | Best for chaotic systems; UQ | Inference cost | No |
| Neural Differentiator | Derivative prediction + integrator | 7× SOTA; stable rollout | Integrator coupling | No |
| Physics-Mamba (SSM) | State space recurrence | Linear-time inference; $\Delta t$-aware | Spatial receptive field | **Yes** |
| GNN-PB | Graph + physics bottleneck | Irregular geometry; exact+soft conservation | GNN scaling limits | **Yes (now better-supported)** |

---

## See Also

- [[pfm-architecture-approaches]] — full 7-paradigm comparison
- [[physics-foundation-models]] — capability requirements driving these choices
- [[autoregressive-rollout-stability]] — error accumulation problem in AR baselines
- [[diffusion-models-physics]] — evidence base for diffusion approach
- [[neural-operators]] — discretization-invariant operator learning
- [[message-passing-belief-propagation]] — foundation for GNN-PB
- [[mixture-of-experts]] — connection to physics bottleneck design
- [[equivariant-gnns]] — equivariant GNNs and conservation law constraints
- [[gns-graph-network-simulators]] — empirical backbone for GNN-PB
- [[dynami-cal-graphnet]] — exact conservation constraints for GNN-PB
- [[equiformer-v3]] — SE(3)-equivariant GNN design patterns
- [[gphyt-physics-foundation-model]], [[walrus-paper]], [[pisd-physics-informed-spectral-diffusion]]
