# Iterative Refinement Architecture for Physics Simulation

**Type:** Architecture Proposal / Concept Note  
**Date:** 2026-06-26  
**Related Concepts:** [[pfm-architecture-approaches]], [[diffusion-models-physics]], [[autoregressive-rollout-stability]], [[physics-foundation-models]], [[world-models-physics-ai]], [[hamiltonian-message-passing]], [[action-based-noether-enforcement]]  
**Related Summaries:** [[pisd-physics-informed-spectral-diffusion]], [[noether-networks]], [[kepler-newton-inductive-biases]]

---

## The Core Idea

Instead of predicting the next physical state in a single forward pass (standard autoregressive) or sampling it from a fully diffused distribution (standard diffusion), a **progressive refinement architecture** generates the next state through a cascade of increasingly constrained predictions:

1. **Coarse pass (fuzzy prediction):** Generate a rough estimate of the next state, capturing coarse structure — where the vortex is, where the shock front has moved — without enforcing any conservation laws.
2. **Refinement step 1:** Refine the coarse estimate to enforce the first conservation law (e.g., mass conservation / divergence-free condition).
3. **Refinement step 2:** Further refine to enforce the second conservation law (e.g., linear momentum).
4. **Refinement step 3:** Further refine to enforce energy conservation.
5. **Refinement step $k$:** Enforce higher Casimir invariants (helicity, enstrophy), boundary conditions, or higher-order PDE constraints.

The **key insight:** conservation laws are hierarchical in how tightly they constrain the solution space. Enforcing them one at a time reduces the search space progressively rather than requiring all constraints to be satisfied simultaneously in one shot.

---

## Mathematical Framework

Let $x_t$ denote the physical state at time $t$ and $\mathcal{C}_k$ denote the $k$-th physics constraint set (e.g., $\mathcal{C}_1 = \{u : \nabla \cdot u = 0\}$ for incompressibility). The prediction cascade is:

$$x_{t+1}^{(0)} = f_\theta^{(0)}(x_t) \qquad \text{(coarse prediction, unconstrained)}$$

$$x_{t+1}^{(k)} = \Pi_{\mathcal{C}_k}\!\left(x_{t+1}^{(k-1)}\right) + f_\theta^{(k)}\!\left(x_{t+1}^{(k-1)},\ x_t\right), \quad k = 1, 2, \ldots, K$$

where $\Pi_{\mathcal{C}_k}$ is a projection (or soft push) toward the constraint manifold $\mathcal{C}_k$, and $f_\theta^{(k)}$ is a learned correction network that refines the prediction given both the current estimate and the original context.

The final prediction is $\hat{x}_{t+1} = x_{t+1}^{(K)}$.

**Physics encoding spectrum placement:** Each refinement step corresponds to progressively stronger physics encoding:
- $k=0$: Level 1 (data-driven only)
- $k=1$: Level 2–3 (soft or test-time constraint)
- $k=K$: Level 4–5 (architectural constraint or hard projection)

---

## Relationship to Diffusion Models

This architecture is **structurally analogous** to a reverse diffusion process, but with crucial differences:

| | Standard Diffusion (DDPM) | Progressive Physics Refinement |
|---|---|---|
| Initial state | Pure noise $x^{(T)} \sim \mathcal{N}(0, I)$ | Coarse physics prediction $x^{(0)} = f_\theta^{(0)}(x_t)$ |
| Each step | Remove noise guided by score function | Enforce one additional physics constraint |
| Step count $K$ | 10–1000 (tunable; quality ↑ with more steps) | $\leq$ number of independent conservation laws |
| What each step does | Moves toward data manifold | Moves toward physics-feasible manifold |
| Termination | Reached data distribution | All conservation laws satisfied |

The progressive refinement idea replaces the **noise schedule** with a **physics constraint schedule**: instead of denoising from $\sigma_T$ to $\sigma_0$, the model enforces constraints from $\mathcal{C}_0$ (none) to $\mathcal{C}_K$ (all conservation laws).

This is equivalent to **diffusion posterior sampling (DPS)** as implemented in PISD, but with the guidance operators ordered by physical importance and applied progressively rather than all at once.

---

## Autoregressive Generation of Time Sequences

The user's intuition is correct: this architecture is **not purely autoregressive at the refinement level**, but **is autoregressive across timesteps**:

- **Within one timestep $t \to t+1$:** Iterative refinement (diffusion-like, $K$ steps).
- **Across timesteps:** The fully-refined $\hat{x}_{t+1}$ is used as input to generate $\hat{x}_{t+2}$, and so on.

$$\hat{x}_{t+1} \xrightarrow{\text{refine}} \hat{x}_{t+1}^{(K)} \xrightarrow{f_\theta^{(0)}} \hat{x}_{t+2}^{(0)} \xrightarrow{\text{refine}} \hat{x}_{t+2}^{(K)} \xrightarrow{\cdots}$$

This is a **compound architecture**: iterative within each step, autoregressive across steps. Compared to pure autoregressive (single pass per step) and pure diffusion (all steps from noise):
- **Cheaper per timestep than diffusion** (only $K \ll 1000$ refinement steps, all physics-guided)
- **More physically faithful than single-pass AR** (conservation laws enforced at each step)
- **Still accumulates autoregressive error across timesteps** (same fundamental issue as all AR approaches)

---

## Connection to Existing Literature

This idea is a synthesis / generalization of several approaches already in the wiki:

1. **PISD (DPS guidance)** [[pisd-physics-informed-spectral-diffusion]]: At inference time, steers the diffusion process using PDE residual gradients. Progressive refinement generalizes this to a per-constraint schedule rather than a single PDE residual.

2. **Noether Networks** [[noether-networks]]: Test-time tailoring to enforce conservation laws. In progressive refinement, the tailoring is made structural — built into the refinement steps rather than a separate meta-learning loop.

3. **Multigrid solvers (classical numerics)**: Directly analogous. In multigrid: solve on coarse grid first (low-frequency components), then refine on progressively finer grids. Here: predict coarse physics first, then enforce increasingly fine constraints. Both exploit the same hierarchical structure.

4. **ADMM (Alternating Direction Method of Multipliers)**: Classical optimization method that solves $\min_x f(x)$ s.t. $\mathcal{C}_1, \ldots, \mathcal{C}_K$ by alternating projections. Progressive refinement can be seen as a learned ADMM, where each step is a learned projection + correction.

5. **Coarse-to-fine prediction in computer vision**: Super-resolution, image pyramids, hierarchical image generation (eDiff-I, MDT) all use this principle. The physics version simply replaces "scale hierarchy" with "conservation law hierarchy."

---

## Does Prediction Get Easier with More Context Over Time?

**Yes and no — two competing effects:**

### Effect 1: Richer Context → Easier Dynamics Identification (Helps)

With $n$ past states, the model can estimate:
- **First-order time derivative:** $\partial_t u \approx (x_t - x_{t-1})/\Delta t$ (needs 2 snapshots)
- **Second-order:** $\partial_{tt} u \approx (x_{t+1} - 2x_t + x_{t-1})/\Delta t^2$ (needs 3 snapshots)
- **Frequency content:** With $n \gg 1$ states, the model can identify periodic/quasi-periodic behavior
- **Governing law identification:** With sufficient context, the model can identify Reynolds number, Mach number, characteristic length scales implicitly

From the Kepler/Newton paper [[kepler-newton-inductive-biases]]: more context enables "Keplerian" (global pattern fitting) mode, which achieves higher prediction accuracy. In the limit of long context over a periodic orbit, the model essentially has the full trajectory and can perfectly extrapolate.

### Effect 2: Error Accumulation → Context Quality Degrades (Hurts)

In **autoregressive rollout**, the "past states" at step $T$ are not ground truth — they are predicted states with accumulated error $\|e_t\|$ growing over $t$. So:

$$x_{\text{context}}^{(T)} = \hat{x}_T \approx x_T + \epsilon_T, \quad \|\epsilon_T\| \approx \lambda^T \|\epsilon_1\|$$

for some error growth factor $\lambda > 1$ (in unstable regimes). Long context over predicted states is increasingly corrupted context.

**Net result:** Prediction is easier with more context *when the context is ground truth* (teacher forcing mode, as in training). During inference rollout, longer context helps identify governing dynamics but the quality of that context degrades due to error accumulation. The two effects compete.

**The progressive refinement architecture partially resolves this tension:** by enforcing conservation laws at each step, the error growth is bounded by physical constraints. An energy-conserving refinement step cannot amplify kinetic energy beyond what momentum allows, reducing the instability of the Lyapunov exponent $\lambda$.

---

## Is a Single Snapshot Sufficient?

**For first-order systems (parabolic PDEs):** Yes, in principle. The heat equation $\partial_t u = \kappa \nabla^2 u$ is fully determined by $u(x, t)$ alone — one snapshot contains all information needed to compute $\partial_t u$, hence $u(x, t + \Delta t)$.

**For second-order systems (hyperbolic PDEs, Navier-Stokes):** No. The wave equation $\partial_{tt} u = c^2 \nabla^2 u$ requires both $u$ and $\partial_t u$ (velocity) to determine the future. From a single snapshot $u(x, t_0)$, the velocity $\partial_t u$ is not directly observable. Two snapshots $u(x, t_0), u(x, t_0 - \Delta t)$ allow estimation of velocity via finite difference.

This is directly the **ODE order / context length relationship** from [[kepler-newton-inductive-biases]]:
$$\text{minimum context length for world model} = \text{order of governing ODE}$$

For Navier-Stokes (second order in time for the velocity field): minimum context = 2 snapshots.

**What single-snapshot AR models are actually doing:** They have implicitly learned the stationary *velocity* from training data distributions. A single-snapshot model trained on Navier-Stokes has internalized the typical relationship between field configuration and its rate of change across the training dataset. This is **Keplerian mode** (curve-fitting from the distribution), not **Newtonian mode** (causal mechanistic reasoning). The model appears to work from one snapshot but is actually exploiting statistical regularities rather than mechanistic physics.

**Practical implication for PFM:** A physically principled PFM should receive at minimum context length = order of the governing PDE. For second-order physical systems:
- Minimum: 2 snapshots
- Recommended: 4–8 snapshots (to estimate both velocity and acceleration for stability)
- Maximum: limited by attention cost and autoregressive error accumulation in early context

GP$_{\text{hy}}$T's derivative prediction approach is a neat partial solution: the model learns to estimate $\partial_t u$ and $\partial_{tt} u$ from a **single snapshot** by learning these derivative statistics from training data. This achieves second-order accuracy without requiring 2 context snapshots, at the cost of approximating the derivative.

---

## The Progressive Refinement Architecture: Summary Assessment

**Strengths:**
- Naturally decouples "approximate dynamics" from "physical feasibility" — the coarse pass handles the former, refinement handles the latter.
- Conservation law enforcement at each refinement step bounds error growth, partially solving the long-rollout instability problem.
- Differentiable and trainable end-to-end (each refinement step can be a small neural network or a learned projection).
- Generalizable: the number of refinement steps $K$ and which constraints to enforce is a design choice, allowing a spectrum from fast/approximate to slow/exact.
- Compatible with autoregressive rollout across time: refine within each step, then roll forward.

**Weaknesses:**
- Each refinement step adds inference cost. $K$ steps per timestep vs. 1 for standard AR.
- Requires specifying the constraint ordering — which conservation laws to enforce first. This ordering may be problem-dependent.
- Projections onto constraint manifolds $\mathcal{C}_k$ may themselves be expensive (e.g., enforcing divergence-free requires solving a Poisson equation).
- Does not eliminate autoregressive error accumulation across timesteps — only bounds it via physics constraints within each step.

**[AI Inference]:** The most natural implementation would combine:
- A small **diffusion backbone** (5–20 refinement steps rather than 1000) for the within-step refinement
- **DPS-style guidance** (PISD) at each step using a conserved quantity as the guidance signal
- **Noether Network** tailoring to discover and enforce approximately-conserved quantities specific to the physical system in context

This gives a model that is cheaper than standard diffusion (few steps), more principled than standard AR (physics constraints at each step), and generalizable across physical systems via test-time constraint specification.

---

## See Also

- [[diffusion-models-physics]] — standard diffusion approach; this is a physics-structured specialization
- [[autoregressive-rollout-stability]] — error accumulation problem this architecture partially addresses
- [[pfm-architecture-approaches]] — full landscape of architecture approaches
- [[world-models-physics-ai]] — context-length / ODE-order relationship; Keplerian vs. Newtonian modes
- [[action-based-noether-enforcement]] — source of conservation laws for the refinement schedule
- [[hamiltonian-message-passing]] — exact conservation enforcement; relevant to hard refinement steps
- [[pisd-physics-informed-spectral-diffusion]] — DPS physics guidance at inference time
- [[noether-networks]] — test-time tailoring; connects to refinement step design
- [[kepler-newton-inductive-biases]] — context length and ODE order; single-snapshot analysis
