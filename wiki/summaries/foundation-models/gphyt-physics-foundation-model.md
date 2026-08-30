# Summary: Towards a Physics Foundation Model (GP$_{\text{hy}}$T)

**Source:** `raw/Towards a Physics Foundation Model.md`
**Authors:** Florian Wiesner, Matthias Wessling, Stephen Baek (University of Virginia / RWTH Aachen)
**arXiv:** 2509.13805v3
**Date Ingested:** 2026-04-11 · **Deepened:** 2026-06-29

---

## Overview

Introduces the **General Physics Transformer (GP$_{\text{hy}}$T)**, a transformer trained on **1.8 TB** of diverse simulation data that demonstrates "train once, deploy anywhere" foundation-model behavior for physics. The central thesis is sharp and contrarian: rather than baking physics into the architecture (PINNs, neural operators, equivariant nets), GP$_{\text{hy}}$T shows that a **general-purpose video transformer plus *one* lightweight physics modification** (predict the time-derivative, integrate numerically) can outperform specialized multi-physics models *and* exhibit emergent in-context learning of governing dynamics. Three headline claims:

1. **7× lower NMSE** than the next-best baseline across heterogeneous multi-physics tasks.
2. **Zero-shot generalization via in-context learning** — it infers unseen dynamics/boundary conditions from the input context, with no equation specification and no finetuning.
3. **Stable long-horizon autoregressive rollouts** (competitive with [[poseidon-pde-foundation-model]] up to 24 steps).

It sits at the **data-driven / hybrid** pole of the [[pfm-architecture-approaches]] spectrum, and is the clearest demonstration in the wiki that *in-context physics learning* — the analog of LLM few-shot prompting — is real.

---

## The Hybrid Neural-Differentiator + Integrator Architecture

GP$_{\text{hy}}$T's defining idea is to **not predict the next state directly**. Instead it predicts a *time-derivative field* and hands it to a classical integrator:

$$X_{t_{i+1}} = f\!\left(X_{t_i},\,\frac{\partial X}{\partial t}\bigg|_{t_i},\,\Delta t\right).$$

- **Neural differentiator:** a spatiotemporal transformer ingesting $n$ past snapshots and outputting $\partial X/\partial t$. Input is tokenized via **4D spatiotemporal tubelet patches** (a $p\times p$ spatial patch over $\tau$ frames; see [[spatiotemporal-tubelet-tokens]]) — the same video-ViT tokenization later used by [[walrus-paper]].
- **Numerical integrator:** first-order **Forward Euler** by default, $X_{t+\Delta t}=X_t+\Delta t\cdot\widehat{\partial_t X}$, explicitly extensible to Runge–Kutta (the design that [[arch-neural-differentiator]] formalizes with RK4 and an ODE error-bound analysis).

### Why predict the derivative instead of the next state?

This is the conceptual core, and it has three consequences:

1. **Δt-agnosticism / temporal-scale generalization.** Predicting $\partial_t X$ decouples the learned object from the sampling interval. The same network serves any $\Delta t$ because the integrator, not the network, consumes $\Delta t$. Directly predicting $X_{t+1}$ ties the model to the training stride. This is GP$_{\text{hy}}$T's answer to the timescale problem that [[poseidon-pde-foundation-model]] solves differently (continuous lead-time LayerNorm) — see [[pfm-interface-design]].
2. **Residual/tendency learning is easier.** $\partial_t X$ is typically small and structured relative to $X$ itself; learning the *tendency* is a better-conditioned regression than reproducing the full state (the same reason residual prediction stabilizes [[arch-autoregressive-transformer]]).
3. **Built-in physical prior.** The form $X_{t+1}=X_t+\Delta t\,\partial_t X$ *is* the structure of every evolution PDE $\partial_t u=\mathcal L(u)$. The transformer is, in effect, learning the spatial operator $\mathcal{L}$ — i.e. doing **operator learning through a derivative head**, a soft architectural bias (level 1–2 on the encoding spectrum) without committing to any specific equation.

### Explicit derivative features (the second lightweight modification)

Before tokenization, GP$_{\text{hy}}$T **concatenates finite-difference derivative channels** to the raw fields: spatial central differences $\partial_x X,\partial_y X$ and temporal $\partial_t X$. These hand the model pre-computed gradient information so it does not have to relearn differentiation from patches. Effect: **~1 order of magnitude** improvement in resolving sharp gradients (shockwaves, phase interfaces) over long rollouts. This is the seed of the derivative-augmented input channels $[u,\partial_x u,\partial_{xx}u,\partial_t u]$ in [[arch-neural-differentiator]].

---

## Training Data (7 datasets, 1.8 TB)

| Dataset | Trajectories | Origin | Physics |
|---|---|---|---|
| Shear flow (incompressible NS) | 1120 | The Well | shear-layer turbulence |
| Rayleigh–Bénard convection | 1750 | The Well | buoyant convection |
| Euler (compressible) | 5000 | The Well | shocks, compressible gas |
| Obstacle flow | 1266 | Own | flow past bluff body, wakes |
| Thermal flow | 354 | Own | coupled heat + flow |
| Rayleigh–Bénard 2 | 228 | Own | convection (2nd regime) |
| Two-phase flow | 816 | Own | interface dynamics |

Two augmentations are doing quiet but essential work:
- **Variable $\Delta t$ sub-sampling** — the model never sees a fixed timestep, *forcing it to infer the temporal scale from the context window itself*. This is the precondition for in-context temporal generalization.
- **Per-dataset normalization** — by removing absolute spatial scale, the model must *infer spatial scale from context* too.

Together these are why GP$_{\text{hy}}$T can do in-context learning at all: training deliberately withholds the absolute scales, so the only way to predict well is to read them off the prompt. **[AI Inference]:** this is the physics analog of how LLMs become few-shot learners by training on heterogeneous text without task labels — the heterogeneity *is* the curriculum that produces in-context ability.

---

## In-Context Learning for Physics (the emergent capability)

The most consequential finding. Given only a context window of snapshots — no equations, no PDE coefficients, no boundary-condition flags — GP$_{\text{hy}}$T:
- **Infers new boundary conditions from the prompt alone** (e.g. adapts to an obstacle/wall configuration it must read from the data).
- **Stays below NMSE = 1 on entirely novel physics** (supersonic flow, turbulent radiative layer) — *the only model tested to do so*, i.e. it degrades gracefully rather than diverging on regimes outside its training distribution.

Mechanistically this parallels [[poseidon-pde-foundation-model]]'s compositional-primitive reuse: both learn reusable physical operators during pretraining and recombine them for unseen tasks; GP$_{\text{hy}}$T does it *at inference time from context* (no weight updates), Poseidon does it *via few-sample finetuning*. See [[in-context-learning-physics]] and [[world-models-physics-ai]] (the context-length-controls-world-model-type story is directly relevant: a derivative-predicting model with a short context is operating near the "Newtonian/local" regime).

---

## Results in Detail

- **Multi-physics accuracy:** lowest NMSE on **6 of 7** datasets; overall **7× better than DPOT** (next best).
- **Long-horizon stability:** best overall rollout stability to **24 prediction steps**; competitive with Poseidon — notable because Poseidon uses a fundamentally different (operator-learning) training scheme.
- **In-context generalization:** new BCs inferred from prompt; novel physics (supersonic, radiative) kept below NMSE 1.
- **Model scaling:** monotonic accuracy improvement **9M → 385M** parameters — clean scaling behavior, suggesting headroom.
- **Context length:** **4 input snapshots** is the sweet spot; gains are log-linear with diminishing returns beyond 2. (Consistent with [[kepler-newton-inductive-biases]]: minimum useful context ≈ ODE order; more context helps prediction but with diminishing physical-information return.)

---

## Key Insights

1. **Predict $\partial_t X$, not $X_{t+1}$** → cross-temporal-scale generalization and better-conditioned learning.
2. **Explicit derivative features** are critical for sharp gradients (shocks, interfaces) — cheap, high-impact.
3. **General video transformer + minimal physics tweaks > specialized multi-physics architecture** — a strong vote for the "scale + generality" school over the "hard inductive bias" school.
4. **In-context physics learning is achievable** — the defining foundation-model property, demonstrated for PDEs.

---

## Limitations

- **2D only** (architecture directly extensible to 3D, but unproven there).
- **Fluids + heat transfer only** — no solid mechanics, optics, molecular dynamics, EM.
- **Fixed 256×128 resolution** (no resolution invariance — a limitation the neural-operator framing of [[poseidon-pde-foundation-model]]/[[neural-operators]] and the compression-token approach of [[learned-query-compression-tokens]] are designed to address).
- **Accuracy still far below numerical solvers** for engineering-grade long-term prediction.
- First-order Euler integration limits accuracy; RK4 is an obvious upgrade ([[arch-neural-differentiator]]).

---

## Relevance to the Physics Foundation Model Goal

GP$_{\text{hy}}$T and [[poseidon-pde-foundation-model]] are the two leading feasibility proofs in the wiki, representing **two complementary PFM design poles**:

| | GP$_{\text{hy}}$T | Poseidon |
|---|---|---|
| Core target | next-state via derivative head | full solution operator $\mathcal S(t,a)$ |
| Timescale handling | predict $\partial_t X$, integrator takes $\Delta t$ | continuous lead-time LayerNorm |
| Generalization mode | **in-context** (no weight update) | **few-sample finetuning** |
| Backbone | flat spatiotemporal ViT | multiscale SwinV2 U-Net |
| Tokenization | tubelet patches + derivative channels | hierarchical windowed patches |

The neural-differentiator + integrator hybrid is one of the strongest candidate PFM backbones and is the basis for benchmark architecture [[arch-neural-differentiator]].

**[AI Inference]:** The derivative-feature approach could be extended to **higher-order curvature** ($\partial^2 X/\partial x^2$, mixed derivatives), which may help turbulence closure (subgrid stresses depend on velocity-gradient invariants). Feeding the strain-rate and vorticity tensors as channels would inject the exact quantities closure models use.

**[AI Inference]:** The clean 9M→385M scaling plus stable rollouts suggests a **1B+ version trained jointly on GP$_{\text{hy}}$T-style derivative prediction *and* Poseidon-style operator learning** could combine in-context adaptation with full-trajectory generation — i.e. read the regime from context (GP$_{\text{hy}}$T) *and* emit any-time solutions (Poseidon). The two papers' tricks are non-conflicting and could be merged in one model.

**[AI Inference]:** Because the model already ingests explicit finite-difference channels, it is a natural host for **physics-residual auxiliary losses**: the predicted $\partial_t X$ and the supplied spatial-derivative channels can be combined into a PDE residual $\partial_t X - \mathcal L(X,\partial_x X,\ldots)$ whenever an equation *is* known, moving it up the encoding spectrum (level 1 → 2) without architectural change.

---

## Links

- [[physics-foundation-models]] — the broader goal
- [[poseidon-pde-foundation-model]] — the complementary operator-learning PFM (compare/contrast)
- [[in-context-learning-physics]] — the key emergent capability demonstrated here
- [[arch-neural-differentiator]] — benchmark architecture formalizing GP$_{\text{hy}}$T's hybrid
- [[transformer-architectures]] — spatiotemporal transformer backbone
- [[spatiotemporal-tubelet-tokens]] — its tubelet token representation
- [[autoregressive-rollout-stability]] — long-horizon stability challenge
- [[world-models-physics-ai]] / [[kepler-newton-inductive-biases]] — context length and world-model type
- [[partial-differential-equations]] / [[navier-stokes-equations]] — governing equations modeled
- [[walrus-paper]] — companion large-scale model (1.3B params)
- [[pde-transformer-paper]] — competing architecture (ICML 2025)
