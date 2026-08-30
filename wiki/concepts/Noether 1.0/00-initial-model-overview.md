# Initial PFM Model — Overview & Design Commitments

**Type:** Concept hub (folder: Noether 1.0)
**Status:** The theoretical home for the *first concrete model*, now named **[[noether-1.0]]**. Everything in this folder specifies one coherent architecture, not the full design space (that lives in [[pfm-architecture-approaches]] and the token/attention folders).
**Related Concepts:** [[graph-tokenizer]], [[normalization-scheme]], [[initial-model-architecture]], [[symmetric-attention-physics]], [[00-attention-overview]], [[pfm-interface-design]], [[pfm-architecture-approaches]], [[pipeline-contract]], [[training-curriculum]], [[noether-1.0]]
**Related Summaries:** [[gphyt-physics-foundation-model]], [[walrus-paper]], [[poseidon-pde-foundation-model]], [[dynami-cal-graphnet]]

---

## What this folder is

The rest of the wiki maps the *space* of physics-foundation-model designs. This folder commits to a *single point* in that space — the first model to build — and works out its tokenizer, attention, normalization, conservation mechanism, and end-to-end workflow in implementation-ready detail. It is the bridge from the research/design layer (this vault) to the build layer (the [GitHub repo](https://github.com/nonidino/physics-foundation-model)). That model is named **Noether 1.0** — see [[noether-1.0]] for the concrete sizing, hyperparameters, and eval protocol.

---

## The design commitments (from the founding brainstorm)

These are fixed decisions; everything downstream follows from them.

### 1. Breadth from the ground up — multiphysics, not single-domain
Accuracy is hypothesized to come **with scale**, as in LLMs. The model is therefore designed for **multiphysics from the start** (fluids, heat, EM, elasticity, many-body), not specialized to one equation family. An initial smoke-test may run on one problem (e.g. 2D incompressible Navier–Stokes or 1D Burgers, per [[possible-architectures]]), but the architecture and training target broad coverage. This sides with the Walrus/GP$_{\text{hy}}$T "diversity-first" lesson and Poseidon's evidence that diverse pretraining drives cross-family generalization ([[walrus-paper]], [[poseidon-pde-foundation-model]]).

### 2. Direct state prediction — not a neural differentiator
The model predicts the **next physical state directly**, $\hat x_{t+1} = f_\theta(\text{context})$ — *not* a derivative handed to a classical integrator (the GP$_{\text{hy}}$T route). Rationale: a classical integrator couples the model permanently to numerical machinery that may not transfer to systems where it does not apply cleanly (quantum, relativistic, particle-level). Direct state prediction is more general in principle, even if noisier; the noise is addressed by the conservation mechanism and optional refinement, not by outsourcing the dynamics. (Internally the model predicts the **residual** $\Delta = \hat x_{t+1}-x_t$ for stability — a representational choice, still direct-state.)

### 3. Output refinement — diffusion as generative core for chaos, PINN when the equation is known
**Decision (adopting [[open-architectural-problems]] S5.1):** refinement is *not* a uniform bolt-on. The design splits scales LES-style — the transformer regresses the resolved (large, energy-containing) scales, and a **physics-guided latent diffusion** model *generates* the chaotic sub-grid detail on the high-frequency residual channel (the S3.1 dual path, commitment-linked below). Two modes:
- **Diffusion (generative core for chaos, *equation-agnostic*)** — generates/denoises the sub-grid residual, **constraint-projected** so the detail stays physical ([[arch-diffusion-backbone]], [[pisd-physics-informed-spectral-diffusion]]); yields ensemble uncertainty. For turbulent/chaotic regimes this is **load-bearing** (a deterministic map provably blurs to the conditional mean); for smooth/laminar regimes it reduces to optional cleanup.
- **PINN residual correction** (optional, *when the governing equation is known*) — a few gradient steps on the PDE residual ([[physicsformer-pinn-ns]]).
The backbone must still stand alone on smooth physics; diffusion becomes the core specifically where determinism fails.

### 4. Headline goals — the continuum–particle bridge and long-horizon stability
The two capabilities the model is *built around*:
- **Continuum–particle bridge** (= multiphysics): one representation for grid/mesh fields **and** N-body/particle systems. The *homogeneous* case is solved at the tokenizer ([[graph-tokenizer]]) — both regimes become graph nodes carrying neighborhood latents. The *coupled* case (fluid + suspended particles, plasma) is addressed by a **typed-edge encoder** with per-node-type embeddings and a shared momentum-consistent message protocol (adopting [[open-architectural-problems]] S4.1).
- **Long-horizon stability via an inbuilt mechanism** — conservation is *architectural*: symmetric/conservative attention + structure-preserving edges, **plus a mandatory projection onto conserved totals** (adopting [[open-architectural-problems]] S1.1) that converts depth-conservation into trajectory (time) conservation. The projection is no longer an ablation — it is on the critical path.

### 5. Symmetric attention — the one concrete mechanism choice
The interaction mechanism is **symmetric attention** ([[symmetric-attention-physics]]): reciprocal scores ($A_{ij}=A_{ji}$, Newton's 3rd law), signed for repulsion, paired with a **skew-symmetric conservative channel** for energy/information preservation across depth. This is the inbuilt long-horizon-stability mechanism. Large contexts are handled by layering [[index-share-sparse-attention]] and [[hierarchical-query-attention]] on top without breaking symmetry.

---

## How the commitments resolve into components

| Commitment | Component | Page |
|---|---|---|
| Multiphysics / continuum–particle bridge | Unified graph tokenizer (nodes = particles or patch centroids) + **typed-edge encoder** for coupled systems (S4.1) | [[graph-tokenizer]] |
| Resolution-free, continuous, local+global; sharp-feature-preserving | Query-compression node encoder + **dual-path high-frequency residual** (S3.1) + supernode hierarchy | [[graph-tokenizer]] |
| Direct state prediction | Residual next-state head + coordinate-implicit decoder | [[initial-model-architecture]] |
| Inbuilt conservation & stability | Symmetric + skew-symmetric attention; antisymmetric edges; **mandatory projection onto conserved totals** (S1.1) | [[symmetric-attention-physics]] |
| Preserve scale/structure | Nondimensionalization + structure-preserving norm (no LayerNorm) | [[normalization-scheme]] |
| Large contexts | Index Share + hierarchical queries | [[00-attention-overview]] |
| Refinement / generative core | **Physics-guided latent diffusion on the sub-grid residual** (S5.1); PINN residual head | [[initial-model-architecture]] |
| Non-text task/regime spec | 5-input physical interface | [[pfm-interface-design]] |

---

## What is deliberately left open

Per the founding stance ("hard to pick without evidence"), some choices are intentionally *not* frozen and are flagged as ablations in [[initial-model-architecture]]:

- **Conservation enforcement strength** — **RESOLVED (2026-06-30):** the hard projection head (S1.1) is now *mandatory*, not an ablation — it converts depth-conservation into time-conservation ([[open-architectural-problems]]). Multi-step rollout training is retained as a *complement*, not a substitute. The symmetric/skew-symmetric mechanism still supplies the aggregate momentum + energy/information preservation the projection then makes exact.
- **Refinement default** — **RESOLVED (2026-06-30):** diffusion is the generative core for chaotic regimes (S5.1), optional cleanup for smooth ones, PINN when the equation is known; an **in-context regime classifier routes** which path runs (S5.2, adopted).
- **Multi-token prediction horizon** $k$ — predicting several future steps at once (GLM MTP, [[glm-index-share-attention]]) as a stability aid; horizon **still TBD**.
- **Validation complements** — **ADOPTED (2026-06-30):** reconstruction fidelity floor (S3.4), coupled benchmark (S4.4), regime routing (S5.2), push-forward curriculum (S1.5); thresholds set at implementation.
- **Local-constraint enforcement (Problem 2)** — **PLANNED (2026-06-30):** joint physical-space projection folding divergence-free + boundary conditions + conservation into one solve (P2.1–P2.4), living in the S1.1 projection stage; the elliptic solve runs on the supernode hierarchy. See the **second-pass audit** in [[open-architectural-problems]].
- **Still genuinely open** — the cross-type message *form* (Problem 4), multi-term loss balancing (Problem 6), equivariance (Problem 9), training stability (Problem 10), and inference cost (Problem 11).

---

## See Also

- [[initial-model-architecture]] — the assembled end-to-end pipeline + diagram
- [[graph-tokenizer]] — the unified tokenizer
- [[normalization-scheme]] — the LayerNorm decision
- [[symmetric-attention-physics]] / [[00-attention-overview]] — the attention mechanism
- [[pfm-interface-design]] — the non-text input interface
- [[pfm-architecture-approaches]] — the broader design space this narrows from
- [[possible-architectures]] — the benchmark plan for an initial smoke test
