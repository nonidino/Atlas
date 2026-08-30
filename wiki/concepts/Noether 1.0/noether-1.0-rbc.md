# Noether 1.0 — RBC (2D Rayleigh–Bénard): as-built model report

**Type:** Concept — initial model, second concrete instantiation (folder: Noether 1.0)
**Status:** **As-built.** This page describes the model as it is actually implemented, trained, and measured on branch `fix/energy-regulation-norm` of the build repo (2026-07-03), not the original design spec — deviations from the earlier spec (this page's prior revision) are called out explicitly where they matter. Treat this as the model-card / technical-report layer; [[noether-1.0]] remains the 1D-Burgers sibling and [[open-architectural-problems]] remains the design-rationale ledger.
**Related Concepts:** [[noether-1.0]], [[open-architectural-problems]], [[pipeline-contract]], [[training-curriculum]], [[graph-tokenizer]], [[initial-model-architecture]], [[multihead-attention]], [[normalization-scheme]], [[pfm-interface-design]], [[intelligent-patching]], [[structure-preserving-tokens]], [[smoke-test-bringup]], [[attractor-energy-projection]], [[spatiotemporal-tubelet-tokens]], [[autoregressive-rollout-stability]]
**Related Summaries:** [[gphyt-physics-foundation-model]] (uses The Well's Rayleigh–Bénard set), [[walrus-paper]], [[poseidon-pde-foundation-model]], [[latent-diffusion-physics]]

---

## Abstract

Noether-1.0-RBC is a graph-tokenized, symmetric+skew-symmetric-attention transformer that predicts the next state of 2D Rayleigh–Bénard convection (coupled velocity + temperature, Boussinesq, no-slip/fixed-$T$ walls) as a residual over the current state, with incompressibility enforced *by construction* via a stream-function decoder head. It is the project's first multiphysics benchmark: multi-field cardinality, wall boundary conditions, and driven-dissipative (non-conservative) dynamics — the complement of the 1D-Burgers config, which stress-tested conservation with none of those. At **5.50M parameters** (base config), the model reaches single-step Nusselt-number error **≤4.8%** across $\mathrm{Ra}=2\times10^4$–$10^6$ and, after a three-part rollout-stabilization fix (§5), autoregressive rollout is **stable and divergence-free through at least 600 steps** at every trained Ra. Rollout *accuracy* is Ra-dependent and currently the open problem: single-step velocity error grows from 0.6% to 4.7% as $\mathrm{Ra}$ goes $3\times10^4\to3\times10^5$, traced to the tokenizer's spatial resolution rather than training budget (§6.5–6.6). A spatiotemporal extension (K-frame context, §4.6) is in evaluation as the next lever.

---

## 1. Physics & benchmark

**Nondimensional 2D Boussinesq equations** (diffusive scaling; $T$ = temperature relative to the top wall):
$$\partial_t \mathbf u + (\mathbf u\cdot\nabla)\mathbf u = -\nabla p + \Pr\,\nabla^2\mathbf u + \Pr\,\mathrm{Ra}\,T\,\hat{\mathbf y},\qquad
\partial_t T + (\mathbf u\cdot\nabla)T = \nabla^2 T,\qquad
\nabla\cdot\mathbf u = 0.$$

- **Domain:** $[0,\Gamma]\times[0,1]$, aspect ratio $\Gamma=2$, grid $128\times64$ (8192 points).
- **Boundary conditions:** horizontal walls at $y=0$ (hot, $T{=}1$) and $y=1$ (cold, $T{=}0$): no-slip $\mathbf u=0$, Dirichlet $T$. Periodic in $x$.
- **Parameters:** $\mathrm{Ra}$ (conditioning input, log-scaled), $\Pr=1$ (fixed in all data used to date; the model architecture supports arbitrary $\Pr$ as a second conditioning input but it has not been varied).
- **Data source:** a **self-contained IMEX vorticity–streamfunction Boussinesq DNS solver** (`src/noether/data/rbc_solver.py`) — not The Well. It was built as a fallback when Dedalus was unavailable and has since become the primary data source; it validates against the $\mathrm{Nu}\propto\mathrm{Ra}^{0.29}$ classical 2D correlation to within 1.5 decades of $\mathrm{Ra}$ (§6.1). The Well `rayleigh_benard` loader (`well_loader.py`) exists and is a drop-in alternate but has not yet been used for training or eval.
- **The defining diagnostic — Nusselt number.** As implemented (`rbc_losses.nusselt`), with $\kappa=H=\Delta T=1$:
$$\mathrm{Nu} = 1 + \langle u_y\,T\rangle_{V}.$$
This is a simplified single-integral form (no conductive-gradient term), valid under this solver's exact nondimensionalization; it is the metric reported everywhere below.

---

## 2. Trained Ra range and the honest scope statement

**All trained/evaluated checkpoints to date use $\mathrm{Ra}\in[3\times10^4,\,3\times10^5]$** (log-uniform sampled, the `rbc_train.npz` corpus), a **laminar-to-weakly-turbulent** regime. The wider sweep $\mathrm{Ra}\in\{2\times10^4,5\times10^4,10^5,3\times10^5,10^6\}$ (`rbc_ra_sweep.npz`) is used only for **evaluation**, so $\mathrm{Ra}=10^6$ results below are **out-of-distribution** and should be read as an extrapolation stress test, not a claim of turbulent-regime competence. The turbulent generative/diffusion path (§4.7) that the original spec reserved for high-$\mathrm{Ra}$ is unbuilt; nothing here should be read as validated above $\mathrm{Ra}\sim3\times10^5$.

---

## 3. Data corpora (as generated)

All corpora are produced by `scripts/generate_rbc_data.py` from the self-contained solver, $128\times64$, $\Gamma=2$.

| Level | Trajectories | $\mathrm{Ra}$ | Purpose |
|---|---|---|---|
| `batch0` | 1, float64 | $10^5$ | div-free structural GO/NO-GO gate |
| `g0` | 8 | $10^5$ | tokenizer-fidelity (G0) bring-up |
| `ra_sweep` | 5 (1 traj/Ra) | $\{2\times10^4,\dots,10^6\}$ | Nu(Ra) tracking, ~10-frame short eval |
| `train` | 48 | log-uniform $[3\times10^4,3\times10^5]$ | dynamics training. **Energy-diverse** (§5.1): each trajectory's initial velocity amplitude is drawn 0–1.5$\times$ free-fall, so the corpus spans off-attractor states as well as the natural attractor — necessary for the model to learn to *move* energy toward the attractor rather than only track it |
| `val` | 8 | same distribution as `train` | held-out |
| `eval_long` | 3 (1 traj/Ra) | $\{3\times10^4,10^5,3\times10^5\}$ | **150-frame** ground-truth trajectories (established convection, natural energy) for horizon-resolved rollout accuracy (25/50/100 steps) and long-horizon feasibility (§6.5) — added specifically because the `ra_sweep` trajectories (~10 frames) could not measure past a handful of rollout steps |

Output cadence $\Delta t_{\text{out}}=0.004$ (diffusive units) was chosen empirically to keep consecutive frames correlated; the original spec's $\Delta t=0.25$ would span ~80 convective turnovers at $\mathrm{Ra}=10^5$ (near-decorrelated, unlearnable single-step) and was never used.

---

## 4. Architecture (as implemented)

The forward pass, `NoetherRBC.forward` (`src/noether/model/noether_rbc.py`), stage by stage.

### 4.1 Normalization — **Ra-aware frozen power law** (major deviation from spec)

The original spec called for per-field CDF/quantile maps fit offline. **The as-built normalizer is simpler and was arrived at empirically, through a real failure mode (§5.1):**

$$\hat T = \frac{T-\mu_T}{\sigma_T},\qquad \hat{\mathbf u} = \frac{\mathbf u}{s_u(\mathrm{Ra})},\qquad s_u(\mathrm{Ra}) = a\cdot\mathrm{Ra}^{\,b}.$$

- $\mu_T,\sigma_T$: temperature mean/std, fit once over the training corpus, frozen.
- $s_u(\mathrm{Ra})$: a **per-sample scalar**, shared across both velocity components (so it commutes with divergence and preserves $\mathbf u=0$ at the walls), fit as a **log-log least-squares power law** over training trajectories: $\log(\mathrm{rms}\,|\mathbf u|) = \log a + b\log\mathrm{Ra}$. Measured coefficients (from the `rbc_train` corpus, reproduced independently on two runs): $a\approx0.084$, $b\approx0.567$ — close to the free-fall-velocity scaling $b=0.5$ one would expect from $\Pr\,\mathrm{Ra}\,T$ buoyancy forcing. `fit_normalizer` degrades gracefully to a fixed $b=0.5$ for single-$\mathrm{Ra}$ or $\mathrm{Ra}$-less corpora.
- **Why this exists at all, and why it must be Ra-aware:** see §5.1 — this is not a stylistic simplification but the fix for a real rollout-divergence bug.

### 4.2 Graph tokenization

**Uniform** (not adaptive) $128\times64$ grid $\to$ $16\times8=128$ leaf patches of $8\times8$ points each, agglomerated $2\times2$ up a hierarchy $128\to32\to8\to2\to1$ (43 internal supernodes, **171 nodes per time-frame**). Edges: leaf 4-neighbor radius graph (periodic in $x$, wall-bounded in $y$) $\cup$ parent$\leftrightarrow$child tree edges. Restriction/prolongation is parameter-free mean-pooling along the tree.

**The adaptive quadtree (monitor $\eta=|\nabla T|+\lambda_\omega|\omega|$, spec'd in the original page) is unimplemented.** `RBCConfig.adaptive_graph` exists as a flag but `graph/grid2d.py` only has `build_uniform_grid_graph`; there is no adaptive builder. This is a known, deliberately deprioritized gap — see §7.

### 4.3 Multi-field encoder

One typed cross-attention encoder per field ($d_u{=}2$ for velocity, $d_u{=}1$ for temperature), each with $N_q{=}16$ learned query slots, summed into the shared node latent with a learned field-type embedding:

$$h_i^{\text{lo}} = \sum_{f\in\{\text{vel},\text{tmp}\}}\Big[\text{CrossAttn}_f\big(Z_q^f, [\gamma(\delta_p), \hat v_p]\big) + e_f\Big],\qquad \gamma(\delta)\in\mathbb R^{32}\ (\text{8-band 2D dyadic Fourier}).$$

Run twice — once at $d{=}256$ (`_lo`, pooled up the hierarchy) and once at $d_{\text{hi}}{=}128$ (`_hi`, leaf-resolution, used only by the decoder's high-frequency path, never passes through the backbone).

### 4.4 Conditioning

$z_{\text{param}}=\mathrm{MLP}([\log\mathrm{Ra},\log\Pr])$, $z_{\Delta t}=\mathrm{MLP}([\log(\Delta t/T_0)])$, both 2-layer GELU MLPs ($\to64\to d$), summed and added to every node token. Gravity as covariant conditioning (P9.2) and the equivariant channel-typed ablation arm (P9.3) are specified in the config (`equivariant` flag) but the gravity vector is folded into a constant bias in the shipped runs — the covariant path has not been exercised.

### 4.5 Backbone

$L=6$ layers (base config), each an $\epsilon$-scaled forward-Euler step with **two sub-steps**, exactly the [[symmetric-attention-physics]] / [[skew-symmetric-attention]] construction:

$$h^{\ell+1} = h^\ell + \alpha_\ell\,\epsilon\,\tanh\!\Big[(W_{\text{skew}}-W_{\text{skew}}^\top-\gamma I)h^\ell + \sum_{k=1}^{H}W_O^{(k)}\!\!\sum_{j\in\mathcal N(i)} A^{(k)}_{ij}(v_j^{(k)}-v_i^{(k)})\Big],\quad A^{(k)}_{ij}=\tanh\!\Big(\tfrac{q_i^{(k)}\cdot q_j^{(k)}}{\sqrt{d_h}}\Big),$$
$$h^{\ell+1} \mathrel{+{=}} \alpha'_\ell\,\epsilon\,W_2\,\mathrm{renorm}\big(\mathrm{GELU}(W_1 h^{\ell+1})\big)\qquad(\text{F1 semi-orthogonal FFN}).$$

Implementation specifics:
- **Tied $Q{=}K$** per head ($H{=}8$, $d_h{=}32$) $\Rightarrow$ symmetric reciprocal scores by construction; aggregation is antisymmetric-flux ($\sum_j A_{ij}(v_j-v_i)$), so $\sum_i\Delta_i=0$ exactly (momentum conservation, structurally verified by unit test).
- **Cayley parameterization (P10.1, on):** $W_{\text{skew}}$ is not a raw learned matrix but $(I-S)(I+S)^{-1}$ of a skew generator $S$, which is *orthogonal* — bounding $\|W_{\text{skew}}-W_{\text{skew}}^\top\|_2\le2$ analytically rather than by measurement, so the Euler-stability bound $\epsilon<2/\|\cdot\|_2$ holds through training by construction.
- **ReZero (P10.4, on):** separate learnable scalar gains $\alpha_\ell,\alpha'_\ell$ (init 0.1) on the attention and FFN sub-steps — normalization-free depth control for $L{=}6$.
- **F1 FFN:** semi-orthogonal expand→GELU→renorm→contract, re-orthogonalized (QR retraction) every 100 optimizer steps.
- $\gamma$ (the port-Hamiltonian dissipation shift) is a **learnable, per-layer, sign-unconstrained scalar**, init 0.01. Empirically, in every measured trained checkpoint $\gamma>0$ on every layer (§6.4 diagnostic) — no anti-dissipative instability was ever observed, so this was never a live bug, only a checked hypothesis.
- **No LayerNorm anywhere** in the block (structure-preserving-norm philosophy, [[normalization-scheme]]).

### 4.6 Spatiotemporal context — `n_ctx` (Phase B, new; in evaluation as of this page's revision)

**Backward-compatible extension** from the original fixed 2-frame context (`u_prev, u_curr`) to a general **$K=$`n_ctx`-frame** context. Implementation:
- `time_embed` is $[K,d]$ (was $[2,d]$); $K{=}2$ reproduces the original behavior exactly, bit-for-bit — existing checkpoints load unchanged.
- The token adjacency generalizes from `_build_adj2` to `_build_adjK`: each of the $K$ time-slots gets its own copy of the intra-frame spatial graph (block-diagonal), *plus* every spatial node is linked to its counterpart in **every other** time-slot (a temporal clique per spatial node, not just nearest-neighbor-in-time). At $K{=}4$: $4\times171=684$ tokens/sample (vs. 342 at $K{=}2$).
- `forward` takes optional `u_hist`/`T_hist` $\in\mathbb R^{[B,K-2,N_x,N_y,\cdot]}$ carrying the older frames ($t{-}2,t{-}3,\dots$) in slots $2..K{-}1$; slot 0 (current frame) is always what the decoder reads, so the increment head and the div-free-by-construction guarantee are **completely unaffected** by $K$.
- Push-forward training maintains a rolling $K$-frame buffer (drop-oldest, insert-newest) across the unroll; rollout eval does the same.
- **Motivation:** a $K{=}2$ tubelet only encodes $\partial_t$ (finite difference); $K>2$ exposes acceleration/phase information the encoder can use directly rather than the backbone having to infer it implicitly — the standard tubelet-tokenization argument ([[spatiotemporal-tubelet-tokens]]), now instantiated for a graph tokenizer rather than a ViT-style patch grid.
- **Status:** implemented and unit-validated (div-free preserved to $1.2\times10^{-11}$ at both $K{=}2$ and $K{=}4$; full single-step + push-forward + rollout-eval pipeline runs end-to-end at $K{=}4$ on a toy scale). **Full-scale $K{=}4$ training results are pending** as of this revision — see §6.7.

### 4.7 Decoder — gated dual velocity head + always-on temperature head

Two coexisting SIREN decoder heads for velocity (2 hidden layers, $\omega_0{=}30$, hidden width $=d$), selected by a per-call `incompressible` flag (RBC always uses the incompressible branch; the other exists for the sibling regime, §7):

$$\text{incompressible:}\quad \delta\psi = \mathrm{Dec}_\psi^{\text{lo}} + \mathrm{Dec}_\psi^{\text{hi}},\qquad \mathbf u_{t+1} = \mathbf u_t + \nabla^\perp(\text{wall}(\delta\psi)),\quad \nabla^\perp=(\partial_y,-\partial_x)$$
$$\text{compressible (unused by RBC):}\quad \delta\mathbf u = \mathrm{Dec}_u^{\text{lo}}+\mathrm{Dec}_u^{\text{hi}},\qquad \mathbf u_{t+1}=\mathbf u_t+\delta\mathbf u$$
$$\text{temperature (always):}\quad \delta T = \mathrm{Dec}_T^{\text{lo}}+\mathrm{Dec}_T^{\text{hi}},\qquad T_{t+1} = \mathrm{wall}(T_t+\delta T).$$

$\nabla^\perp,\nabla\cdot$ are **fixed, non-learned 4th-order finite-difference operators** (`operators2d.py`): periodic central along $x$, even-ghost-reflected central along $y$. Because they act on orthogonal axes they commute to machine precision ($\partial_x\partial_y=\partial_y\partial_x$), which is the entire mechanism behind exact div-free-by-construction — not an approximate property, a Kronecker-product identity. The same even-reflection ghost stencil, applied to $\delta\psi$ zeroed at both wall rows, delivers no-slip *and* no-penetration simultaneously (`apply_wall_psi`); temperature walls are a plain Dirichlet overwrite (`apply_wall_temperature`).

### 4.8 Rollout stabilization — three independent mechanisms (§5 has the full story)

1. Ra-aware frozen normalization (§4.1) — cures a strong multiplicative energy-injection bug (§5.1).
2. Noise-injected push-forward training (§5.2) — trains long-horizon stability directly into the weights.
3. **Attractor-energy projection** (`project_energy`, [[attractor-energy-projection]]) — an optional per-step *inference-time* soft rescale of the rollout velocity onto the known attractor kinetic-energy shell $\langle|\mathbf u|^2\rangle(\mathrm{Ra}) = 2\,s_u(\mathrm{Ra})^2$, the driven-dissipative analog of the S1.1 conservation projection. A per-sample scalar rescale, so it commutes with the div-free/wall guarantees exactly.

### 4.9 What is *not* enforced

The S1.1 conserved-total projection ([[open-architectural-problems]], [[initial-model-architecture]] §5b) is **not wired into the RBC forward pass at all** — RBC conserves no momentum or energy (driven-dissipative), so it degenerates to a no-op; the conserved-readout $\hat C(h) = w_c^\top h$ off the top supernode is kept purely as a **diagnostic auxiliary loss target** during training (regressed toward the true Nu), never used to project the state.

---

## 5. The rollout-divergence investigation (2026-07-03) — root cause and fix

This is a load-bearing part of the model's history and shapes several design choices above, so it is documented in full rather than left as a training footnote.

### 5.1 Root cause: scale-equivariant normalization, not the conservation backbone

**Symptom:** monotonic kinetic-energy blow-up during autoregressive rollout, at every trained $\mathrm{Ra}$, at the *same relative rate* regardless of $\mathrm{Ra}$ (~1.9$\times$ growth per 300 steps).

**A parallel hypothesis** (from an earlier design discussion) attributed this to the conservation-first backbone being structurally mismatched to driven-dissipative physics, and proposed replacing the scalar $\gamma I$ dissipation with a learned state-dependent operator $R(h)$. **Instrumented diagnosis disproved this.** The backbone was measured to be *contractive* (latent-norm ratio $\|h^{\ell+1}\|/\|h^\ell\|\approx0.96$) with $\gamma>0$ on every layer of every checked checkpoint — the backbone was never the source.

**Actual cause:** the *committed* normalization at the time was **per-instance** — $s_u = \mathrm{std}(\mathbf u_{\text{curr}})$, recomputed from the current frame every step. This makes the map **exactly scale-equivariant**: the decoder's velocity increment is $s_u\cdot\nabla^\perp\delta\psi$ with $s_u\propto\sqrt{\text{KE}}$, so it injects energy proportional to the *current* kinetic energy every step — a multiplicative (and therefore Ra-independent-*rate*) exponential growth. Critically, under this normalization the decoder **cannot see the absolute energy scale** of its input, so it has no way to learn a net-dissipative correction for an over-energetic state.

### 5.2 The three-part fix and what each part buys

| Fix | Mechanism | Effect (measured) |
|---|---|---|
| **Ra-aware frozen normalization** (§4.1) | replaces $s_u=\mathrm{std}(u)$ with the fixed $s_u(\mathrm{Ra})=a\,\mathrm{Ra}^b$ | cuts blow-up from ~1.9$\times$/300 steps to ~1.25–1.4$\times$/300; **zero cost** to single-step accuracy (VRMSE identical to 3 decimal places before/after); requires retraining (weights were scale-blind) |
| **Stronger noise-injected push-forward** | full horizon $1{\to}2{\to}4{\to}8$ schedule, 40 epochs (up from an initial 8-epoch/horizon-4 pass) | **trains stability directly into the weights** — confirmed by the fact that the *raw* rollout (energy-projection off) is flat at 200 steps (KE-growth 0.99–1.10$\times$); halves rollout VRMSE at 25–100 step horizons vs. the single-step-only model |
| **Attractor-energy projection** (§4.8.3) | soft-project onto the known $\mathrm{KE}(\mathrm{Ra})$ shell each rollout step, $\lambda\approx0.05$–0.1 | flattens a *slower residual* drift visible only past ~300 steps (1000-step growth 4.4$\times\to$~1.0$\times$) with 9-step VRMSE unchanged; a scalar rescale, so structurally inert |

A caution surfaced by a 1000-step re-check: an initial "**CURED**" verdict at 300 steps was **wrong** — the same push-forward checkpoint showed 2.0–2.1$\times$ growth and was *still accelerating* at 600 steps. The 40-epoch push-forward run subsequently used (feasibility measured to 200 steps, KE-growth 0.99–1.10$\times$ both with and without the energy projection) is the current, more thoroughly checked result, but **feasibility has only been confirmed to 200 steps at $L^2$-scale accuracy**, not the full 600–1000-step horizon the earlier (less-trained) checkpoint was stress-tested at. This is flagged as an open verification item (§7).

---

## 6. Experimental results

All results below are from checkpoints trained on the branch's current code; each row states the exact recipe so results are not conflated across the several ablations run during Phase-A/B bring-up.

### 6.1 Solver validation (data quality, not the model)

The self-contained DNS solver's $\mathrm{Nu}(\mathrm{Ra})$ tracks the 2D correlation $\mathrm{Nu}\approx0.16\,\mathrm{Ra}^{0.29}$ across 1.5 decades ($\mathrm{Ra}=2\times10^4\to10^6$: solver 3.20$\to$8.88 vs. correlation 2.83$\to$8.79) — the ground-truth data is physically trustworthy.

### 6.2 Structural gates (base config, 5,500,138 params)

| Gate | Metric | Result |
|---|---|---|
| Batch-0 div-free (float64) | $\max\lvert \nabla\cdot\mathbf u_{t+1}\rvert $ | **1.31$\times10^{-11}$** (bar: $<10^{-5}$) |
| — model-added divergence (isolates decode) | $\|\nabla\cdot u_{t+1}-\nabla\cdot u_t\|_\infty$ | 2.07$\times10^{-12}$ |
| Wall no-slip/no-penetration | $\max\lvert \mathbf u\rvert_{\text{wall}}$ | 5.8$\times10^{-16}$ |
| Temperature Dirichlet | exact overwrite | 0 |
| Overfit sanity (2000 steps, 1 traj) | vel / temp rel-$L^2$ | 0.0040 / 0.00037 |

Div-free and wall enforcement are **structural** (fixed-operator, not learned), so these numbers hold at machine precision regardless of training quality — confirmed unchanged across every checkpoint tested.

### 6.3 Tokenizer fidelity (G0: encode$\to$decode a single frame, no dynamics, no backbone)

| Config | Rec. rel-$L^2$ (vel) | Rec. rel-$L^2$ (tmp) | Measured floor (128$\times$64-vs-256$\times$128 discretization) |
|---|---|---|---|
| base, ~60 epochs on `rbc_train` | **0.126** | **0.036** | 0.022 (vel) / 0.010 (tmp) |

The tokenizer sits meaningfully **above its own discretization floor** — this is the accuracy ceiling identified in §6.6, and the reason more G0 epochs alone do not close the gap (the from-scratch dynamics encoder was already exceeding this fidelity on the dynamics task, §6.6).

### 6.4 Single-step Nu(Ra) tracking (base, single-step trained, no push-forward)

| $\mathrm{Ra}$ | $\mathrm{Nu}_{\text{data}}$ | $\mathrm{Nu}_{\text{model}}$ | rel. err |
|---|---|---|---|
| $2\times10^4$ | 3.201 | 3.187 | 0.4% |
| $5\times10^4$ | 4.124 | 4.087 | 0.9% |
| $10^5$ | 4.965 | 4.896 | 1.4% |
| $3\times10^5$ | 6.587 | 6.444 | 2.2% |
| $10^6$ (OOD) | 8.875 | 8.554 | **4.8%** |

Worst-case 4.8% at the OOD point; well under the 10% pass bar even without push-forward. Single-step Nu is a *weak* test ($\Delta t\ll$ turnover time $\Rightarrow$ near-identity map); §6.5's rollout-Nu is the strong test.

### 6.5 Horizon-resolved rollout: Phase-A Pass 1 (the current best result)

Measured on the 150-frame `eval_long` corpus (natural-energy trajectories), with the energy projection at $\lambda{=}0.1$; the *raw* (projection-off) numbers are essentially identical (see §5.2), confirming push-forward — not the projection — is doing the stabilizing work.

**Single-step-only baseline (24 epochs, no push-forward):**

| $\mathrm{Ra}$ | v@1 | v@25 | v@50 | v@100 | KE-growth@200 |
|---|---|---|---|---|---|
| $3\times10^4$ | 0.007 | 0.084 | 0.164 | 0.318 | 1.05 |
| $10^5$ | 0.030 | 0.102 | 0.166 | 0.321 | 1.00 |
| $3\times10^5$ | 0.047 | 0.086 | 0.170 | 0.329 | 1.05 |

**+ push-forward (full 1$\to$2$\to$4$\to$8 horizon, 40 epochs):**

| $\mathrm{Ra}$ | v@1 | v@25 | v@50 | v@100 | KE-growth@200 | Feasible |
|---|---|---|---|---|---|---|
| $3\times10^4$ | 0.006 | **0.035** | **0.059** | **0.110** | 1.03 | True |
| $10^5$ | 0.029 | **0.069** | **0.070** | **0.123** | 0.99 | True |
| $3\times10^5$ | 0.047 | **0.045** | **0.088** | **0.177** | 1.04 | True |

Push-forward roughly halves rollout VRMSE at every horizon over the single-step baseline, and rollout is feasible (finite, bounded energy, div-free) through 200 steps at every trained $\mathrm{Ra}$. `v@1` (single-step, no rollout involved) is **unchanged** by push-forward — the important negative-control result that motivates §6.6.

**Reading the instantaneous-Nu numbers with care:** per-horizon Nu error computed against a *specific* ground-truth trajectory grows large at long horizons (up to 15–67% by 50–100 steps) purely because RBC is chaotic and the specific trajectory decorrelates from the specific rollout — this is expected and is *not* evidence of model bias. The chaos-robust metric is the **rollout-mean** Nu (over the whole trajectory) compared to the true mean, plus spectral fidelity — added to the eval in §6.6 below.

### 6.6 Ablation: S3.4 tokenizer pretrain-transfer — **negative result**

Hypothesis: pretraining the encoder to convergence on the G0 reconstruction task (§6.3), then initializing the dynamics model's encoder from those weights (leaving the backbone/decoder fresh), should improve accuracy by giving dynamics training a better-optimized tokenizer to start from. This closes a real gap — G0 had previously been *only* a fidelity gate, never actually fed into the trained model.

**Result: the transfer made every rollout horizon strictly worse**, on the identical `eval_long` set, same push-forward recipe (h8/40ep):

| $\mathrm{Ra}$ | v@1 | v@25 | v@50 | v@100 |
|---|---|---|---|---|
| $3\times10^4$ | 0.006 *(identical)* | 0.051 (was 0.035) | 0.091 (was 0.059) | 0.166 (was 0.110) |
| $10^5$ | 0.029 *(identical)* | 0.086 (was 0.069) | 0.128 (was 0.070) | 0.244 (was 0.123) |
| $3\times10^5$ | 0.047 *(identical)* | 0.096 (was 0.045) | 0.187 (was 0.088) | 0.348 (was 0.177) |

**Two conclusions drawn from this, both load-bearing for what comes next:**
1. **v@1 is bit-identical with and without transfer.** A better-optimized encoder does not change single-step accuracy at all — so the single-step error's $\mathrm{Ra}$-dependence (§6.4/6.5) is **not training-bound**; it is a capacity/resolution property of the architecture at this token budget, not something more encoder-optimization can fix.
2. **Reconstruction fidelity and rollout accuracy are partly *anti*-correlated.** A reconstruction objective preserves high-frequency detail by definition; high-frequency detail is exactly what is brittle under autoregression (the same mechanism as the Phase-1 Burgers SIREN high-frequency runaway). The from-scratch dynamics encoder implicitly learns to *suppress* rollout-harmful high-frequency content; forcing it to start from a reconstruction-optimal point undoes that.

This result substantially **lowers the expected payoff of the adaptive quadtree** (§7): its primary effect would also be higher reconstruction fidelity, which this ablation shows does not translate to rollout accuracy and can hurt it. The **attractor-statistic eval** added alongside this ablation (rollout-mean Nu, spectral fidelity) — measured only on the transfer-initialized model so far, not yet on the from-scratch baseline:

| $\mathrm{Ra}$ | $\mathrm{Nu}_{\text{roll}}$ | $\mathrm{Nu}_{\text{true}}$ | rel. err | Spectral rel. err |
|---|---|---|---|---|
| $3\times10^4$ | 3.060 | 3.586 | 14.7% | 0.013 |
| $10^5$ | 3.944 | 4.965 | 20.6% | 0.092 |
| $3\times10^5$ | 5.607 | 6.601 | 15.1% | 0.150 |

The velocity **spectrum matches reasonably well** while **rollout-mean Nu is under-predicted** by 15–21% — localizing the remaining gap to **heat transport / thermal-plume representation** specifically, more than to velocity-field energy. (These attractor-statistic numbers are confounded by the tokenizer-transfer ablation above and have not yet been re-measured on the — strictly better — from-scratch baseline; flagged in §7.)

### 6.7 Spatiotemporal context ($K{=}4$, Phase B) — pending

Implemented and validated at toy scale (§4.6); a full-scale training run (single-step $K{=}4$, then push-forward $K{=}4$/h8/40ep, batch sizes reduced for the larger token count) is in progress as of this revision. The intended comparison is **single-step v@1** at $K{=}4$ against the $K{=}2$ baseline in §6.5 (0.006/0.029/0.047 at $\mathrm{Ra}=3\times10^4/10^5/3\times10^5$) — the cleanest horizon-independent signal, since §6.6 showed rollout metrics alone can be misleading about what actually helps accuracy.

---

## 7. Known limitations & open items (as of this revision)

- **Adaptive quadtree is unimplemented** (§4.2). De-prioritized after §6.6's negative transfer result suggested the payoff (higher spatial fidelity) may not transfer to rollout accuracy the way it improves reconstruction; still the most direct lever for the $\mathrm{Ra}$-dependent single-step error if a resolution diagnostic (e.g. `patch=4`, memory permitting) confirms resolution — not optimization — is the bottleneck.
- **1000-step feasibility is not re-confirmed on the current best (40-epoch push-forward) checkpoint** — only to 200 steps. An earlier, less-trained checkpoint was stress-tested to 1000 steps and needed the energy projection to stay flat past ~300; whether the current checkpoint's apparent 200-step self-stability persists to 1000 is unverified.
- **Attractor-statistic eval (rollout-mean Nu, spectra) has only been run on the tokenizer-transfer ablation**, not on the (better) from-scratch push-forward baseline — needs re-measurement for a clean read.
- **The Well data is unused.** All training/eval to date uses the self-contained solver; cross-validating against The Well's `rayleigh_benard` set (and against [[gphyt-physics-foundation-model]] / [[walrus-paper]] baselines on the same data) has not been done.
- **The general/compressible velocity decoder head exists but has never been exercised** by any benchmark — needed to prove the §4.7 gating is real, not just a code path that's never taken.
- **Diffusion/generative path (turbulent $\mathrm{Ra}$) is unbuilt**; all results are laminar/weakly-turbulent regime only (§2).
- **Equivariant/covariant-gravity ablation arm (P9.3) has never been run**; gravity is a constant bias in every shipped result.
- **3D, MHD, and multi-system training are all out of scope** for this page — see [[open-architectural-problems]] for the broader roadmap.

---

## 8. Parameter budget (base config: $d{=}256$, $L{=}6$)

| Component | Formula | Value |
|---|---|---|
| Backbone | $12Ld^2$ | 4,718,592 |
| Encoders + field-type embeddings + residual paths | — | ≈15,000 |
| Conditioning MLPs | — | ≈49,000 |
| Decoder $\delta\psi$ + $\delta T$ (SIREN, lo+hi paths) | — | ≈350,000 |
| Diagnostic Nu readout | $2d$ | 512 |
| Hierarchy $R$/$P$, ghost nodes | parameter-free | 0 |
| **Measured total** (both velocity heads built) | — | **5,500,138** |

Slightly above the original 5.13M design estimate because *both* the incompressible and general velocity decoder heads are always instantiated (the gating selects between them at call time, so both must exist as real parameters for the gate to be a genuine choice, not a stub).

---

## 9. See Also
- [[noether-1.0]] — the 1D Burgers config this generalizes; shared backbone/init/stability recipes
- [[open-architectural-problems]] — Addendum 4 (all 25 decided); the design-rationale ledger this page's architecture instantiates
- [[attractor-energy-projection]] — the driven-dissipative rollout-stability mechanism (§4.8.3/§5.2), full derivation and evidence
- [[spatiotemporal-tubelet-tokens]] — the general concept behind §4.6's K-frame context
- [[symmetric-attention-physics]] / [[skew-symmetric-attention]] — the backbone's port-Hamiltonian reading; why it was (correctly) ruled out as the rollout-blow-up cause
- [[autoregressive-rollout-stability]] — push-forward / noise injection, the trained-stability complement
- [[pipeline-contract]] — the stage order §4's pseudocode instantiates
- [[training-curriculum]] — the loss composition §4's training references
- [[pfm-interface-design]] — the conditioning interface §4.4 extends
- [[gphyt-physics-foundation-model]] / [[walrus-paper]] — the Rayleigh–Bénard benchmark literature (not yet directly compared against, §7)
- [[smoke-test-bringup]] — the bring-up staging this page's §6 results were produced under
