# Smoke-Test Bring-Up: First Batch, Corpus Ladder, Implementation Readiness

**Type:** Concept — initial model bring-up plan (folder: Noether 1.0)
**Status:** The answer to "what do we actually run first." Companion to [[noether-1.0]] (the sizing/hyperparameter spec) — this page is the *experimental sequence* for going from zero to a validated first result on the 1D Burgers smoke test, plus a scoped implementation-readiness checklist.
**Related Concepts:** [[noether-1.0]], [[training-curriculum]], [[pipeline-contract]], [[open-architectural-problems]], [[graph-tokenizer]], [[intelligent-patching]]
**Related Summaries:** [[possible-architectures]] (the benchmark protocol this bring-up leads into)

---

## 0. Noether-1.0-core: the narrowed scope for a first implementation

Not every component in [[noether-1.0]] needs to exist to answer "does the architecture work." For **1D periodic Burgers specifically**, several pieces are structurally inactive or safely deferrable — narrowing what needs to be built before the first batch can run:

| Component | Status for Burgers day-1 | Why |
|---|---|---|
| Dual-path encoder, backbone (attn+skew+F1 FFN), S1.1 projection ($k{=}1$), additive decoder | **Active — build this** | the core loop; nothing here is Burgers-specific, all reusable at 2D NS |
| **Adaptive graph rebuild** (monitor-function bisection, [[noether-1.0]] §3.2) | **Active — build this (decision 2026-07-01)** | promoted from deferred to active: cheaply testable in 1D, and gives an early empirical check of [[open-architectural-problems]] Problem 7 (does conservation survive topology churn) well before the harder 2D/3D case |
| Typed-edge encoder (S4.1) | **Inactive** | one field, one node type — trivially a no-op until [[open-architectural-problems]] S4.4 |
| P2.1 elliptic/Poisson solve, P2.2 joint KKT projection | **Inactive** | periodic BC needs no ghost-node enforcement; no divergence-free constraint exists in 1D Burgers. Stage 10 of [[pipeline-contract]] reduces to [[noether-1.0]] §3.7's single closed-form rank-1 correction, now anchored to the fixed canonical grid so it's robust to the adaptive graph above — **no Poisson solve needed for this benchmark** |
| MTP head ($k{=}4$) | **Deferred** | start single-step; add per [[training-curriculum]] Phase 2 |
| Spectral loss, S6.1 uncertainty weighting | **Deferred / moot** | one field channel — nothing to balance yet; spectral loss adds real value once shocks are in the corpus (§3 Level 3) |
| Tier-2 SFA discovery loss (Group C) | **Deferred** | [[training-curriculum]] Phase 3, after B validates; and [[noether-1.0]] §3.7 already shows Burgers' only real invariant is known, not hidden — Group C has little to *discover* here, so it's a low-priority build item for this benchmark specifically |
| Diffusion module (S5.1) + regime router (S5.2) | **Deferred** | moderate-$\nu$ Burgers is shock-forming but not sustained-chaotic; per [[noether-1.0]] §3.9 this is "implement and unit-test, not load-bearing" — safe to build *after* the core validates |

This is **Noether-1.0-core**: tokenizer + adaptive graph + backbone + $k{=}1$ projection (canonical-grid anchored) + decoder + single-step head. Everything else in [[noether-1.0]] is real and eventually needed, but none of it blocks the first batch.

---

## 1. Batch 0 — architecture sanity / expressivity overfit test

Before any real curriculum, the literal first thing to run: **can this architecture fit anything at all?** Standard ML practice — a tiny fixed batch, trained to near-zero loss, with no train/val split. It answers a narrower, more urgent question than "does it generalize": does the differentiable pipeline (tokenizer round-trip, adaptive graph rebuild, backbone, the semi-orthogonal FFN's Stiefel retraction, the S1.1 projection) actually carry a gradient, and does it have the raw capacity to represent a nontrivial map, or does it collapse to a degenerate solution (constant output, identity, zero)?

| Setting | Value |
|---|---|
| Batch size | $B=8$ trajectories — small enough to memorize fast, large enough to catch a batch-dimension bug |
| $\nu$ | fixed at $0.01$ (moderate — shocks form but aren't razor-thin; not a pathological first target) |
| Initial condition | $u_0(x) = A\sin(2\pi x+\varphi)$, single mode, $A\sim\mathrm{Unif}(0.5,1.5)$, $\varphi\sim\mathrm{Unif}(0,2\pi)$ — the simplest nontrivial IC, chosen specifically because it *does* form a shock, so the adaptive graph (below) has something to refine toward from the very first test |
| Grid / graph | 128 canonical grid points; **adaptive bisection graph, 32 leaves + 31 ancestors = 63 nodes/snapshot** ([[noether-1.0]] §3.2) — exercised from Batch 0, not deferred, since a graph-construction bug is far cheaper to catch on 8 trajectories than after a real training run |
| Task | **single next-step only**, one $(t,t{+}1)$ pair per trajectory: context $(u_t,u_{t-1})$ [2-step, [[noether-1.0]] §2] $\to$ target $u_{t+1}$, solved via RK45 to $t=0.05$ |
| Loss | plain residual MSE; **S1.1's $k{=}1$ mass projection stays on**, anchored to the canonical grid ([[noether-1.0]] §3.7) — cheap, always-on, and better to catch a projection or graph-rebuild bug in batch 0 than bolt it on untested later |
| Off | MTP, spectral loss, Group C, diffusion, router, typed edges |
| Success criteria | (1) relative $L^2$ error on this *same fixed batch* $<0.01$ within a few thousand optimizer steps; (2) the 63-node graph is rebuilt correctly each step (exactly 31 splits reached, patches denser where $\eta$ is high — a visual/logged sanity check, not a loss term); (3) mass conservation holds to $<10^{-4}$ even with the graph changing shape step to step. **Failure on (1)/(2) is an implementation bug, not a generalization gap** — there is nothing to generalize to yet. Failure on (3) specifically would be evidence against [[noether-1.0]] §3.7's Problem-7 argument and needs its own investigation, not just a code fix |

If Batch 0 doesn't converge, the fix belongs in code (dead gradients through the Stiefel retraction, a degenerate projection nulling the useful signal, a broken skip connection, a bisection-tree bug), not in the design — the wiki-level design is not what Batch 0 tests.

---

## 2. Batch 1 — G0 tokenizer pretraining (first *real* curriculum batch)

Once Batch 0 confirms the pipeline is alive, real training starts at [[training-curriculum]] Phase 0 — the tokenizer fidelity gate ([[open-architectural-problems]] S3.4) — which is **encode $\to$ decode reconstruction of single snapshots**, not paired-timestep dynamics. This is the cheapest possible real data: it doesn't require treating the PDE as a time-evolution problem at all yet, only as a source of representative field *shapes*.

| Setting | Value |
|---|---|
| Data | snapshots $u(x,t_i)$ sampled at **random times across evolving trajectories**, not just initial conditions — deliberately mixing smooth early-time fields with shock-formed late-time fields, since the S3.1 residual channel specifically needs shock content to prove itself |
| $\nu$ range | log-uniform over $[10^{-3},10^{-1}]$ even at this early stage — tokenizer fidelity should hold across the physics range, not just one $\nu$ |
| Batch size | 64–128 snapshots, drawn from $\sim$100 trajectories $\times$ several sampled times each |
| Task | encode (dual-path) $\to$ decode (additive two-head, [[noether-1.0]] §3.8) $\to$ reconstruction loss |
| Target (S3.4 fidelity floor) | reconstruction error below the RK45 solver's own discretization error at this resolution. The earlier "$L^2<0.01$ as a workable first threshold" was a **placeholder** — **now refined (2026-07-02, build layer) against the promised doubled-resolution (256-point) reference solve.** Measured 128-vs-256 pooled discretization floor over the level-2 recipe = **0.0179** (relative $L^2$), *entirely* from $\nu<5\times10^{-3}$ shocks (per-band 0.0285) that are $\sim$1 grid cell wide and genuinely under-resolved at 128 points; mid/high-$\nu$ floor $\approx0$. **The 0.01 placeholder sat below this physical floor and was unachievable** without fitting sub-grid discretization artifacts. The real S3.4 gate is therefore: pooled reconstruction $\lesssim$ the 0.0179 floor **and** an excellent typical-snapshot median ($<0.008$). The Noether-1.0-base tokenizer meets this (pooled $0.019\approx$ floor, median $0.006$) — the latent is provably adequate, not the bottleneck. |

---

## 3. First dynamics batch (Phase 1, Group A)

Once G0 passes, dynamics training begins per [[training-curriculum]] — paired $(u_t,u_{t-1})\to u_{t+1}$ examples, Group A loss (residual regression; spectral term optional until Level 3 below introduces shock-heavy ICs), batch size 64–128 trajectories, matching [[noether-1.0]] §6.2's optimizer settings. This is the literal start of the "real" training run everything else in this folder was designed for.

---

## 4. Corpus expansion ladder — and the decided quick training set

[[possible-architectures]] already notes 10,000+ Burgers trajectories are generatable in minutes on CPU via RK45 — **data was never the constraint**, and for a sub-1M-parameter model it still isn't. The ladder below isn't about data economy; it's staged *complexity*, so a failure at any level points at a specific, narrow cause rather than an undifferentiated "it doesn't work."

| Level | Corpus | New axis of variation | Purpose |
|---|---|---|---|
| **0** | 8 trajectories (Batch 0) | none — fixed batch | expressivity / pipeline sanity (§1) |
| **1** | ~100–200 trajectories, fixed $\nu$ | random IC amplitude/phase/1–4 modes | does it generalize across ICs at fixed physics? |
| **2 — DECIDED quick training set** | **1,500 trajectories** | $\nu$ now varies (log-uniform) | exercises $z_{\text{param}}$ conditioning for the first time — generalization across physics regimes; see §4.1 for the full recipe |
| **3** | ~2,000–5,000 trajectories | IC families beyond sinusoids: multi-mode sums, Gaussian bumps, near-step-function ICs | stress-tests the S3.1 high-frequency residual channel and the S3.4 fidelity floor against genuinely sharp content |
| **4** | 10,000+ trajectories | full [[possible-architectures]] protocol | the destination: 10-step context $\to$ 90-step rollout, all four [[training-curriculum]] phases (MTP, Group C, diffusion), full [[noether-1.0]] §7 ablation suite (inviscid-Burgers conservative-channel test, F1-vs-F3, projection on/off) |

Level 4 is exactly [[noether-1.0]] §7's protocol — this ladder is the on-ramp to it, not a replacement.

### 4.1 Decision: Level 2, fully specified

Level 2 is the answer to "does our model perform" — enough to exercise the full single-field architecture (encoder, backbone, projection, decoder, *and* the $z_{\text{param}}$ conditioning pathway) with a genuine train/val generalization test, without paying for the full Level-4 curriculum (MTP, Group C, diffusion) this early. Levels 0/1 remain mandatory debugging gates *before* this run (a bug caught at Level 0 is cheap; the same bug caught only after a 1,500-trajectory run is not).

| Setting | Value |
|---|---|
| Trajectory count | **1,500** total — **1,200 train / 300 val** (80/20 split) |
| $\nu$ | $\log\nu\sim\mathrm{Unif}(\log10^{-3},\log10^{-1})$ (log-uniform, matching Level 2's stated range) |
| Initial condition | $u_0(x)=\sum_{k=1}^{3}A_k\sin(2\pi kx+\varphi_k)$, up to 3 modes; $A_k\sim\mathrm{Unif}(0.3,1.0)/k$ (energy tapering with $k$, a physically reasonable initial spectrum); $\varphi_k\sim\mathrm{Unif}(0,2\pi)$ |
| Solver | RK45 (`scipy.integrate.solve_ivp`), **spectral (FFT-based) spatial derivatives** — accurate and simple for periodic 1D Burgers, avoiding finite-difference upwinding choices near shocks |
| Grid / time | 128 points, $x\in[0,1]$ periodic; $\Delta t=0.01$, $t\in[0,2]$ (200 steps/trajectory) |
| Training pairs | context $(u_t,u_{t-1})\to$ target $u_{t+1}$, sampled from each trajectory; subsample every 5th valid window to reduce redundancy ($\approx$38 pairs/trajectory $\times$ 1,200 $\approx$ 45,600 training examples) |
| Loss | Group A only: residual MSE + spectral term, 2-way homoscedastic uncertainty weighting (S6.1) — **only 2 terms to balance**, since Burgers has a single field channel (no per-channel weighting needed, per [[training-curriculum]]) |
| Always on | S1.1's $k{=}1$ mass projection ([[noether-1.0]] §3.7) |
| Off | MTP, Group C, diffusion, router, typed edges — per §0's table |
| Success signal | held-out single-step relative $L^2$ **clearly below the persistence baseline** ($\hat u_{t+1}=u_t$) and monotonically decreasing with training — this is the "does it perform" bar. Approaching [[possible-architectures]]'s $<0.05$ full-rollout target is a stretch goal here, not a requirement — that's Level 4's job |
| Expected cost | $\approx$45,600 examples $\div$ batch 128 $\approx$ 357 steps/epoch; 10–20 epochs is enough to show a clear learning-or-not signal, well within [[possible-architectures]]'s 30-minute single-GPU budget given the model's size |

---

## 5. Implementation readiness checklist

| Status | Item |
|---|---|
| **Resolved, ready to code** | Dual-path encoder/decoder ([[graph-tokenizer]], [[noether-1.0]] §3.8's additive-decode rule, now also the residual head per §3.6's fix); **adaptive bisection graph rebuild** ([[noether-1.0]] §3.2 — monitor function, split rule, ancestor-based hierarchy); backbone attention + skew-symmetric channel + F1 FFN ([[symmetric-attention-physics]], [[skew-symmetric-attention]], [[open-architectural-problems]] Problem 12); S1.1 projection, correctly scoped to $k{=}1$ and canonical-grid-anchored so it's robust to the adaptive graph ([[noether-1.0]] §3.7); optimizer/schedule ([[noether-1.0]] §6.2); RK45 data generator (trivial — `scipy.integrate.solve_ivp`, per [[possible-architectures]]) |
| **Deferred by design, not by ambiguity** | Typed edges, P2.1/P2.2, MTP, spectral loss, Group C, diffusion + router — all per §0's table; each has a fully specified design elsewhere in this folder, just not needed for Burgers day-1 |
| **Pure engineering, zero remaining research question** | trajectory generator script; periodic radius-graph + bisection-tree construction; model modules in code; training loop + logging; Batch 0 as a permanent regression test (a broken future refactor should fail Batch 0 first, before wasting a full training run) |
| **Genuinely open, doesn't block this benchmark** | [[open-architectural-problems]] Problem 4 (cross-type message — blocks S4.4, not Burgers); Problem 7's *persistent-identity* sub-case (moving particles that must retain identity across steps — genuinely different from the *adaptive-patching* sub-case now active and tested here, [[noether-1.0]] §3.2/§3.7); Problem 8 (variable field cardinality — blocks true multiphysics, not single-field Burgers); full scored passes on Problems 9/10 (provisional defaults already suffice here) |

**Net answer to "what's between us and implementing":** for the narrow goal of Noether-1.0-core on 1D Burgers, nothing at the design level — every active component (row 1) is fully specified down to shapes and formulas. What's left is writing code and running Batch 0.

---

## [AI Inference]

**[AI Inference]:** [[noether-1.0]] §3.7's exact energy-decay identity ($-\nu\int u_x^2\,dx$) is more than a correction — it's a free, ground-truth-backed unit test that a hidden-invariant-discovery mechanism (S6.4) doesn't get to use elsewhere. Comparing the backbone's *learned* $\gamma$ against this closed form is a rare case in this project where "is the model learning the right physics" has an exact numerical answer to check against, rather than only a qualitative or statistical one — worth instrumenting from Batch 0 onward even though $\gamma$-tracking isn't part of the primary loss.

**[AI Inference]:** The corpus ladder's real function is fault isolation, not data budgeting: because each level introduces exactly one new axis of variation (IC diversity, then $\nu$ range, then IC sharpness, then the full rollout/curriculum), a failure at Level 2 but not Level 1 specifically implicates the $z_{\text{param}}$ conditioning pathway, and a failure at Level 3 but not Level 2 specifically implicates the S3.1 residual channel — the ladder turns "the model doesn't work" into a localized, actionable diagnosis for free, which a single jump straight to the full 10K-trajectory Level-4 corpus would not provide.

---

## See Also

- [[noether-1.0]] — the full architecture/implementation spec this bring-up sequence targets
- [[training-curriculum]] — the four-phase loss curriculum Batches 1+ instantiate
- [[pipeline-contract]] — the forward-pass stage order; §0's table maps directly onto which stages are active/inactive for Burgers
- [[possible-architectures]] — the full benchmark protocol that Level 4 of the corpus ladder reaches
- [[open-architectural-problems]] — Problems 4/7/8/9/10, confirmed non-blocking for this scope in §5
- [[graph-tokenizer]] — the dual-path encoder and S3.4 fidelity gate exercised at Batch 1
