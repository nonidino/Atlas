# Wiki Build Log

Append-only record of wiki operations. Format: `## [YYYY-MM-DD] operation | description`

---

## [2026-04-11] init | Wiki initialization — batch ingest of 13 sources

**Operation:** Initial wiki build from `/raw` directory scan.  
**Sources processed:** 13 documents across 14 wiki pages (Walrus has both blog + paper).  
**Pages created:**
- **Summaries (14):** gphyt-physics-foundation-model, walrus-paper, walrus-overview, pde-transformer-paper, pde-transformer-landing, deeponet-multi-operator, navier-stokes-nonuniform-grids, latent-diffusion-physics, message-passing-cyclicity, pc-deeponet-cfd, multiscale-diffusion-solar, varmion-viscous-flows, deep-memory-dissipative, aion-1-astronomy
- **Concepts (11):** physics-foundation-models, neural-operators, partial-differential-equations, navier-stokes-equations, transformer-architectures, in-context-learning-physics, diffusion-models-physics, autoregressive-rollout-stability, neural-surrogates, transfer-learning-fine-tuning, message-passing-belief-propagation, mixture-of-experts
- **Index:** index.md
- **Log:** log.md (this file)

**Total wiki pages:** 27  
**Sources not yet with full summaries:** The `2511.15684v1 (1).pdf` file in `/raw` appears to be the PDF version of the Walrus paper (already covered by the two Walrus pages). No additional content.

**Key themes identified:**
1. Physics Foundation Models — the central organizing theme
2. Polymathic AI cluster — Walrus, Latent Diffusion, Multiscale Solar, AION-1
3. Neural operator family — DeepONet, PC-DeepONet, VarMiON, D2NO
4. Transformer architectures — GP$_{\text{hy}}$T, PDE-Transformer, Walrus, AION-1
5. Generative emulators — diffusion models for physics
6. Statistical physics / networks — message passing cyclicity

**Cross-links established:** All summary pages link to relevant concept pages and to related summaries. All concept pages link to relevant summaries and to each other.

**AI Inferences written:** 14 total (one per summary), clearly marked with **[AI Inference]** prefix.

---

## [2026-04-11] note | PDF file in /raw not yet processed

**File:** `raw/2511.15684v1 (1).pdf`  
**Status:** Likely the preprint PDF of the Walrus paper (arXiv 2511.15684). Content covered by `summaries/walrus-paper.md`. To verify and potentially extract additional detail, process this PDF in a future session.

---

## [2026-04-12] note | Synthesis concept page: PFM architecture approaches

**Operation:** Created synthesis concept page from user query.  
**Page created:** `concepts/pfm-architecture-approaches.md`  
**Content:** 7 architectural paradigms compared (AR next-step, neural differentiator, diffusion/generative, masked multimodal, neural operators, hard physics constraints, spectral methods); pros/cons table; hybrid synthesis roadmap; 4 open questions. Cross-links to all relevant summaries and concepts.  
**Trigger:** User asked for a comparison of tested PFM approaches — page compiled from wiki contents.

---

## [2026-04-11] ingest | Two new sources added by user

**Operation:** Incremental ingest of 2 new files added to `/raw`.  
**Sources processed:**
1. `raw/An Intuitive Tutorial to Gaussian Process Regression.md` (arXiv 2009.10862v5, Jie Wang)
2. `raw/Physics-informed diffusion models in spectral space.md` (arXiv 2602.09708v1, Gallon et al.)

**Pages created:**
- `summaries/gaussian-process-regression.md` — GPR tutorial; GP definition, RBF/Matérn kernels, posterior conditioning ($O(N^3)$), Cholesky algorithm, hyperparameter optimization via log marginal likelihood; AI inferences on attention-as-kernel-regression and GP as evaluation baseline
- `summaries/pisd-physics-informed-spectral-diffusion.md` — PISD paper; spectral encoding with Sobolev regularity lemma; Adam-based DPS guidance for physics constraints; 100–10,000× lower PDE residuals vs. DiffusionPDE; 3–15× faster inference; AI inferences on training-free physics conditioning and spectral PFM
- `concepts/gaussian-processes.md` — New concept page; full GP theory with kernel derivations, NTK connection, attention-as-kernel-regression, physics-informed kernels (Green's functions, derivative GPs), comparison vs. neural surrogates

**Pages updated:**
- `index.md` — Added new summaries and concept; updated page counts (27 → 31); added PISD to Physics Encoding Spectrum; new "Statistical / Probabilistic Methods" summary section
- `log.md` — This entry

**Total wiki pages:** 31  
**Running total sources ingested:** 15

---

## [2026-04-13] note | Architecture benchmark planning: 5-way comparison + Burgers' benchmark

**Operation:** Created synthesis/planning concept page from user query.  
**Page created:** `concepts/possible-architectures.md`  
**Content:** Five architectures selected for direct empirical comparison — three from existing wiki research (AR Transformer, Diffusion Backbone, Neural Differentiator) and two new proposals (Physics-Mamba SSM, GNN with Physics Bottleneck). Includes full design rationale, key risks, AI inferences, and a minimal benchmark spec (1D Burgers' equation, 1M–10M params, relative $L^2$ metric).  
**Trigger:** User asked for two new architecture ideas beyond the three best-supported paradigms, plus a simple scaling benchmark equivalent to training a transformer on Shakespeare.

---

## [2026-04-13] ingest | Batch ingest of 4 new sources from `/new` directory

**Operation:** Incremental ingest of 4 files from `/new`.  
**Sources processed:**
1. `new/sanchez-gonzalez20a.md` — GNS: Graph Network-based Simulators (Sanchez-Gonzalez et al., ICML 2020)
2. `new/Dynami-CAL GraphNet...md` — Physics-informed GNN with exact linear + angular momentum conservation (Sharma & Fink, arXiv 2501.07373)
3. `new/EquiformerV3...md` — SE(3)-equivariant graph attention transformer for atomistic modeling (Liao et al., arXiv 2604.09130)
4. `new/Equivariant Quantum Neural Networks for High Energy Physics Analysis at the LHC.md` — EQNN with p4m symmetry for HEP image classification (GSoC 2024 / ML4SCI)

**Summaries created (4):**
- `summaries/gns-graph-network-simulators.md` — encode-process-decode particle GNN; multi-material physics; 34× generalization; noise injection for rollout stability
- `summaries/dynami-cal-graphnet.md` — antisymmetric edge-local frame; exact conservation of linear + angular momentum; ghost-node boundaries; edge memory; outperforms GNS and equivariant GNNs on granular/biomolecular tasks
- `summaries/equiformer-v3.md` — merged layer norm; SwiGLU-S² activations; smooth radius cutoff; 1.75× speedup; SOTA on OC20/OMat24/Matbench Discovery
- `summaries/eqnn-hep-lhc.md` — quantum ML; p4m equivariant embedding + twirling ansatz + invariant measurement; HEP jet classification; less central to continuum PFM goal

**Concept page created (1):**
- `concepts/equivariant-gnns.md` — Three-level taxonomy of physics inductive biases in GNNs: spatial (GNS), SE(3) symmetry (EquiformerV3, EGNN), conservation laws (Dynami-CAL); comparison with other PFM paradigms; path to GNN-scale PFMs

**Concept pages updated (2):**
- `concepts/pfm-architecture-approaches.md` — Added Paradigm 8: Particle GNNs (GNS / Dynami-CAL); updated hybrid architecture synthesis; added 2 new open questions; new cross-links
- `concepts/possible-architectures.md` — Upgraded GNN-PB proposal with new empirical evidence from GNS (backbone) and Dynami-CAL (hard conservation); revised architecture to combine exact momentum conservation + soft energy bottleneck; added EquiformerV3 AI inference; updated cross-links

**Total wiki pages:** 39  
**Running total sources ingested:** 19

**Key themes from this batch:**
1. **Particle-based GNN simulation** — a distinct paradigm from continuum field methods; handles irregular geometry, many-body interactions, and generalizes far beyond training distribution
2. **Hard conservation laws in GNNs** — Dynami-CAL proves exact Newton's-third-law conservation is architecturally achievable, generalizable, and robust to dissipation/external forces; upgrades physics encoding spectrum level 5 for GNNs
3. **SE(3)-equivariant scaling** — EquiformerV3 shows equivariant GNNs can scale to competitive sizes with smart algorithmic choices ($O(L_{\max}^4)$ tensor products, merged normalization, smooth cutoffs)
4. **GNN-PB proposal now better-supported** — originally a speculative proposal; now has direct empirical grounding from GNS (architecture) and Dynami-CAL (conservation)

---

## [2026-04-14] ingest | Conditional Memory via Scalable Lookup (Engram)

**Operation:** Incremental ingest of 1 LLM architecture paper with physics analogy.  
**Source processed:**
- `new/Conditional Memory via Scalable Lookup...md` — Engram: hybrid static memory (n-gram lookup) + dynamic MoE for sparse transformers (Cheng et al., arXiv 2601.07372)

**Summary created (1):**
- `summaries/conditional-memory-engram.md` — Engram mechanism; U-shaped scaling law between memory and compute; mechanistic benefits (early-layer relief, attention reallocation); LLM results; physics analogy: bifurcate computation into pre-computed canonical solutions (memory) vs. learned heterogeneous corrections (MoE)

**Concept page created (1):**
- `concepts/memory-augmented-physics-models.md` — Framework for hybrid static-dynamic PFM architectures; static physics library (Green's functions, canonical solutions, bases) + dynamic neural correction; U-shaped scaling law for optimal memory-compute split; multi-scale application; placement on physics encoding spectrum; connections to neural differentiators and neural operators; open questions on library design, routing, multi-physics

**Total wiki pages:** 41  
**Running total sources ingested:** 20

**Key theme:**
The Engram paper inspires a new architectural direction for PFM: **memory-augmented transformers** that delegate frequent/canonical physics tasks to a pre-computed lookup table, freeing the neural backbone for heterogeneous corrections and out-of-distribution regimes. The U-shaped scaling law suggests there is an optimal balance between memory size and compute size, providing empirical guidance for architecture design.

---

## [2026-04-14] ingest | Genesis: Universal Physics Engine (3 sources → 1 summary)

**Operation:** Batch ingest and synthesis of 3 Genesis sources into 1 unified summary.  
**Sources processed:**
1. `new/Genesis.md` — Official project page; overview of capabilities, benchmarks, gallery
2. `new/Genesis🌌 A Revolutionary Platform for Physics and Embodied AI.md` — Medium article; feature overview, installation, mission
3. `new/Genesis — Genesis 0.4.5 documentation.md` — Official documentation; API, architecture, long-term missions

**Summary created (1):**
- `summaries/genesis-physics-engine.md` — Genesis as unified physics engine + generative data platform; performance (43M FPS, 10–80× speedup); unified solver framework (rigid, MPM, deformable, soft, fluid); VLM-based generative agent for automated multi-modal data synthesis (videos, policies, scenes, trajectories); differentiable simulation; relevance to PFM: (1) training data source at scale, (2) differentiable physics for gradient-based learning, (3) architectural inspiration for modular solver composition, (4) sim2real policy transfer

**Total wiki pages:** 42  
**Running total sources ingested:** 23

**Key themes:**
1. **Genesis as PFM training data infrastructure:** Can generate millions of physically-accurate trajectories, videos, and policies overnight. Addresses the "data bottleneck" limiting neural physics model scaling.
2. **Unified solver framework as architectural pattern:** Genesis successfully integrates multiple physics solvers (rigid, soft, fluid, deformable) into one framework. Suggests PFM should have **modular solver routing** (per-regime specialization via MoE-like routing) rather than monolithic end-to-end learning.
3. **Differentiable simulation:** Genesis supports gradient flow through physics (MPM, Tool Solver implemented; rigid-body coming). Enables **physics-in-the-loop learning** — training PFM by differentiating against Genesis ground truth.
4. **Generative simulation at scale:** VLM agent automatically decomposes natural language prompts into physics simulation API calls. Suggests PFM could self-improve via active learning: iteratively ask Genesis for harder scenarios, train, repeat.

---

## [2026-05-14] note | Architecture concept pages — full internal design specs for all 5 benchmark architectures

**Operation:** Created 5 detailed architecture specification pages in `wiki/concepts/`, one per architecture in `concepts/possible-architectures.md`. Each page is designed to support direct model implementation (not just conceptual overview).

**Pages created (5):**
- `concepts/arch-autoregressive-transformer.md` — AR Transformer (Arch 1): tubelet patching, patch jittering, axial RoPE, T5 temporal bias, QK normalization, factorized space-time attention, residual prediction, variable Δt, push-forward training, in-context learning mechanism; synthesizes Walrus + GP$_{\text{hy}}$T design decisions
- `concepts/arch-diffusion-backbone.md` — Diffusion Backbone (Arch 2): physics autoencoder with saturating latent bounds, spectral normalization with Sobolev regularity lemma, rectified flow noise schedule, DiT denoiser with AdaLN conditioning, Adam-based DPS physics guidance, temporal bundling, UQ via ensembles; synthesizes PISD + Latent Diffusion work
- `concepts/arch-neural-differentiator.md` — Neural Differentiator + Integrator (Arch 3): derivative-augmented input channels ($\partial_x, \partial_{xx}, \partial_t$), transformer backbone, derivative prediction head, Euler/RK4 integration, ODE error bound analysis, derivative target normalization; primary evidence from GP$_{\text{hy}}$T (7× SOTA)
- `concepts/arch-physics-mamba.md` — Physics-Mamba SSM (Arch 4): Mamba selective SSM with HiPPO initialization, Δt-as-explicit-input via log-linear shift on discretization step, axial 2D scan (bidirectional), spatial-temporal Mamba interleaving, optional global attention for elliptic PDEs, linear-time inference recurrence; new proposal
- `concepts/arch-gnn-physics-bottleneck.md` — GNN-PB (Arch 5): radius graph with ghost-node boundaries, antisymmetric edge-local frame for exact momentum conservation (Dynami-CAL), symmetric edge embeddings, edge memory across sub-steps, physics bottleneck layer (Π_conserved projection through physical coordinates), energy-only soft bottleneck, encode-process-decode with Euler integrator; synthesizes GNS + Dynami-CAL

**Index updated:** Added "Benchmark Architecture Specifications" section; page count 42 → 47.

**AI Inferences written:** 3 per page × 5 pages = 15 total, all clearly marked with `**[AI Inference]:**`.

**Key new content:**
1. **Mamba SSM internal mechanics** for physics: HiPPO initialization, log-linear Δt input shift, bidirectional axial scan, per-PDE-type global attention tradeoff
2. **GNN-PB full message-passing pseudocode** with Dynami-CAL antisymmetric frame + energy bottleneck, including exact conservation verification procedure
3. **Diffusion backward process** with Adam-based DPS guidance schedule and timing (full $T \to 1$, not just final steps)
4. **Derivative-augmented channels** for Burgers' specifically: $[u, \partial_x u, \partial_{xx} u, \partial_t u]$ — direct connection to Burgers' RHS

**Total wiki pages:** 47
**Running total sources ingested:** 23 (no new sources; these are synthesis pages from existing wiki content)

---

---

## [2026-05-16] ingest | Batch ingest of 3 new sources from `/new` directory

**Operation:** Scheduled task — incremental ingest of 3 files from `/new`; sources moved to `/raw`.  
**Sources processed:**
1. `new/A Mathematical Explanation of Transformers for Large Language Models and GPTs.md` (arXiv:2510.03989, Tai, Liu, Li, Chan)
2. `new/PhysicsFormer An Efficient and Fast Attention-Based Physics-Informed Neural Network for Solving Incompressible Navier–Stokes Equations.md` (arXiv:2601.03613, Barman, Chatterjee, Ray)
3. `new/From Kepler to Newton Inductive Biases Guide Learned World Models in Transformers.md` (arXiv:2602.06923, Liu, Sanborn, Ganguli, Tolias — ICML)

**Summaries created (3):**
- `summaries/transformer-mathematical-framework.md` — Rigorous derivation of Transformer as discretization of an integro-differential equation via Lie operator splitting; attention = non-local integral operator; LayerNorm = projection to constraint set $S_1$; ReLU = projection to $S_2 = \{u\geq 0\}$; each layer = one time step; unifies Transformers, CNNs, UNets under PDE discretization lens
- `summaries/physicsformer-pinn-ns.md` — Encoder-decoder transformer PINN; pseudo-sequence data embedder converting pointwise $(x,t)$ to temporal sequences; novel $w\sin(t)$ trainable activation; dynamic loss weighting; MSE $\approx 10^{-6}$ on Burgers + 2D NS; 0% inverse problem error (clean); 3× faster than PINNsFormer; ~500MB GPU; handles high-frequency solutions ($\beta=50$ convection) where standard PINNs fail
- `summaries/kepler-newton-inductive-biases.md` — Three inductive biases for world-model transformers: (1) spatial smoothness (continuous coords or small vocabulary $V$), (2) spatial stability (noisy context learning $\sigma\approx 0.1$), (3) temporal locality (context length = ODE order); context length controls phase transition between Newtonian (local/causal) and Keplerian (global/geometric) world models; scaling law $1-R^2 \approx AD^{-1.15}V^{1.33}$

**Concept pages created (1):**
- `concepts/world-models-physics-ai.md` — New concept; Newtonian vs. Keplerian dichotomy; three inductive biases; linear probing for world-model verification; tension with in-context learning (long context = better prediction, short context = better physics discovery); AI inferences on mixed-context attention and symbolic interface needs

**Concept pages updated (2):**
- `concepts/transformer-architectures.md` — Added: (1) mathematical foundation section (transformer as PDE discretization), (2) context length and world model type table; updated related sources and cross-links
- `concepts/navier-stokes-equations.md` — Added PhysicsFormer row to ML approaches table + detailed PhysicsFormer PINN section; updated cross-links

**Index updated:** Page count 47 → 52; sources 23 → 26.

**Total wiki pages:** 52  
**Running total sources ingested:** 26

**Key themes from this batch:**
1. **Transformer = PDE discretization** — Tai et al. provide the first rigorous continuous-to-discrete proof. This closes a theoretical gap and provides a principled basis for embedding physical operators into attention kernels (hard architectural constraint design).
2. **World models vs. curve-fitting** — Liu et al. (ICML) identify the precise architectural choices (context length, input continuity, noisy training) that determine whether a transformer discovers causal physics vs. fits geometric patterns. Critical for PFM architectures aiming at scientific discovery.
3. **Transformer-PINN synthesis** — PhysicsFormer shows transformer + PINN is computationally viable, efficient, and achieves exceptional inverse problem accuracy. Extends the PFM toolkit for scenarios requiring physics enforcement without large training datasets.

---

## [2026-05-18] ingest | A Mathematical Perspective on Transformers (Geshkovski et al., arXiv:2312.10794)

**Operation:** Scheduled task — incremental ingest of 1 file from `/new`; source moved to `/raw`.  
**Source processed:**
1. `new/A mathematical perspective on Transformers.md` (arXiv:2312.10794v5, Geshkovski, Letrouit, Polyanskiy, Rigollet — MIT + Paris-Saclay)

**Summary created (1):**
- `summaries/transformers-particle-systems-clustering.md` — Transformer tokens modeled as a **mean-field interacting particle system** on $\mathbb{S}^{d-1}$; interaction energy $\mathsf{E}_\beta[\mu] = \frac{1}{2\beta}\iint e^{\beta\langle x,x'\rangle} d\mu d\mu'$ increases monotonically under dynamics, driving clustering; Transformer = Wasserstein gradient flow with energy-weighted metric; clustering theorems for $d\geq 3$ (any $\beta$), $d\geq n$ (exponential rate); two-phase metastable dynamics (fast clustering, slow merging); connections to Kuramoto oscillators, Krause model, optimal sphere configurations

**Concept pages updated (1):**
- `concepts/transformer-architectures.md` — Added "Dynamical Systems Perspective: Token Clustering" section with interacting particle model equations, clustering summary table, and implications for physics transformer design; updated See Also links

**Index updated:** Page count 52 → 53; sources 26 → 27.

**Total wiki pages:** 53  
**Running total sources ingested:** 27

**Key themes:**
1. **Statistical mechanics of transformers** — Complements the PDE discretization view (Tai et al.) with a dynamical systems / mean-field perspective. Together these two frameworks give a complete picture: Transformers discretize a continuous IDE *and* their long-time behavior is governed by a clustering energy functional.
2. **Token clustering as fundamental transformer property** — Rigorous proof that $d\geq 3$ always causes consensus; practically, trained models live in a metastable clustered state. This is the mathematical basis for "over-smoothing" and "rank collapse" pathologies.
3. **PFM design implication** — Residual connections, multi-scale architectures, and moderate depth counteract over-clustering. Transformer depth should be chosen to reach the metastable state but not the infinite-time limit. The interaction energy perspective suggests physics transformers should explicitly preserve multiple cluster identities (one per physical field or scale).

---

## [2026-05-18] note | Duplicate source detected and archived

**Operation:** Scheduled task ingest scan — 1 file found in `/new`.  
**File:** `new/2023_transformers2.md` — full text of "A Mathematical Perspective on Transformers" (Geshkovski, Letrouit, Polyanskiy, Rigollet; arXiv:2312.10794v5).  
**Status:** Duplicate — this paper was already ingested on 2026-05-18 from `new/A mathematical perspective on Transformers.md`. Summary exists at `summaries/transformers-particle-systems-clustering.md`.  
**Action:** Source moved to `/raw/2023_transformers2.md`; no new wiki pages created; no index or concept updates needed.

---

## [2026-05-18] ingest+note | Five new sources + four extension-architecture concept pages

**Operation:** User-directed extension. Following a brainstorming session on extending the "transformer = fully-connected GNN, attention as a physical force" interpretation, four extension concepts were identified that go beyond the existing 5-architecture benchmark plan. Each is grounded in canonical external literature retrieved via web search and arXiv fetches.

**Sources processed (5, abstract-and-extract format with arXiv links):**
1. `raw/Multipole Graph Neural Operator for Parametric Partial Differential Equations.md` — Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, Anandkumar; NeurIPS 2020; arXiv:2006.09535. FMM-inspired multi-level graph neural network; $O(N)$ linear complexity capturing interactions at all length scales; equivalent to multi-resolution matrix factorization; discretization-invariant.
2. `raw/Hamiltonian Neural Networks.md` — Greydanus, Dzamba, Yosinski; NeurIPS 2019; arXiv:1906.01563. Parameterize $H_\theta(q,p)$; recover dynamics via Hamilton's equations through autodiff; exact energy conservation; time-reversibility; foundational for physics-structured neural ODEs.
3. `raw/Dissipative Hamiltonian Neural Networks.md` — Sosanya, Greydanus; 2022; arXiv:2201.10085. Parameterize Hamiltonian + Rayleigh dissipation function jointly; Helmholtz decomposition; recovers HNN when D=0; learns friction separately from inertia.
4. `raw/Lagrangian Neural Networks.md` — Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho; ICLR 2020 workshop; arXiv:2003.04630. Parameterize $L_\theta(q, \dot q)$; solve Euler-Lagrange for $\ddot q$ via mass-matrix inversion; works without canonical coordinates (relativistic, constrained systems).
5. `raw/Noether Networks Meta-Learning Useful Conserved Quantities.md` — Alet, Doblar, Zhou, Tenenbaum, Kawaguchi, Finn; NeurIPS 2021; arXiv:2112.03321. Meta-learn auxiliary conserved quantity $g_\phi$; bilevel optimization with test-time tailoring; discovers task-specific approximate conservation laws.
6. `raw/Anti-Symmetric DGN.md` — Gravina, Bacciu, Gallicchio; ICLR 2023 (Best Student Paper); arXiv:2210.09789. Skew-symmetric weight matrix $W - W^\top - \gamma I$; purely imaginary Jacobian eigenvalues; resolves over-squashing and gradient pathologies; enables stable deep stacking.

**Summary pages created (5):**
- `summaries/multipole-graph-neural-operator.md` — MGNO mathematical framework; V-cycle implementation; equivalence to $\mathcal{H}$-matrices and multigrid; computational properties; relation to classical FMM/multigrid; relevance to GNN-PB long-range gap.
- `summaries/hamiltonian-neural-networks.md` — HNN parameterization and loss; energy conservation as architectural property; experiments (mass-spring, pendulum, two-body, pixel pendulum); limitations and extensions (canonical coordinates, conservative systems, autonomous Hamiltonian).
- `summaries/dissipative-hamiltonian-neural-networks.md` — Helmholtz decomposition; H+D joint parameterization; reduces to HNN when D→0; experiments on damped systems; connection to port-Hamiltonian systems.
- `summaries/lagrangian-neural-networks.md` — Euler-Lagrange equation solve via mass-matrix inversion; advantages over HNN (relativistic, constrained, field-theoretic); Lagrangian Graph Networks for many-body systems; symbolic distillation pathway.
- `summaries/noether-networks.md` — Noether's theorem in practical form; conservation loss; inner-loop tailoring; outer-loop meta-learning; theoretical recovery guarantees; experiments on physics and video prediction.
- `summaries/anti-symmetric-dgn.md` — Continuous-time ODE reformulation; skew-symmetric weight; Jacobian eigenvalue analysis; stability theorem; relation to AntisymmetricRNN and broader structured-neural-ODE literature; Hamiltonian/symplectic connection.

**Concept pages created (4) — Extension Architecture Proposals:**
- `concepts/multiscale-hierarchical-gnn.md` — Full FMM-analog architecture proposal: hierarchy of graphs, V-cycle message passing, learned restriction/prolongation, combination with Dynami-CAL and EquiformerV3 irreps. Closes long-range coupling gap in GNN-PB. Position on physics encoding spectrum: level 4-5 depending on combination.
- `concepts/hamiltonian-message-passing.md` — Per-edge pairwise potential $V_{ij}$ + Rayleigh dissipation $D_{ij}$; conservative force from autodiff gradient; symplectic integrator at outer loop; combined with Dynami-CAL antisymmetric edge frame gives exact momentum + exact energy (conservative limit). Upgrades GNN-PB energy bottleneck from level 4 → level 5.
- `concepts/action-based-noether-enforcement.md` — Action functional + explicit symmetry enforcement + Noether-discovered auxiliary conservation. Subsumes equivariant networks (equivariance derives from action symmetry). The action-based framing is closest to fundamental physics (mechanics, EM, GR, QFT all action-based). Combination LNN + Noether Networks = strongest known conservation framework.
- `concepts/antisymmetric-signed-attention-transformer.md` — Three architectural fixes to standard transformer attention: symmetric attention score, force-style antisymmetric aggregation, skew-symmetric linear part. Resolves the clustering/over-smoothing pathology proved in [[transformers-particle-systems-clustering]]. Restores Newton's 3rd law at the attention level; supports signed (repulsive) coupling; preserves information across deep stacks.

**Index updated:** Added "Hamiltonian / Lagrangian / Action-Based Papers" summary section; added "Extension Architecture Proposals" concept section; updated MGNO and A-DGN entries in Graph Network / Equivariant GNN Papers section; updated Physics Encoding Spectrum to include new approaches; updated page count (53 → 62) and source count (27 → 32).

**AI Inferences written:** Multiple per page, all clearly marked with `**[AI Inference]:**`.

**Total wiki pages:** 62
**Running total sources ingested:** 32

**Key themes from this batch:**

1. **Action-based / Lagrangian formulation as PFM endpoint.** Fundamental physics (mechanics, EM, GR, QFT) is action-based. The Lagrangian formulation is the most general representation and naturally extends to field theories, relativistic systems, gauge theories, and constrained mechanics. A PFM based on action functional + Noether discovery + meta-learned auxiliary conservation is the architectural endpoint that this body of literature points toward.

2. **Hard energy conservation now achievable in GNNs.** The wiki previously had hard momentum conservation (Dynami-CAL) but only soft energy conservation (GNN-PB bottleneck). The Hamiltonian message passing concept page closes this gap: parameterizing pairwise interactions via potential functions gives exact energy conservation in the conservative limit, with explicit Rayleigh dissipation for the dissipative part. This is a substantive upgrade to the physics encoding spectrum's level-5 category for graph networks.

3. **Long-range coupling for irregular geometries is now addressable.** The MGNO + GNN-PB combination resolves what was previously the wiki's single largest architectural gap: long-range elliptic coupling (Poisson, Stokes, gravitational, Coulomb) on irregular geometries. Multipole hierarchy gives $O(N)$ all-range coupling at linear cost.

4. **Transformer attention pathology has a rigorous fix.** The [[transformers-particle-systems-clustering]] paper proved that standard transformer attention is structurally dissipative (drives tokens to Dirac mass). A-DGN's skew-symmetric formulation gives the rigorous mathematical fix: replace softmax with symmetric+signed attention and add a skew-symmetric linear part. The resulting "antisymmetric attention transformer" preserves information indefinitely and enforces Newton's 3rd law at the attention level.

5. **The four extensions compose.** Each of the four new concept pages is independent in principle, but they compose cleanly: multipole hierarchy provides multi-scale structure; Hamiltonian message passing provides exact conservation per scale; action-based / Noether provides symmetry-derived and discovered conservation; antisymmetric attention provides stable deep stacking. The compositional endpoint — Lagrangian Functional + Multipole Hierarchy + Antisymmetric Attention + Noether Networks — is the architectural maximum the wiki currently points toward. No published work realizes this combination.

**Triggering context:**
User-initiated brainstorm: "Since a transformer is just a fully connected GNN mathematically, and mathematically speaking the tokens can be thought of as a multiparticle system interacting through attention as an attractive force, can we extend this to create a GNN for physics?" Discussion identified four extension directions; this batch processes the canonical literature for each and formalizes the resulting architectures.

---

## [2026-06-26] note | Concept overview page: What is a PFM, capabilities, and gaps

**Operation:** Synthesis concept page created from user query.  
**Page created:** `concepts/pfm-concept-overview.md`  
**Content:** Three-part synthesis —
1. **Definition** (one-paragraph LLM analogy; "train once, deploy anywhere" hypothesis)
2. **What a PFM would do** (7 capabilities: in-context simulation, zero-shot transfer, multi-physics coupling, inverse problem solving, uncertainty quantification, world-model/scientific discovery, speed over classical solvers) — each with equations and literature grounding
3. **What is currently lacking** (8 gaps: unified representation, error accumulation, scale gap, physics enforcement without generality loss, irregular geometry at scale, multi-scale without supervision, benchmark infrastructure, training data) — each with current state vs. needed, plus summary gap table

**AI Inferences:** One (near-term hybrid architecture synthesis), clearly marked.  
**Trigger:** User asked for a concept page covering what a PFM is, what it would do, and what is currently lacking.

**Total wiki pages:** 63  
**Running total sources ingested:** 32 (no new sources; synthesis from existing wiki)

---

## [2026-06-26] ingest | Multimodal Learning With Transformers: A Survey (Wu et al., 2023)

**Operation:** Incremental ingest of 1 file from `/new`; source moved to `/raw`.  
**Source processed:**  
- `new/Multimodal Learning With Transformers A Survey.md` — Medium blog post reviewing Wu et al. (2023), "Multimodal Large Language Models: A Survey", arXiv:2311.13165v1.

**Summary created (1):**
- `summaries/multimodal-transformers-survey.md` — 4-phase history of multimodal ML; 5 core technical components (tokenization taxonomy: region/grid/patch; learning objectives ITC/MLM/MVM/ITM; model structure encoder-only vs. encoder-decoder; fusion encoder vs. dual encoder; prompts); notable models (BLIP-2 Q-Former, Flamingo Perceiver Resampler, LLaVA, MiniGPT-4, LLaMA-Adapter); challenges (modality expansion, training cost, catastrophic forgetting); PFM relevance analysis.

**Concept pages created (2):**
- `concepts/multimodal-tokenization.md` — Full taxonomy of visual/physical tokenization approaches synthesizing source + web research: (1) patch continuous (ViT), (2) discrete VQ (VQ-VAE/GAN/DALL-E), (3) dual-codebook (TokenFlow CVPR 2025, SemHiTok), (4) Q-Former compression (BLIP-2, fixed 32 tokens), (5) Perceiver Resampler (Flamingo, fixed 64 tokens), (6) continuous projector (LLaVA), (7) video/tube (Walrus, Cosmos), (8) wave-particle dual (arXiv:2511.01593); comparison table; 5 specific gaps for physics tokenization; physics-conditioned Q-Former proposal.
- `concepts/iterative-refinement-pfm.md` — Progressive physics refinement architecture (user-proposed); mathematical formulation; relationship to diffusion/PISD/multigrid/ADMM; autoregressive-across-time structure; analysis of context-helps vs. error-accumulates competing effects; minimum context length = ODE order; single-snapshot sufficiency analysis; assessment of strengths and weaknesses.

**Index updated:** Added survey to Foundation Model Papers; added both concept pages; page count 63 → 66; source count 32 → 33.

**Total wiki pages:** 66  
**Running total sources ingested:** 33

**Key themes from this batch:**
1. **Tokenization is a first-class design problem.** Current PFMs (Walrus, GP$_{\text{hy}}$T) use ViT-style patch tokenization by default, but this is not the only option. Q-Former compression offers resolution-invariance; dual-codebook offers unified understanding + generation; physics-conditioned variants could embed physical semantics into the token representation.
2. **The iterative refinement architecture synthesizes diffusion and AR.** Within each timestep: iterative constraint-enforcement (diffusion-like). Across timesteps: autoregressive rollout. Progressive conservation law enforcement provides a principled ordering for the refinement schedule and bounds autoregressive error growth.
3. **Minimum context = ODE order.** Single-snapshot AR models work in Keplerian (distribution-fitting) mode, not Newtonian (causal) mode. True world-model physics reasoning from a single snapshot is impossible for second-order PDEs without derivative estimation (GP$_{\text{hy}}$T approach).
4. **Cosmos tokenizer** — NVIDIA's purpose-built video tokenizer (8× compression, 12× faster, trained on 20M hours of physical data) is a directly relevant external development; noted but not yet fully ingested as a primary source.

---

## [2026-06-26] note | PFM interface design: communication without text, timescale handling, regime specification

**Operation:** Synthesis concept page from user questions.  
**Page created:** `concepts/pfm-interface-design.md`  
**Trigger:** User raised three design concerns: (1) how PFM capabilities work without text communication, (2) how timesteps are handled across 30 orders of magnitude, (3) how the model handles different physical regimes without text prompts.

**Key content:**
- Tasks are computational patterns (forward, inverse, uncertainty) — no text instruction needed; model invoked differently by orchestration layer
- Nondimensionalization + $\hat{\Delta t}$ conditioning token unifies all timescales; derivative prediction (GP$_{\text{hy}}$T) makes output $\Delta t$-agnostic
- Minimal 5-input physical interface: field data, $\hat{\Delta t}$, field type codes (~100 code vocabulary), dimensionless governing parameters, boundary condition specification
- In-context inference handles most configuration automatically (governing parameters inferred from trajectory statistics)
- Scientific law discovery is NOT a native PFM capability — requires post-processing (symbolic regression, latent probing)
- Text capability as a separate frozen-backbone + projector + LLM layer (BLIP-2/LLaVA pattern), not jointly trained

**Correction noted:** `concepts/pfm-concept-overview.md` overstated "scientific law discovery" as a native PFM output — this requires a downstream symbolic regression pipeline.

**Total wiki pages:** 67  
**Running total sources ingested:** 33 (no new sources; synthesis page)

---

## [2026-06-29] reorg | Vault restructured into themed subfolders + GitHub repo wired in

**Operation:** User-directed reorganization of the prelim-research wiki for navigability. No content changed inside existing pages except wikilink format.

**Structural changes:**
- `summaries/` (flat, 33 files) → 8 themed subfolders: `foundation-models/` (5), `transformers-theory/` (7), `neural-operators/` (3), `graph-networks/` (6), `hamiltonian-lagrangian/` (4), `diffusion-generative/` (3), `solvers-and-simulation/` (2), `other-methods/` (3).
- `concepts/` (flat, 31 files) → 5 themed subfolders: `00-pfm-core/` (5), `physics-foundations/` (3), `ml-building-blocks/` (14), `benchmark-architectures/` (5), `extension-architectures/` (4).
- `pisd-physics-informed-spectral-diffusion` recategorized from transformers → `diffusion-generative/` (it is a diffusion model).

**Link migration:** All 960 wikilinks converted from path-prefixed (`[[summaries/x]]`, `[[concepts/x]]`) to **bare (`[[x]]`)** form. All filenames are unique, so bare links resolve regardless of folder — the vault is now reorganization-proof. Verified zero dangling links.

**New pages (2):**
- `00-start-here.md` — conceptual on-ramp with a 6-stage guided reading path and the Physics Encoding Spectrum framing.
- `resources/code-and-papers.md` — links the project GitHub repo (`nonidino/physics-foundation-model`) plus a per-paper table of arXiv + reference-implementation code links (some marked *(find)* pending verification).

**Files updated:** `index.md` (rebuilt to mirror folder structure, GitHub + start-here banners, counts 67 → 69), root `CLAUDE.md` (folder map, bare-link convention, ingest workflow now writes to themed subfolders + code-and-papers, `new/` marked do-not-move).

**GitHub:** Repo `github.com/nonidino/physics-foundation-model` established as the canonical code/build layer; wiki is the research/design layer. Framework implementation pages to be added later as the project moves from research → build.

**Total wiki pages:** 68 (33 summaries + 31 concepts + 00-start-here + index + log + code-and-papers)
**Running total sources ingested:** 33 (no new sources)

---

## [2026-06-29] ingest+note | Poseidon ingest + deepened FM summaries + new unified-token-representation folder

**Operation:** Three-part user-directed task — (1) in-depth ingest of the Poseidon source from `/new`; (2) substantial deepening of all 5 foundation-model summaries; (3) new `concepts/unified-token-representation/` folder.

**Source processed (1):**
- `new/Poseidon Efficient Foundation Models for PDEs.md` (arXiv:2405.19101v2, Herde, Raonić, Rohner, Käppeli, Molinaro, de Bézenac, Mishra — ETH Zurich; NeurIPS 2024). Moved to `/raw` after ingest.

**Summary created (1):**
- `summaries/foundation-models/poseidon-pde-foundation-model.md` — in-depth: operator-learning framing (OLT ≠ next-step prediction); scOT architecture (SwinV2 windowed/multiscale U-Net, patch embedding, scaled-cosine attention, relative-log position bias, ConvNeXt skips, lead-time-conditioned LayerNorm for continuous-in-time eval); **all2all semi-group training** ($O(K^2)$ pairs, 5.11M examples); split-LR + frozen-latent finetuning; PDEgym pretraining corpus (6 operators); 15 OOD downstream tasks; results (median ~20 samples = FNO@1024; best on 14/15; architecture>size; biphasic scaling law; vs. DPOT/MPP/CNO-FM); **three case studies** (CE-RPUI compositional reuse, ACE reaction-diffusion frozen-latent, Poisson-Gauss warmup→fast-learning); 3 AI inferences.

**FM summaries deepened (5):** `gphyt-physics-foundation-model`, `walrus-paper`, `walrus-overview`, `aion-1-astronomy`, `multimodal-transformers-survey` — each substantially expanded (architecture/training mechanics, deeper PFM analysis, Poseidon compare/contrast, links into the new token folder, additional AI inferences). Established the **four-paradigm framing** (operator / in-context-derivative / autoregressive / masked) across Poseidon/GP$_{\text{hy}}$T/Walrus/AION-1.

**New concept folder created (15 pages): `concepts/unified-token-representation/`**
- Hub: `00-token-representation-overview` (8-criterion scorecard; continuous-vs-discrete / local-vs-global / grid-tied-vs-resolution-free axes; folder map).
- Used by past FMs (11): `patch-embedding-tokens`, `spatiotemporal-tubelet-tokens`, `hierarchical-windowed-tokens`, `adaptive-compute-tokens`, `vector-quantized-tokens`, `scalar-cdf-tokens`, `spectral-fourier-tokens`, `branch-trunk-operator-tokens`, `graph-mesh-tokens`, `coordinate-implicit-tokens`, `learned-query-compression-tokens`.
- Proposed better alternatives (3): `physics-conditioned-query-tokens`, `wave-particle-dual-tokens`, `structure-preserving-tokens`.
- Each page: **intuition, mathematics, pros/cons**, used-by, relationships, AI inferences, cross-links.

**Concept pages updated:** `pfm-architecture-approaches` (rewrote Neural Operator Learning paradigm 5 — Poseidon broke the scale/cross-family-generalization objections; added all2all; See Also), `physics-foundation-models`, `neural-operators`, `transfer-learning-fine-tuning`, `autoregressive-rollout-stability`, `transformer-architectures`, `in-context-learning-physics`, `navier-stokes-equations`, `multimodal-tokenization` (linked the new folder as its physics-specific complement).

**Bookkeeping:** `index.md` — added Poseidon summary row, new Unified Token Representation concept section (15 pages), new cross-cutting themes (four paradigms, token representation), updated counts (68→84 pages, 33→34 sources). `code-and-papers.md` — added Poseidon row (arXiv 2405.19101 + camlab-ethz code/PDEgym); filled GP$_{\text{hy}}$T arXiv and Walrus code links.

**Total wiki pages:** 84 (34 summaries + 46 concepts + start-here + index + log + code-and-papers)
**Running total sources ingested:** 34

**Key themes from this batch:**
1. **Operator learning ≠ next-step prediction, and it scales.** Poseidon is the wiki's strongest feasibility proof for PFMs: a 6-operator fluid pretraining set generalizes to waves, reaction-diffusion, and elliptic PDEs. Mechanism (from case studies): **compositional reuse of physical primitives**, not equation-space interpolation — upgrading Walrus's "diversity-first" lesson from empirical to mechanistic.
2. **Continuous lead-time conditioning** is a clean answer to the timescale-specification problem ([[pfm-interface-design]]) — one model, any $t$ — complementary to GP$_{\text{hy}}$T's derivative-prediction route.
3. **Tokenization is the PFM bottleneck.** All strong PFMs use the weakest token corner (continuous/local/grid-tied patches) and engineer around it; the new folder maps the design space and argues the next capability jump may come from the tokenizer (operator-valued, structure-preserving, dual representations), not the backbone.
4. **all2all / semi-group training** is an architecture-agnostic data-amplification principle applicable to every time-dependent-PDE PFM in the wiki.

---

## [2026-06-30] note | Prereqs and learning resources page

**Operation:** New `misc/` folder + reference page.
**Page created:** `misc/prereqs-and-resources.md`
**Trigger:** User requested a structured learning roadmap for someone entering with ML + calculus + ODEs + basic mechanics/fluids.

**Content:** Tiered curriculum — Tier 1 (PDEs, Fourier/spectral methods, numerical methods for PDEs), Tier 2 (Lagrangian/Hamiltonian mechanics, functional analysis, deeper fluid mechanics), Tier 3 (statistical mechanics, electrodynamics) — each with specific textbooks, free resources, and priority topics. Includes project-specific resources (Brunton & Kutz, FNO paper, Poseidon paper) and a week-by-week suggested order.

**Index updated:** Added `misc/` section; page count 85 → 86.

**Total wiki pages:** 86
**Running total sources ingested:** 34

---

## [2026-06-30] note | Deep-dive concept page: the three tokenization trade-off axes

**Operation:** Synthesis concept page from user query.
**Page created:** `concepts/unified-token-representation/tokenization-tradeoff-axes.md`
**Trigger:** User asked for an in-depth explanation of the continuous/discrete, local/global, and grid-tied/resolution-free axes that organize the unified-token-representation folder.

**Key content:**
- **Continuous vs. Discrete** — nature of the token (real-valued vector vs. integer symbol); why physics is continuous but LLMs are discrete; where each excels (continuous: forward emulation, physics-informed training, rollout; discrete: shared vocabulary, masked modeling, any-to-any inference, generation); VQ/FSQ/LFQ/CDF on the discrete side; wave-particle dual as the resolution.
- **Local vs. Global** — what the token sees (bounded region vs. whole domain); maps directly onto PDE classification (elliptic = global, hyperbolic = local, parabolic = intermediate); local pros/cons (cheap, flexible, handles shocks, loses long-range elliptic coupling); global pros/cons (natively captures Poisson/Stokes/Coulomb coupling, resolution-free, restricted to nice geometries, Gibbs ringing); multiscale hierarchy as interpolation.
- **Grid-tied vs. Resolution-free** — formal distinction between learning a function (grid-tied) and learning an operator (resolution-free); aliasing mechanics and how it drives rollout error; CSM as middle ground; resolution-free options (spectral, DeepONet, Q-Former); why this axis = the operator-learning/emulator distinction.
- **Interaction table** — which axis combinations are coherent and which are contradictory.
- **Scorecard mapping** — which of the 8 criteria in the overview each axis controls.
- **Design roadmap** — 4-step incremental path from current (Continuous, Local, Grid-tied) to full (Continuous+Discrete, Local+Global, Resolution-free).

**Pages updated:**
- `concepts/unified-token-representation/00-token-representation-overview.md` — added "Deep dives" table row linking to new page.
- `index.md` — added row in Unified Token Representation section; page count 84 → 85.

**Total wiki pages:** 85
**Running total sources ingested:** 34 (no new sources; synthesis page)

---

## [2026-06-30] ingest+note | Initial PFM model design + two attention sources + two new concept folders

**Operation:** User-directed founding-design session for the *first concrete model*. (1) Ingested 2 attention sources from `/new`; (2) created `concepts/adjusted-attention-mechanism/`; (3) created `concepts/initial-pfm-model/` with the graph tokenizer, normalization decision, and full architecture + diagram. Source files left in `/new` (do-not-move inbox).

**Founding design commitments (from brainstorm):** broad multiphysics from the ground up (accuracy-with-scale hypothesis); **direct state prediction** (no classical integrator); optional output refinement (diffusion default / PINN when equation known); headline goals = **continuum–particle bridge** + **long-horizon stability via inbuilt mechanism**; **symmetric attention** as the one fixed mechanism choice.

**Sources processed (2):**
1. `new/GLM 5.2 Architecture Deep Dive...md` — Index Share sparse attention (group-shared local/global/landmark indices), multi-token prediction (k=4), KV-cache redesign (GQA + sparse eviction + INT8). Vendor blog (Zhipu AI GLM 5.2).
2. `new/Q-Former Architecture.md` — learned-query multimodal alignment (BLIP-2); interleaved self/cross-attention; HierarQ hierarchical local+global query streams + compressing memory; DisenQ disentangled query groups; LoRA/AdaLoRA PEFT.

**Summaries created (2):** `summaries/transformers-theory/glm-index-share-attention.md`, `summaries/transformers-theory/q-former-architecture.md`.

**New concept folder `adjusted-attention-mechanism/` (4):**
- `00-attention-overview.md` — hub; why standard attention is wrong for physics (asymmetric/attractive/dissipative); the **reciprocal ≠ conservative** distinction; catalog + composition.
- `symmetric-attention-physics.md` — the core mechanism; symmetric score = Newton's 3rd law; the key result that **symmetric attention is intrinsically diffusion (dissipative)** and must be paired with a skew-symmetric conservative channel for energy/long-horizon stability; attention = learnable graph Laplacian. Builds on (does not duplicate) [[antisymmetric-signed-attention-transformer]].
- `index-share-sparse-attention.md` — GLM-derived; local/global/landmark = hyperbolic/elliptic locality; mutual-membership condition to preserve symmetry; landmarks = multigrid coarse level.
- `hierarchical-query-attention.md` — Q-Former/HierarQ-derived; resolution-free compression as the per-node encoder; local+global query streams; compressing scene-memory for bounded long-rollout context.

**New concept folder `initial-pfm-model/` (4):**
- `00-initial-model-overview.md` — the 5 design commitments → components map; deferred forks flagged.
- `graph-tokenizer.md` — unified graph tokenizer: nodes = particles OR field patch-centroids carrying neighborhood latents; query-compression node encoder → resolution-free; relative-displacement + antisymmetric edges; supernode hierarchy for global coupling; coordinate-implicit decode. The continuum–particle bridge solved at the tokenizer. Mixes graph-mesh + learned-query + structure-preserving + coordinate-implicit + multipole.
- `normalization-scheme.md` — **LayerNorm decision**: no standard LayerNorm (mean-subtraction mixes channels/breaks equivariance; norm-division destroys energy). Use input nondimensionalization + grouped-RMS / equivariant gain + AdaLN-style conditional scale; no normalization inside conservative attention blocks.
- `initial-model-architecture.md` — **full end-to-end pipeline + mermaid diagram**; 8 stages; inbuilt conservation/stability table; non-text 5-input interface; training plan (diverse multiphysics, all2all amplification, push-forward + noise injection, single-system smoke test); 5 planned ablations; risks.

**Bookkeeping:** `index.md` — two new concept sections (Initial PFM Model ★★, Adjusted Attention Mechanisms); 2 summary rows; counts 86→96 pages, 34→36 sources. `code-and-papers.md` — added GLM + Q-Former rows (BLIP-2/HierarQ/DisenQ arXiv links, LAVIS code).

**Total wiki pages:** 96 (36 summaries + 55 concepts + 1 misc + start-here + index + log + code-and-papers)
**Running total sources ingested:** 36

**Key decisions recorded:**
1. **Reciprocal ≠ conservative.** Symmetric attention score restores Newton's 3rd law (conserves momentum) but is mathematically graph-diffusion (dissipative in energy). Long-horizon stability requires *also* a skew-symmetric/A-DGN conservative channel. This is the load-bearing insight of the attention design and the #1 planned ablation.
2. **Continuum–particle bridge is a tokenizer problem.** If particle nodes and field-centroid nodes produce the same kind of neighborhood-latent, one symmetric-attention backbone evolves both with no architectural fork.
3. **Put physics in the representation + interaction geometry, not the loss.** Conservation is architectural (antisymmetric edges + reciprocal aggregation + skew-symmetric channel + structure-preserving norm), so stability should be an emergent property at any scale.
4. **Deferred forks (need evidence):** hard projection head vs. trained-in stability; MTP horizon k; refinement default; direct-state vs derivative cost.

---

## [2026-06-30] note | Initial-model follow-ups: diagram fix, conditioning, skew-symmetric/intelligent-patching/MHA pages

**Operation:** User-directed refinements to the initial-model design from the same session.

**Fixes & edits to `initial-model-architecture.md`:**
- **Mermaid diagram fix** — removed numbered prefixes (`1.`, `2.`, …) and `•` bullets from node labels; these triggered Obsidian's "Unsupported markdown: List" render error. Relabeled as "Step N" plain text, ASCII-safe.
- **Added "How physical quantities (Reynolds, Mach, …) enter the model"** — answers how dimensionless numbers are included: produced by nondimensionalization, log-embedded, injected three ways (additive conditioning tokens, AdaLN conditional gain, query conditioning), and optionally left as an "unknown" mask token inferred in-context.
- **Added "Do we need sparse attention?"** — no for the initial model: the radius-graph tokenizer already provides locality-sparsity, and smoke-test token counts favor dense symmetric attention; Index Share is a deferred scaling option that largely coincides with extending the graph's neighbor structure.

**Concept pages created (3):**
- `adjusted-attention-mechanism/skew-symmetric-attention.md` — the conservative channel as a first-class building block: intuition (rotation vs. stretching), math (skew-symmetric spectrum, orthogonal norm-preserving flow, symplectic/Hamiltonian link, A-DGN parameterization, Cayley/exp to orthogonal), the two-distinct-(anti)symmetries-in-attention table, pros/cons, and a **survey of models** — established skew-symmetric RNN/GNN/ODE work (AntisymmetricRNN, A-DGN, LEM, orthogonal/unitary RNNs, HNN/Symplectic-ODE-Net, RevNet/Reformer) vs. the near-art for *attention* specifically (Sinkformers, modern Hopfield/Energy Transformer) — concluding that skew-symmetric attention per se is essentially unbuilt (consistent with [[antisymmetric-signed-attention-transformer]]).
- `initial-pfm-model/intelligent-patching.md` — adaptive-resolution tokenization (AMR for tokens): refinement indicator ($|\nabla u|$, vorticity, spectral energy, or learned uncertainty), quadtree/octree subdivision, token-budget control; composes with the graph tokenizer (variable patch size → variable node density; resolution-free encoder absorbs variable point counts; **refinement tree = supernode hierarchy = Index Share landmarks**); pros/cons; uncertainty-driven closed-loop patching AI inference.
- `initial-pfm-model/multihead-attention.md` — is MHA needed (yes: single head = one Laplacian; multiphysics needs simultaneous elliptic+hyperbolic+parabolic coupling), the conservation problem with dense $W_O$, and the proposed **physically-structured MHA** (per-head symmetric+skew-symmetric; combine by summation or orthogonal $W_O$; heads assigned by field-type × scale × interaction-type; field-group heads share index sets = Index Share); heads as a learned basis of physical operators.

**Bookkeeping:** `index.md` — added 3 rows (skew-symmetric in Adjusted Attention; intelligent-patching + multihead in Initial PFM Model); counts 96→99 pages (sources unchanged at 36).

**Total wiki pages:** 99 (36 summaries + 58 concepts + 1 misc + start-here + index + log + code-and-papers)
**Running total sources ingested:** 36

**Key answers recorded:**
1. **Reynolds/Mach inclusion** = conditioning, not input channels: nondimensionalization generates them → log-embed → inject as additive tokens + conditional gain + query conditioning; unknown ⇒ infer in-context.
2. **Sparse attention** is *not needed yet*; graph radius-tokenizer supplies locality, and sparse attention later coincides with extending neighbor structure.
3. **Skew-symmetric attention** is an open direction — the building block is mature in RNNs/GNNs; inside attention it is essentially unbuilt (nearest: Sinkformers, Hopfield/Energy Transformers).
4. **Multi-head attention** is needed but should be physically structured (heads = physical coupling operators), combined conservatively (summation / orthogonal $W_O$).

---

## [2026-06-30] note | Clarified graph construction vs. interaction learning in the tokenizer

**Operation:** Answered a user query and filed it into `graph-tokenizer.md`.
**Trigger:** "How will the graph tokenizer generate a graph? Doesn't it need to be trained to recognize interactions (gravity, EM)?"

**New section added to `graph-tokenizer.md`:** "How is the graph built — and does it have to learn the interactions?" Key points: (1) **topology** is geometric/not-learned (radius graph / kNN), **features** are learned (query encoder), **force law** is learned by the *backbone* and inferred in-context — the tokenizer never "recognizes gravity"; (2) edges carry only relative geometry, so message passing learns the law from trajectories (equation-agnostic, Newtonian-world-model mechanism); (3) the real subtlety is **reachability vs. range** — radius graphs miss long-range forces (gravity/Coulomb/elliptic), fixed by the multiscale supernode hierarchy = neural FMM/Barnes–Hut; (4) many-body effects via deep message passing or angular/triplet terms (no hyperedges needed); (5) optional learned topology (Neural Relational Inference) for non-spatial interaction structure. Principle: *tokenizer decides reachability; backbone learns the force.*

No new pages; page count unchanged (99).

---

## [2026-06-30] note | Critical self-audit filed: open architectural problems

**Operation:** User asked for a critical audit of the initial-model design ("aside from training data, what are we missing?"). Filed the evaluation as a new page.

**Page created:** `initial-pfm-model/open-architectural-problems.md` — 11 gaps in two tiers.

**Tier 1 (threatens core claims):**
1. **Depth-conservation ≠ time-conservation** — A-DGN/skew-symmetric properties hold across the L layers of one forward pass, not across the autoregressive rollout; direct-state choice (commitment #2) severed the integrator link that Dynami-CAL/HNN use to make antisymmetry→physical conservation, so commitment #4 (inbuilt stability) is aspirational. *The crux; causally upstream of #2 and #5.*
2. **Hard constraints conditioned, not enforced** — incompressibility (pressure projection) and BCs are not satisfied by conditioning codes; blocking for the 2D NS smoke test.
3. **Tokenizer is an untrained lossy autoencoder** — latent adequacy and separate-vs-joint training unspecified; loss compounds over rollout.
4. **Continuum–particle bridge solved only for homogeneous case** — coupled heterogeneous-node systems (FSI, suspensions) — the actual hard part — unsolved; overselling.
5. **Deterministic core vs. chaotic physics** — diffusion may be load-bearing, not optional, for turbulence.

**Tier 2 (normal eng. gaps):** loss design (per-field + spectral); dynamic graph topology over rollout (edge churn vs. momentum conservation); variable field cardinality per node; equivariance commit-or-not; backbone training stability (ε drift, no LayerNorm, init/optimizer); no sizing/compute/eval/inference-cost.

**Unifying theme:** the deferred **enforce-vs-learn** decision underlies all of Tier 1. Recorded recommendation: settle which invariants/constraints are hard (projected/integrated) vs. soft (penalized/corrected) — collapses most of Tier 1. Also recorded: the **tokenizer, not the backbone, is the least-complete and riskiest component** (Problems 3+4) and should be prototyped/stress-tested first (encode→decode fidelity on turbulent data).

**Bookkeeping:** `index.md` — added row to Initial PFM Model section; count 99→100 pages (sources unchanged at 36).

**Total wiki pages:** 100 (36 summaries + 59 concepts + 1 misc + start-here + index + log + code-and-papers)
**Running total sources ingested:** 36

---

## [2026-06-30] note | Scored candidate solutions added to open-architectural-problems (Problems 1, 3, 4, 5)

**Operation:** Added a `#### Candidate solutions (scored /10)` block under Problems 1, 3, 4, and 5 in [[open-architectural-problems]], in response to a design query on Problems 1/3/4/5.
**Scoring rubric:** leverage × feasibility × evidence, 1–10, for the *initial* model.
**Solutions filed:**
- **P1 (depth vs. time conservation):** S1.1 projection-head-at-top-supernode (8) · S1.2 hybrid symplectic subspace (7) · S1.3 learned invariants (6) · S1.4 soft penalty + corrector (5) · S1.5 push-forward training (5).
- **P3 (lossy tokenizer):** reframed to "lossless above cutoff"; S3.1 dual-path residual (8) · S3.2 spectral/wavelet basis (7) · S3.3 capacity matching (7) · S3.4 reconstruction fidelity floor (8) · S3.5 invertible tokenizer (4); non-spectrally-biased decoder as required complement.
- **P4 (continuum–particle coupling):** split interface vs. coupling; S4.1 typed-edge encoder (8) · S4.2 immersed-boundary coupling (7) · S4.3 shared encoder + committed type code (8) · S4.4 concrete coupled benchmark (6).
- **P5 (determinism vs. chaos):** S5.1 LES-split deterministic core + guided latent diffusion on residual (8) · S5.2 regime routing (7) · S5.3 full generative core (6) · S5.4 probabilistic head (6) · S5.5 spectral loss only (4).
**Cross-cutting:** added an [AI Inference] noting P1/P3/P5 compose into one enforce/regress/generate three-tier scheme scaffolded by the supernode hierarchy.
**Bookkeeping:** no new pages; edits confined to [[open-architectural-problems]] and this log. Page/source counts unchanged (100 / 36).

---

## [2026-06-30] note | Threaded S1.1/S3.1/S4.1/S5.1 decisions through initial-pfm-model + second-pass audit

**Operation:** Adopted four candidate solutions as design decisions and propagated them across all initial-pfm-model pages, then re-audited.
**Decisions adopted:** S1.1 (mandatory conservation projection on final latent; tiered known-hard / discovered-soft invariants, folding in S1.3) · S3.1 (dual-path smooth + high-frequency residual encoding) · S4.1 (typed-edge encoder for heterogeneous coupling) · S5.1 (LES scale split: deterministic resolved core + physics-guided latent diffusion generating the sub-grid residual).
**Pages adjusted:**
- [[00-initial-model-overview]] — commitments #3 (diffusion core), #4 (typed edges + mandatory projection); components table; "left open" section marked resolved/partially-resolved.
- [[graph-tokenizer]] — new Typed-edge and Dual-path subsections; open problems #3/#4 annotated with adopted solutions.
- [[initial-model-architecture]] — Stage 2–3 (typed edges + dual-path), new Stage 5b (projection), Stage 7 (diffusion core), conservation table (projection mandatory), ASCII pipeline, risks list.
- [[normalization-scheme]] — projection-after-gain ordering constraint (magnitude = energy).
- [[intelligent-patching]] — dual-path complement note.
- [[multihead-attention]] — node-type-coupling heads (S4.1).
- [[open-architectural-problems]] — **additive only** (nothing removed): decision banners under Problems 1/3/4/5, S1.1 refined with per-block/latent + tiered-invariant clarifications, and a new **Second-pass audit** section.
**Audit verdict:** the four picks are mechanisms, not closures — each ~half-covers its problem, lacks its validation complement (S3.4/S4.4/S5.2/S1.5), and adds interface holes. Problem 2 (local constraint enforcement) is now the #1 smoke-test blocker; Problems 6/7/10/11 aggravated; 8/9 untouched. Risk has shifted from components to interfaces → next step is a specified pipeline contract.
**Bookkeeping:** no new pages; edits confined to the seven initial-pfm-model pages + this log. Counts unchanged (100 / 36).

---

## [2026-06-30] note | Adopted validation complements + filed Problem 2 projection solutions

**Operation:** Adopted the four validation complements and filed a scored solution set for Problem 2 (constraint enforcement), driven by two design questions (can BCs be forced inside the projection? how do projections work?).
**Complements adopted:** S3.4 (tokenizer reconstruction fidelity floor, gating pretraining) · S3.3 (capacity matching) · S4.4 (settling-particles coupled benchmark) · S5.2 (in-context regime routing for the diffusion core) · S1.5 (push-forward with a $1\to2\to4\to8$ curriculum).
**Problem 2 solutions filed (scored):** P2.1 Leray/Helmholtz div-free projection via Poisson solve on the supernode hierarchy (8) · P2.2 joint KKT projection folding conservation + Dirichlet BCs + divergence into one physical-space solve (8) · P2.3 ghost-node/immersed-boundary BC handling (7) · P2.4 divergence-free-by-construction stream/vector potential (7). Includes shared "how a projection works" machinery (linear closed-form; elliptic Leray/Poisson; quadratic Lagrange step).
**Key refinement:** global-scalar conservation projects in *latent* space (S1.1 readout); div-free + BCs are local/elliptic → project in *physical* space post-decode as one joint KKT solve. Refined pipeline order recorded in [[initial-model-architecture]] Stage 5b.
**Pages adjusted:** [[open-architectural-problems]] (Problem 2 solution set; four banners updated to complements-adopted; audit §C Problem 2 and §D rewritten) · [[initial-model-architecture]] (training plan: fidelity gate, multi-term loss, push-forward curriculum, coupled benchmark; Stage 7 routing; Stage 5b physical-space constraint projection) · [[00-initial-model-overview]] (left-open section) · [[graph-tokenizer]] (open problems #3/#4).
**Status shift:** Problem 2 downgraded from #1 blocker to engineering (elliptic-solve implementation + compatibility bookkeeping); genuinely-open now = Problem 4 cross-type message form, 6 (loss balance), 9 (equivariance), 10 (training stability), 11 (inference cost).
**Bookkeeping:** no new pages; counts unchanged (100 / 36).

---

## [2026-06-30] note | New page: The Pipeline Contract (initial model)

**Operation:** Created [[pipeline-contract]] — the binding interface spec the second-pass audit called for (risk moved from components to interfaces).
**Contents:** objects/notation table; G0 fidelity gate + 12-stage per-step sequence with per-stage space and exit invariants; mermaid + ASCII flow; the **latent-vs-physical space map** (two authoritative constraint sites: latent Stage-7 global-scalar pre-projection, physical Stage-10 joint div-free∧BC∧conservation projection); the **three-tier message bus** (enforce/regress/generate via hierarchy restriction/prolongation); **8 load-bearing ordering invariants** each with its silent-failure symptom; the autoregressive contract (deterministic-encode/stochastic-predict resolution; S1.5 curriculum); three regime reductions (particle-only, incompressible, coupled); open interface risks.
**Key formalization:** a constraint may be projected in latent space *iff* it is (near-)linear through the decoder — separates global integrals (latent) from pointwise differential constraints (physical). Model framed as a "projection sandwich": learn freely in the interior, enforce exactly at each step's boundary. Load-bearing primitive = the single learned restriction/prolongation ($R/P$) pair (tokenization + global coupling + Poisson solve + tier bus).
**Bookkeeping:** [[index.md]] — added Pipeline-contract row to Initial PFM Model section; count 100→101 pages (60 concepts). Sources unchanged at 36.

**Total wiki pages:** 101 (36 summaries + 60 concepts + 1 misc + start-here + index + log + code-and-papers)
**Running total sources ingested:** 36

---

## [2026-07-01] note | Problem 6 (loss) resolved + Problem 12 (FFN) discovered and filed; Problem 9/10 provisional defaults

**Operation:** Solved [[open-architectural-problems]] Problem 6 (loss function) and a newly-surfaced Problem 12 (no per-node nonlinear/FFN block), triggered by a design question about whether the backbone was "just shuffling tokens" without one. Also filed provisional (not fully scored) defaults for Problems 9 (equivariance) and 10 (training stability) to unblock parameter sizing (Problem 11), per user request to clear remaining blockers before sizing.
**Problem 12 (new) — solutions filed (scored):** F1 semi-orthogonal norm-bounded expand/contract FFN sub-step (8, **adopted**) · F2 equivariant gated nonlinearity, upgrade path if Problem 9→hard-equivariance (7) · F3 no-FFN baseline, mandatory ablation (4) · F4 unconstrained dense FFN, survivable only via the projection-sandwich safety net for conservation, not equivariance (3). Diagnosis: attention moves information between nodes (communication); an FFN computes new nonlinear per-node features (computation); the backbone as specified had only a single elementwise activation, risking rank collapse and under-fitting local constitutive nonlinearities.
**Problem 6 — solutions filed (scored):** S6.1 homoscedastic per-channel uncertainty weighting (8, adopted) · S6.2 GradNorm cross-group balancing (7, held in reserve) · S6.3 staged curriculum across 4 loss groups (8, adopted) · S6.4 SFA-style slowness+discriminative loss for tier-2 invariant discovery (7, adopted, [AI Inference] novel synthesis) · S6.5 phase-separated backbone-then-diffusion training (8, adopted) · S6.6 fixed hand-tuned weights (4, rejected as sole mechanism). Key clarification: tier-1 known invariants need no loss term (enforced by the S1.1 projection); only tier-2 discovered invariants need Group C's loss.
**New page:** [[training-curriculum]] — the training-time companion to [[pipeline-contract]]; 4 loss groups (core fit / temporal / conservation-discovery / generative), within-group uncertainty-weighting formula, 5-phase curriculum table, SFA discovery-loss derivation, phase-separated diffusion rationale.
**Problem 9/10 provisional defaults:** Problem 9 → no hard equivariance for the initial model, data augmentation as a soft proxy, reversible; Problem 12's F2 as the piecemeal upgrade path. Problem 10 → bound $\|J\|_2$ analytically (Cayley/Householder or spectral norm on the skew-symmetric term and the F1 FFN), compute $\epsilon<2/\|J\|_2$ per sub-step rather than tuning empirically. Both flagged as provisional, not fully scored passes.
**Pages adjusted:** [[open-architectural-problems]] (new Problem 12 full entry with F1–F4; Problem 6 candidate-solutions block; Problem 9/10/11 provisional-default notes; Priority-ordering addendum; second-pass audit §E; See Also/Related Concepts expanded — additive only, nothing removed) · [[initial-model-architecture]] (Stage 4 backbone equation gains the FFN sub-step; Training plan §3 rewritten to reference the resolved curriculum; Risks list; See Also) · [[multihead-attention]] (new section scoping heads vs. the head-agnostic FFN sub-step) · [[pipeline-contract]] (Stage 4 row annotated with the FFN sub-step; Related Concepts).
**Status:** parameter sizing (Problem 11) is now the concrete next step, unblocked by Problem 9/10/12 defaults. Still fully open: Problem 4's cross-type message form, Problems 7–8, and full scored passes on 9/10 if more rigor is wanted first.
**Bookkeeping:** [[index.md]] — added Training-curriculum row; count 101→102 pages (61 concepts). Sources unchanged at 36.

**Total wiki pages:** 102 (36 summaries + 61 concepts + 1 misc + start-here + index + log + code-and-papers)
**Running total sources ingested:** 36

---

## [2026-07-01] note | Noether 1.0 — named, sized model (resolves Problem 11)

**Operation:** Resolved [[open-architectural-problems]] Problem 11 (sizing, compute, eval protocol) by deriving a full engineering spec for the concrete model, named **Noether 1.0**, grounded in [[possible-architectures]]'s existing 100K→1M→10M benchmark sweep rather than inventing new scale targets.
**New page:** [[noether-1.0]] — nano/base/large configs (≈118K / 847K / 8.1M params); backbone parameter derivation $12Ld^2$ (self-term $d^2$ + $H$-head interaction $3d^2$, $H$-invariant + F1 FFN sub-step $8d^2$ at 4× expansion) — noted as identical to a vanilla transformer block's per-layer cost, meaning the physics constraints (symmetric score, skew-symmetric channel, summed heads, semi-orthogonal FFN) cost no extra parameters, only restricted expressivity per parameter; full non-backbone budget (dual-path encoder, typed-edge encoder, conditioning MLPs, conserved-readout, SIREN decoder) worked for the base config; optional diffusion add-on (~103K) scoped as implement-but-not-load-bearing at Burgers scale; 1D-Burgers-specific graph/hierarchy hyperparameters (patch size, radius, 3-level supernode hierarchy, MTP horizon); backbone hyperparameters ($\epsilon$/$\gamma$ schedule instantiating Problem 10's analytic-bound default, Stiefel QR-retraction scheme for F1); optimizer/curriculum schedule tied to [[training-curriculum]]'s phases; full eval protocol (accuracy, conservation on/off ablation, F1-vs-F3 rank-collapse diagnostic, inviscid-Burgers caveat for the symmetric-only ablation, deferred-to-2D-NS list for Problem 9/P2.2/S4.1 ablations).
**Key finding (§7, honest not falsely-precise):** at nano/base scale, inference wall-clock is dominated by kernel-launch/graph-rebuild overhead, not FLOPs — "faster than classical solvers" is not testable at this scale and the 2D NS follow-on benchmark should be chosen at a resolution where classical solve cost is already the bottleneck, so the comparison is meaningful the first time it's run.
**Pages adjusted:** [[00-initial-model-overview]] (Status line + Related Concepts name the model; folder intro links to [[noether-1.0]]) · [[open-architectural-problems]] (Problem 11 marked resolved, additive note, nothing removed).
**Status:** all four Tier-1 problems (1,2,3,5) plus Problems 6, 11, 12 are now resolved/decided; Problems 9 and 10 have provisional (not fully scored) defaults instantiated concretely in [[noether-1.0]] §4. Remaining fully open: Problem 4 (cross-type message form), 7 (dynamic topology cost), 8 (variable field cardinality) — none block implementing and training Noether-1.0-base on the 1D Burgers smoke test.
**Bookkeeping:** [[index.md]] — added ★ Noether 1.0 row; count 102→103 pages (62 concepts). Sources unchanged at 36.

**Total wiki pages:** 103 (36 summaries + 62 concepts + 1 misc + start-here + index + log + code-and-papers)
**Running total sources ingested:** 36

---

## [2026-07-01] note | Smoke-test bring-up plan; two corrections to Noether 1.0

**Operation:** Answered three implementation-planning questions — first training batch, corpus expansion, and implementation-readiness gap — by creating [[smoke-test-bringup]] as a companion to [[noether-1.0]]. Working out the answers surfaced two genuine errors in [[noether-1.0]], corrected in place.
**Corrections to [[noether-1.0]] (§2.2a, §2.2b):**
- **Conserved quantity was wrong.** The original spec listed $k\approx3$ ("mass-, momentum-, energy-analogs") for the S1.1 tier-1 projection, by loose analogy with Navier–Stokes. Worked out exactly for periodic 1D Burgers: only $\int u\,dx$ is truly conserved (any $\nu$, all time, even post-shock — it's the integral form of the conservation law itself). "Momentum" isn't separate from this (Burgers is scalar, no density/velocity split). **Energy $\int u^2dx$ is *not* conserved for $\nu>0$** — exact identity $\frac{d}{dt}\int\frac12u^2dx=-\nu\int u_x^2dx$ — and even at $\nu=0$ only pre-shock (entropy solutions dissipate energy at shocks). Hard-projecting energy would have enforced incorrect physics. Corrected to $k=1$; parameter budget and eval-protocol table both updated; the exact decay identity is kept as a diagnostic to check the backbone's learned $\gamma$ against.
- **Dual-path decode combination was unspecified.** Resolved: two small decoder heads (smooth + residual), summed additively in physical space, so the S3.1 residual channel is guaranteed to only ever *add* a correction, never overwrite the deterministic decode — also the clean interface for S5.1 diffusion later.
- **Parameter counts revised** (decoder now two heads): base config 847K→882K; nano 118K→122K; large 8.1M→8.2M; diffusion add-on total 950K→985K; FLOPs estimate updated.
**New page:** [[smoke-test-bringup]] — §0 scopes "Noether-1.0-core" (which components are active/inactive for 1D Burgers specifically: typed edges, P2.1/P2.2, MTP, Group C, diffusion+router all deferred by design, not by unresolved ambiguity); §1 Batch 0 (8-trajectory fixed-batch overfit test, expressivity/pipeline sanity, not a generalization test); §2 Batch 1 (G0 tokenizer-fidelity pretraining on mixed smooth/shock snapshots); §3 first dynamics batch; §4 a 5-level corpus ladder (8 → ~150 → ~1.5K → ~3.5K → 10K+ trajectories, each level isolating one new axis of variation for fault localization); §5 implementation-readiness checklist (resolved/deferred-by-design/pure-engineering/genuinely-open).
**Net finding:** for Noether-1.0-core on 1D Burgers, nothing remains open at the design level — every active component is fully specified; what's left is code plus running Batch 0.
**Pages adjusted:** [[noether-1.0]] (§2.2a/§2.2b new subsections; parameter tables and eval-protocol table corrected throughout; Related Concepts + See Also link to the new page).
**Bookkeeping:** [[index.md]] — added Smoke-test-bringup row; count 103→104 pages (63 concepts). Sources unchanged at 36.

**Total wiki pages:** 104 (36 summaries + 63 concepts + 1 misc + start-here + index + log + code-and-papers)
**Running total sources ingested:** 36

---

## [2026-07-01] note | Tier 3: 12 new problems (13–24) for general 2D/3D geometry, scored and decided

**Operation:** Filed the general-2D/3D-geometry critical audit (previously given as an unfiled analysis) as a new **Tier 3** section in [[open-architectural-problems]], additive only — nothing removed from the existing Tier 1/2 content. Every problem scored in the same `/10` format as Problems 1–6 and 12, each with a decision, following the user's explicit requirement that the same architecture stay viable end-to-end (evaluated as "extend vs. fork" for every candidate solution).
**Problems 13–24 filed (scored, decisions in parentheses):**
- **P13** unstructured patch/mesh definition — S13.1 generic clustering (6) · S13.2 native mesh (8) · S13.3 learned soft clustering (6) · **S13.4 extend [[intelligent-patching]]'s octree universally (9, adopted)**, composed with S13.2 when a mesh exists.
- **P14** curved/multi-type boundary enforcement — S14.1 SDF + analytic normals (8, adopted default) · S14.2 mesh-native faces (8, preferred when mesh exists) · **S14.3 generalize P2.2 to multi-region row-blocks (9, mandatory)** · S14.4 point-cloud plane-fitting fallback (6).
- **P15** equivariance default's 1D-specific justification doesn't survive 2D/3D — S15.1 commit to full equivariance now (6) · **S15.2 explicit scored ablation before deciding (8, adopted first)** · S15.3 selective/channel-typed equivariance (8, likely destination) · S15.4 soft equivariance loss (5).
- **P16** no irrep-typing mechanism — **S16.1 extend existing field-type codes with an irrep tag (9, adopted)** · S16.2 learned-then-committed classifier, mirroring S4.3 (7, fallback).
- **P17** 3D-specific representation gaps (pseudovector vorticity, rank-2 tensors, cost) — S17.1 extend taxonomy with parity/rank (8) + S17.2 bound $L_{\max}$ (8), both adopted together.
- **P18** elliptic solve (P2.1) sized only for the trivial case — S18.1 agglomeration multigrid on the existing hierarchy (8) + S18.2 Neumann BC from Problem 14's region labels (8), both adopted together.
- **P19** coordinate encoding for curved geometry — **S19.1 additive SDF input to the decoder (8, adopted)** vs. S19.2 full curvilinear fork (6, rejected as a fork).
- **P20** multi-scale nondimensionalization — **S20.1 per-supernode-region scales (8, adopted)** vs. S20.2 rely on in-context inference (5).
- **P21** sparse attention now mandatory at 3D scale, never validated — **S21.1 scheduled milestone, no new design needed (8, adopted)** + S21.2 synthetic large-$N$ pre-validation (6, parallel insurance).
- **P22** moving/deforming boundaries (ALE/FSI), a genuinely new regime — S22.1 ALE atop Problem 7's machinery (7) vs. **S22.2 explicit scope-out for now (7, adopted)**.
- **P23** mesh-native edges vs. radius graph — **S23.1 prefer mesh when available (8, adopted)** vs. S23.2 always radius graph (4).
- **P24** rotating reference frames, missing conditioning slot — **S24.1 add $\Omega$ as a 6th conditioning input (8, adopted)** vs. S24.2 hard-code Coriolis terms (4, rejected — breaks the project's equation-agnostic philosophy).
**Key synthesis (`[AI Inference]`):** nearly every adopted solution reuses or extends machinery already committed to elsewhere (the octree, P2.2, the conditioning interface, the supernode hierarchy, S4.3's known-vs-inferred-then-committed pattern, the mesh-vs-generic fallback decided three separate times) rather than introducing new components. The **one genuinely new primitive across all 12 problems is the SDF boundary representation** (S14.1) — reused three more times elsewhere (S13.4, S18.2, S19.1). This is the concrete evidence that the architecture stays viable end-to-end by extension, not by forking, directly answering the user's stated requirement.
**Pages adjusted:** [[open-architectural-problems]] only — new Tier 3 section (12 problems, ~30 scored solutions) inserted between Problem 11 and "What is actually solid"; Priority-ordering addendum 2 (Tier 3 internal ordering); See Also and Related Concepts expanded. Purely additive — no existing content removed or altered.
**Bookkeeping:** no new pages; page/concept counts unchanged (104 / 63). Sources unchanged at 36.

---

## [2026-07-01] note | Noether 1.0 rewritten to implementation depth; quick training set decided

**Operation:** Rewrote [[noether-1.0]] from a sizing/hyperparameter spec into a full implementation-depth architecture document (every tensor shape, module, and init recipe needed to write code), and converted [[smoke-test-bringup]]'s corpus ladder Level 2 from an option into a fully-specified decision, per the user's request to move toward actual implementation.
**Noether 1.0 rewrite — new content:**
- **§2, full end-to-end forward-pass pseudocode** for one Noether-1.0-base training step on 1D Burgers, shape-annotated at every [[pipeline-contract]] stage.
- **§3, module specifications** for all 8 components, including a resolved ambiguity: [[graph-tokenizer]]'s node latent was specified as $\mathbb R^{N_q\times D}$ (a *set* of query outputs), never reconciled with the single-$d$-vector backbone used everywhere else in this folder. Resolved by setting each query slot's width to $d_{\text{head}}=d/N_q$ and concatenating — no pooling layer needed, and **materially cheaper** than an implicit full-width-per-query assumption.
- **Two more corrected numbers** (in addition to the prior session's $k{=}1$/additive-decode fixes): (a) the conditioning MLP's input dimension was 3 (copied from the general multiphysics interface) — corrected to 1, since 1D Burgers has exactly one physical parameter ($\nu$); (b) the token count used in the parameter/FLOPs discussion was 64 (32 fine nodes × 2 timesteps), omitting the 9 hierarchy nodes (8 mid + 1 top) that also exist per timestep — corrected to 82.
- **§3.7, exact closed-form mass projection** derived in full: for $k{=}1$ (periodic Burgers), the general rank-$k$ projection formula collapses to a single uniform DC-offset subtraction, and a clarification that the projection should use the *directly computed* mass (cheap, exact for Burgers) rather than trusting the learned readout $\hat C(h)$, which stays only as a training diagnostic.
- **§5, initialization scheme** — concrete recipes for every constrained component (skew-symmetric init scale tied to the §6.1 stability bound, Stiefel QR-init + periodic re-orthogonalization, SIREN's standard init for the decoder, standard choices for everything else).
- **Revised parameter totals** from the encoder correction: nano 122K→126K, base 882K→874K, large 8.2M→8.17M; diffusion add-on 985K→977K total; FLOPs estimate updated for the corrected 82-token count.
**Smoke-test bring-up — training set decision:** Level 2 of the corpus ladder is now the **decided quick training set**: 1,500 trajectories (1,200/300 train/val), $\nu$ log-uniform over $[10^{-3},10^{-1}]$, up to 3-mode sinusoidal ICs, spectral-derivative RK45 solve, single-step Group-A training (2-term uncertainty weighting, only channel-trivial since Burgers has one field), S1.1 projection always on, success defined as beating the persistence baseline with monotonically decreasing validation error (not the full Level-4 rollout target, which remains a stretch goal). All internal `[[noether-1.0]]` section cross-references updated to match the new numbering.
**Pages adjusted:** [[noether-1.0]] (full rewrite) · [[smoke-test-bringup]] (§4.1 new decision subsection; cross-reference fixes throughout).
**Bookkeeping:** no new pages; counts unchanged (104 / 63). Sources unchanged at 36.

---

## [2026-07-01] note | Adaptive graph rebuild enabled for Burgers day-1; a residual-head bug caught and fixed

**Operation:** In response to a design question ("isn't the graph autoregressively rebuilt each call, allowing it to change?"), confirmed the general [[pipeline-contract]] architecture does exactly this (Stage 2 sits inside the autoregressive loop), clarified that [[noether-1.0]]'s prior "static graph" was a Burgers-specific optimization (fixed-geometry graph-build is a deterministic function of unchanging sample points, so caching is equivalent to recomputing, not an architecture fork), and — per the user's explicit choice — **enabled adaptive patching now** rather than keeping it deferred, specifically to get an early, cheap empirical test of [[open-architectural-problems]] Problem 7 (momentum/mass conservation under topology churn) in the simplest possible setting before the harder 2D/3D case.
**Bug caught and fixed while working this through:** the prior draft's Stage 5 "residual head" produced one flat scalar per *patch* (32 values) silently indexed as if it lined up 1:1 with the 128 *grid points* — it doesn't (a patch spans multiple, now variable-width, points). Fixed by removing the separate head entirely: the coordinate-implicit decoder directly outputs the continuous residual field at any queried point, additive across the dual path — matching what [[initial-model-architecture]] Stage 5 actually specifies, and eliminating a component that was quietly double-counting the decoder.
**Adaptive graph design (new [[noether-1.0]] §3.2):** monitor function $\eta=|\partial u/\partial x|$ (parameter-free shock detector); 1D recursive bisection (this benchmark's instance of [[intelligent-patching]]'s quadtree/octree) — always exactly 31 splits from 1 root to reach the 32-leaf token budget; hierarchy = the bisection tree's ancestor structure (exactly 31 internal nodes for 32 leaves, the standard full-binary-tree identity), not a fixed "group by 4" rule; restriction/prolongation stay parameter-free. Total nodes/snapshot: 63 (up from the static case's 41); tokens (2-step context): 126 (up from 82).
**Conservation fix this forces (new [[noether-1.0]] §3.7 content):** with variable-width patches, an unweighted average over *nodes* is wrong for a mass/momentum check (a wide and narrow patch shouldn't count equally) — the general S1.1 projection is moved to run strictly *after* decode, anchored to the fixed 128-point canonical grid, never to the current adaptive partition. Stated as a generalizable pattern: any global-integral constraint should anchor to a fixed reference measure, not the current adaptive tokenization.
**Argument for why this resolves Problem 7's adaptive-patching sub-case (not the harder persistent-identity sub-case):** the graph is ephemeral (rebuilt fresh from canonical state every step, never matched/tracked across steps) and the conservation check is anchored to a fixed measure — together, nothing is left for topology churn to corrupt. This is stated as a checkable claim, not just an argument: a new eval-protocol row (adaptive graph vs. an ablation forcing the static graph) directly tests it, and Batch 0's success criteria now include verifying mass conservation holds even as the graph changes shape step to step.
**Pages adjusted:** [[noether-1.0]] (§2 pseudocode Stage 2/5/6/5b-10 rewritten; §3.2 fully rewritten; §3.6 rewritten as a bug-fix note; §3.7 extended with the canonical-anchoring fix and Problem-7 argument; §7 new eval row; §8 FLOPs/token-count updated 82→126; §9 Problem 7 entry updated to "partially addressed"; new [AI Inference] entry; Related Concepts/See Also updated) · [[smoke-test-bringup]] (§0 table: adaptive graph moved from absent to active; Batch 0 spec updated with new success criteria; readiness checklist rows updated; Related Concepts) · [[open-architectural-problems]] (Problem 7 given an additive "partial resolution" note, same pattern as Problems 9/10/11 — nothing removed).
**Bookkeeping:** no new pages; counts unchanged (104 / 63). Sources unchanged at 36.

---

## [2026-07-01] note | Problem 25 (inequality/positivity/realizability constraints) filed; every problem header tagged with resolution status

**Operation:** Filed a new **Problem 25** in [[open-architectural-problems]] — the projection machinery (S1.1, P2.1–P2.4) is equality-only, and general physical fields carry inequality/positivity/realizability constraints (density $\ge0$, phase fractions in $[0,1]$, PSD Reynolds-stress realizability) that nothing currently enforces. Surfaced during a "general physics" audit; 1D Burgers never exposed it since $u$ has no sign constraint. Also tagged **all 25 problem headers** with a resolution-status line (additive only — no problem text removed or altered beyond the new tag), per explicit request.
**Problem 25 — solutions filed (scored):** S25.1 reparameterize for admissibility-by-construction — exp/softplus for positivity, Cholesky factor for PSD tensors (9, **adopted default**) · S25.2 explicit clipping/box projection (6) · S25.3 Dykstra's alternating projection, to combine with an existing equality constraint like S1.1 (7, **adopted complement**) · S25.4 eigenvalue clipping for PSD realizability (6, fallback) · S25.5 soft penalty only (4, rejected as sole mechanism, consistent with the project's stance on Problems 1/6). Framed as Tier-1-adjacent in severity (threatens physical admissibility generally) but outside Tier 3's dimensionality frame (applies even in 1D) — placed after Tier 3, before "What is actually solid."
**Status tags added to all 25 problem headers:** **DECIDED** (1, 2, 3, 6, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 24, 25 — 17 problems); **DECIDED with an explicit caveat** (4 — mechanism only, message form still open; 5 — mechanism only, guided-diffusion projection still to spec; 15 — process only, ablation decided but the equivariance question itself still open; 22 — scoped out, decided to defer); **PARTIALLY DECIDED** (7 — adaptive-patching sub-case resolved, persistent-identity sub-case open); **PROVISIONAL DEFAULT** (9, 10 — full scored passes still available on request); **OPEN** (8 — no candidate solutions scored yet, flagged as the most direct blocker to general multi-field physics training alongside the new Problem 25).
**New Priority-ordering addendum 3:** ties Problem 25 to Problem 8 as the two most direct blockers to general (multi-field) physics training, and gives the overall status tally (21 decided-family / 1 partial / 2 provisional / 1 open) as a single at-a-glance summary.
**Pages adjusted:** [[open-architectural-problems]] only — status tag added to each of the 25 problem headers; new Problem 25 section (5 scored solutions + decision); Priority-ordering Addendum 3. Purely additive; nothing removed or reworded elsewhere.
**Bookkeeping:** no new pages; page/concept counts unchanged (104 / 63). Sources unchanged at 36.

---

## [2026-07-01] note | Closed all remaining open architectural problems; Rayleigh–Bénard set as first multiphysics target

**Operation:** A "make Noether 1.0 trainable on general physics" pass that decided the six problems in [[open-architectural-problems]] not yet fully resolved (P4 message-form, P5 diffusion-projection, P7 persistent-identity, P8 field-cardinality, P9 equivariance, P10 stability). Goal per user: train Burgers first (already fully specced in [[noether-1.0]]), then a multiphysics test. **First multiphysics target chosen: 2D Rayleigh–Bénard convection** (coupled velocity + temperature + pressure, Boussinesq buoyancy, no particles). Scope: decide *everything* fully now, including the particle problems (mechanisms decided, implementation deferred to the S4.4 particle benchmark).
**Problem 8 (field cardinality) — DECIDED (was the last fully-OPEN problem):** S8.1 per-field-type typed encoder summing into a shared irrep-tagged node latent, with a by-arity fallback for unseen field types (9, adopted) + S8.5 universal per-field CDF/quantile normalization for scale-commensurability (8, adopted) + S8.4 field-type hypernetwork as the zero-shot upgrade path (7). S8.2 fields-as-tokens rejected (forks the one-node-per-location invariant); S8.3 padded-superset rejected (closed vocabulary). Instantiated for RBC's velocity(vector)/temperature(scalar) set.
**Problem 9 (equivariance) — DECIDED (full pass):** the "commit or not" binary is *dissolved* — RBC's gravity breaks SO(2), so hard rotational equivariance would be *wrong physics*; the fix (P9.2, 9) is to feed symmetry-breaking fields (gravity, Ω, EM) as **covariant conditioning vectors** so the model is equivariant to the full group acting jointly on state + external fields (making equivariance always correct, reducing the question to pure cost). Realized via P9.3 channel-typed selective equivariance (F2 on vector/tensor, F1 on scalars, 8), P9.1 augmentation over the true residual group as first-pass baseline (7), S15.2 ablation decides cost. Generalizes S24.1's Ω-as-covariant-input pattern.
**Problem 10 (training stability) — DECIDED (full pass):** P10.1 analytic spectral-norm control via Cayley/Householder skew term + Stiefel F1 (9) + P10.4 ReZero/bounded per-layer gain for normalization-free depth stability (8, the piece the 1D provisional default could omit but deeper RBC cannot) + P10.3 init/warmup/clip/AdamW bundle (8); P10.5 fused kernel deferred, P10.2 implicit integrator reserved. Promotes [[noether-1.0]] §5/§6.1 from provisional to decided.
**Problem 4 (cross-type message form) — DECIDED:** nondimensional scale-bridge via S20.1 scales (P4.3, 8) → IBM adjoint spread/interpolate as the physical message form (P4.1, 8) → conserved-flux antisymmetry folding into S1.1 (P4.4, 7) + learned residual fallback for EM/plasma (P4.2, 6). Implementation deferred to S4.4.
**Problem 5 (diffusion constraint-projection) — DECIDED:** confine diffusion to the conserved-orthogonal residual channel (P5.5, 8, → conservation free) + generate div-free part as stream function (P5.3, 8, → incompressibility free, no in-loop Leray solve) + x₀-projection for BCs via shared P2.2 machinery (P5.2, 9); P5.4 guidance as cheap complement. Load-bearing for turbulent RBC, deferrable for laminar (as Burgers).
**Problem 7 (persistent identity) — DECIDED:** persistent identity on nodes + ephemeral rebuilt edges (P7.1, 9, momentum lives on nodes so edge churn can't inject it) + particle-sum conservation anchoring generalizing §3.7 (P7.2, 8) + smooth radial cutoff making churn continuous (P7.4, 7); P7.3 soft re-identification rejected. Fragmentation flagged as narrow open edge case. Implementation deferred to S4.4.
**Overall status now: all 25 problems DECIDED.** Remaining items are implementation or explicitly-deferred: P4/P7 particle-coupling implementations (S4.4), P9's equivariance-*cost* ablation (first 2D run), P22 moving boundaries (scoped out), P7 fragmentation (narrow edge case). Direct RBC blockers (P8, P2 div-free, P18 elliptic solve, P25 admissibility) are all decided — what remains is building, not designing.
**Pages adjusted:** [[open-architectural-problems]] (P4/P5/P7 headers updated + new scored subsections; P8/P9/P10 headers updated + full scored passes; top Status line; earlier addendum's "remain open" clause marked superseded; new Addendum 4 with all-decided tally) · [[noether-1.0]] (§9 "What this leaves open" rewritten to reflect closures + flags the RBC config as next artifact). Purely additive to the problem texts; nothing removed.
**Bookkeeping:** no new pages; page/concept counts unchanged (104 / 63). Sources unchanged at 36. **Flagged next artifact:** a `noether-1.0-rbc` implementation-spec page (the RBC analogue of [[noether-1.0]]).

---

## [2026-07-01] note | New page [[noether-1.0-rbc]] — first multiphysics config (2D Rayleigh–Bénard), implementation-ready

**Operation:** Created the Rayleigh–Bénard analogue of [[noether-1.0]], specced to implementation depth (every tensor shape, module, init, hyperparameter) so the project can go straight to coding. Instantiates the six decisions from the "make Noether trainable on general physics" pass onto the first multiphysics benchmark.
**Key RBC-specific design choices (all following prior decisions):** (1) **stream-function velocity output (P2.4)** — decode a stream-function *increment* $\delta\psi$, set $u_{t+1}=u_t+\nabla^\perp\delta\psi$ → incompressible by construction, zero solve cost, sidesteps the P2.1/S18 pressure projection entirely in 2D; (2) **multi-field typed encoder (P8: S8.1+S8.5)** — per-field-type cross-attention summed into shared latent with type embeddings, per-field CDF normalization; (3) **gravity as covariant conditioning (P9.2)** — since gravity breaks SO(2), feed $\hat g$ as a covariant vector; baseline = P9.1 augmentation (horizontal translation/reflection), P9.3 selective equivariance is the S15.2 ablation; (4) **wall BCs (S14.1/S14.3)** — analytic box SDF, ghost-node reflection for no-slip/no-penetration on $\psi$, Dirichlet overwrite for $T$ — the first benchmark to exercise boundaries at all; (5) **P10.1 Cayley + P10.4 ReZero** stability for the deeper $L{=}6$ backbone; (6) **diffusion (P5) deferred** for laminar, specced for turbulent Ra.
**Honest framing recorded:** RBC is the *complement* of Burgers, not a harder version — it stresses multi-field cardinality / incompressibility / wall BCs while **conservation goes nearly silent** (driven-dissipative: no momentum/energy conservation, mass free via stream function). S1.1's projection is ~a no-op here; the conserved-readout is kept only as a Nu diagnostic. Passing *both* Burgers and RBC exercises disjoint halves of the machinery.
**Sizing:** grid $128\times64$; 128 leaf patches ($8\times8$); hierarchy $128\to32\to8\to2\to1$ (43 internal); 171 nodes/snapshot, 342 tokens. Configs nano/base/large ≈ 0.9M / 5.1M / 18M; primary target RBC-base ($d{=}256,L{=}6,H{=}8$), backbone $12Ld^2=4.72$M. Bring-up staged (Batch 0 = single trajectory, uniform patches, div-free assertion as go/no-go), mirroring [[smoke-test-bringup]].
**Data source:** The Well `rayleigh_benard` (same set [[gphyt-physics-foundation-model]] / [[walrus-paper]] use); Nusselt number $\mathrm{Nu}(\mathrm{Ra})$ is the physics success criterion.
**Pages adjusted:** new [[noether-1.0-rbc]]; [[index]] (added row under initial-pfm-model, page count 104→105 / concepts 63→64, updated open-architectural-problems description to "25 problems all decided"); [[noether-1.0]] + [[open-architectural-problems]] See Also cross-links.
**Bookkeeping:** +1 concept page (105 / 64). Sources unchanged at 36. **Next:** coding — Batch 0 (multi-field encoder + stream-function decode + div-free assertion) is the first thing to build.

---

## [2026-07-01] lint | Build-layer audit fix: adaptive-patch minimum sizes in [[noether-1.0]] / [[noether-1.0-rbc]] were arithmetically forced-uniform

**Operation:** During Phase-1 implementation in the build repo (github.com/nonidino/physics-foundation-model), a genuine spec inconsistency was caught in both config pages' graph-construction sections: the leaf-count budget × minimum patch size exactly equaled the full grid, so the "adaptive" refinement could only ever terminate in the uniform partition.
- **[[noether-1.0]] §3.2/§2:** 32 leaves × min width 4 = 128 = the whole grid ⇒ every leaf forced to width exactly 4; "variable width, denser near the shock" was arithmetically impossible, and §7's conservation-under-topology-churn ablation (adaptive vs. static) would have compared identical graphs. **Fix:** min width 2 (max depth 6). Preserves the 32-leaf/63-node/126-token budget and all parameter/FLOP numbers; the static-ablation baseline stays uniform 32×4.
- **[[noether-1.0-rbc]] §3.1/§2:** 128 leaves × min patch 8×8 (64 pts) = 8192 = the whole 128×64 grid ⇒ same forced-uniform degeneracy; additionally 8×8 sits at quadtree depth 3 (under the two square 64×64 roots the …→2→1 hierarchy implies), contradicting the page's own "max depth 4". **Fix:** min patch 4×4 (making max depth 4 self-consistent). Preserves the 128-leaf/171-node/342-token budget and all sizing.

**Pages adjusted:** [[noether-1.0]] (§2 Stage-2 pseudocode; §3.2 refinement bullet + fix note) · [[noether-1.0-rbc]] (§2 Stage-2 pseudocode; §3.1 patches bullet + fix note).
**Bookkeeping:** no new pages; counts unchanged (105 / 36 sources).

---

## [2026-07-01] lint | Build-layer audit fix 2: rollout length "next 90" was inconsistent with the t=2 pass bar

**Operation:** Second inconsistency caught during Phase-1 implementation: [[possible-architectures]]'s benchmark table said "given first 10 timesteps, autoregressively predict next 90," but with $\Delta t=0.01$ on $t\in[0,2]$ (200 steps) that reaches only $t=1$, while the same page's pass bar (and [[noether-1.0]] §7's primary-accuracy row) measure relative $L^2$ **at $t=2$**. The two numbers cannot both hold. **Fix:** harmonized to the pass bar (the repeatedly-cited number): rollout = the remaining **190** steps to $t=2$. The build layer's eval reports both $t=1$ and $t=2$ for continuity with literature protocols that use 10→90.
**Pages adjusted:** [[possible-architectures]] (task row) · [[noether-1.0]] (§7 conservation row; §8 rollout-cost sentence).
**Bookkeeping:** no new pages; counts unchanged.

---

## [2026-07-02] note | Measured the G0 tokenizer discretization floor (256-point reference), refined the placeholder threshold

**Operation:** During Phase-1 bring-up, the G0 tokenizer-fidelity gate ([[smoke-test-bringup]] §2, S3.4) trained on a GPU to a hard plateau: pooled reconstruction relative $L^2 = 0.019$, **median $0.006$**, failing the $<0.01$ placeholder. Rather than accept the fail or relax the number arbitrarily, ran the refinement [[smoke-test-bringup]] §2 explicitly deferred ("refine against a doubled-resolution 256-point reference solve once available"): a new build-layer script (`scripts/g0_floor.py`) solves the level-2 recipe at **both 128 and 256 points** and measures the pooled discretization error over 2200 snapshots.
**Result — the placeholder was below the physical floor.** Measured 128-vs-256 pooled discretization floor = **0.0179**, *entirely* attributable to $\nu<5\times10^{-3}$ shocks (per-band floor **0.0285**; mid-$\nu$ 0.0005; high-$\nu$ $\approx0$). Those shocks are $\sim$1 grid cell wide at 128 points and genuinely under-resolved, so **no tokenizer can reconstruct them below $\approx0.018$ pooled** — the $0.01$ target demanded reproducing the 128-field *more* faithfully than the 128-field itself represents the true PDE solution (i.e. fitting sub-grid Gibbs artifacts). The tokenizer's pooled $0.019\approx$ floor with median $0.006$ means it **reached the discretization floor**; the latent is provably adequate (S3.4 satisfied), not the bottleneck.
**Refinement adopted:** the S3.4 gate is now "pooled reconstruction $\lesssim$ the measured 0.0179 floor (10% sampling band) **and** median $<0.008$," not the $<0.01$ placeholder. This is the intended-all-along refinement, not a goalpost move — the $0.01$ was flagged "workable first threshold" from the start. **[AI Inference]:** this is a general lesson for the whole project's eval design — a fidelity threshold must be stated *relative to the data's own discretization error*, or it silently asks the model to fit numerical noise; the same caution applies to the RBC VRMSE bars in [[noether-1.0-rbc]] §7, which should likewise be floored against The Well's native resolution before being read as pass/fail.
**Pages adjusted:** [[smoke-test-bringup]] (§2 target row — placeholder → measured floor). Build layer: `scripts/g0_floor.py` (measurement, committed), `scripts/g0_gate.py` (threshold refined, reports pooled/median/floor/ratio).
**Bookkeeping:** no new wiki pages; counts unchanged.

---

## [2026-07-02] note | Phase-1 rollout validated: single-step works, rollout needs S1.5 noise-injection (confirmed empirically)

**Operation:** Full 1D-Burgers Phase-1 sequence trained on GPU (batch0 → G0 → level1/2 → f3/no-skew ablations → level4 push-forward → §7 eval). **Result:** single-step prediction is genuinely good (val rel-L2 **0.013**, beats persistence 0.018), and all structural gates pass (batch0 0.003; G0 at the 0.74× discretization floor; mass conservation exact to 1.9e-6; adaptive-vs-static graph churn matches — [[noether-1.0]] §3.7's Problem-7 argument holds empirically). **But autoregressive rollout blew up** (rel-L2 13 @t=1, 482 @t=2 on the varying-ν test set), worst at *high* ν (smooth): the true field decays viscously while the model **injects energy** (§7 excess-decay diagnostic hugely negative) and holds a jagged patch-scale field, so relative error explodes as the truth shrinks.
**Diagnosis — training regime, not architecture.** The linear backbone is provably stable at the trained weights ($\|W_{\text{skew}}-W_{\text{skew}}^\top\|_2\approx2.8 < 2/\epsilon$ bound); the energy injection is the SIREN decoder's high-frequency residual compounding over rollout. The build layer had **omitted the S1.5 context-noise injection** the curriculum specifies ("push-forward … with context-noise injection") — and level4 push-forward ran only 2 epochs (horizon 8 barely reached). **Adding noise injection cut nu=0.01 rollout rel-L2 from 24.9 → 1.57 at t=2 (16×)**, turning blowup into bounded tracking drift. So the S1.5 spec was right; the omission was a build bug, now fixed. **[AI Inference]:** this is a general reminder for Phase 2 (RBC) and beyond — *single-step accuracy is necessary but not sufficient; rollout stability is a distinct property that only noise-injected push-forward (S1.5) delivers*, and it must be budgeted with adequate epochs/horizon at the level4-equivalent stage, not tacked on for 2 epochs. RBC's turbulent-Ra path (P5 diffusion on the hi residual) is the eventual complement to this for the genuinely chaotic regime, but the deterministic core still needs S1.5 first.
**Pages adjusted:** none (spec was already correct). Build layer: `src/noether/train/loop.py` (pf_noise), `scripts/train.py` (--pf-noise), verification in the Phase-1 results.
**Bookkeeping:** no new pages; counts unchanged.

---

## [2026-07-03] note | Root-caused RBC rollout blow-up (normalization, not architecture); added the driven-dissipative S1.1 analog

**Operation:** Phase-2 (RBC) rollout evaluation diverged (monotonic kinetic-energy blow-up at all Ra). A separate analysis attributed it to the conservation-first backbone being a mismatch for driven-dissipative physics and proposed an architectural rewrite (state-dependent dissipation operator $R(h)$, regime-gated attention, explicit buoyancy). A build-layer diagnosis on the released base checkpoints **disproved that** and traced the cause to a normalization bug.
**Diagnosis.** The checkpoints were trained with **per-instance** velocity normalization ($\text{sd}_u=\text{std}(\mathbf u_{\text{curr}})$), which makes the map exactly **scale-equivariant**: the decoder increment is $\text{sd}_u\!\cdot\!\nabla^\perp\delta\psi$ with $\text{sd}_u\propto\sqrt{\mathrm{KE}}$, so it injects energy $\propto\mathrm{KE}$ every step — a *multiplicative*, hence **Ra-independent-rate**, slow exponential blow-up ($\sim1.9\times$/300 steps at every Ra). The backbone itself was innocent (contractive, $\gamma>0$ on every layer; no high-$k$ runaway). The energy enters through the unconstrained SIREN increment, which under instance-norm cannot see absolute magnitude to regulate it.
**Fixes (three-layer stack, no architecture change).** (1) **Ra-aware frozen normalization** ([[normalization-scheme]]): replace the per-instance std with a frozen power-law scale $\mathrm{rms}\,|\mathbf u|\approx a\,\mathrm{Ra}^b$ ($a\!\approx\!0.084,b\!\approx\!0.57$; fit offline, $<1.5\%$ error), exposing absolute energy so the decoder can regulate it — cut the blow-up $1.9\times\!\to\!1.25\times$/300 at **zero** single-step-accuracy cost (VRMSE 0.057 unchanged), div-free/walls intact. (2) **Stronger noise-injected push-forward** (horizon $1\!\to\!2\!\to\!4\!\to\!8$, 24 epochs): near-flat short/medium horizon but diminishing returns on the slow drift (trainable horizon $\ll$ drift-manifestation horizon). (3) **Attractor-energy projection** (new, [[attractor-energy-projection]]): the residual is a pure energy-magnitude drift, removed by softly projecting velocity onto the known attractor shell $E^\*(\mathrm{Ra})=2(a\,\mathrm{Ra}^b)^2$ each step — flattens 1000-step growth $4.4\times\!\to\!\sim1.0\times$ with 9-step VRMSE unchanged, div-free-preserving, no retrain.
**Pages created:** [[attractor-energy-projection]] (the driven-dissipative analog of S1.1). **Pages adjusted:** [[noether-1.0-rbc]] (§3.7 — the projection that *does* apply where conserved-total S1.1 is a no-op); index.md (new concept row; 105→106). Build layer (branch `fix/energy-regulation-norm`): Ra-aware `fit_normalizer`/`_vel_scale`, `NoetherRBC.project_energy`, `eval_rbc --energy-project`, unit tests (70 green).
**[AI Inference]:** The general lesson for the project: **for driven-dissipative regimes, project onto the attractor's known invariant statistics (energy, enstrophy, Nu, flux balance), not onto conservation laws that do not hold** — the driven-dissipative sibling of S1.1's tiered invariants. And a methodological one: an elaborate architectural hypothesis (the $R(h)$ rewrite) was not warranted; the failure was a normalization bug plus a slow global mode, each fixable without touching the backbone — diagnose the operative mechanism before re-architecting.

---

## [2026-07-03] reorg | Rewrote [[noether-1.0-rbc]] as an as-built model report (was a design-spec page)

**Operation:** The RBC page had drifted significantly from the build layer — three build-only sessions of work (rollout-divergence root-cause + fix, Phase-A horizon-resolved eval + a negative tokenizer-transfer ablation, Phase-B spatiotemporal extension) were only in `MEMORY.md`/git history, not the wiki. Rewrote the page top-to-bottom as an as-built technical report (abstract, architecture-as-implemented, experimental-results tables with exact recipes, a limitations section), reading the current `src/noether/model/*.py` / `train/*.py` / `eval/*.py` directly rather than trusting the prior spec text.
**Material corrections vs. the prior page:** (1) **Normalization** is not the spec'd per-field CDF/quantile map — it's a much simpler *Ra-aware frozen power law* $s_u(\mathrm{Ra})=a\,\mathrm{Ra}^b$ ($a\approx0.084,b\approx0.567$, fit offline), arrived at as the fix for a real scale-equivariance rollout-divergence bug, documented in the new §5. (2) **The adaptive quadtree is unimplemented** — the prior page's §3.1 described it as the "Phase-2+ upgrade" without flagging that no adaptive builder exists in `graph/grid2d.py`; now stated plainly as an open item, further de-prioritized by (4). (3) **S1.1 conservation is not merely "near-no-op" — it is not wired into the forward pass at all**; only the diagnostic Nu readout remains. Its rollout-stability role is filled by the new [[attractor-energy-projection]] mechanism instead. (4) **A completed ablation (S3.4 tokenizer-pretrain transfer) returned a negative result**: transferring a reconstruction-pretrained encoder into the dynamics model made every rollout horizon worse while leaving single-step accuracy bit-identical — evidence that (a) the Ra-dependent single-step error is not training-bound, and (b) reconstruction fidelity and rollout accuracy are partly anti-correlated (high-frequency detail is rollout-brittle). This was not in the wiki at all. (5) **A new spatiotemporal K-frame context (`n_ctx`) is implemented** (backward-compatible, K=2 reproduces the original two-frame path exactly) — the general concept was already covered by [[spatiotemporal-tubelet-tokens]] but its RBC-specific instantiation (temporal-clique adjacency generalization, K-frame push-forward buffer) had no page.
**Pages adjusted:** [[noether-1.0-rbc]] (full rewrite), [[attractor-energy-projection]] (stale §3.7 cross-references corrected to the new section numbers), `index.md` (row description updated to reflect the as-built framing).
**Bookkeeping:** no new pages; counts unchanged (106).
**[AI Inference]:** The gap between this page and the build layer is itself a data point: for an actively-developed model, the wiki's "answer a query → file the answer back as a page" workflow needs to run *continuously* against the build repo, not just at query time, or the design-layer documentation silently becomes historical fiction. A useful discipline going forward: treat any wiki page describing a *specific, currently-being-built* model (as opposed to a general concept) as needing re-sync whenever a build-layer session produces a result that would change a sentence on that page — not just when someone asks a question that happens to touch it.

---

## [2026-07-04] reorg | Renamed initial-pfm-model → "Noether 1.0"; created "Noether 1.1" architecture folder (9 new portion pages)

**Operation:** A design session stepped back from the [[noether-1.0-rbc]] build, which had drifted into a **smoothed-near-copy** failure mode (rollout keeps the initial frame nearly unchanged: velocity spectrum matches, rollout-mean Nu under-predicted 15–21% — the low-pass-smoother signature, §6.6 there) and a **loss of generality** (a bespoke incompressible-fluid solver, not the equation-agnostic model [[noether-1.0]] intended). Diagnosis: the smoothing is the trivial fixed point of a single-step MSE loss on near-identical frames ($\Delta t_{\text{out}}=0.004$) under a residual head + contractive backbone + heavy stability regularization, over an *information-starved* representation. Reframe adopted: **the model is a pattern recognizer, not just a solver** — it should carry an over-complete, redundant representation and *discover* structure rather than have it hard-coded.
**Folder rename.** `concepts/initial-pfm-model/` → `concepts/Noether 1.0/`. Bare-link convention means no links broke; `index.md` section header + folder path updated, rename noted inline.
**New folder `concepts/Noether 1.1/`** — the second-generation architecture, one page per model portion under a shared four-section contract (intuition → mathematics → physics relevance → how fundamental constants enter), linked from a hub overview.
**Pages created (9):** [[00-noether-1.1-overview]] (hub + end-to-end pipeline + design invariants); [[conditioning-and-constants-1.1]] (unified constant-handling contract: ruler/dial/arrow; constants reach into topology and the conserved set, not just a bias); [[graph-tokenizer-1.1]] (redundant derived node features); [[field-token-streams-1.1]] (**one token per field on a shared graph — removes 1.0-RBC's lossy summation**); [[edge-generation-1.1]] (**dedicated**: same-field KNN + cross-field KNN + **learned long-range edge model**; conservative-geometric vs. non-conservative-learned split); [[backbone-1.1]] (momentum-antisymmetry demoted to a **gateable channel**); [[decoder-1.1]] (**general increment head default; div-free no longer hard-wired** — restores compressible + particle generality); [[discovered-conservation-1.1]] (**learned invariants $\{C_k\}$; threshold-gated-hard + learned-soft; local/divergence discovery first**); [[post-decoder-diffusion-1.1]] (**dedicated**: generative fill-in of the chaotic high-$k$ residual band the deterministic decode blurs).
**Design decisions locked this session (user):** separate token per field on a shared graph; learned long-range edges (NRI-style, additive on a geometric base); momentum-antisymmetry default-on-but-**gateable**; conservation enforcement = threshold-gated-hard + learned-soft **hybrid**; first-cut conservation scope = **local (divergence) discovery** on NS fields, then RBC; keep everything **general to particles too** (no fluid-specific hard constraints — hence retiring div-free-by-construction and the pressure/streamfunction token); **all fields share one hierarchy** for now.
**Pages adjusted:** `index.md` (renamed 1.0 section; new Noether 1.1 section with 9 rows; page count 106→115, concepts 65→74; date).
**Status:** design only — no training has run on the 1.1 revision; numbers inherited from [[noether-1.0]] are design estimates. Build-layer implementation deferred pending this wiki clean-up (explicitly requested first).
**[AI Inference]:** The single highest-leverage representational fix is likely the smallest one — *not summing* the per-field encoders ([[field-token-streams-1.1]]). Even before learned edges or discovered conservation, keeping velocity and temperature latents separate should let the backbone hold the phase relationship (hot fluid rising) that a summed vector structurally cannot — a cheap, testable prediction to run before the larger 1.1 machinery.

---

## [2026-07-04] note | Added Noether 1.1 substep page: field descriptors & simulation-conditioned queries

**Operation:** Fleshed out a design idea raised in session — appending learned per-field values that capture "type + forces," and making the tokenizer's query bank simulation-specific — into its own portion page and wired it into the 1.1 stream as a tokenization substep (structurally parallel to how [[intelligent-patching]] hangs off the tokenizer).
**Idea.** One learned object, a **field descriptor $\theta_f$** (structured symmetry tags: rank/parity/rep-type + conserved-flag; plus a free learned block), does two jobs: (1) concatenated into a reserved token slot — *not* added, same anti-summation logic as [[field-token-streams-1.1]] — so the backbone always knows the field's type and its *force affordances* (which couplings it may engage, gating edge types in [[edge-generation-1.1]]); (2) generates the field's **simulation-conditioned query bank** $Z_q^f = Z_q^{\text{base}} + \Delta_q(\theta_f, z_{\text{sim}})$, where the simulation descriptor $z_{\text{sim}}=\operatorname{pool}_f\theta_f \Vert z_{\text{cond}}$ is *derived* from the present field set + constants, never a hand label. Recommended the low-rank/FiLM modulation (B2) over per-field tables (B1), hypernetwork (B3, = S8.4), or basis-mixture (B4) — keep a shared base bank so cross-simulation transfer (the foundation-model premise) survives. Instantiates [[physics-conditioned-query-tokens]] ("queries = physical operators") for the graph tokenizer and fleshes the S8.4 path for [[open-architectural-problems]] Problem 8 (variable field cardinality).
**Pages created (1):** [[field-descriptors-1.1]].
**Pages adjusted:** [[00-noether-1.1-overview]] (substep row 2b in the portion table + pseudocode + related-concepts); [[field-token-streams-1.1]] (token equation now concatenates $W_e\theta_f$; upgraded from opaque $e_f$); [[graph-tokenizer-1.1]] ($Z_q$ noted as simulation-conditioned, links to [[physics-conditioned-query-tokens]]); `index.md` (new substep row; 115→116, concepts 74→75).
**[AI Inference]:** Deriving the "simulation type" as $\operatorname{pool}_f\theta_f\Vert z_{\text{cond}}$ rather than a categorical label is the piece that keeps this from re-introducing the hard-coded-domain problem 1.1 exists to remove — the model's notion of "which physics" becomes a continuous point in descriptor+constant space, so unseen systems interpolate instead of falling off a lookup table.

---

## [2026-07-04] note | Deepened [[discovered-conservation-1.1]]: rollout mechanism (soft vs. hard) + three-tier generalization story

**Operation:** A design-session Q&A pushed past the page's original statement of the soft/hard hybrid to two questions the page didn't yet answer precisely: (1) a soft loss never executes at inference — so what does "it helps rollout" actually mean, mechanically? (2) the discovery machinery is trained on a fixed corpus — so what, concretely, transfers to a system the model has never seen?
**Rollout mechanism, resolved as two genuinely different effects, not one.** *Soft* is indirect: training pressure shapes the backbone/decoder weights so single-step predictions already tend not to move low-$\sigma_k$ quantities — nothing computes at inference, the trained function carries the effect forward — but this only holds per-step unless $D_k$ is measured **through a push-forward unroll**, not single steps alone. Cited direct precedent already in the vault: [[noether-1.0-rbc]] §5.2's push-forward result ("trains stability directly into the weights," flat raw rollout at 200 steps) is exactly this mechanism, previously validated for one hand-built quantity (energy) only. *Hard* is direct: the projection is **re-executed live at every rollout step**, and RBC's own measured split (push-forward flattens short/medium horizon; the explicit energy projection catches a slower residual drift push-forward misses) is cited as evidence the two channels are complementary, not redundant — the same split now generalized from one quantity to the whole discovered $\{C_k\}$ set.
**Generalization, resolved as three tiers of decreasing confidence.** Tier A (new parameter regime, same system type) — free, because $\sigma_k(z_{\text{cond}})$ is already specified as a *function* of conditioning, not a fixed scalar, so a new Ra/Ma is just a new point in an already-continuous space. Tier B (new system type, known field vocabulary in a new combination) — requires a **genuinely new requirement, added to the page**: symbolic seeds must be templated on field-descriptor rank/parity tags ([[field-descriptors-1.1]]'s $\theta_f$), never on a named field, so a generic "mass-like"/"momentum-like" seed auto-instantiates on whatever rank-0/rank-1 streams a new combination exposes. Tier C (genuinely novel-form physics, zero exposure) — **honestly left partially open**: route 1's search runs at training time only, so a truly novel invariant *form* cannot be discovered live on a single new rollout; proposed (not yet specified) a cheap online-recalibration of the trained prior $\sigma_k$ from a short context trajectory (pure function evaluation, empirical-Bayes-flavored), while flagging that genuine live discovery of a new-form invariant needs [[noether-networks]]-style test-time tailoring or route 2 ([[action-based-noether-enforcement]]), both unbuilt.
**Pages adjusted:** [[discovered-conservation-1.1]] — two new sections ("How this behaves during rollout," "Generalization to new systems"), the $C_k$-seed definition amended for rank/parity genericity, one new open item (online recalibration, unspecified), Related Concepts / See Also extended ([[field-descriptors-1.1]], [[autoregressive-rollout-stability]], [[in-context-learning-physics]]).
**Bookkeeping:** no new pages; page counts unchanged.
**[AI Inference]:** The Tier A/B/C split is itself a reusable pattern worth watching for elsewhere in the 1.1 design: "generalizes for free via conditioning" (A) vs. "generalizes because the mechanism is templated on structure, not names" (B) vs. "does not generalize without a genuinely new capability, flag honestly" (C) is a useful three-way test to apply to any other 1.1 mechanism that claims generality — e.g. [[edge-generation-1.1]]'s learned scorer or [[field-descriptors-1.1]]'s query-bank modulation likely sit at Tier B today and would need the same honest Tier-C accounting if pushed.

---

## [2026-07-04] note | Deepened [[post-decoder-diffusion-1.1]]: forward/reverse process spelled out, design commitments re-derived

**Operation:** A design-session walkthrough of the diffusion equation found the page stated the training loss but left the actual generative mechanism implicit — "sample $r$ at inference" was asserted, not shown. Expanded the "What it does — mathematically" section to make every piece explicit rather than compressed.
**Additions.** (1) Disambiguated the diffusion timestep $\tau$ from the physical timestep $t/\Delta t$ — a notational collision worth flagging explicitly since both processes have their own "time." (2) Unpacked the forward noising process: $\bar\alpha_\tau$ is a fixed (not learned) schedule interpolating between clean residual and pure noise; explained why $\epsilon$-prediction (not direct $r$-prediction) is the standard choice (comparably-scaled training signal across noise levels); named what each of the three conditioning arguments ($\bar v_{t+1}$, $\{h_i\}$, $z_{\text{cond}}$) individually contributes. (3) **Added the reverse (sampling) process in full** — the ancestral denoising loop from pure noise down to a sampled residual — which was previously entirely absent from the page; also named few-step samplers (DDIM/consistency-model style) as the concrete mechanism behind the existing "mitigated by few-step samplers" open item. (4) Re-derived each of the four design commitments with a concrete "why," notably: the run-after-conservation ordering is safe *because* [[discovered-conservation-1.1]]'s projection is provably minimal-disturbance (cross-referencing that page's own derivation from the prior session); phase-separated training is justified by the same nonstationary-target argument that motivates freezing an autoencoder before training latent diffusion in [[latent-diffusion-physics]].
**Pages adjusted:** [[post-decoder-diffusion-1.1]] — "What it does — mathematically" section expanded (forward process, network/loss, reverse process, re-derived commitments); no structural/frontmatter changes, no new pages.
**Bookkeeping:** no new pages; page counts unchanged.
**[AI Inference]:** Across this session's three "chunk it down" passes (backbone, decoder, discovered-conservation) and now diffusion, the recurring gap has been the same shape each time: a page states a compressed equation plus a one-line consequence, and the missing content is always the *procedure* the equation is shorthand for (the tied-QK head expansion; the SIREN internals + Leray-solve cost argument; the projection derivation; now the reverse sampling loop). Worth treating as a standing lint check for the rest of Noether 1.1: any page whose math section ends in "...and then X happens" without showing *how* X happens is probably under-specified in the same way.

---

## [2026-07-04] note | Swept Noether 1.1 for the "compressed procedure" gap; deepened 5 pages, reviewed the rest

**Operation:** Following the pattern flagged after the [[post-decoder-diffusion-1.1]] deepening — a page states a compressed equation plus a one-line consequence, and the missing content is always the *procedure* the equation is shorthand for — swept every page in `concepts/Noether 1.1/` for the same gap and fixed what was found, rather than waiting for each to be asked about individually.
**Fixed (content largely already drafted across this session's earlier chat turns, now folded in):**
- [[backbone-1.1]] — added the per-head tied-$Q{=}K$ attention expansion (where $A_{ij}^{(\tau)}$ and $v^{(\tau)}$ actually come from), the explicit F1 FFN sub-step equation, the Cayley parameterization formula $(I-S)(I+S)^{-1}$, and the reciprocity-vs-antisymmetry decomposition of Newton's third law into its two independently-enforced halves.
- [[decoder-1.1]] — added the Fourier coordinate-encoding rationale, the SIREN layer/init specification, and — the most substantive addition — split "structure-preserving heads" into its two variants at genuinely different cost: the stream-function head (free, an exact mixed-partials identity) vs. the Helmholtz-projected head (an actual Poisson/Leray solve), previously conflated as one bullet.
- [[edge-generation-1.1]] — added the GumbelSigmoid score-to-edge relaxation mechanics (straight-through estimator, temperature annealing) and the forward/backward NRI-style training-loop explanation (no edge labels; the sparsity budget vs. prediction-loss gradient tension that makes an edge "earn its place").
- [[graph-tokenizer-1.1]] — clarified that the high-frequency path $h_i^{\text{hi}}$ is a second independent query-compression encoder, not a literal subtraction — "residual" names a role (what the lo-path's training pressure leaves unclaimed), not a computed difference.
- [[conditioning-and-constants-1.1]] — gave $\gamma_{\text{vec}}$ a concrete construction (linear + invariant-scalar-gated, no bias, no bare nonlinearity on the vector) and named why each excluded operation would break the rotation identity — the standard gated-equivariant-nonlinearity pattern, cross-linked to [[equivariant-gnns]].
**Reviewed, no fix needed:** [[field-token-streams-1.1]] and [[field-descriptors-1.1]] already carry explicit formulas rather than black-box verbs; [[00-noether-1.1-overview]] correctly defers all mechanism detail to sub-pages by design (a hub page restating sub-page math would be the opposite failure).
**Bookkeeping:** no new pages; five pages edited, three reviewed and left unchanged; page counts unchanged.
**[AI Inference]:** The five fixes cluster into two repeatable failure shapes, worth naming for future pages written in this vault: (1) **a symbol used before it's defined** (backbone's $A_{ij}^{(\tau)}$, decoder's $\mathrm{Dec}$, edge-generation's $s_{ij}\to a_{ij}$) — the equation names something the surrounding prose never constructs; (2) **a word implying a mechanism the math doesn't show** ("residual" in the tokenizer, "steerable" in conditioning) — natural-language shorthand that reads as more specified than it is. A useful authoring check going forward: for every symbol in a display equation, can a reader point to the sentence that builds it; for every adjective describing a mechanism, does an equation exist that earns it.

---

## [2026-07-08] note | Closed five Noether 1.1 under-specifications + added [[training-scheme-1.1]]

**Operation:** A design-session Q&A raised eight probing questions about Noether 1.1; five exposed genuine gaps in the portion pages, fixed in-place, and two of the "which is trained when / on what" questions were spun into a new master-spec page (the user's explicit split: fix under-specs inline, but give the training scheme its own file).

**Under-specifications fixed inline:**
- [[backbone-1.1]] — **(a)** the antisymmetry gate $g_\tau$ was implicitly a sigmoid, which can never reach $1$ exactly and so leaks momentum $\propto(1-g)$ every layer (≈linear accumulation over a rollout); specified a **clamped/hardtanh** (or one-sided $1-\mathrm{ReLU}$) gate that hits $1$ and $0$ *exactly on open sets*, is sticky at the rails, with a hard-concrete alternative — plus the crucial framing that even exact-$1$ only sharpens the *soft* bias (the per-node $\tanh$ breaks per-layer conservation; the *hard* guarantee is [[discovered-conservation-1.1]]'s live projection, Problem 1). **(b)** Addressed "summed heads → normalization?": *not* a norm-drift problem (ReZero absorbs the scale; forces are physically additive, so $1/H$ averaging is wrong), only a $\tanh$-saturation-at-init concern fixed by a constant $1/\sqrt H$ that preserves the $g{=}1$ telescoping. **(c)** Specified *where* vector constants couple into the backbone — invariant contractions into scores/gates (anisotropy) + additive vector source term (body force), the latter coinciding with the site $g_\tau$ must open; default components-as-channels vs. hard-equivariant covariant channel.
- [[conditioning-and-constants-1.1]] — added the cross-reference for where $\gamma_{\text{vec}}$ actually acts in the dynamics (pointer to backbone's two entry points; default vs. hard-equivariant).
- [[discovered-conservation-1.1]] — the SFA discovery objective is a generalized eigenproblem; added the **whitening + shrinkage** step, argued as a *correctness* condition not hygiene (implements the non-triviality constraint; the smallest — conserved — eigenvectors are the ones poor conditioning destroys; and the design's own redundancy thesis guarantees a rank-deficient $\Sigma$, forcing shrinkage/reduced-rank whitening). Open-item updated.
- [[graph-tokenizer-1.1]] — answered "no past states?": history *is* ingested (n-frame context) but the fusion was unstated; specified **temporal-derivative channels** ($\partial_t v,\partial_{tt}v$) as the default (Markov/low-order), with spatiotemporal tokens / a carried recurrent latent as the non-Markovian upgrades, and the copy-the-input caveat tying it to the multi-step loss. Added to the $\phi_p$ feature bundle.
- [[00-noether-1.1-overview]] — new subsection **"Shared vs. specialized parameters — and when 1.1 becomes an MoE"** answering the shared-vs-regime-specific question: almost everything is shared weights + conditioning, nothing per-regime; litmus test *operator changes form (→ MoE/basis) vs. only magnitude (→ conditioning)*; the model is already MoE-flavoured ($z_{\text{sim}}$ is a soft router, heads are a dense expert basis). INPUT line now notes the context trajectory.

**Page created (1):** [[training-scheme-1.1]] — module training order (6 phases P0–P5: tokenizer+descriptor reconstruction → single-step geometric-edge dynamics → learned edges + push-forward → conservation discovery + freed gates → phase-separated diffusion → new-regime extension) with a learnable-inventory table, plus the physics-regime curriculum (6 rungs adding one generality axis each: 1D Burgers → 2D NS → RBC → compressible → settling particles → MHD), freezing discipline, and the dependency rationale. Load-bearing decisions: **learned edges wait for a motion-demanding loss** (or they learn to do nothing) and **antisymmetry gates freeze at $g{=}1$ until conservation discovery** (so "which edges force" and "what is conserved" are decided jointly). Builds on, does not duplicate, [[training-curriculum]] (which owns the loss composition).

**Pages adjusted:** the five above + `index.md` (new Noether 1.1 row; 116→117, concepts 75→76; date 07-04→07-08).

**[AI Inference]:** Several answers converged on shared axes worth noting for future design work — Q "shared vs. regime-specific params" and Q "how is the conserved set stored" are the *same* conditioning-vs-MoE decision seen at features and at the invariant set ($\sigma_k(z_{\text{cond}})$ is the conditioning choice, a codebook would be the MoE choice); and the gate ($g_\tau$), the vector-constant entry point, and the anti-drift projection all meet at the *non-conservative forcing channel*. The recurring correct pattern across all five fixes is one stance: **shared base + conditioned correction for anything that must transfer; hard enforcement only for what is confidently measured; honest bounded drift for the rest.**

---

## [2026-07-08] reorg | Created `Noether 1.1 implementation/` folder (15 build-spec pages) + reorganized the code repo into noether-1.0 / noether-1.1 branches

**Operation:** Translated the Noether 1.1 *design* into an executable *build* layer, at the user's request: a dedicated implementation subfolder of the concept folder, plus a GitHub branch reorganization.
**Folder created:** `concepts/Noether 1.1/Noether 1.1 implementation/` (15 pages). Anchor [[00-implementation-plan]] lays out three macro-phases (Phase I foundations: scaffold/GUI/data; Phase II deterministic core: tokenizer→edges→backbone→decoder→harness, built in [[training-scheme-1.1]]'s P0–P2 order; Phase III: discovery/diffusion/eval, P3–P4), a dependency graph, milestones M0–M7, and the branch strategy. Its build order deliberately follows the *training-scheme* order, not the forward-pass order, because learned edges / discovered conservation / diffusion are unstable if built-and-trained early.
**Size configs:** [[noether-1.1-medium]] (~44M; $d{=}384,L{=}12$) and [[noether-1.1-large]] (~327M; $d{=}768,L{=}24$), each with a param-budget formula extending [[noether-1.0]] §4 for 1.1's extra modules (per-type attention $\Rightarrow (9{+}3T)d^2$/layer $=21d^2$ at $T{=}4$; plus descriptors, query modulation, edge scorer, per-field decoder heads, diffusion add-on). Both explicitly answer the user's embedded question — **the backbone is a deep stacked-attention model like an LLM** (medium 12 layers, large 24), transformer-like in skeleton but per-edge-type + port-Hamiltonian per layer, depth = latent time.
**Per-portion agent specs (11):** [[impl-repo-scaffold]], [[impl-gui-2d]], [[impl-data-generation]], [[impl-tokenizer-descriptors]], [[impl-edge-generation]], [[impl-backbone]], [[impl-decoder]], [[impl-training-harness]], [[impl-discovered-conservation]], [[impl-diffusion]], [[impl-eval-benchmarks]] — each self-contained (objective, design-page links, interface contract, build steps, acceptance tests, pitfalls) so a future agent can execute one independently. Build-progress tracked in [[implementation-log]] (a separate log from this one, per the user's request).
**GitHub reorg:** froze the complete 1.0 build (Burgers+RBC+NS, tip of the former `fix/energy-regulation-norm`) as branch **`noether-1.0`** (reference only, never edited) and opened **`noether-1.1`** for all new work; both pushed.
**Pages adjusted:** `index.md` (new "Noether 1.1 Implementation" subsection; 117→132, concepts 76→91). No design pages changed — the implementation layer references the concept pages as source-of-truth and never overrides them.
**[AI Inference]:** The implementation folder is the first vault content that is *downstream* of the design rather than part of it — worth keeping a firewall: if a spec file and a concept page ever disagree, the concept page is authoritative and the spec file is the bug (stated in [[00-implementation-plan]] §0). This keeps the design layer and build layer from silently diverging as code gets written.

## [2026-07-15] note | Fineness / token-density failure mode + fix (Noether 1.1)

**Trigger:** the P2 checkpoint rolls out visibly "pixelated" — coarse near-constant blocks vs. smooth ground truth.
**Diagnosis (confirmed on NS val):** the blocks *are* the tokens. At `patch_domain_size=0.25` on the $[0,2]\times[0,1]$ corpus the tokenizer makes only $8\times4=32$ patches (one token per $16\times16$ block); the decoder broadcasts one token across its whole patch, so sub-patch detail must be painted by a single SIREN → patch-mean → blocks.
**Key finding:** token count $P$ is a **runtime graph dimension, not a weight shape** (audited tokenizer/backbone/decoder/edges; all four densities load & run on the same checkpoint). So the answer to "must I restart training?" is **no — warm-start/fine-tune**. But finer density *zero-shot* doesn't help (rel-$L^2$ flat ~1.28, and the SIREN *rings* at out-of-trained-scale offsets): tokens are capacity, training fills them.
**New page:** [[noether-1.1-fineness-and-token-density]] (full diagnosis, evidence table, fix, procedure). Cross-linked from [[graph-tokenizer-1.1]], [[decoder-1.1]], [[noether-1.1-medium]], [[noether-1.1-large]]; index page count updated.
**Design→build sync:** `patch_domain_size` / `patch_size_jitter` now appear in the [[noether-1.1-medium]] / [[noether-1.1-large]] spec tables (the code sets them; the wiki stays authoritative).
**[AI Inference]:** uniform-finer + multi-scale jitter is the immediate fix; the token-efficient successor is adaptive placement ([[intelligent-patching]]) — the two compose, since jitter-trained heads are what an adaptive patcher needs to trust a variable patch scale.

---

## [2026-08-07] note | Regime-Conditioned MoE architecture proposal filed as a competing alternative to Noether 1.1

User brought in a synthesis from a separate design conversation (first-principles survey of ~20 physics regimes, 8 deep-dived) arguing that a single dense conditioned backbone — the direction [[00-noether-1.1-overview]] and this vault's existing [[pfm-architecture-approaches]] hybrid roadmap both converge on — may be structurally wrong for the breadth a PFM needs, and that a large MoE partitioned by equation family is the better decomposition. Filed as new page **[[regime-moe-architecture]]** in `concepts/00-pfm-core/`. Key content: local/global attention selection should track equation type (elliptic/hyperbolic/parabolic), not physics domain, and recurs identically across fluids/EM/gravity; regime crossovers split into smooth (Kn, Ma, $v/c$ — FiLM/adaLN conditioning) vs. threshold (yield surfaces, shock formation, fracture — a separate discontinuity channel), and conflating the two either over-smooths real discontinuities or introduces spurious ones; "simpler" asymptotic limits (hypersonic flow, Kn~1 kinetic transition) are not always cheaper to train first, breaking a naive curriculum-by-simplicity assumption; dimensionless numbers (already established as conditioning tokens in [[pfm-interface-design]]) should drive the MoE gate's logits directly rather than being re-derived from raw fields. Cross-linked into [[mixture-of-experts]] (concrete instantiation of its existing hard-routing-across-families AI Inference) and [[pfm-architecture-approaches]] (added as open question 7 — unresolved head-to-head against the existing hybrid roadmap).

**Explicitly scoped as documentation-only** (user decision, 2026-08-07): does **not** change [[00-noether-1.1-overview]]'s status, does **not** touch [[implementation-log]] or [[00-implementation-plan]] — the live Noether 1.1 build (277 tests passing at time of writing) continues unaffected. This is a proposal on record for future evaluation, not an enacted pivot.

---

## [2026-08-07] note | Vision page (multi-agent deployment target + 5 worked examples) and incremental transfer-learning build roadmap

Two new pages, continuing the same 2026-08-07 thread as [[regime-moe-architecture]]. **[[pfm-purpose-and-direction]]** fixes a concrete deployment target for the PFM — physical objects as agents interacting through the model when in proximity — with the moon/mars base as the paradigm case plus four further worked examples chosen to stress different axes: wildfire spread at the wildland-urban interface (widest scale range, radiative-reach proximity, threshold ignition), EV battery pack thermal runaway (safety-critical, mid-simulation regime reclassification), atmospheric reentry (multi-regime-on-one-object rather than multi-agent, the concrete case for [[regime-moe-architecture]]'s "simple limit ≠ cheap" finding), and an offshore wind farm over its operating life (proximity is wind-direction-dependent and regime-conditional — wake coupling is long-range, fatigue is purely local, same agent pair). Cross-example pattern: proximity/edge topology can't be a single fixed graph-construction rule. Reframes [[pfm-concept-overview]]'s 8 gaps against this target and adds 3 new ones (cross-boundary consistency between independently-computed agents, composability as inference-time not training-time generalization, verified fallback/abstention for safety-critical use).

**[[incremental-transfer-roadmap]]** answers the user's build-strategy question directly: given limited time/resources, bootstrap [[regime-moe-architecture]] from existing pretrained weights expert-by-expert rather than a from-scratch build (current [[00-noether-1.1-overview]] approach) or a from-scratch MoE. Checked every candidate donor model against [[code-and-papers]]'s actual resource catalog: **only Poseidon and Walrus have confirmed public checkpoints**; GNS, GP$_{\text{hy}}$T, Dynami-CAL, AION-1, and PISD all show `(find)` — unconfirmed — so those can only donate structural patterns (antisymmetric edges, derivative+integrator heads), not weights. Proposes a 4-stage bootstrap (frozen donor wrap → LoRA adapter fine-tuning → structural distillation submodule-by-submodule with a regression gate at each swap → shared-trunk unification), explicitly reconciled against Noether 1.1: the tokenizer/edge-generation/conditioning-token layer ([[graph-tokenizer-1.1]], [[edge-generation-1.1]], [[pfm-interface-design]]) is argued to already be the native scaffold this strategy needs and does not need to change — only [[backbone-1.1]] is the component proposed for donor-wrap-then-gradual-nativization. Calibrates which of the five worked examples becomes reachable at which stage.

**Still documentation-only** — no change to [[implementation-log]], [[00-implementation-plan]], or Noether 1.1's build status.

---

## [2026-08-08] reorg | Atlas: a new named architecture track, full concept folder created (9 pages + hub), implementation folder reserved empty

Continuing the 2026-08-07/08-08 design thread ([[regime-moe-architecture]] → [[pfm-purpose-and-direction]] → [[incremental-transfer-roadmap]] → the rocket-ascent case study), the user worked through a concrete agent/typed-edge graph for a 2D rocket ascent (7 agents: combustion reaction, combustion chamber, airframe, atmosphere-front, combustion outflow, plume, atmosphere-wake; the $b\text{–}g$ edge removed as physically unmotivated — base heating should route through the plume/airframe, not a direct chamber-to-wake-atmosphere shortcut), proposed RL-based edge instantiation as a documented alternative to the existing differentiable scorer, and proposed a Poseidon-style U-Net-of-experts hierarchy with per-level MLP/RL gating. Working through "what experts would this specific case need" surfaced a correction to the informal edge-type-as-expert framing from the prior session: heat/pressure/stress/fluid/conservation are **interface-contract labels**, not one-expert-each — the right cut is governing-equation family, since some type-pairs (e.g. heat+pressure at the reaction→chamber edge) are just two output channels of one tightly-coupled expert, while others (heat+stress at the chamber→airframe edge) are genuinely two weakly-coupled experts.

The user then proposed building this specific case study first — only the experts it needs, observe whether the full pipeline works, expand later — and asked for a new name distinct from "Noether," since conservation-law *discovery* ([[discovered-conservation-1.1]]) isn't this design's forte; composing declared local experts through declared interfaces is. **"Atlas"** was chosen: a manifold's atlas of local coordinate charts (agents) glued by transition maps (typed edges) that keep the composition globally consistent (conservation as an *imposed* constraint, not a discovered one) — the precise structural inverse of Noether 1.1's single-backbone-discovers-everything thesis.

**Full Atlas concept folder created** at `concepts/Atlas/`, mirroring [[00-noether-1.1-overview]]'s one-hub-plus-per-portion-page pattern: [[00-atlas-0.1-overview]] (hub — thesis, forward pass, 7 design invariants, explicit non-deprecation of Noether 1.1), [[agent-definition-atlas-0.1]], [[graph-tokenizer-atlas-0.1]] (reuses [[graph-tokenizer-1.1]] directly), [[edge-generation-atlas-0.1]] (both declared/geometric instantiation as the Stage-1 default *and* RL-based instantiation as a documented, deferred alternative — not before a second scenario exists to learn across), [[expert-library-atlas-0.1]] (the governing-equation-family cut, the rocket's 4-expert table: reacting/internal flow, external flow/plume, thermal-structural, and a **non-learned closed-form rigid-body trajectory expert** at physics-encoding level 5), [[unet-hierarchy-atlas-0.1]] (reuses [[multiscale-hierarchical-gnn]] + [[intelligent-patching]] in reverse; MLP-vs-RL gating mapped onto [[regime-moe-architecture]]'s smooth/threshold finding; resolves that page's open shared-trunk-vs-disjoint-experts question — the pooling structure *is* the shared trunk, per-level expert choice *is* the disjoint axis), [[conservation-as-constraint-atlas-0.1]] (direct contrast with [[discovered-conservation-1.1]]; the naming rationale formalized), [[global-fields-and-topology-atlas-0.1]] (two mechanisms not covered anywhere else in the vault: gravity as a uniform field bypassing edges entirely, and staging/docking as first-class graph-topology mutation — documented as a requirement, not yet designed), [[training-and-bootstrap-atlas-0.1]] (applies [[incremental-transfer-roadmap]]'s 4-stage bootstrap to the 4 rocket experts; only Poseidon has a confirmed-checkpoint donor match here), and [[case-study-rocket-ascent-2d-atlas-0.1]] (the Phase 0–5 plan of action — status: **planned, not built**).

Deliberate v0 scope reductions, each with a stated trigger for revisiting: declared edges not RL, MLP gating not RL gating, a 2-level U-Net not the full 4-level hierarchy, direct flux-matching conservation not learned discovery — chosen so a first end-to-end failure is diagnosable to one subsystem rather than entangled across several simultaneously-untested mechanisms.

**`concepts/Atlas/Atlas implementation/` created and left intentionally empty**, per explicit instruction — reserved for build-phase specs once work begins, mirroring [[00-implementation-plan]]'s role for Noether 1.1.

Cross-linked bidirectionally into [[00-noether-1.1-overview]], [[discovered-conservation-1.1]], [[graph-tokenizer-1.1]], [[edge-generation-1.1]], [[backbone-1.1]], [[multiscale-hierarchical-gnn]], [[intelligent-patching]], [[regime-moe-architecture]], [[pfm-purpose-and-direction]], [[incremental-transfer-roadmap]], and [[mixture-of-experts]]. **Noether 1.1's implementation status is unchanged** — Atlas is explicitly documented as a parallel track, not a supersession.

---

## [2026-08-08] reorg | Atlas → **Atlas 0.1**; full implementation layer written (8 pages) in `Atlas 0.1 implementation/`

**Rename.** `concepts/Atlas/` → `concepts/Atlas 0.1/`, inner folder → `Atlas 0.1 implementation/`, and all 10 concept pages given version-suffixed filenames (`00-atlas-0.1-overview`, `expert-library-atlas-0.1`, `case-study-rocket-ascent-2d-atlas-0.1`, …). All bare links across the vault were rewritten in the same pass; H1 titles updated to "Atlas 0.1: …". Version suffixes follow the Noether 1.0/1.1 precedent so a future Atlas 0.2 ([[impl-atlas-0.1-phase5-expansion]] §3.4) can coexist without filename collisions under this vault's unique-filename bare-link convention.

**Implementation layer.** Eight new pages, each self-contained (written on the explicit instruction that an agent holding *only* that page must be able to implement the phase correctly), each with **intuition → theory → implementation → acceptance tests → pitfalls**:

- [[00-atlas-0.1-implementation-plan]] — the binding global conventions everything else inherits. Fixes a **concrete planar-2D geometry** in which all 7 agents are non-overlapping and every one of the 7 declared edges has a real, non-degenerate interface (chamber $z\!\in\![0.12,0.40]$, throat at $z{=}0.40$ with $\varepsilon{=}3$, nozzle exit at $z{=}0.70$, etc.); separates **$\Delta t_{\text{solver}}$ (CFL, data generation) from $\Delta t_{\text{model}}$ (surrogate step)** — the distinction that makes the amortization claim meaningful; sets $d{=}256$, $L{=}6$, ~1.1k tokens, 15–25M params; repo layout as a new `atlas/` package parallel to `noether11/`.
- [[impl-atlas-0.1-phase0-scope-and-data]] — two classical solvers cover all seven agents (2D compressible NS with a one-step reaction progress variable; conduction + plane-stress elasticity), plus atmosphere and a 3-DOF trajectory module **written once and reused verbatim as the runtime non-learned expert**. Every solver gated against a closed-form oracle (Sod, area–Mach, Rayleigh, Blasius, erf-slab, Tsiolkovsky) *before* generating any corpus. LHS sweep with deliberate **corner-case holdout**; HDF5 schema stores interface fluxes, not just interiors (a corpus without them cannot train the interface layer).
- [[impl-atlas-0.1-phase1-scaffold]] — plumbing before physics: full forward pass with identity experts. Notable: the **agent-isolation test** ($f$ must reach $a$ only in ≥3 message-passing layers, via $f\!-\!e\!-\!b\!-\!a$) as the sharpest check the declared graph is real, and increment prediction so an untrained model degrades to "nothing changes."
- [[impl-atlas-0.1-phase2-experts]] — the governing-equation-family cut argued concretely from the $a\!-\!b$ (one coupled expert) vs. $b\!-\!c$ (two weakly-coupled experts) contrast; a Mach-conditioned causal attention mask at the throat as the one architectural physics encoding; structural gates (**choked-throat**, **zero-load**) held to be worth more than the $L^2$ numbers.
- [[impl-atlas-0.1-phase3-integration]] — the **exchange-cadence rule** $\Delta t_{\text{exch}}=\max(\Delta t_p,\Delta t_q)$, which reproduces the physically correct per-edge coupling rates without hand-specifying them; **characteristic-upwinded** conservation targets (supersonic ⇒ upstream side authoritative, subsonic ⇒ symmetric mean) so the constraint cannot inject acausal upstream influence at the nozzle exit; hard-project mass only, soft-penalize momentum/energy, with the compounding argument for why.
- [[impl-atlas-0.1-phase4-validation]] — falsification criteria stated **before** running; 7 ablations, of which **A4 (declared vs. fully-connected graph)** directly tests [[edge-generation-atlas-0.1]]'s premise and **A2 (conservation off)** directly tests [[conservation-as-constraint-atlas-0.1]]'s; an explicit decision rule including "if the monolithic ablation matches full Atlas, stop and reconsider."
- [[impl-atlas-0.1-phase5-expansion]] — wind-farm scenario 2, reuse levels R1/R2/R3, regression baseline frozen *first*, the fast/slow accumulation split for decade-scale fatigue (multi-rate subcycling does not extend to $10^9$ substeps), and RL edge instantiation finally unlocked with its stability-before-pooling ordering constraint.
- [[atlas-0.1-implementation-log]] — append-only tracker, status board, and carried-forward action items (verifying the Poseidon checkpoint license is item 1 and blocks two of three learned experts).

**Still nothing built.** Noether 1.1's implementation status remains unchanged; Atlas 0.1 is a parallel track.

---

## [2026-08-09] note | Atlas 0.1 **Phase 1 built** — the design layer's first Atlas code

First build against the Atlas implementation layer written the day before. Branch `atlas-0.1` cut from `noether-1.1`; package `src/atlas/` parallel to `src/noether11/`, no shared imports. Milestones **M0 and M3 green**; 32 new tests, full repo suite 318 passed.

Three findings that changed wiki pages, all recorded with root causes in [[atlas-0.1-implementation-log]]:

1. **The agent-isolation test's premise was wrong.** [[impl-atlas-0.1-phase1-scaffold]] expected $f$ to reach $a$ in three message-passing layers via $f\!-\!e\!-\!b\!-\!a$. It never does. The materialized graph carries **only** interface edges, so a signal crossing an agent needs a token that is a boundary token of *two* interfaces — and $e$'s $e\!-\!f$ and $e\!-\!b$ boundary sets are measurably disjoint. Combined with the spec's own ordering (all MP layers, then experts once), nothing traverses an agent's interior during message passing. **Token-graph reachability is not agent-graph reachability**, and that distinction had not been articulated anywhere in the design layer. Measured hops from $f$: $e{=}1$, $g{=}1$, $d{=}2$, $c{=}3$, $b{=}6$, $a$ unreached. The test now asserts a stronger claim — changed-agent set *equals* BFS reach over the materialized graph.
2. **The $d\!-\!g$ interface as written is not on `g`'s boundary**, and the partition implies an **undeclared $d\!-\!f$ interface** (the plume's upstream face is wider than the nozzle exit). [[00-atlas-0.1-implementation-plan]]'s "every edge has a real interface" holds; its converse does not. No edge was added — that is a design decision.
3. Token budget resolved: the plan's "≈1,112" is exactly **1,114**.

Both remaining questions — interleaving experts with message passing, and declaring $d\!-\!f$ — are logged as open action items rather than decided in the build layer. **Noether 1.1's status is unchanged**; Atlas remains a parallel track.

---

## [2026-08-09] note | Atlas 0.1 **Phase 0 built** — solvers green, corpus blocked on compute

The classical-solver layer that is both Atlas's teacher and its judge: 2D compressible NS (MUSCL+minmod, HLLC and Rusanov, viscous fluxes, one-step reaction), Q1 FEM conduction + plane-stress elasticity, US Standard Atmosphere, 3-DOF trajectory, twelve analytic oracles, the LHS sweep with corner-case holdout, and the coupled episode generator with its HDF5 schema. **M1 passes on 10 of 11 oracles.**

Four findings that changed wiki pages, all with root causes in [[atlas-0.1-implementation-log]]:

1. **M2 as specified is unreachable by four orders of magnitude** — measured, not estimated: **820 h per episode, 30.6 years for the 327-episode corpus** on one core. The cause is in the design's own two-timestep table: $\Delta t_{\text{macro}}/\Delta t_{\text{CFL}} \approx 1.5\times10^{5}$, because the CFL limit follows the *acoustic* speed while the flow evolves at the convective one. Three routes out (quasi-steady gas / much shorter episodes / implicit-low-Mach), each of which changes what Atlas learns, so the choice is a design decision and is left open. An 11-episode reduced-fidelity corpus was generated and validated (mass-budget residual $10^{-7}$–$10^{-5}$ against a $10^{-3}$ tolerance) as proof the pipeline runs.
2. **The nozzle area–Mach oracle fails on the declared contour, and it is geometry, not discretization.** Refining made it *worse* (resolution-independent); smoothing the throat corner halved it. The declared contour's hard throat corner curves the sonic line and throws a Prandtl–Meyer fan, so quasi-1D theory does not describe the flow near the throat. The solver is not wrong; the oracle does not apply there.
3. **The injector temperature is pinned by Rayleigh flow, and the obvious value silently deleted the hottest 45% of the sweep.** At 700 K the choking screen rejected 246 of 546 configs, all at the hot end. At 900 K the declared 2200–3400 K range is fully attainable with zero rejections.
4. **Phase 1's `b`/`e` grids were a latent Phase-0 bug**: a bounding box with a masked wall gives the solver a *staircase* nozzle. Both are now body-fitted, token counts unchanged, and `geometry/domains.py` now derives the model's cells *from* the solver's blocks — two independent derivations had already drifted 1.7 mm on an 8 mm shell at the throat.

The one unmet viscous criterion (Blasius 15–19% low) traces to a prescribed-primitive far-field that cannot shed a ~3.5% freestream acceleration; the similarity scaling $\delta\propto\sqrt z$ is reproduced exactly, and a new Stokes shear-layer test grades the viscous flux exactly instead. **Noether 1.1's status is unchanged.**

---

## [2026-08-09] note | Atlas 0.1 **compute & training budget** — the M2 wall priced, and a route past it recommended

Query: *what is the plan of action from Phase 2, given a machine that can neither train nor generate data, and should a vast.ai GPU be rented?* Filed as [[impl-atlas-0.1-compute-and-training-budget]].

The answer inverts the obvious assumption. **Atlas 0.1 is a data-generation problem, not a training one:** ~2,400 CPU-core-hours for the corpus against ~160 GPU-hours for all of Phases 2–4. The model is 22.5 M parameters over ≤29,000 samples — days on one consumer GPU. So renting a GPU now would leave it idle; the first machine to rent is a many-core CPU box, and vast.ai (GPU-priced, interruptible, non-durable disk) is the wrong venue for that leg while being a perfectly good one for the training leg.

Four findings worth carrying:

1. **Only four levers exist**, because the corpus cost is $T_{\text{ep}}/\Delta t_{\text{CFL}}$ — an expression containing neither $\Delta t_{\text{macro}}$ nor $\Delta t_{\text{model}}$. Shorten the episode ($50\times$), coarsen the grid ($r^3$), cheapen the substep (GPU port), or raise $\Delta t_{\text{CFL}}$ (implicit). The first two cost fidelity; only the GPU port costs the design nothing.
2. **Option B read literally is a trap.** "10 s → 10 ms" drags $\Delta t_{\text{model}}$ down with it until the surrogate's step is ~3× the solver's — deleting the speedup claim that motivates the architecture, and collapsing multi-rate subcycling to a no-op. **Route B′** decouples the *snapshot cadence* from the *macro step* (two things the spec conflates): $T_{\text{ep}}=0.2$ s, $\Delta t_{\text{snap}}=10^{-3}$ s, $\Delta t_{\text{macro}}$ unchanged ⇒ 4 macro steps, still 200 snapshots, $50\times$ cheaper, with subcycling and the $3.3\times10^3$ speedup ratio intact. The only loss is long-horizon trajectory — the cheapest available loss, because `rigid_body` is closed-form and no quantity of trajectory data trains it.
3. **Full fidelity is priced once so it stops being re-litigated:** \$5,000–11,000 and 175 days at 64 cores; even a successful GPU port leaves 5,400 GPU-hours. Out of reach under every lever combination short of an implicit scheme.
4. **Phase 2's expert order should not be the spec's order.** `reacting_flow` first — it is the only learned expert with no donor dependency, so it is not blocked on the unresolved Poseidon license question. Then the from-scratch baselines for the other two, which stage 2a's own gate requires anyway; building them second converts the Poseidon bootstrap from a blocker into an upgrade.

Also priced: Phase 3's joint fine-tune is **latency-bound, not FLOP-bound** — 50 sequential forward passes of a 380-token model per macro step, so ~10–20× the FLOP estimate with low GPU utilization that is expected rather than a misconfiguration. And Phase 4's classical baselines cost solver time, so the held-out corner runs should be generated while the CPU box is still rented. Total ≈ \$130–220 and ~9 days of machine time, against a realistic 6–10 week calendar dominated by writing and debugging the experts.

---

## [2026-08-09] lint | Phase 0's "Atlas 0.1 has no dataset" **audited and corrected** — new [[physics-simulation-datasets]]

Query: *is there really no data? There must be combustion datasets. What are the governing equations?* The challenge was correct, and the vault had **no dataset catalog page at all** — a standing gap now filled.

**The claim survives only because of its last clause.** [[impl-atlas-0.1-phase0-scope-and-data]] §1 says nothing public contains the rocket "*as one coupled system with labelled interfaces*"; that conjunct carries all the weight, and the paragraph reads as a much stronger claim than it supports. The architecture is cut by **governing-equation family** ([[expert-library-atlas-0.1]]) and then, one phase later, data availability is assessed **by scenario** — the mismatch that produced the error. Atlas decomposes into only three PDE families plus one closed-form ODE, and agents `a,b,e` and `d,f,g` share an *identical* equation (split by boundary condition, not physics), so one body of public data serves both flow experts.

| Family | Public coverage |
|---|---|
| Compressible flow with shocks | **abundant** — PDEgym (77,840 traj.), The Well, PDEBench |
| Reacting flow / combustion | **exists, and the spec never mentions it** — **BLASTNet 2.0**, 2.2 TB, 744 full-domain samples from 34 DNS |
| External aero over bodies | exists with a regime mismatch — AirfRANS (incompressible subsonic), Poseidon's SE-AF, Flowbench |
| Conduction + thermoelasticity | **genuinely thin** — TFRD and CHT one-offs, all thermal-only; nothing pairs conduction with quasi-static thermal stress |

Consequences, all now recorded on the affected pages rather than only here:

1. **`reacting_flow`'s "from scratch" justification asks about donor *weights* and never about donor *data*.** BLASTNet exists. Flagged in [[impl-atlas-0.1-phase2-experts]] as not-yet-re-evaluated rather than settled.
2. **Atlas already uses public data — laundered through weights.** Poseidon *is* PDEgym. Bootstrapping two experts from it while calling the third "from scratch" is inconsistent once data and weights are seen as the same transfer path.
3. **Two holes are real and permanent.** Confined nozzle flow through a **choking throat** (nothing public; the choked-throat M4 gate depends entirely on generated data), and **interface flux on declared subdomain boundaries** — structurally absent because no other architecture declares interfaces to label. [[pfm-purpose-and-direction]] item 8 already predicted the second. **[AI Inference]:** Atlas's central thesis is therefore untestable on existing benchmarks *by construction* — both the argument for its novelty and the reason its data cost is irreducible.
4. **The generated corpus is re-scoped, not eliminated**: from teaching three governing families to teaching interface behaviour. Three-tier strategy (public pretraining / cheap uncoupled single-agent / coupled with `/iface`) in the new page's §5; [[impl-atlas-0.1-compute-and-training-budget]] revised from 144 to an estimated 60–90 coupled episodes, **pending a with/without-pretraining ablation** rather than cut on the estimate.

The countervailing discipline recorded alongside: "same governing family" ≠ "usable." A four-axis domain-gap discount (boundary conditions, nondimensionalization, dimensionality, regime) applies to every row — PDEgym is *entirely periodic-BC*, and walls are where a nozzle's physics lives. The counter-evidence that the discount is survivable is Poseidon's own narrow-pretraining-still-transfers result across 15 OOD tasks.

---

## [2026-08-09] note | Atlas 0.1 **corpus completion plan** — six decisions, three modules, two new gates

Query: *what do we have to do to complete our training corpora?* Filed as [[impl-atlas-0.1-corpus-completion-plan]].

Almost all of the *code* exists already (solvers, oracles, sweep, schema, coupled generator, an 11-episode validated pilot). What is missing is **six irreversible decisions, three small modules, and a rented machine** — and the discipline that follows from one fact: the corpus is the only artifact that cannot be patched incrementally. A trained expert costs two GPU-hours; the corpus costs ~1,000.

Four findings worth carrying beyond Atlas:

1. **Record more interfaces than you declare.** Open action item 5 ($d\!-\!f$) was filed as a design question; it is also a data question with a wildly asymmetric cost — declaring it after generation means regenerating everything, while recording it and never using it costs a few percent of storage. Recording all geometrically real interfaces with a `declared` flag also makes Phase 4's A4 declared-vs-dense ablation runnable, which it currently is not.
2. **New gate G5 — interface-flux consistency.** The generator couples loosely (Gauss–Seidel, lagged neighbour state); Phase 3 imposes *hard* flux matching. If the lag error exceeds the constraint tolerance, **the constraint fights its own training data**, and the symptom — a conservation loss plateauing at a stubborn floor — points at the model rather than the data. The existing global mass-budget check cannot catch this: it is a whole-system integral, and per-interface errors cancel inside it.
3. **Agent `c` has no CFL limit** — backward Euler conduction plus a once-factorized elastic solve, which is why it never appears in the measured cost table. So `thermostruct`, the expert with the **worst** public-data coverage ([[physics-simulation-datasets]] §3.4), has the **cheapest** data generation: its solo corpus can be full-horizon, full-fidelity, and more diverse than the coupled one, via synthesized boundary-condition sweeps.
4. **Sub-macro snapshots need a stated policy.** At $\Delta t_{\text{snap}} < \Delta t_{\text{macro}}$, agent `c` and the rigid state have not advanced. **Hold-last** is recommended not as a compromise but because it is exactly what the multi-rate stepper does at inference — interpolation would manufacture states the solver never computed and teach the surrogate an artifact.

Also settled: freeze the geometry (round the throat — the one geometry-driven M1 failure, and the corpus bakes it in permanently) and the far-field BC before generating, then **re-run all twelve M1 oracles against the recorded numbers**, not merely for a pass. Revised total ~2,000–2,500 core-hours, ~2 days on 48 cores, ~\$60–100 — the schedule is set by the decisions, not the machine.

---

## [2026-08-09] note | Data augmentation for surrogates — **all2all is a ~$100\times$ win Atlas already licensed and never used**

Query: *does data augmentation not work?* Filed as [[data-augmentation-physics-surrogates]].

It works, but the reframing matters: **augmentation adds zero new physics.** It multiplies along directions where you already know the answer, so the question is whether those directions are the ones you are starved in. For engineering geometries they usually are not.

**What dies, and why.** The Lie point symmetries of compressible NS are properties of the *equation*; training data is a property of the *boundary-value problem*. Translation is broken by the injector face and throat, rotation by the geometry, Galilean boost by walls at rest in the body frame. Reflection survives but is worthless — at $\alpha=0$ the mirror BC already makes the reflected sample bit-identical, and at $\alpha\neq0$ it just yields $-\alpha$. This is the same asymmetry [[physics-simulation-datasets]] §3.5 notes from the other side: **periodic-box benchmarks are far more augmentable than engineering geometries**, which is why augmentation results reported on them transfer poorly to applied surrogates.

**The subtle one, worth generalizing.** Scaling *looks* like free augmentation for compressible flow. It isn't, because **nondimensionalization is a quotient by exactly that group** — two dimensionally-different flows with identical $(\mathrm{Ma},\mathrm{Re},\mathrm{Pr},\gamma)$ are literally the same model input, so rescaling returns duplicates. **[AI Inference]:** the design check is to enumerate the symmetries your representation has already quotiented out and cross them off the augmentation list first. Sample-efficiency gained by good nondimensionalization is augmentation headroom spent; they are the same saving counted once.

**What pays.** [[poseidon-pde-foundation-model]]'s **all2all / semi-group amplification**: for an autonomous PDE every ordered snapshot pair is valid training data given lead-time conditioning, giving $\binom{K}{2}$ pairs instead of $K-1$. [[impl-atlas-0.1-phase2-experts]] §2.1 *already* specifies a $\Delta t$-conditioned expert interface — so Atlas is licensed for this by design and its training spec never invokes it. At 200 snapshots that is 19,900 pairs against 199: **~$100\times$ on the single most expensive artifact in the project.** An unplanned synergy: route B′'s short 0.2 s horizon, chosen purely for cost, also makes the semi-group assumption more nearly exact than the original 10 s episodes would have.

Three limits recorded so the win is not overstated: all2all pairs are heavily correlated, so corpus sizing must still be reasoned in *distinct physical configurations*; lead times must be sampled log-uniformly or the design step size is starved by the many distant pairs; and **patch jittering is specifically unavailable in Atlas** — the log's own body-fitting entry established that solver and tokenizer share cells precisely because interface fluxes do not survive interpolation, and jittering reintroduces it at the worst tokens.

**The general result, and the reason none of this removes the corpus:** every mechanism acts on a solution *field*, while interface supervision is a coupled two-subdomain state. **Augmentation multiplies where you are rich (interiors) and does nothing where you are poor (interfaces)** — the gap [[physics-simulation-datasets]] §4 already established as structural and permanent.

---

## [2026-08-09] note | Phase 2 spec gains **two-stage training**; corpus-generation handoff prompt written

Two edits closing out the session's data thread.

**1. [[impl-atlas-0.1-phase2-experts]] §3.2 now specifies two training stages** rather than one, promoting all2all from a margin note to spec:

- **Stage A — all2all / semi-group operator pretraining.** Every ordered snapshot pair is valid supervision given lead-time conditioning: $\binom{200}{2}=19{,}900$ pairs per episode against 199. Sampling rule is **log-uniform in $\tau$** with the design $\Delta t$ oversampled ~4× — uniform sampling starves the one step size that must work, since distant pairs outnumber adjacent ones ~200:1. Two validity conditions are stated and both checked: lead-time conditioning (satisfied by §2.1's interface) and autonomy (satisfied through FiLM, and *more* nearly exact under route B′ than the original 10 s episodes — flagged for re-examination if long episodes ever return).
- **Stage B — fixed-rate push-forward** at the design $\Delta t_{\text{model}}$, which is what the multi-rate stepper actually executes. ~70/30 budget split, marked [AI Inference].

Three exclusions worth recording: the **elasticity head is excluded from stage A** (quasi-static solve, $\tau$ is meaningless for it — the same root error as applying increment-prediction to it); $\mathcal L_{\text{iface}}$ must be **masked on `t_exch`** so the slow edges' held-constant fluxes are not trained as live signals; and the **conservation projection must not run inside stage A**, being a per-step operation applied to jumps the stepper never takes.

Also recorded: all2all does **not** reduce the episode count — pairs within an episode are heavily correlated and the number of distinct physical regimes is set by the sweep — but it changes the marginal value of the last episode enough that D6's sizing ablation must be run with stage A already enabled.

**2. [[handoff-atlas-0.1-corpus-generation]]** (new, `wiki/misc/`) — a self-contained prompt for a fresh session opened on the *code repo* to execute M2. Carries the architecture thesis, the seven agents and edges, current build state including both unmet oracles and their diagnosed causes, route B′ with an explicit warning against the plain-option-B misreading, D1–D5 as settled decisions, the solo-corpus cost arithmetic, venue guidance, both new acceptance gates, and the log's house rules. Ends by asking the new session to critique the plan before writing code.

---

## [2026-08-18] note | A100 costing for the Atlas 0.1 corpus — new §3.3b in [[impl-atlas-0.1-compute-and-training-budget]]

Query: "where is Atlas 0.1 data generation, what still needs generating, and how long on an A100?" Status read from [[atlas-0.1-implementation-log]] and [[impl-atlas-0.1-corpus-completion-plan]]: Phase 0 code complete, M1 green on 10/11 oracles, **M2 still blocked** — only the 11-episode `full_fidelity: false` proof-of-pipeline corpus exists, and D1–D6 plus modules T3.1–T3.3 remain unwritten.

**The answer worth filing.** The A100 does not run the corpus at all as things stand — `compressible2d.py` is single-threaded numpy, so the measured 16.4 h/episode is a *CPU-core* number and a GPU idles beside it. Ported (lever L3, 1–2 weeks + full M1 revalidation), the arithmetic derived from the log's per-agent table is $1.58\times10^{6}$ substeps and $1.6\times10^{10}$ cell-updates per B′ episode over only 68 736 cells — **too few cells to occupy an A100**, so the port is launch-latency-bound at ~160 µs/substep ⇒ **5–15 min/episode, $65$–$200\times$ one core**, and the whole M2 corpus (T2 + pilot + production + Phase-4 baselines) is **≈20–25 A100-hours**. Batching episodes with `vmap`/`jit` would reach ~45 s/episode (~3 h total), a further 10× for meaningfully more engineering.

**[AI Inference]:** that is *not* faster in calendar time than the already-costed CPU route (~2,000–2,500 core-hours ≈ 2 days on 48 rented cores, \$60–100) once porting is counted — the port is worth doing for L3's other payoff (restoring $T_{\text{ep}} \to 1$–10 s) rather than for M2 itself.

Three findings the page did not previously carry: the solver is **fp64**, so the A100 is the right card for the *solver* leg (9.7 TFLOPS) exactly where §3.4's "4090, not H100" guidance does not transfer — while the 22.5 M-param model leaves it under-used on the *training* leg; **agent `c` should stay on the host** (backward Euler + a once-factorized sparse bordered solve, no CFL limit, seconds per history) so a GPU sparse factorization buys nothing; and fp64→fp32 is the tempting A100 optimization that would quietly move the isentropic-vortex order off its recorded 1.86.

---

## [2026-08-18] note | Atlas 0.1 M2: decisions D1–D6 taken in code, T3 modules built, four plan assumptions corrected

Executed the [[impl-atlas-0.1-corpus-completion-plan]] up to the point where rented compute is needed. All six irreversible decisions are now settled *in the code and recorded in every generated file*, and the three T3 modules exist on the `atlas-0.1` branch. Full details in [[atlas-0.1-implementation-log]].

**Settled.** D1 route B′ with **hold-last** (`dt_snap` decoupled from `dt_macro`; a cadence that does not divide the macro step raises rather than silently straddling it). D2 throat **rounded at $R = 1.25\,h_t$**. D3 characteristic far field **implemented, graded, not adopted**. D4 **10 interfaces recorded, 7 declared**. D5 exchange stamps plus both cadences per interface. D6 corpus size refused as a constant and encoded as named plans with the ablation rule attached.

**Built.** Content-addressed resumable generation (`episode_id` hashes the sweep point, the timing, the decisions *and* the config fingerprint, so a throat-radius change invalidates stale files that still have the right name), atomic staging with checksums, HDF5 schema v2, a `--gates` mode that grades a corpus against M2, and 19 tests.

**The four measurements worth carrying out of the build layer** — each contradicts something a design page asserted:

1. **A rounded throat trades two errors, and 1.25 $h_t$ is the knee.** Area–Mach error over the quasi-1D window falls monotonically with fillet radius (3.73% → 1.91%), but the *exit* error — what the thrust integral depends on — bottoms out near $R = 0.03$ and rises again. The adopted arc keeps its lowest point *at* the declared throat, so throat area, $\dot m$ and every $A_t$-dependent oracle are untouched.
2. **[AI Inference], now falsified.** D3 asserted the Blasius shortfall came from a prescribed far field. Measured: $u_e/u_\infty = 1.033$ under *every* far-field treatment, and making the outflow characteristic too drives $\delta_{99}$ from 0.846 to 0.650. The gap is real and its cause is still unknown — one candidate eliminated, and a specific adoption test (far-field-width invariance) written down in place of the hypothesis.
3. **A teacher should not imitate its student's schedule.** The exchange-cadence rule $\Delta t_{\text{exch}} = \max(\Delta t_p, \Delta t_q)$ belongs to the surrogate's multi-rate stepper; imposing it on the classical solver does not yield a stale *record* of a correct run but a *wrong* run — at 5 ms the plume never sees the nozzle start-up (live/held mass flux $7.4\times$). Generalizable: **when generating training data with a coupled solver, the teacher's coupling should be as tight as the solver allows, and the student's cadence should be recorded as metadata, not enforced on the physics.**
4. **The interface gate was measuring resolution, not lag.** G5's source-vs-destination comparison read 35–120% at the nozzle-exit interface at every cadence, because agent `e` resolves the exit with 96 transverse cells and agent `f` with 8, and the remap *sampled* rather than integrated. A conservative remap takes the coupling error to **exactly zero**. The general lesson, which applies to any multi-domain corpus: **at a non-matching interface, interpolate conservatively or the conservation law you plan to impose later is violated by your own training data** — and split the gate, because one number mixing lag with discretization cannot tell you which to fix.

**Still open, and now stated precisely:** the corpus itself (T3.5–T3.9 need a rented many-core box — see [[impl-atlas-0.1-compute-and-training-budget]]); the Blasius cause; and a new item — agent `f`'s transverse resolution at the exit plane leaves a ~0.47 src-vs-dst jump that Phase 3 will project onto, whose fix moves the token budget and is therefore a design decision, not a build-layer one.

---

## [2026-08-18] note | Atlas 0.1 data generation restructured for a 5-hour session cap; T2 solo tier built and the shell portion generated

Constraint: no more than **5 hours of compute per rented session**, corpus may be small, augmentation is available. Three consequences, all now in code on the `atlas-0.1` branch and written up in [[impl-atlas-0.1-compute-and-training-budget]] §5 and [[impl-atlas-0.1-corpus-completion-plan]] §3.3b.

**1. The unit of work changes.** The corpus stops being one artifact and becomes six **portions**, each sized to fit one session, accumulating into one directory with resume-by-content: `shell` (0.5 core-h), `nozzle` (18), `external` (14), `coupled_a` (105), `coupled_b` (105), `baselines` (53). A session that hits its wall stops cleanly and prints what remains.

**2. The cheap tier goes first, and it is not a compromise.** The T2 solo sweeps total ~33 core-hours and carry two of the three Phase-2 structural gates. The **choked-throat gate cannot be graded on the coupled corpus at all** — a coupled episode has exactly one back pressure, whatever the altitude gave it — so back pressure swept at *fixed chamber pressure* has to come from a solo sweep. Agent `c`, meanwhile, has no CFL limit, so the expert with the worst public-data coverage gets full 10 s histories for half a core-hour. **Generated: 64 shell runs in 5 minutes on 8 workers.**

**3. Coarsening is not the lever; the horizon is.** Measured: 65.6 core-hours per second of simulated flight at coarsen 1, 14.1 at coarsen 2 — **4.7×, not the 8× the cell count promises**, because numpy is overhead-bound at these array sizes. And coarsening would break the solver↔tokenizer cell correspondence that body-fitting exists to guarantee. Cost is linear in the horizon and independent of both timesteps, so the coupled episode shortens to 0.1 s while **keeping 200 snapshots** — snapshot cadence is a storage decision that costs nothing.

**The generalizable point, and the one worth carrying to any small-data surrogate project:** augmentation multiplies *pairs*, never *configurations*. all2all takes 32 coupled episodes from 6,368 consecutive-pair samples to 636,800 lead-time pairs, a real 100× in supervision — and leaves the number of distinct physical regimes at exactly 32. That is why the plan buys a second coupled session rather than a longer first one, and why the corpus reports both numbers instead of the flattering one.

---

## [2026-08-19] note | Alternate Atlas 0.1 case study — domain-decomposed 2D Rayleigh-Benard, zero new data, zero new experts

**Query.** The rocket-ascent first instantiation ([[case-study-rocket-ascent-2d-atlas-0.1]]) is blocked: it needs ~2,400 CPU-core-hours of corpus generation ([[impl-atlas-0.1-compute-and-training-budget]]) and three from-scratch experts. Find a case study that needs neither, whose purpose is to validate the **multi-agent communication framework** in isolation and be extended agent-by-agent by later case studies.

**Answer, filed as [[case-study-rbc-decomposition-atlas-0.1]].** Stop searching for a new *scenario* and cut an existing one: partition 2D Rayleigh-Benard into three stacked agents (hot boundary layer / convective bulk / cold boundary layer) joined by two horizontal typed edges, with the already-trained [[noether-1.0-rbc]] checkpoint frozen and weight-shared across all three expert slots. One expert, used three times; existing `rbc_train.npz` / `rbc_ra_sweep.npz` as ground truth; the built [[impl-atlas-0.1-phase1-scaffold]] reused verbatim.

**The finding that unlocks it — [[physics-simulation-datasets]] §4 is narrower than it reads.** That page concludes interface flux data "genuinely cannot be found." True when the two sides of an interface come from *different solvers*. **False when the cut is internal to one monolithic field**: on a horizontal line through a stored RBC snapshot, $q_n = u_yT-\partial_yT$, $\sigma_{ny}$ and $\phi_m=u_y$ are a numpy expression over an existing `.npz`, not a simulation. Any monolithic PDE corpus converts into a labelled *homogeneous*-interface corpus this way; the heterogeneous case (combustion-structure) keeps §4's cost intact.

**Why RBC specifically, over the incompressible-NS and Poseidon/PDEgym alternatives:** it is the only candidate with a **label-free physical-plausibility gate**. In statistically steady RBC the horizontally-averaged vertical heat flux is height-independent, $\mathrm{Nu}(y)=\langle u_yT\rangle_x-\partial_y\langle T\rangle_x=\text{const}$, so the flux residual across each declared interface is a first-principles identity rather than a fitted label — which finally makes [[conservation-as-constraint-atlas-0.1]] measurable. Paired with a **monolithic gold standard** (the same frozen expert run undivided), which no Atlas test has ever had.

**The central experiment.** The frozen expert has never seen a non-wall $y$-boundary, so agent $\beta$ is doubly out of distribution. R1 = overlapping Schwarz agents with halo tokens (positive control, no OOD boundary at all) vs. R2 = non-overlapping declared flux-BC tokens ([[edge-generation-atlas-0.1]] Mechanism A as built). R2 matching R1 validates the declared-edge contract; R1-only would mean halo exchange is a required part of the framework and would change Mechanism A.

**Also settled for free:** Phase C puts a real expert in the scaffold's slot, making [[atlas-0.1-implementation-log]]'s open action item 4 (interleave experts with message-passing layers, or add an intra-agent mixer?) measurable instead of hypothetical. And the three agents share one equation while occupying different local regimes (parabolic BLs, hyperbolic bulk) — a live test of [[expert-library-atlas-0.1]]'s "condition one expert, don't split it" invariant.

**Scope discipline, stated plainly:** this validates the communication layer with the physics held constant, and proves nothing about heterogeneous expert routing, the MoE gate, or donor transfer. It is the control experiment the rocket case study skipped.

---

## [2026-08-19] note | Buildable engineering case study for Atlas 0.1 — 2D wind-farm wake, no new experts, no new data

**Query, restated.** The first answer ([[case-study-rbc-decomposition-atlas-0.1]]) met the no-new-data / no-new-experts constraints but by *cutting up a synthetic benchmark*. The requirement is a **real engineering system** under the same constraints — the rocket was right in kind, wrong only in that combustion has neither an expert nor a corpus.

**The screen.** Available expert families, exhaustively: incompressible NS (Noether 1.1 P2 checkpoint, Poseidon, Walrus), compressible Euler (Poseidon, Walrus), Boussinesq convection ([[noether-1.0-rbc]]), and closed-form mechanics (no training by definition). Missing: combustion, structural/thermoelastic, electromagnetics, electrochemistry, rarefied. So a buildable engineering scenario must decompose **entirely into fluid families plus closed-form mechanics.** Wildfire fails on combustion *and* radiation; compartment fire is close but runs at $\mathrm{Ra}\sim10^{10}$ against the RBC expert's trained $\le3\times10^5$; aircraft is steady (no rollout) and aeroelasticity needs the missing structural expert; car is 3D and conjugate; battery and reentry each need two missing families.

**Answer, filed as [[case-study-wind-farm-wake-2d-atlas-0.1]].** The offshore wind farm of [[impl-atlas-0.1-phase5-expansion]] needs exactly two new experts — `electromagnetic` (generator) and `wave_structure` (foundation, sea) — and **neither is attached to the physics under test.** Delete them, take the farm onshore, cut a hub-height plan view, and replace the rotor with a **closed-form actuator disk**. What remains is wake-induced power loss in a turbine array: 7 agents (inflow, 2 rotors, near/far/exit wake, bypass corridor), 7 typed edges, and **2 experts — one frozen incompressible-NS checkpoint used five times, one zero-parameter algebraic disk.**

**Why the actuator disk is the point, not a concession.** It is the same move [[expert-library-atlas-0.1]] already made for the rocket's `rigid_body` expert (spectrum level 5, closed form, not learned), and it creates a genuinely **heterogeneous** interface — a PDE expert and an algebraic expert exchanging momentum across a declared edge — which is a scaled copy of the rocket's hardest coupling. It also yields the plan's best gate: the disk asserts $T=2\rho AU_\infty^2a(1-a)$ while the fluid expert independently produces a momentum-deficit flux across the same cut, so **two different experts compute the same number by different physics and must agree.** The residual $r_T$ is the acceptance gate, and it needs no data.

**Physical plausibility comes from closed-form theory, not a dataset:** Betz limit $C_P\le16/27$, the induction relation $U_{\text{disk}}=U_\infty(1-a)$, and the Jensen / Bastankhah-Porte-Agel analytical wake models as a validation corridor. Array efficiency $P_2/P_1$ is reported with an explicit 2D discount (no vortex stretching, inverse cascade) as a sanity band rather than a prediction.

**The extension path is the actual deliverable.** Phase G grows the scenario by adding one agent at a time to a framework that already works: **step 1 is a stratified inflow agent using the already-trained Boussinesq expert — a second case study with still zero new training** — then tower/blades (structural, first genuinely new family), then the generator (EM, restoring Phase 5's original scope), then fatigue (a new mechanism, not a new expert). Phase F's yaw axis makes an edge's *existence* state-dependent, which is the stated precondition for RL edge instantiation ([[edge-generation-atlas-0.1]] Mechanism B).

**Status of the earlier answer.** [[case-study-rbc-decomposition-atlas-0.1]] is retained and demoted to the **single-expert control** — one family, one expert, monolithic gold standard — which isolates communication-layer error more cleanly than any engineering scenario can, and is worth running as the wind farm's Phase-B control. Its §2 finding stands independently: interface flux data is free across a cut internal to a monolithic corpus, and unobtainable only across *heterogeneous* interfaces, narrowing [[physics-simulation-datasets]] §4.

---

---

## [2026-08-19] note + reorg | Wind-farm case study specified in full; Atlas 0.1 folder split into common/ and per-case-study folders

**Reorg.** `concepts/Atlas 0.1/` now separates the architecture from the scenario, which it never did before:

- **`common/`** — [[agent-definition-atlas-0.1]], [[graph-tokenizer-atlas-0.1]], [[edge-generation-atlas-0.1]], [[expert-library-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[conservation-as-constraint-atlas-0.1]], [[global-fields-and-topology-atlas-0.1]], [[training-and-bootstrap-atlas-0.1]]. The test applied: *would this page survive swapping the case study?*
- **`case-study-rocket-ascent/`** — the rocket page plus the whole former `Atlas 0.1 implementation/` folder, renamed to `implementation/`. All ten pages are rocket-specific (geometry, agent table, corpus plan, expert set, budget, build log), which is exactly why they belong under the scenario rather than beside the architecture.
- **`case-study-wind-farm-wake/`**, **`case-study-rbc-decomposition/`** — one folder per remaining case study.

**Zero links changed.** The vault's bare-`[[page-name]]` convention did the work it was adopted for; this is the first reorganization that actually tested it, and it held.

**New page — [[spec-wind-farm-wake-atlas-0.1]].** The binding specification for the wind-farm case study, distinct from [[case-study-wind-farm-wake-2d-atlas-0.1]] which carries the rationale and the phased plan.

**8 agents / 15 edges.** Domain $[-6,18]\times[-4,4]$ in rotor diameters, turbines at $x=0$ and $x=7$: inflow corridor, two rotor *strips* of thickness $0.1D$, near wake ($x\le3$), far wake, exit wake, and two bypass corridors. Three partition decisions carry reasoning rather than convention: the near/far split at $3D$ is the **top-hat → self-similar-Gaussian** regime boundary; the bypass exists as an agent because **wake recovery *is* entrainment from it**, so a decomposition without it holds a permanent deficit; and $B^\pm$ are split rather than one two-component agent, which avoids Atlas's first non-simply-connected agent and buys a free mirror-symmetry diagnostic. The graph has **cycles** — the rocket's was near-chain — making path-dependence of message passing a real concern for the first time.

**The $C_T'$ correction, which is load-bearing.** Freestream-referenced actuator-disk theory is unusable for turbine 2: $U_\infty$ is undefined inside a wake, and using the domain inlet would make turbine 2's thrust independent of the wake it sits in — silently deleting the coupling the case study exists to test. The spec uses the local-induction form $T=\tfrac12\rho A C_T'\langle U_d\rangle^2$, $C_T'=4a/(1-a)$, so each turbine reads its own inflow.

**The four conservation laws and their enforcement — the point of the page.**

1. **Mass, level 5, free.** With $\mathbf u=\nabla^\perp\psi$, flux across a cut equals the *endpoint difference* of $\psi$. So intra-agent divergence vanishes identically, and cross-interface mass conservation collapses to **two agents agreeing on $\psi$ at two endpoints** — a handful of scalars, exact, at no cost. **[AI Inference]:** this generalizes to any 2D incompressible Atlas decomposition and is stronger than what [[conservation-as-constraint-atlas-0.1]] currently claims for the mass channel; promote it there if it survives implementation. Fallback recorded for checkpoints without the optional stream-function head of [[decoder-1.1]] (Leray projection at level 4, or a soft residual at level 2 — the choice must be logged, not left implicit).
2. **Thrust/momentum at the four disk faces, level 5.** The disk side is algebraically exact, so enforcement is a **minimum-norm correction on the fluid side, applied in $\psi$-space** — which preserves div-free identically, since $\nabla\!\cdot\!\nabla^\perp\equiv0$ whatever the correction is. The residual is quadratic in $\psi$, so the multiplier is the root of a scalar quadratic: closed form, one step, no iteration.
3. **Momentum at the eleven fluid–fluid edges — measured, not enforced.** Neither side is closed-form, so there is nothing to project onto, and with frozen experts there is no loss to penalize with either. The page says this plainly rather than implying the flux-matching machinery covers it. The disk edges are what calibrate a "good" residual for the rest.
4. **Betz, level 5, by parameterization.** The disk outputs $a$, never $C_P$, so $C_P\le16/27$ is a property of the formula. It becomes a **canary on the composition**: a measured super-Betz turbine means the projection has a sign error or a mismatched control volume.

**Also specified:** fixed-point relaxation for the algebraic↔PDE two-way coupling ($T$ needs $\langle U_d\rangle$ needs $\mathbf f$ needs $T$), with non-convergence itself a gate; composite boundary conditions including a gust ramp to force unsteadiness; the declared/learned/closed-form ledger; gates W0–W10 with **W4 (pre- *and* post-projection thrust agreement) and W9 (decomposed vs. monolithic) named as the two that matter**; a failure-mode table mapping each symptom to what it would mean for Atlas; and 5 open questions, including whether agents $N$ and $F$ should merge — which is a direct test of [[expert-library-atlas-0.1]]'s cut rule.

**Still true, and stated in the page three times:** $\mathrm{Re}_D\approx7\times10^7$ is not resolved by anything here. The plan view runs at an eddy-viscosity $\mathrm{Re}_{\text{eff}}\in[10^3,10^4]$ supplied as a dial. Wake *recovery rates* are therefore not bankable; interface *consistency* is unaffected, being an identity rather than a correlation.

---

## [2026-08-19] note | Prior-art audit: what in Atlas is actually novel, and the $O(K^2)$ problem that would kill the F1 vision

**Query.** Is the Atlas thesis truly novel? What has been done before, what problem does it solve, and does it expand as more experts are added? Prior-art claims were **checked against sources**, not recalled.

**Filed as [[prior-art-and-novelty-atlas-0.1]] in `common/`.**

**Not novel, stated plainly.** Subdomain decomposition with interface coupling is Schwarz, 1870. Coupling heterogeneous solvers through declared interfaces with mesh mapping, multi-rate stepping and implicit iteration is co-simulation/FMI, and **preCICE** is a mature open implementation of exactly that. **Enforcing flux continuity at subdomain interfaces as the conservation mechanism is cPINN (Jagtap et al., 2020)** — [[conservation-as-constraint-atlas-0.1]] currently reads as if the idea is new and must be corrected. Composing a *library* of pretrained blocks instead of one large pretrain is **CompNO** (Jan 2026). A universal cross-domain interface vocabulary is bond graphs (Paynter, 1959) and port-Hamiltonian theory. One model spanning many PDE families is Poseidon/Walrus/MPP/GPhyT.

**Three things that are open.**
1. **Partitioned-coupling stability theory does not cover neural subsolvers.** Every co-simulation convergence result assumes consistent discretizations with controllable step-size error and a Lipschitz input-output map. A frozen surrogate has none of these — and the structural difference that matters most: **the universal remedy for coupling instability is a smaller communication step, and that remedy is unavailable when the expert was trained at a native $\Delta t$.** Nothing replaces it yet.
2. **Composition, rather than pretraining breadth, as the scaling axis** — with *does composition error grow sub-linearly in interface count?* as the measurable. CompNO composes operators **functionally** (convection ∘ diffusion, one domain); Atlas composes subsystems **spatially** across interfaces. Orthogonal axes; Atlas's question is unanswered because nobody has assembled both a heterogeneous expert library and a framework to compose it in.
3. **Field and lumped experts as first-class peers.** A monolithic PDE foundation model represents fields on grids and has no representation for a torque curve — and no amount of extra pretraining gives it one, because the object is not a field. **This is the strongest form of the F1 argument and the only one that does not depend on speed or accuracy:** a car is a system of coupled field *and* lumped subsystems, and only a composition framework admits both. The wind farm's algebraic disk expert exchanging momentum with a learned PDE expert is that claim in miniature.

**The problem actually being solved is not capability.** Coupled aero-thermal-structural simulation of a race car is routine industrial practice. What changes is **cost and character of one coupled evaluation**: hours → seconds, and non-differentiable → differentiable end-to-end. That is a 4–6 order-of-magnitude change in the unit economics of design search, and **differentiability, not speed, is the enabling property** — a fast non-differentiable surrogate still leaves you doing derivative-free search. Certification does not change: the honest deployment story is *search wide with the composed model, verify the shortlist with the classical stack.*

**The scaling problem, and the fix — the most useful finding.** Atlas's edge types (`heat`, `pressure`, `stress`, `fluid`, `momentum`, `mass`, `shear`, `conservation`) **mix categories** — quantities, phenomena, a medium, and a constraint — and grew by accretion, one case study at a time. If the interface contract stays per-expert-pair, $K$ families need up to $K(K+1)/2$ adapters: 10 at $K{=}4$, but **120–210 at the $K\approx15$–20 an F1 car needs.** That integration burden, not accuracy, is what kills the vision.

The fix is bond-graph **effort–flow power bonds** — force·velocity, pressure·flow, temperature·entropy-flow, voltage·current, chemical-potential·molar-flow. Three consequences: the contract becomes **per-quantity, so $O(K)$ port declarations replace $O(K^2)$ pairwise adapters** and the twentieth expert costs what the fourth did; **energy conservation becomes one global residual computable from coupling values alone** (already standard in power-bond co-simulation), generalizing [[conservation-as-constraint-atlas-0.1]] from a per-edge special case; and it connects to [[backbone-1.1]]'s existing symmetric+skew port-Hamiltonian structure, where composition of passive subsystems is passive by theorem. **[AI Inference]:** relabel now — one editing pass today, near-impossible after ten case studies depend on the current vocabulary.

**Four falsification criteria, recorded in advance** so the goalposts cannot move: super-linear composition error in interface count (measured by the wind farm's Phase-F $N{=}2,3,5,8$ sweep, now the single most important experiment in the plan); frozen experts requiring joint fine-tuning to couple stably (then it is a monolith with extra steps and Walrus wins); adapter engineering exceeding the cost of a monolithic coupled model (**track integration hours per added expert from the first expert, not reconstructed later**); and a monolithic FM reaching the needed coverage first (defended only by the lumped-subsystem argument).

**Seven concrete changes recommended**, including adopting preCICE's conservative-vs-consistent mapping distinction per edge (currently decided implicitly, by accident), replacing plain fixed-point relaxation with quasi-Newton interface acceleration (IQN-ILS) if it needs >6 iterations, and scheduling a **third** case study explicitly as the expert-reuse test — the first two case studies test coupling stability and field+lumped coupling, but neither tests the foundation-model claim itself.

---

## [2026-08-19] note | Effort–flow port algebra adopted across all edge types; end-goal roadmap written

Two new pages in `Atlas 0.1/common/`, and a relabelling pass across every page that names an edge type.

### [[port-algebra-atlas-0.1]] — the interface contract, closed

The old vocabulary (`heat`, `pressure`, `stress`, `fluid`, `momentum`, `mass`, `shear`, `conservation`) mixed **four categories**: conserved quantities, phenomena, a medium, and a constraint. It grew one case study at a time, which is what made it $O(K^2)$: a contract defined per expert-pair needs up to $K(K+1)/2$ adapters — 10 at $K{=}4$, **120–210 at the $K\approx15$–20** the F1 target requires. That integration burden, not accuracy, was the thing that would have ended the program.

Replaced by **five ports, each an effort–flow pair whose product is power** (bond graphs, Paynter 1959; port-Hamiltonian theory): `MECH` $(\boldsymbol\sigma\!\cdot\!\mathbf n,\ \mathbf v)$, `ROT` $(\tau,\omega)$, `THERM` $(T,\ q_n/T)$, `ELEC` $(\phi,\ \mathbf j\!\cdot\!\mathbf n)$, plus the `ADVEC` multiport carrying $\dot m$ with $h_0$ and species as passengers. **The list is closed** — a case study may not add to it without amending that page.

**Four results fell out that were not the point of the exercise:**

1. **`MECH` absorbs four old labels, exactly.** $\boldsymbol\sigma=-p\mathbf I+\boldsymbol\tau$, so `pressure` and `stress` are the isotropic and deviatoric parts of one tensor and `shear` is its tangential component. This **resolves [[spec-wind-farm-wake-atlas-0.1]] open question 1 structurally** rather than by argument, and it independently confirms [[expert-library-atlas-0.1]]'s conclusion — "one expert per label" would have built three experts for one quantity.
2. **The migration is a correction, not a rename.** At an interface with mass crossing it, heat moves *two* ways: conduction (`THERM`) and advected enthalpy (a passenger on `ADVEC`). The single `heat` label conflated them, so **four of the seven rocket edges were silently missing the advective half.** Only $b\!-\!c$, a wall, was right as written.
3. **`conservation` was never a quantity.** It is the *definition* of a port connection ($e_A=e_B$, $f_A=-f_B$), not a label applied to some of them. It becomes a residual **reported at every port**, with a `conservative` flag deciding whether it is also enforced — and the enforce-or-measure criterion is now structural: **hard projection where one side is closed-form, exact-by-construction where a potential representation exists, measurement only where both sides are learned and frozen.** That third row is uncomfortable and is now stated plainly instead of implied away.
4. **Unconnected ports make missing subsystems visible.** The wind-farm rotors declare a `ROT` shaft port with nothing attached — under the old vocabulary the absent generator was simply an absence; as an open port it is a **measurable power flow**. Adding the generator later becomes *connect a port*, not *integrate a subsystem*. This is the mechanism the whole roadmap grows by.

Also adopted: preCICE's **conservative-vs-consistent mapping** distinction, per port (flows map preserving their integral, efforts preserving pointwise values) — Atlas had been making that choice implicitly, per edge, by accident. And a **global power residual $\mathcal R(t)$** computable from coupling values alone, which generalizes [[conservation-as-constraint-atlas-0.1]] from a per-edge special case to one number valid in every case study, now and future.

**Propagated to:** [[edge-generation-atlas-0.1]] (Mechanism A materializes ports), [[conservation-as-constraint-atlas-0.1]] (plus the **cPINN citation it was missing** — flux continuity at subdomain interfaces is Jagtap et al. 2020, not Atlas's idea; the real distinction is jointly-trained vs. frozen-at-inference), [[expert-library-atlas-0.1]], [[00-atlas-0.1-overview]], [[case-study-rocket-ascent-2d-atlas-0.1]], [[case-study-rbc-decomposition-atlas-0.1]], [[case-study-wind-farm-wake-2d-atlas-0.1]], [[spec-wind-farm-wake-atlas-0.1]] (§5 edge table, §6 contract, §7 ledger, new gate W11). The **rocket implementation folder was deliberately left on the old labels** with a migration banner — it is blocked at M2, and rewriting a blocked spec is churn; the banner says apply the migration before unblocking, not after.

### [[f1-pathmap-and-end-goal]] — the end goal, and the ladder

End goal stated once, plainly: **a complete, coupled, differentiable model of a real machine — an F1 car — fast and smooth enough that an optimizer, then an agent, can *search* the design space rather than sample it.** Chosen because it is the densest multiphysics coupling in one object a small team can reason about completely, because its couplings *are* the physics (aero↔ride height, brake heat↔tyre grip↔load path), and above all because it is **field and lumped subsystems inseparably** — the one thing a monolithic PDE foundation model structurally cannot represent, at any scale of pretraining.

Scope fixed so it cannot drift: **not replacing CFD/FEA.** Coupled race-car simulation already exists. What changes is designs-evaluated-per-day and whether gradients exist. **Search wide with the composed model; verify the shortlist with the classical stack.**

**Twelve rungs**, each adding exactly one capability with a gate: scaffold (done) → RBC decomposition → wind farm (field↔lumped) → + stratification (second learned expert, free) → **expert-reuse test** → aerofoil+structure → thermal loop → powertrain (`ELEC`/`ROT`) → tyre/contact → full-vehicle graph → design gradients → optimizer → agent layer. **Rungs 1–4 need no new experts and no new data; rung 5 is where cost returns** — and [[physics-simulation-datasets]] §3.4 already found no public corpus pairing transient conduction with thermoelastic stress, so rung 5 is the underestimated one.

Rung 4 is scheduled **explicitly** because it is the one most likely to be skipped by accident and the only one that tests the foundation-model claim itself. Rung 9 decides everything; the wind farm's $N{=}2,3,5,8$ sweep is its cheap early warning and a bad result there is a reason to stop, not to press on.

**Five falsification criteria recorded in advance.** The underrated one is **F5: differentiable ≠ optimizable** — frozen surrogates can have gradients that are well-defined and useless, dominated by artefacts of the learned representation. **[AI Inference]:** the conservation projections may help by pinning output to a physical manifold and removing spurious gradient directions, but that is speculation until rung 10. Also flagged, and not yet on the ladder: **abstention becomes the load-bearing safety property of the whole vision**, because with 15–20 experts the probability that one is out of distribution approaches 1 on any novel design — and a novel design is the entire point. An optimizer will find and exploit exactly the regions where the model is confidently wrong.

**The program has a floor**, and it is worth stating: even if full-vehicle composition proves unreachable, rungs 1–4 answer *"does composition of frozen neural experts preserve physical validity, and how does error scale with interface count?"* — currently unanswered, and useful either way.

---

## [2026-08-19] note | Wind-farm agent graph drawn; phased implementation guide written; a geometry bug caught by the drawing

Three new pages in `case-study-wind-farm-wake/`, plus an `implementation/` subfolder mirroring the rocket's.

### The figure caught a real bug before any code existed

[[wind-farm-agent-graph-figure]] — a to-scale plan view of the eight agent regions over $[-6,18]\times[-4,4]$, a mermaid port graph, and both schedules in full. A rendered HTML version sits beside it in the folder.

**Drawing the partition to scale exposed two errors in [[spec-wind-farm-wake-atlas-0.1]] §3.** The rotor strips were placed at $\lvert x\rvert\le0.05$, which **overlaps agent $I$** on $x\in[-0.05,0)$; and the bypass corridors were tabulated as $x\in[-6,18]$, which overlaps $I$ on its entire span **and contradicts the page's own edge table**, where $I\!-\!B^\pm$ sits at $x=0$. Fixed: rotors at $x\in[0,0.1]$ and $[7,7.1]$, bypasses at $x\in[0,18]$, with $N$ and $W$ carrying a notch handled by the active-mask channel the Phase-1 scaffold already built for the rocket's agents $b$ and $e$. Left in place this would have fired the scaffold's import-time assertion that every interface curve lies on *both* agents' boundaries — the same class of error the rocket hit at $d\!-\!g$ — or, worse, passed silently and double-counted tokens.

**A consequence worth keeping:** upstream of the first rotor the full height is **one** agent, because there is no wake there to separate from a bypass. The wake/bypass split begins at $x=0$, not at the inlet.

**Topology, now visible:** the graph has **cycles** ($I\to N\to F\to W$ and $I\to B^\pm\to W$ are distinct paths between the same agents), so message passing is path-dependent — the rocket's near-chain graph never tested that. Diameter is 4, so the scaffold's $n_{\text{mp}}{=}4$ carries over. The bypass is load-bearing: delete it and the wake agents have no momentum source, so the deficit becomes permanent. And the two disks are the only heterogeneous nodes — every solid edge touching them is a closed-form expert meeting a learned one.

### [[impl-wind-farm-guide]] — phases W0–W6, written for a cold start

Self-contained enough to be executed by an agent holding only it and the spec. **W0** verify the two ingredients exist (and determine the decoder mode, which decides whether mass conservation is level 5, level 4, or level 2) → **W1** geometry and ports with no model, all assertions → **W2** the zero-parameter disk, unit-tested against Betz to machine precision → **W3 the teacher-forced single-interface go/no-go**, R1 halo vs. R2 flux-BC → **W4** metrics validated on known answers *before* the system that produces them → **W5** couple the graph → **W6** run the sweep. Gates W0–W11, a failure-mode triage table, and a definition of done.

**Three rules stated up front because they override convenience:**

1. **Train nothing.** If a phase seems to need training, the phase or the expert is wrong. Fine-tuning "a little" destroys the entire claim, which is that *independently pretrained frozen experts compose*.
2. **Report pre-projection residuals.** The thrust projection drives $r_T$ to machine zero **by construction**; a post-projection number proves the projection works and nothing else. The result is the residual *before* it. Any table showing only the post-projection number is wrong.
3. **A failed gate stops the phase.** Several of the most valuable outcomes here are negative results.

**The constraint with no classical analogue, now written where a builder will hit it:** in partitioned co-simulation the universal remedy for coupling instability is a smaller communication step. **That remedy is unavailable here** — the expert was trained at a native $\Delta t$ and going below it is out of distribution. The available responses are halo exchange, IQN-ILS interface acceleration, or a negative result.

**Also scheduled rather than deferred:** the $N=2,3,5,8$ scaling sweep, which is the cheap early warning for falsification criterion F1 of [[f1-pathmap-and-end-goal]]. Super-linear composition error there is a reason to stop the program and rethink, not to proceed to the next case study.

**Reporting rules fixed in advance:** always three curves (decomposed / monolithic / analytical corridor); the 2D discount stated every time, with the distinction that internal-consistency gates are *identities* and trustworthy while wake-recovery gates are *correlations* and only corridors; and never claim conservation for the eleven ports that are measured rather than enforced.

### [[wind-farm-implementation-log]]

Opens with three empty ledgers to be filled as the build happens rather than reconstructed: the four standing facts (checkpoint, decoder mode, native $\Delta t$, interface variant), the **integration-hours ledger** for falsification criterion F3, and the gate table.


## [2026-08-20] note | Wind-farm case study — phases W0–W3 built and measured

New page: [[results-w0-w3-wind-farm]]. Build narrative and every number in context appended to [[wind-farm-implementation-log]]. Code on the new branch **`atlas-0.1-windfarm`** of `nonidino/physics-foundation-model` (commit `2645cd0`) — `src/atlas/cases/windfarm/`, `src/atlas/ports/`, `tests/atlas/windfarm/`. **Nothing was trained.**

**The go/no-go passed.** A frozen `camlab-ethz/Poseidon-T`, handed a body force it never saw in training, keeps the streamwise momentum budget closed across a rotor to $r_T=2.6\times10^{-3}$ (R2 flux-BC tokens) and $6.2\times10^{-4}$ (R1 Schwarz halo) against a $5\%$ threshold — teacher-forced, pre-projection, and at the floor a purpose-built spectral Navier–Stokes solver reaches on the identical problem. Falsification criterion **F2** of [[f1-pathmap-and-end-goal]] is not triggered. Both variants passing means the declared-edge contract of [[edge-generation-atlas-0.1]] needs no change; **R2 adopted**.

**Four findings that outrank the headline number.**

1. **The gate is weak.** Run against null models, $r_T<5\%$ rejects a do-nothing operator and nothing else — *rigid advection*, with no pressure, no viscosity and no dynamics of any kind, passes at $9.5\times10^{-3}$. What the W3 pass actually rests on is the field comparison the gate does not constrain: $\langle U_d\rangle$ within $0.8$–$1.1\%$ of a real solver and thrust within $1.6$–$2.1\%$, against $3.0\%$ for advection.
2. **The checkpoint destroys a uniform flow** — an exact steady solution of the periodic incompressible equations at any viscosity. Mean velocity $1.000\to0.969$ in one step, $\to-0.083$ in 1200; its pretraining set is zero-mean by construction and a wind farm is nothing but mean flow. Fixed by an exact Galilean change of frame (evolve the fluctuation in the moving frame, transport the mean by spectral translation), not by a correction term.
3. **A frozen expert has a minimum viable coupling step well above its native lead, and they are different numbers.** Stepping *below* native is bad as [[impl-wind-farm-guide]] §7.2 warns ($18.5\%$ error against $2.0\%$) — but stepping *at* native for 1200 steps is unstable, while $5\times$ native is not. The spec's $t=60$ rollout cannot be run at the spec's $\Delta t$ with this checkpoint. This inverts the guide's own warning and is a constraint on W5/W6 that is now recorded rather than waiting to be discovered.
4. **Window size is not a free parameter.** $\mathrm{Re}_{\text{eff}}=T_s/(\nu_pL^2)$ for a fixed-resolution pretrained operator, so doubling the physical area one window covers quarters the effective Reynolds number. And the effective viscosity is not a single number at all: modes above a cutoff show no decay above the per-step bias, below it decay is clean and exponential — the checkpoint behaves like an LES with a spectral cutoff. **[AI Inference]:** if this generalizes beyond Poseidon it belongs in [[expert-library-atlas-0.1]] as a second, independent constraint on how finely a domain must be cut, alongside the governing-family cut rule.

**Corrections to [[spec-wind-farm-wake-atlas-0.1]]**, both caught by W1's assertions: the agent-graph diameter is **3**, not 4 (the quoted path is not a geodesic; $n_{\text{mp}}=4$ retained as conservative), and the rotor notch is **sub-token** at the spec's own $0.25$ spacing, so agent masks need exact fractional occupancy or the rotor is double-counted. Mass conservation drops from level 5 to **level 4** — scOT has no stream-function head, so the $\psi$-endpoint construction of §7.2 is unavailable.

**A method note worth carrying to the other case studies.** A pseudo-spectral reference solver was added that drives the *identical* probe code, so a difference between it and the expert is a difference between them. It was not in the plan and paid for itself three times: it caught a runaway, it exposed a missing mean pressure gradient at a converged state where a $30\%$ residual could not be blamed on a transient, and it establishes the floor the expert is graded against. Without it, "$2.6\times10^{-3}$" is a number rather than a result — which is [[impl-wind-farm-guide]] §5's own argument, arriving a phase early.


## [2026-08-21] note | HydroGym (*Nature*) assessed against the ladder — RL's third role formalized

New page: [[rl-for-flow-control-and-coordination]], filed under `concepts/ml-building-blocks/`. Prompted by the query *"does HydroGym help Atlas as an agent?"* **The paper was not read in full — the *Nature* text is paywalled; the page is built from the [arXiv:2512.17534](https://arxiv.org/abs/2512.17534) preprint, the repo, and the PMLR version, and says so at the top. It is an assessment, not a summary, and a proper `summaries/` page is still owed.**

**The answer is no on the direct question, and the reason is a vocabulary collision worth fixing.** An Atlas *agent* is a chart; an RL *agent* is a policy. HydroGym's environments **are** solvers — the object an expert exists to replace — and the set contains no actuator-disk or rotor case at all, so the one physical feature [[case-study-wind-farm-wake-2d-atlas-0.1]] is built around is the one it does not have. Nothing in W5–W11 changes. [[agent-definition-atlas-0.1]] should adopt *chart-agent* / *control policy* before any RL page is written.

**Two places it is genuinely useful, both already scheduled.** Rung 4 of [[f1-pathmap-and-end-goal]] — the expert-reuse test the ladder flags as most likely to be skipped by accident — specifies a rung but no test set, and HydroGym is one off the shelf (61 envs, $\mathrm{Re}\,30$ to $4\!\times\!10^5$, fixed API), gradeable with the W4 metrics and no RL involved. Constrained by two W0/W3 findings: the checkpoint takes no forcing input and destroys a uniform mean flow. Second, at rung 10 its JAX backend becomes a differentiable gradient reference. **Not** a replacement for `reference.py` at W5/W6 — the incumbent is built, trusted, and has already caught three bugs.

**The zero-shot result is weaker evidence than it looks.** Channel $\to$ 3D wing at $38\%$ friction reduction is transfer of a *policy* $\pi: s\mapsto a$, which needs only a local sensor-to-actuator map to generalize — not of an operator $\mathcal G_{\Delta t}$, which must reproduce everything the policy is free to ignore. It raises the prior on the shared hypothesis behind [[expert-library-atlas-0.1]]'s by-governing-family bet; it does not measure it. It *is* direct outside corroboration of the deployment story in [[f1-pathmap-and-end-goal]] §1.2.

**The synthesis, and the reason the page exists.** RL had two documented roles in this vault, both deferred with the same trigger (topology via [[edge-generation-atlas-0.1]] Mechanism B, routing via [[unet-hierarchy-atlas-0.1]]); HydroGym supplies no cross-scenario graph experience, so **the trigger is unchanged — not before scenario 2**. The third role was never written down: **a control action is the assignment of exactly one variable of an effort–flow pair at an unconnected port**, with the model supplying the conjugate — which is bond-graph causality assignment, i.e. [[port-algebra-atlas-0.1]] used in the direction it was already built for. The wind farm's two unconnected `ROT` shafts are the existing example. Reward $r_t=\sum_p e_pf_p\Delta t-\lambda\lvert\mathcal R(t)\rvert$.

**[AI Inference]:** the residual term is the strongest architectural argument for Atlas-as-RL-environment. A policy trained against a learned surrogate classically learns to exploit the surrogate's error, and detecting that normally needs an external check; Atlas measures a conservation residual at every port by construction, so exploitation has a **visible signature inside the environment itself**. Falsifiable cheaply at Phase F: run $\lambda=0$ and test whether reward correlates with $\mathcal R(t)$.

**One hard design constraint, recorded now rather than after it bites:** model-side RL (topology, routing) and control-side RL **must never share a policy**. A merged reward lets the controller raise its score by steering the system into a regime its own model predicts well — concretely, yawing the turbines until no wake strikes a downstream rotor, deleting the hardest coupling in the graph and the array power with it.

**Index correction:** page count recounted from disk — 171, not 167; the figure had drifted by 3.


## [2026-08-22] note | Wind-farm W5 — the pressure solve built; C1 reaches machine precision, C2 blocked by a second, independent cause

Implements the fix [[wind-farm-implementation-log]] proposed on 2026-08-21. New module `pressure.py` in the build repo, 21 new tests, no page created — the finding lives in the log and the results page, both updated.

**The bug, stated exactly.** Two spectral operators, both validated on W3's genuinely periodic probe window, both reused at W5 where the domain is not periodic. Graded against an *analytic* Taylor–Green pressure on a non-periodic sub-window, the spectral solve errs by $56\%$ at $n=32$ and **$70\%$ at $n=256$** — the error *grows* under refinement, which is the signature of an inconsistent operator rather than an inaccurate one. That is the difference between a discretization error and the wrong equation, and it is worth having as a diagnostic pattern: **refine the grid; if the error does not fall, the operator is wrong, not coarse.**

**The replacement.** Two Poisson solves diagonalized *exactly* by discrete cosine transforms — DCT-II for Neumann faces, DCT-IV for a Neumann/Dirichlet pair — so they are direct solves at $O(N\log N)$, not iterative ones at a tolerance. Cost: $0.08$ s on a $512\times1536$ domain, **$3\%$ of one expert sweep**, which retires [[impl-wind-farm-guide]]'s budgeting of "a Poisson solve per agent per step" as the expensive fallback. The open **outlet** is the load-bearing part: a Dirichlet face is the only boundary through which the rotors' displaced pressure can leave, and its eigenvalues never vanish, so the projection needs no compatibility condition.

**Result 1 — a conservation law moves from measured to enforced.** [[spec-wind-farm-wake-atlas-0.1]] §7.2 declared level-5 mass conservation unavailable without a stream-function decoder, and W5 reported $L_2\le6.6\times10^{-2}$ as the documented fallback. The assembled field is now divergence free to $\mathbf{5.6\times10^{-14}}$ over the whole domain including its boundary cells. **The spec's $10^{-8}$ is met on the velocity-decoder path by a route the spec did not anticipate** — which is a small but real correction to the enforce-or-measure ledger of [[conservation-as-constraint-atlas-0.1]]: *the decoder's output space bounds what can be enforced pointwise, not what can be enforced globally.*

**Result 2 — the instrument at the rotor now works.** C2 was blocked because the rotor control volume read $19$–$300\times$ the turbine-free floor, so no residual there could be attributed to anything. It now reads $2.5\times$ the floor and carries an explicit error bar (the Neumann solve's compatibility defect, $3.6$–$6.7\%$ of $T$), and one rotor's residual sits *at* that defect — i.e. at the instrument's resolution.

**Result 3 — a prediction failed, and that is the useful part.** The 2026-08-21 entry claimed one diagnosis covered three symptoms and one fix would resolve all three. It resolved one. $P_2/P_1$ moved $1.140\to1.138$; upstream induction moved $17.8\%\to17.3\%$. **[AI Inference]:** the residual induction is plausibly *correct 2D physics* rather than a defect — a 3D actuator disk at $a=1/3$ induces $\sim3.5\%$ at one diameter upstream, but in 2D the rotor is an infinite-span strip at $12.5\%$ channel blockage with far slower decay, and the "should be $10\%$" it was called wrong against is 3D intuition. Cheaply checkable with the existing W3 reference solver; if it holds, W10's $0.4$–$0.8$ array-efficiency band is what needs revising, not the coupling. This is exactly the class of error [[case-study-wind-farm-wake-2d-atlas-0.1]]'s 2D discount exists to catch, applied to a gate threshold rather than to a result.

**Result 4 — the fix exposed a second block, and the two had been hiding each other.** With the pressure correct, enabling the thrust projection destroyed the flow ($\langle U_d\rangle$ at $770\times$ freestream) **while reporting post-projection $r_T=7.5\times10^{-11}$, four orders inside its gate.** Cause: spec §7.3's min-norm correction direction is the Leray projection of a slab indicator, and a field varying only in $x$ and pointing along $x$ is *pure gradient in every mode*, so the projection leaves only $k=0$ — the direction is a **uniform field**. A uniform velocity is a Galilean shift, and a correctly-recovered pressure makes the momentum balance invariant under one ($dg/d\lambda=10^{-6}$ against a residual of $10^{-1}$). **The periodic pressure's inability to carry a mean gradient is precisely what had made that sensitivity look nonzero** — the same missing degree of freedom made the target wrong *and* made the direction appear viable.

**The methodological point, on its third appearance.** W3's $r_T$ could not reject rigid advection; W5's first gate reported PASS on a reversed flow; now a projection graded its own arithmetic to $10^{-11}$ while wrecking the field. **A gate that measures whether a solve converged is not measuring the physics the solve was for.** A guard now refuses degenerate corrections and marks the item BLOCKED rather than passing. Worth promoting out of this case study: every one of the three was caught by a null control or a physicality check bolted on *after* the gate was written, never by the gate itself.


## [2026-08-22] note | Wind-farm — the monolithic 2D baseline built, and an AI inference retracted by measurement

The entry above proposed, as an **[AI Inference]**, that W5's excessive upstream induction might be correct 2D physics graded against 3D theory, and called it cheaply checkable. It was checked, and it is **wrong**. Recorded because a proposed check that gets run is worth more than one that stays proposed, and because the retraction is the useful part.

**Three answers, two of them exact.** A 3D actuator disk induces $3.5\%$ at one diameter upstream (the vortex-cylinder law); a 2D actuator strip induces $9.8\%$, because the 2D wake is bounded by two semi-infinite trailing vortex sheets rather than a closed cylinder. **The inference was right that 2D is nearly $3\times$ stronger.** But 2D Navier–Stokes on the actual domain, with the actual boundary conditions and the actual disk, gives $\mathbf{5.7\%}$ — converged to a drift of $3\times10^{-5}$ and insensitive to halving $\nu$, halving $dx$, and thinning the strip $2.5\times$. The coupled system gives $\mathbf{25.9\%}$. **A $4.5\times$ defect, not physics.** It is also *drifting*: $9.3\%$ at $t=2$ rising monotonically to $25.9\%$ at $t=20$, passing through the correct answer at about $t=1$ — where a steady blockage should equilibrate within one or two convective times, as the baseline does. "Something is accumulating" is a sharper and more findable claim than "the number is too big".

**The baseline exists because the pressure fix made it cheap.** It is [[impl-wind-farm-guide]] §7.3's **W9 monolithic baseline** arriving several phases early — exactly what the W3 entry predicted would happen once a reference solver existed. It reuses the same boundary-condition-aware projection the coupled system now uses, so the *only* difference between it and the coupled run is that the frozen expert is replaced by a solver. That makes it the W9 comparison by construction rather than by arrangement.

**The result that was not the point, and matters more.** In genuine 2D the wake barely recovers: $28.3\%$ deficit still standing sixteen diameters downstream, because 2D has no three-dimensional entrainment to refill it and the channel is closed at the sides. The coupled system has $1.6\%$. **The frozen expert dissipates the wake roughly twenty times too fast** — the per-call dissipation measured at W0 (half-life $1.75$ convective times) arriving in the coupled system exactly as W0 warned it would. So the inverted array efficiency has **two** causes and they push the same way: turbine 1 over-throttled by excess induction, turbine 2 under-throttled by a wake dissipated before it arrives. They are separable, and they live in different places — one in the coupling, one in the checkpoint.

**A second inference refuted by the same run.** The entry above suggested W10's $0.4$–$0.8$ array-efficiency band might have been set from 3D intuition and need revising. The 2D baseline gives $P_2/P_1 = 0.403$, $0.421$, $0.429$ across three configurations — inside the band at its lower edge, which is where a 2D case with a poorly-recovering wake belongs. **The band was right; the coupled system is wrong.**

**[AI Inference]:** the general lesson is about what a decomposed system must be graded against. Both errors here are invisible without a monolithic baseline on the *identical* configuration — $25.9\%$ looks defensible beside a textbook, and $1.6\%$ at sixteen diameters looks like healthy wake recovery to anyone carrying 3D field intuition. Only against the same domain solved undivided do they read as $4.5\times$ and $20\times$. If that holds generally, [[f1-pathmap-and-end-goal]]'s rung structure should treat the monolithic baseline as a **prerequisite for reading any coupled result**, not as a late validation step. The wind farm got away with building it late only because it turned out to be cheap.


## [2026-08-23] note | Wind-farm W6–W11 — the metrics reach the system at last, and W9 clears the decomposition

New page: [[open-problems-atlas-0.1]] (`concepts/Atlas 0.1/common/`), holding problems found by measurement, diagnosed, and deliberately deferred. Four entries so far. Full numbers in [[wind-farm-implementation-log]], entry 2026-08-23.

**Four gates had been blocked on an adapter, not on physics.** W4 built every metric and validated each against a known answer, exactly as [[impl-wind-farm-guide]] §5 demands — and then had no way to point one at the coupled system, because the metrics take an analytic `Field` with a `velocity(x,y)` method and the system produces a lattice. Thirty lines of bilinear sampling unblocked W7, W8, W10 and W11 at once. **The planning lesson is worth keeping**: the guide sequenced metrics before the system precisely to avoid unverifiable numbers, and the cost of that ordering is that the *join* between them belongs to neither phase and gets skipped.

**The adapter was validated before it was trusted, and it caught two bugs that would have reached a gate as physics claims.** First, cell averages are not point values: `body_force` interpolated a lattice field holding area fractions while the quadrature point-samples, giving an $8.6\%$ error in the power ledger's work term — the W4 entry's *"fractional occupancy is not a quadrature weight"* in a second costume, and the fix is the same closed form both times. Second, `d_energy_dt` defaults to zero, which its own docstring says is exact only for steady analytic fields; omitting it on a running system inflated $|\mathcal R|$ to $88\%$ of extracted power.

**Gate W9 is the result, and it exonerates the decomposition.** Three runs differing in one thing each: eight agents; the *same 124 tiles with the agents removed*; and a real 2D Navier–Stokes solver on the same domain.

| pair | isolates | rel-$L_2$ |
|---|---|---|
| $A-B$ | composition error | $\mathbf{0.273}$ |
| $B-C$ | expert error | $\mathbf{1.762}$ |

**Composition error is $6.5\times$ smaller than expert error.** Sharper still at the rotors: the *monolithic* run also inverts the array efficiency ($P_2/P_1 = 1.220$, against the partitioned $1.143$ and the classical solver's $0.404$). **So the case study's most visible defect is not caused by Atlas's decomposition at all** — remove every agent and every port and it persists; only replacing the frozen expert with a solver fixes it. The gate still *fails* as written, because $0.273$ exceeds the expert's own single-step error $50\times$, and both statements are true at once: composition error is large against the gate's yardstick and small against the checkpoint's contribution. §7.3 is vindicated — none of this is sayable without the baseline.

**W6 passes at exactly $16/27$, and the canary fired on the gate script itself** before it fired on anything else: normalizing $C_P$ by $\langle U_d\rangle$ instead of the undisturbed local inflow reported $C_P = 2.0$, which is the identical error W4 records it catching on the viewer four days earlier. A canary that catches the same mistake twice in different code is doing better than a regression test could.

**W11 turns out not to be measurable with this checkpoint, and that is the gate's fault rather than the system's** — new **OP-4**. The instrument was validated first (W4's identity to $10^{-12}$, outflux to $2\times10^{-4}$ and dissipation to $3\times10^{-5}$ on smooth fields), so the $240\times$ miss belongs elsewhere: the ledger's dissipation term needs a single $\nu$, and smoothing the field shows **$81\%$ of its dissipation lives below $0.125\,D$** — exactly the cutoff at which W0 established that this checkpoint has no single $\nu$ and *"behaves like an LES with a spectral cutoff, not a Newtonian fluid"*. A power balance needs a dissipation term; a dissipation term needs a viscosity; this expert does not have one.

**[AI Inference]:** OP-4 generalizes past the wind farm and is the entry most likely to matter later. *Any* frozen expert has some dissipation and none of them ship a documented $\nu$, so a power ledger that **measures its experts' dissipation as an operator property** rather than assuming a fluid one is what [[expert-library-atlas-0.1]] needs for every case study, not just this one. If that holds, [[port-algebra-atlas-0.1]]'s global residual $\mathcal R(t)$ needs a per-expert dissipation term in its *definition* — a change to the framework, not to the wind farm. Cheaply falsifiable: apply an expert to a field, measure the energy it actually removed per call, and check whether that number is stable enough across fields to serve as an operator property.

**Where the case study now stands.** Before this week the summary was *"the framework composes frozen experts stably and conservatively, and does not yet reproduce the physics."* W9 sharpens the second clause: **the physics it fails is failed by the expert, not by the composition.** That is a better position for falsification criterion F2 of [[f1-pathmap-and-end-goal]] than the gate table alone suggests, and a worse one for [[expert-library-atlas-0.1]]'s working assumption that a pretrained checkpoint can be dropped in as an expert without qualification.


## [2026-08-23] note | Composition-layer harness — and the finding that the *windowed architecture*, not the expert, carries most of the error

Two pieces of machinery built together, and a result that **corrects the previous entry's central attribution**. Numbers in [[wind-farm-implementation-log]], entry 2026-08-23; new **OP-5** in [[open-problems-atlas-0.1]].

**Why a solver-backed expert.** W9 put composition error at $0.273$ against an expert error of $1.762$, and **a coupling layer cannot be developed under a $6.5\times$ noise floor**. `SolverExpert` swaps the frozen checkpoint for a spectral solver while holding *everything* else identical — same tiling, window size, per-window periodicity, Galilean framing, explicit impulse, assembly. It preserves a uniform flow exactly, where the checkpoint takes its mean from $1$ to $0.969$. The per-window periodicity is deliberately left in place: fixing it too would confound "learned vs exact" with "periodic vs not".

**Partition-of-unity assembly.** Tiles overlap by $0.5\,D$ (32 cells), so much of the domain has two to four independent expert predictions — and `scatter` kept the nearest one and discarded the rest, so adjacent cells across an ownership boundary came from *different expert calls*. Smoothstep blending (chosen because $S(t)+S(1-t)=1$ identically, making a pairwise overlap a genuine partition rather than a smoothing, and because it is $C^1$) cuts the tile-seam excess from $2.95\times$ the interior to $1.73\times$ — $-48\%$ in the mean, $-56\%$ at p99. Blending is **within an agent only**: blending across agents would drive every port residual to zero by erasing the distinction the framework exists to test.

**A null control corrected the premise.** The measurement that motivated this showed agent-boundary cell pairs at $4.07\times$ the interior, which read as interface error. The *monolithic* field — no agents at all — shows $4.31\times$ at the identical sites. Those boundaries sit on the wake/bypass shear layers and rotor planes, so most of that ratio is real flow structure. Only the tile-seam excess was artefact. **Fourth time in this case study a null control has changed a conclusion rather than confirming one.**

**The finding.** With the harness in place the attribution is clean for the first time. Of the $0.739$ gap in $P_2/P_1$ between the starting configuration ($1.143$) and the correct undivided answer ($0.404$): the **assembly rule** accounts for $19\%$, the **checkpoint** for $15\%$, and **the windowed architecture for $66\%$**. With a *perfect solver* inside every window and **no agent decomposition whatsoever**, $P_2/P_1$ is still $0.907$ — wrong by a factor of $2.2$ with nothing learned anywhere in the loop.

**The correction.** The 2026-08-22 entry read $B-C = 1.762$ as "expert error" and concluded the checkpoint dominates. $B-C$ compares a *tiled, periodic-windowed* run against an undivided solver, so it never was expert error alone — it conflated the checkpoint with the windowing, and the conflation flattered the framework by charging the whole difference to the expert. Separated, the larger share belongs to Atlas. W5 had already recorded the mechanism — *"the frozen expert's input shape has imposed a decomposition finer than the one Atlas declares"* — and this is its price, measured.

**What it changes for the deferred checkpoint decision.** Swapping the expert buys at most $\sim15\%$ of the error. A better checkpoint is worth having and **is not the leverage**. The leverage is in how a fixed-window operator forces the domain to be cut, and the three candidates — per-window periodicity, the $123\times$ macro-step ratio, and one-pass exchange versus Schwarz iteration to convergence — are all framework work needing no new expert.

**[AI Inference]:** the macro-step is the most likely single culprit and it connects to a gap [[prior-art-and-novelty-atlas-0.1]] §2 already named — *partitioned-coupling stability theory assumes error shrinks with the macro step, which fails for an expert trained at a native $\Delta t$*. If a step sweep confirms it, the general statement is that **a frozen expert's native $\Delta t$ sets a floor on composition *accuracy*, not merely on stability**, which is sharper than anything the case study has claimed so far and belongs in [[expert-library-atlas-0.1]] as an expert-selection criterion. Testable directly: `SolverExpert` can step below any native lead, the frozen one cannot.

---

## [2026-08-23] note | The frozen expert is not mirror-equivariant — W7's cause was misattributed

Gate W7 asks for a mirror residual below $10^{-6}$ under symmetric inflow, and its failure was recorded as *path-dependent message passing around the graph cycle*, which [[impl-wind-farm-guide]] §10 predicts in advance. **That attribution is wrong.** Recorded as **OP-6** in [[open-problems-atlas-0.1]]; numbers in [[wind-farm-implementation-log]].

Equivariance is a property of the **operator** before it is a property of the composition. Tested directly: feed `FrozenFluidExpert` an exactly mirror-symmetric field and measure the output's own mirror residual. A **uniform inflow** — the most trivial symmetric field there is, and the one W4 uses as its null control — comes back asymmetric at $2.42\times10^{-2}$ after **one** call. That is $2.4\times10^4$ times the gate, with no graph, no ports, no agents and no tiling anywhere in the loop. Three other symmetric fields give $2.6$–$3.2\times10^{-2}$.

**Systematic, not noise.** Two calls on identical input agree to exactly $0$ — the checkpoint is deterministic. On a closed window the residual grows monotonically, $3.16\times10^{-2}\to1.02\times10^{-1}$ over eight calls, bracketing W7's measured $2.77\times10^{-2}$ at $t=5$ and $7.42\times10^{-2}$ at $t=10$.

**The control that makes it trustworthy.** `SolverExpert` through the identical harness and the identical reflection convention gives $1.06\times10^{-15}$. A wrong mirror convention would have failed the solver too.

**The mechanism is grid-locked.** The antisymmetric part of the output on a *featureless* input is a comb at $32$, $16$ and $8$ cells. The checkpoint's patch size is $4$ and its shifted-window stride is $8$ patches $=32$ cells: the dominant wavelength is exactly the shift stride, the others its harmonics. The asymmetry is keyed to the attention grid, not to the flow.

**[AI Inference]:** the fix belongs in the composition layer, not the expert — a group average $\tilde E(\mathbf u)=\tfrac12[E(\mathbf u)+\mathcal M^{-1}E(\mathcal M\mathbf u)]$ costs $2\times$ the calls, trains nothing, and makes W7 pass by construction. That is a sharper claim for Atlas than any gate result: **the composition layer restoring a symmetry the frozen expert provably lacks.** By Noether it is also the same class of defect as a broken conservation law, so [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule should arguably cover declared symmetries too.

**Fifth time a control has overturned a conclusion here rather than confirming it** — and the first time the overturned conclusion was one the guide predicted in advance.

---

## [2026-08-23] note | Gate W7 passes — the composition layer supplies an equivariance the frozen expert lacks

Built the fix OP-6 called for, and it works: **W7's mirror residual goes from $2.77\times10^{-2}$ and growing to $2.90\times10^{-7}$ and flat**, under the $10^{-6}$ gate. New concept page [[symmetry-averaging-atlas-0.1]]; OP-6 in [[open-problems-atlas-0.1]] marked resolved; numbers in [[wind-farm-implementation-log]].

**The construction.** The Reynolds average over a declared finite group,

$$\tilde E(\mathbf u) = \frac{1}{|G|}\sum_{g\in G} g^{-1}E(g\,\mathbf u)$$

is exactly $G$-equivariant for **any** $E$ — nonlinear, learned, or actively hostile to the symmetry — because substituting $g'=gh$ permutes the sum. That substitution is valid precisely because $G$ is **closed**, which is the whole reason the group axioms are load-bearing rather than decorative: average over a mere *set* and the result is a smoothing that is not equivariant and whose output says nothing about the difference. `SymmetryGroup` therefore checks closure numerically at construction, which also catches an op whose action on fields and on the frame velocity disagree, and one that reindexes without re-signing the normal component.

**Measured.** Isolated, the averaged operator is symmetric **bitwise** — exactly $0$, not machine precision — because on a lattice with symmetric cell centres the reflection is an array reversal and floating-point addition is commutative. In the coupled system, $2.90\times10^{-7}$ at $t=5$, flat from $t\approx1$, for $1.7\times$ the wall time. Mean and divergence untouched, since averaging is linear.

**Why it is not zero, traced rather than tolerated.** The residual enters at the **first scatter**, not the pressure solve. The expert runs in float32 and its output depends on a window's *position in the batch*: the same window at batch sizes $8$ and $40$ differs by $\sim1.3\times10^{-6}$, about $11\varepsilon_{32}$, while an identical batch repeats bitwise. Mirror-paired tiles occupy different slots. **The construction is exact; the floor is arithmetic** — and W7's $10^{-6}$ threshold sits only $\sim2\times$ above it, so the gate is graded near the precision of the expert it grades.

**What it does not do.** The physics is untouched. OP-5's $66\%$ is exactly where it was and W8/W9/W10 fail for the reasons they did. This makes the system symmetric, not right.

**[AI Inference]:** the durable result is the pattern, not the gate. *An invariance a frozen expert provably lacks can be created by the composition layer, exactly, without training* answers a piece of what [[f1-pathmap-and-end-goal]] calls F2 directly — and the gates were not built to elicit it, since they ask whether the composed system matches truth rather than whether composition can **add** a guarantee. It also suggests a design preference for the whole invariant layer: prefer enforcement by averaging over a declared group to enforcement by projection along a chosen direction, since an average cannot be degenerate the way OP-1's direction was. Whether a *continuous* group averaged this way would enforce the Noether-conjugate conserved quantity is open, testable, and would be a much stronger claim than the discrete case.

## [2026-08-23] note | OP-5 qualified — the $66\%$ is a subtraction bucket, and the baseline it is measured against moves

A review of the composition-layer entry above, filed into **OP-5** in [[open-problems-atlas-0.1]]. No new measurement; the *ranking* (windowing $>$ assembly $>$ checkpoint) survives every check below. What does not survive is quoting $66\%$ as a measurement of windowing alone.

**The third factor was never varied.** The $2\times2$ crossed {assembly rule} $\times$ {expert}, and **every one of its four cells is windowed** — by design, so the expert swap would be attributable. So there is no run with windowing off and the rest of the coupled path on; the only windowing-off leg is `ChannelNS`, a different solver with different BCs, timestep and nondimensionalization. "Windowing $=66\%$" is therefore *everything that differs between `SolverExpert`-in-124-windows and `ChannelNS`-undivided*, plus every interaction term. **That is the same shape as the error the entry corrects** — one leg still differs in more than one way.

**The target is not fixed.** The 2026-08-22 baseline sweep concluded it was *"insensitive to everything it should be insensitive to."* True of the **induction** column ($5.7\%$, flat — a potential-flow quantity). **Not true of $P_2/P_1$ in the same table**, which runs $0.246$–$0.429$ across those same runs, a spread of $0.183$ or roughly $25\%$ of the $0.739$ gap being decomposed. The verdict was read off the flat column and applied to the sensitive one; array efficiency is a wake-*survival* measure and has no business being $\nu$-insensitive.

**A Reynolds mismatch may be hiding in the bucket.** $\nu=0.03$ at $D=U_\infty=1$ is $\mathrm{Re}_D\approx33$, against [[spec-wind-farm-wake-atlas-0.1]] §2.3's declared $\mathrm{Re}_{\text{eff}}\in[10^3,10^4]$ and W0's analytic check at $\nu=10^{-3}$ — a factor of $\sim30$. At $dx=0.0625$ that $\nu$ gives a cell Reynolds number of $2.08$, on the central-difference limit, which suggests it was **set by the grid rather than chosen as physics**. Whether `SpectralNS`'s in-window $\nu_p$ maps through the $L=2D,\,U_s=4U_\infty,\,T_s=0.5$ scaling to the same physical $\mathrm{Re}_{\text{eff}}$ is not recorded anywhere. The sign is consistent with a mismatch: the coupled system behaves over-viscous, and more viscosity raises $P_2/P_1$.

**And the split is additive on a cubed quantity.** $P_2/P_1=(U_2/U_1)^3$ exactly at every row. In velocity space the shares are $15/13/73$, not $19/15/66$ — ranking unchanged, digits not metric-invariant.

**What to run first, and it is not the three-way sweep.** The missing piece is a **null control for the windowing factor**: sweep window count $124\to\sim31\to\sim8\to1$ through the identical harness with `SolverExpert`, rescaling $\nu_p$ at each level to hold $\mathrm{Re}_{\text{eff}} = T_s/(\nu_p L^2)$ fixed. Holding $\mathrm{Re}_{\text{eff}}$ constant across window sizes is precisely what owning a solver expert buys — it turns the case study's most-quoted confound ("window size is not a free parameter") into a controlled variable. If $P_2/P_1\to0.404$ as $N\to1$, windowing is confirmed *and* quantified against seam count in one sweep; if it plateaus short, most of the $66\%$ is harness rather than architecture.

**[AI Inference]:** OP-5's candidates (2) macro-step and (3) one-pass exchange may be **one knob, not two**. If `SolverExpert` substeps internally to CFL, $\Delta t=0.25$ is not an integration step but an **exchange interval** — how long a window runs on stale boundary data — and only the *frozen* expert genuinely conflates the two, because its native lead forces them equal. If that holds, the general statement sharpens from "a native $\Delta t$ floors composition accuracy" to *a frozen expert's native $\Delta t$ imposes an exchange interval it cannot refine* — a constraint on **coupling** rather than integration, and a cleaner [[expert-library-atlas-0.1]] selection criterion.

**The pattern worth naming.** Five times a control has overturned a conclusion here, and the composition-layer entry was itself one of those corrections — written and published the same day it was measured. A correction is not automatically better-controlled than what it corrects.

## [2026-08-23] note | Schwarz iteration planned — and the plan argues its own fix is the minor term

New page: [[schwarz-iteration-atlas-0.1]], a **build plan, nothing built**. It is OP-5's candidate (3) worked out properly, and working it out changed what it is for.

**Three separations do the work.**

**(1) The word already means something else here.** [[impl-wind-farm-guide]] §6.2's **R1 "Schwarz halo"** was built, passed W3 at $r_T=6.2\times10^{-4}$, and was *not adopted* — R2 declared flux-BC tokens is in use. R1-vs-R2 asks **what data crosses an interface**; this asks **how many times per macro-step**. Orthogonal, and the adopted contract needs no change. Nothing here reopens W3.

**(2) The machinery already exists; the convergence test is on the wrong quantity.** [[spec-wind-farm-wake-atlas-0.1]] §8.3's thrust fixed point re-runs the fluid sweep — W5 measured **3.02 iterations mean, 5 max, every step** — and is stopped by $|T^{(k+1)}-T^{(k)}|/T<10^{-4}$, **a scalar describing one rotor**. So the field exchange is already iterated ~3×, with a criterion that says nothing about whether interfaces agree. **[AI Inference]:** if so the work is adding an interface residual to an existing test, not building a loop — *verify the fixed point re-runs the full sweep rather than re-evaluating the disk against a cached field before scoping it.*

**(3) The theorem has a precondition this build violates, and that is the result.** A Schwarz fixed point is the global solution **only if each local solve is the exact restriction of the global operator.** A **periodic window is not** — it identifies its own inflow and outflow edges, so imposing distinct neighbour data on them contradicts the operator's own structure. Iterating converges to the fixed point of the *wrong map*: the pieces agree with each other ever more precisely while staying wrong in the same way, **and the interface residual — the only visible symptom — is what iteration removes.** So **OP-5's candidates (1) and (3) are ordered, not parallel**: per-window periodicity is a precondition of iteration helping at all. OP-5 updated accordingly.

**The prediction, made before the build.** Within a macro-step the subdomain problem is an evolution, not an elliptic one — the Schwarz-waveform-relaxation setting, where sufficient overlap gives convergence in a *finite* number of iterations. Every transport mechanism sits inside the overlap: advection $U_\infty\Delta t=0.25\,D$ against a $0.5\,D$ tile overlap, diffusion $\sqrt{\nu\Delta t}\approx0.016\,D$, and **elliptic pressure coupling is already global** via the 2026-08-22 DCT Poisson solve. **[AI Inference]:** iterating tile seams should therefore buy very little, and the discriminating test is that declared **ports** are marginal where seams are comfortable — the band rule fixes $\text{band}=U_\infty\Delta t_{\text{macro}}=0.25\,D$ *exactly on* the one-iteration condition rather than inside it. Effect at ports and not at seams confirms the reasoning; the reverse refutes it.

**Additive, never multiplicative — and the reason is a new kind of constraint.** Sequential sweeps converge faster but are order-dependent, which (a) reinstalls exactly the graph-cycle path dependence the 2026-08-23 W7 reattribution just ruled out, and (b) **breaks [[symmetry-averaging-atlas-0.1]]'s exactness**, because mirror-paired tiles would be visited at different points in the iteration and would no longer see equivalent data. **[AI Inference]:** that is the first instance of two composition-layer guarantees *conflicting*, and it exposes a gap — [[port-algebra-atlas-0.1]] defines what a port carries and says nothing about the order in which ports may be resolved. Additive also batches, which is what the hardware wants ($0.117$ s/window alone, $0.022$ s at batch 32).

**Four stages, each able to end the sequence.** **S0** instruments the interface residual against the *existing* loop and changes nothing — the cheapest decisive measurement OP-5 has, and if the residual has already reached the expert's per-call noise floor ($\sim3\times10^{-2}U_\infty$) by iteration 3, S1–S3 should not be built. **S1** moves the convergence test off the thrust scalar. **S2** gives `SolverExpert` a non-periodic window solve and is the real experiment. **S3** is a coarse space, and only if $k_{\max}$ binds — one-level Schwarz advances information one subdomain per iteration, so $\lceil 7D/1.5D\rceil = 5$ iterations before turbine 1 can reach turbine 2 and 16 to cross the domain, against a loop currently running three.

**[AI Inference]:** S3 notes something the framework has and has never used — **Atlas's 8 declared agents over 124 tiles is already a two-level decomposition**, the $H/h$ hierarchy of a two-level method arrived at for an unrelated reason (the expert's fixed input shape). What is missing is a coarse *solve*, and the frozen expert cannot supply one: coarsening means enlarging $L$, and $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p L^2)$ makes that a change of physics rather than of resolution. A classical coarse solve would work but imports a non-expert solver into the loop — a question for [[expert-library-atlas-0.1]] and [[f1-pathmap-and-end-goal]], not a coding decision.

**[AI Inference]:** if S2 succeeds the general statement is sharper than "iterate the coupling" — *a frozen expert can be composed beyond its native field of view if and only if its local solve is a faithful restriction of the global problem*, which makes **boundary-condition flexibility rather than accuracy the primary expert-selection criterion**, and would mean a checkpoint pretrained only on periodic data is uncomposable at any accuracy.

## [2026-08-23] note | OP-5's fix built — and Schwarz iteration turned out to be provably a no-op without it

[[schwarz-iteration-atlas-0.1]] moves from **plan** to **built**. Stage S0 was answered without being instrumented, stage S1 was **refused as unbuildable**, and stage S2 — the non-periodic window solve — is the only one that turned out to exist. Full record in [[wind-farm-implementation-log]]; **175 existing tests unaffected, 11 new.**

**The proof that reorders the work.** The plan argued that iterating an interface exchange around *periodic* local solves converges to the fixed point of the wrong map. The code says something stronger: **the iteration does not move at all.** `gather` reads each tile's whole $128^2$ window out of the global field at $t^n$ and the operator returns one field, so the sweep is a **constant map in the iteration index** — two passes from the same base agree to `np.array_equal`, bitwise, not to a tolerance. A periodic window has no boundary channel, so additive Schwarz on it is the identity. `CoupleConfig` now refuses `schwarz > 1` unless the windows are non-periodic, quoting the measurement in the error.

**And the related question the plan flagged is settled the same way.** `SpectralNS.step` sub-steps internally to CFL, so for a solver expert $\Delta t = 0.25$ is an **exchange interval, not an integration step** — OP-5's candidates (2) and (3) are **one knob rather than two**, and only the frozen expert conflates them, because its native lead forces the two equal.

**`WindowNS`** supplies the missing channel: 2-D NS on one window with a Dirichlet ring one cell wide, ramped from $t^n$ to the neighbours' iterate while the interior initial condition stays pinned at $t^n$. Validated against exact answers *before* being wired to anything — uniform flow preserved **exactly** ($0.0$; the frozen checkpoint gives $0.969$ and `SpectralNS` passes only by carrying the mean analytically), Taylor–Green second-order convergent, projection divergence $7.6\times10^{-16}$, and a Gaussian wake diffusing within **0.44% of the analytic rate**. That last one is the gate that matters: a local solve which damps would be worse than the defect it was built to remove.

**Two wrong turns worth keeping, because both would have read as small boundary errors.** The projection first inverted the compact five-point Laplacian while *applying* the wide $2h$ one, leaving an interior divergence of $2.5$ instead of zero — the exact mismatch `pressure.wide_eigenvalues_neumann` documents one module over. The obvious repair was worse: a MAC projection's `centre → face → centre` round trip is the filter $[1,2,1]/4$, gain $0.962$ **per sub-step** at a 16-cell wavelength, so $0.962^{51}\approx0.14$ — an 86% loss of exactly the scales the wake lives at, hidden inside a projection, in a case study whose open defect is over-dissipation.

**A composition-layer guarantee needed extending, as §4 predicted.** The transmission ring is a *vector field on the same window*, so [[symmetry-averaging-atlas-0.1]] has to mirror it along with the interior — otherwise the operator gets a boundary condition belonging to the unmirrored problem **and the average still comes back looking symmetric.** Now transformed alongside `force` and `frame`; symmetric input plus symmetric ring returns mirror-symmetric to $10^{-12}$.

**The Reynolds mismatch is confirmed and the target moves.** `CoupledSystem` runs at $\nu=1/255$ ($\mathrm{Re}_D=255$); the $0.404$ target was produced at $\nu=0.03$ ($\mathrm{Re}_D=33$), a cell Reynolds number of $2.08$ — set by the grid, not chosen as physics. Re-running the baseline with $dx$ refined in step: $P_2/P_1 = 0.4044$ at $\mathrm{Re}_D=33$ (reproducing the published number *and* its disk velocities exactly) and $\mathbf{0.2663}$ at $\mathrm{Re}_D=67$. **Doubling Reynolds moves the target by 34%**, and the coupled system sits nearly four times further along that axis. So OP-5's $0.739$ gap is measured against the wrong number, the true gap at matched Reynolds is *larger*, and the correction runs against the framework rather than for it. Both rungs are at $t=20$ and neither is steady, which bounds the claim.

**Cost, and the asymmetry that is the result.** A Dirichlet sweep of 124 windows costs $\sim14\times$ the periodic one, all of it the Galilean frame: the periodic path advects the *fluctuation* (2 sub-steps), the Dirichlet path advects the *flow* (51). **A frozen expert cannot pay this at all, having no ring to impose.** The fix for OP-5's dominant term is available to a solver and unavailable to a checkpoint — an [[expert-library-atlas-0.1]] selection criterion rather than a fact about this code.

**First signal, recorded as a signal.** Rollouts to $t=20$ are running and unfinished. Step 1 is **bitwise identical** between periodic and Dirichlet, which is correct — both preserve a uniform flow exactly, so they cannot differ until structure exists, and that makes any later divergence attributable to the boundary condition alone. At $t=0.5$, the first step where they *can* differ: periodic $P_2/P_1 = 1.0130$, Dirichlet $\mathbf{0.9880}$. **The sign flips immediately** — turbine 2 reads below turbine 1 with a non-periodic window and above it with a periodic one, which is the case study's most visible defect appearing and not appearing under a one-line change of boundary condition. **Two steps into eighty, and $P_2/P_1$ is a $7D$-range quantity that means nothing before $t\gtrsim7$.** The number for the gate ledger is the one at $t=20$, against a target the Reynolds ladder says is no longer $0.404$.

## [2026-08-24] note | OP-5 mechanism hunt: the thrust loop is exonerated, two prior claims are retracted, and the divergence is localised to an inverted upstream induction

Six candidate mechanisms have now been proposed for the wind-farm rollout's divergence and five refuted. This session refuted two of them, withdrew two claims recorded in the previous two sessions, and found the first candidate that survives every constraint. Full record in [[wind-farm-implementation-log]]; artifacts in `results/windfarm/schwarz/op5_mechanism_2026-08-24.json` and `relax_probe.json`.

**The disk–fluid loop is a contraction, and provably so.** Perturbing turbine 1's thrust by $5\%$ and running a full 124-tile sweep gives $S=d\langle U_d\rangle/dT=-0.1702$ at the state where the run was failing, against $-0.1703$ from a uniform field — monotone, near-linear, and **state-independent to four digits**. Because more thrust always means a slower disk, $S<0$, and the relaxed slope $\tfrac12+\tfrac12C_T'\langle U_d\rangle S$ is confined to $[0,\tfrac12)$ at *every* state. It cannot reach $1$. The mechanism is closed by a bound rather than by sampling.

**Two claims withdrawn, both mine, both from over-reading a gate.** The previous entry read the fixed point's failure to converge at $t=1.00$ as evidence that the fluid operator responds non-monotonically to a body force — [[spec-wind-farm-wake-atlas-0.1]] §8.3's stated meaning for that gate, and the most serious outcome the spec anticipates. Re-run with the accelerator off and the cap raised, the same iteration on the same state converges **monotonically in 11 steps** with an observed contraction ratio of $0.410$ against the independently measured $0.413$. The gate fired on arithmetic: a $0.410$ contraction from $\text{rel}=0.357$ is still at $6.9\times10^{-4}$ after seven iterations and the tolerance is $10^{-4}$. **`max_iter = 6` is unreachable by construction for this map**, which makes the spec's "typically 2–4 iterations" wrong about the rate while right about the contraction. The second withdrawal: monotonic energy growth was called numerical injection. It is not. $\langle u\rangle = 1.000000000000$ to machine precision and $\int u\,dy = 8.000000$ at *every* $x$ — mass conservation, mandatory — so $E=\tfrac12(\langle u\rangle^2+\operatorname{Var}u+\operatorname{Var}v)$ and energy growth **is** variance growth at conserved flux.

**A general lesson worth more than either retraction.** A convergence-failure gate does not measure the operator. It measures the operator, the accelerator, the tolerance and the iteration cap together, and fires on whichever is weakest. Reading it as a statement about physics requires exonerating the other three first, and that had not been done. This is the second time in the case study that a gate built to catch a physical defect fired on a numerical one — the pattern, not the instance, is the finding.

**What survives.** The centreline induction upstream of turbine 1 has the **wrong sign**: $-0.1425$ at one diameter upstream — a $14\%$ *excess* over freestream where the monolithic reference gives a $4.7\%$ deficit — while the bypass at $y=+2$ sits at $0.949$, exactly inverting the physical slow-core/fast-bypass pattern. It grows by $\times6.08$, $\times6.18$, $\times6.33$ at $x=-3$, $-2$, $-1.5$ between the two checkpoints: **a common growth factor at three separated stations is one coherent mode, not local noise.** It is rotor-driven (the null control is exactly zero), invisible to the local disk loop, indifferent to transmission condition and Schwarz count, and absent from the monolithic reference — every constraint the four refuted candidates failed.

**Narrowed, not closed.** A single forced window reproduces the *correct* upstream sign, so the inversion belongs to the assembly rather than the local solve. The prime suspect is the macro-step: §8.1 prescribes $\Delta t_{\text{macro}}=0.05$ and the code runs $0.25$, so under impulse forcing a single instantaneous hit removes $\Delta u/\langle U_d\rangle = 0.696$ — **seventy per cent of the disk velocity in one step** — for the pressure solve to redistribute. At the specified step it would be $14\%$. Two runs separate this and neither has been started.

**Process note.** Overlap adequacy and the cross-agent handover at turbine 1 were both refuted at **zero compute cost**, from geometry and from the saved checkpoints. The checkpointing rule adopted the previous session paid for itself immediately: every measurement in this entry except one was made against saved state from a run that had already been killed.

## [2026-08-24] note | OP-5 mechanism found: the window's outflow condition is inconsistent once the outflow carries streamwise structure, and the inconsistency is self-amplifying

The wind-farm rollout's divergence is located. It is not the thrust loop, not the force discretization, not the tiling, not the seams, and not the expert — it is the **transmission condition on the window outlet**. Full record in [[wind-farm-implementation-log]]; artifacts in `results/windfarm/schwarz/op5_mechanism_2026-08-24.json`.

**The test that found it.** Take the last good field, feed one window to `WindowNS`, **switch the rotor force off**, freeze the ring, and vary only the number of sub-steps inside the macro-step. `dirichlet` transmission returns $1.2182$, $1.2182$, $1.2183$ at 48, 190 and 379 sub-steps — converged to four decimals. `characteristic` returns $1.1730$, $3.1044$, $\mathbf{11.8699}$. **Refining the time step makes the answer worse**, which is never a stability problem; it is the signature of a boundary condition inconsistent with the equations. The outlet velocity is what runs away, reversing from $+0.92$ to $-3.74$ within a single macro-step at the finest refinement.

**And it is not zero-gradient outflow as such.** A clean $x$-invariant synthetic wake is stable under *both* transmissions and converges to six digits. A zero normal gradient is **exact** when the outflow is $x$-invariant. A window sitting $1.5\,D$ downstream of a disk has an outflow that is still developing, so the condition is wrong there in proportion to the streamwise structure crossing it — **and the error it commits is itself streamwise structure**, which makes the next application more wrong. The refinement sensitivity tracks the pathology: exactly stable on a uniform field, mild at $t=0.5$, NaN by $t=1.0$.

**It explains every result that had not made sense.** Why the null control passed *exactly* — rotors off gives an $x$-invariant outflow, where the condition is exact, so the control was passed trivially and could never have caught this. Why more Schwarz iterations made things worse — each sweep is a full set of sub-steps, so `schwarz=2` integrates an inconsistent boundary twice as long between ring refreshes. Why the spec's $\Delta t=0.05$ looked better — fewer sub-steps per refresh, less drift before the reset. And why both transmissions failed differently: `dirichlet` is stable and traps the wake, `characteristic` releases the wake and is inconsistent. **Neither is a usable open boundary.**

**A retraction, and the shape of the mistake.** The 2026-08-23 entry that adopted `characteristic` recorded Taylor–Green errors of $1.40\times10^{-1}$, $1.18\times10^{-1}$, $1.01\times10^{-1}$ under it — "barely improving with resolution, which is the correct signature of an *inconsistent boundary condition*" — and then argued the wind farm was exempt because its outflow is "nearly $x$-invariant." **The diagnosis was right; the exemption was wrong.** The evidence was measured, correctly interpreted, labelled as such, and then reasoned past by an argument about the intended configuration instead of a test on it. The test that catches it costs seconds and was available the same day. That pattern — correct measurement, correct reading, excused rather than checked — is the more portable finding.

**Scoreboard.** Seven mechanisms proposed across four sessions, six refuted, one confirmed. Two claims of mine retracted this session on top of the one above: the fixed point's non-convergence was arithmetic (a $0.410$ contraction cannot reach $10^{-4}$ in six iterations), not a non-monotone operator; and the monotone energy growth was variance at conserved flux, not numerical injection. Every one of these was overturned by a control costing seconds against a rollout costing hours.

**The fix this points at** is a genuine open boundary — a convective (Orlanski) outflow, $\partial_t u + U_c \partial_x u = 0$, consistent for structure advecting out rather than assuming there is none. Recorded before attempting it.

## [2026-08-24] note | OP-5: the divergence is fixed and the inverted induction is not — an open-boundary condition was being applied to 117 interior tile seams, and the measurement that justified it used a synthetic ring

The wind-farm divergence is fixed, and the fix is a deletion of scope rather than a new numerical scheme. Full record in [[wind-farm-implementation-log]].

**The defect.** An open-boundary condition is a statement about the **domain**. The window solve applied one to whichever faces of *any* tile the flow happened to leave through — interior seams included. On a seam the neighbour's ring is not merely available, it is the right answer, which is the whole content of classical Schwarz; extrapolating over it discards good data and substitutes a condition that is inconsistent wherever streamwise structure crosses. **Only 7 of the 124 tiles have a face on the real outlet.** The other 117 were being handed an open boundary on interior seams. Declaring the faces geometrically, an interior tile goes from a sub-step spread of $10.70$ (and NaN at one state) to $0.0000$, converged at every state, with both open conditions collapsing onto the `dirichlet` control — which is what they should always have reduced to there.

**The retraction, and it is the one that started everything.** `dirichlet` transmission was abandoned on 2026-08-23 because *"with the ring pinned, the wake is forced back to freestream at the outflow cells."* That test built the ring as `np.ones(...)` — **a synthetic uniform freestream**, which is the domain-outlet situation and no interior seam. What an interior tile is actually handed, read from the assembled field at the state that diverged, is a ring carrying $0.9744$ on its outflow face: the wake. `dirichlet` reproduces it to five digits and is sub-step converged. **A pinned ring never trapped the wake on an interior window.** The divergence, three killed rollouts and four refuted mechanisms all descend from a control whose boundary data was unrepresentative of the windows it was generalised to.

**Two failed repairs, kept.** A convective (Orlanski) outflow — the remedy the previous entry named — was built and is *worse* on its own ($28.2$ against $11.9$ at 379 sub-steps), because `characteristic`'s per-sub-step overwrite was also erasing the bias left by the flux patch. A shape-preserving replacement for that patch then recovered two states of three, and instrumenting it showed the scale factor sitting at $0.9996$–$1.0006$ — so the flux patch had never been the driver either. Both repairs were aimed at the wrong object, and both are recorded as such.

**The generalisable finding.** Three times in four sessions a measurement was taken correctly, interpreted correctly, and then applied to a configuration it was not a measurement of. The Taylor–Green error was correctly labelled "the signature of an inconsistent boundary condition" and excused by an argument about the intended flow. The wake-trapping test was correct for a freestream ring and generalised to seams. The convergence gate was read as a statement about the fluid when it measures the operator, the accelerator, the tolerance and the cap together. **In each case the cheap check existed and was skipped in favour of an argument.** The regression test now carries this too: the first synthetic field written for it was too gentle to reproduce the bug at all ($2.9\times10^{-4}$ spread), and only a sharp streamwise ramp sitting on the face under test shows it ($4.4\times10^{-2}$, a factor of $354$) — the same blind spot as the original single-window checks, now written down inside the test that would have inherited it.

**Status, now measured.** 176 existing tests unaffected, 2 added (178 total). Resuming from the state one step before the `rhs` rollout died and changing nothing but the open-face declaration: $t=1.50$ gives $P_2/P_1 = 1.0216$ where the unfixed run gave **NaN**, and $t=1.75$ gives $0.9522$ — the first time any non-periodic rollout has put turbine 2 *below* turbine 1, which is the physically correct ordering. **The blow-up is prevented.** The inverted induction is **not**: $-0.0985$ one diameter upstream against a monolithic $+0.0473$, still a jet and still an order of magnitude out. It fell on the last step at all four upstream stations, alongside energy and $\max u$ — the first decrease in this case study — but that is *one step*, and two numbers quoted from early steps have already been withdrawn this session. What is on the record is the blow-up, not the recovery.

**And the safeguard collided with itself.** The checkpoint directory was keyed on the configuration tag rather than the output name, so the verification run overwrote the `last_good` of the rollout it had just resumed from. Nothing irreplaceable was lost and it is now keyed on `--out`, but a rule about *keeping* artifacts turned out to need a companion rule about *not colliding* on them, and only the first had been written down.

## [2026-08-24] note | the induction is recovering — eight monotone steps and a sign change — and the rollouts get a measured case for a GPU rather than an assumed one

Two things settled in one session, and they are independent. Full record in [[wind-farm-implementation-log]]; the machine-side procedure is [[vast-ai-windfarm-runbook]].

**The physics.** The entry above said the open-face fix prevented the blow-up and that whether the inverted induction recovered was *"not yet a measurement."* Eight macro-steps later it is. $P_2/P_1$ falls monotonically from $1.0216$ to $\mathbf{0.8144}$ with the rate of decrease *growing* ($-0.0025, -0.0174, -0.0197, -0.0221, -0.0237$), and the upstream induction moves toward the reference at all four stations on every step. At $x=-0.5$ it has **changed sign** — $-0.0546$ at $t=1.50$ to $+0.0013$ at $t=3.50$ — which is the first correctly-signed upstream induction anywhere in this case study's coupled runs. The fixed point converged in 3 iterations throughout and mass closed at $4.3 \times 10^{-14}$.

**And it is still wrong.** $0.8144$ against a monolithic $0.403$–$0.429$; $-0.0418$ against $+0.0473$ one diameter upstream. The downstream deficit falls to $0.0\%$ by $x=16$ where the 2-D reference still holds $28.3\%$, so **OP-3 is untouched and plainly visible**. What is established is that the trajectory heads toward the reference from a state that used to diverge — not where it lands. **[AI Inference]** that the remaining gap may be transient, flagged as a hypothesis precisely because the *identical* reasoning was applied to the periodic rollout in W9 and was wrong: that one drifted monotonically to $25.9\%$ instead of settling. A monotone trend has already fooled this case study once from exactly this position.

**Under-claiming paid.** The previous entry could have called the recovery on four indicators turning together and would have been right; it said "one step" instead, at a cost of one paragraph. The two numbers withdrawn earlier this session ($0.9880$, $0.777$) were both cases where that caution was available and skipped.

**The compute.** "Rent a GPU" was measured before it was done. Profiling one macro-step puts **95.2%** of $182$ s in `WindowNS`'s batched finite-difference stencils on a $[124,128,128]$ float64 array and **0.1%** in the global pressure solve — the piece that *looks* expensive, being global and serial, is free. The cheap alternative was then tested and **refuted**: a thread pool over the batch saturates at $2.81\times$ on 8 threads and gets *worse* at 22, which is a memory-bandwidth wall rather than a core shortage. So a bigger CPU box buys almost nothing, and a GPU's advantage here is specific and quantifiable — bandwidth on a 5-point stencil, not FLOPs. The recommendation is a measurement, which is the standard this case study has had to learn to hold.

**Ported so that it could be verified without a GPU.** There is none on this machine, so `WindowNS` is written once against a small backend interface and torch runs the identical code on CPU: the equivalence suite pins torch-CPU against numpy across all three transmissions, the `open_faces` mask, ramped rings and forcing, with the tolerance calibrated against a $10^{-6}$ change in $\nu$ so it cannot pass vacuously. `device='cuda'` is the only thing left untested off the box, and a bring-up script closes it there before any hours are spent. Two defects surfaced that way: the DCT is now a matmul against a matrix built by scipy itself rather than a hand-rolled FFT reordering (a $\sqrt{2}$ slip in the $k=0$ row alone would have produced a plausible pressure field with a wrong constant mode, inside a solve already singular in that mode), and the body force was left as numpy while the fields became tensors — which works on CPU through numpy's interop protocol and **raises on CUDA**, so it would have passed every test here and failed on the rented box. It was caught only because a `DeprecationWarning` was read instead of ignored.

**A third safeguard failure, and it is the same shape as the last two.** The generic upload path reuses the training sync's hardening rather than re-deriving it, and extracting that shared path exposed the hazard: the in-flight lock is **coalescing**, so leaking it is silent — every later upload reports "previous upload still in flight" and skips, and the run keeps going and saves nothing, which is exactly what the sync exists to prevent. There is now one path every early return must take, and its regression test was **verified to fail when the fix is removed** rather than assumed to cover it. Three sessions running, the safeguards have failed in ways the safeguards could not see.

**Status.** 229 passed, 1 skipped (the 38 added are 12 backend-equivalence tests and 26 publish tests). 7 failures are pre-existing and environmental — the `windfarm` extra (`transformers` + scOT) is not installed in this venv — **confirmed by running them in a clean worktree at the previous commit, where they fail identically**, rather than by arguing that they looked unrelated.

---

## [2026-08-24] note | the wind-farm case study has its answer: composing a frozen expert inverts the array efficiency, and the architecture around it passes W10

Measured on a rented RTX 5090. Full record in [[results-w6-w11-wind-farm]]; machine-side procedure in [[vast-ai-windfarm-runbook]].

**The answer.** Composing a frozen pretrained checkpoint through declared ports gives $P_2/P_1 = 1.0162$ — turbine 2 out-producing turbine 1 — against a classical 2-D reference of $0.404$, with upstream induction $5.6\times$ the reference and $74\%$ of the wake destroyed before the second turbine. **Negative, and it is the result rather than a setback.**

**The architecture is not what breaks it.** Replace the expert with a solver, apply the 2026-08-24 open-face fix, and the *same* graph, tiles, ports and assembly give $\mathbf{0.7663}$ — **gate W10 passes for the first time** — with upstream induction of $+0.0389$ against the reference's $+0.0473$: correct sign, within $18\%$. OP-2, open since W5, is effectively closed for the solver path.

**The two rows cannot be made to differ in one variable, and that is the finding.** The frozen row is also periodic-windowed and impulse-forced, because a one-shot field-in/field-out operator has nowhere to accept a boundary ring — so the repair that produced the middle row is *structurally unavailable* to a frozen expert. `CoupleConfig` refuses the combination rather than ignoring it. **[AI Inference]:** the constraint is the interface, not the weights. An expert exposing even a one-cell ring as an input would be eligible for the fix that moved $P_2/P_1$ from $1.02$ to $0.77$. Untested — no such expert was run.

**A gate that turned out to be no test at all, caught by the cheap check.** W8 grades the far wake against 3-D Jensen/BPA corridors. Run on the *classical reference* — the known-correct answer — it scores $\mathbf{0/6}$ stations inside the corridor, because a 2-D wake recovers slower than any 3-D model allows. [[impl-wind-farm-guide]] §9.2 predicted precisely this and the check confirmed it rather than the argument being trusted; reporting "W8 corridor: FAIL" for the system would have flagged it for failing something the right answer also fails. Regraded against the 2-D reference, the test discriminates cleanly: mean deficit retained is $0.62\times$ (solver) and $0.26\times$ (frozen), which is **OP-3 quantified against a valid reference** instead of a per-call estimate. W8's *shape* half is valid and the reference passes it ($0.033$ collapse spread) while both system configurations fail ($0.127$, $0.350$).

**W9 fails and W11 is closed unmeasurable.** Composition error $0.3293$ against the expert's own single-step error $0.0055$ — declaring agents and exchanging ports costs sixty times one expert call. Recorded as a framework result. W11's power residual needs a single $\nu$ and $81\%$ of the dissipation lives below the scale where one exists (OP-4), so it is closed as not measurable and **no figure is quoted anywhere**, per [[conservation-as-constraint-atlas-0.1]]'s enforce-or-measure rule.

**Compute.** The whole set is about five minutes of GPU; it was six hours of CPU that morning. The window solve was ported to torch after profiling showed $95.2\%$ of a macro-step in batched stencils and a CPU thread pool saturating at $2.81\times$ — a bandwidth wall, not a core shortage. On the 5090 `step_batch` runs $33.8\times$ faster than numpy and agrees to $4\times10^{-13}$; the full coupled system reproduced the CPU result exactly. No GitHub credential was placed on the rented box: source went over as a `git archive`, results came back by `scp`.

**Still open.** The $N=2,3,5,8$ scaling curve does not exist and is **not a flag** — the agent graph hardcodes `R1`/`R2` with fixed interfaces, priority order and domain extent, so it needs `geometry.py` parametrised and a new monolithic baseline per $N$. It is the largest remaining piece and the one most directly about the framework rather than this case study.

---

## [2026-08-25] note | the composition layer scales: tripling the agent graph moves the answer by under 1%, and the farm-level error grows for a different reason

The item the previous entry called the largest remaining piece is closed. Full record in [[results-n-sweep-wind-farm]].

**The framework result.** $N = 2,3,5,8$ takes the graph from 8 agents and 15 declared interfaces to **26 and 63**, and $P_2/P_1$ reads $0.7663 / 0.7588 / 0.7581 / 0.7584$ — **under $1\%$ of movement while the graph triples.** Every run finite, gate W1 passing at every $N$, the Schwarz fixed point converging in 3 iterations throughout. **Composition error is bounded in graph size rather than accumulating in it.** That is the most load-bearing claim this case study can make about the architecture, and it is now measured instead of assumed.

**And the farm-level number still degrades, for a reason that is not that.** Array efficiency overpredicts by $1.258\times$ at $N=2$ rising to $\mathbf{1.793\times}$ at $N=8$. Not because composition worsens — per-turbine error is flat — but because the **plateau**, the regime where the expert is wrong, occupies a growing share of the array: one turbine of two, then seven of eight. A fixed per-unit error applied to more units. The two readings come from one measurement and conflating them would have turned a clean framework result into a muddy one.

**Both curves have the same shape, which is the part worth noticing.** Drop at $T_2$, plateau from $T_3$, small upturn at the last turbine — the deep-array asymptote, reproduced qualitatively by the composed system without being told to. **The trailing upturn appears in the classical reference too** ($0.3437 \to 0.3446$), so it is the domain's outlet and not the composition layer. That was one cheap check standing between a real observation and a fabricated artefact, and this case study has now been saved by that same check twice — the first was W8's corridor.

**Where it is still wrong.** At the $N=8$ plateau the system retains $0.2593/0.6566 = \mathbf{0.395}$ of the reference's momentum deficit: **under 40%**, OP-3 at farm scale, consistent in direction with W8's $0.62\times$ centreline figure and worse because the plateau gives a too-fast-recovering wake the most room to be wrong.

**A defect caught only because $N=2$ is a guard and not a data point.** `CoupleConfig.force_mode` defaults to `'impulse'`; the run behind the committed $0.7663$ used `--force-mode rhs`. The sweep script did not set it. Every $N$ would still have produced a plausible, internally consistent, monotone curve — **of a configuration the case study's result is not about.** Nothing would have looked wrong. It surfaced because the $N=2$ row has to reproduce $0.7663$ before the other rows are allowed to mean anything, which is the only reason to keep re-running a number already known.

**Under-claiming paid again, and so did over-verifying.** $N=2$ was pinned three independent ways before the generalisation was trusted: structural graph identity, **bit-for-bit** field equality against `456ec7e` (`max|diff| = 0.0` on the full $512\times1536$ $u$ and $v$, not "within tolerance"), and $t=20$ end-to-end returning $0.7663$ to four decimals. The third is what caught `force_mode`; the first two would both have passed with it wrong.

**Compute.** $20$ minutes of GPU for the sweep, $4$ for the regression, and the four classical references took **$36$ seconds total** — they are a single explicit-projection grid and were run *concurrently* on the box's idle CPU rather than after. Whole rental about $40$ minutes, **≈ \$0.23**. Source over `git archive` and `scp`; no credential on rented hardware; instance destroyed on completion.

**Still open.** The frozen-expert path was **not** swept — only the solver configuration — so nothing here says how a frozen checkpoint scales. Spacing fixed at $7D$, single aligned row, uniform inflow: the offsets, yaw and staggering where real array-efficiency questions live are all untouched.

## [2026-08-26] note | agreement is an operator, not a scalar condition — the worst-agent error bound is a theorem with three hypotheses, and the one it names carries the least weight

Query: *if the agents agree perfectly and everything is conserved across the interaction graph, is the simulation error on the order of the worst agent's independent accuracy?* Filed as [[composition-error-theory]] — version-independent, so the filename carries no Atlas revision suffix.

**The answer is no, and the three reasons are separate mechanisms rather than three views of one.** (1) **Agreement is a consistency condition, not a correctness one.** A converged composition does not approximately solve the intended problem; it *exactly solves a nearby one*, and the distance between them is not controlled by local accuracy. Worse, the failure is silent by construction — the interface residual is the only symptom, and agreement is defined as driving it to zero. This vault already holds the extreme case: under periodic windows the sweep is a **constant map in the iteration index**, two passes agreeing *bitwise*, so the residual is identically zero while $P_2/P_1$ is wrong by $2\times$. (2) **Conservation constrains finitely many linear functionals of an infinite-dimensional error.** Every flux-neutral redistribution — a wake decaying too fast, a displaced profile with the right integral — lies in its kernel. Mass was *enforced* at $4.3\times10^{-14}$ and bought nothing on the quantity of interest. (3) **Errors amplify and transport.** Per-step defects accumulate as $(L^n-1)/(L-1)$, and along an advective chain agent $i$'s error is agent $j$'s boundary data, so the honest aggregator is an adjoint-weighted sum. The same $\varepsilon = 10^{-2}$ yields $10^{-2}$, $8\times10^{-2}$ or $0.5$ depending on which of the three you ask.

**The correct bound, and the substitution the conjecture makes without saying so.** $\lVert u-u^\star\rVert \le C_S(\max_i \tau_i + \gamma)$ — Lax-shaped: consistency plus stability gives convergence, agreement alone gives neither. The conjecture is this with $\gamma=0$ (fair), $C_S=O(1)$ (a hypothesis), and $\tau_i=\varepsilon_i$ — **and that last one is false.** $\varepsilon_i$ is benchmark accuracy on the agent's own distribution; $\tau_i$ is the defect on the true global solution's restriction under the boundary data its *neighbours* impose. **Composition is a distribution-shift machine by construction**, and the vault has the sharpest possible instance already recorded: a frozen expert was *structurally ineligible* for the boundary fix that moved $P_2/P_1$ from $1.02$ to $0.77$, because a field-in/field-out operator has nowhere to put a boundary ring. That is $\tau$ undefined while $\varepsilon$ is excellent.

**Graded against what has actually been measured.** H1 faithfulness — **violated, and measured**. H2 stability — **half-cleared**: [[results-n-sweep-wind-farm]] tripled the graph and moved the answer under $1\%$, so the amplification factor is $O(1)$ *in graph size*; it has never been established *in time*, and OP-2's monotone drift is the counter-signal. H3 exact agreement — achievable, and the least valuable. **The hypothesis the conjecture names is the one that carries the least weight**, which is why tightening agreement has never been where the leverage was.

**The reframing that makes the question constructive.** The exact interface condition is not "match the values" but the **Steklov–Poincaré (Dirichlet-to-Neumann) operator**; composing with it reproduces the global solution exactly, and *there* the worst-agent bound is a theorem. It is global and nonlocal, so every practical transmission condition approximates it, and they form a ladder from Dirichlet through Robin and Ventcell. **Composition accuracy is set by how well the operator is approximated, not by how tightly the approximation is satisfied.** Atlas sits on the leftmost rung — a one-cell Dirichlet ring — and then iterates it to convergence, which is precision applied to the wrong quantity.

**Four constructions, and three of them close open problems that are already logged.** **(1) Passivity rather than conservation** — a storage function plus a dissipation inequality per expert, composed through the Dirac interconnection the port algebra *already is*. Plain passivity bounds the state; **incremental** passivity is what bounds the error, and the distinction is easy to lose. The measurable defect $\pi_i$ composes **additively** and needs **no viscosity**, which *derives* OP-4's recommended fix instead of choosing it — OP-4 is stuck precisely because the dissipation term needs a single $\nu$ the checkpoint does not have. **[AI Inference]** it also predicts OP-2: a per-call passivity defect integrates to a monotone accumulation rather than a wrong steady state, which is OP-2's unexplained signature. **(2) Weak mortar matching with an inf-sup constant $\beta$** — this is the conjecture *as a theorem*, and $\beta$ is the hypothesis it omits; it also says pointwise effort-and-flow matching between two inexact operators is the over-constrained rung, generically infeasible. **(3) Optimized Robin/Ventcell transmission** — the principled form of the Orlanski outflow OP-2 already names as its remedy. **(4) Adjoint/DWR localization** — attribution per agent and per seam *by construction*, in one backward pass, instead of OP-5's subtraction buckets, which that entry itself records were contaminated by a Reynolds-number mismatch. [[prior-art-and-novelty-atlas-0.1]] already claims end-to-end differentiability as a distinguishing property, and end-to-end differentiability **is** the adjoint: the machinery for the strongest available error-localization method is a headline feature spent on nothing.

**What was already right and is now load-bearing.** [[port-algebra-atlas-0.1]] §2 asserts in one line that passivity composes, and never uses it. The connection rule $e_A=e_B$, $f_A=-f_B$ **is** a power-preserving interconnection — the hard structural half is built. What is missing is a storage function per expert, which is one declaration. It also answers [[schwarz-iteration-atlas-0.1]] §4's named gap, that the port algebra says nothing about resolution *order*: an ordering is admissible iff it preserves the power-preserving interconnection, which additive sweeps do and sequential ones do not — the same reason multiplicative Schwarz breaks symmetry averaging's exactness.

**Five falsifiable measurements are on the page, cheapest first**, and two are independent and immediate: the frozen checkpoint's passivity defect (hours, no training) and $\tau$ against $\varepsilon$ for one expert (one rollout, against a classical reference that already exists). W9's composition error $0.3293$ against a single-step $0.0055$ is a $60\times$ hint at the second before it is run.

**Under-claiming note.** The literature results invoked — port-Hamiltonian composition, mortar optimality under inf-sup, optimized Schwarz, dual-weighted residual — are established for linear, variationally-posed problems with exact local solves. [[prior-art-and-novelty-atlas-0.1]] §2 already records that **a frozen neural surrogate is none of those**, so every application of them here is marked **[AI Inference]** on the page and is a guide to what to measure rather than a guarantee to quote.

## [2026-08-26] note | the agreement operator can be built outside the expert — probing a black box for its DtN map turns four constructions into one assembled matrix

Query: *a better way to make the independent agents agree, from a theoretical perspective.* Filed as [[probed-dtn-coupling]] — version-independent, so no Atlas revision suffix, following [[composition-error-theory]]'s precedent.

**The starting point was a contradiction in this vault's own prescription.** [[composition-error-theory]] §4.0 established that composition accuracy is set by $\lVert\Lambda_i-\tilde\Lambda_i\rVert$ — how well the interface operator approximates Steklov–Poincaré — and not by how tightly the approximation is satisfied. It then prescribed climbing a ladder toward it: Robin, Ventcell, optimized transmission conditions. **Every rung of that ladder asks something of the inside of the expert, and this vault has twice measured that frozen experts cannot supply it.** A Robin condition needs a flux-plus-value combination on the boundary, and a field-in/field-out operator was already recorded as *structurally ineligible* for even a Dirichlet ring. A coarse space needs enlarging $L$, and $\mathrm{Re}_{\text{eff}}=T_s/(\nu_p L^2)$ means that changes the physics rather than the resolution — [[schwarz-iteration-atlas-0.1]] §S3 marked it *do not build without settling*. So the framework's own diagnosis prescribed three fixes and the agents it is built around were ineligible for all three.

**The observation that dissolves it is almost trivial once stated, which is the reason to state it.** Dirichlet-to-Neumann is *by definition* Dirichlet-in, Neumann-out. An expert that merely **accepts** a boundary ring and returns a field **already implements $\Lambda_i$** — it has simply never been asked for the answer in that form. You cannot feed it a Robin condition, and you do not need to: with $\Lambda_i$ in hand, every rung above Dirichlet is a linear-algebraic rearrangement performed **in the composition layer**, on data the expert has already returned. **The agreement operator does not have to live inside the agent; it has to live between them** — which is exactly [[symmetry-averaging-atlas-0.1]]'s move, the composition layer supplying a property the expert lacks, applied now to the interface condition. Assembly is $m+1$ Dirichlet solves per block, and `reference.WindowNS` — validated to $0.44\%$ of analytic on wake diffusion before it was wired — makes it runnable today with nothing trained.

**It is the coarse space §S3 could not build, and that is the strongest single argument.** [[schwarz-iteration-atlas-0.1]] §3.2 records a disqualification it has no answer to: one-level additive Schwarz advances information one overlap-connected neighbour per iteration, so at $2D$ windows with $0.5D$ overlap it takes $\lceil 7D/1.5D\rceil = 5$ iterations for turbine 1 to reach turbine 2 and $16$ to cross the domain, against a loop running three. **A Schur complement is dense across the whole interface**, so the $7D$ signal crosses **in one solve**, at any $N$ — and because every call is a native-resolution call on a native-sized window, $L$ never changes and §S3's $\mathrm{Re}_{\text{eff}}$ blocker never arises. The two-level structure the vault noticed it already had — 8 declared agents over 124 fine tiles — becomes usable. **Under-claimed deliberately:** the classical $\kappa \le CH^{-2}(1+H/\delta)$ estimates assume linear, SPD, variationally-posed problems with exact local solves, which [[prior-art-and-novelty-atlas-0.1]] §2 already records a frozen neural surrogate is none of; the claim made is only about the *sparsity pattern*, which holds regardless.

**Then four of [[composition-error-theory]] §4's constructions collapse into one measurement, and this is the part that changes the programme's ordering.** Incremental passivity — which that page correctly distinguishes from plain passivity as the thing that bounds *error* rather than *state* — is exactly **positive-realness of the probed matrix**, $\tfrac12(S_i+S_i^\top)\succeq0$, because the port pairing is power by construction. So H2 becomes an **eigenvalue computation on a matrix already assembled**, and the eigenvector **names which interface mode is being amplified**, which nothing in the vault currently does. $\beta=\sigma_{\min}(S)$ and $\kappa(S)$ become printable per port, generalizing OP-1's expensively-learned conditioning lesson from the thrust constraint to every seam. The optimal Robin $\alpha^\star_k$ is the **measured diagonal** in a Fourier basis — and this is the only route that exists, because OP-4 establishes the checkpoint has no documented $\nu$ and therefore no symbol to Fourier-analyze, closing every analytic path. And $\tau$ against $\varepsilon$ — [[composition-error-theory]]'s load-bearing claim — becomes $\lVert\Lambda^{\text{expert}}-\Lambda^{\text{ref}}\rVert$, **with no rollout to contaminate it**, which does not repeat the OP-5 subtraction-bucket failure where two runs differed in more than one variable. **Assemble first, and passivity, inf-sup and Robin become reports rather than projects.**

**[[port-algebra-atlas-0.1]] turns out to have specified the probe already.** `MECH`'s effort–flow pair $(\boldsymbol\sigma\!\cdot\!\mathbf n,\ \mathbf v)$, read in the order flow-in/effort-out, **is** $\Lambda_i$ — right units, right sign convention, every port type in the closed vocabulary. What the page has never done is treat the pair as an *operator to be identified* rather than two numbers to be matched. This also disposes of the construction's weakest detail: reading $\partial_n u$ off a one-cell ring by one-sided differencing is noisy, but reading the conservative momentum flux is not, and that is the `MECH` effort the port already declares.

**A composability index that returns exactly zero on the case already proved uncomposable.** $\Xi_i = \lVert\Lambda^{\text{expert}}\rVert/\lVert\Lambda^{\text{ref}}\rVert$ turns [[expert-library-atlas-0.1]]'s selection criterion from a yes/no into a norm. A periodic window has no boundary channel, so $\partial\mathcal E_i/\partial\lambda \equiv 0$ — which is precisely the bitwise `np.array_equal` result [[schwarz-iteration-atlas-0.1]] §3.1 measured, restated as an operator. The consequence is **stronger than "the sweep is the identity"**: $S\equiv0$ means the interface problem is not ill-conditioned but **empty** — every trace is equally consistent, because no agent's output depends on any agent's input. That the index scores exactly zero there is the construction's positive control, and it should be run first despite being the least informative measurement.

**The probe floor is OP-6's, not W0's, and getting this backwards would have killed the idea on paper.** The obvious objection is that the checkpoint's noise floor is $0.031\,U_\infty$ and finite differences divide by $\epsilon$. But W0's figure is **systematic, state-dependent model error, not call-to-call randomness** — the same input returns the same output bitwise — and a finite difference **subtracts it off**. What survives is genuine non-reproducibility, and the vault has measured exactly one source: **OP-6's $10^{-6}$ dependence of the forward pass on a window's position in the batch**, previously logged only as an irritating float32 floor under gate W7. At $\epsilon = 10^{-2}U_\infty$ that gives $\sim10^{-4}$, four orders below signal — and it is *removable*, since pinning batch layout across a probe pair leaves the ring as the only difference. Better still, a **JVP** needs no $\epsilon$ at all: **the second unspent use of the end-to-end differentiability [[prior-art-and-novelty-atlas-0.1]] claims as a headline property**, the first being the adjoint that [[composition-error-theory]] §4.4 already noted is claimed and never taken.

**Honest about cost, and it is the main risk.** Naive assembly is $\approx 82$ min per macro-step against a $\sim24$ s step — **$200\times$, unaffordable as stated** — from a Dirichlet sweep at $14\times$ the periodic one. Two reductions bring it to roughly $2$–$3\times$: **equivalence classes** (an aligned row of identical turbines has interior agents that are translates, and $|G|=2$ is already exploited) and **refresh every $K$ steps** rather than rebuilding, which [[results-n-sweep-wind-farm]]'s insensitivity result makes plausible and measurement 2 exists to test. These are estimates from recorded timings, **not measurements**, and both soft spots are named on the page.

**Honest about nonlinearity.** For Navier–Stokes $\Lambda_i$ is not an operator; a probe recovers the tangent map, and the method is **Newton–Krylov–Schur** — nonlinear substructuring, the FETI-DP/BDDC family. Every metric above is therefore **local to a state** and must be reported with the state it was measured at. One thing improves rather than degrades: a Schur solve is **inherently order-free**, so it satisfies [[schwarz-iteration-atlas-0.1]] §4's additive requirement by construction and does not reopen the conflict that page flagged between two composition-layer guarantees.

**What it does not fix, stated on the page rather than discovered later.** It attacks $\gamma$ and $C_S$ and leaves $\tau_i$ untouched, so OP-5's finding that the **windowed architecture carries 66%** of the error is unaddressed. It diagnoses a bad expert precisely and repairs none of it. It does not supersede *enforce, measure, or decline*. And it is unavailable to the frozen checkpoint — $\Lambda\equiv0$ means there is nothing to probe — which is the same eligibility boundary §S2 already drew.

**Six falsifiable measurements are on the page, cheapest first**, and three are independent and immediate: assemble one `WindowNS` block and check the null space is exactly one-dimensional (a wrong dimension indicts the probe, not the physics, and is caught before anything is built on it); $\Xi$ for the frozen checkpoint, which should be exactly zero; and whether **OP-2's drift projects onto a negative passivity mode**, which would diagnose without a rollout a problem that has resisted rollout-based analysis — or eliminate passivity as its cause just as cheaply.

## [2026-08-26] lint | four pages carried LaTeX mangled into control characters by an earlier write; index.md's newest table row was broken across five lines

Found while positioning the [[probed-dtn-coupling]] entry, and worth recording because the failure is **silent in a text editor and invisible to `grep`**. An earlier write processed backslash escapes in LaTeX source, so `\tau` became a literal TAB plus `au`, `\rVert` became a newline plus `Vert`, `\varepsilon` a vertical tab, `\beta` a backspace, `\nu` a newline. A scan of all `.md` files for `[\t\r\v\f\b\a]` found **four affected**, and no others:

| file | damage | fix |
|---|---|---|
| `index.md` | the [[composition-error-theory]] row was **followed by four orphaned fragments** — a shorter duplicate of its own tail, split across lines 103–106 by interpreted `\r` and `\n`. The row itself was intact and complete | fragments deleted after verifying every phrase in them appears in the intact row; **no unique content lost** |
| `spec-wind-farm-wake-atlas-0.1` | `\boldsymbol` → backspace + `oldsymbol`, breaking the traction-vector equation in the `MECH` discussion | restored |
| `pfm-concept-overview` | stray leading TAB before `### 6. Discover Sci...`, which renders the heading as a **code block** | removed |
| `normalization-scheme` | stray leading TAB before the body text under `## The question`, same rendering failure | removed |

**The lesson is about the writing path, not the content.** All four came from LaTeX passed through a layer that interpreted escapes; the fix for future writes is to author page content through a path that does no escape processing, and the check is one scan for control characters. **The same bug was then reproduced once while inserting the new index row** — the row was written through a shell heredoc into a Python string literal, `\t`/`\n`/`\v`/`\b` were interpreted again, and the row split into five lines. It was backed out and the row was instead written to a file verbatim and spliced in. **A failure mode that recurs the moment you stop thinking about it is worth a written rule**, and the rule is: never put LaTeX inside a shell or Python string literal.

Verification after repair: **zero control characters across all `.md` files in the vault**, and the new row is a single line with exactly four pipes.

## [2026-08-26] note | the rollout bound: an autoregressive monolith obeys the identical error law, so the case for decomposition is an inequality about $L$ — plus multi-step agreement as multiple shooting, and why conservation is free and bounds nothing

Query: *(1) per-step error is agent error plus conservation/communication error and grows exponentially — but doesn't an autoregressive model do the same? (2) over longer rollouts, can agreement be enforced across several steps rather than one? (3) how is conservation/communication error actually enforced?* Filed as [[temporal-error-accumulation]].

**(1) Yes, identically — and that is the most useful fact available here.** With $L=\mathrm{Lip}(\Phi)$ the recursion $\lVert e^{n+1}\rVert\le L\lVert e^n\rVert+\tau^{n+1}+\gamma^{n+1}$ closes to $\lVert e^N\rVert\le L^N\lVert e^0\rVert+\sum_n L^{N-n}(\tau^n+\gamma^n)$, and a monolithic autoregressive model is the **same expression with $\gamma\equiv0$**. So no argument for Atlas may rest on composed systems having a better error *law*; they have the same law with different constants, and the page exists partly to stop that argument being made.

**Two corrections to the premise, both load-bearing.** Growth is **exponential only in one of three regimes** — $\delta/(1-L)$ *bounded uniformly in $N$* for $L<1$, $N\delta$ linear at $L=1$, $L^N\delta$ exponential above — and calling it exponential builds the bad case into the premise when the bad case is the one thing on the page that is a **design choice**. Dissipative Navier–Stokes is contractive on its attractor in the energy norm, so a local solve that inherits the physics' dissipativity inherits $L\le1$ with it: **error does not accumulate because time passes, it accumulates because the map is expansive.** Second, the agent term is $\tau$ (consistency defect under neighbour-imposed boundary data), not the benchmark $\varepsilon$ — [[composition-error-theory]] §3.2, and W9's $0.3293$ against a single-step $0.0055$ is the $60\times$ instance.

**The architecture's thesis, restated as the inequality it actually is.** Decomposition pays iff $L_{\text{comp}}<L_{\text{mono}}$ by enough to cover $\gamma$. It can, for two reasons: $L$ **factors** as $L_{\mathcal A}\cdot\max_i L_i$ with the coupling half **designed rather than trained**, and **incremental passivity composes** through the port algebra's Dirac interconnection, giving $L\le1$ for any graph at any $N$ with no global analysis. **A monolithic network has no per-part handle on its Lipschitz constant.** Against it: composition is a distribution-shift machine, so there are $N$ shift problems instead of one, plus the interfaces. **[AI Inference]** OP-2 is best read as a measurement that the composed map is *marginally expansive*, $L=1+\eta$ — a monotone accumulation is exactly that signature — and the reportable quantity is $\eta$, not the drift magnitude at a chosen $t$.

**And the vault's one measurement on the comparison points the same way.** The monolithic run gives $P_2/P_1=0.907$ against the partitioned $0.889$ — **removing every agent made it slightly worse** — while OP-5 attributes $66\%$ to the windowed architecture and $15\%$ to the checkpoint. On this problem $\tau$ dominates and $\gamma$ is not binding: *agent decomposition is not the defect, the local solve is.*

**(2) Yes, and it is the strongest of the three ideas. It is Schwarz waveform relaxation / multiple shooting**, and the interface unknown is promoted from a trace to a trajectory $\lambda(t)$. Three mechanisms, and the obvious one is not the important one. **(A)** Solving a window as one coupled problem reduces the *count* of $\gamma$ injections by a factor $W$ — real, but conditional on the window problem still converging to the same tolerance. **(B)** SWR on a bounded window converges **superlinearly** rather than linearly, and **finitely** for advection with finite propagation speed — but the rate **degrades as the window lengthens**, so this pushes toward *short* windows while (A) pushes toward long ones. **(C) The real payoff: windowing converts a forward march into a simultaneous solve**, so the multiple-shooting Jacobian's conditioning is governed by $L^W$ — the $W$-step propagator — **and not $L^N$**. That is why multiple shooting is the standard tool for unstable trajectories, and it applies here unchanged: it converts an exponential in the rollout length into an exponential in **a length you choose**. Design rule $W^\star\approx\min(1/\ln L,\ W_{\text{SWR}})$; at [[composition-error-theory]]'s own worked $L=1.05$, $W\approx20$ cuts the method's contribution from $1.05^{80}\approx49$ to $1.05^{20}\approx2.7$.

**Prediction recorded before running: windowing buys little on the current build.** Mechanism (A) helps only where $\gamma$ is a meaningful share of the defect, and OP-5 says it is not — the same *faithfulness-is-a-precondition-not-a-peer* ordering [[schwarz-iteration-atlas-0.1]] §5.3 established for iteration, applied to windowing for the identical reason. Mechanism (C) is the exception, since it acts on $L$ rather than $\gamma$. **One structural point strongly in windowing's favour:** a frozen expert *cannot refine below its native $\Delta t$* — recorded as a constraint on coupling — but multi-step agreement asks for the **opposite direction**, a coarser exchange interval, which it supplies by simply being called $W$ times. **Windowing is the one axis on the page a frozen expert can move along.** **[AI Inference]** it also makes the passivity story honest rather than complicating it: over a window the probed $\Lambda_i$ becomes a **transfer function** $\Lambda_i(s)$, and incremental passivity *is* positive-realness for $\mathrm{Re}(s)>0$ (the positive-real lemma) — so [[probed-dtn-coupling]]'s matrix condition is that statement at one frequency, and a probe sweeping window length is a **frequency sweep of the expert**, saying at which timescales it is non-passive.

**(3) Conservation and agreement are different constraints, and only one is free.** Conservation is about **sums** ($\oint_\Gamma(f_A+f_B)=0$); agreement is about **values** ($e_A=e_B$ pointwise). [[port-algebra-atlas-0.1]]'s connection rule asks for **both**, and demanding both pointwise between two inexact operators is the over-constrained rung. **Conservation is enforced structurally and for free by a single-valued numerical flux** — compute the face flux once, hand the same number to both sides with opposite signs — after which it is an identity of the data structure with no residual to monitor. The wind farm's $4.3\times10^{-14}$ is *machine epsilon*, not the output of an enforcement step. **And it bought nothing**: the answer was still $2\times$ wrong, because a conservation law kills $m\sim10$–$100$ linear functionals of an error living in $10^5$–$10^7$ dimensions, and every flux-neutral redistribution is in its kernel. **Enforce it because it is free and because $\mathcal R\neq0$ is an excellent falsifier; never read $\mathcal R=0$ as a bound.**

**Agreement is not enforced at all — it is solved for, and that is $\gamma$.** The mechanisms are iteration (Schwarz), a direct interface solve ([[probed-dtn-coupling]]), or either of those over a window. It is **the cheapest constant to drive to zero and the least valuable to have driven there**, because doing so removes the only symptom of an unfaithful local solve — the vault holds the extreme case where $\gamma\equiv0$ *bitwise* while the answer was $2\times$ wrong. A four-row ladder is tabulated with what each mechanism guarantees: single-valued flux (exact, free, bounds nothing), min-norm projection (subject to **OP-1's conditioning trap** — a degenerate correction direction is a lie reported as a tiny residual), weak/mortar enforcement (a real bound, with constant $1/\beta$), and the dissipation inequality. **Only the last bounds $\lVert e\rVert$ rather than finitely many functionals of it, and the reason is structural: conservation and agreement act on the per-step defect $\delta$, which the bound multiplies by $L^{N-n}$, while passivity acts on $L^{N-n}$ itself.**

**The ordering that falls out is the inverse of the intuitive one.** Intuition prioritizes agreement, then conservation, then stability. The bound prioritizes **stability ($L$), then faithfulness ($\tau$), then agreement ($\gamma$)** — and every vault measurement that overlaps the bound agrees with it. **[AI Inference]** the single highest value-per-hour measurement now available anywhere in the Atlas programme is **fitting $L$ from one paired rollout**: perturb a converged state, roll both trajectories, fit $\lVert e^n\rVert$ against $L^n$. It decides which of the three terms deserves the next session, it tests whether OP-2's drift rate matches the fitted $\eta$, and **every stability claim the framework makes is currently unfalsifiable without it.**

## [2026-08-26] lint | [[autoregressive-rollout-stability]] claimed rollout error grows as $\epsilon^k$ — corrected

Found while grounding [[temporal-error-accumulation]] §2.1, which needed that page as its point of comparison. Its §1 read *"if the one-step error is $\epsilon$, after $k$ steps the error grows approximately as $\epsilon^k$ for stable systems."* **For $\epsilon<1$ that expression decays** — it would say a stable system forgets accumulated error faster than it commits new error, which inverts the phenomenon the page is about. It also contradicted [[composition-error-theory]] §2.3, which states the same recursion correctly, so the vault held both versions simultaneously.

Replaced with the standard result $\lVert e^k\rVert\le\frac{L^k-1}{L-1}\epsilon$ and its three regimes, with the chaotic case recovered as $L\approx e^{\lambda\Delta t}$ so the page's existing Lyapunov statement still follows. **The governing quantity is $L$, not $\epsilon$** — and the correction matters beyond bookkeeping, because the whole argument of [[temporal-error-accumulation]] is that $L\le1$ is attainable by construction while nothing similar is true of $\epsilon$. A page whose headline formula said otherwise would have made the composed-vs-monolithic comparison unreadable.

## [2026-08-26] note | the per-step defect is three terms, not two — the master bound, the coupling scheme as a compiled declaration, and ten gaps to generality

Query: *work out the full theoretical formulation of the error bound; generalize the Schwarz mechanism so it can be implemented for any case; then, what else is needed to generalize the framework to any system given proper agents?* Filed as [[master-error-bound]], [[general-coupling-scheme]] and [[generalization-requirements]].

**The formulation's one real discovery: the defect splits three ways, and the middle term was missing from every previous page.** Telescoping through three interface traces — $\lambda^\star$ (what the *true* solution puts on $\Gamma$), $\lambda^\dagger$ (the exact root of the interface problem you actually *posed*), and $\lambda^{(k)}$ (what the solver returned) — gives $d = \tau + \sigma + \gamma$: **agent infidelity**, **transmission infidelity**, **solve incompleteness**. [[composition-error-theory]] and [[probed-dtn-coupling]] were both *about* $\sigma$ and neither carried it as a term; it was folded into $\gamma$ or left implicit.

**Three consequences arrive immediately, and each converts an existing narrative result into one line of algebra.** (1) **Schwarz iteration reduces $\gamma$ and nothing else** — it drives $\lambda^{(k)}\to\lambda^\dagger$, the root of the problem you posed, so if $\tilde\Lambda$ is wrong it converges *harder onto the wrong trace*. That is the wrapped-river theorem, now derived rather than argued. (2) **$\sigma$ is invisible to every diagnostic the framework reports** — the interface residual measures $\gamma$, the conservation residual measures a few functionals, and nothing measures $\sigma$. (3) **The vault's sharpest failure is a pure-$\sigma$ failure**: under periodic windows $\Lambda\equiv0$ so $\tilde\Lambda$ is maximally wrong, $\gamma\equiv0$ *bitwise*, and the answer was $2\times$ off. **The two-term split cannot even express that configuration**; the three-term split predicts it.

**The $\sigma$ bound is the theorem the "agreement is an operator" claim was reaching for.** $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$ — accuracy set by the **operator gap**, amplified by $1/\beta$, the inf-sup constant [[composition-error-theory]] §4.2 correctly named as the missing hypothesis and could not place in a formula. **Both factors are exactly what a probe returns**, so [[probed-dtn-coupling]]'s construction and this bound were built for each other without either page noticing. $\gamma$ is then owned by the accelerator — $\rho^k$ for Richardson (the $H^{-2}$), $\kappa$-governed for Krylov, **roundoff for a direct Schur solve** — so the honest statement of what probing buys is that it makes $\gamma$ vanish *and* reduces $\sigma$, the second being the larger effect.

**A structural result worth more than either page that asked for $\beta$ gave.** $L\le\lVert\mathcal A\rVert\cdot\max_i\mathrm{Lip}(\mathcal E_i)\cdot(1+C_\mu C_{\mathcal G}/\beta)$, so **$\beta$ sits in the denominator of both $\sigma$ and $L$: an ill-conditioned interface hurts twice — additively in the per-step defect, multiplicatively in the accumulation.** The master bound is $\lVert e^N\rVert\le L^N\lVert e^0\rVert+\sum_n L^{N-n}(\tau^n+\sigma^n+\gamma^n)$, with the passivity theorem supplying $L\le1$ structurally at any $N$. **Every prior result is a specialization**: a monolithic autoregressive model is $\sigma=\gamma=0$; and the worst-agent conjecture is $\sigma=\gamma=0$ with $L\le1$ — **a theorem, whose three hypotheses are now measurable quantities rather than assumptions.** The summary line: *the framework measures the smallest term, controls the second-largest, and has never reported the other two.*

**And the bound's vacuity condition, which was nowhere in the vault.** Past $T_{\text{pred}}\approx\lambda_1^{-1}\ln(\delta_{\text{tol}}/\delta)$ a trajectory bound proves nothing, and **no construction on the page changes that** — the divergence is the physics, not the method. Passivity and windowing buy the *method's* share of the amplification, never the system's.

**Generalizing Schwarz: it is a seven-parameter scheme, and every method the vault has debated is one point in it.** $\Sigma=(\mathcal D,\tilde\Lambda,\mathcal O,\mathcal K,\mathcal C,W,\varepsilon_{\text{tol}})$ — decomposition, transmission rung, ordering, accelerator, levels, window, tolerance. One-pass exchange, the R1 halo, the §8.3 thrust fixed point, refused stage S1, `WindowNS`, the probed Schur solve and waveform relaxation are seven coordinate settings, and **the thrust fixed point turns out to be a mis-specified $\varepsilon_{\text{tol}}$ rather than a different method**. **$\tau$ appears in no row** — the formal version of *faithfulness is a precondition, not a peer*.

**The scheme is compiled, not designed.** Experts publish a capability record (`bc_channel`, `bc_time_varying`, `differentiable`, `dt_native`, `regime_law`, `ports`, `storage`, `equivariances`, `validity`) and **nine admissibility rules** derive $\Sigma$ from it. The consequential one is **R2**: `probed-DtN` requires only `dirichlet`, because the higher rung is built *outside* the expert — **it decouples the achievable rung from the expert's own interface**, which is why the ladder is not closed to black boxes. **R4** records the asymmetry that a frozen expert can coarsen the exchange interval but never refine it; **R8** discharges §S3's coarse-space blocker since a probed Schur complement never changes $L$; **R9** requires that under multirate, conservation be enforced on the **time-integrated** flux. **The compiler sets $\varepsilon_{\text{tol}}$ relative to $\min(\hat\tau,\hat\sigma)$ and refuses anything tighter** — converging the interface below the dominant term is exactly the mistake of spending compute to shrink the negligible term while removing the only alarm. The existing `schwarz > 1` guard becomes the first instance of a general refusal mechanism rather than a special case. **The `emit` line in the macro-step pseudocode is the contract, not optional instrumentation.**

**What is still missing for arbitrary systems: ten gaps, and the premise removes fewer than it looks.** *"Assuming we have the proper agents"* sets $\tau\approx0$ — retiring OP-3, OP-5 and most of OP-6 — and leaves **$\sigma$, $\gamma$ and $L$ entirely untouched**, two of which have never been measured. **Three gaps are theory that exists nowhere in this vault, confirmed by search.** **G1 cross-points:** where three or more subdomains meet the transmission conditions are not independent — the difficulty FETI-DP and BDDC exist to handle. Overlapping-with-PoU avoids it, which is why 124 tiles meeting four-at-a-corner have never caused trouble; **[[probed-dtn-coupling]] introduces it by moving to a non-overlapping formulation, and does not mention it.** **G4 multirate stability:** subcycling is *forced* by differing `dt_native`, and pointwise-in-time flux matching across different clocks **is not conservative while the residual looks perfectly fine**. **G6 the chaotic regime:** zero mentions of shadowing, ergodicity or long-time statistics anywhere — chaos is discussed only when choosing architectures, which is a modelling decision and not an error theory, and this is most of the intended domain.

**Three more are mechanisms with no stated correctness condition.** **G3 field↔lumped** needs the reduction and prolongation to be **adjoint with respect to the port pairing**, or the interface silently leaks power and the Dirac interconnection the passivity theorem depends on is broken — this is the unstated condition beneath F1's *field and lumped experts as first-class peers*, and the wind farm's actuator disk is an instance built without it being written down. **G5 decomposition policy** — and the criterion falls out of the $\sigma$ bound: **cut where the exact DtN operator is closest to local**, i.e. where physical coupling across the surface is weakest, never along a shear layer, wake centreline or reaction front — and since both $\lVert\Lambda-\tilde\Lambda\rVert$ and $\beta$ come from a probe, **cut quality is measurable before any rollout is run.** **G9 topology mutation** reconciles as a restart with $e^0\neq0$.

**The two blocking gaps are both about silence.** Neither a mishandled cross-point nor non-conservative subcycling raises an error, fails a gate, or produces a suspicious number — the first degrades $\beta$, the second yields a residual computed correctly that means nothing. **That is the same failure mode as the bitwise-zero interface residual, and it is the mode this vault has been caught by most often.** Closes with six concrete conditions for "generalized", of which **emitting the full instrumentation $(\tau,\sigma,\gamma,L,\beta,\kappa,\pi,\mathcal R,T_{\text{pred}})$ is nearly free** — every quantity is a by-product of machinery the coupled solve already builds. **[AI Inference]** the recurring failure has not been computing the wrong thing; it has been *reporting the term that was easy to compute.*

## [2026-08-26] note | the gap list re-cut by goal — accurate composition and generic composition share only four items out of twenty

Query: *list all the gaps preventing (a) accurate composition given sufficiently accurate agents, and (b) a generalized architecture where experts plug in instead of every case study being a rework.* [[generalization-requirements]] was restructured around this split rather than a fourth page being added; nine gaps were found that the first cut missed, and G-numbers were kept stable because other pages cite them.

**The headline is the shape of the answer, not any single gap: the two lists overlap in four items out of twenty.** Accurate composition and generic composition are close to **independent programmes**, and progress on one should not be expected to buy the other. That is worth knowing before effort is allocated, because the roadmap has implicitly treated them as the same push.

**What the premise buys is smaller than it looks.** "Sufficiently accurate agents" sets $\tau\approx0$ — retiring OP-3, OP-5's checkpoint share and most of OP-6, which is genuinely a lot — and leaves **$\sigma$, $\gamma$, $L$ and the assembly operator untouched**. So **Goal A is almost entirely about the constants nobody has measured, not about the agents at all.** Its top item (G20) is exactly that: three of the four constants in [[master-error-bound]] have never been measured, and the framework reports $\gamma$, *the smallest term*, and nothing else. The premise also needs one sharpening it does not survive on its own: across a fluid–structure seam there is no single global operator whose restrictions the agents approximate, so **"an accurate agent" is undefined at a multiphysics interface** until G2 is settled.

**Three gaps were found by re-reading [[port-algebra-atlas-0.1]] rather than by new theory, and two of them were already written down.** **G11:** §9.3 records that **conservative-vs-consistent interface mapping is being chosen implicitly, per edge, by accident** — an `ADVEC` flow must map conservatively (integral preserved), a `THERM` effort consistently (pointwise values preserved), and getting it backwards is a classic partitioned-coupling bug. It is flagged, unfixed, cheap, and it silently destroys either conservation or accuracy. **G18:** the `nondim` block exists in the expert declaration but nothing *translates* at a port, so two experts on different reference scales will not agree — recorded as real, unavoidable and $O(K)$. **G14, and this is the sharpest finding of the session for Goal B:** the connection rule reads *"two agents may share an edge for port $P$ iff both declare $P$ and their interface geometries coincide."* **That single clause excludes non-conforming interfaces by assumption, and therefore excludes exactly the independently-built experts that plug-in exists for.** Nothing in the vault covers non-conforming transfer or mesh mismatch — zero hits on either term. Mortar projection with a measured inf-sup constant is the removal, which connects it back to $\beta$.

**Six further gaps were new.** **G12:** OP-5 charges **19% of total error to the assembly rule**, and that share survives perfect agents *and* perfect windows — a number already measured and never diagnosed, with no stated condition that a partition-of-unity blend be at least as accurate as the local solves it blends. **G13:** the interface temporal resolution is **floored at the expert's native $\Delta t$** — $\lambda(t)$ is represented at the exchange cadence, which is a component of $\sigma$ in *time* rather than space, and convergence does not remove it. **G15:** routing between *library* experts is a different problem from [[regime-moe-architecture]]'s in-model gating and is unaddressed. **G16:** port-vocabulary completeness is unproven — closure is claimed for the cases considered, and `ADVEC`'s variable passenger list is already flagged as where case-study-specific growth reappears. **G17:** moving and deforming interfaces (FSI, free surfaces, fronts, contact) invalidate a cached interface operator every step. **G19:** validating a new case needs a classical reference at the matched regime, which for arbitrary systems may not exist or may cost more than the surrogate — without it no accuracy claim is checkable.

**The four that block both goals** are G3 (field↔lumped adjointness — leaks power *and* is F1's unbuildable claim), G8 (abstention — every bound is vacuous outside validity, *and* a pluggable library without validity predicates is unsafe by construction), G18 (nondimensionalization — silent disagreement *and* the $O(K)$ cost plug-in is meant to remove), and G7/G12′ (differentiability and runtime estimation — the only route to an error bar without a reference, *and* what the compiler needs to choose a scheme).

**Priority, and it does not match the numbering.** Strictly blocking, in the sense that the framework produces wrong answers with no signal: **G11** (already identified, cheap), **G1** cross-points, **G4** multirate conservation, **G3** adjointness. Highest value per hour on Goal A: **G20** — fit $L$ from one paired rollout, then assemble one $\tilde\Lambda$ for $\beta$, $\kappa$ and $\sigma$; *nothing else on the list can be prioritized rationally until those three numbers exist*. On Goal B: **G23 + G24**, the capability record and the compiler, which together **are** the rework-every-case-study problem stated exactly, and whose specs are already written.

**The observation worth more than the ordering: every strictly-blocking gap is a gap about silence.** A mishandled cross-point degrades $\beta$; a backwards mapping conserves the wrong integral; non-conservative subcycling produces a residual computed correctly that means nothing. **None of them raises an error, fails a gate, or produces a suspicious number** — the identical failure mode to the interface residual that was zero *bitwise* while the answer was $2\times$ wrong, which is the mode this vault has been caught by more often than any other. **[AI Inference]** that argues the highest-leverage class of work is not a construction at all but instrumentation: emitting $(\tau,\sigma,\gamma,L,\beta,\kappa,\pi,\mathcal R,T_{\text{pred}})$ from every run, all of which are by-products of machinery the coupled solve already builds. The recurring failure has not been computing the wrong thing; it has been reporting the term that was easy to compute.

## [2026-08-26] note | the gap list filed as a tracked worklist — 26 items, five tiers, ordered by what unblocks the most

[[gap-worklist]] opened as the actionable counterpart to [[generalization-requirements]]: that page says *what is missing and why*, this one says *what to do, in what order, and how you know it is done*. Every row carries a **definition of done precise enough that the item cannot be quietly declared finished** — which is the failure mode a plain gap list invites. Rows are never deleted; the Status column moves, and `refused` is a legitimate terminal state with a reason, the way `schwarz > 1` was refused at config time.

**Tier 0 is measurement and is not negotiable, because every ranking below it is currently a guess.** W1 fit $L$ from one paired rollout (yielding $T_{\text{pred}}$ and a verdict on whether OP-2's drift matches $\eta = L-1$); W2 assemble one $\tilde\Lambda$ for $\beta$, $\kappa$ and the null-space dimension — which **must come out exactly 1**, and anything else indicts the probe rather than the physics, which is why it is the item's definition of done rather than a footnote; W3 split one macro-step's defect three ways; W4 emit all nine quantities from every run. **Each is under a day, and together they decide the ordering of the other twenty-two.** The justification is [[master-error-bound]] §7: the framework reports $\gamma$, the smallest term, and has never reported the other two.

**Tier 1 is the silent-correctness class**, and its defining property is that no item in it raises an error, fails a gate, or produces a suspicious number. W5 make conservative-vs-consistent mapping explicit **per port type rather than per edge** — already diagnosed in [[port-algebra-atlas-0.1]] §9.3 and left unfixed; W6 a **cross-point rule**, required *before* [[probed-dtn-coupling]] is built on any non-chain graph; W7 multirate flux matching on the **time-integrated** flux; W8 field↔lumped **adjointness**. One dependency is recorded explicitly because it is easy to miss: **W8 silently disables the passivity theorem, so W12 is worthless until W8 is checked.**

**Tier 2 is the plug-in architecture** — capability record (W9), coupling compiler (W10), non-conforming interface transfer (W11). W10's definition of done is the sharp one: *a new case runs with no hand-written coupling config, and inadmissible schemes are refused citing the rule number*. **Tier 3** is structural guarantees — storage functions, abstention, typing claims as trajectory-vs-statistical by $T_{\text{pred}}$, and adjoint localization, which would spend the differentiability claim for the first time. **Tier 4** is honestly deferred rather than silently dropped.

## [2026-08-26] lint | full pass — 183 pages, index coverage perfect, two stale claims created by this session's own work and fixed, three housekeeping defects filed

Scripted checks across all 183 `.md` files: **duplicate basenames NONE** (the bare-link convention is safe), **folder-qualified links NONE** outside the convention's own syntax examples, **control characters NONE**, **index coverage: every page listed, and no index entry points at a nonexistent page**.

**A methodology note worth keeping, because it nearly produced a fabricated finding.** The first index-coverage run reported **45 pages missing from the index**, including `port-algebra-atlas-0.1`, which is plainly there. Cause: the check normalized names with `os.path.splitext`, which splits at the *last* dot — so every page whose name contains a version number (`-atlas-0.1`, `-1.1`) was truncated to `port-algebra-atlas-0` and failed to match. **A lint that reports 45 defects in a vault this well-maintained is reporting a bug in the lint**, and the correct response was to distrust the tool rather than open 45 tickets. Corrected result: zero.

**Broken links: three, of which one is real.** `[[page-name]]`, `[[x]]`, `[[summaries/x]]`, `[[concepts/x]]` and `[[index.md]]` are all inside backticks in passages *documenting the link convention* — false positives, correctly ignored. `[[pattern-recognizer-vs-solver]]` is annotated *"if broken out as its own page later (link seeded)"* — deliberate, and the annotation is what makes it fine. **`[[phase1-resume-prompt]]` is real**: referenced from three Noether 1.1 pages as *"a self-contained pickup prompt"* that does not exist. Filed as W24 rather than fixed here, since it is unrelated to this session's subject and the honest fix is the owner's call.

**Two stale claims, both created by this session's own work, both fixed.** [[schwarz-iteration-atlas-0.1]] §S3 still carried **"Do not build S3 without settling that"** after [[probed-dtn-coupling]] settled it — a resolution note was added inline in the vault's usual append-don't-rewrite style, recording *why* the blocker dissolved rather than merely that it did: the blocker was on **coarsening**, and a probed Schur complement does not coarsen. And [[composition-error-theory]] §3 is titled *"The correct statement"* while giving the **two-term** bound that [[master-error-bound]] has since split into three; a superseded-in-one-respect box was added at the head of the section, because a reader arriving at a section with that title deserves to be told it is incomplete on the spot rather than in a See Also line.

**Also filed:** W25, `prereqs-and-resources` is the vault's only orphan under its own convention — reachable from `index.md` but through a markdown link rather than a bare wikilink. W26, four historical log headers use combined operations (`ingest+note`, `note + reorg`) outside the declared set `{init, ingest, query, lint, note, reorg}`; **the log is append-only, so the fix is to widen the convention note, not to rewrite history.**

**No contradictions found between pages.** The one class specifically hunted — a page claiming conservation implies accuracy — returned only hits in *summaries of other architectures* describing what those architectures lack, which is correct usage.

## [2026-08-26] lint | systematic finding: 11 table rows across the vault carried unescaped pipes inside math or code, silently breaking cell rendering

Follow-on from the same day's full pass, and the most useful class it found because it is **systematic rather than incidental**. A GitHub-flavored-markdown table row is split on `|` *before* inline parsing, so a pipe inside `$...$` **or inside a code span** ends the cell early — backticks do not protect it. The symptom is invisible in source and invisible to a link checker: the row simply renders with the wrong number of columns, and the tail of the description silently becomes a new column.

Found by scanning every line beginning with `|` for a bare pipe inside a math or code span. **11 rows across 9 files**, including two in `index.md` itself — the symmetry-averaging row (the Reynolds average, $\lvert G\rvert^{-1}\sum_{g}$) and the $N$-sweep row (the bit-for-bit `max|diff|` check). The rest were absolute values and one restriction bar, in [[composition-error-theory]], [[open-problems-atlas-0.1]], [[world-models-physics-ai]], [[attractor-energy-projection]], [[noether-1.0-rbc]], [[impl-wind-farm-guide]] and [[wind-farm-implementation-log]].

**Fixed by removing the character rather than escaping it**, which is the better repair: math now uses `\lvert` / `\rvert`, which is correct LaTeX *and* contains no pipe at all, so the row cannot break again if escaping rules change. Only the single genuine code span was escaped, since there is no alternative spelling inside backticks. Re-scan returns **0**.

**The generalizable rule, and it belongs with the conventions:** *inside a markdown table, never write a bare pipe — in math use `\lvert`/`\rvert` (or `\lVert`/`\rVert` for norms), and in code spans escape it.*

**Worth adding to a permanent lint script, because this is the third distinct failure this session that was invisible in an editor and invisible to grep** — after the escape-mangled control characters and the orphaned duplicate index fragment. All three shared one property: the file looked fine and the render did not.

**And the same class bit once more while this very entry was being written.** The text was passed to `python -c` inside a double-quoted shell string, so every backtick in it triggered command substitution and the content arrived with all code spans silently deleted — the entry appended cleanly, reported zero control characters, and had holes in it. It was removed and re-appended through the file-splice path. The standing rule *never put LaTeX or markdown inside a shell or Python string literal* was written after two instances earlier in the session; **this is the third, and the lesson is that the rule needs to be followed for prose containing backticks, not only for LaTeX.**

---

*Log format: `## [YYYY-MM-DD] operation | description` where operation ∈ {init, ingest, query, lint, note, reorg}*

## [2026-08-27] note | Theory closure audit — is the theory actually closed? No, and the holes sort into three classes

Asked directly, before committing to an implementation push: *"are all of the issues we are currently facing solved from a theoretical standpoint? After this, I don't want to go back to the drawing board for a while."* Filed as [[theory-closure-audit]].

**Verdict: not closed** — but the useful content is the sort, not the verdict. Re-cutting the twenty-four gaps by **kind of work** rather than by goal ([[generalization-requirements]]) or by priority ([[gap-worklist]]) gives three classes with very different costs: **4 CLOSED** (the master bound, the three-way split, the $\sigma$ bound, the worst-agent theorem — safe to build on), **9 IMPORTABLE** (solved in the classical DD/coupling literature, merely absent here; each needs a *decision recorded*, not research), **6 OPEN**, and 5 that are engineering rather than theory at all. **Nine of fourteen theory gaps being transcription is why the honest verdict is "not closed" rather than "back to the drawing board."**

**Verified against the vault rather than against the pages' own claims**, which changed one item's severity. A term-frequency sweep confirms `shadowing` and `ergodic` occur in exactly two pages — [[master-error-bound]] §9 and [[generalization-requirements]] G6 — **both of which mention them only to say they are missing**; `moving interface` occurs only in the worklist and the index; and `mortar` occurs six times but **never as a construction**, only as a cited gap. So three items previously recorded as "identified" are more precisely **named and undeveloped**, which is a different thing.

**The page's most useful section is the one nobody had written: §3, the seven-hypothesis envelope.** The master bound is a theorem, but on a class of systems no page had ever delimited — fixed graph, static and geometrically coincident interfaces, a single global evolution operator (so that $\tau$ has a referent), one macro-step clock, claims inside $T_{\text{pred}}$, bounded assembly, and a power-preserving interconnection with adjoint reduction/prolongation. Each hypothesis maps to exactly one gap, and each gap **is** the removal of that hypothesis. Stating the envelope converts "the bound might not apply" from a worry into seven yes/no questions.

**The finding that came out of writing it down:** the wind farm satisfies **four of the seven and has never checked the other three** ($T_{\text{pred}}$ never computed, the 19% assembly share never diagnosed, adjointness never stated let alone verified). **The flagship case study is not known to sit inside the envelope of its own error bound.** All three checks are cheap — they are W1, W3 and W8 — which is an argument for doing them, not against the case study.

**One open gap ranks above the rest and it is not the one the priority list had first.** O1 — *no claim class beyond $T_{\text{pred}}$* — is the only open item that is a hole **inside** the current envelope; the other five are envelope-*widening*. Past the horizon the trajectory bound is vacuous and no construction changes it, so the deliverable must become statistical (invariant measure, spectra, structure functions) — and turbulence, climate and plasma, the entire intended domain, lie mostly past it. The vault has no page developing that theory, no gate set for statistical claims, and no definition of what $\tau$ means against a measure rather than a trajectory.

**[AI Inference], and it decided the next step:** five of the six open gaps are invisible until a case violates their hypothesis, and the only case study that exists satisfies every one of them. An implementation push would therefore hit all five late and expensively. **Writing the architecture down end to end — every layer from expert declaration to emitted diagnostic, with the envelope as an explicit admissibility check — surfaces them on paper at the cost of a chat rather than a rewrite**, and produces the artifact the framework does not have: a specification of the whole system rather than a set of correct pages about its parts. Next session is that outline, not the build.

## [2026-08-27] note | The end-to-end architecture spec — nine layers on the envelope's spine, five named holes, and the claim-typing decision

Written in direct execution of [[theory-closure-audit]] §6's operational conclusion: *write the architecture down end to end before implementing, because the five envelope-widening gaps are invisible while the only built case study satisfies every hypothesis they remove.* Filed as [[end-to-end-architecture-spec]]. **No new theory** — the page specifies the *system*, which is the artifact the framework did not have: a specification of the whole thing rather than a set of individually correct pages about its parts.

**The organizing device is the seven-hypothesis envelope, and making it the spine is what the page buys.** Nine layers — expert declaration and capability record, decomposition and cut placement, the port algebra and connection admissibility, construction of the transmission operator, the interface solve and accelerator, assembly, time integration and multirate, the emitted diagnostic set, the typing of claims — each declaring which of E1–E7 it consumes, what it must do when one fails, and where the compiler refuses. The envelope becomes a **per-run stamp** with values `holds` / `fails` / `unchecked`, and the asymmetry that falls out of tabulating it is the useful one: **E1–E4 are static properties of a declaration and are decidable at compile time, before any compute is spent**; only E5, E6 and E7 can be `unchecked` at all — which is exactly the three the audit records as unverified for the wind farm.

**Three verdicts, not two, and this is invented on the page rather than derived.** `admit` / `admit-uncertified` / `refuse`, split by one rule: **refuse the silent-wrongness class, decertify the unverified-hypothesis class.** A spec with only *run* and *refuse* is either unusably strict or silently permissive, and the vault has been bitten by the second. Every Tier-1 worklist item — cross-points, reversed conservative-vs-consistent mapping, non-conservative subcycling, non-adjoint field-to-lumped reduction — is silent-wrongness and therefore a refusal; every unchecked envelope hypothesis is a decertification that still runs and still emits.

**Five named slots, one per envelope-widening gap, none solved — and each specified by five fields of which the fifth is the point.** `SeamReference` (O2, $\tau$ undefined at a multiphysics seam), `InterfaceMotion` (O4, moving and deforming $\Gamma$), `TopologyEvent` (O5, conservation across a topology mutation), `AssemblyCertificate` (O3, the missing consistency condition on the blend), `PortAmendment` (O6, vocabulary extension as a defined operation rather than a redesign). The fifth field is **the measurement the missing rule would constrain, emitted from every run that touches the slot even though nothing yet says what value it should take** — the ledger for a mutation, the operator drift for a moving seam, the blend defect for the assembly, the `UNDEFINED`-$\tau$ seam list for multiphysics. That is the vault's instrumentation lesson turned on its *unsolved* problems rather than its solved ones, and it costs one line per slot in the emit contract.

**One slot got smaller while being written, and the narrowing is new.** **[AI Inference]:** a probe never mentions a governing equation — it measures the expert's own response to imposed boundary data — so the probed transmission operator, and with it the read-off optimal Robin coefficient, is **well-defined across a fluid–structure seam where no single global operator exists.** The *rung choice* survives E3's failure; $\tau$ does not. So O2 splits into an **interface** component definable today from a one-sided reference operator via $\lVert\Lambda^{\text{expert}}-\Lambda^{\text{ref}}\rVert$, and an **interior** component that genuinely has no referent. The hole is real and smaller than the audit states it, and differently shaped.

**The decision the session was asked to make and made: claims are typed trajectory-inside-$T_{\text{pred}}$, statistical-outside.** Adopted, with the statistical type **declared and refused rather than approximated** — the type is reserved, its interface is specified (the functional of the invariant measure, the averaging window against the decorrelation time, an estimator with a sampling error, and a gate), and emitting one is a refusal until all four can be filled. The reasoning recorded: the alternative is the status quo, and the status quo *is* the failure mode; the trajectory half costs one arithmetic step past W1; the cost of adopting is that the framework **looks** worse because most of the intended domain becomes a refused claim, which is the accurate appearance; and the cost of not deciding is that the emit set, the gate set and G10's verification protocol all get built for trajectory claims only and the statistical path becomes a rewrite of three things instead of an enum. **What is explicitly not decided is the statistical error theory — that is O1 and it stays open.** A typing discipline is a decision; a statistical theory is research.

**A third claim state was forced by the decision and is the uncomfortable one.** $\hat L$ unknown ⟹ $T_{\text{pred}}$ unknown ⟹ no type is decidable, so the claim is **`UNTYPED`** — and since $L$ has never been measured, **every result currently in the vault types as `UNTYPED` under its own architecture's rule.**

**The horizon turns out to have three branches and only one is written down anywhere.** Read directly off [[master-error-bound]] §8's specialization table: $L>1$ gives the familiar exponential $T_{\text{pred}}\approx\lambda_1^{-1}\ln(\delta_{\text{tol}}/(\tau+\sigma+\gamma))$; $L=1$ gives a **linear** $N_{\text{pred}}\approx(\delta_{\text{tol}}-\lVert e^0\rVert)/\max_n(\tau+\sigma+\gamma)$; $L<1$ gives **no horizon at all** provided $\max_n(\tau+\sigma+\gamma)/(1-L)<\delta_{\text{tol}}$. **This connects E7 to E5, which no page does:** establishing incremental passivity is not only the structural route to $L\le1$, it is what converts a short exponential horizon into a long linear one — the passivity certificate and the claim type are the same investment. And **[AI Inference]** on the emit contract: $\delta_{\text{tol}}$ sits inside the logarithm and tolerance is a property of the *quantity claimed*, so a run has one $L$, one defect, and **as many horizons as it has quoted quantities** — the emitted $T_{\text{pred}}$ must be a table keyed by claim, not a scalar.

**Two compile traces, and they are the payoff for specifying the whole system rather than its parts.** The **wind farm as built** stamps E1–E4 `holds`, E5–E7 `unchecked`, and is refused three times before any rollout: mapping class unset (C3), the actuator disk's field-to-lumped reduction with no adjointness certificate (C6), and — the one worth the page — **$\Xi=0$ at L4**, where $\Lambda\equiv0 \Rightarrow S\equiv0 \Rightarrow \mathcal G(\lambda)=-\chi$ for every $\lambda$, so **the interface problem is empty rather than hard, every trace is equally consistent, and the honest description of the run is an ensemble of independent local solves.** *The case study's negative composed result was available as a compile-time refusal for the cost of one probe.* That is not a criticism of the case study, which produced the finding it was built to produce; it is an argument that the mechanism which surfaces it is a config-time rule rather than a better experiment. The **rocket ascent as specified does not compile** — different `governing_family` at the fluid–structure seam (E3), a non-static combustion front and plume boundary (E2), lumped-to-field ports needing adjointness, and differing `dt_native` forcing multirate (E4) — **tripping three of the five named slots in its declared scope, with `TopologyEvent` arriving when staging does.** **[AI Inference]:** the ordering that implies is that the rocket should not be the next build, since three of its refusals are research while every wind-farm refusal is transcription.

**Ten places the spec is guessing are listed in their own section**, because a specification that does not separate its decisions from its inventions is worse than none: the three-verdict system, the completeness of the nine-layer decomposition (G15 routing is a known missing layer, and training and the learned interior are out of scope by declaration), the inherited-and-not-upgraded **[AI Inference]** status of the G5 cut-placement criterion, the E3/probe asymmetry, the collar-solve shape guessed for `SeamReference`, the invented blend-defect $\alpha$ and its conjectured partition-of-unity form, the six-field `PortAmendment` checklist, the assumption that a mutation rule will constrain jumps in conserved functionals, and the per-claim horizon table.

**Worklist effect, filed into [[gap-worklist]] as a new Tier 5 plus a reframe table.** **Closes: nothing outright**, and saying so plainly matters more than finding something to claim — a specification closes no measurement. The nearest thing is that **W14 is no longer blocked on a judgement call**, since §11.1 supplies the rule it was waiting for. **Reframes ten items**, the sharpest being W6 (a cross-point *refusal* now exists in the spec even though the *rule* does not — which is the correct order), W8 (from a check to run into a connection precondition every field-to-lumped port must pass), W10 (this page is the compiler's specification), W16 (promoted from Tier 4 to the specified decomposition policy with its speculative status preserved verbatim), W17 (reclassified from a $\Delta t$ constraint into a component of $\sigma$ *in time*), and W22 (splits into two slots with different interfaces and different owning layers). **Adds six**, W27–W32, all of them field-5 measurements: the envelope stamp, the assembly certificate, the multiphysics seam declaration and its operator-norm surrogate, the interface motion class with the operator drift priced even on static runs, the mutation ledger, and the port amendment procedure. **All six are small, and that is the argument for them** — each instruments a hole rather than attempting to fill it.

## [2026-08-27] note | Is the theory decided? No — the composition operation itself was never specified, and six of the nine importable mechanisms are one object

**The question.** Asked of [[end-to-end-architecture-spec]] the same day it was written: *is all of the theory decided — enough for a build that fully outlines accurate composition given the agents, and the plug-in architecture of the framework?* **The answer is no, and the residue sorts into three parts, only one of which was a surprise.**

**Part 1 — the nine importable mechanisms had gates but no rules.** [[end-to-end-architecture-spec]] specifies where the compiler **refuses** a connection whose `mapping` is unset, whose reduction/prolongation pair is non-adjoint, whose interfaces do not coincide. It does not say what to set them to, and neither does any other page — [[theory-closure-audit]] §4 classifies all nine as *"adoption, not research"* and estimates *"a week of writing rules down."* **A refusal without a rule is unbuildable**, so the rules are now written in [[interface-transfer-theory]].

**Part 2 — the six genuinely open gaps are correctly held open.** O1 is a recorded decision, O2–O6 are five named slots with declared interfaces and required measurements. Nothing on this list moved and nothing should have.

**Part 3 — three properties the plug-in architecture asserts and had never stated.** This is the part that was not on any list. [[generalization-requirements]] tracks twenty-four gaps and [[theory-closure-audit]] sorts them into three classes; **none of them is closure under composition, substitution, or conformance**, because every prior gap is about a *mechanism* and these are about the **composition operation itself**, which no page had treated as an object with algebraic properties to check. They became visible only when the whole path was written down end to end and then asked to run twice. Now [[plug-in-composition-theorems]], and four new gaps: **G25–G28**.

---

### [[interface-transfer-theory]] — six mechanisms are one object

**Declare a common interface space $M$ per seam and one prolongation $P_i:M\to V_i$ per side. The reduction is not declared — it is forced.** Requiring the transfer to neither create nor destroy interface power gives $\langle R_ie_i,\mu\rangle_M=\langle e_i,P_i\mu\rangle_{V_i}$, which is the definition of an adjoint, so $R_i=P_i^\ast$ and **any other choice leaks power at the interface.**

- **G11 and G3 are the same condition.** The two mapping classes are the two halves of one adjoint pair: a consistent map is an interpolation, a conservative map is its adjoint. So efforts map through $P_i$ (consistent), flows through $P_i^\ast$ (conservative), and **getting it backwards stops being expressible** — the classic partitioned-coupling bug requires two independent declarations to disagree, and there is now only one. Field↔lumped adjointness is the special case $\dim M=1$: **the actuator disk was not missing a check, it was missing a declaration.** These were filed in two different tiers of [[generalization-requirements]] — one a silent correctness bug, one a plug-in blocker — and the reason neither page saw the connection is that G11 was stated in the language of *quantities* and G3 in the language of *operators*.
- **G14 is a restriction of the connection rule, not of the construction.** [[port-algebra-atlas-0.1]] §5's *"their interface geometries coincide"* is called by [[generalization-requirements]] *the single most restrictive line in the framework for plug-in use*, and it was written before [[probed-dtn-coupling]] existed. The probe never touches either agent's discretization — it hands over a trace and takes back a flux — so $\tilde\Lambda_i^M=P_i^\ast\Lambda_iP_i$ is a Galerkin compression, and its **congruence structure is what carries positive-realness through a non-conforming transfer.** It fails to, precisely when the pair is not adjoint: **that is the one-line mechanism of "G3 silently disables G22."** The connection rule is amended in place on [[port-algebra-atlas-0.1]] §5, with geometric coincidence retained as the special case $M=V_A=V_B$, $P_i=\mathrm{id}$, so nothing that works today stops working. **[AI Inference]**, and cheap to falsify — probe two agents at deliberately mismatched resolutions and watch $\beta$.
- **The multiplier space and the probe basis are the same space**, $\dim M=\min_i m_i^{\text{eff}}$, so non-conforming coupling costs no probes beyond those already budgeted, and the inf-sup constant is **measured rather than proved** — the only version available to a black box with no approximation theory.
- **$\sigma$ gains two named components**, $\sigma_{\text{nc}}$ from the interface space and $\sigma_{\text{time}}$ from the temporal representation of $\lambda(t)$. Neither is reduced by iteration, for the same reason $\sigma$ itself is not.
- **Multirate and waveform relaxation are the time-axis pair of the space-axis rule.** R9 becomes a stated condition with a refusal, plus two consequences it did not carry: the interface representation order **caps the scheme order** (a second-order expert with a constant interface ring is a first-order method), and the stability limit is set by **coupling stiffness**, which can be shorter than any agent's native step — the opposite direction from R4 and not implied by it.
- **Nondimensionalization gains a checkable identity**, $s_e s_f=s_P$ per port, and scale-set completeness becomes decidable at compile time from the port list alone. A scale pair that does not satisfy it silently destroys the power bond, after which $\mathcal R(t)$ reports a unit error in watts.
- **Cross-points gain a recorded decision** conditional on the decomposition axis — no action under overlapping+PoU, primal DOFs under non-overlapping — and a detector that rides on [[probed-dtn-coupling]] §2.1's existing null-space check: any excess null direction is a defect.
- **DWR inherits O2, with the same split.** $\eta_\Gamma$ needs only the port residual and is available now, **including at multiphysics seams**; $\eta_i$ needs a governing residual and is undefined where none is declared. This is the identical decomposition [[end-to-end-architecture-spec]] §6.4(b) draws for $\tau$, reached independently.
- Cut placement keeps its **[AI Inference]** status verbatim; the page adds only a measurable form, flagged as the weakest claim on it.

---

### [[plug-in-composition-theorems]] — closure, substitution, conformance, and two horizons

**Closure.** A connected subgraph with its internal ports matched presents a valid capability record, its transmission operator being the **Schur complement** of the internal block — so nesting is not a new mechanism, it is **stopping the existing elimination early.** Legal exactly when $\beta_{\text{int}}>0$, and **grouping-invariant** by the quotient property, so a hierarchy and the flat graph give the same composed operator and a build need not commit to a nesting depth. The record **composes in 8 of 13 fields**, fails in 3 — `L_native`, `regime_law`, `validity`, all *operating-point* rather than *interface* fields — and propagates the `SeamReference` hole in 2. A composite's `validity` is evaluated at the internal solution and is therefore **post-hoc**; decided: it declares `deferred` and raises a declination from inside its own step.

**The attribution theorem, and it is the uncomfortable one.** The master bound at $N=1$ with $e^0=0$ gives $\tau_{\mathcal C}\le\tau^{\text{int}}+\sigma^{\text{int}}+\gamma^{\text{int}}$: **a subassembly's transmission and solve infidelity become the agent infidelity of the expert one level up.** They are not removed by nesting and not double-counted — they are **relabelled**. Three consequences: the three-way split is **not invariant under regrouping**; every emitted defect must carry the **depth** it was measured at, and [[end-to-end-architecture-spec]] §10.2's emit contract does not; and **[AI Inference]** OP-5's expert-versus-architecture share is depth-relative, since `WindowNS` over 124 tiles is itself an internally composed object — checkable by re-cutting one run.

**Substitution.** A swap moves $\beta$ by at most $\lVert\Delta\rVert$ (Weyl), giving a **certificate computable from two probes and no rollout**: the swap preserves admissibility iff $\lVert\Delta\rVert<\beta-\beta_{\min}$. And the modularity theorem: **passivity is the only property in the framework that survives a substitution without re-certification**, because $L\le1$ follows from a per-part property plus the interconnection structure and references no global quantity. That **promotes G22 from an accuracy nicety to the enabling condition of plug-in itself**, and it is currently on neither list's blocking tier. **What is not true: there is no monotonicity theorem.** A strictly better expert — smaller $\tau$, smaller benchmark error — can produce a worse composition, because its $\tilde\Lambda$ can lower $\beta$, which sits in the denominator of **both** $\sigma$ and $L$. Same failure shape as $\Xi=0$ on an accurate checkpoint: accuracy and composability are different axes, and here they are shown able to point in opposite directions.

**Conformance.** Every guarantee is inherited from a declaration and **nothing checks that a declaration is true** — tolerable when the declarer built the expert, foundational when the point of plug-in is that they did not. Four fields fail silently, and the three consequential ones are certified by machinery **the probe already builds**: `bc_channel` by $\Xi_i>0$, `storage` by the symmetric part's spectrum, `equivariances` by one extra solve per generator. **The vault has already failed two of these tests the slow way** — $\Xi=0$ for the frozen checkpoint is a failed `bc_channel` conformance test, and OP-6's mirror-equivariance failure is a failed `equivariances` one; both were framed as findings about a checkpoint and both would have been caught at admission. **`validity` is falsifiable but not verifiable** — certifying it requires knowing where the expert is wrong, which requires the reference the predicate exists to avoid needing — so the certificate records *"not falsified on suite X"*, never *"valid"*, and **the architecture rests on exactly one unverifiable declaration.** **[AI Inference]:** that reframes G19 as a **cost curve on the single axiom** rather than an independent gap.

**Two horizons.** Abstention semantics decided — a declination halts and types the claim on $[0,t_{\text{decline}})$; a fallback is a mid-rollout substitution and reconciles as a restart with $e^0\neq0$, sharing machinery with `TopologyEvent`. And $T_{\text{abs}}\approx\Delta t/(Kp)$, so $T_{\text{usable}}=\min(T_{\text{pred}},T_{\text{abs}})$. **$T_{\text{abs}}$ degrades linearly in library size and $T_{\text{pred}}$ does not** — so for a large enough library the binding limit on a rollout is **the framework, not the physics**, which is the first quantity to price a needlessly conservative validity region. **[AI Inference]**, and the independence assumption is known to be wrong in both directions: the $K$-scaling is the claim, the constant is not.

**Routing (G15)** gains a criterion — rank the feasible set by composability $\Xi$ with $\tau$ as tie-break, and **refuse on an empty feasible set** rather than falling back to the nearest expert, since routing by benchmark accuracy is §2.5's substitution failure committed automatically at every subdomain. Static routing is compile-time; **dynamic routing is a topology mutation**, and **[AI Inference]** dynamic routing, abstention fallback and topology mutation are argued to be one operation, which would make `TopologyEvent` more central than it was filed as.

---

**Filed:** two index rows after [[end-to-end-architecture-spec]]; [[gap-worklist]] gains **Tier 6** (W33–W36) plus a reframe table. **The worklist grows by four rows for two pages of theory, and that is the honest accounting** — W5, W8 and W11 **merge into one declaration**, W6 and W7 turn from open questions into written rules, W15 splits, and W12 is promoted to both goal lists. Reciprocal links added to nine pages, including two **in-place amendments** on [[port-algebra-atlas-0.1]] — §5's connection rule and §9.3's mapping recommendation — so the vault does not carry a superseded rule and its replacement side by side. Seven guesses enumerated in [[plug-in-composition-theorems]] §7, seven undecided items in [[interface-transfer-theory]] §10.

## [2026-08-27] lint | Atlas section — two live escape corruptions found and fixed, and the standing control-character scan has a blind spot

**Scope:** the Atlas 0.1 section (47 pages), plus `index.md` and `log.md`, plus a vault-wide pass for the structural checks. Run immediately after filing [[interface-transfer-theory]] and [[plug-in-composition-theorems]].

**Contradictions found and resolved — all four were the new pages superseding an older rule, and all four are fixed *in place* rather than left to be reconciled by a reader:**

1. **[[port-algebra-atlas-0.1]] §5, the connection rule.** *"Their interface geometries coincide"* is superseded by [[interface-transfer-theory]] §4.1's declared common interface space and prolongation. Amended in place, with geometric coincidence retained as the special case, so the vault does not carry a rule and its replacement side by side.
2. **[[port-algebra-atlas-0.1]] §9.3, implementation note 3.** *"Adopt preCICE's conservative-vs-consistent mapping distinction per port"* is superseded: the class is **derived** from the single declared prolongation rather than adopted as a field. Marked superseded in place.
3. **[[end-to-end-architecture-spec]] §3.2 and §5.2.** The `PortDecl` record's `mapping`, `reduction`, `prolongation` and `adjointness` collapse to `interface_space` plus `prolongation`; checklist conditions **C2, C3 and C6 are one condition**. Both amended in place. C8 additionally gains the composition-level abstention semantics it did not have.
4. **[[theory-closure-audit]] §4 and [[generalization-requirements]] §A/§B.** Several *"zero coverage"* and *"no construction page anywhere in the vault"* entries are now stale. Status updates added to both, with the honest correction attached: **the audit's estimate was right about the effort and wrong about the structure** — the nine importable mechanisms were filed as nine independent transcriptions and six of them are one object.

**Two live corruptions of the escape class, found and fixed:**

- **`wind-farm-implementation-log.md`, the W3 go/no-go row.** `\rangle` had been interpreted as a carriage return, so the row read `$\langle U_d` / newline / `angle$ within $0.140\%$` — a table row split across two physical lines, which **stops the table rendering from that row down**. Restored to `$\langle U_d\rangle$`.
- **`index.md` line 121.** A 460-character **orphaned duplicate** of the wind-farm spec row's tail, beginning mid-formula at `\nabla^\perp\psi$` — the identical signature. The correct text is already present and intact on the spec's own row at line 116, so the fragment was deleted rather than repaired.

**The finding that outlives both fixes.** The vault's standing verification — *scan every `.md` for control characters, which must come back zero* — **cannot see the most common instance of the bug it exists to catch.** Reading a file in text mode applies universal-newline translation, so a lone `\r` is converted to `\n` before the scan ever sees it; and `\r` is the leading character of `\rangle`, `\rho`, `\rVert`, `\right`. Both corruptions above sat in the vault **passing a clean control-character scan.** The scan must read **bytes**, and it must be paired with two structural checks that catch the symptom rather than the cause: **table column-count consistency**, and an **orphan-continuation check** — a non-pipe, non-blank line immediately following a table block. All three now run vault-wide. Recorded as W37 (the two instances, fixed) and W38 (the method, fixed) in [[gap-worklist]].

**One process note worth keeping, because it cost a revert.** The first repair script generalized from `\rangle` to a table of TeX control-word tails, including `u` for `\nu`. That tail matched seven legitimate line beginnings — `u_t = ...`, `uv pip install`, `unchoked operation` — and joined them. All seven were reverted exactly, verified against their surrounding context. **A repair script for this class must be anchored on the specific corrupted site, not on a pattern**, because the pattern is *"a common English or LaTeX word"* and it will always over-match.

**Structural checks, final state:** control characters **0** (byte scan, whole vault, 188 pages); bare pipes inside math or code spans on table rows **0**; malformed or split table rows **0** across the Atlas section, `index.md` and `log.md`; dangling wikilinks in the Atlas section **0** (the one reported, `page-name`, is a backticked documentation example in [[00-atlas-0.1-overview]] §on linking, not a link); index coverage complete except `prereqs-and-resources`, which is the pre-existing orphan already tracked as W25. The two new pages carry 12 and 13 outbound links with none dangling, and 13 inbound each.

## [2026-08-27] note | The specification is implemented — nine findings, and seven of them are things the theory got wrong

**What was built.** `atlas/`, a generalized plug-in composition layer implementing [[end-to-end-architecture-spec]], [[interface-transfer-theory]] and [[plug-in-composition-theorems]] end to end: 23 modules, 7,200 lines, 120 tests over 1,400, `numpy` only. Drop-in for `src/atlas/` on the `atlas-0.1` branch of the build repo ([[code-and-papers]]).

**The claim it exists to make true, and it is now executable:** *a new case study is a graph of declared agent capability records plus port connections, with zero hand-written coupling code.* `tests/test_compiler.py::TestZeroHandWrittenCouplingCode` is a three-agent chain nobody wrote coupling code for, compiled from declarations alone to `admit-uncertified` with a probed-DtN rung and a direct Schur solve. The two existing case studies are **fixtures, not targets**: they exist so the compiler has concrete graphs to be tested against.

**What the compiler reproduces, from the declarations and nothing else.** The wind farm as built compiles to **refuse**, with the stamp $(\text{E1 holds},\ \text{E2 holds},\ \text{E3 holds},\ \text{E4 holds},\ \text{E5}/\text{E6}/\text{E7 unchecked})$ — exactly [[end-to-end-architecture-spec]] §13.1. L3 refuses for a missing declaration, the probe returns $\Xi=0$ so the word *coupled* is refused on the output while the run itself is not, R6 refuses the direct and Krylov solvers, and the claim types `UNTYPED`. **No rollout is involved anywhere in that trace**, which is the finding the case study cost a build to produce. The rocket does not compile and trips three of the five named slots in its declared scope, with the fourth arriving when staging is declared.

---

### The seven places the theory was wrong, or underspecified enough to build wrong

**1. E2 and L3 admissibility are not the same test, and no page says so.** [[end-to-end-architecture-spec]] §1 states E2 as *"interfaces static and geometrically coincident"*; the 2026-08-27 amendment at §5.2 changed the **connection rule**, not the hypothesis. A seam can therefore satisfy E2 — static, coincident — and still be refused at L3 for want of the declaration the amended rule requires. **That is the wind farm exactly**, and §13.1's own stamp is only reproducible if the two are kept apart: collapse them and E2 reads `fails`, contradicting the trace on the same page. Implemented as a separate `geometrically_coincident` declaration, which **nothing checks** — it is a label like any other and [[plug-in-composition-theorems]] §3's argument applies to it.

**2. A missing declaration leaves E7 `unchecked`, not `fails`.** §5.3 reads *"C5 or C6 unmet ⟹ refuse — this is the only envelope hypothesis whose failure is a refusal"*, which conflates the connection **verdict** with the stamp **value**. A refusal for want of a declaration establishes nothing either way; only a *measured* non-adjoint pair or a *measured* negative passivity mode fails E7. §13.1's stamp is again the correct reading and §5.3's phrasing is not.

**3. The transmission rung must be decided before L2, so the nine-layer stack is mis-ordered.** The layers are declared *"ordered by dependency, not by execution"*, but the rung (R1 lifted by R2, decidable from L1's records alone) fixes the decomposition axis, and the axis decides whether cross-points exist at all. Deciding the rung at L5 lets a compile check cross-points against the **declared** axis and then silently switch to the one that has them — the silent-wrongness class, inside the compiler that exists to catch it. The true order is L1 → rung → L2. This is visible the moment the wind farm is given a boundary-capable expert: R2 lifts the rung to `probed-DtN`, the axis goes non-overlapping, and L2 must refuse on cross-points. §4.3 names the trade and does not draw the ordering consequence.

**4. The expected null-space dimension is a property of the assembled seam, not of one side.** [[interface-transfer-theory]] §7 writes $\dim\ker(\tilde\Lambda^M)=1+\#\{\text{untreated cross-points}\}$, the $1$ being incompressibility. At a field↔lumped seam the lumped side **responds in the direction incompressibility leaves open**, so the assembled operator is not singular and the expected count is $0$. The identity is per-seam and depends on which agents meet there. Declaring it per port type produces a false defect report on every rotor face.

**5. A null-space *deficit* has no verdict anywhere.** Both pages give only the excess case. A measured null space smaller than declared still indicts one of two things — the probe is not resolving a constraint the physics has, or the declaration is wrong — and it is free to check. Implemented as a decertification: not the cross-point signature, and not something that should pass silently either.

**6. The passivity defect needs a noise floor, and as stated it does not have one.** $\pi_i=\lvert\min(0,\lambda_{\min}(\tfrac12(S+S^{\!\top})))\rvert$ taken literally reports arithmetic noise as a physical defect on **every** symmetric operator the probe ever assembles — measured here at $1.3\times10^{-15}$ on a matrix that is positive semidefinite by construction, which flipped E7 to `fails` on an otherwise clean graph. Implemented with a scale-relative floor, keeping the raw eigenvalue alongside so the clipping is visible rather than silent.

**7. The $K$-threshold refusal at L1 is a specified gate that cannot fire.** §3.3 requires a **refusal** for any graph at or above the $K$ at which $P(\text{some agent OOD})\to1$, and *"a threshold this spec does not set"*. The compiler decertifies and says exactly that rather than inventing a number. It is a hole in the spec wearing the shape of a rule.

### Two smaller ones

**8. A record can claim a derivative it cannot supply.** `differentiable: jvp` with no JVP callable is a contradiction **inside the declaration**, decidable at L1, and it is not among L1's listed refusals — it surfaced as a probe failure at L4 instead, three layers late.

**9. The zero-probe term is load-bearing at runtime, not only in the probe.** $\chi=-\sum_i\partial_n\mathcal E_i[0]$ is what makes the interface problem inhomogeneous; with $\chi\equiv0$ the converged trace is trivially zero and every seam diagnostic is vacuous. [[probed-dtn-coupling]] §2.2 treats $\mathcal E_i[0]$ only as the probe's bias-removal term.

**And one the port algebra does not say:** the two sides of an `ADVEC` seam must declare the **same passenger list**. A multibond's passenger list is part of its type, so a disagreement is an unnamed sixth port type in disguise and belongs in a `PortAmendment`. Caught by C1 on the rocket's $e\!-\!f$ seam when the plume and the combustion outflow were declared with different lists.

---

### What the implementation confirms rather than corrects

The unification of [[interface-transfer-theory]] §2 holds up under construction and is the single best thing in the three pages. One declared prolongation per side, with $R_i=P_i^\ast$ **derived**, makes the reversed-mapping bug *not expressible*: `port_decl` refuses the four fields the amendment dropped, `mapping_class` takes no declaration and therefore cannot be overridden, and the field↔lumped case falls out as $\dim M=1$ with the reduction forced to be the weighted integral. The hand-written reduction that the actuator disk historically carried is reproducible only by deliberately setting an escape-hatch field, and the conformance test catches it at $10^{-1}$ against a machine-precision baseline.

Likewise **§6.4(b) survives contact with code**: the rocket's fluid–structure seam has E3 `fails` and $\tau$ `UNDEFINED`, and its $\beta$, $\kappa$, passivity spectrum and optimal Robin coefficients are all assembled anyway, because the probe calls `boundary_response` and never mentions a governing equation. The transmission layer is robust to E3's failure; only the attribution is not.

---

**Filed:** `atlas/` at the vault root with its own `README.md` recording seven implementation decisions and two things the implementation could not do. [[gap-worklist]] status column updated across 18 rows, with six new items **W39–W44** for the findings above. Amendment notes added in place on [[end-to-end-architecture-spec]] §1 and §5.3 and on [[interface-transfer-theory]] §7, so the vault does not carry a rule and its correction side by side.

> **The honest accounting.** The specification survived implementation. Nothing in the master bound, the port algebra, the three-verdict system, the transfer unification or the composition theorems needed changing. What needed changing was **six places where a stamp value and a gate verdict were conflated, an ordering was implied but not stated, or a quantity was specified without the tolerance that makes it measurable** — which is the expected failure profile for theory written before a build, and a considerably better one than the alternative.

## [2026-08-27] note | The six corrections are written into the theory, and the amendment notes are gone

The seven findings from *"The specification is implemented"* were recorded as **amendment blockquotes pinned beside the lines they contradicted**, and as [[gap-worklist]] W39–W44. That was the right way to *flag* them and the wrong way to *leave* them. **All six are now folded into the body text of the five pages they indicted, and the blockquotes are removed.**

**Why the notes had to go rather than stay as history.** A page that states a rule and corrects it four lines later is a page whose **primary text is what gets implemented**. The correction is read once, by whoever is already looking for it; the rule is read every time. This vault's own convention — *"never blend an inference with factual content"* — has a mirror image nobody had written down: **never leave a superseded rule in the position of the operative one.** The provenance is not lost; it lives in this log, in W39–W44's `done` cells, and in [[atlas-implementation]] §2, which is where a reader holding an older copy will look.

**What changed, page by page.**

| Page | Was | Is |
|---|---|---|
| [[end-to-end-architecture-spec]] | E2 and E7 rows set a stamp value and a gate verdict in the same cell, making §13.1's own trace unreproducible from §1 | New **§1.1**, *"a stamp value is not a gate verdict"*, with the general rule: **only a measurement writes `fails`**. Absence of evidence is `unchecked`. §5.3's two rows say which of the two they set |
| [[end-to-end-architecture-spec]] | the rung was chosen at §6.2 (L4), three layers after L2 gated on the axis it fixes | New **§2.1**: L1 produces the rung, L2 consumes it, §6.2 assembles at a rung already fixed. The wind-farm lift is worked as the case that fires it |
| [[end-to-end-architecture-spec]] | L1 demanded a **refusal** at a $K$ threshold *"this spec does not set"* | New **§3.4**: retracted. A decertification at every $K$, with $K$ emitted, and $K^\star(\epsilon)=\lceil\log(1-\epsilon)/\log(1-p)\rceil$ named as what a measured $p$ would reinstate. Plus a fifth L1 refusal row: `differentiable: jvp` with no JVP callable |
| [[interface-transfer-theory]] | $\dim\ker\tilde\Lambda^M=1+\#\{\text{cross-points}\}$, with the $1$ carried as a constant | §7 states $n_0(\Gamma)$ **per assembled seam** — $1$ fluid–fluid, $0$ field↔lumped — with a three-row verdict table: excess refuses, deficit decertifies, equal passes |
| [[probed-dtn-coupling]] | §2.1's free correctness check read *"not one-dimensional"*, which is the origin of the per-port-type error | stated per seam, pointing at §7 for the count and the verdicts |
| [[probed-dtn-coupling]] | $\pi_i=\lvert\min(0,\mu_i)\rvert$, which reports roundoff as physics | §4.2 clips against $\varepsilon_i=c\,\varepsilon_{\text{mach}}\lVert S_i\rVert$ and emits $\mu_i$ raw beside it, so the clipping is visible |
| [[port-algebra-atlas-0.1]] | §5's connection rule required only *"both declare $P$"* | *"declare $P$ **with the same passenger list**"* — vacuous for the four simple types, binding for `ADVEC`. §3.2 carries the per-face argument and the example record shows the per-face form |

**One correction found a second instance of itself.** W41 was opened against [[interface-transfer-theory]] §7, but the *"not one-dimensional"* phrasing originates in [[probed-dtn-coupling]] §2.1 and §7 was quoting it. Fixing only the page the item named would have left the sentence that produced the error sitting in the page the probe is built from — and would have broken the quotation, which is how it was noticed.

**One correction got weaker on contact and is better for it.** The W41 note claimed $n_0(\Gamma)$ should be *computed by the compiler from the graph*. It cannot be: $n_0=1$ at a fluid–fluid `MECH` seam comes from **incompressibility**, a property of the governing family, and a compiler that inferred it would be special-casing a physics it is built never to name. So $n_0$ is **declared per seam and unverified**, which puts it in the same class as `validity` under [[plug-in-composition-theorems]] §3 — and the asymmetric verdicts matter *because* of that: an $n_0$ declared too large silently masks a cross-point, and nothing detects it.

**W43 closed by retraction, and the row says so.** The unfireable refusal is gone; the threshold is still unset. **Retiring a rule that could not fire is not the same as building the gate it promised**, and the `done` cell names $p$ (W13) as the measurement that separates the two.

**`atlas/` changed in one place** — the $K$-threshold decision now cites §3.4, reports the graph's $K$, and names $K^\star$ rather than reporting the missing threshold as a hole in the spec. 120 tests pass unchanged, which is the useful signal: **every other correction was already what the code did**, and the six edits closed the gap from the theory's side.

> **What this does not change.** No graph reaches `admit`. $L$ is unmeasured (W1), the assembly accuracy condition does not exist, and the six corrections were all about *saying the right thing*, not about measuring anything. The Tier 0 rows have not moved and could not have.

---

## [2026-08-27] note | Tier 0 measured — the probe is sound, the equation it feeds is not, and the bound holds

**Page:** [[tier0-measurements]] (new) · **Code:** `atlas/cases/window_ns.py` (new), `scripts/tier0_window_ns.py` (new), `atlas/probe.py` (one emit fix) · **Artefacts:** `out/tier0/`

**W1, W2 and W3 have been run against a real forward pass.** Every graph the compiler had seen before this used a hand-declared `boundary_response`; `atlas/cases/window_ns.py` calls `reference.WindowNS`. The configuration is four $128^2$ windows tiling $255^2$ at $h=1/64$ with a **shared seam layer**, against a monolithic reference that is the *same class* at $n=255$ — identical stencil, advection form, projection and sub-step rule, so the only difference between the two sides of every comparison is the cut.

**The shared layer is the choice everything else rests on.** $\Lambda_A+\Lambda_B$ is the Steklov–Poincaré operator only if both blocks act on the *same* trace; a cell-centred tiling with no shared layer gives two cells half a step apart, two different traces, and a sum that means nothing. $127+128=255$ is what makes the seam DOF a single object owned by both sides.

**The cost estimate was wrong by two orders and that reorders the page it came from.** W2 said *hours*; the whole tier ran in under fifteen minutes on a laptop, because one `WindowNS` macro-step is $0.069$ s and $m{+}1=17$ solves is seconds. The estimate had been made when the only expert in view was a neural checkpoint. **Anything on [[gap-worklist]] priced against the frozen checkpoint is mispriced for a classical expert.**

### W2 — the operator is healthy, and one declaration is not

$\beta\in[3.10,6.44]\times10^{-3}$, $\kappa\in[18.8,38.5]$, $\mu>0$ and therefore $\pi=0$ on all four seams — **every assembled seam is passive**, and §4.2's noise floor never had to fire. The finite difference is stable to **seven digits across four decades** of $\epsilon$, because a deterministic solver's probe floor is machine epsilon rather than OP-6's $10^{-6}$; the response is linear over at least six decades of trace amplitude, which is the cheapest available confirmation that a probe of a nonlinear operator is measuring a derivative and not a chord.

**The null-space check failed on its first real use, and what it caught is in the theory.** Declared $n_0(\Gamma)=1$, measured $\mathbf{0}$, on four seams and in every control, with no spectral gap at the bottom. [[probed-dtn-coupling]] §2.1 asserts both that *the elliptic part stays global* — so $\Lambda_i$ is the advection–diffusion DtN — and that *incompressibility leaves a rank-one null space* in $S$. **Those cannot both hold**: exclude pressure from $\Lambda_i$ and the constraint that produces the null space goes with it. [[interface-transfer-theory]] §7's verdict table handled it exactly as designed (deficit → `admit-uncertified`, *"either the probe is not resolving a constraint or $n_0$ is wrong"*), so the **check is vindicated even though the constant is not** — which is the outcome that machinery exists for.

**Two fluxes, one operator, and a per-block hazard.** [[probed-dtn-coupling]] proposes $\partial_n$ in §2.2 and the conservative momentum flux in §4.1 and never compares them. On the assembled seam they agree to $1.11\times10^{-9}$: the advective term is evaluated at the shared ring cell where the two sides carry the same value and opposite normals, so it cancels identically — **§4.1's refinement is a no-op exactly where it is claimed to matter**. Per block it is a factor of $10^3$, and it **flips the passivity verdict**: the upstream window reads $\pi=0$ under §2.2's flux and $\pi=2.39$ under §4.1's, on a transport term that carries no dissipation. W33 would refuse a record at admission on that number, for a graph whose seam is provably passive.

**A second block-level trap, from a control rather than an argument.** The downstream block has $\beta=2.67\times10^{-5}$, $\kappa=1529$ while its own seam is $\kappa=23$ — and a **uniform-flow control reproduces it within $8\%$ on a field with no structure at all**, so it is a property of being downstream, not of the wake. A zero-state control pins the sign convention independently: both blocks return the identical positive eigenvalue, as two mirror-image windows in still fluid must. **$\beta$ and $\pi$ belong to the seam; printing them per block invites the wrong refusal.**

$S$ drifts $0.02\%$–$0.35\%$ over ten macro-steps, so §6's amortization is priced rather than assumed and **W30 closes**. The cut score orders the seams $60$–$84$ across the flow against $237$–$247$ along it, with $\kappa$ and the operator asymmetry agreeing independently — **W16's first real data**, and cutting perpendicular to the flow is $3\times$ better, measured before any rollout.

### W1 — $L$ is below one, and OP-2's implied $\eta$ has the other sign

$L=0.948441\pm0.0048$ composed, $0.979649\pm0.00041$ monolith, **amplitude-independent to six digits**. OP-2's recorded drift implies $L=1.0143$. **Opposite signs, and it is stated as no match rather than as a match with caveats** — the two configurations differ in expert, transmission and $\Delta t$ simultaneously, which is the confound OP-2 has been narrowing for four days. What the measurement does establish: **$L>1$ is not a property of windowed decomposition as such**, since the composed map is measurably *more* contractive than the monolith it approximates — a pinned Dirichlet ring is a sink for perturbations. And it eliminates passivity as OP-2's cause here, agreeing with the probe's $\mu>0$ by a completely independent route.

### W3 — $\gamma$ is negligible by eleven orders, and $\tau$ is misnamed

Total $3.7071\times10^{-4}$; $\tau=3.7014\times10^{-4}$ ($99.8\%$), $\sigma=2.6995\times10^{-5}$, $\gamma=3.48\times10^{-15}$. **The prediction $\gamma\lll\tau,\sigma$ is confirmed by eleven orders of magnitude** — the framework's habit of reporting $\gamma$ and nothing else was reporting the term that does not matter, and that is now measured rather than argued. One correction to the row's wording: the three **do not sum** to the total and cannot, being norms of partly cancelling errors.

**$\tau$ dominates while the agent *is* the reference solver, which is only possible if $\tau$ is carrying something else.** Three measurements identify it: it is flat across a $25\times$ range of $\Delta t$ (so $\tau/\Delta t$ *diverges* — an inconsistent decomposition, not an inaccurate one), it lives away from the seams ($42\%$ beyond 32 cells), and it is the **per-window pressure projection**, identified exactly: `tile.last_mismatch` $=$ `tile.last_div` $=2.7607\times10^{-4}$ to ten significant figures, against a monolith's $8\times10^{-15}$. Each window has $\sim10^{-3}$ of net boundary flux; a homogeneous-Neumann Poisson solve cannot honour it and absorbs it as a uniform divergence. **Incompressibility is a global constraint and a per-window projection under an unbalanced ring cannot enforce it** — which is also [[probed-dtn-coupling]] §2.1's *other* premise failing, in the build the page names as runnable.

> **[[plug-in-composition-theorems]] §1.4's attribution theorem, observed rather than argued.** The dominant $\tau$ is a decomposed elliptic solve — a $\sigma$ one level down, wearing a $\tau$ label because of where the harness cuts. Its **[AI Inference]** that OP-5's expert-versus-architecture share is depth-relative now has a measurement behind it, and **W34's depth tag stops being cosmetic**: without it, this number reads as *"the expert is bad"*.

### The finding that costs the most: the interface condition is a steady condition

**Solving $\sum_i\Lambda_i\lambda=\chi$ exactly makes one composed macro-step $8.9\times$ worse than not solving it.** The interface residual falls $500\times$ while the answer degrades — [[schwarz-iteration-atlas-0.1]]'s warning about removing the only visible symptom, arriving from a direction that page did not anticipate.

**The decisive measurement is one line and it should have been on [[probed-dtn-coupling]] §9's list.** Substitute the reference solution's *own* trace into the interface residual: it goes $2.6421\times10^{-5}\to2.6520\times10^{-5}$ — **the true solution does not satisfy the condition**, and marginally less so than doing nothing. So the condition is not an approximation of a true statement about $\lambda$; it is a different statement, and driving it to zero moves $\lambda$ **$41\times$ past** the truth. Refining $\Delta t$ makes the overshoot **grow** — $41\times$, $75\times$, $150\times$ at $0.05$, $0.01$, $0.002$ — which names the mechanism: **a condition whose solution is $\Delta t$-independent is a steady-state condition**, and the trace's own equation after one explicit macro-step is not.

$\mathcal E_i$ is written throughout §2 as the solution operator of a *boundary-value* problem. Every expert this vault can probe is the one-step map of an *initial*-boundary-value problem, and the page never states the condition for that case. **The right condition is on the time-integrated flux — [[general-coupling-scheme]]'s R9, so far written only for multirate — with $\lambda$ carried as a function of time, which is waveform relaxation.** W7 and W17 are both promoted; W17 stops being a deferred refinement and becomes the correct formulation.

**Why the damage is that large is the bound's own arithmetic.** $\delta=-S^{-1}r$ multiplies by $1/\beta\approx190$, so a residual that is a modelling artifact rather than a physical imbalance arrives as a large trace error. [[master-error-bound]] §4 says $\beta$ does exactly this; what no page says is that **the numerator has to be a real imbalance for the bound to be about anything**.

> **The construction's own diagnostics were all green while it was doing this.** $\kappa\approx20$, $\beta$ three orders above the floor, $\mu>0$, $\gamma$ at machine zero, the probe linear over six decades — every number [[probed-dtn-coupling]] §9 proposes to check came back healthy. **A construction can pass every diagnostic it proposes for itself while being wrong about what it is solving**, and the missing test costs nothing: check that the reference's own answer reduces your residual.

### The master bound, evaluated

With all constants measured: $(\tau+\sigma)/(1-L)=7.70\times10^{-3}$. **It holds at every step, is tight at $N=1$ (ratio $0.93$) and $14\times$ loose asymptotically** — the right shape, since at one step the bound *is* $\tau+\sigma$ and over a rollout it assumes worst-case alignment of defects that partly cancel. **The measured error saturates** at $5.4\times10^{-4}$ from $t=2$ through $t=6$: a plateau, not a drift, and the first coupled configuration in this vault observed to settle. **The first time the bound has been a number rather than a shape.**

### What did not move, and is not being rounded up

**$\lVert\mathcal A\rVert$ was not computed.** `AssemblyCertificate` stays a named hole, `condition` stays empty, E6 stays `unchecked`; the partition-of-unity residual is machine zero *by construction* on a shared-layer tiling, which is weaker than a check that could have failed. $\Xi$ was not measured, so §4.5's positive control is still unrun. $\lVert\Lambda-\tilde\Lambda\rVert$ was not measured, so the $\sigma$ above is an observed field defect and **not** the bound's product form — §4's factorization remains untested. `differentiable` is `NONE`, and that is **tested, not assumed**: `step_batch` returns numpy at the class boundary by deliberate design, so `torch.func.jvp` raises `Cannot access data pointer of Tensor that doesn't have storage` whatever the backend, and §5.2's "second unspent use of the differentiability claim" stays unspent. The cross-point was declared, not treated. **No graph reaches `admit`**, and nothing here was going to make one: $\lVert\mathcal A\rVert$ is unmeasured and the assembly has no accuracy condition.

**Every number is one configuration, one state, one expert** — a developed wake at $t=5$, $\Delta t=0.05$, $\nu=1/255$. [[probed-dtn-coupling]] §7 says this out loud about probes; it is equally true of $L$.

**`atlas/` changed in one place.** `SeamOperator.as_dict` dropped `passivity_lambda_min`, so the assembled operator — the one the scheme is built on — emitted only the clipped $\pi$, against §4.2's explicit requirement that $\mu$ be emitted raw beside it so the clipping is visible rather than silent. The block emitted it and the seam did not. 120 tests pass unchanged.

---

## [2026-08-27] note | Tier 8 acted on — two new compiler rules, and the composition becomes faithful rather than merely close

**Page:** [[tier0-measurements]] §8 · **Code:** `atlas/capability.py`, `atlas/compiler.py`, `atlas/assembly.py`, `atlas/graph.py`, `atlas/cases/window_ns.py`, `scripts/tier0_window_ns.py` · **Artefacts:** `out/tier0b/`

The morning's measurement found two things wrong and left them as worklist rows. Both are now **rules in `atlas/`** rather than notes beside it, and the numbers moved by one to three orders of magnitude in every diagnostic.

### The three-ingredient trap, which is the methodological result

The diagnosis said $\tau$ was the per-window pressure projection. Three fixes followed from it, and **each was measured alone and each looked like a failure**: widening the halo from 1 to 31 cells bought $28\%$; a partition of unity that vanishes at artificial edges bought **nothing**; a global projection applied after assembly bought $9\%$ and, at larger halos, made things *worse*. Three refutations of a correct hypothesis.

**All three are necessary and none is sufficient.** Together, $\tau$ goes $2.935\times10^{-4}\to1.335\times10^{-6}$, a factor of **220**. The control that should have been run before any of them — four windows each covering the *whole* domain, which returns exactly $0.0$ — would have established the harness was sound and sent the search straight to the elliptic part. **A composition fix is not separable into independent tests when its ingredients are a boundary condition, the region it contaminates, and the weight that region is given.**

### R10 — an embedded elliptic sub-solve is not decomposable

`capability.EllipticSubsolve` is a new declared field and **L2 refuses** a decomposition of an agent that embeds a global elliptic solve. A projection method's pressure Poisson solve is global over whatever domain it runs on, so cutting the domain cuts the operator; the error that follows is elliptic — **flat in distance from the cut, flat in $\Delta t$, and untouched by any halo, partition of unity or interface condition**. It is the silent-wrongness class exactly: no exception, no failed gate, and a number that reads as a bad expert.

**It also poisons the interface operator, which nobody predicted.** Taking the projection out and running it in the composition layer improves $\beta$ by $\mathbf{69\times}$ ($5.05\times10^{-3}\to0.349$), $\kappa$ by $\mathbf{18\times}$ ($21.7\to1.196$), the cut score by $570\times$, and the operator's asymmetry by $\mathbf{76\times}$, to $0.002$. **With the elliptic part where [[probed-dtn-coupling]] §2.1 says it belongs, the advection–diffusion DtN is nearly self-adjoint** — the pressure coupling was what made it non-normal. An interface problem at $\kappa\approx1.2$ is not a hard problem, and the morning's $\kappa\approx20$ was measuring a decomposed pressure solve rather than a seam.

### R2b — probed-DtN needs an implicit macro-step

`capability.TimeDiscretization` is the second new field. $\sum_i\Lambda_i\lambda=\chi$ is the interface condition of a **boundary-value problem**; discretize implicitly and every macro-step is one, discretize explicitly and there is none. R2 lifts the rung from what the expert *accepts*; **R2b gates it on what the expert *is***. Recorded as an `admit` rather than a refusal, because the compiler corrects the rung and leaves nothing silently wrong — which is the split rule the three verdicts turn on.

A **halo rule** joins them: `required_halo() = stencil_radius * substeps_per_macro_step`, the distance an artificial boundary's influence travels before the next exchange corrects it. The working configuration declares $2\times10=20$ and carries $21$.

### $n_0(\Gamma)$, resolved rather than merely contradicted

The morning measured $\dim\ker=0$ against a declared $1$ and concluded the declaration was wrong. **The resolution is better than that and it rescues [[interface-transfer-theory]] §7:** $n_0$ is not a property of the seam alone, it depends on **where the elliptic solve lives**. Inside $\Lambda_i$ incompressibility constrains the trace and $n_0=1$; in the composition layer $\Lambda_i$ is the pure advection–diffusion DtN and $n_0=0$. The declaring field is `elliptic_subsolve`, and §7's verdict table did its job — the deficit row's second reading was right, for a reason the row could not have anticipated.

### The strongest single number

| map | $L$ | std. err. |
|---|---|---|
| monolith $n=255$ | $0.979650$ | $0.00041$ |
| **corrected composition** | $\mathbf{0.979644}$ | $0.00041$ |
| as-built composition | $0.972776$ | $0.00043$ |

**The fixed composition's stability constant matches the monolith it approximates to five decimal places**, well inside a standard error; the as-built one is seventeen standard errors away. That is the difference between a composition that is *accurate* and one that is *faithful* — it is not producing a similar answer, it is the same dynamical map to the precision the fit resolves. **And it settles something for OP-2**: a correctly built decomposition does not change $L$ at all. The morning's $0.948$ was a defective *assembly* over-damping, not decomposition damping.

### W45 — the ingest path that did not exist

`_Context.unmeasured()` hard-coded `["L", "sigma", "tau", "C_mu"]` and consulted no record; `ExpertCapabilities.L_fitted` existed and **was read by nothing**. So a constant could be measured, published and still reported missing. `graph.MeasuredConstants` now carries $L$, $\tau$, $\sigma$, $\gamma$, $C_\mu$ and $\lVert\mathcal A\rVert$ **with probe state, scheme and depth**, because a constant measured at one state under one scheme at one depth is not the same constant elsewhere. Verified: five unmeasured constants go to zero, and E5 flips `unchecked` to `holds` with the horizon branch named.

### $\lVert\mathcal A\rVert$ was a field that could never report anything

`PartitionOfUnity.norm_A()` computed $\lVert\sum_i R_i^\top\chi_iR_i\rVert_2$ — which **is $\lVert I\rVert=1$ whenever the identity holds**, so it could only return a number once the thing it was checking had already failed. Fixed to the constant the bound actually needs, $\max_j\sqrt{\sum_i\chi_{i,j}^2}$, and the answer is **$1$ for any partition of unity, by proof rather than measurement** — attained at every single-owner cell, and every decomposition has an interior. So the master bound's $\lVert\mathcal A\rVert$ factor is not a free constant, and W28's second field closes without a measurement. A `GridPartitionOfUnity` was added alongside, because the dense form needs a $65025$-square matrix ($34$ GB) for a $255^2$ field.

### The one measurement that came out badly, and it opens a gap

[[master-error-bound]] §4's product form $\sigma\le\frac{C_\mu}{\beta}\lVert\Lambda-\tilde\Lambda\rVert\lVert\lambda^\star\rVert$ is now fully measurable — and **vacuous for an overlapping scheme**. A halo scheme uses no interface operator, so $\tilde\Lambda=0$ and the bound reads $\sigma\le\mathbf{1.72}$ against a measured $3.73\times10^{-8}$: an overestimate of $4.6\times10^{7}$, i.e. $C_\mu^{\text{implied}}=2.2\times10^{-8}$. The reason is structural — the bound assumes the error enters through an interface *solve*, and **there is no term for overlap width anywhere in the derivation**, which is where a halo scheme's accuracy actually comes from. $C_\mu$ is left **unmeasured**: back-fitting it from $\sigma$ would make the bound tautological, and $10^{-8}$ is a diagnosis of the factorization rather than a value for it. Opened as **W49**.

### Where the compile lands, and what is left

`as-built` **refuses** with one refusal (R10). `split-step` is **`admit-uncertified` and runnable with zero refusals**, stamp E1–E5 and E7 `holds`, E6 `unchecked`. **`admit` is now blocked by exactly one thing, and it is a named hole rather than a missing measurement**: `AssemblyCertificate`'s `condition` — nothing states that a blend must be at least as accurate as the local solves it blends, so E6 cannot be stamped. The blend defect *is* measured ($\alpha=5.7\times10^{-15}$, negative: the blend is not worse than the worst thing it blends) and unconstrained.

$\Xi$ is measured for the first time and it is nonzero: **$0.19$–$0.53$** for the as-built agent against the corrected one, with $\lVert\Lambda^{\text{emb}}-\Lambda^{\text{exp}}\rVert/\lVert\Lambda^{\text{exp}}\rVert=0.78$–$0.92$. That is R10 in operator norm, from two assemblies and no rollout. It is **not** §4.5's positive control, which needs a frozen periodic checkpoint and stays unrun. W48 is stronger than it looked: the downstream block's near-singularity **survives** the fix ($\kappa=7061$ against its seam's $1.196$, now $5900\times$ apart), so it is genuinely a property of being the outflow side, and a per-block gate would refuse a seam whose conditioning is essentially perfect.

**120 tests pass throughout.** Everything here is one configuration, one state, one expert.

---

## [2026-08-28] note | Atlas 0.1 — the assembly condition closed, the σ bound scoped, and the time-integrated interface condition refuted

Second session against `reference.WindowNS`. Full record: [[tier0-measurements]] §9; the ledger is §9.8. **Everything from 2026-08-27 was reproduced from scratch before anything was touched** — the compile, $L=0.979644$, $\kappa=1.196$, $\tau=1.3353\times10^{-6}$, $\sigma=3.7334\times10^{-8}$, 141 tests — and then four of that session's conclusions were corrected.

### The last named hole is closed, and the condition is on $\chi$

`AssemblyCertificate.condition` asked for *the inequality $\alpha$ must satisfy*, checkable without the exact solution. **No inequality on $\alpha$ can be it**, and the measurement is what says so: both candidate definitions of the blend defect read as *passing* on a deliberately non-convex partition whose composed macro-step is $166\times$ worse. $\alpha$ compares the blend against the *local solves*, and a blend can beat every one of them while still leaving their hull.

The condition is **L6/C1: $\chi\ge0$ with $\sum_i\chi_i=1$**, and it follows from the cellwise identity

$$\bigl|A(u)-u^\star\bigr|^2=\sum_i\chi_i\bigl|u_i-u^\star\bigr|^2-V_\chi,\qquad V_\chi=\sum_i\chi_iu_i^2-\Bigl(\sum_i\chi_iu_i\Bigr)^2$$

which holds for *any* weights summing to one and whose $\chi$-weighted variance $V_\chi$ — needing no exact solution — is non-negative for all data **exactly when** $\chi\ge0$. So convexity is necessary as well as sufficient, and it is read off the declaration. **R11** refuses; E6 stamps `holds`; the certificate is issued at compile time from the declaration alone. Measured: the identity closes to $10^{-15}$ on the real tiling for all six partitions tried, signed included.

**Two corrections came with it.** §8.7's $\alpha=5.7\times10^{-15}$ was measured on inputs the PoU identity zeroes for *any* partition — the reference cut into pieces and blended back — so it was the `norm_A` failure repeated one field over. And $\lVert\mathcal A\rVert=1$ holds for any **convex** partition of unity: the word was missing, and a signed one measures $1.7038$.

### W49 — a σ bound that applies, and the first measured $C_\mu$

§4 of [[master-error-bound]] is now scoped to the **substructuring** branch and **§4.1** carries the overlapping one:

$$\Pi=\max_j\sum_i\chi_{ij}\mathbf 1\!\left[b_i(j)\le d_i\right],\qquad \sigma\le C_\mu\,\Pi\,\lVert\delta\lambda\rVert$$

$\Pi$ is the weight the assembly gives to cells a stale artificial-boundary datum can have reached; it is $1$ when the overlap is narrower than the domain of dependence and **$0$** when the partition vanishes over the contaminated band. **The bound holds on 16 configurations at $1.02$–$4.04\times$**, against §4's $4.6\times10^{7}$ on the same scheme. $C_\mu$ is **measured** — $[0.228, 1.178]$, so $1.2$ — and quoting it is a different act from the back-fit §8.8 refused, because this factorization keeps one constant while §4's needs $2.2\times10^{-8}$.

**The row asked for "a term for overlap width" and the measurement says that is the wrong term.** At fixed ramp, widening the halo $11\to61$ leaves $\Pi$ and $\sigma$ essentially unchanged; widening the *ramp* at fixed halo moves $\sigma$ by $1.9\times10^{4}$. Overlap width is a precondition; the partition of unity is the mechanism.

### The time-integrated interface condition was built, and it does not work

§5.1 promoted W7 and W17 on the grounds that the right condition for a one-step map is time-integrated. **Built as a $W=2$ waveform and measured, it fails both of §5's tests identically to the pointwise one** — reference-trace ratios $1.002$–$1.040$ under both conventions at three $\Delta t$, and solving it degrades the composed step by $2.00\times$ against pointwise's $1.59\times$.

The reason is an identity. At a shared-layer seam both sides pin the same cell, so $F_A+F_B=-\nu h\,\partial_{nn}w$ **exactly** — verified bit-exactly on all four seams against the monolith's own solution. **The flux-balance residual is a discrete second derivative, not a jump**: $O(h)$ on the exact solution, vanishing only where it is linear across the seam. The defect is *spatial*, so integrating in time cannot touch it. R9 reverts to being a multirate rule; **the construction valid for an explicit one-step map is the overlapping halo update itself**, with $\Pi$ as its bound. That closes W46.

### Five configurations, and one of §8's headline numbers broke

$\mathrm{Re}$ and $\Delta t$ each varied $4\times$. Every §8 claim survives, and the composed $L$ matches the monolith to within **$0.08$ standard errors everywhere** — stronger than §8.6's single-configuration statement. But **$220\times$ is not a property of the split-step construction**: at $\Delta t=0.025$ and $0.10$ it collapsed to $1.8\times$ and $2.1\times$, with $\tau$ two orders too large and $\sigma$ and every probe diagnostic unchanged. `SUBSTEPS = 10` is a constant tuned to $\Delta t=0.05$, where it happens to equal the monolith's own internal sub-step count; the composition layer applies the exposed elliptic part once per exchange, so a different cadence makes the composed step a *different splitting* of the same equations. Matched, the improvement is $138\times$–$361\times$. That is **R10b**.

> **This is the attribution theorem for the third time on this project, and the third time it was a composition-layer defect wearing an agent's label** — after the decomposed pressure solve and the partition of unity with full weight at the artificial edge. All three were silent, all three read as agent infidelity, and W34's depth tag catches none of them. Opened as **W54**: a defect needs the harness parameter it is a function of, not only its depth.

### $\Xi=0$, the positive control, run

`reference.SpectralNS` through the whole compiler: $\lVert\Lambda\rVert$ **bitwise zero** on 34 solver calls, `L5/R6` refuses the solvers, and the run carries **`refused claims: ["the word 'coupled' on any output involving seam sx0"]`** — spec §6.4(a), from a real periodic solver. So the $\Xi=0.19$–$0.53$ of §8.3 is not an artifact of a probe that manufactures small operators.

### Where the compile lands

`split-step`: **`admit-uncertified`, zero refusals, zero unmeasured constants, all seven envelope hypotheses `holds`, and one decertification.** §8.9's *"blocked by exactly one thing"* was wrong — there were three, and one of them (`eps_tol`) was the W45 bug at a second call site, asserting $\tau$ and $\sigma$ unmeasured on a graph that declares both. What is left is **`L2/G5/W16`**, the cut-score decomposition policy's **[AI Inference]** status: identified, never derived, never tested. **A theory-status item, not a missing measurement** — the first time this vault can say what a graph needs to reach `admit` and have the answer be a page rather than a run.

**191 tests pass** (141 before this session). New: `tests/test_tier9_assembly_condition.py`, `tests/test_tier9_interface_condition.py`. New scripts: `l6_assembly_condition.py`, `l6_macro_consequence.py`, `w49_sigma_halo.py`, `w7_time_integrated_interface.py`, `tier0_sweep.py`, `xi_positive_control.py`, and `vault_scan.py` — W38's byte-level scan, as a script rather than a method note. **Still one expert (W55), one topology, one governing family.**

## [2026-08-28] note | The decomposition policy is derived and its scalarization is falsified — the first `admit`

`G5/W16` was the last thing between `cases/window_ns.py` in `split-step` mode and `admit`, and [[tier0-measurements]] §9.7 was explicit about what kind of thing it was: *"cut where the exact operator is closest to local and the conditioning is comfortable"*, adopted as policy with its **[AI Inference]** status preserved — identified, never derived, never tested. Five configurations agreed with it, and **agreement across five configurations of one graph is not a derivation**. §10 closes it both ways it could be closed, and the two halves have opposite verdicts.

### Derived — L2/C2, and "closest to local" finally denotes something

For subdomain $i$, let $D_i=\mathcal E_iR_i-R_i\mathcal E$ — the failure of the exact one-exchange-interval operator to commute with restriction to $\Omega_i$. Given the partition of unity,

$$\mathcal A\bigl(\{\mathcal E_iR_iu\}\bigr)-\mathcal Eu \;=\; \sum_i R_i^\top\chi_i\,D_iu$$

**exactly**, for any local maps, no PDE and no convexity required. Add $\chi\ge0$ — L6/C1, which R11 already enforces — and it bounds cellwise by $\sum_i\chi_i\lvert D_i\rvert\le\max_i\lvert D_i\rvert$. **So the criterion is: minimize the $\chi$-weighted restriction defect**, a theorem on hypotheses the compiler already checks. It is the same cellwise argument that closed `AssemblyCertificate`: **L6/C1 read on the solution is the assembly condition; read on the defect it is the cut condition.**

Measured: the identity closes to $1.95\times10^{-12}$ on the strip model over 16 runs and to $3.6\times10^{-11}$ against `reference.WindowNS`'s four windows, with the cellwise bound violated by at most $2.2\times10^{-16}$ anywhere. On the real tiling the $\chi$-weighted bound is $2.6268\times10^{-7}$ against a measured one-interval defect of $2.6216\times10^{-7}$ — **tight to $0.2\%$** — while the max form is $5.7525\times10^{-5}$, $220\times$ loose. The gap between the two forms *is* the ramped partition of unity's contribution, since $\chi$ is small exactly where $D$ is large.

The criterion also has a **reference-free** form: on the overlap $\mathcal E_iR_iu-\mathcal E_jR_ju=D_i-D_j$, so the neighbours' disagreement is the part of the defect the cut itself creates, and both local solves are already computed. It **equals** the max form exactly when the agents' contaminated bands are pairwise disjoint — gap $0.000$ at every halo $\ge2\rho s$ and $6.6\times10^{-3}$ below it. On the four-window tiling that condition *fails* (1120 declared cells shared) and the surrogate agrees to $1.00007$ regardless, which is a measurement and not a licence.

### Falsified — $Q$ ranks cuts backwards, three separate ways

No graph in this vault could test a cut criterion: the four-window tiling's cut is at the centre by construction. `tests/strip_model.py` is the instrument — a linear advection–diffusion torus with two overlapping strips and a rotating cut, everything else held fixed. **Neither a fixture nor a case study**; it earns nothing on Tier 0 and measures no constant of any real expert. It is a model problem for a *rule*.

| criterion | rank correlation vs the measured composed defect |
|---|---|
| $Q=\beta^{-1}\lVert S-\operatorname{diag}S\rVert/\lVert S\rVert$ | $\mathbf{-0.853}$ |
| the un-normalized off-diagonal mass alone | $\mathbf{+0.853}$ |
| $\lVert\sum_i\chi_i\lvert D_i\rvert\rVert$ (L2/C2) | $\mathbf{+1.000}$ |

**Following $Q$ costs $92\%$ of the available range** — it picks very nearly the worst cut on offer. And the diagnosis is surgical: the off-diagonal mass carries the right information and **both operations $Q$ performs on it invert the sign**. Normalizing by $\lVert S\rVert$ removes the magnitude that matters; dividing by $\beta$ imports the *interface solve's* amplifier into a scheme that has no interface solve. **This is §8.8's error one field over** — W49 already scoped [[master-error-bound]] §4 to substructuring for exactly this reason, and $Q$ was still applying §4's amplifier to cut placement. The mechanism is one line: $\beta\propto\nu$ for a diffusive DtN, so $Q$ prefers cutting where the coupling is strongest, and strong coupling is what a lagged artificial boundary gets wrong.

Two further falsifications, independent of the ranking. **$Q$ is not a function of the decomposition**: replacing a declared prolongation $P$ by $PU$ for orthogonal $U$ declares the *same* interface space and leaves the scheme bit-identical, while $Q$ moves from $0.5016$ (Fourier) to $0.001388$ (eigenbasis of the symmetric part) — a $361\times$ orbit with $\beta$ invariant. Since §8.3 measured the split-step operator's asymmetry at $0.002$, every seam of the real graph could be declared into a frame where $Q$ reads zero. And **$Q$ is blind where the answer is not in the operator**: with a cut-independent operator it is constant to $8\times10^{-11}$ while the measured defect spreads $2.29\times$. The decision a wind farm actually faces — cut through the wake or beside it — is not expressible in the operator alone.

`cut_score` is **retired as a criterion** and kept as a substructuring diagnostic with its scope in its own docstring, for the same reason §9.1 kept the max-of-norms blend defect: deleting the thing a finding is about leaves the finding unreproducible.

### W56 — `admit` was reachable with an unmeasured constant

Found by the first graph that ever reached `admit`, which is the only way it could have been found. `_w49_sigma_branch` quoted the module default `C_MU_HALO` for graphs declaring no $C_\mu$, while the same compile listed it on `unmeasured` — **a field every compile reported and nothing read**. Invisible for as long as every graph carried a decertification anyway. **The W45 / W52 bug at a third call site**: `holes.Unmeasured` exists so that `float(L)` *raises* rather than returning a plausible number, and $1.2$ is what sixteen configurations of one solver measured, not a default for somebody else's. Fixed at the site, and backstopped: a non-empty `unmeasured` now forces `admit-uncertified` citing `L8/W56`, so the fourth such call site does not have to be found the hard way.

### Where it lands

**`split-step`: `admit`. Zero refusals, zero decertifications, zero unmeasured constants, E1–E7 all `holds`.** The first `admit` this package has issued for any graph, and `README.md`'s standing sentence *"`compile_scheme` still never returns `admit`"* retires with it. Stated narrowly: the bound applies as a theorem because the axis is overlapping and the partition convex, and its value is measured and declared with provenance. **It does not mean the cut is optimal** — the criterion ranks cuts and this graph declares one — and it does not extend past this expert, state, cadence or topology.

**213 tests pass** (191 before). New: `tests/test_tier10_cut_policy.py` (19 tests on the rule and the algebra), `tests/strip_model.py`, `scripts/w16_cut_policy.py`. Opened: **W57** (no cut criterion is derived for the substructuring branch — L2 decertifies there citing it, so the gap is a verdict rather than a silence) and **W58** (the record does not say which of the two measurements `cut_defect_bound` is). **W55 — one expert — is untouched and remains the largest caveat.**

## [2026-08-28] note | W55 closed — two more experts, every rule survives, and a learned operator finds a live bug

**W55 was the largest standing caveat on this project**: every constant in the vault — $C_\mu$, $\Pi$'s calibration, R10b's cadence rule, $L$, $\tau$, $\sigma$, $\beta$, $\kappa$, $\mu$, `cut_defect_bound` — came from `reference.WindowNS` and its no-projection subclass, plus one `SpectralNS` control that measures zero by construction. Two more real experts now exist and the whole Tier 0 stack has been re-run against both.

- **`reference.ChannelNS`** — a second **discretization**: advective-form advection instead of skew-symmetric, a rolled Laplacian, and `pressure.project_outflow`, a MAC projection with **Neumann at the inlet and walls and Dirichlet at the outlet**.
- **Poseidon-T** — a frozen **20.8M-parameter neural operator**, which is not a discretization at all.

### R10 survives losing the mechanism it was measured by

The sharpest result, and it was set up to be able to fail. §4.1 identified the elliptic defect by a *specific mechanism*: `WindowNS`'s all-Neumann pressure solve is **singular in the constant mode**, so it needs the ring's net flux to balance, and a subdomain of a through-flow violates that by construction. That was 99.8% of the first composed defect and it is the measurement R10 came out of.

**`project_outflow` has no compatibility condition at all** — the outlet Dirichlet lets the imbalance leave. So R10's measured mechanism is *absent* on the second expert while its stated reason — an elliptic operator is global over whatever domain it runs on — is untouched. Measured: $\tau$ for the embedded agent is $2.6977\times10^{-2}$, $2.6251\times10^{-2}$, $2.5686\times10^{-2}$ at $\Delta t = 0.05, 0.01, 0.002$ — **flat to a ratio of $1.05$ over a $25\times$ range**, which is §4.1's signature exactly. **The rule generalizes past its own evidence**, and this is the first time that has been shown here.

The magnitude does not transfer and was never expected to: `ChannelNS`'s embedded defect is $92\times$ worse than `WindowNS`'s, so the split-step improvement is $21257\times$ against $217\times$.

### What transferred, and one thing that did not

| | `WindowNS` | `ChannelNS` |
|---|---|---|
| L2/C2 identity residual | $3.59\times10^{-11}$ | $3.22\times10^{-11}$ |
| L2/C2 bound **tightness** | $0.9980$ | $\mathbf{0.9983}$ |
| reference-free surrogate ratio | $1.00007$ | $1.00006$ |
| blend defect, real local solves | $-1.041\times10^{-2}$ | $-1.045\times10^{-2}$ |
| composed $L$ vs its own monolith | within $0.01$ se | within $\mathbf{0.00}$ se |
| R10b, cost of a cadence mismatch | $\sim200\times$ | $\mathbf{1231\times}$ |

**The one clean disagreement is attributable, which is what makes it worth having.** §8.3 read `WindowNS`'s near-self-adjointness ($0.002$) as the signature of the pressure coupling being gone. `ChannelNS` reproduces it across the flow ($0.0038$) and **not along it** ($0.29$) — because the skew-symmetric form is antisymmetric by construction and the advective form is not. **A property that looked like a property of the exposed elliptic part is partly a property of the advection form**, and only a second discretization could separate them.

### $\Pi$ transfers exactly; $C_\mu$ transfers as a bound and not as a calibration

Six configurations of `ChannelNS`, varying the ramp and the halo. **$\Pi$'s two limits reproduce to the digit** — $0.750$ at ramp 1, $0.020$ at ramp 21 — because $\Pi$ is computed from the declared geometry and **is not a measurement of the expert at all**. §9.2's mechanism claim reproduces too: halo $11\to41$ leaves $\Pi$ unchanged and $\sigma$ within 15%, while ramp $1\to21$ moves $\sigma$ by $313\times$. *Overlap width is a precondition; the partition of unity is the mechanism.*

**$C_\mu$ is the one that splits.** Implied $[0.0881, 0.7075]$ here against $[0.228, 1.178]$ there — **overlapping**, with $C_\mu=1.2$ bounding all 22 configurations of both experts. So it is a safe **bound** across experts and a **calibration** only within one: $1.2$ is conservative by $1.7\times$ for `ChannelNS`, and `ChannelNS`'s $0.088$ would be wrong for `WindowNS` by $13\times$. A page quoting $C_\mu$ should now say which of those two facts it needs.

### The first probe floor this vault has measured

Poseidon-T is bitwise deterministic on repeat calls and has a **batch-position spread of $6.618\times10^{-7}$** — OP-6 asserted $10^{-6}$ and had never been tested; the measurement is $0.66\times$ it. The finite-difference probe is stable in $\lVert S\rVert$ over $\epsilon\in[10^{-1},10^{-3}]$ and then scales as $1/\epsilon$ below $10^{-4}$, implying a noise floor of $\sim1.8\times10^{-7}$ — **agreeing with the batch-position measurement to within a factor of 3, by an independent route**. §1's *"the probe floor is machine epsilon, not OP-6's $10^{-6}$"* is correct **and scoped to a float64 solver**.

**And $\beta$ is untrustworthy where $\lVert S\rVert$ is not**: it moves $2.2\times$ across an $\epsilon$ window where the norm is stable to 0.5%, because the smallest singular value inherits the whole noise floor. Since $1/\beta$ is the substructuring amplifier, **the quantity the bound divides by is the least measurable thing on a learned expert** — one more reason §10.2's retirement of a $1/\beta$ cut criterion matters.

The learned seam is **passive** ($\mu = +1.53\times10^{-3}$, $\pi = 0$), so E7's hypothesis is tested on a genuine expert swap for the first time. And $\Xi = 0.0061$ — the first *cross-solver* composability index, giving the vault three points: $0$ exactly (no channel, §9.5), $0.006$ (a learned map whose ring is an initial condition), $1$ (a solver's Dirichlet ring).

**What it cannot do is structural.** The checkpoint is fixed at $128\times128$ and the wrapper raises otherwise, so there is **no same-class monolith at any resolution** and $\tau$, $\sigma$, $C_\mu$ and L2/C2's reference form are unavailable rather than unmeasured. L2/C2's **reference-free** surrogate is the only form of the cut criterion it can carry — which is the case §10.1 derived it for, arrived at from the other direction.

### W61 — R2b left the rung lifted, and only this expert could have shown it

**R2b** exists because §5 measured that solving the flux-balance interface condition makes a composed step $8.9\times$ worse than not solving it *while every diagnostic reads healthy*. For `explicit` it corrects the rung. For **`unknown`** it emitted a decertification ending *"It is not assumed to"* — **and left the rung lifted**, giving a **runnable** probed-DtN scheme on a non-overlapping axis. That is the silent-wrongness class by the split rule's own definition.

**Nothing found it because nothing had ever declared `unknown` on purpose.** `window_ns` and `channel_ns` declare `explicit`; `wind_farm` and `rocket` declared nothing, and `unknown` is the field's **default** — so the fixture whose documented purpose is the rung-lift chain was getting its lift *from an omission*. Poseidon-T is the first expert for which `unknown` is the honest answer: a frozen one-shot map is neither explicit nor implicit.

Fixed: the rung is held at `dirichlet` and **the decertification stays**, which differs from the `explicit` branch on purpose — there we know probed-DtN is wrong, here we know only that it is unestablished. Both fixtures now declare `time_discretization` explicitly.

> **W45, W52 and W56 were all *"a measured value exists and one call site does not consult it"*. W61 is the neighbouring failure: a default that reads as permission.** `unknown` should be the most conservative value of its field and it was the most permissive. The other enum with an implicit-permission default is `EllipticSubsolve`, which has no `unknown` at all — opened as **W60**.

**224 tests pass** (213 before). New: `tests/test_tier11_second_expert.py`, `atlas/cases/channel_ns.py`, `atlas/cases/poseidon.py`, `scripts/w55_second_expert.py`. Opened: **W59** (`BCChannel` cannot express a ring that is an initial condition — $\Xi=0.0061$ is the size of the difference), **W60** (`EllipticSubsolve` has no `unknown`; `poseidon.elliptic_signature()` measures the statistic against §8.3's two calibration points and reads *consistent with embedded* for Poseidon-T, as a diagnosis and not a proof), **W62** (`ChannelNS`'s constants are measured and not yet declared). **One topology still, on all three experts.**

## [2026-08-28] note | The topology — W33 run, W6 treated, and three defects found by aiming existing machinery at real objects

Three rows had the same shape and had had it for a long time: **the machinery exists and has never been pointed at anything real.** W33's conformance suite was built with every field's test implemented and had never certified an expert. W6's cross-point rule was written, conditional on the decomposition, with a definition of done — *"a test on a 4-agent junction showing $\beta$ does not collapse"* — that had never been run on any graph. W5's `ADVEC` passengers, `ROT` port and field-to-lumped seam were fixture-tested only.

All three ran. **Every one produced a finding, and none of them needed anything new built.**

### W33 — and the suite's first run committed the error its own theory page had closed

Five real records plus one that lies. The liar is the same expert with the same weights and **one field changed** — `bc_channel` claiming `DIRICHLET` where the probe measures $\Xi=0$ — and it is **refused at admission**, which is the row's definition of done. `validity` decertifies on all six, which is correct rather than a gap: it is falsifiable and not verifiable, so the certificate says *"not falsified on suite X"* and never *"valid"*.

**W63.** `_test_storage` refused when the probed **block** had a positive passivity defect. W48 was closed earlier the same day with the statement *"the conformance suite certifies on the seam and reports blocks as diagnostics"*, and the suite did not. The vault had already measured why it matters twice — a seam at $\kappa=19$ whose block was $2260$ (§2.3), and $1.196$ against $7061$ (§8.5). Then the suite ran and refused **Poseidon-T** for a **block** defect of $3.218\times10^{-3}$ on an expert whose **seam** defect is $0$.

Fixed: with a seam measurement the verdict is the seam's and the block is a diagnostic; without one the block number is reported and the field **decertifies**, because refusing on a block is the error W48 named.

> **The fourth instance of one pattern, and a new sub-species of it.** W45, W52, W56 and W61 were *"the theory says X and one call site does not do X"* about a **measured constant** or a **default**. This one is about a **level** — the seam versus the block — and the page said which level for a whole session before the code did.

### W6 — the test passes, and the defect is somewhere else

**First, the thing that made it untreatable.** A cross-point coupling is the off-diagonal block $\partial(\text{flux on face }B)/\partial(\text{trace on face }A)$ for two faces of **one** agent. The declared interface is `Callable[[str, ndarray], ndarray]` — *"trace on $V_i$ $\to$ flux on $V_i$"* — one port in, that same port's flux out. **No sequence of single-port calls produces an off-diagonal block**, because each call restarts from the base state, so a per-seam assembly is block-diagonal *by construction* and $\beta_{\text{global}}$ equals $\min_s\beta_s$ for reasons that have nothing to do with the physics. **The vault has been declaring cross-points and refusing on them at L2 without ever being able to measure one.** Opened as **W64**; the generalized probe lives in the harness because putting it on the record is a `PortAmendment`-sized decision.

With the multi-port probe, on the real four-window junction: the seams **do** see each other — about $1.8\%$ of the global operator's norm is off the seam-diagonal — and **$\beta$ does not collapse**, $2.8773\times10^{-1}$ either way, a factor of $1.00$. W6's stated test passes. The primal treatment costs $64\to61$ multipliers and leaves $\beta$ unchanged to four digits.

**$\beta$ was the wrong instrument.** Each seam carries its own multipliers over its own ring, so the shared cell gets a value from every seam containing it — and they differ, because each seam truncates a *different* function to the same 16 modes:

| component | seams | values | exact | spread |
|---|---|---|---|---|
| $u$ | `sx0`, `sx1` | $1.08511$, $1.06876$ | $1.12383$ | $1.62\%$ |
| $v$ | `sy0`, `sy1` | $0.01154$, $-0.02530$ | $0.01848$ | **$66.7\%$, and the sign disagrees** |

**The cross-point cell is multi-valued, badly, in the transverse component, and $\beta$ says everything is fine.** It is a property of the **declared interface space** rather than of the expert, so no better probe removes it — which is exactly what a primal single-valued corner degree of freedom removes, for a reason the rule did not state because the rule was written about $\beta$. And under the overlapping tiling there is **no common cell at all**, so the *"no action under overlapping"* branch is confirmed geometrically.

### The eight-agent topology, with real experts

`cases/wind_farm_real.py` keeps the fixture's topology exactly and replaces every `boundary_response` with a real one: six `reference.WindowNS` windows cut from six *different* places, two `disk.ActuatorDisk` rotors. Counted rather than claimed: **11 `MECH` + 4 `ADVEC`** connections, four `ADVEC` ports carrying a real `h0` passenger with agent `N` carrying one on **two different faces** (the per-face case a per-agent list cannot express), and **two `ROT` ports, `OUT`, unconnected** — genuinely open ports with a measurable power flow. `split-step` compiles with **zero refusals**. The experts and ports are real; **the geometry is schematic and the file says so.**

One claim did not survive being derived: this began as *"six cross-points instead of one"* and the fifteen edges contain exactly **one** 3-clique, $\{N,F,B_p\}$ — the **same** count as the four-window tiling.

**W66 — the scale set is checked and the callable is not.** The `ADVEC` port's first version returned $(\mathbf u\!\cdot\!\mathbf n)h_0$, the enthalpy **power**, against a normal-velocity trace, and the compile failed **E7** with a passivity defect of $3.915\times10^{-2}$ on the assembled seam `e5a`. The number was real and measuring nothing: the pairing was (velocity $\times$ power), which is not a power. The declared scale set says the conjugate pair is (specific total enthalpy, mass flux), so a velocity trace demands an **effort** response; returning $h_0$ makes **E7 stamp `holds`**. `check_scales` takes `(port_type, scales, passengers)`, never sees the callable, and `PortDecl` has no field naming which conjugate half the response is.

> **This is §2.2 / W47 arriving on a new port type.** There a *transport* term in the MECH flux convention reported a passivity defect of $2.39$ on a passive block, and the fix was to pin the diffusive form as normative. An `ADVEC` port has **no diffusive form to fall back on** — transport is the whole of what it carries — so the conjugate pairing is the only defence, and it is the thing nothing validates. It is the silent-wrongness class **inverted**: a false alarm rather than a false pass, and no cheaper for it.

**W65 — two rules that are each right, meeting on one record.** The rotors declare `elliptic_subsolve = none`, which is true, and `governing_family = "incompressible-navier-stokes-2d"`, which they must or E3 fails at every rotor face. R10's guard challenges exactly that pair: *"an incompressible solver almost always contains a pressure solve."* Right about solvers, wrong about closures, and **the guide's own instruction is what walks every lumped closure into it.** It decertifies rather than refuses, which is what keeps it survivable.

**234 tests pass** (226 before). New: `tests/test_tier12_topology.py`, `atlas/cases/wind_farm_real.py`, `scripts/w33_conformance.py`, `scripts/w6_cross_point.py`. Opened: **W64**, **W65**, **W66**. `W6`, `W33` and `W5`'s port-algebra cases are `done`.

> **The recurring shape, now five times.** §5 (every diagnostic green, the wrong equation), §9.1 (both blend defects passing, $166\times$ worse), §9.4 (every probe diagnostic healthy, two orders in $\tau$), §12.1 (a block verdict condemning a passive seam), §12.2 (a cross-point cell multi-valued while $\beta$ reads perfect). **The reported diagnostic is healthy and the thing that matters is not.**

## [2026-08-29] note | The fifth expert — E3 and the bc_channel ladder meet a real disagreement, the conjugate pairing becomes a declaration, and the substructuring branch gets its criterion

§12.5 ended with a diagnostic rather than a number: four mechanisms produce the holes on this project, and **only the first is found by auditing.** Mechanisms 2–4 — *the declared interface cannot express a phenomenon that exists*, *validation checks the declaration and never calls the callable*, *composed diagnostics never proven jointly sufficient* — are found only by pointing unmodified machinery at an object nobody wrote with this framework in mind. This session was planned on that basis. **Mechanism 1 produced nothing, because there was nothing left to audit; mechanisms 2, 3 and 4 produced everything below.**

Everything rests on `solvers/compressible2d.py` and `solvers/thermostruct2d.py` in the build repo, imported **unmodified**, each graded against closed-form oracles by that repo's own M1 suite.

### The fifth real case study

`cases/thermal_seam.py` — a conjugate-heat-transfer seam: a compressible gas duct against a thermoelastic shell segment across one `THERM` port. Four things in this vault that had never met a genuine disagreement meet one here: **two different `governing_family` strings**, **two different `bc_channel` values** (`DIRICHLET` against **`ROBIN`**, the first real expert above `dirichlet` on the ladder), **`IMPLICIT`** time discretization, and an honestly **`EMBEDDED`** elliptic sub-solve that is not a pressure projection. The experts and the seam are real; **the geometry is schematic and the file says so.**

### E3, against a disagreement that is a fact about the solvers — W55's remaining half

The spec's claim is precise: E3 fails, $\tau$ goes `UNDEFINED`, and *the composition still runs*, because probing mentions no governing equation. On `split-step` at a matched clock, a graph with **zero refusals**: E3 **`fails`**, $\tau$ **`UNDEFINED`** on the seam, and the probe assembles $\beta = 4.8062$, $\kappa = 1.000231$, $\dim\ker = 0$. **Every clause holds.**

And one thing nobody asked for: **the multiphysics seam is the best-conditioned interface problem in this vault** — $\kappa = 1.0002$ against `window_ns`'s $1.20$ split-step and $21.7$ as-built — which inverts the intuition that different equations must couple worse than the same ones. Opened as **W71**.

### W66 — the conjugate pairing, and why a declaration is the only fix

`PORT_SPECS[THERM]` declares the bond $(T,\ q_n/T)$ and says in its own note that $(T, q_n)$ is *"the pseudo-bond most co-simulation codes exchange."* **Every thermal solver in the build repo computes $q_n$.** So the wrong convention arrives by itself rather than by construction, and three candidate checks were measured against it.

**The operator cannot say.** Returning the pseudo-bond leaves verdict, stamp, null count and passivity defect **numerically identical**; only $\beta$ moves, from $4.806$ to $456.5$, and nothing checks $\beta$. Fitted, $S_{\text{heat}} = 94.97\,S_{\text{entropy}}$ to $6.4\times10^{-5}$ — a **positive rescale**, and $\operatorname{sym}(cS) = c\operatorname{sym}(S)$ keeps every eigenvalue's sign, so E7's passivity test is *exactly* invariant. **On `ADVEC` this class was a false alarm; on `THERM` it is a false pass.**

**The magnitude cannot say.** The power identity C4 enforces is $s_e s_f = s_P$, so a properly nondimensionalized port declares all three halves at scale $1$. Measured on `wind_farm_real`, where $U_\infty = 1$: the correct `ADVEC` response and the incorrect one differ by **5%**.

**The power balance cannot say either.** $\mathrm{d}H/\mathrm{d}t$ against $\int_\Gamma e\cdot f$ separates by a factor of $900$ on `THERM` — but only because `THERM` is dimensional and the factor is $T$ in kelvin. On the nondimensional port: $0.473$ against $0.499$.

One statement covers all three: **the wrong half differs from the right one by an $O(1)$ factor once nondimensionalized, and no dimensionless diagnostic separates $O(1)$ from $O(1)$.** The framework's own hygiene requirement destroys the information the check would need.

So it is declared: `ports.ResponseHalf`, `PortDecl.response_half`, and **L3/C9** — same half both sides `admit`, different halves **refuse** (the assembled operator would add an effort to a flow), undeclared **decertifies**. **The positive control is the number that matters**: before C9, `window_ns` — the only graph in this vault that reaches `admit` — compiled to **`admit`**, E7 `holds`, zero refusals, with one side of every seam returning the *velocity* where its declared pair says *traction*.

**And what it does not do**, recorded rather than absorbed: a *false* declaration is undetectable by all three legs. `response_half` is the **second unverifiable declaration** in this architecture after `validity`, where §12.1 said there was exactly one. **W69**.

### W64 — decided, and L2's refusal made honest

**The record is not extended.** §12.2 measured what a multi-port response would buy and it buys a diagnostic no rule consumes: the coupling is $1.8\%$ of the operator norm, $\beta$ does **not** collapse ($1.00\times$), and the real defect is multi-valuedness, which a *declaration* fixes and no probe does. A `PortAmendment`-sized widening of the one interface every expert must implement, for a number nothing reads, is a cost with no verdict attached. `CROSS_POINT_COUPLING_SCOPE` states it permanently and both cross-point branches cite it in the artifact.

**L2's refusal stands and its reason is replaced.** It cited $\beta$ and *"degrades the conditioning"*; §12.2 measured both as false. It now cites `cross_point_multivaluedness`, which is **decidable from the declaration** — a cell in two seams under a non-overlapping axis is multi-valued, full stop.

### W57 — L2/C3, derived and measured, and §4's own product form ranks backwards

`master-error-bound` §4 chains an identity and then a bound. Stopping one step earlier removes the operator mismatch entirely:

$$S_M(\lambda^\dagger - a^\star) = \chi_M - S_M a^\star =: -r \implies \lVert\lambda^\dagger - a^\star\rVert \le \frac{\lVert r\rVert}{\beta}, \qquad Q_{\text{sub}}(\Gamma) = \frac{\lVert S_M a^\star - \chi_M\rVert}{\beta}$$

Measured on `tests/substructure_model.py` — a steady advection–diffusion torus cut by the **exact algebraic Schur complement**, a movable cut, everything else fixed, and **exact to $5.3\times10^{-16}$ at the full interface space** so every defect it reports is the declared space's truncation. Over **11 configurations**: ranks **positive in 11/11** ($+0.579$ to $+1.000$), tight to a factor of $1.7$, and following it costs at most $\mathbf{1.105\times}$ the best available cut — against the retired $Q$'s **92%** of the range.

**Two results beyond the criterion.** $1/\beta$ **helps here and inverted the ranking there**, beating the bare residual in 11/11 — which measures §4's own scoping box instead of restating it. And **§4's product form ranks negative in 9 of 11**: it is a bound and must never be a criterion — and it is the shape the retired $Q$ had, which is the cleanest explanation of why that one ranked at $-0.853$.

### W60 — closed by an enum, because the measurement meant to close it is unsound

W60 proposed promoting `elliptic_signature` from a diagnosis to a gate once it had more than two calibration points. **They arrived, from physics that is not Navier–Stokes, and they falsify the promotion.** The shell's backward-Euler conduction is a *known* embedded elliptic part and the statistic reads $\kappa = 1.0002$, asymmetry $8.8\times10^{-5}$ — which its own thresholds call *"consistent with EXPOSED or NONE"* — and stays wrong across a $10{,}000\times$ range of exchange interval.

**The diagnosis is exact:** the statistic measures **non-normality** and has been read as measuring **globality**. Those coincide for a pressure projection, where the elliptic part is a constraint that makes the interface operator strongly non-normal, and come apart for a **self-adjoint** operator — $(M/\Delta t + K)$ is SPD, so its Schur complement is near-normal *however global it is*. The class it cannot see contains every diffusion problem. Opened as **W68**; `EllipticSubsolve.UNKNOWN` is added and decertifies at `L2/R10/W60`.

### W59 — not closed, deliberately, and sharpened instead

The bar was a second learned expert or a re-reading showing the enum load-bearing wrong. **Neither arrived, so no value was added.** What did arrive is the exact trigger, read off the compiler's own rung table: `bc_channel` decides the rung **only** when `time_discretization = implicit`. Poseidon-T declares `unknown`, so **W61's fix currently masks W59 completely** — the rung is `dirichlet` either way. The two errors cancel and the cancellation is not stable.

### W72 — the standing verification had the same blind spot twice

W38 fixed the `\r` blind spot by reading bytes. **A tab is a legal markdown character**, so `\t` — `\tau`, `\theta`, `\times`, `\top`, `\tilde`, `\text` — corrupts into something the control-character sweep is *right* not to flag. Found the way W38 was: **one edit produced both signatures at once**, `\beta` into 0x08 (caught) and `\tau` into a tab (not). The vault uses no tabs for any purpose, so the check is free; a planted-tab positive control is part of the fix.

### State

**256 tests pass** (234 before). `python scripts/vault_scan.py wiki` → 190 files, 0 problems. `python scripts/tier0_window_ns.py` reproduces **bit-exact** from cold — 545 numbers across `w1`, `w2`, `w3`, zero drift. **`window_ns` in `split-step` still reaches `admit`** with L3/C9 in place, which is the point of adding it that way.

New: `atlas/cases/thermal_seam.py`, `tests/substructure_model.py`, `tests/test_tier13_pairing_and_substructuring.py`, `scripts/w57_substructuring.py`, `scripts/w67_thermal_seam.py`. Closed: **W55** (remaining half), **W66**, **W64**, **W57**, **W60**, **W72**. Opened: **W68**, **W69**, **W70**, **W71**. Sharpened, not closed: **W59**.

> **The recurring shape, now six times, and a seventh of a new kind.** §5, §9.1, §9.4, §12.1, §12.2, and now §13.3 — where a wrong conjugate pairing leaves verdict, stamp, null count and passivity defect *bit-identical* to the correct one. The new kind is §13.6: **a diagnostic that measures the right thing about the wrong quantity**, correct on the two calibration points it was fitted to and wrong on the first independent object it ever met.

## [2026-08-30] note | Four of Tier 13's five open rows were one question, and answering it moved beta by 12.6x

§13 closed five holes and opened five. **Four of the five opened turned out to be one question** — *how much of a probed block is the expert's operator, and how much is the boundary condition sitting on top of it?* — and one statistic answers all four. Measuring it opened a sixth nobody predicted.

### The statistic

$\omega = \lVert S - cI\rVert_F / \lVert S\rVert_F$ with $c = \operatorname{tr}S/n$: how much of a probed block is **not** a scalar. Nothing already in the record could see this, because a multiple of the identity is healthy by every diagnostic the probe emits — $\beta$ fine, $\kappa = 1$, null count $0$, passivity defect $0$ — and all four are then properties of a **declared constant**.

`thermal_seam`'s seam is that case. It assembles from $+h_{\text{in}}I$ (shell) and $-k_{\text{gas}}/\delta n \cdot I$ (gas): **the difference of two film coefficients**, $\omega = 2.0\times10^{-5}$. Against `window_ns`'s $0.5268$ (embedded) and $0.0467$ (exposed) on the same instrument, a **2370$\times$** separation, with the probe's own $\omega$ floor another $10\times$ below.

In the spatial domain it is unmistakable: a delta at one seam cell moves the **gas** response at that cell by $9.26\times10^{-2}$ and every other cell by **bit-zero**, and moves the **shell** over a clean 5-cell exponential — $4.71\times10^{-1}$, $1.59\times10^{-3}$, $1.83\times10^{-4}$, … So the shell's globality is **present and resolved**, sitting $300\times$ below the boundary condition on top of it.

### W68 — the 2026-08-29 diagnosis was right and one level too high

§13.6 said `elliptic_signature` measures non-normality and was read as measuring globality. True, and it predicts that measuring the shell's **own block** rather than the assembled seam would help. **It does not** — the block reads $\kappa = 1.0001$ too, because the block *is* $5.0001\,I$. The statistic was **swamped, not blind**, and no threshold on $\kappa$ or the asymmetry could have recovered anything. So the fix is a **precondition and a third outcome** — `NO OPERATOR RESOLVED`, which is neither calibration point and not a weaker form of either — plus `L4/operator-content`, which decertifies a thin seam and names $\beta$: every bound carrying $1/\beta$ there is scaled by a declared constant rather than a measured operator.

### W71 — a closed form instead of a paragraph

$$\omega \approx C\,\mathrm{Bi}\,(k_{\max}\ell)^2, \qquad \mathrm{Bi} = \frac{hd}{k}, \quad \ell = \sqrt{\alpha\Delta t}$$

$C \in [0.135, 0.307]$ over **11 points spanning five decades of $\Pi$**, from two independent sweeps that agree. The two groups are the two ways an operator can hide — the film swamps the body, or the exchange interval is too short to cross one interface wavelength — and **either alone suffices**. That is why §13.6's $10{,}000\times$ sweep of the exchange interval moved nothing: it varied $\ell$ with $\mathrm{Bi} = 0.033$ held fixed. **Conditioning at a seam is set by locality, not by whether the two sides solve the same equations.**

### W74 — the probe was linearizing about 0 K

Found while writing W69's `stencil_radius` test, which measured the shell's spread as $0$ where the standalone said $5$. The base trace was **hard-wired to zero**, and this port's effort is a temperature in kelvin: the gas was asked for its wall flux against a $0$ K wall.

| | base $0$ K | base = operating point |
|---|---|---|
| $\beta$ | $4.8062$ | $0.3807$ |
| $\lVert S\rVert_F$ | $19.227$ | $1.523$ |
| verdict, decertifications | `admit-uncertified`, 7 | **identical** |

**The conclusions were robust; every number the certificate carried was not** — while `probe_state` said `"T_hot=900 K"` throughout. For an affine expert the base is genuinely free and the old docstring's bias-cancellation argument is correct; it says nothing about the linearization point, which is the part that is not free. And the cause is **the declared bond itself**: `PORT_SPECS[THERM]` pairs $(T, q_n/T)$ so effort times flow is a power — the choice §13.3 made deliberately — and dividing by $T$ is exactly what makes the response nonlinear. The pseudo-bond would have been affine. Not radiation: `radiate=False` changes nothing to six digits.

`probe_base` is added, defaulting to zeros, so every earlier measurement reproduces bit-for-bit — verified from cold, 6058 of 6059 numbers identical, the one difference being wall-clock seconds.

### W69 — three unverifiable declarations, not one, and a rule missing an input

§12.1 said one; §13.3 made it two; the audit says **three**, and it was three before W66. `governing_family` is a free-form string, compared across a seam by E3, read by nothing else, with **no test at all** — and unlike the other two, a matching false pair **promotes**: E3 reads `holds` and $\tau$ is attributed under a hypothesis that does not hold.

Two adjacent-but-testable fields now have tests: `deterministic` (one solve; it sets the probe's step, so a false value contaminates every column of every block) and `stencil_radius` (two solves; along-seam spread is a **lower bound** on the reach, so it falsifies an under-declaration and never confirms one). Writing the second found `required_halo()` returning `stencil_radius × substeps` **without ever consulting `time_discretization`** — an *explicit* agent's domain of dependence applied to both, while `_halo_rule` **refuses** an overlap below it.

### W70 — the coupling needs no port; the output needs disclosure

The 2026-08-29 framing was aimed at the wrong target. **The thermoelastic coupling is not between two agents** — it is one expert's internals, and one-way (`step_thermal` takes no mechanical argument), so the thermal subsystem is closed and E7 against `storage` is not wrong. The elastic energy is $2.0\times10^{-5}$ of the thermal.

What is real is the sensitivity: over the retained band $\delta\lVert\sigma\rVert/\lVert\sigma\rVert$ **rises** $5.6\times$ to $1.74\times$ the certified flow's, while $\delta\lVert u\rVert/\lVert u\rVert$ **falls** $1207\times$. Displacement integrates and stress differentiates, so the modes a 16-mode truncation discards drive the stress hardest — **the truncation argument that bounds the certified flow points the wrong way for the uncertified stress.** Recorded as `VOLUMETRIC_COUPLING_SCOPE` rather than a disclosure field, because the field would be a fourth unverifiable declaration.

### W72 — and it found a live corruption

Adding one byte to a hand-written list leaves a hand-written list. The escape table is now **derived from Python's own decoder**; a **`$`-parity check** catches the class with no list at all, including `\n` — legal markdown, invisible to any byte rule, and the leading character of `\nu`, `\nabla`, `\neq`; and a **generated positive control per byte** makes a missing detector a failing test.

On its first run it found `spec-wind-farm-wake-atlas-0.1.md`'s **W11 go/no-go row** split by an eaten `\rvert`, **in the vault since 2026-08-26** and through both of W38's fixes. It passed for three independent reasons, each now closed: the `\r` arrived as a plain LF; the split left a two-line fragment below the ragged-table check's three-line minimum — *a corruption that splits a row disables the check by destroying the contiguity it counts*; and the tail was `vert`, while the list carried `\rVert`.

### State

**285 tests pass** (256 before). `python scripts/vault_scan.py wiki` → 190 files, 0 problems. `python scripts/tier0_window_ns.py` reproduces bit-exact from cold. `window_ns` in `split-step` still reaches **`admit`** with 0 refusals and 0 decertifications.

New: `scripts/w73_locality_and_scope.py`, `tests/test_tier14_locality_and_scope.py`, `probe.operator_content`, `capability.probe_base`, `L4/operator-content`, three conformance tests. Closed: **W68**, **W69**, **W70**, **W71**, **W72**. Opened: **W74** (instance fixed, class open), **W75**, **W76**, **W77**.

> **The mechanism tally, and it sharpens §12.5 rather than confirming it.** Every hole closed here was reachable by **auditing**, and not one had been found that way — `required_halo()` sat through four audits, and the probe's zero base is one line with a docstring that discusses the choice and omits the assumption. What made both findable was a measurement taken for an entirely different purpose. **Auditing finds mechanism 1 once you know which rule to audit; contact is what tells you which.**

## [2026-08-29] note | The probe's base belonged to the seam, not the expert — and the elliptic gate was looking through the wrong basis

§14 closed W74's instance and left the class in a sentence: *nothing detects that a port spec's own choice of bond has invalidated a probe assumption.* Chasing it found that the field §14 added was **on the wrong object**, closed W77 with it, inverted W75, and cost the vault its only clean `admit`.

> **Dating.** §14 and its log entry are stamped `2026-08-30`; they were written on **2026-08-29** — the system clock, the environment date and the mtimes of `atlas/probe.py` and `wiki/log.md` all agree. Nothing is renamed, because this log is append-only and rewriting it would hide the drift rather than record it. So this entry sits below an 08-30 one and the ordering is by append.

### W74's class — a sum of Jacobians is a Jacobian only if the terms share a point

$$\Lambda_M = \sum_i P_i^{*}\,\Lambda_i\,P_i$$

`probe_base` went on `ExpertCapabilities`, one record per **expert**, so each side of a seam names its own and nothing compared them. On `cht` the two are each honest and independently correct and **500 K apart**: the gas linearizes at the wall temperature it sees ($400$ K), the shell at the gas temperature it sees ($900$ K), on one interface variable.

The consequence is not a shifted number. Over the whole physically admissible interval — the interface temperature is trapped between the reservoirs, $250$ K and $900$ K — a **common** base gives $\beta \in [0.4200,\ 1.7689]$, and the mismatched probe reports $\mathbf{0.37565}$: **below the entire range**, so it is a value attained at no admissible state. At the consistent $\lambda^{*} = 371.97$ K where the two one-step fluxes balance, $\beta = 1.23807$ — a further $3.30\times$ after §14's $12.6\times$, in the opposite direction.

`base_sensitivity` is the check whose *passing* promotes: below $10^{-6}$ the response is affine, the base is provably free, and every base decertification goes away. The affine fixture reads $5.4\times10^{-11}$; the gas $6.1\times10^{-3}$ and the shell $1.4\times10^{-2}$ per 1% of base moved. **This is not $\varepsilon$-stability**, which §14 measured at $1.0003\times$ over four decades and drew nothing from: that is curvature at the base, this is whether the base is the right place.

### W75 — falsified, then closed with a different instrument

§14 opened W75 guessing the resolving cadence might be unrunnable. **Three of eight cadences satisfy both requirements**, so that is falsified. It also does not matter, because on the *same shell under two solvers* — backward Euler against explicit sub-stepping, same mesh, material and film coefficient — `elliptic_signature` reads

| $\Delta t$ | implicit $\kappa$ | explicit $\kappa$ |
|---|---|---|
| $1$ | $1.0214$ | $1.0210$ |
| $100$ | $3.1965$ | $\mathbf{9.6687}$ |

identical at the first and **ranked backwards** at the last, asymmetry flat at $5\times10^{-6}$ throughout.

The reason is the **basis**: a Fourier mode is global by construction, so a block built from smooth modes cannot report whether the operator behind it was local — and `EllipticSubsolve` is defined as *"a solve with an INFINITE domain of dependence"*, a statement about support. Poke a delta instead. An implicit macro-step inverts $(M/\Delta t + K)$ and **the inverse of a sparse SPD matrix is dense**; an explicit march's response is exactly zero past `radius * substeps`. Five known points — $1.000$, $0.188$, $0.021$, $0.986$, $0.210$ — every declaration corroborated, two solves each, no threshold on any spectral quantity, and it **resolves `EllipticSubsolve.UNKNOWN`**, which W60 asked for and W68 refused to grant from the spectrum.

Only the *global* reading is positive: a dense tail that underflows is indistinguishable from a compact one, and the same implicit shell reads $0.875$ at the matched clock where $\sqrt{\alpha\Delta t}$ is $60\times$ below one seam cell. The gate wants a **short** interval, the spectral one a long one; they are complements.

### W76 — the certificate that cannot fail

$\operatorname{diag}(1, 10^{-12}) + \operatorname{diag}(0,1)$ has ideal diagnostics and a singular block, so non-sufficiency needs no case study. Measured: `sx0`'s worst block $\kappa$ is $\mathbf{4170.9}$ under an assembled $1.196$ — $3486\times$ — and its $\beta$ is $1.67\times10^{-5}$ under an assembled $0.349$. Read off `alpha_star`, **the first rule ever to read it**, the dominant side holds a median $0.987$ at all 16 modes on `sx0` against $0.543$ and none above $0.90$ on `sy0`. The vertical seams are one-sided because the flow is.

The derivable consequence: a substitution moves the assembled operator by at most the swapped agent's own block, so when $\lVert S_i\rVert < \beta - \beta_{\min}$ an **under-responding** replacement cannot be caught — including an expert that ignores its boundary data entirely, which is the `bc_channel` failure conformance calls the foundational hole. Replacing `sx0`'s 16.7% agent with exactly that is certified at every $\beta_{\min}$ up to $0.20$, while the same swap at the balanced `sy0` is refused from $0.10$ and moves $\beta$ by $45.7\%$. **And nothing in this framework derives $\beta_{\min}$** — W81.

### W77 — derived, not declared

`probe_state` read `"duct, T_hot=900 K, T_wall=400 K"` on every `thermal_seam` certificate while the probe ran at $0$ K. It never needed declaring: the probe knows its base and the base *is* the state. Derived, it reads `gas@400; shell@900 INCONSISTENT(spread 223.6)` — the declared string was not wrong about the values, it was wrong about there being **one** of them. This removes a declaration rather than adding a fourth to W69's three.

### State, and what it cost

| graph | verdict | refuse | decert |
|---|---|---|---|
| `window_ns` split-step | `admit-uncertified` | 0 | **2** (was `admit` 0/0) |
| `wind_farm_real` | `admit-uncertified` | 0 | 11 ($+5$) |
| `thermal_seam` split-step, matched | `admit-uncertified` | 0 | 8 ($+1$) |

**The vault no longer has a clean `admit`.** `window_ns` split-step keeps zero refusals and its composition is untouched — $\sigma$ and $L$ rest on the assembled $\beta$, which is sound. What it lost is the claim that its per-agent substitutions mean anything, and that claim was never true. `wind_farm_real` firing on five seams is corroboration: a wake is one-way, so one-sidedness tracks the advection direction.

**314 tests pass** (285 before). `scripts/tier0_window_ns.py` from cold: $6107$ numbers compared, **$0$ drifted**. `scripts/vault_scan.py wiki` → 190 files, 0 problems.

New: `scripts/w78_base_blocks_and_reach.py`, `tests/test_tier15_base_blocks_and_reach.py`, `probe.support_reach` / `base_sensitivity` / `base_disagreement` / `SeamOperator.derived_probe_state`, `composition.SubstitutionCertificate.blind`, `conformance._test_elliptic_subsolve`, `L4/probe-base`, `L4/block-share`. Closed: **W74** (class), **W75**, **W76**, **W77**. Opened: **W79**, **W80**, **W81**, **W82**.

> **The mechanism tally, and it inverts §14's own lesson.** §14 concluded that contact with a *new object* is what tells you which rule to audit. This session had no new object — it ran on §14's leftovers — and still found three defects by contact with measurements taken for other purposes, while reading the code and thinking about it found nothing, in either session. **The generalization is not about new objects: a rule becomes auditable only after some measurement has named it.**

## [2026-08-29] note | E3 was gating the wrong thing for four sessions — tau at a multiphysics seam needs a referent, not a shared equation

The standing reason no multiphysics graph in this vault could be certified: `E3` compares two `governing_family` strings, and when they differ the compiler emits $\tau$ as `UNDEFINED` for both sides. `rocket.py` — seven agents, a real fluid–structure seam — is kept as a fixture rather than a target partly because of it.

### What $\tau$ is actually measured as

From `scripts/tier0_window_ns.py`, unchanged since Tier 0: $\tau = \lVert u_t - u_{\text{ref}}\rVert$ with $u_t$ the composed step given the **true** trace, and $\sigma = \lVert u_c - u_t\rVert$ with $u_c$ the lagged one. **Nothing in that mentions a governing equation.** It needs a *reference trajectory*, and at a multiphysics seam one is constructible from the agents themselves — the **tightly coupled** solve, with the interface converged inside the macro-step instead of lagged across it. That is the exact analogue of the single-physics monolith, which is likewise not ground truth but "the same expert applied without the cut".

Sharing a governing family is what lets two agents share a **monolithic** reference. $\tau$ needs a **reference pair** — and `lambda_ref`, on the record since the end-to-end spec and documented as *"tau at a multiphysics seam"*, was the declaration that one exists. It was checked for presence and **never consumed**.

### Does an attribution attribute?

Replace the gas with the same solver carrying a *known* conductivity error, leave the shell, reference against the two real solvers:

| gas expert | $\tau_{\text{gas}}$ | $\tau_{\text{shell}}$ | injected |
|---|---|---|---|
| reference (itself) | $0.0000$ | $0.0000$ | $0\%$ |
| $k\times1.05$ | $\mathbf{5.0000\times10^{-2}}$ | $0.0000$ | $5\%$ |
| $k\times1.25$ | $\mathbf{2.5000\times10^{-1}}$ | $0.0000$ | $25\%$ |
| $k\times2.00$ | $\mathbf{1.0000}$ | $0.0000$ | $100\%$ |

Exact, per agent, unswapped side identically zero, and none of it needed the families to match.

### The norm, which is why this is a module and not a rule

The master bound sums $\tau + \sigma + \gamma$ as scalars, presuming one norm. There is none: conserved variables against kelvin. And an agent's own norm can be **blind** — the gas's state is **bit-identical** at wall temperatures of 352, 400 and 450 K over one $10^{-4}$ s step while its flow moves 24%, because the isothermal wall enters through a ghost state the interior has not felt.

`PORT_SPECS` already pairs every port so **effort times flow is a power**. That is a unit neither side owns, both agree on, and every port type has, so the defect is measured in **interface power**. It inherits `response_half` (W69's second unverifiable) and refuses rather than guessing when it is undeclared.

> W66 chose $(T, q_n/T)$ over the pseudo-bond $(T, q_n)$ for a thermodynamic reason. It turns out to be what makes multiphysics attribution possible at all — the pseudo-bond is not a power. The same choice **cost** once, in W74, by making the response nonlinear in the effort.

### Two things that had to be got right on the way

**A scalar secant is not a Newton solve, and the failure is structural.** Iterating on the mean residual moves only a *uniform* shift of the trace, so the along-seam variation is untouchable: it stalls at $3.04\times10^{-3}$ from $2.32\times10^{2}$, where Newton on a finite-difference Jacobian reaches $1.32\times10^{-7}$ in **9** iterations. Reproduced in the algebra with a non-uniform root.

**$\sigma$ is a function of the lag, not a number.** At the initial condition it is $1.0000$ *exactly* — the shell starts at the lagged temperature and transmits zero power there. At the lag a real run carries (the previous step's converged trace) it is $3.392\times10^{-5}$, a $30{,}000\times$ range, and it falls to $10^{-11}$ at zero lag, which is the check that it measures the lag and nothing else. Linear only to leading order: interface power is *bilinear*, so the per-decade ratio drifts $9.99 \to 4.97 \to 3.91$ as the lag grows.

### W84 — a rule that degenerated the moment $\tau$ could be zero

`eps_tol = min(tau, sigma)`, and $\tau$ is $0$ for a composition of exact solvers. That sets the interface tolerance to zero, which no solve can meet. It had never come up because no term had ever been exactly zero — $\tau$ was `UNDEFINED` across a family boundary and nonzero within one. A term that is identically zero names no scale, so the minimum now runs over the terms that carry one.

### State

| graph | verdict | refuse | decert |
|---|---|---|---|
| `thermal_seam` split-step, matched | `admit-uncertified` | **0** | **6** (was 8) |
| `thermal_seam` as-built, matched | `refuse` | 1 | 6 (was 8) |

`tau_undefined_seams` is **empty** on a multiphysics graph for the first time, `L1/E3` **admits**, `L5/eps_tol` clears, and `unmeasured` falls 5 → 3. E3 itself still *fails* and always will — the two sides really do solve different equations. What its failure costs is the monolithic reference and nothing else.

**344 tests pass** (314 before). `scripts/tier0_window_ns.py` from cold: $6443$ numbers compared, **$0$ drifted**. `scripts/vault_scan.py wiki` → 0 problems.

New: `atlas/multiphysics.py`, `scripts/w83_multiphysics_attribution.py`, `tests/test_tier16_multiphysics.py`, `thermal_seam.MEASURED_W83`. Closed: **W88**, **W89**, W84's degeneracy. Opened: **W84** (the derivation), **W85**, **W86**, **W87**.

> **The multiphysics ledger, plainly.** Closed: attribution across a family boundary, and a commensurable norm. Still open: **multirate** (native clocks at 500:1 still refuse on E4), **interface motion**, **lumped-to-field** at $\dim M = 1$, **topology events**, and **two-way volumetric coupling** between two agents. One of four foundational blockers, and it was the one that made the class structurally uncertifiable rather than merely unfinished.

> **[AI Inference]:** nothing here had to be invented — the referent is the agents, the norm is the port algebra's own bond, the split is Tier 0's with the state norm swapped out. What was missing was noticing that E3's string comparison had been standing in for a question about *referents* since the spec was written. That is W74's failure mode again — a field on the wrong object — and it suggests the next audit is not over rules but over **what each declaration is a proxy for**.

---

## [2026-08-30] tier17 | The referent's loose ends, and a multirate rule aimed at the smaller term

Full record: [[tier0-measurements]] §17 · [[gap-worklist]] Tier 17. Reproduce with `python scripts/w87_referent_cost_and_lag.py` and `python scripts/w7_multirate_matching.py`; artifacts `out/w87/w87.json`, `out/w7m/w7m.json`.

> **Dating.** Measured **2026-08-29**, written just after midnight and stamped by the clock as **2026-08-30**, per Tier 15's rule. Across Tiers 14–17 the log reads 08-30, 08-29, 08-29, 08-30; nothing is renamed, because this log is append-only.

Tier 16's three loose ends (**W87**, **W85**, **W86**), then the blocker it had listed as foundational: multirate. All four answers differ from what the rows predicted.

### W87 — `lambda_ref` names an experiment, so it is not a fourth unverifiable declaration

Tier 16 logged it as a fourth field of `validity`'s kind because it is a free-form string. **The string is not what the rule consumes.** The rule consumes the claim that a *reference pair converges*, and `multiphysics.tight_couple` settles that in one solve — so `conformance._test_lambda_ref` runs it and the count stays at three. **A declaration that can be derived is an unwritten derivation (W77); a declaration that names an experiment is an unrun experiment.**

**And then the experiment was run against four deliberately wrong pairs, and four of five converge.** A blind reference, and even a sign-flipped one, still balance — at a different trace ($425.69$ K against the true $372.14$ K). What converging falsifies is exactly one claim: that the interface problem is not *empty*. The admissible interval is not a second discriminator either — measured, it separates one wrong pair by $1.0\times10^{-5}$ of the reservoir span. A false referent corrupts $\tau$'s **magnitude** (exactly $2.0000$ for a sign flip) while leaving its **localization** correct: every defect lands on the agent whose referent was corrupted.

### W85 — the probed $S$ is not a Jacobian, and the obstruction is dimensional

Five operating points on `thermal_seam`, $\omega$ across $17.8\times$: **not one of ten $S$ rows converges.** The step $PS^{-1}Rr$ lives in $\operatorname{range}(P)$, and $\dim M = 16$ against $\dim V = 48$ leaves 32 directions untouchable. Measured at the stopping point, the residual **in** the subspace is $10^{-13}$ and the plateau **is** the orthogonal part, $1.04\times10^{-4}$, unmoved across the whole $\omega$ range. **No threshold on a spectral quantity can fix a subspace, and the data licenses none.** What $S$ is worth is a warm start at $4.8\times$ fewer solves — and only at a consistent seam base, since at the split base it is $245\times$ worse, which is **W74's class getting a numerical consequence for the first time**.

### W86 — the slope is the seam's, the lag is the run's, and the declared number is neither

Walking the reference trajectory five macro-steps: the **slope** $\mathrm{d}\sigma/\mathrm{d}(\text{uniform lag})$ is constant to five digits ($3.45001$–$3.45013\times10^{-2}$ K$^{-1}$); the **lag** ranges $3.43\times$, non-monotone, and is **not derivable at compile time** — nothing has stepped yet, so W77's fix does not transfer. And a scalar-lag check cannot certify $\sigma$ anyway: at drifts agreeing to $8\%$, $\sigma$ differs by $1.48\times$, because its argument is the lag *profile*. **`MEASURED_W83.sigma` turns out to be a uniform-shift extrapolation** — it reproduces exactly as slope $\times$ assumed drift — and its `source` cited an artifact carrying a different number ($1.08\times10^{-11}$, the $\sigma$ at zero lag).

### W7 / R9 — the two-rate test, owed since 2026-08-27, and it is a negative

`thermal_seam` at `clocks="native"` ($500{:}1$) compiled to `refuse` on `L7/R9`. One 500-step march, sliced: the leak is **$0$ exactly at ratio 1** — the control, since one sub-step *is* the interval — and $8.6\times10^{-5}$ at $500{:}1$. Put both multirate terms in interface power over one exchange interval:

| term | value |
|---|---|
| R9's — pointwise against integrated flux | $5.9972\times10^{-5}$ |
| the lag — a stale trace over the whole interval | $\mathbf{3.7431\times10^{-3}}$ |
| ratio | $\mathbf{62.4}$ |

**R9 has refused every multirate graph over the term $62\times$ smaller than the one it does not mention.** The rule is still right in principle — over an interval the conserved quantity *is* the integral — so `graph.FluxMatching` now supplies the declaration the refusal always named, and L7 requires three things of it rather than trust: the clocks nest, both sides supply `boundary_response_integrated` (`boundary_response` restarts each call, so the integral is not recoverable from it), and `solve._port_fluxes` actually calls it. At $n = 1$ the integrated response must equal the plain one exactly, so it is not a fifth unverifiable declaration. `L7/R9/lag` decertifies beside the admission carrying the $62.4$.

### State

| graph | verdict | refuse | decert |
|---|---|---|---|
| `thermal_seam` split-step, **native** ($500{:}1$) | **`admit-uncertified`** | **0** | 10 — was **`refuse`** |
| `thermal_seam` split-step, matched | `admit-uncertified` | 0 | 6 |
| `thermal_seam` native, `flux_matching=POINTWISE` | `refuse` | 1 | the pre-2026-08-30 compile, kept reachable |
| `window_ns` split-step | `admit-uncertified` | 0 | 2 |

**And one of L7's own three requirements was false when it was written.** The admission rests on *"`solve.coupled_step` calls it"*, and it did not: `_port_fluxes` gained the branch and the scheme was never passed to it, so the branch was unreachable and the `assert` guarding it could never fire. The compile was admitting R9 on a promise the run does not keep, which is the class the three-verdict split exists to separate, arriving inside the code that implements it. No test exercised a time-integrated coupled *step* — only the compile that authorizes one. Two do now, the second being the control that the pointwise path never touches the integral.

**374 tests pass** (344 before). `scripts/vault_scan.py` → 0 problems.

New: `scripts/w87_referent_cost_and_lag.py`, `scripts/w7_multirate_matching.py`, `tests/test_tier17_referent_and_multirate.py`, `conformance._test_lambda_ref`, `graph.FluxMatching`, `MeasuredConstants.sigma_lag`, `capability.boundary_response_integrated`, `multiphysics.seam_jacobian` / `lag_distance` / `check_sigma_lag`, `compiler._r9_flux_matching`. Closed: **W87**, **W85**, **W86**, W7's multirate half. Opened: **W90** (the multirate defect has no rule), **W91** (the lag profile), **W92** (the run has the consistent base for free).

> **[AI Inference]:** three of the four findings share a shape — the rule's premise was true and its magnitude was wrong. W87's check works and sees one case in five; $S$ is the right operator in the wrong subspace; R9's integral is the conserved quantity and is $62\times$ too small to matter here. **A rule can be correct, derived, and aimed at the wrong order of magnitude, and nothing but a measurement in a common norm will say so.** Interface power has been that norm since §16.3, and the multirate ledger row has stood since 2026-08-27 because nobody put the two terms in it.

---

## [2026-08-30] tier18 | Real turbine geometry, and the attribution machinery on a frozen expert

Full record: [[tier0-measurements]] §18 · [[gap-worklist]] Tier 18. Reproduce with `python scripts/w93_wake_array.py --steps 60`; artifacts `out/w93/w93.json`, `out/w93/state.npz`. New case study `atlas/cases/wake_array.py` — the **sixth** real one, and the first whose *geometry* is real.

Tiers 14–17 built an attribution stack — `support_reach`, `operator_content`, `assemble_seam` with a seam base, `seam_defect_split`, `certify_substitution` — and every number it had produced came from a two-agent duct or from one solver tiled against itself. This tier points it at **Poseidon-T inside a real composed wake**: three turbines at $3.5\,D$ in an L, six frozen $128^2$ windows, three zero-parameter actuator disks, and the quantity a wind-farm operator is paid in.

### The geometry is chosen by the checkpoint, and the Reynolds band is the price

One macro-step must be one *native* lead, so a window spanning $S$ rotor diameters forces $\Delta t = S/20$ **and** $\mathrm{Re} = 1020/S$. The band $[10^3,10^4]$ and the lateral room a wake needs pull against each other; this case spends the band ($\mathrm{Re} = 255$ at $S = 4$) to buy $11.0 \times 7.5\,D$ of domain, and **the turbine spacing is a whole number of window strides by construction**. Two more declarations fall out of W0 §4.2's cutoff rather than out of convention: `effective_resolution` is a *wavelength* (33 modes on a 128-cell face, **9** on a 32-cell rotor face), and a **port is per face SEGMENT** — a rotor spans $1\,D$ of a $4\,D$ face, so the plane carries two ports on one ring.

### W93 — the halo rule reads a declaration, and the probe measures the same thing

W69 corrected `required_halo` once, in these words: *"that product is an EXPLICIT agent's domain of dependence."* It fixed the `implicit` branch and **left the branch where the record says nothing**, which is what a frozen learned one-shot map declares under R2b. Measured: the record says **2 cells**; `support_reach` — the same quantity, by poking a delta — reads **64**, nonzero in all 128 seam cells at every amplitude from $1$ to $10^{-3}$; the 16-cell overlap was passing on the 2. `required_halo` now returns `None` there. **A neural operator's receptive field is global by construction**, so the repair is not a bigger integer. And the same measurement resolves `elliptic_subsolve` to `embedded` — which `elliptic_signature`, the other instrument, **independently agrees with** on the first black box both have had signal on — and declaring it makes `L2/R10` refuse the whole graph.

### $\tau$ on a real pretrained expert, with its controls

Calibration first, on the reference pair at the same seam: injected errors of $5\%$, $25\%$ and $100\%$ come back as $\tau = 0.050000$, $0.250000$, $1.000000$ **exactly**, unswapped side identically zero. Then the measurement, against the WindowNS pair at $\nu = 3.92\times10^{-3}$:

| actual pair | $\tau[\text{F10}]$ | $\tau[\text{F20}]$ | $\sigma$ | sub-additive |
|---|---|---|---|---|
| the reference pair (control) | $0$ | $0$ | $0.80721$ | yes |
| **Poseidon-T on F10 only** | $\mathbf{10.74}$ | $0$ | $0.37838$ | yes |
| **Poseidon-T on F20 only** | $0$ | $\mathbf{2.02}$ | $0.80721$ | yes |

**The framework catches it and localizes it**: swapping one side leaves the other's $\tau$ at exactly zero. $\Xi = 0.0382$ corroborates — the checkpoint reproduces under $4\%$ of the boundary response the reference does, and the mechanism is **W59**, an overwritten initial condition being a weaker object than a Dirichlet channel held through a step. And the referent is under-determined by $27\times$ in cell Reynolds number while $\tau$ moves only $1.05\times$, which is what lets the $10.7$ be quoted at all.

### W95 — the referent cannot be built from the checkpoint's own pair

$96\times96$ dense FD Jacobian, $\kappa = 2.17\times10^4$, full rank at a $10^{-8}$ cut — and the same $J$ at a $2\times$ smaller probe step differs by $4.09\times10^{-2}$, so **65 of 96 singular directions sit below the probe's own reproducibility**. Newton diverges from the first iteration; truncating to the 31 resolved directions does not help, and damping to $0.2$ does not help. **Three mechanisms eliminated by measurement — W85's subspace, conditioning, step length — and the fourth unnamed.** The cost, though, inverts W85's worry: `seam_defect_split` converges the *reference* pair and asks the checkpoint for four forward passes.

### W97 — a rotor seam is blind by the cell Reynolds number

§2.2's normative `MECH` effort is $\nu\,\partial w/\partial n$ and an actuator disk's traction carries no $\nu$, so a field-to-lumped seam is one-sided by $\mathrm{Re}_h$ **before any expert is chosen**. Measured: the fluid–fluid wake seam **refuses** the WindowNS $\to$ Poseidon-T swap at every $\beta_{\min}$ tried, and the rotor seam certifies it `blind` at every one — over a range $81\times$ the fluid block's own norm. W76 found this as a property of one asymmetric tiling; it is a property of a **port-type pairing**.

### The consequence, in the units the case study is about

Four runs each, 60 macro-steps, with a turbine-free control and a single-turbine control:

| | Poseidon-T | `reference.WindowNS` |
|---|---|---|
| turbine-free drift in $\langle U_d\rangle$ | $+0.046\%$ to $+0.184\%$ | $0$ exactly |
| **R2's wake loss, against R1 off** | $+57.58\%$ | $+79.54\%$ |
| **array loss** | $\mathbf{46.79\%}$ | $\mathbf{25.27\%}$ |

**$21.5$ percentage points of array loss from swapping the fluid expert**, on identical geometry, disks, controls and macro-step — which is the whole of the quantity micro-siting and wake steering are bought to move. Neither column is validated against data: the classical solver is the *declared referent*, not ground truth, and the one independent check (momentum theory at the disk) does not rank them, because $\langle U_d\rangle$ is sampled $0.25\,D$ upstream where both must read high.

### And the absolute trace, which is what made half of this visible

`window_ns`, `channel_ns`, `poseidon` and `wind_farm_real` all write the trace as a **perturbation**, so `probe_base` is identically zero on every fluid seam in this vault and `base_disagreement` has read *consistent* on all of them — while the two sides sat at their own states. `wake_array` writes it **absolutely**; `L4/probe-base` then fires on **10 of 13 seams**, up to $123\%$ of the base norm, and the three that agree are exactly the three constructed to agree. It is also a precondition: interface power is $\int_\Gamma e\,f$, and a perturbation times a perturbation is not a power.

### State

| graph | verdict | refuse | decert |
|---|---|---|---|
| `wake_array`, `elliptic_subsolve=unknown` | **`admit-uncertified`** | **0** | 50 across 44 rules |
| `wake_array`, `elliptic_subsolve=embedded` — *what §18.4 measures* | **`refuse`** | 1 (`L2/R10`) | — |
| `wake_array`, `elliptic_subsolve=none` | `admit-uncertified` | 0 | — |

`tau_undefined_seams` is empty; `E7` **fails**, and it fails because of the learned expert rather than the topology ($\pi = 3.4\times10^{-4}$ under the checkpoint against exactly $0$ under the reference).

**397 tests pass** (374 before). `scripts/vault_scan.py wiki` → 0 problems. `scripts/w93_wake_array.py` run from cold against the same script run from a cached state: **1322 numbers compared, 0 drifted** — the checkpoint is bit-deterministic and every number on this page traces to one invocation.

New: `atlas/cases/wake_array.py`, `scripts/w93_wake_array.py`, `tests/test_tier18_wake_array.py`. Changed: `capability.required_halo`, `compiler._halo_rule`. Closed: **W93**. Opened: **W94** (the disk has no bond, by two routes), **W95** (no referent from the checkpoint's own pair), **W96** (`wind_farm_real`'s rotor returns a force density against a `stress` key), **W97** (rotor seams blind by $\mathrm{Re}_h$).

> **[AI Inference]:** three of the four findings are rules that were right and had never been evaluated where their premise fails, and the fourth is a ledger row no fixture could reach. What made all four visible is the same thing — **real geometry**. A schematic rotor hangs off any ring, so it never forces the bond question; a synthetic seam has a base of zero, so it never forces the linearization question; a solver you can run at any resolution has a monolith, so it never forces the referent question. The five case studies before this one were each honest about being a fixture in exactly one dimension, and each of those dimensions was hiding a rule.

---

## [2026-08-30] note | Wake array — a readable summary page and a live viewer

New page [[case-study-wake-array-atlas-0.1]], the short companion to [[tier0-measurements]] §18: what the sixth real case study is, how it is assembled (agents, ports, the thirteen seams), and what came out — in one readable page rather than a measurement record. Everything on it is quoted from `out/w93/w93.json`; nothing is estimated.

**And the case study is now visible rather than only tabulated.** `scripts/w93_frames.py` re-marches both experts saving the streamwise velocity every second macro-step at half resolution, quantized to `uint8` against a stated fixed range so the two animations are directly comparable; `scripts/w93_build_viewer.py` inlines those frames into a self-contained page. It carries a to-scale plan view of the array (windows, overlaps, seams, rotors, dimensioned legs), the two flow fields side by side on one timeline with the geometry overlaid on demand, the per-turbine inflow history, and the power and attribution tables. Published as an artifact.

> **A colour ramp is a claim too.** The map is fixed across both runs and both experts rather than normalized per frame, and it is centred so that the freestream $u = 1$ lands on a neutral grey: wake reads cool and dark, accelerated bypass flow reads warm. A per-frame normalization would have made the two experts' fields look alike while their velocity ranges differ by a factor of two, which is precisely the difference the page exists to show.

Verified without a browser, since a private artifact cannot be opened by an unauthenticated one: the page script was extracted and run against a minimal DOM stub, which exercises the frame decode, the colour lookup, the plan-view construction, the readout wiring and the chart, and fails loudly on a wrong element id or a bad index. Decoded fields come back at $u \in [0.137, 1.377]$ for Poseidon-T and $[0.368, 1.438]$ for the classical solver at $t = 11.6$, inside the declared ramp with no clipping.

New: `wiki/concepts/Atlas 0.1/case-study-wind-farm-wake/case-study-wake-array-atlas-0.1.md`, `scripts/w93_frames.py`, `scripts/w93_build_viewer.py`, `out/w93/frames.json`, `out/w93/wake-array.html`. Changed: `march()` in `scripts/w93_wake_array.py` gains an optional snapshot hook, which no measurement path uses. `scripts/vault_scan.py wiki` → 0 problems; 397 tests still pass.

---

## [2026-08-30] note | W98 — the viewer found a wake standing upstream of the only turbine

The page built yesterday to *show* the wake array instead exposed a defect in it,
and Tier 18 has been re-measured end to end. In the animation, Poseidon-T's panel
carried a full wake profile the whole length of the domain — including $3.5\,D$
**upstream** of R1, and along a row whose only turbine is far downstream — where
the classical referent beside it held freestream. A wake with no cause.

**The mechanism, once looked for, is that two global operators were being bought
from a per-window API.** `build` declares `pressure` a `GlobalField`; the march
asked `step_many` for `project=True`, which runs an exact Leray projection on
each 128-cell window separately and periodically, and the same call's `frame`
argument translates each window with `spectral_shift`, periodic on the window. So
a wake reaching a window's outflow edge re-entered its own inflow edge — one
circulation every $N/6.4 = 20$ macro-steps, which is why the phantom deficit was
flat in $x$: it is the wake's own $y$-profile, smeared along the direction it kept
going round. Measured with R1 the only disk, at $1.25\,D$ upstream after ten
macro-steps: $0.8244$ before, $0.9495$ after, referent $0.9966$.

> **It is not a halo failure and no overlap fixes it.** The first hypothesis was
> the partition of unity's ramp, whose docstring claims it "gives the wrapped
> strip zero weight" — it does not, the weight only reaches zero at the first
> cell. Giving the strip a true dead zone moves the deficit by $0.012$. W93's
> measured support reach is the whole window, so the contamination was never
> confined to the strip the translation wrapped, and no integer overlap was ever
> going to be the answer.

`wake_array.transport_and_project` is the repair: one projection and one
translation, on the domain, extended downstream by `PAD_CELLS` $= N$ cells of
fluctuation tapered to zero. The extension is periodic-compatible, so the
projection is exact, the translation is a phase factor rather than an
interpolation, and what wraps onto the inlet is the taper's zero — freestream.

**What it was worth.** The checkpoint's array loss falls from $46.79\%$ to
$14.62\%$ against a referent's $25.27\%$ that did not move by a digit: the
framework had been contributing about twice the error of the model it was
measuring. The disagreement also changed character rather than merely shrinking —
R2, in a developed wake, is now $4.6$ pp from the referent, while R3, in the
bypass flow beside one, differs in *sign*. And $\tau$ **rose**, $10.74 \to 11.58$,
while the trajectories converged: W98 had been degrading both sides of
the seam together, which partly cancels in a difference, so repairing the
composition made the trajectory better and the attribution sharper at once.

> **The methodological point is the one worth keeping.** No assertion caught this
> and no reasonable assertion would have: the manufactured deficit is smooth,
> bounded, physically shaped and in the right units, and every control passed
> with it present — $\tau$ calibrated exactly, the certificate refused, the
> turbine-free run sat at its noise floor. A page of correct-looking tables
> described a graph that was not the one running. **A rendering of the state is a
> measurement instrument**, and on this evidence it belongs in the same tier as
> the probes rather than in the write-up.

Also opened: **W99**, `FrozenFluidExpert.step` and `.step_many` drop `force`
silently when `galilean` is false — the block is nested inside `if gal:` — and
the same branch pins each window's output to zero mean, deleting the momentum
deficit a disk has just deposited. Both are one line in the build repo and
neither announces itself; the repair needs `galilean=False`, so both had to be
handled in `march`.

Four regression tests added, three of which need no checkpoint: the outlet does
not reach the inlet, the translation is exact and undamped, the projection is a
projection and guards **three** null wavenumbers rather than one (a Nyquist-zero
convention leaves $k = 0$ at three places on a rectangle, and guarding only the
origin returns NaN on the first step — found the hard way). The fourth marches
the checkpoint and gates at $0.92$, between the measured before and after.

Re-measured artifact `out/w93/w93.json`; the superseded one kept beside it as
`out/w93/w93.pre-w98.json`. Cold against cached: **1321 numeric values, 1314
identical to the bit**, the other seven wall-clock timings. Viewer rebuilt and
republished, with the ramp re-anchored after the frame script's new clipping
check caught it overflowing — a ramp that silently clips understates exactly the
wake the page is about. Changed: [[case-study-wake-array-atlas-0.1]],
[[tier0-measurements]] §18.5 / §18.5.1 / §18.6 / §18.6.1 / §18.6.2 / §18.11,
[[gap-worklist]] Tier 18, `atlas/cases/wake_array.py`,
`scripts/w93_wake_array.py`, `scripts/w93_frames.py`,
`tests/test_tier18_wake_array.py`. 401 tests pass; `vault_scan` 0 problems.

---

## [2026-08-30] note | The plan rescheduled — climb the ladder classically, substitute learned experts afterwards

Two new pages, written after a fresh read of the whole program against [[f1-pathmap-and-end-goal]]'s own ladder: [[case-study-ladder-to-f1]], which reschedules it, and [[cs7-scaling-ladder-pickup]], the self-contained brief for the next build.

**Where the program actually is, counted rather than felt.** Rung 1–2 of 12. Six real case studies, **all six 2-D incompressible Navier–Stokes** plus one thermoelastic seam; two governing families ever coupled, at exactly one seam; largest real graph **nine** agents against rung 9's $15$–$20$. And the three things the pathmap's own §5 calls most likely to fail — composition error versus interface count (**F1**), gradients through a composed stack (**F5**), and expert reuse (rung 4) — have **never been measured**, while each is cheaper than the case study just finished. That inversion is the scheduling error, and it is the whole reason for the first page.

**The revision, which is one observation and its consequences.** `atlas/` does not know what an expert *is*: an `ExpertCapabilities` record needs a `boundary_response`, a `dt_native`, a `governing_family` and a port list, and a finite-volume solver satisfies that exactly as a checkpoint does — four of the six real case studies are classical solvers wearing the expert interface. `composition.certify_substitution` exists for precisely one purpose, swapping one for another at a seam. So the program splits: **Claims A and B are establishable with classical experts, and learned experts arrive afterwards through a certified substitution campaign, one seam at a time.**

Four blockers leave the critical path together. **W93/R10** — a globally-receptive pretrained operator cannot be certified as a decomposition, which is nearly all of them — becomes a property of the substitution step, priced per seam, instead of a wall standing in front of every rung. **W95** — a learned expert cannot be its own referent — stops being a blocker and becomes the reason the classical expert had to exist anyway, because it *is* the referent. **F3** gets measured on the cheap expert first, so the trend is visible before it is expensive. And **rung 5's data cost is deferred rather than incurred**, on a fact about the build repo rather than an argument: `solvers/thermostruct2d.py` carries backward-Euler conduction *and* quasi-static plane-stress elasticity, with `solve_mechanical(T, p_in, p_out, ...)` taking pressure loads from both sides, graded against closed-form oracles by the build repo's own M1 suite and already imported **unmodified** by `cases/thermal_seam.py`. Its docstring says what it was built for: *"exactly the b-c and c-d coupling Atlas will later have to reproduce through its typed edges."* **The structural expert the pathmap calls the point at which the project stops being free already exists** — as a solver. What is deferred is the speed, and nothing gates on speed until rung 10.

> **What the revision costs, and it is stated on the page rather than left implicit.** A classical F1 graph is a coupled simulation, which exists already and is slow, so climbing classically proves Claims A and B and proves *nothing about the vision*. Sub-linear composition error for a family of classical solvers does not imply it for checkpoints — which is why CS-7 runs **both columns at every size** rather than one. And it defers the hardest question instead of answering it: if the substitution campaign refuses at every seam of every case study, the honest outcome is a coupling framework, exactly as §7 of the pathmap already says.

**The second reordering is by coupling kind rather than by subsystem.** An F1 car is not twenty physics problems, it is four kinds repeated — field–field surface, field–lumped surface, two-way volumetric, moving interface — and the port algebra has bonds for one and a half: the field-to-lumped certificate is **blind by the cell Reynolds number** (W97), two-way volumetric has **no port and no bond** (W70, W94), and `motion_class` is `static` with anything else refused (W22, W30). Building a subsystem before its bond exists means every subsystem case study rediscovers the same gap, which is the per-pair integration burden [[port-algebra-atlas-0.1]] was written to avoid, reappearing one level up. So the ladder builds bonds first.

**Ten case studies, CS-7 to CS-16, in three phases.** Phase A decides the forks with no new experts and no data — **CS-7** the scaling ladder (Claim B), **CS-8** the reuse probe, sharpened from *does the expert work elsewhere* into **is a certificate a property of the expert or of the state it was probed at**, which decides whether the plug-in claim is per-expert or per-*design* and therefore whether the economic argument survives. Phase B builds the missing bonds with no data — **CS-9** a volumetric bond, exercising `PortAmendment` for the first time; **CS-10** a wing in ground effect, which is the first moving interface, the first real *design parameter*, and the place W97 is closed or admitted permanently; **CS-11** a bound for the multirate lag. Phase C is the subsystems, each classical first, each scheduled only after the bond it needs exists — and **CS-14 is the rung that connects the wake array's own open `ROT` port**, which is what the pathmap means by an incomplete model stating its own incompleteness.

One **[AI Inference]** worth flagging because nobody has tried it: `support_reach` asks whether the response is *nonzero* past the declared radius, and the useful question is whether it exceeds the defect the composition already tolerates. A **tolerance-halo** — the radius beyond which the response falls below $\varepsilon_{\text{tol}}$, with the truncated tail carried as an explicit term in $\sigma$ — would be a bounded halo for an operator whose support is formally global, and `eps_tol` already exists in the compiler. Not derived; measurable with the probe that exists; attack it at CS-7.

**And a stopping rule the pathmap does not have.** If the substitution campaign returns `refuse` or `blind` at every seam of every case study, the composed model cannot be certified with learned experts at all. That end state is worth naming a checkpoint for rather than approaching asymptotically: **after CS-12**, by which point three expert families and all four coupling kinds have been tried.

**[[cs7-scaling-ladder-pickup]] is the brief for the next build**, written to be the only page a fresh session needs — which is what W24 has been asking for since 2026-08-26 and what no page in this vault yet was. It carries the construction, the five sizes with **two of them controls** ($N=1$ must return exactly zero defect, the one-line control [[tier0-measurements]] §8 says would have redirected the parent's whole search; $N=6$ must reproduce the wake array's $14.62\%$ and $25.27\%$), the six environment gotchas that have each cost a session, the three W-rows to close with a definition of done each (**W58**, **W54**, **W81**), and six measured traps — including that W98's class makes the defect look like it *improves* with size, which would be a false confirmation of the very claim the run exists to test.

New: `wiki/concepts/Atlas 0.1/common/case-study-ladder-to-f1.md`, `wiki/concepts/Atlas 0.1/common/cs7-scaling-ladder-pickup.md`. Changed: `wiki/index.md` (two rows under Atlas 0.1). No code changed and no measurement taken — **this is a plan, and a plan closes no row**. `scripts/vault_scan.py` → 193 files, 0 problems.

---

## [2026-08-31] note | Atlas mapped onto the four classical domain-decomposition families

> **Dating.** The read, the date verification (twice, `date` → 2026-08-30) and the page were done on **2026-08-30**; the filing landed eight seconds past midnight and is stamped by the clock as **2026-08-31**, per Tier 15's rule. The page body reads "written 2026-08-30" because that is when it was written; nothing is renamed.

New page: [[atlas-and-standard-dd-theory]] (`wiki/concepts/Atlas 0.1/common/`), a reading companion for someone working through Toselli & Widlund, *Domain Decomposition Methods — Algorithms and Theory*. It states, family by family, where Atlas's coupling machinery sits and which classical theorems stop applying once the subdomain solver is a frozen neural operator.

**Where Atlas sits.** Overlapping Schwarz: the as-built wind-farm coupling is **one-level restricted-additive overlapping Schwarz** run for a single sweep — Schwarz waveform relaxation converged in one step because $U_\infty\Delta t < \delta$ ([[schwarz-iteration-atlas-0.1]] §5.1) — with no classical coarse space (coarsening changes $\mathrm{Re}_{\text{eff}}$) and a $\sigma$ term set by the partition-of-unity contamination weight $\Pi$ rather than by $1/\beta$ ([[master-error-bound]] §4.1). Substructuring: the `probed-DtN` rung is **primal iterative substructuring with the Schur complement assembled by black-box probing** ([[probed-dtn-coupling]]), Newton–Krylov–Schur under nonlinearity, with the Neumann compatibility condition used as a correctness check and **no cross-point rule** — the FETI-DP/BDDC corner-primal construction is the acknowledged missing piece (G1 / [[gap-worklist]] W6). Mortar: pointwise flux matching is the over-constrained end today; [[interface-transfer-theory]]'s declared common interface space $M$ with prolongations $P_i$ **is** a mortar formulation, with the inf-sup constant $\beta$ obtained by measurement because a frozen operator has no approximation theory.

**Which theorems break, and why.** §7 tabulates it against [[prior-art-and-novelty-atlas-0.1]] §2.1's four root causes (no consistency order, no CFL, no statable Lipschitz bound, error not reducible by refinement): the Schwarz $H^{-2}$ and FETI-DP polylog condition-number bounds, optimized Schwarz's analytic optimal coefficient, Lax–Richtmyer equivalence, mortar optimality, substructuring exactness, and the SWR rate estimates all lose their hypotheses. What survives is the geometry — the additive/multiplicative argument, the dense-interface sparsity argument, the SWR condition $U\Delta t<\delta$, the partition-of-unity identity — plus one condition Atlas *adds* rather than imports: the convexity $\chi_i\ge0$ that a single-sweep blend needs and an iterated method does not (L6/C1). Every broken *number* becomes an Atlas *measurement*.

**Flagged gap.** Optimization-based / virtual-control DD (Lions, Glowinski, Gunzburger) has **no vault page** — a likely genuine gap, and a natural fit, because Atlas already claims the end-to-end adjoint a control formulation is built from and [[composition-error-theory]] §4.4 / [[probed-dtn-coupling]] §5.2 both record it as spent on nothing. Opened as [[gap-worklist]] Tier 4 row **W101**, related to W85 (a $\dim M$ probed operator handed to a $\dim V$ Newton solve stalls; a control formulation sidesteps the Jacobian).

New: `wiki/concepts/Atlas 0.1/common/atlas-and-standard-dd-theory.md`. Changed: `wiki/concepts/Atlas 0.1/common/prior-art-and-novelty-atlas-0.1.md` (See Also), `wiki/concepts/Atlas 0.1/common/general-coupling-scheme.md` (See Also), `wiki/concepts/Atlas 0.1/common/gap-worklist.md` (Tier 4 row W101), `wiki/index.md` (one row under Atlas 0.1). No code touched; `atlas/`, `scripts/`, `tests/` and the cs7 / scaling-ladder files were left alone. This is a synthesis note — it closes no measurement.

---

## [2026-08-31] note | Expert donor survey — public pretrained weights across seven governing families

Every Atlas case study to date has used one expert family, 2-D incompressible Navier-Stokes, so [[incremental-transfer-roadmap]]'s donor table ("only Poseidon and Walrus have confirmed public checkpoints") had never been tested against the question multiphysics actually asks. This note runs that test. New page: [[expert-donor-survey]] (`wiki/concepts/Atlas 0.1/common/`), surveying public literature and model hubs for downloadable weights across **compressible/transonic flow, heat conduction, solid mechanics/linear elasticity, thermoelastic coupling, electromagnetics, reacting flow, and multiphase/free-surface flow**. Each record carries license, architecture family, dimension and native resolution, **boundary-conditions-as-input**, gradient exposure (JVP/VJP), native timestep, training distribution, a frozen-vs-transfer verdict, and — the field that decides certifiability — an explicit **locality** flag.

**Two families have no donor at all.** Coupled **thermoelasticity**: nothing, anywhere, with weights; the literature is uniformly one-off DeepONets and constitutive ANNs. This independently confirms [[expert-library-atlas-0.1]]'s physics argument that the thermal-structural expert must be two experts exchanging the thermal-expansion term — the market cannot supply the merged one even if you wanted it. **Spatially-resolved reacting flow**: also nothing (BLASTNet is data). But that expert *factors*: DeepFlame's DNN chemistry integrators are a downloadable **pointwise** map from a thermochemical state to its state one chemistry substep later, so only the confined-transport half is from-scratch — a direct refinement of [[training-and-bootstrap-atlas-0.1]]'s reacting-flow row, and the split is the operator splitting reacting-flow solvers already use.

**The licensing answer is restrictive.** Poseidon's weights are **CC-BY-NC-4.0** — [[incremental-transfer-roadmap]]'s first unstarted action item, answered against the roadmap, which had built two of four bootstrap stages on Poseidon. The permissive alternatives with comparable or better compressible coverage are Walrus (MIT), DPOT (Apache-2.0) and **GPhyT (MIT, weights public at `flwi/Physics-Foundation-Model`)** — the last of which upgrades a row the vault had as "structural pattern only, checkpoint unconfirmed" from *pattern to reimplement* to *weights to load*. The NC term also plausibly propagates to Therm-FM, which is a Poseidon fine-tune and the only conduction donor found.

**Almost nothing accepts a boundary condition as an input.** [[composition-error-theory]] makes BC flexibility a selection axis; measured against real donors it mostly collapses. Poseidon, Walrus, DPOT, MPP, Bubbleformer and PhysiX have no boundary-data channel at all (trained periodic or fixed-BC); GPhyT handles new BCs in-context, which is impressive and unauditable. Only **PDEformer-2** (signed-distance-function BCs inside a symbolic PDE computation graph — and MindSpore-only, so every probe and adjoint in `atlas/` would need a bridge) and **NeuberNet** (boundary displacement *is* the input) take boundary data as a declared argument. A frozen donor with no BC channel can be driven by a neighbour only by overwriting a halo, which is the mechanism W93 decertified.

**The locality verdict is near-unanimous, and its exceptions share an architecture.** Of roughly twenty entries, three have a bounded domain of dependence: DeepFlame's chemistry MLP (zero-dimensional — the domain of dependence is a point), NeuberNet (a boundary-driven notch patch), and **MACE-MP-0** (finite radial cutoff times message-passing layers — a number derivable from the config and checkable by the same delta-poke that produced W93's 64). GNS would be a fourth if anyone had published its weights; verified this pass that `learning_to_simulate` ships datasets only. All of them are local operators or finite-radius message passing; none is a transformer or a spectral operator. So the continuum-fluid community has converged unanimously on the architectures this framework cannot certify, and **MACE is the bounded-receptive-field existence proof [[case-study-ladder-to-f1]]'s constrained-expert rung was waiting on** — it just comes from atomistics rather than from continuum fluids.

**One correction to how the halo rule should be applied.** For frequency-domain electromagnetics and quasi-static elasticity the governing equation is elliptic, so the exact solution operator is globally coupled *as physics*: even a perfect local expert has an unbounded domain of dependence. For those families the halo rule should be declared inapplicable rather than merely relaxed, with the seam certified through [[probed-dtn-coupling]] instead — otherwise Atlas will keep decertifying experts for a property their physics requires them to have.

Also recorded, so the next survey does not repeat it: searched for and **not found** — VOF/level-set free-surface weights; transonic external-aero checkpoints (the datasets exist, GAOT's RAE2822 span and SuperWing; the weights do not); engineering-regime electromagnetics outside integrated photonics; general linear elasticity.

New: `wiki/concepts/Atlas 0.1/common/expert-donor-survey.md`. Changed: `wiki/resources/code-and-papers.md` (new "Expert donor candidates" section, ~22 rows by governing family; GPhyT's `(find)` code cell filled; Poseidon's NC license, Walrus's MIT + fine-tunes, and GNS's datasets-only status annotated), `wiki/concepts/Atlas 0.1/common/expert-library-atlas-0.1.md` (new "What the donor market says about this cut" section, the internal-vs-external open question updated against six confirmed compressible donors, Related Concepts and See Also), `wiki/index.md` (one row under Atlas 0.1). No code touched; `atlas/`, `scripts/`, `tests/` and the cs7 / scaling-ladder files were left alone. This is a survey note — it closes no measurement, and every "confirmed" line was checked against the primary repository or model card on 2026-08-31.

---

## [2026-08-31] note | CS-7 — the scaling ladder, F1 measured for the first time, and a composed rollout that does not survive its own length

**The seventh real case study**, `atlas/cases/scaling_ladder.py`, and the first whose *variable* is the size of the graph. One geometry grown from $1$ to $24$ coupled windows with $\mathrm{d}x$, $\Delta t_{\text{macro}}$, the $16$-cell overlap, the $8$-cell ramp, the referent's viscosity and the disk model all held fixed, in two columns — a classical solver that has a monolith and a frozen checkpoint that has none at any resolution ever (**W95**). Full record [[tier0-measurements]] §19; rows on [[gap-worklist]] Tier 19; readable companion [[case-study-scaling-ladder-atlas-0.1]].

**The criterion.** [[f1-pathmap-and-end-goal]] §5's **F1** fails if composition error grows *super-linearly in interface count*; §3.2 calls rung 9 *the rung that decides everything* and schedules this sweep as its early warning. It had never been run at any size, and [[case-study-ladder-to-f1]] schedules nine further case studies on the answer.

**F1 is not falsified.** Composed defect against overlapping window pairs, four rungs, ordinary least squares on the logs with $t_{0.975,2} = 4.303$: **$+0.804\ [+0.641, +0.967]$** classically, **$+0.477\ [+0.362, +0.593]$** on the checkpoint, **$+0.851\ [+0.748, +0.954]$** against the monolith. Sub-linear in every column. The test is deliberately one-sided — F1 fails on an interval lying *above* $1$, and gating on "upper bound at or below 1" would report a linear result as a failure, which is the criterion tightened after the fact.

**Both controls pass.** $N=1$ has no interface, so the partition of unity is identically one and the composed step *is* the monolith: the composed defect is **exactly zero, bit for bit**, which is [[tier0-measurements]] §8's own lesson about the control nobody ran. $N=6$ is the wake array and reproduces its $25.27\%$ / $14.62\%$ array loss **to the digit**, on a driver that is a transcription of `w93_wake_array.march` rather than a re-derivation — a control against a re-implementation controls nothing. And the parent survives the refactor that made the ladder possible: `ArrayTiling` now carries its own layout, and **1314 of 1321 numeric values in `out/w93/w93.json` are bit-identical afterwards**, the other seven being wall-clock timings.

> **The control that turned out to be worth more than the headline, and it was not in the brief.** A bigger array is not only more interfaces — it is harder physics, with $u_{\min}$ falling from $0.509$ at $N=2$ to $0.183$ at $N=24$. So the same three numbers were measured again with **one turbine running at every rung**, where the flow near it is identical at every size and every added window sits in near-freestream. The interface-only exponent is **$+0.14$ to $+0.20$** against the headline's $+0.48$ to $+0.85$: **a $68\times$ increase in interfaces buys a $2.2\times$ increase in composed defect.** Most of the headline is the flow, not the cut — which makes a sub-linear result on a growing-physics ladder a *weaker* claim than it reads as (**W102**), and no other case study in this vault separates the two.

**And the run found what it was not looking for.** Continue the wake array's own march past the $60$ macro-steps it published: **the classical composed rollout goes unstable.** It leaves the band between step $70$ and $80$ at $N=6$ and is not finite by $82$, while the monolith from the same state under the same forcing holds $u_{\max}$ at $1.33$ throughout and the checkpoint's column is stable to $110$. The mechanism is one line — each window returns a field that is divergence-free *on its own window*, and a partition-of-unity blend of two divergence-free fields is not, because on the overlap $\nabla\cdot(\chi_1u_1 + \chi_2u_2) = \nabla\chi_1\cdot(u_1-u_2)$. **The assembly creates divergence exactly where the local solves disagree and nothing in the classical column removes it**: the assembled divergence grows $0.0132 \to 0.0957 \to 0.1507 \to 0.2502$ with the graph and the onset arrives earlier as it does, macro-step $14$, then $9$, then $8$.

**The positive control closes the mechanism.** One global Leray projection on the *assembled* field each macro-step — R10b's own prescription, and exactly what the checkpoint's column already does — makes **every rung stable** and holds the divergence flat or falling, with nothing else changed. So **`L2/R10` is vindicated by measurement for the first time**: the rule has refused the reference column of every graph in this vault since Tier 0, and nothing had ever shown the refusal was about anything real. It is. Two consequences stated plainly: no shipped case study applies that projection to a classical column (**W100**), and **Tier 18's published state sits on a diverging trajectory** — its numbers stand, because the state exists and the instruments read what they read, but the trajectory does not continue and $60$ steps was ten to twenty short of showing it.

**Three instrument defects closed along the way, and two of them were larger than their rows.**

**W58** — which `cut_defect_bound` was measured. Built: `MeasuredConstants.cut_defect_bound_form`, four named forms, `assembly.contaminated_multiplicity` deciding the disjointness condition **from the declaration with no run**, and `L2/C2/W58` decertifying a value with no form, an unknown form, a substructuring form on the wrong branch, or a reference-free form whose hypothesis fails. Then the measurement found **a second hypothesis nobody had stated**: `probe.neighbour_disagreement` is a max over neighbour *pairs* of a norm and L2/C2's bound is a norm over the whole grid of a cellwise max, and those diverge as the tiling grows whatever the contaminated sets do. Across the ladder the correctly-aggregated surrogate holds at $0.765 \to 0.802$ of the bound while the pairwise one falls to **$0.234$** — so the hypothesis the framework named moves the answer by $4\%$ and the one it never named moves it by a factor of three. The row has teeth on this run's own compiles: the checkpoint can only supply a reference-free number, and `L2/C2` **admits** it at $N=2$ where the geometry is clean and **decertifies** it at $N=6$, $12$ and $24$ — the same declaration losing its certification at a graph size, with nothing about the checkpoint changing.

**W54** — an emitted defect carries its harness and not only its depth. `emit.HarnessParameters` holds the overlap, $\chi$'s shape, the exchange interval and cadence and the elliptic placement, **derived from the graph** so it cannot disagree with what ran; `RunArtifact.validate` refuses an artifact whose $\tau$, $\sigma$ or $\gamma$ is measured and whose harness is absent; and the row's own definition of done — *a test that changing a harness parameter changes the attribution* — is a test in which one fixed set of restriction defects attributed under two ramps gives two constants **while the depth tag is identical on both**. The tier then supplied a fifth instance of the pattern unprompted: a construction that froze the disk force diverged, and that instability is the disk model's and would have been charged to the composition.

**W81** — $\beta_{\min}$, undefined framework-wide. Both branches the row allows are built. *Derived*: $\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ over the terms carrying a **positive** scale, the positivity filter being W84's caveat, since $\tau = 0$ is a real measurement and a plain minimum sets the tolerance to zero — which is the $10^{-12}$ default arriving from the other direction. *Reported*: `blind` and `passes` are both monotone in $\beta_{\min}$, so the whole verdict function is two numbers, and the certificate now returns `visible_above` and `fails_above` **instead of a verdict** when no tolerance is supplied. And the misleading sibling is gone — `schur_complement`'s parameter is renamed `beta_int_floor`, because it guards a singularity on the internal block where $10^{-12}$ is exactly right. Measured on one wake seam per rung, both thresholds are **negative** at all four and the derived tolerance agrees with the whole swept range: the swap is refused at every admissible $\beta_{\min}$ and is never blind, which set against **W97**'s rotor seam — blind on the entire axis — is the two seam types separated by two numbers rather than by a verdict.

**Three more results worth the line.** L2/C2's bound is tight to $4\%$ and **flat over the whole $24\times$ range** ($1.057, 1.040, 1.040, 1.040$), so the transferable property is the flatness rather than §10's $0.2\%$. W49's $\Pi$ **saturates at $0.7250$ from $N=6$ on, to every digit**, making it a property of the tiling's local pattern rather than of its size — so $C_\mu = 1.2$ carries along the ladder with no re-measurement. And the compiler's own reporting is **exactly** $\#\text{seams} + \#\text{agents} + 9$ decertifications with no cross terms, which is a small result and worth having: an audit trail that grew quadratically would be unreadable at rung 9 whatever the physics did.

**F3 is measured for the first time, and the two columns answer oppositely.** Claim B’s other half is $O(1)$ integration work per added agent. `scripts/w100_timing.py` interleaves the rungs, takes the minimum over repeats and **reports its own reproducibility** — and it had to fail first to be worth believing: a back-to-back version returned $27.0$ and $103.1$ ms per agent for the same rung minutes apart on a shared desktop, and the interleaved one at two repeats reported a $34\%$ spread and refused to quote a ratio. At six repeats the worst spread is **$6.0\%$** and the two ratios are $1.34$ and $1.14$ in log units against a log-noise of $0.058$. Per agent at $N=24$ against $N=2$: **Poseidon-T $0.32\times$** — it gets *cheaper* per agent as the graph grows, because one batched forward pass amortizes its overhead, which is more than Claim B asks — and **WindowNS $3.84\times$**. The confound is eliminated rather than argued: the sub-step count `WindowNS` picks from the advective CFL is **$21$ at every rung from $N=2$ up at an identical $u_{\max} = 1.302$**, so the per-sub-step ratio is the same and the arithmetic per agent is constant by construction. **The classical column’s rise is therefore the memory hierarchy and not the composition** — its *work* per added agent is $O(1)$ and its *time* is not, with the jump between $84$k and $163$k cells, and neither statement transfers off this host (**W104**).

**The mechanism tally.** Tier 18's lesson was *what made all four findings visible is the same thing — real geometry*. This tier's is different and three findings share it: **the measurement that decides is the one that removes a variable, and in each case the removed variable was one nobody had noticed was moving.** Remove the physics and the exponent falls fivefold; remove the elliptic placement and an instability disappears; remove the aggregation and W58's stated hypothesis turns out to be second-order. In all three the headline measurement was correct and *uninterpretable on its own*. A ladder measures a derivative, and a derivative taken while two things move measures neither.

New: `atlas/cases/scaling_ladder.py`, `scripts/w100_scaling_ladder.py`, `scripts/w100_timing.py`, `scripts/w100_frames.py`, `scripts/w100_build_viewer.py`, `scripts/w100_refit.py`, `tests/test_tier19_scaling_ladder.py`, `wiki/concepts/Atlas 0.1/common/case-study-scaling-ladder-atlas-0.1.md`. Changed: `atlas/graph.py` (`cut_defect_bound_form`, `CUT_DEFECT_FORMS`), `atlas/assembly.py` (`contaminated_multiplicity`, `ramp_cells`, `profile`), `atlas/probe.py` (`aggregated_neighbour_disagreement`, `restriction_defect_bound`'s `n_global` guard), `atlas/emit.py` (`HarnessParameters`), `atlas/composition.py` (`beta_min_from_tolerance`, the two thresholds, `beta_int_floor`), `atlas/compiler.py` (L2/C2/W58, the harness on the artifact), `atlas/cases/wake_array.py` (`ArrayTiling` carries its layout), `atlas/cases/window_ns.py` (declares its form), `wiki/index.md`, [[tier0-measurements]] §19, [[gap-worklist]] Tier 19. Artifacts `out/w100/w100.json`, `out/w100/state_<rung>.npz`, `out/w100/solo_<rung>.npz`, `out/w100/scaling-ladder.html`.

## [2026-08-31] note | W100 closed — the assembly gets a second condition, and the repair the tier proposed is falsified by its own instrument

**Query:** make the global Leray projection that CS-7 identified as the repair for the unstable classical rollout into a first-class, declared composition-layer step, reproduce the instability past $90$ macro-steps, and show every rung stable to $\ge 110$.

**The declaration was built and the repair was wrong.** Both halves are the result.

**Reproduced first, exactly.** Continuing [[case-study-scaling-ladder-atlas-0.1]]'s own march past the $60$ steps Tier 18 published: the classical composed column at $N=6$ is **not finite at macro-step $82$**, with $\lVert\nabla\cdot u\rVert_{\text{rms}}$ climbing $0.026 \to 2.57$ between steps $51$ and $81$, while $N=2$ holds flat at $0.0132$ through $95$. From the developed state it leaves the band at step **$14$**, matching [[tier0-measurements]] §19.6 to the digit.

**Then the positive control failed the same test.** §19.6 called one global Leray projection on the assembled field the repair, on $20$ macro-steps from each rung's developed state. Marched to $120$ at $N=6$ from the freestream, reporting where $\lvert u\rvert$ leaves the band $3$: **$74$** with no projection, **$51$** with a global spectral one, **$33$** with the classical solver's own Neumann one. From the developed state the control is stable over the $20$ steps it was measured over and dies at **$38$**. *Tier 18 published a state on a trajectory nobody had marched far enough; §19.6 published a repair on a trajectory nobody had marched far enough.* The horizon that exposed the first was $80$ steps; the horizon that exposes the second is $40$.

**Adding a projection to agents that already project applies the pressure twice.** Each `WindowNS` window has already answered the disk's momentum sink on its own subdomain, inside its own sub-steps; a global solve after the assembly answers it again. **And the divergence is not the proximate cause**: the Neumann variant holds $\lVert\nabla\cdot u\rVert$ at $0.0089$ against the bare column's $0.69$ — $78\times$ cleaner — and dies soonest of the three.

**What survives is `L2/R10`'s own prescription, which no classical column here had ever been built to satisfy.** Take the elliptic part *out* of the agent and let the composition layer apply it once, to the assembled field. `wake_array.exposed_reference_solver` is that agent — `window_ns._no_projection_class` with `_project` replaced by the identity and the velocity update untouched, the composition layer declining to use *part of* an expert rather than editing one. **It is the first classical column in this vault that `R10` does not refuse**, and it is the one that holds $120$ macro-steps.

**The declaration, which stands regardless.** `assembly.L6_C2` is the conservative-assembly condition, stated at $N$ windows rather than two: for a linear $\mathcal C$ with $\mathcal C u_i = 0$ on every subdomain, $\mathcal C(\sum_i \chi_i u_i) = \sum_i [\mathcal C, \chi_i] u_i = \sum_i \nabla\chi_i\cdot(u_i - w)$ for **any** $w$, because $\sum_i \nabla\chi_i = \nabla(1) = 0$ — so the residual is a functional of the *disagreement*, manufactured inside the overlap rather than transported into it. `assembly.ProjectedAssembly` is a partition of unity **and** an `assembly.ConstraintProjection`, declared together so a case study cannot apply the projection without the record saying so or declare it without the driver doing it. **`R12`** decides it from the declaration with no run, in six branches, and `emit.HarnessParameters.assembly_projection` puts it on every emitted defect — **W54's fifth instance**, and the one whose consequence is a trajectory that stops existing rather than a factor in $\tau$. **L6/C1 and L6/C2 are independent**: every convex partition in this vault satisfies the first and violates the second.

**A new gap, and it is sharper than the one it replaces.** The control that says the projection is load-bearing — exposed agents with **no** global projection — runs all $120$ macro-steps without leaving the band, at $\lVert\nabla\cdot u\rVert = 1.25$. **Stable, and not incompressible.** `R10` moves the elliptic part out, `R10b` fixes the cadence and assumes it happens, `R12` is silent because L6/C2's hypothesis genuinely fails on an all-exposed graph — so a graph can clear every rule and never apply the operator. **W105.** It is the first place here where *a rollout that does not blow up* and *a rollout that solves the equations* come apart, and every stability instrument in this vault reads the first.

**Controls held.** $N=1$ composed-equals-monolith is bit-exact and untouched — the blend is unchanged and at one window $\nabla\chi \equiv 0$, so the projected assembly and the bare one are the *same operator*, which is why a case study run at a single size could not have found any of this. The $N=6$ array loss is unchanged. All $443$ prior tests pass, plus $30$ new ones.

Changed: `atlas/assembly.py` (`L6_C2`, `ConstraintProjection`, `ProjectedAssembly`, `AssemblyCondition`'s C2 half, `AssemblyCertificate.conservative`), `atlas/compiler.py` (`R12`, `CONSTRAINED_FAMILIES`), `atlas/graph.py` (`CaseGraph.assembly_projection`), `atlas/holes.py` (the slot's fifth field), `atlas/emit.py` (`HarnessParameters.assembly_projection`), `atlas/cases/wake_array.py` (`divergence_rms`, `project_assembled`, `leray_projection`, `projected_assembly`, `assemble_conservative`, `exposed_reference_solver`, the `reference_exposed` kind), `atlas/cases/scaling_ladder.py`, `scripts/w100_scaling_ladder.py` (`stage_r12`, `stage_long_march`, `--only`, `--long-steps`), `tests/test_tier19_scaling_ladder.py`, `atlas/README.md`, [[tier0-measurements]] §19.6, [[gap-worklist]] Tier 19, [[case-study-scaling-ladder-atlas-0.1]] §5. Artifact `out/w100/w100.json` keys `r12` and `long_march`.

---

## [2026-08-31] note | CS-8 `reuse_probe.py` — is a substitution certificate a property of the expert, or of the state it was probed at?

The **eighth** real case study, and the first whose variable is the state a measurement is taken at rather than anything about the graph. Full record [[tier0-measurements]] §20; rows [[gap-worklist]] Tier 20; readable summary [[case-study-reuse-probe-atlas-0.1]]. Reproduce with `python scripts/w106_reuse_probe.py --steps 110`; artifact `out/w106/w106.json`, states `out/w106/state_N6.npz`, `out/w106/state_N12.npz`.

[[case-study-ladder-to-f1]] §4 names CS-8 as the only rung testing the *foundation-model* claim rather than the coupling claim. A twenty-agent car carries ~60 certificates; if each is state-specific, **the plug-in claim is per-design rather than per-expert** and the economic argument for a reusable expert library collapses into re-certifying every design. Nine tiers have quoted $\beta$, $\Xi$, $\tau$ and a substitution verdict, **each measured once**, and no page had asked what a second reading would say.

**The answer: neither noun is right, and the state half dominates.** One swap — `reference.WindowNS` $\to$ Poseidon-T — probed at $55$ cells across four factors. Median movement as a fraction of each quantity's own level: $\lVert S_i\rVert$ $2.49\%$, $\lVert\Lambda^{\text{expert}}-\Lambda^{\text{ref}}\rVert$ $3.63\%$, $\beta$ $6.46\%$ with the probe state — but $\Xi$ **$55.9\%$** and **both $\beta_{\min}$ thresholds $155\%$, more than their own size**. Meanwhile *doubling the graph* moves everything under $4\%$: a certificate is twenty to forty times more portable across designs of different **size** than across **operating points** of one design, which is the wrong way round for the library argument.

**The mechanism is arithmetic and therefore general.** Each threshold is $\beta$ minus a norm; here $\beta \approx 0.21$ and both norms are $\approx 0.22$, so each threshold is $\sim 10^{-3}$ — two orders below the terms it is built from — and a $3\%$ movement in one against a $6\%$ in the other lands as $155\%$ in their difference. **Any certificate decided near $\lVert\Delta\rVert \approx \beta$ inherits that amplification, and that is exactly the regime a *good* substitution is in.** **W107.**

**Both controls pass exactly, and the second one costs the framework a test.** Control ZERO: at the freestream, seams the taxonomy calls different regimes return the **same operator** — worst range $\mathbf{0.0}$ over eight quantities in all three port groups, by `==` rather than to a tolerance, which is what makes the seam factor interpretable at all. Control FLOOR: the identical cell re-probed with the state reloaded off disk and the experts rebuilt disagrees by **exactly $\mathbf{0.0}$** on all eight quantities. The pipeline is bit-reproducible — so `travel_verdict`, which asks whether a movement exceeds the floor, answers `state-dependent` for a $0.037\%$ movement and a $155\%$ one alike. **W106.** What replaces it: the quantity's own level, and a same-regime replicate.

**The verdict travels perfectly and it is the least informative true thing here.** Swept over the whole admissible $\beta_{\min}$ axis at all $55$ cells: `refuse`, everywhere. It is `refuse` because $\lVert\Delta\rVert/\beta > 1$ — but that ratio ranges $[1.00254, 1.38952]$ and its **minimum is a quarter of one percent from flipping**, at the seam with the *smallest* state-dependence in the study. **W109.** The same shape in W81's derived tolerance: $\varepsilon_{\text{tol}} = \min(\tau,\sigma)$ moves $35\times$ down one trajectory, $\tau$ moves $465\times$, and at the freestream the tolerance is **undefined** — `seam_defect_split` correctly refuses a relative defect where no power crosses the reference interface. W81 closed *where does the number come from*; this opens *which state is it the number for*. **W108.** So §19.8's *"$\tau$ is not monotone in $N$"* was the state, sampled once per rung.

**The practical result, and it is the one to carry forward.** The seam regime is a **rule over the layout** — count the rotors of a seam's own window row lying upstream of it — decidable before any march runs. It captures **$5$ to $53$ times** the seam-to-seam variation on every quantity except $\Xi$. So a library carries neither one certificate per expert nor one per design but **one per expert per regime**, a cost growing with the vocabulary of flow situations rather than with the designs searched. The exception is on the record: $\Xi$, the axis [[expert-library-atlas-0.1]] proposes to *rank* by, disagrees by up to $49\%$ between two seams the rule calls the same regime, and **its reproducibility had never been quoted**.

**Trajectory discipline.** Every probe state came off the R10-compliant classical column — `wake_array.exposed_reference_solver` with one `assembly.ProjectedAssembly` after the blend — because §19.6's embedded column is not finite past macro-step $82$. **Tier 18's published state was excluded for that reason**, so every state here had to be re-marched; both marches reproduce §19.6's column to three digits ($N{=}6$: $\lVert\nabla\cdot u\rVert = 0.06560$ against $0.06556$; $N{=}12$: $0.04043$ against $0.04011$). `reuse_probe.assert_projected` ran on **every one of the 130 graphs built** (**W105**), and all four compiles return `admit-uncertified` with **zero refusals in both columns at both sizes** — the first real graph in this vault to manage it, and the direct consequence of moving the elliptic part out of the agent.

**W110**, small and self-inflicted: the divergence guard's ratio form is inert because its reference is the first snapshot, which is the freestream, where the divergence is identically zero. Absolute values were checked and are healthy.

Added: `atlas/cases/reuse_probe.py` (the seam taxonomy, the probe-state schedule and health check, `assert_projected`, `CertificateReading`, `spread`, `travel_verdict`, `verdict_at`), `scripts/w106_reuse_probe.py`, `tests/test_tier20_reuse_probe.py`, [[case-study-reuse-probe-atlas-0.1]]. Changed: [[tier0-measurements]] §20, [[gap-worklist]] Tier 20, [[case-study-ladder-to-f1]] §7's rung-4 row and §8, `wiki/index.md`.
---

## [2026-09-01] note | PoC 1a — the adjoint, spent: gradient-based wind-farm design through the composed graph

The framework has claimed end-to-end differentiability as a headline feature since Tier 0 and used it for nothing. This spends it. Full record [[poc1-results-differentiable-design]]; specification and summary [[atlas-proof-of-concept-1]] §9; artefact `out/w111/w111.json`; reproduce with `python scripts/w111_wind_farm_design.py --stage grad --case K12 --opt-steps 30`.

**Read the asterisk first, because it is large.** Every rollout used `wake_array.exposed_reference_solver` — the *classical* solver — as the fluid agent, per the build brief and because it is the only classical arrangement that survives a long march (W100). So the **partitioning** half of [[prior-art-and-novelty-atlas-0.1]]'s novelty claim and the **field + lumped peers** half (§2.3) are exercised; the **frozen-pretrained** half (§2.1) is not. Poseidon-T is differentiable and the swap is a `kind=` argument, but it has not been run.

**The number.** Design vector $(x_k,y_k,\gamma_k)$ at $K=12$ (36 variables, $4\times3$ windows) and $K=25$ (75 variables, $6\times4$ windows); objective the disks' own `ROT`-port power time-averaged over the last five macro-steps, minus a spacing penalty; both methods from the identical start with the identical box-and-spacing projection. **The gradient reaches its own 2 % tolerance in 26 and 24 composed rollouts. CMA-ES spent 700 and 928 evaluations and never reached it** — ratios $>26.9\times$ and $>38.7\times$. At matched intermediate tolerance the measured ratio grows with the design dimension, $17.9\times \to 48.3\times$ from 36 to 75 variables, which is the trend that would carry §4's "two to three orders of magnitude"; **at these sizes it reaches one to nearly two, not three, and that is stated rather than rounded up**. A coordinate finite difference is $2n+1$ by construction: $73\times$ and $151\times$ per gradient step. **One adjoint costs 6–9 forward evaluations**, checkpointed one macro-step at a time because the un-checkpointed tape is $\sim\!10^2$ GB.

**The gradient is right, and the check earned its keep.** Adjoint against central differences at nine components: median relative error $\mathbf{5.9\times10^{-9}}$ at $h=10^{-5}$, cosine $1.0000000000$, with a textbook truncation/cancellation curve either side. The one component reading a relative error of exactly $1.0$ is turbine 6's yaw, where adjoint ($-1.0\times10^{-16}$) and difference ($-1.1\times10^{-10}$) are both numerically zero because it sits in the layout's **symmetric centre row** — the instrument is dividing two zeros. The check had already found a real defect: the disk's smearing thickness $\Delta_d=\langle U_d\rangle\Delta t$ was `detach()`-ed as "a statement about the discretization", and the differences reported a systematic $0.2\%$ deficit in *every* component, which is exactly the size of the path that had been cut.

**The ablation is the result worth carrying.** The same objective differentiated at the same point with the rollout **frozen** — the gradient a static wake surrogate would give — returns a vector three to eleven times larger and nearly orthogonal to the truth: cosine $0.238$ at $K=12$ and $\mathbf{0.054}$ at $K=25$. Its streamwise component at $K=25$ is *anti*-correlated ($\cos=-0.115$) and its signs agree with the truth $40\%$ of the time, worse than chance; yaw transfers fine ($\cos=0.95$) because the $\cos^3\gamma$ loss is local. **So a static model gets yaw roughly right and sites turbines in the wrong direction**, which is the sharpest argument this vault has for putting the seam inside the differentiated path.

**Provenance, because the column moved to torch and then to CUDA.** The torch agent is `wake_array.exposed_reference_solver` **bitwise** on a real `step_batch` — $\max\lvert\Delta u\rvert = 0.0$ — on CPU *and* on CUDA; the torch blend-plus-global-Leray reproduces `assemble_conservative` through the declared `ProjectedAssembly` to $2.2\times10^{-16}$; $J(\theta)$ is bit-reproducible and equal between devices (exactly at K12, $3.6\times10^{-16}$ at K25). Getting there required replacing `Tensor.scatter_add`, whose CUDA `atomicAdd` makes the summation order of overlapping window contributions vary run to run: **a finite difference of a function that moves in its last digits measures the noise, and CMA-ES ranking a population by such a number is a different algorithm.** The assembly now accumulates in a fixed order, which is the operational content of OP-6's "pin the batch layout".

**The blocking dependency is closed.** N24 had never been marched past 20 macro-steps; it is confirmed **stable to 60** with all 25 disks live — $u_{\max}=1.5192$ against a band of $3.0$, assembled divergence flat-to-falling (developed-tail trend $0.963$).

**And §5's honest framing, once measured.** Quasi-steady and not converged: settling $1.9\times10^{-2}$ and $3.0\times10^{-2}$ at the two optima, with the gain moving $+236\%$ at 30 macro-steps to $+315$–$336\%$ across 40–70, so the spec's 40–60 band is load-bearing below 40. Neither optimiser converged inside its budget, which makes both ratios conservative. Both optima sit **on** the spacing bound ($2.000\,D$ exactly) and the domain box, so the answer is partly a statement about the box. And the headline $+315\%$ is large mostly because the starting grid is pathological — $19\%$ of the unwaked ideal at $2.83\,D$ in-line spacing — so **the ratio and the wall-clock are the claim, not the percentage**. One artefact defect on the record: `OptTrace.as_dict` serialised only the endpoints of the design trajectory, so the yaw panel shows the optimum rather than the convergence to it; the record was written after every step and every write kept no history, so the trajectory existed at no point in time. Fixed in the code, not in this run's data.

Added: `atlas/cases/wind_farm_design.py`, `scripts/w111_wind_farm_design.py`, `tests/test_tier21_wind_farm_design.py` (36 assertions), [[poc1-results-differentiable-design]], `out/w111/` (artefact, four figures, both fields). Changed: [[atlas-proof-of-concept-1]] §9, `atlas/README.md`'s case table, `wiki/index.md`.
---

## [2026-09-01] note | PoC 1a's interactive demo — the composed march, live and editable in a browser

`atlas/demo/` and `scripts/w112_farm_demo.py`, documented at `atlas/demo/README.md`, tested by `tests/test_tier22_demo.py` (25 assertions). Run it with `python scripts/w112_farm_demo.py` and open `http://127.0.0.1:8011/`. Section added to [[atlas-proof-of-concept-1]]; the measurement it demonstrates is [[poc1-results-differentiable-design]].

**The measurement that decided the architecture.** One composed macro-step, forward, on the development box: $212$ ms at $6$ windows ($4.7$ Hz), $259$ ms at $12$, $445$ ms at $24$. **The field is therefore genuinely real-time** — drag a turbine and the wake re-forms over the next few seconds of wall-clock, because those are the same seconds of simulated time. A gradient macro-step is $5$–$7\times$ a forward one, so the optimiser is **warm-started and short-horizon**: one persistent field, and each Adam iteration differentiates the next $H$ macro-steps of it. **The rollout the gradient is taken through IS the live march**, so the field keeps animating while the optimiser thinks rather than freezing for the length of a rollout. Measured: $5.7$ s an optimiser iteration at $6$ windows and $H = 5$.

**That is a different estimator from the paper's and the demo says so.** [[poc1-results-differentiable-design]]'s objective marches from the freestream on every evaluation, which makes $J$ a function of the layout alone and costs $21$ s a rollout — right for a measurement, wrong for a demonstration where nothing may take twenty seconds to show a first frame. The demo's number is warm-started, so it carries the history of every layout already dragged through, and short, so it sees less downstream interaction. **The verification panel is what converts it back into a defensible number**: on demand the exact layout on screen is marched twice from still air, once through the composed graph and once through `scaling_ladder.reference_monolith` — the same discretization with **no cut** — and the two powers are shown side by side with the difference. Measured at the default: composed $1.565$, classical $1.772$, $\mathbf{-11.7\%}$. Cached by layout hash, run on its own thread, never blocking the optimiser.

**And the panel reports something the demo could have quietly omitted.** The classical monolith is **not slower** than the composed graph at these sizes ($8.1$ s against $13.8$ s for $30$ steps at $6$ windows): splitting a domain into windows and re-assembling them costs more than solving it in one piece. The composed graph's advantage here is not forward speed, it is that it can be **differentiated end to end**, and the panel says exactly that rather than implying a speedup it does not have.

**The validity panel is a citation, not a mood.** Every row is a predicate something already declared, with its number, its limit and its provenance: `reference.WindowNS`'s own $h\lvert u\rvert/\nu \le 8$ (the case study sits at $7.97$, so the wind-speed slider has almost no headroom above $1.0$); W100's stability band $\lvert u\rvert \le 3$; the global Leray projection's declared hypothesis that *"the march holds the inlet and both laterals at $(U_\infty, 0)$"*, which makes **wind direction the knob that leaves the regime** — turn it past $2^\circ$ and the panel goes red. **One row is deliberately a diagnostic rather than a gate**: the same cell-Reynolds predicate evaluated on the *state* is $\approx 11$–$13$ in any developed wake, above $8$ at every inflow the demo allows, which the case study records rather than hides. Gating on it would have painted the panel red permanently for a condition the user cannot act on, and a warning that is always on is not a warning.

**Three defects found by building it, all of them in the demo layer.** (i) The default layout on the smallest domain **opened with a spacing violation** — the $2\times1$ rung's box is $4.0 \times 1.0\,D$ once the freestream band and outlet margin are removed, which holds three turbines at the $2\,D$ packing limit and the demo's minimum is four. The rung is gone and `capacity()` now bounds the turbine slider; zero violations across every preset and count. (ii) Staggering a layout by adding an offset to a finished grid and **clamping at the wall** put two turbines on one point ($1.12\,D$ measured against a $2\,D$ limit); the offset is now taken out of the row range before the rows are placed. (iii) A browser reconnect left the page holding the **previous session's** turbine list, and the next drag pushed those stale positions back — silently moving a turbine. Nothing is sent upstream now until the client has adopted the server's state.

**Nothing in the optimiser was modified.** The one class that extends it, `engine.DemoRollout`, overrides `band` and `project` so the freestream can be pointed and scaled, and at the default inflow it is asserted **bitwise identical** to `wind_farm_design.Rollout` on a real `macro_step`. That assertion is the licence for the subclass to exist: it is what says the animation is the column the PoC measured, and not something that resembles it.

Added: `atlas/demo/` (`engine.py`, `server.py`, `cli.py`, `static/index.html`, `README.md`), `scripts/w112_farm_demo.py`, `tests/test_tier22_demo.py`. Changed: [[atlas-proof-of-concept-1]] (interactive-demo section), `wiki/index.md`.
---

## [2026-09-01] note | PoC 1a's two claims, separated on screen — and a timing number retracted

`atlas/demo/` gains a second window, a play/pause replay, and a correction. Tests `tests/test_tier22_demo.py` (25 -> 29 assertions). Packaged for other machines on the `poc1-windfarm-demo` branch, documented at `atlas/demo/PACKAGING.md`. Section 10 of [[atlas-proof-of-concept-1]] rewritten.

**The retraction first, because it invalidates a number this log already carries.** The earlier entry today, and the first version of §10, said one composed macro-step is $212$ ms at $6$ windows ($4.7$ Hz), $259$ at $12$, $445$ at $24$. **Those figures do not reproduce.** The same code on the same box, unchanged, measured **$\approx 890$ ms** at $6$ windows; a repeat forty minutes later gave $435$ ms; a third gave $943$ ms. The box is a Core Ultra 7 155H and `CurrentClockSpeed` was $1.4$ GHz against a `MaxClockSpeed` of $3.8$. **A wall-clock constant is a claim about a machine in a power state**, and this one was quoted as a property of the code. Nothing had regressed and nothing needed fixing in the physics — what needed fixing was the practice of writing the number down.

**So the demo measures itself.** `Engine.eta_s` returns an exponential moving average of what iterations have actually cost on the machine in front of the user; `_publish` ships it with an `eta_measured` flag and the screen marks it *(estimate)* only until the first real iteration lands. The per-domain table that used to answer this is demoted to a cold-start guess with a docstring explaining why it must not be trusted. The head-to-head times **both** solvers per macro-step, holds the first step out of the median (it pays for FFT plans and allocation: $779$ ms against a $943$ ms median on one side, $419$ against $611$ on the other), and reports the warm-up separately rather than dropping it. What survives as a stable claim is a **ratio**, not a duration: a gradient macro-step costs $\approx 5.5$ forward ones, and both scale with the window count.

**Two claims, and they were tangled.** The demo argues (i) that a layout can be **optimised** by differentiating the whole coupled simulation, which the classical solver cannot do at all, and (ii) that a **single layout** can be solved by both and compared. These are different claims with different evidence and they were sharing one screen, with (ii) as a panel under the fold. (ii) is now its own window at `/compare`, and both are labelled on the main page. The classical column is where the honesty lives: it returns a number and no derivative, so searching with it means one run per perturbed layout.

**The head-to-head, and why it alternates.** One layout, marched from still air by the composed column and by `scaling_ladder.reference_monolith` — the same discretization with **no cut** — **one macro-step each, alternately**. Running them concurrently would make a better animation and a worthless measurement: each per-step number would be a function of what the other was doing. Alternating gives every timed region the machine to itself, and because both sides sit at step $i$ at the same instant their fields are directly comparable, which is where the worst-single-cell velocity difference comes from. For the same reason **the live march parks itself** for the duration — `Engine._parked` is an `Event` the timing thread waits on, because *asking* the worker to stop and assuming it did would time the first steps against an optimiser iteration still in flight. Measured, $6$ windows, $6$ turbines, $30$ steps, on an optimised layout: coupled $943$ ms/step and $2.73$ power, classical $611$ ms/step and $3.27$, difference $\mathbf{-16.5\%}$, worst cell $0.475$ of freestream. **The classical side is the cheaper one, by $1.54\times$** — which the panel states in those words.

**Replay plays and pauses.** Only design vectors are recorded, not fields — a field per iteration is $\approx 1.3$ MB a step — so the replay **re-marches**: the turbines jump to step $i$'s layout and the wake re-forms around them. `REPLAY_HOLD = 2` macro-steps per cursor advance, because at one the layout runs ahead of the field that is supposed to explain it. Pausing does not stop the fluid, deliberately: pausing on a layout lets its wakes settle, which is the only way to see what that layout was doing rather than what it looked like in passing.

**Three defects found by building it.** (i) A finished comparison went on **counting**: `elapsed_s` was recomputed from `started` on every poll, so a $46$ s run reported "$118$ s total" purely because the page had been open that long. (ii) The second window seeded its step count from the **dataclass defaults** rather than the running config, so a server launched with `--steps 14` quietly ran $30$ and labelled the result $30$. (iii) The cost sentence named the **wrong solver** — it printed the cheaper one and then said it cost $1.54\times$ the cheaper one. All three were found by reading the rendered page against the API, not by the tests, which is the argument for looking at the thing.

**A fourth was not in the demo at all.** A demo server left running from an earlier session had been consuming a core since 08:07 and was the reason the first re-measurement read $915$ ms. The check that was supposed to catch it, `ps -W | grep -ci "anaconda3/python"`, misses the backslash form Windows reports for a detached process, so it had reported zero. **Every timing in this entry was taken after that process was killed and re-checked.**

**Packaged.** The `poc1-windfarm-demo` orphan branch carries the framework, the fluid expert vendored at `vendor/src/atlas/cases/windfarm/`, both test tiers, the drivers and four wiki pages, with `run.sh` / `run.cmd` that build a venv, install six packages, self-test and start the server. The vendored layout is chosen so that `ATLAS_BUILD_REPO=./vendor` resolves the expert **with no source file modified**, which is what makes the two test tiers on that branch the same files character for character as these ones — verified there: $29$ passed and $31$ passed with $5$ skipped, from a bare checkout with the environment unset.

Added: `atlas/demo/static/compare.html`, `atlas/demo/PACKAGING.md`. Changed: `atlas/demo/engine.py` (`Comparison`, replay, self-measured ETA, responsive stop), `atlas/demo/server.py` (`/compare` and the compare API), `atlas/demo/static/index.html`, `atlas/demo/cli.py` (`--device`, `--steps`, `--open`), `atlas/demo/README.md`, `tests/test_tier22_demo.py`, [[atlas-proof-of-concept-1]] §10, `wiki/index.md`.
---

## [2026-09-01] note | A second machine, the ladder's hybrid revision, and two new gap rows

[[poc1-retrospective-and-hybrid-roadmap]] (new page), section 9 of [[case-study-ladder-to-f1]], and Tier 21 of [[gap-worklist]] (**W111**, **W112**). Prompted by a cross-machine re-run of the PoC 1a demo — a Mac measured $10$–$25\%$ **slower** on the decomposed all-classical column against a classical monolith, where the dev box measured $-16.5\%$ under the same comparison. Same sign, two independent memory hierarchies.

**What the PoC actually demonstrated, read against that number.** The demo has no neural network anywhere in it — its "coupled graph" is windows of the *same* classical solver (`reference.WindowNS`), its "classical solver" is that solver undivided. So the comparison was never classical-vs-learned; it is decomposed-vs-monolithic with the expert held fixed, and the Mac's result is the **second machine to find the same sign** as [[tier0-measurements]] §19.10's finding that a classical column's per-agent cost *rises* with window count while a frozen checkpoint's *falls* — a finding that page explicitly declined to claim transfers off its own host (**W104**). It now has cross-machine evidence for its qualitative direction, if not its ratio. Read plainly: PoC 1a proved end-to-end differentiability through a decomposed classical graph and a working gradient optimizer (verified against finite differences); it did not prove a speed advantage, attempt multiphysics (every cell is the same governing equation — the "seams" are computational partitions, not physical boundaries, which is the user's own correct critique), or decide where any seam should go.

**Classical solvers as experts, answered with numbers already in the vault.** [[tier0-measurements]] §19.10's table shows a genuine crossover between $N=6$ and $N=12$ windows: classical wins by up to $4.9\times$ per agent at $N=1$ (Poseidon-T pays a large near-fixed call overhead a handful of windows cannot amortize), the frozen checkpoint wins by up to $3.96\times$ at $N=24$ (one batched forward pass spreads that overhead over more windows while the classical column's cost is rising from a memory-hierarchy effect). Neither number is asserted to transfer off the host it was measured on. PoC 1a's demo runs at $N=6$ — inside the region where classical wins in that table, which is a second reason the Mac's result is unsurprising once read beside it. Certifiability is asymmetric on top of speed: **W93** refuses nearly every learned donor's decomposition outright (global receptive field), while a classical stencil's halo is exact and known without a probe; [[expert-donor-survey]] finds almost no pretrained donor accepts a boundary condition as an input at all. Classical experts may be the permanently correct choice at many seams, not a stepping stone.

**A genuinely new, unmeasured axis: iterations to converge, not cost per iteration.** [[schwarz-iteration-atlas-0.1]] measured that a periodic window's Schwarz sweep is the identity map bitwise (zero effective iterations) and that a Dirichlet sweep costs $14\times$ more *per iteration* — but nothing has held a seam and a tolerance fixed and counted **sweep count**, classical vs. learned, at the identical seam. Opened as **W111**, on the hypothesis that a local classical operator's boundary response may converge in fewer sweeps than a globally-receptive one, independent of either one's per-sweep speed.

**The ladder reframed, not rebuilt.** [[case-study-ladder-to-f1]] §2 already runs a classical-first, substitute-per-seam strategy in parallel from Phase B onward — what it lacked was a stated destination. Corrected: not "as learned as certification allows" but a **standing per-seam library**, where the fastest *admissible* expert runs at each seam, which today often means classical, permanently, at multiphysics seams especially. No case study's number, scope, or gate changes.

**One insertion: CS-9★ `seam_placement.py`.** [[interface-transfer-theory]] §9's cut-quality score $\mathcal Q(\Gamma)$ has existed since 2026-08-27, flagged as its own page's weakest claim, and has never once been used to choose a decomposition — six case studies have all placed their windows by hand. CS-9★, scheduled between CS-9 and CS-10, needs no new experts or physics: search candidate tilings of CS-9's two-family domain, scored by $\mathcal Q$ under a cost budget, gated against the ladder's own hand-chosen rungs. It is the one item among [[f1-pathmap-and-end-goal]]'s four end-goal capabilities — multiple experts, multiphysics, multi-formulation decomposition, **automatic adaptive seam placement** — with no prior art in [[prior-art-and-novelty-atlas-0.1]]'s own verdict table. Opened as **W112**.

Added: `wiki/concepts/Atlas 0.1/common/poc1-retrospective-and-hybrid-roadmap.md`. Changed: [[case-study-ladder-to-f1]] (§9), [[gap-worklist]] (Tier 21, W111–W112), `wiki/index.md`.

---

## [2026-09-01] note | CS-9: thermal strain is a bond and is not a port

[[case-study-thermal-strain-atlas-0.1]] (new page), section 10 of [[port-algebra-atlas-0.1]] (the binding `PortAmendment` procedure), Tier 22 of [[gap-worklist]] (**W94** and **W32** closed, **W113**-**W116** opened), and section 10 of [[case-study-ladder-to-f1]]. Code: `atlas/cases/thermal_strain.py`, `scripts/w94_thermal_strain.py`, `tests/test_tier23_thermal_strain.py`; artifact `out/w94/w94.json`.

**The ninth real case study, and the first co-located split.** [[case-study-ladder-to-f1]] section 4 schedules CS-9 as *"the single highest-leverage missing piece of vocabulary"*: **W94** found the case with no escape from needing a real interface bond, and the port algebra has five **surface** bonds and nothing else. So `thermostruct2d.ThermoStruct2D` -- the same build-repo solver `thermal_seam` uses, imported unmodified -- was split into a **conduction agent** and an **elasticity agent** coupled through thermal strain, with the unsplit solve as a free referent. The two agents own *the same cells*: cut along the physics rather than the domain, so there is no cut, no overlap, no halo and no partition of unity, and `Decomposition` has no member for it.

**The gate is met.** Carried in full the volume term reproduces the monolith's stress **bit for bit** -- a control on the plumbing and not a measurement, per **W106** -- and lagged one macro-step it costs $7.116\times10^{-3}$ relative, first order in $\Delta t$ ($\log_2$ ratios $1.003, 1.002, 1.001$ over a $8\times$ range). The volumetric power closes the *elasticity agent's own* balance to the same order. The two-way half -- the Biot term `ThermoStruct2D` truncates, restored as a source without editing that solver -- moves $T$ by $1.104$ K and the stress by $3.78\times10^{-3}$ at a coupling number $9.075\times10^{-3}$, and a staggered one-pass split recovers $98.3$% of it.

**And the amendment refuses, which is the more useful half.** `PortAmendment` was exercised for the first time (**W32**, open since Tier 5) and **refused** on two of six fields. The conjugate pair does exist -- $(\Delta T,\ \beta\,\mathrm{tr}\,\dot{\boldsymbol\varepsilon})$, with the reverse half the *same operator transposed*, $G^{\!\top}$ -- so the object **is a bond**. What it is not is a **port**: the **pairing is not unique** (two readings of the same coupling differ by $3395\times$, and their sum is exactly the rate of an energy neither agent owns), and the **carrier has co-dimension zero** so $\dim M = \dim V$ and a probe is $344$ full solves for an operator that *is* the coupled solve. The mechanism is arithmetic: the free energy carries a bilinear cross term belonging to neither agent, measured at $2680\times$ the elastic energy -- and for a free body it is exactly twice the strain energy -- so there is no additive split, and $\mathcal R(t)$ sums per-agent energy rates and therefore presumes one.

> **A port is a bond on an interface of co-dimension at least one between agents whose free energies add. Thermal strain is a bond of co-dimension zero between agents whose free energies do not add.** The instrument is an **operator splitting with a splitting-error bound**, not a transmission condition -- and CS-9 supplies the first such number. The port list stays at five.

**The exercise changed the procedure.** `holes.py`'s own note predicted the six-field checklist might prove insufficient at its first use; it did. **Field 7, `support`** -- the co-dimension of the carrier and, at zero, the argument that the energies are additive -- is now [[port-algebra-atlas-0.1]] section 10.1, binding. Fields 1 to 6 are *all* satisfiable by a co-located pair and not one of them notices the residual has lost its premise.

**W70 is confirmed from the other side, and the reason its reframing held is now a number.** W70 said *"a sixth port type is not the fix, because the object is not a bond"*; measured, the object is a bond and is not a port. The classical equivalent-thermal-pressure reduction is *exact* when $\nabla\Delta T = 0$ -- the exact identity $G = G_{\text{surf}} + G_{\text{body}}$ holds to $10^{-15}$ and the body half is $6.8\times10^{-17}$ of the load at a uniform temperature -- and W70's shell was $8$ mm of aluminium at a Biot number of $3\times10^{-4}$, nearly isothermal.

**The sharpest finding is about the two routes the closed vocabulary already permits.** As a surface `MECH` traction the stress is wrong by $1279\times$: the dropped body force is only $9.6$% of the load, but the two halves nearly cancel on a soft strip so the displacement amplifies $233\times$. As a **`GlobalField`** it is numerically **exact** -- and bypasses L3 entirely, so no scale set, prolongation, adjoint, null space, response half, $\tau$, $\sigma$ or $\beta$ is ever asked for. Compiled side by side on identical physics, the exact-and-uncertified route turns E3 from `fails` to `holds` and E7 from `fails` to `unchecked`, and carries three fewer decertifications than the route that gets the wrong answer -- **four seam properties (`L1/E3`, `L4/E7/passivity`, `L4/block-share`, `L4/operator-content`) traded for one line saying it is not a composition**. Not uniformly better, and the page says so: $\beta$ is a property of a probed seam and becomes unmeasurable. **Declaring a coupling *out* of the port algebra improves its envelope stamp**, it is declarable today by anyone, and no rule catches it.

**And two more the compile produced for free.** The `surface-mech` seam's interface problem is **one-sided to machine zero** -- $\lVert\Lambda_{\text{cond}}\rVert$ is exactly $0$ against $5.619\times10^{11}$, because a temperature field does not know its boundary is moving -- which is `CASE-STUDY-GUIDE` mistake 6 measured, and a sharper instance than **W97**'s rotor seam, whose one-sidedness is a ratio rather than a zero. And $n_0(\Gamma)$ gains a **fourth row**: declared $0$, the seam refused at `L4/null-space`, and inspection found the null direction is the **constant mode** to $5.7\times10^{-13}$ at $\sigma_{\min}/\sigma_{\max}$ $= 2.4\times10^{-15}$, physically rigid-body translation normal to the face. A solid-solid `MECH` seam on a **free** body gets $1$ for neither incompressibility nor lumpedness -- kinematics -- and the guide's table now says so.

**Four things found that were not being looked for.** **W113**: a graph with no connections could not emit an artifact at all -- `_l3_connections` stamped E2 `unchecked` and `emit.validate` refuses that outright, in its own words *"a defect in the compiler, not a property of the case"* -- latent for the compiler's whole life because no graph in this vault had zero seams until now; fixed, E2 holds vacuously over an empty interface set. **W114**: `L2/R10` refuses a co-located split and its own derivation does not reach one, since such a split cuts no domain, and the rule never asks whether the decomposition cuts the agent. **W115**: `Decomposition` has no member for two agents sharing every cell. **W116**: a co-located split's own measured defect is **not declarable** -- `MeasuredConstants` has a slot for a *transmission* infidelity and none for a *splitting* one, so the case study declares `tau = 0` and leaves $\sigma$ unmeasured rather than putting one number in the other's slot, which would be W56 with the sign reversed. Also recorded: $\mathcal R(t)$, the monitor the port algebra asks for every macro-step, is **blind to the coupling under test by six orders of magnitude** here, and the right residual for a co-located split is the receiving subsystem's own balance.

Full suite **$591$ passed, $0$ failed** on the final tree ($23$ of them this tier's). It was **$1$ failed** at the start of the session, and that failure was not this work: a ragged table in [[poc1-retrospective-and-hybrid-roadmap]], caused by two aliased `[[page|alias]]` links written against the vault's own bare-link convention -- the alias pipe is counted as a column separator. Repaired, and `scripts/vault_scan.py` reports $201$ files and $0$ problems.

Added: `atlas/cases/thermal_strain.py`, `scripts/w94_thermal_strain.py`, `tests/test_tier23_thermal_strain.py`, `wiki/concepts/Atlas 0.1/common/case-study-thermal-strain-atlas-0.1.md`. Changed: `atlas/compiler.py` (W113), `atlas/CASE-STUDY-GUIDE.md` (the case list, and a box on co-located splits), `atlas/cases/__init__.py`, `tests/test_tier8_rules.py` (a docstring W113 falsified), [[port-algebra-atlas-0.1]] (section 10), [[gap-worklist]] (Tier 22), [[case-study-ladder-to-f1]] (section 10), [[poc1-retrospective-and-hybrid-roadmap]], `wiki/index.md`.
---

## [2026-09-02] note | W117: the global-field route, opened as a row, and the census that decides it

Tier 22 of [[gap-worklist]] gains **W117** and its census subsection. No code changed and no measurement was rerun; this is a **reading** of the tree CS-9 left, filed the day after it.

CS-9 recorded the `GlobalField` route as a standing warning — [[port-algebra-atlas-0.1]] §10.4, *"declarable today by anyone, nothing in a compile says so"* — and left it as prose rather than a row. Promoted, with two facts the prose did not have.

**First: it is not merely uncaught, it is unreadable.** `global_fields` is referenced nowhere in `compiler.py`, `emit.py`, `composition.py` or `scheme.py`. The declaration reaches `to_dict` and stops, so `GlobalField` is **write-only**. And the dataclass carries `name`, `applies_to` and `note` — none of which records **where the field's value comes from**, which is exactly the discriminator any rule would have to test. The warning is the only instrument available because the declaration does not hold the fact a check would read.

**Second, and the reason the row is a judgement rather than a patch: the trap is already sprung.** All seven live `GlobalField` declarations were read out of their `build` functions. One is `gravity` (external, and what §5.1 wrote the class for), one is a declared absence (`wind_farm`'s `"none"`), one is CS-9's own eigenstrain — and **four are `pressure`**, in `window_ns`, `channel_ns`, `wind_farm_real` and `wake_array`. Pressure is the quantity that mediates the coupling between the windows the graph decomposes.

The candidate discriminator is **provenance**, and on this census it separates cleanly: gravity is external; the four `pressure` fields are a genuine global operation over *every* agent, owned by the composition layer and declared as such; the eigenstrain is **one named agent's state handed into another agent's update**, which is a seam with the seam removed. A rule that fires when a global field is produced by a named *proper subset* of the graph's agents catches the last and clears the rest.

**W117a is the experiment, and it is designed to be able to fail.** Re-declare all seven with a provenance field and compare the firing pattern against each declaration's stated intent. Firing on exactly the eigenstrain closes the row. Firing on the four `pressure` declarations too means either the rule is wrong or those four are the same defect as CS-9's route — and this vault's four largest case studies are then resting on it. `wake_array`'s is the one to watch: its pressure field is repairing a checkpoint channel *"a PLACEHOLDER its loader pins to 0"*, which is closer to CS-9's route than to gravity.

Also corrected in the entry above: it said CS-9 opened **W113**-**W115**, and CS-9 opened **W113**-**W116** — its own body describes W116 and the Tier 22 table carries it.

Changed: [[gap-worklist]] (Tier 22 — W117 and the census).
---

## [2026-09-02] note | W117a ran, and the global-field route is refused rather than warned about

**W117 closed the same day it opened.** Code: `atlas/graph.py` (`GlobalField.produced_by`), `atlas/compiler.py` (`_l3_global_fields`), all seven live declarations, and `tests/test_tier24_global_field_provenance.py` (11 tests). Wiki: [[gap-worklist]] Tier 22 (the W117 row and the W117a result), [[port-algebra-atlas-0.1]] §10.4, [[case-study-thermal-strain-atlas-0.1]], `atlas/CASE-STUDY-GUIDE.md`.

**The gap was one step larger than the entry above recorded.** `global_fields` was referenced **nowhere** in `compiler.py`, `emit.py`, `composition.py` or `scheme.py`: the class had no reader at all, so §10.4's *"nothing in a compile says so"* was not an oversight in a rule but the absence of any rule to overlook it in. And `GlobalField` held `name`, `applies_to` and `note` — none of which says **where the value comes from**, which is the one fact a check would test. The warning was the only available instrument because the declaration did not carry the discriminator.

**The fix is one field and one rule.** `produced_by` distinguishes external (`()`), a global operation owned by the composition layer (every agent), and a proper subset applied outside itself (**refuse** — one agent's state entering another's update with no seam). Undeclared is `None` and is **decertified, not admitted**: a default of `()` would have read every legacy declaration as external and certified exactly the thing the rule exists to catch.

**W117a ran before the rule was adopted, and it fires on exactly one of seven.** CS-9's `global-field` route is refused; `rocket`'s gravity, `wind_farm`'s declared absence, and the split-step pressure projection in `window_ns`, `channel_ns`, `wind_farm_real` and `wake_array` all clear. The four `pressure` declarations were the ones that had to be *checked* rather than assumed, because pressure is what mediates coupling between the very windows those graphs decompose — and they clear for a reason read out of the experts, not tuned into the declaration: **a rotor expert has no pressure input and no pressure state**, so the disk's momentum sink runs through the declared `MECH` seams and the field applies to no agent that does not produce it. Both rotor graphs had left `applies_to` empty, which *means every agent* and so declared the opposite of what the code does; W117 forced them to become precise, which is the rule working before it refused anything.

**It also refused its author.** The first `channel_ns` declaration reused `tuple(tiling.names)` — right for `window_ns`, wrong here, since `channel_ns` borrows that `Tiling` (windows `W`-prefixed) while naming its own agents `C00`–`C11`. The rule caught it at *"names no agent in this graph"* instead of admitting a provenance list matching nothing.

**What it costs CS-9.** That route **compiled** when the case study was written — numerically exact, certified by nothing, three fewer decertifications than the `surface-mech` route that is wrong by $1279\times$. It is now refused, and the route is deliberately kept exactly as written, because a rule with nothing to fire on is not a rule. The case study's §4.4 comparison is unchanged in substance and sharper in shape: a route refused for being uncertifiable against a route admitted and wrong.


---

## [2026-09-02] note | PoC 1a on the frozen checkpoint — W118 closed, and the asterisk retired

**The half of the claim that had never been run, run.** Full record: [[poc1a-frozen-expert-results]]. Code: `atlas/cases/wind_farm_design.py` (`TapedPoseidon`, `PoseidonRollout`, `rollout_for`), `scripts/w111_wind_farm_design.py --expert poseidon` (with two new stages, `verify` and `crosseval`), `atlas/demo/` (`DemoPoseidonRollout` and a `--expert` flag), `tests/test_tier26_poseidon_design.py` — 17 assertions, passing on CPU and on CUDA. Artefact `out/w118/w118.json`. [[poc1-results-differentiable-design]] §0 said the frozen-pretrained half of [[prior-art-and-novelty-atlas-0.1]] §2.1 was not exercised, that the substitution was a `kind=` argument, and that nothing on the page should be read as if it had been run. It is now a `kind=` argument that has been run, at $K=12$ and $K=25$, and **all three of §2's halves are exercised by one column for the first time.**

**What had to be built, because the checkpoint arrives with its differentiability deliberately removed.** `adapters.FrozenFluidExpert` cuts the tape in three places — numpy in, `torch.no_grad()` around the forward, `.detach().cpu().numpy()` out — and its own docstring says why: *"so that a gradient cannot be taken by accident."* `TapedPoseidon` is the same arithmetic with the tape left on, written in the vault rather than in the build repo for the reason `_TapeBackend` and `_no_projection_class` are: **the agent is not ours to change, and what the composition layer may do is decline to use one part of it and supply that part itself.** It is asserted **bitwise** against `step_many(galilean=False, project=False)` on a real 12-window batch, and the whole composed macro-step reproduces `scripts/w93_wake_array.py`'s numpy march to $2.2\times10^{-16}$. 20 774 444 parameters, **zero** requiring a gradient, asserted after a backward pass rather than assumed.

**The checkpoint needs more from the composition layer than the solver does, and that is a real asymmetry rather than plumbing.** `WindowNS` advects, so the composition layer owes it only the global Leray projection. Poseidon-T, run the way W98 settled, advances the *fluctuation* and does not transport it — so the layer owes it the advection too, and `PoseidonRollout` overrides `project` with a transport-and-project rather than reusing it. Three steps of its macro-step are W-numbered findings and none is a choice: the mean restore is W0's E1, the two global operators are **W98**, and the body force lands after the blend because `step_many` drops `force` silently when `galilean` is off, which is **W99**.

**The result nobody had a prediction for: the projected assembly holds a globally receptive expert.** W93 measured this checkpoint's domain of dependence as the *whole window*, which is why `L2/R10` refuses the decomposition outright; a partition-of-unity blend over an 8-cell ramp has, for such an expert, no argument for being valid. Marched from the freestream with all disks live it is stable to **70/70 macro-steps at both rungs** — $u_{\max} = 1.4519$ and $1.4397$ against a band of $3.0$, assembled divergence flat at $8.5\times10^{-2}$ with tail trends $1.020$ and $1.008$. **R10 still refuses and the arithmetic still runs**, and those two facts are the same distinction [[composition-error-theory]] draws between a search instrument and a verifier, arriving as two numbers from one run. A gap row saying "R10 is too strict" would be exactly the wrong lesson and was deliberately not opened.

**The headline moved, and one direction of the move is a reversal.** $J$ goes $2.4782 \to 7.0243$ at K12 (+183.5 %) and $7.0997 \to 14.6951$ at K25 (+107.0 %), in 18 and 30 gradient rollouts to the same 2 %-of-own-gain tolerance the classical column used. **At $K=12$ CMA-ES reached that tolerance in 144 evaluations and finished at $8.1200$ — $15.6\,\%$ *above* the gradient method's own optimum** — where on the classical column it spent 700 and never arrived. The ratio falls from $>26.9\times$ to a measured $\mathbf{8.0\times}$. At $K=25$ it never arrived in 928, so the bound is $\mathbf{>30.9\times}$: **the dimensional trend survives and the level drops about threefold.** The most likely mechanism is OP-3 — a column whose wakes barely reach the next turbine has a flatter, more separable landscape — and it is opened as **W119** rather than asserted, because three separate readings on the results page depend on it.

**The deployment story is no longer a plan.** [[prior-art-and-novelty-atlas-0.1]] §5 commits to *search wide and cheap with the composed model, verify the shortlist with the classical stack*. Three layouts — the grid, the classical column's optimum, this column's optimum — each scored by both composed columns **and by `scaling_ladder.reference_monolith`**, the undivided classical solver. A layout found entirely inside the frozen column, by an optimiser that never called a classical solver, is worth $\mathbf{+266.5\,\%}$ (K12) and $\mathbf{+170.0\,\%}$ (K25) over the starting grid **when the monolith scores it** — against $+374.2\,\%$ and $+239.7\,\%$ for the classical column's layout under the same verifier. **The pre-screen works and captures $71.2\,\%$ and $70.9\,\%$ of the verified gain — the same fraction across a factor of two in design dimension**, which points at a systematic property of the expert rather than at search failure, and is opened as **W120** on the permissively-licensed donors.

**OP-6, corrected rather than confirmed.** The vault carries OP-6 as *"$\sim10^{-6}$ batch-position dependence, removable by pinning the batch layout"*, and [[probed-dtn-coupling]] §5's probe-step derivation rests on the removable reading. Measured here: pinning does work — $J$ is bit-reproducible on both devices — and the finite-difference floor is still $\mathbf{1.4\times10^{-3}}$ at $h=10^{-2}$, against the classical column's $5.9\times10^{-9}$ at $h=10^{-5}$, with a textbook truncation/cancellation curve either side and the cancellation branch arriving two and a half decades earlier. **A pinned batch layout buys reproducibility, not precision**, and the second is what a finite difference needs. Cross-machine, $J$ agrees to $2.8\times10^{-6}$ where the classical column agreed exactly. Opened as **W121**. Also new and not optional: **TF32 is disabled in code and asserted in the tests** — on an Ampere card it would put every one of the checkpoint's matmuls at $\sim10^{-3}$ relative, which is `scatter_add`'s failure one order of magnitude worse.

**The ablation's sharpest number gets sharper.** Differentiating the same objective with the rollout frozen — the gradient a static wake surrogate gives — returns, at K25, a streamwise component with $\cos = \mathbf{-0.458}$ against the truth and signs agreeing $\mathbf{12\,\%}$ of the time. The classical column's was $-0.115$ and $40\,\%$. The aggregate cosine looks *better* here ($0.41$ against $0.054$) and should not be quoted: it is dominated by the lateral component, whose norm is four to six times the streamwise one.

**And the composed graph is finally the fast one.** [[atlas-proof-of-concept-1]] §10 measured the *classical* composed column at $1.54\times$ **slower** than the undivided monolith, and [[poc1-retrospective-and-hybrid-roadmap]] §1 had a second machine agreeing on the sign. On the same desktop, with the checkpoint in the windows, the composed column is $\mathbf{3.15\times}$ (K12) and $\mathbf{3.64\times}$ (K25) **faster** than the monolith. That finding about the cut is unchanged; what changed is what is inside each window, and it is [[tier0-measurements]] §19.10's $N=6$-to-$N=12$ crossover measured at the level of a whole composed rollout rather than a per-agent microbenchmark. A gradient costs **3 to 4.5 forward evaluations** here against 6 to 9 on the classical column, and an A100 is $12.8$–$19.5\times$ the desktop rather than $5$–$6\times$, because a float32 transformer is what a tensor-core GPU is for and a float64 stencil is not. The whole experiment — both optimiser runs, both baselines, two 70-step marches, two finite-difference sweeps, two ablations, four verification panels, two cross-evaluations and a sensitivity sweep — cost **about \$1.15 and 65 minutes on one A100**.

**Not claimed, and the list is longer than the results.** Not accuracy — §7.2 prices the shortfall at 29 % of the verified gain. Not a certified composition — R10 refuses, unchanged. Not two to three orders of magnitude. Not multiphysics, and the seams still separate identical physics, so [[poc1-retrospective-and-hybrid-roadmap]] §1's critique applies word for word. Not a converged optimum at either size, and at K12 the derivative-free baseline found a better one. And **not shippable**: Poseidon-T's weights are CC-BY-NC-4.0, so the frozen-expert claim as demonstrated cannot be carried into anything commercial with this checkpoint in it.

**The demo carries the toggle.** `python scripts/w112_farm_demo.py --expert poseidon`; `DemoPoseidonRollout` is asserted bitwise identical to `PoseidonRollout` at the default inflow, the same licence `DemoRollout` has against the classical one, and the panel **names the expert on screen** rather than leaving a viewer to infer which column produced the power number. A machine without the checkpoint is offered the classical column and told so, rather than being shown a solver under a frozen-expert label.
---

## [2026-09-02] note | CS-9★ — a search chooses the cut, wins at one interval, and loses when marched

**The tenth real case study, and the first exercise of the one end-goal capability with no prior art.** Code: `atlas/cases/seam_placement.py`, `scripts/w112_seam_placement.py`, `scripts/w112_horizon.py`, `scripts/w112_summary.py`, `tests/test_tier25_seam_placement.py` (42 tests). Wiki: [[tier0-measurements]] §21, [[case-study-seam-placement-atlas-0.1]] (new), [[gap-worklist]] Tier 23. Artifacts `out/w112/w112.json` and `out/w112/horizon.json`.

**The row's premise was wrong when it was written, and the record that corrects it is in this vault.** W112 and [[poc1-retrospective-and-hybrid-roadmap]] §4 both say $\mathcal Q(\Gamma)$ *"has never been used to choose a cut"* and has never had a falsification. [[tier0-measurements]] §10.2 — 2026-08-28, four days after $\mathcal Q$ was proposed — measured a rank correlation of $-0.853$ over twelve cut placements, a $361\times$ orbit under an admissible re-declaration of the same interface space, and a criterion constant to $8\times10^{-11}$ where the truth spread $2.29\times$. `probe.cut_score` has read `RETIRED` in its own docstring since that day. What had **never** happened, and is what W112's own *done when* clause asks for, is a search over **decompositions** under a **cost budget** on **real geometry** against a **hand-chosen rung**. That is what ran.

**And the first run's budget was wrong, which is the cheapest finding here.** Under `n_windows <= 6` every one of eight criteria returned the same two-window decomposition at $0.085\times$ the hand-chosen defect, because fewer seams is less defect, always. An upper bound on agent count is a question about *how many*, not *where*. The gate's budget became an **equality** — six windows, halo eight, only the cuts moving — and the degenerate pool is kept and reported, because the shape of a budget is part of the criterion.

**The gate passes at one exchange interval and it is not close.** The search finds a six-window decomposition at $\mathbf{0.2685\times}$ CS-7's hand-chosen tiling on the $N=6$ geometry, with the hand-chosen tiling ranking **138th of 178**; $0.305\times$ at $N=12$ (48th of 51); $0.295\times$ on the one-rotor control. Three criteria find the exact optimum — the reference-free surrogate, L2/C2's $\chi$-weighted bound, and a new one that costs two array gradients.

**The mechanism needed no run at all.** `wake_array.Rotor` is declared on an x-*adjacency*, so a disk sits at the midpoint of an overlap by construction: the hand-chosen x-cuts are at cells $120$ and $232$ and the rotor planes are at cells $120$ and $232$. **Every x-seam of CS-6 and CS-7 passes exactly through a rotor disc**, worth $3.7\times$ in one-interval composed defect, and the tiling rule cannot express anything else — which is why five case studies and a scaling ladder never surfaced it. That is **W124**, it is pinned by a test, and every number those case studies published stands: the seam was where it was and the instruments read what they read.

**Marched, the answer reverses, and that is the result.** Over $240$ exchange intervals the chosen cut crosses $1.0$ at interval $10$, peaks at $\mathbf{1.59\times}$ **worse** than the hand-chosen tiling at interval $90$, and returns to $0.90$ at $240$ — two sign changes, with $u_{\max}$ tracking the monolith to four digits and an identical divergence across all three columns, so it is the cut and not a blow-up. Every criterion degrades over the horizon, **the derived one included**: L2/C2's $\mathcal Q^\star$ ranks the one-interval defect at $+0.9998$, because it is a theorem about that interval and cannot do otherwise, and the eight-interval defect at $+0.718$. The eight-interval optimum — a real $2\times$ — was found by no criterion. **The gate as W112 poses it is passed and the gate as posed is the wrong gate**, and that is **W123**.

**What this says about $\mathcal Q$ specifically, in both directions.** It does **not** rank backwards on placement in a real wake field ($+0.40$ to $+0.87$ across three geometries), so §10.2's headline was too harsh read as a general claim. It **does** rank backwards — exactly $-1.000$, with both of its factors — on the **overlap**, an axis §10.2 never varied: a wider halo makes the operator less diagonal and the scheme more accurate, so it prefers the narrowest overlap when the narrowest is worse by $4.993\times$. And its basis dependence was **understated**: over $120$ real seams the orbit is $1.24\times$ to $85.4\times$ with a median of $5.65\times$, $42.5\%$ of seams admit more than $10\times$, and $\beta$ is invariant to $2.9\times10^{-15}$ throughout.

**CS-9's shared domain gives the sharpest result and the one that generalizes.** The conduction/elasticity interface has co-dimension zero, so there is no family $\Gamma$ to place — CS-9's own finding restated. The spatial cut, which cuts both families, exists: distance from the heat streak ranks $\mathbf{-0.9988}$ against the conduction defect and $\mathbf{+0.9991}$ against the elasticity defect, the two rank at $-0.0259$ against each other over all $785$ candidates, and **elasticity's optimum costs conduction $795\times$**. One $\Gamma$, two physics, opposite preferences — and no scalarization of one seam operator can express that in any basis, because the disagreement is not in the operator. That is **W126**, and three of four Phase C subsystems inherit it.

Also opened: **W125**, that a criterion can have a good argmin and a bad ordering and nothing in the framework distinguishes them. Reframed: **W16/G5** (scope wrong in both directions), **W58** (the aggregation question recurs on criteria, a $2.8\times$ spread between $\mathcal Q$'s two forms), **W105** (the arrangement the march runs in — stable to $240$ intervals and not incompressible).

**And a rule, because this is the third time.** [[tier0-measurements]] §18.5 published a state on a diverging trajectory; §19.6's repair was stable exactly as far as its positive control was marched and died at step $38$; §21.3's placement reverses at interval $10$. **A comparison between two configurations is not a result until it has been marched past the point where the two curves could cross, and the crossing has to be looked for rather than assumed absent.**

Changed: [[tier0-measurements]] (§21, new), [[case-study-seam-placement-atlas-0.1]] (new), [[gap-worklist]] (Tier 23; W112 closed; W123–W126 opened), [[index]].
