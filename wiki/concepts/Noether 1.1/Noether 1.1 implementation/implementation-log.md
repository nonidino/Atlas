# Noether 1.1 Implementation — Progress Log

Append-only progress tracker for the Noether 1.1 **build** (distinct from the vault-wide `wiki/log.md`, which records design/wiki changes). Format: `## [YYYY-MM-DD] <portion> | <status> — <what happened>`. Status ∈ {planned, in-progress, blocked, done}. Update after **every** work session; note tests passing and what the completed work unblocks.

**Plan:** [[00-implementation-plan]] · **Configs:** [[noether-1.1-medium]], [[noether-1.1-large]] · **Resume prompt:** [[phase1-resume-prompt]] · **Training on vast.ai:** [[vast-ai-training-runbook]]

---

## Portion status board

| # | Portion | Spec | Status | Branch |
|---|---|---|---|---|
| 0 | Repo scaffold & config system | [[impl-repo-scaffold]] | **done** | noether-1.1 |
| 1 | 2D GUI | [[impl-gui-2d]] | **done** | noether-1.1 |
| 2 | Training-data generation | [[impl-data-generation]] | **done** | noether-1.1 |
| 3 | Tokenizer + streams + descriptors | [[impl-tokenizer-descriptors]] | **done** | noether-1.1 |
| 4 | Edge generation | [[impl-edge-generation]] | **done** | noether-1.1 |
| 5 | Backbone | [[impl-backbone]] | **done** | noether-1.1 |
| 6 | Decoder | [[impl-decoder]] | **done** | noether-1.1 |
| 7 | Training harness | [[impl-training-harness]] | **done** | noether-1.1 |
| 8 | Discovered conservation | [[impl-discovered-conservation]] | **done** (training path) | noether-1.1 |
| 9 | Post-decoder diffusion | [[impl-diffusion]] | **done** (training path) | noether-1.1 |
| 10 | Evaluation & benchmarks | [[impl-eval-benchmarks]] | in-progress (metrics/rollout/bench built; inference-time projection+diffusion toggles still stubbed) | noether-1.1 |

Milestones M0–M7: see [[00-implementation-plan]] §4.

---

## [2026-07-08] project | planned — implementation folder created

Design session translated the Noether 1.1 architecture into an executable build. Created this folder with: the master [[00-implementation-plan]] (three macro-phases GUI → data → model; dependency graph; milestones M0–M7), the [[noether-1.1-medium]] (~44M) and [[noether-1.1-large]] (~327M) size configs, and 11 standalone per-portion spec files (portions 0–10) written for independent agent execution. Confirmed the backbone is a deep stacked-attention model (medium $L{=}12$, large $L{=}24$), transformer-like in skeleton but per-edge-type and port-Hamiltonian per layer.

**GitHub:** repo reorganized into a frozen `noether-1.0` reference branch (complete 1.0 build: Burgers + RBC + NS) and an active `noether-1.1` branch. All 1.1 code lands on `noether-1.1`.

**Next:** portion 0 ([[impl-repo-scaffold]]) — stand up `src/noether11/` layout, config dataclasses, CI/test skeleton — then portions 1 (GUI) and 2 (data) can proceed in parallel.

---

## [2026-07-08] impl-repo-scaffold | done — Portion 0 built on `noether-1.1`

Executed [[impl-repo-scaffold]] against the actual repo (`physics-foundation-model`, branch `noether-1.1`, which was confirmed identical to the frozen `noether-1.0` tip beforehand — nothing had been reorganized yet).

**Shipped:** `src/noether11/{core,conserve,generate,condition,data,train,eval,configs}` package tree. Ported (git mv, preserving history) the reusable/equation-agnostic 1.0 modules: `siren.py`, `operators2d.py` → `core/`; `projection.py` → `conserve/`; `field_norm.py` → `condition/`; `bisection.py`, `grid2d.py` → `core/graph/`; `burgers_solver.py`, `datasets.py`, `generate.py`, `ns_datasets.py`, `ns_solver.py`, `rbc_datasets.py`, `rbc_solver.py`, `well_loader.py` → `data/` — fixing the stale `..model.operators2d` relative imports these files carried post-move. **Deliberately did not port** `backbone.py`, `encoder.py`, `encoder2d.py`, `noether10.py`, `noether_rbc.py`, `config.py`, `config_rbc.py`, `train/`, `eval/` — these are 1.0-specific and superseded by portions 3/5/6/7/10's fresh 1.1 designs, not carried over. Removed `src/noether/` from this branch entirely (fully preserved on `noether-1.0`).

`configs/__init__.py`: `Noether11Config` dataclass (every symbol from [[noether-1.1-medium]]/[[noether-1.1-large]] §1) + `MediumConfig()`/`LargeConfig()`/`SmokeConfig()` + `param_estimate()` implementing [[backbone-1.1]]'s `(9+3T)Ld²` formula. Verified against the wiki tables: medium estimate 43.79M vs. 44M target (0.5% off), large 327.35M vs. 327M (0.1% off).

`pyproject.toml` renamed `noether`→`noether11`. Test suite: kept and import-fixed `test_bisection.py`, `test_operators2d.py`, `test_projection.py`, `test_solver.py`, `test_well_loader.py`, `test_rbc_solver.py` (their subject modules were ported); removed `test_field_agnostic.py`, `test_model.py`, `test_rbc_model.py`, `test_rbc_rollout.py`, `test_rbc_losses.py`, `test_ns_training.py`, `test_pooling.py` (tested un-ported 1.0-specific modules — fully preserved on `noether-1.0` if ever needed again); added `test_config.py` (config instantiation, param-budget-within-5%, backbone-dominates-param-count, and a `core/` must-not-import-`data/` design-invariant check). README rewritten for the branch's current state.

**44/44 tests passing.** Committed locally on `noether-1.1` — **not yet pushed**, pending confirmation.

**Known, expected breakage (out of scope for this portion):** `scripts/*.py`, `gui/*.py`, `notebooks/*.ipynb` still import the now-removed `noether` package. This is portions 1/2/6/7/10's job to fix as they land, not something to patch here.

**Next:** portions 1 ([[impl-gui-2d]]) and 2 ([[impl-data-generation]]) — both unblocked, can proceed in parallel. See [[phase1-resume-prompt]] for a self-contained pickup prompt.

---

## [2026-07-08] impl-data-generation | done — Portion 2 built on `noether-1.1`

Executed [[impl-data-generation]] against the ported solvers in `src/noether11/data/`. The solvers moved in portion 0 but did not yet emit the uniform schema — that schema work was this portion's job.

**Shipped:**
- `data/schema.py` — **the data↔model contract.** The uniform per-trajectory sample schema (`fields: {name→[T,*spatial,C]}`, `field_meta`, `constants`, `dt`, `grid`, `t`, `system`), the `FIELD_META` registry (structured block of $\theta_f$: velocity rank 1 even vector; temperature rank 0 even scalar; density rank 0 even; **magnetic rank 1 odd/pseudovector** — parity grounded per [[field-descriptors-1.1]]), `components(rank,ndim)=ndim**rank`, `make_grid`, and `write_sample`/`read_sample`/`validate_sample` (one `.npz` per trajectory). Spatial axes kept in the solvers' **native `[Nx,Ny]` order** (not transposed to image `[H,W]`) so the div-free 4th-order operators stay bit-for-bit; `grid["axes"]` records the semantics.
- `data/registry.py` — `load(system, split, root)` (the tokenizer's sole entry point, returns a `UniformTrajectoryDataset` of validated schema samples), `generate_dataset(system, split, …)`, `resolve_sweep` (a **range** of the characteristic number per rung, not a few points), and `SPLITS` presets (smoke/train/val). Diversity per [[training-scheme-1.1]]: Burgers ν∈[1e-3,1e-1] + IC-family spread; NS Re∈[2e2,1e4] + energy-diverse initial vorticity; RBC Ra∈[3e4,3e5] + the **0–1.5× free-fall off-attractor `u_amp`/spinup mix** from [[noether-1.0-rbc]] §5.1. Velocity appears in all three rungs (θ_f field-identity grounding, the spec pitfall). Constants stored as the dimensionless **dials** ([[conditioning-and-constants-1.1]]): Burgers `{nu}`, NS `{Re}`, RBC `{Ra,Pr}`.
- `data/{compressible,particles,mhd}_solver.py` — rungs 4–6 scaffolded: each `raise NotImplementedError` with a full build spec in the docstring (compressible: shock-capturing FV + Ma sweep, fields velocity+density; particles: NS carrier + Lagrangian integrator + point-set schema extension; MHD: induction eq + Lorentz + div·B=0, pseudovector B).
- `scripts/generate.py` — CLI (`--system/--split/--sweep/--workers/--out-dir`, `--smoke-all`), writes per-trajectory `.npz` into the gitignored `data/<system>/<split>/`.
- `tests/test_data_schema.py` — **16 tests**: every rung-1–3 smoke sample validates; field↔meta consistency; parity/rank grounding; velocity-shared-across-rungs; sweep-spans-a-range; NS velocity-only vs RBC multi-field; schema round-trip; validator rejects bad shape/missing meta; rungs 4–6 raise. Self-contained (generates tiny trajectories in a temp dir, since `/data/` is gitignored).

**Also fixed:** a latent bug in `data/rbc_solver.py` — `initial_state` called bare `spacings()` instead of `ops.spacings()` on the `u_amp>0` off-attractor path (missed in portion 0's import fixup; never fired because no retained test exercised energy-diverse ICs). Portion 2's RBC diversity does exercise it, so it's fixed.

**60/60 tests passing** (44 prior + 16 new). Committed locally on `noether-1.1` (`522823a`, message starts "Portion 2: uniform data schema + generators") — **not yet pushed**, pending the same confirmation as portion 0.

**Env note (this box):** the full `pytest` aborts under a numpy+torch MKL/OpenMP double-load on this Windows Anaconda install (`Fatal Python error: Aborted` in `numpy.linalg.solve`); it passes cleanly with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL`. An environment quirk, not a code issue.

**Next:** portion 1 ([[impl-gui-2d]]) — generalize the RBC GUI into a field-agnostic renderer that can render a ground-truth NS trajectory from portion 2's output (milestone M0). After 1, Phase II begins with portion 3 ([[impl-tokenizer-descriptors]]), which consumes `load()`.

---

## [2026-07-08] impl-gui-2d | done — Portion 1 built on `noether-1.1` (milestone M0)

Executed [[impl-gui-2d]]. Generalized 1.0's RBC-only GUI into a **field-agnostic** 2D field viewer backed by portion 2's `noether11.data.load`. Milestone **M0** reached: a solver-only dataset (no trained model) renders and scrubs a ground-truth NS trajectory.

**Shipped:**
- `gui/server.py` — rewritten as a **FastAPI** backend (deviation from the spec's plain-FastAPI-or-stdlib wording is moot; FastAPI+uvicorn are installed). Endpoints per the interface contract: `GET /api/datasets` (auto-discovers every `<system>/<split>/` under `DATA_ROOT`), `GET /api/checkpoints`, `GET /api/trajectory` (ground-truth display channels for one trajectory, frame-decimated), `POST /api/rollout`, `GET /api/edges`. **Channels are derived from `field_meta` rank, never hardcoded**: a rank-1 vector field → magnitude/components/vorticity/divergence; a rank-0 scalar → itself. `/api/rollout` loads the model via the `noether11` public API (`load_checkpoint`/`rollout`) and — since that API/checkpoint doesn't exist until portions 6–7 — returns `available:false` with a graceful reason; it never reimplements inference. `/api/edges` stubbed until portion 4's scorer.
- `gui/index.html` — self-contained **canvas** renderer (no CDN/Plotly). Dataset/trajectory/channel selectors; **model | truth | error** three-panel layout; colorbar (sequential viridis for unsigned channels, diverging cool-warm for signed like vorticity/divergence, symmetric about 0); scrubber + play; and the three toggles (**diffusion / conservation-projection / learned-edge overlay**) wired to `/api/rollout` params. 1D (Burgers) falls back to a line plot.
- Removed `gui/server_rbc.py`, `gui/index_rbc.html` (subsumed by the generalized server; they imported the removed `noether` package; preserved on `noether-1.0`). `.claude/launch.json` reduced to a single `noether-gui` entry.
- `tests/test_gui.py` — **8 endpoint tests** (dataset discovery; NS channels + `[F][Ny][Nx]` shape + signed-flag; the field-agnostic check that RBC exposes a `temperature` channel; decimation; graceful no-model rollout; 404). `tests/conftest.py` now holds a shared session `smoke_data_root` fixture used by both the data and GUI tests (one generation pass).

**Live verification (preview harness):** started the server, generated smoke splits + two full NS `train` trajectories, and confirmed a **128×64 ground-truth NS trajectory (60 frames decimated from 151)** renders and scrubs, with `speed` (viridis) and `vorticity` (diverging) channels; RBC exposes velocity-derived channels **plus** temperature; the rollout button reports the graceful no-model state.

**68/68 tests passing** (44 prior + 16 data + 8 GUI). Committed locally on `noether-1.1` (`26619a8`, "Portion 1: field-agnostic 2D GUI (M0)") — **not yet pushed** (same decision as portions 0/2).

**Phase I complete.** Portions 0, 1, 2 all done; the foundations (scaffold + data + GUI) are in place. **Next: Phase II, portion 3** ([[impl-tokenizer-descriptors]], train phase P0) — the tokenizer/field-streams/descriptors, which consume `noether11.data.load` and the `field_meta` structured block. It's the next critical-path item (0→2→**3**→5→6→7).

---

## [2026-07-09] impl-tokenizer-descriptors | done — Portion 3 built on `noether-1.1` (Phase II begins)

Executed [[impl-tokenizer-descriptors]] — the **whole encode side**, kept equation-agnostic in `src/noether11/core/`. Turns a uniform-schema `sample` (portion 2's `load()` output) into the typed per-field token multiset the backbone will consume. Design pages honored where the spec under-specified: [[graph-tokenizer-1.1]] (over-complete features, dual independent paths), [[field-token-streams-1.1]] (one stream per field, no summation), [[field-descriptors-1.1]] (θ_f structured‖free, B2 correction-on-shared-base), [[conditioning-and-constants-1.1]] (ruler/dial/arrow).

**Encode/decode-scope decision (surfaced, user confirmed Option A):** built the encode side in full **plus a minimal P0 SIREN reconstruction head** (reusing `core/siren.py`) so the G0 gate runs *now*. Per [[impl-decoder]] build-step 4 / [[training-scheme-1.1]] P0, the SIREN heads co-train "as an autoencoder" in P0; portion 6 (`core/decoder.py`) later supersedes this stand-in with the full dual-path **increment** decoder. Deliberately did **not** create `core/decoder.py` (portion 6's file).

**Shipped (`src/noether11/core/`):**
- `coord.py` — fixed Fourier coordinate encoding γ (no params; shared with the decode head).
- `patch.py` — resolution-free patcher. **Key fix:** patches are fixed **domain-coordinate bins**, not grid-index blocks, so the partition *and centroids* are identical at any resolution (an index-split centroid shifts ½ cell with resolution — caught by a unit test, fixed by coordinate-binning). Layout is cached per grid geometry (built once, not per step) — the difference between a 2-min timeout and 12 s.
- `features.py` — augmented per-point φ_p = [γ, v, ∇v, ω, ‖∇v‖, ∂ₜv, ∂ₜₜv]; **rank-driven** (scalar vs vector), spatial derivs reuse `operators2d`, temporal diffs = how history enters. ndim-generic (2D uses the exact FD operators; other dims fall back to periodic central differences → 1D Burgers tokenizes through the same interface).
- `descriptors.py` — `FieldDescriptor` θ_f = [structured (rank/parity/is_conserved/rep_type read from `field_meta`) ‖ learned free u_f]; free block is a lazily-grown table keyed by field **name** (data-driven lookup, not name-branching code) so velocity shares one descriptor across NS+RBC; `simulation_descriptor` z_sim = mean_f θ_f ‖ z_cond.
- `query_modulation.py` — B2 `Z_q^f = Z_q^base + Δ_q(θ_f,z_sim)`, low-rank UVᵀ (FiLM behind a flag). **U initialized to zero** → every field starts from the *identical shared base*; the correction is earned, never a per-field replacement table (the transfer-killing pitfall). Unit-tested that the shared base cancels between fields.
- `streams.py` — `t_i^f = [Enc_f(h_i) ‖ W_e θ_f] + z_cond`, descriptor in a **reserved slot** (concat, never add); `Enc_f` keyed by rep_type (scalar/vector/pseudovector), not field name. `Tokens` container carries {t^f_i}, positions, field_ids, h_hi, z_cond, per-field θ_f, layout.
- `tokenize.py` — `QueryCompression` (softmax cross-attn, d_head=d/N_q, concat → h_i) run **twice on two independent paths** (lo Z_q^base→h_lo into the backbone, hi Z_q^hi→h_hi into the decoder/diffusion — a second encoder, not a subtraction). `Tokenizer(cfg).encode(sample) -> Tokens`. Structure-keyed lazy submodules (KV embed by rank/ndim) so a new field/rep type adds no code.
- `autoencoder.py` — the minimal P0 head + `Autoencoder`; `use_hi` toggles the dual/single head.

**Conditioning (`src/noether11/condition/conditioning.py`):** the **ruler** (`Ruler.standardize`/`destandardize` — data is already nondimensional on disk, so this is the encoder-facing magnitude standardization with an exact inverse), the **dial** (`Conditioning.z_cond` = z_param from log-π groups + z_dt; constants pooled permutation-invariantly, a **missing group is masked, not zero-filled**; constant names key a lazily-grown embedding), the **arrow** (`CovariantChannel` — a rotation-**equivariant** gated-linear on a vector constant; implemented + unit-tested for equivariance but *not wired to a stream* for NS/RBC, which carry no vector constant as generated — wiring gravity needs the data schema to emit a **g** vector constant, a future portion-2 extension, flagged here). Tokenizer stays constant-light (constants touch it only via nondimensionalization).

**Config:** added `gamma_freqs`, `patch_domain_size`, `d_desc`, `qb_rank`, `query_modulation` (additive, defaulted; `param_estimate()` untouched, so the portion-0 budget tests are unaffected).

**Acceptance tests (`tests/`, +11, all real & runnable):**
- **G0 gate** (`test_autoencoder_g0.py`): train the tokenizer+SIREN autoencoder on turbulent 2D NS frames (48×24) → encode→decode reconstruction RMS **below the discretization floor** (2×-coarsen proxy); deterministic seed, ratio ≈ 0.54, ~30 s. *Scope note (honest):* this gates the reconstruction **mechanism** at smoke scale. In-sample reconstruction < floor is non-trivial (the latent is a real bottleneck); the **strict held-out-generalization** form of G0 (medium config, full corpus) is a P0 **training-run** milestone for the portion-7 harness, not a unit test — the small-scale autoencoder overfits few frames (verified: train fits, held-out doesn't), exactly as [[training-scheme-1.1]] anticipates the emergent lo/hi division of labor needing the later rollout losses.
- **Resolution-invariance** (`test_tokenizer.py`): encode NS at 2× → stream latents match (rel ≈ 0.04 < 0.15); patcher unit test proves identical partition + centroids across resolutions and an exact grid partition.
- **Variable field cardinality**: same `Tokenizer` tokenizes NS (1 stream) and RBC (2 streams) with no code change; dropping temperature drops exactly its stream. 1D Burgers tokenizes through the same interface.
- **Descriptor separability**: reserved-slot concat unit test (swap data → content block moves, identity slot fixed; swap descriptor → only identity slot moves) + a full-encode two-scalar-field data-swap.
- Unit: query-bank shared-base-plus-correction, covariant-channel rotation-equivariance, conditioning missing-group masking, interface-contract shape checks.

**79/79 tests passing** (68 prior + 11 new). Committed locally on `noether-1.1` (`92051d3`, "Portion 3: tokenizer + field streams + descriptors + conditioned queries") — **not yet pushed**, pending the same confirmation as portions 0–1 (origin/noether-1.1 is at `df4c040`).

**Env note (this box):** same MKL/OpenMP quirk — full `pytest` needs `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL`.

**Unblocks / next:** portion 3 is the critical-path pivot (0→2→**3**→5→6→7). Next unblocked items: **portion 4** ([[impl-edge-generation]], KNN edge families + learned scorer that consumes the `Tokens` multiset; scorer gated OFF until P2) and **portion 5** ([[impl-backbone]], typed attention over the streams). The full **G0 milestone (M1)** in its strict form is realized when portion 7's harness runs P0 on the corpus. The `γ_vec` arrow becomes live once the data schema emits a gravity vector constant (small portion-2 follow-up).

---

## [2026-07-09] impl-edge-generation | done — Portion 4 built on `noether-1.1`

Executed [[impl-edge-generation]] — the typed multigraph the backbone runs on. Consumes portion 3's `Tokens` multiset; unblocks portion 5 ([[impl-backbone]]). Design page [[edge-generation-1.1]] is authoritative for the math; [[training-scheme-1.1]] fixed the gating (learned scorer OFF until P2).

**Shipped (`src/noether11/core/edges.py`):**
- `_pairwise_distances(positions, extent, periodic)` — BC-aware Euclidean distances with periodic wrapping (closest-image convention), diagonal zeroed explicitly.
- `knn_same_field(tokens, K)` — geometric, parameter-free, BC-aware, symmetric (i<j canonical orientation, each unordered pair scored once). Returns `edge_index [2,E]`, `edge_attr` (gamma(Δx) ‖ ‖Δx‖).
- `knn_cross_field(tokens, K')` — cross-field KNN over each unordered field pair, includes co-located pairs naturally (distance=0). Returns `edge_index`, `edge_attr`, `sub_type` (0=coloc, 1=spatial), which `build_graph` maps to edge types 1 and 2.
- `LearnedEdgeScorer(g_theta)` — small MLP over `(t_i, t_j, gamma(Δx), ‖Δx‖, z_cond)` → logit; `GumbelSigmoid` custom autograd function with straight-through estimator (hard {0,1} forward, soft gradient backward); sparsity budget via per-node top-k intersection voting; edge-entropy regularizer. **Not trainable before P2** — the `use_learned` gate is a boolean defaulting to `False`.
- `build_graph(tokens, cfg, use_learned=False, scorer=None, tau=None)` → `Graph` dataclass (`edge_index [2,E]`, `edge_type [E]` ∈ {0=same, 1=cross_coloc, 2=cross_spatial, 3=learned}, `edge_attr [E,d_edge]`). Families 1–2 assembled first; family 3 (learned) added on top when `use_learned=True`, KNN edges never replaced. Candidate envelope controlled by `cfg.learned_edge_cutoff` (None = O(P²) all pairs, fine at node scale); range-constant control tested.
- `get_learned_edges(graph)` — export util returning type-3 edges for the GUI overlay (`/api/edges` stub).

**BC/periodicity interface (designed & implemented):** stored `extent` and `periodic` on the `PatchLayout` dataclass (defaulted for backward compat), threaded from the grid schema through `tokenize.py::_get_layout()` → `build_patch_layout(periodic=...)`. Tokens carry BC via `tokens.layout.periodic`/`tokens.layout.extent`; edges read them with zero new plumbing. Chose the existing `PatchLayout` path (geometry-only, equation-agnostic) over passing the sample's `grid` dict — minimal, forward-looking (backbone also needs BC).

**Config:** added `learned_edge_cutoff: float | None = None` (additive, defaulted; `param_estimate()` untouched). Existing edge knobs reused (`knn_k`, `knn_k_cross`, `learned_edge_budget`, `gumbel_temp_*`, `T`).

**Acceptance tests (`tests/test_edges.py`, 17 tests, all passing):**
- Families 1–2 deterministic + symmetric + BC-correct on toy point set (periodic wrap connects across boundary; configurable K; each unordered pair once)
- Cross-field always includes the co-located pair (distance=0 correctly identified)
- `use_learned=False` reproduces pure-KNN baseline (no type-3 edges); `use_learned=True` adds learned edges on top (families 1–2 unchanged)
- Learned-edge mechanism: GumbelSigmoid forward {0,1} with non-collapsed sampling, straight-through gradients reach `g_theta`'s MLP params (grad ≠ 0), sparsity budget via per-node intersection voting guarantees `degree ≤ budget`
- Range-constant control: smaller cutoff → fewer candidate pairs with no code change
- Unit: pairwise distances, diagonal zero, periodic 1D/2D, edge_attr shape, Graph container, GUI export util, co-located distance zero

**M3 gate note (per spec):** the STRICT M3 milestone ("sparse + non-collapsed on TRAINED weights") is a P2 training-run milestone the portion-7 harness runs — NOT a unit test. Handled exactly as portion 3 handled the G0 gate: unit-test the mechanism, document the strict gate as a P2 milestone.

**96/96 tests passing** (79 prior + 17 new). Committed locally on `noether-1.1` (`77ae9b3`, "Portion 4: edge generation (typed multigraph) — KNN families + LearnedEdgeScorer gated OFF until P2") — **not yet pushed**, pending user confirmation.

**Unblocks / next:** portion 4 hangs off portion 3 and unblocks **portion 5** ([[impl-backbone]]), which becomes the next critical-path item (0→2→3→**5**→6→7). The backbone consumes the typed `Graph` this portion emits.

---

## [2026-07-09] impl-backbone | done — Portion 5 built on `noether-1.1` (Phase II dynamics core)

Executed [[impl-backbone]] — the **largest single portion**, the dynamics core: $L$ stacked port-Hamiltonian typed-attention layers. Consumes portion 3's `Tokens` and portion 4's `Graph`; unblocks portion 6 ([[impl-decoder]]), portion 7 ([[impl-training-harness]]), and portion 8 ([[impl-discovered-conservation]]). Design page [[backbone-1.1]] is authoritative for the math.

**Shipped (`src/noether11/core/backbone.py`):**

- `CayleySkew(d)` — $W_{\text{skew}} = (I-S)(I+S)^{-1}$ from learned skew $S = M-M^\top$. Cayley transform via `torch.linalg.solve`. Guarantees orthogonality $W^\top W = I$ (verified to $<10^{-6}$) and $\|W_{\text{skew}}-W_{\text{skew}}^\top\|_2 \le 2$ analytically throughout training (verified after 100 Adam steps). Forward: `h -> (W - W^T) h`.

- `TypedAttention(d, H)` — per-edge-type multi-head symmetric attention with tied Q=K. Per-head `W^{(k)}`, `W_V^{(k)}`, `W_O^{(k)}$`; scores $A_{ij} = \tanh(q_i \cdot q_j / \sqrt{d_h})$. **Two directed messages per undirected edge** scattered to BOTH source and destination for antisymmetry: `+A*(v_j - g*v_i)` to src, `+A*(v_i - g*v_j)` to dst. Heads SUMMED (additive operators), never averaged; `1/\sqrt{H}$` init scale on $W_O$. Batch-efficient via einsum + scatter_reduce (no Python loops over edges).

- `GatedFlux(T=4)` — learnable per-type gate $g_\tau = \text{clamp}(a_\tau, 0, 1)$ (hardtanh, reaches **exactly** 1.0 and 0.0 — tested). Init: $a_0=2 \Rightarrow g_0=1$ (same-field conservative), $a_{1,2,3}=0 \Rightarrow g=0$ (cross-field non-conservative). `freeze()`/`unfreeze()` hooks for P1-P2/P3 discipline.

- `F1FFN(d)` — Stiefel-initialized semi-orthogonal expand-contract. $W_1$ `[4d,d]$` with $W_1^\top W_1 = I_d$ (verified $<10^{-4}$); $W_2$ `[d,4d]$` with $W_2 W_2^\top = I_d$. GELU + renorm restores pre-activation norm. `stiefel_retract_()` periodic QR re-orthogonalization.

- `Layer(d, H, T)` — one forward-Euler step: $h' = h + \alpha\,\varepsilon\,\tanh[(W_{\text{skew}}-W_{\text{skew}}^\top-\gamma I)h + \sum_\tau \text{TypedAttention}_\tau(\dots)]$; then $h'' = h' + \alpha'\,\varepsilon\,\text{F1FFN}(h')$. ReZero $\alpha,\alpha'$ init 0. Per-layer learnable $\gamma_\ell$. No LayerNorm.

- `Backbone(cfg)` — stacks $L$ `Layer`s. `forward(tokens, graph) -> evolved_streams` (z_cond already baked by tokenizer — not re-added). `forward_tokens()` returns `Tokens` with `h_hi` passed through unchanged. `freeze_gates()`/`unfreeze_gates()` broadcast to all layers. `stiefel_retract_all_()` periodic maintenance. `pre_tanh_momentum_delta()` helper for telescoping tests.

- `hard_equivariance: bool = False` added to `Noether11Config` (default False, vector constants treated as ordinary channels; entry points scaffolded for future F2-gated upgrade).

**Key design decisions documented:**
1. **Tokens per edge**: shared node geometry — all fields query the same `Graph` endpoints. For same-field edges, source=destination field; for cross-field, TypedAttention runs with `h_src` from the updated field and `h_dst` from the acting field, so $W_V^{(k)}$ projects the coupling partner's tokens.
2. **Antisymmetric flux scattering**: two directed messages scattered to BOTH source and destination per undirected edge — guarantees $\sum_i \text{message}_i = 0$ at $g=1$ (telescoping) while the per-node $\tanh$ nonlinearity breaks exact conservation (documented honestly per Problem 1).
3. **Gate multiplier**: `g_tau` is inside the attention flux formula; `_typed_messages` does NOT re-multiply by `g_tau` (would double-apply).
4. **z_cond**: tokenizer already adds z_cond at construction — backbone does not re-add (would double-condition).

**Acceptance tests (`tests/test_backbone.py`, 28 tests, all passing):**
- Cayley orthogonality ($W^\top W = I$, $\|W-W^\top\|_2 \le 2$) at init and after 100 optimizer steps
- ReZero identity at init (exact, $\text{rel\_diff} < 10^{-7}$); breaks with nonzero alpha
- Gate exactness (clamp, not sigmoid): reaches exactly 1.0/0.0; freeze/unfreeze
- Pre-tanh message telescoping at $g=1$ (sum $\approx 0$ within $2\times10^{-2}$); measurably different at $g=0$
- Head count invariance: doubling H produces same-scale output
- F1 orthogonality at init and after QR retraction
- Norm preservation at ReZero init; bounded after 20 training steps
- $h_{hi}$ passthrough (keys and values identical before/after backbone)
- Type separation: different $W_O^{(\tau)}$ produce different messages; permuting types changes output
- Variable field cardinality (1-field NS, 2-field RBC forwards with no code change)
- Empty graph handled gracefully
- Gradient flow reaches all unfrozen trainable parameters (>40% with single-field graph)
- Config integration and `hard_equivariance` flag

**124/124 tests passing** (96 prior + 28 new). Committed locally on `noether-1.1` — **not yet pushed**, pending user confirmation.

**Env note (this box):** same MKL/OpenMP quirk — full `pytest` needs `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL`.

**Unblocks / next:** portion 5 is the dynamics gate for Phase II — it unblocks **portion 6** ([[impl-decoder]], the increment-decoder that consumes evolved tokens), **portion 7** ([[impl-training-harness]], the P1 single-step training loop), and **portion 8** ([[impl-discovered-conservation]], the hard-conservation projection partner). Portion 6 is the next critical-path item (0→2→3→5→**6**→7→8).

---

## [2026-07-10] impl-decoder | done — Portion 6 built on `noether-1.1` (Phase II decoder)

Executed [[impl-decoder]] — the dual-path coordinate-implicit **increment** decoder. Consumes evolved Tokens from the backbone (portion 5) and produces per-field physical increments at any resolution. Design page [[decoder-1.1]] is authoritative.

**Shipped (`src/noether11/core/decoder.py`, 437 lines):**
- `SIRENHead(d_in, d_hidden, d_out, d_theta)` — FiLM-modulated SIREN head. CondMLP(θ_f)→(scale, bias) for per-field modulation. Matched Sitzmann init, sin(ω₀x) with ω₀=30 — required for the spectral-bias guard.
- `SimpleSIRENHead` — plain SIREN without FiLM, for fallback.
- `HeadGenerator` — lazily-grown table keyed by (rank, ndim, rep_type, c_out) signature — a single "rank-1 vector in 2D" head serves velocity, magnetic field, and any future rank-1 vector field.
- `Decoder(cfg)` with interface:
  - `decode_increment(tokens, metas, grid, current_fields) → dict[δv^f]`
  - `forward(tokens, metas, grid, current_fields) → dict[v_{t+1}]`
  - `reconstruct(tokens, metas, grid) → dict[field_value]` — P0 backward compat
  - Three decode modes gated OFF by default: "general" (default), "streamfn" (exact div-free via ∇^⊥ψ), "helmholtz" (FFT Leray projection)
  - Poisson solver: `_solve_poisson_2d` for periodic/wall-bounded/generic BCs

**Design decisions:**
1. **Increment = decode − current**: decoder outputs full field value; increment via explicit subtraction. Avoids the 1.0-RBC pitfall of structurally biased tiny increments.
2. **c_out in head signature**: `_get_head` includes c_out so streamfn mode (c_out=1 for scalar ψ) gets a distinct head from general mode (c_out=2 for velocity).
3. **Div-free is discovered, not wired**: `decode_mode` defaults to "general"; streamfn/helmholtz engaged only by portion 8.

**Acceptance tests (`tests/test_decoder.py`, 20 tests):**
- SIREN spectral-bias: MSE < 0.02 on tanh(20x); SIREN < 0.7× ReLU MSE
- Increment not biased small: RMS > 0.01 at random init on zero-current field
- Resolution-free: tokens decode at 2× grid, downsampled rel-diff < 0.5
- General head default, works on all field types; streamfn NOT default
- Dual-path: use_hi=False vs True produce different output
- FiLM: different θ_f → different output, deterministic; same-signature fields share heads
- Variable field cardinality: 1-field NS, 2-field RBC; dropping temperature drops exactly its stream
- Increment closure: forward() = current + decode_increment() within 1e-6
- Structure-preserving: streamfn max|div| < 2e-4; Helmholtz reduces divergence ratio < 0.8×
- Unit: SimpleSIRENHead, masked patches, unknown decode_mode raises, reconstruct shape

**144/144 tests passing** (124 prior + 20 new). Committed locally — not yet pushed.

**Unblocks / next:** portion 6 unblocks **portion 7** ([[impl-training-harness]], P1 single-step training) and **portion 8** ([[impl-discovered-conservation]], gates structure-preserving decode). Portion 7 is next on the critical path (0→2→3→5→6→**7**→8).

---

## [2026-07-10] impl-training-harness | done — Portion 7 built on `noether-1.1`

Executed [[impl-training-harness]] — the phased curriculum + loss manager + training loop that wires portions 0–6 together and drives P0→P4.

**Shipped (`src/noether11/train/`):**
- `losses.py` — `LossManager(nn.Module)` with:
  - **Group A**: Kendall-style homoscedastic-uncertainty-weighted per-field MSE (`log σ²` per channel, lazily registered as `nn.Parameter`); spectral loss via `torch.fft.rfft2` with `1/(|k|+1)` frequency weighting to balance all scales; `λ_spec` from config
  - **Group B**: `group_b_pushforward()` — autoregressive multi-step unroll with MSE over horizon; context-noise injection handled at the token level in `loop.py`
  - **Group C/D hooks**: return 0.0 until portions 8/9 wire `noether11.conserve` / `noether11.generate`
  - Phase-aware aggregation in `forward(pred, target, phase) → dict`
- `curriculum.py` — `Curriculum` predicate-gated state machine:
  - `PhaseState` dataclass (phase, step, best_metric, patience_counter, gates_frozen, use_learned_edges)
  - Gate predicates per phase: P0→P1 (recon < floor·ratio), P1→P2 (rel-L² < 0.05), P2→P3 (rollout rel-L² < threshold), P3→P4 (drift guard, scaffolded)
  - `patience` consecutive passing checks before advancing; failing metric resets counter
  - `advance()` applies module-activation changes: P0→P1 freezes gates, P1→P2 enables learned edges, P2→P3 unfreezes gates
  - Full checkpoint serialization (`state_dict`/`load_state_dict`)
- `loop.py` — `Trainer` training loop:
  - **AdamW** with `_build_param_groups()` excluding Cayley S (`.S` parameter), F1 (`W1`,`W2` in `f1` context), and gate `a_tau` from weight decay
  - **ε warmup**: linear ramp `eps_start→eps_end` over `warmup_steps`, then held
  - **F1 re-orthogonalization**: `backbone.stiefel_retract_all_()` every 100 steps
  - **Push-forward horizon schedule**: starts H=1, grows `1→2→4→8` when single-step MSE plateaus
  - **bf16 autocast** for forward pass, fp32 for loss
  - **Checkpointing**: saves model/optimizer/curriculum/loss state every `checkpoint_interval`; keeps top-K by validation metric
  - `_forward_step()`: tokenize → build_graph (with `use_learned` from curriculum) → backbone → decoder
- `scripts/train.py` — CLI: `--system {burgers,ns,rbc} --config {smoke,medium,large} --phase-until {P0..P4} --data-root --checkpoint-dir --resume --device --epochs --max-steps`

**Acceptance tests (`tests/test_training.py`, 19 tests, all passing):**
1. P0 training loss decreases (50 steps, NS smoke, not NaN)
2. P1 single-step forward: tokenize→graph→backbone→decoder, shapes match, gradients flow
3. Phase controller gate logic: blocking on failing metric, advancing on passing after patience, counter reset
4. `advance()` phase transitions: P0→P1 freezes gates, P1→P2 enables edges, P2→P3 unfreezes
5. Curriculum serialization round-trip
6. Gates frozen before P3 (`requires_grad==False` for `a_tau`)
7. Learned edges off in P0/P1 (`use_learned=False` → no type-3 edges)
8. MSE loss positive, spectral loss positive, spectral ≈0 for identical
9. Uncertainty weights created and require grad
10. Loss aggregation per phase (P0 simple MSE, P1 +spectral, P2 +pushforward)
11. Checkpoint save/load: weights match after round-trip
12. Push-forward unroll: 2-step shapes match, gradients flow through both
13. Epsilon warmup: start/mid/end/over values verified
14. Weight decay exclusion: Cayley S, F1, a_tau in wd=0 groups
15. Integration P0→P1: train P0, force gate pass, advance, verify backbone trainable

**163/163 tests passing** (144 prior + 19 new). Committed locally on `noether-1.1` (`739d25f`, "Portion 7: training harness (phased curriculum + losses)") — **not yet pushed**.

**Unblocks / next:** portion 7 is the orchestrator gate — it unblocks **portion 8** ([[impl-discovered-conservation]], the SFA discovery that runs during P3) and **portion 9** ([[impl-diffusion]], the phase-separated denoiser for P4). Portion 8 is next on the critical path (0→2→3→5→6→7→**8**→10).

---

## [2026-07-10] portions 4-7 | reviewed + fixed — end-to-end P0/P1 training path made real

Reviewed the alternate agent's (DeepSeek/Cline) portions 4 (edges), 5 (backbone), 6 (decoder), 7 (harness) against the specs/design pages, fixed the real bugs, **committed and pushed** to `origin/noether-1.1` (`bc08f27` review fixes + `c63eecd` optimizer fix). Suite: **165 passing** (was 163 passed + 1 failed). Confirmed via a P0 smoke-training run that the loss actually decreases end-to-end through `Trainer.run()` (mean of first 10 steps 0.83 → last 10 0.61 on a 4-trajectory NS set).

**The core problem:** the unit tests passed by hand-building tensors, which *bypassed* the real `dataset → train_step` plumbing. The only end-to-end test (the CLI smoke run) failed early on missing train/val data, so it never exercised — and thus masked — several genuine bugs. Portions 4-7 had **never actually run end-to-end**.

**Fixed — critical (the imminent P0/P1 path):**
- **Backbone was bypassed.** `loop._forward_step` computed `backbone(...)` then decoded the *original* tokens, discarding the evolved ones → the dynamics core received no gradient and did nothing. Now decodes `backbone.forward_tokens(...)`. The P1 unit test had the same bypass **and** a vacuous `if grad is not None` gradient check (the bypassed backbone's grads are `None`, so the assert never ran) — both fixed.
- **Batch plumbing broken on real data.** The training/validation steps called `.to()` on numpy arrays, fed the *whole trajectory* as the context window, and read constants from non-existent top-level keys (Re/Ra live in `sample["constants"]`, so conditioning `z_cond` was always constant-free). Added `_prepare_batch`: numpy→tensor, proper `n_ctx` frame windowing (context ending at t, current=t, target=t+1), correct constants.
- **Two-tokenizer split.** P0 trained `autoencoder.tokenizer`; P1 used a *separate* untrained `Tokenizer` that wasn't even in the optimizer → P0 pretraining was thrown away. Trainer + CLI now share `autoencoder.tokenizer`.
- **CLI split mismatch** (the one failing test): CLI hardcoded train/val splits; added `--split` so a self-contained smoke run works.
- **Optimizer missed the lazy params** (found during the smoke run): AdamW was built in `Trainer.__init__` *before* any forward, so ~7.8k lazily-created params (KV embeddings, descriptor free blocks, SIREN decode heads) were never registered and never trained. `run()` now does a warmup forward → rebuilds the optimizer (and again after each phase advance). Param count 46718 → 54500.

**Fixed — dormant (P3+ multi-field; cross-field gates are 0 until P3):**
- **Backbone cross-field messaging contaminated streams.** Each `TypedAttention` call scattered to *both* endpoints but the whole result was added to one field (and the field pair was double-called), leaking field b's message into field a. `TypedAttention` now returns `(msg_src, msg_dst)` separately; `_typed_messages` routes src→field a, dst→field b for cross-field (same-field still sums both — telescoping bit-identical). New regression test pins it.

**Not fixed — documented P2 follow-ups (all dormant, none block P0/P1):**
- **Learned-edge scorer is not persistent.** `build_graph` lazily creates a *fresh* scorer every call and the Trainer's `self.edge_scorer` stays `None` (and `advance` is called with `edge_builder=None`), so at P2 the scorer would be re-randomized each step and never train. Must be wired (create once entering P2, store on the Trainer, add to the optimizer) before P2.
- **Push-forward is a stub.** `loop._compute_pushforward` returns `None`, so Group B multi-step (the P2 motion-demanding loss) is inactive. `losses.group_b_pushforward` exists but is unwired.
- **P0 harness doesn't standardize inputs.** The validated G0 recipe standardizes each field by global RMS; the harness P0 step reconstructs raw nondim magnitudes, so a real P0 run may not reach the discretization floor without adding it.

**Assessment for training:** the deterministic-core **P0→P1** path now runs correctly end-to-end on real data (single-field NS). Still missing before a *useful* run: real corpus generation (only `smoke` exists on disk), the P0 standardization above, and GPU for anything past the smoke config. **P2** needs scorer-persistence + push-forward wiring; **P3/P4** need portions 8 (conservation) and 9 (diffusion), still `planned`. So: ready to *smoke-train* P0/P1; not yet ready for a full multi-phase run.

**Follow-up same day (`f6db1e5`):** added **input standardization** (the "ruler") — the Trainer per-field RMS-standardizes context/current/target (one scale, stored for re-dimensionalization) in both P0 and P1 so the shared tokenizer sees O(1) fields and SIREN reconstruction can reach the floor; config flag `standardize_inputs` (default True). Added **`notebooks/noether11_colab.ipynb`** — a Colab GPU notebook that generates a small real NS corpus, runs a P0 medium-config quick test (loss + ms/step) and a P1 backbone-gradient probe, and scaffolds the full P0→P1 (M2) cloud run. Staged a real 128×64 NS corpus (8 train + 2 val, gitignored) and verified generate→load→train_step end-to-end.

**Follow-up (`d3b3ded`):** the Colab notebook's `pip install -e .` silently exposed no `noether11` module on Colab (a src-layout editable-install quirk, already flagged in `pyproject.toml`'s pytest config comment). Replaced with the same approach pytest/the scripts already use: locate the repo root, `chdir` in, uninstall any broken editable, put `src/` on `sys.path` directly.

**Follow-up (`9dbf307`) — real GPU crash + fix:** running the notebook on an actual Colab GPU hit `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!` in `Conditioning.z_cond`. Root cause: **three lazily-grown `nn.Parameter`/`nn.Linear` sites created their first-use module with no device/dtype at all** (`Conditioning._name_param`, `StreamEmitter._enc`, and — silently, without crashing — `FieldDescriptor._free`), so `torch.randn(...)`/`nn.Linear(...)` defaulted to CPU regardless of where the rest of the model lived. Invisible on this box (no GPU, so device always "matched" by accident); guaranteed to break the moment a **new** constant name / field name / rep-type is first encountered on a GPU run — which is exactly what the P0 warmup cell does. Fixed all three to create directly on the caller's device/dtype (the pattern already used correctly by `tokenize.py::_kv` and `decoder.py::_get_head`); audited every other lazy-parameter site in `src/noether11/` and found no further instances. Added 4 regression tests (`tests/test_tokenizer.py`) using dtype as the observable axis (no GPU here to reproduce cuda/cpu directly) — verified each fails pre-fix (git-stash roundtrip) and passes post-fix, including one that reproduces the identical `RuntimeError` class. **169 passed** (was 165).

**Follow-up (`6aee92d`) — second real GPU crash + fix (autocast dtype mismatch):** the fixed notebook got past the device bug and hit a new one in the same P0 cell: `RuntimeError: Index put requires the source and destination dtypes match, got Float for the destination and BFloat16 for the source` in `Autoencoder.reconstruct`'s unscatter write. Root cause: under `torch.autocast`, matmul/linear/einsum auto-cast to bf16, but manual tensor writes — advanced-indexing assignment and in-place `scatter_reduce_` — are **not** autocast-managed and require an exact dtype match. Three sites captured their destination buffer's dtype from a tensor read *before* the autocast-affected computation ran, so buffer and values could diverge under AMP: `autoencoder.py::reconstruct` (the one that crashed), `decoder.py::_decode_field` (identical pattern, would have crashed next in the first P1 decode), and `backbone.py::TypedAttention.forward`'s `scatter_reduce_` (would have crashed in the first P1 backbone pass — `dtype = h_src.dtype` captured before the autocast-eligible einsum projections run). Fixed all three: derive the buffer's dtype from the *values being written*, not an earlier-captured dtype — correct under any autocast behavior and a no-op when dtypes already match. Audited every other manual tensor write in `core/`/`condition/`; no further instances. Added 3 regression tests — no GPU needed: `torch.autocast(device_type='cpu', dtype=torch.bfloat16)` + advanced indexing reproduces the *identical* RuntimeError deterministically on CPU (verified via direct repro before writing the tests); each new test confirmed to fail pre-fix (git-stash roundtrip) and pass post-fix. **172 passed** (was 169).

**Follow-up — the P0 quick-test cell in `noether11_colab.ipynb` ran successfully on Colab GPU** (past both prior crashes). Next ask: a dedicated, pre-generated **full train+val corpus** for all three implemented rungs, rather than generating on the fly (right call here: the diversity design needs a *fixed* held-out val set across resumes, and the solvers are pure NumPy/CPU-only — regenerating every epoch would just compete with the GPU-bound training loop for the same CPU across a multi-phase run that revisits the same data thousands of times). Confirmed the solvers have **no GPU path at all** (checked `ns_solver.py`/`rbc_solver.py`/`burgers_solver.py` — pure NumPy, only a brief `torch.from_numpy` round-trip to reuse the FD operators) — a Colab GPU runtime doesn't accelerate generation, only its CPU core count does (via `--workers`).

Prepared Git LFS plumbing (`5a48ed7`: `.gitattributes` tracking `data/**/*.npz`, `.gitignore` restructured from a blanket `/data/` ignore to `/data/*` + per-level negation so `data/{burgers,ns,rbc}/{train,val}/` are committed while smoke/scratch/rungs-4-6 stay ignored — a real gotcha: a directory-level ignore blocks any later negation from reaching inside it). Wrote a self-contained prompt for a new Colab notebook (`notebooks/noether11_generate_corpus_colab.ipynb` + its builder `build_noether11_generate_corpus.py`) to generate the full corpus (burgers 64/16, ns 32/8, rbc 48/8 — the real `SPLITS` presets, fully reproducible via fixed seeds) and push it. The notebook that got built (`2a0a45a`) independently decided to commit the corpus as **plain Git blobs, not LFS** (every file is well under GitHub's 100MB per-file limit, so LFS added quota exposure — 1GB/month free tier, ~575MB corpus — with no benefit) — the user confirmed this is the preferred approach, so **LFS was fully reverted** (`9d5ac2d`): removed `.gitattributes`, restored the `.gitignore` negation comment to not reference LFS, fixed a stale "pushes via Git LFS" docstring in the builder script that contradicted the rest of the same file, ran `git lfs uninstall` locally. Verified: `git check-ignore` still correctly un-ignores train/val and ignores smoke/scratch (the negation logic itself is storage-mechanism-independent); no remaining LFS references anywhere except explicit "no LFS" statements; 172 passed (git-config-only change, suite unaffected as expected).

Also fixed a `notebooks/noether11_generate_corpus_colab.ipynb` setup-cell bug (`2a5fe57`), reported without a traceback: hardened the `import noether11` cell against a stale/partially-initialized `sys.modules` entry from an earlier failed run in the same kernel (a classic Jupyter footgun — re-running a cell after a mid-import exception returns the cached broken module instead of re-importing, surfacing as a confusing downstream `ImportError` unrelated to the real cause) — purge `noether11*` from `sys.modules` before importing, and assert `noether11.__file__` resolves into the checkout's `src/` rather than a stray shadow.

**Next real step:** run the corpus-generation notebook on Colab, verify the pushed corpus loads via `noether11.data.load(system, split, root="data")`, then a cloud-GPU P0→P1 run on the real data. Before P2: wire scorer-persistence + `_compute_pushforward`.

---

## [2026-07-14] portions 8-9 | done (training path) + P0→P4 curriculum linked — Phase III review, P3 completion, P4 wiring

Picked up a repo where a prior agent had **drafted** portions 8 (conservation), 9 (diffusion), 10 (eval) plus a per-phase GitHub-checkpoint-sync workflow (`scripts/train.py --phase PX`, `scripts/train_phase.sh`, `train/publish.py`) — but the implementation-log still said "planned", and there were uncommitted in-progress fixes. Reviewed P0–P3 + the Phase-III drafts for bugs, **completed P3**, and **wired P4 diffusion end-to-end so the curriculum trains P0→P4** with routine checkpointing, in sections. Full suite green (256 pre-existing + 3 new = 259 passing on `.venv`; the 9 `test_gui` errors are a pre-existing `fastapi`-not-in-`.venv` env issue, unrelated).

**Bugs found + fixed (the "never ran end-to-end" class the log keeps catching):**
- **SFA discovery input was malformed** (`Trainer._build_discovery_input`, loop.py): it appended the context window `[B,n_ctx,D]` and future frames `[B,n_future,D]` as separate list entries and `torch.cat(..., dim=-1)` — but `n_ctx(3) ≠ n_future(4)`, so that cat *errors*, and even when shapes matched it fused the time axis into the feature axis and mixed fields. This P3 code path had never actually run. Fixed to build a real `[B, n_ctx+n_future, D_obs]` trajectory: context‖future joined on the TIME axis (dim 1), fields concatenated on the feature axis. (The slowness loss reads consecutive frames along dim 1, so this is a correctness fix, not cosmetic.)
- **P3 whitening was dormant.** `ConservationDiscovery.fit_whitening` existed (the design's mandated "condition the search" correctness step) but the Trainer never called it, so `_whiten_fitted` stayed False and `forward()` skipped whitening entirely. Now fitted once at discovery lazy-init on the first batch, **guarded by `cfg.whiten_dim_cap` (default 8192)** — engage the dense `[D,D]` eigendecomposition only when the flattened obs dim is tractable, else skip gracefully (gradient-SFA still trains, unwhitened) so a 128×64 grid (D≈16k) can't OOM.
- **Interpretability heuristics read the wrong matrix.** `Conservation.project` and `integrate_with_decoder` inspected the raw projector `W` for mass-like near-constant rows, but with whitening on, the physical spatial mask is `W @ M_whiten`. Added `ConservationDiscovery.effective_weights()` (= `W @ M_whiten` when fitted, else `W`) and routed both heuristics through it.
- **P3/P4 resume silently dropped weights.** `_warmup_lazy_modules` didn't rebuild the lazily-created `conservation_discovery`/`diffusion`, so `_load_checkpoint`'s `is not None` guard skipped loading their state dicts on a mid-phase resume (same bug class already fixed for the edge scorer). Now the phase-walk warmup builds discovery (phase≥3) and diffusion (phase≥4) before load. Verified by a checkpoint→resume round-trip test (weights restored to <1e-6).

**P4 wired end-to-end (was a pure stub — `group_d_diffusion` returned 0.0, no `ResidualDiffusion` ever instantiated by the Trainer):**
- Trainer lazy-builds `ResidualDiffusion` on the first P4 step (like the discovery module), keyed on the runtime flattened field dim `d_field`; params join a dedicated AdamW group (normal weight decay); saved/loaded in the checkpoint.
- `train_step` phase≥4 computes the Group-D ε-prediction loss on the **detached** deterministic residual `r = target − v̄`, conditioned on the evolved backbone tokens (per-field streams + `z_cond`) and — crucially — everything fed to the denoiser is `.detach().float()` so (a) the denoiser's gradient never flows into the deterministic stack (phase-separated training — deterministic stack is lightly tuned only by the still-active Group A/B/C) and (b) no bf16/fp32 autocast dtype mismatch (the loss runs outside the autocast block). New `Trainer._forward_with_tokens` exposes the evolved tokens; `_forward_step` is now a thin wrapper so every existing caller is unchanged. Weight `cfg.p4_diffusion_weight` (default 1.0).
- **DiT zero-init** added to the denoiser (final `output_proj` layer + each block's `ada_ln_mod`): the denoiser starts predicting ε≈0, so the P4 loss starts at the ideal ‖ε‖²≈1 instead of a ~3000 random spike — a much more stable optimization start (verified: 3041 → 1.07 at init).
- Also fixed (in the prior uncommitted work, now confirmed correct) a real **DDIM sampler bug** (the update had dropped the directional-ε term entirely: `r = c2·r̂₀ + c1·ε_noise`, missing `+ √(1−ᾱ_next−σ²)·ε_θ`), and the naive-timestep-skip "fast" DDPM path now delegates to the proper DDIM sampler.

**Curriculum P0→P4 (verified end-to-end on real plumbing, not hand-built tensors):** a scratchpad harness drove the actual `Trainer.train_step` through P0→P1→P2→P3→P4 on a tiny NS corpus: every phase trains, P3 conservation (Group C) + whitening engage, P4 diffusion (Group D) builds and its denoiser receives gradient (32/32 params), and a checkpoint→resume at P4 restores both P3 and P4 modules bit-exactly. The existing `--phase PX` sectioned workflow (`train_phase.sh` P0..P4, GitHub-release seed/publish per phase) now covers P4 unchanged: each phase's release contains exactly the lazy modules the next phase's warmup rebuilds, so the chain `P0→P1→P2→P3→P4` is resume-consistent across ephemeral (vast.ai) boxes. Config knobs added: `p3_drift_threshold`, `whiten_dim_cap`, `p4_diffusion_weight`.

**Remaining Phase-III item (portion 10, honestly flagged):** the eval-time **ablation toggles** `use_projection` / `use_diffusion` in `eval/rollout.py::rollout_with_toggles` are still enumerated-but-ignored (only `use_learned_edges` is actually applied), and the public `eval/inference.py::rollout` doesn't sample the diffusion residual or apply the hard projection at rollout time. So the M4/M5 *contrast* evidence (blurred-vs-sharp with diffusion, drift-with-vs-without projection) isn't runnable yet — a distinct follow-up from the training path completed here. The training-time machinery it needs (a trained denoiser, a promoted-hard `StrengthState`) now exists.

**Design-vs-impl deviations noted (not bugs, flagged for honesty):** (1) the diffusion denoiser diffuses the **full flattened field**, not the band-limited high-k residual the design specifies — so it's tied to the training resolution (fine for fixed-res benchmarks, not resolution-free like the rest of the model); (2) its DiT "self-attention" runs over a length-1 sequence (effectively an AdaLN-MLP), since the whole field is one token — adequate at smoke/medium scale, a candidate upgrade to per-patch tokens later; (3) whitening is fit on a single batch (design says "a representative batch") — accumulating covariance over several batches would sharpen it. (4) The P3→P4 gate (`conservation_drift < p3_drift_threshold`) must actually pass for the sectioned chain to reach P4 — same operational model as the P0→P2 gates today; the threshold is now a config knob.

## [2026-07-14] training-harness | fixed — phase-boundary checkpoint was pruned, breaking `--phase P{n+1}`

First real multi-phase run on vast.ai surfaced a sectioned-workflow bug: after P1 finished (advanced to P2), `--phase P2` errored "starting at P1: earlier phase(s) not complete." Root cause: the phase's **final** checkpoint `ckpt_final_P{new}.pt` (written at the NEW phase = the seed the next phase resumes from) is saved with metric=inf, and the `keep_top_k` pruning deleted it as the "worst" entry immediately after writing it — before it was pushed to the release. The only surviving local checkpoints were previous-phase periodics, and the phase-blind newest-by-mtime resume picked one. **Fix (`99900ae`):** `_save_checkpoint(prune=False)` for the final save (never rotated out, so it also gets pushed); `_newest_local`/`_resolve_resume` are phase-aware (`--phase PX` only considers `ckpt_*_P{X}.pt`). +2 regression tests (final survives pruning; simulated completed-P1 run resumes P2 without tripping the guard). Recovery for a checkpoint made by the older code: re-run the previous phase once to regenerate the seed (documented in [[vast-ai-training-runbook]] §7).

## [2026-07-15] tokenizer / decoder / training-harness | done — fineness fix: token-density default + multi-scale jitter + inference override

Diagnosed the "pixelated" rollout: at `patch_domain_size=0.25` the 128×64 / [0,2]×[0,1] corpus tokenizes to only **8×4=32 patches**, and the decoder's one-token-per-patch broadcast renders them as near-constant blocks. Confirmed on NS val with `scripts/diagnose_token_density.py` (new): all four densities run on the **same P2 weights** (token count is a runtime dim, not a weight shape → no from-scratch restart), but density alone doesn't improve accuracy (rel-L² flat ~1.28, SIREN rings at out-of-scale offsets). **Changes (noether-1.1 branch):** (1) `configs` — base `patch_domain_size` 0.25→0.0625, medium 0.0625 (512 tokens), large 0.03125 (2048); new `patch_size_jitter` field. (2) `train/loop.py` — `_apply_patch_jitter()` per step (multi-scale training), `validate()` pinned to base density. (3) `eval/inference.py::load_checkpoint(patch_domain_size=…)` + GUI `/api/rollout` `patch_domain_size` field for A/B without retraining. All 109 tests pass; new jitter smoke + density diagnostic green. **Design page:** [[noether-1.1-fineness-and-token-density]]. **Unblocks:** a warm-start fine-tune at the finer density (re-walk P0→P2) to remove the pixelation *and* push rel-L² below persistence.

## [2026-07-15] edge-generation / training-harness | fixed — P2 CUDA OOM from unbounded learned-edge candidates; memory requirements documented

The density fix (32→512 tokens) surfaced a latent `O(P²)` blowup: `learned_edge_cutoff` defaulted to **None = score all pairs**, which [[edge-generation-1.1]]'s cost note called "trivial at node scale" — true at 32 nodes (496 pairs), false at 512 (130 816) and fatal at 2048 (2 096 128 ≈ **19.5 GB** for one activation). Because the learned scorer only comes online at **P2**, P0/P1 passed and the box died later — the "ran a while, then OOM" signature. **Fix:** new `learned_edge_cutoff_patches` (default 8.0) auto-derives the envelope $R = 8s \approx 4r$ from the patch size in `__post_init__`, so candidates-per-node is **density-invariant** (O(P), not O(P²)). Measured: medium 130 816→39 616 (0.62→0.19 GB), large 2 096 128→179 584 (19.5→1.67 GB); ~155/175 candidates per node across the two densities. Also: `batch_size` re-tuned for the new density (medium 48→16, large 24→8), new `--batch-size` / `--patch-domain-size` CLI overrides as in-the-field OOM knobs, and `keep_top_k` / `checkpoint_interval` / `keep_phase_finals` promoted from `getattr` shims to real config fields. **Disk:** phase-final checkpoints are saved `prune=False` (they seed the next phase) and so were the one class the top-K rotation never touched — they accumulated at ~450 MB each; `_sweep_stale_phase_finals()` now removes finals older than `keep_phase_finals`. **Runbook:** [[vast-ai-training-runbook]] gains §2b (min VRAM/disk/RAM per config) and §2c (ordered OOM knobs + why P2 specifically). 261 tests pass. **Checkpoints cleared for the full retrain:** `ckpt-ns-P1`/`ckpt-ns-P2` releases retagged `archive-ns-P{1,2}-pre-density-fix` (un-pullable by `train_phase.sh`, kept for comparison); local checkpoint moved to `checkpoints/_archive_pre_density_fix/`.

## [2026-07-19] training-harness | fixed — periodic checkpoint deleted itself on save (P1 "does not exist; skipping upload")

Live P1 run hit: `checkpoint saved to ckpt_step0008000_P1.pt` immediately followed by `[publish] WARNING: ... does not exist; skipping upload`. Root cause in `_save_checkpoint`'s keep-top-K rotation (`train/loop.py`): the periodic call site passed `result`, the raw `train_step()` LOSS dict, which has no `rel_l2`/`recon_error` key — so `metrics.get("rel_l2", metrics.get("recon_error", float('inf')))` **always** fell back to `+inf`. With `keep_top_k` (3) periodic saves already on disk at equal (+inf) metric, Python's *stable* sort left the just-appended (newest) entry last among the ties, so the very next `pop(-1)` deleted the file the SAME call had just written — happening on every periodic save from the 4th one onward. Not caused by the density/OOM work; pre-existing, just never triggered before because `checkpoint_interval` (1000) rarely got exercised long enough. **Fix, two parts:** (1) `run()` now tracks `self._last_val_metrics` (the last real `validate()` output, which DOES carry `rel_l2`/`recon_error`) and passes that to periodic saves instead of `result`. (2) `_save_checkpoint`'s prune sort key changed to `(metric, -step)` so that even under a remaining all-tied-inf run (e.g. before the first validation), the *newest* entry sorts first and survives — the oldest is evicted, never the one just written. **Regression test** `test_periodic_pruning_never_deletes_the_checkpoint_just_saved` added; confirmed it fails at exactly step 4000 (the 4th periodic save, `keep_top_k=3`) on the pre-fix code and passes after. The pre-existing `test_final_checkpoint_survives_pruning` had used `metrics={}` (always inf) but only asserted survivor *count*, never *identity* — which is why this shipped unnoticed. 262 tests pass.

## [2026-07-19] edge-generation / backbone / training-harness | optimized — ~2.9x (P1) / ~4.9x (P2) step time, bit-for-bit identical output

Profiled a real train step after a live run showed ~0.58 s/step on an A100 (25 epochs / 4300 s at batch 16 = 296 steps/epoch). The model maths was **not** the bottleneck — Python loops with per-edge `.item()` calls were, each one a host-device sync that stalls the GPU:

1. **`_apply_sparsity_budget`** — allocated a dense `[P, E]` vote matrix (20 M bools at medium, 368 MB at large), looped over all P nodes, then looped over all E edges with **two `.item()` calls each** (~80 k syncs per call, 3 calls/step). Rewritten fully vectorized via a stable sort + per-node rank + `scatter_add`. Verified **exactly equal** to the old implementation across a parameter sweep, with the per-node degree ≤ budget invariant preserved. 36–92× faster on CPU alone.
2. **Duplicate scorer pass** — `build_graph` ran the learned-edge MLP over every candidate pair *twice* (once inside `score_and_sample`, once for the budget). Now scored once and reused: halves both compute and the retained autograd graph on P2's largest tensor.
3. **Static geometry recomputed every forward** — the KNN families and the learned-edge candidate set (plus their Fourier edge attributes) depend only on patch centroids, yet were rebuilt 3×/step, each via Python loops with `.item()` (8.7 k syncs in `knn_same_field` alone). Now memoized on the `PatchLayout` (`_geom_cache`), which `Tokenizer._layout_cache` already keys per density, so `patch_size_jitter` gets one cache per rung and `layout.to(device)` gets a fresh one.
4. **Family-1/2 dedup** — another `.item()` loop over all KNN + learned edges; replaced with canonical `min*P+max` pair keys and `torch.isin`, with the static key set cached.
5. **Backbone same-field projections** — `tau=0` passes the *same* stream as both endpoints, so `q_dst`/`v_dst` duplicated `q_src`/`v_src`. Skipped when `h_src is h_dst`. For a single-field system (NS) `tau=0` is the only edge type that fires, so this halves the backbone's projection einsums outright.
6. **DataLoader** — was `num_workers=0` (batch assembly serialized with the GPU step); now `dataloader_workers` (default 4, CUDA-only) with `pin_memory`, `persistent_workers`, `prefetch_factor=4`, and `non_blocking` H2D copies.

**Measured (CPU, batch 4):** P1 8114 → 2798 ms/step (**2.9×**), P2 21092 → 4293 ms/step (**4.9×**). `build_graph` and `_apply_sparsity_budget` no longer appear in the profile's top entries. The GPU gain should be at least as large, since eliminated syncs cost proportionally *more* there. **Correctness:** an end-to-end P2 fingerprint (edge_index, edge_type, edge_attr, reg loss, encoded tokens, evolved tokens, decoded field) is **bit-for-bit identical** to pre-optimization HEAD — these are pure speedups, not approximations.

**Bug found and fixed while profiling (introduced by the 2026-07-15 density work):** `learned_edge_cutoff` was derived once from the *base* `patch_domain_size`, but `patch_size_jitter` changes density per step. On the 0.5× rung a base-derived R covers 4× the area → **642 688 candidates vs 39 616** (16× spike), i.e. an OOM on ~1/3 of steps. R is now derived from the layout's live patch size (`effective_edge_cutoff`, unit-tested). Relatedly, `patch_size_jitter` defaults changed to **(1.0, 1.5, 2.0)** — all multipliers ≥ 1 so the configured `patch_domain_size` is the *finest* rung and peak memory equals the documented per-config VRAM figure, instead of a sub-1 multiplier silently imposing the `large` token load on a `medium` budget. 269 tests pass (7 added: sparsity-budget-vs-reference-oracle sweep, empty-edge case, envelope density scaling, cutoff derivation + fallback).

## [2026-07-19] backbone / losses / tokenizer | optimized (round 2) — hidden CUDA syncs + remaining static recomputation

Second optimization pass, triaged against a fresh profile rather than a checklist. After round 1 the CPU profile is dominated by *real* compute (`run_backward` 56%, backbone 11%, `linear` 9%, `scatter_reduce_` 5%), so the remaining wins are GPU-specific things a CPU profile cannot see — namely **host-device syncs**, which stall the pipeline regardless of how cheap the op looks.

1. **`_typed_messages` re-split the graph every layer.** `if not type_mask.any()` ran per edge type per layer — on CUDA each `.any()` is a sync, so **48 per forward** at $L{=}12,T{=}4$ (144/step at P2), plus 48 redundant boolean masks and advanced-index copies. But `graph.edge_type` is constant across layers: the per-type slices are now computed **once per graph** (`_type_slices`, memoized on the `Graph`, which is rebuilt each forward so it can't go stale), reading `bincount(...).tolist()` once instead of T separate `.any()` calls, and returning `None` for empty types so the caller skips without touching the device.
2. **`gamma(x_p - x_i)` recomputed every forward**, in *both* the tokenizer and the decoder (so 6× per P2 step at $[P,W,\text{gdim}]\approx200$k elements) despite being fixed geometry. Now `PatchLayout.gamma_offsets(bands)`, memoized per band count. Relatedly `PatchLayout.to(device)` is now identity when already on-device — callers passing an explicit layout were otherwise rebuilding it (and discarding its geometry/gamma caches) on every forward.
3. **Spectral loss rebuilt its $1/(|k|+1)$ weight grid** for every channel of every field on every step. Cached per (shape, ndim, device, dtype); verified **bit-exact** against the original formula.
4. **`--compile`** added as an explicit opt-in (`torch.compile(backbone, dynamic=True)`), *off by default and flagged experimental* — this graph has data-dependent shapes (edge counts vary per step at P2, token count varies with `patch_size_jitter`), so recompilation can outweigh the gain. It must be measured on the box, not assumed.

**Measured (CPU, batch 4):** P1 2798 → 2669 ms, P2 4293 → 3900 ms. Cumulative from the pre-optimization baseline: **P1 8114 → 2669 ms (3.0×), P2 21092 → 3900 ms (5.4×)**. The sync fixes should show up larger on CUDA than these CPU numbers suggest, since a sync costs far more there. Forward-path fingerprint remains **bit-for-bit identical** to pre-optimization HEAD; 269 tests pass.

**Suggestions evaluated and deliberately NOT taken** (an external review proposed them; the profile does not support them):
- *"Hand-written message passing → PyG/torch_scatter, 2–5×"* and *"lack of sparse kernels, 2–4×"* — `scatter_reduce_` is **5%** of step time. Even eliminating it entirely gives <1.06×, nowhere near 2–5×, and it would add a heavy CUDA-version-coupled dependency to an ephemeral-box setup. Rejected on measurement.
- *"Small graph (512 nodes) underutilizes the A100"* — the per-edge tensor is $[B,H,E,d_h] = [16,6,8192,64] \approx 50$M elements; these are large kernels, not launch-starved. Larger batch still helps for a *different* reason: `build_graph` scores one graph per batch (it uses `streams[name][0]`), so that cost is batch-independent and amortizes. **`--batch-size 32` is worth trying on a 40 GB+ card**; the default stays 16 so the documented 24 GB floor holds.
- *"Move everything to GPU before training"* — the corpus is ~396 MB and would fit, but H2D is only ~8 MB/batch (<1 ms against a ~580 ms step) and is already overlapped by `pin_memory` + workers + `non_blocking`. Negligible.
- *"Attention computed separately for each edge type"* — for a single-field system (NS) only $\tau{=}0$ has edges, so there is nothing to fuse; revisit for RBC/MHD where $F>1$.
- *CUDA graphs* — incompatible with the per-step-varying edge counts.

## [2026-07-19] repo-scaffold | fixed — checkpoints/ was never actually gitignored

`.gitignore`'s `/checkpoints/` line carried a trailing inline comment. gitignore has no inline-comment syntax (only a line whose first character is `#` is a comment), so the whole line — pattern plus comment text — was one literal, never-matching string; `git check-ignore` confirmed `checkpoints/ckpt_latest.infer.pt` was **not ignored**. No `.pt` was ever actually committed (checked `git log --all --diff-filter=A -- '*.pt'`), so this was a latent risk rather than realized damage — but a real one: a stray `git add -A` would have swept multi-hundred-MB checkpoint binaries (one local file is 157MB, over GitHub's 100MB per-file push limit) into git history, colliding with the documented "checkpoints live in GitHub Releases, never git commits" design. Fixed by moving the comment to its own line and adding a global `*.pt` pattern as a second layer. Verified: `git check-ignore` now returns ignored for real/archived/hypothetical-nested checkpoint paths, and a literal `git add -A` at the repo root stages only `.gitignore` itself. Separately confirmed `train/publish.py` (the checkpoint-publish path) never calls `git push`/`git tag` — it is exclusively `gh release` — so the two channels were already structurally independent; this closes the one way they could have collided. 269 tests pass. See [[vast-ai-training-runbook]] §2b for the checkpoint-disk-hygiene context this sits alongside.

## [2026-07-19] training-harness | fixed — training wedged at `[publish] uploading to existing release …`

A live P1 run stopped dead after that log line. Root cause in `train/publish.py`: `_run_gh` called `subprocess.run(..., capture_output=True, check=True)` with **no `timeout` and no `stdin` redirect**. A stalled upload (flaky box upstream, half-open TCP) therefore blocked *forever*, and because output was captured the operator saw the "uploading…" line and nothing after — training silently dead with no diagnostic. A `gh` blocking on an interactive prompt (expired credentials, confirmation) had the same effect. Compounding it, the ~490 MB upload (335 MB full + 155 MB inference) ran **synchronously inside the training loop** on every checkpoint interval, so even a healthy-but-slow link stalled training for minutes each time.

**Fix, three parts:** (1) every `gh` invocation is now time-bounded (`UPLOAD_TIMEOUT`, default 1800 s via `NOETHER_UPLOAD_TIMEOUT`; short `QUERY_TIMEOUT` for existence checks) and runs with `stdin=DEVNULL` so it fails loudly instead of waiting on a terminal that never answers. (2) **Periodic** uploads moved to a background thread — staging (the local copy + optimizer strip) stays synchronous so the checkpoint pruner can't unlink the source mid-transfer, and only the network call is backgrounded; one upload in flight at a time, with a second one **skipped rather than queued** since the next checkpoint supersedes it. (3) The **phase-end** publish stays synchronous and first drains any in-flight background upload, because it is the seed the next phase's command pulls; `run()` also drains before returning so a caller exiting immediately can't kill a live transfer. Upload logs now report size, elapsed time and MB/s, so slow-vs-hung is distinguishable at a glance.

3 regression tests added (timeout+stdin passed on every `gh` call; a hanging `gh` degrades to `False` instead of blocking; background push returns immediately and coalesces concurrent pushes) — all three verified to fail against the pre-fix module. 272 tests pass. Operator guidance in [[vast-ai-training-runbook]] §2d, including `--no-publish-full` / `--publish-every N` for genuinely slow links.

## [2026-07-19] training-harness | fixed — resume-only CUDA OOM: allocator fragmentation from per-step jitter

A P2 run that trained fine before a pause OOM'd on epoch 2 after resuming — a distinct signature from the earlier (already-fixed) P2 edge-envelope OOM: `36.36 GiB allocated` + `2.49 GiB reserved by PyTorch but unallocated` out of 39.49 GiB, i.e. the allocator itself flagging fragmentation, and the error's own suggested fix (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) is PyTorch's documented mitigation for exactly that pattern.

**Root cause:** `_apply_patch_jitter()` (added 2026-07-15 for the fineness fix) redrew the token density **every single train_step()**, cycling `patch_domain_size` between the three `patch_size_jitter` multipliers. Token count sets the shape of nearly every tensor in the model — backbone attention `[B,H,E,d_h]`, decoder per-point tensors `[B,P,W,C]`, edge candidate tensors — so almost every step allocated a different set of shapes. PyTorch's caching allocator groups free blocks by size class; blocks freed for one shape can't satisfy a request for a different one, so "reserved but unallocated" memory grows monotonically until an otherwise-tiny allocation fails. The checkpoint this session resumed was trained *before* the jitter default landed (fixed shape throughout), so the shape churn — and the fragmentation it causes — was new to this resumed session, not something the original training run had ever exercised.

**Fix, three parts:**
1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** set via `os.environ.setdefault(...)` at the top of `scripts/train.py`, *before* `import torch` (the allocator reads this at first CUDA use; setting it after import is too late) — and redundantly `export`ed in `scripts/train_phase.sh` for direct invocations. Verified: applies when unset, and an operator's explicit override survives untouched.
2. **New `patch_jitter_dwell_steps`** (default 25): `_apply_patch_jitter` now holds one density for N consecutive steps before redrawing, instead of every step — bounding shape churn while still exercising every density many times per epoch. Caught a real bug while implementing this: the dwell state can't be inferred from `tokenizer.patch_domain_size`'s live value, because `validate()` legitimately overwrites that same field to the base density between train_step() calls (correctly — the curriculum gate must be measured at deployment density) — inferring "still dwelling" from it would silently truncate a dwell window to one step whenever a validation check landed inside it. Fixed by tracking the current multiplier in its own field (`_patch_dwell_mult`), independent of what `validate()` does to the tokenizer. Regression test confirms this fails on the naive (infer-from-live-field) version and passes on the explicit-tracking one.
3. **`torch.cuda.empty_cache()` after `_load_checkpoint`** (CUDA only): the resumed checkpoint's full state dict is deserialized directly onto the training device (`map_location=trainer.device`), so for the load's duration a full extra copy of every model's weights briefly coexists with the trainer's own — transient, but the allocator doesn't necessarily hand that memory back to a defragmented pool on its own. Dropping the reference and clearing the cache means a resumed run starts real training from a clean slate instead of carrying the load's fragmentation forward to compound with jitter's.

3 regression tests added (dwell windows held constant; dwell survives an interleaved `validate()` call — verified to fail on the naive implementation; jitter-disabled path unaffected). Full end-to-end fresh→resume CLI smoke run passes. 275 tests pass. Runbook guidance in [[vast-ai-training-runbook]] §2c (a new sub-section, distinguishing this fragmentation signature from the earlier envelope-OOM one by the "reserved but unallocated" tell).

## [2026-07-19] training-harness | fixed — third CUDA OOM signature: genuine occupancy from push-forward horizon, not fragmentation

The fragmentation fix (dwell + `expandable_segments`) shipped earlier today did not fully resolve the resume-time OOM — it recurred with a **different** error signature: `38.78 GiB allocated` + only `112.53 MiB reserved but unallocated` out of 39.49 GiB. Unlike the earlier report (a 2.49 GiB fragmentation gap), this is nearly the whole card genuinely occupied by live tensors — not fragmentation, a real capacity ceiling. Traced via a full stack trace into `TypedAttention.forward`'s `weighted_to_src = ...` line, inside one backbone layer's one edge-type pass.

**Root cause:** `_run_pushforward`'s `H` unroll substeps feed forward autoregressively but *detached* (`cur = {k: pred[k].detach() ...}`, no backprop through the rollout chain — correct, avoids vanishing/exploding gradients over a long rollout). But each substep's own forward still contributes to **one accumulated `loss`** (`loss = loss + masked_mse(...)`) that calls `.backward()` exactly once, at the end. Autograd therefore has to retain every substep's full activations (tokenize→build_graph→backbone→decode, all `L` layers) simultaneously until that single backward pass — memory scales with `H`, not just with batch/density. `pushforward_horizons=(1,2,4,8)` means the curriculum reaches `H=8` once training has run "for a while" (single-step loss has plateaued three times) — exactly the "trained fine, OOM later" pattern, now doubly so combined with `patch_size_jitter`'s dwell mechanism sustaining the priciest token density for many consecutive steps. Confirmed no gradient-checkpointing existed anywhere in the codebase, despite [[noether-1.1-large]] having anticipated exactly this lever ("gradient checkpointing on the backbone to fit memory") for the larger config — it had never actually been implemented.

**Fix:** new `pushforward_grad_checkpoint` config flag (default **on**). `_run_pushforward` now wraps each substep's `self._forward_step(...)` call in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` whenever `H>1` and gradients are being tracked (skipped at `H=1` — nothing to save across a single substep — and skipped under `validate()`'s `@torch.no_grad()` rollout-metric call, where there is no retained graph to trade against, so checkpointing would only add its recompute cost for zero benefit). Verified the checkpoint API handles the actual call shape (dict-valued positional args, a plain-float arg, a dict-valued keyword arg, dict-valued return) before wiring it in, then verified the wired version is **bit-exact** — same loss, same gradients — against the uncheckpointed path at P2, deliberately including the learned-edge scorer's stochastic Gumbel-softmax sampling (checkpoint's automatic RNG save/restore around the recomputed forward must reproduce identical draws, and does). A live run at P2/`H=8` on real NS data (the exact stress condition) completes cleanly.

2 regression tests added: checkpointed vs uncheckpointed equivalence at P2 (loss + gradients, bit-exact); checkpoint correctly skipped at `H=1` and under `no_grad` (verified to fail when that guard is removed). 277 tests pass. Documented as a third, distinct OOM signature in [[vast-ai-training-runbook]] §2c, alongside the two earlier ones (unbounded edge candidates; jitter-driven fragmentation) — distinguished by the size of the "reserved but unallocated" gap in the error message.

---

*Log format: `## [YYYY-MM-DD] <portion> | <status> — <what happened>*`
