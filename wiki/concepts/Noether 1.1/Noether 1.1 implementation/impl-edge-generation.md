# Portion 4 — Edge generation

**Type:** Implementation spec — agent task (folder: Noether 1.1 / Noether 1.1 implementation)
**Portion of:** [[00-implementation-plan]] (Phase II). **Depends on:** [[impl-tokenizer-descriptors]]. **Unblocks:** [[impl-backbone]].
**Design pages:** [[edge-generation-1.1]]. The learned scorer trains in **P2**, *not before* ([[training-scheme-1.1]]).

---

## Objective
Build the typed multigraph the backbone runs on: three edge families — local same-field KNN, local cross-field KNN (smaller K′), and a **learned long-range** family from a small scorer — added on top of each other, each edge tagged by its (field, field) type.

## Deliverables (`noether11/core/edges.py`)
- `knn_same_field(tokens, K)` and `knn_cross_field(tokens, K')` — geometric, parameter-free, BC-aware (periodic/wall), **symmetric** (score each unordered pair once, so the antisymmetric-flux momentum channel is well-defined).
- `LearnedEdgeScorer` (`g_θ`): MLP over `(t_i^f, t_j^f', γ(x_j−x_i), ‖x_j−x_i‖, z_cond)` → logit `s_ij`; `GumbelSigmoid(s_ij, τ_gs)` with straight-through estimator; kept edges added **on top of** families 1–2, scored once per unordered pair, sparsity budget / entropy regularizer.
- `build_graph(tokens, cfg, use_learned: bool)` → typed edge index + edge features + edge-type ids for the 4 types.
- A **gating hook**: `use_learned=False` until P2 (the harness flips it on).

## Interface contract
- `build_graph(...) -> Graph` with `edge_index [2,E]`, `edge_type [E]` ∈ {same, cross_coloc, cross_spatial, learned}, `edge_attr` (γ(Δx), ‖Δx‖).
- Undirected guarantee: each unordered pair appears with a single canonical orientation (backbone symmetrizes) — **required** or the flux cancellation breaks ([[edge-generation-1.1]] risk).
- Cost: scorer is $O(N^2)$ in **nodes** (~$10^2$–$10^3$), trivial — full pairwise candidate scoring within envelope $R$ is fine.

## Build steps
1. KNN families (reuse `torch_cluster` or a batched cdist top-K); enforce BC and symmetry.
2. `LearnedEdgeScorer` + GumbelSigmoid + straight-through; temperature anneal schedule (from cfg).
3. Sparsity budget (top-k or expected-degree penalty) + small edge-entropy regularizer.
4. `build_graph` assembling all families with type ids; the `use_learned` gate.
5. Debug endpoint/util to export the learned edges for the GUI overlay ([[impl-gui-2d]]).

## Acceptance tests
- Families 1–2 are deterministic and symmetric (unit test on a toy point set).
- With `use_learned=True` on trained weights, the learned graph is **sparse** (within budget) and **non-collapsed** (not all-on/all-off) — milestone **M3**.
- Ablation hook: `use_learned=False` reproduces a pure-KNN baseline (for the "do learned edges help?" ablation).
- Range-constant control: a smaller Debye/cutoff constant yields a smaller KNN radius with no code change ([[conditioning-and-constants-1.1]]).

## Pitfalls
- **Do not train the scorer before P2.** Under a single-step (near-identity-friendly) loss it learns "the topology of doing nothing" ([[edge-generation-1.1]] risk; [[training-scheme-1.1]]). The harness owns the gate; this module just exposes it.
- Straight-through: hard sample forward, soft gradient backward; anneal τ_gs down so inference gets a crisp graph.
- Keep learned edges **undirected** — directed ones break momentum-flux cancellation.

## See Also
- [[impl-backbone]] (consumes the typed multigraph) · [[edge-generation-1.1]] (authoritative) · [[impl-training-harness]] (owns the P2 gate)
