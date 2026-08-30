# Intelligent Patching (adaptive-resolution tokenization)

**Type:** Concept — initial model component (folder: Noether 1.0)
**Status:** Tokenizer refinement for continuous fields — how to place patches/nodes so detail-rich regions get finer resolution. Plugs into the "patch partition" step of [[graph-tokenizer]].
**Related Concepts:** [[graph-tokenizer]], [[initial-model-architecture]], [[adaptive-compute-tokens]], [[hierarchical-windowed-tokens]], [[multiscale-hierarchical-gnn]], [[multihead-attention]]
**Related Summaries:** [[walrus-paper]], [[gns-graph-network-simulators]], [[multipole-graph-neural-operator]]

---

## The problem

Uniform patching spends the token budget evenly across the domain — but physics is *not* evenly hard. A shock, a boundary layer, a vortex core, or a flame front carries almost all the dynamically important structure in a tiny fraction of the area; the rest is smooth. Uniform patches therefore **waste tokens on smooth regions and under-resolve sharp features** — and under-resolved sharp features are exactly where rollout error is born ([[autoregressive-rollout-stability]]).

Classical CFD solved this decades ago with **Adaptive Mesh Refinement (AMR)**: refine the grid where a monitor function says the solution is hard, coarsen where it is easy. Intelligent patching is AMR for tokenization — **put small patches (dense nodes) where there is detail, large patches (sparse nodes) where there is not.**

---

## Mechanism

### Refinement indicator (where to refine)
A scalar **monitor function** $\eta(x)$ flags detail. Standard choices, all cheap:
$$\eta(x) = \big\|\nabla u(x)\big\|\quad(\text{gradients/shocks}),\qquad \eta(x)=\big\|\omega(x)\big\|\quad(\text{vorticity}),$$
$$\eta(x)=\text{local spectral energy above } k_{\text{cut}}\quad(\text{fine-scale content}),\qquad \eta(x)=\text{learned error / uncertainty}.$$
The learned variant is the most powerful: use the model's own **predicted uncertainty** (e.g. the variance of the diffusion corrector, [[initial-model-architecture]] Stage 7) as the indicator — refine where the model is unsure. This couples the refinement criterion to the model rather than hand-designing it.

### Quadtree / octree subdivision (how to refine)
Start from a coarse uniform partition; recursively split any patch whose integrated indicator exceeds a threshold, to a maximum depth:
$$\text{split}(P)\ \text{if}\ \int_P \eta\,dx > \tau, \qquad \text{depth} \le L_{\max}.$$
The result is a tree of patches of varying size — a quadtree in 2D, octree in 3D — with fine leaves clustered on features and coarse leaves over smooth flow.

### Token-budget control
To keep the token count bounded (and batchable), allocate a target budget $N$ by ranking patches on $\eta$ and refining greedily until $N$ is reached — the spatially-*non-uniform* generalization of [[adaptive-compute-tokens]] (Walrus CSM keeps a uniform stride; this concentrates resolution where $\eta$ is large).

---

## Why it composes cleanly with the graph tokenizer

This is the key reason intelligent patching is *natural* here rather than bolted-on:

1. **Variable patch size → variable node density.** Each leaf patch is one centroid node ([[graph-tokenizer]]). Fine leaves give dense nodes on features; coarse leaves give sparse nodes elsewhere. The graph simply has non-uniform node spacing — which graph representations handle natively ([[graph-mesh-tokens]]), unlike fixed grids.
2. **The resolution-free node encoder absorbs the variable point count.** A small patch may contain 4 sample points, a large one 400; the $N_q$-query compression encoder maps *any* point count to the same fixed-dim latent, so heterogeneous patch sizes still yield uniform node tokens. **Intelligent patching needs resolution-freedom, and the tokenizer already provides it** — the two features are co-dependent. **Complement (S3.1):** adaptive patching decides *where* to spend nodes; the dual-path high-frequency residual channel ([[open-architectural-problems]] S3.1, [[graph-tokenizer]]) preserves the sharp-feature *magnitude within* each patch. Together they cover both axes of fine-scale loss — spatial placement and spectral content.
3. **The refinement tree *is* the supernode hierarchy.** Quadtree/octree levels are exactly the coarse→fine levels of the multiscale graph ([[multiscale-hierarchical-gnn]], [[multipole-graph-neural-operator]]). Building the refinement tree once gives both the adaptive tokenization *and* the global-coupling hierarchy (and the Index Share landmarks). One structure, three uses.

---

## Pros

- **Token efficiency** — resolution where it matters; far fewer tokens for the same effective accuracy on feature-dominated flows (shocks, boundary layers, turbulence).
- **Better sharp-feature accuracy** — fine patches on discontinuities reduce the aliasing that drives rollout error.
- **Native multiscale** — the refinement tree doubles as the global-coupling hierarchy.
- **Proven principle** — AMR is decades-validated in classical CFD; this ports it to the tokenizer.
- **Geometry-friendly** — refine toward curved boundaries; pairs with the graph's irregular-geometry support.

## Cons

- **Dynamic token count over rollout** — features move, so the tree changes each step; variable token counts complicate batching (mitigate with a fixed budget $N$ and padding/masking).
- **Refinement criterion choice** — hand-designed indicators ($|\nabla u|$, $|\omega|$) are simple but heuristic; learned indicators add training complexity.
- **Refinement lag** — a purely reactive indicator refines *after* a feature arrives; fast shocks may outrun it. Needs **predictive refinement** (refine along the indicator's advection direction).
- **Tree-construction cost** — building/maintaining the quadtree and the neighbor graph each step is real overhead ([[graph-mesh-tokens]] irregular-memory con).
- **Cross-level edges** — message passing between a fine leaf and an adjacent coarse leaf needs scale-aware edge features so the size mismatch is represented.

---

## Relationship to other representations

- vs. [[adaptive-compute-tokens]] (Walrus CSM): CSM fixes the token budget with a *uniform* resolution-conditioned stride; intelligent patching makes the resolution *spatially non-uniform*, concentrating it on features. CSM is cheaper and order-preserving; intelligent patching is more token-efficient on feature-dominated flows.
- vs. [[hierarchical-windowed-tokens]] (Swin/scOT): Swin's pyramid is a *fixed* multiscale grid; intelligent patching makes the pyramid *content-adaptive* (deep only where $\eta$ is large).
- vs. [[multiscale-hierarchical-gnn]]: that page is the *message-passing* side of the same tree; this page is the *tokenization* side. They share the hierarchy.

---

## [AI Inference]

**[AI Inference]:** The highest-value version closes a loop between the refinement indicator and the model's own uncertainty: tokenize → predict → read the diffusion-corrector variance → refine where variance is high → re-tokenize. This makes patching *active* (the model decides where to look harder), turning tokenization from a fixed preprocessing step into a learned, closed-loop attention-allocation mechanism — the spatial analog of how the multi-token-prediction head allocates *temporal* effort. It also gives a clean training signal: refine exactly where the model would otherwise make its largest error.

**[AI Inference]:** Because the refinement tree, the supernode hierarchy, and the Index Share landmarks are the *same* structure, intelligent patching is not an extra cost on top of the multiscale machinery — it is the multiscale machinery, *driven by data instead of fixed*. Building the model around a single content-adaptive tree (one learned restriction/prolongation pair, refined by $\eta$) would unify adaptive tokenization, global elliptic coupling, and long-range attention into one mechanism — arguably the single most leverage-heavy design decision in the tokenizer.

---

## See Also

- [[graph-tokenizer]] — the host; patch partition step and resolution-free encoder
- [[initial-model-architecture]] — where patching sits (Stage 2) and the uncertainty-driven loop
- [[adaptive-compute-tokens]] — uniform-stride fixed-budget cousin
- [[hierarchical-windowed-tokens]] — fixed multiscale pyramid
- [[multiscale-hierarchical-gnn]] / [[multipole-graph-neural-operator]] — the message-passing side of the refinement tree
- [[graph-mesh-tokens]] — variable node density support
- [[autoregressive-rollout-stability]] — why sharp-feature resolution controls rollout error
- [[unet-hierarchy-atlas-0.1]] — reuses this refinement tree in reverse, as the coarsening path of a graph U-Net
- [[multihead-attention]] — heads can specialize per refinement level / scale
