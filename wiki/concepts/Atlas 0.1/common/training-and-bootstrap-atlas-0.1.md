# Atlas 0.1: Training and Bootstrap Strategy

**Type:** Core Concept — Model Portion (Build Strategy)
**Related Concepts:** [[00-atlas-0.1-overview]], [[incremental-transfer-roadmap]], [[expert-library-atlas-0.1]], [[case-study-rocket-ascent-2d-atlas-0.1]], [[transfer-learning-fine-tuning]]

---

## This page applies, it doesn't replace, [[incremental-transfer-roadmap]]

[[incremental-transfer-roadmap]]'s 4-stage bootstrap (frozen donor wrap → LoRA fine-tuning → structural distillation → shared-trunk unification) and its donor-availability table (only Poseidon and Walrus have confirmed public checkpoints; everything else is a structural pattern to reimplement, not weights to load) apply to Atlas without modification. This page records the specific decisions that follow from applying that roadmap to [[expert-library-atlas-0.1]]'s four rocket experts, and the additional Atlas-specific scope-reduction decisions made to keep the first build diagnosable.

## Per-expert bootstrap decision

| Expert | Bootstrap approach | Reasoning |
|---|---|---|
| Reacting/internal compressible flow | **Train from scratch**, small 2D network | Neither confirmed donor (Poseidon, Walrus) was pretrained on confined reacting-flow/duct-flow data — per [[incremental-transfer-roadmap]]'s table, there is no weight-transfer path here, only the option to reimplement structural patterns from unconfirmed-checkpoint papers (GP$_{\text{hy}}$T's derivative+integrator head is the most relevant one to borrow structurally) |
| External compressible flow / plume | **Bootstrap from Poseidon** (Stage 1: frozen wrap, Stage 2: LoRA) | Closest match to Poseidon's actual pretraining distribution (compressible fluid operators); Walrus is a secondary candidate to compare against given its broader diversity |
| Thermal-structural | **Bootstrap from Poseidon** (Stage 1–2) | Elliptic/parabolic-friendly donor, same reasoning [[incremental-transfer-roadmap]] already gives for this regime family generally |
| Rigid-body trajectory | **Not learned at all** | Closed-form 3-DOF (2D) Newtonian integrator — see [[expert-library-atlas-0.1]]'s physics-encoding-spectrum placement at level 5. No bootstrap stage applies because there is no training. |

## Atlas-specific scope reductions for the first build

These are additional to [[incremental-transfer-roadmap]]'s general staging, made specifically to keep [[case-study-rocket-ascent-2d-atlas-0.1]]'s first end-to-end run diagnosable — if it doesn't work, the cause needs to be isolable to one subsystem, not entangled across several simultaneously-untested mechanisms:

1. **Declared edges, not RL** ([[edge-generation-atlas-0.1]]) — RL edge instantiation is deferred until a second scenario exists to learn across.
2. **MLP gating, not RL gating**, at every level of [[unet-hierarchy-atlas-0.1]] for v0 — even the levels flagged there as better long-term fits for RL gating use a fixed or simple learned routing table initially, since the graph and expert assignment are already declared by [[expert-library-atlas-0.1]]'s table.
3. **2-level U-Net, not 4** ([[unet-hierarchy-atlas-0.1]]) — token-level plus a single whole-vehicle bottleneck; the intermediate agent-region and per-agent pooling levels are not built until an agent count large enough to need them appears.
4. **Direct flux-matching conservation, not learned discovery** ([[conservation-as-constraint-atlas-0.1]]) — [[discovered-conservation-1.1]]'s generality is deliberately not reused for v0.

Each of these is a documented, named alternative already designed elsewhere in this vault (RL edges, RL gating, the full 4-level hierarchy, discovered conservation) — they are deferred, not rejected, and each has a stated trigger condition for when to revisit it (see the respective pages).

## Training order

Because the rigid-body expert requires no training, and the other three experts are largely independent of each other during Stage 1/2 bootstrap (each conditions on the previous expert's output at a declared edge, but doesn't need joint optimization to be individually validated), the natural order is:

1. Bootstrap and validate each of the three learned experts **independently** against its own classical-solver baseline (per-expert accuracy metric, [[case-study-rocket-ascent-2d-atlas-0.1]] Phase 0/2).
2. Wire the full graph together (Phase 3) only once all three pass their individual validation — composing three individually-unvalidated experts would make a pipeline-level failure undiagnosable, the same reasoning behind the scope reductions above.
3. End-to-end rollout validation (Phase 4) is the first point at which cross-expert coupling and the conservation constraint are tested together.

## Data generation

No existing dataset (including The Well, the largest multi-physics corpus referenced in [[pfm-concept-overview]]) covers rocket internal combustion, confined duct flow, and coupled external aero/plume together. Per [[case-study-rocket-ascent-2d-atlas-0.1]]'s Phase 0, training/validation data is synthetic, generated per expert from the simplest physically-defensible classical model available (quasi-1D/2D combustion-and-nozzle theory for the reacting-flow expert; a lightweight 2D compressible Euler/NS solver for the external-flow/plume expert; a 2D conduction+elasticity FEM solver for the thermal-structural expert) — this is Gap 8 from [[pfm-concept-overview]] made concrete at the smallest tractable scope.

---

## See Also

- [[00-atlas-0.1-overview]]
- [[incremental-transfer-roadmap]]
- [[expert-library-atlas-0.1]]
- [[edge-generation-atlas-0.1]]
- [[unet-hierarchy-atlas-0.1]]
- [[conservation-as-constraint-atlas-0.1]]
- [[case-study-rocket-ascent-2d-atlas-0.1]]
- [[transfer-learning-fine-tuning]]
