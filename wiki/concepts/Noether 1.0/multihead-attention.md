# Multi-Head Attention for the PFM — Is It Needed, and What Shape?

**Type:** Concept — initial model component (folder: Noether 1.0)
**Status:** Design decision on the head structure of the symmetric/conservative backbone.
**Related Concepts:** [[symmetric-attention-physics]], [[skew-symmetric-attention]], [[initial-model-architecture]], [[index-share-sparse-attention]], [[graph-tokenizer]], [[equivariant-gnns]]
**Related Summaries:** [[equiformer-v3]], [[dynami-cal-graphnet]], [[glm-index-share-attention]], [[anti-symmetric-dgn]]

---

## The question

Standard transformers use multiple attention heads so different heads can attend in different subspaces / capture different relations. Does a physics model need them? **Yes — but for a sharper reason than in language, and with a constrained structure to protect conservation.**

---

## Why one head is not enough

A single symmetric-attention head is **one learnable graph Laplacian** ([[symmetric-attention-physics]]) — one coupling pattern over the token graph. But real multiphysics requires *several simultaneous, qualitatively different couplings at the same instant*:

- **Pressure** couples *globally and elliptically* (a Poisson solve — every point feels every other instantly).
- **Vorticity / scalars** advect *locally and hyperbolically* (finite-speed transport along characteristics).
- **Diffusion/heat** couples *locally and parabolically* (smoothing).
- **Long-range forces** (gravity, Coulomb) couple at *all ranges* via the multipole hierarchy.

These cannot be expressed by one Laplacian. **Each head is best read as one physical coupling operator**; the head set is a *basis of interaction operators* the model composes per regime. So multi-head attention is needed — but as "multiple physical operators acting at once," not "arbitrary subspaces."

---

## The conservation problem with standard MHA

Standard MHA concatenates per-head outputs and mixes them with a dense learned projection $W_O$:
$$h_i' = W_O\,\big[\,\text{head}_1; \dots; \text{head}_H\,\big].$$
For physics this is dangerous: even if each head is individually symmetric/conservative ([[symmetric-attention-physics]], [[skew-symmetric-attention]]), an **arbitrary dense $W_O$ reintroduces asymmetry and breaks the per-head conservation** — the aggregate momentum/energy guarantees do not survive a generic linear mix. This is the open "per-head antisymmetry" problem flagged in [[antisymmetric-signed-attention-transformer]].

---

## The proposed head structure (physically-structured MHA)

Three constraints make multi-head attention safe and meaningful here:

### 1. Each head is independently symmetric + conservative
Per head $h$: a symmetric score $A^{(h)}_{ij}=A^{(h)}_{ji}$, force-style aggregation (momentum), and its own skew-symmetric linear channel (energy):
$$\Delta h_i^{(h)} = (W^{(h)}-W^{(h)\top}-\gamma_h I)\,h_i + \sum_j A^{(h)}_{ij}\,(V^{(h)}_j - V^{(h)}_i).$$
Each $\gamma_h$ can differ — heads can sit at different points on the conservative↔dissipative spectrum (one near-conservative head for waves, one damped head for viscous diffusion).

### 2. Heads are combined by a *conservation-preserving* operator, not a dense $W_O$
Replace the dense mix with either **summation** (a residual add of per-head updates, which preserves the per-head conservation exactly because the sum of conserved updates is conserved):
$$h_i' = h_i + \epsilon\sum_h \sigma\big(\Delta h_i^{(h)}\big),$$
or, if a learned mix is wanted, an **orthogonal / block-structured $W_O$** that preserves norm and antisymmetry. Summation is the safe default for the initial model.

### 3. Heads are *assigned*, not anonymous — organized by physics
Rather than $H$ interchangeable heads, partition them along physically meaningful axes:
- **By field-type group** — heads for velocity-velocity coupling, velocity-pressure coupling, scalar transport, etc. (the field-type codes from [[pfm-interface-design]] route here). Heads in a field group can **share the neighbor graph / index set** — exactly GLM's Index Share, reinterpreted: co-located field channels share spatial coupling structure ([[index-share-sparse-attention]], [[glm-index-share-attention]]).
- **By scale / refinement level** — one head per level of the multiscale tree ([[intelligent-patching]], [[multiscale-hierarchical-gnn]]): fine-level heads do local interaction, coarse-level heads do global/elliptic coupling.
- **By interaction sign/type** — an attractive head and a repulsive (signed) head, supporting pattern formation ([[symmetric-attention-physics]] signed coupling).
- **By irrep (for equivariance)** — scalar / vector / tensor heads carrying different SO(3) representations, the EquiformerV3 structure ([[equiformer-v3]], [[equivariant-gnns]]).
- **By node-type coupling (S4.1)** — with the typed-edge encoder ([[graph-tokenizer]], [[open-architectural-problems]] S4.1), a head can specialize in a node-type pair (field↔field, particle↔particle, field↔particle), so heterogeneous coupling gets its own conservation-preserving operator instead of being averaged into the field–field heads.

---

## So: how many heads, and is it "needed"?

- **Needed:** yes — single-head attention cannot represent simultaneous elliptic + hyperbolic + parabolic coupling; multiphysics demands several operators at once.
- **Count:** set by the *physics*, not by a round number — roughly (field-type groups) × (scales) × (interaction types), pruned by what the target systems actually contain. For the 1D Burgers smoke test, a handful suffices; for full multiphysics, more.
- **Shape:** physically-structured, conservation-preserving combination (summation or orthogonal $W_O$), heads assigned to physical roles — *not* anonymous dense-mixed heads.

---

## Pros / Cons of this structured MHA

**Pros**
- Represents multiple simultaneous coupling operators (the multiphysics requirement).
- Preserves per-head conservation through summation/orthogonal combination.
- Interpretable — heads map to physical operators/fields/scales; probeable.
- Reuses Index Share (field-group index sharing) and the multiscale tree (scale heads) — no new machinery.

**Cons**
- Structured $W_O$ / summation is less expressive than a dense mix; may cost raw fit on non-physical benchmarks.
- Head assignment adds design choices (which groups, how many scales).
- Equivariant (irrep) heads are expensive (tensor products, [[equiformer-v3]]).
- Verifying conservation survives the chosen combination needs explicit checks (a planned ablation in [[initial-model-architecture]]).

---

## Relationship to the FFN sub-step ([[open-architectural-problems]] Problem 12)

The head structure above governs the **attention sub-step** only — the communication/routing half of each backbone layer. [[initial-model-architecture]] Stage 4 adds a second, **head-agnostic** Euler sub-step (a norm-bounded semi-orthogonal FFN, adopted F1) for per-node nonlinear computation. The two sub-steps have cleanly separated jobs: heads decide *which physical operators couple which nodes*; the FFN sub-step decides *what nonlinear function of its own state each node computes*, independent of head assignment. This keeps the physically-structured head taxonomy above uncontaminated by the FFN's unconstrained-per-node expressivity.

---

## [AI Inference]

**[AI Inference]:** The cleanest interpretation of heads here is a **learned basis of physical operators** — divergence, curl, gradient, Laplacian, advection. Each symmetric head is a learnable second-order operator on the token graph; a small set spans the operators that appear in the PDEs the model targets. This predicts something testable: after training, linear probes on individual heads should recover recognizable differential operators, and ablating a head should specifically damage the physics that operator governs (remove the global/elliptic head → pressure coupling fails). Heads stop being a hyperparameter and become a physical decomposition.

**[AI Inference]:** Head structure, Index Share, and intelligent patching are the *same* organizing principle applied at three places: group by **field** (which channels share a stencil), by **scale** (which refinement level), and by **interaction type**. A single grouping scheme — field × scale × type — could index the heads, the shared neighbor lists, and the refinement tree simultaneously, so "how many heads," "which tokens interact," and "where to refine" are all answered by one learned partition. That unification is the strongest argument that physically-structured MHA is the right shape rather than anonymous heads.

---

## See Also

- [[symmetric-attention-physics]] — the per-head symmetric mechanism
- [[skew-symmetric-attention]] — the per-head conservative channel; $W_O$ antisymmetry problem
- [[index-share-sparse-attention]] — field-group heads sharing index sets
- [[intelligent-patching]] / [[multiscale-hierarchical-gnn]] — scale-assigned heads
- [[equiformer-v3]] / [[equivariant-gnns]] — irrep-structured heads for equivariance
- [[graph-tokenizer]] — the token graph heads operate on
- [[initial-model-architecture]] — where MHA sits; the conservation-survival ablation
- [[pfm-interface-design]] — field-type codes that route field-group heads
