# The Unified Graph Tokenizer (initial model)

**Type:** Concept — initial model component (folder: Noether 1.0)
**Status:** Primary design for the first model's tokenizer. Synthesis of existing token representations selected to satisfy: continuous, local *and* global, resolution-free, physics-structure-preserving.
**Related Concepts:** [[00-initial-model-overview]], [[initial-model-architecture]], [[normalization-scheme]], [[graph-mesh-tokens]], [[learned-query-compression-tokens]], [[structure-preserving-tokens]], [[coordinate-implicit-tokens]], [[symmetric-attention-physics]]
**Related Summaries:** [[gns-graph-network-simulators]], [[dynami-cal-graphnet]], [[q-former-architecture]], [[multipole-graph-neural-operator]]

---

## Design requirement

The tokenizer must map *any* physical state — continuum fields on grids/meshes **and** discrete N-body/particle systems — into one token sequence that:

1. **is continuous** (real-valued latents; smooth, differentiable; criterion 8 + rollout stability),
2. **is local and global** (resolves shocks/fronts *and* long-range elliptic coupling),
3. **is resolution-free** (same tokens at $128^2$ and $512^2$; criterion 1),
4. **preserves physical relations in latent space** (translation/rotation structure, reciprocity, conserved quantities; criterion 5).

The choice: **represent every physical state as a graph**, and make each node latent resolution-free by encoding it with learned-query compression. This is the "graph token representation that subsumes grids" that [[graph-mesh-tokens]] flagged as the highest-value under-exploited direction, upgraded with a resolution-free node encoder and the structure-preserving edge construction of [[structure-preserving-tokens]].

---

## The unified graph representation

A physical state becomes a graph $G=(V,E)$. The *meaning of a node* depends on the regime, but the *graph interface downstream is identical* — this is what unifies particle and continuum physics in one backbone (the continuum–particle bridge, the project's headline goal).

### N-body / particle systems → one node per particle

Node $i$ sits at the particle position $x_i$ and carries its kinematic/material state:

$$f_i = (\,\underbrace{v_i}_{\text{velocity}},\ \underbrace{m_i, q_i}_{\text{mass, charge}},\ \underbrace{\text{type}_i}_{\text{species code}},\ \dots\,), \qquad \mathbf h_i^0 = \phi_v(f_i).$$

Edges connect particles within an interaction radius $r$. This is the [[gns-graph-network-simulators]] / [[dynami-cal-graphnet]] representation directly — geometry-native, handles arbitrary configurations, generalizes far OOD.

### Continuum fields → one node per patch (centroid node)

Partition the domain into patches. Each patch becomes **one node located at the patch centroid** $x_i$. Crucially, the node does **not** store the centroid's single sampled value — it stores a **latent encoding of the field behavior across the patch interior**: the local velocity profile, pressure, vorticity, gradients, sub-patch structure. The node communicates *values* (what the field is doing here), the edges communicate *relative position* (where "here" is relative to neighbors).

$$\mathbf h_i^0 = \mathrm{Enc}\big(\{(\,x_p - x_i,\ u(x_p)\,) : x_p \in \text{patch}_i\}\big).$$

The same downstream graph results whether the input was a uniform grid, an adaptive mesh, or scattered sensors — the node is defined by its centroid and an encoding of its neighborhood, not by a grid index.

### Edges (shared by both regimes)

Edge $(i,j)$ for $\|x_i - x_j\| < r$ carries **relative geometry only** (translation-invariant by construction):

$$\mathbf e_{ij}^0 = \phi_e\big(x_j - x_i,\ \|x_j - x_i\|,\ \mathbf h_i^0, \mathbf h_j^0\big).$$

Building edges from relative displacement gives translation invariance for free; building the scalar features from $\|x_j-x_i\|$ (and steerable/vector features where needed) gives rotation equivariance ([[equivariant-gnns]], [[structure-preserving-tokens]]). Antisymmetric edge construction ($\mathbf m_{ij} = -\mathbf m_{ji}$, [[dynami-cal-graphnet]]) makes the interaction obey Newton's third law — the tokenizer-level seed of momentum conservation that [[symmetric-attention-physics]] then carries through the dynamics.

### Typed edges for coupled (heterogeneous) systems — adopting [[open-architectural-problems]] S4.1

When fields and particles coexist (fluid + suspended grains, plasma), a field-centroid node and a particle node must exchange messages across one edge. Each node carries a **type embedding** $\tau_i$ (field / particle / …), and the edge encoder consumes *both* endpoint types, emitting messages in a **shared latent message space** so heterogeneous nodes speak one protocol:

$$\mathbf e_{ij}^0 = \phi_e\big(x_j - x_i,\ \|x_j - x_i\|,\ \tau_i,\ \tau_j,\ \mathbf h_i^0, \mathbf h_j^0\big), \qquad \mathbf m_{ij} = -\mathbf m_{ji}\ \text{(momentum-consistent across types)}.$$

The antisymmetric construction is retained *across* the type boundary, so a particle pushing on the fluid and the fluid pushing back stay equal-and-opposite — the tokenizer-level momentum seed now spans the coupling, which the S1.1 projection then enforces globally. **Still open:** making the cross-type message *physically* meaningful — field and particle latents live at different scales/units, so what a "momentum-consistent" message *is* across the boundary is the genuinely unsolved research object (see [[open-architectural-problems]] second-pass audit §B).

---

## How is the graph built — and does it have to learn the interactions?

A common and important confusion: **graph construction and interaction modeling are different jobs, handled by different parts of the model.** The tokenizer does *not* need to recognize gravity, EM, or contact to build the graph.

| Job | Decides | Learned? | Where |
|---|---|---|---|
| **Topology** | which nodes connect to which | mostly *not* learned — geometric | tokenizer |
| **Node/edge features** | what each node/edge carries | learned (the query encoder) | tokenizer |
| **Interaction / force law** | what physics acts along an edge (gravity, EM, contact) | learned, *and inferred in-context* | backbone |

### Topology is geometric, not learned
Connectivity is built by a cheap geometric rule — a **radius graph** ($\|x_i-x_j\|<r$) or **k-nearest-neighbours** — O(N) with spatial hashing. This is what [[gns-graph-network-simulators]] / [[dynami-cal-graphnet]] do, and it is principled whenever interactions are dominated by proximity (fluids, granular media, contact, short-range forces).

### The force law is learned by the backbone, not the tokenizer
The edge carries only **relative geometry** ($x_j-x_i$, $\|x_j-x_i\|$). The message/attention function $\psi(h_i,h_j,e_{ij})$ — in the *backbone* — learns what to compute from that geometry: trained on orbital trajectories it learns a $1/r^2$-shaped message; trained on contact dynamics it learns a stiff short-range repulsion. Because the law is read from the **trajectory context**, the same model infers different force laws for different systems — the in-context / Newtonian-world-model mechanism ([[kepler-newton-inductive-biases]], [[world-models-physics-ai]]). You never hardcode the interaction; you make it *reachable* and let the processor learn it. This is precisely what keeps the model equation-agnostic.

### The real subtlety: reachability vs. range
A radius graph only connects *nearby* nodes, so it can only represent interactions whose influence is local. **Long-range interactions — gravity, Coulomb/EM, the elliptic pressure coupling — are exactly the ones a flat radius graph misses** ([[graph-mesh-tokens]] cons). Two fixes:
- **Dense graph** — connect everything, $O(N^2)$. Correct but intractable at scale.
- **Multiscale supernode hierarchy** (already in this tokenizer) — aggregate regions into coarse supernodes so far-field influence travels through the coarse levels in $O(1)$ hops at $O(N)$ total cost. This is the neural analog of the **Fast Multipole Method / Barnes–Hut** tree physicists use for N-body gravity ([[multipole-graph-neural-operator]], [[multiscale-hierarchical-gnn]]). The hierarchy supplies long-range *reachability*; the backbone still learns the actual far-field law along the coarse edges.

So topology must be **co-designed with interaction range** (local edges for short-range, hierarchy for long-range) — but neither requires knowing the *specific* law in advance. Choosing the hierarchy to span all scales is exactly what lets you avoid needing to know the interaction range a priori.

### Many-node (many-body) interactions
Pairwise edges capture two-body interactions directly. Genuinely many-body effects (3-body terms, bond angles) are captured either by **deep message passing** (stacking layers aggregates multi-hop, hence multi-body, information) or by **explicit triplet/angular edge features** (DimeNet / EquiformerV3 style, [[equiformer-v3]]). "Interactions between multiple nodes" therefore does *not* require hyperedges in the basic design — depth plus pairwise edges suffices, with angular terms as an equivariant upgrade.

### Optional: learned graph construction
If interaction structure is *not* spatial (bonded vs. non-bonded atoms; a sparse interaction network unrelated to proximity), the topology itself can be learned — e.g. **Neural Relational Inference** (Kipf et al. 2018) infers a latent interaction graph from trajectories via a VAE over edge types. This is an extension, not the default; for continuum and proximity-dominated particle physics the geometric radius-graph-plus-hierarchy is sufficient and far cheaper.

**[AI Inference]:** The clean principle is *the tokenizer decides reachability; the backbone learns the force.* Hardcoding gravity into the graph would make it a gravity simulator; a geometry-only graph whose edges are interpreted by a learned, in-context processor can represent gravity, EM, contact, or viscous coupling depending on what it was trained on — and infer which applies from context. The only physics the graph construction must respect is the **range** of interaction (handled by the hierarchy), never its **form**.

---

## Resolution-freedom: the per-node set encoder

The node encoder $\mathrm{Enc}(\cdot)$ is the linchpin of resolution-freedom, and it is where **learned-query compression** ([[learned-query-compression-tokens]], [[q-former-architecture]]) enters. Inside a patch there may be $4$ sample points (coarse grid) or $400$ (fine grid). A fixed set of $N_q$ learned query vectors cross-attends over the patch's sample points and emits a **fixed-dim latent regardless of how many points were inside**:

$$\mathbf h_i^0 = \mathrm{CrossAttn}\big(Z_q,\ \{(\gamma(x_p - x_i),\ u(x_p))\}_{p \in \text{patch}_i}\big) \in \mathbb R^{N_q \times D},$$

with $\gamma(\cdot)$ a coordinate (Fourier) encoding ([[coordinate-implicit-tokens]]) of the within-patch offset. Because the queries are fixed in number and the cross-attention is permutation-invariant over an arbitrary point set, **the node latent has the same dimension and meaning at every resolution** — true resolution-freedom at the node level, the property patch embeddings fundamentally lack ([[tokenization-tradeoff-axes]], grid-tied axis). For particles, the same encoder runs over a particle and its radius-neighbors, so particle and field nodes share an encoder interface.

### Dual-path encoding: protecting sharp features — adopting [[open-architectural-problems]] S3.1

The single query-compression latent is *smooth*: it averages sub-patch structure, so shocks, fronts, and high-$k$ turbulence are attenuated at tokenization (the lossy-autoencoder gap, [[open-architectural-problems]] Problem 3). Each node is therefore encoded in **two paths**:

- a **smooth latent** $\mathbf h_i^{\text{lo}}$ (the $N_q$-query compression above) carrying the bulk/large-scale content;
- an explicit **high-frequency residual** $\mathbf h_i^{\text{hi}}$ ([[wave-particle-dual-tokens]]) carrying the sub-patch sharp-feature/high-$k$ content the smooth path drops.

The reframing that makes this principled: *strictly lossless tokenization of an arbitrary field is impossible* — an infinite-dimensional field cannot inject into a finite latent (a counting argument). But physical fields are **band-limited** (a viscous/Kolmogorov cutoff), so the achievable target is **"lossless above the physical cutoff $k_{\max}$,"** and the residual path is where the near-cutoff content is preserved rather than averaged away. Crucially, this residual channel is also the **band the S5.1 diffusion generates into** at prediction time: at *encode* time the residual is deterministically reconstructed from the input; at *predict* time the chaotic residual is *generated* — a deterministic-encode / stochastic-predict tension flagged in the second-pass audit §B. **Still open:** no measured reconstruction-fidelity floor is committed (S3.4 not selected), so this is a mechanism without an adequacy *guarantee*.

---

## Local *and* global

- **Local** is native: the radius graph connects each node only to nearby nodes → linear cost in node count, natural for shocks/fronts/characteristics (hyperbolic physics).
- **Global** is added by a **hierarchy of supernodes** (multipole / multigrid V-cycle, [[multipole-graph-neural-operator]], [[multiscale-hierarchical-gnn]]): coarse "supernode" tokens aggregate regions and provide $O(1)$-hop long-range coupling for elliptic physics (Poisson pressure, Stokes), at $O(N)$ total cost. These supernodes double as the **landmark indices** of [[index-share-sparse-attention]] and the **scene queries** of [[hierarchical-query-attention]] — one hierarchical structure, reused by the tokenizer and both large-context attention mechanisms.

Choosing the proper attention mechanism does much of the local/global work, as anticipated: symmetric attention on the radius graph handles local reciprocal interaction; supernode/landmark edges handle global coupling.

---

## Decoding (resolution-free output)

To return to physical space, decode each node latent with a **coordinate-conditioned implicit decoder** ([[coordinate-implicit-tokens]], branch–trunk style [[branch-trunk-operator-tokens]]): query the node's latent at any coordinate $y$ within its patch to reconstruct the field there,

$$\hat u(y) = \mathrm{Dec}\big(\mathbf h_i,\ \gamma(y - x_i)\big), \qquad y \in \text{patch}_i.$$

Because $y$ is continuous, the output can be sampled at *any* resolution — the decoder is resolution-free like the encoder. For particles, the decoder maps the node latent to the updated kinematic state (e.g. predicted acceleration/velocity, integrated to position).

---

## What each borrowed idea contributes

| Ingredient | From | Contributes |
|---|---|---|
| Graph nodes/edges, message passing | [[graph-mesh-tokens]], [[gns-graph-network-simulators]] | geometry-universal representation; particle+mesh in one |
| Learned-query per-node encoder | [[learned-query-compression-tokens]], [[q-former-architecture]] | resolution-freedom; fixed node latent over arbitrary point sets |
| Relative-displacement edges, antisymmetric frames | [[dynami-cal-graphnet]], [[structure-preserving-tokens]] | translation/rotation structure; Newton's 3rd law seed |
| Coordinate (Fourier) encoding | [[coordinate-implicit-tokens]] | within-patch position; resolution-free decode |
| Supernode hierarchy | [[multipole-graph-neural-operator]], [[multiscale-hierarchical-gnn]] | global/elliptic coupling at $O(N)$; landmarks/scene queries |

This is a deliberate mix-and-match: the graph backbone subsumes grids; the query encoder fixes resolution; the structure-preserving edges preserve physics; the hierarchy supplies global reach.

---

## Position on the trade-off axes ([[tokenization-tradeoff-axes]])

- **Continuous** ✓ (real-valued node/edge latents; differentiable encode/decode).
- **Local + Global** ✓ (radius graph + supernode hierarchy).
- **Resolution-free** ✓ (query-compression encoder + coordinate-conditioned decoder).

It targets the theoretically strong corner (Continuous, Local+Global, Resolution-free) that current grid-tied PFMs avoid for engineering convenience — accepting the higher implementation complexity (graph construction, irregular memory) as the price of the continuum–particle bridge.

---

## Open problems

1. **Patch/centroid definition for unstructured input.** For scattered data without a grid, patches must be defined by spatial clustering (e.g. k-means / Voronoi on sample points) — adds a clustering step and a choice of patch count.
2. **Graph-construction cost for moving particles.** Radius-graph rebuilds each step; neighbor-list maintenance is a real cost (flagged in [[graph-mesh-tokens]]).
3. **Encoder query budget $N_q$.** Too small loses sub-patch turbulence/shocks. **Adopted (S3.1 + S3.3 + S3.4):** the dual-path high-frequency residual channel (above) preserves near-cutoff content; adaptive $N_q$ ([[intelligent-patching]]) matches capacity to local spectral content; and a **reconstruction fidelity floor (S3.4)** gates tokenizer pretraining on encode→decode error below the data's discretization error — supplying the adequacy *guarantee* the mechanism alone lacked.
4. **Mixed continuum–particle graphs.** When fields and particles coexist (e.g. fluid + suspended grains), field-centroid nodes and particle nodes share one graph. **Adopted (S4.1 + S4.4):** the typed-edge encoder (above) gives heterogeneous nodes a shared, momentum-consistent message protocol, validated on a **settling-particles coupled benchmark (S4.4)**. *Still open:* the *physical meaning* of a cross-type message (scale/unit mismatch) — the one genuine research object left in the bridge.

---

## [AI Inference]

**[AI Inference]:** Making the node latent a *set encoding of a neighborhood* rather than a *point sample* is what unifies the particle and continuum cases: a particle node encodes "this particle + its neighbors," a field node encodes "this centroid + its sample points," and downstream they are indistinguishable graph nodes. The continuum–particle bridge the wiki has repeatedly flagged as the missing piece is therefore primarily a **tokenizer** problem, not a backbone problem — if both regimes produce the same kind of node latent, a single symmetric-attention backbone evolves both with no architectural fork.

**[AI Inference]:** Because the same supernode hierarchy serves the tokenizer (global coupling), Index Share (landmarks), and hierarchical queries (scene stream), the model has *one* multiscale structure rather than three. This argues for building the hierarchy **once, at tokenization**, and exposing it to every later stage — a single learned restriction/prolongation operator pair ([[multiscale-hierarchical-gnn]]) reused throughout, which also makes the coarse level a natural home for the compressed long-horizon "scene memory" of [[hierarchical-query-attention]].

---

## See Also

- [[00-initial-model-overview]] — where the tokenizer sits in the first model
- [[initial-model-architecture]] — full pipeline + diagram
- [[normalization-scheme]] — how node latents are normalized without destroying scale/structure
- [[symmetric-attention-physics]] — the backbone that evolves these tokens (attention = graph Laplacian)
- [[graph-mesh-tokens]] — the base representation this extends
- [[learned-query-compression-tokens]] / [[q-former-architecture]] — the resolution-free node encoder
- [[structure-preserving-tokens]] / [[dynami-cal-graphnet]] — physics-preserving edges
- [[coordinate-implicit-tokens]] / [[branch-trunk-operator-tokens]] — resolution-free decode
- [[multipole-graph-neural-operator]] / [[multiscale-hierarchical-gnn]] — global supernode hierarchy
- [[tokenization-tradeoff-axes]] — axis placement
