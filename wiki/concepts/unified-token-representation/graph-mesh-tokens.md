# Graph / Mesh Tokens (Nodes & Edges)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** [[gns-graph-network-simulators]], [[dynami-cal-graphnet]], [[multipole-graph-neural-operator]], [[equiformer-v3]]
**Related:** [[00-token-representation-overview]], [[equivariant-gnns]], [[patch-embedding-tokens]], [[multiscale-hierarchical-gnn]], [[transformer-architectures]]

---

## Intuition

Drop the grid entirely. Represent the physical state as a **graph**: each particle, mesh node, or sample point is a *node token* carrying its local state (position, velocity, mass, field values); pairs of nearby nodes are connected by *edge tokens* carrying relative geometry (displacement, distance). "Attention" becomes **message passing** along edges. This is the natural representation when geometry is irregular — turbine blades, biological tissue, free-surface fluids, molecular systems, astrophysical N-body — where no Cartesian patch grid exists. Because a transformer *is* a fully-connected GNN (attention = message passing over a complete graph; see [[transformer-architectures]], [[antisymmetric-signed-attention-transformer]]), graph tokens and patch tokens are two ends of one spectrum: patches assume a regular dense grid; graph tokens allow arbitrary sparse connectivity.

---

## Mathematics

State as graph $G=(V,E)$. Node token $\mathbf h_i$ for node $i$ at position $x_i$ with features $f_i$; edge token $\mathbf e_{ij}$ for $(i,j)\in E$ (typically a radius graph, $\|x_i-x_j\|<r$). Encode:

$$\mathbf h_i^0 = \phi_v(f_i),\qquad \mathbf e_{ij}^0 = \phi_e(x_j-x_i,\,\|x_j-x_i\|,\,f_i,f_j).$$

One message-passing (encode–process–decode, [[gns-graph-network-simulators]]) step:

$$\mathbf m_{ij}=\psi(\mathbf h_i,\mathbf h_j,\mathbf e_{ij}),\qquad \mathbf h_i' = \gamma\Big(\mathbf h_i,\ \textstyle\bigoplus_{j\in\mathcal N(i)}\mathbf m_{ij}\Big),$$

with permutation-invariant aggregation $\bigoplus$ (sum/mean). Physics structure enters through the edge token:
- **Equivariance** ([[equivariant-gnns]], [[equiformer-v3]]): edge features built from invariants ($\|x_j-x_i\|$) or steerable (spherical-harmonic) representations → SE(3)-equivariant tokens.
- **Hard conservation** ([[dynami-cal-graphnet]]): antisymmetric edge-local frames make messages obey Newton's third law ($\mathbf m_{ij}=-\mathbf m_{ji}$) → exact linear & angular momentum conservation.
- **Multiscale** ([[multipole-graph-neural-operator]]): a hierarchy of graphs (FMM V-cycle) gives $O(N)$ all-range coupling.

---

## Pros

- **Geometry-native** — arbitrary point clouds, meshes, irregular domains, complex boundaries; no grid required (criterion 2, [[00-token-representation-overview]]).
- **Symmetry & conservation can be built into edge tokens** — equivariance (EquiformerV3) and exact momentum conservation (Dynami-CAL) are *architectural*, not learned — encoding-spectrum level 5.
- **Discretization-flexible** — node density can vary spatially (adaptive resolution where physics is active), a natural adaptive-mesh tokenization.
- **Many-body / Lagrangian physics** — natural for particles, granular media, molecular dynamics, N-body gravity.
- **Local by construction** — sparse radius graph → linear cost in nodes (for fixed degree); long-range coupling added via multiscale hierarchy.
- **Generalizes far out of distribution** — GNS shows 34× generalization beyond training configurations.

## Cons

- **Long-range coupling is expensive on a flat graph** — elliptic interactions (Poisson, gravity, Coulomb) need either dense graphs ($O(N^2)$) or a multiscale hierarchy ([[multiscale-hierarchical-gnn]]); a plain radius graph misses them.
- **Graph construction overhead** — building/maintaining the neighbor graph (especially for moving particles) costs compute; choice of radius $r$ / degree is a hyperparameter.
- **Over-smoothing & over-squashing** — deep message passing washes out node distinctions and bottlenecks long-range info ([[anti-symmetric-dgn]] addresses this).
- **Less mature at foundation-model scale for continuum fields** — GNN PFMs lag transformer PFMs ([[walrus-paper]], [[poseidon-pde-foundation-model]]) in breadth, though they dominate molecular/materials ([[equiformer-v3]]).
- **Irregular memory access** — harder to batch/accelerate than dense grid tokens.

---

## Relationship to other representations

- vs. [[patch-embedding-tokens]]: patches = nodes on a regular dense lattice with implicit grid edges; graph tokens generalize to arbitrary connectivity. A patch tokenizer is a special case of a grid-graph tokenizer.
- vs. [[hierarchical-windowed-tokens]] / [[multiscale-hierarchical-gnn]]: Swin's pyramid is the structured-grid analog of the multipole graph hierarchy — same multigrid/FMM idea, different geometry support.
- vs. [[branch-trunk-operator-tokens]]: both handle irregular sampling; DeepONet via a global learned basis, graphs via local message passing — global-summary vs. local-interaction.

**[AI Inference]:** The cleanest path to a *geometry-universal* PFM tokenizer is a **graph token representation that subsumes grids**: represent Cartesian fields as grid-graphs (recovering patch/Swin behavior) and meshes/particles as general graphs, under one message-passing backbone with equivariant + conservation-structured edge tokens. This single representation would satisfy criteria 2 (geometric generality) and 5 (structure preservation) simultaneously — the most under-exploited high-value direction in [[00-token-representation-overview]], given that transformer PFMs currently abandon it for grid convenience.

**[AI Inference]:** Since attention is message passing on a complete graph, a **hybrid sparse+global graph tokenizer** — local edges for fine interactions, a few hierarchical "supernode" tokens for global coupling — is exactly the transformer/GNN unification behind [[multipole-graph-neural-operator]] and [[antisymmetric-signed-attention-transformer]]. Graph tokens are the representation that makes the "transformer = physical GNN" program concrete.

---

## See also

- [[00-token-representation-overview]] — hub; criteria 2 (geometry) and 5 (structure)
- [[equivariant-gnns]] — symmetry/conservation taxonomy for graph tokens
- [[gns-graph-network-simulators]] — encode–process–decode particle tokens
- [[dynami-cal-graphnet]] — antisymmetric edge tokens, exact momentum
- [[multipole-graph-neural-operator]] / [[multiscale-hierarchical-gnn]] — multiscale graph tokens
- [[equiformer-v3]] — SE(3)-equivariant graph attention tokens
- [[patch-embedding-tokens]] — the regular-grid special case
- [[antisymmetric-signed-attention-transformer]] — attention as physical message passing
