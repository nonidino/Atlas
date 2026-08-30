# Structure-Preserving Tokens (Equivariant / Conservation-Aware / Coordinate-&-Scale-Aware)

**Type:** Token representation — *proposed better alternative* (folder: unified-token-representation)
**Status:** Synthesis / proposal drawing on existing equivariant + conservation literature. Mixed established results and **[AI Inference]**.
**Builds on:** [[equivariant-gnns]], [[dynami-cal-graphnet]], [[pc-deeponet-cfd]], [[hamiltonian-message-passing]], [[action-based-noether-enforcement]]
**Related:** [[00-token-representation-overview]], [[patch-embedding-tokens]], [[spectral-fourier-tokens]], [[scalar-cdf-tokens]], [[structure-preserving-tokens]]

---

## Intuition

Every other representation in this folder treats tokenization as *information packaging* — get the field into vectors, then hope the network learns the physics. Structure-preserving tokens take the opposite stance: **build the physics into the tokenizer so that physical laws are true of the tokens by construction, before any learning.** Three structures to preserve, in increasing ambition:

1. **Coordinate & scale awareness** — each token *knows where it is* and *at what regime* (embed spatial coordinates and dimensionless numbers alongside field values), fixing the position-blindness and scale-blindness of [[patch-embedding-tokens]].
2. **Symmetry (equivariance)** — the token representation transforms correctly under translations, rotations, reflections, Galilean boosts; a rotated input yields rotated tokens, so the model never has to learn symmetry from data ([[equivariant-gnns]], [[equiformer-v3]]).
3. **Conservation** — the tokenized state lies in a physically admissible subspace: divergence-free, momentum/energy-conserving, etc., so *no representable token configuration violates the law* ([[dynami-cal-graphnet]], [[pc-deeponet-cfd]]).

This is the only family that targets encoding-spectrum **level 5 (hard architectural constraints)** at the *tokenization* stage, the strongest position on criterion 5 of [[00-token-representation-overview]].

---

## Mathematics

**Coordinate/scale-aware token.** Augment the embedding with positional and regime information:
$$\mathbf v_j = E\big(\,\underbrace{a_j}_{\text{field}},\ \underbrace{\gamma(x_j)}_{\text{coord. encoding}},\ \underbrace{\pi(\mathrm{Re},\mathrm{Ma},\mathrm{Pr},\ldots)}_{\text{dimensionless regime}}\,\big),$$
with $\gamma$ a Fourier/coordinate encoding ([[coordinate-implicit-tokens]]) and $\pi$ from scalar-CDF or continuous conditioning ([[scalar-cdf-tokens]]).

**Equivariant token.** Tokens carry features in irreducible representations of the symmetry group $G$ (e.g. SO(3) via spherical harmonics; [[equiformer-v3]]). For $g\in G$ acting as $\rho_\text{in}(g)$ on inputs and $\rho_\text{out}(g)$ on tokens:
$$E(\rho_\text{in}(g)\cdot a) = \rho_\text{out}(g)\cdot E(a)\qquad(\text{equivariance}).$$
Edge/relative tokens built from invariants ($\|x_i-x_j\|$) or steerable features achieve this exactly ([[graph-mesh-tokens]]).

**Conservation-constrained token.** Restrict the decodable token space to a constraint manifold. E.g. **divergence-free** via a stream-function/projection (Leray):
$$u = \nabla\times\psi \ \Rightarrow\ \nabla\cdot u=0,\quad\text{or spectral projection}\ \hat u(k)\mapsto \hat u(k)-\tfrac{k(k\cdot\hat u(k))}{|k|^2}.$$
**Antisymmetric edge tokens** ($\mathbf m_{ij}=-\mathbf m_{ji}$) give exact linear & angular momentum conservation ([[dynami-cal-graphnet]]); **Hamiltonian/Lagrangian** token structure gives exact energy conservation in the conservative limit ([[hamiltonian-message-passing]], [[action-based-noether-enforcement]]).

---

## Pros

- **Physics true by construction** — symmetries and conservation hold *exactly*, not approximately/learned; encoding-spectrum level 5 at the token level (criterion 5, the one almost every other representation fails).
- **Sample efficiency & generalization** — not having to learn symmetry/conservation from data means far fewer samples and better OOD transfer (the consistent finding across [[equivariant-gnns]], [[dynami-cal-graphnet]]).
- **Coordinate/scale awareness** fixes two long-standing gaps ([[patch-embedding-tokens]] position-blindness, [[multimodal-tokenization]] scale-blindness) — criteria 4 directly.
- **Eliminates stabilization hacks** — an exactly translation-equivariant tokenizer removes the *need* for [[walrus-paper]]'s patch jittering (which only approximates equivariance by averaging over random shifts).
- **Trustworthy & physically admissible outputs** — guaranteed divergence-free / conserving states matter for scientific and engineering use.

## Cons

- **Constraints restrict expressivity / cost** — equivariant tensor products are expensive ($O(L_{\max}^4)$ before tricks; [[equiformer-v3]]) and constraint manifolds can be hard to parameterize, especially on irregular geometry with mixed BCs.
- **Hard to combine many structures at once** — coordinate-aware + equivariant + conservation-constrained + multi-field is a heavy, intricate tokenizer; few systems do more than one well.
- **Constraint must match the physics** — a divergence-free token space is wrong for compressible flow; baking in the *wrong* invariant hurts. Requires knowing which laws hold (against the "in-context, equation-agnostic" ethos of [[gphyt-physics-foundation-model]]).
- **Approximate symmetries** — many real systems have only *broken* or *approximate* symmetries (boundaries break translation; dissipation breaks energy conservation); hard constraints can be too rigid ([[noether-networks]] meta-learns *approximate* conservation as a softer alternative).
- **Least transformer-native** — most mature in the GNN/operator world; integrating into a large vision-transformer PFM is non-trivial.

---

## Why this is a *better* alternative

On the [[00-token-representation-overview]] scorecard, structure-preserving tokens are the *only* family that targets criterion 5 (structure preservation) at full strength, and the coordinate/scale-aware variant also nails criterion 4. The cost is generality and complexity: hard constraints assume known physics, cutting against the equation-agnostic in-context philosophy of the strongest current PFMs. The resolution is almost certainly **a spectrum, not a binary**: use coordinate/scale-aware embedding *universally* (cheap, always helpful), equivariance *where the symmetry is exact* (molecular, free-space), and hard conservation *as an optional projection head* selected per regime — exactly the physics-encoding-spectrum framing the wiki organizes around.

**[AI Inference]:** The single highest-value, lowest-cost upgrade to *current* PFMs is the **coordinate-&-scale-aware** layer alone: append a coordinate encoding and dimensionless-number conditioning to [[patch-embedding-tokens]]. It requires no architectural surgery, addresses two documented gaps, and is compatible with every backbone ([[poseidon-pde-foundation-model]]'s lead-time LayerNorm is already a partial instance — conditioning normalization on the scalar $t$; generalize it to $(\mathrm{Re},\mathrm{Ma},\ldots)$).

**[AI Inference]:** Equivariance and conservation should likely be **selectable per task via routing** ([[mixture-of-experts]]): the tokenizer offers a divergence-free projection, an energy-conserving head, an equivariant frame, etc., and a regime classifier (or in-context inference) chooses which constraints to activate — getting hard-constraint benefits where the law applies without sacrificing generality where it doesn't. This unifies structure-preserving tokens with the in-context generality of [[gphyt-physics-foundation-model]] and [[poseidon-pde-foundation-model]].

**[AI Inference]:** Spectral tokens ([[spectral-fourier-tokens]]) are the cheapest host for conservation constraints (divergence projection is one linear operation per mode), and graph tokens ([[graph-mesh-tokens]]) the cheapest host for momentum conservation (antisymmetric edges). So *which* representation you pick determines *which* structures are cheap to preserve — structure-preservation is not a standalone tokenizer but a property to compose onto the others.

---

## See also

- [[00-token-representation-overview]] — hub; criteria 4 (scale) and 5 (structure)
- [[equivariant-gnns]] — symmetry taxonomy (spatial → SE(3) → conservation)
- [[dynami-cal-graphnet]] — antisymmetric edge tokens, exact momentum
- [[pc-deeponet-cfd]] — divergence-free-by-construction operator tokens
- [[hamiltonian-message-passing]] / [[action-based-noether-enforcement]] — energy/action structure
- [[noether-networks]] — meta-learned *approximate* conservation (softer alternative)
- [[patch-embedding-tokens]] — the baseline lacking coordinate/scale awareness
- [[spectral-fourier-tokens]] / [[graph-mesh-tokens]] — cheap hosts for specific constraints
- [[scalar-cdf-tokens]] / [[coordinate-implicit-tokens]] — scale & coordinate encoding
- [[poseidon-pde-foundation-model]] — lead-time LayerNorm as partial scale-conditioning
