# Edge Generation (Noether 1.1)

**Type:** Concept — model portion, NEW dedicated page in 1.1 (folder: Noether 1.1)
**Status:** Design. The connectivity of the typed multigraph the backbone runs on. Replaces 1.0's single thin 4-neighbor radius graph ([[noether-1.0-rbc]] §4.2 — static, uniform, physics-blind edges) with **three edge families**, the third of which is produced by a small **learned edge-generation model**. Realizes the "reachability is the tokenizer's job, the force law is the backbone's" principle from [[graph-tokenizer]], extended to a typed, partly-learned setting.
**Related Concepts:** [[00-noether-1.1-overview]], [[field-token-streams-1.1]], [[backbone-1.1]], [[graph-tokenizer]], [[edge-generation-1.1]], [[multipole-graph-neural-operator]], [[message-passing-belief-propagation]]
**Related Summaries:** [[gns-graph-network-simulators]], [[dynami-cal-graphnet]], [[multipole-graph-neural-operator]]

---

## What it does — intuitively

Decide **who talks to whom**. A message-passing model can only represent an interaction if an edge carries it, so under-connectivity is a hard ceiling on expressivity — the RBC smoother had, in part, nowhere to *put* plume-to-plume coupling. 1.1 supplies three kinds of edge:

1. **Local same-field (KNN).** Each field token connects to its $K$ nearest same-field neighbors — the transport/diffusion stencil, dense enough to see shear and fronts (1.0's 4-neighbor was too thin).
2. **Local cross-field (KNN, smaller $K'$).** Each token connects to a few *other-field* tokens nearby — the local coupling channel (buoyancy, reaction), kept small because pointwise coupling is short-range.
3. **Learned long-range.** A small model **proposes** edges between *far-apart* tokens — the coupling geometry can't guess: teleconnected plumes, elliptic pressure influence, boundary-layer-to-core links, or genuinely non-spatial interaction networks.

Families 1–2 are cheap geometric rules (translation-invariant, no learning). Family 3 is where a **pattern recognizer for topology** lives — the model discovers connectivity the way it discovers everything else, rather than being told the interaction range.

## What it does — mathematically

Let the token set be $\{t_i^f\}$ at positions $\{x_i\}$ with field tags $f$.

**Family 1 — same-field KNN.** For each $(i,f)$: edges to $\mathrm{KNN}_K(x_i)$ among same-$f$ tokens (periodic / wall-aware per BC). Symmetric (score each pair once) so the **antisymmetric-flux** momentum channel of [[backbone-1.1]] is well-defined on them.

**Family 2 — cross-field KNN.** For each $(i,f)$ and each other field $f'$: edges to $\mathrm{KNN}_{K'}(x_i)$ among $f'$-tokens, $K'<K$. Includes the **co-located** pair $t_i^f\!\leftrightarrow\!t_i^{f'}$ always.

**Family 3 — learned long-range.** A lightweight scorer over candidate pairs within a generous envelope $R\gg r$ (or, at node scale, *all* pairs — see cost note):
$$s_{ij} = g_\theta\big(t_i^{f},\ t_j^{f'},\ \gamma(x_j-x_i),\ \|x_j-x_i\|\big),\qquad a_{ij} = \mathrm{GumbelSigmoid}(s_{ij})\in\{0,1\}\ \text{(or a soft gate)}.$$
Edges are kept where $a_{ij}=1$, **added on top of** families 1–2 (never replacing them), scored **once per unordered pair** to stay undirected, and trained **jointly under the dynamics loss** as latent variables (NRI-style, [[graph-tokenizer]] "learned graph construction"; Kipf et al. 2018). No edge labels are needed. A top-$k$ or entropy budget on $\sum_j a_{ij}$ keeps the learned graph sparse.

**How the score becomes an edge — GumbelSigmoid.** $s_{ij}$ is a real-valued logit; an edge is binary, and a hard threshold is non-differentiable, so $g_\theta$ could never receive a gradient through it. GumbelSigmoid injects noise and a temperature $\tau_{\text{gs}}$: $a_{ij}=\sigma\big(\tfrac{1}{\tau_{\text{gs}}}(s_{ij}+\log u-\log(1-u))\big)$, $u\sim\mathrm{Unif}(0,1)$ — a stochastic relaxation that is soft and differentiable in $(0,1)$ at high $\tau_{\text{gs}}$ and snaps to a hard $\{0,1\}$ sample as $\tau_{\text{gs}}\to0$. Anneal $\tau_{\text{gs}}$ down over training with a straight-through estimator (hard value in the forward pass, soft gradient in the backward pass), so inference gets a crisp graph while training stays differentiable throughout.

**How $g_\theta$ actually learns — no edge labels, ever.** Training is ordinary end-to-end optimization, not a separate procedure: forward, every candidate pair is scored and sampled, the backbone message-passes over base-KNN-edges $\cup$ kept-learned-edges, and the resulting state is decoded and scored against the true next state. Backward, the prediction loss's gradient flows through the (differentiable) $a_{ij}$ into $s_{ij}$ into $g_\theta$'s weights: if connecting $(i,j)$ *reduced* next-state error, $\partial\mathcal L/\partial a_{ij}<0$ pushes $s_{ij}$ up, so the model learns to propose that edge — purely because the resulting messages improved prediction, never from a supervised topology. The sparsity budget's gradient pushes every gate down simultaneously, so a learned edge survives only if it earns its place against that pressure.

**Cost note.** The scorer is $O(N^2)$ in the number of *nodes*, not grid points — with $\mathcal O(10^2)$–$10^3$ nodes, full pairwise scoring is $\sim10^4$–$10^6$ ops, trivial. The $O(N^2)$ objection that kills learned edges at pixel scale does not apply at node scale.

## Physics relevance

- **Reachability co-designed with interaction range.** Local families cover proximity-dominated short-range physics; the learned family + the supernode hierarchy ([[multipole-graph-neural-operator]]) cover long-range elliptic/field-mediated coupling — the FMM/Barnes–Hut analogy, but with the far-field *connectivity* discovered rather than fixed to a tree.
- **Conservative vs. forcing edges, cleanly separated.** Same-field edges carry the **antisymmetric** flux $m_{ij}=-m_{ji}$ ⇒ transport conserves momentum structurally; cross-field and learned edges are **allowed to be non-conservative**, because coupling/forcing (buoyancy, external drive) genuinely is not momentum-conserving. This maps the graph's edge types onto the physics: *conservative geometric edges transport; non-conservative learned edges force.*
- **Discovering non-spatial interaction graphs.** For systems where interaction is not proximity (bonded vs. non-bonded, sparse coupling networks), family 3 is the only family that can represent it — the same mechanism that lets 1.1 leave "which physics" unspecified lets it leave "which topology" unspecified.

## How fundamental constants are included

- **Range constants set the geometric envelopes.** A screening/Debye length, a cutoff radius, or a diffusive length (built from dimensional constants, [[conditioning-and-constants-1.1]]) sets $r$ (KNN radius) and $R$ (learned-edge candidate envelope). Same code, collisionless or collisional, short- or long-range — the constant chooses.
- **The learned scorer is conditioned, not hard-coded.** $g_\theta$ receives $z_{\text{cond}}$, so the *same* scorer proposes denser long-range edges at high Rayleigh/low viscosity (where non-local coupling matters) and sparser ones when the physics is local. It never contains a specific constant's value.
- **Symmetry constants constrain candidate edges.** A vector constant's covariant channel biases which long-range edges are physically admissible (e.g. gravity makes vertical teleconnection more likely than horizontal) — the arrow ([[conditioning-and-constants-1.1]]) reaching into topology.

## Open items / risks

- **A learned edge generator can learn the topology of doing nothing.** If the objective still rewards near-identity ([[00-noether-1.1-overview]]), family 3 will help the model stay still. Edge learning must be trained under a motion-demanding loss (multi-step target), or it inherits the 1.0 failure at the topology level.
- **Keeping momentum-antisymmetry well-defined** requires learned edges stay undirected; directed learned edges would break the flux cancellation — enforced by single-scoring each pair.
- **Gate collapse / posterior collapse** (all edges on, or all off) — mitigated by the sparsity budget and a small edge-entropy regularizer, standard NRI hygiene.

## See Also
- [[field-token-streams-1.1]] — supplies the typed tokens these edges join
- [[backbone-1.1]] — consumes the typed multigraph; where the gated antisymmetric channel lives
- [[graph-tokenizer]] — "tokenizer decides reachability, backbone learns the force"; the NRI extension note
- [[multipole-graph-neural-operator]] — the hierarchy that complements learned long-range edges
- [[edge-generation-atlas-0.1]] — the [[00-atlas-0.1-overview]] track's simplification (declared/geometric instantiation for a known graph) plus a documented RL-based alternative, contrasted against this page's learned-scorer approach
