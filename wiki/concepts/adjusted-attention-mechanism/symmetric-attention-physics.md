# Symmetric Attention for Physics (the core PFM interaction)

**Type:** Concept (folder: adjusted-attention-mechanism)
**Status:** Design choice for the initial model. Builds directly on [[antisymmetric-signed-attention-transformer]]; this page is the focused justification + the dissipative-vs-conservative resolution.
**Related Concepts:** [[00-attention-overview]], [[antisymmetric-signed-attention-transformer]], [[graph-tokenizer]], [[normalization-scheme]], [[initial-model-architecture]], [[equivariant-gnns]]
**Related Summaries:** [[anti-symmetric-dgn]], [[transformers-particle-systems-clustering]], [[dynami-cal-graphnet]], [[transformer-mathematical-framework]]

---

## The decision

The initial model uses a **symmetric attention score** as its core token-interaction mechanism, paired with a **conservative (skew-symmetric) update channel**. This page explains *why symmetric*, *why that alone is not enough*, and *what to pair it with* — so the choice is made for the right reason rather than by analogy.

---

## Why symmetric: reciprocity is Newton's third law

Standard attention computes $A_{ij} = \mathrm{softmax}_j(q_i^\top k_j / \sqrt d)$ with $q_i = W_Q h_i$, $k_j = W_K h_j$. Because $W_Q \neq W_K$ and softmax normalizes each row separately,

$$A_{ij} \neq A_{ji}.$$

Physically, $A_{ij}$ is *how much token $j$ acts on token $i$*. Every fundamental pairwise interaction in physics is **reciprocal**: the force of $i$ on $j$ is equal and opposite to the force of $j$ on $i$ (Newton's third law),

$$\vec F_{ij} = -\vec F_{ji}.$$

Gravity, Coulomb, contact forces, the symmetric stress tensor, the symmetric diffusion/conduction operators — all reciprocal. An attention mechanism that lets $j$ influence $i$ more than $i$ influences $j$ has no physical counterpart and silently injects spurious momentum. **Making the score symmetric, $A_{ij}=A_{ji}$, is the architectural statement that interactions are reciprocal.**

### How to make the score symmetric

Tie the projections or symmetrize the bilinear form. Two clean options:

$$\text{(tied)}\quad q_i = k_i = W h_i \ \Rightarrow\ s_{ij} = (Wh_i)^\top (Wh_j) = h_i^\top W^\top W h_j,$$

which is symmetric because $W^\top W$ is a (symmetric, PSD) Gram matrix; or

$$\text{(symmetrized)}\quad s_{ij} = \tfrac{1}{2}\big(\langle q_i,k_j\rangle + \langle q_j,k_i\rangle\big)/\sqrt{d_k}.$$

To keep repulsion available, bound with $\tanh$ instead of softmax (signed coupling, [[antisymmetric-signed-attention-transformer]] Fix 3):

$$A_{ij} = \tanh(s_{ij}) \in [-1,1], \qquad A_{ij}=A_{ji}.$$

Note this **abandons row-stochastic softmax**: softmax's per-row normalization is exactly what breaks symmetry, so a symmetric mechanism cannot use it. Normalization, if needed, must be *symmetric* — e.g. the symmetric graph-Laplacian style $D^{-1/2} A D^{-1/2}$, which preserves $A_{ij}=A_{ji}$.

---

## The trap: symmetric attention is intrinsically *dissipative*

Here is the part that is easy to get wrong. Suppose we have a symmetric $A$ and use it the obvious way — a force-style (relative) aggregation, which already gives exact momentum conservation:

$$\Delta h_i = \sum_j A_{ij}\,(V_j - V_i).$$

This restores Newton's third law per edge ($A_{ij}(V_j-V_i) = -A_{ji}(V_i-V_j)$) and conserves the aggregate $\sum_i \Delta h_i = 0$. So far so good. **But what dynamical system is this?** Writing it as a continuous flow $\dot V = -L V$, where $L = D - A$ is the graph Laplacian of the (symmetric) attention graph,

$$\dot V_i = \sum_j A_{ij}(V_j - V_i) = -(LV)_i.$$

This is the **heat equation on the attention graph.** For a symmetric PSD $L$, the dynamics:
- conserve the total $\sum_i V_i$ (the zero-eigenvector / "momentum"),
- but **monotonically dissipate the Dirichlet energy** $\tfrac12 \sum_{ij} A_{ij}\|V_i-V_j\|^2$, smoothing everything toward the mean.

In other words, **symmetric reciprocal attention is diffusion.** It is the right operator for *parabolic* physics (heat, viscous diffusion) but it is exactly the over-smoothing / energy-leaking behavior that kills long-horizon rollouts ([[autoregressive-rollout-stability]]). Reciprocity conserves *momentum*; it does **not** conserve *energy*. Confusing "symmetric" with "conservative" is the central error to avoid.

This mirrors the clustering theorem of [[transformers-particle-systems-clustering]]: symmetric, attractive interaction → consensus/Dirac mass. Symmetry alone does not escape dissipation; it only makes the dissipation reciprocal.

---

## The fix: pair reciprocity with a conservative channel

To get energy conservation and information preservation across depth — non-dissipative, oscillatory dynamics — add a **skew-symmetric linear channel** (the A-DGN construction, [[anti-symmetric-dgn]]):

$$L_i = (W - W^\top - \gamma I)\, h_i.$$

A skew-symmetric matrix $W - W^\top$ has **purely imaginary eigenvalues**, so its flow $h(t) = e^{(W-W^\top)t} h(0)$ is **norm-preserving** (orthogonal) — the discrete analog of Hamiltonian/symplectic flow that conserves phase-space volume (Liouville). The $-\gamma I$ term adds a small, *tunable* dissipation: set $\gamma = 0$ for exactly conservative dynamics, or match $\gamma$ to the system's true physical dissipation rate (viscosity, drag, resistivity). This is the **port-Hamiltonian** decomposition — a conservative part + an explicit, controllable dissipative part — and it lets the model represent both ideal (energy-conserving) and real (dissipative) physics with one structure.

Combining reciprocity and conservation, one attention block is:

$$h_i^{\ell+1} = h_i^\ell + \epsilon\,\sigma\!\Big[\underbrace{(W - W^\top - \gamma I)\,h_i^\ell}_{\text{conservative channel (energy)}} \;+\; \underbrace{\sum_j A_{ij}\,(V_j^\ell - V_i^\ell)}_{\text{reciprocal interaction (momentum)}}\Big]$$

with $A_{ij} = \tanh\!\big(\tfrac{1}{2}(\langle q_i,k_j\rangle + \langle q_j,k_i\rangle)/\sqrt{d_k}\big)$ symmetric+signed, forward-Euler step $\epsilon < 2/\|J\|_2$, and a non-expansive activation $\sigma$. This is exactly the update in [[antisymmetric-signed-attention-transformer]]; the contribution here is the *why*: the two terms conserve two different things (momentum vs. energy), and you need both.

| Channel | Conserves | Dynamical role |
|---|---|---|
| Force-style reciprocal aggregation | aggregate momentum $\sum_i h_i$ | reciprocal interaction; diffusive on its own |
| Skew-symmetric linear part | norm / energy / information | non-dissipative depth; tunable $\gamma$ = physical dissipation |

---

## Symmetric attention *is* a learnable graph Laplacian — the tokenizer connection

On the [[graph-tokenizer]], tokens are graph nodes and attention is message passing. A symmetric attention score over a radius graph is precisely a **learnable, content-dependent, dynamic graph Laplacian.** Since graph Laplacians are the discrete generators of diffusion, wave propagation, and the elliptic operators underlying most PDEs, **symmetric attention turns the transformer into a learnable PDE operator on the token graph.** This is the concrete realization of "attention = physical message passing":

- Symmetric score $\leftrightarrow$ symmetric coupling weights $A_{ij}$ on edges.
- Force-style aggregation $\leftrightarrow$ Laplacian $-LV$ (diffusion / parabolic part).
- Skew-symmetric channel $\leftrightarrow$ the conservative / hyperbolic part (waves, oscillation).
- Signed $A$ $\leftrightarrow$ mixed attractive/repulsive coupling (pattern formation).

A model that can weight these channels in-context can represent parabolic, hyperbolic, and mixed dynamics with one mechanism — matching the local-vs-global PDE structure in [[tokenization-tradeoff-axes]].

---

## Interaction with normalization

Standard LayerNorm is **incompatible** with the conservative channel: subtracting the cross-feature mean and dividing by the norm destroys exactly the amplitude/energy the skew-symmetric part is built to preserve, and projects $h_i$ onto a manifold that interferes with the symplectic structure. The initial model therefore avoids LayerNorm inside attention blocks; see [[normalization-scheme]] for the full decision (nondimensionalize at input; RMS/equivariant gain only where calibration is needed; no mean-subtraction inside conservative blocks).

---

## Scaling to large contexts

Symmetric attention is the *mechanism*; at large token counts it must be made sparse or compressed without breaking symmetry:
- **[[index-share-sparse-attention]]** — restrict each node to a symmetric neighborhood (local + global landmark indices); symmetry is preserved as long as the index set is mutual ($j \in \mathcal I(i) \Leftrightarrow i \in \mathcal I(j)$, automatic for radius graphs).
- **[[hierarchical-query-attention]]** — compress to a fixed query budget first, then apply symmetric attention among the (fewer) summary tokens.

Both preserve the symmetric/conservative structure on the reduced interaction set.

---

## Open problems (carried from [[antisymmetric-signed-attention-transformer]])

1. **Multi-head antisymmetry.** The output projection $W_O$ must be structured (orthogonal / antisymmetry-preserving) so combining heads does not reintroduce asymmetry.
2. **Expressivity cost.** Restricting to symmetric scores + skew-symmetric linear parts shrinks the hypothesis class; for physics this is principled, but the accuracy cost vs. unconstrained attention is unmeasured.
3. **Stability-theorem transfer.** A-DGN's proof is for message-passing ODEs; transfer to the symmetric-but-non-softmax attention update needs formal analysis.
4. **FlashAttention compatibility.** The $\tanh$ symmetric score has the same memory pattern as standard attention and is mechanically simpler (no row normalization), but a fused kernel must be written.

---

## [AI Inference]

**[AI Inference]:** The cleanest mental model for the initial model's attention is a **discretized port-Hamiltonian field equation**: $\dot h = (J - R)\nabla \mathcal H(h) + \text{interaction}$, where the skew-symmetric channel supplies $J$ (conservative), $-\gamma I$ supplies $R$ (dissipative), and the symmetric reciprocal aggregation supplies the pairwise interaction. Choosing $\gamma$ and the attention sign structure in-context from the trajectory is then equivalent to the model inferring *how dissipative and how reciprocal* the observed physics is — a learnable placement on the conservative↔dissipative spectrum, which is exactly what distinguishes ideal flow from viscous flow, or a wave from a heat process.

**[AI Inference]:** Because symmetric reciprocal attention is provably diffusion, a PFM that used *only* symmetric attention would be biased toward parabolic (smoothing) behavior and would systematically over-damp turbulence and waves — the opposite of the desired long-horizon energy preservation. This predicts a concrete failure mode to test for: a symmetric-only ablation should show anomalous energy decay on the Euler / wave benchmarks, recovered once the skew-symmetric channel is added. The conservative channel is not a refinement; it is load-bearing.

---

## See Also

- [[antisymmetric-signed-attention-transformer]] — the full prior synthesis (three fixes, pipeline, benchmark plan)
- [[00-attention-overview]] — where this sits among PFM attention mechanisms
- [[anti-symmetric-dgn]] — skew-symmetric ODE; stability theorem; the conservative channel's foundation
- [[transformers-particle-systems-clustering]] — proof that symmetric attractive attention clusters/dissipates
- [[graph-tokenizer]] — symmetric attention as a learnable graph Laplacian on tokens
- [[normalization-scheme]] — why LayerNorm is dropped inside conservative blocks
- [[dynami-cal-graphnet]] — the graph-side antisymmetric-edge realization of the same momentum conservation
- [[index-share-sparse-attention]] / [[hierarchical-query-attention]] — scaling the mechanism to large contexts
- [[initial-model-architecture]] — assembled pipeline
