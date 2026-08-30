# Physics-Conditioned Query Tokens (Proposed Alternative)

**Type:** Token representation — *proposed better alternative* (folder: unified-token-representation)
**Status:** Synthesis / proposal — no published PFM realizes this yet. Heavy **[AI Inference]**.
**Builds on:** [[learned-query-compression-tokens]], [[multimodal-tokenization]] (physics-conditioned Q-Former proposal)
**Related:** [[00-token-representation-overview]], [[scalar-cdf-tokens]], [[structure-preserving-tokens]], [[neural-operators]], [[pfm-interface-design]]

---

## Intuition

Take the resolution-invariant Q-Former idea ([[learned-query-compression-tokens]]) and **give the queries physical meaning**. Instead of $Q$ opaque learnable vectors, initialize each query to extract a *specific physical quantity* — vorticity, divergence, kinetic energy, strain-rate invariants, pressure-gradient magnitude, boundary-layer thickness, spectral-energy peak — and train it with physics-informed objectives so it *stays* meaningful. The result: a fixed-size set of tokens that is (1) resolution-invariant by construction, (2) physically interpretable, and (3) scale-aware (energy/vorticity carry the regime). Each token answers a physically motivated question of the field, regardless of how the field was discretized. This is the representation that scores well on the *most* criteria in [[00-token-representation-overview]] simultaneously — but it is a proposal, not yet built.

---

## Mathematics

Same cross-attention compression as a Q-Former, $\mathbf z=\text{CrossAttn}(\mathbf q,X)\in\mathbb{R}^{Q\times d}$, with three physics modifications:

**(1) Physics-initialized / structured queries.** Query $q_k$ is associated with a physical functional $\Phi_k[u]$ (e.g. $\Phi_\text{vort}[u]=\nabla\times u$, $\Phi_\text{div}[u]=\nabla\cdot u$, $\Phi_E[u]=\tfrac12|u|^2$). Queries can be *seeded* so that, at init, $z_k\approx \langle \Phi_k[u]\rangle$ (the field's projection onto that functional), e.g. by constructing the key/value projections from differential operators.

**(2) Physics-informed auxiliary loss.** Train so each token predicts its target functional, anchoring meaning:
$$\mathcal L_\text{phys}=\sum_k \big\| g_k(z_k) - \Phi_k[u]\big\|^2 \;+\; \lambda\,\big\|\,\text{PDE-residual reconstructed from }\{z_k\}\big\|^2,$$
so the tokens must both *be* their physical quantities and *jointly* suffice to satisfy the governing equation.

**(3) Scale/parameter fusion.** Concatenate scalar-CDF parameter tokens ([[scalar-cdf-tokens]]) or continuous FiLM modulation (Re, Ma, Pr) into the query set, so the compressed representation is explicitly conditioned on the regime.

Output: $Q$ physically-labeled, resolution-invariant tokens; a decoder (coordinate/trunk head, [[coordinate-implicit-tokens]]) reconstructs the field at arbitrary points.

---

## Pros

- **Resolution-invariant *and* physically meaningful** — combines the Q-Former's fixed-budget resolution invariance (criteria 1, 7) with interpretable, structure-bearing tokens (criterion 5). Few representations hit both.
- **Scale-aware by design** — energy/vorticity/parameter tokens carry the regime; closes the scale-awareness gap of [[multimodal-tokenization]] and [[patch-embedding-tokens]] (criterion 4).
- **Interpretable & debuggable** — you can read what the model attends to ("it's tracking vorticity and pressure gradient"), a rarity in PFM internals; aids scientific trust and discovery.
- **Conditioning interface for free** — steerable queries double as the task/parameter interface ([[pfm-interface-design]]).
- **Field-unifying** — different field sets map to a shared set of physical-functional tokens (vorticity exists whether or not temperature does), easing multi-physics unification (criterion 3).
- **Physics-residual anchoring resists drift** — the auxiliary loss keeps tokens meaningful through training, unlike opaque queries that can collapse.

## Cons

- **Unproven** — no published PFM uses physics-conditioned queries; all benefits are projected.
- **Lossy compression remains** — still $Q$ tokens; if the chosen functionals don't span the dynamics, information is lost. Requires a good *basis of physical functionals*, which is problem-dependent (turbulence needs different functionals than reaction–diffusion).
- **Design burden** — choosing/initializing the functional set $\{\Phi_k\}$ is a modeling decision; a bad set bakes in blind spots.
- **Auxiliary-loss balancing** — multi-objective training (reconstruction + functional + residual) is delicate (the PINN-style stiffness of [[physicsformer-pinn-ns]]).
- **Reconstruction from semantic tokens is hard** — decoding a full high-res field from $Q$ physical summaries needs a strong decoder; fine detail may need a residual path ([[wave-particle-dual-tokens]]).
- **Differential-operator queries assume some smoothness** — sharp shocks complicate vorticity/divergence functionals.

---

## Why this is a *better* alternative

Against the [[00-token-representation-overview]] scorecard, physics-conditioned query tokens are the rare representation that targets resolution invariance (1), field unification (3), scale awareness (4), structure preservation (5), and compute tractability (7) *together*, while remaining transformer-native (unlike pure spectral/operator tokens that are geometry-restricted). Its weaknesses (lossiness, decoder difficulty) are shared with all compression schemes and can be mitigated by an adaptive $Q$ or a dual residual path.

**[AI Inference]:** The minimal viable experiment is a **physics-conditioned Q-Former front-end on a Poseidon/GP$_{\text{hy}}$T backbone**: ~32 queries seeded to vorticity, divergence, energy, enstrophy, pressure-gradient, and a handful of learnable extras; auxiliary functional + PDE-residual losses; coordinate-net decoder. Success criterion: match patch-token accuracy while running unchanged across resolutions and exposing interpretable token activations.

**[AI Inference]:** Physics-conditioned queries are a natural substrate for **scientific discovery**: if the learnable (un-seeded) queries converge to reproducible functionals across many systems, those discovered functionals are candidate *new conserved/organizing quantities* — connecting tokenization to the Noether-discovery program of [[noether-networks]] and the world-model analysis of [[world-models-physics-ai]].

---

## See also

- [[00-token-representation-overview]] — hub; this scores on criteria 1,3,4,5,7
- [[learned-query-compression-tokens]] — the Q-Former base being upgraded
- [[multimodal-tokenization]] — original physics-conditioned Q-Former inference
- [[scalar-cdf-tokens]] — parameter fusion into the query set
- [[coordinate-implicit-tokens]] — decoder head for reconstruction
- [[structure-preserving-tokens]] — complementary hard-constraint approach
- [[wave-particle-dual-tokens]] — add a residual detail path
- [[pfm-interface-design]] — steerable queries as conditioning interface
- [[noether-networks]] / [[world-models-physics-ai]] — discovered-functional connection
