# Coordinate / Implicit-Neural-Representation Tokens

**Type:** Token representation (folder: unified-token-representation)
**Used by:** PINNs ([[physicsformer-pinn-ns]]), implicit neural representations (INR/SIREN/NeRF lineage), DeepONet trunk ([[branch-trunk-operator-tokens]])
**Related:** [[00-token-representation-overview]], [[branch-trunk-operator-tokens]], [[neural-operators]], [[spectral-fourier-tokens]], [[gaussian-processes]]

---

## Intuition

Don't tokenize the field — tokenize *space itself*. In a coordinate/implicit representation the input "token" is a **coordinate** $(x,t)$, and the field is a neural function evaluated there: $u_\theta(x,t)$. The continuous field is stored entirely in the network weights; you read it out by querying coordinates. This is how PINNs ([[physicsformer-pinn-ns]]) and NeRF-style implicit representations work. The appeal for physics is profound: the representation is **mesh-free and resolution-free by definition** (query anywhere), differentiation is **exact via autodiff** (so PDE residuals are exact, not finite-difference approximations), and boundary/initial conditions are imposed at arbitrary points. The catch: a plain coordinate-MLP per problem is a *fit*, not a *foundation model* — it must be retrained for each new field unless coupled to a conditioning mechanism (branch code, latent token, hypernetwork).

---

## Mathematics

A coordinate network maps position → field value: $u_\theta:\mathbb{R}^{d+1}\to\mathbb{R}^m$, $u_\theta(x,t)$. Raw MLPs have a **spectral bias** (struggle with high frequencies), fixed by **Fourier features** or sinusoidal activations (SIREN):

$$\gamma(x)=[\sin(2\pi B x),\cos(2\pi B x)],\qquad u_\theta(x)=\text{MLP}(\gamma(x)),$$

with $B$ a (fixed or learned) frequency matrix. [[physicsformer-pinn-ns]] uses a trainable $w\sin(t)$ activation to capture high-frequency temporal solutions. The **PDE-residual loss** uses exact autodiff derivatives:

$$\mathcal L_\text{PDE}=\big\|\partial_t u_\theta + \mathcal N(u_\theta,\nabla u_\theta,\nabla^2 u_\theta)\big\|^2,\qquad \nabla u_\theta,\ \partial_t u_\theta\ \text{by autodiff (exact)}.$$

**Conditioned / generalizable variants** turn the per-problem fit into a representation: a latent code $z$ (from an encoder/branch, [[branch-trunk-operator-tokens]]) modulates the coordinate net, $u_\theta(x;z)$ — via concatenation, FiLM/AdaLN, or a hypernetwork $\theta=H(z)$. The DeepONet trunk is exactly a coordinate net whose basis is combined with branch coefficients.

---

## Pros

- **Mesh-free & resolution-free** — query the field at any continuous coordinate; no grid, no fixed token count (criteria 1–2, [[00-token-representation-overview]]).
- **Exact derivatives via autodiff** — PDE residuals, conservation diagnostics, and physics-informed losses are exact, not discretization-approximated (best-in-class for criterion 5).
- **Arbitrary geometry & BCs** — impose conditions at any points; complex domains are trivial.
- **Compact for smooth fields** — a small MLP can store a whole continuous field.
- **Composes with conditioning** — latent-modulated coordinate nets become a generalizable, resolution-free decoder head for a PFM.

## Cons

- **Per-problem fitting (unconditioned)** — a bare PINN/INR memorizes one field; not a foundation model without a conditioning/encoder mechanism. PINNs are notoriously slow to train per instance.
- **Spectral bias** — plain MLPs miss high frequencies; needs Fourier features / SIREN, and even then sharp shocks are hard.
- **Optimization difficulty** — PINN losses are stiff/multi-objective (data + PDE + BC weighting); convergence is delicate ([[physicsformer-pinn-ns]] uses dynamic loss weighting).
- **Global weight coupling** — the whole field lives in shared weights, so local edits and locality are unnatural (opposite of patch/graph locality).
- **Querying many points is sequential-ish** — to render a full field you evaluate the net at every output coordinate (though batched).

---

## Relationship to other representations

- vs. [[branch-trunk-operator-tokens]]: the trunk **is** a coordinate net; DeepONet = coordinate-net basis (trunk) + data-driven coefficients (branch). This page is the "trunk in isolation," generalized.
- vs. [[spectral-fourier-tokens]]: Fourier-feature coordinate nets and spectral tokens both lean on a frequency basis; spectral tokens store *coefficients* explicitly, coordinate nets store them *implicitly* in weights. Spectral is efficient on grids; coordinate nets are geometry-free.
- vs. [[patch-embedding-tokens]] / [[graph-mesh-tokens]]: those discretize the *field*; coordinate nets discretize *queries* and keep the field continuous — the most "function-space" of all the representations, alongside neural operators.

**[AI Inference]:** The highest-value role for coordinate/implicit tokens in a PFM is as a **resolution-free, autodiff-differentiable decoder head** sitting on top of a transformer encoder: the encoder produces a latent (or set of latent tokens) summarizing the state; a latent-conditioned coordinate net decodes the solution at arbitrary $(x,t)$, with exact PDE-residual guidance available at decode time. This marries [[poseidon-pde-foundation-model]]'s expressive encoder with PINN-grade exact physics — and is essentially the trunk-decoder idea floated in [[branch-trunk-operator-tokens]].

**[AI Inference]:** Because autodiff derivatives are exact, a coordinate-decoder PFM can enforce **hard test-time physics constraints** (DPS-style guidance, [[pisd-physics-informed-spectral-diffusion]]) far more accurately than finite-difference tokenizers — pushing the model up the physics-encoding spectrum (level 3→5) without changing the encoder.

---

## See also

- [[00-token-representation-overview]] — hub; criteria 1, 2, 5
- [[physicsformer-pinn-ns]] — transformer-PINN coordinate representation
- [[branch-trunk-operator-tokens]] — DeepONet trunk = conditioned coordinate net
- [[neural-operators]] / [[spectral-fourier-tokens]] — resolution-free relatives
- [[pisd-physics-informed-spectral-diffusion]] — exact-residual test-time guidance
- [[poseidon-pde-foundation-model]] — candidate encoder for a coordinate decoder head
- [[gaussian-processes]] — function-space view; kernel/coordinate connections
