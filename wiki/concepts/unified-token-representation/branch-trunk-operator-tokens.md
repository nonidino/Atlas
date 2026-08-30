# Branch–Trunk Operator Tokens (DeepONet-Style)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** DeepONet family ([[deeponet-multi-operator]], [[pc-deeponet-cfd]], [[varmion-viscous-flows]])
**Related:** [[00-token-representation-overview]], [[neural-operators]], [[spectral-fourier-tokens]], [[coordinate-implicit-tokens]], [[gaussian-processes]]

---

## Intuition

Separate *what the function is* from *where you want to evaluate it*. DeepONet splits the representation into two encoders: a **branch** net reads the input function (sampled at fixed sensor points) and produces a vector of coefficients; a **trunk** net reads a single query coordinate $y$ and produces a vector of learned basis values at that point. The output at $y$ is their inner product — i.e. the field is reconstructed as a **learned basis expansion** whose coefficients come from the input. The "tokens," loosely, are (i) the branch coefficient vector summarizing the whole input function and (ii) the per-query trunk vectors. This makes the representation **resolution-free in the output** (query any $y$, including off-grid) and **mesh-flexible** (sensors need not be a regular grid), which is why it is a foundational neural-operator tokenization.

---

## Mathematics

DeepONet approximates an operator $\mathcal G:u\mapsto \mathcal G(u)$ as

$$\mathcal G(u)(y)\approx \sum_{k=1}^{q} \underbrace{b_k(u(x_1),\ldots,u(x_M))}_{\text{branch}} \cdot \underbrace{t_k(y)}_{\text{trunk}} + b_0,$$

where the branch net $\mathbf b\in\mathbb{R}^q$ encodes the input function sampled at $M$ sensors, and the trunk net $\mathbf t(y)\in\mathbb{R}^q$ is a learned basis evaluated at query point $y$. The universal approximation theorem for operators guarantees this form can approximate any continuous operator. Extensions in the wiki:
- **D2NO / multi-operator** ([[deeponet-multi-operator]]): branch parameters specialized per operator family; physics-informed zero-shot fine-tuning + LoRA.
- **Physics-constrained DeepONet** ([[pc-deeponet-cfd]]): outputs a stream function whose curl is divergence-free by construction — a *hard* constraint baked into the trunk output.
- **VarMiON** ([[varmion-viscous-flows]]): branch architecture derived from the *variational (weak) form* of the PDE, so the token structure mirrors the discretized weak operator.

The trunk is closely related to a **learned kernel / basis**; in the linear limit the branch–trunk inner product is a kernel regression, connecting to [[gaussian-processes]] and to attention-as-kernel.

---

## Pros

- **Resolution-free output** — query the solution at any coordinate, on or off grid; the defining operator-learning property (criterion 1, [[00-token-representation-overview]]).
- **Mesh / geometry flexible** — sensors and query points are arbitrary point sets; handles irregular domains better than grid patches or FFT.
- **Separates function identity from evaluation location** — clean factorization; the branch token is a compact whole-field summary (useful for conditioning / parameter inference).
- **Learned adaptive basis** — the trunk learns problem-appropriate basis functions (vs. the fixed Fourier basis of [[spectral-fourier-tokens]]), better for non-periodic/complex geometry.
- **Natural home for hard constraints** — trunk outputs can be constrained (divergence-free via stream function, [[pc-deeponet-cfd]]) — encoding-spectrum level 5.
- **Physics-informed & few-shot friendly** — pairs well with PDE-residual losses and LoRA adaptation ([[deeponet-multi-operator]]).

## Cons

- **Fixed sensor layout for the branch** — the branch typically expects inputs at fixed sensor locations; changing the input sampling needs care (a partial resolution-dependence on the *input* side, unlike the output).
- **Single global branch code can bottleneck** — summarizing a complex multiscale field into one coefficient vector $\mathbf b$ loses local detail (mitigated by multi-branch / stacked variants).
- **Limited spatial locality in the basis** — global trunk bases can struggle with sharp local features unless many basis functions are used.
- **Less mature at foundation-model scale** — DeepONets are proven on specific operators; none has matched the breadth of [[poseidon-pde-foundation-model]] / [[walrus-paper]] as a general PFM (though Poseidon is itself framed as operator learning).
- **Two-network training** can be finicky to balance (branch vs. trunk capacity, output normalization).

---

## Relationship to other representations

- vs. [[spectral-fourier-tokens]]: both are neural-operator tokenizations giving resolution-free output. FNO uses the *fixed Fourier* basis (efficient, periodic grids); DeepONet learns an *adaptive* basis (flexible geometry, less efficient). Complementary strengths.
- vs. [[coordinate-implicit-tokens]]: the trunk net *is* a coordinate-based (implicit) representation of the basis — DeepONet = coordinate-MLP basis (trunk) + data-driven coefficients (branch). The two pages are close cousins.
- vs. [[learned-query-compression-tokens]]: the branch coefficient vector is a fixed-length compression of the input function, analogous to Q-Former's fixed query tokens — both compress an arbitrary input to a fixed-size code.

**[AI Inference]:** [[poseidon-pde-foundation-model]] is explicitly an *operator-learning* model (it learns $\mathcal S(t,a)$) but uses **patch tokens**, not branch–trunk tokens. A Poseidon-style multiscale transformer with a **trunk-style resolution-free decoder head** (query the solution at arbitrary $(x,t)$) would combine the transformer's expressivity with DeepONet's resolution invariance — directly addressing Poseidon's grid-tied limitation. The branch role is played by the encoder; the trunk replaces patch recovery.

**[AI Inference]:** The branch token — a compact whole-field summary — is an ideal **conditioning / inverse-inference token**: feeding it (plus a scalar-CDF parameter token, [[scalar-cdf-tokens]]) into a masked-modeling head ([[aion-1-astronomy]]) would let one model do forward operator evaluation *and* parameter inference from the same representation.

---

## See also

- [[00-token-representation-overview]] — hub
- [[neural-operators]] — DeepONet/FNO operator-learning framework
- [[deeponet-multi-operator]] / [[pc-deeponet-cfd]] / [[varmion-viscous-flows]] — DeepONet variants
- [[spectral-fourier-tokens]] — fixed-basis operator tokenization
- [[coordinate-implicit-tokens]] — the trunk as a coordinate-MLP
- [[poseidon-pde-foundation-model]] — operator learning with patch (not branch–trunk) tokens
- [[gaussian-processes]] — branch–trunk inner product as kernel regression
