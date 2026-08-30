# Spatiotemporal Tubelet Tokens (3D / Video Patches)

**Type:** Token representation (folder: unified-token-representation)
**Used by:** [[gphyt-physics-foundation-model]] (4D tubelets), [[walrus-paper]] (history-window tubes), NVIDIA Cosmos tokenizer
**Related:** [[00-token-representation-overview]], [[patch-embedding-tokens]], [[transformer-architectures]], [[autoregressive-rollout-stability]], [[multimodal-tokenization]]

---

## Intuition

A physical state is not a snapshot but a *movie*. A tubelet token is a patch extended through time: instead of one $p\times p$ spatial tile, you take a $p\times p$ tile across $\tau$ consecutive frames — a little space-time "tube" — and project the whole block to one vector. The token therefore **carries short-time dynamics inside it**: a single tubelet already encodes local velocity and acceleration, not just an instantaneous value. This is the natural representation when the model must infer *how things are moving* from a context window (the precondition for in-context timescale inference in [[gphyt-physics-foundation-model]]).

---

## Mathematics

For a spatiotemporal field stacked over a history window, a tubelet covers $p\times p$ spatial cells over $\tau$ frames, $P_j^{x,y,t}\in\mathbb{R}^{p\times p\times \tau\times n}$, linearly embedded:

$$\mathbf z_j = \mathbf W_E\,\text{flatten}(P_j^{x,y,t}) + \mathbf b_E,\qquad \mathbf W_E\in\mathbb{R}^{C\times (p^2\tau n)}.$$

The token grid is now 3D, indexed $(j_x,j_y,j_t)$. Two attention patterns dominate:
- **Joint space-time attention** over all $N_x N_y N_t$ tokens — full coupling, cost $O((N_xN_yN_t)^2)$.
- **Factorized (axial) attention** — alternate spatial-only and temporal-only attention, cost $O(N_t(N_xN_y)^2)+O(N_xN_y N_t^2)$. Used by Walrus (axial RoPE spatially, T5 causal bias temporally).

A tubelet finite-difference interpretation: with $\tau=2$, the projection can represent $\partial_t u\approx (u_{t}-u_{t-1})/\Delta t$ — i.e. the tubelet *natively expresses the time derivative* that [[gphyt-physics-foundation-model]] then exploits with its $X_{t+1}=X_t+\Delta t\,\partial_t X$ integrator.

---

## Pros

- **Encodes dynamics in the token** — velocity/tendency available without extra channels; ideal for derivative-prediction PFMs.
- **In-context timescale inference** — combined with variable-$\Delta t$ training (GP$_{\text{hy}}$T), the model reads temporal scale off the tube.
- **Compresses the temporal axis** — $\tau$ frames → fewer tokens than treating each frame separately.
- **Reuses ViT machinery** — same linear-embedding simplicity as [[patch-embedding-tokens]], one dimension up.
- **Causal-friendly** — temporal axis supports causal masking to respect the arrow of time.

## Cons

- **Inherits all patch-token weaknesses** (resolution-locking, aliasing, no physics semantics, regular grid) — now in space *and* time.
- **Temporal aliasing** — a fixed $\tau$ and frame stride alias fast dynamics (high temporal frequencies), worsening rollout error.
- **Cost grows with the temporal window** — naive joint attention is quartic-ish; forces factorization.
- **Fixed history length** — $\tau$ is baked into $\mathbf W_E$; variable-length history needs padding or re-embedding.
- **Minimum-context coupling** — too short a tube cannot represent the order of the dynamics ([[kepler-newton-inductive-biases]]: minimum context ≈ ODE order); too long wastes compute and can over-fit geometry.

---

## Relationship to other representations

- vs. [[patch-embedding-tokens]]: tubelets = patches + time axis.
- vs. derivative-augmented channels (GP$_{\text{hy}}$T): both inject dynamics, but tubelets do it *implicitly* via the temporal extent while derivative channels do it *explicitly* via finite differences — they are complementary and GP$_{\text{hy}}$T uses both.
- vs. operator-learning lead-time conditioning ([[poseidon-pde-foundation-model]]): Poseidon avoids a temporal token axis entirely, encoding time as a *continuous LayerNorm modulation* instead — a fundamentally different way to put time into the model.

**[AI Inference]:** The NVIDIA Cosmos tokenizer (8× compression, 12× faster, trained on 20M hours of physical/robotic video) shows tubelet tokenization scales to enormous physical-video corpora. A PFM could **pretrain a Cosmos-style tubelet tokenizer on simulation + real experimental video jointly**, giving a shared dynamical token space across synthetic and measured data — a concrete route to the sim-to-real data unification that [[genesis-physics-engine]] and [[walrus-paper]] both want.

**[AI Inference]:** Because a $\tau=2$ tubelet linearly encodes $\partial_t$, a tubelet tokenizer + a derivative-prediction head is *almost* a discretized neural ODE. Making the temporal embedding an explicit finite-difference stencil (rather than a free linear layer) would turn the tokenizer into a structure-preserving time-derivative operator — bridging tubelet tokens and [[structure-preserving-tokens]].

---

## See also

- [[00-token-representation-overview]] — hub
- [[patch-embedding-tokens]] — the spatial-only base
- [[gphyt-physics-foundation-model]] — 4D tubelets + derivative integrator
- [[walrus-paper]] — history-window tubes + axial attention
- [[autoregressive-rollout-stability]] — temporal aliasing and rollout error
- [[kepler-newton-inductive-biases]] — minimum context length = ODE order
- [[structure-preserving-tokens]] — finite-difference / equivariant temporal embedding
